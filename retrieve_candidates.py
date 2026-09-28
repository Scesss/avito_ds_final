"""Кандидатогенерация: TF-IDF + история формулировок + география из train.

Без нейросетей, внешних API и ручных ответов под query_id.
Весовые коэффициенты выбраны на локальной dev-выборке (experiment.py).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from validate_answer import validate

QUERY_COLUMNS = [
    'search_query', 'search_location_id', 'search_is_delivery_search',
    'search_infm_params_text', 'search_category',
]
TRAIN_COLUMNS = QUERY_COLUMNS + ['item_title_raw', 'item_location_id']


def as_text(values):
    return values.fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


def title_key(text):
    return re.sub(r'\W+', ' ', str(text).lower()).strip()


def fit_index(documents, analyzer, ngrams, min_df, max_features):
    vectorizer = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngrams, min_df=min_df,
        max_features=max_features, sublinear_tf=True, dtype=np.float32,
    )
    return vectorizer, vectorizer.fit_transform(documents)


def build_text_indexes(items):
    title = as_text(items.item_title_raw)
    params = as_text(items.item_infm_params_text)
    description = as_text(items.item_description_raw)
    documents = title + ' ' + title + ' ' + title + ' ' + params + ' ' + params + ' ' + description
    print('Строим TF-IDF по словам...', flush=True)
    word = fit_index(documents, 'word', (1, 2), 2, 250_000)
    print('Строим TF-IDF по фрагментам слов...', flush=True)
    char = fit_index(documents, 'char_wb', (3, 5), 3, 300_000)
    return word, char


def build_history_index(items, train):
    """Переносим поисковые формулировки на объявления с таким же заголовком."""
    history = train[['item_title_raw', 'search_query']].copy()
    history['title_key'] = history.item_title_raw.map(title_key)
    texts = history.groupby('title_key').search_query.agg(lambda x: ' '.join(sorted(set(x))))
    documents = items.item_title_raw.map(title_key).map(texts).fillna('')
    print('Строим TF-IDF по историческим запросам...', flush=True)
    return fit_index(documents, 'word', (1, 2), 1, 200_000)


def predict(items, queries, train, text_indexes=None, history_index=None, batch_size=24):
    """Возвращает индексы 50 кандидатов на запрос. Кеши можно передать для эксперимента."""
    if text_indexes is None:
        text_indexes = build_text_indexes(items)
    if history_index is None:
        history_index = build_history_index(items, train)
    word, char = text_indexes
    text = as_text(queries.search_query)
    documents = text + ' ' + text + ' ' + text + ' ' + as_text(queries.search_infm_params_text)
    qw, qc = word[0].transform(documents), char[0].transform(documents)
    qh = history_index[0].transform(text)

    transitions = train.groupby(['search_location_id', 'item_location_id']).size()
    location_counts = {loc: counts.droplevel(0) for loc, counts in transitions.groupby(level=0)}
    locations, location_codes = np.unique(items.item_location_id, return_inverse=True)
    categories = items.item_category_id.to_numpy()
    predictions = []

    for start in tqdm(range(0, len(queries), batch_size), desc='Поиск', unit='batch'):
        stop = start + batch_size
        scores = .58 * (qw[start:stop] @ word[1].T).toarray()
        scores += .42 * (qc[start:stop] @ char[1].T).toarray()
        scores += .30 * (qh[start:stop] @ history_index[1].T).toarray()
        for offset, query in enumerate(queries.iloc[start:stop].itertuples()):
            counts = location_counts.get(query.search_location_id, pd.Series(dtype=float)).copy()
            counts.loc[query.search_location_id] = counts.get(query.search_location_id, 0) + 10
            geo = counts.reindex(locations, fill_value=0).to_numpy(dtype=np.float32)
            geo /= max(geo.max(), 1)
            scores[offset] *= 1 + 9 * geo[location_codes]
            if query.search_category != 0:
                scores[offset] *= np.where(categories == query.search_category, 1., .15)
            k = min(50, len(items))
            indices = np.argpartition(scores[offset], -k)[-k:]
            indices = indices[np.argsort(scores[offset, indices])[::-1]]
            predictions.append(indices)
    return predictions


def save_answer(items, queries, predictions, path):
    ids = items.item_id.to_numpy()
    answer = pd.DataFrame({
        'query_id': queries.query_id.to_numpy(),
        'answer': [' '.join(ids[row]) for row in predictions],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(path, index=False, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    items = pd.read_parquet(args.data_dir / 'benchmark_items.parquet')
    queries = pd.read_parquet(args.data_dir / 'benchmark_queries.parquet')
    train = pd.read_parquet(args.data_dir / 'train.parquet', columns=TRAIN_COLUMNS)
    predictions = predict(items, queries, train)
    path = args.output_dir / 'answer.csv'
    save_answer(items, queries, predictions, path)
    validate(path, args.data_dir)
    print(f'Файл для Stepik: {path}')


if __name__ == '__main__':
    main()
