"""Локальные эксперименты. Разметка benchmark нигде не используется."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from baseline_retriever import (
    QUERY_COLUMNS, as_text, make_item_documents, make_query_documents,
    normalize_title, build_history, top_indices,
)

CACHE = Path('outputs/experiments')
CACHE.mkdir(parents=True, exist_ok=True)


def split_data(train, items):
    """Один запрос на текст; 63% текстов целиком убираем из обучения.

    Оставшиеся тексты можно видеть в других локациях/с другими фильтрами.
    Все строки проверяемого запроса удаляем. Считаем recall по позитивам,
    доступным в фиксированном корпусе benchmark_items.
    """
    rng = np.random.default_rng(20260928)
    unique = train[QUERY_COLUMNS].drop_duplicates().reset_index(drop=True)
    unique['group'] = np.arange(len(unique))
    pairs = train.merge(unique, on=QUERY_COLUMNS)
    eligible = pairs[pairs.item_id.isin(items.item_id)]
    candidates = unique[unique.group.isin(eligible.group)].copy()
    candidates['text'] = as_text(candidates.search_query)
    candidates = candidates.sample(frac=1, random_state=42).drop_duplicates('text')
    benchmark = pd.read_parquet('data/benchmark_queries.parquet')
    lengths = benchmark.search_query.str.split().str.len().clip(upper=7)
    candidates['length'] = candidates.search_query.str.split().str.len().clip(upper=7)
    chosen = []
    for length, proportion in lengths.value_counts(normalize=True).items():
        subset = candidates[candidates.length == length]
        chosen.append(subset.sample(n=min(len(subset), round(proportion * 1200)), random_state=17))
    valid = pd.concat(chosen).sample(frac=1, random_state=23).reset_index(drop=True)
    cold = rng.random(len(valid)) < .63
    cold_texts = set(valid.loc[cold, 'text'])
    keep = ~as_text(pairs.search_query).isin(cold_texts) & ~pairs.group.isin(valid.group)
    history = pairs.loc[keep, train.columns].copy()
    positives = eligible.groupby('group').item_id.agg(lambda x: set(x))
    truth = [positives[g] for g in valid.group]
    valid['query_id'] = valid.group.map(lambda x: f'validation{x:06d}')
    valid['seen_text'] = valid.text.isin(as_text(history.search_query))
    valid['partition'] = np.where(np.arange(len(valid)) % 3 == 0, 'test', 'dev')
    print('Validation:', len(valid), 'seen text:', valid.seen_text.mean(), flush=True)
    return history, valid, truth


def matrix(name, documents, analyzer, ngrams, min_df, max_features):
    path = CACHE / (name + '.joblib')
    if path.exists():
        return joblib.load(path)
    print('Building', name, flush=True)
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams,
        min_df=min_df, max_features=max_features, sublinear_tf=True, dtype=np.float32)
    mat = vectorizer.fit_transform(documents)
    joblib.dump((vectorizer, mat), path)
    print('Saved', name, mat.shape, flush=True)
    return vectorizer, mat


def build_features():
    items = pd.read_parquet('data/benchmark_items.parquet')
    train = pd.read_parquet('data/train.parquet', columns=QUERY_COLUMNS +
        ['item_id', 'item_title_raw', 'item_location_id', 'item_microcat_id'])
    history, valid, truth = split_data(train, items)
    joblib.dump((history, valid, truth), CACHE / 'split.joblib')
    docs = make_item_documents(items)
    bw, iw = matrix('baseline_word', docs, 'word', (1, 2), 2, 250000)
    bc, ic = matrix('baseline_char', docs, 'char_wb', (3, 5), 3, 300000)
    tw, itw = matrix('title_word', as_text(items.item_title_raw), 'word', (1, 2), 1, 200000)
    tc, itc = matrix('title_char', as_text(items.item_title_raw), 'char_wb', (3, 5), 2, 200000)
    history['title_key'] = history.item_title_raw.map(normalize_title)
    title_queries = history.groupby('title_key').search_query.agg(lambda x: ' '.join(sorted(set(x))))
    expanded = items.item_title_raw.map(normalize_title).map(title_queries).fillna('')
    ew, iew = matrix('history_queries', expanded, 'word', (1, 2), 1, 200000)

    ids = items.item_id.to_numpy()
    ids_to_index = {x: i for i, x in enumerate(ids)}
    from collections import defaultdict
    title_index = defaultdict(list)
    for i, title in enumerate(items.item_title_raw):
        title_index[normalize_title(title)].append(i)
    exact, by_text = build_history(history, ids_to_index, title_index)
    locations = items.item_location_id.to_numpy()
    categories = items.item_category_id.to_numpy()
    transitions = history.groupby(['search_location_id', 'item_location_id']).size()
    matrices = [iw, ic, itw, itc, iew]
    qdocs = make_query_documents(valid)
    queries = [bw.transform(qdocs), bc.transform(qdocs), tw.transform(as_text(valid.search_query)),
               tc.transform(as_text(valid.search_query)), ew.transform(as_text(valid.search_query))]
    results = []
    for start in range(0, len(valid), 24):
        dense = [(q[start:start+24] @ m.T).toarray() for q, m in zip(queries, matrices)]
        for j, query in enumerate(valid.iloc[start:start+24].itertuples()):
            fields = np.stack([scores[j] for scores in dense], axis=1)
            base = .58 * fields[:, 0] + .42 * fields[:, 1]
            local = np.flatnonzero(locations == query.search_location_id)
            key = tuple(getattr(query, c) for c in QUERY_COLUMNS)
            h = set(exact.get(key, []) + by_text.get(query.search_query, []))
            baseline_pool = set(top_indices(base, 180)) | set(local[top_indices(base[local], 50)]) | h
            base_score = base + .07 * (locations == query.search_location_id)
            if query.search_category:
                base_score += .10 * (categories == query.search_category)
            hp = np.fromiter(h, dtype=int)
            base_score[hp] += .35
            bp = np.array(sorted(baseline_pool))
            original = bp[top_indices(base_score[bp], 50)]
            counts = transitions.get(query.search_location_id, pd.Series(dtype=float)).to_dict()
            counts[query.search_location_id] = counts.get(query.search_location_id, 0) + 10
            geo = np.array([counts.get(loc, 0) for loc in locations], dtype=np.float32)
            geo /= max(geo.max(), 1)
            pool = set(baseline_pool)
            for f in range(fields.shape[1]):
                pool.update(top_indices(fields[:, f], 200))
                pool.update(top_indices(fields[:, f] * (.1 + geo), 200))
            pool = np.array(sorted(pool))
            hits = np.array([ids[i] in truth[start+j] for i in pool])
            results.append(dict(pool=pool, fields=fields[pool], geo=geo[pool],
                same=(locations[pool] == query.search_location_id).astype(np.float32),
                category=((categories[pool] == query.search_category) | (query.search_category == 0)).astype(np.float32),
                history=np.isin(pool, hp).astype(np.float32), hits=hits, ntrue=len(truth[start+j]),
                baseline=len(set(ids[original]) & truth[start+j]) / len(truth[start+j])))
        print('Features', min(start+24, len(valid)), '/', len(valid), flush=True)
    joblib.dump(results, CACHE / 'features.joblib')


def finalize():
    """Один итоговый замер на test, затем предсказания с использованием всего train."""
    from retrieve_candidates import predict, save_answer, TRAIN_COLUMNS
    from validate_answer import validate

    items = pd.read_parquet('data/benchmark_items.parquet')
    history, valid, truth = joblib.load(CACHE / 'split.joblib')
    results = joblib.load(CACHE / 'features.joblib')
    text_indexes = (joblib.load(CACHE / 'baseline_word.joblib'),
                    joblib.load(CACHE / 'baseline_char.joblib'))
    historical_index = joblib.load(CACHE / 'history_queries.joblib')
    predictions = predict(items, valid, history, text_indexes, historical_index)
    ids = items.item_id.to_numpy()
    recall = np.array([len(set(ids[p]) & t) / len(t) for p, t in zip(predictions, truth)])
    comparison = valid[['query_id', 'search_query', 'seen_text', 'partition']].copy()
    comparison['baseline'] = [r['baseline'] for r in results]
    comparison['improved'] = recall
    comparison.to_csv(CACHE / 'comparison.csv', index=False)
    summary = comparison.groupby('partition')[['baseline', 'improved']].mean()
    print('FULL CORPUS EVALUATION', summary.to_string(), flush=True)
    print('By seen text', comparison.groupby(['partition', 'seen_text'])[['baseline', 'improved']].mean().to_string(), flush=True)
    report = {'validation_queries': len(valid), 'seen_text_fraction': float(valid.seen_text.mean()),
        'metrics': summary.to_dict('index'), 'official_baseline_recall': .540141,
        'official_improved_recall': None}
    with (CACHE / 'metrics.json').open('w') as f:
        json.dump(report, f, indent=2)

    train = pd.read_parquet('data/train.parquet', columns=TRAIN_COLUMNS)
    queries = pd.read_parquet('data/benchmark_queries.parquet')
    predictions = predict(items, queries, train, text_indexes=text_indexes)
    save_answer(items, queries, predictions, Path('outputs/answer.csv'))
    validate(Path('outputs/answer.csv'))


def evaluate_simple():
    results = joblib.load(CACHE / 'features.joblib')
    _, valid, _ = joblib.load(CACHE / 'split.joblib')
    dev = np.flatnonzero(valid.partition.eq('dev'))
    profiles = {
        'old_text': [.58, .42, 0, 0],
        'all_fields': [.29, .21, .25, .25],
        'word_title': [.5, 0, .25, .25],
        'word_char_title': [.5, 0, 0, .5],
        'title_heavy': [.25, 0, .25, .5],
        'titles': [0, 0, .5, .5],
        'word_only': [.5, 0, .5, 0],
    }
    records = []
    for name, weights in profiles.items():
        for expansion in [0, .15, .3, .6]:
            for geo_weight in [0, 1, 3, 9, 19, 49]:
                scores = []
                for i in dev:
                    r = results[i]
                    s = r['fields'][:, :4] @ np.array(weights, dtype=np.float32)
                    s += expansion * r['fields'][:, 4]
                    s *= 1 + geo_weight * r['geo']
                    s *= .15 + .85 * r['category']
                    scores.append(r['hits'][top_indices(s, 50)].sum() / r['ntrue'])
                records.append(dict(profile=name, weights=weights, expansion=expansion,
                    geo_weight=geo_weight, recall=float(np.mean(scores))))
    report = pd.DataFrame(records).sort_values('recall', ascending=False)
    report.to_csv(CACHE / 'simple_tuning.csv', index=False)
    print(report.head(25).to_string(index=False), flush=True)
    print('Best per profile', report.groupby('profile').recall.max().to_dict())


def evaluate():
    results = joblib.load(CACHE / 'features.joblib')
    _, valid, _ = joblib.load(CACHE / 'split.joblib')
    dev = np.flatnonzero(valid.partition.eq('dev'))
    print('Baseline dev', np.mean([results[i]['baseline'] for i in dev]))
    records = []
    for title_weight in [0, .5, 1, 2]:
        for history_weight in [0, .15, .35]:
            for geo_weight in [0, 1, 3, 9]:
                for expansion in [0, .3, .7]:
                    scores = []
                    for i in dev:
                        r = results[i]; f = r['fields']
                        s = (.58*f[:, 0] + .42*f[:, 1] + title_weight*(.5*f[:, 2]+.5*f[:, 3])
                             + history_weight*r['history'] + expansion*f[:, 4])
                        s *= (1 + geo_weight * r['geo'])
                        s *= .15 + .85 * r['category']
                        order = top_indices(s, 50)
                        scores.append(r['hits'][order].sum() / r['ntrue'])
                    records.append(dict(title_weight=title_weight, history_weight=history_weight,
                        geo_weight=geo_weight, expansion=expansion, recall=float(np.mean(scores))))
    report = pd.DataFrame(records).sort_values('recall', ascending=False)
    report.to_csv(CACHE / 'tuning.csv', index=False)
    print(report.head(20).to_string(index=False), flush=True)


if __name__ == '__main__':
    import sys
    if sys.argv[-1] == 'finalize':
        finalize()
    elif sys.argv[-1] == 'simple':
        evaluate_simple()
    elif sys.argv[-1] == 'evaluate':
        evaluate()
    else:
        build_features()
        evaluate()
