"""Функции исходного варианта для сравнения в experiment.py."""

from collections import defaultdict
import re

import numpy as np

QUERY_COLUMNS = [
    'search_query', 'search_location_id', 'search_is_delivery_search',
    'search_infm_params_text', 'search_category',
]


def as_text(values):
    return values.fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


def make_item_documents(items):
    title = as_text(items['item_title_raw'])
    params = as_text(items['item_infm_params_text'])
    description = as_text(items['item_description_raw'])
    return title + ' ' + title + ' ' + title + ' ' + params + ' ' + params + ' ' + description


def make_query_documents(queries):
    query = as_text(queries['search_query'])
    filters = as_text(queries['search_infm_params_text'])
    return query + ' ' + query + ' ' + query + ' ' + filters


def normalize_title(text):
    return re.sub(r'\W+', ' ', str(text).lower()).strip()


def top_indices(scores, k):
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    candidate = np.argpartition(scores, -k)[-k:]
    return candidate[np.argsort(scores[candidate])[::-1]]


def build_history(train, item_to_index, corpus_titles):
    exact = defaultdict(list)
    by_text = defaultdict(list)
    for row in train.itertuples(index=False):
        item_index = item_to_index.get(row.item_id)
        key = tuple(getattr(row, col) for col in QUERY_COLUMNS)
        if item_index is not None:
            exact[key].append(item_index)
        title_key = normalize_title(row.item_title_raw)
        if title_key:
            by_text[row.search_query].extend(corpus_titles.get(title_key, ()))
    return exact, by_text
