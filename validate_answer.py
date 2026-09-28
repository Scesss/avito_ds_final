"""Проверка формата перед отправкой на Stepik; скрытую метрику не оценивает."""
import argparse
from pathlib import Path
import re

import pandas as pd


def validate(path, data_dir=Path('data')):
    answer = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8')
    queries = pd.read_parquet(data_dir / 'benchmark_queries.parquet', columns=['query_id'])
    items = set(pd.read_parquet(data_dir / 'benchmark_items.parquet', columns=['item_id']).item_id)
    if answer.columns.tolist() != ['query_id', 'answer']:
        raise ValueError('Нужны ровно две колонки: query_id,answer')
    if not answer.query_id.is_unique or set(answer.query_id) != set(queries.query_id):
        raise ValueError('Пропуски, лишние query_id или повторяющиеся запросы')
    counts = []
    for row in answer.itertuples(index=False):
        if len(row.query_id) != 16:
            raise ValueError(f'Неверная длина query_id: {row.query_id}')
        ids = row.answer.split(' ') if row.answer else []
        if len(ids) > 50 or len(ids) != len(set(ids)):
            raise ValueError(f'{row.query_id}: больше 50 ID либо есть повторы')
        if any(re.fullmatch('[0-9a-f]{16}', item) is None for item in ids):
            raise ValueError(f'{row.query_id}: нужны 16-символьные ID через один пробел')
        if not set(ids).issubset(items):
            raise ValueError(f'{row.query_id}: неизвестные item_id')
        counts.append(len(ids))
    print(f'OK: {len(answer)} запросов; от {min(counts)} до {max(counts)} ID на запрос.')
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path, nargs='?', default=Path('outputs/answer.csv'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    validate(args.path, args.data_dir)
