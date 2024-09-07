import os
import pickle

import polars as pl
from sklearn.tree import DecisionTreeRegressor


def extract_features(data: pl.DataFrame, max_lag: int = 7) -> pl.DataFrame:
    return (
        data.sort(['item_nbr', 'date'])
        .with_columns([
            pl.col('unit_sales')
            .shift(n)
            .over('store_nbr')
            .name.suffix(f'_lag{n}')
            for n in range(1, max_lag + 1)
        ])
        .with_columns(
            pl.col('date').dt.weekday().alias('weekday'),
            pl.col('date').dt.day().alias('day'),
        )
        .drop_nulls()
        .select(
            [f'unit_sales_lag{n}' for n in range(1, max_lag + 1)]
            + ['weekday', 'day', 'unit_sales']
        )
    )


def train_dt(data_with_features: pl.DataFrame) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor()
    X = data_with_features.drop('unit_sales')
    y = data_with_features['unit_sales']

    model.fit(X, y)

    return model


def save_model(model: DecisionTreeRegressor, path: str = 'models/model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)
