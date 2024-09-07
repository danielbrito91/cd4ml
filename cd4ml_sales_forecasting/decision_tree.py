import argparse
import os
import pickle

import polars as pl
import yaml
from sklearn.tree import DecisionTreeRegressor


def load_data(config_path: str) -> pl.DataFrame:
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return pl.read_parquet(config['data_split']['train_path'])


def extract_features(data, config_path: str) -> pl.DataFrame:
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    max_lags = config['features']['max_lags']
    return (
        data.sort(['item_nbr', 'date'])
        .with_columns([
            pl.col('unit_sales')
            .shift(n)
            .over('store_nbr')
            .name.suffix(f'_lag{n}')
            for n in range(1, max_lags + 1)
        ])
        .with_columns(
            pl.col('date').dt.weekday().alias('weekday'),
            pl.col('date').dt.day().alias('day'),
        )
        .drop_nulls()
        .select(
            [f'unit_sales_lag{n}' for n in range(1, max_lags + 1)]
            + ['weekday', 'day', 'unit_sales']
        )
    )


def train_dt(data_with_features: pl.DataFrame) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(random_state=1)
    X = data_with_features.drop('unit_sales')
    y = data_with_features['unit_sales']

    model.fit(X, y)

    return model


def save_model(model: DecisionTreeRegressor, config_path: str) -> None:
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(config['train']['model_path']), exist_ok=True)

    with open(config['train']['model_path'], 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data = load_data(args.config)
    data_with_features = extract_features(data, args.config)
    model = train_dt(data_with_features)
    save_model(model, args.config)
