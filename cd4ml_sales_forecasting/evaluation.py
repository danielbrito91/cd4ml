import argparse
import json
import os
import pickle

import polars as pl
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cd4ml_sales_forecasting.decision_tree import extract_features


def load_data(config_path: str) -> pl.DataFrame:
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return pl.read_parquet(config['data_split']['test_path'])


def evaluate(test_data, config_path: str) -> dict:
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    os.makedirs(
        os.path.dirname(config['evaluate']['metrics_path']), exist_ok=True
    )
    test_data_with_fts = extract_features(test_data, config_path)

    with open(config['train']['model_path'], 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(test_data_with_fts.drop('unit_sales'))
    y_true = test_data_with_fts['unit_sales']

    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    with open(config['evaluate']['metrics_path'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f)

    return metrics


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data = load_data(args.config)
    evaluate(data, args.config)
