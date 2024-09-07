import json
import os
import pickle

import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cd4ml_sales_forecasting.decision_tree import extract_features


def evaluate(
    test_data: pl.DataFrame,
    model_path: str = 'models/model.pkl',
    report_path: str = 'reports/metrics.json',
) -> dict:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    test_data_with_fts = extract_features(test_data)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(test_data_with_fts.drop('unit_sales'))
    y_true = test_data_with_fts['unit_sales']

    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f)

    return metrics
