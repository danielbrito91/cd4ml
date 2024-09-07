import argparse
import os
from datetime import datetime

import polars as pl
import yaml


def split_data(
    config_path: str,
) -> None:
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data = pl.read_parquet(config['data_load']['data_path'])
    test_start = datetime.strptime(
        config['data_split']['test_start_date'], '%Y-%m-%d'
    )

    train = data.filter(pl.col('date') < test_start)
    test = data.filter(pl.col('date') >= test_start)

    os.makedirs(
        os.path.dirname(config['data_split']['train_path']), exist_ok=True
    )

    train.write_parquet(config['data_split']['train_path'], compression='zstd')
    test.write_parquet(config['data_split']['test_path'], compression='zstd')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    split_data(config_path=args.config)
