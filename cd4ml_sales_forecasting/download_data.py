import argparse
import os

import polars as pl
import pyarrow.dataset as ds
import yaml
from dotenv import load_dotenv

_ = load_dotenv()

RAW_DATA = os.getenv('RAW_DATA')


def download_data(config_path: str) -> None:
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dset = ds.dataset(RAW_DATA, format='parquet')
    data = pl.scan_pyarrow_dataset(dset).collect()
    os.makedirs(
        os.path.dirname(config['data_load']['data_path']), exist_ok=True
    )
    data.write_parquet(config['data_load']['data_path'], compression='zstd')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    download_data(config_path=args.config)
