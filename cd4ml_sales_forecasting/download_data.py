import os

import polars as pl
import pyarrow.dataset as ds
from dotenv import load_dotenv

_ = load_dotenv()

RAW_DATA = os.getenv('RAW_DATA')


def download_data():
    dset = ds.dataset(RAW_DATA, format='parquet')
    return pl.scan_pyarrow_dataset(dset).collect()
