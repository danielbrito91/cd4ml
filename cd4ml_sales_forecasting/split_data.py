from datetime import datetime
from typing import Tuple

import polars as pl


def split_data(
    data, test_start: datetime = datetime(2016, 11, 1)
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    train = data.filter(pl.col('date') < test_start)
    test = data.filter(pl.col('date') >= test_start)

    return train, test
