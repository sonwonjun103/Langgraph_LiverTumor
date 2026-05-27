"""Train/test split helper based on register_dice_sweep.xlsx.

Sorts the sweep by ``sort_column`` (default ``mean1``) ascending; the first
``test_size`` rows become the test set, the rest become the train set.
The sweep is now the only source of truth for case lists — alldata_metrics
is no longer consulted.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def split_train_test(
    sweep_xlsx: str,
    test_size: int,
    sort_column: str = "mean1",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sweep = pd.read_excel(sweep_xlsx)
    sweep_sorted = sweep.sort_values(sort_column).reset_index(drop=True)
    test_df = sweep_sorted.head(test_size).reset_index(drop=True)
    train_df = sweep_sorted.iloc[test_size:].reset_index(drop=True)
    return train_df, test_df
