"""Train/test split helper based on register_dice_sweep.xlsx.

Treats the cases with the lowest ``mean1`` in ``register_dice_sweep.xlsx`` as
the held-out test set, and everything else in ``alldata_metrics.xlsx`` (i.e.
the rest of the catalog including cases that never appeared in the sweep) as
the train set.

Single entry point::

    train_df, test_df = split_train_test(metrics_xlsx, sweep_xlsx, test_size)

``train_df`` keeps the original row order from ``alldata_metrics.xlsx``;
``test_df`` is ordered the same way as the sweep (ascending mean1).
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def _norm(value) -> str:
    """Match main.format_path_part / inference flows: ints/floats integers
    become plain string, NaN becomes ``'0'``."""
    if pd.isna(value):
        return "0"
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _add_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_key"] = list(zip(df["subject"].map(_norm), df["date"].map(_norm)))
    return df


def split_train_test(
    metrics_xlsx: str,
    sweep_xlsx: str,
    test_size: int,
    sort_column: str = "mean1",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = _add_key(pd.read_excel(metrics_xlsx))
    sweep = _add_key(pd.read_excel(sweep_xlsx))

    test_keys_ordered = sweep.sort_values(sort_column).head(test_size)["_key"].tolist()
    test_set = set(test_keys_ordered)

    train_df = metrics[~metrics["_key"].isin(test_set)].drop(columns=["_key"]).reset_index(drop=True)

    metrics_lookup = metrics.set_index("_key")
    test_rows = [metrics_lookup.loc[k] for k in test_keys_ordered if k in metrics_lookup.index]
    if test_rows:
        test_df = pd.DataFrame(test_rows).reset_index(drop=True)
        if "_key" in test_df.columns:
            test_df = test_df.drop(columns=["_key"])
    else:
        test_df = metrics.iloc[0:0].drop(columns=["_key"]).copy()

    return train_df, test_df
