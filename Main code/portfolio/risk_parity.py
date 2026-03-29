"""Inverse-vol weights from diagonal vol proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(vol_per_asset: pd.Series) -> pd.Series:
    v = vol_per_asset.replace(0, np.nan).fillna(vol_per_asset.median())
    inv = 1.0 / v
    return inv / inv.sum()
