"""Scale gross exposure to target annualized vol using forecast portfolio vol."""

from __future__ import annotations

import pandas as pd


def vol_target_scale(
    weights: pd.Series,
    forecast_ann_vol: float,
    target_ann_vol: float,
    max_leverage: float,
) -> pd.Series:
    if forecast_ann_vol <= 1e-8:
        return weights
    scale = min(target_ann_vol / forecast_ann_vol, max_leverage)
    return weights * scale
