"""Shared helpers for alpha construction (alignment, trimming)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pre.settings import AppSettings


def ewma_vol(series: pd.Series, lam: float) -> float:
    s = series.dropna()
    if len(s) < 2:
        return float(np.nan)
    alpha = 1.0 - lam
    v = s.ewm(alpha=alpha, adjust=False).std().iloc[-1]
    return float(v) if np.isfinite(v) else float(np.nan)


def returns_for_alpha(lr1: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    """Optionally drop the most recent bar(s) so signals use only fully settled returns."""
    skip = int(getattr(settings.alpha, "mom_skip_recent_bars", 0) or 0)
    if skip > 0 and len(lr1) > skip:
        return lr1.iloc[:-skip]
    return lr1
