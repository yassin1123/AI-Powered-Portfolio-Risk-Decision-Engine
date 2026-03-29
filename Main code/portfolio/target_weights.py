"""Map combined signals to long-only or long-short weights, before constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pre.settings import AppSettings


def signals_to_weights(signals: pd.Series, settings: AppSettings) -> pd.Series:
    s = signals.fillna(0.0)
    if settings.portfolio.long_only:
        s = s.clip(lower=0.0)
    if s.abs().sum() < 1e-12:
        n = len(s)
        return pd.Series(1.0 / max(n, 1), index=s.index)
    if settings.portfolio.long_only:
        w = s / s.sum()
    else:
        w = s / s.abs().sum()
    return w.fillna(0.0)
