"""Short-horizon trend / autocorr proxy for regime features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def return_autocorr_1d(log_returns: pd.Series, window: int = 60) -> float:
    x = log_returns.dropna().iloc[-window:]
    if len(x) < 10:
        return 0.0
    a = x.iloc[:-1].values
    b = x.iloc[1:].values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
