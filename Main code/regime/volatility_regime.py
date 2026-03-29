"""Rolling realized volatility feature for regime / HMM stack."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ann_vol(log_returns: pd.Series, window: int = 21) -> float:
    x = log_returns.dropna().iloc[-window:]
    if len(x) < 5:
        return 0.2
    return float(np.sqrt(252) * x.std(ddof=1))
