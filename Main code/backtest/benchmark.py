"""Baseline weight schemes for ladder comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.risk_parity import inverse_vol_weights


def equal_weight(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    v = 1.0 / n if n else 0.0
    return pd.Series({t: v for t in tickers})


def inverse_vol_from_returns(lr: pd.DataFrame, tickers: list[str], window: int) -> pd.Series:
    sub = lr[tickers].iloc[-window:]
    vol = sub.std(ddof=1).replace(0, np.nan)
    return inverse_vol_weights(vol.fillna(vol.median()))
