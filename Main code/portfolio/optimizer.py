"""Regime-conditioned: STRESSED min-var, TRANSITION capped, CALM signal-weighted."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pre.settings import AppSettings


def optimize_weights(
    regime_label: str,
    signals: pd.Series,
    sigma: np.ndarray,
    tickers: list[str],
    settings: AppSettings,
) -> pd.Series:
    n = len(tickers)
    s = signals.reindex(tickers).fillna(0.0).values.astype(float)
    if regime_label == "STRESSED":
        return _min_variance(sigma, tickers, settings)
    if regime_label == "TRANSITION":
        w = _signal_inv_vol(signals, sigma, tickers)
        w = w * settings.portfolio.max_gross_leverage * 0.75
        return w / (w.sum() + 1e-12)
    return _signal_inv_vol(signals, sigma, tickers)


def _min_variance(sigma: np.ndarray, tickers: list[str], settings: AppSettings) -> pd.Series:
    n = len(tickers)

    def obj(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    x0 = np.ones(n) / n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, settings.portfolio.max_single_weight)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x if res.success else x0
    w = np.clip(w, 0, settings.portfolio.max_single_weight)
    w = w / w.sum()
    return pd.Series(w, index=tickers)


def _signal_inv_vol(signals: pd.Series, sigma: np.ndarray, tickers: list[str]) -> pd.Series:
    """Long-only signal sleeve: negative scores are zeroed before vol scaling (no short legs)."""
    d = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
    sig = pd.Series(d, index=tickers)
    raw = signals.reindex(tickers).fillna(0.0)
    pos = raw.clip(lower=0.0)
    if pos.sum() < 1e-12:
        w = 1.0 / sig
    else:
        w = pos / sig
    w = w / w.sum()
    return w.fillna(0.0)
