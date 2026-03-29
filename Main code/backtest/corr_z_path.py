"""Expanding average-pairwise correlation z-score path (same definition as hero signal)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.correlation_regime_signal import correlation_regime_signal
from features.returns import compute_log_returns
from pre.settings import AppSettings


def _cov_to_corr(sigma: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
    R = sigma / np.outer(d, d)
    n = R.shape[0]
    if n < 2:
        return R, 0.0
    ac = float((np.sum(R) - np.trace(R)) / (n * (n - 1)))
    return R, ac


def correlation_z_series(closes: pd.DataFrame, settings: AppSettings, *, warmup: int = 130) -> pd.Series:
    """One z per end-date t (uses data ≤ t only). Index aligned to `closes.index[t]`.

    Uses the same log-return definition as `compute_features` / backtest (`compute_log_returns`).
    """
    tickers = list(closes.columns)
    lr_full = compute_log_returns(closes)
    z_vals: list[float] = []
    z_idx: list[pd.Timestamp] = []
    corr_hist: list[float] = []

    for t in range(warmup, len(closes)):
        lr_use = lr_full.iloc[: t + 1][tickers].dropna()
        if len(lr_use) < 40:
            continue
        tail = lr_use.iloc[-min(252, len(lr_use)) :]
        sigma = np.cov(tail.values.T)
        sigma = np.nan_to_num(sigma, nan=1e-6)
        _, avg_corr = _cov_to_corr(sigma)
        corr_hist.append(avg_corr)
        if len(corr_hist) > 400:
            corr_hist = corr_hist[-400:]
        res = correlation_regime_signal(corr_hist, avg_corr, settings)
        z_vals.append(res.corr_z)
        z_idx.append(closes.index[t])

    return pd.Series(z_vals, index=pd.DatetimeIndex(z_idx), name="corr_z")
