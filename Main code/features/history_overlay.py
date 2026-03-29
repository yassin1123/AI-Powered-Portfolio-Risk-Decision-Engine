"""Full-span price history overlay for dashboard (2010→present): drawdown + corr z vs time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.correlation_regime_signal import correlation_regime_signal
from features.returns import portfolio_drawdown
from pre.settings import AppSettings


def _mean_pairwise_corr_upper(block: np.ndarray) -> float:
    """Mean of upper-triangle correlations; block shape (T, n)."""
    if block.shape[1] < 2 or block.shape[0] < 2:
        return float("nan")
    if not np.all(np.isfinite(block)):
        return float("nan")
    c = np.corrcoef(block.T)
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(c[iu]))


def rolling_mean_pairwise_corr(
    lr: pd.DataFrame, window: int, *, max_evals: int = 2500
) -> pd.Series:
    """Trailing average pairwise correlation over ``window`` bars.

    Uses striding + time interpolation so ~15y of daily data stays responsive in the UI.
    """
    lr = lr.dropna(how="any")
    if len(lr) < window or window < 5:
        return pd.Series(dtype=float)
    arr = lr.values.astype(np.float64)
    idx = lr.index
    T = len(idx)
    out = np.full(T, np.nan, dtype=float)
    span = T - window
    stride = max(1, (span + max_evals - 1) // max_evals)
    for t in range(window, T, stride):
        out[t] = _mean_pairwise_corr_upper(arr[t - window : t, :])
    out[T - 1] = _mean_pairwise_corr_upper(arr[T - window : T, :])
    ser = pd.Series(out, index=idx)
    if isinstance(ser.index, pd.DatetimeIndex):
        return ser.interpolate(method="time").bfill().ffill()
    return ser.interpolate().bfill().ffill()


def per_bar_corr_z_series(rho: pd.Series, settings: AppSettings) -> pd.Series:
    """Same z definition as live ``correlation_regime_signal``, applied along a ρ time series."""
    z_out = pd.Series(np.nan, index=rho.index)
    hist: list[float] = []
    for i in range(len(rho)):
        rv = float(rho.iloc[i])
        if not np.isfinite(rv):
            continue
        hist.append(rv)
        if len(hist) > 500:
            hist = hist[-500:]
        res = correlation_regime_signal(hist, rv, settings)
        z_out.iloc[i] = res.corr_z
    return z_out


def _corr_bucket_label(z: float, settings: AppSettings) -> str:
    cfg = settings.correlation_signal
    if not np.isfinite(z):
        return "NORMAL"
    if z > cfg.z_high:
        return "STRESSED"
    if z < cfg.z_low:
        return "CALM"
    return "NORMAL"


def build_full_span_overlay(
    closes: pd.DataFrame,
    lr1: pd.DataFrame,
    weights: pd.Series,
    settings: AppSettings,
) -> dict[str, object] | None:
    """
    Build aligned series for killer chart: calendar dates from ``history_start`` through last bar.
    """
    lr = lr1.dropna(how="any")
    if len(lr) < settings.covariance_window + 15:
        return None
    closes_a = closes.reindex(lr.index).dropna(how="any", axis=0)
    lr = lr.reindex(closes_a.index).dropna(how="any")
    closes_a = closes_a.reindex(lr.index).dropna(how="any", axis=0)
    if len(lr) < 30 or closes_a.shape[1] < 2:
        return None

    w = weights.reindex(closes_a.columns).fillna(0.0)
    if w.sum() != 0:
        w = w / w.sum()

    dd = portfolio_drawdown(closes_a, w)
    rho = rolling_mean_pairwise_corr(lr, settings.covariance_window)
    cz = per_bar_corr_z_series(rho, settings)
    reg = [_corr_bucket_label(float(cz.loc[i]) if i in cz.index and np.isfinite(cz.loc[i]) else float("nan"), settings) for i in lr.index]

    dates = [str(ix.date()) if hasattr(ix, "date") else str(ix) for ix in lr.index]
    dd_pct = [float(x) * 100.0 if np.isfinite(x) else float("nan") for x in dd.reindex(lr.index)]
    cz_l = [float(cz.loc[i]) if i in cz.index and np.isfinite(cz.loc[i]) else float("nan") for i in lr.index]

    return {
        "dates": dates,
        "drawdown_pct": dd_pct,
        "corr_z": cz_l,
        "regime": reg,
        "n_bars": len(dates),
        "history_start": str(lr.index[0].date()) if hasattr(lr.index[0], "date") else str(lr.index[0]),
        "history_end": str(lr.index[-1].date()) if hasattr(lr.index[-1], "date") else str(lr.index[-1]),
    }
