"""Flagship: z-score of DCC average correlation vs its rolling history → risk / positioning regime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pre.settings import AppSettings


@dataclass
class CorrRegimeSignalResult:
    corr_t: float
    corr_mean: float
    corr_std: float
    corr_z: float
    bucket: str
    """crisis_spike | diversification | normal"""

    reduce_exposure: bool
    activate_hedge: bool
    allow_more_risk: bool


def correlation_regime_signal(
    avg_corr_history: list[float] | np.ndarray,
    current_avg_corr: float,
    settings: AppSettings,
) -> CorrRegimeSignalResult:
    cfg = settings.correlation_signal
    hist = np.asarray(list(avg_corr_history), dtype=float)
    w = min(cfg.rolling_window, len(hist))
    if w < 10:
        tail = hist if len(hist) else np.array([current_avg_corr])
        m = float(np.mean(tail))
        s = float(np.std(tail, ddof=1)) if len(tail) > 1 else 1.0
    else:
        tail = hist[-w:]
        m = float(np.mean(tail))
        s = float(np.std(tail, ddof=1)) + cfg.eps_std
    z = (float(current_avg_corr) - m) / s if s > 0 else 0.0

    if z > cfg.z_high:
        bucket = "crisis_spike"
        reduce_exposure = True
        activate_hedge = True
        allow_more_risk = False
    elif z < cfg.z_low:
        bucket = "diversification"
        reduce_exposure = False
        activate_hedge = False
        allow_more_risk = True
    else:
        bucket = "normal"
        reduce_exposure = False
        activate_hedge = False
        allow_more_risk = False

    return CorrRegimeSignalResult(
        corr_t=float(current_avg_corr),
        corr_mean=m,
        corr_std=float(s),
        corr_z=float(z),
        bucket=bucket,
        reduce_exposure=reduce_exposure,
        activate_hedge=activate_hedge,
        allow_more_risk=allow_more_risk,
    )
