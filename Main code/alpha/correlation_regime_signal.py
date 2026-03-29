"""Hero signal (elite brief §4.3): correlation-stress z-score.

**Definition (frozen for research):** Let ρ_t be the sample average pairwise correlation of
log-return residuals over the trailing covariance window (same window as the backtest loop).
Let μ_t and σ_t be the mean and (ddof=1) standard deviation of the last W values of ρ in the
expanding history (W = `correlation_signal.rolling_window`, or full history if W < 10 bars).
The **hero score** is z_t = (ρ_t − μ_t) / (σ_t + ε) with ε = `correlation_signal.eps_std`.

**Buckets:** z_t > z_high → crisis_spike; z_t < z_low → diversification; else normal.
This is intentionally simple and testable; eigenvalue-share variants can plug in behind the same API later.
"""

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


def neutral_corr_regime_signal(current_avg_corr: float) -> CorrRegimeSignalResult:
    """Ablations: correlation path present but non-informative (always normal bucket)."""
    return CorrRegimeSignalResult(
        corr_t=float(current_avg_corr),
        corr_mean=float(current_avg_corr),
        corr_std=1.0,
        corr_z=0.0,
        bucket="normal",
        reduce_exposure=False,
        activate_hedge=False,
        allow_more_risk=False,
    )


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
