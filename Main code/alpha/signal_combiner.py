"""Combine signal families with config weights; correlation regime adds global tilt."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.carry import carry_scores
from alpha.correlation_regime_signal import CorrRegimeSignalResult
from alpha.mean_reversion import mean_reversion_scores
from alpha.momentum import momentum_scores
from alpha.cross_sectional import cross_sectional_scores
from pre.settings import AppSettings


@dataclass
class CombinedSignals:
    per_asset: pd.Series
    correlation_regime: CorrRegimeSignalResult
    breakdown: dict[str, float]


def _dynamic_sleeve_weights(
    w: dict[str, float],
    corr_result: CorrRegimeSignalResult,
    settings: AppSettings,
) -> tuple[float, float, float, float]:
    wm = float(w.get("momentum", 0.25))
    wmr = float(w.get("mean_reversion", 0.2))
    wxs = float(w.get("cross_sectional", 0.2))
    wc = float(w.get("carry", 0.1))
    if not settings.alpha.regime_dynamic_weights:
        return wm, wmr, wxs, wc
    if corr_result.bucket == "crisis_spike":
        mult = np.array([0.78, 1.12, 0.92, 1.0], dtype=float)
    elif corr_result.bucket == "diversification":
        mult = np.array([1.1, 0.88, 1.08, 1.02], dtype=float)
    else:
        mult = np.ones(4, dtype=float)
    v = np.array([wm, wmr, wxs, wc], dtype=float) * mult
    s0 = wm + wmr + wxs + wc
    s1 = float(v.sum())
    if s1 > 1e-12 and s0 > 1e-12:
        v = v * (s0 / s1)
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])


def combine_signals(
    lr1: pd.DataFrame,
    closes: pd.DataFrame,
    corr_result: CorrRegimeSignalResult,
    settings: AppSettings,
) -> CombinedSignals:
    mom = momentum_scores(lr1, settings)
    mr = mean_reversion_scores(lr1, settings)
    cr = carry_scores(closes, settings)
    xs = cross_sectional_scores(mom, settings)

    w = settings.alpha.combine_weights
    wm, wmr, wxs, wc = _dynamic_sleeve_weights(w, corr_result, settings)
    comb = wm * mom + wmr * mr + wxs * xs + wc * cr
    if settings.alpha.normalize_cross_section and len(comb) >= 2:
        sd = float(comb.std(ddof=1))
        if sd > 1e-10:
            comb = (comb - float(comb.mean())) / sd
    # Global tilt from correlation regime (scaled into level)
    crw = w.get("correlation_regime", 0.25)
    if corr_result.bucket == "crisis_spike":
        tilt = -crw
    elif corr_result.bucket == "diversification":
        tilt = crw
    else:
        tilt = 0.0
    comb = comb + tilt
    breakdown = {
        "momentum_mean": float(mom.mean()),
        "mr_mean": float(mr.mean()),
        "carry_mean": float(cr.mean()),
        "xsec_mean": float(xs.mean()),
        "corr_tilt": float(tilt),
        "weight_momentum": wm,
        "weight_mean_reversion": wmr,
        "weight_cross_sectional": wxs,
        "weight_carry": wc,
    }
    return CombinedSignals(
        per_asset=comb.fillna(0.0),
        correlation_regime=corr_result,
        breakdown=breakdown,
    )


def combine_signals_correlation_only(
    tickers: list[str],
    corr_result: CorrRegimeSignalResult,
    settings: AppSettings,
) -> CombinedSignals:
    """Only correlation-regime global tilt (no momentum/MR/carry/xsec sleeves)."""
    w = settings.alpha.combine_weights
    crw = w.get("correlation_regime", 0.25)
    if corr_result.bucket == "crisis_spike":
        tilt = -crw
    elif corr_result.bucket == "diversification":
        tilt = crw
    else:
        tilt = 0.0
    comb = pd.Series(tilt, index=tickers, dtype=float)
    return CombinedSignals(
        per_asset=comb,
        correlation_regime=corr_result,
        breakdown={"corr_tilt_only": float(tilt)},
    )
