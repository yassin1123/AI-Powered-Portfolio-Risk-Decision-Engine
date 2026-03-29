"""Assemble unified market_state dict for elite snapshot (backend brief §4.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from alpha.correlation_regime_signal import CorrRegimeSignalResult
from pre.settings import AppSettings
from regime.regime_state import RegimeOutput


def _norm_drivers_from_trace(drivers: list[dict[str, Any]] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not drivers:
        return out
    for d in drivers:
        name = str(d.get("name", "unknown"))
        out[name] = float(d.get("delta", 0.0))
    return out


def build_market_state(
    *,
    settings: AppSettings,
    ro: RegimeOutput,
    corr_result: CorrRegimeSignalResult,
    avg_corr: float,
    anom_count: int,
    portfolio_drawdown: float,
    forecast_ann_vol: float,
    tail_mult: float,
    risk_disagreement: bool,
    trace_drivers: list[dict[str, Any]] | None = None,
    stability_score: float = 0.5,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    feats = ro.features or {}
    driver_scores = _norm_drivers_from_trace(trace_drivers)

    z_hi = settings.correlation_signal.z_high
    z_lo = settings.correlation_signal.z_low
    triggers = {
        "high_corr_z": bool(corr_result.corr_z > z_hi),
        "low_corr_z": bool(corr_result.corr_z < z_lo),
        "corr_crisis_bucket": corr_result.bucket == "crisis_spike",
        "stressed_regime": ro.label == "STRESSED",
        "transition_regime": ro.label == "TRANSITION",
        "anomaly_elevated": anom_count >= settings.decision_engine.anomaly_suppress_count,
        "drawdown_watch": portfolio_drawdown <= -settings.anomaly.drawdown_watch,
    }

    return {
        "timestamp": now,
        "regime": ro.label,
        "regime_confidence": float(ro.confidence),
        "corr_level": float(avg_corr),
        "corr_z": float(corr_result.corr_z),
        "corr_bucket": corr_result.bucket,
        "anomaly_count": int(anom_count),
        "portfolio_drawdown": float(portfolio_drawdown),
        "vol_ann_forecast": float(forecast_ann_vol),
        "vol_ann_target": float(settings.portfolio.target_ann_vol),
        "tail_multiplier": float(tail_mult),
        "risk_disagreement_hs_mc": bool(risk_disagreement),
        "driver_scores": driver_scores,
        "trigger_flags": triggers,
        "stability_score": float(stability_score),
        "regime_feature_snapshot": {k: float(v) for k, v in feats.items() if isinstance(v, (int, float))},
        "prob_vector": dict(ro.prob_vector),
        "duration_bars": int(ro.duration_bars),
    }
