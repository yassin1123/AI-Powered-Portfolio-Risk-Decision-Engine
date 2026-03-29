"""Vol / correlation / tail multipliers and counterfactual branches (backend brief §10).

All functions return **new** dicts; they never mutate inputs. They do not run the full
optimizer — they adjust the JSON-shaped objects a future UI or API might compare to baseline.
"""

from __future__ import annotations

from typing import Any


def delta_vs_base(base: dict[str, Any], shocked: dict[str, Any]) -> dict[str, Any]:
    """Shallow diff for scalar leaves present in both (nested dicts compared recursively one level)."""
    out: dict[str, Any] = {}
    for k, v in shocked.items():
        if k not in base:
            out[k] = {"base": None, "shocked": v}
            continue
        b = base[k]
        if isinstance(v, dict) and isinstance(b, dict):
            out[k] = delta_vs_base(b, v)
        elif v != b:
            out[k] = {"base": b, "shocked": v}
    return out


def shock_market_state(
    market_state: dict[str, Any],
    *,
    vol_ann_mult: float = 1.0,
    corr_z_add: float = 0.0,
    tail_mult_mult: float = 1.0,
    portfolio_drawdown_add: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Apply multiplicative / additive stress to key elite snapshot fields.

    Returns ``(shocked_market_state, deltas)`` where ``deltas`` holds numeric changes only.
    """
    ms = dict(market_state)
    deltas: dict[str, Any] = {}

    if vol_ann_mult != 1.0:
        old = float(ms.get("vol_ann_forecast", 0.0))
        new = old * vol_ann_mult
        ms["vol_ann_forecast"] = new
        deltas["vol_ann_forecast"] = new - old

    if corr_z_add != 0.0:
        old = float(ms.get("corr_z", 0.0))
        new = old + corr_z_add
        ms["corr_z"] = new
        deltas["corr_z"] = corr_z_add

    if tail_mult_mult != 1.0:
        old = float(ms.get("tail_multiplier", 1.0))
        new = old * tail_mult_mult
        ms["tail_multiplier"] = new
        deltas["tail_multiplier"] = new - old

    if portfolio_drawdown_add != 0.0:
        old = float(ms.get("portfolio_drawdown", 0.0))
        new = old + portfolio_drawdown_add
        ms["portfolio_drawdown"] = new
        deltas["portfolio_drawdown"] = portfolio_drawdown_add

    return ms, deltas


def apply_decision_overrides(
    decision: dict[str, Any],
    *,
    risk_multiplier: float | None = None,
    activate_hedge: bool | None = None,
    suppress_non_defensive: bool | None = None,
    decision_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy decision snapshot with optional scalar overrides (provisional what-if)."""
    d = dict(decision)
    ch: dict[str, Any] = {}
    if risk_multiplier is not None:
        ch["risk_multiplier"] = {"before": d.get("risk_multiplier"), "after": risk_multiplier}
        d["risk_multiplier"] = float(risk_multiplier)
        d["exposure_scale"] = float(risk_multiplier)
    if activate_hedge is not None:
        ch["activate_hedge"] = {"before": d.get("activate_hedge"), "after": activate_hedge}
        d["activate_hedge"] = bool(activate_hedge)
    if suppress_non_defensive is not None:
        ch["suppress_non_defensive"] = {
            "before": d.get("suppress_non_defensive"),
            "after": suppress_non_defensive,
        }
        d["suppress_non_defensive"] = bool(suppress_non_defensive)
    if decision_label is not None:
        ch["decision_label"] = {"before": d.get("decision_label"), "after": decision_label}
        d["decision_label"] = str(decision_label)
    return d, ch


def counterfactual_force_exposure(decision: dict[str, Any], exposure_scale: float = 1.0) -> dict[str, Any]:
    """Force baseline exposure (mirrors ``AblationFlags`` full decision path conceptually)."""
    d, _ = apply_decision_overrides(decision, risk_multiplier=exposure_scale)
    d["counterfactual_note"] = "forced_exposure_scale"
    return d


def counterfactual_zero_anomalies(market_state: dict[str, Any]) -> dict[str, Any]:
    """Hypothetical: anomaly detectors read zero (cf. anomaly gating ablation)."""
    m = dict(market_state)
    m["anomaly_count"] = 0
    tf = dict(m.get("trigger_flags") or {})
    tf["anomaly_elevated"] = False
    m["trigger_flags"] = tf
    m["counterfactual_note"] = "anomaly_count_forced_zero"
    return m


def counterfactual_disable_regime_gating_note() -> dict[str, Any]:
    """No portfolio recompute — documents the branch aligned with ``no_regime_gating`` ablation."""
    return {
        "regime_gating": "disabled_hypothetical",
        "reference_ablation": "no_regime_gating",
        "implementation": "use_backtest_engine_with_AblationFlags_use_regime_gating_False",
    }
