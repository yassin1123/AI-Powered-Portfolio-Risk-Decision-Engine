"""Deterministic decision trace: rule id, drivers, confidence, mechanical audit lines.

Maps the ordered policy in DecisionEngine.decide to a quant-style audit trail.
"""

from __future__ import annotations

from typing import Any

from pre.settings import AppSettings


# Evaluation order must match DecisionEngine.decide (first match wins).
RULE_IDS: tuple[str, ...] = (
    "R1_STRESS_HIGH_CORR",
    "R2_CORR_CRISIS_Z",
    "R3_ANOMALY_SUPPRESS",
    "R4_STRESSED_REGIME",
    "R5_TRANSITION_REGIME",
    "R6_VAR_LIMIT_BREACH",
    "R7_DIVERSIFICATION_REGIME",
    "R8_DEFAULT_NORMAL",
)

PRIORITY_TO_RULE: dict[str, str] = {
    "stress_corr_override": "R1_STRESS_HIGH_CORR",
    "corr_crisis": "R2_CORR_CRISIS_Z",
    "anomaly_suppress": "R3_ANOMALY_SUPPRESS",
    "stressed_regime": "R4_STRESSED_REGIME",
    "transition": "R5_TRANSITION_REGIME",
    "var_breach_risk": "R6_VAR_LIMIT_BREACH",
    "diversification_regime": "R7_DIVERSIFICATION_REGIME",
    "normal": "R8_DEFAULT_NORMAL",
    "signals_only_neutral": "R8_DEFAULT_NORMAL",
}

PRIORITY_TO_SIGNAL: dict[str, str] = {
    "stress_corr_override": "REDUCE_RISK",
    "corr_crisis": "REDUCE_RISK",
    "anomaly_suppress": "REDUCE_RISK",
    "stressed_regime": "REDUCE_RISK",
    "var_breach_risk": "REDUCE_RISK",
    "transition": "TRANSITION_DEFENSIVE",
    "diversification_regime": "ALLOW_EXPAND_RISK",
    "normal": "MAINTAIN",
    "signals_only_neutral": "MAINTAIN",
}


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def compute_confidence(
    settings: AppSettings,
    *,
    regime_confidence: float,
    anomaly_count: int,
    drivers: list[dict[str, Any]],
    priority: str,
) -> float:
    """Higher when regime classifier is clear, fewer anomalies, multiple drivers agree on stress."""
    cfg = settings.decision_engine
    div = max(float(cfg.confidence_anomaly_divisor), 1e-9)
    rc = _clip(float(regime_confidence), 0.0, 1.0)
    noise = _clip(float(anomaly_count) / div, 0.0, 1.0)
    mass = sum(float(d.get("delta", 0.0)) for d in drivers)
    driver_agreement = _clip(mass / 0.85, 0.0, 1.0)
    trans_penalty = (
        float(cfg.confidence_transition_penalty) if priority == "transition" else 0.0
    )
    raw = 0.22 + 0.38 * rc + 0.28 * (1.0 - noise) + 0.22 * driver_agreement - trans_penalty
    return round(_clip(raw, 0.18, 0.92), 4)


def build_driver_facts(
    settings: AppSettings,
    *,
    regime_label: str,
    corr_z: float,
    anomaly_count: int,
    var_99: float,
    var_limit: float,
    drawdown: float,
    vol_ann: float,
    avg_corr: float,
    stress_high_corr: bool,
    high_corr_z: bool,
    anomaly: bool,
    stressed: bool,
    transition: bool,
    var_hot: bool,
    divers: bool,
) -> list[dict[str, Any]]:
    """All measurable inputs with score mass for traceability (not all fire the winning rule)."""
    cfg = settings.decision_engine
    cs = settings.correlation_signal
    drivers: list[dict[str, Any]] = []

    drivers.append(
        {
            "name": "correlation_instability",
            "delta": 0.22 if high_corr_z else (0.10 if corr_z > cs.z_high * 0.65 else 0.0),
            "detail": f"corr_z={corr_z:.3f} (z_high={cs.z_high}, z_low={cs.z_low})",
        }
    )
    drivers.append(
        {
            "name": "anomaly_stack",
            "delta": 0.28 if anomaly else (0.12 if anomaly_count >= 2 else 0.0),
            "detail": f"anomaly_count={anomaly_count} (suppress threshold={cfg.anomaly_suppress_count})",
        }
    )
    lim = float(var_limit)
    v99 = float(var_99)
    if v99 > lim:
        vd = 0.18
        vtxt = f"VaR99={v99*100:.2f}% > limit={lim*100:.2f}%"
    elif lim > 0 and v99 > 0.72 * lim:
        vd = 0.09
        vtxt = f"VaR99={v99*100:.2f}% within limits but elevated vs {lim*100:.2f}%"
    else:
        vd = 0.0
        vtxt = f"VaR99={v99*100:.2f}% (limit={lim*100:.2f}%)"
    drivers.append({"name": "tail_var", "delta": vd, "detail": vtxt})

    drivers.append(
        {
            "name": "drawdown",
            "delta": 0.14 if drawdown <= -0.05 else (0.07 if drawdown <= -0.03 else 0.0),
            "detail": f"portfolio_drawdown={drawdown*100:.2f}%",
        }
    )
    drivers.append(
        {
            "name": "macro_regime",
            "delta": 0.16 if stressed else (0.11 if transition else 0.0),
            "detail": f"regime_label={regime_label}",
        }
    )
    drivers.append(
        {
            "name": "stress_corr_combo",
            "delta": 0.24 if stress_high_corr else 0.0,
            "detail": f"STRESSED ∧ corr_z>{cfg.stress_corr_z_threshold} → systemic co-movement risk",
        }
    )
    drivers.append(
        {
            "name": "diversification_corr",
            "delta": 0.08 if divers else 0.0,
            "detail": f"corr_z<{cs.z_low} → correlation unusually low vs history",
        }
    )
    drivers.append(
        {
            "name": "vol_level",
            "delta": 0.06 if vol_ann > settings.portfolio.target_ann_vol * 1.2 else 0.0,
            "detail": f"predicted_ann_vol={vol_ann*100:.1f}% vs target={settings.portfolio.target_ann_vol*100:.1f}%",
        }
    )
    drivers.append(
        {
            "name": "diversification_breakdown",
            "delta": 0.12 if avg_corr > settings.avg_corr_alert else (0.06 if avg_corr > 0.45 else 0.0),
            "detail": f"avg_pairwise_corr={avg_corr:.3f} (alert>{settings.avg_corr_alert})",
        }
    )
    return drivers


def build_mechanical_lines(
    *,
    corr_z: float,
    anomaly_count: int,
    var_99: float,
    var_limit: float,
    drawdown: float,
    avg_corr: float,
    diversification_score: float,
    vol_ann: float,
    z_high: float,
    anomaly_thr: int,
) -> list[str]:
    cz_note = "↑ risk clustering" if corr_z > z_high else "within crisis band" if corr_z > z_high * 0.7 else "within typical band"
    return [
        f"Correlation instability: Z = {corr_z:.2f} ({cz_note})",
        f"Anomaly triggers: {anomaly_count} active (policy suppress at ≥{anomaly_thr})",
        f"VaR 99% (MC, 1d): {var_99*100:.2f}% vs limit {var_limit*100:.2f}%",
        f"Drawdown: {drawdown*100:.2f}%",
        f"Avg pairwise correlation: {avg_corr:.3f}",
        f"Diversification score (1 − avg_corr): {diversification_score:.3f}",
        f"Predicted ann. vol: {vol_ann*100:.1f}%",
    ]


def build_conclusion(priority: str, exposure_scale: float, regime_label: str) -> str:
    ex = float(exposure_scale)
    if priority == "transition":
        return (
            f"Conclusion: policy selects TRANSITION path. "
            f"Decision-layer risk multiplier = {ex:.2f}× on signals until regime clarity improves."
        )
    if priority in ("stress_corr_override", "corr_crisis", "anomaly_suppress", "stressed_regime", "var_breach_risk"):
        return (
            f"Conclusion: defensive path ({priority}). "
            f"Risk multiplier = {ex:.2f}×; hedge/arming per rule table."
        )
    if priority == "diversification_regime":
        return (
            f"Conclusion: correlation regime allows modest risk expansion. "
            f"Multiplier = {ex:.2f}× subject to portfolio constraints."
        )
    return (
        f"Conclusion: no stress override; maintain standard path under regime {regime_label}. "
        f"Multiplier = {ex:.2f}×."
    )


def format_driver_lines(drivers: list[dict[str, Any]]) -> list[str]:
    out = []
    for d in sorted(drivers, key=lambda x: -float(x.get("delta", 0.0))):
        delta = float(d.get("delta", 0.0))
        if delta <= 1e-9:
            continue
        out.append(f"{d.get('name', '?')}: +{delta:.2f} mass — {d.get('detail', '')}")
    return out


def build_decision_trace_dict(
    settings: AppSettings,
    *,
    priority: str,
    exposure_scale: float,
    regime_label: str,
    corr_z: float,
    anomaly_count: int,
    var_99: float,
    var_limit: float,
    drawdown: float,
    vol_ann: float,
    avg_corr: float,
    regime_confidence: float,
    stress_high_corr: bool,
    high_corr_z: bool,
    anomaly: bool,
    stressed: bool,
    transition: bool,
    var_hot: bool,
    divers: bool,
) -> dict[str, Any]:
    cfg = settings.decision_engine
    cs = settings.correlation_signal
    diversification_score = float(_clip(1.0 - avg_corr, 0.0, 1.0))

    drivers = build_driver_facts(
        settings,
        regime_label=regime_label,
        corr_z=corr_z,
        anomaly_count=anomaly_count,
        var_99=var_99,
        var_limit=var_limit,
        drawdown=drawdown,
        vol_ann=vol_ann,
        avg_corr=avg_corr,
        stress_high_corr=stress_high_corr,
        high_corr_z=high_corr_z,
        anomaly=anomaly,
        stressed=stressed,
        transition=transition,
        var_hot=var_hot,
        divers=divers,
    )
    confidence = compute_confidence(
        settings,
        regime_confidence=regime_confidence,
        anomaly_count=anomaly_count,
        drivers=drivers,
        priority=priority,
    )
    mechanical = build_mechanical_lines(
        corr_z=corr_z,
        anomaly_count=anomaly_count,
        var_99=var_99,
        var_limit=var_limit,
        drawdown=drawdown,
        avg_corr=avg_corr,
        diversification_score=diversification_score,
        vol_ann=vol_ann,
        z_high=cs.z_high,
        anomaly_thr=cfg.anomaly_suppress_count,
    )
    conclusion = build_conclusion(priority, exposure_scale, regime_label)

    rule_fired = PRIORITY_TO_RULE.get(priority, "R8_DEFAULT_NORMAL")

    return {
        "system_signal": PRIORITY_TO_SIGNAL.get(priority, "MAINTAIN"),
        "confidence": confidence,
        "decision_priority": priority,
        "winning_rule_id": rule_fired,
        "rule_evaluation_order": list(RULE_IDS),
        "condition_flags": {
            "R1_STRESS_HIGH_CORR": bool(stress_high_corr),
            "R2_CORR_CRISIS_Z": bool(high_corr_z),
            "R3_ANOMALY_SUPPRESS": bool(anomaly),
            "R4_STRESSED_REGIME": bool(stressed),
            "R5_TRANSITION_REGIME": bool(transition),
            "R6_VAR_LIMIT_BREACH": bool(var_hot),
            "R7_DIVERSIFICATION_REGIME": bool(divers),
        },
        "policy_note": (
            "First true branch in order R1→R2→…→R8 wins (see decision_policy.md). "
            f"Selected: {rule_fired}."
        ),
        "drivers": drivers,
        "driver_lines": format_driver_lines(drivers),
        "mechanical_lines": mechanical,
        "conclusion": conclusion,
        "positioning": {
            "risk_multiplier": float(exposure_scale),
            "lines": [],
        },
        "diversification_score": diversification_score,
        "inputs": {
            "corr_z": corr_z,
            "vol_ann": vol_ann,
            "var_99": var_99,
            "var_limit": var_limit,
            "drawdown": drawdown,
            "anomaly_count": anomaly_count,
            "avg_corr": avg_corr,
            "regime_label": regime_label,
            "regime_confidence": regime_confidence,
        },
    }
