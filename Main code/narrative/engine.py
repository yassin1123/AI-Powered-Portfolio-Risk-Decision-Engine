"""Rule-based narrative from state + decision (backend brief §5; no LLM)."""

from __future__ import annotations

from typing import Any


def build_narrative(
    *,
    regime: str,
    corr_z: float,
    corr_bucket: str,
    anomaly_count: int,
    var_99: float,
    var_limit: float,
    forecast_vol: float,
    target_vol: float,
    risk_multiplier: float,
    decision_priority: str,
    activate_hedge: bool,
    breach_today: bool,
    var_trend: str,
) -> dict[str, Any]:
    # Sentence 1: state
    if regime == "STRESSED":
        s1 = f"Market structure is stressed (regime {regime}) with correlation in the {corr_bucket} bucket."
    elif regime == "TRANSITION":
        s1 = f"Markets are in a transition regime; correlation instability reads as {corr_bucket}."
    else:
        s1 = f"Conditions are relatively calm under regime {regime}; correlation bucket is {corr_bucket}."

    # Sentence 2: drivers
    parts2 = []
    if corr_z > 1.2:
        parts2.append("cross-asset correlation is elevated versus its own history")
    elif corr_z < -0.5:
        parts2.append("correlation is unusually low, so diversification is comparatively available")
    if forecast_vol > target_vol * 1.15:
        parts2.append("forecast volatility sits above the portfolio target")
    if anomaly_count >= 2:
        parts2.append("multiple anomaly detectors are firing")
    if var_99 > var_limit * 0.85:
        parts2.append("tail risk (VaR) is elevated relative to the configured limit")
    s2 = (
        "Drivers: " + "; ".join(parts2) + "."
        if parts2
        else "Drivers: no single stress dimension is dominating the read."
    )

    # Sentence 3: action
    if risk_multiplier < 0.95:
        act = f"The engine scales risk to about {risk_multiplier:.0%} of baseline signal exposure."
    else:
        act = "The engine maintains a standard risk budget on signals."
    if activate_hedge:
        act += " Hedge overlay is armed."
    s3 = act

    # Sentence 4: qualifier
    qual = []
    if breach_today:
        qual.append("Today's return breached the estimated VaR tail.")
    if var_trend == "increasing":
        qual.append("VaR estimates have been drifting up recently.")
    if anomaly_count >= 4:
        qual.append("High anomaly count warrants caution on signal reliability.")
    s4 = " ".join(qual) if qual else "No extreme qualifier flags beyond the baseline read."

    summary = f"{s1} {s2} {s3} {s4}"

    if risk_multiplier < 0.75 or decision_priority in (
        "stress_corr_override",
        "corr_crisis",
        "anomaly_suppress",
    ):
        headline = "Defensive posture — exposure reduced"
    elif regime == "TRANSITION":
        headline = "Transition regime — conviction trimmed"
    elif corr_bucket == "diversification":
        headline = "Diversification supportive — standard risk path"
    else:
        headline = "Risk-aware monitoring — maintain discipline"

    why_lines = [s1.strip(), s2.strip(), s3.strip(), s4.strip()]

    return {
        "headline": headline,
        "summary": summary,
        "why_lines": why_lines,
        "action_line": s3,
    }
