"""Legacy decision copy used when elite narrative is thin."""

from __future__ import annotations

from typing import Any

from dashboard import theme

CORR_Z_SPIKE = theme.CORR_Z_SPIKE


def action_line(ss: dict[str, Any]) -> str:
    dec = ss.get("decision") or {}
    pct = float(dec.get("exposure_scale") or 1.0) * 100
    hedge = bool(dec.get("activate_hedge"))
    pr = str(dec.get("priority") or ss.get("decision_priority") or "normal")
    if pr in ("normal", "diversification_regime", "signals_only_neutral") and pct >= 99 and not hedge:
        return "Maintain standard risk budget; no mandatory de-risk or hedge activation."
    parts = [f"Scale target exposure to ~{pct:.0f}% of baseline"]
    if hedge:
        parts.append("activate hedge overlay")
    return " + ".join(parts) + "."


def decision_explanation_text(ss: dict[str, Any]) -> str:
    dec = ss.get("decision") or {}
    pr = str(dec.get("priority") or ss.get("decision_priority") or "normal")
    cz = float(ss.get("corr_z") or 0.0)
    reg = str(ss.get("regime") or "")
    bucket = str(dec.get("corr_bucket") or ss.get("corr_bucket") or "")
    lines = {
        "stress_corr_override": (
            "Stressed regime coincides with very high correlation instability. "
            "That combination usually means systemic co-movement and fragile diversification. "
            "The engine overrides discretionary risk-taking, cuts exposure sharply, and prioritises defensive positioning."
        ),
        "corr_crisis": (
            "Correlation Instability (Z-score) has spiked above the crisis threshold. "
            "That indicates co-movement is unusually high versus its own history—diversification is doing less work when you need it most. "
            "The system scales risk down and turns hedging on to limit drawdowns."
        ),
        "anomaly_suppress": (
            "Multiple independent anomaly detectors fired together. "
            "That suggests the return distribution or structure of the book may be shifting. "
            "The system suppresses non-defensive risk until the picture stabilises."
        ),
        "stressed_regime": (
            "The regime classifier labels conditions as stressed (vol and/or correlation elevated). "
            "Exposure is reduced and hedges may be armed depending on the correlation bucket."
        ),
        "transition": (
            "Markets are in a transition regime—signals are noisier and correlations less stable. "
            "The engine dials gross exposure back until the state becomes clearer."
        ),
        "var_breach_risk": (
            "Estimated tail loss (VaR) is elevated versus your configured limit. "
            "The system trims risk budget before a larger shock crystallises."
        ),
        "diversification_regime": (
            "Correlation instability is unusually low—dispersion and diversification are more available. "
            "The engine can allow slightly fuller risk within constraints."
        ),
        "normal": (
            "No stress override is active: correlation instability, regime, anomalies, and VaR are not jointly breaching aggressive thresholds. "
            "The book follows the standard signal and risk-budget path."
        ),
        "signals_only_neutral": (
            "Decision layer is neutral (signals-only backtest mode); no regime-based de-risking is applied here."
        ),
    }
    body = lines.get(pr, lines["normal"])
    extra = []
    if cz > CORR_Z_SPIKE:
        extra.append(f"Current Z-score {cz:.2f} supports a correlation-focused read.")
    if reg and pr != "normal":
        extra.append(f"Regime label: {reg}.")
    if bucket and bucket != "none":
        extra.append(f"Correlation bucket: {bucket}.")
    tail = " " + " ".join(extra) if extra else ""
    return body + tail
