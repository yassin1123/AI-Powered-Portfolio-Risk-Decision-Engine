"""Signals + risk + regime → trade / de-risk / hedge / suppress.

Single primary decision_priority per bar (ordered policy — not stacked mins).
Formal trace: core.decision.decision_trace (rule id, drivers, confidence).
See docs/decision_policy.md for the hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alpha.correlation_regime_signal import CorrRegimeSignalResult
from core.decision.decision_trace import build_decision_trace_dict
from pre.settings import AppSettings


@dataclass
class Decision:
    trade_allowed: bool
    exposure_scale: float
    override_signals: bool
    suppress_non_defensive: bool
    activate_hedge: bool
    """Primary policy id: stress_corr_override | corr_crisis | anomaly_suppress | ..."""

    decision_priority: str
    secondary_reasons: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    narrative: str = ""
    trace: dict[str, Any] | None = None


class DecisionEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def decide(
        self,
        regime_label: str,
        corr_result: CorrRegimeSignalResult,
        anomaly_count: int,
        var_99: float,
        risk_limit_var: float,
        *,
        regime_confidence: float = 0.65,
        drawdown: float = 0.0,
        vol_ann: float = 0.12,
        avg_corr: float = 0.35,
    ) -> Decision:
        cfg = self.settings.decision_engine
        cs = self.settings.correlation_signal

        stress_high_corr = (
            regime_label == "STRESSED"
            and corr_result.corr_z > cfg.stress_corr_z_threshold
        )
        high_corr_z = corr_result.corr_z > cs.z_high
        anomaly = anomaly_count >= cfg.anomaly_suppress_count
        stressed = regime_label == "STRESSED"
        transition = regime_label == "TRANSITION"
        var_hot = var_99 > risk_limit_var
        divers = corr_result.corr_z < cs.z_low

        secondary: list[str] = []
        if stress_high_corr:
            secondary.append("fact_stress_high_corr")
        if high_corr_z:
            secondary.append("fact_high_corr_z")
        if anomaly:
            secondary.append("fact_anomaly")
        if stressed:
            secondary.append("fact_stressed_regime")
        if transition:
            secondary.append("fact_transition")
        if var_hot:
            secondary.append("fact_var_hot")
        if divers:
            secondary.append("fact_diversification_corr")

        priority = "normal"
        override = False
        suppress = False
        hedge = False
        exposure = 1.0

        if stress_high_corr:
            priority = "stress_corr_override"
            override = True
            suppress = True
            hedge = True
            exposure = min(cfg.exposure_scale_stress, 0.5)
        elif high_corr_z:
            priority = "corr_crisis"
            hedge = True
            exposure = 0.55
        elif anomaly:
            priority = "anomaly_suppress"
            suppress = True
            exposure = 0.65
        elif stressed:
            priority = "stressed_regime"
            hedge = corr_result.activate_hedge
            exposure = cfg.exposure_scale_stress
        elif transition:
            priority = "transition"
            exposure = cfg.exposure_scale_transition
        elif var_hot:
            priority = "var_breach_risk"
            exposure = 0.7
        elif divers:
            priority = "diversification_regime"
            exposure = min(1.05, self.settings.portfolio.max_gross_leverage)
        else:
            priority = "normal"
            exposure = 1.0
            hedge = corr_result.activate_hedge and corr_result.bucket == "crisis_spike"

        primary_fact = {
            "stress_corr_override": "fact_stress_high_corr",
            "corr_crisis": "fact_high_corr_z",
            "anomaly_suppress": "fact_anomaly",
            "stressed_regime": "fact_stressed_regime",
            "transition": "fact_transition",
            "var_breach_risk": "fact_var_hot",
            "diversification_regime": "fact_diversification_corr",
            "normal": "",
        }.get(priority, "")
        codes = [s for s in secondary if s != primary_fact]

        narrative = f"{priority}"
        if codes:
            narrative += " | also: " + ",".join(codes)

        trace = build_decision_trace_dict(
            self.settings,
            priority=priority,
            exposure_scale=float(exposure),
            regime_label=regime_label,
            corr_z=float(corr_result.corr_z),
            anomaly_count=int(anomaly_count),
            var_99=float(var_99),
            var_limit=float(risk_limit_var),
            drawdown=float(drawdown),
            vol_ann=float(vol_ann),
            avg_corr=float(avg_corr),
            regime_confidence=float(regime_confidence),
            stress_high_corr=stress_high_corr,
            high_corr_z=high_corr_z,
            anomaly=anomaly,
            stressed=stressed,
            transition=transition,
            var_hot=var_hot,
            divers=divers,
        )

        return Decision(
            trade_allowed=True,
            exposure_scale=float(exposure),
            override_signals=override,
            suppress_non_defensive=suppress,
            activate_hedge=hedge,
            decision_priority=priority,
            secondary_reasons=codes,
            reason_codes=codes,
            narrative=narrative,
            trace=trace,
        )


def apply_decision_to_signals(
    signals: pd.Series, decision: Decision, regime_label: str
) -> pd.Series:
    s = signals.copy()
    if decision.override_signals:
        s = s * 0.25
    if decision.suppress_non_defensive:
        s = s * 0.5
    s = s * decision.exposure_scale
    if regime_label == "STRESSED" and decision.decision_priority not in (
        "diversification_regime",
        "normal",
        "signals_only_neutral",
    ):
        s = s.clip(-1.0, 1.0) * 0.8
    return s


def neutral_decision_for_signals_only() -> Decision:
    """Backtest mode: alpha + sizing without risk/regime decision layer."""
    return Decision(
        trade_allowed=True,
        exposure_scale=1.0,
        override_signals=False,
        suppress_non_defensive=False,
        activate_hedge=False,
        decision_priority="signals_only_neutral",
        secondary_reasons=[],
        reason_codes=[],
        narrative="signals_only: no decision-engine scaling",
        trace=None,
    )
