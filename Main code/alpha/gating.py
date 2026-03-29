"""Regime- and anomaly-conditioned multipliers on raw signal strength."""

from __future__ import annotations

import pandas as pd


def gate_signals(
    raw: pd.Series,
    regime_label: str,
    anomaly_count: int,
    *,
    stress_threshold_anomalies: int = 3,
    apply_regime: bool = True,
    apply_anomaly: bool = True,
) -> pd.Series:
    mult = 1.0
    if apply_regime:
        if regime_label == "STRESSED":
            mult *= 0.35
        elif regime_label == "TRANSITION":
            mult *= 0.7
    if apply_anomaly and anomaly_count >= stress_threshold_anomalies:
        mult *= 0.5
    return raw * mult
