"""Rule-based regime: CALM / TRANSITION / STRESSED from vol, corr, tail, DD, anomalies."""

from __future__ import annotations

import numpy as np

from pre.settings import AppSettings


def rule_regime_features(
    tail_multiplier: float,
    avg_pairwise_corr: float,
    med_vol: float,
    portfolio_drawdown: float,
    anomaly_count: int,
    settings: AppSettings,
) -> dict[str, float]:
    ann_vol = float(med_vol * np.sqrt(252)) if med_vol > 0 else 0.0
    return {
        "tail_multiplier": float(tail_multiplier),
        "avg_pairwise_corr": float(avg_pairwise_corr),
        "med_daily_vol": float(med_vol),
        "ann_vol_proxy": ann_vol,
        "portfolio_drawdown": float(portfolio_drawdown),
        "anomaly_count": float(anomaly_count),
    }


def rule_regime_label(feats: dict[str, float], settings: AppSettings) -> str:
    rc = settings.regime
    tm = feats["tail_multiplier"]
    ac = feats["avg_pairwise_corr"]
    av = feats["ann_vol_proxy"]
    dd = feats["portfolio_drawdown"]
    an = int(feats["anomaly_count"])

    if tm > rc.stressed_tail_mult or ac > rc.stressed_avg_corr:
        return "STRESSED"
    if (
        tm > 1.15
        or ac > 0.4
        or av > 0.25
        or dd < -0.08
        or an >= rc.transition_anomaly_count
    ):
        return "TRANSITION"
    return "CALM"
