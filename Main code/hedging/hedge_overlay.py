"""Human-readable hedge recommendation for snapshot / decision log."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.decision.decision_engine import Decision
from hedging.hedge_rules import HedgeBundle, compute_hedge_bundle
from pre.settings import AppSettings


@dataclass
class HedgeRecommendation:
    spy_short_fraction: float
    narrative: str
    bundle: HedgeBundle


def recommend_hedge(
    decision: Decision,
    betas: pd.Series,
    weights: pd.Series,
    tail_mult: float,
    settings: AppSettings,
) -> HedgeRecommendation:
    b = compute_hedge_bundle(decision, betas, weights, tail_mult, settings)
    narrative = f"SPY hedge ~{b.spy_hedge_notional_fraction:.1%} notional; {b.notes}"
    return HedgeRecommendation(
        spy_short_fraction=b.spy_hedge_notional_fraction,
        narrative=narrative,
        bundle=b,
    )
