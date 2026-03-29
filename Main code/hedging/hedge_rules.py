"""Compose hedge actions from decision engine + tail + beta."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.decision.decision_engine import Decision
from hedging.beta_hedge import portfolio_beta_to_spy
from hedging.tail_hedge import tail_hedge_fraction
from pre.settings import AppSettings


@dataclass
class HedgeBundle:
    spy_hedge_notional_fraction: float
    tail_overlay: float
    notes: str


def compute_hedge_bundle(
    decision: Decision,
    betas: pd.Series,
    weights: pd.Series,
    tail_mult: float,
    settings: AppSettings,
) -> HedgeBundle:
    pb = portfolio_beta_to_spy(betas, weights)
    spy_frac = 0.0
    if decision.activate_hedge:
        spy_frac = min(0.35, max(0.0, pb * 0.4))
    tail = tail_hedge_fraction(tail_mult, settings.tail_multiplier_hedge * 0.9)
    notes = []
    if decision.activate_hedge:
        notes.append("corr_or_stress_hedge")
    if tail > 0:
        notes.append("tail_overlay")
    return HedgeBundle(
        spy_hedge_notional_fraction=float(spy_frac + tail * 0.5),
        tail_overlay=float(tail),
        notes=",".join(notes) if notes else "none",
    )
