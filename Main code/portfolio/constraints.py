"""Position caps, turnover, cash floor."""

from __future__ import annotations

import pandas as pd

from pre.settings import AppSettings


def apply_constraints(
    weights: pd.Series,
    prior: pd.Series | None,
    settings: AppSettings,
    *,
    apply_turnover_cap: bool = True,
) -> pd.Series:
    w = weights.reindex(weights.index).fillna(0.0)
    cap = settings.portfolio.max_single_weight
    w = w.clip(lower=0.0 if settings.portfolio.long_only else -cap, upper=cap)
    gross = w.abs().sum()
    if gross > settings.portfolio.max_gross_leverage:
        w = w * (settings.portfolio.max_gross_leverage / gross)
    if apply_turnover_cap and prior is not None and len(prior):
        p = prior.reindex(w.index).fillna(0.0)
        turn = float((w - p).abs().sum())
        if turn > settings.portfolio.turnover_cap and turn > 0:
            # shrink step toward prior
            alpha = settings.portfolio.turnover_cap / turn
            w = p + alpha * (w - p)
    # cash floor: scale down if we model cash explicitly — here implicit in gross < 1
    if settings.portfolio.min_cash_weight > 0:
        max_gross = 1.0 - settings.portfolio.min_cash_weight
        g = w.sum() if settings.portfolio.long_only else w.abs().sum()
        if g > max_gross:
            w = w * (max_gross / g)
    return w
