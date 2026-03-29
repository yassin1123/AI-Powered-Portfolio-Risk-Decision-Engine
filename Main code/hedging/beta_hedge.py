"""Suggest SPY notional to offset portfolio beta."""

from __future__ import annotations

import pandas as pd


def portfolio_beta_to_spy(betas: pd.Series, weights: pd.Series) -> float:
    b = betas.reindex(weights.index).fillna(0.0)
    w = weights.fillna(0.0)
    return float((b * w).sum() / (w.sum() + 1e-12))
