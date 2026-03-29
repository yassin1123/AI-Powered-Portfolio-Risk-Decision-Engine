"""Apply target weights at mark prices (next-bar assumption in engine)."""

from __future__ import annotations

import pandas as pd


def weights_to_holdings_notional(
    weights: pd.Series, equity: float, prices: pd.Series
) -> tuple[pd.Series, float]:
    w = weights.reindex(prices.index).fillna(0.0)
    if w.sum() > 1e-12:
        w = w / w.sum()
    dollars = equity * w
    shares = dollars / prices.replace(0, pd.NA)
    shares = shares.fillna(0.0)
    invested = float((shares * prices).sum())
    cash = equity - invested
    return shares, cash
