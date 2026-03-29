"""Period PnL from holdings and price change."""

from __future__ import annotations

import pandas as pd


def period_pnl(holdings: pd.Series, px0: pd.Series, px1: pd.Series) -> float:
    h = holdings.reindex(px1.index).fillna(0.0)
    return float((h * (px1 - px0)).sum())
