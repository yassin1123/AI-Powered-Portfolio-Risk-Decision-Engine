"""Cash and positions bookkeeping (simplified long-only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PortfolioBook:
    cash: float
    holdings: pd.Series
    equity: float = 0.0
    history: list[float] = field(default_factory=list)

    def mark(self, prices: pd.Series) -> float:
        v = float((self.holdings * prices).sum() + self.cash)
        self.equity = v
        self.history.append(v)
        return v
