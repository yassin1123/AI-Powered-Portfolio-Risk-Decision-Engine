"""Portfolio weights respect caps."""

from __future__ import annotations

import pandas as pd

from portfolio.constraints import apply_constraints
from pre.settings import AppSettings, PortfolioConfig


def test_max_weight_cap() -> None:
    s = AppSettings(portfolio=PortfolioConfig(max_single_weight=0.2, max_gross_leverage=1.0))
    w = pd.Series({"a": 0.9, "b": 0.1})
    out = apply_constraints(w, None, s)
    assert out.max() <= 0.200001
