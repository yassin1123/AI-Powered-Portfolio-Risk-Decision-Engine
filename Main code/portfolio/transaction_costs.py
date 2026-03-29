"""Proportional turnover cost."""

from __future__ import annotations

import pandas as pd

from pre.settings import AppSettings


def turnover_cost(prior: pd.Series, new: pd.Series, settings: AppSettings) -> float:
    p = prior.reindex(new.index).fillna(0.0)
    turn = float((new - p).abs().sum())
    return turn * settings.portfolio.cost_bps / 10000.0
