"""Aggregate metric by regime label series."""

from __future__ import annotations

import pandas as pd


def summarize_by_regime(series: pd.Series, regime: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"x": series, "regime": regime})
    return df.groupby("regime")["x"].agg(["mean", "std", "count"])
