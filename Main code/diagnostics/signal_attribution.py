"""Rough PnL attribution by multiplying lagged signal sleeve means × forward returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_signal_pnl_attribution(
    signal_panel: pd.DataFrame,
    forward_ret: pd.Series,
) -> pd.Series:
    """signal_panel columns = sleeves, aligned index with forward_ret."""
    out = {}
    for c in signal_panel.columns:
        s = signal_panel[c].shift(1)
        out[c] = float((s * forward_ret).sum())
    return pd.Series(out)
