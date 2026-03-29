"""Curve carry: level + change in long-duration vs cash proxy; scaled by asset rate sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pre.settings import AppSettings

# Rough rate-beta tiers for common sleeve names (long-only sleeve uses positives only).
_HIGH_DURATION = frozenset({"TLT", "LTPZ", "EDV", "VGLT"})
_INTERMEDIATE = frozenset({"IEF", "VGIT", "IEI", "SPTL", "SCHR", "GOVT"})
_LOW_CASHLIKE = frozenset({"SHY", "BIL", "SGOV", "GBIL", "MINT", "JPST"})


def _carry_multiplier(ticker: str) -> float:
    u = str(ticker).upper()
    if u in _HIGH_DURATION:
        return 1.0
    if u in _INTERMEDIATE:
        return 0.72
    if u in _LOW_CASHLIKE:
        return 0.35
    return 0.45


def carry_scores(closes: pd.DataFrame, settings: AppSettings) -> pd.Series:
    cols = list(closes.columns)
    out = pd.Series(0.0, index=cols, dtype=float)
    if "TLT" not in cols or "SHY" not in cols:
        return out

    TLT = closes["TLT"].astype(float)
    SHY = closes["SHY"].astype(float)
    safe = (TLT > 0) & (SHY > 0)
    if not bool(safe.iloc[-1]):
        return out

    level = float(np.log(TLT.iloc[-1] / SHY.iloc[-1]))
    lb = int(max(settings.alpha.carry_change_bars, 1))
    if len(TLT) > lb and bool(safe.iloc[-1 - lb]):
        past = float(np.log(TLT.iloc[-1 - lb] / SHY.iloc[-1 - lb]))
        chg = level - past
    else:
        chg = 0.0

    wl = float(settings.alpha.carry_level_weight)
    wc = float(settings.alpha.carry_change_weight)
    s = float(settings.alpha.carry_tanh_scale)
    raw = wl * level + wc * chg
    base = float(np.tanh(raw * s))

    for c in cols:
        out[c] = base * _carry_multiplier(c)
    return out
