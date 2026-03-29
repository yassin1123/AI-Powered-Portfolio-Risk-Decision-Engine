"""Time-series momentum: multi-horizon return / EWMA-vol z-scores, horizon-weighted blend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha._signal_utils import ewma_vol, returns_for_alpha
from pre.settings import AppSettings


def momentum_scores(lr1: pd.DataFrame, settings: AppSettings) -> pd.Series:
    r = returns_for_alpha(lr1, settings)
    if len(r) < 2:
        return pd.Series(0.0, index=lr1.columns, dtype=float)

    lam = float(settings.alpha.mom_ewma_lambda)
    lam = min(max(lam, 0.5), 0.999)
    windows = [int(max(w, 1)) for w in settings.alpha.mom_windows]
    raw_w = list(settings.alpha.mom_window_weights or [])
    if raw_w and len(raw_w) == len(windows):
        hw = np.maximum(np.array(raw_w, dtype=float), 0.0)
        if hw.sum() < 1e-12:
            hw = np.ones(len(windows), dtype=float)
        hw = hw / hw.sum()
    else:
        hw = np.ones(len(windows), dtype=float) / max(len(windows), 1)

    floor_q = float(settings.alpha.mom_vol_floor_quantile)
    floor_q = min(max(floor_q, 0.0), 0.45)

    out: dict[str, float] = {}
    for col in lr1.columns:
        col_lr = r[col].astype(float)
        vols: list[float] = []
        for w in windows:
            seg = col_lr.iloc[-w:]
            if seg.notna().sum() < 2:
                continue
            v = ewma_vol(seg, lam)
            if np.isfinite(v) and v > 0:
                vols.append(v)
        vfloor = max(float(np.quantile(vols, floor_q)), 1e-8) if vols else 1e-8

        hz: list[float] = []
        for w in windows:
            seg = col_lr.iloc[-w:]
            if seg.notna().sum() < max(2, min(3, w // 2)):
                hz.append(0.0)
                continue
            acc = float(seg.sum())
            vol = ewma_vol(seg, lam)
            vol_eff = max(vol, vfloor) if np.isfinite(vol) else vfloor
            denom = vol_eff * np.sqrt(float(w))
            hz.append(float(np.clip(acc / denom, -3.0, 3.0)) if denom > 0 else 0.0)

        out[col] = float(np.dot(hw, np.array(hz, dtype=float)))

    return pd.Series(out, dtype=float).reindex(lr1.columns).fillna(0.0)
