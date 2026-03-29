"""Mean reversion: residual short-horizon return vs slow drift (or legacy inverted short sum)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha._signal_utils import ewma_vol, returns_for_alpha
from pre.settings import AppSettings


def mean_reversion_scores(lr1: pd.DataFrame, settings: AppSettings) -> pd.Series:
    r = returns_for_alpha(lr1, settings)
    w = int(max(settings.alpha.mr_window, 1))
    lam = float(settings.alpha.mom_ewma_lambda)
    lam = min(max(lam, 0.5), 0.999)

    if not settings.alpha.mr_use_residual or len(r) < w + 2:
        seg = r.iloc[-w:] if len(r) >= w else r
        if len(seg) < 2:
            return pd.Series(0.0, index=lr1.columns, dtype=float)
        acc = seg.sum()
        vol = seg.std(ddof=1).replace(0, np.nan)
        z = -(acc / vol / np.sqrt(float(w))).clip(-3, 3)
        return z.reindex(lr1.columns).fillna(0.0).astype(float)

    S = int(max(settings.alpha.mr_slow_window, w + 1))
    out: dict[str, float] = {}
    for col in lr1.columns:
        col_lr = r[col].astype(float)
        if len(col_lr) < S:
            out[col] = 0.0
            continue
        slow_mean = float(col_lr.iloc[-S:].mean())
        fast = col_lr.iloc[-w:]
        if fast.notna().sum() < 2:
            out[col] = 0.0
            continue
        fast_sum = float(fast.sum())
        drift_exp = slow_mean * float(w)
        residual = fast_sum - drift_exp
        vol = ewma_vol(fast, lam)
        if not np.isfinite(vol) or vol <= 0:
            vol = float(fast.std(ddof=1) or 1e-8)
        vol_eff = max(vol, 1e-8)
        z = -(residual / vol_eff / np.sqrt(float(w)))
        out[col] = float(np.clip(z, -3.0, 3.0))

    return pd.Series(out, dtype=float).reindex(lr1.columns).fillna(0.0)
