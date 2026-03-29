"""Cross-sectional relative strength: Gaussian rank scores or legacy quantile long/short."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pre.settings import AppSettings


def cross_sectional_scores(mom: pd.Series, settings: AppSettings) -> pd.Series:
    full_idx = mom.index
    m = mom.dropna()
    if len(m) < 2:
        return pd.Series(0.0, index=full_idx, dtype=float)

    if settings.alpha.xsec_use_rank_gaussian:
        r = m.rank(method="average")
        n = float(len(m))
        u = ((r - 0.5) / n).clip(0.001, 0.999)
        g = pd.Series(stats.norm.ppf(u.values), index=m.index, dtype=float)
        return g.reindex(full_idx).fillna(0.0).clip(-2.5, 2.5)

    q = float(settings.alpha.xsec_quantile)
    q = min(max(q, 0.05), 0.45)
    r = m.rank(pct=True)
    s = pd.Series(0.0, index=m.index)
    s[r >= 1.0 - q] = 1.0
    if not settings.portfolio.long_only:
        s[r <= q] = -1.0
    return s.reindex(full_idx).fillna(0.0).astype(float)
