from __future__ import annotations

import numpy as np
import pandas as pd

from data.data_quality import compute_panel_quality


def test_quality_ok_on_full_synthetic_panel() -> None:
    rng = np.random.default_rng(0)
    n, k = 200, 4
    d = pd.date_range("2019-01-01", periods=n, freq="B")
    x = rng.standard_normal((n, k)).cumsum(axis=0) * 0.01 + 100
    df = pd.DataFrame(x, index=d, columns=[f"A{i}" for i in range(k)])
    r = compute_panel_quality(df)
    assert r.ok_for_backtest
    assert r.overlap_bars_all_valid == n
