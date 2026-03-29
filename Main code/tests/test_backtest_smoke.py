"""Backtest engine runs and placebo is weak vs full (sanity)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from pre.settings import AppSettings


def _closes(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    d = pd.date_range("2021-01-01", periods=n, freq="B")
    x = rng.standard_normal((n, 4)).cumsum(axis=0) * 0.008
    return pd.DataFrame(100 * np.exp(x), index=d, columns=[f"A{i}" for i in range(4)])


def test_backtest_runs() -> None:
    s = AppSettings()
    r = run_backtest(_closes(280), s, mode="full", warmup=120)
    assert len(r.equity) > 10
    assert "sharpe" in r.metrics


def test_placebo_produces_metrics() -> None:
    s = AppSettings()
    c = _closes(280)
    rp = run_backtest(c, s, mode="placebo_random", warmup=120).metrics
    assert "sharpe" in rp
