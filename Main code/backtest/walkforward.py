"""Rolling train/test windows over a long close history."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.engine import BacktestResult, run_backtest
from pre.settings import AppSettings


@dataclass
class WalkforwardSlice:
    train_end: int
    test_start: int
    test_end: int
    result: BacktestResult


def walk_forward(
    closes: pd.DataFrame,
    settings: AppSettings,
    *,
    mode: str = "full",
) -> list[WalkforwardSlice]:
    tr = settings.backtest.walkforward_train_bars
    te = settings.backtest.walkforward_test_bars
    out: list[WalkforwardSlice] = []
    start = tr
    while start + te < len(closes):
        sub = closes.iloc[: start + te].copy()
        res = run_backtest(sub, settings, mode=mode)  # type: ignore[arg-type]
        out.append(
            WalkforwardSlice(
                train_end=start,
                test_start=start,
                test_end=start + te,
                result=res,
            )
        )
        start += te
    return out
