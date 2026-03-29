"""Slice decision_log by regime; performance + sample-size warnings (backend brief §8.2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest.evaluation import summarize_backtest

MIN_BARS_WARN = 30


def performance_by_regime(
    decision_log: list[dict[str, Any]],
    *,
    risk_free_annual: float = 0.0,
    min_bars_warn: int = MIN_BARS_WARN,
) -> dict[str, Any]:
    """
    Build a synthetic equity curve per regime from contiguous slices of ``pnl_frac`` rows
    in that regime (not a full backtest path — useful for coarse regime attribution).
    """
    if not decision_log:
        return {"by_regime": {}, "warnings": ["empty_decision_log"]}

    df = pd.DataFrame(decision_log)
    if "regime" not in df.columns or "pnl_frac" not in df.columns:
        return {"by_regime": {}, "warnings": ["missing_regime_or_pnl_frac"]}

    by_regime: dict[str, Any] = {}
    warnings: list[str] = []

    for regime, g in df.groupby("regime"):
        r = g["pnl_frac"].astype(float).dropna()
        n = len(r)
        label = str(regime)
        if n < min_bars_warn:
            warnings.append(f"regime_{label}_sample_small_n={n}")
        if n < 5:
            by_regime[label] = {
                "n_bars": n,
                "metrics": {},
                "warning": "too_few_bars_for_metrics",
            }
            continue
        log_eq = np.cumsum(r.values)
        eq = pd.Series(np.exp(log_eq), index=range(n))
        turn = pd.Series(0.0, index=eq.index)
        m = summarize_backtest(eq, turn, risk_free_annual)
        by_regime[label] = {
            "n_bars": n,
            "metrics": {k: float(v) for k, v in m.items()},
        }

    return {"by_regime": by_regime, "warnings": warnings}
