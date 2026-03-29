"""Research scripts and generated outputs."""

from research.backend_entrypoints import (
    AblationFlags,
    WalkforwardSlice,
    run_backtest,
    walk_forward,
    write_walkforward_manifest,
)
from research.by_regime_metrics import performance_by_regime
from research.failure_analysis import analyze_failure_windows, failure_summary_from_backtest

__all__ = [
    "AblationFlags",
    "WalkforwardSlice",
    "analyze_failure_windows",
    "failure_summary_from_backtest",
    "performance_by_regime",
    "run_backtest",
    "walk_forward",
    "write_walkforward_manifest",
]
