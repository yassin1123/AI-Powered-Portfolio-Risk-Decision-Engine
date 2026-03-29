"""Stable import paths for walk-forward, ablations, and backtest engine (elite brief §3 research).

Canonical CLIs remain:

- ``python -m backtest.walkforward``
- ``python scripts/run_ablations.py``
- ``python -m backtest.run``

This module re-exports the underlying callables for programmatic use and docs.
"""

from __future__ import annotations

from backtest.ablation import AblationFlags
from backtest.engine import run_backtest
from backtest.walkforward import WalkforwardSlice, walk_forward, write_walkforward_manifest

__all__ = [
    "AblationFlags",
    "WalkforwardSlice",
    "run_backtest",
    "walk_forward",
    "write_walkforward_manifest",
]
