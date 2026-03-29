"""
DashboardSnapshot: single writer (async risk pipeline), read-only Dash callbacks.

Frozen after construction; pipeline builds a new instance each cycle and replaces
the published reference (see main.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DashboardSnapshot:
    """Immutable UI state for one risk cycle."""

    generated_at: datetime
    cycle_ms: float
    regime: str
    header: dict[str, Any] = field(default_factory=dict)
    var_panel: dict[str, Any] = field(default_factory=dict)
    correlation: dict[str, Any] = field(default_factory=dict)
    mc_distribution: dict[str, Any] = field(default_factory=dict)
    risk_attribution: dict[str, Any] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    garch_vol_paths: dict[str, list[float]] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    target_weights: dict[str, float] = field(default_factory=dict)
    tickers: list[str] = field(default_factory=list)
    system_state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    overlay_series: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def empty(message: str | None = None) -> DashboardSnapshot:
        return DashboardSnapshot(
            generated_at=datetime.now(timezone.utc),
            cycle_ms=0.0,
            regime="CALM",
            error=message,
            overlay_series={},
        )
