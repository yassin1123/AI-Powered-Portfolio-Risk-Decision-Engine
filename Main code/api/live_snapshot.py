"""Versioned JSON contract for live UI (backend brief §11.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.schemas import LiveSnapshotV1, SnapshotMeta


def build_live_snapshot_v1(
    *,
    settings_profile: str,
    cycle: int,
    cycle_ms: float,
    data_quality_warnings: list[str],
    market_state: dict[str, Any],
    decision: dict[str, Any],
    narrative: dict[str, Any],
    risk: dict[str, Any],
    portfolio: dict[str, Any],
    recent_changes: dict[str, Any],
    timeline: dict[str, Any],
    analogs: dict[str, Any],
    research_links: dict[str, Any],
    decision_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    meta = SnapshotMeta(
        schema_version="live_snapshot_v1",
        generated_at_utc=now.isoformat(),
        as_of_utc=now.isoformat(),
        cycle=cycle,
        cycle_ms=cycle_ms,
        universe_profile=settings_profile,
        data_quality_warnings=data_quality_warnings,
    )
    risk_out = dict(risk)
    if decision_trace is not None:
        risk_out["decision_trace"] = decision_trace

    snap = LiveSnapshotV1(
        meta=meta,
        market_state=market_state,
        decision=decision,
        narrative=narrative,
        risk=risk_out,
        portfolio=portfolio,
        recent_changes=recent_changes,
        timeline=timeline,
        analogs=analogs,
        research_links=research_links,
    )
    return snap.to_dict()


def elite_snapshot_minimal_placeholders() -> dict[str, Any]:
    """Explicit empty sections so UI never guesses missing vs failed."""
    return {
        "timeline": {"segments": [], "note": "not_populated"},
        "analogs": {"neighbors": [], "note": "not_populated"},
        "research_links": {"walkforward": "python -m backtest.walkforward", "ablations": "python scripts/run_ablations.py"},
    }
