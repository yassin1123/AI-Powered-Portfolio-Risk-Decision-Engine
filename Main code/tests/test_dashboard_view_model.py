"""View model merge: elite_snapshot vs legacy snapshot."""

from __future__ import annotations

from datetime import datetime, timezone

from core.snapshot import DashboardSnapshot
from dashboard.view_model import build_ui_model


def test_build_ui_model_with_elite() -> None:
    elite = {
        "schema_version": "live_snapshot_v1",
        "meta": {"cycle": 3},
        "market_state": {"regime": "CALM", "regime_confidence": 0.8, "corr_z": 0.1},
        "narrative": {"headline": "H", "summary": "S", "why_lines": ["a"], "action_line": "act"},
        "decision": {"risk_multiplier": 1.0},
        "risk": {"tail": {}, "vs_target": {}},
        "recent_changes": {},
        "timeline": {},
        "analogs": {},
        "research_links": {"k": "v"},
    }
    snap = DashboardSnapshot(
        generated_at=datetime.now(timezone.utc),
        cycle_ms=1.0,
        regime="CALM",
        report={"elite_snapshot": elite},
        system_state={"regime": "STRESSED"},
    )
    ui = build_ui_model(snap)
    assert ui["has_elite"] is True
    assert ui["regime"] == "CALM"
    assert ui["narrative"]["headline"] == "H"


def test_build_ui_model_legacy_fallback() -> None:
    snap = DashboardSnapshot(
        generated_at=datetime.now(timezone.utc),
        cycle_ms=1.0,
        regime="TRANSITION",
        report={},
        system_state={
            "regime": "TRANSITION",
            "corr_z": 0.5,
            "predicted_ann_vol_pct": 12.0,
            "target_ann_vol_pct": 10.0,
            "decision": {"priority": "normal", "exposure_scale": 1.0},
            "decision_trace": {"confidence": 0.6},
        },
        anomalies=[{}, {}],
    )
    ui = build_ui_model(snap)
    assert ui["has_elite"] is False
    assert ui["regime"] == "TRANSITION"
    assert ui["market_state"]["anomaly_count"] == 2
    assert ui["market_state"]["vol_ann_forecast"] == 0.12
