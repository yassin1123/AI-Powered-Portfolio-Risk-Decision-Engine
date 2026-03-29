"""Contract tests for elite snapshot builders, narrative, recent changes, scenario, research helpers."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import pytest

from alpha.correlation_regime_signal import CorrRegimeSignalResult
from api.live_snapshot import build_live_snapshot_v1
from context.recent_changes import EliteTickSnapshot, compute_recent_changes, ring_append
from features.state_builder import build_market_state
from narrative.engine import build_narrative
from pre.settings import AppSettings
from regime.regime_state import RegimeOutput
from research.by_regime_metrics import performance_by_regime
from research.failure_analysis import analyze_failure_windows, failure_summary_from_backtest
from scenario.shocks import shock_market_state


def _ro(label: str = "CALM") -> RegimeOutput:
    return RegimeOutput(
        label=label,
        prob_vector={label: 1.0},
        confidence=0.75,
        duration_bars=3,
        last_transition_iso="",
        transition_matrix=None,
        persistence_note="",
        features={"med_vol": 0.12},
    )


def _cr(z: float) -> CorrRegimeSignalResult:
    return CorrRegimeSignalResult(0.4, 0.3, 0.1, z, "normal", False, False, False)


def test_build_market_state_core_keys() -> None:
    s = AppSettings()
    ms = build_market_state(
        settings=s,
        ro=_ro("STRESSED"),
        corr_result=_cr(1.51),
        avg_corr=0.55,
        anom_count=2,
        portfolio_drawdown=-0.08,
        forecast_ann_vol=0.22,
        tail_mult=1.1,
        risk_disagreement=True,
        trace_drivers=[{"name": "corr", "delta": 0.4}],
        stability_score=0.4,
    )
    for k in (
        "timestamp",
        "regime",
        "regime_confidence",
        "corr_z",
        "corr_bucket",
        "anomaly_count",
        "vol_ann_forecast",
        "vol_ann_target",
        "trigger_flags",
        "stability_score",
    ):
        assert k in ms
    assert ms["regime"] == "STRESSED"
    assert ms["trigger_flags"]["high_corr_z"] is True


def test_build_narrative_golden_headline() -> None:
    n = build_narrative(
        regime="CALM",
        corr_z=0.1,
        corr_bucket="normal",
        anomaly_count=0,
        var_99=0.01,
        var_limit=0.05,
        forecast_vol=0.12,
        target_vol=0.15,
        risk_multiplier=1.0,
        decision_priority="standard",
        activate_hedge=False,
        breach_today=False,
        var_trend="flat",
    )
    assert n["headline"]
    assert "why_lines" in n and len(n["why_lines"]) == 4
    assert "summary" in n


def test_recent_changes_deltas() -> None:
    ring: deque[EliteTickSnapshot] = deque(maxlen=30)
    for i in range(8):
        ring_append(
            ring,
            EliteTickSnapshot(
                corr_z=float(i) * 0.1,
                var_99=0.02 + i * 1e-4,
                regime="CALM",
                confidence=0.5,
                risk_multiplier=1.0,
            ),
        )
    rc = compute_recent_changes(ring)
    assert rc["corr_z_delta_1"] is not None
    assert rc["corr_z_delta_5"] is not None
    assert rc["regime_changed"] is False


def test_live_snapshot_v1_keys_and_version() -> None:
    snap = build_live_snapshot_v1(
        settings_profile="test",
        cycle=1,
        cycle_ms=12.0,
        data_quality_warnings=[],
        market_state={"regime": "CALM"},
        decision={"risk_multiplier": 1.0},
        narrative={"headline": "h"},
        risk={"vs_target": {}, "tail": {}},
        portfolio={},
        recent_changes={},
        timeline={},
        analogs={},
        research_links={},
    )
    assert snap["schema_version"] == "live_snapshot_v1"
    for k in (
        "meta",
        "market_state",
        "decision",
        "narrative",
        "risk",
        "portfolio",
        "recent_changes",
        "timeline",
        "analogs",
        "research_links",
    ):
        assert k in snap
    assert snap["meta"]["cycle"] == 1


def test_shock_market_state_deltas() -> None:
    base = {"vol_ann_forecast": 0.2, "corr_z": 0.5, "tail_multiplier": 1.0, "portfolio_drawdown": -0.02}
    shocked, deltas = shock_market_state(base, vol_ann_mult=1.5, corr_z_add=0.25, tail_mult_mult=2.0)
    assert shocked["vol_ann_forecast"] == pytest.approx(0.3)
    assert shocked["corr_z"] == pytest.approx(0.75)
    assert shocked["tail_multiplier"] == pytest.approx(2.0)
    assert "vol_ann_forecast" in deltas


def test_analyze_failure_windows_tags() -> None:
    log = []
    for i in range(15):
        log.append(
            {
                "t": i,
                "timestamp": f"2020-01-{i+1:02d}",
                "regime": "STRESSED" if 5 <= i <= 9 else "CALM",
                "pnl_frac": -0.02 if 6 <= i <= 8 else 0.001,
                "decision_priority": "standard",
                "var_99": 0.01,
            }
        )
    out = analyze_failure_windows(log, window=5, top_k=2)
    assert out["worst_windows"]
    assert "tag_counts" in out


def test_performance_by_regime_warns_small_sample() -> None:
    log = [{"regime": "CALM", "pnl_frac": 0.001} for _ in range(5)]
    out = performance_by_regime(log, min_bars_warn=30)
    assert "warnings" in out
    assert any("sample_small" in w for w in out["warnings"])


def test_failure_summary_equity_metrics() -> None:
    log = [{"t": i, "timestamp": str(i), "regime": "CALM", "pnl_frac": 0.0, "var_99": 0.01} for i in range(20)]
    eq = pd.Series(np.exp(np.cumsum(np.zeros(20))), index=range(20))
    summary = failure_summary_from_backtest(log, eq, risk_free_annual=0.0)
    assert "failure_windows" in summary
    assert "equity_metrics" in summary
