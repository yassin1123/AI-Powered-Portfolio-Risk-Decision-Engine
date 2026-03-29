"""Typed backend snapshot shapes (quant backend brief §11). JSON-serializable via to_jsonable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, lists, dicts; skip non-JSON types as str."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return str(obj)


@dataclass
class SnapshotMeta:
    schema_version: str = "live_snapshot_v1"
    generated_at_utc: str = ""
    as_of_utc: str = ""
    cycle: int = 0
    cycle_ms: float = 0.0
    universe_profile: str = ""
    data_quality_warnings: list[str] = field(default_factory=list)


@dataclass
class MarketState:
    timestamp: str
    regime: str
    regime_confidence: float
    corr_level: float
    corr_z: float
    corr_bucket: str
    anomaly_count: int
    portfolio_drawdown: float
    vol_ann_forecast: float
    vol_ann_target: float
    tail_multiplier: float
    risk_disagreement_hs_mc: bool
    driver_scores: dict[str, float] = field(default_factory=dict)
    trigger_flags: dict[str, bool] = field(default_factory=dict)
    stability_score: float = 0.0
    regime_feature_snapshot: dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionSnapshot:
    timestamp: str
    decision_label: str
    winning_rule: str
    risk_multiplier: float
    previous_risk_multiplier: float | None
    exposure_scale: float
    activate_hedge: bool
    suppress_non_defensive: bool
    conditions_met: dict[str, bool] = field(default_factory=dict)
    pre_filter_signals: dict[str, float] = field(default_factory=dict)
    post_filter_signals: dict[str, float] = field(default_factory=dict)
    signal_adjustment_ratio: dict[str, float] = field(default_factory=dict)
    rationale_codes: list[str] = field(default_factory=list)


@dataclass
class NarrativeBlock:
    headline: str
    summary: str
    why_lines: list[str] = field(default_factory=list)
    action_line: str = ""


@dataclass
class RiskVsTarget:
    forecast_ann_vol: float
    realized_ann_vol_proxy: float
    target_ann_vol: float
    deviation_bps: float
    deviation_pct_of_target: float
    narrative_hint: str = ""


@dataclass
class TailRiskBlock:
    hs_var_99_1d: float
    mc_var_99_1d: float
    cf_cvar_99: float
    tail_multiplier: float
    var_trend_label: str
    breaches_30d: int
    breach_today: bool
    breach_cluster_note: str = ""


@dataclass
class AllocationDelta:
    gross_exposure_before: float
    gross_exposure_after: float
    top_increases: list[dict[str, float]] = field(default_factory=list)
    top_decreases: list[dict[str, float]] = field(default_factory=list)


@dataclass
class RecentChanges:
    corr_z_delta_1: float | None = None
    corr_z_delta_5: float | None = None
    corr_z_delta_20: float | None = None
    var_99_delta_1: float | None = None
    var_99_delta_5: float | None = None
    regime_changed: bool = False
    confidence_delta: float | None = None
    risk_multiplier_delta: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class LiveSnapshotV1:
    meta: SnapshotMeta
    market_state: dict[str, Any]
    decision: dict[str, Any]
    narrative: dict[str, Any]
    risk: dict[str, Any]
    portfolio: dict[str, Any]
    recent_changes: dict[str, Any]
    timeline: dict[str, Any]
    analogs: dict[str, Any]
    research_links: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.meta.schema_version,
            "meta": to_jsonable(self.meta),
            "market_state": to_jsonable(self.market_state),
            "decision": to_jsonable(self.decision),
            "narrative": to_jsonable(self.narrative),
            "risk": to_jsonable(self.risk),
            "portfolio": to_jsonable(self.portfolio),
            "recent_changes": to_jsonable(self.recent_changes),
            "timeline": to_jsonable(self.timeline),
            "analogs": to_jsonable(self.analogs),
            "research_links": to_jsonable(self.research_links),
        }
