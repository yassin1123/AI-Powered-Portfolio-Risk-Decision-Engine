"""1d/5d/20d deltas for elite snapshot (backend brief §5.3)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class EliteTickSnapshot:
    corr_z: float
    var_99: float
    regime: str
    confidence: float
    risk_multiplier: float


def ring_append(ring: deque[EliteTickSnapshot], snap: EliteTickSnapshot) -> None:
    ring.append(snap)


def compute_recent_changes(ring: deque[EliteTickSnapshot]) -> dict[str, Any]:
    if len(ring) < 2:
        return {
            "corr_z_delta_1": None,
            "corr_z_delta_5": None,
            "corr_z_delta_20": None,
            "var_99_delta_1": None,
            "var_99_delta_5": None,
            "regime_changed": False,
            "confidence_delta": None,
            "risk_multiplier_delta": None,
            "notes": ["insufficient_history"],
        }
    cur = ring[-1]
    notes: list[str] = []

    def _lag(k: int) -> EliteTickSnapshot | None:
        if len(ring) <= k:
            return None
        return ring[-1 - k]

    d1 = _lag(1)
    d5 = _lag(5)
    d20 = _lag(20)

    cz1 = (cur.corr_z - d1.corr_z) if d1 else None
    cz5 = (cur.corr_z - d5.corr_z) if d5 else None
    cz20 = (cur.corr_z - d20.corr_z) if d20 else None
    v1 = (cur.var_99 - d1.var_99) if d1 else None
    v5 = (cur.var_99 - d5.var_99) if d5 else None
    reg_changed = bool(d1 and d1.regime != cur.regime)
    conf_d = (cur.confidence - d1.confidence) if d1 else None
    rm_d = (cur.risk_multiplier - d1.risk_multiplier) if d1 else None

    if reg_changed:
        notes.append("regime_label_changed_vs_prior_bar")
    if d5 and abs(cur.corr_z - d5.corr_z) > 0.5:
        notes.append("corr_z_moved_materially_5b")

    return {
        "corr_z_delta_1": round(cz1, 6) if cz1 is not None else None,
        "corr_z_delta_5": round(cz5, 6) if cz5 is not None else None,
        "corr_z_delta_20": round(cz20, 6) if cz20 is not None else None,
        "var_99_delta_1": round(v1, 8) if v1 is not None else None,
        "var_99_delta_5": round(v5, 8) if v5 is not None else None,
        "regime_changed": reg_changed,
        "confidence_delta": round(conf_d, 4) if conf_d is not None else None,
        "risk_multiplier_delta": round(rm_d, 4) if rm_d is not None else None,
        "notes": notes,
    }
