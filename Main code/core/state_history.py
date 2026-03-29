"""Regime transition history and timeline segments (backend brief §4.3, §8.3)."""

from __future__ import annotations

from typing import Any


def normalize_regime_for_transition(label: str) -> str:
    u = (label or "").upper()
    if "STRESS" in u:
        return "STRESSED"
    if "TRANS" in u:
        return "TRANSITION"
    return "CALM"


def transition_event_type(prev: str | None, cur: str) -> str:
    if prev is None:
        return "init"
    p, c = normalize_regime_for_transition(prev), normalize_regime_for_transition(cur)
    if p == c:
        return "no_change"
    if c == "STRESSED" and p != "STRESSED":
        return "enter_stress"
    if p == "STRESSED" and c != "STRESSED":
        return "exit_stress"
    if c == "TRANSITION":
        return "enter_transition"
    if p == "TRANSITION" and c == "CALM":
        return "exit_transition"
    if p == "CALM" and c == "STRESSED":
        return "enter_stress"
    return "regime_shift"


def append_history_row(
    rows: list[dict[str, Any]],
    *,
    timestamp_iso: str,
    regime: str,
    prev_regime: str | None,
    confidence: float,
    corr_z: float,
    max_rows: int = 2000,
) -> None:
    evt = transition_event_type(prev_regime, regime)
    rows.append(
        {
            "timestamp": timestamp_iso,
            "regime": regime,
            "previous_regime": prev_regime,
            "confidence": float(confidence),
            "corr_z": float(corr_z),
            "transition_event": evt,
        }
    )
    if len(rows) > max_rows:
        del rows[: len(rows) - max_rows]


def rolling_transition_stats(rows: list[dict[str, Any]], last_n: int = 20) -> dict[str, Any]:
    if not rows:
        return {"transitions_last_20": 0, "avg_confidence": None, "persistence_hint": ""}
    tail = rows[-last_n:]
    trans = sum(1 for r in tail if r.get("transition_event") not in (None, "no_change", "init"))
    confs = [float(r["confidence"]) for r in tail if "confidence" in r]
    avg_c = sum(confs) / len(confs) if confs else None
    return {
        "transitions_last_n": trans,
        "window_bars": len(tail),
        "avg_confidence": round(avg_c, 4) if avg_c is not None else None,
        "persistence_hint": "elevated_churn" if trans >= 4 else "stable",
    }


def build_timeline_segments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse consecutive same-regime rows into segments."""
    if not rows:
        return {"segments": []}
    segments: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i].get("regime") != rows[start].get("regime"):
            chunk = rows[start:i]
            cz = [float(r["corr_z"]) for r in chunk if "corr_z" in r]
            cf = [float(r["confidence"]) for r in chunk if "confidence" in r]
            segments.append(
                {
                    "start_timestamp": chunk[0].get("timestamp"),
                    "end_timestamp": chunk[-1].get("timestamp"),
                    "regime": chunk[0].get("regime"),
                    "bars": len(chunk),
                    "mean_corr_z": round(sum(cz) / len(cz), 4) if cz else None,
                    "mean_confidence": round(sum(cf) / len(cf), 4) if cf else None,
                }
            )
            start = i
    return {"segments": segments[-30:]}
