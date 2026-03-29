"""Post-hoc tagging of rough patches from decision_log + equity (backend brief §9.4).

Pure functions; safe to call from notebooks or to embed summaries in ``research_links``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest.evaluation import summarize_backtest


def _window_tags(rows: list[dict[str, Any]], i0: int, i1: int) -> list[str]:
    """Heuristic labels for log index range [i0, i1] inclusive."""
    slice_rows = rows[i0 : i1 + 1]
    if not slice_rows:
        return []
    tags: list[str] = []
    regimes = [str(r.get("regime", "")) for r in slice_rows]
    pnls = [float(r.get("pnl_frac") or 0.0) for r in slice_rows]
    prios = [str(r.get("decision_priority", "")) for r in slice_rows]
    var99 = [float(r.get("var_99") or 0.0) for r in slice_rows]

    cum = float(np.sum(pnls))
    mean_reg = max(set(regimes), key=regimes.count) if regimes else ""
    breach_ct = sum(1 for r, v in zip(pnls, var99) if v > 0 and r < -v)

    if mean_reg == "STRESSED" and cum > 0.002 * len(slice_rows):
        tags.append("false_stress_candidate")
    if cum < -0.03 * max(1, len(slice_rows) ** 0.5):
        if i0 > 0 and str(rows[i0 - 1].get("regime", "")) == "CALM":
            if not any(p.startswith("stress") or p == "corr_crisis" for p in prios):
                tags.append("late_derisk_candidate")
        tags.append("severe_drawdown_window")
    if breach_ct >= 3:
        tags.append("var_breach_cluster")
    if any("anomaly_suppress" == p for p in prios) and cum < -0.02:
        tags.append("anomaly_suppress_stress_persisted")
    return tags


def analyze_failure_windows(
    decision_log: list[dict[str, Any]],
    *,
    window: int = 5,
    top_k: int = 5,
) -> dict[str, Any]:
    """Find worst rolling cumulative log-return windows and attach heuristic tags."""
    n = len(decision_log)
    if n < window or window < 2:
        return {
            "worst_windows": [],
            "tag_counts": {},
            "window_bars": window,
            "note": "insufficient_history",
        }

    pnls = np.array([float(r.get("pnl_frac") or 0.0) for r in decision_log], dtype=float)
    rolling = np.convolve(pnls, np.ones(window), mode="valid")
    worst_idx = np.argsort(rolling)[:top_k]

    worst_windows: list[dict[str, Any]] = []
    tag_counts: dict[str, int] = {}
    for rank, start in enumerate(worst_idx):
        i0 = int(start)
        i1 = i0 + window - 1
        tags = _window_tags(decision_log, i0, i1)
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        worst_windows.append(
            {
                "rank": rank + 1,
                "start_t": decision_log[i0].get("t"),
                "end_t": decision_log[i1].get("t"),
                "start_timestamp": str(decision_log[i0].get("timestamp", "")),
                "end_timestamp": str(decision_log[i1].get("timestamp", "")),
                "rolling_cum_log_return": float(rolling[start]),
                "tags": tags,
            }
        )

    return {
        "worst_windows": worst_windows,
        "tag_counts": tag_counts,
        "window_bars": window,
        "note": "heuristic_tags_not_investment_advice",
    }


def failure_summary_from_backtest(
    decision_log: list[dict[str, Any]],
    equity: pd.Series,
    *,
    risk_free_annual: float = 0.0,
) -> dict[str, Any]:
    """JSON-serializable bundle for ``elite_snapshot.research_links`` or offline reports."""
    windows = analyze_failure_windows(decision_log)
    turn = pd.Series(0.0, index=equity.index) if len(equity) else pd.Series(dtype=float)
    overall = summarize_backtest(equity, turn, risk_free_annual) if len(equity) > 2 else {}
    return {
        "failure_windows": windows,
        "equity_metrics": {k: float(v) for k, v in overall.items()},
        "log_bars": len(decision_log),
    }
