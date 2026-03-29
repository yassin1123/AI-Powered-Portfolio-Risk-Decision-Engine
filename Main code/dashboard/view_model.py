"""Merge elite_snapshot (live_snapshot_v1) with legacy snapshot fields for Dash."""

from __future__ import annotations

from typing import Any

from core.snapshot import DashboardSnapshot


def _empty_narrative() -> dict[str, Any]:
    return {
        "headline": "",
        "summary": "",
        "why_lines": [],
        "action_line": "",
    }


def _legacy_narrative_from_ss(ss: dict[str, Any]) -> dict[str, Any]:
    """Minimal narrative shape when elite is absent."""
    dec = ss.get("decision") or {}
    pr = str(dec.get("priority") or ss.get("decision_priority") or "normal")
    reg = str(ss.get("regime") or "")
    return {
        "headline": f"Regime {reg or '—'} · priority {pr}",
        "summary": "",
        "why_lines": [],
        "action_line": "",
    }


def _legacy_market_state(ss: dict[str, Any], header: dict[str, Any], *, anomaly_count: int = 0) -> dict[str, Any]:
    dec = ss.get("decision") or {}
    pred_pct = float(ss.get("predicted_ann_vol_pct") or 0.0)
    tgt_pct = float(ss.get("target_ann_vol_pct") or 10.0)
    return {
        "timestamp": str(header.get("timestamp") or ""),
        "regime": str(ss.get("regime") or header.get("regime") or ""),
        "regime_confidence": float((ss.get("decision_trace") or {}).get("confidence") or dec.get("confidence") or 0.55),
        "corr_level": float(ss.get("avg_pairwise_corr") or ss.get("corr_level") or 0.0),
        "corr_z": float(ss.get("corr_z") or 0.0),
        "corr_bucket": str(dec.get("corr_bucket") or ss.get("corr_bucket") or "normal"),
        "anomaly_count": int(anomaly_count),
        "portfolio_drawdown": float(ss.get("portfolio_drawdown") or 0.0),
        "vol_ann_forecast": pred_pct / 100.0 if pred_pct > 0 else 0.15,
        "vol_ann_target": tgt_pct / 100.0 if tgt_pct > 0 else 0.10,
        "tail_multiplier": float(ss.get("tail_multiplier") or 1.0),
        "risk_disagreement_hs_mc": bool(ss.get("risk_disagreement") or False),
        "driver_scores": {},
        "trigger_flags": {},
        "stability_score": float(ss.get("stability_score") or 0.5),
        "regime_feature_snapshot": {},
    }


def _legacy_decision(ss: dict[str, Any]) -> dict[str, Any]:
    dec = ss.get("decision") or {}
    tr = ss.get("decision_trace") or {}
    return {
        "timestamp": ss.get("timestamp"),
        "decision_label": str(dec.get("priority") or ss.get("decision_priority") or "normal"),
        "winning_rule": str(tr.get("winning_rule_id") or dec.get("winning_rule_id") or ""),
        "risk_multiplier": float(dec.get("exposure_scale") or 1.0),
        "previous_risk_multiplier": None,
        "exposure_scale": float(dec.get("exposure_scale") or 1.0),
        "activate_hedge": bool(dec.get("activate_hedge") or False),
        "suppress_non_defensive": bool(dec.get("suppress_non_defensive") or False),
        "conditions_met": dict(tr.get("condition_flags") or {}),
        "pre_filter_signals_top": {},
        "post_gate_signals_top": {},
        "post_decision_signals_top": {},
        "signal_adjustment_ratio_sample": {},
    }


def _legacy_risk(vp: dict[str, Any], vb: dict[str, Any], ss: dict[str, Any]) -> dict[str, Any]:
    tail = {
        "hs_var_99_1d": float(vp.get("hs_var_99_1d") or 0.0),
        "mc_var_99_1d": float(vp.get("mc_var_99_1d") or 0.0),
        "cf_cvar_99": float(vp.get("cf_cvar_99") or 0.0),
        "tail_multiplier": float(vp.get("tail_multiplier") or ss.get("tail_multiplier") or 1.0),
        "var_trend_label": str(vp.get("var_trend_label") or "flat"),
        "breaches_30d": int(vp.get("breaches_30d") or 0),
        "breach_today": bool(vb.get("var_breach") or False),
        "breach_cluster_note": "",
    }
    pred = float(ss.get("predicted_ann_vol_pct") or 0.0)
    tgt = float(ss.get("target_ann_vol_pct") or 10.0)
    f_ann = pred / 100.0 if pred > 1.5 else float(ss.get("predicted_ann_vol") or 0.15)
    t_ann = tgt / 100.0 if tgt > 1.5 else 0.10
    vs_target = {
        "forecast_ann_vol": f_ann,
        "realized_ann_vol_proxy": f_ann,
        "target_ann_vol": t_ann,
        "deviation_bps": (f_ann - t_ann) * 10000.0,
        "deviation_pct_of_target": ((f_ann - t_ann) / t_ann * 100.0) if t_ann > 1e-9 else 0.0,
        "narrative_hint": "",
    }
    return {"vs_target": vs_target, "tail": tail, "decision_trace": ss.get("decision_trace")}


def build_ui_model(snap: DashboardSnapshot) -> dict[str, Any]:
    report = snap.report or {}
    elite = report.get("elite_snapshot")
    ss = snap.system_state or {}
    vp = snap.var_panel or {}
    vb = report.get("var_block") or {}
    h = snap.header or {}

    if isinstance(elite, dict) and elite.get("schema_version") == "live_snapshot_v1":
        return {
            "has_elite": True,
            "elite": elite,
            "meta": elite.get("meta") or {},
            "regime": str((elite.get("market_state") or {}).get("regime") or ss.get("regime") or ""),
            "confidence": float(
                (elite.get("market_state") or {}).get("regime_confidence")
                or (ss.get("decision_trace") or {}).get("confidence")
                or 0.55
            ),
            "narrative": elite.get("narrative") or _empty_narrative(),
            "market_state": elite.get("market_state") or {},
            "decision": elite.get("decision") or {},
            "risk": elite.get("risk") or {},
            "portfolio": elite.get("portfolio") or {},
            "recent_changes": elite.get("recent_changes") or {},
            "timeline": elite.get("timeline") or {},
            "analogs": elite.get("analogs") or {},
            "research_links": elite.get("research_links") or {},
            "system_state": ss,
            "var_panel": vp,
            "var_block": vb,
            "header": h,
        }

    ms = _legacy_market_state(ss, h, anomaly_count=len(snap.anomalies or []))
    narrative = _legacy_narrative_from_ss(ss)
    risk = _legacy_risk(vp, vb, ss)
    return {
        "has_elite": False,
        "elite": None,
        "meta": {},
        "regime": str(ms.get("regime") or ""),
        "confidence": float(ms.get("regime_confidence") or 0.55),
        "narrative": narrative,
        "market_state": ms,
        "decision": _legacy_decision(ss),
        "risk": risk,
        "portfolio": {},
        "recent_changes": {},
        "timeline": {},
        "analogs": {},
        "research_links": {},
        "system_state": ss,
        "var_panel": vp,
        "var_block": vb,
        "header": h,
    }
