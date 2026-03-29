"""Risk-vs-target, tail block, allocation delta for elite snapshot (backend brief §7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pre.settings import AppSettings
from risk.var import VaRResult


def build_risk_vs_target(
    *,
    settings: AppSettings,
    forecast_ann_vol: float,
    lr1_portfolio: pd.Series | None,
) -> dict[str, Any]:
    tgt = float(settings.portfolio.target_ann_vol)
    realized_proxy = float(forecast_ann_vol)
    if lr1_portfolio is not None and len(lr1_portfolio) >= 20:
        tail = lr1_portfolio.dropna().iloc[-min(60, len(lr1_portfolio)) :]
        if len(tail) >= 10:
            realized_proxy = float(tail.std(ddof=1) * np.sqrt(252))
    dev = forecast_ann_vol - tgt
    dev_bps = dev * 10000.0
    pct = (dev / tgt * 100.0) if tgt > 1e-12 else 0.0
    if forecast_ann_vol > tgt * 1.1:
        hint = "Forecast vol above target — vol targeting will lean toward de-leverage."
    elif forecast_ann_vol < tgt * 0.85:
        hint = "Forecast vol below target — room to add risk within caps."
    else:
        hint = "Forecast vol near target."
    return {
        "forecast_ann_vol": round(forecast_ann_vol, 6),
        "realized_ann_vol_proxy": round(realized_proxy, 6),
        "target_ann_vol": round(tgt, 6),
        "deviation_bps": round(dev_bps, 2),
        "deviation_pct_of_target": round(pct, 2),
        "narrative_hint": hint,
    }


def build_tail_risk_block(
    *,
    var_res: VaRResult,
    var_meta: dict[str, Any],
    recent_breaches: list[bool] | None = None,
) -> dict[str, Any]:
    cluster_note = ""
    if recent_breaches and len(recent_breaches) >= 5:
        recent = recent_breaches[-10:]
        if sum(1 for x in recent if x) >= 3:
            cluster_note = "Multiple VaR breaches clustered in the last 10 bars."
    return {
        "hs_var_99_1d": float(var_res.hs_var.get((0.99, 1), 0.0)),
        "mc_var_99_1d": float(var_res.mc_var.get((0.99, 1), 0.0)),
        "hs_cvar_99_1d": float(var_res.hs_cvar.get((0.99, 1), 0.0)),
        "mc_cvar_99_1d": float(var_res.mc_cvar.get((0.99, 1), 0.0)),
        "cf_cvar_99": float(var_res.cf_cvar_99),
        "tail_multiplier": float(var_res.tail_multiplier),
        "var_trend_label": str(var_meta.get("var_trend_label", "flat")),
        "breaches_30d": int(var_meta.get("breaches_30d", 0)),
        "breach_today": bool(var_meta.get("breach_today", False)),
        "breach_cluster_note": cluster_note,
        "tail_direction": str(var_meta.get("var_trend_label", "flat")),
    }


def build_allocation_delta(
    w_before: pd.Series,
    w_after: pd.Series,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    b = w_before.reindex(w_after.index).fillna(0.0)
    a = w_after.fillna(0.0)
    delta = (a - b).astype(float)
    gross_b = float(b.abs().sum()) if len(b) else 0.0
    gross_a = float(a.abs().sum()) if len(a) else 0.0
    inc = delta[delta > 1e-8].sort_values(ascending=False).head(top_n)
    dec = (-delta[delta < -1e-8]).sort_values(ascending=False).head(top_n)
    return {
        "gross_exposure_before": round(gross_b, 6),
        "gross_exposure_after": round(gross_a, 6),
        "top_increases": [{"asset": str(i), "delta": float(v)} for i, v in inc.items()],
        "top_decreases": [{"asset": str(i), "delta": float(-delta.loc[i])} for i in dec.index],
    }
