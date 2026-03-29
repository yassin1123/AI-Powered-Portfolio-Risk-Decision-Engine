"""L1–L8 risk cycle: features → DCC → VaR → anomalies → regime → alpha → decision → portfolio → snapshot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from alpha.correlation_regime_signal import correlation_regime_signal
from alpha.gating import gate_signals
from alpha.signal_combiner import combine_signals
from core.decision.decision_engine import DecisionEngine, apply_decision_to_signals
from core.snapshot import DashboardSnapshot
from data.fetcher import DataFetcher
from data.universe import UNIVERSE, AssetClass
from detection.anomaly import AnomalyPipeline
from diagnostics.contagion import contagion_index
from features.returns import compute_features
from hedging.hedge_overlay import recommend_hedge
from portfolio.constraints import apply_constraints
from portfolio.optimizer import optimize_weights
from portfolio.risk_targeting import vol_target_scale
from pre.settings import AppSettings
from regime.hmm_regime import build_feature_matrix_row
from regime.regime_state import classify_regime_full, regime_output_to_dict
from reports.generator import basel_traffic_light, build_json_report
from risk.garch import GARCHDCCResult, cholesky_cached, fit_garch_dcc
from risk.portfolio import PortfolioRisk
from risk.var import compute_full_var
from signals.rebalance import RebalancingEngine
from stress.scenarios import ScenarioLibrary, reverse_stress_test, run_scenario


@dataclass
class PipelineState:
    prev_sigma: np.ndarray | None = None
    prev_L: np.ndarray | None = None
    last_garch: GARCHDCCResult | None = None
    cycle: int = 0
    last_garch_refit_cycle: int = -10**9
    var_99_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    corr_history: list[float] = field(default_factory=list)
    prev_regime_label: str | None = None
    regime_duration: int = 0
    last_transition_iso: str = ""
    feature_rows: list[np.ndarray] = field(default_factory=list)
    overlay_drawdown: list[float] = field(default_factory=list)
    overlay_corr_z: list[float] = field(default_factory=list)
    overlay_regime: list[str] = field(default_factory=list)
    var_breach_flags: list[bool] = field(default_factory=list)


def _bond_weight_share(w: pd.Series, tickers: list[str]) -> float:
    s = 0.0
    for t in tickers:
        meta = UNIVERSE.get(str(t), {})
        if meta.get("asset_class") == AssetClass.FIXED_INCOME:
            s += float(w.get(t, 0.0))
    return s


def _var_institutional_block(
    state: PipelineState,
    var_99: float,
    last_ret_log: float,
) -> dict[str, Any]:
    from scipy.stats import percentileofscore

    hist_before = [float(x) for x in state.var_99_history if np.isfinite(x)]
    pct: float | None = None
    if len(hist_before) >= 8:
        pct = float(percentileofscore(hist_before, float(var_99), kind="rank"))
    breach_now = bool(float(last_ret_log) < -float(var_99))
    state.var_breach_flags.append(breach_now)
    state.var_99_history.append(float(var_99))
    state.loss_history.append(float(last_ret_log))
    if len(state.var_99_history) > 300:
        state.var_99_history = state.var_99_history[-252:]
        state.loss_history = state.loss_history[-252:]
        state.var_breach_flags = state.var_breach_flags[-252:]

    breach_30 = sum(1 for x in state.var_breach_flags[-30:] if x)
    trend = "flat"
    h = [float(x) for x in state.var_99_history if np.isfinite(x)]
    if len(h) >= 14:
        d = np.diff(np.array(h[-15:], dtype=float))
        m = float(np.mean(d))
        if m > 1e-7:
            trend = "increasing"
        elif m < -1e-7:
            trend = "decreasing"

    return {
        "var_99_percentile_vs_history": pct,
        "breaches_30d": int(breach_30),
        "var_trend_label": trend,
        "var_99_series": h[-90:],
        "breach_today": breach_now,
    }


def _equal_weights(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    v = 1.0 / n
    return pd.Series({t: v for t in tickers})


def _align_returns(lr1: pd.DataFrame, lr10: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common = [c for c in lr1.columns if c in lr10.columns]
    if not common:
        return np.zeros((0, 0)), np.zeros((0, 0)), []
    win = lr1.index[-min(252, len(lr1)) :]
    cov1 = lr1.reindex(win)[common].notna().mean(axis=0)
    cov10 = lr10.reindex(win)[common].notna().mean(axis=0)
    min_cov = 0.97
    tight = [c for c in common if float(cov1[c]) >= min_cov and float(cov10[c]) >= min_cov]
    use = tight if len(tight) >= 5 else common
    a1 = lr1[use].dropna(how="any")
    a10 = lr10[use].reindex(a1.index).dropna(how="any")
    a1 = a1.reindex(a10.index).dropna(how="any")
    a10 = a10.reindex(a1.index).dropna(how="any")
    n = len(use)
    if len(a1) < 30:
        z = np.zeros((0, n), dtype=float)
        return z, z.copy(), use
    return a1.values.astype(float), a10.values.astype(float), use


def _coherent_sigma_portfolio(
    sigma_t: np.ndarray,
    R_t: np.ndarray,
    d_diag: np.ndarray,
    lr1_sub: pd.DataFrame,
    common: list[str],
    w_sub: pd.Series,
) -> tuple[np.ndarray, PortfolioRisk, np.ndarray, np.ndarray]:
    """If GARCH Σ implies portfolio vol far above sample Σ, use sample cov for VaR/optimizer this bar."""
    n = len(common)
    pr_g = PortfolioRisk.from_weights_sigma(w_sub, sigma_t, common)
    tail = lr1_sub[common].dropna(how="any")
    if len(tail) > 252:
        tail = tail.iloc[-252:]
    if len(tail) < 40:
        return sigma_t, pr_g, R_t, d_diag
    sc = np.cov(tail.values.astype(float), rowvar=False)
    sc = np.nan_to_num(sc, nan=1e-8, posinf=0.05, neginf=-0.05)
    sc = (sc + sc.T) / 2.0 + np.eye(n) * 1e-10
    pr_s = PortfolioRisk.from_weights_sigma(w_sub, sc, common)
    thr = max(0.035, 2.2 * max(pr_s.portfolio_vol, 1e-8))
    if pr_g.portfolio_vol <= thr:
        return sigma_t, pr_g, R_t, d_diag
    d_new = np.sqrt(np.clip(np.diag(sc), 1e-16, None))
    outer = np.outer(d_new, d_new)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_new = np.where(outer > 1e-18, sc / outer, np.eye(n))
    np.fill_diagonal(R_new, 1.0)
    R_new = (R_new + R_new.T) / 2.0
    return sc, pr_s, R_new, d_new


def run_risk_cycle(
    settings: AppSettings,
    fetcher: DataFetcher,
    state: PipelineState,
    target_weights: pd.Series | None = None,
) -> DashboardSnapshot:
    t0 = time.perf_counter()
    tickers = fetcher.tickers
    closes = fetcher.latest_closes()
    if closes.empty or len(closes) < 30:
        return DashboardSnapshot.empty("Insufficient price history")

    tw = target_weights if target_weights is not None else _equal_weights(list(closes.columns))
    w_series = tw.reindex(closes.columns).fillna(0.0)
    if w_series.sum() != 0:
        w_series = w_series / w_series.sum()

    feat = compute_features(closes, settings, w_series)

    lr1_df = feat.log_returns_1d
    lr10_df = feat.log_returns_10d
    lr1_m, lr10_m, common = _align_returns(lr1_df, lr10_df)
    if len(common) < 5:
        return DashboardSnapshot.empty("Too few assets after alignment")
    if lr1_m.shape[0] < 30:
        return DashboardSnapshot.empty(
            "Warm-up: need ≥30 rows where all assets have both 1d and 10d log returns. "
            "With a wide universe, any missing bar drops the row—simulation will fill in after more steps."
        )

    lr1_sub = lr1_df[common]
    if state.cycle - state.last_garch_refit_cycle >= max(1, settings.garch_refit_days):
        state.last_garch = fit_garch_dcc(lr1_sub, settings)
        state.last_garch_refit_cycle = state.cycle
    gd = state.last_garch
    if gd is None:
        state.last_garch = fit_garch_dcc(lr1_sub, settings)
        gd = state.last_garch
    assert gd is not None

    n = len(common)
    if gd.sigma_t.shape != (n, n):
        sigma_t = np.eye(n) * 1e-4
        R_t = np.eye(n)
        d_diag = np.ones(n) * 0.01
    else:
        sigma_t = gd.sigma_t
        R_t = gd.R_t
        d_diag = gd.D_diag

    w_sub = w_series.reindex(common).fillna(0.0)
    if w_sub.sum() != 0:
        w_sub = w_sub / w_sub.sum()
    sigma_t, pr, R_t, d_diag = _coherent_sigma_portfolio(
        sigma_t, R_t, d_diag, lr1_sub, common, w_sub
    )

    L, _ = cholesky_cached(
        sigma_t,
        state.prev_sigma,
        state.prev_L,
        settings.cholesky_frobenius_threshold,
    )
    state.prev_sigma = sigma_t.copy()
    state.prev_L = L.copy()

    w = w_series.reindex(common).fillna(0.0).values.astype(float)
    if w.sum() != 0:
        w = w / w.sum()
    mu_1d = lr1_m.mean(axis=0)
    mu_10d = lr10_m.mean(axis=0)

    rng = np.random.default_rng()
    var_res = compute_full_var(
        lr1_m,
        lr10_m,
        w,
        sigma_t,
        mu_1d,
        mu_10d,
        settings.var_confidence_levels,
        settings.mc_sims,
        rng,
    )
    avg_corr = float((np.sum(R_t) - np.trace(R_t)) / (n * (n - 1) + 1e-9))
    med_vol = float(np.median(d_diag)) if len(d_diag) else 0.01

    state.corr_history.append(avg_corr)
    if len(state.corr_history) > 500:
        state.corr_history = state.corr_history[-500:]

    anomalies = AnomalyPipeline(settings).run(feat, w_series)
    anom_count = len(anomalies)

    pdd = float(feat.drawdown_portfolio.iloc[-1]) if len(feat.drawdown_portfolio) else 0.0
    row_f = build_feature_matrix_row(med_vol, avg_corr, var_res.tail_multiplier, pdd)
    state.feature_rows.append(row_f)
    if len(state.feature_rows) > 300:
        state.feature_rows = state.feature_rows[-300:]
    fh_arr = (
        np.stack(state.feature_rows[-200:], axis=0)
        if len(state.feature_rows) >= 10
        else None
    )

    ro = classify_regime_full(
        settings,
        var_res.tail_multiplier,
        avg_corr,
        med_vol,
        pdd,
        anom_count,
        prev_label=state.prev_regime_label,
        prev_duration=state.regime_duration,
        last_transition_iso=state.last_transition_iso,
        feature_history=fh_arr,
    )
    state.prev_regime_label = ro.label
    state.regime_duration = ro.duration_bars
    state.last_transition_iso = ro.last_transition_iso

    regime = ro.label
    corr_result = correlation_regime_signal(state.corr_history, avg_corr, settings)
    var_99 = var_res.mc_var.get((0.99, 1), 0.0)
    forecast_ann_vol = float(pr.portfolio_vol * np.sqrt(252))
    decision_engine = DecisionEngine(settings)
    decision = decision_engine.decide(
        ro.label,
        corr_result,
        anom_count,
        var_99,
        settings.risk_limit_var_99,
        regime_confidence=float(ro.confidence),
        drawdown=float(pdd),
        vol_ann=forecast_ann_vol,
        avg_corr=float(avg_corr),
    )

    closes_c = closes.reindex(columns=common).dropna(how="all")
    comb = combine_signals(lr1_sub, closes_c, corr_result, settings)
    raw = comb.per_asset.reindex(common).fillna(0.0)
    raw = gate_signals(raw, ro.label, anom_count)
    raw = apply_decision_to_signals(raw, decision, ro.label)
    w_target = optimize_weights(ro.label, raw, sigma_t, common, settings)
    w_target = vol_target_scale(
        w_target,
        forecast_ann_vol,
        settings.portfolio.target_ann_vol,
        settings.portfolio.max_gross_leverage,
    )
    w_target = apply_constraints(w_target, w_sub, settings)

    decision_trace_out: dict[str, Any] | None = None
    if decision.trace is not None:
        tr = dict(decision.trace)
        ws = w_series.reindex(common).fillna(0.0)
        wt = w_target.reindex(common).fillna(0.0)
        cb = _bond_weight_share(ws, common)
        tb = _bond_weight_share(wt, common)
        dpp = (tb - cb) * 100.0
        pos = dict(tr.get("positioning") or {})
        pos_lines = [
            f"Target gross exposure ≈ {float(wt.sum()) * 100:.1f}%",
            f"Decision risk multiplier: {float(decision.exposure_scale):.2f}× (signal gating before optimizer)",
            f"Single-asset cap: {settings.portfolio.max_single_weight * 100:.0f}%",
        ]
        if abs(dpp) >= 0.2:
            pos_lines.append(f"Defensive (fixed income) vs current book: {dpp:+.1f} pp")
        pos.update(
            {
                "target_gross_pct": float(wt.sum() * 100),
                "defensive_shift_pp": float(dpp),
                "lines": pos_lines,
            }
        )
        tr["positioning"] = pos
        decision_trace_out = tr

    plr = (lr1_sub * w_series.reindex(common).fillna(0.0)).sum(axis=1)
    last_ret = float(plr.iloc[-1]) if len(plr) else 0.0
    last_simple = float(np.expm1(last_ret)) if np.isfinite(last_ret) else 0.0
    daily_pnl = last_simple * settings.portfolio_total_value
    var_meta = _var_institutional_block(state, var_99, last_ret)
    zone = basel_traffic_light(
        np.array(state.loss_history),
        np.array(state.var_99_history),
    )

    engine = RebalancingEngine(settings)
    signals = engine.run(
        weights=w_series,
        target_weights=w_target.reindex(w_series.index).fillna(0.0),
        var_99_portfolio=var_99,
        risk_contributions=pr.risk_contributions,
        tail_multiplier=var_res.tail_multiplier,
        avg_pairwise_corr=avg_corr,
        portfolio_drawdown=pdd,
    )

    hedge_rec = recommend_hedge(
        decision,
        feat.rolling_beta.reindex(common).fillna(0.0),
        w_sub,
        var_res.tail_multiplier,
        settings,
    )

    lib = ScenarioLibrary()
    stress_out: dict[str, Any] = {}
    garch_mult = float(np.median(d_diag)) / 0.01 if np.median(d_diag) > 0 else 1.0
    for name in lib.names():
        sr = run_scenario(
            name, w_series, common, sigma_t, garch_mult, 0.02, lib
        )
        stress_out[name] = {
            "portfolio_pnl": sr.portfolio_pnl,
            "direct": sr.direct,
            "correlation_effect": sr.correlation_effect,
            "vol_effect": sr.vol_effect,
            "by_asset": sr.by_asset,
        }
    stress_out["reverse_target_15pct"] = reverse_stress_test(
        w_series, common, sigma_t, -0.15
    )

    garch_paths: dict[str, list[float]] = {}
    vps = getattr(gd, "vol_paths", None) if gd is not None else None
    if vps:
        tail_n = 60
        for t in common[:15]:
            ts = str(t)
            s = vps.get(ts)
            if s and len(s) >= 2:
                tail = s[-min(tail_n, len(s)) :]
                garch_paths[t] = [float(x) for x in tail]
            else:
                garch_paths[t] = risk_garch_path_safe(lr1_sub[t], settings.garch_window)
    else:
        garch_paths = {
            t: risk_garch_path_safe(lr1_sub[t], settings.garch_window)
            for t in common[:15]
        }

    cont_ix = contagion_index(R_t)
    top_rc = (
        pr.risk_contributions.abs().sort_values(ascending=False).head(3)
        if len(pr.risk_contributions)
        else pd.Series(dtype=float)
    )
    top_sig = raw.abs().sort_values(ascending=False).head(3)

    _dd_live = float(feat.drawdown_portfolio.iloc[-1]) if len(feat.drawdown_portfolio) else 0.0
    state.overlay_drawdown.append(_dd_live)
    state.overlay_corr_z.append(float(corr_result.corr_z))
    state.overlay_regime.append(str(regime))
    _cap_ov = 200
    if len(state.overlay_drawdown) > _cap_ov:
        state.overlay_drawdown = state.overlay_drawdown[-_cap_ov:]
        state.overlay_corr_z = state.overlay_corr_z[-_cap_ov:]
        state.overlay_regime = state.overlay_regime[-_cap_ov:]

    system_state = {
        "regime": regime,
        "regime_confidence": ro.confidence,
        "regime_detail": regime_output_to_dict(ro),
        "predicted_ann_vol_pct": round(forecast_ann_vol * 100, 2),
        "target_ann_vol_pct": round(settings.portfolio.target_ann_vol * 100, 2),
        "gross_exposure": float(w_target.sum()),
        "corr_z": round(corr_result.corr_z, 4),
        "corr_bucket": corr_result.bucket,
        "contagion_index": round(cont_ix, 4),
        "risk_limit_var_99": float(settings.risk_limit_var_99),
        "top_risk_assets": {str(k): float(v) for k, v in top_rc.items()},
        "top_signal_assets": {str(k): float(v) for k, v in top_sig.items()},
        "decision": {
            "priority": decision.decision_priority,
            "narrative": decision.narrative,
            "codes": decision.reason_codes,
            "secondary": decision.secondary_reasons,
            "hedge": hedge_rec.narrative,
            "suppress": decision.suppress_non_defensive,
            "override_signals": decision.override_signals,
            "exposure_scale": float(decision.exposure_scale),
            "activate_hedge": bool(decision.activate_hedge),
            "system_signal": (decision_trace_out or {}).get("system_signal"),
            "confidence": (decision_trace_out or {}).get("confidence"),
            "winning_rule_id": (decision_trace_out or {}).get("winning_rule_id"),
        },
        "decision_trace": decision_trace_out,
        "decision_priority": decision.decision_priority,
        "last_rebalance_reason": decision.decision_priority + (
            f" ({decision.narrative})" if decision.narrative else ""
        ),
    }

    header = {
        "total_value": settings.portfolio_total_value,
        "daily_pnl": daily_pnl,
        # Weighted sum of asset log returns (approx portfolio log return for small moves).
        "daily_return": last_ret,
        "daily_return_simple_approx": last_simple,
        "var_95_1d": var_res.mc_var.get((0.95, 1), 0.0),
        "var_99_1d": var_99,
        "regime": regime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    var_panel = {
        "hs_var_95_1d": var_res.hs_var.get((0.95, 1), 0.0),
        "hs_var_99_1d": var_res.hs_var.get((0.99, 1), 0.0),
        "mc_var_95_1d": var_res.mc_var.get((0.95, 1), 0.0),
        "mc_var_99_1d": var_res.mc_var.get((0.99, 1), 0.0),
        "hs_var_95_10d": var_res.hs_var.get((0.95, 10), 0.0),
        "hs_var_99_10d": var_res.hs_var.get((0.99, 10), 0.0),
        "mc_var_95_10d": var_res.mc_var.get((0.95, 10), 0.0),
        "mc_var_99_10d": var_res.mc_var.get((0.99, 10), 0.0),
        "cf_cvar_99": var_res.cf_cvar_99,
        "tail_multiplier": var_res.tail_multiplier,
        "backtesting_zone": zone,
        **var_meta,
    }
    corr_regime = "NORMAL" if avg_corr < 0.5 else "ELEVATED"
    diversification_score = float(max(0.0, min(1.0, 1.0 - avg_corr)))
    correlation = {
        "matrix": R_t.tolist(),
        "tickers": common,
        "avg_pairwise": avg_corr,
        "vs_252d_avg": 0.0,
        "regime": corr_regime,
        "contagion_index": cont_ix,
        "corr_z": corr_result.corr_z,
        "diversification_score": diversification_score,
        "diversification_note": (
            "Diversification weak (avg correlation high) → clustering risk ↑"
            if avg_corr > settings.avg_corr_alert
            else (
                "Diversification moderate"
                if avg_corr > 0.35
                else "Diversification comparatively strong vs avg correlation"
            )
        ),
    }
    mc_dist = {
        "simulations": var_res.mc_port_1d_sims.tolist()[:2000],
        "full_count": len(var_res.mc_port_1d_sims),
    }
    sigma_p = float(pr.portfolio_vol)
    if sigma_p > 1e-18:
        rc_share = (pr.risk_contributions / sigma_p).astype(float)
    else:
        rc_share = pr.risk_contributions * 0.0
    risk_attr = {
        # Absolute (≈ daily σ units); sum ≈ portfolio_vol — not comparable to budget directly.
        "contributions": pr.risk_contributions.to_dict(),
        # Fraction of total portfolio σ; sum ≈ 1 — compare to risk_budget_default (e.g. 0.05 = 5%).
        "contribution_share": {k: float(v) for k, v in rc_share.items()},
        "portfolio_vol": sigma_p,
        "budget": settings.risk_budget_default,
    }

    state.cycle += 1
    cycle_ms = (time.perf_counter() - t0) * 1000

    report_payload = build_json_report(
        cycle_ms=cycle_ms,
        portfolio={
            "total_value": settings.portfolio_total_value,
            "daily_pnl": daily_pnl,
            "daily_return": last_ret,
            "weights": w_series.reindex(common).fillna(0.0).to_dict(),
            "target_weights": w_target.reindex(common).fillna(0.0).to_dict(),
        },
        var_block={
            "hs_var_95_1d": var_res.hs_var.get((0.95, 1), 0.0),
            "hs_var_99_1d": var_res.hs_var.get((0.99, 1), 0.0),
            "mc_var_95_1d": var_res.mc_var.get((0.95, 1), 0.0),
            "mc_var_99_1d": var_res.mc_var.get((0.99, 1), 0.0),
            "cf_cvar_99": var_res.cf_cvar_99,
            "tail_multiplier": var_res.tail_multiplier,
            "var_breach": last_ret < -var_99,
            "backtesting_zone": zone,
        },
        volatility={
            "garch_forecasts": {t: float(gd.garch_vol_forecast.get(t, 0.01)) for t in common},
            "regime": regime,
            "persistence": 0.94,
        },
        correlation={
            "avg_pairwise": avg_corr,
            "max_pairwise": float(np.max(R_t[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
            "vs_252d_avg": 0.0,
            "regime": corr_regime,
            "corr_z": corr_result.corr_z,
            "contagion_index": cont_ix,
        },
        risk_contributions={k: float(v) for k, v in pr.risk_contributions.items()},
        anomalies=anomalies,
        signals=signals,
        stress_tests=stress_out,
    )
    report_payload["system_state"] = system_state
    report_payload["decision_engine"] = {
        "decision_priority": decision.decision_priority,
        "narrative": decision.narrative,
        "codes": decision.reason_codes,
        "secondary": decision.secondary_reasons,
    }

    return DashboardSnapshot(
        generated_at=datetime.now(timezone.utc),
        cycle_ms=cycle_ms,
        regime=regime,
        header=header,
        var_panel=var_panel,
        correlation=correlation,
        mc_distribution=mc_dist,
        risk_attribution=risk_attr,
        anomalies=anomalies,
        garch_vol_paths=garch_paths,
        report=report_payload,
        weights=w_series.reindex(common).fillna(0.0).to_dict(),
        target_weights=w_target.reindex(common).fillna(0.0).to_dict(),
        tickers=common,
        system_state=system_state,
        overlay_series={
            "drawdown": list(state.overlay_drawdown),
            "corr_z": list(state.overlay_corr_z),
            "regime": list(state.overlay_regime),
        },
    )


def risk_garch_path_safe(series: pd.Series, window: int) -> list[float]:
    from risk.garch import garch_vol_history_path

    try:
        return garch_vol_history_path(series, window)
    except Exception:
        return [0.01] * 10
