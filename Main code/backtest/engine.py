"""Daily backtest: signals at t use data ≤ t; P&L from t→t+1; decision log.

VaR into `DecisionEngine.decide` uses `risk.var.compute_full_var` (same MC stack as live),
with sample Σ and `prior_w`; see `docs/backtest_assumptions.md` vs DCC in `pre/pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from alpha.correlation_regime_signal import correlation_regime_signal
from alpha.gating import gate_signals
from alpha.signal_combiner import combine_signals, combine_signals_correlation_only
from core.decision.decision_engine import (
    DecisionEngine,
    apply_decision_to_signals,
    neutral_decision_for_signals_only,
)
from detection.anomaly import AnomalyPipeline
from features.returns import compute_features
from hedging.hedge_overlay import recommend_hedge
from portfolio.constraints import apply_constraints
from portfolio.optimizer import optimize_weights
from portfolio.risk_parity import inverse_vol_weights
from portfolio.risk_targeting import vol_target_scale
from portfolio.target_weights import signals_to_weights
from portfolio.transaction_costs import turnover_cost
from pre.settings import AppSettings
from regime.regime_state import classify_regime_full
from risk.portfolio import PortfolioRisk
from risk.var import compute_full_var
from scipy import stats

from backtest.evaluation import summarize_backtest

Mode = Literal[
    "full",
    "baseline",
    "vol_target_only",
    "signals_only",
    "corr_signal_only",
    "placebo_random",
]


@dataclass
class BacktestState:
    corr_history: list[float] = field(default_factory=list)
    prev_label: str | None = None
    regime_duration: int = 0
    last_transition_iso: str = ""
    feature_rows: list[np.ndarray] = field(default_factory=list)


@dataclass
class BacktestResult:
    equity: pd.Series
    turnover: pd.Series
    decision_log: list[dict[str, Any]]
    metrics: dict[str, float]
    mode: str


def _decision_engine_var99(
    tail_lr1: pd.DataFrame,
    lr10: pd.DataFrame,
    tickers: list[str],
    prior_w: pd.Series,
    sigma: np.ndarray,
    settings: AppSettings,
    rng: np.random.Generator,
) -> tuple[float, str]:
    """1d 99% VaR (positive loss) for decision layer; aligns with `compute_full_var` in live code."""
    t1 = tail_lr1.reindex(columns=tickers)
    t10 = lr10.reindex(columns=tickers)
    common_idx = t1.index.intersection(t10.index)
    t1 = t1.loc[common_idx].dropna(how="any")
    t10 = t10.loc[t1.index]
    ok = ~(t1.isna().any(axis=1) | t10.isna().any(axis=1))
    t1 = t1.loc[ok]
    t10 = t10.loc[ok]
    if len(t1) < 30:
        wn = prior_w.reindex(tickers).fillna(0.0).astype(float)
        if float(wn.sum()) > 1e-12:
            wn = wn / float(wn.sum())
        else:
            wn = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        pr = PortfolioRisk.from_weights_sigma(wn, sigma, tickers)
        z = float(-stats.norm.ppf(0.01))
        return z * pr.portfolio_vol, "parametric_normal_99"
    w_arr = prior_w.reindex(tickers).fillna(0.0).values.astype(float)
    sw = float(w_arr.sum())
    if sw > 1e-12:
        w_arr = w_arr / sw
    else:
        w_arr = np.ones(len(tickers), dtype=float) / len(tickers)
    lr1_m = t1.values.astype(float)
    lr10_m = t10.values.astype(float)
    mu_1d = lr1_m.mean(axis=0)
    mu_10d = lr10_m.mean(axis=0)
    var_res = compute_full_var(
        lr1_m,
        lr10_m,
        w_arr,
        sigma,
        mu_1d,
        mu_10d,
        settings.var_confidence_levels,
        settings.backtest.var_mc_sims,
        rng,
    )
    return float(var_res.mc_var.get((0.99, 1), 0.0)), "mc_full_var"


def _cov_to_corr(sigma: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
    R = sigma / np.outer(d, d)
    n = R.shape[0]
    if n < 2:
        return R, 0.0
    ac = float((np.sum(R) - np.trace(R)) / (n * (n - 1)))
    return R, ac


def run_backtest(
    closes: pd.DataFrame,
    settings: AppSettings,
    *,
    mode: Mode = "full",
    random_seed: int = 42,
    warmup: int = 130,
) -> BacktestResult:
    rng = np.random.default_rng(random_seed)
    tickers = list(closes.columns)
    log_rows: list[dict[str, Any]] = []
    equity_vals: list[float] = []
    turn_vals: list[float] = []
    prior_w = pd.Series(1.0 / len(tickers), index=tickers)
    cash = settings.backtest.initial_cash
    equity = cash

    st = BacktestState()
    de = DecisionEngine(settings)

    for t in range(warmup, len(closes) - 1):
        sub = closes.iloc[: t + 1]
        w_eq = pd.Series(1.0 / len(tickers), index=tickers)
        feat = compute_features(sub, settings, w_eq)
        lr1 = feat.log_returns_1d.dropna(how="all")
        if len(lr1) < 60:
            continue
        lr_use = lr1[tickers].dropna()
        if len(lr_use) < 40:
            continue
        tail = lr_use.iloc[-min(252, len(lr_use)) :]
        sigma = np.cov(tail.values.T)
        sigma = np.nan_to_num(sigma, nan=1e-6)
        R, avg_corr = _cov_to_corr(sigma)
        st.corr_history.append(avg_corr)
        if len(st.corr_history) > 400:
            st.corr_history = st.corr_history[-400:]

        corr_res = correlation_regime_signal(st.corr_history, avg_corr, settings)

        tail_m = 1.0
        med_vol = float(np.sqrt(np.median(np.diag(sigma))) + 1e-8)
        pdd = float(feat.drawdown_portfolio.iloc[-1]) if len(feat.drawdown_portfolio) else 0.0

        anomalies = AnomalyPipeline(settings).run(feat, w_eq)
        anom_count = len(anomalies)

        row_feat = np.array(
            [med_vol * np.sqrt(252), avg_corr, tail_m, abs(min(pdd, 0.0))],
            dtype=float,
        )
        st.feature_rows.append(row_feat)
        fh = np.stack(st.feature_rows[-min(200, len(st.feature_rows)) :], axis=0)

        ro = classify_regime_full(
            settings,
            tail_m,
            avg_corr,
            med_vol,
            pdd,
            anom_count,
            prev_label=st.prev_label,
            prev_duration=st.regime_duration,
            last_transition_iso=st.last_transition_iso or "",
            feature_history=fh if len(fh) >= settings.regime.hmm_min_history else None,
        )
        st.prev_label = ro.label
        st.regime_duration = ro.duration_bars
        st.last_transition_iso = ro.last_transition_iso

        var_99, var_99_method = _decision_engine_var99(
            tail, feat.log_returns_10d, tickers, prior_w, sigma, settings, rng
        )
        pr = PortfolioRisk.from_weights_sigma(w_eq.reindex(tickers).fillna(0), sigma, tickers)
        forecast_ann_vol = pr.portfolio_vol * np.sqrt(252)

        if mode == "signals_only":
            decision = neutral_decision_for_signals_only()
        elif mode == "corr_signal_only":
            decision = neutral_decision_for_signals_only()
        elif mode in ("baseline", "vol_target_only", "placebo_random"):
            decision = de.decide(
                ro.label, corr_res, anom_count, var_99, settings.risk_limit_var_99
            )
        else:
            decision = de.decide(
                ro.label, corr_res, anom_count, var_99, settings.risk_limit_var_99
            )

        if mode == "baseline":
            w_star = pd.Series(1.0 / len(tickers), index=tickers)
        elif mode == "vol_target_only":
            vols = tail.std(ddof=1).replace(0, np.nan).fillna(0.01)
            w_star = inverse_vol_weights(vols)
        elif mode == "placebo_random":
            raw = pd.Series(rng.standard_normal(len(tickers)), index=tickers)
            w_star = signals_to_weights(raw, settings)
        elif mode == "signals_only":
            comb = combine_signals(lr_use, sub, corr_res, settings)
            raw = comb.per_asset.reindex(tickers).fillna(0.0)
            raw = gate_signals(raw, ro.label, anom_count)
            raw = apply_decision_to_signals(raw, decision, ro.label)
            w_star = optimize_weights("CALM", raw, sigma, tickers, settings)
        elif mode == "corr_signal_only":
            comb = combine_signals_correlation_only(tickers, corr_res, settings)
            raw = comb.per_asset.reindex(tickers).fillna(0.0)
            raw = gate_signals(raw, ro.label, anom_count)
            raw = apply_decision_to_signals(raw, decision, ro.label)
            w_star = optimize_weights("CALM", raw, sigma, tickers, settings)
        else:
            comb = combine_signals(lr_use, sub, corr_res, settings)
            raw = comb.per_asset.reindex(tickers).fillna(0.0)
            raw = gate_signals(raw, ro.label, anom_count)
            raw = apply_decision_to_signals(raw, decision, ro.label)
            w_star = optimize_weights(ro.label, raw, sigma, tickers, settings)

        w_star = vol_target_scale(
            w_star,
            forecast_ann_vol,
            settings.portfolio.target_ann_vol,
            settings.portfolio.max_gross_leverage,
        )
        w_star = apply_constraints(w_star, prior_w, settings)

        cost = turnover_cost(prior_w, w_star, settings)
        equity *= 1.0 - cost

        px0 = closes.iloc[t]
        px1 = closes.iloc[t + 1]
        ret = np.log(px1 / px0).reindex(tickers).fillna(0.0)
        port_ret = float((w_star * ret).sum())
        equity *= np.exp(port_ret)

        hedge = recommend_hedge(
            decision, feat.rolling_beta.reindex(tickers).fillna(0), prior_w, tail_m, settings
        )

        turn = float((w_star - prior_w).abs().sum())
        turn_vals.append(turn)
        prior_w = w_star.copy()
        equity_vals.append(equity)

        log_rows.append(
            {
                "t": t,
                "timestamp": str(closes.index[t]),
                "regime": ro.label,
                "corr_z": corr_res.corr_z,
                "corr_bucket": corr_res.bucket,
                "decision_priority": decision.decision_priority,
                "decision_secondary": "|".join(decision.secondary_reasons),
                "decision": decision.reason_codes,
                "decision_narrative": decision.narrative,
                "anomaly_count": anom_count,
                "var_99": var_99,
                "var_99_method": var_99_method,
                "gross_exposure": float(w_star.sum()),
                "hedge": hedge.narrative,
                "pnl_frac": port_ret,
            }
        )

    idx = closes.index[warmup + 1 : warmup + 1 + len(equity_vals)]
    if len(idx) != len(equity_vals):
        idx = pd.RangeIndex(len(equity_vals))
    eq = pd.Series(equity_vals, index=idx[: len(equity_vals)])
    turn_s = pd.Series(turn_vals, index=eq.index)
    metrics = summarize_backtest(eq, turn_s, settings.risk_free_annual)
    metrics["var_breach_rate"] = 0.0
    return BacktestResult(
        equity=eq,
        turnover=turn_s,
        decision_log=log_rows,
        metrics=metrics,
        mode=mode,
    )
