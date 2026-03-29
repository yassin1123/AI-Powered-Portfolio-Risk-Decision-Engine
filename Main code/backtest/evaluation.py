"""Sharpe, Sortino, max DD, Calmar, turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def annualized_sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    x = returns.dropna()
    if len(x) < 5 or x.std(ddof=1) < 1e-12:
        return 0.0
    daily_rf = (1 + rf) ** (1 / 252) - 1
    return float(np.sqrt(252) * (x.mean() - daily_rf) / x.std(ddof=1))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    x = returns.dropna()
    daily_rf = (1 + rf) ** (1 / 252) - 1
    downside = x[x < daily_rf] - daily_rf
    if len(downside) < 2:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd < 1e-12:
        return 0.0
    return float(np.sqrt(252) * (x.mean() - daily_rf) / dstd)


def var_breach_stats(log_rows: list[dict]) -> dict[str, float]:
    """Approximate daily VaR breaches: log return worse than negative MC VaR (same units as engine log)."""
    if not log_rows:
        return {"var_breach_rate": 0.0, "var_breach_cluster_max": 0.0}
    n = 0
    breaches = 0
    run = 0
    max_run = 0
    for row in log_rows:
        v = row.get("var_99")
        pnl = row.get("pnl_frac")
        if v is None or pnl is None:
            continue
        vf = float(v)
        pf = float(pnl)
        n += 1
        hit = pf < -vf
        if hit:
            breaches += 1
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    rate = breaches / n if n else 0.0
    return {"var_breach_rate": float(rate), "var_breach_cluster_max": float(max_run)}


def summarize_backtest(
    equity_curve: pd.Series,
    turnover_series: pd.Series,
    rf_annual: float,
) -> dict[str, float]:
    r = equity_curve.pct_change().dropna()
    years = len(r) / 252.0
    cagr = (
        float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(years, 1e-6)) - 1)
        if len(equity_curve) > 1 and equity_curve.iloc[0] > 0
        else 0.0
    )
    return {
        "cagr": cagr,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
        "sharpe": annualized_sharpe(r, rf_annual),
        "sortino": sortino_ratio(r, rf_annual),
        "max_dd": max_drawdown(equity_curve),
        "calmar": float(cagr / abs(max_drawdown(equity_curve)))
        if max_drawdown(equity_curve) < -1e-8
        else 0.0,
        "mean_turnover": float(turnover_series.mean()) if len(turnover_series) else 0.0,
    }
