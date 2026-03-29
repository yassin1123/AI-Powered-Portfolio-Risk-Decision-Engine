"""Log returns, EWMA covariance (λ=0.94), 10d overlapping returns, beta, Sharpe, drawdown."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pre.settings import AppSettings

EWMA_LAMBDA = 0.94


@dataclass
class FeatureBundle:
    closes: pd.DataFrame
    log_returns_1d: pd.DataFrame
    log_returns_10d: pd.DataFrame
    sample_cov: pd.DataFrame
    ewma_cov: pd.DataFrame
    rolling_beta: pd.Series
    sharpe: pd.Series
    portfolio_sharpe: float
    drawdown_asset: pd.DataFrame
    drawdown_portfolio: pd.Series


def compute_log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    return np.log(closes / closes.shift(1)).dropna(how="all")


def compute_10d_log_returns(log_r_1d: pd.DataFrame) -> pd.DataFrame:
    """Overlapping 10-day log-returns: sum of 10 daily log-returns (Decision record §1)."""
    return log_r_1d.rolling(window=10, min_periods=10).sum().dropna(how="all")


def sample_covariance(returns: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    w = returns if window is None else returns.iloc[-window:]
    return w.cov()


def ewma_covariance(returns: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """RiskMetrics-style EWMA covariance (brief §2.3)."""
    r = returns.dropna()
    n, k = r.shape
    if n < 2:
        return pd.DataFrame(np.eye(k) * 1e-6, index=r.columns, columns=r.columns)
    w = np.zeros((n, 1))
    w[-1] = 1.0
    for i in range(n - 2, -1, -1):
        w[i] = (1 - lam) * (lam ** (n - 2 - i))
    w = w / w.sum()
    mean = (r.values * w).sum(axis=0)
    xc = r.values - mean
    cov = (xc.T * w.ravel()) @ xc
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


def rolling_beta_vs_spy(
    log_returns: pd.DataFrame, spy_col: str = "SPY", window: int = 252
) -> pd.Series:
    if spy_col not in log_returns.columns:
        return pd.Series(0.0, index=log_returns.columns)
    spy = log_returns[spy_col].values
    betas = {}
    for c in log_returns.columns:
        if c == spy_col:
            betas[c] = 1.0
            continue
        y = log_returns[c].values
        m = min(len(y), len(spy))
        y, s = y[-m:], spy[-m:]
        mask = np.isfinite(y) & np.isfinite(s)
        y, s = y[mask], s[mask]
        if len(y) < window:
            betas[c] = float("nan")
            continue
        yw, sw = y[-window:], s[-window:]
        lr = LinearRegression(fit_intercept=True)
        lr.fit(sw.reshape(-1, 1), yw)
        betas[c] = float(lr.coef_[0])
    return pd.Series(betas)


def annualized_sharpe(
    log_returns: pd.DataFrame, rf_annual: float, window: int = 252
) -> pd.Series:
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    out = {}
    for c in log_returns.columns:
        x = log_returns[c].dropna().iloc[-window:]
        if len(x) < 10:
            out[c] = float("nan")
            continue
        excess = x.mean() - rf_daily
        vol = x.std(ddof=1)
        out[c] = float(np.sqrt(252) * excess / vol) if vol > 0 else float("nan")
    return pd.Series(out)


def drawdown_series(prices: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    dd = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for c in prices.columns:
        p = prices[c]
        roll_max = p.cummax()
        dd[c] = p / roll_max - 1.0
    return dd


def portfolio_drawdown(closes: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(closes.columns).fillna(0.0)
    norm = closes / closes.iloc[0]
    port = (norm * w).sum(axis=1)
    rm = port.cummax()
    return port / rm - 1.0


def compute_features(
    closes: pd.DataFrame,
    settings: AppSettings,
    weights: pd.Series,
) -> FeatureBundle:
    lr1 = compute_log_returns(closes)
    lr10 = compute_10d_log_returns(lr1)
    cov_w = settings.covariance_window
    sample = sample_covariance(lr1, cov_w)
    ewma = ewma_covariance(lr1.iloc[-cov_w:])
    beta = rolling_beta_vs_spy(lr1, "SPY", settings.beta_window)
    sharpe = annualized_sharpe(lr1, settings.risk_free_annual, settings.beta_window)
    w = weights.reindex(closes.columns).fillna(0.0)
    w = w / w.sum() if w.sum() != 0 else w
    plr = (lr1 * w).sum(axis=1)
    rf_d = (1 + settings.risk_free_annual) ** (1 / 252) - 1
    ps = (
        float(np.sqrt(252) * (plr.mean() - rf_d) / plr.std(ddof=1))
        if plr.std(ddof=1) > 0
        else float("nan")
    )
    dd_a = drawdown_series(closes)
    dd_p = portfolio_drawdown(closes, w)
    return FeatureBundle(
        closes=closes,
        log_returns_1d=lr1,
        log_returns_10d=lr10,
        sample_cov=sample,
        ewma_cov=ewma,
        rolling_beta=beta,
        sharpe=sharpe,
        portfolio_sharpe=ps,
        drawdown_asset=dd_a,
        drawdown_portfolio=dd_p.reindex(closes.index),
    )
