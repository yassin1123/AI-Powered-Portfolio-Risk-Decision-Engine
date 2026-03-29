"""Portfolio risk: risk contributions (brief §6.1), Calmar, drawdown stats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def risk_contributions(weights: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    RC_i = w_i * (Sigma @ w)_i / sqrt(w^T @ Sigma @ w)  (brief §6.1)
    """
    w = np.asarray(weights, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float)
    sw = s @ w
    den = float(np.sqrt(max(w @ sw, 1e-18)))
    return w * sw / den


def calmar_ratio(
    log_returns: pd.Series, max_drawdown: float, trading_days: int = 252
) -> float:
    if max_drawdown >= 0 or abs(max_drawdown) < 1e-12:
        return float("nan")
    ann = float(log_returns.mean() * trading_days)
    return ann / abs(max_drawdown)


@dataclass
class PortfolioRisk:
    weights: pd.Series
    sigma_t: np.ndarray
    risk_contributions: pd.Series
    portfolio_vol: float

    @classmethod
    def from_weights_sigma(cls, weights: pd.Series, sigma: np.ndarray, tickers: list[str]) -> PortfolioRisk:
        w = weights.reindex(tickers).fillna(0.0).values.astype(float)
        if w.sum() != 0:
            w = w / w.sum()
        rc = risk_contributions(w, sigma)
        pv = float(np.sqrt(max(w @ sigma @ w, 0.0)))
        return cls(
            weights=pd.Series(w, index=tickers),
            sigma_t=sigma,
            risk_contributions=pd.Series(rc, index=tickers),
            portfolio_vol=pv,
        )
