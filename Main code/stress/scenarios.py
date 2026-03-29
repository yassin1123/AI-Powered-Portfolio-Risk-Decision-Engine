"""Historical scenario library and reverse stress test (brief §5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.optimize import minimize

from pre.settings import AppSettings


@dataclass
class ScenarioResult:
    name: str
    portfolio_pnl: float
    by_asset: dict[str, float]
    direct: float
    correlation_effect: float
    vol_effect: float


def _default_shocks() -> dict[str, dict[str, float]]:
    """Illustrative shock weights by asset class keyword (brief §5.1)."""
    return {
        "rates_shock_200bps": {
            "SHY": -0.02,
            "IEF": -0.08,
            "TLT": -0.15,
            "HYG": -0.12,
            "EMB": -0.10,
            "SPY": -0.10,
            "QQQ": -0.12,
            "GLD": 0.03,
        },
        "equity_crash": {
            "SPY": -0.28,
            "QQQ": -0.32,
            "IWM": -0.30,
            "^VIX": 1.8,
            "UVXY": 0.5,
            "TLT": 0.06,
        },
        "credit_crisis": {
            "HYG": -0.35,
            "EMB": -0.25,
            "SPY": -0.22,
            "IEF": 0.06,
        },
        "usd_surge": {
            "UUP": 0.12,
            "EMB": -0.15,
            "BTC-USD": -0.08,
            "GLD": -0.05,
            "PDBC": -0.10,
        },
        "oil_spike": {
            "USO": 0.22,
            "XLE": 0.15,
            "XOM": 0.12,
            "CVX": 0.10,
            "SPY": -0.05,
        },
        "crypto_contagion": {
            "BTC-USD": -0.60,
            "ETH-USD": -0.65,
            "SOL-USD": -0.70,
            "BNB-USD": -0.55,
            "SPY": -0.08,
            "QQQ": -0.08,
        },
        "stagflation": {
            "SPY": -0.25,
            "TLT": -0.15,
            "GLD": 0.08,
            "USO": 0.05,
            "PDBC": 0.12,
        },
    }


class ScenarioLibrary:
    def __init__(self) -> None:
        self.scenarios = _default_shocks()

    def names(self) -> list[str]:
        return list(self.scenarios.keys())


def run_scenario(
    name: str,
    weights: pd.Series,
    tickers: list[str],
    sigma: np.ndarray,
    garch_mult: float,
    rho_shift: float,
    library: ScenarioLibrary | None = None,
) -> ScenarioResult:
    """
    Apply shock dict to assets; decompose P&amp;L (brief §5.2).
    """
    lib = library or ScenarioLibrary()
    shocks = lib.scenarios.get(name, {})
    w = weights.reindex(tickers).fillna(0.0).values.astype(float)
    if w.sum() != 0:
        w = w / w.sum()
    asset_ret = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        asset_ret[i] = shocks.get(t, shocks.get(t.replace("^", ""), 0.0))
    direct = float(w @ asset_ret)
    avg_corr = float((np.sum(sigma) - np.trace(sigma)) / (len(tickers) * (len(tickers) - 1) + 1e-9))
    corr_effect = float(rho_shift * avg_corr * direct)
    vol_effect = float((garch_mult - 1.0) * direct)
    total = direct + corr_effect + vol_effect
    by_asset = {tickers[i]: float(w[i] * asset_ret[i]) for i in range(len(tickers))}
    return ScenarioResult(
        name=name,
        portfolio_pnl=total,
        by_asset=by_asset,
        direct=direct,
        correlation_effect=corr_effect,
        vol_effect=vol_effect,
    )


def reverse_stress_test(
    weights: pd.Series,
    tickers: list[str],
    sigma: np.ndarray,
    target_loss: float,
) -> dict[str, Any]:
    """
    Minimize Mahalanobis distance ||shock||_Sigma^{-1} s.t. w^T shock = target_loss (brief §5.3).
    """
    w = weights.reindex(tickers).fillna(0.0).values.astype(float)
    if w.sum() != 0:
        w = w / w.sum()
    n = len(w)
    sig = np.asarray(sigma, dtype=float) + np.eye(n) * 1e-8
    sig_inv = pinv(sig)

    def obj(x: np.ndarray) -> float:
        return float(x @ sig_inv @ x)

    cons = {"type": "eq", "fun": lambda x: float(w @ x) - target_loss}
    x0 = np.full(n, target_loss / (w @ w + 1e-9) * w)
    bounds = [(-0.99, 0.99)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return {
        "shock": {tickers[i]: float(res.x[i]) for i in range(n)},
        "mahalanobis_sq": float(res.fun),
        "success": res.success,
    }
