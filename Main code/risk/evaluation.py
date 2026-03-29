"""VaR backtest: Kupiec unconditional coverage; Christoffersen conditional coverage; vol RMSE."""

from __future__ import annotations

import numpy as np
from scipy import stats


def kupiec_lr_stat(violations: np.ndarray, expected_rate: float) -> tuple[float, float]:
    """LR statistic for unconditional coverage; returns (stat, p-value approx chi2(1))."""
    n = len(violations)
    if n == 0:
        return 0.0, 1.0
    x = int(np.sum(violations))
    p = expected_rate
    p_hat = x / n
    if p_hat in (0, 1) or p <= 0 or p >= 1:
        return 0.0, 1.0
    ll1 = x * np.log(p_hat) + (n - x) * np.log(1 - p_hat)
    ll0 = x * np.log(p) + (n - x) * np.log(1 - p)
    lr = -2 * (ll0 - ll1)
    pv = 1 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pv)


def christoffersen_conditional(violations: np.ndarray) -> tuple[float, float]:
    """Simple independence test on violation clusters (Markov chain LR)."""
    v = violations.astype(int)
    if len(v) < 10:
        return 0.0, 1.0
    n00 = n01 = n10 = n11 = 0
    for i in range(len(v) - 1):
        a, b = v[i], v[i + 1]
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    if min(n0, n1, n01, n00, n10, n11) <= 0:
        return 0.0, 1.0
    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / (n0 + n1)
    ll1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
    ll0 = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    lr = -2 * (ll0 - ll1)
    pv = 1 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pv)


def vol_rmse_mae(forecast_vol: np.ndarray, realized_vol: np.ndarray) -> dict[str, float]:
    e = forecast_vol - realized_vol
    return {"rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e)))}
