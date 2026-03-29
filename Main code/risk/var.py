"""HS-VaR, MC-VaR (Cholesky), Cornish–Fisher tail adjustment (brief §3.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.linalg import cholesky


def cornish_fisher_z(z: float, skew: float, excess_kurt: float) -> float:
    """Cornish–Fisher expansion (brief §3.1)."""
    g1, g2 = skew, excess_kurt
    return (
        z
        + (z**2 - 1) * g1 / 6.0
        + (z**3 - 3 * z) * g2 / 24.0
        - (2 * z**3 - 5 * z) * g1**2 / 36.0
    )


def historical_var_cvar(
    portfolio_returns: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """
    Sort P&L ascending; VaR = -quantile at (1-alpha); CVaR = mean of tail beyond VaR.
    Returns (VaR, CVaR) as positive loss magnitudes.
    """
    x = np.asarray(portfolio_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 0.0, 0.0
    q = np.quantile(x, 1.0 - alpha)
    var = float(-q)
    tail = x[x <= q]
    cvar = float(-tail.mean()) if len(tail) else var
    return var, max(cvar, var)


def monte_carlo_var_cvar(
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    n_sims: int,
    alpha: float,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray, float, float]:
    """
    R_sim = mu + L @ Z; portfolio = w^T R_sim per draw.
    Returns (var, cvar, sim_port, skew, excess_kurtosis) for CF multiplier.
    """
    rng = rng or np.random.default_rng()
    n = len(mu)
    w = np.asarray(weights, dtype=float).ravel()
    sig = np.asarray(sigma, dtype=float) + np.eye(n) * 1e-10
    L = cholesky(sig, lower=True, check_finite=False)
    z = rng.standard_normal((n, n_sims))
    r = mu.reshape(-1, 1) + L @ z
    port = (w @ r).ravel()
    var, cvar = historical_var_cvar(port, alpha)
    skew = float(stats.skew(port))
    ex_k = float(stats.kurtosis(port, fisher=True))
    return var, cvar, port, skew, ex_k


def simulate_portfolio_returns(
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    n_sims: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Single MC batch: R = mu + L @ Z, portfolio = w^T R."""
    rng = rng or np.random.default_rng()
    n = len(mu)
    w = np.asarray(weights, dtype=float).ravel()
    sig = np.asarray(sigma, dtype=float) + np.eye(n) * 1e-10
    L = cholesky(sig, lower=True, check_finite=False)
    z = rng.standard_normal((n, n_sims))
    out = (w @ (mu.reshape(-1, 1) + L @ z)).ravel()
    return np.asarray(out, dtype=np.float64)


def cornish_fisher_cvar_adjustment(
    port_sims: np.ndarray, alpha: float
) -> tuple[float, float]:
    """CF-adjusted CVaR vs normal CVaR at same alpha; returns (cf_cvar, tail_multiplier)."""
    var_n, cvar_n = historical_var_cvar(port_sims, alpha)
    skew = float(stats.skew(port_sims))
    ex_k = float(stats.kurtosis(port_sims, fisher=True))
    z = stats.norm.ppf(alpha)
    z_cf = cornish_fisher_z(z, skew, ex_k)
    # approximate tail mean shift using empirical quantile mapped through CF
    # scale tail losses by ratio of CF z to normal z for reporting
    if abs(z) > 1e-8:
        mult = abs(z_cf / z)
    else:
        mult = 1.0
    cf_cvar = float(cvar_n * mult)
    tail_mult = cf_cvar / cvar_n if cvar_n > 1e-12 else 1.0
    return cf_cvar, tail_mult


@dataclass
class VaRResult:
    hs_var: dict[tuple[float, int], float]
    hs_cvar: dict[tuple[float, int], float]
    mc_var: dict[tuple[float, int], float]
    mc_cvar: dict[tuple[float, int], float]
    cf_cvar_99: float
    tail_multiplier: float
    mc_port_1d_sims: np.ndarray


def compute_full_var(
    lr1: np.ndarray,
    lr10: np.ndarray,
    w: np.ndarray,
    sigma_t: np.ndarray,
    mu_1d: np.ndarray,
    mu_10d: np.ndarray,
    alphas: list[float],
    n_sims: int,
    rng: np.random.Generator | None = None,
) -> VaRResult:
    """
    lr1, lr10: (T, n) aligned; w: (n,); sigma_t from D @ R @ D; mu from trailing window means.
    """
    rng = rng or np.random.default_rng()
    hs_v: dict[tuple[float, int], float] = {}
    hs_c: dict[tuple[float, int], float] = {}
    mc_v: dict[tuple[float, int], float] = {}
    mc_c: dict[tuple[float, int], float] = {}
    port_hist_1 = (lr1 @ w).ravel()
    port_hist_1 = port_hist_1[np.isfinite(port_hist_1)]
    port_hist_10 = (lr10 @ w).ravel()
    port_hist_10 = port_hist_10[np.isfinite(port_hist_10)]
    port1 = simulate_portfolio_returns(mu_1d, sigma_t, w, n_sims, rng)
    # 10d MC: scale covariance to 10-period (i.i.d. multivariate normal: Cov_10d = 10 * Cov_1d).
    sigma_10d = 10.0 * np.asarray(sigma_t, dtype=float)
    port10 = simulate_portfolio_returns(mu_10d, sigma_10d, w, n_sims, rng)
    for a in alphas:
        v1, c1 = historical_var_cvar(port_hist_1, a)
        hs_v[(a, 1)] = v1
        hs_c[(a, 1)] = c1
        v10, c10 = historical_var_cvar(port_hist_10, a)
        hs_v[(a, 10)] = v10
        hs_c[(a, 10)] = c10
        mv1, mc1 = historical_var_cvar(port1, a)
        mc_v[(a, 1)] = mv1
        mc_c[(a, 1)] = mc1
        mv10, mc10 = historical_var_cvar(port10, a)
        mc_v[(a, 10)] = mv10
        mc_c[(a, 10)] = mc10
    cf_cvar, tail_m = cornish_fisher_cvar_adjustment(port1, 0.99)
    return VaRResult(
        hs_var=hs_v,
        hs_cvar=hs_c,
        mc_var=mc_v,
        mc_cvar=mc_c,
        cf_cvar_99=cf_cvar,
        tail_multiplier=tail_m,
        mc_port_1d_sims=port1,
    )
