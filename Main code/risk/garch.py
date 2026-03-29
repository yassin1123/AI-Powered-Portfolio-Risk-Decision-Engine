"""GARCH(1,1) marginal vols + DCC correlation; Sigma_t = D @ R_t @ D (brief §3.2–3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.linalg import cholesky

from pre.settings import AppSettings


@dataclass
class GARCHDCCResult:
    sigma_t: np.ndarray
    R_t: np.ndarray
    D_diag: np.ndarray
    L_chol: np.ndarray
    garch_vol_forecast: dict[str, float]
    standardized_residuals: np.ndarray
    """Per-asset conditional σ series (same window as fit); for dashboard paths without refitting."""
    vol_paths: dict[str, list[float]] | None = None


def _fit_one_garch_forecast(
    r: np.ndarray, max_iter: int = 250
) -> tuple[float, np.ndarray]:
    """
    Returns (one-step conditional vol forecast for last date, conditional vol series h_t).
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 120:
        s = float(np.std(r, ddof=1)) if len(r) > 2 else 0.01
        win = min(20, max(5, len(r) // 3))
        ser = pd.Series(r)
        roll = ser.rolling(win, min_periods=min(3, len(r))).std().bfill().ffill().fillna(s)
        h = np.maximum(roll.to_numpy(dtype=float), 1e-6)
        return float(max(s, 1e-6)), h
    try:
        am = arch_model(r, mean="Constant", vol="Garch", p=1, q=1, rescale=True)
        res = am.fit(disp="off", options={"maxiter": max_iter})
        fc = res.forecast(horizon=1, reindex=False)
        var = float(fc.variance.values[-1, 0])
        cond_vol = np.asarray(res.conditional_volatility, dtype=float)
        cond_vol = np.maximum(cond_vol, 1e-12)
        sig_f = float(np.sqrt(max(var, 1e-12)))
        return max(sig_f, 1e-6), cond_vol
    except Exception:
        s = float(np.std(r, ddof=1)) if len(r) > 1 else 0.01
        ser = pd.Series(r)
        roll = ser.rolling(20, min_periods=5).std().bfill().ffill().fillna(s)
        h = np.maximum(roll.to_numpy(dtype=float), 1e-6)
        return max(s, 1e-6), h


def dcc_R_from_epsilon(epsilon: np.ndarray, a: float = 0.04, b: float = 0.94) -> np.ndarray:
    """
    DCC recursion on standardized residuals eps (T, n). Returns R at T-1 (brief §3.3).
    """
    eps = np.asarray(epsilon, dtype=float)
    t, n = eps.shape
    if t < 5:
        return np.eye(n)
    qbar = np.corrcoef(eps.T)
    qbar = np.nan_to_num(qbar, nan=0.0)
    np.fill_diagonal(qbar, 1.0)
    q = qbar.copy()
    ab = a + b
    if ab >= 1.0:
        a, b = 0.04, 0.94
    for tt in range(1, t):
        e = eps[tt - 1]
        outer = np.outer(e, e)
        q = (1 - a - b) * qbar + a * outer + b * q
    d = np.sqrt(np.clip(np.diag(q), 1e-12, None))
    dinv = np.diag(1.0 / d)
    r_mat = dinv @ q @ dinv
    r_mat = (r_mat + r_mat.T) / 2.0
    np.fill_diagonal(r_mat, 1.0)
    return r_mat


def conditional_covariance_drd(d_diag: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Sigma_t = D @ R @ D."""
    d = np.diag(d_diag)
    s = d @ r @ d
    s = (s + s.T) / 2.0
    return s


def cholesky_cached(
    sigma: np.ndarray,
    prev_sigma: np.ndarray | None,
    prev_L: np.ndarray | None,
    frob_thresh: float,
) -> tuple[np.ndarray, bool]:
    if prev_sigma is not None and prev_L is not None:
        diff = np.linalg.norm(sigma - prev_sigma, ord="fro")
        if diff < frob_thresh:
            return prev_L, False
    n = sigma.shape[0]
    sig = sigma + np.eye(n) * 1e-10
    L = cholesky(sig, lower=True)
    return L, True


def fit_garch_dcc(
    log_returns: pd.DataFrame,
    settings: AppSettings,
    a_dcc: float = 0.04,
    b_dcc: float = 0.94,
) -> GARCHDCCResult:
    """
    Per-asset GARCH(1,1) on last garch_window rows; D from 1-step vol forecast;
    R_t from DCC on standardized residuals (full window).
    """
    tickers = list(log_returns.columns)
    r = log_returns.iloc[-settings.garch_window :].dropna(how="all")
    r = r.dropna(axis=1, how="all")
    tickers = [c for c in tickers if c in r.columns]
    r = r[tickers].dropna()
    if len(r) < 60:
        n = len(tickers)
        sig = np.eye(n) * 0.0001
        return GARCHDCCResult(
            sigma_t=sig,
            R_t=np.eye(n),
            D_diag=np.sqrt(np.diag(sig)),
            L_chol=cholesky(sig + np.eye(n) * 1e-10, lower=True),
            garch_vol_forecast={t: 0.01 for t in tickers},
            standardized_residuals=np.zeros((len(r), n)),
            vol_paths=None,
        )

    n = len(tickers)
    T = len(r)
    eps = np.zeros((T, n))
    d_diag = np.zeros(n)
    vols: dict[str, float] = {}
    cond_vols = np.zeros((T, n))
    for j, t in enumerate(tickers):
        col = r[t].values.astype(float)
        sig_f, hv = _fit_one_garch_forecast(col)
        vols[t] = sig_f
        d_diag[j] = sig_f
        if len(hv) < T:
            hv = np.pad(hv, (T - len(hv), 0), mode="edge")
        hv = hv[-T:]
        cond_vols[:, j] = hv
        eps[:, j] = col / np.maximum(hv, 1e-12)

    # Cap explosive one-step vol forecasts (VIX/UVXY/commodities can yield GARCH σ that
    # poisons Σ and blows up MC-VaR / ann. vol). Cap vs median so typical names unchanged.
    med = float(np.median(np.maximum(d_diag, 1e-12)))
    soft_cap = min(0.28, max(med * 8.0, 0.05))
    d_diag = np.clip(d_diag, 1e-8, soft_cap)

    R_t = dcc_R_from_epsilon(eps, a=a_dcc, b=b_dcc)
    sigma_t = conditional_covariance_drd(d_diag, R_t)
    L = cholesky(sigma_t + np.eye(n) * 1e-10, lower=True)
    vol_paths = {tickers[j]: cond_vols[:, j].astype(float).tolist() for j in range(n)}
    return GARCHDCCResult(
        sigma_t=sigma_t,
        R_t=R_t,
        D_diag=d_diag,
        L_chol=L,
        garch_vol_forecast=vols,
        standardized_residuals=eps,
        vol_paths=vol_paths,
    )


def garch_vol_history_path(
    log_returns: pd.Series, window: int = 504
) -> list[float]:
    """For dashboard: conditional vol path from last fit."""
    r = log_returns.dropna().iloc[-window:].values.astype(float)
    _, hv = _fit_one_garch_forecast(r)
    tail = hv[-252:] if len(hv) > 252 else hv
    return [float(x) for x in tail]
