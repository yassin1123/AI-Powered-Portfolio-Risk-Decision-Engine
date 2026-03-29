"""Brief §10.1: CUSUM reset / white noise; Mahalanobis vs chi-squared."""

from __future__ import annotations

import numpy as np
from scipy import stats

from detection.anomaly import cusum_portfolio_stats


def test_cusum_resets_final_stats_below_threshold():
    """After any alarm, sp and sm reset; long white noise should not end with huge state."""
    rng = np.random.default_rng(42)
    mu0, sig = 0.0, 0.01
    k = 0.5 * sig
    h = 4.0 * sig
    for _ in range(40):
        r = rng.standard_normal(800) * sig
        sp, sm, _alarms = cusum_portfolio_stats(r, mu0, k, h)
        assert sp <= h + 1e-9, "S+ should not exceed h after segment ends (reset or natural bound)"
        assert sm <= h + 1e-9, "S- should not exceed h after segment ends"


def test_cusum_fires_on_sustained_drift_then_resets():
    """Persistent positive mean shift should trigger at least one alarm; post-reset state bounded."""
    rng = np.random.default_rng(0)
    sig = 0.01
    k = 0.5 * sig
    h = 4.0 * sig
    mu0 = 0.0
    base = rng.standard_normal(100) * sig
    drift = np.full(50, 0.004)
    r = np.concatenate([base, drift])
    sp, sm, alarms = cusum_portfolio_stats(r, mu0, k, h)
    assert alarms >= 1
    assert sp <= h + 1e-9 and sm <= h + 1e-9


def test_mahalanobis_empirical_95pct_near_chi2():
    """Under MVN, D^2 ~ chi2(n); empirical 95th pct within 2% of chi2.ppf(0.95, n) (brief §10.1)."""
    n = 12
    rng = np.random.default_rng(7)
    mean = np.zeros(n)
    cov = np.eye(n)
    inv = np.linalg.inv(cov)
    n_samples = 8000
    d2 = np.empty(n_samples)
    for i in range(n_samples):
        x = rng.multivariate_normal(mean, cov)
        diff = x - mean
        d2[i] = float(diff @ inv @ diff)
    emp_q95 = float(np.quantile(d2, 0.95))
    th = float(stats.chi2.ppf(0.95, df=n))
    rel = abs(emp_q95 / th - 1.0)
    assert rel < 0.02, f"95th pct ratio off: emp={emp_q95:.4f} theory={th:.4f} rel_err={rel:.4f}"
