"""Correlation regime z-score and rule branches."""

from __future__ import annotations

import numpy as np

from alpha.correlation_regime_signal import correlation_regime_signal
from pre.settings import AppSettings, CorrelationRegimeSignalConfig


def test_corr_z_spike_triggers_hedge() -> None:
    s = AppSettings(
        correlation_signal=CorrelationRegimeSignalConfig(
            rolling_window=50, z_high=1.5, z_low=-1.0, eps_std=1e-6
        )
    )
    hist = [0.2] * 80
    r = correlation_regime_signal(hist, 0.65, s)
    assert r.corr_z > 1.5
    assert r.activate_hedge and r.reduce_exposure


def test_diversification_bucket() -> None:
    s = AppSettings()
    hist = list(np.linspace(0.5, 0.6, 80))
    r = correlation_regime_signal(hist, 0.15, s)
    assert r.allow_more_risk or r.bucket == "diversification"
