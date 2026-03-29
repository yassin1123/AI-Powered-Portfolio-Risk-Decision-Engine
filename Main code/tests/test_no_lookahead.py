"""API-level: correlation regime signal only uses caller-supplied history (past corrs)."""

from __future__ import annotations

import numpy as np

from alpha.correlation_regime_signal import correlation_regime_signal
from pre.settings import AppSettings


def test_correlation_signal_finite_with_short_history() -> None:
    settings = AppSettings()
    hist = [0.15 + 0.01 * i for i in range(30)]
    r = correlation_regime_signal(hist, hist[-1], settings)
    assert np.isfinite(r.corr_z)
