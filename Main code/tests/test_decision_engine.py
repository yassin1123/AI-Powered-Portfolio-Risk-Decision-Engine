"""Decision engine stress + corr_z path and priority ordering."""

from __future__ import annotations

from alpha.correlation_regime_signal import CorrRegimeSignalResult
from core.decision.decision_engine import DecisionEngine
from pre.settings import AppSettings


def _cr(z: float) -> CorrRegimeSignalResult:
    return CorrRegimeSignalResult(0.5, 0.3, 0.1, z, "normal", False, False, False)


def test_stress_corr_z_overrides() -> None:
    s = AppSettings()
    de = DecisionEngine(s)
    cr = _cr(1.2)
    d = de.decide("STRESSED", cr, 0, 0.02, 0.05)
    assert d.exposure_scale < 1.0
    assert d.decision_priority == "stress_corr_override"


def test_stress_high_corr_beats_anomaly() -> None:
    """STRESSED + corr_z above stress threshold wins over anomaly_suppress."""
    s = AppSettings()
    de = DecisionEngine(s)
    z_thr = s.decision_engine.stress_corr_z_threshold
    cr = _cr(z_thr + 0.25)
    d = de.decide("STRESSED", cr, 99, 0.02, 0.05)
    assert d.decision_priority == "stress_corr_override"


def test_anomaly_when_not_stress_combo() -> None:
    s = AppSettings()
    de = DecisionEngine(s)
    cr = _cr(0.0)
    d = de.decide("CALM", cr, 99, 0.02, 0.05)
    assert d.decision_priority == "anomaly_suppress"


def test_corr_crisis_when_calm_high_z() -> None:
    s = AppSettings()
    de = DecisionEngine(s)
    z = s.correlation_signal.z_high + 0.1
    cr = _cr(z)
    d = de.decide("CALM", cr, 0, 0.02, 0.05)
    assert d.decision_priority == "corr_crisis"
