"""Decision engine stress + corr_z path and priority ordering."""

from __future__ import annotations

from alpha.correlation_regime_signal import CorrRegimeSignalResult
from core.decision.decision_engine import DecisionEngine
from core.decision.decision_trace import compute_confidence
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


def test_confidence_transition_many_anomalies_can_reach_eighties() -> None:
    """Default confidence_* settings allow strong regime + driver mass to read ~80%+."""
    s = AppSettings()
    drivers = [{"delta": 0.85}]
    c = compute_confidence(
        s,
        regime_confidence=1.0,
        anomaly_count=10,
        drivers=drivers,
        priority="transition",
    )
    assert c >= 0.80


def test_confidence_legacy_strict_divisor_stays_below_eighty() -> None:
    """Old divisor (6) + full transition penalty caps transition+10 anomalies under 80%."""
    s = AppSettings()
    s.decision_engine.confidence_anomaly_divisor = 6.0
    s.decision_engine.confidence_transition_penalty = 0.12
    drivers = [{"delta": 0.85}]
    c = compute_confidence(
        s,
        regime_confidence=1.0,
        anomaly_count=10,
        drivers=drivers,
        priority="transition",
    )
    assert c < 0.80
