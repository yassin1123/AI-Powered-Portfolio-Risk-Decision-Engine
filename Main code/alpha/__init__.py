"""Alpha and signal layer: correlation regime (flagship) + academic baselines."""

from alpha.correlation_regime_signal import CorrRegimeSignalResult, correlation_regime_signal
from alpha.signal_combiner import (
    CombinedSignals,
    combine_signals,
    combine_signals_correlation_only,
)

__all__ = [
    "CorrRegimeSignalResult",
    "correlation_regime_signal",
    "CombinedSignals",
    "combine_signals",
    "combine_signals_correlation_only",
]
