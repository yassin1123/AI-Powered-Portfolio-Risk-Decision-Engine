"""Signal combiner smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.correlation_regime_signal import CorrRegimeSignalResult
from alpha.signal_combiner import combine_signals
from pre.settings import AppSettings


def test_combine_signals_runs() -> None:
    rng = np.random.default_rng(0)
    d = pd.date_range("2020-01-01", periods=80, freq="B")
    lr = pd.DataFrame(rng.standard_normal((80, 3)), index=d, columns=list("ABC"))
    closes = 100 * np.exp(lr.cumsum())
    cr = CorrRegimeSignalResult(0.3, 0.3, 0.05, 0.0, "normal", False, False, False)
    s = AppSettings()
    out = combine_signals(lr, closes, cr, s)
    assert len(out.per_asset) == 3
