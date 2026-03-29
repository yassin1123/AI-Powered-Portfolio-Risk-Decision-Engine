from __future__ import annotations

import pandas as pd

from alpha.gating import gate_signals


def test_stress_scales_down() -> None:
    s = pd.Series({"a": 1.0, "b": 1.0})
    g = gate_signals(s, "STRESSED", 0)
    assert float(g.sum()) < float(s.sum())
