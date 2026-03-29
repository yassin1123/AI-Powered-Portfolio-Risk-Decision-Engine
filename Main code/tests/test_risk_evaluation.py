from __future__ import annotations

import numpy as np

from risk.evaluation import christoffersen_conditional, kupiec_lr_stat


def test_kupiec_runs() -> None:
    v = np.zeros(100)
    v[:5] = 1
    lr, p = kupiec_lr_stat(v, 0.05)
    assert lr >= 0
    assert 0 <= p <= 1


def test_christoffersen_runs() -> None:
    v = np.array([0, 1, 1, 0, 0, 1, 0] * 5)
    lr, p = christoffersen_conditional(v)
    assert lr >= 0
