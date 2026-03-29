from __future__ import annotations

import numpy as np

from diagnostics.contagion import contagion_index


def test_contagion_high_for_uniform_corr() -> None:
    n = 5
    R = np.ones((n, n)) * 0.8
    np.fill_diagonal(R, 1.0)
    c = contagion_index(R)
    assert c > 0.5
