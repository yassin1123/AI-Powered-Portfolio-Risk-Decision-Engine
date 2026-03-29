"""Average pairwise correlation from correlation matrix R (DCC output)."""

from __future__ import annotations

import numpy as np


def avg_pairwise_correlation(R: np.ndarray) -> float:
    n = R.shape[0]
    if n < 2:
        return 0.0
    s = float((np.sum(R) - np.trace(R)) / (n * (n - 1)))
    return s
