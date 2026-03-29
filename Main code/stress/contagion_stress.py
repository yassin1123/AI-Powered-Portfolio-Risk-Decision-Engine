"""Blend correlation matrix toward high-corr crisis template."""

from __future__ import annotations

import numpy as np


def stress_correlation_matrix(R: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    n = R.shape[0]
    crisis = np.ones((n, n)) * 0.85
    np.fill_diagonal(crisis, 1.0)
    return (1 - intensity) * R + intensity * crisis
