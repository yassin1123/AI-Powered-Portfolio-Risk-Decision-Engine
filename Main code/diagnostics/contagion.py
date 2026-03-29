"""Correlation concentration: λ_max / sum(λ)."""

from __future__ import annotations

import numpy as np


def contagion_index(R: np.ndarray) -> float:
    w, _v = np.linalg.eigh(R)
    w = np.clip(w[::-1], 0, None)
    s = float(w.sum())
    if s <= 1e-12:
        return 0.0
    return float(w[0] / s)
