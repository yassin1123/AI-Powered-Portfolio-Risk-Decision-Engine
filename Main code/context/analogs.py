"""Past-only kNN on standardized state vectors (backend brief §8.1)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _standardize_rows(mat: np.ndarray) -> np.ndarray:
    """mat shape (n, d); skip if n<3."""
    if mat.shape[0] < 3:
        return mat
    mu = np.nanmean(mat, axis=0)
    sig = np.nanstd(mat, axis=0, ddof=1) + 1e-9
    return (mat - mu) / sig


def find_similar_states(
    history_features: list[dict[str, float]],
    current: dict[str, float],
    *,
    k: int = 5,
    keys: tuple[str, ...] = ("corr_z", "vol_norm", "anomaly_norm", "dd_norm", "var_norm"),
) -> dict[str, Any]:
    """
    history_features: oldest-first rows; current is latest observation.
    No lookahead: neighbors are only indices < len(history_features) (excludes appending current before call).
    """
    if len(history_features) < 4:
        return {"neighbors": [], "note": "insufficient_history"}

    rows = []
    for h in history_features:
        rows.append([float(h.get(ky, 0.0)) for ky in keys])
    X = np.asarray(rows, dtype=float)
    q = np.array([float(current.get(ky, 0.0)) for ky in keys], dtype=float)
    Xs = _standardize_rows(X)
    qs = (q - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0, ddof=1) + 1e-9)
    dists = np.linalg.norm(Xs - qs.reshape(1, -1), axis=1)
    idx = np.argsort(dists)[:k]
    neighbors = []
    for j in idx:
        neighbors.append(
            {
                "history_index": int(j),
                "distance": float(dists[j]),
                "features": {keys[i]: float(X[j, i]) for i in range(len(keys))},
                "forward_5d_return": None,
                "forward_10d_vol": None,
            }
        )
    return {
        "neighbors": neighbors,
        "feature_keys": list(keys),
        "note": "forward_outcomes_null_in_live_loop",
    }
