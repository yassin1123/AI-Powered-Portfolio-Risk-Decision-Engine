"""K-means sequence → soft posteriors + empirical transition matrix (HMM-lite)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def hmm_posteriors_and_transitions(
    feature_history: np.ndarray, n_states: int = 3
) -> tuple[np.ndarray | None, list[list[float]] | None]:
    """
    feature_history: (T, k) features.
    Returns posterior over states for last row (inverse distance to centers),
    and empirical transition matrix from hard cluster sequence.
    """
    X = np.asarray(feature_history, dtype=float)
    if len(X) < 30:
        return None, None
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-9
    Z = (X - mu) / sig
    km = KMeans(n_clusters=n_states, n_init=10, random_state=42)
    labels = km.fit_predict(Z)
    last = Z[-1:]
    d = np.linalg.norm(km.cluster_centers_ - last, axis=1)
    w = 1.0 / (d + 1e-6)
    post = (w / w.sum()).astype(float)
    Tmat = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        Tmat[labels[i], labels[i + 1]] += 1.0
    row_sums = Tmat.sum(axis=1, keepdims=True) + 1e-9
    P = (Tmat / row_sums).tolist()
    return post, P


def build_feature_matrix_row(
    med_vol: float,
    avg_corr: float,
    tail_m: float,
    dd: float,
) -> np.ndarray:
    return np.array(
        [
            float(med_vol * np.sqrt(252)),
            float(avg_corr),
            float(tail_m),
            float(abs(min(dd, 0.0))),
        ],
        dtype=float,
    )
