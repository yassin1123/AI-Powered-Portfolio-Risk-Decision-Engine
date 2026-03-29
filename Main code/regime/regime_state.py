"""Regime label, probabilities, persistence, optional transition matrix from feature history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pre.settings import AppSettings
from regime.hmm_regime import build_feature_matrix_row, hmm_posteriors_and_transitions
from regime.rule_based import rule_regime_features, rule_regime_label


@dataclass
class RegimeOutput:
    label: str
    prob_vector: dict[str, float]
    confidence: float
    duration_bars: int
    last_transition_iso: str
    transition_matrix: list[list[float]] | None
    persistence_note: str
    features: dict[str, float] = field(default_factory=dict)


def classify_regime_full(
    settings: AppSettings,
    tail_multiplier: float,
    avg_pairwise_corr: float,
    med_vol: float,
    portfolio_drawdown: float,
    anomaly_count: int,
    *,
    prev_label: str | None,
    prev_duration: int,
    last_transition_iso: str,
    feature_history: np.ndarray | None = None,
) -> RegimeOutput:
    now = datetime.now(timezone.utc).isoformat()
    feats = rule_regime_features(
        tail_multiplier,
        avg_pairwise_corr,
        med_vol,
        portfolio_drawdown,
        anomaly_count,
        settings,
    )
    label = rule_regime_label(feats, settings)

    if prev_label is None or prev_label != label:
        duration = 1
        trans_iso = now
    else:
        duration = prev_duration + 1
        trans_iso = last_transition_iso

    prob_vector = {label: 1.0}
    trans_matrix: list[list[float]] | None = None
    confidence = 0.72

    if (
        feature_history is not None
        and len(feature_history) >= settings.regime.hmm_min_history
        and settings.regime.use_hmm_features
    ):
        post, trans_matrix = hmm_posteriors_and_transitions(feature_history, n_states=3)
        if post is not None:
            names = ["CALM", "TRANSITION", "STRESSED"]
            prob_vector = {
                names[i]: float(post[i]) for i in range(min(len(names), len(post)))
            }
            label = max(prob_vector, key=prob_vector.get)  # type: ignore[arg-type, return-value]
            confidence = float(max(post))

    return RegimeOutput(
        label=label,
        prob_vector=prob_vector,
        confidence=confidence,
        duration_bars=duration,
        last_transition_iso=trans_iso,
        transition_matrix=trans_matrix,
        persistence_note=f"In {label} for {duration} bar(s).",
        features=feats,
    )


def regime_output_to_dict(ro: RegimeOutput) -> dict[str, Any]:
    return {
        "label": ro.label,
        "prob_vector": ro.prob_vector,
        "confidence": ro.confidence,
        "duration_bars": ro.duration_bars,
        "last_transition_iso": ro.last_transition_iso,
        "transition_matrix": ro.transition_matrix,
        "persistence_note": ro.persistence_note,
        "features": ro.features,
    }
