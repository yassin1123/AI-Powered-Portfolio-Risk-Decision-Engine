"""What-if shocks and counterfactual decision dicts (API-only; no live book mutation)."""

from scenario.shocks import (
    apply_decision_overrides,
    counterfactual_disable_regime_gating_note,
    counterfactual_force_exposure,
    counterfactual_zero_anomalies,
    delta_vs_base,
    shock_market_state,
)

__all__ = [
    "apply_decision_overrides",
    "counterfactual_disable_regime_gating_note",
    "counterfactual_force_exposure",
    "counterfactual_zero_anomalies",
    "delta_vs_base",
    "shock_market_state",
]
