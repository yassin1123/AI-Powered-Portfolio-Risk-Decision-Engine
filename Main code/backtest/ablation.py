"""Toggle major engine components for ablation studies (elite brief §5.2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationFlags:
    use_correlation_signal: bool = True
    use_regime_gating: bool = True
    use_anomaly_gating: bool = True
    use_vol_target: bool = True
    use_transaction_costs: bool = True
    use_decision_engine: bool = True
    use_turnover_cap: bool = True
