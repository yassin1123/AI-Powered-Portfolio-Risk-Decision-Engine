"""Risk budget and rebalancing signals (brief §6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from pre.settings import AppSettings

Priority = Literal["HIGH", "MEDIUM", "CRITICAL"]
SignalType = Literal[
    "REDUCE_RISK",
    "REBALANCE",
    "HEDGE",
    "CORRELATION_ALERT",
    "DRAWDOWN_STOP",
    "INCREASE",
]


@dataclass
class Signal:
    type: SignalType
    priority: Priority
    message: str
    detail: dict[str, Any]


def generate_signals(
    settings: AppSettings,
    weights: pd.Series,
    target_weights: pd.Series,
    var_99_portfolio: float,
    risk_contributions: pd.Series,
    tail_multiplier: float,
    avg_pairwise_corr: float,
    portfolio_drawdown: float,
) -> list[Signal]:
    sigs: list[Signal] = []
    if var_99_portfolio > settings.risk_limit_var_99:
        sigs.append(
            Signal(
                type="REDUCE_RISK",
                priority="HIGH",
                message="Portfolio VaR(99%) exceeds limit",
                detail={"var_99": var_99_portfolio, "limit": settings.risk_limit_var_99},
            )
        )
    budget = settings.risk_budget_default
    sigma_p = float(pd.Series(risk_contributions, dtype=float).sum())
    if sigma_p > 1e-18:
        for t, rc in risk_contributions.items():
            if pd.isna(rc):
                continue
            share = float(rc) / sigma_p
            if share > budget * 1.25:
                sigs.append(
                    Signal(
                        type="REDUCE_RISK",
                        priority="HIGH",
                        message=f"Risk share breach on {t}",
                        detail={
                            "ticker": t,
                            "risk_share": share,
                            "budget": budget,
                            "rc_sigma_units": float(rc),
                        },
                    )
                )
            elif share < budget * 0.15:
                sigs.append(
                    Signal(
                        type="INCREASE",
                        priority="MEDIUM",
                        message=f"Low risk share vs budget: {t}",
                        detail={"ticker": t, "risk_share": share, "budget": budget},
                    )
                )
    dev = (weights - target_weights.reindex(weights.index).fillna(0.0)).abs()
    if dev.max() > settings.weight_drift_threshold:
        sigs.append(
            Signal(
                type="REBALANCE",
                priority="MEDIUM",
                message="Weights drifted from target",
                detail={"max_dev": float(dev.max())},
            )
        )
    if tail_multiplier > settings.tail_multiplier_hedge:
        sigs.append(
            Signal(
                type="HEDGE",
                priority="HIGH",
                message="Cornish–Fisher tail multiplier elevated",
                detail={"tail_multiplier": tail_multiplier},
            )
        )
    if avg_pairwise_corr > settings.avg_corr_alert:
        sigs.append(
            Signal(
                type="CORRELATION_ALERT",
                priority="CRITICAL",
                message="Average pairwise correlation high (diversification breakdown)",
                detail={"avg_corr": avg_pairwise_corr},
            )
        )
    if portfolio_drawdown < -settings.drawdown_hard_stop:
        sigs.append(
            Signal(
                type="DRAWDOWN_STOP",
                priority="CRITICAL",
                message="Portfolio drawdown hard stop",
                detail={"drawdown": portfolio_drawdown},
            )
        )
    return sigs


class RebalancingEngine:
    """Rule-based recommendations (not an optimiser)."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def run(
        self,
        weights: pd.Series,
        target_weights: pd.Series,
        var_99_portfolio: float,
        risk_contributions: pd.Series,
        tail_multiplier: float,
        avg_pairwise_corr: float,
        portfolio_drawdown: float,
    ) -> list[dict[str, Any]]:
        sigs = generate_signals(
            self.settings,
            weights=weights,
            target_weights=target_weights,
            var_99_portfolio=var_99_portfolio,
            risk_contributions=risk_contributions,
            tail_multiplier=tail_multiplier,
            avg_pairwise_corr=avg_pairwise_corr,
            portfolio_drawdown=portfolio_drawdown,
        )
        return [
            {
                "type": s.type,
                "priority": s.priority,
                "message": s.message,
                "detail": s.detail,
            }
            for s in sigs
        ]
