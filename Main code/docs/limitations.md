# Limitations

Explicit limitations **increase trust** with reviewers. This engine is serious but not institutional data or execution.

See also: [`backtest_assumptions.md`](backtest_assumptions.md), [`model_risk.md`](model_risk.md), [`failure_analysis.md`](failure_analysis.md).

## Data

- **yfinance** (and similar) provide **daily** adjusted series, not tick-level institutional feeds; latency and corporate-action handling differ from Bloomberg/Refinitiv.
- **ETF proxies** are not futures or OTC markets; roll, basis, and liquidity differ from prop desks.
- **Survivorship** and **selection bias** must be stated if the universe changes over time.
- **Timezone / calendar** alignment across asset classes requires careful indexing (sessions differ).

## Models

- **GARCH/DCC** can be **slow or numerically fragile** for large universes; regularization and subset universes may be required.
- **DCC** and **Gaussian MC-VaR** understate joint tails vs true data-generating process.
- **10d MC scaling** (e.g. `10 × Σ` for i.i.d. log-returns) is an approximation; realized 10d dependence may differ.
- **Regime** labels (rules or HMM) are **probabilistic**, not ground truth.

## Backtest & signals

- **Execution** is simplified: assumptions on **next-bar fills**, **slippage**, and **capacity** must be stated.
- **Alpha** modules use **standard academic factors**; they are not proprietary edge and must be **evaluated OOS**, not cherry-picked IS.
- **Crowding** and **decay** of published patterns are not fully modeled.

## System

- **Live** mode is simulation/replay unless wired to paid feeds; sub-second latency claims need real tick infrastructure.
- **Dashboard** is a research/operator UI, not a certified risk system of record.

## Forward work

- Liquidity-adjusted VaR, execution microstructure, richer factor models, and formal model risk governance are natural extensions.
