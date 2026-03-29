# Ablation summary

Ablation runs compare the **full** engine against configurations with one major component removed. They answer: *what is actually doing the work?*

## How to generate

From `Main code`:

```bash
python scripts/run_ablations.py --synthetic
# or after building real data:
python scripts/run_ablations.py
```

Output: [`research/outputs/ablation_results.csv`](../research/outputs/ablation_results.csv) (created on first run).

## Grid (see `scripts/run_ablations.py`)

| Run | What changes |
|-----|----------------|
| `full_baseline` | No change |
| `no_correlation_signal` | Hero correlation path neutralised |
| `no_regime_gating` | Regime multipliers in `gate_signals` disabled |
| `no_anomaly_gating` | Anomaly multipliers in `gate_signals` disabled |
| `no_vol_target` | Vol targeting step skipped |
| `no_transaction_costs` | Turnover costs set to zero |
| `no_decision_engine` | Decision layer neutral; signals-only scaling |
| `no_turnover_cap` | Turnover cap in `apply_constraints` disabled |

Interpret **CAGR / Sharpe** together with **max drawdown**, **mean turnover**, and **var_breach_rate** from the ladder or full backtest logs—not any single number in isolation.
