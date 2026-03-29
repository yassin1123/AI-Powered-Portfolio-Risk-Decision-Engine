# Results summary

## Ladder table (regenerated)

Run from `Main code`:

```bash
python -m backtest.run
```

Output: [`research/outputs/ladder_table.csv`](../research/outputs/ladder_table.csv)

The CSV includes **five** named strategies plus placebo:

| Strategy (CSV `Strategy` column) | Mode |
|----------------------------------|------|
| Baseline | `baseline` |
| Vol targeting only | `vol_target_only` |
| Signals only | `signals_only` |
| Correlation signal only | `corr_signal_only` |
| Full system | `full` |
| placebo_random_signals | `placebo_random` |

Columns include at least: `Strategy`, `Sharpe`, `max_dd`, `mean_turnover`, `var_breach_rate`, `cagr`.

## Interpreting the ladder when everything loses money

Negative Sharpe and negative CAGR across **Baseline**, **Full system**, and **Random** are **common** in short samples, synthetic paths, or when the **equity risk premium** is not in your favor. That is **not** by itself a bug in the risk engine.

**How to frame it**

- **This codebase is not** “I built a money-printing strategy.” **It is** “I built machinery to **measure and govern** portfolio risk (VaR, stress, regime, correlation dynamics, decisions).”
- **Signals only** worse than baseline → the **alpha layer** (as implemented) is **weak or costly** (turnover); fix signals or costs separately from the risk stack.
- **Correlation signal only** matching **vol targeting only** → the correlation overlay is **not** a return engine by itself; it is **risk conditioning**.
- **Random (placebo) ≈ worst** → sanity check: the stack is **not** indistinguishable from noise; the ordering still carries information.
- **Full system worse than baseline on return metrics** → with **weak alpha**, **risk controls** (gating, de-risking) can **lower CAGR** while changing **drawdowns, VaR breaches, or tail mix**. That is a real economic trade-off to report, not hide.

Use the ladder to compare **components** and **placebos**, not to assert profitability without a separate **alpha** thesis and dataset.

For **why generic sleeves underperform** and how to improve alpha **without** stacking indicators (regime-gated momentum hypothesis, universe splits, turnover), see [`alpha_and_risk.md`](alpha_and_risk.md).

After a normal run (not `--placebo` / `--no-extras`), the same command also refreshes:

- [`research/outputs/leadlag_summary.csv`](../research/outputs/leadlag_summary.csv) — forward 5-bar outcomes vs `corr_z`
- [`research/outputs/decision_breakdown.csv`](../research/outputs/decision_breakdown.csv) — share of bars by `decision_priority`
- [`research/figures/killer_overlay.png`](../research/figures/killer_overlay.png) — drawdown, `corr_z`, regime (requires `matplotlib` in dev extras)

Use `python -m backtest.run --leadlag` to regenerate only the lead–lag CSV from an existing `decision_log.csv`.

## Figures (target set)

1. Cumulative returns (`equity_curve.csv`)
2. Drawdown + correlation + regime — `killer_overlay.png`
3. Rolling Sharpe
4. VaR vs realized loss
5. Regime overlay on PnL

Place outputs under `research/figures/` (generate via notebook or script).

## Statistical evaluation

- **Kupiec / Christoffersen:** `risk.evaluation` on stored VaR and return series.
- **Key narrative:** [`research/key_findings.md`](../research/key_findings.md) (includes an auto-generated quantitative block).

*Last updated: after each `python -m backtest.run`.*
