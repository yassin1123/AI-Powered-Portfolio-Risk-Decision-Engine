# Implementation roadmap — 8-layer / fund-grade upgrade

Phased plan aligned with README research questions. **Tier 1** delivers the largest credibility jump.

---

## Tier 1 (mandatory for “9.5 story”)

| # | Deliverable | Suggested location | Notes |
|---|-------------|-------------------|--------|
| 1 | **Regime detection** (3 states: calm / stress / transition) | `regime/rule_based.py`, `regime/regime_state.py` | Start rules (vol + avg corr + drawdown + anomaly count); add `regime/hmm_regime.py` optional |
| 2 | **Alpha layer** (≥4 families) | `alpha/momentum.py`, `mean_reversion.py`, `carry.py`, `cross_sectional.py`, `signal_combiner.py` | Z-scored, documented; one “smart” signal e.g. residual momentum or anomaly-conditioned |
| 3 | **Regime-conditioned gating** | `alpha/gating.py` or inside combiner | Table: calm vs transition vs stress → thresholds, gross cap, allowed styles |
| 4 | **Portfolio construction** | `portfolio/risk_targeting.py`, `target_weights.py`, `constraints.py` | Vol target %, max weight, turnover cap, min cash; **inverse-vol** + **signal × inv-vol** baselines |
| 5 | **Backtest engine** | `backtest/engine.py`, `portfolio_book.py`, `pnl.py`, `fills.py` | Daily bars; **no lookahead**; explicit fill rule; proportional costs |
| 6 | **Baselines** | `backtest/benchmark.py` | Buy-hold EW, inv-vol, risk-only (no alpha), alpha-no-gating, **full** |
| 7 | **Walk-forward** | `backtest/walkforward.py` | Rolling or expanding; YAML config |
| 8 | **Ablations A1–A8** | `research/ablation_configs/` + script or notebook | CSV table: Sharpe, maxDD, breach%, turnover, crisis return |
| 9 | **Results artifacts** | `research/figures/`, `research/ablation_results.csv` | Plots: equity, DD, calibration |

**Tests to add:** `tests/test_no_lookahead.py`, `tests/test_backtest_pnl.py`, `tests/test_constraints.py`, `tests/test_regime_gating.py`.

---

## Tier 2 (major boost)

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Risk **forecast evaluation** — Kupiec, Christoffersen, vol RMSE | `risk/evaluation.py` or `backtest/evaluation.py` |
| 2 | **Anomaly predictive value** | `research/anomaly_predictive.py` |
| 3 | **Hedging** rules | `hedging/hedge_rules.py`, `beta_hedge.py`, `tail_hedge.py` |
| 4 | Extended **stress** library | extend `stress/` — historical replay tags, liquidity, contagion |
| 5 | **PDF / narrative report** | `docs/research_report.pdf` (export from notebook or ReportLab) |
| 6 | Dashboard **panels**: regime prob, forecast vs realized, signal heatmap | `dashboard/app.py` + snapshot fields |

---

## Tier 3 (elite polish)

- **HMM** regime (`regime/hmm_regime.py`) alongside rule baseline.
- **Experiment registry** — `experiments/registry.csv` + `run_configs/` (git hash, seed, config path).
- **Correlation contagion** index (concentration / cluster stress).
- **Regime-conditioned optimizer** (e.g. stress → min-var tilt).
- **Forecast combination** (ensemble: sample / EWMA / DCC).
- **Diagnostics** package: marginal risk contribution, factor rolling β by regime.
- **Performance benchmarks** — extend `pytest-benchmark` matrix in CI doc.

---

## Standout combo (recommended)

Implement **regime-conditioned portfolio** (Tier 1 gating + sizing) + **anomaly-conditioned signal suppression** (Tier 2): *signals allowed only when anomaly intensity &lt; threshold or stress-approved styles.*

---

## Ideal repo tree (end state)

```text
README.md  config.yaml  pyproject.toml  main.py  .env.example
data/  features/  risk/  detection/  signals/  stress/  reports/  core/  pre/  dashboard/  tests/
regime/  alpha/  portfolio/  hedging/  backtest/  diagnostics/   # new or merged
docs/
  methodology.md
  limitations.md
  IMPLEMENTATION_ROADMAP.md
  research_report.pdf   # generated
  architecture.png      # optional export from mermaid
research/
  notebooks/
  ablation_results.csv
  walkforward_results.csv
  figures/
experiments/
  registry.csv
  run_configs/
  outputs/
```

**Migration note:** Packages can stay at top level initially; `core/` consolidation is a refactor pass once imports are centralized (e.g. single `pre.pipeline` facade).

---

## Experiment registry columns (minimum)

`run_id`, `git_hash`, `config_path`, `seed`, `start`, `end`, `universe`, `sharpe`, `max_dd`, `var_breach_rate`, `turnover`, `notes`

---

## Dashboard “explainability” cards (L8)

For each snapshot cycle (extend `DashboardSnapshot`):

- Top 3 **risk** drivers (contribution / vol / corr)
- Top 3 **signal** contributors
- Gross / net / cash
- **Regime** + duration
- Last **de-risk** trigger (module + rule)

---

## What not to build early

- Heavy DL for prices without OOS protocol  
- Huge universe before backtest is stable  
- UI animations over evaluation rigor  

**Win condition:** rigor, clean ablations, reproducible configs, honest limitations.
