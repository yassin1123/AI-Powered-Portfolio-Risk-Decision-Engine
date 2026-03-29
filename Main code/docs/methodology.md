# Methodology

**Central thesis:** test whether **correlation stress / instability** (the hero z-score of average pairwise correlation) improves **risk-aware** portfolio behaviour versus static baselines—see [`docs/RESEARCH_NOTE.md`](RESEARCH_NOTE.md) and `alpha/correlation_regime_signal.py`.

This document ties **research questions**, **layers**, and **evaluation** to the codebase. Implementation status lives in [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md).

**Code (current):** `regime/`, `alpha/` (incl. `correlation_regime_signal.py`), `core/decision/` (`DecisionEngine` — priority policy in [`decision_policy.md`](decision_policy.md)), `portfolio/`, `hedging/`, `backtest/`, `diagnostics/` (contagion, attribution helpers), `risk/evaluation.py`; live wiring in [`pre/pipeline.py`](../pre/pipeline.py).

---

## Research questions (summary)

- **RQ1 — Dynamic covariance:** Does DCC-GARCH (with **Σ = D R D**) improve **1-step and multi-step** risk forecasts vs sample cov and EWMA? Metrics: breach rates, **Kupiec** (unconditional coverage), **Christoffersen** (independence), vol forecast **RMSE/MAE**, calibration plots.
- **RQ2 — Regime & anomaly:** Do regime labels and anomaly concurrence **predict** shorter-horizon drawdowns, vol spikes, or correlation jumps? Metrics: precision/recall tables, hit rates by regime, portfolio path with/without gating.
- **RQ3 — Sizing & gating:** Does **vol targeting** + **regime-conditioned signal gating** improve **OOS Sharpe**, Sortino, and tail loss vs equal-weight and inverse-vol baselines?

---

## Eight-layer logical architecture

| Layer | Responsibility | Evidence / outputs |
|-------|------------------|-------------------|
| L1 Data | Clean ingestion, calendars, staleness, bias notes | Data QA logs, missing-data policy |
| L2 Features | Returns, features for risk & signals | Feature store, reproducible transforms |
| L3 Risk forecasting | VaR, vol, cov forecasts | **Realized vs forecast**, breach stats, tests |
| L4 Regime + anomaly | Calm / stress / transition + detectors | Regime probs, anomaly timeline, duration |
| L5 Alpha | Momentum, MR, carry, X-section; optional residual/dispersion | IC, turnover, regime-stratified PnL |
| L6 Portfolio | Vol target, constraints, costs, **hedges** | Weights, gross/net, hedge notionals |
| L7 Backtest | Walk-forward, ablations, benchmarks | Tables: Sharpe, DD, turnover, crisis windows |
| L8 Dashboard | Monitoring + research controls | Regime, calibration, attribution, explain cards |

---

## Target package layout (incremental migration)

Current packages live at repo root (`data/`, `risk/`, …). **Target** consolidation (optional refactor):

```text
core/   # or keep flat packages; roadmap tracks either
  data/
  features/
  risk/
  regime/          # rule_based.py, hmm_regime.py, regime_state.py
  anomaly/         # migrate from detection/
  alpha/           # momentum, mean_reversion, carry, cross_sectional, combiner
  portfolio/       # target_weights, risk_targeting, constraints, optimizer
  hedging/         # beta_hedge, tail_hedge, overlay rules
  stress/          # extend stress/scenarios
  backtest/        # engine, book, pnl, benchmark, evaluation
  diagnostics/     # attribution, factor exposure
  ui/              # dashboard wrappers
```

Minimum **new** modules for “fund credibility” (priority order):

1. `regime/` — rule baseline + optional **Gaussian HMM** on (vol, avg corr, return).
2. `alpha/` — 3–4 signal families + **regime gating** table.
3. `portfolio/` — vol targeting, caps, simple **inverse-vol** and **signal-weighted** weights.
4. `backtest/` — daily loop, **no lookahead**, costs, baselines (EW, inv-vol, risk-only, full).
5. `research/` — ablation configs + CSV outputs; walk-forward splits.

---

## Ablations (minimum set)

| ID | Variant |
|----|---------|
| A1 | No DCC — sample cov only |
| A2 | No anomaly layer |
| A3 | No regime filter |
| A4 | No vol targeting |
| A5 | No transaction costs |
| A6 | No reverse-stress / tail action |
| A7 | No signal gating |
| A8 | Full system |

Report: Sharpe, max DD, VaR breach rate, turnover, crisis-window return, tail loss.

---

## Statistical evaluation checklist

- VaR **monotonicity** (99% ≥ 95%) and CVaR ≥ VaR (tests).
- **Kupiec / Christoffersen** on VaR exceedances (when backtest exists).
- Vol forecast **RMSE/MAE** vs realized (e.g. realized from squared returns or Parkinson if extended).
- **Mahalanobis / χ²** sanity (already in tests).
- **Walk-forward:** train window → test window → roll; config: expanding vs rolling.

---

## Predictive value of anomalies

For each detector flag (and **conjunction count**), measure:

- Future 1d / 5d / 10d return and drawdown
- Future 5d realized vol
- Future correlation spike (e.g. avg pairwise corr change)

Output: table **Detector × horizon × metric** (+ precision/recall if binary “bad week” defined).

---

## Factor / exposure (macro-aware)

Rolling **β** to SPY, rates proxy (TLT/IEF), vol (^VIX), USD (UUP), gold (GLD) where tickers exist—report **rolling loadings** and **regime-conditional** averages.

---

## References (implementation)

- Original engineering brief: Portfolio Risk Engine v1.0 (internal).
- Roadmap: [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md).
- Limitations: [`limitations.md`](limitations.md).
