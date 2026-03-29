# Research note — regime-aware correlation stress and portfolio control

**Version:** working draft aligned with the engineering brief (March 2026).  
**Audience:** technical reviewers; this note under-claims performance and foregrounds validation design.

---

## 1. Thesis

We test whether **abnormal correlation expansion and instability** (relative to each asset panel’s own history) contain information useful for **portfolio risk control**—signal weighting, exposure scaling, and defensive allocation—compared with **static** baselines (equal weight, inverse volatility, naive momentum).

The hero signal is the **z-score of average pairwise correlation** of log returns over a rolling covariance window; see `alpha/correlation_regime_signal.py` for the frozen definition.

This is **not** marketed as a standalone alpha engine. Primary evidence lives in **conditional forward risk** (volatility, short-horizon drawdown behaviour by signal bucket), **gross vs net** paths under turnover costs, **walk-forward** slices with **past-only** fitted thresholds, and **ablations** that remove one subsystem at a time.

---

## 2. Data

- **Source:** Yahoo Finance via `yfinance`, built by `scripts/build_data_panel.py` into `data/raw/` (per-ticker OHLCV) and `data/processed/` (aligned close panel + `panel_metadata.json`).
- **Policy:** no silent pre-inception history; alignment uses inner dates after limited ffill/bfill; see metadata for `first_valid_date_by_ticker`.
- **Windows:** prefer ~2010–present where coverage is clean. Short 2020s-only samples are labelled **stress-heavy evaluation**, not “all history.”
- **Quality gate:** `data/data_quality.py` writes JSON to `research/outputs/data_quality_report.json` when running `python -m backtest.run` (unless `--no-qc`).

---

## 3. Methodology (summary)

1. **Features & regime:** rule-based and optional HMM features (`regime/regime_state.py`); anomalies via `detection/anomaly.py`.
2. **Hero signal:** correlation z and buckets feed the combiner and decision engine.
3. **Risk:** historical + Monte Carlo + Cornish–Fisher stack (`risk/var.py`); **risk disagreement** flag when HS vs MC 1d 99% VaR diverge beyond a relative threshold (`backtest.risk_disagreement_rel_threshold`).
4. **Portfolio:** regime-conditioned optimizer (`portfolio/optimizer.py`); vol targeting and turnover cap in `portfolio/constraints.py`.
5. **Backtest:** daily bar, no lookahead; proportional costs on turnover; optional **rebalance stride** (`backtest.rebalance_every_bars`); **gross and net** equity logged.

---

## 4. Validation design

| Tool | Location | Role |
|------|----------|------|
| Strategy ladder + placebo | `python -m backtest.run` | Baselines vs full vs random-signal sanity |
| Walk-forward | `python -m backtest.walkforward` (add `--fast` for a shorter MC path + smaller synthetic panel) | Fit corr z-threshold quantiles on train only per fold; OOS metrics on test block |
| Hero predictive tables | `python research/hero_signal_validation.py` | Bucketed forward vol / DD / avg corr |
| Ablations | `python scripts/run_ablations.py` | Remove one component at a time → `ablation_results.csv` |
| Cost sweep | `python -m backtest.run --cost-sweep` | Net performance vs `cost_bps` |

---

## 5. Results (populate from `research/outputs/`)

After you run the pipelines on your machine, paste or summarise:

- `ladder_table.csv` — Sharpe, CAGR, max DD, **gross vs net**, VaR breach rate.
- `hero_signal_validation_buckets.csv` — does **high** correlation stress precede higher forward vol?
- `walkforward_manifest.json` — stability of OOS metrics across folds.
- `ablation_results.csv` — which removals hurt tail or turnover behaviour.

**Honest failures:** document at least one weak subperiod or threshold sensitivity in `research/failure_cases.md` and `docs/failure_analysis.md`.

---

## 6. Limitations and next steps

- yfinance data quirks, splits, and survivorship (ETF inception) remain operational risks; metadata and QC reports are mandatory context.
- Backtest VaR uses sample covariance in-loop vs DCC in live (`docs/backtest_assumptions.md`).
- **Next:** richer failure maps by calendar year and regime; optional eigenvalue-concentration hero metric; historical “similar state” retrieval for the dashboard.

---

## References in-repo

- `docs/methodology.md`, `docs/backtest_assumptions.md`, `docs/model_risk.md`, `docs/ablation_summary.md`.
