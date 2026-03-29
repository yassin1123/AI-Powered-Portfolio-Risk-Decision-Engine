# Backtest assumptions

## Execution timing

- Signals and portfolio weights at end of bar **t** use only information available through **t** (returns and prices through **t**).
- Portfolio return applied for the interval **t → t+1** uses weights chosen at **t** (next-bar / close-to-close alignment).
- Live `main.py` loop uses the same decision stack as `backtest.engine` for consistency of rules (not identical data path).

## VaR fed to the decision engine (backtest vs live)

- **Live pipeline** (`pre/pipeline.py`): 1-day **99% MC VaR** is `var_res.mc_var[(0.99, 1)]` from `risk.var.compute_full_var`, using **DCC-GARCH** covariance `sigma_t`, aligned 1d/10d return windows, and **current** book weights.
- **Backtest** (`backtest/engine.py`): uses the **same** `compute_full_var` function (historical + Monte Carlo + Cornish–Fisher tail step) so the **estimator matches production**, but with (a) **rolling sample covariance** `sigma` over the same tail window as the optimizer, (b) weights = **`prior_w`** (holdings at **t** before rebalance), (c) a separate MC draw count **`backtest.var_mc_sims`** (default 1500 in `config.yaml`) to keep runtime reasonable—live uses **`mc_sims`** (often 10k).
- If the aligned 1d/10d history is too short (`< 30` rows), the backtest falls back to **parametric normal** 99% VaR: \(z_{0.99} \times \sigma_p\) with \(\sigma_p = \sqrt{w'\Sigma w}\) from the same sample \(\Sigma\); `decision_log.var_99_method` is `parametric_normal_99` vs `mc_full_var`.

**Interview answer:** The backtest does **not** use a fixed 2% VaR constant; it uses the shared VaR module with explicit methodology flags in the log. It is still **not** identical to live because covariance is **sample** in-loop rather than **DCC**-refit.

## Portfolio optimizer (CALM / TRANSITION)

- `_signal_inv_vol` in `portfolio/optimizer.py` applies **inverse-vol scaling to non-negative signals only** (`raw.clip(lower=0.0)`). Negative alpha scores do not create short positions; this matches a **long-only** book for the default universe. **STRESSED** still uses a long-only minimum-variance sleeve.

## Cost model

- Proportional cost on one-way turnover: `turnover * cost_bps / 10000` per rebalance (see `portfolio.transaction_costs` and `config.yaml` `portfolio.cost_bps`).
- No explicit market impact or bid–ask model unless extended.

## Data frequency

- Daily business-day bars (synthetic or fetched). Staleness and gaps: see `docs/limitations.md` and optional `data/quality` hooks.

## Universe

- Defined by `data/universe.py` / fetcher. Survivorship and lookahead: documented in `limitations.md`.

## Liquidity

- Implicit full execution at close; no delayed fill or capacity constraint in the baseline engine.

## Placebo validation

- Run `python -m backtest.run --placebo` (with `--synthetic` implied in the default CLI path). **Random signals** through the same constraint and cost shell typically produce **weak or negative risk-adjusted performance**, which supports that reported results are **not** purely structural artifacts.
