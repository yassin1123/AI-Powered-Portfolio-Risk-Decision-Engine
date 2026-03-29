# Failure analysis — when the system underperforms

For dated, run-specific incidents, add rows to [`research/failure_cases.md`](../research/failure_cases.md).

## Low-volatility, sideways markets

- **Momentum** and **cross-sectional** sleeves may churn; costs drag **Sharpe**.
- **Vol targeting** scales exposure up in calm periods, which can **cap upside** if a slow grind persists without realized vol.

## High-noise regimes

- **Correlation z-score** can oscillate around thresholds → **whipsaw** in hedging and de-risk rules.

## Turnover constraints

- Tight **turnover caps** prevent full alignment with signals → **lagged** response to correlation spikes.

## DCC instability

- With **few observations** or **many assets**, covariance estimates degrade; optimizer and VaR can be misleading until the model falls back to simpler cov.

## Anomaly false positives

- Elevated **anomaly counts** may **suppress** alpha when the stack is overly sensitive, reducing participation in recoveries.

Use `research/outputs/` decision logs and `research/key_findings.md` to quantify these effects on your data.
