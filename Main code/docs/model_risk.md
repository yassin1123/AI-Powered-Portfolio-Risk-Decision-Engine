# Model risk

## Dynamic covariance (DCC-GARCH)

- Estimation can **fail or be unstable** with small samples, missing data, or ill-conditioned matrices.
- Parameters are **not** structural constants; **regime shifts** can invalidate short estimation windows.

## Small-sample bias

- Rolling correlation, `corr_z`, and regime features are **noisy** when windows are short relative to the number of assets.

## Parameter sensitivity

- Thresholds (`corr_z` high/low, anomaly counts, vol target) materially affect turnover and drawdowns. **Sensitivity analysis** (ablations / ladder) is required before strong claims.

## Signal overfitting

- Combining multiple sleeves increases **data mining** risk. The **placebo** and **walk-forward** protocols mitigate but do not eliminate it.

## Regime misclassification

- Rule and cluster-based regimes are **proxies**. Labels can lag true shifts; **transition** states are especially ambiguous.

## Structural breaks

- Crisis periods may violate stationarity assumptions underlying GARCH/DCC and correlation z-scores.

See also [`failure_analysis.md`](failure_analysis.md) and [`limitations.md`](limitations.md).
