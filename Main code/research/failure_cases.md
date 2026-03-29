# Failure notebook (live document)

Use this file to record **honest** cases where the engine behaved poorly: false stress alarms, missed drawdowns, threshold instability out-of-sample, or cost sensitivity that kills net returns.

## Template

| Date (UTC) | Context | What happened | Hypothesis | Follow-up |
|------------|---------|---------------|------------|-----------|
| YYYY-MM-DD | e.g. walk-forward fold 3, core universe | e.g. over-defended in calm tape | threshold too tight on train quantiles | log sensitivity run |

_Add rows as you analyse `research/outputs/` and live simulation._

## Known structural limitations

- Short macro eras (e.g. 2020s only) exaggerate correlation clustering; widen the sample before claiming robustness.
- VaR breach metric in backtest is a **rough** log-return vs VaR comparison for diagnostics, not regulatory backtesting.
