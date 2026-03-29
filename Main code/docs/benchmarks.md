# Engineering benchmarks

Timing targets (document on your machine):

| Stage | Notes |
|-------|--------|
| DCC-GARCH refit | `pytest tests --benchmark-only` (extend markers) |
| Signal + decision stack | Per `cycle_ms` in `DashboardSnapshot` |
| Backtest 400d synthetic | `python -m backtest.run` wall time |

**Caching:** Cholesky reuse in `risk.garch.cholesky_cached`; feature history capped in `PipelineState`.
