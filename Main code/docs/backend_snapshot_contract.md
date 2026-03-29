# Backend snapshot contract (`elite_snapshot` / `live_snapshot_v1`)

This document describes the JSON tree attached to each risk cycle as `report["elite_snapshot"]` from `pre.pipeline.run_risk_cycle`. The current Dash UI does not render these fields; they are a **stable contract** for a future UI or external API.

## Top-level shape

- **`schema_version`**: string, `"live_snapshot_v1"`.
- **`meta`**: `SnapshotMeta` — UTC timestamps, `cycle`, `cycle_ms`, `universe_profile`, `data_quality_warnings`.
- **`market_state`**: unified regime / correlation / vol / anomaly / drawdown read (see `features.state_builder.build_market_state`).
- **`decision`**: enriched decision snapshot — priorities, `conditions_met`, top pre/gate/post signal dicts, `signal_adjustment_ratio_sample`, exposure fields.
- **`narrative`**: rule-based `headline`, `summary`, `why_lines`, `action_line` from `narrative.engine.build_narrative` (no LLM).
- **`risk`**: `vs_target` (forecast vs realized proxy vs target), `tail` (VaR / CVaR / breach stats), optional `decision_trace` echo.
- **`portfolio`**: `allocation_delta`, `weights_current`, `weights_target`.
- **`recent_changes`**: 1/5/20-bar style deltas from an in-memory ring (`context.recent_changes`).
- **`timeline`**: regime segments + `transition_stats` from `core.state_history`.
- **`analogs`**: past-only kNN neighbors (`context.analogs.find_similar_states`).
- **`research_links`**: CLI strings and module paths for walk-forward, ablations, failure analysis, by-regime metrics, scenario helpers.

## Versioning

Bump `schema_version` and this file together when removing or renaming fields. Prefer additive changes (new optional keys) within `live_snapshot_v1` when possible.

## Related modules

| Area | Module |
|------|--------|
| Assembly | `api.live_snapshot.build_live_snapshot_v1` |
| Types | `core.schemas` |
| Shocks / counterfactuals | `scenario.shocks` |
| Failure tagging | `research.failure_analysis` |
| Regime slices | `research.by_regime_metrics` |
| Programmatic backtest / WF | `research.backend_entrypoints` |

## Canonical research CLIs

- Walk-forward: `python -m backtest.walkforward`
- Ablations grid: `python scripts/run_ablations.py`
- Full backtest export (decision log, equity): `python -m backtest.run`
