# Experiments

Track **reproducible runs**: config, seed, git hash, metrics.

Suggested files:

| File | Purpose |
|------|---------|
| `registry.csv` | One row per run (see columns in `docs/IMPLEMENTATION_ROADMAP.md`) |
| `run_configs/` | Frozen YAML/JSON copies per run id |
| `outputs/` | Logs, equity curves, tables (large files → gitignore + artifact store) |

Workflow:

1. Copy `config.yaml` → `run_configs/<run_id>.yaml` and edit flags (ablation).
2. Run backtest / pipeline; append row to `registry.csv`.
3. Attach key plots to `research/figures/` and reference run_id in caption.
