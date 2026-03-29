# Decision policy (priority hierarchy)

The decision engine assigns **one** primary `decision_priority` per bar by walking the list below; the first matching condition wins. Parameters (`exposure_scale`, `override_signals`, `suppress_non_defensive`, `activate_hedge`) come **only** from that tier. Additional boolean facts are listed in `secondary_reasons` / `reason_codes` for audit (excluding redundancy with the primary).

## Ordered ladder

1. **`stress_corr_override`** — Regime is `STRESSED` and `corr_z > stress_corr_z_threshold` (config). Full defensive path: override signals, suppress non-defensive, hedge, low exposure cap.
2. **`corr_crisis`** — `corr_z > z_high` (correlation spike) if not already captured above.
3. **`anomaly_suppress`** — `anomaly_count >= anomaly_suppress_count`.
4. **`stressed_regime`** — `STRESSED` without a higher-priority match.
5. **`transition`** — `TRANSITION` taper (reduced exposure scale).
6. **`var_breach_risk`** — Portfolio `var_99` above the risk limit (when not subsumed earlier).
7. **`diversification_regime`** — `corr_z < z_low` (allow modest tilt up within caps).
8. **`normal`** — Default path; may still activate hedge from correlation bucket rules.

## Backtest-only neutral tier

- **`signals_only_neutral`** — Used when the backtest runs **Signals only** or **Correlation signal only**: no decision-engine scaling; `apply_decision_to_signals` skips extra stressed clipping for this id.

Implementation: `core/decision/decision_engine.py`.

## Deterministic rule IDs (audit / UI)

Each bar, the engine exposes `winning_rule_id` and a full `condition_flags` map (booleans) plus `driver_lines` (score mass for traceability). IDs align with the ladder above:

| Rule ID | Priority when selected |
|---------|-------------------------|
| `R1_STRESS_HIGH_CORR` | `stress_corr_override` |
| `R2_CORR_CRISIS_Z` | `corr_crisis` |
| `R3_ANOMALY_SUPPRESS` | `anomaly_suppress` |
| `R4_STRESSED_REGIME` | `stressed_regime` |
| `R5_TRANSITION_REGIME` | `transition` |
| `R6_VAR_LIMIT_BREACH` | `var_breach_risk` |
| `R7_DIVERSIFICATION_REGIME` | `diversification_regime` |
| `R8_DEFAULT_NORMAL` | `normal` |

Trace builder: `core/decision/decision_trace.py`. Dashboard: **SYSTEM SIGNAL** + **Decision trace** panels.
