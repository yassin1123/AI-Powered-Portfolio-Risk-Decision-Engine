# Key findings (populate after runs)

Run `python -m backtest.run` from `Main code` to generate `research/outputs/ladder_table.csv`, lead–lag and decision breakdown CSVs, `killer_overlay.png` (with dev extras), and to refresh the **auto-generated** section below. You can still edit narrative above or below that block freely.

Template statements (replace placeholders):

1. **Correlation regime signal:** Average pairwise correlation **z-scores above 1.5** coincided with **[X]%** higher subsequent **5-day realized volatility** vs baseline periods (or: co-move with drawdowns — report honestly).
2. **Anomaly concurrence:** Multi-layer anomaly spikes **predicted** short-horizon **vol expansion**; **gating** reduced **[Y]** (e.g. tail loss metric) vs ungated signals in the synthetic ladder.
3. **Regime transitions:** Large moves in **avg correlation** clustered within **±Z days** of **regime label changes** (see `research/regime_transition_study.py` output).
4. **Vol targeting:** Realized vol **tracked** target **10%** within **[±bp]** but **CAGR** was **[lower/higher]** than unlevered EW in the placebo-rich sample.
5. **Placebo:** **Random-signal** Sharpe vs full system — if random is **worse**, the stack is **not** pure noise; if full is **still negative**, separate **alpha** from **risk architecture** in the write-up.

**Stress vs reality:** Compare worst **scenario PnL** from `stress` to worst **realized** windows using `research/stress_vs_reality.py`.

---

## How to read the ladder (honest framing)

The auto-generated numbers below can show **negative Sharpe and negative CAGR for every row**. That does **not** mean the project failed if your goal is **risk analysis and control**, not **“profitable strategy.”**

| Observation | Takeaway |
|---------------|----------|
| **Signals only** looks bad | **Alpha** (as wired today) is weak or **turnover** eats edge—treat as a **signal problem**, not a VaR/GARCH bug. |
| **Correlation signal only** ≈ **vol targeting only** | Correlation regime is **risk conditioning**, not a standalone return source. |
| **Random / placebo** is worst | Good: the system is **not** equivalent to **random** rebalancing. |
| **Full system** below **Baseline** on return | **Risk controls** can **cut exposure** when rules fire; if signals do not pay, **returns** can lag while **risk metrics** change. Report both. |

For a **trading** story you still need **positive edge** and costs data; for a **risk engineering** story the ladder supports **ablations and placebo discipline** even when no row is “green” on Sharpe.

<!-- AUTO-GENERATED START -->

### Quantitative snapshot (auto-generated)

**Five-strategy ladder (synthetic sample):**

- **Baseline:** Sharpe **-1.1964**, max DD **-0.119128**, mean turnover **0.0**
- **Vol targeting only:** Sharpe **-1.2279**, max DD **-0.119044**, mean turnover **0.002675**
- **Signals only:** Sharpe **-2.2329**, max DD **-0.186071**, mean turnover **0.236617**
- **Correlation signal only:** Sharpe **-1.2279**, max DD **-0.119044**, mean turnover **0.002675**
- **Full system:** Sharpe **-2.2329**, max DD **-0.186071**, mean turnover **0.236617**
- **placebo_random_signals:** Sharpe **-3.5181**, max DD **-0.209486**, mean turnover **0.49205**

**Lead–lag (corr_z vs forward 5-bar outcomes):**

- **corr_z > 1.5** (n=50): avg forward 5d max drawdown **-0.007789567875187171**, avg ann. vol proxy **0.06040108987616369**, day breach rate **0.0**
- **|corr_z| <= 1.5 (baseline)** (n=191): avg forward 5d max drawdown **-0.00961797246177563**, avg ann. vol proxy **0.06410488835752332**, day breach rate **0.0**

**Decision mix (`decision_priority`):**

- **normal:** **46.0967%** (124 bars)
- **anomaly_suppress:** **29.368%** (79 bars)
- **corr_crisis:** **18.9591%** (51 bars)
- **transition:** **3.3457%** (9 bars)
- **diversification_regime:** **2.2305%** (6 bars)

Figure: `research/figures/killer_overlay.png` (after matplotlib install and backtest run).

<!-- AUTO-GENERATED END -->
