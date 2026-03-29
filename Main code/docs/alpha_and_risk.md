# Alpha vs risk: what this project is asking

## Two different questions

**Risk (what this stack is built to answer):**  
“When is risk high? How should exposure and hedges respond?”

**Alpha (a separate question):**  
“What should outperform, under what conditions, and why?”

You do **not** fix weak alpha by stacking more generic indicators. You fix it either by **reframing the project** or by **adding one testable alpha hypothesis** tied to structure you already measure.

---

## Why the current sleeves look “weak”

The default combine path uses **momentum, mean reversion, cross-sectional, carry, and correlation tilts**. Those are reasonable **building blocks** or **baseline sleeves**, not by themselves a sharp economic story. **Correlation gating** is doing useful **risk state** work; it is not guaranteed to be a **return** engine.

**Honest summary:** alpha reads weak because the signals are **mostly generic** and the **universe is mixed** (equities, rates, vol, commodities, crypto in one cross-section). That is often **good for risk**, **hard for pure signal**.

---

## Option 1 — Keep this a risk system (clean path)

Position the work as a **portfolio risk and decision engine**, not an alpha factory. Then **mediocre raw Sharpe** in ablations is **not a failure**; it shows you separate **signal quality** from **risk control**. That is **defensible** for internships and research write-ups.

**Strong closing line (use in interviews):**  
“I found that risk and state-detection were much stronger than the raw alpha sleeves. Rather than pretending otherwise, I studied how **weak alpha interacts with robust risk control** and how **regime-aware gating** can improve **signal discipline**.”

---

## Option 2 — Add one real alpha hypothesis

Do **not** add five new indicators. Add **one** hypothesis, test it **properly**, compare to **always-on** and to **current full system**.

### Route A — Regime-conditioned cross-sectional momentum (best fit here)

**Idea:** momentum is not equally reliable in every **market structure**. In **calm**, **dispersed** conditions it often behaves better; when **correlation spikes** or **regime is stressed**, momentum can fail.

**Hypothesis:** when **regime is CALM** and **corr_z** is below a threshold, **cross-sectional momentum** (e.g. **60d return / 60d vol** per name) works **better** than in **TRANSITION** / **STRESS** or when **corr_z** is extreme.

**Implementation sketch:**

- Score: `score_i = return_60d_i / vol_60d_i` (or similar risk-adjusted rank).
- **Activate** only when: `regime == CALM`, `corr_z < 1.0`, `anomaly_count` below a cutoff.
- **Scale down or off** when correlation **crisis** / **stress** flags fire.
- Long-only: long **top bucket**; optional long-short if you extend the book.

**Why it’s stronger:** the signal becomes **“momentum conditional on market structure”**, not **“always-on momentum”**.

### Route B — Mean reversion after correlation spikes

**Hypothesis:** after **sharp correlation spikes**, some names **overshoot** vs the basket and **partially mean revert** over **3–5 days**.

**Sketch:** when **corr_z** is very high, flag **worst short-horizon relative** names; after **anomaly / transition** triggers, test **forward** mean reversion. **Not guaranteed** — but it is **tied to the project pillar** (correlation dynamics).

### Route C — Relative value **within** sleeves

**Idea:** one giant mixed universe blurs comparable stories. **Within-sleeve** cross-section (e.g. equity vs equity, rates vs rates) is often cleaner.

---

## What probably drags results today

1. **Mixed universe** — fine for risk; noisy for a single alpha engine. **Mitigation:** for alpha **experiments**, split **equity / macro ETF / vol / crypto** sleeves and test **inside** each.
2. **High turnover** — costs eat **weak** edge. **Mitigation:** minimum **signal change** to trade, **holding buffers**, **stronger cost penalty**, **less frequent** rebalance (e.g. **weekly** for that test).
3. **Very short horizons** — noise and costs dominate. **Mitigation:** compare **20d / 60d / 120d** explicitly in write-ups.
4. **Correlation as risk, not return** — use it to **gate** alpha, not to **replace** a return thesis.

---

## Recommended minimal experiment (if you pursue Option 2)

1. Restrict **alpha testing** to **equities + broad liquid ETFs** (drop vol ETPs / crypto from the **alpha** sleeve for that run; `universe_profile: core` already helps the live book).
2. One signal: **60d return / 60d vol**, cross-sectional rank.
3. **Gate:** only on when **CALM**, **corr_z &lt; 1.0**, **anomalies** below threshold.
4. Rebalance **weekly** (not daily) for that experiment.
5. **Buffer:** only change weights if **rank / weight** change exceeds a floor.
6. **Compare:** always-on momentum vs **regime-gated** vs **current full** system — report **Sharpe**, **turnover**, **behaviour in calm vs stress**.

**Success is not** “huge Sharpe”. **Success is:** less bad than before, **lower turnover**, **clearer story** of **where** the sleeve works and **where** it dies.

---

## Bottom line

You may **not** fully “fix alpha” in one project cycle. A **strong** outcome is still: **honest ablations**, **clear separation** of risk vs alpha, and **one** well-motivated hypothesis if you choose Option 2.

See also: [`results_summary.md`](results_summary.md) (ladder interpretation), [`methodology.md`](methodology.md), [`limitations.md`](limitations.md).
