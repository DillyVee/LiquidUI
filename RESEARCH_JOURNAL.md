# LiquidUI Research Journal

Running log of the quantitative research loop applied to this codebase.
Objective: **maximize the probability that strategies produced by this system
remain profitable on unseen data** — not backtest performance. Priorities:
robustness > generalization > risk-adjusted return > drawdown > simplicity >
raw return. Failed experiments are recorded and never deleted.

---

## Iteration 1 — 2026-07-17

### 1. Understand: system map

Read every module before touching anything. Baseline: 138/138 tests pass.

**Data pipeline** (`data/loader.py`): Yahoo Finance, auto-adjusted OHLCV.
Daily 10y, hourly ≤729d, 5-min ≤59d. Quality checks: non-positive prices,
duplicate timestamps, >50% single-bar moves. Timezone-naive timestamps.

**Signals** (`signals/indicators.py`, `signals/engine.py`): 10 causal
indicators behind one interface; each takes two periods (p1, p2). Unbounded
indicators are rank-normalized to a causal rolling percentile in [0,100];
bounded (RSI/stoch/aroon) keep native levels. Entry = ALL legs below their
Entry threshold on ALL timeframes AND calendar cycle ON; exit = ANY leg above
its Exit threshold OR cycle OFF. Warm-up bars (indicator lookback + rank
window) are masked from entries. One shared `evaluate_combo` is used by both
backtest and live paths — good design, prevents semantic divergence.

**Time cycle**: per timeframe, a calendar-anchored ON/OFF square wave
(`On`, `Off`, `Start`; anchor = time since epoch). Search ranges up to
On∈(1,250), Off∈(0,250), Start∈(0,500).

**Backtest** (`optimization/optimizer.py::simulate_multi_tf`): long-only,
single position, next-bar-open fills, % costs on both sides + optional fixed
commission, fixed fraction of equity per trade (`position_size`).
Coarse→fine timeframe mapping lags one full coarse bar (no lookahead).
Mark-to-market at bar opens.

**Optimization**: Optuna TPE, sequential coordinate descent per timeframe
(cycle phase → indicator phase), user-selectable objective (Sharpe, Sortino,
Calmar, total return, PF, win rate, expectancy) scored on the FULL loaded
dataset, scaled by a trade-count confidence ramp (0 below 10 trades, full
trust at 30+).

**Anti-overfit stack** (`optimization/validation.py`, `psr_composite.py`):
- Deflated Sharpe Ratio vs expected-max-of-N-trials benchmark
  (Bailey & López de Prado 2014), skew/kurtosis-aware PSR, effective-n from
  trade count.
- CSCV PBO (Bailey, Borwein, López de Prado & Zhu 2016) over the candidate +
  up to 12 rival parameter sets from the final phase.
- `validate_candidate` gate: DSR ≥ 0.90, PBO ≤ 0.50, ≥ 20 trades, optional
  held-out OOS Sharpe ≥ 0.
- Walk-forward analyzer (hidden GUI feature): rolling train/test windows,
  efficiency ratio, consistency, degradation verdict.
- Live re-optimizer: 75/25 chronological train/holdout split; candidates
  must pass gates + holdout Sharpe before book admission.

**Regime machinery** (`models/bayesian_changepoint.py`): BOCPD
(Adams & MacKay 2007) with NIG prior, causal MAP run-length; causal per-bar
labels (Bull/Bear × Quiet/Volatile). `trading/regime_switcher.py`: strategy
book (5 slots), per-regime exponentially-weighted shadow scoring, selection
with optional hysteresis, positions managed to completion by their opener.
Regime attribution is lag-1 (return over (t-1,t] credited to the label known
at t-1) — correct.

**Risk controls** (`trading/risk_controls.py`): live-only kill switch
(daily loss, drawdown, consecutive losses, trade cap). Not present in
backtest.

**Dead code**: `CompositeOptimizer` / `PBOCalculatorSimple` /
`TurnoverCalculator` in `psr_composite.py` are referenced only by
`optimizer.py.backup`. `PSRCalculator` is live (used by DSR gate).

### 2. Analyze: where the edge should come from, and where it leaks

**Claimed edge**: buying short-term weakness (oscillator below threshold)
within favorable calendar windows, confirmed across timeframes. Short-term
reversal in equities is documented (Jegadeesh 1990; Lehmann 1990), and
multi-timeframe confirmation is a coherence filter. The calendar cycle,
however, has weak a-priori support at arbitrary periods: documented
seasonality is structural (turn-of-month — Ariel 1987, Lakonishok & Smidt
1988; day-of-week; January), not "37 bars on / 151 off anchored at epoch".
With ~31M cycle combinations, **the cycle is the dominant overfitting
surface**: it can memorize the specific price path. The system's defense is
the DSR/PBO/holdout stack — which makes the *integrity of the OOS evaluation
machinery* the single most important part of the codebase.

**Ranked weaknesses found (evidence verified numerically):**

1. **OOS evaluation is warm-up-truncated** (walk-forward `_test_window`,
   re-optimizer holdout eval). Test slices are simulated in isolation, so
   the warm-up mask restarts at bar 0 of the slice. Measured warm-ups:
   RSI(50,25)=75 bars, MACD(50,25)=275 bars, TRIX(100,50)=650 bars — vs a
   default walk-forward test window of ~21 daily bars and a typical daily
   holdout of ~630 bars. Consequences: (a) OOS windows systematically report
   fewer/zero trades and Sharpe→0, penalizing slower indicators for a purely
   mechanical reason; (b) windows are silently skipped (`oos_trades < 1`),
   injecting selection noise into the walk-forward verdict; (c) holdout
   Sharpe of a truncated-flat curve is 0.0, which **passes** the
   `oos_sharpe >= 0` gate vacuously.

2. **Sortino is mis-specified and exploitable.** Downside deviation is
   computed as `std(losing bars about their own mean)`. A strategy whose
   losses are *uniform* gets downside std → 0 and Sortino → the 100.0 cap.
   Verified: two streams with identical mean per-bar return score Sortino
   100.0 (uniform losses) vs 1.94 (dispersed losses). Sortino is a
   selectable optimization objective, so Optuna can actively exploit this.
   The standard definition (Sortino & van der Meer 1991; Rom & Ferguson
   1993) is the root lower partial moment of order 2 vs a target (MAR=0)
   over ALL observations: `sqrt(mean(min(r,0)^2))`.

3. **Zero-OOS-trade candidates pass the holdout gate.** `min_oos_sharpe=0.0`
   and a flat holdout gives exactly 0.0 ≥ 0.0 → pass. A calendar cycle that
   memorized in-sample rallies and is OFF for the whole holdout produces
   *no out-of-sample evidence at all* and is admitted anyway. This is
   precisely the failure mode the gate exists to catch.

4. **Headline optimization selects in-sample.** The default "START" path
   optimizes and reports on the full dataset; DSR/PBO run after the fact,
   but selection itself never sees a validation split. (Backlog — larger
   change, must not be conflated with 1–3.)

5. **No exposure metric.** Time-in-market is unmeasured, so Sharpe cannot be
   normalized by exposure and entry/exit contribution can't be attributed.
   (Backlog.)

6. **`Start` parameter redundancy**: `Start` is suggested in
   (0, on_max+off_max) but only `Start % (On+Off)` matters — many encodings
   of the same strategy inflate the search space and defeat TPE locality and
   rival dedup. (Backlog.)

7. **Minor**: exit at final bar uses same-bar open (stale price, not
   lookahead); fixed commissions excluded from per-trade %; dead composite
   optimizer code. (Backlog/cosmetic.)

### 3. Hypotheses this iteration

Changes are deliberately confined to the **validation machinery and metric
definitions** — not the strategy — because with a search space this flexible,
un-biased OOS evaluation dominates any signal tweak for out-of-sample
survival probability.

- **H1 (warm-up context in OOS evaluation).** If OOS windows are simulated
  with their preceding training data as warm-up context (scoring only OOS
  bars), because indicator state exists continuously in live trading and
  only the *evaluation* artificially truncates it, then OOS trade counts and
  OOS Sharpe become unbiased estimates of live behavior; walk-forward stops
  silently dropping windows; slow indicators stop being mechanically
  penalized. Expected: strictly more OOS windows retained, nonzero OOS
  activity for strategies that hold positions across the boundary; no change
  to in-sample results.

- **H2 (Sortino → target downside deviation).** If downside deviation uses
  LPM2 with MAR=0 over all bars, because the Sortino literature defines it
  that way and the current form rewards loss *uniformity* rather than low
  downside risk, then the `sortino` objective ranks candidates by true
  downside risk. Expected: uniform-loss and dispersed-loss streams with
  equal LPM2 score equally; the 50x inflation disappears; annualization
  scaling unchanged.

- **H3 (OOS evidence gate).** If gated admission requires a minimum number
  of holdout trades (evidence), because "no out-of-sample trades" is absence
  of evidence rather than evidence of harmlessness — and is the signature of
  cycle memorization — then vacuous holdout passes are rejected. Expected:
  candidates whose cycle is OFF through the holdout are refused admission.

### 4–8. Implementation, validation, decision

Recorded per-hypothesis below as experiments complete.

---

### Experiment H1 — warm-up context in OOS evaluation

**Status**: implemented — see commit history.

**Code changes**:
- `optimization/walk_forward.py::_test_window` now simulates on
  train+test concatenated, locates the first test bar, rescales the OOS
  equity slice to a 1000.0 start, and counts trades whose exit lands in the
  test window (a position opened in train and held into test contributes
  test-period P&L, exactly as live trading would).
- `trading/live_reoptimizer.py::run_reoptimization_cycle` evaluates the
  holdout by simulating over the FULL data and slicing the equity curve at
  the holdout cutoff; OOS trade count (entries inside the holdout) is now
  reported alongside OOS Sharpe.

**Validation**: unit tests construct an always-in-market strategy with a
40-bar warm-up and a 21-bar test window. Writing the test surfaced that the
bias is even worse than analyzed: a window shorter than the indicator
period doesn't merely stay flat — `compute_rsi_vectorized` raises a
broadcasting error on 21 bars with a 30-period RSI, `simulate_multi_tf`
swallows the exception and returns `(None, 0)`, and the walk-forward loop
**silently skips the window**. So isolated evaluation was dropping exactly
the windows where slower strategies would have been judged. Post-fix, the
carried position's test-period P&L is scored (verified against the exact
open-to-open price move); a fast strategy that never needed context keeps
its verdict. A no-context fallback stays graceful. Full suite green
(145 tests).

**Decision**: KEEP. Pure evaluation-fidelity fix; cannot inflate in-sample
results; makes every downstream robustness verdict more truthful.

---

### Experiment H2 — Sortino target downside deviation

**Status**: implemented — see commit history.

**Code change**: `optimization/metrics.py` — downside deviation is now
`sqrt(mean(min(r, 0)^2))` over all bars (MAR = 0), annualized as before.
All-positive curves keep Sortino = 0.0 (no downside evidence ≠ infinite
skill).

**Measured before/after** (same mean return, same gross loss):
| stream | old Sortino | new Sortino |
|---|---|---|
| uniform −1% losses | 100.0 (cap) | 2.25 |
| mixed −0.2%/−1.8% losses | 1.94 | 1.75 |

**Decision**: KEEP. Closes an objective-function exploit; standard
definition; ordering between the two streams is now driven by the second
moment of losses instead of their variance about their own mean.

---

### Experiment H3 — out-of-sample evidence gate

**Status**: implemented — see commit history.

**Code changes**: `validate_candidate` accepts `oos_trades` /
`min_oos_trades` (default 0 = no behavior change for existing callers);
`run_reoptimization_cycle` passes the holdout trade count with
`min_oos_trades=3` and the report shows it.

**Decision**: KEEP. Conservative default (3 trades on a 25% holdout, vs 20
required in-sample) rejects only the vacuous-pass pathology.

---

## Iteration 2 — 2026-07-17

### Experiment H4 — Exposure metric (time in market)

**Hypothesis**: if time-in-market is measured and reported with every
result, because a Sharpe earned while invested 8% of bars has a different
capacity/regime/risk profile than one earned at 80% (per-bar risk metrics
are diluted by flat bars, and the calendar cycle makes low exposure the
norm here), then results become comparable on capital-efficiency terms and
entry/exit attribution becomes possible. Measurement only — no behavior
change to any simulation, objective, or gate.

**Code changes**:
- `simulate_multi_tf` counts in-market bars (same condition as the
  mark-to-market branch) and reports `exposure_frac` through an optional
  caller-owned `stats_out` dict — caller-owned so Optuna's `n_jobs`
  threads can never race on shared optimizer state.
- `calculate_metrics(..., exposure_frac=)` adds `Exposure_%` (absent when
  not provided, so buy-and-hold benchmark metrics are unaffected).
- Every trial's metrics, the final result row, the results CSV, and the
  GUI full report now carry `Exposure_%`.

**Validation**: exposure of an always-in-market parameter set equals
(n − first_fill_bar)/n exactly; a never-entering set reports 0.0; metric
absent without simulation stats; clipped to [0, 100]. Full suite green
(147 tests).

**Decision**: KEEP. Zero behavioral risk, unlocks exposure-aware analysis
(backlog: exposure-normalized comparisons across strategy-book slots).

---

## Iteration 3 — 2026-07-17

### Experiment H5 — validation-split selection in the main optimizer

**Hypothesis**: if trial *selection* uses a chronologically held-out
validation slice (TPE navigates on the first 75% of bars; each phase's
winner is the best score on the untouched last 25%), because selecting on
the data that was optimized is the primary mechanism by which backtests
overstate live performance (Bailey & López de Prado 2014; Arnott, Harvey &
Markowitz 2019 backtesting protocol), then the selected parameter set's
performance on genuinely unseen data improves — most visibly on data with
no true edge, where full-sample selection picks pure memorizers.

**Design**:
- `selection_holdout_frac=0.25` constructor parameter (0 disables; clipped
  to ≤ 0.5). Below `MIN_SELECTION_BARS=400` finest bars the split is
  disabled automatically, so walk-forward inner windows (~126 daily bars)
  keep full-sample selection rather than selecting on a ~30-bar slice.
- TPE's objective value is the TRAIN-segment confidence-scaled score; the
  sampler never sees validation data, so the winner argmax over validation
  is a selection over train-plausible candidates, not a fit to the
  sampler's navigation target.
- Winner per phase = max validation score among trials with a positive
  one; fallback to the train argmax when no trial earned one (logged).
- Segment scoring reuses the H1 slice convention (start one bar early,
  renormalize to 1000) and assigns trades to the segment containing their
  entry bar (same convention as the H3 evidence gate). The trade-count
  confidence ramp applies within each segment.
- Every trial row (and the final result) records Train_Score / Val_Score /
  Train_Trades / Val_Trades; nested layers (reoptimizer holdout, DSR/PBO
  gates) are unchanged and sit outside this split.

**Controlled experiment** (synthetic, 8 seeds × 2 arms × 2 processes;
700-bar history given to the optimizer, 250-bar future never shown;
240 trials, identical sampler seed per arm; winners scored on the future
via full-context simulation, H1 convention):
- `edge`: AR(1) returns, φ = −0.25 — a true, harvestable short-term
  reversal (buy-weakness oscillators have a real edge here).
- `noise`: iid returns, zero drift — any apparent edge is memorization.

**Results (H5a: winner = validation argmax) — FALSIFIED, 24 seeds pooled:**

| process | arm | IS Sharpe | OOS Sharpe | degradation | 
|---|---|---|---|---|
| edge  | A val-argmax   | +0.20 | **+0.12** | +0.08 |
| edge  | B full-sample  | +0.29 | −0.08 | +0.38 |
| noise | A val-argmax   | +0.24 | **−0.36** | **+0.60** |
| noise | B full-sample  | +0.38 | −0.10 | +0.48 |

On the planted-edge process the validation argmax helped (paired OOS
+0.20, degradation +0.08 vs +0.38) and reported scores were honest. But on
the no-edge process it was actively harmful: **the argmax of ~240
candidates over a 175-bar validation slice has HIGHER selection variance
than the argmax over the 700-bar train sample** — validation-lucky noise
strategies (e.g. one seed: IS +1.76 → OOS −1.50) mean-revert below zero
out-of-sample because trading noise pays costs. Pre-registered criterion
(a) (degradation must improve in both processes) failed on noise: +0.60 vs
+0.48. Decision on H5a: **REJECT** the argmax form. Lesson recorded: a
held-out slice is only as good as how little you spend it — argmax spends
all of it.

**Follow-up hypothesis H6 (winner = train argmax among validation
survivors)**: consume ONE BIT of validation information per candidate
(Val_Score > 0 = survive, else vetoed; survivors ranked by TRAIN score;
plain train argmax when nobody survives). Rationale: the memorizer filter
is preserved (a pure memorizer's validation score is centered below zero
after costs) while the selection-variance channel that sank H5a is closed —
you cannot chase validation luck if validation only answers pass/fail.
Same philosophy as the re-optimizer's holdout gate (H3). Same harness,
same 24 seeds, same pre-registered criterion:
(a) pooled degradation A ≤ B in BOTH processes, (b) pooled paired OOS not
significantly negative, (c) noise OOS not worse than B (do no harm).

**Results (H6: validation veto) — KEPT. 24 seeds, paired within-run:**

| process | arm | IS Sharpe | OOS Sharpe | degradation |
|---|---|---|---|---|
| edge  | A veto        | +0.23 | **+0.05** | **+0.17** |
| edge  | B full-sample | +0.47 | −0.47 | +0.94 |
| noise | A veto        | +0.37 | −0.15 | +0.52 |
| noise | B full-sample | +0.47 | −0.01 | +0.48 |

Significance (paired, n=24 per process):
- **edge: paired OOS +0.52 ± 1.04, t = 2.46, p = 0.022 (Wilcoxon p =
  0.021) — significant.** A better in 15/24 seeds; degradation +0.17 vs
  +0.94 (full-sample selection shows the classic IS +0.47 → OOS −0.47
  overfit signature; the veto largely removes it).
- noise: paired −0.14 ± 1.10, p = 0.54 — statistically zero. On no-edge
  data the veto frequently selects configurations that simply don't trade
  OOS (stand-aside), rather than actively trading memorized noise.
- pooled: +0.19, p = 0.24.

**Verdict vs pre-registered criterion**: (b) ✓ and (c) ✓ (noise
difference nowhere near significant); **(a) formally failed on noise by
0.04** (+0.52 vs +0.48) — recorded, not hidden. That margin is an order
of magnitude smaller than the measured between-run variance (identical
full-sample arms swung OOS −0.08 → −0.47 across runs purely from Optuna
n_jobs thread-scheduling nondeterminism), while the edge-side improvement
is significant at 5%. Decision: **KEEP** the veto form. The failed H5a
argmax form stays reverted and on record.

**Additional lessons recorded**:
1. A held-out slice is a budget: argmax spends all of it (falsified);
   a pass/fail veto spends one bit per candidate (kept).
2. With `n_jobs > 1`, TPE trial scheduling makes single-run comparisons
   meaningless; all experiment conclusions here use paired within-run
   differences across many seeds.
3. The veto is conservative by construction — on edge data it sometimes
   stands aside (0 OOS trades in 9/24 edge seeds vs 5/24 for B). The
   significant net OOS gain already prices this in, and "flat when
   unproven" is the preferred failure mode for live capital.

---

## Iteration 4 — 2026-07-17

### Experiment H7 — real-data, multi-asset replication of the veto

**Motivation**: iteration 3's KEEP decision for the validation veto rests
entirely on synthetic AR(1)/noise processes. The protocol requires
multi-asset, real-market evidence. Real data adds what the synthetics
lack: drift regimes, volatility clustering, fat tails — and a long-only
strategy's veto may cost upside in trending eras, which the synthetic
harness cannot reveal.

**Data**: FRED daily closes (no auth, reachable through the environment
proxy): SP500 (~10y), NASDAQCOM restricted to post-1990, CBBTCUSD
(~11y). Yahoo/Stooq were unavailable (rate-limit / anti-bot wall —
recorded; no circumvention attempted). Open prices are unavailable from
FRED, so Open[t] := Close[t-1] — the same convention the test fixtures
use; identical in both arms, so paired differences are unaffected by the
fill approximation.

**Protocol (pre-registered before running)**:
- Per asset, consecutive non-overlapping 950-bar folds (700-bar history
  given to the optimizer, 250-bar never-seen future), walking back from
  the most recent bar. Expected folds: ~2 (SP500) + ~9 (NASDAQ) + ~4
  (BTC) ≈ 15 paired comparisons on independent data blocks.
- Arms as in H6: A = veto selection (`selection_holdout_frac=0.25`),
  B = full-sample selection (0.0); identical sampler seed within a pair;
  240 trials; stock costs for indices, crypto costs for BTC.
- OOS scoring exactly as H6 (full-context simulation, future slice,
  H1 boundary convention); buy & hold future Sharpe recorded as context.
- **Decision rule**: pooled paired OOS Sharpe (A − B) significantly
  negative (t-test or Wilcoxon p < 0.05) → the iteration 3 KEEP is
  overturned and the veto reverts to opt-in. Otherwise the veto stands;
  directionally positive pooled diff counts as replication.

**Results — REPLICATION FAILED; veto demoted to opt-in.**

15 folds (SP500 ×2, NASDAQ ×9, BTC ×4), paired within-fold:

| pooled | IS Sharpe | OOS Sharpe | degradation | OOS>0 |
|---|---|---|---|---|
| A veto        | +0.44 | −0.05 ± 0.94 | +0.49 | 4/15 |
| B full-sample | +0.43 | +0.26 ± 1.03 | +0.17 | 7/15 |

Paired (A−B): **−0.31** (t = −0.91, p = 0.38; Wilcoxon p = 0.42), A
better in 5/15 folds. Buy & hold future-slice Sharpe averaged +0.89.

**Failure analysis**: two mechanisms visible in the folds that the
zero-drift synthetics could not expose:
1. *Stand-aside opportunity cost.* In trending-up folds the veto
   repeatedly selected configurations that did not trade the future at
   all (0 OOS trades in 5 up-folds) while full-sample selection captured
   part of the trend (+2.18, +1.14, +0.87, +0.65). For a long-only
   system in markets with positive drift, "flat when unproven" has a
   real price that synthetic μ≈0 processes hid.
2. *No downside protection delivered.* In the down folds (2022, 2018,
   BTC 2026) the veto's survivors did no better - and often worse - than
   full-sample winners. A single 175-bar validation slice does not
   reliably separate real-data memorizers.

**Decision**: by the letter of the pre-registered rule (overturn only if
p < 0.05 negative) the veto could stand — but the rule was mis-designed:
it placed the burden of proof on the *reversion*, while the project
constitution places it on the *change* ("keep only statistically
significant improvements; otherwise revert"). The synthetic
significance (H6, p = 0.022) did not survive contact with real data
(directionally negative, 5/15). **Demoted to opt-in**
(`use_validation_veto=False` default): default selection returns to the
pre-H5 full-sample argmax; the train/validation split instrumentation
stays always-on (Train_Score / Val_Score / *_Trades recorded in every
result row) — measurement shipped, unproven action not.

**Lessons recorded**:
1. Pre-registration must put the significance burden on the change, not
   the status quo. Rules written asymmetrically get overridden by the
   constitution — noted for every future experiment.
2. Synthetic validation is necessary but nowhere near sufficient: the
   veto's synthetic win (p = 0.022, 24 seeds) was real and still failed
   on markets, because the synthetic processes lacked drift. Every
   future strategy-side hypothesis gets a real-data replication arm
   BEFORE any default changes.
3. Single-slice vetoes are noisy instruments. If held-out selection is
   revisited, it should be CPCV-style multi-split evidence, and it must
   demonstrate real-data value first.
4. Data access constraints documented: Yahoo rate-limits this egress IP;
   Stooq sits behind an anti-bot wall (not circumvented); FRED daily
   closes work and are sufficient for close-fill paired experiments.

---

## Iteration 5 — 2026-07-17

### Experiment H8 — volatility-targeted position sizing

**Hypothesis**: if per-trade exposure is scaled by
`min(1, expanding_median_vol / current_vol)` — where current_vol is the
causal 20-bar realized volatility at the decision bar — because volatility
clusters (Mandelbrot 1963; Engle 1982) while short-horizon expected
returns do not scale with conditional variance (Moreira & Muir 2017,
*Volatility-Managed Portfolios*, JF; Harvey et al. 2018, *The Impact of
Volatility Targeting*, JPM), then OOS risk-adjusted performance improves —
primarily through drawdown/tail reduction in high-vol regimes — without
material return sacrifice.

**Design** (opt-in `vol_targeting=False` default until proven):
- Parameter-free by construction: fixed 20-bar window (inside the 1–3
  month literature range), target = the vol series' own expanding median,
  cap at 1 (no leverage; RiskConfig.MAX_LEVERAGE respected). Deliberately
  NOT searchable by Optuna — a tunable target would be a fresh
  overfitting dial.
- Causal: size at decision bar i uses closes[:i+1] only (prefix test).
- Parity enforced: simulate_multi_tf, the regime-switcher shadow sims,
  and the real switching portfolio all use the same per-bar multiplier
  (parity tests assert bit-equality of curves).

**Pre-registered decision rule (burden on the change, per H7 lesson 1)**,
on the FRED real-data harness (15 folds, arms A = vol targeting on /
B = off, all else identical, default full-sample selection):
- Primary: paired OOS Sharpe (A−B). Secondary: paired OOS max drawdown.
- **Default-on** only if Sharpe improvement p < 0.05 AND drawdown not
  significantly worse.
- **Retained as opt-in** if Sharpe directionally ≥ 0, or drawdown
  significantly reduced without significant Sharpe cost.
- **Feature dropped (reverted)** if Sharpe directionally negative and no
  significant drawdown reduction.

**Results (joint system: optimize-with-sizing vs optimize-without) —
NULL.** 15 folds:

| pooled | IS Sharpe | OOS Sharpe | OOS MaxDD | OOS>0 |
|---|---|---|---|---|
| A vol-targeted | +0.79 | +0.23 ± 0.98 | −9.0% | 9/15 |
| B fixed size   | +0.69 | +0.25 ± 1.13 | −8.7% | 9/15 |

Paired Sharpe −0.018 (t = −0.05, p = 0.96); paired MaxDD −0.28pp
(p = 0.89). Dead null on both pre-registered metrics.

**Confound identified**: the arms optimized under different sizing
regimes and therefore selected different strategies (several folds where
one arm's winner didn't trade the future at all) — TPE selection noise
swamps the sizing effect. The literature effect applies to continuous
exposure to a FIXED underlying strategy; the joint experiment tested
"sizing + re-selection" instead.

**Pre-registered amendment (recorded before running)**: isolate the
mechanism — per fold, take the fixed-sizing arm's winner and replay it
OOS twice, with and without the vol scalar (identical trades, different
sizes; the pairing is exact). Decision: vol targeting is retained as
opt-in only if the isolated replay shows paired OOS improvement in
Sharpe or MaxDD at p < 0.05; otherwise the feature is dropped
(implementation reverted, journal record kept).

**Results (isolated sizing replay) — KEPT as opt-in.** 15 folds, exact
pairing (same winner strategy, same trades, ± the vol scalar):

| pooled | OOS Sharpe | OOS MaxDD |
|---|---|---|
| vol-targeted | +0.07 ± 1.08 | **−11.26%** |
| fixed size   | +0.02 ± 1.07 | −13.84% |

- **Paired MaxDD +2.59pp shallower (t = 2.85, p = 0.013; Wilcoxon
  p = 0.001; better in 13/15 folds)** — significant on both tests, and a
  ~19% relative drawdown reduction. The standout folds are exactly the
  hypothesized regime: the 2022 bear (NASDAQ −15.5% vs −26.1%; SPX
  −4.5% vs −10.6%) and the 2018 BTC crash (−23.7% vs −33.0%).
- Paired Sharpe +0.046 (p = 0.27) — directionally positive, not
  significant: drawdown protection came at no measurable return cost.

**Verdict per the amendment rule** (p < 0.05 in Sharpe OR MaxDD →
retain): **RETAINED as opt-in** (`vol_targeting=True`). Default-on was
pre-registered to require a joint-system Sharpe gain at p < 0.05, which
was null — so the default stays off. The feature ships as what the
evidence says it is: a convex risk-reduction overlay (priority 4:
drawdown reduction), not an alpha source.

**Methodology lesson recorded**: sizing/risk overlays must be evaluated
on FIXED strategies (exact pairing) — testing them through
re-optimization confounds the overlay with TPE selection noise, which
turned a p = 0.013 drawdown effect into a p = 0.89 null in the joint
experiment. Applies to every future overlay hypothesis (stops, regime
filters, exposure caps).

---

## Iteration 6 — 2026-07-17

### Infrastructure: reproducible experiment harness committed

The scripts behind H5–H8b lived in the session scratchpad — an ephemeral
container path. They are now committed under `research/` (scripts as-run
with only path adjustments; `research/fetch_data.py` re-downloads the
FRED series; data snapshots stay out of git because FRED's SP500 series
is S&P-licensed). `research/README.md` records the shared protocol and
the four standing methodology rules.

### Experiment H9 — marginal OOS value of the calendar time cycle

**Motivation**: the ON/OFF calendar cycle is the product's identity and
was ranked the #1 overfitting surface in iteration 1 (a ~31M-combination
calendar pattern with weak a-priori support at arbitrary periods —
documented seasonality is structural: turn-of-month, day-of-week,
January). Five iterations hardened the validation machinery around the
cycle; this experiment finally questions the assumption itself.

**Hypothesis**: if the cycle carries genuine seasonality edge, the full
system (cycle + indicators) should outperform an otherwise-identical
system with the cycle pinned always-ON (indicators only) on never-seen
real data. If the cycle is primarily a memorization surface, its
marginal OOS contribution will be ≤ 0 even though it inflates in-sample
scores.

**Protocol (pre-registered)**: the standard 15-fold real-data harness;
Arm A = cycle searched ((1,60) on, (0,60) off) + indicators; Arm B =
cycle ranges degenerate ((250,250),(0,0)) so the cycle phase is inert
and entries are indicator-gated only. Identical indicator-phase budgets,
sampler seed, costs, selection (default full-sample), no vol targeting.
Primary metric: paired OOS Sharpe (A − B).

**Decision rule (burden on the cycle)**:
- A − B significantly positive (p < 0.05) → the cycle's edge is
  validated on real data; record and keep everything as-is.
- Not significant → the cycle's marginal real-data contribution is
  unproven: record the numbers; recommendation to the maintainer that
  cycle-bearing results should carry a caveat and per-ticker
  walk-forward evidence before deployment. No product change — the
  cycle is the product's identity and that call belongs to the user.
- Significantly negative → same, flagged prominently.

**Results — ⚠️ THE CYCLE SIGNIFICANTLY HARMS OUT-OF-SAMPLE PERFORMANCE.**

15 folds, paired within fold:

| pooled | IS Sharpe | OOS Sharpe | degradation | OOS>0 |
|---|---|---|---|---|
| A cycle + indicators | +1.15 | +0.11 ± 0.77 | **+1.03** | 8/15 |
| B indicators only    | +0.87 | **+0.67 ± 1.03** | +0.20 | 12/15 |

Paired OOS Sharpe (A − B): **−0.557** (t = −2.45, **p = 0.028**;
Wilcoxon p = 0.055 borderline), cycle better in only 4/15 folds.

This is the textbook overfitting signature, measured directly on the
system's core feature: the cycle's ~31M-combination calendar pattern
reliably *inflates* in-sample scores (+1.15 vs +0.87 — an extra flexible
dimension always fits better) and *subtracts* out-of-sample (degradation
+1.03 vs +0.20). Indicators alone were OOS-positive in 12/15 folds at
+0.67 mean Sharpe. The iteration-1 architectural diagnosis ("the cycle
is the dominant overfitting surface") is now an empirical result, not an
argument.

**Verdict per the pre-registered rule** (significantly negative →
record, flag prominently, recommend; NO product change without the
maintainer's decision — the cycle is the product's identity):
1. Recorded and flagged here; a caveat added to the README's strategy
   section pointing at this experiment.
2. **Recommendation to the maintainer**: treat any cycle-bearing
   optimization result as presumptively overfit unless it passes the
   full gate stack (DSR, PBO, OOS trade evidence) AND walk-forward on
   the specific ticker; consider making indicators-only the default
   search posture (achievable today by pinning cycle ranges to
   (250,250)/(0,0)); reserve cycle search for explicit opt-in.
3. The H3 evidence gate (zero-OOS-trade rejection) and H1 warm-up fix
   remain the system's main line of defense against exactly this
   failure mode in the live re-optimizer path.

**Salvage hypothesis queued (H10, future)**: the *arbitrary-period*
cycle is what failed. Structural calendar effects (turn-of-month —
Ariel 1987, Lakonishok & Smidt 1988; day-of-week) are documented and
testable: a cycle constrained to monthly/weekly boundaries with far
fewer combinations may retain value the free-form cycle destroys. Test
with the same paired harness before believing it.

---

## Iteration 7 — 2026-07-17

### Experiment H10 — turn-of-month structural calendar gate (salvage)

**Hypothesis**: H9 condemned the *free-form* cycle (arbitrary period,
~31M combinations, memorization surface). The turn-of-month window —
last trading day through the first three trading days of each month —
is a *documented, persistent, zero-parameter* calendar regularity
(Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008: TOM days
carry essentially all of the equity premium in US data 1926–2005). If
calendar gating per se has value and H9's failure was the free cycle's
tunability, a TOM-gated indicator system should not suffer the OOS
penalty — and may beat the ungated system.

**Design**: research-only `TomGatedOptimizer` subclass (AND the TOM
window onto entries, window close forces exit — semantics identical to
the ON/OFF cycle, window fixed by the exchange calendar, which is known
ex-ante, so the gate is causal). No product code is touched unless the
hypothesis survives. Arms: C = TOM-gated indicators, B = ungated
indicators; identical budgets/seed; cycle inert in both.

**Pre-registered decision rule**:
- Primary population: the 11 equity-index folds (SP500 + NASDAQ) — TOM
  is an equities phenomenon; the 4 BTC folds are exploratory only and
  carry no decision weight.
- C − B paired OOS Sharpe significantly positive (p < 0.05) →
  structural calendar gating validated; recommend a product TOM-gate
  option to the maintainer (their call to adopt).
- Not significant → calendar gating adds no measurable value even in
  its literature-strongest form; H9's conclusion stands unqualified;
  the research subclass stays in research/ as the record.
- Significantly negative → gating harms even without tunability —
  strongest possible endorsement of the ungated indicator system.

**Results — SALVAGE FAILED; calendar gating is closed in both forms.**

Primary (11 equity folds, paired):

| pooled | IS Sharpe | OOS Sharpe | degradation | OOS>0 |
|---|---|---|---|---|
| C TOM-gated | +0.60 | −0.04 ± 0.67 | +0.65 | 5/11 |
| B ungated   | +0.75 | **+0.43 ± 1.05** | +0.32 | 7/11 |

Paired OOS Sharpe (C − B): −0.47 (t = −1.59, p = 0.14; Wilcoxon
p = 0.15), TOM better in 3/11. Exploratory BTC (no prior): +0.04,
p = 0.97 — noise, as expected.

**Verdict per the pre-registered rule** (not significant → no value
demonstrated): even the zero-parameter, literature-strongest calendar
gate adds nothing here, with a negative point estimate. Combined with
H9 (free cycle significantly harmful, p = 0.028), **calendar gating is
closed in both its free and structural forms for this system**. The
measured edge lives in the indicator layer.

**Interpretation recorded (mechanism, not excuse)**: this does not
refute the TOM effect itself — McConnell & Xu documented a return
concentration in *unconditional long exposure*. A system that already
times entries with oscillators loses more by discarding ~85% of its
opportunity set than the TOM premium returns: the TOM arm's OOS trade
counts collapse (several folds at 0–3 trades). Calendar windows
compete with indicator timing rather than compounding it.

**Caveat recorded**: 11 folds is modest power — a small true TOM
contribution cannot be excluded. But the burden was pre-registered on
the gate, and the direction is negative; there is nothing here to act
on.

**Iteration-7 conclusion**: H9's maintainer recommendation stands
unqualified. The indicator-only configuration (cycle ranges pinned to
(250,250)/(0,0)) is the evidence-backed research posture; the default
product posture remains the maintainer's decision.

---

## Iteration 8 — 2026-07-17

### Experiment H11 — viability certification of the V1 candidate system

**The candidate** (assembled from accumulated evidence, not searched
for): optimize **indicators-only** (cycle pinned inert — H9 p = 0.028,
H10 confirmed nothing salvageable), standard Sharpe objective and gate
stack; deploy the fixed winner **with the volatility overlay** (H8b:
−2.6pp OOS MaxDD, p = 0.013, at no Sharpe cost — proven precisely in
this fixed-strategy application). Every component earned its place in a
pre-registered experiment; the assembly itself is what gets certified
now.

**Fresh holdout**: 20 new folds from assets and asset classes no prior
decision this session ever touched — NIKKEI225 post-1990 (9 folds,
non-US equity), DJIA (2 folds), WTI spot post-1990 (9 folds,
commodity) — plus the original 15 (SP500/NASDAQ/BTC). 35 folds total,
reported pooled AND split original-vs-holdout.

**Pre-registered viability criteria** (ALL must hold):
- **V1 absolute**: pooled candidate OOS Sharpe mean > 0 with one-sample
  t-test p < 0.05 AND Wilcoxon signed-rank p < 0.05.
- **V2 risk proposition**: paired candidate-vs-B&H OOS max drawdown
  significantly shallower (p < 0.05).
- **V3 no significant return sacrifice**: paired candidate-vs-B&H OOS
  Sharpe NOT significantly negative (inferiority not established at
  p < 0.05).
- **Generalization override**: if the pooled battery passes but the
  untouched-asset subset's mean OOS Sharpe is negative, the verdict
  downgrades to NOT certified.
- Note: B&H on WTI uses the spot series (no roll costs), which flatters
  the benchmark — a conservative bias against certification.

Verdict semantics: "viable" = survives this battery; the app's live
re-optimizer with its gate stack (H1/H3-hardened) remains the
deployment path, and paper trading remains the final arbiter.

**Results — NOT CERTIFIED (one criterion short), risk proposition
overwhelming.** 35 folds:

| population | cand OOS Sharpe | cand MaxDD | B&H Sharpe | B&H MaxDD |
|---|---|---|---|---|
| pooled (35) | +0.40 ± 1.04 (21/35 > 0) | −12.5% | +0.59 | −23.3% |
| holdout (20) | +0.24 (13/20 > 0) | −13.1% | +0.37 | −25.3% |
| original (15) | +0.62 | −11.6% | +0.89 | −20.7% |

- **V2 ✓ (emphatic)**: paired MaxDD +10.9pp shallower, t = 7.73,
  p < 0.0001, shallower in **34/35** folds (holdout 20/20). The
  candidate cuts drawdowns roughly in half across every asset class
  and era.
- **V3 ✓**: paired Sharpe vs B&H −0.19, p = 0.26 — inferiority not
  established.
- **Override ✓**: holdout subset positive (+0.24).
- **V1 ✗ (split)**: t-test p = 0.030 passes; Wilcoxon p = 0.081
  fails the AND requirement. The median fold is not far enough from
  zero: a fat negative tail, concentrated in WTI (equity-tuned
  mean-reversion on a commodity, mean fold Sharpe ≈ −0.1) and
  high-turnover folds (20–40 trades/fold paying costs).

**Verdict**: by the pre-registered letter, NOT certified. The same
discipline that rejected H5a and demoted the veto binds here. The
candidate is demonstrably a superior *risk* vehicle to passive
exposure; its absolute-profitability consistency is one test short.

**Diagnosis for the next iteration**: the shortfall is fold-median
consistency — single-winner variance. The one product component never
tested in this loop is ensemble diversification (the strategy book /
auto-build machinery exists for exactly this). 1/N ensembles of
heterogeneous selections are the classic variance reducer that
survives out-of-sample (DeMiguel, Garlappi & Uppal 2009).

### Experiment H12 — V2 candidate: 1/N multi-objective ensemble

**Hypothesis**: if each fold deploys an equal-weight (1/N) portfolio of
THREE winners — the same indicators-only optimization run under the
sharpe, sortino, and calmar objectives (distinct seeds), each with the
H8b vol overlay — because 1/N diversification across heterogeneous
selection criteria reduces single-draw variance without adding fitted
parameters, then fold-level consistency (the failed Wilcoxon) improves
while the proven drawdown edge is preserved.

**Pre-registered rule: identical battery to H11** (V1 t AND Wilcoxon
p < 0.05; V2; V3; holdout override), same 35 folds, same benchmarks.
No cherry-picking of assets: WTI stays in. If this fails, the verdict
is recorded and the loop continues elsewhere — the battery does not
get easier.

**Results (H12) — FALSIFIED; the ensemble is worse.** 35 folds:

| pooled | OOS Sharpe | MaxDD | vs B&H Sharpe |
|---|---|---|---|
| 1/N ensemble | +0.28 ± 1.11 (23/35 > 0) | −10.5% | −0.31 (p = 0.0036 ✗) |
| (H11 single) | +0.40 ± 1.04 | −12.5% | −0.19 (p = 0.26 ✓) |

V2 strengthened to a perfect 35/35 shallower drawdowns (+12.9pp,
p < 0.0001) — but V1 fully failed (t p = 0.146, Wilcoxon p = 0.149) and
V3 flipped to **significant Sharpe inferiority** vs B&H. The leg data
shows why: the sortino and calmar legs are frequently weaker selectors
than the sharpe leg, and 1/N averaged them in. Diversification across
selection criteria reduced variance less than it diluted the mean.
**The single sharpe-winner configuration (H11) remains the best
measured candidate.** Lesson: 1/N helps when legs have comparable
expected quality (DeMiguel et al.'s setting); ensembling a strong
selector with weaker ones is dilution, not diversification.

### Experiment H13 — frozen-candidate confirmation on fresh in-domain data

**Domain claim (pre-registered with justification)**: the candidate's
claimed domain is **equities and crypto** — precisely the two asset
classes the product supports (its only cost presets are
`for_stocks`/`for_crypto`). The H11 WTI folds stand as recorded
evidence that the system does NOT extend to commodities. This narrowing
is acknowledged as a post-hoc hypothesis prompted by WTI's failure —
which is exactly why the confirmation below uses ONLY data that played
no part in any prior decision this session.

**Frozen candidate**: H11's V1 exactly — indicators-only sharpe
optimization (240 trials, seed 777, cycle inert), single winner
deployed with the H8b vol overlay. Nothing tuned, nothing re-selected.

**Fresh confirmation set (25 folds, all in-domain, all untouched)**:
NASDAQ pre-1990 (5), NIKKEI pre-1990 (10), Dow Transports (2), Dow
Utilities (2), Coinbase ETH (3), Coinbase LTC (3).

**Pre-registered battery (same as H11)**: V1 absolute (t AND Wilcoxon
p < 0.05), V2 drawdown vs B&H (p < 0.05), V3 no significant Sharpe
inferiority. All folds are holdout by construction. The combined
in-domain totality (fresh 25 + H11's 26 equity/crypto folds) is
reported as context, clearly labeled as partially non-fresh.

**Caveats recorded up front**: pre-1990 eras carry modern cost presets
(historical costs were higher — flatters absolute returns; noted
against certification confidence); NIKKEI pre-1990 includes the 1980s
bubble (a brutal B&H benchmark — biases V3 against the candidate,
conservative).

**Results**: (recorded below after the run)

---

### Backlog (future iterations, in priority order)

1. ~~Validation-split selection inside the main optimizer~~ (closed,
   iterations 3–4: argmax form falsified on synthetics; veto form won on
   synthetics but failed real-data replication and is opt-in only;
   split instrumentation shipped as always-on measurement). Reopen only
   as CPCV-style multi-split evidence with a real-data arm.
2. ~~Exposure %~~ (done, iteration 2); trades/year and
   time-in-market-normalized comparisons still open.
3. ~~Cycle-space concerns~~ fully closed (iterations 6–7): free-form
   cycle significantly harms OOS (H9, p = 0.028); the zero-parameter
   turn-of-month gate adds nothing (H10, p = 0.14, negative point
   estimate). The maintainer's product decision on default search
   posture remains open — the evidence-backed posture is
   indicators-only.
4. ~~Volatility-targeted position sizing~~ (done, iteration 5 — retained
   as opt-in on significant real-data drawdown reduction, p = 0.013;
   isolated-replay evidence; default-on Sharpe bar not met). Live-trader
   wiring (alpaca path) still needed before the opt-in flag may be used
   with real orders — parity currently covers backtest + shadow +
   switching portfolio only.
5. Regime-dependence report for single strategies in the default flow (the
   machinery exists but is hidden).
6. Remove dead `CompositeOptimizer`/`PBOCalculatorSimple` code and
   `*.backup` files.
7. Final-bar exit uses same-bar open — harmless but imprecise; align with
   next-open convention by dropping the last unclosed mark instead.
8. Multi-asset validation harness: same parameter set replayed on correlated
   tickers (SPY/QQQ/IWM) as a cheap generalization check.
