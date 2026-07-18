# Research experiments

Reproduction scripts for the experiments recorded in
[`../RESEARCH_JOURNAL.md`](../RESEARCH_JOURNAL.md). The journal is the
canonical record of hypotheses, pre-registered decision rules, numbers,
and verdicts; these scripts are how those numbers were produced.

## Setup

```bash
pip install -r ../requirements.txt
python fetch_data.py        # downloads FRED daily series into research/data/
```

`research/data/` is gitignored: the SP500 series is licensed to FRED by
S&P, so snapshots are not committed. Numbers will drift slightly as FRED
appends new observations; the journal records the as-of date of each run
(2026-07-17).

## Protocol shared by the real-data experiments

- Per asset (SP500, NASDAQ post-1990, BTC-USD), consecutive
  **non-overlapping 950-bar folds** walking back from the most recent bar:
  700 bars of history are given to the optimizer, the next 250 bars are a
  never-seen future.
- Winners are re-simulated over the **full fold** (so indicator warm-up
  has context, matching live state) and scored on the future slice only.
- All comparisons are **paired within fold** (Optuna's `n_jobs` thread
  scheduling makes single-run cross-comparisons meaningless).
- FRED provides closes only; `Open[t] := Close[t-1]`. The approximation is
  identical in both arms of every experiment, so paired differences are
  unaffected.

## Scripts

| script | journal experiment | question |
|---|---|---|
| `experiments/h5_experiment.py` | H5a / H6 | does held-out selection beat full-sample selection on synthetic planted-edge vs noise processes? (ran twice: against the argmax `_phase_winner`, then the veto variant - pair each run with the optimizer code at the commit the journal records) |
| `experiments/h7_realdata.py` | H7 | does the validation veto replicate on real markets? |
| `experiments/h8_voltarget.py` | H8 | does volatility targeting improve the jointly re-optimized system? |
| `experiments/h8b_isolated.py` | H8b | does the vol scalar help when the strategy is held fixed (exact pairing)? |
| `experiments/h9_cycle_value.py` | H9 | what is the marginal OOS value of the calendar time cycle itself? |

Each script prints per-fold rows plus pooled paired statistics
(t-test and Wilcoxon). Expect ~10-20 minutes per real-data script
(~30 optimizer runs of 240 trials each).

## Rules these experiments follow

1. Decision rules are pre-registered in the journal **before** results are
   seen, with the significance burden on the change, never on the status
   quo.
2. Synthetic evidence alone never changes a default - every strategy-side
   hypothesis gets a real-data arm.
3. Overlays (sizing, stops, filters) are evaluated on fixed strategies
   with exact pairing, never through re-optimization.
4. Failed experiments are kept in the journal and in this directory.
