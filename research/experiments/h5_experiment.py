"""
H5 experiment: does validation-split selection improve OOS survival?

Arms (identical data, trials, sampler seed):
  A: selection_holdout_frac = 0.25  (H5: winners picked on validation slice)
  B: selection_holdout_frac = 0.0   (pre-H5: winners picked on full sample)

Processes:
  edge : AR(1) returns with phi = -0.25 (true short-term reversal ->
         buy-weakness oscillators have a real, harvestable edge)
  noise: iid returns, zero drift (any apparent edge is memorization)

Protocol per seed: generate 950 bars; optimizer sees ONLY the first 700
("history"); the last 250 ("future") are never shown to either arm.
Winner params are then simulated over all 950 bars (warm-up context) and
scored on the future slice alone. Report per-arm mean OOS Sharpe and
degradation (in-sample Sharpe - OOS Sharpe).
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from config.settings import TransactionCosts  # noqa: E402
from optimization.optimizer import MultiTimeframeOptimizer  # noqa: E402

N_HISTORY = 700
N_FUTURE = 250
N_TRIALS = 240
SEEDS = range(24)
ANN = 252.0

PARAM_PREFIXES = ("MN1_", "MN2_", "Entry_", "Exit_", "On_", "Off_", "Start_", "IND")


def gen_prices(seed, phi, n=N_HISTORY + N_FUTURE, mu=0.0002, sigma=0.012):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = mu + phi * (r[t - 1] - mu) + eps[t]
    closes = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(
        {
            "Datetime": pd.bdate_range("2021-01-04", periods=n),
            "Open": np.concatenate([[100.0], closes[:-1]]),
            "Close": closes,
        }
    )


def opt_kwargs(frac):
    return dict(
        n_trials=N_TRIALS,
        time_cycle_ranges=((1, 60), (0, 60), (0, 120)),
        mn1_range=(5, 40),
        mn2_range=(3, 15),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="H5EXP",
        timeframes=["daily"],
        transaction_costs=TransactionCosts.for_stocks(),
        position_size=1.0,
        seed=777,
        selection_holdout_frac=frac,
    )


def sharpe_of(eq_slice):
    r = np.diff(eq_slice) / eq_slice[:-1]
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ANN))


def run_arm(df_full, frac):
    hist = {"daily": df_full.iloc[:N_HISTORY].reset_index(drop=True)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = MultiTimeframeOptimizer(df_dict=hist, **opt_kwargs(frac))
        opt.run()
        if not opt.all_results:
            return None
        winner = {
            k: v
            for k, v in opt.all_results[0].items()
            if any(k.startswith(p) for p in PARAM_PREFIXES)
        }
        is_sharpe = float(opt.all_results[0].get("Sharpe_Ratio", 0.0))

        # OOS: simulate over history+future, score the future slice only
        ev = MultiTimeframeOptimizer(
            df_dict={"daily": df_full.reset_index(drop=True)},
            **{**opt_kwargs(0.0), "n_trials": 1},
        )
        stats = {}
        eq, _, trades = ev.simulate_multi_tf(winner, return_trades=True, stats_out=stats)
    if eq is None:
        return None
    cutoff = df_full["Datetime"].iloc[N_HISTORY]
    oos_sharpe = sharpe_of(eq[N_HISTORY - 1 :])
    oos_trades = sum(1 for t in trades if pd.Timestamp(t["Entry_Date"]) >= cutoff)
    return dict(is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, oos_trades=oos_trades)


def main():
    for label, phi in (("edge (AR1 phi=-0.25)", -0.25), ("noise (iid)", 0.0)):
        rows = {"A_val": [], "B_full": []}
        for seed in SEEDS:
            df_full = gen_prices(seed, phi)
            a = run_arm(df_full, 0.25)
            b = run_arm(df_full, 0.0)
            if a is None or b is None:
                print(f"seed {seed}: arm failed, skipping")
                continue
            rows["A_val"].append(a)
            rows["B_full"].append(b)
            print(
                f"[{label}] seed {seed}: "
                f"A(val-select) IS {a['is_sharpe']:+.2f} -> OOS {a['oos_sharpe']:+.2f} "
                f"({a['oos_trades']} tr) | "
                f"B(full-select) IS {b['is_sharpe']:+.2f} -> OOS {b['oos_sharpe']:+.2f} "
                f"({b['oos_trades']} tr)"
            )
        print(f"\n=== {label} summary over {len(rows['A_val'])} seeds ===")
        for arm, data in rows.items():
            iss = np.array([d["is_sharpe"] for d in data])
            oos = np.array([d["oos_sharpe"] for d in data])
            print(
                f"  {arm:7s}: IS Sharpe {iss.mean():+.2f}+/-{iss.std():.2f} | "
                f"OOS Sharpe {oos.mean():+.2f}+/-{oos.std():.2f} | "
                f"degradation {np.mean(iss - oos):+.2f} | "
                f"OOS>0 in {int((oos > 0).sum())}/{len(oos)} seeds"
            )
        a_oos = np.array([d["oos_sharpe"] for d in rows["A_val"]])
        b_oos = np.array([d["oos_sharpe"] for d in rows["B_full"]])
        diff = a_oos - b_oos
        print(
            f"  paired OOS Sharpe (A - B): mean {diff.mean():+.2f}, "
            f"A better in {int((diff > 0).sum())}/{len(diff)} seeds\n"
        )


if __name__ == "__main__":
    main()
