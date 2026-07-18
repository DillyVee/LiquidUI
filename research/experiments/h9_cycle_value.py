"""
H9: marginal out-of-sample value of the calendar time cycle.

The ON/OFF calendar cycle is the system's identity and its largest
overfitting surface (~31M parameter combinations of pure calendar
pattern). This experiment measures what the cycle actually contributes
out-of-sample on real data:

  Arm A: status quo - cycle searched ((1,60),(0,60)) + indicators
  Arm B: cycle pinned always-ON ((250,250),(0,0)) - indicators only,
         identical indicator-phase trial budget

Paired within fold, same sampler seed, default selection, no vol
targeting. Pre-registered decision rule in RESEARCH_JOURNAL.md.
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

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
N_HISTORY, N_FUTURE = 700, 250
FOLD = N_HISTORY + N_FUTURE
N_TRIALS = 240
ANN = 252.0
PARAM_PREFIXES = ("MN1_", "MN2_", "Entry_", "Exit_", "On_", "Off_", "Start_", "IND")

CYCLE_RANGES = ((1, 60), (0, 60), (0, 120))  # arm A, as in H7/H8
NO_CYCLE_RANGES = ((250, 250), (0, 0), (0, 0))  # arm B: degenerate always-ON

ASSETS = [
    ("SP500", "SP500.csv", None, TransactionCosts.for_stocks),
    ("NASDAQ", "NASDAQCOM.csv", "1990-01-01", TransactionCosts.for_stocks),
    ("BTC", "CBBTCUSD.csv", None, TransactionCosts.for_crypto),
]


def load_fred(path, min_date=None):
    df = pd.read_csv(os.path.join(HERE, path), na_values=".")
    df.columns = ["Datetime", "Close"]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.dropna().reset_index(drop=True)
    if min_date:
        df = df[df["Datetime"] >= min_date].reset_index(drop=True)
    df["Open"] = df["Close"].shift(1)
    return df.dropna().reset_index(drop=True)[["Datetime", "Open", "Close"]]


def folds(df):
    out, end = [], len(df)
    while end - FOLD >= 0:
        out.append(df.iloc[end - FOLD : end].reset_index(drop=True))
        end -= FOLD
    return list(reversed(out))


def opt_kwargs(costs, cycle_ranges, n_trials=N_TRIALS):
    return dict(
        n_trials=n_trials,
        time_cycle_ranges=cycle_ranges,
        mn1_range=(5, 40),
        mn2_range=(3, 15),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="H9",
        timeframes=["daily"],
        transaction_costs=costs(),
        position_size=1.0,
        seed=777,
    )


def sharpe_of(eq):
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ANN))


def run_arm(df_fold, costs, cycle_ranges):
    hist = {"daily": df_fold.iloc[:N_HISTORY].reset_index(drop=True)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = MultiTimeframeOptimizer(df_dict=hist, **opt_kwargs(costs, cycle_ranges))
        opt.run()
        if not opt.all_results:
            return None
        winner = {
            k: v
            for k, v in opt.all_results[0].items()
            if any(k.startswith(p) for p in PARAM_PREFIXES)
        }
        is_sharpe = float(opt.all_results[0].get("Sharpe_Ratio", 0.0))
        ev = MultiTimeframeOptimizer(
            df_dict={"daily": df_fold.reset_index(drop=True)},
            **{**opt_kwargs(costs, cycle_ranges), "n_trials": 1},
        )
        eq, _, trades = ev.simulate_multi_tf(winner, return_trades=True)
    if eq is None:
        return None
    cutoff = df_fold["Datetime"].iloc[N_HISTORY]
    return dict(
        is_sharpe=is_sharpe,
        oos_sharpe=sharpe_of(eq[N_HISTORY - 1 :]),
        oos_trades=sum(1 for t in trades if pd.Timestamp(t["Entry_Date"]) >= cutoff),
    )


def main():
    rows = []
    for name, path, min_date, costs in ASSETS:
        blocks = folds(load_fred(path, min_date))
        print(f"{name}: {len(blocks)} folds")
        for i, blk in enumerate(blocks):
            a = run_arm(blk, costs, CYCLE_RANGES)
            b = run_arm(blk, costs, NO_CYCLE_RANGES)
            if a is None or b is None:
                print(f"  fold {i}: arm failed, skipped")
                continue
            rows.append(dict(asset=name, fold=i, a=a, b=b))
            span = f"{blk['Datetime'].iloc[N_HISTORY].date()}..{blk['Datetime'].iloc[-1].date()}"
            print(
                f"  fold {i} [{span}] "
                f"A(cycle) IS {a['is_sharpe']:+.2f} -> OOS {a['oos_sharpe']:+.2f} "
                f"({a['oos_trades']} tr) | "
                f"B(no-cycle) IS {b['is_sharpe']:+.2f} -> OOS {b['oos_sharpe']:+.2f} "
                f"({b['oos_trades']} tr)"
            )

    from scipy import stats

    a_s = np.array([r["a"]["oos_sharpe"] for r in rows])
    b_s = np.array([r["b"]["oos_sharpe"] for r in rows])
    a_i = np.array([r["a"]["is_sharpe"] for r in rows])
    b_i = np.array([r["b"]["is_sharpe"] for r in rows])

    print(f"\n=== pooled over {len(rows)} folds ===")
    print(
        f"A cycle+ind : IS {a_i.mean():+.2f} | OOS {a_s.mean():+.2f}+/-{a_s.std():.2f} "
        f"| degradation {np.mean(a_i - a_s):+.2f} | OOS>0 {int((a_s>0).sum())}/{len(rows)}"
    )
    print(
        f"B ind only  : IS {b_i.mean():+.2f} | OOS {b_s.mean():+.2f}+/-{b_s.std():.2f} "
        f"| degradation {np.mean(b_i - b_s):+.2f} | OOS>0 {int((b_s>0).sum())}/{len(rows)}"
    )
    d = a_s - b_s
    t, p = stats.ttest_rel(a_s, b_s)
    try:
        wp = stats.wilcoxon(a_s, b_s, zero_method="zsplit").pvalue
    except ValueError:
        wp = float("nan")
    print(
        f"paired OOS Sharpe (A-B): mean {d.mean():+.3f} sd {d.std(ddof=1):.3f}  "
        f"t={t:+.2f} p={p:.3f}  wilcoxon p={wp:.3f}  cycle better {int((d>0).sum())}/{len(d)}"
    )


if __name__ == "__main__":
    main()
