"""
H7: real-data multi-asset replication of the H6 validation veto.

Per asset: consecutive non-overlapping 950-bar folds walking back from
the most recent bar; 700-bar history -> optimizer, 250-bar future ->
never-seen evaluation. Arms A (veto, frac=0.25) vs B (full-sample,
frac=0.0), identical sampler seed within a pair. OOS scored exactly as
the H6 harness (full-context simulation, future slice). Buy & hold
future Sharpe recorded as context. Pre-registered decision rule in
RESEARCH_JOURNAL.md.
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

ASSETS = [
    ("SP500", "SP500.csv", None, TransactionCosts.for_stocks()),
    ("NASDAQ", "NASDAQCOM.csv", "1990-01-01", TransactionCosts.for_stocks()),
    ("BTC", "CBBTCUSD.csv", None, TransactionCosts.for_crypto()),
]


def load_fred(path, min_date=None):
    df = pd.read_csv(os.path.join(HERE, path), na_values=".")
    df.columns = ["Datetime", "Close"]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.dropna().reset_index(drop=True)
    if min_date:
        df = df[df["Datetime"] >= min_date].reset_index(drop=True)
    df["Open"] = df["Close"].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df[["Datetime", "Open", "Close"]]


def folds(df):
    """Non-overlapping FOLD-bar blocks, walking back from the last bar"""
    out = []
    end = len(df)
    while end - FOLD >= 0:
        out.append(df.iloc[end - FOLD : end].reset_index(drop=True))
        end -= FOLD
    return list(reversed(out))


def opt_kwargs(frac, costs):
    return dict(
        n_trials=N_TRIALS,
        time_cycle_ranges=((1, 60), (0, 60), (0, 120)),
        mn1_range=(5, 40),
        mn2_range=(3, 15),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="H7",
        timeframes=["daily"],
        transaction_costs=costs,
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


def run_arm(df_fold, frac, costs):
    hist = {"daily": df_fold.iloc[:N_HISTORY].reset_index(drop=True)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = MultiTimeframeOptimizer(df_dict=hist, **opt_kwargs(frac, costs))
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
            **{**opt_kwargs(0.0, costs), "n_trials": 1},
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
        df = load_fred(path, min_date)
        blocks = folds(df)
        print(f"{name}: {len(df)} bars -> {len(blocks)} folds")
        for i, blk in enumerate(blocks):
            a = run_arm(blk, 0.25, costs)
            b = run_arm(blk, 0.0, costs)
            if a is None or b is None:
                print(f"  fold {i}: arm failed, skipped")
                continue
            bh = sharpe_of(blk["Close"].to_numpy()[N_HISTORY - 1 :])
            span = f"{blk['Datetime'].iloc[N_HISTORY].date()}..{blk['Datetime'].iloc[-1].date()}"
            rows.append(dict(asset=name, fold=i, a=a, b=b, bh=bh))
            print(
                f"  fold {i} [{span}] B&H OOS {bh:+.2f} | "
                f"A(veto) IS {a['is_sharpe']:+.2f} -> OOS {a['oos_sharpe']:+.2f} "
                f"({a['oos_trades']} tr) | "
                f"B(full) IS {b['is_sharpe']:+.2f} -> OOS {b['oos_sharpe']:+.2f} "
                f"({b['oos_trades']} tr)"
            )

    a_oos = np.array([r["a"]["oos_sharpe"] for r in rows])
    b_oos = np.array([r["b"]["oos_sharpe"] for r in rows])
    a_is = np.array([r["a"]["is_sharpe"] for r in rows])
    b_is = np.array([r["b"]["is_sharpe"] for r in rows])
    d = a_oos - b_oos

    from scipy import stats

    t, p = stats.ttest_rel(a_oos, b_oos)
    try:
        wp = stats.wilcoxon(a_oos, b_oos, zero_method="zsplit").pvalue
    except ValueError:
        wp = float("nan")
    print(f"\n=== pooled over {len(rows)} folds ===")
    print(
        f"A veto : IS {a_is.mean():+.2f} | OOS {a_oos.mean():+.2f}+/-{a_oos.std():.2f} "
        f"| degradation {np.mean(a_is - a_oos):+.2f} | OOS>0 {int((a_oos>0).sum())}/{len(rows)}"
    )
    print(
        f"B full : IS {b_is.mean():+.2f} | OOS {b_oos.mean():+.2f}+/-{b_oos.std():.2f} "
        f"| degradation {np.mean(b_is - b_oos):+.2f} | OOS>0 {int((b_oos>0).sum())}/{len(rows)}"
    )
    print(
        f"paired (A-B): mean {d.mean():+.3f} sd {d.std(ddof=1):.3f}  "
        f"t={t:+.2f} p={p:.3f}  wilcoxon p={wp:.3f}  A better {int((d>0).sum())}/{len(rows)}"
    )
    bh = np.array([r["bh"] for r in rows])
    print(f"buy&hold OOS Sharpe mean {bh.mean():+.2f}")


if __name__ == "__main__":
    main()
