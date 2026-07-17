"""
H10: does a STRUCTURALLY constrained calendar gate (turn-of-month)
behave differently from the free-form cycle H9 condemned?

The turn-of-month effect - equity returns concentrating in the last
trading day through the first three trading days of each month - is
documented and persistent (Ariel 1987; Lakonishok & Smidt 1988;
McConnell & Xu 2008). Unlike the free cycle's ~31M arbitrary patterns,
the TOM window has ZERO tunable parameters, so it cannot memorize.

  Arm C: indicators gated to the TOM window (entries only inside it,
         window close forces exit - same semantics the free cycle used)
  Arm B: same indicator search, no calendar gate

The TOM gate is a research-only subclass of the optimizer - no product
code changes unless the hypothesis survives. Paired within fold, same
seed. Primary population: the 11 equity-index folds (TOM is an equities
phenomenon); BTC folds reported as exploratory only. Pre-registered
decision rule in RESEARCH_JOURNAL.md.
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
NO_CYCLE_RANGES = ((250, 250), (0, 0), (0, 0))  # cycle inert in both arms

ASSETS = [
    ("SP500", "SP500.csv", None, TransactionCosts.for_stocks, True),
    ("NASDAQ", "NASDAQCOM.csv", "1990-01-01", TransactionCosts.for_stocks, True),
    ("BTC", "CBBTCUSD.csv", None, TransactionCosts.for_crypto, False),  # exploratory
]


def tom_mask(datetimes) -> np.ndarray:
    """True on the last trading day of each month and the first three
    trading days of the following month. Exchange calendars are known in
    advance, so this gate is causal."""
    dt = pd.DatetimeIndex(pd.to_datetime(datetimes))
    n = len(dt)
    mask = np.zeros(n, dtype=bool)
    ym = np.asarray(dt.year * 12 + dt.month)
    month_ends = np.flatnonzero(np.diff(ym) != 0)  # i = last bar of its month
    mask[month_ends] = True
    for i in month_ends:
        mask[i + 1 : i + 4] = True
    return mask


class TomGatedOptimizer(MultiTimeframeOptimizer):
    """Research-only: AND the turn-of-month window onto the entry gate and
    force exits when it closes - identical semantics to the ON/OFF cycle,
    with the window fixed by the calendar instead of searched."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tom = tom_mask(self.np_data[self.finest_tf]["datetime"])

    def compute_signals(self, params):
        enter, exit_ = super().compute_signals(params)
        return enter & self._tom, exit_ | ~self._tom


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


def opt_kwargs(costs, n_trials=N_TRIALS):
    return dict(
        n_trials=n_trials,
        time_cycle_ranges=NO_CYCLE_RANGES,
        mn1_range=(5, 40),
        mn2_range=(3, 15),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="H10",
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


def run_arm(df_fold, costs, cls):
    hist = {"daily": df_fold.iloc[:N_HISTORY].reset_index(drop=True)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = cls(df_dict=hist, **opt_kwargs(costs))
        opt.run()
        if not opt.all_results:
            return None
        winner = {
            k: v
            for k, v in opt.all_results[0].items()
            if any(k.startswith(p) for p in PARAM_PREFIXES)
        }
        is_sharpe = float(opt.all_results[0].get("Sharpe_Ratio", 0.0))
        ev = cls(
            df_dict={"daily": df_fold.reset_index(drop=True)},
            **opt_kwargs(costs, n_trials=1),
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


def report(rows, label):
    from scipy import stats

    if not rows:
        print(f"\n=== {label}: no folds ===")
        return
    c_s = np.array([r["c"]["oos_sharpe"] for r in rows])
    b_s = np.array([r["b"]["oos_sharpe"] for r in rows])
    c_i = np.array([r["c"]["is_sharpe"] for r in rows])
    b_i = np.array([r["b"]["is_sharpe"] for r in rows])
    d = c_s - b_s
    t, p = stats.ttest_rel(c_s, b_s)
    try:
        wp = stats.wilcoxon(c_s, b_s, zero_method="zsplit").pvalue
    except ValueError:
        wp = float("nan")
    print(f"\n=== {label} ({len(rows)} folds) ===")
    print(
        f"C tom-gated : IS {c_i.mean():+.2f} | OOS {c_s.mean():+.2f}+/-{c_s.std():.2f} "
        f"| degradation {np.mean(c_i - c_s):+.2f} | OOS>0 {int((c_s>0).sum())}/{len(rows)}"
    )
    print(
        f"B ungated   : IS {b_i.mean():+.2f} | OOS {b_s.mean():+.2f}+/-{b_s.std():.2f} "
        f"| degradation {np.mean(b_i - b_s):+.2f} | OOS>0 {int((b_s>0).sum())}/{len(rows)}"
    )
    print(
        f"paired OOS Sharpe (C-B): mean {d.mean():+.3f} sd {d.std(ddof=1):.3f}  "
        f"t={t:+.2f} p={p:.3f}  wilcoxon p={wp:.3f}  TOM better {int((d>0).sum())}/{len(d)}"
    )


def main():
    equity_rows, btc_rows = [], []
    for name, path, min_date, costs, is_primary in ASSETS:
        blocks = folds(load_fred(path, min_date))
        print(f"{name}: {len(blocks)} folds")
        for i, blk in enumerate(blocks):
            c = run_arm(blk, costs, TomGatedOptimizer)
            b = run_arm(blk, costs, MultiTimeframeOptimizer)
            if c is None or b is None:
                print(f"  fold {i}: arm failed, skipped")
                continue
            (equity_rows if is_primary else btc_rows).append(
                dict(asset=name, fold=i, c=c, b=b)
            )
            span = f"{blk['Datetime'].iloc[N_HISTORY].date()}..{blk['Datetime'].iloc[-1].date()}"
            print(
                f"  fold {i} [{span}] "
                f"C(tom) IS {c['is_sharpe']:+.2f} -> OOS {c['oos_sharpe']:+.2f} "
                f"({c['oos_trades']} tr) | "
                f"B(none) IS {b['is_sharpe']:+.2f} -> OOS {b['oos_sharpe']:+.2f} "
                f"({b['oos_trades']} tr)"
            )

    report(equity_rows, "PRIMARY: equity indices")
    report(btc_rows, "exploratory: BTC (no TOM prior)")


if __name__ == "__main__":
    main()
