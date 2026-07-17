"""
H8b: isolated volatility-sizing replay.

Per fold: optimize ONCE with fixed sizing (the default system), take the
winner, then replay it over the full fold twice - vol_targeting on and
off. Signals (and therefore trades) are identical; only per-trade size
differs, so the pairing isolates the sizing mechanism from TPE selection
noise. Scored on the OOS slice as in H7/H8.
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


def opt_kwargs(costs, vol, n_trials=N_TRIALS):
    return dict(
        n_trials=n_trials,
        time_cycle_ranges=((1, 60), (0, 60), (0, 120)),
        mn1_range=(5, 40),
        mn2_range=(3, 15),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="H8B",
        timeframes=["daily"],
        transaction_costs=costs(),
        position_size=1.0,
        seed=777,
        vol_targeting=vol,
    )


def sharpe_of(eq):
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ANN))


def maxdd_of(eq):
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0)) * 100.0


def main():
    rows = []
    for name, path, min_date, costs in ASSETS:
        blocks = folds(load_fred(path, min_date))
        print(f"{name}: {len(blocks)} folds")
        for i, blk in enumerate(blocks):
            hist = {"daily": blk.iloc[:N_HISTORY].reset_index(drop=True)}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                opt = MultiTimeframeOptimizer(df_dict=hist, **opt_kwargs(costs, False))
                opt.run()
                if not opt.all_results:
                    continue
                winner = {
                    k: v
                    for k, v in opt.all_results[0].items()
                    if any(k.startswith(p) for p in PARAM_PREFIXES)
                }
                res = {}
                for vol in (True, False):
                    ev = MultiTimeframeOptimizer(
                        df_dict={"daily": blk.reset_index(drop=True)},
                        **opt_kwargs(costs, vol, n_trials=1),
                    )
                    eq, n_trades = ev.simulate_multi_tf(winner)
                    if eq is None:
                        res = None
                        break
                    oos = eq[N_HISTORY - 1 :]
                    res[vol] = dict(
                        s=sharpe_of(oos), dd=maxdd_of(oos), n=n_trades
                    )
            if not res:
                print(f"  fold {i}: failed, skipped")
                continue
            if res[True]["n"] == 0:
                print(f"  fold {i}: winner never trades, skipped")
                continue
            rows.append(dict(asset=name, fold=i, r=res))
            print(
                f"  fold {i}: vol OOS {res[True]['s']:+.2f} DD {res[True]['dd']:.1f}% "
                f"| fix OOS {res[False]['s']:+.2f} DD {res[False]['dd']:.1f}% "
                f"({res[False]['n']} trades total)"
            )

    from scipy import stats

    a_s = np.array([r["r"][True]["s"] for r in rows])
    b_s = np.array([r["r"][False]["s"] for r in rows])
    a_d = np.array([r["r"][True]["dd"] for r in rows])
    b_d = np.array([r["r"][False]["dd"] for r in rows])

    print(f"\n=== isolated sizing replay, {len(rows)} folds ===")
    print(f"vol : OOS Sharpe {a_s.mean():+.3f}+/-{a_s.std():.3f} | MaxDD {a_d.mean():.2f}%")
    print(f"fix : OOS Sharpe {b_s.mean():+.3f}+/-{b_s.std():.3f} | MaxDD {b_d.mean():.2f}%")
    for label, x, y in (("Sharpe", a_s, b_s), ("MaxDD", a_d, b_d)):
        d = x - y
        nz = d[np.abs(d) > 1e-12]
        t, p = stats.ttest_rel(x, y)
        try:
            wp = stats.wilcoxon(x, y, zero_method="zsplit").pvalue
        except ValueError:
            wp = float("nan")
        print(
            f"paired {label} (vol-fix): mean {d.mean():+.3f}  t={t:+.2f} p={p:.3f} "
            f"wilcoxon p={wp:.3f}  vol better {int((d > 0).sum())}/{len(d)} "
            f"(nonzero diffs: {len(nz)})"
        )


if __name__ == "__main__":
    main()
