"""
H11: viability certification of the V1 candidate system.

Candidate (assembled from pre-registered evidence, journal iterations
1-7): optimize indicators-only (cycle pinned inert - H9/H10), then
deploy the fixed winner with the volatility overlay (H8b). Certified
against buy & hold on 35 folds across 6 assets, 20 of which (NIKKEI,
DJIA, WTI) were never used in any prior decision this session.

Pre-registered criteria (RESEARCH_JOURNAL.md, iteration 8):
  V1: pooled OOS Sharpe > 0, t-test AND Wilcoxon p < 0.05
  V2: paired MaxDD vs B&H significantly shallower (p < 0.05)
  V3: paired Sharpe vs B&H not significantly negative
  Generalization override: untouched-asset subset mean must be >= 0
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
NO_CYCLE_RANGES = ((250, 250), (0, 0), (0, 0))

# (name, file, min_date, costs, holdout) - holdout=True marks assets no
# prior session decision ever touched
ASSETS = [
    ("SP500", "SP500.csv", None, TransactionCosts.for_stocks, False),
    ("NASDAQ", "NASDAQCOM.csv", "1990-01-01", TransactionCosts.for_stocks, False),
    ("BTC", "CBBTCUSD.csv", None, TransactionCosts.for_crypto, False),
    ("NIKKEI", "NIKKEI225.csv", "1990-01-01", TransactionCosts.for_stocks, True),
    ("DJIA", "DJIA.csv", None, TransactionCosts.for_stocks, True),
    ("WTI", "DCOILWTICO.csv", "1990-01-01", TransactionCosts.for_stocks, True),
]


def load_fred(path, min_date=None):
    df = pd.read_csv(os.path.join(HERE, path), na_values=".")
    df.columns = ["Datetime", "Close"]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.dropna().reset_index(drop=True)
    df = df[df["Close"] > 0].reset_index(drop=True)  # WTI went negative 2020-04-20
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
        time_cycle_ranges=NO_CYCLE_RANGES,
        mn1_range=(5, 40),
        mn2_range=(3, 15),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="H11",
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


def run_fold(blk, costs):
    hist = {"daily": blk.iloc[:N_HISTORY].reset_index(drop=True)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = MultiTimeframeOptimizer(df_dict=hist, **opt_kwargs(costs, False))
        opt.run()
        if not opt.all_results:
            return None
        winner = {
            k: v
            for k, v in opt.all_results[0].items()
            if any(k.startswith(p) for p in PARAM_PREFIXES)
        }
        out = {}
        for tag, vol in (("cand", True), ("raw", False)):
            ev = MultiTimeframeOptimizer(
                df_dict={"daily": blk.reset_index(drop=True)},
                **opt_kwargs(costs, vol, n_trials=1),
            )
            eq, n_trades = ev.simulate_multi_tf(winner)
            if eq is None:
                return None
            oos = eq[N_HISTORY - 1 :]
            out[tag] = dict(s=sharpe_of(oos), dd=maxdd_of(oos), n=n_trades)
    closes = blk["Close"].to_numpy()[N_HISTORY - 1 :]
    out["bh"] = dict(s=sharpe_of(closes), dd=maxdd_of(closes))
    return out


def battery(rows, label):
    from scipy import stats

    c_s = np.array([r["res"]["cand"]["s"] for r in rows])
    c_d = np.array([r["res"]["cand"]["dd"] for r in rows])
    r_s = np.array([r["res"]["raw"]["s"] for r in rows])
    b_s = np.array([r["res"]["bh"]["s"] for r in rows])
    b_d = np.array([r["res"]["bh"]["dd"] for r in rows])

    print(f"\n=== {label} ({len(rows)} folds) ===")
    print(
        f"candidate : OOS Sharpe {c_s.mean():+.3f}+/-{c_s.std():.3f} "
        f"| MaxDD {c_d.mean():.2f}% | Sharpe>0 in {int((c_s>0).sum())}/{len(rows)}"
    )
    print(f"no-overlay: OOS Sharpe {r_s.mean():+.3f} (context)")
    print(f"buy&hold  : OOS Sharpe {b_s.mean():+.3f} | MaxDD {b_d.mean():.2f}%")

    t1, p1 = stats.ttest_1samp(c_s, 0.0)
    try:
        w1 = stats.wilcoxon(c_s, zero_method="zsplit").pvalue
    except ValueError:
        w1 = float("nan")
    print(f"V1 absolute: mean {c_s.mean():+.3f}  t={t1:+.2f} p={p1:.4f}  wilcoxon p={w1:.4f}")

    t2, p2 = stats.ttest_rel(c_d, b_d)
    try:
        w2 = stats.wilcoxon(c_d, b_d, zero_method="zsplit").pvalue
    except ValueError:
        w2 = float("nan")
    print(
        f"V2 drawdown vs B&H: paired {np.mean(c_d - b_d):+.2f}pp  "
        f"t={t2:+.2f} p={p2:.4f}  wilcoxon p={w2:.4f}  shallower in "
        f"{int((c_d > b_d).sum())}/{len(rows)}"
    )

    t3, p3 = stats.ttest_rel(c_s, b_s)
    print(
        f"V3 sharpe vs B&H: paired {np.mean(c_s - b_s):+.3f}  t={t3:+.2f} p={p3:.4f}"
        f"  (inferiority requires t<0 with p<0.05)"
    )
    return dict(c_s=c_s, p1=p1, w1=w1, dd_diff=np.mean(c_d - b_d), p2=p2,
                sh_diff=np.mean(c_s - b_s), t3=t3, p3=p3)


def main():
    rows = []
    for name, path, min_date, costs, holdout in ASSETS:
        blocks = folds(load_fred(path, min_date))
        print(f"{name}: {len(blocks)} folds{' [HOLDOUT]' if holdout else ''}")
        for i, blk in enumerate(blocks):
            res = run_fold(blk, costs)
            if res is None:
                print(f"  fold {i}: failed, skipped")
                continue
            rows.append(dict(asset=name, fold=i, holdout=holdout, res=res))
            span = f"{blk['Datetime'].iloc[N_HISTORY].date()}..{blk['Datetime'].iloc[-1].date()}"
            c, b = res["cand"], res["bh"]
            print(
                f"  fold {i} [{span}] cand OOS {c['s']:+.2f} DD {c['dd']:.1f}% "
                f"({c['n']} tr) | B&H {b['s']:+.2f} DD {b['dd']:.1f}%"
            )

    battery(rows, "POOLED: all folds")
    battery([r for r in rows if r["holdout"]], "HOLDOUT: untouched assets")
    battery([r for r in rows if not r["holdout"]], "original assets")


if __name__ == "__main__":
    main()
