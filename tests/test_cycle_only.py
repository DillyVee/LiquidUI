"""Tests for the time-cycle-only strategy mode (no indicators)"""

import os

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from config.settings import TransactionCosts
from signals.engine import CYCLE_ONLY, evaluate_combo, params_warmup
from trading.regime_switcher import StoredStrategy, run_switching_backtest


def _df_dict(n=400, seed=5):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.011, n))
    return {
        "daily": pd.DataFrame(
            {
                "Datetime": pd.bdate_range("2022-01-03", periods=n),
                "Open": np.concatenate([[100.0], closes[:-1]]),
                "Close": closes,
            }
        )
    }


def _optimizer(df_dict, cycle_only=True, n_trials=1):
    from optimization.optimizer import MultiTimeframeOptimizer

    return MultiTimeframeOptimizer(
        df_dict=df_dict,
        n_trials=n_trials,
        time_cycle_ranges=((1, 20), (0, 20), (0, 40)),
        mn1_range=(5, 20),
        mn2_range=(3, 10),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="CYCTEST",
        timeframes=["daily"],
        transaction_costs=TransactionCosts(),
        position_size=1.0,
        cycle_only=cycle_only,
    )


# ============================================================
# Signal engine
# ============================================================
def test_engine_cycle_only_no_indicator_conditions():
    closes = 100 + np.arange(200, dtype=float)
    params = {"IND1_daily": CYCLE_ONLY}
    entry_ok, exit_ok, warmup = evaluate_combo(closes, params, "daily")

    assert entry_ok.all(), "cycle-only entries must always be indicator-approved"
    assert not exit_ok.any(), "cycle-only must never emit an indicator exit"
    assert warmup == 0


def test_params_warmup_cycle_only_and_robustness():
    assert params_warmup({"IND1_daily": CYCLE_ONLY}, "daily") == 0
    # Params without any indicator keys must not raise (e.g. On/Off/Start only)
    assert params_warmup({"On_daily": 5, "Off_daily": 3}, "daily") == 0
    # Legacy path unchanged
    assert params_warmup({"MN1_daily": 10, "MN2_daily": 4}, "daily") == 14


def test_cycle_only_trades_follow_the_cycle_exactly():
    """With entries always allowed, positions must be open exactly during
    the ON phase (shifted one bar for next-open fills)"""
    df_dict = _df_dict()
    opt = _optimizer(df_dict)
    params = {
        "IND1_daily": CYCLE_ONLY,
        "On_daily": 5,
        "Off_daily": 5,
        "Start_daily": 0,
    }
    enter, exit_ = opt.compute_signals(params)

    anchor = opt.np_data["daily"]["anchor"]
    cycle = ((anchor - 0) % 10) < 5
    assert np.array_equal(enter, cycle)
    assert np.array_equal(exit_, ~cycle)

    eq_curve, trades = opt.simulate_multi_tf(params)
    assert eq_curve is not None
    assert trades > 5, "a 5-on/5-off cycle over 400 bars must trade repeatedly"


# ============================================================
# Optimizer end-to-end
# ============================================================
def test_optimizer_cycle_only_run_produces_cycle_only_params(tmp_path):
    opt = _optimizer(_df_dict(), cycle_only=True, n_trials=60)
    opt.run()

    assert opt.all_results, "cycle-only optimization must produce a result"
    best = opt.all_results[0]
    assert best["IND1_daily"] == CYCLE_ONLY
    assert "MN1_daily" not in best, "cycle-only result must carry no indicator params"
    for key in ("On_daily", "Off_daily", "Start_daily"):
        assert key in best
    # Full metric suite present regardless of mode
    for key in ("Sharpe_Ratio", "Sortino_Ratio", "Percent_Gain_%", "Objective"):
        assert key in best

    # The stored parameter set must be simulable as-is (book/live path)
    eq_curve, _ = opt.simulate_multi_tf(best)
    assert eq_curve is not None


def test_switching_backtest_accepts_cycle_only_strategy():
    """A cycle-only strategy competes in the regime-switching backtest"""
    df_dict = _df_dict()
    opt = _optimizer(df_dict)

    cyc = StoredStrategy(
        name="cyc",
        params={
            "IND1_daily": CYCLE_ONLY,
            "On_daily": 5, "Off_daily": 5, "Start_daily": 0,
        },
        base_score=0.5,
        objective="total_return",
    )
    rsi = StoredStrategy(
        name="rsi",
        params={
            "MN1_daily": 10, "MN2_daily": 3,
            "Entry_daily": 35.0, "Exit_daily": 65.0,
            "On_daily": 10, "Off_daily": 0, "Start_daily": 0,
        },
        base_score=0.4,
    )

    n = opt.np_data["daily"]["length"]
    labels = np.array(["Bull-Quiet"] * (n // 2) + ["Bear-Vol"] * (n - n // 2))

    result = run_switching_backtest(opt, [cyc, rsi], labels)
    assert len(result.equity_curve) == n
    assert set(result.shadow_curves) == {"cyc", "rsi"}
    assert np.all(np.isfinite(result.equity_curve))
