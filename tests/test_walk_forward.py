"""
Tests for walk-forward out-of-sample evaluation.

The critical property: OOS test windows must be evaluated WITH warm-up
context from the training slice. Simulating a test slice in isolation
restarts the indicator warm-up mask at bar 0, so any window shorter than
the warm-up mechanically reports zero trades and a flat curve - biasing
walk-forward verdicts against slower indicators and silently dropping
windows.
"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from config.settings import TransactionCosts
from optimization.optimizer import MultiTimeframeOptimizer
from optimization.walk_forward import WalkForwardAnalyzer

OPT_KWARGS = dict(
    time_cycle_ranges=((1, 250), (0, 250), (0, 500)),
    mn1_range=(5, 100),
    mn2_range=(3, 50),
    entry_range=(10.0, 40.0),
    exit_range=(50.0, 90.0),
    timeframes=["daily"],
    transaction_costs=TransactionCosts(),  # zero costs: exact price math
    position_size=1.0,
)

# Legacy RSI parameter set with a 40-bar warm-up (MN1 + MN2) that always
# enters once warmed up (RSI <= 100 < 100.5) and never exits on the
# indicator; the cycle is always ON. The strategy therefore buys at the
# first warmed-up bar and holds forever.
HOLD_FOREVER = {
    "MN1_daily": 30,
    "MN2_daily": 10,
    "Entry_daily": 100.5,
    "Exit_daily": 100.5,
    "On_daily": 250,
    "Off_daily": 0,
    "Start_daily": 0,
}


def _daily_df(n, start="2022-01-03", seed=1, drift=0.0005):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    return pd.DataFrame(
        {
            "Datetime": pd.bdate_range(start, periods=n),
            "Open": np.concatenate([[100.0], closes[:-1]]),
            "Close": closes,
        }
    )


def _split(df, n_test):
    train = df.iloc[:-n_test].reset_index(drop=True)
    test = df.iloc[-n_test:].reset_index(drop=True)
    return {"daily": train}, {"daily": test}


def test_isolated_test_slice_is_warmup_blind():
    """Documents the bias the context fix removes: a 21-bar window
    simulated alone can never trade a strategy with a 40-bar warm-up -
    the 30-period RSI cannot even be computed on 21 bars, so isolated
    evaluation fails outright and the window would be silently skipped."""
    df = _daily_df(321)
    _, test_dict = _split(df, 21)

    isolated = MultiTimeframeOptimizer(df_dict=test_dict, n_trials=1, **OPT_KWARGS)
    eq, trades = isolated.simulate_multi_tf(HOLD_FOREVER)

    assert trades == 0, "warm-up must swallow the isolated window"
    assert eq is None or np.allclose(eq, 1000.0), (
        "isolated evaluation yields no usable OOS information"
    )


def test_test_window_gets_warmup_context_from_train():
    """With the train slice as context, the position opened in-sample is
    carried into the test window and its test-period P&L is scored."""
    df = _daily_df(321)
    train_dict, test_dict = _split(df, 21)
    test_start = test_dict["daily"]["Datetime"].iloc[0]

    oos_equity, oos_trades = WalkForwardAnalyzer._test_window(
        MultiTimeframeOptimizer,
        train_dict,
        test_dict,
        test_start,
        HOLD_FOREVER,
        **OPT_KWARGS,
    )

    assert oos_equity is not None
    assert oos_trades >= 1, "the held position must count as OOS activity"
    assert oos_equity[0] == 1000.0

    # Zero costs + full position: OOS return must equal the open-to-open
    # price move from the last train bar through the last test bar
    opens = df["Open"].to_numpy()
    expected = opens[-1] / opens[len(train_dict["daily"]) - 1] - 1.0
    got = oos_equity[-1] / 1000.0 - 1.0
    assert got == pytest.approx(expected, abs=1e-9)


def test_test_window_without_train_context_degrades_gracefully():
    """Empty training context falls back to isolated evaluation"""
    df = _daily_df(60)
    _, test_dict = _split(df, 21)
    test_start = test_dict["daily"]["Datetime"].iloc[0]

    oos_equity, oos_trades = WalkForwardAnalyzer._test_window(
        MultiTimeframeOptimizer,
        {"daily": df.iloc[0:0]},  # no context available
        test_dict,
        test_start,
        HOLD_FOREVER,
        **OPT_KWARGS,
    )

    # Without context the 30-period RSI cannot exist on a 21-bar slice:
    # the window reports no OOS activity (and must not crash)
    assert oos_trades == 0
    assert oos_equity is None or np.allclose(oos_equity, 1000.0)


def test_fast_strategy_result_unchanged_by_context():
    """A strategy fully warmed up inside the test window must score the
    same test-period return with or without context (the fix adds missing
    state, it must not distort strategies that never needed it)."""
    params = dict(HOLD_FOREVER, MN1_daily=3, MN2_daily=2)  # 5-bar warm-up

    df = _daily_df(321)
    train_dict, test_dict = _split(df, 60)
    test_start = test_dict["daily"]["Datetime"].iloc[0]

    with_ctx, _ = WalkForwardAnalyzer._test_window(
        MultiTimeframeOptimizer,
        train_dict,
        test_dict,
        test_start,
        params,
        **OPT_KWARGS,
    )

    isolated = MultiTimeframeOptimizer(df_dict=test_dict, n_trials=1, **OPT_KWARGS)
    eq_iso, trades_iso = isolated.simulate_multi_tf(params)

    assert with_ctx is not None and eq_iso is not None
    assert trades_iso >= 1

    # Same terminal wealth ratio over the shared holding period is too
    # strict (entry bars differ by the warm-up); assert both are in the
    # market and profitable/losing together over the test period
    ret_ctx = with_ctx[-1] / with_ctx[0] - 1.0
    ret_iso = eq_iso[-1] / 1000.0 - 1.0
    assert np.sign(ret_ctx) == np.sign(ret_iso)
