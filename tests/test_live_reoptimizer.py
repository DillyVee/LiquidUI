"""Tests for the gated live re-optimization cycle"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from config.settings import TransactionCosts
from trading.live_reoptimizer import run_reoptimization_cycle, split_train_holdout
from trading.regime_switcher import StrategyBook


def _df_dict(n=900, seed=5, drift=0.0004):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    return {
        "daily": pd.DataFrame(
            {
                "Datetime": pd.bdate_range("2021-01-04", periods=n),
                "Open": np.concatenate([[100.0], closes[:-1]]),
                "Close": closes,
            }
        )
    }


OPT_KWARGS = dict(
    time_cycle_ranges=((1, 20), (0, 20), (0, 40)),
    mn1_range=(5, 20),
    mn2_range=(3, 10),
    entry_range=(10.0, 40.0),
    exit_range=(50.0, 90.0),
    timeframes=["daily"],
    transaction_costs=TransactionCosts.for_stocks(),
    position_size=1.0,
)


def test_split_train_holdout_no_overlap():
    dd = _df_dict(400)
    train, hold = split_train_holdout(dd, holdout_fraction=0.25)
    t, h = train["daily"], hold["daily"]
    assert len(t) + len(h) == 400
    assert abs(len(h) - 100) <= 2
    assert t["Datetime"].max() < h["Datetime"].min(), "train must precede holdout"


def test_reopt_cycle_runs_and_gates(tmp_path):
    """A full cycle completes and produces a coherent gated verdict.
    On random-walk data with a small trial budget the candidate usually
    fails the gates - either way the invariants must hold."""
    book = StrategyBook(tmp_path / "book.json")
    result = run_reoptimization_cycle(
        _df_dict(),
        book,
        OPT_KWARGS,
        ticker="TEST",
        n_trials=80,
        holdout_fraction=0.25,
    )

    assert isinstance(result.message, str) and result.message
    if result.admitted:
        assert result.report is not None and result.report.passed
        assert len(book) == 1
        stored = book.strategies[0]
        assert stored.base_score == pytest.approx(result.report.dsr)
        assert "DSR" in stored.metrics
    else:
        # Nothing may enter the book when the gates fail
        assert len(book) == 0
        assert result.report is None or not result.report.passed or "beat" in result.message


def test_reopt_cycle_insufficient_data(tmp_path):
    book = StrategyBook(tmp_path / "book.json")
    result = run_reoptimization_cycle(
        _df_dict(200), book, OPT_KWARGS, ticker="TEST", n_trials=50
    )
    assert not result.admitted
    assert "not enough" in result.message
    assert len(book) == 0


def test_reopt_candidate_params_are_simulable(tmp_path):
    """Whatever the cycle admits (or would admit) must simulate cleanly on
    fresh data - i.e. the extracted candidate keys are complete"""
    from optimization.optimizer import MultiTimeframeOptimizer

    book = StrategyBook(tmp_path / "book.json")
    result = run_reoptimization_cycle(
        _df_dict(seed=9, drift=0.001),
        book,
        OPT_KWARGS,
        ticker="TEST",
        n_trials=80,
        # Loose gates so we exercise the admission path deterministically
        dsr_threshold=0.0,
        pbo_threshold=1.0,
        min_trades=1,
        min_oos_sharpe=-99.0,
        min_oos_trades=0,
    )
    assert result.admitted, f"loose gates should admit: {result.message}"

    # The 225-bar holdout was evaluated (with train warm-up context), so
    # the stored strategy must carry its OOS evidence count
    stored_metrics = book.strategies[0].metrics
    assert "OOS_Trades" in stored_metrics
    assert stored_metrics["OOS_Trades"] >= 0

    fresh = MultiTimeframeOptimizer(
        df_dict=_df_dict(seed=10), n_trials=1, ticker="FRESH", **OPT_KWARGS
    )
    eq, trades = fresh.simulate_multi_tf(book.strategies[0].params)
    assert eq is not None and len(eq) == 900
