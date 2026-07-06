"""Tests for Bayesian Online Change Point Detection and regime attribution"""

import numpy as np
import pandas as pd
import pytest

from models.bayesian_changepoint import (
    BOCPDDetector,
    RegimeStrategyAnalyzer,
    classify_segments,
    label_bars,
    label_runs,
    online_regime_labels,
)


@pytest.fixture
def shifted_returns():
    """Two clearly different regimes: quiet bull then volatile bear"""
    rng = np.random.default_rng(7)
    bull = rng.normal(0.0012, 0.006, 400)
    bear = rng.normal(-0.0025, 0.025, 400)
    return np.concatenate([bull, bear])


def test_detects_known_changepoint(shifted_returns):
    detector = BOCPDDetector(hazard_lambda=200.0, min_segment_bars=30)
    result = detector.detect(shifted_returns)

    assert len(result.changepoints) >= 1, "must detect the regime shift"
    # At least one change point within 50 bars of the true shift at 400
    assert any(abs(cp - 400) <= 50 for cp in result.changepoints), (
        f"no change point near 400, got {result.changepoints}"
    )


def test_no_false_positives_on_stationary_series():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.01, 800)
    detector = BOCPDDetector(hazard_lambda=400.0, min_segment_bars=30)
    result = detector.detect(returns)
    assert len(result.changepoints) <= 2, (
        f"too many change points on stationary data: {result.changepoints}"
    )


def test_detector_is_causal(shifted_returns):
    """Run-length posterior before the shift must not depend on future data"""
    detector = BOCPDDetector(hazard_lambda=200.0, min_segment_bars=30)
    full = detector.detect(shifted_returns)
    partial = detector.detect(shifted_returns[:300])
    assert np.allclose(full.cp_probability[:300], partial.cp_probability), (
        "cp probabilities before bar 300 changed when future data was added"
    )


def test_segment_classification(shifted_returns):
    detector = BOCPDDetector(hazard_lambda=200.0, min_segment_bars=30)
    result = detector.detect(shifted_returns)
    segments = classify_segments(shifted_returns, result.changepoints, 252.0)

    # Segments must partition the series
    assert segments[0].start_idx == 0
    assert segments[-1].end_idx == len(shifted_returns)
    for a, b in zip(segments[:-1], segments[1:]):
        assert a.end_idx == b.start_idx

    assert segments[0].regime == "Bull-Quiet"
    assert segments[-1].regime.startswith("Bear")
    assert segments[-1].ann_vol > segments[0].ann_vol


def test_label_bars(shifted_returns):
    detector = BOCPDDetector(hazard_lambda=200.0, min_segment_bars=30)
    result = detector.detect(shifted_returns)
    segments = classify_segments(shifted_returns, result.changepoints, 252.0)
    labels = label_bars(len(shifted_returns), segments)
    assert len(labels) == len(shifted_returns)
    assert (labels != "Unclassified").all()


def test_short_series_returns_no_changepoints():
    detector = BOCPDDetector(min_segment_bars=20)
    result = detector.detect(np.random.default_rng(0).normal(0, 0.01, 10))
    assert result.changepoints == []


def test_online_labels_are_causal(shifted_returns):
    """Live labels must not repaint: label[:k] identical on prefix vs full"""
    detector = BOCPDDetector(hazard_lambda=200.0, min_segment_bars=30)
    full_res = detector.detect(shifted_returns)
    full_labels = online_regime_labels(shifted_returns, full_res.map_run_length)

    for k in (200, 400, 600):
        prefix_res = detector.detect(shifted_returns[:k])
        prefix_labels = online_regime_labels(shifted_returns[:k], prefix_res.map_run_length)
        assert list(prefix_labels) == list(full_labels[:k]), (
            f"live labels repainted on the first {k} bars"
        )


def test_online_labels_update_bar_to_bar():
    """A section that starts bearish then reverses must be labeled bearish
    in real time BEFORE the reversal - the live label is not the ex-post one"""
    rng = np.random.default_rng(3)
    seg1 = rng.normal(0.0015, 0.006, 400)
    seg2 = np.concatenate(
        [rng.normal(-0.004, 0.02, 120), rng.normal(0.006, 0.02, 120)]
    )
    returns = np.concatenate([seg1, seg2])
    detector = BOCPDDetector(hazard_lambda=200.0, min_segment_bars=30)
    res = detector.detect(returns)
    labels = online_regime_labels(returns, res.map_run_length)

    # Early in the second section (bearish part) the live label is Bear...
    assert str(labels[470]).startswith("Bear")
    # ...even though by the end that same BOCPD segment reads bullish ex-post
    ex_post = classify_segments(returns, res.changepoints, 252.0)
    last_seg = ex_post[-1]
    if last_seg.start_idx <= 470 < last_seg.end_idx:
        assert last_seg.regime.startswith("Bull"), "test needs a reversing section"


def test_label_runs_partition():
    labels = np.array(["A", "A", "B", "B", "B", "A"], dtype=object)
    runs = label_runs(labels)
    assert runs == [(0, 2, "A"), (2, 5, "B"), (5, 6, "A")]


def test_regime_strategy_analyzer(shifted_returns):
    n = len(shifted_returns)
    detector = BOCPDDetector(hazard_lambda=200.0, min_segment_bars=30)
    result = detector.detect(shifted_returns)
    labels = online_regime_labels(shifted_returns, result.map_run_length)

    prices = 100 * np.cumprod(1 + shifted_returns)
    # Strategy that rides the asset at half exposure
    equity = 1000 * np.cumprod(1 + shifted_returns * 0.5)
    datetimes = pd.bdate_range("2022-01-03", periods=n).values

    trade_log = pd.DataFrame(
        {
            "Entry_Date": [datetimes[50], datetimes[500]],
            "Percent_Change": [2.5, -1.0],
        }
    )

    summary = RegimeStrategyAnalyzer.analyze(
        equity, prices, datetimes, labels, trade_log, 252.0
    )

    assert not summary.empty
    assert set(summary.columns) >= {
        "Regime",
        "Bars",
        "Time_%",
        "Strategy_Return_%",
        "BuyHold_Return_%",
        "Ann_Sharpe",
        "Max_DD_%",
        "Trades",
        "Win_Rate_%",
    }
    # Bars attributed across regimes cover the series minus the few
    # unclassified "Forming" bars at the very start
    assert n - 5 <= summary["Bars"].sum() <= n
    assert summary["Time_%"].sum() <= 100.5
    # Both trades assigned to some regime
    assert summary["Trades"].sum() == 2

    report = RegimeStrategyAnalyzer.build_report(summary, labels, datetimes, "TEST")
    assert "LIVE" in report.upper()
