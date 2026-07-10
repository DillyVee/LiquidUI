"""Tests for the causal indicator library"""

import numpy as np
import pytest

from signals.indicators import (
    INDICATORS,
    compute_indicator,
    indicator_oscillator,
    indicator_warmup,
    rank_normalize,
)


@pytest.fixture
def closes():
    rng = np.random.default_rng(11)
    return 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 600))


@pytest.mark.parametrize("name", INDICATORS)
def test_indicator_shapes_and_finite(name, closes):
    raw = compute_indicator(name, closes, 14, 5)
    osc = indicator_oscillator(name, closes, 14, 5)
    assert raw.shape == closes.shape
    assert osc.shape == closes.shape
    assert np.all(np.isfinite(raw)), f"{name} produced non-finite raw values"
    assert np.all(np.isfinite(osc))
    assert osc.min() >= 0.0 and osc.max() <= 100.0


@pytest.mark.parametrize("name", INDICATORS)
def test_indicator_is_causal(name, closes):
    """Value at bar i must not change when future bars are appended"""
    full = indicator_oscillator(name, closes, 14, 5)
    partial = indicator_oscillator(name, closes[:400], 14, 5)
    assert np.allclose(full[:400], partial, atol=1e-9), (
        f"{name} repaints: past values changed when future data was added"
    )


@pytest.mark.parametrize("name", INDICATORS)
def test_warmup_is_sane(name):
    w = indicator_warmup(name, 14, 5)
    assert 10 <= w <= 500


def test_rank_normalize_basic():
    x = np.arange(100, dtype=float)  # strictly increasing
    r = rank_normalize(x, 20)
    # In an uptrend every value is the max of its window -> rank 100
    assert np.allclose(r[25:], 100.0)
    # And a new minimum ranks low
    y = np.concatenate([np.arange(50, dtype=float), [-100.0]])
    assert rank_normalize(y, 20)[-1] == pytest.approx(100.0 / 20)


def test_rank_normalize_is_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    full = rank_normalize(x, 50)
    part = rank_normalize(x[:200], 50)
    assert np.allclose(full[:200], part)


def test_rsi_matches_legacy_path(closes):
    """The library's rsi raw output must equal the legacy RSI+smooth chain"""
    from optimization.metrics import PerformanceMetrics

    legacy = PerformanceMetrics.smooth_vectorized(
        PerformanceMetrics.compute_rsi_vectorized(closes, 14), 5
    )
    assert np.allclose(compute_indicator("rsi", closes, 14, 5), legacy)


def test_directional_sanity():
    """Bounded oscillators read high after a rally; rank-normalized ones
    read high where the move is extreme vs recent history (rally onset)"""
    rng = np.random.default_rng(1)
    flat = 100 + rng.normal(0, 0.05, 300)
    rally = flat[-1] * np.cumprod(1 + np.abs(rng.normal(0.004, 0.002, 100)))
    closes = np.concatenate([flat, rally])

    for name in ("rsi", "stoch"):
        osc = indicator_oscillator(name, closes, 14, 5)
        assert osc[-1] > 60, f"{name} should read high after a rally, got {osc[-1]:.1f}"

    for name in ("roc", "emacross", "bbz", "macd"):
        osc = indicator_oscillator(name, closes, 14, 5)
        onset = osc[300:340]  # right after the rally starts at bar 300
        assert onset.max() > 90, (
            f"{name} should rank extreme at rally onset, max {onset.max():.1f}"
        )


def test_unknown_indicator_raises(closes):
    with pytest.raises(ValueError):
        compute_indicator("vibes", closes, 14, 5)
