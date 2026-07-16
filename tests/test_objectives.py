"""Tests for the optimization objective registry"""

import numpy as np
import pytest

from optimization.metrics import PerformanceMetrics
from optimization.objectives import (
    AUTO_BUILD_DEFAULTS,
    DEFAULT_OBJECTIVE,
    OBJECTIVES,
    get_objective,
    objective_score,
    scored,
    trade_confidence_multiplier,
)


@pytest.fixture
def metrics():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.001, 0.01, 400)
    eq_curve = 1000.0 * np.cumprod(1 + returns)
    trade_returns = rng.normal(0.4, 2.0, 60)
    return PerformanceMetrics.calculate_metrics(
        eq_curve, annualization_factor=252.0, trade_returns=trade_returns
    )


def test_registry_shape():
    assert DEFAULT_OBJECTIVE in OBJECTIVES
    for name in AUTO_BUILD_DEFAULTS:
        assert name in OBJECTIVES
    # Every advertised goal is resolvable and self-consistent
    for name, spec in OBJECTIVES.items():
        assert spec.name == name
        assert spec.label and spec.metric_key and spec.description
        assert get_objective(name) is spec


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="Unknown objective"):
        get_objective("alpha_to_the_moon")


@pytest.mark.parametrize("name", sorted(OBJECTIVES))
def test_every_objective_scores_from_metrics(name, metrics):
    value = objective_score(name, metrics)
    assert np.isfinite(value)
    # The score must be exactly the metric it claims to read
    assert value == pytest.approx(float(metrics[OBJECTIVES[name].metric_key]))


def test_missing_metrics_score_zero():
    assert objective_score("sharpe", None) == 0.0
    assert objective_score("sharpe", {}) == 0.0
    assert objective_score("win_rate", {"Sharpe_Ratio": 2.0}) == 0.0  # key absent
    assert objective_score("sharpe", {"Sharpe_Ratio": float("nan")}) == 0.0
    assert objective_score("sharpe", {"Sharpe_Ratio": float("inf")}) == 0.0


def test_trade_confidence_multiplier_gates_small_samples():
    assert trade_confidence_multiplier(0) == 0.0
    assert trade_confidence_multiplier(9) == 0.0
    assert trade_confidence_multiplier(20) == pytest.approx(0.5)
    assert trade_confidence_multiplier(30) == 1.0
    assert trade_confidence_multiplier(500) == 1.0


def test_scored_applies_multiplier(metrics):
    raw, score = scored("sharpe", metrics, trade_count=20)
    assert raw == pytest.approx(float(metrics["Sharpe_Ratio"]))
    assert score == pytest.approx(raw * 0.5)
    raw_full, score_full = scored("sharpe", metrics, trade_count=100)
    assert score_full == pytest.approx(raw_full)


def test_objectives_rank_better_strategies_higher():
    """A strategy with higher return and lower drawdown must outscore a
    losing one under every objective"""
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 0.008, 500)

    good_eq = 1000.0 * np.cumprod(1 + noise + 0.0015)
    bad_eq = 1000.0 * np.cumprod(1 + noise - 0.0015)
    good_trades = rng.normal(0.8, 1.0, 50)
    bad_trades = rng.normal(-0.8, 1.0, 50)

    good = PerformanceMetrics.calculate_metrics(good_eq, trade_returns=good_trades)
    bad = PerformanceMetrics.calculate_metrics(bad_eq, trade_returns=bad_trades)

    for name in OBJECTIVES:
        assert objective_score(name, good) > objective_score(name, bad), name
