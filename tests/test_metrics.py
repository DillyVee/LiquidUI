"""
Unit tests for performance metrics and PSR calculation
"""

import numpy as np

from optimization.metrics import PerformanceMetrics
from optimization.psr_composite import PSRCalculator


class TestPerformanceMetrics:
    def test_calculate_metrics_basic(self):
        eq_curve = np.linspace(1000.0, 1500.0, 100)
        metrics = PerformanceMetrics.calculate_metrics(eq_curve)

        assert metrics is not None
        assert metrics["Final_Equity_$"] == 1500.0
        assert metrics["Percent_Gain_%"] == 50.0
        assert metrics["Max_Drawdown_%"] == 0.0

    def test_calculate_metrics_with_drawdown(self):
        eq_curve = np.array([1000.0, 1200.0, 900.0, 1100.0])
        metrics = PerformanceMetrics.calculate_metrics(eq_curve)

        assert metrics is not None
        assert metrics["Max_Drawdown_%"] == -25.0  # 1200 -> 900

    def test_calculate_metrics_invalid(self):
        assert PerformanceMetrics.calculate_metrics(None) is None
        assert PerformanceMetrics.calculate_metrics(np.array([])) is None
        assert PerformanceMetrics.calculate_metrics(np.array([1000.0, np.nan])) is None
        assert PerformanceMetrics.calculate_metrics(np.array([1000.0, -5.0])) is None

    def test_sortino_uses_annualization_factor(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 500)
        eq_curve = 1000.0 * np.cumprod(1 + returns)

        daily = PerformanceMetrics.calculate_metrics(eq_curve, annualization_factor=252.0)
        hourly = PerformanceMetrics.calculate_metrics(eq_curve, annualization_factor=252.0 * 6.5)

        # Same curve annualized at a higher frequency must scale by sqrt of the ratio
        assert daily is not None and hourly is not None
        assert daily["Sortino_Ratio"] != 0
        ratio = hourly["Sortino_Ratio"] / daily["Sortino_Ratio"]
        assert abs(ratio - np.sqrt(6.5)) < 0.01

    def test_rsi_bounds(self):
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 300))
        rsi = PerformanceMetrics.compute_rsi_vectorized(prices, 14)

        assert len(rsi) == len(prices)
        assert np.all(rsi >= 0)
        assert np.all(rsi <= 100)

    def test_rsi_uptrend_high(self):
        prices = np.linspace(100, 200, 100)  # monotonic uptrend
        rsi = PerformanceMetrics.compute_rsi_vectorized(prices, 14)
        assert np.all(rsi[20:] > 95)

    def test_smooth_vectorized(self):
        arr = np.arange(10, dtype=float)
        smoothed = PerformanceMetrics.smooth_vectorized(arr, 3)
        assert len(smoothed) == len(arr)
        # Rolling mean of [2,3,4] is 3
        assert abs(smoothed[4] - 3.0) < 1e-12

    def test_buyhold_return(self):
        prices = np.array([100.0, 150.0])
        assert PerformanceMetrics.calculate_buyhold_return(prices) == 50.0
        assert PerformanceMetrics.calculate_buyhold_return(np.array([100.0])) == 0.0


class TestPSRCalculator:
    def test_psr_in_bounds(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.001, 0.01, 500)
        psr = PSRCalculator.calculate_psr(returns)
        assert 0.0 < psr < 1.0

    def test_psr_positive_returns_beats_negative(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(0, 0.01, 500)
        psr_pos = PSRCalculator.calculate_psr(noise + 0.002)
        psr_neg = PSRCalculator.calculate_psr(noise - 0.002)
        assert psr_pos > psr_neg

    def test_psr_insufficient_data(self):
        assert PSRCalculator.calculate_psr(np.array([0.01] * 5)) == 0.5
        assert PSRCalculator.calculate_psr(None) == 0.5

    def test_sharpe_from_equity(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.001, 0.01, 500)
        eq_curve = 1000.0 * np.cumprod(1 + returns)

        sharpe = PSRCalculator.calculate_sharpe_from_equity(eq_curve)
        assert -5 <= sharpe <= 10
        assert sharpe > 0  # positive drift

    def test_sharpe_from_flat_equity(self):
        assert PSRCalculator.calculate_sharpe_from_equity(np.full(100, 1000.0)) == 0.0
