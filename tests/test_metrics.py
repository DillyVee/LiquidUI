"""
Unit tests for performance metrics and PSR calculation
"""

import numpy as np
import pytest

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


class TestExtendedMetrics:
    """The institutional metric suite added for objective-based optimization"""

    @staticmethod
    def _curve(seed=42, drift=0.001, n=500):
        rng = np.random.default_rng(seed)
        returns = rng.normal(drift, 0.01, n)
        return 1000.0 * np.cumprod(1 + returns)

    def test_extended_keys_present(self):
        metrics = PerformanceMetrics.calculate_metrics(self._curve())
        for key in (
            "CAGR_%", "Sharpe_Ratio", "Calmar_Ratio", "Volatility_Ann_%",
            "Ulcer_Index", "VaR_95_%", "CVaR_95_%",
        ):
            assert key in metrics, key

    def test_trade_level_stats(self):
        trades = np.array([2.0, -1.0, 3.0, -1.0, 1.0])
        metrics = PerformanceMetrics.calculate_metrics(
            self._curve(), trade_returns=trades
        )
        assert metrics["Win_Rate_%"] == 60.0
        # Trade-based profit factor: (2+3+1) / (1+1) = 3
        assert metrics["Profit_Factor"] == 3.0
        assert metrics["Avg_Win_%"] == 2.0
        assert metrics["Avg_Loss_%"] == -1.0
        assert metrics["Payoff_Ratio"] == 2.0
        assert metrics["Expectancy_%"] == 0.8
        assert metrics["Best_Trade_%"] == 3.0
        assert metrics["Worst_Trade_%"] == -1.0

    def test_no_trade_stats_without_trade_returns(self):
        metrics = PerformanceMetrics.calculate_metrics(self._curve())
        assert "Win_Rate_%" not in metrics
        assert "Profit_Factor" in metrics  # per-bar fallback stays

    def test_cagr_matches_total_return_horizon(self):
        # 252 bars at 252/year = exactly one year: CAGR == total return
        eq = np.linspace(1000.0, 1210.0, 253)
        metrics = PerformanceMetrics.calculate_metrics(eq[1:], annualization_factor=252.0)
        assert abs(metrics["CAGR_%"] - metrics["Percent_Gain_%"]) < 1.0

    def test_calmar_sign_and_scale(self):
        metrics = PerformanceMetrics.calculate_metrics(self._curve(drift=0.002))
        assert metrics["Calmar_Ratio"] > 0
        losing = PerformanceMetrics.calculate_metrics(self._curve(drift=-0.002))
        assert losing["Calmar_Ratio"] < 0

    def test_var_cvar_ordering(self):
        metrics = PerformanceMetrics.calculate_metrics(self._curve())
        # Expected shortfall is at least as deep as VaR, both positive here
        assert metrics["CVaR_95_%"] >= metrics["VaR_95_%"] > 0

    def test_ratios_are_capped(self):
        # A nearly risk-free ramp must not produce infinite ratios
        eq = 1000.0 * np.cumprod(1 + np.full(300, 0.001))
        eq[100] *= 0.99999  # one microscopic down bar
        metrics = PerformanceMetrics.calculate_metrics(eq)
        assert np.isfinite(metrics["Sortino_Ratio"])
        assert abs(metrics["Sortino_Ratio"]) <= 100.0
        assert np.isfinite(metrics["Profit_Factor"])

    def test_sortino_is_target_downside_deviation(self):
        """Sortino must use the LPM2 target semideviation (MAR=0, over ALL
        bars - Sortino & van der Meer 1991), not the std of losing bars
        about their own mean. The latter rewards loss UNIFORMITY: a stream
        of identical -1% losses had downside std ~ 0 and rode the ratio to
        the 100.0 cap, which the optimizer could exploit."""
        rng = np.random.default_rng(0)
        wins = np.full(250, 0.012)
        # Same mean return, same gross loss; only the loss dispersion differs
        uniform_losses = np.full(250, -0.01) + rng.normal(0, 1e-5, 250)
        mixed_losses = np.concatenate([np.full(125, -0.002), np.full(125, -0.018)])

        def metrics_for(losses):
            r = np.empty(500)
            r[0::2] = wins
            r[1::2] = losses
            eq = 1000.0 * np.cumprod(1 + np.concatenate([[0.0], r]))
            m = PerformanceMetrics.calculate_metrics(eq)
            expected = (
                np.mean(r)
                / np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
                * np.sqrt(252.0)
            )
            return m["Sortino_Ratio"], expected

        got_u, exp_u = metrics_for(uniform_losses)
        got_m, exp_m = metrics_for(mixed_losses)

        assert got_u == pytest.approx(exp_u, abs=5e-3)
        assert got_m == pytest.approx(exp_m, abs=5e-3)
        # The uniform-loss stream must no longer explode to the cap; both
        # streams carry comparable downside risk and must score comparably
        assert got_u < 10.0
        assert abs(got_u - got_m) < 1.0

    def test_trades_per_year(self):
        # 505-bar curve at 252/year ~= 2 years; 5 trades -> 2.5/year
        eq = self._curve(n=505)
        trades = np.array([2.0, -1.0, 3.0, -1.0, 1.0])
        metrics = PerformanceMetrics.calculate_metrics(eq, trade_returns=trades)
        assert metrics["Trades_Per_Year"] == pytest.approx(5 / (505 / 252.0), abs=0.01)
        assert "Trades_Per_Year" not in PerformanceMetrics.calculate_metrics(eq)

    def test_exposure_metric(self):
        metrics = PerformanceMetrics.calculate_metrics(
            self._curve(), exposure_frac=0.375
        )
        assert metrics["Exposure_%"] == 37.5

        # Absent unless the simulation provides it; clipped to [0, 100]
        assert "Exposure_%" not in PerformanceMetrics.calculate_metrics(self._curve())
        capped = PerformanceMetrics.calculate_metrics(self._curve(), exposure_frac=1.7)
        assert capped["Exposure_%"] == 100.0

    def test_sortino_zero_without_downside(self):
        """No losing bars = no downside evidence, not infinite skill"""
        eq = 1000.0 * np.cumprod(1 + np.full(300, 0.001))
        metrics = PerformanceMetrics.calculate_metrics(eq)
        assert metrics["Sortino_Ratio"] == 0.0

    def test_buyhold_metrics(self):
        rng = np.random.default_rng(9)
        prices = 50 * np.cumprod(1 + rng.normal(0.0008, 0.012, 400))
        bh = PerformanceMetrics.calculate_buyhold_metrics(prices)
        assert bh is not None
        expected = (prices[-1] / prices[0] - 1) * 100
        assert bh["Percent_Gain_%"] == pytest.approx(expected, abs=0.01)
        assert PerformanceMetrics.calculate_buyhold_metrics(np.array([100.0])) is None
