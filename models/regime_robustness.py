"""
Robustness Testing for Regime Models
White's Reality Check, Hansen's SPA Test, Block Bootstrap

Features:
1. White's Reality Check (tests for data mining bias)
2. Hansen's Superior Predictive Ability (SPA) Test
3. Stationary Block Bootstrap (for autocorrelated data)
4. Multiple testing correction
5. Statistical significance validation
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class RobustnessTestResults:
    """Results from robustness testing"""

    test_statistic: float
    p_value: float
    is_significant: bool  # True if p < 0.05
    null_hypothesis: str
    test_name: str
    confidence_level: float
    interpretation: str


class WhiteRealityCheck:
    """
    White's Reality Check (2000)

    Tests whether a trading strategy's performance is due to skill or data mining

    Null Hypothesis: The best strategy found is no better than random
    Alternative: The strategy has genuine predictive power

    Uses bootstrap to simulate performance distribution under null
    """

    def __init__(self, n_bootstrap: int = 1000, block_size: int = 10):
        """
        Initialize White's Reality Check

        Args:
            n_bootstrap: Number of bootstrap samples
            block_size: Block size for stationary bootstrap (preserves autocorrelation)
        """
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size

    def test(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        alternative_strategies: Optional[List[np.ndarray]] = None,
    ) -> RobustnessTestResults:
        """
        Perform White's Reality Check

        Args:
            strategy_returns: Returns from strategy being tested [T]
            benchmark_returns: Returns from benchmark (e.g., buy-and-hold) [T]
            alternative_strategies: List of alternative strategies tested
                                   (for data mining adjustment)

        Returns:
            RobustnessTestResults with p-value and interpretation
        """
        # Calculate excess returns over benchmark
        excess_returns = strategy_returns - benchmark_returns

        # Test statistic: mean excess return (Sharpe would be alternative)
        observed_statistic = np.mean(excess_returns)

        print(f"\n{'='*70}")
        print(f"WHITE'S REALITY CHECK")
        print(f"{'='*70}")
        print(f"Observed excess return: {observed_statistic:.4f}")
        print(f"Running {self.n_bootstrap} bootstrap simulations...")

        # Bootstrap distribution under null (no skill)
        bootstrap_statistics = []

        for b in range(self.n_bootstrap):
            # Stationary block bootstrap (preserves autocorrelation)
            bootstrap_excess = self._stationary_block_bootstrap(excess_returns)

            # Recenter to enforce null hypothesis (mean = 0)
            bootstrap_excess_centered = bootstrap_excess - np.mean(bootstrap_excess)

            # Calculate statistic on bootstrap sample
            bootstrap_stat = np.mean(bootstrap_excess_centered)
            bootstrap_statistics.append(bootstrap_stat)

        bootstrap_statistics = np.array(bootstrap_statistics)

        # Calculate p-value: fraction of bootstrap samples >= observed
        p_value = np.mean(bootstrap_statistics >= observed_statistic)

        # Adjust for multiple testing if alternative strategies provided
        if alternative_strategies is not None:
            n_strategies = len(alternative_strategies) + 1
            # Bonferroni correction
            p_value_adjusted = min(p_value * n_strategies, 1.0)
            print(
                f"Data mining adjustment: {n_strategies} strategies tested, "
                f"p-value: {p_value:.4f} → {p_value_adjusted:.4f}"
            )
            p_value = p_value_adjusted

        is_significant = p_value < 0.05

        interpretation = self._interpret_result(
            observed_statistic, p_value, is_significant
        )

        print(f"\nP-value: {p_value:.4f}")
        print(f"Result: {'✅ SIGNIFICANT' if is_significant else '❌ NOT SIGNIFICANT'}")
        print(f"{'='*70}\n")

        return RobustnessTestResults(
            test_statistic=float(observed_statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            null_hypothesis="Strategy has no skill (excess return = 0)",
            test_name="White's Reality Check",
            confidence_level=0.95,
            interpretation=interpretation,
        )

    def _stationary_block_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """
        Stationary block bootstrap (Politis & Romano, 1994)

        Preserves autocorrelation structure in time series

        Args:
            data: Time series data

        Returns:
            Bootstrap sample
        """
        n = len(data)
        bootstrap_sample = []

        while len(bootstrap_sample) < n:
            # Random starting point
            start_idx = np.random.randint(0, n)

            # Random block length (geometric distribution)
            block_length = np.random.geometric(1.0 / self.block_size)
            block_length = min(block_length, n)

            # Extract block (with wraparound)
            for i in range(block_length):
                idx = (start_idx + i) % n
                bootstrap_sample.append(data[idx])

        return np.array(bootstrap_sample[:n])

    def _interpret_result(
        self, statistic: float, p_value: float, is_significant: bool
    ) -> str:
        """Generate interpretation of test result"""
        if is_significant:
            if statistic > 0:
                return (
                    f"✅ ROBUST: Strategy shows statistically significant outperformance "
                    f"(p={p_value:.3f}). Performance is unlikely due to data mining."
                )
            else:
                return (
                    f"❌ UNDERPERFORMANCE: Strategy significantly underperforms "
                    f"(p={p_value:.3f})."
                )
        else:
            return (
                f"⚠️  NOT ROBUST: Cannot reject null hypothesis (p={p_value:.3f}). "
                f"Performance may be due to luck or data mining."
            )


class HansenSPATest:
    """
    Hansen's Superior Predictive Ability (SPA) Test (2005)

    More powerful than White's Reality Check
    Tests whether best strategy is significantly better than benchmark

    Key improvement: Accounts for testing multiple strategies simultaneously
    """

    def __init__(self, n_bootstrap: int = 1000, block_size: int = 10):
        """
        Initialize Hansen's SPA Test

        Args:
            n_bootstrap: Number of bootstrap samples
            block_size: Block size for stationary bootstrap
        """
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size

    def test(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        all_strategies: List[np.ndarray],
    ) -> RobustnessTestResults:
        """
        Perform Hansen's SPA Test

        Args:
            strategy_returns: Returns from best strategy
            benchmark_returns: Benchmark returns
            all_strategies: All strategies tested (including losers)

        Returns:
            RobustnessTestResults
        """
        # Performance relative to benchmark for all strategies
        n_strategies = len(all_strategies)
        relative_performance = np.array(
            [strat - benchmark_returns for strat in all_strategies]
        )

        # Find best strategy
        mean_performance = np.mean(relative_performance, axis=1)
        best_idx = np.argmax(mean_performance)
        best_performance = mean_performance[best_idx]

        print(f"\n{'='*70}")
        print(f"HANSEN'S SUPERIOR PREDICTIVE ABILITY (SPA) TEST")
        print(f"{'='*70}")
        print(f"Number of strategies tested: {n_strategies}")
        print(f"Best strategy excess return: {best_performance:.4f}")
        print(f"Running {self.n_bootstrap} bootstrap simulations...")

        # Bootstrap under null (no strategy beats benchmark)
        bootstrap_max_stats = []

        for b in range(self.n_bootstrap):
            # Bootstrap each strategy
            bootstrap_perfs = []
            for strat_perf in relative_performance:
                boot_sample = self._stationary_block_bootstrap(strat_perf)
                # Recenter to enforce null
                boot_centered = boot_sample - np.mean(strat_perf)
                bootstrap_perfs.append(np.mean(boot_centered))

            # Maximum across strategies (accounts for multiple testing)
            max_stat = np.max(bootstrap_perfs)
            bootstrap_max_stats.append(max_stat)

        bootstrap_max_stats = np.array(bootstrap_max_stats)

        # P-value: fraction of bootstrap maxima >= observed best
        p_value = np.mean(bootstrap_max_stats >= best_performance)

        is_significant = p_value < 0.05

        interpretation = self._interpret_spa_result(
            best_performance, p_value, is_significant, n_strategies
        )

        print(f"\nP-value: {p_value:.4f}")
        print(f"Result: {'✅ SIGNIFICANT' if is_significant else '❌ NOT SIGNIFICANT'}")
        print(f"{'='*70}\n")

        return RobustnessTestResults(
            test_statistic=float(best_performance),
            p_value=float(p_value),
            is_significant=is_significant,
            null_hypothesis="No strategy beats benchmark",
            test_name="Hansen's SPA Test",
            confidence_level=0.95,
            interpretation=interpretation,
        )

    def _stationary_block_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """Stationary block bootstrap (same as White's RC)"""
        n = len(data)
        bootstrap_sample = []

        while len(bootstrap_sample) < n:
            start_idx = np.random.randint(0, n)
            block_length = np.random.geometric(1.0 / self.block_size)
            block_length = min(block_length, n)

            for i in range(block_length):
                idx = (start_idx + i) % n
                bootstrap_sample.append(data[idx])

        return np.array(bootstrap_sample[:n])

    def _interpret_spa_result(
        self, statistic: float, p_value: float, is_significant: bool, n_strategies: int
    ) -> str:
        """Generate interpretation"""
        if is_significant:
            return (
                f"✅ SUPERIOR PREDICTIVE ABILITY: Best strategy significantly "
                f"outperforms benchmark (p={p_value:.3f}) even after accounting for "
                f"{n_strategies} strategies tested. Performance is robust."
            )
        else:
            return (
                f"⚠️  NO SUPERIOR ABILITY: Cannot conclude that any strategy beats "
                f"benchmark (p={p_value:.3f}). Observed performance may be due to "
                f"data mining across {n_strategies} tested strategies."
            )


class BlockBootstrapValidator:
    """
    Overlapping Block Bootstrap for Time Series

    Generates confidence intervals that respect autocorrelation
    """

    def __init__(self, block_size: int = 10, n_bootstrap: int = 1000):
        """
        Initialize block bootstrap

        Args:
            block_size: Size of blocks (typically sqrt(T))
            n_bootstrap: Number of bootstrap samples
        """
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap

    def confidence_interval(
        self, data: np.ndarray, statistic_func=np.mean, confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval using block bootstrap

        Args:
            data: Time series data
            statistic_func: Function to calculate statistic (default: mean)
            confidence_level: Confidence level (0.95 = 95%)

        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        # Observed statistic
        observed = statistic_func(data)

        # Bootstrap distribution
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = self._block_bootstrap(data)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Percentile confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return observed, lower, upper

    def _block_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """
        Overlapping block bootstrap

        Args:
            data: Time series

        Returns:
            Bootstrap sample
        """
        n = len(data)
        n_blocks = int(np.ceil(n / self.block_size))

        bootstrap_sample = []
        for _ in range(n_blocks):
            # Random starting point (overlapping blocks)
            start = np.random.randint(0, max(1, n - self.block_size + 1))
            end = min(start + self.block_size, n)
            block = data[start:end]
            bootstrap_sample.extend(block)

        return np.array(bootstrap_sample[:n])

    def sharpe_ratio_ci(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate Sharpe ratio confidence interval

        Args:
            returns: Return series
            confidence_level: Confidence level

        Returns:
            (sharpe, lower_ci, upper_ci)
        """

        def sharpe_func(rets):
            if len(rets) < 2:
                return 0.0
            return np.mean(rets) / np.std(rets) * np.sqrt(252)

        return self.confidence_interval(returns, sharpe_func, confidence_level)


def run_full_robustness_suite(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    alternative_strategies: List[np.ndarray] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, RobustnessTestResults]:
    """
    Run complete robustness testing suite

    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        alternative_strategies: Other strategies tested
        n_bootstrap: Bootstrap samples

    Returns:
        Dict of test results
    """
    results = {}

    # 1. White's Reality Check
    wrc = WhiteRealityCheck(n_bootstrap=n_bootstrap)
    results["whites_rc"] = wrc.test(
        strategy_returns, benchmark_returns, alternative_strategies
    )

    # 2. Hansen's SPA Test (if alternatives provided)
    if alternative_strategies is not None:
        all_strategies = [strategy_returns] + alternative_strategies
        spa = HansenSPATest(n_bootstrap=n_bootstrap)
        results["hansen_spa"] = spa.test(
            strategy_returns, benchmark_returns, all_strategies
        )

    # 3. Sharpe Ratio Confidence Interval
    bootstrap = BlockBootstrapValidator(n_bootstrap=n_bootstrap)
    sharpe, sharpe_lower, sharpe_upper = bootstrap.sharpe_ratio_ci(
        strategy_returns - benchmark_returns
    )

    print(f"\n{'='*70}")
    print(f"SHARPE RATIO CONFIDENCE INTERVAL (95%)")
    print(f"{'='*70}")
    print(f"Point Estimate: {sharpe:.3f}")
    print(f"95% CI: [{sharpe_lower:.3f}, {sharpe_upper:.3f}]")
    print(f"{'='*70}\n")

    results["sharpe_ci"] = {
        "point_estimate": sharpe,
        "lower_bound": sharpe_lower,
        "upper_bound": sharpe_upper,
        "confidence_level": 0.95,
    }

    return results


def interpret_robustness_results(results: Dict[str, RobustnessTestResults]) -> str:
    """
    Generate summary interpretation of all robustness tests

    Args:
        results: Dict of test results

    Returns:
        Formatted interpretation string
    """
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║              ROBUSTNESS TESTING SUMMARY                      ║
╚══════════════════════════════════════════════════════════════╝
"""

    # Count significant tests
    significant_count = sum(
        1
        for r in results.values()
        if isinstance(r, RobustnessTestResults) and r.is_significant
    )
    total_tests = sum(
        1 for r in results.values() if isinstance(r, RobustnessTestResults)
    )

    report += f"\n✅ Significant Tests: {significant_count}/{total_tests}\n\n"

    for test_name, result in results.items():
        if isinstance(result, RobustnessTestResults):
            icon = "✅" if result.is_significant else "❌"
            report += f"{icon} {result.test_name}\n"
            report += f"   P-value: {result.p_value:.4f}\n"
            report += f"   {result.interpretation}\n\n"

    # Overall verdict
    if significant_count == total_tests:
        verdict = "✅ HIGHLY ROBUST: Strategy passes all statistical tests. Suitable for live trading."
    elif significant_count >= total_tests / 2:
        verdict = "⚠️  MODERATELY ROBUST: Strategy passes most tests. Use with caution and monitoring."
    else:
        verdict = "❌ NOT ROBUST: Strategy fails most tests. High risk of data mining. DO NOT TRADE."

    report += f"\n{'═'*62}\n"
    report += f"FINAL VERDICT:\n{verdict}\n"
    report += f"{'═'*62}\n"

    return report
