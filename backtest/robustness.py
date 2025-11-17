"""
Robustness Testing Framework
Nested CV, walk-forward, parameter stability, Monte Carlo, and statistical validation
"""

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from infrastructure.logger import quant_logger

logger = quant_logger.get_logger("robustness")


@dataclass
class RobustnessResult:
    """Result from robustness testing"""

    test_name: str
    passed: bool
    score: float
    confidence_interval: Optional[Tuple[float, float]]
    p_value: Optional[float]
    details: Dict[str, Any]


class NestedCrossValidation:
    """
    Nested cross-validation for hyperparameter tuning and generalization testing

    Outer loop: True out-of-sample performance
    Inner loop: Hyperparameter optimization
    """

    def __init__(
        self,
        n_outer_splits: int = 5,
        n_inner_splits: int = 3,
        min_train_size: int = 252,  # 1 year of daily data
    ):
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.min_train_size = min_train_size

    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]],
        metric_func: Callable[[pd.Series], float],
    ) -> Dict[str, Any]:
        """
        Run nested cross-validation

        Args:
            data: Time series data
            strategy_func: Strategy function(data, **params) -> returns
            param_grid: Dictionary of parameter ranges
            metric_func: Function to evaluate performance (e.g., Sharpe ratio)

        Returns:
            Dictionary with results
        """
        logger.info(
            f"Starting nested CV: {self.n_outer_splits} outer, {self.n_inner_splits} inner splits"
        )

        outer_cv = TimeSeriesSplit(n_splits=self.n_outer_splits)
        inner_cv = TimeSeriesSplit(n_splits=self.n_inner_splits)

        outer_scores = []
        best_params_per_fold = []

        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
            outer_cv.split(data)
        ):
            logger.info(f"Outer fold {fold_idx + 1}/{self.n_outer_splits}")

            outer_train = data.iloc[outer_train_idx]
            outer_test = data.iloc[outer_test_idx]

            # Inner loop: hyperparameter optimization
            best_inner_score = -np.inf
            best_params = None

            param_combinations = list(itertools.product(*param_grid.values()))
            param_names = list(param_grid.keys())

            for params_tuple in param_combinations:
                params = dict(zip(param_names, params_tuple))

                inner_scores = []

                for inner_train_idx, inner_val_idx in inner_cv.split(outer_train):
                    inner_train = outer_train.iloc[inner_train_idx]
                    inner_val = outer_train.iloc[inner_val_idx]

                    try:
                        # Run strategy on validation set
                        returns = strategy_func(inner_train, inner_val, **params)
                        score = metric_func(returns)
                        inner_scores.append(score)

                    except Exception as e:
                        logger.warning(f"Inner fold failed with params {params}: {e}")
                        inner_scores.append(-np.inf)

                avg_inner_score = np.mean(inner_scores)

                if avg_inner_score > best_inner_score:
                    best_inner_score = avg_inner_score
                    best_params = params

            logger.info(
                f"Best params for fold {fold_idx}: {best_params}, score: {best_inner_score:.4f}"
            )
            best_params_per_fold.append(best_params)

            # Outer loop: test with best params
            try:
                returns = strategy_func(outer_train, outer_test, **best_params)
                outer_score = metric_func(returns)
                outer_scores.append(outer_score)
                logger.info(f"Outer fold {fold_idx} test score: {outer_score:.4f}")

            except Exception as e:
                logger.error(f"Outer fold {fold_idx} failed: {e}")
                outer_scores.append(np.nan)

        # Aggregate results
        outer_scores = np.array(outer_scores)
        valid_scores = outer_scores[~np.isnan(outer_scores)]

        results = {
            "mean_score": np.mean(valid_scores),
            "std_score": np.std(valid_scores),
            "min_score": np.min(valid_scores),
            "max_score": np.max(valid_scores),
            "all_scores": outer_scores.tolist(),
            "best_params_per_fold": best_params_per_fold,
            "n_successful_folds": len(valid_scores),
        }

        logger.info(
            f"Nested CV complete: mean={results['mean_score']:.4f}, "
            f"std={results['std_score']:.4f}"
        )

        return results


class WalkForwardAnalysis:
    """
    Walk-forward analysis with rolling optimization and out-of-sample testing
    """

    def __init__(
        self,
        train_window: int = 252,  # 1 year
        test_window: int = 63,  # 3 months
        step_size: int = 21,  # 1 month
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]],
        metric_func: Callable[[pd.Series], float],
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis

        Args:
            data: Time series data
            strategy_func: Strategy function
            param_grid: Parameter grid for optimization
            metric_func: Performance metric

        Returns:
            Results dictionary
        """
        logger.info(
            f"Starting walk-forward: train={self.train_window}, "
            f"test={self.test_window}, step={self.step_size}"
        )

        results = {
            "train_scores": [],
            "test_scores": [],
            "best_params": [],
            "train_periods": [],
            "test_periods": [],
        }

        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        start_idx = self.train_window

        while start_idx + self.test_window <= len(data):
            train_start = start_idx - self.train_window
            train_end = start_idx
            test_end = min(start_idx + self.test_window, len(data))

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[train_end:test_end]

            logger.info(
                f"Window: train {train_data.index[0]} to {train_data.index[-1]}, "
                f"test {test_data.index[0]} to {test_data.index[-1]}"
            )

            # Optimize on training window
            best_score = -np.inf
            best_params = None

            for params_tuple in param_combinations:
                params = dict(zip(param_names, params_tuple))

                try:
                    # Score on training set (could split into train/val)
                    returns = strategy_func(train_data, train_data, **params)
                    score = metric_func(returns)

                    if score > best_score:
                        best_score = score
                        best_params = params

                except Exception as e:
                    logger.warning(f"Param combination failed: {params}, {e}")
                    continue

            # Test on out-of-sample window
            try:
                test_returns = strategy_func(train_data, test_data, **best_params)
                test_score = metric_func(test_returns)

                results["train_scores"].append(best_score)
                results["test_scores"].append(test_score)
                results["best_params"].append(best_params)
                results["train_periods"].append(
                    (train_data.index[0], train_data.index[-1])
                )
                results["test_periods"].append(
                    (test_data.index[0], test_data.index[-1])
                )

                logger.info(
                    f"Train score: {best_score:.4f}, Test score: {test_score:.4f}, "
                    f"Params: {best_params}"
                )

            except Exception as e:
                logger.error(f"Test window failed: {e}")

            # Move forward
            start_idx += self.step_size

        # Calculate degradation (train vs test)
        train_scores = np.array(results["train_scores"])
        test_scores = np.array(results["test_scores"])

        results["avg_train_score"] = np.mean(train_scores)
        results["avg_test_score"] = np.mean(test_scores)
        results["score_degradation"] = (
            results["avg_train_score"] - results["avg_test_score"]
        )
        results["degradation_pct"] = (
            results["score_degradation"] / abs(results["avg_train_score"])
            if results["avg_train_score"] != 0
            else np.nan
        )

        logger.info(
            f"Walk-forward complete: {len(results['test_scores'])} windows, "
            f"avg test score: {results['avg_test_score']:.4f}, "
            f"degradation: {results['degradation_pct']*100:.1f}%"
        )

        return results


class MonteCarloValidation:
    """
    Monte Carlo validation via bootstrap and permutation tests
    """

    def __init__(self, n_simulations: int = 1000, random_seed: Optional[int] = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def bootstrap_returns(
        self, returns: pd.Series, metric_func: Callable[[pd.Series], float]
    ) -> Tuple[float, float, List[float]]:
        """
        Bootstrap confidence intervals for a performance metric

        Args:
            returns: Return series
            metric_func: Metric to calculate

        Returns:
            (mean, std, distribution)
        """
        logger.info(f"Running bootstrap with {self.n_simulations} simulations")

        bootstrap_metrics = []

        for i in range(self.n_simulations):
            # Resample with replacement
            sample_returns = returns.sample(n=len(returns), replace=True)
            metric = metric_func(sample_returns)
            bootstrap_metrics.append(metric)

        bootstrap_metrics = np.array(bootstrap_metrics)

        return (
            np.mean(bootstrap_metrics),
            np.std(bootstrap_metrics),
            bootstrap_metrics.tolist(),
        )

    def block_bootstrap(
        self,
        returns: pd.Series,
        block_size: int,
        metric_func: Callable[[pd.Series], float],
    ) -> Tuple[float, float, List[float]]:
        """
        Block bootstrap for time series (preserves autocorrelation)

        Args:
            returns: Return series
            block_size: Size of blocks to resample
            metric_func: Metric to calculate

        Returns:
            (mean, std, distribution)
        """
        logger.info(
            f"Running block bootstrap: block_size={block_size}, n_sims={self.n_simulations}"
        )

        n_obs = len(returns)
        n_blocks = int(np.ceil(n_obs / block_size))

        bootstrap_metrics = []

        for i in range(self.n_simulations):
            # Sample random blocks
            sampled_returns = []

            for _ in range(n_blocks):
                start_idx = np.random.randint(0, max(1, n_obs - block_size + 1))
                block = returns.iloc[start_idx : start_idx + block_size]
                sampled_returns.extend(block.values)

            # Trim to original length
            sampled_returns = pd.Series(sampled_returns[:n_obs])

            metric = metric_func(sampled_returns)
            bootstrap_metrics.append(metric)

        bootstrap_metrics = np.array(bootstrap_metrics)

        return (
            np.mean(bootstrap_metrics),
            np.std(bootstrap_metrics),
            bootstrap_metrics.tolist(),
        )

    def permutation_test(
        self,
        returns: pd.Series,
        metric_func: Callable[[pd.Series], float],
        null_hypothesis_value: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Permutation test for statistical significance

        Tests if the observed metric is significantly different from null hypothesis

        Args:
            returns: Return series
            metric_func: Metric to test
            null_hypothesis_value: Expected value under null (e.g., 0 for no alpha)

        Returns:
            (observed_metric, p_value)
        """
        logger.info("Running permutation test")

        observed_metric = metric_func(returns)

        # Generate null distribution by shuffling
        null_distribution = []

        for i in range(self.n_simulations):
            shuffled = returns.sample(frac=1.0).reset_index(drop=True)
            null_metric = metric_func(shuffled)
            null_distribution.append(null_metric)

        null_distribution = np.array(null_distribution)

        # Calculate p-value (two-tailed)
        p_value = np.mean(
            np.abs(null_distribution - null_hypothesis_value)
            >= np.abs(observed_metric - null_hypothesis_value)
        )

        logger.info(f"Observed: {observed_metric:.4f}, p-value: {p_value:.4f}")

        return observed_metric, p_value


class ParameterStabilityAnalyzer:
    """
    Analyze parameter stability and sensitivity
    """

    def sensitivity_analysis(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        base_params: Dict[str, Any],
        param_ranges: Dict[str, np.ndarray],
        metric_func: Callable[[pd.Series], float],
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on parameters

        Args:
            data: Time series data
            strategy_func: Strategy function
            base_params: Base parameter set
            param_ranges: Ranges to test for each parameter
            metric_func: Performance metric

        Returns:
            DataFrame with sensitivity results
        """
        logger.info("Running parameter sensitivity analysis")

        results = []

        for param_name, param_values in param_ranges.items():
            logger.info(f"Testing parameter: {param_name}")

            for value in param_values:
                # Create params with one parameter varied
                test_params = base_params.copy()
                test_params[param_name] = value

                try:
                    returns = strategy_func(data, data, **test_params)
                    score = metric_func(returns)

                    results.append(
                        {"parameter": param_name, "value": value, "score": score}
                    )

                except Exception as e:
                    logger.warning(f"Failed for {param_name}={value}: {e}")
                    results.append(
                        {"parameter": param_name, "value": value, "score": np.nan}
                    )

        return pd.DataFrame(results)

    def stability_score(
        self, sensitivity_df: pd.DataFrame, threshold_pct: float = 0.20
    ) -> Dict[str, float]:
        """
        Calculate stability score for each parameter

        Stability = what % of nearby parameter values achieve >80% of best performance

        Args:
            sensitivity_df: Output from sensitivity_analysis
            threshold_pct: Performance threshold (0.80 = 80% of best)

        Returns:
            Dictionary of stability scores per parameter
        """
        stability_scores = {}

        for param_name in sensitivity_df["parameter"].unique():
            param_data = sensitivity_df[sensitivity_df["parameter"] == param_name]
            scores = param_data["score"].dropna()

            if len(scores) == 0:
                stability_scores[param_name] = 0.0
                continue

            best_score = scores.max()
            threshold = best_score * (1 - threshold_pct)

            # What fraction of parameter values are above threshold?
            stable_fraction = (scores >= threshold).mean()
            stability_scores[param_name] = stable_fraction

        return stability_scores


class RegimeAnalysis:
    """
    Test strategy performance across different market regimes
    """

    @staticmethod
    def detect_volatility_regimes(
        returns: pd.Series, window: int = 60, n_regimes: int = 3
    ) -> pd.Series:
        """
        Detect volatility regimes

        Args:
            returns: Return series
            window: Rolling window for volatility
            n_regimes: Number of regimes (e.g., low/medium/high vol)

        Returns:
            Series with regime labels
        """
        vol = returns.rolling(window=window).std()

        # Quantile-based regimes
        if n_regimes == 2:
            quantiles = [0, 0.5, 1.0]
        elif n_regimes == 3:
            quantiles = [0, 0.33, 0.67, 1.0]
        else:
            quantiles = np.linspace(0, 1, n_regimes + 1)

        regime_thresholds = vol.quantile(quantiles)

        regimes = pd.cut(
            vol, bins=regime_thresholds, labels=range(n_regimes), include_lowest=True
        )

        return regimes

    @staticmethod
    def detect_trend_regimes(prices: pd.Series, window: int = 60) -> pd.Series:
        """
        Detect trend regimes (up/down/sideways)

        Args:
            prices: Price series
            window: Window for trend detection

        Returns:
            Series with regime labels: -1 (down), 0 (sideways), 1 (up)
        """
        sma = prices.rolling(window=window).mean()

        # Trend strength
        trend = (prices - sma) / sma

        regimes = pd.Series(index=prices.index, dtype=int)
        regimes[trend > 0.02] = 1  # Up trend
        regimes[trend < -0.02] = -1  # Down trend
        regimes[(trend >= -0.02) & (trend <= 0.02)] = 0  # Sideways

        return regimes

    def performance_by_regime(
        self, returns: pd.Series, regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate performance metrics by regime

        Args:
            returns: Strategy returns
            regimes: Regime labels

        Returns:
            DataFrame with performance by regime
        """
        results = []

        for regime_label in regimes.unique():
            if pd.isna(regime_label):
                continue

            regime_mask = regimes == regime_label
            regime_returns = returns[regime_mask]

            if len(regime_returns) < 10:  # Skip if too few observations
                continue

            metrics = {
                "regime": regime_label,
                "n_periods": len(regime_returns),
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "sharpe": (
                    regime_returns.mean() / regime_returns.std()
                    if regime_returns.std() > 0
                    else 0
                ),
                "hit_rate": (regime_returns > 0).mean(),
            }

            results.append(metrics)

        return pd.DataFrame(results)
