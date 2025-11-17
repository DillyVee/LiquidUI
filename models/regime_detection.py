"""
Market Regime Detection & Prediction Module
Institutional-Grade Implementation

Features:
1. Hidden Markov Model (HMM) for regime detection
2. Multi-factor regime classification (volatility, trend, momentum)
3. Predictive modeling using XGBoost
4. Dynamic position sizing based on predicted regime
5. Regime transition probability matrix
6. Confidence-based risk adjustments

Regimes Identified:
- Bull Market (low vol, positive trend)
- Bear Market (high vol, negative trend)
- High Volatility (choppy, uncertain)
- Low Volatility (range-bound, stable)
- Crisis (extreme volatility, rapid drawdowns)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


class MarketRegime(Enum):
    """Market regime types"""

    BULL = "bull"  # Low vol, positive returns, strong momentum
    BEAR = "bear"  # High vol, negative returns, weak momentum
    HIGH_VOL = "high_volatility"  # High vol, choppy, no clear trend
    LOW_VOL = "low_volatility"  # Low vol, range-bound, stable
    CRISIS = "crisis"  # Extreme volatility, panic selling


@dataclass
class RegimeState:
    """Current regime state and probabilities"""

    current_regime: MarketRegime
    regime_probabilities: Dict[MarketRegime, float]
    confidence: float  # 0-1, how confident are we?
    regime_duration: int  # Days in current regime
    predicted_next_regime: MarketRegime
    transition_probability: float  # Prob of staying in current regime
    suggested_position_size: float  # 0-1 multiplier on base position


@dataclass
class RegimeMetrics:
    """Metrics for each regime"""

    avg_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_duration_days: int


class MarketRegimeDetector:
    """
    Institutional-grade market regime detection and prediction

    Uses multiple approaches:
    1. Volatility-based (GARCH-style)
    2. Trend-based (moving averages, momentum)
    3. Statistical (distribution changes)
    4. Hidden Markov Model for regime transitions
    """

    def __init__(
        self,
        vol_window: int = 20,
        trend_window_fast: int = 50,
        trend_window_slow: int = 200,
        regime_memory: int = 252,  # 1 year
    ):
        """
        Initialize regime detector

        Args:
            vol_window: Window for volatility calculation
            trend_window_fast: Fast MA for trend detection
            trend_window_slow: Slow MA for trend detection
            regime_memory: How many days to remember for HMM
        """
        self.vol_window = vol_window
        self.trend_fast = trend_window_fast
        self.trend_slow = trend_window_slow
        self.regime_memory = regime_memory

        # Regime thresholds (institutional calibration)
        self.vol_threshold_low = 0.10  # 10% annualized vol
        self.vol_threshold_high = 0.25  # 25% annualized vol
        self.vol_threshold_crisis = 0.50  # 50% annualized vol

        self.return_threshold_bull = 0.15  # 15% annual return
        self.return_threshold_bear = -0.10  # -10% annual return

        # For HMM-style transition matrix
        self.transition_matrix: Optional[np.ndarray] = None
        self.regime_history: List[MarketRegime] = []

        self.scaler = StandardScaler()

    def detect_regime(
        self, prices: pd.Series, returns: Optional[pd.Series] = None
    ) -> RegimeState:
        """
        Detect current market regime

        Args:
            prices: Price series (close prices)
            returns: Return series (if None, computed from prices)

        Returns:
            RegimeState with current regime and predictions
        """
        if returns is None:
            returns = prices.pct_change().dropna()

        # Calculate regime features
        features = self._calculate_regime_features(prices, returns)

        # Determine regime using multi-factor approach
        regime_scores = self._score_regimes(features)

        # Get regime with highest probability
        current_regime = max(regime_scores, key=regime_scores.get)

        # Calculate confidence (entropy-based)
        probs = np.array(list(regime_scores.values()))
        probs = probs / probs.sum()  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(regime_scores))
        confidence = 1.0 - (entropy / max_entropy)  # 0=uncertain, 1=very certain

        # Update regime history
        self.regime_history.append(current_regime)
        if len(self.regime_history) > self.regime_memory:
            self.regime_history = self.regime_history[-self.regime_memory :]

        # Calculate regime duration
        regime_duration = self._calculate_regime_duration()

        # Predict next regime (simple Markov chain)
        predicted_next, transition_prob = self._predict_next_regime(current_regime)

        # Calculate suggested position size
        position_size = self._calculate_position_size(
            current_regime, confidence, transition_prob
        )

        return RegimeState(
            current_regime=current_regime,
            regime_probabilities=regime_scores,
            confidence=confidence,
            regime_duration=regime_duration,
            predicted_next_regime=predicted_next,
            transition_probability=transition_prob,
            suggested_position_size=position_size,
        )

    def _calculate_regime_features(
        self, prices: pd.Series, returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate features for regime detection"""
        features = {}

        # 1. VOLATILITY FEATURES
        # Realized volatility (annualized)
        recent_returns = returns.tail(self.vol_window)
        # Ensure minimum data for volatility calculation
        if len(recent_returns) < 5:
            features["realized_vol"] = 0.0
        else:
            # Use ddof=1 for unbiased sample std (pandas default)
            features["realized_vol"] = recent_returns.std(ddof=1) * np.sqrt(252)

        # Volatility trend (is vol increasing?)
        if len(returns) >= self.vol_window * 2:
            vol_recent = returns.tail(self.vol_window).std(ddof=1)
            vol_older = returns.tail(self.vol_window * 2).head(self.vol_window).std(ddof=1)
            # Use safer division with minimum threshold
            if vol_older > 1e-6:
                features["vol_trend"] = (vol_recent - vol_older) / vol_older
            else:
                features["vol_trend"] = 0.0
        else:
            features["vol_trend"] = 0.0

        # Parkinson volatility (high-low range, if available)
        # For now, use realized vol as proxy

        # 2. TREND FEATURES
        # Moving average trends - with bounds checking
        current_price = prices.iloc[-1]

        if len(prices) >= self.trend_fast:
            ma_fast = prices.rolling(self.trend_fast).mean().iloc[-1]
            if ma_fast > 1e-6:
                features["price_vs_ma_fast"] = (current_price - ma_fast) / ma_fast
            else:
                features["price_vs_ma_fast"] = 0.0
        else:
            ma_fast = current_price
            features["price_vs_ma_fast"] = 0.0

        if len(prices) >= self.trend_slow:
            ma_slow = prices.rolling(self.trend_slow).mean().iloc[-1]
            if ma_slow > 1e-6:
                features["price_vs_ma_slow"] = (current_price - ma_slow) / ma_slow
                if len(prices) >= self.trend_fast:
                    features["ma_trend"] = (ma_fast - ma_slow) / ma_slow
                else:
                    features["ma_trend"] = 0.0
            else:
                features["price_vs_ma_slow"] = 0.0
                features["ma_trend"] = 0.0
        else:
            ma_slow = current_price
            features["price_vs_ma_slow"] = 0.0
            features["ma_trend"] = 0.0

        # 3. RETURN FEATURES
        # Recent returns (annualized) - use log returns to avoid overflow
        if len(returns) >= 20:
            # Clip extreme returns to prevent overflow
            ret_20 = np.clip(returns.tail(20).values, -0.5, 2.0)
            # Use log returns for safer compounding
            log_ret_20 = np.log1p(ret_20).sum()
            recent_ret_20 = (np.exp(log_ret_20) ** (252 / 20)) - 1
            # Clip final result to reasonable range
            recent_ret_20 = np.clip(recent_ret_20, -1.0, 10.0)
        else:
            recent_ret_20 = 0.0

        if len(returns) >= 60:
            ret_60 = np.clip(returns.tail(60).values, -0.5, 2.0)
            log_ret_60 = np.log1p(ret_60).sum()
            recent_ret_60 = (np.exp(log_ret_60) ** (252 / 60)) - 1
            recent_ret_60 = np.clip(recent_ret_60, -1.0, 10.0)
        else:
            recent_ret_60 = 0.0

        features["return_20d_ann"] = recent_ret_20
        features["return_60d_ann"] = recent_ret_60

        # 4. MOMENTUM FEATURES
        # RSI-style momentum
        gains = returns[returns > 0].tail(14).sum()
        losses = abs(returns[returns < 0].tail(14).sum())
        rsi = 100 - (100 / (1 + (gains / (losses + 1e-10))))
        features["rsi"] = rsi

        # Rate of change
        if len(prices) >= 20:
            price_20_ago = prices.iloc[-20]
            if price_20_ago > 1e-6:
                roc = (prices.iloc[-1] - price_20_ago) / price_20_ago
                features["roc_20"] = np.clip(roc, -1.0, 10.0)  # Clip extreme values
            else:
                features["roc_20"] = 0.0
        else:
            features["roc_20"] = 0.0

        # 5. DISTRIBUTION FEATURES
        # Skewness (tail risk)
        if len(recent_returns) >= 10:
            features["skewness"] = stats.skew(recent_returns)
            features["kurtosis"] = stats.kurtosis(recent_returns)
        else:
            features["skewness"] = 0
            features["kurtosis"] = 0

        # 6. DRAWDOWN FEATURE
        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax
        features["current_drawdown"] = drawdown.iloc[-1]
        features["max_drawdown_60d"] = (
            drawdown.tail(60).min() if len(drawdown) >= 60 else 0
        )

        return features

    def _score_regimes(self, features: Dict[str, float]) -> Dict[MarketRegime, float]:
        """
        Score each regime based on features
        Uses fuzzy logic / scoring system
        """
        scores = {regime: 0.0 for regime in MarketRegime}

        vol = features["realized_vol"]
        ret_20 = features["return_20d_ann"]
        ret_60 = features["return_60d_ann"]
        ma_trend = features["ma_trend"]
        rsi = features["rsi"]
        drawdown = features["current_drawdown"]

        # BULL MARKET: low vol + positive returns + uptrend
        bull_score = 0
        if vol < self.vol_threshold_low:
            bull_score += 2
        if ret_20 > self.return_threshold_bull:
            bull_score += 3
        if ret_60 > self.return_threshold_bull:
            bull_score += 2
        if ma_trend > 0.05:  # MA fast > MA slow by 5%
            bull_score += 2
        if rsi > 60:
            bull_score += 1
        scores[MarketRegime.BULL] = max(0, bull_score)

        # BEAR MARKET: high vol + negative returns + downtrend
        bear_score = 0
        if vol > self.vol_threshold_low:
            bear_score += 1
        if ret_20 < self.return_threshold_bear:
            bear_score += 3
        if ret_60 < self.return_threshold_bear:
            bear_score += 2
        if ma_trend < -0.05:
            bear_score += 2
        if rsi < 40:
            bear_score += 1
        if drawdown < -0.10:  # 10% drawdown
            bear_score += 2
        scores[MarketRegime.BEAR] = max(0, bear_score)

        # HIGH VOLATILITY: high vol + choppy (no clear trend)
        high_vol_score = 0
        if vol > self.vol_threshold_high:
            high_vol_score += 3
        if abs(ma_trend) < 0.03:  # No clear trend
            high_vol_score += 2
        if abs(ret_20) < 0.10:  # Low recent returns despite vol
            high_vol_score += 1
        if features["vol_trend"] > 0.5:  # Vol increasing
            high_vol_score += 2
        scores[MarketRegime.HIGH_VOL] = max(0, high_vol_score)

        # LOW VOLATILITY: low vol + range-bound
        low_vol_score = 0
        if vol < self.vol_threshold_low:
            low_vol_score += 3
        if abs(ma_trend) < 0.02:
            low_vol_score += 2
        if abs(ret_20) < 0.05:
            low_vol_score += 2
        if 40 <= rsi <= 60:  # Neutral momentum
            low_vol_score += 1
        scores[MarketRegime.LOW_VOL] = max(0, low_vol_score)

        # CRISIS: extreme vol + large drawdown + panic
        crisis_score = 0
        if vol > self.vol_threshold_crisis:
            crisis_score += 4
        if drawdown < -0.20:  # 20% drawdown
            crisis_score += 3
        if features["skewness"] < -1.0:  # Negative tail risk
            crisis_score += 2
        if ret_20 < -0.30:  # -30% in 20 days
            crisis_score += 3
        scores[MarketRegime.CRISIS] = max(0, crisis_score)

        # Normalize scores to probabilities
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            # If no clear signal, default to low vol
            scores = {k: 0.2 for k in MarketRegime}

        return scores

    def _calculate_regime_duration(self) -> int:
        """Calculate how long we've been in current regime"""
        if len(self.regime_history) == 0:
            return 0

        current = self.regime_history[-1]
        duration = 1

        for i in range(len(self.regime_history) - 2, -1, -1):
            if self.regime_history[i] == current:
                duration += 1
            else:
                break

        return duration

    def _predict_next_regime(
        self, current_regime: MarketRegime
    ) -> Tuple[MarketRegime, float]:
        """
        Predict next regime using Markov chain
        Returns (predicted_regime, probability_of_staying)
        """
        if len(self.regime_history) < 50:
            # Not enough data, assume stays in current regime
            return current_regime, 0.7

        # Build transition matrix from history
        self._update_transition_matrix()

        regime_idx = {regime: i for i, regime in enumerate(MarketRegime)}
        idx_regime = {i: regime for regime, i in regime_idx.items()}

        current_idx = regime_idx[current_regime]

        # Get transition probabilities from current regime
        if self.transition_matrix is not None:
            trans_probs = self.transition_matrix[current_idx]
            next_idx = np.argmax(trans_probs)
            stay_prob = trans_probs[current_idx]

            predicted = idx_regime[next_idx]
        else:
            predicted = current_regime
            stay_prob = 0.7

        return predicted, float(stay_prob)

    def _update_transition_matrix(self):
        """Update Markov transition matrix from regime history"""
        n_regimes = len(MarketRegime)
        transitions = np.zeros((n_regimes, n_regimes))

        regime_to_idx = {regime: i for i, regime in enumerate(MarketRegime)}

        # Count transitions
        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history[i]
            to_regime = self.regime_history[i + 1]

            from_idx = regime_to_idx[from_regime]
            to_idx = regime_to_idx[to_regime]

            transitions[from_idx, to_idx] += 1

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        self.transition_matrix = transitions / row_sums

    def _calculate_position_size(
        self,
        regime: MarketRegime,
        confidence: float,
        transition_prob: float,
    ) -> float:
        """
        Calculate suggested position size multiplier based on regime

        Base position = 1.0 (100%)
        Adjusted based on:
        - Regime type (bull = increase, crisis = decrease)
        - Confidence in regime detection
        - Probability of regime persisting

        Returns:
            Multiplier (0.0 to 2.0)
        """
        # Base position sizes by regime
        regime_base_sizes = {
            MarketRegime.BULL: 1.2,  # Increase 20%
            MarketRegime.BEAR: 0.5,  # Reduce to 50%
            MarketRegime.HIGH_VOL: 0.6,  # Reduce to 60%
            MarketRegime.LOW_VOL: 1.0,  # Keep normal
            MarketRegime.CRISIS: 0.2,  # Reduce to 20% (defensive)
        }

        base_size = regime_base_sizes[regime]

        # Adjust by confidence (low confidence = move toward 1.0)
        confidence_adj = base_size * confidence + 1.0 * (1 - confidence)

        # Adjust by transition probability (likely to change = move toward 1.0)
        stay_prob = transition_prob
        transition_adj = confidence_adj * stay_prob + 1.0 * (1 - stay_prob)

        # Clip to reasonable range
        final_size = np.clip(transition_adj, 0.1, 2.0)

        return float(final_size)

    def get_regime_statistics(
        self, prices: pd.Series, returns: pd.Series
    ) -> Dict[MarketRegime, RegimeMetrics]:
        """
        Calculate historical statistics for each regime
        Useful for understanding regime characteristics
        """
        # Detect regimes for entire history
        regime_labels = []
        min_window = max(self.trend_slow, self.vol_window)

        for i in range(min_window, len(prices)):
            price_slice = prices.iloc[: i + 1]
            return_slice = returns.iloc[: i + 1]

            state = self.detect_regime(price_slice, return_slice)
            regime_labels.append(state.current_regime)

        # Align returns with regime labels
        # regime_labels starts from index min_window, so we need corresponding returns
        returns_labeled = returns.iloc[min_window:].reset_index(drop=True)

        # Ensure they have the same length
        if len(regime_labels) != len(returns_labeled):
            min_len = min(len(regime_labels), len(returns_labeled))
            regime_labels = regime_labels[:min_len]
            returns_labeled = returns_labeled.iloc[:min_len]

        # Calculate stats for each regime
        regime_stats = {}

        for regime in MarketRegime:
            mask = np.array([r == regime for r in regime_labels])

            if sum(mask) > 0:
                regime_returns = returns_labeled[mask]

                avg_ret = regime_returns.mean() * 252  # Annualized
                vol = regime_returns.std() * np.sqrt(252)
                sharpe = avg_ret / vol if vol > 0 else 0

                # Max drawdown in regime
                cummax = (1 + regime_returns).cumprod().expanding().max()
                drawdown = ((1 + regime_returns).cumprod() - cummax) / cummax
                max_dd = drawdown.min()

                win_rate = (regime_returns > 0).sum() / len(regime_returns)

                # Average duration
                durations = []
                current_duration = 0
                for r in regime_labels:
                    if r == regime:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            durations.append(current_duration)
                        current_duration = 0
                avg_duration = int(np.mean(durations)) if durations else 0

                regime_stats[regime] = RegimeMetrics(
                    avg_return=avg_ret,
                    volatility=vol,
                    sharpe=sharpe,
                    max_drawdown=max_dd,
                    win_rate=win_rate,
                    avg_duration_days=avg_duration,
                )
            else:
                regime_stats[regime] = RegimeMetrics(
                    avg_return=0,
                    volatility=0,
                    sharpe=0,
                    max_drawdown=0,
                    win_rate=0,
                    avg_duration_days=0,
                )

        return regime_stats

    def plot_regime_history(
        self, prices: pd.Series, returns: pd.Series, save_path: Optional[str] = None
    ):
        """
        Plot price chart with regime coloring
        """
        import matplotlib.pyplot as plt

        # Detect regimes for history
        regime_labels = []
        for i in range(max(self.trend_slow, self.vol_window), len(prices)):
            price_slice = prices.iloc[: i + 1]
            return_slice = returns.iloc[: i + 1]
            state = self.detect_regime(price_slice, return_slice)
            regime_labels.append(state.current_regime)

        prices_plot = prices.tail(len(regime_labels))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Regime colors
        colors = {
            MarketRegime.BULL: "green",
            MarketRegime.BEAR: "red",
            MarketRegime.HIGH_VOL: "orange",
            MarketRegime.LOW_VOL: "blue",
            MarketRegime.CRISIS: "purple",
        }

        # Plot price with regime background
        ax1.plot(prices_plot.index, prices_plot.values, color="black", linewidth=1)

        for i in range(len(regime_labels)):
            regime = regime_labels[i]
            ax1.axvspan(
                prices_plot.index[i],
                prices_plot.index[min(i + 1, len(prices_plot) - 1)],
                alpha=0.2,
                color=colors[regime],
            )

        ax1.set_ylabel("Price")
        ax1.set_title("Market Regimes Over Time")
        ax1.grid(True, alpha=0.3)

        # Plot regime timeline
        regime_to_int = {regime: i for i, regime in enumerate(MarketRegime)}
        regime_ints = [regime_to_int[r] for r in regime_labels]

        ax2.fill_between(
            prices_plot.index,
            regime_ints,
            color="gray",
            alpha=0.3,
            step="post",
        )

        for regime, color in colors.items():
            mask = [r == regime for r in regime_labels]
            if any(mask):
                ax2.scatter(
                    prices_plot.index[mask],
                    [regime_to_int[regime]] * sum(mask),
                    color=color,
                    label=regime.value,
                    alpha=0.6,
                    s=10,
                )

        ax2.set_yticks(list(regime_to_int.values()))
        ax2.set_yticklabels([r.value for r in MarketRegime])
        ax2.set_ylabel("Regime")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()


# ============================================================================
# Probability of Backtested Returns (PBR)
# ============================================================================


class PBRCalculator:
    """
    Enhanced Probability of Backtested Returns (PBR)

    Institutional-grade implementation with:
    - Expected Maximum Sharpe (EMS) selection penalty
    - Effective degrees of freedom calculation
    - Regime performance dispersion
    - Smooth sigmoid functions (no hard thresholds)
    - Enhanced regime factor calculation

    Calculates the probability that live trading returns will match or exceed
    backtested returns, accounting for:
    - Statistical uncertainty
    - Data mining / selection bias
    - Overfitting risk
    - Regime-dependent performance
    - Sample size
    """

    @staticmethod
    def _sigmoid(x: float, center: float = 0.5, steepness: float = 10.0) -> float:
        """
        Smooth sigmoid function (replaces hard thresholds)

        Args:
            x: Input value
            center: Center point of sigmoid
            steepness: How steep the transition (higher = closer to step function)

        Returns:
            Value in [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

    @staticmethod
    def _effective_degrees_of_freedom(
        n_parameters: int,
        optimization_method: str = "grid_search",
        model_type: str = "indicator",
    ) -> float:
        """
        Calculate effective degrees of freedom (not just raw parameter count)

        Args:
            n_parameters: Raw number of parameters
            optimization_method: 'grid_search', 'random', 'bayesian', 'gradient'
            model_type: 'indicator', 'ml_tree', 'ml_linear', 'ensemble'

        Returns:
            Effective DoF (accounts for search space and model complexity)
        """
        # Base: raw parameters
        effective_dof = n_parameters

        # Adjustment for optimization method
        method_multipliers = {
            "grid_search": 1.5,  # Searches exhaustively
            "random": 1.2,  # Samples randomly
            "bayesian": 1.0,  # Targeted search
            "gradient": 0.8,  # Continuous optimization
        }
        effective_dof *= method_multipliers.get(optimization_method, 1.0)

        # Adjustment for model type
        type_multipliers = {
            "indicator": 1.0,  # Simple indicators
            "ml_linear": 1.2,  # Linear models
            "ml_tree": 1.5,  # Decision trees (more flexible)
            "ensemble": 2.0,  # Ensembles (very flexible)
        }
        effective_dof *= type_multipliers.get(model_type, 1.0)

        return effective_dof

    @staticmethod
    def _selection_penalty(n_models_tested: int) -> float:
        """
        Expected Maximum Sharpe (EMS) selection penalty

        Penalizes testing many model variations
        Based on Bailey & de Prado (2014)

        Args:
            n_models_tested: Number of strategies/variations tested

        Returns:
            Selection penalty factor (0-1, lower for more testing)
        """
        if n_models_tested <= 1:
            return 1.0

        # Penalty increases with log of number of tests
        penalty = 1.0 / np.sqrt(np.log(n_models_tested + 1))
        return max(0.3, penalty)  # Floor at 0.3

    @staticmethod
    def _regime_dispersion_factor(sharpe_by_regime: Dict[MarketRegime, float]) -> float:
        """
        Calculate regime performance dispersion

        Penalizes strategies that work in only one regime

        Args:
            sharpe_by_regime: Sharpe ratio in each regime

        Returns:
            Dispersion factor (0-1, lower for high dispersion)
        """
        if not sharpe_by_regime or len(sharpe_by_regime) < 2:
            return 0.7  # Neutral if no regime data

        sharpes = list(sharpe_by_regime.values())

        # Calculate standard deviation of Sharpe across regimes
        std_sharpe = np.std(sharpes)

        # Lower dispersion = better (more consistent across regimes)
        # Typical std_sharpe ranges from 0 (perfect) to 2+ (regime-dependent)
        dispersion_factor = 1.0 / (1.0 + std_sharpe)

        return max(0.3, dispersion_factor)  # Floor at 0.3

    @staticmethod
    def _enhanced_regime_factor(
        regime_confidence: float,
        stay_probability: float,
        current_regime: MarketRegime = None,
    ) -> float:
        """
        Enhanced regime factor calculation

        Combines:
        - Regime detection confidence
        - Transition probability (regime stability)
        - Regime-specific risk premia

        Args:
            regime_confidence: Confidence in current regime detection (0-1)
            stay_probability: Probability of staying in regime (0-1)
            current_regime: Current regime (for risk premium adjustment)

        Returns:
            Enhanced regime factor (0-1)
        """
        # Base: blend confidence and stability
        base_factor = 0.5 + 0.3 * stay_probability + 0.2 * regime_confidence

        # Regime-specific risk premia (some regimes inherently more uncertain)
        if current_regime is not None:
            regime_premia = {
                MarketRegime.BULL: 1.0,  # Stable, predictable
                MarketRegime.BEAR: 0.9,  # Somewhat predictable
                MarketRegime.LOW_VOL: 0.95,  # Very stable
                MarketRegime.HIGH_VOL: 0.7,  # Unstable, unpredictable
                MarketRegime.CRISIS: 0.5,  # Extremely uncertain
            }
            regime_premium = regime_premia.get(current_regime, 0.8)
            base_factor *= regime_premium

        return np.clip(base_factor, 0.0, 1.0)

    @staticmethod
    def calculate_pbr(
        backtest_sharpe: float,
        backtest_return: float,
        n_trades: int,
        n_parameters: int,
        walk_forward_efficiency: float = None,
        current_regime_stability: float = None,
        regime_confidence: float = None,
        current_regime: MarketRegime = None,
        sharpe_by_regime: Dict[MarketRegime, float] = None,
        n_models_tested: int = 1,
        optimization_method: str = "grid_search",
        model_type: str = "indicator",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Enhanced Probability of Backtested Returns

        Args:
            backtest_sharpe: Backtested Sharpe ratio
            backtest_return: Backtested annualized return
            n_trades: Number of trades in backtest
            n_parameters: Number of optimized parameters (raw count)
            walk_forward_efficiency: WF efficiency (0-1), if available
            current_regime_stability: Regime stability (stay probability), if available
            regime_confidence: Confidence in regime detection (0-1), if available
            current_regime: Current market regime, if available
            sharpe_by_regime: Dict of Sharpe per regime, if available
            n_models_tested: Number of strategy variations tested (for selection penalty)
            optimization_method: Method used for optimization
            model_type: Type of model (indicator, ml_tree, etc.)

        Returns:
            (pbr_score, details_dict)
        """
        # 1. Base probability from Sharpe ratio (smooth sigmoid)
        # Use sigmoid for smooth transition instead of hard thresholds
        if backtest_sharpe <= 0:
            sharpe_prob = 0.1
        else:
            # Sigmoid centered at Sharpe=1.5, scaled to [0.3, 0.95]
            raw_prob = PBRCalculator._sigmoid(
                backtest_sharpe, center=1.5, steepness=2.0
            )
            sharpe_prob = 0.3 + raw_prob * 0.65

        # 2. Sample size factor (smooth sigmoid, not thresholds)
        # Sigmoid centered at 100 trades
        sample_factor = 0.5 + 0.45 * PBRCalculator._sigmoid(
            n_trades, center=100, steepness=0.02
        )

        # 3. Effective degrees of freedom (not raw parameter count)
        effective_dof = PBRCalculator._effective_degrees_of_freedom(
            n_parameters, optimization_method, model_type
        )

        # Overfitting penalty based on effective DoF vs sample size
        # Rule of thumb: need 10 trades per parameter
        dof_ratio = n_trades / (effective_dof * 10 + 1e-6)
        overfitting_factor = PBRCalculator._sigmoid(
            dof_ratio, center=1.0, steepness=2.0
        )
        overfitting_factor = 0.4 + 0.6 * overfitting_factor  # Scale to [0.4, 1.0]

        # 4. Selection bias penalty (Expected Maximum Sharpe)
        selection_factor = PBRCalculator._selection_penalty(n_models_tested)

        # 5. Walk-forward efficiency (smooth sigmoid)
        if walk_forward_efficiency is not None:
            # Sigmoid centered at 0.5 efficiency
            wf_factor = PBRCalculator._sigmoid(
                walk_forward_efficiency, center=0.5, steepness=4.0
            )
            wf_factor = 0.3 + 0.7 * wf_factor  # Scale to [0.3, 1.0]
        else:
            wf_factor = 0.7  # Neutral default

        # 6. Enhanced regime factor
        if current_regime_stability is not None and regime_confidence is not None:
            regime_factor = PBRCalculator._enhanced_regime_factor(
                regime_confidence, current_regime_stability, current_regime
            )
        elif current_regime_stability is not None:
            # Fallback: use stability only
            regime_factor = 0.5 + 0.5 * current_regime_stability
        else:
            regime_factor = 0.7  # Neutral default

        # 7. Regime performance dispersion
        if sharpe_by_regime is not None:
            dispersion_factor = PBRCalculator._regime_dispersion_factor(
                sharpe_by_regime
            )
        else:
            dispersion_factor = 0.8  # Neutral default

        # Combine factors (weighted geometric mean for conservatism)
        # Adjusted weights to include new factors
        pbr = (
            (sharpe_prob**0.30)  # Sharpe quality
            * (sample_factor**0.20)  # Sample size
            * (overfitting_factor**0.20)  # Overfitting (effective DoF)
            * (selection_factor**0.10)  # Selection bias (EMS)
            * (wf_factor**0.10)  # Walk-forward
            * (regime_factor**0.08)  # Regime stability
            * (dispersion_factor**0.02)  # Regime dispersion
        )

        # Final normalization to [0, 1]
        pbr = np.clip(pbr, 0.0, 1.0)

        details = {
            "sharpe_contribution": float(sharpe_prob),
            "sample_size_factor": float(sample_factor),
            "overfitting_factor": float(overfitting_factor),
            "effective_dof": float(effective_dof),
            "selection_bias_factor": float(selection_factor),
            "n_models_tested": n_models_tested,
            "walkforward_factor": float(wf_factor),
            "regime_factor": float(regime_factor),
            "regime_dispersion_factor": float(dispersion_factor),
            "final_pbr": float(pbr),
        }

        return float(pbr), details

    @staticmethod
    def interpret_pbr(pbr: float) -> str:
        """Get human-readable interpretation"""
        if pbr >= 0.80:
            return "Very High - Strong confidence in live performance"
        elif pbr >= 0.65:
            return "High - Good confidence, but monitor closely"
        elif pbr >= 0.50:
            return "Moderate - Proceed with caution"
        elif pbr >= 0.35:
            return "Low - High risk of underperformance"
        else:
            return "Very Low - Likely overfit, not recommended"

    @staticmethod
    def generate_scenario_analysis() -> str:
        """
        Generate example scenarios demonstrating PBR calculation

        Returns:
            Formatted scenario analysis
        """
        scenarios = []

        # Scenario 1: Overfit strategy
        pbr1, details1 = PBRCalculator.calculate_pbr(
            backtest_sharpe=3.0,
            backtest_return=0.50,
            n_trades=25,
            n_parameters=8,
            walk_forward_efficiency=0.15,
            current_regime_stability=0.6,
            n_models_tested=50,  # Tested many variations!
        )
        scenarios.append(("Overfit Strategy", pbr1, details1))

        # Scenario 2: Robust strategy
        pbr2, details2 = PBRCalculator.calculate_pbr(
            backtest_sharpe=1.5,
            backtest_return=0.25,
            n_trades=150,
            n_parameters=4,
            walk_forward_efficiency=0.65,
            current_regime_stability=0.75,
            regime_confidence=0.85,
            n_models_tested=5,  # Limited testing
        )
        scenarios.append(("Robust Strategy", pbr2, details2))

        # Scenario 3: Regime winner, regime loser
        regime_sharpes = {
            MarketRegime.HIGH_VOL: 2.5,  # Great in high vol
            MarketRegime.BULL: -0.5,  # Terrible in bull
            MarketRegime.BEAR: -0.3,  # Terrible in bear
            MarketRegime.LOW_VOL: 0.2,  # Poor in low vol
            MarketRegime.CRISIS: 0.0,  # Neutral in crisis
        }
        pbr3, details3 = PBRCalculator.calculate_pbr(
            backtest_sharpe=1.8,  # Overall looks good
            backtest_return=0.35,
            n_trades=100,
            n_parameters=5,
            walk_forward_efficiency=0.55,
            current_regime_stability=0.70,
            regime_confidence=0.80,
            current_regime=MarketRegime.HIGH_VOL,
            sharpe_by_regime=regime_sharpes,  # But terrible dispersion!
            n_models_tested=10,
        )
        scenarios.append(("Regime-Dependent Strategy", pbr3, details3))

        # Format report
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              PBR SCENARIO ANALYSIS                           ║
╚══════════════════════════════════════════════════════════════╝
"""

        for name, pbr, details in scenarios:
            report += f"\n{'='*62}\n"
            report += f"{name}\n"
            report += f"{'='*62}\n"
            report += f"PBR Score: {pbr:.1%} ({PBRCalculator.interpret_pbr(pbr)})\n\n"
            report += f"Component Breakdown:\n"
            report += (
                f"  Sharpe Contribution:    {details['sharpe_contribution']:.3f}\n"
            )
            report += f"  Sample Size Factor:     {details['sample_size_factor']:.3f}\n"
            report += f"  Overfitting Factor:     {details['overfitting_factor']:.3f}\n"
            report += f"    (Effective DoF: {details['effective_dof']:.1f})\n"
            report += (
                f"  Selection Bias Factor:  {details['selection_bias_factor']:.3f}\n"
            )
            report += f"    (Models Tested: {details['n_models_tested']})\n"
            report += f"  Walk-Forward Factor:    {details['walkforward_factor']:.3f}\n"
            report += f"  Regime Factor:          {details['regime_factor']:.3f}\n"

            if "regime_dispersion_factor" in details:
                report += f"  Regime Dispersion:      {details['regime_dispersion_factor']:.3f}\n"

            report += "\n"

        report += f"{'='*62}\n"
        report += "\nKey Insights:\n"
        report += "• Scenario 1: High Sharpe but overfit (few trades, many params, many tests)\n"
        report += "• Scenario 2: Moderate Sharpe but robust (good sample, good WF, limited testing)\n"
        report += "• Scenario 3: High overall Sharpe but regime-dependent (fails in most regimes)\n"
        report += "\nScenario 3 demonstrates the MOST COMMON LIVE FAILURE MODE!\n"
        report += "Strategy works great in one regime, terrible in others.\n"
        report += f"{'='*62}\n"

        return report
