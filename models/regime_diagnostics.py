"""
Enhanced Regime Detection Diagnostics
Transition Latency, False Transition Rate, Regime Persistence

Features:
1. Transition latency (how quickly do we detect regime shifts?)
2. False transition rate (how often do we incorrectly detect transitions?)
3. Regime persistence half-life
4. Stability metrics
5. Detection quality scores
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models.regime_detection import MarketRegime, MarketRegimeDetector


@dataclass
class TransitionDiagnostics:
    """Diagnostics for regime transitions"""

    transition_latency_days: float  # Average detection delay
    false_transition_rate: float  # Fraction of false transitions
    true_transition_rate: float  # Fraction of correctly detected transitions
    missed_transition_rate: float  # Fraction of missed transitions
    regime_persistence_halflife: Dict[
        MarketRegime, float
    ]  # Days until 50% prob of exit
    stability_score: float  # Overall stability (0-1, higher is more stable)
    transition_matrix: np.ndarray  # Empirical transition probabilities


@dataclass
class RegimeQualityMetrics:
    """Quality metrics for regime detection"""

    detection_accuracy: float  # How often is regime correct?
    average_confidence: float  # Average confidence score
    regime_purity: Dict[MarketRegime, float]  # How pure is each regime?
    confusion_matrix: np.ndarray  # Actual vs predicted regimes
    regime_durations: Dict[
        MarketRegime, List[int]
    ]  # Duration of each regime occurrence


class RegimeDiagnosticAnalyzer:
    """
    Comprehensive diagnostics for regime detection system

    Analyzes:
    - How quickly we detect transitions (latency)
    - How many false alarms we generate (FTR)
    - How stable regimes are (persistence)
    - Overall detection quality
    """

    def __init__(self, detector: MarketRegimeDetector):
        """
        Initialize diagnostic analyzer

        Args:
            detector: MarketRegimeDetector instance
        """
        self.detector = detector

    def calculate_transition_latency(
        self,
        detected_regimes: List[MarketRegime],
        true_transition_points: List[int],
        window_size: int = 10,
    ) -> float:
        """
        Calculate transition latency

        Measures how many periods it takes to detect a true regime shift

        Args:
            detected_regimes: Detected regime sequence
            true_transition_points: Indices where true transitions occur
            window_size: Look-ahead window for detecting transition

        Returns:
            Average latency in days
        """
        latencies = []

        for true_idx in true_transition_points:
            # Find when detector catches the transition
            # (within window_size periods after true transition)
            detected_idx = None

            for offset in range(window_size):
                check_idx = true_idx + offset
                if check_idx >= len(detected_regimes):
                    break

                # Check if regime changed
                if check_idx > 0:
                    if detected_regimes[check_idx] != detected_regimes[check_idx - 1]:
                        detected_idx = check_idx
                        break

            if detected_idx is not None:
                latency = detected_idx - true_idx
                latencies.append(latency)
            else:
                # Transition not detected within window
                latencies.append(window_size)  # Maximum latency

        if latencies:
            return float(np.mean(latencies))
        else:
            return 0.0

    def calculate_false_transition_rate(
        self,
        detected_regimes: List[MarketRegime],
        true_regimes: List[MarketRegime],
        min_duration: int = 5,
    ) -> Tuple[float, float, float]:
        """
        Calculate false transition rate

        A transition is "false" if:
        1. Detected regime changes but true regime doesn't
        2. Detected regime changes back quickly (whipsaw)

        Args:
            detected_regimes: Detected regime sequence
            true_regimes: Ground truth regime sequence
            min_duration: Minimum duration for valid regime (else considered noise)

        Returns:
            (false_transition_rate, true_transition_rate, missed_transition_rate)
        """
        # Count detected transitions
        detected_transitions = []
        for i in range(1, len(detected_regimes)):
            if detected_regimes[i] != detected_regimes[i - 1]:
                detected_transitions.append(i)

        # Count true transitions
        true_transitions = []
        for i in range(1, len(true_regimes)):
            if true_regimes[i] != true_regimes[i - 1]:
                true_transitions.append(i)

        # Classify detected transitions
        true_positives = 0  # Correctly detected transition
        false_positives = 0  # Incorrectly detected transition

        for det_idx in detected_transitions:
            # Check if there's a true transition nearby (within ±3 periods)
            is_true = any(abs(det_idx - true_idx) <= 3 for true_idx in true_transitions)

            if is_true:
                true_positives += 1
            else:
                false_positives += 1

        # Calculate missed transitions
        false_negatives = 0  # Missed true transition
        for true_idx in true_transitions:
            # Check if detected within ±3 periods
            is_detected = any(
                abs(true_idx - det_idx) <= 3 for det_idx in detected_transitions
            )
            if not is_detected:
                false_negatives += 1

        # Calculate rates
        total_detected = len(detected_transitions)
        total_true = len(true_transitions)

        false_transition_rate = (
            false_positives / total_detected if total_detected > 0 else 0.0
        )
        true_transition_rate = (
            true_positives / total_detected if total_detected > 0 else 0.0
        )
        missed_transition_rate = false_negatives / total_true if total_true > 0 else 0.0

        return false_transition_rate, true_transition_rate, missed_transition_rate

    def calculate_regime_persistence(
        self, regime_sequence: List[MarketRegime]
    ) -> Dict[MarketRegime, float]:
        """
        Calculate regime persistence half-life

        Half-life: Expected number of periods until 50% probability of exiting regime

        Args:
            regime_sequence: Sequence of detected regimes

        Returns:
            Dict mapping regime to half-life in periods
        """
        # Collect all regime durations
        regime_durations = {regime: [] for regime in MarketRegime}

        current_regime = regime_sequence[0]
        current_duration = 1

        for i in range(1, len(regime_sequence)):
            if regime_sequence[i] == current_regime:
                current_duration += 1
            else:
                # Regime changed
                regime_durations[current_regime].append(current_duration)
                current_regime = regime_sequence[i]
                current_duration = 1

        # Add final regime duration
        regime_durations[current_regime].append(current_duration)

        # Calculate half-life for each regime
        half_lives = {}

        for regime, durations in regime_durations.items():
            if durations:
                # Half-life ≈ median duration (50th percentile)
                half_life = float(np.median(durations))
                half_lives[regime] = half_life
            else:
                half_lives[regime] = 0.0

        return half_lives

    def calculate_stability_score(self, regime_sequence: List[MarketRegime]) -> float:
        """
        Calculate overall stability score

        Stable system:
        - Fewer transitions
        - Longer regime durations
        - Lower false transition rate

        Args:
            regime_sequence: Sequence of detected regimes

        Returns:
            Stability score (0-1, higher is better)
        """
        if len(regime_sequence) < 2:
            return 1.0

        # Count transitions
        n_transitions = sum(
            1
            for i in range(1, len(regime_sequence))
            if regime_sequence[i] != regime_sequence[i - 1]
        )

        # Calculate transition rate
        transition_rate = n_transitions / len(regime_sequence)

        # Stability is inverse of transition rate
        # Normalize to 0-1 range (assume transition_rate ∈ [0, 0.5])
        stability = max(0.0, 1.0 - 2 * transition_rate)

        return float(stability)

    def build_transition_matrix(
        self, regime_sequence: List[MarketRegime]
    ) -> np.ndarray:
        """
        Build empirical transition probability matrix

        Args:
            regime_sequence: Sequence of regimes

        Returns:
            Transition matrix [n_regimes, n_regimes]
            Entry (i, j) = P(regime_j | regime_i)
        """
        n_regimes = len(MarketRegime)
        transition_counts = np.zeros((n_regimes, n_regimes))

        # Map regimes to indices
        regime_to_idx = {regime: i for i, regime in enumerate(MarketRegime)}

        # Count transitions
        for i in range(len(regime_sequence) - 1):
            from_regime = regime_sequence[i]
            to_regime = regime_sequence[i + 1]

            from_idx = regime_to_idx[from_regime]
            to_idx = regime_to_idx[to_regime]

            transition_counts[from_idx, to_idx] += 1

        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        transition_matrix = transition_counts / row_sums

        return transition_matrix

    def calculate_regime_purity(
        self,
        regime_sequence: List[MarketRegime],
        returns: pd.Series,
        volatility: pd.Series,
    ) -> Dict[MarketRegime, float]:
        """
        Calculate regime purity

        Purity: How well does regime match its expected characteristics?

        For example:
        - BULL regime should have positive returns, moderate vol
        - CRISIS regime should have negative returns, high vol

        Args:
            regime_sequence: Detected regimes
            returns: Return series
            volatility: Volatility series

        Returns:
            Dict mapping regime to purity score (0-1)
        """
        purity_scores = {}

        # Expected characteristics
        regime_expectations = {
            MarketRegime.BULL: {
                "return_sign": 1,  # Positive
                "volatility": "low-moderate",  # <0.3
            },
            MarketRegime.BEAR: {
                "return_sign": -1,  # Negative
                "volatility": "moderate-high",  # >0.2
            },
            MarketRegime.HIGH_VOL: {
                "return_sign": 0,  # Any
                "volatility": "high",  # >0.4
            },
            MarketRegime.LOW_VOL: {
                "return_sign": 0,  # Any
                "volatility": "low",  # <0.15
            },
            MarketRegime.CRISIS: {
                "return_sign": -1,  # Negative
                "volatility": "very-high",  # >0.5
            },
        }

        for regime in MarketRegime:
            # Get periods in this regime
            mask = [r == regime for r in regime_sequence]
            if not any(mask):
                purity_scores[regime] = 0.0
                continue

            regime_returns = returns[mask]
            regime_volatility = volatility[mask]

            # Calculate purity based on expectations
            expectations = regime_expectations[regime]

            purity_components = []

            # Return sign purity
            if expectations["return_sign"] != 0:
                correct_sign = (
                    np.sign(regime_returns.mean()) == expectations["return_sign"]
                )
                purity_components.append(1.0 if correct_sign else 0.0)

            # Volatility purity
            avg_vol = regime_volatility.mean()
            if expectations["volatility"] == "low":
                vol_purity = (
                    1.0 if avg_vol < 0.15 else max(0.0, 1.0 - (avg_vol - 0.15) / 0.15)
                )
            elif expectations["volatility"] == "low-moderate":
                vol_purity = (
                    1.0 if avg_vol < 0.3 else max(0.0, 1.0 - (avg_vol - 0.3) / 0.2)
                )
            elif expectations["volatility"] == "moderate-high":
                vol_purity = 1.0 if avg_vol > 0.2 else max(0.0, avg_vol / 0.2)
            elif expectations["volatility"] == "high":
                vol_purity = 1.0 if avg_vol > 0.4 else max(0.0, avg_vol / 0.4)
            else:  # very-high
                vol_purity = 1.0 if avg_vol > 0.5 else max(0.0, avg_vol / 0.5)

            purity_components.append(vol_purity)

            # Overall purity
            purity_scores[regime] = float(np.mean(purity_components))

        return purity_scores

    def generate_full_diagnostic_report(
        self,
        detected_regimes: List[MarketRegime],
        true_regimes: List[MarketRegime],
        returns: pd.Series,
        volatility: pd.Series,
    ) -> TransitionDiagnostics:
        """
        Generate comprehensive diagnostic report

        Args:
            detected_regimes: Detected regime sequence
            true_regimes: Ground truth regimes
            returns: Return series
            volatility: Volatility series

        Returns:
            TransitionDiagnostics with all metrics
        """
        print(f"\n{'='*70}")
        print(f"REGIME DETECTION DIAGNOSTICS")
        print(f"{'='*70}")

        # Find true transition points
        true_transitions = [
            i
            for i in range(1, len(true_regimes))
            if true_regimes[i] != true_regimes[i - 1]
        ]

        # Calculate metrics
        latency = self.calculate_transition_latency(detected_regimes, true_transitions)

        ftr, ttr, mtr = self.calculate_false_transition_rate(
            detected_regimes, true_regimes
        )

        half_lives = self.calculate_regime_persistence(detected_regimes)

        stability = self.calculate_stability_score(detected_regimes)

        transition_matrix = self.build_transition_matrix(detected_regimes)

        print(f"\n📊 TRANSITION QUALITY:")
        print(f"   Latency: {latency:.1f} days (detection delay)")
        print(f"   False Transition Rate: {ftr:.1%} (false alarms)")
        print(f"   True Transition Rate: {ttr:.1%} (correctly detected)")
        print(f"   Missed Transition Rate: {mtr:.1%} (missed shifts)")

        print(f"\n⏱️  REGIME PERSISTENCE (Half-Life):")
        for regime, hl in sorted(half_lives.items(), key=lambda x: x[1], reverse=True):
            print(f"   {regime.value:15s}: {hl:.1f} days")

        print(f"\n📈 STABILITY SCORE: {stability:.1%}")

        print(f"\n{'='*70}\n")

        return TransitionDiagnostics(
            transition_latency_days=latency,
            false_transition_rate=ftr,
            true_transition_rate=ttr,
            missed_transition_rate=mtr,
            regime_persistence_halflife=half_lives,
            stability_score=stability,
            transition_matrix=transition_matrix,
        )


def assess_detection_quality(diagnostics: TransitionDiagnostics) -> str:
    """
    Assess overall quality of regime detection

    Args:
        diagnostics: TransitionDiagnostics result

    Returns:
        Quality assessment string
    """
    # Scoring rubric
    score = 0.0

    # Low latency is good
    if diagnostics.transition_latency_days < 2:
        score += 25
    elif diagnostics.transition_latency_days < 5:
        score += 15
    elif diagnostics.transition_latency_days < 10:
        score += 5

    # Low false transition rate is good
    if diagnostics.false_transition_rate < 0.2:
        score += 25
    elif diagnostics.false_transition_rate < 0.4:
        score += 15
    elif diagnostics.false_transition_rate < 0.6:
        score += 5

    # High true transition rate is good
    if diagnostics.true_transition_rate > 0.8:
        score += 25
    elif diagnostics.true_transition_rate > 0.6:
        score += 15
    elif diagnostics.true_transition_rate > 0.4:
        score += 5

    # High stability is good
    if diagnostics.stability_score > 0.7:
        score += 25
    elif diagnostics.stability_score > 0.5:
        score += 15
    elif diagnostics.stability_score > 0.3:
        score += 5

    # Overall assessment
    if score >= 80:
        quality = "EXCELLENT"
        emoji = "🏆"
        recommendation = "Regime detection is highly reliable. Safe for live trading."
    elif score >= 60:
        quality = "GOOD"
        emoji = "✅"
        recommendation = (
            "Regime detection is solid. Suitable for trading with monitoring."
        )
    elif score >= 40:
        quality = "ADEQUATE"
        emoji = "⚠️"
        recommendation = "Regime detection is acceptable but needs improvement."
    else:
        quality = "POOR"
        emoji = "❌"
        recommendation = (
            "Regime detection needs significant improvement before trading."
        )

    return f"""
{'='*70}
{emoji} DETECTION QUALITY: {quality} (Score: {score:.0f}/100)
{'='*70}

📊 Component Scores:
   Latency: {'✅' if diagnostics.transition_latency_days < 5 else '⚠️'} {diagnostics.transition_latency_days:.1f} days
   FTR: {'✅' if diagnostics.false_transition_rate < 0.3 else '⚠️'} {diagnostics.false_transition_rate:.1%}
   TTR: {'✅' if diagnostics.true_transition_rate > 0.7 else '⚠️'} {diagnostics.true_transition_rate:.1%}
   Stability: {'✅' if diagnostics.stability_score > 0.6 else '⚠️'} {diagnostics.stability_score:.1%}

💡 RECOMMENDATION:
   {recommendation}

{'='*70}
"""
