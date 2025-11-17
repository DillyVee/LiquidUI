"""
Multi-Horizon Regime Agreement Index
Institutional-Grade Consensus Scoring

Features:
1. Agreement across multiple time horizons
2. Consistency scoring
3. Signal strength based on horizon alignment
4. Whipsaw detection and filtering
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from models.regime_detection import MarketRegime


@dataclass
class HorizonPrediction:
    """Prediction at specific horizon"""

    horizon_days: int
    predicted_regime: MarketRegime
    confidence: float
    probabilities: Dict[MarketRegime, float]


@dataclass
class AgreementAnalysis:
    """Multi-horizon agreement analysis"""

    agreement_index: float  # 0-1, 1 = perfect agreement
    consensus_regime: MarketRegime  # Most agreed-upon regime
    consensus_strength: float  # How strongly regimes agree
    horizon_predictions: List[HorizonPrediction]
    disagreement_regimes: List[MarketRegime]  # Regimes with no consensus
    signal_quality: str  # 'strong', 'moderate', 'weak', 'conflicting'
    recommendation: str  # Trading recommendation based on agreement


class MultiHorizonAgreementIndex:
    """
    Calculates agreement index across multiple prediction horizons

    Key Insight:
    - If 1-day, 5-day, 20-day all predict BULL → strong signal
    - If short-term ≠ medium-term ≠ long-term → avoid allocation

    Used by institutions to reduce whipsaws and noise
    """

    def __init__(self, horizons: List[int] = None):
        """
        Initialize agreement calculator

        Args:
            horizons: List of prediction horizons in days
                     Default: [1, 5, 10, 20] (short, medium-short, medium, long)
        """
        self.horizons = horizons or [1, 5, 10, 20]

    def calculate_agreement(
        self, predictions: List[HorizonPrediction]
    ) -> AgreementAnalysis:
        """
        Calculate agreement index across horizons

        Args:
            predictions: List of predictions at different horizons

        Returns:
            AgreementAnalysis with consensus and strength
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 horizon predictions")

        # 1. Count regime votes across horizons
        regime_votes = {}
        confidence_weights = {}

        for pred in predictions:
            regime = pred.predicted_regime
            regime_votes[regime] = regime_votes.get(regime, 0) + 1

            # Weight by confidence
            if regime not in confidence_weights:
                confidence_weights[regime] = []
            confidence_weights[regime].append(pred.confidence)

        # 2. Calculate weighted agreement
        n_horizons = len(predictions)

        # Simple agreement: what fraction agree on most common regime?
        max_votes = max(regime_votes.values())
        simple_agreement = max_votes / n_horizons

        # Weighted agreement: account for confidence
        weighted_scores = {}
        for regime, votes in regime_votes.items():
            avg_confidence = np.mean(confidence_weights[regime])
            weighted_scores[regime] = votes * avg_confidence

        total_weighted = sum(weighted_scores.values())
        max_weighted = max(weighted_scores.values())
        weighted_agreement = max_weighted / total_weighted if total_weighted > 0 else 0

        # Combined agreement index (geometric mean)
        agreement_index = np.sqrt(simple_agreement * weighted_agreement)

        # 3. Determine consensus regime
        consensus_regime = max(weighted_scores, key=weighted_scores.get)

        # 4. Calculate consensus strength
        # How confident are we in the consensus?
        consensus_confidences = confidence_weights.get(consensus_regime, [0])
        consensus_strength = np.mean(consensus_confidences)

        # 5. Identify disagreement
        disagreement_regimes = [r for r in regime_votes.keys() if r != consensus_regime]

        # 6. Classify signal quality
        signal_quality = self._classify_signal_quality(
            agreement_index, consensus_strength, len(disagreement_regimes)
        )

        # 7. Generate recommendation
        recommendation = self._generate_recommendation(
            consensus_regime, agreement_index, consensus_strength, signal_quality
        )

        return AgreementAnalysis(
            agreement_index=float(agreement_index),
            consensus_regime=consensus_regime,
            consensus_strength=float(consensus_strength),
            horizon_predictions=predictions,
            disagreement_regimes=disagreement_regimes,
            signal_quality=signal_quality,
            recommendation=recommendation,
        )

    def _classify_signal_quality(
        self, agreement_index: float, consensus_strength: float, n_disagreements: int
    ) -> str:
        """
        Classify signal quality based on agreement metrics

        Args:
            agreement_index: Overall agreement (0-1)
            consensus_strength: Confidence in consensus
            n_disagreements: Number of disagreeing regimes

        Returns:
            Quality classification: 'strong', 'moderate', 'weak', 'conflicting'
        """
        if agreement_index >= 0.8 and consensus_strength >= 0.7:
            return "strong"
        elif agreement_index >= 0.6 and consensus_strength >= 0.6:
            return "moderate"
        elif n_disagreements >= 3:
            return "conflicting"
        else:
            return "weak"

    def _generate_recommendation(
        self,
        consensus_regime: MarketRegime,
        agreement_index: float,
        consensus_strength: float,
        signal_quality: str,
    ) -> str:
        """
        Generate trading recommendation based on agreement

        Args:
            consensus_regime: Most agreed-upon regime
            agreement_index: Agreement score
            consensus_strength: Confidence in consensus
            signal_quality: Quality classification

        Returns:
            Trading recommendation string
        """
        if signal_quality == "strong":
            if consensus_regime == MarketRegime.BULL:
                return "STRONG BUY: All horizons agree on bullish regime. Increase position size."
            elif consensus_regime == MarketRegime.BEAR:
                return "STRONG SELL: All horizons agree on bearish regime. Reduce exposure significantly."
            elif consensus_regime == MarketRegime.CRISIS:
                return "RISK OFF: All horizons signal crisis. Move to cash/defensive assets."
            elif consensus_regime == MarketRegime.HIGH_VOL:
                return "REDUCE SIZE: All horizons show high volatility. Tighten stops, reduce leverage."
            else:  # LOW_VOL
                return "NEUTRAL: Low volatility regime. Standard allocation, watch for breakout."

        elif signal_quality == "moderate":
            if consensus_regime == MarketRegime.BULL:
                return "MODERATE BUY: Majority bullish, but some uncertainty. Standard allocation."
            elif consensus_regime == MarketRegime.BEAR:
                return (
                    "MODERATE SELL: Majority bearish. Reduce position size moderately."
                )
            else:
                return "CAUTIOUS: Moderate agreement. Standard risk management, monitor closely."

        elif signal_quality == "conflicting":
            return (
                "NO POSITION: Conflicting signals across horizons. "
                "Avoid new positions until clarity emerges."
            )

        else:  # weak
            return (
                "WAIT: Weak agreement across horizons. "
                "Maintain current positions but avoid increasing exposure."
            )

    def calculate_horizon_stability(
        self, predictions: List[HorizonPrediction]
    ) -> float:
        """
        Calculate stability: how much do predictions change across horizons?

        Stable predictions (same regime across all horizons) = high stability
        Unstable predictions (regime changes frequently) = low stability

        Args:
            predictions: Ordered list of predictions (short to long horizon)

        Returns:
            Stability score (0-1, 1 = perfectly stable)
        """
        if len(predictions) < 2:
            return 1.0

        # Count regime transitions across horizons
        transitions = 0
        for i in range(len(predictions) - 1):
            if predictions[i].predicted_regime != predictions[i + 1].predicted_regime:
                transitions += 1

        # Stability is inverse of transition rate
        max_transitions = len(predictions) - 1
        stability = 1.0 - (transitions / max_transitions)

        return stability

    def detect_regime_shift(self, predictions: List[HorizonPrediction]) -> Dict:
        """
        Detect if a regime shift is occurring

        Pattern: Short-term predicts different regime than long-term
        Example: 1-day = BEAR, but 20-day = BULL → regime shift in progress

        Args:
            predictions: Ordered list of predictions (short to long horizon)

        Returns:
            Dict with shift detection results
        """
        if len(predictions) < 2:
            return {"shift_detected": False}

        short_term = predictions[0].predicted_regime  # 1-day
        long_term = predictions[-1].predicted_regime  # 20-day

        shift_detected = short_term != long_term

        if shift_detected:
            # Determine direction of shift
            if short_term == MarketRegime.BEAR and long_term == MarketRegime.BULL:
                direction = "deteriorating"
                message = "Short-term bearish, long-term bullish → regime deteriorating"
            elif short_term == MarketRegime.BULL and long_term == MarketRegime.BEAR:
                direction = "improving"
                message = "Short-term bullish, long-term bearish → regime improving"
            elif short_term == MarketRegime.CRISIS:
                direction = "crisis_entry"
                message = "Short-term crisis signal → entering crisis regime"
            elif long_term == MarketRegime.CRISIS:
                direction = "crisis_exit"
                message = "Long-term crisis but short-term recovery → exiting crisis"
            else:
                direction = "transitioning"
                message = f"Regime shift: {long_term.value} → {short_term.value}"

            return {
                "shift_detected": True,
                "direction": direction,
                "from_regime": long_term,
                "to_regime": short_term,
                "message": message,
            }
        else:
            return {
                "shift_detected": False,
                "stable_regime": short_term,
                "message": f"Stable {short_term.value} regime across all horizons",
            }

    def generate_report(self, analysis: AgreementAnalysis) -> str:
        """
        Generate human-readable agreement report

        Args:
            analysis: AgreementAnalysis result

        Returns:
            Formatted report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           MULTI-HORIZON AGREEMENT ANALYSIS                   ║
╚══════════════════════════════════════════════════════════════╝

📊 AGREEMENT INDEX: {analysis.agreement_index:.1%}
{'✅' if analysis.agreement_index > 0.7 else '⚠️' if analysis.agreement_index > 0.5 else '❌'} Signal Quality: {analysis.signal_quality.upper()}

🎯 CONSENSUS REGIME: {analysis.consensus_regime.value.upper()}
   Consensus Strength: {analysis.consensus_strength:.1%}

📈 HORIZON BREAKDOWN:
"""

        for pred in sorted(analysis.horizon_predictions, key=lambda x: x.horizon_days):
            marker = "✓" if pred.predicted_regime == analysis.consensus_regime else "✗"
            report += f"   {marker} {pred.horizon_days:2d}-day: {pred.predicted_regime.value:10s} (conf: {pred.confidence:.1%})\n"

        if analysis.disagreement_regimes:
            report += f"\n⚠️  DISAGREEMENTS:\n"
            for regime in analysis.disagreement_regimes:
                report += f"   • {regime.value}\n"

        report += f"\n💡 RECOMMENDATION:\n   {analysis.recommendation}\n"

        report += f"\n{'═'*62}\n"

        return report
