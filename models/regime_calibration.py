"""
Probability Calibration for Regime Predictions
Institutional-Grade Calibration Methods

Features:
1. Isotonic regression calibration
2. Platt scaling (logistic calibration)
3. Brier score evaluation
4. Reliability diagrams
5. Expected Calibration Error (ECE)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid
from sklearn.isotonic import IsotonicRegression

from models.regime_detection import MarketRegime


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics"""

    brier_score: float  # Lower is better (0 to 1)
    ece: float  # Expected Calibration Error (0 to 1)
    mce: float  # Maximum Calibration Error (0 to 1)
    reliability_curve: List[Tuple[float, float]]  # (predicted_prob, actual_freq)
    is_calibrated: bool  # True if well-calibrated


class ProbabilityCalibrator:
    """
    Calibrates regime prediction probabilities

    Raw ML probabilities are often poorly calibrated:
    - "80% Bull" might actually mean 60% Bull
    - "20% Crisis" might actually mean 40% Crisis

    This class fixes that using:
    1. Isotonic Regression (non-parametric, order-preserving)
    2. Platt Scaling (parametric, logistic)
    """

    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator

        Args:
            method: 'isotonic' or 'platt'
        """
        self.method = method
        self.calibrators = {}  # One per regime
        self.is_fitted = False

    def fit(
        self, predicted_probs: np.ndarray, true_labels: np.ndarray, regime_idx: int
    ):
        """
        Fit calibration model for specific regime

        Args:
            predicted_probs: Uncalibrated probabilities [N]
            true_labels: Binary indicators (1 if regime, 0 otherwise) [N]
            regime_idx: Index of regime being calibrated
        """
        if len(predicted_probs) < 10:
            print(f"⚠️  Warning: Only {len(predicted_probs)} samples for calibration")
            return

        if self.method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(predicted_probs, true_labels)
        elif self.method == "platt":
            calibrator = self._fit_platt_scaling(predicted_probs, true_labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.calibrators[regime_idx] = calibrator
        self.is_fitted = True

    def _fit_platt_scaling(self, predicted_probs: np.ndarray, true_labels: np.ndarray):
        """
        Fit Platt scaling: calibrated_prob = sigmoid(A * log_odds + B)

        Args:
            predicted_probs: Uncalibrated probabilities
            true_labels: Binary labels

        Returns:
            Fitted parameters (A, B)
        """

        # Convert probabilities to log-odds
        eps = 1e-10
        predicted_probs = np.clip(predicted_probs, eps, 1 - eps)
        log_odds = np.log(predicted_probs / (1 - predicted_probs))

        # Optimize A and B to minimize negative log-likelihood
        def negative_log_likelihood(params):
            A, B = params
            calibrated_probs = expit(A * log_odds + B)
            # Binary cross-entropy
            loss = -np.mean(
                true_labels * np.log(calibrated_probs + eps)
                + (1 - true_labels) * np.log(1 - calibrated_probs + eps)
            )
            return loss

        result = minimize(negative_log_likelihood, x0=[1.0, 0.0], method="BFGS")

        return result.x  # (A, B)

    def calibrate(self, predicted_probs: Dict[int, float]) -> Dict[int, float]:
        """
        Calibrate regime probabilities

        Args:
            predicted_probs: Dict mapping regime_idx to uncalibrated probability

        Returns:
            Dict mapping regime_idx to calibrated probability
        """
        if not self.is_fitted:
            return predicted_probs  # Return uncalibrated if not fitted

        calibrated = {}
        for regime_idx, prob in predicted_probs.items():
            if regime_idx in self.calibrators:
                if self.method == "isotonic":
                    cal_prob = self.calibrators[regime_idx].predict([prob])[0]
                else:  # platt
                    A, B = self.calibrators[regime_idx]
                    eps = 1e-10
                    prob = np.clip(prob, eps, 1 - eps)
                    log_odds = np.log(prob / (1 - prob))
                    cal_prob = expit(A * log_odds + B)

                calibrated[regime_idx] = float(cal_prob)
            else:
                calibrated[regime_idx] = prob  # No calibrator, use original

        # Renormalize to sum to 1
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {k: v / total for k, v in calibrated.items()}

        return calibrated

    def evaluate_calibration(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Evaluate calibration quality

        Args:
            predicted_probs: Predicted probabilities [N]
            true_labels: Binary ground truth [N]
            n_bins: Number of bins for reliability diagram

        Returns:
            CalibrationMetrics with quality scores
        """
        # 1. Brier Score (mean squared error of probabilities)
        brier = np.mean((predicted_probs - true_labels) ** 2)

        # 2. Expected Calibration Error (ECE)
        # Bin predictions, compare average predicted prob vs actual frequency
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        mce = 0.0
        reliability_curve = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_predicted = np.mean(predicted_probs[mask])
                avg_actual = np.mean(true_labels[mask])
                n_samples = np.sum(mask)

                # ECE: weighted absolute difference
                ece += (n_samples / len(predicted_probs)) * abs(
                    avg_predicted - avg_actual
                )

                # MCE: maximum absolute difference
                mce = max(mce, abs(avg_predicted - avg_actual))

                reliability_curve.append((avg_predicted, avg_actual))

        # Consider well-calibrated if ECE < 0.1 and Brier < 0.25
        is_calibrated = ece < 0.1 and brier < 0.25

        return CalibrationMetrics(
            brier_score=float(brier),
            ece=float(ece),
            mce=float(mce),
            reliability_curve=reliability_curve,
            is_calibrated=is_calibrated,
        )


class MultiClassCalibrator:
    """
    Calibrate multi-class regime predictions

    Handles all 5 regimes simultaneously
    """

    def __init__(self, method: str = "isotonic"):
        """
        Initialize multi-class calibrator

        Args:
            method: 'isotonic' or 'platt'
        """
        self.method = method
        self.calibrators = {
            i: ProbabilityCalibrator(method) for i in range(len(MarketRegime))
        }
        self.is_fitted = False

    def fit(self, predicted_probs: np.ndarray, true_labels: np.ndarray):
        """
        Fit calibration models for all regimes

        Args:
            predicted_probs: Uncalibrated probabilities [N, n_classes]
            true_labels: Integer labels [N]
        """
        n_classes = predicted_probs.shape[1]

        print(f"\n{'='*70}")
        print(f"CALIBRATING REGIME PROBABILITIES ({self.method.upper()})")
        print(f"{'='*70}")

        for regime_idx in range(n_classes):
            # Binary labels: 1 if this regime, 0 otherwise
            binary_labels = (true_labels == regime_idx).astype(int)
            probs_for_regime = predicted_probs[:, regime_idx]

            # Fit calibrator
            self.calibrators[regime_idx].fit(
                probs_for_regime, binary_labels, regime_idx
            )

            # Evaluate before/after
            metrics_before = self.calibrators[regime_idx].evaluate_calibration(
                probs_for_regime, binary_labels
            )

            regime_name = list(MarketRegime)[regime_idx].value
            print(f"\n{regime_name.upper()}")
            print(
                f"  Before: Brier={metrics_before.brier_score:.3f}, ECE={metrics_before.ece:.3f}"
            )

        self.is_fitted = True
        print(f"\n✅ Calibration complete!")
        print(f"{'='*70}\n")

    def calibrate(
        self, predicted_probs: Dict[MarketRegime, float]
    ) -> Dict[MarketRegime, float]:
        """
        Calibrate regime probabilities

        Args:
            predicted_probs: Dict mapping regime to uncalibrated probability

        Returns:
            Dict mapping regime to calibrated probability
        """
        if not self.is_fitted:
            return predicted_probs

        # Convert to index-based dict
        regime_to_idx = {regime: i for i, regime in enumerate(MarketRegime)}
        idx_probs = {regime_to_idx[r]: p for r, p in predicted_probs.items()}

        # Calibrate each
        calibrated_idx = {}
        for regime, prob in predicted_probs.items():
            regime_idx = regime_to_idx[regime]
            cal_prob = self.calibrators[regime_idx].calibrate({regime_idx: prob})
            calibrated_idx[regime_idx] = cal_prob[regime_idx]

        # Renormalize
        total = sum(calibrated_idx.values())
        if total > 0:
            calibrated_idx = {k: v / total for k, v in calibrated_idx.items()}

        # Convert back to regime keys
        idx_to_regime = {i: regime for regime, i in regime_to_idx.items()}
        calibrated = {idx_to_regime[i]: p for i, p in calibrated_idx.items()}

        return calibrated

    def evaluate_all(
        self, predicted_probs: np.ndarray, true_labels: np.ndarray
    ) -> Dict[MarketRegime, CalibrationMetrics]:
        """
        Evaluate calibration for all regimes

        Args:
            predicted_probs: Predicted probabilities [N, n_classes]
            true_labels: Integer labels [N]

        Returns:
            Dict mapping regime to CalibrationMetrics
        """
        results = {}

        for regime_idx, regime in enumerate(MarketRegime):
            binary_labels = (true_labels == regime_idx).astype(int)
            probs_for_regime = predicted_probs[:, regime_idx]

            metrics = self.calibrators[regime_idx].evaluate_calibration(
                probs_for_regime, binary_labels
            )

            results[regime] = metrics

        return results
