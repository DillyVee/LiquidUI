"""
FINAL FIXED PSR Composite System (Corrected Version)
---------------------------------------------------
Key improvements:
1. Correct PSR variance scaling for annualized Sharpe
2. Effective sample size (trade count) support
3. Conservative fallback for negative variance
4. Adaptive z-score clipping based on sample size
5. Same API as your original version
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import optuna


# ============================================================
# Weights Dataclass
# ============================================================
@dataclass
class CompositeWeights:
    """Weights for composite score"""
    psr: float = 0.70
    pbo_penalty: float = 0.20
    turnover: float = 0.05
    drawdown: float = 0.05

    def validate(self):
        total = self.psr + self.pbo_penalty + self.turnover + self.drawdown
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights sum to {total:.3f}, should be ~1.0")


# ============================================================
# PSR Calculator (Corrected)
# ============================================================
class PSRCalculator:
    """
    FINAL FIXED Probabilistic Sharpe Ratio Calculator
    - Correct variance scaling for annualized Sharpe
    - Properly handles low trade counts
    - Conservative fallback for negative variance
    - Adaptive confidence clipping
    """

    @staticmethod
    def calculate_psr(
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
        annualization_factor: float = 252.0,
        trade_count: int = None
    ) -> float:
        """Calculate PSR with correct scaling and trade-count awareness"""
        if returns is None or len(returns) == 0:
            return 0.5

        returns = returns[~(np.isnan(returns) | np.isinf(returns))]
        if len(returns) < 30:
            return 0.5

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0 or std_ret < 1e-12:
            return 0.95 if mean_ret > 0 else 0.05

        # per-period Sharpe (not annualized)
        s_p = mean_ret / std_ret

        # Annualized Sharpe for reporting and comparison
        observed_sharpe = s_p * np.sqrt(annualization_factor)
        observed_sharpe = np.clip(observed_sharpe, -10.0, 10.0)

        try:
            skew = stats.skew(returns, bias=False)
            kurt = stats.kurtosis(returns, bias=False, fisher=True)
            skew = np.clip(skew, -10.0, 10.0)
            kurt = np.clip(kurt, -10.0, 50.0)
        except Exception:
            skew = 0.0
            kurt = 0.0

        n = len(returns)
        if trade_count is not None and trade_count < n / 10:
            effective_n = max(int(trade_count), 10)
        else:
            effective_n = n

        try:
            # Corrected variance computation
            numerator = (1.0 - (skew * s_p) + (((kurt - 1.0) / 4.0) * (s_p ** 2)))
            denom = max(effective_n - 1.0, 1.0)
            var_sp = numerator / denom
            variance_annual_sharpe = annualization_factor * var_sp

            if variance_annual_sharpe <= 0 or not np.isfinite(variance_annual_sharpe):
                base_se = 1.0 / np.sqrt(max(effective_n - 1.0, 1.0))
                skew_penalty = 1.0 + min(5.0, abs(skew)) * 0.3
                kurt_penalty = 1.0 + min(10.0, abs(kurt)) * 0.15
                sharpe_std = base_se * skew_penalty * kurt_penalty * np.sqrt(annualization_factor)
            else:
                sharpe_std = np.sqrt(variance_annual_sharpe)

            if sharpe_std < 1e-10 or not np.isfinite(sharpe_std):
                sharpe_std = 0.2

            z_score = (observed_sharpe - benchmark_sharpe) / sharpe_std

            if effective_n < 30:
                max_z = 2.0
            elif effective_n < 100:
                max_z = 2.5
            else:
                max_z = 3.0

            z_score = np.clip(z_score, -max_z, max_z)
            psr = stats.norm.cdf(z_score)
            psr = np.clip(psr, 0.001, 0.999)

        except Exception:
            psr = 0.75 if observed_sharpe > benchmark_sharpe else 0.25

        return float(psr)

    @staticmethod
    def calculate_sharpe_from_equity(
        equity_curve: np.ndarray,
        annualization_factor: float = 252.0
    ) -> float:
        """Calculate annualized Sharpe ratio from equity curve"""
        if len(equity_curve) < 2:
            return 0.0

        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~(np.isnan(returns) | np.isinf(returns))]

        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0 or std_ret < 1e-10:
            return 0.0

        sharpe = mean_ret / std_ret * np.sqrt(annualization_factor)
        return np.clip(sharpe, -5, 10)


# ============================================================
# Simplified PBO Calculator
# ============================================================
class PBOCalculatorSimple:
    """Simplified PBO estimation"""

    @staticmethod
    def estimate_pbo_from_returns(returns: np.ndarray) -> float:
        """Estimate overfitting probability from return characteristics"""
        if len(returns) < 20:
            return 0.5

        returns = returns[~(np.isnan(returns) | np.isinf(returns))]
        if len(returns) < 20:
            return 0.5

        mid = len(returns) // 2
        first_half = returns[:mid]
        second_half = returns[mid:]

        sharpe_1 = (np.mean(first_half) / (np.std(first_half, ddof=1) + 1e-10))
        sharpe_2 = (np.mean(second_half) / (np.std(second_half, ddof=1) + 1e-10))

        if sharpe_1 <= 0 and sharpe_2 <= 0:
            pbo = 0.9
        elif sharpe_1 > 1.0 and sharpe_2 < 0:
            pbo = 0.8
        elif sharpe_1 > sharpe_2 * 3:
            pbo = 0.7
        elif sharpe_1 > sharpe_2 * 1.5:
            pbo = 0.5
        elif sharpe_2 > sharpe_1:
            pbo = 0.2
        elif abs(sharpe_1 - sharpe_2) < 0.2:
            pbo = 0.3
        else:
            pbo = 0.4

        return np.clip(pbo, 0.0, 1.0)


# ============================================================
# Turnover Calculator
# ============================================================
class TurnoverCalculator:
    """Calculate strategy turnover"""

    @staticmethod
    def calculate_annual_turnover(
        trade_count: int,
        total_days: int,
        position_size: float = 1.0
    ) -> float:
        """Estimate annualized turnover"""
        if total_days == 0:
            return 0.0

        trades_per_day = trade_count / total_days
        annual_trades = trades_per_day * 252
        turnover = annual_trades * 2 * position_size
        return turnover


# ============================================================
# Composite Optimizer
# ============================================================
class CompositeOptimizer:
    """Composite optimizer with fixed PSR"""

    def __init__(
        self,
        weights: Optional[CompositeWeights] = None,
        benchmark_sharpe: float = 0.0,
        max_acceptable_turnover: float = 200.0,
        max_acceptable_dd: float = 0.50,
        min_trades: int = 20
    ):
        self.weights = weights or CompositeWeights()
        self.weights.validate()

        self.benchmark_sharpe = benchmark_sharpe
        self.max_acceptable_turnover = max_acceptable_turnover
        self.max_acceptable_dd = max_acceptable_dd
        self.min_trades = min_trades

    def calculate_composite_score(
        self,
        equity_curve: np.ndarray,
        trade_count: int,
        total_days: int,
        annualization_factor: float = 252.0
    ) -> Dict[str, float]:
        """Calculate composite score with all components"""
        if len(equity_curve) < 2:
            raise ValueError("Equity curve too short for composite score")

        if trade_count < self.min_trades:
            trade_penalty = 1.0 - (trade_count / self.min_trades)
        else:
            trade_penalty = 0.0

        returns = np.diff(equity_curve) / equity_curve[:-1]

        psr = PSRCalculator.calculate_psr(
            returns,
            benchmark_sharpe=self.benchmark_sharpe,
            annualization_factor=annualization_factor,
            trade_count=trade_count
        )
        psr_score = psr

        pbo = PBOCalculatorSimple.estimate_pbo_from_returns(returns)
        pbo_penalty = pbo

        annual_turnover = TurnoverCalculator.calculate_annual_turnover(
            trade_count, total_days
        )

        if annual_turnover < self.max_acceptable_turnover:
            turnover_penalty = 0.0
        else:
            turnover_penalty = (annual_turnover - self.max_acceptable_turnover) / 200.0
            turnover_penalty = np.clip(turnover_penalty, 0.0, 1.0)

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown)
        dd_penalty = np.clip(max_dd / self.max_acceptable_dd, 0.0, 1.0)

        composite = (
            self.weights.psr * psr_score
            - self.weights.pbo_penalty * pbo_penalty
            - self.weights.turnover * turnover_penalty
            - self.weights.drawdown * dd_penalty
            - (0.3 * trade_penalty)
        )

        composite = np.clip(composite, -1.0, 1.0)

        return {
            "composite_score": composite,
            "psr": psr,
            "psr_score": psr_score,
            "pbo": pbo_penalty,
            "annual_turnover": annual_turnover,
            "turnover_penalty": turnover_penalty,
            "max_drawdown": max_dd,
            "dd_penalty": dd_penalty,
            "trade_penalty": trade_penalty,
            "trade_count": trade_count,
        }

    @staticmethod
    def create_optuna_study(
        study_name: str = "composite_optimization",
        storage_path: str = "sqlite:///optuna_composite.db",
        load_if_exists: bool = True
    ) -> optuna.Study:
        """Create Optuna study"""
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=30,
            n_ei_candidates=24,
            multivariate=False,
            warn_independent_sampling=False,
            seed=42,
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=20, n_warmup_steps=5, interval_steps=1
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage_path,
            load_if_exists=load_if_exists,
        )

        return study
