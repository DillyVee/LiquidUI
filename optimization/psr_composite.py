"""
FIXED PSR Composite System
Issues Fixed:
1. PSR calculation capped at realistic values (not always 1.00)
2. Reduced turnover penalty to allow more trades
3. Added minimum trade count requirement
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import optuna


@dataclass
class CompositeWeights:
    """
    ADJUSTED weights to allow more trades
    - Reduced turnover penalty from 15% to 5%
    - Increased PSR weight
    """
    psr: float = 0.70           # Probabilistic Sharpe Ratio (increased)
    pbo_penalty: float = 0.20   # Overfitting penalty
    turnover: float = 0.05      # Trading cost penalty (REDUCED)
    drawdown: float = 0.05      # Risk penalty
    
    def validate(self):
        """Ensure weights sum to ~1.0"""
        total = self.psr + self.pbo_penalty + self.turnover + self.drawdown
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights sum to {total:.3f}, should be ~1.0")


class PSRCalculator:
    """
    FIXED Probabilistic Sharpe Ratio Calculator
    
    Issues Fixed:
    1. Returns NaN/inf values properly
    2. Caps PSR at realistic values
    3. Requires minimum sample size
    """
    
    @staticmethod
    def calculate_psr(
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
        annualization_factor: float = 252.0
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio with proper validation
        
        Returns:
            PSR between 0.0 and 1.0 (NEVER exactly 1.00 unless very strong evidence)
        """
        # ✅ FIX: Require minimum 30 returns for reliable PSR
        if len(returns) < 30:
            return 0.5  # Unknown - neutral score
        
        # Clean returns
        returns = returns[~(np.isnan(returns) | np.isinf(returns))]
        
        if len(returns) < 30:
            return 0.5
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        # ✅ FIX: If std is zero, return based on mean
        if std_ret == 0 or std_ret < 1e-10:
            return 0.95 if mean_ret > 0 else 0.05  # Strong but not 1.00
        
        # Calculate observed Sharpe
        observed_sharpe = mean_ret / std_ret * np.sqrt(annualization_factor)
        
        # ✅ FIX: Cap unrealistic Sharpe ratios
        if observed_sharpe > 10:
            observed_sharpe = 10  # Extremely high Sharpe, cap it
        elif observed_sharpe < -5:
            observed_sharpe = -5
        
        # Calculate skewness and kurtosis with error handling
        try:
            skew = stats.skew(returns, bias=False)
            kurt = stats.kurtosis(returns, bias=False, fisher=True)
            
            # ✅ FIX: Cap extreme values
            skew = np.clip(skew, -5, 5)
            kurt = np.clip(kurt, -5, 10)
        except:
            skew = 0
            kurt = 0
        
        n = len(returns)
        
        # PSR formula with higher moments
        try:
            sharpe_std = np.sqrt(
                (1 + (0.5 * observed_sharpe**2) - 
                 (skew * observed_sharpe) + 
                 (((kurt - 3) / 4) * observed_sharpe**2)) / n
            )
            
            # ✅ FIX: Ensure sharpe_std is reasonable
            if sharpe_std < 1e-10:
                sharpe_std = 0.01  # Small but non-zero
            
            # Z-score
            z_score = (observed_sharpe - benchmark_sharpe) / sharpe_std
            
            # ✅ FIX: Cap z-score to prevent PSR = 1.000000
            z_score = np.clip(z_score, -5, 5)  # ~99.9999% confidence at ±5
            
            # PSR from CDF
            psr = stats.norm.cdf(z_score)
            
            # ✅ FIX: Never return exactly 1.0 or 0.0
            psr = np.clip(psr, 0.001, 0.999)
            
        except:
            # Fallback calculation
            if observed_sharpe > benchmark_sharpe:
                psr = 0.75  # Moderate confidence
            else:
                psr = 0.25
        
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
        
        # Cap Sharpe at realistic values
        return np.clip(sharpe, -5, 10)


class PBOCalculatorSimple:
    """Simplified PBO without requiring multiple equity curves"""
    
    @staticmethod
    def estimate_pbo_from_returns(returns: np.ndarray) -> float:
        """
        Estimate PBO using return characteristics
        
        ✅ FIXED: Better heuristics for overfitting detection
        """
        if len(returns) < 20:
            return 0.5  # Unknown
        
        returns = returns[~(np.isnan(returns) | np.isinf(returns))]
        
        if len(returns) < 20:
            return 0.5
        
        # Split returns into two halves
        mid = len(returns) // 2
        first_half = returns[:mid]
        second_half = returns[mid:]
        
        # Calculate Sharpe for each half
        sharpe_1 = (np.mean(first_half) / (np.std(first_half, ddof=1) + 1e-10))
        sharpe_2 = (np.mean(second_half) / (np.std(second_half, ddof=1) + 1e-10))
        
        # ✅ FIXED: More nuanced PBO estimation
        if sharpe_1 <= 0 and sharpe_2 <= 0:
            pbo = 0.9  # Both halves negative - bad strategy
        elif sharpe_1 > 1.0 and sharpe_2 < 0:
            pbo = 0.8  # High overfitting risk
        elif sharpe_1 > sharpe_2 * 3:  # First much better than second
            pbo = 0.7  # Moderate-high risk
        elif sharpe_1 > sharpe_2 * 1.5:
            pbo = 0.5  # Moderate risk
        elif sharpe_2 > sharpe_1:
            pbo = 0.2  # Good sign (improving)
        elif abs(sharpe_1 - sharpe_2) < 0.2:
            pbo = 0.3  # Consistent
        else:
            pbo = 0.4  # Slightly degrading but acceptable
        
        return np.clip(pbo, 0.0, 1.0)


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


class CompositeOptimizer:
    """
    FIXED Composite Optimizer
    - Allows more trades
    - Better PSR calculation
    - Minimum trade requirement
    """
    
    def __init__(
        self,
        weights: Optional[CompositeWeights] = None,
        benchmark_sharpe: float = 0.0,
        max_acceptable_turnover: float = 200.0,  # ✅ INCREASED from 100
        max_acceptable_dd: float = 0.50,
        min_trades: int = 20  # ✅ NEW: Minimum required trades
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
        """
        Calculate composite score with fixes:
        1. Better PSR calculation
        2. Penalty for too few trades
        3. Reduced turnover penalty
        """
        # ✅ PENALTY for insufficient trades
        if trade_count < self.min_trades:
            trade_penalty = 1.0 - (trade_count / self.min_trades)
        else:
            trade_penalty = 0.0
        
        # Component 1: PSR (FIXED)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        psr = PSRCalculator.calculate_psr(
            returns,
            benchmark_sharpe=self.benchmark_sharpe,
            annualization_factor=annualization_factor
        )
        psr_score = psr
        
        # Component 2: PBO penalty (FIXED)
        pbo = PBOCalculatorSimple.estimate_pbo_from_returns(returns)
        pbo_penalty = pbo
        
        # Component 3: Turnover penalty (REDUCED IMPACT)
        annual_turnover = TurnoverCalculator.calculate_annual_turnover(
            trade_count, total_days
        )
        
        # ✅ FIX: Gentler turnover penalty
        # Only penalize if > 200 trades/year (was 100)
        if annual_turnover < self.max_acceptable_turnover:
            turnover_penalty = 0.0  # No penalty for reasonable trading
        else:
            turnover_penalty = (annual_turnover - self.max_acceptable_turnover) / 200.0
            turnover_penalty = np.clip(turnover_penalty, 0.0, 1.0)
        
        # Component 4: Drawdown penalty
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown)
        dd_penalty = np.clip(max_dd / self.max_acceptable_dd, 0.0, 1.0)
        
        # Calculate composite score
        composite = (
            self.weights.psr * psr_score -
            self.weights.pbo_penalty * pbo_penalty -
            self.weights.turnover * turnover_penalty -
            self.weights.drawdown * dd_penalty -
            (0.3 * trade_penalty)  # ✅ NEW: Penalty for too few trades
        )
        
        # ✅ Cap composite score to reasonable range
        composite = np.clip(composite, -1.0, 1.0)
        
        return {
            'composite_score': composite,
            'psr': psr,
            'psr_score': psr_score,
            'pbo': pbo_penalty,
            'annual_turnover': annual_turnover,
            'turnover_penalty': turnover_penalty,
            'max_drawdown': max_dd,
            'dd_penalty': dd_penalty,
            'trade_penalty': trade_penalty,
            'trade_count': trade_count
        }
    
    @staticmethod
    def create_optuna_study(
        study_name: str = "composite_optimization",
        storage_path: str = "sqlite:///optuna_composite.db",
        load_if_exists: bool = True
    ) -> optuna.Study:
        """Create Optuna study with optimal settings"""
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=30,
            n_ei_candidates=24,
            multivariate=False,
            warn_independent_sampling=False,
            seed=42
        )
        
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=5,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage_path,
            load_if_exists=load_if_exists
        )
        
        return study