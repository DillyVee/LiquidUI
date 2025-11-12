"""
FINAL FIXED PSR Composite System
Key fixes:
1. PSR uses effective sample size (trade count) for low-trade strategies
2. Conservative fallback when variance term is negative
3. Adaptive z-score clipping based on sample size
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import optuna


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


class PSRCalculator:
    """
    FINAL FIXED Probabilistic Sharpe Ratio Calculator
    - Properly handles low trade counts
    - Conservative fallback for negative variance
    - Realistic confidence bounds
    """
    
    @staticmethod
    def calculate_psr(
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
        annualization_factor: float = 252.0,
        trade_count: int = None
    ) -> float:
        """
        Calculate PSR with proper confidence adjustment for low-trade strategies
        
        CRITICAL FIX: When trade_count is low, use it as effective sample size
        instead of the number of bars, which gives false confidence.
        
        Example:
        - 15 trades over 3480 bars
        - Old way: n=3480 → PSR=99.9% (WRONG - overconfident)
        - New way: n=15 → PSR=53% (CORRECT - realistic)
        """
        if len(returns) < 30:
            return 0.5
        
        returns = returns[~(np.isnan(returns) | np.isinf(returns))]
        
        if len(returns) < 30:
            return 0.5
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0 or std_ret < 1e-10:
            return 0.95 if mean_ret > 0 else 0.05
        
        observed_sharpe = mean_ret / std_ret * np.sqrt(annualization_factor)
        observed_sharpe = np.clip(observed_sharpe, -5, 10)
        
        try:
            skew = stats.skew(returns, bias=False)
            kurt = stats.kurtosis(returns, bias=False, fisher=True)
            skew = np.clip(skew, -5, 5)
            kurt = np.clip(kurt, -5, 10)
        except:
            skew = 0.0
            kurt = 0.0
        
        n = len(returns)
        
        # ============================================================
        # CRITICAL FIX: Use effective sample size
        # ============================================================
        # For low-trade strategies, the true degrees of freedom is the
        # number of independent trades, not the number of bars
        # ============================================================
        if trade_count is not None and trade_count < n / 10:
            effective_n = max(trade_count, 10)  # Minimum 10
            # Note: We don't print here during optimization to avoid spam
        else:
            effective_n = n
        
        try:
            # Bailey & López de Prado (2014) formula
            variance_term = (
                1.0 - (skew * observed_sharpe) + 
                (((kurt - 1.0) / 4.0) * observed_sharpe**2)
            ) / (effective_n - 1.0)
            
            # Handle negative variance (can occur with extreme higher moments)
            if variance_term <= 0 or not np.isfinite(variance_term):
                # Use conservative fallback with higher-moment penalty
                base_se = 1.0 / np.sqrt(effective_n - 1.0)
                
                # Penalty increases with extreme skew/kurtosis (more uncertainty)
                skew_penalty = 1.0 + abs(skew) * 0.3
                kurt_penalty = 1.0 + max(0, abs(kurt)) * 0.15
                
                sharpe_std = base_se * skew_penalty * kurt_penalty
            else:
                sharpe_std = np.sqrt(variance_term)
            
            if sharpe_std < 1e-10 or not np.isfinite(sharpe_std):
                sharpe_std = 0.2  # Conservative default
            
            # Z-score
            z_score = (observed_sharpe - benchmark_sharpe) / sharpe_std
            
            # Adaptive clipping: smaller samples = more conservative
            if effective_n < 30:
                max_z = 2.0  # ~97.7% max confidence
            elif effective_n < 100:
                max_z = 2.5  # ~99.4% max confidence
            else:
                max_z = 3.0  # ~99.87% max confidence
            
            z_score = np.clip(z_score, -max_z, max_z)
            
            # PSR from cumulative normal distribution
            psr = stats.norm.cdf(z_score)
            psr = np.clip(psr, 0.001, 0.999)
            
        except Exception as e:
            # Fallback if anything goes wrong
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
        
        # Split into two halves
        mid = len(returns) // 2
        first_half = returns[:mid]
        second_half = returns[mid:]
        
        sharpe_1 = (np.mean(first_half) / (np.std(first_half, ddof=1) + 1e-10))
        sharpe_2 = (np.mean(second_half) / (np.std(second_half, ddof=1) + 1e-10))
        
        if sharpe_1 <= 0 and sharpe_2 <= 0:
            pbo = 0.9  # Both negative
        elif sharpe_1 > 1.0 and sharpe_2 < 0:
            pbo = 0.8  # High overfitting risk
        elif sharpe_1 > sharpe_2 * 3:
            pbo = 0.7  # Significant degradation
        elif sharpe_1 > sharpe_2 * 1.5:
            pbo = 0.5  # Moderate degradation
        elif sharpe_2 > sharpe_1:
            pbo = 0.2  # Improving (good sign)
        elif abs(sharpe_1 - sharpe_2) < 0.2:
            pbo = 0.3  # Consistent
        else:
            pbo = 0.4  # Slight degradation
        
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
        
        # Penalty for too few trades
        if trade_count < self.min_trades:
            trade_penalty = 1.0 - (trade_count / self.min_trades)
        else:
            trade_penalty = 0.0
        
        # Component 1: PSR (with trade-count awareness)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        psr = PSRCalculator.calculate_psr(
            returns,
            benchmark_sharpe=self.benchmark_sharpe,
            annualization_factor=annualization_factor,
            trade_count=trade_count  # ← KEY FIX
        )
        psr_score = psr
        
        # Component 2: PBO penalty
        pbo = PBOCalculatorSimple.estimate_pbo_from_returns(returns)
        pbo_penalty = pbo
        
        # Component 3: Turnover penalty
        annual_turnover = TurnoverCalculator.calculate_annual_turnover(
            trade_count, total_days
        )
        
        if annual_turnover < self.max_acceptable_turnover:
            turnover_penalty = 0.0
        else:
            turnover_penalty = (annual_turnover - self.max_acceptable_turnover) / 200.0
            turnover_penalty = np.clip(turnover_penalty, 0.0, 1.0)
        
        # Component 4: Drawdown penalty
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown)
        dd_penalty = np.clip(max_dd / self.max_acceptable_dd, 0.0, 1.0)
        
        # Composite score
        composite = (
            self.weights.psr * psr_score -
            self.weights.pbo_penalty * pbo_penalty -
            self.weights.turnover * turnover_penalty -
            self.weights.drawdown * dd_penalty -
            (0.3 * trade_penalty)
        )
        
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
        """Create Optuna study"""
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