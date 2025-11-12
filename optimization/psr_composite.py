"""
PSR-Based Composite Optimization System

Optimizes for:
1. Probabilistic Sharpe Ratio (PSR) - statistical significance
2. Walk-Forward Stability - robustness to regime changes
3. PBO penalty - overfitting detection
4. Turnover penalty - execution realism
5. Drawdown penalty - risk management

This replaces single-metric optimization with a composite score that
rewards robust, tradeable strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


@dataclass
class CompositeWeights:
    """Weights for composite objective function"""
    psr: float = 0.40           # Probabilistic Sharpe Ratio
    wfa_sharpe: float = 0.30    # Walk-Forward mean Sharpe
    pbo_penalty: float = 0.15   # Overfitting penalty
    turnover: float = 0.10      # Trading cost penalty
    drawdown: float = 0.05      # Risk penalty
    
    def validate(self):
        """Ensure weights sum to ~1.0"""
        total = self.psr + self.wfa_sharpe + self.pbo_penalty + self.turnover + self.drawdown
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights sum to {total:.3f}, should be ~1.0")


class PSRCalculator:
    """
    Probabilistic Sharpe Ratio Calculator
    
    PSR = probability that true Sharpe > benchmark Sharpe
    
    PSR > 0.95 = very confident strategy is good
    PSR > 0.75 = reasonable confidence
    PSR < 0.50 = likely false positive
    """
    
    @staticmethod
    def calculate_psr(
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
        annualization_factor: float = 252.0
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio
        
        Args:
            returns: Array of returns (NOT equity curve)
            benchmark_sharpe: Minimum acceptable Sharpe (default 0)
            annualization_factor: 252 for daily, 252*6.5 for hourly, etc.
            
        Returns:
            PSR value between 0 and 1
        """
        if len(returns) < 10:
            return 0.0
        
        # Remove invalid values
        returns = returns[~(np.isnan(returns) | np.isinf(returns))]
        
        if len(returns) < 10:
            return 0.0
        
        # Calculate observed Sharpe
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            return 0.0
        
        observed_sharpe = mean_ret / std_ret * np.sqrt(annualization_factor)
        
        # Calculate skewness and kurtosis
        skew = stats.skew(returns, bias=False)
        kurt = stats.kurtosis(returns, bias=False, fisher=True)  # Excess kurtosis
        
        # PSR formula from Bailey & Lopez de Prado
        n = len(returns)
        
        # Standard error of Sharpe ratio accounting for higher moments
        sharpe_std = np.sqrt(
            (1 + (0.5 * observed_sharpe**2) - (skew * observed_sharpe) + (((kurt - 3) / 4) * observed_sharpe**2)) / n
        )
        
        if sharpe_std == 0:
            return 1.0 if observed_sharpe > benchmark_sharpe else 0.0
        
        # Z-score
        z_score = (observed_sharpe - benchmark_sharpe) / sharpe_std
        
        # PSR is the cumulative distribution function of standard normal at z_score
        psr = stats.norm.cdf(z_score)
        
        return np.clip(psr, 0.0, 1.0)
    
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
        
        if std_ret == 0:
            return 0.0
        
        return mean_ret / std_ret * np.sqrt(annualization_factor)


class PBOCalculator:
    """
    Probability of Backtest Overfitting (PBO)
    
    Estimates likelihood that observed performance is due to overfitting
    
    PBO < 0.3 = low overfitting risk
    PBO > 0.5 = high overfitting risk
    """
    
    @staticmethod
    def calculate_pbo_lite(
        equity_curves: List[np.ndarray],
        selection_metric: str = 'sharpe'
    ) -> float:
        """
        Lightweight PBO approximation using multiple equity curves
        
        Args:
            equity_curves: List of equity curves from different folds/periods
            selection_metric: How curves were selected
            
        Returns:
            PBO estimate (0 = no overfitting, 1 = severe overfitting)
        """
        if len(equity_curves) < 3:
            return 0.5  # Unknown, assume moderate risk
        
        # Calculate performance rank correlations between halves
        n_curves = len(equity_curves)
        
        # Split curves into first half and second half
        sharpes_first = []
        sharpes_second = []
        
        for curve in equity_curves:
            if len(curve) < 20:
                continue
            
            mid = len(curve) // 2
            first_half = curve[:mid]
            second_half = curve[mid:]
            
            sharpe_1 = PSRCalculator.calculate_sharpe_from_equity(first_half)
            sharpe_2 = PSRCalculator.calculate_sharpe_from_equity(second_half)
            
            sharpes_first.append(sharpe_1)
            sharpes_second.append(sharpe_2)
        
        if len(sharpes_first) < 3:
            return 0.5
        
        # Calculate rank correlation
        try:
            rho, _ = stats.spearmanr(sharpes_first, sharpes_second)
            
            # PBO is roughly the probability that rho <= 0
            # Convert rho to PBO: rho=-1 -> PBO=1, rho=0 -> PBO=0.5, rho=1 -> PBO=0
            pbo = (1 - rho) / 2
            
            return np.clip(pbo, 0.0, 1.0)
        except:
            return 0.5


class WalkForwardLite:
    """
    Lightweight Walk-Forward Analysis for optimization
    
    Uses 3-6 small folds to assess out-of-sample stability
    Much faster than full walk-forward
    """
    
    @staticmethod
    def evaluate_wfa(
        simulate_func,
        params: Dict,
        df_dict: Dict[str, pd.DataFrame],
        n_folds: int = 4,
        train_ratio: float = 0.7
    ) -> Tuple[float, List[np.ndarray]]:
        """
        Quick WFA evaluation
        
        Args:
            simulate_func: Function to run backtest
            params: Strategy parameters
            df_dict: Data dictionary
            n_folds: Number of train/test folds
            train_ratio: Fraction of data for training in each fold
            
        Returns:
            (mean_oos_sharpe, list_of_oos_equity_curves)
        """
        # Get finest timeframe
        tf_order = {'5min': 0, 'hourly': 1, 'daily': 2}
        timeframes = list(df_dict.keys())
        finest_tf = sorted(timeframes, key=lambda x: tf_order.get(x, 99))[0]
        
        df_finest = df_dict[finest_tf]
        n_total = len(df_finest)
        
        if n_total < 100:
            return 0.0, []
        
        fold_size = n_total // n_folds
        oos_sharpes = []
        oos_curves = []
        
        for fold in range(n_folds):
            # Define test window
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, n_total)
            
            if test_end - test_start < 20:
                continue
            
            # Create test data subset
            test_dict = {}
            for tf in timeframes:
                df_tf = df_dict[tf].iloc[test_start:test_end].copy()
                test_dict[tf] = df_tf.reset_index(drop=True)
            
            # Simulate on test fold
            try:
                eq_curve, trades = simulate_func(params, test_dict)
                
                if eq_curve is not None and len(eq_curve) > 10:
                    sharpe = PSRCalculator.calculate_sharpe_from_equity(eq_curve)
                    oos_sharpes.append(sharpe)
                    oos_curves.append(eq_curve)
            except:
                continue
        
        if len(oos_sharpes) == 0:
            return 0.0, []
        
        mean_sharpe = np.mean(oos_sharpes)
        return mean_sharpe, oos_curves


class TurnoverCalculator:
    """
    Calculate strategy turnover and estimate transaction costs
    """
    
    @staticmethod
    def calculate_annual_turnover(
        trade_count: int,
        total_days: int,
        position_size: float = 1.0
    ) -> float:
        """
        Estimate annualized turnover
        
        Args:
            trade_count: Number of round-trip trades
            total_days: Total days in backtest
            position_size: Average position size (1.0 = 100%)
            
        Returns:
            Annualized turnover (trades per year * position size)
        """
        if total_days == 0:
            return 0.0
        
        # Trades per day
        trades_per_day = trade_count / total_days
        
        # Annualize (252 trading days)
        annual_trades = trades_per_day * 252
        
        # Each trade involves entry + exit
        turnover = annual_trades * 2 * position_size
        
        return turnover
    
    @staticmethod
    def estimate_cost_drag(
        turnover: float,
        commission: float = 0.0005,
        slippage: float = 0.0005,
        spread: float = 0.0001
    ) -> float:
        """
        Estimate annual performance drag from costs
        
        Args:
            turnover: Annual turnover (from calculate_annual_turnover)
            commission: Commission per trade (decimal, e.g., 0.0005 = 0.05%)
            slippage: Slippage per trade
            spread: Bid-ask spread
            
        Returns:
            Annual cost drag (decimal, e.g., 0.10 = 10% drag)
        """
        cost_per_trade = commission + slippage + spread
        total_drag = turnover * cost_per_trade
        
        return total_drag


class CompositeOptimizer:
    """
    Main composite optimization engine
    
    Combines PSR, WFA, PBO, turnover, and drawdown into single objective
    """
    
    def __init__(
        self,
        weights: Optional[CompositeWeights] = None,
        benchmark_sharpe: float = 0.0,
        max_acceptable_turnover: float = 100.0,
        max_acceptable_dd: float = 0.50
    ):
        self.weights = weights or CompositeWeights()
        self.weights.validate()
        
        self.benchmark_sharpe = benchmark_sharpe
        self.max_acceptable_turnover = max_acceptable_turnover
        self.max_acceptable_dd = max_acceptable_dd
    
    def calculate_composite_score(
        self,
        equity_curve: np.ndarray,
        trade_count: int,
        total_days: int,
        oos_equity_curves: List[np.ndarray],
        annualization_factor: float = 252.0
    ) -> Dict[str, float]:
        """
        Calculate composite optimization score
        
        Args:
            equity_curve: In-sample equity curve
            trade_count: Number of trades
            total_days: Days in backtest
            oos_equity_curves: Out-of-sample curves from WFA
            annualization_factor: For Sharpe calculation
            
        Returns:
            Dictionary with component scores and total
        """
        # Component 1: PSR
        returns = np.diff(equity_curve) / equity_curve[:-1]
        psr = PSRCalculator.calculate_psr(
            returns,
            benchmark_sharpe=self.benchmark_sharpe,
            annualization_factor=annualization_factor
        )
        psr_score = psr  # Already 0-1
        
        # Component 2: WFA mean Sharpe (normalized)
        if len(oos_equity_curves) > 0:
            oos_sharpes = [
                PSRCalculator.calculate_sharpe_from_equity(curve, annualization_factor)
                for curve in oos_equity_curves
            ]
            mean_oos_sharpe = np.mean(oos_sharpes)
            
            # Normalize: Sharpe 2.0 = excellent, 0.5 = minimum acceptable
            wfa_sharpe_norm = np.clip((mean_oos_sharpe - 0.5) / 1.5, 0.0, 1.0)
        else:
            wfa_sharpe_norm = 0.0
        
        # Component 3: PBO penalty
        if len(oos_equity_curves) > 0:
            pbo = PBOCalculator.calculate_pbo_lite(oos_equity_curves)
            pbo_penalty = pbo  # Higher PBO = worse
        else:
            pbo_penalty = 0.5  # Unknown
        
        # Component 4: Turnover penalty
        annual_turnover = TurnoverCalculator.calculate_annual_turnover(
            trade_count, total_days
        )
        
        # Normalize: 0-20 trades/year = good, >100 = excessive
        turnover_penalty = np.clip(annual_turnover / self.max_acceptable_turnover, 0.0, 1.0)
        
        # Component 5: Drawdown penalty
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown)
        
        # Normalize: 0-10% DD = good, >50% = severe
        dd_penalty = np.clip(max_dd / self.max_acceptable_dd, 0.0, 1.0)
        
        # Calculate composite score
        composite = (
            self.weights.psr * psr_score +
            self.weights.wfa_sharpe * wfa_sharpe_norm -
            self.weights.pbo_penalty * pbo_penalty -
            self.weights.turnover * turnover_penalty -
            self.weights.drawdown * dd_penalty
        )
        
        return {
            'composite_score': composite,
            'psr': psr,
            'psr_score': psr_score,
            'wfa_sharpe': mean_oos_sharpe if len(oos_equity_curves) > 0 else 0.0,
            'wfa_sharpe_norm': wfa_sharpe_norm,
            'pbo': pbo_penalty,
            'annual_turnover': annual_turnover,
            'turnover_penalty': turnover_penalty,
            'max_drawdown': max_dd,
            'dd_penalty': dd_penalty
        }
    
    @staticmethod
    def create_optuna_study(
        study_name: str = "composite_optimization",
        storage_path: str = "sqlite:///optuna_composite.db",
        load_if_exists: bool = True
    ) -> optuna.Study:
        """
        Create Optuna study with optimal settings
        
        Args:
            study_name: Name for the study
            storage_path: SQLite database path
            load_if_exists: Load existing study if found
            
        Returns:
            Optuna study object
        """
        sampler = TPESampler(
            n_startup_trials=30,      # Random trials before TPE kicks in
            n_ei_candidates=24,        # Candidates for expected improvement
            multivariate=True,         # Consider parameter interactions
            seed=42                    # Reproducibility
        )
        
        pruner = MedianPruner(
            n_startup_trials=20,       # Don't prune early trials
            n_warmup_steps=5,          # Steps before pruning starts
            interval_steps=1           # Check every step
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
    
    def objective_function(
        self,
        trial: optuna.Trial,
        simulate_func,
        df_dict: Dict[str, pd.DataFrame],
        param_ranges: Dict,
        n_wfa_folds: int = 4
    ) -> float:
        """
        Optuna objective function for composite optimization
        
        Args:
            trial: Optuna trial object
            simulate_func: Function(params, df_dict) -> (equity_curve, trades)
            df_dict: Data dictionary
            param_ranges: Dictionary of parameter ranges
            n_wfa_folds: Number of WFA folds
            
        Returns:
            Composite score to maximize
        """
        # Sample parameters
        params = {}
        for tf in param_ranges['timeframes']:
            params[f'MN1_{tf}'] = trial.suggest_int(
                f'MN1_{tf}',
                param_ranges['mn1_range'][0],
                param_ranges['mn1_range'][1]
            )
            params[f'MN2_{tf}'] = trial.suggest_int(
                f'MN2_{tf}',
                param_ranges['mn2_range'][0],
                param_ranges['mn2_range'][1]
            )
            params[f'Entry_{tf}'] = trial.suggest_float(
                f'Entry_{tf}',
                param_ranges['entry_range'][0],
                param_ranges['entry_range'][1],
                step=0.5
            )
            params[f'Exit_{tf}'] = trial.suggest_float(
                f'Exit_{tf}',
                param_ranges['exit_range'][0],
                param_ranges['exit_range'][1],
                step=0.5
            )
            params[f'On_{tf}'] = trial.suggest_int(
                f'On_{tf}',
                param_ranges['on_range'][0],
                param_ranges['on_range'][1]
            )
            params[f'Off_{tf}'] = trial.suggest_int(
                f'Off_{tf}',
                param_ranges['off_range'][0],
                param_ranges['off_range'][1]
            )
            params[f'Start_{tf}'] = trial.suggest_int(
                f'Start_{tf}',
                0,
                param_ranges['on_range'][1] + param_ranges['off_range'][1]
            )
        
        # Run full backtest
        try:
            eq_curve, trades = simulate_func(params, df_dict)
            
            if eq_curve is None or len(eq_curve) < 50:
                return 0.0
            
            # Quick WFA evaluation
            mean_oos_sharpe, oos_curves = WalkForwardLite.evaluate_wfa(
                simulate_func,
                params,
                df_dict,
                n_folds=n_wfa_folds
            )
            
            # Calculate total days
            tf_order = {'5min': 0, 'hourly': 1, 'daily': 2}
            finest_tf = sorted(param_ranges['timeframes'], key=lambda x: tf_order.get(x, 99))[0]
            df_finest = df_dict[finest_tf]
            
            if 'Datetime' in df_finest.columns:
                total_days = (df_finest['Datetime'].max() - df_finest['Datetime'].min()).days
            else:
                total_days = len(df_finest)
            
            # Calculate composite score
            scores = self.calculate_composite_score(
                eq_curve,
                trades,
                max(total_days, 1),
                oos_curves,
                annualization_factor=252.0 if finest_tf == 'daily' else 252.0 * 6.5
            )
            
            # Report intermediate values for pruning
            trial.report(scores['composite_score'], step=0)
            
            # Store all metrics
            trial.set_user_attr('psr', scores['psr'])
            trial.set_user_attr('wfa_sharpe', scores['wfa_sharpe'])
            trial.set_user_attr('pbo', scores['pbo'])
            trial.set_user_attr('annual_turnover', scores['annual_turnover'])
            trial.set_user_attr('max_drawdown', scores['max_drawdown'])
            
            return scores['composite_score']
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0


# Example usage function
def create_composite_objective(
    optimizer_instance,
    df_dict: Dict[str, pd.DataFrame],
    param_ranges: Dict,
    weights: Optional[CompositeWeights] = None
):
    """
    Factory function to create composite objective for MultiTimeframeOptimizer
    
    Args:
        optimizer_instance: Instance of MultiTimeframeOptimizer
        df_dict: Data dictionary
        param_ranges: Parameter ranges
        weights: Optional custom weights
        
    Returns:
        Objective function for Optuna
    """
    composite = CompositeOptimizer(weights=weights)
    
    def objective(trial):
        return composite.objective_function(
            trial,
            simulate_func=optimizer_instance.simulate_multi_tf,
            df_dict=df_dict,
            param_ranges=param_ranges,
            n_wfa_folds=4
        )
    
    return objective