"""
Multi-Timeframe Optimization Engine - FIXED VERSION
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
import optuna

from optimization.metrics import PerformanceMetrics
from config.settings import RETRACEMENT_ZONES

from optimization.psr_composite import (
    CompositeOptimizer,
    CompositeWeights,
    PSRCalculator,
    WalkForwardLite,
    PBOCalculator,
    TurnoverCalculator
)

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MultiTimeframeOptimizer(QThread):
    """
    Multi-timeframe strategy optimizer with sequential optimization
    """
    progress = pyqtSignal(int)
    new_best = pyqtSignal(dict)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    phase_update = pyqtSignal(str)
    stopped = False

    def __init__(
        self,
        df_dict: Dict[str, pd.DataFrame],
        n_trials: int,
        time_cycle_ranges: Tuple,
        mn1_range: Tuple[int, int],
        mn2_range: Tuple[int, int],
        entry_range: Tuple[float, float],
        exit_range: Tuple[float, float],
        ticker: str = "",
        timeframes: Optional[List[str]] = None,
        optimize_equity_curve: bool = False,
        batch_size: int = 500,
        transaction_costs: Optional['TransactionCosts'] = None,
        composite_weights: Optional[CompositeWeights] = None
    ):
        super().__init__()
        self.df_dict = df_dict
        self.timeframes = timeframes or list(df_dict.keys())
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.time_cycle_ranges = time_cycle_ranges
        self.mn1_range = mn1_range
        self.mn2_range = mn2_range
        self.entry_range = entry_range
        self.exit_range = exit_range
        self.ticker = ticker
        self.optimize_equity_curve = optimize_equity_curve
        self.all_results = []
        self.best_params_per_tf = {}
        self.base_eq_curve = None
        self.stopped = False
        
        # Transaction costs
        if transaction_costs is None:
            from config.settings import TransactionCosts
            self.transaction_costs = TransactionCosts()
        else:
            self.transaction_costs = transaction_costs
        
        # Composite optimization setup
        self.composite_optimizer = CompositeOptimizer(
            weights=composite_weights,
            benchmark_sharpe=0.0,
            max_acceptable_turnover=100.0,
            max_acceptable_dd=0.50
        )
        
        # Create Optuna study with persistence
        self.study = CompositeOptimizer.create_optuna_study(
            study_name=f"{ticker}_composite_opt" if ticker else "composite_opt",
            storage_path=f"sqlite:///optuna_{ticker}.db" if ticker else "sqlite:///optuna.db",
            load_if_exists=True
        )
        
        # Pre-process data
        self._preprocess_data()
        self._align_timeframes()

    def calculate_composite_metrics(self, params: Dict) -> Dict[str, float]:
        """
        Calculate composite optimization metrics for given parameters
        
        Returns all component scores plus composite total
        """
        # Run full backtest
        eq_curve, trade_count = self.simulate_multi_tf(params)
        
        if eq_curve is None:
            return {
                'composite_score': 0.0,
                'psr': 0.0,
                'wfa_sharpe': 0.0,
                'pbo': 1.0,
                'annual_turnover': 0.0,
                'max_drawdown': 1.0
            }
        
        # Run lightweight WFA
        _, oos_curves = WalkForwardLite.evaluate_wfa(
            simulate_func=self.simulate_multi_tf,
            params=params,
            df_dict=self.df_dict,
            n_folds=4
        )
        
        # Calculate total days
        df_finest = self.df_dict[self.finest_tf]
        if 'Datetime' in df_finest.columns:
            total_days = (df_finest['Datetime'].max() - df_finest['Datetime'].min()).days
        else:
            total_days = len(df_finest)
        
        # Get annualization factor
        if self.finest_tf == 'daily':
            ann_factor = 252.0
        elif self.finest_tf == 'hourly':
            ann_factor = 252.0 * 6.5
        else:  # 5min
            ann_factor = 252.0 * 6.5 * 12
        
        # Calculate composite score
        scores = self.composite_optimizer.calculate_composite_score(
            eq_curve,
            trade_count,
            max(total_days, 1),
            oos_curves,
            annualization_factor=ann_factor
        )
        
        return scores

    def _preprocess_data(self):
        """Convert all DataFrames to numpy arrays once for speed"""
        print("🚀 Pre-processing data to numpy arrays...")
        self.np_data = {}
        
        for tf in self.timeframes:
            df = self.df_dict[tf]
            self.np_data[tf] = {
                'close': df["Close"].to_numpy(dtype=np.float64),
                'open': df["Open"].to_numpy(dtype=np.float64),
                'datetime': df["Datetime"].values,
                'length': len(df)
            }
            print(f"  {tf}: {self.np_data[tf]['length']} bars cached")

    def _align_timeframes(self):
        """Align all timeframes to the finest granularity"""
        tf_order = {'1min': 0, '5min': 1, 'hourly': 2, 'daily': 3}
        sorted_tfs = sorted(self.timeframes, key=lambda x: tf_order.get(x, 99))
        finest_tf = sorted_tfs[0]
        
        finest_df = self.df_dict[finest_tf].copy()
        
        # Ensure Datetime column exists
        for tf in self.timeframes:
            if 'Datetime' not in self.df_dict[tf].columns:
                self.df_dict[tf]['Datetime'] = pd.to_datetime(self.df_dict[tf].index)
        
        # Pre-compute all index mappings once
        self.tf_indices = {}
        self.tf_indices[finest_tf] = np.arange(len(finest_df))
        
        for tf in self.timeframes:
            if tf == finest_tf:
                continue
                
            coarse_df = self.df_dict[tf]
            indices = np.full(len(finest_df), -1, dtype=np.int32)
            
            for idx, coarse_date in enumerate(coarse_df['Datetime']):
                if tf == 'daily':
                    mask = finest_df['Datetime'].dt.date == coarse_date.date()
                elif tf == 'hourly':
                    mask = (
                        (finest_df['Datetime'].dt.date == coarse_date.date()) &
                        (finest_df['Datetime'].dt.hour == coarse_date.hour)
                    )
                elif tf == '5min':
                    mask = (
                        (finest_df['Datetime'].dt.date == coarse_date.date()) &
                        (finest_df['Datetime'].dt.hour == coarse_date.hour) &
                        ((finest_df['Datetime'].dt.minute // 5) == (coarse_date.minute // 5))
                    )
                else:
                    continue
                    
                indices[mask] = idx
            
            self.tf_indices[tf] = indices
            print(f"  Aligned {tf} to {finest_tf}")
        
        self.finest_tf = finest_tf

    def simulate_multi_tf(self, params: Dict, return_trades: bool = False):
        """
        Heavily optimized backtest using pre-computed numpy arrays
        
        Args:
            params: Strategy parameters
            return_trades: Whether to return trade log
            
        Returns:
            Tuple of (equity_curve, trade_count, [trades])
        """
        if self.stopped:
            return (None, 0, []) if return_trades else (None, 0)

        try:
            # Get finest timeframe data
            close_finest = self.np_data[self.finest_tf]['close']
            open_finest = self.np_data[self.finest_tf]['open']
            datetime_finest = self.np_data[self.finest_tf]['datetime']
            n_bars = len(close_finest)
            
            # Pre-allocate all signal arrays
            enter_signal = np.ones(n_bars, dtype=bool)
            exit_signal = np.zeros(n_bars, dtype=bool)
            
            # Calculate signals for each timeframe
            # Calculate signals for each timeframe
            for tf in self.timeframes:
                mn1 = int(params[f'MN1_{tf}'])
                mn2 = int(params[f'MN2_{tf}'])
                entry = params[f'Entry_{tf}']
                exit_val = params[f'Exit_{tf}']
                
                # ✅ DEBUG: Print what parameters are actually being used
                print(f"  🔍 {tf}: MN1={mn1}, MN2={mn2}, Entry<{entry:.1f}, Exit>{exit_val:.1f}")
                
                # Use cached numpy arrays
                close_tf = self.np_data[tf]['close']
                
                # Vectorized RSI calculation
                rsi = PerformanceMetrics.compute_rsi_vectorized(close_tf, mn1)
                rsi_smooth = PerformanceMetrics.smooth_vectorized(rsi, mn2)
                
                # ✅ DEBUG: Print RSI statistics
                print(f"     RSI range: {np.nanmin(rsi_smooth):.1f} to {np.nanmax(rsi_smooth):.1f}")
                print(f"     Trades possible: {np.sum(rsi_smooth < entry)} entry signals")
                # Vectorized cycle calculation
                on = int(params[f'On_{tf}'])
                off = int(params[f'Off_{tf}'])
                start = int(params[f'Start_{tf}'])
                cycle = ((np.arange(len(close_tf)) - start) % (on + off)) < on
                
                # Map to finest timeframe if needed
                if tf != self.finest_tf:
                    indices = self.tf_indices[tf]
                    valid_mask = indices >= 0
                    indices_clipped = np.clip(indices, 0, len(rsi_smooth) - 1)
                    rsi_smooth_mapped = np.zeros(n_bars)
                    cycle_mapped = np.zeros(n_bars, dtype=bool)
                    rsi_smooth_mapped[valid_mask] = rsi_smooth[indices_clipped[valid_mask]]
                    cycle_mapped[valid_mask] = cycle[indices_clipped[valid_mask]]
                else:
                    rsi_smooth_mapped = rsi_smooth
                    cycle_mapped = cycle
                
                # Vectorized signal logic
                enter_signal &= (rsi_smooth_mapped < entry) & cycle_mapped
                exit_signal |= (rsi_smooth_mapped > exit_val) | (~cycle_mapped)
            
            # Vectorized backtest simulation
            equity_curve = np.zeros(n_bars)
            equity = 1000.0
            position = False
            entry_price = 0.0
            entry_idx = 0
            trade_count = 0
            trades = [] if return_trades else None

            for i in range(n_bars):
                if not position and enter_signal[i]:
                    if i + 1 < n_bars:
                        entry_price = open_finest[i + 1]
                        entry_idx = i + 1
                        
                        # Apply transaction costs on entry
                        entry_cost_pct = self.transaction_costs.TOTAL_PCT
                        entry_price_with_costs = entry_price * (1 + entry_cost_pct)
                        
                        # Apply fixed commission if any
                        if self.transaction_costs.COMMISSION_FIXED > 0:
                            equity -= self.transaction_costs.COMMISSION_FIXED
                        
                        position = True
                        trade_count += 1
                        entry_price = entry_price_with_costs
                        
                elif position and exit_signal[i]:
                    if i + 1 < n_bars:
                        exit_price = open_finest[i + 1]
                        exit_idx = i + 1
                    else:
                        exit_price = open_finest[i]
                        exit_idx = i
                    
                    # Apply transaction costs on exit
                    exit_cost_pct = self.transaction_costs.TOTAL_PCT
                    exit_price_with_costs = exit_price * (1 - exit_cost_pct)
                    
                    # Calculate profit/loss
                    pct_change = (exit_price_with_costs / entry_price - 1) * 100
                    equity *= exit_price_with_costs / entry_price
                    
                    # Apply fixed commission if any
                    if self.transaction_costs.COMMISSION_FIXED > 0:
                        equity -= self.transaction_costs.COMMISSION_FIXED
                    
                    if return_trades:
                        trades.append({
                            'Entry_Date': datetime_finest[entry_idx],
                            'Entry_Price': entry_price,
                            'Exit_Date': datetime_finest[exit_idx],
                            'Exit_Price': exit_price_with_costs,
                            'Percent_Change': pct_change,
                            'Equity_Value': equity,
                            'Transaction_Cost_Entry': entry_cost_pct * 100,
                            'Transaction_Cost_Exit': exit_cost_pct * 100,
                            'Total_Cost_PCT': (entry_cost_pct + exit_cost_pct) * 100
                        })
                    
                    position = False
                    
                equity_curve[i] = equity * (open_finest[i] / entry_price) if position else equity

            # Close any open position at end
            if position:
                exit_price = open_finest[-1]
                exit_cost_pct = self.transaction_costs.TOTAL_PCT
                exit_price_with_costs = exit_price * (1 - exit_cost_pct)
                
                pct_change = (exit_price_with_costs / entry_price - 1) * 100
                equity *= exit_price_with_costs / entry_price
                
                if self.transaction_costs.COMMISSION_FIXED > 0:
                    equity -= self.transaction_costs.COMMISSION_FIXED
                
                if return_trades:
                    trades.append({
                        'Entry_Date': datetime_finest[entry_idx],
                        'Entry_Price': entry_price,
                        'Exit_Date': datetime_finest[-1],
                        'Exit_Price': exit_price_with_costs,
                        'Percent_Change': pct_change,
                        'Equity_Value': equity,
                        'Transaction_Cost_Entry': entry_cost_pct * 100,
                        'Transaction_Cost_Exit': exit_cost_pct * 100,
                        'Total_Cost_PCT': (entry_cost_pct + exit_cost_pct) * 100
                    })
                
                equity_curve[-1] = equity

            if return_trades:
                return equity_curve, trade_count, trades
            return equity_curve, trade_count
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return (None, 0, []) if return_trades else (None, 0)

    def simulate_equity_curve_retracement(
        self, 
        base_eq_curve: np.ndarray, 
        entry_zone_idx: int, 
        exit_zone_idx: int
    ) -> Tuple[np.ndarray, int]:
        """Optimize trades on equity curve using retracement zones"""
        try:
            entry_zone = RETRACEMENT_ZONES[entry_zone_idx]
            exit_zone = RETRACEMENT_ZONES[exit_zone_idx]
            
            # Vectorized retracement calculation
            peak = np.maximum.accumulate(base_eq_curve)
            retracement = (peak - base_eq_curve) / peak
            
            enter_signal = (retracement >= entry_zone[0]) & (retracement <= entry_zone[1])
            exit_signal = (
                ((retracement >= exit_zone[0]) & (retracement <= exit_zone[1])) | 
                (retracement < 0.001)
            )
            
            # Vectorized trade execution
            final_equity_curve = np.zeros(len(base_eq_curve))
            portfolio_equity = 1000.0
            position = False
            entry_eq = 0.0
            trade_count = 0
            
            for i in range(len(base_eq_curve)):
                if not position and enter_signal[i]:
                    entry_eq = base_eq_curve[i]
                    position = True
                    trade_count += 1
                elif position and exit_signal[i]:
                    exit_eq = base_eq_curve[i]
                    portfolio_equity *= exit_eq / entry_eq
                    position = False
                
                final_equity_curve[i] = (
                    portfolio_equity * (base_eq_curve[i] / entry_eq) 
                    if position 
                    else portfolio_equity
                )
            
            if position:
                exit_eq = base_eq_curve[-1]
                portfolio_equity *= exit_eq / entry_eq
                final_equity_curve[-1] = portfolio_equity
            
            return final_equity_curve, trade_count
            
        except Exception as e:
            print(f"Equity curve retracement optimization error: {e}")
            return base_eq_curve, 0

    def run(self):
        """Sequential optimization using PSR composite objective"""
        try:
            print(f"\n{'='*60}")
            print(f"PSR COMPOSITE OPTIMIZATION")
            print(f"Ticker: {self.ticker}")
            print(f"Timeframes: {self.timeframes}")
            print(f"Total trials: {self.n_trials}")
            print(f"{'='*60}\n")
            
            print(f"📊 Optimization Weights:")
            print(f"   PSR: {self.composite_optimizer.weights.psr:.2f}")
            print(f"   WFA Sharpe: {self.composite_optimizer.weights.wfa_sharpe:.2f}")
            print(f"   PBO Penalty: {self.composite_optimizer.weights.pbo_penalty:.2f}")
            print(f"   Turnover Penalty: {self.composite_optimizer.weights.turnover:.2f}")
            print(f"   Drawdown Penalty: {self.composite_optimizer.weights.drawdown:.2f}")
            print()
            
            on_range, off_range, start_range = self.time_cycle_ranges
            
            phases_per_tf = 2
            total_phases = len(self.timeframes) * phases_per_tf
            trials_per_phase = max(50, self.n_trials // (len(self.timeframes) * phases_per_tf))
            
            phase_counter = 0
            
            # Optimize each timeframe sequentially
            for tf_idx, tf in enumerate(self.timeframes):
                if self.stopped:
                    break
                
                # PHASE: Optimize Cycle
                phase_counter += 1
                phase_msg = f"Phase {phase_counter}/{total_phases}: {tf.upper()} Time Cycle..."
                self.phase_update.emit(phase_msg)
                print(f"\n{'='*60}")
                print(f"PHASE {phase_counter}: {tf.upper()} TIME CYCLE")
                print(f"{'='*60}")
                
                # ✅ FIX: Create NEW study for cycle phase
                cycle_study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=10,
                        multivariate=False,  # ✅ FIX: Set to False to avoid warnings
                        warn_independent_sampling=False
                    )
                )
                
                trial_count = [0]
                
                def objective_cycle(trial):
                    if self.stopped:
                        raise optuna.exceptions.OptunaError("Stopped by user")
                    
                    trial_count[0] += 1
                    
                    if trial_count[0] % 50 == 0:
                        print(f"  Phase {phase_counter} - Trial {trial_count[0]}/{trials_per_phase}")
                    
                    # Build params
                    params = {}
                    
                    # Add previous TF params
                    for prev_tf in self.timeframes[:tf_idx]:
                        params.update(self.best_params_per_tf[prev_tf])
                    
                    # Default values for current TF
                    mn1_default = (self.mn1_range[0] + self.mn1_range[1]) // 2
                    mn2_default = (self.mn2_range[0] + self.mn2_range[1]) // 2
                    entry_default = (self.entry_range[0] + self.entry_range[1]) / 2
                    exit_default = (self.exit_range[0] + self.exit_range[1]) / 2
                    
                    params[f'MN1_{tf}'] = mn1_default
                    params[f'MN2_{tf}'] = mn2_default
                    params[f'Entry_{tf}'] = entry_default
                    params[f'Exit_{tf}'] = exit_default
                    
                    # Optimize cycle params
                    params[f'On_{tf}'] = trial.suggest_int(f"On_{tf}", *on_range)
                    params[f'Off_{tf}'] = trial.suggest_int(f"Off_{tf}", *off_range)
                    params[f'Start_{tf}'] = trial.suggest_int(f"Start_{tf}", 0, on_range[1] + off_range[1])
                    
                    # Add future TF defaults
                    for future_tf in self.timeframes[tf_idx+1:]:
                        params[f'MN1_{future_tf}'] = mn1_default
                        params[f'MN2_{future_tf}'] = mn2_default
                        params[f'Entry_{future_tf}'] = entry_default
                        params[f'Exit_{future_tf}'] = exit_default
                        params[f'On_{future_tf}'] = on_range[0]
                        params[f'Off_{future_tf}'] = off_range[0]
                        params[f'Start_{future_tf}'] = 0
                    
                    # Calculate composite score (lightweight for cycle phase)
                    eq_curve, trades = self.simulate_multi_tf(params)
                    
                    if eq_curve is None or len(eq_curve) < 50:
                        return 0.0
                    
                    sharpe = PSRCalculator.calculate_sharpe_from_equity(eq_curve)
                    trade_penalty = min(trades / 50.0, 1.0)
                    score = sharpe - trade_penalty
                    
                    # Update progress
                    progress_pct = ((phase_counter - 1) / total_phases) * 100
                    phase_pct = (trial_count[0] / trials_per_phase) * (100 / total_phases)
                    self.progress.emit(int(progress_pct + phase_pct))
                    
                    return score
                
                cycle_study.optimize(objective_cycle, n_trials=trials_per_phase, n_jobs=1,
                                catch=(Exception,), show_progress_bar=False)
                
                # Store cycle params
                best_cycle = {
                    f'On_{tf}': cycle_study.best_params[f'On_{tf}'],
                    f'Off_{tf}': cycle_study.best_params[f'Off_{tf}'],
                    f'Start_{tf}': cycle_study.best_params[f'Start_{tf}']
                }
                
                if tf not in self.best_params_per_tf:
                    self.best_params_per_tf[tf] = {}
                self.best_params_per_tf[tf].update(best_cycle)
                
                print(f"✓ Phase {phase_counter} Complete - Cycle params optimized")
                print(f"  Stored: {best_cycle}")
                
                # PHASE: Optimize RSI
                phase_counter += 1
                self.phase_update.emit(f"Phase {phase_counter}/{total_phases}: {tf.upper()} RSI...")
                print(f"\nPHASE {phase_counter}: {tf.upper()} RSI (COMPOSITE)")
                
                # ✅ FIX: Create NEW study for RSI phase
                rsi_study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=10,
                        multivariate=False,  # ✅ FIX: Set to False
                        warn_independent_sampling=False
                    )
                )
                
                trial_count = [0]
                
                def objective_rsi(trial):
                    if self.stopped:
                        raise optuna.exceptions.OptunaError("Stopped by user")
                    
                    trial_count[0] += 1
                    
                    if trial_count[0] % 50 == 0:
                        print(f"  Phase {phase_counter} - Trial {trial_count[0]}/{trials_per_phase}")
                    
                    # Build params
                    params = {}
                    for prev_tf in self.timeframes[:tf_idx]:
                        params.update(self.best_params_per_tf[prev_tf])
                    
                    params.update(best_cycle)
                    
                    # Optimize RSI params
                    params[f'MN1_{tf}'] = trial.suggest_int(f"MN1_{tf}", *self.mn1_range)
                    params[f'MN2_{tf}'] = trial.suggest_int(f"MN2_{tf}", *self.mn2_range)
                    params[f'Entry_{tf}'] = trial.suggest_float(f"Entry_{tf}", *self.entry_range, step=0.5)
                    params[f'Exit_{tf}'] = trial.suggest_float(f"Exit_{tf}", *self.exit_range, step=0.5)
                    
                    # Add future TF defaults
                    for future_tf in self.timeframes[tf_idx+1:]:
                        mn1_default = (self.mn1_range[0] + self.mn1_range[1]) // 2
                        mn2_default = (self.mn2_range[0] + self.mn2_range[1]) // 2
                        entry_default = (self.entry_range[0] + self.entry_range[1]) / 2
                        exit_default = (self.exit_range[0] + self.exit_range[1]) / 2
                        
                        params[f'MN1_{future_tf}'] = mn1_default
                        params[f'MN2_{future_tf}'] = mn2_default
                        params[f'Entry_{future_tf}'] = entry_default
                        params[f'Exit_{future_tf}'] = exit_default
                        params[f'On_{future_tf}'] = on_range[0]
                        params[f'Off_{future_tf}'] = off_range[0]
                        params[f'Start_{future_tf}'] = 0
                    
                    # Calculate FULL composite score
                    scores = self.calculate_composite_metrics(params)
                    
                    # Store component scores
                    trial.set_user_attr('psr', scores['psr'])
                    trial.set_user_attr('wfa_sharpe', scores['wfa_sharpe'])
                    trial.set_user_attr('pbo', scores['pbo'])
                    trial.set_user_attr('annual_turnover', scores['annual_turnover'])
                    trial.set_user_attr('max_drawdown', scores['max_drawdown'])
                    
                    # Update progress
                    progress_pct = ((phase_counter - 1) / total_phases) * 100
                    phase_pct = (trial_count[0] / trials_per_phase) * (100 / total_phases)
                    self.progress.emit(int(progress_pct + phase_pct))
                    
                    return scores['composite_score']
                
                rsi_study.optimize(objective_rsi, n_trials=trials_per_phase, n_jobs=1,
                                catch=(Exception,), show_progress_bar=False)
                
                # Store RSI params
                best_rsi = {
                    f'MN1_{tf}': rsi_study.best_params[f'MN1_{tf}'],
                    f'MN2_{tf}': rsi_study.best_params[f'MN2_{tf}'],
                    f'Entry_{tf}': rsi_study.best_params[f'Entry_{tf}'],
                    f'Exit_{tf}': rsi_study.best_params[f'Exit_{tf}']
                }
                
                self.best_params_per_tf[tf].update(best_rsi)
                
                print(f"✓ Phase {phase_counter} Complete - RSI optimized")
                print(f"  Stored: {best_rsi}")
                print(f"  Complete params for {tf}: {self.best_params_per_tf[tf]}")
            
            # Compile final results
            print(f"\n{'='*60}")
            print(f"Compiling final results...")
            print(f"Timeframes optimized: {list(self.best_params_per_tf.keys())}")
            
            base_params = {}
            for tf in self.timeframes:
                if tf in self.best_params_per_tf:
                    base_params.update(self.best_params_per_tf[tf])
                    print(f"  Added {tf}: {len(self.best_params_per_tf[tf])} params")
                else:
                    print(f"  ⚠️  WARNING: {tf} not in best_params_per_tf!")
            
            print(f"\nFinal params keys: {list(base_params.keys())}")
            
            # Calculate ALL metrics for display
            base_eq_curve, base_trade_count = self.simulate_multi_tf(base_params)
            
            if base_eq_curve is None:
                raise ValueError("Final simulation failed - no equity curve generated")
            
            base_metrics = PerformanceMetrics.calculate_metrics(base_eq_curve)
            
            if base_metrics is None:
                raise ValueError("Failed to calculate performance metrics")
            
            base_metrics['Trade_Count'] = base_trade_count
            
            # Add composite metrics
            composite_scores = self.calculate_composite_metrics(base_params)
            base_metrics.update({
                'Composite_Score': composite_scores['composite_score'],
                'PSR': composite_scores['psr'],
                'WFA_Sharpe': composite_scores['wfa_sharpe'],
                'PBO': composite_scores['pbo'],
                'Annual_Turnover': composite_scores['annual_turnover']
            })
            
            # Save results
            final_result = {**base_params, **base_metrics, 'Curve_Optimized': False}
            self.all_results.append(final_result)
            self.new_best.emit(final_result)
            
            df_results = pd.DataFrame([final_result])
            filename = f"{self.ticker}_psr_composite.csv"
            df_results.to_csv(filename, index=False)
            
            print(f"\n{'='*60}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Composite Score: {composite_scores['composite_score']:.3f}")
            print(f"  PSR: {composite_scores['psr']:.3f}")
            print(f"  WFA Sharpe: {composite_scores['wfa_sharpe']:.2f}")
            print(f"  PBO: {composite_scores['pbo']:.3f}")
            print(f"  Annual Turnover: {composite_scores['annual_turnover']:.1f}")
            print(f"  Max DD: {composite_scores['max_drawdown']*100:.1f}%")
            
            print(f"\n📊 Traditional Metrics:")
            print(f"  Return: {final_result['Percent_Gain_%']:.2f}%")
            print(f"  Sortino: {final_result['Sortino_Ratio']:.2f}")
            print(f"  Profit Factor: {final_result['Profit_Factor']:.2f}")
            print(f"  Trades: {final_result['Trade_Count']}")
            
            # ✅ FIX: Display all optimized parameters by timeframe
            print(f"\n⚙️  Optimized Parameters:")
            for tf in self.timeframes:
                if tf in self.best_params_per_tf:
                    params = self.best_params_per_tf[tf]
                    print(f"\n  {tf.upper()}:")
                    
                    # Time Cycle
                    if f'On_{tf}' in params:
                        print(f"    Time Cycle: ON={params[f'On_{tf}']}, OFF={params[f'Off_{tf}']}, START={params[f'Start_{tf}']}")
                    
                    # RSI
                    if f'MN1_{tf}' in params:
                        print(f"    RSI: MN1={params[f'MN1_{tf}']}, MN2={params[f'MN2_{tf}']}")
                        print(f"    Thresholds: ENTRY<{params[f'Entry_{tf}']:.1f}, EXIT>{params[f'Exit_{tf}']:.1f}")
            
            print(f"{'='*60}\n")
            
            self.finished.emit(df_results)
            
        except Exception as e:
            if not self.stopped:
                error_msg = f"Optimization error: {e}"
                print(f"\n✗ {error_msg}")
                self.error.emit(error_msg)
                import traceback
                traceback.print_exc()