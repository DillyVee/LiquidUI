"""
Multi-Timeframe Optimization Engine - COMPLETE WITH ORGANIZED STORAGE
PSR calculation NO LONGER includes walk-forward analysis
Walk-forward is now a separate button/function
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from config.settings import RETRACEMENT_ZONES, Paths
from optimization.metrics import PerformanceMetrics
from optimization.psr_composite import PSRCalculator

if TYPE_CHECKING:
    from config.settings import TransactionCosts

optuna.logging.set_verbosity(optuna.logging.WARNING)


class MultiTimeframeOptimizer(QThread):
    """Multi-timeframe strategy optimizer with PSR (no WFA in optimization)"""

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
        batch_size: int = 500,
        transaction_costs: Optional["TransactionCosts"] = None,
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
        self.all_results = []
        self.best_params_per_tf = {}
        self.base_eq_curve = None
        self.stopped = False

        if transaction_costs is None:
            from config.settings import TransactionCosts

            self.transaction_costs = TransactionCosts()
        else:
            self.transaction_costs = transaction_costs

        storage_path = Paths.get_optuna_path(ticker) if ticker else None

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=30, multivariate=False, warn_independent_sampling=False
            ),
            study_name=f"{ticker}_psr_opt" if ticker else "psr_opt",
            storage=storage_path,
            load_if_exists=True if storage_path else False,
        )

        self._preprocess_data()
        self._align_timeframes()

    def _preprocess_data(self):
        """Convert all DataFrames to numpy arrays once for speed"""
        print("🚀 Pre-processing data to numpy arrays...")
        self.np_data = {}

        for tf in self.timeframes:
            df = self.df_dict[tf]
            self.np_data[tf] = {
                "close": df["Close"].to_numpy(dtype=np.float64),
                "open": df["Open"].to_numpy(dtype=np.float64),
                "datetime": df["Datetime"].values,
                "length": len(df),
            }
            print(f"  {tf}: {self.np_data[tf]['length']} bars cached")

    def _align_timeframes(self):
        """Align all timeframes to the finest granularity"""
        tf_order = {"1min": 0, "5min": 1, "hourly": 2, "daily": 3}
        sorted_tfs = sorted(self.timeframes, key=lambda x: tf_order.get(x, 99))
        finest_tf = sorted_tfs[0]

        finest_df = self.df_dict[finest_tf].copy()

        # Ensure Datetime column exists
        for tf in self.timeframes:
            if "Datetime" not in self.df_dict[tf].columns:
                self.df_dict[tf]["Datetime"] = pd.to_datetime(self.df_dict[tf].index)

        # Pre-compute all index mappings once
        self.tf_indices = {}
        self.tf_indices[finest_tf] = np.arange(len(finest_df))

        for tf in self.timeframes:
            if tf == finest_tf:
                continue

            coarse_df = self.df_dict[tf]
            indices = np.full(len(finest_df), -1, dtype=np.int32)

            for idx, coarse_date in enumerate(coarse_df["Datetime"]):
                if tf == "daily":
                    mask = finest_df["Datetime"].dt.date == coarse_date.date()
                elif tf == "hourly":
                    mask = (finest_df["Datetime"].dt.date == coarse_date.date()) & (
                        finest_df["Datetime"].dt.hour == coarse_date.hour
                    )
                elif tf == "5min":
                    mask = (
                        (finest_df["Datetime"].dt.date == coarse_date.date())
                        & (finest_df["Datetime"].dt.hour == coarse_date.hour)
                        & (
                            (finest_df["Datetime"].dt.minute // 5)
                            == (coarse_date.minute // 5)
                        )
                    )
                else:
                    continue

                indices[mask] = idx

            self.tf_indices[tf] = indices
            print(f"  Aligned {tf} to {finest_tf}")

        self.finest_tf = finest_tf

    def calculate_psr(self, params: Dict) -> Tuple[float, float]:
        """
        Calculate PSR and Sharpe for given parameters
        Returns (psr, sharpe) tuple
        """
        # Run full backtest
        eq_curve, trade_count = self.simulate_multi_tf(params)

        if eq_curve is None or len(eq_curve) < 50 or trade_count < 10:
            return 0.0, 0.0

        # Calculate returns
        returns = np.diff(eq_curve) / eq_curve[:-1]
        returns = returns[~(np.isnan(returns) | np.isinf(returns))]

        if len(returns) < 30:
            return 0.0, 0.0

        # Get annualization factor based on finest timeframe
        if self.finest_tf == "daily":
            ann_factor = 252.0
        elif self.finest_tf == "hourly":
            ann_factor = 252.0 * 6.5
        else:  # 5min
            ann_factor = 252.0 * 6.5 * 12

        # Calculate PSR (with trade-count awareness for realistic confidence)
        psr = PSRCalculator.calculate_psr(
            returns,
            benchmark_sharpe=0.0,
            annualization_factor=ann_factor,
            trade_count=trade_count,
        )

        # Calculate Sharpe Ratio
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)
            sharpe = np.clip(sharpe, -5, 10)
        else:
            sharpe = 0.0

        return float(psr), float(sharpe)

    def simulate_multi_tf(self, params: Dict, return_trades: bool = False):
        """
        FIXED VERSION - Now properly returns trade log with actual returns
        """
        if self.stopped:
            return (None, 0, []) if return_trades else (None, 0)

        try:
            # Get finest timeframe data
            close_finest = self.np_data[self.finest_tf]["close"]
            open_finest = self.np_data[self.finest_tf]["open"]
            datetime_finest = self.np_data[self.finest_tf]["datetime"]
            n_bars = len(close_finest)

            # Pre-allocate signal arrays
            enter_signal = np.ones(n_bars, dtype=bool)
            exit_signal = np.zeros(n_bars, dtype=bool)

            # Calculate signals for each timeframe
            for tf in self.timeframes:
                mn1 = int(params[f"MN1_{tf}"])
                mn2 = int(params[f"MN2_{tf}"])
                entry = params[f"Entry_{tf}"]
                exit_val = params[f"Exit_{tf}"]

                close_tf = self.np_data[tf]["close"]

                # Vectorized RSI
                rsi = PerformanceMetrics.compute_rsi_vectorized(close_tf, mn1)
                rsi_smooth = PerformanceMetrics.smooth_vectorized(rsi, mn2)

                # Vectorized cycle
                on = int(params[f"On_{tf}"])
                off = int(params[f"Off_{tf}"])
                start = int(params[f"Start_{tf}"])
                cycle = ((np.arange(len(close_tf)) - start) % (on + off)) < on

                # Map to finest timeframe
                if tf != self.finest_tf:
                    indices = self.tf_indices[tf]
                    valid_mask = indices >= 0
                    indices_clipped = np.clip(indices, 0, len(rsi_smooth) - 1)
                    rsi_smooth_mapped = np.zeros(n_bars)
                    cycle_mapped = np.zeros(n_bars, dtype=bool)
                    rsi_smooth_mapped[valid_mask] = rsi_smooth[
                        indices_clipped[valid_mask]
                    ]
                    cycle_mapped[valid_mask] = cycle[indices_clipped[valid_mask]]
                else:
                    rsi_smooth_mapped = rsi_smooth
                    cycle_mapped = cycle

                # Signals
                enter_signal &= (rsi_smooth_mapped < entry) & cycle_mapped
                exit_signal |= (rsi_smooth_mapped > exit_val) | (~cycle_mapped)

            # Backtest simulation
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

                        # Apply costs
                        entry_cost_pct = self.transaction_costs.TOTAL_PCT
                        entry_price_with_costs = entry_price * (1 + entry_cost_pct)

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

                    exit_cost_pct = self.transaction_costs.TOTAL_PCT
                    exit_price_with_costs = exit_price * (1 - exit_cost_pct)

                    # Calculate actual percent change
                    pct_change = (exit_price_with_costs / entry_price - 1) * 100

                    # Update equity
                    equity_before = equity
                    equity *= exit_price_with_costs / entry_price

                    if self.transaction_costs.COMMISSION_FIXED > 0:
                        equity -= self.transaction_costs.COMMISSION_FIXED

                    # Store trades with ACTUAL returns
                    if return_trades:
                        trades.append(
                            {
                                "Entry_Date": datetime_finest[entry_idx],
                                "Entry_Price": entry_price,
                                "Exit_Date": datetime_finest[exit_idx],
                                "Exit_Price": exit_price_with_costs,
                                "Percent_Change": pct_change,
                                "Equity_Before": equity_before,
                                "Equity_After": equity,
                                "Transaction_Cost_Entry": entry_cost_pct * 100,
                                "Transaction_Cost_Exit": exit_cost_pct * 100,
                                "Total_Cost_PCT": (entry_cost_pct + exit_cost_pct)
                                * 100,
                            }
                        )

                    position = False

                equity_curve[i] = (
                    equity * (open_finest[i] / entry_price) if position else equity
                )

            # Close final position
            if position:
                exit_price = open_finest[-1]
                exit_cost_pct = self.transaction_costs.TOTAL_PCT
                exit_price_with_costs = exit_price * (1 - exit_cost_pct)

                pct_change = (exit_price_with_costs / entry_price - 1) * 100
                equity_before = equity
                equity *= exit_price_with_costs / entry_price

                if self.transaction_costs.COMMISSION_FIXED > 0:
                    equity -= self.transaction_costs.COMMISSION_FIXED

                if return_trades:
                    trades.append(
                        {
                            "Entry_Date": datetime_finest[entry_idx],
                            "Entry_Price": entry_price,
                            "Exit_Date": datetime_finest[-1],
                            "Exit_Price": exit_price_with_costs,
                            "Percent_Change": pct_change,
                            "Equity_Before": equity_before,
                            "Equity_After": equity,
                            "Transaction_Cost_Entry": entry_cost_pct * 100,
                            "Transaction_Cost_Exit": exit_cost_pct * 100,
                            "Total_Cost_PCT": (entry_cost_pct + exit_cost_pct) * 100,
                        }
                    )

                equity_curve[-1] = equity

            if return_trades:
                return equity_curve, trade_count, trades

            return equity_curve, trade_count

        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback

            traceback.print_exc()
            return (None, 0, []) if return_trades else (None, 0)

    def run(self):
        """Sequential optimization with parallel batching and incremental CSV saves"""
        try:
            import gc
            import multiprocessing

            # Determine optimal CPU usage
            n_cpus = multiprocessing.cpu_count()
            n_jobs = max(1, n_cpus - 1)  # Leave 1 core for OS

            print(f"\n{'='*60}")
            print(f"PSR COMPOSITE OPTIMIZATION (PARALLEL BATCHED)")
            print(f"{'='*60}")
            print(f"Ticker: {self.ticker}")
            print(f"Timeframes: {self.timeframes}")
            print(f"Total trials: {self.n_trials}")
            print(f"Batch size: {self.batch_size}")
            print(f"CPU cores: {n_cpus} (using {n_jobs} for optimization)")
            print(f"{'='*60}\n")

            print("📊 Using PSR Composite Optimization")
            print(f"   Parallel Processing: {n_jobs} workers per batch")
            print(f"   Memory Management: Clear after each batch")
            print(f"   Incremental Saves: Append to CSV after each batch")
            print()

            on_range, off_range, start_range = self.time_cycle_ranges

            phases_per_tf = 2
            total_phases = len(self.timeframes) * phases_per_tf
            trials_per_phase = max(
                50, self.n_trials // (len(self.timeframes) * phases_per_tf)
            )

            # Calculate batches per phase
            batches_per_phase = max(1, trials_per_phase // self.batch_size)
            trials_per_batch = trials_per_phase // batches_per_phase

            print(f"📦 Batch Configuration:")
            print(f"   Trials per phase: {trials_per_phase}")
            print(f"   Batches per phase: {batches_per_phase}")
            print(f"   Trials per batch: {trials_per_batch}")
            print()

            phase_counter = 0

            # Initialize results CSV
            results_path = Paths.get_results_path(self.ticker, suffix="_psr_batched")
            csv_initialized = False

            # Optimize each timeframe sequentially
            for tf_idx, tf in enumerate(self.timeframes):
                if self.stopped:
                    break

                # ===================================================================
                # PHASE: Optimize Cycle (with batching)
                # ===================================================================
                phase_counter += 1
                phase_msg = (
                    f"Phase {phase_counter}/{total_phases}: {tf.upper()} Time Cycle..."
                )
                self.phase_update.emit(phase_msg)
                print(f"\n{'='*60}")
                print(f"PHASE {phase_counter}: {tf.upper()} TIME CYCLE (BATCHED)")
                print(f"{'='*60}")

                cycle_study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=min(10, trials_per_batch),
                        multivariate=False,
                        warn_independent_sampling=False,
                    ),
                )

                # Pre-build base parameters once (optimization!)
                base_params_cycle = {}
                for prev_tf in self.timeframes[:tf_idx]:
                    base_params_cycle.update(self.best_params_per_tf[prev_tf])

                mn1_default = (self.mn1_range[0] + self.mn1_range[1]) // 2
                mn2_default = (self.mn2_range[0] + self.mn2_range[1]) // 2
                entry_default = (self.entry_range[0] + self.entry_range[1]) / 2
                exit_default = (self.exit_range[0] + self.exit_range[1]) / 2

                # Current timeframe defaults
                base_params_cycle[f"MN1_{tf}"] = mn1_default
                base_params_cycle[f"MN2_{tf}"] = mn2_default
                base_params_cycle[f"Entry_{tf}"] = entry_default
                base_params_cycle[f"Exit_{tf}"] = exit_default

                # Future timeframes defaults
                for future_tf in self.timeframes[tf_idx + 1 :]:
                    base_params_cycle[f"MN1_{future_tf}"] = mn1_default
                    base_params_cycle[f"MN2_{future_tf}"] = mn2_default
                    base_params_cycle[f"Entry_{future_tf}"] = entry_default
                    base_params_cycle[f"Exit_{future_tf}"] = exit_default
                    base_params_cycle[f"On_{future_tf}"] = on_range[0]
                    base_params_cycle[f"Off_{future_tf}"] = off_range[0]
                    base_params_cycle[f"Start_{future_tf}"] = 0

                # Run cycle optimization in batches
                for batch_idx in range(batches_per_phase):
                    if self.stopped:
                        break

                    print(f"\n📦 Batch {batch_idx + 1}/{batches_per_phase}")

                    trial_count = [0]

                    def objective_cycle(trial):
                        if self.stopped:
                            raise optuna.exceptions.OptunaError("Stopped by user")

                        trial_count[0] += 1

                        # Copy base params (fast shallow copy)
                        params = base_params_cycle.copy()

                        # Only update current timeframe cycle params (trial-specific)
                        params[f"On_{tf}"] = trial.suggest_int(f"On_{tf}", *on_range)
                        params[f"Off_{tf}"] = trial.suggest_int(f"Off_{tf}", *off_range)
                        params[f"Start_{tf}"] = trial.suggest_int(
                            f"Start_{tf}", 0, on_range[1] + off_range[1]
                        )

                        eq_curve, trades = self.simulate_multi_tf(params)

                        if eq_curve is None or len(eq_curve) < 50:
                            return 0.0

                        sharpe = PSRCalculator.calculate_sharpe_from_equity(eq_curve)
                        trade_penalty = min(trades / 50.0, 1.0)
                        score = sharpe - trade_penalty

                        # Update progress
                        batch_progress = batch_idx / batches_per_phase
                        phase_progress = (phase_counter - 1) / total_phases
                        batch_trial_progress = (trial_count[0] / trials_per_batch) * (
                            1 / batches_per_phase
                        )
                        total_progress = (
                            phase_progress
                            + batch_progress / total_phases
                            + batch_trial_progress / total_phases
                        ) * 100
                        self.progress.emit(int(total_progress))

                        return score

                    # Run batch with parallel processing
                    cycle_study.optimize(
                        objective_cycle,
                        n_trials=trials_per_batch,
                        n_jobs=n_jobs,  # ✅ PARALLEL PROCESSING
                        catch=(Exception,),
                        show_progress_bar=False,
                    )

                    print(f"   ✓ Batch {batch_idx + 1} complete")

                    # Memory cleanup after batch
                    gc.collect()

                best_cycle = {
                    f"On_{tf}": cycle_study.best_params[f"On_{tf}"],
                    f"Off_{tf}": cycle_study.best_params[f"Off_{tf}"],
                    f"Start_{tf}": cycle_study.best_params[f"Start_{tf}"],
                }

                if tf not in self.best_params_per_tf:
                    self.best_params_per_tf[tf] = {}
                self.best_params_per_tf[tf].update(best_cycle)

                print(f"✓ Phase {phase_counter} Complete - Cycle params optimized")

                # ===================================================================
                # PHASE: Optimize RSI (with batching + CSV saves)
                # ===================================================================
                phase_counter += 1
                self.phase_update.emit(
                    f"Phase {phase_counter}/{total_phases}: {tf.upper()} RSI..."
                )
                print(f"\nPHASE {phase_counter}: {tf.upper()} RSI (BATCHED PSR)")

                rsi_study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=min(10, trials_per_batch),
                        multivariate=False,
                        warn_independent_sampling=False,
                    ),
                )

                # Pre-build base parameters once (optimization!)
                base_params_rsi = {}
                for prev_tf in self.timeframes[:tf_idx]:
                    base_params_rsi.update(self.best_params_per_tf[prev_tf])

                base_params_rsi.update(best_cycle)

                # Future timeframes defaults
                for future_tf in self.timeframes[tf_idx + 1 :]:
                    mn1_default = (self.mn1_range[0] + self.mn1_range[1]) // 2
                    mn2_default = (self.mn2_range[0] + self.mn2_range[1]) // 2
                    entry_default = (self.entry_range[0] + self.entry_range[1]) / 2
                    exit_default = (self.exit_range[0] + self.exit_range[1]) / 2

                    base_params_rsi[f"MN1_{future_tf}"] = mn1_default
                    base_params_rsi[f"MN2_{future_tf}"] = mn2_default
                    base_params_rsi[f"Entry_{future_tf}"] = entry_default
                    base_params_rsi[f"Exit_{future_tf}"] = exit_default
                    base_params_rsi[f"On_{future_tf}"] = on_range[0]
                    base_params_rsi[f"Off_{future_tf}"] = off_range[0]
                    base_params_rsi[f"Start_{future_tf}"] = 0

                # Run RSI optimization in batches with CSV saves
                for batch_idx in range(batches_per_phase):
                    if self.stopped:
                        break

                    print(f"\n📦 Batch {batch_idx + 1}/{batches_per_phase}")

                    trial_count = [0]
                    batch_results = []

                    def objective_rsi(trial):
                        if self.stopped:
                            raise optuna.exceptions.OptunaError("Stopped by user")

                        trial_count[0] += 1

                        # Copy base params (fast shallow copy)
                        params = base_params_rsi.copy()

                        # Only update current timeframe RSI params (trial-specific)
                        params[f"MN1_{tf}"] = trial.suggest_int(
                            f"MN1_{tf}", *self.mn1_range
                        )
                        params[f"MN2_{tf}"] = trial.suggest_int(
                            f"MN2_{tf}", *self.mn2_range
                        )
                        params[f"Entry_{tf}"] = trial.suggest_float(
                            f"Entry_{tf}", *self.entry_range, step=0.5
                        )
                        params[f"Exit_{tf}"] = trial.suggest_float(
                            f"Exit_{tf}", *self.exit_range, step=0.5
                        )

                        # Run simulation once and cache results
                        eq_curve, trade_count = self.simulate_multi_tf(params)

                        if eq_curve is None or len(eq_curve) < 50 or trade_count < 10:
                            psr, sharpe = 0.0, 0.0
                        else:
                            # Calculate PSR and Sharpe from cached equity curve
                            returns = np.diff(eq_curve) / eq_curve[:-1]
                            returns = returns[~(np.isnan(returns) | np.isinf(returns))]

                            if len(returns) < 30:
                                psr, sharpe = 0.0, 0.0
                            else:
                                # Get annualization factor
                                if self.finest_tf == "daily":
                                    ann_factor = 252.0
                                elif self.finest_tf == "hourly":
                                    ann_factor = 252.0 * 6.5
                                else:  # 5min or 1min
                                    ann_factor = 252.0 * 6.5 * 12

                                # Calculate PSR
                                psr = PSRCalculator.calculate_psr(
                                    returns,
                                    benchmark_sharpe=0.0,
                                    annualization_factor=ann_factor,
                                    trade_count=trade_count,
                                )

                                # Calculate Sharpe
                                mean_ret = np.mean(returns)
                                std_ret = np.std(returns, ddof=1)
                                if std_ret > 0:
                                    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)
                                    sharpe = np.clip(sharpe, -5, 10)
                                else:
                                    sharpe = 0.0

                        # Store for batch saving (no need to re-simulate!)
                        trial.set_user_attr("params", params)
                        trial.set_user_attr("psr", float(psr))
                        trial.set_user_attr("sharpe", float(sharpe))
                        trial.set_user_attr("eq_curve", eq_curve)
                        trial.set_user_attr("trade_count", trade_count)

                        # Update progress
                        batch_progress = batch_idx / batches_per_phase
                        phase_progress = (phase_counter - 1) / total_phases
                        batch_trial_progress = (trial_count[0] / trials_per_batch) * (
                            1 / batches_per_phase
                        )
                        total_progress = (
                            phase_progress
                            + batch_progress / total_phases
                            + batch_trial_progress / total_phases
                        ) * 100
                        self.progress.emit(int(total_progress))

                        return psr

                    # Run batch with parallel processing
                    rsi_study.optimize(
                        objective_rsi,
                        n_trials=trials_per_batch,
                        n_jobs=n_jobs,  # ✅ PARALLEL PROCESSING
                        catch=(Exception,),
                        show_progress_bar=False,
                    )

                    # ✅ SAVE BATCH RESULTS TO CSV
                    print(f"   💾 Saving batch results to CSV...")

                    # Get top 5 results from this batch
                    batch_trials = sorted(
                        [
                            t
                            for t in rsi_study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ],
                        key=lambda t: t.value,
                        reverse=True,
                    )[:5]

                    for trial in batch_trials:
                        params = trial.user_attrs.get("params", {})
                        if not params:
                            continue

                        # Use cached results instead of re-simulating!
                        eq_curve = trial.user_attrs.get("eq_curve")
                        trade_count = trial.user_attrs.get("trade_count", 0)

                        if eq_curve is None:
                            continue

                        metrics = PerformanceMetrics.calculate_metrics(eq_curve)
                        if metrics is None:
                            continue

                        metrics["Trade_Count"] = trade_count
                        metrics["PSR"] = trial.user_attrs.get("psr", 0.0)
                        metrics["Sharpe_Ratio"] = trial.user_attrs.get("sharpe", 0.0)
                        metrics["Batch"] = batch_idx + 1
                        metrics["Phase"] = phase_counter

                        result = {**params, **metrics, "Curve_Optimized": False}
                        batch_results.append(result)

                    # Append to CSV
                    if batch_results:
                        df_batch = pd.DataFrame(batch_results)

                        if not csv_initialized:
                            # First batch - create new file
                            df_batch.to_csv(results_path, index=False, mode="w")
                            csv_initialized = True
                            print(f"   ✓ Created CSV: {results_path}")
                        else:
                            # Append to existing file
                            df_batch.to_csv(
                                results_path, index=False, mode="a", header=False
                            )
                            print(f"   ✓ Appended {len(batch_results)} results to CSV")

                    print(f"   ✓ Batch {batch_idx + 1} complete")

                    # Memory cleanup after batch
                    batch_results.clear()
                    gc.collect()

                best_rsi = {
                    f"MN1_{tf}": rsi_study.best_params[f"MN1_{tf}"],
                    f"MN2_{tf}": rsi_study.best_params[f"MN2_{tf}"],
                    f"Entry_{tf}": rsi_study.best_params[f"Entry_{tf}"],
                    f"Exit_{tf}": rsi_study.best_params[f"Exit_{tf}"],
                }

                self.best_params_per_tf[tf].update(best_rsi)

                print(f"✓ Phase {phase_counter} Complete - RSI optimized")

            # ===================================================================
            # Compile final best result
            # ===================================================================
            print(f"\n{'='*60}")
            print(f"Compiling final best result...")

            base_params = {}
            for tf in self.timeframes:
                if tf in self.best_params_per_tf:
                    base_params.update(self.best_params_per_tf[tf])

            # Get equity curve AND trade log for GUI display
            base_eq_curve, base_trade_count, base_trades = self.simulate_multi_tf(
                base_params, return_trades=True
            )

            if base_eq_curve is None:
                raise ValueError("Final simulation failed")

            base_metrics = PerformanceMetrics.calculate_metrics(base_eq_curve)

            if base_metrics is None:
                raise ValueError("Failed to calculate performance metrics")

            base_metrics["Trade_Count"] = base_trade_count

            # Calculate PSR and Sharpe
            psr, sharpe = self.calculate_psr(base_params)
            base_metrics["PSR"] = psr
            base_metrics["Sharpe_Ratio"] = sharpe
            base_metrics["Batch"] = "FINAL"
            base_metrics["Phase"] = "FINAL"

            print(f"✓ Calculated PSR: {psr:.3f}, Sharpe: {sharpe:.2f}")

            # Append final result to CSV
            final_result = {**base_params, **base_metrics, "Curve_Optimized": False}
            self.all_results.append(final_result)
            self.new_best.emit(final_result)

            df_final = pd.DataFrame([final_result])
            df_final.to_csv(results_path, index=False, mode="a", header=False)
            print(f"✓ Appended FINAL result to CSV")

            print(f"\n✅ Complete results saved to: {results_path}")

            # Build results DataFrame for GUI with equity curve and trade log
            # Create a new DataFrame instead of loading from CSV to avoid issues with complex objects
            final_result_for_gui = final_result.copy()
            final_result_for_gui["equity_curve"] = base_eq_curve
            if base_trades:
                final_result_for_gui["trade_log"] = pd.DataFrame(base_trades)
            else:
                final_result_for_gui["trade_log"] = pd.DataFrame()

            df_results = pd.DataFrame([final_result_for_gui])

            print(f"\n{'='*60}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Total saved results: {len(df_results)}")
            print(f"Best PSR: {psr:.3f} ({psr*100:.1f}%)")
            print(f"Best Sharpe Ratio: {sharpe:.2f}")

            print(f"\n📊 Traditional Metrics:")
            print(f"  Return: {final_result['Percent_Gain_%']:.2f}%")
            print(f"  Sortino: {final_result['Sortino_Ratio']:.2f}")
            print(f"  Max Drawdown: {final_result['Max_Drawdown_%']:.2f}%")
            print(f"  Profit Factor: {final_result['Profit_Factor']:.2f}")
            print(f"  Trades: {final_result['Trade_Count']}")

            print(f"\n⚙️  Optimized Parameters:")
            for tf in self.timeframes:
                if tf in self.best_params_per_tf:
                    params = self.best_params_per_tf[tf]
                    print(f"\n  {tf.upper()}:")

                    if f"On_{tf}" in params:
                        print(
                            f"    Time Cycle: ON={params[f'On_{tf}']}, OFF={params[f'Off_{tf}']}, START={params[f'Start_{tf}']}"
                        )

                    if f"MN1_{tf}" in params:
                        print(
                            f"    RSI: MN1={params[f'MN1_{tf}']}, MN2={params[f'MN2_{tf}']}"
                        )
                        print(
                            f"    Thresholds: ENTRY<{params[f'Entry_{tf}']:.1f}, EXIT>{params[f'Exit_{tf}']:.1f}"
                        )

            print(f"{'='*60}\n")

            self.finished.emit(df_results)

        except Exception as e:
            if not self.stopped:
                error_msg = f"Optimization error: {e}"
                print(f"\n✗ {error_msg}")
                self.error.emit(error_msg)
                import traceback

                traceback.print_exc()
