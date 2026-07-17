"""
Multi-Timeframe Optimization Engine

Optimizes the time-cycle + indicator-combo strategy per timeframe with a
user-selectable objective (Sharpe, Sortino, Calmar, total return, profit
factor, win rate, expectancy - see optimization.objectives). Every trial's
full metric suite is computed once and saved, so results CSVs carry the
complete institutional picture regardless of which goal was optimized.

Modes:
  - cycle_only=True    optimize ONLY the time cycle (no indicators at all)
  - indicator_search   mix-and-match indicators from the signal library,
                       restricted to `allowed_indicators`, with an optional
                       AND-combined second leg (`combine_indicators`)
  - legacy             indicator_search=False reproduces the original
                       RSI-only parameter search

Selection instrumentation: when enough data is loaded, every trial also
records its score on a chronological TRAIN segment and a held-out
VALIDATION slice (Train_Score / Val_Score / *_Trades in every result
row) so train-vs-validation degradation is always visible.

By default this is measurement only - selection remains the full-sample
argmax. The opt-in `use_validation_veto` mode makes TPE navigate on the
train segment and picks each phase winner as the best train score among
trials with a positive validation score. The veto significantly improved
OOS results on synthetic planted-edge data (paired +0.52, p=0.022) but
FAILED to replicate on real multi-asset data (15 folds, paired -0.31,
p=0.38, stand-aside opportunity cost in trending markets), so it is not
the default - see RESEARCH_JOURNAL.md experiments H5a/H6/H7.

Walk-forward analysis stays a separate function (optimization.walk_forward)
and the anti-overfit gates live in optimization.validation.
"""

import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from config.settings import Paths
from optimization.metrics import PerformanceMetrics
from optimization.objectives import DEFAULT_OBJECTIVE, get_objective, scored
from signals.engine import CYCLE_ONLY, evaluate_combo
from signals.indicators import INDICATORS

if TYPE_CHECKING:
    from config.settings import TransactionCosts

optuna.logging.set_verbosity(optuna.logging.WARNING)


class MultiTimeframeOptimizer(QThread):
    """Multi-timeframe strategy optimizer with selectable objective"""

    # Below this many finest-timeframe bars a 25% validation slice is too
    # short for stable trade statistics; selection falls back to the
    # full-sample argmax (status quo). Keeps walk-forward inner windows
    # (~126 daily bars) unaffected.
    MIN_SELECTION_BARS = 400

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
        position_size: float = 1.0,
        indicator_search: bool = True,
        objective: str = DEFAULT_OBJECTIVE,
        cycle_only: bool = False,
        allowed_indicators: Optional[Sequence[str]] = None,
        combine_indicators: bool = True,
        seed: Optional[int] = None,
        selection_holdout_frac: float = 0.25,
        use_validation_veto: bool = False,
        vol_targeting: bool = False,
    ):
        super().__init__()
        # Validates the objective name up front (fail fast, not mid-run)
        self.objective = get_objective(objective).name

        # Search-space switches
        self.cycle_only = bool(cycle_only)
        self.indicator_search = bool(indicator_search)
        self.combine_indicators = bool(combine_indicators)
        allowed = list(allowed_indicators) if allowed_indicators else list(INDICATORS)
        unknown = [i for i in allowed if i not in INDICATORS]
        if unknown:
            raise ValueError(f"Unknown indicators {unknown} (choose from {INDICATORS})")
        if not allowed:
            raise ValueError("allowed_indicators must not be empty")
        self.allowed_indicators = allowed

        # Reproducibility: fixed sampler seed makes single-worker runs
        # repeatable (with n_jobs > 1, scheduling order still varies)
        self.seed = seed

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
        # Inputs for the anti-overfit gates (optimization.validation):
        # every trial's annualized Sharpe, and the top parameter sets of the
        # final phase as CSCV rivals
        self.trial_sharpes: List[float] = []
        self.rival_params: List[Dict] = []
        self.base_eq_curve = None
        self.stopped = False

        # All rows destined for the results CSV. The file is rewritten in
        # full after each batch: appending with header=False corrupted the
        # CSV whenever different batches had different parameter columns
        # (e.g. one batch's best used IND2, the next didn't).
        self._result_rows: List[Dict] = []

        # Fraction of equity deployed per trade. Must match live position
        # sizing or backtest metrics won't describe the live system.
        self.position_size = float(np.clip(position_size, 0.001, 1.0))

        # Opt-in volatility targeting (Moreira & Muir 2017; Harvey et al.
        # 2018): per-trade size is scaled by min(1, median_vol / vol) where
        # vol is the causal 20-bar realized volatility at the decision bar
        # and median_vol its own expanding median - "scale down when
        # volatility is elevated relative to its own history". Causal and
        # parameter-free (fixed 20-bar window, no leverage), so nothing new
        # for the optimizer to overfit. The per-bar multiplier is built in
        # _build_size_frac after data preprocessing.
        self.vol_targeting = bool(vol_targeting)

        if transaction_costs is None:
            from config.settings import TransactionCosts

            self.transaction_costs = TransactionCosts()
        else:
            # Copy so GUI spinbox edits can't mutate an in-flight optimization
            self.transaction_costs = copy.copy(transaction_costs)

        self._preprocess_data()
        self._align_timeframes()
        self._build_size_frac()

        # Chronological train/validation split. Always used to RECORD
        # per-segment scores (degradation instrumentation); it only
        # affects selection when use_validation_veto is opted in - the
        # veto failed to replicate on real multi-asset data (journal H7)
        # so measurement is on by default and action is not.
        self.use_validation_veto = bool(use_validation_veto)
        frac = float(np.clip(selection_holdout_frac, 0.0, 0.5))
        n_finest = self.np_data[self.finest_tf]["length"]
        if frac > 0.0 and n_finest >= self.MIN_SELECTION_BARS:
            self.selection_cut: Optional[int] = int(n_finest * (1.0 - frac))
            if self.use_validation_veto:
                print(
                    f"🔍 Selection: VETO mode - TPE on bars [0, {self.selection_cut}), "
                    f"winners must score > 0 on bars [{self.selection_cut}, {n_finest})"
                )
            else:
                print(
                    f"🔍 Selection: full-sample; train/validation scores recorded "
                    f"(split at bar {self.selection_cut}) for degradation visibility"
                )
        else:
            self.selection_cut = None
            if frac > 0.0:
                print(
                    f"🔍 Selection: {n_finest} bars < {self.MIN_SELECTION_BARS} - "
                    "no train/validation split (slice too short)"
                )

    def _preprocess_data(self):
        """Convert all DataFrames to numpy arrays once for speed"""
        print("🚀 Pre-processing data to numpy arrays...")
        self.np_data = {}

        for tf in self.timeframes:
            # Ensure Datetime column exists before it is read anywhere
            if "Datetime" not in self.df_dict[tf].columns:
                self.df_dict[tf]["Datetime"] = pd.to_datetime(self.df_dict[tf].index)

            df = self.df_dict[tf]
            self.np_data[tf] = {
                "close": df["Close"].to_numpy(dtype=np.float64),
                "open": df["Open"].to_numpy(dtype=np.float64),
                "datetime": df["Datetime"].values,
                "anchor": PerformanceMetrics.cycle_anchor_index(df["Datetime"].values, tf),
                "length": len(df),
            }
            print(f"  {tf}: {self.np_data[tf]['length']} bars cached")

    def _align_timeframes(self):
        """Align all timeframes to the finest granularity"""
        tf_order = {"1min": 0, "5min": 1, "hourly": 2, "daily": 3}
        sorted_tfs = sorted(self.timeframes, key=lambda x: tf_order.get(x, 99))
        finest_tf = sorted_tfs[0]

        finest_df = self.df_dict[finest_tf].copy()

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
                        & ((finest_df["Datetime"].dt.minute // 5) == (coarse_date.minute // 5))
                    )
                else:
                    continue

                indices[mask] = idx

            # Shift to the PREVIOUS completed coarse bar. A coarse bar's
            # close (and therefore its indicators) is only known once that
            # bar finishes, so mapping a coarse bar onto the finest bars
            # inside the same period would leak future information
            # (lookahead bias).
            indices = np.where(indices >= 0, indices - 1, -1).astype(np.int32)

            self.tf_indices[tf] = indices
            print(f"  Aligned {tf} to {finest_tf} (lagged one {tf} bar, no lookahead)")

        self.finest_tf = finest_tf

    # Realized-vol window for volatility targeting, in finest-tf bars.
    # Fixed (not searchable) on purpose: ~1 month of daily bars, inside the
    # 1-3 month range the volatility-targeting literature uses. Making it
    # tunable would hand the optimizer a fresh overfitting dial.
    VOL_WINDOW = 20

    def _build_size_frac(self):
        """Causal per-bar position-size multiplier for volatility targeting
        (None when disabled). size_frac[i] uses closes[: i + 1] only, so a
        trade decided at bar i (filled at open i+1) never reads the future."""
        if not self.vol_targeting:
            self._size_frac = None
            return
        closes = self.np_data[self.finest_tf]["close"]
        r = np.zeros(len(closes))
        r[1:] = np.diff(closes) / np.where(closes[:-1] > 1e-12, closes[:-1], 1.0)
        vol = (
            pd.Series(r)
            .rolling(self.VOL_WINDOW, min_periods=self.VOL_WINDOW)
            .std()
            .to_numpy()
        )
        med = pd.Series(vol).expanding(min_periods=self.VOL_WINDOW).median().to_numpy()
        scalar = np.ones(len(closes))
        valid = np.isfinite(vol) & np.isfinite(med) & (vol > 1e-12)
        scalar[valid] = np.minimum(1.0, med[valid] / vol[valid])
        self._size_frac = scalar
        print(
            f"📐 Volatility targeting ON: {self.VOL_WINDOW}-bar realized vol vs "
            f"expanding median (mean size multiplier "
            f"{float(np.mean(scalar)):.2f}, no leverage)"
        )

    def stop(self):
        """Request a graceful stop of the optimization"""
        self.stopped = True

    @property
    def annualization_factor(self) -> float:
        """Periods per year for the finest timeframe being simulated"""
        if self.finest_tf == "daily":
            return 252.0
        if self.finest_tf == "hourly":
            return 252.0 * 6.5
        if self.finest_tf == "5min":
            return 252.0 * 6.5 * 12
        return 252.0 * 6.5 * 60  # 1min

    def _segment_score(
        self, eq_curve: np.ndarray, trade_pcts: np.ndarray, start: int, end: int
    ) -> Tuple[float, int]:
        """
        Confidence-scaled objective score of one chronological segment
        [start, end) of the equity curve. The slice begins one bar early so
        the first segment bar's return is included, and is renormalized to
        the 1000.0 start every metric assumes. Trades belong to the segment
        their ENTRY bar falls in.
        """
        lo = max(start - 1, 0)
        eq_slice = eq_curve[lo:end]
        if len(eq_slice) < 2 or eq_slice[0] <= 0:
            return 0.0, 0
        eq_norm = eq_slice * (1000.0 / eq_slice[0])
        seg_metrics = PerformanceMetrics.calculate_metrics(
            eq_norm,
            annualization_factor=self.annualization_factor,
            trade_returns=trade_pcts,
        )
        n_trades = len(trade_pcts)
        _, seg_score = scored(self.objective, seg_metrics, n_trades)
        return float(seg_score), n_trades

    def evaluate_params(self, params: Dict) -> Tuple[float, float, Optional[Dict], int]:
        """
        Simulate one parameter set and score it under the configured
        objective.

        Returns:
            (raw_objective_value, sampler_score, metrics, trades). `raw` is
            the full-sample objective value (reporting continuity).
            `sampler_score` is what TPE maximizes: the full-sample
            confidence-scaled score by default; in opt-in veto mode with a
            split active it is the TRAIN-segment score instead (the
            validation slice is reserved for the veto). When a split is
            active, Train_Score / Val_Score / *_Trades are always recorded
            in the metrics dict regardless of mode.
        """
        sim_stats: Dict = {}
        eq_curve, trade_count, trade_pcts = self.simulate_multi_tf(
            params, return_trade_pcts=True, stats_out=sim_stats
        )
        if eq_curve is None or len(eq_curve) < 50:
            return 0.0, 0.0, None, 0

        metrics = PerformanceMetrics.calculate_metrics(
            eq_curve,
            annualization_factor=self.annualization_factor,
            trade_returns=trade_pcts,
            exposure_frac=sim_stats.get("exposure_frac"),
        )
        if metrics is None:
            return 0.0, 0.0, None, int(trade_count)

        raw, score = scored(self.objective, metrics, trade_count)

        cut = self.selection_cut
        if cut is not None:
            entry_idxs = np.asarray(
                sim_stats.get("trade_entry_idxs", []), dtype=np.int64
            )
            in_train = entry_idxs < cut
            train_score, train_trades = self._segment_score(
                eq_curve, trade_pcts[in_train], 0, cut
            )
            val_score, val_trades = self._segment_score(
                eq_curve, trade_pcts[~in_train], cut, len(eq_curve)
            )
            metrics["Train_Score"] = round(train_score, 6)
            metrics["Val_Score"] = round(val_score, 6)
            metrics["Train_Trades"] = int(train_trades)
            metrics["Val_Trades"] = int(val_trades)
            if self.use_validation_veto:
                score = train_score

        return raw, score, metrics, int(trade_count)

    def compute_signals(self, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (enter_signal, exit_signal) boolean arrays on the finest
        timeframe for a parameter set. Shared by simulate_multi_tf and the
        regime-switching engine so signal semantics can never diverge.

        Indicator semantics live in signals.engine.evaluate_combo (legacy
        RSI params, mixed indicator combos, and cycle-only sets alike);
        this method adds the calendar-anchored time cycle, the warm-up
        mask, and the lag-one coarse-to-fine timeframe mapping.
        """
        n_bars = len(self.np_data[self.finest_tf]["close"])

        # Pre-allocate signal arrays
        enter_signal = np.ones(n_bars, dtype=bool)
        exit_signal = np.zeros(n_bars, dtype=bool)

        # Calculate signals for each timeframe
        for tf in self.timeframes:
            close_tf = self.np_data[tf]["close"]

            # Indicator combo entry/exit conditions (causal)
            entry_ok, exit_ok, warmup = evaluate_combo(close_tf, params, tf)

            # Vectorized cycle (guard against a zero-length period).
            # Anchored to calendar time so the phase is identical across
            # backtests, walk-forward windows, and live trading.
            on = int(params[f"On_{tf}"])
            off = int(params[f"Off_{tf}"])
            start = int(params[f"Start_{tf}"])
            period = max(1, on + off)
            anchor = self.np_data[tf]["anchor"]
            cycle = ((anchor - start) % period) < on

            # Block entries during the indicator warm-up region, where
            # oscillators are computed from partial windows
            warmed_up = np.arange(len(close_tf)) >= warmup

            # Map to finest timeframe
            if tf != self.finest_tf:
                indices = self.tf_indices[tf]
                valid_mask = indices >= 0
                indices_clipped = np.clip(indices, 0, len(entry_ok) - 1)
                entry_mapped = np.zeros(n_bars, dtype=bool)
                exit_mapped = np.zeros(n_bars, dtype=bool)
                cycle_mapped = np.zeros(n_bars, dtype=bool)
                warmed_up_mapped = np.zeros(n_bars, dtype=bool)
                entry_mapped[valid_mask] = entry_ok[indices_clipped[valid_mask]]
                exit_mapped[valid_mask] = exit_ok[indices_clipped[valid_mask]]
                cycle_mapped[valid_mask] = cycle[indices_clipped[valid_mask]]
                warmed_up_mapped[valid_mask] = warmed_up[indices_clipped[valid_mask]]
            else:
                entry_mapped = entry_ok
                exit_mapped = exit_ok
                cycle_mapped = cycle
                warmed_up_mapped = warmed_up

            # Signals
            enter_signal &= entry_mapped & cycle_mapped & warmed_up_mapped
            exit_signal |= exit_mapped | (~cycle_mapped)

        return enter_signal, exit_signal

    def simulate_multi_tf(
        self,
        params: Dict,
        return_trades: bool = False,
        return_trade_pcts: bool = False,
        stats_out: Optional[Dict] = None,
    ):
        """
        Simulate a parameter set on the finest timeframe.

        Args:
            stats_out: Optional caller-owned dict that receives simulation
                statistics ("exposure_frac": fraction of bars marked
                in-market). Caller-owned so concurrent trials (Optuna
                n_jobs threads) can never race on shared state.

        Returns:
            (equity_curve, trade_count) by default;
            (equity_curve, trade_count, trades) with return_trades=True;
            (equity_curve, trade_count, trade_pcts) with
            return_trade_pcts=True, where trade_pcts is a float array of
            per-trade percent changes (cheap - used for trade-level metrics
            on every optimization trial).
        """
        def _empty():
            if return_trades:
                return None, 0, []
            if return_trade_pcts:
                return None, 0, np.array([])
            return None, 0

        if self.stopped:
            return _empty()

        try:
            # Get finest timeframe data
            close_finest = self.np_data[self.finest_tf]["close"]
            open_finest = self.np_data[self.finest_tf]["open"]
            datetime_finest = self.np_data[self.finest_tf]["datetime"]
            n_bars = len(close_finest)

            enter_signal, exit_signal = self.compute_signals(params)

            # Backtest simulation
            equity_curve = np.zeros(n_bars)
            equity = 1000.0
            position = False
            entry_price = 0.0
            entry_idx = 0
            trade_count = 0
            trades = [] if return_trades else None
            trade_pcts: List[float] = []
            trade_entry_idxs: List[int] = []
            bars_in_market = 0
            # Fraction of equity this trade deploys; with volatility
            # targeting it is fixed per trade at the decision bar
            trade_f = self.position_size

            for i in range(n_bars):
                if not position and enter_signal[i]:
                    # Enter at the NEXT bar's open (no lookahead)
                    if i + 1 < n_bars:
                        entry_price = open_finest[i + 1]
                        entry_idx = i + 1
                        trade_f = self.position_size * (
                            self._size_frac[i] if self._size_frac is not None else 1.0
                        )

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

                    # Calculate actual percent change (price move net of costs)
                    pct_change = (exit_price_with_costs / entry_price - 1) * 100
                    trade_pcts.append(pct_change)
                    trade_entry_idxs.append(entry_idx)

                    # Update equity, deploying this trade's fraction of
                    # equity (must match live sizing)
                    equity_before = equity
                    equity *= 1.0 + trade_f * (
                        exit_price_with_costs / entry_price - 1.0
                    )

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
                                "Total_Cost_PCT": (entry_cost_pct + exit_cost_pct) * 100,
                            }
                        )

                    position = False

                # Mark-to-market only once the entry bar has been reached;
                # before that the position is not yet open
                if position and i >= entry_idx:
                    equity_curve[i] = equity * (
                        1.0 + trade_f * (open_finest[i] / entry_price - 1.0)
                    )
                    bars_in_market += 1
                else:
                    equity_curve[i] = equity

            # Close final position
            if position:
                exit_price = open_finest[-1]
                exit_cost_pct = self.transaction_costs.TOTAL_PCT
                exit_price_with_costs = exit_price * (1 - exit_cost_pct)

                pct_change = (exit_price_with_costs / entry_price - 1) * 100
                trade_pcts.append(pct_change)
                trade_entry_idxs.append(entry_idx)
                equity_before = equity
                equity *= 1.0 + trade_f * (
                    exit_price_with_costs / entry_price - 1.0
                )

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

            if stats_out is not None:
                stats_out["exposure_frac"] = (
                    bars_in_market / n_bars if n_bars else 0.0
                )
                stats_out["trade_entry_idxs"] = trade_entry_idxs

            if return_trades:
                return equity_curve, trade_count, trades
            if return_trade_pcts:
                return equity_curve, trade_count, np.asarray(trade_pcts)

            return equity_curve, trade_count

        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback

            traceback.print_exc()
            return _empty()

    # ------------------------------------------------------------------
    # Optimization phases
    # ------------------------------------------------------------------

    def _make_study(self, phase_index: int, trials_per_batch: int) -> optuna.Study:
        """One TPE study per phase (seeded per phase when a seed is set)"""
        return optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=min(10, trials_per_batch),
                multivariate=False,
                warn_independent_sampling=False,
                seed=None if self.seed is None else self.seed + phase_index,
            ),
        )

    def _default_indicator_params(self, tf: str) -> Dict:
        """Neutral indicator parameters for a timeframe that has not been
        optimized yet (cycle-only mode disables indicators entirely)"""
        if self.cycle_only:
            return {f"IND1_{tf}": CYCLE_ONLY}
        return {
            f"MN1_{tf}": (self.mn1_range[0] + self.mn1_range[1]) // 2,
            f"MN2_{tf}": (self.mn2_range[0] + self.mn2_range[1]) // 2,
            f"Entry_{tf}": (self.entry_range[0] + self.entry_range[1]) / 2,
            f"Exit_{tf}": (self.exit_range[0] + self.exit_range[1]) / 2,
        }

    def _record_trial(self, trial: optuna.trial.Trial, params: Dict):
        """Evaluate a trial's params, attach everything the reporting needs
        as user attrs, and return the score to maximize"""
        raw, score, metrics, trade_count = self.evaluate_params(params)

        if metrics is not None:
            self.trial_sharpes.append(float(metrics.get("Sharpe_Ratio", 0.0)))

        trial.set_user_attr("params", params)
        trial.set_user_attr("objective_raw", float(raw))
        trial.set_user_attr("score", float(score))
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("trade_count", trade_count)
        trial.set_user_attr(
            "val_score",
            float(metrics["Val_Score"]) if metrics and "Val_Score" in metrics else None,
        )
        return score

    def _phase_winner(self, study: optuna.Study) -> optuna.trial.FrozenTrial:
        """
        The trial whose suggested params win this phase.

        Default: the plain sampler-score argmax (full-sample selection).

        Opt-in veto mode: the best TRAIN score among trials whose
        validation-slice score is positive. Two variants were tested
        against pre-registered criteria (RESEARCH_JOURNAL.md): the
        validation ARGMAX was falsified on synthetic no-edge data (a max
        over a short slice has higher selection variance than over the
        full sample, H5a); the VETO won significantly on synthetic
        planted-edge data (H6) but failed to replicate on real
        multi-asset folds (H7) - hence opt-in, not default. Falls back to
        the plain argmax when no trial has a positive validation score.
        """
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        train_key = lambda t: t.value if t.value is not None else -np.inf  # noqa: E731
        if self.use_validation_veto and self.selection_cut is not None:
            survivors = [
                t for t in completed if (t.user_attrs.get("val_score") or 0.0) > 0.0
            ]
            if survivors:
                winner = max(survivors, key=train_key)
                print(
                    f"   🏁 winner: train score {winner.value:.4f} among "
                    f"{len(survivors)}/{len(completed)} validation survivors "
                    f"(val score {winner.user_attrs['val_score']:.4f})"
                )
                return winner
            print("   🏁 no validation survivors - winner by train score")
        return max(completed, key=train_key)

    def _save_batch_results(
        self, study: optuna.Study, n_before: int, batch_idx: int, phase_counter: int,
        results_path,
    ):
        """Append this batch's top-5 rows and rewrite the results CSV.
        Rewriting (instead of appending without a header) keeps columns
        aligned even when parameter sets differ across batches."""
        batch_trials = sorted(
            [
                t
                for t in study.trials[n_before:]
                if t.state == optuna.trial.TrialState.COMPLETE
            ],
            key=lambda t: t.value,
            reverse=True,
        )[:5]

        added = 0
        for trial in batch_trials:
            params = trial.user_attrs.get("params", {})
            metrics = trial.user_attrs.get("metrics")
            if not params or metrics is None:
                continue

            row = dict(params)
            row.update(metrics)
            row["Trade_Count"] = trial.user_attrs.get("trade_count", 0)
            row["Objective"] = self.objective
            row["Objective_Score"] = trial.user_attrs.get("objective_raw", 0.0)
            row["Batch"] = batch_idx + 1
            row["Phase"] = phase_counter
            self._result_rows.append(row)
            added += 1

        if added:
            pd.DataFrame(self._result_rows).to_csv(results_path, index=False)
            print(f"   💾 Saved {added} results (CSV now {len(self._result_rows)} rows)")

    def _collect_rivals(self, study: optuna.Study):
        """Keep the top parameter sets of the final phase as CSCV rivals
        for the anti-overfit validation"""
        completed = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("params")
        ]
        completed.sort(key=lambda t: t.value, reverse=True)
        self.rival_params = [t.user_attrs["params"] for t in completed[:12]]

    def run(self):
        """Sequential optimization with parallel batching and incremental CSV saves"""
        try:
            import gc
            import multiprocessing

            # Determine optimal CPU usage
            n_cpus = multiprocessing.cpu_count()
            n_jobs = max(1, n_cpus - 1)  # Leave 1 core for OS

            spec = get_objective(self.objective)

            print(f"\n{'='*60}")
            print(f"STRATEGY OPTIMIZATION (objective: {spec.label})")
            print(f"{'='*60}")
            print(f"Ticker: {self.ticker}")
            print(f"Timeframes: {self.timeframes}")
            print(f"Mode: {'TIME CYCLE ONLY' if self.cycle_only else 'cycle + indicators'}")
            if not self.cycle_only and self.indicator_search:
                print(f"Indicators: {self.allowed_indicators}")
                print(f"Combine two indicators: {self.combine_indicators}")
            print(f"Total trials: {self.n_trials}")
            print(f"Batch size: {self.batch_size}")
            print(f"CPU cores: {n_cpus} (using {n_jobs} for optimization)")
            print(f"{'='*60}\n")

            on_range, off_range, start_range = self.time_cycle_ranges

            phases_per_tf = 1 if self.cycle_only else 2
            total_phases = len(self.timeframes) * phases_per_tf
            trials_per_phase = max(50, self.n_trials // total_phases)

            # Calculate batches per phase
            batches_per_phase = max(1, trials_per_phase // self.batch_size)
            trials_per_batch = trials_per_phase // batches_per_phase

            print(f"📦 Batch Configuration:")
            print(f"   Trials per phase: {trials_per_phase}")
            print(f"   Batches per phase: {batches_per_phase}")
            print(f"   Trials per batch: {trials_per_batch}")
            print()

            phase_counter = 0

            results_path = Paths.get_results_path(
                self.ticker, suffix=f"_{self.objective}"
            )

            final_study = None

            # Optimize each timeframe sequentially
            for tf_idx, tf in enumerate(self.timeframes):
                if self.stopped:
                    break

                # ===================================================================
                # PHASE: Optimize Time Cycle (with batching)
                # ===================================================================
                phase_counter += 1
                phase_msg = f"Phase {phase_counter}/{total_phases}: {tf.upper()} Time Cycle..."
                self.phase_update.emit(phase_msg)
                print(f"\n{'='*60}")
                print(f"PHASE {phase_counter}: {tf.upper()} TIME CYCLE (BATCHED)")
                print(f"{'='*60}")

                cycle_study = self._make_study(phase_counter, trials_per_batch)

                # Pre-build base parameters once
                base_params_cycle = {}
                for prev_tf in self.timeframes[:tf_idx]:
                    base_params_cycle.update(self.best_params_per_tf[prev_tf])

                # Current timeframe indicator defaults (neutral placeholder;
                # the indicator phase optimizes them afterwards)
                base_params_cycle.update(self._default_indicator_params(tf))

                # Future timeframes defaults
                for future_tf in self.timeframes[tf_idx + 1 :]:
                    base_params_cycle.update(self._default_indicator_params(future_tf))
                    base_params_cycle[f"On_{future_tf}"] = on_range[0]
                    base_params_cycle[f"Off_{future_tf}"] = off_range[0]
                    base_params_cycle[f"Start_{future_tf}"] = 0

                # Run cycle optimization in batches
                for batch_idx in range(batches_per_phase):
                    if self.stopped:
                        break

                    print(f"\n📦 Batch {batch_idx + 1}/{batches_per_phase}")

                    trial_count = [0]
                    n_trials_before_batch = len(cycle_study.trials)

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

                        score = self._record_trial(trial, params)

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
                        n_jobs=n_jobs,
                        catch=(Exception,),
                        show_progress_bar=False,
                    )

                    # In cycle-only mode this is the phase that produces the
                    # results people care about - persist it
                    if self.cycle_only:
                        self._save_batch_results(
                            cycle_study, n_trials_before_batch, batch_idx,
                            phase_counter, results_path,
                        )

                    print(f"   ✓ Batch {batch_idx + 1} complete")

                    # Memory cleanup after batch
                    gc.collect()

                if self.stopped:
                    break

                if not any(t.state == optuna.trial.TrialState.COMPLETE for t in cycle_study.trials):
                    raise ValueError(
                        f"No completed trials in {tf} cycle phase - "
                        "check data quality and parameter ranges"
                    )

                cycle_winner = self._phase_winner(cycle_study)
                won = int(cycle_winner.params[f"On_{tf}"])
                woff = int(cycle_winner.params[f"Off_{tf}"])
                # Canonicalize the phase: only Start % (On + Off) matters
                # to the cycle, so store one encoding per strategy (keeps
                # CSVs comparable and CSCV rivals deduplicable)
                best_cycle = {
                    f"On_{tf}": won,
                    f"Off_{tf}": woff,
                    f"Start_{tf}": int(cycle_winner.params[f"Start_{tf}"]) % max(1, won + woff),
                }

                if tf not in self.best_params_per_tf:
                    self.best_params_per_tf[tf] = {}
                self.best_params_per_tf[tf].update(best_cycle)

                if self.cycle_only:
                    # Strategy for this timeframe is the cycle alone
                    self.best_params_per_tf[tf][f"IND1_{tf}"] = CYCLE_ONLY
                    final_study = cycle_study
                    print(f"✓ Phase {phase_counter} Complete - Cycle params optimized")
                    continue

                final_study = cycle_study
                print(f"✓ Phase {phase_counter} Complete - Cycle params optimized")

                # ===================================================================
                # PHASE: Optimize Indicators (with batching + CSV saves)
                # ===================================================================
                phase_counter += 1
                self.phase_update.emit(
                    f"Phase {phase_counter}/{total_phases}: {tf.upper()} Indicators..."
                )
                print(f"\nPHASE {phase_counter}: {tf.upper()} INDICATORS (BATCHED)")

                ind_study = self._make_study(phase_counter, trials_per_batch)

                # Pre-build base parameters once
                base_params_ind = {}
                for prev_tf in self.timeframes[:tf_idx]:
                    base_params_ind.update(self.best_params_per_tf[prev_tf])

                base_params_ind.update(best_cycle)

                # Future timeframes defaults
                for future_tf in self.timeframes[tf_idx + 1 :]:
                    base_params_ind.update(self._default_indicator_params(future_tf))
                    base_params_ind[f"On_{future_tf}"] = on_range[0]
                    base_params_ind[f"Off_{future_tf}"] = off_range[0]
                    base_params_ind[f"Start_{future_tf}"] = 0

                # Run indicator optimization in batches with CSV saves
                for batch_idx in range(batches_per_phase):
                    if self.stopped:
                        break

                    print(f"\n📦 Batch {batch_idx + 1}/{batches_per_phase}")

                    trial_count = [0]

                    def objective_ind(trial):
                        if self.stopped:
                            raise optuna.exceptions.OptunaError("Stopped by user")

                        trial_count[0] += 1

                        # Copy base params (fast shallow copy)
                        params = base_params_ind.copy()

                        # Only update current timeframe indicator params
                        # (trial-specific)
                        params[f"MN1_{tf}"] = trial.suggest_int(f"MN1_{tf}", *self.mn1_range)
                        params[f"MN2_{tf}"] = trial.suggest_int(f"MN2_{tf}", *self.mn2_range)
                        params[f"Entry_{tf}"] = trial.suggest_float(
                            f"Entry_{tf}", *self.entry_range, step=0.5
                        )
                        params[f"Exit_{tf}"] = trial.suggest_float(
                            f"Exit_{tf}", *self.exit_range, step=0.5
                        )

                        if self.indicator_search:
                            # Mix-and-match: which indicator drives leg 1,
                            # optionally AND-ed with a second indicator
                            params[f"IND1_{tf}"] = trial.suggest_categorical(
                                f"IND1_{tf}", list(self.allowed_indicators)
                            )
                            params[f"IND1_Inv_{tf}"] = trial.suggest_int(
                                f"IND1_Inv_{tf}", 0, 1
                            )
                            if self.combine_indicators:
                                ind2 = trial.suggest_categorical(
                                    f"IND2_{tf}", ["none"] + list(self.allowed_indicators)
                                )
                            else:
                                ind2 = "none"
                            params[f"IND2_{tf}"] = ind2
                            if ind2 != "none":
                                params[f"IND2_P1_{tf}"] = trial.suggest_int(
                                    f"IND2_P1_{tf}", *self.mn1_range
                                )
                                params[f"IND2_P2_{tf}"] = trial.suggest_int(
                                    f"IND2_P2_{tf}", *self.mn2_range
                                )
                                params[f"IND2_Entry_{tf}"] = trial.suggest_float(
                                    f"IND2_Entry_{tf}", *self.entry_range, step=0.5
                                )
                                params[f"IND2_Exit_{tf}"] = trial.suggest_float(
                                    f"IND2_Exit_{tf}", *self.exit_range, step=0.5
                                )
                                params[f"IND2_Inv_{tf}"] = trial.suggest_int(
                                    f"IND2_Inv_{tf}", 0, 1
                                )

                        score = self._record_trial(trial, params)

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

                    # Snapshot trial count so only THIS batch's trials get
                    # saved below (the study accumulates across batches)
                    n_trials_before_batch = len(ind_study.trials)

                    # Run batch with parallel processing
                    ind_study.optimize(
                        objective_ind,
                        n_trials=trials_per_batch,
                        n_jobs=n_jobs,
                        catch=(Exception,),
                        show_progress_bar=False,
                    )

                    self._save_batch_results(
                        ind_study, n_trials_before_batch, batch_idx,
                        phase_counter, results_path,
                    )

                    print(f"   ✓ Batch {batch_idx + 1} complete")

                    # Memory cleanup after batch
                    gc.collect()

                if self.stopped:
                    break

                if not any(t.state == optuna.trial.TrialState.COMPLETE for t in ind_study.trials):
                    raise ValueError(
                        f"No completed trials in {tf} indicator phase - "
                        "check data quality and parameter ranges"
                    )

                # Take every suggested key of the winning trial (all keys
                # are namespaced with _{tf}); with indicator_search this
                # includes the IND1/IND2 choices and their sub-params
                best_ind = dict(self._phase_winner(ind_study).params)
                if self.indicator_search and not self.combine_indicators:
                    # IND2 was forced off, not suggested - record it so the
                    # stored parameter set is self-describing
                    best_ind[f"IND2_{tf}"] = "none"

                self.best_params_per_tf[tf].update(best_ind)
                final_study = ind_study

                print(f"✓ Phase {phase_counter} Complete - Indicators optimized")

            # ===================================================================
            # Compile final best result
            # ===================================================================
            if self.stopped:
                print("\nOptimization stopped by user")
                self.finished.emit(pd.DataFrame())
                return

            if final_study is not None:
                self._collect_rivals(final_study)

            print(f"\n{'='*60}")
            print(f"Compiling final best result...")

            base_params = {}
            for tf in self.timeframes:
                if tf in self.best_params_per_tf:
                    base_params.update(self.best_params_per_tf[tf])

            # Get equity curve AND trade log for GUI display
            final_stats: Dict = {}
            base_eq_curve, base_trade_count, base_trades = self.simulate_multi_tf(
                base_params, return_trades=True, stats_out=final_stats
            )

            if base_eq_curve is None:
                raise ValueError("Final simulation failed")

            trade_pcts = np.asarray(
                [t["Percent_Change"] for t in base_trades], dtype=np.float64
            )
            base_metrics = PerformanceMetrics.calculate_metrics(
                base_eq_curve,
                annualization_factor=self.annualization_factor,
                trade_returns=trade_pcts,
                exposure_frac=final_stats.get("exposure_frac"),
            )

            if base_metrics is None:
                raise ValueError("Failed to calculate performance metrics")

            objective_raw, _ = scored(self.objective, base_metrics, base_trade_count)

            # Attach the compiled set's train/validation segment scores so
            # the result row shows how it fared on the untouched slice
            if self.selection_cut is not None:
                _, _, split_metrics, _ = self.evaluate_params(base_params)
                if split_metrics:
                    for key in ("Train_Score", "Val_Score", "Train_Trades", "Val_Trades"):
                        if key in split_metrics:
                            base_metrics[key] = split_metrics[key]

            base_metrics["Trade_Count"] = base_trade_count
            base_metrics["Objective"] = self.objective
            base_metrics["Objective_Score"] = objective_raw
            base_metrics["Batch"] = "FINAL"
            base_metrics["Phase"] = "FINAL"

            # Buy & hold benchmark over the same bars, for honest context
            bh_metrics = PerformanceMetrics.calculate_buyhold_metrics(
                self.np_data[self.finest_tf]["close"],
                annualization_factor=self.annualization_factor,
            )
            if bh_metrics:
                base_metrics["BuyHold_Return_%"] = bh_metrics["Percent_Gain_%"]
                base_metrics["BuyHold_Sharpe"] = bh_metrics["Sharpe_Ratio"]
                base_metrics["BuyHold_MaxDD_%"] = bh_metrics["Max_Drawdown_%"]

            print(
                f"✓ {spec.label}: {objective_raw:{spec.fmt}} | "
                f"Sharpe: {base_metrics['Sharpe_Ratio']:.2f} | "
                f"Sortino: {base_metrics['Sortino_Ratio']:.2f}"
            )
            if "Val_Score" in base_metrics:
                print(
                    f"   Validation slice: score {base_metrics['Val_Score']:.4f} "
                    f"({base_metrics.get('Val_Trades', 0)} trades) | "
                    f"train score {base_metrics.get('Train_Score', 0.0):.4f}"
                )

            # Append final result and rewrite CSV
            final_result = {**base_params, **base_metrics}
            self.all_results.append(final_result)
            self.new_best.emit(final_result)

            self._result_rows.append(final_result)
            pd.DataFrame(self._result_rows).to_csv(results_path, index=False)
            print(f"\n✅ Complete results saved to: {results_path}")

            # Build results DataFrame for GUI with equity curve and trade log
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
            print(f"Objective ({spec.label}): {objective_raw:{spec.fmt}}")

            print(f"\n📊 Metrics:")
            print(f"  Return: {final_result['Percent_Gain_%']:.2f}%")
            if "BuyHold_Return_%" in final_result:
                print(f"  Buy & Hold: {final_result['BuyHold_Return_%']:.2f}%")
            print(f"  Sharpe: {final_result['Sharpe_Ratio']:.2f}")
            print(f"  Sortino: {final_result['Sortino_Ratio']:.2f}")
            print(f"  Calmar: {final_result['Calmar_Ratio']:.2f}")
            print(f"  Max Drawdown: {final_result['Max_Drawdown_%']:.2f}%")
            print(f"  Profit Factor: {final_result['Profit_Factor']:.2f}")
            if "Win_Rate_%" in final_result:
                print(f"  Win Rate: {final_result['Win_Rate_%']:.1f}%")
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

                    if params.get(f"IND1_{tf}") == CYCLE_ONLY:
                        print("    Indicators: none (time cycle only)")
                    elif f"MN1_{tf}" in params:
                        ind1 = params.get(f"IND1_{tf}", "rsi")
                        print(
                            f"    {str(ind1).upper()}: P1={params[f'MN1_{tf}']}, "
                            f"P2={params[f'MN2_{tf}']}"
                        )
                        print(
                            f"    Thresholds: ENTRY<{params[f'Entry_{tf}']:.1f}, EXIT>{params[f'Exit_{tf}']:.1f}"
                        )
                        ind2 = params.get(f"IND2_{tf}", "none")
                        if ind2 and ind2 != "none":
                            print(
                                f"    + {str(ind2).upper()}: "
                                f"P1={params.get(f'IND2_P1_{tf}')}, "
                                f"P2={params.get(f'IND2_P2_{tf}')}"
                            )

            print(f"{'='*60}\n")

            self.finished.emit(df_results)

        except Exception as e:
            if self.stopped:
                # Let the GUI reset its buttons after a user-initiated stop
                self.finished.emit(pd.DataFrame())
            else:
                error_msg = f"Optimization error: {e}"
                print(f"\n✗ {error_msg}")
                self.error.emit(error_msg)
                import traceback

                traceback.print_exc()
