"""
Live Re-Optimization with Gated Book Admission

Periodically re-optimizes on fresh data so the system keeps searching for
the current edge - but a fresh fit NEVER trades directly. Every candidate
must survive, in order:

  1. Train/holdout split: optimization only ever sees the older part of
     the data; the newest slice stays untouched for true out-of-sample
     evaluation.
  2. Anti-overfit gates (optimization.validation): Deflated Sharpe Ratio
     (prices in how many trials were burned), CSCV probability of backtest
     overfitting, minimum trade count, and non-negative held-out Sharpe.
  3. StrategyBook admission: limited slots; a gated candidate still has to
     beat the weakest stored strategy's score to displace it.

Even after admission, the regime-switching engine only lets the newcomer
place real trades once its SHADOW performance earns the selection in the
live regime. Three independent layers between "the optimizer liked it"
and "it trades your money".

run_reoptimization_cycle() is pure logic (testable, no Qt/network);
LiveReoptimizer wraps it in a QThread with a data refresh per cycle.
"""

import datetime as _dt
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from optimization.psr_composite import PSRCalculator
from optimization.validation import ValidationReport, validate_candidate
from trading.regime_switcher import StoredStrategy, StrategyBook


@dataclass
class ReoptimizationResult:
    """Outcome of one re-optimization cycle"""

    admitted: bool
    message: str
    report: Optional[ValidationReport] = None
    strategy: Optional[StoredStrategy] = None


def split_train_holdout(
    df_dict: Dict[str, pd.DataFrame], holdout_fraction: float = 0.25
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Chronological split: train = oldest (1 - holdout_fraction) of the data,
    holdout = the newest slice, cut at one timestamp across all timeframes
    so no holdout information can leak into training.
    """
    tf_order = {"1min": 0, "5min": 1, "hourly": 2, "daily": 3}
    finest = sorted(df_dict.keys(), key=lambda x: tf_order.get(x, 99))[0]
    dt_finest = df_dict[finest]["Datetime"]
    cut_idx = int(len(dt_finest) * (1.0 - holdout_fraction))
    cut_idx = min(max(cut_idx, 1), len(dt_finest) - 1)
    cutoff = dt_finest.iloc[cut_idx]

    train, hold = {}, {}
    for tf, df in df_dict.items():
        train[tf] = df[df["Datetime"] < cutoff].reset_index(drop=True)
        hold[tf] = df[df["Datetime"] >= cutoff].reset_index(drop=True)
    return train, hold


def run_reoptimization_cycle(
    df_dict: Dict[str, pd.DataFrame],
    book: StrategyBook,
    optimizer_kwargs: Dict,
    ticker: str = "",
    n_trials: int = 300,
    holdout_fraction: float = 0.25,
    dsr_threshold: float = 0.90,
    pbo_threshold: float = 0.50,
    min_trades: int = 20,
    min_holdout_bars: int = 150,
    min_oos_sharpe: float = 0.0,
    min_oos_trades: int = 3,
) -> ReoptimizationResult:
    """
    One full re-optimization cycle: optimize on train, gate, admit.

    Args:
        df_dict: Fresh market data per timeframe
        book: The strategy book candidates are admitted into
        optimizer_kwargs: Ranges/costs/etc. for MultiTimeframeOptimizer
            (everything except df_dict, n_trials, ticker)
        n_trials: Trial budget for the re-optimization (kept modest -
            the DSR gate punishes big budgets on noise)
        min_oos_trades: Minimum trades the candidate must ENTER inside the
            holdout window (when one is evaluated). A candidate that never
            trades out-of-sample has produced no OOS evidence and must not
            be admitted on a vacuous Sharpe of 0.0.
    """
    from optimization.optimizer import MultiTimeframeOptimizer

    train_dict, hold_dict = split_train_holdout(df_dict, holdout_fraction)

    tf_order = {"1min": 0, "5min": 1, "hourly": 2, "daily": 3}
    finest = sorted(df_dict.keys(), key=lambda x: tf_order.get(x, 99))[0]
    if len(train_dict[finest]) < 300:
        return ReoptimizationResult(
            False, f"not enough training data ({len(train_dict[finest])} bars)"
        )

    # 1) Optimize on the TRAIN slice only
    optimizer = MultiTimeframeOptimizer(
        df_dict=train_dict, n_trials=n_trials, ticker=ticker or "REOPT",
        **optimizer_kwargs,
    )
    optimizer.run()  # synchronous (same pattern as walk-forward)

    if not optimizer.all_results:
        return ReoptimizationResult(False, "optimization produced no result")

    candidate = {
        k: v
        for k, v in optimizer.all_results[0].items()
        if any(
            k.startswith(p)
            for p in ("MN1_", "MN2_", "Entry_", "Exit_", "On_", "Off_", "Start_", "IND")
        )
    }

    # 2) True out-of-sample evaluation on the untouched holdout.
    # Simulate over the FULL data (train supplies indicator warm-up
    # context, exactly as live trading would have it) and score only the
    # holdout bars - simulating the holdout in isolation restarts the
    # warm-up mask, which for slow indicators can swallow most of the
    # holdout and bias OOS Sharpe toward 0 with zero trades.
    oos_sharpe = None
    oos_trades: Optional[int] = None
    if len(hold_dict[finest]) >= min_holdout_bars:
        eval_optimizer = MultiTimeframeOptimizer(
            df_dict=df_dict, n_trials=1, ticker="HOLDOUT", **optimizer_kwargs
        )
        full_curve, _, full_trades = eval_optimizer.simulate_multi_tf(
            candidate, return_trades=True
        )
        if full_curve is not None and len(full_curve) > 2:
            cutoff = pd.Timestamp(hold_dict[finest]["Datetime"].iloc[0])
            finest_dt = df_dict[finest]["Datetime"]
            cut_idx = int(finest_dt.searchsorted(cutoff, side="left"))
            if 0 < cut_idx < len(full_curve):
                # Slice one bar early so the first holdout bar's return
                # is included in the OOS Sharpe
                oos_curve = full_curve[cut_idx - 1 :]
                oos_sharpe = float(
                    PSRCalculator.calculate_sharpe_from_equity(
                        oos_curve,
                        annualization_factor=eval_optimizer.annualization_factor,
                    )
                )
                oos_trades = sum(
                    1
                    for t in full_trades
                    if pd.Timestamp(t["Entry_Date"]) >= cutoff
                )

    # 3) Anti-overfit gates
    report = validate_candidate(
        simulate_fn=optimizer.simulate_multi_tf,
        params=candidate,
        rival_params=optimizer.rival_params,
        trial_sharpes=optimizer.trial_sharpes,
        annualization_factor=optimizer.annualization_factor,
        dsr_threshold=dsr_threshold,
        pbo_threshold=pbo_threshold,
        min_trades=min_trades,
        oos_sharpe=oos_sharpe,
        min_oos_sharpe=min_oos_sharpe,
        oos_trades=oos_trades,
        min_oos_trades=min_oos_trades,
    )

    if not report.passed:
        return ReoptimizationResult(
            False, "candidate failed overfit gates:\n" + report.summary(), report
        )

    # 4) Book admission (score = DSR so stored strategies compete on
    # deflated, luck-adjusted evidence rather than raw fit quality)
    name = f"auto_{ticker}_{_dt.datetime.now().strftime('%m%d-%H%M%S')}"
    strategy = StoredStrategy(
        name=name,
        params=candidate,
        base_score=float(report.dsr),
        ticker=ticker,
        timeframes=sorted(df_dict.keys()),
        metrics={
            "DSR": report.dsr,
            "PBO": report.pbo if report.pbo is not None else -1.0,
            "OOS_Sharpe": oos_sharpe if oos_sharpe is not None else 0.0,
            "OOS_Trades": oos_trades if oos_trades is not None else -1,
            "Trades": report.trade_count,
        },
    )
    added, replaced = book.add(strategy)
    if not added:
        return ReoptimizationResult(
            False,
            f"candidate passed gates (DSR {report.dsr:.3f}) but did not beat "
            "the weakest stored strategy",
            report,
            strategy,
        )

    msg = f"admitted '{name}' (DSR {report.dsr:.3f}"
    if oos_sharpe is not None:
        msg += f", OOS Sharpe {oos_sharpe:.2f}"
    msg += ")"
    if replaced and replaced != name:
        msg += f", replacing '{replaced}'"
    return ReoptimizationResult(True, msg, report, strategy)


class LiveReoptimizer(QThread):
    """Background re-optimization loop feeding the strategy book"""

    status_update = pyqtSignal(str)
    strategy_admitted = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        ticker: str,
        book_path,
        optimizer_kwargs: Dict,
        timeframes: Optional[list] = None,
        interval_minutes: int = 240,
        n_trials: int = 300,
    ):
        super().__init__()
        self.ticker = ticker
        self.book_path = book_path
        self.optimizer_kwargs = optimizer_kwargs
        self.timeframes = timeframes
        self.interval_minutes = int(interval_minutes)
        self.n_trials = int(n_trials)
        self.running = False

    def run(self):
        from data import DataLoader

        self.running = True
        self.status_update.emit(
            f"🔬 Live re-optimizer started (every {self.interval_minutes} min, "
            f"{self.n_trials} trials/cycle, gated admission)"
        )

        while self.running:
            try:
                self.status_update.emit(f"🔬 Re-optimizing {self.ticker}...")
                df_dict, err = DataLoader.load_yfinance_data(self.ticker)
                if err or not df_dict:
                    self.error.emit(f"Reopt data load failed: {err}")
                else:
                    if self.timeframes:
                        df_dict = {
                            tf: df for tf, df in df_dict.items() if tf in self.timeframes
                        }
                    book = StrategyBook(self.book_path)
                    result = run_reoptimization_cycle(
                        df_dict,
                        book,
                        self.optimizer_kwargs,
                        ticker=self.ticker,
                        n_trials=self.n_trials,
                    )
                    if result.admitted:
                        self.status_update.emit(f"✅ Reopt: {result.message}")
                        self.strategy_admitted.emit(
                            {"name": result.strategy.name, "message": result.message}
                        )
                    else:
                        self.status_update.emit(f"🚫 Reopt: {result.message}")
            except Exception as e:
                self.error.emit(f"Reopt cycle error: {e}")

            self._sleep(self.interval_minutes * 60)

        self.status_update.emit("Re-optimizer stopped")

    def _sleep(self, seconds: float):
        deadline = time.time() + seconds
        while self.running and time.time() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.time())))

    def stop(self):
        self.running = False
