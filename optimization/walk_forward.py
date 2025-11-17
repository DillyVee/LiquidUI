"""
Walk-Forward Analysis Module - FIXED VERSION

Key fixes:
1. Better data validation and error handling
2. Proper datetime handling for all timeframes
3. Smart window sizing based on available data
4. Progress feedback during long optimizations
5. Graceful degradation for edge cases
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class WalkForwardResults:
    """Results from walk-forward analysis"""

    in_sample_returns: List[float]
    out_of_sample_returns: List[float]
    in_sample_equity_curves: List[np.ndarray]
    out_of_sample_equity_curves: List[np.ndarray]
    window_dates: List[Tuple[str, str, str, str]]
    best_params_per_window: List[Dict]

    avg_in_sample_return: float
    avg_out_of_sample_return: float
    efficiency_ratio: float
    consistency: float
    return_degradation: float
    is_overfit: bool


class WalkForwardAnalyzer:
    """Walk-Forward Analysis for Trading Strategies"""

    @staticmethod
    def run_walk_forward(
        optimizer_class,
        df_dict: Dict[str, pd.DataFrame],
        train_days: int = 180,
        test_days: int = 30,
        min_trades: int = 5,
        **optimizer_kwargs,
    ) -> WalkForwardResults:
        """
        Run walk-forward analysis with robust error handling

        Args:
            optimizer_class: The MultiTimeframeOptimizer class
            df_dict: Dictionary of dataframes by timeframe
            train_days: Days for training
            test_days: Days for testing
            min_trades: Minimum trades per window
            **optimizer_kwargs: Additional optimizer arguments
        """
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD ANALYSIS - FIXED VERSION")
        print(f"{'='*70}")

        # Step 1: Validate and prepare data
        df_dict_prepared = WalkForwardAnalyzer._prepare_data(df_dict)

        # Step 2: Determine finest timeframe and validate data range
        timeframes = optimizer_kwargs.get("timeframes", list(df_dict_prepared.keys()))
        finest_tf = WalkForwardAnalyzer._get_finest_timeframe(timeframes)

        df_finest = df_dict_prepared[finest_tf]
        total_days = (df_finest["Datetime"].max() - df_finest["Datetime"].min()).days

        print(f"\n📊 Data Summary:")
        print(f"   Finest timeframe: {finest_tf}")
        print(f"   Total days: {total_days}")
        print(
            f"   Date range: {df_finest['Datetime'].min().date()} to {df_finest['Datetime'].max().date()}"
        )

        # Step 3: Validate window settings
        window_size = train_days + test_days
        if total_days < window_size:
            raise ValueError(
                f"Insufficient data!\n"
                f"  Available: {total_days} days\n"
                f"  Required: {window_size} days (train={train_days} + test={test_days})\n"
                f"  Suggestion: train={total_days//3}, test={total_days//6}"
            )

        # Calculate number of windows
        n_windows = max(1, (total_days - train_days) // test_days)
        print(f"   Windows to process: {n_windows}")
        print(f"   Window size: {train_days} train + {test_days} test")

        if n_windows < 2:
            print(f"\n⚠️  WARNING: Only {n_windows} window(s) possible")
            print(f"   Walk-forward works best with 3+ windows")

        print(f"{'='*70}\n")

        # Step 4: Run walk-forward windows
        results_collector = {
            "in_sample_returns": [],
            "out_of_sample_returns": [],
            "in_sample_curves": [],
            "out_of_sample_curves": [],
            "window_dates": [],
            "best_params": [],
        }

        for window_idx in range(n_windows):
            print(f"\n{'─'*70}")
            print(f"WINDOW {window_idx + 1}/{n_windows}")
            print(f"{'─'*70}")

            try:
                # Define date ranges
                start_offset = window_idx * test_days
                train_start = df_finest["Datetime"].min() + pd.Timedelta(
                    days=start_offset
                )
                train_end = train_start + pd.Timedelta(days=train_days)
                test_start = train_end
                test_end = test_start + pd.Timedelta(days=test_days)

                # Check bounds
                if test_end > df_finest["Datetime"].max():
                    print(f"⚠️  Window exceeds data, stopping at window {window_idx}")
                    break

                print(f"Training:  {train_start.date()} to {train_end.date()}")
                print(f"Testing:   {test_start.date()} to {test_end.date()}")

                # Split data
                train_dict, test_dict = WalkForwardAnalyzer._split_data(
                    df_dict_prepared,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                    timeframes,
                )

                # Validate split
                if not WalkForwardAnalyzer._validate_split(
                    train_dict, test_dict, finest_tf
                ):
                    print(f"⚠️  Insufficient data in window {window_idx + 1}, skipping")
                    continue

                # Run optimization on training data
                print(f"\n📊 Optimizing on training data...")
                best_params, is_equity, is_trades = (
                    WalkForwardAnalyzer._optimize_window(
                        optimizer_class, train_dict, window_idx + 1, **optimizer_kwargs
                    )
                )

                if best_params is None or is_trades < min_trades:
                    print(f"⚠️  Insufficient trades in-sample ({is_trades}), skipping")
                    continue

                is_return = (is_equity[-1] / 1000.0 - 1) * 100
                print(f"✓ In-sample: {is_return:+.2f}% ({is_trades} trades)")

                # Test on out-of-sample data
                print(f"\n📈 Testing on out-of-sample data...")
                oos_equity, oos_trades = WalkForwardAnalyzer._test_window(
                    optimizer_class, test_dict, best_params, **optimizer_kwargs
                )

                if oos_equity is None or oos_trades < 1:
                    print(f"⚠️  No trades out-of-sample, skipping")
                    continue

                oos_return = (oos_equity[-1] / 1000.0 - 1) * 100
                print(f"✓ Out-of-sample: {oos_return:+.2f}% ({oos_trades} trades)")

                # Calculate degradation
                degradation = (
                    ((is_return - oos_return) / abs(is_return) * 100)
                    if is_return != 0
                    else 0
                )
                print(f"  Degradation: {degradation:.1f}%")

                if degradation > 50:
                    print(f"  ⚠️  HIGH DEGRADATION - potential overfitting!")
                elif degradation < 0:
                    print(f"  ✨ OOS better than IS - good sign!")

                # Store results
                results_collector["in_sample_returns"].append(is_return)
                results_collector["out_of_sample_returns"].append(oos_return)
                results_collector["in_sample_curves"].append(is_equity)
                results_collector["out_of_sample_curves"].append(oos_equity)
                results_collector["window_dates"].append(
                    (
                        train_start.strftime("%Y-%m-%d"),
                        train_end.strftime("%Y-%m-%d"),
                        test_start.strftime("%Y-%m-%d"),
                        test_end.strftime("%Y-%m-%d"),
                    )
                )
                results_collector["best_params"].append(best_params)

            except Exception as e:
                print(f"✗ Error in window {window_idx + 1}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Step 5: Calculate summary statistics
        if len(results_collector["out_of_sample_returns"]) == 0:
            raise ValueError("No valid walk-forward windows completed")

        return WalkForwardAnalyzer._calculate_summary(results_collector)

    @staticmethod
    def _prepare_data(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Ensure all dataframes have proper Datetime column"""
        print(f"\n🔧 Preparing data...")
        prepared = {}

        for tf, df in df_dict.items():
            df_copy = df.copy()

            # Ensure Datetime column exists
            if "Datetime" not in df_copy.columns:
                if "Date" in df_copy.columns:
                    df_copy = df_copy.rename(columns={"Date": "Datetime"})
                else:
                    df_copy["Datetime"] = pd.to_datetime(df_copy.index)

            # Ensure Datetime is datetime64
            if df_copy["Datetime"].dtype != "datetime64[ns]":
                df_copy["Datetime"] = pd.to_datetime(df_copy["Datetime"])

            # Remove timezone if present
            if (
                hasattr(df_copy["Datetime"].dtype, "tz")
                and df_copy["Datetime"].dtype.tz is not None
            ):
                df_copy["Datetime"] = df_copy["Datetime"].dt.tz_localize(None)

            prepared[tf] = df_copy

            dt = df_copy["Datetime"]
            days = (dt.max() - dt.min()).days
            print(f"   {tf}: {len(df_copy)} bars, {days} days")

        return prepared

    @staticmethod
    def _get_finest_timeframe(timeframes: List[str]) -> str:
        """Determine finest timeframe from list"""
        tf_order = {"1min": 0, "5min": 1, "hourly": 2, "daily": 3}
        return sorted(timeframes, key=lambda x: tf_order.get(x, 99))[0]

    @staticmethod
    def _split_data(
        df_dict: Dict[str, pd.DataFrame],
        train_start,
        train_end,
        test_start,
        test_end,
        timeframes: List[str],
    ) -> Tuple[Dict, Dict]:
        """Split data into train and test sets"""
        train_dict = {}
        test_dict = {}

        for tf in timeframes:
            if tf not in df_dict:
                continue

            df_tf = df_dict[tf].copy()

            # Training data
            train_mask = (df_tf["Datetime"] >= train_start) & (
                df_tf["Datetime"] < train_end
            )
            train_dict[tf] = df_tf[train_mask].copy().reset_index(drop=True)

            # Testing data
            test_mask = (df_tf["Datetime"] >= test_start) & (
                df_tf["Datetime"] < test_end
            )
            test_dict[tf] = df_tf[test_mask].copy().reset_index(drop=True)

            print(f"  {tf}: Train={len(train_dict[tf])}, Test={len(test_dict[tf])}")

        return train_dict, test_dict

    @staticmethod
    def _validate_split(train_dict: Dict, test_dict: Dict, finest_tf: str) -> bool:
        """Validate that split has enough data"""
        if finest_tf not in train_dict or finest_tf not in test_dict:
            return False

        train_bars = len(train_dict[finest_tf])
        test_bars = len(test_dict[finest_tf])

        return train_bars >= 100 and test_bars >= 10

    @staticmethod
    def _optimize_window(
        optimizer_class, train_dict: Dict, window_num: int, **optimizer_kwargs
    ):
        """Run optimization on training window"""
        try:
            optimizer = optimizer_class(df_dict=train_dict, **optimizer_kwargs)

            optimizer.stopped = False
            optimizer.run()
            optimizer.wait()

            if not optimizer.all_results or len(optimizer.all_results) == 0:
                return None, None, 0

            best_params = optimizer.all_results[0]
            is_equity, is_trades = optimizer.simulate_multi_tf(best_params)

            return best_params, is_equity, is_trades

        except Exception as e:
            print(f"  Optimization error: {e}")
            return None, None, 0

    @staticmethod
    def _test_window(
        optimizer_class, test_dict: Dict, best_params: Dict, **optimizer_kwargs
    ):
        """Test on out-of-sample window"""
        try:
            # Remove n_trials from kwargs for testing
            test_kwargs = {k: v for k, v in optimizer_kwargs.items() if k != "n_trials"}

            test_optimizer = optimizer_class(
                df_dict=test_dict, n_trials=1, **test_kwargs
            )

            oos_equity, oos_trades = test_optimizer.simulate_multi_tf(best_params)
            return oos_equity, oos_trades

        except Exception as e:
            print(f"  Testing error: {e}")
            return None, 0

    @staticmethod
    def _calculate_summary(results: Dict) -> WalkForwardResults:
        """Calculate summary statistics from results"""
        is_returns = results["in_sample_returns"]
        oos_returns = results["out_of_sample_returns"]

        avg_is = np.mean(is_returns)
        avg_oos = np.mean(oos_returns)

        efficiency = (avg_oos / avg_is) if avg_is != 0 else 0

        oos_profitable = sum(1 for r in oos_returns if r > 0)
        consistency = oos_profitable / len(oos_returns)

        degradation = avg_is - avg_oos
        degradation_pct = (degradation / abs(avg_is) * 100) if avg_is != 0 else 0

        is_overfit = (efficiency < 0.3) or (consistency < 0.4) or (degradation_pct > 70)

        return WalkForwardResults(
            in_sample_returns=is_returns,
            out_of_sample_returns=oos_returns,
            in_sample_equity_curves=results["in_sample_curves"],
            out_of_sample_equity_curves=results["out_of_sample_curves"],
            window_dates=results["window_dates"],
            best_params_per_window=results["best_params"],
            avg_in_sample_return=avg_is,
            avg_out_of_sample_return=avg_oos,
            efficiency_ratio=efficiency,
            consistency=consistency,
            return_degradation=degradation_pct,
            is_overfit=is_overfit,
        )

    @staticmethod
    def plot_walk_forward_results(results: WalkForwardResults, ticker: str = ""):
        """Plot walk-forward analysis results"""
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#121212")

        # Plot 1: Returns by window
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_facecolor("#121212")

        windows = range(1, len(results.in_sample_returns) + 1)

        ax1.plot(
            windows,
            results.in_sample_returns,
            "o-",
            color="#2979ff",
            linewidth=2,
            markersize=8,
            label="In-Sample",
        )
        ax1.plot(
            windows,
            results.out_of_sample_returns,
            "s-",
            color="#ff9800",
            linewidth=2,
            markersize=8,
            label="Out-of-Sample",
        )
        ax1.axhline(y=0, color="#888888", linestyle="--", alpha=0.5)

        ax1.set_xlabel("Window Number", color="white")
        ax1.set_ylabel("Return (%)", color="white")
        ax1.set_title("Returns by Window", color="white", fontweight="bold")
        ax1.tick_params(colors="white")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.2)

        # Plot 2: Cumulative equity
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_facecolor("#121212")

        cumulative_is = 1000.0
        cumulative_oos = 1000.0

        is_cumulative = [1000.0]
        oos_cumulative = [1000.0]

        for is_curve, oos_curve in zip(
            results.in_sample_equity_curves, results.out_of_sample_equity_curves
        ):
            is_norm = (is_curve / 1000.0) * cumulative_is
            is_cumulative.extend(is_norm[1:])
            cumulative_is = is_norm[-1]

            oos_norm = (oos_curve / 1000.0) * cumulative_oos
            oos_cumulative.extend(oos_norm[1:])
            cumulative_oos = oos_norm[-1]

        ax2.plot(
            is_cumulative,
            color="#2979ff",
            linewidth=2,
            label=f"IS: ${cumulative_is:.0f}",
        )
        ax2.plot(
            oos_cumulative,
            color="#ff9800",
            linewidth=2,
            label=f"OOS: ${cumulative_oos:.0f}",
        )
        ax2.axhline(y=1000, color="#888888", linestyle="--", alpha=0.5)

        ax2.set_xlabel("Cumulative Bars", color="white")
        ax2.set_ylabel("Equity ($)", color="white")
        ax2.set_title("Cumulative Equity", color="white", fontweight="bold")
        ax2.tick_params(colors="white")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.2)

        # Plot 3: Degradation
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_facecolor("#121212")

        degradations = []
        for is_ret, oos_ret in zip(
            results.in_sample_returns, results.out_of_sample_returns
        ):
            deg = ((is_ret - oos_ret) / abs(is_ret) * 100) if is_ret != 0 else 0
            degradations.append(deg)

        colors = [
            "#ff4444" if d > 50 else "#ffaa00" if d > 20 else "#44ff44"
            for d in degradations
        ]

        ax3.bar(windows, degradations, color=colors, alpha=0.7, edgecolor="white")
        ax3.axhline(y=0, color="#888888", linestyle="-", linewidth=1)
        ax3.axhline(y=50, color="#ff4444", linestyle="--", alpha=0.5)

        ax3.set_xlabel("Window", color="white")
        ax3.set_ylabel("Degradation (%)", color="white")
        ax3.set_title("Performance Degradation", color="white", fontweight="bold")
        ax3.tick_params(colors="white")
        ax3.grid(True, alpha=0.2)

        # Plot 4: Summary
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_facecolor("#121212")
        ax4.axis("off")

        summary = f"""
╔════════════════════════════════════════╗
║   WALK-FORWARD ANALYSIS SUMMARY        ║
╚════════════════════════════════════════╝

📊 RESULTS:
   In-Sample Avg:     {results.avg_in_sample_return:+.2f}%
   Out-of-Sample Avg: {results.avg_out_of_sample_return:+.2f}%

🎯 METRICS:
   Efficiency Ratio:  {results.efficiency_ratio:.2f}
   Consistency:       {results.consistency*100:.1f}%
   Degradation:       {results.return_degradation:.1f}%

✅ VERDICT:
"""

        if results.is_overfit:
            summary += "   ❌ LIKELY OVERFIT\n"
        elif results.efficiency_ratio > 0.7:
            summary += "   ✅ ROBUST STRATEGY\n"
        else:
            summary += "   ⚠️  MIXED RESULTS\n"

        summary += f"\nWindows: {len(results.in_sample_returns)}"

        ax4.text(
            0.05,
            0.95,
            summary,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            color="white",
        )

        title = (
            f"Walk-Forward Analysis - {ticker}" if ticker else "Walk-Forward Analysis"
        )
        plt.suptitle(title, color="white", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return fig

    @staticmethod
    def generate_walk_forward_report(results: WalkForwardResults) -> str:
        """Generate detailed text report"""
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║          WALK-FORWARD ANALYSIS REPORT                         ║
╚═══════════════════════════════════════════════════════════════╝

📊 SUMMARY:
{'─'*65}
Windows Analyzed:           {len(results.in_sample_returns)}
Average In-Sample:          {results.avg_in_sample_return:+.2f}%
Average Out-of-Sample:      {results.avg_out_of_sample_return:+.2f}%

Efficiency Ratio (OOS/IS):  {results.efficiency_ratio:.3f}
Consistency (% profitable): {results.consistency*100:.1f}%
Degradation:                {results.return_degradation:.1f}%

🎯 ASSESSMENT:
{'─'*65}
"""

        if results.is_overfit:
            report += "❌ STRATEGY IS LIKELY OVERFIT\n\n"
            report += "   DO NOT TRADE LIVE\n"
        elif results.efficiency_ratio > 0.7:
            report += "✅ STRATEGY APPEARS ROBUST\n\n"
            report += "   READY FOR PAPER TRADING\n"
        else:
            report += "⚠️  MIXED RESULTS - USE CAUTION\n\n"

        report += f"\n{'═'*65}\n"

        return report
