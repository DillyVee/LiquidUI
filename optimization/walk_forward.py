"""
Walk-Forward Analysis Module

Tests strategy robustness by:
1. Training on historical data (in-sample)
2. Testing on future data (out-of-sample)
3. Rolling this process forward through time
4. Comparing in-sample vs out-of-sample performance

This detects overfitting: If in-sample is great but out-of-sample is poor,
your strategy is curve-fitted to past data and won't work live.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class WalkForwardResults:
    """Results from walk-forward analysis"""
    in_sample_returns: List[float]
    out_of_sample_returns: List[float]
    in_sample_equity_curves: List[np.ndarray]
    out_of_sample_equity_curves: List[np.ndarray]
    window_dates: List[Tuple[str, str, str, str]]  # (train_start, train_end, test_start, test_end)
    best_params_per_window: List[Dict]
    
    # Summary statistics
    avg_in_sample_return: float
    avg_out_of_sample_return: float
    efficiency_ratio: float  # OOS / IS (should be > 0.5 for robust strategy)
    consistency: float  # % of OOS windows that were profitable
    
    # Degradation metrics
    return_degradation: float  # How much worse is OOS vs IS
    is_overfit: bool  # True if severe degradation detected


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for Trading Strategies
    
    Methodology:
    1. Split data into windows (e.g., 6 months train, 1 month test)
    2. Optimize on training window
    3. Test on following window (out-of-sample)
    4. Roll forward and repeat
    5. Combine all OOS results
    """
    
    @staticmethod
    def run_walk_forward(
        optimizer_class,
        df_dict: Dict[str, pd.DataFrame],
        train_days: int = 180,
        test_days: int = 30,
        min_trades: int = 10,
        **optimizer_kwargs
    ) -> WalkForwardResults:
        """
        Run walk-forward analysis
        
        Args:
            optimizer_class: The MultiTimeframeOptimizer class
            df_dict: Dictionary of dataframes by timeframe
            train_days: Days of data for training (in-sample)
            test_days: Days of data for testing (out-of-sample)
            min_trades: Minimum trades required per window
            **optimizer_kwargs: Additional arguments for optimizer
            
        Returns:
            WalkForwardResults object
        """
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD ANALYSIS")
        print(f"{'='*70}")
        print(f"Training window: {train_days} days")
        print(f"Testing window: {test_days} days")
        print(f"Minimum trades: {min_trades}")
        
        # 🔥 FIX: Ensure ALL dataframes have Datetime column FIRST
        print(f"\n🔧 Preparing data...")
        df_dict_fixed = {}
        for tf, df in df_dict.items():
            df_copy = df.copy()
            
            # Ensure Datetime column exists
            if 'Datetime' not in df_copy.columns:
                print(f"   Adding Datetime column to {tf}")
                df_copy['Datetime'] = pd.to_datetime(df_copy.index)
            
            # Ensure Datetime is datetime64
            if df_copy['Datetime'].dtype != 'datetime64[ns]':
                print(f"   Converting {tf} Datetime to datetime64")
                df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'])
            
            df_dict_fixed[tf] = df_copy
            
            # Show what we have
            dt = df_copy['Datetime']
            days = (dt.max() - dt.min()).days
            print(f"   {tf}: {len(df_copy)} bars, {days} days ({dt.min().date()} to {dt.max().date()})")
        
        # Use fixed dict from now on
        df_dict = df_dict_fixed
        
        # Get finest timeframe
        tf_order = {'5min': 0, 'hourly': 1, 'daily': 2}
        timeframes = list(df_dict.keys())
        finest_tf = sorted(timeframes, key=lambda x: tf_order.get(x, 99))[0]
        
        df_finest = df_dict[finest_tf].copy()
        
        print(f"\n📊 Using {finest_tf} as primary timeframe")
        print(f"   Shape: {df_finest.shape}")
        
        # Calculate number of windows
        total_days = (df_finest['Datetime'].max() - df_finest['Datetime'].min()).days
        window_size = train_days + test_days
        n_windows = max(1, (total_days - train_days) // test_days)
        
        print(f"Total data: {total_days} days")
        print(f"Number of windows: {n_windows}")
        
        # ⚠️ VALIDATION CHECK
        if total_days < window_size:
            raise ValueError(
                f"Insufficient data for walk-forward analysis!\n"
                f"Available: {total_days} days\n"
                f"Required: {window_size} days (train={train_days} + test={test_days})\n"
                f"Solution: Reduce train_days to {total_days // 3} and test_days to {total_days // 6}"
            )
        
        print(f"{'='*70}\n")
        
        # Storage for results
        in_sample_returns = []
        out_of_sample_returns = []
        in_sample_curves = []
        out_of_sample_curves = []
        window_dates = []
        best_params_list = []
        
        for window_idx in range(n_windows):
            print(f"\n{'─'*70}")
            print(f"WINDOW {window_idx + 1}/{n_windows}")
            print(f"{'─'*70}")
            
            # Define date ranges
            start_offset = window_idx * test_days
            
            train_start = df_finest['Datetime'].min() + pd.Timedelta(days=start_offset)
            train_end = train_start + pd.Timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_days)
            
            # Check if we have enough data
            if test_end > df_finest['Datetime'].max():
                print(f"⚠️  Window {window_idx + 1} exceeds available data, stopping")
                break
            
            print(f"Training:  {train_start.date()} to {train_end.date()}")
            print(f"Testing:   {test_start.date()} to {test_end.date()}")
            
            # Split data into train/test for all timeframes
            train_dict = {}
            test_dict = {}
            
            for tf in timeframes:
                df_tf = df_dict[tf].copy()
                
                # Training data
                train_mask = (df_tf['Datetime'] >= train_start) & (df_tf['Datetime'] < train_end)
                train_dict[tf] = df_tf[train_mask].copy().reset_index(drop=True)
                
                # Testing data
                test_mask = (df_tf['Datetime'] >= test_start) & (df_tf['Datetime'] < test_end)
                test_dict[tf] = df_tf[test_mask].copy().reset_index(drop=True)
                
                print(f"  {tf}: Train={len(train_dict[tf])} bars, Test={len(test_dict[tf])} bars")
            
            # Check if we have enough data
            if len(train_dict[finest_tf]) < 100 or len(test_dict[finest_tf]) < 10:
                print(f"⚠️  Insufficient data in window {window_idx + 1}, skipping")
                continue
            
            try:
                # PHASE 1: OPTIMIZE on training data (IN-SAMPLE)
                print(f"\n📊 Phase 1: Optimizing on training data...")
                
                optimizer = optimizer_class(
                    df_dict=train_dict,
                    **optimizer_kwargs
                )
                
                # Run optimization
                optimizer.stopped = False
                optimizer.run()
                
                # Wait for completion
                optimizer.wait()
                
                if not optimizer.all_results or len(optimizer.all_results) == 0:
                    print(f"⚠️  Optimization failed in window {window_idx + 1}")
                    continue
                
                # Get best parameters
                best_params = optimizer.all_results[0]
                best_params_list.append(best_params)
                
                # Calculate in-sample performance
                is_equity_curve, is_trades = optimizer.simulate_multi_tf(best_params)
                
                if is_equity_curve is None or is_trades < min_trades:
                    print(f"⚠️  Insufficient trades in-sample ({is_trades}), skipping")
                    continue
                
                is_return = (is_equity_curve[-1] / 1000.0 - 1) * 100
                in_sample_returns.append(is_return)
                in_sample_curves.append(is_equity_curve)
                
                print(f"✓ In-sample: {is_return:+.2f}% ({is_trades} trades)")
                
                # PHASE 2: TEST on out-of-sample data
                print(f"\n📈 Phase 2: Testing on out-of-sample data...")
                
                # Create optimizer with test data (no optimization, just simulation)
                test_optimizer = optimizer_class(
                    df_dict=test_dict,
                    n_trials=1,  # We're not optimizing, just simulating
                    **{k: v for k, v in optimizer_kwargs.items() if k != 'n_trials'}
                )
                
                # Simulate with best params from training
                oos_equity_curve, oos_trades = test_optimizer.simulate_multi_tf(best_params)
                
                if oos_equity_curve is None or oos_trades < 1:
                    print(f"⚠️  No trades out-of-sample, skipping")
                    continue
                
                oos_return = (oos_equity_curve[-1] / 1000.0 - 1) * 100
                out_of_sample_returns.append(oos_return)
                out_of_sample_curves.append(oos_equity_curve)
                
                print(f"✓ Out-of-sample: {oos_return:+.2f}% ({oos_trades} trades)")
                
                # Calculate degradation
                degradation = ((is_return - oos_return) / abs(is_return) * 100) if is_return != 0 else 0
                print(f"  Degradation: {degradation:.1f}%")
                
                if degradation > 50:
                    print(f"  ⚠️  HIGH DEGRADATION - potential overfitting!")
                elif degradation < 0:
                    print(f"  ✨ BETTER OOS than IS - good sign!")
                
                # Store dates
                window_dates.append((
                    train_start.strftime('%Y-%m-%d'),
                    train_end.strftime('%Y-%m-%d'),
                    test_start.strftime('%Y-%m-%d'),
                    test_end.strftime('%Y-%m-%d')
                ))
                
            except Exception as e:
                print(f"✗ Error in window {window_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate summary statistics
        if len(out_of_sample_returns) == 0:
            raise ValueError("No valid walk-forward windows completed")
        
        avg_is = np.mean(in_sample_returns)
        avg_oos = np.mean(out_of_sample_returns)
        
        # Efficiency ratio: OOS / IS (higher is better, > 0.5 is good)
        efficiency = (avg_oos / avg_is) if avg_is != 0 else 0
        
        # Consistency: % of OOS windows that were profitable
        oos_profitable = sum(1 for r in out_of_sample_returns if r > 0)
        consistency = oos_profitable / len(out_of_sample_returns)
        
        # Degradation
        degradation = avg_is - avg_oos
        degradation_pct = (degradation / abs(avg_is) * 100) if avg_is != 0 else 0
        
        # Overfitting detection
        is_overfit = (efficiency < 0.3) or (consistency < 0.4) or (degradation_pct > 70)
        
        results = WalkForwardResults(
            in_sample_returns=in_sample_returns,
            out_of_sample_returns=out_of_sample_returns,
            in_sample_equity_curves=in_sample_curves,
            out_of_sample_equity_curves=out_of_sample_curves,
            window_dates=window_dates,
            best_params_per_window=best_params_list,
            avg_in_sample_return=avg_is,
            avg_out_of_sample_return=avg_oos,
            efficiency_ratio=efficiency,
            consistency=consistency,
            return_degradation=degradation_pct,
            is_overfit=is_overfit
        )
        
        return results

    def plot_walk_forward_results(results: WalkForwardResults, ticker: str = ""):
        """
        Plot walk-forward analysis results
        
        Creates 3 plots:
        1. In-sample vs Out-of-sample returns by window
        2. Cumulative equity curves
        3. Return degradation analysis
        """
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#121212')
        
        # Plot 1: Returns by window
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_facecolor('#121212')
        
        windows = range(1, len(results.in_sample_returns) + 1)
        
        ax1.plot(windows, results.in_sample_returns, 'o-', 
                color='#2979ff', linewidth=2, markersize=8, label='In-Sample')
        ax1.plot(windows, results.out_of_sample_returns, 's-', 
                color='#ff9800', linewidth=2, markersize=8, label='Out-of-Sample')
        ax1.axhline(y=0, color='#888888', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Window Number', color='white')
        ax1.set_ylabel('Return (%)', color='white')
        ax1.set_title('Returns by Walk-Forward Window', color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Cumulative equity
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_facecolor('#121212')
        
        # Concatenate all equity curves
        cumulative_is = 1000.0
        cumulative_oos = 1000.0
        
        is_cumulative_curve = [1000.0]
        oos_cumulative_curve = [1000.0]
        
        for is_curve, oos_curve in zip(results.in_sample_equity_curves, 
                                        results.out_of_sample_equity_curves):
            # Append in-sample
            is_curve_norm = (is_curve / 1000.0) * cumulative_is
            is_cumulative_curve.extend(is_curve_norm[1:])
            cumulative_is = is_curve_norm[-1]
            
            # Append out-of-sample
            oos_curve_norm = (oos_curve / 1000.0) * cumulative_oos
            oos_cumulative_curve.extend(oos_curve_norm[1:])
            cumulative_oos = oos_curve_norm[-1]
        
        ax2.plot(is_cumulative_curve, color='#2979ff', linewidth=2, 
                label=f'In-Sample (Final: ${cumulative_is:.0f})')
        ax2.plot(oos_cumulative_curve, color='#ff9800', linewidth=2, 
                label=f'Out-of-Sample (Final: ${cumulative_oos:.0f})')
        ax2.axhline(y=1000, color='#888888', linestyle='--', alpha=0.5, label='Break-even')
        
        ax2.set_xlabel('Cumulative Bars', color='white')
        ax2.set_ylabel('Equity ($)', color='white')
        ax2.set_title('Cumulative Equity Curves', color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.2)
        
        # Plot 3: Degradation analysis
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_facecolor('#121212')
        
        degradations = []
        for is_ret, oos_ret in zip(results.in_sample_returns, results.out_of_sample_returns):
            deg = ((is_ret - oos_ret) / abs(is_ret) * 100) if is_ret != 0 else 0
            degradations.append(deg)
        
        colors = ['#ff4444' if d > 50 else '#ffaa00' if d > 20 else '#44ff44' for d in degradations]
        
        ax3.bar(windows, degradations, color=colors, alpha=0.7, edgecolor='white')
        ax3.axhline(y=0, color='#888888', linestyle='-', linewidth=1)
        ax3.axhline(y=50, color='#ff4444', linestyle='--', alpha=0.5, label='Danger Zone (>50%)')
        
        ax3.set_xlabel('Window Number', color='white')
        ax3.set_ylabel('Degradation (%)', color='white')
        ax3.set_title('Performance Degradation (IS - OOS)', color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.2)
        
        # Plot 4: Summary statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_facecolor('#121212')
        ax4.axis('off')
        
        # Create summary text
        summary = f"""
╔════════════════════════════════════════════════════╗
║     WALK-FORWARD ANALYSIS SUMMARY                  ║
╚════════════════════════════════════════════════════╝

📊 PERFORMANCE METRICS:
   Average In-Sample:     {results.avg_in_sample_return:+.2f}%
   Average Out-of-Sample: {results.avg_out_of_sample_return:+.2f}%
   
🎯 ROBUSTNESS INDICATORS:
   Efficiency Ratio:      {results.efficiency_ratio:.2f}
      (OOS/IS, >0.5 = good, >0.8 = excellent)
   
   Consistency:           {results.consistency*100:.1f}%
      (% of OOS windows profitable)
   
   Return Degradation:    {results.return_degradation:.1f}%
      (<30% = good, <50% = acceptable, >70% = danger)

✅ VERDICT:
"""
        
        if results.is_overfit:
            summary += "   ❌ LIKELY OVERFIT - Strategy may fail live\n"
            summary += "   • High degradation from IS to OOS\n"
            summary += "   • Consider: Fewer parameters, more data\n"
        elif results.efficiency_ratio > 0.7:
            summary += "   ✅ EXCELLENT - Strategy appears robust\n"
            summary += "   • OOS performance close to IS\n"
            summary += "   • Good consistency across windows\n"
        elif results.efficiency_ratio > 0.5:
            summary += "   ✅ GOOD - Strategy shows promise\n"
            summary += "   • Acceptable OOS performance\n"
            summary += "   • Monitor in paper trading\n"
        else:
            summary += "   ⚠️  MARGINAL - Use with caution\n"
            summary += "   • Significant performance drop OOS\n"
            summary += "   • Consider refinement\n"
        
        summary += f"\n📅 WINDOWS ANALYZED: {len(results.in_sample_returns)}"
        
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                color='white')
        
        title = f'Walk-Forward Analysis - {ticker}' if ticker else 'Walk-Forward Analysis'
        plt.suptitle(title, color='white', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_walk_forward_report(results: WalkForwardResults) -> str:
        """Generate detailed text report"""
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║          WALK-FORWARD ANALYSIS DETAILED REPORT                ║
╚═══════════════════════════════════════════════════════════════╝

📊 SUMMARY STATISTICS:
{'─'*65}
Windows Analyzed:           {len(results.in_sample_returns)}

Average In-Sample Return:   {results.avg_in_sample_return:+.2f}%
Average Out-of-Sample:      {results.avg_out_of_sample_return:+.2f}%

Efficiency Ratio (OOS/IS):  {results.efficiency_ratio:.3f}
Consistency (% OOS profit): {results.consistency*100:.1f}%
Return Degradation:         {results.return_degradation:.1f}%

🎯 INTERPRETATION:
{'─'*65}
"""
        
        # Efficiency interpretation
        if results.efficiency_ratio > 0.8:
            report += "✅ Efficiency Ratio EXCELLENT (>0.8)\n"
            report += "   OOS performance is 80%+ of IS performance\n"
        elif results.efficiency_ratio > 0.5:
            report += "✅ Efficiency Ratio GOOD (>0.5)\n"
            report += "   OOS performance is acceptable\n"
        elif results.efficiency_ratio > 0.3:
            report += "⚠️  Efficiency Ratio MARGINAL (0.3-0.5)\n"
            report += "   Significant performance drop out-of-sample\n"
        else:
            report += "❌ Efficiency Ratio POOR (<0.3)\n"
            report += "   Strategy fails out-of-sample - likely overfit\n"
        
        report += "\n"
        
        # Consistency interpretation
        if results.consistency > 0.7:
            report += "✅ Consistency EXCELLENT (>70%)\n"
            report += "   Most OOS windows are profitable\n"
        elif results.consistency > 0.5:
            report += "✅ Consistency GOOD (>50%)\n"
            report += "   More wins than losses out-of-sample\n"
        else:
            report += "❌ Consistency POOR (<50%)\n"
            report += "   Strategy loses more often than it wins OOS\n"
        
        report += "\n"
        
        # Degradation interpretation
        if results.return_degradation < 30:
            report += "✅ Degradation MINIMAL (<30%)\n"
            report += "   Strategy maintains performance OOS\n"
        elif results.return_degradation < 50:
            report += "⚠️  Degradation MODERATE (30-50%)\n"
            report += "   Some performance loss, but acceptable\n"
        elif results.return_degradation < 70:
            report += "⚠️  Degradation HIGH (50-70%)\n"
            report += "   Significant performance loss OOS\n"
        else:
            report += "❌ Degradation SEVERE (>70%)\n"
            report += "   Strategy is likely overfit to in-sample data\n"
        
        report += f"\n\n📈 WINDOW-BY-WINDOW BREAKDOWN:\n{'─'*65}\n"
        
        for i in range(len(results.in_sample_returns)):
            is_ret = results.in_sample_returns[i]
            oos_ret = results.out_of_sample_returns[i]
            deg = ((is_ret - oos_ret) / abs(is_ret) * 100) if is_ret != 0 else 0
            
            dates = results.window_dates[i]
            
            status = "✅" if oos_ret > 0 else "❌"
            deg_status = "🔥" if deg > 50 else "⚠️ " if deg > 20 else "✓ "
            
            report += f"Window {i+1}: {dates[0]} to {dates[3]}\n"
            report += f"  IS:  {is_ret:+.2f}% | OOS: {oos_ret:+.2f}% {status}\n"
            report += f"  Degradation: {deg:.1f}% {deg_status}\n\n"
        
        report += f"\n{'═'*65}\n"
        report += "FINAL VERDICT:\n"
        report += f"{'═'*65}\n"
        
        if results.is_overfit:
            report += "❌ STRATEGY IS LIKELY OVERFIT\n\n"
            report += "   DO NOT TRADE THIS STRATEGY LIVE\n\n"
            report += "   Recommendations:\n"
            report += "   • Simplify strategy (fewer parameters)\n"
            report += "   • Use more data for optimization\n"
            report += "   • Add robustness constraints\n"
            report += "   • Consider different approach\n"
        elif results.efficiency_ratio > 0.7 and results.consistency > 0.6:
            report += "✅ STRATEGY APPEARS ROBUST\n\n"
            report += "   READY FOR PAPER TRADING\n\n"
            report += "   Next steps:\n"
            report += "   • Run Monte Carlo simulation\n"
            report += "   • Start paper trading\n"
            report += "   • Monitor live performance\n"
        else:
            report += "⚠️  STRATEGY SHOWS MIXED RESULTS\n\n"
            report += "   USE WITH CAUTION\n\n"
            report += "   Recommendations:\n"
            report += "   • Paper trade with small size\n"
            report += "   • Monitor performance closely\n"
            report += "   • Be ready to stop if underperforming\n"
        
        report += f"\n{'═'*65}\n"
        
        return report