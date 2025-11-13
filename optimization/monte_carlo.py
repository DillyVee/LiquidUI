"""
Monte Carlo Simulation for Strategy Robustness Testing

Monte Carlo methods randomize trade sequences to assess:
1. How sensitive is the strategy to trade order?
2. What's the range of possible outcomes?
3. How likely is the strategy to meet expectations?
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation"""
    original_equity: float
    mean_equity: float
    median_equity: float
    percentile_5: float
    percentile_95: float
    std_dev: float
    min_equity: float
    max_equity: float
    probability_profit: float
    simulations: List[np.ndarray]


class MonteCarloSimulator:
    """
    Monte Carlo simulation for backtesting validation
    
    Two main approaches:
    1. Trade Randomization: Randomly reorder trades
    2. Return Bootstrap: Resample trade returns with replacement
    """
    
    """
COMPLETE FIX FOR MONTE CARLO SIMULATION

The issue: Your Monte Carlo is showing all identical results because either:
1. The trade log has identical returns (bug in simulate_multi_tf), OR
2. The randomization isn't working (bug in simulate_trade_randomization)

Apply these fixes:
"""

# ============================================================
# FIX 1: Update optimization/monte_carlo.py
# ============================================================

# Replace the simulate_trade_randomization method with this:

    @staticmethod
    def simulate_trade_randomization(
        trades: List[Dict],
        n_simulations: int = 1000,
        initial_equity: float = 1000.0
    ) -> MonteCarloResults:
        """
        Randomize trade order to see range of outcomes
        
        FIXED: Now properly extracts and shuffles returns
        """
        if not trades or len(trades) == 0:
            raise ValueError("No trades provided for Monte Carlo simulation")

        # Extract returns as numpy array
        returns = np.array([trade['Percent_Change'] / 100.0 for trade in trades], dtype=np.float64)
        
        # 🔍 DIAGNOSTIC OUTPUT
        print(f"\n{'='*60}")
        print(f"MONTE CARLO SIMULATION SETUP")
        print(f"{'='*60}")
        print(f"Total trades: {len(returns)}")
        print(f"Returns range: [{returns.min()*100:.2f}%, {returns.max()*100:.2f}%]")
        print(f"Returns mean: {returns.mean()*100:.2f}%")
        print(f"Returns std: {returns.std()*100:.2f}%")
        print(f"Unique returns: {len(np.unique(returns))} ({len(np.unique(returns))/len(returns)*100:.1f}%)")
        
        # ⚠️ Check for problematic scenarios
        if len(np.unique(returns)) == 1:
            print(f"\n⚠️  WARNING: All returns are identical ({returns[0]*100:.2f}%)")
            print(f"Monte Carlo will show no variation. This indicates:")
            print(f"  - All trades hit same profit target")
            print(f"  - All trades have same hold time")
            print(f"  - Strategy has very tight parameters")
        
        # Show first few returns
        print(f"\nFirst 10 returns:")
        for i, ret in enumerate(returns[:10]):
            print(f"  Trade {i+1}: {ret*100:+.2f}%")
        
        # Original equity curve (trades in actual order)
        original_equity_curve = MonteCarloSimulator._calculate_equity_curve(
            returns, initial_equity
        )
        original_final = original_equity_curve[-1]
        
        print(f"\nOriginal sequence: ${original_final:,.2f} ({(original_final/initial_equity-1)*100:+.2f}%)")
        
        # Run simulations
        final_equities = []
        all_curves = []
        
        # Show first few shuffles to verify randomization
        print(f"\nTesting randomization (first 3 returns of 3 shuffles):")
        for test_idx in range(3):
            test_shuffle = np.random.permutation(returns)
            print(f"  Shuffle {test_idx+1}: [{', '.join([f'{r*100:+.1f}%' for r in test_shuffle[:3]])}...]")
        
        print(f"\nRunning {n_simulations} simulations...")
        
        for sim_idx in range(n_simulations):
            # ✅ Shuffle returns (not trade objects!)
            sampled_returns = np.random.permutation(returns)
            
            # Calculate equity curve for this shuffled sequence
            eq_curve = MonteCarloSimulator._calculate_equity_curve(
                sampled_returns, initial_equity
            )
            
            final_equities.append(eq_curve[-1])
            all_curves.append(eq_curve)
            
            # Show progress
            if sim_idx in [0, 1, 2, 99, 499, 999]:
                print(f"  Sim {sim_idx+1}: ${eq_curve[-1]:,.2f}")
        
        final_equities = np.array(final_equities)
        
        # Results
        print(f"\n{'='*60}")
        print(f"MONTE CARLO RESULTS")
        print(f"{'='*60}")
        print(f"Min:    ${final_equities.min():,.2f}")
        print(f"Max:    ${final_equities.max():,.2f}")
        print(f"Mean:   ${final_equities.mean():,.2f}")
        print(f"Median: ${np.median(final_equities):,.2f}")
        print(f"Std:    ${final_equities.std():,.2f}")
        print(f"Range:  ${final_equities.max() - final_equities.min():,.2f}")
        print(f"Unique outcomes: {len(np.unique(final_equities))}")
        print(f"{'='*60}\n")
        
        # Calculate statistics
        return MonteCarloResults(
            original_equity=original_final,
            mean_equity=np.mean(final_equities),
            median_equity=np.median(final_equities),
            percentile_5=np.percentile(final_equities, 5),
            percentile_95=np.percentile(final_equities, 95),
            std_dev=np.std(final_equities),
            min_equity=np.min(final_equities),
            max_equity=np.max(final_equities),
            probability_profit=np.mean(final_equities > initial_equity),
            simulations=all_curves
        )

    
    @staticmethod
    def simulate_bootstrap(
        trades: List[Dict],
        n_simulations: int = 1000,
        initial_equity: float = 1000.0,
        n_trades_per_sim: Optional[int] = None
    ) -> MonteCarloResults:
        """
        Bootstrap resampling: randomly sample trades with replacement
        
        Method: Create new trade sequences by sampling from original trades
        Useful when you want to test with different trade counts
        
        Args:
            trades: List of trade dictionaries
            n_simulations: Number of bootstrap samples
            initial_equity: Starting capital
            n_trades_per_sim: Trades per simulation (None = same as original)
            
        Returns:
            MonteCarloResults object
        """
        if not trades or len(trades) == 0:
            raise ValueError("No trades provided for Monte Carlo simulation")
        
        returns = np.array([trade['Percent_Change'] / 100.0 for trade in trades])
        
        if n_trades_per_sim is None:
            n_trades_per_sim = len(returns)
        
        # Original equity
        original_equity_curve = MonteCarloSimulator._calculate_equity_curve(
            returns, initial_equity
        )
        original_final = original_equity_curve[-1]
        
        # Run simulations
        final_equities = []
        all_curves = []
        
        for _ in range(n_simulations):
            # Bootstrap sample (with replacement)
            sampled_returns = np.random.choice(returns, size=n_trades_per_sim, replace=True)
            
            # Calculate equity curve
            eq_curve = MonteCarloSimulator._calculate_equity_curve(
                sampled_returns, initial_equity
            )
            
            final_equities.append(eq_curve[-1])
            all_curves.append(eq_curve)
        
        final_equities = np.array(final_equities)
        
        return MonteCarloResults(
            original_equity=original_final,
            mean_equity=np.mean(final_equities),
            median_equity=np.median(final_equities),
            percentile_5=np.percentile(final_equities, 5),
            percentile_95=np.percentile(final_equities, 95),
            std_dev=np.std(final_equities),
            min_equity=np.min(final_equities),
            max_equity=np.max(final_equities),
            probability_profit=np.mean(final_equities > initial_equity),
            simulations=all_curves
        )
    
    @staticmethod
    def simulate_with_drawdown_constraint(
        trades: List[Dict],
        n_simulations: int = 1000,
        initial_equity: float = 1000.0,
        max_drawdown_pct: float = 20.0
    ) -> Dict:
        """
        Monte Carlo with stop-loss: Stop simulation if drawdown exceeds threshold
        
        Tests: How often would risk management have stopped the strategy?
        
        Args:
            trades: List of trade dictionaries
            n_simulations: Number of simulations
            initial_equity: Starting capital
            max_drawdown_pct: Stop if drawdown exceeds this (e.g., 20.0 = 20%)
            
        Returns:
            Dictionary with results including stop-out statistics
        """
        returns = np.array([trade['Percent_Change'] / 100.0 for trade in trades])
        
        stopped_count = 0
        completed_count = 0
        final_equities_completed = []
        final_equities_stopped = []
        
        for _ in range(n_simulations):
            sampled_returns = np.random.permutation(returns)
            equity = initial_equity
            peak = initial_equity
            stopped = False
            
            for ret in sampled_returns:
                equity *= (1 + ret)
                peak = max(peak, equity)
                
                # Check drawdown
                drawdown = (peak - equity) / peak * 100
                
                if drawdown > max_drawdown_pct:
                    stopped = True
                    stopped_count += 1
                    final_equities_stopped.append(equity)
                    break
            
            if not stopped:
                completed_count += 1
                final_equities_completed.append(equity)
        
        return {
            'stopped_count': stopped_count,
            'completed_count': completed_count,
            'stop_out_rate': stopped_count / n_simulations,
            'avg_equity_completed': np.mean(final_equities_completed) if final_equities_completed else 0,
            'avg_equity_stopped': np.mean(final_equities_stopped) if final_equities_stopped else 0,
            'max_drawdown_threshold': max_drawdown_pct
        }
    
    @staticmethod
    def _calculate_equity_curve(returns: np.ndarray, initial_equity: float) -> np.ndarray:
        """Calculate equity curve from returns"""
        equity_curve = np.zeros(len(returns) + 1)
        equity_curve[0] = initial_equity
        
        for i, ret in enumerate(returns):
            equity_curve[i + 1] = equity_curve[i] * (1 + ret)
        
        return equity_curve

        



    @staticmethod
    def plot_monte_carlo_results(
        results: MonteCarloResults,
        title: str = "Monte Carlo Simulation Results",
        max_paths: int = 100
    ):
        """
        Plot Monte Carlo simulation results
        
        Args:
            results: MonteCarloResults object
            title: Plot title
            max_paths: Maximum number of paths to show (avoid clutter)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.patch.set_facecolor('#121212')
        
        # Plot 1: Equity curves (sample of paths)
        ax1.set_facecolor('#121212')
        
        # Show random sample of paths
        n_paths = min(max_paths, len(results.simulations))
        sample_indices = np.random.choice(len(results.simulations), n_paths, replace=False)
        
        for idx in sample_indices:
            ax1.plot(results.simulations[idx], color='#2979ff', alpha=0.1, linewidth=0.5)
        
        # Show percentile bands
        if results.simulations:
            n_steps = len(results.simulations[0])
            percentile_5_curve = np.zeros(n_steps)
            percentile_95_curve = np.zeros(n_steps)
            median_curve = np.zeros(n_steps)
            
            for i in range(n_steps):
                values = [sim[i] for sim in results.simulations]
                percentile_5_curve[i] = np.percentile(values, 5)
                percentile_95_curve[i] = np.percentile(values, 95)
                median_curve[i] = np.median(values)
            
            ax1.fill_between(
                range(n_steps), percentile_5_curve, percentile_95_curve,
                color='#FFA500', alpha=0.3, label='5-95% Range'
            )
            ax1.plot(median_curve, color='#FFA500', linewidth=2, label='Median Path')
        
        ax1.axhline(y=1000, color='#888888', linestyle='--', alpha=0.5, label='Break-even')
        ax1.set_xlabel('Trade Number', color='white')
        ax1.set_ylabel('Equity ($)', color='white')
        ax1.set_title('Monte Carlo Equity Paths', color='white')
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Distribution of final equities
        ax2.set_facecolor('#121212')
        
        final_equities = np.array([sim[-1] for sim in results.simulations])
        
        # ✅ ROBUST BIN CALCULATION
        data_range = np.ptp(final_equities)  # max - min
        n_unique = len(np.unique(final_equities))
        n_data = len(final_equities)
        
        # Handle edge cases
        if n_unique <= 1:
            # All values are the same - show single bar
            n_bins = 1
        elif n_unique <= 5:
            # Very few unique values - use exact number
            n_bins = n_unique
        elif data_range < 1e-6:
            # Virtually no range - use few bins
            n_bins = min(n_unique, 5)
        else:
            # Normal case - use adaptive calculation
            # Sturges' rule: k = ceil(log2(n) + 1)
            sturges = int(np.ceil(np.log2(n_data) + 1))
            
            # Rice rule: k = ceil(2 * n^(1/3))
            rice = int(np.ceil(2 * (n_data ** (1/3))))
            
            # Use the minimum to avoid too many bins
            n_bins = min(sturges, rice, n_unique, 50)
            n_bins = max(n_bins, 10)  # At least 10 bins for normal data
        
        print(f"📊 Histogram: {n_data} points, {n_unique} unique, range={data_range:.2f}, using {n_bins} bins")
        
        try:
            # Try to create histogram with calculated bins
            counts, bins, patches = ax2.hist(
                final_equities, 
                bins=n_bins, 
                color='#2979ff', 
                alpha=0.7, 
                edgecolor='white',
                linewidth=0.5
            )
        except (ValueError, RuntimeError) as e:
            print(f"⚠️  Histogram warning: {e}")
            # Fallback: let matplotlib decide
            try:
                counts, bins, patches = ax2.hist(
                    final_equities, 
                    bins='auto', 
                    color='#2979ff', 
                    alpha=0.7, 
                    edgecolor='white',
                    linewidth=0.5
                )
            except Exception as e2:
                print(f"⚠️  Histogram fallback failed: {e2}")
                # Last resort: use fixed small number
                counts, bins, patches = ax2.hist(
                    final_equities, 
                    bins=max(n_unique, 5), 
                    color='#2979ff', 
                    alpha=0.7, 
                    edgecolor='white',
                    linewidth=0.5
                )
        
        # Add vertical lines for statistics
        ax2.axvline(results.original_equity, color='#00ff00', linestyle='--', 
                linewidth=2, label=f'Original: ${results.original_equity:.0f}')
        ax2.axvline(results.median_equity, color='#FFA500', linestyle='-', 
                linewidth=2, label=f'Median: ${results.median_equity:.0f}')
        ax2.axvline(results.percentile_5, color='#ff4444', linestyle=':', 
                linewidth=2, label=f'5%: ${results.percentile_5:.0f}')
        ax2.axvline(results.percentile_95, color='#44ff44', linestyle=':', 
                linewidth=2, label=f'95%: ${results.percentile_95:.0f}')
        
        ax2.set_xlabel('Final Equity ($)', color='white')
        ax2.set_ylabel('Frequency', color='white')
        ax2.set_title('Distribution of Final Equity', color='white')
        ax2.tick_params(colors='white')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.2)
        
        plt.suptitle(title, color='white', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def generate_monte_carlo_report(results: MonteCarloResults, initial_equity: float = 1000.0) -> str:
        """Generate text report from Monte Carlo results"""
        original_return = (results.original_equity / initial_equity - 1) * 100
        mean_return = (results.mean_equity / initial_equity - 1) * 100
        median_return = (results.median_equity / initial_equity - 1) * 100
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║          MONTE CARLO SIMULATION REPORT                   ║
╚══════════════════════════════════════════════════════════╝

📊 ORIGINAL BACKTEST:
   Final Equity:  ${results.original_equity:,.2f}
   Return:        {original_return:+.2f}%

📈 MONTE CARLO RESULTS ({len(results.simulations)} simulations):
   Mean Equity:   ${results.mean_equity:,.2f} ({mean_return:+.2f}%)
   Median Equity: ${results.median_equity:,.2f} ({median_return:+.2f}%)
   
🎯 CONFIDENCE INTERVALS:
   95% Confidence: ${results.percentile_5:,.2f} to ${results.percentile_95:,.2f}
   Range:          ${results.max_equity - results.min_equity:,.2f}
   Std Dev:        ${results.std_dev:,.2f}
   
⚠️  RISK ASSESSMENT:
   Best Case:      ${results.max_equity:,.2f}
   Worst Case:     ${results.min_equity:,.2f}
   Probability of Profit: {results.probability_profit * 100:.1f}%

🎲 INTERPRETATION:
"""
        
        # Add interpretation
        if results.probability_profit < 0.5:
            report += "   ❌ Strategy is more likely to LOSE than win in random scenarios\n"
        elif results.probability_profit < 0.7:
            report += "   ⚠️  Strategy has moderate probability of profit\n"
        elif results.probability_profit < 0.9:
            report += "   ✅ Strategy has good probability of profit\n"
        else:
            report += "   ✅✅ Strategy is very robust to trade randomization\n"
        
        if results.original_equity > results.percentile_95:
            report += "   🎉 Original result is in TOP 5% (may be lucky!)\n"
        elif results.original_equity < results.percentile_5:
            report += "   ⚠️  Original result is in BOTTOM 5% (may be unlucky!)\n"
        else:
            report += "   ✓ Original result is within normal range\n"
        
        deviation = abs(results.original_equity - results.median_equity) / results.std_dev
        if deviation > 2:
            report += "   ⚠️  Original result is >2 std devs from median (unusual!)\n"
        
        report += "\n" + "═" * 60 + "\n"
        
        return report
