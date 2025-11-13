"""
Enhanced Monte Carlo Simulation with Comprehensive Quantitative Metrics

This module extends the basic Monte Carlo simulation with professional-grade
quantitative finance metrics including VaR, CVaR, drawdown analysis, 
risk-adjusted returns, and statistical tests.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats
from scipy.stats import skewtest, kurtosistest


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


@dataclass
class AdvancedMonteCarloMetrics:
    """Advanced quantitative finance metrics for Monte Carlo analysis"""
    # Value at Risk metrics
    var_95: float  # 95% VaR (5th percentile loss)
    var_99: float  # 99% VaR (1st percentile loss)
    cvar_95: float  # Conditional VaR / Expected Shortfall at 95%
    cvar_99: float  # Conditional VaR / Expected Shortfall at 99%
    
    # Drawdown metrics
    max_drawdown_mean: float
    max_drawdown_median: float
    max_drawdown_95: float
    max_drawdown_99: float
    avg_drawdown_duration: float
    avg_recovery_time: float
    
    # Risk-adjusted return metrics
    sharpe_mean: float
    sharpe_median: float
    sharpe_std: float
    sharpe_5th: float
    sharpe_95th: float
    sortino_mean: float
    sortino_std: float
    calmar_mean: float  # Return / Max Drawdown
    calmar_median: float
    omega_ratio: float  # Probability-weighted gains/losses
    
    # Distribution characteristics
    return_skewness: float
    return_kurtosis: float
    return_skew_pval: float  # p-value for normality test
    return_kurt_pval: float  # p-value for excess kurtosis
    
    # Path dependency metrics
    path_dependency_score: float  # How much does order matter?
    order_sensitivity: float  # Coefficient of variation of outcomes
    
    # Confidence intervals (95%)
    mean_return_ci: Tuple[float, float]
    sharpe_ci: Tuple[float, float]
    max_dd_ci: Tuple[float, float]
    
    # Tail risk
    tail_ratio: float  # 95th percentile / 5th percentile
    gain_to_pain_ratio: float  # Sum(positive returns) / abs(Sum(negative returns))
    
    # Win rate stability
    win_rate_mean: float
    win_rate_std: float
    win_rate_ci: Tuple[float, float]


class AdvancedMonteCarloAnalyzer:
    """
    Advanced Monte Carlo analyzer with comprehensive quantitative metrics
    
    This class provides professional-grade risk analysis including:
    - Value at Risk (VaR) and Conditional VaR (CVaR)
    - Drawdown analysis and recovery metrics
    - Risk-adjusted return ratios (Sharpe, Sortino, Calmar, Omega)
    - Statistical tests for distribution characteristics
    - Path dependency and sensitivity analysis
    - Confidence intervals for all major metrics
    """
    
    @staticmethod
    def calculate_advanced_metrics(
        results: MonteCarloResults,
        initial_equity: float = 1000.0,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0
    ) -> AdvancedMonteCarloMetrics:
        """
        Calculate comprehensive quantitative finance metrics
        
        Args:
            results: Basic MonteCarloResults object
            initial_equity: Starting equity
            risk_free_rate: Risk-free rate for Sharpe/Sortino (annualized)
            annualization_factor: Factor for annualizing metrics (252 for daily)
            
        Returns:
            AdvancedMonteCarloMetrics with all calculated metrics
        """
        # Extract final equities and returns from simulations
        final_equities = np.array([sim[-1] for sim in results.simulations])
        returns_pct = ((final_equities - initial_equity) / initial_equity) * 100
        
        # Calculate equity curves returns for each simulation
        all_returns = []
        for sim in results.simulations:
            sim_returns = np.diff(sim) / sim[:-1]
            all_returns.append(sim_returns)
        
        all_returns_flat = np.concatenate(all_returns)
        all_returns_flat = all_returns_flat[~(np.isnan(all_returns_flat) | np.isinf(all_returns_flat))]
        
        # 1. VALUE AT RISK METRICS
        var_95, var_99, cvar_95, cvar_99 = AdvancedMonteCarloAnalyzer._calculate_var_metrics(
            returns_pct, final_equities, initial_equity
        )
        
        # 2. DRAWDOWN METRICS
        dd_metrics = AdvancedMonteCarloAnalyzer._calculate_drawdown_metrics(results.simulations)
        
        # 3. RISK-ADJUSTED RETURN METRICS
        risk_adj_metrics = AdvancedMonteCarloAnalyzer._calculate_risk_adjusted_metrics(
            results.simulations, initial_equity, risk_free_rate, annualization_factor
        )
        
        # 4. DISTRIBUTION CHARACTERISTICS
        skew, kurt, skew_pval, kurt_pval = AdvancedMonteCarloAnalyzer._calculate_distribution_stats(
            returns_pct
        )
        
        # 5. PATH DEPENDENCY
        path_dep, order_sens = AdvancedMonteCarloAnalyzer._calculate_path_dependency(
            results.simulations, results.original_equity
        )
        
        # 6. CONFIDENCE INTERVALS
        mean_return_ci = AdvancedMonteCarloAnalyzer._bootstrap_ci(returns_pct)
        sharpe_ci = AdvancedMonteCarloAnalyzer._bootstrap_ci(risk_adj_metrics['sharpe_ratios'])
        max_dd_ci = AdvancedMonteCarloAnalyzer._bootstrap_ci(dd_metrics['max_drawdowns'])
        
        # 7. TAIL RISK
        tail_ratio = np.percentile(returns_pct, 95) / abs(np.percentile(returns_pct, 5)) if np.percentile(returns_pct, 5) != 0 else 0
        
        positive_returns = all_returns_flat[all_returns_flat > 0]
        negative_returns = all_returns_flat[all_returns_flat < 0]
        gain_to_pain = abs(np.sum(positive_returns) / np.sum(negative_returns)) if len(negative_returns) > 0 else 0
        
        # 8. WIN RATE STABILITY
        win_rates = []
        for sim in results.simulations:
            sim_returns = np.diff(sim) / sim[:-1]
            sim_returns = sim_returns[~(np.isnan(sim_returns) | np.isinf(sim_returns))]
            if len(sim_returns) > 0:
                win_rate = np.sum(sim_returns > 0) / len(sim_returns)
                win_rates.append(win_rate)
        
        win_rate_mean = np.mean(win_rates)
        win_rate_std = np.std(win_rates)
        win_rate_ci = AdvancedMonteCarloAnalyzer._bootstrap_ci(np.array(win_rates))
        
        return AdvancedMonteCarloMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown_mean=dd_metrics['max_dd_mean'],
            max_drawdown_median=dd_metrics['max_dd_median'],
            max_drawdown_95=dd_metrics['max_dd_95'],
            max_drawdown_99=dd_metrics['max_dd_99'],
            avg_drawdown_duration=dd_metrics['avg_dd_duration'],
            avg_recovery_time=dd_metrics['avg_recovery_time'],
            sharpe_mean=risk_adj_metrics['sharpe_mean'],
            sharpe_median=risk_adj_metrics['sharpe_median'],
            sharpe_std=risk_adj_metrics['sharpe_std'],
            sharpe_5th=risk_adj_metrics['sharpe_5th'],
            sharpe_95th=risk_adj_metrics['sharpe_95th'],
            sortino_mean=risk_adj_metrics['sortino_mean'],
            sortino_std=risk_adj_metrics['sortino_std'],
            calmar_mean=risk_adj_metrics['calmar_mean'],
            calmar_median=risk_adj_metrics['calmar_median'],
            omega_ratio=risk_adj_metrics['omega_ratio'],
            return_skewness=skew,
            return_kurtosis=kurt,
            return_skew_pval=skew_pval,
            return_kurt_pval=kurt_pval,
            path_dependency_score=path_dep,
            order_sensitivity=order_sens,
            mean_return_ci=mean_return_ci,
            sharpe_ci=sharpe_ci,
            max_dd_ci=max_dd_ci,
            tail_ratio=tail_ratio,
            gain_to_pain_ratio=gain_to_pain,
            win_rate_mean=win_rate_mean,
            win_rate_std=win_rate_std,
            win_rate_ci=win_rate_ci
        )
    
    @staticmethod
    def _calculate_var_metrics(
        returns_pct: np.ndarray,
        final_equities: np.ndarray,
        initial_equity: float
    ) -> Tuple[float, float, float, float]:
        """Calculate VaR and CVaR at 95% and 99% confidence levels"""
        # VaR is the percentile loss
        var_95 = np.percentile(returns_pct, 5)  # 5th percentile (95% confidence)
        var_99 = np.percentile(returns_pct, 1)  # 1st percentile (99% confidence)
        
        # CVaR (Expected Shortfall) is the mean of losses beyond VaR
        losses_beyond_95 = returns_pct[returns_pct <= var_95]
        cvar_95 = np.mean(losses_beyond_95) if len(losses_beyond_95) > 0 else var_95
        
        losses_beyond_99 = returns_pct[returns_pct <= var_99]
        cvar_99 = np.mean(losses_beyond_99) if len(losses_beyond_99) > 0 else var_99
        
        return var_95, var_99, cvar_95, cvar_99
    
    @staticmethod
    def _calculate_drawdown_metrics(simulations: List[np.ndarray]) -> Dict:
        """Calculate comprehensive drawdown statistics"""
        max_drawdowns = []
        drawdown_durations = []
        recovery_times = []
        
        for sim in simulations:
            # Calculate drawdown curve
            peak = np.maximum.accumulate(sim)
            drawdown = (peak - sim) / peak
            max_dd = np.max(drawdown) * 100  # Convert to percentage
            max_drawdowns.append(max_dd)
            
            # Calculate drawdown duration
            in_drawdown = drawdown > 0.01  # More than 1% drawdown
            if np.any(in_drawdown):
                # Find consecutive drawdown periods
                dd_periods = np.diff(np.concatenate([[0], in_drawdown.astype(int), [0]]))
                dd_starts = np.where(dd_periods == 1)[0]
                dd_ends = np.where(dd_periods == -1)[0]
                
                if len(dd_starts) > 0 and len(dd_ends) > 0:
                    durations = dd_ends - dd_starts
                    drawdown_durations.extend(durations)
                    
                    # Calculate recovery time (time from max DD to recovery)
                    for start, end in zip(dd_starts, dd_ends):
                        dd_segment = drawdown[start:end]
                        if len(dd_segment) > 0:
                            max_dd_idx = start + np.argmax(dd_segment)
                            recovery = end - max_dd_idx
                            recovery_times.append(recovery)
        
        return {
            'max_drawdowns': np.array(max_drawdowns),
            'max_dd_mean': np.mean(max_drawdowns),
            'max_dd_median': np.median(max_drawdowns),
            'max_dd_95': np.percentile(max_drawdowns, 95),
            'max_dd_99': np.percentile(max_drawdowns, 99),
            'avg_dd_duration': np.mean(drawdown_durations) if drawdown_durations else 0,
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0
        }
    
    @staticmethod
    def _calculate_risk_adjusted_metrics(
        simulations: List[np.ndarray],
        initial_equity: float,
        risk_free_rate: float,
        annualization_factor: float
    ) -> Dict:
        """Calculate Sharpe, Sortino, Calmar, and Omega ratios"""
        sharpe_ratios = []
        sortino_ratios = []
        calmar_ratios = []
        omega_numerators = []
        omega_denominators = []
        
        for sim in simulations:
            returns = np.diff(sim) / sim[:-1]
            returns = returns[~(np.isnan(returns) | np.isinf(returns))]
            
            if len(returns) < 2:
                continue
            
            mean_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1)
            
            # Sharpe Ratio
            if std_ret > 0:
                sharpe = (mean_ret - risk_free_rate / annualization_factor) / std_ret * np.sqrt(annualization_factor)
                sharpe_ratios.append(np.clip(sharpe, -5, 10))
            
            # Sortino Ratio (using downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns, ddof=1)
                if downside_std > 0:
                    sortino = (mean_ret - risk_free_rate / annualization_factor) / downside_std * np.sqrt(annualization_factor)
                    sortino_ratios.append(np.clip(sortino, -5, 10))
            
            # Calmar Ratio (Return / Max Drawdown)
            peak = np.maximum.accumulate(sim)
            drawdown = (peak - sim) / peak
            max_dd = np.max(drawdown)
            
            if max_dd > 0:
                total_return = (sim[-1] / initial_equity - 1)
                calmar = total_return / max_dd
                calmar_ratios.append(calmar)
            
            # Omega Ratio components
            threshold = 0  # Can be set to risk-free rate
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]
            
            omega_numerators.append(np.sum(gains))
            omega_denominators.append(np.sum(losses))
        
        # Calculate Omega Ratio
        total_gains = np.sum(omega_numerators)
        total_losses = np.sum(omega_denominators)
        omega_ratio = total_gains / total_losses if total_losses > 0 else 0
        
        return {
            'sharpe_ratios': np.array(sharpe_ratios),
            'sharpe_mean': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'sharpe_median': np.median(sharpe_ratios) if sharpe_ratios else 0,
            'sharpe_std': np.std(sharpe_ratios) if sharpe_ratios else 0,
            'sharpe_5th': np.percentile(sharpe_ratios, 5) if sharpe_ratios else 0,
            'sharpe_95th': np.percentile(sharpe_ratios, 95) if sharpe_ratios else 0,
            'sortino_mean': np.mean(sortino_ratios) if sortino_ratios else 0,
            'sortino_std': np.std(sortino_ratios) if sortino_ratios else 0,
            'calmar_mean': np.mean(calmar_ratios) if calmar_ratios else 0,
            'calmar_median': np.median(calmar_ratios) if calmar_ratios else 0,
            'omega_ratio': omega_ratio
        }
    
    @staticmethod
    def _calculate_distribution_stats(returns_pct: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate skewness and kurtosis with significance tests"""
        if len(returns_pct) < 8:
            return 0.0, 0.0, 1.0, 1.0
        
        skew = stats.skew(returns_pct, bias=False)
        kurt = stats.kurtosis(returns_pct, bias=False, fisher=True)
        
        # Test for normality using skewness and kurtosis tests
        try:
            skew_stat, skew_pval = skewtest(returns_pct)
        except:
            skew_pval = 1.0
        
        try:
            kurt_stat, kurt_pval = kurtosistest(returns_pct)
        except:
            kurt_pval = 1.0
        
        return skew, kurt, skew_pval, kurt_pval
    
    @staticmethod
    def _calculate_path_dependency(
        simulations: List[np.ndarray],
        original_equity: float
    ) -> Tuple[float, float]:
        """
        Measure how much trade order affects outcomes
        
        Returns:
            path_dependency_score: Normalized std dev of outcomes (0=no dependency, 1=high dependency)
            order_sensitivity: Coefficient of variation
        """
        final_equities = np.array([sim[-1] for sim in simulations])
        
        # Coefficient of variation as measure of sensitivity
        mean_equity = np.mean(final_equities)
        std_equity = np.std(final_equities)
        cv = std_equity / mean_equity if mean_equity > 0 else 0
        
        # Normalized path dependency score
        equity_range = np.max(final_equities) - np.min(final_equities)
        normalized_range = equity_range / original_equity if original_equity > 0 else 0
        
        # Path dependency: 0 = all paths identical, 1 = huge variation
        path_dependency = min(normalized_range / 2.0, 1.0)  # Normalize to 0-1
        
        return path_dependency, cv
    
    @staticmethod
    def _bootstrap_ci(
        data: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(data) < 2:
            return (0.0, 0.0)
        
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrapped_means, alpha/2 * 100)
        upper = np.percentile(bootstrapped_means, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    @staticmethod
    def generate_enhanced_report(
        basic_results: MonteCarloResults,
        advanced_metrics: AdvancedMonteCarloMetrics,
        initial_equity: float = 1000.0
    ) -> str:
        """Generate comprehensive Monte Carlo report with all metrics"""
        original_return = (basic_results.original_equity / initial_equity - 1) * 100
        mean_return = (basic_results.mean_equity / initial_equity - 1) * 100
        
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║     COMPREHENSIVE MONTE CARLO ANALYSIS REPORT                 ║
╚═══════════════════════════════════════════════════════════════╝

📊 BASIC RESULTS ({len(basic_results.simulations)} simulations):
{'─'*65}
Original Return:        {original_return:+.2f}%
Mean Return:            {mean_return:+.2f}%
Median Return:          {(basic_results.median_equity/initial_equity-1)*100:+.2f}%
Std Deviation:          ${basic_results.std_dev:,.2f}

Range:                  ${basic_results.min_equity:,.2f} to ${basic_results.max_equity:,.2f}
95% Confidence:         ${basic_results.percentile_5:,.2f} to ${basic_results.percentile_95:,.2f}

Probability of Profit:  {basic_results.probability_profit*100:.1f}%

💰 VALUE AT RISK (VaR):
{'─'*65}
95% VaR:                {advanced_metrics.var_95:+.2f}% {'⚠️' if advanced_metrics.var_95 < -10 else ''}
99% VaR:                {advanced_metrics.var_99:+.2f}% {'❌' if advanced_metrics.var_99 < -20 else ''}

95% CVaR (ES):          {advanced_metrics.cvar_95:+.2f}%
99% CVaR (ES):          {advanced_metrics.cvar_99:+.2f}%

"""
        
        # VaR interpretation
        if advanced_metrics.var_95 > -5:
            report += "   ✅ Low downside risk at 95% confidence\n"
        elif advanced_metrics.var_95 > -15:
            report += "   ⚠️  Moderate downside risk\n"
        else:
            report += "   ❌ HIGH downside risk - significant loss potential\n"
        
        report += f"""
📉 DRAWDOWN ANALYSIS:
{'─'*65}
Mean Max Drawdown:      {advanced_metrics.max_drawdown_mean:.2f}%
Median Max Drawdown:    {advanced_metrics.max_drawdown_median:.2f}%
95th Percentile DD:     {advanced_metrics.max_drawdown_95:.2f}% {'❌' if advanced_metrics.max_drawdown_95 > 40 else '⚠️' if advanced_metrics.max_drawdown_95 > 25 else '✅'}
99th Percentile DD:     {advanced_metrics.max_drawdown_99:.2f}%

Avg DD Duration:        {advanced_metrics.avg_drawdown_duration:.1f} periods
Avg Recovery Time:      {advanced_metrics.avg_recovery_time:.1f} periods

95% CI for Max DD:      {advanced_metrics.max_dd_ci[0]:.2f}% to {advanced_metrics.max_dd_ci[1]:.2f}%

📈 RISK-ADJUSTED RETURNS:
{'─'*65}
Sharpe Ratio:
  Mean:                 {advanced_metrics.sharpe_mean:.3f} {'✅' if advanced_metrics.sharpe_mean > 1.0 else '⚠️' if advanced_metrics.sharpe_mean > 0.5 else '❌'}
  Median:               {advanced_metrics.sharpe_median:.3f}
  Std Dev:              {advanced_metrics.sharpe_std:.3f}
  95% CI:               {advanced_metrics.sharpe_ci[0]:.3f} to {advanced_metrics.sharpe_ci[1]:.3f}
  Range (5th-95th):     {advanced_metrics.sharpe_5th:.3f} to {advanced_metrics.sharpe_95th:.3f}

Sortino Ratio:
  Mean:                 {advanced_metrics.sortino_mean:.3f}
  Std Dev:              {advanced_metrics.sortino_std:.3f}

Calmar Ratio:
  Mean:                 {advanced_metrics.calmar_mean:.3f}
  Median:               {advanced_metrics.calmar_median:.3f}

Omega Ratio:            {advanced_metrics.omega_ratio:.3f} {'✅' if advanced_metrics.omega_ratio > 1.5 else '⚠️' if advanced_metrics.omega_ratio > 1.0 else '❌'}

📊 RETURN DISTRIBUTION:
{'─'*65}
Skewness:               {advanced_metrics.return_skewness:.3f}
"""
        
        if advanced_metrics.return_skewness > 0.5:
            report += "                        (Positively skewed - good! ✅)\n"
        elif advanced_metrics.return_skewness < -0.5:
            report += "                        (Negatively skewed - more downside ⚠️)\n"
        else:
            report += "                        (Approximately symmetric)\n"
        
        report += f"""Kurtosis:               {advanced_metrics.return_kurtosis:.3f}
"""
        
        if advanced_metrics.return_kurtosis > 3:
            report += "                        (Fat tails - extreme events likely ⚠️)\n"
        elif advanced_metrics.return_kurtosis < -1:
            report += "                        (Thin tails - fewer extremes ✅)\n"
        else:
            report += "                        (Normal tail behavior)\n"
        
        report += f"""
Skewness p-value:       {advanced_metrics.return_skew_pval:.4f}
Kurtosis p-value:       {advanced_metrics.return_kurt_pval:.4f}
"""
        
        if advanced_metrics.return_skew_pval < 0.05 or advanced_metrics.return_kurt_pval < 0.05:
            report += "   ⚠️  Returns significantly non-normal (p < 0.05)\n"
        
        report += f"""
🎲 PATH DEPENDENCY & SENSITIVITY:
{'─'*65}
Path Dependency Score:  {advanced_metrics.path_dependency_score:.3f}
"""
        
        if advanced_metrics.path_dependency_score < 0.2:
            report += "                        (Low - order doesn't matter much ✅)\n"
        elif advanced_metrics.path_dependency_score < 0.5:
            report += "                        (Moderate - some order sensitivity ⚠️)\n"
        else:
            report += "                        (High - very order-dependent ❌)\n"
        
        report += f"""
Order Sensitivity (CV): {advanced_metrics.order_sensitivity:.3f}

🎯 TAIL RISK METRICS:
{'─'*65}
Tail Ratio (95th/5th):  {advanced_metrics.tail_ratio:.3f}
"""
        
        if advanced_metrics.tail_ratio > 2.0:
            report += "                        (Favorable - upside > downside ✅)\n"
        elif advanced_metrics.tail_ratio > 1.0:
            report += "                        (Balanced risk/reward)\n"
        else:
            report += "                        (Unfavorable - downside > upside ❌)\n"
        
        report += f"""
Gain-to-Pain Ratio:     {advanced_metrics.gain_to_pain_ratio:.3f} {'✅' if advanced_metrics.gain_to_pain_ratio > 2.0 else '⚠️' if advanced_metrics.gain_to_pain_ratio > 1.0 else '❌'}

🎲 WIN RATE CONSISTENCY:
{'─'*65}
Mean Win Rate:          {advanced_metrics.win_rate_mean*100:.1f}%
Win Rate Std Dev:       {advanced_metrics.win_rate_std*100:.1f}%
95% CI:                 {advanced_metrics.win_rate_ci[0]*100:.1f}% to {advanced_metrics.win_rate_ci[1]*100:.1f}%

✅ OVERALL ASSESSMENT:
{'─'*65}
"""
        
        # Overall risk assessment
        risk_score = 0
        if advanced_metrics.sharpe_mean > 1.0:
            risk_score += 2
        elif advanced_metrics.sharpe_mean > 0.5:
            risk_score += 1
        
        if advanced_metrics.max_drawdown_mean < 20:
            risk_score += 2
        elif advanced_metrics.max_drawdown_mean < 35:
            risk_score += 1
        
        if advanced_metrics.var_95 > -10:
            risk_score += 2
        elif advanced_metrics.var_95 > -20:
            risk_score += 1
        
        if basic_results.probability_profit > 0.7:
            risk_score += 2
        elif basic_results.probability_profit > 0.5:
            risk_score += 1
        
        if advanced_metrics.path_dependency_score < 0.3:
            risk_score += 2
        elif advanced_metrics.path_dependency_score < 0.5:
            risk_score += 1
        
        if risk_score >= 8:
            report += "✅✅ EXCELLENT - Strategy appears robust across all metrics\n"
        elif risk_score >= 6:
            report += "✅ GOOD - Strategy shows positive characteristics\n"
        elif risk_score >= 4:
            report += "⚠️  MODERATE - Some concerns, proceed with caution\n"
        else:
            report += "❌ POOR - Multiple red flags, not recommended for live trading\n"
        
        report += f"\n{'═'*65}\n"
        
        return report
    
    @staticmethod
    def plot_enhanced_distributions(
        basic_results: MonteCarloResults,
        advanced_metrics: AdvancedMonteCarloMetrics,
        title: str = "Enhanced Monte Carlo Analysis"
    ):
        """Create comprehensive visualization with distribution plots"""
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#121212')
        
        # 1. Final equity distribution
        ax1 = plt.subplot(3, 3, 1)
        ax1.set_facecolor('#121212')
        
        final_equities = np.array([sim[-1] for sim in basic_results.simulations])
        
        ax1.hist(final_equities, bins=50, color='#2979ff', alpha=0.7, edgecolor='white')
        ax1.axvline(basic_results.original_equity, color='#00ff00', linestyle='--', 
                   linewidth=2, label=f'Original: ${basic_results.original_equity:.0f}')
        ax1.axvline(basic_results.mean_equity, color='#FFA500', linestyle='-', 
                   linewidth=2, label=f'Mean: ${basic_results.mean_equity:.0f}')
        ax1.axvline(basic_results.percentile_5, color='#ff4444', linestyle=':', 
                   linewidth=2, label=f'5th %ile: ${basic_results.percentile_5:.0f}')
        ax1.axvline(basic_results.percentile_95, color='#44ff44', linestyle=':', 
                   linewidth=2, label=f'95th %ile: ${basic_results.percentile_95:.0f}')
        
        ax1.set_xlabel('Final Equity ($)', color='white')
        ax1.set_ylabel('Frequency', color='white')
        ax1.set_title('Final Equity Distribution', color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.2)
        
        # 2. Drawdown distribution
        ax2 = plt.subplot(3, 3, 2)
        ax2.set_facecolor('#121212')
        
        max_drawdowns = []
        for sim in basic_results.simulations:
            peak = np.maximum.accumulate(sim)
            dd = (peak - sim) / peak * 100
            max_drawdowns.append(np.max(dd))
        
        ax2.hist(max_drawdowns, bins=50, color='#ff4444', alpha=0.7, edgecolor='white')
        ax2.axvline(advanced_metrics.max_drawdown_mean, color='#FFA500', linestyle='-', 
                   linewidth=2, label=f'Mean: {advanced_metrics.max_drawdown_mean:.1f}%')
        ax2.axvline(advanced_metrics.max_drawdown_95, color='#ff0000', linestyle='--', 
                   linewidth=2, label=f'95th: {advanced_metrics.max_drawdown_95:.1f}%')
        
        ax2.set_xlabel('Max Drawdown (%)', color='white')
        ax2.set_ylabel('Frequency', color='white')
        ax2.set_title('Maximum Drawdown Distribution', color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.2)
        
        # 3. Sharpe ratio distribution
        ax3 = plt.subplot(3, 3, 3)
        ax3.set_facecolor('#121212')
        
        sharpe_ratios = []
        for sim in basic_results.simulations:
            returns = np.diff(sim) / sim[:-1]
            returns = returns[~(np.isnan(returns) | np.isinf(returns))]
            if len(returns) > 0:
                sharpe = np.mean(returns) / (np.std(returns, ddof=1) + 1e-10) * np.sqrt(252)
                sharpe_ratios.append(np.clip(sharpe, -5, 10))
        
        ax3.hist(sharpe_ratios, bins=50, color='#00ff88', alpha=0.7, edgecolor='white')
        ax3.axvline(advanced_metrics.sharpe_mean, color='#FFA500', linestyle='-', 
                   linewidth=2, label=f'Mean: {advanced_metrics.sharpe_mean:.2f}')
        ax3.axvline(0, color='#888888', linestyle='--', linewidth=1)
        
        ax3.set_xlabel('Sharpe Ratio', color='white')
        ax3.set_ylabel('Frequency', color='white')
        ax3.set_title('Sharpe Ratio Distribution', color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.2)
        
        # 4. Sample equity paths
        ax4 = plt.subplot(3, 3, 4)
        ax4.set_facecolor('#121212')
        
        n_paths = min(100, len(basic_results.simulations))
        for i in range(n_paths):
            ax4.plot(basic_results.simulations[i], color='#2979ff', alpha=0.05, linewidth=0.5)
        
        # Overlay percentiles
        n_steps = len(basic_results.simulations[0])
        median_path = np.zeros(n_steps)
        p5_path = np.zeros(n_steps)
        p95_path = np.zeros(n_steps)
        
        for i in range(n_steps):
            values = [sim[i] for sim in basic_results.simulations]
            median_path[i] = np.median(values)
            p5_path[i] = np.percentile(values, 5)
            p95_path[i] = np.percentile(values, 95)
        
        ax4.plot(median_path, color='#FFA500', linewidth=2, label='Median')
        ax4.fill_between(range(n_steps), p5_path, p95_path, 
                        color='#FFA500', alpha=0.2, label='5-95% Range')
        ax4.axhline(y=1000, color='#888888', linestyle='--', alpha=0.5)
        
        ax4.set_xlabel('Trade Number', color='white')
        ax4.set_ylabel('Equity ($)', color='white')
        ax4.set_title('Sample Equity Paths', color='white', fontweight='bold')
        ax4.tick_params(colors='white')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.2)
        
        # 5. Return distribution with normal overlay
        ax5 = plt.subplot(3, 3, 5)
        ax5.set_facecolor('#121212')
        
        returns_pct = ((final_equities - 1000) / 1000) * 100
        
        n, bins, patches = ax5.hist(returns_pct, bins=50, density=True,
                                     color='#2979ff', alpha=0.7, edgecolor='white')
        
        # Overlay normal distribution
        mu, sigma = np.mean(returns_pct), np.std(returns_pct)
        x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
        ax5.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2, 
                label=f'Normal({mu:.1f}, {sigma:.1f})')
        
        ax5.set_xlabel('Return (%)', color='white')
        ax5.set_ylabel('Density', color='white')
        ax5.set_title('Return Distribution vs Normal', color='white', fontweight='bold')
        ax5.tick_params(colors='white')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.2)
        
        # 6. Q-Q plot for normality
        ax6 = plt.subplot(3, 3, 6)
        ax6.set_facecolor('#121212')
        
        stats.probplot(returns_pct, dist="norm", plot=ax6)
        ax6.get_lines()[0].set_color('#2979ff')
        ax6.get_lines()[0].set_markersize(3)
        ax6.get_lines()[1].set_color('#ff4444')
        ax6.get_lines()[1].set_linewidth(2)
        
        ax6.set_xlabel('Theoretical Quantiles', color='white')
        ax6.set_ylabel('Sample Quantiles', color='white')
        ax6.set_title('Q-Q Plot (Normality Test)', color='white', fontweight='bold')
        ax6.tick_params(colors='white')
        ax6.grid(True, alpha=0.2)
        
        # 7. Win rate distribution
        ax7 = plt.subplot(3, 3, 7)
        ax7.set_facecolor('#121212')
        
        win_rates = []
        for sim in basic_results.simulations:
            returns = np.diff(sim) / sim[:-1]
            returns = returns[~(np.isnan(returns) | np.isinf(returns))]
            if len(returns) > 0:
                wr = np.sum(returns > 0) / len(returns)
                win_rates.append(wr * 100)
        
        ax7.hist(win_rates, bins=30, color='#00ff88', alpha=0.7, edgecolor='white')
        ax7.axvline(advanced_metrics.win_rate_mean * 100, color='#FFA500', 
                   linestyle='-', linewidth=2, label=f'Mean: {advanced_metrics.win_rate_mean*100:.1f}%')
        ax7.axvline(50, color='#888888', linestyle='--', linewidth=1, label='50%')
        
        ax7.set_xlabel('Win Rate (%)', color='white')
        ax7.set_ylabel('Frequency', color='white')
        ax7.set_title('Win Rate Distribution', color='white', fontweight='bold')
        ax7.tick_params(colors='white')
        ax7.legend(loc='upper right', fontsize=8)
        ax7.grid(True, alpha=0.2)
        
        # 8. Calmar ratio distribution
        ax8 = plt.subplot(3, 3, 8)
        ax8.set_facecolor('#121212')
        
        calmar_ratios = []
        for sim in basic_results.simulations:
            total_return = (sim[-1] / 1000 - 1)
            peak = np.maximum.accumulate(sim)
            dd = (peak - sim) / peak
            max_dd = np.max(dd)
            if max_dd > 0:
                calmar = total_return / max_dd
                if -10 < calmar < 10:  # Filter outliers
                    calmar_ratios.append(calmar)
        
        if calmar_ratios:
            ax8.hist(calmar_ratios, bins=50, color='#9c27b0', alpha=0.7, edgecolor='white')
            ax8.axvline(advanced_metrics.calmar_mean, color='#FFA500', linestyle='-', 
                       linewidth=2, label=f'Mean: {advanced_metrics.calmar_mean:.2f}')
            ax8.axvline(0, color='#888888', linestyle='--', linewidth=1)
        
        ax8.set_xlabel('Calmar Ratio', color='white')
        ax8.set_ylabel('Frequency', color='white')
        ax8.set_title('Calmar Ratio Distribution', color='white', fontweight='bold')
        ax8.tick_params(colors='white')
        if calmar_ratios:
            ax8.legend(loc='upper right', fontsize=8)
        ax8.grid(True, alpha=0.2)
        
        # 9. Summary metrics box
        ax9 = plt.subplot(3, 3, 9)
        ax9.set_facecolor('#121212')
        ax9.axis('off')
        
        summary = f"""
╔════════════════════════════════════╗
║   KEY METRICS SUMMARY              ║
╚════════════════════════════════════╝

📊 Returns:
   Mean: {(basic_results.mean_equity/1000-1)*100:+.1f}%
   VaR 95%: {advanced_metrics.var_95:+.1f}%
   
📉 Risk:
   Avg Max DD: {advanced_metrics.max_drawdown_mean:.1f}%
   95th DD: {advanced_metrics.max_drawdown_95:.1f}%
   
📈 Risk-Adjusted:
   Sharpe: {advanced_metrics.sharpe_mean:.2f}
   Sortino: {advanced_metrics.sortino_mean:.2f}
   Omega: {advanced_metrics.omega_ratio:.2f}
   
🎲 Distribution:
   Skew: {advanced_metrics.return_skewness:.2f}
   Kurt: {advanced_metrics.return_kurtosis:.2f}
   
🎯 Robustness:
   Path Dep: {advanced_metrics.path_dependency_score:.2f}
   Win Rate: {advanced_metrics.win_rate_mean*100:.1f}%
   Prob Profit: {basic_results.probability_profit*100:.1f}%
"""
        
        ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                color='white')
        
        plt.suptitle(title, color='white', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        return fig


# Keep the original MonteCarloSimulator for backward compatibility
class MonteCarloSimulator:
    """
    Original Monte Carlo simulator - kept for backward compatibility
    Enhanced version available through AdvancedMonteCarloAnalyzer
    """
    
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
        
        # Diagnostic output
        print(f"\n{'='*60}")
        print(f"MONTE CARLO SIMULATION SETUP")
        print(f"{'='*60}")
        print(f"Total trades: {len(returns)}")
        print(f"Returns range: [{returns.min()*100:.2f}%, {returns.max()*100:.2f}%]")
        print(f"Returns mean: {returns.mean()*100:.2f}%")
        print(f"Returns std: {returns.std()*100:.2f}%")
        print(f"Unique returns: {len(np.unique(returns))} ({len(np.unique(returns))/len(returns)*100:.1f}%)")
        
        if len(np.unique(returns)) == 1:
            print(f"\n⚠️  WARNING: All returns are identical ({returns[0]*100:.2f}%)")
            print(f"Monte Carlo will show no variation.")
        
        # Show first few returns
        print(f"\nFirst 10 returns:")
        for i, ret in enumerate(returns[:10]):
            print(f"  Trade {i+1}: {ret*100:+.2f}%")
        
        # Original equity curve
        original_equity_curve = MonteCarloSimulator._calculate_equity_curve(
            returns, initial_equity
        )
        original_final = original_equity_curve[-1]
        
        print(f"\nOriginal sequence: ${original_final:,.2f} ({(original_final/initial_equity-1)*100:+.2f}%)")
        
        # Run simulations
        final_equities = []
        all_curves = []
        
        print(f"\nRunning {n_simulations} simulations...")
        
        for sim_idx in range(n_simulations):
            sampled_returns = np.random.permutation(returns)
            eq_curve = MonteCarloSimulator._calculate_equity_curve(
                sampled_returns, initial_equity
            )
            
            final_equities.append(eq_curve[-1])
            all_curves.append(eq_curve)
            
            if sim_idx in [0, 1, 2, 99, 499, 999]:
                print(f"  Sim {sim_idx+1}: ${eq_curve[-1]:,.2f}")
        
        final_equities = np.array(final_equities)
        
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
    def _calculate_equity_curve(returns: np.ndarray, initial_equity: float) -> np.ndarray:
        """Calculate equity curve from returns"""
        equity_curve = np.zeros(len(returns) + 1)
        equity_curve[0] = initial_equity
        
        for i, ret in enumerate(returns):
            equity_curve[i + 1] = equity_curve[i] * (1 + ret)
        
        return equity_curve
    
    @staticmethod
    def generate_monte_carlo_report(results: MonteCarloResults, initial_equity: float = 1000.0) -> str:
        """Generate basic Monte Carlo report (kept for compatibility)"""
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


# Example usage
if __name__ == "__main__":
    print("Enhanced Monte Carlo Analysis Module")
    print("=====================================")
    print("\nThis module provides comprehensive quantitative finance metrics")
    print("for Monte Carlo simulation analysis, including:")
    print("\n✓ Value at Risk (VaR) and Conditional VaR")
    print("✓ Drawdown analysis and recovery metrics")
    print("✓ Risk-adjusted returns (Sharpe, Sortino, Calmar, Omega)")
    print("✓ Distribution analysis (skewness, kurtosis, normality tests)")
    print("✓ Path dependency and order sensitivity")
    print("✓ Confidence intervals for all metrics")
    print("✓ Win rate stability analysis")
    print("✓ Comprehensive reporting and visualization")
    print("\nUse AdvancedMonteCarloAnalyzer.calculate_advanced_metrics()")
    print("to enhance your Monte Carlo results with professional metrics.")
