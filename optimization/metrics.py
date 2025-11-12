"""
Performance Metrics Calculation
"""
from typing import Optional, Dict
import numpy as np


class PerformanceMetrics:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_metrics(eq_curve: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            eq_curve: Equity curve as numpy array
            
        Returns:
            Dictionary of metrics or None if invalid
        """
        if eq_curve is None or len(eq_curve) == 0:
            return None
        
        final_equity = eq_curve[-1]
        if final_equity <= 0 or np.isnan(final_equity) or np.isinf(final_equity):
            return None
        
        # Percent gain
        initial_equity = 1000.0
        percent_gain = (final_equity - initial_equity) / initial_equity * 100
        
        # Returns calculation
        returns = np.diff(eq_curve) / eq_curve[:-1]
        valid_mask = ~(np.isnan(returns) | np.isinf(returns))
        returns = returns[valid_mask]
        
        if len(returns) == 0:
            return None
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-10
        avg_return = np.mean(returns)
        sortino = (avg_return / downside_std) * np.sqrt(252 * 390) if downside_std > 1e-10 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(eq_curve)
        drawdown = (eq_curve - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Profit factor
        equity_changes = np.diff(eq_curve)
        gross_profit = np.sum(equity_changes[equity_changes > 0])
        gross_loss = np.abs(np.sum(equity_changes[equity_changes < 0]))
        profit_factor = (
            gross_profit / gross_loss 
            if gross_loss > 0 
            else (gross_profit * 1000 if gross_profit > 0 else 1.0)
        )
        
        return {
            "Final_Equity_$": round(final_equity, 2),
            "Percent_Gain_%": round(percent_gain, 2),
            "Sortino_Ratio": round(sortino, 3),
            "Max_Drawdown_%": round(max_drawdown, 2),
            "Profit_Factor": round(profit_factor, 3)
        }
    
    @staticmethod
    def compute_rsi_vectorized(prices: np.ndarray, length: int) -> np.ndarray:
        """
        Vectorized RSI computation (3x faster than pandas rolling)
        
        Args:
            prices: Array of prices
            length: RSI period
            
        Returns:
            RSI values as numpy array
        """
        length = max(1, int(length))
        delta = np.diff(prices, prepend=prices[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        
        # Use cumsum for faster rolling mean
        gain_sum = np.cumsum(gain)
        loss_sum = np.cumsum(loss)
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        # Rolling window
        avg_gain[length-1:] = (
            gain_sum[length-1:] - np.concatenate([[0], gain_sum[:-length]])
        ) / length
        avg_loss[length-1:] = (
            loss_sum[length-1:] - np.concatenate([[0], loss_sum[:-length]])
        ) / length
        
        # First window
        avg_gain[:length] = gain_sum[:length] / np.arange(1, length+1)
        avg_loss[:length] = loss_sum[:length] / np.arange(1, length+1)
        
        # Calculate RSI
        rs = np.divide(
            avg_gain, avg_loss, 
            out=np.zeros_like(avg_gain), 
            where=avg_loss != 0
        )
        rsi = 100 - 100 / (1 + rs)
        
        return rsi
    
    @staticmethod
    def smooth_vectorized(arr: np.ndarray, length: int) -> np.ndarray:
        """
        Vectorized smoothing (faster than pandas rolling)
        
        Args:
            arr: Array to smooth
            length: Smoothing period
            
        Returns:
            Smoothed array
        """
        length = max(1, int(length))
        cumsum = np.cumsum(arr)
        result = np.zeros_like(arr)
        
        # Rolling window
        result[length-1:] = (
            cumsum[length-1:] - np.concatenate([[0], cumsum[:-length]])
        ) / length
        
        # First window
        result[:length] = cumsum[:length] / np.arange(1, length+1)
        
        return result
    
    @staticmethod
    def calculate_buyhold_return(prices: np.ndarray) -> float:
        """Calculate buy and hold return percentage"""
        if len(prices) < 2:
            return 0.0
        return (prices[-1] / prices[0] - 1) * 100
