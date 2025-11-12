"""
Multi-Timeframe Trading Optimizer with Alpaca Live Trading

A comprehensive algorithmic trading application featuring:
- Multi-timeframe strategy optimization
- Walk-forward analysis for overfitting detection
- Equity curve optimization
- Live paper trading with Alpaca
- Risk management controls
"""

__version__ = "1.0.0"
__author__ = "Trading App Team"

from gui import MainWindow
from optimization import MultiTimeframeOptimizer, PerformanceMetrics
from data import DataLoader
from trading import AlpacaLiveTrader

__all__ = [
    'MainWindow',
    'MultiTimeframeOptimizer',
    'PerformanceMetrics',
    'DataLoader',
    'AlpacaLiveTrader'
]
