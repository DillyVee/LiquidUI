"""
Configuration and Constants
"""
import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class OptimizationConfig:
    """Optimization parameters"""
    DEFAULT_TRIALS: int = 900
    DEFAULT_BATCH_SIZE: int = 500
    MIN_BARS_FOR_OPTIMIZATION: int = 200
    
    # Timeframe data limits
    DAILY_MAX_DAYS: int = 365 * 10  # 10 years
    HOURLY_MAX_DAYS: int = 730       # ~2 years
    FIVEMIN_MAX_DAYS: int = 60       # ~2 months


@dataclass
class RiskConfig:
    """Risk management settings"""
    DEFAULT_POSITION_SIZE: float = 0.05  # 5%
    DEFAULT_MAX_POSITIONS: int = 20
    DEFAULT_STOP_LOSS: float = 0.02      # 2%
    MAX_DRAWDOWN_THRESHOLD: float = 0.50  # 50%


@dataclass
class TransactionCosts:
    """Transaction cost settings for realistic backtesting"""
    # Commission per trade (percentage of trade value)
    COMMISSION_PCT: float = 0.001  # 0.1% = $10 per $10k trade
    
    # Alternative: Fixed commission per trade (set to 0 if using percentage)
    COMMISSION_FIXED: float = 0.0  # e.g., $1.00 per trade
    
    # Slippage (percentage of price - market impact)
    SLIPPAGE_PCT: float = 0.0005  # 0.05% = ~0.5 cents per $10 stock
    
    # Bid-ask spread (percentage of price)
    SPREAD_PCT: float = 0.0001  # 0.01% = 1 basis point
    
    # Total cost per trade (convenience method)
    @property
    def TOTAL_PCT(self) -> float:
        """Total percentage cost per trade (one-way)"""
        return self.COMMISSION_PCT + self.SLIPPAGE_PCT + self.SPREAD_PCT
    
    # Realistic presets for different asset types
    @classmethod
    def for_stocks(cls):
        """Transaction costs for US stocks"""
        return cls(
            COMMISSION_PCT=0.0,      # Many brokers now commission-free
            COMMISSION_FIXED=0.0,     # Zero commission
            SLIPPAGE_PCT=0.0005,     # 0.05% slippage
            SPREAD_PCT=0.0001        # 0.01% spread
        )
    
    @classmethod
    def for_crypto(cls):
        """Transaction costs for cryptocurrency"""
        return cls(
            COMMISSION_PCT=0.001,     # 0.1% taker fee (Coinbase/Binance)
            COMMISSION_FIXED=0.0,
            SLIPPAGE_PCT=0.002,       # 0.2% slippage (more volatile)
            SPREAD_PCT=0.0005         # 0.05% spread
        )
    
    @classmethod
    def for_forex(cls):
        """Transaction costs for forex"""
        return cls(
            COMMISSION_PCT=0.0,
            COMMISSION_FIXED=0.0,
            SLIPPAGE_PCT=0.0001,      # 0.01% slippage
            SPREAD_PCT=0.0002         # 0.02% spread (2 pips typical)
        )


@dataclass
class AlpacaConfig:
    """Alpaca API configuration"""
    # SECURITY: Load from environment variables
    API_KEY: str = os.environ.get('ALPACA_API_KEY', '')
    SECRET_KEY: str = os.environ.get('ALPACA_SECRET_KEY', '')
    BASE_URL: str = "https://paper-api.alpaca.markets"
    
    # Ticker mapping: yfinance -> Alpaca format
    TICKER_MAP = {
        # Crypto - Alpaca uses slash format
        'BTC-USD': 'BTC/USD',
        'ETH-USD': 'ETH/USD',
        'BTCUSD': 'BTC/USD',
        'ETHUSD': 'ETH/USD',
        'SOL-USD': 'SOL/USD',
        'AVAX-USD': 'AVAX/USD',
        'DOGE-USD': 'DOGE/USD',
        'LTC-USD': 'LTC/USD',
        'BCH-USD': 'BCH/USD',
        'LINK-USD': 'LINK/USD',
        'UNI-USD': 'UNI/USD',
    }
    
    @staticmethod
    def get_alpaca_symbol(yfinance_symbol: str) -> str:
        """Convert yfinance symbol to Alpaca format"""
        return AlpacaConfig.TICKER_MAP.get(yfinance_symbol, yfinance_symbol)

@dataclass
class IndicatorRanges:
    MN1_RANGE: Tuple[int, int] = (5, 50)   # ✅ Not 2-100!
    MN2_RANGE: Tuple[int, int] = (1, 20)   # ✅ Not 2-100!
    ENTRY_RANGE: Tuple[float, float] = (25.0, 45.0)
    EXIT_RANGE: Tuple[float, float] = (55.0, 75.0)
    ON_RANGE: Tuple[int, int] = (1, 200)   # ✅ Much wider range
    OFF_RANGE: Tuple[int, int] = (0, 100)  # ✅ Much wider range


class Paths:
    """File paths"""
    TICKERS_FILE = "config/large_cap_us_stocks.txt"
    LEADERBOARD_FILE = "Batch_Leaderboard.csv"


# UI Constants
DARK_THEME_STYLESHEET = """
    QMainWindow { background-color: #121212; color: #fff; }
    QLabel { color: #ddd; font-size: 11pt; }
    QPushButton { 
        background-color: #1e1e1e; color: #fff; 
        border-radius: 6px; padding: 6px 12px;
        font-size: 10pt;
    }
    QPushButton:hover { background-color: #2a2a2a; }
    QPushButton:disabled { background-color: #0a0a0a; color: #555; }
    QSpinBox, QDoubleSpinBox, QLineEdit { 
        background-color: #1e1e1e; color: #fff; 
        border-radius: 4px; padding: 4px;
        font-size: 10pt;
    }
    QProgressBar { 
        border: 1px solid #555; border-radius: 5px; 
        text-align: center; background: #1e1e1e; color: #fff;
    }
    QProgressBar::chunk { background-color: #2979ff; }
    QCheckBox, QRadioButton { color: #ddd; font-size: 10pt; }
    QComboBox {
        background-color: #1e1e1e; color: #fff;
        border-radius: 4px; padding: 4px;
        font-size: 10pt;
    }
    QComboBox:hover { background-color: #2a2a2a; }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #1e1e1e; color: #fff;
        selection-background-color: #2979ff;
    }
    QGroupBox { 
        color: #ddd; font-weight: bold;
        border: 1px solid #555; border-radius: 4px;
        margin-top: 6px; padding-top: 6px;
    }
"""

RETRACEMENT_ZONES = [
    (0.059, 0.0625),   # ~6%
    (0.114, 0.125),    # ~12%
    (0.214, 0.25),     # ~23%
    (0.33333, 0.382),  # ~36%
    (0.618, 0.66666),  # ~64%
    (0.75, 0.786),     # ~77%
    (0.875, 0.886),    # ~88%
    (0.9375, 0.941)    # ~94%
]
