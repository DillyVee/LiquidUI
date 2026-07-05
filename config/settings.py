"""
Configuration Settings - Updated with Organized Data Folders
"""

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================
@dataclass
class Paths:
    """Organized directory structure for generated output files"""

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data_output"

    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_results_path(cls, ticker: str, suffix: str = "") -> Path:
        """Get path for optimization results CSV"""
        filename = f"{ticker}_results{suffix}.csv"
        return cls.DATA_DIR / filename


# Initialize directories on module import
Paths.ensure_directories()


# ============================================================
# OPTIMIZATION CONFIG
# ============================================================
@dataclass
class OptimizationConfig:
    """Optimization configuration"""

    DEFAULT_TRIALS = 2000
    DEFAULT_BATCH_SIZE = 500
    MIN_TRADES = 10

    # Data limits to prevent overload
    FIVEMIN_MAX_DAYS = 30
    HOURLY_MAX_DAYS = 180

    # Equity curve retracement zones
    RETRACEMENT_ZONES = [
        (0.00, 0.05),  # 0-5% retracement
        (0.05, 0.15),  # 5-15% retracement
        (0.15, 0.30),  # 15-30% retracement
        (0.30, 0.50),  # 30-50% retracement
        (0.50, 1.00),  # 50%+ retracement
    ]


# For backward compatibility
RETRACEMENT_ZONES = OptimizationConfig.RETRACEMENT_ZONES


# ============================================================
# RISK MANAGEMENT
# ============================================================
@dataclass
class RiskConfig:
    """Risk management settings"""

    DEFAULT_POSITION_SIZE = 0.05  # 5% of account per trade
    DEFAULT_MAX_POSITIONS = 1
    MAX_LEVERAGE = 1.0


# ============================================================
# TRANSACTION COSTS
# ============================================================
class TransactionCosts:
    """Transaction cost configuration"""

    def __init__(self):
        # Percentage-based costs (as decimals, e.g., 0.001 = 0.1%)
        self.COMMISSION_PCT = 0.0
        self.SLIPPAGE_PCT = 0.0
        self.SPREAD_PCT = 0.0

        # Fixed costs per trade
        self.COMMISSION_FIXED = 0.0

    @property
    def TOTAL_PCT(self) -> float:
        """Total percentage cost per trade"""
        return self.COMMISSION_PCT + self.SLIPPAGE_PCT + self.SPREAD_PCT

    @classmethod
    def for_stocks(cls):
        """Typical costs for US stocks"""
        costs = cls()
        costs.COMMISSION_PCT = 0.0
        costs.SLIPPAGE_PCT = 0.0002
        costs.SPREAD_PCT = 0.0004
        return costs

    @classmethod
    def for_crypto(cls):
        """Typical costs for cryptocurrency"""
        costs = cls()
        costs.COMMISSION_PCT = 0.0025
        costs.SLIPPAGE_PCT = 0.0005
        costs.SPREAD_PCT = 0.0020
        return costs


# ============================================================
# INDICATOR RANGES
# ============================================================
@dataclass
class IndicatorRanges:
    """Parameter ranges for optimization"""

    MN1_RANGE = (5, 100)
    MN2_RANGE = (3, 50)
    ENTRY_RANGE = (10.0, 40.0)
    EXIT_RANGE = (50.0, 90.0)
    ON_RANGE = (1, 250)
    OFF_RANGE = (0, 250)


# ============================================================
# ALPACA CONFIG
# ============================================================
@dataclass
class AlpacaConfig:
    """Alpaca API configuration"""

    # Paper trading endpoint
    BASE_URL = "https://paper-api.alpaca.markets"

    # Your API credentials (KEEP THESE SECRET!)
    API_KEY = "your_api_key_here"
    SECRET_KEY = "your_secret_key_here"

    # Ticker mapping
    TICKER_MAP = {
        "BTC-USD": "BTC/USD",
        "ETH-USD": "ETH/USD",
        "DOGE-USD": "DOGE/USD",
        "SOL-USD": "SOL/USD",
        "AVAX-USD": "AVAX/USD",
    }

    @classmethod
    def get_alpaca_symbol(cls, yfinance_symbol: str) -> str:
        """Convert yfinance symbol to Alpaca format"""
        return cls.TICKER_MAP.get(yfinance_symbol, yfinance_symbol)
