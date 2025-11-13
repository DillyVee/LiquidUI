"""
Configuration Settings - Updated with Organized Data Folders
"""
from pathlib import Path
from dataclasses import dataclass

# ============================================================
# PROJECT PATHS
# ============================================================
@dataclass
class Paths:
    """Organized directory structure for all project files"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data_output"
    PLOTS_DIR = BASE_DIR / "plots"
    LOGS_DIR = BASE_DIR / "logs"
    OPTUNA_DIR = BASE_DIR / "optuna_studies"
    
    # Create directories on import
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories if they don't exist"""
        for attr_name in dir(cls):
            if attr_name.endswith('_DIR'):
                directory = getattr(cls, attr_name)
                directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_results_path(cls, ticker: str, suffix: str = "") -> Path:
        """Get path for optimization results CSV"""
        filename = f"{ticker}_results{suffix}.csv"
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_plot_path(cls, ticker: str, plot_type: str) -> Path:
        """Get path for plots"""
        filename = f"{ticker}_{plot_type}.png"
        return cls.PLOTS_DIR / filename
    
    @classmethod
    def get_optuna_path(cls, ticker: str) -> str:
        """Get path for Optuna database"""
        db_path = cls.OPTUNA_DIR / f"optuna_{ticker}.db"
        return f"sqlite:///{db_path}"
    
    @classmethod
    def get_log_path(cls, log_name: str) -> Path:
        """Get path for log files"""
        return cls.LOGS_DIR / f"{log_name}.log"


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
        'BTC-USD': 'BTC/USD',
        'ETH-USD': 'ETH/USD',
        'DOGE-USD': 'DOGE/USD',
        'SOL-USD': 'SOL/USD',
        'AVAX-USD': 'AVAX/USD',
    }
    
    @classmethod
    def get_alpaca_symbol(cls, yfinance_symbol: str) -> str:
        """Convert yfinance symbol to Alpaca format"""
        return cls.TICKER_MAP.get(yfinance_symbol, yfinance_symbol)


"""
Modern Professional Dark Theme for Trading Optimizer
Replace the DARK_THEME_STYLESHEET in config/settings.py with this
"""

DARK_THEME_STYLESHEET = """
/* ========================================
   MODERN PROFESSIONAL DARK THEME
   ======================================== */

/* Main Window */
QMainWindow {
    background-color: #0d1117;
    color: #c9d1d9;
}

QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
}

/* Labels */
QLabel {
    color: #c9d1d9;
    background-color: transparent;
    padding: 2px;
}

/* Buttons - Modern flat design with hover effects */
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 28px;
}

QPushButton:hover {
    background-color: #30363d;
    border: 1px solid #58a6ff;
}

QPushButton:pressed {
    background-color: #161b22;
    border: 1px solid #58a6ff;
}

QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border: 1px solid #21262d;
}

/* Special Button Styles */
QPushButton#startButton {
    background-color: #238636;
    border: 1px solid #2ea043;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#startButton:hover {
    background-color: #2ea043;
    border: 1px solid #3fb950;
}

QPushButton#stopButton {
    background-color: #da3633;
    border: 1px solid #f85149;
    color: #ffffff;
}

QPushButton#stopButton:hover {
    background-color: #f85149;
    border: 1px solid #ff7b72;
}

/* Input Fields - Clean modern style */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: #1f6feb;
    min-height: 26px;
}

QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border: 1px solid #58a6ff;
}

QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 1px solid #58a6ff;
    background-color: #161b22;
}

/* Spin Box Arrows */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #21262d;
    border: none;
    border-top-right-radius: 6px;
    width: 20px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #30363d;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #21262d;
    border: none;
    border-bottom-right-radius: 6px;
    width: 20px;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #30363d;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 6px solid #8b949e;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #8b949e;
}

/* ComboBox - Dropdown style */
QComboBox {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 26px;
}

QComboBox:hover {
    border: 1px solid #58a6ff;
}

QComboBox:focus {
    border: 1px solid #58a6ff;
    background-color: #161b22;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid #8b949e;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    border-radius: 4px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #21262d;
}

/* Progress Bar - Modern sleek design */
QProgressBar {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    text-align: center;
    color: #c9d1d9;
    font-weight: 600;
    height: 24px;
}

QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb,
        stop:1 #58a6ff
    );
    border-radius: 7px;
}

/* Group Box - Card-style containers */
QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: 600;
    color: #58a6ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 4px 8px;
    background-color: #161b22;
    border-radius: 4px;
}

/* Checkboxes - Modern toggle style */
QCheckBox {
    color: #c9d1d9;
    spacing: 8px;
    padding: 4px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border: 2px solid #30363d;
    border-radius: 4px;
    background-color: #0d1117;
}

QCheckBox::indicator:hover {
    border: 2px solid #58a6ff;
}

QCheckBox::indicator:checked {
    background-color: #1f6feb;
    border: 2px solid #1f6feb;
    image: none;
}

QCheckBox::indicator:checked:after {
    content: "✓";
    color: white;
    font-size: 14px;
    font-weight: bold;
}

QCheckBox::indicator:disabled {
    background-color: #161b22;
    border: 2px solid #21262d;
}

/* Scrollbars - Minimal modern design */
QScrollBar:vertical {
    background-color: #0d1117;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #0d1117;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 6px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #484f58;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Tooltips */
QToolTip {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
}

/* Menu Bar (if you add one later) */
QMenuBar {
    background-color: #0d1117;
    color: #c9d1d9;
    border-bottom: 1px solid #21262d;
    padding: 4px;
}

QMenuBar::item {
    padding: 6px 12px;
    border-radius: 6px;
}

QMenuBar::item:selected {
    background-color: #21262d;
}

QMenu {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 6px;
}

QMenu::item:selected {
    background-color: #21262d;
}

/* Status Bar (if you add one) */
QStatusBar {
    background-color: #0d1117;
    color: #8b949e;
    border-top: 1px solid #21262d;
}

/* Tab Widget (if you add tabs) */
QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 8px;
    background-color: #161b22;
    top: -1px;
}

QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #0d1117;
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}

QTabBar::tab:hover:!selected {
    background-color: #21262d;
}

/* Table Widget (if you add tables) */
QTableWidget {
    background-color: #0d1117;
    alternate-background-color: #161b22;
    gridline-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    selection-background-color: #1f6feb;
}

QTableWidget::item {
    padding: 6px;
    border: none;
}

QHeaderView::section {
    background-color: #161b22;
    color: #c9d1d9;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #30363d;
    font-weight: 600;
}
"""

# Also update the specific button styles in gui/styles.py:
LIVE_TRADING_BUTTON_ACTIVE = """
    QPushButton { 
        background-color: #238636;
        border: 1px solid #2ea043;
        color: #ffffff;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 16px;
    }
    QPushButton:hover { 
        background-color: #2ea043;
        border: 1px solid #3fb950;
    }
    QPushButton:disabled { 
        background-color: #161b22; 
        color: #484f58;
        border: 1px solid #21262d;
    }
"""

LIVE_TRADING_BUTTON_STOPPED = """
    QPushButton { 
        background-color: #da3633;
        border: 1px solid #f85149;
        color: #ffffff;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 16px;
    }
    QPushButton:hover { 
        background-color: #f85149;
        border: 1px solid #ff7b72;
    }
"""

# Color constants (updated for modern theme)
COLOR_SUCCESS = "#3fb950"    # GitHub green
COLOR_ERROR = "#f85149"      # GitHub red
COLOR_WARNING = "#d29922"    # GitHub yellow
COLOR_PRIMARY = "#58a6ff"    # GitHub blue
COLOR_BACKGROUND = "#0d1117" # GitHub dark background
COLOR_TEXT = "#c9d1d9"       # GitHub text