"""
Professional Multi-Tab GUI for LiquidUI Trading Platform
Institutional-Grade UX/UI Design
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QSplitter,
)

# Import existing components
from config.settings import (
    AlpacaConfig,
    IndicatorRanges,
    OptimizationConfig,
    Paths,
    RiskConfig,
    TransactionCosts,
)
from data import DataLoader
from gui.styles import (
    MAIN_STYLESHEET,
    COLOR_SUCCESS,
    COLOR_WARNING,
    LIVE_TRADING_BUTTON_ACTIVE,
    LIVE_TRADING_BUTTON_STOPPED,
)

# Regime analysis imports
from models.regime_detection import (
    MarketRegime,
    MarketRegimeDetector,
    PBRCalculator,
    RegimeState,
)
from models.regime_predictor import RegimePredictor, RegimeBasedPositionSizer
from models.regime_calibration import MultiClassCalibrator
from models.regime_cross_asset import CrossAssetRegimeAnalyzer, load_multi_asset_data
from models.regime_diagnostics import RegimeDiagnosticAnalyzer
from models.regime_agreement import HorizonPrediction, MultiHorizonAgreementIndex
from models.regime_robustness import (
    WhiteRealityCheck,
    HansenSPATest,
    BlockBootstrapValidator,
    run_full_robustness_suite,
)

# Optimization imports
from optimization import MultiTimeframeOptimizer, PerformanceMetrics
from optimization.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResults,
    AdvancedMonteCarloAnalyzer,
    AdvancedMonteCarloMetrics,
)
from optimization.walk_forward import WalkForwardAnalyzer, WalkForwardResults

# Trading imports
from trading import ALPACA_AVAILABLE, AlpacaLiveTrader


# Worker thread for optimization
class OptimizationWorker(QThread):
    """Background worker for optimization"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, optimizer, df_dict, ticker, parent=None):
        super().__init__(parent)
        self.optimizer = optimizer
        self.df_dict = df_dict
        self.ticker = ticker
        self.stopped = False

    def run(self):
        try:
            self.progress.emit("Starting optimization...")
            results = self.optimizer.optimize(self.df_dict)
            if not self.stopped:
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Professional Multi-Tab Trading Platform Interface

    Tabs:
    1. Data & Setup - Load and configure data
    2. Strategy Builder - Optimization and backtesting
    3. Regime Analysis - Institutional-grade regime detection
    4. Live Trading - Real-time trading and monitoring
    5. Settings - Risk management and configuration
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidUI - Professional Trading Platform")
        self.setMinimumSize(1400, 900)  # Larger for professional feel

        # Initialize data structures
        self.df_dict_full = None
        self.current_ticker = None
        self.best_params = None
        self.best_results = None

        # Regime analysis
        self.regime_detector = MarketRegimeDetector()
        self.regime_predictor = None
        self.regime_calibrator = None
        self.cross_asset_analyzer = None

        # Trading and workers
        self.live_trader = None
        self.worker = None

        # Optimizers
        self.optimizer = None
        self.monte_carlo_results = None
        self.walk_forward_results = None

        # Apply professional stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

        # Build UI
        self.init_professional_ui()

    def init_professional_ui(self):
        """Initialize professional tabbed interface"""

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Add professional header
        self._add_professional_header(main_layout)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # Professional flat look
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Style tabs professionally
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2d2d2d;
                background: #1a1a1a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #2d2d2d;
                color: #cccccc;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #1a1a1a;
                color: #00ff88;
                border-bottom: 2px solid #00ff88;
            }
            QTabBar::tab:hover:!selected {
                background: #3a3a3a;
                color: #ffffff;
            }
        """)

        # Create all tabs
        self.tabs.addTab(self._create_data_tab(), "📊 Data & Setup")
        self.tabs.addTab(self._create_strategy_tab(), "⚙️ Strategy Builder")
        self.tabs.addTab(self._create_regime_tab(), "🏛️ Regime Analysis")
        self.tabs.addTab(self._create_trading_tab(), "🔴 Live Trading")
        self.tabs.addTab(self._create_settings_tab(), "⚙️ Settings")

        main_layout.addWidget(self.tabs)

        # Add status bar at bottom
        self._add_professional_statusbar()

    def _add_professional_header(self, layout):
        """Add professional header with branding"""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 10)

        # Logo/Title
        title_label = QLabel("⚡ LiquidUI")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #00ff88;
            padding: 8px;
        """)
        header_layout.addWidget(title_label)

        # Subtitle
        subtitle = QLabel("Institutional-Grade Quantitative Trading Platform")
        subtitle.setStyleSheet("""
            font-size: 12px;
            color: #888888;
            padding: 8px;
        """)
        header_layout.addWidget(subtitle)

        header_layout.addStretch()

        # Connection status indicator
        self.connection_indicator = QLabel("● Disconnected")
        self.connection_indicator.setStyleSheet("""
            color: #ff4444;
            font-size: 11px;
            padding: 8px;
        """)
        header_layout.addWidget(self.connection_indicator)

        layout.addWidget(header)

        # Add separator
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background: #2d2d2d;")
        layout.addWidget(separator)

    def _create_data_tab(self):
        """Tab 1: Data & Setup"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Welcome message for first-time users
        welcome = QGroupBox("👋 Welcome")
        welcome_layout = QVBoxLayout()
        welcome_text = QLabel(
            "Start by loading market data for the asset you want to analyze.\n"
            "You can load individual tickers or batch process multiple symbols."
        )
        welcome_text.setWordWrap(True)
        welcome_text.setStyleSheet("color: #cccccc; padding: 10px;")
        welcome_layout.addWidget(welcome_text)
        welcome.setLayout(welcome_layout)
        layout.addWidget(welcome)

        # Data loading section
        data_group = QGroupBox("📥 Data Loading")
        data_layout = QVBoxLayout()

        # Ticker input row
        ticker_row = QHBoxLayout()
        ticker_label = QLabel("Ticker Symbol:")
        ticker_label.setFixedWidth(100)
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("e.g., AAPL, SPY, BTC-USD...")
        self.ticker_input.setMinimumWidth(200)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background: #00ff88;
                color: #000000;
                font-weight: bold;
                padding: 10px 30px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #00dd77;
            }
        """)
        self.load_btn.clicked.connect(self.load_data)

        ticker_row.addWidget(ticker_label)
        ticker_row.addWidget(self.ticker_input)
        ticker_row.addWidget(self.load_btn)
        ticker_row.addStretch()

        data_layout.addLayout(ticker_row)

        # Batch loading button
        self.batch_btn = QPushButton("📋 Load Multiple Tickers")
        self.batch_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #ffffff;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #5a5a5a;
            }
        """)
        data_layout.addWidget(self.batch_btn)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Data info display
        info_group = QGroupBox("ℹ️ Data Information")
        info_layout = QVBoxLayout()

        self.data_info_display = QTextEdit()
        self.data_info_display.setReadOnly(True)
        self.data_info_display.setMaximumHeight(200)
        self.data_info_display.setPlaceholderText(
            "Data information will appear here after loading..."
        )
        self.data_info_display.setStyleSheet("""
            background: #0a0a0a;
            color: #cccccc;
            border: 1px solid #2d2d2d;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        """)

        info_layout.addWidget(self.data_info_display)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Timeframe selection
        tf_group = QGroupBox("⏱️ Active Timeframes")
        tf_layout = QHBoxLayout()

        self.tf_checkboxes = {}
        for tf in ["5min", "hourly", "daily"]:
            cb = QCheckBox(tf.capitalize())
            cb.setChecked(True)
            self.tf_checkboxes[tf] = cb
            tf_layout.addWidget(cb)

        tf_layout.addStretch()
        tf_group.setLayout(tf_layout)
        layout.addWidget(tf_group)

        layout.addStretch()

        # Next step hint
        hint = QLabel("💡 After loading data, go to 'Strategy Builder' to optimize your trading strategy.")
        hint.setStyleSheet("color: #888888; font-style: italic; padding: 10px;")
        layout.addWidget(hint)

        return tab

    def _create_strategy_tab(self):
        """Tab 2: Strategy Builder"""
        tab = QWidget()

        # Use splitter for results on right side
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        # Parameter ranges
        param_group = QGroupBox("📐 Parameter Ranges")
        param_layout = QVBoxLayout()

        # MN1
        mn1_row = QHBoxLayout()
        mn1_row.addWidget(QLabel("MN1 (Fast MA):"))
        self.mn1_min = QSpinBox()
        self.mn1_min.setRange(5, 100)
        self.mn1_min.setValue(10)
        self.mn1_max = QSpinBox()
        self.mn1_max.setRange(5, 100)
        self.mn1_max.setValue(30)
        mn1_row.addWidget(QLabel("Min:"))
        mn1_row.addWidget(self.mn1_min)
        mn1_row.addWidget(QLabel("Max:"))
        mn1_row.addWidget(self.mn1_max)
        mn1_row.addStretch()
        param_layout.addLayout(mn1_row)

        # MN2
        mn2_row = QHBoxLayout()
        mn2_row.addWidget(QLabel("MN2 (Slow MA):"))
        self.mn2_min = QSpinBox()
        self.mn2_min.setRange(20, 200)
        self.mn2_min.setValue(30)
        self.mn2_max = QSpinBox()
        self.mn2_max.setRange(20, 200)
        self.mn2_max.setValue(60)
        mn2_row.addWidget(QLabel("Min:"))
        mn2_row.addWidget(self.mn2_min)
        mn2_row.addWidget(QLabel("Max:"))
        mn2_row.addWidget(self.mn2_max)
        mn2_row.addStretch()
        param_layout.addLayout(mn2_row)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # Optimization controls
        opt_group = QGroupBox("⚡ Optimization")
        opt_layout = QVBoxLayout()

        # PSR button
        self.psr_btn = QPushButton("▶ Run PSR Optimization")
        self.psr_btn.setStyleSheet("""
            QPushButton {
                background: #00ff88;
                color: #000000;
                font-weight: bold;
                padding: 15px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #00dd77;
            }
            QPushButton:disabled {
                background: #2d2d2d;
                color: #666666;
            }
        """)
        opt_layout.addWidget(self.psr_btn)

        # Monte Carlo button
        self.monte_carlo_btn = QPushButton("🎲 Monte Carlo Simulation")
        self.monte_carlo_btn.setEnabled(False)
        self.monte_carlo_btn.setStyleSheet("""
            QPushButton {
                background: #9B59B6;
                color: #ffffff;
                font-weight: bold;
                padding: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #8E44AD;
            }
            QPushButton:disabled {
                background: #2d2d2d;
                color: #666666;
            }
        """)
        opt_layout.addWidget(self.monte_carlo_btn)

        # Walk-Forward button
        self.walk_forward_btn = QPushButton("📊 Walk-Forward Analysis")
        self.walk_forward_btn.setEnabled(False)
        self.walk_forward_btn.setStyleSheet("""
            QPushButton {
                background: #00BCD4;
                color: #ffffff;
                font-weight: bold;
                padding: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #00ACC1;
            }
            QPushButton:disabled {
                background: #2d2d2d;
                color: #666666;
            }
        """)
        opt_layout.addWidget(self.walk_forward_btn)

        opt_group.setLayout(opt_layout)
        left_layout.addWidget(opt_group)

        left_layout.addStretch()

        # Right side: Results display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        results_label = QLabel("📈 Optimization Results")
        results_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff88;")
        right_layout.addWidget(results_label)

        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText(
            "Results will appear here after optimization...\n\n"
            "Metrics include:\n"
            "• Sharpe Ratio\n"
            "• Max Drawdown\n"
            "• Win Rate\n"
            "• Total Return\n"
            "• Risk-Adjusted Metrics"
        )
        self.results_display.setStyleSheet("""
            background: #0a0a0a;
            color: #cccccc;
            border: 1px solid #2d2d2d;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        """)
        right_layout.addWidget(self.results_display)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Left panel
        splitter.setStretchFactor(1, 2)  # Right panel (larger)

        # Main tab layout
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(splitter)

        return tab

    def _create_regime_tab(self):
        """Tab 3: Institutional Regime Analysis"""
        tab = QWidget()

        # Use splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        # Basic regime detection
        basic_group = QGroupBox("🔍 Regime Detection")
        basic_layout = QVBoxLayout()

        self.detect_regime_btn = QPushButton("Detect Current Regime")
        self.detect_regime_btn.setStyleSheet("""
            QPushButton {
                background: #FF6B35;
                color: #ffffff;
                padding: 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #FF5722;
            }
        """)
        basic_layout.addWidget(self.detect_regime_btn)

        self.train_predictor_btn = QPushButton("🤖 Train ML Predictor")
        self.train_predictor_btn.setStyleSheet("""
            QPushButton {
                background: #4ECDC4;
                color: #ffffff;
                padding: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #45B7AF;
            }
        """)
        basic_layout.addWidget(self.train_predictor_btn)

        basic_group.setLayout(basic_layout)
        left_layout.addWidget(basic_group)

        # Institutional analysis
        inst_group = QGroupBox("🏛️ Institutional Analysis")
        inst_layout = QVBoxLayout()

        self.calibrate_btn = QPushButton("📐 Calibrate Probabilities")
        self.calibrate_btn.setStyleSheet("background: #9B59B6; color: white; padding: 10px; border-radius: 4px;")
        inst_layout.addWidget(self.calibrate_btn)

        self.robustness_btn = QPushButton("🔬 Robustness Tests")
        self.robustness_btn.setStyleSheet("background: #E74C3C; color: white; padding: 10px; border-radius: 4px;")
        inst_layout.addWidget(self.robustness_btn)

        self.cross_asset_btn = QPushButton("🌐 Cross-Asset Analysis")
        self.cross_asset_btn.setStyleSheet("background: #F39C12; color: white; padding: 10px; border-radius: 4px;")
        inst_layout.addWidget(self.cross_asset_btn)

        self.diagnostics_btn = QPushButton("📋 Full Diagnostics")
        self.diagnostics_btn.setStyleSheet("background: #16A085; color: white; padding: 10px; border-radius: 4px;")
        inst_layout.addWidget(self.diagnostics_btn)

        inst_group.setLayout(inst_layout)
        left_layout.addWidget(inst_group)

        left_layout.addStretch()

        # Right: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        results_label = QLabel("📊 Analysis Results")
        results_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff88;")
        right_layout.addWidget(results_label)

        self.regime_display = QTextEdit()
        self.regime_display.setReadOnly(True)
        self.regime_display.setPlaceholderText(
            "Regime analysis results will appear here...\n\n"
            "Analysis includes:\n"
            "• Current market regime\n"
            "• Confidence levels\n"
            "• Regime probabilities\n"
            "• Statistical tests\n"
            "• Cross-asset correlations"
        )
        self.regime_display.setStyleSheet("""
            background: #0a0a0a;
            color: #cccccc;
            border: 1px solid #2d2d2d;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Courier New', monospace;
        """)
        right_layout.addWidget(self.regime_display)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(splitter)

        return tab

    def _create_trading_tab(self):
        """Tab 4: Live Trading"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Warning banner
        warning = QLabel("⚠️ LIVE TRADING - Real money at risk. Use with extreme caution.")
        warning.setStyleSheet("""
            background: #ff4444;
            color: #ffffff;
            padding: 15px;
            font-weight: bold;
            font-size: 13px;
            border-radius: 4px;
        """)
        layout.addWidget(warning)

        # Alpaca configuration
        config_group = QGroupBox("🔑 Alpaca Configuration")
        config_layout = QVBoxLayout()

        # API keys
        api_row = QHBoxLayout()
        api_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Your Alpaca API key...")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_row.addWidget(self.api_key_input)
        config_layout.addLayout(api_row)

        secret_row = QHBoxLayout()
        secret_row.addWidget(QLabel("Secret Key:"))
        self.api_secret_input = QLineEdit()
        self.api_secret_input.setPlaceholderText("Your Alpaca secret key...")
        self.api_secret_input.setEchoMode(QLineEdit.EchoMode.Password)
        secret_row.addWidget(self.api_secret_input)
        config_layout.addLayout(secret_row)

        # Paper trading checkbox
        self.paper_trading_check = QCheckBox("Paper Trading (recommended for testing)")
        self.paper_trading_check.setChecked(True)
        self.paper_trading_check.setStyleSheet("color: #00ff88; font-weight: bold;")
        config_layout.addWidget(self.paper_trading_check)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Trading controls
        controls_group = QGroupBox("🎮 Trading Controls")
        controls_layout = QVBoxLayout()

        self.start_trading_btn = QPushButton("▶ START LIVE TRADING")
        self.start_trading_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                padding: 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #229954;
            }
        """)
        controls_layout.addWidget(self.start_trading_btn)

        self.stop_trading_btn = QPushButton("⏹ STOP TRADING")
        self.stop_trading_btn.setEnabled(False)
        self.stop_trading_btn.setStyleSheet("""
            QPushButton {
                background: #c0392b;
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                padding: 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #a93226;
            }
            QPushButton:disabled {
                background: #2d2d2d;
                color: #666666;
            }
        """)
        controls_layout.addWidget(self.stop_trading_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Status display
        status_group = QGroupBox("📊 Trading Status")
        status_layout = QVBoxLayout()

        self.trading_status_display = QTextEdit()
        self.trading_status_display.setReadOnly(True)
        self.trading_status_display.setMaximumHeight(300)
        self.trading_status_display.setPlaceholderText(
            "Trading status will appear here...\n\n"
            "Displays:\n"
            "• Connection status\n"
            "• Account balance\n"
            "• Open positions\n"
            "• Recent trades\n"
            "• P&L summary"
        )
        self.trading_status_display.setStyleSheet("""
            background: #0a0a0a;
            color: #cccccc;
            border: 1px solid #2d2d2d;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Courier New', monospace;
        """)
        status_layout.addWidget(self.trading_status_display)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        layout.addStretch()

        return tab

    def _create_settings_tab(self):
        """Tab 5: Settings & Risk Management"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Risk management
        risk_group = QGroupBox("⚙️ Risk Management")
        risk_layout = QVBoxLayout()

        # Max position size
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Max Position Size:"))
        self.max_position = QDoubleSpinBox()
        self.max_position.setRange(0.01, 1.0)
        self.max_position.setValue(0.1)
        self.max_position.setSuffix(" (10% of portfolio)")
        pos_row.addWidget(self.max_position)
        pos_row.addStretch()
        risk_layout.addLayout(pos_row)

        # Stop loss
        stop_row = QHBoxLayout()
        stop_row.addWidget(QLabel("Stop Loss:"))
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0.01, 0.50)
        self.stop_loss.setValue(0.02)
        self.stop_loss.setSuffix(" (2%)")
        stop_row.addWidget(self.stop_loss)
        stop_row.addStretch()
        risk_layout.addLayout(stop_row)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

        # Transaction costs
        cost_group = QGroupBox("💰 Transaction Costs")
        cost_layout = QVBoxLayout()

        cost_row = QHBoxLayout()
        cost_row.addWidget(QLabel("Commission:"))
        self.commission = QDoubleSpinBox()
        self.commission.setRange(0.0, 1.0)
        self.commission.setValue(0.0006)
        self.commission.setDecimals(4)
        self.commission.setSuffix(" per share")
        cost_row.addWidget(self.commission)
        cost_row.addStretch()
        cost_layout.addLayout(cost_row)

        # Quick presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        stocks_btn = QPushButton("Stocks (0.06%)")
        stocks_btn.clicked.connect(lambda: self.commission.setValue(0.0006))
        preset_row.addWidget(stocks_btn)

        crypto_btn = QPushButton("Crypto (0.35%)")
        crypto_btn.clicked.connect(lambda: self.commission.setValue(0.0035))
        preset_row.addWidget(crypto_btn)

        zero_btn = QPushButton("Zero")
        zero_btn.clicked.connect(lambda: self.commission.setValue(0.0))
        preset_row.addWidget(zero_btn)

        preset_row.addStretch()
        cost_layout.addLayout(preset_row)

        cost_group.setLayout(cost_layout)
        layout.addWidget(cost_group)

        layout.addStretch()

        # Save settings button
        save_btn = QPushButton("💾 Save Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background: #00ff88;
                color: #000000;
                font-weight: bold;
                padding: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #00dd77;
            }
        """)
        layout.addWidget(save_btn)

        return tab

    def _add_professional_statusbar(self):
        """Add professional status bar"""
        statusbar = self.statusBar()
        statusbar.setStyleSheet("""
            QStatusBar {
                background: #1a1a1a;
                color: #888888;
                border-top: 1px solid #2d2d2d;
                padding: 5px;
            }
        """)

        # Status label
        self.status_label = QLabel("Ready")
        statusbar.addWidget(self.status_label)

        # Version info
        version_label = QLabel("v1.0.0")
        statusbar.addPermanentWidget(version_label)

    def load_data(self):
        """Load data from Yahoo Finance"""
        ticker = self.ticker_input.text().strip().upper()

        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a ticker symbol")
            return

        self.status_label.setText(f"Loading {ticker}...")

        # Load data using DataLoader
        df_dict, error = DataLoader.load_yfinance_data(ticker)

        if error:
            QMessageBox.critical(self, "Data Error", f"Failed to load {ticker}:\n{error}")
            self.status_label.setText("Error loading data")
            return

        self.df_dict_full = df_dict
        self.current_ticker = ticker

        # Update data info display
        info_text = f"✅ Successfully loaded {ticker}\n\n"
        info_text += "=" * 50 + "\n"

        for tf, df in df_dict.items():
            info_text += f"\n{tf.upper()}:\n"
            info_text += f"  Rows: {len(df):,}\n"
            info_text += f"  Date Range: {df['Datetime'].min()} to {df['Datetime'].max()}\n"
            info_text += f"  Columns: {', '.join(df.columns[:5])}\n"

        self.data_info_display.setText(info_text)
        self.status_label.setText(f"Loaded {ticker} successfully")

        # Enable strategy tab
        self.psr_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "Success",
            f"Successfully loaded {ticker}!\n\nGo to 'Strategy Builder' tab to optimize."
        )


# Entry point
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
