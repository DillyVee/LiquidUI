"""
Professional Multi-Tab GUI for LiquidUI Trading Platform
Complete implementation with all features from original GUI
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from config.settings import (
    AlpacaConfig,
    IndicatorRanges,
    OptimizationConfig,
    Paths,
    RiskConfig,
    TransactionCosts,
)
from gui.styles import (
    MAIN_STYLESHEET,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    LIVE_TRADING_BUTTON_ACTIVE,
    LIVE_TRADING_BUTTON_STOPPED,
)
from optimization import (
    MultiTimeframeOptimizer,
    MonteCarloSimulator,
    WalkForwardAnalyzer,
)
from models.regime_detection import MarketRegimeDetector
from models.regime_predictor import RegimePredictor
from models.regime_calibration import MultiClassCalibrator
from models.regime_agreement import MultiHorizonAgreementIndex
from models.regime_diagnostics import RegimeDiagnosticAnalyzer
from models.regime_cross_asset import CrossAssetRegimeAnalyzer
from data import DataLoader
from trading import AlpacaLiveTrader, ALPACA_AVAILABLE


class MainWindow(QMainWindow):
    """Professional multi-tab trading platform GUI"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidUI - Professional Trading Platform")
        self.setGeometry(100, 100, 1600, 1000)

        # Data storage
        self.df_dict: Optional[Dict[str, pd.DataFrame]] = None
        self.df_dict_full: Optional[Dict[str, pd.DataFrame]] = None
        self.current_ticker: str = ""
        self.data_source: str = "yfinance"
        self.last_trade_log: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict] = None
        self.best_results: Optional[pd.DataFrame] = None

        # Worker threads
        self.worker: Optional[MultiTimeframeOptimizer] = None

        # Regime detection objects
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.regime_predictor: Optional[RegimePredictor] = None
        self.regime_calibrator: Optional[MultiClassCalibrator] = None
        self.multi_horizon_agreement: Optional[MultiHorizonAgreementIndex] = None
        self.regime_diagnostics: Optional[RegimeDiagnosticAnalyzer] = None
        self.cross_asset_analyzer: Optional[CrossAssetRegimeAnalyzer] = None
        self.current_regime_state = None
        self.calibrated_predictions = None

        # Live trading
        self.live_trader: Optional[AlpacaLiveTrader] = None

        # Settings
        self.position_size_pct = RiskConfig.DEFAULT_POSITION_SIZE
        self.max_positions = RiskConfig.DEFAULT_MAX_POSITIONS
        self.transaction_costs = TransactionCosts()

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the tabbed interface"""
        self.setStyleSheet(MAIN_STYLESHEET)

        # Central widget with tab container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 12px 24px;
                margin-right: 2px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 120px;
                font-size: 13px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #4CAF50;
                border-bottom: 2px solid #4CAF50;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3a3a3a;
            }
        """
        )

        # Create all tabs
        self.tab1 = self._create_data_tab()
        self.tab2 = self._create_strategy_tab()
        self.tab3 = self._create_regime_tab()
        self.tab4 = self._create_trading_tab()
        self.tab5 = self._create_settings_tab()

        # Add tabs to widget
        self.tabs.addTab(self.tab1, "📊 Data & Setup")
        self.tabs.addTab(self.tab2, "⚙️ Strategy Optimization")
        self.tabs.addTab(self.tab3, "🏛️ Regime Analysis")
        self.tabs.addTab(self.tab4, "🔴 Live Trading")
        self.tabs.addTab(self.tab5, "⚙️ Settings")

        main_layout.addWidget(self.tabs)

        # Status bar
        self.statusBar().showMessage("Ready to load data")

    # =============================================================================
    # TAB 1: DATA & SETUP
    # =============================================================================

    def _create_data_tab(self) -> QWidget:
        """Create data loading and visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Welcome message
        welcome_label = QLabel("👋 Welcome to LiquidUI Professional Trading Platform")
        welcome_label.setStyleSheet(
            f"font-size: 18px; color: {COLOR_SUCCESS}; font-weight: bold; padding: 15px;"
        )
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)

        # Data loading section
        data_group = QGroupBox("Load Market Data")
        data_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        data_layout = QVBoxLayout()

        # Single ticker input
        ticker_layout = QHBoxLayout()
        ticker_layout.addWidget(QLabel("Ticker Symbol:"))
        self.ticker_input = QLineEdit("SPY")
        self.ticker_input.setPlaceholderText("Enter ticker (e.g., SPY, AAPL, BTC-USD)")
        self.ticker_input.setMaximumWidth(200)
        ticker_layout.addWidget(self.ticker_input)

        self.load_yf_btn = QPushButton("📈 Load from Yahoo Finance")
        self.load_yf_btn.setStyleSheet(
            f"background-color: {COLOR_SUCCESS}; font-weight: bold; padding: 10px;"
        )
        self.load_yf_btn.clicked.connect(self.load_yfinance)
        ticker_layout.addWidget(self.load_yf_btn)
        ticker_layout.addStretch()
        data_layout.addLayout(ticker_layout)

        # Batch ticker loading
        batch_layout = QHBoxLayout()
        self.batch_ticker_btn = QPushButton("📁 Load Multiple Tickers from CSV")
        self.batch_ticker_btn.clicked.connect(self.load_ticker_list)
        batch_layout.addWidget(self.batch_ticker_btn)
        batch_layout.addStretch()
        data_layout.addLayout(batch_layout)

        # Date range display
        self.date_range_label = QLabel("No data loaded")
        self.date_range_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 5px;"
        )
        data_layout.addWidget(self.date_range_label)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Timeframe selection
        tf_group = QGroupBox("Select Timeframes")
        tf_layout = QHBoxLayout()

        self.tf_checkboxes = {}
        for tf_name, tf_label in [
            ("daily", "Daily"),
            ("hourly", "Hourly"),
            ("5min", "5-Minute"),
        ]:
            cb = QCheckBox(tf_label)
            cb.setChecked(tf_name == "daily")
            cb.stateChanged.connect(self.on_timeframe_changed)
            self.tf_checkboxes[tf_name] = cb
            tf_layout.addWidget(cb)

        tf_layout.addStretch()
        tf_group.setLayout(tf_layout)
        layout.addWidget(tf_group)

        # Chart display
        chart_group = QGroupBox("Price Chart")
        chart_layout = QVBoxLayout()

        self.figure = plt.figure(figsize=(12, 6), facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)

        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)

        # Next steps hint
        hint_label = QLabel(
            "💡 Next Step: Load data, then go to 'Strategy Optimization' tab to run PSR optimization"
        )
        hint_label.setStyleSheet(
            "color: #FFA726; padding: 10px; background-color: #2d2d2d; border-radius: 4px;"
        )
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        layout.addStretch()
        return tab

    # =============================================================================
    # TAB 2: STRATEGY OPTIMIZATION
    # =============================================================================

    def _create_strategy_tab(self) -> QWidget:
        """Create strategy optimization tab"""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        # Parameter ranges
        params_group = QGroupBox("Strategy Parameter Ranges")
        params_layout = QVBoxLayout()

        self._add_param_range(params_layout, "MN1 (Fast Moving Average)", "mn1")
        self._add_param_range(params_layout, "MN2 (Slow Moving Average)", "mn2")
        self._add_param_range(
            params_layout, "Entry Threshold", "entry", is_decimal=True
        )
        self._add_param_range(params_layout, "Exit Threshold", "exit", is_decimal=True)
        self._add_param_range(params_layout, "On Cycle", "on")
        self._add_param_range(params_layout, "Off Cycle", "off")

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Optimization settings
        opt_group = QGroupBox("Optimization Settings")
        opt_layout = QVBoxLayout()

        # Trials
        trials_layout = QHBoxLayout()
        trials_layout.addWidget(QLabel("Trials:"))
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(300, 100000)
        self.trials_spin.setValue(OptimizationConfig.DEFAULT_TRIALS)
        self.trials_spin.setSingleStep(100)
        trials_layout.addWidget(self.trials_spin)
        trials_layout.addStretch()
        opt_layout.addLayout(trials_layout)

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(50, 2000)
        self.batch_spin.setValue(OptimizationConfig.DEFAULT_BATCH_SIZE)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        opt_layout.addLayout(batch_layout)

        # Objective
        obj_layout = QHBoxLayout()
        obj_layout.addWidget(QLabel("Objective:"))
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(["psr", "sharpe", "sortino", "calmar"])
        obj_layout.addWidget(self.objective_combo)
        obj_layout.addStretch()
        opt_layout.addLayout(obj_layout)

        opt_group.setLayout(opt_layout)
        left_layout.addWidget(opt_group)

        # Action buttons
        btn_group = QGroupBox("Actions")
        btn_layout = QVBoxLayout()

        # PSR Optimization
        opt_btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶️ START PSR OPTIMIZATION")
        self.start_btn.setStyleSheet(
            f"background-color: {COLOR_SUCCESS}; font-weight: bold; padding: 15px; font-size: 14px;"
        )
        self.start_btn.clicked.connect(self.start_optimization)
        opt_btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ STOP")
        self.stop_btn.setStyleSheet(
            f"background-color: {COLOR_DANGER}; font-weight: bold; padding: 15px;"
        )
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        opt_btn_layout.addWidget(self.stop_btn)
        btn_layout.addLayout(opt_btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)

        # Phase label
        self.phase_label = QLabel("Ready to optimize")
        self.phase_label.setStyleSheet("color: #888; font-style: italic;")
        btn_layout.addWidget(self.phase_label)

        # Monte Carlo
        self.monte_carlo_btn = QPushButton("🎲 Run Monte Carlo Simulation")
        self.monte_carlo_btn.clicked.connect(self.run_monte_carlo)
        btn_layout.addWidget(self.monte_carlo_btn)

        # Monte Carlo simulations
        mc_layout = QHBoxLayout()
        mc_layout.addWidget(QLabel("MC Simulations:"))
        self.mc_simulations_spin = QSpinBox()
        self.mc_simulations_spin.setRange(100, 10000)
        self.mc_simulations_spin.setValue(1000)
        mc_layout.addWidget(self.mc_simulations_spin)
        mc_layout.addStretch()
        btn_layout.addLayout(mc_layout)

        # Walk-Forward Analysis
        self.walk_forward_btn = QPushButton("📈 Run Walk-Forward Analysis")
        self.walk_forward_btn.clicked.connect(self.run_walk_forward)
        btn_layout.addWidget(self.walk_forward_btn)

        # WF settings
        wf_layout = QVBoxLayout()
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train Days:"))
        self.wf_train_days_spin = QSpinBox()
        self.wf_train_days_spin.setRange(30, 730)
        self.wf_train_days_spin.setValue(180)
        train_layout.addWidget(self.wf_train_days_spin)
        train_layout.addStretch()
        wf_layout.addLayout(train_layout)

        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test Days:"))
        self.wf_test_days_spin = QSpinBox()
        self.wf_test_days_spin.setRange(7, 365)
        self.wf_test_days_spin.setValue(30)
        test_layout.addWidget(self.wf_test_days_spin)
        test_layout.addStretch()
        wf_layout.addLayout(test_layout)

        trials_layout = QHBoxLayout()
        trials_layout.addWidget(QLabel("WF Trials:"))
        self.wf_trials_spin = QSpinBox()
        self.wf_trials_spin.setRange(50, 5000)
        self.wf_trials_spin.setValue(500)
        trials_layout.addWidget(self.wf_trials_spin)
        trials_layout.addStretch()
        wf_layout.addLayout(trials_layout)

        btn_layout.addLayout(wf_layout)

        btn_group.setLayout(btn_layout)
        left_layout.addWidget(btn_group)

        left_layout.addStretch()

        # Right panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Results display
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout()

        self.best_params_label = QLabel("No optimization run yet")
        self.best_params_label.setWordWrap(True)
        self.best_params_label.setStyleSheet(
            "padding: 10px; background-color: #2d2d2d; border-radius: 4px;"
        )
        results_layout.addWidget(self.best_params_label)

        # Metrics display
        metrics_layout = QHBoxLayout()

        # PSR
        psr_box = QVBoxLayout()
        psr_label = QLabel("PSR")
        psr_label.setStyleSheet("font-size: 11px; color: #888;")
        psr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.psr_label = QLabel("--")
        self.psr_label.setStyleSheet(
            f"font-size: 24px; font-weight: bold; color: {COLOR_SUCCESS}; background-color: #2d2d2d; padding: 10px; border-radius: 4px;"
        )
        self.psr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        psr_box.addWidget(psr_label)
        psr_box.addWidget(self.psr_label)
        metrics_layout.addLayout(psr_box)

        # Sharpe
        sharpe_box = QVBoxLayout()
        sharpe_label = QLabel("Sharpe")
        sharpe_label.setStyleSheet("font-size: 11px; color: #888;")
        sharpe_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sharpe_label = QLabel("--")
        self.sharpe_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #e0e0e0; background-color: #2d2d2d; padding: 10px; border-radius: 4px;"
        )
        self.sharpe_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sharpe_box.addWidget(sharpe_label)
        sharpe_box.addWidget(self.sharpe_label)
        metrics_layout.addLayout(sharpe_box)

        # Sortino
        sortino_box = QVBoxLayout()
        sortino_label = QLabel("Sortino")
        sortino_label.setStyleSheet("font-size: 11px; color: #888;")
        sortino_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sortino_label = QLabel("--")
        self.sortino_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #e0e0e0; background-color: #2d2d2d; padding: 10px; border-radius: 4px;"
        )
        self.sortino_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sortino_box.addWidget(sortino_label)
        sortino_box.addWidget(self.sortino_label)
        metrics_layout.addLayout(sortino_box)

        results_layout.addLayout(metrics_layout)

        # Show full report button
        self.show_report_btn = QPushButton("📊 Show Full PSR Report")
        self.show_report_btn.clicked.connect(self.show_psr_report)
        results_layout.addWidget(self.show_report_btn)

        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        # Equity curve chart
        chart_group = QGroupBox("Equity Curve")
        chart_layout = QVBoxLayout()

        self.results_figure = plt.figure(figsize=(10, 6), facecolor="#1e1e1e")
        self.results_canvas = FigureCanvas(self.results_figure)
        chart_layout.addWidget(self.results_canvas)

        chart_group.setLayout(chart_layout)
        right_layout.addWidget(chart_group)

        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)

        return tab

    # =============================================================================
    # TAB 3: REGIME ANALYSIS
    # =============================================================================

    def _create_regime_tab(self) -> QWidget:
        """Create regime analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Basic regime detection
        basic_group = QGroupBox("Basic Regime Detection")
        basic_layout = QVBoxLayout()

        self.detect_regime_btn = QPushButton("🔍 Detect Current Market Regime")
        self.detect_regime_btn.setStyleSheet(
            f"background-color: {COLOR_SUCCESS}; font-weight: bold; padding: 12px;"
        )
        self.detect_regime_btn.clicked.connect(self.detect_market_regime)
        basic_layout.addWidget(self.detect_regime_btn)

        self.regime_display = QLabel("No regime detected yet")
        self.regime_display.setStyleSheet(
            "padding: 15px; background-color: #2d2d2d; border-radius: 4px; font-size: 14px;"
        )
        self.regime_display.setWordWrap(True)
        basic_layout.addWidget(self.regime_display)

        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # ML Predictor
        ml_group = QGroupBox("Machine Learning Regime Predictor")
        ml_layout = QVBoxLayout()

        self.train_predictor_btn = QPushButton("🤖 Train Regime Predictor")
        self.train_predictor_btn.clicked.connect(self.train_regime_predictor)
        ml_layout.addWidget(self.train_predictor_btn)

        self.prediction_display = QLabel("No prediction yet")
        self.prediction_display.setStyleSheet(
            "padding: 15px; background-color: #2d2d2d; border-radius: 4px;"
        )
        self.prediction_display.setWordWrap(True)
        ml_layout.addWidget(self.prediction_display)

        ml_group.setLayout(ml_layout)
        layout.addWidget(ml_group)

        # Advanced institutional features
        inst_group = QGroupBox("Institutional-Grade Analysis")
        inst_layout = QVBoxLayout()

        # Row 1
        row1 = QHBoxLayout()
        self.calc_pbr_btn = QPushButton("📊 Calculate PBR")
        self.calc_pbr_btn.clicked.connect(self.calculate_pbr)
        row1.addWidget(self.calc_pbr_btn)

        self.calibrate_btn = QPushButton("🎯 Calibrate Probabilities")
        self.calibrate_btn.clicked.connect(self.calibrate_probabilities)
        row1.addWidget(self.calibrate_btn)
        inst_layout.addLayout(row1)

        # Row 2
        row2 = QHBoxLayout()
        self.agreement_btn = QPushButton("🔄 Multi-Horizon Agreement")
        self.agreement_btn.clicked.connect(self.check_multi_horizon_agreement)
        row2.addWidget(self.agreement_btn)

        self.robustness_btn = QPushButton("🛡️ Robustness Tests")
        self.robustness_btn.clicked.connect(self.run_robustness_tests)
        row2.addWidget(self.robustness_btn)
        inst_layout.addLayout(row2)

        # Row 3
        row3 = QHBoxLayout()
        self.diagnostics_btn = QPushButton("🔬 Regime Diagnostics")
        self.diagnostics_btn.clicked.connect(self.run_regime_diagnostics)
        row3.addWidget(self.diagnostics_btn)

        self.cross_asset_btn = QPushButton("🌍 Cross-Asset Analysis")
        self.cross_asset_btn.clicked.connect(self.run_cross_asset_analysis)
        row3.addWidget(self.cross_asset_btn)
        inst_layout.addLayout(row3)

        # Results display
        self.institutional_display = QTextEdit()
        self.institutional_display.setReadOnly(True)
        self.institutional_display.setMaximumHeight(200)
        self.institutional_display.setPlaceholderText(
            "Institutional analysis results will appear here..."
        )
        self.institutional_display.setStyleSheet(
            "background-color: #2d2d2d; border: 1px solid #3a3a3a;"
        )
        inst_layout.addWidget(self.institutional_display)

        # PBR display
        self.pbr_display = QLabel("PBR: Not calculated")
        self.pbr_display.setStyleSheet(
            "padding: 10px; background-color: #2d2d2d; border-radius: 4px; font-weight: bold;"
        )
        inst_layout.addWidget(self.pbr_display)

        inst_group.setLayout(inst_layout)
        layout.addWidget(inst_group)

        layout.addStretch()
        return tab

    # =============================================================================
    # TAB 4: LIVE TRADING
    # =============================================================================

    def _create_trading_tab(self) -> QWidget:
        """Create live trading tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Warning banner
        warning_label = QLabel("⚠️ DANGER: Live Trading Mode - Real Money at Risk!")
        warning_label.setStyleSheet(
            f"background-color: {COLOR_DANGER}; color: white; font-size: 16px; font-weight: bold; padding: 15px; border-radius: 4px;"
        )
        warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warning_label)

        # API Configuration
        api_group = QGroupBox("Alpaca API Configuration")
        api_layout = QVBoxLayout()

        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter Alpaca API Key")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        key_layout.addWidget(self.api_key_input)
        api_layout.addLayout(key_layout)

        secret_layout = QHBoxLayout()
        secret_layout.addWidget(QLabel("API Secret:"))
        self.api_secret_input = QLineEdit()
        self.api_secret_input.setPlaceholderText("Enter Alpaca API Secret")
        self.api_secret_input.setEchoMode(QLineEdit.EchoMode.Password)
        secret_layout.addWidget(self.api_secret_input)
        api_layout.addLayout(secret_layout)

        # Paper trading toggle
        self.paper_trading_cb = QCheckBox("Paper Trading (Recommended)")
        self.paper_trading_cb.setChecked(True)
        self.paper_trading_cb.setStyleSheet(
            f"color: {COLOR_SUCCESS}; font-weight: bold;"
        )
        api_layout.addWidget(self.paper_trading_cb)

        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Trading controls
        control_group = QGroupBox("Trading Controls")
        control_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.live_trading_btn = QPushButton("▶️ START LIVE TRADING")
        self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_STOPPED)
        self.live_trading_btn.clicked.connect(self.toggle_live_trading)
        btn_layout.addWidget(self.live_trading_btn)
        control_layout.addLayout(btn_layout)

        # Status display
        self.trading_status_label = QLabel("Status: Not Running")
        self.trading_status_label.setStyleSheet(
            "padding: 15px; background-color: #2d2d2d; border-radius: 4px; font-size: 14px;"
        )
        control_layout.addWidget(self.trading_status_label)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Trade log
        log_group = QGroupBox("Trade Log")
        log_layout = QVBoxLayout()

        self.trade_log_display = QTextEdit()
        self.trade_log_display.setReadOnly(True)
        self.trade_log_display.setPlaceholderText(
            "Trade executions will appear here..."
        )
        self.trade_log_display.setStyleSheet(
            "background-color: #2d2d2d; border: 1px solid #3a3a3a;"
        )
        log_layout.addWidget(self.trade_log_display)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        return tab

    # =============================================================================
    # TAB 5: SETTINGS
    # =============================================================================

    def _create_settings_tab(self) -> QWidget:
        """Create settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Risk Management
        risk_group = QGroupBox("Risk Management")
        risk_layout = QVBoxLayout()

        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Position Size (%):"))
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.1, 100.0)
        self.position_size_spin.setValue(self.position_size_pct)
        self.position_size_spin.setSingleStep(1.0)
        self.position_size_spin.setSuffix("%")
        self.position_size_spin.valueChanged.connect(self.on_risk_settings_changed)
        pos_layout.addWidget(self.position_size_spin)
        pos_layout.addStretch()
        risk_layout.addLayout(pos_layout)

        max_pos_layout = QHBoxLayout()
        max_pos_layout.addWidget(QLabel("Max Positions:"))
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 20)
        self.max_positions_spin.setValue(self.max_positions)
        self.max_positions_spin.valueChanged.connect(self.on_risk_settings_changed)
        max_pos_layout.addWidget(self.max_positions_spin)
        max_pos_layout.addStretch()
        risk_layout.addLayout(max_pos_layout)

        self.leverage_warning_label = QLabel("")
        self.leverage_warning_label.setStyleSheet(
            f"color: {COLOR_WARNING}; font-weight: bold;"
        )
        risk_layout.addWidget(self.leverage_warning_label)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

        # Transaction Costs
        tc_group = QGroupBox("Transaction Costs")
        tc_layout = QVBoxLayout()

        comm_layout = QHBoxLayout()
        comm_layout.addWidget(QLabel("Commission (%):"))
        self.commission_pct_spin = QDoubleSpinBox()
        self.commission_pct_spin.setRange(0.0, 1.0)
        self.commission_pct_spin.setValue(self.transaction_costs.commission_pct * 100)
        self.commission_pct_spin.setSingleStep(0.01)
        self.commission_pct_spin.setSuffix("%")
        self.commission_pct_spin.valueChanged.connect(self.on_transaction_costs_changed)
        comm_layout.addWidget(self.commission_pct_spin)
        comm_layout.addStretch()
        tc_layout.addLayout(comm_layout)

        slip_layout = QHBoxLayout()
        slip_layout.addWidget(QLabel("Slippage (%):"))
        self.slippage_pct_spin = QDoubleSpinBox()
        self.slippage_pct_spin.setRange(0.0, 1.0)
        self.slippage_pct_spin.setValue(self.transaction_costs.slippage_pct * 100)
        self.slippage_pct_spin.setSingleStep(0.01)
        self.slippage_pct_spin.setSuffix("%")
        self.slippage_pct_spin.valueChanged.connect(self.on_transaction_costs_changed)
        slip_layout.addWidget(self.slippage_pct_spin)
        slip_layout.addStretch()
        tc_layout.addLayout(slip_layout)

        spread_layout = QHBoxLayout()
        spread_layout.addWidget(QLabel("Spread (%):"))
        self.spread_pct_spin = QDoubleSpinBox()
        self.spread_pct_spin.setRange(0.0, 1.0)
        self.spread_pct_spin.setValue(self.transaction_costs.spread_pct * 100)
        self.spread_pct_spin.setSingleStep(0.01)
        self.spread_pct_spin.setSuffix("%")
        self.spread_pct_spin.valueChanged.connect(self.on_transaction_costs_changed)
        spread_layout.addWidget(self.spread_pct_spin)
        spread_layout.addStretch()
        tc_layout.addLayout(spread_layout)

        # Preset buttons
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))

        stocks_btn = QPushButton("📈 Stocks")
        stocks_btn.clicked.connect(self.set_costs_for_stocks)
        preset_layout.addWidget(stocks_btn)

        crypto_btn = QPushButton("₿ Crypto")
        crypto_btn.clicked.connect(self.set_costs_for_crypto)
        preset_layout.addWidget(crypto_btn)

        zero_btn = QPushButton("0️⃣ Zero")
        zero_btn.clicked.connect(self.set_costs_zero)
        preset_layout.addWidget(zero_btn)

        preset_layout.addStretch()
        tc_layout.addLayout(preset_layout)

        tc_group.setLayout(tc_layout)
        layout.addWidget(tc_group)

        layout.addStretch()
        return tab

    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    def _add_param_range(
        self, layout: QVBoxLayout, label: str, param_name: str, is_decimal: bool = False
    ):
        """Add parameter range controls"""
        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel(f"{label}:"))

        group_layout.addWidget(QLabel("Min:"))
        if is_decimal:
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-100.0, 100.0)
            min_spin.setSingleStep(0.1)
            min_spin.setDecimals(2)
        else:
            min_spin = QSpinBox()
            min_spin.setRange(1, 500)

        setattr(self, f"{param_name}_min", min_spin)
        group_layout.addWidget(min_spin)

        group_layout.addWidget(QLabel("Max:"))
        if is_decimal:
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-100.0, 100.0)
            max_spin.setSingleStep(0.1)
            max_spin.setDecimals(2)
        else:
            max_spin = QSpinBox()
            max_spin.setRange(1, 500)

        setattr(self, f"{param_name}_max", max_spin)
        group_layout.addWidget(max_spin)

        group_layout.addStretch()
        layout.addLayout(group_layout)

        # Set default values
        if param_name == "mn1":
            min_spin.setValue(IndicatorRanges.MN1_RANGE[0])
            max_spin.setValue(IndicatorRanges.MN1_RANGE[1])
        elif param_name == "mn2":
            min_spin.setValue(IndicatorRanges.MN2_RANGE[0])
            max_spin.setValue(IndicatorRanges.MN2_RANGE[1])
        elif param_name == "entry":
            min_spin.setValue(IndicatorRanges.ENTRY_RANGE[0])
            max_spin.setValue(IndicatorRanges.ENTRY_RANGE[1])
        elif param_name == "exit":
            min_spin.setValue(IndicatorRanges.EXIT_RANGE[0])
            max_spin.setValue(IndicatorRanges.EXIT_RANGE[1])
        elif param_name == "on":
            min_spin.setValue(IndicatorRanges.ON_RANGE[0])
            max_spin.setValue(IndicatorRanges.ON_RANGE[1])
        elif param_name == "off":
            min_spin.setValue(IndicatorRanges.OFF_RANGE[0])
            max_spin.setValue(IndicatorRanges.OFF_RANGE[1])

    # =============================================================================
    # DATA LOADING METHODS
    # =============================================================================

    def load_yfinance(self):
        """Load single ticker from Yahoo Finance"""
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a ticker symbol")
            return

        self.statusBar().showMessage(f"Loading {ticker}...")

        try:
            # Load data using DataLoader
            df_dict, error = DataLoader.load_yfinance_data(ticker)

            if error:
                QMessageBox.critical(
                    self, "Data Loading Error", f"Failed to load {ticker}:\n{error}"
                )
                self.statusBar().showMessage("Error loading data")
                return

            self.df_dict_full = df_dict

            # Filter to selected timeframes
            self._filter_timeframes()

            self.current_ticker = ticker
            self.data_source = "yfinance"

            # Update date range display
            if "daily" in self.df_dict:
                df = self.df_dict["daily"]
                start_date = pd.to_datetime(df["Datetime"].iloc[0]).strftime("%Y-%m-%d")
                end_date = pd.to_datetime(df["Datetime"].iloc[-1]).strftime("%Y-%m-%d")
                days = len(df)
                self.date_range_label.setText(
                    f"✅ Loaded {ticker}: {start_date} to {end_date} ({days} days)"
                )
                self.date_range_label.setStyleSheet(
                    f"color: {COLOR_SUCCESS}; font-weight: bold; padding: 5px;"
                )

            # Update chart
            self.on_timeframe_changed()

            # Initialize regime detector
            if "daily" in self.df_dict:
                self.regime_detector = MarketRegimeDetector(self.df_dict["daily"])

            self.statusBar().showMessage(f"Successfully loaded {ticker}")

        except Exception as e:
            QMessageBox.critical(
                self, "Data Loading Error", f"Failed to load {ticker}:\n{str(e)}"
            )
            self.statusBar().showMessage("Error loading data")

    def load_ticker_list(self):
        """Load multiple tickers from CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Ticker List CSV", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Read CSV
            df = pd.read_csv(file_path)
            if "ticker" not in df.columns:
                QMessageBox.critical(
                    self, "CSV Error", "CSV must have a 'ticker' column"
                )
                return

            tickers = df["ticker"].tolist()

            # Show confirmation
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setText(f"Load and optimize {len(tickers)} tickers?")
            msg.setInformativeText("This may take a while...")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if msg.exec() != QMessageBox.StandardButton.Yes:
                return

            # Load each ticker
            for ticker in tickers:
                try:
                    self.statusBar().showMessage(f"Loading {ticker}...")
                    self._load_and_optimize_ticker(ticker)
                except Exception as e:
                    print(f"Error loading {ticker}: {e}")

            self.statusBar().showMessage(
                f"Completed batch loading {len(tickers)} tickers"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Batch Load Error", f"Failed to load ticker list:\n{str(e)}"
            )

    def _load_and_optimize_ticker(self, ticker: str):
        """Load single ticker and run optimization"""
        df_dict, error = DataLoader.load_yfinance_data(ticker)

        if error:
            print(f"Error loading {ticker}: {error}")
            return

        self.df_dict_full = df_dict
        self._filter_timeframes()
        self.current_ticker = ticker
        self.start_optimization()

    def _filter_timeframes(self):
        """Filter data to selected timeframes"""
        if not self.df_dict_full:
            return

        self.df_dict = {}
        for tf_name, cb in self.tf_checkboxes.items():
            if cb.isChecked() and tf_name in self.df_dict_full:
                self.df_dict[tf_name] = self.df_dict_full[tf_name]

    def on_timeframe_changed(self):
        """Handle timeframe checkbox changes"""
        self._filter_timeframes()

        # Update chart with first selected timeframe
        for tf_name, cb in self.tf_checkboxes.items():
            if cb.isChecked() and tf_name in self.df_dict:
                df = self.df_dict[tf_name]
                self._update_chart(df["Close"].values, tf_name)
                break

    def _update_chart(self, close_arr: np.ndarray, timeframe: str):
        """Update price chart"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor="#1e1e1e")

        ax.plot(
            close_arr,
            color="#4CAF50",
            linewidth=2,
            label=f"{self.current_ticker} ({timeframe})",
        )
        ax.set_title(
            f"{self.current_ticker} Price Chart",
            color="white",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Index", color="white")
        ax.set_ylabel("Price", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#2d2d2d", edgecolor="#3a3a3a", labelcolor="white")
        ax.grid(True, alpha=0.2, color="white")

        self.canvas.draw()

    # =============================================================================
    # OPTIMIZATION METHODS
    # =============================================================================

    def start_optimization(self):
        """Start PSR optimization"""
        if not self.df_dict:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self, "Already Running", "Optimization is already running"
            )
            return

        # Get parameters
        params = {
            "mn1": (self.mn1_min.value(), self.mn1_max.value()),
            "mn2": (self.mn2_min.value(), self.mn2_max.value()),
            "entry": (self.entry_min.value(), self.entry_max.value()),
            "exit": (self.exit_min.value(), self.exit_max.value()),
            "on": (self.on_min.value(), self.on_max.value()),
            "off": (self.off_min.value(), self.off_max.value()),
        }

        trials = self.trials_spin.value()
        batch_size = self.batch_spin.value()
        objective = self.objective_combo.currentText()

        # Create worker
        self.worker = MultiTimeframeOptimizer(
            df_dict=self.df_dict,
            indicator_ranges=params,
            n_trials=trials,
            batch_size=batch_size,
            objective=objective,
            transaction_costs=self.transaction_costs,
        )

        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.new_best.connect(self.update_best_label)
        self.worker.phase_update.connect(self.update_phase_label)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Starting optimization...")

        # Start
        self.worker.start()
        self.statusBar().showMessage("Optimization running...")

    def stop_optimization(self):
        """Stop running optimization"""
        if self.worker:
            self.worker.stop()
            self.phase_label.setText("Stopping...")
            self.stop_btn.setEnabled(False)

    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_best_label(self, best_params: Dict):
        """Update best parameters display"""
        self.best_params = best_params

        # Update metrics
        psr = best_params.get("psr", 0.0)
        sharpe = best_params.get("sharpe_ratio", 0.0)
        sortino = best_params.get("sortino_ratio", 0.0)

        self.psr_label.setText(f"{psr:.3f}")
        self.sharpe_label.setText(f"{sharpe:.3f}")
        self.sortino_label.setText(f"{sortino:.3f}")

        # Update full display
        text = f"Best Parameters Found:\n"
        text += f"PSR: {psr:.3f} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f}\n"
        text += f"MN1: {best_params.get('mn1', 0)} | MN2: {best_params.get('mn2', 0)}\n"
        text += f"Entry: {best_params.get('entry', 0):.2f} | Exit: {best_params.get('exit', 0):.2f}\n"
        text += f"On: {best_params.get('on', 0)} | Off: {best_params.get('off', 0)}"

        self.best_params_label.setText(text)

    def update_phase_label(self, phase_text: str):
        """Update phase info"""
        self.phase_label.setText(phase_text)

    def show_results(self, df_results: pd.DataFrame):
        """Show optimization results"""
        self.best_results = df_results

        # Re-enable buttons
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.phase_label.setText("Optimization complete!")

        # Plot results
        if not df_results.empty:
            best = df_results.iloc[0]
            self._plot_results(best)

            # Store trade log for Monte Carlo
            if "trade_log" in best:
                self.last_trade_log = best["trade_log"]

        self.statusBar().showMessage("Optimization completed successfully")

    def _plot_results(self, best: pd.Series):
        """Plot equity curve with buy/sell signals"""
        self.results_figure.clear()

        if "equity_curve" not in best or best["equity_curve"] is None:
            return

        equity = best["equity_curve"]
        trade_log = best.get("trade_log")

        # Main plot
        ax1 = self.results_figure.add_subplot(211, facecolor="#1e1e1e")
        ax1.plot(equity, color="#4CAF50", linewidth=2, label="Equity")
        ax1.set_title("Equity Curve", color="white", fontweight="bold")
        ax1.set_ylabel("Equity", color="white")
        ax1.tick_params(colors="white")
        ax1.grid(True, alpha=0.2, color="white")
        ax1.legend(facecolor="#2d2d2d", edgecolor="#3a3a3a", labelcolor="white")

        # Add buy/sell markers if we have trade log
        if trade_log is not None and not trade_log.empty:
            buys = trade_log[trade_log["side"] == "buy"]
            sells = trade_log[trade_log["side"] == "sell"]

            if not buys.empty:
                ax1.scatter(
                    buys.index,
                    equity[buys.index],
                    color="#2196F3",
                    marker="^",
                    s=100,
                    label="Buy",
                    zorder=5,
                )
            if not sells.empty:
                ax1.scatter(
                    sells.index,
                    equity[sells.index],
                    color="#F44336",
                    marker="v",
                    s=100,
                    label="Sell",
                    zorder=5,
                )

        # Drawdown plot
        ax2 = self.results_figure.add_subplot(212, facecolor="#1e1e1e")
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#F44336", alpha=0.3)
        ax2.plot(drawdown, color="#F44336", linewidth=1)
        ax2.set_title("Drawdown %", color="white", fontweight="bold")
        ax2.set_xlabel("Time", color="white")
        ax2.set_ylabel("Drawdown %", color="white")
        ax2.tick_params(colors="white")
        ax2.grid(True, alpha=0.2, color="white")

        self.results_figure.tight_layout()
        self.results_canvas.draw()

    def show_psr_report(self):
        """Show detailed PSR report"""
        if not self.best_params:
            QMessageBox.information(self, "No Results", "Run optimization first")
            return

        report = f"""
=== PSR COMPOSITE REPORT ===

PSR: {self.best_params.get('psr', 0.0):.4f}

Performance Metrics:
  Sharpe Ratio: {self.best_params.get('sharpe_ratio', 0.0):.4f}
  Sortino Ratio: {self.best_params.get('sortino_ratio', 0.0):.4f}
  Calmar Ratio: {self.best_params.get('calmar_ratio', 0.0):.4f}

Risk Metrics:
  Max Drawdown: {self.best_params.get('max_drawdown', 0.0):.2f}%
  Profit Factor: {self.best_params.get('profit_factor', 0.0):.4f}

Trading Metrics:
  Total Trades: {self.best_params.get('total_trades', 0)}
  Win Rate: {self.best_params.get('win_rate', 0.0):.2f}%
  Annual Turnover: {self.best_params.get('annual_turnover', 0.0):.2f}

Parameters:
  MN1: {self.best_params.get('mn1', 0)}
  MN2: {self.best_params.get('mn2', 0)}
  Entry: {self.best_params.get('entry', 0.0):.2f}
  Exit: {self.best_params.get('exit', 0.0):.2f}
  On Cycle: {self.best_params.get('on', 0)}
  Off Cycle: {self.best_params.get('off', 0)}
"""

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("PSR Composite Report")
        msg.setText(report)
        msg.setStyleSheet("QLabel{min-width: 500px; font-family: monospace;}")
        msg.exec()

    def show_error(self, error_msg: str):
        """Show error message"""
        QMessageBox.critical(self, "Error", error_msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.phase_label.setText("Error occurred")
        self.statusBar().showMessage("Error")

    # =============================================================================
    # MONTE CARLO
    # =============================================================================

    def run_monte_carlo(self):
        """Run Monte Carlo simulation"""
        if not self.last_trade_log or self.last_trade_log.empty:
            QMessageBox.warning(
                self, "No Trade Log", "Run optimization first to get trade log"
            )
            return

        if len(self.last_trade_log) < 10:
            QMessageBox.warning(
                self, "Insufficient Trades", "Need at least 10 trades for Monte Carlo"
            )
            return

        self.statusBar().showMessage("Running Monte Carlo simulation...")

        try:
            from core.monte_carlo import MonteCarloAnalyzer

            n_sims = self.mc_simulations_spin.value()

            analyzer = MonteCarloAnalyzer(self.last_trade_log)
            results = analyzer.run_all_methods(n_simulations=n_sims)

            # Show summary
            msg = f"""
Monte Carlo Results ({n_sims} simulations):

Trade Randomization:
  Mean Sharpe: {results['trade_randomization']['mean_sharpe']:.3f}
  95% CI: [{results['trade_randomization']['ci_lower']:.3f}, {results['trade_randomization']['ci_upper']:.3f}]

Bootstrap:
  Mean Sharpe: {results['bootstrap']['mean_sharpe']:.3f}
  95% CI: [{results['bootstrap']['ci_lower']:.3f}, {results['bootstrap']['ci_upper']:.3f}]

Parametric:
  Mean Sharpe: {results['parametric']['mean_sharpe']:.3f}
  95% CI: [{results['parametric']['ci_lower']:.3f}, {results['parametric']['ci_upper']:.3f}]

Block Bootstrap:
  Mean Sharpe: {results['block_bootstrap']['mean_sharpe']:.3f}
  95% CI: [{results['block_bootstrap']['ci_lower']:.3f}, {results['block_bootstrap']['ci_upper']:.3f}]
"""

            QMessageBox.information(self, "Monte Carlo Results", msg)
            self.statusBar().showMessage("Monte Carlo completed")

        except Exception as e:
            QMessageBox.critical(
                self, "Monte Carlo Error", f"Failed to run Monte Carlo:\n{str(e)}"
            )
            self.statusBar().showMessage("Monte Carlo failed")

    # =============================================================================
    # WALK-FORWARD
    # =============================================================================

    def run_walk_forward(self):
        """Run walk-forward analysis"""
        if not self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        train_days = self.wf_train_days_spin.value()
        test_days = self.wf_test_days_spin.value()
        trials = self.wf_trials_spin.value()

        self.statusBar().showMessage("Running walk-forward analysis...")

        try:
            from core.walk_forward import WalkForwardAnalyzer

            params = {
                "mn1": (self.mn1_min.value(), self.mn1_max.value()),
                "mn2": (self.mn2_min.value(), self.mn2_max.value()),
                "entry": (self.entry_min.value(), self.entry_max.value()),
                "exit": (self.exit_min.value(), self.exit_max.value()),
                "on": (self.on_min.value(), self.on_max.value()),
                "off": (self.off_min.value(), self.off_max.value()),
            }

            analyzer = WalkForwardAnalyzer(
                df_dict=self.df_dict,
                indicator_ranges=params,
                train_days=train_days,
                test_days=test_days,
                trials_per_window=trials,
                transaction_costs=self.transaction_costs,
            )

            results = analyzer.run()

            # Show summary
            msg = f"""
Walk-Forward Analysis Results:

Windows: {len(results.windows)}
Efficiency: {results.efficiency:.2%}

In-Sample Sharpe: {results.in_sample_sharpe:.3f}
Out-Sample Sharpe: {results.out_sample_sharpe:.3f}

Total Return: {results.total_return:.2%}
Max Drawdown: {results.max_drawdown:.2%}

Plots saved to: {Paths.RESULTS_DIR}/walk_forward/
"""

            QMessageBox.information(self, "Walk-Forward Results", msg)
            self.statusBar().showMessage("Walk-forward analysis completed")

        except Exception as e:
            QMessageBox.critical(
                self, "Walk-Forward Error", f"Failed to run walk-forward:\n{str(e)}"
            )
            self.statusBar().showMessage("Walk-forward failed")

    # =============================================================================
    # REGIME ANALYSIS
    # =============================================================================

    def detect_market_regime(self):
        """Detect current market regime"""
        if not self.regime_detector:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        try:
            regime_state = self.regime_detector.detect_regime()
            self.current_regime_state = regime_state

            text = f"""
Current Market Regime:

Regime: {regime_state.regime_type.upper()}
Confidence: {regime_state.confidence:.2%}

Volatility: {regime_state.volatility:.2%}
Trend Strength: {regime_state.trend_strength:.2f}
Mean Return: {regime_state.mean_return:.2%}
"""

            self.regime_display.setText(text)
            self.statusBar().showMessage("Regime detected")

        except Exception as e:
            QMessageBox.critical(self, "Regime Detection Error", str(e))

    def train_regime_predictor(self):
        """Train ML regime predictor"""
        if not self.regime_detector:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        self.statusBar().showMessage("Training regime predictor...")

        try:
            self.regime_predictor = RegimePredictor()

            # Get historical regimes
            df = self.df_dict["daily"]
            X, y = self.regime_detector.get_historical_features_and_labels(df)

            # Train
            results = self.regime_predictor.train(X, y)

            # Predict current
            current_features = self.regime_detector.get_current_features()
            prediction = self.regime_predictor.predict(current_features)

            text = f"""
Regime Predictor Trained:

Accuracy: {results['accuracy']:.2%}
Precision: {results['precision']:.2%}
Recall: {results['recall']:.2%}
F1 Score: {results['f1']:.2%}

Predicted Next Regime: {prediction['regime'].upper()}
Confidence: {prediction['confidence']:.2%}
"""

            self.prediction_display.setText(text)
            self.statusBar().showMessage("Predictor trained successfully")

        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
            self.statusBar().showMessage("Training failed")

    def calculate_pbr(self):
        """Calculate Probability of Backtested Returns"""
        if not self.best_results or self.best_results.empty:
            QMessageBox.warning(self, "No Results", "Run optimization first")
            return

        try:
            from core.pbr import PBRCalculator

            calculator = PBRCalculator()
            pbr_score = calculator.calculate(
                backtest_results=self.best_results.iloc[0],
                regime_state=self.current_regime_state,
            )

            self.pbr_display.setText(f"PBR Score: {pbr_score:.2%}")

            # Add to institutional display
            text = f"PBR (Probability of Backtested Returns): {pbr_score:.2%}\n"
            text += "Interpretation: Likelihood that backtest results are achievable in live trading\n"
            text += f"Status: {'HIGH CONFIDENCE' if pbr_score > 0.7 else 'MEDIUM CONFIDENCE' if pbr_score > 0.5 else 'LOW CONFIDENCE'}"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("PBR calculated")

        except Exception as e:
            QMessageBox.critical(self, "PBR Error", str(e))

    def calibrate_probabilities(self):
        """Calibrate regime prediction probabilities"""
        if not self.regime_predictor:
            QMessageBox.warning(self, "No Predictor", "Train predictor first")
            return

        self.statusBar().showMessage("Calibrating probabilities...")

        try:
            self.regime_calibrator = MultiClassCalibrator()

            # Get predictions and true labels
            df = self.df_dict["daily"]
            X, y_true = self.regime_detector.get_historical_features_and_labels(df)
            y_pred_proba = self.regime_predictor.predict_proba(X)

            # Fit calibrator
            self.regime_calibrator.fit(y_pred_proba, y_true)

            # Get calibrated predictions
            y_calibrated = self.regime_calibrator.predict_proba(y_pred_proba)
            self.calibrated_predictions = y_calibrated

            text = "Probability Calibration Complete\n"
            text += "Method: Isotonic Regression\n"
            text += "Predictions are now calibrated for accurate probability estimates"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Calibration complete")

        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", str(e))
            self.statusBar().showMessage("Calibration failed")

    def check_multi_horizon_agreement(self):
        """Check multi-horizon prediction agreement"""
        if not self.regime_predictor:
            QMessageBox.warning(self, "No Predictor", "Train predictor first")
            return

        self.statusBar().showMessage("Checking multi-horizon agreement...")

        try:
            self.multi_horizon_agreement = MultiHorizonAgreementIndex()

            # Get predictions for different horizons
            horizons = [1, 5, 10, 20]  # days
            predictions = {}

            current_features = self.regime_detector.get_current_features()

            for h in horizons:
                pred = self.regime_predictor.predict_horizon(
                    current_features, horizon=h
                )
                predictions[f"{h}d"] = pred

            # Calculate agreement
            agreement_score = self.multi_horizon_agreement.calculate(predictions)

            text = f"Multi-Horizon Agreement Index: {agreement_score:.2%}\n\n"
            text += "Predictions by Horizon:\n"
            for horizon, pred in predictions.items():
                text += f"  {horizon}: {pred['regime'].upper()} ({pred['confidence']:.1%})\n"
            text += f"\nConsensus: {'STRONG' if agreement_score > 0.8 else 'MODERATE' if agreement_score > 0.6 else 'WEAK'}"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Multi-horizon check complete")

        except Exception as e:
            QMessageBox.critical(self, "Agreement Error", str(e))
            self.statusBar().showMessage("Agreement check failed")

    def run_robustness_tests(self):
        """Run White's Reality Check and Hansen's SPA"""
        if not self.best_results or self.best_results.empty:
            QMessageBox.warning(self, "No Results", "Run optimization first")
            return

        self.statusBar().showMessage("Running robustness tests...")

        try:
            from core.robustness import RobustnessAnalyzer

            analyzer = RobustnessAnalyzer()
            results = analyzer.run_tests(
                strategy_returns=self.best_results.iloc[0]["returns"],
                n_bootstrap=500,  # Reduced for GUI performance
            )

            text = "Robustness Tests:\n\n"
            text += f"White's Reality Check:\n"
            text += f"  p-value: {results['wrc_pvalue']:.4f}\n"
            text += (
                f"  Result: {'PASS' if results['wrc_pvalue'] < 0.05 else 'FAIL'}\n\n"
            )
            text += f"Hansen's SPA Test:\n"
            text += f"  p-value: {results['spa_pvalue']:.4f}\n"
            text += (
                f"  Result: {'PASS' if results['spa_pvalue'] < 0.05 else 'FAIL'}\n\n"
            )
            text += f"Sharpe Ratio 95% CI: [{results['sharpe_ci_lower']:.3f}, {results['sharpe_ci_upper']:.3f}]"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Robustness tests complete")

        except Exception as e:
            QMessageBox.critical(self, "Robustness Error", str(e))
            self.statusBar().showMessage("Robustness tests failed")

    def run_regime_diagnostics(self):
        """Run regime stability and persistence diagnostics"""
        if not self.regime_detector:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        self.statusBar().showMessage("Running regime diagnostics...")

        try:
            self.regime_diagnostics = RegimeDiagnosticAnalyzer(self.regime_detector)

            results = self.regime_diagnostics.analyze()

            text = "Regime Diagnostic Analysis:\n\n"
            text += f"Regime Stability: {results['stability']:.2%}\n"
            text += f"Average Persistence: {results['avg_persistence']:.1f} days\n"
            text += (
                f"Transition Frequency: {results['transition_freq']:.1f} per month\n\n"
            )
            text += "Regime Distribution:\n"
            for regime, pct in results["distribution"].items():
                text += f"  {regime.upper()}: {pct:.1%}\n"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Diagnostics complete")

        except Exception as e:
            QMessageBox.critical(self, "Diagnostics Error", str(e))
            self.statusBar().showMessage("Diagnostics failed")

    def run_cross_asset_analysis(self):
        """Run cross-asset regime analysis"""
        self.statusBar().showMessage(
            "Running cross-asset analysis (loading SPY, TLT, GLD, BTC)..."
        )

        try:
            self.cross_asset_analyzer = CrossAssetRegimeAnalyzer()

            # This will load and analyze multiple assets
            results = self.cross_asset_analyzer.analyze_global_regime()

            text = "Cross-Asset Regime Analysis:\n\n"
            text += f"Global Regime: {results['global_regime'].upper()}\n"
            text += f"Agreement Score: {results['agreement']:.2%}\n\n"
            text += "Individual Assets:\n"
            for asset, regime in results["asset_regimes"].items():
                text += f"  {asset}: {regime.upper()}\n"
            text += f"\nCorrelation Regime: {results['correlation_regime']}"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Cross-asset analysis complete")

        except Exception as e:
            QMessageBox.critical(self, "Cross-Asset Error", str(e))
            self.statusBar().showMessage("Cross-asset analysis failed")

    # =============================================================================
    # LIVE TRADING
    # =============================================================================

    def toggle_live_trading(self):
        """Toggle live trading on/off"""
        if self.live_trader and self.live_trader.is_running:
            self.stop_live_trading()
        else:
            self.start_live_trading()

    def start_live_trading(self):
        """Start live trading"""
        if not self.best_params:
            QMessageBox.warning(
                self, "No Strategy", "Run optimization first to get strategy parameters"
            )
            return

        # Get API credentials
        api_key = self.api_key_input.text().strip()
        api_secret = self.api_secret_input.text().strip()

        if not api_key or not api_secret:
            QMessageBox.warning(
                self, "Missing Credentials", "Please enter API key and secret"
            )
            return

        # Confirm
        paper = self.paper_trading_cb.isChecked()
        mode = "PAPER" if paper else "LIVE"

        msg = QMessageBox()
        msg.setIcon(
            QMessageBox.Icon.Warning if not paper else QMessageBox.Icon.Question
        )
        msg.setWindowTitle("Confirm Trading")
        msg.setText(f"Start {mode} trading?")
        msg.setInformativeText("Make sure you understand the risks involved.")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if msg.exec() != QMessageBox.StandardButton.Yes:
            return

        try:
            # Create live trader
            self.live_trader = AlpacaLiveTrader(
                api_key=api_key,
                api_secret=api_secret,
                paper=paper,
                strategy_params=self.best_params,
                position_size_pct=self.position_size_pct,
                max_positions=self.max_positions,
            )

            # Connect signals
            self.live_trader.status_update.connect(self.update_trading_status)
            self.live_trader.trade_executed.connect(self.on_trade_executed)
            self.live_trader.error.connect(self.on_trading_error)

            # Start
            self.live_trader.start()

            # Update UI
            self.live_trading_btn.setText("⏹️ STOP LIVE TRADING")
            self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_ACTIVE)
            self.trading_status_label.setText(f"Status: {mode} TRADING ACTIVE")
            self.trading_status_label.setStyleSheet(
                f"padding: 15px; background-color: {COLOR_SUCCESS}; color: white; border-radius: 4px; font-size: 14px; font-weight: bold;"
            )

            self.statusBar().showMessage(f"{mode} trading started")

        except Exception as e:
            QMessageBox.critical(
                self, "Trading Error", f"Failed to start trading:\n{str(e)}"
            )

    def stop_live_trading(self):
        """Stop live trading"""
        if not self.live_trader:
            return

        try:
            self.live_trader.stop()

            # Update UI
            self.live_trading_btn.setText("▶️ START LIVE TRADING")
            self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_STOPPED)
            self.trading_status_label.setText("Status: Not Running")
            self.trading_status_label.setStyleSheet(
                "padding: 15px; background-color: #2d2d2d; border-radius: 4px; font-size: 14px;"
            )

            self.statusBar().showMessage("Trading stopped")

        except Exception as e:
            QMessageBox.critical(
                self, "Stop Error", f"Failed to stop trading:\n{str(e)}"
            )

    def update_trading_status(self, status: str):
        """Update trading status display"""
        self.trading_status_label.setText(f"Status: {status}")

    def on_trade_executed(self, trade_info: Dict):
        """Handle trade execution notification"""
        text = f"[{trade_info['timestamp']}] {trade_info['side'].upper()} {trade_info['qty']} {trade_info['symbol']} @ ${trade_info['price']:.2f}\n"
        self.trade_log_display.append(text)

    def on_trading_error(self, error_msg: str):
        """Handle trading error"""
        self.trade_log_display.append(f"ERROR: {error_msg}\n")
        QMessageBox.critical(self, "Trading Error", error_msg)

    # =============================================================================
    # SETTINGS
    # =============================================================================

    def on_risk_settings_changed(self):
        """Handle risk settings changes"""
        self.position_size_pct = self.position_size_spin.value()
        self.max_positions = self.max_positions_spin.value()

        # Check for over-leverage
        total_exposure = self.position_size_pct * self.max_positions
        if total_exposure > 100:
            self.leverage_warning_label.setText(
                f"⚠️ WARNING: Total exposure is {total_exposure:.0f}% (using leverage)"
            )
        else:
            self.leverage_warning_label.setText("")

    def on_transaction_costs_changed(self):
        """Handle transaction cost changes"""
        self.transaction_costs.commission_pct = self.commission_pct_spin.value() / 100
        self.transaction_costs.slippage_pct = self.slippage_pct_spin.value() / 100
        self.transaction_costs.spread_pct = self.spread_pct_spin.value() / 100

    def set_costs_for_stocks(self):
        """Set transaction costs for stocks"""
        self.commission_pct_spin.setValue(0.05)
        self.slippage_pct_spin.setValue(0.05)
        self.spread_pct_spin.setValue(0.01)

    def set_costs_for_crypto(self):
        """Set transaction costs for crypto"""
        self.commission_pct_spin.setValue(0.1)
        self.slippage_pct_spin.setValue(0.2)
        self.spread_pct_spin.setValue(0.05)

    def set_costs_zero(self):
        """Zero out transaction costs"""
        self.commission_pct_spin.setValue(0.0)
        self.slippage_pct_spin.setValue(0.0)
        self.spread_pct_spin.setValue(0.0)

    # =============================================================================
    # CLEANUP
    # =============================================================================

    def closeEvent(self, event):
        """Handle window close"""
        # Stop worker threads
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        # Stop live trading
        if self.live_trader and self.live_trader.is_running:
            self.live_trader.stop()

        event.accept()
