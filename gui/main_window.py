"""
LiquidUI Trading Platform - Main Window

Professional multi-tab GUI. Tabs and features that are hidden by default can
be re-enabled with the feature flags below.
"""

from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.settings import (
    IndicatorRanges,
    OptimizationConfig,
    RiskConfig,
    TransactionCosts,
)
from data import DataLoader
from gui.styles import (
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    LIVE_TRADING_BUTTON_ACTIVE,
    LIVE_TRADING_BUTTON_STOPPED,
    MAIN_STYLESHEET,
)
from models.bayesian_changepoint import (
    BOCPDDetector,
    RegimeStrategyAnalyzer,
    online_regime_labels,
    plot_regimes,
)
from models.regime_agreement import HorizonPrediction, MultiHorizonAgreementIndex
from models.regime_calibration import MultiClassCalibrator
from models.regime_detection import MarketRegime, MarketRegimeDetector, PBRCalculator
from models.regime_diagnostics import RegimeDiagnosticAnalyzer
from models.regime_predictor import RegimePredictor
from models.regime_robustness import BlockBootstrapValidator, WhiteRealityCheck
from optimization import MultiTimeframeOptimizer, WalkForwardAnalyzer
from optimization.monte_carlo import AdvancedMonteCarloAnalyzer, MonteCarloSimulator
from trading import AlpacaLiveTrader
from trading.regime_switcher import (
    StoredStrategy,
    StrategyBook,
    build_switching_report,
    plot_switching_backtest,
    run_switching_backtest,
)

# Feature flags - set to True to re-enable hidden features
SHOW_REGIME_TAB = False
SHOW_LIVE_TRADING_TAB = False
SHOW_MONTE_CARLO = False
SHOW_WALK_FORWARD = False

# All timeframes the platform understands, ordered coarse -> fine
TIMEFRAMES = [("daily", "Daily"), ("hourly", "Hourly"), ("5min", "5-Minute")]


class MainWindow(QMainWindow):
    """Professional multi-tab trading platform GUI"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidUI - Professional Trading Platform")
        self.setGeometry(100, 100, 1600, 1000)

        # Data storage
        self.df_dict: Dict[str, pd.DataFrame] = {}
        self.df_dict_full: Dict[str, pd.DataFrame] = {}
        self.current_ticker: str = ""
        self.last_trade_log: Optional[pd.DataFrame] = None
        self.last_equity_curve: Optional[np.ndarray] = None
        self.best_params: Optional[Dict] = None
        self.best_results: Optional[pd.DataFrame] = None

        # Batch processing queue (tickers waiting for optimization)
        self._batch_queue: List[str] = []

        # Worker threads
        self.worker: Optional[MultiTimeframeOptimizer] = None

        # Regime detection objects
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.regime_predictor: Optional[RegimePredictor] = None
        self.regime_calibrator: Optional[MultiClassCalibrator] = None
        self.current_regime_state = None

        # Live trading
        self.live_trader: Optional[AlpacaLiveTrader] = None

        # Settings (position size is stored as a percentage, e.g. 5.0 = 5%)
        self.position_size_pct = RiskConfig.DEFAULT_POSITION_SIZE * 100
        self.max_positions = RiskConfig.DEFAULT_MAX_POSITIONS
        # Default to realistic stock costs - optimizing with zero costs
        # systematically selects high-turnover parameters that lose after
        # real-world fees/slippage
        self.transaction_costs = TransactionCosts.for_stocks()

        self.init_ui()

    def init_ui(self):
        """Initialize the tabbed interface"""
        self.setStyleSheet(MAIN_STYLESHEET)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
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
        """)

        self.tabs.addTab(self._create_data_tab(), "📊 Data & Setup")
        self.tabs.addTab(self._create_strategy_tab(), "⚙️ Strategy Optimization")
        if SHOW_REGIME_TAB:
            self.tabs.addTab(self._create_regime_tab(), "🏛️ Regime Analysis")
        if SHOW_LIVE_TRADING_TAB:
            self.tabs.addTab(self._create_trading_tab(), "🔴 Live Trading")
        self.tabs.addTab(self._create_settings_tab(), "⚙️ Settings")

        main_layout.addWidget(self.tabs)

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

        ticker_layout = QHBoxLayout()
        ticker_layout.addWidget(QLabel("Ticker Symbol:"))
        self.ticker_input = QLineEdit("SPY")
        self.ticker_input.setPlaceholderText("Enter ticker (e.g., SPY, AAPL, BTC-USD)")
        self.ticker_input.setMaximumWidth(200)
        self.ticker_input.returnPressed.connect(self.load_yfinance)
        ticker_layout.addWidget(self.ticker_input)

        self.load_yf_btn = QPushButton("📈 Load from Yahoo Finance")
        self.load_yf_btn.setStyleSheet(
            f"background-color: {COLOR_SUCCESS}; font-weight: bold; padding: 10px;"
        )
        self.load_yf_btn.clicked.connect(self.load_yfinance)
        ticker_layout.addWidget(self.load_yf_btn)
        ticker_layout.addStretch()
        data_layout.addLayout(ticker_layout)

        batch_layout = QHBoxLayout()
        self.batch_ticker_btn = QPushButton("📁 Load Multiple Tickers from CSV")
        self.batch_ticker_btn.setToolTip(
            "CSV must contain a 'ticker' column. Each ticker is loaded and "
            "optimized sequentially with the current settings."
        )
        self.batch_ticker_btn.clicked.connect(self.load_ticker_list)
        batch_layout.addWidget(self.batch_ticker_btn)
        batch_layout.addStretch()
        data_layout.addLayout(batch_layout)

        self.date_range_label = QLabel("No data loaded")
        self.date_range_label.setStyleSheet("color: #888; font-style: italic; padding: 5px;")
        data_layout.addWidget(self.date_range_label)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Timeframe selection
        tf_group = QGroupBox("Select Timeframes")
        tf_layout = QHBoxLayout()

        self.tf_checkboxes = {}
        for tf_name, tf_label in TIMEFRAMES:
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

        params_group = QGroupBox("Strategy Parameter Ranges")
        params_layout = QVBoxLayout()

        self._add_param_range(params_layout, "MN1 (RSI1 Length)", "mn1")
        self._add_param_range(params_layout, "MN2 (RSI2 Length)", "mn2")
        self._add_param_range(params_layout, "Entry Threshold", "entry", is_decimal=True)
        self._add_param_range(params_layout, "Exit Threshold", "exit", is_decimal=True)
        self._add_param_range(params_layout, "On Cycle", "on")
        self._add_param_range(params_layout, "Off Cycle", "off")

        start_note = QLabel("Note: START cycle range is auto-calculated as 0 to (ON + OFF)")
        start_note.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        start_note.setWordWrap(True)
        params_layout.addWidget(start_note)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Optimization settings
        opt_group = QGroupBox("Optimization Settings")
        opt_layout = QVBoxLayout()

        trials_layout = QHBoxLayout()
        trials_layout.addWidget(QLabel("Trials:"))
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(300, 100000)
        self.trials_spin.setValue(OptimizationConfig.DEFAULT_TRIALS)
        self.trials_spin.setSingleStep(100)
        trials_layout.addWidget(self.trials_spin)
        trials_layout.addStretch()
        opt_layout.addLayout(trials_layout)

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(50, 2000)
        self.batch_spin.setValue(OptimizationConfig.DEFAULT_BATCH_SIZE)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        opt_layout.addLayout(batch_layout)

        opt_group.setLayout(opt_layout)
        left_layout.addWidget(opt_group)

        # Action buttons
        btn_group = QGroupBox("Actions")
        btn_layout = QVBoxLayout()

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

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)

        self.phase_label = QLabel("Ready to optimize")
        self.phase_label.setStyleSheet("color: #888; font-style: italic;")
        btn_layout.addWidget(self.phase_label)

        if SHOW_MONTE_CARLO:
            self.monte_carlo_btn = QPushButton("🎲 Run Monte Carlo Simulation")
            self.monte_carlo_btn.clicked.connect(self.run_monte_carlo)
            btn_layout.addWidget(self.monte_carlo_btn)

            mc_layout = QHBoxLayout()
            mc_layout.addWidget(QLabel("MC Simulations:"))
            self.mc_simulations_spin = QSpinBox()
            self.mc_simulations_spin.setRange(100, 10000)
            self.mc_simulations_spin.setValue(1000)
            mc_layout.addWidget(self.mc_simulations_spin)
            mc_layout.addStretch()
            btn_layout.addLayout(mc_layout)

        if SHOW_WALK_FORWARD:
            self.walk_forward_btn = QPushButton("📈 Run Walk-Forward Analysis")
            self.walk_forward_btn.clicked.connect(self.run_walk_forward)
            btn_layout.addWidget(self.walk_forward_btn)

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

        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout()

        self.best_params_label = QLabel("No optimization run yet")
        self.best_params_label.setWordWrap(True)
        self.best_params_label.setStyleSheet(
            "padding: 10px; background-color: #2d2d2d; border-radius: 4px;"
        )
        results_layout.addWidget(self.best_params_label)

        metrics_layout = QHBoxLayout()
        self.psr_label = self._add_metric_box(metrics_layout, "PSR", large=True)
        self.sharpe_label = self._add_metric_box(metrics_layout, "Sharpe")
        self.sortino_label = self._add_metric_box(metrics_layout, "Sortino")
        results_layout.addLayout(metrics_layout)

        self.show_report_btn = QPushButton("📊 Show Full PSR Report")
        self.show_report_btn.clicked.connect(self.show_psr_report)
        results_layout.addWidget(self.show_report_btn)

        self.regime_breakdown_btn = QPushButton("🔀 Regime Breakdown (Bayesian CPD)")
        self.regime_breakdown_btn.clicked.connect(self.show_regime_breakdown)
        results_layout.addWidget(self.regime_breakdown_btn)

        book_layout = QHBoxLayout()
        self.save_to_book_btn = QPushButton("💾 Save Strategy to Book")
        self.save_to_book_btn.clicked.connect(self.save_strategy_to_book)
        book_layout.addWidget(self.save_to_book_btn)

        self.switching_backtest_btn = QPushButton("🔁 Regime-Switching Backtest")
        self.switching_backtest_btn.clicked.connect(self.run_switching_backtest_gui)
        book_layout.addWidget(self.switching_backtest_btn)
        results_layout.addLayout(book_layout)

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

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)

        return tab

    @staticmethod
    def _add_metric_box(layout: QHBoxLayout, title: str, large: bool = False) -> QLabel:
        """Add a titled metric display box and return its value label"""
        box = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 11px; color: #888;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        value_label = QLabel("--")
        size = 24 if large else 20
        color = COLOR_SUCCESS if large else "#e0e0e0"
        value_label.setStyleSheet(
            f"font-size: {size}px; font-weight: bold; color: {color}; "
            "background-color: #2d2d2d; padding: 10px; border-radius: 4px;"
        )
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        box.addWidget(title_label)
        box.addWidget(value_label)
        layout.addLayout(box)
        return value_label

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

        lookback_layout = QHBoxLayout()
        lookback_label = QLabel("Lookback Days:")
        lookback_label.setStyleSheet("font-weight: bold;")
        self.regime_lookback_spin = QSpinBox()
        self.regime_lookback_spin.setRange(20, 1000)
        self.regime_lookback_spin.setValue(252)  # Default 1 year
        self.regime_lookback_spin.setSuffix(" days")
        self.regime_lookback_spin.setToolTip(
            "Number of historical days to analyze for regime detection (20-1000)"
        )
        lookback_layout.addWidget(lookback_label)
        lookback_layout.addWidget(self.regime_lookback_spin)
        lookback_layout.addStretch()
        basic_layout.addLayout(lookback_layout)

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

        row1 = QHBoxLayout()
        self.calc_pbr_btn = QPushButton("📊 Calculate PBR")
        self.calc_pbr_btn.clicked.connect(self.calculate_pbr)
        row1.addWidget(self.calc_pbr_btn)

        self.calibrate_btn = QPushButton("🎯 Calibrate Probabilities")
        self.calibrate_btn.clicked.connect(self.calibrate_probabilities)
        row1.addWidget(self.calibrate_btn)
        inst_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.agreement_btn = QPushButton("🔄 Multi-Horizon Agreement")
        self.agreement_btn.clicked.connect(self.check_multi_horizon_agreement)
        row2.addWidget(self.agreement_btn)

        self.robustness_btn = QPushButton("🛡️ Robustness Tests")
        self.robustness_btn.clicked.connect(self.run_robustness_tests)
        row2.addWidget(self.robustness_btn)
        inst_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.diagnostics_btn = QPushButton("🔬 Regime Diagnostics")
        self.diagnostics_btn.clicked.connect(self.run_regime_diagnostics)
        row3.addWidget(self.diagnostics_btn)

        self.cross_asset_btn = QPushButton("🌍 Cross-Asset Analysis")
        self.cross_asset_btn.clicked.connect(self.run_cross_asset_analysis)
        row3.addWidget(self.cross_asset_btn)
        inst_layout.addLayout(row3)

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

        self.paper_trading_cb = QCheckBox("Paper Trading (Recommended)")
        self.paper_trading_cb.setChecked(True)
        self.paper_trading_cb.setStyleSheet(f"color: {COLOR_SUCCESS}; font-weight: bold;")
        api_layout.addWidget(self.paper_trading_cb)

        self.use_book_cb = QCheckBox(
            "Use Strategy Book (regime switching - needs 2+ stored strategies)"
        )
        self.use_book_cb.setChecked(False)
        api_layout.addWidget(self.use_book_cb)

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
        self.trade_log_display.setPlaceholderText("Trade executions will appear here...")
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
        self.leverage_warning_label.setStyleSheet(f"color: {COLOR_WARNING}; font-weight: bold;")
        risk_layout.addWidget(self.leverage_warning_label)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

        # Transaction Costs
        tc_group = QGroupBox("Transaction Costs")
        tc_layout = QVBoxLayout()

        self.commission_pct_spin = self._add_cost_row(tc_layout, "Commission (%):")
        self.slippage_pct_spin = self._add_cost_row(tc_layout, "Slippage (%):")
        self.spread_pct_spin = self._add_cost_row(tc_layout, "Spread (%):")

        # Show the active (stocks-preset) defaults instead of zeros
        self.commission_pct_spin.setValue(self.transaction_costs.COMMISSION_PCT * 100)
        self.slippage_pct_spin.setValue(self.transaction_costs.SLIPPAGE_PCT * 100)
        self.spread_pct_spin.setValue(self.transaction_costs.SPREAD_PCT * 100)

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

    def _add_cost_row(self, layout: QVBoxLayout, label: str) -> QDoubleSpinBox:
        """Add a transaction cost spinbox row and return the spinbox"""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setSuffix("%")
        spin.valueChanged.connect(self.on_transaction_costs_changed)
        row.addWidget(spin)
        row.addStretch()
        layout.addLayout(row)
        return spin

    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    def _add_param_range(
        self, layout: QVBoxLayout, label: str, param_name: str, is_decimal: bool = False
    ):
        """Add parameter range controls"""
        defaults = {
            "mn1": IndicatorRanges.MN1_RANGE,
            "mn2": IndicatorRanges.MN2_RANGE,
            "entry": IndicatorRanges.ENTRY_RANGE,
            "exit": IndicatorRanges.EXIT_RANGE,
            "on": IndicatorRanges.ON_RANGE,
            "off": IndicatorRanges.OFF_RANGE,
        }

        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel(f"{label}:"))

        spins = []
        for bound in ("Min", "Max"):
            group_layout.addWidget(QLabel(f"{bound}:"))
            if is_decimal:
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 100.0)
                spin.setSingleStep(0.1)
                spin.setDecimals(2)
            else:
                spin = QSpinBox()
                spin.setRange(0, 500)
            setattr(self, f"{param_name}_{bound.lower()}", spin)
            group_layout.addWidget(spin)
            spins.append(spin)

        group_layout.addStretch()
        layout.addLayout(group_layout)

        if param_name in defaults:
            spins[0].setValue(defaults[param_name][0])
            spins[1].setValue(defaults[param_name][1])

    def _get_param_ranges(self) -> Dict:
        """Read and validate parameter ranges from the UI (min <= max)"""
        params = {}
        for name in ("mn1", "mn2", "entry", "exit", "on", "off"):
            low = getattr(self, f"{name}_min").value()
            high = getattr(self, f"{name}_max").value()
            if low > high:
                low, high = high, low
            params[name] = (low, high)
        return params

    @staticmethod
    def _get_tf_value(container, name: str, default="N/A"):
        """Look up a per-timeframe parameter (e.g. MN1_daily) in a dict/Series"""
        for tf in ("daily", "hourly", "5min"):
            value = container.get(f"{name}_{tf}")
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
        return default

    @staticmethod
    def _fmt(value, spec: str = ".1f") -> str:
        """Format numbers, pass through non-numeric values"""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return format(float(value), spec)
        return str(value)

    # =============================================================================
    # DATA LOADING METHODS
    # =============================================================================

    def load_yfinance(self):
        """Load single ticker from Yahoo Finance"""
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a ticker symbol")
            return

        self._load_ticker(ticker, interactive=True)

    def _load_ticker(self, ticker: str, interactive: bool = True) -> bool:
        """Load a ticker and refresh the UI. Returns True on success."""
        self.statusBar().showMessage(f"Loading {ticker}...")

        try:
            df_dict, error = DataLoader.load_yfinance_data(ticker)

            if error:
                if interactive:
                    QMessageBox.critical(
                        self, "Data Loading Error", f"Failed to load {ticker}:\n{error}"
                    )
                self.statusBar().showMessage(f"Error loading {ticker}: {error}")
                return False

            self.df_dict_full = df_dict
            self._filter_timeframes()
            self.current_ticker = ticker

            if "daily" in self.df_dict_full:
                df = self.df_dict_full["daily"]
                start_date = pd.to_datetime(df["Datetime"].iloc[0]).strftime("%Y-%m-%d")
                end_date = pd.to_datetime(df["Datetime"].iloc[-1]).strftime("%Y-%m-%d")
                self.date_range_label.setText(
                    f"✅ Loaded {ticker}: {start_date} to {end_date} ({len(df)} days)"
                )
                self.date_range_label.setStyleSheet(
                    f"color: {COLOR_SUCCESS}; font-weight: bold; padding: 5px;"
                )

            self.on_timeframe_changed()
            self.regime_detector = MarketRegimeDetector()
            self.statusBar().showMessage(f"Successfully loaded {ticker}")
            return True

        except Exception as e:
            if interactive:
                QMessageBox.critical(
                    self, "Data Loading Error", f"Failed to load {ticker}:\n{str(e)}"
                )
            self.statusBar().showMessage(f"Error loading {ticker}: {e}")
            return False

    def load_ticker_list(self):
        """Load multiple tickers from CSV and optimize them sequentially"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self, "Already Running", "Wait for the current optimization to finish"
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Ticker List CSV", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            if "ticker" not in df.columns:
                QMessageBox.critical(self, "CSV Error", "CSV must have a 'ticker' column")
                return

            tickers = [str(t).strip().upper() for t in df["ticker"].dropna() if str(t).strip()]
            # De-duplicate while preserving order
            tickers = list(dict.fromkeys(tickers))

            if not tickers:
                QMessageBox.warning(self, "CSV Error", "No tickers found in CSV")
                return

            reply = QMessageBox.question(
                self,
                "Batch Optimization",
                f"Load and optimize {len(tickers)} tickers sequentially?\n"
                "This may take a while. Results are saved per ticker in data_output/.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            self._batch_queue = tickers
            self._process_batch_next()

        except Exception as e:
            QMessageBox.critical(self, "Batch Load Error", f"Failed to load ticker list:\n{str(e)}")

    def _process_batch_next(self):
        """Load and optimize the next ticker in the batch queue"""
        while self._batch_queue:
            ticker = self._batch_queue.pop(0)
            self.ticker_input.setText(ticker)
            if self._load_ticker(ticker, interactive=False):
                self.start_optimization()
                return
            # Load failed - skip to the next ticker

        self.statusBar().showMessage("Batch processing complete")

    def _filter_timeframes(self):
        """Filter data to selected timeframes"""
        self.df_dict = {
            tf_name: self.df_dict_full[tf_name]
            for tf_name, cb in self.tf_checkboxes.items()
            if cb.isChecked() and tf_name in self.df_dict_full
        }

    def on_timeframe_changed(self):
        """Handle timeframe checkbox changes"""
        if not self.df_dict_full:
            return

        self._filter_timeframes()

        # Update chart with first selected timeframe (or clear if none selected)
        for tf_name, _ in TIMEFRAMES:
            if tf_name in self.df_dict:
                self._update_chart(self.df_dict[tf_name]["Close"].values, tf_name)
                return

        self.figure.clear()
        self.canvas.draw()

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
        ax.set_xlabel("Bar", color="white")
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
            QMessageBox.warning(
                self, "No Data", "Load data and select at least one timeframe first"
            )
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "Optimization is already running")
            return

        params = self._get_param_ranges()
        time_cycle_ranges = (
            params["on"],
            params["off"],
            (0, params["on"][1] + params["off"][1]),
        )

        self.worker = MultiTimeframeOptimizer(
            df_dict=self.df_dict,
            n_trials=self.trials_spin.value(),
            time_cycle_ranges=time_cycle_ranges,
            mn1_range=params["mn1"],
            mn2_range=params["mn2"],
            entry_range=params["entry"],
            exit_range=params["exit"],
            ticker=self.current_ticker,
            batch_size=self.batch_spin.value(),
            transaction_costs=self.transaction_costs,
            position_size=self.position_size_pct / 100.0,
        )

        self.worker.progress.connect(self.update_progress)
        self.worker.new_best.connect(self.update_best_label)
        self.worker.phase_update.connect(self.update_phase_label)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Starting optimization...")

        self.worker.start()
        self.statusBar().showMessage(f"Optimization running for {self.current_ticker}...")

    def stop_optimization(self):
        """Stop running optimization"""
        if self.worker:
            self._batch_queue.clear()
            self.worker.stop()
            self.phase_label.setText("Stopping...")
            self.stop_btn.setEnabled(False)

    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_best_label(self, best_params: Dict):
        """Update best parameters display during optimization"""
        self.best_params = best_params

        psr = best_params.get("PSR", 0.0)
        sharpe = best_params.get("Sharpe_Ratio", 0.0)
        sortino = best_params.get("Sortino_Ratio", 0.0)

        self.psr_label.setText(f"{psr:.3f}")
        self.sharpe_label.setText(f"{sharpe:.2f}")
        self.sortino_label.setText(f"{sortino:.2f}")

        get = self._get_tf_value
        fmt = self._fmt
        text = (
            "Best Parameters Found:\n"
            f"PSR: {psr:.3f} | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}\n"
            f"RSI1 Length (MN1): {get(best_params, 'MN1')} | "
            f"RSI2 Length (MN2): {get(best_params, 'MN2')}\n"
            f"Entry: {fmt(get(best_params, 'Entry'))} | "
            f"Exit: {fmt(get(best_params, 'Exit'))}\n"
            f"Time Cycle - ON: {get(best_params, 'On')} | "
            f"OFF: {get(best_params, 'Off')} | START: {get(best_params, 'Start')}"
        )
        self.best_params_label.setText(text)

    def update_phase_label(self, phase_text: str):
        """Update phase info"""
        self.phase_label.setText(phase_text)

    def show_results(self, df_results: pd.DataFrame):
        """Show optimization results"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if df_results.empty:
            # Optimization was stopped before producing a result
            self.phase_label.setText("Optimization stopped")
            self.statusBar().showMessage("Optimization stopped")
            return

        self.best_results = df_results
        self.progress_bar.setValue(100)
        self.phase_label.setText("Optimization complete!")

        best = df_results.iloc[0]
        self._plot_results(best)

        if "trade_log" in best:
            self.last_trade_log = best["trade_log"]

        equity = best.get("equity_curve")
        if equity is not None and not (isinstance(equity, float) and np.isnan(equity)):
            self.last_equity_curve = equity

        self._update_results_display(best)
        self.statusBar().showMessage(f"Optimization completed for {self.current_ticker}")

        # Continue batch processing if more tickers are queued
        if self._batch_queue:
            self._process_batch_next()

    def _update_results_display(self, best: pd.Series):
        """Update the optimization results display panel"""
        psr = best.get("PSR", 0.0)
        sharpe = best.get("Sharpe_Ratio", 0.0)
        sortino = best.get("Sortino_Ratio", 0.0)

        self.psr_label.setText(f"{psr:.3f}")
        self.sharpe_label.setText(f"{sharpe:.2f}")
        self.sortino_label.setText(f"{sortino:.2f}")

        get = self._get_tf_value
        fmt = self._fmt
        text = (
            "✅ Optimization Complete!\n\n"
            "📊 Performance Metrics:\n"
            f"  PSR: {psr:.3f} ({psr * 100:.1f}%) | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}\n"
            f"  Return: {best.get('Percent_Gain_%', 0.0):.2f}% | "
            f"Max DD: {best.get('Max_Drawdown_%', 0.0):.2f}% | "
            f"PF: {best.get('Profit_Factor', 0.0):.2f}\n"
            f"  Total Trades: {best.get('Trade_Count', 0)}\n\n"
            "⚙️  Best Parameters:\n"
            f"  RSI1 Length (MN1): {get(best, 'MN1')}\n"
            f"  RSI2 Length (MN2): {get(best, 'MN2')}\n"
            f"  Entry Threshold: {fmt(get(best, 'Entry'))}\n"
            f"  Exit Threshold: {fmt(get(best, 'Exit'))}\n"
            f"  Time Cycle: ON={get(best, 'On')}, OFF={get(best, 'Off')}, "
            f"START={get(best, 'Start')}"
        )
        self.best_params_label.setText(text)

    def _plot_results(self, best: pd.Series):
        """Plot equity curve with drawdown and buy & hold comparison"""
        self.results_figure.clear()

        equity = best.get("equity_curve")
        if equity is None or (isinstance(equity, float) and np.isnan(equity)):
            self.results_canvas.draw()
            return

        buy_hold_equity = self._calculate_buy_hold_equity(len(equity))

        # Equity curve plot
        ax1 = self.results_figure.add_subplot(211, facecolor="#1e1e1e")
        ax1.plot(equity, color="#4CAF50", linewidth=2, label="Strategy", zorder=2)

        if buy_hold_equity is not None:
            ax1.plot(
                buy_hold_equity,
                color="#FFA500",
                linewidth=2,
                linestyle="--",
                label="Buy & Hold",
                alpha=0.8,
                zorder=1,
            )

        ax1.set_title("Equity Curve vs Buy & Hold", color="white", fontweight="bold")
        ax1.set_ylabel("Equity", color="white")
        ax1.tick_params(colors="white")
        ax1.grid(True, alpha=0.2, color="white")
        ax1.legend(facecolor="#2d2d2d", edgecolor="#3a3a3a", labelcolor="white")

        # Drawdown plot
        ax2 = self.results_figure.add_subplot(212, facecolor="#1e1e1e")

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#F44336", alpha=0.3)
        ax2.plot(drawdown, color="#F44336", linewidth=1, label="Strategy DD")

        if buy_hold_equity is not None:
            bh_running_max = np.maximum.accumulate(buy_hold_equity)
            bh_drawdown = (buy_hold_equity - bh_running_max) / bh_running_max * 100
            ax2.plot(
                bh_drawdown,
                color="#FFA500",
                linewidth=1,
                linestyle="--",
                label="Buy & Hold DD",
                alpha=0.7,
            )

        ax2.set_title("Drawdown % Comparison", color="white", fontweight="bold")
        ax2.set_xlabel("Time", color="white")
        ax2.set_ylabel("Drawdown %", color="white")
        ax2.tick_params(colors="white")
        ax2.grid(True, alpha=0.2, color="white")
        ax2.legend(facecolor="#2d2d2d", edgecolor="#3a3a3a", labelcolor="white")

        self.results_figure.tight_layout()
        self.results_canvas.draw()

    def _calculate_buy_hold_equity(self, n_periods: int) -> Optional[np.ndarray]:
        """Calculate buy & hold equity curve for comparison"""
        try:
            if not self.df_dict or not self.current_ticker:
                return None

            # Prefer the finest selected timeframe (matches the strategy curve)
            timeframe = None
            for tf_name in ("5min", "hourly", "daily"):
                if tf_name in self.df_dict:
                    timeframe = tf_name
                    break
            if timeframe is None:
                return None

            df = self.df_dict[timeframe]
            if df is None or df.empty:
                return None

            prices = df["Close"].values
            if len(prices) < 2:
                return None

            n_use = min(len(prices), n_periods)
            prices = prices[-n_use:]

            initial_equity = 1000.0
            buy_hold = initial_equity * (prices / prices[0])

            if len(buy_hold) < n_periods:
                padding = np.full(n_periods - len(buy_hold), initial_equity)
                buy_hold = np.concatenate([padding, buy_hold])

            return buy_hold

        except Exception as e:
            print(f"Failed to calculate buy & hold: {e}")
            return None

    def show_psr_report(self):
        """Show detailed PSR report"""
        if not self.best_params:
            QMessageBox.information(self, "No Results", "Run optimization first")
            return

        p = self.best_params
        get = self._get_tf_value
        fmt = self._fmt
        report = f"""
=== PSR COMPOSITE REPORT ===

Ticker: {self.current_ticker or 'N/A'}
PSR: {p.get('PSR', 0.0):.4f}

Performance Metrics:
  Sharpe Ratio: {p.get('Sharpe_Ratio', 0.0):.4f}
  Sortino Ratio: {p.get('Sortino_Ratio', 0.0):.4f}
  Total Return: {p.get('Percent_Gain_%', 0.0):.2f}%
  Final Equity: ${p.get('Final_Equity_$', 0.0):,.2f}

Risk Metrics:
  Max Drawdown: {p.get('Max_Drawdown_%', 0.0):.2f}%
  Profit Factor: {p.get('Profit_Factor', 0.0):.4f}

Trading Metrics:
  Total Trades: {p.get('Trade_Count', 0)}

Parameters:
  MN1 (RSI1 Length): {get(p, 'MN1')}
  MN2 (RSI2 Length): {get(p, 'MN2')}
  Entry Threshold: {fmt(get(p, 'Entry'), '.2f')}
  Exit Threshold: {fmt(get(p, 'Exit'), '.2f')}
  On Cycle: {get(p, 'On')}
  Off Cycle: {get(p, 'Off')}
  Start Cycle: {get(p, 'Start')}
"""

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("PSR Composite Report")
        msg.setText(report)
        msg.setStyleSheet("QLabel{min-width: 500px; font-family: monospace;}")
        msg.exec()

    def show_regime_breakdown(self):
        """Detect market regimes with Bayesian change point detection and
        show the strategy's performance broken down by regime type"""
        if self.last_equity_curve is None:
            QMessageBox.warning(
                self, "No Results", "Run an optimization first - the regime breakdown "
                "analyzes the optimized strategy's equity curve."
            )
            return

        if not self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        # Finest timeframe drives the equity curve's bar alignment
        tf_order = {"1min": 0, "5min": 1, "hourly": 2, "daily": 3}
        finest_tf = sorted(self.df_dict.keys(), key=lambda x: tf_order.get(x, 99))[0]
        df = self.df_dict[finest_tf]

        equity = np.asarray(self.last_equity_curve, dtype=np.float64)
        if len(equity) != len(df):
            QMessageBox.warning(
                self,
                "Data Mismatch",
                "The loaded data no longer matches the optimization results "
                f"({len(df)} bars vs {len(equity)} equity points).\n"
                "Re-run optimization with the current data.",
            )
            return

        closes = df["Close"].to_numpy(dtype=np.float64)
        datetimes = df["Datetime"].values
        returns = np.zeros(len(closes))
        returns[1:] = np.diff(closes) / closes[:-1]

        if finest_tf == "daily":
            ann_factor = 252.0
        elif finest_tf == "hourly":
            ann_factor = 252.0 * 6.5
        else:
            ann_factor = 252.0 * 6.5 * 12

        self.statusBar().showMessage("Running Bayesian change point detection...")
        try:
            labels, result = self._causal_regime_labels(returns)

            summary = RegimeStrategyAnalyzer.analyze(
                equity_curve=equity,
                prices=closes,
                datetimes=datetimes,
                labels=labels,
                trade_log=self.last_trade_log,
                annualization_factor=ann_factor,
            )
            report = RegimeStrategyAnalyzer.build_report(
                summary, labels, datetimes, self.current_ticker
            )
        except Exception as e:
            self.statusBar().showMessage("Regime breakdown failed")
            QMessageBox.critical(self, "Regime Breakdown Error", str(e))
            return

        n_switches = int(np.sum(labels[1:] != labels[:-1]))
        self.statusBar().showMessage(
            f"Regime breakdown complete: {len(result.changepoints)} change point(s), "
            f"{n_switches} live regime switch(es)"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Regime Breakdown (BOCPD) - {self.current_ticker}")
        dialog.setGeometry(80, 80, 1400, 950)
        dialog.setStyleSheet(MAIN_STYLESHEET)

        layout = QVBoxLayout()

        try:
            fig = plot_regimes(
                prices=closes,
                equity_curve=equity,
                datetimes=datetimes,
                labels=labels,
                cp_probability=result.cp_probability,
                ticker=self.current_ticker,
            )
            layout.addWidget(FigureCanvas(fig), stretch=3)
        except Exception as e:
            print(f"Failed to create regime plot: {e}")

        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setPlainText(report)
        report_text.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background-color: #1e1e1e; color: white;"
        )
        layout.addWidget(report_text, stretch=2)

        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.setLayout(layout)
        dialog.exec()

    @staticmethod
    def _causal_regime_labels(returns: np.ndarray):
        """Causal bar-by-bar regime labels via BOCPD (shared by the regime
        breakdown and the switching backtest)"""
        n = len(returns)
        detector = BOCPDDetector(
            hazard_lambda=float(np.clip(n // 8, 60, 2000)),
            min_segment_bars=int(max(20, n // 100)),
        )
        result = detector.detect(returns)
        labels = online_regime_labels(returns, result.map_run_length)
        return labels, result

    def _strategy_book(self) -> StrategyBook:
        """The per-ticker persistent strategy book (max 3 slots)"""
        ticker = self.current_ticker or "DEFAULT"
        return StrategyBook(Paths.DATA_DIR / f"{ticker}_strategy_book.json")

    def _book_strategies_for_current_data(self, book: StrategyBook) -> list:
        """Stored strategies whose params cover every loaded timeframe"""
        required = [
            f"{key}_{tf}"
            for tf in self.df_dict.keys()
            for key in ("MN1", "MN2", "Entry", "Exit", "On", "Off", "Start")
        ]
        return [s for s in book.strategies if all(k in s.params for k in required)]

    def save_strategy_to_book(self):
        """Store the current optimization result in the strategy book
        (max 3 slots; a stronger newcomer replaces the weakest slot)"""
        if not self.best_params:
            QMessageBox.warning(
                self, "No Results", "Run an optimization first - the book stores "
                "optimized parameter sets."
            )
            return

        best = (
            self.best_results.iloc[0]
            if getattr(self, "best_results", None) is not None
            else {}
        )
        psr = float(best.get("PSR", 0.0) or 0.0)
        tfs = sorted({k.split("_", 1)[1] for k in self.best_params if k.startswith("MN1_")})

        name = (
            f"{self.current_ticker}_{'+'.join(tfs)}_"
            f"{datetime.now().strftime('%m%d-%H%M')}"
        )
        strategy = StoredStrategy(
            name=name,
            params={k: v for k, v in self.best_params.items()},
            base_score=psr,
            ticker=self.current_ticker,
            timeframes=tfs,
            metrics={
                k: float(best.get(k, 0.0) or 0.0)
                for k in ("PSR", "Sharpe_Ratio", "Sortino_Ratio",
                          "Percent_Gain_%", "Max_Drawdown_%")
                if k in best
            },
        )

        book = self._strategy_book()
        added, replaced = book.add(strategy)

        if not added:
            QMessageBox.information(
                self,
                "Not Stored",
                f"Book is full ({StrategyBook.MAX_STRATEGIES} slots) and this "
                f"strategy's PSR ({psr:.3f}) does not beat the weakest stored "
                "one. It was not saved.",
            )
            return

        msg = f"Stored '{name}' (PSR {psr:.3f})."
        if replaced and replaced != name:
            msg += f"\nReplaced weakest slot: '{replaced}'."
        msg += f"\n\nBook now holds {len(book)}/{StrategyBook.MAX_STRATEGIES}:"
        for s in book.strategies:
            msg += f"\n  • {s.name}  (score {s.base_score:.3f})"
        QMessageBox.information(self, "Strategy Saved", msg)
        self.statusBar().showMessage(
            f"Strategy book: {len(book)}/{StrategyBook.MAX_STRATEGIES} slots used"
        )

    def run_switching_backtest_gui(self):
        """Backtest the regime-switching meta-strategy: all stored
        strategies run in shadow, and only the best for the live regime
        places trades"""
        if not self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Wait for the running optimization")
            return

        book = self._strategy_book()
        strategies = self._book_strategies_for_current_data(book)
        if len(strategies) < 2:
            QMessageBox.warning(
                self,
                "Need More Strategies",
                f"The switching backtest needs at least 2 stored strategies "
                f"covering timeframes {list(self.df_dict.keys())}.\n"
                f"Book has {len(book)} stored, {len(strategies)} usable.\n\n"
                "Run optimizations and use 'Save Strategy to Book'.",
            )
            return

        self.statusBar().showMessage("Running regime-switching backtest...")
        try:
            params_ranges = self._get_param_ranges()
            temp_optimizer = MultiTimeframeOptimizer(
                df_dict=self.df_dict,
                n_trials=1,
                time_cycle_ranges=(
                    params_ranges["on"],
                    params_ranges["off"],
                    (0, params_ranges["on"][1] + params_ranges["off"][1]),
                ),
                mn1_range=params_ranges["mn1"],
                mn2_range=params_ranges["mn2"],
                entry_range=params_ranges["entry"],
                exit_range=params_ranges["exit"],
                ticker=self.current_ticker,
                transaction_costs=self.transaction_costs,
                position_size=self.position_size_pct / 100.0,
            )

            closes = temp_optimizer.np_data[temp_optimizer.finest_tf]["close"]
            datetimes = temp_optimizer.np_data[temp_optimizer.finest_tf]["datetime"]
            returns = np.zeros(len(closes))
            returns[1:] = np.diff(closes) / closes[:-1]

            labels, _ = self._causal_regime_labels(returns)

            result = run_switching_backtest(temp_optimizer, strategies, labels)
            report = build_switching_report(
                result,
                strategies,
                annualization_factor=temp_optimizer.annualization_factor,
                ticker=self.current_ticker,
            )
        except Exception as e:
            self.statusBar().showMessage("Switching backtest failed")
            QMessageBox.critical(self, "Switching Backtest Error", str(e))
            return

        self.statusBar().showMessage(
            f"Switching backtest: {len(result.trades)} trades, "
            f"{len(result.switch_log)} strategy switches"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Regime-Switching Backtest - {self.current_ticker}")
        dialog.setGeometry(80, 80, 1400, 950)
        dialog.setStyleSheet(MAIN_STYLESHEET)
        layout = QVBoxLayout()

        try:
            fig = plot_switching_backtest(
                result, strategies, datetimes, ticker=self.current_ticker
            )
            layout.addWidget(FigureCanvas(fig), stretch=3)
        except Exception as e:
            print(f"Failed to plot switching backtest: {e}")

        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setPlainText(report)
        report_text.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background-color: #1e1e1e; color: white;"
        )
        layout.addWidget(report_text, stretch=2)

        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.setLayout(layout)
        dialog.exec()

    def show_error(self, error_msg: str):
        """Show optimization error"""
        QMessageBox.critical(self, "Error", error_msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.phase_label.setText("Error occurred")
        self.statusBar().showMessage("Error")

        # Don't let one bad ticker abort a batch run
        if self._batch_queue:
            self._process_batch_next()

    # =============================================================================
    # MONTE CARLO
    # =============================================================================

    def run_monte_carlo(self):
        """Run comprehensive Monte Carlo simulation with full quant metrics"""
        if self.last_trade_log is None or (
            isinstance(self.last_trade_log, pd.DataFrame) and self.last_trade_log.empty
        ):
            QMessageBox.warning(self, "No Trade Log", "Run optimization first to get trade log")
            return

        if len(self.last_trade_log) < 10:
            QMessageBox.warning(
                self, "Insufficient Trades", "Need at least 10 trades for Monte Carlo"
            )
            return

        self.statusBar().showMessage("Running comprehensive Monte Carlo analysis...")

        try:
            n_sims = self.mc_simulations_spin.value()

            if isinstance(self.last_trade_log, pd.DataFrame):
                trades = self.last_trade_log.to_dict("records")
            else:
                trades = self.last_trade_log

            basic_results = MonteCarloSimulator.simulate_trade_randomization(
                trades=trades, n_simulations=n_sims, initial_equity=1000.0
            )

            advanced_metrics = AdvancedMonteCarloAnalyzer.calculate_advanced_metrics(
                basic_results, initial_equity=1000.0, risk_free_rate=0.0
            )

            full_report = AdvancedMonteCarloAnalyzer.generate_enhanced_report(
                basic_results, advanced_metrics, initial_equity=1000.0
            )

            self._show_monte_carlo_results(basic_results, advanced_metrics, full_report)
            self.statusBar().showMessage("Monte Carlo analysis completed")

        except Exception as e:
            QMessageBox.critical(self, "Monte Carlo Error", f"Failed to run Monte Carlo:\n{str(e)}")
            self.statusBar().showMessage("Monte Carlo failed")

    def _show_monte_carlo_results(self, basic_results, advanced_metrics, report):
        """Display comprehensive Monte Carlo results in a new window"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Comprehensive Monte Carlo Analysis")
        dialog.setGeometry(100, 100, 1400, 900)
        dialog.setStyleSheet(MAIN_STYLESHEET)

        layout = QVBoxLayout()

        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setPlainText(report)
        report_text.setStyleSheet(
            "font-family: monospace; font-size: 11px; background-color: #1e1e1e; color: white;"
        )
        layout.addWidget(report_text)

        try:
            fig = AdvancedMonteCarloAnalyzer.plot_enhanced_distributions(
                basic_results, advanced_metrics, "Monte Carlo Analysis - Full Suite"
            )
            layout.addWidget(FigureCanvas(fig))
        except Exception as e:
            print(f"Failed to create Monte Carlo visualization: {e}")

        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.setLayout(layout)
        dialog.exec()

    # =============================================================================
    # WALK-FORWARD
    # =============================================================================

    def run_walk_forward(self):
        """Run walk-forward analysis"""
        if not self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        self.statusBar().showMessage("Running walk-forward analysis...")

        try:
            params = self._get_param_ranges()
            time_cycle_ranges = (
                params["on"],
                params["off"],
                (0, params["on"][1] + params["off"][1]),
            )

            results = WalkForwardAnalyzer.run_walk_forward(
                optimizer_class=MultiTimeframeOptimizer,
                df_dict=self.df_dict,
                train_days=self.wf_train_days_spin.value(),
                test_days=self.wf_test_days_spin.value(),
                n_trials=self.wf_trials_spin.value(),
                time_cycle_ranges=time_cycle_ranges,
                mn1_range=params["mn1"],
                mn2_range=params["mn2"],
                entry_range=params["entry"],
                exit_range=params["exit"],
                ticker=self.current_ticker,
                transaction_costs=self.transaction_costs,
                position_size=self.position_size_pct / 100.0,
            )

            overfit_text = "⚠️ LIKELY OVERFIT" if results.is_overfit else "✅ Not overfit"
            msg = f"""
Walk-Forward Analysis Results:

Windows: {len(results.out_of_sample_returns)}
Efficiency Ratio (OOS/IS): {results.efficiency_ratio:.2f}
Consistency: {results.consistency:.2%}

Avg In-Sample Return: {results.avg_in_sample_return:.2f}%
Avg Out-of-Sample Return: {results.avg_out_of_sample_return:.2f}%
Return Degradation: {results.return_degradation:.1f}%

Verdict: {overfit_text}
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
        """Detect current market regime using specified lookback period"""
        if not self.df_dict or "daily" not in self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        try:
            prices = self.df_dict["daily"]["Close"]
            lookback_days = self.regime_lookback_spin.value()

            if len(prices) > lookback_days:
                prices_to_use = prices.iloc[-lookback_days:]
            else:
                prices_to_use = prices

            # Scale detection windows to the lookback period so feature
            # calculation windows are appropriate for the data size
            vol_window = max(10, int(lookback_days / 10))
            trend_fast = max(20, int(lookback_days / 5))
            trend_slow = max(50, int(lookback_days / 1.25))

            self.regime_detector = MarketRegimeDetector(
                vol_window=vol_window,
                trend_window_fast=trend_fast,
                trend_window_slow=trend_slow,
                regime_memory=min(252, lookback_days),
            )

            regime_state = self.regime_detector.detect_regime(prices_to_use)
            self.current_regime_state = regime_state

            text = f"""
Current Market Regime (using {lookback_days} days):

Regime: {regime_state.current_regime.value.upper()}
Confidence: {regime_state.confidence:.2%}
Duration: {regime_state.regime_duration} days

Next Predicted: {regime_state.predicted_next_regime.value.upper()}
Transition Probability: {regime_state.transition_probability:.2%}
Suggested Position Size: {regime_state.suggested_position_size:.2f}x

Detection Windows:
  Volatility: {vol_window} days
  Trend Fast: {trend_fast} days
  Trend Slow: {trend_slow} days
"""

            self.regime_display.setText(text)
            self.statusBar().showMessage(f"Regime detected using {lookback_days} days")

        except Exception as e:
            QMessageBox.critical(self, "Regime Detection Error", str(e))

    def train_regime_predictor(self):
        """Train ML regime predictor"""
        if not self.regime_detector or not self.df_dict or "daily" not in self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        self.statusBar().showMessage("Training regime predictor...")

        try:
            self.regime_predictor = RegimePredictor(detector=self.regime_detector)

            prices = self.df_dict["daily"]["Close"]
            returns = prices.pct_change().fillna(0)

            results = self.regime_predictor.train(prices, returns)

            avg_accuracy = (
                results.accuracy_1day + results.accuracy_5day + results.accuracy_20day
            ) / 3

            text = f"""
Regime Predictor Trained:

Training Accuracy: {self.regime_predictor.train_accuracy:.2%}
Validation Accuracy: {self.regime_predictor.test_accuracy:.2%}
Multi-horizon Accuracy: {avg_accuracy:.2%}

Model: Random Forest
Features: {len(self.regime_predictor.feature_names)}
"""

            self.prediction_display.setText(text)
            self.statusBar().showMessage("Predictor trained successfully")

        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
            self.statusBar().showMessage("Training failed")

    def calculate_pbr(self):
        """Calculate Probability of Backtested Returns"""
        if self.best_results is None or self.best_results.empty:
            QMessageBox.warning(self, "No Results", "Run optimization first")
            return

        try:
            best = self.best_results.iloc[0]

            backtest_sharpe = float(best.get("Sharpe_Ratio", 0.0))
            backtest_return = float(best.get("Percent_Gain_%", 0.0)) / 100.0
            n_trades = int(best.get("Trade_Count", 0))

            if n_trades == 0:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    "No trades found in optimization results",
                )
                return

            # Optimized parameters: MN1, MN2, Entry, Exit, On, Off (per timeframe)
            n_parameters = 6 * max(1, len(self.df_dict))

            regime_stability = None
            if self.current_regime_state:
                regime_stability = self.current_regime_state.transition_probability

            pbr_score, pbr_details = PBRCalculator.calculate_pbr(
                backtest_sharpe=backtest_sharpe,
                backtest_return=backtest_return,
                n_trades=n_trades,
                n_parameters=n_parameters,
                walk_forward_efficiency=None,
                current_regime_stability=regime_stability,
            )

            self.pbr_display.setText(f"PBR Score: {pbr_score:.1%}")

            interpretation = PBRCalculator.interpret_pbr(pbr_score)
            text = (
                "PBR (Probability of Backtested Returns):\n\n"
                f"Score: {pbr_score:.1%}\n"
                f"Interpretation: {interpretation}\n\n"
                "Backtest Metrics:\n"
                f"  Sharpe Ratio: {backtest_sharpe:.2f}\n"
                f"  Total Return: {backtest_return:.1%}\n"
                f"  Number of Trades: {n_trades}\n"
                f"  Parameters: {n_parameters}\n\n"
                "Contributing Factors:\n"
                f"  Sharpe Factor: {pbr_details.get('sharpe_contribution', 0.0):.1%}\n"
                f"  Sample Size: {pbr_details.get('sample_size_factor', 0.0):.1%}\n"
                f"  Overfitting Penalty: {pbr_details.get('overfitting_factor', 0.0):.1%}\n"
            )

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("PBR calculated successfully")

        except Exception as e:
            QMessageBox.critical(self, "PBR Error", f"Failed to calculate PBR:\n{e}")
            self.statusBar().showMessage("PBR calculation failed")

    def calibrate_probabilities(self):
        """Calibrate regime detection probabilities against realized regimes"""
        if not self.regime_detector or not self.df_dict or "daily" not in self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data and detect regime first")
            return

        self.statusBar().showMessage("Calibrating probabilities (this may take a moment)...")

        try:
            prices = self.df_dict["daily"]["Close"]

            # Walk forward through history collecting the detector's predicted
            # probabilities at each step and the regime realized at the next step
            min_history = 60
            max_samples = 300  # cap for GUI responsiveness
            start = max(min_history, len(prices) - max_samples)

            detector = MarketRegimeDetector()
            regime_list = list(MarketRegime)
            regime_to_int = {r: i for i, r in enumerate(regime_list)}

            prob_rows = []
            labels = []
            prev_probs = None
            for i in range(start, len(prices)):
                try:
                    state = detector.detect_regime(prices.iloc[:i])
                except Exception:
                    prev_probs = None
                    continue

                if prev_probs is not None:
                    prob_rows.append(prev_probs)
                    labels.append(regime_to_int[state.current_regime])

                prev_probs = [state.regime_probabilities.get(r, 0.0) for r in regime_list]

            if len(labels) < 50:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Need at least 50 historical regime observations for "
                    f"calibration. Found: {len(labels)}",
                )
                return

            predicted_probs = np.asarray(prob_rows, dtype=float)
            true_labels = np.asarray(labels)

            self.regime_calibrator = MultiClassCalibrator(method="isotonic")
            self.regime_calibrator.fit(predicted_probs, true_labels)

            text = (
                "Probability Calibration Complete\n\n"
                "Method: Isotonic Regression\n"
                f"Historical Observations: {len(true_labels)}\n"
                f"Regime Classes: {len(regime_list)}\n\n"
                "Detector probabilities are now calibrated against realized "
                "next-day regimes, improving the reliability of confidence scores."
            )

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Calibration complete")

        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", str(e))
            self.statusBar().showMessage("Calibration failed")

    def check_multi_horizon_agreement(self):
        """Check multi-horizon prediction agreement"""
        if not self.regime_detector:
            QMessageBox.warning(self, "No Data", "Load data and detect regime first")
            return

        if not self.df_dict or "daily" not in self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        self.statusBar().showMessage("Checking multi-horizon agreement...")

        try:
            prices = self.df_dict["daily"]["Close"]

            # Detect regimes at different lookback periods (simulating horizons)
            horizons = [1, 5, 10, 20]  # days
            predictions = []

            for h in horizons:
                try:
                    lookback_prices = prices.iloc[-h * 10 :] if len(prices) > h * 10 else prices
                    regime_state = self.regime_detector.detect_regime(lookback_prices)

                    predictions.append(
                        HorizonPrediction(
                            horizon_days=h,
                            predicted_regime=regime_state.current_regime,
                            confidence=regime_state.confidence,
                            probabilities=regime_state.regime_probabilities,
                        )
                    )
                except Exception as e:
                    print(f"Error detecting regime for horizon {h}: {e}")

            if len(predictions) < 2:
                QMessageBox.warning(
                    self,
                    "Insufficient Predictions",
                    "Could not generate enough horizon predictions",
                )
                return

            agreement_analyzer = MultiHorizonAgreementIndex(horizons=horizons)
            analysis = agreement_analyzer.calculate_agreement(predictions)

            text = (
                "Multi-Horizon Agreement Analysis\n\n"
                f"Agreement Index: {analysis.agreement_index:.2%}\n"
                f"Consensus Regime: {analysis.consensus_regime.value.upper()}\n"
                f"Consensus Strength: {analysis.consensus_strength:.2%}\n"
                f"Signal Quality: {analysis.signal_quality.upper()}\n\n"
                "Predictions by Horizon:\n"
            )
            for pred in analysis.horizon_predictions:
                text += (
                    f"  {pred.horizon_days:2d}-day: "
                    f"{pred.predicted_regime.value.upper():<15s} "
                    f"({pred.confidence:.1%})\n"
                )
            text += f"\nRecommendation: {analysis.recommendation}"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Multi-horizon check complete")

        except Exception as e:
            QMessageBox.critical(self, "Agreement Error", str(e))
            self.statusBar().showMessage("Agreement check failed")

    def run_robustness_tests(self):
        """Run White's Reality Check and bootstrap Sharpe confidence interval"""
        if self.best_results is None or self.best_results.empty:
            QMessageBox.warning(self, "No Results", "Run optimization first")
            return

        if not self.df_dict or "daily" not in self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first for benchmark")
            return

        self.statusBar().showMessage("Running robustness tests (this may take a minute)...")

        try:
            prices = self.df_dict["daily"]["Close"]
            benchmark_returns = prices.pct_change().dropna().values

            strategy_returns = None

            if self.last_equity_curve is not None and len(self.last_equity_curve) > 1:
                equity_series = pd.Series(self.last_equity_curve)
                strategy_returns = equity_series.pct_change().dropna().values
            elif self.last_trade_log is not None and len(self.last_trade_log) > 0:
                if isinstance(self.last_trade_log, pd.DataFrame):
                    trades = self.last_trade_log
                else:
                    trades = pd.DataFrame(self.last_trade_log)

                if "Percent_Change" in trades.columns:
                    trade_returns = trades["Percent_Change"].values / 100.0
                    equity = 1000.0 * np.cumprod(1 + trade_returns)
                    equity = np.concatenate([[1000.0], equity])
                    strategy_returns = pd.Series(equity).pct_change().dropna().values

            if strategy_returns is None or len(strategy_returns) < 30:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    "Need at least 30 strategy returns for robustness testing. "
                    "Run optimization with more trades or longer history.",
                )
                return

            # Align lengths (use shorter series)
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]

            if min_len < 30:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Need at least 30 aligned returns. Found: {min_len}",
                )
                return

            # White's Reality Check (reduced bootstrap for GUI performance)
            wrc = WhiteRealityCheck(n_bootstrap=500, block_size=10)
            wrc_result = wrc.test(
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                alternative_strategies=None,
            )

            # Sharpe ratio confidence interval on excess returns
            bootstrap = BlockBootstrapValidator(n_bootstrap=500, block_size=10)
            sharpe, sharpe_lower, sharpe_upper = bootstrap.sharpe_ratio_ci(
                strategy_returns - benchmark_returns
            )

            text = (
                "Robustness Tests\n\n"
                f"Sample Size: {len(strategy_returns)} periods\n"
                "Bootstrap Samples: 500\n\n"
                "White's Reality Check:\n"
                f"  Test Statistic: {wrc_result.test_statistic:.6f}\n"
                f"  P-value: {wrc_result.p_value:.6f}\n"
            )
            icon = "✅" if wrc_result.is_significant else "❌"
            text += (
                f"  {icon} Result: "
                f"{'SIGNIFICANT' if wrc_result.is_significant else 'NOT SIGNIFICANT'}\n"
                f"  Null: {wrc_result.null_hypothesis}\n\n"
                "Sharpe Ratio (Excess Returns):\n"
                f"  Point Estimate: {sharpe:.3f}\n"
                f"  95% CI: [{sharpe_lower:.3f}, {sharpe_upper:.3f}]\n\n"
            )

            # Verdict requires both statistical significance and a CI that
            # excludes zero; Sharpe magnitude sets the confidence level
            if wrc_result.is_significant and sharpe > 1.0 and sharpe_lower > 0:
                verdict = (
                    "✅ ROBUST: Statistically significant outperformance with good Sharpe ratio"
                )
            elif wrc_result.is_significant and sharpe > 0.5 and sharpe_lower > 0:
                verdict = "⚠️ MODERATELY ROBUST: Statistically significant but modest Sharpe ratio"
            elif wrc_result.is_significant and sharpe > 0:
                verdict = "⚠️ WEAK SIGNAL: Statistically significant but very low Sharpe ratio - be cautious"
            elif wrc_result.is_significant and sharpe <= 0:
                verdict = "❌ SIGNIFICANT UNDERPERFORMANCE: Strategy loses money"
            elif sharpe_lower < 0 < sharpe_upper:
                verdict = "❌ NOT ROBUST: Confidence interval includes zero - no reliable edge"
            else:
                verdict = "⚠️ NOT ROBUST: Performance may be due to luck or overfitting"

            text += f"Verdict: {verdict}\n\n"
            text += f"Interpretation:\n{wrc_result.interpretation}"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Robustness tests complete")

        except Exception as e:
            QMessageBox.critical(self, "Robustness Error", str(e))
            self.statusBar().showMessage("Robustness tests failed")

    def run_regime_diagnostics(self):
        """Run regime stability and persistence diagnostics"""
        if not self.regime_detector:
            QMessageBox.warning(self, "No Data", "Load data and detect regime first")
            return

        if not self.df_dict or "daily" not in self.df_dict:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        self.statusBar().showMessage("Running regime diagnostics...")

        try:
            prices = self.df_dict["daily"]["Close"]

            returns = prices.pct_change().dropna()
            volatility = returns.rolling(window=20).std().dropna()

            # Detect historical regimes (capped for GUI responsiveness)
            max_observations = 500
            start = max(20, len(prices) - max_observations)
            regime_sequence = []
            for i in range(start, len(prices)):
                try:
                    window_prices = prices.iloc[max(0, i - 100) : i + 1]
                    regime_state = self.regime_detector.detect_regime(window_prices)
                    regime_sequence.append(regime_state.current_regime)
                except Exception:
                    continue

            if len(regime_sequence) < 30:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Need at least 30 regime observations. Found: {len(regime_sequence)}",
                )
                return

            returns_aligned = returns.iloc[-len(regime_sequence) :]
            volatility_aligned = volatility.iloc[-len(regime_sequence) :]

            analyzer = RegimeDiagnosticAnalyzer(self.regime_detector)

            persistence = analyzer.calculate_regime_persistence(regime_sequence)
            stability = analyzer.calculate_stability_score(regime_sequence)
            transition_matrix = analyzer.build_transition_matrix(regime_sequence)
            purity = analyzer.calculate_regime_purity(
                regime_sequence, returns_aligned, volatility_aligned
            )

            text = (
                "Regime Diagnostics\n\n"
                f"Observations: {len(regime_sequence)}\n"
                f"Stability Score: {stability:.2%}\n\n"
                "Regime Persistence (Half-Life in Days):\n"
            )
            for regime, hl in sorted(persistence.items(), key=lambda x: x[1], reverse=True):
                text += f"  {regime.value.upper():<15s}: {hl:.1f} days\n"

            text += "\nRegime Purity (Quality Scores):\n"
            for regime, score in sorted(purity.items(), key=lambda x: x[1], reverse=True):
                icon = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
                text += f"  {icon} {regime.value.upper():<15s}: {score:.2%}\n"

            text += "\nTransition Matrix (From → To Probabilities):\n"
            regime_list = list(MarketRegime)
            text += "     " + "".join(f"{r.value[:4]:>8s}" for r in regime_list) + "\n"
            for i, from_regime in enumerate(regime_list):
                text += f"{from_regime.value[:4]:<4s} "
                for j in range(len(regime_list)):
                    text += f"{transition_matrix[i, j]:>7.1%} "
                text += "\n"

            if stability > 0.7 and np.mean(list(purity.values())) > 0.6:
                assessment = "✅ EXCELLENT: Regime detection is highly stable and accurate"
            elif stability > 0.5 and np.mean(list(purity.values())) > 0.5:
                assessment = "⚠️ GOOD: Regime detection is acceptable with some noise"
            else:
                assessment = "❌ POOR: Regime detection needs improvement"

            text += f"\nAssessment: {assessment}"

            self.institutional_display.append(text)
            self.institutional_display.append("\n" + "=" * 50 + "\n")

            self.statusBar().showMessage("Regime diagnostics complete")

        except Exception as e:
            QMessageBox.critical(self, "Diagnostics Error", str(e))
            self.statusBar().showMessage("Regime diagnostics failed")

    def run_cross_asset_analysis(self):
        """Run cross-asset regime analysis"""
        self.statusBar().showMessage(
            "Running cross-asset analysis (loading SPY, TLT, GLD, BTC-USD)..."
        )

        try:
            tickers = ["SPY", "TLT", "GLD", "BTC-USD"]
            asset_regimes = {}

            for ticker in tickers:
                try:
                    df_dict, error = DataLoader.load_yfinance_data(ticker)
                    if error or "daily" not in df_dict:
                        print(f"Failed to load {ticker}: {error}")
                        continue

                    detector = MarketRegimeDetector()
                    regime_state = detector.detect_regime(df_dict["daily"]["Close"])
                    asset_regimes[ticker] = regime_state.current_regime

                except Exception as e:
                    print(f"Error analyzing {ticker}: {e}")

            if len(asset_regimes) < 2:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Need at least 2 assets for cross-asset analysis. "
                    f"Loaded: {len(asset_regimes)}",
                )
                return

            regime_values = list(asset_regimes.values())
            most_common = max(set(regime_values), key=regime_values.count)
            agreement = regime_values.count(most_common) / len(regime_values)

            text = (
                "Cross-Asset Regime Analysis:\n\n"
                f"Assets Analyzed: {len(asset_regimes)}\n"
                f"Consensus Regime: {most_common.value.upper()}\n"
                f"Agreement Score: {agreement:.2%}\n\n"
                "Individual Assets:\n"
            )
            for asset, regime in asset_regimes.items():
                text += f"  {asset}: {regime.value.upper()}\n"

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

        if not self.df_dict or not self.current_ticker:
            QMessageBox.warning(self, "No Data", "Load data first")
            return

        # All selected timeframes need optimized parameters
        timeframes = list(self.df_dict.keys())
        missing = [tf for tf in timeframes if f"MN1_{tf}" not in self.best_params]
        if missing:
            QMessageBox.warning(
                self,
                "Missing Parameters",
                f"No optimized parameters for timeframe(s): {', '.join(missing)}.\n"
                "Re-run optimization with the current timeframe selection.",
            )
            return

        api_key = self.api_key_input.text().strip()
        api_secret = self.api_secret_input.text().strip()

        if not api_key or not api_secret:
            QMessageBox.warning(self, "Missing Credentials", "Please enter API key and secret")
            return

        paper = self.paper_trading_cb.isChecked()
        mode = "PAPER" if paper else "LIVE"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning if not paper else QMessageBox.Icon.Question)
        msg.setWindowTitle("Confirm Trading")
        msg.setText(f"Start {mode} trading of {self.current_ticker}?")
        msg.setInformativeText("Make sure you understand the risks involved.")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if msg.exec() != QMessageBox.StandardButton.Yes:
            return

        # Optional regime-switching mode from the stored strategy book
        strategy_book = None
        if getattr(self, "use_book_cb", None) is not None and self.use_book_cb.isChecked():
            book = self._strategy_book()
            usable = self._book_strategies_for_current_data(book)
            if len(usable) < 2:
                QMessageBox.warning(
                    self,
                    "Strategy Book Unavailable",
                    f"Regime switching needs 2+ stored strategies covering "
                    f"{list(self.df_dict.keys())}; book has {len(usable)} usable. "
                    "Starting with the single optimized strategy instead.",
                )
            else:
                strategy_book = book

        try:
            self.live_trader = AlpacaLiveTrader(
                api_key=api_key,
                secret_key=api_secret,
                symbol=self.current_ticker,
                params=self.best_params,
                df_dict=self.df_dict,
                timeframes=timeframes,
                position_size_pct=self.position_size_pct / 100.0,
                paper=paper,
                strategy_book=strategy_book,
            )

            self.live_trader.status_update.connect(self.update_trading_status)
            self.live_trader.trade_executed.connect(self.on_trade_executed)
            self.live_trader.error.connect(self.on_trading_error)

            self.live_trader.start()

            self.live_trading_btn.setText("⏹️ STOP LIVE TRADING")
            self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_ACTIVE)
            self.trading_status_label.setText(f"Status: {mode} TRADING ACTIVE")
            self.trading_status_label.setStyleSheet(
                f"padding: 15px; background-color: {COLOR_SUCCESS}; color: white; "
                "border-radius: 4px; font-size: 14px; font-weight: bold;"
            )

            self.statusBar().showMessage(f"{mode} trading started")

        except Exception as e:
            QMessageBox.critical(self, "Trading Error", f"Failed to start trading:\n{str(e)}")

    def stop_live_trading(self):
        """Stop live trading"""
        if not self.live_trader:
            return

        try:
            self.live_trader.stop()

            self.live_trading_btn.setText("▶️ START LIVE TRADING")
            self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_STOPPED)
            self.trading_status_label.setText("Status: Not Running")
            self.trading_status_label.setStyleSheet(
                "padding: 15px; background-color: #2d2d2d; border-radius: 4px; font-size: 14px;"
            )

            self.statusBar().showMessage("Trading stopped")

        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Failed to stop trading:\n{str(e)}")

    def update_trading_status(self, status: str):
        """Update trading status display"""
        self.trading_status_label.setText(f"Status: {status}")

    def on_trade_executed(self, trade_info: Dict):
        """Handle trade execution notification"""
        text = (
            f"[{trade_info['time']}] {trade_info['action']} "
            f"{trade_info['shares']} {trade_info['symbol']} "
            f"@ ${trade_info['price']:.2f}\n"
        )
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

        total_exposure = self.position_size_pct * self.max_positions
        if total_exposure > 100:
            self.leverage_warning_label.setText(
                f"⚠️ WARNING: Total exposure is {total_exposure:.0f}% (using leverage)"
            )
        else:
            self.leverage_warning_label.setText("")

    def on_transaction_costs_changed(self):
        """Handle transaction cost changes"""
        self.transaction_costs.COMMISSION_PCT = self.commission_pct_spin.value() / 100
        self.transaction_costs.SLIPPAGE_PCT = self.slippage_pct_spin.value() / 100
        self.transaction_costs.SPREAD_PCT = self.spread_pct_spin.value() / 100

    def set_costs_for_stocks(self):
        """Set transaction costs for stocks (matches TransactionCosts.for_stocks)"""
        costs = TransactionCosts.for_stocks()
        self.commission_pct_spin.setValue(costs.COMMISSION_PCT * 100)
        self.slippage_pct_spin.setValue(costs.SLIPPAGE_PCT * 100)
        self.spread_pct_spin.setValue(costs.SPREAD_PCT * 100)

    def set_costs_for_crypto(self):
        """Set transaction costs for crypto (matches TransactionCosts.for_crypto)"""
        costs = TransactionCosts.for_crypto()
        self.commission_pct_spin.setValue(costs.COMMISSION_PCT * 100)
        self.slippage_pct_spin.setValue(costs.SLIPPAGE_PCT * 100)
        self.spread_pct_spin.setValue(costs.SPREAD_PCT * 100)

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
        self._batch_queue.clear()

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        if self.live_trader and self.live_trader.is_running:
            self.live_trader.stop()
            self.live_trader.wait()

        event.accept()
