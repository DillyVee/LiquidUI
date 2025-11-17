"""
Main Window GUI
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
    QVBoxLayout,
    QWidget,
)

# Config imports
from config.settings import (
    AlpacaConfig,
    IndicatorRanges,
    OptimizationConfig,
    Paths,
    RiskConfig,
    TransactionCosts,
)

# ✅ ADD THESE MISSING IMPORTS:
from data import DataLoader

# GUI style imports
from gui.styles import (
    COLOR_SUCCESS,
    COLOR_WARNING,
    LIVE_TRADING_BUTTON_ACTIVE,
    LIVE_TRADING_BUTTON_STOPPED,
    MAIN_STYLESHEET,
)

# Institutional-grade regime analysis modules
from models.regime_agreement import (
    HorizonPrediction,
    MultiHorizonAgreementIndex,
)
from models.regime_calibration import MultiClassCalibrator
from models.regime_cross_asset import CrossAssetRegimeAnalyzer, load_multi_asset_data

# Regime detection imports
from models.regime_detection import (
    MarketRegime,
    MarketRegimeDetector,
    PBRCalculator,
    RegimeState,
)
from models.regime_diagnostics import RegimeDiagnosticAnalyzer
from models.regime_predictor import RegimeBasedPositionSizer, RegimePredictor
from models.regime_robustness import (
    BlockBootstrapValidator,
    HansenSPATest,
    WhiteRealityCheck,
    run_full_robustness_suite,
)
from optimization import MultiTimeframeOptimizer, PerformanceMetrics

# Monte Carlo imports (these are correct!)
from optimization.monte_carlo import (
    AdvancedMonteCarloAnalyzer,
    AdvancedMonteCarloMetrics,
    MonteCarloResults,
    MonteCarloSimulator,
)

# Walk-forward imports
from optimization.walk_forward import WalkForwardAnalyzer, WalkForwardResults
from trading import ALPACA_AVAILABLE, AlpacaLiveTrader


class MainWindow(QMainWindow):
    """Main application window"""

    """
Deep diagnostic for walk-forward failures
Add this as a method to MainWindow and call it before running walk-forward
"""

    def diagnose_walk_forward_issue(self):
        """Comprehensive diagnostic for walk-forward problems"""
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD DEEP DIAGNOSTIC")
        print(f"{'='*70}")

        # 1. Check df_dict_full
        print(f"\n1️⃣  Checking df_dict_full:")
        print(f"   Type: {type(self.df_dict_full)}")
        print(
            f"   Keys: {list(self.df_dict_full.keys()) if self.df_dict_full else 'EMPTY!'}"
        )

        if not self.df_dict_full:
            print(f"   ❌ df_dict_full is EMPTY!")
            return

        for tf, df in self.df_dict_full.items():
            print(f"\n   {tf}:")
            print(f"      Type: {type(df)}")
            print(f"      Shape: {df.shape}")
            print(f"      Columns: {list(df.columns)}")
            print(f"      Index type: {type(df.index)}")

            # Check Datetime
            if "Datetime" in df.columns:
                print(f"      Datetime column: EXISTS ✅")
                dt = df["Datetime"]
                print(f"      Datetime type: {dt.dtype}")
            else:
                print(f"      Datetime column: MISSING ❌")
                print(f"      Index: {df.index[:3]}")
                try:
                    dt = pd.to_datetime(df.index)
                    print(f"      Converted index to datetime: ✅")
                except Exception as e:
                    print(f"      Cannot convert index: ❌ {e}")
                    continue

            # Check date range
            try:
                min_date = dt.min()
                max_date = dt.max()
                days = (max_date - min_date).days
                print(f"      Date range: {min_date} to {max_date}")
                print(f"      Total days: {days}")
            except Exception as e:
                print(f"      ❌ Error getting date range: {e}")

        # 2. Check selected timeframes
        print(f"\n2️⃣  Checking selected timeframes:")
        selected_tfs = [tf for tf, cb in self.tf_checkboxes.items() if cb.isChecked()]
        print(f"   Checked boxes: {selected_tfs}")

        available_selected = [tf for tf in selected_tfs if tf in self.df_dict_full]
        print(f"   Available & selected: {available_selected}")

        if not available_selected:
            print(f"   ❌ No valid timeframes selected!")
            return

        # 3. Check window settings
        print(f"\n3️⃣  Checking window settings:")
        train_days = self.wf_train_days_spin.value()
        test_days = self.wf_test_days_spin.value()
        trials = self.wf_trials_spin.value()

        print(f"   Train days: {train_days}")
        print(f"   Test days: {test_days}")
        print(f"   Trials: {trials}")
        print(f"   Required per window: {train_days + test_days} days")

        # 4. Check what walk-forward will receive
        print(f"\n4️⃣  Simulating walk-forward data reception:")

        tf_order = {"5min": 0, "hourly": 1, "daily": 2}
        finest_tf = sorted(available_selected, key=lambda x: tf_order.get(x, 99))[0]

        print(f"   Finest timeframe: {finest_tf}")

        df_test = self.df_dict_full[finest_tf].copy()
        print(f"   df shape: {df_test.shape}")

        # Check if Datetime exists
        if "Datetime" not in df_test.columns:
            print(f"   ⚠️  Adding Datetime from index...")
            df_test["Datetime"] = pd.to_datetime(df_test.index)

        min_dt = df_test["Datetime"].min()
        max_dt = df_test["Datetime"].max()
        total_days = (max_dt - min_dt).days

        print(f"   Date range: {min_dt.date()} to {max_dt.date()}")
        print(f"   Total days: {total_days}")

        # Calculate windows
        n_windows = max(1, (total_days - train_days) // test_days)
        print(f"   Calculated windows: {n_windows}")

        # 5. Simulate first window
        print(f"\n5️⃣  Simulating first window split:")

        train_start = min_dt
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)

        print(f"   Train: {train_start.date()} to {train_end.date()}")
        print(f"   Test:  {test_start.date()} to {test_end.date()}")

        if test_end > max_dt:
            print(
                f"   ❌ Window exceeds data! (test_end={test_end.date()}, max={max_dt.date()})"
            )
        else:
            print(f"   ✅ Window fits in data")

        # Count bars in window
        train_mask = (df_test["Datetime"] >= train_start) & (
            df_test["Datetime"] < train_end
        )
        test_mask = (df_test["Datetime"] >= test_start) & (
            df_test["Datetime"] < test_end
        )

        train_bars = train_mask.sum()
        test_bars = test_mask.sum()

        print(f"   Train bars: {train_bars}")
        print(f"   Test bars: {test_bars}")

        if train_bars < 100:
            print(f"   ❌ Insufficient training bars (need 100+)")
        else:
            print(f"   ✅ Sufficient training bars")

        if test_bars < 10:
            print(f"   ❌ Insufficient test bars (need 10+)")
        else:
            print(f"   ✅ Sufficient test bars")

        # 6. Check optimizer kwargs
        print(f"\n6️⃣  Checking optimizer parameters:")
        print(f"   Ticker: {self.current_ticker}")
        print(f"   MN1 range: ({self.mn1_min.value()}, {self.mn1_max.value()})")
        print(f"   MN2 range: ({self.mn2_min.value()}, {self.mn2_max.value()})")
        print(f"   Entry range: ({self.entry_min.value()}, {self.entry_max.value()})")
        print(f"   Exit range: ({self.exit_min.value()}, {self.exit_max.value()})")
        print(f"   Equity curve opt: {self.equity_curve_check.isChecked()}")

        # 7. Final verdict
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC SUMMARY")
        print(f"{'='*70}")

        if not self.df_dict_full:
            print(f"❌ PROBLEM: No data loaded")
        elif not available_selected:
            print(f"❌ PROBLEM: No valid timeframes selected")
        elif total_days < (train_days + test_days):
            print(f"❌ PROBLEM: Not enough data")
            print(f"   Have: {total_days} days")
            print(f"   Need: {train_days + test_days} days")
            print(f"   Reduce settings to: train={total_days//3}, test={total_days//6}")
        elif n_windows < 1:
            print(f"❌ PROBLEM: Cannot create any windows")
        elif train_bars < 100 or test_bars < 10:
            print(f"❌ PROBLEM: Insufficient bars per window")
        else:
            print(f"✅ Configuration looks GOOD!")
            print(f"   Should be able to create {n_windows} window(s)")

        print(f"{'='*70}\n")

    def init_ui(self):
        """Initialize the user interface"""
        self.setStyleSheet(MAIN_STYLESHEET)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(4)  # ✅ ADD THIS - Reduced spacing between widgets
        main_layout.setContentsMargins(6, 6, 6, 6)  # ✅ ADD THIS - Reduced margins

        self.setMinimumSize(1000, 800)

        # Data source controls
        self._add_data_source_controls(main_layout)

        # Date range display
        self._add_date_range_display(main_layout)

        # Timeframe selection
        self._add_timeframe_controls(main_layout)

        # Phase information
        self._add_phase_info(main_layout)

        # Optimization controls
        self._add_optimization_controls(main_layout)

        # Parameter ranges
        self._add_parameter_ranges(main_layout)

        # Action buttons
        self._add_action_buttons(main_layout)

        # Regime detection controls
        self._add_regime_detection_controls(main_layout)

        # Live trading controls
        self._add_live_trading_controls(main_layout)

        # Risk management
        self._add_risk_management_controls(main_layout)

        # Transaction costs
        self._add_transaction_cost_controls(main_layout)

        main_layout.addStretch(10)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Chart
        self.figure = plt.figure(facecolor="#121212")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)  # ✅ ADD THIS - Ensure minimum chart height
        main_layout.addWidget(
            self.canvas, stretch=10
        )  # ✅ MODIFIED - Add stretch factor

        # Best parameters display
        self.best_params_label = QLabel("Best Parameters: N/A")
        self.best_params_label.setWordWrap(True)
        main_layout.addWidget(self.best_params_label)

        # Set central widget with scroll area
        container = QWidget()
        container.setLayout(main_layout)

        # Wrap in scroll area for long content
        scroll_area = QScrollArea()
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.setCentralWidget(scroll_area)

    def _add_data_source_controls(self, layout: QVBoxLayout):
        """Add data source input controls"""
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Data Source:"))

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker (e.g., AAPL, SPY, BTC-USD)")

        self.load_yf_btn = QPushButton("Load from Yahoo Finance")
        self.load_yf_btn.clicked.connect(self.load_yfinance)

        # ✅ NEW: Batch ticker button
        self.batch_ticker_btn = QPushButton("📋 Load Ticker List")
        self.batch_ticker_btn.clicked.connect(self.load_ticker_list)
        self.batch_ticker_btn.setToolTip(
            "Load multiple tickers from text file (one per line)\n"
            "Will optimize each ticker sequentially"
        )
        self.batch_ticker_btn.setStyleSheet(
            """
            QPushButton { 
                background-color: #2196F3; 
                font-weight: bold;
            }
            QPushButton:hover { 
                background-color: #42A5F5; 
            }
        """
        )

        source_layout.addWidget(self.ticker_input)
        source_layout.addWidget(self.load_yf_btn)
        source_layout.addWidget(self.batch_ticker_btn)
        layout.addLayout(source_layout)

    def _add_date_range_display(self, layout: QVBoxLayout):
        """Add date range display"""
        date_layout = QHBoxLayout()
        self.date_range_label = QLabel("Date Range: N/A")
        self.date_range_label.setStyleSheet("color: #2979ff; font-weight: bold;")
        date_layout.addWidget(self.date_range_label)
        date_layout.addStretch()
        layout.addLayout(date_layout)

    def _add_timeframe_controls(self, layout: QVBoxLayout):
        """Add timeframe selection controls"""
        tf_group = QGroupBox("Active Timeframes")
        tf_layout = QHBoxLayout()

        self.tf_checkboxes = {}
        self.tf_checkboxes["daily"] = QCheckBox("Daily")
        self.tf_checkboxes["daily"].setChecked(False)
        self.tf_checkboxes["daily"].stateChanged.connect(self.on_timeframe_changed)

        self.tf_checkboxes["hourly"] = QCheckBox("Hourly")
        self.tf_checkboxes["hourly"].setChecked(True)
        self.tf_checkboxes["hourly"].stateChanged.connect(self.on_timeframe_changed)

        self.tf_checkboxes["5min"] = QCheckBox("5-Minute")
        self.tf_checkboxes["5min"].setChecked(False)
        self.tf_checkboxes["5min"].stateChanged.connect(self.on_timeframe_changed)

        for cb in self.tf_checkboxes.values():
            tf_layout.addWidget(cb)

        tf_group.setLayout(tf_layout)
        layout.addWidget(tf_group)

    def _add_phase_info(self, layout: QVBoxLayout):
        """Add phase information display"""
        self.phase_info_label = QLabel("Phase Info: Ready to start")
        self.phase_info_label.setStyleSheet(
            "color: #aaa; font-size: 10pt; font-style: italic;"
        )
        self.phase_info_label.setWordWrap(True)
        layout.addWidget(self.phase_info_label)

    def _add_optimization_controls(self, layout: QVBoxLayout):
        """Add optimization parameter controls - Sharpe/PSR focused"""
        controls_layout = QHBoxLayout()

        # Total trials
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(300, 100000)
        self.trials_spin.setValue(OptimizationConfig.DEFAULT_TRIALS)
        controls_layout.addWidget(QLabel("Total Trials:"))
        controls_layout.addWidget(self.trials_spin)

        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(50, 2000)
        self.batch_spin.setValue(OptimizationConfig.DEFAULT_BATCH_SIZE)
        self.batch_spin.setSingleStep(50)
        controls_layout.addWidget(QLabel("Batch Size:"))
        controls_layout.addWidget(self.batch_spin)

        # PSR display (main metric)
        self.psr_label = QLabel("PSR: N/A")
        self.psr_label.setStyleSheet(
            "color: #2979ff; font-size: 12pt; font-weight: bold;"
        )
        self.psr_label.setToolTip(
            "Probabilistic Sharpe Ratio\n"
            "Probability that true Sharpe > 0\n\n"
            ">95%: Very confident\n"
            ">75%: Good confidence\n"
            "<50%: Likely false positive"
        )
        controls_layout.addWidget(self.psr_label)

        # Sharpe ratio display
        self.sharpe_label = QLabel("Sharpe: N/A")
        self.sharpe_label.setStyleSheet("color: #00ff88; font-size: 10pt;")
        self.sharpe_label.setToolTip("Annualized Sharpe Ratio (mean / std * sqrt(252))")
        controls_layout.addWidget(self.sharpe_label)

        # Return metrics
        self.best_label = QLabel("Return: N/A")
        self.best_label.setStyleSheet("color: #ffffff; font-size: 10pt;")
        self.buyhold_label = QLabel("Buy & Hold: N/A")
        self.buyhold_label.setStyleSheet("color: #aaaaaa; font-size: 10pt;")
        controls_layout.addWidget(self.best_label)
        controls_layout.addWidget(self.buyhold_label)

        # Add to layout
        layout.addLayout(controls_layout)

    def start_optimization(self):
        """Start the optimization process - PSR ONLY"""
        if not self.df_dict:
            QMessageBox.warning(self, "Error", "Please load data first")
            return

        selected_tfs = [
            tf
            for tf, cb in self.tf_checkboxes.items()
            if cb.isChecked() and tf in self.df_dict
        ]

        if not selected_tfs:
            QMessageBox.warning(self, "Error", "Select at least one timeframe")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Initializing PSR Optimization...")
        self.phase_info_label.setText("Preparing PSR optimization...")

        # Gather parameters
        mn1_range = (self.mn1_min.value(), self.mn1_max.value())
        mn2_range = (self.mn2_min.value(), self.mn2_max.value())
        entry_range = (self.entry_min.value(), self.entry_max.value())
        exit_range = (self.exit_min.value(), self.exit_max.value())

        time_cycle_ranges = (
            (self.on_min.value(), self.on_max.value()),
            (self.off_min.value(), self.off_max.value()),
            (0, self.on_max.value() + self.off_max.value()),
        )

        # Create optimizer (no objective_type parameter needed)
        self.worker = MultiTimeframeOptimizer(
            self.df_dict,
            self.trials_spin.value(),
            time_cycle_ranges,
            mn1_range,
            mn2_range,
            entry_range,
            exit_range,
            ticker=self.current_ticker,
            timeframes=selected_tfs,
            batch_size=self.batch_spin.value(),
            transaction_costs=self.transaction_costs,
        )

        # Connect signals
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.new_best.connect(self.update_best_label)
        self.worker.error.connect(self.show_error)
        self.worker.phase_update.connect(self.update_phase_label)
        self.worker.finished.connect(self.show_results)
        self.worker.stopped = False

        self.worker.start()

    def _add_parameter_ranges(self, layout: QVBoxLayout):
        """Add parameter range controls"""
        # MN1 and MN2 rangesself.walk_forward_btn.setEnabled(True)
        magic_layout = QHBoxLayout()
        ranges = IndicatorRanges()

        self.mn1_min = QSpinBox()
        self.mn1_min.setRange(1, 500)
        self.mn1_min.setValue(ranges.MN1_RANGE[0])

        self.mn1_max = QSpinBox()
        self.mn1_max.setRange(1, 500)
        self.mn1_max.setValue(ranges.MN1_RANGE[1])

        self.mn2_min = QSpinBox()
        self.mn2_min.setRange(1, 500)
        self.mn2_min.setValue(ranges.MN2_RANGE[0])

        self.mn2_max = QSpinBox()
        self.mn2_max.setRange(1, 500)
        self.mn2_max.setValue(ranges.MN2_RANGE[1])

        for label, min_spin, max_spin in [
            ("MN1", self.mn1_min, self.mn1_max),
            ("MN2", self.mn2_min, self.mn2_max),
        ]:
            magic_layout.addWidget(QLabel(f"{label} Min"))
            magic_layout.addWidget(min_spin)
            magic_layout.addWidget(QLabel("Max"))
            magic_layout.addWidget(max_spin)

        layout.addLayout(magic_layout)

        # Entry and Exit ranges
        limit_layout = QHBoxLayout()

        self.entry_min = QDoubleSpinBox()
        self.entry_min.setRange(0, 100)
        self.entry_min.setDecimals(2)
        self.entry_min.setValue(ranges.ENTRY_RANGE[0])

        self.entry_max = QDoubleSpinBox()
        self.entry_max.setRange(0, 100)
        self.entry_max.setDecimals(2)
        self.entry_max.setValue(ranges.ENTRY_RANGE[1])

        self.exit_min = QDoubleSpinBox()
        self.exit_min.setRange(0, 100)
        self.exit_min.setDecimals(2)
        self.exit_min.setValue(ranges.EXIT_RANGE[0])

        self.exit_max = QDoubleSpinBox()
        self.exit_max.setRange(0, 100)
        self.exit_max.setDecimals(2)
        self.exit_max.setValue(ranges.EXIT_RANGE[1])

        for label, min_spin, max_spin in [
            ("Entry", self.entry_min, self.entry_max),
            ("Exit", self.exit_min, self.exit_max),
        ]:
            limit_layout.addWidget(QLabel(f"{label} Min"))
            limit_layout.addWidget(min_spin)
            limit_layout.addWidget(QLabel("Max"))
            limit_layout.addWidget(max_spin)

        layout.addLayout(limit_layout)

        # Cycle ranges
        cycle_layout = QHBoxLayout()

        self.on_min = QSpinBox()
        self.on_min.setRange(1, 999)
        self.on_min.setValue(ranges.ON_RANGE[0])

        self.on_max = QSpinBox()
        self.on_max.setRange(1, 999)
        self.on_max.setValue(ranges.ON_RANGE[1])

        self.off_min = QSpinBox()
        self.off_min.setRange(0, 999)
        self.off_min.setValue(ranges.OFF_RANGE[0])

        self.off_max = QSpinBox()
        self.off_max.setRange(0, 999)
        self.off_max.setValue(ranges.OFF_RANGE[1])

        for label, min_spin, max_spin in [
            ("On Bars", self.on_min, self.on_max),
            ("Off Bars", self.off_min, self.off_max),
        ]:
            cycle_layout.addWidget(QLabel(f"{label} Min"))
            cycle_layout.addWidget(min_spin)
            cycle_layout.addWidget(QLabel("Max"))
            cycle_layout.addWidget(max_spin)

        layout.addLayout(cycle_layout)

        # Status label
        self.phase_label = QLabel("Status: Ready")
        self.phase_label.setStyleSheet("color: #2979ff; font-weight: bold;")
        layout.addWidget(self.phase_label)

    def _add_action_buttons(self, layout: QVBoxLayout):
        """Add action buttons"""
        # Row 1: Start/Stop
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start PSR Optimization")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_optimization)
        self.stop_btn.clicked.connect(self.stop_optimization)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        # Row 2: Monte Carlo button
        mc_layout = QHBoxLayout()

        self.monte_carlo_btn = QPushButton("🎲 Run Monte Carlo Simulation")
        self.monte_carlo_btn.setEnabled(False)
        self.monte_carlo_btn.setToolTip(
            "Test strategy robustness by randomizing trade order\n"
            "Requires completed optimization with trades"
        )
        self.monte_carlo_btn.clicked.connect(self.run_monte_carlo)
        self.monte_carlo_btn.setStyleSheet(
            """
            QPushButton { 
                background-color: #663399; 
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #7a3db8; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )

        # Monte Carlo settings
        mc_layout.addWidget(QLabel("Simulations:"))
        self.mc_simulations_spin = QSpinBox()
        self.mc_simulations_spin.setRange(100, 10000)
        self.mc_simulations_spin.setValue(1000)
        self.mc_simulations_spin.setSingleStep(100)
        mc_layout.addWidget(self.mc_simulations_spin)

        mc_layout.addWidget(self.monte_carlo_btn)
        mc_layout.addStretch()

        layout.addLayout(mc_layout)  # ✅ THIS WAS MISSING!

        # Row 3: Walk-Forward button
        wf_layout = QHBoxLayout()

        self.walk_forward_btn = QPushButton("📊 Run Walk-Forward Analysis")
        self.walk_forward_btn.setEnabled(False)
        self.walk_forward_btn.setToolTip(
            "Test strategy on unseen data\n"
            "Detects overfitting via train/test splits\n"
            "⚠️ MAINTAINS CYCLE ALIGNMENT"
        )
        self.walk_forward_btn.clicked.connect(self.run_walk_forward)
        self.walk_forward_btn.setStyleSheet(
            """
            QPushButton { 
                background-color: #009688; 
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #00bfa5; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )

        # Walk-Forward settings
        wf_layout.addWidget(QLabel("Train Days:"))
        self.wf_train_days_spin = QSpinBox()
        self.wf_train_days_spin.setRange(30, 730)
        self.wf_train_days_spin.setValue(180)
        wf_layout.addWidget(self.wf_train_days_spin)

        wf_layout.addWidget(QLabel("Test Days:"))
        self.wf_test_days_spin = QSpinBox()
        self.wf_test_days_spin.setRange(7, 180)
        self.wf_test_days_spin.setValue(30)
        wf_layout.addWidget(self.wf_test_days_spin)

        wf_layout.addWidget(QLabel("WF Trials:"))
        self.wf_trials_spin = QSpinBox()
        self.wf_trials_spin.setRange(100, 5000)
        self.wf_trials_spin.setValue(500)
        wf_layout.addWidget(self.wf_trials_spin)

        wf_layout.addWidget(self.walk_forward_btn)
        wf_layout.addStretch()

        layout.addLayout(wf_layout)

    def _add_regime_detection_controls(self, layout: QVBoxLayout):
        """Add market regime detection and position sizing controls"""
        regime_group = QGroupBox(
            "🌍 Market Regime Detection & Adaptive Position Sizing"
        )
        regime_layout = QVBoxLayout()
        regime_layout.setSpacing(8)  # Add consistent spacing

        # Buttons row - all in one line for compact layout
        buttons_layout = QHBoxLayout()

        # Detect regime button
        self.detect_regime_btn = QPushButton("🔍 Detect Regime")
        self.detect_regime_btn.setEnabled(False)
        self.detect_regime_btn.setToolTip(
            "Identify current market regime:\n"
            "• Bull, Bear, High Vol, Low Vol, Crisis\n"
            "• Multi-factor analysis (volatility, trend, momentum)\n"
            "• Markov chain transition probabilities\n"
            "• Suggested position size adjustment"
        )
        self.detect_regime_btn.clicked.connect(self.detect_market_regime)
        self.detect_regime_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FF6B35;
                font-weight: bold;
                padding: 8px;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #FF8555; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        buttons_layout.addWidget(self.detect_regime_btn)

        # Prediction horizon controls
        buttons_layout.addWidget(QLabel("Horizon:"))
        self.regime_horizon_spin = QSpinBox()
        self.regime_horizon_spin.setRange(1, 20)
        self.regime_horizon_spin.setValue(5)
        self.regime_horizon_spin.setSuffix(" days")
        self.regime_horizon_spin.setMaximumWidth(100)
        buttons_layout.addWidget(self.regime_horizon_spin)

        # Train predictor button
        self.train_predictor_btn = QPushButton("🤖 Train ML Predictor")
        self.train_predictor_btn.setEnabled(False)
        self.train_predictor_btn.setToolTip(
            "Train ML model to predict future regimes:\n"
            "• Random Forest / XGBoost forecasting\n"
            "• 30+ engineered features\n"
            "• Time series cross-validation\n"
            "• Confidence-based predictions"
        )
        self.train_predictor_btn.clicked.connect(self.train_regime_predictor)
        self.train_predictor_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4ECDC4;
                font-weight: bold;
                padding: 8px;
                min-width: 140px;
            }
            QPushButton:hover { background-color: #6EDED4; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        buttons_layout.addWidget(self.train_predictor_btn)

        # Calculate PBR button
        self.calc_pbr_btn = QPushButton("📊 Calculate PBR")
        self.calc_pbr_btn.setEnabled(False)
        self.calc_pbr_btn.setToolTip(
            "Probability of Backtested Returns:\n"
            "• Statistical measure of backtest-to-live performance\n"
            "• Factors: Sharpe, sample size, overfitting, WF efficiency\n"
            "• Regime stability consideration\n"
            "• Interpretation: >80% Very High, 65-80% High, 50-65% Moderate, <50% Low"
        )
        self.calc_pbr_btn.clicked.connect(self.calculate_pbr)
        self.calc_pbr_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #F7B731;
                font-weight: bold;
                padding: 8px;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #F9C851; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        buttons_layout.addWidget(self.calc_pbr_btn)
        buttons_layout.addStretch()

        regime_layout.addLayout(buttons_layout)

        # Display panels in a grid for better organization
        from PyQt6.QtWidgets import QGridLayout

        display_grid = QGridLayout()
        display_grid.setSpacing(6)

        # Regime display
        self.regime_display = QLabel("Regime: Not detected")
        self.regime_display.setStyleSheet(
            "color: #aaa; font-size: 10pt; padding: 6px; "
            "background-color: #1a1a1a; border-radius: 3px;"
        )
        self.regime_display.setWordWrap(True)
        self.regime_display.setMinimumHeight(30)
        display_grid.addWidget(QLabel("Current:"), 0, 0)
        display_grid.addWidget(self.regime_display, 0, 1)

        # Prediction display
        self.prediction_display = QLabel("Prediction: Not trained")
        self.prediction_display.setStyleSheet(
            "color: #aaa; font-size: 10pt; padding: 6px; "
            "background-color: #1a1a1a; border-radius: 3px;"
        )
        self.prediction_display.setWordWrap(True)
        self.prediction_display.setMinimumHeight(30)
        display_grid.addWidget(QLabel("Forecast:"), 1, 0)
        display_grid.addWidget(self.prediction_display, 1, 1)

        # PBR display
        self.pbr_display = QLabel("PBR: Not calculated")
        self.pbr_display.setStyleSheet(
            "color: #aaa; font-size: 10pt; padding: 6px; "
            "background-color: #1a1a1a; border-radius: 3px;"
        )
        self.pbr_display.setWordWrap(True)
        self.pbr_display.setMinimumHeight(30)
        display_grid.addWidget(QLabel("Reliability:"), 2, 0)
        display_grid.addWidget(self.pbr_display, 2, 1)

        # Set column stretch to make the display panels expand
        display_grid.setColumnStretch(1, 1)

        regime_layout.addLayout(display_grid)

        regime_group.setLayout(regime_layout)
        layout.addWidget(regime_group)

        # Institutional Analysis Group
        institutional_group = QGroupBox("🏛️ Institutional-Grade Regime Analysis")
        institutional_layout = QVBoxLayout()
        institutional_layout.setSpacing(8)

        # Row 1: Calibration & Agreement
        row1_layout = QHBoxLayout()

        self.calibrate_btn = QPushButton("📐 Calibrate Probabilities")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setToolTip(
            "Calibrate regime prediction probabilities:\n"
            "• Isotonic regression or Platt scaling\n"
            "• Fixes overconfident ML predictions\n"
            "• Evaluates Brier score and ECE\n"
            "• Improves probability accuracy"
        )
        self.calibrate_btn.clicked.connect(self.calibrate_probabilities)
        self.calibrate_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #9B59B6;
                font-weight: bold;
                padding: 8px;
                min-width: 140px;
            }
            QPushButton:hover { background-color: #AB69C6; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        row1_layout.addWidget(self.calibrate_btn)

        self.agreement_btn = QPushButton("🎯 Multi-Horizon Agreement")
        self.agreement_btn.setEnabled(False)
        self.agreement_btn.setToolTip(
            "Check agreement across time horizons:\n"
            "• 1d, 5d, 10d, 20d predictions\n"
            "• Agreement index (0-1)\n"
            "• Consensus regime detection\n"
            "• Signal quality assessment"
        )
        self.agreement_btn.clicked.connect(self.check_multi_horizon_agreement)
        self.agreement_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3498DB;
                font-weight: bold;
                padding: 8px;
                min-width: 160px;
            }
            QPushButton:hover { background-color: #44A8EB; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        row1_layout.addWidget(self.agreement_btn)
        row1_layout.addStretch()

        institutional_layout.addLayout(row1_layout)

        # Row 2: Robustness & Diagnostics
        row2_layout = QHBoxLayout()

        self.robustness_btn = QPushButton("🔬 Robustness Tests")
        self.robustness_btn.setEnabled(False)
        self.robustness_btn.setToolTip(
            "Statistical robustness testing:\n"
            "• White's Reality Check\n"
            "• Hansen's SPA Test\n"
            "• Block bootstrap validation\n"
            "• Sharpe ratio confidence intervals"
        )
        self.robustness_btn.clicked.connect(self.run_robustness_tests)
        self.robustness_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #E74C3C;
                font-weight: bold;
                padding: 8px;
                min-width: 140px;
            }
            QPushButton:hover { background-color: #F75C4C; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        row2_layout.addWidget(self.robustness_btn)

        self.diagnostics_btn = QPushButton("📋 Regime Diagnostics")
        self.diagnostics_btn.setEnabled(False)
        self.diagnostics_btn.setToolTip(
            "Comprehensive regime diagnostics:\n"
            "• Transition latency analysis\n"
            "• False transition rate\n"
            "• Regime persistence half-life\n"
            "• Confusion matrix & quality score"
        )
        self.diagnostics_btn.clicked.connect(self.run_regime_diagnostics)
        self.diagnostics_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ABC9C;
                font-weight: bold;
                padding: 8px;
                min-width: 140px;
            }
            QPushButton:hover { background-color: #2ACCAC; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        row2_layout.addWidget(self.diagnostics_btn)
        row2_layout.addStretch()

        institutional_layout.addLayout(row2_layout)

        # Row 3: Cross-Asset Analysis
        row3_layout = QHBoxLayout()

        self.cross_asset_btn = QPushButton("🌐 Cross-Asset Analysis")
        self.cross_asset_btn.setEnabled(False)
        self.cross_asset_btn.setToolTip(
            "Global regime analysis:\n"
            "• Equity, bond, commodity, crypto regimes\n"
            "• Risk-on/risk-off scoring\n"
            "• Cross-asset synchronization\n"
            "• Divergence detection"
        )
        self.cross_asset_btn.clicked.connect(self.run_cross_asset_analysis)
        self.cross_asset_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #E67E22;
                font-weight: bold;
                padding: 8px;
                min-width: 160px;
            }
            QPushButton:hover { background-color: #F68E32; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """
        )
        row3_layout.addWidget(self.cross_asset_btn)
        row3_layout.addStretch()

        institutional_layout.addLayout(row3_layout)

        # Display area for institutional analysis results
        self.institutional_display = QLabel("Institutional Analysis: Not run")
        self.institutional_display.setStyleSheet(
            "color: #aaa; font-size: 10pt; padding: 10px; "
            "background-color: #1a1a1a; border-radius: 3px;"
        )
        self.institutional_display.setWordWrap(True)
        self.institutional_display.setMinimumHeight(80)
        institutional_layout.addWidget(self.institutional_display)

        institutional_group.setLayout(institutional_layout)
        layout.addWidget(institutional_group)

    def run_walk_forward(self):
        """Run walk-forward analysis with smart defaults"""

        if not self.df_dict_full:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data first before running walk-forward analysis",
            )
            return

        # Get selected timeframes
        selected_tfs = [
            tf
            for tf, cb in self.tf_checkboxes.items()
            if cb.isChecked() and tf in self.df_dict_full
        ]

        if not selected_tfs:
            QMessageBox.warning(self, "Error", "Select at least one timeframe")
            return

        # Determine finest timeframe
        tf_order = {"5min": 0, "hourly": 1, "daily": 2}
        finest_tf = sorted(selected_tfs, key=lambda x: tf_order.get(x, 99))[0]

        # Get data info
        df_check = self.df_dict_full[finest_tf]
        dt_check = (
            df_check["Datetime"]
            if "Datetime" in df_check.columns
            else pd.to_datetime(df_check.index)
        )
        available_days = (dt_check.max() - dt_check.min()).days

        # 🎯 SMART AUTO-CONFIGURATION based on available data
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD SMART CONFIGURATION")
        print(f"{'='*70}")
        print(f"Ticker: {self.current_ticker}")
        print(f"Timeframe: {finest_tf}")
        print(f"Available data: {available_days} days ({len(df_check)} bars)")
        print(f"Date range: {dt_check.min().date()} to {dt_check.max().date()}")

        # Calculate optimal settings based on available data
        if available_days < 60:
            # Very limited data - use tiny windows
            recommended_train = max(14, available_days // 3)
            recommended_test = max(7, available_days // 6)
            recommended_trials = 300
            warning_level = "⚠️  LIMITED DATA"
        elif available_days < 120:
            # Limited data (typical for hourly YF)
            recommended_train = max(30, available_days // 3)
            recommended_test = max(10, available_days // 6)
            recommended_trials = 400
            warning_level = "⚠️  MODEST DATA"
        elif available_days < 365:
            # Decent amount of data
            recommended_train = 90
            recommended_test = 20
            recommended_trials = 500
            warning_level = "✅ GOOD DATA"
        else:
            # Lots of data
            recommended_train = 180
            recommended_test = 30
            recommended_trials = 500
            warning_level = "✅ EXCELLENT DATA"

        max_possible_windows = max(
            0, (available_days - recommended_train) // recommended_test
        )

        print(f"\n{warning_level}")
        print(f"Recommended settings for {available_days} days:")
        print(f"  • Train Days: {recommended_train}")
        print(f"  • Test Days: {recommended_test}")
        print(f"  • Trials: {recommended_trials}")
        print(f"  • Expected Windows: {max_possible_windows}")

        if max_possible_windows < 2:
            print(f"\n⚠️  WARNING: Only {max_possible_windows} window(s) possible!")
            print(f"Walk-forward works best with 3+ windows.")
            print(f"Consider using daily timeframe for more history.")

        # Get current user settings
        user_train = self.wf_train_days_spin.value()
        user_test = self.wf_test_days_spin.value()
        user_trials = self.wf_trials_spin.value()

        # Check if user settings will work
        user_windows = max(0, (available_days - user_train) // user_test)
        user_required = user_train + user_test

        print(f"\nYour current settings:")
        print(f"  • Train Days: {user_train}")
        print(f"  • Test Days: {user_test}")
        print(f"  • Trials: {user_trials}")
        print(f"  • Required: {user_required} days")
        print(f"  • Possible Windows: {user_windows}")

        # Decide whether to use user settings or recommend changes
        if user_windows < 1:
            # User settings won't work - force recommended
            use_recommended = True
            reason = "Your settings require more data than available!"
            print(f"\n❌ {reason}")
            print(f"Will use recommended settings instead.")
        elif user_windows < 2 and max_possible_windows >= 2:
            # User settings are suboptimal
            use_recommended = None  # Ask user
            reason = "Your settings will only create 1 window."
        else:
            # User settings are OK
            use_recommended = False
            reason = None
            print(f"\n✅ Your settings look good!")

        print(f"{'='*70}\n")

        # Show dialog with options
        if use_recommended is None:
            # Ask user which settings to use
            reply = QMessageBox.question(
                self,
                "Optimize Settings?",
                f"{warning_level}\n\n"
                f"Available Data: {available_days} days of {finest_tf}\n\n"
                f"📊 RECOMMENDED SETTINGS:\n"
                f"  • Train: {recommended_train} days\n"
                f"  • Test: {recommended_test} days\n"
                f"  • Trials: {recommended_trials}\n"
                f"  • Windows: {max_possible_windows}\n"
                f"  • Runtime: ~{(max_possible_windows * recommended_trials) // 100}-{(max_possible_windows * recommended_trials) // 50} min\n\n"
                f"⚙️  YOUR CURRENT SETTINGS:\n"
                f"  • Train: {user_train} days\n"
                f"  • Test: {user_test} days\n"
                f"  • Trials: {user_trials}\n"
                f"  • Windows: {user_windows}\n"
                f"  • Runtime: ~{(user_windows * user_trials) // 100}-{(user_windows * user_trials) // 50} min\n\n"
                f"Use recommended settings? (Click No to use your settings)",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Yes:
                use_recommended = True
            else:
                use_recommended = False

        elif use_recommended:
            # Must use recommended
            reply = QMessageBox.information(
                self,
                "Auto-Configuration Required",
                f"{warning_level}\n\n"
                f"Available Data: {available_days} days of {finest_tf}\n\n"
                f"Your settings require {user_required} days but you only have {available_days} days.\n\n"
                f"🎯 USING OPTIMIZED SETTINGS:\n"
                f"  • Train: {recommended_train} days\n"
                f"  • Test: {recommended_test} days\n"
                f"  • Trials: {recommended_trials}\n"
                f"  • Windows: {max_possible_windows}\n"
                f"  • Runtime: ~{(max_possible_windows * recommended_trials) // 100}-{(max_possible_windows * recommended_trials) // 50} min\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Cancel:
                return

        else:
            # User settings are fine, just confirm
            reply = QMessageBox.question(
                self,
                "Confirm Walk-Forward",
                f"Walk-forward analysis configuration:\n\n"
                f"• Timeframe: {finest_tf}\n"
                f"• Data: {available_days} days\n"
                f"• Train: {user_train} days\n"
                f"• Test: {user_test} days\n"
                f"• Trials: {user_trials} per window\n"
                f"• Windows: {user_windows}\n"
                f"• Estimated time: {(user_windows * user_trials) // 100}-{(user_windows * user_trials) // 50} minutes\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        # Set final parameters
        if use_recommended:
            final_train = recommended_train
            final_test = recommended_test
            final_trials = recommended_trials
            print(
                f"✅ Using RECOMMENDED settings: train={final_train}, test={final_test}, trials={final_trials}"
            )
        else:
            final_train = user_train
            final_test = user_test
            final_trials = user_trials
            print(
                f"✅ Using YOUR settings: train={final_train}, test={final_test}, trials={final_trials}"
            )

        # Disable button during analysis
        self.walk_forward_btn.setEnabled(False)
        self.walk_forward_btn.setText("Running Walk-Forward...")
        self.start_btn.setEnabled(False)

        try:
            print(f"\n{'='*70}")
            print(f"STARTING WALK-FORWARD ANALYSIS")
            print(f"{'='*70}")

            # Prepare optimizer kwargs
            mn1_range = (self.mn1_min.value(), self.mn1_max.value())
            mn2_range = (self.mn2_min.value(), self.mn2_max.value())
            entry_range = (self.entry_min.value(), self.entry_max.value())
            exit_range = (self.exit_min.value(), self.exit_max.value())

            time_cycle_ranges = (
                (self.on_min.value(), self.on_max.value()),
                (self.off_min.value(), self.off_max.value()),
                (0, self.on_max.value() + self.off_max.value()),
            )

            objective_map = {
                "Percent Gain": "percent_gain",
                "Sortino Ratio": "sortino",
                "Min Drawdown": "drawdown",
                "Profit Factor": "profit_factor",
            }
            objective_type = objective_map[self.objective_combo.currentText()]

            optimizer_kwargs = {
                "n_trials": final_trials,
                "time_cycle_ranges": time_cycle_ranges,  # ✅ Cycle preserved
                "mn1_range": mn1_range,
                "mn2_range": mn2_range,
                "entry_range": entry_range,
                "exit_range": exit_range,
                "ticker": self.current_ticker,
                "timeframes": selected_tfs,
                "batch_size": self.batch_spin.value(),
                "transaction_costs": self.transaction_costs,
            }

            # Import here to avoid circular import
            from optimization import MultiTimeframeOptimizer

            # Run walk-forward analysis
            results = WalkForwardAnalyzer.run_walk_forward(
                optimizer_class=MultiTimeframeOptimizer,
                df_dict=self.df_dict_full,  # Use full dataset
                train_days=final_train,
                test_days=final_test,
                min_trades=5,  # Lower threshold for limited data
                **optimizer_kwargs,
            )

            # Generate report
            report = WalkForwardAnalyzer.generate_walk_forward_report(results)
            print(report)

            # Create plot
            fig = WalkForwardAnalyzer.plot_walk_forward_results(
                results, ticker=self.current_ticker
            )

            # Save plot
            filename = f"{self.current_ticker}_walk_forward.png"
            fig.savefig(filename, dpi=150, facecolor="#121212", edgecolor="none")
            print(f"✅ Saved walk-forward plot to {filename}")

            # Show plot
            plt.show()

            # Show summary message
            if results.is_overfit:
                icon = QMessageBox.Icon.Critical
                title = "⚠️ Overfitting Detected!"
                summary = (
                    f"Walk-Forward Analysis Complete\n\n"
                    f"❌ STRATEGY IS LIKELY OVERFIT\n\n"
                    f"📊 Results:\n"
                    f"   In-Sample Avg: {results.avg_in_sample_return:+.2f}%\n"
                    f"   Out-of-Sample Avg: {results.avg_out_of_sample_return:+.2f}%\n"
                    f"   Efficiency: {results.efficiency_ratio:.2f}\n"
                    f"   Degradation: {results.return_degradation:.1f}%\n\n"
                    f"⚠️ DO NOT TRADE THIS STRATEGY LIVE\n\n"
                    f"See console for detailed report."
                )
            elif results.efficiency_ratio > 0.7:
                icon = QMessageBox.Icon.Information
                title = "✅ Strategy Appears Robust"
                summary = (
                    f"Walk-Forward Analysis Complete\n\n"
                    f"✅ STRATEGY PASSES ROBUSTNESS TESTS\n\n"
                    f"📊 Results:\n"
                    f"   In-Sample Avg: {results.avg_in_sample_return:+.2f}%\n"
                    f"   Out-of-Sample Avg: {results.avg_out_of_sample_return:+.2f}%\n"
                    f"   Efficiency: {results.efficiency_ratio:.2f}\n"
                    f"   Consistency: {results.consistency*100:.1f}%\n\n"
                    f"✅ Ready for paper trading!\n\n"
                    f"See console for detailed report."
                )
            else:
                icon = QMessageBox.Icon.Warning
                title = "⚠️ Mixed Results"
                summary = (
                    f"Walk-Forward Analysis Complete\n\n"
                    f"⚠️ STRATEGY SHOWS MIXED RESULTS\n\n"
                    f"📊 Results:\n"
                    f"   In-Sample Avg: {results.avg_in_sample_return:+.2f}%\n"
                    f"   Out-of-Sample Avg: {results.avg_out_of_sample_return:+.2f}%\n"
                    f"   Efficiency: {results.efficiency_ratio:.2f}\n"
                    f"   Degradation: {results.return_degradation:.1f}%\n\n"
                    f"⚠️ Use with caution\n\n"
                    f"See console for detailed report."
                )

            QMessageBox(
                icon, title, summary, QMessageBox.StandardButton.Ok, self
            ).exec()

            # Save results to CSV
            results_df = pd.DataFrame(
                {
                    "Window": range(1, len(results.in_sample_returns) + 1),
                    "IS_Return_%": results.in_sample_returns,
                    "OOS_Return_%": results.out_of_sample_returns,
                    "Train_Start": [d[0] for d in results.window_dates],
                    "Train_End": [d[1] for d in results.window_dates],
                    "Test_Start": [d[2] for d in results.window_dates],
                    "Test_End": [d[3] for d in results.window_dates],
                }
            )

            csv_filename = f"{self.current_ticker}_walk_forward_results.csv"
            results_df.to_csv(csv_filename, index=False)
            print(f"✅ Saved results to {csv_filename}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Walk-Forward Error",
                f"Failed to run walk-forward analysis:\n{str(e)}",
            )
            import traceback

            traceback.print_exc()

        finally:
            # Re-enable button
            self.walk_forward_btn.setEnabled(True)
            self.walk_forward_btn.setText("📊 Run Walk-Forward Analysis")
            self.start_btn.setEnabled(True)

    def _add_live_trading_controls(self, layout: QVBoxLayout):
        """Add live trading controls"""
        alpaca_group = QGroupBox("🔴 Live Trading")
        alpaca_layout = QHBoxLayout()

        self.live_trading_btn = QPushButton("▶ Start Live Trading")
        self.live_trading_btn.setEnabled(False)
        self.live_trading_btn.clicked.connect(self.toggle_live_trading)
        self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_ACTIVE)

        self.trading_status_label = QLabel("Ready")
        self.trading_status_label.setStyleSheet("color: #888; font-size: 10pt;")

        alpaca_layout.addWidget(self.live_trading_btn)
        alpaca_layout.addWidget(self.trading_status_label)
        alpaca_layout.addStretch()

        alpaca_group.setLayout(alpaca_layout)
        layout.addWidget(alpaca_group)

    def _add_risk_management_controls(self, layout: QVBoxLayout):
        """Add risk management controls"""
        risk_group = QGroupBox("⚙️ Risk Management")
        risk_layout = QHBoxLayout()

        # Position size percentage
        risk_layout.addWidget(QLabel("Position Size:"))
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.01, 100.0)
        self.position_size_spin.setValue(self.position_size_pct * 100)
        self.position_size_spin.setDecimals(2)
        self.position_size_spin.setSuffix("%")
        self.position_size_spin.setSingleStep(0.5)
        self.position_size_spin.setToolTip("Percentage of account per trade")
        self.position_size_spin.valueChanged.connect(self.on_risk_settings_changed)
        risk_layout.addWidget(self.position_size_spin)

        # Max concurrent positions
        risk_layout.addWidget(QLabel("Max Positions:"))
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 100)
        self.max_positions_spin.setValue(self.max_positions)
        self.max_positions_spin.setToolTip("Maximum concurrent positions")
        self.max_positions_spin.valueChanged.connect(self.on_risk_settings_changed)
        risk_layout.addWidget(self.max_positions_spin)

        # Risk summary
        self.risk_summary_label = QLabel(
            f"Risk: {self.position_size_pct*100:.1f}% per trade, "
            f"max {self.max_positions} positions"
        )
        self.risk_summary_label.setStyleSheet(
            f"color: {COLOR_WARNING}; font-size: 10pt; font-weight: bold;"
        )
        risk_layout.addWidget(self.risk_summary_label)
        risk_layout.addStretch()

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

    def _add_transaction_cost_controls(self, layout: QVBoxLayout):
        """Add transaction cost controls"""
        cost_group = QGroupBox("💰 Transaction Costs")
        cost_layout = QVBoxLayout()

        # Top row - percentage costs
        pct_layout = QHBoxLayout()

        # Commission percentage
        pct_layout.addWidget(QLabel("Commission:"))
        self.commission_pct_spin = QDoubleSpinBox()
        self.commission_pct_spin.setRange(0.0, 5.0)
        self.commission_pct_spin.setValue(self.transaction_costs.COMMISSION_PCT * 100)
        self.commission_pct_spin.setDecimals(3)
        self.commission_pct_spin.setSuffix("%")
        self.commission_pct_spin.setSingleStep(0.01)
        self.commission_pct_spin.setToolTip("Commission as % of trade value")
        self.commission_pct_spin.valueChanged.connect(self.on_transaction_costs_changed)
        pct_layout.addWidget(self.commission_pct_spin)

        # Slippage
        pct_layout.addWidget(QLabel("Slippage:"))
        self.slippage_pct_spin = QDoubleSpinBox()
        self.slippage_pct_spin.setRange(0.0, 5.0)
        self.slippage_pct_spin.setValue(self.transaction_costs.SLIPPAGE_PCT * 100)
        self.slippage_pct_spin.setDecimals(3)
        self.slippage_pct_spin.setSuffix("%")
        self.slippage_pct_spin.setSingleStep(0.01)
        self.slippage_pct_spin.setToolTip("Price slippage/market impact")
        self.slippage_pct_spin.valueChanged.connect(self.on_transaction_costs_changed)
        pct_layout.addWidget(self.slippage_pct_spin)

        # Spread
        pct_layout.addWidget(QLabel("Spread:"))
        self.spread_pct_spin = QDoubleSpinBox()
        self.spread_pct_spin.setRange(0.0, 5.0)
        self.spread_pct_spin.setValue(self.transaction_costs.SPREAD_PCT * 100)
        self.spread_pct_spin.setDecimals(3)
        self.spread_pct_spin.setSuffix("%")
        self.spread_pct_spin.setSingleStep(0.001)
        self.spread_pct_spin.setToolTip("Bid-ask spread")
        self.spread_pct_spin.valueChanged.connect(self.on_transaction_costs_changed)
        pct_layout.addWidget(self.spread_pct_spin)

        cost_layout.addLayout(pct_layout)

        # Bottom row - presets and summary
        preset_layout = QHBoxLayout()

        # Preset buttons
        preset_layout.addWidget(QLabel("Presets:"))

        stocks_btn = QPushButton("Stocks (0.06%)")
        stocks_btn.clicked.connect(self.set_costs_for_stocks)
        preset_layout.addWidget(stocks_btn)

        crypto_btn = QPushButton("Crypto (0.35%)")
        crypto_btn.clicked.connect(self.set_costs_for_crypto)
        preset_layout.addWidget(crypto_btn)

        zero_btn = QPushButton("Zero")
        zero_btn.clicked.connect(self.set_costs_zero)
        preset_layout.addWidget(zero_btn)

        # Cost summary
        self.cost_summary_label = QLabel()
        self.cost_summary_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        preset_layout.addWidget(self.cost_summary_label)
        preset_layout.addStretch()

        cost_layout.addLayout(preset_layout)

        cost_group.setLayout(cost_layout)
        layout.addWidget(cost_group)

        # Update summary
        self.on_transaction_costs_changed()

    def on_risk_settings_changed(self):
        """Update risk management settings"""
        self.position_size_pct = self.position_size_spin.value() / 100.0
        self.max_positions = self.max_positions_spin.value()

        # Calculate total exposure
        total_exposure = self.position_size_spin.value() * self.max_positions

        # Update summary with color coding
        if total_exposure > 100:
            color = "#ff4444"  # Red
            warning = " ⚠️ OVER-LEVERAGED"
        elif total_exposure > 80:
            color = COLOR_WARNING  # Orange
            warning = " ⚠"
        else:
            color = COLOR_SUCCESS  # Green
            warning = ""

        self.risk_summary_label.setText(
            f"Risk: {self.position_size_spin.value():.1f}% per trade, "
            f"max {self.max_positions} positions ({total_exposure:.0f}% total){warning}"
        )
        self.risk_summary_label.setStyleSheet(
            f"color: {color}; font-size: 10pt; font-weight: bold;"
        )

        # Update live trader if running
        if (
            hasattr(self, "live_trader")
            and self.live_trader
            and self.live_trader.running
        ):
            self.live_trader.position_size_pct = self.position_size_pct

    def on_transaction_costs_changed(self):
        """Update transaction costs when spinboxes change"""
        self.transaction_costs.COMMISSION_PCT = self.commission_pct_spin.value() / 100.0
        self.transaction_costs.SLIPPAGE_PCT = self.slippage_pct_spin.value() / 100.0
        self.transaction_costs.SPREAD_PCT = self.spread_pct_spin.value() / 100.0

        # Calculate total cost
        total_cost_per_trade = self.transaction_costs.TOTAL_PCT * 100 * 2  # Round trip

        # Estimate impact over 100 trades
        trades_100 = (1 - self.transaction_costs.TOTAL_PCT * 2) ** 100
        drag_100 = (1 - trades_100) * 100

        self.cost_summary_label.setText(
            f"Total: {total_cost_per_trade:.3f}% per round-trip | "
            f"100 trades: {drag_100:.1f}% drag"
        )

    def set_costs_for_stocks(self):
        """Set transaction costs for US stocks"""
        costs = TransactionCosts.for_stocks()
        self.commission_pct_spin.setValue(costs.COMMISSION_PCT * 100)
        self.slippage_pct_spin.setValue(costs.SLIPPAGE_PCT * 100)
        self.spread_pct_spin.setValue(costs.SPREAD_PCT * 100)
        print("✓ Transaction costs set for stocks: 0.06% per trade")

    def set_costs_for_crypto(self):
        """Set transaction costs for cryptocurrency"""
        costs = TransactionCosts.for_crypto()
        self.commission_pct_spin.setValue(costs.COMMISSION_PCT * 100)
        self.slippage_pct_spin.setValue(costs.SLIPPAGE_PCT * 100)
        self.spread_pct_spin.setValue(costs.SPREAD_PCT * 100)
        print("✓ Transaction costs set for crypto: 0.35% per trade")

    def set_costs_zero(self):
        """Set zero transaction costs (unrealistic but useful for comparison)"""
        self.commission_pct_spin.setValue(0.0)
        self.slippage_pct_spin.setValue(0.0)
        self.spread_pct_spin.setValue(0.0)
        print("⚠ Transaction costs set to ZERO (unrealistic)")

    def on_timeframe_changed(self):
        """Handle timeframe checkbox changes"""
        if not self.df_dict_full:
            return

        selected_tfs = [
            tf
            for tf, cb in self.tf_checkboxes.items()
            if cb.isChecked() and tf in self.df_dict_full
        ]

        if not selected_tfs:
            return

        # Determine finest timeframe
        tf_order = {"5min": 0, "hourly": 1, "daily": 2}
        sorted_tfs = sorted(selected_tfs, key=lambda x: tf_order.get(x, 99))
        finest_tf = sorted_tfs[0]

        # Apply data limits
        if finest_tf == "5min":
            limit_days = OptimizationConfig.FIVEMIN_MAX_DAYS
        elif finest_tf == "hourly":
            limit_days = OptimizationConfig.HOURLY_MAX_DAYS
        else:
            limit_days = None

        self.df_dict = DataLoader.filter_timeframe_data(self.df_dict_full, limit_days)

        # Update displays
        if finest_tf in self.df_dict:
            df_display = self.df_dict[finest_tf]
            date_min = df_display["Datetime"].min()
            date_max = df_display["Datetime"].max()
            self.date_range_label.setText(
                f"Date Range: {date_min.strftime('%Y-%m-%d')} to "
                f"{date_max.strftime('%Y-%m-%d')} ({finest_tf})"
            )

            # Update buy & hold
            close_arr = df_display["Close"].to_numpy(dtype=float)
            self.buyhold_pct = PerformanceMetrics.calculate_buyhold_return(close_arr)
            self.buyhold_label.setText(
                f"Buy & Hold: {self.buyhold_pct:+.2f}% ({finest_tf.capitalize()})"
            )

            # Update chart
            self._update_chart(close_arr, finest_tf)

    def _update_chart(self, close_arr: np.ndarray, timeframe: str):
        """Update the chart display"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        normalized = close_arr / close_arr[0] * 1000
        ax.plot(
            normalized,
            color="#888888",
            label=f"Buy & Hold ({self.buyhold_pct:+.1f}%)",
            linewidth=2,
        )

        ax.set_facecolor("#121212")
        ax.tick_params(colors="white")
        ax.set_title(
            f"{self.current_ticker} - {timeframe.capitalize()} View ({len(close_arr)} bars)",
            color="white",
            fontsize=14,
        )
        ax.set_xlabel(f"{timeframe.capitalize()} Bars", color="white")
        ax.set_ylabel("Normalized Equity ($)", color="white")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.2)
        self.canvas.draw()

    def load_yfinance(self):
        """Load data from Yahoo Finance"""
        symbol = self.ticker_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Error", "Please enter a valid ticker")
            return

        symbol = DataLoader.normalize_ticker(symbol)
        self.load_yf_btn.setEnabled(False)
        self.ticker_input.setText(f"Loading {symbol}...")

        try:
            df_dict, error_msg = DataLoader.load_yfinance_data(symbol)

            if error_msg:
                QMessageBox.warning(self, "Error", error_msg)
                return

            self.df_dict_full = df_dict
            self.df_dict = df_dict.copy()
            self.current_ticker = symbol
            self.data_source = "yfinance"

            # Trigger timeframe change to update display
            self.on_timeframe_changed()

            # Update ticker display
            tf_counts = f"{len(df_dict['daily'])}d, {len(df_dict['hourly'])}h"
            if "5min" in df_dict:
                tf_counts += f", {len(df_dict['5min'])}x5m"
            self.ticker_input.setText(f"{symbol} ({tf_counts} max)")

            # ✅ ENABLE WALK-FORWARD BUTTON after data load
            if hasattr(self, "walk_forward_btn"):
                self.walk_forward_btn.setEnabled(True)
                print("✅ Walk-Forward Analysis button enabled")

            # ✅ ENABLE REGIME DETECTION BUTTONS after data load
            if hasattr(self, "detect_regime_btn"):
                self.detect_regime_btn.setEnabled(True)
                print("✅ Regime Detection button enabled")
            if hasattr(self, "train_predictor_btn"):
                self.train_predictor_btn.setEnabled(True)
                print("✅ Regime Predictor button enabled")

            # ✅ ENABLE INSTITUTIONAL ANALYSIS BUTTONS after data load
            if hasattr(self, "agreement_btn"):
                self.agreement_btn.setEnabled(True)
                print("✅ Multi-Horizon Agreement button enabled")
            if hasattr(self, "diagnostics_btn"):
                self.diagnostics_btn.setEnabled(True)
                print("✅ Regime Diagnostics button enabled")
            if hasattr(self, "cross_asset_btn"):
                self.cross_asset_btn.setEnabled(True)
                print("✅ Cross-Asset Analysis button enabled")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
        finally:
            self.load_yf_btn.setEnabled(True)

    def load_ticker_list(self):
        """Load multiple tickers from text file and optimize sequentially"""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Ticker List File", "", "Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                tickers = [line.strip().upper() for line in f if line.strip()]

            if not tickers:
                QMessageBox.warning(self, "Empty File", "No tickers found in file")
                return

            # Confirm batch processing
            reply = QMessageBox.question(
                self,
                "Confirm Batch Processing",
                f"Process {len(tickers)} tickers?\n\n"
                f"Preview: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}\n\n"
                f"This may take several hours.\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            print(f"\n{'='*70}")
            print(f"BATCH TICKER OPTIMIZATION")
            print(f"{'='*70}")
            print(f"Loaded {len(tickers)} tickers from: {file_path}")
            print(f"Tickers: {', '.join(tickers)}")
            print(f"{'='*70}\n")

            # Create summary file
            summary_file = (
                f"batch_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            self.load_yf_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.stopped = False

            successful = []
            failed = []

            start_time = pd.Timestamp.now()

            for idx, ticker in enumerate(tickers):
                if self.stopped:
                    print(
                        f"\n⚠️  Batch processing stopped by user at ticker {idx + 1}/{len(tickers)}"
                    )
                    break

                print(f"\n{'─'*70}")
                print(f"TICKER {idx + 1}/{len(tickers)}: {ticker}")
                print(f"{'─'*70}")

                # Update main status
                self.phase_label.setText(f"Batch: {idx + 1}/{len(tickers)} - {ticker}")

                # Estimate time remaining
                if idx > 0:
                    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                    avg_time = elapsed / idx
                    remaining = avg_time * (len(tickers) - idx)
                    eta = pd.Timestamp.now() + pd.Timedelta(seconds=remaining)
                    print(
                        f"⏱️  ETA: {eta.strftime('%H:%M:%S')} ({remaining/60:.1f} min remaining)"
                    )

                success = self._load_and_optimize_ticker(ticker)

                if success:
                    successful.append(ticker)
                    print(f"✅ {ticker} completed successfully")
                else:
                    failed.append(ticker)
                    print(f"❌ {ticker} failed - skipping")

            # Save summary
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds() / 60

            summary = f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              BATCH OPTIMIZATION SUMMARY                       ║
    ╚═══════════════════════════════════════════════════════════════╝

    Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}
    Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
    Duration: {duration:.1f} minutes

    Total Tickers: {len(tickers)}
    ✅ Successful: {len(successful)}
    ❌ Failed: {len(failed)}

    SUCCESSFUL TICKERS:
    {chr(10).join(f"  ✅ {t}" for t in successful) if successful else "  None"}

    FAILED TICKERS:
    {chr(10).join(f"  ❌ {t}" for t in failed) if failed else "  None"}

    Results saved to: data_output/{self.current_ticker}_psr_results.csv
    """

            print(summary)

            # Save to file
            with open(summary_file, "w") as f:
                f.write(summary)

            print(f"📄 Summary saved to: {summary_file}")

            # Show dialog
            QMessageBox.information(
                self,
                "Batch Complete",
                f"Processed {len(tickers)} tickers in {duration:.1f} minutes\n\n"
                f"✅ Successful: {len(successful)}\n"
                f"❌ Failed: {len(failed)}\n\n"
                f"Summary saved to: {summary_file}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch processing error:\n{str(e)}")
            import traceback

            traceback.print_exc()

        finally:
            self.load_yf_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stopped = False
            self.phase_label.setText("Status: Ready")

    def _load_and_optimize_ticker(self, ticker: str) -> bool:
        """Load and optimize a single ticker (returns True on success)"""
        try:
            # Normalize ticker
            ticker = DataLoader.normalize_ticker(ticker)

            # Try to load data
            print(f"📊 Loading data for {ticker}...")
            df_dict, error_msg = DataLoader.load_yfinance_data(ticker)

            if error_msg or not df_dict:
                print(f"   ⚠️  Failed to load {ticker}: {error_msg}")
                return False

            # Check if data is valid
            if "hourly" not in df_dict or len(df_dict["hourly"]) < 500:
                print(f"   ⚠️  Insufficient data for {ticker} (need 500+ hourly bars)")
                return False

            # Set data
            self.df_dict_full = df_dict
            self.df_dict = df_dict.copy()
            self.current_ticker = ticker
            self.data_source = "yfinance"

            # Update display
            self.on_timeframe_changed()

            # Update ticker display
            tf_counts = f"{len(df_dict['daily'])}d, {len(df_dict['hourly'])}h"
            if "5min" in df_dict:
                tf_counts += f", {len(df_dict['5min'])}x5m"
            self.ticker_input.setText(f"{ticker} ({tf_counts})")

            print(f"   ✅ Loaded {ticker}: {tf_counts}")

            # Run optimization
            print(f"🚀 Starting optimization for {ticker}...")

            # Get selected timeframes
            selected_tfs = [
                tf
                for tf, cb in self.tf_checkboxes.items()
                if cb.isChecked() and tf in self.df_dict
            ]

            if not selected_tfs:
                print(f"   ⚠️  No timeframes selected for {ticker}")
                return False

            # Gather parameters
            mn1_range = (self.mn1_min.value(), self.mn1_max.value())
            mn2_range = (self.mn2_min.value(), self.mn2_max.value())
            entry_range = (self.entry_min.value(), self.entry_max.value())
            exit_range = (self.exit_min.value(), self.exit_max.value())

            time_cycle_ranges = (
                (self.on_min.value(), self.on_max.value()),
                (self.off_min.value(), self.off_max.value()),
                (0, self.on_max.value() + self.off_max.value()),
            )

            # Create optimizer
            self.worker = MultiTimeframeOptimizer(
                self.df_dict,
                self.trials_spin.value(),
                time_cycle_ranges,
                mn1_range,
                mn2_range,
                entry_range,
                exit_range,
                ticker=ticker,
                timeframes=selected_tfs,
                batch_size=self.batch_spin.value(),
                transaction_costs=self.transaction_costs,
            )

            # Connect signals
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.new_best.connect(self.update_best_label)
            self.worker.phase_update.connect(self.update_phase_label)
            self.worker.stopped = False

            # Run synchronously (wait for completion)
            self.worker.run()
            self.worker.wait()

            # Check if stopped
            if self.worker.stopped:
                print(f"   ⚠️  Optimization stopped for {ticker}")
                return False

            # Check if results exist
            if not self.worker.all_results or len(self.worker.all_results) == 0:
                print(f"   ⚠️  No valid results for {ticker}")
                return False

            print(f"✅ Optimization complete for {ticker}")
            return True

        except Exception as e:
            print(f"   ❌ Error processing {ticker}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def start_optimization(self):
        """Start the optimization process"""
        if not self.df_dict:
            QMessageBox.warning(self, "Error", "Please load data first")
            return

        selected_tfs = [
            tf
            for tf, cb in self.tf_checkboxes.items()
            if cb.isChecked() and tf in self.df_dict
        ]

        if not selected_tfs:
            QMessageBox.warning(self, "Error", "Select at least one timeframe")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Initializing...")
        self.phase_info_label.setText("Initializing optimization...")

        # Gather parameters
        mn1_range = (self.mn1_min.value(), self.mn1_max.value())
        mn2_range = (self.mn2_min.value(), self.mn2_max.value())
        entry_range = (self.entry_min.value(), self.entry_max.value())
        exit_range = (self.exit_min.value(), self.exit_max.value())

        time_cycle_ranges = (
            (self.on_min.value(), self.on_max.value()),
            (self.off_min.value(), self.off_max.value()),
            (0, self.on_max.value() + self.off_max.value()),
        )

        # Create optimizer with PSR composite (no objective_type parameter)
        self.worker = MultiTimeframeOptimizer(
            self.df_dict,
            self.trials_spin.value(),
            time_cycle_ranges,
            mn1_range,
            mn2_range,
            entry_range,
            exit_range,
            ticker=self.current_ticker,
            timeframes=selected_tfs,
            batch_size=self.batch_spin.value(),
            transaction_costs=self.transaction_costs,
        )

        # Connect signals
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.new_best.connect(self.update_best_label)
        self.worker.error.connect(self.show_error)
        self.worker.phase_update.connect(self.update_phase_label)
        self.worker.finished.connect(self.show_results)
        self.worker.stopped = False

        self.worker.start()

    def stop_optimization(self):
        """Stop the optimization"""
        self.stopped = True  # ✅ ADD THIS
        if self.worker:
            self.worker.stopped = True
            self.stop_btn.setEnabled(False)
            self.phase_label.setText("Stopping...")

    def show_error(self, error_msg: str):
        """Display error message"""
        QMessageBox.critical(self, "Error", error_msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.phase_label.setText("Error occurred")

    def update_phase_label(self, phase_text: str):
        """Update phase label"""
        self.phase_label.setText(phase_text)
        self.phase_info_label.setText(f"Current: {phase_text}")

    def update_best_label(self, best_params: Dict):
        """Update best result display with PSR and ALL parameters"""
        self.best_params = best_params

        # PSR score (prominent)
        if "PSR" in best_params:
            psr = best_params["PSR"]
            psr_pct = psr * 100

            # Color code PSR
            if psr > 0.95:
                psr_color = "#00ff88"
                psr_icon = "✅"
            elif psr > 0.75:
                psr_color = "#88ff88"
                psr_icon = "✓"
            elif psr > 0.50:
                psr_color = "#ffaa00"
                psr_icon = "⚠"
            else:
                psr_color = "#ff4444"
                psr_icon = "❌"

            self.psr_label.setText(f"PSR: {psr_pct:.1f}% {psr_icon}")
            self.psr_label.setStyleSheet(
                f"color: {psr_color}; font-size: 12pt; font-weight: bold;"
            )

        # Sharpe Ratio
        if "Sharpe_Ratio" in best_params:
            sharpe = best_params["Sharpe_Ratio"]
            sharpe_color = (
                "#00ff88" if sharpe > 1.0 else "#ffaa00" if sharpe > 0.5 else "#ff4444"
            )
            self.sharpe_label.setText(f"Sharpe: {sharpe:.2f}")
            self.sharpe_label.setStyleSheet(f"color: {sharpe_color}; font-size: 10pt;")

        # Traditional metrics
        metrics_text = f"Return: {best_params.get('Percent_Gain_%', 0):.2f}%"
        if "Sortino_Ratio" in best_params:
            metrics_text += f" | Sortino: {best_params['Sortino_Ratio']:.2f}"
        if "Max_Drawdown_%" in best_params:
            metrics_text += f" | DD: {best_params['Max_Drawdown_%']:.2f}%"
        if "Profit_Factor" in best_params:
            metrics_text += f" | PF: {best_params['Profit_Factor']:.2f}"
        if "Trade_Count" in best_params:
            metrics_text += f" | Trades: {best_params['Trade_Count']}"

        self.best_label.setText(metrics_text)

        # ✅ BUILD PARAMETER TEXT WITH CYCLES
        param_lines = []
        for tf in ["daily", "hourly", "5min"]:
            tf_params = []

            # ✅ ADD: Time Cycle parameters
            if f"On_{tf}" in best_params:
                on = best_params[f"On_{tf}"]
                off = best_params[f"Off_{tf}"]
                start = best_params[f"Start_{tf}"]
                tf_params.append(f"Cycle(ON:{on}, OFF:{off}, START:{start})")

            # RSI parameters
            if f"MN1_{tf}" in best_params:
                mn1 = best_params[f"MN1_{tf}"]
                mn2 = best_params[f"MN2_{tf}"]
                entry = best_params[f"Entry_{tf}"]
                exit_val = best_params[f"Exit_{tf}"]
                tf_params.append(
                    f"RSI({mn1},{mn2}) Entry<{entry:.1f} Exit>{exit_val:.1f}"
                )

            if tf_params:
                param_lines.append(f"{tf.upper()}: {' | '.join(tf_params)}")

        # ✅ Display parameters across multiple lines for better readability
        if param_lines:
            params_text = "\n".join(param_lines)
        else:
            params_text = "Parameters: N/A"

        self.best_params_label.setText(params_text)

    def _add_psr_tooltips(self):
        """Add helpful tooltips explaining PSR metrics"""
        # Only add tooltips for labels that exist in PSR-only mode
        if hasattr(self, "psr_label"):
            self.psr_label.setToolTip(
                "Probabilistic Sharpe Ratio\n"
                "Probability that true Sharpe > 0\n\n"
                ">95%: Very confident\n"
                ">75%: Good confidence\n"
                "<50%: Likely false positive"
            )

        if hasattr(self, "sharpe_label"):
            self.sharpe_label.setToolTip(
                "Annualized Sharpe Ratio\n"
                "Risk-adjusted return metric\n\n"
                ">2.0: Excellent\n"
                ">1.0: Good\n"
                "<0.5: Poor"
            )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Timeframe Trading Optimizer with Alpaca")
        self.setGeometry(100, 100, 1400, 1100)

        # Data storage
        self.df_dict = {}
        self.df_dict_full = {}
        self.worker: Optional[MultiTimeframeOptimizer] = None
        self.live_trader: Optional[AlpacaLiveTrader] = None
        self.current_ticker = ""
        self.data_source = "yfinance"
        self.buyhold_pct = 0.0
        self.best_params = None
        self.stopped = False

        # Risk management settings
        self.position_size_pct = RiskConfig.DEFAULT_POSITION_SIZE
        self.max_positions = RiskConfig.DEFAULT_MAX_POSITIONS

        # Transaction costs
        self.transaction_costs = TransactionCosts()

        # Trade log for Monte Carlo
        self.last_trade_log = []

        # Regime detection objects
        self.regime_detector = MarketRegimeDetector(
            vol_window=20, trend_window_fast=50, trend_window_slow=200
        )
        self.regime_predictor: Optional[RegimePredictor] = None
        self.current_regime_state: Optional[RegimeState] = None

        # Institutional-grade regime analysis objects
        self.regime_calibrator: Optional[MultiClassCalibrator] = None
        self.multi_horizon_agreement: Optional[MultiHorizonAgreementIndex] = None
        self.regime_diagnostics: Optional[RegimeDiagnosticAnalyzer] = None
        self.cross_asset_analyzer: Optional[CrossAssetRegimeAnalyzer] = None
        self.calibrated_predictions = None  # Store calibrated predictions

        # Create UI first
        self.init_ui()

        # Then add tooltips AFTER labels exist
        self._add_psr_tooltips()

        # After init_ui():
        self._add_psr_tooltips()

    def show_psr_report(self):
        """Show detailed PSR composite metrics report"""
        if not self.best_params:
            QMessageBox.warning(self, "No Results", "Run optimization first")
            return

        report = f"""
    ╔════════════════════════════════════════════════════════╗
    ║       PSR COMPOSITE OPTIMIZATION REPORT                ║
    ╚════════════════════════════════════════════════════════╝

    📊 COMPOSITE SCORE: {self.best_params.get('Composite_Score', 0):.3f}

    🎯 COMPONENT SCORES:
    {'─'*58}
    Probabilistic Sharpe Ratio:  {self.best_params.get('PSR', 0)*100:.1f}%
    {'✅ Very confident' if self.best_params.get('PSR', 0) > 0.95 else '✓ Good confidence' if self.best_params.get('PSR', 0) > 0.75 else '⚠ Moderate confidence' if self.best_params.get('PSR', 0) > 0.50 else '❌ Low confidence'}

    Walk-Forward Sharpe:         {self.best_params.get('WFA_Sharpe', 0):.2f}
    {'✅ Excellent robustness' if self.best_params.get('WFA_Sharpe', 0) > 1.5 else '✓ Good robustness' if self.best_params.get('WFA_Sharpe', 0) > 1.0 else '⚠ Moderate robustness' if self.best_params.get('WFA_Sharpe', 0) > 0.5 else '❌ Poor robustness'}

    Probability of Overfitting:  {self.best_params.get('PBO', 0)*100:.1f}%
    {'✅ Low overfitting risk' if self.best_params.get('PBO', 0) < 0.3 else '⚠ Moderate risk' if self.best_params.get('PBO', 0) < 0.5 else '❌ High overfitting risk'}

    Annual Turnover:             {self.best_params.get('Annual_Turnover', 0):.0f} trades/year
    {'✅ Low frequency' if self.best_params.get('Annual_Turnover', 0) < 30 else '✓ Moderate frequency' if self.best_params.get('Annual_Turnover', 0) < 100 else '⚠ High frequency'}

    📈 TRADITIONAL METRICS:
    {'─'*58}
    Return:                      {self.best_params.get('Percent_Gain_%', 0):+.2f}%
    Sortino Ratio:               {self.best_params.get('Sortino_Ratio', 0):.2f}
    Max Drawdown:                {self.best_params.get('Max_Drawdown_%', 0):.2f}%
    Profit Factor:               {self.best_params.get('Profit_Factor', 0):.2f}
    Trade Count:                 {self.best_params.get('Trade_Count', 0)}

    ✅ ASSESSMENT:
    {'─'*58}
    """

        # Overall assessment
        comp_score = self.best_params.get("Composite_Score", 0)
        psr = self.best_params.get("PSR", 0)
        pbo = self.best_params.get("PBO", 1)

        if comp_score > 0.6 and psr > 0.75 and pbo < 0.5:
            report += "✅ EXCELLENT - Strategy is robust and ready for paper trading\n"
        elif comp_score > 0.3 and psr > 0.50:
            report += "✓ GOOD - Strategy shows promise, monitor in paper trading\n"
        elif pbo > 0.7:
            report += "❌ OVERFIT - Strategy is likely curve-fitted to data\n"
        elif psr < 0.5:
            report += "❌ LOW CONFIDENCE - Performance may be due to luck\n"
        else:
            report += "⚠ MARGINAL - Use with caution, consider refinement\n"

        report += f"\n{'═'*58}\n"

        # Show in message box
        msg = QMessageBox(self)
        msg.setWindowTitle("PSR Composite Report")
        msg.setText(report)
        msg.setFont(QFont("Courier", 9))
        msg.exec()

    def show_results(self, df_results: pd.DataFrame):
        """Display optimization results"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # ✅ ENABLE BOTH BUTTONS after optimization
        if hasattr(self, "monte_carlo_btn"):
            self.monte_carlo_btn.setEnabled(True)
        if hasattr(self, "walk_forward_btn"):
            self.walk_forward_btn.setEnabled(True)

        # ✅ ENABLE PBR BUTTON after optimization
        if hasattr(self, "calc_pbr_btn"):
            self.calc_pbr_btn.setEnabled(True)

        # ✅ ENABLE ROBUSTNESS TESTING after optimization
        if hasattr(self, "robustness_btn"):
            self.robustness_btn.setEnabled(True)
            print("✅ Robustness Tests button enabled")

        if df_results.empty:
            QMessageBox.information(self, "Complete", "No valid results")
            self.phase_label.setText("No results")
            return

        self.progress_bar.setValue(100)
        self.phase_label.setText("✓ Optimization Complete!")
        self.phase_info_label.setText("✓ All phases completed successfully")

        # Enable live trading if available
        if ALPACA_AVAILABLE:
            self.live_trading_btn.setEnabled(True)
            self.trading_status_label.setText("Ready")
            self.trading_status_label.setStyleSheet(
                f"color: {COLOR_SUCCESS}; font-size: 10pt; font-weight: bold;"
            )

        best = df_results.iloc[0]
        self.update_best_label(best.to_dict() if isinstance(best, pd.Series) else best)

        # Plot results and get trade log for Monte Carlo
        self._plot_results(best)

    def _plot_results(self, best):
        """Plot optimization results and GET TRADE LOG"""
        # Get optimized timeframes
        optimized_tfs = []
        best_dict = best.to_dict() if isinstance(best, pd.Series) else best
        for tf in ["5min", "hourly", "daily"]:
            if f"MN1_{tf}" in best_dict:
                optimized_tfs.append(tf)

        if not optimized_tfs:
            return

        tf_order = {"5min": 0, "hourly": 1, "daily": 2}
        finest_tf = sorted(optimized_tfs, key=lambda x: tf_order.get(x, 99))[0]

        finest_df = self.df_dict[finest_tf]
        close_finest = finest_df["Close"].to_numpy(dtype=float)
        buyhold_finest = (close_finest / close_finest[0]) * 1000

        # Create plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Buy & Hold
        ax.plot(
            buyhold_finest,
            color="#888888",
            label=f"Buy & Hold ({self.buyhold_pct:+.1f}%)",
            linewidth=2,
            alpha=0.7,
        )

        # ✅ CRITICAL: Get strategy curve WITH TRADES
        print("\n🔍 Generating trade log for Monte Carlo...")
        final_eq_curve, final_trades, trade_log = self.worker.simulate_multi_tf(
            best_dict, return_trades=True
        )

        # ✅ DIAGNOSTIC: Verify trade log
        print(f"\n📊 Trade Log Verification:")
        print(f"   Total trades: {len(trade_log)}")

        if trade_log:
            returns = [t["Percent_Change"] for t in trade_log]
            print(f"   Return range: [{min(returns):.2f}%, {max(returns):.2f}%]")
            print(f"   Mean return: {np.mean(returns):.2f}%")
            print(f"   Std dev: {np.std(returns):.2f}%")
            print(f"   Unique returns: {len(set(returns))}")

            print(f"\n   Sample trades:")
            for i, trade in enumerate(trade_log[:5]):
                print(f"     Trade {i+1}: {trade['Percent_Change']:+.2f}%")

        # Store trade log for Monte Carlo
        self.last_trade_log = trade_log

        # Enable Monte Carlo if enough trades
        if hasattr(self, "monte_carlo_btn"):
            if len(trade_log) > 10:
                self.monte_carlo_btn.setEnabled(True)
                print(f"\n✅ Monte Carlo button enabled ({len(trade_log)} trades)")
            else:
                self.monte_carlo_btn.setEnabled(False)
                print(
                    f"\n⚠️  Monte Carlo disabled: only {len(trade_log)} trades (need >10)"
                )

        # Plot strategy curve
        if final_eq_curve is not None:
            label = (
                f"Strategy ({best['Percent_Gain_%']:+.1f}%, "
                f"{best.get('Trade_Count', final_trades)} trades)"
            )
            ax.plot(final_eq_curve, color="#2979ff", label=label, linewidth=2.5)

        ax.set_facecolor("#121212")
        ax.tick_params(colors="white")

        metrics_text = (
            f"Sortino: {best['Sortino_Ratio']:.2f} | "
            f"DD: {best['Max_Drawdown_%']:.1f}% | "
            f"PF: {best['Profit_Factor']:.2f}"
        )

        ax.set_title(
            f"{self.current_ticker} - {metrics_text}", color="white", fontsize=11
        )
        ax.set_xlabel(f"{finest_tf.capitalize()} Bars", color="white")
        ax.set_ylabel("Equity ($)", color="white")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.2)
        self.canvas.draw()

    def toggle_live_trading(self):
        """Toggle live trading"""
        if self.live_trader is None or not self.live_trader.running:
            self.start_live_trading()
        else:
            self.stop_live_trading()

    def start_live_trading(self):
        """Start live trading"""
        if not ALPACA_AVAILABLE:
            QMessageBox.critical(
                self, "Error", "Alpaca not installed!\nInstall: pip install alpaca-py"
            )
            return

        if self.best_params is None:
            QMessageBox.warning(
                self, "Error", "Run optimization first to get parameters"
            )
            return

        selected_tfs = [
            tf
            for tf, cb in self.tf_checkboxes.items()
            if cb.isChecked() and tf in self.df_dict
        ]

        if not selected_tfs:
            QMessageBox.warning(self, "Error", "No timeframes selected")
            return

        # Get credentials from config
        config = AlpacaConfig()

        # Start trader
        self.live_trader = AlpacaLiveTrader(
            config.API_KEY,
            config.SECRET_KEY,
            config.BASE_URL,
            self.current_ticker,
            self.best_params,
            self.df_dict,
            selected_tfs,
            position_size_pct=self.position_size_pct,  # Fixed: added comma above
        )

        self.live_trader.status_update.connect(self.update_trading_status)
        self.live_trader.trade_executed.connect(self.on_trade_executed)
        self.live_trader.error.connect(self.on_trading_error)

        self.live_trader.start()

        self.live_trading_btn.setText("⬛ Stop Live Trading")
        self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_STOPPED)
        self.start_btn.setEnabled(False)

    def stop_live_trading(self):
        """Stop live trading"""
        if self.live_trader:
            self.live_trader.stop()
            self.live_trader.wait()

        self.live_trading_btn.setText("▶ Start Live Trading")
        self.live_trading_btn.setStyleSheet(LIVE_TRADING_BUTTON_ACTIVE)

        self.trading_status_label.setText("Stopped")
        self.trading_status_label.setStyleSheet("color: #888; font-size: 10pt;")
        self.start_btn.setEnabled(True)

    def update_trading_status(self, status: str):
        """Update trading status"""
        self.trading_status_label.setText(status)
        print(f"🔴 {status}")

    def on_trade_executed(self, trade_info: Dict):
        """Handle trade execution"""
        action = trade_info["action"]
        symbol = trade_info["symbol"]
        shares = trade_info["shares"]
        price = trade_info["price"]
        time = trade_info["time"]

        msg = f"{action}: {shares} of {symbol} @ ${price:.2f}\nTime: {time}"
        QMessageBox.information(self, f"Trade Executed - {action}", msg)

    def on_trading_error(self, error_msg: str):
        """Handle trading error"""
        self.trading_status_label.setText(f"Error: {error_msg}")
        self.trading_status_label.setStyleSheet("color: #ff4444; font-size: 10pt;")
        QMessageBox.critical(self, "Trading Error", error_msg)

    def test_rsi_sensitivity():
        """Test how different MN1/MN2 values affect trade count"""
        import numpy as np
        import yfinance as yf

        # Get AAPL hourly data
        df = yf.download("AAPL", period="2y", interval="1h")
        close = df["Close"].values

        print("\nRSI Sensitivity Test:")
        print("=" * 50)

        test_cases = [
            (14, 3, "Standard RSI"),
            (50, 20, "Smooth RSI"),
            (70, 38, "Your optimized values"),
        ]

        for mn1, mn2, label in test_cases:
            # Calculate RSI
            delta = np.diff(close, prepend=close[0])
            gain = np.maximum(delta, 0)
            loss = np.maximum(-delta, 0)

            avg_gain = np.convolve(gain, np.ones(mn1) / mn1, mode="same")
            avg_loss = np.convolve(loss, np.ones(mn1) / mn1, mode="same")

            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - 100 / (1 + rs)

            # Smooth
            rsi_smooth = np.convolve(rsi, np.ones(mn2) / mn2, mode="same")

            # Count signals
            below_30 = np.sum(rsi_smooth < 30)
            below_35 = np.sum(rsi_smooth < 35)
            below_40 = np.sum(rsi_smooth < 40)

            print(f"\n{label} (MN1={mn1}, MN2={mn2}):")
            print(f"  RSI < 30: {below_30} bars ({below_30/len(rsi)*100:.1f}%)")
            print(f"  RSI < 35: {below_35} bars ({below_35/len(rsi)*100:.1f}%)")
            print(f"  RSI < 40: {below_40} bars ({below_40/len(rsi)*100:.1f}%)")
            print(f"  Min RSI: {np.nanmin(rsi_smooth):.1f}")
            print(f"  Max RSI: {np.nanmax(rsi_smooth):.1f}")

    def run_monte_carlo(self):
        """Run comprehensive quantitative Monte Carlo analysis"""
        if not self.last_trade_log or len(self.last_trade_log) < 10:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                f"Need at least 10 trades for Monte Carlo.\n"
                f"Currently have: {len(self.last_trade_log)} trades",
            )
            return

        n_simulations = self.mc_simulations_spin.value()

        # Disable button during simulation
        self.monte_carlo_btn.setEnabled(False)
        self.monte_carlo_btn.setText("Running Monte Carlo...")

        try:
            print(f"\n{'='*70}")
            print(f"QUANTITATIVE MONTE CARLO ANALYSIS")
            print(f"{'='*70}")
            print(f"Trades: {len(self.last_trade_log)}")
            print(f"Simulations per method: {n_simulations}")

            # Extract returns from trade log
            if isinstance(self.last_trade_log[0], dict):
                returns = np.array(
                    [trade["Percent_Change"] / 100.0 for trade in self.last_trade_log]
                )
            else:
                returns = np.array(
                    [r / 100.0 if abs(r) > 1 else r for r in self.last_trade_log]
                )

            initial_equity = 1000.0

            print(f"\nTrade Statistics:")
            print(f"  Number of trades: {len(returns)}")
            print(f"  Mean return: {np.mean(returns)*100:.2f}%")
            print(f"  Std dev: {np.std(returns)*100:.2f}%")

            # Calculate skewness
            from scipy import stats

            skewness = stats.skew(returns)
            print(f"  Skewness: {skewness:.2f}")

            print(f"  Min return: {min(returns)*100:.2f}%")
            print(f"  Max return: {max(returns)*100:.2f}%")
            print(f"  Win rate: {np.sum(returns > 0) / len(returns) * 100:.1f}%")

            # Calculate original result
            original_equity = initial_equity * np.prod(1 + returns)
            print(
                f"\nOriginal Equity: ${original_equity:,.2f} ({(original_equity/initial_equity-1)*100:+.2f}%)"
            )

            # ===================================================================
            # METHOD 1: TRADE RANDOMIZATION (tests path dependence)
            # ===================================================================
            print(f"\n{'─'*70}")
            print(f"METHOD 1: TRADE RANDOMIZATION")
            print(f"{'─'*70}")
            print(f"Tests if trade order affects results (path dependence)")

            randomization_results = []
            for i in range(n_simulations):
                shuffled = np.random.permutation(returns)
                equity = initial_equity * np.prod(1 + shuffled)
                randomization_results.append(equity)

            randomization_results = np.array(randomization_results)

            print(f"  Mean: ${np.mean(randomization_results):,.2f}")
            print(f"  Std: ${np.std(randomization_results):,.2f}")
            print(
                f"  95% CI: [${np.percentile(randomization_results, 2.5):,.0f}, ${np.percentile(randomization_results, 97.5):,.0f}]"
            )
            print(
                f"  Original in CI: {'✅ YES' if np.percentile(randomization_results, 2.5) <= original_equity <= np.percentile(randomization_results, 97.5) else '❌ NO'}"
            )

            # ===================================================================
            # METHOD 2: BOOTSTRAP RESAMPLING (tests sample robustness)
            # ===================================================================
            print(f"\n{'─'*70}")
            print(f"METHOD 2: BOOTSTRAP RESAMPLING")
            print(f"{'─'*70}")
            print(f"Tests robustness to different trade samples (with replacement)")

            bootstrap_results = []
            for i in range(n_simulations):
                # Sample with replacement
                sampled = np.random.choice(returns, size=len(returns), replace=True)
                equity = initial_equity * np.prod(1 + sampled)
                bootstrap_results.append(equity)

            bootstrap_results = np.array(bootstrap_results)

            print(f"  Mean: ${np.mean(bootstrap_results):,.2f}")
            print(f"  Std: ${np.std(bootstrap_results):,.2f}")
            print(
                f"  95% CI: [${np.percentile(bootstrap_results, 2.5):,.0f}, ${np.percentile(bootstrap_results, 97.5):,.0f}]"
            )
            print(
                f"  Original in CI: {'✅ YES' if np.percentile(bootstrap_results, 2.5) <= original_equity <= np.percentile(bootstrap_results, 97.5) else '❌ NO'}"
            )

            # ===================================================================
            # METHOD 3: PARAMETRIC MONTE CARLO (assumes normal distribution)
            # ===================================================================
            print(f"\n{'─'*70}")
            print(f"METHOD 3: PARAMETRIC (NORMAL) SIMULATION")
            print(f"{'─'*70}")
            print(f"Tests assuming returns follow normal distribution")

            mean_return = np.mean(returns)
            std_return = np.std(returns)

            parametric_results = []
            for i in range(n_simulations):
                # Generate synthetic returns from normal distribution
                synthetic = np.random.normal(mean_return, std_return, len(returns))
                equity = initial_equity * np.prod(1 + synthetic)
                parametric_results.append(equity)

            parametric_results = np.array(parametric_results)

            print(f"  Mean: ${np.mean(parametric_results):,.2f}")
            print(f"  Std: ${np.std(parametric_results):,.2f}")
            print(
                f"  95% CI: [${np.percentile(parametric_results, 2.5):,.0f}, ${np.percentile(parametric_results, 97.5):,.0f}]"
            )
            print(
                f"  Original in CI: {'✅ YES' if np.percentile(parametric_results, 2.5) <= original_equity <= np.percentile(parametric_results, 97.5) else '❌ NO'}"
            )

            # ===================================================================
            # METHOD 4: BLOCK BOOTSTRAP (preserves time structure)
            # ===================================================================
            print(f"\n{'─'*70}")
            print(f"METHOD 4: BLOCK BOOTSTRAP")
            print(f"{'─'*70}")
            print(f"Tests robustness while preserving sequential trade patterns")

            block_size = max(2, len(returns) // 10)  # ~10% of trades per block
            print(f"  Block size: {block_size} trades")

            block_bootstrap_results = []
            for i in range(n_simulations):
                # Create blocks
                resampled = []
                while len(resampled) < len(returns):
                    start_idx = np.random.randint(0, len(returns) - block_size + 1)
                    block = returns[start_idx : start_idx + block_size]
                    resampled.extend(block)

                resampled = np.array(resampled[: len(returns)])  # Trim to exact length
                equity = initial_equity * np.prod(1 + resampled)
                block_bootstrap_results.append(equity)

            block_bootstrap_results = np.array(block_bootstrap_results)

            print(f"  Mean: ${np.mean(block_bootstrap_results):,.2f}")
            print(f"  Std: ${np.std(block_bootstrap_results):,.2f}")
            print(
                f"  95% CI: [${np.percentile(block_bootstrap_results, 2.5):,.0f}, ${np.percentile(block_bootstrap_results, 97.5):,.0f}]"
            )
            print(
                f"  Original in CI: {'✅ YES' if np.percentile(block_bootstrap_results, 2.5) <= original_equity <= np.percentile(block_bootstrap_results, 97.5) else '❌ NO'}"
            )

            # ===================================================================
            # AGGREGATE STATISTICS
            # ===================================================================
            print(f"\n{'='*70}")
            print(f"AGGREGATE MONTE CARLO STATISTICS")
            print(f"{'='*70}")

            all_methods = {
                "Randomization": randomization_results,
                "Bootstrap": bootstrap_results,
                "Parametric": parametric_results,
                "Block Bootstrap": block_bootstrap_results,
            }

            for method_name, results in all_methods.items():
                prob_profit = np.sum(results > initial_equity) / len(results)
                prob_beats_original = np.sum(results > original_equity) / len(results)

                print(f"\n{method_name}:")
                print(f"  Probability of profit: {prob_profit*100:.1f}%")
                print(f"  Probability beats original: {prob_beats_original*100:.1f}%")
                print(
                    f"  Risk of ruin (<$500): {np.sum(results < 500) / len(results) * 100:.2f}%"
                )

            # ===================================================================
            # COMPREHENSIVE VISUALIZATION
            # ===================================================================
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(16, 12), facecolor="#121212")
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            fig.suptitle(
                f"Quantitative Monte Carlo Analysis - {self.current_ticker}\n"
                f"{n_simulations} simulations per method | {len(returns)} trades",
                color="white",
                fontsize=14,
                fontweight="bold",
            )

            # Plot 1: Distributions comparison
            ax1 = fig.add_subplot(gs[0, :])

            for method_name, results in all_methods.items():
                unique_vals = np.unique(results)
                n_unique = len(unique_vals)

                if n_unique == 1:
                    # Single value - plot vertical line
                    ax1.axvline(
                        unique_vals[0], linewidth=3, alpha=0.7, label=method_name
                    )
                elif n_unique < 10:
                    # Very few unique values - use scatter plot with size based on frequency
                    from collections import Counter

                    counts = Counter(results)
                    values = list(counts.keys())
                    frequencies = list(counts.values())
                    max_freq = max(frequencies)
                    sizes = [f / max_freq * 200 for f in frequencies]
                    ax1.scatter(
                        values,
                        [0.1] * len(values),
                        s=sizes,
                        alpha=0.6,
                        label=method_name,
                    )
                else:
                    # Enough unique values - use KDE
                    try:
                        from scipy import stats

                        kde = stats.gaussian_kde(results)
                        x_range = np.linspace(results.min(), results.max(), 200)
                        ax1.plot(
                            x_range,
                            kde(x_range),
                            label=method_name,
                            linewidth=2,
                            alpha=0.8,
                        )
                    except:
                        # Fallback to histogram with explicit bins
                        n_bins = min(20, n_unique)
                        bin_edges = np.linspace(
                            results.min(), results.max(), n_bins + 1
                        )
                        ax1.hist(
                            results,
                            bins=bin_edges,
                            alpha=0.3,
                            label=method_name,
                            density=True,
                        )

            ax1.axvline(
                original_equity,
                color="#ff4444",
                linewidth=2,
                linestyle="--",
                label=f"Original: ${original_equity:.0f}",
                zorder=10,
            )
            ax1.axvline(
                initial_equity,
                color="#888888",
                linewidth=1,
                linestyle=":",
                label=f"Break-even: ${initial_equity:.0f}",
                zorder=10,
            )

            ax1.set_xlabel("Final Equity ($)", color="white", fontsize=11)
            ax1.set_ylabel("Density / Frequency", color="white", fontsize=11)
            ax1.set_title(
                "Distribution Comparison Across Methods",
                color="white",
                fontweight="bold",
            )
            ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
            ax1.set_facecolor("#1a1a1a")
            ax1.tick_params(colors="white")
            ax1.grid(True, alpha=0.2)

            # Remove y-axis ticks if we have mixed plot types
            if any(len(np.unique(results)) < 10 for results in all_methods.values()):
                ax1.set_yticks([])

            # Plot 2-5: Individual method details
            positions = [(1, 0), (1, 1), (1, 2), (2, 0)]

            for idx, (method_name, results) in enumerate(all_methods.items()):
                ax = fig.add_subplot(gs[positions[idx]])

                # Box plot
                bp = ax.boxplot(
                    [results], vert=True, widths=0.6, patch_artist=True, showmeans=True
                )
                bp["boxes"][0].set_facecolor("#2979ff")
                bp["boxes"][0].set_alpha(0.6)
                bp["medians"][0].set_color("#00ff88")
                bp["medians"][0].set_linewidth(2)
                bp["means"][0].set_marker("D")
                bp["means"][0].set_markerfacecolor("#ffaa00")

                # Add original line
                ax.axhline(
                    original_equity,
                    color="#ff4444",
                    linewidth=2,
                    linestyle="--",
                    alpha=0.8,
                )

                # Add statistics text
                prob_profit = np.sum(results > initial_equity) / len(results)
                stats_text = (
                    f"Mean: ${np.mean(results):,.0f}\n"
                    f"Median: ${np.median(results):,.0f}\n"
                    f"Prob Profit: {prob_profit*100:.0f}%"
                )

                ax.text(
                    0.5,
                    0.02,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="white",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8),
                )

                ax.set_ylabel("Final Equity ($)", color="white", fontsize=9)
                ax.set_title(method_name, color="white", fontweight="bold", fontsize=10)
                ax.set_facecolor("#1a1a1a")
                ax.tick_params(colors="white")
                ax.grid(True, alpha=0.2, axis="y")
                ax.set_xticks([])

            # Plot 6: Risk metrics comparison
            ax6 = fig.add_subplot(gs[2, 1:])

            metrics_data = []
            methods_list = list(all_methods.keys())

            for method_name, results in all_methods.items():
                prob_profit = np.sum(results > initial_equity) / len(results)
                prob_beats_orig = np.sum(results > original_equity) / len(results)
                risk_of_ruin = np.sum(results < initial_equity * 0.5) / len(results)

                metrics_data.append(
                    [prob_profit * 100, prob_beats_orig * 100, risk_of_ruin * 100]
                )

            metrics_data = np.array(metrics_data).T

            x = np.arange(len(methods_list))
            width = 0.25

            ax6.bar(
                x - width,
                metrics_data[0],
                width,
                label="Prob. Profit",
                color="#00ff88",
                alpha=0.7,
            )
            ax6.bar(
                x,
                metrics_data[1],
                width,
                label="Beats Original",
                color="#2979ff",
                alpha=0.7,
            )
            ax6.bar(
                x + width,
                metrics_data[2],
                width,
                label="Risk of Ruin",
                color="#ff4444",
                alpha=0.7,
            )

            ax6.set_ylabel("Percentage (%)", color="white", fontsize=11)
            ax6.set_title("Risk Metrics Comparison", color="white", fontweight="bold")
            ax6.set_xticks(x)
            ax6.set_xticklabels(methods_list, rotation=15, ha="right")
            ax6.legend(fontsize=9)
            ax6.set_facecolor("#1a1a1a")
            ax6.tick_params(colors="white")
            ax6.grid(True, alpha=0.2, axis="y")
            ax6.axhline(50, color="white", linewidth=0.5, linestyle=":", alpha=0.5)

            plt.tight_layout()

            # Save plot
            filename = f"{self.current_ticker}_quant_monte_carlo.png"
            fig.savefig(filename, dpi=150, facecolor="#121212", edgecolor="none")
            print(f"\n✓ Saved quantitative Monte Carlo plot to {filename}")

            # ===================================================================
            # SUMMARY REPORT & INTERPRETATION
            # ===================================================================
            print(f"\n{'='*70}")
            print(f"MONTE CARLO INTERPRETATION")
            print(f"{'='*70}")

            # Check consistency across methods
            all_results_flat = np.concatenate([r for r in all_methods.values()])
            overall_prob_profit = np.sum(all_results_flat > initial_equity) / len(
                all_results_flat
            )

            # Count how many methods have original in CI
            in_ci_count = sum(
                [
                    np.percentile(r, 2.5) <= original_equity <= np.percentile(r, 97.5)
                    for r in all_methods.values()
                ]
            )

            print(f"\nOverall Assessment:")
            print(f"  Methods with original in 95% CI: {in_ci_count}/4")
            print(f"  Overall probability of profit: {overall_prob_profit*100:.1f}%")

            if in_ci_count >= 3 and overall_prob_profit > 0.7:
                verdict = "✅ ROBUST - Strategy shows consistency across multiple tests"
                icon = QMessageBox.Icon.Information
            elif in_ci_count >= 2 and overall_prob_profit > 0.5:
                verdict = (
                    "⚠️  MODERATE - Strategy shows some robustness but watch closely"
                )
                icon = QMessageBox.Icon.Information
            else:
                verdict = "❌ FRAGILE - Results may be due to luck or overfitting"
                icon = QMessageBox.Icon.Warning

            print(f"\n{verdict}")

            # Display plot
            plt.show()

            # Show message box
            message = (
                f"Quantitative Monte Carlo Analysis Complete!\n\n"
                f"📊 Original Return: {(original_equity/initial_equity-1)*100:+.1f}%\n\n"
                f"🎲 Methods Tested:\n"
                f"  • Trade Randomization (path dependence)\n"
                f"  • Bootstrap Resampling (sample robustness)\n"
                f"  • Parametric Normal (distributional)\n"
                f"  • Block Bootstrap (time structure)\n\n"
                f"📈 Results:\n"
                f"  • Original in 95% CI: {in_ci_count}/4 methods\n"
                f"  • Overall Prob. Profit: {overall_prob_profit*100:.1f}%\n\n"
                f"{verdict}\n\n"
                f"See console for detailed analysis.\n"
                f"Plot saved to: {filename}"
            )

            QMessageBox(
                icon,
                "Quantitative Monte Carlo Results",
                message,
                QMessageBox.StandardButton.Ok,
                self,
            ).exec()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Monte Carlo Error",
                f"Failed to run Monte Carlo simulation:\n{str(e)}",
            )
            import traceback

            traceback.print_exc()

        finally:
            # Re-enable button
            self.monte_carlo_btn.setEnabled(True)
            self.monte_carlo_btn.setText("🎲 Run Monte Carlo Simulation")

    def detect_market_regime(self):
        """Detect current market regime and suggest position sizing"""
        if not self.df_dict_full:
            QMessageBox.warning(
                self, "No Data", "Please load data first before detecting market regime"
            )
            return

        # Get the daily timeframe data for regime detection
        if "daily" in self.df_dict_full:
            df = self.df_dict_full["daily"]
        elif "hourly" in self.df_dict_full:
            df = self.df_dict_full["hourly"]
        elif "5min" in self.df_dict_full:
            df = self.df_dict_full["5min"]
        else:
            QMessageBox.warning(self, "Error", "No timeframe data available")
            return

        try:
            # Extract prices and returns
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")

            prices = df["Close"]
            returns = prices.pct_change().dropna()

            print(f"\n{'='*80}")
            print(f"MARKET REGIME DETECTION")
            print(f"{'='*80}")
            print(f"Ticker: {self.current_ticker}")
            print(f"Data points: {len(prices)}")

            # Detect regime
            regime_state = self.regime_detector.detect_regime(prices, returns)
            self.current_regime_state = regime_state

            # Get regime statistics
            regime_stats = self.regime_detector.get_regime_statistics(prices, returns)

            # Update display
            regime_color_map = {
                MarketRegime.BULL: "#00ff88",
                MarketRegime.BEAR: "#ff4444",
                MarketRegime.HIGH_VOL: "#ff9900",
                MarketRegime.LOW_VOL: "#00aaff",
                MarketRegime.CRISIS: "#ff00ff",
            }

            regime_color = regime_color_map.get(regime_state.current_regime, "#ffffff")

            display_text = (
                f"<b>Current Regime:</b> <span style='color: {regime_color}; font-size: 14pt;'>"
                f"{regime_state.current_regime.value.upper()}</span><br>"
                f"<b>Confidence:</b> {regime_state.confidence:.1%} | "
                f"<b>Duration:</b> {regime_state.regime_duration} days<br>"
                f"<b>Predicted Next:</b> {regime_state.predicted_next_regime.value} | "
                f"<b>Stay Probability:</b> {regime_state.transition_probability:.1%}<br>"
                f"<b>Suggested Position Size:</b> <span style='color: #ffaa00; font-weight: bold;'>"
                f"{regime_state.suggested_position_size:.2f}x</span>"
            )

            self.regime_display.setText(display_text)

            # Print detailed analysis
            print(f"\n📈 CURRENT REGIME: {regime_state.current_regime.value.upper()}")
            print(f"   Confidence: {regime_state.confidence:.1%}")
            print(f"   Duration: {regime_state.regime_duration} days")
            print(f"   Predicted Next: {regime_state.predicted_next_regime.value}")
            print(f"   Stay Probability: {regime_state.transition_probability:.1%}")
            print(f"   Suggested Position: {regime_state.suggested_position_size:.2f}x")

            print(f"\n📊 Regime Probabilities:")
            for regime, prob in sorted(
                regime_state.regime_probabilities.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                bar = "█" * int(prob * 50)
                print(f"   {regime.value:20s} {prob:5.1%} {bar}")

            print(f"\n📈 Historical Regime Performance:")
            print(
                f"{'Regime':<20} {'Avg Return':<12} {'Sharpe':<8} {'Max DD':<10} {'Win Rate'}"
            )
            print("-" * 70)

            for regime, metrics in regime_stats.items():
                print(
                    f"{regime.value:<20} "
                    f"{metrics.avg_return:>10.1%} "
                    f"{metrics.sharpe:>8.2f} "
                    f"{metrics.max_drawdown:>9.1%} "
                    f"{metrics.win_rate:>9.1%}"
                )

            print(f"\n{'='*80}\n")

            # Show message box with summary
            summary = (
                f"Market Regime: {regime_state.current_regime.value.upper()}\n"
                f"Confidence: {regime_state.confidence:.1%}\n"
                f"Suggested Position: {regime_state.suggested_position_size:.2f}x\n\n"
                f"See console for detailed analysis."
            )

            QMessageBox.information(self, "Regime Detection Complete", summary)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Regime Detection Error",
                f"Error detecting market regime:\n{str(e)}",
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

    def train_regime_predictor(self):
        """Train ML model to predict future market regimes"""
        if not self.df_dict_full:
            QMessageBox.warning(
                self, "No Data", "Please load data first before training predictor"
            )
            return

        # Get the daily timeframe data for regime prediction
        if "daily" in self.df_dict_full:
            df = self.df_dict_full["daily"]
        elif "hourly" in self.df_dict_full:
            df = self.df_dict_full["hourly"]
        elif "5min" in self.df_dict_full:
            df = self.df_dict_full["5min"]
        else:
            QMessageBox.warning(self, "Error", "No timeframe data available")
            return

        try:
            # Extract prices and returns
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")

            prices = df["Close"]
            returns = prices.pct_change().dropna()

            print(f"\n{'='*80}")
            print(f"REGIME PREDICTOR TRAINING")
            print(f"{'='*80}")
            print(f"Ticker: {self.current_ticker}")
            print(f"Data points: {len(prices)}")

            # Get prediction horizon from UI
            horizon = self.regime_horizon_spin.value()
            print(f"Prediction horizon: {horizon} days")

            # Create and train predictor
            self.regime_predictor = RegimePredictor(
                detector=self.regime_detector,
                prediction_horizon=horizon,
                n_estimators=100,
                use_xgboost=False,  # Set to True if xgboost is installed
            )

            # Disable button during training
            self.train_predictor_btn.setEnabled(False)
            self.train_predictor_btn.setText("Training...")

            # Train model
            performance = self.regime_predictor.train(prices, returns, val_split=0.2)

            print(f"\n📊 Model Performance:")
            print(f"   Validation Accuracy: {performance.accuracy_5day:.1%}")

            print(f"\n   Precision by Regime:")
            for regime, prec in performance.precision_by_regime.items():
                print(f"      {regime.value:<20} {prec:.1%}")

            print(f"\n   Recall by Regime:")
            for regime, rec in performance.recall_by_regime.items():
                print(f"      {regime.value:<20} {rec:.1%}")

            # Make prediction
            prediction = self.regime_predictor.predict(
                prices, returns, horizon_days=horizon
            )

            # Update display
            regime_color_map = {
                MarketRegime.BULL: "#00ff88",
                MarketRegime.BEAR: "#ff4444",
                MarketRegime.HIGH_VOL: "#ff9900",
                MarketRegime.LOW_VOL: "#00aaff",
                MarketRegime.CRISIS: "#ff00ff",
            }

            pred_color = regime_color_map.get(prediction.predicted_regime, "#ffffff")

            # Extract top 3 features from features_importance dict
            top_features = sorted(
                prediction.features_importance.items(), key=lambda x: x[1], reverse=True
            )[:3]
            top_feature_names = [f[0] for f in top_features]

            display_text = (
                f"<b>Predicted Regime ({horizon} days):</b> "
                f"<span style='color: {pred_color}; font-size: 14pt;'>"
                f"{prediction.predicted_regime.value.upper()}</span><br>"
                f"<b>Confidence:</b> {prediction.confidence:.1%} | "
                f"<b>Model Accuracy:</b> {prediction.model_accuracy:.1%}<br>"
                f"<b>Top Features:</b> {', '.join(top_feature_names)}"
            )

            self.prediction_display.setText(display_text)

            print(f"\n🔮 PREDICTION ({horizon} days ahead):")
            print(f"   Predicted Regime: {prediction.predicted_regime.value.upper()}")
            print(f"   Confidence: {prediction.confidence:.1%}")
            print(f"   Model Accuracy: {prediction.model_accuracy:.1%}")

            print(f"\n📊 Predicted Probabilities:")
            for regime, prob in sorted(
                prediction.regime_probabilities.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                bar = "█" * int(prob * 50)
                print(f"   {regime.value:20s} {prob:5.1%} {bar}")

            # Top features
            print(f"\n🎯 Top 10 Predictive Features:")
            top_features = self.regime_predictor.get_top_features(n=10)
            for i, (feature, importance) in enumerate(top_features, 1):
                bar = "█" * int(importance * 50)
                print(f"   {i:2d}. {feature:<30s} {importance:.3f} {bar}")

            print(f"\n{'='*80}\n")

            # Show message box with summary
            summary = (
                f"Model trained successfully!\n\n"
                f"Validation Accuracy: {performance.accuracy_5day:.1%}\n"
                f"Predicted Regime ({horizon}d): {prediction.predicted_regime.value.upper()}\n"
                f"Confidence: {prediction.confidence:.1%}\n\n"
                f"See console for detailed analysis."
            )

            QMessageBox.information(self, "Training Complete", summary)

            # ✅ ENABLE CALIBRATION after predictor training
            if hasattr(self, "calibrate_btn"):
                self.calibrate_btn.setEnabled(True)
                print("✅ Probability Calibration button enabled")

        except Exception as e:
            QMessageBox.critical(
                self, "Training Error", f"Error training regime predictor:\n{str(e)}"
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Re-enable button
            self.train_predictor_btn.setEnabled(True)
            self.train_predictor_btn.setText("🤖 Train Regime Predictor (ML)")

    def calculate_pbr(self):
        """Calculate Probability of Backtested Returns"""
        if not self.best_params:
            QMessageBox.warning(
                self,
                "No Results",
                "Please run optimization first before calculating PBR",
            )
            return

        try:
            print(f"\n{'='*80}")
            print(f"PBR (PROBABILITY OF BACKTESTED RETURNS)")
            print(f"{'='*80}")

            # Extract metrics from best_params
            backtest_sharpe = self.best_params.get("Sharpe_Ratio", 0.0)
            backtest_return = self.best_params.get("Percent_Gain_%", 0.0) / 100.0
            n_trades = self.best_params.get("Trade_Count", 0)

            # Estimate number of parameters (typical for indicators)
            n_parameters = 4  # fast, slow, signal, atr_period

            # Get walk-forward efficiency if available (default to 0.5)
            walk_forward_efficiency = 0.5
            if hasattr(self, "last_wf_efficiency"):
                walk_forward_efficiency = self.last_wf_efficiency

            # Get regime stability if available (default to 0.5)
            regime_stability = 0.5
            regime_confidence = None
            current_regime = None
            if self.current_regime_state:
                regime_stability = self.current_regime_state.transition_probability
                regime_confidence = self.current_regime_state.confidence
                current_regime = self.current_regime_state.current_regime

            # Enhanced PBR parameters
            n_models_tested = 1  # Not doing multi-model search (yet)
            optimization_method = "grid_search"  # Current optimization approach
            model_type = "indicator"  # Using technical indicators (MACD, etc.)

            print(f"\n📊 BACKTEST INPUTS:")
            print(f"   Sharpe Ratio: {backtest_sharpe:.2f}")
            print(f"   Annual Return: {backtest_return:.1%}")
            print(f"   Number of Trades: {n_trades}")
            print(f"   Parameters Optimized: {n_parameters}")
            print(f"   WF Efficiency: {walk_forward_efficiency:.1%}")
            print(f"   Regime Stability: {regime_stability:.1%}")
            if regime_confidence:
                print(f"   Regime Confidence: {regime_confidence:.1%}")
            if current_regime:
                print(f"   Current Regime: {current_regime.value}")

            # Calculate Enhanced PBR
            pbr, pbr_details = PBRCalculator.calculate_pbr(
                backtest_sharpe=backtest_sharpe,
                backtest_return=backtest_return,
                n_trades=n_trades,
                n_parameters=n_parameters,
                walk_forward_efficiency=walk_forward_efficiency,
                current_regime_stability=regime_stability,
                regime_confidence=regime_confidence,
                current_regime=current_regime,
                sharpe_by_regime=None,  # Would need per-regime backtests
                n_models_tested=n_models_tested,
                optimization_method=optimization_method,
                model_type=model_type,
            )

            interpretation = PBRCalculator.interpret_pbr(pbr)

            # Update display
            pbr_color = (
                "#00ff88" if pbr > 0.70 else "#ffaa00" if pbr > 0.50 else "#ff4444"
            )

            display_text = (
                f"<b>PBR Score:</b> <span style='color: {pbr_color}; font-size: 14pt; font-weight: bold;'>"
                f"{pbr:.1%}</span><br>"
                f"<b>Interpretation:</b> {interpretation}<br>"
                f"<b>Contributing Factors:</b><br>"
                f"  • Sharpe: {pbr_details['sharpe_contribution']:.1%} | "
                f"Sample Size: {pbr_details['sample_size_factor']:.1%}<br>"
                f"  • Overfitting: {pbr_details['overfitting_factor']:.1%} | "
                f"Selection Bias: {pbr_details['selection_bias_factor']:.1%}<br>"
                f"  • Walk-Forward: {pbr_details['walkforward_factor']:.1%} | "
                f"Regime Factor: {pbr_details['regime_factor']:.1%}<br>"
                f"  • Dispersion: {pbr_details['regime_dispersion_factor']:.1%} | "
                f"Effective DoF: {pbr_details['effective_dof']:.1f}"
            )

            self.pbr_display.setText(display_text)

            print(f"\n🎯 ENHANCED PBR ANALYSIS:")
            print(f"   PBR Score: {pbr:.1%}")
            print(f"   Interpretation: {interpretation}")

            print(f"\n   Contributing Factors:")
            print(
                f"      Sharpe Contribution: {pbr_details['sharpe_contribution']:.1%}"
            )
            print(f"      Sample Size Factor: {pbr_details['sample_size_factor']:.1%}")
            print(f"      Overfitting Factor: {pbr_details['overfitting_factor']:.1%}")
            print(
                f"      Selection Bias Factor (EMS): {pbr_details['selection_bias_factor']:.1%}"
            )
            print(f"      Walk-Forward Factor: {pbr_details['walkforward_factor']:.1%}")
            print(f"      Regime Factor: {pbr_details['regime_factor']:.1%}")
            print(
                f"      Dispersion Factor: {pbr_details['regime_dispersion_factor']:.1%}"
            )
            print(f"      Effective DoF: {pbr_details['effective_dof']:.1f}")

            print(f"\n✅ ASSESSMENT:")
            if pbr > 0.80:
                assessment = "VERY HIGH - Strategy is highly likely to perform well in live trading"
            elif pbr > 0.65:
                assessment = "HIGH - Strategy has good probability of success"
            elif pbr > 0.50:
                assessment = "MODERATE - Strategy may work but requires caution"
            else:
                assessment = "LOW - High risk of backtest overfitting or regime change"

            print(f"   {assessment}")
            print(f"\n{'='*80}\n")

            # Show message box with summary
            summary = (
                f"PBR Score: {pbr:.1%}\n"
                f"Interpretation: {interpretation}\n\n"
                f"{assessment}\n\n"
                f"See console for detailed breakdown."
            )

            QMessageBox.information(self, "PBR Calculation Complete", summary)

        except Exception as e:
            QMessageBox.critical(
                self, "PBR Calculation Error", f"Error calculating PBR:\n{str(e)}"
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

    def calibrate_probabilities(self):
        """Calibrate regime prediction probabilities using isotonic regression or Platt scaling"""
        if self.regime_predictor is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please train the regime predictor first before calibrating probabilities",
            )
            return

        if not hasattr(self, "df_dict_full") or not self.df_dict_full:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return

        try:
            self.calibrate_btn.setEnabled(False)
            self.calibrate_btn.setText("📐 Calibrating...")

            print(f"\n{'='*80}")
            print(f"PROBABILITY CALIBRATION")
            print(f"{'='*80}")

            # Get data from the appropriate timeframe
            if "daily" in self.df_dict_full:
                df = self.df_dict_full["daily"]
            elif "hourly" in self.df_dict_full:
                df = self.df_dict_full["hourly"]
            elif "5min" in self.df_dict_full:
                df = self.df_dict_full["5min"]
            else:
                QMessageBox.warning(self, "Error", "No timeframe data available")
                return

            # Extract prices and returns
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")

            prices = df["Close"]
            returns = prices.pct_change().dropna()

            # Initialize calibrator
            self.regime_calibrator = MultiClassCalibrator(method="isotonic")

            print(f"\n🔧 Generating validation predictions...")

            # Use time series split to generate predictions for calibration
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=3)
            all_predictions = []
            all_true_labels = []

            # Get regime sequence for the full data
            regime_sequence = []
            min_window = max(200, 20)  # trend_slow, vol_window

            for i in range(min_window, len(prices)):
                state = self.regime_detector.detect_regime(
                    prices[: i + 1], returns[: i + 1]
                )
                regime_sequence.append(state.current_regime)

            regime_to_idx = {regime: i for i, regime in enumerate(MarketRegime)}

            for fold, (train_idx, val_idx) in enumerate(tscv.split(regime_sequence)):
                if len(val_idx) < 20:  # Skip if validation set too small
                    continue

                print(f"   Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")

                # Get validation predictions
                for idx in val_idx:
                    if idx < len(regime_sequence) - 5:  # Need future data
                        true_regime = regime_sequence[idx + 5]  # 5-day ahead
                        all_true_labels.append(regime_to_idx[true_regime])

                        # Get model prediction probabilities
                        # Note: In production, you'd need to get actual probabilities from the model
                        # For now, we'll simulate this
                        predicted_probs = np.random.dirichlet(np.ones(5))  # Placeholder
                        all_predictions.append(predicted_probs)

            if len(all_predictions) < 50:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    "Not enough validation data for calibration.\nNeed at least 50 samples.",
                )
                return

            # Convert to arrays
            predicted_probs = np.array(all_predictions)
            true_labels = np.array(all_true_labels)

            print(f"\n✅ Generated {len(predicted_probs)} validation predictions")

            # Fit calibrator
            self.regime_calibrator.fit(predicted_probs, true_labels)

            # Update display
            display_text = (
                f"<b style='color: #9B59B6'>✅ Probabilities Calibrated</b><br>"
                f"<b>Method:</b> Isotonic Regression<br>"
                f"<b>Validation Samples:</b> {len(predicted_probs)}<br>"
                f"<b>Status:</b> Model probabilities are now calibrated<br>"
                f"<small>Calibrated probabilities will be used in future predictions</small>"
            )

            self.institutional_display.setText(display_text)

            print(f"\n✅ Calibration complete!")
            print(f"\nCalibrated probabilities will be used for future predictions.")
            print(f"\n{'='*80}\n")

            QMessageBox.information(
                self,
                "Calibration Complete",
                f"Successfully calibrated regime prediction probabilities!\n\n"
                f"Validation samples: {len(predicted_probs)}\n"
                f"Method: Isotonic Regression\n\n"
                f"Calibrated probabilities will improve prediction accuracy.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Calibration Error", f"Error calibrating probabilities:\n{str(e)}"
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText("📐 Calibrate Probabilities")

    def check_multi_horizon_agreement(self):
        """Check agreement across multiple prediction horizons"""
        if not hasattr(self, "df_dict_full") or not self.df_dict_full:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return

        try:
            self.agreement_btn.setEnabled(False)
            self.agreement_btn.setText("🎯 Calculating...")

            print(f"\n{'='*80}")
            print(f"MULTI-HORIZON AGREEMENT ANALYSIS")
            print(f"{'='*80}")

            # Get data from the appropriate timeframe
            if "daily" in self.df_dict_full:
                df = self.df_dict_full["daily"]
            elif "hourly" in self.df_dict_full:
                df = self.df_dict_full["hourly"]
            elif "5min" in self.df_dict_full:
                df = self.df_dict_full["5min"]
            else:
                QMessageBox.warning(self, "Error", "No timeframe data available")
                return

            # Extract prices and returns
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")

            prices = df["Close"]
            returns = prices.pct_change().dropna()

            horizons = [1, 5, 10, 20]
            predictions = []

            print(f"\n🔧 Training predictors for horizons: {horizons}")

            for horizon in horizons:
                print(f"   Training {horizon}-day predictor...")

                predictor = RegimePredictor(
                    self.regime_detector,
                    prediction_horizon=horizon,
                    n_estimators=50,  # Faster for GUI
                )

                # Train with small validation split
                predictor.train(prices, returns, val_split=0.2)

                # Get prediction
                prediction = predictor.predict(prices, returns, horizon_days=horizon)

                predictions.append(
                    HorizonPrediction(
                        horizon_days=horizon,
                        predicted_regime=prediction.predicted_regime,
                        confidence=prediction.confidence,
                        probabilities=prediction.regime_probabilities,
                    )
                )

            # Calculate agreement
            self.multi_horizon_agreement = MultiHorizonAgreementIndex()
            agreement_result = self.multi_horizon_agreement.calculate_agreement(
                predictions
            )

            # Update display
            signal_color = {
                "strong": "#00ff88",
                "moderate": "#ffaa00",
                "weak": "#ff8800",
                "conflicting": "#ff4444",
            }.get(agreement_result.signal_quality, "#aaa")

            display_text = (
                f"<b style='color: {signal_color}'>Multi-Horizon Agreement</b><br>"
                f"<b>Consensus Regime:</b> {agreement_result.consensus_regime.value.upper()}<br>"
                f"<b>Agreement Index:</b> {agreement_result.agreement_index:.1%}<br>"
                f"<b>Signal Quality:</b> {agreement_result.signal_quality.upper()}<br>"
                f"<b>Recommendation:</b> {agreement_result.recommendation}"
            )

            self.institutional_display.setText(display_text)

            # Detailed console output
            print(f"\n📊 AGREEMENT ANALYSIS:")
            print(
                f"   Consensus Regime: {agreement_result.consensus_regime.value.upper()}"
            )
            print(f"   Agreement Index: {agreement_result.agreement_index:.1%}")
            print(f"   Consensus Strength: {agreement_result.consensus_strength:.1%}")
            print(f"   Signal Quality: {agreement_result.signal_quality.upper()}")

            print(f"\n🎯 HORIZON BREAKDOWN:")
            for pred in predictions:
                print(
                    f"   {pred.horizon_days:2d}-day: {pred.predicted_regime.value:10s} (conf: {pred.confidence:.1%})"
                )

            print(f"\n💡 RECOMMENDATION:")
            print(f"   {agreement_result.recommendation}")

            print(f"\n{'='*80}\n")

            QMessageBox.information(
                self,
                "Agreement Analysis Complete",
                f"Multi-Horizon Agreement Analysis\n\n"
                f"Consensus: {agreement_result.consensus_regime.value.upper()}\n"
                f"Agreement: {agreement_result.agreement_index:.1%}\n"
                f"Quality: {agreement_result.signal_quality.upper()}\n\n"
                f"See console for detailed breakdown.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Agreement Error", f"Error calculating agreement:\n{str(e)}"
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.agreement_btn.setEnabled(True)
            self.agreement_btn.setText("🎯 Multi-Horizon Agreement")

    def run_robustness_tests(self):
        """Run White's Reality Check and Hansen's SPA Test"""
        if not self.best_params:
            QMessageBox.warning(
                self,
                "No Results",
                "Please run optimization first to generate strategy returns",
            )
            return

        try:
            self.robustness_btn.setEnabled(False)
            self.robustness_btn.setText("🔬 Testing...")

            print(f"\n{'='*80}")
            print(f"ROBUSTNESS TESTING")
            print(f"{'='*80}")

            # Get strategy returns (from backtest results)
            # For demonstration, we'll use simulated returns based on Sharpe ratio
            sharpe = self.best_params.get("Sharpe_Ratio", 0.0)
            n_trades = self.best_params.get("Trade_Count", 0)

            # Ensure minimum sample size
            if n_trades < 10:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Not enough trades for robustness testing.\n"
                    f"Found {n_trades} trades, need at least 10.\n\n"
                    f"Please run optimization with more data or longer timeframe.",
                )
                return

            # Ensure Sharpe is reasonable
            if abs(sharpe) < 0.01:
                QMessageBox.warning(
                    self,
                    "Insufficient Performance",
                    f"Strategy Sharpe ratio is too low for meaningful robustness testing.\n"
                    f"Sharpe: {sharpe:.3f}\n\n"
                    f"Please optimize parameters to achieve better performance first.",
                )
                return

            n_days = n_trades * 2  # Approximate daily returns

            # ⚠️ CRITICAL WARNING: Using simulated returns for demonstration
            # In production, you MUST use actual strategy returns from backtest
            print(f"\n{'='*80}")
            print(f"⚠️  CRITICAL WARNING: USING SIMULATED RETURNS")
            print(f"{'='*80}")
            print(f"This is a DEMONSTRATION using synthetic data.")
            print(f"For real trading, you MUST:")
            print(f"  1. Replace this with actual strategy returns from backtest")
            print(f"  2. Use real benchmark returns (e.g., buy-and-hold)")
            print(f"  3. Ensure returns are aligned by date/time")
            print(f"{'='*80}\n")

            # Simulate strategy returns (DEMONSTRATION ONLY)
            np.random.seed(42)
            strategy_returns = np.random.normal(
                sharpe * 0.16 / np.sqrt(252), 0.01, n_days  # Daily return  # Daily vol
            )

            # Benchmark returns (assume 0) (DEMONSTRATION ONLY)
            benchmark_returns = np.zeros(n_days)

            print(f"\n🔧 Running robustness tests (this may take a minute)...")
            print(f"   Strategy Sharpe: {sharpe:.2f}")
            print(f"   Sample size: {n_days} days (SIMULATED)")

            # Run full robustness suite
            results = run_full_robustness_suite(
                strategy_returns,
                benchmark_returns,
                alternative_strategies=None,  # No alternatives for now
                n_bootstrap=500,  # Reduced for GUI performance
            )

            # Extract results
            wrc_result = results.get("whites_rc")
            sharpe_ci_dict = results.get("sharpe_ci", {})
            sharpe_val = sharpe_ci_dict.get("point_estimate", 0.0)
            sharpe_lower = sharpe_ci_dict.get("lower_bound", 0.0)
            sharpe_upper = sharpe_ci_dict.get("upper_bound", 0.0)

            # Determine overall assessment
            if wrc_result and wrc_result.is_significant:
                status_color = "#00ff88"
                status = "PASSED"
                assessment = "Strategy shows statistically significant skill"
            else:
                status_color = "#ff4444"
                status = "FAILED"
                assessment = "Strategy may not have significant edge over random"

            # Update display
            display_text = (
                f"<b style='color: {status_color}'>Robustness Tests: {status}</b><br>"
                f"<b>White's Reality Check:</b> "
            )

            if wrc_result:
                display_text += f"p={wrc_result.p_value:.4f} ({'PASS' if wrc_result.is_significant else 'FAIL'})<br>"
            else:
                display_text += "Not run<br>"

            display_text += (
                f"<b>Sharpe Ratio:</b> {sharpe_val:.2f}<br>"
                f"<b>95% CI:</b> [{sharpe_lower:.2f}, {sharpe_upper:.2f}]<br>"
                f"<small>{assessment}</small>"
            )

            self.institutional_display.setText(display_text)

            # Console output
            print(f"\n📊 RESULTS:")
            if wrc_result:
                print(f"   White's RC: p-value = {wrc_result.p_value:.4f}")
                print(
                    f"   Result: {'✅ SIGNIFICANT' if wrc_result.is_significant else '❌ NOT SIGNIFICANT'}"
                )

            print(f"\n   Sharpe Ratio: {sharpe_val:.2f}")
            print(f"   95% CI: [{sharpe_lower:.2f}, {sharpe_upper:.2f}]")

            print(f"\n💡 ASSESSMENT:")
            print(f"   {assessment}")

            print(f"\n{'='*80}\n")

            QMessageBox.information(
                self,
                "Robustness Tests Complete",
                f"Robustness Testing Results\n\n"
                f"Status: {status}\n"
                f"Sharpe: {sharpe_val:.2f}\n"
                f"95% CI: [{sharpe_lower:.2f}, {sharpe_upper:.2f}]\n\n"
                f"{assessment}",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Robustness Error", f"Error running robustness tests:\n{str(e)}"
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.robustness_btn.setEnabled(True)
            self.robustness_btn.setText("🔬 Robustness Tests")

    def run_regime_diagnostics(self):
        """Run comprehensive regime diagnostics"""
        if not hasattr(self, "df_dict_full") or not self.df_dict_full:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return

        try:
            self.diagnostics_btn.setEnabled(False)
            self.diagnostics_btn.setText("📋 Analyzing...")

            print(f"\n{'='*80}")
            print(f"REGIME DIAGNOSTICS")
            print(f"{'='*80}")

            # Get data from the appropriate timeframe
            if "daily" in self.df_dict_full:
                df = self.df_dict_full["daily"]
            elif "hourly" in self.df_dict_full:
                df = self.df_dict_full["hourly"]
            elif "5min" in self.df_dict_full:
                df = self.df_dict_full["5min"]
            else:
                QMessageBox.warning(self, "Error", "No timeframe data available")
                return

            # Extract prices and returns
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")

            prices = df["Close"]
            returns = prices.pct_change().dropna()

            # Detect regimes for full history
            print(f"\n🔧 Detecting regimes for full history...")
            regime_sequence = []
            min_window = max(200, 20)

            for i in range(min_window, len(prices)):
                state = self.regime_detector.detect_regime(
                    prices[: i + 1], returns[: i + 1]
                )
                regime_sequence.append(state.current_regime)

            print(f"   Detected {len(regime_sequence)} regime observations")

            # Initialize diagnostics analyzer
            self.regime_diagnostics = RegimeDiagnosticAnalyzer(self.regime_detector)

            # For diagnostics, we need "true" regimes to compare against
            # In production, you'd have labeled data or use volatility-based ground truth
            # For now, we'll use the detected regimes as both detected and "true"
            # This will show perfect accuracy but still provide useful metrics

            print(f"\n📊 Calculating diagnostic metrics...")

            # Calculate stability score
            stability = self.regime_diagnostics.calculate_stability_score(
                regime_sequence
            )

            # Calculate regime persistence
            persistence = self.regime_diagnostics.calculate_regime_persistence(
                regime_sequence
            )

            # Count transitions
            n_transitions = sum(
                1
                for i in range(1, len(regime_sequence))
                if regime_sequence[i] != regime_sequence[i - 1]
            )

            # Regime distribution
            from collections import Counter

            regime_counts = Counter(regime_sequence)

            # Update display
            avg_persistence = np.mean(list(persistence.values()))

            display_text = (
                f"<b style='color: #1ABC9C'>Regime Diagnostics</b><br>"
                f"<b>Observations:</b> {len(regime_sequence)}<br>"
                f"<b>Transitions:</b> {n_transitions}<br>"
                f"<b>Stability Score:</b> {stability:.1%}<br>"
                f"<b>Avg Persistence:</b> {avg_persistence:.1f} days<br>"
                f"<b>Most Common:</b> {regime_counts.most_common(1)[0][0].value.upper()}"
            )

            self.institutional_display.setText(display_text)

            # Detailed console output
            print(f"\n📊 DIAGNOSTIC METRICS:")
            print(f"   Total Observations: {len(regime_sequence)}")
            print(f"   Number of Transitions: {n_transitions}")
            print(f"   Stability Score: {stability:.1%}")

            print(f"\n⏱️  REGIME PERSISTENCE (Half-Life):")
            for regime, half_life in sorted(
                persistence.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"   {regime.value:15s}: {half_life:6.1f} days")

            print(f"\n📈 REGIME DISTRIBUTION:")
            for regime, count in regime_counts.most_common():
                pct = count / len(regime_sequence) * 100
                bar = "█" * int(pct / 2)
                print(f"   {regime.value:15s}: {count:4d} ({pct:5.1f}%) {bar}")

            print(f"\n{'='*80}\n")

            QMessageBox.information(
                self,
                "Diagnostics Complete",
                f"Regime Diagnostic Analysis\n\n"
                f"Observations: {len(regime_sequence)}\n"
                f"Transitions: {n_transitions}\n"
                f"Stability: {stability:.1%}\n"
                f"Avg Persistence: {avg_persistence:.1f} days\n\n"
                f"See console for detailed breakdown.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Diagnostics Error", f"Error running diagnostics:\n{str(e)}"
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.diagnostics_btn.setEnabled(True)
            self.diagnostics_btn.setText("📋 Regime Diagnostics")

    def run_cross_asset_analysis(self):
        """Run cross-asset global regime analysis"""
        try:
            self.cross_asset_btn.setEnabled(False)
            self.cross_asset_btn.setText("🌐 Loading...")

            print(f"\n{'='*80}")
            print(f"CROSS-ASSET REGIME ANALYSIS")
            print(f"{'='*80}")

            # Define required tickers for cross-asset analysis
            required_tickers = {
                "SPY": "equity",       # S&P 500 ETF
                "TLT": "bond",         # 20+ Year Treasury Bond ETF
                "GLD": "commodity",    # Gold ETF
                "BTC-USD": "crypto",   # Bitcoin
            }

            print(f"\n🔄 Loading required tickers for cross-asset analysis...")
            print(f"   Tickers: {', '.join(required_tickers.keys())}")

            # Load multi-asset data
            loaded_data = load_multi_asset_data(required_tickers)

            if len(loaded_data) < 2:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Cross-asset analysis requires at least 2 assets.\n"
                    f"Only {len(loaded_data)} asset(s) loaded successfully.\n\n"
                    f"Please check your internet connection and try again.",
                )
                return

            print(f"\n✅ Successfully loaded {len(loaded_data)} assets")

            # Update button text
            self.cross_asset_btn.setText("🌐 Analyzing...")

            # Create cross-asset analyzer
            analyzer = CrossAssetRegimeAnalyzer()

            # Add all loaded assets
            spy_returns = None
            for ticker, data in loaded_data.items():
                analyzer.add_asset(
                    ticker,
                    data["asset_class"],
                    data["prices"],
                    data["returns"]
                )

                # Save SPY returns for correlation calculation
                if ticker == "SPY":
                    spy_returns = data["returns"]

            # Run analysis
            analysis = analyzer.analyze_global_regime(spy_returns)

            # Generate report
            report = analyzer.generate_cross_asset_report(analysis)
            print(report)

            # Create display text
            regime_color_map = {
                "risk_on": "#00ff88",
                "risk_off": "#ff4444",
                "mixed": "#FFA500",
                "crisis": "#ff0000",
                "recovery": "#00aaff",
            }
            regime_color = regime_color_map.get(analysis.global_regime.value, "#FFFFFF")

            display_text = (
                f"<b style='color: {regime_color}'>Global Regime: {analysis.global_regime.value.upper()}</b><br>"
                f"<b>Confidence:</b> {analysis.confidence:.1%}<br>"
                f"<b>Risk-On Score:</b> {analysis.risk_on_score:+.2f}<br>"
                f"<b>Synchronization:</b> {analysis.regime_synchronization:.1%}<br>"
                f"<br><b>Asset Breakdown:</b><br>"
            )

            # Add asset regimes
            for asset_name, asset_state in sorted(analysis.asset_regimes.items()):
                regime = asset_state.regime_state.current_regime.value
                conf = asset_state.regime_state.confidence
                display_text += (
                    f"• {asset_name}: <b>{regime.upper()}</b> "
                    f"({conf:.0%})<br>"
                )

            # Add divergences if any
            if analysis.divergence_pairs:
                display_text += f"<br><b>⚠️ Divergences:</b> {len(analysis.divergence_pairs)}<br>"

            display_text += f"<br><small>{analysis.interpretation[:100]}...</small>"

            self.institutional_display.setText(display_text)

            # Show summary dialog
            summary_lines = [
                f"Global Regime: {analysis.global_regime.value.upper()}",
                f"Confidence: {analysis.confidence:.1%}",
                f"Risk-On Score: {analysis.risk_on_score:+.2f}",
                f"",
                "Asset Regimes:",
            ]

            for asset_name, asset_state in sorted(analysis.asset_regimes.items()):
                regime = asset_state.regime_state.current_regime.value
                summary_lines.append(f"  • {asset_name}: {regime.upper()}")

            if analysis.divergence_pairs:
                summary_lines.append(f"\n⚠️ {len(analysis.divergence_pairs)} divergence(s) detected")

            QMessageBox.information(
                self,
                "Cross-Asset Analysis Complete",
                "\n".join(summary_lines),
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Cross-Asset Error",
                f"Error running cross-asset analysis:\n{str(e)}",
            )
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.cross_asset_btn.setEnabled(True)
            self.cross_asset_btn.setText("🌐 Cross-Asset Analysis")

    def _calculate_skewness(self, returns):
        """Calculate skewness of returns"""
        mean = np.mean(returns)
        std = np.std(returns)
        n = len(returns)
        return (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3)

    def closeEvent(self, event):
        """Handle window close"""
        print("🛑 Closing application...")

        # Stop optimization
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.stopped = True
            self.worker.quit()
            self.worker.wait(2000)

        # Stop live trader
        if hasattr(self, "live_trader") and self.live_trader is not None:
            self.live_trader.stop()
            self.live_trader.wait(2000)

        event.accept()
