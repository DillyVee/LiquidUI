"""
Main Window GUI
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QProgressBar, QMessageBox, QCheckBox, QComboBox, QGroupBox
)
from PyQt6.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtGui import QFont

from optimization.walk_forward import WalkForwardAnalyzer, WalkForwardResults

from config.settings import (
    OptimizationConfig, RiskConfig, AlpacaConfig, 
    IndicatorRanges, Paths, TransactionCosts
)
from gui.styles import (
    MAIN_STYLESHEET, LIVE_TRADING_BUTTON_ACTIVE, 
    LIVE_TRADING_BUTTON_STOPPED, COLOR_SUCCESS, COLOR_WARNING
)
from data import DataLoader
from optimization import MultiTimeframeOptimizer, PerformanceMetrics, MonteCarloSimulator
from trading import AlpacaLiveTrader, ALPACA_AVAILABLE


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
        print(f"   Keys: {list(self.df_dict_full.keys()) if self.df_dict_full else 'EMPTY!'}")
        
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
            if 'Datetime' in df.columns:
                print(f"      Datetime column: EXISTS ✅")
                dt = df['Datetime']
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
        selected_tfs = [
            tf for tf, cb in self.tf_checkboxes.items() 
            if cb.isChecked()
        ]
        print(f"   Checked boxes: {selected_tfs}")
        
        available_selected = [
            tf for tf in selected_tfs 
            if tf in self.df_dict_full
        ]
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
        
        tf_order = {'5min': 0, 'hourly': 1, 'daily': 2}
        finest_tf = sorted(available_selected, key=lambda x: tf_order.get(x, 99))[0]
        
        print(f"   Finest timeframe: {finest_tf}")
        
        df_test = self.df_dict_full[finest_tf].copy()
        print(f"   df shape: {df_test.shape}")
        
        # Check if Datetime exists
        if 'Datetime' not in df_test.columns:
            print(f"   ⚠️  Adding Datetime from index...")
            df_test['Datetime'] = pd.to_datetime(df_test.index)
        
        min_dt = df_test['Datetime'].min()
        max_dt = df_test['Datetime'].max()
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
            print(f"   ❌ Window exceeds data! (test_end={test_end.date()}, max={max_dt.date()})")
        else:
            print(f"   ✅ Window fits in data")
        
        # Count bars in window
        train_mask = (df_test['Datetime'] >= train_start) & (df_test['Datetime'] < train_end)
        test_mask = (df_test['Datetime'] >= test_start) & (df_test['Datetime'] < test_end)
        
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
        
        # Live trading controls
        self._add_live_trading_controls(main_layout)
        
        # Risk management
        self._add_risk_management_controls(main_layout)
        
        # Transaction costs
        self._add_transaction_cost_controls(main_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Chart
        self.figure = plt.figure(facecolor="#121212")
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Best parameters display
        self.best_params_label = QLabel("Best Parameters: N/A")
        self.best_params_label.setWordWrap(True)
        main_layout.addWidget(self.best_params_label)

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def _add_data_source_controls(self, layout: QVBoxLayout):
        """Add data source input controls"""
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Data Source:"))
        
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker (e.g., AAPL, SPY, BTC-USD)")
        self.load_yf_btn = QPushButton("Load from Yahoo Finance")
        self.load_yf_btn.clicked.connect(self.load_yfinance)
        
        source_layout.addWidget(self.ticker_input)
        source_layout.addWidget(self.load_yf_btn)
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
        self.tf_checkboxes['daily'] = QCheckBox("Daily")
        self.tf_checkboxes['daily'].setChecked(False)
        self.tf_checkboxes['daily'].stateChanged.connect(self.on_timeframe_changed)
        
        self.tf_checkboxes['hourly'] = QCheckBox("Hourly")
        self.tf_checkboxes['hourly'].setChecked(True)
        self.tf_checkboxes['hourly'].stateChanged.connect(self.on_timeframe_changed)
        
        self.tf_checkboxes['5min'] = QCheckBox("5-Minute")
        self.tf_checkboxes['5min'].setChecked(False)
        self.tf_checkboxes['5min'].stateChanged.connect(self.on_timeframe_changed)
        
        for cb in self.tf_checkboxes.values():
            tf_layout.addWidget(cb)
        
        self.equity_curve_check = QCheckBox("Optimize on Equity Curve")
        self.equity_curve_check.setToolTip("Optimize trades on equity curve retracement zones")
        tf_layout.addWidget(self.equity_curve_check)
        
        self.show_base_curve_check = QCheckBox("Show Base Strategy")
        self.show_base_curve_check.setChecked(True)
        tf_layout.addWidget(self.show_base_curve_check)
        
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
            tf for tf, cb in self.tf_checkboxes.items() 
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
            (0, self.on_max.value() + self.off_max.value())
        )

        optimize_equity_curve = self.equity_curve_check.isChecked()

        # Create optimizer (no objective_type parameter needed)
        self.worker = MultiTimeframeOptimizer(
            self.df_dict, 
            self.trials_spin.value(), 
            time_cycle_ranges, 
            mn1_range, mn2_range, entry_range, exit_range,
            ticker=self.current_ticker,
            timeframes=selected_tfs, 
            optimize_equity_curve=optimize_equity_curve,
            batch_size=self.batch_spin.value(),
            transaction_costs=self.transaction_costs
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
            ("MN2", self.mn2_min, self.mn2_max)
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
            ("Exit", self.exit_min, self.exit_max)
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
            ("Off Bars", self.off_min, self.off_max)
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
        self.monte_carlo_btn.setStyleSheet("""
            QPushButton { 
                background-color: #663399; 
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #7a3db8; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """)

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
        self.walk_forward_btn.setStyleSheet("""
            QPushButton { 
                background-color: #009688; 
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #00bfa5; }
            QPushButton:disabled { background-color: #0a0a0a; color: #555; }
        """)

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

    def run_walk_forward(self):
        """Run walk-forward analysis with smart defaults"""
        
        if not self.df_dict_full:
            QMessageBox.warning(
                self, "No Data",
                "Please load data first before running walk-forward analysis"
            )
            return

        # Get selected timeframes
        selected_tfs = [
            tf for tf, cb in self.tf_checkboxes.items() 
            if cb.isChecked() and tf in self.df_dict_full
        ]

        if not selected_tfs:
            QMessageBox.warning(self, "Error", "Select at least one timeframe")
            return

        # Determine finest timeframe
        tf_order = {'5min': 0, 'hourly': 1, 'daily': 2}
        finest_tf = sorted(selected_tfs, key=lambda x: tf_order.get(x, 99))[0]
        
        # Get data info
        df_check = self.df_dict_full[finest_tf]
        dt_check = df_check['Datetime'] if 'Datetime' in df_check.columns else pd.to_datetime(df_check.index)
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
        
        max_possible_windows = max(0, (available_days - recommended_train) // recommended_test)
        
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
                self, "Optimize Settings?",
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
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
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
                self, "Auto-Configuration Required",
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
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
        
        else:
            # User settings are fine, just confirm
            reply = QMessageBox.question(
                self, "Confirm Walk-Forward",
                f"Walk-forward analysis configuration:\n\n"
                f"• Timeframe: {finest_tf}\n"
                f"• Data: {available_days} days\n"
                f"• Train: {user_train} days\n"
                f"• Test: {user_test} days\n"
                f"• Trials: {user_trials} per window\n"
                f"• Windows: {user_windows}\n"
                f"• Estimated time: {(user_windows * user_trials) // 100}-{(user_windows * user_trials) // 50} minutes\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Set final parameters
        if use_recommended:
            final_train = recommended_train
            final_test = recommended_test
            final_trials = recommended_trials
            print(f"✅ Using RECOMMENDED settings: train={final_train}, test={final_test}, trials={final_trials}")
        else:
            final_train = user_train
            final_test = user_test
            final_trials = user_trials
            print(f"✅ Using YOUR settings: train={final_train}, test={final_test}, trials={final_trials}")
        
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
                (0, self.on_max.value() + self.off_max.value())
            )

            objective_map = {
                "Percent Gain": "percent_gain",
                "Sortino Ratio": "sortino",
                "Min Drawdown": "drawdown",
                "Profit Factor": "profit_factor"
            }
            objective_type = objective_map[self.objective_combo.currentText()]

            optimizer_kwargs = {
                'n_trials': final_trials,
                'time_cycle_ranges': time_cycle_ranges,  # ✅ Cycle preserved
                'mn1_range': mn1_range,
                'mn2_range': mn2_range,
                'entry_range': entry_range,
                'exit_range': exit_range,
                'ticker': self.current_ticker,
                'timeframes': selected_tfs,
                'optimize_equity_curve': self.equity_curve_check.isChecked(),
                'batch_size': self.batch_spin.value(),
                'transaction_costs': self.transaction_costs
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
                **optimizer_kwargs
            )

            # Generate report
            report = WalkForwardAnalyzer.generate_walk_forward_report(results)
            print(report)

            # Create plot
            fig = WalkForwardAnalyzer.plot_walk_forward_results(
                results,
                ticker=self.current_ticker
            )

            # Save plot
            filename = f"{self.current_ticker}_walk_forward.png"
            fig.savefig(filename, dpi=150, facecolor='#121212', edgecolor='none')
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

            QMessageBox(icon, title, summary, QMessageBox.StandardButton.Ok, self).exec()

            # Save results to CSV
            results_df = pd.DataFrame({
                'Window': range(1, len(results.in_sample_returns) + 1),
                'IS_Return_%': results.in_sample_returns,
                'OOS_Return_%': results.out_of_sample_returns,
                'Train_Start': [d[0] for d in results.window_dates],
                'Train_End': [d[1] for d in results.window_dates],
                'Test_Start': [d[2] for d in results.window_dates],
                'Test_End': [d[3] for d in results.window_dates]
            })
        
            csv_filename = f"{self.current_ticker}_walk_forward_results.csv"
            results_df.to_csv(csv_filename, index=False)
            print(f"✅ Saved results to {csv_filename}")

        except Exception as e:
            QMessageBox.critical(
                self, "Walk-Forward Error",
                f"Failed to run walk-forward analysis:\n{str(e)}"
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
        if hasattr(self, 'live_trader') and self.live_trader and self.live_trader.running:
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
            tf for tf, cb in self.tf_checkboxes.items() 
            if cb.isChecked() and tf in self.df_dict_full
        ]
        
        if not selected_tfs:
            return
        
        # Determine finest timeframe
        tf_order = {'5min': 0, 'hourly': 1, 'daily': 2}
        sorted_tfs = sorted(selected_tfs, key=lambda x: tf_order.get(x, 99))
        finest_tf = sorted_tfs[0]
        
        # Apply data limits
        if finest_tf == '5min':
            limit_days = OptimizationConfig.FIVEMIN_MAX_DAYS
        elif finest_tf == 'hourly':
            limit_days = OptimizationConfig.HOURLY_MAX_DAYS
        else:
            limit_days = None
        
        self.df_dict = DataLoader.filter_timeframe_data(self.df_dict_full, limit_days)
        
        # Update displays
        if finest_tf in self.df_dict:
            df_display = self.df_dict[finest_tf]
            date_min = df_display['Datetime'].min()
            date_max = df_display['Datetime'].max()
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
            linewidth=2
        )
        
        ax.set_facecolor("#121212")
        ax.tick_params(colors="white")
        ax.set_title(
            f"{self.current_ticker} - {timeframe.capitalize()} View ({len(close_arr)} bars)", 
            color="white", fontsize=14
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
            if '5min' in df_dict:
                tf_counts += f", {len(df_dict['5min'])}x5m"
            self.ticker_input.setText(f"{symbol} ({tf_counts} max)")
            
            # ✅ ENABLE WALK-FORWARD BUTTON after data load
            if hasattr(self, 'walk_forward_btn'):
                self.walk_forward_btn.setEnabled(True)
                print("✅ Walk-Forward Analysis button enabled")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
        finally:
            self.load_yf_btn.setEnabled(True)

    def start_optimization(self):
        """Start the optimization process"""
        if not self.df_dict:
            QMessageBox.warning(self, "Error", "Please load data first")
            return
    
        selected_tfs = [
            tf for tf, cb in self.tf_checkboxes.items() 
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
            (0, self.on_max.value() + self.off_max.value())
        )

        optimize_equity_curve = self.equity_curve_check.isChecked()

        # Create optimizer with PSR composite (no objective_type parameter)
        self.worker = MultiTimeframeOptimizer(
            self.df_dict, 
            self.trials_spin.value(), 
            time_cycle_ranges, 
            mn1_range, mn2_range, entry_range, exit_range,
            ticker=self.current_ticker,
            timeframes=selected_tfs, 
            optimize_equity_curve=optimize_equity_curve,
            batch_size=self.batch_spin.value(),
            transaction_costs=self.transaction_costs
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
        """Update best result display with PSR"""
        self.best_params = best_params
        
        # PSR score (prominent)
        if 'PSR' in best_params:
            psr = best_params['PSR']
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
        if 'Sharpe_Ratio' in best_params:
            sharpe = best_params['Sharpe_Ratio']
            sharpe_color = "#00ff88" if sharpe > 1.0 else "#ffaa00" if sharpe > 0.5 else "#ff4444"
            self.sharpe_label.setText(f"Sharpe: {sharpe:.2f}")
            self.sharpe_label.setStyleSheet(f"color: {sharpe_color}; font-size: 10pt;")
        
        # Traditional metrics
        metrics_text = f"Return: {best_params['Percent_Gain_%']:.2f}%"
        if 'Sortino_Ratio' in best_params:
            metrics_text += f" | Sortino: {best_params['Sortino_Ratio']:.2f}"
        if 'Max_Drawdown_%' in best_params:
            metrics_text += f" | DD: {best_params['Max_Drawdown_%']:.2f}%"
        if 'Profit_Factor' in best_params:
            metrics_text += f" | PF: {best_params['Profit_Factor']:.2f}"
        if 'Trade_Count' in best_params:
            metrics_text += f" | Trades: {best_params['Trade_Count']}"
        
        self.best_label.setText(metrics_text)
        
        # Build parameter text
        param_lines = []
        for tf in ['daily', 'hourly', '5min']:
            if f'MN1_{tf}' in best_params:
                param_lines.append(
                    f"{tf.upper()}: RSI({best_params[f'MN1_{tf}']},{best_params[f'MN2_{tf}']}) "
                    f"Entry<{best_params[f'Entry_{tf}']:.1f} "
                    f"Exit>{best_params[f'Exit_{tf}']:.1f}"
                )
        
        self.best_params_label.setText(" | ".join(param_lines))

    def _add_psr_tooltips(self):
        """Add helpful tooltips explaining PSR metrics"""
        # Only add tooltips for labels that exist in PSR-only mode
        if hasattr(self, 'psr_label'):
            self.psr_label.setToolTip(
                "Probabilistic Sharpe Ratio\n"
                "Probability that true Sharpe > 0\n\n"
                ">95%: Very confident\n"
                ">75%: Good confidence\n"
                "<50%: Likely false positive"
            )
        
        if hasattr(self, 'sharpe_label'):
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
    
        # Risk management settings
        self.position_size_pct = RiskConfig.DEFAULT_POSITION_SIZE
        self.max_positions = RiskConfig.DEFAULT_MAX_POSITIONS
    
        # Transaction costs
        self.transaction_costs = TransactionCosts()

        # Trade log for Monte Carlo
        self.last_trade_log = []

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
        comp_score = self.best_params.get('Composite_Score', 0)
        psr = self.best_params.get('PSR', 0)
        pbo = self.best_params.get('PBO', 1)
        
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
        if hasattr(self, 'monte_carlo_btn'):
            self.monte_carlo_btn.setEnabled(True)
        if hasattr(self, 'walk_forward_btn'):
            self.walk_forward_btn.setEnabled(True)
        
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
        """Plot optimization results on chart"""
        # Get optimized timeframes
        optimized_tfs = []
        best_dict = best.to_dict() if isinstance(best, pd.Series) else best
        for tf in ['5min', 'hourly', 'daily']:
            if f'MN1_{tf}' in best_dict:
                optimized_tfs.append(tf)
        
        if not optimized_tfs:
            return
        
        tf_order = {'5min': 0, 'hourly': 1, 'daily': 2}
        finest_tf = sorted(optimized_tfs, key=lambda x: tf_order.get(x, 99))[0]
        
        finest_df = self.df_dict[finest_tf]
        close_finest = finest_df["Close"].to_numpy(dtype=float)
        buyhold_finest = (close_finest / close_finest[0]) * 1000
        
        # Create plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Buy & Hold
        ax.plot(
            buyhold_finest, color="#888888", 
            label=f"Buy & Hold ({self.buyhold_pct:+.1f}%)", 
            linewidth=2, alpha=0.7
        )
        
        # Strategy curve - STORE TRADE LOG
        final_eq_curve, final_trades, trade_log = self.worker.simulate_multi_tf(
            best_dict, return_trades=True
        )
        
        # Store trade log for Monte Carlo
        self.last_trade_log = trade_log
        
        # ✅ FIX: Check if monte_carlo_btn exists before enabling
        if hasattr(self, 'monte_carlo_btn'):
            if len(trade_log) > 10:  # Need at least 10 trades for meaningful MC
                self.monte_carlo_btn.setEnabled(True)
                print(f"✓ {len(trade_log)} trades available for Monte Carlo simulation")
            else:
                self.monte_carlo_btn.setEnabled(False)
                print(f"⚠ Only {len(trade_log)} trades - need >10 for Monte Carlo")
        
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
        
        ax.set_title(f"{self.current_ticker} - {metrics_text}", color="white", fontsize=11)
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
                self, "Error", 
                "Alpaca not installed!\nInstall: pip install alpaca-py"
            )
            return
        
        if self.best_params is None:
            QMessageBox.warning(
                self, "Error", 
                "Run optimization first to get parameters"
            )
            return
        
        selected_tfs = [
            tf for tf, cb in self.tf_checkboxes.items() 
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
            position_size_pct=self.position_size_pct  # Fixed: added comma above
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
        action = trade_info['action']
        symbol = trade_info['symbol']
        shares = trade_info['shares']
        price = trade_info['price']
        time = trade_info['time']
        
        msg = f"{action}: {shares} of {symbol} @ ${price:.2f}\nTime: {time}"
        QMessageBox.information(self, f"Trade Executed - {action}", msg)
    
    def on_trading_error(self, error_msg: str):
        """Handle trading error"""
        self.trading_status_label.setText(f"Error: {error_msg}")
        self.trading_status_label.setStyleSheet("color: #ff4444; font-size: 10pt;")
        QMessageBox.critical(self, "Trading Error", error_msg)

    def test_rsi_sensitivity():
        """Test how different MN1/MN2 values affect trade count"""
        import yfinance as yf
        import numpy as np
        
        # Get AAPL hourly data
        df = yf.download("AAPL", period="2y", interval="1h")
        close = df['Close'].values
        
        print("\nRSI Sensitivity Test:")
        print("="*50)
        
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
            
            avg_gain = np.convolve(gain, np.ones(mn1)/mn1, mode='same')
            avg_loss = np.convolve(loss, np.ones(mn1)/mn1, mode='same')
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - 100 / (1 + rs)
            
            # Smooth
            rsi_smooth = np.convolve(rsi, np.ones(mn2)/mn2, mode='same')
            
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
        """Run Monte Carlo simulation on trade results"""
        if not self.last_trade_log or len(self.last_trade_log) < 10:
            QMessageBox.warning(
                self, "Insufficient Data",
                f"Need at least 10 trades for Monte Carlo.\n"
                f"Currently have: {len(self.last_trade_log)} trades"
            )
            return
        
        n_simulations = self.mc_simulations_spin.value()
        
        # Disable button during simulation
        self.monte_carlo_btn.setEnabled(False)
        self.monte_carlo_btn.setText("Running Monte Carlo...")
        
        try:
            print(f"\n{'='*60}")
            print(f"MONTE CARLO SIMULATION")
            print(f"{'='*60}")
            print(f"Trades: {len(self.last_trade_log)}")
            print(f"Simulations: {n_simulations}")
            print(f"Method: Trade Randomization")
            
            # Run Monte Carlo simulation
            results = MonteCarloSimulator.simulate_trade_randomization(
                trades=self.last_trade_log,
                n_simulations=n_simulations,
                initial_equity=1000.0
            )
            
            # Generate and display report
            report = MonteCarloSimulator.generate_monte_carlo_report(results, initial_equity=1000.0)
            print(report)
            
            # Create plot
            fig = MonteCarloSimulator.plot_monte_carlo_results(
                results,
                title=f"Monte Carlo Simulation - {self.current_ticker} ({n_simulations} runs)",
                max_paths=100
            )
            
            # Save plot
            filename = f"{self.current_ticker}_monte_carlo.png"
            fig.savefig(filename, dpi=150, facecolor='#121212', edgecolor='none')
            print(f"✓ Saved Monte Carlo plot to {filename}")
            
            # Show results in message box
            original_return = (results.original_equity / 1000.0 - 1) * 100
            mean_return = (results.mean_equity / 1000.0 - 1) * 100
            
            message = (
                f"Monte Carlo Simulation Complete!\n\n"
                f"📊 Original Return: {original_return:+.1f}%\n"
                f"📈 Mean Return: {mean_return:+.1f}%\n"
                f"📉 Median Return: {(results.median_equity/1000-1)*100:+.1f}%\n\n"
                f"🎯 95% Confidence Interval:\n"
                f"   ${results.percentile_5:,.0f} to ${results.percentile_95:,.0f}\n\n"
                f"⚠️  Worst Case: ${results.min_equity:,.0f}\n"
                f"🎉 Best Case: ${results.max_equity:,.0f}\n\n"
                f"✅ Probability of Profit: {results.probability_profit*100:.1f}%\n\n"
            )
            
            if results.probability_profit < 0.5:
                message += "❌ WARNING: Strategy more likely to lose!\n"
                icon = QMessageBox.Icon.Warning
            elif results.probability_profit < 0.7:
                message += "⚠️  Moderate confidence in strategy\n"
                icon = QMessageBox.Icon.Information
            else:
                message += "✅ Strategy appears robust!\n"
                icon = QMessageBox.Icon.Information
            
            message += f"\nPlot saved to: {filename}\nSee console for full report."
            
            # Display plot
            plt.show()
            
            # Show message box
            QMessageBox(icon, "Monte Carlo Results", message, QMessageBox.StandardButton.Ok, self).exec()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Monte Carlo Error",
                f"Failed to run Monte Carlo simulation:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
        
        finally:
            # Re-enable button
            self.monte_carlo_btn.setEnabled(True)
            self.monte_carlo_btn.setText("🎲 Run Monte Carlo Simulation")

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
