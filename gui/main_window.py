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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setStyleSheet(MAIN_STYLESHEET)
        
        main_layout = QVBoxLayout()
        
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
        """Add optimization parameter controls"""
        controls_layout = QHBoxLayout()
        
        # Trials
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
        
        # Objective
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "Percent Gain",
            "Sortino Ratio", 
            "Min Drawdown",
            "Profit Factor"
        ])
        controls_layout.addWidget(QLabel("Objective:"))
        controls_layout.addWidget(self.objective_combo)

        # Results display
        self.best_label = QLabel("Best Result: N/A")
        self.buyhold_label = QLabel("Buy & Hold: N/A")
        controls_layout.addWidget(self.best_label)
        controls_layout.addWidget(self.buyhold_label)
        
        layout.addLayout(controls_layout)

    def _add_parameter_ranges(self, layout: QVBoxLayout):
        """Add parameter range controls"""
        # MN1 and MN2 ranges
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
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Multi-Timeframe Optimization")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        
        self.start_btn.clicked.connect(self.start_optimization)
        self.stop_btn.clicked.connect(self.stop_optimization)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # Monte Carlo button (separate row)
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
        
        layout.addLayout(mc_layout)

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

        objective_map = {
            "Percent Gain": "percent_gain",
            "Sortino Ratio": "sortino",
            "Min Drawdown": "drawdown",
            "Profit Factor": "profit_factor"
        }
        objective_type = objective_map[self.objective_combo.currentText()]
        optimize_equity_curve = self.equity_curve_check.isChecked()

        # Create optimizer
        self.worker = MultiTimeframeOptimizer(
            self.df_dict, 
            self.trials_spin.value(), 
            time_cycle_ranges, 
            mn1_range, mn2_range, entry_range, exit_range,
            ticker=self.current_ticker, 
            objective_type=objective_type,
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
        """Update best result display"""
        self.best_params = best_params
        
        # Build metrics text
        metrics_text = f"Gain: {best_params['Percent_Gain_%']:.2f}%"
        if 'Sortino_Ratio' in best_params:
            metrics_text += f" | Sortino: {best_params['Sortino_Ratio']:.3f}"
        if 'Max_Drawdown_%' in best_params:
            metrics_text += f" | DD: {best_params['Max_Drawdown_%']:.2f}%"
        if 'Profit_Factor' in best_params:
            metrics_text += f" | PF: {best_params['Profit_Factor']:.2f}"
        if 'Trade_Count' in best_params:
            metrics_text += f" | Trades: {best_params['Trade_Count']}"
        
        self.best_label.setText(f"Best: {metrics_text}")
        
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

    def show_results(self, df_results: pd.DataFrame):
        """Display optimization results"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if df_results.empty:
            QMessageBox.information(self, "Complete", "No valid results")
            self.phase_label.setText("No results")
            return
            
        self.progress_bar.setValue(100)
        self.phase_label.setText("✓ Optimization Complete!")
        self.phase_info_label.setText("✓ All phases completed successfully")
        
        # Enable live trading
        if ALPACA_AVAILABLE:
            self.live_trading_btn.setEnabled(True)
            self.trading_status_label.setText("Ready")
            self.trading_status_label.setStyleSheet(
                f"color: {COLOR_SUCCESS}; font-size: 10pt; font-weight: bold;"
            )

        best = df_results.iloc[0]
        self.update_best_label(best.to_dict() if isinstance(best, pd.Series) else best)

        # Plot results and GET TRADE LOG
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
        
        # Enable Monte Carlo button if we have trades
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
