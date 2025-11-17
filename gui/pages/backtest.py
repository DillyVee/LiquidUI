"""
Backtest Page - Run Backtests with Real-time Visualization
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit, QGroupBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class BacktestWorker(QThread):
    """Worker thread for running backtests"""
    
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
    
    def run(self):
        """Run backtest in background"""
        try:
            self.log.emit("Initializing backtest engine...")
            self.progress.emit(10)
            
            self.log.emit(f"Loading data for {self.params['symbol']}...")
            self.progress.emit(30)
            
            # Simulate backtest (replace with actual backtest engine)
            import time
            for i in range(30, 90, 10):
                time.sleep(0.3)
                self.progress.emit(i)
                self.log.emit(f"Processing {i}% complete...")
            
            # Generate dummy results
            np.random.seed(42)
            n_days = 252
            returns = np.random.normal(0.001, 0.02, n_days)
            equity_curve = 100000 * (1 + returns).cumprod()
            
            results = {
                'equity_curve': equity_curve,
                'returns': returns,
                'total_return': (equity_curve[-1] / equity_curve[0] - 1) * 100,
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
                'max_drawdown': np.min((equity_curve / np.maximum.accumulate(equity_curve) - 1)) * 100,
                'win_rate': 0.625,
                'total_trades': 145,
                'avg_win': 234.50,
                'avg_loss': -156.20,
            }
            
            self.progress.emit(100)
            self.log.emit("✓ Backtest complete!")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class BacktestPage(QWidget):
    """Backtest runner page"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backtest_worker = None
        self.results = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the backtest UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("🔬 Backtest Runner")
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #E0E0E0;")
        layout.addWidget(header)
        
        # Configuration section
        config_group = QGroupBox("Backtest Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        config_layout = QHBoxLayout(config_group)
        
        # Left column - basic settings
        left_col = QVBoxLayout()
        
        # Symbol
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["SPY", "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"])
        symbol_layout.addWidget(self.symbol_combo)
        left_col.addLayout(symbol_layout)
        
        # Strategy
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["RSI Mean Reversion", "MACD Crossover", "Bollinger Breakout", "Moving Average"])
        strategy_layout.addWidget(self.strategy_combo)
        left_col.addLayout(strategy_layout)
        
        # Initial capital
        capital_layout = QHBoxLayout()
        capital_layout.addWidget(QLabel("Initial Capital:"))
        self.capital_input = QDoubleSpinBox()
        self.capital_input.setRange(1000, 10000000)
        self.capital_input.setValue(100000)
        self.capital_input.setPrefix("$")
        capital_layout.addWidget(self.capital_input)
        left_col.addLayout(capital_layout)
        
        config_layout.addLayout(left_col)
        
        # Right column - advanced settings
        right_col = QVBoxLayout()
        
        # Commission
        comm_layout = QHBoxLayout()
        comm_layout.addWidget(QLabel("Commission:"))
        self.comm_input = QDoubleSpinBox()
        self.comm_input.setRange(0, 1)
        self.comm_input.setValue(0.001)
        self.comm_input.setSingleStep(0.0001)
        self.comm_input.setSuffix("%")
        comm_layout.addWidget(self.comm_input)
        right_col.addLayout(comm_layout)
        
        # Slippage
        slip_layout = QHBoxLayout()
        slip_layout.addWidget(QLabel("Slippage:"))
        self.slip_input = QDoubleSpinBox()
        self.slip_input.setRange(0, 1)
        self.slip_input.setValue(0.0002)
        self.slip_input.setSingleStep(0.0001)
        self.slip_input.setSuffix("%")
        slip_layout.addWidget(self.slip_input)
        right_col.addLayout(slip_layout)
        
        # Position size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Position Size:"))
        self.size_input = QSpinBox()
        self.size_input.setRange(1, 1000)
        self.size_input.setValue(100)
        size_layout.addWidget(self.size_input)
        right_col.addLayout(size_layout)
        
        config_layout.addLayout(right_col)
        
        layout.addWidget(config_group)
        
        # Run button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.run_btn = QPushButton("▶️ Run Backtest")
        self.run_btn.setMinimumWidth(150)
        self.run_btn.clicked.connect(self._run_backtest)
        button_layout.addWidget(self.run_btn)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Results section
        results_layout = QHBoxLayout()
        
        # Left: Chart
        chart_group = QGroupBox("Equity Curve")
        chart_layout = QVBoxLayout(chart_group)
        
        self.figure = Figure(figsize=(8, 5), facecolor='#1E1E1E')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1E1E1E')
        self.ax.tick_params(colors='#E0E0E0')
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        chart_layout.addWidget(self.canvas)
        results_layout.addWidget(chart_group, 3)
        
        # Right: Metrics + Log
        right_panel = QVBoxLayout()
        
        # Metrics table
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_table = QTableWidget(8, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setMaximumHeight(300)
        
        metrics_layout.addWidget(self.metrics_table)
        right_panel.addWidget(metrics_group)
        
        # Log
        log_group = QGroupBox("Backtest Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        right_panel.addWidget(log_group)
        
        results_layout.addLayout(right_panel, 2)
        
        layout.addLayout(results_layout)
    
    def _run_backtest(self):
        """Start backtest"""
        # Collect parameters
        params = {
            'symbol': self.symbol_combo.currentText(),
            'strategy': self.strategy_combo.currentText(),
            'capital': self.capital_input.value(),
            'commission': self.comm_input.value() / 100,
            'slippage': self.slip_input.value() / 100,
            'position_size': self.size_input.value(),
        }
        
        # Clear previous results
        self.log_text.clear()
        self.ax.clear()
        self.canvas.draw()
        
        # Disable button and show progress
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Create and start worker
        self.backtest_worker = BacktestWorker(params)
        self.backtest_worker.progress.connect(self.progress_bar.setValue)
        self.backtest_worker.log.connect(self._append_log)
        self.backtest_worker.finished.connect(self._on_backtest_complete)
        self.backtest_worker.error.connect(self._on_backtest_error)
        self.backtest_worker.start()
    
    def _append_log(self, message: str):
        """Append message to log"""
        self.log_text.append(message)
    
    def _on_backtest_complete(self, results: dict):
        """Handle backtest completion"""
        self.results = results
        
        # Plot equity curve
        self.ax.clear()
        self.ax.plot(results['equity_curve'], color='#42A5F5', linewidth=2)
        self.ax.set_title('Equity Curve', color='#E0E0E0', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Days', color='#E0E0E0')
        self.ax.set_ylabel('Portfolio Value ($)', color='#E0E0E0')
        self.ax.grid(True, alpha=0.2, color='#444444')
        self.ax.set_facecolor('#1E1E1E')
        self.ax.tick_params(colors='#E0E0E0')
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update metrics table
        metrics = [
            ("Total Return", f"{results['total_return']:.2f}%"),
            ("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}"),
            ("Max Drawdown", f"{results['max_drawdown']:.2f}%"),
            ("Win Rate", f"{results['win_rate']*100:.1f}%"),
            ("Total Trades", str(results['total_trades'])),
            ("Avg Win", f"${results['avg_win']:.2f}"),
            ("Avg Loss", f"${results['avg_loss']:.2f}"),
            ("Final Value", f"${results['equity_curve'][-1]:,.2f}"),
        ]
        
        self.metrics_table.setRowCount(len(metrics))
        for row, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(value))
        
        # Re-enable button
        self.run_btn.setEnabled(True)
        self.progress_bar.hide()
    
    def _on_backtest_error(self, error: str):
        """Handle backtest error"""
        self.run_btn.setEnabled(True)
        self.progress_bar.hide()
        self._append_log(f"✗ Error: {error}")
        QMessageBox.critical(self, "Backtest Error", f"Backtest failed:\n{error}")
