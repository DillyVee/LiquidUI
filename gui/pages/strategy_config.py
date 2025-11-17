"""
Strategy Configuration Page
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QSpinBox, QTableWidget, QTableWidgetItem,
                             QTextEdit, QVBoxLayout, QWidget)


class StrategyConfigPage(QWidget):
    """Strategy configuration page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the strategy config UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        header = QLabel("⚙️ Strategy Configuration")
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #E0E0E0;")
        layout.addWidget(header)

        # Strategy selection
        strategy_group = QGroupBox("Select Strategy")
        strategy_layout = QHBoxLayout(strategy_group)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(
            [
                "RSI Mean Reversion",
                "MACD Crossover",
                "Bollinger Breakout",
                "Moving Average Crossover",
                "Custom Strategy",
            ]
        )
        self.strategy_combo.currentIndexChanged.connect(self._load_strategy_params)
        strategy_layout.addWidget(self.strategy_combo)

        new_btn = QPushButton("➕ New Strategy")
        strategy_layout.addWidget(new_btn)

        layout.addWidget(strategy_group)

        # Parameters
        params_group = QGroupBox("Strategy Parameters")
        self.params_layout = QVBoxLayout(params_group)

        # Will be populated based on selected strategy
        self._load_strategy_params(0)

        layout.addWidget(params_group)

        # Risk settings
        risk_group = QGroupBox("Risk Management")
        risk_layout = QVBoxLayout(risk_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Max Position Size:"))
        max_pos = QSpinBox()
        max_pos.setRange(1, 10000)
        max_pos.setValue(1000)
        row1.addWidget(max_pos)

        row1.addWidget(QLabel("Stop Loss:"))
        stop_loss = QDoubleSpinBox()
        stop_loss.setRange(0, 100)
        stop_loss.setValue(2.0)
        stop_loss.setSuffix("%")
        row1.addWidget(stop_loss)

        risk_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Take Profit:"))
        take_profit = QDoubleSpinBox()
        take_profit.setRange(0, 100)
        take_profit.setValue(5.0)
        take_profit.setSuffix("%")
        row2.addWidget(take_profit)

        row2.addWidget(QLabel("Max Daily Loss:"))
        max_loss = QDoubleSpinBox()
        max_loss.setRange(0, 100000)
        max_loss.setValue(5000)
        max_loss.setPrefix("$")
        row2.addWidget(max_loss)

        risk_layout.addLayout(row2)

        layout.addWidget(risk_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_btn = QPushButton("💾 Save Configuration")
        button_layout.addWidget(save_btn)

        test_btn = QPushButton("🧪 Test Strategy")
        button_layout.addWidget(test_btn)

        layout.addLayout(button_layout)

        layout.addStretch()

    def _load_strategy_params(self, index: int):
        """Load parameters for selected strategy"""
        # Clear existing params
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Load strategy-specific params
        if index == 0:  # RSI Mean Reversion
            params = [
                ("RSI Period", 14, 1, 100),
                ("Oversold Level", 30, 1, 50),
                ("Overbought Level", 70, 50, 100),
            ]
        elif index == 1:  # MACD Crossover
            params = [
                ("Fast Period", 12, 1, 50),
                ("Slow Period", 26, 1, 100),
                ("Signal Period", 9, 1, 50),
            ]
        else:
            params = [
                ("Parameter 1", 10, 1, 100),
                ("Parameter 2", 20, 1, 100),
            ]

        for label, default, min_val, max_val in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label + ":"))
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            row.addWidget(spin)
            row.addStretch()
            self.params_layout.addLayout(row)
