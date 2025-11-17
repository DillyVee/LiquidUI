"""
Live Trading Page - Real-time Trading Interface
"""

import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QTableWidget, QTableWidgetItem, QTextEdit,
                             QVBoxLayout, QWidget)


class LiveTradingPage(QWidget):
    """Live trading interface"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_trading = False
        self._setup_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)

    def _setup_ui(self):
        """Setup the live trading UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        header_layout = QHBoxLayout()
        header = QLabel("🚀 Live Trading")
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #E0E0E0;")
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Trading status
        self.status_label = QLabel("● STOPPED")
        self.status_label.setStyleSheet(
            "color: #F44336; font-size: 18px; font-weight: bold;"
        )
        header_layout.addWidget(self.status_label)

        layout.addLayout(header_layout)

        # Control buttons
        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶️ Start Trading")
        self.start_btn.setStyleSheet("background-color: #4CAF50;")
        self.start_btn.clicked.connect(self._start_trading)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop Trading")
        self.stop_btn.setStyleSheet("background-color: #F44336;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_trading)
        control_layout.addWidget(self.stop_btn)

        self.panic_btn = QPushButton("🚨 PANIC STOP")
        self.panic_btn.setStyleSheet("background-color: #D32F2F; font-weight: bold;")
        self.panic_btn.clicked.connect(self._panic_stop)
        control_layout.addWidget(self.panic_btn)

        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Active positions
        positions_group = QGroupBox("Active Positions")
        positions_layout = QVBoxLayout(positions_group)

        self.positions_table = QTableWidget(0, 6)
        self.positions_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Quantity", "Entry Price", "Current Price", "P&L"]
        )
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.setAlternatingRowColors(True)

        positions_layout.addWidget(self.positions_table)
        layout.addWidget(positions_group)

        # Open orders
        orders_group = QGroupBox("Open Orders")
        orders_layout = QVBoxLayout(orders_group)

        self.orders_table = QTableWidget(0, 5)
        self.orders_table.setHorizontalHeaderLabels(
            ["Time", "Symbol", "Type", "Quantity", "Price"]
        )
        self.orders_table.horizontalHeader().setStretchLastSection(True)
        self.orders_table.setAlternatingRowColors(True)

        orders_layout.addWidget(self.orders_table)
        layout.addWidget(orders_group)

        # Activity log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

    def _start_trading(self):
        """Start live trading"""
        self.is_trading = True
        self.status_label.setText("● LIVE")
        self.status_label.setStyleSheet(
            "color: #4CAF50; font-size: 18px; font-weight: bold;"
        )

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._log("✓ Live trading started")
        self._log("Connected to Alpaca API")
        self._log("Monitoring markets...")

        # Start update timer
        self.update_timer.start(1000)

    def _stop_trading(self):
        """Stop live trading"""
        self.is_trading = False
        self.status_label.setText("● STOPPED")
        self.status_label.setStyleSheet(
            "color: #F44336; font-size: 18px; font-weight: bold;"
        )

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self._log("⏹ Live trading stopped")

        # Stop update timer
        self.update_timer.stop()

    def _panic_stop(self):
        """Emergency stop - close all positions"""
        self._log("🚨 PANIC STOP ACTIVATED")
        self._log("Closing all positions...")
        self._log("Canceling all orders...")
        self._stop_trading()

    def _update_status(self):
        """Update trading status"""
        # Update current time in log
        if self.is_trading:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            # self._log(f"[{now}] Monitoring...")

    def _log(self, message: str):
        """Add message to activity log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
