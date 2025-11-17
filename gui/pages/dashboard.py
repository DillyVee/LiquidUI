"""
Dashboard Page - Portfolio Overview and Performance Metrics
"""

import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)


class MetricCard(QFrame):
    """Card widget for displaying a single metric"""

    def __init__(self, title: str, value: str, change: str = None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 15px;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(title_label)

        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet("color: #E0E0E0; font-size: 28px; font-weight: bold;")
        layout.addWidget(value_label)

        # Change (optional)
        if change:
            change_label = QLabel(change)
            color = "#4CAF50" if "+" in change else "#F44336"
            change_label.setStyleSheet(f"color: {color}; font-size: 14px;")
            layout.addWidget(change_label)


class DashboardPage(QWidget):
    """Main dashboard page showing portfolio overview"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        header = QLabel("📊 Dashboard")
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #E0E0E0;")
        layout.addWidget(header)

        # Welcome message
        welcome = QLabel(
            f"Welcome back! Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        welcome.setStyleSheet("color: #888888; font-size: 14px;")
        layout.addWidget(welcome)

        # Metrics grid
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(15)

        # Portfolio metrics (example data)
        metrics = [
            ("Portfolio Value", "$125,430.50", "+$2,345.20 (+1.9%)"),
            ("Today's P&L", "+$1,234.56", "+0.98%"),
            ("Total Return", "+25.4%", "Since Jan 2025"),
            ("Sharpe Ratio", "2.15", "Last 30 days"),
            ("Win Rate", "67.3%", "145 / 215 trades"),
            ("Max Drawdown", "-8.2%", "Feb 15, 2025"),
        ]

        row, col = 0, 0
        for title, value, change in metrics:
            card = MetricCard(title, value, change)
            metrics_layout.addWidget(card, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        layout.addLayout(metrics_layout)

        # Recent trades section
        recent_label = QLabel("📈 Recent Trades")
        recent_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #E0E0E0; margin-top: 10px;"
        )
        layout.addWidget(recent_label)

        # Trades table
        table = QTableWidget(5, 6)
        table.setHorizontalHeaderLabels(
            ["Time", "Symbol", "Side", "Quantity", "Price", "P&L"]
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        # Sample data
        trades = [
            ("10:23:15", "SPY", "BUY", "100", "$452.30", "+$234.50"),
            ("10:45:32", "AAPL", "SELL", "50", "$178.20", "+$125.30"),
            ("11:15:08", "TSLA", "BUY", "25", "$245.60", "-$45.20"),
            ("14:30:21", "NVDA", "SELL", "30", "$875.40", "+$890.15"),
            ("15:55:42", "MSFT", "BUY", "40", "$425.80", "+$156.20"),
        ]

        for row_idx, trade_data in enumerate(trades):
            for col_idx, value in enumerate(trade_data):
                item = QTableWidgetItem(value)

                # Color P&L
                if col_idx == 5:
                    if "+" in value:
                        item.setForeground(Qt.GlobalColor.green)
                    else:
                        item.setForeground(Qt.GlobalColor.red)

                table.setItem(row_idx, col_idx, item)

        layout.addWidget(table)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.clicked.connect(self._refresh_dashboard)
        button_layout.addWidget(refresh_btn)

        export_btn = QPushButton("📊 Export Report")
        button_layout.addWidget(export_btn)

        layout.addLayout(button_layout)

        layout.addStretch()

    def _refresh_dashboard(self):
        """Refresh dashboard data"""
        # TODO: Implement data refresh
        print("Refreshing dashboard...")
