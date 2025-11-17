"""
Risk Monitor Page - Real-time Risk Metrics and Alerts
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class RiskCard(QFrame):
    """Card widget for displaying a risk metric"""
    
    def __init__(self, title: str, value: str, limit: str, percent: float, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(title_label)
        
        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet("color: #E0E0E0; font-size: 24px; font-weight: bold;")
        layout.addWidget(value_label)
        
        # Limit
        limit_label = QLabel(f"Limit: {limit}")
        limit_label.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(limit_label)
        
        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(int(percent))
        progress.setTextVisible(False)
        progress.setMaximumHeight(8)
        
        # Color based on percentage
        if percent < 50:
            color = "#4CAF50"  # Green
        elif percent < 80:
            color = "#FFC107"  # Yellow
        else:
            color = "#F44336"  # Red
        
        progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #333333;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)
        
        layout.addWidget(progress)


class RiskMonitorPage(QWidget):
    """Risk monitoring dashboard"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the risk monitor UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        header = QLabel("🛡️ Risk Monitor")
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #E0E0E0;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Overall status
        status = QLabel("● ALL SYSTEMS NORMAL")
        status.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(status)
        
        layout.addLayout(header_layout)
        
        # Risk metrics cards
        metrics_layout = QHBoxLayout()
        
        risk_cards = [
            ("Daily P&L", "$1,234", "$5,000", 24.7),
            ("Position Exposure", "$45,230", "$100,000", 45.2),
            ("VaR (95%)", "$2,156", "$10,000", 21.6),
            ("Margin Used", "32%", "80%", 40.0),
        ]
        
        for title, value, limit, percent in risk_cards:
            card = RiskCard(title, value, limit, percent)
            metrics_layout.addWidget(card)
        
        layout.addLayout(metrics_layout)
        
        # Charts section
        charts_layout = QHBoxLayout()
        
        # P&L chart
        pnl_group = QGroupBox("P&L Distribution")
        pnl_layout = QVBoxLayout(pnl_group)
        
        self.pnl_figure = Figure(figsize=(5, 3), facecolor='#1E1E1E')
        self.pnl_canvas = FigureCanvas(self.pnl_figure)
        self.pnl_ax = self.pnl_figure.add_subplot(111)
        
        # Generate sample P&L distribution
        pnl_data = np.random.normal(500, 1000, 100)
        self.pnl_ax.hist(pnl_data, bins=20, color='#42A5F5', alpha=0.7, edgecolor='#64B5F6')
        self.pnl_ax.axvline(0, color='#F44336', linestyle='--', linewidth=2)
        self.pnl_ax.set_facecolor('#1E1E1E')
        self.pnl_ax.tick_params(colors='#E0E0E0')
        self.pnl_ax.spines['bottom'].set_color('#444444')
        self.pnl_ax.spines['left'].set_color('#444444')
        self.pnl_ax.spines['top'].set_visible(False)
        self.pnl_ax.spines['right'].set_visible(False)
        self.pnl_figure.tight_layout()
        
        pnl_layout.addWidget(self.pnl_canvas)
        charts_layout.addWidget(pnl_group)
        
        # Drawdown chart
        dd_group = QGroupBox("Drawdown Over Time")
        dd_layout = QVBoxLayout(dd_group)
        
        self.dd_figure = Figure(figsize=(5, 3), facecolor='#1E1E1E')
        self.dd_canvas = FigureCanvas(self.dd_figure)
        self.dd_ax = self.dd_figure.add_subplot(111)
        
        # Generate sample drawdown
        days = np.arange(100)
        drawdown = -np.abs(np.random.normal(0, 3, 100))
        self.dd_ax.fill_between(days, drawdown, 0, color='#F44336', alpha=0.5)
        self.dd_ax.plot(days, drawdown, color='#D32F2F', linewidth=2)
        self.dd_ax.set_facecolor('#1E1E1E')
        self.dd_ax.tick_params(colors='#E0E0E0')
        self.dd_ax.spines['bottom'].set_color('#444444')
        self.dd_ax.spines['left'].set_color('#444444')
        self.dd_ax.spines['top'].set_visible(False)
        self.dd_ax.spines['right'].set_visible(False)
        self.dd_ax.set_ylabel('Drawdown (%)', color='#E0E0E0')
        self.dd_figure.tight_layout()
        
        dd_layout.addWidget(self.dd_canvas)
        charts_layout.addWidget(dd_group)
        
        layout.addLayout(charts_layout)
        
        # Risk alerts table
        alerts_group = QGroupBox("⚠️ Risk Alerts & Warnings")
        alerts_layout = QVBoxLayout(alerts_group)
        
        self.alerts_table = QTableWidget(3, 4)
        self.alerts_table.setHorizontalHeaderLabels(["Time", "Level", "Type", "Message"])
        self.alerts_table.horizontalHeader().setStretchLastSection(True)
        self.alerts_table.setAlternatingRowColors(True)
        self.alerts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Sample alerts
        alerts = [
            ("15:23:45", "⚠️ WARNING", "Position Size", "SPY position approaching 80% of limit"),
            ("14:15:22", "ℹ️ INFO", "Volatility", "Market volatility increased by 25%"),
            ("10:05:11", "⚠️ WARNING", "Drawdown", "Daily drawdown reached -5.2%"),
        ]
        
        for row_idx, (time, level, type_, msg) in enumerate(alerts):
            self.alerts_table.setItem(row_idx, 0, QTableWidgetItem(time))
            self.alerts_table.setItem(row_idx, 1, QTableWidgetItem(level))
            self.alerts_table.setItem(row_idx, 2, QTableWidgetItem(type_))
            self.alerts_table.setItem(row_idx, 3, QTableWidgetItem(msg))
        
        alerts_layout.addWidget(self.alerts_table)
        layout.addWidget(alerts_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh Metrics")
        button_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("📊 Export Risk Report")
        button_layout.addWidget(export_btn)
        
        kill_switch_btn = QPushButton("🚨 ACTIVATE KILL SWITCH")
        kill_switch_btn.setStyleSheet("background-color: #D32F2F; font-weight: bold;")
        button_layout.addWidget(kill_switch_btn)
        
        layout.addLayout(button_layout)
