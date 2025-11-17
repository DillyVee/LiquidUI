"""
LiquidUI - Professional Quantitative Trading Platform
Main Window and Application Entry Point
"""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor

# Import page modules (will create these)
from gui.pages.dashboard import DashboardPage
from gui.pages.data_manager import DataManagerPage
from gui.pages.backtest import BacktestPage
from gui.pages.strategy_config import StrategyConfigPage
from gui.pages.live_trading import LiveTradingPage
from gui.pages.risk_monitor import RiskMonitorPage


class NavigationButton(QPushButton):
    """Custom styled navigation button"""

    def __init__(self, text: str, icon_name: str = None, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Modern styling
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #E0E0E0;
                border: none;
                border-left: 3px solid transparent;
                text-align: left;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.05);
                color: #FFFFFF;
            }
            QPushButton:checked {
                background-color: rgba(66, 165, 245, 0.15);
                border-left: 3px solid #42A5F5;
                color: #42A5F5;
            }
        """)


class SideNavigation(QFrame):
    """Modern side navigation panel"""

    page_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Dark background
        self.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-right: 1px solid #333333;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Logo/Title
        title_widget = QWidget()
        title_widget.setStyleSheet("background-color: #252525; padding: 20px;")
        title_layout = QVBoxLayout(title_widget)

        title = QLabel("💧 LiquidUI")
        title.setStyleSheet("color: #42A5F5; font-size: 24px; font-weight: bold;")
        subtitle = QLabel("Quantitative Trading Platform")
        subtitle.setStyleSheet("color: #888888; font-size: 11px;")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addWidget(title_widget)

        # Navigation buttons
        self.nav_buttons = []

        pages = [
            ("📊 Dashboard", 0),
            ("📁 Data Manager", 1),
            ("🔬 Backtest", 2),
            ("⚙️ Strategy Config", 3),
            ("🚀 Live Trading", 4),
            ("🛡️ Risk Monitor", 5),
        ]

        for text, page_idx in pages:
            btn = NavigationButton(text)
            btn.clicked.connect(lambda checked, idx=page_idx: self._on_button_clicked(idx))
            self.nav_buttons.append(btn)
            layout.addWidget(btn)

        # Set first button as active
        self.nav_buttons[0].setChecked(True)

        # Spacer
        layout.addStretch()

        # Version info at bottom
        version = QLabel("v1.0.0")
        version.setStyleSheet("color: #555555; padding: 10px 20px; font-size: 10px;")
        layout.addWidget(version)

    def _on_button_clicked(self, page_idx: int):
        """Handle navigation button click"""
        # Uncheck all buttons
        for btn in self.nav_buttons:
            btn.setChecked(False)

        # Check clicked button
        self.nav_buttons[page_idx].setChecked(True)

        # Emit signal
        self.page_changed.emit(page_idx)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidUI - Quantitative Trading Platform")
        self.setMinimumSize(1400, 900)

        # Center window on screen
        self._center_on_screen()

        # Setup UI
        self._setup_ui()

        # Apply dark theme
        self._apply_theme()

    def _setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout (horizontal: navigation + content)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Side navigation
        self.navigation = SideNavigation()
        self.navigation.page_changed.connect(self._change_page)
        main_layout.addWidget(self.navigation)

        # Content area (stacked widget for different pages)
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background-color: #2B2B2B;")

        # Add pages
        self.pages.addWidget(DashboardPage())
        self.pages.addWidget(DataManagerPage())
        self.pages.addWidget(BacktestPage())
        self.pages.addWidget(StrategyConfigPage())
        self.pages.addWidget(LiveTradingPage())
        self.pages.addWidget(RiskMonitorPage())

        main_layout.addWidget(self.pages)

    def _change_page(self, page_idx: int):
        """Change the displayed page"""
        self.pages.setCurrentIndex(page_idx)

    def _center_on_screen(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def _apply_theme(self):
        """Apply modern dark theme to the application"""
        palette = QPalette()

        # Dark theme colors
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
        palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(224, 224, 224))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(224, 224, 224))
        palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
        palette.setColor(QPalette.ColorRole.Button, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
        palette.setColor(QPalette.ColorRole.Link, QColor(66, 165, 245))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(66, 165, 245))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))

        self.setPalette(palette)

        # Global stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2B2B2B;
            }
            QLabel {
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #42A5F5;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #64B5F6;
            }
            QPushButton:pressed {
                background-color: #2196F3;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #1E1E1E;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                color: #E0E0E0;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 1px solid #42A5F5;
            }
            QTableWidget {
                background-color: #1E1E1E;
                alternate-background-color: #252525;
                gridline-color: #333333;
                border: 1px solid #333333;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #E0E0E0;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #42A5F5;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #1E1E1E;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            QScrollBar:horizontal {
                background-color: #1E1E1E;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #555555;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #666666;
            }
        """)


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("LiquidUI")
    app.setOrganizationName("LiquidUI")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
