"""
LiquidUI - Multi-Timeframe Trading Optimizer

Main entry point. Launches the PyQt6 desktop application.
"""

import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    """Launch the trading application"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
