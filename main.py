"""
Multi-Timeframe Trading Optimizer - Main Entry Point
"""

import sys
import os

from PyQt6.QtWidgets import QApplication

# Use new professional tabbed GUI by default
# Set environment variable USE_OLD_GUI=1 to use legacy single-scroll GUI
USE_OLD_GUI = os.environ.get('USE_OLD_GUI', '0') == '1'

if USE_OLD_GUI:
    from gui.main_window import MainWindow
    print("Using legacy single-scroll GUI (set USE_OLD_GUI=0 to use new tabbed GUI)")
else:
    from gui.main_window_v2 import MainWindow
    print("Using professional tabbed GUI (set USE_OLD_GUI=1 to use legacy GUI)")


def main():
    """Launch the trading application"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
