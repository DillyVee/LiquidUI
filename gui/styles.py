"""
GUI Styling and Constants
"""
from config.settings import DARK_THEME_STYLESHEET

# Export the dark theme
MAIN_STYLESHEET = DARK_THEME_STYLESHEET

# Additional button styles
LIVE_TRADING_BUTTON_ACTIVE = """
    QPushButton { background-color: #0a6e0a; font-weight: bold; }
    QPushButton:hover { background-color: #0d8f0d; }
    QPushButton:disabled { background-color: #0a0a0a; }
"""

LIVE_TRADING_BUTTON_STOPPED = """
    QPushButton { background-color: #8e0a0a; font-weight: bold; }
    QPushButton:hover { background-color: #b00d0d; }
"""

# Color constants
COLOR_SUCCESS = "#0a6e0a"
COLOR_ERROR = "#ff4444"
COLOR_WARNING = "#FFA500"
COLOR_PRIMARY = "#2979ff"
COLOR_BACKGROUND = "#121212"
COLOR_TEXT = "#ddd"
