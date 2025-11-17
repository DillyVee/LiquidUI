"""
GUI Styling and Constants
"""

from config.settings import DARK_THEME_STYLESHEET

# Export the dark theme as MAIN_STYLESHEET
MAIN_STYLESHEET = DARK_THEME_STYLESHEET

# Live Trading Button Styles
LIVE_TRADING_BUTTON_ACTIVE = """
    QPushButton { 
        background-color: #238636;
        border: 1px solid #2ea043;
        color: #ffffff;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 16px;
    }
    QPushButton:hover { 
        background-color: #2ea043;
        border: 1px solid #3fb950;
    }
    QPushButton:disabled { 
        background-color: #161b22; 
        color: #484f58;
        border: 1px solid #21262d;
    }
"""

LIVE_TRADING_BUTTON_STOPPED = """
    QPushButton { 
        background-color: #da3633;
        border: 1px solid #f85149;
        color: #ffffff;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 16px;
    }
    QPushButton:hover { 
        background-color: #f85149;
        border: 1px solid #ff7b72;
    }
"""

# Color constants
COLOR_SUCCESS = "#4CAF50"  # Material Green
COLOR_ERROR = "#F44336"  # Material Red
COLOR_DANGER = "#F44336"  # Material Red (alias for ERROR)
COLOR_WARNING = "#FFA726"  # Material Orange
COLOR_PRIMARY = "#2196F3"  # Material Blue
COLOR_BACKGROUND = "#0d1117"  # GitHub dark background
COLOR_TEXT = "#c9d1d9"  # GitHub text
