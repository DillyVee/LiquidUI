"""
GUI Styling and Constants

Modern professional dark theme for the trading platform.
"""

MAIN_STYLESHEET = """
/* Main Window */
QMainWindow {
    background-color: #0d1117;
    color: #c9d1d9;
}

QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
}

/* Labels */
QLabel {
    color: #c9d1d9;
    background-color: transparent;
    padding: 1px;
}

/* Buttons */
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: 500;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #30363d;
    border: 1px solid #58a6ff;
}

QPushButton:pressed {
    background-color: #161b22;
    border: 1px solid #58a6ff;
}

QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border: 1px solid #21262d;
}

/* Input Fields */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: #1f6feb;
    min-height: 20px;
}

QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border: 1px solid #58a6ff;
}

QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 1px solid #58a6ff;
    background-color: #161b22;
}

/* Spin Box Arrows */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #21262d;
    border: none;
    border-top-right-radius: 4px;
    width: 16px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #30363d;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #21262d;
    border: none;
    border-bottom-right-radius: 4px;
    width: 16px;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #30363d;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 5px solid #8b949e;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 5px solid #8b949e;
}

/* ComboBox */
QComboBox {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 20px;
}

QComboBox:hover {
    border: 1px solid #58a6ff;
}

QComboBox:focus {
    border: 1px solid #58a6ff;
    background-color: #161b22;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #8b949e;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 4px 8px;
    border-radius: 3px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #21262d;
}

/* Progress Bar */
QProgressBar {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    text-align: center;
    color: #c9d1d9;
    font-weight: 600;
    height: 18px;
}

QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb,
        stop:1 #58a6ff
    );
    border-radius: 5px;
}

/* Group Box */
QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 10px;
    font-weight: 600;
    color: #58a6ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 2px 6px;
    background-color: #161b22;
    border-radius: 3px;
}

/* Checkboxes */
QCheckBox {
    color: #c9d1d9;
    spacing: 6px;
    padding: 2px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #30363d;
    border-radius: 3px;
    background-color: #0d1117;
}

QCheckBox::indicator:hover {
    border: 2px solid #58a6ff;
}

QCheckBox::indicator:checked {
    background-color: #1f6feb;
    border: 2px solid #1f6feb;
    image: none;
}

QCheckBox::indicator:disabled {
    background-color: #161b22;
    border: 2px solid #21262d;
}

/* Scrollbars */
QScrollBar:vertical {
    background-color: #0d1117;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #0d1117;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 5px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #484f58;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Tooltips */
QToolTip {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 9pt;
}

/* Menu Bar */
QMenuBar {
    background-color: #0d1117;
    color: #c9d1d9;
    border-bottom: 1px solid #21262d;
    padding: 2px;
}

QMenuBar::item {
    padding: 4px 10px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #21262d;
}

QMenu {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px;
}

QMenu::item {
    padding: 6px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #21262d;
}

/* Status Bar */
QStatusBar {
    background-color: #0d1117;
    color: #8b949e;
    border-top: 1px solid #21262d;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 6px;
    background-color: #161b22;
    top: -1px;
}

QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 6px 12px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #0d1117;
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}

QTabBar::tab:hover:!selected {
    background-color: #21262d;
}

/* Table Widget */
QTableWidget {
    background-color: #0d1117;
    alternate-background-color: #161b22;
    gridline-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    selection-background-color: #1f6feb;
}

QTableWidget::item {
    padding: 4px;
    border: none;
}

QHeaderView::section {
    background-color: #161b22;
    color: #c9d1d9;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #30363d;
    font-weight: 600;
}

/* Text Edit */
QTextEdit {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    selection-background-color: #1f6feb;
}
"""

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
