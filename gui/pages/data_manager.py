"""
Data Manager Page - Download, Validate, and View Market Data
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QComboBox,
    QDateEdit, QTextEdit, QGroupBox, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataDownloadWorker(QThread):
    """Worker thread for downloading data"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(self, symbol: str, start_date: str, end_date: str):
        super().__init__()
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
    
    def run(self):
        """Download data in background"""
        try:
            self.progress.emit(f"Downloading {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                self.error.emit(f"No data found for {self.symbol}")
            else:
                self.progress.emit(f"Downloaded {len(df)} bars")
                self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e))


class DataManagerPage(QWidget):
    """Data management page"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self.download_worker = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the data manager UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("📁 Data Manager")
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #E0E0E0;")
        layout.addWidget(header)
        
        # Download section
        download_group = QGroupBox("Download Market Data")
        download_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        download_layout = QVBoxLayout(download_group)
        
        # Input form
        form_layout = QHBoxLayout()
        
        # Symbol input
        symbol_label = QLabel("Symbol:")
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g., SPY, AAPL, TSLA")
        self.symbol_input.setText("SPY")
        self.symbol_input.setMaximumWidth(150)
        
        # Start date
        start_label = QLabel("Start Date:")
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-365))
        self.start_date.setCalendarPopup(True)
        self.start_date.setMaximumWidth(150)
        
        # End date
        end_label = QLabel("End Date:")
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        self.end_date.setMaximumWidth(150)
        
        # Download button
        self.download_btn = QPushButton("📥 Download Data")
        self.download_btn.clicked.connect(self._download_data)
        
        form_layout.addWidget(symbol_label)
        form_layout.addWidget(self.symbol_input)
        form_layout.addWidget(start_label)
        form_layout.addWidget(self.start_date)
        form_layout.addWidget(end_label)
        form_layout.addWidget(self.end_date)
        form_layout.addWidget(self.download_btn)
        form_layout.addStretch()
        
        download_layout.addLayout(form_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.hide()
        download_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to download data")
        self.status_label.setStyleSheet("color: #888888;")
        download_layout.addWidget(self.status_label)
        
        layout.addWidget(download_group)
        
        # Data preview section
        preview_label = QLabel("📊 Data Preview")
        preview_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #E0E0E0; margin-top: 10px;")
        layout.addWidget(preview_label)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.data_table)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        validate_btn = QPushButton("✓ Validate Data")
        validate_btn.clicked.connect(self._validate_data)
        button_layout.addWidget(validate_btn)
        
        save_btn = QPushButton("💾 Save to Database")
        button_layout.addWidget(save_btn)
        
        export_btn = QPushButton("📤 Export CSV")
        button_layout.addWidget(export_btn)
        
        layout.addLayout(button_layout)
    
    def _download_data(self):
        """Download market data"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
        
        start = self.start_date.date().toString("yyyy-MM-dd")
        end = self.end_date.date().toString("yyyy-MM-dd")
        
        # Disable button and show progress
        self.download_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(f"Downloading {symbol}...")
        
        # Create and start worker thread
        self.download_worker = DataDownloadWorker(symbol, start, end)
        self.download_worker.progress.connect(self._on_progress)
        self.download_worker.finished.connect(self._on_download_complete)
        self.download_worker.error.connect(self._on_download_error)
        self.download_worker.start()
    
    def _on_progress(self, message: str):
        """Update progress message"""
        self.status_label.setText(message)
    
    def _on_download_complete(self, df: pd.DataFrame):
        """Handle successful download"""
        self.current_data = df
        self._display_data(df)
        
        self.download_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText(f"✓ Downloaded {len(df)} bars successfully")
        self.status_label.setStyleSheet("color: #4CAF50;")
    
    def _on_download_error(self, error: str):
        """Handle download error"""
        self.download_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText(f"✗ Error: {error}")
        self.status_label.setStyleSheet("color: #F44336;")
        
        QMessageBox.critical(self, "Download Error", f"Failed to download data:\n{error}")
    
    def _display_data(self, df: pd.DataFrame):
        """Display data in table"""
        # Setup table
        self.data_table.setRowCount(min(len(df), 100))  # Show first 100 rows
        self.data_table.setColumnCount(len(df.columns) + 1)  # +1 for date
        self.data_table.setHorizontalHeaderLabels(["Date"] + list(df.columns))
        
        # Fill data
        for row_idx in range(min(len(df), 100)):
            # Date column
            date_item = QTableWidgetItem(df.index[row_idx].strftime("%Y-%m-%d"))
            self.data_table.setItem(row_idx, 0, date_item)
            
            # Data columns
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx][col_name]
                if isinstance(value, float):
                    item = QTableWidgetItem(f"{value:.2f}")
                else:
                    item = QTableWidgetItem(str(value))
                self.data_table.setItem(row_idx, col_idx + 1, item)
        
        self.data_table.resizeColumnsToContents()
    
    def _validate_data(self):
        """Validate downloaded data"""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please download data first")
            return
        
        # Simple validation checks
        issues = []
        
        # Check for missing values
        missing = self.current_data.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        # Check for duplicates
        duplicates = self.current_data.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate dates: {duplicates}")
        
        # Check date order
        if not self.current_data.index.is_monotonic_increasing:
            issues.append("Dates are not in ascending order")
        
        # Show results
        if issues:
            QMessageBox.warning(
                self,
                "Validation Issues",
                "Data validation found issues:\n\n" + "\n".join(issues)
            )
        else:
            QMessageBox.information(
                self,
                "Validation Passed",
                "✓ Data validation passed all checks!"
            )
