# 💧 LiquidUI - Graphical User Interface

Modern, professional GUI for the quantitative trading platform. No command line required!

## 🚀 Quick Start

### Windows
Double-click `launch_gui.bat` or run:
```cmd
python launch_gui.py
```

### Linux/Mac
```bash
python3 launch_gui.py
```

## 📱 Features

### 1. 📊 Dashboard
- Portfolio overview with key metrics
- Real-time P&L tracking
- Recent trades history
- Performance statistics

### 2. 📁 Data Manager
- Download market data from Yahoo Finance
- Interactive date range selection
- Data validation tools
- Preview downloaded data in table format
- Export to CSV or save to database

### 3. 🔬 Backtest Runner
- Visual strategy backtesting
- Multiple built-in strategies:
  - RSI Mean Reversion
  - MACD Crossover
  - Bollinger Breakout
  - Moving Average Crossover
- Real-time equity curve visualization
- Comprehensive performance metrics
- Configurable transaction costs

### 4. ⚙️ Strategy Configuration
- Configure strategy parameters
- Risk management settings:
  - Position sizing
  - Stop loss / Take profit
  - Maximum daily loss limits
- Save and load configurations
- Test strategies before deployment

### 5. 🚀 Live Trading
- Start/stop live trading with one click
- Monitor active positions in real-time
- View open orders
- Activity log with timestamps
- Emergency **PANIC STOP** button

### 6. 🛡️ Risk Monitor
- Real-time risk metrics dashboard
- Visual risk indicators with color-coded alerts
- P&L distribution analysis
- Drawdown tracking
- Risk alerts and warnings table
- **KILL SWITCH** for emergency stops

## 🎨 Interface

- **Modern dark theme** optimized for long trading sessions
- **Intuitive navigation** with sidebar menu
- **Real-time updates** using background workers
- **Interactive charts** powered by Matplotlib
- **Color-coded metrics** for quick visual assessment

## 💡 Usage Tips

### Downloading Data
1. Go to **Data Manager**
2. Enter symbol (e.g., SPY, AAPL)
3. Select date range
4. Click **Download Data**
5. Preview and validate before saving

### Running a Backtest
1. Go to **Backtest Runner**
2. Select symbol and strategy
3. Configure parameters (capital, commission, etc.)
4. Click **Run Backtest**
5. View equity curve and performance metrics

### Going Live
1. Configure your strategy in **Strategy Config**
2. Set risk limits and parameters
3. Go to **Live Trading**
4. Click **Start Trading**
5. Monitor positions and activity log
6. Use **Stop Trading** to safely exit

### Monitoring Risk
1. Go to **Risk Monitor**
2. Check risk metric cards (green = safe, yellow = caution, red = danger)
3. Review P&L distribution and drawdown charts
4. Watch for alerts in the warnings table
5. Use **Kill Switch** if emergency action needed

## 🔧 Architecture

```
gui/
├── main_window.py          # Main application window and navigation
├── pages/
│   ├── dashboard.py        # Portfolio overview dashboard
│   ├── data_manager.py     # Data download and management
│   ├── backtest.py         # Backtest runner with charts
│   ├── strategy_config.py  # Strategy configuration
│   ├── live_trading.py     # Live trading interface
│   └── risk_monitor.py     # Risk monitoring dashboard
└── widgets/                # Custom reusable widgets (future)
```

## 🎯 Keyboard Shortcuts

- `Ctrl+Q` - Quit application (future)
- `F5` - Refresh current page (future)
- `Ctrl+S` - Save configuration (future)

## 🔌 Integration

The GUI integrates with the core trading pipeline:
- `data_layer/` for market data
- `backtest/` for backtesting engine
- `execution/` for order routing
- `risk/` for risk management
- `monitoring/` for metrics

## 🛠️ Customization

### Adding New Strategies

Edit `gui/pages/strategy_config.py`:
```python
self.strategy_combo.addItems([
    "Your Custom Strategy",
    # ... existing strategies
])
```

### Changing Theme Colors

Edit `gui/main_window.py` in the `_apply_theme()` method to customize colors.

### Adding New Pages

1. Create new page in `gui/pages/your_page.py`
2. Import in `gui/main_window.py`
3. Add to pages stack and navigation

## 📝 Notes

- The GUI uses **PyQt6** for the interface
- Charts are rendered with **Matplotlib**
- Data downloads use **yfinance** API (free, no authentication required)
- Background operations run in separate threads to keep UI responsive

## ⚡ Performance Tips

- Close unused pages to save memory
- Limit data preview to first 100 rows
- Use date range filters when downloading large datasets
- Monitor system resources during live trading

## 🐛 Troubleshooting

### GUI won't start
```cmd
pip install PyQt6 matplotlib
```

### Charts not displaying
```cmd
pip install matplotlib
```

### Data download fails
- Check internet connection
- Verify symbol ticker is valid
- Try a different date range

## 🔐 Security

- **Never share screenshots** containing API keys or account details
- Use **separate credentials** for testing vs. live trading
- The **PANIC STOP** button closes all positions immediately
- The **KILL SWITCH** stops ALL trading activity instantly

## 📚 Learn More

- See `examples/` directory for strategy implementation examples
- Read main `README.md` for architecture details
- Check `CONTRIBUTING.md` for development guidelines

---

**Happy Trading! 📈💰**
