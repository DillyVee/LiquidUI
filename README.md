# Multi-Timeframe Trading Optimizer

A professional-grade algorithmic trading application with multi-timeframe optimization, walk-forward validation, and live paper trading capabilities.

## 🚀 Features

- **Multi-Timeframe Optimization**: Simultaneously optimize across daily, hourly, and 5-minute timeframes
- **Walk-Forward Analysis**: Detect overfitting with train/test split validation
- **Equity Curve Optimization**: Advanced retracement zone-based trade timing
- **Live Paper Trading**: Integrate with Alpaca for paper trading execution
- **Risk Management**: Position sizing and portfolio controls
- **Batch Processing**: Optimize multiple tickers sequentially
- **Professional GUI**: Dark-themed PyQt6 interface with real-time updates

## 📁 Project Structure

```
trading_app/
├── main.py                    # Application entry point
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration constants
├── data/
│   ├── __init__.py
│   └── loader.py             # Yahoo Finance data loading
├── optimization/
│   ├── __init__.py
│   ├── optimizer.py          # Multi-timeframe optimizer
│   └── metrics.py            # Performance calculations
├── trading/
│   ├── __init__.py
│   └── alpaca_trader.py      # Live trading engine
├── gui/
│   ├── __init__.py
│   ├── main_window.py        # Main UI
│   └── styles.py             # UI styling
└── utils/
    ├── __init__.py
    └── helpers.py            # Utility functions
```

## 🔧 Installation

### 1. Clone or download the project

```bash
git clone <repository-url>
cd trading_app
```

### 2. Create virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys (for live trading)

**IMPORTANT**: Never hardcode API keys in your code!

Set environment variables:

```bash
# Windows
set ALPACA_API_KEY=your_api_key_here
set ALPACA_SECRET_KEY=your_secret_key_here

# macOS/Linux
export ALPACA_API_KEY=your_api_key_here
export ALPACA_SECRET_KEY=your_secret_key_here
```

Or create a `.env` file:

```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

## 🎯 Quick Start

### Run the application

```bash
python main.py
```

### Basic workflow

1. **Load Data**: Enter a ticker (e.g., `AAPL`, `SPY`, `BTC-USD`) and click "Load from Yahoo Finance"
2. **Select Timeframes**: Check the timeframes you want to optimize (daily, hourly, 5-minute)
3. **Configure Parameters**: Set RSI periods, entry/exit thresholds, and cycle parameters
4. **Start Optimization**: Click "Start Multi-Timeframe Optimization"
5. **View Results**: Review the equity curve and performance metrics
6. **Live Trading** (optional): Click "Start Live Trading" for paper trading

## 📊 Configuration

### Optimization Settings (config/settings.py)

```python
@dataclass
class OptimizationConfig:
    DEFAULT_TRIALS: int = 900
    DEFAULT_BATCH_SIZE: int = 500
    DAILY_MAX_DAYS: int = 365 * 10
    HOURLY_MAX_DAYS: int = 730
    FIVEMIN_MAX_DAYS: int = 60
```

### Risk Management

```python
@dataclass
class RiskConfig:
    DEFAULT_POSITION_SIZE: float = 0.05  # 5% per trade
    DEFAULT_MAX_POSITIONS: int = 20
    DEFAULT_STOP_LOSS: float = 0.02      # 2%
```

## 🔍 Key Improvements from Original Code

### 1. **Modular Architecture**
- Separated concerns into logical modules
- Easy to test and maintain individual components
- Clear dependencies between modules

### 2. **Security**
- API keys loaded from environment variables
- No hardcoded credentials in source code
- Safe for version control

### 3. **Configuration Management**
- Centralized settings in `config/settings.py`
- Easy to adjust parameters without code changes
- Type-safe with dataclasses

### 4. **Better Error Handling**
- Specific exception types
- Graceful degradation
- User-friendly error messages

### 5. **Code Organization**
- Single Responsibility Principle
- DRY (Don't Repeat Yourself)
- Clear naming conventions

## 🛠️ Development

### Adding New Features

1. **New Data Source**: Extend `data/loader.py`
2. **New Indicator**: Add to `optimization/metrics.py`
3. **New UI Control**: Modify `gui/main_window.py`
4. **New Configuration**: Update `config/settings.py`

### Running Tests (when implemented)

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black trading_app/

# Check types
mypy trading_app/

# Lint
flake8 trading_app/
```

## 📝 Usage Examples

### Example 1: Optimize Single Ticker

```python
from data import DataLoader
from optimization import MultiTimeframeOptimizer

# Load data
df_dict, error = DataLoader.load_yfinance_data("AAPL")

# Create optimizer
optimizer = MultiTimeframeOptimizer(
    df_dict=df_dict,
    n_trials=900,
    time_cycle_ranges=((1, 50), (0, 50), (0, 100)),
    mn1_range=(2, 100),
    mn2_range=(2, 100),
    entry_range=(30.0, 60.0),
    exit_range=(40.0, 70.0),
    ticker="AAPL",
    objective_type="percent_gain",
    timeframes=['hourly', 'daily']
)

# Run optimization
optimizer.start()
optimizer.wait()
```

### Example 2: Paper Trading

```python
from trading import AlpacaLiveTrader
from config import AlpacaConfig

config = AlpacaConfig()

trader = AlpacaLiveTrader(
    api_key=config.API_KEY,
    secret_key=config.SECRET_KEY,
    base_url=config.BASE_URL,
    symbol="AAPL",
    params=best_params,
    df_dict=df_dict,
    timeframes=['hourly'],
    position_size_pct=0.05
)

trader.start()
```

## ⚠️ Important Notes

### Data Limitations

- **5-minute data**: Limited to ~60 days (Yahoo Finance restriction)
- **Hourly data**: Limited to ~730 days
- **Daily data**: Up to 10 years available

### Performance

- **Optimization time**: 5-30 minutes depending on trials and timeframes
- **Memory usage**: ~500MB-2GB depending on data size
- **Recommended**: 8GB RAM minimum

### Risk Warning

⚠️ **This software is for educational purposes only.**
- Always test strategies thoroughly before live trading
- Use paper trading first to validate strategies
- Past performance does not guarantee future results
- Never risk money you cannot afford to lose

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Issue: "Alpaca not installed"
**Solution**: `pip install alpaca-py`

### Issue: "No data loaded"
**Solution**: Check ticker symbol format (e.g., `BTC-USD` for crypto)

### Issue: "API key not found"
**Solution**: Set environment variables for Alpaca credentials

### Issue: "Out of memory during optimization"
**Solution**: Reduce number of trials or use fewer timeframes

## 📚 Additional Resources

- [Alpaca Documentation](https://alpaca.markets/docs/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Optuna Documentation](https://optuna.org/)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Contact: support@example.com

---

**Happy Trading! 📈**
