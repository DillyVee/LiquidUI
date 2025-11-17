# LiquidUI Examples

This directory contains example scripts demonstrating how to use the LiquidUI quantitative trading pipeline.

## 📚 Examples

### 1. Basic Backtesting (`01_basic_backtest.py`)

**What it demonstrates:**
- Downloading market data with yfinance
- Computing technical indicators (RSI, ATR)
- Running a simple mean reversion strategy
- Analyzing performance metrics

**Run:**
```bash
python examples/01_basic_backtest.py
```

**Expected output:**
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Equity curve plot (if matplotlib is installed)

---

### 2. Walk-Forward Validation (`02_walk_forward_validation.py`)

**What it demonstrates:**
- Robust testing methodology
- Rolling train/test windows
- Parameter optimization
- Out-of-sample validation
- Overfitting detection

**Run:**
```bash
python examples/02_walk_forward_validation.py
```

**Expected output:**
- Training vs testing performance
- Degradation metrics
- Best parameters per fold

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: for plotting
pip install matplotlib seaborn
```

### Running Examples

```bash
# From project root
cd /path/to/LiquidUI

# Run example 1
python examples/01_basic_backtest.py

# Run example 2
python examples/02_walk_forward_validation.py
```

## 📖 Learning Path

We recommend running the examples in order:

1. **01_basic_backtest.py** - Learn the basics
2. **02_walk_forward_validation.py** - Learn robust testing

## 💡 Tips

### Modifying Examples

All examples are designed to be easily modified:

```python
# Change the symbol
symbol = "GOOGL"  # Instead of "AAPL"

# Change the date range
df = yf.download(symbol, start="2022-01-01", end="2024-12-31")

# Adjust strategy parameters
rsi_oversold = 25  # Instead of 30
rsi_overbought = 75  # Instead of 70
```

### Adding Your Own Strategy

```python
def my_custom_strategy(bar, portfolio, engine):
    """Your custom trading logic"""

    # Example: Moving average crossover
    if bar['sma_fast'] > bar['sma_slow']:
        # Golden cross - buy signal
        engine.submit_order(symbol, OrderSide.BUY, 100)
    elif bar['sma_fast'] < bar['sma_slow']:
        # Death cross - sell signal
        if symbol in portfolio.positions:
            qty = portfolio.positions[symbol].quantity
            engine.submit_order(symbol, OrderSide.SELL, qty)
```

## 🎯 What to Try Next

After running these examples, explore:

1. **Risk Management** - Add position sizing and stop losses
2. **Multiple Assets** - Test strategies across portfolios
3. **Transaction Costs** - Experiment with different cost models
4. **Advanced Features** - Try MACD, Bollinger Bands, etc.
5. **Optimization** - Use Optuna for hyperparameter tuning
6. **Live Trading** - Deploy to paper trading (see main README)

## 📊 Expected Results

### Example 1 - Basic Backtest

Typical results for AAPL RSI strategy (2020-2024):
- **Sharpe Ratio**: 1.5 - 2.0
- **Max Drawdown**: -10% to -15%
- **Win Rate**: 55% - 60%

*Note: Actual results will vary based on market conditions and exact date range.*

### Example 2 - Walk-Forward

- **Degradation**: < 20% indicates robust strategy
- **Windows**: ~40-50 test windows (depending on parameters)

## ⚠️ Important Notes

### Data Requirements

Examples use yfinance for data download. Note:
- Free API with rate limits
- May have occasional downtime
- For production, use paid data providers

### Computation Time

- **Example 1**: ~1-2 minutes
- **Example 2**: ~5-10 minutes (due to walk-forward optimization)

### Risk Disclaimer

**These examples are for educational purposes only.**

- Not financial advice
- Past performance doesn't guarantee future results
- Always test on paper accounts before using real money
- Understand the risks involved in trading

## 🐛 Troubleshooting

### Common Issues

**"Module not found" error:**
```bash
# Make sure you're in the project root
cd /path/to/LiquidUI

# Install dependencies
pip install -r requirements.txt
```

**"No data downloaded":**
```bash
# Check internet connection
# Try a different symbol or date range
# yfinance occasionally has issues, try again later
```

**"Matplotlib not found":**
```bash
# Install optional plotting dependencies
pip install matplotlib seaborn
```

## 📚 Additional Resources

- **Main README**: `../README.md` - Full project documentation
- **API Reference**: See docstrings in source files
- **Contributing**: `../CONTRIBUTING.md` - How to contribute

## 🤝 Questions?

- Open an issue on GitHub
- Check the main documentation
- Review the inline code comments

---

Happy backtesting! 📈
