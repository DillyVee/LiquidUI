# LiquidUI - Multi-Timeframe Trading Optimizer

**Desktop application for optimizing, validating, and paper-trading an RSI + time-cycle strategy across multiple timeframes.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Features

- **Data loading** - Pull daily, hourly, and 5-minute bars from Yahoo Finance for stocks and crypto, with automatic data-quality checks (bad prices, duplicate timestamps, extreme moves)
- **PSR optimization** - Optuna-driven parameter search that maximizes the Probabilistic Sharpe Ratio, with batched parallel trials, incremental CSV saves, and live progress in the GUI
- **Batch mode** - Load a CSV of tickers and optimize them sequentially, unattended
- **Results dashboard** - Equity curve vs. buy & hold, drawdown comparison, and a full PSR report
- **Transaction cost modeling** - Commission, slippage, and spread settings with stock/crypto presets, applied inside every backtest
- **Monte Carlo simulation** - Trade-order randomization with advanced risk metrics *(hidden by default)*
- **Walk-forward analysis** - Out-of-sample validation with overfitting detection *(hidden by default)*
- **Regime analysis** - Market regime detection, ML regime prediction, probability calibration, and robustness tests *(hidden by default)*
- **Live trading** - Alpaca paper/live trading driven by the optimized parameters *(hidden by default)*

Hidden features can be re-enabled with the feature flags at the top of `gui/main_window.py`.

---

## Project Structure

```
LiquidUI/
├── main.py                # Application entry point
├── gui/                   # PyQt6 interface
│   ├── main_window.py     # Main tabbed window
│   └── styles.py          # Dark theme stylesheet
├── config/
│   └── settings.py        # Paths, defaults, parameter ranges
├── data/
│   └── loader.py          # Yahoo Finance data loading & cleaning
├── optimization/
│   ├── optimizer.py       # Multi-timeframe PSR optimizer (Optuna)
│   ├── metrics.py         # Performance metrics
│   ├── psr_composite.py   # Probabilistic Sharpe Ratio calculator
│   ├── monte_carlo.py     # Monte Carlo simulation
│   └── walk_forward.py    # Walk-forward analysis
├── models/                # Market regime detection & prediction
├── trading/
│   └── alpaca_trader.py   # Alpaca live/paper trading loop
├── tests/                 # Test suite
└── pinescript_*.pine      # TradingView ports of the strategy
```

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/DillyVee/LiquidUI.git
cd LiquidUI

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. **Data & Setup** - Enter a ticker (e.g. `SPY`, `AAPL`, `BTC-USD`) and load data, then select the timeframes to optimize
2. **Strategy Optimization** - Set parameter ranges and trial count, then run PSR optimization; results and equity curves update live
3. **Settings** - Configure position sizing and transaction costs (stock/crypto presets available)

Optimization results are also written to `data_output/<TICKER>_results_psr_batched.csv`.

### Strategy

The strategy combines a smoothed RSI with an on/off time cycle, evaluated on every selected timeframe:

- **Entry**: smoothed RSI below the entry threshold on all timeframes while the cycle is ON
- **Exit**: smoothed RSI above the exit threshold on any timeframe, or the cycle turning OFF

The optimizer searches RSI lengths (MN1/MN2), entry/exit thresholds, and cycle parameters (ON/OFF/START) per timeframe. TradingView ports of the regime indicator and strategy are included as `.pine` files.

---

## Testing

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

---

## License

MIT - see [LICENSE](LICENSE).

---

**⚠️ Risk Warning**: Trading involves substantial risk of loss. This software is provided for research and educational purposes. Always test thoroughly on paper accounts before deploying real capital.
