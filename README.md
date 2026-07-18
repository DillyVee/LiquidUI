# LiquidUI - Multi-Timeframe Trading Optimizer

**Desktop application for optimizing, validating, and paper-trading indicator + time-cycle strategies across multiple timeframes.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Features

- **Data loading** - Pull daily, hourly, and 5-minute bars from Yahoo Finance for stocks and crypto, with automatic data-quality checks (bad prices, duplicate timestamps, extreme moves); buy & hold benchmark stats appear the moment data loads
- **Selectable optimization goals** - Optuna-driven parameter search maximizing your choice of Sharpe, Sortino, Calmar, total return, profit factor, win rate, or expectancy - with batched parallel trials, incremental CSV saves, and live progress in the GUI
- **Ten-indicator library** - RSI, Stochastic, MACD, ROC, EMA cross, Bollinger z-score, CCI, TRIX, DPO, and Aroon behind one causal interface; pick which indicators the optimizer may use, optionally AND-combining two per timeframe
- **Time-cycle-only mode** - One toggle optimizes the calendar ON/OFF/START cycle alone, with no indicators
- **Auto-build strategy book** - Run one optimization per selected goal, auto-save each result to the per-ticker strategy book, then backtest the regime-switching meta-strategy across them
- **Anti-overfit gates** - Deflated Sharpe Ratio and CSCV probability-of-backtest-overfitting checks run on every optimization result
- **Batch mode** - Load a CSV of tickers and optimize them sequentially, unattended
- **Results dashboard** - Equity curve vs. buy & hold, drawdown comparison, and a full institutional report (CAGR, Calmar, Ulcer index, VaR/CVaR, win rate, expectancy, payoff ratio, ...)
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
│   ├── optimizer.py       # Multi-timeframe optimizer (Optuna)
│   ├── objectives.py      # Optimization goal registry
│   ├── metrics.py         # Institutional performance metrics
│   ├── validation.py      # Anti-overfit gates (DSR, CSCV PBO)
│   ├── psr_composite.py   # PSR machinery used by the DSR gate
│   ├── monte_carlo.py     # Monte Carlo simulation
│   └── walk_forward.py    # Walk-forward analysis
├── signals/
│   ├── indicators.py      # Causal 10-indicator library
│   └── engine.py          # Combo signal engine (shared backtest/live)
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

1. **Data & Setup** - Enter a ticker (e.g. `SPY`, `AAPL`, `BTC-USD`) and load data; the buy & hold benchmark appears immediately. Select the timeframes to optimize
2. **Strategy Optimization** - Pick an optimization goal (Sharpe, Sortino, Calmar, ...), choose which indicators the search may use (or toggle time-cycles-only), set ranges and trials, then press START; results and equity curves update live. Use **🤖 Auto-Build Strategy Book** to optimize several goals back-to-back, save them to the strategy book, and run the regime-switching backtest. **⭐ Apply Certified Preset** configures the evidence-backed strategy (below) in one click: calendar cycle disabled, all indicators, Sharpe objective, volatility targeting on
3. **Settings** - Configure position sizing and transaction costs (stock/crypto presets available)

Optimization results are also written to `data_output/<TICKER>_results_<objective>.csv`.

### Strategy

The strategy combines a normalized indicator oscillator (or two, AND-combined) with an on/off calendar time cycle, evaluated on every selected timeframe:

- **Entry**: every indicator leg below its entry threshold on all timeframes while the cycle is ON
- **Exit**: any leg above its exit threshold on any timeframe, or the cycle turning OFF
- **Cycle-only mode**: no indicator legs - entries/exits come from the cycle alone

The optimizer searches the indicator choice, its periods (P1/P2), entry/exit thresholds, and cycle parameters (ON/OFF/START) per timeframe. TradingView ports of the regime indicator and strategy are included as `.pine` files.

> **⚠️ Research note on the time cycle**: a controlled 15-fold experiment on real market data (SP500/NASDAQ/BTC, see `RESEARCH_JOURNAL.md`, experiment H9) found that the free-form calendar cycle *inflates in-sample scores and significantly reduces out-of-sample Sharpe* (paired −0.56, p = 0.028) versus running the same indicator search with the cycle disabled. Treat cycle-bearing results as presumptively overfit unless they pass the anti-overfit gates **and** walk-forward validation on your specific ticker.

### Evidence-backed configuration

A 60-fold out-of-sample study across 8 markets and 7 decades (`RESEARCH_JOURNAL.md`, experiments H8–H13) certified one configuration as viable — a **risk-managed long exposure** with roughly half the drawdowns of buy & hold at statistically non-inferior returns (pooled OOS Sharpe ≈ +0.5, t p = 0.003 / Wilcoxon p = 0.010; drawdown shallower in 58/60 folds, p < 1e-8).

**In the GUI**: press **⭐ Apply Certified Preset** on the Strategy Optimization tab, load an equity or crypto ticker, and press START. The same configuration in code:

```python
MultiTimeframeOptimizer(
    df_dict=data,                                    # equities or crypto only
    time_cycle_ranges=((250, 250), (0, 0), (0, 0)),  # calendar cycle DISABLED
    objective="sharpe",
    vol_targeting=True,                              # proven drawdown overlay
    # ... standard ranges, costs preset for your asset class
)
```

Keep the anti-overfit gates on, re-optimize on ~3 years of daily history, and paper-trade through the live gate stack before any real capital. Full definition, measured profile, and limitations: `RESEARCH_JOURNAL.md`, "THE CERTIFIED STRATEGY".

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
