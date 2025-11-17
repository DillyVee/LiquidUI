# LiquidUI GUI Feature Map

**Everything runs from `python main.py` - No separate modules to configure!**

---

## 🖥️ Main GUI Layout (Top to Bottom)

When you run `python main.py`, you get ONE window with ALL features:

```
┌─────────────────────────────────────────────────────────────┐
│  LiquidUI - Quantitative Trading Platform                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 DATA LOADING SECTION                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Data Source: [AAPL____________] [Load from Yahoo]  │   │
│  │             [📋 Load Ticker List]                   │   │
│  │ Date Range: 2020-01-01 to 2024-01-01              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ⏱️  TIMEFRAME SELECTION                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Active Timeframes:                                  │   │
│  │ [✓] Daily    [✓] Hourly    [ ] 5-Minute           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ⚙️  STRATEGY PARAMETERS                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Fast MA: [5 to 20]    Slow MA: [20 to 50]         │   │
│  │ RSI Period: [10 to 20]  RSI Oversold: [20 to 40]  │   │
│  │ MACD Fast: [8 to 15]   MACD Slow: [20 to 30]      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  🎯 OPTIMIZATION CONTROLS                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Total Trials: [5000]   Batch Size: [200]           │   │
│  │ PSR: 94.2%  Sharpe: 1.85  Return: 23.4%            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  🚀 ACTION BUTTONS                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [▶️ Start Optimization]  [⏸️ Stop]  [💾 Export]     │   │
│  │ [🔬 Walk-Forward Analysis]                          │   │
│  │ [🎲 Run Monte Carlo Simulation]                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  📈 LIVE TRADING (PAPER/REAL)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Alpaca API: [●] Connected                           │   │
│  │ [▶️ Start Live Trading]  [⏹️ Stop Trading]          │   │
│  │ Status: Running | P&L: +$1,234                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  🛡️  RISK MANAGEMENT                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Initial Capital: [$100,000]                         │   │
│  │ Max Drawdown: [-10%]   Max Daily Loss: [$5,000]    │   │
│  │ Position Size: [50%]   Stop Loss: [-3%]            │   │
│  │ [✓] Enable Kill Switch                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  💰 TRANSACTION COSTS                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Commission: [0.1%]   Spread: [0.05%]               │   │
│  │ Slippage: [0.02%]                                   │   │
│  │ Presets: [Stocks] [Crypto] [Zero Costs]            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  [Progress: ████████████░░░░░░░░ 65%]                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│  📊 RESULTS & CHARTS                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │        📈 Equity Curve                              │   │
│  │                                                     │   │
│  │        📉 Drawdown Chart                            │   │
│  │                                                     │   │
│  │        🎯 Trade Markers on Price Chart              │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  📋 BEST PARAMETERS                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Fast MA: 12  |  Slow MA: 38  |  RSI: 14            │   │
│  │ Sharpe: 1.89  |  Return: 24.7%  |  Drawdown: -8.2% │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ What's Included in the GUI

### 1. **Data Management** (Top Section)
- ✅ Load single ticker from Yahoo Finance
- ✅ Load multiple tickers from file
- ✅ Automatic multi-timeframe data download
- ✅ Date range display

**How to use:**
- Enter ticker → Click "Load from Yahoo Finance"
- OR click "📋 Load Ticker List" for batch processing

---

### 2. **Timeframe Selection** (Multi-Timeframe Trading)
- ✅ Daily charts
- ✅ Hourly charts
- ✅ 5-minute charts
- ✅ Can select multiple simultaneously

**How to use:**
- Check boxes for timeframes you want to trade
- Algorithm finds patterns across all selected timeframes

---

### 3. **Strategy Configuration** (Parameter Ranges)
- ✅ Moving Average periods
- ✅ RSI settings (period, overbought, oversold)
- ✅ MACD settings (fast, slow, signal)
- ✅ All parameters have min/max ranges for optimization

**How to use:**
- Set ranges for each parameter (e.g., Fast MA: 5 to 20)
- Optimizer tests all combinations within these ranges

---

### 4. **Backtesting & Optimization** (Main Engine)
- ✅ **Start Optimization** - Tests thousands of parameter combinations
- ✅ **Real-time progress** - Shows trials completed and current best
- ✅ **PSR (Probabilistic Sharpe Ratio)** - Statistical confidence metric
- ✅ **Sharpe Ratio** - Risk-adjusted returns
- ✅ **Export Results** - Save to Excel/CSV

**How to use:**
- Click "▶️ Start Optimization"
- Watch real-time updates
- Best parameters automatically saved

---

### 5. **Walk-Forward Analysis** (Overfitting Detection)
- ✅ **Automated rolling window testing**
- ✅ **In-sample vs out-of-sample comparison**
- ✅ **Efficiency metrics**
- ✅ **Visual results**

**How to use:**
- Click "🔬 Walk-Forward Analysis"
- Reviews consistency across time periods
- Shows if strategy is robust or overfit

---

### 6. **Monte Carlo Simulation** (Risk Analysis)
- ✅ **1,000+ simulations**
- ✅ **Confidence intervals (95%)**
- ✅ **Probability of loss**
- ✅ **Best/worst case scenarios**
- ✅ **Fan chart visualization**

**How to use:**
- Click "🎲 Run Monte Carlo Simulation"
- See distribution of possible outcomes
- Understand risk exposure

---

### 7. **Live/Paper Trading** (Alpaca Integration)
- ✅ **Connect to Alpaca API**
- ✅ **Paper trading (simulated)**
- ✅ **Live trading (real money)**
- ✅ **Real-time P&L tracking**
- ✅ **Auto trade execution**

**How to use:**
- Add API keys to `.env` file
- Click "Connect to Alpaca"
- Click "▶️ Start Live Trading"
- Monitor in real-time

---

### 8. **Risk Management** (Built-in Safety)
- ✅ **Initial capital setting**
- ✅ **Max drawdown limits**
- ✅ **Max daily loss limits**
- ✅ **Position sizing**
- ✅ **Stop loss percentages**
- ✅ **Kill switch** (auto-shutdown on breach)

**How to use:**
- Configure your risk limits
- System automatically enforces them
- Stops trading if limits exceeded

---

### 9. **Transaction Costs** (Realistic Modeling)
- ✅ **Commission rates**
- ✅ **Bid-ask spread**
- ✅ **Slippage**
- ✅ **Quick presets** (Stocks, Crypto, Zero)

**How to use:**
- Click preset buttons OR
- Manually enter your broker's fees
- Costs automatically included in backtest

---

### 10. **Visual Results** (Charts & Metrics)
- ✅ **Equity curve** (account balance over time)
- ✅ **Drawdown chart** (losses over time)
- ✅ **Price chart with trade markers** (buy/sell points)
- ✅ **Performance metrics** (Sharpe, returns, win rate)

**How to use:**
- Automatically updates after each optimization
- Scroll down to see all charts
- Export charts as images

---

## 🚫 What's NOT in the GUI (Standalone Modules)

These are **optional** advanced features you can run separately:

### 1. **MLflow Experiment Tracking** (`models/experiment_tracking.py`)
- **Purpose:** Advanced ML experiment logging
- **Run separately:** For data scientists who want detailed versioning
- **GUI alternative:** Results are shown in GUI, just not ML-specific tracking

### 2. **Infrastructure/Airflow** (`infrastructure/airflow/`)
- **Purpose:** Production workflow orchestration
- **Run separately:** For automated daily strategy runs
- **GUI alternative:** Manual execution via GUI buttons

### 3. **Example Scripts** (`examples/`)
- **Purpose:** Learning and testing individual components
- **Run separately:** Educational purposes
- **GUI alternative:** All functionality available in GUI

### 4. **Monitoring Dashboard** (`monitoring/metrics.py`)
- **Purpose:** Prometheus metrics for production monitoring
- **Run separately:** For ops teams running in production
- **GUI alternative:** Live trading panel shows key metrics

---

## 📝 Usage Summary

### For 99% of Users - Use ONLY the GUI:
```bash
python main.py
```

**You get:**
- ✅ Data loading
- ✅ Backtesting
- ✅ Optimization
- ✅ Walk-forward analysis
- ✅ Monte Carlo simulation
- ✅ Risk management
- ✅ Paper/live trading
- ✅ All charts and metrics

### For Advanced Users - Optional Standalone:
```bash
# Run example backtest script (learning)
python examples/01_basic_backtest.py

# Run walk-forward script (testing)
python examples/02_walk_forward_validation.py

# Start Airflow (production automation)
airflow scheduler
```

---

## 🎯 Quick Start Workflow (All in GUI)

```
1. Launch GUI
   → python main.py

2. Load Data
   → Enter "AAPL"
   → Click "Load from Yahoo Finance"

3. Configure Strategy (optional, has defaults)
   → Set parameter ranges
   → Select timeframes

4. Run Optimization
   → Click "▶️ Start Optimization"
   → Wait for completion

5. Validate Strategy
   → Click "🔬 Walk-Forward Analysis"
   → Click "🎲 Run Monte Carlo"

6. Review Results
   → Scroll through charts
   → Check metrics

7. Paper Trade (optional)
   → Add Alpaca keys to .env
   → Click "Start Live Trading"

Everything happens in ONE window!
```

---

## 💡 Pro Tips

1. **Don't run separate modules** - Everything you need is in the GUI
2. **Example scripts are for learning** - Not required for normal use
3. **MLflow/Airflow are for production** - Skip unless you're deploying at scale
4. **One ticker at a time** - Or use "Load Ticker List" for batch processing

---

**The GUI is your complete trading workstation - no assembly required!** 🚀
