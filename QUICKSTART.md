# LiquidUI Quick Start Guide

A simple guide to get you started with institutional-grade quantitative trading.

---

## 📋 Table of Contents
1. [Initial Setup](#initial-setup)
2. [First Time Launch](#first-time-launch)
3. [Loading Market Data](#loading-market-data)
4. [Running Your First Backtest](#running-your-first-backtest)
5. [Strategy Optimization](#strategy-optimization)
6. [Walk-Forward Analysis](#walk-forward-analysis)
7. [Monte Carlo Simulation](#monte-carlo-simulation)
8. [Risk Management](#risk-management)
9. [Paper Trading (Optional)](#paper-trading)

---

## 🚀 Initial Setup

### Step 1: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 2: Launch the Application
```bash
# Run the GUI application
python main.py
```

**What you'll see:** A modern trading application window with multiple sections.

---

## 🎯 First Time Launch

When you first open the application, you'll see several sections:

### Main Window Layout
```
┌─────────────────────────────────────────────────────┐
│ 📊 Data Manager                                     │
│ ├─ Symbol: [AAPL] [Load Data]                      │
│ └─ Date Range: [Start] to [End]                    │
├─────────────────────────────────────────────────────┤
│ ⚙️  Strategy Configuration                          │
│ ├─ Fast MA: 10    Slow MA: 30                      │
│ ├─ RSI: 14        MACD: 12/26/9                    │
│ └─ Timeframes: [x] 1D  [ ] 1H  [ ] 15m             │
├─────────────────────────────────────────────────────┤
│ 🎯 Backtest & Optimization                          │
│ ├─ [Run Backtest]                                  │
│ ├─ [Optimize Strategy]                             │
│ └─ [Walk-Forward Analysis]                         │
├─────────────────────────────────────────────────────┤
│ 📈 Results & Charts                                 │
│ └─ (Equity curves, performance metrics)             │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Loading Market Data

### Step 1: Choose a Symbol
1. **Find the Symbol input box** (top section)
2. **Enter a ticker symbol** (e.g., `AAPL`, `MSFT`, `TSLA`, `SPY`)

### Step 2: Set Date Range
1. **Start Date:** When your backtest begins (e.g., `2020-01-01`)
2. **End Date:** When your backtest ends (e.g., `2024-01-01`)

💡 **Tip:** Start with 2-3 years of data for meaningful results

### Step 3: Load the Data
1. **Click "Load Data"** button
2. **Wait for download** (shows progress bar)
3. **Verify:** You should see a success message

**What happens:** The app downloads historical price data from Yahoo Finance (OHLCV - Open, High, Low, Close, Volume).

---

## 🎮 Running Your First Backtest

### Step 1: Configure Your Strategy

The default strategy is a **Moving Average Crossover**:

**Parameters:**
- **Fast MA Period:** 10 (short-term moving average)
- **Slow MA Period:** 30 (long-term moving average)

**Trading Logic:**
- **Buy Signal:** When Fast MA crosses ABOVE Slow MA
- **Sell Signal:** When Fast MA crosses BELOW Slow MA

### Step 2: Select Timeframe
- ✅ Check `1D` (daily data) - **recommended for beginners**
- You can also try: `1H` (hourly), `15m` (15-minute)

### Step 3: Run Backtest
1. **Click "Run Backtest"** button
2. **Wait for completion** (usually 5-30 seconds)

### Step 4: View Results

**Performance Metrics:**
```
Total Return:        15.3%
Sharpe Ratio:        1.42
Max Drawdown:        -8.2%
Win Rate:            54%
Total Trades:        23
```

**What these mean:**
- **Total Return:** How much money you made/lost (%)
- **Sharpe Ratio:** Risk-adjusted returns (>1.0 is good, >2.0 is excellent)
- **Max Drawdown:** Worst peak-to-trough loss (smaller is better)
- **Win Rate:** Percentage of profitable trades
- **Total Trades:** Number of buy/sell cycles

**Charts:**
- **Equity Curve:** Your account balance over time
- **Drawdown Chart:** Shows when you lost money
- **Trade Markers:** Buy/sell points on price chart

---

## ⚙️ Strategy Optimization

Instead of manually guessing parameters, let the optimizer find the best ones!

### Step 1: Choose What to Optimize

**Common Parameters:**
- Fast MA Period: 5 to 20
- Slow MA Period: 20 to 50
- RSI Period: 10 to 20
- RSI Oversold: 20 to 40
- RSI Overbought: 60 to 80

### Step 2: Set Optimization Range
1. **Min Value:** Minimum parameter value to test
2. **Max Value:** Maximum parameter value to test
3. **Step:** Increment between tests

Example:
```
Fast MA:  Min=5,  Max=20,  Step=1  → Tests 16 values
Slow MA:  Min=20, Max=50,  Step=5  → Tests 7 values
Total combinations: 16 × 7 = 112 backtests
```

### Step 3: Choose Optimization Metric
- **Sharpe Ratio** (best for risk-adjusted returns) ⭐ **Recommended**
- **Total Return** (maximize profit)
- **Win Rate** (maximize winning trades)
- **Max Drawdown** (minimize losses)

### Step 4: Run Optimization
1. **Click "Optimize Strategy"** button
2. **Wait for completion** (can take 1-5 minutes)
3. **Review best parameters**

**Output:**
```
Best Parameters Found:
├─ Fast MA: 8
├─ Slow MA: 35
├─ Sharpe Ratio: 1.89
└─ Total Return: 24.7%
```

💡 **Tip:** The optimizer automatically uses these best parameters for future backtests.

---

## 🔬 Walk-Forward Analysis

**Purpose:** Detect overfitting and validate strategy robustness.

### What is Walk-Forward?
Think of it like a **time machine test**:
1. Train on past data (e.g., 2020)
2. Test on future unseen data (e.g., 2021)
3. Repeat rolling forward through time

### How to Use

**Step 1: Configure Walk-Forward**
```
Training Period: 252 days (1 year)
Testing Period:  63 days  (3 months)
```

**Step 2: Run Analysis**
1. **Click "Walk-Forward Analysis"** button
2. **Wait for completion** (5-10 minutes)

**Step 3: Interpret Results**

**Good Signs (Strategy is Robust):**
- ✅ In-sample and out-of-sample returns are similar
- ✅ Out-of-sample Sharpe > 1.0
- ✅ Consistency across windows

**Warning Signs (Overfitting):**
- ❌ In-sample: +50%, Out-of-sample: -10%
- ❌ Huge variance between windows
- ❌ Out-of-sample Sharpe < 0.5

**Example Output:**
```
Walk-Forward Results:
├─ In-Sample Sharpe:      1.85
├─ Out-of-Sample Sharpe:  1.42  ← Should be close to in-sample
├─ Efficiency:            76%   ← >70% is good
└─ Consistency:           Good  ✅
```

---

## 🎲 Monte Carlo Simulation

**Purpose:** Stress-test your strategy with randomness.

### What is Monte Carlo?
Simulates **1,000+ alternate realities** by shuffling your trade returns.

**Questions it answers:**
- What's my **best-case** scenario?
- What's my **worst-case** scenario?
- How likely am I to lose money?

### How to Use

**Step 1: Configure Simulation**
```
Number of Simulations: 1000
Confidence Level: 95%
```

**Step 2: Run Simulation**
1. **Click "Monte Carlo Simulation"** button
2. **Wait for completion** (30-60 seconds)

**Step 3: Interpret Results**

**Output:**
```
Monte Carlo Analysis:
├─ Median Return:         18.2%
├─ 95% Confidence Range:  [8.4%, 31.7%]
├─ Probability of Loss:   12%
├─ Best Case (95th %ile): 31.7%
└─ Worst Case (5th %ile): 8.4%
```

**Chart Shows:**
- **Fan of equity curves** (each line = one simulation)
- **Confidence bands** (95% of outcomes fall within)
- **Median path** (most likely outcome)

💡 **Interpretation:**
- **Low Probability of Loss:** Good (< 20%)
- **Tight Confidence Bands:** More predictable
- **Wide Bands:** Higher uncertainty/risk

---

## 🛡️ Risk Management

### Built-in Risk Controls

**1. Position Sizing**
- **Max Position Size:** $100,000 per trade
- **Max Portfolio Exposure:** 100% of capital

**2. Drawdown Limits**
- **Max Daily Loss:** -$10,000 → Stops trading for the day
- **Max Drawdown:** -10% → Kill switch activated

**3. Risk Metrics Monitoring**
```
Current Status:
├─ Daily P&L:        +$1,234  ✅
├─ Drawdown:         -2.3%    ✅
├─ VaR (95%):        $8,500
└─ Position Count:   3/10     ✅
```

### How to Configure

**In the Risk Settings Panel:**
1. Set your **initial capital** (e.g., $100,000)
2. Set **max drawdown %** (e.g., 10%)
3. Set **max daily loss** (e.g., $5,000)
4. Enable **kill switch** for automatic shutdown

---

## 💰 Paper Trading (Optional)

**Note:** Requires Alpaca API account (free).

### Step 1: Get Alpaca API Keys
1. Sign up at [alpaca.markets](https://alpaca.markets) (free)
2. Get **Paper Trading** API keys (not real money!)
3. Copy API Key and Secret Key

### Step 2: Configure Environment
```bash
# Create .env file
cp .env.example .env

# Edit .env and add:
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Step 3: Enable Paper Trading
1. **Go to Live Trading tab** in GUI
2. **Click "Connect to Alpaca"**
3. **Verify connection** (shows account balance)

### Step 4: Start Trading
1. **Load your optimized strategy**
2. **Click "Start Paper Trading"**
3. **Monitor live trades** in real-time

**What happens:**
- App monitors market in real-time
- Generates signals based on your strategy
- Automatically executes trades via Alpaca
- Tracks performance live

⚠️ **Safety:** Only use paper trading (simulated money) until you're confident!

---

## 🎓 Learning Path

### Beginner (Week 1)
1. ✅ Load data for `SPY` (S&P 500 ETF)
2. ✅ Run backtest with default settings
3. ✅ Understand performance metrics
4. ✅ Try different symbols (`AAPL`, `MSFT`, etc.)

### Intermediate (Week 2-3)
1. ✅ Run optimizer to find best parameters
2. ✅ Try different strategies (RSI, MACD)
3. ✅ Analyze equity curves and drawdowns
4. ✅ Run Monte Carlo simulations

### Advanced (Week 4+)
1. ✅ Perform walk-forward analysis
2. ✅ Test multi-timeframe strategies
3. ✅ Implement custom risk rules
4. ✅ Paper trade your strategy

---

## 💡 Pro Tips

### 1. Start Simple
- Use **1 symbol** (SPY)
- Use **1 timeframe** (daily)
- Use **default parameters**
- **Then** increase complexity

### 2. Avoid Overfitting
- Always run **walk-forward analysis**
- Don't over-optimize (>1000 combinations)
- Test on **multiple symbols**
- Use **out-of-sample testing**

### 3. Understand Your Metrics
- **Sharpe > 1.5:** Good strategy
- **Max Drawdown < -15%:** Acceptable risk
- **Win Rate 50-60%:** Normal for trend-following
- **Trades < 10/year:** May be under-trading

### 4. Risk Management First
- Never risk more than **2% per trade**
- Always use **stop losses**
- Set **maximum drawdown limits**
- Start with **paper trading**

### 5. Continuous Learning
- Track **why** trades win/lose
- Keep a **trading journal**
- Test during **different market conditions**
- Read backtest reports carefully

---

## 🆘 Common Issues & Solutions

### Issue: "No data loaded"
**Solution:** Click "Load Data" button and wait for download to complete.

### Issue: "Not enough data for backtest"
**Solution:** Increase date range to at least 1 year of data.

### Issue: "Optimization taking too long"
**Solution:** Reduce parameter ranges or increase step size.

### Issue: "All trades are losses"
**Solution:** Your strategy may not fit this market. Try different parameters or symbols.

### Issue: "Can't see charts"
**Solution:** Make sure you're running on a system with GUI support (not headless server).

---

## 📚 Next Steps

### Explore Advanced Features
1. **Custom Indicators:** Add your own technical indicators
2. **Multi-Strategy:** Combine multiple strategies
3. **Portfolio Backtesting:** Test multiple symbols together
4. **Machine Learning:** Integrate ML models (see `models/` folder)

### Read Documentation
- `README.md` - Project overview
- `CONTRIBUTING.md` - How to extend the code
- `SECURITY.md` - Security best practices
- `examples/` folder - Sample strategies

### Join the Community
- Report bugs on GitHub Issues
- Share your strategies (without giving away secrets!)
- Contribute improvements

---

## 🎯 Quick Reference

### Essential Keyboard Shortcuts
- `Ctrl+L` - Load Data
- `Ctrl+B` - Run Backtest
- `Ctrl+O` - Optimize
- `Ctrl+W` - Walk-Forward
- `Ctrl+M` - Monte Carlo

### Best Practices Checklist
- ✅ Always backtest on **multiple symbols**
- ✅ Always run **walk-forward analysis**
- ✅ Always check **Monte Carlo** worst-case
- ✅ Always set **risk limits**
- ✅ Always **paper trade** before live
- ✅ Always keep **trading journals**

---

## 🚀 You're Ready!

Start with the **Beginner path**, work through each feature systematically, and **most importantly**: understand WHY your strategy makes money before risking real capital.

**Happy Trading!** 📈

---

*Last Updated: 2025-11-17*
*Version: 1.0.0*
