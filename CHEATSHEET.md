# LiquidUI Cheat Sheet

Quick reference for common tasks.

---

## 🚀 Getting Started (5 Minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python main.py

# 3. In the GUI:
#    - Enter symbol: AAPL
#    - Set dates: 2020-01-01 to 2024-01-01
#    - Click "Load Data"
#    - Click "Run Backtest"
```

---

## 📊 Performance Metrics Quick Guide

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **Sharpe Ratio** | > 2.0 | 1.0 - 2.0 | < 1.0 |
| **Max Drawdown** | < -10% | -10% to -20% | > -20% |
| **Win Rate** | > 60% | 50% - 60% | < 50% |
| **Total Return** | > 20%/yr | 10% - 20%/yr | < 10%/yr |

---

## ⚙️ Strategy Parameters

### Moving Average Crossover
```python
Fast MA:  5-20   (typical: 10)
Slow MA:  20-50  (typical: 30)
```

### RSI Strategy
```python
Period:     10-20  (typical: 14)
Oversold:   20-35  (typical: 30)
Overbought: 65-80  (typical: 70)
```

### MACD Strategy
```python
Fast:   8-15   (typical: 12)
Slow:   20-30  (typical: 26)
Signal: 7-12   (typical: 9)
```

---

## 🔬 Analysis Workflow

### Basic Workflow
```
1. Load Data
2. Run Backtest
3. Check Sharpe Ratio
4. If Sharpe > 1.0 → Proceed
5. If Sharpe < 1.0 → Optimize
```

### Advanced Workflow
```
1. Load Data
2. Optimize Strategy
3. Run Walk-Forward Analysis
4. Check Out-of-Sample Performance
5. Run Monte Carlo Simulation
6. If all look good → Paper Trade
7. Monitor for 1 month
8. Review results → Go Live (carefully!)
```

---

## 🎯 Optimization Quick Tips

### Fast Optimization (< 1 min)
```
Fast MA:  [5, 15, step=5]   → 3 values
Slow MA:  [20, 40, step=10] → 3 values
Total: 3 × 3 = 9 backtests
```

### Medium Optimization (1-5 min)
```
Fast MA:  [5, 20, step=1]   → 16 values
Slow MA:  [20, 50, step=5]  → 7 values
Total: 16 × 7 = 112 backtests
```

### Deep Optimization (5-30 min)
```
Fast MA:  [5, 20, step=1]    → 16 values
Slow MA:  [20, 50, step=2]   → 16 values
RSI:      [10, 20, step=2]   → 6 values
Total: 16 × 16 × 6 = 1,536 backtests
```

⚠️ **Warning:** > 10,000 combinations = overfitting risk!

---

## 🛡️ Risk Management Presets

### Conservative
```python
Max Drawdown:    -5%
Max Daily Loss:  -$2,000
Position Size:   25% of capital
Stop Loss:       -2%
```

### Moderate (Recommended)
```python
Max Drawdown:    -10%
Max Daily Loss:  -$5,000
Position Size:   50% of capital
Stop Loss:       -3%
```

### Aggressive
```python
Max Drawdown:    -20%
Max Daily Loss:  -$10,000
Position Size:   100% of capital
Stop Loss:       -5%
```

---

## 📈 Symbol Recommendations

### Beginners (High Liquidity, Low Volatility)
- `SPY` - S&P 500 ETF
- `QQQ` - Nasdaq ETF
- `AAPL` - Apple
- `MSFT` - Microsoft

### Intermediate (Moderate Volatility)
- `TSLA` - Tesla
- `NVDA` - NVIDIA
- `AMD` - AMD
- `AMZN` - Amazon

### Advanced (High Volatility)
- `BTC-USD` - Bitcoin
- `COIN` - Coinbase
- Penny stocks (not recommended!)

---

## 🕐 Timeframe Selection

| Timeframe | Best For | Data Needed | Trades/Year |
|-----------|----------|-------------|-------------|
| **1D** (Daily) | Swing trading | 2-5 years | 10-50 |
| **1H** (Hourly) | Day trading | 6-12 months | 50-200 |
| **15m** (15-min) | Scalping | 1-3 months | 200-1000 |

💡 **Tip:** Start with daily (1D) data!

---

## 🔍 Interpreting Walk-Forward Results

### Excellent Strategy
```
In-Sample Sharpe:  1.85
Out-Sample Sharpe: 1.72  ← Close to in-sample ✅
Efficiency:        93%   ← Very high ✅
```

### Good Strategy
```
In-Sample Sharpe:  1.60
Out-Sample Sharpe: 1.35  ← Reasonable drop ✅
Efficiency:        84%   ← Good ✅
```

### Warning Signs
```
In-Sample Sharpe:  2.50
Out-Sample Sharpe: 0.45  ← Massive drop ❌
Efficiency:        18%   ← Very low ❌
```

**Efficiency Formula:**
```
Efficiency = (Out-of-Sample Sharpe / In-Sample Sharpe) × 100%

> 80%: Excellent
60-80%: Good
40-60%: Acceptable
< 40%: Overfitting!
```

---

## 🎲 Monte Carlo Interpretation

### Interpreting Confidence Intervals

```
Example Result:
├─ Median Return: 18.2%
├─ 95% CI: [8.4%, 31.7%]
└─ Prob of Loss: 12%
```

**What it means:**
- **50% chance** return is above 18.2%
- **95% chance** return is between 8.4% and 31.7%
- **12% chance** of losing money

### Risk Levels

| Prob of Loss | Risk Level | Action |
|--------------|------------|--------|
| < 10% | Very Low | Proceed confidently |
| 10-20% | Low | Acceptable |
| 20-30% | Moderate | Review strategy |
| 30-40% | High | Improve or reject |
| > 40% | Very High | Reject strategy |

---

## 💰 Paper Trading Checklist

Before enabling paper trading:

- [ ] Sharpe Ratio > 1.0
- [ ] Max Drawdown < -15%
- [ ] Walk-Forward Efficiency > 70%
- [ ] Monte Carlo Prob(Loss) < 20%
- [ ] Tested on multiple symbols
- [ ] Reviewed all trades manually
- [ ] Set risk limits
- [ ] Connected to Alpaca Paper API
- [ ] Monitoring dashboard ready

**After 1 Month:**
- [ ] Review actual vs. backtested performance
- [ ] Check if Sharpe Ratio holds
- [ ] Analyze unexpected losses
- [ ] Fine-tune parameters if needed

---

## 🐛 Troubleshooting

### Problem: Low Sharpe Ratio (< 0.5)
**Solutions:**
1. Optimize parameters
2. Try different strategy
3. Test different symbol
4. Check for transaction costs

### Problem: High Drawdown (> -20%)
**Solutions:**
1. Add stop losses
2. Reduce position size
3. Add trend filter
4. Use diversification

### Problem: Too Few Trades (< 10/year)
**Solutions:**
1. Reduce MA periods
2. Use shorter timeframe
3. Lower entry thresholds
4. Add more signals

### Problem: Too Many Trades (> 500/year)
**Solutions:**
1. Increase MA periods
2. Use longer timeframe
3. Raise entry thresholds
4. Filter by trend

---

## 📐 Common Formula Reference

### Sharpe Ratio
```
Sharpe = (Return - Risk-Free Rate) / Volatility
```
Higher is better (> 2.0 is excellent)

### Max Drawdown
```
Max DD = (Peak - Trough) / Peak
```
Smaller absolute value is better (e.g., -5% better than -20%)

### Win Rate
```
Win Rate = (Winning Trades / Total Trades) × 100%
```
50-60% is typical for good strategies

### Profit Factor
```
Profit Factor = Gross Profit / Gross Loss
```
\> 1.5 is good, > 2.0 is excellent

---

## 🎯 One-Page Strategy Evaluation

**Use this checklist before going live:**

| Criterion | Target | Your Result | ✓/✗ |
|-----------|--------|-------------|-----|
| Sharpe Ratio | > 1.0 | _____ | ☐ |
| Max Drawdown | < -15% | _____ | ☐ |
| Win Rate | > 50% | _____ | ☐ |
| Total Trades | 20-200 | _____ | ☐ |
| Walk-Forward Eff | > 70% | _____ | ☐ |
| MC Prob(Loss) | < 20% | _____ | ☐ |
| Tested Symbols | ≥ 3 | _____ | ☐ |
| Backtest Period | ≥ 2 years | _____ | ☐ |

**If 7/8 checked → Consider paper trading**
**If 8/8 checked → Strong candidate for live trading**

---

## 🔗 Quick Links

- Full Guide: `QUICKSTART.md`
- Examples: `examples/` folder
- API Docs: `README.md`
- Report Issues: GitHub Issues

---

*Keep this cheat sheet handy while trading!*
