# Transaction Costs Guide

## 📊 What Are Transaction Costs?

Transaction costs are the **real-world expenses** of executing trades. Ignoring them leads to **unrealistic backtests** that will fail in live trading.

### Components of Transaction Costs

1. **Commission** - Fee charged by broker per trade
2. **Slippage** - Difference between expected and actual execution price
3. **Spread** - Difference between bid and ask price

## 🎯 Why This Matters

### Example: The Reality Check

**Strategy without costs:**
- 100 trades @ +2% average = +200% gain ✨

**Same strategy WITH 0.2% costs per trade:**
- 100 trades @ +2% - 0.4% (round-trip) = +160% gain
- **20% less profit!** ⚠️

### High-Frequency Trap

**1000 trades per year:**
- No costs: Looks amazing! 📈
- With costs: Loses money 📉
- **Transaction costs destroy high-frequency strategies**

## ⚙️ How to Use in the App

### 1. **Open the Transaction Costs Panel**

Look for the **💰 Transaction Costs** section in the GUI.

### 2. **Choose a Preset** (Recommended)

Click the preset button that matches your asset:

- **Stocks (0.06%)** - For US stocks with commission-free brokers
  - Commission: 0%
  - Slippage: 0.05%
  - Spread: 0.01%

- **Crypto (0.35%)** - For cryptocurrency exchanges
  - Commission: 0.1% (typical taker fee)
  - Slippage: 0.2% (higher volatility)
  - Spread: 0.05%

- **Zero** - No costs (unrealistic, for comparison only)

### 3. **Or Customize Manually**

Adjust each component:

```
Commission: 0.100%  (your broker's fee)
Slippage:   0.050%  (depends on liquidity)
Spread:     0.010%  (check order book)
```

### 4. **Check the Summary**

The app shows:
- **Total per round-trip** - Cost of buy + sell
- **100 trades drag** - Performance impact over 100 trades

Example:
```
Total: 0.120% per round-trip | 100 trades: 10.7% drag
```

## 📈 Realistic Cost Examples

### Commission-Free Stocks (Robinhood, Webull)
```python
Commission: 0.000%
Slippage:   0.050%
Spread:     0.010%
Total:      0.060% per trade (0.12% round-trip)
```

### Interactive Brokers (Stocks)
```python
Commission: 0.0035% ($0.35 per 100 shares, ~$10k trade)
Slippage:   0.050%
Spread:     0.010%
Total:      0.0635% per trade
```

### Coinbase/Binance (Crypto)
```python
Commission: 0.100% (taker fee)
Slippage:   0.200% (volatile, wide spreads)
Spread:     0.050%
Total:      0.350% per trade (0.70% round-trip)
```

### Forex (Major Pairs)
```python
Commission: 0.000%
Slippage:   0.010%
Spread:     0.020% (2 pips on EUR/USD)
Total:      0.030% per trade
```

## 💡 Best Practices

### 1. **Always Include Costs**
Never optimize without transaction costs - you'll get false results.

### 2. **Be Conservative**
It's better to **overestimate** costs than underestimate:
- Use higher slippage for illiquid stocks
- Add 50% buffer for market impact on large orders

### 3. **Optimize for Net Returns**
The strategy that looks best **before costs** is often different from the one that performs best **after costs**.

### 4. **Check Trade Frequency**
```
High frequency (500+ trades/year) = Need very low costs
Medium frequency (100-200 trades/year) = Moderate costs OK
Low frequency (10-50 trades/year) = Costs less critical
```

## 📊 Impact Analysis

### How Costs Affect Strategy Performance

| Strategy Type | Trades/Year | Cost per Trade | Annual Drag |
|--------------|-------------|----------------|-------------|
| Day Trading | 1000 | 0.10% | -20% |
| Swing Trading | 200 | 0.10% | -4% |
| Position Trading | 50 | 0.10% | -1% |

**Key Insight**: High-frequency strategies need **very low costs** to be profitable.

### Real Example

**Strategy: Mean Reversion on SPY**

Without costs:
- 300 trades/year
- Average gain: +0.8% per trade
- Annual return: +240% 🎉

With realistic costs (0.06% per trade):
- 300 trades/year  
- Average gain: +0.8% - 0.12% (round-trip) = +0.68%
- Annual return: +204%
- **15% less profit!**

## 🔧 Configuration Code

### Setting Costs Programmatically

```python
from config.settings import TransactionCosts

# Custom costs
custom_costs = TransactionCosts(
    COMMISSION_PCT=0.001,   # 0.1%
    SLIPPAGE_PCT=0.0005,    # 0.05%
    SPREAD_PCT=0.0001       # 0.01%
)

# Use presets
stock_costs = TransactionCosts.for_stocks()
crypto_costs = TransactionCosts.for_crypto()
forex_costs = TransactionCosts.for_forex()

# Pass to optimizer
optimizer = MultiTimeframeOptimizer(
    # ... other params ...
    transaction_costs=stock_costs
)
```

## ⚠️ Common Mistakes

### Mistake 1: Forgetting Costs
```python
# ❌ BAD - Unrealistic
optimizer = MultiTimeframeOptimizer(...)  # Uses default tiny costs

# ✅ GOOD - Realistic
optimizer = MultiTimeframeOptimizer(
    ...,
    transaction_costs=TransactionCosts.for_crypto()
)
```

### Mistake 2: Underestimating Slippage
```python
# ❌ BAD - Too optimistic
slippage = 0.01%  # You won't get this in real market

# ✅ GOOD - Conservative
slippage = 0.05%  # Realistic for liquid stocks
slippage = 0.20%  # Realistic for crypto
```

### Mistake 3: Using Same Costs for All Assets
```python
# ❌ BAD - One size fits all
costs = 0.1%  # Same for stocks and crypto

# ✅ GOOD - Asset-specific
stock_costs = TransactionCosts.for_stocks()    # 0.06%
crypto_costs = TransactionCosts.for_crypto()   # 0.35%
```

## 📉 When Costs Kill Strategies

### Red Flags

Your strategy might not be viable if:

1. **Average trade profit < 3x costs**
   - If costs = 0.1%, need >0.3% average profit

2. **Win rate < 60% with small wins**
   - Costs eat into small gains quickly

3. **Many trades per day**
   - 10 trades/day = 2500/year = costs add up!

4. **Low Sharpe ratio before costs**
   - If barely profitable without costs, won't survive with them

## 🎓 Advanced: Modeling Market Impact

For **large positions** (>$100k):

```python
# Add size-based slippage
position_size_usd = 500000  # $500k position
avg_daily_volume = 10000000  # $10M daily volume

# Rule of thumb: 0.1% slippage per 1% of daily volume
size_ratio = position_size_usd / avg_daily_volume
additional_slippage = size_ratio * 0.1  # 0.5% extra slippage

total_slippage = base_slippage + additional_slippage
```

## 📊 Testing Impact

### A/B Test Your Strategy

1. Run optimization **without costs** (set to zero)
2. Run optimization **with realistic costs**
3. Compare results:

```
No costs:      +150% return, 500 trades
With costs:    +85% return, 250 trades
Difference:    43% less profit!
```

If performance drops >30%, your strategy is **too sensitive to costs**.

## ✅ Checklist

Before going live:

- [ ] Transaction costs configured for your asset type
- [ ] Used conservative estimates (not optimistic)
- [ ] Tested strategy with costs enabled
- [ ] Verified trade frequency is reasonable
- [ ] Compared with/without costs
- [ ] Performance still good after costs
- [ ] Trade log shows per-trade costs
- [ ] Understand annual cost drag

## 🎯 Summary

**Key Takeaways:**

1. ✅ Always include transaction costs in backtests
2. ✅ Use realistic (conservative) estimates
3. ✅ Different assets have different costs
4. ✅ High-frequency strategies need very low costs
5. ✅ Test strategy with and without costs
6. ✅ If profit drops >30% with costs, reconsider strategy

**Remember:** A strategy that works in backtest without costs but fails with costs **will fail in live trading**. Better to discover this now than with real money!

---

**Questions?** See README.md or open an issue on GitHub.
