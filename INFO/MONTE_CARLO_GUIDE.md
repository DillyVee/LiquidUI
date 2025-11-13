# Monte Carlo Simulation Guide

## 🎲 What is Monte Carlo Simulation?

Monte Carlo simulation tests **strategy robustness** by randomizing trade sequences thousands of times to see:

1. **Range of possible outcomes** - Best/worst case scenarios
2. **Probability of success** - How often does the strategy actually profit?
3. **Luck vs. skill** - Was your backtest result typical or lucky?
4. **Confidence intervals** - What returns can you realistically expect?

## 🎯 Why This Matters

### The Problem: Order Dependency

Your backtest shows one specific sequence of trades. But **trade order matters**!

**Example:**
```
Original sequence: +5%, -3%, +8%, -2% = +7.9% total
Random shuffle:    -3%, -2%, +5%, +8% = +7.9% total (same!)
Another shuffle:   -3%, -2%, -3%, +5% = -3.1% total (different!)
```

**Key Insight:** If results vary wildly with trade order, your strategy may not be robust.

### Real-World Scenario

**Your Backtest:**
- 100 trades
- Final return: +150%
- Looks amazing! 🎉

**Monte Carlo (1000 simulations):**
- Median return: +85%
- 95% confidence: +20% to +180%
- Probability of profit: 75%

**Reality Check:** Your backtest was in the top 20% of outcomes (lucky!). Realistic expectation is +85%, not +150%.

## 🔧 How to Use in the App

### Step 1: Complete Optimization

First, run a normal optimization:
1. Load data for a ticker
2. Select timeframes
3. Start optimization
4. Wait for completion

### Step 2: Run Monte Carlo

Look for the **🎲 Run Monte Carlo Simulation** button (appears after optimization)

```
┌────────────────────────────────────┐
│ Simulations: [1000]  🎲 Run MC     │
└────────────────────────────────────┘
```

**Settings:**
- **100 simulations** - Quick test (30 seconds)
- **1000 simulations** - Standard (2-3 minutes)
- **5000+ simulations** - High confidence (10+ minutes)

### Step 3: Interpret Results

The app shows two plots and a report:

#### Plot 1: Equity Curves
Shows 100 random paths your strategy could take:
- **Blue spaghetti** - Individual simulation paths
- **Orange band** - 5th to 95th percentile range
- **Orange line** - Median path

#### Plot 2: Distribution
Histogram of final equity values:
- **Green line** - Your original result
- **Orange line** - Median of simulations
- **Red/Green dotted** - 5th/95th percentiles

## 📊 Reading the Report

```
╔══════════════════════════════════════════════════════════╗
║          MONTE CARLO SIMULATION REPORT                   ║
╚══════════════════════════════════════════════════════════╝

📊 ORIGINAL BACKTEST:
   Final Equity:  $2,450.00
   Return:        +145.0%

📈 MONTE CARLO RESULTS (1000 simulations):
   Mean Equity:   $1,850.00 (+85.0%)
   Median Equity: $1,800.00 (+80.0%)
   
🎯 CONFIDENCE INTERVALS:
   95% Confidence: $1,200.00 to $3,100.00
   Range:          $1,900.00
   Std Dev:        $425.00
   
⚠️  RISK ASSESSMENT:
   Best Case:      $4,200.00
   Worst Case:     $850.00
   Probability of Profit: 82.5%

🎲 INTERPRETATION:
   ✅ Strategy has good probability of profit
   🎉 Original result is in TOP 5% (may be lucky!)
   ✓ Result within 2 std devs of median
```

### What Each Metric Means

**Original vs. Mean:**
- If original >> mean: You got lucky in backtest
- If original ≈ mean: Result is typical
- If original << mean: You got unlucky (or data issue)

**Probability of Profit:**
- <50%: ❌ Strategy more likely to lose
- 50-70%: ⚠️ Moderate confidence
- 70-90%: ✅ Good confidence
- >90%: ✅✅ Very robust

**95% Confidence Interval:**
- The range where 95% of outcomes fall
- Wide range = High uncertainty
- Narrow range = Consistent results

## ✅ What Makes a Good Result

### Excellent (Robust Strategy)
```
✅ Probability of profit: 85%+
✅ Original result within median ±1 std dev
✅ Narrow confidence interval
✅ Consistent equity curves
```

### Acceptable (Decent Strategy)
```
✅ Probability of profit: 65-85%
⚠️  Original result within median ±2 std dev
✅ Moderate confidence interval
✅ Most paths above break-even
```

### Warning Signs (Risky Strategy)
```
❌ Probability of profit: <60%
❌ Original result >2 std devs from median
❌ Very wide confidence interval
❌ Many paths below break-even
```

## 🚨 Common Patterns and What They Mean

### Pattern 1: Original Result is Top 5%

```
Original: $2,500
Median:   $1,200
```

**Meaning:** Your backtest got lucky! Real trading will likely be worse.

**Action:** Lower your expectations. Plan for median, not original.

### Pattern 2: Wide Confidence Interval

```
95% CI: $500 to $5,000
```

**Meaning:** High uncertainty. Results vary wildly with trade order.

**Action:** Strategy may be too sensitive to market conditions.

### Pattern 3: Probability < 50%

```
Probability of profit: 45%
```

**Meaning:** Strategy more likely to lose than win!

**Action:** Don't trade this strategy. Go back to optimization.

### Pattern 4: Bimodal Distribution

```
Two peaks in histogram: One at $800, one at $2,000
```

**Meaning:** Strategy has two distinct outcomes (good vs. bad).

**Action:** Investigate what causes the split. Market regime?

## 📈 Example Interpretations

### Example 1: Robust Strategy ✅

```
Original Return:  +120%
Monte Carlo Mean: +115%
Probability:      88%
95% CI:           +80% to +150%
```

**Verdict:** Excellent! Original is close to mean, high probability, narrow range.

**Action:** ✅ Safe to trade. Expect ~115% return.

### Example 2: Lucky Backtest ⚠️

```
Original Return:  +200%
Monte Carlo Mean: +90%
Probability:      72%
95% CI:           +20% to +180%
```

**Verdict:** Original result was lucky (top 10%). Real expectation is +90%.

**Action:** ⚠️ Can trade, but expect half the backtest return.

### Example 3: Unreliable Strategy ❌

```
Original Return:  +80%
Monte Carlo Mean: +15%
Probability:      56%
95% CI:           -20% to +100%
```

**Verdict:** High variance, barely profitable, could lose money.

**Action:** ❌ Don't trade. Redesign strategy.

## 🔬 Advanced Usage

### Adjusting Simulation Count

```python
# Quick test (100 sims) - 30 seconds
Fast but less precise

# Standard (1000 sims) - 2-3 minutes  
Good balance of speed and accuracy

# High precision (5000 sims) - 10+ minutes
Very accurate confidence intervals
```

**Rule of thumb:** Use 1000 for routine testing, 5000 for final validation.

### Bootstrap vs. Randomization

The app uses **Trade Randomization** by default:
- Shuffles actual trades randomly
- Preserves trade distribution
- Shows order dependency

Alternative: **Bootstrap** (in code):
- Samples trades with replacement
- Can test with different trade counts
- Shows sampling uncertainty

### Combining with Walk-Forward

**Best practice:**
1. Run walk-forward analysis (tests time dependency)
2. Run Monte Carlo on out-of-sample results
3. Both should look good before trading

## 💡 Pro Tips

### Tip 1: Run MC on Multiple Optimizations

```
Optimization 1: MC shows 75% profit probability ✅
Optimization 2: MC shows 45% profit probability ❌
Optimization 3: MC shows 80% profit probability ✅

Action: Trade strategies 1 and 3, skip strategy 2
```

### Tip 2: Compare Assets

```python
# Run MC for each asset
AAPL MC: 85% profit prob, tight CI ✅
TSLA MC: 60% profit prob, wide CI ⚠️
BTC MC:  55% profit prob, huge CI ❌

Action: Allocate more to AAPL, less to TSLA, skip BTC
```

### Tip 3: Set Realistic Expectations

```
Don't plan for: Original backtest result
Do plan for:    Monte Carlo median - 1 std dev

This gives you margin of safety!
```

### Tip 4: Check Worst Case

```
If worst case MC result = lose 50%:
   Can you handle that emotionally?
   Can your account survive?

If no: Don't trade or reduce position size
```

## 📊 Statistical Concepts Explained

### What is "95% Confidence Interval"?

**Plain English:** In 95 out of 100 alternate universes, your result will be in this range.

**Example:**
```
95% CI: $1,200 to $2,000

Means: 95% chance your real result will be $1,200-$2,000
       5% chance it will be outside this range
```

### What is Standard Deviation?

**Plain English:** Typical distance from average.

**Example:**
```
Mean: $1,500
Std Dev: $300

Means: Most results are $1,200-$1,800 (mean ± 1 std dev)
       Almost all are $900-$2,100 (mean ± 2 std dev)
```

### What is "Probability of Profit"?

**Plain English:** In how many simulations did you make money?

**Example:**
```
Probability: 82%

Means: In 820 out of 1000 simulations, final equity > $1,000
       In 180 simulations, you lost money
```

## 🎓 When to Run Monte Carlo

### Always run Monte Carlo:
- ✅ Before going live with any strategy
- ✅ After each major optimization
- ✅ When backtest looks "too good to be true"
- ✅ When you have fewer than 100 trades

### Monte Carlo is less critical:
- If you have 500+ trades (law of large numbers kicks in)
- If walk-forward shows consistent results
- For very simple strategies with obvious edge

## ✅ Checklist Before Live Trading

- [ ] Strategy optimized with transaction costs
- [ ] Walk-forward validation passed
- [ ] Monte Carlo shows >70% profit probability
- [ ] Your original result within 1 std dev of median
- [ ] Worst-case scenario is acceptable
- [ ] 95% confidence interval is reasonable
- [ ] Ready for median result, not best case

## 🎯 Summary

**Key Takeaways:**

1. ✅ Monte Carlo tests if your strategy is robust or lucky
2. ✅ Run 1000+ simulations for reliable results
3. ✅ Probability >70% = good, >80% = excellent
4. ✅ Plan for median result, not original backtest
5. ✅ Wide confidence interval = high uncertainty
6. ✅ Original >> median = you got lucky
7. ✅ Always run MC before live trading

**Remember:** A strategy that only looks good in one specific trade sequence is not robust. Monte Carlo reveals the truth!

---

**Questions?** Check the console output for detailed statistics.
