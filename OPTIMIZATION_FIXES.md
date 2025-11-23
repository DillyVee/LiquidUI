# Optimization Objective Fixes

## Changes Implemented

### 1. Fixed Cycle Phase Objective (CRITICAL)

**Location:** `optimization/optimizer.py:499-510`

**Problem:** The previous implementation penalized strategies with MORE trades, which is backwards:
```python
# BEFORE (WRONG):
trade_penalty = min(trades / 50.0, 1.0)
score = sharpe - trade_penalty
# Result: 50 trades = penalty of 1.0, so Sharpe 2.0 becomes score 1.0
#         1 trade = penalty of 0.02, so Sharpe 1.5 becomes score 1.48
```

**Solution:** Now rewards having enough trades for statistical confidence:
```python
# AFTER (CORRECT):
if trades < 10:
    trade_multiplier = 0.0  # Insufficient trades
elif trades < 30:
    trade_multiplier = trades / 30.0  # Scale up to 30
else:
    trade_multiplier = 1.0  # Sufficient statistical power

score = sharpe * trade_multiplier
```

**Impact:**
- Strategies with 1-9 trades: score = 0.0 (rejected)
- Strategies with 10-29 trades: score scales linearly (incentivizes more)
- Strategies with 30+ trades: full Sharpe score (optimal)

---

### 2. Fixed RSI Phase with Smooth Trade Count Penalty

**Location:** `optimization/optimizer.py:664-674`

**Problem:** Hard cutoff at 10 trades returned 0.0 PSR, treating 1 trade the same as 9 trades.

**Solution:** Smooth multiplier that scales from 0.0 to 1.0:
```python
if trade_count < 10:
    trade_multiplier = 0.0  # Insufficient trades
elif trade_count < 30:
    trade_multiplier = (trade_count - 10) / 20.0  # 0.0 to 1.0
else:
    trade_multiplier = 1.0  # Sufficient statistical power

psr = psr * trade_multiplier
sharpe = sharpe * trade_multiplier
```

**Impact:**
- Creates smooth gradient encouraging more trades
- Prevents optimizer from getting stuck at boundary conditions
- Aligns RSI phase incentives with cycle phase

---

## Recommended Parameter Range Changes

### Current Problem with Time Cycle Ranges

The current default ranges allow extreme duty cycles:
```python
time_cycle_ranges=((1, 50), (0, 50), (0, 100))
#                  on_range  off_range  start_range
```

**Worst case:** ON=1, OFF=50 → duty cycle = 1/(1+50) = 2% trading time!

### Recommended Changes

Update the GUI parameter ranges to prevent pathological cases:

```python
# RECOMMENDED:
time_cycle_ranges=((5, 30), (5, 30), (0, 60))
#                  on: 5-30  off: 5-30  start: 0-60
```

**Benefits:**
- Minimum duty cycle: 5/(5+30) = 14%
- Maximum duty cycle: 30/(30+5) = 86%
- Prevents single-trade strategies
- Ensures reasonable trading activity

**How to Implement:**
In your GUI code (e.g., `gui/main_window_v2.py`), update the spinbox ranges:
```python
# Example (adjust based on your GUI implementation):
self.on_min.setMinimum(5)   # was: 1
self.on_min.setValue(5)     # was: 1
self.off_min.setMinimum(5)  # was: 0
self.off_min.setValue(5)    # was: 0
self.on_max.setValue(30)    # was: 50
self.off_max.setValue(30)   # was: 50
```

---

## Expected Results

With these fixes, you should see:

1. **Consistent Trade Counts:** 30+ trades across all batches
2. **Monotonic Improvement:** Objectives improve or stabilize over batches
3. **No Extreme Parameters:** Cycle parameters stay in reasonable ranges
4. **Statistical Validity:** PSR scores based on sufficient samples

---

## Testing Recommendations

1. **Run a small test optimization** (100-200 trials) with new fixes
2. **Monitor trade counts** across batches - should stay ≥30
3. **Check cycle parameters** - ON and OFF should be balanced
4. **Compare PSR progression** - should improve or stabilize

---

## Notes

- The objective fixes are implemented in `optimization/optimizer.py`
- Parameter range changes require GUI updates (not yet implemented)
- Consider running a comparison: old params vs new params on same data
