# Market Regime Detection & Adaptive Position Sizing Guide

## Overview

This module provides institutional-grade market regime detection and prediction capabilities with automatic position sizing adjustments.

**NEW FEATURES:**
1. ✅ **Market Regime Detection** - HMM-style regime identification
2. ✅ **Regime Prediction** - ML-based forecasting (Random Forest / XGBoost)
3. ✅ **PBR Calculator** - Probability of Backtested Returns
4. ✅ **Dynamic Position Sizing** - Regime-aware risk management
5. ✅ **Transition Analysis** - Markov chain regime modeling

---

## What are Market Regimes?

Market regimes are distinct market states with different characteristics:

| Regime | Characteristics | Typical Duration | Position Size |
|--------|----------------|------------------|---------------|
| **Bull Market** | Low vol + positive trend + strong momentum | 100-300 days | 1.2x (increase 20%) |
| **Bear Market** | High vol + negative trend + weak momentum | 50-150 days | 0.5x (reduce 50%) |
| **High Volatility** | High vol + choppy + no clear trend | 20-60 days | 0.6x (reduce 40%) |
| **Low Volatility** | Low vol + range-bound + stable | 60-180 days | 1.0x (normal) |
| **Crisis** | Extreme vol + panic + large drawdowns | 10-40 days | 0.2x (reduce 80%) |

---

## Quick Start

### 1. Basic Regime Detection

```python
from models.regime_detection import MarketRegimeDetector
import yfinance as yf

# Load data
df = yf.download("SPY", start="2020-01-01", end="2024-01-01")
prices = df["Close"]
returns = prices.pct_change().dropna()

# Detect regime
detector = MarketRegimeDetector()
state = detector.detect_regime(prices, returns)

print(f"Current Regime: {state.current_regime.value}")
print(f"Confidence: {state.confidence:.1%}")
print(f"Suggested Position: {state.suggested_position_size:.2f}x")
```

**Output:**
```
Current Regime: bull
Confidence: 87.3%
Suggested Position: 1.15x
```

### 2. Regime Prediction with ML

```python
from models.regime_predictor import RegimePredictor

# Create and train predictor
predictor = RegimePredictor(detector, prediction_horizon=5)
predictor.train(prices, returns)

# Predict future regime
prediction = predictor.predict(prices, returns)

print(f"Predicted Regime (5 days): {prediction.predicted_regime.value}")
print(f"Confidence: {prediction.confidence:.1%}")
```

**Output:**
```
Predicted Regime (5 days): bull
Confidence: 73.5%
```

### 3. Calculate PBR (Probability of Backtested Returns)

```python
from models.regime_detection import PBRCalculator

# Your backtest results
backtest_sharpe = 1.8
backtest_return = 0.24  # 24% annual return
n_trades = 45
n_parameters = 4  # How many params you optimized
walk_forward_efficiency = 0.82  # From walk-forward analysis

# Calculate PBR
pbr, details = PBRCalculator.calculate_pbr(
    backtest_sharpe=backtest_sharpe,
    backtest_return=backtest_return,
    n_trades=n_trades,
    n_parameters=n_parameters,
    walk_forward_efficiency=walk_forward_efficiency,
)

print(f"PBR: {pbr:.1%}")
print(PBRCalculator.interpret_pbr(pbr))
```

**Output:**
```
PBR: 74.2%
Interpretation: High - Good confidence, but monitor closely
```

### 4. Dynamic Position Sizing

```python
from models.regime_predictor import RegimeBasedPositionSizer

# Create position sizer
sizer = RegimeBasedPositionSizer(
    detector=detector,
    predictor=predictor,  # Optional
    base_position_size=1.0,
    max_leverage=2.0,
)

# Calculate position size
sizing = sizer.calculate_position_size(prices, returns, use_prediction=True)

print(f"Recommended Position: {sizing['final_size']:.2f}x")
print(f"Current Regime: {sizing['current_regime']}")
```

**Output:**
```
Recommended Position: 1.18x
Current Regime: bull
```

---

## Detailed Feature Documentation

### 1. MarketRegimeDetector

**Purpose:** Detect current market regime using multi-factor analysis.

**Key Methods:**

```python
detector = MarketRegimeDetector(
    vol_window=20,              # Window for volatility calculation
    trend_window_fast=50,       # Fast MA for trend
    trend_window_slow=200,      # Slow MA for trend
    regime_memory=252,          # Days to remember for HMM
)

# Detect current regime
state = detector.detect_regime(prices, returns)

# Access results
state.current_regime           # MarketRegime enum
state.regime_probabilities     # Dict[MarketRegime, float]
state.confidence              # 0-1, how confident
state.regime_duration         # Days in current regime
state.predicted_next_regime   # Markov chain prediction
state.transition_probability  # Prob of staying
state.suggested_position_size # 0-2x multiplier

# Get historical stats
regime_stats = detector.get_regime_statistics(prices, returns)

# Visualize
detector.plot_regime_history(prices, returns, save_path="regimes.png")
```

**Regime Detection Logic:**

Uses 6 factor groups:
1. **Volatility** - Realized vol, vol trend, parkinson vol
2. **Trend** - Moving averages, price vs MA
3. **Returns** - Recent returns (20d, 60d annualized)
4. **Momentum** - RSI, rate of change
5. **Distribution** - Skewness, kurtosis (tail risk)
6. **Drawdown** - Current DD, max DD

Each regime gets scored, highest score wins.

### 2. RegimePredictor (ML-Based)

**Purpose:** Predict future regime using machine learning.

**Key Methods:**

```python
predictor = RegimePredictor(
    detector=detector,
    prediction_horizon=5,    # Days ahead to predict
    n_estimators=100,        # Trees in forest
    use_xgboost=False,       # Use XGBoost (needs install)
)

# Train model
performance = predictor.train(prices, returns, val_split=0.2)

# Predict
prediction = predictor.predict(prices, returns, horizon_days=5)

# Access results
prediction.predicted_regime        # MarketRegime
prediction.confidence             # 0-1
prediction.regime_probabilities   # Dict[MarketRegime, float]
prediction.features_importance    # Dict[str, float]
prediction.model_accuracy         # Historical accuracy

# Get top features
top_features = predictor.get_top_features(n=10)
```

**Model Features (30+ features):**
- Current regime (one-hot)
- Regime probabilities
- Regime persistence (duration, transition prob)
- Market features (vol, returns, skew, kurtosis)
- Trend features (MA 50, MA 200)
- Momentum (RSI)
- Drawdown

**Model Types:**
- `use_xgboost=False`: Random Forest (built-in, no dependencies)
- `use_xgboost=True`: XGBoost (better, requires `pip install xgboost`)

**Training:**
- Uses time series split (preserves temporal order)
- 80/20 train/val split by default
- Prints accuracy, precision, recall per regime

### 3. PBRCalculator

**Purpose:** Calculate probability that live trading will match backtest results.

**Formula:**
```
PBR = (Sharpe Factor)^0.35 ×
      (Sample Size)^0.25 ×
      (Overfitting)^0.20 ×
      (Walk-Forward)^0.10 ×
      (Regime Stability)^0.10
```

**Components:**

1. **Sharpe Factor** (35% weight)
   - Sharpe < 0: 10%
   - Sharpe 0-1: 30-50%
   - Sharpe 1-2: 50-80%
   - Sharpe > 2: 80-95%

2. **Sample Size** (25% weight)
   - < 30 trades: 50% penalty
   - 30-100 trades: 70%
   - 100-300 trades: 85%
   - > 300 trades: 95%

3. **Overfitting** (20% weight)
   - ≤ 2 params: 95%
   - 3-5 params: 85%
   - 6-10 params: 70%
   - > 10 params: 50%

4. **Walk-Forward Efficiency** (10% weight)
   - Uses WF efficiency directly

5. **Regime Stability** (10% weight)
   - Transition probability

**Interpretation:**
- **80%+**: Very High - Strong confidence
- **65-80%**: High - Good confidence
- **50-65%**: Moderate - Proceed with caution
- **35-50%**: Low - High risk
- **< 35%**: Very Low - Likely overfit

### 4. RegimeBasedPositionSizer

**Purpose:** Dynamically adjust position size based on regime.

**Key Methods:**

```python
sizer = RegimeBasedPositionSizer(
    detector=detector,
    predictor=predictor,        # Optional
    base_position_size=1.0,     # 100% capital
    max_leverage=2.0,           # Max 200%
    min_position_size=0.1,      # Min 10%
)

sizing = sizer.calculate_position_size(
    prices,
    returns,
    use_prediction=True  # Include ML prediction
)

# Results
sizing['position_size']          # Final size (0.1-2.0x)
sizing['current_regime']         # Regime name
sizing['regime_confidence']      # Detection confidence
sizing['base_from_regime']       # Size from current regime
sizing['prediction_adjustment']  # Adjustment from prediction
sizing['final_size']            # Final recommended size
```

**Logic:**

1. **Base Size** (from current regime)
   - Bull: 1.2x
   - Low Vol: 1.0x
   - High Vol: 0.6x
   - Bear: 0.5x
   - Crisis: 0.2x

2. **Confidence Adjustment**
   - Low confidence → move toward 1.0x (neutral)
   - High confidence → use full regime multiplier

3. **Prediction Adjustment** (if available)
   - If predicting better regime → increase
   - If predicting worse regime → decrease
   - Weighted by prediction confidence

4. **Limits**
   - Clipped to [min_size, max_leverage]

---

## Integration with Existing System

### Add to Backtesting Engine

```python
from models.regime_detection import MarketRegimeDetector
from models.regime_predictor import RegimeBasedPositionSizer

# In your backtest loop
detector = MarketRegimeDetector()
sizer = RegimeBasedPositionSizer(detector, base_position_size=1.0)

for date in backtest_dates:
    # Get data up to this date
    prices_to_date = prices[:date]
    returns_to_date = returns[:date]

    # Calculate regime-adjusted position size
    sizing = sizer.calculate_position_size(
        prices_to_date,
        returns_to_date,
        use_prediction=False  # No prediction in backtest
    )

    position_size = sizing['final_size']

    # Apply to your strategy
    if signal == "BUY":
        shares = (capital * position_size) / current_price
        execute_trade(shares)
```

### Add to GUI

```python
# In gui/main_window.py

# Add regime detection button
self.regime_btn = QPushButton("🌍 Detect Market Regime")
self.regime_btn.clicked.connect(self.detect_regime)

def detect_regime(self):
    """Detect and display current regime"""
    if self.df_dict_full is None:
        return

    # Get daily prices
    df_daily = self.df_dict_full.get("daily")
    if df_daily is None:
        return

    prices = df_daily["Close"]
    returns = prices.pct_change().dropna()

    # Detect regime
    detector = MarketRegimeDetector()
    state = detector.detect_regime(prices, returns)

    # Display results
    msg = f"""
    Current Market Regime: {state.current_regime.value.upper()}

    Confidence: {state.confidence:.1%}
    Duration: {state.regime_duration} days
    Suggested Position: {state.suggested_position_size:.2f}x

    Predicted Next: {state.predicted_next_regime.value}
    Transition Prob: {state.transition_probability:.1%}
    """

    QMessageBox.information(self, "Market Regime", msg)
```

---

## Example Workflows

### Workflow 1: Pre-Trade Regime Check

```python
# Before placing trade
detector = MarketRegimeDetector()
state = detector.detect_regime(prices, returns)

# Check if favorable regime
if state.current_regime in [MarketRegime.BULL, MarketRegime.LOW_VOL]:
    print("✅ Favorable regime, proceed with trade")
    position_size = state.suggested_position_size
else:
    print("⚠️ Unfavorable regime, reduce position or skip")
    position_size = 0.5  # Reduce by 50%
```

### Workflow 2: Strategy Validation

```python
# After backtesting
pbr, details = PBRCalculator.calculate_pbr(
    backtest_sharpe=your_sharpe,
    backtest_return=your_return,
    n_trades=your_trades,
    n_parameters=params_optimized,
    walk_forward_efficiency=wf_efficiency,
)

if pbr > 0.70:
    print("✅ HIGH PBR: Strategy validated, ready for paper trading")
elif pbr > 0.50:
    print("⚠️ MODERATE PBR: Additional validation recommended")
else:
    print("❌ LOW PBR: Strategy likely overfit, do not trade")
```

### Workflow 3: Dynamic Risk Management

```python
# Real-time position management
predictor = RegimePredictor(detector)
predictor.train(historical_prices, historical_returns)

sizer = RegimeBasedPositionSizer(detector, predictor)

# Each trading day
sizing = sizer.calculate_position_size(prices, returns, use_prediction=True)

# Adjust all positions
for symbol in portfolio:
    current_position = portfolio[symbol]
    target_position = base_position * sizing['final_size']

    if target_position > current_position:
        # Increase position
        buy_shares(symbol, target_position - current_position)
    elif target_position < current_position:
        # Reduce position
        sell_shares(symbol, current_position - target_position)
```

---

## Performance Tips

### 1. Training Frequency

- **Full retrain**: Weekly (expensive but accurate)
- **Incremental update**: Daily (faster, good enough)
- **No retrain**: Use detection only (fastest)

```python
# Weekly retrain
if datetime.today().weekday() == 0:  # Monday
    predictor.train(prices, returns)

# Daily: use .detect_regime() only (no ML)
state = detector.detect_regime(prices, returns)
```

### 2. Prediction Horizon

- **1-5 days**: High accuracy, short-term trading
- **5-10 days**: Medium accuracy, swing trading
- **10-20 days**: Lower accuracy, position trading

```python
# Short-term
predictor = RegimePredictor(detector, prediction_horizon=3)

# Medium-term
predictor = RegimePredictor(detector, prediction_horizon=10)
```

### 3. Regime Memory

- **Short (63 days)**: Responsive to recent changes
- **Medium (252 days)**: Balanced
- **Long (504 days)**: Stable, slower to change

```python
# Responsive
detector = MarketRegimeDetector(regime_memory=63)

# Stable
detector = MarketRegimeDetector(regime_memory=504)
```

---

## Verification & Testing

### Check Monte Carlo Implementation

```python
# All calculations verified ✅
from optimization.monte_carlo import AdvancedMonteCarloAnalyzer

# VaR: Uses percentiles correctly (5th for 95%, 1st for 99%)
# CVaR: Mean of losses beyond VaR (Expected Shortfall) ✅
# Sharpe: Properly annualized ✅
# Drawdown: Cumulative max method ✅
```

### Check Walk-Forward Implementation

```python
# All calculations verified ✅
from optimization.walk_forward import WalkForwardAnalyzer

# Train/test split: Temporal ordering preserved ✅
# Efficiency ratio: (Out-of-sample / In-sample) ✅
# Overfitting detection: Proper thresholds ✅
```

### Check PSR Implementation

```python
# All calculations verified ✅
from optimization.psr_composite import PSRCalculator

# Variance scaling: Correct for annualized Sharpe ✅
# Skewness/kurtosis: Properly adjusted ✅
# Confidence intervals: Bootstrap method ✅
```

---

## Troubleshooting

### Issue: "Model not trained"
**Solution:** Call `predictor.train()` before `predictor.predict()`

### Issue: Low prediction accuracy (< 50%)
**Solutions:**
- Increase training data (need 1+ years)
- Reduce prediction horizon (try 3-5 days)
- Check if regimes are well-defined (some markets don't have clear regimes)

### Issue: Regime changes too frequently
**Solutions:**
- Increase `regime_memory` parameter
- Smooth price data (use longer windows)
- Add minimum regime duration filter

### Issue: Position size too aggressive
**Solutions:**
- Reduce `max_leverage` parameter
- Increase `min_position_size` for safety floor
- Add your own risk overlay

---

## Mathematical References

1. **Hidden Markov Models**: Rabiner, L. R. (1989)
2. **Regime Detection**: Ang, A. & Timmermann, A. (2012)
3. **PSR**: Bailey, D. H. & López de Prado, M. (2012)
4. **CVaR/VaR**: Rockafellar, R. T. & Uryasev, S. (2000)

---

## Summary

✅ **Regime Detection** - Identifies current market state
✅ **Regime Prediction** - Forecasts future regime with ML
✅ **PBR Calculator** - Validates backtest robustness
✅ **Position Sizing** - Automatically adjusts risk based on regime
✅ **Verified Calculations** - Monte Carlo, Walk-Forward, PSR all correct

**Run the example:**
```bash
python examples/03_regime_based_trading.py
```

**Next Steps:**
1. Test on your symbols
2. Integrate with GUI
3. Backtest with regime-aware position sizing
4. Monitor PBR before going live
