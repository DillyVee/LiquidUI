"""
Example: Market Regime Detection and Adaptive Position Sizing

This example demonstrates:
1. Detecting market regimes using HMM-style analysis
2. Predicting future regimes with ML
3. Calculating PBR (Probability of Backtested Returns)
4. Dynamic position sizing based on regime

Usage:
    python examples/03_regime_based_trading.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.regime_detection import MarketRegimeDetector, PBRCalculator, RegimeMetrics
from models.regime_predictor import RegimeBasedPositionSizer, RegimePredictor


def main():
    """Main example"""
    print("=" * 80)
    print("MARKET REGIME DETECTION & ADAPTIVE POSITION SIZING")
    print("=" * 80)

    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n📊 Loading market data...")

    ticker = "SPY"  # S&P 500 ETF
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    prices = df["Close"]
    returns = prices.pct_change().dropna()

    print(f"✅ Loaded {len(prices)} days of {ticker} data")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # ============================================================================
    # 2. DETECT MARKET REGIMES
    # ============================================================================
    print("\n🔍 Detecting market regimes...")

    detector = MarketRegimeDetector(
        vol_window=20,
        trend_window_fast=50,
        trend_window_slow=200,
    )

    # Detect current regime
    current_state = detector.detect_regime(prices, returns)

    print(f"\n📈 CURRENT MARKET REGIME: {current_state.current_regime.value.upper()}")
    print(f"   Confidence:        {current_state.confidence:.1%}")
    print(f"   Duration:          {current_state.regime_duration} days")
    print(f"   Predicted Next:    {current_state.predicted_next_regime.value}")
    print(f"   Stay Probability:  {current_state.transition_probability:.1%}")
    print(f"   Suggested Position:{current_state.suggested_position_size:.2f}x")

    print(f"\n📊 Regime Probabilities:")
    for regime, prob in sorted(
        current_state.regime_probabilities.items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(prob * 50)
        print(f"   {regime.value:20s} {prob:5.1%} {bar}")

    # ============================================================================
    # 3. GET HISTORICAL REGIME STATISTICS
    # ============================================================================
    print("\n📈 Historical Regime Performance:")

    regime_stats = detector.get_regime_statistics(prices, returns)

    print(
        f"\n{'Regime':<20} {'Avg Return':<12} {'Sharpe':<8} {'Max DD':<10} {'Win Rate':<10} {'Avg Days'}"
    )
    print("-" * 80)

    for regime, metrics in regime_stats.items():
        print(
            f"{regime.value:<20} "
            f"{metrics.avg_return:>10.1%} "
            f"{metrics.sharpe:>8.2f} "
            f"{metrics.max_drawdown:>9.1%} "
            f"{metrics.win_rate:>9.1%} "
            f"{metrics.avg_duration_days:>8d}"
        )

    # ============================================================================
    # 4. TRAIN PREDICTIVE MODEL
    # ============================================================================
    print("\n🤖 Training predictive model...")

    predictor = RegimePredictor(
        detector=detector,
        prediction_horizon=5,  # 5 days ahead
        n_estimators=100,
        use_xgboost=False,  # Set to True if xgboost installed
    )

    # Train on historical data
    performance = predictor.train(prices, returns, val_split=0.2)

    print(f"\n📊 Model Performance:")
    print(f"   Validation Accuracy: {performance.accuracy_5day:.1%}")

    print(f"\n   Precision by Regime:")
    for regime, prec in performance.precision_by_regime.items():
        print(f"      {regime.value:<20} {prec:.1%}")

    print(f"\n   Recall by Regime:")
    for regime, rec in performance.recall_by_regime.items():
        print(f"      {regime.value:<20} {rec:.1%}")

    # ============================================================================
    # 5. PREDICT FUTURE REGIME
    # ============================================================================
    print("\n🔮 Predicting future regime...")

    prediction = predictor.predict(prices, returns, horizon_days=5)

    print(
        f"\n📈 PREDICTED REGIME (5 days ahead): {prediction.predicted_regime.value.upper()}"
    )
    print(f"   Confidence:        {prediction.confidence:.1%}")
    print(f"   Model Accuracy:    {prediction.model_accuracy:.1%}")

    print(f"\n📊 Predicted Probabilities:")
    for regime, prob in sorted(
        prediction.regime_probabilities.items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(prob * 50)
        print(f"   {regime.value:20s} {prob:5.1%} {bar}")

    # Top features
    print(f"\n🎯 Top 10 Predictive Features:")
    top_features = predictor.get_top_features(n=10)
    for i, (feature, importance) in enumerate(top_features, 1):
        bar = "█" * int(importance * 50)
        print(f"   {i:2d}. {feature:<30s} {importance:.3f} {bar}")

    # ============================================================================
    # 6. CALCULATE PBR (PROBABILITY OF BACKTESTED RETURNS)
    # ============================================================================
    print("\n💰 Calculating PBR (Probability of Backtested Returns)...")

    # Simulate backtest results
    backtest_sharpe = 1.8
    backtest_return = 0.24  # 24% annual return
    n_trades = 45
    n_parameters = 4
    walk_forward_efficiency = 0.82
    regime_stability = current_state.transition_probability

    pbr, pbr_details = PBRCalculator.calculate_pbr(
        backtest_sharpe=backtest_sharpe,
        backtest_return=backtest_return,
        n_trades=n_trades,
        n_parameters=n_parameters,
        walk_forward_efficiency=walk_forward_efficiency,
        current_regime_stability=regime_stability,
    )

    print(f"\n📊 BACKTEST ASSUMPTIONS:")
    print(f"   Sharpe Ratio:         {backtest_sharpe:.2f}")
    print(f"   Annual Return:        {backtest_return:.1%}")
    print(f"   Number of Trades:     {n_trades}")
    print(f"   Parameters Optimized: {n_parameters}")
    print(f"   WF Efficiency:        {walk_forward_efficiency:.1%}")
    print(f"   Regime Stability:     {regime_stability:.1%}")

    print(f"\n🎯 PBR ANALYSIS:")
    print(f"   PBR Score:            {pbr:.1%}")
    print(f"   Interpretation:       {PBRCalculator.interpret_pbr(pbr)}")

    print(f"\n   Contributing Factors:")
    print(f"      Sharpe:           {pbr_details['sharpe_contribution']:.1%}")
    print(f"      Sample Size:      {pbr_details['sample_size_factor']:.1%}")
    print(f"      Overfitting:      {pbr_details['overfitting_factor']:.1%}")
    print(f"      Walk-Forward:     {pbr_details['walkforward_factor']:.1%}")
    print(f"      Regime Stability: {pbr_details['regime_stability_factor']:.1%}")

    # ============================================================================
    # 7. DYNAMIC POSITION SIZING
    # ============================================================================
    print("\n⚖️  Calculating dynamic position sizes...")

    position_sizer = RegimeBasedPositionSizer(
        detector=detector,
        predictor=predictor,
        base_position_size=1.0,
        max_leverage=2.0,
        min_position_size=0.1,
    )

    # Without prediction
    sizing_no_pred = position_sizer.calculate_position_size(
        prices, returns, use_prediction=False
    )

    # With prediction
    sizing_with_pred = position_sizer.calculate_position_size(
        prices, returns, use_prediction=True
    )

    print(f"\n📊 POSITION SIZING:")
    print(f"\n   Without Prediction (Current Regime Only):")
    print(f"      Current Regime:        {sizing_no_pred['current_regime']}")
    print(f"      Regime Confidence:     {sizing_no_pred['regime_confidence']:.1%}")
    print(f"      Suggested Position:    {sizing_no_pred['position_size']:.2f}x")

    print(f"\n   With Prediction (Forward-Looking):")
    print(f"      Current Regime:        {sizing_with_pred['current_regime']}")
    print(
        f"      Prediction Adjustment: {sizing_with_pred['prediction_adjustment']:.2f}x"
    )
    print(f"      Final Position:        {sizing_with_pred['final_size']:.2f}x")

    # Calculate impact
    capital = 100000  # $100k
    position_basic = capital * 1.0
    position_regime_only = capital * sizing_no_pred["position_size"]
    position_regime_pred = capital * sizing_with_pred["final_size"]

    print(f"\n💵 Position Size Comparison (for ${capital:,} capital):")
    print(f"      Static (100%):         ${position_basic:,.0f}")
    print(f"      Regime-Aware:          ${position_regime_only:,.0f}")
    print(f"      Regime + Prediction:   ${position_regime_pred:,.0f}")

    # ============================================================================
    # 8. VISUALIZE REGIME HISTORY
    # ============================================================================
    print("\n📊 Generating regime visualization...")

    detector.plot_regime_history(prices, returns, save_path="regime_history.png")

    print(f"✅ Saved regime visualization to: regime_history.png")

    # ============================================================================
    # 9. SUMMARY & RECOMMENDATIONS
    # ============================================================================
    print("\n" + "=" * 80)
    print("TRADING RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n🎯 Current Market State:")
    print(f"   • Regime: {current_state.current_regime.value.upper()}")
    print(f"   • Confidence: {current_state.confidence:.1%}")
    print(f"   • PBR Score: {pbr:.1%}")

    print(f"\n📈 Position Sizing:")
    print(f"   • Recommended: {sizing_with_pred['final_size']:.2f}x base position")
    if sizing_with_pred["final_size"] > 1.2:
        print(f"   • Action: INCREASE position size (favorable conditions)")
    elif sizing_with_pred["final_size"] < 0.8:
        print(f"   • Action: DECREASE position size (unfavorable conditions)")
    else:
        print(f"   • Action: MAINTAIN standard position size")

    print(f"\n⚠️  Risk Warnings:")
    if pbr < 0.50:
        print(f"   • LOW PBR: High overfitting risk, trade carefully")
    if current_state.regime_duration < 5:
        print(f"   • RECENT REGIME CHANGE: Regime unstable, reduce size")
    if current_state.confidence < 0.70:
        print(f"   • LOW CONFIDENCE: Unclear regime, trade defensively")

    print(f"\n✅ Confidence Indicators:")
    if pbr > 0.70:
        print(f"   • HIGH PBR: Strategy likely to perform well")
    if current_state.transition_probability > 0.80:
        print(f"   • STABLE REGIME: Likely to persist")
    if prediction.model_accuracy > 0.70:
        print(f"   • RELIABLE PREDICTION: Model has good track record")

    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    main()
