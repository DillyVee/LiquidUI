"""
Example 2: Walk-Forward Validation
Demonstrates robust testing methodology with walk-forward analysis
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine, OrderSide
from backtest.robustness import WalkForwardAnalysis
from data_layer.feature_engineering import MACD, RSI, FeaturePipeline


def create_strategy_function(rsi_oversold=30, rsi_overbought=70):
    """Factory function to create strategy with specific parameters"""

    def strategy(bar, portfolio, engine):
        if "rsi_14" not in bar or pd.isna(bar["rsi_14"]):
            return

        rsi = bar["rsi_14"]
        symbol = "AAPL"  # Fixed for this example

        if rsi < rsi_oversold and symbol not in portfolio.positions:
            engine.submit_order(symbol, OrderSide.BUY, 100)
        elif rsi > rsi_overbought and symbol in portfolio.positions:
            qty = portfolio.positions[symbol].quantity
            if qty > 0:
                engine.submit_order(symbol, OrderSide.SELL, qty)

    return strategy


def run_backtest_with_params(train_data, test_data, **params):
    """Run backtest with given parameters"""
    strategy = create_strategy_function(
        rsi_oversold=params.get("rsi_oversold", 30),
        rsi_overbought=params.get("rsi_overbought", 70),
    )

    engine = BacktestEngine(
        initial_cash=100_000, commission_pct=0.001, spread_pct=0.0005
    )

    results = engine.run_backtest(test_data, strategy, "AAPL")
    metrics = engine.get_performance_metrics()

    return pd.Series([metrics["sharpe_ratio"]])


def main():
    """Run walk-forward validation example"""

    print("=" * 70)
    print("Example 2: Walk-Forward Validation")
    print("=" * 70)

    # Download data
    print("\n📥 Downloading data...")
    df = yf.download("AAPL", start="2020-01-01", end="2024-12-31", interval="1d")

    # Compute features
    print("🔧 Computing features...")
    pipeline = FeaturePipeline(name="features")
    pipeline.add_feature(RSI(period=14))
    pipeline.add_feature(MACD())

    features_df = pipeline.compute_features(df)

    # Define parameter grid
    param_grid = {
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
    }

    # Run walk-forward analysis
    print("\n🔄 Running walk-forward analysis...")
    print(f"   Training window: 252 days (1 year)")
    print(f"   Testing window:  63 days (3 months)")
    print(f"   Step size:       21 days (1 month)")
    print(
        f"   Parameter combinations: {len(param_grid['rsi_oversold']) * len(param_grid['rsi_overbought'])}"
    )

    wfa = WalkForwardAnalysis(train_window=252, test_window=63, step_size=21)

    # This would take a while with real execution
    print("\n⏳ Running analysis (this may take a few minutes)...")

    try:
        results = wfa.run(
            data=features_df,
            strategy_func=run_backtest_with_params,
            param_grid=param_grid,
            metric_func=lambda x: x.mean(),
        )

        # Display results
        print("\n📊 Walk-Forward Results:")
        print("-" * 70)
        print(f"Average Training Sharpe:  {results['avg_train_score']:>8.2f}")
        print(f"Average Test Sharpe:      {results['avg_test_score']:>8.2f}")
        print(f"Degradation:              {results['degradation_pct']*100:>8.1f}%")
        print(f"Number of Windows:        {len(results['test_scores']):>8}")

        # Interpretation
        print("\n💡 Interpretation:")
        if results["degradation_pct"] < 0.20:
            print("   ✅ GOOD: Less than 20% performance degradation")
            print("   Strategy shows good out-of-sample consistency")
        elif results["degradation_pct"] < 0.40:
            print("   ⚠️  MODERATE: 20-40% performance degradation")
            print("   Strategy may be slightly overfit. Use with caution.")
        else:
            print("   ❌ POOR: More than 40% performance degradation")
            print("   Strategy is likely overfit to training data")

        # Show best parameters from each fold
        print("\n🎯 Best Parameters per Fold:")
        for i, params in enumerate(results["best_params_per_fold"][:5]):  # Show first 5
            print(f"   Fold {i+1}: {params}")

    except Exception as e:
        print(f"\n⚠️  Note: Full walk-forward requires significant computation time")
        print(f"   Error: {e}")
        print(f"   In production, this would run on a dedicated server")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
