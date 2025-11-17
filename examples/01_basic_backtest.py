"""
Example 1: Basic Backtesting
Simple RSI mean reversion strategy with realistic transaction costs
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine, OrderSide
from data_layer.feature_engineering import ATR, RSI, FeaturePipeline


def main():
    """Run a basic backtest example"""

    print("=" * 60)
    print("Example 1: Basic RSI Mean Reversion Backtest")
    print("=" * 60)

    # ========================================================================
    # Step 1: Download Data
    # ========================================================================
    print("\n📥 Downloading data...")
    symbol = "AAPL"
    df = yf.download(symbol, start="2020-01-01", end="2024-12-31", interval="1d")

    print(f"✓ Downloaded {len(df)} bars for {symbol}")

    # ========================================================================
    # Step 2: Compute Features
    # ========================================================================
    print("\n🔧 Computing features...")
    pipeline = FeaturePipeline(name="basic_features")
    pipeline.add_feature(RSI(period=14))
    pipeline.add_feature(ATR(period=14))

    features_df = pipeline.compute_features(df)
    print(f"✓ Computed {len(features_df.columns)} features")

    # ========================================================================
    # Step 3: Define Strategy
    # ========================================================================
    def rsi_strategy(bar, portfolio, engine):
        """
        Simple RSI mean reversion strategy

        Rules:
        - Buy when RSI < 30 (oversold)
        - Sell when RSI > 70 (overbought)
        """
        if "rsi_14" not in bar or pd.isna(bar["rsi_14"]):
            return

        rsi = bar["rsi_14"]

        # Entry: Oversold
        if rsi < 30:
            if symbol not in portfolio.positions:
                # Buy 100 shares
                engine.submit_order(symbol, OrderSide.BUY, 100)

        # Exit: Overbought
        elif rsi > 70:
            if symbol in portfolio.positions:
                qty = portfolio.positions[symbol].quantity
                if qty > 0:
                    engine.submit_order(symbol, OrderSide.SELL, qty)

    # ========================================================================
    # Step 4: Run Backtest
    # ========================================================================
    print("\n🚀 Running backtest...")

    engine = BacktestEngine(
        initial_cash=100_000,
        commission_pct=0.001,  # 0.1% commission
        spread_pct=0.0005,  # 5 bps spread
        slippage_pct=0.0002,  # 2 bps slippage
    )

    results_df = engine.run_backtest(features_df, rsi_strategy, symbol)

    # ========================================================================
    # Step 5: Analyze Results
    # ========================================================================
    print("\n📊 Performance Metrics:")
    print("-" * 60)

    metrics = engine.get_performance_metrics()

    print(f"Total Return:     {metrics['total_return']*100:>10.2f}%")
    print(f"Annual Return:    {metrics['annual_return']*100:>10.2f}%")
    print(f"Volatility:       {metrics['volatility']*100:>10.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']*100:>10.2f}%")
    print(f"Total Trades:     {metrics['total_trades']:>10}")
    print(f"Win Rate:         {metrics['win_rate']*100:>10.2f}%")
    print(f"Final Equity:     ${metrics['final_equity']:>10,.2f}")

    print("\n" + "=" * 60)

    # Optional: Plot equity curve
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        results_df["equity"].plot(title=f"{symbol} - RSI Strategy Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("examples/equity_curve.png", dpi=150)
        print("📈 Equity curve saved to examples/equity_curve.png")
    except ImportError:
        print("💡 Install matplotlib to generate equity curve plot")

    return metrics


if __name__ == "__main__":
    main()
