"""
Integration Test: End-to-End Pipeline
Tests the full workflow from data ingestion to backtest execution
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, OrderSide
from data_layer.feature_engineering import MACD, RSI, FeaturePipeline
from data_layer.storage import VersionedDataStore
from data_layer.validation import DataValidator
from monitoring.metrics import MetricsCollector
from risk.risk_manager import RiskManager


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data"""
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "Open": 100 + np.random.randn(len(dates)).cumsum(),
            "High": 105 + np.random.randn(len(dates)).cumsum(),
            "Low": 95 + np.random.randn(len(dates)).cumsum(),
            "Close": 100 + np.random.randn(len(dates)).cumsum(),
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    # Ensure OHLC consistency
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


def test_full_pipeline_end_to_end(temp_data_dir, sample_ohlcv_data):
    """
    Integration test: Full pipeline from ingestion to backtest

    Steps:
    1. Ingest data to versioned storage
    2. Validate data quality
    3. Compute features
    4. Run backtest
    5. Check risk limits
    6. Record metrics
    """
    symbol = "TEST"

    # ========================================================================
    # Step 1: Data Ingestion
    # ========================================================================
    data_store = VersionedDataStore(temp_data_dir)

    version_id = data_store.write_raw_data(
        df=sample_ohlcv_data, symbol=symbol, data_source="test", metadata={"test": True}
    )

    assert version_id is not None
    print(f"✓ Data ingested: version {version_id}")

    # ========================================================================
    # Step 2: Data Validation
    # ========================================================================
    validator = DataValidator()
    results = validator.validate(sample_ohlcv_data, symbol, expected_frequency="1d")

    assert not validator.has_critical_failures(), "Data validation failed"
    print(f"✓ Data validation passed: {len(results)} checks")

    # ========================================================================
    # Step 3: Feature Engineering
    # ========================================================================
    pipeline = FeaturePipeline(name="test_features")
    pipeline.add_feature(RSI(period=14))
    pipeline.add_feature(MACD(fast=12, slow=26, signal=9))

    features_df = pipeline.compute_features(sample_ohlcv_data)

    assert "rsi_14" in features_df.columns
    assert "macd" in features_df.columns
    print(
        f"✓ Features computed: {len(features_df.columns)} columns, {len(features_df)} rows"
    )

    # ========================================================================
    # Step 4: Backtest Execution
    # ========================================================================
    def simple_strategy(bar, portfolio, engine):
        """Simple RSI strategy"""
        if "rsi_14" not in bar or pd.isna(bar["rsi_14"]):
            return

        # Entry: RSI < 30 (oversold)
        if bar["rsi_14"] < 30:
            if symbol not in portfolio.positions:
                engine.submit_order(symbol, OrderSide.BUY, 100)

        # Exit: RSI > 70 (overbought)
        elif bar["rsi_14"] > 70:
            if symbol in portfolio.positions:
                qty = portfolio.positions[symbol].quantity
                if qty > 0:
                    engine.submit_order(symbol, OrderSide.SELL, qty)

    # Run backtest
    engine = BacktestEngine(
        initial_cash=100000, commission_pct=0.001, spread_pct=0.0005
    )

    results_df = engine.run_backtest(features_df, simple_strategy, symbol)

    assert len(results_df) > 0
    print(f"✓ Backtest completed: {len(results_df)} periods")

    # ========================================================================
    # Step 5: Performance Metrics
    # ========================================================================
    metrics = engine.get_performance_metrics()

    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "total_return" in metrics

    print(f"✓ Performance metrics:")
    print(f"  - Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  - Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  - Total Trades: {metrics['total_trades']}")

    # ========================================================================
    # Step 6: Risk Management Checks
    # ========================================================================
    risk_config = {
        "position_limits": {
            "max_gross_exposure": 1_000_000,
            "max_single_position": 100_000,
        },
        "pnl_limits": {
            "max_daily_loss": -10_000,
            "max_drawdown_pct": -0.50,
        },
    }

    risk_manager = RiskManager(risk_config)

    # Simulate current state
    positions = {symbol: 100}
    prices = {symbol: features_df.iloc[-1]["Close"]}
    portfolio_value = metrics["final_equity"]

    risk_level, violations = risk_manager.check_all_limits(
        positions=positions,
        prices=prices,
        portfolio_value=portfolio_value,
        realized_pnl_today=0,
    )

    print(f"✓ Risk check: {risk_level.value}")
    if violations:
        for v in violations:
            print(f"  - {v}")

    # ========================================================================
    # Step 7: Metrics Collection
    # ========================================================================
    metrics_collector = MetricsCollector()

    metrics_collector.record_metric(
        "strategy.sharpe_ratio",
        metrics["sharpe_ratio"],
        tags={"strategy": "test", "symbol": symbol},
    )
    metrics_collector.record_metric(
        "strategy.total_return",
        metrics["total_return"],
        tags={"strategy": "test", "symbol": symbol},
    )

    # Retrieve metrics
    sharpe_series = metrics_collector.get_metric_series("strategy.sharpe_ratio")
    assert len(sharpe_series) > 0

    print(f"✓ Metrics recorded: {len(metrics_collector.metrics)} metric types")

    # ========================================================================
    # Assertions
    # ========================================================================
    # Check that strategy executed trades
    assert metrics["total_trades"] > 0, "No trades executed"

    # Check that equity changed
    assert metrics["final_equity"] != metrics["initial_equity"], "No P&L change"

    # Check that we have fills
    assert len(engine.fills) > 0, "No fills recorded"

    print("\n✅ End-to-end integration test PASSED")


def test_data_versioning_integrity(temp_data_dir, sample_ohlcv_data):
    """Test data versioning and integrity checks"""
    data_store = VersionedDataStore(temp_data_dir)

    # Write data
    version_id = data_store.write_raw_data(
        df=sample_ohlcv_data, symbol="TEST", data_source="test"
    )

    # Validate integrity
    is_valid, error_msg = data_store.validate_data_integrity(version_id)

    assert is_valid, f"Data integrity check failed: {error_msg}"

    # List versions
    versions = data_store.list_versions("TEST", "raw")
    assert len(versions) == 1
    assert versions[0].version_id == version_id

    print("✅ Data versioning integrity test PASSED")


def test_feature_pipeline_reproducibility(sample_ohlcv_data):
    """Test that feature pipeline produces consistent results"""
    pipeline = FeaturePipeline(name="reproducibility_test")
    pipeline.add_feature(RSI(period=14))

    # Compute features twice
    features_1 = pipeline.compute_features(sample_ohlcv_data.copy())
    features_2 = pipeline.compute_features(sample_ohlcv_data.copy())

    # Should be identical
    pd.testing.assert_frame_equal(features_1, features_2)

    print("✅ Feature pipeline reproducibility test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
