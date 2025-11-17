"""
Airflow DAG for Daily Quant Strategy Pipeline
Orchestrates: Data ingestion → Feature engineering → Backtesting → Validation → Deployment
"""

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.insert(0, "/app")

from backtest.engine import BacktestEngine
from backtest.robustness import WalkForwardAnalysis
from data_layer.feature_engineering import FeaturePipeline, FeatureStore
from data_layer.storage import VersionedDataStore
from data_layer.validation import DataValidator
from models.experiment_tracking import ExperimentTracker
from monitoring.metrics import MetricsCollector
from risk.risk_manager import RiskManager

# DAG default arguments
default_args = {
    "owner": "quant_team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email": ["alerts@quantfirm.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# Create DAG
dag = DAG(
    "daily_quant_strategy_pipeline",
    default_args=default_args,
    description="End-to-end quant strategy pipeline",
    schedule_interval="0 2 * * *",  # Run at 2 AM daily
    catchup=False,
    max_active_runs=1,
    tags=["production", "quant", "daily"],
)


# ============================================================================
# Task Functions
# ============================================================================


def ingest_market_data(**context):
    """Task 1: Ingest fresh market data"""
    import yfinance as yf

    from data.loader import DataLoader

    execution_date = context["execution_date"]

    symbols = Variable.get("trading_symbols", deserialize_json=True)
    data_store = VersionedDataStore("/app/data_layer")

    for symbol in symbols:
        print(f"Ingesting data for {symbol}")

        # Download data
        df = yf.download(symbol, period="1y", interval="1d")

        if df.empty:
            print(f"No data for {symbol}")
            continue

        # Write to versioned storage
        version_id = data_store.write_raw_data(
            df=df,
            symbol=symbol,
            data_source="yfinance",
            metadata={"execution_date": str(execution_date)},
        )

        print(f"Ingested {len(df)} rows for {symbol}, version: {version_id}")

    return "Data ingestion completed"


def validate_data_quality(**context):
    """Task 2: Validate data quality"""
    from pathlib import Path

    symbols = Variable.get("trading_symbols", deserialize_json=True)
    data_store = VersionedDataStore("/app/data_layer")
    validator = DataValidator()

    failed_symbols = []

    for symbol in symbols:
        print(f"Validating {symbol}")

        try:
            df = data_store.read_raw_data(symbol)

            results = validator.validate(df, symbol, expected_frequency="1d")

            if validator.has_critical_failures():
                failed_symbols.append(symbol)
                print(f"CRITICAL FAILURES for {symbol}")

            # Generate report
            report_path = f'/app/logs/validation_{symbol}_{context["ds"]}.json'
            validator.generate_validation_report(report_path)

        except Exception as e:
            print(f"Validation failed for {symbol}: {e}")
            failed_symbols.append(symbol)

    if failed_symbols:
        raise ValueError(f"Data quality check failed for: {', '.join(failed_symbols)}")

    return "Data validation passed"


def engineer_features(**context):
    """Task 3: Feature engineering"""
    symbols = Variable.get("trading_symbols", deserialize_json=True)
    data_store = VersionedDataStore("/app/data_layer")
    feature_store = FeatureStore(data_store)

    # Create feature pipeline
    pipeline = FeaturePipeline(name="standard_features")
    pipeline.add_standard_features()

    feature_store.register_pipeline(pipeline)

    for symbol in symbols:
        print(f"Computing features for {symbol}")

        # Load raw data
        df = data_store.read_raw_data(symbol)

        # Compute and store features
        version_id = feature_store.compute_and_store(
            df=df,
            symbol=symbol,
            pipeline_name="standard_features",
            metadata={"execution_date": context["ds"]},
        )

        print(f"Features computed for {symbol}, version: {version_id}")

    return "Feature engineering completed"


def run_backtest(**context):
    """Task 4: Run strategy backtest"""
    import numpy as np

    symbols = Variable.get("trading_symbols", deserialize_json=True)
    data_store = VersionedDataStore("/app/data_layer")
    feature_store = FeatureStore(data_store)

    backtest_results = {}

    for symbol in symbols:
        print(f"Running backtest for {symbol}")

        # Load features
        features_df = feature_store.get_features(
            symbol=symbol, pipeline_name="standard_features"
        )

        # Simple strategy: RSI mean reversion
        def simple_strategy(bar, portfolio, engine):
            if "rsi_14" not in bar:
                return

            rsi = bar["rsi_14"]

            # Entry signals
            if rsi < 30 and symbol not in [p for p in portfolio.positions.keys()]:
                # Oversold - buy
                quantity = 100
                engine.submit_order(symbol, "buy", quantity)

            elif rsi > 70 and symbol in portfolio.positions.keys():
                # Overbought - sell
                if portfolio.positions[symbol].quantity > 0:
                    quantity = portfolio.positions[symbol].quantity
                    engine.submit_order(symbol, "sell", quantity)

        # Run backtest
        engine = BacktestEngine(
            initial_cash=100000, commission_pct=0.001, spread_pct=0.0005
        )

        results = engine.run_backtest(features_df, simple_strategy, symbol)

        # Get metrics
        metrics = engine.get_performance_metrics()
        backtest_results[symbol] = metrics

        print(
            f"Backtest complete for {symbol}: Sharpe = {metrics.get('sharpe_ratio', 0):.2f}"
        )

    # Store results in XCom
    context["task_instance"].xcom_push(key="backtest_results", value=backtest_results)

    return backtest_results


def validate_robustness(**context):
    """Task 5: Robustness validation"""
    import pandas as pd

    backtest_results = context["task_instance"].xcom_pull(
        task_ids="run_backtest", key="backtest_results"
    )

    # Check if Sharpe ratios meet minimum threshold
    min_sharpe = Variable.get(
        "min_sharpe_ratio", default_var=1.0, deserialize_json=False
    )

    failed_strategies = []

    for symbol, metrics in backtest_results.items():
        sharpe = metrics.get("sharpe_ratio", 0)

        if sharpe < float(min_sharpe):
            failed_strategies.append(f"{symbol} (Sharpe: {sharpe:.2f})")

    if failed_strategies:
        print(f"WARNING: Strategies below threshold: {', '.join(failed_strategies)}")
        # Don't fail the DAG, just log warning

    return "Robustness validation completed"


def update_risk_limits(**context):
    """Task 6: Update risk limits based on backtest results"""
    backtest_results = context["task_instance"].xcom_pull(
        task_ids="run_backtest", key="backtest_results"
    )

    risk_config = {}

    for symbol, metrics in backtest_results.items():
        max_dd = abs(metrics.get("max_drawdown", 0.1))

        # Set position limit based on max drawdown
        # More volatile = smaller position
        position_limit = max(10000, 100000 * (0.05 / max(max_dd, 0.01)))

        risk_config[symbol] = {
            "max_position": position_limit,
            "max_drawdown": max_dd * 1.5,  # Allow 1.5x historical max DD
        }

    # Store risk config
    Variable.set("risk_config", risk_config, serialize_json=True)

    return "Risk limits updated"


def deploy_to_production(**context):
    """Task 7: Deploy approved strategies to production"""
    backtest_results = context["task_instance"].xcom_pull(
        task_ids="run_backtest", key="backtest_results"
    )

    # Only deploy strategies that meet criteria
    min_sharpe = float(Variable.get("min_sharpe_ratio", default_var=1.0))

    approved_strategies = [
        symbol
        for symbol, metrics in backtest_results.items()
        if metrics.get("sharpe_ratio", 0) >= min_sharpe
    ]

    if approved_strategies:
        Variable.set("approved_strategies", approved_strategies, serialize_json=True)
        print(f"Deployed strategies: {', '.join(approved_strategies)}")
    else:
        print("No strategies meet deployment criteria")

    return f"Deployed {len(approved_strategies)} strategies"


def send_daily_report(**context):
    """Task 8: Generate and send daily report"""
    backtest_results = context["task_instance"].xcom_pull(
        task_ids="run_backtest", key="backtest_results"
    )

    # Generate summary report
    report_lines = [
        "=" * 60,
        "Daily Quant Strategy Report",
        f"Execution Date: {context['ds']}",
        "=" * 60,
        "",
    ]

    for symbol, metrics in backtest_results.items():
        report_lines.extend(
            [
                f"Strategy: {symbol}",
                f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
                f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%",
                f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%",
                f"  Num Trades: {metrics.get('total_trades', 0)}",
                "",
            ]
        )

    report = "\n".join(report_lines)

    print(report)

    # In production: send via email/Slack
    # For now, write to file
    report_path = f'/app/logs/daily_report_{context["ds"]}.txt'
    with open(report_path, "w") as f:
        f.write(report)

    return "Daily report sent"


# ============================================================================
# Define Tasks
# ============================================================================

task_ingest = PythonOperator(
    task_id="ingest_market_data", python_callable=ingest_market_data, dag=dag
)

task_validate = PythonOperator(
    task_id="validate_data_quality", python_callable=validate_data_quality, dag=dag
)

task_features = PythonOperator(
    task_id="engineer_features", python_callable=engineer_features, dag=dag
)

task_backtest = PythonOperator(
    task_id="run_backtest", python_callable=run_backtest, dag=dag
)

task_robustness = PythonOperator(
    task_id="validate_robustness", python_callable=validate_robustness, dag=dag
)

task_risk = PythonOperator(
    task_id="update_risk_limits", python_callable=update_risk_limits, dag=dag
)

task_deploy = PythonOperator(
    task_id="deploy_to_production", python_callable=deploy_to_production, dag=dag
)

task_report = PythonOperator(
    task_id="send_daily_report", python_callable=send_daily_report, dag=dag
)

# ============================================================================
# Define Dependencies
# ============================================================================

task_ingest >> task_validate >> task_features >> task_backtest
task_backtest >> [task_robustness, task_risk]
[task_robustness, task_risk] >> task_deploy >> task_report
