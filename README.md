# LiquidUI - Professional Quantitative Trading Pipeline

**Production-grade algorithmic trading infrastructure with institutional-level robustness, monitoring, and governance.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/DillyVee/LiquidUI/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/DillyVee/LiquidUI.svg)](https://github.com/DillyVee/LiquidUI/issues)

---

## 🚀 Features

### Core Infrastructure
- **Advanced Data Layer**: Versioned, immutable storage with Parquet backend and data lineage tracking
- **Feature Engineering Pipeline**: Modular feature computation with automatic caching and dependency management
- **Production Backtesting Engine**: Realistic market simulation with order book replay and fill models
- **Transaction Cost Modeling**: Almgren-Chriss market impact, slippage models, and capacity analysis
- **Smart Execution System**: TWAP, VWAP, POV algorithms with intelligent order routing

### Risk & Monitoring
- **Real-Time Risk Management**: Position limits, P&L controls, and automated kill switches
- **Observability Stack**: Prometheus metrics, Grafana dashboards, and drift detection
- **Audit Trail**: Comprehensive logging with correlation IDs and regulatory compliance

### Research & Validation
- **Robustness Testing Suite**: Nested CV, walk-forward analysis, Monte Carlo validation
- **Parameter Stability Analysis**: Sensitivity surfaces and regime-aware testing
- **Experiment Tracking**: MLflow-compatible system with model registry

### Deployment & Orchestration
- **Docker Containerization**: Multi-stage builds for dev, test, and production
- **Airflow Orchestration**: End-to-end daily pipeline with failure recovery
- **Kubernetes Ready**: Horizontal scaling and high availability

### Governance
- **Model Cards**: Comprehensive documentation following Google's framework
- **Data Quality Framework**: Automated validation with Great Expectations patterns
- **Reproducibility**: Pinned dependencies, versioned data, and artifact tracking

---

## 📁 Project Structure

```
LiquidUI/
├── data_layer/               # Data infrastructure
│   ├── storage.py            # Versioned Parquet storage
│   ├── validation.py         # Data quality checks
│   └── feature_engineering.py # Feature pipeline
│
├── backtest/                 # Backtesting engine
│   ├── engine.py             # Core backtest engine
│   ├── transaction_costs.py  # Cost models
│   └── robustness.py         # Validation suite
│
├── execution/                # Execution system
│   └── order_router.py       # Smart order routing
│
├── risk/                     # Risk management
│   └── risk_manager.py       # Real-time risk controls
│
├── monitoring/               # Observability
│   └── metrics.py            # Metrics & drift detection
│
├── models/                   # Model management
│   └── experiment_tracking.py # Experiment tracking
│
├── governance/               # Governance & compliance
│   └── model_card.py         # Model documentation
│
├── infrastructure/           # Infrastructure code
│   ├── logger.py             # Structured logging
│   ├── docker/               # Docker configs
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── airflow/              # Airflow DAGs
│       └── dags/
│
├── tests/                    # Test suite
│   ├── unit/
│   ├── integration/
│   └── stress/
│
├── requirements.txt          # Python dependencies
├── requirements-test.txt     # Test dependencies
└── README.md                 # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for full stack)
- PostgreSQL 15+ (or use Docker)
- Redis (or use Docker)

### Quick Start (Local Development)

```bash
# Clone the repository
git clone https://github.com/yourorg/LiquidUI.git
cd LiquidUI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest tests/ -v

# Start GUI application
python main.py
```

### Docker Deployment (Production)

```bash
# Build and start all services
docker-compose -f infrastructure/docker/docker-compose.yml up -d

# Services will be available at:
# - Airflow Web UI: http://localhost:8080
# - Grafana Dashboard: http://localhost:3000
# - Prometheus: http://localhost:9090
# - Jupyter Lab: http://localhost:8888
# - Main App: http://localhost:8000

# View logs
docker-compose -f infrastructure/docker/docker-compose.yml logs -f quant_app

# Stop all services
docker-compose -f infrastructure/docker/docker-compose.yml down
```

---

## 📊 Usage Examples

### 1. Data Ingestion & Validation

```python
from data_layer.storage import VersionedDataStore
from data_layer.validation import DataValidator
import yfinance as yf

# Initialize storage
data_store = VersionedDataStore('/path/to/data')

# Download and store data
df = yf.download('AAPL', start='2020-01-01', end='2024-12-31')

version_id = data_store.write_raw_data(
    df=df,
    symbol='AAPL',
    data_source='yfinance',
    metadata={'description': 'Daily OHLCV data'}
)

# Validate data quality
validator = DataValidator()
results = validator.validate(df, 'AAPL', expected_frequency='1d')

if validator.has_critical_failures():
    print("Data quality issues detected!")
    validator.generate_validation_report('validation_report.json')
```

### 2. Feature Engineering

```python
from data_layer.feature_engineering import FeaturePipeline, RSI, MACD, ATR

# Create pipeline
pipeline = FeaturePipeline(name='momentum_features')

# Add features
pipeline.add_feature(RSI(period=14))
pipeline.add_feature(MACD(fast=12, slow=26, signal=9))
pipeline.add_feature(ATR(period=14))

# Compute features
features_df = pipeline.compute_features(df)

print(features_df.head())
```

### 3. Backtesting with Transaction Costs

```python
from backtest.engine import BacktestEngine, OrderSide
from backtest.transaction_costs import AlmgrenChrissModel

# Initialize backtest engine
engine = BacktestEngine(
    initial_cash=100000,
    commission_pct=0.001,
    spread_pct=0.0005,
    slippage_pct=0.0002
)

# Define strategy
def rsi_strategy(bar, portfolio, engine):
    if bar['rsi_14'] < 30:
        # Oversold - buy
        engine.submit_order('AAPL', OrderSide.BUY, 100)
    elif bar['rsi_14'] > 70:
        # Overbought - sell
        if 'AAPL' in portfolio.positions:
            qty = portfolio.positions['AAPL'].quantity
            engine.submit_order('AAPL', OrderSide.SELL, qty)

# Run backtest
results = engine.run_backtest(features_df, rsi_strategy, symbol='AAPL')

# Get metrics
metrics = engine.get_performance_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
```

### 4. Robustness Testing

```python
from backtest.robustness import WalkForwardAnalysis, MonteCarloValidation

# Walk-forward analysis
wfa = WalkForwardAnalysis(train_window=252, test_window=63, step_size=21)

param_grid = {
    'rsi_period': [10, 14, 20],
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75]
}

results = wfa.run(data=features_df, strategy_func=rsi_strategy, param_grid=param_grid)

print(f"Average test Sharpe: {results['avg_test_score']:.2f}")
print(f"Performance degradation: {results['degradation_pct']*100:.1f}%")

# Monte Carlo validation
mc = MonteCarloValidation(n_simulations=1000)
returns = results_df['equity'].pct_change()

mean, std, dist = mc.bootstrap_returns(returns, lambda r: r.mean() / r.std())
print(f"Bootstrap Sharpe: {mean:.2f} ± {std:.2f}")
```

### 5. Risk Management

```python
from risk.risk_manager import RiskManager

# Initialize risk manager
risk_config = {
    'position_limits': {
        'max_gross_exposure': 1_000_000,
        'max_single_position': 100_000,
        'max_leverage': 1.5
    },
    'pnl_limits': {
        'max_daily_loss': -10_000,
        'max_drawdown_pct': -0.10,
        'trailing_stop_pct': -0.05
    }
}

risk_manager = RiskManager(risk_config)

# Check limits
positions = {'AAPL': 1000, 'GOOGL': 500}
prices = {'AAPL': 180, 'GOOGL': 140}

risk_level, violations = risk_manager.check_all_limits(
    positions=positions,
    prices=prices,
    portfolio_value=500_000,
    realized_pnl_today=-5_000
)

if risk_level.value == 'red':
    print("KILL SWITCH ACTIVATED!")

print(f"Risk Level: {risk_level.value}")
for v in violations:
    print(f"  - {v}")
```

### 6. Experiment Tracking

```python
from models.experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker('/path/to/experiments')

# Create experiment
exp_id = tracker.create_experiment('rsi_optimization', 'Optimize RSI parameters')

# Start run
run_id = tracker.start_run('run_001', tags={'strategy': 'rsi', 'symbol': 'AAPL'})

# Log parameters
tracker.log_params({'rsi_period': 14, 'oversold': 30, 'overbought': 70})

# Log metrics
tracker.log_metrics({'sharpe_ratio': 1.85, 'max_drawdown': -0.12, 'win_rate': 0.58})

# End run
tracker.end_run('completed')
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test suite
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run stress tests
pytest tests/stress/ -v

# Property-based testing (using Hypothesis)
pytest tests/unit/test_backtest.py -v
```

---

## 📈 Monitoring & Observability

### Metrics Dashboard (Grafana)

Access: `http://localhost:3000` (default credentials: admin/admin)

Key dashboards:
- **Strategy Performance**: Real-time P&L, Sharpe ratio, drawdown
- **Risk Metrics**: Position limits, VaR, exposure
- **Execution Quality**: Fill rates, slippage, latency
- **Data Quality**: Missing data, outliers, validation failures

### Prometheus Metrics

Example metrics exposed:
- `strategy_sharpe_ratio{strategy="rsi", symbol="AAPL"}`
- `strategy_total_return{strategy="rsi"}`
- `strategy_num_trades`
- `risk_gross_exposure`
- `risk_leverage`
- `data_validation_failures`

### Logs

Structured JSON logs with correlation IDs:

```bash
# View application logs
tail -f logs/quant_20251117.jsonl | jq .

# View audit logs
tail -f logs/audit/audit_20251117.jsonl | jq .

# Search logs by correlation ID
cat logs/quant_20251117.jsonl | jq 'select(.correlation_id=="abc123")'
```

---

## 🛡️ Security & Compliance

### API Keys & Secrets

**NEVER commit API keys to version control.**

Use environment variables or secrets management:

```bash
# .env file (add to .gitignore)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
POSTGRES_PASSWORD=strong_password
AIRFLOW_FERNET_KEY=generate_fernet_key
```

### Audit Trail

All critical operations are logged to audit trail:
- Trade executions
- Risk events (kill switch, limit breaches)
- Model deployments
- Configuration changes

Audit logs are **append-only** and stored in `/logs/audit/`.

---

## 🚨 Production Checklist

Before deploying to production:

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Walk-forward validation shows <20% degradation
- [ ] Monte Carlo confidence intervals calculated
- [ ] Transaction costs calibrated to live fills
- [ ] Risk limits configured and tested
- [ ] Kill switches tested
- [ ] Monitoring dashboards configured
- [ ] Alert routing set up (email, Slack, PagerDuty)
- [ ] Model card generated and reviewed
- [ ] Data validation pipeline running
- [ ] Backup and disaster recovery plan documented
- [ ] Regulatory compliance verified
- [ ] Capacity limits estimated and documented

---

## 📚 Documentation

- **Architecture Overview**: `docs/architecture.md`
- **API Reference**: `docs/api.md`
- **Deployment Guide**: `docs/deployment.md`
- **Contributing Guidelines**: `CONTRIBUTING.md`
- **Changelog**: `CHANGELOG.md`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Code Standards:**
- Follow PEP 8 style guide
- Add type hints to all functions
- Write unit tests for new features (target: 80% coverage)
- Update documentation

---

## 📜 License

Proprietary - All Rights Reserved

---

## 🙏 Acknowledgments

- **Almgren & Chriss** - Optimal execution framework
- **Great Expectations** - Data validation patterns
- **MLflow** - Experiment tracking inspiration
- **Prometheus & Grafana** - Monitoring stack

---

## 📞 Support

For questions or issues:
- Email: support@example.com
- Slack: #quant-trading
- Issues: https://github.com/yourorg/LiquidUI/issues

---

**⚠️ Risk Warning**: Trading involves substantial risk of loss. This software is provided for research and educational purposes. Always test thoroughly on paper accounts before deploying real capital.

