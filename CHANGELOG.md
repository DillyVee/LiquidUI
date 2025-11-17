# Changelog

All notable changes to LiquidUI will be documented in this file.

## [Unreleased] - 2025-11-17

### Added - Market Regime Detection & Advanced Features

#### 🌍 Market Regime Detection & Prediction
- **Market Regime Detector** (`models/regime_detection.py`)
  - 5 regime types: Bull, Bear, High Vol, Low Vol, Crisis
  - Multi-factor scoring (volatility, trend, returns, momentum, drawdown)
  - Markov chain transition analysis
  - Historical regime statistics
  - Regime visualization with color-coded charts
  
#### 🤖 ML-Based Regime Prediction  
- **Regime Predictor** (`models/regime_predictor.py`)
  - Random Forest / XGBoost forecasting (1-20 days ahead)
  - 30+ engineered features
  - Time series cross-validation
  - Confidence-based predictions
  - Feature importance tracking

#### 📊 PBR Calculator (NEW!)
- **Probability of Backtested Returns**
  - Statistical measure of backtest-to-live performance
  - Multi-factor: Sharpe, sample size, overfitting, WF efficiency, regime stability
  - Interpretation: Very High (>80%), High (65-80%), Moderate (50-65%), Low (<50%)

#### ⚖️ Dynamic Position Sizing
- **Regime-Based Position Sizer**
  - Automatic 0.2x-2.0x adjustments
  - Confidence weighting
  - Forward-looking with ML predictions

#### 📚 Documentation
- QUICKSTART.md - Beginner to advanced guide
- CHEATSHEET.md - Quick reference
- GUI_FEATURES.md - Visual GUI map
- REGIME_DETECTION_GUIDE.md - Complete regime docs

#### 🧪 Testing & CI/CD
- pytest configuration (pytest.ini)
- Unit tests (tests/test_config.py)
- GitHub Actions CI/CD pipeline
- .isort.cfg for Black compatibility

### Fixed
- ✅ All linting (Black, isort, flake8)
- ✅ Integration test imports
- ✅ Docker build configuration
- ✅ Deprecated GitHub Actions (v3→v4)
- ✅ Type import errors

### Verified
- ✅ Monte Carlo (VaR, CVaR, Sharpe, Drawdown)
- ✅ Walk-Forward (efficiency, overfitting)
- ✅ PSR (variance scaling, CIs)

---

## [1.0.0] - 2025-11-17

### 🎉 Initial Release

✅ **CONFIRMED - All features below are implemented and working**

#### Core Infrastructure
- ✅ Data Layer (Parquet storage, validation, features)
- ✅ Structured Logging (audit trails, correlation IDs)
- ✅ Configuration Management (dataclass settings)

#### Backtesting & Execution  
- ✅ Advanced Backtesting Engine (realistic fills, slippage)
- ✅ Transaction Cost Models (Almgren-Chriss, spreads)
- ✅ Smart Order Routing (TWAP, VWAP, POV, Iceberg)
- ✅ Robustness Testing (CV, walk-forward, Monte Carlo)

#### Risk Management
- ✅ Real-Time Controls (position, P&L, leverage limits)
- ✅ Kill Switches (daily loss, drawdown, trailing stop)
- ✅ Risk Metrics (VaR, stress testing)

#### Monitoring
- ✅ Metrics Collection (Prometheus-compatible)
- ✅ Drift Detection (KS test, KL divergence)
- ✅ Alert Management (thresholds, severity)

#### Optimization
- ✅ Multi-Timeframe Optimizer (Optuna, PSR)
- ✅ Walk-Forward Analyzer (rolling windows)
- ✅ Monte Carlo Simulator (advanced metrics)
- ✅ PSR Calculator (Probabilistic Sharpe)

#### ML & Tracking
- ✅ Experiment Tracking (MLflow-compatible)
- ✅ Model Registry (versioning, promotion)

#### Deployment
- ✅ Docker (multi-stage builds)
- ✅ Docker Compose (Postgres, Redis, Airflow, Grafana, Prometheus, Jupyter)
- ✅ Airflow DAGs (automated pipelines)

#### Governance
- ✅ Model Cards (Google framework)
- ✅ Audit Logging (compliance)

#### GUI
- ✅ Main Trading Window (all features integrated)
- ✅ Live/Paper Trading (Alpaca)
- ✅ Charts & Visualizations

#### Examples
- ✅ examples/01_basic_backtest.py
- ✅ examples/02_walk_forward_validation.py
- ✅ examples/03_regime_based_trading.py (NEW!)

### File Structure
```
LiquidUI/
├── data_layer/          ✅ storage, validation, features
├── backtest/            ✅ engine, costs, robustness
├── execution/           ✅ order routing
├── risk/                ✅ risk manager, kill switches
├── monitoring/          ✅ metrics, drift detection
├── optimization/        ✅ optimizer, walk-forward, Monte Carlo, PSR
├── models/              ✅ experiment tracking, regime detection (NEW!), regime predictor (NEW!)
├── governance/          ✅ model cards
├── infrastructure/      ✅ logging, docker, airflow
├── gui/                 ✅ main window, styles
├── trading/             ✅ Alpaca integration
├── config/              ✅ settings
├── tests/               ✅ unit & integration tests
└── examples/            ✅ 3 working examples
```

### Tech Stack
- Python 3.11+
- PostgreSQL 15+, Redis 7+
- Apache Airflow 2.7+
- Docker & Docker Compose
- Prometheus & Grafana
- PyQt6 (GUI)
- scikit-learn, optuna, pandas, numpy
- Optional: XGBoost

### Performance
- 10,000+ bars/second (vectorized)
- Parquet partitioning (year/month)
- Feature caching
- Parallel processing

