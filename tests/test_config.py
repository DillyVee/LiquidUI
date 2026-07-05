"""
Basic configuration and smoke tests
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_python_version():
    """Test that we're running on Python 3.11+"""
    assert sys.version_info >= (3, 11), "Requires Python 3.11 or higher"


def test_project_structure():
    """Test that core project directories exist"""
    for directory in ("config", "data", "gui", "models", "optimization", "trading"):
        assert (PROJECT_ROOT / directory).is_dir(), f"{directory} directory should exist"


def test_requirements_files():
    """Test that requirements files exist"""
    assert (PROJECT_ROOT / "requirements.txt").exists()
    assert (PROJECT_ROOT / "requirements-test.txt").exists()


def test_import_config():
    """Test that config module can be imported and exposes expected classes"""
    from config import settings

    for attr in (
        "OptimizationConfig",
        "RiskConfig",
        "TransactionCosts",
        "IndicatorRanges",
        "Paths",
    ):
        assert hasattr(settings, attr), f"{attr} should exist in settings"


def test_transaction_costs():
    """Test transaction cost math"""
    from config.settings import TransactionCosts

    costs = TransactionCosts()
    assert costs.TOTAL_PCT == 0.0

    costs.COMMISSION_PCT = 0.001
    costs.SLIPPAGE_PCT = 0.0005
    costs.SPREAD_PCT = 0.0005
    assert abs(costs.TOTAL_PCT - 0.002) < 1e-12

    stocks = TransactionCosts.for_stocks()
    assert stocks.TOTAL_PCT > 0

    crypto = TransactionCosts.for_crypto()
    assert crypto.TOTAL_PCT > stocks.TOTAL_PCT
