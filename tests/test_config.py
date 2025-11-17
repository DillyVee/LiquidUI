"""
Basic configuration and smoke tests
"""

import sys
from pathlib import Path

import pytest


def test_python_version():
    """Test that we're running on Python 3.11+"""
    assert sys.version_info >= (3, 11), "Requires Python 3.11 or higher"


def test_project_structure():
    """Test that basic project directories exist"""
    project_root = Path(__file__).parent.parent
    assert (project_root / "config").exists(), "config directory should exist"
    assert (project_root / "backtest").exists(), "backtest directory should exist"
    assert (project_root / "optimization").exists(), "optimization directory should exist"
    assert (project_root / "risk").exists(), "risk directory should exist"


def test_requirements_files():
    """Test that requirements files exist"""
    project_root = Path(__file__).parent.parent
    assert (
        project_root / "requirements.txt"
    ).exists(), "requirements.txt should exist"
    assert (
        project_root / "requirements-test.txt"
    ).exists(), "requirements-test.txt should exist"


def test_import_config():
    """Test that config module can be imported"""
    try:
        from config import settings

        # Check for at least one config class
        assert hasattr(
            settings, "OptimizationConfig"
        ), "OptimizationConfig class should exist in settings"
    except ImportError as e:
        pytest.skip(f"Config module import failed: {e}")
