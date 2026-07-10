"""
Pytest root configuration.

Ensures the repository root is importable (config, data, models,
optimization, signals, trading, gui) regardless of how pytest is invoked.
Bare `pytest tests/` does NOT add the current directory to sys.path the way
`python -m pytest` does, which broke test collection in CI.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Qt must never try to open a real display during tests (CI is headless)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
