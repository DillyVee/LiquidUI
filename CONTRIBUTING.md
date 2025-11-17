# Contributing to LiquidUI

Thank you for your interest in contributing to LiquidUI! This document provides guidelines and instructions for contributing to this project.

## 🎯 Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to keep our community respectful and inclusive.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker and Docker Compose (for full stack development)
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Fork and Clone**
   ```bash
   git fork https://github.com/DillyVee/LiquidUI.git
   cd LiquidUI
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   ```

4. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

6. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## 📝 How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS)
   - Relevant logs or screenshots

### Suggesting Features

Feature requests are welcome! Please:
1. Check if the feature has already been suggested
2. Use the feature request template
3. Clearly describe:
   - The problem your feature would solve
   - How it should work
   - Why it would be useful to others
   - Possible implementation approach (optional)

### Submitting Changes

#### Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

   Branch naming conventions:
   - `feature/` - New features
   - `fix/` - Bug fixes
   - `docs/` - Documentation updates
   - `refactor/` - Code refactoring
   - `test/` - Test additions/improvements
   - `chore/` - Maintenance tasks

2. **Make Your Changes**
   - Write clean, readable code
   - Follow the style guide (see below)
   - Add/update tests
   - Update documentation

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=. --cov-report=html

   # Run linting
   black .
   flake8 .
   mypy .
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "type: brief description

   Detailed explanation of changes (if needed)

   Fixes #issue_number"
   ```

   Commit message types:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `style:` - Code style (formatting, no logic change)
   - `refactor:` - Code refactoring
   - `test:` - Adding or updating tests
   - `chore:` - Maintenance, dependencies

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a Pull Request on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/examples (if applicable)

## 💻 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters (not 79)
- **Quotes**: Prefer double quotes for strings
- **Imports**: Organized using isort
  - Standard library
  - Third-party packages
  - Local imports

Example:
```python
"""
Module docstring explaining purpose
"""
import os
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from data_layer.storage import VersionedDataStore
```

### Type Hints

All functions must have type hints:

```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
```

### Documentation

- **Docstrings**: Use Google style
- **Comments**: Explain WHY, not WHAT
- **README**: Update if you add new features

Example docstring:
```python
def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "mean_variance"
) -> Dict[str, float]:
    """
    Optimize portfolio weights.

    Args:
        returns: DataFrame of asset returns (assets as columns)
        method: Optimization method ('mean_variance', 'risk_parity')

    Returns:
        Dictionary of {asset: weight}

    Raises:
        ValueError: If method is not supported

    Example:
        >>> returns = pd.DataFrame(...)
        >>> weights = optimize_portfolio(returns)
        >>> print(weights)
        {'AAPL': 0.4, 'GOOGL': 0.6}
    """
```

### Testing

#### Required Tests

- **Unit Tests**: Test individual functions
- **Integration Tests**: Test component interaction
- **Property Tests**: Use Hypothesis for edge cases

#### Test Structure

```python
# tests/unit/test_backtest.py
import pytest
import pandas as pd
from backtest.engine import BacktestEngine

@pytest.fixture
def sample_data():
    """Fixture for test data"""
    return pd.DataFrame(...)

def test_backtest_basic(sample_data):
    """Test basic backtest execution"""
    engine = BacktestEngine(initial_cash=100000)
    results = engine.run_backtest(sample_data, strategy, 'TEST')

    assert len(results) > 0
    assert 'equity' in results.columns

def test_backtest_no_trades_on_empty_data():
    """Test that empty data produces no trades"""
    engine = BacktestEngine()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        engine.run_backtest(empty_df, strategy, 'TEST')
```

#### Coverage Requirements

- **Minimum**: 80% code coverage
- **Preferred**: 90%+ for critical modules (risk management, execution)
- Run: `pytest tests/ --cov=. --cov-report=html`

### Performance

- Use vectorized operations (NumPy/Pandas) over loops
- Profile before optimizing: `python -m cProfile script.py`
- Document any optimization trade-offs

### Security

- **Never commit**:
  - API keys or passwords
  - .env files with real credentials
  - Database dumps
  - Personal data

- **Always**:
  - Use environment variables for secrets
  - Validate user inputs
  - Sanitize SQL queries (use parameterized queries)
  - Keep dependencies updated

## 📦 Module Guidelines

### Adding New Modules

1. Create module directory: `module_name/`
2. Add `__init__.py` with version
3. Create main module file
4. Add tests in `tests/unit/test_module_name.py`
5. Update documentation

### Module Structure

```
module_name/
├── __init__.py          # Module initialization
├── core.py              # Core functionality
├── utils.py             # Utility functions
└── types.py             # Type definitions
```

## 🔍 Review Process

### What Reviewers Look For

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New code has tests (80%+ coverage)
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance is acceptable
- [ ] Breaking changes are documented

### Review Checklist (for Contributors)

Before requesting review:

- [ ] Code is clean and well-documented
- [ ] Tests added and passing
- [ ] Linters pass (black, flake8, mypy)
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Environment variables used for config
- [ ] CHANGELOG.md updated (if significant change)

## 🎨 Commit Message Guidelines

Format:
```
type(scope): brief description (50 chars or less)

More detailed explanation (if needed). Wrap at 72 characters.
Explain WHAT and WHY, not HOW.

Fixes #123
Closes #456
```

Examples:
```
feat(backtest): add market impact model using Almgren-Chriss

Implements permanent and temporary market impact following
Almgren & Chriss (2001) framework. Includes capacity analysis.

Closes #42
```

```
fix(risk): correct leverage calculation in position limits

Previous calculation used gross instead of net exposure.
This could allow over-leveraging in hedged positions.

Fixes #89
```

## 📚 Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)

## ❓ Questions?

- Open an issue with the `question` label
- Join our discussions on GitHub Discussions
- Email: support@liquidui.dev (if applicable)

## 🙏 Recognition

Contributors will be recognized in:
- README.md Contributors section
- CHANGELOG.md for significant contributions
- Release notes

---

Thank you for contributing to LiquidUI! 🚀
