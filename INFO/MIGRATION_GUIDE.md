# Migration Guide: From Monolithic to Modular

This guide helps you transition from the original 2000+ line monolithic script to the new modular architecture.

## 📋 What Changed

### File Structure

**BEFORE:**
```
your_script.py  (2000+ lines)
```

**AFTER:**
```
trading_app/
├── main.py                    # Entry point (15 lines)
├── config/settings.py         # All constants (150 lines)
├── data/loader.py             # Data loading (180 lines)
├── optimization/
│   ├── optimizer.py           # Core optimization (400 lines)
│   └── metrics.py             # Calculations (120 lines)
├── trading/alpaca_trader.py   # Live trading (300 lines)
└── gui/
    ├── main_window.py         # UI logic (350 lines)
    └── styles.py              # Styling (40 lines)
```

## 🔄 Key Changes Explained

### 1. Configuration Management

**BEFORE (Hardcoded):**
```python
api_key = "PKHEQYQDXFEGVR6KB2AGRQDGKZ"  # ⚠️ EXPOSED!
```

**AFTER (Environment Variables):**
```python
# config/settings.py
import os

@dataclass
class AlpacaConfig:
    API_KEY: str = os.environ.get('ALPACA_API_KEY', '')
    SECRET_KEY: str = os.environ.get('ALPACA_SECRET_KEY', '')
```

**Action Required:**
```bash
# Set environment variables
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

### 2. Data Loading

**BEFORE (Inline):**
```python
def load_yfinance(self):
    # 100 lines of data loading code mixed with UI
    df = yf.download(...)
    # Process data
    # Update UI
```

**AFTER (Separated):**
```python
# data/loader.py
class DataLoader:
    @classmethod
    def load_yfinance_data(cls, symbol: str):
        # Pure data loading logic
        return df_dict, error_msg

# gui/main_window.py
def load_yfinance(self):
    df_dict, error = DataLoader.load_yfinance_data(symbol)
    # Update UI only
```

### 3. Optimization Logic

**BEFORE (Single Class):**
```python
class MultiTimeframeOptimizer:
    def __init__(self, ...):
        # 50 parameters
        
    def run(self):
        # 300+ lines
        
    def simulate(self):
        # 200+ lines
        
    def calculate_metrics(self):
        # 150+ lines
```

**AFTER (Split Responsibilities):**
```python
# optimization/optimizer.py
class MultiTimeframeOptimizer:
    def run(self):
        # Core optimization only
        
    def simulate_multi_tf(self):
        # Simulation logic
        
# optimization/metrics.py  
class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(eq_curve):
        # Pure calculation
```

### 4. GUI Organization

**BEFORE (One Giant Method):**
```python
def init_ui(self):
    # 400 lines creating all widgets
    layout = QVBoxLayout()
    # ... 50 widgets ...
    # ... 50 connections ...
```

**AFTER (Helper Methods):**
```python
def init_ui(self):
    self._add_data_source_controls(layout)
    self._add_timeframe_controls(layout)
    self._add_optimization_controls(layout)
    # Clear, organized structure

def _add_data_source_controls(self, layout):
    # Focused on one responsibility
```

## 🚀 How to Use the New Code

### Basic Import Pattern

```python
# Old way (everything in one file)
from your_script import MultiTimeframeOptimizer

# New way (clear imports)
from optimization import MultiTimeframeOptimizer
from data import DataLoader
from trading import AlpacaLiveTrader
from config import AlpacaConfig
```

### Running the Application

**Old:**
```python
# Run the giant script
python your_script.py
```

**New:**
```python
# Run the entry point
python main.py

# Or import as a module
from gui import MainWindow
app = QApplication(sys.argv)
window = MainWindow()
window.show()
```

### Customizing Settings

**Old:**
```python
# Edit hardcoded values throughout 2000 lines
trials = 900  # Line 42
position_size = 0.05  # Line 856
stop_loss = 0.02  # Line 1234
```

**New:**
```python
# Edit one file: config/settings.py
@dataclass
class OptimizationConfig:
    DEFAULT_TRIALS: int = 1200  # Changed here

@dataclass
class RiskConfig:
    DEFAULT_POSITION_SIZE: float = 0.03  # Changed here
```

## 🔧 Common Migration Patterns

### Pattern 1: Accessing Configuration

**Before:**
```python
# Scattered throughout code
api_key = "..."
position_size = 0.05
max_dd = 0.50
```

**After:**
```python
from config.settings import AlpacaConfig, RiskConfig

config = AlpacaConfig()
api_key = config.API_KEY

risk = RiskConfig()
position_size = risk.DEFAULT_POSITION_SIZE
```

### Pattern 2: Data Loading

**Before:**
```python
# In MainWindow.__init__
self.load_yf_btn.clicked.connect(self.load_yfinance)

def load_yfinance(self):
    # 100 lines of yfinance logic mixed with UI
```

**After:**
```python
# In MainWindow
def load_yfinance(self):
    df_dict, error = DataLoader.load_yfinance_data(symbol)
    if error:
        self.show_error(error)
    else:
        self.update_data(df_dict)
```

### Pattern 3: Optimization

**Before:**
```python
# Everything in one giant class
optimizer = MultiTimeframeOptimizer(
    param1, param2, ... param20
)
```

**After:**
```python
from optimization import MultiTimeframeOptimizer
from config.settings import IndicatorRanges

ranges = IndicatorRanges()
optimizer = MultiTimeframeOptimizer(
    df_dict=df_dict,
    n_trials=900,
    mn1_range=ranges.MN1_RANGE,
    # Clear parameter names
)
```

### Pattern 4: Live Trading

**Before:**
```python
# Hardcoded credentials in class
class AlpacaLiveTrader:
    def __init__(self):
        self.api_key = "HARDCODED"  # ⚠️
```

**After:**
```python
from trading import AlpacaLiveTrader
from config import AlpacaConfig

config = AlpacaConfig()  # Gets from env vars
trader = AlpacaLiveTrader(
    api_key=config.API_KEY,
    secret_key=config.SECRET_KEY,
    # ...
)
```

## ✅ Checklist for Migration

- [ ] Set environment variables for API keys
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Update any custom modifications to use new module structure
- [ ] Test data loading with `DataLoader`
- [ ] Test optimization with new `MultiTimeframeOptimizer`
- [ ] Verify live trading connects to Alpaca
- [ ] Update any scripts that imported the old monolithic file

## 💡 Benefits You'll Notice

1. **Faster Development**: Change one module without touching others
2. **Easier Testing**: Test individual components in isolation
3. **Better Collaboration**: Multiple developers can work on different modules
4. **Clearer Bugs**: Stack traces point to specific, small files
5. **Simpler Maintenance**: Find code quickly by module responsibility
6. **Safer Credentials**: No more accidental API key commits

## 🐛 Troubleshooting Migration Issues

### Issue: "ModuleNotFoundError: No module named 'config'"

**Cause**: Not running from correct directory

**Fix:**
```bash
# Make sure you're in trading_app directory
cd trading_app
python main.py
```

### Issue: "API keys not loading"

**Cause**: Environment variables not set

**Fix:**
```bash
# Windows
set ALPACA_API_KEY=your_key
python main.py

# Linux/Mac
export ALPACA_API_KEY=your_key
python main.py

# Or use .env file with python-dotenv
```

### Issue: "Import errors between modules"

**Cause**: Relative imports not resolving

**Fix:**
```python
# Use absolute imports
from config.settings import AlpacaConfig  # ✅
from .settings import AlpacaConfig        # ❌ (unless in package)
```

## 📚 Next Steps

1. **Read the Code**: Start with `main.py`, follow the imports
2. **Experiment**: Modify `config/settings.py` and see changes
3. **Extend**: Add new features in appropriate modules
4. **Test**: Write unit tests for individual modules
5. **Document**: Add docstrings to your custom modifications

## 🎓 Learning Resources

- [Python Modules Tutorial](https://docs.python.org/3/tutorial/modules.html)
- [Package Structure Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Clean Architecture in Python](https://www.youtube.com/watch?v=DJtef410XaM)

---

**Questions?** Open an issue or check the README.md for more details!
