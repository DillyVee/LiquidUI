"""
Causal Indicator Library

Ten structurally different indicators behind one uniform interface so the
optimizer can mix and match them per timeframe:

    rsi       - Cutler RSI (momentum oscillator)
    stoch     - Stochastic %K smoothed to %D (range position)
    macd      - MACD histogram (trend acceleration)
    roc       - Rate of change (raw momentum)
    emacross  - Fast/slow EMA spread (trend direction/strength)
    bbz       - Bollinger z-score (mean-reversion stretch)
    cci       - Commodity Channel Index (deviation from typical price)
    trix      - Triple-EMA rate of change histogram (smoothed momentum)
    dpo       - Detrended Price Oscillator, causal variant (cycle swings)
    aroon     - Aroon oscillator (time since high vs time since low)

Every indicator takes exactly two integer periods (p1, p2) and is CAUSAL:
the value at bar i uses only closes[: i + 1]. Raw outputs live on wildly
different scales, so `indicator_oscillator` rank-normalizes them to a
causal rolling percentile in [0, 100]. That gives every indicator the same
threshold semantics the RSI strategy already uses (enter below Entry, exit
above Exit), makes thresholds adaptive to recent history, and lets the
optimizer compare indicators on equal footing.

Warm-up: values before `indicator_warmup(...)` bars are computed from
partial windows and must not generate entries (enforced by the signal
engine, same as the existing RSI warm-up mask).
"""

from typing import Tuple

import numpy as np
import pandas as pd

INDICATORS = (
    "rsi",
    "stoch",
    "macd",
    "roc",
    "emacross",
    "bbz",
    "cci",
    "trix",
    "dpo",
    "aroon",
)

# Human-readable names for GUI display
INDICATOR_LABELS = {
    "rsi": "RSI",
    "stoch": "Stochastic",
    "macd": "MACD",
    "roc": "Rate of Change",
    "emacross": "EMA Cross",
    "bbz": "Bollinger Z",
    "cci": "CCI",
    "trix": "TRIX",
    "dpo": "Detrended Price Osc",
    "aroon": "Aroon Osc",
}

# Rolling window used for percentile-rank normalization, as a multiple of
# the indicator's own length (floored so tiny lengths still get context)
_RANK_WINDOW_MIN = 50
_RANK_WINDOW_MULT = 2


def _sma(arr: np.ndarray, length: int) -> np.ndarray:
    """Simple moving average with expanding partial windows at the start"""
    length = max(1, int(length))
    cumsum = np.cumsum(arr, dtype=np.float64)
    out = np.empty_like(cumsum)
    out[length - 1 :] = (
        cumsum[length - 1 :] - np.concatenate([[0.0], cumsum[:-length]])
    ) / length
    out[: length - 1] = cumsum[: length - 1] / np.arange(1, length)
    return out


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (causal)"""
    return (
        pd.Series(arr).ewm(span=max(1, int(span)), adjust=False).mean().to_numpy()
    )


def _rolling_min_max(arr: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    s = pd.Series(arr)
    lo = s.rolling(max(1, int(length)), min_periods=1).min().to_numpy()
    hi = s.rolling(max(1, int(length)), min_periods=1).max().to_numpy()
    return lo, hi


def _rolling_std(arr: np.ndarray, length: int) -> np.ndarray:
    return (
        pd.Series(arr)
        .rolling(max(2, int(length)), min_periods=2)
        .std()
        .fillna(0.0)
        .to_numpy()
    )


def _rolling_mad(arr: np.ndarray, length: int) -> np.ndarray:
    """Rolling mean absolute deviation from the rolling mean (causal).
    Early bars use the expanding window of available history."""
    length = max(2, int(length))
    n = len(arr)
    out = np.zeros(n)

    head = min(length - 1, n)
    for i in range(head):
        window = arr[: i + 1]
        out[i] = np.mean(np.abs(window - np.mean(window)))

    if n >= length:
        from numpy.lib.stride_tricks import sliding_window_view

        views = sliding_window_view(arr, length)
        means = views.mean(axis=1, keepdims=True)
        out[length - 1 :] = np.abs(views - means).mean(axis=1)
    return out


def _rolling_argmax_age(arr: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bars since the rolling-window high and low (0 = this bar). Causal;
    early bars use the expanding window."""
    length = max(2, int(length))
    n = len(arr)
    age_hi = np.zeros(n)
    age_lo = np.zeros(n)

    head = min(length - 1, n)
    for i in range(head):
        window = arr[: i + 1]
        age_hi[i] = i - int(np.argmax(window))
        age_lo[i] = i - int(np.argmin(window))

    if n >= length:
        from numpy.lib.stride_tricks import sliding_window_view

        views = sliding_window_view(arr, length)
        # argmax/argmin return the FIRST occurrence; position length-1 is
        # the current bar, so age = (length - 1) - position
        age_hi[length - 1 :] = (length - 1) - views.argmax(axis=1)
        age_lo[length - 1 :] = (length - 1) - views.argmin(axis=1)
    return age_hi, age_lo


def compute_indicator(name: str, closes: np.ndarray, p1: int, p2: int) -> np.ndarray:
    """
    Raw (un-normalized) indicator series. Causal: out[i] depends only on
    closes[: i + 1].

    Args:
        name: One of INDICATORS
        closes: Close price array
        p1: Primary period (main lookback)
        p2: Secondary period (smoothing / signal line)
    """
    closes = np.asarray(closes, dtype=np.float64)
    p1 = max(2, int(p1))
    p2 = max(1, int(p2))

    if name == "rsi":
        from optimization.metrics import PerformanceMetrics

        rsi = PerformanceMetrics.compute_rsi_vectorized(closes, p1)
        return PerformanceMetrics.smooth_vectorized(rsi, p2)

    if name == "stoch":
        lo, hi = _rolling_min_max(closes, p1)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        k = np.where(hi - lo > 1e-12, (closes - lo) / span * 100.0, 50.0)
        return _sma(k, p2)

    if name == "macd":
        fast = _ema(closes, p1)
        slow = _ema(closes, p1 * 2)
        macd_line = (fast - slow) / np.where(closes > 1e-12, closes, 1.0) * 100.0
        signal = _ema(macd_line, p2)
        return macd_line - signal  # histogram

    if name == "roc":
        roc = np.zeros_like(closes)
        roc[p1:] = (closes[p1:] / np.where(closes[:-p1] > 1e-12, closes[:-p1], 1.0) - 1.0) * 100.0
        return _sma(roc, p2)

    if name == "emacross":
        fast = _ema(closes, p2)
        slow = _ema(closes, p1 + p2)  # guaranteed slower than fast
        return (fast / np.where(slow > 1e-12, slow, 1.0) - 1.0) * 100.0

    if name == "bbz":
        mid = _sma(closes, p1)
        std = _rolling_std(closes, p1)
        z = np.divide(closes - mid, std, out=np.zeros_like(closes), where=std > 1e-12)
        return _sma(z, p2)

    if name == "cci":
        # Close-based CCI: (close - SMA) / (0.015 * rolling mean |deviation|)
        mid = _sma(closes, p1)
        mad = _rolling_mad(closes, p1)
        denom = np.where(mad > 1e-12, 0.015 * mad, 1.0)
        cci = np.where(mad > 1e-12, (closes - mid) / denom, 0.0)
        return _sma(cci, p2)

    if name == "trix":
        # Triple-smoothed EMA 1-bar rate of change, minus its signal line
        e3 = _ema(_ema(_ema(closes, p1), p1), p1)
        trix = np.zeros_like(e3)
        prev = np.where(np.abs(e3[:-1]) > 1e-12, e3[:-1], 1.0)
        trix[1:] = (e3[1:] / prev - 1.0) * 100.0
        signal = _ema(trix, p2)
        return trix - signal

    if name == "dpo":
        # Causal detrended price oscillator: price vs the SMA from half a
        # period ago (classic DPO displacement, past-shifted so out[i]
        # never reads future bars)
        shift = p1 // 2 + 1
        sma = _sma(closes, p1)
        detrended = np.zeros_like(closes)
        detrended[shift:] = closes[shift:] - sma[:-shift]
        return _sma(detrended, p2)

    if name == "aroon":
        # Aroon oscillator: how recently the rolling high vs low printed,
        # rescaled from [-100, 100] to [0, 100] (bounded, absolute levels)
        up, down = _rolling_argmax_age(closes, p1)
        aroon_up = (p1 - up) / p1 * 100.0
        aroon_down = (p1 - down) / p1 * 100.0
        osc = (aroon_up - aroon_down + 100.0) / 2.0
        return _sma(osc, p2)

    raise ValueError(f"Unknown indicator '{name}' (choose from {INDICATORS})")


def rank_normalize(values: np.ndarray, window: int) -> np.ndarray:
    """
    Causal rolling percentile rank in [0, 100]: out[i] is the percentage of
    the last `window` values (ending at and including i) that are <= x[i].
    Early bars use the expanding window of whatever history exists.
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    window = max(2, int(window))
    out = np.empty(n)

    head = min(window - 1, n)
    for i in range(head):
        out[i] = np.mean(x[: i + 1] <= x[i]) * 100.0

    if n >= window:
        from numpy.lib.stride_tricks import sliding_window_view

        views = sliding_window_view(x, window)  # (n - window + 1, window)
        out[window - 1 :] = (views <= views[:, -1:]).mean(axis=1) * 100.0

    return out


def _rank_window(p1: int, p2: int) -> int:
    return max(_RANK_WINDOW_MIN, _RANK_WINDOW_MULT * (int(p1) + int(p2)))


# Bounded indicators are natively 0-100 and keep their classic absolute
# semantics; unbounded ones are rank-normalized to a causal rolling
# percentile so thresholds mean the same thing everywhere
_BOUNDED = ("rsi", "stoch", "aroon")


def indicator_warmup(name: str, p1: int, p2: int) -> int:
    """Bars before the oscillator is fully warmed up (indicator lookback
    plus, for unbounded indicators, the rank-normalization window)"""
    p1, p2 = int(p1), int(p2)
    if name == "macd":
        lookback = p1 * 2 + p2
    elif name == "trix":
        lookback = p1 * 3 + p2
    elif name == "dpo":
        lookback = p1 + p1 // 2 + 1 + p2
    else:
        lookback = p1 + p2
    if name in _BOUNDED:
        return lookback
    return lookback + _rank_window(p1, p2)


def indicator_oscillator(
    name: str, closes: np.ndarray, p1: int, p2: int
) -> np.ndarray:
    """
    Uniform [0, 100] oscillator with enter-below / exit-above threshold
    semantics for every indicator:

    - rsi/stoch: native 0-100 values (classic absolute levels)
    - macd/roc/emacross/bbz: causal rolling percentile rank ("how extreme
      is this reading vs its own recent history")
    """
    raw = compute_indicator(name, closes, p1, p2)
    if name in _BOUNDED:
        return np.clip(raw, 0.0, 100.0)
    return rank_normalize(raw, _rank_window(p1, p2))
