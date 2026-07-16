"""
Combo Signal Engine

Turns a parameter set into entry/exit conditions for one timeframe. This is
THE single definition of indicator signal semantics - the backtest
(optimizer.compute_signals) and the live trader (_eval_params_signals) both
call it, so they can never diverge.

Parameter schema per timeframe `tf`:

Legacy (no IND1_{tf} key) - bit-exact with the original RSI strategy:
    MN1_{tf}, MN2_{tf}       raw Cutler RSI length + SMA smoothing
    Entry_{tf}, Exit_{tf}    enter below / exit above thresholds

Combo (IND1_{tf} present) - mix-and-match indicators:
    IND1_{tf}                indicator name (signals.indicators.INDICATORS)
    IND1_Inv_{tf}            1 to invert the oscillator (buy strength
                             instead of buy weakness)
    MN1_{tf}, MN2_{tf}       leg-1 periods (reused keys)
    Entry_{tf}, Exit_{tf}    leg-1 thresholds (reused keys)
    IND2_{tf}                optional second indicator or "none"
    IND2_P1/P2/Entry/Exit/Inv_{tf}   leg-2 periods/thresholds/invert

Cycle-only (IND1_{tf} == "none") - the strategy trades on the time cycle
alone: entries are always indicator-approved, exits only come from the
cycle turning off, and there is no indicator warm-up.

Entries require EVERY leg's oscillator below its Entry threshold (AND);
exits fire when ANY leg's oscillator is above its Exit threshold (OR).
The time cycle and coarse-timeframe mapping stay in the callers.
"""

from typing import Dict, List, Tuple

import numpy as np

from signals.indicators import indicator_oscillator, indicator_warmup

CYCLE_ONLY = "none"  # sentinel value for IND1_{tf} meaning "no indicators"


def _legs(params: Dict, tf: str) -> List[Tuple[str, int, int, float, float, bool]]:
    """(name, p1, p2, entry, exit, invert) for each active leg"""
    legs = []
    if str(params.get(f"IND1_{tf}", "")) == CYCLE_ONLY:
        # Cycle-only strategy: no indicator legs at all
        return legs
    if f"IND1_{tf}" in params:
        legs.append(
            (
                str(params[f"IND1_{tf}"]),
                int(params[f"MN1_{tf}"]),
                int(params[f"MN2_{tf}"]),
                float(params[f"Entry_{tf}"]),
                float(params[f"Exit_{tf}"]),
                bool(int(params.get(f"IND1_Inv_{tf}", 0))),
            )
        )
        second = str(params.get(f"IND2_{tf}", "none"))
        if second and second != "none":
            legs.append(
                (
                    second,
                    int(params[f"IND2_P1_{tf}"]),
                    int(params[f"IND2_P2_{tf}"]),
                    float(params[f"IND2_Entry_{tf}"]),
                    float(params[f"IND2_Exit_{tf}"]),
                    bool(int(params.get(f"IND2_Inv_{tf}", 0))),
                )
            )
    else:
        # Legacy pure-RSI parameter sets (stored strategies, old CSVs)
        legs.append(
            (
                "rsi",
                int(params[f"MN1_{tf}"]),
                int(params[f"MN2_{tf}"]),
                float(params[f"Entry_{tf}"]),
                float(params[f"Exit_{tf}"]),
                False,
            )
        )
    return legs


def params_warmup(params: Dict, tf: str) -> int:
    """Warm-up bars this parameter set needs on `tf` (max over legs)"""
    if str(params.get(f"IND1_{tf}", "")) == CYCLE_ONLY:
        return 0  # cycle-only: nothing to warm up
    if f"IND1_{tf}" not in params:
        # Legacy path: identical to the original MN1+MN2 warm-up
        if f"MN1_{tf}" not in params:
            return 0
        return int(params[f"MN1_{tf}"]) + int(params[f"MN2_{tf}"])
    return max(
        indicator_warmup(name, p1, p2) for name, p1, p2, _, _, _ in _legs(params, tf)
    )


def evaluate_combo(
    closes: np.ndarray, params: Dict, tf: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Evaluate the indicator combo on a close series.

    Returns:
        (entry_ok, exit_ok, warmup) - boolean arrays over `closes` plus the
        warm-up bar count. Causal: values at bar i use closes[: i + 1] only.
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    entry_ok = np.ones(n, dtype=bool)
    exit_ok = np.zeros(n, dtype=bool)
    warmup = 0

    legacy = f"IND1_{tf}" not in params

    for name, p1, p2, entry_thr, exit_thr, invert in _legs(params, tf):
        if legacy:
            # Bit-exact original chain: raw RSI + SMA smoothing, no clip
            from optimization.metrics import PerformanceMetrics

            osc = PerformanceMetrics.smooth_vectorized(
                PerformanceMetrics.compute_rsi_vectorized(closes, p1), p2
            )
            leg_warmup = p1 + p2
        else:
            osc = indicator_oscillator(name, closes, p1, p2)
            leg_warmup = indicator_warmup(name, p1, p2)

        if invert:
            osc = 100.0 - osc

        entry_ok &= osc < entry_thr
        exit_ok |= osc > exit_thr
        warmup = max(warmup, leg_warmup)

    return entry_ok, exit_ok, warmup
