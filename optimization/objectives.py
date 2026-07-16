"""
Optimization Objective Registry

One place that defines what the optimizer can maximize. Every objective
reads from the metric dict produced by PerformanceMetrics.calculate_metrics
so the number shown in the GUI is exactly the number that was optimized.

All objectives are formulated as MAXIMIZE. Objectives that need trade-level
statistics (win rate, expectancy, profit factor) require the metric dict to
have been built with per-trade returns; they fall back to 0.0 when the
trade-level keys are missing.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class ObjectiveSpec:
    """One optimization goal"""

    name: str  # registry key, stable identifier saved with results
    label: str  # human-readable name for the GUI
    metric_key: str  # key in the metrics dict this objective reads
    description: str
    fmt: str = ".3f"  # display format for the score


# Ordered so the GUI shows risk-adjusted goals first
OBJECTIVES: Dict[str, ObjectiveSpec] = {
    spec.name: spec
    for spec in (
        ObjectiveSpec(
            "sharpe",
            "Sharpe Ratio",
            "Sharpe_Ratio",
            "Annualized mean return over volatility (rf=0)",
            ".2f",
        ),
        ObjectiveSpec(
            "sortino",
            "Sortino Ratio",
            "Sortino_Ratio",
            "Annualized mean return over downside deviation",
            ".2f",
        ),
        ObjectiveSpec(
            "calmar",
            "Calmar Ratio",
            "Calmar_Ratio",
            "CAGR over maximum drawdown magnitude",
            ".2f",
        ),
        ObjectiveSpec(
            "total_return",
            "Total Return %",
            "Percent_Gain_%",
            "Raw percent gain over the backtest",
            ".2f",
        ),
        ObjectiveSpec(
            "profit_factor",
            "Profit Factor",
            "Profit_Factor",
            "Gross profit over gross loss across closed trades",
            ".2f",
        ),
        ObjectiveSpec(
            "win_rate",
            "Win Rate %",
            "Win_Rate_%",
            "Percentage of closed trades that were profitable",
            ".1f",
        ),
        ObjectiveSpec(
            "expectancy",
            "Expectancy %/trade",
            "Expectancy_%",
            "Average percent return per closed trade",
            ".3f",
        ),
    )
}

DEFAULT_OBJECTIVE = "sharpe"

# Sensible default set for the auto-build pipeline: three structurally
# different goals so the strategy book holds diverse behavior
AUTO_BUILD_DEFAULTS = ("sharpe", "sortino", "calmar")


def get_objective(name: str) -> ObjectiveSpec:
    """Look up an objective, raising a helpful error for unknown names"""
    try:
        return OBJECTIVES[name]
    except KeyError:
        raise ValueError(
            f"Unknown objective '{name}' (choose from {sorted(OBJECTIVES)})"
        ) from None


def objective_score(name: str, metrics: Optional[Dict[str, float]]) -> float:
    """
    Score a metric dict under the named objective. Returns 0.0 for missing
    metrics (failed simulations score as 'no edge', not as an error).
    """
    spec = get_objective(name)
    if not metrics:
        return 0.0
    value = metrics.get(spec.metric_key)
    if value is None:
        return 0.0
    value = float(value)
    if not (value == value and abs(value) != float("inf")):  # NaN/inf guard
        return 0.0
    return value


def trade_confidence_multiplier(trade_count: int) -> float:
    """
    Scale scores by statistical confidence in the trade sample: fewer than
    10 trades is noise (score 0), 10-30 ramps linearly, 30+ is fully
    trusted. Applied uniformly to every objective so 'few lucky trades'
    can never win an optimization under any goal.
    """
    if trade_count < 10:
        return 0.0
    if trade_count < 30:
        return (trade_count - 10) / 20.0
    return 1.0


def scored(
    name: str, metrics: Optional[Dict[str, float]], trade_count: int
) -> Tuple[float, float]:
    """(raw objective value, confidence-scaled score) for one trial"""
    raw = objective_score(name, metrics)
    return raw, raw * trade_confidence_multiplier(int(trade_count))


ObjectiveFn = Callable[[Optional[Dict[str, float]], int], float]
