"""
Anti-Overfitting Validation Toolkit

Statistical defenses against selection bias - the reason optimized
backtests look better than live results:

1. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): PSR measured
   against the Sharpe you'd expect the BEST of N random trials to show.
   The optimizer burns thousands of trials; the best one is partly luck,
   and DSR prices that luck in.

2. CSCV Probability of Backtest Overfitting (Bailey, Borwein, Lopez de
   Prado & Zhu, 2016): split history into S blocks, and for every
   train/test block combination check whether the config that won in-train
   also wins out-of-train. PBO = how often the in-sample winner lands in
   the bottom half out-of-sample.

3. validate_candidate(): one gated verdict combining DSR, PBO, and a
   minimum trade count - used before a strategy may enter the book.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from optimization.psr_composite import PSRCalculator

_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """
    Expected maximum Sharpe of N unskilled trials (annualized units),
    E[max] ~= sigma_trials * ((1-gamma)*z(1-1/N) + gamma*z(1-1/(N*e)))
    """
    sharpes = np.asarray(
        [s for s in trial_sharpes if np.isfinite(s)], dtype=np.float64
    )
    n = len(sharpes)
    if n < 2:
        return 0.0

    sigma = float(np.std(sharpes, ddof=1))
    if sigma <= 1e-12:
        return 0.0

    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sigma * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    trial_sharpes: Sequence[float],
    annualization_factor: float = 252.0,
    trade_count: Optional[int] = None,
) -> float:
    """
    DSR: probability the candidate's Sharpe beats the best-of-N-noise
    benchmark rather than zero. Uses the same PSR machinery (skew/kurtosis
    aware, trade-count effective sample size), just with a benchmark that
    accounts for how many trials the optimizer tried.

    Args:
        returns: Per-bar returns of the CANDIDATE strategy
        trial_sharpes: Annualized Sharpe of every optimization trial
        annualization_factor: Periods per year
        trade_count: Trades behind the return series (effective n)
    """
    benchmark = expected_max_sharpe(trial_sharpes)
    return PSRCalculator.calculate_psr(
        returns,
        benchmark_sharpe=benchmark,
        annualization_factor=annualization_factor,
        trade_count=trade_count,
    )


def cscv_pbo(returns_matrix: np.ndarray, n_blocks: int = 10) -> float:
    """
    Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation.

    Args:
        returns_matrix: (T, K) per-bar returns of K candidate configurations
            over the same T bars
        n_blocks: Even number of contiguous time blocks (C(S, S/2) splits)

    Returns:
        PBO in [0, 1]: fraction of splits where the in-sample winner ranks
        in the bottom half out-of-sample. ~0.5 = pure luck; small = robust.
    """
    m = np.asarray(returns_matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T, K)")
    t_bars, n_cfg = m.shape
    if n_cfg < 2:
        raise ValueError("need at least 2 configurations")
    n_blocks = int(n_blocks)
    if n_blocks % 2:
        n_blocks += 1
    if t_bars < n_blocks * 10:
        n_blocks = max(4, (t_bars // 10) // 2 * 2)

    # Per-block sufficient statistics so any block-union Sharpe is exact
    bounds = np.linspace(0, t_bars, n_blocks + 1, dtype=int)
    n_b = np.diff(bounds).astype(np.float64)  # (S,)
    s1 = np.array([m[a:b].sum(axis=0) for a, b in zip(bounds[:-1], bounds[1:])])
    s2 = np.array([(m[a:b] ** 2).sum(axis=0) for a, b in zip(bounds[:-1], bounds[1:])])

    def union_sharpe(block_idx) -> np.ndarray:
        n = n_b[list(block_idx)].sum()
        u1 = s1[list(block_idx)].sum(axis=0)
        u2 = s2[list(block_idx)].sum(axis=0)
        mean = u1 / n
        var = np.maximum((u2 - u1**2 / n) / max(n - 1.0, 1.0), 1e-18)
        return mean / np.sqrt(var)

    all_blocks = set(range(n_blocks))
    logits = []
    for train in combinations(range(n_blocks), n_blocks // 2):
        test = tuple(all_blocks - set(train))
        sr_train = union_sharpe(train)
        sr_test = union_sharpe(test)

        winner = int(np.argmax(sr_train))
        # Relative rank of the in-sample winner out-of-sample, in (0, 1)
        rank = stats.rankdata(sr_test)[winner] / (n_cfg + 1.0)
        logits.append(np.log(rank / (1.0 - rank)))

    logits = np.asarray(logits)
    return float(np.mean(logits <= 0.0))


@dataclass
class ValidationReport:
    """Verdict of the anti-overfit gates for one candidate strategy"""

    passed: bool
    dsr: float
    pbo: Optional[float]
    n_trials: int
    trade_count: int
    sharpe: float
    oos_sharpe: Optional[float] = None
    oos_trades: Optional[int] = None
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Overfit check: {verdict}",
            f"  Deflated Sharpe Ratio: {self.dsr:.3f} "
            f"(deflated for {self.n_trials} optimization trials)",
        ]
        if self.pbo is not None:
            lines.append(f"  Prob. of Backtest Overfitting (CSCV): {self.pbo:.2f}")
        lines.append(f"  Trades: {self.trade_count}  |  Sharpe: {self.sharpe:.2f}")
        if self.oos_sharpe is not None:
            oos = f"  Held-out OOS Sharpe: {self.oos_sharpe:.2f}"
            if self.oos_trades is not None:
                oos += f"  ({self.oos_trades} OOS trades)"
            lines.append(oos)
        for r in self.reasons:
            lines.append(f"  ⚠ {r}")
        return "\n".join(lines)


def validate_candidate(
    simulate_fn: Callable,
    params: Dict,
    rival_params: Sequence[Dict],
    trial_sharpes: Sequence[float],
    annualization_factor: float = 252.0,
    dsr_threshold: float = 0.90,
    pbo_threshold: float = 0.50,
    min_trades: int = 20,
    oos_sharpe: Optional[float] = None,
    min_oos_sharpe: float = 0.0,
    oos_trades: Optional[int] = None,
    min_oos_trades: int = 0,
) -> ValidationReport:
    """
    Run the anti-overfit gates on a candidate parameter set.

    Args:
        simulate_fn: params -> (equity_curve, trade_count); typically
            optimizer.simulate_multi_tf
        params: The candidate
        rival_params: Other top parameter sets from the same optimization
            (needed for CSCV; skipped when fewer than 4 rivals)
        trial_sharpes: Annualized Sharpe of every trial the optimizer ran
        oos_sharpe: Optional true held-out Sharpe (from the live
            re-optimizer's untouched validation window)
        oos_trades: Optional count of trades ENTERED inside the held-out
            window. A flat holdout scores Sharpe 0.0, which passes a
            `>= 0` threshold vacuously - but zero OOS trades is absence
            of evidence, not evidence of harmlessness (and is the
            signature of a calendar cycle that memorized in-sample
            rallies). Gate it separately.
        min_oos_trades: Minimum OOS trades required when oos_trades is
            provided (0 disables the check)
    """
    reasons: List[str] = []

    eq_curve, trade_count = simulate_fn(params)
    if eq_curve is None or len(eq_curve) < 50:
        return ValidationReport(
            passed=False, dsr=0.0, pbo=None, n_trials=len(trial_sharpes),
            trade_count=0, sharpe=0.0, reasons=["simulation produced no usable curve"],
        )

    returns = np.diff(eq_curve) / eq_curve[:-1]
    returns = returns[np.isfinite(returns)]

    mean, std = np.mean(returns), np.std(returns, ddof=1)
    sharpe = float(mean / std * np.sqrt(annualization_factor)) if std > 1e-12 else 0.0

    dsr = deflated_sharpe_ratio(
        returns, trial_sharpes, annualization_factor, trade_count
    )
    if dsr < dsr_threshold:
        reasons.append(
            f"DSR {dsr:.3f} < {dsr_threshold} - performance is consistent "
            "with being the luckiest of the trials"
        )

    pbo = None
    rivals = [p for p in rival_params if p]
    if len(rivals) >= 4:
        curves = [returns]
        for rp in rivals:
            rc, _ = simulate_fn(rp)
            if rc is not None and len(rc) == len(eq_curve):
                r = np.diff(rc) / rc[:-1]
                curves.append(np.where(np.isfinite(r), r, 0.0))
        if len(curves) >= 5:
            matrix = np.column_stack([c[: len(returns)] for c in curves])
            pbo = cscv_pbo(matrix)
            if pbo > pbo_threshold:
                reasons.append(
                    f"PBO {pbo:.2f} > {pbo_threshold} - in-sample winners "
                    "don't stay winners out-of-sample"
                )

    if trade_count < min_trades:
        reasons.append(f"only {trade_count} trades (< {min_trades})")

    if oos_sharpe is not None and oos_sharpe < min_oos_sharpe:
        reasons.append(
            f"held-out OOS Sharpe {oos_sharpe:.2f} < {min_oos_sharpe:.2f}"
        )

    if (
        oos_trades is not None
        and min_oos_trades > 0
        and oos_trades < min_oos_trades
    ):
        reasons.append(
            f"only {oos_trades} trades entered in the held-out window "
            f"(< {min_oos_trades}) - no out-of-sample evidence"
        )

    return ValidationReport(
        passed=len(reasons) == 0,
        dsr=float(dsr),
        pbo=pbo,
        n_trials=len(trial_sharpes),
        trade_count=int(trade_count),
        sharpe=sharpe,
        oos_sharpe=oos_sharpe,
        oos_trades=oos_trades,
        reasons=reasons,
    )
