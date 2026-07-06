"""
Bayesian Online Change Point Detection (BOCPD) for Market Regimes

Implements the Adams & MacKay (2007) algorithm on strategy-timeframe returns
using a Normal-Inverse-Gamma conjugate prior (Student-t predictive), the
standard Bayesian change point method in quantitative finance.

Pipeline:
  1. BOCPDDetector.detect()      - find change points in the return series
  2. classify_segments()         - label each post-shift segment as a regime
                                   (Bull/Bear x Quiet/Volatile) from its
                                   annualized drift and volatility
  3. RegimeStrategyAnalyzer      - break the strategy's performance down by
                                   regime type (return, Sharpe, drawdown,
                                   trades, win rate vs buy & hold)

The detector is online/causal: the run-length posterior at bar t uses only
bars <= t. Segment regime labels are assigned ex post from full-segment
statistics, which is appropriate for performance attribution (not for
generating tradable signals).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REGIME_TYPES = ["Bull-Quiet", "Bull-Volatile", "Bear-Quiet", "Bear-Volatile"]

REGIME_COLORS = {
    "Bull-Quiet": "#1b5e20",
    "Bull-Volatile": "#7cb342",
    "Bear-Quiet": "#b71c1c",
    "Bear-Volatile": "#ff7043",
}


@dataclass
class RegimeSegment:
    """A contiguous market segment between detected change points"""

    start_idx: int
    end_idx: int  # exclusive
    regime: str
    ann_return: float
    ann_vol: float

    @property
    def n_bars(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class BOCPDResult:
    """Output of BOCPD detection"""

    changepoints: List[int]  # bar indices where a new segment starts
    cp_probability: np.ndarray  # P(run length <= 1) at each bar
    map_run_length: np.ndarray  # MAP run length at each bar


class BOCPDDetector:
    """
    Bayesian Online Change Point Detection (Adams & MacKay, 2007)

    Observation model: Normal with unknown mean and variance under a
    Normal-Inverse-Gamma prior, giving a Student-t posterior predictive.
    A constant hazard H = 1/hazard_lambda encodes the prior expectation
    that regimes last ~hazard_lambda bars.
    """

    def __init__(
        self,
        hazard_lambda: float = 250.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: Optional[float] = None,
        min_segment_bars: int = 20,
        prune_threshold: float = 1e-10,
    ):
        """
        Args:
            hazard_lambda: Prior expected regime length in bars
            mu0, kappa0, alpha0, beta0: Normal-Inverse-Gamma prior. beta0
                defaults to a scale matched to the data (set in detect())
            min_segment_bars: Minimum bars between reported change points
            prune_threshold: Drop run-length hypotheses below this mass
        """
        self.hazard_lambda = float(hazard_lambda)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = beta0
        self.min_segment_bars = int(min_segment_bars)
        self.prune_threshold = float(prune_threshold)

    def detect(self, returns: np.ndarray) -> BOCPDResult:
        """
        Run BOCPD over a return series.

        Args:
            returns: 1-D array of per-bar returns

        Returns:
            BOCPDResult with change point indices (into the returns array)
        """
        x_all = np.asarray(returns, dtype=np.float64)
        x_all = np.where(np.isfinite(x_all), x_all, 0.0)
        n = len(x_all)
        if n < 2 * self.min_segment_bars:
            return BOCPDResult([], np.zeros(n), np.zeros(n))

        # Match the prior scale to the data so the detector works for any
        # bar frequency (daily vs 5-min returns differ by orders of
        # magnitude). Calibrated from an initial window only, so adding
        # future data never changes earlier posterior values (causality)
        beta0 = self.beta0
        if beta0 is None:
            calib = x_all[: min(len(x_all), 100)]
            sample_var = float(np.var(calib))
            beta0 = max(sample_var * self.alpha0, 1e-12)

        hazard = 1.0 / self.hazard_lambda

        # Run-length hypotheses (parallel arrays)
        r_probs = np.array([1.0])
        run_lengths = np.array([0], dtype=np.int64)
        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([beta0])

        cp_probability = np.zeros(n)
        map_run_length = np.zeros(n, dtype=np.int64)

        for t in range(n):
            x = x_all[t]

            # Student-t posterior predictive for each run-length hypothesis
            df = 2.0 * alpha
            scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
            pred = stats.t.pdf(x, df, loc=mu, scale=scale)
            pred = np.maximum(pred, 1e-300)

            # Growth (no change) and change point (reset) masses
            growth = r_probs * pred * (1.0 - hazard)
            cp_mass = float(np.sum(r_probs * pred * hazard))

            # New run-length distribution
            r_probs = np.concatenate([[cp_mass], growth])
            run_lengths = np.concatenate([[0], run_lengths + 1])

            # Posterior parameter updates (prepend prior for run length 0)
            mu_new = (kappa * mu + x) / (kappa + 1.0)
            beta_new = beta + kappa * (x - mu) ** 2 / (2.0 * (kappa + 1.0))
            mu = np.concatenate([[self.mu0], mu_new])
            kappa = np.concatenate([[self.kappa0], kappa + 1.0])
            alpha = np.concatenate([[self.alpha0], alpha + 0.5])
            beta = np.concatenate([[beta0], beta_new])

            # Normalize and prune negligible hypotheses
            total = r_probs.sum()
            if total <= 0 or not np.isfinite(total):
                # Numerical breakdown - reset to a fresh run
                r_probs = np.array([1.0])
                run_lengths = np.array([0], dtype=np.int64)
                mu = np.array([self.mu0])
                kappa = np.array([self.kappa0])
                alpha = np.array([self.alpha0])
                beta = np.array([beta0])
            else:
                r_probs = r_probs / total
                keep = r_probs > self.prune_threshold
                keep[0] = True  # always keep the reset hypothesis
                r_probs = r_probs[keep]
                r_probs = r_probs / r_probs.sum()
                run_lengths = run_lengths[keep]
                mu = mu[keep]
                kappa = kappa[keep]
                alpha = alpha[keep]
                beta = beta[keep]

            cp_probability[t] = float(np.sum(r_probs[run_lengths <= 1]))
            map_run_length[t] = int(run_lengths[np.argmax(r_probs)])

        changepoints = self._extract_changepoints(map_run_length)
        return BOCPDResult(changepoints, cp_probability, map_run_length)

    def _extract_changepoints(self, map_run_length: np.ndarray) -> List[int]:
        """
        Turn the MAP run-length trajectory into change point indices.

        A sharp drop in the MAP run length at bar t means the posterior
        believes a new regime started at bar t - run_length[t].
        """
        changepoints: List[int] = []
        n = len(map_run_length)

        for t in range(1, n):
            drop = map_run_length[t - 1] + 1 - map_run_length[t]
            if drop > max(2, self.min_segment_bars // 2):
                cp = int(t - map_run_length[t])
                cp = max(0, min(cp, n - 1))
                if not changepoints or cp - changepoints[-1] >= self.min_segment_bars:
                    if cp >= self.min_segment_bars:
                        changepoints.append(cp)

        return changepoints


def classify_segments(
    returns: np.ndarray,
    changepoints: Sequence[int],
    annualization_factor: float = 252.0,
) -> List[RegimeSegment]:
    """
    Split the return series at change points and classify each segment.

    Regimes: Bull/Bear from the sign of annualized drift, Quiet/Volatile
    from segment volatility vs the full-sample volatility.

    Args:
        returns: Per-bar return series (same array the detector saw)
        changepoints: Segment start indices from BOCPDDetector
        annualization_factor: Periods per year for the bar frequency

    Returns:
        List of RegimeSegment covering every bar exactly once
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    if n == 0:
        return []

    overall_vol = float(np.std(returns)) * np.sqrt(annualization_factor)

    boundaries = [0] + [int(c) for c in changepoints if 0 < int(c) < n] + [n]
    boundaries = sorted(set(boundaries))

    segments: List[RegimeSegment] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg = returns[start:end]
        ann_ret = float(np.mean(seg)) * annualization_factor
        ann_vol = float(np.std(seg)) * np.sqrt(annualization_factor)

        direction = "Bull" if ann_ret >= 0 else "Bear"
        vol_state = "Volatile" if ann_vol >= overall_vol else "Quiet"

        segments.append(
            RegimeSegment(
                start_idx=start,
                end_idx=end,
                regime=f"{direction}-{vol_state}",
                ann_return=ann_ret,
                ann_vol=ann_vol,
            )
        )

    return segments


def label_bars(n_bars: int, segments: Sequence[RegimeSegment]) -> np.ndarray:
    """Per-bar regime label array from segments"""
    labels = np.empty(n_bars, dtype=object)
    labels[:] = "Unclassified"
    for seg in segments:
        labels[seg.start_idx : min(seg.end_idx, n_bars)] = seg.regime
    return labels


class RegimeStrategyAnalyzer:
    """Break strategy performance down by detected regime type"""

    @staticmethod
    def analyze(
        equity_curve: np.ndarray,
        prices: np.ndarray,
        datetimes: np.ndarray,
        segments: Sequence[RegimeSegment],
        trade_log: Optional[pd.DataFrame] = None,
        annualization_factor: float = 252.0,
    ) -> pd.DataFrame:
        """
        Per-regime performance table.

        Args:
            equity_curve: Strategy equity per bar (aligned with prices)
            prices: Close prices per bar
            datetimes: Bar timestamps (aligned), used to assign trades
            segments: Regime segments from classify_segments()
            trade_log: Optional trade log with Entry_Date / Percent_Change
            annualization_factor: Periods per year

        Returns:
            DataFrame with one row per regime type present in the data
        """
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        n = len(equity_curve)

        labels = label_bars(n, segments)

        strat_returns = np.zeros(n)
        strat_returns[1:] = np.diff(equity_curve) / equity_curve[:-1]
        bh_returns = np.zeros(n)
        bh_returns[1:] = np.diff(prices) / prices[:-1]
        strat_returns = np.where(np.isfinite(strat_returns), strat_returns, 0.0)
        bh_returns = np.where(np.isfinite(bh_returns), bh_returns, 0.0)

        # Assign each trade to the regime active at its entry
        trade_regimes: List[str] = []
        trade_pnls: List[float] = []
        if trade_log is not None and len(trade_log) > 0:
            dt_index = pd.DatetimeIndex(pd.to_datetime(datetimes))
            entries = pd.to_datetime(pd.DataFrame(trade_log)["Entry_Date"])
            pnls = pd.DataFrame(trade_log)["Percent_Change"].astype(float)
            positions = dt_index.searchsorted(entries, side="right") - 1
            for pos, pnl in zip(positions, pnls):
                pos = int(np.clip(pos, 0, n - 1))
                trade_regimes.append(str(labels[pos]))
                trade_pnls.append(float(pnl))

        rows = []
        for regime in REGIME_TYPES:
            mask = labels == regime
            bars = int(mask.sum())
            if bars == 0:
                continue

            r_strat = strat_returns[mask]
            r_bh = bh_returns[mask]

            total_ret = float(np.prod(1.0 + r_strat) - 1.0) * 100
            bh_ret = float(np.prod(1.0 + r_bh) - 1.0) * 100

            std = float(np.std(r_strat, ddof=1)) if bars > 2 else 0.0
            sharpe = (
                float(np.mean(r_strat)) / std * np.sqrt(annualization_factor)
                if std > 1e-12
                else 0.0
            )

            # Drawdown of the compounded within-regime sub-curve
            sub_curve = np.cumprod(1.0 + r_strat)
            peak = np.maximum.accumulate(sub_curve)
            max_dd = float(np.min(sub_curve / peak - 1.0)) * 100 if bars > 1 else 0.0

            regime_trades = [p for rg, p in zip(trade_regimes, trade_pnls) if rg == regime]
            n_trades = len(regime_trades)
            n_wins = sum(1 for p in regime_trades if p > 0)

            rows.append(
                {
                    "Regime": regime,
                    "Bars": bars,
                    "Time_%": round(bars / n * 100, 1),
                    "Strategy_Return_%": round(total_ret, 2),
                    "BuyHold_Return_%": round(bh_ret, 2),
                    "Ann_Sharpe": round(sharpe, 2),
                    "Max_DD_%": round(max_dd, 2),
                    "Trades": n_trades,
                    "Win_Rate_%": round(n_wins / n_trades * 100, 1) if n_trades else 0.0,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def build_report(
        summary: pd.DataFrame,
        segments: Sequence[RegimeSegment],
        datetimes: np.ndarray,
        ticker: str = "",
    ) -> str:
        """Human-readable text report of the per-regime breakdown"""
        dt = pd.DatetimeIndex(pd.to_datetime(datetimes))
        lines = [
            "=" * 72,
            f"REGIME BREAKDOWN (Bayesian Online Change Point Detection){' - ' + ticker if ticker else ''}",
            "=" * 72,
            "",
            f"Detected change points: {max(len(segments) - 1, 0)}",
            f"Segments: {len(segments)}",
            "",
            "PER-REGIME STRATEGY PERFORMANCE",
            "-" * 72,
        ]

        if summary.empty:
            lines.append("No classified regime bars.")
        else:
            lines.append(summary.to_string(index=False))

        lines += ["", "SEGMENTS", "-" * 72]
        for i, seg in enumerate(segments, 1):
            start = dt[seg.start_idx].date()
            end = dt[min(seg.end_idx, len(dt)) - 1].date()
            lines.append(
                f"{i:>3}. {start} to {end}  {seg.regime:<14} "
                f"drift {seg.ann_return * 100:+7.1f}%/yr  vol {seg.ann_vol * 100:6.1f}%/yr  "
                f"({seg.n_bars} bars)"
            )

        lines.append("=" * 72)
        return "\n".join(lines)


def plot_regimes(
    prices: np.ndarray,
    equity_curve: np.ndarray,
    datetimes: np.ndarray,
    segments: Sequence[RegimeSegment],
    cp_probability: Optional[np.ndarray] = None,
    ticker: str = "",
    figure=None,
):
    """
    Plot price + strategy equity with regime shading and change points.

    Args:
        figure: Optional existing matplotlib Figure to draw into (cleared);
            created if None

    Returns:
        The matplotlib Figure
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    dt = pd.DatetimeIndex(pd.to_datetime(datetimes))

    if figure is None:
        figure = plt.figure(figsize=(14, 9), facecolor="#121212")
    else:
        figure.clear()
        figure.set_facecolor("#121212")

    n_rows = 3 if cp_probability is not None else 2
    axes = figure.subplots(
        n_rows, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2, 1][:n_rows]}
    )
    ax_price, ax_equity = axes[0], axes[1]

    for ax in axes:
        ax.set_facecolor("#121212")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2)

    def shade(ax):
        for seg in segments:
            color = REGIME_COLORS.get(seg.regime, "#555555")
            ax.axvspan(
                dt[seg.start_idx],
                dt[min(seg.end_idx, len(dt)) - 1],
                color=color,
                alpha=0.18,
            )
        for seg in segments[1:]:
            ax.axvline(dt[seg.start_idx], color="#ffee58", alpha=0.6, linewidth=1)

    ax_price.plot(dt, prices, color="#90caf9", linewidth=1.2)
    shade(ax_price)
    ax_price.set_ylabel("Price", color="white")
    title = f"Regimes via BOCPD{' - ' + ticker if ticker else ''}"
    ax_price.set_title(title, color="white", fontweight="bold")

    ax_equity.plot(dt, equity_curve, color="#4CAF50", linewidth=1.4)
    shade(ax_equity)
    ax_equity.set_ylabel("Strategy Equity ($)", color="white")

    if cp_probability is not None:
        ax_cp = axes[2]
        ax_cp.fill_between(dt, cp_probability, color="#ffee58", alpha=0.6)
        ax_cp.set_ylabel("P(change)", color="white")
        ax_cp.set_ylim(0, 1)

    present = {seg.regime for seg in segments}
    handles = [
        Patch(facecolor=REGIME_COLORS[r], alpha=0.4, label=r)
        for r in REGIME_TYPES
        if r in present
    ]
    if handles:
        ax_price.legend(handles=handles, loc="upper left", fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    figure.autofmt_xdate()
    try:
        figure.tight_layout()
    except Exception:
        pass

    return figure
