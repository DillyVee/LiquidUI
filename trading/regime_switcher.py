"""
Regime-Adaptive Strategy Switching

Stores the best optimized parameter sets (max 3 "slots") in a persistent
StrategyBook and runs every stored strategy as a continuous SHADOW
simulation. Each shadow's bar returns are scored per regime type (using the
causal BOCPD regime labels), and real trades are only taken from whichever
strategy currently scores best in the live regime:

  - All stored strategies always run in shadow (signals tracked, no orders).
  - Per-regime score = exponentially weighted mean of the shadow's per-bar
    returns observed while that regime was the live label (recency-weighted,
    so "started to perform better" is captured).
  - When a different strategy's score in the current regime passes the
    selected one's, selection switches - but the open position (if any)
    is still managed to completion by the strategy that opened it.
  - The newly selected strategy begins trading at its NEXT entry signal.
    While waiting (flat), selection re-evaluates every bar, so another
    strategy can take the slot by passing it in shadow first.

The same engine drives both the historical switching backtest
(run_switching_backtest) and the live trader, so the logic cannot diverge.
"""

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Strategy storage
# ============================================================
@dataclass
class StoredStrategy:
    """One optimized parameter set stored in the book"""

    name: str
    params: Dict[str, float]
    base_score: float  # e.g. PSR from the optimization that produced it
    ticker: str = ""
    timeframes: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    created: str = ""


class StrategyBook:
    """
    Persistent store of up to MAX_STRATEGIES parameter sets, plus the
    engine's shadow-score snapshot so live switching survives restarts.
    """

    MAX_STRATEGIES = 3

    def __init__(self, path):
        self.path = Path(path)
        self.strategies: List[StoredStrategy] = []
        self.shadow_snapshot: Dict = {}
        if self.path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self.strategies)

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.strategies]

    def get(self, name: str) -> Optional[StoredStrategy]:
        for s in self.strategies:
            if s.name == name:
                return s
        return None

    def add(self, strategy: StoredStrategy) -> Tuple[bool, Optional[str]]:
        """
        Add a strategy. Same name overwrites its slot. When all slots are
        full, the lowest-base_score slot is replaced only if the newcomer
        scores higher.

        Returns:
            (added, replaced_name)
        """
        if not strategy.created:
            strategy.created = _dt.datetime.now().isoformat(timespec="seconds")

        existing = self.get(strategy.name)
        if existing is not None:
            idx = self.strategies.index(existing)
            self.strategies[idx] = strategy
            self._drop_shadow_for(strategy.name)
            self.save()
            return True, strategy.name

        if len(self.strategies) < self.MAX_STRATEGIES:
            self.strategies.append(strategy)
            self.save()
            return True, None

        worst = min(self.strategies, key=lambda s: s.base_score)
        if strategy.base_score <= worst.base_score:
            return False, None

        self.strategies[self.strategies.index(worst)] = strategy
        self._drop_shadow_for(worst.name)
        self._drop_shadow_for(strategy.name)
        self.save()
        return True, worst.name

    def remove(self, name: str) -> bool:
        s = self.get(name)
        if s is None:
            return False
        self.strategies.remove(s)
        self._drop_shadow_for(name)
        self.save()
        return True

    def _drop_shadow_for(self, name: str):
        """Stale shadow scores must not carry over to a replaced slot"""
        for key in ("scores", "obs"):
            if key in self.shadow_snapshot:
                self.shadow_snapshot[key].pop(name, None)
        if self.shadow_snapshot.get("selected") == name:
            self.shadow_snapshot.pop("selected", None)

    def save(self):
        payload = {
            "strategies": [asdict(s) for s in self.strategies],
            "shadow_snapshot": self.shadow_snapshot,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(self.path)

    def load(self):
        payload = json.loads(self.path.read_text())
        self.strategies = [StoredStrategy(**s) for s in payload.get("strategies", [])]
        self.shadow_snapshot = payload.get("shadow_snapshot", {})


# ============================================================
# Switching engine (shared by backtest and live)
# ============================================================
class RegimeSwitchingEngine:
    """
    Tracks per-strategy, per-regime shadow scores and picks which strategy
    is allowed to open the next trade.
    """

    def __init__(
        self,
        strategy_names: Sequence[str],
        base_scores: Dict[str, float],
        halflife_bars: float = 200.0,
        min_regime_bars: int = 30,
        switch_margin: float = 0.0,
        require_positive_score: bool = True,
    ):
        """
        Args:
            strategy_names: Names of the stored strategies
            base_scores: Optimization score per strategy (bootstrap ranking
                before any regime has enough shadow history)
            halflife_bars: Halflife of the exponentially weighted per-regime
                score (recency weighting of shadow performance)
            min_regime_bars: Shadow bars a strategy must accumulate in a
                regime before its score there is trusted
            switch_margin: Challenger must beat the incumbent's score by this
                much (per-bar return units) to take the selection
            require_positive_score: Stand aside (no new entries) when even
                the best trusted score in the current regime is <= 0
        """
        if not strategy_names:
            raise ValueError("Need at least one strategy")

        self.names = list(strategy_names)
        self.base_scores = dict(base_scores)
        self._alpha = 1.0 - 0.5 ** (1.0 / float(halflife_bars))
        self.min_regime_bars = int(min_regime_bars)
        self.switch_margin = float(switch_margin)
        self.require_positive_score = bool(require_positive_score)

        self.scores: Dict[str, Dict[str, float]] = {n: {} for n in self.names}
        self.obs: Dict[str, Dict[str, int]] = {n: {} for n in self.names}

        self.bootstrap = max(self.names, key=lambda n: self.base_scores.get(n, 0.0))
        self.selected = self.bootstrap
        self.switch_log: List[Dict] = []

    def update_shadows(self, shadow_returns: Dict[str, float], regime: str):
        """Fold one bar of shadow returns (earned under `regime`) into the
        per-regime scores"""
        if regime is None:
            return
        regime = str(regime)
        a = self._alpha
        for name, ret in shadow_returns.items():
            if name not in self.scores or not np.isfinite(ret):
                continue
            prev = self.scores[name].get(regime)
            self.scores[name][regime] = (
                float(ret) if prev is None else (1.0 - a) * prev + a * float(ret)
            )
            self.obs[name][regime] = self.obs[name].get(regime, 0) + 1

    def select(self, regime: str, bar_index=None, timestamp=None) -> str:
        """
        Re-evaluate which strategy holds the selection for `regime`.
        Call every bar; switches are recorded in switch_log.
        """
        regime = str(regime)
        eligible = [
            n for n in self.names if self.obs[n].get(regime, 0) >= self.min_regime_bars
        ]

        if not eligible:
            choice = self.bootstrap
        else:
            best = max(eligible, key=lambda n: self.scores[n][regime])
            incumbent = self.selected
            if (
                incumbent in eligible
                and best != incumbent
                and self.scores[best][regime]
                <= self.scores[incumbent][regime] + self.switch_margin
            ):
                best = incumbent  # hysteresis: not decisively better
            choice = best

        if choice != self.selected:
            self.switch_log.append(
                {
                    "bar_index": bar_index,
                    "timestamp": str(timestamp) if timestamp is not None else None,
                    "regime": regime,
                    "from": self.selected,
                    "to": choice,
                    "from_score": self.scores[self.selected].get(regime),
                    "to_score": self.scores[choice].get(regime),
                }
            )
            self.selected = choice

        return self.selected

    def entry_allowed(self, regime: str) -> bool:
        """Whether the selected strategy may open a NEW position in this
        regime (existing positions are always managed to completion)"""
        regime = str(regime)
        n_obs = self.obs[self.selected].get(regime, 0)
        if n_obs < self.min_regime_bars:
            # Not enough shadow history in this regime to distrust it -
            # trade the bootstrap/selected strategy
            return True
        if self.require_positive_score:
            return self.scores[self.selected].get(regime, 0.0) > 0.0
        return True

    def snapshot(self) -> Dict:
        return {
            "scores": {n: dict(v) for n, v in self.scores.items()},
            "obs": {n: dict(v) for n, v in self.obs.items()},
            "selected": self.selected,
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        }

    def restore(self, snapshot: Dict):
        if not snapshot:
            return
        for name in self.names:
            self.scores[name].update(
                {k: float(v) for k, v in snapshot.get("scores", {}).get(name, {}).items()}
            )
            self.obs[name].update(
                {k: int(v) for k, v in snapshot.get("obs", {}).get(name, {}).items()}
            )
        sel = snapshot.get("selected")
        if sel in self.names:
            self.selected = sel

    def score_table(self) -> pd.DataFrame:
        """Per-strategy x per-regime scores in basis points per bar"""
        regimes = sorted({r for n in self.names for r in self.scores[n]})
        rows = []
        for name in self.names:
            row = {"Strategy": name}
            for r in regimes:
                score = self.scores[name].get(r)
                n_obs = self.obs[name].get(r, 0)
                row[f"{r} (bp/bar)"] = (
                    round(score * 1e4, 3) if score is not None else None
                )
                row[f"{r} bars"] = n_obs
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================
# Switching backtest
# ============================================================
@dataclass
class SwitchingBacktestResult:
    """Output of the regime-switching backtest"""

    equity_curve: np.ndarray
    shadow_curves: Dict[str, np.ndarray]
    trades: pd.DataFrame
    switch_log: pd.DataFrame
    score_table: pd.DataFrame
    labels: np.ndarray
    engine: RegimeSwitchingEngine


class _ShadowSim:
    """One strategy's shadow position state, mirroring simulate_multi_tf
    semantics exactly (next-bar-open fills, cost model, sized exposure)"""

    def __init__(self, enter_signal, exit_signal, open_prices, costs, position_size):
        self.enter = enter_signal
        self.exit = exit_signal
        self.opens = open_prices
        self.costs = costs
        self.f = position_size
        self.equity = 1000.0
        self.position = False
        self.entry_price = 0.0
        self.entry_idx = 0
        self.n = len(open_prices)

    def step(self, i: int) -> float:
        """Advance one bar; returns the marked equity at bar i"""
        if not self.position and self.enter[i]:
            if i + 1 < self.n:
                price = self.opens[i + 1]
                self.entry_price = price * (1 + self.costs.TOTAL_PCT)
                self.entry_idx = i + 1
                if self.costs.COMMISSION_FIXED > 0:
                    self.equity -= self.costs.COMMISSION_FIXED
                self.position = True
        elif self.position and self.exit[i]:
            exit_price = self.opens[i + 1] if i + 1 < self.n else self.opens[i]
            exit_with_costs = exit_price * (1 - self.costs.TOTAL_PCT)
            self.equity *= 1.0 + self.f * (exit_with_costs / self.entry_price - 1.0)
            if self.costs.COMMISSION_FIXED > 0:
                self.equity -= self.costs.COMMISSION_FIXED
            self.position = False

        if self.position and i >= self.entry_idx:
            return self.equity * (
                1.0 + self.f * (self.opens[i] / self.entry_price - 1.0)
            )
        return self.equity


def run_switching_backtest(
    optimizer,
    strategies: Sequence[StoredStrategy],
    labels: np.ndarray,
    halflife_bars: float = 200.0,
    min_regime_bars: int = 30,
    switch_margin: float = 0.0,
    require_positive_score: bool = True,
) -> SwitchingBacktestResult:
    """
    Backtest the regime-switching meta-strategy over the optimizer's data.

    All strategies run in shadow bar by bar; the engine scores them per
    regime with the causal labels and only the currently selected strategy
    opens real positions (handover on its next entry signal; open positions
    are exited by the strategy that opened them).

    Args:
        optimizer: A constructed MultiTimeframeOptimizer (provides data,
            compute_signals, costs, position size)
        strategies: Stored strategies to compete (2-3 recommended)
        labels: Causal per-bar regime labels on the finest timeframe
            (from online_regime_labels)
    """
    if len(strategies) < 1:
        raise ValueError("Need at least one stored strategy")

    opens = optimizer.np_data[optimizer.finest_tf]["open"]
    datetimes = optimizer.np_data[optimizer.finest_tf]["datetime"]
    n = len(opens)
    labels = np.asarray(labels, dtype=object)
    if len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != {n} bars")

    costs = optimizer.transaction_costs
    f = optimizer.position_size

    signals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    shadows: Dict[str, _ShadowSim] = {}
    for s in strategies:
        enter, exit_ = optimizer.compute_signals(s.params)
        signals[s.name] = (enter, exit_)
        shadows[s.name] = _ShadowSim(enter, exit_, opens, costs, f)

    engine = RegimeSwitchingEngine(
        [s.name for s in strategies],
        {s.name: s.base_score for s in strategies},
        halflife_bars=halflife_bars,
        min_regime_bars=min_regime_bars,
        switch_margin=switch_margin,
        require_positive_score=require_positive_score,
    )

    shadow_curves = {name: np.zeros(n) for name in shadows}
    equity_curve = np.zeros(n)

    equity = 1000.0
    position = False
    owner: Optional[str] = None
    entry_price = 0.0
    entry_idx = 0
    entry_regime = ""
    trades: List[Dict] = []

    for i in range(n):
        # 1) Advance every shadow and fold its bar return into the scores.
        #    The return over (i-1, i] was earned under the regime known at
        #    i-1 (lag-1, no look-ahead)
        shadow_rets = {}
        for name, sim in shadows.items():
            eq_i = sim.step(i)
            shadow_curves[name][i] = eq_i
            if i > 0:
                prev = shadow_curves[name][i - 1]
                shadow_rets[name] = eq_i / prev - 1.0 if prev > 0 else 0.0
        if i > 0:
            engine.update_shadows(shadow_rets, labels[i - 1])

        # 2) Selection re-evaluates every bar (so a pending handover can be
        #    re-taken by a strategy passing it in shadow before entry)
        selected = engine.select(labels[i], bar_index=i, timestamp=datetimes[i])

        # 3) Real portfolio: only the selected strategy opens positions;
        #    the opener manages its own exit
        if not position:
            if engine.entry_allowed(labels[i]) and signals[selected][0][i]:
                if i + 1 < n:
                    price = opens[i + 1]
                    entry_price = price * (1 + costs.TOTAL_PCT)
                    entry_idx = i + 1
                    if costs.COMMISSION_FIXED > 0:
                        equity -= costs.COMMISSION_FIXED
                    position = True
                    owner = selected
                    entry_regime = str(labels[i])
        elif position and signals[owner][1][i]:
            exit_price = opens[i + 1] if i + 1 < n else opens[i]
            exit_with_costs = exit_price * (1 - costs.TOTAL_PCT)
            pct = (exit_with_costs / entry_price - 1.0) * 100
            equity *= 1.0 + f * (exit_with_costs / entry_price - 1.0)
            if costs.COMMISSION_FIXED > 0:
                equity -= costs.COMMISSION_FIXED
            trades.append(
                {
                    "Entry_Date": datetimes[entry_idx],
                    "Exit_Date": datetimes[min(i + 1, n - 1)],
                    "Strategy": owner,
                    "Regime_At_Entry": entry_regime,
                    "Percent_Change": pct,
                    "Equity_After": equity,
                }
            )
            position = False
            owner = None

        if position and i >= entry_idx:
            equity_curve[i] = equity * (1.0 + f * (opens[i] / entry_price - 1.0))
        else:
            equity_curve[i] = equity

    # Close any open position at the final bar (mirrors simulate_multi_tf)
    if position:
        exit_with_costs = opens[-1] * (1 - costs.TOTAL_PCT)
        pct = (exit_with_costs / entry_price - 1.0) * 100
        equity *= 1.0 + f * (exit_with_costs / entry_price - 1.0)
        if costs.COMMISSION_FIXED > 0:
            equity -= costs.COMMISSION_FIXED
        trades.append(
            {
                "Entry_Date": datetimes[entry_idx],
                "Exit_Date": datetimes[-1],
                "Strategy": owner,
                "Regime_At_Entry": entry_regime,
                "Percent_Change": pct,
                "Equity_After": equity,
            }
        )
        equity_curve[-1] = equity

    return SwitchingBacktestResult(
        equity_curve=equity_curve,
        shadow_curves=shadow_curves,
        trades=pd.DataFrame(trades),
        switch_log=pd.DataFrame(engine.switch_log),
        score_table=engine.score_table(),
        labels=labels,
        engine=engine,
    )


def plot_switching_backtest(
    result: SwitchingBacktestResult,
    strategies: Sequence[StoredStrategy],
    datetimes: np.ndarray,
    ticker: str = "",
    figure=None,
):
    """
    Plot the switching backtest: combined equity vs each strategy's shadow
    curve, regime shading, switch markers, and the active-strategy timeline.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from models.bayesian_changepoint import REGIME_COLORS, REGIME_TYPES, label_runs

    dt = pd.DatetimeIndex(pd.to_datetime(datetimes))
    n = len(dt)
    runs = label_runs(result.labels)

    if figure is None:
        figure = plt.figure(figsize=(14, 9), facecolor="#121212")
    else:
        figure.clear()
        figure.set_facecolor("#121212")

    ax_eq, ax_sel = figure.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    for ax in (ax_eq, ax_sel):
        ax.set_facecolor("#121212")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2)

    # Regime shading
    for start, end, regime in runs:
        color = REGIME_COLORS.get(regime, "#555555")
        for ax in (ax_eq, ax_sel):
            ax.axvspan(dt[start], dt[min(end, n) - 1], color=color, alpha=0.15)

    # Shadow curves + combined equity
    shadow_palette = ["#90caf9", "#ce93d8", "#ffcc80"]
    for i, s in enumerate(strategies):
        curve = result.shadow_curves.get(s.name)
        if curve is not None:
            ax_eq.plot(
                dt, curve,
                color=shadow_palette[i % len(shadow_palette)],
                linewidth=1.0, alpha=0.8, label=f"shadow: {s.name}",
            )
    ax_eq.plot(dt, result.equity_curve, color="#4CAF50", linewidth=2.0,
               label="SWITCHING (real)")

    # Switch markers
    if not result.switch_log.empty:
        for _, row in result.switch_log.iterrows():
            bi = row.get("bar_index")
            if bi is not None and 0 <= int(bi) < n:
                ax_eq.axvline(dt[int(bi)], color="#ffee58", alpha=0.7,
                              linewidth=1.2, linestyle="--")

    ax_eq.set_ylabel("Equity ($)", color="white")
    title = f"Regime-Switching Backtest{' - ' + ticker if ticker else ''}"
    ax_eq.set_title(title, color="white", fontweight="bold")
    ax_eq.legend(loc="upper left", fontsize=8)

    # Active-strategy timeline (piecewise from bootstrap through switches)
    names = [s.name for s in strategies]
    selected_idx = np.zeros(n, dtype=int)
    current = names.index(result.engine.bootstrap) if result.engine.bootstrap in names else 0
    pointer = 0
    switch_rows = (
        result.switch_log.to_dict("records") if not result.switch_log.empty else []
    )
    for i in range(n):
        while pointer < len(switch_rows) and switch_rows[pointer].get("bar_index") == i:
            to = switch_rows[pointer].get("to")
            if to in names:
                current = names.index(to)
            pointer += 1
        selected_idx[i] = current
    ax_sel.step(dt, selected_idx, where="post", color="#4CAF50", linewidth=1.5)
    ax_sel.set_yticks(range(len(names)))
    ax_sel.set_yticklabels(names, fontsize=8)
    ax_sel.set_ylim(-0.5, len(names) - 0.5)
    ax_sel.set_ylabel("Selected", color="white")

    present = {regime for _, _, regime in runs}
    handles = [
        Patch(facecolor=REGIME_COLORS[r], alpha=0.4, label=r)
        for r in REGIME_TYPES
        if r in present
    ]
    if handles:
        ax_sel.legend(handles=handles, loc="upper left", fontsize=7, ncol=len(handles))

    ax_sel.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    figure.autofmt_xdate()
    try:
        figure.tight_layout()
    except Exception:
        pass

    return figure


def build_switching_report(
    result: SwitchingBacktestResult,
    strategies: Sequence[StoredStrategy],
    annualization_factor: float = 252.0,
    ticker: str = "",
) -> str:
    """Text report for the switching backtest"""
    from optimization.metrics import PerformanceMetrics

    lines = [
        "=" * 76,
        f"REGIME-SWITCHING BACKTEST{' - ' + ticker if ticker else ''}",
        "=" * 76,
        "",
        f"Strategies in book ({len(strategies)}/{StrategyBook.MAX_STRATEGIES} slots):",
    ]
    for s in strategies:
        lines.append(f"   {s.name:<28} base score {s.base_score:.3f}")

    metrics = PerformanceMetrics.calculate_metrics(
        result.equity_curve, annualization_factor=annualization_factor
    )
    lines += ["", "COMBINED (switching) PERFORMANCE", "-" * 76]
    if metrics:
        for k, v in metrics.items():
            lines.append(f"   {k:<20} {v}")
    lines.append(f"   {'Trades':<20} {len(result.trades)}")
    lines.append(f"   {'Switches':<20} {len(result.switch_log)}")

    for s in strategies:
        curve = result.shadow_curves.get(s.name)
        if curve is not None and len(curve):
            ret = (curve[-1] / 1000.0 - 1.0) * 100
            lines.append(f"   {'Shadow ' + s.name:<28} standalone return {ret:+.2f}%")

    lines += ["", "PER-REGIME SHADOW SCORES (EW return, basis points per bar)", "-" * 76]
    if result.score_table.empty:
        lines.append("   (no scores yet)")
    else:
        lines.append(result.score_table.to_string(index=False))

    if not result.trades.empty:
        lines += ["", "TRADES BY STRATEGY / REGIME", "-" * 76]
        pivot = (
            result.trades.groupby(["Strategy", "Regime_At_Entry"])
            .agg(
                Trades=("Percent_Change", "size"),
                Avg_Pct=("Percent_Change", "mean"),
                Win_Rate=("Percent_Change", lambda x: (x > 0).mean() * 100),
            )
            .round(2)
            .reset_index()
        )
        lines.append(pivot.to_string(index=False))

    lines += ["", "SWITCH LOG (last 20)", "-" * 76]
    if result.switch_log.empty:
        lines.append("   No switches - one strategy held the selection throughout.")
    else:
        for _, row in result.switch_log.tail(20).iterrows():
            ts = str(row.get("timestamp", ""))[:19]
            lines.append(
                f"   {ts}  [{row['regime']}]  {row['from']} -> {row['to']}"
            )

    lines.append("=" * 76)
    return "\n".join(lines)
