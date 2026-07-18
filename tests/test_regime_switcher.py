"""Tests for the regime-adaptive strategy switching system"""

import numpy as np
import pandas as pd
import pytest

from config.settings import TransactionCosts
from trading.regime_switcher import (
    RegimeSwitchingEngine,
    StoredStrategy,
    StrategyBook,
    run_switching_backtest,
)


# ============================================================
# StrategyBook
# ============================================================
def _strat(name, score):
    return StoredStrategy(name=name, params={"id": name}, base_score=score)


def test_book_caps_at_max_and_replaces_worst(tmp_path):
    book = StrategyBook(tmp_path / "book.json")
    n_max = StrategyBook.MAX_STRATEGIES

    # Fill every slot; "S1" gets the lowest score of the batch
    names = [f"S{i}" for i in range(1, n_max + 1)]
    scores = [0.6 + 0.05 * i for i in range(n_max)]  # S1 weakest at 0.6
    for name, score in zip(names, scores):
        added, replaced = book.add(_strat(name, score))
        assert added and replaced is None

    # Weaker than the worst slot (S1, 0.6) -> rejected
    added, replaced = book.add(_strat("REJECT", 0.5))
    assert not added and len(book) == n_max and "REJECT" not in book.names

    # Stronger -> replaces the worst (S1)
    added, replaced = book.add(_strat("WINNER", 0.99))
    assert added and replaced == "S1"
    assert "WINNER" in book.names and "S1" not in book.names
    assert len(book) == n_max


def test_book_same_name_overwrites_slot(tmp_path):
    book = StrategyBook(tmp_path / "book.json")
    book.add(_strat("A", 0.5))
    book.add(_strat("A", 0.9))
    assert len(book) == 1 and book.get("A").base_score == 0.9


def test_book_persistence_roundtrip(tmp_path):
    path = tmp_path / "book.json"
    book = StrategyBook(path)
    book.add(_strat("A", 0.8))
    book.shadow_snapshot = {"scores": {"A": {"Bull-Quiet": 0.001}}, "selected": "A"}
    book.save()

    book2 = StrategyBook(path)
    assert book2.names == ["A"]
    assert book2.shadow_snapshot["scores"]["A"]["Bull-Quiet"] == 0.001


def test_book_drops_shadow_scores_of_replaced_slot(tmp_path):
    book = StrategyBook(tmp_path / "book.json")
    # Fill every slot, with "B" as the weakest
    book.add(_strat("B", 0.2))
    for i in range(StrategyBook.MAX_STRATEGIES - 1):
        book.add(_strat(f"S{i}", 0.7 + 0.01 * i))
    book.shadow_snapshot = {
        "scores": {"B": {"Bear-Quiet": 0.01}},
        "obs": {"B": {"Bear-Quiet": 500}},
        "selected": "B",
    }
    book.add(_strat("D", 0.9))  # replaces B
    assert "B" not in book.shadow_snapshot["scores"]
    assert book.shadow_snapshot.get("selected") != "B"


# ============================================================
# RegimeSwitchingEngine
# ============================================================
def test_engine_bootstrap_is_best_base_score():
    eng = RegimeSwitchingEngine(["A", "B"], {"A": 0.4, "B": 0.9})
    assert eng.selected == "B"


def test_engine_switches_when_challenger_passes():
    eng = RegimeSwitchingEngine(
        ["A", "B"], {"A": 1.0, "B": 0.5}, halflife_bars=10, min_regime_bars=5
    )
    for _ in range(10):
        eng.update_shadows({"A": 0.001, "B": 0.003}, "Bear-Volatile")
        eng.select("Bear-Volatile")

    assert eng.selected == "B"
    assert len(eng.switch_log) == 1
    assert eng.switch_log[0]["from"] == "A" and eng.switch_log[0]["to"] == "B"


def test_engine_pending_can_be_retaken_before_entry():
    """A -> B switch, then C passes B in shadow before any entry: C takes it"""
    eng = RegimeSwitchingEngine(
        ["A", "B", "C"],
        {"A": 1.0, "B": 0.5, "C": 0.2},
        halflife_bars=10,
        min_regime_bars=5,
    )
    for _ in range(8):
        eng.update_shadows({"A": 0.000, "B": 0.002, "C": 0.001}, "R")
        eng.select("R")
    assert eng.selected == "B"

    for _ in range(20):
        eng.update_shadows({"A": 0.000, "B": 0.002, "C": 0.008}, "R")
        eng.select("R")
    assert eng.selected == "C"

    transitions = [(s["from"], s["to"]) for s in eng.switch_log]
    assert transitions == [("A", "B"), ("B", "C")]


def test_engine_hysteresis_margin_holds_incumbent():
    eng = RegimeSwitchingEngine(
        ["A", "B"], {"A": 1.0, "B": 0.5},
        halflife_bars=10, min_regime_bars=5, switch_margin=0.005,
    )
    for _ in range(20):
        eng.update_shadows({"A": 0.001, "B": 0.0012}, "R")  # B barely ahead
        eng.select("R")
    assert eng.selected == "A" and len(eng.switch_log) == 0


def test_engine_entry_gate():
    eng = RegimeSwitchingEngine(
        ["A"], {"A": 1.0}, halflife_bars=10, min_regime_bars=5
    )
    # Bootstrap phase: not enough shadow history -> trading allowed
    assert eng.entry_allowed("R")
    # Losing regime -> stand aside
    for _ in range(10):
        eng.update_shadows({"A": -0.002}, "R")
    eng.select("R")
    assert not eng.entry_allowed("R")
    # Unless the gate is disabled
    eng.require_positive_score = False
    assert eng.entry_allowed("R")


def test_engine_snapshot_restore_roundtrip():
    eng = RegimeSwitchingEngine(["A", "B"], {"A": 1.0, "B": 0.5}, min_regime_bars=3)
    for _ in range(5):
        eng.update_shadows({"A": 0.001, "B": 0.004}, "R")
        eng.select("R")
    snap = eng.snapshot()

    eng2 = RegimeSwitchingEngine(["A", "B"], {"A": 1.0, "B": 0.5}, min_regime_bars=3)
    eng2.restore(snap)
    assert eng2.selected == eng.selected
    assert eng2.scores["B"]["R"] == pytest.approx(eng.scores["B"]["R"])
    assert eng2.obs["A"]["R"] == eng.obs["A"]["R"]


# ============================================================
# Switching backtest - scripted scenario with full control
# ============================================================
class FakeOptimizer:
    """Minimal stand-in exposing what run_switching_backtest needs"""

    def __init__(self, opens, signal_map, position_size=1.0):
        n = len(opens)
        self.finest_tf = "daily"
        self.np_data = {
            "daily": {
                "open": np.asarray(opens, dtype=np.float64),
                "close": np.asarray(opens, dtype=np.float64),
                "datetime": pd.bdate_range("2023-01-02", periods=n).values,
            }
        }
        self.transaction_costs = TransactionCosts()  # zero costs
        self.position_size = position_size
        self._signal_map = signal_map

    def compute_signals(self, params):
        return self._signal_map[params["id"]]


@pytest.fixture
def scripted_scenario():
    """
    Bars 0-99 ('R1'): steady uptrend - A (always long, periodic exits) wins.
    Bars 100-199 ('R2'): sawtooth downtrend - A bleeds; B trades only the
    up-legs of the sawtooth and profits.
    """
    n = 200
    opens = np.zeros(n)
    for i in range(100):
        opens[i] = 100.0 + i  # uptrend
    wave = {0: 0.0, 1: -2.0, 2: 2.0, 3: 0.0}
    for i in range(100, 200):
        opens[i] = 200.0 - 0.5 * (i - 100) + wave[i % 4]

    a_enter = np.ones(n, dtype=bool)
    a_exit = np.array([(i % 25) == 24 for i in range(n)])

    b_enter = np.array([(i % 4) == 0 and i >= 100 for i in range(n)])
    b_exit = np.array([(i % 4) == 2 for i in range(n)])

    labels = np.array(["R1"] * 100 + ["R2"] * 100, dtype=object)
    signal_map = {"A": (a_enter, a_exit), "B": (b_enter, b_exit)}
    return FakeOptimizer(opens, signal_map), labels


def test_switching_backtest_hands_over_between_regimes(scripted_scenario):
    fake_opt, labels = scripted_scenario
    strategies = [
        StoredStrategy("A", {"id": "A"}, base_score=1.0),
        StoredStrategy("B", {"id": "B"}, base_score=0.5),
    ]

    result = run_switching_backtest(
        fake_opt,
        strategies,
        labels,
        halflife_bars=20.0,
        min_regime_bars=15,
        require_positive_score=True,
    )

    assert len(result.equity_curve) == 200
    assert not result.trades.empty

    # Early trades belong to A (bootstrap, wins the uptrend regime)
    first = result.trades.iloc[0]
    assert first["Strategy"] == "A" and first["Regime_At_Entry"] == "R1"

    # The engine must have switched to B inside R2
    assert not result.switch_log.empty
    r2_switches = result.switch_log[result.switch_log["regime"] == "R2"]
    assert any(r2_switches["to"] == "B"), "expected an A -> B handover in R2"
    switch_bar = int(r2_switches.iloc[0]["bar_index"])
    assert switch_bar >= 100

    # After the handover, new entries come from B with B's settings
    later = result.trades[
        pd.to_datetime(result.trades["Entry_Date"])
        > pd.Timestamp(fake_opt.np_data["daily"]["datetime"][switch_bar])
    ]
    assert not later.empty
    assert (later["Strategy"] == "B").all(), (
        f"post-switch entries not all B: {later[['Strategy', 'Regime_At_Entry']]}"
    )
    # B's sawtooth trades should be profitable
    assert later["Percent_Change"].mean() > 0


def test_switching_backtest_open_position_managed_by_owner(scripted_scenario):
    """A position opened by A must be exited by A's exit signal even if the
    selection moves to B while it is open"""
    fake_opt, labels = scripted_scenario
    strategies = [
        StoredStrategy("A", {"id": "A"}, base_score=1.0),
        StoredStrategy("B", {"id": "B"}, base_score=0.5),
    ]
    result = run_switching_backtest(
        fake_opt, strategies, labels, halflife_bars=20.0, min_regime_bars=15
    )

    a_trades = result.trades[result.trades["Strategy"] == "A"]
    dts = pd.DatetimeIndex(fake_opt.np_data["daily"]["datetime"])
    for _, t in a_trades.iterrows():
        exit_idx = dts.searchsorted(pd.Timestamp(t["Exit_Date"]))
        # A's exits fire at i % 25 == 24 -> fill at bar i+1 (or last bar)
        assert exit_idx % 25 == 0 or exit_idx == len(dts) - 1, (
            f"A trade exited at bar {exit_idx}, not an A exit fill"
        )


# ============================================================
# Shadow parity with the real backtest engine
# ============================================================
def test_shadow_simulation_matches_simulate_multi_tf():
    """The shadow sim must replicate simulate_multi_tf exactly - if these
    ever diverge, shadow scores stop describing the real strategies"""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from optimization.optimizer import MultiTimeframeOptimizer

    rng = np.random.default_rng(5)
    n = 400
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    opens = np.concatenate([[100.0], closes[:-1]])
    df = pd.DataFrame(
        {
            "Datetime": pd.bdate_range("2023-01-02", periods=n),
            "Open": opens,
            "Close": closes,
        }
    )

    opt = MultiTimeframeOptimizer(
        df_dict={"daily": df},
        n_trials=1,
        time_cycle_ranges=((1, 20), (0, 20), (0, 40)),
        mn1_range=(5, 20),
        mn2_range=(3, 10),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        timeframes=["daily"],
        transaction_costs=TransactionCosts.for_stocks(),
        position_size=0.5,
    )

    params = {
        "MN1_daily": 10, "MN2_daily": 5,
        "Entry_daily": 45.0, "Exit_daily": 55.0,
        "On_daily": 15, "Off_daily": 5, "Start_daily": 2,
    }

    sim_curve, sim_trades = opt.simulate_multi_tf(params)
    assert sim_trades > 0, "test needs a params set that actually trades"

    labels = np.array(["Bull-Quiet"] * n, dtype=object)
    result = run_switching_backtest(
        opt,
        [StoredStrategy("only", params, base_score=1.0)],
        labels,
        require_positive_score=False,
    )

    shadow = result.shadow_curves["only"]
    # Last bar differs by design: simulate closes the final open position,
    # shadows keep marking it
    assert np.allclose(shadow[:-1], sim_curve[:-1]), (
        "shadow simulation diverged from simulate_multi_tf"
    )


def test_shadow_and_real_parity_with_vol_targeting():
    """With volatility targeting enabled, the shadow sims AND the real
    switching portfolio must still replicate simulate_multi_tf exactly -
    per-trade sizing is part of the parity contract"""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from optimization.optimizer import MultiTimeframeOptimizer

    rng = np.random.default_rng(5)
    n = 400
    # Alternating quiet/volatile blocks so the size multiplier truly varies
    sigma = np.where((np.arange(n) // 50) % 2 == 0, 0.006, 0.025)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 1.0, n) * sigma)
    df = pd.DataFrame(
        {
            "Datetime": pd.bdate_range("2023-01-02", periods=n),
            "Open": np.concatenate([[100.0], closes[:-1]]),
            "Close": closes,
        }
    )

    opt = MultiTimeframeOptimizer(
        df_dict={"daily": df},
        n_trials=1,
        time_cycle_ranges=((1, 20), (0, 20), (0, 40)),
        mn1_range=(5, 20),
        mn2_range=(3, 10),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        timeframes=["daily"],
        transaction_costs=TransactionCosts.for_stocks(),
        position_size=0.8,
        vol_targeting=True,
    )
    assert opt._size_frac is not None
    assert np.min(opt._size_frac) < 0.999, "multiplier must actually bind"

    params = {
        "MN1_daily": 10, "MN2_daily": 5,
        "Entry_daily": 45.0, "Exit_daily": 55.0,
        "On_daily": 15, "Off_daily": 5, "Start_daily": 2,
    }

    sim_curve, sim_trades = opt.simulate_multi_tf(params)
    assert sim_trades > 0

    labels = np.array(["Bull-Quiet"] * n, dtype=object)
    result = run_switching_backtest(
        opt,
        [StoredStrategy("only", params, base_score=1.0)],
        labels,
        require_positive_score=False,
    )

    assert np.allclose(result.shadow_curves["only"][:-1], sim_curve[:-1]), (
        "vol-targeted shadow diverged from simulate_multi_tf"
    )
    assert np.allclose(result.equity_curve, sim_curve), (
        "vol-targeted real switching portfolio diverged from simulate_multi_tf"
    )
