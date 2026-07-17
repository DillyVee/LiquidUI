"""Integration tests: objective selection and search-space toggles on the
MultiTimeframeOptimizer"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from config.settings import Paths, TransactionCosts
from optimization.optimizer import MultiTimeframeOptimizer
from signals.indicators import INDICATORS


def _df_dict(n=450, seed=8):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0.0006, 0.012, n))
    return {
        "daily": pd.DataFrame(
            {
                "Datetime": pd.bdate_range("2021-06-01", periods=n),
                "Open": np.concatenate([[100.0], closes[:-1]]),
                "Close": closes,
            }
        )
    }


def _optimizer(**overrides):
    kwargs = dict(
        df_dict=_df_dict(),
        n_trials=60,
        time_cycle_ranges=((1, 20), (0, 20), (0, 40)),
        mn1_range=(5, 20),
        mn2_range=(3, 10),
        entry_range=(10.0, 40.0),
        exit_range=(50.0, 90.0),
        ticker="OBJTEST",
        timeframes=["daily"],
        transaction_costs=TransactionCosts(),
        position_size=1.0,
        seed=1234,
    )
    kwargs.update(overrides)
    return MultiTimeframeOptimizer(**kwargs)


def test_unknown_objective_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown objective"):
        _optimizer(objective="nope")


def test_unknown_indicator_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown indicators"):
        _optimizer(allowed_indicators=["rsi", "hodl"])


@pytest.mark.parametrize("objective", ["sharpe", "profit_factor"])
def test_run_records_objective_and_full_metrics(objective):
    opt = _optimizer(objective=objective)
    opt.run()

    assert opt.all_results
    best = opt.all_results[0]
    assert best["Objective"] == objective
    assert np.isfinite(best["Objective_Score"])
    for key in (
        "Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio", "Percent_Gain_%",
        "Max_Drawdown_%", "Profit_Factor", "CAGR_%", "Trade_Count",
        "BuyHold_Return_%",
    ):
        assert key in best, key
    # DSR inputs collected for the anti-overfit gates
    assert len(opt.trial_sharpes) > 0
    assert opt.rival_params
    # Exposure (time in market) recorded alongside the other metrics
    assert "Exposure_%" in best
    assert 0.0 <= best["Exposure_%"] <= 100.0
    # Validation-split selection active on 450 bars: segment scores present
    assert "Train_Score" in best and "Val_Score" in best
    # Cycle Start stored canonically (one encoding per strategy)
    period = max(1, int(best["On_daily"]) + int(best["Off_daily"]))
    assert 0 <= int(best["Start_daily"]) < period


def test_cycle_start_canonicalization_is_semantically_free():
    """Start % (On + Off) must not change the simulation at all"""
    opt = _optimizer()
    period = PARAMS_RSI["On_daily"] + PARAMS_RSI["Off_daily"]
    shifted = dict(PARAMS_RSI, Start_daily=PARAMS_RSI["Start_daily"] + 3 * period)

    eq1, t1 = opt.simulate_multi_tf(PARAMS_RSI)
    eq2, t2 = opt.simulate_multi_tf(shifted)
    assert t1 == t2
    assert np.allclose(eq1, eq2)


def test_simulate_reports_exposure_stats():
    opt = _optimizer()

    # Always-in-market once warmed up: enter below 100.5 (always), never
    # exit; cycle always ON. Exposure = bars from first entry to the end.
    hold = {
        "MN1_daily": 5, "MN2_daily": 3,
        "Entry_daily": 100.5, "Exit_daily": 100.5,
        "On_daily": 20, "Off_daily": 0, "Start_daily": 0,
    }
    stats = {}
    eq, trades = opt.simulate_multi_tf(hold, stats_out=stats)
    n = len(eq)
    enter, _ = opt.compute_signals(hold)
    first_entry = int(np.argmax(enter)) + 1  # fills at next bar's open
    assert stats["exposure_frac"] == pytest.approx((n - first_entry) / n)

    # Cycle permanently OFF: no trades, zero exposure
    never = dict(hold, On_daily=1, Off_daily=0, Start_daily=0)
    # On=1, Off=0 -> period 1, cycle always ON; instead block via Entry=0
    never["Entry_daily"] = 0.0  # oscillator can never be below 0
    stats2 = {}
    eq2, trades2 = opt.simulate_multi_tf(never, stats_out=stats2)
    assert trades2 == 0
    assert stats2["exposure_frac"] == 0.0


def test_allowed_indicators_restricts_search():
    allowed = ["stoch", "aroon"]
    opt = _optimizer(allowed_indicators=allowed)
    opt.run()

    best = opt.all_results[0]
    assert best["IND1_daily"] in allowed
    ind2 = best.get("IND2_daily", "none")
    assert ind2 in ["none"] + allowed


def test_combine_indicators_off_forces_single_leg():
    opt = _optimizer(combine_indicators=False)
    opt.run()

    best = opt.all_results[0]
    assert best["IND1_daily"] in INDICATORS
    assert best["IND2_daily"] == "none"
    assert "IND2_P1_daily" not in best


def test_results_csv_columns_stay_aligned():
    """Rows with different parameter shapes (with/without IND2) must land
    in one readable CSV with a consistent header"""
    opt = _optimizer(objective="sortino")
    results_path = Paths.get_results_path("OBJTEST", suffix="_sortino")
    if results_path.exists():
        results_path.unlink()

    opt.run()

    assert results_path.exists()
    df = pd.read_csv(results_path)
    assert len(df) == len(opt._result_rows)
    assert "Objective" in df.columns and "Sharpe_Ratio" in df.columns
    # Every row parses back with the same schema; the final row is the best
    assert df["Objective"].eq("sortino").all()
    results_path.unlink()


PARAMS_RSI = {
    "MN1_daily": 10, "MN2_daily": 3,
    "Entry_daily": 35.0, "Exit_daily": 65.0,
    "On_daily": 10, "Off_daily": 5, "Start_daily": 0,
}


def test_evaluate_params_scales_by_trade_confidence():
    """Without a validation split the sampler score is the full-sample
    objective scaled by the trade-confidence ramp (pre-H5 contract)"""
    opt = _optimizer(selection_holdout_frac=0.0)
    assert opt.selection_cut is None

    raw, score, metrics, trade_count = opt.evaluate_params(PARAMS_RSI)
    assert metrics is not None
    assert "Val_Score" not in metrics
    if trade_count >= 30:
        assert score == pytest.approx(raw)
    elif trade_count < 10:
        assert score == 0.0
    else:
        assert score == pytest.approx(raw * (trade_count - 10) / 20.0)


def test_evaluate_params_with_validation_split():
    """With the split active the sampler only ever sees the TRAIN-segment
    score; the validation slice is recorded for winner selection"""
    opt = _optimizer()  # 450 bars >= MIN_SELECTION_BARS -> split active
    assert opt.selection_cut == int(450 * 0.75)

    raw, score, metrics, trade_count = opt.evaluate_params(PARAMS_RSI)
    assert metrics is not None
    for key in ("Train_Score", "Val_Score", "Train_Trades", "Val_Trades"):
        assert key in metrics, key
    assert score == pytest.approx(metrics["Train_Score"])
    assert metrics["Train_Trades"] + metrics["Val_Trades"] == trade_count
    # Full-sample raw objective is preserved for reporting
    assert np.isfinite(raw)


def test_selection_split_disabled_on_small_data():
    """Walk-forward-sized windows must keep full-sample selection"""
    opt = _optimizer(df_dict=_df_dict(n=300))
    assert opt.selection_cut is None


def test_phase_winner_validation_veto():
    """Validation acts as a VETO (survivors ranked by train score), not an
    argmax - the argmax variant was falsified on no-edge data (journal
    experiment H5a: max over a short slice has higher selection variance
    than max over the full sample)"""
    import optuna

    opt = _optimizer()  # split active
    dists = {"x": optuna.distributions.IntDistribution(0, 10)}

    def add(study, x, train, val):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": x},
                distributions=dists,
                value=train,
                user_attrs={"val_score": val, "params": {"x": x}},
            )
        )

    # Best train candidate survives validation -> it wins (val magnitude
    # beyond the veto is deliberately ignored)
    study = optuna.create_study(direction="maximize")
    add(study, 0, train=1.0, val=0.1)  # best train, survives
    add(study, 1, train=0.5, val=0.9)  # best validation, weaker train
    add(study, 2, train=0.8, val=None)  # no validation evidence
    assert opt._phase_winner(study).params["x"] == 0

    # Best train candidate vetoed (val <= 0) -> best surviving train wins
    study2 = optuna.create_study(direction="maximize")
    add(study2, 0, train=1.0, val=-0.2)
    add(study2, 1, train=0.5, val=0.3)
    add(study2, 2, train=0.7, val=0.0)
    assert opt._phase_winner(study2).params["x"] == 1

    # No survivors anywhere -> fall back to plain train argmax
    study3 = optuna.create_study(direction="maximize")
    add(study3, 0, train=0.3, val=0.0)
    add(study3, 1, train=0.9, val=-0.5)
    assert opt._phase_winner(study3).params["x"] == 1

    # Split inactive -> validation attrs are ignored entirely
    small = _optimizer(df_dict=_df_dict(n=300))
    study4 = optuna.create_study(direction="maximize")
    add(study4, 0, train=0.9, val=-0.8)
    add(study4, 1, train=0.2, val=0.8)
    assert small._phase_winner(study4).params["x"] == 0
