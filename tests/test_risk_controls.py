"""Tests for live risk controls"""

import datetime as dt

from trading.risk_controls import RiskControls, RiskLimits


def _t(day, hour=10):
    return dt.datetime(2026, 7, day, hour)


def test_daily_loss_kill_switch():
    rc = RiskControls(RiskLimits(max_daily_loss_pct=3.0))
    assert rc.update_equity(10000, _t(1)) is None
    assert rc.update_equity(9800, _t(1, 11)) is None  # -2%
    msg = rc.update_equity(9690, _t(1, 12))  # -3.1%
    assert msg is not None and "KILL" in msg
    assert not rc.entry_allowed()
    assert "kill switch" in rc.entry_block_reason()


def test_daily_anchor_resets_next_day():
    rc = RiskControls(RiskLimits(max_daily_loss_pct=3.0, max_drawdown_pct=50.0))
    rc.update_equity(10000, _t(1))
    rc.update_equity(9750, _t(1, 15))  # -2.5%, no trip
    # Next day anchors at the lower equity; -2.5% from 9750 is fine
    assert rc.update_equity(9750, _t(2)) is None
    assert rc.update_equity(9550, _t(2, 12)) is None  # -2.05% vs day start
    assert rc.entry_allowed()


def test_drawdown_from_high_water_mark():
    rc = RiskControls(RiskLimits(max_daily_loss_pct=99.0, max_drawdown_pct=10.0))
    rc.update_equity(10000, _t(1))
    rc.update_equity(12000, _t(2))  # HWM = 12000
    assert rc.update_equity(11000, _t(3)) is None  # -8.3%
    msg = rc.update_equity(10700, _t(4))  # -10.8%
    assert msg is not None and "drawdown" in msg


def test_consecutive_loss_halt():
    rc = RiskControls(RiskLimits(max_consecutive_losses=3))
    rc.update_equity(10000, _t(1))
    assert rc.record_trade_result(-1.0) is None
    assert rc.record_trade_result(-0.5) is None
    msg = rc.record_trade_result(-2.0)
    assert msg is not None and "consecutive" in msg
    assert not rc.entry_allowed()


def test_win_resets_loss_streak():
    rc = RiskControls(RiskLimits(max_consecutive_losses=3))
    rc.update_equity(10000, _t(1))
    rc.record_trade_result(-1.0)
    rc.record_trade_result(-1.0)
    rc.record_trade_result(0.5)  # win resets streak
    assert rc.record_trade_result(-1.0) is None
    assert rc.entry_allowed()


def test_daily_trade_cap():
    rc = RiskControls(RiskLimits(max_trades_per_day=2, max_consecutive_losses=99))
    rc.update_equity(10000, _t(1))
    rc.record_trade_result(0.1)
    rc.record_trade_result(0.1)
    assert not rc.entry_allowed()
    assert "cap" in rc.entry_block_reason()
    # New day clears the cap
    rc.update_equity(10000, _t(2))
    assert rc.entry_allowed()


def test_reset_requires_explicit_call():
    rc = RiskControls(RiskLimits(max_daily_loss_pct=1.0))
    rc.update_equity(10000, _t(1))
    rc.update_equity(9800, _t(1, 11))
    assert rc.kill_switch_active
    # Even a recovery or a new day does NOT auto-reset
    rc.update_equity(10500, _t(2))
    assert rc.kill_switch_active
    rc.reset("user")
    assert rc.entry_allowed()


def test_status_dict():
    rc = RiskControls()
    rc.update_equity(10000, _t(1))
    s = rc.status()
    assert s["kill_switch_active"] is False
    assert s["day_start_equity"] == 10000
