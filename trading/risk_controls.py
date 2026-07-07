"""
Live Risk Controls

Hard limits enforced by the live trader independently of any strategy
signal. Once the kill switch trips, no new entries are allowed until it is
explicitly reset - a strategy bug or a broken regime cannot keep firing
orders into a losing day.

Pure logic (no broker calls) so every rule is unit-testable; the trader
feeds it equity marks and trade results and obeys its verdicts.
"""

import datetime as _dt
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RiskLimits:
    """Hard limits for live trading"""

    max_daily_loss_pct: float = 3.0  # % of day-start equity
    max_drawdown_pct: float = 15.0  # % below session high-water mark
    max_consecutive_losses: int = 6  # losing trades in a row
    max_trades_per_day: int = 20


class RiskControls:
    """Kill-switch state machine driven by equity marks and trade results"""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.kill_switch_active = False
        self.kill_reason: Optional[str] = None

        self._day: Optional[_dt.date] = None
        self._day_start_equity: Optional[float] = None
        self._high_water_mark: Optional[float] = None
        self._consecutive_losses = 0
        self._trades_today = 0

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def update_equity(self, equity: float, now: Optional[_dt.datetime] = None) -> Optional[str]:
        """
        Feed the latest account equity. Returns a violation message when a
        limit trips (kill switch activates), else None.
        """
        if not (equity > 0):
            return None
        now = now or _dt.datetime.now()

        if self._day != now.date():
            self._day = now.date()
            self._day_start_equity = equity
            self._trades_today = 0

        if self._high_water_mark is None or equity > self._high_water_mark:
            self._high_water_mark = equity

        if self.kill_switch_active:
            return None

        daily_loss_pct = (1.0 - equity / self._day_start_equity) * 100.0
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            return self._trip(
                f"daily loss {daily_loss_pct:.1f}% >= "
                f"{self.limits.max_daily_loss_pct:.1f}% limit"
            )

        dd_pct = (1.0 - equity / self._high_water_mark) * 100.0
        if dd_pct >= self.limits.max_drawdown_pct:
            return self._trip(
                f"drawdown {dd_pct:.1f}% >= "
                f"{self.limits.max_drawdown_pct:.1f}% limit"
            )

        return None

    def record_trade_result(self, pnl_pct: float) -> Optional[str]:
        """
        Feed a closed trade's P&L (%). Returns a violation message when the
        consecutive-loss limit trips, else None.
        """
        self._trades_today += 1
        if pnl_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self.kill_switch_active:
            return None
        if self._consecutive_losses >= self.limits.max_consecutive_losses:
            return self._trip(
                f"{self._consecutive_losses} consecutive losing trades"
            )
        return None

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------
    def entry_allowed(self) -> bool:
        """Whether a NEW position may be opened right now"""
        if self.kill_switch_active:
            return False
        if self._trades_today >= self.limits.max_trades_per_day:
            return False
        return True

    def entry_block_reason(self) -> Optional[str]:
        if self.kill_switch_active:
            return f"kill switch active: {self.kill_reason}"
        if self._trades_today >= self.limits.max_trades_per_day:
            return (
                f"daily trade cap reached "
                f"({self._trades_today}/{self.limits.max_trades_per_day})"
            )
        return None

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------
    def _trip(self, reason: str) -> str:
        self.kill_switch_active = True
        self.kill_reason = reason
        return f"🚨 KILL SWITCH: {reason}"

    def reset(self, authorized_by: str = "user"):
        """Manual reset (e.g. next trading day, after review)"""
        self.kill_switch_active = False
        self.kill_reason = None
        self._consecutive_losses = 0

    def status(self) -> Dict:
        return {
            "kill_switch_active": self.kill_switch_active,
            "kill_reason": self.kill_reason,
            "day_start_equity": self._day_start_equity,
            "high_water_mark": self._high_water_mark,
            "consecutive_losses": self._consecutive_losses,
            "trades_today": self._trades_today,
        }
