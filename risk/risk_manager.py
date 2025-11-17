"""
Risk Management System
Real-time risk controls, position limits, kill switches, and P&L monitoring
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from infrastructure.logger import quant_logger


logger = quant_logger.get_logger('risk_management')


class RiskLevel(Enum):
    """Risk alert levels"""
    GREEN = "green"      # Normal
    YELLOW = "yellow"    # Warning
    ORANGE = "orange"    # Elevated
    RED = "red"          # Critical - kill switch


@dataclass
class RiskLimit:
    """Risk limit configuration"""
    name: str
    limit_type: str  # 'max', 'min', 'range'
    value: float
    threshold_warning: float  # Warning at X% of limit
    threshold_critical: float  # Critical at X% of limit
    current_value: float = 0.0
    status: RiskLevel = RiskLevel.GREEN


@dataclass
class RiskEvent:
    """Risk event/breach"""
    timestamp: datetime
    event_type: str
    severity: RiskLevel
    description: str
    current_value: float
    limit_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionLimits:
    """Position and concentration limits"""

    def __init__(self, config: Dict[str, Any]):
        self.max_gross_exposure = config.get('max_gross_exposure', 1000000)
        self.max_net_exposure = config.get('max_net_exposure', 500000)
        self.max_single_position = config.get('max_single_position', 100000)
        self.max_sector_concentration = config.get('max_sector_concentration', 0.30)
        self.max_leverage = config.get('max_leverage', 1.0)

    def check_position_limits(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Check position limits

        Args:
            positions: Dict of {symbol: quantity}
            prices: Dict of {symbol: price}
            portfolio_value: Total portfolio value

        Returns:
            (risk_level, list_of_violations)
        """
        violations = []
        max_risk_level = RiskLevel.GREEN

        # Calculate exposures
        long_exposure = sum(
            max(0, qty) * prices.get(sym, 0)
            for sym, qty in positions.items()
        )

        short_exposure = abs(sum(
            min(0, qty) * prices.get(sym, 0)
            for sym, qty in positions.items()
        ))

        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # Check gross exposure
        if gross_exposure > self.max_gross_exposure:
            violations.append(
                f"Gross exposure ${gross_exposure:,.0f} exceeds limit ${self.max_gross_exposure:,.0f}"
            )
            max_risk_level = RiskLevel.RED

        elif gross_exposure > self.max_gross_exposure * 0.9:
            violations.append(f"Gross exposure at {gross_exposure/self.max_gross_exposure*100:.1f}% of limit")
            max_risk_level = max(max_risk_level, RiskLevel.YELLOW)

        # Check net exposure
        if abs(net_exposure) > self.max_net_exposure:
            violations.append(
                f"Net exposure ${abs(net_exposure):,.0f} exceeds limit ${self.max_net_exposure:,.0f}"
            )
            max_risk_level = max(max_risk_level, RiskLevel.ORANGE)

        # Check leverage
        leverage = gross_exposure / portfolio_value if portfolio_value > 0 else 0

        if leverage > self.max_leverage:
            violations.append(f"Leverage {leverage:.2f}x exceeds limit {self.max_leverage:.2f}x")
            max_risk_level = RiskLevel.RED

        # Check single position limits
        for symbol, qty in positions.items():
            position_value = abs(qty) * prices.get(symbol, 0)

            if position_value > self.max_single_position:
                violations.append(
                    f"Position in {symbol} ${position_value:,.0f} exceeds limit ${self.max_single_position:,.0f}"
                )
                max_risk_level = max(max_risk_level, RiskLevel.ORANGE)

        return max_risk_level, violations


class PnLLimits:
    """P&L-based limits and stop-loss"""

    def __init__(self, config: Dict[str, Any]):
        self.max_daily_loss = config.get('max_daily_loss', -10000)
        self.max_drawdown_pct = config.get('max_drawdown_pct', -0.10)  # -10%
        self.profit_target = config.get('profit_target', 20000)
        self.trailing_stop_pct = config.get('trailing_stop_pct', -0.05)  # -5% from peak

        self.daily_start_equity = 0.0
        self.peak_equity = 0.0
        self.hwm = 0.0  # High water mark

    def check_pnl_limits(
        self,
        current_equity: float,
        realized_pnl_today: float
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Check P&L limits

        Args:
            current_equity: Current portfolio equity
            realized_pnl_today: Realized P&L for today

        Returns:
            (risk_level, list_of_violations)
        """
        violations = []
        max_risk_level = RiskLevel.GREEN

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Update HWM
        if current_equity > self.hwm:
            self.hwm = current_equity

        # Check daily loss limit
        daily_pnl = current_equity - self.daily_start_equity

        if daily_pnl < self.max_daily_loss:
            violations.append(
                f"Daily loss ${daily_pnl:,.0f} exceeds limit ${self.max_daily_loss:,.0f} - KILL SWITCH"
            )
            max_risk_level = RiskLevel.RED

        elif daily_pnl < self.max_daily_loss * 0.8:
            violations.append(f"Daily loss at {daily_pnl/self.max_daily_loss*100:.0f}% of limit")
            max_risk_level = max(max_risk_level, RiskLevel.YELLOW)

        # Check drawdown from peak
        if self.peak_equity > 0:
            drawdown_pct = (current_equity - self.peak_equity) / self.peak_equity

            if drawdown_pct < self.max_drawdown_pct:
                violations.append(
                    f"Drawdown {drawdown_pct*100:.1f}% exceeds limit {self.max_drawdown_pct*100:.1f}% - KILL SWITCH"
                )
                max_risk_level = RiskLevel.RED

        # Check trailing stop
        if self.hwm > 0:
            trailing_loss_pct = (current_equity - self.hwm) / self.hwm

            if trailing_loss_pct < self.trailing_stop_pct:
                violations.append(
                    f"Trailing stop triggered: {trailing_loss_pct*100:.1f}% from high water mark"
                )
                max_risk_level = max(max_risk_level, RiskLevel.ORANGE)

        # Check profit target (optional de-risk)
        if daily_pnl > self.profit_target:
            violations.append(f"Profit target reached: ${daily_pnl:,.0f} (consider de-risking)")
            # Not a violation, just informational

        return max_risk_level, violations

    def reset_daily_limits(self, current_equity: float):
        """Reset daily limits at start of day"""
        self.daily_start_equity = current_equity
        logger.info(f"Daily limits reset at equity ${current_equity:,.0f}")


class RiskManager:
    """
    Centralized risk manager with kill switch functionality
    """

    def __init__(self, config: Dict[str, Any]):
        self.position_limits = PositionLimits(config.get('position_limits', {}))
        self.pnl_limits = PnLLimits(config.get('pnl_limits', {}))

        self.enabled = True
        self.kill_switch_active = False
        self.risk_events: List[RiskEvent] = []

        self.volatility_limit = config.get('volatility_limit', 0.03)  # 3% daily vol
        self.var_limit = config.get('var_limit', -50000)  # $50k 1-day 95% VaR

        logger.info("Risk Manager initialized")

    def check_all_limits(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        realized_pnl_today: float
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Run all risk checks

        Args:
            positions: Current positions
            prices: Current prices
            portfolio_value: Portfolio value
            realized_pnl_today: Realized P&L today

        Returns:
            (overall_risk_level, list_of_violations)
        """
        if not self.enabled:
            return RiskLevel.GREEN, []

        all_violations = []

        # Position limits
        pos_level, pos_violations = self.position_limits.check_position_limits(
            positions, prices, portfolio_value
        )
        all_violations.extend(pos_violations)

        # P&L limits
        pnl_level, pnl_violations = self.pnl_limits.check_pnl_limits(
            portfolio_value, realized_pnl_today
        )
        all_violations.extend(pnl_violations)

        # Overall risk level is the maximum
        overall_level = max(pos_level, pnl_level, key=lambda x: ['green', 'yellow', 'orange', 'red'].index(x.value))

        # Trigger kill switch if RED
        if overall_level == RiskLevel.RED and not self.kill_switch_active:
            self.trigger_kill_switch("Risk limit breach")
            all_violations.append("🚨 KILL SWITCH ACTIVATED 🚨")

        # Log risk events
        if all_violations:
            event = RiskEvent(
                timestamp=datetime.now(),
                event_type="limit_check",
                severity=overall_level,
                description="; ".join(all_violations),
                current_value=portfolio_value,
                limit_value=0,
                metadata={
                    'positions': positions,
                    'prices': prices
                }
            )
            self.risk_events.append(event)

            if overall_level in [RiskLevel.ORANGE, RiskLevel.RED]:
                logger.error(f"Risk event {overall_level.value}: {event.description}")
            elif overall_level == RiskLevel.YELLOW:
                logger.warning(f"Risk event {overall_level.value}: {event.description}")

        return overall_level, all_violations

    def trigger_kill_switch(self, reason: str):
        """
        Trigger emergency kill switch

        In production, this would:
        - Cancel all pending orders
        - Close all positions (or flatten)
        - Disable new order submission
        - Send alerts to traders/risk managers
        """
        self.kill_switch_active = True
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason} 🚨")

        # Log to audit trail
        from infrastructure.logger import quant_logger
        audit = quant_logger.audit

        audit.log_risk_event(
            event_type="kill_switch",
            details={
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'action': 'kill_switch_activated'
            }
        )

        # In production: send alerts (email, SMS, Slack, PagerDuty)
        self._send_alert(f"KILL SWITCH: {reason}")

    def reset_kill_switch(self, authorized_by: str):
        """Reset kill switch (requires authorization)"""
        self.kill_switch_active = False
        logger.info(f"Kill switch reset by {authorized_by}")

    def _send_alert(self, message: str):
        """Send alert (placeholder for actual alerting)"""
        logger.critical(f"ALERT: {message}")
        # In production: integrate with alerting system
        # - Email
        # - SMS (Twilio)
        # - Slack/Teams webhook
        # - PagerDuty
        pass

    def calculate_var(
        self,
        positions: Dict[str, float],
        returns_history: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            positions: Current positions
            returns_history: Historical returns for each symbol
            confidence_level: VaR confidence level (0.95 = 95%)

        Returns:
            VaR value (negative = potential loss)
        """
        # Portfolio returns
        position_weights = pd.Series(positions)

        # Calculate historical portfolio returns
        portfolio_returns = (returns_history * position_weights).sum(axis=1)

        # VaR is the percentile of return distribution
        var = portfolio_returns.quantile(1 - confidence_level)

        return var

    def calculate_var_parametric(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate parametric VaR using covariance matrix

        Args:
            positions: Current positions
            prices: Current prices
            covariance_matrix: Covariance matrix of returns
            confidence_level: Confidence level

        Returns:
            VaR value
        """
        # Position values
        position_values = pd.Series({
            sym: positions.get(sym, 0) * prices.get(sym, 0)
            for sym in covariance_matrix.columns
        })

        # Portfolio variance
        portfolio_variance = position_values.dot(covariance_matrix.dot(position_values))
        portfolio_std = np.sqrt(portfolio_variance)

        # VaR (assuming normal distribution)
        z_score = stats.norm.ppf(confidence_level)
        var = -z_score * portfolio_std

        return var

    def stress_test(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        scenarios: List[Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Run stress test scenarios

        Args:
            positions: Current positions
            prices: Current prices
            scenarios: List of scenarios, each a dict of {symbol: price_change_pct}

        Returns:
            DataFrame with scenario results
        """
        results = []

        for i, scenario in enumerate(scenarios):
            scenario_pnl = 0

            for symbol, qty in positions.items():
                if symbol in scenario:
                    price = prices.get(symbol, 0)
                    price_change_pct = scenario[symbol]
                    pnl = qty * price * price_change_pct
                    scenario_pnl += pnl

            results.append({
                'scenario': f'Scenario {i+1}',
                'pnl': scenario_pnl,
                'changes': scenario
            })

        return pd.DataFrame(results)

    def get_risk_summary(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Get comprehensive risk summary"""

        # Calculate exposures
        long_exposure = sum(
            max(0, qty) * prices.get(sym, 0)
            for sym, qty in positions.items()
        )

        short_exposure = abs(sum(
            min(0, qty) * prices.get(sym, 0)
            for sym, qty in positions.items()
        ))

        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        summary = {
            'portfolio_value': portfolio_value,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'leverage': gross_exposure / portfolio_value if portfolio_value > 0 else 0,
            'num_positions': len([p for p in positions.values() if p != 0]),
            'kill_switch_active': self.kill_switch_active,
            'recent_events': len([e for e in self.risk_events if e.timestamp > datetime.now() - timedelta(hours=1)])
        }

        return summary
