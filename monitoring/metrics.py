"""
Monitoring and Observability
Metrics collection, drift detection, and alerting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from scipy import stats
import json

from infrastructure.logger import quant_logger


logger = quant_logger.get_logger('monitoring')


@dataclass
class MetricSnapshot:
    """Snapshot of a metric at a point in time"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    metric_name: str
    message: str
    current_value: float
    threshold: float
    metadata: Dict[str, Any]


class MetricsCollector:
    """
    Collect and store time-series metrics
    Compatible with Prometheus/Grafana
    """

    def __init__(self):
        self.metrics: Dict[str, List[MetricSnapshot]] = {}
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a metric value

        Args:
            name: Metric name (e.g., 'portfolio.sharpe_ratio')
            value: Metric value
            tags: Tags for grouping (e.g., {'strategy': 'momentum'})
            timestamp: Timestamp (default: now)
        """
        if name not in self.metrics:
            self.metrics[name] = []

        snapshot = MetricSnapshot(
            timestamp=timestamp or datetime.now(),
            metric_name=name,
            value=value,
            tags=tags or {}
        )

        self.metrics[name].append(snapshot)

    def get_metric_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Get metric as time series

        Args:
            name: Metric name
            start_time: Start time filter
            end_time: End time filter
            tags: Tag filters

        Returns:
            Pandas Series indexed by timestamp
        """
        if name not in self.metrics:
            return pd.Series(dtype=float)

        snapshots = self.metrics[name]

        # Filter by time
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]

        # Filter by tags
        if tags:
            snapshots = [
                s for s in snapshots
                if all(s.tags.get(k) == v for k, v in tags.items())
            ]

        if not snapshots:
            return pd.Series(dtype=float)

        # Convert to series
        times = [s.timestamp for s in snapshots]
        values = [s.value for s in snapshots]

        return pd.Series(values, index=times)

    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None

        return self.metrics[name][-1].value

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format"""
        lines = []

        for metric_name, snapshots in self.metrics.items():
            if not snapshots:
                continue

            latest = snapshots[-1]

            # Format: metric_name{tag1="value1",tag2="value2"} value timestamp
            tag_str = ",".join(f'{k}="{v}"' for k, v in latest.tags.items())
            if tag_str:
                line = f'{metric_name}{{{tag_str}}} {latest.value} {int(latest.timestamp.timestamp() * 1000)}'
            else:
                line = f'{metric_name} {latest.value} {int(latest.timestamp.timestamp() * 1000)}'

            lines.append(line)

        return "\n".join(lines)


class DriftDetector:
    """
    Detect data drift and model performance drift
    """

    def __init__(self, reference_window: int = 252, test_window: int = 63):
        self.reference_window = reference_window
        self.test_window = test_window

    def detect_distribution_drift(
        self,
        reference_data: pd.Series,
        test_data: pd.Series,
        method: str = 'ks'
    ) -> Tuple[bool, float, str]:
        """
        Detect distribution drift using statistical tests

        Args:
            reference_data: Reference/baseline data
            test_data: Test/current data
            method: Test method ('ks', 'kl', 't_test')

        Returns:
            (has_drift, test_statistic, interpretation)
        """
        if method == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(reference_data, test_data)

            has_drift = p_value < 0.05
            interpretation = f"KS test: p-value={p_value:.4f}, " + \
                           ("DRIFT DETECTED" if has_drift else "No drift")

            return has_drift, p_value, interpretation

        elif method == 'kl':
            # KL divergence (requires discretization)
            ref_hist, bins = np.histogram(reference_data, bins=50, density=True)
            test_hist, _ = np.histogram(test_data, bins=bins, density=True)

            # Add small epsilon to avoid log(0)
            ref_hist = ref_hist + 1e-10
            test_hist = test_hist + 1e-10

            kl_div = np.sum(test_hist * np.log(test_hist / ref_hist))

            has_drift = kl_div > 0.1  # Threshold
            interpretation = f"KL divergence: {kl_div:.4f}, " + \
                           ("DRIFT DETECTED" if has_drift else "No drift")

            return has_drift, kl_div, interpretation

        elif method == 't_test':
            # T-test for mean difference
            statistic, p_value = stats.ttest_ind(reference_data, test_data)

            has_drift = p_value < 0.05
            interpretation = f"T-test: p-value={p_value:.4f}, " + \
                           ("DRIFT DETECTED" if has_drift else "No drift")

            return has_drift, p_value, interpretation

        else:
            raise ValueError(f"Unknown method: {method}")

    def detect_performance_degradation(
        self,
        recent_returns: pd.Series,
        baseline_sharpe: float,
        window: int = 63
    ) -> Tuple[bool, float, str]:
        """
        Detect performance degradation

        Args:
            recent_returns: Recent return series
            baseline_sharpe: Expected Sharpe ratio
            window: Rolling window for calculation

        Returns:
            (has_degraded, current_sharpe, message)
        """
        if len(recent_returns) < window:
            return False, 0.0, "Insufficient data"

        # Calculate recent Sharpe
        recent = recent_returns.tail(window)
        current_sharpe = recent.mean() / recent.std() * np.sqrt(252) if recent.std() > 0 else 0

        # Check if significantly below baseline
        degradation_threshold = 0.7  # 30% degradation

        has_degraded = current_sharpe < baseline_sharpe * degradation_threshold

        message = f"Current Sharpe: {current_sharpe:.2f}, Baseline: {baseline_sharpe:.2f}"

        if has_degraded:
            message += " - DEGRADATION DETECTED"

        return has_degraded, current_sharpe, message

    def detect_regime_change(
        self,
        returns: pd.Series,
        window: int = 60
    ) -> Tuple[bool, str]:
        """
        Detect market regime changes

        Args:
            returns: Return series
            window: Window for regime detection

        Returns:
            (regime_changed, current_regime)
        """
        if len(returns) < window * 2:
            return False, "unknown"

        # Compare recent vs historical volatility
        recent_vol = returns.tail(window).std()
        historical_vol = returns.tail(window * 2).head(window).std()

        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1

        # Regime classification
        if vol_ratio > 1.5:
            current_regime = "high_volatility"
            changed = True
        elif vol_ratio < 0.67:
            current_regime = "low_volatility"
            changed = True
        else:
            current_regime = "normal"
            changed = False

        return changed, current_regime


class AlertManager:
    """
    Manage alerts and notifications
    """

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}

    def add_rule(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = '>',  # '>', '<', '==', '>=', '<='
        severity: str = 'warning',
        message_template: str = "Metric {metric} is {value} (threshold: {threshold})"
    ):
        """
        Add an alerting rule

        Args:
            metric_name: Metric to monitor
            threshold: Alert threshold
            comparison: Comparison operator
            severity: Alert severity
            message_template: Alert message template
        """
        self.alert_rules[metric_name] = {
            'threshold': threshold,
            'comparison': comparison,
            'severity': severity,
            'message_template': message_template
        }

        logger.info(f"Added alert rule: {metric_name} {comparison} {threshold}")

    def check_alert(
        self,
        metric_name: str,
        current_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """
        Check if metric triggers an alert

        Args:
            metric_name: Metric name
            current_value: Current value
            metadata: Additional metadata

        Returns:
            Alert if triggered, None otherwise
        """
        if metric_name not in self.alert_rules:
            return None

        rule = self.alert_rules[metric_name]
        threshold = rule['threshold']
        comparison = rule['comparison']

        # Check condition
        triggered = False

        if comparison == '>':
            triggered = current_value > threshold
        elif comparison == '<':
            triggered = current_value < threshold
        elif comparison == '>=':
            triggered = current_value >= threshold
        elif comparison == '<=':
            triggered = current_value <= threshold
        elif comparison == '==':
            triggered = abs(current_value - threshold) < 1e-6

        if triggered:
            message = rule['message_template'].format(
                metric=metric_name,
                value=current_value,
                threshold=threshold
            )

            alert = Alert(
                alert_id=f"{metric_name}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                severity=rule['severity'],
                metric_name=metric_name,
                message=message,
                current_value=current_value,
                threshold=threshold,
                metadata=metadata or {}
            )

            self.alerts.append(alert)

            # Log alert
            if alert.severity == 'critical':
                logger.critical(f"ALERT: {message}")
            elif alert.severity == 'warning':
                logger.warning(f"ALERT: {message}")
            else:
                logger.info(f"ALERT: {message}")

            return alert

        return None

    def get_active_alerts(self, lookback_hours: int = 24) -> List[Alert]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        return [a for a in self.alerts if a.timestamp >= cutoff]

    def export_alerts_json(self, filepath: str):
        """Export alerts to JSON"""
        alerts_data = [asdict(a) for a in self.alerts]

        # Convert datetime to ISO format
        for alert in alerts_data:
            alert['timestamp'] = alert['timestamp'].isoformat()

        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)


class PerformanceMonitor:
    """
    Monitor strategy performance in real-time
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.baseline_metrics: Dict[str, float] = {}

    def record_performance_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Dict[str, Any]],
        strategy_name: str = 'default'
    ):
        """
        Record comprehensive performance metrics

        Args:
            returns: Return series
            equity_curve: Equity curve
            trades: List of trade records
            strategy_name: Strategy identifier
        """
        tags = {'strategy': strategy_name}

        # Calculate metrics
        if len(returns) > 0:
            # Returns
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            self.metrics.record_metric('strategy.total_return', total_return, tags)

            # Sharpe ratio
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            self.metrics.record_metric('strategy.sharpe_ratio', sharpe, tags)

            # Sortino ratio
            downside = returns[returns < 0].std()
            sortino = returns.mean() / downside * np.sqrt(252) if downside > 0 else 0
            self.metrics.record_metric('strategy.sortino_ratio', sortino, tags)

            # Max drawdown
            cummax = equity_curve.cummax()
            drawdown = (equity_curve - cummax) / cummax
            max_dd = drawdown.min()
            self.metrics.record_metric('strategy.max_drawdown', max_dd, tags)

            # Volatility
            vol = returns.std() * np.sqrt(252)
            self.metrics.record_metric('strategy.volatility', vol, tags)

            # Win rate
            if len(trades) > 0:
                winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
                win_rate = winning_trades / len(trades)
                self.metrics.record_metric('strategy.win_rate', win_rate, tags)

                # Number of trades
                self.metrics.record_metric('strategy.num_trades', len(trades), tags)

                # Average trade P&L
                avg_pnl = np.mean([t.get('pnl', 0) for t in trades])
                self.metrics.record_metric('strategy.avg_trade_pnl', avg_pnl, tags)

        logger.info(f"Recorded performance metrics for {strategy_name}")

    def compare_to_baseline(
        self,
        current_sharpe: float,
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Compare current performance to baseline

        Args:
            current_sharpe: Current Sharpe ratio
            strategy_name: Strategy name

        Returns:
            Comparison results
        """
        baseline_key = f"{strategy_name}_sharpe"

        if baseline_key not in self.baseline_metrics:
            # First run - set as baseline
            self.baseline_metrics[baseline_key] = current_sharpe
            return {
                'is_baseline': True,
                'baseline_sharpe': current_sharpe,
                'current_sharpe': current_sharpe,
                'degradation_pct': 0.0
            }

        baseline_sharpe = self.baseline_metrics[baseline_key]
        degradation_pct = (current_sharpe - baseline_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 0

        return {
            'is_baseline': False,
            'baseline_sharpe': baseline_sharpe,
            'current_sharpe': current_sharpe,
            'degradation_pct': degradation_pct,
            'is_degraded': degradation_pct < -0.20  # More than 20% worse
        }
