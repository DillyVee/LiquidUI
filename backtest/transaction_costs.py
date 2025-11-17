"""
Transaction Cost Models
Almgren-Chriss market impact, slippage models, and capacity analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from infrastructure.logger import quant_logger


logger = quant_logger.get_logger('transaction_costs')


@dataclass
class MarketImpactParams:
    """Parameters for market impact model"""
    # Permanent impact coefficient
    permanent_impact: float = 0.1

    # Temporary impact coefficient
    temporary_impact: float = 0.01

    # Volatility scaling
    volatility_scaling: float = 1.0

    # Market depth (average daily volume)
    avg_daily_volume: float = 1_000_000


class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model
    Models permanent and temporary impact of large trades

    Reference: Almgren & Chriss (2001) "Optimal execution of portfolio transactions"
    """

    def __init__(self, params: Optional[MarketImpactParams] = None):
        self.params = params or MarketImpactParams()

    def permanent_impact(
        self,
        trade_size: float,
        avg_daily_volume: float,
        volatility: float
    ) -> float:
        """
        Calculate permanent market impact (price moves and doesn't recover)

        Args:
            trade_size: Number of shares traded
            avg_daily_volume: Average daily volume
            volatility: Daily volatility

        Returns:
            Permanent impact as fraction of price
        """
        participation_rate = trade_size / avg_daily_volume

        # Permanent impact ~ gamma * sigma * (Q / V)^0.6
        impact = (
            self.params.permanent_impact *
            volatility *
            (participation_rate ** 0.6)
        )

        return impact

    def temporary_impact(
        self,
        trade_size: float,
        avg_daily_volume: float,
        volatility: float,
        trade_duration_minutes: float = 1.0
    ) -> float:
        """
        Calculate temporary market impact (reverts after trade)

        Args:
            trade_size: Number of shares traded
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            trade_duration_minutes: Duration of trade execution

        Returns:
            Temporary impact as fraction of price
        """
        participation_rate = trade_size / avg_daily_volume

        # Temporary impact ~ eta * sigma * (Q / V)^0.6 / sqrt(T)
        time_factor = 1.0 / np.sqrt(trade_duration_minutes)

        impact = (
            self.params.temporary_impact *
            volatility *
            (participation_rate ** 0.6) *
            time_factor
        )

        return impact

    def total_cost(
        self,
        trade_size: float,
        price: float,
        avg_daily_volume: float,
        volatility: float,
        is_buy: bool = True,
        spread_bps: float = 5.0,
        commission_bps: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs

        Args:
            trade_size: Number of shares
            price: Current price
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            is_buy: True for buy, False for sell
            spread_bps: Bid-ask spread in basis points
            commission_bps: Commission in basis points

        Returns:
            Dictionary with cost breakdown
        """
        # Permanent impact
        perm_impact_pct = self.permanent_impact(trade_size, avg_daily_volume, volatility)

        # Temporary impact
        temp_impact_pct = self.temporary_impact(trade_size, avg_daily_volume, volatility)

        # Spread cost (half-spread)
        spread_pct = (spread_bps / 2) / 10000

        # Commission
        commission_pct = commission_bps / 10000

        # Total cost (in basis points)
        total_pct = perm_impact_pct + temp_impact_pct + spread_pct + commission_pct

        # Dollar cost
        notional = trade_size * price
        total_cost_dollars = notional * total_pct

        return {
            'permanent_impact_pct': perm_impact_pct,
            'temporary_impact_pct': temp_impact_pct,
            'spread_pct': spread_pct,
            'commission_pct': commission_pct,
            'total_pct': total_pct,
            'total_bps': total_pct * 10000,
            'total_dollars': total_cost_dollars,
            'notional': notional
        }

    def optimal_execution_schedule(
        self,
        total_shares: float,
        price: float,
        avg_daily_volume: float,
        volatility: float,
        execution_horizon_minutes: float = 60.0,
        risk_aversion: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optimal execution schedule (Almgren-Chriss)

        Args:
            total_shares: Total shares to execute
            price: Current price
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            execution_horizon_minutes: Time to execute (minutes)
            risk_aversion: Risk aversion parameter (lambda)

        Returns:
            (times, shares_remaining) schedule
        """
        # Simplified optimal trajectory
        # Full implementation would solve the optimization problem

        n_slices = max(int(execution_horizon_minutes / 5), 1)  # 5-minute slices
        times = np.linspace(0, execution_horizon_minutes, n_slices + 1)

        # Exponential decay (simple approximation)
        # True Almgren-Chriss uses sinh/cosh functions
        kappa = np.sqrt(risk_aversion * volatility ** 2 / self.params.temporary_impact)
        tau = times / execution_horizon_minutes

        shares_remaining = total_shares * (1 - tau)  # Linear for simplicity

        return times, shares_remaining


class SlippageModel:
    """
    Empirical slippage model based on order size and market conditions
    """

    def __init__(self):
        self.base_slippage_bps = 2.0  # Base slippage in bps

    def calculate_slippage(
        self,
        order_size: float,
        avg_daily_volume: float,
        volatility: float,
        spread_bps: float,
        is_aggressive: bool = True
    ) -> float:
        """
        Calculate expected slippage

        Args:
            order_size: Order size in shares
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            spread_bps: Bid-ask spread in bps
            is_aggressive: True for market orders, False for limit orders

        Returns:
            Slippage in basis points
        """
        participation_rate = order_size / avg_daily_volume

        # Base slippage
        slippage = self.base_slippage_bps

        # Size effect
        slippage += 10 * (participation_rate ** 0.5) * 10000  # Convert to bps

        # Volatility effect
        slippage += volatility * 100  # Higher vol = more slippage

        # Spread effect (aggressive orders cross spread)
        if is_aggressive:
            slippage += spread_bps / 2

        return slippage

    def calculate_realized_slippage(
        self,
        execution_price: float,
        benchmark_price: float,
        side: str
    ) -> float:
        """
        Calculate realized slippage after execution

        Args:
            execution_price: Actual fill price
            benchmark_price: Benchmark price (e.g., arrival price)
            side: 'buy' or 'sell'

        Returns:
            Slippage in basis points (positive = cost)
        """
        if side.lower() == 'buy':
            slippage_pct = (execution_price - benchmark_price) / benchmark_price
        else:
            slippage_pct = (benchmark_price - execution_price) / benchmark_price

        return slippage_pct * 10000


class CapacityAnalyzer:
    """
    Analyze strategy capacity and scaling limits
    """

    def __init__(self, impact_model: Optional[AlmgrenChrissModel] = None):
        self.impact_model = impact_model or AlmgrenChrissModel()

    def estimate_capacity(
        self,
        turnover: float,
        avg_daily_volume: float,
        volatility: float,
        target_return_pct: float,
        max_impact_pct: float = 0.001  # Max 10 bps impact
    ) -> Dict[str, float]:
        """
        Estimate strategy capacity

        Args:
            turnover: Annual turnover (e.g., 5.0 for 500%)
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            target_return_pct: Target annual return
            max_impact_pct: Maximum acceptable market impact

        Returns:
            Dictionary with capacity estimates
        """
        # Binary search for max AUM
        aum_low = 1000
        aum_high = avg_daily_volume * 250 * 10  # Start with 10x annual volume

        while aum_high - aum_low > 1000:
            aum_mid = (aum_low + aum_high) / 2

            # Calculate daily trading volume
            daily_turnover = turnover / 252
            daily_trade_volume = aum_mid * daily_turnover

            # Average trade size
            avg_trade_size = daily_trade_volume / avg_daily_volume

            # Calculate impact
            costs = self.impact_model.total_cost(
                trade_size=avg_trade_size,
                price=100,  # Normalized price
                avg_daily_volume=avg_daily_volume,
                volatility=volatility
            )

            total_annual_cost = costs['total_pct'] * turnover

            # Check if costs exceed threshold
            if total_annual_cost > max_impact_pct:
                aum_high = aum_mid
            else:
                aum_low = aum_mid

        max_capacity = aum_low

        # Calculate break-even capacity (where costs = returns)
        breakeven_capacity = max_capacity * (target_return_pct / max_impact_pct)

        return {
            'max_capacity_dollars': max_capacity,
            'breakeven_capacity_dollars': breakeven_capacity,
            'annual_turnover': turnover,
            'avg_daily_volume': avg_daily_volume,
            'max_participation_rate': (max_capacity * turnover / 252) / avg_daily_volume
        }

    def capacity_curve(
        self,
        aum_range: np.ndarray,
        turnover: float,
        avg_daily_volume: float,
        volatility: float,
        gross_return_pct: float
    ) -> pd.DataFrame:
        """
        Generate capacity curve showing net returns vs AUM

        Args:
            aum_range: Array of AUM values to test
            turnover: Annual turnover
            avg_daily_volume: Average daily volume
            volatility: Daily volatility
            gross_return_pct: Gross annual return

        Returns:
            DataFrame with capacity analysis
        """
        results = []

        for aum in aum_range:
            daily_turnover = turnover / 252
            daily_trade_volume = aum * daily_turnover
            avg_trade_size = daily_trade_volume

            costs = self.impact_model.total_cost(
                trade_size=avg_trade_size,
                price=100,
                avg_daily_volume=avg_daily_volume,
                volatility=volatility
            )

            annual_cost_pct = costs['total_pct'] * turnover
            net_return_pct = gross_return_pct - annual_cost_pct

            results.append({
                'aum': aum,
                'gross_return': gross_return_pct,
                'transaction_costs': annual_cost_pct,
                'net_return': net_return_pct,
                'participation_rate': (daily_trade_volume / avg_daily_volume),
                'cost_ratio': annual_cost_pct / gross_return_pct if gross_return_pct > 0 else np.nan
            })

        return pd.DataFrame(results)


class LiquidityAnalyzer:
    """
    Analyze market liquidity and trading constraints
    """

    @staticmethod
    def calculate_market_depth(
        volume_profile: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling market depth

        Args:
            volume_profile: Time series of volume
            window: Rolling window

        Returns:
            Market depth metric
        """
        return volume_profile.rolling(window=window).mean()

    @staticmethod
    def calculate_amihud_illiquidity(
        returns: pd.Series,
        dollar_volume: pd.Series
    ) -> float:
        """
        Calculate Amihud illiquidity ratio

        Amihud (2002): measures price impact per dollar traded

        Args:
            returns: Return series
            dollar_volume: Dollar volume series

        Returns:
            Illiquidity ratio
        """
        illiquidity = (returns.abs() / dollar_volume).mean()
        return illiquidity

    @staticmethod
    def calculate_roll_spread(
        prices: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Roll's spread estimator

        Roll (1984): estimate spread from serial covariance of price changes

        Args:
            prices: Price series
            window: Rolling window

        Returns:
            Estimated spread
        """
        price_changes = prices.diff()

        def roll_estimate(x):
            if len(x) < 2:
                return np.nan
            cov = np.cov(x[:-1], x[1:])[0, 1]
            if cov >= 0:
                return 0  # No bid-ask bounce
            return 2 * np.sqrt(-cov)

        spread = price_changes.rolling(window=window).apply(roll_estimate, raw=True)
        return spread

    @staticmethod
    def liquidity_score(
        avg_daily_volume: float,
        bid_ask_spread_bps: float,
        market_cap: float
    ) -> float:
        """
        Calculate composite liquidity score

        Args:
            avg_daily_volume: Average daily volume
            bid_ask_spread_bps: Bid-ask spread in bps
            market_cap: Market capitalization

        Returns:
            Liquidity score (0-100, higher is more liquid)
        """
        # Volume score (0-40 points)
        volume_score = min(40, (avg_daily_volume / 1_000_000) * 4)

        # Spread score (0-40 points)
        spread_score = max(0, 40 - (bid_ask_spread_bps / 10) * 4)

        # Market cap score (0-20 points)
        mcap_billion = market_cap / 1_000_000_000
        mcap_score = min(20, mcap_billion * 2)

        total_score = volume_score + spread_score + mcap_score

        return min(100, total_score)
