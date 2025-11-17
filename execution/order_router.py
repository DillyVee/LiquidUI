"""
Execution System with Smart Order Routing
Simulates production execution with TWAP, VWAP, POV algorithms and FIX-like connectivity
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from infrastructure.logger import quant_logger

logger = quant_logger.get_logger("execution")


class ExecutionAlgo(Enum):
    """Execution algorithm types"""

    MARKET = "market"  # Immediate market order
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"  # Percentage of Volume
    ICEBERG = "iceberg"  # Iceberg / hidden orders
    SMART = "smart"  # Smart routing with opportunistic limit orders


@dataclass
class ExecutionInstruction:
    """Parent order / execution instruction"""

    instruction_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: float
    algorithm: ExecutionAlgo
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    # Algorithm-specific parameters
    limit_price: Optional[float] = None
    target_pov: float = 0.10  # Target 10% of volume for POV
    max_pov: float = 0.30  # Max 30% of volume
    urgency: float = 0.5  # 0 = patient, 1 = aggressive

    # State tracking
    filled_quantity: float = 0.0
    child_orders: List[Any] = field(default_factory=list)
    status: str = "active"  # active, completed, cancelled


@dataclass
class ChildOrder:
    """Child order (sliced from parent)"""

    child_id: str
    parent_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str  # 'market', 'limit'
    limit_price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"  # pending, filled, cancelled


class TWAPSlicer:
    """
    Time-Weighted Average Price execution
    Splits order evenly across time
    """

    def slice_order(
        self, instruction: ExecutionInstruction, slice_interval_minutes: int = 5
    ) -> List[ChildOrder]:
        """
        Slice parent order into TWAP child orders

        Args:
            instruction: Parent execution instruction
            slice_interval_minutes: Time between slices

        Returns:
            List of child orders
        """
        duration = (instruction.end_time - instruction.start_time).total_seconds() / 60
        n_slices = max(1, int(duration / slice_interval_minutes))

        slice_size = instruction.total_quantity / n_slices

        child_orders = []

        for i in range(n_slices):
            slice_time = instruction.start_time + timedelta(
                minutes=i * slice_interval_minutes
            )

            # Use limit orders if not urgent, market orders if urgent
            if instruction.urgency < 0.7:
                order_type = "limit"
                # Set limit slightly away from mid to avoid adverse selection
                if instruction.side == "buy":
                    limit_price = (
                        instruction.limit_price * 0.9999
                        if instruction.limit_price
                        else None
                    )
                else:
                    limit_price = (
                        instruction.limit_price * 1.0001
                        if instruction.limit_price
                        else None
                    )
            else:
                order_type = "market"
                limit_price = None

            child = ChildOrder(
                child_id=str(uuid.uuid4()),
                parent_id=instruction.instruction_id,
                symbol=instruction.symbol,
                side=instruction.side,
                quantity=slice_size,
                order_type=order_type,
                limit_price=limit_price,
                timestamp=slice_time,
            )

            child_orders.append(child)

        logger.info(
            f"TWAP: Sliced {instruction.total_quantity} into {n_slices} orders of {slice_size:.2f}"
        )

        return child_orders


class VWAPSlicer:
    """
    Volume-Weighted Average Price execution
    Slices order proportional to expected volume profile
    """

    def __init__(self):
        # Typical intraday volume profile (US market)
        # Higher at open/close, lower at lunch
        self.hourly_volume_profile = {
            9: 0.15,  # 9:30-10:30 (market open)
            10: 0.12,
            11: 0.08,
            12: 0.06,  # Lunch
            13: 0.07,
            14: 0.09,
            15: 0.13,  # 3:00-4:00
            16: 0.30,  # Last hour (close)
        }

    def slice_order(
        self, instruction: ExecutionInstruction, slice_interval_minutes: int = 15
    ) -> List[ChildOrder]:
        """
        Slice order following VWAP profile

        Args:
            instruction: Parent execution instruction
            slice_interval_minutes: Time between slices

        Returns:
            List of child orders
        """
        child_orders = []

        current_time = instruction.start_time
        remaining_quantity = instruction.total_quantity

        while current_time < instruction.end_time and remaining_quantity > 0:
            hour = current_time.hour

            # Get expected volume weight for this hour
            volume_weight = self.hourly_volume_profile.get(hour, 0.08)

            # Allocate quantity proportional to volume
            slice_size = min(
                remaining_quantity,
                instruction.total_quantity
                * volume_weight
                * (slice_interval_minutes / 60),
            )

            child = ChildOrder(
                child_id=str(uuid.uuid4()),
                parent_id=instruction.instruction_id,
                symbol=instruction.symbol,
                side=instruction.side,
                quantity=slice_size,
                order_type="limit" if instruction.urgency < 0.7 else "market",
                limit_price=instruction.limit_price,
                timestamp=current_time,
            )

            child_orders.append(child)

            remaining_quantity -= slice_size
            current_time += timedelta(minutes=slice_interval_minutes)

        logger.info(
            f"VWAP: Sliced {instruction.total_quantity} into {len(child_orders)} orders"
        )

        return child_orders


class POVSlicer:
    """
    Percentage of Volume execution
    Targets a percentage of market volume
    """

    def slice_order(
        self,
        instruction: ExecutionInstruction,
        market_volume_profile: pd.Series,
        slice_interval_minutes: int = 5,
    ) -> List[ChildOrder]:
        """
        Slice order based on POV target

        Args:
            instruction: Parent execution instruction
            market_volume_profile: Expected market volume over time
            slice_interval_minutes: Time between slices

        Returns:
            List of child orders
        """
        child_orders = []

        current_time = instruction.start_time
        remaining_quantity = instruction.total_quantity

        while current_time < instruction.end_time and remaining_quantity > 0:
            # Get expected market volume for this interval
            # (In production, this would use real-time market data)
            expected_market_vol = market_volume_profile.get(current_time, 10000)

            # Calculate target participation
            target_quantity = expected_market_vol * instruction.target_pov

            # Cap at max POV
            max_quantity = expected_market_vol * instruction.max_pov

            slice_size = min(target_quantity, max_quantity, remaining_quantity)

            child = ChildOrder(
                child_id=str(uuid.uuid4()),
                parent_id=instruction.instruction_id,
                symbol=instruction.symbol,
                side=instruction.side,
                quantity=slice_size,
                order_type="limit",
                limit_price=instruction.limit_price,
                timestamp=current_time,
            )

            child_orders.append(child)

            remaining_quantity -= slice_size
            current_time += timedelta(minutes=slice_interval_minutes)

        logger.info(
            f"POV: Sliced {instruction.total_quantity} into {len(child_orders)} orders (target {instruction.target_pov*100}% POV)"
        )

        return child_orders


class SmartOrderRouter:
    """
    Smart order routing with opportunistic execution
    Combines multiple strategies dynamically
    """

    def __init__(self):
        self.twap_slicer = TWAPSlicer()
        self.vwap_slicer = VWAPSlicer()
        self.pov_slicer = POVSlicer()

    def route_order(
        self,
        instruction: ExecutionInstruction,
        market_data: Optional[pd.DataFrame] = None,
    ) -> List[ChildOrder]:
        """
        Route order using specified algorithm

        Args:
            instruction: Execution instruction
            market_data: Real-time market data (for POV/VWAP)

        Returns:
            List of child orders
        """
        logger.info(
            f"Routing {instruction.algorithm.value} order: {instruction.symbol} {instruction.side} {instruction.total_quantity}"
        )

        if instruction.algorithm == ExecutionAlgo.MARKET:
            # Single market order
            return [
                ChildOrder(
                    child_id=str(uuid.uuid4()),
                    parent_id=instruction.instruction_id,
                    symbol=instruction.symbol,
                    side=instruction.side,
                    quantity=instruction.total_quantity,
                    order_type="market",
                    timestamp=instruction.start_time,
                )
            ]

        elif instruction.algorithm == ExecutionAlgo.TWAP:
            return self.twap_slicer.slice_order(instruction)

        elif instruction.algorithm == ExecutionAlgo.VWAP:
            return self.vwap_slicer.slice_order(instruction)

        elif instruction.algorithm == ExecutionAlgo.POV:
            # Need volume profile
            if market_data is not None and "Volume" in market_data.columns:
                volume_profile = market_data["Volume"]
            else:
                # Use dummy profile
                volume_profile = pd.Series(
                    10000,
                    index=pd.date_range(
                        instruction.start_time, instruction.end_time, freq="5min"
                    ),
                )

            return self.pov_slicer.slice_order(instruction, volume_profile)

        elif instruction.algorithm == ExecutionAlgo.SMART:
            # Adaptive strategy
            return self._smart_routing(instruction, market_data)

        else:
            logger.error(f"Unknown algorithm: {instruction.algorithm}")
            return []

    def _smart_routing(
        self, instruction: ExecutionInstruction, market_data: Optional[pd.DataFrame]
    ) -> List[ChildOrder]:
        """
        Smart adaptive routing

        Uses TWAP as baseline, but adapts based on:
        - Market conditions (volatility, spread, volume)
        - Urgency
        - Fill rate
        """
        # Start with TWAP baseline
        child_orders = self.twap_slicer.slice_order(instruction)

        # Adapt based on urgency
        if instruction.urgency > 0.8:
            # High urgency: use market orders
            for order in child_orders:
                order.order_type = "market"
                order.limit_price = None

        elif instruction.urgency < 0.3:
            # Low urgency: use passive limit orders
            for order in child_orders:
                order.order_type = "limit"
                # Set limit further from mid for better price
                if instruction.limit_price:
                    if instruction.side == "buy":
                        order.limit_price = instruction.limit_price * 0.999
                    else:
                        order.limit_price = instruction.limit_price * 1.001

        return child_orders


class ExecutionSimulator:
    """
    Simulate execution of child orders against market data
    """

    def __init__(self, spread_bps: float = 5.0, fill_probability: float = 0.95):
        self.spread_bps = spread_bps
        self.fill_probability = fill_probability

    def simulate_execution(
        self, child_orders: List[ChildOrder], market_data: pd.DataFrame
    ) -> Tuple[List[ChildOrder], Dict[str, Any]]:
        """
        Simulate execution of child orders

        Args:
            child_orders: List of child orders to execute
            market_data: Market OHLCV data

        Returns:
            (filled_orders, execution_stats)
        """
        filled_orders = []
        execution_stats = {
            "total_filled": 0,
            "total_quantity": sum(o.quantity for o in child_orders),
            "avg_fill_price": 0,
            "total_slippage_bps": 0,
            "fill_rate": 0,
        }

        total_filled_value = 0
        total_filled_quantity = 0

        for order in child_orders:
            # Find the market bar closest to order timestamp
            if order.timestamp not in market_data.index:
                closest_time = (
                    market_data.index[market_data.index >= order.timestamp][0]
                    if any(market_data.index >= order.timestamp)
                    else market_data.index[-1]
                )
            else:
                closest_time = order.timestamp

            bar = market_data.loc[closest_time]

            # Simulate fill
            if order.order_type == "market":
                # Market order: always fills
                if order.side == "buy":
                    fill_price = bar["Close"] * (1 + self.spread_bps / 2 / 10000)
                else:
                    fill_price = bar["Close"] * (1 - self.spread_bps / 2 / 10000)

                order.filled_quantity = order.quantity
                order.avg_fill_price = fill_price
                order.status = "filled"

            elif order.order_type == "limit":
                # Limit order: fills if price touches limit
                filled = False

                if order.side == "buy" and order.limit_price is not None:
                    if bar["Low"] <= order.limit_price:
                        filled = np.random.random() < self.fill_probability
                        fill_price = min(order.limit_price, bar["Open"])
                elif order.side == "sell" and order.limit_price is not None:
                    if bar["High"] >= order.limit_price:
                        filled = np.random.random() < self.fill_probability
                        fill_price = max(order.limit_price, bar["Open"])

                if filled:
                    order.filled_quantity = order.quantity
                    order.avg_fill_price = fill_price
                    order.status = "filled"
                else:
                    order.status = "unfilled"

            # Track statistics
            if order.status == "filled":
                filled_orders.append(order)
                total_filled_quantity += order.filled_quantity
                total_filled_value += order.filled_quantity * order.avg_fill_price

        # Calculate statistics
        if total_filled_quantity > 0:
            execution_stats["total_filled"] = total_filled_quantity
            execution_stats["avg_fill_price"] = (
                total_filled_value / total_filled_quantity
            )
            execution_stats["fill_rate"] = (
                total_filled_quantity / execution_stats["total_quantity"]
            )

            # Calculate slippage vs initial mid price
            initial_price = market_data.iloc[0]["Close"]
            slippage_pct = (
                execution_stats["avg_fill_price"] - initial_price
            ) / initial_price
            execution_stats["total_slippage_bps"] = slippage_pct * 10000

        logger.info(
            f"Execution complete: {execution_stats['total_filled']:.0f} / {execution_stats['total_quantity']:.0f} filled "
            f"({execution_stats['fill_rate']*100:.1f}%), "
            f"avg price: {execution_stats['avg_fill_price']:.2f}, "
            f"slippage: {execution_stats['total_slippage_bps']:.2f} bps"
        )

        return filled_orders, execution_stats


class PreTradeRiskCheck:
    """
    Pre-trade risk checks before order submission
    """

    def __init__(self, config: Dict[str, Any]):
        self.max_order_size = config.get("max_order_size", 10000)
        self.max_position_size = config.get("max_position_size", 100000)
        self.max_concentration = config.get(
            "max_concentration", 0.10
        )  # 10% of portfolio
        self.max_adv_participation = config.get(
            "max_adv_participation", 0.25
        )  # 25% of ADV

    def check_order(
        self,
        instruction: ExecutionInstruction,
        current_position: float,
        portfolio_value: float,
        avg_daily_volume: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Run pre-trade checks

        Args:
            instruction: Execution instruction
            current_position: Current position in symbol
            portfolio_value: Total portfolio value
            avg_daily_volume: Average daily volume

        Returns:
            (passed, error_message)
        """
        # Check order size
        if instruction.total_quantity > self.max_order_size:
            return (
                False,
                f"Order size {instruction.total_quantity} exceeds max {self.max_order_size}",
            )

        # Check resulting position size
        if instruction.side == "buy":
            resulting_position = current_position + instruction.total_quantity
        else:
            resulting_position = current_position - instruction.total_quantity

        if abs(resulting_position) > self.max_position_size:
            return (
                False,
                f"Resulting position {resulting_position} exceeds max {self.max_position_size}",
            )

        # Check concentration
        # (Simplified: assumes $100 per share)
        position_value = abs(resulting_position) * 100
        concentration = position_value / portfolio_value if portfolio_value > 0 else 0

        if concentration > self.max_concentration:
            return (
                False,
                f"Position concentration {concentration*100:.1f}% exceeds max {self.max_concentration*100:.1f}%",
            )

        # Check participation rate
        participation = (
            instruction.total_quantity / avg_daily_volume if avg_daily_volume > 0 else 0
        )

        if participation > self.max_adv_participation:
            return (
                False,
                f"Participation rate {participation*100:.1f}% exceeds max {self.max_adv_participation*100:.1f}%",
            )

        return True, None
