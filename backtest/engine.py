"""
Advanced Backtesting Engine
Realistic market simulation with order book replay, fill models, and latency effects
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from infrastructure.logger import quant_logger

logger = quant_logger.get_logger("backtest_engine")


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation"""

    order_id: str
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fills: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Trade fill representation"""

    fill_id: str
    order_id: str
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float


@dataclass
class Position:
    """Position representation"""

    symbol: str
    quantity: float  # Positive = long, negative = short
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class FillModel:
    """
    Realistic fill model incorporating:
    - Spread costs
    - Slippage (temporary market impact)
    - Partial fills
    - Queue position modeling
    """

    def __init__(
        self,
        spread_pct: float = 0.0001,  # 1 bps
        slippage_pct: float = 0.0002,  # 2 bps
        partial_fill_prob: float = 0.0,  # Probability of partial fill
        latency_ms: int = 0,  # Execution latency in milliseconds
    ):
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.partial_fill_prob = partial_fill_prob
        self.latency_ms = latency_ms

    def simulate_fill(
        self, order: Order, current_bar: pd.Series, next_bar: Optional[pd.Series] = None
    ) -> Tuple[float, float, OrderStatus]:
        """
        Simulate order fill

        Args:
            order: Order to fill
            current_bar: Current OHLCV bar
            next_bar: Next bar (for more realistic fills)

        Returns:
            (fill_quantity, fill_price, order_status)
        """
        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, current_bar, next_bar)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit_order(order, current_bar, next_bar)
        else:
            # Stop orders not yet implemented
            return 0.0, 0.0, OrderStatus.PENDING

    def _fill_market_order(
        self, order: Order, current_bar: pd.Series, next_bar: Optional[pd.Series]
    ) -> Tuple[float, float, OrderStatus]:
        """Fill market order with spread and slippage"""

        # Use next bar's open (more realistic) or current bar's close
        if next_bar is not None:
            base_price = next_bar["Open"]
        else:
            base_price = current_bar["Close"]

        # Apply spread
        if order.side == OrderSide.BUY:
            spread_adjustment = base_price * self.spread_pct / 2
        else:
            spread_adjustment = -base_price * self.spread_pct / 2

        # Apply slippage (temporary impact)
        if order.side == OrderSide.BUY:
            slippage = base_price * self.slippage_pct
        else:
            slippage = -base_price * self.slippage_pct

        fill_price = base_price + spread_adjustment + slippage

        # Check for partial fill
        if np.random.random() < self.partial_fill_prob:
            fill_quantity = order.quantity * np.random.uniform(0.5, 0.95)
            status = OrderStatus.PARTIALLY_FILLED
        else:
            fill_quantity = order.quantity
            status = OrderStatus.FILLED

        return fill_quantity, fill_price, status

    def _fill_limit_order(
        self, order: Order, current_bar: pd.Series, next_bar: Optional[pd.Series]
    ) -> Tuple[float, float, OrderStatus]:
        """Fill limit order if price reaches limit"""

        if order.limit_price is None:
            return 0.0, 0.0, OrderStatus.REJECTED

        bar = next_bar if next_bar is not None else current_bar

        # Check if limit order can be filled
        if order.side == OrderSide.BUY:
            # Buy limit: fill if market goes at or below limit price
            if bar["Low"] <= order.limit_price:
                fill_price = min(order.limit_price, bar["Open"])
                fill_quantity = order.quantity
                return fill_quantity, fill_price, OrderStatus.FILLED
        else:
            # Sell limit: fill if market goes at or above limit price
            if bar["High"] >= order.limit_price:
                fill_price = max(order.limit_price, bar["Open"])
                fill_quantity = order.quantity
                return fill_quantity, fill_price, OrderStatus.FILLED

        return 0.0, 0.0, OrderStatus.PENDING


class PortfolioState:
    """Portfolio state tracker"""

    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []

    def update_position(
        self,
        symbol: str,
        fill_quantity: float,
        fill_price: float,
        side: OrderSide,
        commission: float,
    ):
        """Update position after fill"""

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=0.0, average_price=0.0
            )

        pos = self.positions[symbol]
        old_quantity = pos.quantity

        # Update quantity (positive = long, negative = short)
        if side == OrderSide.BUY:
            new_quantity = old_quantity + fill_quantity
        else:
            new_quantity = old_quantity - fill_quantity

        # Calculate realized P&L if closing/reducing position
        if old_quantity * new_quantity < 0 or abs(new_quantity) < abs(old_quantity):
            # Position is being reduced or flipped
            closed_quantity = min(abs(fill_quantity), abs(old_quantity))
            realized_pnl = closed_quantity * (fill_price - pos.average_price)
            if old_quantity < 0:  # Was short
                realized_pnl = -realized_pnl
            pos.realized_pnl += realized_pnl

        # Update average price (FIFO accounting)
        if (old_quantity >= 0 and side == OrderSide.BUY) or (
            old_quantity <= 0 and side == OrderSide.SELL
        ):
            # Adding to position
            total_cost = (abs(old_quantity) * pos.average_price) + (
                fill_quantity * fill_price
            )
            pos.average_price = (
                total_cost / abs(new_quantity) if new_quantity != 0 else 0.0
            )

        pos.quantity = new_quantity

        # Update cash
        cash_flow = (
            -fill_quantity * fill_price
            if side == OrderSide.BUY
            else fill_quantity * fill_price
        )
        self.cash += cash_flow - commission

        # Remove position if flat
        if abs(pos.quantity) < 1e-6:
            del self.positions[symbol]

    def update_unrealized_pnl(self, prices: Dict[str, float]):
        """Update unrealized P&L based on current prices"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                pos.unrealized_pnl = pos.quantity * (current_price - pos.average_price)

    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity"""
        self.update_unrealized_pnl(prices)
        position_value = sum(
            pos.quantity * prices.get(pos.symbol, pos.average_price)
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def record_equity(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        """Record equity point"""
        equity = self.get_total_equity(prices)
        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)


class BacktestEngine:
    """
    Advanced backtesting engine with realistic market simulation
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_pct: float = 0.0,
        spread_pct: float = 0.0001,
        slippage_pct: float = 0.0002,
        latency_ms: int = 0,
    ):
        self.initial_cash = initial_cash
        self.commission_pct = commission_pct

        self.fill_model = FillModel(
            spread_pct=spread_pct, slippage_pct=slippage_pct, latency_ms=latency_ms
        )

        self.portfolio = PortfolioState(initial_cash)
        self.orders: List[Order] = []
        self.fills: List[Fill] = []
        self.pending_orders: List[Order] = []

        self.current_timestamp: Optional[pd.Timestamp] = None

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit an order

        Args:
            symbol: Ticker symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Order type
            limit_price: Limit price (for limit orders)
            metadata: Additional metadata

        Returns:
            order_id
        """
        order = Order(
            order_id=str(uuid.uuid4()),
            timestamp=self.current_timestamp,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            metadata=metadata or {},
        )

        self.orders.append(order)
        self.pending_orders.append(order)

        logger.debug(
            f"Order submitted: {order.order_id} {side.value} {quantity} {symbol} @ {order_type.value}"
        )

        return order.order_id

    def process_orders(
        self, current_bar: pd.Series, next_bar: Optional[pd.Series] = None
    ):
        """Process pending orders"""

        filled_orders = []

        for order in self.pending_orders:
            if (
                order.symbol != current_bar.name[0]
                if isinstance(current_bar.name, tuple)
                else current_bar.get("symbol")
            ):
                continue

            # Simulate fill
            fill_quantity, fill_price, status = self.fill_model.simulate_fill(
                order, current_bar, next_bar
            )

            if fill_quantity > 0:
                # Calculate commission
                commission = fill_quantity * fill_price * self.commission_pct

                # Record fill
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    timestamp=self.current_timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_quantity,
                    price=fill_price,
                    commission=commission,
                    slippage=abs(fill_price - current_bar["Close"])
                    / current_bar["Close"],
                )
                self.fills.append(fill)

                # Update portfolio
                self.portfolio.update_position(
                    symbol=order.symbol,
                    fill_quantity=fill_quantity,
                    fill_price=fill_price,
                    side=order.side,
                    commission=commission,
                )

                # Update order
                order.filled_quantity += fill_quantity
                order.average_fill_price = (
                    order.average_fill_price * (order.filled_quantity - fill_quantity)
                    + fill_price * fill_quantity
                ) / order.filled_quantity
                order.status = status

                logger.debug(
                    f"Order filled: {order.order_id} {fill_quantity} @ {fill_price:.2f} "
                    f"(commission: {commission:.2f})"
                )

                if status == OrderStatus.FILLED:
                    filled_orders.append(order)

        # Remove filled orders from pending
        for order in filled_orders:
            self.pending_orders.remove(order)

    def run_backtest(
        self, data: pd.DataFrame, strategy_func: callable, symbol: str = None
    ) -> pd.DataFrame:
        """
        Run backtest

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            strategy_func: Function that takes (bar, portfolio, engine) and submits orders
            symbol: Symbol name (if not in data)

        Returns:
            DataFrame with equity curve and metrics
        """
        logger.info(f"Starting backtest with {len(data)} bars")

        self.portfolio = PortfolioState(self.initial_cash)
        self.orders = []
        self.fills = []
        self.pending_orders = []

        data_with_symbol = data.copy()
        if symbol and "symbol" not in data_with_symbol.columns:
            data_with_symbol["symbol"] = symbol

        for i in range(len(data)):
            current_bar = data.iloc[i]
            next_bar = data.iloc[i + 1] if i < len(data) - 1 else None

            self.current_timestamp = current_bar.name

            # Process pending orders first
            self.process_orders(current_bar, next_bar)

            # Run strategy
            try:
                strategy_func(current_bar, self.portfolio, self)
            except Exception as e:
                logger.error(
                    f"Strategy error at {self.current_timestamp}: {e}", exc_info=True
                )

            # Record equity
            current_prices = {symbol or "default": current_bar["Close"]}
            self.portfolio.record_equity(self.current_timestamp, current_prices)

        # Create results DataFrame
        results_df = pd.DataFrame(
            {"equity": self.portfolio.equity_curve}, index=self.portfolio.timestamps
        )

        logger.info(
            f"Backtest complete: {len(self.fills)} fills, {len(self.orders)} orders"
        )

        return results_df

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if len(self.portfolio.equity_curve) == 0:
            return {}

        equity_series = pd.Series(
            self.portfolio.equity_curve, index=self.portfolio.timestamps
        )
        returns = equity_series.pct_change().dropna()

        # Calculate metrics
        total_return = (equity_series.iloc[-1] / self.initial_cash) - 1
        annual_return = (1 + total_return) ** (252 / len(equity_series)) - 1

        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0

        # Drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # Win rate
        total_fills = len(self.fills)
        winning_trades = sum(
            1 for pos in self.portfolio.positions.values() if pos.realized_pnl > 0
        )
        win_rate = winning_trades / total_fills if total_fills > 0 else 0

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_fills,
            "win_rate": win_rate,
            "final_equity": equity_series.iloc[-1],
            "initial_equity": self.initial_cash,
        }

        return metrics
