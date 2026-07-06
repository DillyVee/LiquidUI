"""
Alpaca Live Trading Module
"""

import datetime
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from config.settings import AlpacaConfig
from optimization.metrics import PerformanceMetrics

# Try to import Alpaca - will be used if available
try:
    from alpaca.data.historical import (
        CryptoHistoricalDataClient,
        StockHistoricalDataClient,
    )
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠ Alpaca not installed. Install with: pip install alpaca-py")


class AlpacaLiveTrader(QThread):
    """Live trading thread that monitors signals and executes trades"""

    status_update = pyqtSignal(str)
    trade_executed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        params: Dict,
        df_dict: Dict,
        timeframes: List[str],
        position_size_pct: float = 0.05,
        paper: bool = True,
    ):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.yfinance_symbol = symbol
        self.symbol = AlpacaConfig.get_alpaca_symbol(symbol)
        self.params = params
        self.df_dict = df_dict
        self.timeframes = timeframes
        self.running = False
        self.position = False
        self.last_bar_time = {}
        self.position_size_pct = position_size_pct
        # Quantity actually bought by this bot (from the confirmed fill), so
        # exits only sell what the bot bought - never manually-held assets
        self.entry_qty: Optional[float] = None

        print(f"🔄 Ticker mapping: {self.yfinance_symbol} (yfinance) -> {self.symbol} (Alpaca)")

    @property
    def is_running(self) -> bool:
        """Whether the trading loop is active"""
        return self.running

    def run(self):
        """Main trading loop"""
        if not ALPACA_AVAILABLE:
            self.error.emit("Alpaca library not installed!")
            return

        try:
            # Initialize Alpaca clients
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)

            # Check if crypto to use crypto data client
            is_crypto = "/" in self.symbol

            if is_crypto:
                try:
                    self.data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
                    self.is_crypto = True
                    self.status_update.emit(f"✓ Using Crypto API for {self.symbol}")
                    print(f"📊 Initialized Crypto Data Client for {self.symbol}")
                except ImportError:
                    self.error.emit("Crypto API not available in alpaca-py version")
                    return
            else:
                self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                self.is_crypto = False
                self.status_update.emit(f"✓ Using Stock API for {self.symbol}")

            # Get account info
            account = self.trading_client.get_account()
            mode = "Paper" if self.paper else "LIVE"
            self.status_update.emit(f"✓ Connected to Alpaca {mode} Trading")
            self.status_update.emit(f"Account Balance: ${float(account.cash):.2f}")

            # Check current position
            try:
                position = self.trading_client.get_open_position(self.symbol)
                self.position = True
                self.entry_qty = abs(float(position.qty))
                self.status_update.emit(f"⚠ Already in position: {position.qty} of {self.symbol}")
            except Exception:
                self.position = False
                self.entry_qty = None
                self.status_update.emit(f"No existing position in {self.symbol}")

            self.running = True
            self.status_update.emit("🔴 LIVE - Monitoring for signals...")

            # Main monitoring loop
            while self.running:
                try:
                    # Get latest bars for all timeframes
                    current_signals = self.get_current_signals()

                    if current_signals is None:
                        self._sleep(10)
                        continue

                    # Check for entry signal
                    if not self.position and current_signals["should_enter"]:
                        self.execute_entry()

                    # Check for exit signal
                    elif self.position and current_signals["should_exit"]:
                        self.execute_exit()

                    # Sleep before next check (interruptible so stop() takes
                    # effect within ~1 second)
                    if "5min" in self.timeframes:
                        self._sleep(30)
                    elif "hourly" in self.timeframes:
                        self._sleep(60)
                    else:
                        self._sleep(300)

                except Exception as e:
                    self.error.emit(f"Monitoring error: {e}")
                    self._sleep(10)

        except Exception as e:
            self.error.emit(f"Alpaca connection error: {e}")

    def get_current_signals(self) -> Dict:
        """Fetch latest data and calculate signals"""
        try:
            signals_dict = {}

            for tf in self.timeframes:
                mn1 = int(self.params[f"MN1_{tf}"])
                mn2 = int(self.params[f"MN2_{tf}"])

                # Bars required for a fully-warmed-up RSI + smoothing, plus buffer
                bars_needed = mn1 + mn2 + 10

                # Determine timeframe for Alpaca and a calendar window that is
                # guaranteed to contain enough TRADING bars. Passing limit
                # without start is unreliable (Alpaca counts forward from its
                # default start, so the newest bars may be missing entirely).
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                if tf == "5min":
                    alpaca_tf = TimeFrame(5, TimeFrameUnit.Minute)
                    # ~6.5 trading hours/day, weekends: ~6x calendar buffer
                    window = datetime.timedelta(minutes=5 * bars_needed * 6)
                elif tf == "hourly":
                    alpaca_tf = TimeFrame(1, TimeFrameUnit.Hour)
                    window = datetime.timedelta(hours=bars_needed * 6)
                elif tf == "daily":
                    alpaca_tf = TimeFrame(1, TimeFrameUnit.Day)
                    window = datetime.timedelta(days=int(bars_needed * 1.7) + 10)
                else:
                    continue

                # Get historical bars
                if self.is_crypto:
                    request = CryptoBarsRequest(
                        symbol_or_symbols=self.symbol,
                        timeframe=alpaca_tf,
                        start=now_utc - window,
                    )
                    bars = self.data_client.get_crypto_bars(request)
                else:
                    request = StockBarsRequest(
                        symbol_or_symbols=self.symbol,
                        timeframe=alpaca_tf,
                        start=now_utc - window,
                    )
                    bars = self.data_client.get_stock_bars(request)

                df = bars.df

                if df.empty:
                    return None

                # Reset index
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)

                df = df.reset_index()
                df = df.rename(columns={"timestamp": "Datetime"})

                # Normalize timezone to match the backtest data convention
                # (yfinance: exchange time for stocks, UTC for crypto - both
                # timezone-naive). Required so the calendar-anchored cycle
                # lands on the same phase as the backtest.
                if isinstance(df["Datetime"].dtype, pd.DatetimeTZDtype):
                    target_tz = "UTC" if self.is_crypto else "America/New_York"
                    df["Datetime"] = (
                        df["Datetime"].dt.tz_convert(target_tz).dt.tz_localize(None)
                    )

                # Signal off closed bars only - drop the still-forming bar
                # so signals never repaint
                self.last_bar_time[tf] = df["Datetime"].iloc[-1]
                signal_df = df.iloc[:-1].copy()

                if len(signal_df) < mn1 + mn2 + 1:
                    self.status_update.emit(
                        f"⏳ Waiting for indicator warm-up on {tf}: "
                        f"{len(signal_df)}/{mn1 + mn2 + 1} bars"
                    )
                    return None

                # Calculate RSI
                close = signal_df["close"].to_numpy(dtype=float)

                rsi = PerformanceMetrics.compute_rsi_vectorized(close, mn1)
                rsi_smooth = PerformanceMetrics.smooth_vectorized(rsi, mn2)

                # Calculate cycle (guard against a zero-length period).
                # Anchored to calendar time exactly like the backtest - the
                # previous index-in-window approach froze the cycle at a
                # constant phase because the fetch window slides with time.
                on = int(self.params[f"On_{tf}"])
                off = int(self.params[f"Off_{tf}"])
                start = int(self.params[f"Start_{tf}"])
                period = max(1, on + off)
                last_closed = signal_df["Datetime"].iloc[-1]
                anchor = int(
                    PerformanceMetrics.cycle_anchor_index(
                        np.array([last_closed.to_datetime64()]), tf
                    )[0]
                )
                cycle = ((anchor - start) % period) < on

                # Get current signal
                current_rsi = rsi_smooth[-1]
                entry_threshold = self.params[f"Entry_{tf}"]
                exit_threshold = self.params[f"Exit_{tf}"]

                signals_dict[tf] = {
                    "rsi": current_rsi,
                    "cycle": cycle,
                    "entry": entry_threshold,
                    "exit": exit_threshold,
                    "should_enter": (current_rsi < entry_threshold) and cycle,
                    "should_exit": (current_rsi > exit_threshold) or (not cycle),
                }

            # Combine signals (AND for entry, OR for exit)
            should_enter = all(signals_dict[tf]["should_enter"] for tf in self.timeframes)
            should_exit = any(signals_dict[tf]["should_exit"] for tf in self.timeframes)

            return {
                "should_enter": should_enter,
                "should_exit": should_exit,
                "signals": signals_dict,
            }

        except Exception as e:
            print(f"Signal calculation error: {e}")
            return None

    def execute_entry(self):
        """Execute buy order and confirm the fill before updating state"""
        try:
            account = self.trading_client.get_account()
            cash = float(account.cash)

            current_price = self._latest_price()

            if self.is_crypto:
                # Crypto: use notional order
                notional = cash * self.position_size_pct

                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )
            else:
                # Stocks: use qty
                shares = int((cash * self.position_size_pct) / current_price)

                if shares < 1:
                    self.error.emit("Insufficient funds for position")
                    return

                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )

            order = self.trading_client.submit_order(order_data)

            # Never assume a market order filled - a rejected/unfilled order
            # would otherwise desync the bot's position state
            filled_qty, avg_price = self._wait_for_fill(order.id)

            if filled_qty <= 0:
                self.error.emit(
                    f"Entry order for {self.symbol} did not fill "
                    "(rejected/cancelled/timeout) - no position opened"
                )
                return

            self.position = True
            self.entry_qty = filled_qty
            fill_price = avg_price if avg_price > 0 else current_price
            shares_text = f"{filled_qty:g} units"

            trade_info = {
                "action": "BUY",
                "symbol": self.symbol,
                "shares": shares_text,
                "price": fill_price,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            self.trade_executed.emit(trade_info)
            self.status_update.emit(f"✓ BUY: {shares_text} of {self.symbol} @ ${fill_price:.2f}")

        except Exception as e:
            self.error.emit(f"Entry execution error: {e}")

    def execute_exit(self):
        """Execute sell order and confirm the fill before updating state"""
        try:
            position = self.trading_client.get_open_position(self.symbol)
            available_qty = abs(float(position.qty))

            # Sell only what this bot bought (tracked from the entry fill),
            # never manually-held assets in the same symbol
            qty = min(self.entry_qty, available_qty) if self.entry_qty else available_qty

            if self.is_crypto:
                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                )
            else:
                qty = int(qty)
                if qty < 1:
                    self.position = False
                    self.entry_qty = None
                    return
                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )

            order = self.trading_client.submit_order(order_data)
            filled_qty, avg_price = self._wait_for_fill(order.id)

            if filled_qty <= 0:
                self.error.emit(
                    f"Exit order for {self.symbol} did not fill - still in position, "
                    "will retry on next signal check"
                )
                return

            if filled_qty < qty * 0.999:
                # Partial fill: remain in position with the remainder
                self.entry_qty = qty - filled_qty
                self.error.emit(
                    f"Exit partially filled ({filled_qty:g}/{qty:g}) - "
                    f"{self.entry_qty:g} units remain, will retry"
                )
            else:
                self.position = False
                self.entry_qty = None

            fill_price = avg_price if avg_price > 0 else self._latest_price()
            shares_text = f"{filled_qty:g} units"

            trade_info = {
                "action": "SELL",
                "symbol": self.symbol,
                "shares": shares_text,
                "price": fill_price,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            self.trade_executed.emit(trade_info)
            self.status_update.emit(
                f"✓ SELL: {shares_text} of {self.symbol} @ ${fill_price:.2f}"
            )

        except Exception as e:
            self.error.emit(f"Exit execution error: {e}")

    def _latest_price(self) -> float:
        """Fetch the most recent 1-minute close (used for sizing/reporting)"""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        # 5-day window covers weekends/holidays for stocks
        start = now_utc - datetime.timedelta(days=5)

        if self.is_crypto:
            request = CryptoBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
            )
            bars = self.data_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
            )
            bars = self.data_client.get_stock_bars(request)

        return float(bars.df["close"].iloc[-1])

    def _wait_for_fill(self, order_id, timeout: float = 60.0) -> Tuple[float, float]:
        """
        Poll an order until it reaches a terminal state or times out.

        Returns:
            (filled_qty, avg_fill_price); filled_qty is 0 if nothing filled
        """
        deadline = time.time() + timeout
        terminal = {OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}

        while time.time() < deadline:
            order = self.trading_client.get_order_by_id(order_id)
            filled_qty = float(order.filled_qty or 0)
            avg_price = float(order.filled_avg_price or 0)

            if order.status == OrderStatus.FILLED:
                return filled_qty, avg_price
            if order.status in terminal:
                return filled_qty, avg_price

            time.sleep(1)

        # Timeout: cancel the remainder, then report what actually filled
        try:
            self.trading_client.cancel_order_by_id(order_id)
        except Exception:
            pass
        time.sleep(1)
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return float(order.filled_qty or 0), float(order.filled_avg_price or 0)
        except Exception:
            return 0.0, 0.0

    def _sleep(self, seconds: float):
        """Sleep in 1s slices so stop() takes effect quickly"""
        deadline = time.time() + seconds
        while self.running and time.time() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.time())))

    def stop(self):
        """Stop the trading loop"""
        self.running = False
