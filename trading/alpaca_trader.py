"""
Alpaca Live Trading Module
"""
import datetime
import time
from typing import Dict, List
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from config.settings import AlpacaConfig
from optimization.metrics import PerformanceMetrics

# Try to import Alpaca - will be used if available
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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
        base_url: str,
        symbol: str, 
        params: Dict, 
        df_dict: Dict, 
        timeframes: List[str],
        position_size_pct: float = 0.05
    ):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.yfinance_symbol = symbol
        self.symbol = AlpacaConfig.get_alpaca_symbol(symbol)
        self.params = params
        self.df_dict = df_dict
        self.timeframes = timeframes
        self.running = False
        self.position = False
        self.last_bar_time = {}
        self.position_size_pct = position_size_pct

        print(f"🔄 Ticker mapping: {self.yfinance_symbol} (yfinance) -> {self.symbol} (Alpaca)")
        
    def run(self):
        """Main trading loop"""
        if not ALPACA_AVAILABLE:
            self.error.emit("Alpaca library not installed!")
            return
            
        try:
            # Initialize Alpaca clients
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
            
            # Check if crypto to use crypto data client
            is_crypto = '/' in self.symbol
            
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
            self.status_update.emit(f"✓ Connected to Alpaca Paper Trading")
            self.status_update.emit(f"Account Balance: ${float(account.cash):.2f}")
            
            # Check current position
            try:
                position = self.trading_client.get_open_position(self.symbol)
                self.position = True
                self.status_update.emit(f"⚠ Already in position: {position.qty} of {self.symbol}")
            except:
                self.position = False
                self.status_update.emit(f"No existing position in {self.symbol}")
            
            self.running = True
            self.status_update.emit("🔴 LIVE - Monitoring for signals...")
            
            # Main monitoring loop
            while self.running:
                try:
                    # Get latest bars for all timeframes
                    current_signals = self.get_current_signals()
                    
                    if current_signals is None:
                        time.sleep(5)
                        continue
                    
                    # Check for entry signal
                    if not self.position and current_signals['should_enter']:
                        self.execute_entry()
                    
                    # Check for exit signal
                    elif self.position and current_signals['should_exit']:
                        self.execute_exit()
                    
                    # Sleep before next check
                    if '5min' in self.timeframes:
                        time.sleep(5)
                    elif 'hourly' in self.timeframes:
                        time.sleep(30)
                    else:
                        time.sleep(300)
                        
                except Exception as e:
                    self.error.emit(f"Monitoring error: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            self.error.emit(f"Alpaca connection error: {e}")
            
    def get_current_signals(self) -> Dict:
        """Fetch latest data and calculate signals"""
        try:
            signals_dict = {}
            
            for tf in self.timeframes:
                # Determine timeframe for Alpaca
                if tf == '5min':
                    alpaca_tf = TimeFrame(5, TimeFrameUnit.Minute)
                    bars_needed = 200
                elif tf == 'hourly':
                    alpaca_tf = TimeFrame(1, TimeFrameUnit.Hour)
                    bars_needed = 200
                elif tf == 'daily':
                    alpaca_tf = TimeFrame(1, TimeFrameUnit.Day)
                    bars_needed = 200
                else:
                    continue
                
                # Get historical bars
                if self.is_crypto:
                    request = CryptoBarsRequest(
                        symbol_or_symbols=self.symbol,
                        timeframe=alpaca_tf,
                        limit=bars_needed
                    )
                    bars = self.data_client.get_crypto_bars(request)
                else:
                    request = StockBarsRequest(
                        symbol_or_symbols=self.symbol,
                        timeframe=alpaca_tf,
                        limit=bars_needed
                    )
                    bars = self.data_client.get_stock_bars(request)
                
                df = bars.df
                
                if df.empty:
                    return None
                
                # Reset index
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)
                
                df = df.reset_index()
                df = df.rename(columns={'timestamp': 'Datetime'})
                
                # Check if we have a new bar (no repainting)
                latest_time = df['Datetime'].iloc[-1]
                if tf in self.last_bar_time and latest_time == self.last_bar_time[tf]:
                    signal_df = df.iloc[:-1].copy()
                else:
                    self.last_bar_time[tf] = latest_time
                    signal_df = df.iloc[:-1].copy()
                
                if len(signal_df) < 2:
                    return None
                
                # Calculate RSI
                close = signal_df['close'].to_numpy(dtype=float)
                mn1 = int(self.params[f'MN1_{tf}'])
                mn2 = int(self.params[f'MN2_{tf}'])
                
                rsi = PerformanceMetrics.compute_rsi_vectorized(close, mn1)
                rsi_smooth = PerformanceMetrics.smooth_vectorized(rsi, mn2)
                
                # Calculate cycle
                on = int(self.params[f'On_{tf}'])
                off = int(self.params[f'Off_{tf}'])
                start = int(self.params[f'Start_{tf}'])
                bar_index = len(signal_df) - 1
                cycle = ((bar_index - start) % (on + off)) < on
                
                # Get current signal
                current_rsi = rsi_smooth[-1]
                entry_threshold = self.params[f'Entry_{tf}']
                exit_threshold = self.params[f'Exit_{tf}']
                
                signals_dict[tf] = {
                    'rsi': current_rsi,
                    'cycle': cycle,
                    'entry': entry_threshold,
                    'exit': exit_threshold,
                    'should_enter': (current_rsi < entry_threshold) and cycle,
                    'should_exit': (current_rsi > exit_threshold) or (not cycle)
                }
            
            # Combine signals (AND for entry, OR for exit)
            should_enter = all(signals_dict[tf]['should_enter'] for tf in self.timeframes)
            should_exit = any(signals_dict[tf]['should_exit'] for tf in self.timeframes)
            
            return {
                'should_enter': should_enter,
                'should_exit': should_exit,
                'signals': signals_dict
            }
            
        except Exception as e:
            print(f"Signal calculation error: {e}")
            return None
    
    def execute_entry(self):
        """Execute buy order"""
        try:
            account = self.trading_client.get_account()
            cash = float(account.cash)
        
            # Get current price
            if self.is_crypto:
                request = CryptoBarsRequest(
                    symbol_or_symbols=self.symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    limit=1
                )
                bars = self.data_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=self.symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    limit=1
                )
                bars = self.data_client.get_stock_bars(request)

            current_price = bars.df['close'].iloc[-1]
            is_crypto = '/' in self.symbol

            position_size_pct = self.position_size_pct

            if is_crypto:
                # Crypto: use notional order
                notional = cash * position_size_pct

                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )

                order = self.trading_client.submit_order(order_data)
                shares_text = f"${notional:.2f}"
            else:
                # Stocks: use qty
                shares = int((cash * position_size_pct) / current_price)

                if shares < 1:
                    self.error.emit(f"Insufficient funds for position")
                    return

                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )

                order = self.trading_client.submit_order(order_data)
                shares_text = f"{shares} shares"

            self.position = True
            trade_info = {
                'action': 'BUY',
                'symbol': self.symbol,
                'shares': shares_text,
                'price': current_price,
                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            self.trade_executed.emit(trade_info)
            self.status_update.emit(
                f"✓ BUY: {shares_text} of {self.symbol} @ ${current_price:.2f}"
            )

        except Exception as e:
            self.error.emit(f"Entry execution error: {e}")
    
    def execute_exit(self):
        """Execute sell order"""
        try:
            position = self.trading_client.get_open_position(self.symbol)
            is_crypto = '/' in self.symbol
            
            # Get current price
            if self.is_crypto:
                request = CryptoBarsRequest(
                    symbol_or_symbols=self.symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    limit=1
                )
                bars = self.data_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=self.symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    limit=1
                )
                bars = self.data_client.get_stock_bars(request)
            
            current_price = bars.df['close'].iloc[-1]
            
            if is_crypto:
                self.trading_client.close_position(self.symbol)
                shares_text = f"{position.qty} (fractional)"
            else:
                shares = abs(int(position.qty))
                
                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order_data)
                shares_text = f"{shares} shares"
            
            self.position = False
            trade_info = {
                'action': 'SELL',
                'symbol': self.symbol,
                'shares': shares_text,
                'price': current_price,
                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.trade_executed.emit(trade_info)
            self.status_update.emit(f"✓ SELL: {shares_text} of {self.symbol} @ ${current_price:.2f}")
            
        except Exception as e:
            self.error.emit(f"Exit execution error: {e}")
    
    def stop(self):
        """Stop the trading loop"""
        self.running = False
