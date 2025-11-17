"""
Data Loading Module
"""
import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

from config.settings import OptimizationConfig


class DataLoader:
    """Handle data loading from various sources"""
    
    # Crypto ticker normalization
    CRYPTO_MAP = {
        'BTC': 'BTC-USD',
        'BITCOIN': 'BTC-USD',
        'ETH': 'ETH-USD',
        'ETHEREUM': 'ETH-USD',
        'SOL': 'SOL-USD',
        'SOLANA': 'SOL-USD',
        'AVAX': 'AVAX-USD',
        'AVALANCHE': 'AVAX-USD',
        'DOGE': 'DOGE-USD',
        'DOGECOIN': 'DOGE-USD'
    }
    
    @classmethod
    def normalize_ticker(cls, symbol: str) -> str:
        """Normalize crypto ticker symbols"""
        symbol = symbol.strip().upper()
        return cls.CRYPTO_MAP.get(symbol, symbol)
    
    @classmethod
    def is_crypto(cls, symbol: str) -> bool:
        """Check if symbol is cryptocurrency"""
        return '-USD' in symbol or '/USD' in symbol
    
    @classmethod
    def load_yfinance_data(cls, symbol: str) -> Tuple[Dict[str, pd.DataFrame], str]:
        """
        Load maximum available data from Yahoo Finance
        
        Returns:
            Tuple of (df_dict, error_message)
            df_dict contains 'daily', 'hourly', '5min' dataframes
        """
        symbol = cls.normalize_ticker(symbol)
        end_date = datetime.datetime.today()
        
        try:
            print(f"\n📥 Downloading data for {symbol}...")
            
            is_crypto = cls.is_crypto(symbol)
            if is_crypto:
                print(f"  📊 Crypto detected")
            
            # Load daily data (10 years max)
            start_daily = end_date - datetime.timedelta(days=10*365)
            print(f"  Daily: {start_daily.date()} to {end_date.date()}")
            df_daily = yf.download(
                symbol, 
                start=start_daily.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"), 
                interval="1d",
                progress=False, 
                auto_adjust=True
            )
            
            if df_daily.empty:
                return {}, f"No data found for {symbol}. Check ticker format."
            
            # Load hourly data (~730 days max)
            start_hourly = end_date - datetime.timedelta(days=729)
            print(f"  Hourly: {start_hourly.date()} to {end_date.date()}")
            df_hourly = yf.download(
                symbol,
                start=start_hourly.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1h",
                progress=False,
                auto_adjust=True
            )
            
            # Retry hourly with shorter window if needed
            if df_hourly.empty and not df_daily.empty:
                print(f"  ⚠ Retrying hourly with 90-day window...")
                start_hourly = end_date - datetime.timedelta(days=89)
                df_hourly = yf.download(
                    symbol,
                    start=start_hourly.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1h",
                    progress=False,
                    auto_adjust=True
                )
            
            if df_hourly.empty:
                return {}, f"Could not load hourly data for {symbol}"
            
            # Load 5-minute data (~60 days max)
            start_5min = end_date - datetime.timedelta(days=59)
            print(f"  5-minute: {start_5min.date()} to {end_date.date()}")
            df_5min = yf.download(
                symbol,
                start=start_5min.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="5m",
                progress=False,
                auto_adjust=True
            )
            
            # Retry 5min with shorter window if needed
            if df_5min.empty:
                print(f"  ⚠ Retrying 5-minute with 30-day window...")
                start_5min = end_date - datetime.timedelta(days=29)
                df_5min = yf.download(
                    symbol,
                    start=start_5min.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="5m",
                    progress=False,
                    auto_adjust=True
                )
            
            # Process dataframes
            df_daily = cls._process_dataframe(df_daily)
            df_hourly = cls._process_dataframe(df_hourly)
            
            df_dict = {
                'daily': df_daily,
                'hourly': df_hourly
            }
            
            if not df_5min.empty:
                df_5min = cls._process_dataframe(df_5min)
                df_dict['5min'] = df_5min
                print(f"  ✓ 5-minute: {len(df_5min)} bars")
            
            print(f"✓ Loaded {symbol}: Daily={len(df_daily)}, Hourly={len(df_hourly)}")
            
            return df_dict, ""
            
        except Exception as e:
            return {}, f"Failed to load {symbol}: {str(e)}"
    
    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe format"""
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Drop NaN and reset index
        df = df.dropna().reset_index(drop=False)
        
        # Rename Date/Datetime column
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'Datetime'})
        elif 'Datetime' not in df.columns and df.index.name in ['Date', 'Datetime']:
            df = df.reset_index()
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'Datetime'})
        
        # Ensure timezone-naive
        if 'Datetime' in df.columns:
            if pd.api.types.is_datetime64tz_dtype(df['Datetime']):
                df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        
        return df
    
    @staticmethod
    def filter_timeframe_data(
        df_dict_full: Dict[str, pd.DataFrame],
        limit_days: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter all timeframes to match a specific date limit
        
        Args:
            df_dict_full: Full data dictionary
            limit_days: Number of days to limit to (None = no limit)
        
        Returns:
            Filtered data dictionary
        """
        if limit_days is None:
            return {k: v.copy() for k, v in df_dict_full.items()}
        
        df_dict = {}
        end_date = None
        
        for tf in ['daily', 'hourly', '5min']:
            if tf not in df_dict_full:
                continue
            
            df = df_dict_full[tf].copy()
            
            # Get end date from data
            if end_date is None:
                end_date = df['Datetime'].max()
            
            # Calculate cutoff
            cutoff_date = end_date - pd.Timedelta(days=limit_days)
            
            # Filter
            df = df[df['Datetime'] >= cutoff_date].copy().reset_index(drop=True)
            df_dict[tf] = df
        
        return df_dict
