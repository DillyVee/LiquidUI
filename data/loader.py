"""
Data Loading Module
"""
import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf


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

            if cls.is_crypto(symbol):
                print(f"  📊 Crypto detected")

            # Daily data is required (10 years max)
            df_daily = cls._download(symbol, end_date, interval="1d", days=10 * 365)
            if df_daily.empty:
                return {}, f"No data found for {symbol}. Check ticker format."

            df_dict = {'daily': cls._process_dataframe(df_daily)}

            # Intraday data is optional - Yahoo limits how far back it goes
            # (hourly ~730 days, 5-minute ~60 days) and it is not available
            # for all tickers
            df_hourly = cls._download(
                symbol, end_date, interval="1h", days=729, retry_days=89
            )
            if not df_hourly.empty:
                df_dict['hourly'] = cls._process_dataframe(df_hourly)

            df_5min = cls._download(
                symbol, end_date, interval="5m", days=59, retry_days=29
            )
            if not df_5min.empty:
                df_dict['5min'] = cls._process_dataframe(df_5min)

            summary = ", ".join(f"{tf}={len(df)}" for tf, df in df_dict.items())
            print(f"✓ Loaded {symbol}: {summary}")

            return df_dict, ""

        except Exception as e:
            return {}, f"Failed to load {symbol}: {str(e)}"

    @staticmethod
    def _download(
        symbol: str,
        end_date: datetime.datetime,
        interval: str,
        days: int,
        retry_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Download one interval from Yahoo, optionally retrying a shorter window"""
        start = end_date - datetime.timedelta(days=days)
        print(f"  {interval}: {start.date()} to {end_date.date()}")
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty and retry_days is not None:
            print(f"  ⚠ Retrying {interval} with {retry_days}-day window...")
            start = end_date - datetime.timedelta(days=retry_days)
            df = yf.download(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

        return df
    
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
            if isinstance(df['Datetime'].dtype, pd.DatetimeTZDtype):
                df['Datetime'] = df['Datetime'].dt.tz_localize(None)

        # Data quality validation
        if len(df) > 0:
            # Check for required columns
            required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  Warning: Missing columns: {missing_cols}")

            # Check for zero/negative prices (data error)
            if 'Close' in df.columns:
                invalid_prices = (df['Close'] <= 0).sum()
                if invalid_prices > 0:
                    print(f"⚠️  Warning: Found {invalid_prices} zero/negative prices, removing...")
                    df = df[df['Close'] > 0].copy()

            # Check for duplicate timestamps
            if 'Datetime' in df.columns:
                duplicates = df['Datetime'].duplicated().sum()
                if duplicates > 0:
                    print(f"⚠️  Warning: Found {duplicates} duplicate timestamps, removing...")
                    df = df.drop_duplicates(subset=['Datetime'], keep='last').copy()

            # Check for extreme price movements (potential data errors)
            if 'Close' in df.columns and len(df) > 1:
                returns = df['Close'].pct_change()
                extreme_moves = (returns.abs() > 0.5).sum()  # >50% move in one bar
                if extreme_moves > 0:
                    print(f"⚠️  Warning: Found {extreme_moves} extreme price movements (>50%)")

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
