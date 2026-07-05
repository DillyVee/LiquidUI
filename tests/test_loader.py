"""
Unit tests for the data loader (offline paths only)
"""

import numpy as np
import pandas as pd

from data.loader import DataLoader


class TestTickerNormalization:
    def test_crypto_shortcuts(self):
        assert DataLoader.normalize_ticker("btc") == "BTC-USD"
        assert DataLoader.normalize_ticker(" ETH ") == "ETH-USD"
        assert DataLoader.normalize_ticker("DOGECOIN") == "DOGE-USD"

    def test_regular_tickers_untouched(self):
        assert DataLoader.normalize_ticker("spy") == "SPY"
        assert DataLoader.normalize_ticker("AAPL") == "AAPL"

    def test_is_crypto(self):
        assert DataLoader.is_crypto("BTC-USD")
        assert DataLoader.is_crypto("ETH/USD")
        assert not DataLoader.is_crypto("AAPL")


class TestProcessDataframe:
    @staticmethod
    def _sample_df(n=10):
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": np.linspace(100, 110, n),
                "High": np.linspace(101, 111, n),
                "Low": np.linspace(99, 109, n),
                "Close": np.linspace(100, 110, n),
                "Volume": np.full(n, 1_000_000),
            }
        ).set_index("Date")

    def test_renames_date_and_strips_timezone(self):
        df = DataLoader._process_dataframe(self._sample_df())
        assert "Datetime" in df.columns
        assert df["Datetime"].dt.tz is None

    def test_removes_nonpositive_prices(self):
        raw = self._sample_df().reset_index()
        raw.loc[3, "Close"] = -1.0
        raw.loc[5, "Close"] = 0.0
        df = DataLoader._process_dataframe(raw.set_index("Date"))
        assert (df["Close"] > 0).all()
        assert len(df) == 8

    def test_removes_duplicate_timestamps(self):
        raw = self._sample_df().reset_index()
        raw.loc[4, "Date"] = raw.loc[3, "Date"]
        df = DataLoader._process_dataframe(raw.set_index("Date"))
        assert not df["Datetime"].duplicated().any()


class TestFilterTimeframeData:
    def test_no_limit_copies(self):
        df = pd.DataFrame(
            {
                "Datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
                "Close": range(5),
            }
        )
        result = DataLoader.filter_timeframe_data({"daily": df}, limit_days=None)
        assert len(result["daily"]) == 5
        assert result["daily"] is not df

    def test_limit_days(self):
        df = pd.DataFrame(
            {
                "Datetime": pd.date_range("2024-01-01", periods=100, freq="D"),
                "Close": range(100),
            }
        )
        result = DataLoader.filter_timeframe_data({"daily": df}, limit_days=10)
        assert len(result["daily"]) == 11  # inclusive cutoff
