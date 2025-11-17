"""
Feature Engineering Pipeline
Production-grade feature computation with caching, lineage tracking, and validation
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hashlib
import json
from functools import wraps

from infrastructure.logger import quant_logger


logger = quant_logger.get_logger('feature_engineering')


@dataclass
class FeatureMetadata:
    """Metadata for a feature"""
    name: str
    description: str
    feature_type: str  # 'price', 'volume', 'microstructure', 'derived'
    dependencies: List[str]  # List of required input columns
    parameters: Dict[str, Any]
    version: str


class FeatureTransform(ABC):
    """Base class for feature transforms"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.dependencies: List[str] = []
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the feature"""
        pass

    def get_metadata(self) -> FeatureMetadata:
        """Get feature metadata"""
        return FeatureMetadata(
            name=self.name,
            description=self.description,
            feature_type=self.__class__.__name__,
            dependencies=self.dependencies,
            parameters=self.parameters,
            version="1.0"
        )

    def validate_inputs(self, df: pd.DataFrame):
        """Validate that required columns exist"""
        missing = [col for col in self.dependencies if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {self.name}: {missing}")


# ============================================================================
# Price-based Features
# ============================================================================

class Returns(FeatureTransform):
    """Simple or log returns"""

    def __init__(self, periods: int = 1, log: bool = False):
        super().__init__(f'returns_{periods}', f'{periods}-period {"log" if log else "simple"} returns')
        self.dependencies = ['Close']
        self.parameters = {'periods': periods, 'log': log}
        self.periods = periods
        self.log = log

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)
        if self.log:
            return np.log(df['Close'] / df['Close'].shift(self.periods))
        else:
            return df['Close'].pct_change(self.periods)


class RollingVolatility(FeatureTransform):
    """Rolling volatility (standard deviation of returns)"""

    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(f'volatility_{window}', f'{window}-period rolling volatility')
        self.dependencies = ['Close']
        self.parameters = {'window': window, 'annualize': annualize}
        self.window = window
        self.annualize = annualize

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)
        returns = df['Close'].pct_change()
        vol = returns.rolling(window=self.window).std()

        if self.annualize:
            # Annualization factor depends on data frequency
            # Assume daily for now; can be parameterized
            vol = vol * np.sqrt(252)

        return vol


class RSI(FeatureTransform):
    """Relative Strength Index"""

    def __init__(self, period: int = 14):
        super().__init__(f'rsi_{period}', f'{period}-period RSI')
        self.dependencies = ['Close']
        self.parameters = {'period': period}
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class BollingerBands(FeatureTransform):
    """Bollinger Bands position"""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f'bb_position_{window}', f'{window}-period Bollinger Band position')
        self.dependencies = ['Close']
        self.parameters = {'window': window, 'num_std': num_std}
        self.window = window
        self.num_std = num_std

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        sma = df['Close'].rolling(window=self.window).mean()
        std = df['Close'].rolling(window=self.window).std()

        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)

        # Return position within bands: -1 (at lower), 0 (at middle), +1 (at upper)
        bb_position = (df['Close'] - sma) / (upper_band - lower_band)

        return bb_position


class TrueRange(FeatureTransform):
    """True Range (for ATR calculation)"""

    def __init__(self):
        super().__init__('true_range', 'True Range')
        self.dependencies = ['High', 'Low', 'Close']

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr


class ATR(FeatureTransform):
    """Average True Range"""

    def __init__(self, period: int = 14):
        super().__init__(f'atr_{period}', f'{period}-period Average True Range')
        self.dependencies = ['High', 'Low', 'Close']
        self.parameters = {'period': period}
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        tr = TrueRange().compute(df)
        atr = tr.rolling(window=self.period).mean()

        return atr


# ============================================================================
# Volume-based Features
# ============================================================================

class VolumeMA(FeatureTransform):
    """Volume moving average"""

    def __init__(self, period: int = 20):
        super().__init__(f'volume_ma_{period}', f'{period}-period volume MA')
        self.dependencies = ['Volume']
        self.parameters = {'period': period}
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)
        return df['Volume'].rolling(window=self.period).mean()


class VolumeRatio(FeatureTransform):
    """Volume relative to moving average"""

    def __init__(self, period: int = 20):
        super().__init__(f'volume_ratio_{period}', f'Volume / {period}-MA')
        self.dependencies = ['Volume']
        self.parameters = {'period': period}
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)
        volume_ma = df['Volume'].rolling(window=self.period).mean()
        return df['Volume'] / volume_ma


class VWAP(FeatureTransform):
    """Volume Weighted Average Price (intraday)"""

    def __init__(self):
        super().__init__('vwap', 'Volume Weighted Average Price')
        self.dependencies = ['High', 'Low', 'Close', 'Volume']

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

        return vwap


class OBV(FeatureTransform):
    """On-Balance Volume"""

    def __init__(self):
        super().__init__('obv', 'On-Balance Volume')
        self.dependencies = ['Close', 'Volume']

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv


# ============================================================================
# Microstructure Features
# ============================================================================

class HighLowSpread(FeatureTransform):
    """High-Low spread as proxy for intrabar volatility"""

    def __init__(self):
        super().__init__('hl_spread', 'High-Low spread')
        self.dependencies = ['High', 'Low', 'Close']

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)
        return (df['High'] - df['Low']) / df['Close']


class RollImpact(FeatureTransform):
    """Roll's spread estimator (measure of bid-ask bounce)"""

    def __init__(self, window: int = 20):
        super().__init__(f'roll_spread_{window}', 'Roll spread estimator')
        self.dependencies = ['Close']
        self.parameters = {'window': window}
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        price_changes = df['Close'].diff()
        autocovariance = price_changes.rolling(window=self.window).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1], raw=True
        )

        # Roll's estimate: spread = 2 * sqrt(-cov)
        # If cov is positive, there's no bid-ask bounce
        spread = 2 * np.sqrt(np.maximum(-autocovariance, 0))

        return spread


# ============================================================================
# Momentum & Trend Features
# ============================================================================

class MACD(FeatureTransform):
    """MACD (Moving Average Convergence Divergence)"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__('macd', 'MACD indicator')
        self.dependencies = ['Close']
        self.parameters = {'fast': fast, 'slow': slow, 'signal': signal}
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        ema_fast = df['Close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        # Return MACD line (can also return signal line and histogram)

        return macd


class ADX(FeatureTransform):
    """Average Directional Index (trend strength)"""

    def __init__(self, period: int = 14):
        super().__init__(f'adx_{period}', f'{period}-period ADX')
        self.dependencies = ['High', 'Low', 'Close']
        self.parameters = {'period': period}
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        self.validate_inputs(df)

        # Calculate +DM and -DM
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # True Range
        tr = TrueRange().compute(df)

        # Smooth
        atr = tr.rolling(window=self.period).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.period).mean() / atr)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=self.period).mean()

        return adx


# ============================================================================
# Feature Pipeline
# ============================================================================

class FeaturePipeline:
    """
    Feature engineering pipeline with caching and lineage tracking
    """

    def __init__(self, name: str):
        self.name = name
        self.transforms: List[FeatureTransform] = []
        self.feature_metadata: Dict[str, FeatureMetadata] = {}

    def add_feature(self, transform: FeatureTransform):
        """Add a feature transform to the pipeline"""
        self.transforms.append(transform)
        self.feature_metadata[transform.name] = transform.get_metadata()
        logger.info(f"Added feature: {transform.name}")

    def add_standard_features(self):
        """Add a standard set of features for quant trading"""
        logger.info("Adding standard feature set")

        # Price features
        self.add_feature(Returns(periods=1))
        self.add_feature(Returns(periods=5))
        self.add_feature(Returns(periods=20))
        self.add_feature(RollingVolatility(window=20))
        self.add_feature(RSI(period=14))
        self.add_feature(BollingerBands(window=20))
        self.add_feature(ATR(period=14))

        # Volume features
        self.add_feature(VolumeRatio(period=20))
        self.add_feature(OBV())

        # Microstructure
        self.add_feature(HighLowSpread())

        # Momentum
        self.add_feature(MACD())
        self.add_feature(ADX())

    def compute_features(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """
        Compute all features in the pipeline

        Args:
            df: Input OHLCV DataFrame
            dropna: Whether to drop NaN values

        Returns:
            DataFrame with all computed features
        """
        logger.info(f"Computing {len(self.transforms)} features")

        result = df.copy()

        for transform in self.transforms:
            try:
                feature_series = transform.compute(df)
                result[transform.name] = feature_series
                logger.debug(f"Computed feature: {transform.name}")

            except Exception as e:
                logger.error(f"Failed to compute feature {transform.name}: {e}", exc_info=True)
                # Add NaN column to maintain consistency
                result[transform.name] = np.nan

        if dropna:
            initial_len = len(result)
            result = result.dropna()
            logger.info(f"Dropped {initial_len - len(result)} rows with NaN values")

        logger.info(f"Feature computation complete: {len(result)} rows, {len(result.columns)} columns")

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return [t.name for t in self.transforms]

    def get_feature_config(self) -> Dict[str, Any]:
        """Get pipeline configuration for versioning"""
        return {
            'name': self.name,
            'features': {
                name: asdict(meta) for name, meta in self.feature_metadata.items()
            },
            'version': self._compute_config_hash()
        }

    def _compute_config_hash(self) -> str:
        """Compute hash of pipeline configuration for versioning"""
        config_str = json.dumps(self.get_feature_config(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]


# ============================================================================
# Feature Store Integration
# ============================================================================

class FeatureStore:
    """
    Feature store for caching and retrieving computed features
    Integrates with VersionedDataStore
    """

    def __init__(self, data_store):
        self.data_store = data_store
        self.pipelines: Dict[str, FeaturePipeline] = {}

    def register_pipeline(self, pipeline: FeaturePipeline):
        """Register a feature pipeline"""
        self.pipelines[pipeline.name] = pipeline
        logger.info(f"Registered feature pipeline: {pipeline.name}")

    def compute_and_store(
        self,
        df: pd.DataFrame,
        symbol: str,
        pipeline_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compute features using a pipeline and store them

        Args:
            df: Input OHLCV data
            symbol: Ticker symbol
            pipeline_name: Name of registered pipeline
            metadata: Additional metadata

        Returns:
            version_id of stored features
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not registered")

        pipeline = self.pipelines[pipeline_name]

        # Compute features
        features_df = pipeline.compute_features(df)

        # Store
        version_id = self.data_store.write_features(
            df=features_df,
            symbol=symbol,
            feature_set_name=pipeline_name,
            feature_config=pipeline.get_feature_config(),
            metadata=metadata
        )

        return version_id

    def get_features(
        self,
        symbol: str,
        pipeline_name: str,
        start_date=None,
        end_date=None,
        version_id=None
    ) -> pd.DataFrame:
        """Retrieve stored features"""
        return self.data_store.read_features(
            symbol=symbol,
            feature_set_name=pipeline_name,
            start_date=start_date,
            end_date=end_date,
            version_id=version_id
        )
