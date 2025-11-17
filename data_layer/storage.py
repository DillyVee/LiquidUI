"""
Data Storage Layer - Immutable, Versioned, Time-Series Optimized
Supports Parquet with partitioning, data versioning, and metadata tracking
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, date
import json
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, asdict
import shutil

from infrastructure.logger import quant_logger


logger = quant_logger.get_logger('data_storage')


@dataclass
class DataVersion:
    """Data version metadata"""
    version_id: str
    created_at: str
    data_type: str  # 'raw', 'processed', 'features'
    symbol: str
    start_date: str
    end_date: str
    record_count: int
    checksum: str
    schema_version: str
    metadata: Dict[str, Any]


class VersionedDataStore:
    """
    Immutable, versioned data store with Parquet backend
    Inspired by data lakehouse patterns (Delta Lake, Iceberg)
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Directory structure
        self.raw_path = self.base_path / 'raw'
        self.processed_path = self.base_path / 'processed'
        self.features_path = self.base_path / 'features'
        self.metadata_path = self.base_path / '_metadata'

        for path in [self.raw_path, self.processed_path, self.features_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)

    def write_raw_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Write raw market data with automatic partitioning and versioning

        Args:
            df: DataFrame with DatetimeIndex
            symbol: Ticker symbol
            data_source: Source identifier (e.g., 'yahoo', 'alpaca')
            metadata: Additional metadata

        Returns:
            version_id: Unique version identifier
        """
        logger.info(f"Writing raw data for {symbol}, {len(df)} records")

        # Generate version ID
        version_id = self._generate_version_id(df, symbol, 'raw')

        # Partition path: raw/symbol/year=YYYY/month=MM/
        if df.index.name != 'timestamp' and isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index.name = 'timestamp'

        # Add partition columns
        df_with_partitions = df.reset_index()
        df_with_partitions['year'] = df_with_partitions['timestamp'].dt.year
        df_with_partitions['month'] = df_with_partitions['timestamp'].dt.month
        df_with_partitions['symbol'] = symbol

        # Write partitioned parquet
        output_path = self.raw_path / symbol / version_id
        table = pa.Table.from_pandas(df_with_partitions)

        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['year', 'month'],
            compression='snappy',
            use_dictionary=True,
            write_statistics=True
        )

        # Write metadata
        version_meta = DataVersion(
            version_id=version_id,
            created_at=datetime.utcnow().isoformat(),
            data_type='raw',
            symbol=symbol,
            start_date=df.index.min().isoformat(),
            end_date=df.index.max().isoformat(),
            record_count=len(df),
            checksum=self._calculate_checksum(df),
            schema_version='1.0',
            metadata=metadata or {}
        )
        version_meta.metadata['data_source'] = data_source

        self._write_version_metadata(version_meta)

        logger.info(f"Written version {version_id} for {symbol}")
        return version_id

    def read_raw_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        version_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read raw data with optional date filtering

        Args:
            symbol: Ticker symbol
            start_date: Start date filter
            end_date: End date filter
            version_id: Specific version (None = latest)

        Returns:
            DataFrame with DatetimeIndex
        """
        if version_id is None:
            version_id = self._get_latest_version(symbol, 'raw')

        data_path = self.raw_path / symbol / version_id

        if not data_path.exists():
            raise FileNotFoundError(f"No data found for {symbol} version {version_id}")

        # Read with partition pruning
        filters = []
        if start_date:
            filters.append(('year', '>=', start_date.year))
        if end_date:
            filters.append(('year', '<=', end_date.year))

        df = pq.read_table(
            str(data_path),
            filters=filters if filters else None
        ).to_pandas()

        # Set index and filter by exact dates
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        # Drop partition columns
        df = df.drop(columns=['year', 'month', 'symbol'], errors='ignore')

        logger.info(f"Read {len(df)} records for {symbol} version {version_id}")
        return df

    def write_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_set_name: str,
        feature_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Write computed features with lineage tracking"""
        logger.info(f"Writing features '{feature_set_name}' for {symbol}")

        version_id = self._generate_version_id(df, symbol, f'features_{feature_set_name}')

        # Store with partitioning
        df_with_partitions = df.reset_index()
        df_with_partitions['year'] = df_with_partitions['timestamp'].dt.year
        df_with_partitions['month'] = df_with_partitions['timestamp'].dt.month

        output_path = self.features_path / symbol / feature_set_name / version_id
        table = pa.Table.from_pandas(df_with_partitions)

        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['year', 'month'],
            compression='snappy'
        )

        # Metadata with feature lineage
        meta = metadata or {}
        meta['feature_config'] = feature_config
        meta['feature_columns'] = list(df.columns)

        version_meta = DataVersion(
            version_id=version_id,
            created_at=datetime.utcnow().isoformat(),
            data_type=f'features_{feature_set_name}',
            symbol=symbol,
            start_date=df.index.min().isoformat(),
            end_date=df.index.max().isoformat(),
            record_count=len(df),
            checksum=self._calculate_checksum(df),
            schema_version='1.0',
            metadata=meta
        )

        self._write_version_metadata(version_meta)

        logger.info(f"Written feature version {version_id}")
        return version_id

    def read_features(
        self,
        symbol: str,
        feature_set_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        version_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Read feature data"""
        if version_id is None:
            version_id = self._get_latest_version(symbol, f'features_{feature_set_name}')

        data_path = self.features_path / symbol / feature_set_name / version_id

        filters = []
        if start_date:
            filters.append(('year', '>=', start_date.year))
        if end_date:
            filters.append(('year', '<=', end_date.year))

        df = pq.read_table(
            str(data_path),
            filters=filters if filters else None
        ).to_pandas()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        df = df.drop(columns=['year', 'month'], errors='ignore')

        return df

    def list_versions(self, symbol: str, data_type: str = 'raw') -> List[DataVersion]:
        """List all versions for a symbol and data type"""
        versions = []
        metadata_files = self.metadata_path.glob(f'{symbol}_{data_type}_*.json')

        for meta_file in metadata_files:
            with open(meta_file) as f:
                version_data = json.load(f)
                versions.append(DataVersion(**version_data))

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_version_metadata(self, version_id: str) -> Optional[DataVersion]:
        """Get metadata for a specific version"""
        meta_files = list(self.metadata_path.glob(f'*_{version_id}.json'))
        if not meta_files:
            return None

        with open(meta_files[0]) as f:
            return DataVersion(**json.load(f))

    def _generate_version_id(self, df: pd.DataFrame, symbol: str, data_type: str) -> str:
        """Generate unique version ID based on content hash and timestamp"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        content_hash = self._calculate_checksum(df)[:8]
        return f'{timestamp}_{content_hash}'

    def _calculate_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum of DataFrame for integrity verification"""
        # Use a sample for large DataFrames
        sample_df = df.head(10000) if len(df) > 10000 else df
        content = pd.util.hash_pandas_object(sample_df).values
        return hashlib.sha256(content.tobytes()).hexdigest()

    def _write_version_metadata(self, version: DataVersion):
        """Write version metadata to JSON"""
        meta_file = self.metadata_path / f'{version.symbol}_{version.data_type}_{version.version_id}.json'
        with open(meta_file, 'w') as f:
            json.dump(asdict(version), f, indent=2)

    def _get_latest_version(self, symbol: str, data_type: str) -> str:
        """Get the latest version ID for a symbol and data type"""
        versions = self.list_versions(symbol, data_type)
        if not versions:
            raise FileNotFoundError(f"No versions found for {symbol} {data_type}")
        return versions[0].version_id

    def validate_data_integrity(self, version_id: str) -> Tuple[bool, Optional[str]]:
        """Validate data integrity by recalculating checksum"""
        meta = self.get_version_metadata(version_id)
        if not meta:
            return False, "Version metadata not found"

        try:
            # Read the data
            if meta.data_type == 'raw':
                df = self.read_raw_data(meta.symbol, version_id=version_id)
            elif meta.data_type.startswith('features_'):
                feature_set = meta.data_type.replace('features_', '')
                df = self.read_features(meta.symbol, feature_set, version_id=version_id)
            else:
                return False, f"Unknown data type: {meta.data_type}"

            # Recalculate checksum
            current_checksum = self._calculate_checksum(df)

            if current_checksum == meta.checksum:
                return True, None
            else:
                return False, f"Checksum mismatch: expected {meta.checksum}, got {current_checksum}"

        except Exception as e:
            return False, f"Validation error: {str(e)}"
