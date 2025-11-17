"""
Data Validation Framework
Automated quality checks, schema validation, and anomaly detection
Inspired by Great Expectations
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from infrastructure.logger import quant_logger

logger = quant_logger.get_logger("data_validation")


class ValidationSeverity(Enum):
    """Validation check severity levels"""

    CRITICAL = "critical"  # Blocks pipeline
    ERROR = "error"  # Logged but may continue
    WARNING = "warning"  # Informational


@dataclass
class ValidationResult:
    """Result of a validation check"""

    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["severity"] = self.severity.value
        return result


class DataValidator:
    """
    Comprehensive data validation for market data
    Implements checks for completeness, accuracy, consistency, and timeliness
    """

    def __init__(self):
        self.checks: List[Callable] = []
        self.results: List[ValidationResult] = []

    def validate(
        self, df: pd.DataFrame, symbol: str, **context
    ) -> List[ValidationResult]:
        """
        Run all validation checks on the DataFrame

        Args:
            df: DataFrame to validate
            symbol: Ticker symbol
            **context: Additional context (data_source, expected_timeframe, etc.)

        Returns:
            List of validation results
        """
        self.results = []

        logger.info(f"Starting validation for {symbol}, {len(df)} records")

        # Core checks
        self._check_not_empty(df, symbol)
        self._check_index_sorted(df, symbol)
        self._check_no_duplicates(df, symbol)
        self._check_no_nulls_in_critical_columns(df, symbol)
        self._check_price_sanity(df, symbol)
        self._check_volume_sanity(df, symbol)
        self._check_no_gaps(df, symbol, context.get("expected_frequency"))
        self._check_timezone(df, symbol)
        self._check_ohlc_consistency(df, symbol)
        self._check_statistical_anomalies(df, symbol)

        # Log results
        critical_failures = sum(
            1
            for r in self.results
            if not r.passed and r.severity == ValidationSeverity.CRITICAL
        )
        errors = sum(
            1
            for r in self.results
            if not r.passed and r.severity == ValidationSeverity.ERROR
        )
        warnings = sum(
            1
            for r in self.results
            if not r.passed and r.severity == ValidationSeverity.WARNING
        )

        logger.info(
            f"Validation complete for {symbol}: "
            f"{critical_failures} critical, {errors} errors, {warnings} warnings"
        )

        return self.results

    def get_critical_failures(self) -> List[ValidationResult]:
        """Get all critical validation failures"""
        return [
            r
            for r in self.results
            if not r.passed and r.severity == ValidationSeverity.CRITICAL
        ]

    def has_critical_failures(self) -> bool:
        """Check if there are any critical failures"""
        return len(self.get_critical_failures()) > 0

    def _add_result(
        self,
        check_name: str,
        passed: bool,
        severity: ValidationSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Add a validation result"""
        result = ValidationResult(
            check_name=check_name,
            passed=passed,
            severity=severity,
            message=message,
            details=details or {},
            timestamp=datetime.utcnow().isoformat(),
        )
        self.results.append(result)

        if not passed:
            log_func = (
                logger.critical
                if severity == ValidationSeverity.CRITICAL
                else (
                    logger.error
                    if severity == ValidationSeverity.ERROR
                    else logger.warning
                )
            )
            log_func(f"Validation {check_name}: {message}")

    def _check_not_empty(self, df: pd.DataFrame, symbol: str):
        """Check that DataFrame is not empty"""
        passed = len(df) > 0
        self._add_result(
            "not_empty",
            passed,
            ValidationSeverity.CRITICAL,
            f"DataFrame has {len(df)} records" if passed else "DataFrame is empty",
            {"record_count": len(df)},
        )

    def _check_index_sorted(self, df: pd.DataFrame, symbol: str):
        """Check that index is sorted chronologically"""
        if not isinstance(df.index, pd.DatetimeIndex):
            self._add_result(
                "index_type",
                False,
                ValidationSeverity.CRITICAL,
                f"Index is not DatetimeIndex, got {type(df.index)}",
                {"index_type": str(type(df.index))},
            )
            return

        is_sorted = df.index.is_monotonic_increasing
        self._add_result(
            "index_sorted",
            is_sorted,
            ValidationSeverity.CRITICAL,
            (
                "Index is properly sorted"
                if is_sorted
                else "Index is not sorted chronologically"
            ),
            {"is_sorted": is_sorted},
        )

    def _check_no_duplicates(self, df: pd.DataFrame, symbol: str):
        """Check for duplicate timestamps"""
        duplicates = df.index.duplicated().sum()
        passed = duplicates == 0
        self._add_result(
            "no_duplicate_timestamps",
            passed,
            ValidationSeverity.ERROR,
            (
                f"No duplicate timestamps"
                if passed
                else f"Found {duplicates} duplicate timestamps"
            ),
            {"duplicate_count": int(duplicates)},
        )

    def _check_no_nulls_in_critical_columns(self, df: pd.DataFrame, symbol: str):
        """Check for null values in critical price columns"""
        critical_cols = ["Open", "High", "Low", "Close", "Volume"]
        existing_cols = [col for col in critical_cols if col in df.columns]

        for col in existing_cols:
            null_count = df[col].isnull().sum()
            passed = null_count == 0
            self._add_result(
                f"no_nulls_{col.lower()}",
                passed,
                ValidationSeverity.CRITICAL,
                (
                    f"No nulls in {col}"
                    if passed
                    else f"Found {null_count} nulls in {col}"
                ),
                {
                    "column": col,
                    "null_count": int(null_count),
                    "null_pct": float(null_count / len(df)),
                },
            )

    def _check_price_sanity(self, df: pd.DataFrame, symbol: str):
        """Check for unrealistic price values"""
        price_cols = ["Open", "High", "Low", "Close"]
        existing_cols = [col for col in price_cols if col in df.columns]

        for col in existing_cols:
            # Check for negative or zero prices
            invalid_prices = (df[col] <= 0).sum()
            passed = invalid_prices == 0
            self._add_result(
                f"positive_prices_{col.lower()}",
                passed,
                ValidationSeverity.CRITICAL,
                (
                    f"All {col} prices are positive"
                    if passed
                    else f"Found {invalid_prices} non-positive {col} prices"
                ),
                {"column": col, "invalid_count": int(invalid_prices)},
            )

            # Check for extreme price movements (>50% in one bar - likely data error)
            if col == "Close" and len(df) > 1:
                returns = df[col].pct_change().abs()
                extreme_moves = (returns > 0.5).sum()
                passed = extreme_moves == 0
                self._add_result(
                    f"no_extreme_moves_{col.lower()}",
                    passed,
                    ValidationSeverity.WARNING,
                    (
                        f"No extreme price movements"
                        if passed
                        else f"Found {extreme_moves} extreme movements (>50%)"
                    ),
                    {
                        "column": col,
                        "extreme_count": int(extreme_moves),
                        "max_move": float(returns.max()),
                    },
                )

    def _check_volume_sanity(self, df: pd.DataFrame, symbol: str):
        """Check for volume anomalies"""
        if "Volume" not in df.columns:
            return

        # Check for negative volumes
        negative_vol = (df["Volume"] < 0).sum()
        passed = negative_vol == 0
        self._add_result(
            "non_negative_volume",
            passed,
            ValidationSeverity.ERROR,
            (
                "All volumes are non-negative"
                if passed
                else f"Found {negative_vol} negative volumes"
            ),
            {"negative_count": int(negative_vol)},
        )

        # Check for suspiciously low volume (potential data issue)
        zero_vol = (df["Volume"] == 0).sum()
        zero_vol_pct = zero_vol / len(df)
        passed = zero_vol_pct < 0.05  # Less than 5% zero volume
        self._add_result(
            "sufficient_volume",
            passed,
            ValidationSeverity.WARNING,
            (
                f"Only {zero_vol_pct:.1%} zero volume bars"
                if passed
                else f"{zero_vol_pct:.1%} bars have zero volume"
            ),
            {
                "zero_volume_count": int(zero_vol),
                "zero_volume_pct": float(zero_vol_pct),
            },
        )

    def _check_no_gaps(
        self, df: pd.DataFrame, symbol: str, expected_frequency: Optional[str]
    ):
        """Check for unexpected time gaps"""
        if expected_frequency is None:
            return

        if len(df) < 2:
            return

        # Infer frequency
        try:
            freq_mapping = {
                "1min": pd.Timedelta(minutes=1),
                "5min": pd.Timedelta(minutes=5),
                "1h": pd.Timedelta(hours=1),
                "1d": pd.Timedelta(days=1),
            }
            expected_delta = freq_mapping.get(expected_frequency)

            if expected_delta is None:
                return

            # Check gaps (excluding weekends for daily data)
            time_diffs = df.index.to_series().diff()

            if expected_frequency == "1d":
                # Allow for weekends (up to 3 days gap)
                max_allowed_gap = pd.Timedelta(days=3)
            else:
                # Intraday: allow 2x expected frequency
                max_allowed_gap = expected_delta * 2

            gaps = time_diffs[time_diffs > max_allowed_gap]
            passed = len(gaps) == 0

            self._add_result(
                "no_unexpected_gaps",
                passed,
                ValidationSeverity.WARNING,
                (
                    f"No unexpected time gaps"
                    if passed
                    else f"Found {len(gaps)} unexpected gaps"
                ),
                {
                    "gap_count": len(gaps),
                    "expected_frequency": expected_frequency,
                    "largest_gap": str(gaps.max()) if len(gaps) > 0 else None,
                },
            )

        except Exception as e:
            logger.warning(f"Could not check gaps: {e}")

    def _check_timezone(self, df: pd.DataFrame, symbol: str):
        """Check that timezone is set"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return

        has_tz = df.index.tz is not None
        self._add_result(
            "has_timezone",
            has_tz,
            ValidationSeverity.WARNING,
            (
                f"Timezone is set to {df.index.tz}"
                if has_tz
                else "No timezone information"
            ),
            {"timezone": str(df.index.tz) if has_tz else None},
        )

    def _check_ohlc_consistency(self, df: pd.DataFrame, symbol: str):
        """Check OHLC bar consistency (High >= Low, etc.)"""
        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            return

        # High should be >= Low
        violations = (df["High"] < df["Low"]).sum()
        passed = violations == 0
        self._add_result(
            "high_gte_low",
            passed,
            ValidationSeverity.ERROR,
            (
                "High >= Low in all bars"
                if passed
                else f"Found {violations} bars where High < Low"
            ),
            {"violation_count": int(violations)},
        )

        # High should be >= Open and Close
        violations_open = (df["High"] < df["Open"]).sum()
        violations_close = (df["High"] < df["Close"]).sum()
        total_violations = violations_open + violations_close
        passed = total_violations == 0
        self._add_result(
            "high_gte_open_close",
            passed,
            ValidationSeverity.ERROR,
            "High >= Open/Close" if passed else f"Found {total_violations} violations",
            {
                "violations_vs_open": int(violations_open),
                "violations_vs_close": int(violations_close),
            },
        )

        # Low should be <= Open and Close
        violations_open = (df["Low"] > df["Open"]).sum()
        violations_close = (df["Low"] > df["Close"]).sum()
        total_violations = violations_open + violations_close
        passed = total_violations == 0
        self._add_result(
            "low_lte_open_close",
            passed,
            ValidationSeverity.ERROR,
            "Low <= Open/Close" if passed else f"Found {total_violations} violations",
            {
                "violations_vs_open": int(violations_open),
                "violations_vs_close": int(violations_close),
            },
        )

    def _check_statistical_anomalies(self, df: pd.DataFrame, symbol: str):
        """Check for statistical anomalies that might indicate data quality issues"""
        if "Close" not in df.columns or len(df) < 30:
            return

        # Calculate returns
        returns = df["Close"].pct_change().dropna()

        # Check for outliers (returns > 5 standard deviations)
        mean_return = returns.mean()
        std_return = returns.std()
        outliers = (np.abs(returns - mean_return) > 5 * std_return).sum()

        passed = outliers == 0
        self._add_result(
            "no_statistical_outliers",
            passed,
            ValidationSeverity.WARNING,
            (
                "No statistical outliers"
                if passed
                else f"Found {outliers} statistical outliers (>5σ)"
            ),
            {
                "outlier_count": int(outliers),
                "mean_return": float(mean_return),
                "std_return": float(std_return),
                "max_z_score": float(
                    np.abs((returns - mean_return) / std_return).max()
                ),
            },
        )

    def generate_validation_report(self, output_path: str):
        """Generate JSON validation report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "critical_failures": sum(
                1
                for r in self.results
                if not r.passed and r.severity == ValidationSeverity.CRITICAL
            ),
            "checks": [r.to_dict() for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report written to {output_path}")


class CorporateActionsHandler:
    """
    Handle corporate actions (splits, dividends) for historical data
    Essential for avoiding survivorship bias
    """

    @staticmethod
    def adjust_for_splits(df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
        """
        Adjust historical prices for stock splits

        Args:
            df: OHLCV DataFrame
            splits: Series of split ratios (e.g., 2.0 for 2:1 split) indexed by date

        Returns:
            Adjusted DataFrame
        """
        df = df.copy()
        price_cols = ["Open", "High", "Low", "Close"]

        for split_date, ratio in splits.items():
            mask = df.index < split_date
            for col in price_cols:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio

            if "Volume" in df.columns:
                df.loc[mask, "Volume"] = df.loc[mask, "Volume"] * ratio

        return df

    @staticmethod
    def adjust_for_dividends(df: pd.DataFrame, dividends: pd.Series) -> pd.DataFrame:
        """
        Adjust historical prices for dividends (total return calculation)

        Args:
            df: OHLCV DataFrame
            dividends: Series of dividend amounts indexed by ex-dividend date

        Returns:
            Adjusted DataFrame
        """
        df = df.copy()
        price_cols = ["Open", "High", "Low", "Close"]

        for div_date, div_amount in dividends.items():
            if div_date not in df.index:
                continue

            close_price = df.loc[div_date, "Close"]
            adjustment_factor = 1 - (div_amount / close_price)

            mask = df.index < div_date
            for col in price_cols:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] * adjustment_factor

        return df
