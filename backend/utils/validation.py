"""
🚀 AUTO-ANALYST PLATFORM - INPUT VALIDATION UTILITIES
==================================================

Production-ready validation utilities with focus on performance, maintainability,
and type safety. Provides comprehensive data validation for ML pipelines.

Key Features:
- Modular architecture with clear separation of concerns
- High-performance validation with chunked processing
- Type-safe operations with comprehensive error handling
- Async support for I/O heavy operations
- Memory-efficient processing for large datasets
- Comprehensive logging and monitoring integration

Components:
- BaseValidator: Abstract validation interface
- DataValidator: Core data validation logic
- SchemaValidator: Schema compliance checking
- TypeValidator: Data type validation and conversion
- QualityValidator: Data quality assessment

Dependencies:
- pandas>=2.0.0: Data manipulation
- numpy>=1.24.0: Numerical operations
- pydantic>=2.0.0: Data validation
- typing-extensions: Enhanced type hints
"""

import asyncio
import logging
import re
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Set,
    Protocol, TypeVar, Generic, Callable, AsyncGenerator
)

# Core dependencies
import numpy as np
import pandas as pd

# Validation dependencies
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing_extensions import Annotated

# Optional dependencies
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress common warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)
ValidationResultType = TypeVar('ValidationResultType')


# =============================================================================
# CORE ENUMS & CONSTANTS
# =============================================================================

class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(str, Enum):
    """Supported data types for validation."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


class IssueType(str, Enum):
    """Types of validation issues."""
    MISSING_COLUMN = "missing_column"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    OUTLIERS = "outliers"
    CONSTRAINT_VIOLATION = "constraint_violation"


# =============================================================================
# CONFIGURATION & MODELS
# =============================================================================

class ValidationConfig(BaseModel):
    """Configuration for validation behavior."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True
    )

    # General settings
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    max_errors: int = Field(default=100, ge=1, le=1000)
    chunk_size: int = Field(default=10000, ge=1000, le=100000)
    timeout_seconds: int = Field(default=300, ge=10, le=3600)

    # Quality thresholds
    missing_value_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    outlier_threshold: float = Field(default=3.0, ge=1.0, le=5.0)

    # Performance settings
    enable_parallel: bool = True
    sample_large_datasets: bool = True
    max_sample_size: int = Field(default=100000, ge=1000)

    @field_validator('max_errors')
    @classmethod
    def validate_max_errors(cls, v: int) -> int:
        """Ensure max_errors is reasonable."""
        if v < 1:
            raise ValueError("max_errors must be positive")
        return min(v, 1000)  # Cap at 1000 for performance


@dataclass
class ValidationIssue:
    """Individual validation issue with metadata."""

    issue_type: IssueType
    severity: ValidationStatus
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    details: Dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationResult:
    """Comprehensive validation result."""

    # Status and timing
    status: ValidationStatus = ValidationStatus.PASSED
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Dataset metrics
    dataset_shape: Tuple[int, int] = (0, 0)
    memory_usage_mb: float = 0.0

    # Issues and quality
    issues: List[ValidationIssue] = field(default_factory=list)
    quality_score: float = 1.0

    # Summary statistics
    missing_ratio: float = 0.0
    duplicate_ratio: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len([i for i in self.issues if i.severity == ValidationStatus.ERROR])

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len([i for i in self.issues if i.severity == ValidationStatus.WARNING])


# =============================================================================
# VALIDATION PROTOCOLS & INTERFACES
# =============================================================================

class ValidatorProtocol(Protocol):
    """Protocol for all validators."""

    async def validate(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> ValidationResult:
        """Validate the DataFrame."""
        ...


class BaseValidator(ABC):
    """Abstract base validator with common functionality."""

    def __init__(self, config: ValidationConfig):
        """Initialize with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def validate(self, df: pd.DataFrame, **kwargs) -> ValidationResult:
        """Abstract validation method."""
        pass

    def _add_issue(
            self,
            issues: List[ValidationIssue],
            issue_type: IssueType,
            severity: ValidationStatus,
            message: str,
            **kwargs
    ) -> None:
        """Helper to add validation issues."""
        if len(issues) < self.config.max_errors:
            issues.append(ValidationIssue(
                issue_type=issue_type,
                severity=severity,
                message=message,
                **kwargs
            ))

    def _calculate_memory_usage(self, df: pd.DataFrame) -> float:
        """Calculate DataFrame memory usage in MB."""
        try:
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        except Exception:
            return 0.0

    async def _process_chunks(
            self,
            df: pd.DataFrame,
            processor: Callable[[pd.DataFrame], Any]
    ) -> List[Any]:
        """Process DataFrame in chunks for memory efficiency."""
        results = []
        chunk_size = self.config.chunk_size

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]

            # Allow other coroutines to run
            await asyncio.sleep(0)

            result = await asyncio.get_event_loop().run_in_executor(
                None, processor, chunk
            )
            results.append(result)

        return results


# =============================================================================
# CORE VALIDATORS
# =============================================================================

class SchemaValidator(BaseValidator):
    """Validates DataFrame schema compliance."""

    async def validate(
            self,
            df: pd.DataFrame,
            required_columns: Optional[List[str]] = None,
            expected_types: Optional[Dict[str, DataType]] = None,
            **kwargs
    ) -> ValidationResult:
        """Validate DataFrame schema."""
        start_time = time.time()
        result = ValidationResult(dataset_shape=df.shape)
        issues = []

        try:
            # Validate required columns
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                for col in missing_cols:
                    self._add_issue(
                        issues, IssueType.MISSING_COLUMN, ValidationStatus.ERROR,
                        f"Required column '{col}' is missing",
                        column=col,
                        suggestion=f"Add column '{col}' to your dataset"
                    )

            # Validate column types
            if expected_types:
                for col, expected_type in expected_types.items():
                    if col in df.columns:
                        type_issues = await self._validate_column_type(
                            df[col], col, expected_type
                        )
                        issues.extend(type_issues)

            # Determine overall status
            if any(i.severity == ValidationStatus.ERROR for i in issues):
                result.status = ValidationStatus.ERROR
            elif issues:
                result.status = ValidationStatus.WARNING

            result.issues = issues
            result.duration_seconds = time.time() - start_time
            result.memory_usage_mb = self._calculate_memory_usage(df)

            return result

        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
            result.status = ValidationStatus.CRITICAL
            result.issues = [ValidationIssue(
                issue_type=IssueType.CONSTRAINT_VIOLATION,
                severity=ValidationStatus.CRITICAL,
                message=f"Schema validation error: {str(e)}"
            )]
            return result

    async def _validate_column_type(
            self,
            series: pd.Series,
            column: str,
            expected_type: DataType
    ) -> List[ValidationIssue]:
        """Validate a single column's type."""
        issues = []

        try:
            # Skip empty series
            if series.empty or series.isna().all():
                return issues

            # Type-specific validation
            type_checkers = {
                DataType.INTEGER: self._is_integer_series,
                DataType.FLOAT: self._is_numeric_series,
                DataType.STRING: self._is_string_series,
                DataType.BOOLEAN: self._is_boolean_series,
                DataType.DATETIME: self._is_datetime_series,
                DataType.CATEGORICAL: self._is_categorical_series,
            }

            checker = type_checkers.get(expected_type)
            if checker:
                is_valid, invalid_count = await checker(series)

                if not is_valid and invalid_count > 0:
                    severity = (ValidationStatus.ERROR
                                if self.config.validation_level == ValidationLevel.STRICT
                                else ValidationStatus.WARNING)

                    issues.append(ValidationIssue(
                        issue_type=IssueType.TYPE_MISMATCH,
                        severity=severity,
                        message=f"Column '{column}' has {invalid_count} values incompatible with {expected_type.value}",
                        column=column,
                        details={
                            'expected_type': expected_type.value,
                            'invalid_count': invalid_count,
                            'total_count': len(series)
                        },
                        suggestion=f"Convert column '{column}' to {expected_type.value} type"
                    ))

        except Exception as e:
            self.logger.warning(f"Type validation failed for column {column}: {e}")

        return issues

    # Type checking methods
    async def _is_integer_series(self, series: pd.Series) -> Tuple[bool, int]:
        """Check if series contains integer values."""
        try:
            non_null = series.dropna()
            if non_null.empty:
                return True, 0

            # Try converting to numeric
            numeric = pd.to_numeric(non_null, errors='coerce')
            valid_numeric = numeric.notna()

            # Check if integers
            is_int = (numeric == numeric.round()).fillna(False)
            invalid_count = (~(valid_numeric & is_int)).sum()

            return invalid_count == 0, int(invalid_count)
        except Exception:
            return False, len(series)

    async def _is_numeric_series(self, series: pd.Series) -> Tuple[bool, int]:
        """Check if series contains numeric values."""
        try:
            non_null = series.dropna()
            if non_null.empty:
                return True, 0

            numeric = pd.to_numeric(non_null, errors='coerce')
            invalid_count = numeric.isna().sum()

            return invalid_count == 0, int(invalid_count)
        except Exception:
            return False, len(series)

    async def _is_string_series(self, series: pd.Series) -> Tuple[bool, int]:
        """Check if series can be treated as strings."""
        return True, 0  # Most data can be converted to strings

    async def _is_boolean_series(self, series: pd.Series) -> Tuple[bool, int]:
        """Check if series contains boolean-like values."""
        try:
            non_null = series.dropna()
            if non_null.empty:
                return True, 0

            str_series = non_null.astype(str).str.lower().str.strip()
            bool_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f'}

            is_bool = str_series.isin(bool_values)
            invalid_count = (~is_bool).sum()

            return invalid_count == 0, int(invalid_count)
        except Exception:
            return False, len(series)

    async def _is_datetime_series(self, series: pd.Series) -> Tuple[bool, int]:
        """Check if series contains datetime values."""
        try:
            non_null = series.dropna()
            if non_null.empty:
                return True, 0

            # Try pandas datetime conversion
            dt_series = pd.to_datetime(non_null, errors='coerce')
            invalid_count = dt_series.isna().sum()

            return invalid_count == 0, int(invalid_count)
        except Exception:
            return False, len(series)

    async def _is_categorical_series(self, series: pd.Series) -> Tuple[bool, int]:
        """Check if series is suitable for categorical type."""
        # Most series can be categorical, but warn if too many unique values
        try:
            unique_ratio = series.nunique() / len(series.dropna())
            return unique_ratio < 0.8, 0  # Warn if >80% unique values
        except Exception:
            return True, 0


class QualityValidator(BaseValidator):
    """Validates data quality metrics."""

    async def validate(self, df: pd.DataFrame, **kwargs) -> ValidationResult:
        """Validate data quality."""
        start_time = time.time()
        result = ValidationResult(dataset_shape=df.shape)
        issues = []
        quality_components = []

        try:
            # Missing values assessment
            missing_score, missing_issues = await self._assess_missing_values(df)
            issues.extend(missing_issues)
            quality_components.append(missing_score)

            # Duplicate rows assessment
            duplicate_score, duplicate_issues = await self._assess_duplicates(df)
            issues.extend(duplicate_issues)
            quality_components.append(duplicate_score)

            # Outliers assessment (if scipy available)
            if SCIPY_AVAILABLE:
                outlier_score, outlier_issues = await self._assess_outliers(df)
                issues.extend(outlier_issues)
                quality_components.append(outlier_score)

            # Calculate overall quality score
            result.quality_score = np.mean(quality_components) if quality_components else 1.0

            # Set summary metrics
            result.missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            result.duplicate_ratio = df.duplicated().sum() / len(df)

            # Determine status
            if result.quality_score < 0.5:
                result.status = ValidationStatus.ERROR
            elif result.quality_score < 0.8:
                result.status = ValidationStatus.WARNING
            else:
                result.status = ValidationStatus.PASSED

            result.issues = issues
            result.duration_seconds = time.time() - start_time
            result.memory_usage_mb = self._calculate_memory_usage(df)

            return result

        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
            result.status = ValidationStatus.CRITICAL
            result.quality_score = 0.0
            return result

    async def _assess_missing_values(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """Assess missing values in dataset."""
        issues = []

        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0

            # Check individual columns
            column_missing = df.isnull().sum() / len(df)
            problematic_cols = column_missing[column_missing > self.config.missing_value_threshold]

            for col in problematic_cols.index:
                severity = (ValidationStatus.ERROR
                            if column_missing[col] > 0.8
                            else ValidationStatus.WARNING)

                issues.append(ValidationIssue(
                    issue_type=IssueType.MISSING_VALUES,
                    severity=severity,
                    message=f"Column '{col}' has {column_missing[col]:.1%} missing values",
                    column=col,
                    details={'missing_ratio': column_missing[col]},
                    suggestion="Consider imputation or data collection"
                ))

            # Calculate score (penalize missing values)
            score = max(0, 1 - (missing_ratio * 2))
            return score, issues

        except Exception as e:
            self.logger.warning(f"Missing values assessment failed: {e}")
            return 0.5, issues

    async def _assess_duplicates(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """Assess duplicate rows."""
        issues = []

        try:
            total_rows = len(df)
            duplicate_rows = df.duplicated().sum()
            duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0

            if duplicate_ratio > self.config.duplicate_threshold:
                severity = (ValidationStatus.ERROR
                            if duplicate_ratio > 0.3
                            else ValidationStatus.WARNING)

                issues.append(ValidationIssue(
                    issue_type=IssueType.DUPLICATE_ROWS,
                    severity=severity,
                    message=f"Dataset has {duplicate_rows} duplicate rows ({duplicate_ratio:.1%})",
                    details={
                        'duplicate_count': duplicate_rows,
                        'duplicate_ratio': duplicate_ratio
                    },
                    suggestion="Remove duplicate rows to improve quality"
                ))

            # Calculate score
            score = max(0, 1 - (duplicate_ratio * 2))
            return score, issues

        except Exception as e:
            self.logger.warning(f"Duplicates assessment failed: {e}")
            return 0.5, issues

    async def _assess_outliers(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """Assess outliers in numeric columns."""
        issues = []
        outlier_ratios = []

        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) < 10:  # Skip small columns
                        continue

                    # Z-score based outlier detection
                    z_scores = np.abs(stats.zscore(col_data))
                    outliers = z_scores > self.config.outlier_threshold
                    outlier_ratio = outliers.sum() / len(col_data)

                    outlier_ratios.append(outlier_ratio)

                    if outlier_ratio > 0.05:  # More than 5% outliers
                        issues.append(ValidationIssue(
                            issue_type=IssueType.OUTLIERS,
                            severity=ValidationStatus.WARNING,
                            message=f"Column '{col}' has {outlier_ratio:.1%} potential outliers",
                            column=col,
                            details={'outlier_ratio': outlier_ratio},
                            suggestion="Review outliers for data quality"
                        ))

                except Exception as col_error:
                    self.logger.debug(f"Outlier detection failed for {col}: {col_error}")

            # Calculate average outlier score
            avg_outlier_ratio = np.mean(outlier_ratios) if outlier_ratios else 0
            score = max(0, 1 - (avg_outlier_ratio * 3))

            return score, issues

        except Exception as e:
            self.logger.warning(f"Outlier assessment failed: {e}")
            return 0.5, issues


# =============================================================================
# MAIN DATA VALIDATOR
# =============================================================================

class DataValidator:
    """
    Main data validator orchestrating all validation components.

    Provides high-level interface for comprehensive data validation
    with performance optimization and error handling.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize with configuration."""
        self.config = config or ValidationConfig()
        self.schema_validator = SchemaValidator(self.config)
        self.quality_validator = QualityValidator(self.config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def validate_dataset(
            self,
            df: pd.DataFrame,
            required_columns: Optional[List[str]] = None,
            expected_types: Optional[Dict[str, DataType]] = None,
            validate_quality: bool = True
    ) -> ValidationResult:
        """
        Comprehensive dataset validation.

        Args:
            df: DataFrame to validate
            required_columns: Required column names
            expected_types: Expected column types
            validate_quality: Whether to perform quality validation

        Returns:
            Comprehensive validation result
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting validation for dataset: {df.shape}")

            # Sample large datasets for performance
            validation_df = await self._prepare_dataset(df)

            # Run validations concurrently
            tasks = []

            # Schema validation
            if required_columns or expected_types:
                tasks.append(self.schema_validator.validate(
                    validation_df,
                    required_columns=required_columns,
                    expected_types=expected_types
                ))

            # Quality validation
            if validate_quality:
                tasks.append(self.quality_validator.validate(validation_df))

            # Wait for all validations to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []

            # Combine results
            combined_result = self._combine_results(df, results)
            combined_result.duration_seconds = time.time() - start_time

            self.logger.info(
                f"Validation completed: {combined_result.status.value} "
                f"({len(combined_result.issues)} issues, {combined_result.duration_seconds:.2f}s)"
            )

            return combined_result

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")

            # Return error result
            error_result = ValidationResult(
                status=ValidationStatus.CRITICAL,
                dataset_shape=df.shape,
                duration_seconds=time.time() - start_time,
                issues=[ValidationIssue(
                    issue_type=IssueType.CONSTRAINT_VIOLATION,
                    severity=ValidationStatus.CRITICAL,
                    message=f"Validation failed: {str(e)}"
                )]
            )
            return error_result

    async def _prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset for validation (sampling, etc.)."""
        if (self.config.sample_large_datasets and
                len(df) > self.config.max_sample_size):

            sample_size = min(self.config.max_sample_size, len(df))
            self.logger.info(f"Sampling dataset: {len(df)} -> {sample_size} rows")

            # Stratified sampling if possible
            return df.sample(n=sample_size, random_state=42)

        return df

    def _combine_results(
            self,
            original_df: pd.DataFrame,
            results: List[Union[ValidationResult, Exception]]
    ) -> ValidationResult:
        """Combine multiple validation results."""
        combined = ValidationResult(
            dataset_shape=original_df.shape,
            memory_usage_mb=self._calculate_memory_usage(original_df)
        )

        all_issues = []
        quality_scores = []
        statuses = []

        for result in results:
            if isinstance(result, Exception):
                # Handle validation exceptions
                all_issues.append(ValidationIssue(
                    issue_type=IssueType.CONSTRAINT_VIOLATION,
                    severity=ValidationStatus.ERROR,
                    message=f"Validation error: {str(result)}"
                ))
                statuses.append(ValidationStatus.ERROR)
            elif isinstance(result, ValidationResult):
                all_issues.extend(result.issues)
                quality_scores.append(result.quality_score)
                statuses.append(result.status)

                # Update metrics
                if result.missing_ratio > 0:
                    combined.missing_ratio = result.missing_ratio
                if result.duplicate_ratio > 0:
                    combined.duplicate_ratio = result.duplicate_ratio

        # Set combined metrics
        combined.issues = all_issues
        combined.quality_score = np.mean(quality_scores) if quality_scores else 1.0

        # Determine overall status
        if ValidationStatus.CRITICAL in statuses:
            combined.status = ValidationStatus.CRITICAL
        elif ValidationStatus.ERROR in statuses:
            combined.status = ValidationStatus.ERROR
        elif ValidationStatus.WARNING in statuses:
            combined.status = ValidationStatus.WARNING
        else:
            combined.status = ValidationStatus.PASSED

        return combined

    def _calculate_memory_usage(self, df: pd.DataFrame) -> float:
        """Calculate DataFrame memory usage in MB."""
        try:
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        except Exception:
            return 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def validate_dataset(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        expected_types: Optional[Dict[str, DataType]] = None,
        config: Optional[ValidationConfig] = None
) -> ValidationResult:
    """
    Convenience function for dataset validation.

    Args:
        df: DataFrame to validate
        required_columns: Required column names
        expected_types: Expected column types
        config: Validation configuration

    Returns:
        Validation result
    """
    validator = DataValidator(config)
    return await validator.validate_dataset(
        df, required_columns, expected_types
    )


def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Quick missing values check.

    Args:
        df: DataFrame to check
        threshold: Threshold for concerning missing values

    Returns:
        Missing value analysis
    """
    try:
        missing_counts = df.isnull().sum()
        missing_ratios = missing_counts / len(df)

        total_missing = missing_counts.sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_ratio = total_missing / total_cells if total_cells > 0 else 0

        concerning_columns = missing_ratios[missing_ratios > threshold]

        return {
            'total_missing_cells': int(total_missing),
            'overall_missing_ratio': float(overall_ratio),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'concerning_columns': concerning_columns.to_dict(),
            'is_concerning': len(concerning_columns) > 0 or overall_ratio > threshold
        }
    except Exception as e:
        logger.error(f"Missing values check failed: {e}")
        return {'error': str(e)}


def check_duplicate_records(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick duplicate records check.

    Args:
        df: DataFrame to check

    Returns:
        Duplicate analysis
    """
    try:
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0

        return {
            'total_rows': total_rows,
            'duplicate_rows': int(duplicate_rows),
            'duplicate_ratio': float(duplicate_ratio),
            'unique_rows': total_rows - duplicate_rows,
            'has_duplicates': duplicate_rows > 0
        }
    except Exception as e:
        logger.error(f"Duplicate check failed: {e}")
        return {'error': str(e)}


def validate_file_format(file_path: Union[str, Path], expected_format: str) -> bool:
    """
    Validate file format matches expectation.

    Args:
        file_path: Path to file
        expected_format: Expected format ('csv', 'excel', 'json')

    Returns:
        True if format matches
    """
    try:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        format_mapping = {
            'csv': ['.csv'],
            'excel': ['.xlsx', '.xls'],
            'json': ['.json'],
            'parquet': ['.parquet']
        }

        expected_extensions = format_mapping.get(expected_format.lower(), [])
        return extension in expected_extensions
    except Exception as e:
        logger.error(f"File format validation failed: {e}")
        return False


def sanitize_input(
        value: Any,
        expected_type: Optional[type] = None,
        max_length: Optional[int] = None
) -> Any:
    """
    Sanitize input value.

    Args:
        value: Input value to sanitize
        expected_type: Expected type
        max_length: Maximum length for strings

    Returns:
        Sanitized value
    """
    try:
        if value is None:
            return None

        # Type conversion
        if expected_type:
            if expected_type == str:
                value = str(value)
            elif expected_type == int:
                value = int(float(str(value)))
            elif expected_type == float:
                value = float(str(value))
            elif expected_type == bool:
                if isinstance(value, str):
                    value = value.lower() in ('true', 'yes', '1', 'on')
                else:
                    value = bool(value)

        # String sanitization
        if isinstance(value, str):
            # Remove potentially harmful characters
            value = re.sub(r'[<>"\']', '', value)

            # Limit length
            if max_length and len(value) > max_length:
                value = value[:max_length]

            value = value.strip()

        return value
    except Exception as e:
        logger.warning(f"Input sanitization failed: {e}")
        return str(value) if value is not None else None


def is_valid_dataset(df: pd.DataFrame, min_rows: int = 10, min_cols: int = 2) -> bool:
    """
    Quick dataset validity check.

    Args:
        df: DataFrame to check
        min_rows: Minimum required rows
        min_cols: Minimum required columns

    Returns:
        True if dataset is valid
    """
    try:
        if df.empty:
            return False

        if len(df) < min_rows:
            return False

        if len(df.columns) < min_cols:
            return False

        # Check if all values are null
        if df.isnull().all().all():
            return False

        return True
    except Exception as e:
        logger.error(f"Dataset validity check failed: {e}")
        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'DataValidator', 'SchemaValidator', 'QualityValidator', 'BaseValidator',

    # Configuration and models
    'ValidationConfig', 'ValidationResult', 'ValidationIssue',

    # Enums
    'ValidationLevel', 'ValidationStatus', 'DataType', 'IssueType',

    # Functions
    'validate_dataset', 'check_missing_values', 'check_duplicate_records',
    'validate_file_format', 'sanitize_input', 'is_valid_dataset',

    # Protocols
    'ValidatorProtocol'
]

# Initialize logging
logger.info("Validation utilities loaded successfully")
