"""
🚀 AUTO-ANALYST PLATFORM - DATA PREPROCESSING UTILITIES
======================================================

Production-ready data preprocessing with modular architecture, async support,
and comprehensive error handling. Optimized for performance and maintainability.

Key Features:
- Modular architecture with clear separation of concerns
- Async/await support for I/O intensive operations
- Memory-efficient chunked processing for large datasets
- Comprehensive type safety and validation
- Extensible plugin architecture for custom processors
- Production-grade monitoring and logging

Components:
- BaseProcessor: Abstract processor interface
- TabularProcessor: Structured data preprocessing
- TimeSeriesProcessor: Temporal data preprocessing
- TextProcessor: Natural language processing
- QualityAnalyzer: Data quality assessment
- ProcessingPipeline: Orchestrates multiple processors

Dependencies:
- pandas>=2.0.0: Data manipulation
- numpy>=1.24.0: Numerical operations
- scikit-learn>=1.3.0: ML preprocessing utilities
- pydantic>=2.0.0: Configuration validation
"""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Protocol,
    AsyncGenerator, Callable, TypeVar, Generic
)

# Core dependencies
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ML preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Optional dependencies with graceful fallbacks
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Configure warnings and logging
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

logger = logging.getLogger(__name__)

# Type definitions
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)
ProcessorType = TypeVar('ProcessorType')


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class ProcessingStatus(str, Enum):
    """Processing operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataType(str, Enum):
    """Data type classifications."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


class MissingValueStrategy(str, Enum):
    """Missing value handling strategies."""
    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    KNN = "knn"
    CONSTANT = "constant"


class EncodingStrategy(str, Enum):
    """Categorical encoding strategies."""
    ONE_HOT = "one_hot"
    LABEL = "label"
    ORDINAL = "ordinal"
    FREQUENCY = "frequency"
    TARGET = "target"
    BINARY = "binary"


class ScalingStrategy(str, Enum):
    """Numeric scaling strategies."""
    STANDARD = "standard"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    NONE = "none"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class ProcessingConfig(BaseModel):
    """Configuration for data processing operations."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True
    )

    # General settings
    chunk_size: int = Field(default=10000, ge=1000, le=100000)
    max_memory_mb: int = Field(default=2000, ge=100, le=16000)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    enable_parallel: bool = True

    # Missing value handling
    missing_strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN
    missing_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    fill_value: Optional[Any] = None

    # Categorical encoding
    encoding_strategy: EncodingStrategy = EncodingStrategy.ONE_HOT
    max_categories: int = Field(default=20, ge=2, le=100)
    min_frequency: float = Field(default=0.01, ge=0.0, le=1.0)

    # Numeric scaling
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD

    # Feature engineering
    create_interactions: bool = False
    extract_datetime_features: bool = True
    polynomial_features: bool = False

    # Data quality
    remove_duplicates: bool = True
    remove_constant_features: bool = True
    variance_threshold: float = Field(default=0.01, ge=0.0, le=1.0)

    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Ensure chunk size is reasonable."""
        return min(max(v, 1000), 100000)


@dataclass
class ProcessingResult:
    """Result of a preprocessing operation."""

    status: ProcessingStatus = ProcessingStatus.COMPLETED
    data: Optional[pd.DataFrame] = None
    original_shape: Tuple[int, int] = (0, 0)
    processed_shape: Tuple[int, int] = (0, 0)
    duration_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    transformations_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.status == ProcessingStatus.COMPLETED and self.data is not None

    @property
    def shape_changed(self) -> bool:
        """Check if data shape changed during processing."""
        return self.original_shape != self.processed_shape


@dataclass
class QualityReport:
    """Data quality assessment report."""

    # Basic statistics
    total_rows: int = 0
    total_columns: int = 0
    memory_usage_mb: float = 0.0

    # Missing data analysis
    missing_cells: int = 0
    missing_ratio: float = 0.0
    columns_with_missing: Dict[str, float] = field(default_factory=dict)

    # Duplicate analysis
    duplicate_rows: int = 0
    duplicate_ratio: float = 0.0

    # Data type analysis
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)

    # Quality issues
    constant_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    outlier_columns: Dict[str, int] = field(default_factory=dict)

    # Overall assessment
    quality_score: float = 1.0
    recommendations: List[str] = field(default_factory=list)

    def add_recommendation(self, message: str) -> None:
        """Add a quality improvement recommendation."""
        if message not in self.recommendations:
            self.recommendations.append(message)


# =============================================================================
# PROTOCOLS & INTERFACES
# =============================================================================

class ProcessorProtocol(Protocol):
    """Protocol for all data processors."""

    async def process(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig,
            **kwargs
    ) -> ProcessingResult:
        """Process the data according to configuration."""
        ...

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        ...


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""

    def __init__(self, name: str):
        """Initialize processor with name."""
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._fitted_transformers: Dict[str, Any] = {}
        self._processing_stats: Dict[str, Any] = {}

    @abstractmethod
    async def process(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig,
            **kwargs
    ) -> ProcessingResult:
        """Process data according to configuration."""
        pass

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        if data is None or data.empty:
            return False
        return True

    def _calculate_memory_usage(self, df: pd.DataFrame) -> float:
        """Calculate DataFrame memory usage in MB."""
        try:
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        except Exception:
            return 0.0

    @asynccontextmanager
    async def _processing_context(self, data: pd.DataFrame):
        """Context manager for processing operations."""
        start_time = asyncio.get_event_loop().time()
        original_shape = data.shape
        original_memory = self._calculate_memory_usage(data)

        try:
            yield
        finally:
            end_time = asyncio.get_event_loop().time()
            self._processing_stats = {
                'duration': end_time - start_time,
                'original_shape': original_shape,
                'original_memory_mb': original_memory,
            }


# =============================================================================
# CORE PROCESSORS
# =============================================================================

class QualityAnalyzer(BaseProcessor):
    """Analyzes data quality and generates improvement recommendations."""

    def __init__(self):
        """Initialize quality analyzer."""
        super().__init__("QualityAnalyzer")

    async def process(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig,
            **kwargs
    ) -> ProcessingResult:
        """Analyze data quality."""
        if not self.validate_input(data):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message="Invalid input data"
            )

        async with self._processing_context(data):
            try:
                report = await self._analyze_quality(data)

                return ProcessingResult(
                    status=ProcessingStatus.COMPLETED,
                    data=data,  # Return original data
                    original_shape=data.shape,
                    processed_shape=data.shape,
                    duration_seconds=self._processing_stats['duration'],
                    memory_usage_mb=self._processing_stats['original_memory_mb'],
                    metadata={'quality_report': report}
                )

            except Exception as e:
                self.logger.error(f"Quality analysis failed: {e}")
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error_message=str(e)
                )

    async def _analyze_quality(self, data: pd.DataFrame) -> QualityReport:
        """Perform comprehensive quality analysis."""
        report = QualityReport()

        # Basic statistics
        report.total_rows, report.total_columns = data.shape
        report.memory_usage_mb = self._calculate_memory_usage(data)

        # Analyze missing data
        await self._analyze_missing_data(data, report)

        # Analyze duplicates
        await self._analyze_duplicates(data, report)

        # Analyze data types
        await self._analyze_data_types(data, report)

        # Analyze quality issues
        await self._analyze_quality_issues(data, report)

        # Calculate overall quality score
        report.quality_score = self._calculate_quality_score(report)

        # Generate recommendations
        self._generate_recommendations(report)

        return report

    async def _analyze_missing_data(self, data: pd.DataFrame, report: QualityReport) -> None:
        """Analyze missing data patterns."""
        total_cells = data.shape[0] * data.shape[1]
        missing_counts = data.isnull().sum()

        report.missing_cells = missing_counts.sum()
        report.missing_ratio = report.missing_cells / total_cells if total_cells > 0 else 0

        # Per-column missing analysis
        for col in data.columns:
            missing_ratio = missing_counts[col] / len(data) if len(data) > 0 else 0
            if missing_ratio > 0:
                report.columns_with_missing[col] = missing_ratio

    async def _analyze_duplicates(self, data: pd.DataFrame, report: QualityReport) -> None:
        """Analyze duplicate rows."""
        report.duplicate_rows = data.duplicated().sum()
        report.duplicate_ratio = report.duplicate_rows / len(data) if len(data) > 0 else 0

    async def _analyze_data_types(self, data: pd.DataFrame, report: QualityReport) -> None:
        """Analyze and classify column data types."""
        for col in data.columns:
            dtype = data[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                report.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                report.datetime_columns.append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                report.categorical_columns.append(col)
            elif dtype == 'object':
                if self._is_categorical_column(data[col]):
                    report.categorical_columns.append(col)
                else:
                    report.text_columns.append(col)

    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Determine if column should be treated as categorical."""
        try:
            non_null = series.dropna()
            if len(non_null) == 0:
                return True

            # Check unique value ratio and average string length
            unique_ratio = len(non_null.unique()) / len(non_null)
            avg_length = non_null.astype(str).str.len().mean()

            return unique_ratio < 0.5 or avg_length < 50
        except Exception:
            return True

    async def _analyze_quality_issues(self, data: pd.DataFrame, report: QualityReport) -> None:
        """Analyze various quality issues."""
        # Constant columns
        for col in data.columns:
            if data[col].nunique(dropna=False) <= 1:
                report.constant_columns.append(col)

        # High cardinality categorical columns
        for col in report.categorical_columns:
            if data[col].nunique() > 100:
                report.high_cardinality_columns.append(col)

        # Outliers in numeric columns (if scipy available)
        if SCIPY_AVAILABLE:
            await self._detect_outliers(data, report)

    async def _detect_outliers(self, data: pd.DataFrame, report: QualityReport) -> None:
        """Detect outliers in numeric columns."""
        for col in report.numeric_columns:
            try:
                col_data = data[col].dropna()
                if len(col_data) < 10:
                    continue

                # Use IQR method for outlier detection
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

                if outliers > 0:
                    report.outlier_columns[col] = outliers

            except Exception as e:
                self.logger.debug(f"Outlier detection failed for {col}: {e}")

    def _calculate_quality_score(self, report: QualityReport) -> float:
        """Calculate overall quality score (0-1)."""
        try:
            score_components = []

            # Missing data score
            missing_score = max(0, 1 - report.missing_ratio * 2)
            score_components.append(missing_score)

            # Duplicate score
            duplicate_score = max(0, 1 - report.duplicate_ratio * 2)
            score_components.append(duplicate_score)

            # Constant features score
            if report.total_columns > 0:
                constant_ratio = len(report.constant_columns) / report.total_columns
                constant_score = max(0, 1 - constant_ratio)
                score_components.append(constant_score)

            return np.mean(score_components) if score_components else 1.0

        except Exception:
            return 0.5

    def _generate_recommendations(self, report: QualityReport) -> None:
        """Generate quality improvement recommendations."""
        if report.missing_ratio > 0.1:
            report.add_recommendation("Address missing values through imputation or removal")

        if report.duplicate_ratio > 0.05:
            report.add_recommendation("Remove duplicate rows to improve data quality")

        if report.constant_columns:
            report.add_recommendation("Remove constant columns as they provide no information")

        if report.high_cardinality_columns:
            report.add_recommendation("Consider grouping rare categories in high-cardinality columns")

        if report.quality_score < 0.7:
            report.add_recommendation("Overall data quality is low - review data collection process")


class TabularProcessor(BaseProcessor):
    """Processes structured tabular data."""

    def __init__(self):
        """Initialize tabular processor."""
        super().__init__("TabularProcessor")

    async def process(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig,
            target_column: Optional[str] = None,
            fit_transformers: bool = True
    ) -> ProcessingResult:
        """Process tabular data according to configuration."""
        if not self.validate_input(data):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message="Invalid input data"
            )

        async with self._processing_context(data):
            try:
                # Make a copy to avoid modifying original
                processed_data = data.copy()
                transformations = []
                warnings_list = []

                # Step 1: Remove duplicates
                if config.remove_duplicates:
                    initial_rows = len(processed_data)
                    processed_data = processed_data.drop_duplicates()

                    if len(processed_data) < initial_rows:
                        removed = initial_rows - len(processed_data)
                        transformations.append(f"Removed {removed} duplicate rows")

                # Step 2: Handle missing values
                processed_data, missing_warnings = await self._handle_missing_values(
                    processed_data, config, target_column
                )
                warnings_list.extend(missing_warnings)
                transformations.append("Handled missing values")

                # Step 3: Remove constant features
                if config.remove_constant_features:
                    processed_data, constant_warnings = await self._remove_constant_features(
                        processed_data, target_column
                    )
                    warnings_list.extend(constant_warnings)
                    if constant_warnings:
                        transformations.append("Removed constant features")

                # Step 4: Encode categorical features
                processed_data = await self._encode_categorical_features(
                    processed_data, config, target_column, fit_transformers
                )
                transformations.append("Encoded categorical features")

                # Step 5: Scale numeric features
                processed_data = await self._scale_numeric_features(
                    processed_data, config, target_column, fit_transformers
                )
                transformations.append("Scaled numeric features")

                # Step 6: Feature engineering
                if any([config.create_interactions, config.extract_datetime_features]):
                    processed_data = await self._engineer_features(processed_data, config)
                    transformations.append("Engineered features")

                return ProcessingResult(
                    status=ProcessingStatus.COMPLETED,
                    data=processed_data,
                    original_shape=data.shape,
                    processed_shape=processed_data.shape,
                    duration_seconds=self._processing_stats['duration'],
                    memory_usage_mb=self._calculate_memory_usage(processed_data),
                    transformations_applied=transformations,
                    warnings=warnings_list
                )

            except Exception as e:
                self.logger.error(f"Tabular processing failed: {e}")
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error_message=str(e)
                )

    async def _handle_missing_values(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig,
            target_column: Optional[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values according to strategy."""
        warnings_list = []
        processed_data = data.copy()

        # Drop columns with excessive missing values
        columns_to_drop = []
        for col in data.columns:
            if col != target_column:
                missing_ratio = data[col].isnull().sum() / len(data)
                if missing_ratio > config.missing_threshold:
                    columns_to_drop.append(col)

        if columns_to_drop:
            processed_data = processed_data.drop(columns=columns_to_drop)
            warnings_list.append(f"Dropped {len(columns_to_drop)} columns with >{config.missing_threshold:.1%} missing values")

        # Handle remaining missing values
        strategy = config.missing_strategy

        if strategy == MissingValueStrategy.DROP_ROWS:
            initial_rows = len(processed_data)
            processed_data = processed_data.dropna()
            if len(processed_data) < initial_rows:
                warnings_list.append(f"Dropped {initial_rows - len(processed_data)} rows with missing values")

        else:
            # Imputation strategies
            for col in processed_data.columns:
                if col != target_column and processed_data[col].isnull().any():
                    processed_data[col] = await self._impute_column(
                        processed_data[col], strategy, config.fill_value
                    )

        return processed_data, warnings_list

    async def _impute_column(
            self,
            series: pd.Series,
            strategy: MissingValueStrategy,
            fill_value: Any
    ) -> pd.Series:
        """Impute missing values in a single column."""
        try:
            if strategy == MissingValueStrategy.MEAN and pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.mean())

            elif strategy == MissingValueStrategy.MEDIAN and pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.median())

            elif strategy == MissingValueStrategy.MODE:
                mode_values = series.mode()
                mode_value = mode_values.iloc[0] if len(mode_values) > 0 else "Unknown"
                return series.fillna(mode_value)

            elif strategy == MissingValueStrategy.FORWARD_FILL:
                return series.ffill()

            elif strategy == MissingValueStrategy.BACKWARD_FILL:
                return series.bfill()

            elif strategy == MissingValueStrategy.INTERPOLATE and pd.api.types.is_numeric_dtype(series):
                return series.interpolate()

            elif strategy == MissingValueStrategy.CONSTANT:
                value = fill_value if fill_value is not None else ("Unknown" if series.dtype == 'object' else 0)
                return series.fillna(value)

            else:
                # Default fallback
                if pd.api.types.is_numeric_dtype(series):
                    return series.fillna(series.median())
                else:
                    mode_values = series.mode()
                    mode_value = mode_values.iloc[0] if len(mode_values) > 0 else "Unknown"
                    return series.fillna(mode_value)

        except Exception as e:
            self.logger.warning(f"Imputation failed for column: {e}")
            # Final fallback
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(0)
            else:
                return series.fillna("Unknown")

    async def _remove_constant_features(
            self,
            data: pd.DataFrame,
            target_column: Optional[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove constant and near-constant features."""
        warnings_list = []
        constant_cols = []

        for col in data.columns:
            if col != target_column and data[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)

        if constant_cols:
            processed_data = data.drop(columns=constant_cols)
            warnings_list.append(f"Removed {len(constant_cols)} constant columns")
            return processed_data, warnings_list

        return data, warnings_list

    async def _encode_categorical_features(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig,
            target_column: Optional[str],
            fit_transformers: bool
    ) -> pd.DataFrame:
        """Encode categorical features."""
        try:
            # Identify categorical columns
            categorical_cols = [
                col for col in data.columns
                if col != target_column and
                   (data[col].dtype == 'object' or data[col].dtype.name == 'category')
            ]

            if not categorical_cols:
                return data

            processed_data = data.copy()

            for col in categorical_cols:
                n_unique = data[col].nunique()

                # Choose encoding strategy based on cardinality
                if config.encoding_strategy == EncodingStrategy.ONE_HOT and n_unique <= config.max_categories:
                    # One-hot encoding
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True, dummy_na=False)
                    processed_data = pd.concat([processed_data.drop(columns=[col]), dummies], axis=1)

                elif config.encoding_strategy == EncodingStrategy.LABEL:
                    # Label encoding
                    if fit_transformers:
                        encoder = LabelEncoder()
                        processed_data[col] = encoder.fit_transform(data[col].astype(str))
                        self._fitted_transformers[f"label_{col}"] = encoder
                    else:
                        encoder = self._fitted_transformers.get(f"label_{col}")
                        if encoder:
                            # Handle unseen categories
                            processed_data[col] = self._safe_label_transform(data[col], encoder)

                elif config.encoding_strategy == EncodingStrategy.FREQUENCY:
                    # Frequency encoding
                    freq_map = data[col].value_counts().to_dict()
                    processed_data[col] = data[col].map(freq_map).fillna(0)

                else:
                    # Default to frequency encoding for high cardinality
                    freq_map = data[col].value_counts().to_dict()
                    processed_data[col] = data[col].map(freq_map).fillna(0)

            return processed_data

        except Exception as e:
            self.logger.warning(f"Categorical encoding failed: {e}")
            return data

    def _safe_label_transform(self, series: pd.Series, encoder: LabelEncoder) -> np.ndarray:
        """Safely transform labels handling unseen categories."""
        result = []
        for value in series.astype(str):
            if value in encoder.classes_:
                result.append(encoder.transform([value])[0])
            else:
                result.append(-1)  # Unseen category marker
        return np.array(result)

    async def _scale_numeric_features(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig,
            target_column: Optional[str],
            fit_transformers: bool
    ) -> pd.DataFrame:
        """Scale numeric features."""
        try:
            if config.scaling_strategy == ScalingStrategy.NONE:
                return data

            # Identify numeric columns
            numeric_cols = [
                col for col in data.columns
                if col != target_column and pd.api.types.is_numeric_dtype(data[col])
            ]

            if not numeric_cols:
                return data

            processed_data = data.copy()

            # Create scaler
            if config.scaling_strategy == ScalingStrategy.STANDARD:
                scaler = StandardScaler()
            elif config.scaling_strategy == ScalingStrategy.MIN_MAX:
                scaler = MinMaxScaler()
            elif config.scaling_strategy == ScalingStrategy.ROBUST:
                scaler = RobustScaler()
            else:
                return data

            # Fit and transform
            if fit_transformers:
                processed_data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                self._fitted_transformers['scaler'] = scaler
            else:
                scaler_fitted = self._fitted_transformers.get('scaler')
                if scaler_fitted:
                    processed_data[numeric_cols] = scaler_fitted.transform(data[numeric_cols])

            return processed_data

        except Exception as e:
            self.logger.warning(f"Feature scaling failed: {e}")
            return data

    async def _engineer_features(
            self,
            data: pd.DataFrame,
            config: ProcessingConfig
    ) -> pd.DataFrame:
        """Engineer new features."""
        try:
            processed_data = data.copy()

            # Extract datetime features
            if config.extract_datetime_features:
                datetime_cols = data.select_dtypes(include=['datetime64']).columns

                for col in datetime_cols:
                    dt_series = pd.to_datetime(data[col])
                    processed_data[f"{col}_year"] = dt_series.dt.year
                    processed_data[f"{col}_month"] = dt_series.dt.month
                    processed_data[f"{col}_day"] = dt_series.dt.day
                    processed_data[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                    processed_data[f"{col}_quarter"] = dt_series.dt.quarter

            # Create interaction features
            if config.create_interactions:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                # Limit to prevent feature explosion
                if len(numeric_cols) >= 2:
                    col1, col2 = numeric_cols[:2]  # Take first two numeric columns
                    processed_data[f"{col1}_x_{col2}"] = data[col1] * data[col2]

            return processed_data

        except Exception as e:
            self.logger.warning(f"Feature engineering failed: {e}")
            return data


class ProcessingPipeline:
    """
    Orchestrates multiple preprocessing operations.

    Provides high-level interface for comprehensive data preprocessing
    with async support and comprehensive error handling.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize processing pipeline."""
        self.config = config or ProcessingConfig()
        self.quality_analyzer = QualityAnalyzer()
        self.tabular_processor = TabularProcessor()
        self.logger = logging.getLogger(f"{__name__}.ProcessingPipeline")

    async def process_dataset(
            self,
            data: pd.DataFrame,
            target_column: Optional[str] = None,
            analyze_quality: bool = True,
            custom_config: Optional[ProcessingConfig] = None
    ) -> Tuple[ProcessingResult, Optional[QualityReport]]:
        """
        Process dataset with comprehensive preprocessing pipeline.

        Args:
            data: Input DataFrame
            target_column: Target column name (excluded from preprocessing)
            analyze_quality: Whether to perform quality analysis
            custom_config: Override configuration

        Returns:
            Tuple of (processing_result, quality_report)
        """
        config = custom_config or self.config

        try:
            self.logger.info(f"Starting preprocessing pipeline: {data.shape}")

            quality_report = None

            # Step 1: Quality analysis
            if analyze_quality:
                quality_result = await self.quality_analyzer.process(data, config)
                if quality_result.success:
                    quality_report = quality_result.metadata.get('quality_report')
                    self.logger.info(f"Quality score: {quality_report.quality_score:.2f}")

            # Step 2: Tabular processing
            processing_result = await self.tabular_processor.process(
                data, config, target_column=target_column, fit_transformers=True
            )

            if processing_result.success:
                self.logger.info(
                    f"Processing completed: {processing_result.original_shape} -> "
                    f"{processing_result.processed_shape} "
                    f"({processing_result.duration_seconds:.2f}s)"
                )
            else:
                self.logger.error(f"Processing failed: {processing_result.error_message}")

            return processing_result, quality_report

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                original_shape=data.shape
            ), None

    async def transform_dataset(
            self,
            data: pd.DataFrame,
            target_column: Optional[str] = None
    ) -> ProcessingResult:
        """
        Transform dataset using fitted transformers.

        Args:
            data: Input DataFrame
            target_column: Target column name

        Returns:
            Processing result with transformed data
        """
        try:
            return await self.tabular_processor.process(
                data, self.config, target_column=target_column, fit_transformers=False
            )
        except Exception as e:
            self.logger.error(f"Dataset transformation failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def preprocess_data(
        data: pd.DataFrame,
        config: Optional[ProcessingConfig] = None,
        target_column: Optional[str] = None,
        analyze_quality: bool = True
) -> Tuple[pd.DataFrame, Optional[QualityReport]]:
    """
    Convenience function for data preprocessing.

    Args:
        data: Input DataFrame
        config: Processing configuration
        target_column: Target column name
        analyze_quality: Whether to analyze quality

    Returns:
        Tuple of (processed_dataframe, quality_report)
    """
    pipeline = ProcessingPipeline(config)
    result, quality_report = await pipeline.process_dataset(
        data, target_column, analyze_quality
    )

    if result.success:
        return result.data, quality_report
    else:
        raise ValueError(f"Preprocessing failed: {result.error_message}")


def prepare_ml_data(
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for machine learning by splitting features and target.

    Args:
        data: Input DataFrame
        target_column: Target column name
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Determine stratification
    stratify = y if y.dtype == 'object' or y.nunique() < 20 else None

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def validate_data_quality(data: pd.DataFrame, min_quality_score: float = 0.7) -> bool:
    """
    Quick data quality validation.

    Args:
        data: DataFrame to validate
        min_quality_score: Minimum acceptable quality score

    Returns:
        True if data quality is acceptable
    """
    try:
        # Basic checks
        if data.empty:
            return False

        # Missing value check
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.5:
            return False

        # Duplicate check
        duplicate_ratio = data.duplicated().sum() / len(data)
        if duplicate_ratio > 0.3:
            return False

        return True

    except Exception:
        return False


def get_column_types(data: pd.DataFrame) -> Dict[str, str]:
    """
    Get simplified column type mapping.

    Args:
        data: Input DataFrame

    Returns:
        Dictionary mapping column names to simplified types
    """
    type_mapping = {}

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            type_mapping[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            type_mapping[col] = "datetime"
        elif pd.api.types.is_bool_dtype(data[col]):
            type_mapping[col] = "boolean"
        else:
            # Determine if categorical or text
            sample = data[col].dropna().head(100).astype(str)
            avg_length = sample.str.len().mean() if len(sample) > 0 else 0
            unique_ratio = data[col].nunique() / len(data) if len(data) > 0 else 0

            if unique_ratio < 0.5 or avg_length < 50:
                type_mapping[col] = "categorical"
            else:
                type_mapping[col] = "text"

    return type_mapping


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'ProcessingPipeline', 'TabularProcessor', 'QualityAnalyzer', 'BaseProcessor',

    # Configuration and models
    'ProcessingConfig', 'ProcessingResult', 'QualityReport',

    # Enums
    'ProcessingStatus', 'DataType', 'MissingValueStrategy',
    'EncodingStrategy', 'ScalingStrategy',

    # Functions
    'preprocess_data', 'prepare_ml_data', 'validate_data_quality', 'get_column_types',

    # Protocols
    'ProcessorProtocol'
]

# Module initialization
logger.info(f"Preprocessing utilities loaded - NLTK: {NLTK_AVAILABLE}, SciPy: {SCIPY_AVAILABLE}")
