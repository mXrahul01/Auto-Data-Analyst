"""
Comprehensive Data Preprocessing Module for Auto-Analyst Platform

This module provides production-ready preprocessing functions for tabular, time series,
and text data. The functions are designed to be modular, reusable, and integrate
seamlessly with the Auto-Analyst ML pipeline and dashboard.

Features:
- Tabular Data Preprocessing: Missing values, encoding, scaling, feature engineering
- Time Series Preprocessing: Resampling, smoothing, rolling windows, trend analysis
- Text Preprocessing: Cleaning, tokenization, normalization, feature extraction
- Robust Error Handling: Comprehensive validation and graceful failure handling
- Performance Optimization: Vectorized operations and memory-efficient processing
- Type Safety: Full type hints and runtime validation

Components:
- TabularPreprocessor: Handles structured tabular data preprocessing
- TimeSeriesPreprocessor: Specialized for temporal data preprocessing  
- TextPreprocessor: Natural language processing and text cleaning
- DataQualityAnalyzer: Data quality assessment and reporting
- PreprocessingPipeline: Orchestrates multiple preprocessing steps

Usage:
    # Tabular preprocessing
    processor = TabularPreprocessor()
    cleaned_data = processor.preprocess_dataset(df, config)
    
    # Time series preprocessing
    ts_processor = TimeSeriesPreprocessor()
    processed_ts = ts_processor.preprocess_timeseries(ts_data)
    
    # Text preprocessing
    text_processor = TextPreprocessor()
    cleaned_text = text_processor.preprocess_text_data(text_data)

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine learning preprocessing utilities
- scipy: Statistical functions and signal processing
- nltk: Natural language processing (optional)
- spacy: Advanced NLP processing (optional)
"""

import logging
import warnings
import re
import unicodedata
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# Core data processing imports
import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.interpolate import interp1d

# Scikit-learn preprocessing imports
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# Text processing imports (optional)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')

# Configure logging
logger = logging.getLogger(__name__)

class DataType(str, Enum):
    """Data type classifications."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"

class MissingValueStrategy(str, Enum):
    """Missing value handling strategies."""
    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE = "interpolate"
    KNN = "knn"
    CONSTANT = "constant"

class EncodingStrategy(str, Enum):
    """Categorical encoding strategies."""
    ONE_HOT = "onehot"
    LABEL = "label"
    ORDINAL = "ordinal"
    TARGET = "target"
    BINARY = "binary"
    FREQUENCY = "frequency"

class ScalingStrategy(str, Enum):
    """Numeric scaling strategies."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    QUANTILE = "quantile"
    POWER = "power"
    NONE = "none"

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations."""
    
    # Missing value handling
    missing_value_strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN
    missing_value_threshold: float = 0.5  # Drop columns with >50% missing
    fill_value: Any = None
    
    # Categorical encoding
    categorical_encoding: EncodingStrategy = EncodingStrategy.ONE_HOT
    max_categories: int = 20  # Limit for one-hot encoding
    min_category_frequency: float = 0.01  # Min frequency to keep category
    
    # Numeric scaling
    numeric_scaling: ScalingStrategy = ScalingStrategy.STANDARD
    
    # Feature engineering
    create_interactions: bool = False
    interaction_degree: int = 2
    extract_datetime_features: bool = True
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Data quality
    remove_duplicates: bool = True
    remove_constant_features: bool = True
    variance_threshold: float = 0.01
    
    # Text processing
    lowercase_text: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    stem_words: bool = False
    lemmatize_words: bool = True
    min_text_length: int = 3
    
    # Time series
    resample_frequency: Optional[str] = None
    interpolation_method: str = "linear"
    smoothing_window: Optional[int] = None
    
    # Performance
    chunk_size: int = 10000
    n_jobs: int = -1

@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    
    # Basic statistics
    n_rows: int = 0
    n_columns: int = 0
    memory_usage: float = 0.0
    
    # Missing values
    missing_values_count: Dict[str, int] = field(default_factory=dict)
    missing_values_ratio: Dict[str, float] = field(default_factory=dict)
    columns_with_missing: List[str] = field(default_factory=list)
    
    # Data types
    column_types: Dict[str, str] = field(default_factory=dict)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    
    # Data quality issues
    duplicate_rows: int = 0
    constant_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    outlier_counts: Dict[str, int] = field(default_factory=dict)
    
    # Overall quality score
    quality_score: float = 0.0
    quality_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class DataQualityAnalyzer:
    """Comprehensive data quality assessment and reporting."""
    
    def __init__(self):
        """Initialize data quality analyzer."""
        self.report = DataQualityReport()
    
    def analyze_dataset(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Detailed data quality report
        """
        try:
            if df.empty:
                raise ValueError("Cannot analyze empty DataFrame")
            
            # Initialize report
            report = DataQualityReport()
            
            # Basic statistics
            report.n_rows, report.n_columns = df.shape
            report.memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Analyze missing values
            self._analyze_missing_values(df, report)
            
            # Analyze data types
            self._analyze_data_types(df, report)
            
            # Analyze data quality issues
            self._analyze_quality_issues(df, report)
            
            # Calculate overall quality score
            self._calculate_quality_score(report)
            
            # Generate recommendations
            self._generate_recommendations(report)
            
            logger.info(f"Data quality analysis completed: {report.quality_score:.2f}/1.0")
            return report
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {str(e)}")
            raise
    
    def _analyze_missing_values(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Analyze missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_ratios = missing_counts / len(df)
        
        report.missing_values_count = missing_counts.to_dict()
        report.missing_values_ratio = missing_ratios.to_dict()
        report.columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    def _analyze_data_types(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Analyze and classify data types."""
        report.column_types = df.dtypes.astype(str).to_dict()
        
        for column in df.columns:
            dtype = df[column].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                report.numeric_columns.append(column)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                report.datetime_columns.append(column)
            elif pd.api.types.is_bool_dtype(dtype):
                # Treat boolean as categorical
                report.categorical_columns.append(column)
            elif pd.api.types.is_object_dtype(dtype):
                # Determine if object is categorical or text
                if self._is_categorical_column(df[column]):
                    report.categorical_columns.append(column)
                else:
                    report.text_columns.append(column)
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Determine if a column should be treated as categorical."""
        try:
            # Remove missing values for analysis
            non_null_series = series.dropna()
            
            if len(non_null_series) == 0:
                return True  # Default to categorical for all-null columns
            
            # Check unique value ratio
            unique_ratio = len(non_null_series.unique()) / len(non_null_series)
            
            # Check if values look like categories vs text
            sample_values = non_null_series.head(100).astype(str)
            avg_length = sample_values.str.len().mean()
            
            # Categorical if: low unique ratio OR short average length
            is_categorical = unique_ratio < 0.5 or avg_length < 50
            
            return is_categorical
            
        except Exception:
            return True  # Default to categorical on error
    
    def _analyze_quality_issues(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Analyze various data quality issues."""
        # Duplicate rows
        report.duplicate_rows = df.duplicated().sum()
        
        # Constant columns
        for column in df.columns:
            if df[column].nunique(dropna=False) <= 1:
                report.constant_columns.append(column)
        
        # High cardinality columns
        for column in report.categorical_columns:
            if df[column].nunique() > 100:
                report.high_cardinality_columns.append(column)
        
        # Outliers in numeric columns
        for column in report.numeric_columns:
            if df[column].dtype in ['float64', 'int64']:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                if outliers > 0:
                    report.outlier_counts[column] = outliers
    
    def _calculate_quality_score(self, report: DataQualityReport) -> None:
        """Calculate overall data quality score."""
        try:
            score_components = []
            
            # Missing values score (0-1)
            if report.columns_with_missing:
                avg_missing_ratio = np.mean(list(report.missing_values_ratio.values()))
                missing_score = max(0, 1 - avg_missing_ratio * 2)
            else:
                missing_score = 1.0
            score_components.append(missing_score)
            
            # Duplicates score (0-1)
            if report.n_rows > 0:
                duplicate_ratio = report.duplicate_rows / report.n_rows
                duplicate_score = max(0, 1 - duplicate_ratio * 2)
            else:
                duplicate_score = 1.0
            score_components.append(duplicate_score)
            
            # Constant features score (0-1)
            if report.n_columns > 0:
                constant_ratio = len(report.constant_columns) / report.n_columns
                constant_score = max(0, 1 - constant_ratio * 2)
            else:
                constant_score = 1.0
            score_components.append(constant_score)
            
            # High cardinality score (0-1)
            if report.categorical_columns:
                high_card_ratio = len(report.high_cardinality_columns) / len(report.categorical_columns)
                cardinality_score = max(0, 1 - high_card_ratio)
            else:
                cardinality_score = 1.0
            score_components.append(cardinality_score)
            
            # Overall score
            report.quality_score = np.mean(score_components)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {str(e)}")
            report.quality_score = 0.5
    
    def _generate_recommendations(self, report: DataQualityReport) -> None:
        """Generate data quality improvement recommendations."""
        recommendations = []
        issues = []
        
        # Missing values
        if report.columns_with_missing:
            high_missing_cols = [col for col, ratio in report.missing_values_ratio.items() if ratio > 0.5]
            if high_missing_cols:
                issues.append(f"High missing values in columns: {high_missing_cols}")
                recommendations.append("Consider dropping columns with >50% missing values")
            else:
                recommendations.append("Handle missing values with appropriate imputation strategy")
        
        # Duplicates
        if report.duplicate_rows > 0:
            issues.append(f"Found {report.duplicate_rows} duplicate rows")
            recommendations.append("Remove duplicate rows to improve data quality")
        
        # Constant columns
        if report.constant_columns:
            issues.append(f"Constant columns detected: {report.constant_columns}")
            recommendations.append("Remove constant columns as they provide no information")
        
        # High cardinality
        if report.high_cardinality_columns:
            issues.append(f"High cardinality columns: {report.high_cardinality_columns}")
            recommendations.append("Consider grouping rare categories or using feature hashing")
        
        # Outliers
        if report.outlier_counts:
            total_outliers = sum(report.outlier_counts.values())
            issues.append(f"Detected {total_outliers} outliers across numeric columns")
            recommendations.append("Review outliers - remove if erroneous, transform if valid extreme values")
        
        # Data size
        if report.n_rows < 1000:
            issues.append("Small dataset size may limit model performance")
            recommendations.append("Consider collecting more data or using simpler models")
        
        report.quality_issues = issues
        report.recommendations = recommendations

class TabularPreprocessor:
    """Comprehensive tabular data preprocessing with configurable strategies."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize tabular preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.fitted_transformers: Dict[str, Any] = {}
        self.column_info: Dict[str, Dict[str, Any]] = {}
        
    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        fit_transformers: bool = True
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Complete preprocessing pipeline for tabular data.
        
        Args:
            df: Input DataFrame
            target_column: Target column name (excluded from preprocessing)
            fit_transformers: Whether to fit new transformers
            
        Returns:
            Tuple of (preprocessed_dataframe, quality_report)
        """
        try:
            logger.info(f"Starting tabular preprocessing: {df.shape}")
            
            if df.empty:
                raise ValueError("Cannot preprocess empty DataFrame")
            
            # Make a copy to avoid modifying original
            processed_df = df.copy()
            
            # Analyze data quality
            quality_analyzer = DataQualityAnalyzer()
            quality_report = quality_analyzer.analyze_dataset(df)
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col != target_column]
            if target_column and target_column not in df.columns:
                logger.warning(f"Target column '{target_column}' not found in DataFrame")
            
            # Store column information
            if fit_transformers:
                self._analyze_columns(processed_df[feature_columns])
            
            # Preprocessing steps
            processed_df = self._handle_duplicates(processed_df)
            processed_df = self._handle_missing_values(processed_df, feature_columns, fit_transformers)
            processed_df = self._remove_constant_features(processed_df, feature_columns)
            processed_df = self._encode_categorical_features(processed_df, fit_transformers)
            processed_df = self._scale_numeric_features(processed_df, fit_transformers)
            processed_df = self._engineer_features(processed_df, target_column)
            
            logger.info(f"Tabular preprocessing completed: {processed_df.shape}")
            return processed_df, quality_report
            
        except Exception as e:
            logger.error(f"Tabular preprocessing failed: {str(e)}")
            raise
    
    def _analyze_columns(self, df: pd.DataFrame) -> None:
        """Analyze and classify columns for preprocessing."""
        self.column_info = {}
        
        for column in df.columns:
            col_info = {
                'dtype': str(df[column].dtype),
                'nunique': df[column].nunique(),
                'missing_ratio': df[column].isnull().sum() / len(df)
            }
            
            # Classify column type
            if pd.api.types.is_numeric_dtype(df[column]):
                col_info['type'] = DataType.NUMERIC
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                col_info['type'] = DataType.DATETIME
            elif pd.api.types.is_bool_dtype(df[column]):
                col_info['type'] = DataType.BOOLEAN
            else:
                # Determine if categorical or text
                sample_values = df[column].dropna().head(100).astype(str)
                avg_length = sample_values.str.len().mean() if len(sample_values) > 0 else 0
                unique_ratio = col_info['nunique'] / len(df) if len(df) > 0 else 0
                
                if unique_ratio < 0.5 or avg_length < 50:
                    col_info['type'] = DataType.CATEGORICAL
                else:
                    col_info['type'] = DataType.TEXT
            
            self.column_info[column] = col_info
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate rows in the dataset."""
        if not self.config.remove_duplicates:
            return df
        
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        
        if len(df_clean) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        return df_clean
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        fit_transformers: bool
    ) -> pd.DataFrame:
        """Handle missing values using configured strategy."""
        try:
            # Drop columns with too many missing values
            columns_to_drop = []
            for column in feature_columns:
                missing_ratio = df[column].isnull().sum() / len(df)
                if missing_ratio > self.config.missing_value_threshold:
                    columns_to_drop.append(column)
            
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                logger.info(f"Dropped columns with high missing values: {columns_to_drop}")
            
            # Handle remaining missing values
            remaining_columns = [col for col in feature_columns if col not in columns_to_drop]
            
            for column in remaining_columns:
                if df[column].isnull().any():
                    df[column] = self._impute_missing_values(
                        df[column], column, fit_transformers
                    )
            
            return df
            
        except Exception as e:
            logger.error(f"Missing value handling failed: {str(e)}")
            return df
    
    def _impute_missing_values(
        self,
        series: pd.Series,
        column_name: str,
        fit_transformers: bool
    ) -> pd.Series:
        """Impute missing values for a single column."""
        try:
            strategy = self.config.missing_value_strategy
            col_info = self.column_info.get(column_name, {})
            col_type = col_info.get('type', DataType.NUMERIC)
            
            if strategy == MissingValueStrategy.DROP:
                return series.dropna()
            
            elif strategy == MissingValueStrategy.CONSTANT:
                fill_value = self.config.fill_value
                if fill_value is None:
                    fill_value = 0 if col_type == DataType.NUMERIC else "Unknown"
                return series.fillna(fill_value)
            
            elif strategy == MissingValueStrategy.FORWARD_FILL:
                return series.fillna(method='ffill')
            
            elif strategy == MissingValueStrategy.BACKWARD_FILL:
                return series.fillna(method='bfill')
            
            elif strategy == MissingValueStrategy.INTERPOLATE:
                if col_type == DataType.NUMERIC:
                    return series.interpolate()
                else:
                    return series.fillna(series.mode().iloc[0] if len(series.mode()) > 0 else "Unknown")
            
            elif strategy == MissingValueStrategy.KNN:
                # Use KNN imputation for numeric columns
                if col_type == DataType.NUMERIC and fit_transformers:
                    imputer_key = f"knn_imputer_{column_name}"
                    if imputer_key not in self.fitted_transformers:
                        imputer = KNNImputer(n_neighbors=5)
                        values_reshaped = series.values.reshape(-1, 1)
                        imputed_values = imputer.fit_transform(values_reshaped)
                        self.fitted_transformers[imputer_key] = imputer
                        return pd.Series(imputed_values.flatten(), index=series.index)
                    else:
                        imputer = self.fitted_transformers[imputer_key]
                        values_reshaped = series.values.reshape(-1, 1)
                        imputed_values = imputer.transform(values_reshaped)
                        return pd.Series(imputed_values.flatten(), index=series.index)
                else:
                    # Fallback to mode for non-numeric
                    return series.fillna(series.mode().iloc[0] if len(series.mode()) > 0 else "Unknown")
            
            else:
                # Default strategies (mean, median, mode)
                if col_type == DataType.NUMERIC:
                    if strategy == MissingValueStrategy.MEAN:
                        return series.fillna(series.mean())
                    elif strategy == MissingValueStrategy.MEDIAN:
                        return series.fillna(series.median())
                    else:
                        return series.fillna(series.mode().iloc[0] if len(series.mode()) > 0 else 0)
                else:
                    # Mode for categorical
                    return series.fillna(series.mode().iloc[0] if len(series.mode()) > 0 else "Unknown")
            
        except Exception as e:
            logger.warning(f"Imputation failed for column {column_name}: {str(e)}")
            return series.fillna(0 if pd.api.types.is_numeric_dtype(series) else "Unknown")
    
    def _remove_constant_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Remove constant or near-constant features."""
        if not self.config.remove_constant_features:
            return df
        
        try:
            # Remove truly constant columns
            constant_columns = []
            for column in feature_columns:
                if column in df.columns and df[column].nunique() <= 1:
                    constant_columns.append(column)
            
            if constant_columns:
                df = df.drop(columns=constant_columns)
                logger.info(f"Removed constant columns: {constant_columns}")
            
            # Remove low-variance numeric features
            numeric_columns = [col for col in df.columns 
                             if col in feature_columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if numeric_columns and self.config.variance_threshold > 0:
                selector = VarianceThreshold(threshold=self.config.variance_threshold)
                selector.fit(df[numeric_columns])
                
                low_variance_cols = [col for col, keep in zip(numeric_columns, selector.get_support()) if not keep]
                if low_variance_cols:
                    df = df.drop(columns=low_variance_cols)
                    logger.info(f"Removed low-variance columns: {low_variance_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Constant feature removal failed: {str(e)}")
            return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit_transformers: bool) -> pd.DataFrame:
        """Encode categorical features using configured strategy."""
        try:
            categorical_columns = [col for col in df.columns 
                                 if col in self.column_info and 
                                 self.column_info[col].get('type') == DataType.CATEGORICAL]
            
            if not categorical_columns:
                return df
            
            encoded_dfs = [df.drop(columns=categorical_columns)]
            
            for column in categorical_columns:
                encoded_column = self._encode_single_categorical(
                    df[column], column, fit_transformers
                )
                
                if isinstance(encoded_column, pd.DataFrame):
                    encoded_dfs.append(encoded_column)
                else:
                    encoded_dfs.append(pd.DataFrame({column: encoded_column}))
            
            # Combine all encoded features
            result_df = pd.concat(encoded_dfs, axis=1)
            logger.info(f"Encoded {len(categorical_columns)} categorical columns")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Categorical encoding failed: {str(e)}")
            return df
    
    def _encode_single_categorical(
        self,
        series: pd.Series,
        column_name: str,
        fit_transformers: bool
    ) -> Union[pd.Series, pd.DataFrame]:
        """Encode a single categorical column."""
        try:
            strategy = self.config.categorical_encoding
            
            # Handle high cardinality columns
            n_unique = series.nunique()
            if n_unique > self.config.max_categories and strategy == EncodingStrategy.ONE_HOT:
                strategy = EncodingStrategy.FREQUENCY  # Fallback for high cardinality
            
            if strategy == EncodingStrategy.LABEL:
                encoder_key = f"label_encoder_{column_name}"
                if fit_transformers:
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(series.astype(str))
                    self.fitted_transformers[encoder_key] = encoder
                else:
                    encoder = self.fitted_transformers.get(encoder_key)
                    if encoder:
                        # Handle unseen categories
                        try:
                            encoded = encoder.transform(series.astype(str))
                        except ValueError:
                            # Handle unseen labels by assigning a default value
                            encoded = []
                            for value in series.astype(str):
                                if value in encoder.classes_:
                                    encoded.append(encoder.transform([value])[0])
                                else:
                                    encoded.append(-1)  # Unseen category
                            encoded = np.array(encoded)
                    else:
                        encoded = series.astype('category').cat.codes
                
                return pd.Series(encoded, index=series.index, name=column_name)
            
            elif strategy == EncodingStrategy.ONE_HOT:
                encoder_key = f"onehot_encoder_{column_name}"
                if fit_transformers:
                    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(series.values.reshape(-1, 1))
                    self.fitted_transformers[encoder_key] = encoder
                    feature_names = [f"{column_name}_{cat}" for cat in encoder.categories_[0][1:]]
                else:
                    encoder = self.fitted_transformers.get(encoder_key)
                    if encoder:
                        encoded = encoder.transform(series.values.reshape(-1, 1))
                        feature_names = [f"{column_name}_{cat}" for cat in encoder.categories_[0][1:]]
                    else:
                        # Fallback to dummy encoding
                        return pd.get_dummies(series, prefix=column_name, drop_first=True)
                
                return pd.DataFrame(encoded, columns=feature_names, index=series.index)
            
            elif strategy == EncodingStrategy.FREQUENCY:
                # Frequency encoding
                freq_map = series.value_counts().to_dict()
                return series.map(freq_map).fillna(0)
            
            elif strategy == EncodingStrategy.ORDINAL:
                encoder_key = f"ordinal_encoder_{column_name}"
                if fit_transformers:
                    # For ordinal, assume natural ordering of categories
                    categories = sorted(series.dropna().unique())
                    encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
                    encoded = encoder.fit_transform(series.values.reshape(-1, 1))
                    self.fitted_transformers[encoder_key] = encoder
                else:
                    encoder = self.fitted_transformers.get(encoder_key)
                    if encoder:
                        encoded = encoder.transform(series.values.reshape(-1, 1))
                    else:
                        # Fallback to label encoding
                        encoded = series.astype('category').cat.codes.values.reshape(-1, 1)
                
                return pd.Series(encoded.flatten(), index=series.index, name=column_name)
            
            else:
                # Default: return as is
                return series
            
        except Exception as e:
            logger.warning(f"Encoding failed for column {column_name}: {str(e)}")
            return series.astype('category').cat.codes
    
    def _scale_numeric_features(self, df: pd.DataFrame, fit_transformers: bool) -> pd.DataFrame:
        """Scale numeric features using configured strategy."""
        try:
            numeric_columns = [col for col in df.columns 
                             if pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns or self.config.numeric_scaling == ScalingStrategy.NONE:
                return df
            
            strategy = self.config.numeric_scaling
            scaler_key = f"scaler_{strategy.value}"
            
            if fit_transformers:
                # Create scaler based on strategy
                if strategy == ScalingStrategy.STANDARD:
                    scaler = StandardScaler()
                elif strategy == ScalingStrategy.MINMAX:
                    scaler = MinMaxScaler()
                elif strategy == ScalingStrategy.ROBUST:
                    scaler = RobustScaler()
                elif strategy == ScalingStrategy.MAXABS:
                    scaler = MaxAbsScaler()
                elif strategy == ScalingStrategy.QUANTILE:
                    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
                elif strategy == ScalingStrategy.POWER:
                    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                else:
                    return df
                
                # Fit and transform
                df_scaled = df.copy()
                df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                self.fitted_transformers[scaler_key] = scaler
                
            else:
                # Use fitted scaler
                scaler = self.fitted_transformers.get(scaler_key)
                if scaler:
                    df_scaled = df.copy()
                    df_scaled[numeric_columns] = scaler.transform(df[numeric_columns])
                else:
                    df_scaled = df
            
            logger.info(f"Scaled {len(numeric_columns)} numeric columns using {strategy.value}")
            return df_scaled
            
        except Exception as e:
            logger.error(f"Numeric scaling failed: {str(e)}")
            return df
    
    def _engineer_features(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Engineer new features from existing ones."""
        try:
            engineered_df = df.copy()
            
            # Extract datetime features
            if self.config.extract_datetime_features:
                engineered_df = self._extract_datetime_features(engineered_df)
            
            # Create polynomial features
            if self.config.create_polynomial_features:
                engineered_df = self._create_polynomial_features(engineered_df, target_column)
            
            # Create interaction features
            if self.config.create_interactions:
                engineered_df = self._create_interaction_features(engineered_df, target_column)
            
            if len(engineered_df.columns) > len(df.columns):
                new_features = len(engineered_df.columns) - len(df.columns)
                logger.info(f"Created {new_features} new features")
            
            return engineered_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            return df
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        try:
            datetime_columns = [col for col in df.columns 
                              if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            for column in datetime_columns:
                dt_series = pd.to_datetime(df[column])
                
                # Extract common datetime features
                df[f"{column}_year"] = dt_series.dt.year
                df[f"{column}_month"] = dt_series.dt.month
                df[f"{column}_day"] = dt_series.dt.day
                df[f"{column}_dayofweek"] = dt_series.dt.dayofweek
                df[f"{column}_quarter"] = dt_series.dt.quarter
                df[f"{column}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
                
                # Hour and minute for datetime with time
                if dt_series.dt.hour.nunique() > 1:
                    df[f"{column}_hour"] = dt_series.dt.hour
                    df[f"{column}_minute"] = dt_series.dt.minute
            
            return df
            
        except Exception as e:
            logger.warning(f"Datetime feature extraction failed: {str(e)}")
            return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Create polynomial features from numeric columns."""
        try:
            numeric_columns = [col for col in df.columns 
                             if col != target_column and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_columns) < 2:  # Need at least 2 numeric columns
                return df
            
            # Limit to prevent feature explosion
            selected_columns = numeric_columns[:5]  # Max 5 columns
            
            poly = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                include_bias=False,
                interaction_only=False
            )
            
            poly_features = poly.fit_transform(df[selected_columns])
            feature_names = poly.get_feature_names_out(selected_columns)
            
            # Add only new polynomial features (exclude original features)
            original_feature_count = len(selected_columns)
            new_features = poly_features[:, original_feature_count:]
            new_feature_names = [f"poly_{name}" for name in feature_names[original_feature_count:]]
            
            # Add to dataframe
            poly_df = pd.DataFrame(new_features, columns=new_feature_names, index=df.index)
            result_df = pd.concat([df, poly_df], axis=1)
            
            return result_df
            
        except Exception as e:
            logger.warning(f"Polynomial feature creation failed: {str(e)}")
            return df
    
    def _create_interaction_features(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        try:
            numeric_columns = [col for col in df.columns 
                             if col != target_column and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_columns) < 2:
                return df
            
            # Create pairwise interactions (limit to prevent explosion)
            selected_columns = numeric_columns[:4]  # Max 4 columns for interactions
            
            interaction_df = df.copy()
            
            for i, col1 in enumerate(selected_columns):
                for j, col2 in enumerate(selected_columns[i+1:], i+1):
                    # Multiplication interaction
                    interaction_name = f"{col1}_x_{col2}"
                    interaction_df[interaction_name] = df[col1] * df[col2]
                    
                    # Division interaction (with safety check)
                    if (df[col2] != 0).all():
                        division_name = f"{col1}_div_{col2}"
                        interaction_df[division_name] = df[col1] / df[col2]
            
            return interaction_df
            
        except Exception as e:
            logger.warning(f"Interaction feature creation failed: {str(e)}")
            return df

class TimeSeriesPreprocessor:
    """Specialized preprocessing for time series data."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize time series preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
    
    def preprocess_timeseries(
        self,
        df: pd.DataFrame,
        time_column: str,
        value_columns: Optional[List[str]] = None,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Comprehensive time series preprocessing.
        
        Args:
            df: Input DataFrame with time series data
            time_column: Name of the time/date column
            value_columns: Columns to treat as time series values
            frequency: Target frequency for resampling
            
        Returns:
            Preprocessed time series DataFrame
        """
        try:
            logger.info(f"Starting time series preprocessing: {df.shape}")
            
            if df.empty:
                raise ValueError("Cannot preprocess empty DataFrame")
            
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found in DataFrame")
            
            # Make a copy
            ts_df = df.copy()
            
            # Convert time column to datetime
            ts_df[time_column] = pd.to_datetime(ts_df[time_column])
            
            # Set time column as index
            ts_df = ts_df.set_index(time_column)
            
            # Sort by time
            ts_df = ts_df.sort_index()
            
            # Determine value columns if not specified
            if value_columns is None:
                value_columns = [col for col in ts_df.columns 
                               if pd.api.types.is_numeric_dtype(ts_df[col])]
            
            # Preprocessing steps
            ts_df = self._handle_missing_timestamps(ts_df, frequency)
            ts_df = self._handle_missing_values_ts(ts_df, value_columns)
            ts_df = self._resample_timeseries(ts_df, value_columns, frequency)
            ts_df = self._smooth_timeseries(ts_df, value_columns)
            ts_df = self._create_time_features(ts_df)
            ts_df = self._create_lag_features(ts_df, value_columns)
            ts_df = self._create_rolling_features(ts_df, value_columns)
            
            logger.info(f"Time series preprocessing completed: {ts_df.shape}")
            return ts_df
            
        except Exception as e:
            logger.error(f"Time series preprocessing failed: {str(e)}")
            raise
    
    def _handle_missing_timestamps(self, df: pd.DataFrame, frequency: Optional[str]) -> pd.DataFrame:
        """Handle missing timestamps in time series."""
        try:
            if frequency is None:
                # Try to infer frequency
                try:
                    inferred_freq = pd.infer_freq(df.index)
                    if inferred_freq:
                        frequency = inferred_freq
                    else:
                        # Use median time difference as frequency
                        time_diffs = df.index.to_series().diff().dropna()
                        median_diff = time_diffs.median()
                        
                        if median_diff.total_seconds() < 3600:  # Less than 1 hour
                            frequency = f"{int(median_diff.total_seconds())}S"
                        elif median_diff.total_seconds() < 86400:  # Less than 1 day
                            frequency = f"{int(median_diff.total_seconds()/3600)}H"
                        else:
                            frequency = f"{int(median_diff.total_seconds()/86400)}D"
                except Exception:
                    frequency = "D"  # Default to daily
            
            # Create complete time range
            start_time = df.index.min()
            end_time = df.index.max()
            complete_range = pd.date_range(start=start_time, end=end_time, freq=frequency)
            
            # Reindex to fill missing timestamps
            df_complete = df.reindex(complete_range)
            
            if len(df_complete) > len(df):
                missing_count = len(df_complete) - len(df)
                logger.info(f"Added {missing_count} missing timestamps")
            
            return df_complete
            
        except Exception as e:
            logger.warning(f"Missing timestamp handling failed: {str(e)}")
            return df
    
    def _handle_missing_values_ts(self, df: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
        """Handle missing values in time series data."""
        try:
            for column in value_columns:
                if column in df.columns and df[column].isnull().any():
                    method = self.config.interpolation_method
                    
                    if method == "linear":
                        df[column] = df[column].interpolate(method='linear')
                    elif method == "polynomial":
                        df[column] = df[column].interpolate(method='polynomial', order=2)
                    elif method == "spline":
                        df[column] = df[column].interpolate(method='spline', order=3)
                    elif method == "forward":
                        df[column] = df[column].fillna(method='ffill')
                    elif method == "backward":
                        df[column] = df[column].fillna(method='bfill')
                    else:
                        # Default to linear interpolation
                        df[column] = df[column].interpolate(method='linear')
                    
                    # Fill any remaining missing values at edges
                    df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.warning(f"Time series missing value handling failed: {str(e)}")
            return df
    
    def _resample_timeseries(
        self,
        df: pd.DataFrame,
        value_columns: List[str],
        frequency: Optional[str]
    ) -> pd.DataFrame:
        """Resample time series to target frequency."""
        try:
            if frequency is None or frequency == self.config.resample_frequency:
                return df
            
            target_freq = self.config.resample_frequency or frequency
            
            # Resample numeric columns
            resampled_data = {}
            
            for column in value_columns:
                if column in df.columns:
                    # Use appropriate aggregation method
                    if 'price' in column.lower() or 'value' in column.lower():
                        # Use mean for price/value columns
                        resampled_data[column] = df[column].resample(target_freq).mean()
                    elif 'count' in column.lower() or 'volume' in column.lower():
                        # Use sum for count/volume columns
                        resampled_data[column] = df[column].resample(target_freq).sum()
                    else:
                        # Default to mean
                        resampled_data[column] = df[column].resample(target_freq).mean()
            
            # Handle non-numeric columns
            for column in df.columns:
                if column not in value_columns:
                    try:
                        resampled_data[column] = df[column].resample(target_freq).first()
                    except Exception:
                        continue
            
            resampled_df = pd.DataFrame(resampled_data)
            
            if len(resampled_df) != len(df):
                logger.info(f"Resampled time series from {len(df)} to {len(resampled_df)} records")
            
            return resampled_df
            
        except Exception as e:
            logger.warning(f"Time series resampling failed: {str(e)}")
            return df
    
    def _smooth_timeseries(self, df: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
        """Apply smoothing to time series data."""
        try:
            if not self.config.smoothing_window:
                return df
            
            window_size = self.config.smoothing_window
            
            for column in value_columns:
                if column in df.columns:
                    # Apply rolling mean smoothing
                    original_col = f"{column}_original"
                    smoothed_col = f"{column}_smooth"
                    
                    # Keep original values
                    df[original_col] = df[column]
                    
                    # Apply smoothing
                    df[smoothed_col] = df[column].rolling(window=window_size, center=True).mean()
                    
                    # Optional: Replace original with smoothed
                    # df[column] = df[smoothed_col]
            
            return df
            
        except Exception as e:
            logger.warning(f"Time series smoothing failed: {str(e)}")
            return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from the datetime index."""
        try:
            # Basic time features
            df['hour'] = df.index.hour
            df['day'] = df.index.day
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            
            # Cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            
            # Binary features
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            df['is_month_start'] = df.index.is_month_start.astype(int)
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
            
            logger.info("Created time-based features")
            return df
            
        except Exception as e:
            logger.warning(f"Time feature creation failed: {str(e)}")
            return df
    
    def _create_lag_features(self, df: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
        """Create lag features for time series forecasting."""
        try:
            lag_periods = [1, 7, 30]  # 1 day, 1 week, 1 month lags
            
            for column in value_columns:
                if column in df.columns:
                    for lag in lag_periods:
                        lag_col_name = f"{column}_lag_{lag}"
                        df[lag_col_name] = df[column].shift(lag)
            
            logger.info(f"Created lag features for {len(value_columns)} columns")
            return df
            
        except Exception as e:
            logger.warning(f"Lag feature creation failed: {str(e)}")
            return df
    
    def _create_rolling_features(self, df: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
        """Create rolling window statistical features."""
        try:
            windows = [7, 30]  # 7-day and 30-day windows
            
            for column in value_columns:
                if column in df.columns:
                    for window in windows:
                        # Rolling statistics
                        df[f"{column}_roll_mean_{window}"] = df[column].rolling(window=window).mean()
                        df[f"{column}_roll_std_{window}"] = df[column].rolling(window=window).std()
                        df[f"{column}_roll_min_{window}"] = df[column].rolling(window=window).min()
                        df[f"{column}_roll_max_{window}"] = df[column].rolling(window=window).max()
                        df[f"{column}_roll_median_{window}"] = df[column].rolling(window=window).median()
                        
                        # Rolling percentiles
                        df[f"{column}_roll_q25_{window}"] = df[column].rolling(window=window).quantile(0.25)
                        df[f"{column}_roll_q75_{window}"] = df[column].rolling(window=window).quantile(0.75)
            
            logger.info(f"Created rolling features for {len(value_columns)} columns")
            return df
            
        except Exception as e:
            logger.warning(f"Rolling feature creation failed: {str(e)}")
            return df

class TextPreprocessor:
    """Comprehensive text preprocessing for NLP tasks."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize text preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.stopwords_set = set()
        self.stemmer = None
        self.lemmatizer = None
        
        # Initialize NLP tools if available
        self._initialize_nlp_tools()
    
    def _initialize_nlp_tools(self) -> None:
        """Initialize NLP tools (NLTK, spaCy)."""
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                import nltk
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                
                # Initialize tools
                self.stopwords_set = set(stopwords.words('english'))
                self.stemmer = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
                
                logger.info("NLTK tools initialized")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {str(e)}")
    
    def preprocess_text_data(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Comprehensive text preprocessing pipeline.
        
        Args:
            df: Input DataFrame with text data
            text_columns: Columns containing text data
            
        Returns:
            DataFrame with preprocessed text
        """
        try:
            logger.info(f"Starting text preprocessing: {df.shape}")
            
            if df.empty:
                raise ValueError("Cannot preprocess empty DataFrame")
            
            # Identify text columns if not specified
            if text_columns is None:
                text_columns = [col for col in df.columns 
                              if df[col].dtype == 'object' and self._is_text_column(df[col])]
            
            if not text_columns:
                logger.warning("No text columns found for preprocessing")
                return df
            
            processed_df = df.copy()
            
            # Process each text column
            for column in text_columns:
                if column in processed_df.columns:
                    processed_df[column] = self._preprocess_text_column(processed_df[column])
                    
                    # Create additional text features
                    processed_df = self._create_text_features(processed_df, column)
            
            logger.info(f"Text preprocessing completed: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            raise
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Determine if a column contains text data."""
        try:
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False
            
            # Check average text length
            avg_length = sample.astype(str).str.len().mean()
            
            # Check for typical text characteristics
            has_spaces = sample.astype(str).str.contains(' ').mean() > 0.5
            long_enough = avg_length > 10
            
            return has_spaces and long_enough
            
        except Exception:
            return False
    
    def _preprocess_text_column(self, series: pd.Series) -> pd.Series:
        """Preprocess a single text column."""
        try:
            # Convert to string and handle missing values
            text_series = series.fillna("").astype(str)
            
            # Apply preprocessing steps
            if self.config.lowercase_text:
                text_series = text_series.str.lower()
            
            # Remove special characters and normalize
            text_series = text_series.apply(self._clean_text)
            
            if self.config.remove_punctuation:
                text_series = text_series.apply(self._remove_punctuation)
            
            # Tokenization and advanced processing
            if NLTK_AVAILABLE:
                if self.config.remove_stopwords:
                    text_series = text_series.apply(self._remove_stopwords)
                
                if self.config.stem_words and self.stemmer:
                    text_series = text_series.apply(self._stem_text)
                elif self.config.lemmatize_words and self.lemmatizer:
                    text_series = text_series.apply(self._lemmatize_text)
            
            # Filter by minimum length
            if self.config.min_text_length > 0:
                text_series = text_series.apply(
                    lambda x: x if len(x.split()) >= self.config.min_text_length else ""
                )
            
            return text_series
            
        except Exception as e:
            logger.warning(f"Text column preprocessing failed: {str(e)}")
            return series.fillna("").astype(str)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        try:
            # Handle Unicode normalization
            text = unicodedata.normalize('NFKD', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove extra punctuation
            text = re.sub(r'[^\w\s\.]', ' ', text)
            
            return text.strip()
            
        except Exception:
            return text
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        try:
            # Keep letters, numbers, and spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception:
            return text
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        try:
            if not self.stopwords_set:
                return text
            
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in self.stopwords_set]
            return ' '.join(filtered_words)
            
        except Exception:
            return text
    
    def _stem_text(self, text: str) -> str:
        """Apply stemming to text."""
        try:
            if not self.stemmer:
                return text
            
            words = text.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
            
        except Exception:
            return text
    
    def _lemmatize_text(self, text: str) -> str:
        """Apply lemmatization to text."""
        try:
            if not self.lemmatizer:
                return text
            
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
            
        except Exception:
            return text
    
    def _create_text_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Create additional features from text data."""
        try:
            # Basic text statistics
            df[f"{text_column}_length"] = df[text_column].str.len()
            df[f"{text_column}_word_count"] = df[text_column].str.split().str.len()
            df[f"{text_column}_char_count"] = df[text_column].str.len()
            
            # Advanced text features
            df[f"{text_column}_sentence_count"] = df[text_column].str.count(r'[.!?]+')
            df[f"{text_column}_avg_word_length"] = df[text_column].apply(self._avg_word_length)
            df[f"{text_column}_uppercase_ratio"] = df[text_column].apply(self._uppercase_ratio)
            
            # Linguistic features
            if NLTK_AVAILABLE:
                df[f"{text_column}_unique_word_ratio"] = df[text_column].apply(self._unique_word_ratio)
                df[f"{text_column}_complexity_score"] = df[text_column].apply(self._text_complexity_score)
            
            return df
            
        except Exception as e:
            logger.warning(f"Text feature creation failed: {str(e)}")
            return df
    
    def _avg_word_length(self, text: str) -> float:
        """Calculate average word length in text."""
        try:
            words = text.split()
            if not words:
                return 0.0
            return sum(len(word) for word in words) / len(words)
        except Exception:
            return 0.0
    
    def _uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase characters."""
        try:
            if not text:
                return 0.0
            uppercase_count = sum(1 for char in text if char.isupper())
            return uppercase_count / len(text)
        except Exception:
            return 0.0
    
    def _unique_word_ratio(self, text: str) -> float:
        """Calculate ratio of unique words to total words."""
        try:
            words = text.split()
            if not words:
                return 0.0
            unique_words = set(words)
            return len(unique_words) / len(words)
        except Exception:
            return 0.0
    
    def _text_complexity_score(self, text: str) -> float:
        """Calculate a simple text complexity score."""
        try:
            words = text.split()
            if not words:
                return 0.0
            
            # Average word length
            avg_word_len = sum(len(word) for word in words) / len(words)
            
            # Sentence count approximation
            sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
            
            # Words per sentence
            words_per_sentence = len(words) / sentence_count
            
            # Simple complexity score
            complexity = (avg_word_len * 0.4) + (words_per_sentence * 0.6)
            return min(complexity / 10, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.0

class PreprocessingPipeline:
    """Orchestrates multiple preprocessing steps for different data types."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
        # Initialize specialized preprocessors
        self.tabular_preprocessor = TabularPreprocessor(config)
        self.timeseries_preprocessor = TimeSeriesPreprocessor(config)
        self.text_preprocessor = TextPreprocessor(config)
        
        self.quality_report: Optional[DataQualityReport] = None
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        data_type: str = "tabular",
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        text_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Comprehensive data preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            data_type: Type of data ('tabular', 'timeseries', 'text')
            target_column: Target column for supervised learning
            time_column: Time column for time series data
            text_columns: Text columns for NLP preprocessing
            
        Returns:
            Tuple of (preprocessed_dataframe, quality_report)
        """
        try:
            logger.info(f"Starting {data_type} preprocessing pipeline")
            
            if df.empty:
                raise ValueError("Cannot preprocess empty DataFrame")
            
            processed_df = df.copy()
            quality_report = DataQualityReport()
            
            if data_type.lower() == "timeseries":
                if not time_column:
                    raise ValueError("Time column required for time series preprocessing")
                
                processed_df = self.timeseries_preprocessor.preprocess_timeseries(
                    processed_df, time_column
                )
                
                # Analyze quality after time series preprocessing
                analyzer = DataQualityAnalyzer()
                quality_report = analyzer.analyze_dataset(processed_df)
                
            elif data_type.lower() == "text":
                processed_df = self.text_preprocessor.preprocess_text_data(
                    processed_df, text_columns
                )
                
                # For text data, also apply basic tabular preprocessing
                processed_df, quality_report = self.tabular_preprocessor.preprocess_dataset(
                    processed_df, target_column
                )
                
            else:  # Default to tabular
                processed_df, quality_report = self.tabular_preprocessor.preprocess_dataset(
                    processed_df, target_column
                )
            
            self.quality_report = quality_report
            
            logger.info(f"Preprocessing pipeline completed: {processed_df.shape}")
            return processed_df, quality_report
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise

# Convenience functions for easy import and usage

def preprocess_data(
    df: pd.DataFrame,
    config: Optional[PreprocessingConfig] = None,
    data_type: str = "tabular",
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    text_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Main preprocessing function with automatic pipeline selection.
    
    Args:
        df: Input DataFrame
        config: Preprocessing configuration
        data_type: Type of data processing needed
        target_column: Target column name
        time_column: Time column for time series
        text_columns: Text columns for NLP
        
    Returns:
        Tuple of (processed_dataframe, quality_report)
    """
    pipeline = PreprocessingPipeline(config)
    return pipeline.preprocess_data(
        df, data_type, target_column, time_column, text_columns
    )

def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_missing: bool = True,
    missing_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Quick dataset cleaning function.
    
    Args:
        df: Input DataFrame
        remove_duplicates: Remove duplicate rows
        handle_missing: Handle missing values
        missing_threshold: Threshold for dropping columns with missing values
        
    Returns:
        Cleaned DataFrame
    """
    try:
        cleaned_df = df.copy()
        
        if remove_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
        
        if handle_missing:
            # Drop columns with too many missing values
            missing_ratios = cleaned_df.isnull().sum() / len(cleaned_df)
            cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            
            # Fill remaining missing values
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype in ['object', 'category']:
                    cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Dataset cleaning failed: {str(e)}")
        return df

def normalize_features(
    df: pd.DataFrame,
    method: str = "standard",
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize numeric features in DataFrame.
    
    Args:
        df: Input DataFrame
        method: Normalization method ('standard', 'minmax', 'robust')
        exclude_columns: Columns to exclude from normalization
        
    Returns:
        DataFrame with normalized features
    """
    try:
        normalized_df = df.copy()
        exclude_columns = exclude_columns or []
        
        # Find numeric columns
        numeric_columns = [col for col in df.columns 
                         if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_columns]
        
        if not numeric_columns:
            return normalized_df
        
        # Apply normalization
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        logger.info(f"Normalized {len(numeric_columns)} columns using {method} scaling")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Feature normalization failed: {str(e)}")
        return df

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    fill_value: Any = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Missing value strategy ('mean', 'median', 'mode', 'constant')
        fill_value: Value to use for constant strategy
        
    Returns:
        DataFrame with handled missing values
    """
    try:
        handled_df = df.copy()
        
        for column in df.columns:
            if df[column].isnull().any():
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
                    handled_df[column] = df[column].fillna(df[column].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
                    handled_df[column] = df[column].fillna(df[column].median())
                elif strategy == "mode":
                    mode_value = df[column].mode()
                    fill_val = mode_value.iloc[0] if len(mode_value) > 0 else "Unknown"
                    handled_df[column] = df[column].fillna(fill_val)
                elif strategy == "constant":
                    fill_val = fill_value if fill_value is not None else "Unknown"
                    handled_df[column] = df[column].fillna(fill_val)
                else:
                    # Default to median/mode
                    if pd.api.types.is_numeric_dtype(df[column]):
                        handled_df[column] = df[column].fillna(df[column].median())
                    else:
                        mode_value = df[column].mode()
                        fill_val = mode_value.iloc[0] if len(mode_value) > 0 else "Unknown"
                        handled_df[column] = df[column].fillna(fill_val)
        
        return handled_df
        
    except Exception as e:
        logger.error(f"Missing value handling failed: {str(e)}")
        return df

def encode_categorical_features(
    df: pd.DataFrame,
    method: str = "onehot",
    max_categories: int = 20
) -> pd.DataFrame:
    """
    Encode categorical features in DataFrame.
    
    Args:
        df: Input DataFrame
        method: Encoding method ('onehot', 'label', 'frequency')
        max_categories: Maximum categories for one-hot encoding
        
    Returns:
        DataFrame with encoded categorical features
    """
    try:
        encoded_df = df.copy()
        
        # Find categorical columns
        categorical_columns = [col for col in df.columns 
                             if df[col].dtype == 'object' or df[col].dtype.name == 'category']
        
        for column in categorical_columns:
            n_unique = df[column].nunique()
            
            if method == "onehot" and n_unique <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                encoded_df = pd.concat([encoded_df.drop(columns=[column]), dummies], axis=1)
                
            elif method == "label":
                # Label encoding
                le = LabelEncoder()
                encoded_df[column] = le.fit_transform(df[column].astype(str))
                
            elif method == "frequency":
                # Frequency encoding
                freq_map = df[column].value_counts().to_dict()
                encoded_df[column] = df[column].map(freq_map)
                
            else:
                # Default to frequency encoding for high cardinality
                freq_map = df[column].value_counts().to_dict()
                encoded_df[column] = df[column].map(freq_map)
        
        logger.info(f"Encoded {len(categorical_columns)} categorical columns")
        return encoded_df
        
    except Exception as e:
        logger.error(f"Categorical encoding failed: {str(e)}")
        return df

def scale_numeric_features(
    df: pd.DataFrame,
    method: str = "standard",
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Scale numeric features (alias for normalize_features for backward compatibility).
    
    Args:
        df: Input DataFrame
        method: Scaling method
        exclude_columns: Columns to exclude
        
    Returns:
        DataFrame with scaled features
    """
    return normalize_features(df, method, exclude_columns)

def extract_features(
    df: pd.DataFrame,
    create_interactions: bool = False,
    extract_datetime: bool = True
) -> pd.DataFrame:
    """
    Extract additional features from existing columns.
    
    Args:
        df: Input DataFrame
        create_interactions: Create interaction features
        extract_datetime: Extract datetime features
        
    Returns:
        DataFrame with additional features
    """
    try:
        feature_df = df.copy()
        
        # Extract datetime features
        if extract_datetime:
            datetime_columns = [col for col in df.columns 
                              if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            for column in datetime_columns:
                dt_series = pd.to_datetime(feature_df[column])
                feature_df[f"{column}_year"] = dt_series.dt.year
                feature_df[f"{column}_month"] = dt_series.dt.month
                feature_df[f"{column}_dayofweek"] = dt_series.dt.dayofweek
                feature_df[f"{column}_quarter"] = dt_series.dt.quarter
        
        # Create simple interaction features
        if create_interactions:
            numeric_columns = [col for col in df.columns 
                             if pd.api.types.is_numeric_dtype(df[col])]
            
            # Limit to prevent feature explosion
            selected_columns = numeric_columns[:3]
            
            for i, col1 in enumerate(selected_columns):
                for j, col2 in enumerate(selected_columns[i+1:], i+1):
                    feature_df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        
        if len(feature_df.columns) > len(df.columns):
            new_features = len(feature_df.columns) - len(df.columns)
            logger.info(f"Created {new_features} additional features")
        
        return feature_df
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return df

def transform_data_types(df: pd.DataFrame, type_hints: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Transform data types based on hints or automatic detection.
    
    Args:
        df: Input DataFrame
        type_hints: Dictionary mapping column names to desired types
        
    Returns:
        DataFrame with transformed data types
    """
    try:
        transformed_df = df.copy()
        
        for column in df.columns:
            target_type = type_hints.get(column) if type_hints else None
            
            if target_type:
                if target_type.lower() == "datetime":
                    transformed_df[column] = pd.to_datetime(df[column], errors='coerce')
                elif target_type.lower() in ["int", "integer"]:
                    transformed_df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                elif target_type.lower() == "float":
                    transformed_df[column] = pd.to_numeric(df[column], errors='coerce')
                elif target_type.lower() in ["str", "string", "category"]:
                    transformed_df[column] = df[column].astype(str)
            else:
                # Automatic type detection
                if df[column].dtype == 'object':
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[column], errors='coerce')
                    if not numeric_series.isnull().all():
                        transformed_df[column] = numeric_series
                    else:
                        # Try to convert to datetime
                        try:
                            datetime_series = pd.to_datetime(df[column], errors='coerce')
                            if not datetime_series.isnull().all():
                                transformed_df[column] = datetime_series
                        except Exception:
                            pass
        
        return transformed_df
        
    except Exception as e:
        logger.error(f"Data type transformation failed: {str(e)}")
        return df

def prepare_ml_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare dataset for machine learning by splitting features and target.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of dataset for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    try:
        from sklearn.model_selection import train_test_split
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
            if y.dtype == 'object' or y.nunique() < 20 else None
        )
        
        logger.info(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"ML dataset preparation failed: {str(e)}")
        raise

# Export key functions and classes
__all__ = [
    # Main classes
    'PreprocessingPipeline', 'TabularPreprocessor', 'TimeSeriesPreprocessor', 'TextPreprocessor',
    'DataQualityAnalyzer',
    
    # Configuration classes
    'PreprocessingConfig', 'DataQualityReport',
    
    # Enums
    'DataType', 'MissingValueStrategy', 'EncodingStrategy', 'ScalingStrategy',
    
    # Main functions
    'preprocess_data', 'clean_dataset', 'normalize_features', 'handle_missing_values',
    'encode_categorical_features', 'scale_numeric_features', 'extract_features',
    'transform_data_types', 'prepare_ml_dataset'
]

# Initialize module
logger.info(f"Preprocessing module loaded - NLTK available: {NLTK_AVAILABLE}, spaCy available: {SPACY_AVAILABLE}")
