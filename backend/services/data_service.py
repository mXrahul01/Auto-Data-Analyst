"""
Data Service for Auto-Analyst Platform

This module provides comprehensive dataset management capabilities including:
- Multi-format dataset loading (CSV, Excel, JSON, Parquet)
- Intelligent data cleaning and preprocessing
- Automatic data type inference and validation
- Memory-efficient processing for large datasets
- Feature engineering and transformation pipelines
- Integration with ML pipelines and validation systems
- Data quality assessment and reporting

Features:
- Support for multiple file formats with automatic detection
- Chunked processing for large datasets to prevent memory issues
- Automatic handling of missing values, duplicates, and inconsistencies
- Intelligent column type detection (numerical, categorical, datetime, text)
- Configurable preprocessing pipelines
- Data quality metrics and validation reporting
- Integration with validation.py and ml_service.py
- Production-ready error handling and logging
- Async/await support for non-blocking operations
- Caching and performance optimization

Usage:
    # Initialize service
    data_service = DataService()
    
    # Load and process dataset
    result = await data_service.load_dataset('data.csv')
    processed_df = await data_service.preprocess_dataset(result.dataframe)
    
    # Get data insights
    insights = await data_service.analyze_dataset(processed_df)
    
    # Prepare for ML pipeline
    ml_ready_data = await data_service.prepare_for_ml(processed_df, target_column='target')
"""

import asyncio
import logging
import warnings
import os
import tempfile
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Literal
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import io
import json
import time
import gc

# Data processing libraries
import pandas as pd
import numpy as np
from scipy import stats

# File format support
try:
    import openpyxl  # For Excel support
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_SUPPORT = True
except ImportError:
    PARQUET_SUPPORT = False

try:
    import fastparquet
    FASTPARQUET_SUPPORT = True
except ImportError:
    FASTPARQUET_SUPPORT = False

# Data validation and cleaning
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

class FileFormat(str, Enum):
    """Supported file formats."""
    CSV = "csv"
    EXCEL = "xlsx"
    JSON = "json"
    PARQUET = "parquet"
    TSV = "tsv"
    AUTO = "auto"

class DataType(str, Enum):
    """Data types for columns."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class ProcessingStrategy(str, Enum):
    """Data processing strategies."""
    MEMORY_EFFICIENT = "memory_efficient"
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_FOCUSED = "quality_focused"
    BALANCED = "balanced"

@dataclass
class DataLoadingConfig:
    """Configuration for data loading operations."""
    
    # File reading settings
    max_file_size_mb: int = 500
    chunk_size: int = 10000
    encoding: str = "utf-8"
    low_memory: bool = True
    
    # CSV-specific settings
    csv_separator: Optional[str] = None  # Auto-detect if None
    csv_decimal: str = "."
    csv_thousands: Optional[str] = None
    csv_skip_blank_lines: bool = True
    csv_comment: Optional[str] = None
    
    # Excel-specific settings
    excel_engine: Optional[str] = None  # Auto-select
    excel_sheet_name: Union[str, int, List, None] = 0  # First sheet by default
    
    # JSON-specific settings
    json_orient: str = "records"  # records, index, values, split, table
    json_lines: bool = False
    
    # Data type inference
    infer_datetime_format: bool = True
    parse_dates: bool = True
    date_format: Optional[str] = None
    
    # Memory management
    use_chunked_loading: bool = False
    auto_enable_chunking: bool = True
    memory_threshold_mb: int = 100
    
    # Data quality
    max_missing_ratio: float = 0.95  # Drop columns with >95% missing
    drop_duplicates: bool = True
    handle_mixed_types: bool = True

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing operations."""
    
    # Missing value handling
    numeric_missing_strategy: Literal["mean", "median", "mode", "drop", "knn", "interpolate"] = "median"
    categorical_missing_strategy: Literal["mode", "constant", "drop"] = "mode"
    missing_indicator: bool = False
    
    # Outlier handling
    detect_outliers: bool = True
    outlier_method: Literal["iqr", "zscore", "isolation_forest"] = "iqr"
    outlier_threshold: float = 3.0
    handle_outliers: Literal["remove", "cap", "transform", "ignore"] = "cap"
    
    # Data transformation
    normalize_numeric: bool = False
    encode_categorical: bool = True
    categorical_encoding: Literal["label", "onehot", "target", "frequency"] = "label"
    handle_high_cardinality: bool = True
    max_cardinality: int = 50
    
    # Feature engineering
    create_datetime_features: bool = True
    create_interaction_features: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Data validation
    validate_data_types: bool = True
    validate_ranges: bool = True
    validate_uniqueness: bool = True
    
    # Performance
    processing_strategy: ProcessingStrategy = ProcessingStrategy.BALANCED
    parallel_processing: bool = True
    n_jobs: int = -1

@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    
    file_path: str
    file_format: FileFormat
    file_size_bytes: int
    n_rows: int
    n_columns: int
    column_names: List[str]
    column_types: Dict[str, DataType]
    memory_usage_mb: float
    encoding: str
    has_header: bool
    separator: Optional[str] = None
    
    # Data quality metrics
    missing_value_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    data_quality_score: float = 1.0
    
    # Processing metadata
    load_time: float = 0.0
    preprocessing_time: float = 0.0
    chunks_processed: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class DataProcessingResult:
    """Result of data processing operations."""
    
    dataframe: pd.DataFrame
    info: DatasetInfo
    transformations_applied: List[str]
    preprocessing_config: PreprocessingConfig
    quality_report: Dict[str, Any]
    feature_mapping: Dict[str, str] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    created_features: List[str] = field(default_factory=list)

class DataService:
    """
    Comprehensive data service for dataset management and preprocessing.
    
    This service handles all aspects of dataset loading, cleaning, and preparation
    for ML pipelines in the Auto-Analyst platform.
    """
    
    def __init__(
        self,
        loading_config: Optional[DataLoadingConfig] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the data service.
        
        Args:
            loading_config: Configuration for data loading operations
            preprocessing_config: Configuration for preprocessing operations
            cache_dir: Directory for caching processed datasets
        """
        self.loading_config = loading_config or DataLoadingConfig()
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "auto_analyst_cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_stats = {
            'files_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("DataService initialized successfully")
    
    async def load_dataset(
        self,
        file_path: Union[str, Path, io.IOBase],
        file_format: FileFormat = FileFormat.AUTO,
        **kwargs
    ) -> DataProcessingResult:
        """
        Load a dataset from file with intelligent format detection and processing.
        
        Args:
            file_path: Path to the dataset file or file-like object
            file_format: File format (auto-detected if AUTO)
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            DataProcessingResult with loaded dataset and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or data is invalid
            MemoryError: If file is too large for memory
        """
        try:
            start_time = time.time()
            
            # Check cache first
            if isinstance(file_path, (str, Path)):
                cache_key = await self._generate_cache_key(file_path)
                cached_result = await self._load_from_cache(cache_key)
                if cached_result:
                    self.performance_stats['cache_hits'] += 1
                    return cached_result
                self.performance_stats['cache_misses'] += 1
            
            # Validate file
            file_info = await self._validate_file(file_path)
            
            # Detect format if needed
            if file_format == FileFormat.AUTO:
                file_format = await self._detect_file_format(file_path)
            
            # Load dataset based on format
            if file_format == FileFormat.CSV:
                df, info = await self._load_csv(file_path, file_info, **kwargs)
            elif file_format == FileFormat.EXCEL:
                df, info = await self._load_excel(file_path, file_info, **kwargs)
            elif file_format == FileFormat.JSON:
                df, info = await self._load_json(file_path, file_info, **kwargs)
            elif file_format == FileFormat.PARQUET:
                df, info = await self._load_parquet(file_path, file_info, **kwargs)
            elif file_format == FileFormat.TSV:
                df, info = await self._load_csv(file_path, file_info, sep='\t', **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Perform initial data analysis
            info = await self._analyze_dataset(df, info)
            
            # Create processing result
            result = DataProcessingResult(
                dataframe=df,
                info=info,
                transformations_applied=["initial_load"],
                preprocessing_config=self.preprocessing_config,
                quality_report=await self._generate_quality_report(df, info)
            )
            
            info.load_time = time.time() - start_time
            self._update_performance_stats(info.load_time)
            
            # Cache result if applicable
            if isinstance(file_path, (str, Path)):
                await self._save_to_cache(cache_key, result)
            
            logger.info(f"Dataset loaded successfully: {info.n_rows} rows, {info.n_columns} columns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    async def preprocess_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        config: Optional[PreprocessingConfig] = None
    ) -> DataProcessingResult:
        """
        Comprehensive dataset preprocessing for ML readiness.
        
        Args:
            df: Input dataframe
            target_column: Target column for supervised learning (optional)
            config: Custom preprocessing configuration
            
        Returns:
            DataProcessingResult with preprocessed dataset
        """
        try:
            start_time = time.time()
            config = config or self.preprocessing_config
            
            # Create a copy to avoid modifying original
            df_processed = df.copy()
            transformations = []
            dropped_columns = []
            created_features = []
            
            logger.info(f"Starting preprocessing: {len(df_processed)} rows, {len(df_processed.columns)} columns")
            
            # Step 1: Data Quality Assessment
            quality_before = await self._assess_data_quality(df_processed)
            
            # Step 2: Handle Missing Values
            if self._has_missing_values(df_processed):
                df_processed, dropped_cols = await self._handle_missing_values(
                    df_processed, config, target_column
                )
                transformations.append("missing_values_handled")
                dropped_columns.extend(dropped_cols)
            
            # Step 3: Remove Duplicates
            if config.preprocessing_config.drop_duplicates:
                initial_rows = len(df_processed)
                df_processed = df_processed.drop_duplicates()
                if len(df_processed) < initial_rows:
                    transformations.append("duplicates_removed")
                    logger.info(f"Removed {initial_rows - len(df_processed)} duplicate rows")
            
            # Step 4: Data Type Conversion and Validation
            df_processed, type_changes = await self._optimize_data_types(df_processed)
            if type_changes:
                transformations.append("data_types_optimized")
            
            # Step 5: Outlier Detection and Handling
            if config.detect_outliers:
                df_processed, outlier_info = await self._handle_outliers(
                    df_processed, config, target_column
                )
                if outlier_info['outliers_found'] > 0:
                    transformations.append("outliers_handled")
            
            # Step 6: Feature Engineering
            if config.create_datetime_features or config.create_interaction_features:
                df_processed, new_features = await self._engineer_features(df_processed, config)
                transformations.extend(new_features)
                created_features.extend(new_features)
            
            # Step 7: Categorical Encoding
            if config.encode_categorical:
                df_processed, encoding_info = await self._encode_categorical_features(
                    df_processed, config, target_column
                )
                if encoding_info['encoded_columns']:
                    transformations.append("categorical_encoded")
            
            # Step 8: Numeric Scaling/Normalization
            if config.normalize_numeric:
                df_processed, scaling_info = await self._normalize_numeric_features(df_processed)
                if scaling_info['scaled_columns']:
                    transformations.append("numeric_normalized")
            
            # Step 9: Final Data Validation
            validation_result = await self._validate_processed_data(df_processed, config)
            if not validation_result['valid']:
                logger.warning(f"Data validation issues: {validation_result['issues']}")
            
            # Step 10: Memory Optimization
            df_processed = await self._optimize_memory_usage(df_processed)
            transformations.append("memory_optimized")
            
            # Generate final dataset info
            info = DatasetInfo(
                file_path="processed_dataset",
                file_format=FileFormat.CSV,  # Processed data is always tabular
                file_size_bytes=df_processed.memory_usage(deep=True).sum(),
                n_rows=len(df_processed),
                n_columns=len(df_processed.columns),
                column_names=list(df_processed.columns),
                column_types=await self._infer_column_types(df_processed),
                memory_usage_mb=df_processed.memory_usage(deep=True).sum() / (1024 * 1024),
                encoding="utf-8",
                has_header=True,
                preprocessing_time=time.time() - start_time
            )
            
            # Generate quality report
            quality_after = await self._assess_data_quality(df_processed)
            quality_report = {
                'quality_before': quality_before,
                'quality_after': quality_after,
                'improvement': quality_after['overall_score'] - quality_before['overall_score'],
                'transformations_applied': transformations,
                'validation_result': validation_result
            }
            
            result = DataProcessingResult(
                dataframe=df_processed,
                info=info,
                transformations_applied=transformations,
                preprocessing_config=config,
                quality_report=quality_report,
                dropped_columns=dropped_columns,
                created_features=created_features
            )
            
            logger.info(f"Preprocessing completed: {len(transformations)} transformations applied")
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    async def prepare_for_ml(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Prepare dataset specifically for ML pipeline consumption.
        
        Args:
            df: Preprocessed dataframe
            target_column: Target column name
            task_type: ML task type ('classification', 'regression', etc.)
            test_size: Proportion of data to reserve for testing
            
        Returns:
            Dictionary with ML-ready data splits and metadata
        """
        try:
            logger.info("Preparing dataset for ML pipeline")
            
            # Validate inputs
            if target_column and target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Separate features and target
            if target_column:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                # Infer task type if not provided
                if not task_type:
                    task_type = await self._infer_task_type(y)
            else:
                X = df.copy()
                y = None
                task_type = task_type or 'unsupervised'
            
            # Ensure all features are numeric for ML
            X_processed, feature_info = await self._ensure_numeric_features(X)
            
            # Handle any remaining missing values
            if X_processed.isnull().any().any():
                imputer = SimpleImputer(strategy='median')
                numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    X_processed[numeric_cols] = imputer.fit_transform(X_processed[numeric_cols])
            
            # Create train/test split if requested
            splits = {}
            if test_size > 0 and len(X_processed) > 10:
                from sklearn.model_selection import train_test_split
                
                if y is not None:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y, test_size=test_size, random_state=42,
                        stratify=y if task_type == 'classification' and len(y.unique()) < len(y) * 0.5 else None
                    )
                    splits = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test
                    }
                else:
                    X_train, X_test = train_test_split(X_processed, test_size=test_size, random_state=42)
                    splits = {
                        'X_train': X_train,
                        'X_test': X_test
                    }
            else:
                splits = {
                    'X_train': X_processed,
                    'y_train': y
                }
            
            # Generate feature names and types for ML pipeline
            feature_names = list(X_processed.columns)
            feature_types = {}
            for col in feature_names:
                if X_processed[col].dtype in ['int64', 'float64']:
                    feature_types[col] = 'numeric'
                else:
                    feature_types[col] = 'categorical'
            
            ml_ready_data = {
                'splits': splits,
                'feature_names': feature_names,
                'feature_types': feature_types,
                'target_column': target_column,
                'task_type': task_type,
                'n_samples': len(X_processed),
                'n_features': len(feature_names),
                'preprocessing_info': feature_info,
                'dataset_info': {
                    'memory_usage_mb': X_processed.memory_usage(deep=True).sum() / (1024 * 1024),
                    'data_types': dict(X_processed.dtypes.astype(str)),
                    'missing_values': X_processed.isnull().sum().to_dict(),
                    'shape': X_processed.shape
                }
            }
            
            logger.info(f"ML preparation complete: {task_type} task with {len(feature_names)} features")
            return ml_ready_data
            
        except Exception as e:
            logger.error(f"ML preparation failed: {str(e)}")
            raise
    
    async def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis and insights generation.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with detailed dataset analysis
        """
        try:
            logger.info("Performing comprehensive dataset analysis")
            
            analysis = {
                'basic_info': await self._get_basic_info(df),
                'data_quality': await self._assess_data_quality(df),
                'statistical_summary': await self._get_statistical_summary(df),
                'column_analysis': await self._analyze_columns(df),
                'correlation_analysis': await self._analyze_correlations(df),
                'missing_value_analysis': await self._analyze_missing_values(df),
                'outlier_analysis': await self._analyze_outliers(df),
                'distribution_analysis': await self._analyze_distributions(df),
                'recommendations': await self._generate_recommendations(df)
            }
            
            logger.info("Dataset analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {str(e)}")
            raise
    
    # Private helper methods
    
    async def _validate_file(self, file_path: Union[str, Path, io.IOBase]) -> Dict[str, Any]:
        """Validate file existence and basic properties."""
        if isinstance(file_path, io.IOBase):
            return {
                'exists': True,
                'size_bytes': 0,  # Cannot determine for file-like objects
                'is_readable': file_path.readable(),
                'path': 'file_object'
            }
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > self.loading_config.max_file_size_mb:
            if not self.loading_config.auto_enable_chunking:
                raise MemoryError(f"File too large: {size_mb:.1f}MB > {self.loading_config.max_file_size_mb}MB")
            else:
                logger.warning(f"Large file detected ({size_mb:.1f}MB), enabling chunked loading")
        
        return {
            'exists': True,
            'size_bytes': size_bytes,
            'size_mb': size_mb,
            'is_readable': os.access(file_path, os.R_OK),
            'path': str(file_path)
        }
    
    async def _detect_file_format(self, file_path: Union[str, Path, io.IOBase]) -> FileFormat:
        """Detect file format from extension or content."""
        if isinstance(file_path, io.IOBase):
            # Try to detect from content for file-like objects
            return FileFormat.CSV  # Default assumption
        
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        format_mapping = {
            '.csv': FileFormat.CSV,
            '.xlsx': FileFormat.EXCEL,
            '.xls': FileFormat.EXCEL,
            '.json': FileFormat.JSON,
            '.parquet': FileFormat.PARQUET,
            '.tsv': FileFormat.TSV,
            '.txt': FileFormat.CSV  # Assume CSV for txt files
        }
        
        detected_format = format_mapping.get(extension, FileFormat.CSV)
        logger.debug(f"Detected file format: {detected_format} for {file_path}")
        
        return detected_format
    
    async def _load_csv(
        self,
        file_path: Union[str, Path, io.IOBase],
        file_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load CSV file with intelligent parameter detection."""
        try:
            # Prepare loading parameters
            load_params = {
                'encoding': self.loading_config.encoding,
                'low_memory': self.loading_config.low_memory,
                'skip_blank_lines': self.loading_config.csv_skip_blank_lines,
                **kwargs
            }
            
            # Auto-detect separator if not provided
            if 'sep' not in load_params and not self.loading_config.csv_separator:
                separator = await self._detect_csv_separator(file_path)
                load_params['sep'] = separator
            elif self.loading_config.csv_separator:
                load_params['sep'] = self.loading_config.csv_separator
            
            # Handle large files with chunking
            if (file_info['size_mb'] > self.loading_config.memory_threshold_mb or 
                self.loading_config.use_chunked_loading):
                return await self._load_csv_chunked(file_path, file_info, load_params)
            
            # Standard loading
            df = pd.read_csv(file_path, **load_params)
            
            # Create dataset info
            info = DatasetInfo(
                file_path=file_info['path'],
                file_format=FileFormat.CSV,
                file_size_bytes=file_info['size_bytes'],
                n_rows=len(df),
                n_columns=len(df.columns),
                column_names=list(df.columns),
                column_types=await self._infer_column_types(df),
                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
                encoding=self.loading_config.encoding,
                has_header=True,
                separator=load_params.get('sep', ',')
            )
            
            return df, info
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {str(e)}")
            raise
    
    async def _load_csv_chunked(
        self,
        file_path: Union[str, Path, io.IOBase],
        file_info: Dict[str, Any],
        load_params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load large CSV files in chunks to manage memory."""
        logger.info(f"Loading large CSV file in chunks: {file_info['size_mb']:.1f}MB")
        
        chunks = []
        chunk_count = 0
        
        try:
            # Read file in chunks
            chunk_reader = pd.read_csv(
                file_path,
                chunksize=self.loading_config.chunk_size,
                **load_params
            )
            
            for chunk in chunk_reader:
                chunks.append(chunk)
                chunk_count += 1
                
                # Memory management
                if chunk_count % 10 == 0:
                    logger.debug(f"Processed {chunk_count} chunks")
                    gc.collect()
            
            # Combine chunks
            logger.info(f"Combining {chunk_count} chunks into single dataframe")
            df = pd.concat(chunks, ignore_index=True)
            
            # Clean up
            del chunks
            gc.collect()
            
            info = DatasetInfo(
                file_path=file_info['path'],
                file_format=FileFormat.CSV,
                file_size_bytes=file_info['size_bytes'],
                n_rows=len(df),
                n_columns=len(df.columns),
                column_names=list(df.columns),
                column_types=await self._infer_column_types(df),
                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
                encoding=self.loading_config.encoding,
                has_header=True,
                separator=load_params.get('sep', ','),
                chunks_processed=chunk_count
            )
            
            return df, info
            
        except Exception as e:
            logger.error(f"Chunked CSV loading failed: {str(e)}")
            raise
    
    async def _load_excel(
        self,
        file_path: Union[str, Path, io.IOBase],
        file_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load Excel file with sheet detection."""
        if not EXCEL_SUPPORT:
            raise ImportError("Excel support not available. Install openpyxl: pip install openpyxl")
        
        try:
            load_params = {
                'engine': self.loading_config.excel_engine,
                'sheet_name': self.loading_config.excel_sheet_name,
                **kwargs
            }
            
            # Remove None values
            load_params = {k: v for k, v in load_params.items() if v is not None}
            
            df = pd.read_excel(file_path, **load_params)
            
            info = DatasetInfo(
                file_path=file_info['path'],
                file_format=FileFormat.EXCEL,
                file_size_bytes=file_info['size_bytes'],
                n_rows=len(df),
                n_columns=len(df.columns),
                column_names=list(df.columns),
                column_types=await self._infer_column_types(df),
                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
                encoding="utf-8",
                has_header=True
            )
            
            return df, info
            
        except Exception as e:
            logger.error(f"Failed to load Excel file: {str(e)}")
            raise
    
    async def _load_json(
        self,
        file_path: Union[str, Path, io.IOBase],
        file_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load JSON file with format detection."""
        try:
            load_params = {
                'orient': self.loading_config.json_orient,
                'lines': self.loading_config.json_lines,
                **kwargs
            }
            
            # Try different JSON formats
            try:
                df = pd.read_json(file_path, **load_params)
            except ValueError:
                # Try with lines=True for JSON Lines format
                if not load_params.get('lines', False):
                    load_params['lines'] = True
                    df = pd.read_json(file_path, **load_params)
                else:
                    raise
            
            info = DatasetInfo(
                file_path=file_info['path'],
                file_format=FileFormat.JSON,
                file_size_bytes=file_info['size_bytes'],
                n_rows=len(df),
                n_columns=len(df.columns),
                column_names=list(df.columns),
                column_types=await self._infer_column_types(df),
                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
                encoding="utf-8",
                has_header=True
            )
            
            return df, info
            
        except Exception as e:
            logger.error(f"Failed to load JSON file: {str(e)}")
            raise
    
    async def _load_parquet(
        self,
        file_path: Union[str, Path, io.IOBase],
        file_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load Parquet file."""
        if not PARQUET_SUPPORT and not FASTPARQUET_SUPPORT:
            raise ImportError("Parquet support not available. Install pyarrow or fastparquet")
        
        try:
            engine = 'pyarrow' if PARQUET_SUPPORT else 'fastparquet'
            df = pd.read_parquet(file_path, engine=engine, **kwargs)
            
            info = DatasetInfo(
                file_path=file_info['path'],
                file_format=FileFormat.PARQUET,
                file_size_bytes=file_info['size_bytes'],
                n_rows=len(df),
                n_columns=len(df.columns),
                column_names=list(df.columns),
                column_types=await self._infer_column_types(df),
                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
                encoding="utf-8",
                has_header=True
            )
            
            return df, info
            
        except Exception as e:
            logger.error(f"Failed to load Parquet file: {str(e)}")
            raise
    
    async def _detect_csv_separator(self, file_path: Union[str, Path, io.IOBase]) -> str:
        """Detect CSV separator by analyzing the first few lines."""
        if isinstance(file_path, io.IOBase):
            # Reset position and read sample
            file_path.seek(0)
            sample = file_path.read(1024)
            file_path.seek(0)
        else:
            with open(file_path, 'r', encoding=self.loading_config.encoding) as f:
                sample = f.read(1024)
        
        # Count occurrences of common separators
        separators = [',', ';', '\t', '|']
        separator_counts = {}
        
        for sep in separators:
            separator_counts[sep] = sample.count(sep)
        
        # Return the most common separator
        detected_sep = max(separator_counts, key=separator_counts.get)
        
        # Validation: ensure it appears consistently
        lines = sample.split('\n')[:5]  # Check first 5 lines
        if len(lines) > 1:
            counts = [line.count(detected_sep) for line in lines if line.strip()]
            if len(set(counts)) > 2:  # Too much variation
                detected_sep = ','  # Fallback to comma
        
        logger.debug(f"Detected CSV separator: '{detected_sep}'")
        return detected_sep
    
    async def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, DataType]:
        """Infer data types for each column."""
        column_types = {}
        
        for column in df.columns:
            col_data = df[column]
            
            # Skip if all null
            if col_data.isnull().all():
                column_types[column] = DataType.UNKNOWN
                continue
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                column_types[column] = DataType.DATETIME
            # Check for boolean
            elif pd.api.types.is_bool_dtype(col_data):
                column_types[column] = DataType.BOOLEAN
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(col_data):
                column_types[column] = DataType.NUMERIC
            # Check for string/object
            elif pd.api.types.is_object_dtype(col_data):
                # Further analyze string columns
                non_null_data = col_data.dropna()
                if len(non_null_data) == 0:
                    column_types[column] = DataType.UNKNOWN
                    continue
                
                # Check if it's categorical (low cardinality relative to size)
                unique_ratio = non_null_data.nunique() / len(non_null_data)
                avg_length = non_null_data.astype(str).str.len().mean()
                
                if unique_ratio < 0.1 and avg_length < 50:
                    column_types[column] = DataType.CATEGORICAL
                elif avg_length > 100:
                    column_types[column] = DataType.TEXT
                else:
                    # Try to detect if it's actually numeric/datetime in string format
                    try:
                        pd.to_numeric(non_null_data.iloc[:100])
                        column_types[column] = DataType.NUMERIC
                    except (ValueError, TypeError):
                        try:
                            pd.to_datetime(non_null_data.iloc[:100])
                            column_types[column] = DataType.DATETIME
                        except (ValueError, TypeError):
                            column_types[column] = DataType.CATEGORICAL
            else:
                column_types[column] = DataType.MIXED
        
        return column_types
    
    async def _analyze_dataset(self, df: pd.DataFrame, info: DatasetInfo) -> DatasetInfo:
        """Perform initial dataset analysis and update info."""
        # Calculate data quality metrics
        total_cells = info.n_rows * info.n_columns
        missing_cells = df.isnull().sum().sum()
        info.missing_value_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        # Calculate duplicate ratio
        duplicate_rows = df.duplicated().sum()
        info.duplicate_ratio = duplicate_rows / info.n_rows if info.n_rows > 0 else 0
        
        # Calculate overall data quality score
        quality_factors = [
            1 - info.missing_value_ratio,  # Missing value penalty
            1 - min(info.duplicate_ratio * 2, 1),  # Duplicate penalty
            min(info.n_rows / 1000, 1),  # Size adequacy
            min(info.n_columns / 10, 1)   # Feature adequacy
        ]
        info.data_quality_score = sum(quality_factors) / len(quality_factors)
        
        return info
    
    async def _generate_quality_report(self, df: pd.DataFrame, info: DatasetInfo) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        return {
            'overall_score': info.data_quality_score,
            'missing_value_ratio': info.missing_value_ratio,
            'duplicate_ratio': info.duplicate_ratio,
            'column_quality': {
                col: {
                    'missing_ratio': df[col].isnull().sum() / len(df),
                    'unique_ratio': df[col].nunique() / len(df),
                    'data_type': info.column_types.get(col, DataType.UNKNOWN).value
                }
                for col in df.columns
            },
            'recommendations': await self._generate_quality_recommendations(df, info)
        }
    
    async def _generate_quality_recommendations(self, df: pd.DataFrame, info: DatasetInfo) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        if info.missing_value_ratio > 0.1:
            recommendations.append("High missing value ratio detected - consider imputation or feature removal")
        
        if info.duplicate_ratio > 0.05:
            recommendations.append("Significant duplicates detected - consider removing duplicate rows")
        
        if info.n_rows < 100:
            recommendations.append("Small dataset - results may have high variance")
        
        if info.n_columns > info.n_rows:
            recommendations.append("More features than samples - consider dimensionality reduction")
        
        # Check for columns with single values
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_columns:
            recommendations.append(f"Remove constant columns: {constant_columns}")
        
        # Check for high cardinality categorical columns
        high_cardinality_cols = []
        for col, dtype in info.column_types.items():
            if dtype == DataType.CATEGORICAL and df[col].nunique() > 50:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            recommendations.append(f"High cardinality categorical columns may need special encoding: {high_cardinality_cols}")
        
        return recommendations
    
    async def _handle_missing_values(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values based on configuration."""
        df_processed = df.copy()
        dropped_columns = []
        
        # First, drop columns with too many missing values
        for col in df_processed.columns:
            if col != target_column:  # Don't drop target column
                missing_ratio = df_processed[col].isnull().sum() / len(df_processed)
                if missing_ratio > self.loading_config.max_missing_ratio:
                    df_processed = df_processed.drop(columns=[col])
                    dropped_columns.append(col)
                    logger.info(f"Dropped column '{col}' with {missing_ratio:.1%} missing values")
        
        # Handle remaining missing values
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                col_type = await self._get_column_type(df_processed[col])
                
                if col_type == DataType.NUMERIC:
                    if config.numeric_missing_strategy == "mean":
                        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    elif config.numeric_missing_strategy == "median":
                        df_processed[col].fillna(df_processed[col].median(), inplace=True)
                    elif config.numeric_missing_strategy == "mode":
                        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    elif config.numeric_missing_strategy == "interpolate":
                        df_processed[col].interpolate(inplace=True)
                    elif config.numeric_missing_strategy == "knn":
                        # Use KNN imputation for numeric columns
                        imputer = KNNImputer(n_neighbors=5)
                        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
                
                elif col_type in [DataType.CATEGORICAL, DataType.TEXT]:
                    if config.categorical_missing_strategy == "mode":
                        mode_value = df_processed[col].mode()
                        if len(mode_value) > 0:
                            df_processed[col].fillna(mode_value[0], inplace=True)
                        else:
                            df_processed[col].fillna("Unknown", inplace=True)
                    elif config.categorical_missing_strategy == "constant":
                        df_processed[col].fillna("Missing", inplace=True)
                
                elif col_type == DataType.DATETIME:
                    # Use forward fill for datetime columns
                    df_processed[col].fillna(method='ffill', inplace=True)
        
        return df_processed, dropped_columns
    
    async def _handle_outliers(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle outliers in numeric columns."""
        df_processed = df.copy()
        outlier_info = {'outliers_found': 0, 'columns_processed': []}
        
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)  # Don't modify target
        
        for col in numeric_columns:
            col_data = df_processed[col].dropna()
            
            if len(col_data) < 10:  # Skip if too few values
                continue
            
            outliers_mask = None
            
            if config.outlier_method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                
            elif config.outlier_method == "zscore":
                z_scores = np.abs(stats.zscore(col_data))
                outliers_mask = pd.Series(False, index=df_processed.index)
                outliers_mask[col_data.index] = z_scores > config.outlier_threshold
                
            elif config.outlier_method == "isolation_forest":
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                    outliers_mask = pd.Series(False, index=df_processed.index)
                    outliers_mask[col_data.index] = outliers == -1
                except ImportError:
                    logger.warning("IsolationForest not available, using IQR method")
                    continue
            
            if outliers_mask is not None and outliers_mask.sum() > 0:
                outlier_count = outliers_mask.sum()
                outlier_info['outliers_found'] += outlier_count
                outlier_info['columns_processed'].append(col)
                
                if config.handle_outliers == "remove":
                    df_processed = df_processed[~outliers_mask]
                elif config.handle_outliers == "cap":
                    # Cap at percentiles
                    lower_cap = col_data.quantile(0.01)
                    upper_cap = col_data.quantile(0.99)
                    df_processed.loc[df_processed[col] < lower_cap, col] = lower_cap
                    df_processed.loc[df_processed[col] > upper_cap, col] = upper_cap
                elif config.handle_outliers == "transform":
                    # Log transformation for positive skewed data
                    if (col_data > 0).all():
                        df_processed[col] = np.log1p(df_processed[col])
                
                logger.info(f"Handled {outlier_count} outliers in column '{col}'")
        
        return df_processed, outlier_info
    
    async def _optimize_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Optimize data types for memory efficiency."""
        df_optimized = df.copy()
        changes = []
        
        for col in df_optimized.columns:
            original_type = str(df_optimized[col].dtype)
            
            # Optimize integer columns
            if df_optimized[col].dtype == 'int64':
                if df_optimized[col].min() >= 0:
                    if df_optimized[col].max() < 255:
                        df_optimized[col] = df_optimized[col].astype('uint8')
                        changes.append(f"{col}: {original_type} -> uint8")
                    elif df_optimized[col].max() < 65535:
                        df_optimized[col] = df_optimized[col].astype('uint16')
                        changes.append(f"{col}: {original_type} -> uint16")
                    elif df_optimized[col].max() < 4294967295:
                        df_optimized[col] = df_optimized[col].astype('uint32')
                        changes.append(f"{col}: {original_type} -> uint32")
                else:
                    if df_optimized[col].min() >= -128 and df_optimized[col].max() < 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                        changes.append(f"{col}: {original_type} -> int8")
                    elif df_optimized[col].min() >= -32768 and df_optimized[col].max() < 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                        changes.append(f"{col}: {original_type} -> int16")
                    elif df_optimized[col].min() >= -2147483648 and df_optimized[col].max() < 2147483647:
                        df_optimized[col] = df_optimized[col].astype('int32')
                        changes.append(f"{col}: {original_type} -> int32")
            
            # Optimize float columns
            elif df_optimized[col].dtype == 'float64':
                if pd.api.types.is_integer_dtype(df_optimized[col].dropna()):
                    # Convert float to int if no decimal values
                    df_optimized[col] = df_optimized[col].astype('int64')
                    changes.append(f"{col}: {original_type} -> int64")
                else:
                    # Check if float32 is sufficient
                    try:
                        float32_col = df_optimized[col].astype('float32')
                        if np.allclose(df_optimized[col].dropna(), float32_col.dropna(), equal_nan=True):
                            df_optimized[col] = float32_col
                            changes.append(f"{col}: {original_type} -> float32")
                    except:
                        pass
            
            # Optimize object columns
            elif df_optimized[col].dtype == 'object':
                # Try to convert to category if low cardinality
                nunique = df_optimized[col].nunique()
                total_count = df_optimized[col].count()
                
                if nunique / total_count < 0.5 and nunique < 100:
                    df_optimized[col] = df_optimized[col].astype('category')
                    changes.append(f"{col}: {original_type} -> category")
        
        if changes:
            logger.info(f"Optimized data types for {len(changes)} columns")
        
        return df_optimized, changes
    
    async def _engineer_features(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create engineered features based on configuration."""
        df_engineered = df.copy()
        new_features = []
        
        # Datetime feature engineering
        if config.create_datetime_features:
            datetime_cols = df_engineered.select_dtypes(include=['datetime64']).columns
            
            for col in datetime_cols:
                dt_series = pd.to_datetime(df_engineered[col])
                
                # Extract datetime components
                df_engineered[f"{col}_year"] = dt_series.dt.year
                df_engineered[f"{col}_month"] = dt_series.dt.month
                df_engineered[f"{col}_day"] = dt_series.dt.day
                df_engineered[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                df_engineered[f"{col}_hour"] = dt_series.dt.hour
                df_engineered[f"{col}_quarter"] = dt_series.dt.quarter
                
                new_features.extend([
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_dayofweek", f"{col}_hour", f"{col}_quarter"
                ])
        
        # Interaction features (for numeric columns)
        if config.create_interaction_features:
            numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Create pairwise interactions (limited to avoid explosion)
                from itertools import combinations
                
                pairs = list(combinations(numeric_cols[:5], 2))  # Limit to top 5 columns
                
                for col1, col2 in pairs[:10]:  # Limit to 10 interactions
                    interaction_name = f"{col1}_x_{col2}"
                    df_engineered[interaction_name] = df_engineered[col1] * df_engineered[col2]
                    new_features.append(interaction_name)
        
        # Polynomial features
        if config.polynomial_features and config.polynomial_degree > 1:
            try:
                from sklearn.preprocessing import PolynomialFeatures
                
                numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and len(numeric_cols) <= 5:  # Limit to prevent explosion
                    poly = PolynomialFeatures(
                        degree=config.polynomial_degree,
                        interaction_only=True,
                        include_bias=False
                    )
                    
                    poly_features = poly.fit_transform(df_engineered[numeric_cols])
                    feature_names = poly.get_feature_names_out(numeric_cols)
                    
                    # Add only new features (not original ones)
                    for i, name in enumerate(feature_names):
                        if name not in numeric_cols:
                            df_engineered[f"poly_{name}"] = poly_features[:, i]
                            new_features.append(f"poly_{name}")
                
            except ImportError:
                logger.warning("PolynomialFeatures not available")
        
        if new_features:
            logger.info(f"Created {len(new_features)} engineered features")
        
        return df_engineered, new_features
    
    async def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical features based on configuration."""
        df_encoded = df.copy()
        encoding_info = {'encoded_columns': [], 'encoding_method': config.categorical_encoding}
        
        categorical_columns = []
        for col in df_encoded.columns:
            if col != target_column and (
                df_encoded[col].dtype == 'object' or 
                df_encoded[col].dtype.name == 'category' or
                (df_encoded[col].dtype in ['int64', 'float64'] and df_encoded[col].nunique() < 10)
            ):
                categorical_columns.append(col)
        
        for col in categorical_columns:
            nunique = df_encoded[col].nunique()
            
            # Handle high cardinality columns
            if nunique > config.max_cardinality and config.handle_high_cardinality:
                if config.categorical_encoding == "frequency":
                    # Frequency encoding for high cardinality
                    freq_map = df_encoded[col].value_counts().to_dict()
                    df_encoded[f"{col}_freq"] = df_encoded[col].map(freq_map)
                    df_encoded = df_encoded.drop(columns=[col])
                    encoding_info['encoded_columns'].append(col)
                    continue
                else:
                    # Keep only top categories
                    top_categories = df_encoded[col].value_counts().head(config.max_cardinality).index
                    df_encoded[col] = df_encoded[col].where(
                        df_encoded[col].isin(top_categories), 'Other'
                    )
            
            # Apply encoding strategy
            if config.categorical_encoding == "label":
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoding_info['encoded_columns'].append(col)
                
            elif config.categorical_encoding == "onehot":
                if nunique <= 10:  # Limit one-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                    encoding_info['encoded_columns'].append(col)
                else:
                    # Fallback to label encoding
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    encoding_info['encoded_columns'].append(col)
            
            elif config.categorical_encoding == "target" and target_column:
                # Target encoding (mean encoding)
                try:
                    target_mean = df_encoded.groupby(col)[target_column].mean()
                    df_encoded[f"{col}_target"] = df_encoded[col].map(target_mean)
                    df_encoded = df_encoded.drop(columns=[col])
                    encoding_info['encoded_columns'].append(col)
                except:
                    # Fallback to label encoding
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    encoding_info['encoded_columns'].append(col)
            
            elif config.categorical_encoding == "frequency":
                freq_map = df_encoded[col].value_counts().to_dict()
                df_encoded[f"{col}_freq"] = df_encoded[col].map(freq_map)
                df_encoded = df_encoded.drop(columns=[col])
                encoding_info['encoded_columns'].append(col)
        
        if encoding_info['encoded_columns']:
            logger.info(f"Encoded {len(encoding_info['encoded_columns'])} categorical columns using {config.categorical_encoding}")
        
        return df_encoded, encoding_info
    
    async def _normalize_numeric_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize numeric features."""
        df_normalized = df.copy()
        scaling_info = {'scaled_columns': [], 'scaler_type': 'StandardScaler'}
        
        numeric_columns = df_normalized.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            df_normalized[numeric_columns] = scaler.fit_transform(df_normalized[numeric_columns])
            scaling_info['scaled_columns'] = list(numeric_columns)
            
            logger.info(f"Normalized {len(numeric_columns)} numeric columns")
        
        return df_normalized, scaling_info
    
    async def _ensure_numeric_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Ensure all features are numeric for ML consumption."""
        df_numeric = df.copy()
        conversion_info = {'converted_columns': [], 'dropped_columns': []}
        
        for col in df_numeric.columns:
            if not pd.api.types.is_numeric_dtype(df_numeric[col]):
                # Try to convert to numeric
                try:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                    conversion_info['converted_columns'].append(col)
                except:
                    # If conversion fails and it's categorical, encode it
                    if df_numeric[col].nunique() < 100:  # Reasonable cardinality
                        le = LabelEncoder()
                        df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
                        conversion_info['converted_columns'].append(col)
                    else:
                        # Drop high cardinality non-numeric columns
                        df_numeric = df_numeric.drop(columns=[col])
                        conversion_info['dropped_columns'].append(col)
        
        # Handle any remaining missing values after conversion
        if df_numeric.isnull().any().any():
            imputer = SimpleImputer(strategy='median')
            df_numeric = pd.DataFrame(
                imputer.fit_transform(df_numeric),
                columns=df_numeric.columns,
                index=df_numeric.index
            )
        
        return df_numeric, conversion_info
    
    # Additional helper methods for analysis and utilities
    
    async def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtypes': dict(df.dtypes.astype(str)),
            'null_counts': dict(df.isnull().sum()),
            'duplicate_rows': df.duplicated().sum()
        }
    
    async def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        quality_metrics = {
            'overall_score': 0.0,
            'missing_value_ratio': missing_cells / total_cells if total_cells > 0 else 0,
            'duplicate_ratio': df.duplicated().sum() / len(df) if len(df) > 0 else 0,
            'constant_columns': len([col for col in df.columns if df[col].nunique() <= 1]),
            'high_cardinality_columns': len([col for col in df.columns if df[col].nunique() > len(df) * 0.5]),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Calculate overall quality score
        quality_factors = [
            1 - quality_metrics['missing_value_ratio'],
            1 - min(quality_metrics['duplicate_ratio'] * 2, 1),
            1 - min(quality_metrics['constant_columns'] / max(1, df.shape[1]), 1),
            min(df.shape[0] / 1000, 1)  # Sample size adequacy
        ]
        
        quality_metrics['overall_score'] = sum(quality_factors) / len(quality_factors)
        
        return quality_metrics
    
    async def _infer_task_type(self, target: pd.Series) -> str:
        """Infer ML task type from target variable."""
        if pd.api.types.is_numeric_dtype(target):
            # Check if it's actually classification (few unique values)
            unique_ratio = target.nunique() / len(target)
            if unique_ratio < 0.05 and target.nunique() < 20:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    
    # Cache management methods
    
    async def _generate_cache_key(self, file_path: Union[str, Path]) -> str:
        """Generate cache key for file."""
        file_path = Path(file_path)
        stat = file_path.stat()
        
        # Create hash from file path, size, and modification time
        key_string = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _load_from_cache(self, cache_key: str) -> Optional[DataProcessingResult]:
        """Load cached processing result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                import pickle
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                logger.debug(f"Loaded from cache: {cache_key}")
                return result
        except Exception as e:
            logger.debug(f"Cache load failed: {str(e)}")
        
        return None
    
    async def _save_to_cache(self, cache_key: str, result: DataProcessingResult) -> None:
        """Save processing result to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.debug(f"Cache save failed: {str(e)}")
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics."""
        self.performance_stats['files_processed'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        self.performance_stats['average_processing_time'] = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['files_processed']
        )
    
    async def _has_missing_values(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has missing values."""
        return df.isnull().any().any()
    
    async def _get_column_type(self, series: pd.Series) -> DataType:
        """Get the data type of a single column."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME
        elif pd.api.types.is_bool_dtype(series):
            return DataType.BOOLEAN
        elif pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERIC
        elif pd.api.types.is_object_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:
                return DataType.CATEGORICAL
            else:
                return DataType.TEXT
        else:
            return DataType.MIXED
    
    async def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final memory optimization."""
        # Remove any temporary columns or optimize final data types
        return df.copy()
    
    async def _validate_processed_data(self, df: pd.DataFrame, config: PreprocessingConfig) -> Dict[str, Any]:
        """Validate processed data meets requirements."""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            validation_result['issues'].append("Infinite values found in numeric columns")
            validation_result['valid'] = False
        
        # Check for remaining missing values
        if df.isnull().any().any():
            validation_result['warnings'].append("Some missing values remain after preprocessing")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            validation_result['warnings'].append(f"Constant columns found: {constant_cols}")
        
        return validation_result
    
    async def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of the dataset."""
        summary = {}
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical'] = {}
            for col in categorical_cols:
                summary['categorical'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                    'frequency': df[col].value_counts().head(5).to_dict()
                }
        
        return summary
    
    async def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual columns."""
        column_analysis = {}
        
        for col in df.columns:
            analysis = {
                'dtype': str(df[col].dtype),
                'unique_count': df[col].nunique(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                })
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    async def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        correlation_matrix = df[numeric_cols].corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations
        }
    
    async def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        missing_analysis = {
            'total_missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': {},
            'missing_patterns': {}
        }
        
        # Analyze missing values by column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_analysis['columns_with_missing'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        # Analyze missing value patterns (simplified)
        if len(missing_analysis['columns_with_missing']) > 1:
            missing_cols = list(missing_analysis['columns_with_missing'].keys())
            if len(missing_cols) <= 5:  # Limit to prevent combinatorial explosion
                missing_patterns = df[missing_cols].isnull().value_counts()
                missing_analysis['missing_patterns'] = missing_patterns.to_dict()
        
        return missing_analysis
    
    async def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            outlier_analysis[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(col_data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers.tolist()[:10]  # Limit to first 10
            }
        
        return outlier_analysis
    
    async def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_analysis = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            
            distribution_analysis[col] = {
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'normality_test': None  # Would add statistical tests here
            }
            
            # Simple normality assessment
            if abs(distribution_analysis[col]['skewness']) < 0.5:
                distribution_analysis[col]['distribution_type'] = 'approximately_normal'
            elif distribution_analysis[col]['skewness'] > 1:
                distribution_analysis[col]['distribution_type'] = 'right_skewed'
            elif distribution_analysis[col]['skewness'] < -1:
                distribution_analysis[col]['distribution_type'] = 'left_skewed'
            else:
                distribution_analysis[col]['distribution_type'] = 'moderately_skewed'
        
        return distribution_analysis
    
    async def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate data processing recommendations."""
        recommendations = []
        
        # Sample size recommendations
        if len(df) < 100:
            recommendations.append("Dataset is very small - consider collecting more data")
        elif len(df) < 1000:
            recommendations.append("Dataset is relatively small - results may have high variance")
        
        # Missing value recommendations
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.2:
            recommendations.append("High missing value ratio - consider imputation strategies")
        
        # Duplicate recommendations
        if df.duplicated().sum() > 0:
            recommendations.append("Duplicate rows detected - consider removing duplicates")
        
        # Feature recommendations
        if df.shape[1] > df.shape[0]:
            recommendations.append("More features than samples - consider dimensionality reduction")
        
        # Categorical feature recommendations
        high_cardinality_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 50:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            recommendations.append(f"High cardinality categorical columns detected: {high_cardinality_cols}")
        
        return recommendations

# Factory function for easy service creation
def create_data_service(
    loading_config: Optional[DataLoadingConfig] = None,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    cache_dir: Optional[str] = None
) -> DataService:
    """
    Factory function to create a DataService instance.
    
    Args:
        loading_config: Custom loading configuration
        preprocessing_config: Custom preprocessing configuration
        cache_dir: Custom cache directory
        
    Returns:
        Configured DataService instance
    """
    return DataService(
        loading_config=loading_config,
        preprocessing_config=preprocessing_config,
        cache_dir=cache_dir
    )

# Convenience function for dependency injection
def get_data_service() -> DataService:
    """Get DataService instance for dependency injection."""
    return create_data_service()

# Example usage
if __name__ == "__main__":
    async def example_usage():
        """Example usage of the DataService."""
        
        print("🔄 DataService Example Usage")
        print("=" * 50)
        
        # Initialize service
        data_service = create_data_service()
        
        # Example 1: Load a CSV file
        print("\n📁 Loading CSV file...")
        try:
            # Create sample data for demonstration
            import tempfile
            
            sample_data = pd.DataFrame({
                'age': [25, 35, 45, 30, 28, None, 40],
                'income': [50000, 75000, 90000, 60000, 55000, 80000, 85000],
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
                'date': pd.date_range('2023-01-01', periods=7),
                'target': [1, 0, 1, 0, 1, 0, 1]
            })
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                sample_data.to_csv(f.name, index=False)
                temp_file = f.name
            
            # Load dataset
            result = await data_service.load_dataset(temp_file)
            print(f"✅ Loaded dataset: {result.info.n_rows} rows, {result.info.n_columns} columns")
            print(f"   Data quality score: {result.info.data_quality_score:.2f}")
            
            # Clean up
            os.unlink(temp_file)
            
        except Exception as e:
            print(f"❌ Loading failed: {str(e)}")
        
        # Example 2: Preprocess dataset
        print("\n🔧 Preprocessing dataset...")
        try:
            preprocessed = await data_service.preprocess_dataset(
                result.dataframe,
                target_column='target'
            )
            print(f"✅ Preprocessing completed: {len(preprocessed.transformations_applied)} transformations applied")
            print(f"   Transformations: {preprocessed.transformations_applied}")
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
        
        # Example 3: Prepare for ML
        print("\n🤖 Preparing for ML pipeline...")
        try:
            ml_data = await data_service.prepare_for_ml(
                preprocessed.dataframe,
                target_column='target'
            )
            print(f"✅ ML preparation completed")
            print(f"   Task type: {ml_data['task_type']}")
            print(f"   Features: {ml_data['n_features']}")
            print(f"   Samples: {ml_data['n_samples']}")
            
        except Exception as e:
            print(f"❌ ML preparation failed: {str(e)}")
        
        # Example 4: Dataset analysis
        print("\n📊 Analyzing dataset...")
        try:
            analysis = await data_service.analyze_dataset(result.dataframe)
            print(f"✅ Analysis completed")
            print(f"   Data quality: {analysis['data_quality']['overall_score']:.2f}")
            print(f"   Recommendations: {len(analysis['recommendations'])} generated")
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
        
        print(f"\n🎯 DataService example completed successfully!")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except Exception as e:
        print(f"Example failed: {str(e)}")
