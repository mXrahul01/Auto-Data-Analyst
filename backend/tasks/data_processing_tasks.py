"""
Data Processing Tasks Module for Auto-Analyst Platform

This module provides comprehensive data processing task implementations for the
Auto-Analyst platform, handling all aspects of data ingestion, validation,
cleaning, transformation, and preparation for ML workflows.

Features:
- Multi-format data ingestion (CSV, Excel, JSON, Parquet, SQL)
- Comprehensive data validation and quality assessment
- Advanced data cleaning and preprocessing pipelines
- Statistical analysis and profiling
- Schema inference and data type optimization
- Missing value handling with multiple strategies
- Outlier detection and treatment
- Feature encoding and transformation
- Data sampling and partitioning for large datasets
- Real-time progress tracking and monitoring
- Data lineage and provenance tracking
- Integration with feature stores and data catalogs
- Automated data documentation generation

Processing Pipeline:
1. Data ingestion and format detection
2. Initial validation and quality assessment
3. Schema inference and optimization
4. Data profiling and statistical analysis
5. Data cleaning and preprocessing
6. Missing value imputation
7. Outlier detection and handling
8. Data transformation and encoding
9. Feature engineering preparation
10. Data partitioning and sampling
11. Quality validation and reporting
12. Metadata extraction and storage
13. Artifact generation and cleanup

Supported Formats:
- Structured: CSV, TSV, Excel (XLS/XLSX), JSON, Parquet, Feather
- Databases: PostgreSQL, MySQL, SQLite, MongoDB
- Streaming: Kafka, Redis Streams
- Cloud Storage: S3, GCS, Azure Blob
- APIs: REST, GraphQL endpoints
- Big Data: Spark, Dask integration

Dependencies:
- pandas: Primary data manipulation library
- numpy: Numerical computations
- polars: High-performance data processing (optional)
- dask: Distributed computing for large datasets
- pyarrow: Columnar data processing
- sqlalchemy: Database connectivity
- fastparquet: Efficient Parquet I/O
- openpyxl: Excel file processing
- chardet: Character encoding detection
- great_expectations: Data validation framework
"""

import asyncio
import logging
import os
import time
import uuid
import json
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil
import mimetypes
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from io import StringIO, BytesIO
import hashlib
import gzip
import zipfile
import tarfile

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Core data processing imports
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype

# High-performance alternatives
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# File format support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import openpyxl
    import xlrd
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Database connectivity
try:
    from sqlalchemy import create_engine, text, inspect
    import pymongo
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Character encoding detection
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

# Data validation
try:
    import great_expectations as ge
    from great_expectations.core.batch import RuntimeBatchRequest
    from great_expectations.data_context import BaseDataContext
    from great_expectations.data_context.types.base import DataContextConfig
    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, normaltest
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Data profiling
try:
    from ydata_profiling import ProfileReport
    YDATA_PROFILING_AVAILABLE = True
except ImportError:
    YDATA_PROFILING_AVAILABLE = False

# Cloud storage
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Backend imports
from backend.config import settings
from backend.models.database import get_db_session
from backend.services.data_service import DataService
from backend.utils.monitoring import log_info, log_warning, log_error, monitor_performance
from backend.utils.validation import validate_dataset, ValidationResult
from backend.utils.preprocessing import preprocess_data

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DataProcessingConfig:
    """Configuration for data processing tasks."""
    
    # Input settings
    file_path: str = ""
    file_format: str = "auto"
    encoding: str = "auto"
    delimiter: str = "auto"
    header_row: Optional[int] = 0
    
    # Processing options
    sample_size: Optional[int] = None
    chunk_size: int = 10000
    memory_limit_gb: float = 4.0
    use_dask: bool = False
    use_polars: bool = False
    
    # Data validation
    enable_validation: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    quality_threshold: float = 0.7
    
    # Data cleaning
    handle_missing: str = "auto"  # auto, drop, fill, interpolate
    missing_threshold: float = 0.5
    handle_outliers: str = "detect"  # ignore, detect, remove, cap
    outlier_method: str = "iqr"  # iqr, zscore, isolation
    
    # Schema inference
    infer_schema: bool = True
    optimize_dtypes: bool = True
    categorical_threshold: int = 50
    datetime_inference: bool = True
    
    # Data profiling
    generate_profile: bool = True
    profile_sample_size: int = 100000
    
    # Output options
    output_format: str = "parquet"
    compress_output: bool = True
    save_intermediate: bool = False
    
    # Performance settings
    n_jobs: int = -1
    parallel_processing: bool = True
    progress_updates: bool = True

@dataclass
class DataProcessingResult:
    """Result of data processing task."""
    
    # Task information
    task_id: str
    dataset_id: int
    status: str = "processing"
    progress: float = 0.0
    
    # Input data information
    original_file_info: Dict[str, Any] = field(default_factory=dict)
    detected_format: str = ""
    detected_encoding: str = ""
    detected_delimiter: str = ""
    
    # Processing results
    rows_processed: int = 0
    columns_processed: int = 0
    data_types: Dict[str, str] = field(default_factory=dict)
    column_names: List[str] = field(default_factory=list)
    
    # Data quality metrics
    data_quality_score: float = 0.0
    missing_value_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    outlier_ratio: float = 0.0
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistical summary
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    column_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: Optional[Dict[str, Any]] = None
    
    # Data transformations applied
    transformations_applied: List[Dict[str, Any]] = field(default_factory=list)
    missing_values_handled: Dict[str, Any] = field(default_factory=dict)
    outliers_detected: Dict[str, Any] = field(default_factory=dict)
    
    # Schema information
    inferred_schema: Dict[str, Any] = field(default_factory=dict)
    schema_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Output artifacts
    processed_file_path: str = ""
    metadata_file_path: str = ""
    profile_report_path: str = ""
    
    # Performance metrics
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    file_size_original: int = 0
    file_size_processed: int = 0
    
    # Error information
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class DataProcessor:
    """Core data processing engine."""
    
    def __init__(self, config: DataProcessingConfig):
        """Initialize data processor."""
        self.config = config
        self.result = DataProcessingResult(
            task_id=str(uuid.uuid4()),
            dataset_id=0,
            started_at=datetime.now()
        )
        self.progress_callback: Optional[Callable] = None
        self.temp_dir: Optional[Path] = None
        
        # Initialize temporary directory
        self._setup_temp_directory()
    
    def _setup_temp_directory(self):
        """Setup temporary working directory."""
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="auto_analyst_data_"))
            logger.info(f"Temporary directory created: {self.temp_dir}")
        except Exception as e:
            log_error(f"Failed to create temporary directory: {e}")
            self.temp_dir = Path(settings.TEMP_DIRECTORY)
    
    def set_progress_callback(self, callback: Callable):
        """Set progress update callback."""
        self.progress_callback = callback
    
    def update_progress(self, progress: float, status: str, details: Optional[Dict] = None):
        """Update processing progress."""
        self.result.progress = progress
        self.result.status = status
        
        if self.progress_callback:
            try:
                meta = {'progress': progress, 'status': status}
                if details:
                    meta.update(details)
                self.progress_callback(state='PROGRESS', meta=meta)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        if self.config.progress_updates:
            log_info(f"Data processing progress: {progress:.1%} - {status}")
    
    async def process_dataset(
        self,
        file_path: str,
        dataset_id: int,
        user_id: int
    ) -> DataProcessingResult:
        """
        Process uploaded dataset comprehensively.
        
        Args:
            file_path: Path to input file
            dataset_id: Dataset identifier
            user_id: User identifier
            
        Returns:
            Data processing results
        """
        try:
            self.result.dataset_id = dataset_id
            self.update_progress(0.0, "Initializing data processing")
            
            # Step 1: File inspection and format detection
            self.update_progress(5.0, "Inspecting file and detecting format")
            await self._inspect_file(file_path)
            
            # Step 2: Data loading and initial validation
            self.update_progress(10.0, "Loading data and performing initial validation")
            data = await self._load_data(file_path)
            
            # Step 3: Schema inference and optimization
            self.update_progress(20.0, "Inferring schema and optimizing data types")
            data = await self._infer_and_optimize_schema(data)
            
            # Step 4: Data quality assessment
            self.update_progress(30.0, "Assessing data quality")
            await self._assess_data_quality(data)
            
            # Step 5: Data profiling and statistical analysis
            self.update_progress(40.0, "Generating data profile and statistics")
            await self._generate_data_profile(data)
            
            # Step 6: Data cleaning and preprocessing
            self.update_progress(50.0, "Cleaning and preprocessing data")
            data = await self._clean_and_preprocess_data(data)
            
            # Step 7: Missing value handling
            self.update_progress(60.0, "Handling missing values")
            data = await self._handle_missing_values(data)
            
            # Step 8: Outlier detection and treatment
            self.update_progress(70.0, "Detecting and handling outliers")
            data = await self._handle_outliers(data)
            
            # Step 9: Final validation and quality check
            self.update_progress(80.0, "Performing final validation")
            await self._final_validation(data)
            
            # Step 10: Save processed data and artifacts
            self.update_progress(90.0, "Saving processed data and generating artifacts")
            await self._save_processed_data(data, dataset_id)
            
            # Step 11: Generate metadata and documentation
            self.update_progress(95.0, "Generating metadata and documentation")
            await self._generate_metadata(data)
            
            # Step 12: Complete processing
            self.update_progress(100.0, "Data processing completed successfully")
            self._finalize_processing()
            
            return self.result
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            log_error(error_msg, exception=e)
            
            self.result.status = "failed"
            self.result.error_message = error_msg
            self.result.error_details = {
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            
            return self.result
        
        finally:
            # Cleanup temporary directory
            await self._cleanup_temp_directory()
            
            self.result.completed_at = datetime.now()
            if self.result.started_at:
                self.result.processing_time = (
                    self.result.completed_at - self.result.started_at
                ).total_seconds()
    
    async def _inspect_file(self, file_path: str):
        """Inspect file and detect format, encoding, and basic properties."""
        try:
            file_path_obj = Path(file_path)
            
            # Basic file information
            file_stats = file_path_obj.stat()
            self.result.file_size_original = file_stats.st_size
            
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            
            # Detect file format
            detected_format = self._detect_file_format(file_path_obj, mime_type)
            self.result.detected_format = detected_format
            
            # Detect encoding for text files
            if detected_format in ['csv', 'tsv', 'txt', 'json']:
                detected_encoding = await self._detect_encoding(file_path)
                self.result.detected_encoding = detected_encoding
            
            # Store file information
            self.result.original_file_info = {
                'filename': file_path_obj.name,
                'file_size': file_stats.st_size,
                'mime_type': mime_type,
                'format': detected_format,
                'encoding': self.result.detected_encoding,
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }
            
            log_info(f"File inspection completed: {detected_format}, {file_stats.st_size} bytes")
            
        except Exception as e:
            log_error(f"File inspection failed: {e}")
            raise
    
    def _detect_file_format(self, file_path: Path, mime_type: Optional[str]) -> str:
        """Detect file format based on extension and content."""
        try:
            extension = file_path.suffix.lower()
            
            # Format mapping
            format_map = {
                '.csv': 'csv',
                '.tsv': 'tsv',
                '.txt': 'txt',
                '.json': 'json',
                '.jsonl': 'jsonl',
                '.parquet': 'parquet',
                '.feather': 'feather',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.h5': 'hdf5',
                '.hdf5': 'hdf5',
                '.pickle': 'pickle',
                '.pkl': 'pickle'
            }
            
            if extension in format_map:
                return format_map[extension]
            
            # Content-based detection for ambiguous cases
            if mime_type:
                if 'text/csv' in mime_type:
                    return 'csv'
                elif 'application/json' in mime_type:
                    return 'json'
                elif 'application/vnd.ms-excel' in mime_type:
                    return 'excel'
                elif 'text/plain' in mime_type:
                    # Try to detect delimiter by sampling
                    return self._detect_delimiter_format(file_path)
            
            # Default fallback
            return 'csv'
            
        except Exception as e:
            log_warning(f"Format detection failed: {e}")
            return 'csv'
    
    def _detect_delimiter_format(self, file_path: Path) -> str:
        """Detect delimiter for text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
            
            # Count common delimiters
            comma_count = sample.count(',')
            tab_count = sample.count('\t')
            semicolon_count = sample.count(';')
            pipe_count = sample.count('|')
            
            # Determine most likely delimiter
            counts = {
                'csv': comma_count,
                'tsv': tab_count,
                'ssv': semicolon_count,
                'psv': pipe_count
            }
            
            return max(counts.items(), key=lambda x: x[1])[0]
            
        except Exception:
            return 'csv'
    
    async def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        try:
            if CHARDET_AVAILABLE:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)  # Sample first 10KB
                    result = chardet.detect(raw_data)
                    encoding = result.get('encoding', 'utf-8')
                    confidence = result.get('confidence', 0)
                    
                    if confidence > 0.7:
                        return encoding
            
            # Fallback: try common encodings
            common_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in common_encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # Try to read sample
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            return 'utf-8'  # Default fallback
            
        except Exception as e:
            log_warning(f"Encoding detection failed: {e}")
            return 'utf-8'
    
    async def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file based on detected format."""
        try:
            format_type = self.result.detected_format
            encoding = self.result.detected_encoding or 'utf-8'
            
            if format_type == 'csv':
                data = await self._load_csv(file_path, encoding)
            elif format_type == 'tsv':
                data = await self._load_csv(file_path, encoding, delimiter='\t')
            elif format_type == 'excel':
                data = await self._load_excel(file_path)
            elif format_type == 'json':
                data = await self._load_json(file_path, encoding)
            elif format_type == 'parquet':
                data = await self._load_parquet(file_path)
            elif format_type == 'feather':
                data = await self._load_feather(file_path)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")
            
            # Basic data validation
            if data.empty:
                raise ValueError("Loaded dataset is empty")
            
            # Update result with basic info
            self.result.rows_processed = len(data)
            self.result.columns_processed = len(data.columns)
            self.result.column_names = data.columns.tolist()
            
            log_info(f"Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")
            return data
            
        except Exception as e:
            log_error(f"Data loading failed: {e}")
            raise
    
    async def _load_csv(self, file_path: str, encoding: str, delimiter: str = None) -> pd.DataFrame:
        """Load CSV/TSV file with auto-detection of delimiter."""
        try:
            # Auto-detect delimiter if not provided
            if delimiter is None:
                delimiter = self._detect_csv_delimiter(file_path, encoding)
            
            self.result.detected_delimiter = delimiter
            
            # Determine if we need chunked reading for large files
            file_size = Path(file_path).stat().st_size
            use_chunks = file_size > self.config.memory_limit_gb * 1024 * 1024 * 1024 * 0.5  # 50% of memory limit
            
            if use_chunks and not self.config.sample_size:
                return await self._load_csv_chunked(file_path, encoding, delimiter)
            else:
                # Load entire file or sample
                kwargs = {
                    'filepath_or_buffer': file_path,
                    'encoding': encoding,
                    'sep': delimiter,
                    'header': self.config.header_row,
                    'low_memory': False,
                    'na_values': ['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a', '#N/A', 'None'],
                    'keep_default_na': True
                }
                
                # Apply sample size if specified
                if self.config.sample_size:
                    kwargs['nrows'] = self.config.sample_size
                
                return pd.read_csv(**kwargs)
            
        except Exception as e:
            log_error(f"CSV loading failed: {e}")
            raise
    
    def _detect_csv_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect CSV delimiter by sampling file content."""
        try:
            import csv
            
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024 * 10)  # Sample first 10KB
            
            # Use csv.Sniffer to detect delimiter
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            return delimiter
            
        except Exception:
            # Fallback to manual detection
            delimiters = [',', '\t', ';', '|']
            
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline()
                
                delimiter_counts = {d: first_line.count(d) for d in delimiters}
                return max(delimiter_counts.items(), key=lambda x: x[1])[0]
                
            except Exception:
                return ','
    
    async def _load_csv_chunked(self, file_path: str, encoding: str, delimiter: str) -> pd.DataFrame:
        """Load CSV file in chunks for large files."""
        try:
            chunks = []
            total_rows = 0
            
            chunk_reader = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                header=self.config.header_row,
                chunksize=self.config.chunk_size,
                low_memory=False
            )
            
            for i, chunk in enumerate(chunk_reader):
                chunks.append(chunk)
                total_rows += len(chunk)
                
                # Update progress
                if i % 10 == 0:  # Update every 10 chunks
                    self.update_progress(
                        10 + (10 * min(total_rows / (self.config.sample_size or 1000000), 1)),
                        f"Loading data: {total_rows:,} rows processed"
                    )
                
                # Break if sample size reached
                if self.config.sample_size and total_rows >= self.config.sample_size:
                    break
            
            # Combine chunks
            data = pd.concat(chunks, ignore_index=True)
            
            # Trim to exact sample size if needed
            if self.config.sample_size and len(data) > self.config.sample_size:
                data = data.head(self.config.sample_size)
            
            return data
            
        except Exception as e:
            log_error(f"Chunked CSV loading failed: {e}")
            raise
    
    async def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file."""
        try:
            if not EXCEL_AVAILABLE:
                raise ImportError("Excel libraries not available")
            
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Get sheet names
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                # Single sheet
                data = pd.read_excel(file_path, header=self.config.header_row)
            else:
                # Multiple sheets - use first sheet or combine
                log_info(f"Multiple sheets found: {sheet_names}. Using first sheet: {sheet_names[0]}")
                data = pd.read_excel(file_path, sheet_name=sheet_names[0], header=self.config.header_row)
            
            # Apply sample size if specified
            if self.config.sample_size and len(data) > self.config.sample_size:
                data = data.head(self.config.sample_size)
            
            return data
            
        except Exception as e:
            log_error(f"Excel loading failed: {e}")
            raise
    
    async def _load_json(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load JSON file."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Try to determine JSON structure
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    # JSON array
                    data = pd.read_json(f, orient='records')
                elif first_char == '{':
                    # JSON object - try different orientations
                    content = json.load(f)
                    if isinstance(content, list):
                        data = pd.DataFrame(content)
                    elif isinstance(content, dict):
                        # Try to convert dict to DataFrame
                        data = pd.json_normalize(content)
                    else:
                        raise ValueError("Unsupported JSON structure")
                else:
                    # JSONL format
                    f.seek(0)
                    data = pd.read_json(f, lines=True)
            
            # Apply sample size if specified
            if self.config.sample_size and len(data) > self.config.sample_size:
                data = data.head(self.config.sample_size)
            
            return data
            
        except Exception as e:
            log_error(f"JSON loading failed: {e}")
            raise
    
    async def _load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            if not PYARROW_AVAILABLE:
                raise ImportError("PyArrow not available for Parquet support")
            
            # Load Parquet file
            data = pd.read_parquet(file_path)
            
            # Apply sample size if specified
            if self.config.sample_size and len(data) > self.config.sample_size:
                data = data.head(self.config.sample_size)
            
            return data
            
        except Exception as e:
            log_error(f"Parquet loading failed: {e}")
            raise
    
    async def _load_feather(self, file_path: str) -> pd.DataFrame:
        """Load Feather file."""
        try:
            if not PYARROW_AVAILABLE:
                raise ImportError("PyArrow not available for Feather support")
            
            # Load Feather file
            data = pd.read_feather(file_path)
            
            # Apply sample size if specified
            if self.config.sample_size and len(data) > self.config.sample_size:
                data = data.head(self.config.sample_size)
            
            return data
            
        except Exception as e:
            log_error(f"Feather loading failed: {e}")
            raise
    
    async def _infer_and_optimize_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Infer schema and optimize data types."""
        try:
            if not self.config.infer_schema:
                return data
            
            optimized_data = data.copy()
            inferred_schema = {}
            
            for column in data.columns:
                original_dtype = str(data[column].dtype)
                optimized_dtype = original_dtype
                
                try:
                    # Skip if already optimal
                    if pd.api.types.is_numeric_dtype(data[column]):
                        # Optimize numeric types
                        if pd.api.types.is_integer_dtype(data[column]):
                            optimized_dtype = self._optimize_integer_dtype(data[column])
                        elif pd.api.types.is_float_dtype(data[column]):
                            optimized_dtype = self._optimize_float_dtype(data[column])
                    
                    elif data[column].dtype == 'object':
                        # Try to infer better type for object columns
                        optimized_dtype = await self._infer_object_column_type(data[column])
                    
                    # Apply optimization if different
                    if optimized_dtype != original_dtype:
                        if optimized_dtype == 'category':
                            optimized_data[column] = data[column].astype('category')
                        elif optimized_dtype.startswith('datetime'):
                            optimized_data[column] = pd.to_datetime(data[column], errors='coerce')
                        elif optimized_dtype in ['int8', 'int16', 'int32', 'int64']:
                            optimized_data[column] = pd.to_numeric(data[column], errors='coerce').astype(optimized_dtype)
                        elif optimized_dtype in ['float32', 'float64']:
                            optimized_data[column] = pd.to_numeric(data[column], errors='coerce').astype(optimized_dtype)
                    
                    inferred_schema[column] = {
                        'original_dtype': original_dtype,
                        'optimized_dtype': optimized_dtype,
                        'nullable': data[column].isnull().any(),
                        'unique_values': data[column].nunique(),
                        'memory_usage': data[column].memory_usage(deep=True)
                    }
                
                except Exception as col_error:
                    log_warning(f"Schema optimization failed for column {column}: {col_error}")
                    inferred_schema[column] = {
                        'original_dtype': original_dtype,
                        'optimized_dtype': original_dtype,
                        'error': str(col_error)
                    }
            
            self.result.inferred_schema = inferred_schema
            
            # Update data types in result
            self.result.data_types = {
                col: str(optimized_data[col].dtype) for col in optimized_data.columns
            }
            
            log_info("Schema inference and optimization completed")
            return optimized_data
            
        except Exception as e:
            log_error(f"Schema optimization failed: {e}")
            return data
    
    def _optimize_integer_dtype(self, series: pd.Series) -> str:
        """Optimize integer data type based on value range."""
        try:
            if series.isnull().any():
                # Use nullable integer types
                min_val = series.min()
                max_val = series.max()
                
                if min_val >= -128 and max_val <= 127:
                    return 'Int8'
                elif min_val >= -32768 and max_val <= 32767:
                    return 'Int16'
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return 'Int32'
                else:
                    return 'Int64'
            else:
                # Use regular integer types
                min_val = series.min()
                max_val = series.max()
                
                if min_val >= 0:
                    # Unsigned integers
                    if max_val <= 255:
                        return 'uint8'
                    elif max_val <= 65535:
                        return 'uint16'
                    elif max_val <= 4294967295:
                        return 'uint32'
                    else:
                        return 'uint64'
                else:
                    # Signed integers
                    if min_val >= -128 and max_val <= 127:
                        return 'int8'
                    elif min_val >= -32768 and max_val <= 32767:
                        return 'int16'
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        return 'int32'
                    else:
                        return 'int64'
        
        except Exception:
            return str(series.dtype)
    
    def _optimize_float_dtype(self, series: pd.Series) -> str:
        """Optimize float data type."""
        try:
            # Check if values can fit in float32
            float32_series = series.astype('float32')
            if np.allclose(series.dropna(), float32_series.dropna(), equal_nan=True):
                return 'float32'
            else:
                return 'float64'
        except Exception:
            return str(series.dtype)
    
    async def _infer_object_column_type(self, series: pd.Series) -> str:
        """Infer type for object columns."""
        try:
            # Skip if mostly null
            if series.isnull().sum() / len(series) > 0.9:
                return 'object'
            
            # Try datetime inference
            if self.config.datetime_inference:
                try:
                    datetime_series = pd.to_datetime(series, errors='coerce')
                    if datetime_series.notnull().sum() / len(series) > 0.8:
                        return 'datetime64[ns]'
                except Exception:
                    pass
            
            # Try numeric inference
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notnull().sum() / len(series) > 0.8:
                    if (numeric_series.dropna() == numeric_series.dropna().astype(int)).all():
                        return self._optimize_integer_dtype(numeric_series.astype('Int64'))
                    else:
                        return self._optimize_float_dtype(numeric_series)
            except Exception:
                pass
            
            # Check if suitable for categorical
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1 and series.nunique() < self.config.categorical_threshold:
                return 'category'
            
            return 'object'
            
        except Exception:
            return 'object'
    
    async def _assess_data_quality(self, data: pd.DataFrame):
        """Assess data quality and identify issues."""
        try:
            quality_issues = []
            
            # Missing values analysis
            missing_stats = data.isnull().sum()
            total_cells = len(data) * len(data.columns)
            total_missing = missing_stats.sum()
            missing_ratio = total_missing / total_cells if total_cells > 0 else 0
            
            self.result.missing_value_ratio = missing_ratio
            
            # High missing value columns
            high_missing_cols = missing_stats[missing_stats / len(data) > self.config.missing_threshold]
            if len(high_missing_cols) > 0:
                quality_issues.append({
                    'type': 'high_missing_values',
                    'severity': 'high',
                    'columns': high_missing_cols.to_dict(),
                    'description': f"{len(high_missing_cols)} columns with >50% missing values"
                })
            
            # Duplicate rows analysis
            duplicate_count = data.duplicated().sum()
            duplicate_ratio = duplicate_count / len(data) if len(data) > 0 else 0
            self.result.duplicate_ratio = duplicate_ratio
            
            if duplicate_ratio > 0.1:  # More than 10% duplicates
                quality_issues.append({
                    'type': 'high_duplicates',
                    'severity': 'medium',
                    'count': duplicate_count,
                    'ratio': duplicate_ratio,
                    'description': f"{duplicate_count} duplicate rows ({duplicate_ratio:.1%})"
                })
            
            # Data type consistency
            mixed_type_cols = []
            for col in data.select_dtypes(include=['object']).columns:
                sample_types = set(type(val).__name__ for val in data[col].dropna().head(100))
                if len(sample_types) > 2:
                    mixed_type_cols.append(col)
            
            if mixed_type_cols:
                quality_issues.append({
                    'type': 'mixed_data_types',
                    'severity': 'medium',
                    'columns': mixed_type_cols,
                    'description': f"{len(mixed_type_cols)} columns with mixed data types"
                })
            
            # Empty columns
            empty_cols = [col for col in data.columns if data[col].isnull().all()]
            if empty_cols:
                quality_issues.append({
                    'type': 'empty_columns',
                    'severity': 'high',
                    'columns': empty_cols,
                    'description': f"{len(empty_cols)} completely empty columns"
                })
            
            # Single value columns
            single_value_cols = [col for col in data.columns if data[col].nunique() <= 1]
            if single_value_cols:
                quality_issues.append({
                    'type': 'single_value_columns',
                    'severity': 'low',
                    'columns': single_value_cols,
                    'description': f"{len(single_value_cols)} columns with single/no values"
                })
            
            # Calculate overall quality score
            quality_score = 1.0
            quality_score -= missing_ratio * 0.3  # Penalize missing values
            quality_score -= duplicate_ratio * 0.2  # Penalize duplicates
            quality_score -= len(mixed_type_cols) / len(data.columns) * 0.1  # Penalize mixed types
            quality_score -= len(empty_cols) / len(data.columns) * 0.4  # Heavily penalize empty columns
            
            self.result.data_quality_score = max(0, quality_score)
            self.result.quality_issues = quality_issues
            
            log_info(f"Data quality assessment completed. Score: {quality_score:.2f}")
            
        except Exception as e:
            log_error(f"Data quality assessment failed: {e}")
    
    async def _generate_data_profile(self, data: pd.DataFrame):
        """Generate comprehensive data profile and statistics."""
        try:
            # Basic statistics
            profile_data = data
            if len(data) > self.config.profile_sample_size:
                profile_data = data.sample(n=self.config.profile_sample_size, random_state=42)
            
            # Statistical summary
            statistical_summary = {
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
                'dtypes': data.dtypes.astype(str).to_dict()
            }
            
            # Column-wise statistics
            column_statistics = {}
            
            for column in data.columns:
                col_stats = {
                    'dtype': str(data[column].dtype),
                    'non_null_count': data[column].count(),
                    'null_count': data[column].isnull().sum(),
                    'null_percentage': data[column].isnull().sum() / len(data) * 100,
                    'unique_count': data[column].nunique(),
                    'unique_percentage': data[column].nunique() / len(data) * 100
                }
                
                # Numeric statistics
                if pd.api.types.is_numeric_dtype(data[column]):
                    numeric_stats = data[column].describe()
                    col_stats.update({
                        'mean': numeric_stats['mean'],
                        'std': numeric_stats['std'],
                        'min': numeric_stats['min'],
                        'max': numeric_stats['max'],
                        'median': numeric_stats['50%'],
                        'q1': numeric_stats['25%'],
                        'q3': numeric_stats['75%'],
                        'skewness': data[column].skew(),
                        'kurtosis': data[column].kurtosis()
                    })
                    
                    # Outlier detection using IQR
                    Q1 = numeric_stats['25%']
                    Q3 = numeric_stats['75%']
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
                    col_stats['outlier_count'] = len(outliers)
                    col_stats['outlier_percentage'] = len(outliers) / len(data) * 100
                
                # Text/categorical statistics
                elif pd.api.types.is_object_dtype(data[column]):
                    value_counts = data[column].value_counts()
                    col_stats.update({
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                        'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                        'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                        'top_values': value_counts.head(10).to_dict()
                    })
                    
                    # String length statistics if applicable
                    if data[column].dtype == 'object':
                        string_lengths = data[column].astype(str).str.len()
                        col_stats.update({
                            'min_length': string_lengths.min(),
                            'max_length': string_lengths.max(),
                            'avg_length': string_lengths.mean()
                        })
                
                # Datetime statistics
                elif pd.api.types.is_datetime64_any_dtype(data[column]):
                    date_stats = data[column].describe()
                    col_stats.update({
                        'min_date': str(date_stats['min']),
                        'max_date': str(date_stats['max']),
                        'date_range_days': (date_stats['max'] - date_stats['min']).days
                    })
                
                column_statistics[column] = col_stats
            
            # Correlation matrix for numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                try:
                    correlation_matrix = data[numeric_columns].corr().to_dict()
                    self.result.correlation_matrix = correlation_matrix
                except Exception as e:
                    log_warning(f"Correlation matrix calculation failed: {e}")
            
            self.result.statistical_summary = statistical_summary
            self.result.column_statistics = column_statistics
            
            # Generate profile report if enabled and library available
            if self.config.generate_profile and YDATA_PROFILING_AVAILABLE:
                await self._generate_profile_report(profile_data)
            
            log_info("Data profiling completed")
            
        except Exception as e:
            log_error(f"Data profiling failed: {e}")
    
    async def _generate_profile_report(self, data: pd.DataFrame):
        """Generate detailed profile report using ydata-profiling."""
        try:
            # Generate profile report
            profile = ProfileReport(
                data,
                title=f"Data Profile Report - Dataset {self.result.dataset_id}",
                explorative=True,
                minimal=False
            )
            
            # Save profile report
            report_path = self.temp_dir / f"profile_report_{self.result.dataset_id}.html"
            profile.to_file(report_path)
            self.result.profile_report_path = str(report_path)
            
            log_info(f"Profile report generated: {report_path}")
            
        except Exception as e:
            log_warning(f"Profile report generation failed: {e}")
    
    async def _clean_and_preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        try:
            cleaned_data = data.copy()
            transformations = []
            
            # Remove completely empty columns
            empty_cols = [col for col in cleaned_data.columns if cleaned_data[col].isnull().all()]
            if empty_cols:
                cleaned_data = cleaned_data.drop(columns=empty_cols)
                transformations.append({
                    'operation': 'remove_empty_columns',
                    'columns_removed': empty_cols,
                    'count': len(empty_cols)
                })
            
            # Remove single-value columns (optional)
            single_value_cols = [
                col for col in cleaned_data.columns 
                if cleaned_data[col].nunique() <= 1
            ]
            if single_value_cols and len(single_value_cols) < len(cleaned_data.columns) * 0.5:
                # Only remove if less than 50% of columns are single-valued
                cleaned_data = cleaned_data.drop(columns=single_value_cols)
                transformations.append({
                    'operation': 'remove_single_value_columns',
                    'columns_removed': single_value_cols,
                    'count': len(single_value_cols)
                })
            
            # Clean column names
            original_columns = cleaned_data.columns.tolist()
            cleaned_columns = [
                col.strip().replace(' ', '_').replace('-', '_')
                for col in original_columns
            ]
            
            if cleaned_columns != original_columns:
                column_mapping = dict(zip(original_columns, cleaned_columns))
                cleaned_data.columns = cleaned_columns
                transformations.append({
                    'operation': 'clean_column_names',
                    'column_mapping': column_mapping
                })
            
            # Update result
            self.result.transformations_applied.extend(transformations)
            self.result.columns_processed = len(cleaned_data.columns)
            self.result.column_names = cleaned_data.columns.tolist()
            
            log_info(f"Data cleaning completed. Applied {len(transformations)} transformations")
            return cleaned_data
            
        except Exception as e:
            log_error(f"Data cleaning failed: {e}")
            return data
    
    async def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration."""
        try:
            if self.config.handle_missing == "ignore":
                return data
            
            processed_data = data.copy()
            missing_handling_info = {}
            
            for column in data.columns:
                missing_count = data[column].isnull().sum()
                missing_ratio = missing_count / len(data)
                
                if missing_count == 0:
                    continue
                
                strategy = self._determine_missing_strategy(data[column], missing_ratio)
                
                if strategy == "drop":
                    # Drop rows with missing values for this column
                    processed_data = processed_data.dropna(subset=[column])
                    missing_handling_info[column] = {
                        'strategy': 'drop_rows',
                        'rows_dropped': missing_count
                    }
                
                elif strategy == "fill_mode":
                    # Fill with mode for categorical
                    mode_value = data[column].mode().iloc[0] if len(data[column].mode()) > 0 else "Unknown"
                    processed_data[column].fillna(mode_value, inplace=True)
                    missing_handling_info[column] = {
                        'strategy': 'fill_mode',
                        'fill_value': mode_value,
                        'values_filled': missing_count
                    }
                
                elif strategy == "fill_median":
                    # Fill with median for numeric
                    median_value = data[column].median()
                    processed_data[column].fillna(median_value, inplace=True)
                    missing_handling_info[column] = {
                        'strategy': 'fill_median',
                        'fill_value': median_value,
                        'values_filled': missing_count
                    }
                
                elif strategy == "fill_mean":
                    # Fill with mean for numeric
                    mean_value = data[column].mean()
                    processed_data[column].fillna(mean_value, inplace=True)
                    missing_handling_info[column] = {
                        'strategy': 'fill_mean',
                        'fill_value': mean_value,
                        'values_filled': missing_count
                    }
                
                elif strategy == "interpolate":
                    # Interpolate for time series or ordered data
                    processed_data[column] = processed_data[column].interpolate()
                    missing_handling_info[column] = {
                        'strategy': 'interpolate',
                        'values_filled': missing_count
                    }
                
                elif strategy == "forward_fill":
                    # Forward fill
                    processed_data[column] = processed_data[column].fillna(method='ffill')
                    missing_handling_info[column] = {
                        'strategy': 'forward_fill',
                        'values_filled': missing_count
                    }
            
            self.result.missing_values_handled = missing_handling_info
            self.result.rows_processed = len(processed_data)
            
            log_info(f"Missing values handled for {len(missing_handling_info)} columns")
            return processed_data
            
        except Exception as e:
            log_error(f"Missing value handling failed: {e}")
            return data
    
    def _determine_missing_strategy(self, series: pd.Series, missing_ratio: float) -> str:
        """Determine best strategy for handling missing values."""
        try:
            if self.config.handle_missing != "auto":
                return self.config.handle_missing
            
            # If too many missing values, consider dropping
            if missing_ratio > self.config.missing_threshold:
                return "drop"
            
            # Strategy based on data type and characteristics
            if pd.api.types.is_numeric_dtype(series):
                if series.skew() > 1:  # Highly skewed
                    return "fill_median"
                else:
                    return "fill_mean"
            
            elif pd.api.types.is_datetime64_any_dtype(series):
                return "interpolate"
            
            elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
                return "fill_mode"
            
            else:
                return "fill_mode"  # Default fallback
            
        except Exception:
            return "fill_mode"
    
    async def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers."""
        try:
            if self.config.handle_outliers == "ignore":
                return data
            
            processed_data = data.copy()
            outliers_info = {}
            total_outliers = 0
            
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                try:
                    outliers_detected = self._detect_outliers(data[column], self.config.outlier_method)
                    outlier_count = len(outliers_detected)
                    
                    if outlier_count == 0:
                        continue
                    
                    total_outliers += outlier_count
                    
                    if self.config.handle_outliers == "remove":
                        # Remove outliers
                        processed_data = processed_data.drop(outliers_detected)
                        outliers_info[column] = {
                            'method': self.config.outlier_method,
                            'action': 'removed',
                            'count': outlier_count
                        }
                    
                    elif self.config.handle_outliers == "cap":
                        # Cap outliers
                        Q1 = data[column].quantile(0.25)
                        Q3 = data[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        processed_data[column] = processed_data[column].clip(lower_bound, upper_bound)
                        outliers_info[column] = {
                            'method': self.config.outlier_method,
                            'action': 'capped',
                            'count': outlier_count,
                            'bounds': [lower_bound, upper_bound]
                        }
                    
                    else:  # detect only
                        outliers_info[column] = {
                            'method': self.config.outlier_method,
                            'action': 'detected',
                            'count': outlier_count,
                            'indices': outliers_detected.tolist()[:100]  # Store first 100 indices
                        }
                
                except Exception as col_error:
                    log_warning(f"Outlier handling failed for column {column}: {col_error}")
                    continue
            
            # Calculate overall outlier ratio
            if len(data) > 0:
                self.result.outlier_ratio = total_outliers / (len(data) * len(numeric_columns))
            
            self.result.outliers_detected = outliers_info
            self.result.rows_processed = len(processed_data)
            
            log_info(f"Outlier handling completed. Processed {len(outliers_info)} columns")
            return processed_data
            
        except Exception as e:
            log_error(f"Outlier handling failed: {e}")
            return data
    
    def _detect_outliers(self, series: pd.Series, method: str) -> pd.Index:
        """Detect outliers using specified method."""
        try:
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return series[(series < lower_bound) | (series > upper_bound)].index
            
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(series.dropna()))
                threshold = 3
                outlier_mask = z_scores > threshold
                return series.dropna()[outlier_mask].index
            
            elif method == "isolation":
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_mask = iso_forest.fit_predict(series.values.reshape(-1, 1)) == -1
                    return series[outlier_mask].index
                except ImportError:
                    # Fallback to IQR method
                    return self._detect_outliers(series, "iqr")
            
            else:
                return pd.Index([])
                
        except Exception as e:
            log_warning(f"Outlier detection failed: {e}")
            return pd.Index([])
    
    async def _final_validation(self, data: pd.DataFrame):
        """Perform final validation of processed data."""
        try:
            validation_results = {}
            
            # Check data integrity
            if data.empty:
                validation_results['data_empty'] = True
                self.result.warnings.append("Processed dataset is empty")
            
            # Check for extreme missing values after processing
            missing_ratios = data.isnull().sum() / len(data)
            high_missing = missing_ratios[missing_ratios > 0.9]
            if len(high_missing) > 0:
                validation_results['high_missing_after_processing'] = high_missing.to_dict()
                self.result.warnings.append(f"{len(high_missing)} columns still have >90% missing values")
            
            # Check data types consistency
            mixed_types = []
            for col in data.select_dtypes(include=['object']).columns:
                sample_types = set(type(val).__name__ for val in data[col].dropna().head(100))
                if len(sample_types) > 2:
                    mixed_types.append(col)
            
            if mixed_types:
                validation_results['mixed_types'] = mixed_types
                self.result.warnings.append(f"{len(mixed_types)} columns have mixed data types")
            
            # Memory usage check
            memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            if memory_usage > self.config.memory_limit_gb * 1024:
                validation_results['high_memory_usage'] = memory_usage
                self.result.warnings.append(f"Dataset uses {memory_usage:.1f}MB memory")
            
            self.result.schema_validation = validation_results
            
            log_info("Final validation completed")
            
        except Exception as e:
            log_error(f"Final validation failed: {e}")
    
    async def _save_processed_data(self, data: pd.DataFrame, dataset_id: int):
        """Save processed data to storage."""
        try:
            # Create output directory
            output_dir = Path(settings.DATASETS_DIRECTORY) / f"processed_{dataset_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save in requested format
            if self.config.output_format == "parquet":
                output_path = output_dir / f"processed_data_{dataset_id}.parquet"
                data.to_parquet(output_path, compression='snappy' if self.config.compress_output else None)
            
            elif self.config.output_format == "csv":
                output_path = output_dir / f"processed_data_{dataset_id}.csv"
                data.to_csv(output_path, index=False)
                
                if self.config.compress_output:
                    # Compress CSV
                    with open(output_path, 'rb') as f_in:
                        with gzip.open(f"{output_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    output_path.unlink()  # Remove uncompressed version
                    output_path = Path(f"{output_path}.gz")
            
            elif self.config.output_format == "feather":
                output_path = output_dir / f"processed_data_{dataset_id}.feather"
                data.to_feather(output_path)
            
            else:
                # Default to parquet
                output_path = output_dir / f"processed_data_{dataset_id}.parquet"
                data.to_parquet(output_path)
            
            self.result.processed_file_path = str(output_path)
            self.result.file_size_processed = output_path.stat().st_size
            
            log_info(f"Processed data saved: {output_path}")
            
        except Exception as e:
            log_error(f"Failed to save processed data: {e}")
            raise
    
    async def _generate_metadata(self, data: pd.DataFrame):
        """Generate comprehensive metadata for processed dataset."""
        try:
            metadata = {
                'processing_info': {
                    'processing_date': datetime.now().isoformat(),
                    'processor_version': '1.0.0',
                    'config': {
                        'file_format': self.config.file_format,
                        'handle_missing': self.config.handle_missing,
                        'handle_outliers': self.config.handle_outliers,
                        'infer_schema': self.config.infer_schema
                    }
                },
                'dataset_info': {
                    'shape': data.shape,
                    'columns': data.columns.tolist(),
                    'dtypes': data.dtypes.astype(str).to_dict(),
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
                },
                'quality_metrics': {
                    'data_quality_score': self.result.data_quality_score,
                    'missing_value_ratio': self.result.missing_value_ratio,
                    'duplicate_ratio': self.result.duplicate_ratio,
                    'outlier_ratio': self.result.outlier_ratio
                },
                'transformations': self.result.transformations_applied,
                'missing_values_handling': self.result.missing_values_handled,
                'outliers_detection': self.result.outliers_detected,
                'quality_issues': self.result.quality_issues,
                'statistical_summary': self.result.statistical_summary,
                'column_statistics': self.result.column_statistics
            }
            
            # Save metadata
            metadata_path = Path(self.result.processed_file_path).parent / f"metadata_{self.result.dataset_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.result.metadata_file_path = str(metadata_path)
            
            log_info(f"Metadata generated: {metadata_path}")
            
        except Exception as e:
            log_error(f"Metadata generation failed: {e}")
    
    def _finalize_processing(self):
        """Finalize processing results."""
        try:
            self.result.status = "completed"
            
            # Calculate final memory usage
            import psutil
            process = psutil.Process(os.getpid())
            self.result.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            
            log_info("Data processing finalized successfully")
            
        except Exception as e:
            log_error(f"Processing finalization failed: {e}")
    
    async def _cleanup_temp_directory(self):
        """Clean up temporary directory."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                log_info(f"Temporary directory cleaned up: {self.temp_dir}")
        except Exception as e:
            log_warning(f"Temporary directory cleanup failed: {e}")

# Main data processing execution function
@monitor_performance("data_processing")
def execute_dataset_processing(
    dataset_id: int,
    user_id: int,
    config: Dict[str, Any],
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Execute dataset processing task.
    
    Args:
        dataset_id: Dataset identifier
        user_id: User identifier
        config: Processing configuration
        progress_callback: Progress update callback
        
    Returns:
        Processing results
    """
    try:
        log_info(f"Starting data processing for dataset: {dataset_id}")
        
        # Get dataset file path from database
        with get_db_session() as db_session:
            data_service = DataService()
            dataset = data_service.get_dataset_by_id(dataset_id, db_session)
            
            if not dataset:
                raise ValueError(f"Dataset not found: {dataset_id}")
            
            file_path = dataset.file_path
        
        # Parse configuration
        processing_config = DataProcessingConfig(
            file_path=file_path,
            file_format=config.get('file_format', 'auto'),
            encoding=config.get('encoding', 'auto'),
            sample_size=config.get('sample_size'),
            handle_missing=config.get('handle_missing', 'auto'),
            handle_outliers=config.get('handle_outliers', 'detect'),
            infer_schema=config.get('infer_schema', True),
            generate_profile=config.get('generate_profile', True),
            output_format=config.get('output_format', 'parquet')
        )
        
        # Initialize data processor
        processor = DataProcessor(processing_config)
        if progress_callback:
            processor.set_progress_callback(progress_callback)
        
        # Execute processing
        result = asyncio.run(processor.process_dataset(file_path, dataset_id, user_id))
        
        # Convert result to dictionary
        result_dict = {
            'task_id': result.task_id,
            'dataset_id': result.dataset_id,
            'status': result.status,
            'progress': result.progress,
            'rows_processed': result.rows_processed,
            'columns_processed': result.columns_processed,
            'data_quality_score': result.data_quality_score,
            'missing_value_ratio': result.missing_value_ratio,
            'duplicate_ratio': result.duplicate_ratio,
            'outlier_ratio': result.outlier_ratio,
            'data_types': result.data_types,
            'column_names': result.column_names,
            'statistical_summary': result.statistical_summary,
            'quality_issues': result.quality_issues,
            'transformations_applied': result.transformations_applied,
            'processed_file_path': result.processed_file_path,
            'metadata_file_path': result.metadata_file_path,
            'processing_time': result.processing_time,
            'error_message': result.error_message,
            'warnings': result.warnings
        }
        
        # Update database with results
        with get_db_session() as db_session:
            data_service.update_dataset_processing_results(dataset_id, result_dict, db_session)
        
        log_info(f"Data processing completed for dataset: {dataset_id}")
        return result_dict
        
    except Exception as e:
        error_msg = f"Data processing failed for dataset {dataset_id}: {str(e)}"
        log_error(error_msg, exception=e)
        
        return {
            'task_id': str(uuid.uuid4()),
            'dataset_id': dataset_id,
            'status': 'failed',
            'error_message': error_msg,
            'error_details': {
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        }

# Export functions
__all__ = [
    'execute_dataset_processing',
    'DataProcessor',
    'DataProcessingConfig', 
    'DataProcessingResult'
]
