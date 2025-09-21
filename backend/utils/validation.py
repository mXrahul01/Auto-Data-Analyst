"""
Comprehensive Data Validation Module for Auto-Analyst Platform

This module provides production-ready data validation capabilities for the Auto-Analyst
platform, ensuring datasets are clean, complete, and compatible with ML pipelines
before processing begins.

Features:
- Schema Validation: Column existence, types, constraints validation
- Data Quality Checks: Missing values, duplicates, outliers, consistency
- Type Conversion: Safe casting and automatic type inference
- Integrity Validation: Range checks, format validation, business rules
- Performance Optimization: Efficient validation for large datasets
- Comprehensive Reporting: Detailed validation reports with actionable insights
- Monitoring Integration: Prometheus metrics and drift detection

Components:
- DataValidator: Main validation orchestrator
- SchemaValidator: Column and type validation
- QualityValidator: Data quality assessment
- TypeConverter: Safe type conversion utilities
- ValidationReporter: Report generation and formatting
- ValidationConfig: Configuration management

Usage:
    # Basic validation
    validator = DataValidator()
    result = validator.validate_dataset(df, schema)
    
    # Custom validation with config
    config = ValidationConfig(strict_mode=True)
    validator = DataValidator(config)
    result = validator.validate_dataset(df, schema)
    
    # Type conversion
    converter = TypeConverter()
    converted_df = converter.convert_types(df, type_hints)

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- pydantic: Data validation and serialization
- jsonschema: JSON schema validation
- dateutil: Date parsing utilities
- re: Regular expressions for pattern matching
"""

import logging
import warnings
import re
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# Core data processing imports
import pandas as pd
import numpy as np
from dateutil import parser as date_parser

# Validation and schema imports
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    from pydantic.dataclasses import dataclass as pydantic_dataclass
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import jsonschema
    from jsonschema import validate, ValidationError as JSONValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

# Monitoring integration
try:
    from .monitoring import log_error, log_warning, log_info, monitor_performance
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')

# Configure logging
logger = logging.getLogger(__name__)

class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"          # Fail on any validation error
    MODERATE = "moderate"      # Warn on minor issues, fail on major ones
    LENIENT = "lenient"        # Log issues but continue processing
    REPORT_ONLY = "report_only"  # Only report issues, never fail

class DataType(str, Enum):
    """Supported data types for validation."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    CATEGORICAL = "categorical"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    CURRENCY = "currency"

class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"

class IssueType(str, Enum):
    """Types of validation issues."""
    MISSING_COLUMN = "missing_column"
    EXTRA_COLUMN = "extra_column"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_VALUES = "missing_values"
    INVALID_VALUES = "invalid_values"
    DUPLICATE_ROWS = "duplicate_rows"
    OUTLIERS = "outliers"
    RANGE_VIOLATION = "range_violation"
    FORMAT_ERROR = "format_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    SCHEMA_ERROR = "schema_error"
    DATA_QUALITY = "data_quality"

@dataclass
class ColumnConstraint:
    """Constraints for individual columns."""
    
    name: str
    data_type: DataType
    required: bool = True
    nullable: bool = True
    unique: bool = False
    
    # Numeric constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # String constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    
    # Categorical constraints
    allowed_values: Optional[Set[str]] = None
    
    # Custom validation function
    custom_validator: Optional[Callable] = None
    
    # Metadata
    description: Optional[str] = None
    default_value: Any = None

@dataclass
class ValidationIssue:
    """Individual validation issue."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issue_type: IssueType = IssueType.DATA_QUALITY
    severity: ValidationStatus = ValidationStatus.WARNING
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None
    can_auto_fix: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationConfig:
    """Configuration for validation behavior."""
    
    # General settings
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    max_errors: int = 100
    sample_size: Optional[int] = None  # For large datasets
    
    # Schema validation settings
    strict_schema: bool = False
    allow_extra_columns: bool = True
    auto_convert_types: bool = True
    
    # Data quality settings
    missing_value_threshold: float = 0.5  # Fail if >50% missing
    duplicate_threshold: float = 0.1      # Warn if >10% duplicates
    outlier_detection: bool = True
    outlier_threshold: float = 3.0        # Z-score threshold
    
    # Type conversion settings
    date_formats: List[str] = field(default_factory=lambda: [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
        '%d-%m-%Y', '%Y/%m/%d', '%B %d, %Y'
    ])
    
    # Performance settings
    chunk_size: int = 10000
    enable_parallel: bool = True
    timeout_seconds: int = 300
    
    # Reporting settings
    include_suggestions: bool = True
    detailed_reporting: bool = True
    include_statistics: bool = True

@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    
    validation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ValidationStatus = ValidationStatus.PASSED
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Dataset information
    dataset_shape: Tuple[int, int] = (0, 0)
    dataset_size_mb: float = 0.0
    columns_validated: int = 0
    rows_validated: int = 0
    
    # Validation results
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0
    critical_count: int = 0
    
    # Schema compliance
    schema_valid: bool = True
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    type_mismatches: Dict[str, str] = field(default_factory=dict)
    
    # Data quality metrics
    missing_value_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    outlier_ratio: float = 0.0
    data_quality_score: float = 1.0
    
    # Performance metrics
    validation_duration: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    auto_fixes_available: bool = False
    
    # Raw statistics
    column_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class SchemaValidator:
    """Handles schema validation for datasets."""
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize schema validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_schema(
        self, 
        df: pd.DataFrame, 
        schema: Dict[str, ColumnConstraint]
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate DataFrame against schema constraints.
        
        Args:
            df: DataFrame to validate
            schema: Schema definition with column constraints
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check for missing required columns
            required_columns = {name for name, constraint in schema.items() if constraint.required}
            missing_columns = required_columns - set(df.columns)
            
            for col in missing_columns:
                issues.append(ValidationIssue(
                    issue_type=IssueType.MISSING_COLUMN,
                    severity=ValidationStatus.CRITICAL,
                    column=col,
                    message=f"Required column '{col}' is missing from dataset",
                    suggestion=f"Add column '{col}' to your dataset or mark it as optional in schema"
                ))
            
            # Check for extra columns if strict mode
            if not self.config.allow_extra_columns:
                expected_columns = set(schema.keys())
                extra_columns = set(df.columns) - expected_columns
                
                for col in extra_columns:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.EXTRA_COLUMN,
                        severity=ValidationStatus.WARNING,
                        column=col,
                        message=f"Unexpected column '{col}' found in dataset",
                        suggestion=f"Remove column '{col}' or add it to the schema definition"
                    ))
            
            # Validate individual columns
            for col_name, constraint in schema.items():
                if col_name in df.columns:
                    col_issues = self._validate_column(df[col_name], constraint)
                    issues.extend(col_issues)
            
            # Overall schema validation status
            critical_issues = [issue for issue in issues if issue.severity == ValidationStatus.CRITICAL]
            is_valid = len(critical_issues) == 0 or self.config.validation_level == ValidationLevel.LENIENT
            
            return is_valid, issues
            
        except Exception as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            issues.append(ValidationIssue(
                issue_type=IssueType.SCHEMA_ERROR,
                severity=ValidationStatus.CRITICAL,
                message=f"Schema validation error: {str(e)}",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            ))
            return False, issues
    
    def _validate_column(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate a single column against its constraints."""
        issues = []
        column_name = constraint.name
        
        try:
            # Check for required non-null values
            if not constraint.nullable:
                null_count = series.isnull().sum()
                if null_count > 0:
                    null_indices = series[series.isnull()].index.tolist()
                    issues.append(ValidationIssue(
                        issue_type=IssueType.MISSING_VALUES,
                        severity=ValidationStatus.ERROR,
                        column=column_name,
                        row_indices=null_indices,
                        message=f"Column '{column_name}' contains {null_count} null values but is marked as non-nullable",
                        details={'null_count': null_count, 'total_rows': len(series)},
                        suggestion="Either remove rows with null values or mark column as nullable"
                    ))
            
            # Type validation
            type_issues = self._validate_column_type(series, constraint)
            issues.extend(type_issues)
            
            # Range validation for numeric types
            if constraint.data_type in [DataType.INTEGER, DataType.FLOAT]:
                range_issues = self._validate_numeric_range(series, constraint)
                issues.extend(range_issues)
            
            # String length validation
            if constraint.data_type == DataType.STRING:
                length_issues = self._validate_string_constraints(series, constraint)
                issues.extend(length_issues)
            
            # Pattern validation
            if constraint.pattern:
                pattern_issues = self._validate_pattern(series, constraint)
                issues.extend(pattern_issues)
            
            # Allowed values validation
            if constraint.allowed_values:
                values_issues = self._validate_allowed_values(series, constraint)
                issues.extend(values_issues)
            
            # Uniqueness validation
            if constraint.unique:
                uniqueness_issues = self._validate_uniqueness(series, constraint)
                issues.extend(uniqueness_issues)
            
            # Custom validation
            if constraint.custom_validator:
                custom_issues = self._validate_custom(series, constraint)
                issues.extend(custom_issues)
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.SCHEMA_ERROR,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Column validation error: {str(e)}",
                details={'error': str(e)}
            ))
        
        return issues
    
    def _validate_column_type(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate column data type."""
        issues = []
        column_name = constraint.name
        expected_type = constraint.data_type
        
        try:
            # Skip validation for completely null columns
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                return issues
            
            type_validation_map = {
                DataType.INTEGER: self._is_integer_type,
                DataType.FLOAT: self._is_numeric_type,
                DataType.STRING: self._is_string_type,
                DataType.BOOLEAN: self._is_boolean_type,
                DataType.DATETIME: self._is_datetime_type,
                DataType.DATE: self._is_date_type,
                DataType.EMAIL: self._is_email_type,
                DataType.URL: self._is_url_type,
                DataType.CATEGORICAL: self._is_categorical_type,
                DataType.CURRENCY: self._is_currency_type
            }
            
            validator_func = type_validation_map.get(expected_type)
            if validator_func:
                is_valid, invalid_indices = validator_func(non_null_series)
                
                if not is_valid and len(invalid_indices) > 0:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.TYPE_MISMATCH,
                        severity=ValidationStatus.ERROR if self.config.strict_schema else ValidationStatus.WARNING,
                        column=column_name,
                        row_indices=invalid_indices[:100],  # Limit to first 100 for performance
                        message=f"Column '{column_name}' contains {len(invalid_indices)} values incompatible with type '{expected_type.value}'",
                        details={
                            'expected_type': expected_type.value,
                            'actual_type': str(series.dtype),
                            'invalid_count': len(invalid_indices)
                        },
                        suggestion=f"Convert values to {expected_type.value} type or update schema",
                        can_auto_fix=self.config.auto_convert_types
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Type validation error: {str(e)}",
                details={'error': str(e)}
            ))
        
        return issues
    
    def _validate_numeric_range(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate numeric range constraints."""
        issues = []
        column_name = constraint.name
        
        try:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            
            if constraint.min_value is not None:
                below_min = numeric_series < constraint.min_value
                if below_min.any():
                    below_indices = series[series.index.isin(numeric_series[below_min].index)].index.tolist()
                    issues.append(ValidationIssue(
                        issue_type=IssueType.RANGE_VIOLATION,
                        severity=ValidationStatus.ERROR,
                        column=column_name,
                        row_indices=below_indices[:50],
                        message=f"Column '{column_name}' has {below_min.sum()} values below minimum {constraint.min_value}",
                        details={'min_value': constraint.min_value, 'violation_count': below_min.sum()},
                        suggestion=f"Remove or correct values below {constraint.min_value}"
                    ))
            
            if constraint.max_value is not None:
                above_max = numeric_series > constraint.max_value
                if above_max.any():
                    above_indices = series[series.index.isin(numeric_series[above_max].index)].index.tolist()
                    issues.append(ValidationIssue(
                        issue_type=IssueType.RANGE_VIOLATION,
                        severity=ValidationStatus.ERROR,
                        column=column_name,
                        row_indices=above_indices[:50],
                        message=f"Column '{column_name}' has {above_max.sum()} values above maximum {constraint.max_value}",
                        details={'max_value': constraint.max_value, 'violation_count': above_max.sum()},
                        suggestion=f"Remove or correct values above {constraint.max_value}"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.RANGE_VIOLATION,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Range validation error: {str(e)}",
                details={'error': str(e)}
            ))
        
        return issues
    
    def _validate_string_constraints(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate string length constraints."""
        issues = []
        column_name = constraint.name
        
        try:
            string_series = series.astype(str)
            lengths = string_series.str.len()
            
            if constraint.min_length is not None:
                too_short = lengths < constraint.min_length
                if too_short.any():
                    short_indices = series[too_short].index.tolist()
                    issues.append(ValidationIssue(
                        issue_type=IssueType.FORMAT_ERROR,
                        severity=ValidationStatus.WARNING,
                        column=column_name,
                        row_indices=short_indices[:50],
                        message=f"Column '{column_name}' has {too_short.sum()} values shorter than minimum length {constraint.min_length}",
                        details={'min_length': constraint.min_length, 'violation_count': too_short.sum()}
                    ))
            
            if constraint.max_length is not None:
                too_long = lengths > constraint.max_length
                if too_long.any():
                    long_indices = series[too_long].index.tolist()
                    issues.append(ValidationIssue(
                        issue_type=IssueType.FORMAT_ERROR,
                        severity=ValidationStatus.WARNING,
                        column=column_name,
                        row_indices=long_indices[:50],
                        message=f"Column '{column_name}' has {too_long.sum()} values longer than maximum length {constraint.max_length}",
                        details={'max_length': constraint.max_length, 'violation_count': too_long.sum()}
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.FORMAT_ERROR,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"String constraint validation error: {str(e)}"
            ))
        
        return issues
    
    def _validate_pattern(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate regex pattern constraints."""
        issues = []
        column_name = constraint.name
        
        try:
            pattern = re.compile(constraint.pattern)
            string_series = series.astype(str)
            matches = string_series.str.match(pattern, na=False)
            
            if not matches.all():
                no_match_indices = series[~matches].index.tolist()
                issues.append(ValidationIssue(
                    issue_type=IssueType.FORMAT_ERROR,
                    severity=ValidationStatus.ERROR,
                    column=column_name,
                    row_indices=no_match_indices[:50],
                    message=f"Column '{column_name}' has {(~matches).sum()} values not matching pattern '{constraint.pattern}'",
                    details={'pattern': constraint.pattern, 'violation_count': (~matches).sum()},
                    suggestion=f"Ensure values match the pattern: {constraint.pattern}"
                ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.FORMAT_ERROR,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Pattern validation error: {str(e)}"
            ))
        
        return issues
    
    def _validate_allowed_values(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate allowed values constraint."""
        issues = []
        column_name = constraint.name
        
        try:
            allowed_values = constraint.allowed_values
            string_series = series.astype(str)
            invalid_mask = ~string_series.isin(allowed_values)
            
            if invalid_mask.any():
                invalid_indices = series[invalid_mask].index.tolist()
                invalid_values = string_series[invalid_mask].unique()[:10]  # Show first 10 unique invalid values
                
                issues.append(ValidationIssue(
                    issue_type=IssueType.CONSTRAINT_VIOLATION,
                    severity=ValidationStatus.ERROR,
                    column=column_name,
                    row_indices=invalid_indices[:50],
                    message=f"Column '{column_name}' has {invalid_mask.sum()} invalid values not in allowed set",
                    details={
                        'allowed_values': list(allowed_values),
                        'invalid_values': invalid_values.tolist(),
                        'violation_count': invalid_mask.sum()
                    },
                    suggestion=f"Use only allowed values: {list(allowed_values)}"
                ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.CONSTRAINT_VIOLATION,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Allowed values validation error: {str(e)}"
            ))
        
        return issues
    
    def _validate_uniqueness(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate uniqueness constraint."""
        issues = []
        column_name = constraint.name
        
        try:
            duplicates = series.duplicated(keep=False)
            if duplicates.any():
                duplicate_indices = series[duplicates].index.tolist()
                duplicate_values = series[duplicates].unique()[:10]
                
                issues.append(ValidationIssue(
                    issue_type=IssueType.DUPLICATE_ROWS,
                    severity=ValidationStatus.ERROR,
                    column=column_name,
                    row_indices=duplicate_indices[:100],
                    message=f"Column '{column_name}' has {duplicates.sum()} duplicate values but should be unique",
                    details={
                        'duplicate_count': duplicates.sum(),
                        'duplicate_values': duplicate_values.tolist() if hasattr(duplicate_values, 'tolist') else list(duplicate_values)
                    },
                    suggestion="Remove duplicate values to ensure uniqueness"
                ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.DUPLICATE_ROWS,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Uniqueness validation error: {str(e)}"
            ))
        
        return issues
    
    def _validate_custom(self, series: pd.Series, constraint: ColumnConstraint) -> List[ValidationIssue]:
        """Validate using custom validation function."""
        issues = []
        column_name = constraint.name
        
        try:
            validation_result = constraint.custom_validator(series)
            
            if isinstance(validation_result, bool):
                if not validation_result:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.CONSTRAINT_VIOLATION,
                        severity=ValidationStatus.ERROR,
                        column=column_name,
                        message=f"Column '{column_name}' failed custom validation",
                        suggestion="Review column values against custom business rules"
                    ))
            elif isinstance(validation_result, tuple):
                is_valid, invalid_indices, message = validation_result
                if not is_valid:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.CONSTRAINT_VIOLATION,
                        severity=ValidationStatus.ERROR,
                        column=column_name,
                        row_indices=invalid_indices[:50] if invalid_indices else None,
                        message=message or f"Column '{column_name}' failed custom validation"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.CONSTRAINT_VIOLATION,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Custom validation error: {str(e)}"
            ))
        
        return issues
    
    # Type validation helper methods
    def _is_integer_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains integer values."""
        try:
            converted = pd.to_numeric(series, errors='coerce')
            is_int = (converted == converted.astype(int, errors='ignore')).fillna(False)
            invalid_indices = series[~is_int].index.tolist()
            return len(invalid_indices) == 0, invalid_indices
        except Exception:
            return False, list(range(len(series)))
    
    def _is_numeric_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains numeric values."""
        try:
            converted = pd.to_numeric(series, errors='coerce')
            invalid_mask = converted.isnull() & series.notnull()
            invalid_indices = series[invalid_mask].index.tolist()
            return len(invalid_indices) == 0, invalid_indices
        except Exception:
            return False, list(range(len(series)))
    
    def _is_string_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains string values."""
        # Most values can be converted to string, so this is usually True
        return True, []
    
    def _is_boolean_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains boolean values."""
        try:
            boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
            string_series = series.astype(str).str.lower()
            is_boolean = string_series.isin(boolean_values) | series.isin([True, False, 1, 0])
            invalid_indices = series[~is_boolean].index.tolist()
            return len(invalid_indices) == 0, invalid_indices
        except Exception:
            return False, list(range(len(series)))
    
    def _is_datetime_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains datetime values."""
        try:
            invalid_indices = []
            for idx, value in series.items():
                try:
                    pd.to_datetime(value)
                except (ValueError, TypeError):
                    invalid_indices.append(idx)
            return len(invalid_indices) == 0, invalid_indices
        except Exception:
            return False, list(range(len(series)))
    
    def _is_date_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains date values."""
        # Similar to datetime but more lenient
        return self._is_datetime_type(series)
    
    def _is_email_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains email addresses."""
        try:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            string_series = series.astype(str)
            is_email = string_series.str.match(email_pattern, na=False)
            invalid_indices = series[~is_email].index.tolist()
            return len(invalid_indices) == 0, invalid_indices
        except Exception:
            return False, list(range(len(series)))
    
    def _is_url_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains URLs."""
        try:
            url_pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
            string_series = series.astype(str)
            is_url = string_series.str.match(url_pattern, na=False)
            invalid_indices = series[~is_url].index.tolist()
            return len(invalid_indices) == 0, invalid_indices
        except Exception:
            return False, list(range(len(series)))
    
    def _is_categorical_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series is suitable for categorical type."""
        # Categorical is usually valid for most data
        return True, []
    
    def _is_currency_type(self, series: pd.Series) -> Tuple[bool, List[int]]:
        """Check if series contains currency values."""
        try:
            currency_pattern = r'^\$?[\d,]+\.?\d{0,2}$'
            string_series = series.astype(str)
            is_currency = string_series.str.match(currency_pattern, na=False)
            invalid_indices = series[~is_currency].index.tolist()
            return len(invalid_indices) == 0, invalid_indices
        except Exception:
            return False, list(range(len(series)))

class QualityValidator:
    """Handles data quality validation and assessment."""
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize quality validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """
        Perform comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (quality_score, list_of_issues)
        """
        issues = []
        quality_components = []
        
        try:
            # Missing values assessment
            missing_score, missing_issues = self._assess_missing_values(df)
            issues.extend(missing_issues)
            quality_components.append(missing_score)
            
            # Duplicate rows assessment
            duplicate_score, duplicate_issues = self._assess_duplicates(df)
            issues.extend(duplicate_issues)
            quality_components.append(duplicate_score)
            
            # Outliers assessment
            if self.config.outlier_detection:
                outlier_score, outlier_issues = self._assess_outliers(df)
                issues.extend(outlier_issues)
                quality_components.append(outlier_score)
            
            # Consistency assessment
            consistency_score, consistency_issues = self._assess_consistency(df)
            issues.extend(consistency_issues)
            quality_components.append(consistency_score)
            
            # Calculate overall quality score
            overall_quality = np.mean(quality_components) if quality_components else 0.5
            
            return overall_quality, issues
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {str(e)}")
            issues.append(ValidationIssue(
                issue_type=IssueType.DATA_QUALITY,
                severity=ValidationStatus.CRITICAL,
                message=f"Data quality validation error: {str(e)}",
                details={'error': str(e)}
            ))
            return 0.0, issues
    
    def _assess_missing_values(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """Assess missing values in the dataset."""
        issues = []
        
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
            
            # Column-wise missing value analysis
            column_missing = df.isnull().sum() / len(df)
            high_missing_cols = column_missing[column_missing > self.config.missing_value_threshold].index.tolist()
            
            # Create issues for high missing value columns
            for col in high_missing_cols:
                missing_count = df[col].isnull().sum()
                missing_indices = df[df[col].isnull()].index.tolist()
                
                issues.append(ValidationIssue(
                    issue_type=IssueType.MISSING_VALUES,
                    severity=ValidationStatus.CRITICAL if column_missing[col] > 0.8 else ValidationStatus.WARNING,
                    column=col,
                    row_indices=missing_indices[:100],
                    message=f"Column '{col}' has {missing_count} missing values ({column_missing[col]:.1%})",
                    details={
                        'missing_count': missing_count,
                        'missing_ratio': column_missing[col],
                        'total_rows': len(df)
                    },
                    suggestion="Consider imputation, removal, or data collection for missing values"
                ))
            
            # Overall missing value score
            missing_score = max(0, 1 - missing_ratio * 2)  # Penalize missing values
            
            return missing_score, issues
            
        except Exception as e:
            self.logger.error(f"Missing values assessment failed: {str(e)}")
            return 0.5, issues
    
    def _assess_duplicates(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """Assess duplicate rows in the dataset."""
        issues = []
        
        try:
            total_rows = len(df)
            duplicate_rows = df.duplicated().sum()
            duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0
            
            if duplicate_ratio > self.config.duplicate_threshold:
                duplicate_indices = df[df.duplicated(keep=False)].index.tolist()
                
                issues.append(ValidationIssue(
                    issue_type=IssueType.DUPLICATE_ROWS,
                    severity=ValidationStatus.WARNING if duplicate_ratio < 0.5 else ValidationStatus.ERROR,
                    row_indices=duplicate_indices[:100],
                    message=f"Dataset contains {duplicate_rows} duplicate rows ({duplicate_ratio:.1%})",
                    details={
                        'duplicate_count': duplicate_rows,
                        'duplicate_ratio': duplicate_ratio,
                        'total_rows': total_rows
                    },
                    suggestion="Remove duplicate rows to improve data quality",
                    can_auto_fix=True
                ))
            
            # Duplicate score
            duplicate_score = max(0, 1 - duplicate_ratio * 2)
            
            return duplicate_score, issues
            
        except Exception as e:
            self.logger.error(f"Duplicates assessment failed: {str(e)}")
            return 0.5, issues
    
    def _assess_outliers(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """Assess outliers in numeric columns."""
        issues = []
        outlier_ratios = []
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                try:
                    # Z-score based outlier detection
                    z_scores = np.abs(stats.zscore(df[col].dropna())) if SCIPY_AVAILABLE else np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = z_scores > self.config.outlier_threshold
                    outlier_count = outliers.sum()
                    outlier_ratio = outlier_count / len(df[col].dropna()) if len(df[col].dropna()) > 0 else 0
                    
                    outlier_ratios.append(outlier_ratio)
                    
                    if outlier_ratio > 0.05:  # More than 5% outliers
                        outlier_indices = df[col][outliers].index.tolist()
                        
                        issues.append(ValidationIssue(
                            issue_type=IssueType.OUTLIERS,
                            severity=ValidationStatus.WARNING,
                            column=col,
                            row_indices=outlier_indices[:50],
                            message=f"Column '{col}' has {outlier_count} potential outliers ({outlier_ratio:.1%})",
                            details={
                                'outlier_count': outlier_count,
                                'outlier_ratio': outlier_ratio,
                                'threshold': self.config.outlier_threshold,
                                'method': 'z-score'
                            },
                            suggestion="Review outliers - remove if erroneous, keep if valid extreme values"
                        ))
                
                except Exception as col_error:
                    self.logger.warning(f"Outlier detection failed for column {col}: {str(col_error)}")
            
            # Overall outlier score
            avg_outlier_ratio = np.mean(outlier_ratios) if outlier_ratios else 0
            outlier_score = max(0, 1 - avg_outlier_ratio * 4)  # Moderate penalty for outliers
            
            return outlier_score, issues
            
        except Exception as e:
            self.logger.error(f"Outliers assessment failed: {str(e)}")
            return 0.5, issues
    
    def _assess_consistency(self, df: pd.DataFrame) -> Tuple[float, List[ValidationIssue]]:
        """Assess data consistency across the dataset."""
        issues = []
        consistency_factors = []
        
        try:
            # Check for mixed data types in object columns
            object_columns = df.select_dtypes(include=['object']).columns
            
            for col in object_columns:
                try:
                    sample_values = df[col].dropna().head(100)
                    if len(sample_values) > 0:
                        # Check type consistency within the column
                        type_variety = len(set(type(val).__name__ for val in sample_values))
                        type_consistency = 1.0 / type_variety if type_variety > 0 else 1.0
                        consistency_factors.append(type_consistency)
                        
                        if type_variety > 2:  # Mixed types detected
                            issues.append(ValidationIssue(
                                issue_type=IssueType.TYPE_MISMATCH,
                                severity=ValidationStatus.WARNING,
                                column=col,
                                message=f"Column '{col}' contains mixed data types",
                                details={'type_variety': type_variety},
                                suggestion="Ensure consistent data types within columns"
                            ))
                
                except Exception as col_error:
                    self.logger.warning(f"Consistency check failed for column {col}: {str(col_error)}")
            
            # Check for encoding issues
            for col in object_columns:
                try:
                    string_values = df[col].dropna().astype(str)
                    encoding_issues = string_values.str.contains('�', na=False).sum()
                    
                    if encoding_issues > 0:
                        issues.append(ValidationIssue(
                            issue_type=IssueType.FORMAT_ERROR,
                            severity=ValidationStatus.WARNING,
                            column=col,
                            message=f"Column '{col}' may have encoding issues ({encoding_issues} problematic values)",
                            details={'encoding_issue_count': encoding_issues},
                            suggestion="Check file encoding and re-import if necessary"
                        ))
                
                except Exception as col_error:
                    self.logger.warning(f"Encoding check failed for column {col}: {str(col_error)}")
            
            # Overall consistency score
            consistency_score = np.mean(consistency_factors) if consistency_factors else 1.0
            
            return consistency_score, issues
            
        except Exception as e:
            self.logger.error(f"Consistency assessment failed: {str(e)}")
            return 0.5, issues

class TypeConverter:
    """Handles safe type conversion and inference."""
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize type converter.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def convert_types(
        self,
        df: pd.DataFrame,
        type_hints: Optional[Dict[str, DataType]] = None,
        infer_types: bool = True
    ) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
        """
        Convert DataFrame column types with validation.
        
        Args:
            df: DataFrame to convert
            type_hints: Explicit type hints for columns
            infer_types: Whether to infer types automatically
            
        Returns:
            Tuple of (converted_dataframe, conversion_issues)
        """
        issues = []
        converted_df = df.copy()
        
        try:
            # Apply explicit type hints first
            if type_hints:
                for col, target_type in type_hints.items():
                    if col in converted_df.columns:
                        converted_col, col_issues = self._convert_column_type(
                            converted_df[col], col, target_type
                        )
                        converted_df[col] = converted_col
                        issues.extend(col_issues)
            
            # Infer and convert remaining columns
            if infer_types:
                for col in converted_df.columns:
                    if type_hints is None or col not in type_hints:
                        inferred_type = self._infer_column_type(converted_df[col])
                        if inferred_type != DataType.STRING:  # Don't convert if already string
                            converted_col, col_issues = self._convert_column_type(
                                converted_df[col], col, inferred_type
                            )
                            converted_df[col] = converted_col
                            issues.extend(col_issues)
            
            return converted_df, issues
            
        except Exception as e:
            self.logger.error(f"Type conversion failed: {str(e)}")
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                message=f"Type conversion error: {str(e)}",
                details={'error': str(e)}
            ))
            return df, issues
    
    def _convert_column_type(
        self,
        series: pd.Series,
        column_name: str,
        target_type: DataType
    ) -> Tuple[pd.Series, List[ValidationIssue]]:
        """Convert a single column to target type."""
        issues = []
        
        try:
            if target_type == DataType.INTEGER:
                return self._convert_to_integer(series, column_name)
            elif target_type == DataType.FLOAT:
                return self._convert_to_float(series, column_name)
            elif target_type == DataType.BOOLEAN:
                return self._convert_to_boolean(series, column_name)
            elif target_type == DataType.DATETIME:
                return self._convert_to_datetime(series, column_name)
            elif target_type == DataType.CATEGORICAL:
                return self._convert_to_categorical(series, column_name)
            elif target_type == DataType.STRING:
                return self._convert_to_string(series, column_name)
            else:
                # Unsupported conversion
                issues.append(ValidationIssue(
                    issue_type=IssueType.TYPE_MISMATCH,
                    severity=ValidationStatus.WARNING,
                    column=column_name,
                    message=f"Unsupported type conversion to {target_type.value}",
                    suggestion="Use supported data types for conversion"
                ))
                return series, issues
                
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Type conversion failed: {str(e)}",
                details={'target_type': target_type.value, 'error': str(e)}
            ))
            return series, issues
    
    def _convert_to_integer(self, series: pd.Series, column_name: str) -> Tuple[pd.Series, List[ValidationIssue]]:
        """Convert series to integer type."""
        issues = []
        
        try:
            # First convert to numeric, handling errors
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Check for conversion losses
            conversion_losses = numeric_series.isnull() & series.notnull()
            if conversion_losses.any():
                loss_count = conversion_losses.sum()
                issues.append(ValidationIssue(
                    issue_type=IssueType.TYPE_MISMATCH,
                    severity=ValidationStatus.WARNING,
                    column=column_name,
                    message=f"Lost {loss_count} values during integer conversion",
                    details={'conversion_losses': loss_count},
                    suggestion="Review non-numeric values that couldn't be converted"
                ))
            
            # Convert to integer, handling NaN values
            integer_series = numeric_series.astype('Int64')  # Nullable integer type
            
            return integer_series, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Integer conversion failed: {str(e)}"
            ))
            return series, issues
    
    def _convert_to_float(self, series: pd.Series, column_name: str) -> Tuple[pd.Series, List[ValidationIssue]]:
        """Convert series to float type."""
        issues = []
        
        try:
            # Convert to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Check for conversion losses
            conversion_losses = numeric_series.isnull() & series.notnull()
            if conversion_losses.any():
                loss_count = conversion_losses.sum()
                issues.append(ValidationIssue(
                    issue_type=IssueType.TYPE_MISMATCH,
                    severity=ValidationStatus.WARNING,
                    column=column_name,
                    message=f"Lost {loss_count} values during float conversion",
                    details={'conversion_losses': loss_count}
                ))
            
            return numeric_series, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Float conversion failed: {str(e)}"
            ))
            return series, issues
    
    def _convert_to_boolean(self, series: pd.Series, column_name: str) -> Tuple[pd.Series, List[ValidationIssue]]:
        """Convert series to boolean type."""
        issues = []
        
        try:
            # Define boolean mappings
            true_values = {'true', 't', 'yes', 'y', '1', 1, True}
            false_values = {'false', 'f', 'no', 'n', '0', 0, False}
            
            # Convert to string for consistent processing
            string_series = series.astype(str).str.lower().str.strip()
            
            # Map to boolean values
            boolean_series = series.copy()
            boolean_series[string_series.isin([str(v).lower() for v in true_values])] = True
            boolean_series[string_series.isin([str(v).lower() for v in false_values])] = False
            
            # Check for unmappable values
            mapped_mask = string_series.isin([str(v).lower() for v in true_values | false_values])
            unmapped_count = (~mapped_mask & series.notnull()).sum()
            
            if unmapped_count > 0:
                issues.append(ValidationIssue(
                    issue_type=IssueType.TYPE_MISMATCH,
                    severity=ValidationStatus.WARNING,
                    column=column_name,
                    message=f"Could not map {unmapped_count} values to boolean",
                    details={'unmapped_count': unmapped_count},
                    suggestion="Use standard boolean values: true/false, yes/no, 1/0"
                ))
            
            # Convert to boolean type
            boolean_series = boolean_series.astype(bool)
            
            return boolean_series, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Boolean conversion failed: {str(e)}"
            ))
            return series, issues
    
    def _convert_to_datetime(self, series: pd.Series, column_name: str) -> Tuple[pd.Series, List[ValidationIssue]]:
        """Convert series to datetime type."""
        issues = []
        
        try:
            # Try pandas automatic datetime parsing first
            try:
                datetime_series = pd.to_datetime(series, errors='coerce')
            except Exception:
                datetime_series = None
            
            # If automatic parsing fails, try with format hints
            if datetime_series is None or datetime_series.isnull().all():
                datetime_series = None
                
                for fmt in self.config.date_formats:
                    try:
                        datetime_series = pd.to_datetime(series, format=fmt, errors='coerce')
                        if not datetime_series.isnull().all():
                            break
                    except Exception:
                        continue
            
            # Final fallback: try dateutil parser
            if datetime_series is None or datetime_series.isnull().all():
                def parse_date(x):
                    try:
                        return date_parser.parse(str(x)) if pd.notnull(x) else None
                    except Exception:
                        return None
                
                datetime_series = series.apply(parse_date)
                datetime_series = pd.to_datetime(datetime_series, errors='coerce')
            
            # Check for conversion losses
            if datetime_series is not None:
                conversion_losses = datetime_series.isnull() & series.notnull()
                if conversion_losses.any():
                    loss_count = conversion_losses.sum()
                    issues.append(ValidationIssue(
                        issue_type=IssueType.TYPE_MISMATCH,
                        severity=ValidationStatus.WARNING,
                        column=column_name,
                        message=f"Could not parse {loss_count} values as datetime",
                        details={'conversion_losses': loss_count},
                        suggestion="Check date format and provide format hints if needed"
                    ))
                
                return datetime_series, issues
            else:
                issues.append(ValidationIssue(
                    issue_type=IssueType.TYPE_MISMATCH,
                    severity=ValidationStatus.ERROR,
                    column=column_name,
                    message="Datetime conversion completely failed",
                    suggestion="Provide valid datetime values or format hints"
                ))
                return series, issues
                
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Datetime conversion failed: {str(e)}"
            ))
            return series, issues
    
    def _convert_to_categorical(self, series: pd.Series, column_name: str) -> Tuple[pd.Series, List[ValidationIssue]]:
        """Convert series to categorical type."""
        issues = []
        
        try:
            # Convert to categorical
            categorical_series = series.astype('category')
            
            # Report on category count
            n_categories = categorical_series.nunique()
            issues.append(ValidationIssue(
                issue_type=IssueType.DATA_QUALITY,
                severity=ValidationStatus.INFO,
                column=column_name,
                message=f"Column converted to categorical with {n_categories} categories",
                details={'category_count': n_categories}
            ))
            
            return categorical_series, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"Categorical conversion failed: {str(e)}"
            ))
            return series, issues
    
    def _convert_to_string(self, series: pd.Series, column_name: str) -> Tuple[pd.Series, List[ValidationIssue]]:
        """Convert series to string type."""
        issues = []
        
        try:
            # Convert to string
            string_series = series.astype(str)
            
            return string_series, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=ValidationStatus.ERROR,
                column=column_name,
                message=f"String conversion failed: {str(e)}"
            ))
            return series, issues
    
    def _infer_column_type(self, series: pd.Series) -> DataType:
        """Infer the most appropriate data type for a column."""
        try:
            # Skip empty or all-null columns
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                return DataType.STRING
            
            # Try integer first
            try:
                converted = pd.to_numeric(non_null_series, errors='coerce')
                if not converted.isnull().any():
                    # Check if all values are integers
                    if (converted == converted.astype(int)).all():
                        return DataType.INTEGER
                    else:
                        return DataType.FLOAT
            except Exception:
                pass
            
            # Try datetime
            try:
                datetime_converted = pd.to_datetime(non_null_series, errors='coerce')
                if not datetime_converted.isnull().any():
                    return DataType.DATETIME
            except Exception:
                pass
            
            # Check for boolean patterns
            if len(non_null_series) > 0:
                sample_str = non_null_series.astype(str).str.lower()
                boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
                if sample_str.isin(boolean_values).all():
                    return DataType.BOOLEAN
            
            # Check if suitable for categorical
            unique_ratio = len(non_null_series.unique()) / len(non_null_series)
            if unique_ratio < 0.5:  # Less than 50% unique values
                return DataType.CATEGORICAL
            
            # Default to string
            return DataType.STRING
            
        except Exception:
            return DataType.STRING

class ValidationReporter:
    """Generates comprehensive validation reports."""
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize validation reporter.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_report(
        self,
        result: ValidationResult,
        format_type: str = "dict"
    ) -> Union[Dict[str, Any], str]:
        """
        Generate comprehensive validation report.
        
        Args:
            result: Validation result to report on
            format_type: Output format ('dict', 'json', 'text')
            
        Returns:
            Formatted validation report
        """
        try:
            if format_type == "dict":
                return self._generate_dict_report(result)
            elif format_type == "json":
                return json.dumps(self._generate_dict_report(result), indent=2, default=str)
            elif format_type == "text":
                return self._generate_text_report(result)
            else:
                raise ValueError(f"Unsupported report format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {"error": f"Report generation failed: {str(e)}"}
    
    def _generate_dict_report(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate dictionary format report."""
        report = {
            "validation_summary": {
                "validation_id": result.validation_id,
                "timestamp": result.timestamp.isoformat(),
                "overall_status": result.status.value,
                "dataset_info": {
                    "shape": result.dataset_shape,
                    "size_mb": result.dataset_size_mb,
                    "columns_validated": result.columns_validated,
                    "rows_validated": result.rows_validated
                },
                "issue_counts": {
                    "total_issues": len(result.issues),
                    "critical": result.critical_count,
                    "errors": result.errors_count,
                    "warnings": result.warnings_count
                },
                "quality_metrics": {
                    "data_quality_score": result.data_quality_score,
                    "missing_value_ratio": result.missing_value_ratio,
                    "duplicate_ratio": result.duplicate_ratio,
                    "outlier_ratio": result.outlier_ratio
                },
                "performance": {
                    "validation_duration": result.validation_duration,
                    "memory_usage_mb": result.memory_usage_mb
                }
            },
            "schema_compliance": {
                "schema_valid": result.schema_valid,
                "missing_columns": result.missing_columns,
                "extra_columns": result.extra_columns,
                "type_mismatches": result.type_mismatches
            },
            "issues": [
                {
                    "id": issue.id,
                    "type": issue.issue_type.value,
                    "severity": issue.severity.value,
                    "column": issue.column,
                    "message": issue.message,
                    "row_count": len(issue.row_indices) if issue.row_indices else 0,
                    "details": issue.details,
                    "suggestion": issue.suggestion,
                    "can_auto_fix": issue.can_auto_fix,
                    "timestamp": issue.timestamp.isoformat()
                }
                for issue in result.issues
            ],
            "recommendations": result.recommendations,
            "auto_fixes_available": result.auto_fixes_available
        }
        
        # Add column statistics if available
        if result.column_stats and self.config.include_statistics:
            report["column_statistics"] = result.column_stats
        
        return report
    
    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate human-readable text report."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("DATA VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Validation ID: {result.validation_id}")
        lines.append(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Overall Status: {result.status.value.upper()}")
        lines.append("")
        
        # Dataset Information
        lines.append("DATASET INFORMATION")
        lines.append("-" * 20)
        lines.append(f"Shape: {result.dataset_shape[0]:,} rows × {result.dataset_shape[1]} columns")
        lines.append(f"Size: {result.dataset_size_mb:.2f} MB")
        lines.append(f"Validation Duration: {result.validation_duration:.2f} seconds")
        lines.append("")
        
        # Issue Summary
        lines.append("ISSUE SUMMARY")
        lines.append("-" * 15)
        lines.append(f"Total Issues: {len(result.issues)}")
        lines.append(f"  Critical: {result.critical_count}")
        lines.append(f"  Errors: {result.errors_count}")
        lines.append(f"  Warnings: {result.warnings_count}")
        lines.append("")
        
        # Quality Metrics
        lines.append("QUALITY METRICS")
        lines.append("-" * 17)
        lines.append(f"Overall Quality Score: {result.data_quality_score:.2f}/1.0")
        lines.append(f"Missing Value Ratio: {result.missing_value_ratio:.1%}")
        lines.append(f"Duplicate Ratio: {result.duplicate_ratio:.1%}")
        lines.append(f"Outlier Ratio: {result.outlier_ratio:.1%}")
        lines.append("")
        
        # Schema Compliance
        if not result.schema_valid:
            lines.append("SCHEMA COMPLIANCE ISSUES")
            lines.append("-" * 25)
            if result.missing_columns:
                lines.append(f"Missing Required Columns: {', '.join(result.missing_columns)}")
            if result.extra_columns:
                lines.append(f"Extra Columns: {', '.join(result.extra_columns)}")
            if result.type_mismatches:
                lines.append("Type Mismatches:")
                for col, issue in result.type_mismatches.items():
                    lines.append(f"  {col}: {issue}")
            lines.append("")
        
        # Detailed Issues
        if result.issues:
            lines.append("DETAILED ISSUES")
            lines.append("-" * 16)
            
            # Group issues by severity
            critical_issues = [i for i in result.issues if i.severity == ValidationStatus.CRITICAL]
            error_issues = [i for i in result.issues if i.severity == ValidationStatus.ERROR]
            warning_issues = [i for i in result.issues if i.severity == ValidationStatus.WARNING]
            
            for severity, issues in [("CRITICAL", critical_issues), ("ERRORS", error_issues), ("WARNINGS", warning_issues)]:
                if issues:
                    lines.append(f"\n{severity}:")
                    for issue in issues[:10]:  # Limit to first 10 per severity
                        lines.append(f"  • {issue.message}")
                        if issue.column:
                            lines.append(f"    Column: {issue.column}")
                        if issue.suggestion and self.config.include_suggestions:
                            lines.append(f"    Suggestion: {issue.suggestion}")
                        lines.append("")
                    
                    if len(issues) > 10:
                        lines.append(f"  ... and {len(issues) - 10} more {severity.lower()}")
                        lines.append("")
        
        # Recommendations
        if result.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 15)
            for i, rec in enumerate(result.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Footer
        lines.append("=" * 60)
        
        return "\n".join(lines)

class DataValidator:
    """Main data validation orchestrator."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize data validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.schema_validator = SchemaValidator(self.config)
        self.quality_validator = QualityValidator(self.config)
        self.type_converter = TypeConverter(self.config)
        self.reporter = ValidationReporter(self.config)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @monitor_performance("dataset_validation") if MONITORING_AVAILABLE else lambda x: x
    def validate_dataset(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict[str, ColumnConstraint]] = None,
        convert_types: bool = True
    ) -> ValidationResult:
        """
        Perform comprehensive dataset validation.
        
        Args:
            df: DataFrame to validate
            schema: Schema constraints (optional)
            convert_types: Whether to perform type conversion
            
        Returns:
            Comprehensive validation result
        """
        start_time = time.time()
        result = ValidationResult()
        
        try:
            self.logger.info(f"Starting validation for dataset: {df.shape}")
            
            # Basic dataset information
            result.dataset_shape = df.shape
            result.dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            result.columns_validated = len(df.columns)
            result.rows_validated = len(df)
            
            # Sample dataset if too large
            validation_df = self._sample_dataset(df)
            
            # Schema validation
            if schema:
                schema_valid, schema_issues = self.schema_validator.validate_schema(validation_df, schema)
                result.schema_valid = schema_valid
                result.issues.extend(schema_issues)
                
                # Extract schema-specific metrics
                result.missing_columns = [
                    issue.column for issue in schema_issues 
                    if issue.issue_type == IssueType.MISSING_COLUMN and issue.column
                ]
                result.extra_columns = [
                    issue.column for issue in schema_issues 
                    if issue.issue_type == IssueType.EXTRA_COLUMN and issue.column
                ]
                result.type_mismatches = {
                    issue.column: issue.message for issue in schema_issues
                    if issue.issue_type == IssueType.TYPE_MISMATCH and issue.column
                }
            
            # Data quality validation
            quality_score, quality_issues = self.quality_validator.validate_data_quality(validation_df)
            result.data_quality_score = quality_score
            result.issues.extend(quality_issues)
            
            # Type conversion if requested
            if convert_types:
                converted_df, conversion_issues = self.type_converter.convert_types(validation_df)
                result.issues.extend(conversion_issues)
                # Note: In production, you might want to return the converted DataFrame
            
            # Calculate quality metrics
            result.missing_value_ratio = validation_df.isnull().sum().sum() / (validation_df.shape[0] * validation_df.shape[1])
            result.duplicate_ratio = validation_df.duplicated().sum() / len(validation_df)
            
            # Outlier ratio calculation for numeric columns
            numeric_cols = validation_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and SCIPY_AVAILABLE:
                total_outliers = 0
                total_numeric_values = 0
                for col in numeric_cols:
                    col_data = validation_df[col].dropna()
                    if len(col_data) > 0:
                        z_scores = np.abs(stats.zscore(col_data))
                        outliers = (z_scores > self.config.outlier_threshold).sum()
                        total_outliers += outliers
                        total_numeric_values += len(col_data)
                
                result.outlier_ratio = total_outliers / total_numeric_values if total_numeric_values > 0 else 0.0
            
            # Generate column statistics
            if self.config.include_statistics:
                result.column_stats = self._generate_column_stats(validation_df)
            
            # Count issues by severity
            result.critical_count = len([i for i in result.issues if i.severity == ValidationStatus.CRITICAL])
            result.errors_count = len([i for i in result.issues if i.severity == ValidationStatus.ERROR])
            result.warnings_count = len([i for i in result.issues if i.severity == ValidationStatus.WARNING])
            
            # Determine overall status
            if result.critical_count > 0:
                result.status = ValidationStatus.CRITICAL
            elif result.errors_count > 0:
                result.status = ValidationStatus.ERROR
            elif result.warnings_count > 0:
                result.status = ValidationStatus.WARNING
            else:
                result.status = ValidationStatus.PASSED
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)
            result.auto_fixes_available = any(issue.can_auto_fix for issue in result.issues)
            
            # Performance metrics
            result.validation_duration = time.time() - start_time
            result.memory_usage_mb = validation_df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Log validation completion
            self.logger.info(
                f"Validation completed: Status={result.status.value}, "
                f"Issues={len(result.issues)}, Duration={result.validation_duration:.2f}s"
            )
            
            # Log metrics if monitoring is available
            if MONITORING_AVAILABLE:
                log_info(
                    f"Dataset validation completed",
                    extra={
                        'validation_id': result.validation_id,
                        'dataset_shape': result.dataset_shape,
                        'status': result.status.value,
                        'issue_count': len(result.issues),
                        'duration': result.validation_duration
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            
            # Create error result
            result.status = ValidationStatus.CRITICAL
            result.validation_duration = time.time() - start_time
            result.issues.append(ValidationIssue(
                issue_type=IssueType.SCHEMA_ERROR,
                severity=ValidationStatus.CRITICAL,
                message=f"Validation process failed: {str(e)}",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            ))
            
            if MONITORING_AVAILABLE:
                log_error(
                    f"Dataset validation failed",
                    exception=e,
                    extra={'dataset_shape': df.shape if df is not None else None}
                )
            
            return result
    
    def _sample_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample dataset if too large for efficient validation."""
        if self.config.sample_size and len(df) > self.config.sample_size:
            self.logger.info(f"Sampling dataset from {len(df)} to {self.config.sample_size} rows")
            return df.sample(n=self.config.sample_size, random_state=42)
        return df
    
    def _generate_column_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate detailed statistics for each column."""
        stats = {}
        
        for col in df.columns:
            try:
                col_stats = {
                    'dtype': str(df[col].dtype),
                    'non_null_count': df[col].count(),
                    'null_count': df[col].isnull().sum(),
                    'unique_count': df[col].nunique(),
                    'memory_usage': df[col].memory_usage(deep=True)
                }
                
                # Numeric statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe()
                    col_stats.update({
                        'mean': desc['mean'],
                        'std': desc['std'],
                        'min': desc['min'],
                        'max': desc['max'],
                        'median': desc['50%'],
                        'q1': desc['25%'],
                        'q3': desc['75%']
                    })
                
                # Categorical statistics
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    value_counts = df[col].value_counts()
                    col_stats.update({
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                        'category_distribution': value_counts.head(10).to_dict()
                    })
                
                stats[col] = col_stats
                
            except Exception as e:
                self.logger.warning(f"Failed to generate stats for column {col}: {str(e)}")
                stats[col] = {'error': str(e)}
        
        return stats
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        try:
            # Critical issues recommendations
            if result.critical_count > 0:
                recommendations.append("Address critical issues immediately before proceeding with analysis")
            
            # Missing columns recommendations
            if result.missing_columns:
                recommendations.append(f"Add missing required columns: {', '.join(result.missing_columns)}")
            
            # Data quality recommendations
            if result.missing_value_ratio > 0.3:
                recommendations.append("High missing value ratio detected - consider data imputation or collection")
            
            if result.duplicate_ratio > 0.1:
                recommendations.append("Significant duplicate rows detected - consider deduplication")
            
            if result.outlier_ratio > 0.1:
                recommendations.append("Many outliers detected - review for data entry errors or valid extreme values")
            
            # Type conversion recommendations
            type_issues = [i for i in result.issues if i.issue_type == IssueType.TYPE_MISMATCH]
            if type_issues and any(i.can_auto_fix for i in type_issues):
                recommendations.append("Enable automatic type conversion to resolve type mismatches")
            
            # Performance recommendations
            if result.dataset_size_mb > 500:  # Large dataset
                recommendations.append("Large dataset detected - consider data sampling or chunked processing")
            
            if result.validation_duration > 60:  # Long validation time
                recommendations.append("Long validation time - consider optimizing data preprocessing pipeline")
            
            # Quality score recommendations
            if result.data_quality_score < 0.7:
                recommendations.append("Low data quality score - comprehensive data cleaning recommended")
            elif result.data_quality_score < 0.9:
                recommendations.append("Moderate data quality - minor cleaning may improve analysis results")
            
            # Auto-fix recommendations
            if result.auto_fixes_available:
                recommendations.append("Automatic fixes are available for some issues - enable auto-correction")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate recommendations: {str(e)}")
            recommendations.append("Review validation results and address identified issues")
        
        return recommendations

# Convenience functions for easy usage

def validate_dataset(
    df: pd.DataFrame,
    schema: Optional[Dict[str, ColumnConstraint]] = None,
    config: Optional[ValidationConfig] = None
) -> ValidationResult:
    """
    Quick dataset validation function.
    
    Args:
        df: DataFrame to validate
        schema: Optional schema constraints
        config: Optional validation configuration
        
    Returns:
        Validation result
    """
    validator = DataValidator(config)
    return validator.validate_dataset(df, schema)

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
        logger.error(f"File format validation failed: {str(e)}")
        return False

def validate_column_types(
    df: pd.DataFrame,
    type_mapping: Dict[str, str]
) -> Tuple[bool, List[str]]:
    """
    Validate column types match expectations.
    
    Args:
        df: DataFrame to validate
        type_mapping: Column name to expected type mapping
        
    Returns:
        Tuple of (all_valid, list_of_mismatches)
    """
    try:
        mismatches = []
        
        for col, expected_type in type_mapping.items():
            if col not in df.columns:
                mismatches.append(f"Column '{col}' not found")
                continue
            
            actual_type = str(df[col].dtype)
            
            # Simple type matching - could be enhanced
            type_matches = {
                'int': 'int' in actual_type,
                'float': 'float' in actual_type,
                'str': 'object' in actual_type,
                'datetime': 'datetime' in actual_type,
                'bool': 'bool' in actual_type
            }
            
            if not type_matches.get(expected_type.lower(), False):
                mismatches.append(f"Column '{col}': expected {expected_type}, got {actual_type}")
        
        return len(mismatches) == 0, mismatches
        
    except Exception as e:
        logger.error(f"Column type validation failed: {str(e)}")
        return False, [f"Validation error: {str(e)}"]

def validate_data_quality(df: pd.DataFrame, quality_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Quick data quality assessment.
    
    Args:
        df: DataFrame to assess
        quality_threshold: Minimum acceptable quality score
        
    Returns:
        Quality assessment results
    """
    try:
        config = ValidationConfig()
        quality_validator = QualityValidator(config)
        
        quality_score, issues = quality_validator.validate_data_quality(df)
        
        return {
            'quality_score': quality_score,
            'passes_threshold': quality_score >= quality_threshold,
            'issue_count': len(issues),
            'issues': [
                {
                    'type': issue.issue_type.value,
                    'severity': issue.severity.value,
                    'message': issue.message
                }
                for issue in issues
            ]
        }
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {str(e)}")
        return {
            'quality_score': 0.0,
            'passes_threshold': False,
            'error': str(e)
        }

def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Check for missing values in DataFrame.
    
    Args:
        df: DataFrame to check
        threshold: Threshold for concerning missing value ratio
        
    Returns:
        Missing value analysis
    """
    try:
        missing_counts = df.isnull().sum()
        missing_ratios = missing_counts / len(df)
        
        concerning_columns = missing_ratios[missing_ratios > threshold].to_dict()
        
        return {
            'total_missing_cells': missing_counts.sum(),
            'overall_missing_ratio': missing_counts.sum() / (df.shape[0] * df.shape[1]),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'concerning_columns': concerning_columns,
            'recommendations': [
                f"Column '{col}' has {ratio:.1%} missing values - consider imputation or removal"
                for col, ratio in concerning_columns.items()
            ]
        }
        
    except Exception as e:
        logger.error(f"Missing values check failed: {str(e)}")
        return {'error': str(e)}

def check_duplicate_records(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for duplicate records in DataFrame.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Duplicate analysis results
    """
    try:
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0
        
        # Find columns with all duplicates
        all_duplicate_mask = df.duplicated(keep=False)
        
        return {
            'total_rows': total_rows,
            'duplicate_rows': duplicate_rows,
            'duplicate_ratio': duplicate_ratio,
            'unique_rows': total_rows - duplicate_rows,
            'has_duplicates': duplicate_rows > 0,
            'duplicate_indices': df[df.duplicated()].index.tolist()[:100],  # First 100
            'recommendation': f"Remove {duplicate_rows} duplicate rows" if duplicate_rows > 0 else "No duplicates found"
        }
        
    except Exception as e:
        logger.error(f"Duplicate check failed: {str(e)}")
        return {'error': str(e)}

def sanitize_input(
    value: Any,
    expected_type: Optional[type] = None,
    max_length: Optional[int] = None
) -> Any:
    """
    Sanitize and validate input value.
    
    Args:
        value: Input value to sanitize
        expected_type: Expected type for the value
        max_length: Maximum length for string values
        
    Returns:
        Sanitized value
    """
    try:
        # Handle None values
        if value is None:
            return None
        
        # Type conversion if specified
        if expected_type:
            if expected_type == str:
                value = str(value)
            elif expected_type == int:
                value = int(float(str(value)))  # Handle string numbers
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
            
            # Strip whitespace
            value = value.strip()
        
        return value
        
    except Exception as e:
        logger.warning(f"Input sanitization failed: {str(e)}")
        return str(value) if value is not None else None

def validate_ml_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate ML configuration parameters.
    
    Args:
        config: ML configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Required parameters
        required_params = ['task_type', 'target_column']
        for param in required_params:
            if param not in config:
                errors.append(f"Required parameter '{param}' missing")
        
        # Validate task type
        if 'task_type' in config:
            valid_tasks = ['classification', 'regression', 'clustering', 'anomaly_detection']
            if config['task_type'] not in valid_tasks:
                errors.append(f"Invalid task_type. Must be one of: {valid_tasks}")
        
        # Validate numeric parameters
        numeric_params = {
            'max_models': (1, 20),
            'max_time': (10, 7200),  # 10 seconds to 2 hours
            'test_size': (0.1, 0.5),
            'random_state': (0, 10000)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in config:
                try:
                    value = float(config[param])
                    if not (min_val <= value <= max_val):
                        errors.append(f"Parameter '{param}' must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param}' must be numeric")
        
        # Validate string parameters
        if 'target_column' in config:
            if not isinstance(config['target_column'], str) or not config['target_column'].strip():
                errors.append("Parameter 'target_column' must be a non-empty string")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"ML config validation failed: {str(e)}")
        return False, [f"Configuration validation error: {str(e)}"]

def validate_analysis_request(request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate analysis request parameters.
    
    Args:
        request: Analysis request dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Required fields
        required_fields = ['dataset_id', 'target_column', 'task_type']
        for field in required_fields:
            if field not in request:
                errors.append(f"Required field '{field}' missing")
            elif not request[field]:
                errors.append(f"Field '{field}' cannot be empty")
        
        # Validate dataset_id
        if 'dataset_id' in request:
            try:
                dataset_id = int(request['dataset_id'])
                if dataset_id <= 0:
                    errors.append("Dataset ID must be positive")
            except (ValueError, TypeError):
                errors.append("Dataset ID must be a valid integer")
        
        # Validate execution mode
        if 'execution_mode' in request:
            valid_modes = ['local_cpu', 'local_gpu', 'cloud']
            if request['execution_mode'] not in valid_modes:
                errors.append(f"Execution mode must be one of: {valid_modes}")
        
        # Validate config if present
        if 'config' in request and request['config']:
            ml_config_valid, ml_config_errors = validate_ml_config(request['config'])
            errors.extend(ml_config_errors)
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"Analysis request validation failed: {str(e)}")
        return False, [f"Request validation error: {str(e)}"]

def is_valid_dataset(df: pd.DataFrame, min_rows: int = 10, min_cols: int = 2) -> bool:
    """
    Quick check if dataset is valid for analysis.
    
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
        logger.error(f"Dataset validity check failed: {str(e)}")
        return False

# Export key functions and classes
__all__ = [
    # Main classes
    'DataValidator', 'SchemaValidator', 'QualityValidator', 'TypeConverter', 'ValidationReporter',
    
    # Configuration and result classes
    'ValidationConfig', 'ValidationResult', 'ValidationIssue', 'ColumnConstraint',
    
    # Enums
    'ValidationLevel', 'DataType', 'ValidationStatus', 'IssueType',
    
    # Main functions
    'validate_dataset', 'validate_file_format', 'validate_column_types', 'validate_data_quality',
    'check_missing_values', 'check_duplicate_records', 'sanitize_input',
    'validate_ml_config', 'validate_analysis_request', 'is_valid_dataset'
]

# Initialize module
logger.info(f"Validation module loaded - Monitoring available: {MONITORING_AVAILABLE}, Pydantic available: {PYDANTIC_AVAILABLE}")
