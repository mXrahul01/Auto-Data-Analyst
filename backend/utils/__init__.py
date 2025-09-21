"""
Utility Functions Package for Auto-Analyst Backend

This package provides essential utility functions for data preprocessing, validation,
and monitoring across the Auto-Analyst backend application. The utilities are designed
to be reusable, efficient, and maintainable across different components of the system.

The utils package contains three main modules:
- preprocessing.py: Data cleaning, transformation, and preparation utilities
- validation.py: Input validation, data quality checks, and error handling utilities  
- monitoring.py: System monitoring, logging, and performance tracking utilities

These utilities support the core backend functionality by providing:
- Standardized data preprocessing pipelines
- Comprehensive input validation and sanitization
- System health monitoring and alerting
- Performance metrics collection and analysis
- Error handling and logging utilities
- Data quality assessment tools

Usage:
    # Import specific utility functions
    from backend.utils import preprocess_data, validate_dataset
    from backend.utils import monitor_performance, log_metrics
    
    # Import entire modules if needed
    from backend.utils import preprocessing, validation, monitoring
    
    # Use preprocessing utilities
    cleaned_data = preprocess_data(raw_data, config)
    
    # Use validation utilities
    is_valid = validate_dataset(dataset, schema)
    
    # Use monitoring utilities  
    monitor_performance('api_endpoint', execution_time)

Design Principles:
- Functions are stateless and side-effect free
- Clear separation of concerns between modules
- Comprehensive error handling and logging
- Type hints and documentation for all public functions
- Performance-optimized implementations
- Consistent API patterns across utilities

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- pydantic: Data validation and serialization
- logging: System logging and monitoring
- typing: Type hints and annotations
"""

import logging
from typing import Any, Dict, List, Optional, Union

# Package metadata
__version__ = "1.0.0"
__author__ = "Auto-Analyst Backend Team"
__description__ = "Utility functions for Auto-Analyst backend"

# Configure package-level logger
logger = logging.getLogger(__name__)

# Import core utility functions with error handling
try:
    # Preprocessing utilities
    from .preprocessing import (
        preprocess_data,
        clean_dataset,
        normalize_features,
        handle_missing_values,
        encode_categorical_features,
        scale_numeric_features,
        extract_features,
        transform_data_types,
        prepare_ml_dataset
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Preprocessing utilities not available: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    # Validation utilities
    from .validation import (
        validate_dataset,
        validate_file_format,
        validate_column_types,
        validate_data_quality,
        check_missing_values,
        check_duplicate_records,
        sanitize_input,
        validate_ml_config,
        validate_analysis_request,
        is_valid_dataset
    )
    VALIDATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Validation utilities not available: {e}")
    VALIDATION_AVAILABLE = False

try:
    # Monitoring utilities
    from .monitoring import (
        monitor_performance,
        log_metrics,
        track_system_health,
        measure_execution_time,
        log_error,
        log_warning,
        log_info,
        create_performance_report,
        get_system_metrics,
        alert_on_threshold
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Monitoring utilities not available: {e}")
    MONITORING_AVAILABLE = False

# Define public API - only include available functions
__all__ = [
    # Package metadata
    "__version__", "__author__", "__description__",
]

# Add preprocessing functions if available
if PREPROCESSING_AVAILABLE:
    __all__.extend([
        "preprocess_data", "clean_dataset", "normalize_features", 
        "handle_missing_values", "encode_categorical_features",
        "scale_numeric_features", "extract_features", 
        "transform_data_types", "prepare_ml_dataset"
    ])

# Add validation functions if available  
if VALIDATION_AVAILABLE:
    __all__.extend([
        "validate_dataset", "validate_file_format", "validate_column_types",
        "validate_data_quality", "check_missing_values", "check_duplicate_records",
        "sanitize_input", "validate_ml_config", "validate_analysis_request",
        "is_valid_dataset"
    ])

# Add monitoring functions if available
if MONITORING_AVAILABLE:
    __all__.extend([
        "monitor_performance", "log_metrics", "track_system_health",
        "measure_execution_time", "log_error", "log_warning", "log_info",
        "create_performance_report", "get_system_metrics", "alert_on_threshold"
    ])

# Utility functions for checking module availability
def get_available_modules() -> Dict[str, bool]:
    """
    Get availability status of utility modules.
    
    Returns:
        Dictionary mapping module names to availability status
    """
    return {
        'preprocessing': PREPROCESSING_AVAILABLE,
        'validation': VALIDATION_AVAILABLE,
        'monitoring': MONITORING_AVAILABLE
    }

def check_module_health() -> Dict[str, Any]:
    """
    Check health status of all utility modules.
    
    Returns:
        Dictionary with health status and module information
    """
    available_modules = get_available_modules()
    total_modules = len(available_modules)
    available_count = sum(available_modules.values())
    
    return {
        'status': 'healthy' if available_count == total_modules else 'degraded',
        'available_modules': available_count,
        'total_modules': total_modules,
        'module_status': available_modules,
        'functions_exported': len(__all__)
    }

# Add utility functions to public API
__all__.extend(['get_available_modules', 'check_module_health'])

# Package initialization logging
logger.info(f"Utils package v{__version__} initialized")
available_modules = get_available_modules()
available_count = sum(available_modules.values())
total_count = len(available_modules)

if available_count == total_count:
    logger.info(f"All utility modules loaded successfully ({available_count}/{total_count})")
else:
    logger.warning(f"Some utility modules unavailable ({available_count}/{total_count})")
    unavailable = [name for name, status in available_modules.items() if not status]
    logger.warning(f"Unavailable modules: {unavailable}")

# No side effects beyond logging - all utility functions are imported but not executed
