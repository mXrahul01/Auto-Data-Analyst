"""
Auto-Analyst Backend Models Package

This package provides database models and validation schemas for the Auto-Analyst platform,
a zero-code AI-powered data analysis web application.

The package contains:
- Database Models: SQLAlchemy ORM models for data persistence
- Pydantic Schemas: Request/response validation models for API endpoints
- Model Registry: Programmatic access to available models and schemas
- Utility Functions: Helper functions for model management and validation

Components:
- database.py: Database ORM models (User, Dataset, Analysis, etc.)
- schemas.py: Pydantic validation schemas for API contracts

Usage:
    # Import database models
    from backend.models import User, Dataset, Analysis
    
    # Import Pydantic schemas  
    from backend.models import UserSchema, DatasetSchema, AnalysisSchema
    
    # Get available models programmatically
    from backend.models import get_database_models, get_schema_models
    
    db_models = get_database_models()
    schemas = get_schema_models()

Features:
- Automatic model discovery and registration
- Safe imports with error handling
- Clean separation between database and validation models
- Registry system for programmatic model access
- Compatible with FastAPI, SQLAlchemy, and Pydantic
- Production-ready with comprehensive error handling
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Type, Union
import importlib
from datetime import datetime

# Package metadata
__version__ = "1.0.0"
__author__ = "Auto-Analyst Backend Team"
__description__ = "Database models and validation schemas for Auto-Analyst platform"

# Configure logging
logger = logging.getLogger(__name__)

# Suppress SQLAlchemy warnings during import
warnings.filterwarnings('ignore', category=UserWarning, module='sqlalchemy')

# Model registries
_DATABASE_MODELS = {}
_SCHEMA_MODELS = {}
_MODEL_REGISTRY = {
    'database': {'available': False, 'module': None, 'models': {}, 'error': None},
    'schemas': {'available': False, 'module': None, 'models': {}, 'error': None}
}

def _safe_import_module(module_name: str, package: str = None) -> Any:
    """
    Safely import a module with comprehensive error handling.
    
    Args:
        module_name: Name of the module to import
        package: Package name for relative imports
        
    Returns:
        Imported module or None if import failed
    """
    try:
        if package:
            module = importlib.import_module(f'.{module_name}', package=package)
        else:
            module = importlib.import_module(module_name)
        
        logger.debug(f"Successfully imported {module_name}")
        return module
        
    except ImportError as e:
        logger.debug(f"Failed to import {module_name}: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error importing {module_name}: {str(e)}")
        return None

def _extract_models_from_module(module: Any, model_type: str) -> Dict[str, Type]:
    """
    Extract model classes from a module based on naming conventions.
    
    Args:
        module: The imported module
        model_type: Type of models to extract ('database' or 'schemas')
        
    Returns:
        Dictionary mapping model names to model classes
    """
    models = {}
    
    try:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Skip private attributes
            if attr_name.startswith('_'):
                continue
            
            # Skip non-classes
            if not isinstance(attr, type):
                continue
            
            # Extract database models (typically SQLAlchemy models)
            if model_type == 'database':
                # Check for SQLAlchemy Base characteristics
                if hasattr(attr, '__tablename__') or hasattr(attr, '__table__'):
                    models[attr_name] = attr
                # Also include common database model patterns
                elif attr_name.endswith(('Model', 'Table')) and not attr_name.startswith('Base'):
                    models[attr_name] = attr
            
            # Extract Pydantic schemas
            elif model_type == 'schemas':
                # Check for Pydantic BaseModel characteristics
                if hasattr(attr, '__fields__') and hasattr(attr, '__config__'):
                    models[attr_name] = attr
                # Also include common schema patterns
                elif attr_name.endswith(('Schema', 'Request', 'Response', 'Create', 'Update')):
                    models[attr_name] = attr
        
        logger.debug(f"Extracted {len(models)} {model_type} models")
        return models
        
    except Exception as e:
        logger.error(f"Error extracting {model_type} models: {str(e)}")
        return {}

def _register_models(module: Any, module_name: str) -> None:
    """Register models from a module in the global registries."""
    try:
        if module_name == 'database':
            models = _extract_models_from_module(module, 'database')
            _DATABASE_MODELS.update(models)
            _MODEL_REGISTRY['database']['models'] = models
            _MODEL_REGISTRY['database']['available'] = True
            _MODEL_REGISTRY['database']['module'] = module
            
        elif module_name == 'schemas':
            models = _extract_models_from_module(module, 'schemas')
            _SCHEMA_MODELS.update(models)
            _MODEL_REGISTRY['schemas']['models'] = models
            _MODEL_REGISTRY['schemas']['available'] = True
            _MODEL_REGISTRY['schemas']['module'] = module
            
    except Exception as e:
        logger.error(f"Error registering {module_name} models: {str(e)}")
        _MODEL_REGISTRY[module_name]['error'] = str(e)

# Import database models
database_module = _safe_import_module('database', __name__)
if database_module:
    _register_models(database_module, 'database')
    
    # Import common database models (these names are typical for such applications)
    try:
        # User and authentication models
        User = getattr(database_module, 'User', None)
        UserSession = getattr(database_module, 'UserSession', None)
        
        # Dataset and file models
        Dataset = getattr(database_module, 'Dataset', None)
        DatasetFile = getattr(database_module, 'DatasetFile', None)
        DatasetMetadata = getattr(database_module, 'DatasetMetadata', None)
        
        # Analysis and ML models
        Analysis = getattr(database_module, 'Analysis', None)
        AnalysisResult = getattr(database_module, 'AnalysisResult', None)
        MLModel = getattr(database_module, 'MLModel', None)
        ModelExperiment = getattr(database_module, 'ModelExperiment', None)
        
        # Pipeline and processing models
        Pipeline = getattr(database_module, 'Pipeline', None)
        PipelineRun = getattr(database_module, 'PipelineRun', None)
        ProcessingJob = getattr(database_module, 'ProcessingJob', None)
        
        # Kaggle integration models
        KaggleToken = getattr(database_module, 'KaggleToken', None)
        RemoteExecution = getattr(database_module, 'RemoteExecution', None)
        
        # Configuration and settings
        UserPreference = getattr(database_module, 'UserPreference', None)
        SystemConfig = getattr(database_module, 'SystemConfig', None)
        
        # Audit and logging models
        AuditLog = getattr(database_module, 'AuditLog', None)
        ErrorLog = getattr(database_module, 'ErrorLog', None)
        
    except AttributeError as e:
        logger.debug(f"Some database models not available: {str(e)}")
else:
    _MODEL_REGISTRY['database']['error'] = "Database module not available"

# Import Pydantic schemas
schemas_module = _safe_import_module('schemas', __name__)
if schemas_module:
    _register_models(schemas_module, 'schemas')
    
    # Import common Pydantic schemas
    try:
        # User schemas
        UserSchema = getattr(schemas_module, 'UserSchema', None)
        UserCreate = getattr(schemas_module, 'UserCreate', None)
        UserUpdate = getattr(schemas_module, 'UserUpdate', None)
        UserResponse = getattr(schemas_module, 'UserResponse', None)
        UserLogin = getattr(schemas_module, 'UserLogin', None)
        
        # Dataset schemas
        DatasetSchema = getattr(schemas_module, 'DatasetSchema', None)
        DatasetCreate = getattr(schemas_module, 'DatasetCreate', None)
        DatasetUpdate = getattr(schemas_module, 'DatasetUpdate', None)
        DatasetResponse = getattr(schemas_module, 'DatasetResponse', None)
        DatasetUpload = getattr(schemas_module, 'DatasetUpload', None)
        
        # Analysis schemas
        AnalysisSchema = getattr(schemas_module, 'AnalysisSchema', None)
        AnalysisRequest = getattr(schemas_module, 'AnalysisRequest', None)
        AnalysisResponse = getattr(schemas_module, 'AnalysisResponse', None)
        AnalysisResult = getattr(schemas_module, 'AnalysisResultSchema', None)
        
        # ML Model schemas
        MLModelSchema = getattr(schemas_module, 'MLModelSchema', None)
        ModelTrainingRequest = getattr(schemas_module, 'ModelTrainingRequest', None)
        ModelPredictionRequest = getattr(schemas_module, 'ModelPredictionRequest', None)
        ModelPredictionResponse = getattr(schemas_module, 'ModelPredictionResponse', None)
        
        # Pipeline schemas
        PipelineSchema = getattr(schemas_module, 'PipelineSchema', None)
        PipelineConfig = getattr(schemas_module, 'PipelineConfigSchema', None)
        PipelineRunRequest = getattr(schemas_module, 'PipelineRunRequest', None)
        PipelineRunResponse = getattr(schemas_module, 'PipelineRunResponse', None)
        
        # Kaggle integration schemas
        KaggleTokenSchema = getattr(schemas_module, 'KaggleTokenSchema', None)
        KaggleConnectionRequest = getattr(schemas_module, 'KaggleConnectionRequest', None)
        RemoteExecutionRequest = getattr(schemas_module, 'RemoteExecutionRequest', None)
        RemoteExecutionResponse = getattr(schemas_module, 'RemoteExecutionResponse', None)
        
        # Common response schemas
        StatusResponse = getattr(schemas_module, 'StatusResponse', None)
        ErrorResponse = getattr(schemas_module, 'ErrorResponse', None)
        MessageResponse = getattr(schemas_module, 'MessageResponse', None)
        
        # Dashboard and visualization schemas
        DashboardData = getattr(schemas_module, 'DashboardDataSchema', None)
        ChartData = getattr(schemas_module, 'ChartDataSchema', None)
        MetricsResponse = getattr(schemas_module, 'MetricsResponseSchema', None)
        
    except AttributeError as e:
        logger.debug(f"Some schema models not available: {str(e)}")
else:
    _MODEL_REGISTRY['schemas']['error'] = "Schemas module not available"

# Public API functions

def get_database_models() -> Dict[str, Type]:
    """
    Get all available database models.
    
    Returns:
        Dictionary mapping model names to database model classes
    """
    return _DATABASE_MODELS.copy()

def get_schema_models() -> Dict[str, Type]:
    """
    Get all available Pydantic schema models.
    
    Returns:
        Dictionary mapping schema names to Pydantic model classes
    """
    return _SCHEMA_MODELS.copy()

def get_all_models() -> Dict[str, Dict[str, Type]]:
    """
    Get all available models organized by type.
    
    Returns:
        Dictionary with 'database' and 'schemas' keys containing respective models
    """
    return {
        'database': get_database_models(),
        'schemas': get_schema_models()
    }

def get_model_registry() -> Dict[str, Any]:
    """
    Get the complete model registry with availability and error information.
    
    Returns:
        Dictionary with detailed model registry information
    """
    return _MODEL_REGISTRY.copy()

def find_model(model_name: str, model_type: Optional[str] = None) -> Optional[Type]:
    """
    Find a model by name, optionally filtered by type.
    
    Args:
        model_name: Name of the model to find
        model_type: Optional type filter ('database' or 'schemas')
        
    Returns:
        Model class if found, None otherwise
    """
    if model_type == 'database':
        return _DATABASE_MODELS.get(model_name)
    elif model_type == 'schemas':
        return _SCHEMA_MODELS.get(model_name)
    else:
        # Search in both registries
        return _DATABASE_MODELS.get(model_name) or _SCHEMA_MODELS.get(model_name)

def list_available_models(model_type: Optional[str] = None) -> List[str]:
    """
    List names of available models, optionally filtered by type.
    
    Args:
        model_type: Optional type filter ('database' or 'schemas')
        
    Returns:
        List of available model names
    """
    if model_type == 'database':
        return list(_DATABASE_MODELS.keys())
    elif model_type == 'schemas':
        return list(_SCHEMA_MODELS.keys())
    else:
        return list(_DATABASE_MODELS.keys()) + list(_SCHEMA_MODELS.keys())

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    model = find_model(model_name)
    
    if not model:
        return {'found': False, 'name': model_name}
    
    info = {
        'found': True,
        'name': model_name,
        'class': model.__name__,
        'module': model.__module__,
        'doc': model.__doc__
    }
    
    # Add database-specific info
    if hasattr(model, '__tablename__'):
        info.update({
            'type': 'database',
            'tablename': model.__tablename__,
            'columns': getattr(model, '__table__', {}).columns.keys() if hasattr(model, '__table__') else []
        })
    
    # Add Pydantic-specific info  
    if hasattr(model, '__fields__'):
        info.update({
            'type': 'schema',
            'fields': list(model.__fields__.keys()),
            'validators': hasattr(model, '__validators__')
        })
    
    return info

def validate_models() -> Dict[str, Any]:
    """
    Validate the models package and check for issues.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'status': 'healthy',
        'issues': [],
        'warnings': [],
        'info': {
            'database_models_count': len(_DATABASE_MODELS),
            'schema_models_count': len(_SCHEMA_MODELS),
            'database_available': _MODEL_REGISTRY['database']['available'],
            'schemas_available': _MODEL_REGISTRY['schemas']['available']
        }
    }
    
    # Check for critical issues
    if not _MODEL_REGISTRY['database']['available']:
        validation['status'] = 'degraded'
        validation['issues'].append('Database models not available')
        if _MODEL_REGISTRY['database']['error']:
            validation['issues'].append(f"Database error: {_MODEL_REGISTRY['database']['error']}")
    
    if not _MODEL_REGISTRY['schemas']['available']:
        validation['status'] = 'degraded'  
        validation['issues'].append('Schema models not available')
        if _MODEL_REGISTRY['schemas']['error']:
            validation['issues'].append(f"Schemas error: {_MODEL_REGISTRY['schemas']['error']}")
    
    # Check for warnings
    if len(_DATABASE_MODELS) == 0 and _MODEL_REGISTRY['database']['available']:
        validation['warnings'].append('No database models found in database module')
        
    if len(_SCHEMA_MODELS) == 0 and _MODEL_REGISTRY['schemas']['available']:
        validation['warnings'].append('No schema models found in schemas module')
    
    # Both modules missing is critical
    if not _MODEL_REGISTRY['database']['available'] and not _MODEL_REGISTRY['schemas']['available']:
        validation['status'] = 'critical'
        validation['issues'].append('No model modules available - check package installation')
    
    return validation

def get_package_status() -> Dict[str, Any]:
    """
    Get comprehensive package status information.
    
    Returns:
        Dictionary with complete package status
    """
    return {
        'package': 'backend.models',
        'version': __version__,
        'description': __description__,
        'timestamp': datetime.now().isoformat(),
        'modules': {
            'database': {
                'available': _MODEL_REGISTRY['database']['available'],
                'model_count': len(_DATABASE_MODELS),
                'models': list(_DATABASE_MODELS.keys()),
                'error': _MODEL_REGISTRY['database']['error']
            },
            'schemas': {
                'available': _MODEL_REGISTRY['schemas']['available'],
                'model_count': len(_SCHEMA_MODELS), 
                'models': list(_SCHEMA_MODELS.keys()),
                'error': _MODEL_REGISTRY['schemas']['error']
            }
        },
        'validation': validate_models()
    }

# Utility functions for common operations

def create_model_mapping(prefix: str = '', suffix: str = '') -> Dict[str, Type]:
    """
    Create a filtered mapping of models based on naming patterns.
    
    Args:
        prefix: Filter models starting with this prefix
        suffix: Filter models ending with this suffix
        
    Returns:
        Dictionary of filtered models
    """
    all_models = {**_DATABASE_MODELS, **_SCHEMA_MODELS}
    filtered = {}
    
    for name, model in all_models.items():
        if (not prefix or name.startswith(prefix)) and (not suffix or name.endswith(suffix)):
            filtered[name] = model
    
    return filtered

def get_user_related_models() -> Dict[str, Type]:
    """Get all user-related models and schemas."""
    return create_model_mapping(prefix='User')

def get_dataset_related_models() -> Dict[str, Type]:
    """Get all dataset-related models and schemas.""" 
    return create_model_mapping(prefix='Dataset')

def get_analysis_related_models() -> Dict[str, Type]:
    """Get all analysis-related models and schemas."""
    return create_model_mapping(prefix='Analysis')

# Define the public API - these are the main exports
__all__ = [
    # Package metadata
    '__version__', '__author__', '__description__',
    
    # Database models (if available)
    'User', 'UserSession', 'Dataset', 'DatasetFile', 'DatasetMetadata',
    'Analysis', 'AnalysisResult', 'MLModel', 'ModelExperiment',
    'Pipeline', 'PipelineRun', 'ProcessingJob',
    'KaggleToken', 'RemoteExecution', 'UserPreference', 'SystemConfig',
    'AuditLog', 'ErrorLog',
    
    # Pydantic schemas (if available)
    'UserSchema', 'UserCreate', 'UserUpdate', 'UserResponse', 'UserLogin',
    'DatasetSchema', 'DatasetCreate', 'DatasetUpdate', 'DatasetResponse', 'DatasetUpload',
    'AnalysisSchema', 'AnalysisRequest', 'AnalysisResponse', 'AnalysisResult',
    'MLModelSchema', 'ModelTrainingRequest', 'ModelPredictionRequest', 'ModelPredictionResponse',
    'PipelineSchema', 'PipelineConfig', 'PipelineRunRequest', 'PipelineRunResponse',
    'KaggleTokenSchema', 'KaggleConnectionRequest', 'RemoteExecutionRequest', 'RemoteExecutionResponse',
    'StatusResponse', 'ErrorResponse', 'MessageResponse',
    'DashboardData', 'ChartData', 'MetricsResponse',
    
    # Registry and utility functions
    'get_database_models', 'get_schema_models', 'get_all_models', 'get_model_registry',
    'find_model', 'list_available_models', 'get_model_info', 'validate_models', 'get_package_status',
    'create_model_mapping', 'get_user_related_models', 'get_dataset_related_models', 'get_analysis_related_models'
]

# Remove None values from __all__ (for missing imports)
__all__ = [item for item in __all__ if globals().get(item) is not None]

# Package initialization logging
logger.info(f"Auto-Analyst models package v{__version__} initialized")
logger.info(f"Database models: {len(_DATABASE_MODELS)} available")
logger.info(f"Schema models: {len(_SCHEMA_MODELS)} available")

# Log any import issues
if _MODEL_REGISTRY['database']['error']:
    logger.warning(f"Database models import issue: {_MODEL_REGISTRY['database']['error']}")
    
if _MODEL_REGISTRY['schemas']['error']:
    logger.warning(f"Schema models import issue: {_MODEL_REGISTRY['schemas']['error']}")

# Final validation check
validation_result = validate_models()
if validation_result['status'] == 'critical':
    logger.error("Critical issues with models package - some functionality may not work")
elif validation_result['status'] == 'degraded':
    logger.warning("Models package has some issues - check validation results")
else:
    logger.info("Models package loaded successfully")
