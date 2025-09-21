"""
Backend Services Package for Auto-Analyst Platform

This package provides comprehensive business logic services for the Auto-Analyst
zero-code AI-powered data analysis web application. The services layer acts as
the orchestration layer between API endpoints, database models, and ML components.

The services package contains:
- DataService: Dataset management, loading, preprocessing, and validation
- MLService: Machine learning pipeline orchestration and model management
- InsightsService: Automated insight generation and business intelligence
- MLOpsService: Experiment tracking, model monitoring, and feature store management

Features:
- Production-ready business logic services
- Clean separation of concerns
- Dependency injection compatible with FastAPI
- Comprehensive error handling and logging
- Async/await support for non-blocking operations
- Scalable architecture for concurrent operations
- Integration with database models and ML pipelines

Service Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DataService   │    │   MLService     │    │ InsightsService │
│                 │    │                 │    │                 │
│ • Data loading  │    │ • Pipeline mgmt │    │ • Insight gen.  │
│ • Preprocessing │────│ • Model training│────│ • Business intel│
│ • Validation    │    │ • Analysis runs │    │ • Recommendations│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  MLOpsService   │
                    │                 │
                    │ • Experiment    │
                    │   tracking      │
                    │ • Model monitor │
                    │ • Feature store │
                    └─────────────────┘

Usage:
    # Import specific services
    from backend.services import DataService, MLService, InsightsService
    
    # Import factory functions
    from backend.services import create_data_service, create_ml_service
    
    # Import for FastAPI dependency injection
    from backend.services import get_data_service, get_ml_service
    
    # Import configuration classes
    from backend.services import DataLoadingConfig, MLConfig, ExperimentConfig
    
    # Use in FastAPI endpoints
    @app.post("/analyses/")
    def create_analysis(
        data_service: DataService = Depends(get_data_service),
        ml_service: MLService = Depends(get_ml_service)
    ):
        # Service logic here
        pass

Module Dependencies:
- data_service.py: pandas, numpy, scikit-learn, optional (openpyxl, pyarrow)
- ml_service.py: ML pipeline integration, database models
- insights_service.py: Natural language processing, explanation services
- mlops_service.py: MLflow, Feast, Evidently, Prometheus (all optional)

Error Handling:
All services include comprehensive error handling with graceful degradation
when optional dependencies are not available. Import errors are logged
but do not prevent package initialization.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

# Package metadata
__version__ = "1.0.0"
__author__ = "Auto-Analyst Backend Team"
__description__ = "Business logic services for Auto-Analyst platform"
__all__ = [
    # Version and metadata
    '__version__', '__author__', '__description__',
    
    # Service classes
    'DataService', 'MLService', 'InsightsService', 'MLOpsService',
    
    # Factory functions
    'create_data_service', 'create_ml_service', 'create_insights_service', 'create_mlops_service',
    
    # Dependency injection functions
    'get_data_service', 'get_ml_service', 'get_insights_service', 'get_mlops_service',
    
    # Configuration classes
    'DataLoadingConfig', 'PreprocessingConfig', 'MLConfig', 'ExperimentConfig',
    'InsightMetadata', 'AnalysisContext', 'DriftReport',
    
    # Enums and constants
    'FileFormat', 'DataType', 'ProcessingStrategy', 'ModelType', 'InsightType',
    'AnalysisStatus', 'ExecutionMode', 'ExperimentStatus', 'MonitoringStatus',
    
    # Result classes
    'DataProcessingResult', 'AnalysisResult', 'InsightsResult', 'ModelMonitoringResult',
    
    # Utility functions
    'get_service_registry', 'get_available_services', 'validate_services',
    'get_service_health', 'create_service_registry'
]

# Configure logging for the services package
logger = logging.getLogger(__name__)

# Suppress common warnings that might occur during service initialization
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')

# Service availability tracking
_SERVICE_AVAILABILITY = {
    'data_service': False,
    'ml_service': False,
    'insights_service': False,
    'mlops_service': False
}

_IMPORT_ERRORS = {}

def _safe_import(module_name: str, items: List[str]) -> Dict[str, Any]:
    """
    Safely import items from a module with error handling.
    
    Args:
        module_name: Name of the module to import from
        items: List of items to import
        
    Returns:
        Dictionary of successfully imported items
    """
    imported_items = {}
    
    try:
        module = __import__(f'backend.services.{module_name}', fromlist=items)
        
        for item in items:
            if hasattr(module, item):
                imported_items[item] = getattr(module, item)
            else:
                logger.warning(f"Item '{item}' not found in module '{module_name}'")
        
        _SERVICE_AVAILABILITY[module_name] = True
        logger.debug(f"Successfully imported from {module_name}: {list(imported_items.keys())}")
        
    except ImportError as e:
        _SERVICE_AVAILABILITY[module_name] = False
        _IMPORT_ERRORS[module_name] = str(e)
        logger.warning(f"Failed to import from {module_name}: {str(e)}")
        
        # Create mock objects for missing services to prevent AttributeError
        for item in items:
            if item.endswith('Service'):
                imported_items[item] = _create_mock_service(item)
            elif item.startswith('create_') or item.startswith('get_'):
                imported_items[item] = _create_mock_factory(item)
            else:
                imported_items[item] = None
    
    except Exception as e:
        _SERVICE_AVAILABILITY[module_name] = False
        _IMPORT_ERRORS[module_name] = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error importing from {module_name}: {str(e)}")
        
        # Create mock objects for failed imports
        for item in items:
            imported_items[item] = None
    
    return imported_items

def _create_mock_service(service_name: str) -> type:
    """Create a mock service class for unavailable services."""
    
    class MockService:
        def __init__(self, *args, **kwargs):
            self.service_name = service_name
            logger.warning(f"{service_name} is not available - using mock implementation")
        
        def __getattr__(self, name):
            def mock_method(*args, **kwargs):
                raise RuntimeError(f"{self.service_name} is not available. Please check dependencies.")
            return mock_method
        
        def get_service_health(self):
            return {
                'status': 'unavailable',
                'service': self.service_name,
                'error': _IMPORT_ERRORS.get(service_name.lower().replace('service', '_service'), 'Service not available')
            }
    
    MockService.__name__ = service_name
    MockService.__qualname__ = service_name
    return MockService

def _create_mock_factory(function_name: str) -> callable:
    """Create a mock factory function for unavailable services."""
    
    def mock_factory(*args, **kwargs):
        service_name = function_name.replace('create_', '').replace('get_', '') + '_service'
        raise RuntimeError(f"Cannot create service - {service_name} is not available")
    
    mock_factory.__name__ = function_name
    return mock_factory

# Import DataService and related components
data_service_imports = _safe_import('data_service', [
    'DataService', 'create_data_service', 'get_data_service',
    'DataLoadingConfig', 'PreprocessingConfig', 'DatasetInfo', 'DataProcessingResult',
    'FileFormat', 'DataType', 'ProcessingStrategy'
])

# Import MLService and related components
ml_service_imports = _safe_import('ml_service', [
    'MLService', 'create_ml_service', 'get_ml_service',
    'MLConfig', 'AnalysisContext', 'AnalysisResult',
    'AnalysisStatus', 'ExecutionMode'
])

# Import InsightsService and related components
insights_service_imports = _safe_import('insights_service', [
    'InsightsService', 'create_insights_service', 'get_insights_service',
    'InsightMetadata', 'Insight', 'InsightsResult',
    'ModelType', 'InsightType', 'ConfidenceLevel'
])

# Import MLOpsService and related components
mlops_service_imports = _safe_import('mlops_service', [
    'MLOpsService', 'create_mlops_service', 'get_mlops_service',
    'ExperimentConfig', 'ExperimentMetadata', 'DriftReport', 'ModelMonitoringResult',
    'ExperimentStatus', 'ModelStage', 'MonitoringStatus', 'DriftType'
])

# Combine all imports
all_imports = {
    **data_service_imports,
    **ml_service_imports,
    **insights_service_imports,
    **mlops_service_imports
}

# Add imports to module namespace
globals().update(all_imports)

# Service registry and management functions

def get_service_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive service registry information.
    
    Returns:
        Dictionary with service availability and status information
    """
    registry = {}
    
    for service_name, available in _SERVICE_AVAILABILITY.items():
        registry[service_name] = {
            'available': available,
            'error': _IMPORT_ERRORS.get(service_name) if not available else None,
            'module': f'backend.services.{service_name}'
        }
        
        # Add service class information if available
        service_class_name = ''.join(word.capitalize() for word in service_name.split('_'))
        if service_class_name in all_imports:
            registry[service_name]['service_class'] = service_class_name
            registry[service_name]['factory_function'] = f'create_{service_name}'
            registry[service_name]['dependency_function'] = f'get_{service_name}'
    
    return registry

def get_available_services() -> List[str]:
    """
    Get list of available services.
    
    Returns:
        List of service names that are available
    """
    return [
        service_name for service_name, available in _SERVICE_AVAILABILITY.items()
        if available
    ]

def get_unavailable_services() -> Dict[str, str]:
    """
    Get list of unavailable services with error messages.
    
    Returns:
        Dictionary mapping service names to error messages
    """
    return {
        service_name: _IMPORT_ERRORS.get(service_name, 'Unknown error')
        for service_name, available in _SERVICE_AVAILABILITY.items()
        if not available
    }

def validate_services() -> Dict[str, Any]:
    """
    Validate service availability and provide recommendations.
    
    Returns:
        Validation results with status and recommendations
    """
    available_services = get_available_services()
    unavailable_services = get_unavailable_services()
    
    # Determine overall status
    if len(available_services) == len(_SERVICE_AVAILABILITY):
        status = 'healthy'
    elif len(available_services) >= len(_SERVICE_AVAILABILITY) * 0.75:
        status = 'degraded'
    else:
        status = 'critical'
    
    # Generate recommendations
    recommendations = []
    if 'data_service' not in available_services:
        recommendations.append("Install data processing dependencies: pip install pandas numpy scikit-learn")
    
    if 'mlops_service' not in available_services:
        recommendations.append("Install MLOps dependencies: pip install mlflow feast evidently prometheus-client")
    
    if 'insights_service' not in available_services:
        recommendations.append("Install insights dependencies: pip install nltk scipy")
    
    return {
        'status': status,
        'available_services': available_services,
        'unavailable_services': unavailable_services,
        'recommendations': recommendations,
        'service_count': {
            'total': len(_SERVICE_AVAILABILITY),
            'available': len(available_services),
            'unavailable': len(unavailable_services)
        }
    }

def get_service_health() -> Dict[str, Any]:
    """
    Get comprehensive health status of all services.
    
    Returns:
        Health status information for monitoring
    """
    health_status = {
        'overall_status': 'healthy',
        'timestamp': None,
        'services': {},
        'summary': {
            'total_services': len(_SERVICE_AVAILABILITY),
            'available_services': len(get_available_services()),
            'critical_issues': 0,
            'warnings': 0
        }
    }
    
    try:
        from datetime import datetime
        health_status['timestamp'] = datetime.now().isoformat()
        
        # Check individual services
        critical_services = ['data_service', 'ml_service']
        
        for service_name, available in _SERVICE_AVAILABILITY.items():
            service_status = {
                'available': available,
                'critical': service_name in critical_services,
                'status': 'healthy' if available else 'unavailable',
                'error': _IMPORT_ERRORS.get(service_name) if not available else None
            }
            
            # Try to get service-specific health if available
            try:
                if available:
                    service_class_name = ''.join(word.capitalize() for word in service_name.split('_'))
                    if service_class_name in globals():
                        service_class = globals()[service_class_name]
                        # This would call get_service_health() if the service supports it
                        # For now, just mark as healthy if importable
                        service_status['details'] = 'Service class loaded successfully'
                
            except Exception as e:
                service_status['status'] = 'degraded'
                service_status['error'] = str(e)
                health_status['summary']['warnings'] += 1
            
            health_status['services'][service_name] = service_status
            
            # Update summary
            if not available and service_name in critical_services:
                health_status['summary']['critical_issues'] += 1
                health_status['overall_status'] = 'critical'
            elif not available:
                health_status['summary']['warnings'] += 1
                if health_status['overall_status'] == 'healthy':
                    health_status['overall_status'] = 'degraded'
        
        # Add package information
        health_status['package_info'] = {
            'version': __version__,
            'description': __description__,
            'total_imports': len(all_imports),
            'successful_imports': len([k for k, v in all_imports.items() if v is not None])
        }
        
    except Exception as e:
        health_status['overall_status'] = 'error'
        health_status['error'] = str(e)
        logger.error(f"Health check failed: {str(e)}")
    
    return health_status

def create_service_registry() -> Dict[str, Any]:
    """
    Create a comprehensive service registry for the application.
    
    Returns:
        Service registry with factory functions and metadata
    """
    registry = {
        'services': {},
        'factories': {},
        'dependency_injection': {},
        'configurations': {},
        'metadata': {
            'version': __version__,
            'total_services': len(_SERVICE_AVAILABILITY),
            'available_services': len(get_available_services())
        }
    }
    
    # Register available services
    for service_name, available in _SERVICE_AVAILABILITY.items():
        if available:
            service_class_name = ''.join(word.capitalize() for word in service_name.split('_'))
            factory_name = f'create_{service_name}'
            di_name = f'get_{service_name}'
            config_name = f'{service_class_name.replace("Service", "")}Config'
            
            if service_class_name in all_imports:
                registry['services'][service_name] = all_imports[service_class_name]
            
            if factory_name in all_imports:
                registry['factories'][service_name] = all_imports[factory_name]
            
            if di_name in all_imports:
                registry['dependency_injection'][service_name] = all_imports[di_name]
            
            if config_name in all_imports:
                registry['configurations'][service_name] = all_imports[config_name]
    
    return registry

# Utility functions for FastAPI integration

def create_service_dependencies() -> Dict[str, callable]:
    """
    Create dependency injection functions for FastAPI.
    
    Returns:
        Dictionary of dependency functions for FastAPI Depends()
    """
    dependencies = {}
    
    if 'get_data_service' in all_imports:
        dependencies['data_service'] = all_imports['get_data_service']
    
    if 'get_ml_service' in all_imports:
        dependencies['ml_service'] = all_imports['get_ml_service']
    
    if 'get_insights_service' in all_imports:
        dependencies['insights_service'] = all_imports['get_insights_service']
    
    if 'get_mlops_service' in all_imports:
        dependencies['mlops_service'] = all_imports['get_mlops_service']
    
    return dependencies

# Initialize package and log status
try:
    # Log package initialization
    available_count = len(get_available_services())
    total_count = len(_SERVICE_AVAILABILITY)
    
    logger.info(f"Backend Services package v{__version__} initialized")
    logger.info(f"Service availability: {available_count}/{total_count} services loaded")
    
    # Log any import issues
    unavailable = get_unavailable_services()
    if unavailable:
        logger.warning(f"Some services unavailable: {list(unavailable.keys())}")
        for service, error in unavailable.items():
            logger.debug(f"  {service}: {error}")
    else:
        logger.info("All services loaded successfully")
    
    # Validate critical services
    validation = validate_services()
    if validation['status'] == 'critical':
        logger.error(f"Critical service issues detected. Recommendations: {validation['recommendations']}")
    elif validation['status'] == 'degraded':
        logger.warning(f"Some services degraded. Recommendations: {validation['recommendations']}")
    
except Exception as e:
    logger.error(f"Package initialization error: {str(e)}")

# Export convenience function for checking service status
def check_service_status() -> None:
    """Print service status to console for debugging."""
    print(f"\n🔧 Auto-Analyst Services Status")
    print("=" * 50)
    
    for service_name, available in _SERVICE_AVAILABILITY.items():
        status_icon = "✅" if available else "❌"
        print(f"{status_icon} {service_name.replace('_', ' ').title()}: {'Available' if available else 'Unavailable'}")
        
        if not available and service_name in _IMPORT_ERRORS:
            print(f"    Error: {_IMPORT_ERRORS[service_name]}")
    
    print(f"\n📊 Summary: {len(get_available_services())}/{len(_SERVICE_AVAILABILITY)} services available")
    
    unavailable = get_unavailable_services()
    if unavailable:
        print(f"\n💡 Recommendations:")
        validation = validate_services()
        for rec in validation.get('recommendations', []):
            print(f"   • {rec}")

# Clean up temporary variables to avoid polluting the namespace
del logging, warnings, logger
