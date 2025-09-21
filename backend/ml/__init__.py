"""
Auto-Analyst ML Package

This package provides a comprehensive suite of machine learning models and utilities
for automated data analysis, supporting tabular, time series, text, anomaly detection,
clustering, and deep learning tasks.

The package is designed for zero-code AI-powered data analysis with automatic model
selection, hyperparameter optimization, and deployment-ready pipelines.

Features:
- Automatic dataset type detection and model selection
- Comprehensive preprocessing and feature engineering
- Multiple model categories (tabular, timeseries, text, anomaly, clustering, deep learning)
- Ensemble methods and advanced optimization
- Model explanation and interpretability
- Production-ready deployment pipelines
- MLOps integration with experiment tracking
- Remote execution support (Kaggle/Colab)

Usage:
    from ml import AutoPipeline, get_available_models, create_analyzer
    
    # Quick analysis
    pipeline = AutoPipeline()
    result = await pipeline.analyze_dataset(df, target_column='target')
    
    # Get available models
    models = get_available_models()
    
    # Create specific analyzer
    analyzer = create_analyzer('tabular', task_type='classification')
"""

import sys
import warnings
import logging
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
import importlib
from datetime import datetime

# Package metadata
__version__ = "1.0.0"
__author__ = "Auto-Analyst AI Lab"
__email__ = "ai-lab@auto-analyst.com"
__description__ = "Comprehensive AutoML package for zero-code data analysis"
__license__ = "MIT"
__url__ = "https://github.com/auto-analyst/ml"

# Package information
PACKAGE_INFO = {
    'name': 'auto-analyst-ml',
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'license': __license__,
    'url': __url__,
    'python_requires': '>=3.8',
    'created': '2025-09-21',
    'last_updated': datetime.now().isoformat()
}

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# Suppress common ML library warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Module availability registry
_MODULE_REGISTRY = {
    'auto_pipeline': {'available': False, 'module': None, 'error': None},
    'tabular_models': {'available': False, 'module': None, 'error': None},
    'timeseries_models': {'available': False, 'module': None, 'error': None},
    'text_models': {'available': False, 'module': None, 'error': None},
    'anomaly_models': {'available': False, 'module': None, 'error': None},
    'clustering_models': {'available': False, 'module': None, 'error': None},
    'deep_models': {'available': False, 'module': None, 'error': None},
    'ensemble_models': {'available': False, 'module': None, 'error': None},
    'feature_engineering': {'available': False, 'module': None, 'error': None},
    'preprocessing': {'available': False, 'module': None, 'error': None},
    'evaluation': {'available': False, 'module': None, 'error': None},
    'explainer': {'available': False, 'module': None, 'error': None},
    'model_selection': {'available': False, 'module': None, 'error': None},
    'validation': {'available': False, 'module': None, 'error': None}
}

# Analyzer registry for different data types
_ANALYZER_REGISTRY = {}

# Model factory registry
_MODEL_FACTORIES = {}

def _safe_import(module_name: str, package: str = None) -> Any:
    """
    Safely import a module with error handling.
    
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
        
        _MODULE_REGISTRY[module_name]['available'] = True
        _MODULE_REGISTRY[module_name]['module'] = module
        _MODULE_REGISTRY[module_name]['error'] = None
        
        logger.debug(f"Successfully imported {module_name}")
        return module
        
    except ImportError as e:
        _MODULE_REGISTRY[module_name]['available'] = False
        _MODULE_REGISTRY[module_name]['module'] = None
        _MODULE_REGISTRY[module_name]['error'] = str(e)
        
        logger.debug(f"Failed to import {module_name}: {str(e)}")
        return None
    
    except Exception as e:
        _MODULE_REGISTRY[module_name]['available'] = False
        _MODULE_REGISTRY[module_name]['module'] = None
        _MODULE_REGISTRY[module_name]['error'] = f"Unexpected error: {str(e)}"
        
        logger.warning(f"Unexpected error importing {module_name}: {str(e)}")
        return None

def _register_analyzer(data_type: str, analyzer_class: Type) -> None:
    """Register an analyzer for a specific data type."""
    _ANALYZER_REGISTRY[data_type] = analyzer_class

def _register_model_factory(model_type: str, factory_function: Any) -> None:
    """Register a model factory function."""
    _MODEL_FACTORIES[model_type] = factory_function

# Core imports - Auto Pipeline (main orchestrator)
auto_pipeline = _safe_import('auto_pipeline', __name__)
if auto_pipeline:
    try:
        # Import main classes
        AutoPipeline = getattr(auto_pipeline, 'PipelineOrchestrator', None)
        PipelineConfig = getattr(auto_pipeline, 'PipelineConfig', None)
        DatasetDetector = getattr(auto_pipeline, 'DatasetDetector', None)
        
        # Import utility functions
        create_auto_pipeline = getattr(auto_pipeline, 'create_auto_pipeline', None)
        quick_auto_analysis = getattr(auto_pipeline, 'quick_auto_analysis', None)
        auto_analyze_dataset = getattr(auto_pipeline, 'auto_analyze_dataset', None)
        
        # Import enums
        DatasetType = getattr(auto_pipeline, 'DatasetType', None)
        TaskType = getattr(auto_pipeline, 'TaskType', None)
        ExecutionMode = getattr(auto_pipeline, 'ExecutionMode', None)
        
    except AttributeError as e:
        logger.warning(f"Some auto_pipeline components not available: {str(e)}")

# Model Selection
model_selection = _safe_import('model_selection', __name__)
if model_selection:
    try:
        ModelSelector = getattr(model_selection, 'ModelSelector', None)
        create_model_selector = getattr(model_selection, 'create_model_selector', None)
        auto_select_and_train_models = getattr(model_selection, 'auto_select_and_train_models', None)
    except AttributeError as e:
        logger.warning(f"Some model_selection components not available: {str(e)}")

# Tabular Models
tabular_models = _safe_import('tabular_models', __name__)
if tabular_models:
    try:
        create_tabular_analyzer = getattr(tabular_models, 'create_tabular_analyzer', None)
        TabularAnalyzer = getattr(tabular_models, 'TabularAnalyzer', None)
        if create_tabular_analyzer:
            _register_model_factory('tabular', create_tabular_analyzer)
        if TabularAnalyzer:
            _register_analyzer('tabular', TabularAnalyzer)
    except AttributeError as e:
        logger.warning(f"Some tabular_models components not available: {str(e)}")

# Time Series Models
timeseries_models = _safe_import('timeseries_models', __name__)
if timeseries_models:
    try:
        create_timeseries_analyzer = getattr(timeseries_models, 'create_timeseries_analyzer', None)
        TimeSeriesAnalyzer = getattr(timeseries_models, 'TimeSeriesAnalyzer', None)
        if create_timeseries_analyzer:
            _register_model_factory('timeseries', create_timeseries_analyzer)
        if TimeSeriesAnalyzer:
            _register_analyzer('timeseries', TimeSeriesAnalyzer)
    except AttributeError as e:
        logger.warning(f"Some timeseries_models components not available: {str(e)}")

# Text Models
text_models = _safe_import('text_models', __name__)
if text_models:
    try:
        create_text_analyzer = getattr(text_models, 'create_text_analyzer', None)
        TextAnalyzer = getattr(text_models, 'TextAnalyzer', None)
        if create_text_analyzer:
            _register_model_factory('text', create_text_analyzer)
        if TextAnalyzer:
            _register_analyzer('text', TextAnalyzer)
    except AttributeError as e:
        logger.warning(f"Some text_models components not available: {str(e)}")

# Anomaly Detection Models
anomaly_models = _safe_import('anomaly_models', __name__)
if anomaly_models:
    try:
        create_anomaly_detector = getattr(anomaly_models, 'create_anomaly_detector', None)
        AnomalyDetector = getattr(anomaly_models, 'AnomalyDetector', None)
        if create_anomaly_detector:
            _register_model_factory('anomaly', create_anomaly_detector)
        if AnomalyDetector:
            _register_analyzer('anomaly', AnomalyDetector)
    except AttributeError as e:
        logger.warning(f"Some anomaly_models components not available: {str(e)}")

# Clustering Models
clustering_models = _safe_import('clustering_models', __name__)
if clustering_models:
    try:
        create_clustering_analyzer = getattr(clustering_models, 'create_clustering_analyzer', None)
        ClusteringAnalyzer = getattr(clustering_models, 'ClusteringAnalyzer', None)
        if create_clustering_analyzer:
            _register_model_factory('clustering', create_clustering_analyzer)
        if ClusteringAnalyzer:
            _register_analyzer('clustering', ClusteringAnalyzer)
    except AttributeError as e:
        logger.warning(f"Some clustering_models components not available: {str(e)}")

# Deep Learning Models
deep_models = _safe_import('deep_models', __name__)
if deep_models:
    try:
        create_deep_learning_analyzer = getattr(deep_models, 'create_deep_learning_analyzer', None)
        DeepLearningAnalyzer = getattr(deep_models, 'DeepLearningAnalyzer', None)
        if create_deep_learning_analyzer:
            _register_model_factory('deep', create_deep_learning_analyzer)
        if DeepLearningAnalyzer:
            _register_analyzer('deep', DeepLearningAnalyzer)
    except AttributeError as e:
        logger.warning(f"Some deep_models components not available: {str(e)}")

# Ensemble Models
ensemble_models = _safe_import('ensemble_models', __name__)
if ensemble_models:
    try:
        create_ensemble_analyzer = getattr(ensemble_models, 'create_ensemble_analyzer', None)
        EnsembleAnalyzer = getattr(ensemble_models, 'EnsembleAnalyzer', None)
        if create_ensemble_analyzer:
            _register_model_factory('ensemble', create_ensemble_analyzer)
        if EnsembleAnalyzer:
            _register_analyzer('ensemble', EnsembleAnalyzer)
    except AttributeError as e:
        logger.warning(f"Some ensemble_models components not available: {str(e)}")

# Feature Engineering
feature_engineering = _safe_import('feature_engineering', __name__)
if feature_engineering:
    try:
        create_feature_engineer = getattr(feature_engineering, 'create_feature_engineer', None)
        FeatureEngineer = getattr(feature_engineering, 'FeatureEngineer', None)
    except AttributeError as e:
        logger.warning(f"Some feature_engineering components not available: {str(e)}")

# Preprocessing
preprocessing = _safe_import('preprocessing', __name__)
if preprocessing:
    try:
        create_preprocessor = getattr(preprocessing, 'create_preprocessor', None)
        DataPreprocessor = getattr(preprocessing, 'DataPreprocessor', None)
    except AttributeError as e:
        logger.warning(f"Some preprocessing components not available: {str(e)}")

# Evaluation
evaluation = _safe_import('evaluation', __name__)
if evaluation:
    try:
        create_evaluator = getattr(evaluation, 'create_evaluator', None)
        ModelEvaluator = getattr(evaluation, 'ModelEvaluator', None)
    except AttributeError as e:
        logger.warning(f"Some evaluation components not available: {str(e)}")

# Explainer
explainer = _safe_import('explainer', __name__)
if explainer:
    try:
        create_explainer = getattr(explainer, 'create_explainer', None)
        ModelExplainer = getattr(explainer, 'ModelExplainer', None)
    except AttributeError as e:
        logger.warning(f"Some explainer components not available: {str(e)}")

# Validation
validation = _safe_import('validation', __name__)
if validation:
    try:
        validate_data = getattr(validation, 'validate_data', None)
        DataValidator = getattr(validation, 'DataValidator', None)
    except AttributeError as e:
        logger.warning(f"Some validation components not available: {str(e)}")

# Public API functions

def get_available_modules() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available modules in the ML package.
    
    Returns:
        Dictionary with module availability, errors, and metadata
    """
    return _MODULE_REGISTRY.copy()

def get_available_models() -> Dict[str, bool]:
    """
    Get availability status of different model types.
    
    Returns:
        Dictionary mapping model types to availability status
    """
    return {
        'tabular': _MODULE_REGISTRY['tabular_models']['available'],
        'timeseries': _MODULE_REGISTRY['timeseries_models']['available'],
        'text': _MODULE_REGISTRY['text_models']['available'],
        'anomaly': _MODULE_REGISTRY['anomaly_models']['available'],
        'clustering': _MODULE_REGISTRY['clustering_models']['available'],
        'deep_learning': _MODULE_REGISTRY['deep_models']['available'],
        'ensemble': _MODULE_REGISTRY['ensemble_models']['available'],
        'auto_pipeline': _MODULE_REGISTRY['auto_pipeline']['available']
    }

def get_available_analyzers() -> List[str]:
    """
    Get list of available analyzer types.
    
    Returns:
        List of available analyzer type names
    """
    return list(_ANALYZER_REGISTRY.keys())

def create_analyzer(
    data_type: str,
    task_type: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Factory function to create an analyzer for a specific data type.
    
    Args:
        data_type: Type of data ('tabular', 'timeseries', 'text', 'anomaly', 'clustering')
        task_type: Specific task type (optional)
        **kwargs: Additional arguments for analyzer creation
        
    Returns:
        Analyzer instance for the specified data type
        
    Raises:
        ValueError: If data_type is not available or supported
    """
    if data_type not in _MODEL_FACTORIES:
        available_types = list(_MODEL_FACTORIES.keys())
        raise ValueError(
            f"Data type '{data_type}' not available. "
            f"Available types: {available_types}"
        )
    
    factory_function = _MODEL_FACTORIES[data_type]
    
    try:
        if task_type:
            return factory_function(task_type=task_type, **kwargs)
        else:
            return factory_function(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create {data_type} analyzer: {str(e)}")

def get_model_capabilities() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed capabilities of available models.
    
    Returns:
        Dictionary with detailed model capabilities and requirements
    """
    capabilities = {}
    
    # Define model capabilities
    model_info = {
        'tabular': {
            'tasks': ['classification', 'regression'],
            'algorithms': ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression', 'SVM'],
            'data_types': ['numerical', 'categorical', 'mixed'],
            'min_samples': 10,
            'max_features': 10000,
            'supports_missing': True,
            'supports_categorical': True,
            'interpretable': True,
            'scalable': True
        },
        'timeseries': {
            'tasks': ['forecasting', 'classification', 'anomaly_detection'],
            'algorithms': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM', 'GRU', 'TFT'],
            'data_types': ['univariate', 'multivariate'],
            'min_samples': 50,
            'max_features': 1000,
            'supports_missing': True,
            'supports_seasonality': True,
            'supports_trends': True,
            'interpretable': False,
            'scalable': True
        },
        'text': {
            'tasks': ['classification', 'regression', 'sentiment_analysis', 'topic_modeling'],
            'algorithms': ['TF-IDF', 'Word2Vec', 'BERT', 'RoBERTa', 'DistilBERT'],
            'data_types': ['text', 'documents'],
            'min_samples': 20,
            'max_features': float('inf'),
            'supports_multilingual': True,
            'supports_preprocessing': True,
            'interpretable': False,
            'scalable': False
        },
        'anomaly': {
            'tasks': ['anomaly_detection', 'outlier_detection'],
            'algorithms': ['IsolationForest', 'OneClassSVM', 'AutoEncoder', 'LOF'],
            'data_types': ['numerical', 'mixed'],
            'min_samples': 100,
            'max_features': 1000,
            'supports_missing': False,
            'unsupervised': True,
            'interpretable': False,
            'scalable': True
        },
        'clustering': {
            'tasks': ['clustering', 'segmentation'],
            'algorithms': ['KMeans', 'DBSCAN', 'AgglomerativeClustering', 'GaussianMixture'],
            'data_types': ['numerical', 'mixed'],
            'min_samples': 50,
            'max_features': 1000,
            'supports_missing': False,
            'unsupervised': True,
            'auto_k_selection': True,
            'interpretable': True,
            'scalable': True
        },
        'deep_learning': {
            'tasks': ['classification', 'regression', 'forecasting', 'nlp'],
            'algorithms': ['MLP', 'CNN', 'LSTM', 'Transformer', 'TabNet'],
            'data_types': ['numerical', 'text', 'images', 'timeseries'],
            'min_samples': 1000,
            'max_features': float('inf'),
            'requires_gpu': False,
            'supports_gpu': True,
            'interpretable': False,
            'scalable': True
        },
        'ensemble': {
            'tasks': ['any'],
            'algorithms': ['Voting', 'Stacking', 'Blending', 'Bagging', 'Boosting'],
            'data_types': ['any'],
            'min_samples': 100,
            'max_features': 1000,
            'requires_base_models': True,
            'improves_performance': True,
            'interpretable': False,
            'scalable': False
        }
    }
    
    # Add availability information
    available_models = get_available_models()
    
    for model_type, info in model_info.items():
        if model_type in available_models:
            capabilities[model_type] = {
                'available': available_models[model_type],
                **info
            }
            
            if not available_models[model_type]:
                error = _MODULE_REGISTRY.get(f'{model_type}_models', {}).get('error')
                if error:
                    capabilities[model_type]['error'] = error
    
    return capabilities

def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information.
    
    Returns:
        Dictionary with package metadata and status
    """
    return {
        **PACKAGE_INFO,
        'modules': get_available_modules(),
        'models': get_available_models(),
        'analyzers': get_available_analyzers(),
        'capabilities': get_model_capabilities(),
        'python_version': sys.version,
        'package_path': str(Path(__file__).parent)
    }

def get_system_requirements() -> Dict[str, Any]:
    """
    Get system requirements and recommendations.
    
    Returns:
        Dictionary with system requirements and recommendations
    """
    requirements = {
        'python_version': '>=3.8',
        'core_dependencies': [
            'pandas>=1.3.0',
            'numpy>=1.20.0',
            'scikit-learn>=1.0.0'
        ],
        'optional_dependencies': {
            'deep_learning': ['torch>=1.9.0', 'tensorflow>=2.6.0'],
            'timeseries': ['prophet>=1.0', 'statsmodels>=0.12.0'],
            'text_processing': ['transformers>=4.10.0', 'sentence-transformers>=2.0.0'],
            'visualization': ['plotly>=5.0.0', 'seaborn>=0.11.0'],
            'optimization': ['optuna>=2.0.0', 'hyperopt>=0.2.0'],
            'experiment_tracking': ['mlflow>=1.20.0'],
            'feature_engineering': ['category-encoders>=2.3.0']
        },
        'system_recommendations': {
            'memory': '8GB+ RAM for large datasets',
            'cpu': '4+ cores recommended for parallel processing',
            'gpu': 'CUDA-compatible GPU for deep learning (optional)',
            'storage': '1GB+ free space for model caching'
        },
        'performance_tips': [
            'Use GPU acceleration for deep learning models',
            'Enable parallel processing for model selection',
            'Consider data sampling for very large datasets',
            'Use feature selection for high-dimensional data'
        ]
    }
    
    return requirements

def validate_installation() -> Dict[str, Any]:
    """
    Validate the ML package installation and dependencies.
    
    Returns:
        Validation results with status and recommendations
    """
    validation_results = {
        'status': 'healthy',
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check core modules
    core_modules = ['auto_pipeline', 'model_selection', 'preprocessing', 'evaluation']
    missing_core = []
    
    for module in core_modules:
        if not _MODULE_REGISTRY[module]['available']:
            missing_core.append(module)
    
    if missing_core:
        validation_results['status'] = 'critical'
        validation_results['issues'].extend([
            f"Core module '{module}' not available" for module in missing_core
        ])
    
    # Check model modules
    model_modules = ['tabular_models', 'timeseries_models', 'text_models']
    available_models = sum(1 for module in model_modules if _MODULE_REGISTRY[module]['available'])
    
    if available_models == 0:
        validation_results['status'] = 'critical'
        validation_results['issues'].append("No model modules available")
    elif available_models < len(model_modules):
        validation_results['warnings'].append(
            f"Only {available_models}/{len(model_modules)} model types available"
        )
    
    # Check optional dependencies
    try:
        import torch
        validation_results['gpu_support'] = torch.cuda.is_available()
    except ImportError:
        validation_results['warnings'].append("PyTorch not available - deep learning disabled")
    
    try:
        import mlflow
        validation_results['experiment_tracking'] = True
    except ImportError:
        validation_results['warnings'].append("MLflow not available - experiment tracking disabled")
    
    # Generate recommendations
    if validation_results['status'] == 'critical':
        validation_results['recommendations'].extend([
            "Install missing core dependencies",
            "Check Python version compatibility",
            "Reinstall the ML package"
        ])
    elif validation_results['warnings']:
        validation_results['recommendations'].extend([
            "Install optional dependencies for full functionality",
            "Consider upgrading to latest package version"
        ])
    else:
        validation_results['recommendations'].append("Installation is complete and healthy")
    
    return validation_results

# Legacy compatibility - maintain backward compatibility
def get_analyzer_registry() -> Dict[str, Type]:
    """Legacy function for backward compatibility."""
    return _ANALYZER_REGISTRY.copy()

def get_model_factory_registry() -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    return _MODEL_FACTORIES.copy()

# Define public API - these are the main exports
__all__ = [
    # Package metadata
    '__version__', '__author__', '__description__',
    'PACKAGE_INFO',
    
    # Core classes (if available)
    'AutoPipeline', 'PipelineConfig', 'DatasetDetector',
    'ModelSelector',
    
    # Analyzer classes (if available)  
    'TabularAnalyzer', 'TimeSeriesAnalyzer', 'TextAnalyzer',
    'AnomalyDetector', 'ClusteringAnalyzer', 'DeepLearningAnalyzer',
    
    # Utility classes
    'FeatureEngineer', 'DataPreprocessor', 'ModelEvaluator', 'ModelExplainer',
    
    # Enums (if available)
    'DatasetType', 'TaskType', 'ExecutionMode',
    
    # Factory functions
    'create_analyzer', 'create_auto_pipeline', 'create_model_selector',
    'create_tabular_analyzer', 'create_timeseries_analyzer', 'create_text_analyzer',
    'create_anomaly_detector', 'create_clustering_analyzer',
    'create_feature_engineer', 'create_preprocessor', 'create_evaluator', 'create_explainer',
    
    # Main analysis functions
    'quick_auto_analysis', 'auto_analyze_dataset', 'auto_select_and_train_models',
    
    # Information and utility functions
    'get_available_modules', 'get_available_models', 'get_available_analyzers',
    'get_model_capabilities', 'get_package_info', 'get_system_requirements',
    'validate_installation',
    
    # Legacy compatibility
    'get_analyzer_registry', 'get_model_factory_registry'
]

# Remove None values from __all__ (for missing imports)
__all__ = [item for item in __all__ if globals().get(item) is not None]

# Package initialization logging
logger.info(f"Auto-Analyst ML package v{__version__} initialized")
available_count = sum(1 for status in _MODULE_REGISTRY.values() if status['available'])
total_count = len(_MODULE_REGISTRY)
logger.info(f"Module availability: {available_count}/{total_count} modules loaded")

# Validate critical components
if not _MODULE_REGISTRY['auto_pipeline']['available']:
    logger.warning("Auto-pipeline not available - core functionality may be limited")

if available_count < total_count // 2:
    logger.warning("Many modules unavailable - check dependencies")

