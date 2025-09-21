"""
ML Service for Auto-Analyst Platform

This service orchestrates the complete machine learning workflow including:
- Dataset validation and preparation
- AutoML pipeline execution and monitoring
- Model training, evaluation, and selection
- Feature engineering and explanation generation
- Remote execution support (Kaggle/Colab)
- Results formatting for dashboard display
- Model persistence and deployment preparation

The ML Service acts as the central coordinator between data processing,
ML pipeline execution, and result presentation, ensuring seamless
integration across the Auto-Analyst platform.

Features:
- Comprehensive AutoML pipeline orchestration
- Real-time analysis monitoring and progress tracking
- Intelligent resource management (CPU/GPU/Remote)
- Advanced error handling and recovery mechanisms
- Model explanation and interpretability generation
- Dashboard-ready result formatting
- Remote execution with Kaggle/Colab integration
- Model versioning and experiment tracking
- Performance optimization and caching
- Production-ready scalability and monitoring

Usage:
    # Initialize ML service
    ml_service = MLService()
    
    # Run analysis
    analysis_id = await ml_service.create_analysis(dataset_id, config)
    result = await ml_service.run_analysis(analysis_id)
    
    # Monitor progress
    status = await ml_service.get_analysis_status(analysis_id)
    
    # Get results
    dashboard_data = await ml_service.get_analysis_results(analysis_id)
"""

import asyncio
import logging
import warnings
import time
import uuid
import json
import pickle
import tempfile
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import concurrent.futures
import multiprocessing as mp

# Core libraries
import pandas as pd
import numpy as np

# Database and models
try:
    from backend.models.database import get_db_session, Dataset, Analysis, MLModel, User
    from backend.models.schemas import (
        AnalysisRequest, AnalysisResponse, AnalysisStatusEnum,
        ExecutionModeEnum, TaskTypeEnum, DatasetTypeEnum
    )
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database models not available")

# ML pipeline integration
try:
    from ml import (
        auto_analyze_dataset, quick_auto_analysis, create_auto_pipeline,
        get_available_models, validate_installation
    )
    ML_PIPELINE_AVAILABLE = True
except ImportError:
    ML_PIPELINE_AVAILABLE = False
    logging.warning("ML pipeline not available")

# Data service integration
try:
    from backend.services.data_service import DataService, create_data_service
    DATA_SERVICE_AVAILABLE = True
except ImportError:
    DATA_SERVICE_AVAILABLE = False
    logging.warning("Data service not available")

# Remote execution support
try:
    from backend.services.kaggle_service import KaggleService, create_kaggle_service
    KAGGLE_SERVICE_AVAILABLE = True
except ImportError:
    KAGGLE_SERVICE_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class AnalysisStatus(str, Enum):
    """Analysis execution status."""
    PENDING = "pending"
    VALIDATING = "validating"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REMOTE_SUBMITTED = "remote_submitted"
    REMOTE_RUNNING = "remote_running"

class ExecutionMode(str, Enum):
    """Execution mode for analyses."""
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu" 
    REMOTE_KAGGLE = "remote_kaggle"
    REMOTE_COLAB = "remote_colab"
    AUTO = "auto"

@dataclass
class MLConfig:
    """Configuration for ML service operations."""
    
    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    max_execution_time: int = 3600  # 1 hour
    enable_parallel: bool = True
    n_jobs: int = -1
    
    # Pipeline settings
    auto_model_selection: bool = True
    max_models_to_try: int = 8
    optimization_budget: int = 100
    enable_ensemble: bool = True
    early_stopping: bool = True
    
    # Feature engineering
    enable_feature_engineering: bool = True
    enable_feature_selection: bool = True
    
    # Evaluation settings
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Explanation settings
    generate_explanations: bool = True
    explanation_samples: int = 1000
    
    # Performance settings
    enable_caching: bool = True
    cache_results: bool = True
    
    # Remote execution
    remote_timeout: int = 3600
    enable_remote_fallback: bool = True
    
    # Monitoring
    enable_progress_tracking: bool = True
    save_intermediate_results: bool = True

@dataclass 
class AnalysisContext:
    """Context information for an analysis run."""
    
    analysis_id: str
    user_id: Optional[int]
    dataset_id: int
    dataset_path: str
    target_column: Optional[str]
    task_type: Optional[str]
    dataset_type: Optional[str]
    execution_mode: ExecutionMode
    config: MLConfig
    
    # Runtime information
    status: AnalysisStatus = AnalysisStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_stage: str = "initialization"
    progress: float = 0.0
    
    # Results
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Remote execution
    remote_execution_id: Optional[str] = None
    remote_platform: Optional[str] = None

@dataclass
class AnalysisResult:
    """Result of ML analysis execution."""
    
    analysis_id: str
    status: AnalysisStatus
    dataset_analysis: Dict[str, Any]
    model_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    explanations: Optional[Dict[str, Any]]
    
    # Dashboard data
    dashboard_data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    
    # Metadata
    execution_time: float
    model_count: int
    best_model_name: str
    data_quality_score: float
    
    # Files and artifacts
    model_artifacts: Optional[Dict[str, str]] = None
    prediction_function: Optional[Callable] = None
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class MLService:
    """
    Comprehensive ML service for orchestrating machine learning workflows.
    
    This service manages the complete ML lifecycle from dataset validation
    through model training, evaluation, and result presentation.
    """
    
    def __init__(self, config: Optional[MLConfig] = None, cache_dir: Optional[str] = None):
        """
        Initialize the ML service.
        
        Args:
            config: ML service configuration
            cache_dir: Directory for caching results and models
        """
        self.config = config or MLConfig()
        
        # Setup directories
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "auto_analyst_ml_cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.cache_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Analysis tracking
        self.active_analyses: Dict[str, AnalysisContext] = {}
        self.analysis_results: Dict[str, AnalysisResult] = {}
        
        # Services
        self.data_service = create_data_service() if DATA_SERVICE_AVAILABLE else None
        self.kaggle_service = create_kaggle_service() if KAGGLE_SERVICE_AVAILABLE else None
        
        # Performance tracking
        self.performance_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_execution_time': 0.0,
            'cache_hits': 0,
            'remote_executions': 0
        }
        
        # Validate dependencies
        self._validate_dependencies()
        
        logger.info("MLService initialized successfully")
    
    async def create_analysis(
        self,
        dataset_id: int,
        user_id: Optional[int] = None,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.AUTO,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new ML analysis.
        
        Args:
            dataset_id: ID of the dataset to analyze
            user_id: ID of the requesting user
            target_column: Target column for supervised learning
            task_type: Type of ML task (classification, regression, etc.)
            execution_mode: Where to execute the analysis
            config: Custom configuration parameters
            
        Returns:
            Analysis ID for tracking
            
        Raises:
            ValueError: If dataset is invalid or configuration is incorrect
            RuntimeError: If dependencies are not available
        """
        try:
            # Generate analysis ID
            analysis_id = str(uuid.uuid4())
            
            # Validate dataset
            dataset_info = await self._validate_dataset(dataset_id)
            
            # Prepare configuration
            analysis_config = self._prepare_config(config)
            
            # Detect execution mode if AUTO
            if execution_mode == ExecutionMode.AUTO:
                execution_mode = await self._detect_optimal_execution_mode(dataset_info, user_id)
            
            # Create analysis context
            context = AnalysisContext(
                analysis_id=analysis_id,
                user_id=user_id,
                dataset_id=dataset_id,
                dataset_path=dataset_info['file_path'],
                target_column=target_column,
                task_type=task_type,
                dataset_type=None,  # Will be detected during analysis
                execution_mode=execution_mode,
                config=analysis_config
            )
            
            # Store context
            self.active_analyses[analysis_id] = context
            
            # Save to database if available
            if DATABASE_AVAILABLE:
                await self._create_analysis_record(context)
            
            logger.info(f"Analysis created: {analysis_id} for dataset {dataset_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Failed to create analysis: {str(e)}")
            raise
    
    async def run_analysis(
        self,
        analysis_id: str,
        background: bool = True
    ) -> Union[AnalysisResult, str]:
        """
        Execute ML analysis.
        
        Args:
            analysis_id: ID of the analysis to run
            background: Whether to run in background (returns status) or wait for completion
            
        Returns:
            Analysis result if background=False, else status message
            
        Raises:
            ValueError: If analysis ID is invalid
            RuntimeError: If execution fails
        """
        try:
            if analysis_id not in self.active_analyses:
                raise ValueError(f"Analysis not found: {analysis_id}")
            
            context = self.active_analyses[analysis_id]
            
            if background:
                # Start analysis in background
                asyncio.create_task(self._execute_analysis(context))
                return f"Analysis {analysis_id} started in background"
            else:
                # Execute and wait for completion
                result = await self._execute_analysis(context)
                return result
                
        except Exception as e:
            logger.error(f"Failed to run analysis {analysis_id}: {str(e)}")
            raise
    
    async def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get current status of an analysis.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Status information dictionary
        """
        try:
            if analysis_id not in self.active_analyses:
                # Check if completed
                if analysis_id in self.analysis_results:
                    result = self.analysis_results[analysis_id]
                    return {
                        'analysis_id': analysis_id,
                        'status': result.status.value,
                        'progress': 100.0,
                        'current_stage': 'completed',
                        'execution_time': result.execution_time,
                        'error_message': result.error_message
                    }
                else:
                    return {
                        'analysis_id': analysis_id,
                        'status': 'not_found',
                        'error': 'Analysis not found'
                    }
            
            context = self.active_analyses[analysis_id]
            
            # Calculate execution time
            execution_time = 0.0
            if context.start_time:
                if context.end_time:
                    execution_time = (context.end_time - context.start_time).total_seconds()
                else:
                    execution_time = (datetime.now() - context.start_time).total_seconds()
            
            return {
                'analysis_id': analysis_id,
                'status': context.status.value,
                'progress': context.progress,
                'current_stage': context.current_stage,
                'execution_time': execution_time,
                'execution_mode': context.execution_mode.value,
                'error_message': context.error_message,
                'remote_execution_id': context.remote_execution_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis status: {str(e)}")
            return {
                'analysis_id': analysis_id,
                'status': 'error',
                'error': str(e)
            }
    
    async def get_analysis_results(self, analysis_id: str) -> Optional[AnalysisResult]:
        """
        Get results of completed analysis.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Analysis results if available
        """
        try:
            if analysis_id in self.analysis_results:
                return self.analysis_results[analysis_id]
            
            # Check database if available
            if DATABASE_AVAILABLE:
                result = await self._load_analysis_from_db(analysis_id)
                if result:
                    return result
            
            logger.warning(f"Analysis results not found: {analysis_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get analysis results: {str(e)}")
            return None
    
    async def cancel_analysis(self, analysis_id: str) -> bool:
        """
        Cancel a running analysis.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            True if cancelled successfully
        """
        try:
            if analysis_id not in self.active_analyses:
                return False
            
            context = self.active_analyses[analysis_id]
            
            # Update status
            context.status = AnalysisStatus.CANCELLED
            context.end_time = datetime.now()
            context.error_message = "Analysis cancelled by user"
            
            # Cancel remote execution if applicable
            if context.remote_execution_id and self.kaggle_service:
                await self.kaggle_service.cancel_execution(context.remote_execution_id)
            
            # Update database
            if DATABASE_AVAILABLE:
                await self._update_analysis_status(analysis_id, AnalysisStatus.CANCELLED)
            
            logger.info(f"Analysis cancelled: {analysis_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel analysis: {str(e)}")
            return False
    
    async def list_analyses(
        self,
        user_id: Optional[int] = None,
        status: Optional[AnalysisStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List analyses with optional filtering.
        
        Args:
            user_id: Filter by user ID
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of analysis summaries
        """
        try:
            analyses = []
            
            # Include active analyses
            for context in self.active_analyses.values():
                if user_id and context.user_id != user_id:
                    continue
                if status and context.status != status:
                    continue
                
                analyses.append({
                    'analysis_id': context.analysis_id,
                    'dataset_id': context.dataset_id,
                    'status': context.status.value,
                    'progress': context.progress,
                    'execution_mode': context.execution_mode.value,
                    'created_at': context.start_time.isoformat() if context.start_time else None
                })
            
            # Include completed analyses
            for result in self.analysis_results.values():
                if user_id and hasattr(result, 'user_id') and result.user_id != user_id:
                    continue
                if status and result.status != status:
                    continue
                
                analyses.append({
                    'analysis_id': result.analysis_id,
                    'status': result.status.value,
                    'best_model': result.best_model_name,
                    'execution_time': result.execution_time,
                    'data_quality': result.data_quality_score
                })
            
            # Sort by creation time and limit
            analyses = sorted(analyses, key=lambda x: x.get('created_at', ''), reverse=True)
            
            return analyses[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list analyses: {str(e)}")
            return []
    
    async def get_model_predictions(
        self,
        analysis_id: str,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Get predictions from trained model.
        
        Args:
            analysis_id: Analysis ID containing the trained model
            input_data: Data to make predictions on
            
        Returns:
            Predictions and metadata
        """
        try:
            # Get analysis result
            result = await self.get_analysis_results(analysis_id)
            if not result:
                raise ValueError(f"Analysis not found: {analysis_id}")
            
            if not result.prediction_function:
                raise ValueError("No prediction function available")
            
            # Make predictions
            predictions = result.prediction_function(input_data)
            
            return {
                'analysis_id': analysis_id,
                'predictions': predictions,
                'model_name': result.best_model_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    # Private execution methods
    
    async def _execute_analysis(self, context: AnalysisContext) -> AnalysisResult:
        """Execute the complete analysis workflow."""
        try:
            # Update status and start tracking
            context.status = AnalysisStatus.VALIDATING
            context.start_time = datetime.now()
            context.progress = 0.0
            
            logger.info(f"Starting analysis execution: {context.analysis_id}")
            
            # Stage 1: Dataset Validation and Preparation (0-10%)
            await self._update_progress(context, "Dataset validation", 5.0)
            
            if not self.data_service:
                raise RuntimeError("Data service not available")
            
            # Load dataset
            dataset_result = await self.data_service.load_dataset(context.dataset_path)
            context.current_stage = "dataset_loaded"
            await self._update_progress(context, "Dataset loaded", 10.0)
            
            # Stage 2: Data Preprocessing (10-30%)
            context.status = AnalysisStatus.PREPARING
            await self._update_progress(context, "Data preprocessing", 15.0)
            
            # Preprocess dataset
            preprocessed_result = await self.data_service.preprocess_dataset(
                dataset_result.dataframe,
                target_column=context.target_column
            )
            await self._update_progress(context, "Data preprocessing complete", 25.0)
            
            # Prepare for ML
            ml_data = await self.data_service.prepare_for_ml(
                preprocessed_result.dataframe,
                target_column=context.target_column,
                task_type=context.task_type
            )
            
            # Update context with detected information
            context.task_type = ml_data['task_type']
            context.dataset_type = 'tabular'  # Default, could be enhanced
            
            await self._update_progress(context, "ML preparation complete", 30.0)
            
            # Stage 3: Choose Execution Path
            if context.execution_mode in [ExecutionMode.REMOTE_KAGGLE, ExecutionMode.REMOTE_COLAB]:
                result = await self._execute_remote_analysis(context, ml_data)
            else:
                result = await self._execute_local_analysis(context, ml_data)
            
            # Stage 4: Finalize Results
            context.status = AnalysisStatus.COMPLETED
            context.end_time = datetime.now()
            context.progress = 100.0
            result.execution_time = (context.end_time - context.start_time).total_seconds()
            
            # Store results
            self.analysis_results[context.analysis_id] = result
            
            # Update performance stats
            self._update_performance_stats(result)
            
            # Save to database
            if DATABASE_AVAILABLE:
                await self._save_analysis_results(result)
            
            # Clean up active analysis
            if context.analysis_id in self.active_analyses:
                del self.active_analyses[context.analysis_id]
            
            logger.info(f"Analysis completed successfully: {context.analysis_id}")
            return result
            
        except Exception as e:
            # Handle errors
            context.status = AnalysisStatus.FAILED
            context.end_time = datetime.now()
            context.error_message = str(e)
            
            # Create error result
            result = self._create_error_result(context, e)
            self.analysis_results[context.analysis_id] = result
            
            # Update performance stats
            self.performance_stats['failed_analyses'] += 1
            
            # Update database
            if DATABASE_AVAILABLE:
                await self._update_analysis_status(context.analysis_id, AnalysisStatus.FAILED, str(e))
            
            logger.error(f"Analysis failed: {context.analysis_id} - {str(e)}")
            logger.error(traceback.format_exc())
            
            return result
    
    async def _execute_local_analysis(
        self,
        context: AnalysisContext,
        ml_data: Dict[str, Any]
    ) -> AnalysisResult:
        """Execute analysis locally using auto_pipeline."""
        try:
            context.status = AnalysisStatus.RUNNING
            await self._update_progress(context, "Starting ML pipeline", 35.0)
            
            if not ML_PIPELINE_AVAILABLE:
                raise RuntimeError("ML pipeline not available")
            
            # Prepare configuration for auto pipeline
            pipeline_config = {
                'execution_mode': 'local',
                'max_execution_time': context.config.max_execution_time,
                'optimization_budget': context.config.optimization_budget,
                'enable_ensemble': context.config.enable_ensemble,
                'generate_explanations': context.config.generate_explanations,
                'cross_validation_folds': context.config.cross_validation_folds,
                'enable_parallel': context.config.enable_parallel
            }
            
            await self._update_progress(context, "Running AutoML pipeline", 40.0)
            
            # Extract data for pipeline
            X = ml_data['splits']['X_train']
            y = ml_data['splits'].get('y_train')
            
            # Create temporary dataset file for pipeline
            temp_df = X.copy()
            if y is not None:
                temp_df[context.target_column or 'target'] = y
            
            # Run auto analysis
            pipeline_result = await auto_analyze_dataset(
                df=temp_df,
                target_column=context.target_column,
                user_id=str(context.user_id) if context.user_id else None,
                config=pipeline_config
            )
            
            await self._update_progress(context, "ML pipeline complete", 85.0)
            
            # Process results
            result = await self._process_pipeline_results(
                context, pipeline_result, ml_data
            )
            
            await self._update_progress(context, "Results processing complete", 95.0)
            
            return result
            
        except Exception as e:
            logger.error(f"Local analysis execution failed: {str(e)}")
            raise
    
    async def _execute_remote_analysis(
        self,
        context: AnalysisContext,
        ml_data: Dict[str, Any]
    ) -> AnalysisResult:
        """Execute analysis on remote platform (Kaggle/Colab)."""
        try:
            context.status = AnalysisStatus.REMOTE_SUBMITTED
            await self._update_progress(context, "Submitting to remote platform", 35.0)
            
            if not self.kaggle_service:
                raise RuntimeError("Kaggle service not available")
            
            # Prepare data for remote execution
            remote_config = await self._prepare_remote_config(context, ml_data)
            
            # Submit to remote platform
            if context.execution_mode == ExecutionMode.REMOTE_KAGGLE:
                execution_result = await self.kaggle_service.submit_analysis(
                    user_id=context.user_id,
                    dataset_data=ml_data,
                    config=remote_config
                )
                context.remote_execution_id = execution_result['execution_id']
                context.remote_platform = 'kaggle'
            
            elif context.execution_mode == ExecutionMode.REMOTE_COLAB:
                # Placeholder for Colab integration
                raise NotImplementedError("Colab execution not yet implemented")
            
            await self._update_progress(context, "Remote execution submitted", 45.0)
            
            # Monitor remote execution
            context.status = AnalysisStatus.REMOTE_RUNNING
            result = await self._monitor_remote_execution(context)
            
            return result
            
        except Exception as e:
            logger.error(f"Remote analysis execution failed: {str(e)}")
            
            # Try local fallback if enabled
            if context.config.enable_remote_fallback:
                logger.info("Attempting local fallback")
                context.execution_mode = ExecutionMode.LOCAL_CPU
                return await self._execute_local_analysis(context, ml_data)
            else:
                raise
    
    async def _monitor_remote_execution(self, context: AnalysisContext) -> AnalysisResult:
        """Monitor remote execution and retrieve results."""
        try:
            start_time = time.time()
            timeout = context.config.remote_timeout
            
            while time.time() - start_time < timeout:
                # Check execution status
                if context.remote_platform == 'kaggle' and self.kaggle_service:
                    status = await self.kaggle_service.get_execution_status(
                        context.remote_execution_id
                    )
                    
                    if status['status'] == 'completed':
                        await self._update_progress(context, "Remote execution completed", 85.0)
                        
                        # Retrieve results
                        remote_results = await self.kaggle_service.get_execution_results(
                            context.remote_execution_id
                        )
                        
                        # Process remote results
                        result = await self._process_remote_results(context, remote_results)
                        await self._update_progress(context, "Results retrieved", 95.0)
                        
                        return result
                    
                    elif status['status'] == 'failed':
                        raise RuntimeError(f"Remote execution failed: {status.get('error', 'Unknown error')}")
                    
                    # Update progress based on remote status
                    remote_progress = status.get('progress', 50)
                    overall_progress = 45 + (remote_progress * 0.4)  # Scale to 45-85% range
                    await self._update_progress(context, f"Remote execution: {status.get('stage', 'running')}", overall_progress)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Timeout reached
            raise TimeoutError(f"Remote execution timeout after {timeout} seconds")
            
        except Exception as e:
            logger.error(f"Remote monitoring failed: {str(e)}")
            raise
    
    async def _process_pipeline_results(
        self,
        context: AnalysisContext,
        pipeline_result: Dict[str, Any],
        ml_data: Dict[str, Any]
    ) -> AnalysisResult:
        """Process results from auto pipeline execution."""
        try:
            # Extract key information
            dataset_analysis = pipeline_result.get('dataset_analysis', {})
            model_results = pipeline_result.get('model_results', {})
            performance_metrics = model_results.get('performance_metrics', {})
            feature_importance = model_results.get('feature_importance', {})
            
            # Generate dashboard data
            dashboard_data = await self._create_dashboard_data(
                context, pipeline_result, ml_data
            )
            
            # Extract insights and recommendations
            insights = pipeline_result.get('insights', [])
            recommendations = pipeline_result.get('recommendations', [])
            
            # Create analysis result
            result = AnalysisResult(
                analysis_id=context.analysis_id,
                status=AnalysisStatus.COMPLETED,
                dataset_analysis=dataset_analysis,
                model_results=model_results,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance,
                explanations=pipeline_result.get('explanations', {}),
                dashboard_data=dashboard_data,
                insights=insights,
                recommendations=recommendations,
                execution_time=0.0,  # Will be set later
                model_count=len(model_results.get('model_comparison', [])),
                best_model_name=model_results.get('best_model_name', 'Unknown'),
                data_quality_score=dataset_analysis.get('data_quality_score', 0.0)
            )
            
            # Save model artifacts if available
            if 'prediction_pipeline_available' in pipeline_result and pipeline_result['prediction_pipeline_available']:
                result.model_artifacts = await self._save_model_artifacts(context, pipeline_result)
                result.prediction_function = self._create_prediction_function(context, pipeline_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline results processing failed: {str(e)}")
            raise
    
    async def _process_remote_results(
        self,
        context: AnalysisContext,
        remote_results: Dict[str, Any]
    ) -> AnalysisResult:
        """Process results from remote execution."""
        try:
            # Remote results should follow the same format as local pipeline
            # This is a simplified version - actual implementation would depend
            # on how remote results are structured
            
            results_data = remote_results.get('results', {})
            
            result = AnalysisResult(
                analysis_id=context.analysis_id,
                status=AnalysisStatus.COMPLETED,
                dataset_analysis=results_data.get('dataset_analysis', {}),
                model_results=results_data.get('model_results', {}),
                performance_metrics=results_data.get('performance_metrics', {}),
                feature_importance=results_data.get('feature_importance', {}),
                explanations=results_data.get('explanations', {}),
                dashboard_data=results_data.get('dashboard_data', {}),
                insights=results_data.get('insights', []),
                recommendations=results_data.get('recommendations', []),
                execution_time=0.0,  # Will be set later
                model_count=len(results_data.get('model_results', {}).get('model_comparison', [])),
                best_model_name=results_data.get('model_results', {}).get('best_model_name', 'Unknown'),
                data_quality_score=results_data.get('dataset_analysis', {}).get('data_quality_score', 0.0)
            )
            
            # Handle model artifacts from remote execution
            if 'model_artifacts_url' in remote_results:
                result.model_artifacts = await self._download_remote_artifacts(
                    context, remote_results['model_artifacts_url']
                )
            
            # Update performance stats
            self.performance_stats['remote_executions'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Remote results processing failed: {str(e)}")
            raise
    
    async def _create_dashboard_data(
        self,
        context: AnalysisContext,
        pipeline_result: Dict[str, Any],
        ml_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create dashboard-ready data visualization."""
        try:
            dashboard_data = pipeline_result.get('dashboard_data', {})
            
            # Enhance with additional data if needed
            if not dashboard_data:
                # Create basic dashboard data
                dashboard_data = {
                    'overview': {
                        'analysis_id': context.analysis_id,
                        'dataset_type': context.dataset_type,
                        'task_type': context.task_type,
                        'execution_mode': context.execution_mode.value,
                        'n_samples': ml_data.get('n_samples', 0),
                        'n_features': ml_data.get('n_features', 0)
                    },
                    'performance': pipeline_result.get('model_results', {}).get('performance_metrics', {}),
                    'model_info': {
                        'best_model': pipeline_result.get('model_results', {}).get('best_model_name', 'Unknown'),
                        'model_count': len(pipeline_result.get('model_results', {}).get('model_comparison', []))
                    },
                    'data_quality': {
                        'score': pipeline_result.get('dataset_analysis', {}).get('data_quality_score', 0.0)
                    }
                }
            
            # Add execution metadata
            dashboard_data['execution_info'] = {
                'analysis_id': context.analysis_id,
                'execution_mode': context.execution_mode.value,
                'start_time': context.start_time.isoformat() if context.start_time else None,
                'dataset_id': context.dataset_id
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard data creation failed: {str(e)}")
            return {}
    
    # Utility and helper methods
    
    async def _validate_dataset(self, dataset_id: int) -> Dict[str, Any]:
        """Validate dataset exists and is accessible."""
        try:
            if DATABASE_AVAILABLE:
                with get_db_session() as db:
                    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
                    if not dataset:
                        raise ValueError(f"Dataset not found: {dataset_id}")
                    
                    if dataset.status != 'processed':
                        raise ValueError(f"Dataset not ready for analysis: {dataset.status}")
                    
                    return {
                        'id': dataset.id,
                        'name': dataset.name,
                        'file_path': dataset.file_path,
                        'n_rows': dataset.num_rows,
                        'n_columns': dataset.num_columns,
                        'file_size': dataset.file_size
                    }
            else:
                # Mock validation for testing
                return {
                    'id': dataset_id,
                    'name': f'Dataset_{dataset_id}',
                    'file_path': f'/tmp/dataset_{dataset_id}.csv',
                    'n_rows': 1000,
                    'n_columns': 10,
                    'file_size': 1024000
                }
                
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            raise
    
    async def _detect_optimal_execution_mode(
        self,
        dataset_info: Dict[str, Any],
        user_id: Optional[int]
    ) -> ExecutionMode:
        """Detect optimal execution mode based on dataset and user capabilities."""
        try:
            # Check dataset size
            size_mb = dataset_info.get('file_size', 0) / (1024 * 1024)
            n_rows = dataset_info.get('n_rows', 0)
            
            # Check if user has remote credentials
            has_kaggle_creds = False
            if user_id and DATABASE_AVAILABLE and KAGGLE_SERVICE_AVAILABLE:
                has_kaggle_creds = await self.kaggle_service.user_has_credentials(user_id)
            
            # Decision logic
            if size_mb > 100 or n_rows > 50000:  # Large dataset
                if has_kaggle_creds:
                    return ExecutionMode.REMOTE_KAGGLE
                else:
                    return ExecutionMode.LOCAL_CPU  # Could be GPU if available
            else:
                return ExecutionMode.LOCAL_CPU
                
        except Exception as e:
            logger.warning(f"Could not detect optimal execution mode: {str(e)}")
            return ExecutionMode.LOCAL_CPU
    
    def _prepare_config(self, config: Optional[Dict[str, Any]]) -> MLConfig:
        """Prepare ML configuration from request parameters."""
        base_config = self.config
        
        if config:
            # Override base config with custom parameters
            for key, value in config.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
        
        return base_config
    
    async def _prepare_remote_config(
        self,
        context: AnalysisContext,
        ml_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare configuration for remote execution."""
        return {
            'analysis_id': context.analysis_id,
            'task_type': context.task_type,
            'target_column': context.target_column,
            'dataset_info': {
                'n_samples': ml_data.get('n_samples', 0),
                'n_features': ml_data.get('n_features', 0),
                'feature_names': ml_data.get('feature_names', [])
            },
            'pipeline_config': {
                'max_execution_time': context.config.max_execution_time,
                'optimization_budget': context.config.optimization_budget,
                'enable_ensemble': context.config.enable_ensemble,
                'generate_explanations': context.config.generate_explanations
            }
        }
    
    async def _update_progress(self, context: AnalysisContext, stage: str, progress: float):
        """Update analysis progress and stage."""
        context.current_stage = stage
        context.progress = min(progress, 100.0)
        
        # Update database if available
        if DATABASE_AVAILABLE:
            try:
                with get_db_session() as db:
                    analysis = db.query(Analysis).filter(Analysis.id == context.analysis_id).first()
                    if analysis:
                        # Store progress in metadata
                        metadata = analysis.metadata or {}
                        metadata.update({
                            'current_stage': stage,
                            'progress': progress,
                            'last_update': datetime.now().isoformat()
                        })
                        analysis.metadata = metadata
                        db.commit()
            except Exception as e:
                logger.warning(f"Failed to update progress in database: {str(e)}")
        
        logger.debug(f"Analysis {context.analysis_id}: {stage} ({progress:.1f}%)")
    
    def _create_error_result(self, context: AnalysisContext, error: Exception) -> AnalysisResult:
        """Create error result for failed analysis."""
        return AnalysisResult(
            analysis_id=context.analysis_id,
            status=AnalysisStatus.FAILED,
            dataset_analysis={},
            model_results={},
            performance_metrics={},
            feature_importance=None,
            explanations=None,
            dashboard_data={
                'error': {
                    'message': str(error),
                    'type': type(error).__name__,
                    'timestamp': datetime.now().isoformat()
                }
            },
            insights=[f"Analysis failed: {str(error)}"],
            recommendations=["Check data quality and try again"],
            execution_time=(context.end_time - context.start_time).total_seconds() if context.start_time and context.end_time else 0.0,
            model_count=0,
            best_model_name="None",
            data_quality_score=0.0,
            error_message=str(error)
        )
    
    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        issues = []
        
        if not DATA_SERVICE_AVAILABLE:
            issues.append("Data service not available")
        
        if not ML_PIPELINE_AVAILABLE:
            issues.append("ML pipeline not available")
        
        # Check ML pipeline status
        if ML_PIPELINE_AVAILABLE:
            try:
                ml_status = validate_installation()
                if ml_status['status'] != 'healthy':
                    issues.append(f"ML pipeline issues: {ml_status.get('issues', [])}")
            except:
                issues.append("ML pipeline validation failed")
        
        if issues:
            logger.warning(f"Dependency issues detected: {issues}")
        else:
            logger.info("All dependencies validated successfully")
    
    def _update_performance_stats(self, result: AnalysisResult):
        """Update performance statistics."""
        self.performance_stats['total_analyses'] += 1
        
        if result.status == AnalysisStatus.COMPLETED:
            self.performance_stats['successful_analyses'] += 1
            
            # Update average execution time
            total_time = (
                self.performance_stats['average_execution_time'] * 
                (self.performance_stats['successful_analyses'] - 1) + 
                result.execution_time
            )
            self.performance_stats['average_execution_time'] = (
                total_time / self.performance_stats['successful_analyses']
            )
    
    async def _save_model_artifacts(
        self,
        context: AnalysisContext,
        pipeline_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Save model artifacts to filesystem."""
        try:
            artifacts = {}
            
            # Create directory for this analysis
            analysis_dir = self.models_dir / context.analysis_id
            analysis_dir.mkdir(exist_ok=True)
            
            # Save model if available
            if 'best_model' in pipeline_result.get('model_results', {}):
                model_path = analysis_dir / "model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(pipeline_result['model_results']['best_model'], f)
                artifacts['model'] = str(model_path)
            
            # Save preprocessing pipeline if available
            if 'preprocessing_pipeline' in pipeline_result:
                prep_path = analysis_dir / "preprocessor.pkl"
                with open(prep_path, 'wb') as f:
                    pickle.dump(pipeline_result['preprocessing_pipeline'], f)
                artifacts['preprocessor'] = str(prep_path)
            
            # Save metadata
            metadata_path = analysis_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'analysis_id': context.analysis_id,
                    'model_name': pipeline_result.get('model_results', {}).get('best_model_name'),
                    'feature_names': pipeline_result.get('feature_names', []),
                    'created_at': datetime.now().isoformat()
                }, f, indent=2)
            artifacts['metadata'] = str(metadata_path)
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {str(e)}")
            return {}
    
    def _create_prediction_function(
        self,
        context: AnalysisContext,
        pipeline_result: Dict[str, Any]
    ) -> Callable:
        """Create prediction function for the trained model."""
        def predict(input_data):
            try:
                # This is a simplified implementation
                # In practice, would need to load and apply the full pipeline
                model = pipeline_result.get('model_results', {}).get('best_model')
                
                if model is None:
                    return {'error': 'No model available'}
                
                # Convert input to appropriate format
                if isinstance(input_data, dict):
                    df = pd.DataFrame([input_data])
                elif isinstance(input_data, list):
                    df = pd.DataFrame(input_data)
                else:
                    df = input_data
                
                # Make predictions
                predictions = model.predict(df)
                
                return {
                    'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                    'status': 'success'
                }
                
            except Exception as e:
                return {
                    'predictions': None,
                    'status': 'error',
                    'error': str(e)
                }
        
        return predict
    
    # Database integration methods
    
    async def _create_analysis_record(self, context: AnalysisContext):
        """Create analysis record in database."""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            with get_db_session() as db:
                analysis = Analysis(
                    name=f"Analysis_{context.analysis_id[:8]}",
                    description=f"Auto-generated analysis for dataset {context.dataset_id}",
                    task_type=context.task_type or 'unknown',
                    dataset_type=context.dataset_type or 'unknown',
                    target_column=context.target_column,
                    status=context.status.value,
                    execution_mode=context.execution_mode.value,
                    user_id=context.user_id,
                    dataset_id=context.dataset_id,
                    pipeline_config=context.config.__dict__,
                    started_at=context.start_time
                )
                db.add(analysis)
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to create analysis record: {str(e)}")
    
    async def _update_analysis_status(
        self,
        analysis_id: str,
        status: AnalysisStatus,
        error_message: Optional[str] = None
    ):
        """Update analysis status in database."""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            with get_db_session() as db:
                analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
                if analysis:
                    analysis.status = status.value
                    if error_message:
                        analysis.error_message = error_message
                    if status == AnalysisStatus.COMPLETED:
                        analysis.completed_at = datetime.now()
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update analysis status: {str(e)}")
    
    async def _save_analysis_results(self, result: AnalysisResult):
        """Save analysis results to database."""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            with get_db_session() as db:
                analysis = db.query(Analysis).filter(Analysis.id == result.analysis_id).first()
                if analysis:
                    analysis.status = result.status.value
                    analysis.best_model_name = result.best_model_name
                    analysis.performance_metrics = result.performance_metrics
                    analysis.feature_importance = result.feature_importance
                    analysis.execution_time = result.execution_time
                    analysis.completed_at = datetime.now()
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Failed to save analysis results: {str(e)}")
    
    async def _load_analysis_from_db(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Load analysis result from database."""
        if not DATABASE_AVAILABLE:
            return None
        
        try:
            with get_db_session() as db:
                analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
                if analysis and analysis.status == 'completed':
                    # Convert database record to AnalysisResult
                    # This is a simplified conversion
                    return AnalysisResult(
                        analysis_id=analysis_id,
                        status=AnalysisStatus(analysis.status),
                        dataset_analysis={},  # Would need to be stored separately
                        model_results={'best_model_name': analysis.best_model_name},
                        performance_metrics=analysis.performance_metrics or {},
                        feature_importance=analysis.feature_importance,
                        explanations={},
                        dashboard_data={},
                        insights=[],
                        recommendations=[],
                        execution_time=analysis.execution_time or 0.0,
                        model_count=1,
                        best_model_name=analysis.best_model_name or 'Unknown',
                        data_quality_score=0.0
                    )
                    
        except Exception as e:
            logger.error(f"Failed to load analysis from database: {str(e)}")
        
        return None
    
    # Health and monitoring methods
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'active_analyses': len(self.active_analyses),
                'completed_analyses': len(self.analysis_results),
                'performance_stats': self.performance_stats,
                'dependencies': {
                    'data_service': DATA_SERVICE_AVAILABLE,
                    'ml_pipeline': ML_PIPELINE_AVAILABLE,
                    'database': DATABASE_AVAILABLE,
                    'kaggle_service': KAGGLE_SERVICE_AVAILABLE,
                    'mlflow': MLFLOW_AVAILABLE
                },
                'cache_info': {
                    'cache_dir': str(self.cache_dir),
                    'models_dir': str(self.models_dir)
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup_old_analyses(self, days: int = 30):
        """Clean up old analysis results and artifacts."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            # Clean up in-memory results
            to_remove = []
            for analysis_id, result in self.analysis_results.items():
                # Check if result is old (would need timestamp in result)
                # This is a simplified cleanup
                if len(self.analysis_results) > 1000:  # Keep only recent 1000
                    to_remove.append(analysis_id)
                    
            for analysis_id in to_remove:
                del self.analysis_results[analysis_id]
                cleaned_count += 1
            
            # Clean up model artifacts
            if self.models_dir.exists():
                for analysis_dir in self.models_dir.iterdir():
                    if analysis_dir.is_dir():
                        # Check directory age
                        dir_time = datetime.fromtimestamp(analysis_dir.stat().st_mtime)
                        if dir_time < cutoff_date:
                            import shutil
                            shutil.rmtree(analysis_dir)
                            cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old analyses")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

# Factory and utility functions

def create_ml_service(config: Optional[MLConfig] = None, cache_dir: Optional[str] = None) -> MLService:
    """
    Factory function to create ML service instance.
    
    Args:
        config: Custom ML service configuration
        cache_dir: Custom cache directory
        
    Returns:
        Configured MLService instance
    """
    return MLService(config=config, cache_dir=cache_dir)

def get_ml_service() -> MLService:
    """Get ML service instance for dependency injection."""
    return create_ml_service()

# Example usage and testing
if __name__ == "__main__":
    async def example_usage():
        """Example usage of the ML service."""
        
        print("🤖 MLService Example Usage")
        print("=" * 50)
        
        # Initialize service
        ml_service = create_ml_service()
        
        # Check service health
        print("\n🏥 Service Health Check:")
        health = ml_service.get_service_health()
        print(f"Status: {health['status']}")
        print(f"Dependencies: {health['dependencies']}")
        
        # Example 1: Create and run analysis
        try:
            print("\n📊 Creating Analysis...")
            
            # Mock dataset ID for testing
            dataset_id = 1
            
            analysis_id = await ml_service.create_analysis(
                dataset_id=dataset_id,
                user_id=1,
                target_column='target',
                task_type='classification',
                execution_mode=ExecutionMode.LOCAL_CPU
            )
            
            print(f"✅ Analysis created: {analysis_id}")
            
            # Start analysis in background
            result_msg = await ml_service.run_analysis(analysis_id, background=True)
            print(f"📈 {result_msg}")
            
            # Monitor progress
            print("\n⏳ Monitoring Progress...")
            for i in range(5):  # Check 5 times
                status = await ml_service.get_analysis_status(analysis_id)
                print(f"   Status: {status['status']} - {status['current_stage']} ({status['progress']:.1f}%)")
                
                if status['status'] in ['completed', 'failed']:
                    break
                    
                await asyncio.sleep(2)  # Wait 2 seconds
            
            # Get final results
            final_status = await ml_service.get_analysis_status(analysis_id)
            if final_status['status'] == 'completed':
                print("✅ Analysis completed successfully!")
                
                results = await ml_service.get_analysis_results(analysis_id)
                if results:
                    print(f"   Best Model: {results.best_model_name}")
                    print(f"   Data Quality: {results.data_quality_score:.2f}")
                    print(f"   Execution Time: {results.execution_time:.1f}s")
                    print(f"   Insights: {len(results.insights)} generated")
            
        except Exception as e:
            print(f"❌ Example failed: {str(e)}")
        
        # Example 2: List analyses
        print("\n📋 Listing Analyses...")
        try:
            analyses = await ml_service.list_analyses(user_id=1, limit=10)
            print(f"Found {len(analyses)} analyses")
            for analysis in analyses[:3]:
                print(f"   {analysis['analysis_id'][:8]}: {analysis['status']}")
                
        except Exception as e:
            print(f"❌ List analyses failed: {str(e)}")
        
        # Example 3: Service statistics
        print("\n📊 Service Statistics:")
        stats = ml_service.performance_stats
        print(f"   Total Analyses: {stats['total_analyses']}")
        print(f"   Success Rate: {stats['successful_analyses']}/{stats['total_analyses']}")
        print(f"   Average Time: {stats['average_execution_time']:.1f}s")
        
        print(f"\n🎯 MLService example completed!")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except Exception as e:
        print(f"Example execution failed: {str(e)}")
