"""
Background Tasks Module for Auto-Analyst Platform

This module provides comprehensive background task management and orchestration
for the Auto-Analyst platform, handling all long-running operations including:

- ML training and model optimization
- Data preprocessing and validation
- Batch predictions and inference
- Model monitoring and drift detection
- Cleanup and maintenance operations
- Notification and alert processing
- Remote execution coordination
- Resource optimization tasks

The module supports multiple execution backends:
- Celery with Redis/RabbitMQ for distributed processing
- AsyncIO for lightweight concurrent tasks
- Threading for CPU-bound operations
- Remote execution on Kaggle, Colab, and cloud platforms

Features:
- Task scheduling and queuing
- Progress tracking and monitoring
- Error handling and retry logic
- Resource management and optimization
- Task dependencies and workflows
- Real-time status updates
- Performance metrics collection
- Automatic cleanup and maintenance

Architecture:
- TaskManager: Central task coordination and management
- TaskExecutor: Task execution with different backends
- TaskMonitor: Progress tracking and performance monitoring
- TaskScheduler: Periodic and scheduled task execution
- RemoteExecutor: Integration with external compute platforms
- TaskQueue: Priority-based task queuing system

Usage:
    from backend.tasks import task_manager, execute_analysis, process_dataset
    
    # Execute ML analysis
    task_id = await execute_analysis(analysis_id="123", config={...})
    
    # Process uploaded dataset
    result = await process_dataset(dataset_id=456, user_id=789)
    
    # Monitor task progress
    status = await task_manager.get_task_status(task_id)
    
    # Schedule periodic tasks
    task_manager.schedule_periodic("cleanup_old_artifacts", interval="1h")

Dependencies:
- Celery: Distributed task processing
- Redis: Message broker and result backend
- AsyncIO: Async task coordination
- MLflow: Experiment tracking integration
- Feast: Feature store operations
- Monitoring: Performance and drift detection
"""

import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import sys

# Core backend imports
from backend.config import settings

# Celery imports (optional)
try:
    from celery import Celery, Task
    from celery.result import AsyncResult
    from celery.signals import task_prerun, task_postrun, task_failure, task_success
    from celery.exceptions import Retry, WorkerLostError
    CELERY_AVAILABLE = True
except ImportError:
    Celery = None
    Task = object
    AsyncResult = None
    CELERY_AVAILABLE = False

# Redis imports (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# Database and services
try:
    from backend.models.database import get_db_session
    from backend.services.ml_service import MLService
    from backend.services.data_service import DataService
    from backend.services.insights_service import InsightsService
    from backend.services.mlops_service import MLOpsService
    from backend.utils.monitoring import MonitoringManager, log_info, log_warning, log_error
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"
    TIMEOUT = "timeout"

class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class TaskType(str, Enum):
    """Types of background tasks."""
    ML_TRAINING = "ml_training"
    DATA_PROCESSING = "data_processing"
    BATCH_PREDICTION = "batch_prediction"
    DRIFT_DETECTION = "drift_detection"
    MODEL_DEPLOYMENT = "model_deployment"
    CLEANUP = "cleanup"
    MONITORING = "monitoring"
    NOTIFICATION = "notification"
    REMOTE_EXECUTION = "remote_execution"

@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    execution_time: Optional[float] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class TaskConfig:
    """Task execution configuration."""
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60
    timeout: Optional[int] = None
    queue: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    callbacks: Dict[str, Callable] = field(default_factory=dict)

class TaskManager:
    """Central task management and coordination system."""
    
    def __init__(self):
        """Initialize task manager."""
        self.active_tasks: Dict[str, TaskResult] = {}
        self.task_history: Dict[str, TaskResult] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="task-")
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        self.monitoring_manager: Optional[MonitoringManager] = None
        self.redis_client: Optional[redis.Redis] = None
        self.celery_app: Optional[Celery] = None
        self._shutdown_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Initialize components
        self._initialize_redis()
        self._initialize_celery()
        self._start_cleanup_thread()
    
    def _initialize_redis(self):
        """Initialize Redis connection for task state."""
        if REDIS_AVAILABLE and settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                self.redis_client.ping()
                logger.info("Redis connection established for task management")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
    
    def _initialize_celery(self):
        """Initialize Celery application."""
        if CELERY_AVAILABLE and settings.REDIS_URL:
            try:
                self.celery_app = Celery(
                    'auto_analyst_tasks',
                    broker=settings.REDIS_URL,
                    backend=settings.REDIS_URL,
                    include=['backend.tasks']
                )
                
                # Configure Celery
                self.celery_app.conf.update(
                    task_serializer='json',
                    accept_content=['json'],
                    result_serializer='json',
                    timezone='UTC',
                    enable_utc=True,
                    task_track_started=True,
                    task_ignore_result=False,
                    result_expires=3600 * 24 * 7,  # 7 days
                    worker_prefetch_multiplier=1,
                    task_acks_late=True,
                    worker_disable_rate_limits=False,
                    task_default_queue='default',
                    task_routes={
                        'backend.tasks.train_model': {'queue': 'training'},
                        'backend.tasks.process_dataset': {'queue': 'processing'},
                        'backend.tasks.batch_predict': {'queue': 'prediction'},
                        'backend.tasks.cleanup_artifacts': {'queue': 'maintenance'},
                    },
                    task_default_retry_delay=60,
                    task_max_retries=3,
                )
                
                logger.info("Celery application initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Celery: {e}")
                self.celery_app = None
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            name="task-cleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker for task cleanup."""
        while not self._shutdown_event.is_set():
            try:
                self._cleanup_completed_tasks()
                self._cleanup_expired_results()
                self._update_task_metrics()
            except Exception as e:
                logger.error(f"Task cleanup error: {e}")
            
            # Sleep for 5 minutes
            self._shutdown_event.wait(300)
    
    def _cleanup_completed_tasks(self):
        """Clean up completed tasks from memory."""
        current_time = datetime.now()
        cleanup_threshold = current_time - timedelta(hours=24)
        
        tasks_to_remove = []
        for task_id, result in self.task_history.items():
            if (result.completed_at and 
                result.completed_at < cleanup_threshold and
                result.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            self.task_history.pop(task_id, None)
            logger.debug(f"Cleaned up task history: {task_id}")
    
    def _cleanup_expired_results(self):
        """Clean up expired task results from Redis."""
        if not self.redis_client:
            return
        
        try:
            # Get all task result keys
            pattern = "task_result:*"
            keys = self.redis_client.keys(pattern)
            
            current_time = time.time()
            expired_keys = []
            
            for key in keys:
                try:
                    result_data = self.redis_client.get(key)
                    if result_data:
                        result = json.loads(result_data)
                        if (result.get('completed_at') and 
                            current_time - result['completed_at'] > 7 * 24 * 3600):  # 7 days
                            expired_keys.append(key)
                except Exception:
                    continue
            
            if expired_keys:
                self.redis_client.delete(*expired_keys)
                logger.debug(f"Cleaned up {len(expired_keys)} expired task results")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired results: {e}")
    
    def _update_task_metrics(self):
        """Update task execution metrics."""
        if self.monitoring_manager:
            try:
                active_count = len(self.active_tasks)
                pending_count = len([t for t in self.active_tasks.values() 
                                   if t.status == TaskStatus.PENDING])
                running_count = len([t for t in self.active_tasks.values() 
                                   if t.status == TaskStatus.RUNNING])
                
                # Update metrics
                self.monitoring_manager.update_gauge('active_tasks_count', active_count)
                self.monitoring_manager.update_gauge('pending_tasks_count', pending_count)
                self.monitoring_manager.update_gauge('running_tasks_count', running_count)
                
            except Exception as e:
                logger.error(f"Failed to update task metrics: {e}")
    
    async def submit_task(
        self,
        task_func: Callable,
        *args,
        task_config: Optional[TaskConfig] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_func: Function to execute
            *args: Function arguments
            task_config: Task configuration
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        config = task_config or TaskConfig()
        
        # Create task result
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            started_at=datetime.now(),
            metadata=config.metadata
        )
        
        self.active_tasks[task_id] = result
        
        try:
            # Use Celery if available and configured
            if self.celery_app and hasattr(task_func, 'delay'):
                celery_result = task_func.delay(*args, **kwargs)
                result.metadata['celery_task_id'] = celery_result.id
                
                # Start monitoring Celery task
                asyncio.create_task(self._monitor_celery_task(task_id, celery_result))
                
            else:
                # Use thread pool for async execution
                if asyncio.iscoroutinefunction(task_func):
                    asyncio.create_task(self._execute_async_task(task_id, task_func, *args, **kwargs))
                else:
                    future = self.thread_pool.submit(self._execute_sync_task, task_id, task_func, *args, **kwargs)
                    result.metadata['thread_future'] = future
            
            log_info(f"Task submitted: {task_id}", extra={'task_id': task_id, 'function': task_func.__name__})
            return task_id
            
        except Exception as e:
            result.status = TaskStatus.FAILURE
            result.error = str(e)
            result.traceback = traceback.format_exc()
            result.completed_at = datetime.now()
            
            log_error(f"Task submission failed: {task_id}", exception=e, extra={'task_id': task_id})
            return task_id
    
    async def _monitor_celery_task(self, task_id: str, celery_result: AsyncResult):
        """Monitor Celery task execution."""
        try:
            result = self.active_tasks[task_id]
            
            while not celery_result.ready():
                # Update progress if available
                task_info = celery_result.info
                if isinstance(task_info, dict) and 'progress' in task_info:
                    result.progress = task_info['progress']
                
                result.status = TaskStatus.RUNNING
                await asyncio.sleep(1)
            
            # Task completed
            if celery_result.successful():
                result.status = TaskStatus.SUCCESS
                result.result = celery_result.result
            else:
                result.status = TaskStatus.FAILURE
                result.error = str(celery_result.info)
            
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - result.started_at).total_seconds()
            
            # Move to history
            self.task_history[task_id] = result
            self.active_tasks.pop(task_id, None)
            
        except Exception as e:
            logger.error(f"Celery task monitoring failed: {e}")
            result.status = TaskStatus.FAILURE
            result.error = str(e)
    
    async def _execute_async_task(self, task_id: str, task_func: Callable, *args, **kwargs):
        """Execute async task."""
        try:
            result = self.active_tasks[task_id]
            result.status = TaskStatus.RUNNING
            
            start_time = time.time()
            task_result = await task_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            result.status = TaskStatus.SUCCESS
            result.result = task_result
            result.execution_time = execution_time
            result.completed_at = datetime.now()
            result.progress = 1.0
            
        except Exception as e:
            result.status = TaskStatus.FAILURE
            result.error = str(e)
            result.traceback = traceback.format_exc()
            result.completed_at = datetime.now()
            
            log_error(f"Async task execution failed: {task_id}", exception=e)
        
        finally:
            # Move to history
            if task_id in self.active_tasks:
                self.task_history[task_id] = self.active_tasks.pop(task_id)
    
    def _execute_sync_task(self, task_id: str, task_func: Callable, *args, **kwargs) -> Any:
        """Execute synchronous task in thread pool."""
        try:
            result = self.active_tasks[task_id]
            result.status = TaskStatus.RUNNING
            
            start_time = time.time()
            task_result = task_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            result.status = TaskStatus.SUCCESS
            result.result = task_result
            result.execution_time = execution_time
            result.completed_at = datetime.now()
            result.progress = 1.0
            
            return task_result
            
        except Exception as e:
            result.status = TaskStatus.FAILURE
            result.error = str(e)
            result.traceback = traceback.format_exc()
            result.completed_at = datetime.now()
            
            log_error(f"Sync task execution failed: {task_id}", exception=e)
            raise
        
        finally:
            # Move to history
            if task_id in self.active_tasks:
                self.task_history[task_id] = self.active_tasks.pop(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status and result."""
        # Check active tasks first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check history
        if task_id in self.task_history:
            return self.task_history[task_id]
        
        # Check Redis if available
        if self.redis_client:
            try:
                result_data = self.redis_client.get(f"task_result:{task_id}")
                if result_data:
                    data = json.loads(result_data)
                    return TaskResult(**data)
            except Exception as e:
                logger.error(f"Failed to get task status from Redis: {e}")
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            result = self.active_tasks.get(task_id)
            if not result:
                return False
            
            # Cancel Celery task if applicable
            if 'celery_task_id' in result.metadata:
                celery_task_id = result.metadata['celery_task_id']
                if self.celery_app:
                    self.celery_app.control.revoke(celery_task_id, terminate=True)
            
            # Cancel thread future if applicable
            if 'thread_future' in result.metadata:
                future = result.metadata['thread_future']
                future.cancel()
            
            result.status = TaskStatus.REVOKED
            result.completed_at = datetime.now()
            
            # Move to history
            self.task_history[task_id] = self.active_tasks.pop(task_id)
            
            log_info(f"Task cancelled: {task_id}")
            return True
            
        except Exception as e:
            log_error(f"Task cancellation failed: {task_id}", exception=e)
            return False
    
    async def get_active_tasks(self) -> Dict[str, TaskResult]:
        """Get all active tasks."""
        return self.active_tasks.copy()
    
    async def get_task_history(self, limit: int = 100) -> List[TaskResult]:
        """Get task execution history."""
        history = list(self.task_history.values())
        # Sort by completion time, most recent first
        history.sort(key=lambda x: x.completed_at or datetime.min, reverse=True)
        return history[:limit]
    
    def shutdown(self):
        """Shutdown task manager gracefully."""
        logger.info("Shutting down task manager...")
        
        # Signal cleanup thread to stop
        self._shutdown_event.set()
        
        # Wait for cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Task manager shutdown complete")

# Global task manager instance
task_manager = TaskManager()

# Task execution decorators and utilities
def celery_task(*args, **kwargs):
    """Decorator for Celery tasks."""
    def decorator(func):
        if CELERY_AVAILABLE and task_manager.celery_app:
            return task_manager.celery_app.task(*args, **kwargs)(func)
        else:
            # Fallback to regular function
            return func
    return decorator

def background_task(
    priority: TaskPriority = TaskPriority.NORMAL,
    max_retries: int = 3,
    timeout: Optional[int] = None,
    queue: Optional[str] = None
):
    """Decorator for background tasks."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            config = TaskConfig(
                priority=priority,
                max_retries=max_retries,
                timeout=timeout,
                queue=queue
            )
            return await task_manager.submit_task(func, *args, task_config=config, **kwargs)
        return wrapper
    return decorator

# Core task implementations
@celery_task(bind=True, name='backend.tasks.train_model')
def train_model(self, analysis_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train ML models for analysis.
    
    Args:
        analysis_id: Analysis identifier
        config: Training configuration
        
    Returns:
        Training results and metrics
    """
    try:
        from backend.tasks.training_tasks import execute_ml_training
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting training'})
        
        # Execute training
        result = execute_ml_training(analysis_id, config, progress_callback=self.update_state)
        
        # Final update
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Training completed'})
        
        log_info(f"Model training completed for analysis: {analysis_id}")
        return result
        
    except Exception as e:
        log_error(f"Model training failed for analysis: {analysis_id}", exception=e)
        raise

@celery_task(bind=True, name='backend.tasks.process_dataset')
def process_dataset(self, dataset_id: int, user_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process uploaded dataset.
    
    Args:
        dataset_id: Dataset identifier
        user_id: User identifier
        config: Processing configuration
        
    Returns:
        Processing results and statistics
    """
    try:
        from backend.tasks.data_processing_tasks import execute_dataset_processing
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting data processing'})
        
        # Execute processing
        result = execute_dataset_processing(dataset_id, user_id, config, progress_callback=self.update_state)
        
        # Final update
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Processing completed'})
        
        log_info(f"Dataset processing completed: {dataset_id}")
        return result
        
    except Exception as e:
        log_error(f"Dataset processing failed: {dataset_id}", exception=e)
        raise

@celery_task(bind=True, name='backend.tasks.batch_predict')
def batch_predict(self, model_id: str, input_file: str, output_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute batch predictions.
    
    Args:
        model_id: Model identifier
        input_file: Input data file path
        output_file: Output predictions file path
        config: Prediction configuration
        
    Returns:
        Prediction results and statistics
    """
    try:
        from backend.tasks.prediction_tasks import execute_batch_prediction
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting batch prediction'})
        
        # Execute prediction
        result = execute_batch_prediction(model_id, input_file, output_file, config, progress_callback=self.update_state)
        
        # Final update
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Prediction completed'})
        
        log_info(f"Batch prediction completed for model: {model_id}")
        return result
        
    except Exception as e:
        log_error(f"Batch prediction failed for model: {model_id}", exception=e)
        raise

@celery_task(bind=True, name='backend.tasks.detect_drift')
def detect_drift(self, model_id: str, data_window: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect model and data drift.
    
    Args:
        model_id: Model identifier
        data_window: Data window configuration
        
    Returns:
        Drift detection results
    """
    try:
        from backend.tasks.monitoring_tasks import execute_drift_detection
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting drift detection'})
        
        # Execute drift detection
        result = execute_drift_detection(model_id, data_window, progress_callback=self.update_state)
        
        # Final update
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Drift detection completed'})
        
        log_info(f"Drift detection completed for model: {model_id}")
        return result
        
    except Exception as e:
        log_error(f"Drift detection failed for model: {model_id}", exception=e)
        raise

@celery_task(bind=True, name='backend.tasks.cleanup_artifacts')
def cleanup_artifacts(self, retention_days: int = 30) -> Dict[str, Any]:
    """
    Clean up old artifacts and temporary files.
    
    Args:
        retention_days: Number of days to retain artifacts
        
    Returns:
        Cleanup statistics
    """
    try:
        from backend.tasks.cleanup_tasks import execute_artifact_cleanup
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting cleanup'})
        
        # Execute cleanup
        result = execute_artifact_cleanup(retention_days, progress_callback=self.update_state)
        
        # Final update
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Cleanup completed'})
        
        log_info(f"Artifact cleanup completed: {result}")
        return result
        
    except Exception as e:
        log_error(f"Artifact cleanup failed", exception=e)
        raise

@celery_task(bind=True, name='backend.tasks.generate_insights')
def generate_insights(self, analysis_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI insights for analysis.
    
    Args:
        analysis_id: Analysis identifier
        config: Insight generation configuration
        
    Returns:
        Generated insights
    """
    try:
        from backend.tasks.insights_tasks import execute_insight_generation
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Generating insights'})
        
        # Execute insight generation
        result = execute_insight_generation(analysis_id, config, progress_callback=self.update_state)
        
        # Final update
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Insights generated'})
        
        log_info(f"Insight generation completed for analysis: {analysis_id}")
        return result
        
    except Exception as e:
        log_error(f"Insight generation failed for analysis: {analysis_id}", exception=e)
        raise

# Convenience functions for task execution
async def execute_analysis(analysis_id: str, config: Dict[str, Any]) -> str:
    """Execute ML analysis asynchronously."""
    return await task_manager.submit_task(train_model, analysis_id, config)

async def process_uploaded_dataset(dataset_id: int, user_id: int, config: Dict[str, Any] = None) -> str:
    """Process uploaded dataset asynchronously."""
    config = config or {}
    return await task_manager.submit_task(process_dataset, dataset_id, user_id, config)

async def execute_batch_predictions(model_id: str, input_file: str, output_file: str, config: Dict[str, Any] = None) -> str:
    """Execute batch predictions asynchronously."""
    config = config or {}
    return await task_manager.submit_task(batch_predict, model_id, input_file, output_file, config)

async def monitor_model_drift(model_id: str, data_window: Dict[str, Any]) -> str:
    """Monitor model drift asynchronously."""
    return await task_manager.submit_task(detect_drift, model_id, data_window)

async def cleanup_old_artifacts(retention_days: int = 30) -> str:
    """Clean up old artifacts asynchronously."""
    return await task_manager.submit_task(cleanup_artifacts, retention_days)

async def create_insights(analysis_id: str, config: Dict[str, Any] = None) -> str:
    """Generate insights asynchronously."""
    config = config or {}
    return await task_manager.submit_task(generate_insights, analysis_id, config)

# Task monitoring and utilities
async def get_task_progress(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task progress information."""
    result = await task_manager.get_task_status(task_id)
    if result:
        return {
            'task_id': task_id,
            'status': result.status.value,
            'progress': result.progress,
            'started_at': result.started_at.isoformat() if result.started_at else None,
            'completed_at': result.completed_at.isoformat() if result.completed_at else None,
            'execution_time': result.execution_time,
            'error': result.error,
            'metadata': result.metadata
        }
    return None

async def list_active_tasks() -> List[Dict[str, Any]]:
    """List all active tasks."""
    active_tasks = await task_manager.get_active_tasks()
    return [
        {
            'task_id': task_id,
            'status': result.status.value,
            'progress': result.progress,
            'started_at': result.started_at.isoformat() if result.started_at else None,
            'metadata': result.metadata
        }
        for task_id, result in active_tasks.items()
    ]

async def get_task_statistics() -> Dict[str, Any]:
    """Get task execution statistics."""
    active_tasks = await task_manager.get_active_tasks()
    task_history = await task_manager.get_task_history(limit=1000)
    
    # Calculate statistics
    total_tasks = len(active_tasks) + len(task_history)
    active_count = len(active_tasks)
    completed_count = len([t for t in task_history if t.status == TaskStatus.SUCCESS])
    failed_count = len([t for t in task_history if t.status == TaskStatus.FAILURE])
    
    # Average execution time
    completed_tasks = [t for t in task_history if t.execution_time is not None]
    avg_execution_time = sum(t.execution_time for t in completed_tasks) / len(completed_tasks) if completed_tasks else 0
    
    return {
        'total_tasks': total_tasks,
        'active_tasks': active_count,
        'completed_tasks': completed_count,
        'failed_tasks': failed_count,
        'success_rate': completed_count / (completed_count + failed_count) if (completed_count + failed_count) > 0 else 0,
        'average_execution_time': avg_execution_time
    }

# Periodic task scheduling
class TaskScheduler:
    """Periodic task scheduler."""
    
    def __init__(self):
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def schedule_periodic(
        self,
        task_name: str,
        task_func: Callable,
        interval: Union[int, str],
        *args,
        **kwargs
    ):
        """Schedule a periodic task."""
        if isinstance(interval, str):
            # Parse interval string (e.g., "5m", "1h", "1d")
            if interval.endswith('m'):
                interval_seconds = int(interval[:-1]) * 60
            elif interval.endswith('h'):
                interval_seconds = int(interval[:-1]) * 3600
            elif interval.endswith('d'):
                interval_seconds = int(interval[:-1]) * 86400
            else:
                interval_seconds = int(interval)
        else:
            interval_seconds = interval
        
        self.scheduled_tasks[task_name] = {
            'func': task_func,
            'interval': interval_seconds,
            'args': args,
            'kwargs': kwargs,
            'last_run': 0,
            'next_run': time.time() + interval_seconds
        }
        
        # Start scheduler if not running
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            self.start_scheduler()
    
    def start_scheduler(self):
        """Start the task scheduler."""
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_worker,
            name="task-scheduler",
            daemon=True
        )
        self.scheduler_thread.start()
    
    def _scheduler_worker(self):
        """Scheduler worker loop."""
        while not self.stop_event.is_set():
            current_time = time.time()
            
            for task_name, task_info in self.scheduled_tasks.items():
                if current_time >= task_info['next_run']:
                    try:
                        # Execute task
                        task_func = task_info['func']
                        args = task_info['args']
                        kwargs = task_info['kwargs']
                        
                        if asyncio.iscoroutinefunction(task_func):
                            # Submit to task manager for async execution
                            asyncio.create_task(
                                task_manager.submit_task(task_func, *args, **kwargs)
                            )
                        else:
                            # Execute directly
                            task_func(*args, **kwargs)
                        
                        # Update schedule
                        task_info['last_run'] = current_time
                        task_info['next_run'] = current_time + task_info['interval']
                        
                        log_info(f"Executed scheduled task: {task_name}")
                        
                    except Exception as e:
                        log_error(f"Scheduled task failed: {task_name}", exception=e)
            
            # Sleep for 30 seconds
            self.stop_event.wait(30)
    
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.stop_event.set()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)

# Global scheduler instance
task_scheduler = TaskScheduler()

# Cleanup function
def shutdown_tasks():
    """Shutdown all task components."""
    logger.info("Shutting down task system...")
    
    # Stop scheduler
    task_scheduler.stop_scheduler()
    
    # Shutdown task manager
    task_manager.shutdown()
    
    logger.info("Task system shutdown complete")

# Initialize periodic tasks
def initialize_periodic_tasks():
    """Initialize default periodic tasks."""
    try:
        # Schedule cleanup task every 6 hours
        task_scheduler.schedule_periodic(
            'cleanup_artifacts',
            cleanup_artifacts,
            '6h',
            retention_days=30
        )
        
        # Schedule drift detection every hour for active models
        task_scheduler.schedule_periodic(
            'monitor_drift',
            _periodic_drift_monitoring,
            '1h'
        )
        
        logger.info("Periodic tasks initialized")
        
    except Exception as e:
        log_error("Failed to initialize periodic tasks", exception=e)

async def _periodic_drift_monitoring():
    """Periodic drift monitoring for all active models."""
    try:
        # This would query active models and schedule drift detection
        # Implementation would depend on specific requirements
        pass
    except Exception as e:
        log_error("Periodic drift monitoring failed", exception=e)

# Signal handlers for graceful shutdown
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down tasks...")
        shutdown_tasks()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Initialize on import
if BACKEND_AVAILABLE:
    try:
        initialize_periodic_tasks()
        setup_signal_handlers()
    except Exception as e:
        logger.error(f"Task system initialization failed: {e}")

# Export public interface
__all__ = [
    # Core classes
    'TaskManager', 'TaskScheduler', 'TaskResult', 'TaskConfig',
    'TaskStatus', 'TaskPriority', 'TaskType',
    
    # Global instances
    'task_manager', 'task_scheduler',
    
    # Task functions
    'execute_analysis', 'process_uploaded_dataset', 'execute_batch_predictions',
    'monitor_model_drift', 'cleanup_old_artifacts', 'create_insights',
    
    # Monitoring functions
    'get_task_progress', 'list_active_tasks', 'get_task_statistics',
    
    # Decorators
    'celery_task', 'background_task',
    
    # Utilities
    'shutdown_tasks', 'initialize_periodic_tasks'
]
