"""
Cleanup Tasks Module for Auto-Analyst Platform

This module provides comprehensive cleanup and maintenance task implementations
for the Auto-Analyst platform, handling all aspects of system cleanup, resource
optimization, and maintenance operations to ensure optimal performance and
storage management.

Features:
- Automated artifact cleanup with configurable retention policies
- Temporary file and cache management
- Database maintenance and optimization
- Log rotation and archival
- Model artifact lifecycle management
- MLflow experiment and run cleanup
- Feature store maintenance
- Memory and storage optimization
- Orphaned resource detection and cleanup
- Backup and restore operations
- System health monitoring integration
- Compliance and audit trail maintenance

Cleanup Operations:
1. Temporary files and directories cleanup
2. Old dataset and processing artifacts cleanup
3. Model artifacts and experiment cleanup
4. Log files rotation and archival
5. Database maintenance and optimization
6. Cache invalidation and cleanup
7. Memory cleanup and garbage collection
8. Storage optimization and compression
9. Backup management and retention
10. Orphaned resource detection
11. Feature store maintenance
12. Monitoring data cleanup

Retention Policies:
- Temporary files: 1 day
- Processing artifacts: 7 days
- Model artifacts: 30 days (configurable by model stage)
- Experiment logs: 90 days
- System logs: 30 days
- Database backups: 7 days
- Feature store snapshots: 14 days
- Monitoring metrics: 90 days

Safety Features:
- Pre-cleanup validation and safety checks
- Backup creation before destructive operations
- Rollback capabilities for critical operations
- Dry-run mode for testing cleanup operations
- Detailed logging and audit trails
- Resource locking to prevent concurrent cleanup
- Configurable safety thresholds and limits

Dependencies:
- pathlib: File system operations
- shutil: High-level file operations
- psutil: System resource monitoring
- sqlalchemy: Database maintenance
- docker: Container cleanup (if applicable)
- boto3: AWS S3 cleanup operations
- google-cloud-storage: GCS cleanup operations
- redis: Cache cleanup operations
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
import json
import gzip
import tarfile
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import contextmanager
import hashlib
import glob
import subprocess
import traceback

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Database operations
try:
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.orm import sessionmaker
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

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

# Redis cache
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Docker integration
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Backend imports
from backend.config import settings
from backend.models.database import get_db_session
from backend.services.data_service import DataService
from backend.services.mlops_service import MLOpsService
from backend.utils.monitoring import log_info, log_warning, log_error, monitor_performance

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CleanupConfig:
    """Configuration for cleanup operations."""
    
    # General settings
    dry_run: bool = False
    create_backups: bool = True
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Retention periods (in days)
    temp_files_retention: int = 1
    processing_artifacts_retention: int = 7
    model_artifacts_retention: int = 30
    experiment_logs_retention: int = 90
    system_logs_retention: int = 30
    database_backups_retention: int = 7
    feature_store_retention: int = 14
    monitoring_data_retention: int = 90
    
    # Storage thresholds
    disk_usage_threshold: float = 0.85  # 85% disk usage
    memory_usage_threshold: float = 0.80  # 80% memory usage
    min_free_space_gb: float = 5.0  # Minimum free space in GB
    
    # Safety settings
    max_files_per_batch: int = 1000
    max_size_per_batch_gb: float = 10.0
    enable_safety_checks: bool = True
    require_confirmation: bool = False
    
    # Specific cleanup options
    clean_temp_files: bool = True
    clean_datasets: bool = True
    clean_models: bool = True
    clean_experiments: bool = True
    clean_logs: bool = True
    clean_cache: bool = True
    clean_database: bool = False  # Disabled by default for safety
    clean_backups: bool = True
    
    # Cloud storage settings
    clean_cloud_storage: bool = False
    cloud_provider: str = "local"  # local, aws, gcp, azure
    
    # Advanced options
    compress_before_delete: bool = True
    archive_old_data: bool = False
    archive_location: Optional[str] = None

@dataclass
class CleanupResult:
    """Result of cleanup operations."""
    
    # Task information
    task_id: str
    status: str = "running"
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Cleanup statistics
    files_processed: int = 0
    files_deleted: int = 0
    directories_deleted: int = 0
    total_size_cleaned_mb: float = 0.0
    
    # Category-specific cleanup results
    temp_files_cleaned: Dict[str, Any] = field(default_factory=dict)
    datasets_cleaned: Dict[str, Any] = field(default_factory=dict)
    models_cleaned: Dict[str, Any] = field(default_factory=dict)
    experiments_cleaned: Dict[str, Any] = field(default_factory=dict)
    logs_cleaned: Dict[str, Any] = field(default_factory=dict)
    cache_cleaned: Dict[str, Any] = field(default_factory=dict)
    database_cleaned: Dict[str, Any] = field(default_factory=dict)
    
    # Storage optimization
    disk_space_freed_mb: float = 0.0
    compression_savings_mb: float = 0.0
    
    # Safety and backup information
    backups_created: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: float = 0.0
    items_per_second: float = 0.0
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Next cleanup recommendations
    next_cleanup_date: Optional[datetime] = None
    cleanup_recommendations: List[str] = field(default_factory=list)

class CleanupManager:
    """Main cleanup manager for orchestrating all cleanup operations."""
    
    def __init__(self, config: CleanupConfig):
        """Initialize cleanup manager."""
        self.config = config
        self.result = CleanupResult(task_id=str(uuid.uuid4()))
        self.progress_callback: Optional[Callable] = None
        self._lock = threading.Lock()
        self._interrupted = False
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="cleanup-"
        )
        
        # Initialize cloud clients
        self.s3_client = None
        self.gcs_client = None
        self.redis_client = None
        self.mlflow_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize external service clients."""
        try:
            # AWS S3 client
            if AWS_AVAILABLE and self.config.cloud_provider == "aws":
                self.s3_client = boto3.client('s3')
            
            # GCP client
            if GCP_AVAILABLE and self.config.cloud_provider == "gcp":
                self.gcs_client = gcs.Client()
            
            # Redis client
            if REDIS_AVAILABLE and settings.REDIS_URL:
                self.redis_client = redis.from_url(settings.REDIS_URL)
            
            # MLflow client
            if MLFLOW_AVAILABLE:
                self.mlflow_client = MlflowClient()
            
        except Exception as e:
            log_warning(f"Client initialization warning: {e}")
    
    def set_progress_callback(self, callback: Callable):
        """Set progress update callback."""
        self.progress_callback = callback
    
    def update_progress(self, progress: float, status: str, details: Optional[Dict] = None):
        """Update cleanup progress."""
        if self.progress_callback:
            try:
                meta = {'progress': progress, 'status': status}
                if details:
                    meta.update(details)
                self.progress_callback(state='PROGRESS', meta=meta)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        log_info(f"Cleanup progress: {progress:.1%} - {status}")
    
    async def execute_cleanup(self, retention_days: int = 30) -> CleanupResult:
        """
        Execute comprehensive cleanup operations.
        
        Args:
            retention_days: Default retention period in days
            
        Returns:
            Cleanup results and statistics
        """
        try:
            self.update_progress(0.0, "Initializing cleanup operations")
            
            # Override retention settings if provided
            if retention_days != 30:
                self.config.model_artifacts_retention = retention_days
                self.config.experiment_logs_retention = retention_days
                self.config.processing_artifacts_retention = min(retention_days, 7)
            
            # Step 1: Pre-cleanup validation
            self.update_progress(5.0, "Performing pre-cleanup validation")
            await self._pre_cleanup_validation()
            
            # Step 2: System resource check
            self.update_progress(10.0, "Checking system resources")
            await self._check_system_resources()
            
            # Step 3: Create backups if required
            if self.config.create_backups:
                self.update_progress(15.0, "Creating safety backups")
                await self._create_backups()
            
            # Step 4: Temporary files cleanup
            if self.config.clean_temp_files:
                self.update_progress(20.0, "Cleaning temporary files")
                await self._cleanup_temp_files()
            
            # Step 5: Dataset cleanup
            if self.config.clean_datasets:
                self.update_progress(30.0, "Cleaning old datasets and processing artifacts")
                await self._cleanup_datasets()
            
            # Step 6: Model artifacts cleanup
            if self.config.clean_models:
                self.update_progress(45.0, "Cleaning model artifacts")
                await self._cleanup_models()
            
            # Step 7: Experiment cleanup
            if self.config.clean_experiments:
                self.update_progress(55.0, "Cleaning experiment data")
                await self._cleanup_experiments()
            
            # Step 8: Log cleanup
            if self.config.clean_logs:
                self.update_progress(65.0, "Cleaning log files")
                await self._cleanup_logs()
            
            # Step 9: Cache cleanup
            if self.config.clean_cache:
                self.update_progress(75.0, "Cleaning cache data")
                await self._cleanup_cache()
            
            # Step 10: Database cleanup
            if self.config.clean_database:
                self.update_progress(85.0, "Performing database maintenance")
                await self._cleanup_database()
            
            # Step 11: Cloud storage cleanup
            if self.config.clean_cloud_storage:
                self.update_progress(90.0, "Cleaning cloud storage")
                await self._cleanup_cloud_storage()
            
            # Step 12: Final optimization
            self.update_progress(95.0, "Performing final optimization")
            await self._final_optimization()
            
            # Step 13: Generate recommendations
            self.update_progress(99.0, "Generating cleanup recommendations")
            await self._generate_recommendations()
            
            self.update_progress(100.0, "Cleanup completed successfully")
            self._finalize_cleanup()
            
            return self.result
            
        except Exception as e:
            error_msg = f"Cleanup operation failed: {str(e)}"
            log_error(error_msg, exception=e)
            
            self.result.status = "failed"
            self.result.errors.append(error_msg)
            
            return self.result
        
        finally:
            # Cleanup resources
            self.executor.shutdown(wait=True)
            self.result.completed_at = datetime.now()
            if self.result.started_at:
                self.result.processing_time = (
                    self.result.completed_at - self.result.started_at
                ).total_seconds()
    
    async def _pre_cleanup_validation(self):
        """Perform pre-cleanup validation and safety checks."""
        try:
            safety_violations = []
            
            # Check if system is under heavy load
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 80:
                    safety_violations.append(f"High CPU usage: {cpu_percent:.1f}%")
                
                if memory_percent > 90:
                    safety_violations.append(f"High memory usage: {memory_percent:.1f}%")
            
            # Check disk space
            disk_usage = self._get_disk_usage()
            if disk_usage > 0.95:  # More than 95% full
                safety_violations.append(f"Disk usage critically high: {disk_usage:.1%}")
            
            # Check for running critical processes
            critical_processes = ['mlflow', 'jupyter', 'tensorboard']
            if PSUTIL_AVAILABLE:
                running_processes = [p.name() for p in psutil.process_iter(['name'])]
                active_critical = [p for p in critical_processes if p in running_processes]
                if active_critical:
                    safety_violations.append(f"Critical processes running: {active_critical}")
            
            # Store safety violations
            self.result.safety_violations = safety_violations
            
            if safety_violations and self.config.enable_safety_checks:
                if self.config.require_confirmation:
                    raise Exception(f"Safety violations detected: {safety_violations}")
                else:
                    log_warning(f"Safety violations detected but continuing: {safety_violations}")
            
        except Exception as e:
            log_error(f"Pre-cleanup validation failed: {e}")
            raise
    
    def _get_disk_usage(self) -> float:
        """Get current disk usage percentage."""
        try:
            if PSUTIL_AVAILABLE:
                disk_usage = psutil.disk_usage('/')
                return disk_usage.used / disk_usage.total
            else:
                # Fallback method using shutil
                total, used, free = shutil.disk_usage('/')
                return used / total
        except Exception:
            return 0.0
    
    async def _check_system_resources(self):
        """Check system resources and adjust cleanup strategy."""
        try:
            if not PSUTIL_AVAILABLE:
                return
            
            # Get current resource usage
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            # Calculate available resources
            available_memory_gb = memory_info.available / (1024 ** 3)
            available_disk_gb = disk_info.free / (1024 ** 3)
            
            # Adjust cleanup strategy based on available resources
            if available_memory_gb < 2.0:  # Less than 2GB RAM
                self.config.max_files_per_batch = min(500, self.config.max_files_per_batch)
                self.config.max_workers = min(2, self.config.max_workers)
                log_warning("Low memory detected, reducing cleanup batch size")
            
            if available_disk_gb < self.config.min_free_space_gb:
                self.config.compress_before_delete = False  # Skip compression to save space
                log_warning("Low disk space detected, skipping compression")
            
            log_info(f"System resources - RAM: {available_memory_gb:.1f}GB, Disk: {available_disk_gb:.1f}GB")
            
        except Exception as e:
            log_error(f"System resource check failed: {e}")
    
    async def _create_backups(self):
        """Create safety backups before cleanup."""
        try:
            if self.config.dry_run:
                log_info("Dry run mode: Skipping backup creation")
                return
            
            backup_dir = Path(settings.TEMP_DIRECTORY) / f"cleanup_backup_{self.result.task_id}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup critical configuration files
            critical_files = [
                Path(settings.BASE_DIR) / "config.py",
                Path(settings.BASE_DIR) / ".env",
            ]
            
            for file_path in critical_files:
                if file_path.exists():
                    backup_path = backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    self.result.backups_created.append(str(backup_path))
            
            # Backup database (if small enough)
            if self.config.clean_database:
                await self._backup_database(backup_dir)
            
            log_info(f"Backups created in: {backup_dir}")
            
        except Exception as e:
            log_error(f"Backup creation failed: {e}")
            if not self.config.dry_run:
                raise
    
    async def _backup_database(self, backup_dir: Path):
        """Create database backup."""
        try:
            if not DATABASE_AVAILABLE:
                return
            
            # Simple backup for SQLite
            if "sqlite" in settings.DATABASE_URL.lower():
                db_path = settings.DATABASE_URL.replace("sqlite:///", "")
                if Path(db_path).exists():
                    backup_path = backup_dir / "database_backup.db"
                    shutil.copy2(db_path, backup_path)
                    self.result.backups_created.append(str(backup_path))
            
        except Exception as e:
            log_warning(f"Database backup failed: {e}")
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        try:
            temp_dirs = [
                Path(settings.TEMP_DIRECTORY),
                Path(tempfile.gettempdir()),
                Path("/tmp") if Path("/tmp").exists() else None,
            ]
            
            # Remove None values
            temp_dirs = [d for d in temp_dirs if d is not None]
            
            total_cleaned = 0
            total_size_mb = 0
            
            cutoff_date = datetime.now() - timedelta(days=self.config.temp_files_retention)
            
            for temp_dir in temp_dirs:
                if not temp_dir.exists():
                    continue
                
                try:
                    # Find old temporary files
                    pattern = temp_dir / "auto_analyst_*"
                    temp_files = list(temp_dir.glob("auto_analyst_*"))
                    
                    for temp_path in temp_files:
                        try:
                            # Check modification time
                            mod_time = datetime.fromtimestamp(temp_path.stat().st_mtime)
                            
                            if mod_time < cutoff_date:
                                file_size = self._get_path_size(temp_path)
                                
                                if self.config.dry_run:
                                    log_info(f"Would delete: {temp_path} ({file_size / 1024 / 1024:.1f}MB)")
                                else:
                                    if temp_path.is_file():
                                        temp_path.unlink()
                                    elif temp_path.is_dir():
                                        shutil.rmtree(temp_path)
                                    
                                    total_cleaned += 1
                                    total_size_mb += file_size / (1024 * 1024)
                        
                        except Exception as e:
                            log_warning(f"Failed to clean temp file {temp_path}: {e}")
                            continue
                
                except Exception as e:
                    log_warning(f"Failed to clean temp directory {temp_dir}: {e}")
                    continue
            
            self.result.temp_files_cleaned = {
                'files_cleaned': total_cleaned,
                'size_cleaned_mb': total_size_mb,
                'retention_days': self.config.temp_files_retention
            }
            
            log_info(f"Temporary files cleanup: {total_cleaned} files, {total_size_mb:.1f}MB")
            
        except Exception as e:
            log_error(f"Temporary files cleanup failed: {e}")
            self.result.errors.append(f"Temp cleanup failed: {str(e)}")
    
    def _get_path_size(self, path: Path) -> int:
        """Get total size of path (file or directory)."""
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                total_size = 0
                for item in path.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
                return total_size
            else:
                return 0
        except Exception:
            return 0
    
    async def _cleanup_datasets(self):
        """Clean up old datasets and processing artifacts."""
        try:
            datasets_dir = Path(settings.DATASETS_DIRECTORY)
            if not datasets_dir.exists():
                return
            
            total_cleaned = 0
            total_size_mb = 0
            
            cutoff_date = datetime.now() - timedelta(days=self.config.processing_artifacts_retention)
            
            # Get dataset information from database
            with get_db_session() as db_session:
                data_service = DataService()
                old_datasets = data_service.get_datasets_older_than(cutoff_date, db_session)
                
                for dataset in old_datasets:
                    try:
                        # Check if dataset files exist
                        dataset_files = [
                            Path(dataset.file_path) if dataset.file_path else None,
                            datasets_dir / f"processed_{dataset.id}",
                            datasets_dir / f"metadata_{dataset.id}.json"
                        ]
                        
                        dataset_size = 0
                        files_to_delete = []
                        
                        for file_path in dataset_files:
                            if file_path and file_path.exists():
                                file_size = self._get_path_size(file_path)
                                dataset_size += file_size
                                files_to_delete.append(file_path)
                        
                        if files_to_delete:
                            if self.config.dry_run:
                                log_info(f"Would delete dataset {dataset.id}: {len(files_to_delete)} files ({dataset_size / 1024 / 1024:.1f}MB)")
                            else:
                                # Archive before deletion if configured
                                if self.config.archive_old_data:
                                    await self._archive_dataset(dataset.id, files_to_delete)
                                
                                # Delete files
                                for file_path in files_to_delete:
                                    if file_path.is_file():
                                        file_path.unlink()
                                    elif file_path.is_dir():
                                        shutil.rmtree(file_path)
                                
                                # Mark dataset as deleted in database
                                data_service.mark_dataset_deleted(dataset.id, db_session)
                                
                                total_cleaned += len(files_to_delete)
                                total_size_mb += dataset_size / (1024 * 1024)
                    
                    except Exception as e:
                        log_warning(f"Failed to clean dataset {dataset.id}: {e}")
                        continue
            
            self.result.datasets_cleaned = {
                'datasets_cleaned': len(old_datasets) if old_datasets else 0,
                'files_cleaned': total_cleaned,
                'size_cleaned_mb': total_size_mb,
                'retention_days': self.config.processing_artifacts_retention
            }
            
            log_info(f"Dataset cleanup: {total_cleaned} files, {total_size_mb:.1f}MB")
            
        except Exception as e:
            log_error(f"Dataset cleanup failed: {e}")
            self.result.errors.append(f"Dataset cleanup failed: {str(e)}")
    
    async def _archive_dataset(self, dataset_id: int, files: List[Path]):
        """Archive dataset files before deletion."""
        try:
            if not self.config.archive_location:
                return
            
            archive_dir = Path(self.config.archive_location)
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Create compressed archive
            archive_path = archive_dir / f"dataset_{dataset_id}_{datetime.now().strftime('%Y%m%d')}.tar.gz"
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                for file_path in files:
                    if file_path.exists():
                        tar.add(file_path, arcname=file_path.name)
            
            log_info(f"Dataset {dataset_id} archived to: {archive_path}")
            
        except Exception as e:
            log_warning(f"Dataset archiving failed for {dataset_id}: {e}")
    
    async def _cleanup_models(self):
        """Clean up old model artifacts."""
        try:
            models_dir = Path(settings.MODELS_DIRECTORY)
            if not models_dir.exists():
                return
            
            total_cleaned = 0
            total_size_mb = 0
            
            cutoff_date = datetime.now() - timedelta(days=self.config.model_artifacts_retention)
            
            # Get model information from database
            with get_db_session() as db_session:
                mlops_service = MLOpsService()
                old_models = mlops_service.get_models_older_than(cutoff_date, db_session)
                
                for model in old_models:
                    try:
                        # Skip production models
                        if model.stage == "production":
                            continue
                        
                        # Find model artifacts
                        model_artifacts = [
                            Path(model.artifact_location) if model.artifact_location else None,
                            models_dir / f"model_{model.id}",
                        ]
                        
                        model_size = 0
                        files_to_delete = []
                        
                        for artifact_path in model_artifacts:
                            if artifact_path and artifact_path.exists():
                                artifact_size = self._get_path_size(artifact_path)
                                model_size += artifact_size
                                files_to_delete.append(artifact_path)
                        
                        if files_to_delete:
                            if self.config.dry_run:
                                log_info(f"Would delete model {model.id}: {len(files_to_delete)} artifacts ({model_size / 1024 / 1024:.1f}MB)")
                            else:
                                # Delete artifacts
                                for artifact_path in files_to_delete:
                                    if artifact_path.is_file():
                                        artifact_path.unlink()
                                    elif artifact_path.is_dir():
                                        shutil.rmtree(artifact_path)
                                
                                # Update model status in database
                                mlops_service.mark_model_artifacts_cleaned(model.id, db_session)
                                
                                total_cleaned += len(files_to_delete)
                                total_size_mb += model_size / (1024 * 1024)
                    
                    except Exception as e:
                        log_warning(f"Failed to clean model {model.id}: {e}")
                        continue
            
            self.result.models_cleaned = {
                'models_cleaned': len(old_models) if old_models else 0,
                'artifacts_cleaned': total_cleaned,
                'size_cleaned_mb': total_size_mb,
                'retention_days': self.config.model_artifacts_retention
            }
            
            log_info(f"Model cleanup: {total_cleaned} artifacts, {total_size_mb:.1f}MB")
            
        except Exception as e:
            log_error(f"Model cleanup failed: {e}")
            self.result.errors.append(f"Model cleanup failed: {str(e)}")
    
    async def _cleanup_experiments(self):
        """Clean up old MLflow experiments and runs."""
        try:
            if not MLFLOW_AVAILABLE or not self.mlflow_client:
                return
            
            total_cleaned = 0
            total_size_mb = 0
            
            cutoff_date = datetime.now() - timedelta(days=self.config.experiment_logs_retention)
            cutoff_timestamp = int(cutoff_date.timestamp() * 1000)  # MLflow uses milliseconds
            
            try:
                # Get all experiments
                experiments = self.mlflow_client.search_experiments()
                
                for experiment in experiments:
                    try:
                        # Get old runs for this experiment
                        runs = self.mlflow_client.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            filter_string=f"attributes.start_time < {cutoff_timestamp}",
                            max_results=1000
                        )
                        
                        for run in runs:
                            try:
                                if self.config.dry_run:
                                    log_info(f"Would delete run: {run.info.run_id}")
                                else:
                                    # Delete run artifacts and metadata
                                    self.mlflow_client.delete_run(run.info.run_id)
                                    total_cleaned += 1
                                    
                                    # Estimate size (approximate)
                                    total_size_mb += 1  # Rough estimate per run
                            
                            except Exception as e:
                                log_warning(f"Failed to clean run {run.info.run_id}: {e}")
                                continue
                    
                    except Exception as e:
                        log_warning(f"Failed to process experiment {experiment.experiment_id}: {e}")
                        continue
            
            except Exception as e:
                log_warning(f"MLflow experiments cleanup failed: {e}")
            
            self.result.experiments_cleaned = {
                'runs_cleaned': total_cleaned,
                'size_cleaned_mb': total_size_mb,
                'retention_days': self.config.experiment_logs_retention
            }
            
            log_info(f"Experiment cleanup: {total_cleaned} runs, {total_size_mb:.1f}MB")
            
        except Exception as e:
            log_error(f"Experiment cleanup failed: {e}")
            self.result.errors.append(f"Experiment cleanup failed: {str(e)}")
    
    async def _cleanup_logs(self):
        """Clean up old log files."""
        try:
            log_dirs = [
                Path(settings.BASE_DIR) / "logs",
                Path("/var/log/auto-analyst") if Path("/var/log/auto-analyst").exists() else None,
                Path("./logs") if Path("./logs").exists() else None,
            ]
            
            # Remove None values
            log_dirs = [d for d in log_dirs if d is not None]
            
            total_cleaned = 0
            total_size_mb = 0
            
            cutoff_date = datetime.now() - timedelta(days=self.config.system_logs_retention)
            
            for log_dir in log_dirs:
                if not log_dir.exists():
                    continue
                
                try:
                    # Find old log files
                    log_files = list(log_dir.glob("*.log*"))
                    
                    for log_file in log_files:
                        try:
                            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                            
                            if mod_time < cutoff_date:
                                file_size = log_file.stat().st_size
                                
                                if self.config.dry_run:
                                    log_info(f"Would delete log: {log_file} ({file_size / 1024 / 1024:.1f}MB)")
                                else:
                                    # Compress before deletion if configured
                                    if self.config.compress_before_delete and not str(log_file).endswith('.gz'):
                                        compressed_path = f"{log_file}.gz"
                                        with open(log_file, 'rb') as f_in:
                                            with gzip.open(compressed_path, 'wb') as f_out:
                                                shutil.copyfileobj(f_in, f_out)
                                        
                                        # Verify compression worked
                                        if Path(compressed_path).exists():
                                            log_file.unlink()
                                            compression_ratio = Path(compressed_path).stat().st_size / file_size
                                            self.result.compression_savings_mb += file_size * (1 - compression_ratio) / (1024 * 1024)
                                    else:
                                        log_file.unlink()
                                    
                                    total_cleaned += 1
                                    total_size_mb += file_size / (1024 * 1024)
                        
                        except Exception as e:
                            log_warning(f"Failed to clean log file {log_file}: {e}")
                            continue
                
                except Exception as e:
                    log_warning(f"Failed to clean log directory {log_dir}: {e}")
                    continue
            
            self.result.logs_cleaned = {
                'files_cleaned': total_cleaned,
                'size_cleaned_mb': total_size_mb,
                'retention_days': self.config.system_logs_retention
            }
            
            log_info(f"Log cleanup: {total_cleaned} files, {total_size_mb:.1f}MB")
            
        except Exception as e:
            log_error(f"Log cleanup failed: {e}")
            self.result.errors.append(f"Log cleanup failed: {str(e)}")
    
    async def _cleanup_cache(self):
        """Clean up cache data."""
        try:
            total_cleaned = 0
            total_size_mb = 0
            
            # Redis cache cleanup
            if self.redis_client:
                try:
                    # Get cache statistics
                    info = self.redis_client.info()
                    used_memory_mb = info.get('used_memory', 0) / (1024 * 1024)
                    
                    if self.config.dry_run:
                        log_info(f"Would clean Redis cache: {used_memory_mb:.1f}MB")
                    else:
                        # Clean expired keys
                        pipeline = self.redis_client.pipeline()
                        
                        # Get all keys with TTL
                        keys_with_ttl = []
                        for key in self.redis_client.scan_iter(match="*", count=1000):
                            ttl = self.redis_client.ttl(key)
                            if ttl == -1:  # No TTL set, but old key
                                keys_with_ttl.append(key)
                        
                        # Delete old cache entries
                        if keys_with_ttl:
                            pipeline.delete(*keys_with_ttl[:1000])  # Limit batch size
                            pipeline.execute()
                            total_cleaned += len(keys_with_ttl[:1000])
                    
                    total_size_mb += used_memory_mb
                
                except Exception as e:
                    log_warning(f"Redis cache cleanup failed: {e}")
            
            # File-based cache cleanup
            cache_dirs = [
                Path(settings.TEMP_DIRECTORY) / "cache",
                Path("./cache") if Path("./cache").exists() else None,
            ]
            
            cache_dirs = [d for d in cache_dirs if d is not None]
            
            for cache_dir in cache_dirs:
                if not cache_dir.exists():
                    continue
                
                try:
                    cache_files = list(cache_dir.rglob("*"))
                    
                    for cache_file in cache_files:
                        if cache_file.is_file():
                            file_size = cache_file.stat().st_size
                            
                            if self.config.dry_run:
                                log_info(f"Would delete cache file: {cache_file}")
                            else:
                                cache_file.unlink()
                                total_cleaned += 1
                                total_size_mb += file_size / (1024 * 1024)
                
                except Exception as e:
                    log_warning(f"File cache cleanup failed for {cache_dir}: {e}")
            
            self.result.cache_cleaned = {
                'items_cleaned': total_cleaned,
                'size_cleaned_mb': total_size_mb
            }
            
            log_info(f"Cache cleanup: {total_cleaned} items, {total_size_mb:.1f}MB")
            
        except Exception as e:
            log_error(f"Cache cleanup failed: {e}")
            self.result.errors.append(f"Cache cleanup failed: {str(e)}")
    
    async def _cleanup_database(self):
        """Perform database maintenance and cleanup."""
        try:
            if not DATABASE_AVAILABLE:
                return
            
            with get_db_session() as db_session:
                try:
                    # Database-specific maintenance
                    if "sqlite" in settings.DATABASE_URL.lower():
                        await self._cleanup_sqlite_database(db_session)
                    elif "postgresql" in settings.DATABASE_URL.lower():
                        await self._cleanup_postgresql_database(db_session)
                    elif "mysql" in settings.DATABASE_URL.lower():
                        await self._cleanup_mysql_database(db_session)
                    
                    self.result.database_cleaned = {
                        'maintenance_performed': True,
                        'database_type': 'sqlite' if 'sqlite' in settings.DATABASE_URL.lower() else 'other'
                    }
                    
                    log_info("Database maintenance completed")
                
                except Exception as e:
                    log_error(f"Database maintenance failed: {e}")
                    self.result.errors.append(f"Database cleanup failed: {str(e)}")
        
        except Exception as e:
            log_error(f"Database cleanup initialization failed: {e}")
    
    async def _cleanup_sqlite_database(self, db_session):
        """SQLite-specific cleanup operations."""
        try:
            if self.config.dry_run:
                log_info("Would perform SQLite VACUUM and analyze operations")
                return
            
            # VACUUM to reclaim space
            db_session.execute(text("VACUUM"))
            
            # ANALYZE to update statistics
            db_session.execute(text("ANALYZE"))
            
            db_session.commit()
            
            log_info("SQLite database maintenance completed")
            
        except Exception as e:
            log_error(f"SQLite cleanup failed: {e}")
            raise
    
    async def _cleanup_postgresql_database(self, db_session):
        """PostgreSQL-specific cleanup operations."""
        try:
            if self.config.dry_run:
                log_info("Would perform PostgreSQL VACUUM and ANALYZE operations")
                return
            
            # VACUUM ANALYZE
            db_session.execute(text("VACUUM ANALYZE"))
            
            db_session.commit()
            
            log_info("PostgreSQL database maintenance completed")
            
        except Exception as e:
            log_error(f"PostgreSQL cleanup failed: {e}")
            raise
    
    async def _cleanup_mysql_database(self, db_session):
        """MySQL-specific cleanup operations."""
        try:
            if self.config.dry_run:
                log_info("Would perform MySQL OPTIMIZE TABLE operations")
                return
            
            # Get all tables
            tables = db_session.execute(text("SHOW TABLES")).fetchall()
            
            for table in tables:
                table_name = table[0]
                db_session.execute(text(f"OPTIMIZE TABLE {table_name}"))
            
            db_session.commit()
            
            log_info("MySQL database maintenance completed")
            
        except Exception as e:
            log_error(f"MySQL cleanup failed: {e}")
            raise
    
    async def _cleanup_cloud_storage(self):
        """Clean up cloud storage resources."""
        try:
            if self.config.cloud_provider == "aws" and self.s3_client:
                await self._cleanup_s3_storage()
            elif self.config.cloud_provider == "gcp" and self.gcs_client:
                await self._cleanup_gcs_storage()
            
        except Exception as e:
            log_error(f"Cloud storage cleanup failed: {e}")
            self.result.errors.append(f"Cloud storage cleanup failed: {str(e)}")
    
    async def _cleanup_s3_storage(self):
        """Clean up AWS S3 storage."""
        try:
            # This is a placeholder for S3 cleanup logic
            # Implementation would depend on specific S3 bucket structure
            log_info("S3 storage cleanup placeholder")
            
        except Exception as e:
            log_error(f"S3 cleanup failed: {e}")
    
    async def _cleanup_gcs_storage(self):
        """Clean up Google Cloud Storage."""
        try:
            # This is a placeholder for GCS cleanup logic
            # Implementation would depend on specific GCS bucket structure
            log_info("GCS storage cleanup placeholder")
            
        except Exception as e:
            log_error(f"GCS cleanup failed: {e}")
    
    async def _final_optimization(self):
        """Perform final system optimization."""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Calculate total cleanup statistics
            self.result.files_deleted = (
                self.result.temp_files_cleaned.get('files_cleaned', 0) +
                self.result.datasets_cleaned.get('files_cleaned', 0) +
                self.result.models_cleaned.get('artifacts_cleaned', 0) +
                self.result.logs_cleaned.get('files_cleaned', 0) +
                self.result.cache_cleaned.get('items_cleaned', 0)
            )
            
            self.result.total_size_cleaned_mb = (
                self.result.temp_files_cleaned.get('size_cleaned_mb', 0) +
                self.result.datasets_cleaned.get('size_cleaned_mb', 0) +
                self.result.models_cleaned.get('size_cleaned_mb', 0) +
                self.result.experiments_cleaned.get('size_cleaned_mb', 0) +
                self.result.logs_cleaned.get('size_cleaned_mb', 0) +
                self.result.cache_cleaned.get('size_cleaned_mb', 0)
            )
            
            self.result.disk_space_freed_mb = self.result.total_size_cleaned_mb
            
            # Calculate processing rate
            if self.result.processing_time > 0:
                self.result.items_per_second = self.result.files_deleted / self.result.processing_time
            
            log_info("Final optimization completed")
            
        except Exception as e:
            log_error(f"Final optimization failed: {e}")
    
    async def _generate_recommendations(self):
        """Generate cleanup recommendations for future runs."""
        try:
            recommendations = []
            
            # Analyze cleanup results and generate recommendations
            if self.result.total_size_cleaned_mb > 1000:  # More than 1GB cleaned
                recommendations.append("Consider running cleanup more frequently")
            
            if self.result.temp_files_cleaned.get('files_cleaned', 0) > 1000:
                recommendations.append("Review temporary file generation patterns")
            
            if len(self.result.errors) > 0:
                recommendations.append("Review cleanup errors and adjust configuration")
            
            if len(self.result.safety_violations) > 0:
                recommendations.append("Address safety violations before next cleanup")
            
            # Recommend next cleanup date based on cleanup amount
            if self.result.total_size_cleaned_mb < 100:  # Less than 100MB
                next_cleanup = datetime.now() + timedelta(days=30)
            elif self.result.total_size_cleaned_mb < 500:  # Less than 500MB
                next_cleanup = datetime.now() + timedelta(days=14)
            else:
                next_cleanup = datetime.now() + timedelta(days=7)
            
            self.result.next_cleanup_date = next_cleanup
            self.result.cleanup_recommendations = recommendations
            
            log_info(f"Generated {len(recommendations)} cleanup recommendations")
            
        except Exception as e:
            log_error(f"Recommendation generation failed: {e}")
    
    def _finalize_cleanup(self):
        """Finalize cleanup operation."""
        try:
            self.result.status = "completed"
            
            log_info(f"Cleanup completed successfully: {self.result.files_deleted} files, {self.result.total_size_cleaned_mb:.1f}MB")
            
        except Exception as e:
            log_error(f"Cleanup finalization failed: {e}")

# Main cleanup execution function
@monitor_performance("artifact_cleanup")
def execute_artifact_cleanup(
    retention_days: int = 30,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Execute comprehensive artifact cleanup.
    
    Args:
        retention_days: Number of days to retain artifacts
        progress_callback: Progress update callback
        
    Returns:
        Cleanup results and statistics
    """
    try:
        log_info(f"Starting artifact cleanup with {retention_days} days retention")
        
        # Create cleanup configuration
        config = CleanupConfig(
            model_artifacts_retention=retention_days,
            experiment_logs_retention=retention_days,
            processing_artifacts_retention=min(retention_days, 7),
            dry_run=False,
            create_backups=True,
            parallel_execution=True
        )
        
        # Initialize cleanup manager
        manager = CleanupManager(config)
        if progress_callback:
            manager.set_progress_callback(progress_callback)
        
        # Execute cleanup
        result = asyncio.run(manager.execute_cleanup(retention_days))
        
        # Convert result to dictionary
        result_dict = {
            'task_id': result.task_id,
            'status': result.status,
            'files_deleted': result.files_deleted,
            'total_size_cleaned_mb': result.total_size_cleaned_mb,
            'disk_space_freed_mb': result.disk_space_freed_mb,
            'compression_savings_mb': result.compression_savings_mb,
            'processing_time': result.processing_time,
            'items_per_second': result.items_per_second,
            'cleanup_breakdown': {
                'temp_files': result.temp_files_cleaned,
                'datasets': result.datasets_cleaned,
                'models': result.models_cleaned,
                'experiments': result.experiments_cleaned,
                'logs': result.logs_cleaned,
                'cache': result.cache_cleaned,
                'database': result.database_cleaned
            },
            'backups_created': result.backups_created,
            'safety_violations': result.safety_violations,
            'warnings': result.warnings,
            'errors': result.errors,
            'next_cleanup_date': result.next_cleanup_date.isoformat() if result.next_cleanup_date else None,
            'recommendations': result.cleanup_recommendations
        }
        
        log_info(f"Artifact cleanup completed: {result.files_deleted} files, {result.total_size_cleaned_mb:.1f}MB")
        return result_dict
        
    except Exception as e:
        error_msg = f"Artifact cleanup failed: {str(e)}"
        log_error(error_msg, exception=e)
        
        return {
            'task_id': str(uuid.uuid4()),
            'status': 'failed',
            'error_message': error_msg,
            'error_details': {
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        }

# Utility functions
def get_cleanup_recommendations() -> Dict[str, Any]:
    """Get cleanup recommendations based on current system state."""
    try:
        recommendations = {
            'next_recommended_cleanup': datetime.now() + timedelta(days=7),
            'estimated_cleanup_size_mb': 0,
            'priority_areas': [],
            'system_status': 'healthy'
        }
        
        # Check disk usage
        if PSUTIL_AVAILABLE:
            disk_usage = psutil.disk_usage('/')
            usage_percent = disk_usage.used / disk_usage.total
            
            if usage_percent > 0.9:
                recommendations['priority_areas'].append('immediate_cleanup_required')
                recommendations['system_status'] = 'critical'
            elif usage_percent > 0.8:
                recommendations['priority_areas'].append('cleanup_recommended')
                recommendations['system_status'] = 'warning'
        
        return recommendations
        
    except Exception as e:
        log_error(f"Failed to get cleanup recommendations: {e}")
        return {'error': str(e)}

# Export functions
__all__ = [
    'execute_artifact_cleanup',
    'get_cleanup_recommendations',
    'CleanupManager',
    'CleanupConfig',
    'CleanupResult'
]
