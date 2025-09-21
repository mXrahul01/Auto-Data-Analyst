"""
MLOps Service for Auto-Analyst Platform

This service provides comprehensive MLOps capabilities including experiment tracking,
feature store management, model monitoring, and observability for machine learning
workflows in the Auto-Analyst platform.

Features:
- Experiment Tracking: MLflow integration for comprehensive experiment management
- Feature Store: Feast integration for feature engineering and serving
- Model Monitoring: Evidently integration for drift detection and performance monitoring
- Model Registry: Centralized model lifecycle management
- Observability: Comprehensive logging, metrics, and alerting
- Production Support: Scalable, concurrent model execution support
- Integration: Seamless integration with ML pipelines and services

Components:
- ExperimentTracker: MLflow-based experiment tracking and model registry
- FeatureStore: Feast-based feature management and serving
- ModelMonitor: Evidently-based model performance and drift monitoring
- ObservabilityManager: Comprehensive logging and metrics collection
- MLOpsOrchestrator: Central coordinator for all MLOps operations

Usage:
    # Initialize MLOps service
    mlops_service = MLOpsService()
    
    # Track experiment
    experiment_id = await mlops_service.start_experiment("model_training")
    await mlops_service.log_metrics(experiment_id, {"accuracy": 0.95})
    
    # Monitor model
    drift_report = await mlops_service.detect_drift(reference_data, current_data)
    
    # Manage features
    features = await mlops_service.get_features(["feature1", "feature2"])
"""

import asyncio
import logging
import warnings
import uuid
import time
import json
import pickle
import tempfile
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Feast integration
try:
    import feast
    from feast import FeatureStore, Entity, Feature, FeatureView, Field, ValueType
    from feast.data_source import FileSource
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False

# Evidently integration for monitoring
try:
    import evidently
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    from evidently.test_suite import TestSuite
    from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

# Monitoring and observability
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Database integration
try:
    from backend.models.database import get_db_session, MLModel, Analysis, Dataset
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')
warnings.filterwarnings('ignore', category=UserWarning, module='feast')

class ExperimentStatus(str, Enum):
    """Experiment execution status."""
    ACTIVE = "active"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelStage(str, Enum):
    """Model lifecycle stages."""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    NONE = "None"

class MonitoringStatus(str, Enum):
    """Model monitoring status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class DriftType(str, Enum):
    """Types of drift detection."""
    DATA_DRIFT = "data_drift"
    TARGET_DRIFT = "target_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"

@dataclass
class ExperimentConfig:
    """Configuration for MLOps operations."""
    
    # MLflow settings
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_registry_uri: Optional[str] = None
    experiment_name: str = "auto_analyst_experiments"
    
    # Feast settings
    feast_repo_path: str = "./feast_repo"
    feast_project: str = "auto_analyst"
    
    # Monitoring settings
    monitoring_enabled: bool = True
    drift_threshold: float = 0.05
    monitoring_frequency: int = 3600  # seconds
    
    # Storage settings
    artifact_store_path: str = "./mlops_artifacts"
    model_registry_path: str = "./model_registry"
    
    # Performance settings
    max_concurrent_experiments: int = 10
    experiment_timeout: int = 7200  # 2 hours
    retry_attempts: int = 3
    
    # Observability settings
    enable_prometheus: bool = True
    metrics_port: int = 8000
    log_level: str = "INFO"

@dataclass
class ExperimentMetadata:
    """Metadata for ML experiments."""
    
    experiment_id: str
    run_id: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Model information
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_stage: ModelStage = ModelStage.NONE
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Artifacts
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    
    # Associated data
    dataset_id: Optional[int] = None
    analysis_id: Optional[str] = None
    user_id: Optional[int] = None

@dataclass
class DriftReport:
    """Model drift detection report."""
    
    report_id: str
    timestamp: datetime
    drift_type: DriftType
    
    # Drift detection results
    drift_detected: bool
    drift_score: float
    threshold: float
    
    # Detailed results
    feature_drifts: Dict[str, float] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    severity: MonitoringStatus = MonitoringStatus.UNKNOWN
    
    # Metadata
    reference_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))
    current_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))
    model_id: Optional[str] = None

@dataclass
class ModelMonitoringResult:
    """Comprehensive model monitoring result."""
    
    monitoring_id: str
    model_id: str
    timestamp: datetime
    status: MonitoringStatus
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Drift reports
    data_drift: Optional[DriftReport] = None
    target_drift: Optional[DriftReport] = None
    prediction_drift: Optional[DriftReport] = None
    
    # Quality metrics
    data_quality_score: float = 1.0
    prediction_quality_score: float = 1.0
    
    # Alerts and recommendations
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Resource usage
    resource_usage: Dict[str, Any] = field(default_factory=dict)

class ExperimentTracker:
    """MLflow-based experiment tracking and model registry."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = None
        self.active_runs = {}
        
        if MLFLOW_AVAILABLE:
            self._initialize_mlflow()
        else:
            logger.warning("MLflow not available - experiment tracking disabled")
    
    def _initialize_mlflow(self):
        """Initialize MLflow client and tracking."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            if self.config.mlflow_registry_uri:
                mlflow.set_registry_uri(self.config.mlflow_registry_uri)
            
            self.client = MlflowClient()
            
            # Create experiment if it doesn't exist
            try:
                experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(self.config.experiment_name)
            except Exception as e:
                logger.warning(f"Could not create experiment: {str(e)}")
            
            logger.info("MLflow initialized successfully")
            
        except Exception as e:
            logger.error(f"MLflow initialization failed: {str(e)}")
            self.client = None
    
    async def start_experiment(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:
        """
        Start a new MLflow experiment run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of the specific run
            tags: Tags to attach to the run
            nested: Whether this is a nested run
            
        Returns:
            Experiment run ID
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return str(uuid.uuid4())  # Return mock ID
        
        try:
            exp_name = experiment_name or self.config.experiment_name
            
            # Set experiment
            mlflow.set_experiment(exp_name)
            
            # Start run
            run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=tags
            )
            
            run_id = run.info.run_id
            
            # Track active run
            self.active_runs[run_id] = {
                'experiment_name': exp_name,
                'start_time': datetime.now(),
                'status': ExperimentStatus.ACTIVE
            }
            
            logger.info(f"Started experiment run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {str(e)}")
            raise
    
    async def log_parameters(self, run_id: str, parameters: Dict[str, Any]) -> None:
        """Log parameters to MLflow run."""
        if not MLFLOW_AVAILABLE or not self.client:
            return
        
        try:
            with mlflow.start_run(run_id=run_id):
                for key, value in parameters.items():
                    # MLflow has parameter value length limits
                    str_value = str(value)
                    if len(str_value) > 250:
                        str_value = str_value[:247] + "..."
                    
                    mlflow.log_param(key, str_value)
            
            logger.debug(f"Logged {len(parameters)} parameters to run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {str(e)}")
    
    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow run."""
        if not MLFLOW_AVAILABLE or not self.client:
            return
        
        try:
            with mlflow.start_run(run_id=run_id):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                        mlflow.log_metric(key, value, step=step)
                    else:
                        logger.warning(f"Skipping invalid metric {key}: {value}")
            
            logger.debug(f"Logged {len(metrics)} metrics to run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
    
    async def log_artifacts(
        self,
        run_id: str,
        artifacts: Dict[str, Any],
        artifact_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Log artifacts to MLflow run."""
        artifact_paths = {}
        
        if not MLFLOW_AVAILABLE or not self.client:
            return artifact_paths
        
        try:
            with mlflow.start_run(run_id=run_id):
                for artifact_name, artifact_data in artifacts.items():
                    
                    if hasattr(artifact_data, 'save'):  # Model object
                        # Save model
                        mlflow.sklearn.log_model(artifact_data, artifact_name)
                        artifact_paths[artifact_name] = f"runs:/{run_id}/{artifact_name}"
                        
                    elif isinstance(artifact_data, pd.DataFrame):
                        # Save DataFrame as CSV
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                            artifact_data.to_csv(f.name, index=False)
                            mlflow.log_artifact(f.name, artifact_path)
                            artifact_paths[artifact_name] = f"runs:/{run_id}/{artifact_path or ''}/{Path(f.name).name}"
                    
                    elif isinstance(artifact_data, (dict, list)):
                        # Save JSON data
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            json.dump(artifact_data, f, indent=2, default=str)
                            mlflow.log_artifact(f.name, artifact_path)
                            artifact_paths[artifact_name] = f"runs:/{run_id}/{artifact_path or ''}/{Path(f.name).name}"
                    
                    elif isinstance(artifact_data, str) and Path(artifact_data).exists():
                        # Log file
                        mlflow.log_artifact(artifact_data, artifact_path)
                        artifact_paths[artifact_name] = f"runs:/{run_id}/{artifact_path or ''}/{Path(artifact_data).name}"
            
            logger.debug(f"Logged {len(artifacts)} artifacts to run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {str(e)}")
        
        return artifact_paths
    
    async def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model",
        stage: ModelStage = ModelStage.STAGING,
        description: Optional[str] = None
    ) -> str:
        """Register model in MLflow model registry."""
        if not MLFLOW_AVAILABLE or not self.client:
            return "mock_model_version"
        
        try:
            # Register model
            model_uri = f"runs:/{run_id}/{model_path}"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={"run_id": run_id}
            )
            
            model_version = registered_model.version
            
            # Transition to specified stage if not None
            if stage != ModelStage.NONE:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version,
                    stage=stage.value,
                    archive_existing_versions=False
                )
            
            # Update description
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version,
                    description=description
                )
            
            logger.info(f"Registered model {model_name} version {model_version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    async def end_experiment(
        self,
        run_id: str,
        status: ExperimentStatus = ExperimentStatus.FINISHED
    ) -> None:
        """End MLflow experiment run."""
        if not MLFLOW_AVAILABLE or not self.client:
            return
        
        try:
            # Update run status
            if status == ExperimentStatus.FINISHED:
                mlflow_status = "FINISHED"
            elif status == ExperimentStatus.FAILED:
                mlflow_status = "FAILED"
            elif status == ExperimentStatus.CANCELLED:
                mlflow_status = "KILLED"
            else:
                mlflow_status = "FINISHED"
            
            self.client.set_terminated(run_id, mlflow_status)
            
            # Update active runs tracking
            if run_id in self.active_runs:
                self.active_runs[run_id]['status'] = status
                self.active_runs[run_id]['end_time'] = datetime.now()
            
            logger.info(f"Ended experiment run {run_id} with status {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to end experiment: {str(e)}")
    
    async def get_experiment_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get experiment metrics and metadata."""
        if not MLFLOW_AVAILABLE or not self.client:
            return {}
        
        try:
            run = self.client.get_run(run_id)
            
            return {
                'run_id': run_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags,
                'artifact_uri': run.info.artifact_uri
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment metrics: {str(e)}")
            return {}

class FeatureStore:
    """Feast-based feature store management."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.store = None
        
        if FEAST_AVAILABLE:
            self._initialize_feast()
        else:
            logger.warning("Feast not available - feature store disabled")
    
    def _initialize_feast(self):
        """Initialize Feast feature store."""
        try:
            # Create feature store directory if it doesn't exist
            repo_path = Path(self.config.feast_repo_path)
            repo_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize or load feature store
            if not (repo_path / "feature_store.yaml").exists():
                # Initialize new feature store
                import subprocess
                result = subprocess.run(
                    ["feast", "init", self.config.feast_project],
                    cwd=repo_path.parent,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logger.warning(f"Feast init failed: {result.stderr}")
            
            # Load feature store
            self.store = FeatureStore(repo_path=str(repo_path))
            logger.info("Feast feature store initialized")
            
        except Exception as e:
            logger.error(f"Feast initialization failed: {str(e)}")
            self.store = None
    
    async def register_feature_view(
        self,
        name: str,
        features: List[Dict[str, Any]],
        data_source_path: str,
        entity_name: str = "user_id"
    ) -> bool:
        """Register a new feature view."""
        if not FEAST_AVAILABLE or not self.store:
            return False
        
        try:
            # Create data source
            data_source = FileSource(
                path=data_source_path,
                timestamp_field="event_timestamp"
            )
            
            # Create entity
            entity = Entity(
                name=entity_name,
                value_type=ValueType.INT64,
                description=f"Entity for {name}"
            )
            
            # Create features
            feature_list = []
            for feature_def in features:
                feature = Field(
                    name=feature_def['name'],
                    dtype=getattr(ValueType, feature_def.get('type', 'FLOAT')),
                    description=feature_def.get('description', '')
                )
                feature_list.append(feature)
            
            # Create feature view
            feature_view = FeatureView(
                name=name,
                entities=[entity_name],
                schema=feature_list,
                source=data_source,
                ttl=timedelta(days=1)
            )
            
            # Apply to feature store
            self.store.apply([entity, feature_view])
            
            logger.info(f"Registered feature view: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature view: {str(e)}")
            return False
    
    async def get_online_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Get online features for real-time serving."""
        if not FEAST_AVAILABLE or not self.store:
            return pd.DataFrame()
        
        try:
            # Get online features
            response = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows
            )
            
            # Convert to DataFrame
            return response.to_df()
            
        except Exception as e:
            logger.error(f"Failed to get online features: {str(e)}")
            return pd.DataFrame()
    
    async def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str]
    ) -> pd.DataFrame:
        """Get historical features for training."""
        if not FEAST_AVAILABLE or not self.store:
            return pd.DataFrame()
        
        try:
            # Get historical features
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs
            ).to_df()
            
            return training_df
            
        except Exception as e:
            logger.error(f"Failed to get historical features: {str(e)}")
            return pd.DataFrame()
    
    async def materialize_features(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: Optional[List[str]] = None
    ) -> bool:
        """Materialize features to online store."""
        if not FEAST_AVAILABLE or not self.store:
            return False
        
        try:
            if feature_views:
                # Materialize specific feature views
                for fv_name in feature_views:
                    self.store.materialize(
                        feature_views=[fv_name],
                        start_date=start_date,
                        end_date=end_date
                    )
            else:
                # Materialize all feature views
                self.store.materialize_incremental(end_date=end_date)
            
            logger.info("Features materialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to materialize features: {str(e)}")
            return False

class ModelMonitor:
    """Evidently-based model monitoring and drift detection."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.monitoring_results = {}
        
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available - model monitoring disabled")
    
    async def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> DriftReport:
        """Detect data drift between reference and current datasets."""
        drift_threshold = threshold or self.config.drift_threshold
        
        if not EVIDENTLY_AVAILABLE:
            return self._create_mock_drift_report(DriftType.DATA_DRIFT, False)
        
        try:
            # Prepare data
            if feature_columns:
                ref_data = reference_data[feature_columns].copy()
                cur_data = current_data[feature_columns].copy()
            else:
                ref_data = reference_data.copy()
                cur_data = current_data.copy()
            
            # Create drift report
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_data, current_data=cur_data)
            
            # Extract results
            drift_results = report.as_dict()
            dataset_drift = drift_results['metrics'][0]['result']
            
            # Calculate feature-level drift
            feature_drifts = {}
            for feature_result in dataset_drift.get('drift_by_columns', {}):
                feature_name = feature_result.get('column_name')
                drift_score = feature_result.get('drift_score', 0)
                feature_drifts[feature_name] = drift_score
            
            # Determine overall drift
            overall_drift_score = dataset_drift.get('dataset_drift_score', 0)
            drift_detected = overall_drift_score > drift_threshold
            
            # Create drift report
            drift_report = DriftReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                drift_type=DriftType.DATA_DRIFT,
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                threshold=drift_threshold,
                feature_drifts=feature_drifts,
                severity=self._assess_drift_severity(overall_drift_score, drift_threshold)
            )
            
            # Generate recommendations
            drift_report.recommendations = self._generate_drift_recommendations(drift_report)
            
            logger.info(f"Data drift detection completed: {drift_detected} (score: {overall_drift_score:.3f})")
            return drift_report
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {str(e)}")
            return self._create_mock_drift_report(DriftType.DATA_DRIFT, False, error=str(e))
    
    async def detect_target_drift(
        self,
        reference_targets: pd.Series,
        current_targets: pd.Series,
        threshold: Optional[float] = None
    ) -> DriftReport:
        """Detect target drift between reference and current targets."""
        drift_threshold = threshold or self.config.drift_threshold
        
        if not EVIDENTLY_AVAILABLE:
            return self._create_mock_drift_report(DriftType.TARGET_DRIFT, False)
        
        try:
            # Prepare data
            ref_df = pd.DataFrame({'target': reference_targets})
            cur_df = pd.DataFrame({'target': current_targets})
            
            # Create drift report
            report = Report(metrics=[TargetDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            
            # Extract results
            drift_results = report.as_dict()
            target_drift = drift_results['metrics'][0]['result']
            
            drift_score = target_drift.get('drift_score', 0)
            drift_detected = drift_score > drift_threshold
            
            # Create drift report
            drift_report = DriftReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                drift_type=DriftType.TARGET_DRIFT,
                drift_detected=drift_detected,
                drift_score=drift_score,
                threshold=drift_threshold,
                severity=self._assess_drift_severity(drift_score, drift_threshold)
            )
            
            drift_report.recommendations = self._generate_drift_recommendations(drift_report)
            
            logger.info(f"Target drift detection completed: {drift_detected} (score: {drift_score:.3f})")
            return drift_report
            
        except Exception as e:
            logger.error(f"Target drift detection failed: {str(e)}")
            return self._create_mock_drift_report(DriftType.TARGET_DRIFT, False, error=str(e))
    
    async def detect_prediction_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
        threshold: Optional[float] = None
    ) -> DriftReport:
        """Detect prediction drift between reference and current predictions."""
        drift_threshold = threshold or self.config.drift_threshold
        
        try:
            # Simple statistical drift detection for predictions
            from scipy.stats import ks_2samp
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = ks_2samp(reference_predictions, current_predictions)
            
            drift_detected = p_value < drift_threshold
            drift_score = ks_statistic
            
            # Create drift report
            drift_report = DriftReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                drift_type=DriftType.PREDICTION_DRIFT,
                drift_detected=drift_detected,
                drift_score=drift_score,
                threshold=drift_threshold,
                statistical_tests={'ks_test': {'statistic': ks_statistic, 'p_value': p_value}},
                severity=self._assess_drift_severity(drift_score, drift_threshold)
            )
            
            drift_report.recommendations = self._generate_drift_recommendations(drift_report)
            
            logger.info(f"Prediction drift detection completed: {drift_detected} (score: {drift_score:.3f})")
            return drift_report
            
        except Exception as e:
            logger.error(f"Prediction drift detection failed: {str(e)}")
            return self._create_mock_drift_report(DriftType.PREDICTION_DRIFT, False, error=str(e))
    
    async def comprehensive_monitoring(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        reference_targets: Optional[pd.Series] = None,
        current_targets: Optional[pd.Series] = None,
        predictions: Optional[pd.Series] = None
    ) -> ModelMonitoringResult:
        """Comprehensive model monitoring including all drift types."""
        try:
            monitoring_id = str(uuid.uuid4())
            
            # Initialize monitoring result
            monitoring_result = ModelMonitoringResult(
                monitoring_id=monitoring_id,
                model_id=model_id,
                timestamp=datetime.now(),
                status=MonitoringStatus.HEALTHY
            )
            
            # Data drift detection
            data_drift = await self.detect_data_drift(reference_data, current_data)
            monitoring_result.data_drift = data_drift
            
            # Target drift detection
            if reference_targets is not None and current_targets is not None:
                target_drift = await self.detect_target_drift(reference_targets, current_targets)
                monitoring_result.target_drift = target_drift
            
            # Prediction drift detection
            if predictions is not None and reference_targets is not None:
                prediction_drift = await self.detect_prediction_drift(reference_targets, predictions)
                monitoring_result.prediction_drift = prediction_drift
            
            # Data quality assessment
            monitoring_result.data_quality_score = await self._assess_data_quality(current_data)
            
            # Overall status assessment
            monitoring_result.status = self._assess_overall_status([
                data_drift.severity if data_drift else MonitoringStatus.HEALTHY,
                target_drift.severity if monitoring_result.target_drift else MonitoringStatus.HEALTHY,
                prediction_drift.severity if monitoring_result.prediction_drift else MonitoringStatus.HEALTHY
            ])
            
            # Generate alerts and recommendations
            monitoring_result.alerts = self._generate_monitoring_alerts(monitoring_result)
            monitoring_result.recommendations = self._generate_monitoring_recommendations(monitoring_result)
            
            # Store monitoring result
            self.monitoring_results[monitoring_id] = monitoring_result
            
            logger.info(f"Comprehensive monitoring completed for model {model_id}: {monitoring_result.status.value}")
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Comprehensive monitoring failed: {str(e)}")
            raise
    
    def _create_mock_drift_report(
        self, 
        drift_type: DriftType, 
        drift_detected: bool, 
        error: Optional[str] = None
    ) -> DriftReport:
        """Create mock drift report when Evidently is not available."""
        return DriftReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            drift_type=drift_type,
            drift_detected=drift_detected,
            drift_score=0.0,
            threshold=self.config.drift_threshold,
            severity=MonitoringStatus.UNKNOWN,
            recommendations=["Evidently not available - cannot perform drift detection"]
        )
    
    def _assess_drift_severity(self, drift_score: float, threshold: float) -> MonitoringStatus:
        """Assess the severity of detected drift."""
        if drift_score <= threshold:
            return MonitoringStatus.HEALTHY
        elif drift_score <= threshold * 2:
            return MonitoringStatus.WARNING
        else:
            return MonitoringStatus.CRITICAL
    
    def _generate_drift_recommendations(self, drift_report: DriftReport) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        if drift_report.drift_detected:
            if drift_report.severity == MonitoringStatus.CRITICAL:
                recommendations.extend([
                    "Critical drift detected - immediate model retraining recommended",
                    "Review data collection process for changes",
                    "Consider rollback to previous model version"
                ])
            elif drift_report.severity == MonitoringStatus.WARNING:
                recommendations.extend([
                    "Moderate drift detected - schedule model retraining",
                    "Monitor performance metrics closely",
                    "Investigate root cause of drift"
                ])
            
            # Feature-specific recommendations
            if drift_report.feature_drifts:
                high_drift_features = [
                    f for f, score in drift_report.feature_drifts.items() 
                    if score > drift_report.threshold * 1.5
                ]
                if high_drift_features:
                    recommendations.append(f"High drift in features: {', '.join(high_drift_features[:5])}")
        else:
            recommendations.append("No significant drift detected - model performance stable")
        
        return recommendations
    
    async def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality score."""
        try:
            quality_factors = []
            
            # Missing values factor
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            quality_factors.append(1 - missing_ratio)
            
            # Data completeness factor
            completeness = data.count().sum() / (data.shape[0] * data.shape[1])
            quality_factors.append(completeness)
            
            # Duplicate factor
            duplicate_ratio = data.duplicated().sum() / len(data)
            quality_factors.append(1 - duplicate_ratio)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            logger.warning(f"Data quality assessment failed: {str(e)}")
            return 0.5
    
    def _assess_overall_status(self, statuses: List[MonitoringStatus]) -> MonitoringStatus:
        """Assess overall monitoring status from individual statuses."""
        if MonitoringStatus.CRITICAL in statuses:
            return MonitoringStatus.CRITICAL
        elif MonitoringStatus.WARNING in statuses:
            return MonitoringStatus.WARNING
        elif MonitoringStatus.HEALTHY in statuses:
            return MonitoringStatus.HEALTHY
        else:
            return MonitoringStatus.UNKNOWN
    
    def _generate_monitoring_alerts(self, result: ModelMonitoringResult) -> List[str]:
        """Generate monitoring alerts."""
        alerts = []
        
        if result.status == MonitoringStatus.CRITICAL:
            alerts.append("CRITICAL: Model requires immediate attention")
        
        if result.data_drift and result.data_drift.drift_detected:
            alerts.append(f"Data drift detected: {result.data_drift.drift_score:.3f}")
        
        if result.target_drift and result.target_drift.drift_detected:
            alerts.append(f"Target drift detected: {result.target_drift.drift_score:.3f}")
        
        if result.data_quality_score < 0.7:
            alerts.append(f"Low data quality: {result.data_quality_score:.2f}")
        
        return alerts
    
    def _generate_monitoring_recommendations(self, result: ModelMonitoringResult) -> List[str]:
        """Generate comprehensive monitoring recommendations."""
        recommendations = []
        
        # Collect recommendations from individual drift reports
        if result.data_drift:
            recommendations.extend(result.data_drift.recommendations)
        
        if result.target_drift:
            recommendations.extend(result.target_drift.recommendations)
        
        if result.prediction_drift:
            recommendations.extend(result.prediction_drift.recommendations)
        
        # Add general recommendations based on status
        if result.status == MonitoringStatus.CRITICAL:
            recommendations.extend([
                "Schedule immediate model retraining",
                "Investigate data pipeline for issues",
                "Consider feature engineering improvements"
            ])
        elif result.status == MonitoringStatus.WARNING:
            recommendations.extend([
                "Monitor model performance closely",
                "Plan model update in next release cycle"
            ])
        
        # Remove duplicates
        return list(dict.fromkeys(recommendations))

class ObservabilityManager:
    """Comprehensive logging, metrics, and observability management."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics_registry = None
        self.prometheus_metrics = {}
        
        if PROMETHEUS_AVAILABLE and config.enable_prometheus:
            self._initialize_prometheus()
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics."""
        try:
            self.metrics_registry = CollectorRegistry()
            
            # Define metrics
            self.prometheus_metrics = {
                'experiment_counter': Counter(
                    'mlops_experiments_total',
                    'Total number of experiments',
                    ['status'],
                    registry=self.metrics_registry
                ),
                'experiment_duration': Histogram(
                    'mlops_experiment_duration_seconds',
                    'Experiment duration in seconds',
                    registry=self.metrics_registry
                ),
                'model_accuracy': Gauge(
                    'mlops_model_accuracy',
                    'Model accuracy score',
                    ['model_name', 'version'],
                    registry=self.metrics_registry
                ),
                'drift_score': Gauge(
                    'mlops_drift_score',
                    'Model drift score',
                    ['model_id', 'drift_type'],
                    registry=self.metrics_registry
                ),
                'data_quality_score': Gauge(
                    'mlops_data_quality_score',
                    'Data quality score',
                    ['dataset_id'],
                    registry=self.metrics_registry
                )
            }
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Prometheus initialization failed: {str(e)}")
    
    async def record_experiment_start(self, experiment_id: str, experiment_type: str) -> None:
        """Record experiment start metrics."""
        try:
            if self.prometheus_metrics:
                self.prometheus_metrics['experiment_counter'].labels(status='started').inc()
            
            logger.info(f"Experiment started: {experiment_id} ({experiment_type})")
            
        except Exception as e:
            logger.error(f"Failed to record experiment start: {str(e)}")
    
    async def record_experiment_completion(
        self,
        experiment_id: str,
        duration: float,
        status: str,
        metrics: Dict[str, float]
    ) -> None:
        """Record experiment completion metrics."""
        try:
            if self.prometheus_metrics:
                self.prometheus_metrics['experiment_counter'].labels(status=status).inc()
                self.prometheus_metrics['experiment_duration'].observe(duration)
                
                # Record model accuracy if available
                if 'accuracy' in metrics:
                    self.prometheus_metrics['model_accuracy'].labels(
                        model_name=f"exp_{experiment_id}",
                        version="latest"
                    ).set(metrics['accuracy'])
            
            logger.info(f"Experiment completed: {experiment_id} ({status}) in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to record experiment completion: {str(e)}")
    
    async def record_drift_detection(
        self,
        model_id: str,
        drift_type: DriftType,
        drift_score: float
    ) -> None:
        """Record drift detection metrics."""
        try:
            if self.prometheus_metrics:
                self.prometheus_metrics['drift_score'].labels(
                    model_id=model_id,
                    drift_type=drift_type.value
                ).set(drift_score)
            
            logger.info(f"Drift recorded: {model_id} ({drift_type.value}): {drift_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to record drift metrics: {str(e)}")
    
    async def record_data_quality(self, dataset_id: str, quality_score: float) -> None:
        """Record data quality metrics."""
        try:
            if self.prometheus_metrics:
                self.prometheus_metrics['data_quality_score'].labels(
                    dataset_id=dataset_id
                ).set(quality_score)
            
            logger.info(f"Data quality recorded: {dataset_id}: {quality_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to record data quality: {str(e)}")
    
    def get_metrics_handler(self):
        """Get Prometheus metrics handler for HTTP endpoint."""
        if PROMETHEUS_AVAILABLE and self.metrics_registry:
            return prometheus_client.generate_latest(self.metrics_registry)
        return "Prometheus not available"

class MLOpsService:
    """
    Comprehensive MLOps service orchestrating experiment tracking,
    feature store, model monitoring, and observability.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize MLOps service with all components."""
        self.config = config or ExperimentConfig()
        
        # Initialize components
        self.experiment_tracker = ExperimentTracker(self.config)
        self.feature_store = FeatureStore(self.config)
        self.model_monitor = ModelMonitor(self.config)
        self.observability = ObservabilityManager(self.config)
        
        # Service state
        self.active_experiments = {}
        self.monitoring_jobs = {}
        self.service_stats = {
            'experiments_tracked': 0,
            'models_monitored': 0,
            'drift_detections': 0,
            'features_served': 0
        }
        
        # Threading for background tasks
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_experiments)
        self.monitoring_active = True
        
        logger.info("MLOps service initialized successfully")
    
    # Experiment Management
    
    async def start_ml_experiment(
        self,
        analysis_id: str,
        experiment_config: Dict[str, Any],
        user_id: Optional[int] = None
    ) -> str:
        """Start comprehensive ML experiment with full MLOps tracking."""
        try:
            # Generate experiment metadata
            experiment_metadata = ExperimentMetadata(
                experiment_id=str(uuid.uuid4()),
                run_id="",  # Will be set by MLflow
                experiment_name=f"analysis_{analysis_id}",
                status=ExperimentStatus.ACTIVE,
                start_time=datetime.now(),
                analysis_id=analysis_id,
                user_id=user_id
            )
            
            # Start MLflow experiment
            run_id = await self.experiment_tracker.start_experiment(
                experiment_name=experiment_metadata.experiment_name,
                run_name=f"run_{experiment_metadata.experiment_id}",
                tags={
                    'analysis_id': analysis_id,
                    'user_id': str(user_id) if user_id else 'unknown',
                    'experiment_type': 'auto_analysis'
                }
            )
            
            experiment_metadata.run_id = run_id
            
            # Log initial parameters
            await self.experiment_tracker.log_parameters(run_id, experiment_config)
            
            # Record observability metrics
            await self.observability.record_experiment_start(
                experiment_metadata.experiment_id, 
                'auto_analysis'
            )
            
            # Store experiment metadata
            self.active_experiments[experiment_metadata.experiment_id] = experiment_metadata
            
            logger.info(f"ML experiment started: {experiment_metadata.experiment_id}")
            return experiment_metadata.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to start ML experiment: {str(e)}")
            raise
    
    async def log_experiment_progress(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None
    ) -> None:
        """Log experiment progress and metrics."""
        try:
            if experiment_id not in self.active_experiments:
                logger.warning(f"Experiment not found: {experiment_id}")
                return
            
            metadata = self.active_experiments[experiment_id]
            
            # Log to MLflow
            await self.experiment_tracker.log_metrics(metadata.run_id, metrics, step)
            
            if parameters:
                await self.experiment_tracker.log_parameters(metadata.run_id, parameters)
            
            # Update metadata
            metadata.metrics.update(metrics)
            if parameters:
                metadata.parameters.update(parameters)
            
            logger.debug(f"Logged progress for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to log experiment progress: {str(e)}")
    
    async def complete_ml_experiment(
        self,
        experiment_id: str,
        final_metrics: Dict[str, float],
        model_artifacts: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        register_model: bool = True
    ) -> Dict[str, Any]:
        """Complete ML experiment with model registration and final logging."""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment not found: {experiment_id}")
            
            metadata = self.active_experiments[experiment_id]
            
            # Log final metrics
            await self.experiment_tracker.log_metrics(metadata.run_id, final_metrics)
            metadata.metrics.update(final_metrics)
            
            # Log artifacts if provided
            artifact_paths = {}
            if model_artifacts:
                artifact_paths = await self.experiment_tracker.log_artifacts(
                    metadata.run_id, model_artifacts
                )
                metadata.artifact_paths.update(artifact_paths)
            
            # Register model if requested
            model_version = None
            if register_model and model_name:
                model_version = await self.experiment_tracker.register_model(
                    metadata.run_id,
                    model_name,
                    description=f"Model from analysis {metadata.analysis_id}"
                )
                metadata.model_name = model_name
                metadata.model_version = model_version
            
            # End experiment
            metadata.status = ExperimentStatus.FINISHED
            metadata.end_time = datetime.now()
            
            await self.experiment_tracker.end_experiment(
                metadata.run_id, ExperimentStatus.FINISHED
            )
            
            # Record observability metrics
            duration = (metadata.end_time - metadata.start_time).total_seconds()
            await self.observability.record_experiment_completion(
                experiment_id, duration, 'finished', final_metrics
            )
            
            # Update service stats
            self.service_stats['experiments_tracked'] += 1
            
            # Move to completed experiments
            del self.active_experiments[experiment_id]
            
            completion_result = {
                'experiment_id': experiment_id,
                'status': 'completed',
                'duration': duration,
                'model_name': model_name,
                'model_version': model_version,
                'artifact_paths': artifact_paths,
                'final_metrics': final_metrics
            }
            
            logger.info(f"ML experiment completed: {experiment_id}")
            return completion_result
            
        except Exception as e:
            logger.error(f"Failed to complete ML experiment: {str(e)}")
            raise
    
    # Model Monitoring
    
    async def start_model_monitoring(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        monitoring_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start continuous model monitoring."""
        try:
            monitoring_job_id = str(uuid.uuid4())
            
            # Store reference data and config
            monitoring_job = {
                'job_id': monitoring_job_id,
                'model_id': model_id,
                'reference_data': reference_data,
                'config': monitoring_config or {},
                'start_time': datetime.now(),
                'status': 'active',
                'last_check': datetime.now()
            }
            
            self.monitoring_jobs[monitoring_job_id] = monitoring_job
            
            # Start background monitoring
            self.executor.submit(self._background_monitoring_loop, monitoring_job_id)
            
            logger.info(f"Model monitoring started: {model_id} (job: {monitoring_job_id})")
            return monitoring_job_id
            
        except Exception as e:
            logger.error(f"Failed to start model monitoring: {str(e)}")
            raise
    
    async def check_model_health(
        self,
        model_id: str,
        current_data: pd.DataFrame,
        current_targets: Optional[pd.Series] = None,
        predictions: Optional[pd.Series] = None
    ) -> ModelMonitoringResult:
        """Perform comprehensive model health check."""
        try:
            # Find monitoring job for model
            monitoring_job = None
            for job in self.monitoring_jobs.values():
                if job['model_id'] == model_id:
                    monitoring_job = job
                    break
            
            if not monitoring_job:
                raise ValueError(f"No monitoring job found for model: {model_id}")
            
            reference_data = monitoring_job['reference_data']
            
            # Perform comprehensive monitoring
            monitoring_result = await self.model_monitor.comprehensive_monitoring(
                model_id=model_id,
                reference_data=reference_data,
                current_data=current_data,
                current_targets=current_targets,
                predictions=predictions
            )
            
            # Record drift metrics
            if monitoring_result.data_drift:
                await self.observability.record_drift_detection(
                    model_id, DriftType.DATA_DRIFT, monitoring_result.data_drift.drift_score
                )
            
            if monitoring_result.target_drift:
                await self.observability.record_drift_detection(
                    model_id, DriftType.TARGET_DRIFT, monitoring_result.target_drift.drift_score
                )
            
            # Record data quality
            await self.observability.record_data_quality(
                f"model_{model_id}", monitoring_result.data_quality_score
            )
            
            # Update service stats
            self.service_stats['drift_detections'] += 1
            
            logger.info(f"Model health check completed: {model_id} - {monitoring_result.status.value}")
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Model health check failed: {str(e)}")
            raise
    
    def _background_monitoring_loop(self, monitoring_job_id: str) -> None:
        """Background monitoring loop for continuous model monitoring."""
        try:
            job = self.monitoring_jobs[monitoring_job_id]
            
            while (self.monitoring_active and 
                   job['status'] == 'active' and 
                   monitoring_job_id in self.monitoring_jobs):
                
                # Sleep for monitoring frequency
                time.sleep(self.config.monitoring_frequency)
                
                # Check if job still exists and is active
                if (monitoring_job_id not in self.monitoring_jobs or 
                    self.monitoring_jobs[monitoring_job_id]['status'] != 'active'):
                    break
                
                # Update last check time
                job['last_check'] = datetime.now()
                
                logger.debug(f"Background monitoring check: {job['model_id']}")
        
        except Exception as e:
            logger.error(f"Background monitoring loop failed: {str(e)}")
            if monitoring_job_id in self.monitoring_jobs:
                self.monitoring_jobs[monitoring_job_id]['status'] = 'failed'
    
    async def stop_model_monitoring(self, monitoring_job_id: str) -> bool:
        """Stop model monitoring job."""
        try:
            if monitoring_job_id in self.monitoring_jobs:
                self.monitoring_jobs[monitoring_job_id]['status'] = 'stopped'
                del self.monitoring_jobs[monitoring_job_id]
                logger.info(f"Model monitoring stopped: {monitoring_job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop model monitoring: {str(e)}")
            return False
    
    # Feature Store Operations
    
    async def register_features(
        self,
        feature_set_name: str,
        features_data: pd.DataFrame,
        feature_definitions: List[Dict[str, Any]]
    ) -> bool:
        """Register features in the feature store."""
        try:
            # Save features data to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
                features_data.to_parquet(f.name)
                data_path = f.name
            
            # Register feature view
            success = await self.feature_store.register_feature_view(
                name=feature_set_name,
                features=feature_definitions,
                data_source_path=data_path
            )
            
            if success:
                logger.info(f"Features registered: {feature_set_name}")
                self.service_stats['features_served'] += len(feature_definitions)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to register features: {str(e)}")
            return False
    
    async def get_features_for_inference(
        self,
        feature_names: List[str],
        entity_ids: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Get features for real-time inference."""
        try:
            features_df = await self.feature_store.get_online_features(
                feature_refs=feature_names,
                entity_rows=entity_ids
            )
            
            logger.debug(f"Retrieved {len(features_df)} feature rows for inference")
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to get features for inference: {str(e)}")
            return pd.DataFrame()
    
    async def get_features_for_training(
        self,
        entity_df: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Get historical features for model training."""
        try:
            training_features = await self.feature_store.get_historical_features(
                entity_df=entity_df,
                feature_refs=feature_names
            )
            
            logger.debug(f"Retrieved training features: {training_features.shape}")
            return training_features
            
        except Exception as e:
            logger.error(f"Failed to get training features: {str(e)}")
            return pd.DataFrame()
    
    # Service Management
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive MLOps service health status."""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'mlflow': MLFLOW_AVAILABLE and self.experiment_tracker.client is not None,
                    'feast': FEAST_AVAILABLE and self.feature_store.store is not None,
                    'evidently': EVIDENTLY_AVAILABLE,
                    'prometheus': PROMETHEUS_AVAILABLE and self.config.enable_prometheus
                },
                'active_experiments': len(self.active_experiments),
                'monitoring_jobs': len(self.monitoring_jobs),
                'service_stats': self.service_stats.copy()
            }
            
            # Check component health
            unhealthy_components = [
                name for name, status in health_status['components'].items() 
                if not status
            ]
            
            if unhealthy_components:
                health_status['status'] = 'degraded'
                health_status['issues'] = f"Components unavailable: {', '.join(unhealthy_components)}"
            
            # Add recent activity
            health_status['recent_activity'] = {
                'last_experiment': max([
                    exp.start_time for exp in self.active_experiments.values()
                ], default=datetime.min).isoformat() if self.active_experiments else None,
                
                'active_monitoring_jobs': [
                    {
                        'job_id': job['job_id'],
                        'model_id': job['model_id'],
                        'status': job['status'],
                        'last_check': job['last_check'].isoformat()
                    }
                    for job in self.monitoring_jobs.values()
                ]
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup_resources(self, older_than_days: int = 30) -> Dict[str, int]:
        """Cleanup old experiments and monitoring data."""
        try:
            cleanup_stats = {
                'experiments_cleaned': 0,
                'monitoring_jobs_cleaned': 0,
                'artifacts_cleaned': 0
            }
            
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            # Cleanup old monitoring jobs
            jobs_to_remove = []
            for job_id, job in self.monitoring_jobs.items():
                if job['start_time'] < cutoff_date and job['status'] != 'active':
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.monitoring_jobs[job_id]
                cleanup_stats['monitoring_jobs_cleaned'] += 1
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return {'error': str(e)}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown MLOps service."""
        try:
            logger.info("Shutting down MLOps service")
            
            # Stop monitoring
            self.monitoring_active = False
            
            # Complete active experiments
            for experiment_id in list(self.active_experiments.keys()):
                try:
                    metadata = self.active_experiments[experiment_id]
                    await self.experiment_tracker.end_experiment(
                        metadata.run_id, ExperimentStatus.CANCELLED
                    )
                except Exception as e:
                    logger.warning(f"Failed to cleanup experiment {experiment_id}: {str(e)}")
            
            # Stop monitoring jobs
            for job_id in list(self.monitoring_jobs.keys()):
                self.monitoring_jobs[job_id]['status'] = 'stopped'
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("MLOps service shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {str(e)}")

# Factory functions for easy service creation

def create_mlops_service(config: Optional[ExperimentConfig] = None) -> MLOpsService:
    """
    Factory function to create MLOps service instance.
    
    Args:
        config: Optional MLOps configuration
        
    Returns:
        Configured MLOpsService instance
    """
    return MLOpsService(config)

def get_mlops_service() -> MLOpsService:
    """Get MLOps service instance for dependency injection."""
    return create_mlops_service()

# Example usage and testing
if __name__ == "__main__":
    async def example_usage():
        """Example usage of the MLOps service."""
        
        print("🔧 MLOpsService Example Usage")
        print("=" * 50)
        
        # Initialize service
        mlops_service = create_mlops_service()
        
        # Check service health
        print("\n🏥 Service Health Check:")
        health = await mlops_service.get_service_health()
        print(f"Status: {health['status']}")
        print(f"Components: {health['components']}")
        
        try:
            # Example 1: Start ML experiment
            print("\n🧪 Starting ML Experiment...")
            
            experiment_config = {
                'model_type': 'RandomForestClassifier',
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
            
            experiment_id = await mlops_service.start_ml_experiment(
                analysis_id="test_analysis_123",
                experiment_config=experiment_config,
                user_id=1
            )
            
            print(f"✅ Experiment started: {experiment_id}")
            
            # Log progress
            print("📊 Logging experiment progress...")
            await mlops_service.log_experiment_progress(
                experiment_id,
                metrics={'accuracy': 0.85, 'precision': 0.83},
                step=1
            )
            
            await mlops_service.log_experiment_progress(
                experiment_id,
                metrics={'accuracy': 0.87, 'precision': 0.85},
                step=2
            )
            
            # Complete experiment
            print("🏁 Completing experiment...")
            final_metrics = {'accuracy': 0.89, 'precision': 0.87, 'recall': 0.86}
            
            completion_result = await mlops_service.complete_ml_experiment(
                experiment_id,
                final_metrics=final_metrics,
                model_name="test_model",
                register_model=True
            )
            
            print(f"✅ Experiment completed: {completion_result['status']}")
            print(f"   Duration: {completion_result['duration']:.1f}s")
            print(f"   Model version: {completion_result['model_version']}")
            
        except Exception as e:
            print(f"❌ Experiment example failed: {str(e)}")
        
        try:
            # Example 2: Model monitoring
            print("\n🔍 Model Monitoring Example...")
            
            # Generate sample data
            np.random.seed(42)
            reference_data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 1000),
                'feature2': np.random.normal(0, 1, 1000),
                'feature3': np.random.uniform(0, 1, 1000)
            })
            
            # Current data with some drift
            current_data = pd.DataFrame({
                'feature1': np.random.normal(0.5, 1.2, 500),  # Mean shift and variance change
                'feature2': np.random.normal(0, 1, 500),
                'feature3': np.random.uniform(0, 1, 500)
            })
            
            # Start monitoring
            monitoring_job_id = await mlops_service.start_model_monitoring(
                model_id="test_model_123",
                reference_data=reference_data
            )
            
            print(f"📈 Monitoring started: {monitoring_job_id}")
            
            # Perform health check
            monitoring_result = await mlops_service.check_model_health(
                model_id="test_model_123",
                current_data=current_data
            )
            
            print(f"🏥 Model health: {monitoring_result.status.value}")
            if monitoring_result.data_drift:
                print(f"   Data drift: {monitoring_result.data_drift.drift_detected} "
                      f"(score: {monitoring_result.data_drift.drift_score:.3f})")
            print(f"   Data quality: {monitoring_result.data_quality_score:.2f}")
            print(f"   Alerts: {len(monitoring_result.alerts)}")
            
            # Stop monitoring
            await mlops_service.stop_model_monitoring(monitoring_job_id)
            print("✅ Monitoring stopped")
            
        except Exception as e:
            print(f"❌ Monitoring example failed: {str(e)}")
        
        # Example 3: Service statistics
        print("\n📊 Service Statistics:")
        final_health = await mlops_service.get_service_health()
        stats = final_health['service_stats']
        print(f"   Experiments tracked: {stats['experiments_tracked']}")
        print(f"   Models monitored: {stats['models_monitored']}")
        print(f"   Drift detections: {stats['drift_detections']}")
        print(f"   Features served: {stats['features_served']}")
        
        # Cleanup
        print("\n🧹 Cleaning up resources...")
        cleanup_result = await mlops_service.cleanup_resources(older_than_days=0)
        print(f"   Cleaned: {cleanup_result}")
        
        # Shutdown
        await mlops_service.shutdown()
        
        print(f"\n🎯 MLOpsService example completed!")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except Exception as e:
        print(f"Example execution failed: {str(e)}")
