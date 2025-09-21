"""
ML Training Tasks Module for Auto-Analyst Platform

This module provides comprehensive ML training task implementations for the
Auto-Analyst platform, handling all aspects of machine learning model training
including automated model selection, hyperparameter optimization, cross-validation,
ensemble creation, and performance evaluation.

Features:
- Automated ML pipeline execution with model selection
- Multi-algorithm training with performance comparison
- Hyperparameter optimization using Optuna/Hyperopt
- Advanced cross-validation strategies
- Ensemble model creation and optimization
- Real-time progress tracking and monitoring
- MLflow experiment tracking and artifact management
- Feature engineering and selection automation
- Model interpretability and explanation generation
- Resource usage monitoring and optimization
- Remote training coordination (Kaggle, Colab, Cloud)
- Distributed training support for large datasets

Training Pipeline:
1. Data validation and preprocessing
2. Feature engineering and selection
3. Model architecture selection
4. Hyperparameter optimization
5. Cross-validation and evaluation
6. Ensemble creation and optimization
7. Model interpretation and explanation
8. Performance benchmarking and comparison
9. Model registration and deployment preparation
10. Artifact management and cleanup

Supported Algorithms:
- Tabular: XGBoost, LightGBM, CatBoost, Random Forest, SVM, Neural Networks
- Time Series: ARIMA, Prophet, LSTM, Transformer models
- Deep Learning: TensorFlow, PyTorch models for various tasks
- Ensemble: Voting, Bagging, Boosting, Stacking ensembles
- Anomaly Detection: Isolation Forest, One-Class SVM, Autoencoders
- Clustering: K-Means, DBSCAN, Hierarchical clustering
- NLP: BERT, GPT, classical ML for text classification

Dependencies:
- MLflow: Experiment tracking and model registry
- Optuna: Hyperparameter optimization
- scikit-learn: Classical ML algorithms
- XGBoost, LightGBM, CatBoost: Gradient boosting
- TensorFlow/Keras: Deep learning models
- SHAP, LIME: Model interpretability
- Feast: Feature store integration
- Evidently: Model monitoring and validation
"""

import asyncio
import logging
import os
import time
import uuid
import json
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Backend imports
from backend.config import settings
from backend.models.database import get_db_session
from backend.services.data_service import DataService
from backend.services.mlops_service import MLOpsService
from backend.utils.monitoring import log_info, log_warning, log_error, monitor_performance
from backend.utils.preprocessing import preprocess_data
from backend.utils.validation import validate_dataset
from backend.ml.auto_pipeline import AutoMLPipeline
from backend.ml.tabular import TabularModels
from backend.ml.ensemble import EnsembleModels
from backend.ml.evaluation import ModelEvaluator
from backend.ml.explainer import ModelExplainer

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for ML training tasks."""
    
    # General settings
    task_type: str = "classification"  # classification, regression, clustering
    target_column: str = ""
    feature_columns: List[str] = field(default_factory=list)
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    
    # Model selection
    algorithms: List[str] = field(default_factory=lambda: ["auto"])
    max_models: int = 10
    time_budget_hours: float = 2.0
    
    # Hyperparameter optimization
    enable_hpo: bool = True
    hpo_trials: int = 100
    hpo_timeout: int = 3600  # seconds
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, timeseries
    
    # Ensemble settings
    enable_ensemble: bool = True
    ensemble_size: int = 5
    ensemble_methods: List[str] = field(default_factory=lambda: ["voting", "stacking"])
    
    # Feature engineering
    enable_feature_engineering: bool = True
    max_features: Optional[int] = None
    feature_selection_method: str = "auto"
    
    # Model interpretability
    enable_explanations: bool = True
    explanation_samples: int = 100
    
    # Performance settings
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None
    
    # Remote execution
    execution_mode: str = "local"  # local, kaggle, colab, aws, gcp
    compute_backend: Optional[str] = None
    
    # Monitoring
    enable_monitoring: bool = True
    drift_detection: bool = True
    
    # Advanced settings
    early_stopping: bool = True
    class_weight: Optional[str] = None
    custom_metrics: List[str] = field(default_factory=list)

@dataclass
class TrainingResult:
    """Result of ML training task."""
    
    # Execution info
    task_id: str
    analysis_id: str
    status: str = "running"
    progress: float = 0.0
    
    # Training results
    models_trained: List[Dict[str, Any]] = field(default_factory=list)
    best_model: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    
    # Performance metrics
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Model comparison
    model_comparison: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Feature information
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_selection_results: Dict[str, Any] = field(default_factory=dict)
    
    # Explanations
    model_explanations: Dict[str, Any] = field(default_factory=dict)
    prediction_explanations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Artifacts
    model_artifacts: Dict[str, str] = field(default_factory=dict)
    experiment_artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Execution statistics
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class MLTrainingExecutor:
    """Core ML training execution engine."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize training executor."""
        self.config = config
        self.result = TrainingResult(
            task_id=str(uuid.uuid4()),
            analysis_id=config.target_column or str(uuid.uuid4()),
            started_at=datetime.now()
        )
        self.progress_callback: Optional[Callable] = None
        self.mlflow_client: Optional[MlflowClient] = None
        self.experiment_id: Optional[str] = None
        self.current_run_id: Optional[str] = None
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            self._initialize_mlflow()
    
    def _initialize_mlflow(self):
        """Initialize MLflow experiment tracking."""
        try:
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            self.mlflow_client = MlflowClient()
            
            # Create or get experiment
            experiment_name = f"auto-analyst-{self.config.task_type}-{datetime.now().strftime('%Y%m%d')}"
            
            try:
                experiment = self.mlflow_client.create_experiment(
                    name=experiment_name,
                    artifact_location=settings.mlflow.artifact_location
                )
                self.experiment_id = experiment
            except mlflow.exceptions.MlflowException:
                # Experiment already exists
                experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
                self.experiment_id = experiment.experiment_id
            
            logger.info(f"MLflow experiment initialized: {experiment_name}")
            
        except Exception as e:
            log_warning(f"MLflow initialization failed: {e}")
            self.mlflow_client = None
    
    def set_progress_callback(self, callback: Callable):
        """Set progress update callback."""
        self.progress_callback = callback
    
    def update_progress(self, progress: float, status: str, details: Optional[Dict] = None):
        """Update training progress."""
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
        
        log_info(f"Training progress: {progress:.1%} - {status}")
    
    async def execute_training(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> TrainingResult:
        """
        Execute complete ML training pipeline.
        
        Args:
            data: Training dataset
            target_column: Target variable column name
            feature_columns: Feature column names (optional)
            
        Returns:
            Training results and artifacts
        """
        try:
            self.update_progress(0.0, "Initializing training pipeline")
            
            # Start MLflow run
            if self.mlflow_client:
                run = self.mlflow_client.create_run(
                    experiment_id=self.experiment_id,
                    tags={"task_type": self.config.task_type}
                )
                self.current_run_id = run.info.run_id
                mlflow.start_run(run_id=self.current_run_id)
            
            # Step 1: Data validation and preprocessing
            self.update_progress(5.0, "Validating and preprocessing data")
            X, y, feature_names = await self._prepare_data(data, target_column, feature_columns)
            
            # Step 2: Data splitting
            self.update_progress(10.0, "Splitting data for training and validation")
            X_train, X_test, y_train, y_test = await self._split_data(X, y)
            
            # Step 3: Feature engineering and selection
            self.update_progress(15.0, "Performing feature engineering")
            X_train_engineered, X_test_engineered, feature_info = await self._engineer_features(
                X_train, X_test, y_train, feature_names
            )
            
            # Step 4: Model selection and training
            self.update_progress(20.0, "Starting model training")
            models_results = await self._train_models(
                X_train_engineered, X_test_engineered, y_train, y_test
            )
            
            # Step 5: Hyperparameter optimization
            if self.config.enable_hpo:
                self.update_progress(60.0, "Optimizing hyperparameters")
                models_results = await self._optimize_hyperparameters(
                    models_results, X_train_engineered, y_train
                )
            
            # Step 6: Ensemble creation
            if self.config.enable_ensemble:
                self.update_progress(75.0, "Creating ensemble models")
                ensemble_results = await self._create_ensembles(
                    models_results, X_train_engineered, X_test_engineered, y_train, y_test
                )
                models_results.extend(ensemble_results)
            
            # Step 7: Model evaluation and comparison
            self.update_progress(85.0, "Evaluating and comparing models")
            best_model = await self._evaluate_models(models_results, X_test_engineered, y_test)
            
            # Step 8: Model interpretation and explanations
            if self.config.enable_explanations:
                self.update_progress(90.0, "Generating model explanations")
                explanations = await self._generate_explanations(
                    best_model, X_train_engineered, X_test_engineered, feature_info['feature_names']
                )
                self.result.model_explanations = explanations
            
            # Step 9: Save artifacts and finalize
            self.update_progress(95.0, "Saving models and artifacts")
            await self._save_artifacts(best_model, models_results, feature_info)
            
            # Step 10: Complete training
            self.update_progress(100.0, "Training completed successfully")
            self._finalize_training(best_model, models_results)
            
            return self.result
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            log_error(error_msg, exception=e)
            
            self.result.status = "failed"
            self.result.error_message = error_msg
            self.result.error_details = {
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            
            return self.result
        
        finally:
            # Clean up MLflow run
            if self.mlflow_client and self.current_run_id:
                try:
                    mlflow.end_run()
                except Exception as e:
                    logger.warning(f"MLflow run cleanup failed: {e}")
            
            self.result.completed_at = datetime.now()
            if self.result.started_at:
                self.result.execution_time = (
                    self.result.completed_at - self.result.started_at
                ).total_seconds()
    
    async def _prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for training."""
        try:
            # Validate target column
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Extract target variable
            y = data[target_column].copy()
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            # Validate feature columns
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                raise ValueError(f"Feature columns not found: {missing_features}")
            
            # Extract features
            X = data[feature_columns].copy()
            
            # Basic data validation
            if X.empty or y.empty:
                raise ValueError("Empty dataset provided")
            
            if len(X) != len(y):
                raise ValueError("Feature and target lengths don't match")
            
            # Handle missing values in target
            if y.isnull().any():
                log_warning("Target variable contains missing values, dropping affected rows")
                valid_indices = y.notnull()
                X = X[valid_indices]
                y = y[valid_indices]
            
            # Log data information
            log_info(f"Prepared data: {len(X)} samples, {len(X.columns)} features")
            
            # Log to MLflow
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_params({
                    'n_samples': len(X),
                    'n_features': len(X.columns),
                    'target_column': target_column,
                    'task_type': self.config.task_type
                })
            
            return X, y, feature_columns
            
        except Exception as e:
            log_error(f"Data preparation failed: {e}")
            raise
    
    async def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data for training and testing."""
        try:
            # Determine stratification
            stratify = None
            if (self.config.stratify and 
                self.config.task_type == "classification" and
                len(y.unique()) > 1):
                stratify = y
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify
            )
            
            log_info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Log to MLflow
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_params({
                    'test_size': self.config.test_size,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                })
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            log_error(f"Data splitting failed: {e}")
            raise
    
    async def _engineer_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Perform feature engineering and selection."""
        try:
            if not self.config.enable_feature_engineering:
                return X_train, X_test, {'feature_names': feature_names}
            
            # Initialize feature engineering pipeline
            from backend.ml.feature_engineering import FeatureEngineer
            
            feature_engineer = FeatureEngineer(
                task_type=self.config.task_type,
                max_features=self.config.max_features,
                selection_method=self.config.feature_selection_method
            )
            
            # Fit and transform features
            X_train_engineered = feature_engineer.fit_transform(X_train, y_train)
            X_test_engineered = feature_engineer.transform(X_test)
            
            # Get feature information
            feature_info = {
                'original_features': feature_names,
                'engineered_features': feature_engineer.get_feature_names(),
                'feature_importance': feature_engineer.get_feature_importance(),
                'selection_results': feature_engineer.get_selection_results(),
                'transformation_steps': feature_engineer.get_transformation_steps()
            }
            
            log_info(f"Feature engineering completed: {len(feature_names)} -> {len(feature_info['engineered_features'])} features")
            
            return X_train_engineered, X_test_engineered, feature_info
            
        except Exception as e:
            log_warning(f"Feature engineering failed, using original features: {e}")
            return X_train, X_test, {'feature_names': feature_names}
    
    async def _train_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> List[Dict[str, Any]]:
        """Train multiple ML models."""
        try:
            models_results = []
            
            # Initialize model trainer
            if self.config.task_type == "classification":
                from backend.ml.tabular import TabularClassifiers
                model_trainer = TabularClassifiers()
            elif self.config.task_type == "regression":
                from backend.ml.tabular import TabularRegressors
                model_trainer = TabularRegressors()
            else:
                raise ValueError(f"Unsupported task type: {self.config.task_type}")
            
            # Get algorithms to train
            algorithms = self.config.algorithms
            if "auto" in algorithms:
                algorithms = model_trainer.get_recommended_algorithms(
                    X_train.shape, self.config.task_type
                )
            
            # Train each algorithm
            total_algorithms = len(algorithms)
            for i, algorithm in enumerate(algorithms):
                try:
                    progress = 20 + (40 * (i + 1) / total_algorithms)
                    self.update_progress(progress, f"Training {algorithm} model")
                    
                    # Train model
                    model_result = await self._train_single_model(
                        algorithm, X_train, X_test, y_train, y_test, model_trainer
                    )
                    
                    if model_result:
                        models_results.append(model_result)
                        
                        # Log to MLflow
                        if MLFLOW_AVAILABLE and mlflow.active_run():
                            with mlflow.start_run(nested=True):
                                mlflow.log_params(model_result['parameters'])
                                mlflow.log_metrics(model_result['metrics'])
                                
                                # Log model
                                if algorithm in ['xgboost', 'lightgbm']:
                                    if algorithm == 'xgboost' and XGBOOST_AVAILABLE:
                                        mlflow.xgboost.log_model(model_result['model'], "model")
                                    elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
                                        mlflow.lightgbm.log_model(model_result['model'], "model")
                                else:
                                    mlflow.sklearn.log_model(model_result['model'], "model")
                    
                except Exception as e:
                    log_warning(f"Training failed for {algorithm}: {e}")
                    continue
            
            if not models_results:
                raise ValueError("No models were trained successfully")
            
            log_info(f"Successfully trained {len(models_results)} models")
            return models_results
            
        except Exception as e:
            log_error(f"Model training failed: {e}")
            raise
    
    async def _train_single_model(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_trainer
    ) -> Optional[Dict[str, Any]]:
        """Train a single ML model."""
        try:
            start_time = time.time()
            
            # Get model and parameters
            model, params = model_trainer.get_model_config(algorithm, self.config.task_type)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train, self.config.task_type)
            test_metrics = self._calculate_metrics(y_test, y_pred_test, self.config.task_type)
            
            # Cross-validation
            cv_scores = None
            if self.config.cv_folds > 1:
                cv_scores = await self._cross_validate_model(model, X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Prepare result
            result = {
                'algorithm': algorithm,
                'model': model,
                'parameters': params,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'training_time': training_time,
                'model_size': self._get_model_size(model),
                'feature_importance': self._get_feature_importance(model, X_train.columns)
            }
            
            return result
            
        except Exception as e:
            log_error(f"Single model training failed for {algorithm}: {e}")
            return None
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            metrics = {}
            
            if task_type == "classification":
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # ROC AUC for binary classification
                if len(np.unique(y_true)) == 2:
                    try:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                    except Exception:
                        pass
                
            elif task_type == "regression":
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y_true, y_pred)
            
            return metrics
            
        except Exception as e:
            log_error(f"Metrics calculation failed: {e}")
            return {}
    
    async def _cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation."""
        try:
            # Choose CV strategy
            if self.config.cv_strategy == "stratified" and self.config.task_type == "classification":
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            elif self.config.cv_strategy == "timeseries":
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            
            # Perform cross-validation
            scoring = 'accuracy' if self.config.task_type == "classification" else 'r2'
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)
            
            return {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            log_error(f"Cross-validation failed: {e}")
            return {}
    
    def _get_model_size(self, model) -> Optional[float]:
        """Get model size in MB."""
        try:
            import sys
            return sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
        except Exception:
            return None
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model."""
        try:
            importance = None
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            
            if importance is not None and len(importance) == len(feature_names):
                return dict(zip(feature_names, importance.tolist()))
            
            return {}
            
        except Exception as e:
            log_error(f"Feature importance extraction failed: {e}")
            return {}
    
    async def _optimize_hyperparameters(
        self,
        models_results: List[Dict[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> List[Dict[str, Any]]:
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            log_warning("Optuna not available, skipping hyperparameter optimization")
            return models_results
        
        try:
            optimized_results = []
            
            # Select top models for optimization
            top_models = sorted(models_results, key=lambda x: list(x['test_metrics'].values())[0], reverse=True)[:3]
            
            for i, model_result in enumerate(top_models):
                try:
                    progress = 60 + (15 * (i + 1) / len(top_models))
                    self.update_progress(progress, f"Optimizing {model_result['algorithm']} hyperparameters")
                    
                    # Create Optuna study
                    study = optuna.create_study(
                        direction='maximize',
                        sampler=TPESampler(seed=self.config.random_state),
                        pruner=MedianPruner()
                    )
                    
                    # Define objective function
                    def objective(trial):
                        return self._optuna_objective(
                            trial, model_result['algorithm'], X_train, y_train
                        )
                    
                    # Optimize
                    study.optimize(
                        objective,
                        n_trials=min(self.config.hpo_trials, 50),
                        timeout=self.config.hpo_timeout // len(top_models),
                        n_jobs=1  # Avoid nested parallelism
                    )
                    
                    # Train final model with best parameters
                    best_params = study.best_params
                    optimized_model = self._train_with_params(
                        model_result['algorithm'], X_train, y_train, best_params
                    )
                    
                    # Update model result
                    model_result['model'] = optimized_model
                    model_result['parameters'].update(best_params)
                    model_result['hpo_score'] = study.best_value
                    model_result['hpo_trials'] = len(study.trials)
                    
                    optimized_results.append(model_result)
                    
                except Exception as e:
                    log_warning(f"HPO failed for {model_result['algorithm']}: {e}")
                    optimized_results.append(model_result)
            
            # Add non-optimized models
            remaining_models = models_results[len(top_models):]
            optimized_results.extend(remaining_models)
            
            return optimized_results
            
        except Exception as e:
            log_error(f"Hyperparameter optimization failed: {e}")
            return models_results
    
    def _optuna_objective(self, trial, algorithm: str, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """Optuna objective function for hyperparameter optimization."""
        try:
            # Define hyperparameter search spaces
            if algorithm == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif algorithm == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif algorithm == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            else:
                # Default parameter space
                params = {}
            
            # Train model with trial parameters
            model = self._train_with_params(algorithm, X_train, y_train, params)
            
            # Cross-validation score
            scoring = 'accuracy' if self.config.task_type == "classification" else 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring, n_jobs=1)
            
            return cv_scores.mean()
            
        except Exception as e:
            log_error(f"Optuna objective failed: {e}")
            return 0.0
    
    def _train_with_params(self, algorithm: str, X_train: pd.DataFrame, y_train: pd.Series, params: Dict) -> Any:
        """Train model with specific parameters."""
        try:
            from backend.ml.tabular import TabularModels
            model_factory = TabularModels()
            
            model = model_factory.create_model(algorithm, self.config.task_type, params)
            model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            log_error(f"Model training with params failed: {e}")
            raise
    
    async def _create_ensembles(
        self,
        models_results: List[Dict[str, Any]],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> List[Dict[str, Any]]:
        """Create ensemble models."""
        try:
            if len(models_results) < 2:
                log_warning("Not enough models for ensemble creation")
                return []
            
            ensemble_results = []
            
            # Select top models for ensemble
            top_models = sorted(
                models_results, 
                key=lambda x: list(x['test_metrics'].values())[0], 
                reverse=True
            )[:self.config.ensemble_size]
            
            from backend.ml.ensemble import EnsembleModels
            ensemble_factory = EnsembleModels()
            
            for method in self.config.ensemble_methods:
                try:
                    self.update_progress(75 + (10 * len(ensemble_results) / len(self.config.ensemble_methods)), 
                                       f"Creating {method} ensemble")
                    
                    # Extract models and prepare ensemble
                    base_models = [(f"model_{i}", result['model']) for i, result in enumerate(top_models)]
                    
                    # Create ensemble
                    ensemble_model = ensemble_factory.create_ensemble(
                        method, base_models, self.config.task_type
                    )
                    
                    # Train ensemble
                    ensemble_model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_train = ensemble_model.predict(X_train)
                    y_pred_test = ensemble_model.predict(X_test)
                    
                    # Calculate metrics
                    train_metrics = self._calculate_metrics(y_train, y_pred_train, self.config.task_type)
                    test_metrics = self._calculate_metrics(y_test, y_pred_test, self.config.task_type)
                    
                    # Prepare ensemble result
                    ensemble_result = {
                        'algorithm': f"ensemble_{method}",
                        'model': ensemble_model,
                        'parameters': {'method': method, 'n_models': len(base_models)},
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'base_models': [result['algorithm'] for result in top_models],
                        'ensemble_type': method,
                        'training_time': sum(result['training_time'] for result in top_models)
                    }
                    
                    ensemble_results.append(ensemble_result)
                    
                except Exception as e:
                    log_warning(f"Ensemble creation failed for {method}: {e}")
                    continue
            
            log_info(f"Created {len(ensemble_results)} ensemble models")
            return ensemble_results
            
        except Exception as e:
            log_error(f"Ensemble creation failed: {e}")
            return []
    
    async def _evaluate_models(
        self,
        models_results: List[Dict[str, Any]],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate and compare all trained models."""
        try:
            if not models_results:
                raise ValueError("No models to evaluate")
            
            # Sort models by performance
            if self.config.task_type == "classification":
                metric_key = 'accuracy'
            else:
                metric_key = 'r2'
            
            sorted_models = sorted(
                models_results,
                key=lambda x: x['test_metrics'].get(metric_key, 0),
                reverse=True
            )
            
            best_model = sorted_models[0]
            
            # Store results
            self.result.models_trained = models_results
            self.result.best_model = best_model
            self.result.best_score = best_model['test_metrics'].get(metric_key, 0)
            
            # Create model comparison
            comparison_data = []
            for result in models_results:
                comparison_data.append({
                    'algorithm': result['algorithm'],
                    'train_score': list(result['train_metrics'].values())[0],
                    'test_score': list(result['test_metrics'].values())[0],
                    'training_time': result.get('training_time', 0),
                    'model_size': result.get('model_size', 0)
                })
            
            self.result.model_comparison = {
                'comparison_data': comparison_data,
                'best_model': best_model['algorithm'],
                'total_models': len(models_results)
            }
            
            # Performance summary
            self.result.performance_summary = {
                'best_' + metric_key: self.result.best_score,
                'models_trained': len(models_results),
                'total_training_time': sum(r.get('training_time', 0) for r in models_results)
            }
            
            log_info(f"Best model: {best_model['algorithm']} with {metric_key}: {self.result.best_score:.4f}")
            
            return best_model
            
        except Exception as e:
            log_error(f"Model evaluation failed: {e}")
            raise
    
    async def _generate_explanations(
        self,
        best_model: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate model explanations and interpretability results."""
        try:
            explanations = {}
            model = best_model['model']
            
            # Global feature importance
            feature_importance = self._get_feature_importance(model, feature_names)
            explanations['feature_importance'] = feature_importance
            
            # SHAP explanations
            if SHAP_AVAILABLE:
                try:
                    explanations['shap'] = await self._generate_shap_explanations(
                        model, X_train, X_test, feature_names
                    )
                except Exception as e:
                    log_warning(f"SHAP explanation failed: {e}")
            
            # LIME explanations
            if LIME_AVAILABLE:
                try:
                    explanations['lime'] = await self._generate_lime_explanations(
                        model, X_train, X_test, feature_names
                    )
                except Exception as e:
                    log_warning(f"LIME explanation failed: {e}")
            
            return explanations
            
        except Exception as e:
            log_error(f"Explanation generation failed: {e}")
            return {}
    
    async def _generate_shap_explanations(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        try:
            # Sample data for explanation (performance optimization)
            sample_size = min(len(X_test), self.config.explanation_samples)
            X_explain = X_test.sample(n=sample_size, random_state=self.config.random_state)
            
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):  # Classification
                explainer = shap.Explainer(model.predict_proba, X_train.sample(n=min(100, len(X_train))))
                shap_values = explainer(X_explain)
            else:  # Regression
                explainer = shap.Explainer(model.predict, X_train.sample(n=min(100, len(X_train))))
                shap_values = explainer(X_explain)
            
            # Extract SHAP information
            shap_data = {
                'feature_importance': dict(zip(feature_names, np.abs(shap_values.values).mean(axis=0))),
                'sample_explanations': shap_values.values[:10].tolist(),  # First 10 samples
                'base_value': float(shap_values.base_values.mean()) if hasattr(shap_values, 'base_values') else 0.0
            }
            
            return shap_data
            
        except Exception as e:
            log_error(f"SHAP explanation failed: {e}")
            return {}
    
    async def _generate_lime_explanations(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate LIME explanations."""
        try:
            # Create LIME explainer
            if self.config.task_type == "classification":
                explainer = LimeTabularExplainer(
                    X_train.values,
                    feature_names=feature_names,
                    class_names=np.unique(model.predict(X_train)).astype(str),
                    mode='classification'
                )
            else:
                explainer = LimeTabularExplainer(
                    X_train.values,
                    feature_names=feature_names,
                    mode='regression'
                )
            
            # Generate explanations for sample instances
            sample_size = min(10, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=self.config.random_state)
            
            explanations = []
            for idx, instance in X_sample.iterrows():
                exp = explainer.explain_instance(
                    instance.values,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=len(feature_names)
                )
                
                # Extract explanation data
                exp_data = {
                    'instance_id': int(idx),
                    'prediction': float(model.predict([instance.values])[0]),
                    'feature_contributions': dict(exp.as_list())
                }
                
                explanations.append(exp_data)
            
            return {'sample_explanations': explanations}
            
        except Exception as e:
            log_error(f"LIME explanation failed: {e}")
            return {}
    
    async def _save_artifacts(
        self,
        best_model: Dict[str, Any],
        models_results: List[Dict[str, Any]],
        feature_info: Dict[str, Any]
    ):
        """Save model artifacts and experiment data."""
        try:
            artifacts = {}
            
            # Create artifacts directory
            artifacts_dir = Path(settings.ARTIFACTS_DIRECTORY) / f"analysis_{self.result.analysis_id}"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best model
            model_path = artifacts_dir / "best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model['model'], f)
            artifacts['best_model'] = str(model_path)
            
            # Save all models summary
            models_summary = [
                {
                    'algorithm': result['algorithm'],
                    'parameters': result['parameters'],
                    'train_metrics': result['train_metrics'],
                    'test_metrics': result['test_metrics'],
                    'training_time': result.get('training_time', 0)
                }
                for result in models_results
            ]
            
            models_path = artifacts_dir / "models_summary.json"
            with open(models_path, 'w') as f:
                json.dump(models_summary, f, indent=2, default=str)
            artifacts['models_summary'] = str(models_path)
            
            # Save feature information
            feature_path = artifacts_dir / "feature_info.json"
            with open(feature_path, 'w') as f:
                json.dump(feature_info, f, indent=2, default=str)
            artifacts['feature_info'] = str(feature_path)
            
            # Save training configuration
            config_path = artifacts_dir / "training_config.json"
            with open(config_path, 'w') as f:
                config_dict = {
                    'task_type': self.config.task_type,
                    'algorithms': self.config.algorithms,
                    'test_size': self.config.test_size,
                    'cv_folds': self.config.cv_folds,
                    'enable_hpo': self.config.enable_hpo,
                    'enable_ensemble': self.config.enable_ensemble
                }
                json.dump(config_dict, f, indent=2)
            artifacts['training_config'] = str(config_path)
            
            self.result.model_artifacts = artifacts
            
            # Log artifacts to MLflow
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_artifacts(str(artifacts_dir))
            
            log_info(f"Artifacts saved to: {artifacts_dir}")
            
        except Exception as e:
            log_error(f"Artifact saving failed: {e}")
    
    def _finalize_training(self, best_model: Dict[str, Any], models_results: List[Dict[str, Any]]):
        """Finalize training results."""
        try:
            self.result.status = "completed"
            
            # Set feature importance from best model
            self.result.feature_importance = best_model.get('feature_importance', {})
            
            # Set performance metrics
            self.result.training_metrics = best_model.get('train_metrics', {})
            self.result.test_metrics = best_model.get('test_metrics', {})
            
            # Log final results to MLflow
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_metrics(self.result.test_metrics)
                mlflow.log_param('best_algorithm', best_model['algorithm'])
                mlflow.log_param('total_models_trained', len(models_results))
                
                # Log feature importance
                if self.result.feature_importance:
                    for feature, importance in list(self.result.feature_importance.items())[:20]:
                        mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            log_info("Training finalized successfully")
            
        except Exception as e:
            log_error(f"Training finalization failed: {e}")

# Main training execution function
@monitor_performance("ml_training")
def execute_ml_training(
    analysis_id: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Execute ML training task.
    
    Args:
        analysis_id: Analysis identifier
        config: Training configuration
        progress_callback: Progress update callback
        
    Returns:
        Training results
    """
    try:
        log_info(f"Starting ML training for analysis: {analysis_id}")
        
        # Parse configuration
        training_config = TrainingConfig(
            task_type=config.get('task_type', 'classification'),
            target_column=config.get('target_column', ''),
            feature_columns=config.get('feature_columns'),
            algorithms=config.get('algorithms', ['auto']),
            max_models=config.get('max_models', 10),
            time_budget_hours=config.get('time_budget_hours', 2.0),
            enable_hpo=config.get('enable_hpo', True),
            enable_ensemble=config.get('enable_ensemble', True),
            cv_folds=config.get('cv_folds', 5)
        )
        
        # Initialize training executor
        executor = MLTrainingExecutor(training_config)
        if progress_callback:
            executor.set_progress_callback(progress_callback)
        
        # Load data
        with get_db_session() as db_session:
            data_service = DataService()
            dataset = data_service.get_dataset_by_analysis_id(analysis_id, db_session)
            
            if not dataset:
                raise ValueError(f"Dataset not found for analysis: {analysis_id}")
            
            # Load dataset
            data = pd.read_parquet(dataset.file_path)  # Assuming parquet format
        
        # Execute training
        result = asyncio.run(executor.execute_training(
            data, training_config.target_column, training_config.feature_columns
        ))
        
        # Convert result to dictionary
        result_dict = {
            'task_id': result.task_id,
            'analysis_id': result.analysis_id,
            'status': result.status,
            'progress': result.progress,
            'best_model_algorithm': result.best_model['algorithm'] if result.best_model else None,
            'best_score': result.best_score,
            'models_trained': len(result.models_trained),
            'training_metrics': result.training_metrics,
            'test_metrics': result.test_metrics,
            'model_comparison': result.model_comparison,
            'feature_importance': result.feature_importance,
            'model_explanations': result.model_explanations,
            'execution_time': result.execution_time,
            'artifacts': result.model_artifacts,
            'error_message': result.error_message
        }
        
        # Update database with results
        with get_db_session() as db_session:
            mlops_service = MLOpsService()
            mlops_service.save_training_results(analysis_id, result_dict, db_session)
        
        log_info(f"ML training completed for analysis: {analysis_id}")
        return result_dict
        
    except Exception as e:
        error_msg = f"ML training failed for analysis {analysis_id}: {str(e)}"
        log_error(error_msg, exception=e)
        
        return {
            'task_id': str(uuid.uuid4()),
            'analysis_id': analysis_id,
            'status': 'failed',
            'error_message': error_msg,
            'error_details': {
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        }

# Remote training coordination functions
async def execute_remote_training(
    analysis_id: str,
    config: Dict[str, Any],
    platform: str = "kaggle"
) -> Dict[str, Any]:
    """Execute training on remote platform."""
    try:
        if platform == "kaggle":
            from backend.tasks.remote_execution import KaggleExecutor
            executor = KaggleExecutor()
        elif platform == "colab":
            from backend.tasks.remote_execution import ColabExecutor
            executor = ColabExecutor()
        else:
            raise ValueError(f"Unsupported remote platform: {platform}")
        
        # Submit training job to remote platform
        job_id = await executor.submit_training_job(analysis_id, config)
        
        # Monitor job execution
        result = await executor.monitor_job(job_id)
        
        return result
        
    except Exception as e:
        log_error(f"Remote training failed: {e}")
        raise

# Export functions
__all__ = [
    'execute_ml_training',
    'execute_remote_training', 
    'MLTrainingExecutor',
    'TrainingConfig',
    'TrainingResult'
]
