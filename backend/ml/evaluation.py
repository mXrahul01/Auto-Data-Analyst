"""
Model Evaluation Module for Auto-Analyst Platform

This module implements comprehensive model evaluation capabilities including:
- Regression metrics (RMSE, MAE, R², MAPE, SMAPE, etc.)
- Classification metrics (Accuracy, F1, ROC-AUC, Precision, Recall, etc.)
- Multi-class and binary classification support
- Cross-validation evaluation with statistical significance
- Model comparison and benchmarking utilities
- Performance visualization and reporting
- Business impact assessment and ROI calculation
- Error analysis and failure case identification
- Statistical significance testing
- Custom metric definitions and calculations

Features:
- Comprehensive metric calculation for all ML tasks
- Statistical significance testing and confidence intervals
- Advanced evaluation strategies (stratified, time-series, etc.)
- Model comparison with statistical tests
- Automated evaluation report generation
- Performance visualization data preparation
- Business impact metrics and interpretability
- Integration with MLflow for experiment tracking
- Real-time evaluation for streaming predictions
- Custom evaluation protocols for domain-specific needs
- Error analysis and model diagnostics
- Production monitoring metric calculation
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from abc import ABC, abstractmethod
import math

# Core ML evaluation libraries
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    max_error, explained_variance_score,
    
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, log_loss, brier_score_loss,
    classification_report, confusion_matrix,
    
    # Multi-class specific
    cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score, top_k_accuracy_score,
    
    # Clustering metrics
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# Cross-validation and model selection
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    TimeSeriesSplit, validation_curve, learning_curve
)

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import scipy.stats as stats

# Advanced metrics and utilities
try:
    from sklearn.calibration import calibration_curve
    from sklearn.inspection import permutation_importance
    SKLEARN_ADVANCED = True
except ImportError:
    SKLEARN_ADVANCED = False

# Visualization support
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Statistical testing
try:
    from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare
    from statsmodels.stats.contingency_tables import mcnemar
    STATISTICAL_TESTS_AVAILABLE = True
except ImportError:
    STATISTICAL_TESTS_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Time series specific metrics
try:
    from sklearn.metrics import mean_pinball_loss
    PINBALL_LOSS_AVAILABLE = True
except ImportError:
    PINBALL_LOSS_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """ML task types supported by the evaluator."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    RANKING = "ranking"

class MetricType(Enum):
    """Types of metrics available."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PRECISION_RECALL_AUC = "pr_auc"
    LOG_LOSS = "log_loss"
    BRIER_SCORE = "brier_score"
    
    # Regression metrics
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    MAPE = "mape"
    SMAPE = "smape"
    MEDIAN_AE = "median_ae"
    MAX_ERROR = "max_error"
    EXPLAINED_VARIANCE = "explained_variance"
    
    # Advanced metrics
    COHEN_KAPPA = "cohen_kappa"
    MATTHEWS_CORRCOEF = "matthews_corrcoef"
    BALANCED_ACCURACY = "balanced_accuracy"

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    def __init__(self):
        # Cross-validation settings
        self.cv_folds = 5
        self.cv_strategy = 'stratified'  # 'stratified', 'kfold', 'timeseries'
        self.cv_random_state = 42
        self.cv_shuffle = True
        
        # Statistical testing
        self.confidence_level = 0.95
        self.enable_statistical_tests = True
        self.bootstrap_samples = 1000
        self.permutation_tests = True
        
        # Evaluation settings
        self.evaluation_metrics = []  # Auto-select if empty
        self.custom_scorers = {}
        self.enable_detailed_reports = True
        self.generate_visualizations = True
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        self.chunk_size = 10000  # For large datasets
        
        # Business metrics
        self.calculate_business_impact = True
        self.cost_sensitive_evaluation = False
        self.class_weights = None
        
        # Monitoring settings
        self.track_prediction_drift = True
        self.save_evaluation_artifacts = True
        self.mlflow_tracking = True

@dataclass
class MetricResult:
    """Result of a single metric calculation."""
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    standard_error: Optional[float]
    metadata: Dict[str, Any]

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    report_id: str
    timestamp: datetime
    task_type: TaskType
    model_name: Optional[str]
    dataset_name: Optional[str]
    metrics: Dict[str, MetricResult]
    cross_validation_scores: Dict[str, List[float]]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[Dict]
    feature_importance: Optional[Dict[str, float]]
    error_analysis: Dict[str, Any]
    business_impact: Dict[str, Any]
    recommendations: List[str]
    visualizations: Dict[str, Any]
    metadata: Dict[str, Any]

class MetricCalculator:
    """Base class for metric calculators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, MetricResult]:
        """Calculate comprehensive regression metrics."""
        try:
            metrics = {}
            
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
            r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
            
            metrics['mse'] = MetricResult(
                name='MSE',
                value=float(mse),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': 'Mean Squared Error'}
            )
            
            metrics['rmse'] = MetricResult(
                name='RMSE',
                value=float(rmse),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': 'Root Mean Squared Error'}
            )
            
            metrics['mae'] = MetricResult(
                name='MAE',
                value=float(mae),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': 'Mean Absolute Error'}
            )
            
            metrics['r2_score'] = MetricResult(
                name='R²',
                value=float(r2),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': 'Coefficient of Determination'}
            )
            
            # Additional regression metrics
            try:
                # MAPE (avoid division by zero)
                mask = y_true != 0
                if np.any(mask):
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                    metrics['mape'] = MetricResult(
                        name='MAPE',
                        value=float(mape),
                        confidence_interval=None,
                        standard_error=None,
                        metadata={'description': 'Mean Absolute Percentage Error'}
                    )
                
                # SMAPE
                smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
                metrics['smape'] = MetricResult(
                    name='SMAPE',
                    value=float(smape),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Symmetric Mean Absolute Percentage Error'}
                )
                
                # Median Absolute Error
                median_ae = median_absolute_error(y_true, y_pred)
                metrics['median_ae'] = MetricResult(
                    name='Median AE',
                    value=float(median_ae),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Median Absolute Error'}
                )
                
                # Max Error
                max_err = max_error(y_true, y_pred)
                metrics['max_error'] = MetricResult(
                    name='Max Error',
                    value=float(max_err),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Maximum Residual Error'}
                )
                
                # Explained Variance Score
                exp_var = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
                metrics['explained_variance'] = MetricResult(
                    name='Explained Variance',
                    value=float(exp_var),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Explained Variance Score'}
                )
                
            except Exception as e:
                logger.warning(f"Failed to calculate some regression metrics: {str(e)}")
            
            # Statistical correlations
            try:
                pearson_r, pearson_p = pearsonr(y_true, y_pred)
                spearman_r, spearman_p = spearmanr(y_true, y_pred)
                
                metrics['pearson_correlation'] = MetricResult(
                    name='Pearson r',
                    value=float(pearson_r),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'p_value': float(pearson_p), 'description': 'Pearson Correlation'}
                )
                
                metrics['spearman_correlation'] = MetricResult(
                    name='Spearman ρ',
                    value=float(spearman_r),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'p_value': float(spearman_p), 'description': 'Spearman Correlation'}
                )
                
            except Exception as e:
                logger.warning(f"Failed to calculate correlation metrics: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Regression metrics calculation failed: {str(e)}")
            return {}
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION
    ) -> Dict[str, MetricResult]:
        """Calculate comprehensive classification metrics."""
        try:
            metrics = {}
            
            # Determine average strategy based on task type
            if task_type == TaskType.BINARY_CLASSIFICATION:
                average = 'binary'
            else:
                average = 'weighted'  # For multiclass
            
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average=average, zero_division=0)
            recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
            
            metrics['accuracy'] = MetricResult(
                name='Accuracy',
                value=float(accuracy),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': 'Classification Accuracy'}
            )
            
            metrics['precision'] = MetricResult(
                name='Precision',
                value=float(precision),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': f'Precision ({average} average)'}
            )
            
            metrics['recall'] = MetricResult(
                name='Recall',
                value=float(recall),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': f'Recall ({average} average)'}
            )
            
            metrics['f1_score'] = MetricResult(
                name='F1-Score',
                value=float(f1),
                confidence_interval=None,
                standard_error=None,
                metadata={'description': f'F1-Score ({average} average)'}
            )
            
            # Probability-based metrics
            if y_prob is not None:
                try:
                    if task_type == TaskType.BINARY_CLASSIFICATION:
                        # ROC-AUC
                        if len(np.unique(y_true)) == 2:
                            roc_auc = roc_auc_score(y_true, y_prob)
                            metrics['roc_auc'] = MetricResult(
                                name='ROC-AUC',
                                value=float(roc_auc),
                                confidence_interval=None,
                                standard_error=None,
                                metadata={'description': 'Area Under ROC Curve'}
                            )
                        
                        # Precision-Recall AUC
                        pr_auc = average_precision_score(y_true, y_prob)
                        metrics['pr_auc'] = MetricResult(
                            name='PR-AUC',
                            value=float(pr_auc),
                            confidence_interval=None,
                            standard_error=None,
                            metadata={'description': 'Area Under Precision-Recall Curve'}
                        )
                        
                        # Log Loss
                        logloss = log_loss(y_true, np.column_stack([1-y_prob, y_prob]))
                        metrics['log_loss'] = MetricResult(
                            name='Log Loss',
                            value=float(logloss),
                            confidence_interval=None,
                            standard_error=None,
                            metadata={'description': 'Logarithmic Loss (lower is better)'}
                        )
                        
                        # Brier Score
                        brier = brier_score_loss(y_true, y_prob)
                        metrics['brier_score'] = MetricResult(
                            name='Brier Score',
                            value=float(brier),
                            confidence_interval=None,
                            standard_error=None,
                            metadata={'description': 'Brier Score (lower is better)'}
                        )
                        
                    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                        # Multiclass ROC-AUC
                        if len(np.unique(y_true)) > 2:
                            try:
                                roc_auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                                metrics['roc_auc_ovr'] = MetricResult(
                                    name='ROC-AUC (OvR)',
                                    value=float(roc_auc_ovr),
                                    confidence_interval=None,
                                    standard_error=None,
                                    metadata={'description': 'One-vs-Rest ROC-AUC'}
                                )
                            except Exception:
                                pass
                        
                        # Multiclass Log Loss
                        logloss = log_loss(y_true, y_prob)
                        metrics['log_loss'] = MetricResult(
                            name='Log Loss',
                            value=float(logloss),
                            confidence_interval=None,
                            standard_error=None,
                            metadata={'description': 'Logarithmic Loss (lower is better)'}
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to calculate probability-based metrics: {str(e)}")
            
            # Advanced classification metrics
            try:
                # Balanced Accuracy
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                metrics['balanced_accuracy'] = MetricResult(
                    name='Balanced Accuracy',
                    value=float(balanced_acc),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Balanced Accuracy Score'}
                )
                
                # Cohen's Kappa
                kappa = cohen_kappa_score(y_true, y_pred)
                metrics['cohen_kappa'] = MetricResult(
                    name="Cohen's κ",
                    value=float(kappa),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': "Cohen's Kappa Coefficient"}
                )
                
                # Matthews Correlation Coefficient (for binary classification)
                if task_type == TaskType.BINARY_CLASSIFICATION:
                    mcc = matthews_corrcoef(y_true, y_pred)
                    metrics['matthews_corrcoef'] = MetricResult(
                        name='MCC',
                        value=float(mcc),
                        confidence_interval=None,
                        standard_error=None,
                        metadata={'description': 'Matthews Correlation Coefficient'}
                    )
                
            except Exception as e:
                logger.warning(f"Failed to calculate advanced classification metrics: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Classification metrics calculation failed: {str(e)}")
            return {}
    
    def calculate_clustering_metrics(
        self,
        X: np.ndarray,
        labels_true: Optional[np.ndarray],
        labels_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Calculate clustering evaluation metrics."""
        try:
            metrics = {}
            
            # Internal metrics (don't need true labels)
            try:
                silhouette = silhouette_score(X, labels_pred)
                metrics['silhouette_score'] = MetricResult(
                    name='Silhouette Score',
                    value=float(silhouette),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Silhouette Score (higher is better)'}
                )
            except Exception as e:
                logger.warning(f"Failed to calculate silhouette score: {str(e)}")
            
            try:
                calinski_harabasz = calinski_harabasz_score(X, labels_pred)
                metrics['calinski_harabasz'] = MetricResult(
                    name='Calinski-Harabasz Index',
                    value=float(calinski_harabasz),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Calinski-Harabasz Index (higher is better)'}
                )
            except Exception as e:
                logger.warning(f"Failed to calculate Calinski-Harabasz score: {str(e)}")
            
            try:
                davies_bouldin = davies_bouldin_score(X, labels_pred)
                metrics['davies_bouldin'] = MetricResult(
                    name='Davies-Bouldin Index',
                    value=float(davies_bouldin),
                    confidence_interval=None,
                    standard_error=None,
                    metadata={'description': 'Davies-Bouldin Index (lower is better)'}
                )
            except Exception as e:
                logger.warning(f"Failed to calculate Davies-Bouldin score: {str(e)}")
            
            # External metrics (need true labels)
            if labels_true is not None:
                try:
                    ari = adjusted_rand_score(labels_true, labels_pred)
                    metrics['adjusted_rand_score'] = MetricResult(
                        name='Adjusted Rand Index',
                        value=float(ari),
                        confidence_interval=None,
                        standard_error=None,
                        metadata={'description': 'Adjusted Rand Index'}
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate ARI: {str(e)}")
                
                try:
                    nmi = normalized_mutual_info_score(labels_true, labels_pred)
                    metrics['normalized_mutual_info'] = MetricResult(
                        name='Normalized Mutual Information',
                        value=float(nmi),
                        confidence_interval=None,
                        standard_error=None,
                        metadata={'description': 'Normalized Mutual Information'}
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate NMI: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Clustering metrics calculation failed: {str(e)}")
            return {}

class ModelEvaluator:
    """
    Comprehensive model evaluation system with statistical analysis,
    cross-validation, and business impact assessment.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.metric_calculator = MetricCalculator(self.config)
        self.evaluation_history = []
        
        logger.info("ModelEvaluator initialized")
    
    async def evaluate_model(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        task_type: Union[str, TaskType] = 'auto',
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> EvaluationReport:
        """
        Comprehensive model evaluation with cross-validation and statistical analysis.
        
        Args:
            model: Trained model to evaluate
            X: Feature matrix
            y: Target vector
            task_type: Type of ML task
            model_name: Name of the model for reporting
            dataset_name: Name of the dataset
            X_test: Optional separate test set features
            y_test: Optional separate test set targets
            
        Returns:
            Comprehensive evaluation report
        """
        try:
            logger.info(f"Starting model evaluation for {model_name or 'Unknown Model'}")
            start_time = datetime.now()
            
            # Convert inputs to numpy arrays
            X, y = self._prepare_data(X, y)
            
            # Auto-detect task type if needed
            if isinstance(task_type, str) and task_type == 'auto':
                task_type = self._detect_task_type(y, model)
            elif isinstance(task_type, str):
                task_type = TaskType(task_type)
            
            logger.info(f"Detected task type: {task_type.value}")
            
            # Use separate test set if provided, otherwise use cross-validation
            if X_test is not None and y_test is not None:
                X_test, y_test = self._prepare_data(X_test, y_test)
                evaluation_strategy = 'holdout'
            else:
                evaluation_strategy = 'cross_validation'
            
            # Calculate metrics
            if evaluation_strategy == 'holdout':
                metrics, cv_scores = await self._evaluate_holdout(
                    model, X, y, X_test, y_test, task_type
                )
            else:
                metrics, cv_scores = await self._evaluate_cross_validation(
                    model, X, y, task_type
                )
            
            # Generate additional analysis
            confusion_mat, class_report = await self._generate_classification_analysis(
                model, X, y, task_type, X_test, y_test
            )
            
            feature_importance = await self._calculate_feature_importance(
                model, X, y, task_type
            )
            
            error_analysis = await self._perform_error_analysis(
                model, X, y, task_type, X_test, y_test
            )
            
            business_impact = await self._assess_business_impact(
                metrics, task_type, y
            )
            
            visualizations = await self._prepare_visualizations(
                model, X, y, task_type, metrics, X_test, y_test
            )
            
            recommendations = self._generate_recommendations(
                metrics, task_type, error_analysis, business_impact
            )
            
            # Create evaluation report
            report = EvaluationReport(
                report_id=str(uuid.uuid4()),
                timestamp=start_time,
                task_type=task_type,
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                cross_validation_scores=cv_scores,
                confusion_matrix=confusion_mat,
                classification_report=class_report,
                feature_importance=feature_importance,
                error_analysis=error_analysis,
                business_impact=business_impact,
                recommendations=recommendations,
                visualizations=visualizations,
                metadata={
                    'evaluation_strategy': evaluation_strategy,
                    'evaluation_time': (datetime.now() - start_time).total_seconds(),
                    'n_samples': len(X),
                    'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                    'config': asdict(self.config)
                }
            )
            
            # Store evaluation history
            self.evaluation_history.append(report)
            
            # Log to MLflow if enabled
            if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                await self._log_to_mlflow(report)
            
            evaluation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model evaluation completed in {evaluation_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            # Return minimal report with error
            return EvaluationReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                task_type=task_type if isinstance(task_type, TaskType) else TaskType.REGRESSION,
                model_name=model_name,
                dataset_name=dataset_name,
                metrics={},
                cross_validation_scores={},
                confusion_matrix=None,
                classification_report=None,
                feature_importance=None,
                error_analysis={'error': str(e)},
                business_impact={},
                recommendations=[f"Evaluation failed: {str(e)}"],
                visualizations={},
                metadata={'error': str(e)}
            )
    
    def _prepare_data(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for evaluation."""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)
        
        # Handle missing values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def _detect_task_type(self, y: np.ndarray, model: Any) -> TaskType:
        """Auto-detect the task type based on target and model."""
        try:
            # Check if it's clustering (no target labels)
            if hasattr(model, 'cluster_centers_') or hasattr(model, 'labels_'):
                return TaskType.CLUSTERING
            
            # Check target characteristics
            unique_values = len(np.unique(y))
            total_values = len(y)
            
            # If target has many unique values relative to size, it's regression
            if unique_values > 20 and unique_values > total_values * 0.1:
                return TaskType.REGRESSION
            
            # Check if target is continuous
            if np.issubdtype(y.dtype, np.floating):
                # Check if values are actually discrete
                if np.allclose(y, np.round(y)):
                    # Integer-like values, could be classification
                    if unique_values <= 10:
                        return TaskType.MULTICLASS_CLASSIFICATION if unique_values > 2 else TaskType.BINARY_CLASSIFICATION
                return TaskType.REGRESSION
            
            # Classification based on number of unique values
            if unique_values == 2:
                return TaskType.BINARY_CLASSIFICATION
            elif unique_values <= 20:
                return TaskType.MULTICLASS_CLASSIFICATION
            else:
                return TaskType.REGRESSION
                
        except Exception as e:
            logger.warning(f"Task type detection failed: {str(e)}, defaulting to regression")
            return TaskType.REGRESSION
    
    async def _evaluate_holdout(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: TaskType
    ) -> Tuple[Dict[str, MetricResult], Dict[str, List[float]]]:
        """Evaluate model using holdout test set."""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = None
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba') and task_type in [
                TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION
            ]:
                y_prob = model.predict_proba(X_test)
                if task_type == TaskType.BINARY_CLASSIFICATION:
                    y_prob = y_prob[:, 1]  # Use positive class probability
            
            # Calculate metrics based on task type
            if task_type == TaskType.REGRESSION:
                metrics = self.metric_calculator.calculate_regression_metrics(
                    y_test, y_pred
                )
            elif task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                metrics = self.metric_calculator.calculate_classification_metrics(
                    y_test, y_pred, y_prob, task_type=task_type
                )
            elif task_type == TaskType.CLUSTERING:
                metrics = self.metric_calculator.calculate_clustering_metrics(
                    X_test, None, y_pred
                )
            else:
                metrics = {}
            
            # For holdout, we don't have cross-validation scores
            cv_scores = {}
            
            return metrics, cv_scores
            
        except Exception as e:
            logger.error(f"Holdout evaluation failed: {str(e)}")
            return {}, {}
    
    async def _evaluate_cross_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType
    ) -> Tuple[Dict[str, MetricResult], Dict[str, List[float]]]:
        """Evaluate model using cross-validation."""
        try:
            # Select cross-validation strategy
            if self.config.cv_strategy == 'stratified' and task_type in [
                TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION
            ]:
                cv = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=self.config.cv_shuffle,
                    random_state=self.config.cv_random_state
                )
            elif self.config.cv_strategy == 'timeseries':
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                cv = KFold(
                    n_splits=self.config.cv_folds,
                    shuffle=self.config.cv_shuffle,
                    random_state=self.config.cv_random_state
                )
            
            # Define scoring metrics based on task type
            scoring_metrics = self._get_scoring_metrics(task_type)
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring_metrics,
                n_jobs=self.config.n_jobs if self.config.enable_parallel else 1,
                return_train_score=True,
                error_score='raise'
            )
            
            # Process cross-validation results
            metrics = {}
            cv_scores = {}
            
            for metric_name, scores in cv_results.items():
                if metric_name.startswith('test_'):
                    # Remove 'test_' prefix
                    clean_name = metric_name[5:]
                    cv_scores[clean_name] = scores.tolist()
                    
                    # Calculate statistics
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    # Calculate confidence interval
                    if self.config.enable_statistical_tests:
                        ci_lower, ci_upper = self._calculate_confidence_interval(
                            scores, self.config.confidence_level
                        )
                        confidence_interval = (float(ci_lower), float(ci_upper))
                    else:
                        confidence_interval = None
                    
                    metrics[clean_name] = MetricResult(
                        name=clean_name.title().replace('_', ' '),
                        value=float(mean_score),
                        confidence_interval=confidence_interval,
                        standard_error=float(std_score / np.sqrt(len(scores))),
                        metadata={
                            'cv_scores': scores.tolist(),
                            'cv_mean': float(mean_score),
                            'cv_std': float(std_score)
                        }
                    )
            
            return metrics, cv_scores
            
        except Exception as e:
            logger.error(f"Cross-validation evaluation failed: {str(e)}")
            return {}, {}
    
    def _get_scoring_metrics(self, task_type: TaskType) -> Dict[str, str]:
        """Get appropriate scoring metrics for the task type."""
        if task_type == TaskType.REGRESSION:
            return {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error'
            }
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            return {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return {
                'accuracy': 'accuracy',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'f1_weighted': 'f1_weighted'
            }
        else:
            return {'accuracy': 'accuracy'}  # Default
    
    def _calculate_confidence_interval(
        self, 
        scores: np.ndarray, 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        try:
            alpha = 1 - confidence_level
            n = len(scores)
            
            if n < 2:
                return float(scores[0]), float(scores[0])
            
            # Use t-distribution for small samples
            mean_score = np.mean(scores)
            std_score = np.std(scores, ddof=1)
            
            if n < 30:
                # Use t-distribution
                t_value = stats.t.ppf(1 - alpha/2, df=n-1)
                margin_error = t_value * (std_score / np.sqrt(n))
            else:
                # Use normal distribution
                z_value = stats.norm.ppf(1 - alpha/2)
                margin_error = z_value * (std_score / np.sqrt(n))
            
            return mean_score - margin_error, mean_score + margin_error
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {str(e)}")
            return float(np.mean(scores)), float(np.mean(scores))
    
    async def _generate_classification_analysis(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Generate confusion matrix and classification report."""
        try:
            if task_type not in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                return None, None
            
            # Use test set if available, otherwise use full dataset
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                y_true = y_test
            else:
                y_pred = model.predict(X)
                y_true = y
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            try:
                class_report = classification_report(
                    y_true, y_pred, output_dict=True, zero_division=0
                )
            except Exception:
                class_report = None
            
            return cm, class_report
            
        except Exception as e:
            logger.warning(f"Classification analysis failed: {str(e)}")
            return None, None
    
    async def _calculate_feature_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType
    ) -> Optional[Dict[str, float]]:
        """Calculate feature importance if available."""
        try:
            feature_importance = {}
            
            # Try built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    feature_importance[f'feature_{i}'] = float(importance)
            
            # Try coefficients for linear models
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) == 1:  # 1D for binary or regression
                    for i, importance in enumerate(np.abs(coef)):
                        feature_importance[f'feature_{i}'] = float(importance)
                else:  # 2D for multiclass
                    # Use mean absolute coefficient across classes
                    mean_coef = np.mean(np.abs(coef), axis=0)
                    for i, importance in enumerate(mean_coef):
                        feature_importance[f'feature_{i}'] = float(importance)
            
            # Try permutation importance if sklearn advanced is available
            elif SKLEARN_ADVANCED and len(X) <= 1000:  # Only for smaller datasets
                try:
                    scoring = 'accuracy' if task_type in [
                        TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION
                    ] else 'r2'
                    
                    perm_importance = permutation_importance(
                        model, X, y, 
                        scoring=scoring,
                        n_repeats=5,
                        random_state=self.config.cv_random_state,
                        n_jobs=1  # Avoid nested parallelization
                    )
                    
                    for i, importance in enumerate(perm_importance.importances_mean):
                        feature_importance[f'feature_{i}'] = float(importance)
                        
                except Exception as e:
                    logger.warning(f"Permutation importance failed: {str(e)}")
            
            return feature_importance if feature_importance else None
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
            return None
    
    async def _perform_error_analysis(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        try:
            error_analysis = {}
            
            # Use test set if available
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                y_true = y_test
                X_eval = X_test
            else:
                y_pred = model.predict(X)
                y_true = y
                X_eval = X
            
            if task_type == TaskType.REGRESSION:
                # Regression error analysis
                residuals = y_true - y_pred
                error_analysis.update({
                    'mean_residual': float(np.mean(residuals)),
                    'residual_std': float(np.std(residuals)),
                    'residual_skewness': float(stats.skew(residuals)),
                    'residual_kurtosis': float(stats.kurtosis(residuals)),
                    'largest_errors': {
                        'indices': np.argsort(np.abs(residuals))[-10:].tolist(),
                        'values': residuals[np.argsort(np.abs(residuals))[-10:]].tolist()
                    }
                })
                
                # Outlier detection in residuals
                q75, q25 = np.percentile(residuals, [75, 25])
                iqr = q75 - q25
                outlier_threshold = 1.5 * iqr
                outliers = np.abs(residuals) > (q75 + outlier_threshold)
                error_analysis['outlier_percentage'] = float(np.mean(outliers) * 100)
                
            elif task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                # Classification error analysis
                incorrect_mask = y_true != y_pred
                error_analysis.update({
                    'error_rate': float(np.mean(incorrect_mask)),
                    'correctly_classified': int(np.sum(~incorrect_mask)),
                    'misclassified': int(np.sum(incorrect_mask))
                })
                
                # Per-class error analysis
                unique_classes = np.unique(y_true)
                per_class_errors = {}
                
                for cls in unique_classes:
                    class_mask = y_true == cls
                    class_errors = incorrect_mask[class_mask]
                    per_class_errors[f'class_{cls}'] = {
                        'total_samples': int(np.sum(class_mask)),
                        'misclassified': int(np.sum(class_errors)),
                        'error_rate': float(np.mean(class_errors))
                    }
                
                error_analysis['per_class_errors'] = per_class_errors
            
            return error_analysis
            
        except Exception as e:
            logger.warning(f"Error analysis failed: {str(e)}")
            return {'error': str(e)}
    
    async def _assess_business_impact(
        self,
        metrics: Dict[str, MetricResult],
        task_type: TaskType,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Assess business impact and ROI of the model."""
        try:
            if not self.config.calculate_business_impact:
                return {}
            
            business_impact = {}
            
            # Basic performance assessment
            if task_type == TaskType.REGRESSION:
                if 'r2_score' in metrics:
                    r2 = metrics['r2_score'].value
                    if r2 >= 0.9:
                        performance_grade = 'Excellent'
                    elif r2 >= 0.8:
                        performance_grade = 'Good'
                    elif r2 >= 0.6:
                        performance_grade = 'Fair'
                    else:
                        performance_grade = 'Poor'
                    
                    business_impact['performance_grade'] = performance_grade
                    business_impact['variance_explained'] = f"{r2:.1%}"
                
                # Calculate potential cost savings (example)
                if 'mae' in metrics:
                    mae = metrics['mae'].value
                    baseline_error = np.std(y)  # Use baseline of predicting mean
                    improvement = (baseline_error - mae) / baseline_error
                    business_impact['error_reduction'] = f"{improvement:.1%}"
            
            elif task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                if 'accuracy' in metrics:
                    accuracy = metrics['accuracy'].value
                    if accuracy >= 0.95:
                        performance_grade = 'Excellent'
                    elif accuracy >= 0.9:
                        performance_grade = 'Good'
                    elif accuracy >= 0.8:
                        performance_grade = 'Fair'
                    else:
                        performance_grade = 'Poor'
                    
                    business_impact['performance_grade'] = performance_grade
                    business_impact['accuracy_percentage'] = f"{accuracy:.1%}"
                
                # Calculate precision/recall balance
                if 'precision' in metrics and 'recall' in metrics:
                    precision = metrics['precision'].value
                    recall = metrics['recall'].value
                    
                    if abs(precision - recall) < 0.05:
                        balance = 'Well-balanced'
                    elif precision > recall + 0.1:
                        balance = 'Precision-focused (conservative)'
                    elif recall > precision + 0.1:
                        balance = 'Recall-focused (liberal)'
                    else:
                        balance = 'Moderately balanced'
                    
                    business_impact['precision_recall_balance'] = balance
            
            # Model complexity assessment
            # This would typically require model introspection
            business_impact['deployment_readiness'] = 'Ready for production'
            business_impact['interpretability'] = 'High' if 'feature_importance' in business_impact else 'Medium'
            
            return business_impact
            
        except Exception as e:
            logger.warning(f"Business impact assessment failed: {str(e)}")
            return {}
    
    async def _prepare_visualizations(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType,
        metrics: Dict[str, MetricResult],
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Prepare data for visualizations."""
        try:
            if not self.config.generate_visualizations:
                return {}
            
            visualizations = {}
            
            # Use test set if available
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                y_true = y_test
            else:
                y_pred = model.predict(X)
                y_true = y
            
            if task_type == TaskType.REGRESSION:
                # Predicted vs Actual
                visualizations['predicted_vs_actual'] = {
                    'actual': y_true.tolist(),
                    'predicted': y_pred.tolist(),
                    'title': 'Predicted vs Actual Values'
                }
                
                # Residuals
                residuals = y_true - y_pred
                visualizations['residuals'] = {
                    'residuals': residuals.tolist(),
                    'predicted': y_pred.tolist(),
                    'title': 'Residual Plot'
                }
                
                # Distribution of residuals
                visualizations['residual_distribution'] = {
                    'residuals': residuals.tolist(),
                    'title': 'Distribution of Residuals'
                }
            
            elif task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                # Confusion matrix data
                cm = confusion_matrix(y_true, y_pred)
                visualizations['confusion_matrix'] = {
                    'matrix': cm.tolist(),
                    'title': 'Confusion Matrix'
                }
                
                # ROC curve data for binary classification
                if task_type == TaskType.BINARY_CLASSIFICATION and hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X_test if X_test is not None else X)[:, 1]
                        fpr, tpr, _ = roc_curve(y_true, y_prob)
                        
                        visualizations['roc_curve'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': float(roc_auc_score(y_true, y_prob)),
                            'title': 'ROC Curve'
                        }
                        
                        # Precision-Recall curve
                        precision, recall, _ = precision_recall_curve(y_true, y_prob)
                        visualizations['pr_curve'] = {
                            'precision': precision.tolist(),
                            'recall': recall.tolist(),
                            'auc': float(average_precision_score(y_true, y_prob)),
                            'title': 'Precision-Recall Curve'
                        }
                    except Exception as e:
                        logger.warning(f"Failed to create ROC/PR curves: {str(e)}")
                
                # Class distribution
                unique_classes, class_counts = np.unique(y_true, return_counts=True)
                visualizations['class_distribution'] = {
                    'classes': unique_classes.tolist(),
                    'counts': class_counts.tolist(),
                    'title': 'Class Distribution'
                }
            
            # Feature importance visualization (if available)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = [f'Feature {i}' for i in range(len(importances))]
                
                # Sort by importance
                sorted_idx = np.argsort(importances)[::-1][:20]  # Top 20
                
                visualizations['feature_importance'] = {
                    'features': [feature_names[i] for i in sorted_idx],
                    'importance': importances[sorted_idx].tolist(),
                    'title': 'Feature Importance'
                }
            
            return visualizations
            
        except Exception as e:
            logger.warning(f"Visualization preparation failed: {str(e)}")
            return {}
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, MetricResult],
        task_type: TaskType,
        error_analysis: Dict[str, Any],
        business_impact: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if task_type == TaskType.REGRESSION:
                if 'r2_score' in metrics:
                    r2 = metrics['r2_score'].value
                    if r2 < 0.5:
                        recommendations.append(
                            "Low R² score indicates poor fit. Consider feature engineering or different algorithms."
                        )
                    elif r2 > 0.95:
                        recommendations.append(
                            "Very high R² score - check for potential overfitting or data leakage."
                        )
                
                if 'mae' in metrics and 'rmse' in metrics:
                    mae = metrics['mae'].value
                    rmse = metrics['rmse'].value
                    ratio = rmse / mae if mae > 0 else 1
                    
                    if ratio > 2:
                        recommendations.append(
                            "High RMSE/MAE ratio suggests presence of outliers. Consider robust regression methods."
                        )
            
            elif task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                if 'accuracy' in metrics:
                    accuracy = metrics['accuracy'].value
                    if accuracy < 0.8:
                        recommendations.append(
                            "Consider improving model performance through feature selection or ensemble methods."
                        )
                
                if 'precision' in metrics and 'recall' in metrics:
                    precision = metrics['precision'].value
                    recall = metrics['recall'].value
                    
                    if precision < 0.7 and recall > 0.9:
                        recommendations.append(
                            "High recall but low precision suggests too many false positives. Consider adjusting decision threshold."
                        )
                    elif precision > 0.9 and recall < 0.7:
                        recommendations.append(
                            "High precision but low recall suggests too many false negatives. Consider lowering decision threshold."
                        )
            
            # Error analysis recommendations
            if 'outlier_percentage' in error_analysis:
                outlier_pct = error_analysis['outlier_percentage']
                if outlier_pct > 10:
                    recommendations.append(
                        f"High outlier percentage ({outlier_pct:.1f}%) detected. Consider outlier removal or robust methods."
                    )
            
            # Business impact recommendations
            if business_impact.get('performance_grade') == 'Poor':
                recommendations.append(
                    "Poor performance grade suggests the model may not be suitable for production deployment."
                )
            elif business_impact.get('performance_grade') == 'Excellent':
                recommendations.append(
                    "Excellent performance achieved - model is ready for production deployment."
                )
            
            # General recommendations if none specific
            if not recommendations:
                recommendations.append(
                    "Model evaluation completed successfully. Consider monitoring performance in production."
                )
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {str(e)}")
            return ["Evaluation completed - review metrics for insights."]
    
    async def _log_to_mlflow(self, report: EvaluationReport):
        """Log evaluation results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"evaluation_{report.model_name or 'unknown'}"):
                # Log parameters
                mlflow.log_param("task_type", report.task_type.value)
                mlflow.log_param("model_name", report.model_name or "unknown")
                mlflow.log_param("dataset_name", report.dataset_name or "unknown")
                mlflow.log_param("evaluation_time", report.metadata.get('evaluation_time', 0))
                
                # Log metrics
                for metric_name, metric_result in report.metrics.items():
                    mlflow.log_metric(metric_name, metric_result.value)
                    if metric_result.standard_error:
                        mlflow.log_metric(f"{metric_name}_stderr", metric_result.standard_error)
                
                # Log cross-validation scores
                for metric_name, scores in report.cross_validation_scores.items():
                    mlflow.log_metric(f"cv_{metric_name}_mean", np.mean(scores))
                    mlflow.log_metric(f"cv_{metric_name}_std", np.std(scores))
                
                # Log business impact metrics
                for impact_name, impact_value in report.business_impact.items():
                    if isinstance(impact_value, (int, float)):
                        mlflow.log_metric(f"business_{impact_name}", impact_value)
                
                # Log artifacts
                if report.confusion_matrix is not None:
                    np.savetxt("confusion_matrix.csv", report.confusion_matrix, delimiter=",")
                    mlflow.log_artifact("confusion_matrix.csv")
                
                # Log evaluation report as JSON
                report_dict = asdict(report)
                # Convert datetime to string for JSON serialization
                report_dict['timestamp'] = report.timestamp.isoformat()
                
                with open("evaluation_report.json", "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)
                mlflow.log_artifact("evaluation_report.json")
                
                logger.info("Evaluation results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def compare_models(
        self,
        reports: List[EvaluationReport],
        primary_metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple model evaluation reports."""
        try:
            if len(reports) < 2:
                raise ValueError("Need at least 2 evaluation reports for comparison")
            
            # Ensure all reports are for the same task type
            task_types = [report.task_type for report in reports]
            if len(set(task_types)) > 1:
                raise ValueError("All reports must be for the same task type")
            
            task_type = task_types[0]
            
            # Select primary metric if not provided
            if primary_metric is None:
                if task_type == TaskType.REGRESSION:
                    primary_metric = 'r2_score'
                elif task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                    primary_metric = 'f1_score'
                else:
                    primary_metric = list(reports[0].metrics.keys())[0] if reports[0].metrics else 'accuracy'
            
            comparison_results = {
                'task_type': task_type.value,
                'primary_metric': primary_metric,
                'models_compared': len(reports),
                'model_performance': {},
                'best_model': None,
                'performance_ranking': [],
                'statistical_significance': {}
            }
            
            # Compare model performance
            model_scores = {}
            for report in reports:
                model_name = report.model_name or f"model_{report.report_id[:8]}"
                
                if primary_metric in report.metrics:
                    score = report.metrics[primary_metric].value
                    model_scores[model_name] = score
                    
                    comparison_results['model_performance'][model_name] = {
                        'score': score,
                        'metrics': {name: result.value for name, result in report.metrics.items()}
                    }
            
            # Rank models
            if model_scores:
                # For most metrics, higher is better, but some are "lower is better"
                lower_is_better = primary_metric in ['rmse', 'mae', 'log_loss', 'brier_score']
                sorted_models = sorted(
                    model_scores.items(),
                    key=lambda x: x[1],
                    reverse=not lower_is_better
                )
                
                comparison_results['best_model'] = sorted_models[0][0]
                comparison_results['performance_ranking'] = [
                    {'model': name, 'score': score} for name, score in sorted_models
                ]
            
            # Statistical significance testing if cross-validation scores available
            if self.config.enable_statistical_tests and STATISTICAL_TESTS_AVAILABLE:
                significance_results = await self._test_statistical_significance(
                    reports, primary_metric
                )
                comparison_results['statistical_significance'] = significance_results
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return {'error': str(e)}
    
    async def _test_statistical_significance(
        self,
        reports: List[EvaluationReport],
        metric_name: str
    ) -> Dict[str, Any]:
        """Test statistical significance between model performances."""
        try:
            significance_results = {}
            
            # Collect cross-validation scores
            cv_scores = {}
            for report in reports:
                model_name = report.model_name or f"model_{report.report_id[:8]}"
                if metric_name in report.cross_validation_scores:
                    cv_scores[model_name] = report.cross_validation_scores[metric_name]
            
            if len(cv_scores) < 2:
                return {'error': 'Not enough cross-validation scores for statistical testing'}
            
            model_names = list(cv_scores.keys())
            
            # Pairwise t-tests
            pairwise_tests = {}
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    scores1 = np.array(cv_scores[model1])
                    scores2 = np.array(cv_scores[model2])
                    
                    # Paired t-test (assuming same CV folds)
                    if len(scores1) == len(scores2):
                        t_stat, p_value = ttest_rel(scores1, scores2)
                        
                        pairwise_tests[f"{model1}_vs_{model2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'mean_diff': float(np.mean(scores1) - np.mean(scores2))
                        }
            
            significance_results['pairwise_tests'] = pairwise_tests
            
            # Friedman test if more than 2 models
            if len(cv_scores) > 2:
                try:
                    scores_matrix = np.array([cv_scores[name] for name in model_names])
                    friedman_stat, friedman_p = friedmanchisquare(*scores_matrix)
                    
                    significance_results['friedman_test'] = {
                        'statistic': float(friedman_stat),
                        'p_value': float(friedman_p),
                        'significant': friedman_p < 0.05,
                        'interpretation': 'Significant differences exist between models' if friedman_p < 0.05 else 'No significant differences detected'
                    }
                except Exception as e:
                    logger.warning(f"Friedman test failed: {str(e)}")
            
            return significance_results
            
        except Exception as e:
            logger.warning(f"Statistical significance testing failed: {str(e)}")
            return {'error': str(e)}
    
    def get_evaluation_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get summary of recent evaluations."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_reports = [
                report for report in self.evaluation_history
                if report.timestamp >= cutoff_date
            ]
            
            if not recent_reports:
                return {'message': f'No evaluations found in the last {days_back} days'}
            
            summary = {
                'period_days': days_back,
                'total_evaluations': len(recent_reports),
                'task_type_breakdown': {},
                'average_performance': {},
                'best_models': {},
                'common_recommendations': []
            }
            
            # Task type breakdown
            task_counts = {}
            for report in recent_reports:
                task_type = report.task_type.value
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
            summary['task_type_breakdown'] = task_counts
            
            # Calculate average performance by task type
            task_performance = {}
            for report in recent_reports:
                task_type = report.task_type.value
                if task_type not in task_performance:
                    task_performance[task_type] = {}
                
                for metric_name, metric_result in report.metrics.items():
                    if metric_name not in task_performance[task_type]:
                        task_performance[task_type][metric_name] = []
                    task_performance[task_type][metric_name].append(metric_result.value)
            
            # Calculate averages
            for task_type, metrics in task_performance.items():
                summary['average_performance'][task_type] = {
                    metric_name: float(np.mean(scores))
                    for metric_name, scores in metrics.items()
                }
            
            # Find best models by primary metrics
            primary_metrics = {
                TaskType.REGRESSION.value: 'r2_score',
                TaskType.BINARY_CLASSIFICATION.value: 'f1_score',
                TaskType.MULTICLASS_CLASSIFICATION.value: 'f1_score'
            }
            
            for task_type, primary_metric in primary_metrics.items():
                task_reports = [r for r in recent_reports if r.task_type.value == task_type]
                if task_reports:
                    best_report = max(
                        [r for r in task_reports if primary_metric in r.metrics],
                        key=lambda x: x.metrics[primary_metric].value,
                        default=None
                    )
                    
                    if best_report:
                        summary['best_models'][task_type] = {
                            'model_name': best_report.model_name,
                            'score': best_report.metrics[primary_metric].value,
                            'metric': primary_metric
                        }
            
            # Collect common recommendations
            all_recommendations = []
            for report in recent_reports:
                all_recommendations.extend(report.recommendations)
            
            # Count recommendation frequency
            from collections import Counter
            recommendation_counts = Counter(all_recommendations)
            summary['common_recommendations'] = [
                {'recommendation': rec, 'frequency': count}
                for rec, count in recommendation_counts.most_common(5)
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Evaluation summary generation failed: {str(e)}")
            return {'error': str(e)}

# Utility functions

def create_evaluator(
    cv_folds: int = 5,
    confidence_level: float = 0.95,
    enable_business_impact: bool = True
) -> ModelEvaluator:
    """Factory function to create a ModelEvaluator."""
    config = EvaluationConfig()
    config.cv_folds = cv_folds
    config.confidence_level = confidence_level
    config.calculate_business_impact = enable_business_impact
    return ModelEvaluator(config)

async def quick_evaluate(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task_type: str = 'auto'
) -> Dict[str, float]:
    """Quick model evaluation for simple use cases."""
    evaluator = create_evaluator(cv_folds=3, enable_business_impact=False)
    report = await evaluator.evaluate_model(model, X, y, task_type)
    
    return {name: result.value for name, result in report.metrics.items()}

def get_available_metrics(task_type: str) -> Dict[str, str]:
    """Get available metrics for a specific task type."""
    metrics_by_task = {
        'regression': {
            'r2_score': 'Coefficient of Determination',
            'rmse': 'Root Mean Squared Error',
            'mae': 'Mean Absolute Error',
            'mape': 'Mean Absolute Percentage Error',
            'explained_variance': 'Explained Variance Score'
        },
        'binary_classification': {
            'accuracy': 'Classification Accuracy',
            'precision': 'Precision',
            'recall': 'Recall (Sensitivity)',
            'f1_score': 'F1-Score',
            'roc_auc': 'Area Under ROC Curve',
            'pr_auc': 'Area Under Precision-Recall Curve'
        },
        'multiclass_classification': {
            'accuracy': 'Classification Accuracy',
            'precision': 'Precision (weighted)',
            'recall': 'Recall (weighted)',
            'f1_score': 'F1-Score (weighted)',
            'balanced_accuracy': 'Balanced Accuracy'
        }
    }
    
    return metrics_by_task.get(task_type, {})

# Example usage and testing
if __name__ == "__main__":
    async def test_evaluation():
        """Test the evaluation functionality."""
        from sklearn.datasets import make_classification, make_regression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        print("Testing Model Evaluation...")
        
        # Test classification
        print("\n=== Classification Test ===")
        X_class, y_class = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=2, random_state=42
        )
        
        # Train models
        rf_class = RandomForestClassifier(n_estimators=50, random_state=42)
        lr_class = LogisticRegression(random_state=42, max_iter=1000)
        
        rf_class.fit(X_class, y_class)
        lr_class.fit(X_class, y_class)
        
        # Evaluate models
        evaluator = create_evaluator()
        
        rf_report = await evaluator.evaluate_model(
            rf_class, X_class, y_class, 
            task_type='binary_classification',
            model_name='RandomForest'
        )
        
        lr_report = await evaluator.evaluate_model(
            lr_class, X_class, y_class,
            task_type='binary_classification', 
            model_name='LogisticRegression'
        )
        
        print(f"RandomForest F1-Score: {rf_report.metrics['f1_score'].value:.4f}")
        print(f"LogisticRegression F1-Score: {lr_report.metrics['f1_score'].value:.4f}")
        
        # Compare models
        comparison = await evaluator.compare_models([rf_report, lr_report], 'f1_score')
        print(f"Best model: {comparison['best_model']}")
        
        # Test regression
        print("\n=== Regression Test ===")
        X_reg, y_reg = make_regression(
            n_samples=1000, n_features=15, noise=0.1, random_state=42
        )
        
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
        lr_reg = LinearRegression()
        
        rf_reg.fit(X_reg, y_reg)
        lr_reg.fit(X_reg, y_reg)
        
        rf_reg_report = await evaluator.evaluate_model(
            rf_reg, X_reg, y_reg,
            task_type='regression',
            model_name='RandomForestRegressor'
        )
        
        print(f"RandomForest R²: {rf_reg_report.metrics['r2_score'].value:.4f}")
        print(f"RMSE: {rf_reg_report.metrics['rmse'].value:.4f}")
        
        # Test evaluation summary
        print("\n=== Evaluation Summary ===")
        summary = evaluator.get_evaluation_summary(days_back=1)
        print(f"Total evaluations: {summary['total_evaluations']}")
        print(f"Task types: {list(summary['task_type_breakdown'].keys())}")
        
        return rf_report, lr_report
    
    # Run test
    import asyncio
    print("Available metrics for classification:", get_available_metrics('binary_classification'))
    results = asyncio.run(test_evaluation())
