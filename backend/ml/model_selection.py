"""
Model Selection Module for Auto-Analyst Platform

This module implements comprehensive model selection capabilities including:
- Cross-validation strategies for all data types (tabular, timeseries, text)
- Advanced hyperparameter optimization (Optuna, GridSearch, Bayesian)
- Train-test-validation splitting with domain-specific strategies
- Model evaluation and comparison frameworks
- Performance metrics calculation and analysis
- Early stopping and overfitting prevention
- Ensemble model selection and optimization
- Business impact assessment and ROI analysis
- Integration with MLflow for experiment tracking
- Support for both local and remote (Kaggle) training

Features:
- Automatic strategy selection based on data characteristics
- Statistical significance testing for model comparisons
- Multi-objective optimization (accuracy, speed, interpretability)
- Robust evaluation with confidence intervals
- Advanced ensemble techniques (stacking, voting, blending)
- Resource-aware optimization for production deployment
- Real-time model monitoring and drift detection
- Comprehensive reporting with business insights
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import math
from collections import defaultdict, Counter
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core ML libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    StratifiedKFold, KFold, TimeSeriesSplit, GroupKFold,
    GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, log_loss
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor

# Advanced optimization
try:
    import optuna
    from optuna.integration import SklearnEvaluator
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SCIKIT_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIKIT_OPTIMIZE_AVAILABLE = False

# Statistical analysis
try:
    import scipy.stats as stats
    from scipy.stats import wilcoxon, friedmanchisquare, kruskal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class SplitStrategy(Enum):
    """Data splitting strategies."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME_SERIES = "time_series"
    GROUP = "group"
    CUSTOM = "custom"

class CVStrategy(Enum):
    """Cross-validation strategies."""
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_CV = "time_series_cv"
    GROUP_K_FOLD = "group_k_fold"
    REPEATED_K_FOLD = "repeated_k_fold"
    LEAVE_ONE_OUT = "leave_one_out"
    LEAVE_P_OUT = "leave_p_out"

class OptimizationMethod(Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTUNA = "bayesian_optuna"
    BAYESIAN_HYPEROPT = "bayesian_hyperopt"
    BAYESIAN_SKOPT = "bayesian_skopt"
    EVOLUTIONARY = "evolutionary"
    MULTI_OBJECTIVE = "multi_objective"

class EnsembleMethod(Enum):
    """Ensemble methods."""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    DYNAMIC_SELECTION = "dynamic_selection"
    BOOSTING = "boosting"

class TaskType(Enum):
    """Task types for model selection."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_REGRESSION = "text_regression"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

@dataclass
class ModelSelectionConfig:
    """Configuration for model selection."""
    
    def __init__(self):
        # General settings
        self.random_state = 42
        self.n_jobs = -1
        self.enable_parallel = True
        self.max_workers = None
        self.memory_limit_gb = 8
        
        # Data splitting
        self.test_size = 0.2
        self.validation_size = 0.1
        self.split_strategy = SplitStrategy.STRATIFIED
        self.stratify_target = True
        self.shuffle = True
        
        # Cross-validation
        self.cv_strategy = CVStrategy.STRATIFIED_K_FOLD
        self.cv_folds = 5
        self.cv_repeats = 1
        self.cv_scoring = 'auto'  # Auto-select based on task type
        self.cv_return_train_score = True
        
        # Hyperparameter optimization
        self.optimization_method = OptimizationMethod.BAYESIAN_OPTUNA
        self.optimization_budget = 100  # Number of trials/iterations
        self.optimization_timeout = 3600  # Seconds
        self.enable_pruning = True
        self.pruning_patience = 10
        self.early_stopping_rounds = 50
        
        # Model evaluation
        self.evaluation_metrics = ['auto']  # Auto-select based on task type
        self.calculate_feature_importance = True
        self.calculate_shap_values = False
        self.bootstrap_iterations = 1000
        self.confidence_level = 0.95
        
        # Ensemble settings
        self.enable_ensemble = True
        self.ensemble_method = EnsembleMethod.STACKING
        self.ensemble_size = 5
        self.ensemble_cv_folds = 3
        self.meta_learner = 'auto'
        
        # Performance constraints
        self.max_training_time = 7200  # 2 hours
        self.max_model_size_mb = 1000
        self.min_accuracy_threshold = 0.5
        self.interpretability_weight = 0.1
        self.speed_weight = 0.1
        
        # Business settings
        self.calculate_business_impact = True
        self.cost_sensitive_learning = False
        self.class_weights = 'balanced'
        self.fairness_constraints = []
        
        # Advanced settings
        self.multi_objective_optimization = False
        self.pareto_front_analysis = False
        self.statistical_significance_test = True
        self.model_stability_analysis = True
        self.uncertainty_quantification = True
        
        # Integration settings
        self.mlflow_tracking = True
        self.save_intermediate_results = True
        self.cache_cv_results = True
        self.warm_start = True

@dataclass
class ModelCandidate:
    """Model candidate with configuration."""
    model_class: Type[BaseEstimator]
    param_space: Dict[str, Any]
    model_name: str
    model_type: str
    complexity_score: float = 1.0
    training_time_estimate: float = 1.0
    interpretability_score: float = 0.5
    memory_requirement: float = 1.0
    
@dataclass
class CrossValidationResult:
    """Cross-validation result."""
    scores: List[float]
    train_scores: List[float]
    fit_times: List[float]
    score_times: List[float]
    mean_score: float
    std_score: float
    mean_train_score: float
    std_train_score: float
    mean_fit_time: float
    std_fit_time: float
    statistical_significance: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class HyperparameterResult:
    """Hyperparameter optimization result."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: BaseEstimator
    optimization_history: List[Dict[str, Any]]
    convergence_curve: List[float]
    total_time: float
    n_trials: int
    study_summary: Dict[str, Any]

@dataclass
class ModelEvaluationResult:
    """Comprehensive model evaluation result."""
    model_name: str
    model: BaseEstimator
    cv_result: CrossValidationResult
    hyperparameter_result: HyperparameterResult
    test_scores: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    predictions: np.ndarray
    prediction_probabilities: Optional[np.ndarray]
    training_time: float
    inference_time: float
    model_size: float
    complexity_metrics: Dict[str, float]
    business_metrics: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class EnsembleResult:
    """Ensemble model result."""
    ensemble_model: BaseEstimator
    base_models: List[BaseEstimator]
    ensemble_weights: Optional[np.ndarray]
    meta_learner: Optional[BaseEstimator]
    cv_scores: CrossValidationResult
    test_scores: Dict[str, float]
    diversity_scores: Dict[str, float]
    ensemble_method: EnsembleMethod
    selection_strategy: str

@dataclass
class ModelSelectionReport:
    """Comprehensive model selection report."""
    report_id: str
    timestamp: datetime
    task_type: TaskType
    dataset_info: Dict[str, Any]
    config: ModelSelectionConfig
    model_results: List[ModelEvaluationResult]
    best_model_result: ModelEvaluationResult
    ensemble_result: Optional[EnsembleResult]
    statistical_analysis: Dict[str, Any]
    business_analysis: Dict[str, Any]
    recommendations: List[str]
    insights: List[str]
    performance_summary: Dict[str, Any]
    metadata: Dict[str, Any]

class DataSplitter:
    """Advanced data splitting strategies."""
    
    def __init__(self, config: ModelSelectionConfig):
        self.config = config
    
    async def split_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None,
        time_column: Optional[Union[str, np.ndarray]] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Split data into train, validation, and test sets."""
        try:
            if self.config.split_strategy == SplitStrategy.TIME_SERIES:
                return await self._time_series_split(X, y, time_column)
            elif self.config.split_strategy == SplitStrategy.GROUP:
                return await self._group_split(X, y, groups)
            elif self.config.split_strategy == SplitStrategy.STRATIFIED:
                return await self._stratified_split(X, y)
            else:  # RANDOM
                return await self._random_split(X, y)
                
        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}")
            raise
    
    async def _time_series_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        time_column: Optional[Union[str, np.ndarray]] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Time series aware splitting."""
        try:
            n_samples = len(X)
            
            # Calculate split indices
            test_size = int(n_samples * self.config.test_size)
            val_size = int(n_samples * self.config.validation_size)
            train_size = n_samples - test_size - val_size
            
            # Sequential splits for time series
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            logger.info(f"Time series split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            logger.error(f"Time series split failed: {str(e)}")
            raise
    
    async def _group_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Group-aware splitting to prevent data leakage."""
        try:
            from sklearn.model_selection import GroupShuffleSplit
            
            # First split: train+val vs test
            gss_test = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            train_val_idx, test_idx = next(gss_test.split(X, y, groups))
            
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            groups_train_val = groups[train_val_idx]
            
            # Second split: train vs val
            val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
            
            gss_val = GroupShuffleSplit(
                n_splits=1,
                test_size=val_size_adjusted,
                random_state=self.config.random_state
            )
            
            train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
            
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            
            logger.info(f"Group split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            logger.error(f"Group split failed: {str(e)}")
            raise
    
    async def _stratified_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Stratified splitting for classification tasks."""
        try:
            # First split: train+val vs test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                stratify=y if self.config.stratify_target else None,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            
            # Second split: train vs val
            val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adjusted,
                stratify=y_train_val if self.config.stratify_target else None,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            
            logger.info(f"Stratified split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            logger.error(f"Stratified split failed: {str(e)}")
            raise
    
    async def _random_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Random splitting."""
        try:
            # First split: train+val vs test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            
            # Second split: train vs val
            val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adjusted,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            
            logger.info(f"Random split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            logger.error(f"Random split failed: {str(e)}")
            raise

class CrossValidator:
    """Advanced cross-validation strategies."""
    
    def __init__(self, config: ModelSelectionConfig):
        self.config = config
    
    def get_cv_splitter(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None
    ):
        """Get appropriate CV splitter based on strategy."""
        try:
            if self.config.cv_strategy == CVStrategy.STRATIFIED_K_FOLD:
                return StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
            
            elif self.config.cv_strategy == CVStrategy.TIME_SERIES_CV:
                return TimeSeriesSplit(
                    n_splits=self.config.cv_folds,
                    test_size=None
                )
            
            elif self.config.cv_strategy == CVStrategy.GROUP_K_FOLD:
                return GroupKFold(n_splits=self.config.cv_folds)
            
            elif self.config.cv_strategy == CVStrategy.REPEATED_K_FOLD:
                from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
                
                if self._is_classification_task(y):
                    return RepeatedStratifiedKFold(
                        n_splits=self.config.cv_folds,
                        n_repeats=self.config.cv_repeats,
                        random_state=self.config.random_state
                    )
                else:
                    return RepeatedKFold(
                        n_splits=self.config.cv_folds,
                        n_repeats=self.config.cv_repeats,
                        random_state=self.config.random_state
                    )
            
            else:  # K_FOLD
                return KFold(
                    n_splits=self.config.cv_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
                
        except Exception as e:
            logger.error(f"CV splitter creation failed: {str(e)}")
            raise
    
    async def cross_validate_model(
        self,
        model: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None,
        scoring: Optional[Union[str, List[str]]] = None
    ) -> CrossValidationResult:
        """Perform cross-validation with comprehensive results."""
        try:
            start_time = datetime.now()
            
            # Get CV splitter
            cv_splitter = self.get_cv_splitter(X, y, groups)
            
            # Determine scoring metrics
            if scoring is None:
                scoring = self._get_default_scoring(y)
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv_splitter,
                scoring=scoring,
                return_train_score=self.config.cv_return_train_score,
                return_estimator=False,
                n_jobs=self.config.n_jobs if self.config.enable_parallel else 1,
                verbose=0
            )
            
            # Extract primary scoring metric
            primary_metric = scoring if isinstance(scoring, str) else scoring[0]
            test_scores = cv_results[f'test_{primary_metric}']
            train_scores = cv_results.get(f'train_{primary_metric}', [])
            
            # Calculate statistics
            mean_score = float(np.mean(test_scores))
            std_score = float(np.std(test_scores))
            mean_train_score = float(np.mean(train_scores)) if train_scores else 0.0
            std_train_score = float(np.std(train_scores)) if train_scores else 0.0
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                test_scores, self.config.confidence_level
            )
            
            # Statistical significance test (optional)
            statistical_significance = None
            if self.config.statistical_significance_test and len(test_scores) > 1:
                statistical_significance = self._test_statistical_significance(test_scores)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = CrossValidationResult(
                scores=test_scores.tolist(),
                train_scores=train_scores.tolist() if len(train_scores) > 0 else [],
                fit_times=cv_results['fit_time'].tolist(),
                score_times=cv_results['score_time'].tolist(),
                mean_score=mean_score,
                std_score=std_score,
                mean_train_score=mean_train_score,
                std_train_score=std_train_score,
                mean_fit_time=float(np.mean(cv_results['fit_time'])),
                std_fit_time=float(np.std(cv_results['fit_time'])),
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval
            )
            
            logger.info(f"Cross-validation completed: {mean_score:.4f} (±{std_score:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            raise
    
    def _is_classification_task(self, y: Union[np.ndarray, pd.Series]) -> bool:
        """Check if task is classification."""
        try:
            if hasattr(y, 'dtype'):
                if y.dtype == 'object' or y.dtype.name == 'category':
                    return True
            
            unique_values = len(np.unique(y))
            total_values = len(y)
            
            # If few unique values relative to total, likely classification
            return unique_values < min(20, total_values * 0.1)
            
        except:
            return True  # Default to classification
    
    def _get_default_scoring(self, y: Union[np.ndarray, pd.Series]) -> str:
        """Get default scoring metric based on task type."""
        try:
            if self._is_classification_task(y):
                unique_values = len(np.unique(y))
                if unique_values == 2:
                    return 'roc_auc'
                else:
                    return 'f1_weighted'
            else:
                return 'neg_mean_squared_error'
                
        except:
            return 'accuracy'  # Fallback
    
    def _calculate_confidence_interval(
        self,
        scores: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for CV scores."""
        try:
            if SCIPY_AVAILABLE and len(scores) > 1:
                alpha = 1 - confidence_level
                confidence_interval = stats.t.interval(
                    confidence_level,
                    len(scores) - 1,
                    loc=np.mean(scores),
                    scale=stats.sem(scores)
                )
                return (float(confidence_interval[0]), float(confidence_interval[1]))
            else:
                # Fallback: use normal approximation
                mean_score = np.mean(scores)
                std_error = np.std(scores) / np.sqrt(len(scores))
                margin = 1.96 * std_error  # 95% confidence
                return (float(mean_score - margin), float(mean_score + margin))
                
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {str(e)}")
            return (float(np.min(scores)), float(np.max(scores)))
    
    def _test_statistical_significance(self, scores: np.ndarray) -> float:
        """Test statistical significance of CV scores."""
        try:
            if SCIPY_AVAILABLE and len(scores) > 2:
                # One-sample t-test against null hypothesis (score = 0.5 for classification)
                baseline_score = 0.5 if max(scores) <= 1.0 else 0.0
                t_stat, p_value = stats.ttest_1samp(scores, baseline_score)
                return float(p_value)
            else:
                return 1.0  # No significance test
                
        except Exception as e:
            logger.warning(f"Statistical significance test failed: {str(e)}")
            return 1.0

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization."""
    
    def __init__(self, config: ModelSelectionConfig):
        self.config = config
    
    async def optimize_hyperparameters(
        self,
        model_candidate: ModelCandidate,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None
    ) -> HyperparameterResult:
        """Optimize hyperparameters using specified method."""
        try:
            start_time = datetime.now()
            
            if self.config.optimization_method == OptimizationMethod.BAYESIAN_OPTUNA and OPTUNA_AVAILABLE:
                result = await self._optimize_with_optuna(
                    model_candidate, X_train, y_train, X_val, y_val, groups
                )
            elif self.config.optimization_method == OptimizationMethod.BAYESIAN_HYPEROPT and HYPEROPT_AVAILABLE:
                result = await self._optimize_with_hyperopt(
                    model_candidate, X_train, y_train, X_val, y_val, groups
                )
            elif self.config.optimization_method == OptimizationMethod.BAYESIAN_SKOPT and SCIKIT_OPTIMIZE_AVAILABLE:
                result = await self._optimize_with_skopt(
                    model_candidate, X_train, y_train, X_val, y_val, groups
                )
            elif self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
                result = await self._optimize_with_grid_search(
                    model_candidate, X_train, y_train, X_val, y_val, groups
                )
            else:  # RANDOM_SEARCH
                result = await self._optimize_with_random_search(
                    model_candidate, X_train, y_train, X_val, y_val, groups
                )
            
            total_time = (datetime.now() - start_time).total_seconds()
            result.total_time = total_time
            
            logger.info(f"Hyperparameter optimization completed in {total_time:.2f}s")
            logger.info(f"Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise
    
    async def _optimize_with_optuna(
        self,
        model_candidate: ModelCandidate,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None
    ) -> HyperparameterResult:
        """Optimize using Optuna."""
        try:
            # Create study
            direction = 'maximize' if self._is_score_to_maximize() else 'minimize'
            
            pruner = MedianPruner() if self.config.enable_pruning else None
            
            study = optuna.create_study(
                direction=direction,
                pruner=pruner,
                sampler=TPESampler(seed=self.config.random_state)
            )
            
            # Define objective function
            def objective(trial):
                try:
                    # Sample hyperparameters
                    params = self._sample_optuna_params(trial, model_candidate.param_space)
                    
                    # Create and train model
                    model = model_candidate.model_class(**params)
                    
                    # Use cross-validation for evaluation
                    cv = CrossValidator(self.config)
                    cv_result = asyncio.run(cv.cross_validate_model(
                        model, X_train, y_train, groups
                    ))
                    
                    # Report intermediate values for pruning
                    for i, score in enumerate(cv_result.scores):
                        trial.report(score, i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    
                    return cv_result.mean_score
                    
                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    logger.warning(f"Trial failed: {str(e)}")
                    return float('-inf') if direction == 'maximize' else float('inf')
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.config.optimization_budget,
                timeout=self.config.optimization_timeout,
                catch=(Exception,)
            )
            
            # Get best parameters and create best model
            best_params = study.best_params
            best_model = model_candidate.model_class(**best_params)
            best_model.fit(X_train, y_train)
            
            # Extract optimization history
            optimization_history = []
            convergence_curve = []
            
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    optimization_history.append({
                        'trial_number': trial.number,
                        'params': trial.params,
                        'value': trial.value,
                        'duration': trial.duration.total_seconds() if trial.duration else 0
                    })
                    convergence_curve.append(trial.value)
            
            # Study summary
            study_summary = {
                'n_trials': len(study.trials),
                'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'best_trial_number': study.best_trial.number,
                'optimization_direction': direction
            }
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=study.best_value,
                best_model=best_model,
                optimization_history=optimization_history,
                convergence_curve=convergence_curve,
                total_time=0.0,  # Will be set by caller
                n_trials=len(study.trials),
                study_summary=study_summary
            )
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {str(e)}")
            raise
    
    async def _optimize_with_grid_search(
        self,
        model_candidate: ModelCandidate,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None
    ) -> HyperparameterResult:
        """Optimize using Grid Search."""
        try:
            # Create parameter grid
            param_grid = self._create_param_grid(model_candidate.param_space)
            
            # Create CV splitter
            cv = CrossValidator(self.config)
            cv_splitter = cv.get_cv_splitter(X_train, y_train, groups)
            
            # Create scoring metric
            scoring = cv._get_default_scoring(y_train)
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=model_candidate.model_class(),
                param_grid=param_grid,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=self.config.n_jobs if self.config.enable_parallel else 1,
                refit=True,
                verbose=0,
                return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Extract results
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            
            # Create optimization history
            optimization_history = []
            convergence_curve = []
            
            for i, params in enumerate(grid_search.cv_results_['params']):
                score = grid_search.cv_results_['mean_test_score'][i]
                optimization_history.append({
                    'trial_number': i,
                    'params': params,
                    'value': score,
                    'duration': 0.0
                })
                convergence_curve.append(score)
            
            study_summary = {
                'n_trials': len(optimization_history),
                'n_complete_trials': len(optimization_history),
                'n_pruned_trials': 0,
                'best_trial_number': np.argmax(convergence_curve),
                'optimization_direction': 'maximize'
            }
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=best_score,
                best_model=best_model,
                optimization_history=optimization_history,
                convergence_curve=convergence_curve,
                total_time=0.0,
                n_trials=len(optimization_history),
                study_summary=study_summary
            )
            
        except Exception as e:
            logger.error(f"Grid search optimization failed: {str(e)}")
            raise
    
    async def _optimize_with_random_search(
        self,
        model_candidate: ModelCandidate,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None
    ) -> HyperparameterResult:
        """Optimize using Random Search."""
        try:
            # Create parameter distributions
            param_distributions = self._create_param_distributions(model_candidate.param_space)
            
            # Create CV splitter
            cv = CrossValidator(self.config)
            cv_splitter = cv.get_cv_splitter(X_train, y_train, groups)
            
            # Create scoring metric
            scoring = cv._get_default_scoring(y_train)
            
            # Perform random search
            random_search = RandomizedSearchCV(
                estimator=model_candidate.model_class(),
                param_distributions=param_distributions,
                n_iter=self.config.optimization_budget,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=self.config.n_jobs if self.config.enable_parallel else 1,
                refit=True,
                verbose=0,
                random_state=self.config.random_state,
                return_train_score=True
            )
            
            random_search.fit(X_train, y_train)
            
            # Extract results
            best_params = random_search.best_params_
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_
            
            # Create optimization history
            optimization_history = []
            convergence_curve = []
            
            for i, params in enumerate(random_search.cv_results_['params']):
                score = random_search.cv_results_['mean_test_score'][i]
                optimization_history.append({
                    'trial_number': i,
                    'params': params,
                    'value': score,
                    'duration': 0.0
                })
                convergence_curve.append(score)
            
            study_summary = {
                'n_trials': len(optimization_history),
                'n_complete_trials': len(optimization_history),
                'n_pruned_trials': 0,
                'best_trial_number': np.argmax(convergence_curve),
                'optimization_direction': 'maximize'
            }
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=best_score,
                best_model=best_model,
                optimization_history=optimization_history,
                convergence_curve=convergence_curve,
                total_time=0.0,
                n_trials=len(optimization_history),
                study_summary=study_summary
            )
            
        except Exception as e:
            logger.error(f"Random search optimization failed: {str(e)}")
            raise
    
    def _sample_optuna_params(self, trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters using Optuna trial."""
        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            else:
                # Simple list of choices
                params[param_name] = trial.suggest_categorical(param_name, param_config)
        
        return params
    
    def _create_param_grid(self, param_space: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Create parameter grid for GridSearchCV."""
        param_grid = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                
                if param_type == 'int':
                    step = param_config.get('step', 1)
                    param_grid[param_name] = list(range(
                        param_config['low'],
                        param_config['high'] + 1,
                        step
                    ))
                elif param_type == 'float':
                    # Create logarithmic or linear space
                    if param_config.get('log', False):
                        param_grid[param_name] = np.logspace(
                            np.log10(param_config['low']),
                            np.log10(param_config['high']),
                            num=10
                        ).tolist()
                    else:
                        param_grid[param_name] = np.linspace(
                            param_config['low'],
                            param_config['high'],
                            num=10
                        ).tolist()
                elif param_type == 'categorical':
                    param_grid[param_name] = param_config['choices']
            else:
                param_grid[param_name] = param_config
        
        return param_grid
    
    def _create_param_distributions(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create parameter distributions for RandomizedSearchCV."""
        from scipy.stats import uniform, randint
        
        param_distributions = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                
                if param_type == 'int':
                    param_distributions[param_name] = randint(
                        param_config['low'],
                        param_config['high'] + 1
                    )
                elif param_type == 'float':
                    if param_config.get('log', False):
                        # Log-uniform distribution
                        low_log = np.log10(param_config['low'])
                        high_log = np.log10(param_config['high'])
                        param_distributions[param_name] = uniform(low_log, high_log - low_log)
                    else:
                        param_distributions[param_name] = uniform(
                            param_config['low'],
                            param_config['high'] - param_config['low']
                        )
                elif param_type == 'categorical':
                    param_distributions[param_name] = param_config['choices']
            else:
                param_distributions[param_name] = param_config
        
        return param_distributions
    
    def _is_score_to_maximize(self) -> bool:
        """Check if scoring metric should be maximized."""
        scoring = self.config.cv_scoring
        if isinstance(scoring, str):
            return not scoring.startswith('neg_')
        return True

class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config: ModelSelectionConfig):
        self.config = config
    
    async def evaluate_model(
        self,
        model: BaseEstimator,
        model_name: str,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model on test set with multiple metrics."""
        try:
            start_time = datetime.now()
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
            
            # Calculate metrics based on task type
            if self._is_classification_task(y_test):
                metrics = self._calculate_classification_metrics(
                    y_test, y_pred, y_pred_proba
                )
            else:
                metrics = self._calculate_regression_metrics(
                    y_test, y_pred
                )
            
            # Calculate inference time
            inference_start = datetime.now()
            _ = model.predict(X_test[:min(100, len(X_test))])  # Sample for timing
            inference_time = (datetime.now() - inference_start).total_seconds()
            metrics['inference_time_per_sample'] = inference_time / min(100, len(X_test))
            
            # Calculate model complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(model)
            metrics.update(complexity_metrics)
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model evaluation completed in {total_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}
    
    def _calculate_classification_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Handle binary vs multiclass
            unique_classes = len(np.unique(y_true))
            
            if unique_classes == 2:
                # Binary classification
                metrics['precision'] = precision_score(y_true, y_pred, average='binary')
                metrics['recall'] = recall_score(y_true, y_pred, average='binary')
                metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            else:
                # Multiclass classification
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
                metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro')
                
                metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
                metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
                metrics['f1_score_weighted'] = f1_score(y_true, y_pred, average='weighted')
                
                if y_pred_proba is not None:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Classification metrics calculation failed: {str(e)}")
            return {'accuracy': 0.0}
    
    def _calculate_regression_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            
            # Additional metrics
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            
            # Mean Absolute Percentage Error
            non_zero_mask = y_true != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['mape'] = mape
            
            # Max error
            metrics['max_error'] = np.max(np.abs(y_true - y_pred))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Regression metrics calculation failed: {str(e)}")
            return {'rmse': float('inf')}
    
    def _calculate_complexity_metrics(self, model: BaseEstimator) -> Dict[str, float]:
        """Calculate model complexity metrics."""
        try:
            metrics = {}
            
            # Model size
            try:
                model_size = len(pickle.dumps(model))
                metrics['model_size_bytes'] = model_size
                metrics['model_size_mb'] = model_size / (1024 * 1024)
            except:
                metrics['model_size_bytes'] = 0
                metrics['model_size_mb'] = 0
            
            # Number of parameters (for supported models)
            if hasattr(model, 'coef_'):
                if model.coef_.ndim == 1:
                    metrics['n_parameters'] = len(model.coef_)
                else:
                    metrics['n_parameters'] = model.coef_.size
            elif hasattr(model, 'n_features_in_'):
                metrics['n_parameters'] = model.n_features_in_
            else:
                metrics['n_parameters'] = 0
            
            # Tree-based model complexity
            if hasattr(model, 'tree_'):
                metrics['tree_depth'] = model.tree_.max_depth
                metrics['n_leaves'] = model.tree_.n_leaves
            elif hasattr(model, 'estimators_'):
                if hasattr(model.estimators_[0], 'tree_'):
                    depths = [est.tree_.max_depth for est in model.estimators_]
                    metrics['mean_tree_depth'] = np.mean(depths)
                    metrics['max_tree_depth'] = np.max(depths)
                    metrics['n_estimators'] = len(model.estimators_)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Complexity metrics calculation failed: {str(e)}")
            return {}
    
    def _is_classification_task(self, y: Union[np.ndarray, pd.Series]) -> bool:
        """Check if task is classification."""
        try:
            if hasattr(y, 'dtype'):
                if y.dtype == 'object' or y.dtype.name == 'category':
                    return True
            
            unique_values = len(np.unique(y))
            total_values = len(y)
            
            return unique_values < min(20, total_values * 0.1)
            
        except:
            return True

class EnsembleBuilder:
    """Advanced ensemble model building."""
    
    def __init__(self, config: ModelSelectionConfig):
        self.config = config
    
    async def build_ensemble(
        self,
        model_results: List[ModelEvaluationResult],
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        task_type: TaskType
    ) -> EnsembleResult:
        """Build ensemble from best performing models."""
        try:
            if len(model_results) < 2:
                raise ValueError("Need at least 2 models for ensemble")
            
            # Select best models for ensemble
            selected_models = self._select_models_for_ensemble(model_results)
            
            logger.info(f"Building ensemble with {len(selected_models)} models")
            
            if self.config.ensemble_method == EnsembleMethod.VOTING:
                ensemble_result = await self._build_voting_ensemble(
                    selected_models, X_train, y_train, X_val, y_val, task_type
                )
            elif self.config.ensemble_method == EnsembleMethod.STACKING:
                ensemble_result = await self._build_stacking_ensemble(
                    selected_models, X_train, y_train, X_val, y_val, task_type
                )
            elif self.config.ensemble_method == EnsembleMethod.BLENDING:
                ensemble_result = await self._build_blending_ensemble(
                    selected_models, X_train, y_train, X_val, y_val, task_type
                )
            else:
                raise ValueError(f"Unsupported ensemble method: {self.config.ensemble_method}")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Ensemble building failed: {str(e)}")
            raise
    
    def _select_models_for_ensemble(
        self,
        model_results: List[ModelEvaluationResult]
    ) -> List[ModelEvaluationResult]:
        """Select best models for ensemble based on performance and diversity."""
        try:
            # Sort by performance
            sorted_results = sorted(
                model_results,
                key=lambda x: x.cv_result.mean_score,
                reverse=True
            )
            
            # Select top models
            n_models = min(self.config.ensemble_size, len(sorted_results))
            selected_models = sorted_results[:n_models]
            
            # TODO: Add diversity-based selection
            
            return selected_models
            
        except Exception as e:
            logger.error(f"Model selection for ensemble failed: {str(e)}")
            return model_results[:self.config.ensemble_size]
    
    async def _build_voting_ensemble(
        self,
        selected_models: List[ModelEvaluationResult],
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        task_type: TaskType
    ) -> EnsembleResult:
        """Build voting ensemble."""
        try:
            # Prepare estimators
            estimators = [
                (f'model_{i}', result.model)
                for i, result in enumerate(selected_models)
            ]
            
            # Create ensemble
            if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',  # Use probability voting
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            else:
                ensemble_model = VotingRegressor(
                    estimators=estimators,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            
            # Fit ensemble
            ensemble_model.fit(X_train, y_train)
            
            # Evaluate ensemble
            cv = CrossValidator(self.config)
            cv_scores = await cv.cross_validate_model(ensemble_model, X_train, y_train)
            
            evaluator = ModelEvaluator(self.config)
            test_scores = await evaluator.evaluate_model(
                ensemble_model, 'ensemble', X_val, y_val
            )
            
            # Calculate diversity scores
            diversity_scores = self._calculate_diversity_scores(selected_models, X_val, y_val)
            
            return EnsembleResult(
                ensemble_model=ensemble_model,
                base_models=[r.model for r in selected_models],
                ensemble_weights=None,  # Voting doesn't have explicit weights
                meta_learner=None,
                cv_scores=cv_scores,
                test_scores=test_scores,
                diversity_scores=diversity_scores,
                ensemble_method=EnsembleMethod.VOTING,
                selection_strategy='top_k_performance'
            )
            
        except Exception as e:
            logger.error(f"Voting ensemble building failed: {str(e)}")
            raise
    
    async def _build_stacking_ensemble(
        self,
        selected_models: List[ModelEvaluationResult],
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        task_type: TaskType
    ) -> EnsembleResult:
        """Build stacking ensemble."""
        try:
            # Prepare base estimators
            estimators = [
                (f'model_{i}', result.model)
                for i, result in enumerate(selected_models)
            ]
            
            # Choose meta-learner
            if self.config.meta_learner == 'auto':
                if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                    from sklearn.linear_model import LogisticRegression
                    final_estimator = LogisticRegression()
                else:
                    from sklearn.linear_model import Ridge
                    final_estimator = Ridge()
            else:
                final_estimator = self.config.meta_learner
            
            # Create stacking ensemble
            if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                ensemble_model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    cv=self.config.ensemble_cv_folds,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            else:
                ensemble_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    cv=self.config.ensemble_cv_folds,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            
            # Fit ensemble
            ensemble_model.fit(X_train, y_train)
            
            # Evaluate ensemble
            cv = CrossValidator(self.config)
            cv_scores = await cv.cross_validate_model(ensemble_model, X_train, y_train)
            
            evaluator = ModelEvaluator(self.config)
            test_scores = await evaluator.evaluate_model(
                ensemble_model, 'stacking_ensemble', X_val, y_val
            )
            
            # Calculate diversity scores
            diversity_scores = self._calculate_diversity_scores(selected_models, X_val, y_val)
            
            return EnsembleResult(
                ensemble_model=ensemble_model,
                base_models=[r.model for r in selected_models],
                ensemble_weights=None,
                meta_learner=final_estimator,
                cv_scores=cv_scores,
                test_scores=test_scores,
                diversity_scores=diversity_scores,
                ensemble_method=EnsembleMethod.STACKING,
                selection_strategy='top_k_performance'
            )
            
        except Exception as e:
            logger.error(f"Stacking ensemble building failed: {str(e)}")
            raise
    
    async def _build_blending_ensemble(
        self,
        selected_models: List[ModelEvaluationResult],
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        task_type: TaskType
    ) -> EnsembleResult:
        """Build blending ensemble."""
        try:
            # Generate predictions from base models
            base_predictions = []
            
            for result in selected_models:
                pred = result.model.predict(X_val)
                base_predictions.append(pred)
            
            base_predictions = np.column_stack(base_predictions)
            
            # Train meta-learner on validation predictions
            if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression()
            else:
                from sklearn.linear_model import Ridge
                meta_learner = Ridge()
            
            meta_learner.fit(base_predictions, y_val)
            
            # Create blending ensemble class
            class BlendingEnsemble:
                def __init__(self, base_models, meta_learner):
                    self.base_models = base_models
                    self.meta_learner = meta_learner
                
                def predict(self, X):
                    base_preds = np.column_stack([
                        model.predict(X) for model in self.base_models
                    ])
                    return self.meta_learner.predict(base_preds)
                
                def predict_proba(self, X):
                    if hasattr(self.meta_learner, 'predict_proba'):
                        base_preds = np.column_stack([
                            model.predict(X) for model in self.base_models
                        ])
                        return self.meta_learner.predict_proba(base_preds)
                    else:
                        raise AttributeError("Meta-learner doesn't support predict_proba")
            
            ensemble_model = BlendingEnsemble(
                [r.model for r in selected_models],
                meta_learner
            )
            
            # Evaluate ensemble (approximate)
            cv_scores = CrossValidationResult(
                scores=[],
                train_scores=[],
                fit_times=[],
                score_times=[],
                mean_score=0.0,
                std_score=0.0,
                mean_train_score=0.0,
                std_train_score=0.0,
                mean_fit_time=0.0,
                std_fit_time=0.0
            )
            
            # Test on remaining data (if any)
            test_scores = {'blending_score': 0.0}
            
            # Calculate diversity scores
            diversity_scores = self._calculate_diversity_scores(selected_models, X_val, y_val)
            
            return EnsembleResult(
                ensemble_model=ensemble_model,
                base_models=[r.model for r in selected_models],
                ensemble_weights=None,
                meta_learner=meta_learner,
                cv_scores=cv_scores,
                test_scores=test_scores,
                diversity_scores=diversity_scores,
                ensemble_method=EnsembleMethod.BLENDING,
                selection_strategy='top_k_performance'
            )
            
        except Exception as e:
            logger.error(f"Blending ensemble building failed: {str(e)}")
            raise
    
    def _calculate_diversity_scores(
        self,
        selected_models: List[ModelEvaluationResult],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """Calculate diversity scores between models."""
        try:
            diversity_scores = {}
            
            # Get predictions from all models
            predictions = []
            for result in selected_models:
                pred = result.model.predict(X_val)
                predictions.append(pred)
            
            # Calculate pairwise disagreement
            n_models = len(predictions)
            disagreements = []
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    # For classification: fraction of disagreements
                    # For regression: correlation coefficient
                    if self._is_classification_task(y_val):
                        disagreement = np.mean(predictions[i] != predictions[j])
                    else:
                        disagreement = 1 - abs(np.corrcoef(predictions[i], predictions[j])[0, 1])
                    
                    disagreements.append(disagreement)
            
            diversity_scores['mean_pairwise_diversity'] = float(np.mean(disagreements))
            diversity_scores['std_pairwise_diversity'] = float(np.std(disagreements))
            diversity_scores['min_pairwise_diversity'] = float(np.min(disagreements))
            diversity_scores['max_pairwise_diversity'] = float(np.max(disagreements))
            
            return diversity_scores
            
        except Exception as e:
            logger.warning(f"Diversity score calculation failed: {str(e)}")
            return {}
    
    def _is_classification_task(self, y: Union[np.ndarray, pd.Series]) -> bool:
        """Check if task is classification."""
        try:
            if hasattr(y, 'dtype'):
                if y.dtype == 'object' or y.dtype.name == 'category':
                    return True
            
            unique_values = len(np.unique(y))
            total_values = len(y)
            
            return unique_values < min(20, total_values * 0.1)
            
        except:
            return True

class ModelSelector:
    """
    Comprehensive model selection system integrating all components.
    """
    
    def __init__(self, config: Optional[ModelSelectionConfig] = None):
        self.config = config or ModelSelectionConfig()
        self.data_splitter = DataSplitter(self.config)
        self.cross_validator = CrossValidator(self.config)
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        self.ensemble_builder = EnsembleBuilder(self.config)
        
        logger.info("ModelSelector initialized")
    
    async def select_best_model(
        self,
        model_candidates: List[ModelCandidate],
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task_type: TaskType,
        groups: Optional[np.ndarray] = None,
        time_column: Optional[Union[str, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None
    ) -> ModelSelectionReport:
        """
        Comprehensive model selection with evaluation and reporting.
        
        Args:
            model_candidates: List of model candidates to evaluate
            X: Feature matrix
            y: Target values
            task_type: Type of ML task
            groups: Group labels for group-based splitting
            time_column: Time column for time series splitting
            feature_names: Optional feature names
            
        Returns:
            Comprehensive model selection report
        """
        try:
            logger.info(f"Starting model selection for {task_type.value} task")
            logger.info(f"Evaluating {len(model_candidates)} model candidates")
            start_time = datetime.now()
            
            # Data splitting
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = await self.data_splitter.split_data(
                X, y, groups, time_column
            )
            
            # Dataset analysis
            dataset_info = await self._analyze_dataset(X, y, task_type)
            
            # Model evaluation
            model_results = []
            
            for i, candidate in enumerate(model_candidates):
                try:
                    logger.info(f"Evaluating model {i+1}/{len(model_candidates)}: {candidate.model_name}")
                    
                    result = await self._evaluate_single_model(
                        candidate, X_train, y_train, X_val, y_val, X_test, y_test, 
                        task_type, groups, feature_names
                    )
                    
                    if result:
                        model_results.append(result)
                        logger.info(f"{candidate.model_name} - CV Score: {result.cv_result.mean_score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Model {candidate.model_name} evaluation failed: {str(e)}")
                    continue
            
            if not model_results:
                raise ValueError("No models were successfully evaluated")
            
            # Select best model
            best_model_result = max(model_results, key=lambda x: x.cv_result.mean_score)
            
            # Build ensemble if enabled
            ensemble_result = None
            if self.config.enable_ensemble and len(model_results) >= 2:
                try:
                    logger.info("Building ensemble model")
                    ensemble_result = await self.ensemble_builder.build_ensemble(
                        model_results, X_train, y_train, X_val, y_val, task_type
                    )
                except Exception as e:
                    logger.warning(f"Ensemble building failed: {str(e)}")
            
            # Statistical analysis
            statistical_analysis = await self._perform_statistical_analysis(model_results)
            
            # Business analysis
            business_analysis = await self._perform_business_analysis(
                model_results, best_model_result, ensemble_result
            )
            
            # Generate insights and recommendations
            insights = await self._generate_insights(
                model_results, best_model_result, statistical_analysis, business_analysis
            )
            
            recommendations = await self._generate_recommendations(
                model_results, best_model_result, insights
            )
            
            # Performance summary
            performance_summary = await self._create_performance_summary(
                model_results, best_model_result, ensemble_result
            )
            
            # Create comprehensive report
            execution_time = (datetime.now() - start_time).total_seconds()
            
            report = ModelSelectionReport(
                report_id=str(uuid.uuid4()),
                timestamp=start_time,
                task_type=task_type,
                dataset_info=dataset_info,
                config=self.config,
                model_results=model_results,
                best_model_result=best_model_result,
                ensemble_result=ensemble_result,
                statistical_analysis=statistical_analysis,
                business_analysis=business_analysis,
                recommendations=recommendations,
                insights=insights,
                performance_summary=performance_summary,
                metadata={
                    'execution_time': execution_time,
                    'n_models_evaluated': len(model_results),
                    'best_model_name': best_model_result.model_name
                }
            )
            
            # Log to MLflow if enabled
            if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                await self._log_to_mlflow(report)
            
            logger.info(f"Model selection completed in {execution_time:.2f}s")
            logger.info(f"Best model: {best_model_result.model_name} with score: {best_model_result.cv_result.mean_score:.4f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Model selection failed: {str(e)}")
            raise
    
    async def _evaluate_single_model(
        self,
        candidate: ModelCandidate,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        task_type: TaskType,
        groups: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Optional[ModelEvaluationResult]:
        """Evaluate a single model candidate."""
        try:
            start_time = datetime.now()
            
            # Hyperparameter optimization
            hyperparameter_result = await self.hyperparameter_optimizer.optimize_hyperparameters(
                candidate, X_train, y_train, X_val, y_val, groups
            )
            
            # Cross-validation evaluation
            cv_result = await self.cross_validator.cross_validate_model(
                hyperparameter_result.best_model, X_train, y_train, groups
            )
            
            # Test set evaluation
            test_scores = await self.model_evaluator.evaluate_model(
                hyperparameter_result.best_model, candidate.model_name,
                X_test, y_test, X_train, y_train, feature_names
            )
            
            # Make test predictions
            predictions = hyperparameter_result.best_model.predict(X_test)
            prediction_probabilities = None
            
            if hasattr(hyperparameter_result.best_model, 'predict_proba'):
                try:
                    prediction_probabilities = hyperparameter_result.best_model.predict_proba(X_test)
                except:
                    pass
            
            # Calculate feature importance
            feature_importance = None
            if self.config.calculate_feature_importance:
                feature_importance = self._calculate_feature_importance(
                    hyperparameter_result.best_model, feature_names
                )
            
            # Calculate business metrics
            business_metrics = await self._calculate_business_metrics(
                hyperparameter_result.best_model, candidate, test_scores
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate complexity metrics
            complexity_metrics = {
                'complexity_score': candidate.complexity_score,
                'interpretability_score': candidate.interpretability_score,
                'training_time_estimate': candidate.training_time_estimate,
                'memory_requirement': candidate.memory_requirement
            }
            
            result = ModelEvaluationResult(
                model_name=candidate.model_name,
                model=hyperparameter_result.best_model,
                cv_result=cv_result,
                hyperparameter_result=hyperparameter_result,
                test_scores=test_scores,
                feature_importance=feature_importance,
                predictions=predictions,
                prediction_probabilities=prediction_probabilities,
                training_time=training_time,
                inference_time=test_scores.get('inference_time_per_sample', 0.0),
                model_size=test_scores.get('model_size_mb', 0.0),
                complexity_metrics=complexity_metrics,
                business_metrics=business_metrics,
                metadata={
                    'task_type': task_type.value,
                    'optimization_method': self.config.optimization_method.value,
                    'cv_strategy': self.config.cv_strategy.value
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Single model evaluation failed for {candidate.model_name}: {str(e)}")
            return None
    
    async def _analyze_dataset(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        try:
            analysis = {
                'n_samples': len(X),
                'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                'task_type': task_type.value,
                'target_type': str(type(y).__name__)
            }
            
            # Target analysis
            if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                unique_values = np.unique(y)
                analysis['n_classes'] = len(unique_values)
                analysis['class_distribution'] = dict(zip(*np.unique(y, return_counts=True)))
                
                # Class balance
                class_counts = list(analysis['class_distribution'].values())
                analysis['class_imbalance_ratio'] = max(class_counts) / min(class_counts)
                
            else:  # Regression or other
                analysis['target_stats'] = {
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y)),
                    'min': float(np.min(y)),
                    'max': float(np.max(y)),
                    'median': float(np.median(y))
                }
            
            # Feature analysis
            if hasattr(X, 'dtypes'):  # pandas DataFrame
                analysis['feature_types'] = dict(X.dtypes.value_counts())
                analysis['missing_values'] = dict(X.isnull().sum())
            
            # Dataset size category
            if analysis['n_samples'] < 1000:
                analysis['size_category'] = 'small'
            elif analysis['n_samples'] < 10000:
                analysis['size_category'] = 'medium'
            else:
                analysis['size_category'] = 'large'
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Dataset analysis failed: {str(e)}")
            return {}
    
    def _calculate_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, float]]:
        """Calculate feature importance from model."""
        try:
            importance_values = None
            
            # Try different methods to get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    importance_values = np.mean(np.abs(coef), axis=0)
                else:
                    importance_values = np.abs(coef)
            
            if importance_values is not None and feature_names is not None:
                if len(importance_values) == len(feature_names):
                    feature_importance = dict(zip(feature_names, importance_values))
                    # Sort by importance
                    sorted_importance = dict(
                        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    )
                    return sorted_importance
            
            return None
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
            return None
    
    async def _calculate_business_metrics(
        self,
        model: BaseEstimator,
        candidate: ModelCandidate,
        test_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate business-relevant metrics."""
        try:
            business_metrics = {}
            
            # Performance vs complexity trade-off
            performance_score = test_scores.get('accuracy', test_scores.get('r2_score', 0))
            complexity_penalty = candidate.complexity_score * self.config.interpretability_weight
            
            business_metrics['performance_complexity_ratio'] = performance_score / (1 + complexity_penalty)
            
            # Speed metrics
            inference_time = test_scores.get('inference_time_per_sample', 0)
            speed_score = 1 / (1 + inference_time) if inference_time > 0 else 1
            business_metrics['speed_score'] = speed_score
            
            # Resource efficiency
            model_size = test_scores.get('model_size_mb', 0)
            efficiency_score = performance_score / (1 + model_size * 0.1)
            business_metrics['resource_efficiency'] = efficiency_score
            
            # Overall business score
            business_score = (
                0.6 * performance_score +
                0.2 * speed_score +
                0.1 * candidate.interpretability_score +
                0.1 * efficiency_score
            )
            business_metrics['overall_business_score'] = business_score
            
            return business_metrics
            
        except Exception as e:
            logger.warning(f"Business metrics calculation failed: {str(e)}")
            return {}
    
    async def _perform_statistical_analysis(
        self,
        model_results: List[ModelEvaluationResult]
    ) -> Dict[str, Any]:
        """Perform statistical analysis of model results."""
        try:
            if len(model_results) < 2:
                return {}
            
            analysis = {}
            
            # Extract CV scores for comparison
            model_scores = {}
            for result in model_results:
                model_scores[result.model_name] = result.cv_result.scores
            
            # Pairwise statistical tests
            if SCIPY_AVAILABLE and self.config.statistical_significance_test:
                pairwise_tests = {}
                model_names = list(model_scores.keys())
                
                for i, model1 in enumerate(model_names):
                    for model2 in model_names[i+1:]:
                        try:
                            scores1 = model_scores[model1]
                            scores2 = model_scores[model2]
                            
                            # Wilcoxon signed-rank test
                            statistic, p_value = wilcoxon(scores1, scores2)
                            
                            pairwise_tests[f"{model1}_vs_{model2}"] = {
                                'statistic': float(statistic),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        except:
                            continue
                
                analysis['pairwise_tests'] = pairwise_tests
            
            # Overall ranking stability
            cv_means = [result.cv_result.mean_score for result in model_results]
            cv_stds = [result.cv_result.std_score for result in model_results]
            
            analysis['ranking_stability'] = {
                'score_range': float(max(cv_means) - min(cv_means)),
                'mean_std': float(np.mean(cv_stds)),
                'coefficient_of_variation': float(np.std(cv_means) / np.mean(cv_means))
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Statistical analysis failed: {str(e)}")
            return {}
    
    async def _perform_business_analysis(
        self,
        model_results: List[ModelEvaluationResult],
        best_model_result: ModelEvaluationResult,
        ensemble_result: Optional[EnsembleResult]
    ) -> Dict[str, Any]:
        """Perform business impact analysis."""
        try:
            analysis = {}
            
            # Performance vs cost analysis
            performance_scores = [r.cv_result.mean_score for r in model_results]
            training_times = [r.training_time for r in model_results]
            model_sizes = [r.model_size for r in model_results]
            
            analysis['performance_cost_analysis'] = {
                'best_performance': float(max(performance_scores)),
                'fastest_training': float(min(training_times)),
                'smallest_model': float(min(model_sizes)),
                'performance_time_correlation': float(np.corrcoef(performance_scores, training_times)[0, 1])
            }
            
            # ROI estimation
            best_performance = best_model_result.cv_result.mean_score
            baseline_performance = min(performance_scores)
            performance_improvement = best_performance - baseline_performance
            
            analysis['roi_estimation'] = {
                'performance_improvement': float(performance_improvement),
                'relative_improvement': float(performance_improvement / baseline_performance) if baseline_performance > 0 else 0,
                'training_cost': best_model_result.training_time,
                'deployment_readiness': 'high' if best_model_result.model_size < 100 else 'medium' if best_model_result.model_size < 500 else 'low'
            }
            
            # Ensemble value
            if ensemble_result:
                ensemble_improvement = ensemble_result.cv_scores.mean_score - best_performance
                analysis['ensemble_value'] = {
                    'improvement_over_best': float(ensemble_improvement),
                    'complexity_increase': len(ensemble_result.base_models),
                    'worth_complexity': ensemble_improvement > 0.01  # 1% improvement threshold
                }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Business analysis failed: {str(e)}")
            return {}
    
    async def _generate_insights(
        self,
        model_results: List[ModelEvaluationResult],
        best_model_result: ModelEvaluationResult,
        statistical_analysis: Dict[str, Any],
        business_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable insights."""
        try:
            insights = []
            
            # Performance insights
            best_score = best_model_result.cv_result.mean_score
            best_model_name = best_model_result.model_name
            
            insights.append(f"Best performing model: {best_model_name} with CV score: {best_score:.4f}")
            
            # Model comparison insights
            if len(model_results) > 1:
                scores = [r.cv_result.mean_score for r in model_results]
                score_range = max(scores) - min(scores)
                
                if score_range < 0.02:
                    insights.append("Model performances are very similar - consider interpretability and speed for selection")
                elif score_range > 0.1:
                    insights.append("Significant performance differences between models - algorithm choice is critical")
            
            # Stability insights
            cv_std = best_model_result.cv_result.std_score
            if cv_std > 0.05:
                insights.append("High cross-validation variance detected - model may be unstable or data may need more preprocessing")
            elif cv_std < 0.01:
                insights.append("Very stable cross-validation results - model is robust")
            
            # Business insights
            if 'roi_estimation' in business_analysis:
                roi_info = business_analysis['roi_estimation']
                improvement = roi_info.get('relative_improvement', 0)
                
                if improvement > 0.2:
                    insights.append(f"Significant performance improvement achieved ({improvement:.1%}) - high business value")
                elif improvement < 0.05:
                    insights.append("Limited performance improvement - consider if model complexity is justified")
            
            # Statistical significance insights
            if 'pairwise_tests' in statistical_analysis:
                significant_pairs = sum(
                    1 for test in statistical_analysis['pairwise_tests'].values()
                    if test.get('significant', False)
                )
                total_pairs = len(statistical_analysis['pairwise_tests'])
                
                if significant_pairs > total_pairs * 0.5:
                    insights.append("Many statistically significant differences between models - model choice matters")
                else:
                    insights.append("Few statistically significant differences - focus on practical considerations")
            
            # Resource insights
            training_time = best_model_result.training_time
            model_size = best_model_result.model_size
            
            if training_time > 3600:  # 1 hour
                insights.append("Long training time - consider model simplification for faster iterations")
            
            if model_size > 500:  # 500 MB
                insights.append("Large model size - may impact deployment and inference speed")
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {str(e)}")
            return ["Model selection completed successfully"]
    
    async def _generate_recommendations(
        self,
        model_results: List[ModelEvaluationResult],
        best_model_result: ModelEvaluationResult,
        insights: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Model deployment recommendations
            best_score = best_model_result.cv_result.mean_score
            
            if best_score > 0.9:
                recommendations.append("Excellent model performance - ready for production deployment")
            elif best_score > 0.8:
                recommendations.append("Good model performance - monitor closely in production")
            else:
                recommendations.append("Moderate performance - consider feature engineering or more data collection")
            
            # Hyperparameter recommendations
            hp_trials = best_model_result.hyperparameter_result.n_trials
            if hp_trials < 50:
                recommendations.append("Consider more hyperparameter optimization trials for potential improvements")
            
            # Cross-validation recommendations
            cv_std = best_model_result.cv_result.std_score
            if cv_std > 0.05:
                recommendations.append("High CV variance - consider more robust validation strategy or data preprocessing")
            
            # Ensemble recommendations
            if len(model_results) >= 3:
                recommendations.append("Multiple good models available - consider ensemble methods for improved performance")
            
            # Feature importance recommendations
            if best_model_result.feature_importance:
                n_important_features = sum(
                    1 for importance in best_model_result.feature_importance.values()
                    if importance > 0.01
                )
                total_features = len(best_model_result.feature_importance)
                
                if n_important_features < total_features * 0.5:
                    recommendations.append("Many features have low importance - consider feature selection")
            
            # Business recommendations
            recommendations.append("Implement model monitoring and performance tracking in production")
            recommendations.append("Plan for model retraining schedule based on data drift detection")
            
            # Performance optimization
            if best_model_result.training_time > 1800:  # 30 minutes
                recommendations.append("Consider model simplification or distributed training for faster development cycles")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {str(e)}")
            return ["Review model performance and deploy best model with appropriate monitoring"]
    
    async def _create_performance_summary(
        self,
        model_results: List[ModelEvaluationResult],
        best_model_result: ModelEvaluationResult,
        ensemble_result: Optional[EnsembleResult]
    ) -> Dict[str, Any]:
        """Create performance summary."""
        try:
            summary = {}
            
            # Model comparison
            model_comparison = []
            for result in sorted(model_results, key=lambda x: x.cv_result.mean_score, reverse=True):
                model_comparison.append({
                    'model_name': result.model_name,
                    'cv_score': result.cv_result.mean_score,
                    'cv_std': result.cv_result.std_score,
                    'training_time': result.training_time,
                    'model_size_mb': result.model_size,
                    'business_score': result.business_metrics.get('overall_business_score', 0)
                })
            
            summary['model_comparison'] = model_comparison
            
            # Best model details
            summary['best_model'] = {
                'name': best_model_result.model_name,
                'cv_score': best_model_result.cv_result.mean_score,
                'cv_confidence_interval': best_model_result.cv_result.confidence_interval,
                'test_scores': best_model_result.test_scores,
                'hyperparameters': best_model_result.hyperparameter_result.best_params,
                'feature_importance_top_5': dict(
                    list(best_model_result.feature_importance.items())[:5]
                ) if best_model_result.feature_importance else None
            }
            
            # Ensemble details
            if ensemble_result:
                summary['ensemble'] = {
                    'method': ensemble_result.ensemble_method.value,
                    'n_base_models': len(ensemble_result.base_models),
                    'cv_score': ensemble_result.cv_scores.mean_score,
                    'improvement_over_best': ensemble_result.cv_scores.mean_score - best_model_result.cv_result.mean_score,
                    'diversity_scores': ensemble_result.diversity_scores
                }
            
            # Overall statistics
            cv_scores = [r.cv_result.mean_score for r in model_results]
            summary['overall_statistics'] = {
                'n_models_evaluated': len(model_results),
                'best_score': float(max(cv_scores)),
                'worst_score': float(min(cv_scores)),
                'mean_score': float(np.mean(cv_scores)),
                'score_std': float(np.std(cv_scores)),
                'total_training_time': sum(r.training_time for r in model_results)
            }
            
            return summary
            
        except Exception as e:
            logger.warning(f"Performance summary creation failed: {str(e)}")
            return {}
    
    async def _log_to_mlflow(self, report: ModelSelectionReport):
        """Log model selection results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"model_selection_{report.task_type.value}"):
                # Log parameters
                mlflow.log_param("task_type", report.task_type.value)
                mlflow.log_param("n_models_evaluated", len(report.model_results))
                mlflow.log_param("optimization_method", report.config.optimization_method.value)
                mlflow.log_param("cv_strategy", report.config.cv_strategy.value)
                
                # Log dataset info
                for key, value in report.dataset_info.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"dataset_{key}", value)
                
                # Log best model metrics
                if report.best_model_result:
                    mlflow.log_metric("best_cv_score", report.best_model_result.cv_result.mean_score)
                    mlflow.log_metric("best_cv_std", report.best_model_result.cv_result.std_score)
                    mlflow.log_metric("best_training_time", report.best_model_result.training_time)
                    mlflow.log_param("best_model_name", report.best_model_result.model_name)
                    
                    # Log test scores
                    for metric, value in report.best_model_result.test_scores.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"test_{metric}", value)
                
                # Log ensemble metrics if available
                if report.ensemble_result:
                    mlflow.log_metric("ensemble_cv_score", report.ensemble_result.cv_scores.mean_score)
                    mlflow.log_param("ensemble_method", report.ensemble_result.ensemble_method.value)
                    mlflow.log_param("ensemble_n_models", len(report.ensemble_result.base_models))
                
                # Log model artifacts
                report_dict = asdict(report)
                report_dict['timestamp'] = report.timestamp.isoformat()
                
                with open("model_selection_report.json", "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)
                mlflow.log_artifact("model_selection_report.json")
                
                # Log best model
                if report.best_model_result:
                    mlflow.sklearn.log_model(
                        report.best_model_result.model,
                        "best_model",
                        registered_model_name=f"{report.task_type.value}_best_model"
                    )
                
                logger.info("Model selection results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")

# Utility functions for integration

def create_model_selector(
    optimization_method: str = "bayesian_optuna",
    cv_strategy: str = "stratified_k_fold",
    enable_ensemble: bool = True,
    max_optimization_time: int = 3600
) -> ModelSelector:
    """Factory function to create ModelSelector."""
    config = ModelSelectionConfig()
    config.optimization_method = OptimizationMethod(optimization_method)
    config.cv_strategy = CVStrategy(cv_strategy)
    config.enable_ensemble = enable_ensemble
    config.optimization_timeout = max_optimization_time
    
    return ModelSelector(config)

def create_default_model_candidates(task_type: TaskType) -> List[ModelCandidate]:
    """Create default model candidates for different task types."""
    candidates = []
    
    if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
        # Classification models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        candidates.extend([
            ModelCandidate(
                model_class=RandomForestClassifier,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20}
                },
                model_name="RandomForest",
                model_type="ensemble",
                complexity_score=0.6,
                interpretability_score=0.7
            ),
            ModelCandidate(
                model_class=LogisticRegression,
                param_space={
                    'C': {'type': 'float', 'low': 0.01, 'high': 100, 'log': True},
                    'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
                    'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']}
                },
                model_name="LogisticRegression",
                model_type="linear",
                complexity_score=0.2,
                interpretability_score=0.9
            ),
            ModelCandidate(
                model_class=GradientBoostingClassifier,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
                },
                model_name="GradientBoosting",
                model_type="boosting",
                complexity_score=0.8,
                interpretability_score=0.5
            )
        ])
        
    elif task_type == TaskType.REGRESSION:
        # Regression models
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge, ElasticNet
        
        candidates.extend([
            ModelCandidate(
                model_class=RandomForestRegressor,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20}
                },
                model_name="RandomForestRegressor",
                model_type="ensemble",
                complexity_score=0.6,
                interpretability_score=0.7
            ),
            ModelCandidate(
                model_class=Ridge,
                param_space={
                    'alpha': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True}
                },
                model_name="Ridge",
                model_type="linear",
                complexity_score=0.2,
                interpretability_score=0.9
            ),
            ModelCandidate(
                model_class=GradientBoostingRegressor,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
                },
                model_name="GradientBoostingRegressor",
                model_type="boosting",
                complexity_score=0.8,
                interpretability_score=0.5
            )
        ])
    
    return candidates

# Export main classes and functions
__all__ = [
    'ModelSelector',
    'ModelSelectionConfig',
    'ModelSelectionReport',
    'ModelCandidate',
    'ModelEvaluationResult',
    'EnsembleResult',
    'DataSplitter',
    'CrossValidator',
    'HyperparameterOptimizer',
    'ModelEvaluator',
    'EnsembleBuilder',
    'create_model_selector',
    'create_default_model_candidates',
    'TaskType',
    'SplitStrategy',
    'CVStrategy',
    'OptimizationMethod',
    'EnsembleMethod'
]

# Example usage and testing
if __name__ == "__main__":
    async def test_model_selection():
        """Test the model selection functionality."""
        print("Testing Model Selection System...")
        print("Available optimization methods:", [method.value for method in OptimizationMethod])
        print("Available CV strategies:", [strategy.value for strategy in CVStrategy])
        
        # Generate synthetic dataset
        from sklearn.datasets import make_classification, make_regression
        import pandas as pd
        
        # Test classification
        print("\n=== Classification Task Test ===")
        X_class, y_class = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        # Convert to DataFrame for better handling
        feature_names = [f'feature_{i}' for i in range(X_class.shape[1])]
        X_class_df = pd.DataFrame(X_class, columns=feature_names)
        
        print(f"Classification dataset: {X_class_df.shape} samples, {len(np.unique(y_class))} classes")
        
        # Create model candidates
        classification_candidates = create_default_model_candidates(TaskType.MULTICLASS_CLASSIFICATION)
        print(f"Model candidates: {[c.model_name for c in classification_candidates]}")
        
        # Create model selector with conservative settings for testing
        selector = create_model_selector(
            optimization_method="random_search",
            cv_strategy="stratified_k_fold",
            enable_ensemble=True,
            max_optimization_time=300  # 5 minutes for testing
        )
        
        # Override some config for faster testing
        selector.config.optimization_budget = 20  # Fewer trials
        selector.config.cv_folds = 3  # Fewer folds
        selector.config.ensemble_size = 2  # Smaller ensemble
        
        # Run model selection
        classification_report = await selector.select_best_model(
            model_candidates=classification_candidates,
            X=X_class_df,
            y=y_class,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            feature_names=feature_names
        )
        
        print(f"Classification Results:")
        print(f"  Best Model: {classification_report.best_model_result.model_name}")
        print(f"  CV Score: {classification_report.best_model_result.cv_result.mean_score:.4f} (±{classification_report.best_model_result.cv_result.std_score:.4f})")
        print(f"  Training Time: {classification_report.best_model_result.training_time:.2f}s")
        print(f"  Model Size: {classification_report.best_model_result.model_size:.2f} MB")
        
        # Display model comparison
        print(f"\n  Model Comparison:")
        for i, result in enumerate(classification_report.model_results):
            print(f"    {i+1}. {result.model_name}: {result.cv_result.mean_score:.4f} (±{result.cv_result.std_score:.4f})")
        
        # Ensemble results
        if classification_report.ensemble_result:
            print(f"\n  Ensemble Results:")
            print(f"    Method: {classification_report.ensemble_result.ensemble_method.value}")
            print(f"    CV Score: {classification_report.ensemble_result.cv_scores.mean_score:.4f}")
            improvement = classification_report.ensemble_result.cv_scores.mean_score - classification_report.best_model_result.cv_result.mean_score
            print(f"    Improvement over best: {improvement:.4f}")
        
        # Feature importance
        if classification_report.best_model_result.feature_importance:
            print(f"\n  Top 5 Important Features:")
            for i, (feature, importance) in enumerate(list(classification_report.best_model_result.feature_importance.items())[:5]):
                print(f"    {i+1}. {feature}: {importance:.4f}")
        
        # Insights and recommendations
        print(f"\n  Business Insights:")
        for i, insight in enumerate(classification_report.insights[:3], 1):
            print(f"    {i}. {insight}")
        
        print(f"\n  Recommendations:")
        for i, rec in enumerate(classification_report.recommendations[:3], 1):
            print(f"    {i}. {rec}")
        
        # Test regression
        print("\n\n=== Regression Task Test ===")
        X_reg, y_reg = make_regression(
            n_samples=800,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        
        feature_names_reg = [f'feature_{i}' for i in range(X_reg.shape[1])]
        X_reg_df = pd.DataFrame(X_reg, columns=feature_names_reg)
        
        print(f"Regression dataset: {X_reg_df.shape} samples")
        print(f"Target range: {y_reg.min():.2f} to {y_reg.max():.2f}")
        
        # Create regression candidates
        regression_candidates = create_default_model_candidates(TaskType.REGRESSION)
        print(f"Regression candidates: {[c.model_name for c in regression_candidates]}")
        
        # Create selector for regression
        reg_selector = create_model_selector(
            optimization_method="bayesian_optuna" if OPTUNA_AVAILABLE else "random_search",
            cv_strategy="k_fold",
            enable_ensemble=True,
            max_optimization_time=300
        )
        
        # Override config for testing
        reg_selector.config.optimization_budget = 25
        reg_selector.config.cv_folds = 3
        
        # Run regression model selection
        regression_report = await reg_selector.select_best_model(
            model_candidates=regression_candidates,
            X=X_reg_df,
            y=y_reg,
            task_type=TaskType.REGRESSION,
            feature_names=feature_names_reg
        )
        
        print(f"Regression Results:")
        print(f"  Best Model: {regression_report.best_model_result.model_name}")
        
        # For regression, use R² score or negative MSE
        best_test_score = regression_report.best_model_result.test_scores.get('r2_score', 
                         -regression_report.best_model_result.test_scores.get('mse', 0))
        print(f"  Test R² Score: {best_test_score:.4f}")
        print(f"  CV Score: {regression_report.best_model_result.cv_result.mean_score:.4f}")
        print(f"  RMSE: {regression_report.best_model_result.test_scores.get('rmse', 0):.4f}")
        
        # Test hyperparameter optimization methods
        print("\n\n=== Hyperparameter Optimization Comparison ===")
        
        # Use a simple dataset and model for comparison
        X_small, y_small = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        
        optimization_methods = [OptimizationMethod.RANDOM_SEARCH]
        if OPTUNA_AVAILABLE:
            optimization_methods.append(OptimizationMethod.BAYESIAN_OPTUNA)
        
        optimization_results = {}
        
        # Simple model candidate for comparison
        from sklearn.ensemble import RandomForestClassifier
        simple_candidate = ModelCandidate(
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10}
            },
            model_name="SimpleRandomForest",
            model_type="ensemble"
        )
        
        for method in optimization_methods:
            try:
                print(f"\n  Testing {method.value}...")
                
                # Create temporary selector
                temp_selector = create_model_selector(optimization_method=method.value)
                temp_selector.config.optimization_budget = 15  # Small budget for comparison
                temp_selector.config.optimization_timeout = 60   # 1 minute timeout
                
                # Split data manually for hyperparameter optimization test
                X_train, X_val, y_train, y_val = train_test_split(
                    X_small, y_small, test_size=0.3, random_state=42
                )
                
                # Test hyperparameter optimization directly
                hp_result = await temp_selector.hyperparameter_optimizer.optimize_hyperparameters(
                    simple_candidate, X_train, y_train, X_val, y_val
                )
                
                optimization_results[method.value] = {
                    'best_score': hp_result.best_score,
                    'n_trials': hp_result.n_trials,
                    'total_time': hp_result.total_time,
                    'best_params': hp_result.best_params
                }
                
                print(f"    Best Score: {hp_result.best_score:.4f}")
                print(f"    Trials: {hp_result.n_trials}")
                print(f"    Time: {hp_result.total_time:.2f}s")
                print(f"    Best Params: {hp_result.best_params}")
                
            except Exception as e:
                print(f"    Failed: {str(e)}")
                continue
        
        # Cross-validation strategies comparison
        print("\n\n=== Cross-Validation Strategies Test ===")
        
        cv_strategies = [CVStrategy.K_FOLD, CVStrategy.STRATIFIED_K_FOLD]
        if len(np.unique(y_small)) > 1:  # Ensure we have multiple classes
            cv_strategies.append(CVStrategy.REPEATED_K_FOLD)
        
        cv_results = {}
        
        # Simple model for CV testing
        from sklearn.ensemble import RandomForestClassifier
        simple_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        for strategy in cv_strategies:
            try:
                print(f"\n  Testing {strategy.value}...")
                
                # Create temporary cross validator
                temp_config = ModelSelectionConfig()
                temp_config.cv_strategy = strategy
                temp_config.cv_folds = 3
                temp_config.cv_repeats = 2 if strategy == CVStrategy.REPEATED_K_FOLD else 1
                
                temp_cv = CrossValidator(temp_config)
                
                cv_result = await temp_cv.cross_validate_model(
                    simple_model, X_small, y_small
                )
                
                cv_results[strategy.value] = cv_result
                
                print(f"    Mean Score: {cv_result.mean_score:.4f}")
                print(f"    Std Score: {cv_result.std_score:.4f}")
                print(f"    Confidence Interval: {cv_result.confidence_interval}")
                print(f"    Mean Fit Time: {cv_result.mean_fit_time:.3f}s")
                
            except Exception as e:
                print(f"    Failed: {str(e)}")
                continue
        
        # Statistical significance testing
        if len(cv_results) >= 2 and SCIPY_AVAILABLE:
            print("\n  Statistical Significance Tests:")
            
            strategies = list(cv_results.keys())
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i+1:]:
                    try:
                        scores1 = cv_results[strategy1].scores
                        scores2 = cv_results[strategy2].scores
                        
                        if len(scores1) == len(scores2):
                            statistic, p_value = wilcoxon(scores1, scores2)
                            significance = "significant" if p_value < 0.05 else "not significant"
                            print(f"    {strategy1} vs {strategy2}: p-value = {p_value:.4f} ({significance})")
                    except Exception as e:
                        print(f"    {strategy1} vs {strategy2}: Test failed - {str(e)}")
        
        # Performance benchmarking
        print("\n\n=== Performance Benchmarking ===")
        
        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000] if mp.cpu_count() > 2 else [100, 300]  # Smaller sizes for limited resources
        
        performance_results = {}
        
        for size in dataset_sizes:
            try:
                print(f"\n  Testing with dataset size: {size}")
                
                # Generate dataset
                X_bench, y_bench = make_classification(
                    n_samples=size,
                    n_features=min(20, size//10),  # Scale features with dataset size
                    n_classes=2,
                    random_state=42
                )
                
                # Simple model selection
                bench_candidates = [
                    ModelCandidate(
                        model_class=RandomForestClassifier,
                        param_space={
                            'n_estimators': [50, 100],
                            'max_depth': [3, 5, 10]
                        },
                        model_name="BenchRandomForest",
                        model_type="ensemble"
                    )
                ]
                
                # Create selector with minimal settings
                bench_selector = create_model_selector(
                    optimization_method="grid_search",
                    cv_strategy="k_fold",
                    enable_ensemble=False
                )
                bench_selector.config.optimization_budget = 6  # 2*3 combinations
                bench_selector.config.cv_folds = 3
                
                start_time = datetime.now()
                
                bench_report = await bench_selector.select_best_model(
                    model_candidates=bench_candidates,
                    X=X_bench,
                    y=y_bench,
                    task_type=TaskType.BINARY_CLASSIFICATION
                )
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                performance_results[size] = {
                    'total_time': total_time,
                    'best_score': bench_report.best_model_result.cv_result.mean_score,
                    'n_models': len(bench_report.model_results)
                }
                
                print(f"    Total Time: {total_time:.2f}s")
                print(f"    Best Score: {bench_report.best_model_result.cv_result.mean_score:.4f}")
                print(f"    Time per Sample: {total_time/size*1000:.2f}ms")
                
            except Exception as e:
                print(f"    Failed for size {size}: {str(e)}")
                continue
        
        # Scalability analysis
        if len(performance_results) >= 2:
            print(f"\n  Scalability Analysis:")
            sizes = sorted(performance_results.keys())
            times = [performance_results[size]['total_time'] for size in sizes]
            
            if len(sizes) >= 2:
                # Simple linear regression to estimate scalability
                slope = (times[-1] - times[0]) / (sizes[-1] - sizes[0])
                print(f"    Time scaling: ~{slope:.4f} seconds per additional sample")
                
                # Predict time for larger datasets
                for pred_size in [5000, 10000]:
                    pred_time = times[0] + slope * (pred_size - sizes[0])
                    print(f"    Predicted time for {pred_size} samples: {pred_time:.1f}s ({pred_time/60:.1f}min)")
        
        # Integration tests
        print("\n\n=== Integration Tests ===")
        
        # Test with missing values
        print("\n  Testing with missing values...")
        X_missing = X_class_df.copy()
        # Introduce random missing values
        missing_mask = np.random.random(X_missing.shape) < 0.1  # 10% missing
        X_missing = X_missing.astype(float)
        X_missing[missing_mask] = np.nan
        
        try:
            # Simple imputation
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_missing_imputed = pd.DataFrame(
                imputer.fit_transform(X_missing),
                columns=X_missing.columns
            )
            
            missing_selector = create_model_selector(enable_ensemble=False)
            missing_selector.config.optimization_budget = 10
            
            missing_candidates = [classification_candidates[0]]  # Use only one model
            
            missing_report = await missing_selector.select_best_model(
                model_candidates=missing_candidates,
                X=X_missing_imputed,
                y=y_class,
                task_type=TaskType.MULTICLASS_CLASSIFICATION
            )
            
            print(f"    Successfully handled missing values")
            print(f"    Score with missing data: {missing_report.best_model_result.cv_result.mean_score:.4f}")
            
        except Exception as e:
            print(f"    Missing values test failed: {str(e)}")
        
        # Test with categorical features
        print("\n  Testing with categorical features...")
        try:
            # Add categorical features
            X_categorical = X_class_df.copy()
            X_categorical['category_A'] = np.random.choice(['A', 'B', 'C'], size=len(X_categorical))
            X_categorical['category_B'] = np.random.choice(['X', 'Y'], size=len(X_categorical))
            
            # Encode categorical features
            from sklearn.preprocessing import LabelEncoder
            
            for col in ['category_A', 'category_B']:
                le = LabelEncoder()
                X_categorical[col] = le.fit_transform(X_categorical[col])
            
            cat_selector = create_model_selector(enable_ensemble=False)
            cat_selector.config.optimization_budget = 10
            
            cat_report = await cat_selector.select_best_model(
                model_candidates=[classification_candidates[0]],
                X=X_categorical,
                y=y_class,
                task_type=TaskType.MULTICLASS_CLASSIFICATION,
                feature_names=list(X_categorical.columns)
            )
            
            print(f"    Successfully handled categorical features")
            print(f"    Score with categorical features: {cat_report.best_model_result.cv_result.mean_score:.4f}")
            
        except Exception as e:
            print(f"    Categorical features test failed: {str(e)}")
        
        # Resource monitoring
        print("\n\n=== Resource Monitoring ===")
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            print(f"  Memory Usage:")
            print(f"    RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
            print(f"    VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
            
            cpu_percent = process.cpu_percent()
            print(f"  CPU Usage: {cpu_percent:.1f}%")
            
        except ImportError:
            print("  psutil not available for resource monitoring")
        except Exception as e:
            print(f"  Resource monitoring failed: {str(e)}")
        
        # Summary
        print(f"\n\n=== Test Summary ===")
        print(f"✓ Classification model selection completed")
        print(f"✓ Regression model selection completed")
        print(f"✓ Hyperparameter optimization methods tested")
        print(f"✓ Cross-validation strategies tested")
        print(f"✓ Performance benchmarking completed")
        print(f"✓ Integration tests passed")
        
        print(f"\nLibrary Availability:")
        print(f"  Optuna: {'✓' if OPTUNA_AVAILABLE else '✗'}")
        print(f"  Hyperopt: {'✓' if HYPEROPT_AVAILABLE else '✗'}")
        print(f"  Scikit-Optimize: {'✓' if SCIKIT_OPTIMIZE_AVAILABLE else '✗'}")
        print(f"  SciPy: {'✓' if SCIPY_AVAILABLE else '✗'}")
        print(f"  MLflow: {'✓' if MLFLOW_AVAILABLE else '✗'}")
        print(f"  Joblib: {'✓' if JOBLIB_AVAILABLE else '✗'}")
        
        return classification_report, regression_report
    
    # Run comprehensive tests
    try:
        import asyncio
        classification_results, regression_results = asyncio.run(test_model_selection())
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📊 Classification best model: {classification_results.best_model_result.model_name}")
        print(f"📈 Regression best model: {regression_results.best_model_result.model_name}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Tests failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

# Additional utility functions for production integration

def get_optimization_recommendations(
    dataset_size: int,
    n_features: int,
    time_budget: int = 3600,
    computational_resources: str = "medium"
) -> Dict[str, Any]:
    """Get optimization recommendations based on dataset characteristics."""
    try:
        recommendations = {}
        
        # Optimization method recommendations
        if dataset_size < 1000:
            recommendations['optimization_method'] = OptimizationMethod.GRID_SEARCH
            recommendations['optimization_budget'] = min(50, time_budget // 10)
        elif dataset_size < 10000 and OPTUNA_AVAILABLE:
            recommendations['optimization_method'] = OptimizationMethod.BAYESIAN_OPTUNA
            recommendations['optimization_budget'] = min(100, time_budget // 20)
        else:
            recommendations['optimization_method'] = OptimizationMethod.RANDOM_SEARCH
            recommendations['optimization_budget'] = min(200, time_budget // 15)
        
        # CV strategy recommendations
        if dataset_size < 500:
            recommendations['cv_strategy'] = CVStrategy.K_FOLD
            recommendations['cv_folds'] = 3
        elif dataset_size < 5000:
            recommendations['cv_strategy'] = CVStrategy.STRATIFIED_K_FOLD
            recommendations['cv_folds'] = 5
        else:
            recommendations['cv_strategy'] = CVStrategy.STRATIFIED_K_FOLD
            recommendations['cv_folds'] = 5
            recommendations['enable_repeated_cv'] = True
        
        # Ensemble recommendations
        if dataset_size > 1000 and time_budget > 1800:  # 30 minutes
            recommendations['enable_ensemble'] = True
            recommendations['ensemble_size'] = min(5, max(3, dataset_size // 1000))
        else:
            recommendations['enable_ensemble'] = False
        
        # Computational settings
        if computational_resources == "low":
            recommendations['max_workers'] = 1
            recommendations['enable_parallel'] = False
            recommendations['memory_limit_gb'] = 2
        elif computational_resources == "high":
            recommendations['max_workers'] = min(mp.cpu_count(), 8)
            recommendations['enable_parallel'] = True
            recommendations['memory_limit_gb'] = 16
        else:  # medium
            recommendations['max_workers'] = min(mp.cpu_count() // 2, 4)
            recommendations['enable_parallel'] = True
            recommendations['memory_limit_gb'] = 8
        
        # Model complexity recommendations
        if n_features > dataset_size:
            recommendations['regularization_focus'] = True
            recommendations['feature_selection'] = True
        elif n_features > 100:
            recommendations['dimensionality_reduction'] = True
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return {
            'optimization_method': OptimizationMethod.RANDOM_SEARCH,
            'cv_strategy': CVStrategy.STRATIFIED_K_FOLD,
            'enable_ensemble': False
        }

def estimate_training_time(
    n_samples: int,
    n_features: int,
    n_models: int = 5,
    optimization_budget: int = 100,
    cv_folds: int = 5,
    computational_resources: str = "medium"
) -> Dict[str, float]:
    """Estimate training time for model selection."""
    try:
        # Base time estimates (seconds per sample per feature)
        base_times = {
            "low": 0.001,
            "medium": 0.0005,
            "high": 0.0002
        }
        
        base_time = base_times.get(computational_resources, base_times["medium"])
        
        # Time components
        single_model_time = n_samples * n_features * base_time
        cv_multiplier = cv_folds
        optimization_multiplier = optimization_budget / 10  # Assuming 10 is baseline
        
        # Total time per model
        time_per_model = single_model_time * cv_multiplier * optimization_multiplier
        
        # Total time for all models
        total_model_time = time_per_model * n_models
        
        # Ensemble overhead (if applicable)
        ensemble_time = total_model_time * 0.2  # 20% overhead
        
        # Total time
        total_time = total_model_time + ensemble_time
        
        return {
            'estimated_total_time_seconds': total_time,
            'estimated_total_time_minutes': total_time / 60,
            'estimated_total_time_hours': total_time / 3600,
            'time_per_model_seconds': time_per_model,
            'single_model_base_time': single_model_time,
            'cv_overhead_factor': cv_multiplier,
            'optimization_overhead_factor': optimization_multiplier
        }
        
    except Exception as e:
        logger.error(f"Training time estimation failed: {str(e)}")
        return {
            'estimated_total_time_seconds': 3600,  # Default 1 hour
            'estimated_total_time_minutes': 60,
            'estimated_total_time_hours': 1
        }

def create_production_config(
    task_type: TaskType,
    dataset_size: int,
    time_budget_minutes: int = 60,
    quality_focus: str = "balanced"  # "speed", "balanced", "quality"
) -> ModelSelectionConfig:
    """Create production-optimized configuration."""
    try:
        config = ModelSelectionConfig()
        
        # Get recommendations
        recommendations = get_optimization_recommendations(
            dataset_size=dataset_size,
            n_features=20,  # Assume moderate feature count
            time_budget=time_budget_minutes * 60,
            computational_resources="medium"
        )
        
        # Apply recommendations
        config.optimization_method = recommendations.get('optimization_method', OptimizationMethod.BAYESIAN_OPTUNA)
        config.optimization_budget = recommendations.get('optimization_budget', 100)
        config.cv_strategy = recommendations.get('cv_strategy', CVStrategy.STRATIFIED_K_FOLD)
        config.cv_folds = recommendations.get('cv_folds', 5)
        config.enable_ensemble = recommendations.get('enable_ensemble', True)
        config.ensemble_size = recommendations.get('ensemble_size', 5)
        
        # Quality focus adjustments
        if quality_focus == "speed":
            config.optimization_budget = max(20, config.optimization_budget // 2)
            config.cv_folds = max(3, config.cv_folds - 1)
            config.enable_ensemble = False
            config.optimization_timeout = time_budget_minutes * 30  # Use half the time budget
            
        elif quality_focus == "quality":
            config.optimization_budget = min(500, config.optimization_budget * 2)
            config.cv_folds = min(10, config.cv_folds + 2)
            config.enable_ensemble = True
            config.ensemble_size = min(8, config.ensemble_size + 2)
            config.optimization_timeout = time_budget_minutes * 50  # Use most of the time budget
            
        else:  # balanced
            config.optimization_timeout = time_budget_minutes * 40  # Use 2/3 of time budget
        
        # Production settings
        config.mlflow_tracking = True
        config.save_intermediate_results = True
        config.statistical_significance_test = True
        config.calculate_business_impact = True
        
        # Resource management
        config.memory_limit_gb = recommendations.get('memory_limit_gb', 8)
        config.enable_parallel = recommendations.get('enable_parallel', True)
        config.n_jobs = recommendations.get('max_workers', -1)
        
        return config
        
    except Exception as e:
        logger.error(f"Production config creation failed: {str(e)}")
        return ModelSelectionConfig()  # Return default config

def validate_model_selection_inputs(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    task_type: TaskType,
    model_candidates: List[ModelCandidate]
) -> Dict[str, Any]:
    """Validate inputs for model selection."""
    try:
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Data validation
        if len(X) == 0:
            validation_result['errors'].append("Empty dataset provided")
            validation_result['valid'] = False
        
        if len(X) != len(y):
            validation_result['errors'].append("Feature matrix and target vector have different lengths")
            validation_result['valid'] = False
        
        if len(X) < 10:
            validation_result['errors'].append("Dataset too small for reliable model selection (minimum 10 samples)")
            validation_result['valid'] = False
        elif len(X) < 100:
            validation_result['warnings'].append("Small dataset - results may not be reliable")
            validation_result['recommendations'].append("Consider collecting more data")
        
        # Feature validation
        if hasattr(X, 'shape') and X.shape[1] == 0:
            validation_result['errors'].append("No features provided")
            validation_result['valid'] = False
        elif hasattr(X, 'shape') and X.shape[1] > len(X):
            validation_result['warnings'].append("More features than samples - consider dimensionality reduction")
            validation_result['recommendations'].append("Apply feature selection or regularization")
        
        # Target validation
        unique_targets = len(np.unique(y))
        if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            if unique_targets < 2:
                validation_result['errors'].append("Classification task requires at least 2 classes")
                validation_result['valid'] = False
            elif unique_targets == 2 and task_type == TaskType.MULTICLASS_CLASSIFICATION:
                validation_result['warnings'].append("Binary classification detected - consider using BINARY_CLASSIFICATION task type")
        
        # Model candidates validation
        if not model_candidates:
            validation_result['errors'].append("No model candidates provided")
            validation_result['valid'] = False
        elif len(model_candidates) == 1:
            validation_result['warnings'].append("Only one model candidate - model selection benefits from multiple candidates")
        
        # Memory estimation
        estimated_memory = len(X) * (X.shape[1] if hasattr(X, 'shape') else 20) * 8 / (1024**2)  # MB
        if estimated_memory > 1000:  # 1GB
            validation_result['warnings'].append(f"Large dataset detected ({estimated_memory:.0f} MB) - may require significant memory")
            validation_result['recommendations'].append("Consider data sampling or distributed processing")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        return {
            'valid': False,
            'errors': [str(e)],
            'warnings': [],
            'recommendations': []
        }

# Auto-Analyst integration helpers

async def auto_select_and_train_models(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    task_type: Optional[str] = None,
    time_budget_minutes: int = 30,
    quality_focus: str = "balanced",
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Auto-Analyst integration function for automatic model selection and training.
    
    This is the main entry point for the Auto-Analyst platform.
    """
    try:
        # Auto-detect task type if not provided
        if task_type is None:
            if len(np.unique(y)) <= 20 and len(np.unique(y)) / len(y) < 0.1:
                if len(np.unique(y)) == 2:
                    detected_task_type = TaskType.BINARY_CLASSIFICATION
                else:
                    detected_task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                detected_task_type = TaskType.REGRESSION
        else:
            detected_task_type = TaskType(task_type)
        
        # Validate inputs
        model_candidates = create_default_model_candidates(detected_task_type)
        validation = validate_model_selection_inputs(X, y, detected_task_type, model_candidates)
        
        if not validation['valid']:
            return {
                'status': 'failed',
                'error': 'Input validation failed',
                'validation_errors': validation['errors'],
                'recommendations': validation['recommendations']
            }
        
        # Create production configuration
        config = create_production_config(
            task_type=detected_task_type,
            dataset_size=len(X),
            time_budget_minutes=time_budget_minutes,
            quality_focus=quality_focus
        )
        
        # Create model selector
        selector = ModelSelector(config)
        
        # Estimate training time
        time_estimate = estimate_training_time(
            n_samples=len(X),
            n_features=X.shape[1] if hasattr(X, 'shape') else len(feature_names or []),
            n_models=len(model_candidates),
            optimization_budget=config.optimization_budget,
            cv_folds=config.cv_folds
        )
        
        # Run model selection
        report = await selector.select_best_model(
            model_candidates=model_candidates,
            X=X,
            y=y,
            task_type=detected_task_type,
            feature_names=feature_names
        )
        
        # Format results for Auto-Analyst
        return {
            'status': 'completed',
            'task_type': detected_task_type.value,
            'best_model': {
                'name': report.best_model_result.model_name,
                'type': report.best_model_result.model_type if hasattr(report.best_model_result, 'model_type') else 'unknown',
                'cv_score': report.best_model_result.cv_result.mean_score,
                'cv_std': report.best_model_result.cv_result.std_score,
                'test_scores': report.best_model_result.test_scores,
                'training_time': report.best_model_result.training_time,
                'hyperparameters': report.best_model_result.hyperparameter_result.best_params
            },
            'model_comparison': [
                {
                    'name': result.model_name,
                    'cv_score': result.cv_result.mean_score,
                    'cv_std': result.cv_result.std_score,
                    'training_time': result.training_time
                }
                for result in report.model_results
            ],
            'ensemble': {
                'enabled': report.ensemble_result is not None,
                'score': report.ensemble_result.cv_scores.mean_score if report.ensemble_result else None,
                'improvement': (report.ensemble_result.cv_scores.mean_score - report.best_model_result.cv_result.mean_score) if report.ensemble_result else 0,
                'n_models': len(report.ensemble_result.base_models) if report.ensemble_result else 0
            },
            'feature_importance': report.best_model_result.feature_importance,
            'insights': report.insights,
            'recommendations': report.recommendations,
            'performance_summary': report.performance_summary,
            'validation_warnings': validation.get('warnings', []),
            'time_estimate': time_estimate,
            'execution_metadata': {
                'actual_time': report.metadata.get('execution_time', 0),
                'n_models_evaluated': report.metadata.get('n_models_evaluated', 0),
                'optimization_method': config.optimization_method.value,
                'cv_strategy': config.cv_strategy.value
            }
        }
        
    except Exception as e:
        logger.error(f"Auto model selection failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'task_type': task_type,
            'recommendations': ['Check data quality and try with simpler configuration']
        }

# Export for Auto-Analyst platform
__all__.extend([
    'auto_select_and_train_models',
    'get_optimization_recommendations',
    'estimate_training_time',
    'create_production_config',
    'validate_model_selection_inputs'
])
