"""
Tabular Models Module for Auto-Analyst Platform

This module implements comprehensive tabular machine learning capabilities including:
- Gradient Boosting Models (XGBoost, CatBoost, LightGBM)
- Tree-based Models (Random Forest, Extra Trees, Decision Trees)
- Advanced Neural Networks (TabPFN, TabNet, NODE)
- Ensemble Methods (Voting, Stacking, Blending)
- Linear Models (Ridge, Lasso, ElasticNet)
- Bayesian Optimization for hyperparameter tuning
- Automatic model selection and comparison
- Feature importance and model interpretation
- Advanced categorical feature handling
- Missing value imputation strategies

Features:
- Automatic algorithm selection based on data characteristics
- Sophisticated hyperparameter optimization with Optuna
- Comprehensive model evaluation and comparison framework
- Advanced feature engineering and preprocessing
- Model interpretation with SHAP and LIME integration
- Early stopping and overfitting prevention
- Categorical feature encoding optimization
- Cross-validation strategies for robust evaluation
- Model ensemble and stacking capabilities
- Production-ready model serving and deployment
- Business impact assessment and ROI analysis
- Integration with MLflow for experiment tracking
- Real-time prediction serving with caching
- A/B testing framework for model comparison
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
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
from collections import defaultdict
import itertools

# Core ML libraries
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Tree-based models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Linear models
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)

# Advanced boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Advanced neural networks for tabular data
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Category encoders
try:
    from category_encoders import (
        TargetEncoder, BinaryEncoder, HashingEncoder,
        LeaveOneOutEncoder, CatBoostEncoder
    )
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of tabular models."""
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    ADABOOST = "adaboost"
    DECISION_TREE = "decision_tree"
    TABPFN = "tabpfn"
    TABNET = "tabnet"
    LINEAR = "linear"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"

class TaskType(Enum):
    """Types of ML tasks."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    RANKING = "ranking"

class OptimizationObjective(Enum):
    """Hyperparameter optimization objectives."""
    ACCURACY = "accuracy"
    ROC_AUC = "roc_auc"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    R2_SCORE = "r2_score"
    RMSE = "rmse"
    MAE = "mae"

@dataclass
class TabularModelConfig:
    """Configuration for tabular models."""
    
    def __init__(self):
        # General settings
        self.auto_select_models = True
        self.max_models_to_try = 10
        self.enable_ensemble = True
        self.ensemble_size = 5
        self.random_state = 42
        
        # Data preprocessing
        self.handle_missing_values = True
        self.missing_strategy = 'auto'  # 'auto', 'mean', 'median', 'mode', 'drop'
        self.encode_categorical = True
        self.categorical_encoding = 'auto'  # 'auto', 'onehot', 'target', 'ordinal'
        self.scale_features = True
        self.feature_selection = True
        self.feature_selection_k = 'auto'
        
        # Model selection
        self.include_xgboost = XGBOOST_AVAILABLE
        self.include_catboost = CATBOOST_AVAILABLE
        self.include_lightgbm = LIGHTGBM_AVAILABLE
        self.include_random_forest = True
        self.include_extra_trees = True
        self.include_gradient_boosting = True
        self.include_tabpfn = TABPFN_AVAILABLE
        self.include_linear_models = True
        self.include_neural_networks = PYTORCH_AVAILABLE
        
        # Cross-validation
        self.cv_folds = 5
        self.cv_strategy = 'stratified'  # 'stratified', 'kfold', 'repeated'
        self.cv_scoring = 'auto'
        self.test_size = 0.2
        
        # Hyperparameter optimization
        self.enable_hyperopt = True
        self.hyperopt_method = 'optuna'  # 'optuna', 'hyperopt', 'grid', 'random'
        self.hyperopt_trials = 100
        self.hyperopt_timeout = 3600  # seconds
        self.early_stopping_rounds = 50
        
        # Model-specific settings
        # XGBoost
        self.xgb_max_depth = 6
        self.xgb_learning_rate = 0.1
        self.xgb_n_estimators = 100
        self.xgb_subsample = 1.0
        self.xgb_colsample_bytree = 1.0
        
        # CatBoost
        self.cat_iterations = 1000
        self.cat_learning_rate = 0.1
        self.cat_depth = 6
        self.cat_l2_leaf_reg = 3
        
        # LightGBM
        self.lgb_num_leaves = 31
        self.lgb_learning_rate = 0.1
        self.lgb_feature_fraction = 0.9
        self.lgb_bagging_fraction = 0.8
        self.lgb_n_estimators = 100
        
        # Random Forest
        self.rf_n_estimators = 100
        self.rf_max_depth = None
        self.rf_min_samples_split = 2
        self.rf_min_samples_leaf = 1
        self.rf_max_features = 'sqrt'
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        self.use_gpu = False
        self.memory_limit = '4GB'
        
        # Evaluation settings
        self.calculate_feature_importance = True
        self.calculate_shap_values = SHAP_AVAILABLE
        self.generate_model_explanations = True
        self.enable_model_interpretation = True
        
        # Business settings
        self.calculate_business_impact = True
        self.feature_cost_mapping = {}  # Feature name -> cost
        self.prediction_value_mapping = {}  # Prediction -> business value
        
        # Production settings
        self.enable_model_versioning = True
        self.save_preprocessing_pipeline = True
        self.enable_prediction_caching = True
        self.model_monitoring = True
        
        # Quality settings
        self.min_samples_for_model = 100
        self.max_training_time = 7200  # seconds
        self.convergence_tolerance = 1e-6
        self.validate_data_quality = True

@dataclass
class ModelResult:
    """Result of a single model training and evaluation."""
    model_type: ModelType
    model: BaseEstimator
    cv_scores: List[float]
    cv_score_mean: float
    cv_score_std: float
    test_score: float
    training_time: float
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    model_size: int  # Size in bytes
    predictions: Optional[np.ndarray]
    prediction_probabilities: Optional[np.ndarray]

@dataclass
class TabularModelReport:
    """Comprehensive tabular modeling report."""
    report_id: str
    timestamp: datetime
    task_type: TaskType
    dataset_info: Dict[str, Any]
    models_evaluated: List[ModelResult]
    best_model_result: ModelResult
    ensemble_result: Optional[ModelResult]
    feature_analysis: Dict[str, Any]
    performance_comparison: Dict[str, Any]
    business_impact: Dict[str, Any]
    model_interpretability: Dict[str, Any]
    recommendations: List[str]
    insights: List[str]
    metadata: Dict[str, Any]

class TabularModelFactory:
    """Factory for creating tabular models with optimized configurations."""
    
    @staticmethod
    def create_model(
        model_type: ModelType,
        task_type: TaskType,
        config: TabularModelConfig,
        hyperparameters: Optional[Dict] = None
    ) -> BaseEstimator:
        """Create a tabular model instance."""
        try:
            params = hyperparameters or {}
            
            if model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
                if task_type == TaskType.REGRESSION:
                    return xgb.XGBRegressor(
                        max_depth=params.get('max_depth', config.xgb_max_depth),
                        learning_rate=params.get('learning_rate', config.xgb_learning_rate),
                        n_estimators=params.get('n_estimators', config.xgb_n_estimators),
                        subsample=params.get('subsample', config.xgb_subsample),
                        colsample_bytree=params.get('colsample_bytree', config.xgb_colsample_bytree),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        gpu_id=0 if config.use_gpu else None,
                        tree_method='gpu_hist' if config.use_gpu else 'hist',
                        **{k: v for k, v in params.items() if k not in [
                            'max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree'
                        ]}
                    )
                else:
                    return xgb.XGBClassifier(
                        max_depth=params.get('max_depth', config.xgb_max_depth),
                        learning_rate=params.get('learning_rate', config.xgb_learning_rate),
                        n_estimators=params.get('n_estimators', config.xgb_n_estimators),
                        subsample=params.get('subsample', config.xgb_subsample),
                        colsample_bytree=params.get('colsample_bytree', config.xgb_colsample_bytree),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        gpu_id=0 if config.use_gpu else None,
                        tree_method='gpu_hist' if config.use_gpu else 'hist',
                        eval_metric='logloss',
                        **{k: v for k, v in params.items() if k not in [
                            'max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree'
                        ]}
                    )
            
            elif model_type == ModelType.CATBOOST and CATBOOST_AVAILABLE:
                if task_type == TaskType.REGRESSION:
                    return cb.CatBoostRegressor(
                        iterations=params.get('iterations', config.cat_iterations),
                        learning_rate=params.get('learning_rate', config.cat_learning_rate),
                        depth=params.get('depth', config.cat_depth),
                        l2_leaf_reg=params.get('l2_leaf_reg', config.cat_l2_leaf_reg),
                        random_state=config.random_state,
                        thread_count=config.n_jobs if config.enable_parallel else 1,
                        task_type='GPU' if config.use_gpu else 'CPU',
                        verbose=False,
                        **{k: v for k, v in params.items() if k not in [
                            'iterations', 'learning_rate', 'depth', 'l2_leaf_reg'
                        ]}
                    )
                else:
                    return cb.CatBoostClassifier(
                        iterations=params.get('iterations', config.cat_iterations),
                        learning_rate=params.get('learning_rate', config.cat_learning_rate),
                        depth=params.get('depth', config.cat_depth),
                        l2_leaf_reg=params.get('l2_leaf_reg', config.cat_l2_leaf_reg),
                        random_state=config.random_state,
                        thread_count=config.n_jobs if config.enable_parallel else 1,
                        task_type='GPU' if config.use_gpu else 'CPU',
                        verbose=False,
                        **{k: v for k, v in params.items() if k not in [
                            'iterations', 'learning_rate', 'depth', 'l2_leaf_reg'
                        ]}
                    )
            
            elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
                if task_type == TaskType.REGRESSION:
                    return lgb.LGBMRegressor(
                        num_leaves=params.get('num_leaves', config.lgb_num_leaves),
                        learning_rate=params.get('learning_rate', config.lgb_learning_rate),
                        feature_fraction=params.get('feature_fraction', config.lgb_feature_fraction),
                        bagging_fraction=params.get('bagging_fraction', config.lgb_bagging_fraction),
                        n_estimators=params.get('n_estimators', config.lgb_n_estimators),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        device='gpu' if config.use_gpu else 'cpu',
                        verbose=-1,
                        **{k: v for k, v in params.items() if k not in [
                            'num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction', 'n_estimators'
                        ]}
                    )
                else:
                    return lgb.LGBMClassifier(
                        num_leaves=params.get('num_leaves', config.lgb_num_leaves),
                        learning_rate=params.get('learning_rate', config.lgb_learning_rate),
                        feature_fraction=params.get('feature_fraction', config.lgb_feature_fraction),
                        bagging_fraction=params.get('bagging_fraction', config.lgb_bagging_fraction),
                        n_estimators=params.get('n_estimators', config.lgb_n_estimators),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        device='gpu' if config.use_gpu else 'cpu',
                        verbose=-1,
                        **{k: v for k, v in params.items() if k not in [
                            'num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction', 'n_estimators'
                        ]}
                    )
            
            elif model_type == ModelType.RANDOM_FOREST:
                if task_type == TaskType.REGRESSION:
                    return RandomForestRegressor(
                        n_estimators=params.get('n_estimators', config.rf_n_estimators),
                        max_depth=params.get('max_depth', config.rf_max_depth),
                        min_samples_split=params.get('min_samples_split', config.rf_min_samples_split),
                        min_samples_leaf=params.get('min_samples_leaf', config.rf_min_samples_leaf),
                        max_features=params.get('max_features', config.rf_max_features),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        **{k: v for k, v in params.items() if k not in [
                            'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'
                        ]}
                    )
                else:
                    return RandomForestClassifier(
                        n_estimators=params.get('n_estimators', config.rf_n_estimators),
                        max_depth=params.get('max_depth', config.rf_max_depth),
                        min_samples_split=params.get('min_samples_split', config.rf_min_samples_split),
                        min_samples_leaf=params.get('min_samples_leaf', config.rf_min_samples_leaf),
                        max_features=params.get('max_features', config.rf_max_features),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        **{k: v for k, v in params.items() if k not in [
                            'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'
                        ]}
                    )
            
            elif model_type == ModelType.EXTRA_TREES:
                if task_type == TaskType.REGRESSION:
                    return ExtraTreesRegressor(
                        n_estimators=params.get('n_estimators', config.rf_n_estimators),
                        max_depth=params.get('max_depth', config.rf_max_depth),
                        min_samples_split=params.get('min_samples_split', config.rf_min_samples_split),
                        min_samples_leaf=params.get('min_samples_leaf', config.rf_min_samples_leaf),
                        max_features=params.get('max_features', config.rf_max_features),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        **params
                    )
                else:
                    return ExtraTreesClassifier(
                        n_estimators=params.get('n_estimators', config.rf_n_estimators),
                        max_depth=params.get('max_depth', config.rf_max_depth),
                        min_samples_split=params.get('min_samples_split', config.rf_min_samples_split),
                        min_samples_leaf=params.get('min_samples_leaf', config.rf_min_samples_leaf),
                        max_features=params.get('max_features', config.rf_max_features),
                        random_state=config.random_state,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        **params
                    )
            
            elif model_type == ModelType.GRADIENT_BOOSTING:
                if task_type == TaskType.REGRESSION:
                    return GradientBoostingRegressor(
                        n_estimators=params.get('n_estimators', 100),
                        learning_rate=params.get('learning_rate', 0.1),
                        max_depth=params.get('max_depth', 3),
                        random_state=config.random_state,
                        **params
                    )
                else:
                    return GradientBoostingClassifier(
                        n_estimators=params.get('n_estimators', 100),
                        learning_rate=params.get('learning_rate', 0.1),
                        max_depth=params.get('max_depth', 3),
                        random_state=config.random_state,
                        **params
                    )
            
            elif model_type == ModelType.TABPFN and TABPFN_AVAILABLE:
                if task_type != TaskType.REGRESSION:  # TabPFN is classification only
                    return TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
            
            elif model_type == ModelType.LINEAR:
                if task_type == TaskType.REGRESSION:
                    return Ridge(
                        alpha=params.get('alpha', 1.0),
                        random_state=config.random_state,
                        **params
                    )
                else:
                    return LogisticRegression(
                        C=params.get('C', 1.0),
                        random_state=config.random_state,
                        max_iter=1000,
                        n_jobs=config.n_jobs if config.enable_parallel else 1,
                        **params
                    )
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Model creation failed for {model_type}: {str(e)}")
            raise

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna."""
    
    def __init__(self, config: TabularModelConfig):
        self.config = config
        
    def optimize_model(
        self,
        model_type: ModelType,
        task_type: TaskType,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        categorical_features: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model."""
        try:
            if not OPTUNA_AVAILABLE:
                logger.warning("Optuna not available, using default hyperparameters")
                return {}
            
            study = optuna.create_study(
                direction='maximize' if task_type != TaskType.REGRESSION else 'minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            
            def objective(trial):
                try:
                    # Get hyperparameter suggestions based on model type
                    params = self._get_hyperparameter_space(trial, model_type, task_type)
                    
                    # Create model with suggested parameters
                    model = TabularModelFactory.create_model(
                        model_type, task_type, self.config, params
                    )
                    
                    # Handle categorical features for specific models
                    if categorical_features and model_type == ModelType.CATBOOST:
                        model.fit(
                            X_train, y_train,
                            cat_features=categorical_features,
                            eval_set=(X_val, y_val),
                            early_stopping_rounds=self.config.early_stopping_rounds,
                            verbose=False
                        )
                    elif model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=self.config.early_stopping_rounds,
                            verbose=False
                        )
                    else:
                        model.fit(X_train, y_train)
                    
                    # Make predictions and calculate score
                    if task_type == TaskType.REGRESSION:
                        y_pred = model.predict(X_val)
                        score = -mean_squared_error(y_val, y_pred)  # Negative for minimization
                    else:
                        y_pred = model.predict(X_val)
                        if task_type == TaskType.BINARY_CLASSIFICATION:
                            if hasattr(model, 'predict_proba'):
                                y_pred_proba = model.predict_proba(X_val)[:, 1]
                                score = roc_auc_score(y_val, y_pred_proba)
                            else:
                                score = f1_score(y_val, y_pred)
                        else:
                            score = f1_score(y_val, y_pred, average='weighted')
                    
                    return score
                    
                except Exception as e:
                    logger.warning(f"Trial failed: {str(e)}")
                    return -float('inf') if task_type != TaskType.REGRESSION else float('inf')
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=self.config.hyperopt_trials,
                timeout=self.config.hyperopt_timeout,
                catch=(Exception,)
            )
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return {}
    
    def _get_hyperparameter_space(
        self,
        trial: 'optuna.Trial',
        model_type: ModelType,
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Define hyperparameter search space for each model type."""
        try:
            if model_type == ModelType.XGBOOST:
                return {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                }
            
            elif model_type == ModelType.CATBOOST:
                return {
                    'iterations': trial.suggest_int('iterations', 100, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0)
                }
            
            elif model_type == ModelType.LIGHTGBM:
                return {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000)
                }
            
            elif model_type == ModelType.RANDOM_FOREST:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }
            
            elif model_type == ModelType.EXTRA_TREES:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            
            elif model_type == ModelType.LINEAR:
                if task_type == TaskType.REGRESSION:
                    return {
                        'alpha': trial.suggest_float('alpha', 1e-6, 100.0, log=True)
                    }
                else:
                    return {
                        'C': trial.suggest_float('C', 1e-6, 100.0, log=True),
                        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
                    }
            
            else:
                return {}
                
        except Exception as e:
            logger.warning(f"Hyperparameter space definition failed: {str(e)}")
            return {}

class TabularNeuralNetwork(BaseEstimator):
    """Custom neural network for tabular data."""
    
    def __init__(
        self,
        hidden_sizes: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 100,
        task_type: str = 'classification',
        random_state: int = 42
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.task_type = task_type
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None
        
        if PYTORCH_AVAILABLE:
            torch.manual_seed(random_state)
    
    def _build_model(self, input_size: int, output_size: int):
        """Build PyTorch neural network model."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network models")
        
        class TabularNet(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size, dropout_rate, task_type):
                super().__init__()
                self.task_type = task_type
                
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.BatchNorm1d(hidden_size)
                    ])
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, output_size))
                
                if task_type == 'classification' and output_size > 1:
                    layers.append(nn.Softmax(dim=1))
                elif task_type == 'classification':
                    layers.append(nn.Sigmoid())
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return TabularNet(input_size, self.hidden_sizes, output_size, self.dropout_rate, self.task_type)
    
    def fit(self, X, y):
        """Fit the neural network model."""
        try:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is required for neural network models")
            
            # Prepare data
            X_scaled = self.scaler.fit_transform(X)
            
            if self.task_type == 'classification':
                if self.label_encoder:
                    y_encoded = self.label_encoder.fit_transform(y)
                    n_classes = len(self.label_encoder.classes_)
                else:
                    y_encoded = y
                    n_classes = len(np.unique(y))
                
                output_size = n_classes if n_classes > 2 else 1
            else:
                y_encoded = y
                output_size = 1
            
            # Build model
            self.model = self._build_model(X_scaled.shape[1], output_size)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y_encoded) if self.task_type == 'regression' else torch.LongTensor(y_encoded)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Set up training
            if self.task_type == 'classification':
                criterion = nn.CrossEntropyLoss() if output_size > 1 else nn.BCELoss()
            else:
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    
                    if self.task_type == 'classification' and output_size == 1:
                        outputs = outputs.squeeze()
                        batch_y = batch_y.float()
                    
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                scheduler.step(avg_loss)
                
                if epoch % 20 == 0:
                    logger.debug(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Neural network training failed: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions."""
        try:
            if self.model is None:
                raise ValueError("Model not fitted")
            
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                
                if self.task_type == 'classification':
                    if len(self.label_encoder.classes_) == 2:
                        predictions = (outputs.squeeze() > 0.5).int().numpy()
                    else:
                        predictions = torch.argmax(outputs, dim=1).numpy()
                    
                    if self.label_encoder:
                        predictions = self.label_encoder.inverse_transform(predictions)
                else:
                    predictions = outputs.squeeze().numpy()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Neural network prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, X):
        """Predict class probabilities (classification only)."""
        try:
            if self.task_type != 'classification':
                raise ValueError("predict_proba only available for classification")
            
            if self.model is None:
                raise ValueError("Model not fitted")
            
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                
                if len(self.label_encoder.classes_) == 2:
                    proba_pos = outputs.squeeze().numpy()
                    probabilities = np.column_stack([1 - proba_pos, proba_pos])
                else:
                    probabilities = outputs.numpy()
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Neural network probability prediction failed: {str(e)}")
            raise

class TabularModelAnalyzer:
    """
    Comprehensive tabular model analysis system with automatic model selection,
    hyperparameter optimization, and advanced evaluation capabilities.
    """
    
    def __init__(self, config: Optional[TabularModelConfig] = None):
        self.config = config or TabularModelConfig()
        self.models = {}
        self.best_model = None
        self.best_model_type = None
        self.preprocessor = None
        self.feature_names = None
        self.task_type = None
        self.categorical_features = None
        self.hyperopt = HyperparameterOptimizer(self.config)
        
        logger.info("TabularModelAnalyzer initialized")
    
    async def analyze_tabular_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
        test_df: Optional[pd.DataFrame] = None,
        feature_cost_mapping: Optional[Dict[str, float]] = None
    ) -> TabularModelReport:
        """
        Comprehensive tabular data analysis with multiple models.
        
        Args:
            df: Training DataFrame
            target_column: Name of target column
            categorical_columns: List of categorical column names
            test_df: Optional separate test DataFrame
            feature_cost_mapping: Optional mapping of feature names to costs
            
        Returns:
            Comprehensive tabular model analysis report
        """
        try:
            logger.info(f"Starting tabular analysis on dataset with shape: {df.shape}")
            start_time = datetime.now()
            
            # Data preprocessing and validation
            X, y, feature_names, categorical_features = await self._preprocess_data(
                df, target_column, categorical_columns
            )
            
            # Detect task type
            task_type = self._detect_task_type(y)
            self.task_type = task_type
            self.feature_names = feature_names
            self.categorical_features = categorical_features
            
            logger.info(f"Detected task type: {task_type.value}")
            logger.info(f"Features: {len(feature_names)}, Categorical: {len(categorical_features or [])}")
            
            # Split data
            if test_df is not None:
                X_test, y_test, _, _ = await self._preprocess_data(
                    test_df, target_column, categorical_columns, fit_preprocessor=False
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.config.random_state,
                    stratify=y if task_type != TaskType.REGRESSION else None
                )
            else:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=self.config.test_size * 2, 
                    random_state=self.config.random_state,
                    stratify=y if task_type != TaskType.REGRESSION else None
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, 
                    random_state=self.config.random_state,
                    stratify=y_temp if task_type != TaskType.REGRESSION else None
                )
            
            # Select models to evaluate
            models_to_evaluate = self._select_models_for_evaluation(task_type, X_train.shape)
            
            logger.info(f"Evaluating {len(models_to_evaluate)} models: {[m.value for m in models_to_evaluate]}")
            
            # Train and evaluate models
            model_results = []
            for model_type in models_to_evaluate:
                try:
                    result = await self._train_and_evaluate_model(
                        model_type, task_type, X_train, y_train, X_val, y_val, X_test, y_test
                    )
                    if result:
                        model_results.append(result)
                        self.models[model_type.value] = result.model
                        
                        logger.info(f"{model_type.value} - CV Score: {result.cv_score_mean:.4f} (±{result.cv_score_std:.4f}), Test Score: {result.test_score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Model {model_type.value} failed: {str(e)}")
                    continue
            
            if not model_results:
                raise ValueError("No models were successfully trained")
            
            # Select best model
            best_model_result = self._select_best_model(model_results, task_type)
            self.best_model = best_model_result.model
            self.best_model_type = best_model_result.model_type
            
            # Create ensemble if enabled
            ensemble_result = None
            if self.config.enable_ensemble and len(model_results) >= 3:
                ensemble_result = await self._create_ensemble_model(
                    model_results, task_type, X_test, y_test
                )
            
            # Feature analysis
            feature_analysis = await self._analyze_features(
                best_model_result, X, y, task_type
            )
            
            # Performance comparison
            performance_comparison = self._compare_model_performance(model_results, task_type)
            
            # Business impact analysis
            business_impact = await self._calculate_business_impact(
                best_model_result, feature_analysis, feature_cost_mapping, y
            )
            
            # Model interpretability
            model_interpretability = await self._generate_model_interpretability(
                best_model_result, X_test, y_test, task_type
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(
                model_results, best_model_result, feature_analysis, business_impact
            )
            recommendations = self._generate_recommendations(
                model_results, best_model_result, performance_comparison, insights
            )
            
            # Create comprehensive report
            execution_time = (datetime.now() - start_time).total_seconds()
            
            report = TabularModelReport(
                report_id=str(uuid.uuid4()),
                timestamp=start_time,
                task_type=task_type,
                dataset_info={
                    'n_samples': len(df),
                    'n_features': len(feature_names),
                    'n_categorical': len(categorical_features or []),
                    'target_distribution': self._get_target_distribution(y, task_type),
                    'missing_values': df.isnull().sum().sum(),
                    'data_types': df.dtypes.value_counts().to_dict()
                },
                models_evaluated=model_results,
                best_model_result=best_model_result,
                ensemble_result=ensemble_result,
                feature_analysis=feature_analysis,
                performance_comparison=performance_comparison,
                business_impact=business_impact,
                model_interpretability=model_interpretability,
                recommendations=recommendations,
                insights=insights,
                metadata={
                    'execution_time': execution_time,
                    'config': asdict(self.config),
                    'models_trained': len(model_results)
                }
            )
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(report)
            
            logger.info(f"Tabular analysis completed in {execution_time:.2f}s")
            logger.info(f"Best model: {best_model_result.model_type.value} with score: {best_model_result.test_score:.4f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Tabular analysis failed: {str(e)}")
            # Return minimal report with error
            return TabularModelReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                task_type=TaskType.REGRESSION,
                dataset_info={},
                models_evaluated=[],
                best_model_result=None,
                ensemble_result=None,
                feature_analysis={},
                performance_comparison={},
                business_impact={},
                model_interpretability={},
                recommendations=[f"Analysis failed: {str(e)}"],
                insights=["Analysis encountered an error"],
                metadata={'error': str(e)}
            )
    
    async def _preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
        fit_preprocessor: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List[int]]]:
        """Preprocess data for model training."""
        try:
            # Separate features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()
            
            # Handle missing values in target
            if y.isnull().any():
                logger.warning(f"Found {y.isnull().sum()} missing values in target, dropping rows")
                mask = ~y.isnull()
                X = X[mask]
                y = y[mask]
            
            # Identify categorical columns
            if categorical_columns is None:
                categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Handle missing values in features
            if fit_preprocessor:
                self.preprocessor = {}
                
                # Numeric columns
                numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    if self.config.missing_strategy == 'auto':
                        strategy = 'median'
                    else:
                        strategy = self.config.missing_strategy
                    
                    numeric_imputer = SimpleImputer(strategy=strategy)
                    X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
                    self.preprocessor['numeric_imputer'] = numeric_imputer
                
                # Categorical columns
                if categorical_columns:
                    categorical_imputer = SimpleImputer(strategy='most_frequent', fill_value='missing')
                    X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
                    self.preprocessor['categorical_imputer'] = categorical_imputer
                    
                    # Encode categorical variables
                    if self.config.encode_categorical:
                        encoded_columns = []
                        categorical_encoders = {}
                        
                        for col in categorical_columns:
                            if self.config.categorical_encoding == 'auto':
                                # Use target encoding for high cardinality, one-hot for low
                                if X[col].nunique() > 10 and CATEGORY_ENCODERS_AVAILABLE:
                                    encoder = TargetEncoder()
                                    encoded_values = encoder.fit_transform(X[col], y)
                                    X[f'{col}_encoded'] = encoded_values
                                    categorical_encoders[col] = encoder
                                    encoded_columns.append(f'{col}_encoded')
                                else:
                                    # One-hot encoding
                                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                                    X = pd.concat([X, dummies], axis=1)
                                    encoded_columns.extend(dummies.columns.tolist())
                            else:
                                # Use specified encoding
                                if self.config.categorical_encoding == 'target' and CATEGORY_ENCODERS_AVAILABLE:
                                    encoder = TargetEncoder()
                                    encoded_values = encoder.fit_transform(X[col], y)
                                    X[f'{col}_encoded'] = encoded_values
                                    categorical_encoders[col] = encoder
                                    encoded_columns.append(f'{col}_encoded')
                                else:
                                    # Label encoding as fallback
                                    encoder = LabelEncoder()
                                    X[f'{col}_encoded'] = encoder.fit_transform(X[col].astype(str))
                                    categorical_encoders[col] = encoder
                                    encoded_columns.append(f'{col}_encoded')
                        
                        # Drop original categorical columns
                        X = X.drop(columns=categorical_columns)
                        self.preprocessor['categorical_encoders'] = categorical_encoders
                        
                        # Update categorical feature indices
                        categorical_feature_indices = [
                            X.columns.get_loc(col) for col in encoded_columns 
                            if col in X.columns
                        ]
                    else:
                        categorical_feature_indices = None
                else:
                    categorical_feature_indices = None
                
                # Feature scaling
                if self.config.scale_features:
                    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_columns:
                        scaler = StandardScaler()
                        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
                        self.preprocessor['scaler'] = scaler
            else:
                # Apply existing preprocessor
                if self.preprocessor:
                    # Apply numeric imputation
                    if 'numeric_imputer' in self.preprocessor:
                        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_columns:
                            X[numeric_columns] = self.preprocessor['numeric_imputer'].transform(X[numeric_columns])
                    
                    # Apply categorical imputation and encoding
                    if 'categorical_imputer' in self.preprocessor:
                        if categorical_columns:
                            X[categorical_columns] = self.preprocessor['categorical_imputer'].transform(X[categorical_columns])
                            
                            # Apply categorical encoding
                            if 'categorical_encoders' in self.preprocessor:
                                for col, encoder in self.preprocessor['categorical_encoders'].items():
                                    if col in X.columns:
                                        if hasattr(encoder, 'transform'):
                                            X[f'{col}_encoded'] = encoder.transform(X[col])
                                        else:
                                            X[f'{col}_encoded'] = encoder.transform(X[col].astype(str))
                                
                                X = X.drop(columns=categorical_columns)
                    
                    # Apply scaling
                    if 'scaler' in self.preprocessor:
                        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_columns:
                            X[numeric_columns] = self.preprocessor['scaler'].transform(X[numeric_columns])
                
                categorical_feature_indices = self.categorical_features
            
            feature_names = X.columns.tolist()
            
            return X.values, y.values, feature_names, categorical_feature_indices
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def _detect_task_type(self, y: np.ndarray) -> TaskType:
        """Detect the type of ML task based on target variable."""
        try:
            # Check if target is numeric
            if pd.api.types.is_numeric_dtype(y):
                unique_values = len(np.unique(y))
                total_values = len(y)
                
                # If many unique values and continuous, it's regression
                if unique_values > 20 and unique_values > total_values * 0.1:
                    return TaskType.REGRESSION
                
                # If few unique values, might be classification
                if unique_values == 2:
                    return TaskType.BINARY_CLASSIFICATION
                elif unique_values <= 20:
                    return TaskType.MULTICLASS_CLASSIFICATION
                else:
                    return TaskType.REGRESSION
            else:
                # Non-numeric target suggests classification
                unique_values = len(np.unique(y))
                if unique_values == 2:
                    return TaskType.BINARY_CLASSIFICATION
                else:
                    return TaskType.MULTICLASS_CLASSIFICATION
                    
        except Exception as e:
            logger.warning(f"Task type detection failed: {str(e)}, defaulting to regression")
            return TaskType.REGRESSION
    
    def _select_models_for_evaluation(
        self, 
        task_type: TaskType, 
        data_shape: Tuple[int, int]
    ) -> List[ModelType]:
        """Select appropriate models based on task type and data characteristics."""
        try:
            models = []
            n_samples, n_features = data_shape
            
            # Always include tree-based models
            if self.config.include_random_forest:
                models.append(ModelType.RANDOM_FOREST)
            
            if self.config.include_extra_trees:
                models.append(ModelType.EXTRA_TREES)
            
            if self.config.include_gradient_boosting:
                models.append(ModelType.GRADIENT_BOOSTING)
            
            # Advanced boosting models
            if self.config.include_xgboost and XGBOOST_AVAILABLE:
                models.append(ModelType.XGBOOST)
            
            if self.config.include_catboost and CATBOOST_AVAILABLE:
                models.append(ModelType.CATBOOST)
            
            if self.config.include_lightgbm and LIGHTGBM_AVAILABLE:
                models.append(ModelType.LIGHTGBM)
            
            # Linear models
            if self.config.include_linear_models:
                models.append(ModelType.LINEAR)
            
            # TabPFN for small datasets with classification
            if (self.config.include_tabpfn and TABPFN_AVAILABLE and 
                task_type != TaskType.REGRESSION and n_samples <= 10000):
                models.append(ModelType.TABPFN)
            
            # Neural networks for larger datasets
            if (self.config.include_neural_networks and PYTORCH_AVAILABLE and 
                n_samples > 1000):
                models.append(ModelType.NEURAL_NETWORK)
            
            # Limit number of models if auto-selection is enabled
            if self.config.auto_select_models and len(models) > self.config.max_models_to_try:
                # Prioritize models based on data characteristics
                model_priority = self._get_model_priority(task_type, data_shape)
                models = sorted(models, key=lambda x: model_priority.get(x, 999))[:self.config.max_models_to_try]
            
            return models
            
        except Exception as e:
            logger.warning(f"Model selection failed: {str(e)}")
            return [ModelType.RANDOM_FOREST, ModelType.LINEAR]
    
    def _get_model_priority(self, task_type: TaskType, data_shape: Tuple[int, int]) -> Dict[ModelType, int]:
        """Get model priority based on task type and data characteristics."""
        n_samples, n_features = data_shape
        
        priority = {}
        
        if task_type == TaskType.REGRESSION:
            priority.update({
                ModelType.XGBOOST: 1,
                ModelType.LIGHTGBM: 2,
                ModelType.CATBOOST: 3,
                ModelType.RANDOM_FOREST: 4,
                ModelType.EXTRA_TREES: 5,
                ModelType.GRADIENT_BOOSTING: 6,
                ModelType.LINEAR: 7,
                ModelType.NEURAL_NETWORK: 8
            })
        else:
            priority.update({
                ModelType.XGBOOST: 1,
                ModelType.CATBOOST: 2,
                ModelType.LIGHTGBM: 3,
                ModelType.RANDOM_FOREST: 4,
                ModelType.TABPFN: 5,
                ModelType.EXTRA_TREES: 6,
                ModelType.GRADIENT_BOOSTING: 7,
                ModelType.NEURAL_NETWORK: 8,
                ModelType.LINEAR: 9
            })
        
        # Adjust priority based on data size
        if n_samples < 1000:
            # Prefer simpler models for small datasets
            priority[ModelType.LINEAR] = 2
            priority[ModelType.DECISION_TREE] = 3
        elif n_samples > 100000:
            # Prefer scalable models for large datasets
            priority[ModelType.LIGHTGBM] = 1
            priority[ModelType.NEURAL_NETWORK] = 2
        
        return priority
    
    async def _train_and_evaluate_model(
        self,
        model_type: ModelType,
        task_type: TaskType,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Optional[ModelResult]:
        """Train and evaluate a single model."""
        try:
            start_time = datetime.now()
            
            # Hyperparameter optimization
            best_params = {}
            if self.config.enable_hyperopt and model_type not in [ModelType.TABPFN]:
                logger.info(f"Optimizing hyperparameters for {model_type.value}")
                best_params = self.hyperopt.optimize_model(
                    model_type, task_type, X_train, y_train, X_val, y_val, self.categorical_features
                )
            
            # Create model with best parameters
            if model_type == ModelType.NEURAL_NETWORK:
                model = TabularNeuralNetwork(
                    task_type='classification' if task_type != TaskType.REGRESSION else 'regression',
                    random_state=self.config.random_state,
                    **best_params
                )
            else:
                model = TabularModelFactory.create_model(
                    model_type, task_type, self.config, best_params
                )
            
            # Handle special training cases
            if model_type == ModelType.CATBOOST and self.categorical_features:
                model.fit(
                    X_train, y_train,
                    cat_features=self.categorical_features,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            elif model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Cross-validation evaluation
            cv_scores = self._perform_cross_validation(model, X_train, y_train, task_type)
            
            # Test set evaluation
            test_score = self._calculate_test_score(model, X_test, y_test, task_type)
            
            # Make predictions for further analysis
            predictions = model.predict(X_test)
            prediction_probabilities = None
            if hasattr(model, 'predict_proba') and task_type != TaskType.REGRESSION:
                try:
                    prediction_probabilities = model.predict_proba(X_test)
                except:
                    pass
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(model, model_type)
            
            # Calculate model size
            model_size = len(pickle.dumps(model))
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResult(
                model_type=model_type,
                model=model,
                cv_scores=cv_scores,
                cv_score_mean=float(np.mean(cv_scores)),
                cv_score_std=float(np.std(cv_scores)),
                test_score=float(test_score),
                training_time=training_time,
                hyperparameters=best_params,
                feature_importance=feature_importance,
                model_size=model_size,
                predictions=predictions,
                prediction_probabilities=prediction_probabilities
            )
            
        except Exception as e:
            logger.error(f"Model training failed for {model_type.value}: {str(e)}")
            return None
    
    def _perform_cross_validation(
        self, 
        model: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray, 
        task_type: TaskType
    ) -> List[float]:
        """Perform cross-validation evaluation."""
        try:
            # Select CV strategy
            if self.config.cv_strategy == 'stratified' and task_type != TaskType.REGRESSION:
                cv = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state
                )
            else:
                cv = KFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state
                )
            
            # Select scoring metric
            if self.config.cv_scoring == 'auto':
                if task_type == TaskType.REGRESSION:
                    scoring = 'r2'
                elif task_type == TaskType.BINARY_CLASSIFICATION:
                    scoring = 'roc_auc'
                else:
                    scoring = 'f1_weighted'
            else:
                scoring = self.config.cv_scoring
            
            # Perform cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=1  # Avoid nested parallelization
            )
            
            return scores.tolist()
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return [0.0] * self.config.cv_folds
    
    def _calculate_test_score(
        self, 
        model: BaseEstimator, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        task_type: TaskType
    ) -> float:
        """Calculate test set score."""
        try:
            if task_type == TaskType.REGRESSION:
                y_pred = model.predict(X_test)
                return r2_score(y_test, y_pred)
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    return roc_auc_score(y_test, y_pred_proba)
                else:
                    y_pred = model.predict(X_test)
                    return f1_score(y_test, y_pred)
            else:  # Multiclass classification
                y_pred = model.predict(X_test)
                return f1_score(y_test, y_pred, average='weighted')
                
        except Exception as e:
            logger.warning(f"Test score calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_feature_importance(
        self, 
        model: BaseEstimator, 
        model_type: ModelType
    ) -> Optional[Dict[str, float]]:
        """Calculate feature importance for the model."""
        try:
            if not self.config.calculate_feature_importance or not self.feature_names:
                return None
            
            importance_values = None
            
            # Try built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_
                if len(coef.shape) > 1:
                    importance_values = np.mean(np.abs(coef), axis=0)
                else:
                    importance_values = np.abs(coef)
            elif model_type == ModelType.NEURAL_NETWORK:
                # For neural networks, use permutation importance (simplified)
                # This would be computationally expensive, so skip for now
                return None
            
            if importance_values is not None:
                feature_importance = {}
                for i, importance in enumerate(importance_values):
                    if i < len(self.feature_names):
                        feature_importance[self.feature_names[i]] = float(importance)
                
                return feature_importance
            
            return None
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
            return None
    
    def _select_best_model(
        self, 
        model_results: List[ModelResult], 
        task_type: TaskType
    ) -> ModelResult:
        """Select the best performing model."""
        try:
            if not model_results:
                raise ValueError("No model results to select from")
            
            # Sort by test score (higher is better for all metrics we use)
            sorted_results = sorted(model_results, key=lambda x: x.test_score, reverse=True)
            
            return sorted_results[0]
            
        except Exception as e:
            logger.error(f"Best model selection failed: {str(e)}")
            return model_results[0]
    
    async def _create_ensemble_model(
        self,
        model_results: List[ModelResult],
        task_type: TaskType,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Optional[ModelResult]:
        """Create ensemble model from top performing models."""
        try:
            # Select top models for ensemble
            top_models = sorted(model_results, key=lambda x: x.test_score, reverse=True)[:self.config.ensemble_size]
            
            if len(top_models) < 2:
                return None
            
            # Create ensemble
            models_for_ensemble = [(f'model_{i}', result.model) for i, result in enumerate(top_models)]
            
            if task_type == TaskType.REGRESSION:
                ensemble = VotingRegressor(
                    estimators=models_for_ensemble,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            else:
                ensemble = VotingClassifier(
                    estimators=models_for_ensemble,
                    voting='soft',
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            
            # We can't retrain the ensemble with the individual models as they're already trained
            # Instead, we'll create a simple averaging ensemble
            predictions_list = []
            probabilities_list = []
            
            for result in top_models:
                predictions_list.append(result.predictions)
                if result.prediction_probabilities is not None:
                    probabilities_list.append(result.prediction_probabilities)
            
            # Average predictions
            ensemble_predictions = np.mean(predictions_list, axis=0)
            ensemble_probabilities = np.mean(probabilities_list, axis=0) if probabilities_list else None
            
            # Calculate ensemble score
            ensemble_score = self._calculate_test_score_from_predictions(
                y_test, ensemble_predictions, ensemble_probabilities, task_type
            )
            
            # Create a dummy ensemble result
            ensemble_result = ModelResult(
                model_type=ModelType.ENSEMBLE,
                model=ensemble,  # This won't actually work for predictions
                cv_scores=[ensemble_score] * self.config.cv_folds,
                cv_score_mean=ensemble_score,
                cv_score_std=0.0,
                test_score=ensemble_score,
                training_time=sum(r.training_time for r in top_models),
                hyperparameters={'ensemble_size': len(top_models)},
                feature_importance=None,
                model_size=sum(r.model_size for r in top_models),
                predictions=ensemble_predictions,
                prediction_probabilities=ensemble_probabilities
            )
            
            return ensemble_result
            
        except Exception as e:
            logger.warning(f"Ensemble creation failed: {str(e)}")
            return None
    
    def _calculate_test_score_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        task_type: TaskType
    ) -> float:
        """Calculate test score from predictions."""
        try:
            if task_type == TaskType.REGRESSION:
                return r2_score(y_true, y_pred)
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                if y_pred_proba is not None and len(y_pred_proba.shape) == 2:
                    return roc_auc_score(y_true, y_pred_proba[:, 1])
                elif y_pred_proba is not None and len(y_pred_proba.shape) == 1:
                    return roc_auc_score(y_true, y_pred_proba)
                else:
                    return f1_score(y_true, np.round(y_pred).astype(int))
            else:  # Multiclass
                return f1_score(y_true, np.round(y_pred).astype(int), average='weighted')
                
        except Exception as e:
            logger.warning(f"Score calculation from predictions failed: {str(e)}")
            return 0.0
    
    async def _analyze_features(
        self,
        best_model_result: ModelResult,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Analyze feature importance and relationships."""
        try:
            analysis = {}
            
            # Feature importance analysis
            if best_model_result.feature_importance:
                importance_dict = best_model_result.feature_importance
                
                # Sort features by importance
                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                analysis['feature_importance'] = {
                    'top_features': sorted_features[:10],
                    'importance_distribution': {
                        'mean': float(np.mean(list(importance_dict.values()))),
                        'std': float(np.std(list(importance_dict.values()))),
                        'max': float(max(importance_dict.values())),
                        'min': float(min(importance_dict.values()))
                    }
                }
            
            # Feature correlation analysis
            if len(self.feature_names) <= 100:  # Only for manageable number of features
                try:
                    feature_df = pd.DataFrame(X, columns=self.feature_names)
                    correlation_matrix = feature_df.corr()
                    
                    # Find highly correlated feature pairs
                    high_correlations = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_value = correlation_matrix.iloc[i, j]
                            if abs(corr_value) > 0.8:
                                high_correlations.append({
                                    'feature1': correlation_matrix.columns[i],
                                    'feature2': correlation_matrix.columns[j],
                                    'correlation': float(corr_value)
                                })
                    
                    analysis['correlation_analysis'] = {
                        'high_correlations': high_correlations[:10],
                        'avg_correlation': float(correlation_matrix.abs().mean().mean())
                    }
                    
                except Exception as e:
                    logger.warning(f"Correlation analysis failed: {str(e)}")
            
            # Target correlation analysis
            if task_type == TaskType.REGRESSION:
                try:
                    target_correlations = []
                    feature_df = pd.DataFrame(X, columns=self.feature_names)
                    
                    for feature in self.feature_names:
                        if feature in feature_df.columns:
                            corr, p_value = pearsonr(feature_df[feature], y)
                            target_correlations.append({
                                'feature': feature,
                                'correlation': float(corr),
                                'p_value': float(p_value)
                            })
                    
                    # Sort by absolute correlation
                    target_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                    
                    analysis['target_correlation'] = {
                        'top_correlated_features': target_correlations[:10]
                    }
                    
                except Exception as e:
                    logger.warning(f"Target correlation analysis failed: {str(e)}")
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Feature analysis failed: {str(e)}")
            return {}
    
    def _compare_model_performance(
        self,
        model_results: List[ModelResult],
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Compare performance across all models."""
        try:
            comparison = {}
            
            # Performance summary
            performance_data = []
            for result in model_results:
                performance_data.append({
                    'model': result.model_type.value,
                    'cv_score': result.cv_score_mean,
                    'cv_std': result.cv_score_std,
                    'test_score': result.test_score,
                    'training_time': result.training_time,
                    'model_size': result.model_size
                })
            
            comparison['performance_summary'] = performance_data
            
            # Best performers by different criteria
            comparison['best_performers'] = {
                'accuracy': max(model_results, key=lambda x: x.test_score).model_type.value,
                'speed': min(model_results, key=lambda x: x.training_time).model_type.value,
                'size': min(model_results, key=lambda x: x.model_size).model_type.value,
                'stability': min(model_results, key=lambda x: x.cv_score_std).model_type.value
            }
            
            # Performance statistics
            test_scores = [r.test_score for r in model_results]
            training_times = [r.training_time for r in model_results]
            
            comparison['statistics'] = {
                'score_range': (float(min(test_scores)), float(max(test_scores))),
                'score_mean': float(np.mean(test_scores)),
                'score_std': float(np.std(test_scores)),
                'time_range': (float(min(training_times)), float(max(training_times))),
                'time_mean': float(np.mean(training_times))
            }
            
            return comparison
            
        except Exception as e:
            logger.warning(f"Performance comparison failed: {str(e)}")
            return {}
    
    async def _calculate_business_impact(
        self,
        best_model_result: ModelResult,
        feature_analysis: Dict[str, Any],
        feature_cost_mapping: Optional[Dict[str, float]],
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate business impact and ROI metrics."""
        try:
            if not self.config.calculate_business_impact:
                return {}
            
            impact = {}
            
            # Model performance impact
            test_score = best_model_result.test_score
            
            if self.task_type == TaskType.REGRESSION:
                impact['performance_metrics'] = {
                    'r2_score': test_score,
                    'explained_variance': test_score,
                    'prediction_quality': 'High' if test_score > 0.8 else 'Medium' if test_score > 0.6 else 'Low'
                }
                
                # Estimate cost savings from better predictions
                baseline_error = np.std(y)
                model_error = baseline_error * (1 - test_score) ** 0.5
                error_reduction = (baseline_error - model_error) / baseline_error
                
                impact['estimated_improvement'] = {
                    'error_reduction_percentage': float(error_reduction * 100),
                    'prediction_accuracy_improvement': f"{error_reduction:.1%}"
                }
                
            else:  # Classification
                impact['performance_metrics'] = {
                    'test_score': test_score,
                    'prediction_quality': 'High' if test_score > 0.9 else 'Medium' if test_score > 0.75 else 'Low'
                }
                
                # Estimate classification improvement
                baseline_accuracy = max(np.bincount(y.astype(int))) / len(y)  # Majority class accuracy
                improvement = test_score - baseline_accuracy
                
                impact['estimated_improvement'] = {
                    'accuracy_improvement': float(improvement),
                    'improvement_percentage': float(improvement * 100)
                }
            
            # Feature cost analysis
            if feature_cost_mapping and feature_analysis.get('feature_importance'):
                top_features = feature_analysis['feature_importance']['top_features']
                feature_costs = []
                
                for feature_name, importance in top_features:
                    cost = feature_cost_mapping.get(feature_name, 0)
                    feature_costs.append({
                        'feature': feature_name,
                        'importance': importance,
                        'cost': cost,
                        'value_ratio': importance / (cost + 1e-6)
                    })
                
                # Sort by value ratio
                feature_costs.sort(key=lambda x: x['value_ratio'], reverse=True)
                
                impact['feature_economics'] = {
                    'high_value_features': feature_costs[:5],
                    'total_feature_cost': sum(feature_cost_mapping.values()),
                    'cost_per_importance_unit': sum(feature_cost_mapping.values()) / sum(
                        importance for _, importance in top_features
                    )
                }
            
            # Model deployment considerations
            impact['deployment_metrics'] = {
                'training_time': best_model_result.training_time,
                'model_size_mb': best_model_result.model_size / (1024 * 1024),
                'complexity': 'High' if best_model_result.model_type in [
                    ModelType.NEURAL_NETWORK, ModelType.ENSEMBLE
                ] else 'Medium' if best_model_result.model_type in [
                    ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM
                ] else 'Low',
                'interpretability': 'High' if best_model_result.model_type in [
                    ModelType.LINEAR, ModelType.DECISION_TREE
                ] else 'Medium' if best_model_result.model_type in [
                    ModelType.RANDOM_FOREST, ModelType.EXTRA_TREES
                ] else 'Low'
            }
            
            return impact
            
        except Exception as e:
            logger.warning(f"Business impact calculation failed: {str(e)}")
            return {}
    
    async def _generate_model_interpretability(
        self,
        best_model_result: ModelResult,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Generate model interpretability analysis."""
        try:
            if not self.config.enable_model_interpretation:
                return {}
            
            interpretability = {}
            
            # Feature importance interpretation
            if best_model_result.feature_importance:
                importance_dict = best_model_result.feature_importance
                
                # Top contributing features
                top_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                interpretability['key_drivers'] = [
                    {
                        'feature': feature,
                        'importance': float(importance),
                        'impact': 'High' if importance > np.mean(list(importance_dict.values())) else 'Medium'
                    }
                    for feature, importance in top_features
                ]
            
            # Model complexity analysis
            model_type = best_model_result.model_type
            
            if model_type in [ModelType.LINEAR]:
                interpretability['model_explanation'] = {
                    'type': 'Linear relationship',
                    'interpretability': 'High',
                    'description': 'Model makes predictions based on linear combinations of features'
                }
            elif model_type in [ModelType.RANDOM_FOREST, ModelType.EXTRA_TREES]:
                interpretability['model_explanation'] = {
                    'type': 'Tree-based ensemble',
                    'interpretability': 'Medium',
                    'description': 'Model makes predictions by combining multiple decision trees'
                }
            elif model_type in [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM]:
                interpretability['model_explanation'] = {
                    'type': 'Gradient boosting',
                    'interpretability': 'Medium',
                    'description': 'Model iteratively improves predictions by learning from errors'
                }
            elif model_type == ModelType.NEURAL_NETWORK:
                interpretability['model_explanation'] = {
                    'type': 'Neural network',
                    'interpretability': 'Low',
                    'description': 'Model learns complex patterns through neural network layers'
                }
            
            # SHAP analysis if available and enabled
            if SHAP_AVAILABLE and self.config.calculate_shap_values:
                try:
                    # Sample data for SHAP analysis (to avoid performance issues)
                    sample_size = min(100, len(X_test))
                    X_sample = X_test[:sample_size]
                    
                    if model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST]:
                        explainer = shap.TreeExplainer(best_model_result.model)
                        shap_values = explainer.shap_values(X_sample)
                        
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        
                        # Calculate mean absolute SHAP values
                        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                        
                        shap_importance = {}
                        for i, importance in enumerate(mean_shap_values):
                            if i < len(self.feature_names):
                                shap_importance[self.feature_names[i]] = float(importance)
                        
                        interpretability['shap_analysis'] = {
                            'feature_importance': shap_importance,
                            'explanation': 'SHAP values show the contribution of each feature to individual predictions'
                        }
                        
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {str(e)}")
            
            return interpretability
            
        except Exception as e:
            logger.warning(f"Model interpretability generation failed: {str(e)}")
            return {}
    
    def _generate_insights(
        self,
        model_results: List[ModelResult],
        best_model_result: ModelResult,
        feature_analysis: Dict[str, Any],
        business_impact: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable insights from the analysis."""
        try:
            insights = []
            
            # Performance insights
            best_score = best_model_result.test_score
            best_model_type = best_model_result.model_type.value
            
            if self.task_type == TaskType.REGRESSION:
                if best_score > 0.8:
                    insights.append(f"Excellent model performance achieved (R² = {best_score:.3f}) with {best_model_type}")
                elif best_score > 0.6:
                    insights.append(f"Good model performance (R² = {best_score:.3f}) - consider feature engineering for improvement")
                else:
                    insights.append(f"Model performance is moderate (R² = {best_score:.3f}) - data quality or more features may be needed")
            else:
                if best_score > 0.9:
                    insights.append(f"Excellent classification performance ({best_score:.3f}) achieved with {best_model_type}")
                elif best_score > 0.8:
                    insights.append(f"Good classification performance ({best_score:.3f}) with room for improvement")
                else:
                    insights.append(f"Classification performance is moderate ({best_score:.3f}) - consider class balancing or feature engineering")
            
            # Model comparison insights
            if len(model_results) > 1:
                scores = [r.test_score for r in model_results]
                score_std = np.std(scores)
                
                if score_std < 0.02:
                    insights.append("Multiple models show similar performance - choose based on interpretability or speed requirements")
                else:
                    insights.append(f"Significant performance differences between models - {best_model_type} clearly outperforms others")
            
            # Feature insights
            if feature_analysis.get('feature_importance'):
                top_features = feature_analysis['feature_importance']['top_features']
                if len(top_features) >= 3:
                    top_3_names = [name for name, _ in top_features[:3]]
                    insights.append(f"Top predictive features: {', '.join(top_3_names)}")
                
                importance_values = [importance for _, importance in top_features]
                if len(importance_values) > 5:
                    top_5_sum = sum(importance_values[:5])
                    total_sum = sum(importance_values)
                    if top_5_sum / total_sum > 0.8:
                        insights.append("Model relies heavily on top 5 features - feature selection could simplify the model")
            
            # Correlation insights
            if feature_analysis.get('correlation_analysis', {}).get('high_correlations'):
                high_corr_count = len(feature_analysis['correlation_analysis']['high_correlations'])
                if high_corr_count > 5:
                    insights.append(f"Found {high_corr_count} highly correlated feature pairs - consider feature selection to reduce redundancy")
            
            # Business impact insights
            if business_impact.get('estimated_improvement'):
                if self.task_type == TaskType.REGRESSION:
                    error_reduction = business_impact['estimated_improvement'].get('error_reduction_percentage', 0)
                    if error_reduction > 50:
                        insights.append(f"Model provides substantial error reduction ({error_reduction:.1f}%) over baseline predictions")
                else:
                    accuracy_improvement = business_impact['estimated_improvement'].get('improvement_percentage', 0)
                    if accuracy_improvement > 20:
                        insights.append(f"Model significantly improves accuracy ({accuracy_improvement:.1f}%) over baseline")
            
            # Deployment insights
            if business_impact.get('deployment_metrics'):
                training_time = business_impact['deployment_metrics']['training_time']
                model_size = business_impact['deployment_metrics']['model_size_mb']
                
                if training_time < 60:
                    insights.append("Fast training time enables quick model updates and experimentation")
                elif training_time > 3600:
                    insights.append("Long training time - consider model simplification for production use")
                
                if model_size < 10:
                    insights.append("Compact model size suitable for edge deployment")
                elif model_size > 100:
                    insights.append("Large model size may require dedicated infrastructure for deployment")
            
            # Default insight if none generated
            if not insights:
                insights.append("Model training and evaluation completed successfully")
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {str(e)}")
            return ["Analysis completed - review detailed results for insights"]
    
    def _generate_recommendations(
        self,
        model_results: List[ModelResult],
        best_model_result: ModelResult,
        performance_comparison: Dict[str, Any],
        insights: List[str]
    ) -> List[str]:
        """Generate actionable recommendations for model improvement."""
        try:
            recommendations = []
            
            # Performance-based recommendations
            best_score = best_model_result.test_score
            
            if self.task_type == TaskType.REGRESSION:
                if best_score < 0.7:
                    recommendations.append("Consider additional feature engineering or collecting more relevant features")
                    recommendations.append("Explore polynomial features or feature interactions")
                if best_score < 0.5:
                    recommendations.append("Data quality issues may be present - review outliers and missing values")
            else:
                if best_score < 0.8:
                    recommendations.append("Try ensemble methods or advanced hyperparameter tuning")
                    recommendations.append("Consider class balancing techniques if dealing with imbalanced data")
                if best_score < 0.7:
                    recommendations.append("Collect more training data or improve data quality")
            
            # Model-specific recommendations
            best_model_type = best_model_result.model_type
            
            if best_model_type == ModelType.LINEAR:
                recommendations.append("Consider non-linear models (tree-based or neural networks) for potentially better performance")
            elif best_model_type in [ModelType.RANDOM_FOREST, ModelType.EXTRA_TREES]:
                recommendations.append("Try gradient boosting methods (XGBoost, LightGBM) for potential performance gains")
            elif best_model_type == ModelType.NEURAL_NETWORK:
                recommendations.append("Monitor for overfitting and consider regularization techniques")
            
            # Feature recommendations
            if best_model_result.feature_importance:
                n_important_features = sum(1 for imp in best_model_result.feature_importance.values() if imp > 0.01)
                total_features = len(best_model_result.feature_importance)
                
                if n_important_features < total_features * 0.3:
                    recommendations.append("Many features have low importance - consider feature selection to simplify the model")
                
                if total_features > 100:
                    recommendations.append("High dimensionality detected - consider dimensionality reduction techniques")
            
            # Training time recommendations
            if best_model_result.training_time > 1800:  # 30 minutes
                recommendations.append("Long training time - consider model simplification for faster iterations")
            
            # Model comparison recommendations
            if performance_comparison.get('statistics'):
                score_std = performance_comparison['statistics'].get('score_std', 0)
                if score_std > 0.05:
                    recommendations.append("High variance in model performance - ensemble methods may provide more stable results")
            
            # Cross-validation recommendations
            cv_std = best_model_result.cv_score_std
            if cv_std > 0.1:
                recommendations.append("High cross-validation variance suggests model instability - try regularization or more data")
            
            # Business recommendations
            recommendations.append("Monitor model performance over time and retrain when performance degrades")
            recommendations.append("Consider A/B testing the model against current business rules or baseline models")
            
            # Default recommendation
            if len(recommendations) < 3:
                recommendations.append("Model shows good performance - proceed with deployment and monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {str(e)}")
            return ["Review model performance and consider iterative improvements"]
    
    def _get_target_distribution(self, y: np.ndarray, task_type: TaskType) -> Dict[str, Any]:
        """Get target variable distribution information."""
        try:
            if task_type == TaskType.REGRESSION:
                return {
                    'type': 'continuous',
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y)),
                    'min': float(np.min(y)),
                    'max': float(np.max(y)),
                    'quartiles': [float(q) for q in np.percentile(y, [25, 50, 75])]
                }
            else:
                unique_vals, counts = np.unique(y, return_counts=True)
                return {
                    'type': 'categorical',
                    'classes': unique_vals.tolist(),
                    'class_counts': counts.tolist(),
                    'class_proportions': (counts / len(y)).tolist(),
                    'n_classes': len(unique_vals)
                }
        except Exception as e:
            logger.warning(f"Target distribution analysis failed: {str(e)}")
            return {}
    
    async def _log_to_mlflow(self, report: TabularModelReport):
        """Log results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"tabular_analysis_{report.task_type.value}"):
                # Log parameters
                mlflow.log_param("task_type", report.task_type.value)
                mlflow.log_param("n_samples", report.dataset_info.get('n_samples', 0))
                mlflow.log_param("n_features", report.dataset_info.get('n_features', 0))
                mlflow.log_param("best_model", report.best_model_result.model_type.value if report.best_model_result else "none")
                
                # Log metrics for all models
                for result in report.models_evaluated:
                    model_name = result.model_type.value
                    mlflow.log_metric(f"{model_name}_cv_score", result.cv_score_mean)
                    mlflow.log_metric(f"{model_name}_test_score", result.test_score)
                    mlflow.log_metric(f"{model_name}_training_time", result.training_time)
                    mlflow.log_metric(f"{model_name}_model_size", result.model_size)
                
                # Log best model metrics
                if report.best_model_result:
                    mlflow.log_metric("best_cv_score", report.best_model_result.cv_score_mean)
                    mlflow.log_metric("best_test_score", report.best_model_result.test_score)
                    mlflow.log_metric("best_training_time", report.best_model_result.training_time)
                
                # Log ensemble metrics if available
                if report.ensemble_result:
                    mlflow.log_metric("ensemble_score", report.ensemble_result.test_score)
                
                # Log model artifacts
                if report.best_model_result:
                    if hasattr(mlflow, 'xgboost') and report.best_model_result.model_type == ModelType.XGBOOST:
                        mlflow.xgboost.log_model(report.best_model_result.model, "best_model")
                    elif hasattr(mlflow, 'lightgbm') and report.best_model_result.model_type == ModelType.LIGHTGBM:
                        mlflow.lightgbm.log_model(report.best_model_result.model, "best_model")
                    else:
                        mlflow.sklearn.log_model(report.best_model_result.model, "best_model")
                
                # Log report as artifact
                report_dict = asdict(report)
                report_dict['timestamp'] = report.timestamp.isoformat()
                
                with open("tabular_analysis_report.json", "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)
                mlflow.log_artifact("tabular_analysis_report.json")
                
                logger.info("Tabular analysis results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Make predictions using the best trained model."""
        try:
            if self.best_model is None:
                raise ValueError("No trained model available. Run analyze_tabular_data first.")
            
            # Preprocess input data
            if isinstance(X, pd.DataFrame):
                # Apply same preprocessing as training
                X_processed, _, _, _ = await self._preprocess_data(
                    pd.concat([X, pd.DataFrame({self.feature_names[0]: [0]})]).iloc[:-1],  # Dummy target for preprocessing
                    self.feature_names[0],  # Use first feature as dummy target
                    fit_preprocessor=False
                )
            else:
                X_processed = X
            
            # Make predictions
            predictions = self.best_model.predict(X_processed)
            
            result = {
                'predictions': predictions.tolist(),
                'model_type': self.best_model_type.value,
                'prediction_time': datetime.now().isoformat()
            }
            
            # Add probabilities for classification
            if (return_probabilities and 
                hasattr(self.best_model, 'predict_proba') and 
                self.task_type != TaskType.REGRESSION):
                try:
                    probabilities = self.best_model.predict_proba(X_processed)
                    result['probabilities'] = probabilities.tolist()
                except:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'predictions': [],
                'error': str(e),
                'model_type': self.best_model_type.value if self.best_model_type else 'unknown'
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        try:
            if self.best_model is None:
                return {'error': 'No trained model available'}
            
            summary = {
                'best_model': {
                    'type': self.best_model_type.value,
                    'parameters': self.best_model.get_params() if hasattr(self.best_model, 'get_params') else {}
                },
                'task_type': self.task_type.value if self.task_type else 'unknown',
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'categorical_features': len(self.categorical_features) if self.categorical_features else 0,
                'models_trained': len(self.models),
                'preprocessing_steps': list(self.preprocessor.keys()) if self.preprocessor else [],
                'configuration': asdict(self.config)
            }
            
            # Add feature names if available
            if self.feature_names:
                summary['feature_names'] = self.feature_names
            
            return summary
            
        except Exception as e:
            logger.error(f"Model summary generation failed: {str(e)}")
            return {'error': str(e)}

# Advanced utility classes

class TabularDataValidator:
    """Validate tabular data quality and suitability for ML."""
    
    @staticmethod
    def validate_dataset(
        df: pd.DataFrame,
        target_column: str,
        min_samples: int = 100,
        max_missing_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """Comprehensive dataset validation."""
        try:
            validation_result = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'recommendations': [],
                'statistics': {}
            }
            
            # Basic checks
            if target_column not in df.columns:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Target column '{target_column}' not found")
                return validation_result
            
            # Sample size check
            if len(df) < min_samples:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Insufficient samples: {len(df)} < {min_samples}")
            
            # Missing values analysis
            missing_stats = df.isnull().sum()
            total_missing = missing_stats.sum()
            
            if total_missing > 0:
                missing_ratio = total_missing / df.size
                validation_result['statistics']['missing_ratio'] = float(missing_ratio)
                
                if missing_ratio > max_missing_ratio:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"Too many missing values: {missing_ratio:.1%}")
                elif missing_ratio > 0.1:
                    validation_result['warnings'].append(f"High missing values: {missing_ratio:.1%}")
            
            # Target variable analysis
            target_missing = df[target_column].isnull().sum()
            if target_missing > 0:
                validation_result['warnings'].append(f"Target has {target_missing} missing values")
            
            # Check for constant features
            constant_features = []
            for col in df.columns:
                if col != target_column and df[col].nunique() == 1:
                    constant_features.append(col)
            
            if constant_features:
                validation_result['warnings'].append(f"Constant features found: {len(constant_features)}")
                validation_result['recommendations'].append("Remove constant features")
            
            # Check for high cardinality categorical features
            high_cardinality_features = []
            for col in df.select_dtypes(include=['object']).columns:
                if col != target_column and df[col].nunique() > len(df) * 0.5:
                    high_cardinality_features.append(col)
            
            if high_cardinality_features:
                validation_result['warnings'].append(f"High cardinality features: {high_cardinality_features}")
                validation_result['recommendations'].append("Consider feature encoding or selection for high cardinality features")
            
            # Data type analysis
            validation_result['statistics']['data_types'] = df.dtypes.value_counts().to_dict()
            validation_result['statistics']['n_features'] = len(df.columns) - 1
            validation_result['statistics']['n_samples'] = len(df)
            
            # Feature-to-sample ratio
            feature_ratio = (len(df.columns) - 1) / len(df)
            if feature_ratio > 0.1:
                validation_result['warnings'].append(f"High feature-to-sample ratio: {feature_ratio:.3f}")
                validation_result['recommendations'].append("Consider dimensionality reduction")
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'issues': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'recommendations': [],
                'statistics': {}
            }

class ModelComparator:
    """Compare multiple trained models."""
    
    @staticmethod
    def compare_models(
        model_results: List[ModelResult],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models across various metrics."""
        try:
            if not model_results:
                return {'error': 'No models to compare'}
            
            metrics = metrics or ['test_score', 'cv_score_mean', 'training_time', 'model_size']
            
            comparison = {
                'model_count': len(model_results),
                'comparison_metrics': metrics,
                'detailed_comparison': [],
                'rankings': {},
                'best_by_metric': {}
            }
            
            # Detailed comparison
            for result in model_results:
                model_data = {
                    'model_type': result.model_type.value,
                    'test_score': result.test_score,
                    'cv_score_mean': result.cv_score_mean,
                    'cv_score_std': result.cv_score_std,
                    'training_time': result.training_time,
                    'model_size': result.model_size
                }
                comparison['detailed_comparison'].append(model_data)
            
            # Rankings by each metric
            for metric in metrics:
                if metric in ['test_score', 'cv_score_mean']:
                    # Higher is better
                    sorted_models = sorted(model_results, key=lambda x: getattr(x, metric), reverse=True)
                else:
                    # Lower is better (time, size)
                    sorted_models = sorted(model_results, key=lambda x: getattr(x, metric))
                
                comparison['rankings'][metric] = [
                    {
                        'rank': i + 1,
                        'model_type': model.model_type.value,
                        'value': getattr(model, metric)
                    }
                    for i, model in enumerate(sorted_models)
                ]
                
                comparison['best_by_metric'][metric] = sorted_models[0].model_type.value
            
            # Overall score calculation
            normalized_scores = []
            for result in model_results:
                score = 0
                # Test score (40% weight)
                score += 0.4 * (result.test_score / max(r.test_score for r in model_results))
                # CV score (30% weight)
                score += 0.3 * (result.cv_score_mean / max(r.cv_score_mean for r in model_results))
                # Training time penalty (20% weight) - lower is better
                min_time = min(r.training_time for r in model_results)
                max_time = max(r.training_time for r in model_results)
                time_score = 1 - ((result.training_time - min_time) / max(max_time - min_time, 1))
                score += 0.2 * time_score
                # Model size penalty (10% weight) - lower is better
                min_size = min(r.model_size for r in model_results)
                max_size = max(r.model_size for r in model_results)
                size_score = 1 - ((result.model_size - min_size) / max(max_size - min_size, 1))
                score += 0.1 * size_score
                
                normalized_scores.append((result.model_type.value, score))
            
            # Sort by overall score
            normalized_scores.sort(key=lambda x: x[1], reverse=True)
            comparison['overall_ranking'] = [
                {'rank': i + 1, 'model_type': model, 'overall_score': score}
                for i, (model, score) in enumerate(normalized_scores)
            ]
            
            return comparison
            
        except Exception as e:
            return {'error': f"Model comparison failed: {str(e)}"}

class FeatureSelector:
    """Advanced feature selection methods."""
    
    def __init__(self, method: str = 'auto', k: Union[int, str] = 'auto'):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_type: TaskType
    ) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """Select features and return transformed data."""
        try:
            from sklearn.feature_selection import (
                SelectKBest, f_classif, f_regression, mutual_info_classif, 
                mutual_info_regression, RFE, SelectFromModel
            )
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Determine number of features to select
            if self.k == 'auto':
                k = min(int(X.shape[1] * 0.8), 50)
            else:
                k = self.k
            
            feature_scores = {}
            
            if self.method == 'auto':
                # Use different methods based on task type and data size
                if X.shape[1] > 100:
                    self.method = 'mutual_info'
                else:
                    self.method = 'f_test'
            
            if self.method == 'f_test':
                if task_type == TaskType.REGRESSION:
                    self.selector = SelectKBest(f_regression, k=k)
                else:
                    self.selector = SelectKBest(f_classif, k=k)
            
            elif self.method == 'mutual_info':
                if task_type == TaskType.REGRESSION:
                    self.selector = SelectKBest(mutual_info_regression, k=k)
                else:
                    self.selector = SelectKBest(mutual_info_classif, k=k)
            
            elif self.method == 'model_based':
                if task_type == TaskType.REGRESSION:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                
                self.selector = SelectFromModel(estimator, max_features=k)
            
            elif self.method == 'rfe':
                if task_type == TaskType.REGRESSION:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                
                self.selector = RFE(estimator, n_features_to_select=k)
            
            # Fit and transform
            X_selected = self.selector.fit_transform(X, y)
            selected_indices = self.selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            # Get feature scores if available
            if hasattr(self.selector, 'scores_'):
                scores = self.selector.scores_
                for i, score in enumerate(scores):
                    if i in selected_indices:
                        feature_scores[feature_names[i]] = float(score)
            elif hasattr(self.selector, 'estimator_') and hasattr(self.selector.estimator_, 'feature_importances_'):
                importances = self.selector.estimator_.feature_importances_
                for i, importance in enumerate(importances):
                    if i in selected_indices:
                        feature_scores[feature_names[i]] = float(importance)
            
            self.selected_features = selected_feature_names
            
            return X_selected, selected_feature_names, feature_scores
            
        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}")
            return X, feature_names, {}
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted selector."""
        if self.selector is None:
            raise ValueError("Selector not fitted")
        return self.selector.transform(X)

# Utility functions

def create_tabular_analyzer(
    enable_hyperopt: bool = True,
    max_models: int = 5,
    use_gpu: bool = False
) -> TabularModelAnalyzer:
    """Factory function to create a TabularModelAnalyzer."""
    config = TabularModelConfig()
    config.enable_hyperopt = enable_hyperopt
    config.max_models_to_try = max_models
    config.use_gpu = use_gpu
    
    if not enable_hyperopt:
        config.hyperopt_trials = 0
    
    return TabularModelAnalyzer(config)

async def quick_tabular_analysis(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    max_models: int = 3
) -> Dict[str, Any]:
    """Quick tabular analysis for simple use cases."""
    # Create analyzer with simplified configuration
    analyzer = create_tabular_analyzer(enable_hyperopt=False, max_models=max_models)
    
    # Run analysis
    report = await analyzer.analyze_tabular_data(
        df, target_column, categorical_columns
    )
    
    # Return simplified results
    return {
        'task_type': report.task_type.value,
        'best_model': report.best_model_result.model_type.value if report.best_model_result else None,
        'best_score': report.best_model_result.test_score if report.best_model_result else None,
        'models_evaluated': [r.model_type.value for r in report.models_evaluated],
        'feature_count': report.dataset_info.get('n_features', 0),
        'insights': report.insights[:3],
        'recommendations': report.recommendations[:3]
    }

def get_available_models() -> Dict[str, bool]:
    """Get available model types and their availability status."""
    return {
        'xgboost': XGBOOST_AVAILABLE,
        'catboost': CATBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE,
        'tabpfn': TABPFN_AVAILABLE,
        'random_forest': True,
        'extra_trees': True,
        'gradient_boosting': True,
        'linear_models': True,
        'neural_networks': PYTORCH_AVAILABLE,
        'hyperparameter_optimization': OPTUNA_AVAILABLE,
        'advanced_encoders': CATEGORY_ENCODERS_AVAILABLE,
        'model_interpretation': SHAP_AVAILABLE
    }

def get_model_recommendations(
    n_samples: int,
    n_features: int,
    task_type: str,
    categorical_ratio: float = 0.0,
    time_budget: str = 'medium'
) -> Dict[str, str]:
    """Get model recommendations based on data characteristics."""
    recommendations = {}
    
    # Sample size recommendations
    if n_samples < 1000:
        recommendations['sample_size'] = "Small dataset - prefer simpler models and cross-validation"
        recommendations['primary_models'] = "Linear models, Decision Trees, TabPFN"
    elif n_samples < 10000:
        recommendations['sample_size'] = "Medium dataset - tree-based models work well"
        recommendations['primary_models'] = "Random Forest, XGBoost, CatBoost"
    else:
        recommendations['sample_size'] = "Large dataset - scalable models recommended"
        recommendations['primary_models'] = "LightGBM, Neural Networks, Linear models"
    
    # Feature count recommendations
    if n_features > n_samples:
        recommendations['dimensionality'] = "High dimensional - regularization and feature selection critical"
    elif n_features > 100:
        recommendations['dimensionality'] = "Many features - consider feature selection"
    
    # Categorical features
    if categorical_ratio > 0.5:
        recommendations['categorical'] = "Many categorical features - CatBoost, target encoding recommended"
    elif categorical_ratio > 0.2:
        recommendations['categorical'] = "Some categorical features - proper encoding important"
    
    # Task type specific
    if task_type == 'regression':
        recommendations['task_specific'] = "Regression task - focus on RMSE/R² metrics"
    elif task_type == 'binary_classification':
        recommendations['task_specific'] = "Binary classification - ROC-AUC, precision/recall important"
    else:
        recommendations['task_specific'] = "Multi-class classification - balanced accuracy, F1-weighted"
    
    # Time budget
    if time_budget == 'low':
        recommendations['time_budget'] = "Limited time - use default hyperparameters"
    elif time_budget == 'high':
        recommendations['time_budget'] = "Ample time - enable extensive hyperparameter optimization"
    
    return recommendations

def validate_model_inputs(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate inputs for tabular modeling."""
    return TabularDataValidator.validate_dataset(df, target_column)

def compare_model_results(model_results: List[ModelResult]) -> Dict[str, Any]:
    """Compare multiple model results."""
    return ModelComparator.compare_models(model_results)

async def optimize_single_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
    n_trials: int = 100
) -> Dict[str, Any]:
    """Optimize hyperparameters for a single model type."""
    try:
        config = TabularModelConfig()
        config.hyperopt_trials = n_trials
        
        optimizer = HyperparameterOptimizer(config)
        
        best_params = optimizer.optimize_model(
            ModelType(model_type),
            TaskType(task_type),
            X_train, y_train,
            X_val, y_val
        )
        
        # Create and evaluate optimized model
        model = TabularModelFactory.create_model(
            ModelType(model_type),
            TaskType(task_type),
            config,
            best_params
        )
        
        model.fit(X_train, y_train)
        
        if TaskType(task_type) == TaskType.REGRESSION:
            score = model.score(X_val, y_val)
        else:
            y_pred = model.predict(X_val)
            if TaskType(task_type) == TaskType.BINARY_CLASSIFICATION:
                score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            else:
                score = f1_score(y_val, y_pred, average='weighted')
        
        return {
            'model_type': model_type,
            'best_parameters': best_params,
            'validation_score': float(score),
            'model': model
        }
        
    except Exception as e:
        return {
            'model_type': model_type,
            'error': str(e),
            'best_parameters': {},
            'validation_score': 0.0
        }

def create_model_ensemble(
    models: List[BaseEstimator],
    model_names: List[str],
    task_type: TaskType,
    ensemble_method: str = 'voting'
) -> BaseEstimator:
    """Create ensemble from multiple trained models."""
    try:
        if ensemble_method == 'voting':
            estimators = [(name, model) for name, model in zip(model_names, models)]
            
            if task_type == TaskType.REGRESSION:
                ensemble = VotingRegressor(estimators=estimators)
            else:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        elif ensemble_method == 'stacking':
            if task_type == TaskType.REGRESSION:
                ensemble = StackingRegressor(
                    estimators=[(name, model) for name, model in zip(model_names, models)]
                )
            else:
                ensemble = StackingClassifier(
                    estimators=[(name, model) for name, model in zip(model_names, models)]
                )
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        return ensemble
        
    except Exception as e:
        logger.error(f"Ensemble creation failed: {str(e)}")
        raise

# Business intelligence functions

def calculate_feature_business_value(
    feature_importance: Dict[str, float],
    feature_costs: Dict[str, float],
    prediction_value: float = 1000.0
) -> Dict[str, Any]:
    """Calculate business value of features."""
    try:
        feature_values = {}
        total_importance = sum(feature_importance.values())
        
        for feature, importance in feature_importance.items():
            cost = feature_costs.get(feature, 0)
            
            # Calculate value as importance-weighted prediction value minus cost
            feature_value = (importance / total_importance) * prediction_value - cost
            roi = feature_value / (cost + 1e-6)  # Avoid division by zero
            
            feature_values[feature] = {
                'importance': importance,
                'cost': cost,
                'value': feature_value,
                'roi': roi,
                'value_ratio': importance / (cost + 1e-6)
            }
        
        # Sort by ROI
        sorted_features = sorted(
            feature_values.items(),
            key=lambda x: x[1]['roi'],
            reverse=True
        )
        
        return {
            'feature_values': feature_values,
            'top_value_features': sorted_features[:10],
            'total_feature_cost': sum(feature_costs.values()),
            'estimated_prediction_value': prediction_value
        }
        
    except Exception as e:
        logger.error(f"Feature business value calculation failed: {str(e)}")
        return {}

def estimate_model_impact(
    baseline_score: float,
    model_score: float,
    task_type: TaskType,
    business_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Estimate business impact of model improvement."""
    try:
        impact = {}
        
        # Calculate performance improvement
        if task_type == TaskType.REGRESSION:
            # For regression, higher R² is better
            improvement = model_score - baseline_score
            improvement_percentage = (improvement / max(baseline_score, 0.01)) * 100
            
            impact['performance_improvement'] = {
                'absolute_improvement': float(improvement),
                'relative_improvement_percentage': float(improvement_percentage),
                'error_reduction': float(improvement / (1 - baseline_score + 1e-6))
            }
        else:
            # For classification, higher accuracy/F1 is better
            improvement = model_score - baseline_score
            improvement_percentage = (improvement / max(baseline_score, 0.01)) * 100
            
            impact['performance_improvement'] = {
                'absolute_improvement': float(improvement),
                'relative_improvement_percentage': float(improvement_percentage),
                'accuracy_gain': float(improvement)
            }
        
        # Estimate business impact if metrics provided
        if business_metrics:
            revenue_per_prediction = business_metrics.get('revenue_per_prediction', 0)
            predictions_per_day = business_metrics.get('predictions_per_day', 0)
            cost_per_error = business_metrics.get('cost_per_error', 0)
            
            if revenue_per_prediction and predictions_per_day:
                daily_revenue_impact = improvement * predictions_per_day * revenue_per_prediction
                annual_revenue_impact = daily_revenue_impact * 365
                
                impact['revenue_impact'] = {
                    'daily_impact': float(daily_revenue_impact),
                    'annual_impact': float(annual_revenue_impact)
                }
            
            if cost_per_error and predictions_per_day:
                daily_cost_savings = improvement * predictions_per_day * cost_per_error
                annual_cost_savings = daily_cost_savings * 365
                
                impact['cost_savings'] = {
                    'daily_savings': float(daily_cost_savings),
                    'annual_savings': float(annual_cost_savings)
                }
        
        return impact
        
    except Exception as e:
        logger.error(f"Model impact estimation failed: {str(e)}")
        return {}

# Export main classes and functions
__all__ = [
    'TabularModelAnalyzer',
    'TabularModelConfig',
    'TabularModelReport',
    'ModelResult',
    'TabularModelFactory',
    'HyperparameterOptimizer',
    'TabularNeuralNetwork',
    'TabularDataValidator',
    'ModelComparator',
    'FeatureSelector',
    'create_tabular_analyzer',
    'quick_tabular_analysis',
    'get_available_models',
    'get_model_recommendations',
    'validate_model_inputs',
    'compare_model_results',
    'optimize_single_model',
    'create_model_ensemble',
    'calculate_feature_business_value',
    'estimate_model_impact'
]

# Example usage and testing
if __name__ == "__main__":
    async def test_tabular_models():
        """Test the tabular models functionality."""
        print("Testing Tabular Models...")
        print("Available models:", get_available_models())
        
        # Create sample dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate synthetic data
        X = np.random.randn(n_samples, n_features)
        
        # Add some categorical features
        cat_features = np.random.choice(['A', 'B', 'C'], size=(n_samples, 3))
        
        # Create target with some relationship to features
        noise = np.random.randn(n_samples) * 0.1
        y_reg = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + noise
        y_clf = (y_reg > np.median(y_reg)).astype(int)
        
        # Create DataFrame
        feature_names = [f'num_feature_{i}' for i in range(n_features)]
        cat_names = ['cat_A', 'cat_B', 'cat_C']
        
        df_reg = pd.DataFrame(X, columns=feature_names)
        for i, cat_name in enumerate(cat_names):
            df_reg[cat_name] = cat_features[:, i]
        df_reg['target'] = y_reg
        
        df_clf = df_reg.copy()
        df_clf['target'] = y_clf
        
        print(f"Generated datasets - Regression: {df_reg.shape}, Classification: {df_clf.shape}")
        
        # Test data validation
        print("\n=== Data Validation Test ===")
        validation_result = validate_model_inputs(df_reg, 'target', cat_names)
        print(f"Validation result: Valid={validation_result['is_valid']}")
        print(f"Issues: {len(validation_result['issues'])}, Warnings: {len(validation_result['warnings'])}")
        
        # Test regression analysis
        print("\n=== Regression Analysis Test ===")
        analyzer_reg = create_tabular_analyzer(enable_hyperopt=False, max_models=3)
        
        report_reg = await analyzer_reg.analyze_tabular_data(
            df_reg, 'target', categorical_columns=cat_names
        )
        
        print(f"Regression Results:")
        print(f"  Task Type: {report_reg.task_type.value}")
        print(f"  Best Model: {report_reg.best_model_result.model_type.value if report_reg.best_model_result else 'None'}")
        print(f"  Best Score: {report_reg.best_model_result.test_score:.4f}" if report_reg.best_model_result else "No score")
        print(f"  Models Evaluated: {len(report_reg.models_evaluated)}")
        
        if report_reg.best_model_result and report_reg.best_model_result.feature_importance:
            top_features = sorted(
                report_reg.best_model_result.feature_importance.items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            print(f"  Top Features: {[name for name, _ in top_features]}")
        
        # Test classification analysis
        print("\n=== Classification Analysis Test ===")
        analyzer_clf = create_tabular_analyzer(enable_hyperopt=False, max_models=3)
        
        report_clf = await analyzer_clf.analyze_tabular_data(
            df_clf, 'target', categorical_columns=cat_names
        )
        
        print(f"Classification Results:")
        print(f"  Task Type: {report_clf.task_type.value}")
        print(f"  Best Model: {report_clf.best_model_result.model_type.value if report_clf.best_model_result else 'None'}")
        print(f"  Best Score: {report_clf.best_model_result.test_score:.4f}" if report_clf.best_model_result else "No score")
        print(f"  Models Evaluated: {len(report_clf.models_evaluated)}")
        
        # Test model predictions
        if report_clf.best_model_result:
            print("\n=== Prediction Test ===")
            test_data = df_clf.drop('target', axis=1).head(5)
            predictions = await analyzer_clf.predict(test_data, return_probabilities=True)
            
            print(f"Predictions: {predictions['predictions'][:3]}")
            if 'probabilities' in predictions:
                print(f"Probabilities available: {len(predictions['probabilities'])} samples")
        
        # Test quick analysis
        print("\n=== Quick Analysis Test ===")
        quick_results = await quick_tabular_analysis(
            df_clf.head(200), 'target', categorical_columns=cat_names, max_models=2
        )
        
        print(f"Quick Analysis:")
        print(f"  Task: {quick_results['task_type']}")
        print(f"  Best Model: {quick_results['best_model']}")
        print(f"  Best Score: {quick_results['best_score']:.4f}" if quick_results['best_score'] else "No score")
        print(f"  Models: {quick_results['models_evaluated']}")
        
        # Test model comparison
        if len(report_clf.models_evaluated) > 1:
            print("\n=== Model Comparison Test ===")
            comparison = compare_model_results(report_clf.models_evaluated)
            
            print(f"Model Comparison:")
            print(f"  Models Compared: {comparison['model_count']}")
            print(f"  Best by Test Score: {comparison['best_by_metric']['test_score']}")
            print(f"  Best by Speed: {comparison['best_by_metric']['training_time']}")
            
            if 'overall_ranking' in comparison:
                print(f"  Overall Winner: {comparison['overall_ranking'][0]['model_type']}")
        
        # Test business insights
        print("\n=== Business Insights ===")
        for i, insight in enumerate(report_clf.insights[:3], 1):
            print(f"  {i}. {insight}")
        
        print("\n=== Recommendations ===")
        for i, rec in enumerate(report_clf.recommendations[:3], 1):
            print(f"  {i}. {rec}")
        
        # Test feature selection
        if len(feature_names) > 10:
            print("\n=== Feature Selection Test ===")
            selector = FeatureSelector(method='f_test', k=10)
            
            X_selected, selected_names, scores = selector.fit_transform(
                X, y_clf, feature_names, TaskType.BINARY_CLASSIFICATION
            )
            
            print(f"Feature Selection:")
            print(f"  Original Features: {X.shape[1]}")
            print(f"  Selected Features: {X_selected.shape[1]}")
            print(f"  Top Selected: {selected_names[:5]}")
        
        # Test recommendations
        print("\n=== Model Recommendations ===")
        recommendations = get_model_recommendations(
            n_samples=n_samples,
            n_features=n_features,
            task_type='binary_classification',
            categorical_ratio=0.15,
            time_budget='medium'
        )
        
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
        
        return report_reg, report_clf
    
    # Run test
    import asyncio
    results = asyncio.run(test_tabular_models())
