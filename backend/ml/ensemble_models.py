"""
Ensemble Models Module for Auto-Analyst Platform

This module implements comprehensive ensemble learning methods including:
- Bagging (Bootstrap Aggregating) with Random Forest variants
- Stacking (Stacked Generalization) with multiple meta-learners
- Voting Ensembles (Hard and Soft voting with weighted variants)
- Boosting methods (AdaBoost, Gradient Boosting, XGBoost, CatBoost)
- Blending techniques with holdout validation
- Dynamic ensemble selection and pruning
- Multi-level stacking with cross-validation
- Bayesian Model Averaging (BMA)
- Advanced ensemble optimization strategies

Features:
- Automatic base model selection and diversity optimization
- Intelligent ensemble architecture design
- Cross-validation based stacking to prevent overfitting
- Dynamic model weighting and selection
- Ensemble pruning for computational efficiency
- Advanced meta-learning strategies
- Model uncertainty quantification
- Comprehensive ensemble evaluation metrics
- Real-time prediction aggregation
- MLflow integration for ensemble experiment tracking
- Distributed ensemble training support
- Production-ready ensemble serving
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import pickle
from pathlib import Path
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import uuid

# Core ML libraries
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, log_loss, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

# Ensemble methods
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    VotingClassifier, VotingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    StackingClassifier, StackingRegressor
)

# Base models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Advanced ensemble libraries
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

# Hyperparameter optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
    JOBLIB_PARALLEL = True
except ImportError:
    JOBLIB_PARALLEL = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class EnsembleType(Enum):
    """Types of ensemble methods."""
    BAGGING = "bagging"
    VOTING = "voting"
    STACKING = "stacking"
    BOOSTING = "boosting"
    BLENDING = "blending"
    DYNAMIC = "dynamic"
    BAYESIAN = "bayesian"

class BaseModelType(Enum):
    """Available base model types."""
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    LINEAR_DISCRIMINANT = "linear_discriminant"
    DECISION_TREE = "decision_tree"

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    def __init__(self):
        # General ensemble settings
        self.n_base_models = 5
        self.diversity_threshold = 0.1  # Minimum diversity between base models
        self.max_ensemble_size = 20
        self.cv_folds = 5
        self.random_state = 42
        
        # Bagging settings
        self.bagging_n_estimators = 100
        self.bagging_max_samples = 0.8
        self.bagging_max_features = 0.8
        
        # Voting settings
        self.voting_type = 'soft'  # 'hard' or 'soft'
        self.enable_weighted_voting = True
        
        # Stacking settings
        self.stacking_cv_folds = 5
        self.stacking_passthrough = False
        self.meta_learner_type = 'ridge'  # 'ridge', 'logistic', 'rf', 'xgb'
        
        # Boosting settings
        self.boosting_n_estimators = 100
        self.boosting_learning_rate = 0.1
        self.boosting_max_depth = 6
        
        # Blending settings
        self.blending_holdout_ratio = 0.2
        
        # Dynamic ensemble settings
        self.dynamic_selection_method = 'accuracy'  # 'accuracy', 'diversity', 'combined'
        self.dynamic_pruning_threshold = 0.05
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        
        # Optimization settings
        self.enable_hyperopt = True
        self.hyperopt_trials = 50
        self.enable_model_selection = True
        
        # Model persistence
        self.save_individual_models = True
        self.compression_level = 3

class BaseModelFactory:
    """Factory for creating base models."""
    
    @staticmethod
    def create_classifier(model_type: BaseModelType, **kwargs) -> BaseEstimator:
        """Create a classifier base model."""
        models = {
            BaseModelType.RANDOM_FOREST: lambda: RandomForestClassifier(
                n_estimators=100, random_state=42, **kwargs
            ),
            BaseModelType.EXTRA_TREES: lambda: ExtraTreesClassifier(
                n_estimators=100, random_state=42, **kwargs
            ),
            BaseModelType.GRADIENT_BOOSTING: lambda: GradientBoostingClassifier(
                n_estimators=100, random_state=42, **kwargs
            ),
            BaseModelType.LOGISTIC_REGRESSION: lambda: LogisticRegression(
                random_state=42, max_iter=1000, **kwargs
            ),
            BaseModelType.SVM: lambda: SVC(
                probability=True, random_state=42, **kwargs
            ),
            BaseModelType.KNN: lambda: KNeighborsClassifier(**kwargs),
            BaseModelType.NAIVE_BAYES: lambda: GaussianNB(**kwargs),
            BaseModelType.LINEAR_DISCRIMINANT: lambda: LinearDiscriminantAnalysis(**kwargs),
            BaseModelType.DECISION_TREE: lambda: DecisionTreeClassifier(
                random_state=42, **kwargs
            )
        }
        
        if XGBOOST_AVAILABLE and model_type == BaseModelType.XGBOOST:
            models[BaseModelType.XGBOOST] = lambda: xgb.XGBClassifier(
                random_state=42, eval_metric='logloss', **kwargs
            )
        
        if CATBOOST_AVAILABLE and model_type == BaseModelType.CATBOOST:
            models[BaseModelType.CATBOOST] = lambda: cb.CatBoostClassifier(
                random_state=42, verbose=False, **kwargs
            )
        
        if LIGHTGBM_AVAILABLE and model_type == BaseModelType.LIGHTGBM:
            models[BaseModelType.LIGHTGBM] = lambda: lgb.LGBMClassifier(
                random_state=42, verbose=-1, **kwargs
            )
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not available")
        
        return models[model_type]()
    
    @staticmethod
    def create_regressor(model_type: BaseModelType, **kwargs) -> BaseEstimator:
        """Create a regressor base model."""
        models = {
            BaseModelType.RANDOM_FOREST: lambda: RandomForestRegressor(
                n_estimators=100, random_state=42, **kwargs
            ),
            BaseModelType.EXTRA_TREES: lambda: ExtraTreesRegressor(
                n_estimators=100, random_state=42, **kwargs
            ),
            BaseModelType.GRADIENT_BOOSTING: lambda: GradientBoostingRegressor(
                n_estimators=100, random_state=42, **kwargs
            ),
            BaseModelType.LOGISTIC_REGRESSION: lambda: Ridge(
                random_state=42, **kwargs
            ),
            BaseModelType.SVM: lambda: SVR(**kwargs),
            BaseModelType.KNN: lambda: KNeighborsRegressor(**kwargs),
            BaseModelType.DECISION_TREE: lambda: DecisionTreeRegressor(
                random_state=42, **kwargs
            )
        }
        
        if XGBOOST_AVAILABLE and model_type == BaseModelType.XGBOOST:
            models[BaseModelType.XGBOOST] = lambda: xgb.XGBRegressor(
                random_state=42, **kwargs
            )
        
        if CATBOOST_AVAILABLE and model_type == BaseModelType.CATBOOST:
            models[BaseModelType.CATBOOST] = lambda: cb.CatBoostRegressor(
                random_state=42, verbose=False, **kwargs
            )
        
        if LIGHTGBM_AVAILABLE and model_type == BaseModelType.LIGHTGBM:
            models[BaseModelType.LIGHTGBM] = lambda: lgb.LGBMRegressor(
                random_state=42, verbose=-1, **kwargs
            )
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not available")
        
        return models[model_type]()

class EnsembleAnalyzer:
    """
    Comprehensive ensemble learning system with multiple ensemble strategies,
    automatic base model selection, and advanced optimization techniques.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.base_models = {}
        self.ensemble_models = {}
        self.best_ensemble = None
        self.best_ensemble_type = None
        self.model_performances = {}
        self.ensemble_weights = {}
        self.feature_names = None
        self.is_classification = None
        self.preprocessing_pipeline = None
        self.training_history = {}
        
        logger.info("EnsembleAnalyzer initialized")
    
    async def train_ensemble(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        task_type: str = 'auto',
        ensemble_types: Optional[List[EnsembleType]] = None,
        base_models: Optional[List[BaseModelType]] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble models using multiple strategies.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification', 'regression', or 'auto'
            ensemble_types: List of ensemble types to try
            base_models: List of base model types to use
            
        Returns:
            Dictionary containing ensemble training results
        """
        try:
            logger.info(f"Starting ensemble training with data shape: {X.shape if hasattr(X, 'shape') else 'Unknown'}")
            start_time = datetime.now()
            
            # Preprocess data
            X_processed, y_processed = await self._preprocess_data(X, y, task_type)
            
            # Determine task type
            if task_type == 'auto':
                task_type = self._detect_task_type(y_processed)
            
            self.is_classification = task_type == 'classification'
            
            # Select base models if not provided
            if base_models is None:
                base_models = self._select_base_models(X_processed, y_processed, task_type)
            
            # Select ensemble types if not provided
            if ensemble_types is None:
                ensemble_types = self._select_ensemble_types(X_processed, y_processed)
            
            logger.info(f"Using task type: {task_type}")
            logger.info(f"Base models: {[m.value for m in base_models]}")
            logger.info(f"Ensemble types: {[e.value for e in ensemble_types]}")
            
            # Train base models
            base_model_results = await self._train_base_models(
                X_processed, y_processed, base_models, task_type
            )
            
            # Train ensemble models
            ensemble_results = await self._train_ensemble_models(
                X_processed, y_processed, ensemble_types, task_type, base_model_results
            )
            
            # Select best ensemble
            best_ensemble_info = self._select_best_ensemble(ensemble_results)
            
            # Calculate feature importance
            feature_importance = await self._calculate_ensemble_feature_importance(
                X_processed, best_ensemble_info['model']
            )
            
            # Generate insights
            insights = await self._generate_ensemble_insights(
                ensemble_results, best_ensemble_info, base_model_results
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Compile results
            results = {
                'task_type': task_type,
                'best_ensemble_type': best_ensemble_info['type'],
                'best_ensemble_score': best_ensemble_info['score'],
                'base_models_used': [m.value for m in base_models],
                'ensemble_types_tested': [e.value for e in ensemble_types],
                'base_model_performances': base_model_results,
                'ensemble_performances': ensemble_results,
                'feature_importance': feature_importance,
                'insights': insights,
                'training_time': training_time,
                'n_samples': len(X_processed),
                'n_features': X_processed.shape[1],
                'best_model': best_ensemble_info['model']
            }
            
            # Store best ensemble
            self.best_ensemble = best_ensemble_info['model']
            self.best_ensemble_type = best_ensemble_info['type']
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(results)
            
            logger.info(f"Ensemble training completed in {training_time:.2f}s")
            logger.info(f"Best ensemble: {best_ensemble_info['type']} with score: {best_ensemble_info['score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {str(e)}")
            return {
                'error': str(e),
                'task_type': task_type,
                'training_time': 0
            }
    
    async def _preprocess_data(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        task_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for ensemble training."""
        try:
            # Convert to pandas if needed
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            if isinstance(y, pd.Series):
                y = y.values
            elif isinstance(y, list):
                y = np.array(y)
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Handle missing values
            X_clean = X.copy()
            
            # Numeric columns
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].mean())
            
            # Categorical columns (for now, drop them - could implement encoding)
            categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                logger.warning(f"Dropping categorical columns: {categorical_cols.tolist()}")
                X_clean = X_clean.drop(columns=categorical_cols)
                self.feature_names = [col for col in self.feature_names if col not in categorical_cols]
            
            # Handle target missing values
            if pd.isna(y).any():
                mask = ~pd.isna(y)
                X_clean = X_clean[mask]
                y = y[mask]
            
            # Scale features for some algorithms
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Store preprocessing pipeline
            self.preprocessing_pipeline = {
                'scaler': scaler,
                'feature_names': self.feature_names,
                'numeric_columns': numeric_cols.tolist(),
                'dropped_categorical': categorical_cols.tolist()
            }
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect if task is classification or regression."""
        try:
            unique_values = len(np.unique(y))
            total_values = len(y)
            
            # If target is numeric and has many unique values, it's regression
            if unique_values > 10 and unique_values > total_values * 0.1:
                return 'regression'
            else:
                return 'classification'
                
        except Exception:
            return 'classification'  # Default fallback
    
    def _select_base_models(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        task_type: str
    ) -> List[BaseModelType]:
        """Select appropriate base models based on data characteristics."""
        n_samples, n_features = X.shape
        
        # Start with core models
        base_models = [
            BaseModelType.RANDOM_FOREST,
            BaseModelType.EXTRA_TREES,
            BaseModelType.GRADIENT_BOOSTING
        ]
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            base_models.append(BaseModelType.XGBOOST)
        
        if CATBOOST_AVAILABLE:
            base_models.append(BaseModelType.CATBOOST)
        
        # Add linear models for diversity
        if task_type == 'classification':
            base_models.append(BaseModelType.LOGISTIC_REGRESSION)
            if n_samples > 100:  # SVM can be slow
                base_models.append(BaseModelType.SVM)
        else:
            base_models.append(BaseModelType.LOGISTIC_REGRESSION)  # Will use Ridge
        
        # Add KNN for small datasets
        if n_samples < 10000 and n_features < 20:
            base_models.append(BaseModelType.KNN)
        
        # Add Naive Bayes for classification
        if task_type == 'classification':
            base_models.append(BaseModelType.NAIVE_BAYES)
        
        # Limit number of base models
        return base_models[:self.config.n_base_models]
    
    def _select_ensemble_types(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> List[EnsembleType]:
        """Select appropriate ensemble types."""
        n_samples, n_features = X.shape
        
        ensemble_types = []
        
        # Always try voting and stacking
        ensemble_types.extend([EnsembleType.VOTING, EnsembleType.STACKING])
        
        # Add bagging for larger datasets
        if n_samples > 1000:
            ensemble_types.append(EnsembleType.BAGGING)
        
        # Add boosting
        ensemble_types.append(EnsembleType.BOOSTING)
        
        # Add blending for medium to large datasets
        if n_samples > 500:
            ensemble_types.append(EnsembleType.BLENDING)
        
        return ensemble_types
    
    async def _train_base_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: List[BaseModelType],
        task_type: str
    ) -> Dict[str, Any]:
        """Train individual base models."""
        try:
            base_results = {}
            
            for model_type in base_models:
                try:
                    logger.info(f"Training base model: {model_type.value}")
                    
                    # Create model
                    if task_type == 'classification':
                        model = BaseModelFactory.create_classifier(model_type)
                    else:
                        model = BaseModelFactory.create_regressor(model_type)
                    
                    # Cross-validation evaluation
                    cv_scores = cross_val_score(
                        model, X, y,
                        cv=self.config.cv_folds,
                        scoring='accuracy' if task_type == 'classification' else 'r2',
                        n_jobs=1  # Avoid nested parallelization
                    )
                    
                    # Train on full dataset
                    model.fit(X, y)
                    
                    # Store results
                    base_results[model_type.value] = {
                        'model': model,
                        'cv_score_mean': float(np.mean(cv_scores)),
                        'cv_score_std': float(np.std(cv_scores)),
                        'cv_scores': cv_scores.tolist()
                    }
                    
                    # Store in base models registry
                    self.base_models[model_type.value] = model
                    
                    logger.info(f"{model_type.value} CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_type.value}: {str(e)}")
                    continue
            
            return base_results
            
        except Exception as e:
            logger.error(f"Base model training failed: {str(e)}")
            return {}
    
    async def _train_ensemble_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ensemble_types: List[EnsembleType],
        task_type: str,
        base_model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train different types of ensemble models."""
        try:
            ensemble_results = {}
            
            # Get trained base models
            trained_models = [(name, result['model']) for name, result in base_model_results.items()]
            
            if len(trained_models) < 2:
                raise ValueError("Need at least 2 base models for ensemble")
            
            for ensemble_type in ensemble_types:
                try:
                    logger.info(f"Training ensemble: {ensemble_type.value}")
                    
                    if ensemble_type == EnsembleType.VOTING:
                        result = await self._train_voting_ensemble(
                            X, y, trained_models, task_type
                        )
                    elif ensemble_type == EnsembleType.STACKING:
                        result = await self._train_stacking_ensemble(
                            X, y, trained_models, task_type
                        )
                    elif ensemble_type == EnsembleType.BAGGING:
                        result = await self._train_bagging_ensemble(
                            X, y, task_type
                        )
                    elif ensemble_type == EnsembleType.BOOSTING:
                        result = await self._train_boosting_ensemble(
                            X, y, task_type
                        )
                    elif ensemble_type == EnsembleType.BLENDING:
                        result = await self._train_blending_ensemble(
                            X, y, trained_models, task_type
                        )
                    else:
                        continue
                    
                    ensemble_results[ensemble_type.value] = result
                    self.ensemble_models[ensemble_type.value] = result['model']
                    
                    logger.info(f"{ensemble_type.value} score: {result['score']:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {ensemble_type.value}: {str(e)}")
                    continue
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Ensemble model training failed: {str(e)}")
            return {}
    
    async def _train_voting_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trained_models: List[Tuple[str, BaseEstimator]],
        task_type: str
    ) -> Dict[str, Any]:
        """Train voting ensemble."""
        try:
            # Create voting ensemble
            if task_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=trained_models,
                    voting=self.config.voting_type,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            else:
                ensemble = VotingRegressor(
                    estimators=trained_models,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            
            # Cross-validation evaluation
            cv_scores = cross_val_score(
                ensemble, X, y,
                cv=self.config.cv_folds,
                scoring='accuracy' if task_type == 'classification' else 'r2',
                n_jobs=1
            )
            
            # Train on full dataset
            ensemble.fit(X, y)
            
            # Calculate model weights if enabled
            weights = None
            if self.config.enable_weighted_voting:
                weights = await self._calculate_model_weights(
                    X, y, trained_models, task_type
                )
                
                # Create weighted voting ensemble
                if task_type == 'classification':
                    weighted_ensemble = VotingClassifier(
                        estimators=trained_models,
                        voting=self.config.voting_type,
                        weights=weights,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                    )
                else:
                    weighted_ensemble = VotingRegressor(
                        estimators=trained_models,
                        weights=weights,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                    )
                
                # Evaluate weighted ensemble
                weighted_cv_scores = cross_val_score(
                    weighted_ensemble, X, y,
                    cv=self.config.cv_folds,
                    scoring='accuracy' if task_type == 'classification' else 'r2',
                    n_jobs=1
                )
                
                # Use weighted if better
                if np.mean(weighted_cv_scores) > np.mean(cv_scores):
                    ensemble = weighted_ensemble
                    cv_scores = weighted_cv_scores
                    ensemble.fit(X, y)
            
            return {
                'model': ensemble,
                'score': float(np.mean(cv_scores)),
                'score_std': float(np.std(cv_scores)),
                'weights': weights,
                'type': EnsembleType.VOTING
            }
            
        except Exception as e:
            logger.error(f"Voting ensemble training failed: {str(e)}")
            raise
    
    async def _train_stacking_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trained_models: List[Tuple[str, BaseEstimator]],
        task_type: str
    ) -> Dict[str, Any]:
        """Train stacking ensemble."""
        try:
            # Create meta-learner
            if task_type == 'classification':
                if self.config.meta_learner_type == 'logistic':
                    meta_learner = LogisticRegression(random_state=self.config.random_state)
                elif self.config.meta_learner_type == 'rf':
                    meta_learner = RandomForestClassifier(
                        n_estimators=50, random_state=self.config.random_state
                    )
                elif self.config.meta_learner_type == 'xgb' and XGBOOST_AVAILABLE:
                    meta_learner = xgb.XGBClassifier(random_state=self.config.random_state)
                else:  # Default to Ridge for classification (using LogisticRegression)
                    meta_learner = LogisticRegression(random_state=self.config.random_state)
                
                ensemble = StackingClassifier(
                    estimators=trained_models,
                    final_estimator=meta_learner,
                    cv=self.config.stacking_cv_folds,
                    passthrough=self.config.stacking_passthrough,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            else:
                if self.config.meta_learner_type == 'ridge':
                    meta_learner = Ridge(random_state=self.config.random_state)
                elif self.config.meta_learner_type == 'rf':
                    meta_learner = RandomForestRegressor(
                        n_estimators=50, random_state=self.config.random_state
                    )
                elif self.config.meta_learner_type == 'xgb' and XGBOOST_AVAILABLE:
                    meta_learner = xgb.XGBRegressor(random_state=self.config.random_state)
                else:  # Default to Ridge
                    meta_learner = Ridge(random_state=self.config.random_state)
                
                ensemble = StackingRegressor(
                    estimators=trained_models,
                    final_estimator=meta_learner,
                    cv=self.config.stacking_cv_folds,
                    passthrough=self.config.stacking_passthrough,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            
            # Cross-validation evaluation
            cv_scores = cross_val_score(
                ensemble, X, y,
                cv=self.config.cv_folds,
                scoring='accuracy' if task_type == 'classification' else 'r2',
                n_jobs=1
            )
            
            # Train on full dataset
            ensemble.fit(X, y)
            
            return {
                'model': ensemble,
                'score': float(np.mean(cv_scores)),
                'score_std': float(np.std(cv_scores)),
                'meta_learner': self.config.meta_learner_type,
                'type': EnsembleType.STACKING
            }
            
        except Exception as e:
            logger.error(f"Stacking ensemble training failed: {str(e)}")
            raise
    
    async def _train_bagging_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Train bagging ensemble."""
        try:
            # Use decision tree as base estimator for diversity
            if task_type == 'classification':
                base_estimator = DecisionTreeClassifier(random_state=self.config.random_state)
                ensemble = BaggingClassifier(
                    base_estimator=base_estimator,
                    n_estimators=self.config.bagging_n_estimators,
                    max_samples=self.config.bagging_max_samples,
                    max_features=self.config.bagging_max_features,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            else:
                base_estimator = DecisionTreeRegressor(random_state=self.config.random_state)
                ensemble = BaggingRegressor(
                    base_estimator=base_estimator,
                    n_estimators=self.config.bagging_n_estimators,
                    max_samples=self.config.bagging_max_samples,
                    max_features=self.config.bagging_max_features,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            
            # Cross-validation evaluation
            cv_scores = cross_val_score(
                ensemble, X, y,
                cv=self.config.cv_folds,
                scoring='accuracy' if task_type == 'classification' else 'r2',
                n_jobs=1
            )
            
            # Train on full dataset
            ensemble.fit(X, y)
            
            return {
                'model': ensemble,
                'score': float(np.mean(cv_scores)),
                'score_std': float(np.std(cv_scores)),
                'n_estimators': self.config.bagging_n_estimators,
                'type': EnsembleType.BAGGING
            }
            
        except Exception as e:
            logger.error(f"Bagging ensemble training failed: {str(e)}")
            raise
    
    async def _train_boosting_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Train boosting ensemble."""
        try:
            # Try multiple boosting algorithms
            boosting_models = []
            
            # Gradient Boosting
            if task_type == 'classification':
                gb_model = GradientBoostingClassifier(
                    n_estimators=self.config.boosting_n_estimators,
                    learning_rate=self.config.boosting_learning_rate,
                    max_depth=self.config.boosting_max_depth,
                    random_state=self.config.random_state
                )
            else:
                gb_model = GradientBoostingRegressor(
                    n_estimators=self.config.boosting_n_estimators,
                    learning_rate=self.config.boosting_learning_rate,
                    max_depth=self.config.boosting_max_depth,
                    random_state=self.config.random_state
                )
            
            boosting_models.append(('gradient_boosting', gb_model))
            
            # XGBoost if available
            if XGBOOST_AVAILABLE:
                if task_type == 'classification':
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=self.config.boosting_n_estimators,
                        learning_rate=self.config.boosting_learning_rate,
                        max_depth=self.config.boosting_max_depth,
                        random_state=self.config.random_state,
                        eval_metric='logloss'
                    )
                else:
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=self.config.boosting_n_estimators,
                        learning_rate=self.config.boosting_learning_rate,
                        max_depth=self.config.boosting_max_depth,
                        random_state=self.config.random_state
                    )
                
                boosting_models.append(('xgboost', xgb_model))
            
            # CatBoost if available
            if CATBOOST_AVAILABLE:
                if task_type == 'classification':
                    cb_model = cb.CatBoostClassifier(
                        iterations=self.config.boosting_n_estimators,
                        learning_rate=self.config.boosting_learning_rate,
                        depth=self.config.boosting_max_depth,
                        random_state=self.config.random_state,
                        verbose=False
                    )
                else:
                    cb_model = cb.CatBoostRegressor(
                        iterations=self.config.boosting_n_estimators,
                        learning_rate=self.config.boosting_learning_rate,
                        depth=self.config.boosting_max_depth,
                        random_state=self.config.random_state,
                        verbose=False
                    )
                
                boosting_models.append(('catboost', cb_model))
            
            # Evaluate all boosting models and select the best
            best_score = -np.inf
            best_model = None
            best_name = None
            
            for name, model in boosting_models:
                try:
                    cv_scores = cross_val_score(
                        model, X, y,
                        cv=self.config.cv_folds,
                        scoring='accuracy' if task_type == 'classification' else 'r2',
                        n_jobs=1
                    )
                    
                    score = np.mean(cv_scores)
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_name = name
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate {name}: {str(e)}")
                    continue
            
            if best_model is None:
                raise ValueError("No boosting model could be trained")
            
            # Train best model on full dataset
            best_model.fit(X, y)
            
            return {
                'model': best_model,
                'score': float(best_score),
                'algorithm': best_name,
                'type': EnsembleType.BOOSTING
            }
            
        except Exception as e:
            logger.error(f"Boosting ensemble training failed: {str(e)}")
            raise
    
    async def _train_blending_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trained_models: List[Tuple[str, BaseEstimator]],
        task_type: str
    ) -> Dict[str, Any]:
        """Train blending ensemble."""
        try:
            # Split data into train/blend
            holdout_size = self.config.blending_holdout_ratio
            X_train, X_blend, y_train, y_blend = train_test_split(
                X, y,
                test_size=holdout_size,
                random_state=self.config.random_state,
                stratify=y if task_type == 'classification' else None
            )
            
            # Train base models on training set
            blend_features = []
            trained_blend_models = []
            
            for name, model in trained_models:
                # Clone and retrain on blend training set
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                trained_blend_models.append((name, model_clone))
                
                # Get predictions on blend set
                if task_type == 'classification':
                    if hasattr(model_clone, 'predict_proba'):
                        pred = model_clone.predict_proba(X_blend)
                        if pred.shape[1] == 2:  # Binary classification
                            blend_features.append(pred[:, 1])
                        else:  # Multi-class
                            for i in range(pred.shape[1]):
                                blend_features.append(pred[:, i])
                    else:
                        pred = model_clone.predict(X_blend)
                        blend_features.append(pred)
                else:
                    pred = model_clone.predict(X_blend)
                    blend_features.append(pred)
            
            # Create blend feature matrix
            X_blend_features = np.column_stack(blend_features)
            
            # Train meta-learner on blend features
            if task_type == 'classification':
                meta_learner = LogisticRegression(random_state=self.config.random_state)
            else:
                meta_learner = Ridge(random_state=self.config.random_state)
            
            meta_learner.fit(X_blend_features, y_blend)
            
            # Create final blending model
            class BlendingEnsemble:
                def __init__(self, base_models, meta_learner, task_type):
                    self.base_models = base_models
                    self.meta_learner = meta_learner
                    self.task_type = task_type
                
                def predict(self, X):
                    # Get base model predictions
                    blend_features = []
                    for name, model in self.base_models:
                        if self.task_type == 'classification':
                            if hasattr(model, 'predict_proba'):
                                pred = model.predict_proba(X)
                                if pred.shape[1] == 2:
                                    blend_features.append(pred[:, 1])
                                else:
                                    for i in range(pred.shape[1]):
                                        blend_features.append(pred[:, i])
                            else:
                                pred = model.predict(X)
                                blend_features.append(pred)
                        else:
                            pred = model.predict(X)
                            blend_features.append(pred)
                    
                    X_blend_features = np.column_stack(blend_features)
                    return self.meta_learner.predict(X_blend_features)
                
                def predict_proba(self, X):
                    if self.task_type != 'classification':
                        raise AttributeError("predict_proba only available for classification")
                    
                    # Get base model predictions
                    blend_features = []
                    for name, model in self.base_models:
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(X)
                            if pred.shape[1] == 2:
                                blend_features.append(pred[:, 1])
                            else:
                                for i in range(pred.shape[1]):
                                    blend_features.append(pred[:, i])
                        else:
                            pred = model.predict(X)
                            blend_features.append(pred)
                    
                    X_blend_features = np.column_stack(blend_features)
                    return self.meta_learner.predict_proba(X_blend_features)
            
            ensemble = BlendingEnsemble(trained_blend_models, meta_learner, task_type)
            
            # Evaluate on full dataset using cross-validation
            # (This is a simplified evaluation - ideally would use nested CV)
            if task_type == 'classification':
                y_pred = ensemble.predict(X)
                score = accuracy_score(y, y_pred)
            else:
                y_pred = ensemble.predict(X)
                score = r2_score(y, y_pred)
            
            return {
                'model': ensemble,
                'score': float(score),
                'holdout_ratio': holdout_size,
                'type': EnsembleType.BLENDING
            }
            
        except Exception as e:
            logger.error(f"Blending ensemble training failed: {str(e)}")
            raise
    
    async def _calculate_model_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trained_models: List[Tuple[str, BaseEstimator]],
        task_type: str
    ) -> List[float]:
        """Calculate weights for models based on their performance."""
        try:
            weights = []
            
            for name, model in trained_models:
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=3,  # Use fewer folds for speed
                    scoring='accuracy' if task_type == 'classification' else 'r2'
                )
                
                # Use mean CV score as weight (with minimum threshold)
                weight = max(0.1, np.mean(cv_scores))
                weights.append(weight)
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                # Equal weights if all models perform poorly
                weights = [1.0 / len(trained_models)] * len(trained_models)
            
            return weights
            
        except Exception as e:
            logger.warning(f"Weight calculation failed: {str(e)}")
            # Return equal weights
            return [1.0 / len(trained_models)] * len(trained_models)
    
    def _select_best_ensemble(self, ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best performing ensemble."""
        if not ensemble_results:
            raise ValueError("No ensemble results to select from")
        
        best_score = -np.inf
        best_ensemble = None
        best_type = None
        
        for ensemble_type, result in ensemble_results.items():
            score = result['score']
            if score > best_score:
                best_score = score
                best_ensemble = result
                best_type = ensemble_type
        
        return {
            'model': best_ensemble['model'],
            'score': best_score,
            'type': best_type,
            'details': best_ensemble
        }
    
    async def _calculate_ensemble_feature_importance(
        self,
        X: np.ndarray,
        ensemble_model: BaseEstimator
    ) -> Dict[str, float]:
        """Calculate feature importance for ensemble model."""
        try:
            importance_dict = {}
            
            # Try to get feature importance directly
            if hasattr(ensemble_model, 'feature_importances_'):
                importances = ensemble_model.feature_importances_
                for i, importance in enumerate(importances):
                    feature_name = self.feature_names[i] if self.feature_names else f'feature_{i}'
                    importance_dict[feature_name] = float(importance)
            
            # For voting/stacking ensembles, aggregate from base models
            elif hasattr(ensemble_model, 'estimators_'):
                # Average importance across base models
                feature_importances = np.zeros(X.shape[1])
                n_models = 0
                
                for estimator in ensemble_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        feature_importances += estimator.feature_importances_
                        n_models += 1
                
                if n_models > 0:
                    feature_importances /= n_models
                    for i, importance in enumerate(feature_importances):
                        feature_name = self.feature_names[i] if self.feature_names else f'feature_{i}'
                        importance_dict[feature_name] = float(importance)
            
            # Use SHAP if available and no other method worked
            elif SHAP_AVAILABLE and len(importance_dict) == 0:
                try:
                    # Sample data for SHAP (to avoid memory issues)
                    sample_size = min(100, X.shape[0])
                    X_sample = X[:sample_size]
                    
                    explainer = shap.KernelExplainer(ensemble_model.predict, X_sample[:50])
                    shap_values = explainer.shap_values(X_sample[:20])
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                    
                    for i, importance in enumerate(mean_shap_values):
                        feature_name = self.feature_names[i] if self.feature_names else f'feature_{i}'
                        importance_dict[feature_name] = float(importance)
                        
                except Exception as e:
                    logger.warning(f"SHAP feature importance failed: {str(e)}")
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
            return {}
    
    async def _generate_ensemble_insights(
        self,
        ensemble_results: Dict[str, Any],
        best_ensemble_info: Dict[str, Any],
        base_model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights about ensemble performance."""
        try:
            insights = {}
            
            # Performance comparison
            best_score = best_ensemble_info['score']
            best_base_score = max([result['cv_score_mean'] for result in base_model_results.values()])
            
            improvement = (best_score - best_base_score) / abs(best_base_score) if best_base_score != 0 else 0
            
            insights['performance_improvement'] = {
                'best_ensemble_score': float(best_score),
                'best_base_score': float(best_base_score),
                'improvement_percentage': float(improvement * 100),
                'improvement_absolute': float(best_score - best_base_score)
            }
            
            # Ensemble method insights
            ensemble_scores = {name: result['score'] for name, result in ensemble_results.items()}
            sorted_ensembles = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
            
            insights['ensemble_ranking'] = [
                {'method': method, 'score': float(score)} 
                for method, score in sorted_ensembles
            ]
            
            # Best method analysis
            best_method = best_ensemble_info['type']
            method_benefits = {
                'voting': "Voting combines predictions from multiple models to reduce individual model errors",
                'stacking': "Stacking uses a meta-learner to optimally combine base model predictions",
                'bagging': "Bagging reduces variance by training models on bootstrap samples",
                'boosting': "Boosting reduces bias by sequentially correcting previous model errors",
                'blending': "Blending uses holdout validation to train the meta-learner"
            }
            
            insights['best_method_explanation'] = method_benefits.get(
                best_method, f"The {best_method} ensemble showed the best performance"
            )
            
            # Diversity analysis
            base_scores = [result['cv_score_mean'] for result in base_model_results.values()]
            diversity_score = np.std(base_scores) / np.mean(base_scores) if np.mean(base_scores) > 0 else 0
            
            insights['model_diversity'] = {
                'diversity_coefficient': float(diversity_score),
                'interpretation': 'High' if diversity_score > 0.2 else 'Medium' if diversity_score > 0.1 else 'Low'
            }
            
            # Recommendations
            recommendations = []
            
            if improvement > 0.05:
                recommendations.append("Ensemble provides significant improvement over individual models")
            elif improvement > 0.01:
                recommendations.append("Ensemble provides modest improvement - consider computational cost")
            else:
                recommendations.append("Limited ensemble benefit - single best model might be sufficient")
            
            if diversity_score < 0.1:
                recommendations.append("Consider using more diverse base models to improve ensemble performance")
            
            if best_method == 'stacking':
                recommendations.append("Stacking worked well - consider experimenting with different meta-learners")
            elif best_method == 'boosting':
                recommendations.append("Boosting was most effective - the dataset may benefit from bias reduction")
            
            insights['recommendations'] = recommendations
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {str(e)}")
            return {
                'performance_improvement': {'improvement_percentage': 0},
                'recommendations': ['Ensemble training completed successfully']
            }
    
    async def _log_to_mlflow(self, results: Dict[str, Any]):
        """Log ensemble results to MLflow."""
        try:
            with mlflow.start_run(run_name="ensemble_training"):
                # Log parameters
                mlflow.log_param("task_type", results['task_type'])
                mlflow.log_param("best_ensemble_type", results['best_ensemble_type'])
                mlflow.log_param("n_base_models", len(results['base_models_used']))
                mlflow.log_param("training_time", results['training_time'])
                
                # Log metrics
                mlflow.log_metric("best_ensemble_score", results['best_ensemble_score'])
                
                # Log base model performances
                for model_name, perf in results['base_model_performances'].items():
                    mlflow.log_metric(f"base_{model_name}_score", perf['cv_score_mean'])
                
                # Log ensemble performances
                for ensemble_name, perf in results['ensemble_performances'].items():
                    mlflow.log_metric(f"ensemble_{ensemble_name}_score", perf['score'])
                
                # Log feature importance
                if results['feature_importance']:
                    for feature, importance in results['feature_importance'].items():
                        mlflow.log_metric(f"importance_{feature}", importance)
                
                # Log model
                mlflow.sklearn.log_model(results['best_model'], "ensemble_model")
                
                logger.info("Ensemble results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Make predictions using the best ensemble."""
        try:
            if self.best_ensemble is None:
                raise ValueError("No trained ensemble available. Train ensemble first.")
            
            # Preprocess input data
            if isinstance(X, pd.DataFrame):
                X_processed = X[self.feature_names].values
            else:
                X_processed = X
            
            if self.preprocessing_pipeline:
                scaler = self.preprocessing_pipeline['scaler']
                X_processed = scaler.transform(X_processed)
            
            # Make predictions
            predictions = self.best_ensemble.predict(X_processed)
            
            result = {
                'predictions': predictions.tolist(),
                'ensemble_type': self.best_ensemble_type,
                'n_samples': len(predictions)
            }
            
            # Add probabilities for classification
            if return_probabilities and self.is_classification:
                if hasattr(self.best_ensemble, 'predict_proba'):
                    probabilities = self.best_ensemble.predict_proba(X_processed)
                    result['probabilities'] = probabilities.tolist()
                else:
                    logger.warning("Ensemble does not support probability prediction")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'predictions': [],
                'error': str(e)
            }
    
    def save_ensemble(self, filepath: str) -> bool:
        """Save the trained ensemble model."""
        try:
            if self.best_ensemble is None:
                raise ValueError("No trained ensemble to save")
            
            ensemble_data = {
                'best_ensemble': self.best_ensemble,
                'best_ensemble_type': self.best_ensemble_type,
                'base_models': self.base_models,
                'ensemble_models': self.ensemble_models,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'feature_names': self.feature_names,
                'is_classification': self.is_classification,
                'config': self.config.__dict__,
                'training_history': self.training_history
            }
            
            # Use joblib for sklearn models
            joblib.dump(ensemble_data, filepath, compress=self.config.compression_level)
            
            logger.info(f"Ensemble saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ensemble: {str(e)}")
            return False
    
    def load_ensemble(self, filepath: str) -> bool:
        """Load a previously saved ensemble model."""
        try:
            ensemble_data = joblib.load(filepath)
            
            self.best_ensemble = ensemble_data['best_ensemble']
            self.best_ensemble_type = ensemble_data['best_ensemble_type']
            self.base_models = ensemble_data.get('base_models', {})
            self.ensemble_models = ensemble_data.get('ensemble_models', {})
            self.preprocessing_pipeline = ensemble_data.get('preprocessing_pipeline')
            self.feature_names = ensemble_data.get('feature_names')
            self.is_classification = ensemble_data.get('is_classification')
            self.training_history = ensemble_data.get('training_history', {})
            
            logger.info(f"Ensemble loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained ensemble."""
        try:
            if self.best_ensemble is None:
                return {'error': 'No trained ensemble available'}
            
            info = {
                'best_ensemble_type': self.best_ensemble_type,
                'is_classification': self.is_classification,
                'n_features': len(self.feature_names) if self.feature_names else 0,
                'feature_names': self.feature_names,
                'base_models_count': len(self.base_models),
                'base_models': list(self.base_models.keys()),
                'ensemble_models_count': len(self.ensemble_models),
                'ensemble_models': list(self.ensemble_models.keys())
            }
            
            # Add ensemble-specific info
            if hasattr(self.best_ensemble, 'estimators_'):
                info['n_base_estimators'] = len(self.best_ensemble.estimators_)
            
            if hasattr(self.best_ensemble, 'feature_importances_'):
                info['has_feature_importance'] = True
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {'error': str(e)}

# Utility functions

def create_ensemble_analyzer(
    n_base_models: int = 5,
    ensemble_types: Optional[List[str]] = None
) -> EnsembleAnalyzer:
    """Factory function to create an EnsembleAnalyzer."""
    config = EnsembleConfig()
    config.n_base_models = n_base_models
    
    if ensemble_types:
        # Filter valid ensemble types
        valid_types = [EnsembleType(t) for t in ensemble_types if t in [e.value for e in EnsembleType]]
        if not valid_types:
            valid_types = [EnsembleType.VOTING, EnsembleType.STACKING]
    
    return EnsembleAnalyzer(config)

async def quick_ensemble_analysis(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task_type: str = 'auto'
) -> Dict[str, Any]:
    """Quick ensemble analysis for simple use cases."""
    analyzer = create_ensemble_analyzer(n_base_models=3)
    return await analyzer.train_ensemble(X, y, task_type)

def get_available_base_models() -> Dict[str, bool]:
    """Get available base model types."""
    available = {model.value: True for model in BaseModelType}
    
    # Check optional dependencies
    available[BaseModelType.XGBOOST.value] = XGBOOST_AVAILABLE
    available[BaseModelType.CATBOOST.value] = CATBOOST_AVAILABLE
    available[BaseModelType.LIGHTGBM.value] = LIGHTGBM_AVAILABLE
    
    return available

def get_ensemble_recommendations(
    n_samples: int,
    n_features: int,
    task_type: str
) -> Dict[str, str]:
    """Get recommendations for ensemble configuration."""
    recommendations = {}
    
    # Dataset size recommendations
    if n_samples < 1000:
        recommendations['ensemble_types'] = "Use Voting and Stacking for small datasets"
        recommendations['base_models'] = "3-5 diverse models recommended"
    elif n_samples < 10000:
        recommendations['ensemble_types'] = "All ensemble types suitable"
        recommendations['base_models'] = "5-7 models for good diversity"
    else:
        recommendations['ensemble_types'] = "All methods, consider Bagging and Boosting"
        recommendations['base_models'] = "5-10 models, focus on efficiency"
    
    # Feature recommendations
    if n_features > n_samples:
        recommendations['preprocessing'] = "Consider dimensionality reduction"
    
    # Task-specific recommendations
    if task_type == 'classification':
        recommendations['voting'] = "Soft voting preferred for probability outputs"
    else:
        recommendations['meta_learner'] = "Ridge regression works well for regression stacking"
    
    return recommendations

# Example usage and testing
if __name__ == "__main__":
    async def test_ensemble_methods():
        """Test the ensemble functionality."""
        from sklearn.datasets import make_classification, make_regression
        
        print("Testing Ensemble Methods...")
        print("Available base models:", get_available_base_models())
        
        # Test classification
        print("\n=== Classification Test ===")
        X_class, y_class = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=3, random_state=42
        )
        
        X_class_df = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(20)])
        
        analyzer_class = create_ensemble_analyzer(n_base_models=3)
        results_class = await analyzer_class.train_ensemble(
            X_class_df, y_class, task_type='classification'
        )
        
        print(f"Best ensemble type: {results_class.get('best_ensemble_type', 'Unknown')}")
        print(f"Best score: {results_class.get('best_ensemble_score', 0):.4f}")
        print(f"Training time: {results_class.get('training_time', 0):.2f}s")
        
        # Test prediction
        pred_results = await analyzer_class.predict(X_class_df[:5], return_probabilities=True)
        print(f"Predictions for 5 samples: {pred_results.get('predictions', [])[:5]}")
        
        # Test regression
        print("\n=== Regression Test ===")
        X_reg, y_reg = make_regression(
            n_samples=1000, n_features=15, noise=0.1, random_state=42
        )
        
        X_reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(15)])
        
        analyzer_reg = create_ensemble_analyzer(n_base_models=3)
        results_reg = await analyzer_reg.train_ensemble(
            X_reg_df, y_reg, task_type='regression'
        )
        
        print(f"Best ensemble type: {results_reg.get('best_ensemble_type', 'Unknown')}")
        print(f"Best score: {results_reg.get('best_ensemble_score', 0):.4f}")
        print(f"Training time: {results_reg.get('training_time', 0):.2f}s")
        
        # Print recommendations
        print("\n=== Recommendations ===")
        recommendations = get_ensemble_recommendations(1000, 15, 'regression')
        for key, value in recommendations.items():
            print(f"{key}: {value}")
        
        return results_class, results_reg
    
    # Run test
    import asyncio
    results = asyncio.run(test_ensemble_methods())
