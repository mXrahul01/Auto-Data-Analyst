"""
Model Explainer Module for Auto-Analyst Platform

This module implements comprehensive model interpretability and explainability including:
- SHAP (SHapley Additive exPlanations) for all model types
- LIME (Local Interpretable Model-agnostic Explanations)
- Permutation feature importance
- Partial dependence plots
- Feature interaction analysis
- Global and local explanations
- Model-agnostic explanation methods
- Tree-specific explanations
- Deep learning explanations
- Time series explanations
- Natural language explanation generation

Features:
- Multiple explanation methods with automatic selection
- Support for all major ML model types
- Global feature importance and local instance explanations
- Interactive visualization data preparation
- Performance optimization for large datasets
- Explanation aggregation and summarization
- Business-friendly interpretation generation
- Explanation stability and consistency analysis
- Custom explanation protocols
- Integration with visualization frameworks
- Real-time explanation generation
- Batch explanation processing
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
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import math
from pathlib import Path

# Core ML libraries
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import export_text
from sklearn.preprocessing import LabelEncoder

# SHAP (primary explainability library)
try:
    import shap
    SHAP_AVAILABLE = True
    
    # Initialize SHAP
    shap.initjs()
    
except ImportError:
    SHAP_AVAILABLE = False

# LIME (alternative explainability library)
try:
    import lime
    import lime.lime_tabular
    import lime.lime_text
    import lime.lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Advanced interpretability libraries
try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from interpret import show
    INTERPRET_AVAILABLE = True
except ImportError:
    INTERPRET_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

# Visualization support
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Deep learning explanations
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Advanced analysis
try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations available."""
    GLOBAL = "global"
    LOCAL = "local"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    FEATURE_INTERACTION = "feature_interaction"
    COUNTERFACTUAL = "counterfactual"
    ANCHOR = "anchor"

class ExplainerMethod(Enum):
    """Available explainer methods."""
    SHAP_TREE = "shap_tree"
    SHAP_LINEAR = "shap_linear"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    SHAP_GRADIENT = "shap_gradient"
    LIME_TABULAR = "lime_tabular"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ELI5_PERMUTATION = "eli5_permutation"
    BUILT_IN_IMPORTANCE = "built_in_importance"

class ModelType(Enum):
    """Supported model types for explanation."""
    TREE_BASED = "tree_based"
    LINEAR = "linear" 
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    UNKNOWN = "unknown"

@dataclass
class ExplanationConfig:
    """Configuration for model explanations."""
    
    def __init__(self):
        # General settings
        self.max_display_features = 20
        self.explanation_sample_size = 1000  # Max samples for explanation
        self.background_sample_size = 100    # Background samples for SHAP
        self.random_state = 42
        
        # SHAP settings
        self.shap_tree_model_output = "probability"  # "probability", "margin", "raw"
        self.shap_check_additivity = False  # Skip for performance
        self.shap_approximate = True        # Use approximation for speed
        
        # LIME settings
        self.lime_num_features = 10
        self.lime_num_samples = 5000
        self.lime_discretize_continuous = True
        
        # Permutation importance
        self.permutation_n_repeats = 10
        self.permutation_scoring = None  # Auto-select
        
        # Partial dependence
        self.pdp_grid_resolution = 100
        self.pdp_percentiles = (0.05, 0.95)
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        self.batch_size = 1000
        
        # Output settings
        self.generate_visualizations = True
        self.save_explanations = True
        self.natural_language = True
        
        # Business settings
        self.include_business_impact = True
        self.feature_cost_mapping = {}  # Feature name -> cost
        self.actionable_features = []   # Features that can be changed
        
        # Advanced settings
        self.explanation_stability_check = True
        self.feature_interaction_analysis = True
        self.counterfactual_generation = False

@dataclass
class FeatureExplanation:
    """Explanation for a single feature."""
    feature_name: str
    importance_score: float
    confidence_interval: Optional[Tuple[float, float]]
    description: str
    business_impact: Optional[str]
    actionable: bool
    feature_type: str  # 'numeric', 'categorical', 'binary'
    value_range: Optional[Tuple[float, float]]
    top_values: Optional[List[Any]]

@dataclass
class InstanceExplanation:
    """Explanation for a single prediction instance."""
    instance_id: str
    prediction: Any
    prediction_probability: Optional[List[float]]
    feature_contributions: Dict[str, float]
    explanation_method: ExplainerMethod
    confidence_score: float
    natural_language_explanation: str
    counterfactuals: Optional[List[Dict[str, Any]]]
    similar_instances: Optional[List[str]]

@dataclass
class GlobalExplanation:
    """Global model explanation."""
    explanation_id: str
    timestamp: datetime
    model_name: Optional[str]
    explainer_method: ExplainerMethod
    feature_importances: Dict[str, FeatureExplanation]
    feature_interactions: Dict[str, Dict[str, float]]
    partial_dependence_data: Dict[str, Dict[str, Any]]
    model_behavior_summary: Dict[str, Any]
    business_insights: List[str]
    recommendations: List[str]
    visualizations: Dict[str, Any]
    metadata: Dict[str, Any]

class ModelTypeDetector:
    """Utility class to detect model type for appropriate explainer selection."""
    
    @staticmethod
    def detect_model_type(model: Any) -> ModelType:
        """Detect the type of ML model for explainer selection."""
        try:
            model_class = type(model).__name__.lower()
            model_module = type(model).__module__.lower()
            
            # Tree-based models
            tree_keywords = [
                'tree', 'forest', 'random', 'extra', 'gradient', 'boosting',
                'xgb', 'lgb', 'catboost', 'gbm'
            ]
            if any(keyword in model_class for keyword in tree_keywords):
                return ModelType.TREE_BASED
            
            # Linear models
            linear_keywords = [
                'linear', 'logistic', 'ridge', 'lasso', 'elastic',
                'sgd', 'perceptron'
            ]
            if any(keyword in model_class for keyword in linear_keywords):
                return ModelType.LINEAR
            
            # Neural networks
            nn_keywords = ['neural', 'mlp', 'keras', 'tensorflow', 'torch', 'pytorch']
            if any(keyword in model_class or keyword in model_module for keyword in nn_keywords):
                return ModelType.NEURAL_NETWORK
            
            # SVM
            if 'svm' in model_class or 'svc' in model_class or 'svr' in model_class:
                return ModelType.SVM
            
            # Naive Bayes
            if 'naive' in model_class or 'bayes' in model_class:
                return ModelType.NAIVE_BAYES
            
            # Ensemble methods
            ensemble_keywords = ['voting', 'bagging', 'ada', 'ensemble']
            if any(keyword in model_class for keyword in ensemble_keywords):
                return ModelType.ENSEMBLE
            
            return ModelType.UNKNOWN
            
        except Exception as e:
            logger.warning(f"Model type detection failed: {str(e)}")
            return ModelType.UNKNOWN

class ExplainerFactory:
    """Factory for creating appropriate explainers based on model type."""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
    
    def create_explainer(
        self,
        model: Any,
        X_background: np.ndarray,
        feature_names: List[str],
        model_type: Optional[ModelType] = None
    ) -> Tuple[Any, ExplainerMethod]:
        """Create the most appropriate explainer for the given model."""
        try:
            if model_type is None:
                model_type = ModelTypeDetector.detect_model_type(model)
            
            logger.info(f"Creating explainer for model type: {model_type.value}")
            
            # Try SHAP explainers first (most comprehensive)
            if SHAP_AVAILABLE:
                explainer, method = self._create_shap_explainer(
                    model, X_background, model_type
                )
                if explainer is not None:
                    return explainer, method
            
            # Fallback to LIME
            if LIME_AVAILABLE:
                explainer, method = self._create_lime_explainer(
                    model, X_background, feature_names
                )
                if explainer is not None:
                    return explainer, method
            
            # Fallback to built-in importance or permutation
            return None, ExplainerMethod.BUILT_IN_IMPORTANCE
            
        except Exception as e:
            logger.error(f"Explainer creation failed: {str(e)}")
            return None, ExplainerMethod.BUILT_IN_IMPORTANCE
    
    def _create_shap_explainer(
        self,
        model: Any,
        X_background: np.ndarray,
        model_type: ModelType
    ) -> Tuple[Any, ExplainerMethod]:
        """Create appropriate SHAP explainer."""
        try:
            # Tree-based models - use TreeExplainer
            if model_type == ModelType.TREE_BASED:
                try:
                    explainer = shap.TreeExplainer(
                        model,
                        X_background,
                        model_output=self.config.shap_tree_model_output,
                        check_additivity=self.config.shap_check_additivity
                    )
                    return explainer, ExplainerMethod.SHAP_TREE
                except Exception:
                    pass
            
            # Linear models - use LinearExplainer
            if model_type == ModelType.LINEAR:
                try:
                    explainer = shap.LinearExplainer(model, X_background)
                    return explainer, ExplainerMethod.SHAP_LINEAR
                except Exception:
                    pass
            
            # Neural networks - try DeepExplainer or GradientExplainer
            if model_type == ModelType.NEURAL_NETWORK:
                try:
                    if TENSORFLOW_AVAILABLE and hasattr(model, 'layers'):
                        # TensorFlow/Keras model
                        explainer = shap.DeepExplainer(model, X_background)
                        return explainer, ExplainerMethod.SHAP_DEEP
                    elif PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                        # PyTorch model
                        explainer = shap.GradientExplainer(model, X_background)
                        return explainer, ExplainerMethod.SHAP_GRADIENT
                except Exception:
                    pass
            
            # Fallback to KernelExplainer (model-agnostic but slower)
            try:
                # Limit background data for KernelExplainer performance
                bg_sample_size = min(self.config.background_sample_size, len(X_background))
                background_sample = X_background[:bg_sample_size]
                
                explainer = shap.KernelExplainer(model.predict, background_sample)
                return explainer, ExplainerMethod.SHAP_KERNEL
            except Exception:
                pass
            
            return None, ExplainerMethod.BUILT_IN_IMPORTANCE
            
        except Exception as e:
            logger.warning(f"SHAP explainer creation failed: {str(e)}")
            return None, ExplainerMethod.BUILT_IN_IMPORTANCE
    
    def _create_lime_explainer(
        self,
        model: Any,
        X_background: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Any, ExplainerMethod]:
        """Create LIME explainer for tabular data."""
        try:
            # Determine if classification or regression
            try:
                # Try to get prediction for classification detection
                test_pred = model.predict(X_background[:1])
                if hasattr(model, 'predict_proba'):
                    mode = 'classification'
                else:
                    mode = 'regression'
            except:
                mode = 'regression'  # Default
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_background,
                feature_names=feature_names,
                mode=mode,
                discretize_continuous=self.config.lime_discretize_continuous,
                random_state=self.config.random_state
            )
            
            return explainer, ExplainerMethod.LIME_TABULAR
            
        except Exception as e:
            logger.warning(f"LIME explainer creation failed: {str(e)}")
            return None, ExplainerMethod.BUILT_IN_IMPORTANCE

class ModelExplainer:
    """
    Comprehensive model explanation system with multiple interpretability methods,
    automatic explainer selection, and business-friendly insights generation.
    """
    
    def __init__(self, config: Optional[ExplanationConfig] = None):
        self.config = config or ExplanationConfig()
        self.explainer_factory = ExplainerFactory(self.config)
        self.explainer = None
        self.explainer_method = None
        self.feature_names = None
        self.model = None
        self.X_background = None
        self.explanation_cache = {}
        
        logger.info("ModelExplainer initialized")
    
    async def fit_explainer(
        self,
        model: Any,
        X_background: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Fit explainer to a trained model and background data.
        
        Args:
            model: Trained ML model
            X_background: Background/training data for explainer
            feature_names: Names of features
            model_name: Optional name for the model
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Fitting explainer for model: {model_name or 'Unknown'}")
            
            # Store model and background data
            self.model = model
            
            # Convert to numpy array
            if isinstance(X_background, pd.DataFrame):
                self.X_background = X_background.values
                if feature_names is None:
                    feature_names = X_background.columns.tolist()
            else:
                self.X_background = X_background
            
            # Set feature names
            if feature_names is None:
                self.feature_names = [f'feature_{i}' for i in range(self.X_background.shape[1])]
            else:
                self.feature_names = feature_names
            
            # Sample background data if too large
            if len(self.X_background) > self.config.explanation_sample_size:
                indices = np.random.choice(
                    len(self.X_background),
                    self.config.explanation_sample_size,
                    replace=False
                )
                self.X_background = self.X_background[indices]
                logger.info(f"Sampled {self.config.explanation_sample_size} background samples")
            
            # Create explainer
            self.explainer, self.explainer_method = self.explainer_factory.create_explainer(
                model, self.X_background, self.feature_names
            )
            
            if self.explainer is None:
                logger.warning("No compatible explainer found, using built-in methods")
            else:
                logger.info(f"Created {self.explainer_method.value} explainer")
            
            return True
            
        except Exception as e:
            logger.error(f"Explainer fitting failed: {str(e)}")
            return False
    
    async def explain_global(
        self,
        max_features: Optional[int] = None,
        include_interactions: bool = True,
        include_pdp: bool = True
    ) -> GlobalExplanation:
        """
        Generate global explanations for the model.
        
        Args:
            max_features: Maximum number of features to explain
            include_interactions: Whether to include feature interactions
            include_pdp: Whether to include partial dependence plots
            
        Returns:
            Global explanation object
        """
        try:
            logger.info("Generating global explanations")
            start_time = datetime.now()
            
            if self.model is None:
                raise ValueError("Model not fitted. Call fit_explainer first.")
            
            max_features = max_features or self.config.max_display_features
            
            # Calculate feature importances
            feature_importances = await self._calculate_global_feature_importance(max_features)
            
            # Feature interactions
            feature_interactions = {}
            if include_interactions and self.config.feature_interaction_analysis:
                feature_interactions = await self._calculate_feature_interactions(max_features)
            
            # Partial dependence plots
            pdp_data = {}
            if include_pdp:
                top_features = list(feature_importances.keys())[:5]  # Top 5 features
                pdp_data = await self._calculate_partial_dependence(top_features)
            
            # Model behavior summary
            behavior_summary = await self._analyze_model_behavior(feature_importances)
            
            # Generate business insights
            business_insights = self._generate_business_insights(
                feature_importances, behavior_summary
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                feature_importances, behavior_summary
            )
            
            # Prepare visualizations
            visualizations = await self._prepare_global_visualizations(
                feature_importances, feature_interactions, pdp_data
            )
            
            # Create global explanation
            explanation = GlobalExplanation(
                explanation_id=str(uuid.uuid4()),
                timestamp=start_time,
                model_name=getattr(self.model, '__class__', {}).get('__name__'),
                explainer_method=self.explainer_method,
                feature_importances=feature_importances,
                feature_interactions=feature_interactions,
                partial_dependence_data=pdp_data,
                model_behavior_summary=behavior_summary,
                business_insights=business_insights,
                recommendations=recommendations,
                visualizations=visualizations,
                metadata={
                    'n_features_explained': len(feature_importances),
                    'explanation_time': (datetime.now() - start_time).total_seconds(),
                    'background_samples': len(self.X_background),
                    'explainer_method': self.explainer_method.value
                }
            )
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE and self.config.save_explanations:
                await self._log_explanation_to_mlflow(explanation)
            
            logger.info(f"Global explanation generated in {explanation.metadata['explanation_time']:.2f}s")
            return explanation
            
        except Exception as e:
            logger.error(f"Global explanation failed: {str(e)}")
            return self._create_empty_global_explanation(str(e))
    
    async def explain_instance(
        self,
        instance: Union[pd.Series, np.ndarray, Dict],
        instance_id: Optional[str] = None,
        top_features: int = 10
    ) -> InstanceExplanation:
        """
        Generate explanation for a single prediction instance.
        
        Args:
            instance: Single data instance to explain
            instance_id: Optional identifier for the instance
            top_features: Number of top features to include in explanation
            
        Returns:
            Instance explanation object
        """
        try:
            if self.model is None:
                raise ValueError("Model not fitted. Call fit_explainer first.")
            
            # Convert instance to proper format
            if isinstance(instance, pd.Series):
                X_instance = instance.values.reshape(1, -1)
            elif isinstance(instance, dict):
                # Convert dict to array using feature names
                X_instance = np.array([
                    instance.get(feature, 0) for feature in self.feature_names
                ]).reshape(1, -1)
            else:
                X_instance = np.array(instance).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(X_instance)[0]
            
            # Get prediction probabilities if available
            pred_proba = None
            if hasattr(self.model, 'predict_proba'):
                pred_proba = self.model.predict_proba(X_instance)[0].tolist()
            
            # Calculate feature contributions
            feature_contributions = await self._calculate_instance_contributions(
                X_instance, top_features
            )
            
            # Generate natural language explanation
            nl_explanation = self._generate_natural_language_explanation(
                prediction, feature_contributions, pred_proba
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_explanation_confidence(
                feature_contributions, pred_proba
            )
            
            # Generate counterfactuals if enabled
            counterfactuals = None
            if self.config.counterfactual_generation:
                counterfactuals = await self._generate_counterfactuals(X_instance)
            
            instance_explanation = InstanceExplanation(
                instance_id=instance_id or str(uuid.uuid4()),
                prediction=prediction,
                prediction_probability=pred_proba,
                feature_contributions=feature_contributions,
                explanation_method=self.explainer_method,
                confidence_score=confidence_score,
                natural_language_explanation=nl_explanation,
                counterfactuals=counterfactuals,
                similar_instances=None  # Could implement similarity search
            )
            
            return instance_explanation
            
        except Exception as e:
            logger.error(f"Instance explanation failed: {str(e)}")
            return InstanceExplanation(
                instance_id=instance_id or str(uuid.uuid4()),
                prediction=None,
                prediction_probability=None,
                feature_contributions={},
                explanation_method=self.explainer_method,
                confidence_score=0.0,
                natural_language_explanation=f"Explanation failed: {str(e)}",
                counterfactuals=None,
                similar_instances=None
            )
    
    async def _calculate_global_feature_importance(
        self,
        max_features: int
    ) -> Dict[str, FeatureExplanation]:
        """Calculate global feature importance using the best available method."""
        try:
            feature_importances = {}
            
            # Method 1: SHAP global importance
            if self.explainer is not None and SHAP_AVAILABLE:
                importance_dict = await self._calculate_shap_global_importance()
                
            # Method 2: Built-in feature importance
            elif hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                importance_dict = {
                    self.feature_names[i]: float(imp) 
                    for i, imp in enumerate(importances)
                }
                
            # Method 3: Permutation importance
            else:
                importance_dict = await self._calculate_permutation_importance()
            
            # Convert to FeatureExplanation objects
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:max_features]
            
            for feature_name, importance in sorted_features:
                feature_idx = self.feature_names.index(feature_name)
                feature_values = self.X_background[:, feature_idx]
                
                # Determine feature type
                if len(np.unique(feature_values)) <= 10 and np.all(feature_values % 1 == 0):
                    feature_type = 'categorical'
                    value_range = None
                    top_values = list(np.unique(feature_values)[:5])
                elif len(np.unique(feature_values)) == 2:
                    feature_type = 'binary'
                    value_range = None
                    top_values = list(np.unique(feature_values))
                else:
                    feature_type = 'numeric'
                    value_range = (float(np.min(feature_values)), float(np.max(feature_values)))
                    top_values = None
                
                # Generate description
                description = self._generate_feature_description(
                    feature_name, importance, feature_type
                )
                
                # Business impact assessment
                business_impact = None
                if self.config.include_business_impact:
                    business_impact = self._assess_feature_business_impact(
                        feature_name, importance, feature_type
                    )
                
                feature_importances[feature_name] = FeatureExplanation(
                    feature_name=feature_name,
                    importance_score=float(importance),
                    confidence_interval=None,  # Could implement with bootstrap
                    description=description,
                    business_impact=business_impact,
                    actionable=feature_name in self.config.actionable_features,
                    feature_type=feature_type,
                    value_range=value_range,
                    top_values=top_values
                )
            
            return feature_importances
            
        except Exception as e:
            logger.error(f"Global feature importance calculation failed: {str(e)}")
            return {}
    
    async def _calculate_shap_global_importance(self) -> Dict[str, float]:
        """Calculate SHAP global feature importance."""
        try:
            # Calculate SHAP values for background data sample
            sample_size = min(self.config.background_sample_size, len(self.X_background))
            X_sample = self.X_background[:sample_size]
            
            if self.explainer_method == ExplainerMethod.SHAP_KERNEL:
                # For KernelExplainer, use smaller sample
                X_sample = X_sample[:min(50, len(X_sample))]
            
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class classification - use first class or average
                if len(shap_values) > 1:
                    shap_values = np.mean(shap_values, axis=0)
                else:
                    shap_values = shap_values[0]
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            importance_dict = {
                self.feature_names[i]: float(importance)
                for i, importance in enumerate(mean_shap_values)
            }
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"SHAP global importance failed: {str(e)}")
            return {}
    
    async def _calculate_permutation_importance(self) -> Dict[str, float]:
        """Calculate permutation importance."""
        try:
            # Determine scoring method
            scoring = self.config.permutation_scoring
            if scoring is None:
                if hasattr(self.model, 'predict_proba'):
                    scoring = 'accuracy'
                else:
                    scoring = 'r2'
            
            # Calculate permutation importance
            result = permutation_importance(
                self.model,
                self.X_background,
                self.model.predict(self.X_background),
                n_repeats=self.config.permutation_n_repeats,
                scoring=scoring,
                random_state=self.config.random_state,
                n_jobs=1  # Avoid nested parallelism
            )
            
            importance_dict = {
                self.feature_names[i]: float(importance)
                for i, importance in enumerate(result.importances_mean)
            }
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Permutation importance failed: {str(e)}")
            return {}
    
    async def _calculate_feature_interactions(self, max_features: int) -> Dict[str, Dict[str, float]]:
        """Calculate feature interactions using SHAP interaction values."""
        try:
            if not SHAP_AVAILABLE or self.explainer is None:
                return {}
            
            # Only calculate for tree-based models (most efficient)
            if self.explainer_method != ExplainerMethod.SHAP_TREE:
                return {}
            
            # Use small sample for performance
            sample_size = min(100, len(self.X_background))
            X_sample = self.X_background[:sample_size]
            
            try:
                interaction_values = self.explainer.shap_interaction_values(X_sample)
                
                # interaction_values shape: (n_samples, n_features, n_features)
                mean_interactions = np.mean(np.abs(interaction_values), axis=0)
                
                interactions = {}
                feature_subset = self.feature_names[:max_features]
                
                for i, feat1 in enumerate(feature_subset):
                    if i < len(self.feature_names):
                        interactions[feat1] = {}
                        for j, feat2 in enumerate(feature_subset):
                            if j < len(self.feature_names) and i != j:
                                if i < mean_interactions.shape[0] and j < mean_interactions.shape[1]:
                                    interactions[feat1][feat2] = float(mean_interactions[i, j])
                
                return interactions
                
            except Exception:
                # Fallback: approximate interactions using correlation of SHAP values
                shap_values = self.explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                interactions = {}
                for i, feat1 in enumerate(self.feature_names[:max_features]):
                    interactions[feat1] = {}
                    for j, feat2 in enumerate(self.feature_names[:max_features]):
                        if i != j and i < shap_values.shape[1] and j < shap_values.shape[1]:
                            correlation = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                            interactions[feat1][feat2] = float(abs(correlation))
                
                return interactions
                
        except Exception as e:
            logger.warning(f"Feature interaction calculation failed: {str(e)}")
            return {}
    
    async def _calculate_partial_dependence(self, features: List[str]) -> Dict[str, Dict[str, Any]]:
        """Calculate partial dependence plots for top features."""
        try:
            pdp_data = {}
            
            for feature in features:
                if feature not in self.feature_names:
                    continue
                
                feature_idx = self.feature_names.index(feature)
                
                try:
                    # Calculate partial dependence
                    pd_results = partial_dependence(
                        self.model,
                        self.X_background,
                        features=[feature_idx],
                        grid_resolution=self.config.pdp_grid_resolution,
                        percentiles=self.config.pdp_percentiles
                    )
                    
                    pdp_values = pd_results['average'][0]
                    feature_grid = pd_results['grid'][0]
                    
                    pdp_data[feature] = {
                        'grid_values': feature_grid.tolist(),
                        'pdp_values': pdp_values.tolist(),
                        'feature_range': (float(feature_grid.min()), float(feature_grid.max()))
                    }
                    
                except Exception as e:
                    logger.warning(f"PDP calculation failed for {feature}: {str(e)}")
                    continue
            
            return pdp_data
            
        except Exception as e:
            logger.warning(f"Partial dependence calculation failed: {str(e)}")
            return {}
    
    async def _calculate_instance_contributions(
        self,
        X_instance: np.ndarray,
        top_features: int
    ) -> Dict[str, float]:
        """Calculate feature contributions for a single instance."""
        try:
            contributions = {}
            
            # Method 1: SHAP values
            if self.explainer is not None and SHAP_AVAILABLE:
                try:
                    shap_values = self.explainer.shap_values(X_instance)
                    
                    if isinstance(shap_values, list):
                        # Multi-class - use first class or class with highest probability
                        if len(shap_values) > 1 and hasattr(self.model, 'predict_proba'):
                            proba = self.model.predict_proba(X_instance)[0]
                            class_idx = np.argmax(proba)
                            shap_values = shap_values[class_idx]
                        else:
                            shap_values = shap_values[0]
                    
                    # Extract values for single instance
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                    
                    contributions = {
                        self.feature_names[i]: float(value)
                        for i, value in enumerate(shap_values)
                        if i < len(self.feature_names)
                    }
                    
                except Exception as e:
                    logger.warning(f"SHAP instance explanation failed: {str(e)}")
            
            # Method 2: LIME
            if not contributions and self.explainer is not None and LIME_AVAILABLE:
                try:
                    if self.explainer_method == ExplainerMethod.LIME_TABULAR:
                        explanation = self.explainer.explain_instance(
                            X_instance[0],
                            self.model.predict,
                            num_features=top_features,
                            num_samples=self.config.lime_num_samples
                        )
                        
                        # Extract feature contributions
                        for feature_idx, contribution in explanation.as_list():
                            if isinstance(feature_idx, str):
                                contributions[feature_idx] = float(contribution)
                            elif feature_idx < len(self.feature_names):
                                contributions[self.feature_names[feature_idx]] = float(contribution)
                                
                except Exception as e:
                    logger.warning(f"LIME instance explanation failed: {str(e)}")
            
            # Method 3: Linear model coefficients (for linear models)
            if not contributions and hasattr(self.model, 'coef_'):
                try:
                    coef = self.model.coef_
                    if len(coef.shape) > 1:
                        coef = coef[0]  # Use first class for multi-class
                    
                    # Multiply coefficients by feature values
                    feature_values = X_instance[0]
                    contributions = {
                        self.feature_names[i]: float(coef[i] * feature_values[i])
                        for i in range(min(len(coef), len(self.feature_names)))
                    }
                    
                except Exception as e:
                    logger.warning(f"Linear coefficients extraction failed: {str(e)}")
            
            # Sort by absolute contribution and return top features
            if contributions:
                sorted_contributions = sorted(
                    contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:top_features]
                contributions = dict(sorted_contributions)
            
            return contributions
            
        except Exception as e:
            logger.error(f"Instance contribution calculation failed: {str(e)}")
            return {}
    
    async def _analyze_model_behavior(
        self,
        feature_importances: Dict[str, FeatureExplanation]
    ) -> Dict[str, Any]:
        """Analyze overall model behavior patterns."""
        try:
            behavior = {}
            
            if not feature_importances:
                return behavior
            
            # Extract importance scores
            importance_scores = [
                feat.importance_score for feat in feature_importances.values()
            ]
            
            # Calculate distribution statistics
            behavior['importance_concentration'] = {
                'gini_coefficient': self._calculate_gini_coefficient(importance_scores),
                'top_feature_dominance': max(importance_scores) / sum(importance_scores),
                'effective_features': sum(1 for score in importance_scores if score > 0.01)
            }
            
            # Feature type analysis
            feature_types = [feat.feature_type for feat in feature_importances.values()]
            type_counts = {ftype: feature_types.count(ftype) for ftype in set(feature_types)}
            behavior['feature_type_distribution'] = type_counts
            
            # Actionable features analysis
            actionable_count = sum(1 for feat in feature_importances.values() if feat.actionable)
            behavior['actionability'] = {
                'actionable_features': actionable_count,
                'total_features': len(feature_importances),
                'actionable_ratio': actionable_count / len(feature_importances)
            }
            
            return behavior
            
        except Exception as e:
            logger.warning(f"Model behavior analysis failed: {str(e)}")
            return {}
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for importance concentration."""
        try:
            if not values or len(values) < 2:
                return 0.0
            
            # Sort values
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(sorted_values)
            return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
            
        except Exception:
            return 0.0
    
    def _generate_feature_description(
        self,
        feature_name: str,
        importance: float,
        feature_type: str
    ) -> str:
        """Generate human-readable description for a feature."""
        try:
            importance_level = "high" if importance > 0.1 else "medium" if importance > 0.05 else "low"
            
            descriptions = {
                'numeric': f"{feature_name} is a numeric feature with {importance_level} importance for predictions.",
                'categorical': f"{feature_name} is a categorical feature with {importance_level} importance for predictions.",
                'binary': f"{feature_name} is a binary feature with {importance_level} importance for predictions."
            }
            
            return descriptions.get(feature_type, f"{feature_name} has {importance_level} importance for predictions.")
            
        except Exception:
            return f"{feature_name} contributes to model predictions."
    
    def _assess_feature_business_impact(
        self,
        feature_name: str,
        importance: float,
        feature_type: str
    ) -> str:
        """Assess business impact of a feature."""
        try:
            if importance > 0.2:
                impact = "High business impact - key driver of predictions"
            elif importance > 0.1:
                impact = "Medium business impact - significant contributor"
            elif importance > 0.05:
                impact = "Low business impact - minor contributor"
            else:
                impact = "Minimal business impact"
            
            # Add actionability context
            if feature_name in self.config.actionable_features:
                impact += " (actionable feature - can be influenced)"
            
            # Add cost context if available
            if feature_name in self.config.feature_cost_mapping:
                cost = self.config.feature_cost_mapping[feature_name]
                impact += f" (data collection cost: {cost})"
            
            return impact
            
        except Exception:
            return "Business impact assessment unavailable"
    
    def _generate_business_insights(
        self,
        feature_importances: Dict[str, FeatureExplanation],
        behavior_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate business-oriented insights from explanations."""
        try:
            insights = []
            
            if not feature_importances:
                return ["No feature importance data available for insights."]
            
            # Top feature insights
            top_feature = max(feature_importances.values(), key=lambda x: x.importance_score)
            insights.append(
                f"The most important feature is '{top_feature.feature_name}' "
                f"with {top_feature.importance_score:.1%} relative importance."
            )
            
            # Concentration insights
            if 'importance_concentration' in behavior_summary:
                dominance = behavior_summary['importance_concentration']['top_feature_dominance']
                if dominance > 0.5:
                    insights.append(
                        "Model predictions are heavily dominated by a single feature, "
                        "which may indicate overfitting or data quality issues."
                    )
                elif dominance < 0.1:
                    insights.append(
                        "Model uses a well-distributed set of features, "
                        "indicating robust and balanced predictions."
                    )
            
            # Actionability insights
            if 'actionability' in behavior_summary:
                actionable_ratio = behavior_summary['actionability']['actionable_ratio']
                if actionable_ratio > 0.7:
                    insights.append(
                        "Most important features are actionable, "
                        "providing good opportunities for intervention."
                    )
                elif actionable_ratio < 0.3:
                    insights.append(
                        "Few important features are actionable, "
                        "limiting opportunities for direct intervention."
                    )
            
            # Feature type insights
            if 'feature_type_distribution' in behavior_summary:
                type_dist = behavior_summary['feature_type_distribution']
                if type_dist.get('categorical', 0) > type_dist.get('numeric', 0):
                    insights.append(
                        "Model relies more heavily on categorical features, "
                        "suggesting rule-based decision patterns."
                    )
            
            return insights
            
        except Exception as e:
            logger.warning(f"Business insights generation failed: {str(e)}")
            return ["Business insights generation encountered an error."]
    
    def _generate_recommendations(
        self,
        feature_importances: Dict[str, FeatureExplanation],
        behavior_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on explanations."""
        try:
            recommendations = []
            
            if not feature_importances:
                return ["No feature importance data available for recommendations."]
            
            # Feature engineering recommendations
            low_importance_features = [
                name for name, feat in feature_importances.items() 
                if feat.importance_score < 0.01
            ]
            
            if len(low_importance_features) > 3:
                recommendations.append(
                    f"Consider removing {len(low_importance_features)} low-importance features "
                    "to reduce model complexity and improve interpretability."
                )
            
            # Data quality recommendations
            if 'importance_concentration' in behavior_summary:
                dominance = behavior_summary['importance_concentration']['top_feature_dominance']
                if dominance > 0.8:
                    recommendations.append(
                        "High feature dominance detected. Investigate potential data leakage "
                        "or consider collecting additional diverse features."
                    )
            
            # Business action recommendations
            actionable_features = [
                feat for feat in feature_importances.values() 
                if feat.actionable and feat.importance_score > 0.1
            ]
            
            if actionable_features:
                top_actionable = max(actionable_features, key=lambda x: x.importance_score)
                recommendations.append(
                    f"Focus business interventions on '{top_actionable.feature_name}' "
                    f"as it's both highly important and actionable."
                )
            
            # Model monitoring recommendations
            recommendations.append(
                "Monitor feature importance stability over time to detect "
                "model drift and data distribution changes."
            )
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {str(e)}")
            return ["Recommendations generation encountered an error."]
    
    def _generate_natural_language_explanation(
        self,
        prediction: Any,
        feature_contributions: Dict[str, float],
        pred_proba: Optional[List[float]]
    ) -> str:
        """Generate natural language explanation for an instance."""
        try:
            if not feature_contributions:
                return "Unable to generate explanation due to insufficient data."
            
            # Sort contributions by absolute value
            sorted_contribs = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Build explanation
            explanation_parts = []
            
            # Prediction statement
            if pred_proba is not None:
                if len(pred_proba) == 2:  # Binary classification
                    confidence = max(pred_proba)
                    prediction_text = f"predicted class {prediction} with {confidence:.1%} confidence"
                else:  # Multi-class
                    confidence = max(pred_proba)
                    prediction_text = f"predicted class {prediction} with {confidence:.1%} confidence"
            else:
                prediction_text = f"predicted value of {prediction:.3f}" if isinstance(prediction, (int, float)) else f"predicted {prediction}"
            
            explanation_parts.append(f"The model {prediction_text}.")
            
            # Top contributing features
            top_positive = [item for item in sorted_contribs if item[1] > 0][:3]
            top_negative = [item for item in sorted_contribs if item[1] < 0][:3]
            
            if top_positive:
                pos_features = [f"{name} (+{abs(contrib):.3f})" for name, contrib in top_positive]
                explanation_parts.append(
                    f"The strongest positive contributors are: {', '.join(pos_features)}."
                )
            
            if top_negative:
                neg_features = [f"{name} (-{abs(contrib):.3f})" for name, contrib in top_negative]
                explanation_parts.append(
                    f"The strongest negative contributors are: {', '.join(neg_features)}."
                )
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.warning(f"Natural language explanation failed: {str(e)}")
            return f"Prediction: {prediction}. Feature contributions available but explanation generation failed."
    
    def _calculate_explanation_confidence(
        self,
        feature_contributions: Dict[str, float],
        pred_proba: Optional[List[float]]
    ) -> float:
        """Calculate confidence score for the explanation."""
        try:
            # Base confidence on prediction probability if available
            if pred_proba is not None:
                prediction_confidence = max(pred_proba)
            else:
                prediction_confidence = 0.5
            
            # Adjust based on feature contribution concentration
            if feature_contributions:
                contrib_values = list(feature_contributions.values())
                total_contrib = sum(abs(val) for val in contrib_values)
                if total_contrib > 0:
                    # Higher concentration of contributions = higher confidence
                    top_3_contrib = sum(abs(val) for val in sorted(contrib_values, key=abs, reverse=True)[:3])
                    contribution_confidence = min(1.0, top_3_contrib / total_contrib)
                else:
                    contribution_confidence = 0.0
            else:
                contribution_confidence = 0.0
            
            # Combined confidence
            return float((prediction_confidence + contribution_confidence) / 2)
            
        except Exception:
            return 0.5  # Default confidence
    
    async def _generate_counterfactuals(self, X_instance: np.ndarray) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations (simplified implementation)."""
        try:
            # This is a simplified counterfactual generation
            # In practice, you'd use dedicated libraries like DiCE or Alibi
            
            counterfactuals = []
            
            # Get original prediction
            original_pred = self.model.predict(X_instance)[0]
            
            # Try modifying each feature slightly
            for i, feature_name in enumerate(self.feature_names[:5]):  # Top 5 features only
                modified_instance = X_instance.copy()
                
                # Modify feature value
                original_value = modified_instance[0, i]
                
                # Try increasing and decreasing the value
                for direction in [1.1, 0.9]:  # +10% and -10%
                    modified_instance[0, i] = original_value * direction
                    new_pred = self.model.predict(modified_instance)[0]
                    
                    # If prediction changed significantly, save as counterfactual
                    if abs(new_pred - original_pred) > 0.1:  # Threshold for significant change
                        counterfactual = {
                            'feature_changed': feature_name,
                            'original_value': float(original_value),
                            'new_value': float(modified_instance[0, i]),
                            'original_prediction': float(original_pred),
                            'new_prediction': float(new_pred),
                            'change_description': f"Changing {feature_name} from {original_value:.3f} to {modified_instance[0, i]:.3f}"
                        }
                        counterfactuals.append(counterfactual)
                        
                        if len(counterfactuals) >= 3:  # Limit to 3 counterfactuals
                            break
                
                # Restore original value
                modified_instance[0, i] = original_value
                
                if len(counterfactuals) >= 3:
                    break
            
            return counterfactuals
            
        except Exception as e:
            logger.warning(f"Counterfactual generation failed: {str(e)}")
            return []
    
    async def _prepare_global_visualizations(
        self,
        feature_importances: Dict[str, FeatureExplanation],
        feature_interactions: Dict[str, Dict[str, float]],
        pdp_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare visualization data for global explanations."""
        try:
            if not self.config.generate_visualizations:
                return {}
            
            visualizations = {}
            
            # Feature importance bar chart
            if feature_importances:
                sorted_features = sorted(
                    feature_importances.items(),
                    key=lambda x: x[1].importance_score,
                    reverse=True
                )[:15]  # Top 15 features
                
                visualizations['feature_importance'] = {
                    'type': 'bar_chart',
                    'data': {
                        'features': [item[0] for item in sorted_features],
                        'importance': [item[1].importance_score for item in sorted_features],
                        'title': 'Feature Importance',
                        'x_label': 'Features',
                        'y_label': 'Importance Score'
                    }
                }
            
            # Feature interaction heatmap
            if feature_interactions:
                visualizations['feature_interactions'] = {
                    'type': 'heatmap',
                    'data': {
                        'interactions': feature_interactions,
                        'title': 'Feature Interactions',
                        'color_scale': 'viridis'
                    }
                }
            
            # Partial dependence plots
            if pdp_data:
                visualizations['partial_dependence'] = {
                    'type': 'line_plots',
                    'data': {
                        'plots': pdp_data,
                        'title': 'Partial Dependence Plots',
                        'x_label': 'Feature Value',
                        'y_label': 'Partial Dependence'
                    }
                }
            
            return visualizations
            
        except Exception as e:
            logger.warning(f"Global visualization preparation failed: {str(e)}")
            return {}
    
    def _create_empty_global_explanation(self, error_message: str) -> GlobalExplanation:
        """Create empty global explanation with error message."""
        return GlobalExplanation(
            explanation_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            model_name=None,
            explainer_method=self.explainer_method or ExplainerMethod.BUILT_IN_IMPORTANCE,
            feature_importances={},
            feature_interactions={},
            partial_dependence_data={},
            model_behavior_summary={},
            business_insights=[f"Explanation failed: {error_message}"],
            recommendations=["Unable to generate recommendations due to explanation failure"],
            visualizations={},
            metadata={'error': error_message}
        )
    
    async def _log_explanation_to_mlflow(self, explanation: GlobalExplanation):
        """Log explanation results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"explanation_{explanation.model_name or 'unknown'}"):
                # Log parameters
                mlflow.log_param("explainer_method", explanation.explainer_method.value)
                mlflow.log_param("model_name", explanation.model_name or "unknown")
                mlflow.log_param("n_features_explained", len(explanation.feature_importances))
                
                # Log feature importance metrics
                for feature_name, feature_exp in explanation.feature_importances.items():
                    mlflow.log_metric(f"importance_{feature_name}", feature_exp.importance_score)
                
                # Log explanation metadata
                if 'explanation_time' in explanation.metadata:
                    mlflow.log_metric("explanation_time", explanation.metadata['explanation_time'])
                
                # Log explanation report as JSON
                explanation_dict = asdict(explanation)
                explanation_dict['timestamp'] = explanation.timestamp.isoformat()
                
                with open("explanation_report.json", "w") as f:
                    json.dump(explanation_dict, f, indent=2, default=str)
                mlflow.log_artifact("explanation_report.json")
                
                logger.info("Explanation logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow explanation logging failed: {str(e)}")
    
    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get summary of explainer capabilities and status."""
        try:
            summary = {
                'explainer_fitted': self.model is not None,
                'explainer_method': self.explainer_method.value if self.explainer_method else None,
                'available_libraries': {
                    'shap': SHAP_AVAILABLE,
                    'lime': LIME_AVAILABLE,
                    'eli5': ELI5_AVAILABLE,
                    'interpret': INTERPRET_AVAILABLE
                },
                'supported_explanation_types': [etype.value for etype in ExplanationType],
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'background_samples': len(self.X_background) if self.X_background is not None else 0,
                'configuration': asdict(self.config)
            }
            
            if self.model is not None:
                model_type = ModelTypeDetector.detect_model_type(self.model)
                summary['detected_model_type'] = model_type.value
            
            return summary
            
        except Exception as e:
            logger.error(f"Explanation summary generation failed: {str(e)}")
            return {'error': str(e)}

# Utility functions

def create_explainer(
    max_features: int = 20,
    explanation_sample_size: int = 1000,
    enable_interactions: bool = True
) -> ModelExplainer:
    """Factory function to create a ModelExplainer."""
    config = ExplanationConfig()
    config.max_display_features = max_features
    config.explanation_sample_size = explanation_sample_size
    config.feature_interaction_analysis = enable_interactions
    return ModelExplainer(config)

async def quick_explain(
    model: Any,
    X_background: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    max_features: int = 10
) -> Dict[str, float]:
    """Quick model explanation for simple use cases."""
    explainer = create_explainer(max_features=max_features)
    await explainer.fit_explainer(model, X_background, feature_names)
    
    global_explanation = await explainer.explain_global(max_features=max_features)
    
    return {
        name: feat.importance_score 
        for name, feat in global_explanation.feature_importances.items()
    }

def get_available_explainers() -> Dict[str, bool]:
    """Get available explanation libraries and methods."""
    return {
        'shap': SHAP_AVAILABLE,
        'lime': LIME_AVAILABLE,
        'eli5': ELI5_AVAILABLE,
        'interpret': INTERPRET_AVAILABLE,
        'sklearn_inspection': True,  # Always available
        'built_in_importance': True  # Always available
    }

def get_explanation_recommendations(
    model_type: str,
    dataset_size: int,
    n_features: int
) -> Dict[str, str]:
    """Get recommendations for explanation configuration."""
    recommendations = {}
    
    # Model type recommendations
    if model_type in ['tree_based', 'ensemble']:
        recommendations['primary_method'] = "SHAP TreeExplainer for fast, exact explanations"
        recommendations['interactions'] = "Enable feature interactions for tree-based models"
    elif model_type == 'linear':
        recommendations['primary_method'] = "SHAP LinearExplainer or built-in coefficients"
        recommendations['interactions'] = "Linear interactions less critical"
    elif model_type == 'neural_network':
        recommendations['primary_method'] = "SHAP DeepExplainer or GradientExplainer"
        recommendations['interactions'] = "Consider gradient-based explanations"
    else:
        recommendations['primary_method'] = "SHAP KernelExplainer (model-agnostic)"
    
    # Dataset size recommendations
    if dataset_size > 10000:
        recommendations['sampling'] = "Use sampling for background data (1000-5000 samples)"
        recommendations['performance'] = "Enable approximation for faster explanations"
    else:
        recommendations['sampling'] = "Can use full dataset for background"
    
    # Feature count recommendations
    if n_features > 50:
        recommendations['feature_selection'] = "Focus on top 20-30 most important features"
        recommendations['visualization'] = "Use feature grouping for better visualization"
    
    return recommendations

# Example usage and testing
if __name__ == "__main__":
    async def test_explainer():
        """Test the explainer functionality."""
        from sklearn.datasets import make_classification, make_regression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        
        print("Testing Model Explainer...")
        print("Available explainers:", get_available_explainers())
        
        # Test classification
        print("\n=== Classification Explanation Test ===")
        X_class, y_class = make_classification(
            n_samples=1000, n_features=10, n_informative=7,
            n_redundant=3, n_classes=2, random_state=42
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X_class.shape[1])]
        X_class_df = pd.DataFrame(X_class, columns=feature_names)
        
        # Train model
        rf_class = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_class.fit(X_class, y_class)
        
        # Create explainer
        explainer = create_explainer(max_features=10)
        
        # Fit explainer
        await explainer.fit_explainer(rf_class, X_class_df, model_name="RandomForestClassifier")
        
        # Global explanation
        global_exp = await explainer.explain_global(max_features=8)
        print(f"Top features: {list(global_exp.feature_importances.keys())[:3]}")
        print(f"Business insights: {len(global_exp.business_insights)}")
        print(f"Recommendations: {len(global_exp.recommendations)}")
        
        # Instance explanation
        test_instance = X_class_df.iloc[0]
        instance_exp = await explainer.explain_instance(test_instance, top_features=5)
        print(f"Instance prediction: {instance_exp.prediction}")
        print(f"Top contributing features: {list(instance_exp.feature_contributions.keys())[:3]}")
        print(f"Confidence: {instance_exp.confidence_score:.3f}")
        
        # Test regression
        print("\n=== Regression Explanation Test ===")
        X_reg, y_reg = make_regression(
            n_samples=500, n_features=8, noise=0.1, random_state=42
        )
        
        feature_names_reg = [f'reg_feature_{i}' for i in range(X_reg.shape[1])]
        X_reg_df = pd.DataFrame(X_reg, columns=feature_names_reg)
        
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_reg.fit(X_reg, y_reg)
        
        explainer_reg = create_explainer(max_features=8)
        await explainer_reg.fit_explainer(rf_reg, X_reg_df, model_name="RandomForestRegressor")
        
        global_exp_reg = await explainer_reg.explain_global(max_features=6)
        print(f"Regression top features: {list(global_exp_reg.feature_importances.keys())[:3]}")
        
        # Quick explain test
        print("\n=== Quick Explain Test ===")
        quick_importance = await quick_explain(rf_class, X_class_df, max_features=5)
        print(f"Quick explanation: {list(quick_importance.keys())[:3]}")
        
        # Get recommendations
        print("\n=== Recommendations ===")
        recommendations = get_explanation_recommendations('tree_based', 1000, 10)
        for key, value in recommendations.items():
            print(f"{key}: {value}")
        
        return global_exp, instance_exp
    
    # Run test
    import asyncio
    results = asyncio.run(test_explainer())
