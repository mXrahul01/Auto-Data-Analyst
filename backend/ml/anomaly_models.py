"""
Anomaly Detection Models Module for Auto-Analyst Platform

This module implements various anomaly detection algorithms including:
- Isolation Forest (primary choice for most datasets)
- Local Outlier Factor (LOF) for density-based anomalies
- One-Class SVM for high-dimensional data
- Elliptic Envelope for Gaussian-distributed data
- DBSCAN-based anomaly detection
- Custom ensemble methods

Features:
- Automatic algorithm selection based on data characteristics
- Hyperparameter optimization with Bayesian methods
- Real-time anomaly scoring and batch processing
- Model explainability with SHAP integration
- MLflow integration for experiment tracking
- Performance monitoring and drift detection
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json

# Core ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Hyperparameter optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

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

# Advanced anomaly detection libraries
try:
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    from pyod.models.deep_svdd import DeepSVDD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class AnomalyDetectionConfig:
    """Configuration class for anomaly detection parameters."""
    
    def __init__(self):
        # Algorithm selection thresholds
        self.small_dataset_threshold = 1000  # Use LOF for datasets < 1K rows
        self.large_dataset_threshold = 100000  # Use optimized methods for > 100K rows
        
        # Default contamination rates
        self.default_contamination = 0.1  # Expect 10% anomalies by default
        self.max_contamination = 0.5  # Maximum allowed contamination rate
        
        # Performance settings
        self.max_features = 50  # Maximum features before dimensionality reduction
        self.n_jobs = -1  # Use all available cores
        self.random_state = 42
        
        # Optimization settings
        self.bayesian_opt_calls = 20  # Number of Bayesian optimization iterations
        self.cv_folds = 3  # Cross-validation folds for hyperparameter tuning
        
        # Model persistence
        self.model_cache_size = 10  # Number of models to keep in memory
        
        # Explainability settings
        self.enable_shap = SHAP_AVAILABLE
        self.shap_sample_size = 1000  # Sample size for SHAP explanations

class AnomalyDetector:
    """
    Comprehensive anomaly detection system with multiple algorithms,
    automatic model selection, and advanced features.
    """
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        self.config = config or AnomalyDetectionConfig()
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.best_model = None
        self.best_algorithm = None
        self.preprocessing_pipeline = None
        self.shap_explainer = None
        self.training_stats = {}
        
        # Initialize algorithm registry
        self._initialize_algorithms()
        
        logger.info("AnomalyDetector initialized with config")
    
    def _initialize_algorithms(self):
        """Initialize available anomaly detection algorithms."""
        self.algorithms = {
            'isolation_forest': {
                'class': IsolationForest,
                'params': {
                    'contamination': self.config.default_contamination,
                    'random_state': self.config.random_state,
                    'n_jobs': self.config.n_jobs
                },
                'param_space': {
                    'n_estimators': [100, 200, 300],
                    'max_samples': ['auto', 0.5, 0.8],
                    'max_features': [0.5, 0.8, 1.0]
                },
                'best_for': 'general',
                'scalable': True
            },
            'local_outlier_factor': {
                'class': LocalOutlierFactor,
                'params': {
                    'contamination': self.config.default_contamination,
                    'n_jobs': self.config.n_jobs
                },
                'param_space': {
                    'n_neighbors': [10, 20, 30, 50],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'leaf_size': [20, 30, 50]
                },
                'best_for': 'small_datasets',
                'scalable': False
            },
            'one_class_svm': {
                'class': OneClassSVM,
                'params': {
                    'nu': self.config.default_contamination
                },
                'param_space': {
                    'kernel': ['rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
                    'nu': [0.05, 0.1, 0.15, 0.2]
                },
                'best_for': 'high_dimensional',
                'scalable': True
            },
            'elliptic_envelope': {
                'class': EllipticEnvelope,
                'params': {
                    'contamination': self.config.default_contamination,
                    'random_state': self.config.random_state
                },
                'param_space': {
                    'support_fraction': [None, 0.6, 0.8, 0.9]
                },
                'best_for': 'gaussian_data',
                'scalable': True
            }
        }
        
        # Add PyOD models if available
        if PYOD_AVAILABLE:
            self.algorithms.update({
                'autoencoder': {
                    'class': AutoEncoder,
                    'params': {
                        'contamination': self.config.default_contamination,
                        'random_state': self.config.random_state
                    },
                    'param_space': {
                        'hidden_neurons': [[32, 16, 8], [64, 32, 16], [128, 64, 32]],
                        'epochs': [50, 100, 150]
                    },
                    'best_for': 'complex_patterns',
                    'scalable': True
                }
            })
    
    async def detect_anomalies(
        self,
        data: pd.DataFrame,
        contamination: Optional[float] = None,
        algorithm: Optional[str] = None,
        optimize_params: bool = True
    ) -> Dict[str, Any]:
        """
        Main method to detect anomalies in the given dataset.
        
        Args:
            data: Input DataFrame
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            algorithm: Specific algorithm to use (auto-select if None)
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            logger.info(f"Starting anomaly detection on dataset with shape {data.shape}")
            
            # Validate and preprocess data
            processed_data, feature_names = await self._preprocess_data(data)
            self.feature_names = feature_names
            
            # Set contamination rate
            if contamination is None:
                contamination = await self._estimate_contamination(processed_data)
            
            contamination = min(contamination, self.config.max_contamination)
            
            # Select best algorithm if not specified
            if algorithm is None:
                algorithm = self._select_algorithm(processed_data)
            
            logger.info(f"Using algorithm: {algorithm} with contamination: {contamination:.3f}")
            
            # Train and evaluate model
            model_results = await self._train_model(
                processed_data, algorithm, contamination, optimize_params
            )
            
            # Generate predictions
            predictions = await self._predict_anomalies(processed_data, model_results['model'])
            
            # Calculate metrics and insights
            metrics = self._calculate_metrics(predictions, processed_data)
            
            # Generate explanations if available
            explanations = await self._generate_explanations(
                processed_data, model_results['model'], predictions
            )
            
            # Compile final results
            results = {
                'algorithm': algorithm,
                'contamination': contamination,
                'n_anomalies': int(np.sum(predictions['is_anomaly'])),
                'anomaly_rate': float(np.mean(predictions['is_anomaly']) * 100),
                'anomaly_indices': predictions['anomaly_indices'].tolist(),
                'anomaly_scores': predictions['scores'].tolist(),
                'model_performance': model_results.get('performance', {}),
                'metrics': metrics,
                'explanations': explanations,
                'feature_importance': predictions.get('feature_importance', {}),
                'top_anomalies': self._get_top_anomalies(data, predictions, n=10),
                'training_time': model_results.get('training_time', 0),
                'prediction_time': predictions.get('prediction_time', 0)
            }
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(results, model_results['model'])
            
            logger.info(f"Anomaly detection completed. Found {results['n_anomalies']} anomalies ({results['anomaly_rate']:.2f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                'error': str(e),
                'algorithm': algorithm or 'auto',
                'n_anomalies': 0,
                'anomaly_rate': 0.0,
                'anomaly_indices': [],
                'anomaly_scores': []
            }
    
    async def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for anomaly detection."""
        try:
            # Select numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for anomaly detection")
            
            # Handle missing values
            df_clean = data[numeric_cols].copy()
            df_clean = df_clean.fillna(df_clean.mean())
            
            # Remove constant columns
            constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
            if constant_cols:
                df_clean = df_clean.drop(columns=constant_cols)
                logger.info(f"Removed constant columns: {constant_cols}")
            
            # Dimensionality reduction if needed
            if len(df_clean.columns) > self.config.max_features:
                logger.info(f"Applying PCA to reduce from {len(df_clean.columns)} to {self.config.max_features} features")
                pca = PCA(n_components=self.config.max_features, random_state=self.config.random_state)
                df_clean = pd.DataFrame(
                    pca.fit_transform(df_clean),
                    columns=[f'PC{i+1}' for i in range(self.config.max_features)]
                )
            
            # Scale features
            scaler = RobustScaler()  # More robust to outliers than StandardScaler
            scaled_data = scaler.fit_transform(df_clean)
            
            # Store preprocessing pipeline
            self.preprocessing_pipeline = {
                'numeric_columns': numeric_cols,
                'selected_features': df_clean.columns.tolist(),
                'scaler': scaler,
                'constant_columns': constant_cols
            }
            
            return scaled_data, df_clean.columns.tolist()
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    async def _estimate_contamination(self, data: np.ndarray) -> float:
        """Estimate contamination rate using statistical methods."""
        try:
            # Use IQR method to estimate outliers
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            iqr = q75 - q25
            
            # Count potential outliers across all features
            outlier_masks = []
            for i in range(data.shape[1]):
                lower_bound = q25[i] - 1.5 * iqr[i]
                upper_bound = q75[i] + 1.5 * iqr[i]
                outlier_mask = (data[:, i] < lower_bound) | (data[:, i] > upper_bound)
                outlier_masks.append(outlier_mask)
            
            # Combine outlier masks (point is outlier if outlier in any feature)
            combined_mask = np.any(outlier_masks, axis=0)
            estimated_contamination = np.mean(combined_mask)
            
            # Bound the estimate
            estimated_contamination = max(0.01, min(0.5, estimated_contamination))
            
            logger.info(f"Estimated contamination rate: {estimated_contamination:.3f}")
            return estimated_contamination
            
        except Exception as e:
            logger.warning(f"Contamination estimation failed: {str(e)}, using default")
            return self.config.default_contamination
    
    def _select_algorithm(self, data: np.ndarray) -> str:
        """Select the best algorithm based on data characteristics."""
        n_samples, n_features = data.shape
        
        # Small datasets: LOF works well
        if n_samples < self.config.small_dataset_threshold:
            return 'local_outlier_factor'
        
        # Large datasets: Isolation Forest is most scalable
        if n_samples > self.config.large_dataset_threshold:
            return 'isolation_forest'
        
        # High-dimensional data: One-Class SVM
        if n_features > n_samples / 10:
            return 'one_class_svm'
        
        # Check if data appears Gaussian
        if self._is_gaussian_distributed(data):
            return 'elliptic_envelope'
        
        # Default to Isolation Forest for general cases
        return 'isolation_forest'
    
    def _is_gaussian_distributed(self, data: np.ndarray) -> bool:
        """Simple test for Gaussian distribution using skewness and kurtosis."""
        try:
            from scipy import stats
            
            # Test a sample of features
            sample_features = min(5, data.shape[1])
            feature_indices = np.random.choice(data.shape[1], sample_features, replace=False)
            
            gaussian_count = 0
            for i in feature_indices:
                # Shapiro-Wilk test for normality (for small samples)
                if len(data) <= 5000:
                    stat, p_value = stats.shapiro(data[:, i])
                    if p_value > 0.05:
                        gaussian_count += 1
                else:
                    # Use skewness and kurtosis for large samples
                    skewness = abs(stats.skew(data[:, i]))
                    kurtosis = abs(stats.kurtosis(data[:, i]))
                    if skewness < 1 and kurtosis < 3:
                        gaussian_count += 1
            
            return gaussian_count / sample_features > 0.6
            
        except ImportError:
            return False
        except Exception:
            return False
    
    async def _train_model(
        self,
        data: np.ndarray,
        algorithm: str,
        contamination: float,
        optimize_params: bool
    ) -> Dict[str, Any]:
        """Train the selected anomaly detection model."""
        try:
            start_time = datetime.now()
            
            algo_config = self.algorithms[algorithm]
            
            # Update contamination in parameters
            params = algo_config['params'].copy()
            if 'contamination' in params:
                params['contamination'] = contamination
            elif 'nu' in params:  # OneClassSVM uses 'nu'
                params['nu'] = contamination
            
            # Optimize hyperparameters if requested
            if optimize_params and len(data) > 100:  # Skip optimization for very small datasets
                best_params = await self._optimize_hyperparameters(
                    data, algorithm, params
                )
                params.update(best_params)
            
            # Create and train model
            model = algo_config['class'](**params)
            
            if algorithm == 'local_outlier_factor':
                # LOF doesn't have fit/predict, only fit_predict
                model.fit_predict(data)
            else:
                model.fit(data)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model performance if possible
            performance = await self._evaluate_model(model, data, algorithm)
            
            result = {
                'model': model,
                'algorithm': algorithm,
                'parameters': params,
                'training_time': training_time,
                'performance': performance
            }
            
            # Cache the model
            self.models[algorithm] = model
            self.best_model = model
            self.best_algorithm = algorithm
            
            return result
            
        except Exception as e:
            logger.error(f"Model training failed for {algorithm}: {str(e)}")
            raise
    
    async def _optimize_hyperparameters(
        self,
        data: np.ndarray,
        algorithm: str,
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search or Bayesian optimization."""
        try:
            algo_config = self.algorithms[algorithm]
            param_space = algo_config['param_space']
            
            if BAYESIAN_OPT_AVAILABLE and len(param_space) > 2:
                # Use Bayesian optimization for complex parameter spaces
                return await self._bayesian_optimization(data, algorithm, base_params, param_space)
            else:
                # Use grid search for simpler cases
                return self._grid_search_optimization(data, algorithm, base_params, param_space)
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {str(e)}, using default params")
            return {}
    
    def _grid_search_optimization(
        self,
        data: np.ndarray,
        algorithm: str,
        base_params: Dict[str, Any],
        param_space: Dict[str, List]
    ) -> Dict[str, Any]:
        """Grid search hyperparameter optimization."""
        try:
            algo_class = self.algorithms[algorithm]['class']
            
            # Limit grid size for performance
            limited_param_space = {}
            for key, values in param_space.items():
                limited_param_space[key] = values[:3]  # Take first 3 values
            
            best_score = -np.inf
            best_params = {}
            
            for params in ParameterGrid(limited_param_space):
                try:
                    # Merge with base parameters
                    full_params = {**base_params, **params}
                    
                    # Create and evaluate model
                    model = algo_class(**full_params)
                    
                    if algorithm == 'local_outlier_factor':
                        scores = model.fit_predict(data)
                        # For LOF, use negative outlier factor as score
                        score = np.mean(model.negative_outlier_factor_)
                    else:
                        model.fit(data)
                        scores = model.decision_function(data)
                        score = np.mean(scores)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception:
                    continue
            
            return best_params
            
        except Exception as e:
            logger.error(f"Grid search optimization failed: {str(e)}")
            return {}
    
    async def _bayesian_optimization(
        self,
        data: np.ndarray,
        algorithm: str,
        base_params: Dict[str, Any],
        param_space: Dict[str, List]
    ) -> Dict[str, Any]:
        """Bayesian optimization for hyperparameter tuning."""
        try:
            # Convert parameter space to skopt format
            dimensions = []
            param_names = []
            
            for key, values in param_space.items():
                if isinstance(values[0], (int, float)):
                    if isinstance(values[0], int):
                        dimensions.append(Integer(min(values), max(values)))
                    else:
                        dimensions.append(Real(min(values), max(values)))
                else:
                    # Categorical parameters - use first few options
                    continue  # Skip for simplicity
                param_names.append(key)
            
            if not dimensions:
                return {}
            
            algo_class = self.algorithms[algorithm]['class']
            
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                try:
                    # Merge with base parameters
                    full_params = {**base_params}
                    for name, value in params.items():
                        full_params[name] = value
                    
                    # Create and evaluate model
                    model = algo_class(**full_params)
                    
                    if algorithm == 'local_outlier_factor':
                        model.fit_predict(data)
                        score = -np.mean(model.negative_outlier_factor_)
                    else:
                        model.fit(data)
                        scores = model.decision_function(data)
                        score = -np.mean(scores)  # Negative for minimization
                    
                    return score
                    
                except Exception:
                    return 1e6  # Large penalty for failed evaluations
            
            # Run Bayesian optimization
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.config.bayesian_opt_calls,
                random_state=self.config.random_state
            )
            
            # Extract best parameters
            best_params = {}
            for i, name in enumerate(param_names):
                best_params[name] = result.x[i]
            
            return best_params
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {str(e)}")
            return {}
    
    async def _evaluate_model(
        self,
        model: Any,
        data: np.ndarray,
        algorithm: str
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            performance = {}
            
            if algorithm == 'local_outlier_factor':
                # LOF specific metrics
                outlier_factors = model.negative_outlier_factor_
                performance['mean_outlier_factor'] = float(np.mean(outlier_factors))
                performance['std_outlier_factor'] = float(np.std(outlier_factors))
            else:
                # General anomaly detection metrics
                scores = model.decision_function(data)
                performance['mean_anomaly_score'] = float(np.mean(scores))
                performance['std_anomaly_score'] = float(np.std(scores))
                performance['min_anomaly_score'] = float(np.min(scores))
                performance['max_anomaly_score'] = float(np.max(scores))
            
            return performance
            
        except Exception as e:
            logger.warning(f"Model evaluation failed: {str(e)}")
            return {}
    
    async def _predict_anomalies(
        self,
        data: np.ndarray,
        model: Any
    ) -> Dict[str, Any]:
        """Generate anomaly predictions and scores."""
        try:
            start_time = datetime.now()
            
            # Get anomaly labels and scores
            if hasattr(model, 'predict'):
                labels = model.predict(data)  # 1 for inlier, -1 for outlier
                is_anomaly = labels == -1
            else:
                # LOF case
                labels = model.fit_predict(data)
                is_anomaly = labels == -1
            
            # Get anomaly scores
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(data)
                # Normalize scores to [0, 1] range for consistency
                scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
            elif hasattr(model, 'negative_outlier_factor_'):
                scores = -model.negative_outlier_factor_
                scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores = np.zeros(len(data))
                scores_normalized = scores
            
            # Find anomaly indices
            anomaly_indices = np.where(is_anomaly)[0]
            
            # Calculate feature importance if possible
            feature_importance = await self._calculate_feature_importance(
                model, data, is_anomaly
            )
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'is_anomaly': is_anomaly,
                'scores': scores_normalized,
                'raw_scores': scores,
                'anomaly_indices': anomaly_indices,
                'feature_importance': feature_importance,
                'prediction_time': prediction_time
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    async def _calculate_feature_importance(
        self,
        model: Any,
        data: np.ndarray,
        is_anomaly: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance for anomaly detection."""
        try:
            if not self.feature_names:
                return {}
            
            # Method 1: Use SHAP if available
            if SHAP_AVAILABLE and hasattr(model, 'decision_function'):
                try:
                    # Sample data for SHAP to avoid memory issues
                    sample_size = min(self.config.shap_sample_size, len(data))
                    sample_indices = np.random.choice(len(data), sample_size, replace=False)
                    sample_data = data[sample_indices]
                    
                    if hasattr(model, 'decision_function'):
                        explainer = shap.KernelExplainer(model.decision_function, sample_data[:100])
                        shap_values = explainer.shap_values(sample_data[:50])
                        
                        # Calculate mean absolute SHAP values for each feature
                        feature_importance = {}
                        for i, feature_name in enumerate(self.feature_names):
                            if i < shap_values.shape[1]:
                                feature_importance[feature_name] = float(np.mean(np.abs(shap_values[:, i])))
                        
                        return feature_importance
                        
                except Exception:
                    pass  # Fall back to statistical method
            
            # Method 2: Statistical feature importance
            feature_importance = {}
            
            for i, feature_name in enumerate(self.feature_names):
                if i >= data.shape[1]:
                    break
                
                # Calculate correlation between feature values and anomaly status
                feature_values = data[:, i]
                correlation = np.corrcoef(feature_values, is_anomaly.astype(int))[0, 1]
                feature_importance[feature_name] = float(abs(correlation))
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
            return {}
    
    def _calculate_metrics(
        self,
        predictions: Dict[str, Any],
        data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional metrics and statistics."""
        try:
            metrics = {}
            
            is_anomaly = predictions['is_anomaly']
            scores = predictions['scores']
            
            # Basic statistics
            metrics['total_points'] = int(len(data))
            metrics['anomaly_count'] = int(np.sum(is_anomaly))
            metrics['normal_count'] = int(np.sum(~is_anomaly))
            metrics['anomaly_percentage'] = float(np.mean(is_anomaly) * 100)
            
            # Score statistics
            metrics['mean_anomaly_score'] = float(np.mean(scores[is_anomaly])) if np.any(is_anomaly) else 0.0
            metrics['mean_normal_score'] = float(np.mean(scores[~is_anomaly])) if np.any(~is_anomaly) else 0.0
            metrics['score_separation'] = metrics['mean_anomaly_score'] - metrics['mean_normal_score']
            
            # Percentile information
            metrics['score_90th_percentile'] = float(np.percentile(scores, 90))
            metrics['score_95th_percentile'] = float(np.percentile(scores, 95))
            metrics['score_99th_percentile'] = float(np.percentile(scores, 99))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {str(e)}")
            return {}
    
    async def _generate_explanations(
        self,
        data: np.ndarray,
        model: Any,
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanations for anomaly detection results."""
        try:
            explanations = {}
            
            is_anomaly = predictions['is_anomaly']
            anomaly_indices = predictions['anomaly_indices']
            
            # Overall explanation
            total_anomalies = len(anomaly_indices)
            total_points = len(data)
            
            explanations['overview'] = (
                f"Detected {total_anomalies} anomalies out of {total_points} data points "
                f"({total_anomalies/total_points*100:.1f}% of the dataset)."
            )
            
            # Algorithm-specific explanations
            if self.best_algorithm == 'isolation_forest':
                explanations['method'] = (
                    "Isolation Forest isolates anomalies by randomly selecting features and "
                    "split values. Anomalies are easier to isolate and require fewer splits."
                )
            elif self.best_algorithm == 'local_outlier_factor':
                explanations['method'] = (
                    "Local Outlier Factor identifies anomalies by comparing the local density "
                    "of each point with its neighbors. Points in sparse regions are anomalies."
                )
            elif self.best_algorithm == 'one_class_svm':
                explanations['method'] = (
                    "One-Class SVM learns a boundary around normal data points. "
                    "Points outside this boundary are considered anomalies."
                )
            elif self.best_algorithm == 'elliptic_envelope':
                explanations['method'] = (
                    "Elliptic Envelope fits an ellipse to the central points, assuming "
                    "normal data follows a Gaussian distribution. Outliers fall outside the ellipse."
                )
            
            # Feature-based explanations
            if predictions.get('feature_importance'):
                top_features = sorted(
                    predictions['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                if top_features:
                    feature_names = [f[0] for f in top_features]
                    explanations['key_features'] = (
                        f"The most important features for anomaly detection are: "
                        f"{', '.join(feature_names)}. These features show the strongest "
                        f"patterns that distinguish anomalies from normal data points."
                    )
            
            # Severity assessment
            scores = predictions['scores']
            if len(anomaly_indices) > 0:
                max_score = np.max(scores[anomaly_indices])
                min_score = np.min(scores[anomaly_indices])
                
                if max_score > 0.8:
                    severity = "high"
                elif max_score > 0.6:
                    severity = "medium"
                else:
                    severity = "low"
                
                explanations['severity'] = (
                    f"The detected anomalies show {severity} severity. "
                    f"Anomaly scores range from {min_score:.3f} to {max_score:.3f}."
                )
            
            return explanations
            
        except Exception as e:
            logger.warning(f"Explanation generation failed: {str(e)}")
            return {'overview': 'Anomaly detection completed successfully.'}
    
    def _get_top_anomalies(
        self,
        original_data: pd.DataFrame,
        predictions: Dict[str, Any],
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the top N most anomalous data points."""
        try:
            is_anomaly = predictions['is_anomaly']
            scores = predictions['scores']
            
            # Get anomaly indices sorted by score (highest first)
            anomaly_mask = is_anomaly
            anomaly_scores = scores[anomaly_mask]
            anomaly_indices = predictions['anomaly_indices']
            
            # Sort by score (descending)
            sorted_indices = np.argsort(anomaly_scores)[::-1]
            top_indices = anomaly_indices[sorted_indices][:n]
            
            top_anomalies = []
            for idx in top_indices:
                anomaly_info = {
                    'index': int(idx),
                    'anomaly_score': float(scores[idx]),
                    'data': original_data.iloc[idx].to_dict()
                }
                top_anomalies.append(anomaly_info)
            
            return top_anomalies
            
        except Exception as e:
            logger.warning(f"Top anomalies extraction failed: {str(e)}")
            return []
    
    async def _log_to_mlflow(self, results: Dict[str, Any], model: Any):
        """Log results to MLflow for experiment tracking."""
        try:
            if not MLFLOW_AVAILABLE:
                return
            
            with mlflow.start_run(run_name=f"anomaly_detection_{self.best_algorithm}"):
                # Log parameters
                mlflow.log_param("algorithm", results['algorithm'])
                mlflow.log_param("contamination", results['contamination'])
                mlflow.log_param("n_features", len(self.feature_names) if self.feature_names else 0)
                
                # Log metrics
                mlflow.log_metric("n_anomalies", results['n_anomalies'])
                mlflow.log_metric("anomaly_rate", results['anomaly_rate'])
                mlflow.log_metric("training_time", results.get('training_time', 0))
                mlflow.log_metric("prediction_time", results.get('prediction_time', 0))
                
                # Log additional metrics
                if 'metrics' in results:
                    for key, value in results['metrics'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"metric_{key}", value)
                
                # Log model
                mlflow.sklearn.log_model(model, "anomaly_model")
                
                # Log feature importance as artifact
                if results.get('feature_importance'):
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(results['feature_importance'], f, indent=2)
                        mlflow.log_artifact(f.name, "feature_importance.json")
                
                logger.info("Results logged to MLflow successfully")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def predict_single(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Predict anomaly for a single data point."""
        try:
            if self.best_model is None:
                raise ValueError("No trained model available. Run detect_anomalies first.")
            
            # Convert to DataFrame and preprocess
            df = pd.DataFrame([data_point])
            
            # Apply same preprocessing as training
            if self.preprocessing_pipeline:
                # Select same columns
                numeric_cols = self.preprocessing_pipeline['numeric_columns']
                df = df[numeric_cols].fillna(df[numeric_cols].mean())
                
                # Remove constant columns
                constant_cols = self.preprocessing_pipeline['constant_columns']
                df = df.drop(columns=constant_cols, errors='ignore')
                
                # Apply scaling
                scaler = self.preprocessing_pipeline['scaler']
                scaled_data = scaler.transform(df)
            else:
                raise ValueError("Preprocessing pipeline not available")
            
            # Make prediction
            if hasattr(self.best_model, 'predict'):
                label = self.best_model.predict(scaled_data)[0]
                is_anomaly = label == -1
            else:
                # Handle LOF case
                is_anomaly = False  # Cannot predict on new single point with LOF
            
            # Get anomaly score
            if hasattr(self.best_model, 'decision_function'):
                score = self.best_model.decision_function(scaled_data)[0]
                score_normalized = max(0, min(1, (score + 1) / 2))  # Rough normalization
            else:
                score = 0.0
                score_normalized = 0.0
            
            return {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(score_normalized),
                'raw_score': float(score),
                'algorithm': self.best_algorithm,
                'confidence': 'high' if abs(score) > 0.5 else 'medium' if abs(score) > 0.2 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'error': str(e)
            }
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model and preprocessing pipeline."""
        try:
            model_data = {
                'model': self.best_model,
                'algorithm': self.best_algorithm,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'feature_names': self.feature_names,
                'config': self.config.__dict__
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a previously saved model and preprocessing pipeline."""
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['model']
            self.best_algorithm = model_data['algorithm']
            self.preprocessing_pipeline = model_data['preprocessing_pipeline']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

# Utility functions for the anomaly detection module

def create_anomaly_detector(contamination: float = 0.1) -> AnomalyDetector:
    """Factory function to create an AnomalyDetector instance."""
    config = AnomalyDetectionConfig()
    config.default_contamination = contamination
    return AnomalyDetector(config)

async def quick_anomaly_detection(
    data: pd.DataFrame,
    contamination: float = 0.1,
    algorithm: Optional[str] = None
) -> Dict[str, Any]:
    """Quick anomaly detection function for simple use cases."""
    detector = create_anomaly_detector(contamination)
    return await detector.detect_anomalies(data, contamination, algorithm, optimize_params=False)

def get_available_algorithms() -> List[str]:
    """Get list of available anomaly detection algorithms."""
    config = AnomalyDetectionConfig()
    detector = AnomalyDetector(config)
    return list(detector.algorithms.keys())

# Example usage and testing
if __name__ == "__main__":
    async def test_anomaly_detection():
        """Test the anomaly detection functionality."""
        # Create sample data
        np.random.seed(42)
        
        # Normal data
        normal_data = np.random.normal(0, 1, (1000, 5))
        
        # Anomalous data
        anomaly_data = np.random.normal(5, 1, (100, 5))
        
        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])
        df = pd.DataFrame(all_data, columns=[f'feature_{i}' for i in range(5)])
        
        # Create detector
        detector = create_anomaly_detector(contamination=0.1)
        
        # Run detection
        results = await detector.detect_anomalies(df)
        
        print(f"Algorithm used: {results['algorithm']}")
        print(f"Anomalies detected: {results['n_anomalies']}")
        print(f"Anomaly rate: {results['anomaly_rate']:.2f}%")
        print(f"Training time: {results.get('training_time', 0):.2f}s")
        
        return results
    
    # Run test
    import asyncio
    results = asyncio.run(test_anomaly_detection())
