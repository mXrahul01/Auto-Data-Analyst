"""
Clustering Models Module for Auto-Analyst Platform

This module implements comprehensive clustering algorithms including:
- K-Means (standard and variants)
- HDBSCAN for density-based clustering with noise handling
- Gaussian Mixture Models (GMM) for probabilistic clustering
- Hierarchical clustering with dendrogram support
- DBSCAN for density-based clustering
- Spectral clustering for graph-based clustering
- BIRCH for large-scale datasets
- Mean Shift for bandwidth-based clustering

Features:
- Automatic optimal cluster number detection
- Intelligent algorithm selection based on data characteristics
- Comprehensive cluster evaluation metrics
- Cluster profiling and characterization
- Visualization-ready outputs
- Real-time clustering for new data points
- MLflow integration for experiment tracking
- Model persistence and caching
- Performance optimization for large datasets
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
import math

# Core ML libraries
from sklearn.cluster import (
    KMeans, DBSCAN, SpectralClustering, 
    Birch, MeanShift, estimate_bandwidth,
    AgglomerativeClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, TSNE
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid

# Advanced clustering
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Hyperparameter optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

# Visualization support
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class ClusteringConfig:
    """Configuration class for clustering parameters."""
    
    def __init__(self):
        # Dataset size thresholds for algorithm selection
        self.small_dataset_threshold = 1000
        self.large_dataset_threshold = 50000
        self.very_large_dataset_threshold = 200000
        
        # Cluster number detection
        self.min_clusters = 2
        self.max_clusters = 20
        self.auto_max_clusters_ratio = 0.1  # Max clusters = n_samples * ratio
        
        # Performance settings
        self.max_features = 50  # PCA threshold
        self.n_jobs = -1
        self.random_state = 42
        
        # Quality thresholds
        self.min_silhouette_score = 0.3
        self.min_cluster_size_ratio = 0.01  # Minimum 1% of data per cluster
        
        # Optimization settings
        self.bayesian_opt_calls = 15
        self.grid_search_cv = 3
        
        # Visualization settings
        self.enable_tsne = True
        self.tsne_perplexity = 30
        self.tsne_n_components = 2
        
        # Model persistence
        self.model_cache_size = 5

class ClusterOptimizer:
    """Utility class for finding optimal number of clusters."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
    
    async def find_optimal_clusters(
        self,
        data: np.ndarray,
        algorithm: str = 'kmeans',
        method: str = 'elbow'
    ) -> int:
        """Find optimal number of clusters using various methods."""
        try:
            max_k = min(
                self.config.max_clusters,
                int(len(data) * self.config.auto_max_clusters_ratio),
                len(data) // 2
            )
            max_k = max(self.config.min_clusters, max_k)
            
            if method == 'elbow':
                return await self._elbow_method(data, algorithm, max_k)
            elif method == 'silhouette':
                return await self._silhouette_method(data, algorithm, max_k)
            elif method == 'gap_statistic':
                return await self._gap_statistic_method(data, algorithm, max_k)
            else:
                # Combined approach
                return await self._combined_method(data, algorithm, max_k)
                
        except Exception as e:
            logger.warning(f"Optimal cluster detection failed: {str(e)}")
            return self.config.min_clusters + 1
    
    async def _elbow_method(self, data: np.ndarray, algorithm: str, max_k: int) -> int:
        """Find optimal clusters using elbow method."""
        try:
            if algorithm != 'kmeans':
                return self.config.min_clusters + 1
            
            inertias = []
            k_range = range(self.config.min_clusters, max_k + 1)
            
            for k in k_range:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.config.random_state,
                    n_init=10,
                    max_iter=300
                )
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point using second derivative
            if len(inertias) < 3:
                return self.config.min_clusters + 1
            
            # Calculate second derivatives
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_derivative = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_derivative)
            
            # Find maximum second derivative (elbow point)
            elbow_index = np.argmax(second_derivatives) + 1
            optimal_k = list(k_range)[elbow_index]
            
            return optimal_k
            
        except Exception as e:
            logger.warning(f"Elbow method failed: {str(e)}")
            return self.config.min_clusters + 1
    
    async def _silhouette_method(self, data: np.ndarray, algorithm: str, max_k: int) -> int:
        """Find optimal clusters using silhouette analysis."""
        try:
            best_score = -1
            best_k = self.config.min_clusters + 1
            
            k_range = range(self.config.min_clusters, max_k + 1)
            
            for k in k_range:
                if algorithm == 'kmeans':
                    clusterer = KMeans(
                        n_clusters=k,
                        random_state=self.config.random_state,
                        n_init=10
                    )
                elif algorithm == 'gmm':
                    clusterer = GaussianMixture(
                        n_components=k,
                        random_state=self.config.random_state
                    )
                else:
                    continue
                
                cluster_labels = clusterer.fit_predict(data)
                
                # Skip if all points in one cluster
                if len(np.unique(cluster_labels)) <= 1:
                    continue
                
                score = silhouette_score(data, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            return best_k if best_score >= self.config.min_silhouette_score else self.config.min_clusters + 1
            
        except Exception as e:
            logger.warning(f"Silhouette method failed: {str(e)}")
            return self.config.min_clusters + 1
    
    async def _gap_statistic_method(self, data: np.ndarray, algorithm: str, max_k: int) -> int:
        """Find optimal clusters using gap statistic."""
        try:
            if algorithm != 'kmeans':
                return self.config.min_clusters + 1
            
            def compute_inertia(data, n_clusters):
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
                kmeans.fit(data)
                return kmeans.inertia_
            
            def compute_gap(data, n_clusters, n_refs=10):
                # Compute inertia for actual data
                actual_inertia = compute_inertia(data, n_clusters)
                
                # Generate reference datasets
                reference_inertias = []
                for _ in range(n_refs):
                    # Create random reference data with same bounds
                    mins = data.min(axis=0)
                    maxs = data.max(axis=0)
                    reference = np.random.uniform(mins, maxs, data.shape)
                    reference_inertias.append(compute_inertia(reference, n_clusters))
                
                # Compute gap
                gap = np.log(np.mean(reference_inertias)) - np.log(actual_inertia)
                return gap
            
            k_range = range(self.config.min_clusters, max_k + 1)
            gaps = []
            
            for k in k_range:
                gap = compute_gap(data, k)
                gaps.append(gap)
            
            # Find first local maximum in gap statistic
            for i in range(1, len(gaps)):
                if gaps[i] < gaps[i-1]:
                    return list(k_range)[i-1]
            
            # If no clear maximum, return cluster number with highest gap
            return list(k_range)[np.argmax(gaps)]
            
        except Exception as e:
            logger.warning(f"Gap statistic method failed: {str(e)}")
            return self.config.min_clusters + 1
    
    async def _combined_method(self, data: np.ndarray, algorithm: str, max_k: int) -> int:
        """Combine multiple methods for robust cluster number selection."""
        try:
            methods = ['elbow', 'silhouette']
            results = []
            
            for method in methods:
                try:
                    if method == 'elbow':
                        result = await self._elbow_method(data, algorithm, max_k)
                    elif method == 'silhouette':
                        result = await self._silhouette_method(data, algorithm, max_k)
                    
                    results.append(result)
                except Exception:
                    continue
            
            if not results:
                return self.config.min_clusters + 1
            
            # Return most common result, or median if no consensus
            from collections import Counter
            counts = Counter(results)
            most_common = counts.most_common(1)[0]
            
            if most_common[1] > 1:  # Consensus
                return most_common[0]
            else:
                return int(np.median(results))
                
        except Exception as e:
            logger.warning(f"Combined method failed: {str(e)}")
            return self.config.min_clusters + 1

class ClusterAnalyzer:
    """
    Comprehensive clustering analysis system with multiple algorithms,
    automatic parameter optimization, and detailed cluster insights.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.best_model = None
        self.best_algorithm = None
        self.best_n_clusters = None
        self.preprocessing_pipeline = None
        self.cluster_profiles = {}
        self.training_stats = {}
        self.optimizer = ClusterOptimizer(self.config)
        
        # Initialize algorithm registry
        self._initialize_algorithms()
        
        logger.info("ClusterAnalyzer initialized")
    
    def _initialize_algorithms(self):
        """Initialize available clustering algorithms."""
        self.algorithms = {
            'kmeans': {
                'class': KMeans,
                'params': {
                    'random_state': self.config.random_state,
                    'n_init': 10,
                    'max_iter': 300
                },
                'param_space': {
                    'init': ['k-means++', 'random'],
                    'algorithm': ['auto', 'full', 'elkan']
                },
                'needs_n_clusters': True,
                'scalable': True,
                'best_for': 'general',
                'handles_noise': False
            },
            'gaussian_mixture': {
                'class': GaussianMixture,
                'params': {
                    'random_state': self.config.random_state,
                    'max_iter': 200
                },
                'param_space': {
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'init_params': ['kmeans', 'random']
                },
                'needs_n_clusters': True,
                'scalable': True,
                'best_for': 'probabilistic',
                'handles_noise': False
            },
            'hierarchical': {
                'class': AgglomerativeClustering,
                'params': {},
                'param_space': {
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'affinity': ['euclidean', 'manhattan', 'cosine']
                },
                'needs_n_clusters': True,
                'scalable': False,
                'best_for': 'small_datasets',
                'handles_noise': False
            },
            'dbscan': {
                'class': DBSCAN,
                'params': {},
                'param_space': {
                    'eps': [0.1, 0.3, 0.5, 0.8, 1.0],
                    'min_samples': [3, 5, 10, 15]
                },
                'needs_n_clusters': False,
                'scalable': True,
                'best_for': 'density_based',
                'handles_noise': True
            },
            'spectral': {
                'class': SpectralClustering,
                'params': {
                    'random_state': self.config.random_state,
                    'n_jobs': self.config.n_jobs
                },
                'param_space': {
                    'affinity': ['rbf', 'nearest_neighbors'],
                    'assign_labels': ['kmeans', 'discretize']
                },
                'needs_n_clusters': True,
                'scalable': False,
                'best_for': 'non_convex',
                'handles_noise': False
            },
            'birch': {
                'class': Birch,
                'params': {},
                'param_space': {
                    'threshold': [0.1, 0.3, 0.5, 0.7],
                    'branching_factor': [50, 100, 150]
                },
                'needs_n_clusters': True,
                'scalable': True,
                'best_for': 'large_datasets',
                'handles_noise': False
            },
            'mean_shift': {
                'class': MeanShift,
                'params': {},
                'param_space': {},
                'needs_n_clusters': False,
                'scalable': False,
                'best_for': 'bandwidth_based',
                'handles_noise': True
            }
        }
        
        # Add HDBSCAN if available
        if HDBSCAN_AVAILABLE:
            self.algorithms['hdbscan'] = {
                'class': hdbscan.HDBSCAN,
                'params': {},
                'param_space': {
                    'min_cluster_size': [5, 10, 15, 20],
                    'min_samples': [1, 3, 5, 10],
                    'cluster_selection_epsilon': [0.0, 0.1, 0.5]
                },
                'needs_n_clusters': False,
                'scalable': True,
                'best_for': 'density_hierarchical',
                'handles_noise': True
            }
    
    async def cluster_data(
        self,
        data: pd.DataFrame,
        algorithm: Optional[str] = None,
        n_clusters: Optional[int] = None,
        optimize_params: bool = True
    ) -> Dict[str, Any]:
        """
        Main method to perform clustering analysis.
        
        Args:
            data: Input DataFrame
            algorithm: Specific algorithm to use (auto-select if None)
            n_clusters: Number of clusters (auto-detect if None)
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Dictionary containing clustering results and analysis
        """
        try:
            logger.info(f"Starting clustering analysis on dataset with shape {data.shape}")
            
            # Validate and preprocess data
            processed_data, feature_names = await self._preprocess_data(data)
            self.feature_names = feature_names
            
            # Select algorithm if not specified
            if algorithm is None:
                algorithm = self._select_algorithm(processed_data)
            
            # Detect optimal number of clusters if needed
            if n_clusters is None and self.algorithms[algorithm]['needs_n_clusters']:
                n_clusters = await self.optimizer.find_optimal_clusters(
                    processed_data, algorithm
                )
            
            logger.info(f"Using algorithm: {algorithm}, n_clusters: {n_clusters}")
            
            # Train model
            model_results = await self._train_model(
                processed_data, algorithm, n_clusters, optimize_params
            )
            
            # Generate cluster assignments
            assignments = await self._assign_clusters(processed_data, model_results['model'])
            
            # Evaluate clustering quality
            evaluation = await self._evaluate_clustering(
                processed_data, assignments['labels'], algorithm
            )
            
            # Analyze and profile clusters
            profiles = await self._profile_clusters(
                data, processed_data, assignments['labels']
            )
            
            # Generate visualizations if possible
            visualizations = await self._prepare_visualizations(
                processed_data, assignments['labels']
            )
            
            # Generate insights and recommendations
            insights = await self._generate_insights(
                data, assignments['labels'], evaluation, profiles
            )
            
            # Compile results
            results = {
                'algorithm': algorithm,
                'n_clusters': n_clusters or len(np.unique(assignments['labels'][assignments['labels'] >= 0])),
                'cluster_labels': assignments['labels'].tolist(),
                'cluster_centers': assignments.get('centers', []),
                'cluster_sizes': self._calculate_cluster_sizes(assignments['labels']),
                'evaluation_metrics': evaluation,
                'cluster_profiles': profiles,
                'insights': insights,
                'visualizations': visualizations,
                'model_info': model_results,
                'preprocessing_info': {
                    'n_features_original': len(data.columns),
                    'n_features_used': len(feature_names),
                    'feature_names': feature_names
                }
            }
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(results, model_results['model'])
            
            logger.info(f"Clustering completed successfully with {results['n_clusters']} clusters")
            
            return results
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
            return {
                'error': str(e),
                'algorithm': algorithm or 'auto',
                'n_clusters': 0,
                'cluster_labels': [],
                'cluster_sizes': {}
            }
    
    async def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for clustering."""
        try:
            # Select numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for clustering")
            
            # Handle missing values
            df_clean = data[numeric_cols].copy()
            df_clean = df_clean.fillna(df_clean.mean())
            
            # Remove constant columns
            constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
            if constant_cols:
                df_clean = df_clean.drop(columns=constant_cols)
                logger.info(f"Removed constant columns: {constant_cols}")
            
            # Remove highly correlated features
            if len(df_clean.columns) > 2:
                correlation_matrix = df_clean.corr().abs()
                upper_triangle = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                high_corr_features = [
                    column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > 0.95)
                ]
                if high_corr_features:
                    df_clean = df_clean.drop(columns=high_corr_features)
                    logger.info(f"Removed highly correlated features: {high_corr_features}")
            
            # Dimensionality reduction if needed
            if len(df_clean.columns) > self.config.max_features:
                logger.info(f"Applying PCA to reduce from {len(df_clean.columns)} to {self.config.max_features} features")
                pca = PCA(n_components=self.config.max_features, random_state=self.config.random_state)
                df_clean = pd.DataFrame(
                    pca.fit_transform(df_clean),
                    columns=[f'PC{i+1}' for i in range(self.config.max_features)]
                )
                
                # Store PCA info
                self.pca_transformer = pca
                self.pca_explained_variance = pca.explained_variance_ratio_
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_clean)
            
            # Store preprocessing pipeline
            self.preprocessing_pipeline = {
                'numeric_columns': numeric_cols,
                'selected_features': df_clean.columns.tolist(),
                'scaler': scaler,
                'constant_columns': constant_cols,
                'high_correlation_removed': high_corr_features if 'high_corr_features' in locals() else []
            }
            
            return scaled_data, df_clean.columns.tolist()
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def _select_algorithm(self, data: np.ndarray) -> str:
        """Select best algorithm based on data characteristics."""
        n_samples, n_features = data.shape
        
        # Very large datasets: use scalable algorithms
        if n_samples > self.config.very_large_dataset_threshold:
            return 'birch'
        
        # Large datasets: prefer scalable algorithms
        if n_samples > self.config.large_dataset_threshold:
            return 'kmeans'
        
        # Small datasets: hierarchical clustering works well
        if n_samples < self.config.small_dataset_threshold:
            return 'hierarchical'
        
        # High-dimensional data: spectral clustering
        if n_features > n_samples / 5:
            return 'spectral'
        
        # Check for potential noise/outliers
        if self._has_potential_noise(data):
            if HDBSCAN_AVAILABLE:
                return 'hdbscan'
            else:
                return 'dbscan'
        
        # Default to k-means for general cases
        return 'kmeans'
    
    def _has_potential_noise(self, data: np.ndarray) -> bool:
        """Check if data potentially contains noise/outliers."""
        try:
            # Use isolation forest to detect potential outliers
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(random_state=self.config.random_state, contamination=0.1)
            outlier_labels = iso_forest.fit_predict(data)
            
            # If more than 5% outliers, consider using noise-robust clustering
            outlier_ratio = np.sum(outlier_labels == -1) / len(data)
            return outlier_ratio > 0.05
            
        except Exception:
            return False
    
    async def _train_model(
        self,
        data: np.ndarray,
        algorithm: str,
        n_clusters: Optional[int],
        optimize_params: bool
    ) -> Dict[str, Any]:
        """Train the clustering model."""
        try:
            start_time = datetime.now()
            
            algo_config = self.algorithms[algorithm]
            params = algo_config['params'].copy()
            
            # Add number of clusters if needed
            if algo_config['needs_n_clusters'] and n_clusters:
                if algorithm == 'gaussian_mixture':
                    params['n_components'] = n_clusters
                else:
                    params['n_clusters'] = n_clusters
            
            # Special handling for Mean Shift
            if algorithm == 'mean_shift':
                if 'bandwidth' not in params:
                    bandwidth = estimate_bandwidth(data, quantile=0.3, n_samples=min(500, len(data)))
                    params['bandwidth'] = bandwidth
            
            # Optimize hyperparameters if requested
            if optimize_params and algo_config['param_space']:
                best_params = await self._optimize_parameters(
                    data, algorithm, params, n_clusters
                )
                params.update(best_params)
            
            # Create and train model
            model = algo_config['class'](**params)
            
            if hasattr(model, 'fit'):
                model.fit(data)
            else:
                # Some algorithms only have fit_predict
                model.fit_predict(data)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model
            self.models[algorithm] = model
            self.best_model = model
            self.best_algorithm = algorithm
            self.best_n_clusters = n_clusters
            
            return {
                'model': model,
                'algorithm': algorithm,
                'parameters': params,
                'training_time': training_time,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.error(f"Model training failed for {algorithm}: {str(e)}")
            raise
    
    async def _optimize_parameters(
        self,
        data: np.ndarray,
        algorithm: str,
        base_params: Dict[str, Any],
        n_clusters: Optional[int]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        try:
            algo_config = self.algorithms[algorithm]
            param_space = algo_config['param_space']
            
            if not param_space:
                return {}
            
            # Special handling for algorithms that don't use silhouette score
            if algorithm in ['dbscan', 'hdbscan', 'mean_shift']:
                return await self._optimize_density_based(data, algorithm, base_params, param_space)
            
            best_score = -1
            best_params = {}
            
            # Limit parameter combinations for performance
            limited_space = {}
            for key, values in param_space.items():
                limited_space[key] = values[:3]  # Take first 3 values
            
            for params in ParameterGrid(limited_space):
                try:
                    # Merge parameters
                    full_params = {**base_params, **params}
                    
                    # Add n_clusters if needed
                    if algo_config['needs_n_clusters'] and n_clusters:
                        if algorithm == 'gaussian_mixture':
                            full_params['n_components'] = n_clusters
                        else:
                            full_params['n_clusters'] = n_clusters
                    
                    # Create and fit model
                    model = algo_config['class'](**full_params)
                    labels = model.fit_predict(data)
                    
                    # Skip if invalid clustering
                    if len(np.unique(labels)) <= 1:
                        continue
                    
                    # Calculate silhouette score
                    score = silhouette_score(data, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception:
                    continue
            
            return best_params
            
        except Exception as e:
            logger.warning(f"Parameter optimization failed: {str(e)}")
            return {}
    
    async def _optimize_density_based(
        self,
        data: np.ndarray,
        algorithm: str,
        base_params: Dict[str, Any],
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize parameters for density-based algorithms."""
        try:
            best_score = -1
            best_params = {}
            
            # For DBSCAN/HDBSCAN, optimize for cluster validity
            for params in ParameterGrid(param_space):
                try:
                    full_params = {**base_params, **params}
                    
                    if algorithm == 'dbscan':
                        model = DBSCAN(**full_params)
                    elif algorithm == 'hdbscan' and HDBSCAN_AVAILABLE:
                        model = hdbscan.HDBSCAN(**full_params)
                    else:
                        continue
                    
                    labels = model.fit_predict(data)
                    
                    # Check cluster validity
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    n_noise = np.sum(labels == -1)
                    
                    # Skip if too few clusters or too much noise
                    if n_clusters < 2 or n_noise > len(data) * 0.5:
                        continue
                    
                    # Calculate custom score (fewer noise points and good cluster count)
                    noise_ratio = n_noise / len(data)
                    cluster_score = min(1.0, n_clusters / 10)  # Prefer moderate number of clusters
                    score = cluster_score * (1 - noise_ratio)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception:
                    continue
            
            return best_params
            
        except Exception as e:
            logger.warning(f"Density-based optimization failed: {str(e)}")
            return {}
    
    async def _assign_clusters(self, data: np.ndarray, model: Any) -> Dict[str, Any]:
        """Assign cluster labels and extract cluster information."""
        try:
            # Get cluster labels
            if hasattr(model, 'predict'):
                labels = model.predict(data)
            else:
                labels = model.fit_predict(data)
            
            # Get cluster centers if available
            centers = []
            if hasattr(model, 'cluster_centers_'):
                centers = model.cluster_centers_.tolist()
            elif hasattr(model, 'means_'):  # Gaussian Mixture
                centers = model.means_.tolist()
            
            # Get cluster probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(data)
                except:
                    pass
            
            return {
                'labels': labels,
                'centers': centers,
                'probabilities': probabilities,
                'unique_labels': np.unique(labels).tolist()
            }
            
        except Exception as e:
            logger.error(f"Cluster assignment failed: {str(e)}")
            raise
    
    async def _evaluate_clustering(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        algorithm: str
    ) -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics."""
        try:
            evaluation = {}
            
            # Filter out noise points for evaluation
            mask = labels >= 0
            if np.sum(mask) < 2:
                return {'error': 'Insufficient valid clusters for evaluation'}
            
            clean_data = data[mask]
            clean_labels = labels[mask]
            
            # Skip evaluation if only one cluster
            unique_labels = np.unique(clean_labels)
            if len(unique_labels) <= 1:
                return {'error': 'Only one cluster found'}
            
            # Silhouette Score
            try:
                evaluation['silhouette_score'] = float(silhouette_score(clean_data, clean_labels))
            except Exception:
                evaluation['silhouette_score'] = 0.0
            
            # Calinski-Harabasz Index
            try:
                evaluation['calinski_harabasz_score'] = float(calinski_harabasz_score(clean_data, clean_labels))
            except Exception:
                evaluation['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin Index (lower is better)
            try:
                evaluation['davies_bouldin_score'] = float(davies_bouldin_score(clean_data, clean_labels))
            except Exception:
                evaluation['davies_bouldin_score'] = float('inf')
            
            # Additional metrics
            evaluation['n_clusters'] = len(unique_labels)
            evaluation['n_noise_points'] = int(np.sum(labels == -1))
            evaluation['noise_ratio'] = float(np.sum(labels == -1) / len(labels))
            
            # Cluster size statistics
            cluster_sizes = []
            for label in unique_labels:
                size = np.sum(clean_labels == label)
                cluster_sizes.append(size)
            
            evaluation['min_cluster_size'] = int(min(cluster_sizes))
            evaluation['max_cluster_size'] = int(max(cluster_sizes))
            evaluation['avg_cluster_size'] = float(np.mean(cluster_sizes))
            evaluation['cluster_size_std'] = float(np.std(cluster_sizes))
            
            # Cluster balance (coefficient of variation of cluster sizes)
            if evaluation['avg_cluster_size'] > 0:
                evaluation['cluster_balance'] = float(
                    evaluation['cluster_size_std'] / evaluation['avg_cluster_size']
                )
            else:
                evaluation['cluster_balance'] = 0.0
            
            return evaluation
            
        except Exception as e:
            logger.warning(f"Clustering evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_cluster_sizes(self, labels: np.ndarray) -> Dict[str, int]:
        """Calculate cluster sizes."""
        unique_labels = np.unique(labels)
        sizes = {}
        
        for label in unique_labels:
            if label == -1:
                sizes['noise'] = int(np.sum(labels == label))
            else:
                sizes[f'cluster_{label}'] = int(np.sum(labels == label))
        
        return sizes
    
    async def _profile_clusters(
        self,
        original_data: pd.DataFrame,
        processed_data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Create detailed profiles for each cluster."""
        try:
            profiles = {}
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    profile_key = 'noise'
                else:
                    profile_key = f'cluster_{label}'
                
                mask = labels == label
                cluster_data = original_data[mask]
                
                profile = {
                    'size': int(np.sum(mask)),
                    'percentage': float(np.sum(mask) / len(labels) * 100)
                }
                
                # Statistical summary for numeric columns
                if len(numeric_cols) > 0:
                    numeric_summary = {}
                    for col in numeric_cols:
                        if col in cluster_data.columns:
                            values = cluster_data[col].dropna()
                            if len(values) > 0:
                                numeric_summary[col] = {
                                    'mean': float(values.mean()),
                                    'std': float(values.std()),
                                    'min': float(values.min()),
                                    'max': float(values.max()),
                                    'median': float(values.median())
                                }
                    
                    profile['numeric_summary'] = numeric_summary
                
                # Categorical summary
                categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    categorical_summary = {}
                    for col in categorical_cols:
                        if col in cluster_data.columns:
                            value_counts = cluster_data[col].value_counts()
                            if len(value_counts) > 0:
                                categorical_summary[col] = {
                                    'most_common': str(value_counts.index[0]),
                                    'most_common_count': int(value_counts.iloc[0]),
                                    'unique_values': int(cluster_data[col].nunique()),
                                    'top_values': value_counts.head(3).to_dict()
                                }
                    
                    profile['categorical_summary'] = categorical_summary
                
                profiles[profile_key] = profile
            
            return profiles
            
        except Exception as e:
            logger.warning(f"Cluster profiling failed: {str(e)}")
            return {}
    
    async def _prepare_visualizations(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data for visualizations."""
        try:
            visualizations = {}
            
            # 2D projection for visualization
            if data.shape[1] > 2:
                # Use t-SNE for 2D visualization
                if self.config.enable_tsne and len(data) <= 5000:  # t-SNE is slow for large datasets
                    try:
                        perplexity = min(self.config.tsne_perplexity, len(data) - 1)
                        tsne = TSNE(
                            n_components=2,
                            perplexity=perplexity,
                            random_state=self.config.random_state,
                            n_iter=1000
                        )
                        data_2d = tsne.fit_transform(data)
                        
                        visualizations['tsne_2d'] = {
                            'x': data_2d[:, 0].tolist(),
                            'y': data_2d[:, 1].tolist(),
                            'labels': labels.tolist()
                        }
                    except Exception as e:
                        logger.warning(f"t-SNE visualization failed: {str(e)}")
                
                # Use PCA as fallback
                if 'tsne_2d' not in visualizations:
                    pca = PCA(n_components=2, random_state=self.config.random_state)
                    data_2d = pca.fit_transform(data)
                    
                    visualizations['pca_2d'] = {
                        'x': data_2d[:, 0].tolist(),
                        'y': data_2d[:, 1].tolist(),
                        'labels': labels.tolist(),
                        'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
                    }
            else:
                # Data is already 2D or 1D
                visualizations['original_2d'] = {
                    'x': data[:, 0].tolist(),
                    'y': data[:, 1].tolist() if data.shape[1] > 1 else [0] * len(data),
                    'labels': labels.tolist()
                }
            
            # Cluster size pie chart data
            unique_labels = np.unique(labels)
            cluster_sizes = []
            cluster_names = []
            
            for label in unique_labels:
                size = np.sum(labels == label)
                if label == -1:
                    cluster_names.append('Noise')
                else:
                    cluster_names.append(f'Cluster {label}')
                cluster_sizes.append(int(size))
            
            visualizations['cluster_sizes'] = {
                'labels': cluster_names,
                'sizes': cluster_sizes
            }
            
            return visualizations
            
        except Exception as e:
            logger.warning(f"Visualization preparation failed: {str(e)}")
            return {}
    
    async def _generate_insights(
        self,
        original_data: pd.DataFrame,
        labels: np.ndarray,
        evaluation: Dict[str, Any],
        profiles: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate natural language insights about the clustering results."""
        try:
            insights = {}
            
            # Overview
            n_clusters = len(np.unique(labels[labels >= 0]))
            n_noise = np.sum(labels == -1)
            total_points = len(labels)
            
            insights['overview'] = (
                f"Clustering analysis identified {n_clusters} distinct groups in the dataset. "
                f"Out of {total_points:,} data points, {n_noise} were classified as noise or outliers."
            )
            
            # Quality assessment
            silhouette = evaluation.get('silhouette_score', 0)
            if silhouette >= 0.7:
                quality = "excellent"
            elif silhouette >= 0.5:
                quality = "good"
            elif silhouette >= 0.3:
                quality = "fair"
            else:
                quality = "poor"
            
            insights['quality_assessment'] = (
                f"The clustering quality is {quality} with a silhouette score of {silhouette:.3f}. "
                f"This indicates {'well-separated' if silhouette > 0.5 else 'overlapping'} clusters."
            )
            
            # Cluster balance
            cluster_balance = evaluation.get('cluster_balance', 0)
            if cluster_balance < 0.5:
                balance_desc = "well-balanced"
            elif cluster_balance < 1.0:
                balance_desc = "moderately balanced"
            else:
                balance_desc = "imbalanced"
            
            insights['cluster_balance'] = (
                f"The clusters are {balance_desc} in size. "
                f"The largest cluster contains {evaluation.get('max_cluster_size', 0)} points "
                f"while the smallest contains {evaluation.get('min_cluster_size', 0)} points."
            )
            
            # Dominant patterns
            dominant_patterns = []
            
            # Find most distinctive features for each cluster
            if profiles and len(profiles) > 1:
                numeric_features = []
                for cluster_id, profile in profiles.items():
                    if cluster_id != 'noise' and 'numeric_summary' in profile:
                        for feature, stats in profile['numeric_summary'].items():
                            numeric_features.append((cluster_id, feature, stats['mean']))
                
                if numeric_features:
                    # Group by feature and find clusters with extreme values
                    feature_groups = {}
                    for cluster_id, feature, mean_val in numeric_features:
                        if feature not in feature_groups:
                            feature_groups[feature] = []
                        feature_groups[feature].append((cluster_id, mean_val))
                    
                    for feature, cluster_values in feature_groups.items():
                        if len(cluster_values) > 1:
                            cluster_values.sort(key=lambda x: x[1])
                            highest = cluster_values[-1]
                            lowest = cluster_values[0]
                            
                            if abs(highest[1] - lowest[1]) > 0.5:  # Significant difference
                                dominant_patterns.append(
                                    f"{highest[0]} shows high {feature} values ({highest[1]:.2f}) "
                                    f"while {lowest[0]} shows low values ({lowest[1]:.2f})"
                                )
            
            if dominant_patterns:
                insights['dominant_patterns'] = ". ".join(dominant_patterns[:3])
            else:
                insights['dominant_patterns'] = "No strong distinguishing patterns detected between clusters."
            
            # Recommendations
            recommendations = []
            
            if silhouette < 0.5:
                recommendations.append("Consider trying a different clustering algorithm or adjusting parameters")
            
            if n_noise > total_points * 0.2:
                recommendations.append("High number of noise points suggests reviewing data quality or using density-based clustering")
            
            if cluster_balance > 1.5:
                recommendations.append("Cluster sizes are imbalanced - consider if this reflects natural data structure")
            
            if n_clusters < 3:
                recommendations.append("Few clusters detected - verify if more granular segmentation is needed")
            
            if not recommendations:
                recommendations.append("Clustering results appear satisfactory for further analysis")
            
            insights['recommendations'] = recommendations
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {str(e)}")
            return {
                'overview': 'Clustering analysis completed successfully.',
                'quality_assessment': 'Quality metrics calculated.',
                'recommendations': ['Review clustering results for business insights.']
            }
    
    async def _log_to_mlflow(self, results: Dict[str, Any], model: Any):
        """Log results to MLflow for experiment tracking."""
        try:
            if not MLFLOW_AVAILABLE:
                return
            
            with mlflow.start_run(run_name=f"clustering_{results['algorithm']}"):
                # Log parameters
                mlflow.log_param("algorithm", results['algorithm'])
                mlflow.log_param("n_clusters", results['n_clusters'])
                mlflow.log_param("n_features", results['preprocessing_info']['n_features_used'])
                
                # Log metrics
                eval_metrics = results['evaluation_metrics']
                for metric_name, metric_value in eval_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value) and math.isfinite(metric_value):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(model, "clustering_model")
                
                logger.info("Results logged to MLflow successfully")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def predict_cluster(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Predict cluster for a single data point."""
        try:
            if self.best_model is None:
                raise ValueError("No trained model available. Run cluster_data first.")
            
            # Convert to DataFrame and preprocess
            df = pd.DataFrame([data_point])
            
            # Apply same preprocessing as training
            if self.preprocessing_pipeline:
                numeric_cols = self.preprocessing_pipeline['numeric_columns']
                df = df[numeric_cols].fillna(df[numeric_cols].mean())
                
                # Remove constant columns
                constant_cols = self.preprocessing_pipeline['constant_columns']
                df = df.drop(columns=constant_cols, errors='ignore')
                
                # Apply scaling
                scaler = self.preprocessing_pipeline['scaler']
                scaled_data = scaler.transform(df)
                
                # Apply PCA if used
                if hasattr(self, 'pca_transformer'):
                    scaled_data = self.pca_transformer.transform(scaled_data)
            else:
                raise ValueError("Preprocessing pipeline not available")
            
            # Make prediction
            if hasattr(self.best_model, 'predict'):
                cluster_id = self.best_model.predict(scaled_data)[0]
            else:
                cluster_id = -1  # Cannot predict with some algorithms
            
            # Get cluster probability if available
            probability = None
            if hasattr(self.best_model, 'predict_proba'):
                try:
                    probabilities = self.best_model.predict_proba(scaled_data)[0]
                    probability = float(max(probabilities))
                except:
                    pass
            
            result = {
                'cluster_id': int(cluster_id),
                'algorithm': self.best_algorithm,
                'confidence': probability,
                'is_noise': bool(cluster_id == -1)
            }
            
            # Add cluster profile if available
            if cluster_id >= 0:
                profile_key = f'cluster_{cluster_id}'
                if profile_key in self.cluster_profiles:
                    result['cluster_profile'] = self.cluster_profiles[profile_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Cluster prediction failed: {str(e)}")
            return {
                'cluster_id': -1,
                'error': str(e),
                'is_noise': True
            }
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model and preprocessing pipeline."""
        try:
            model_data = {
                'model': self.best_model,
                'algorithm': self.best_algorithm,
                'n_clusters': self.best_n_clusters,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'feature_names': self.feature_names,
                'cluster_profiles': self.cluster_profiles,
                'config': self.config.__dict__
            }
            
            if hasattr(self, 'pca_transformer'):
                model_data['pca_transformer'] = self.pca_transformer
                model_data['pca_explained_variance'] = self.pca_explained_variance
            
            joblib.dump(model_data, filepath)
            logger.info(f"Clustering model saved to {filepath}")
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
            self.best_n_clusters = model_data['n_clusters']
            self.preprocessing_pipeline = model_data['preprocessing_pipeline']
            self.feature_names = model_data['feature_names']
            self.cluster_profiles = model_data['cluster_profiles']
            
            if 'pca_transformer' in model_data:
                self.pca_transformer = model_data['pca_transformer']
                self.pca_explained_variance = model_data['pca_explained_variance']
            
            logger.info(f"Clustering model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

# Utility functions for the clustering module

def create_cluster_analyzer(min_clusters: int = 2, max_clusters: int = 20) -> ClusterAnalyzer:
    """Factory function to create a ClusterAnalyzer instance."""
    config = ClusteringConfig()
    config.min_clusters = min_clusters
    config.max_clusters = max_clusters
    return ClusterAnalyzer(config)

async def quick_clustering(
    data: pd.DataFrame,
    algorithm: Optional[str] = None,
    n_clusters: Optional[int] = None
) -> Dict[str, Any]:
    """Quick clustering function for simple use cases."""
    analyzer = create_cluster_analyzer()
    return await analyzer.cluster_data(data, algorithm, n_clusters, optimize_params=False)

def get_available_algorithms() -> List[str]:
    """Get list of available clustering algorithms."""
    config = ClusteringConfig()
    analyzer = ClusterAnalyzer(config)
    return list(analyzer.algorithms.keys())

def get_recommended_algorithm(n_samples: int, n_features: int, has_noise: bool = False) -> str:
    """Get recommended algorithm based on data characteristics."""
    config = ClusteringConfig()
    analyzer = ClusterAnalyzer(config)
    
    # Create dummy data with specified characteristics
    dummy_data = np.random.randn(min(n_samples, 100), min(n_features, 10))
    
    if has_noise:
        # Add some outliers
        noise_indices = np.random.choice(len(dummy_data), size=int(len(dummy_data) * 0.1), replace=False)
        dummy_data[noise_indices] += np.random.normal(5, 2, (len(noise_indices), dummy_data.shape[1]))
    
    return analyzer._select_algorithm(dummy_data)

# Example usage and testing
if __name__ == "__main__":
    async def test_clustering():
        """Test the clustering functionality."""
        # Create sample data with clear clusters
        np.random.seed(42)
        
        # Create 3 distinct clusters
        cluster1 = np.random.normal([2, 2], 0.5, (100, 2))
        cluster2 = np.random.normal([8, 8], 0.5, (100, 2))
        cluster3 = np.random.normal([2, 8], 0.5, (100, 2))
        
        # Add some noise
        noise = np.random.uniform([0, 0], [10, 10], (20, 2))
        
        # Combine all data
        all_data = np.vstack([cluster1, cluster2, cluster3, noise])
        df = pd.DataFrame(all_data, columns=['feature_1', 'feature_2'])
        
        # Create analyzer
        analyzer = create_cluster_analyzer()
        
        # Run clustering
        results = await analyzer.cluster_data(df)
        
        print(f"Algorithm used: {results['algorithm']}")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Silhouette score: {results['evaluation_metrics'].get('silhouette_score', 0):.3f}")
        print(f"Cluster sizes: {results['cluster_sizes']}")
        
        return results
    
    # Run test
    import asyncio
    results = asyncio.run(test_clustering())
