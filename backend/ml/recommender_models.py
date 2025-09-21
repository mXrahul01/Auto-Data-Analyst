"""
Recommender Models Module for Auto-Analyst Platform

This module implements comprehensive recommendation system capabilities including:
- Collaborative Filtering (User-based, Item-based, Memory-based)
- Matrix Factorization (SVD, NMF, ALS, Deep Matrix Factorization)
- Content-Based Filtering with feature engineering
- Hybrid Recommendation Systems with multiple fusion strategies
- Deep Learning Recommenders (Neural Collaborative Filtering, AutoRec)
- Sequential and temporal recommendation models
- Multi-criteria and context-aware recommendations
- Implicit feedback handling and negative sampling
- Cold start problem solutions
- Diversity, novelty, and serendipity optimization

Features:
- Multiple recommendation algorithms with automatic selection
- Scalable implementations for large-scale datasets
- Real-time recommendation serving with caching
- Comprehensive evaluation metrics (Precision, Recall, NDCG, MAP, etc.)
- A/B testing framework for recommendation strategies
- Explanation generation for recommended items
- Business impact assessment and ROI analysis
- Integration with MLflow for experiment tracking
- Advanced hyperparameter optimization
- Multi-objective optimization (accuracy, diversity, novelty)
- Cold start handling for new users and items
- Implicit feedback conversion and confidence weighting
- Temporal dynamics and user preference drift modeling
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
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

# Core ML and matrix operations
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors

# Advanced matrix factorization
try:
    from scipy import sparse
    from scipy.sparse import csr_matrix, csc_matrix
    from scipy.sparse.linalg import svds
    from scipy.linalg import solve
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Recommendation-specific libraries
try:
    import surprise
    from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF
    from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
    from surprise import BaselineOnly, CoClustering
    from surprise.model_selection import cross_validate, GridSearchCV
    from surprise.accuracy import rmse, mae
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

try:
    import implicit
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    from implicit.lmf import LogisticMatrixFactorization
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False

# Deep learning for recommendations
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Advanced analytics
try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class RecommenderType(Enum):
    """Types of recommendation algorithms."""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    MATRIX_FACTORIZATION = "matrix_factorization"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"
    POPULARITY_BASED = "popularity_based"
    ASSOCIATION_RULES = "association_rules"
    CLUSTERING_BASED = "clustering_based"

class CFMethod(Enum):
    """Collaborative filtering methods."""
    USER_BASED = "user_based"
    ITEM_BASED = "item_based"
    MEMORY_BASED = "memory_based"

class MFMethod(Enum):
    """Matrix factorization methods."""
    SVD = "svd"
    NMF = "nmf"
    ALS = "alternating_least_squares"
    BPR = "bayesian_personalized_ranking"
    LMF = "logistic_matrix_factorization"
    DEEP_MF = "deep_matrix_factorization"

class FeedbackType(Enum):
    """Types of user feedback."""
    EXPLICIT = "explicit"  # Ratings, reviews
    IMPLICIT = "implicit"  # Clicks, views, purchases
    MIXED = "mixed"

class EvaluationMetric(Enum):
    """Evaluation metrics for recommendations."""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    NDCG = "ndcg"
    MAP = "mean_average_precision"
    MRR = "mean_reciprocal_rank"
    COVERAGE = "coverage"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    SERENDIPITY = "serendipity"
    RMSE = "rmse"
    MAE = "mae"

@dataclass
class RecommenderConfig:
    """Configuration for recommendation systems."""
    
    def __init__(self):
        # General settings
        self.recommender_type = RecommenderType.MATRIX_FACTORIZATION
        self.feedback_type = FeedbackType.EXPLICIT
        self.min_user_interactions = 5
        self.min_item_interactions = 5
        self.test_size = 0.2
        self.random_state = 42
        
        # Collaborative Filtering settings
        self.cf_method = CFMethod.ITEM_BASED
        self.cf_similarity_metric = 'cosine'  # 'cosine', 'pearson', 'euclidean'
        self.cf_k_neighbors = 50
        self.cf_min_support = 5
        
        # Matrix Factorization settings
        self.mf_method = MFMethod.SVD
        self.mf_factors = 50
        self.mf_learning_rate = 0.01
        self.mf_regularization = 0.1
        self.mf_epochs = 100
        self.mf_early_stopping = True
        
        # Deep Learning settings
        self.dl_embedding_dim = 64
        self.dl_hidden_units = [128, 64, 32]
        self.dl_dropout_rate = 0.2
        self.dl_batch_size = 256
        self.dl_epochs = 50
        
        # Implicit feedback settings
        self.implicit_confidence_alpha = 40
        self.implicit_regularization = 0.01
        self.negative_sampling_ratio = 5
        
        # Recommendation generation
        self.top_k_recommendations = 10
        self.recommendation_threshold = 0.0
        self.diversity_weight = 0.1
        self.novelty_weight = 0.05
        
        # Cold start handling
        self.cold_start_strategy = 'popularity'  # 'popularity', 'content', 'hybrid'
        self.new_user_min_ratings = 3
        self.content_features = []
        
        # Evaluation settings
        self.evaluation_metrics = [
            EvaluationMetric.PRECISION, 
            EvaluationMetric.RECALL,
            EvaluationMetric.NDCG
        ]
        self.evaluation_k_values = [5, 10, 20]
        self.cross_validation_folds = 5
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        self.use_gpu = False
        self.batch_processing = True
        
        # Business settings
        self.enable_explanations = True
        self.explanation_method = 'feature_based'
        self.business_rules = []
        self.filter_seen_items = True
        
        # Advanced settings
        self.temporal_dynamics = False
        self.context_aware = False
        self.multi_criteria = False
        self.online_learning = False
        
        # Caching and serving
        self.enable_caching = True
        self.cache_ttl_hours = 24
        self.precompute_recommendations = True
        
        # Quality settings
        self.min_recommendation_score = 0.1
        self.max_recommendations_per_user = 100
        self.recommendation_diversity_threshold = 0.7

@dataclass
class RecommendationItem:
    """Single recommendation item."""
    item_id: Any
    score: float
    rank: int
    explanation: Optional[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class UserRecommendations:
    """Recommendations for a single user."""
    user_id: Any
    recommendations: List[RecommendationItem]
    algorithm_used: str
    generation_time: datetime
    user_profile: Dict[str, Any]
    diversity_score: float
    novelty_score: float
    metadata: Dict[str, Any]

@dataclass
class RecommenderReport:
    """Comprehensive recommender system evaluation report."""
    report_id: str
    timestamp: datetime
    algorithm_type: RecommenderType
    dataset_info: Dict[str, Any]
    model_parameters: Dict[str, Any]
    evaluation_results: Dict[str, Dict[str, float]]  # metric -> k_value -> score
    business_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendation_samples: List[UserRecommendations]
    insights: List[str]
    recommendations_for_improvement: List[str]
    metadata: Dict[str, Any]

class RecommenderEvaluator:
    """Comprehensive evaluation metrics for recommendation systems."""
    
    @staticmethod
    def precision_at_k(y_true: List, y_pred: List, k: int = 10) -> float:
        """Calculate Precision@K."""
        try:
            if len(y_pred) > k:
                y_pred = y_pred[:k]
            
            relevant_items = set(y_true)
            recommended_items = set(y_pred)
            
            if len(recommended_items) == 0:
                return 0.0
            
            return len(relevant_items & recommended_items) / len(recommended_items)
        except:
            return 0.0
    
    @staticmethod
    def recall_at_k(y_true: List, y_pred: List, k: int = 10) -> float:
        """Calculate Recall@K."""
        try:
            if len(y_pred) > k:
                y_pred = y_pred[:k]
            
            relevant_items = set(y_true)
            recommended_items = set(y_pred)
            
            if len(relevant_items) == 0:
                return 0.0
            
            return len(relevant_items & recommended_items) / len(relevant_items)
        except:
            return 0.0
    
    @staticmethod
    def ndcg_at_k(y_true: List, y_pred: List, k: int = 10) -> float:
        """Calculate NDCG@K."""
        try:
            if len(y_pred) > k:
                y_pred = y_pred[:k]
            
            # Convert to binary relevance
            relevant_items = set(y_true)
            relevance_scores = [1 if item in relevant_items else 0 for item in y_pred]
            
            # DCG
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
            
            # IDCG
            ideal_relevance = [1] * min(len(y_true), k)
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            
            return dcg / idcg if idcg > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def mean_average_precision(y_true_dict: Dict, y_pred_dict: Dict, k: int = 10) -> float:
        """Calculate Mean Average Precision."""
        try:
            aps = []
            for user_id in y_true_dict:
                if user_id in y_pred_dict:
                    y_true = y_true_dict[user_id]
                    y_pred = y_pred_dict[user_id][:k]
                    
                    if len(y_true) == 0:
                        continue
                    
                    relevant_items = set(y_true)
                    precisions = []
                    
                    for i, item in enumerate(y_pred):
                        if item in relevant_items:
                            precision = RecommenderEvaluator.precision_at_k(y_true, y_pred[:i+1], i+1)
                            precisions.append(precision)
                    
                    if precisions:
                        aps.append(sum(precisions) / len(y_true))
            
            return sum(aps) / len(aps) if aps else 0.0
        except:
            return 0.0
    
    @staticmethod
    def coverage(y_pred_dict: Dict, all_items: Set) -> float:
        """Calculate item coverage."""
        try:
            recommended_items = set()
            for recommendations in y_pred_dict.values():
                recommended_items.update(recommendations)
            
            return len(recommended_items) / len(all_items) if all_items else 0.0
        except:
            return 0.0
    
    @staticmethod
    def diversity_score(recommendations: List, item_features: Dict) -> float:
        """Calculate intra-list diversity."""
        try:
            if len(recommendations) < 2:
                return 0.0
            
            similarities = []
            for i, item1 in enumerate(recommendations):
                for item2 in recommendations[i+1:]:
                    if item1 in item_features and item2 in item_features:
                        # Calculate feature-based similarity
                        features1 = item_features[item1]
                        features2 = item_features[item2]
                        
                        # Simple Jaccard similarity for categorical features
                        if isinstance(features1, set) and isinstance(features2, set):
                            similarity = len(features1 & features2) / len(features1 | features2)
                        else:
                            similarity = 0.5  # Default similarity
                        
                        similarities.append(similarity)
            
            return 1 - (sum(similarities) / len(similarities)) if similarities else 0.0
        except:
            return 0.0

class CollaborativeFilteringRecommender(BaseEstimator, RegressorMixin):
    """Collaborative Filtering Recommender implementation."""
    
    def __init__(
        self,
        method: CFMethod = CFMethod.ITEM_BASED,
        similarity_metric: str = 'cosine',
        k_neighbors: int = 50,
        min_support: int = 5
    ):
        self.method = method
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.min_support = min_support
        
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def fit(self, interactions_df: pd.DataFrame):
        """
        Fit the collaborative filtering model.
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        try:
            # Encode user and item IDs
            interactions_df = interactions_df.copy()
            interactions_df['user_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
            interactions_df['item_encoded'] = self.item_encoder.fit_transform(interactions_df['item_id'])
            
            # Create user-item matrix
            self.user_item_matrix = interactions_df.pivot(
                index='user_encoded',
                columns='item_encoded',
                values='rating'
            ).fillna(0)
            
            # Calculate means
            self.global_mean = interactions_df['rating'].mean()
            self.user_means = interactions_df.groupby('user_encoded')['rating'].mean()
            self.item_means = interactions_df.groupby('item_encoded')['rating'].mean()
            
            # Calculate similarity matrix
            if self.method == CFMethod.USER_BASED:
                self.similarity_matrix = self._calculate_user_similarity()
            elif self.method == CFMethod.ITEM_BASED:
                self.similarity_matrix = self._calculate_item_similarity()
            
            return self
            
        except Exception as e:
            logger.error(f"CF model fitting failed: {str(e)}")
            raise
    
    def _calculate_user_similarity(self) -> np.ndarray:
        """Calculate user-user similarity matrix."""
        if self.similarity_metric == 'cosine':
            return cosine_similarity(self.user_item_matrix)
        elif self.similarity_metric == 'pearson':
            return np.corrcoef(self.user_item_matrix.values)
        else:
            # Euclidean distance (converted to similarity)
            distances = euclidean_distances(self.user_item_matrix)
            return 1 / (1 + distances)
    
    def _calculate_item_similarity(self) -> np.ndarray:
        """Calculate item-item similarity matrix."""
        if self.similarity_metric == 'cosine':
            return cosine_similarity(self.user_item_matrix.T)
        elif self.similarity_metric == 'pearson':
            return np.corrcoef(self.user_item_matrix.T.values)
        else:
            # Euclidean distance (converted to similarity)
            distances = euclidean_distances(self.user_item_matrix.T)
            return 1 / (1 + distances)
    
    def predict(self, user_item_pairs: List[Tuple]) -> np.ndarray:
        """Predict ratings for user-item pairs."""
        try:
            predictions = []
            
            for user_id, item_id in user_item_pairs:
                # Encode IDs
                try:
                    user_encoded = self.user_encoder.transform([user_id])[0]
                    item_encoded = self.item_encoder.transform([item_id])[0]
                except ValueError:
                    # Unknown user or item - return global mean
                    predictions.append(self.global_mean)
                    continue
                
                if self.method == CFMethod.USER_BASED:
                    prediction = self._predict_user_based(user_encoded, item_encoded)
                elif self.method == CFMethod.ITEM_BASED:
                    prediction = self._predict_item_based(user_encoded, item_encoded)
                else:
                    prediction = self.global_mean
                
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"CF prediction failed: {str(e)}")
            return np.array([self.global_mean] * len(user_item_pairs))
    
    def _predict_user_based(self, user_encoded: int, item_encoded: int) -> float:
        """Predict rating using user-based CF."""
        try:
            # Get similar users
            user_similarities = self.similarity_matrix[user_encoded]
            user_ratings = self.user_item_matrix.iloc[:, item_encoded]
            
            # Filter users who have rated this item
            rated_users = user_ratings[user_ratings > 0].index
            
            if len(rated_users) == 0:
                return self.item_means.get(item_encoded, self.global_mean)
            
            # Get top-k similar users
            similar_users = []
            for user in rated_users:
                if user != user_encoded:
                    similarity = user_similarities[user]
                    similar_users.append((user, similarity))
            
            similar_users.sort(key=lambda x: x[1], reverse=True)
            top_users = similar_users[:self.k_neighbors]
            
            if not top_users:
                return self.item_means.get(item_encoded, self.global_mean)
            
            # Calculate weighted prediction
            numerator = 0
            denominator = 0
            
            for user, similarity in top_users:
                if abs(similarity) > 1e-6:  # Avoid division by zero
                    user_rating = user_ratings[user]
                    user_mean = self.user_means.get(user, self.global_mean)
                    
                    numerator += similarity * (user_rating - user_mean)
                    denominator += abs(similarity)
            
            if denominator > 0:
                user_mean = self.user_means.get(user_encoded, self.global_mean)
                return user_mean + numerator / denominator
            else:
                return self.item_means.get(item_encoded, self.global_mean)
                
        except Exception as e:
            logger.warning(f"User-based prediction failed: {str(e)}")
            return self.global_mean
    
    def _predict_item_based(self, user_encoded: int, item_encoded: int) -> float:
        """Predict rating using item-based CF."""
        try:
            # Get similar items
            item_similarities = self.similarity_matrix[item_encoded]
            user_ratings = self.user_item_matrix.iloc[user_encoded, :]
            
            # Filter items that user has rated
            rated_items = user_ratings[user_ratings > 0].index
            
            if len(rated_items) == 0:
                return self.user_means.get(user_encoded, self.global_mean)
            
            # Get top-k similar items
            similar_items = []
            for item in rated_items:
                if item != item_encoded:
                    similarity = item_similarities[item]
                    similar_items.append((item, similarity))
            
            similar_items.sort(key=lambda x: x[1], reverse=True)
            top_items = similar_items[:self.k_neighbors]
            
            if not top_items:
                return self.user_means.get(user_encoded, self.global_mean)
            
            # Calculate weighted prediction
            numerator = 0
            denominator = 0
            
            for item, similarity in top_items:
                if abs(similarity) > 1e-6:  # Avoid division by zero
                    item_rating = user_ratings[item]
                    
                    numerator += similarity * item_rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                return numerator / denominator
            else:
                return self.user_means.get(user_encoded, self.global_mean)
                
        except Exception as e:
            logger.warning(f"Item-based prediction failed: {str(e)}")
            return self.global_mean
    
    def recommend(self, user_id: Any, n_recommendations: int = 10, filter_seen: bool = True) -> List[Tuple]:
        """Generate recommendations for a user."""
        try:
            # Encode user ID
            try:
                user_encoded = self.user_encoder.transform([user_id])[0]
            except ValueError:
                # Unknown user - return popular items
                return self._get_popular_items(n_recommendations)
            
            # Get all items
            all_items = list(range(len(self.item_encoder.classes_)))
            
            # Filter seen items if requested
            if filter_seen:
                user_ratings = self.user_item_matrix.iloc[user_encoded, :]
                seen_items = set(user_ratings[user_ratings > 0].index)
                candidate_items = [item for item in all_items if item not in seen_items]
            else:
                candidate_items = all_items
            
            # Predict ratings for candidate items
            user_item_pairs = [(user_id, self.item_encoder.inverse_transform([item])[0]) 
                             for item in candidate_items]
            predictions = self.predict(user_item_pairs)
            
            # Sort by predicted rating
            item_scores = list(zip([pair[1] for pair in user_item_pairs], predictions))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return item_scores[:n_recommendations]
            
        except Exception as e:
            logger.error(f"CF recommendation failed: {str(e)}")
            return self._get_popular_items(n_recommendations)
    
    def _get_popular_items(self, n_recommendations: int) -> List[Tuple]:
        """Get popular items as fallback."""
        try:
            item_popularity = self.user_item_matrix.sum(axis=0).sort_values(ascending=False)
            popular_items = item_popularity.head(n_recommendations).index
            
            return [(self.item_encoder.inverse_transform([item])[0], item_popularity[item]) 
                   for item in popular_items]
        except:
            return []

class MatrixFactorizationRecommender(BaseEstimator, RegressorMixin):
    """Matrix Factorization Recommender using various algorithms."""
    
    def __init__(
        self,
        method: MFMethod = MFMethod.SVD,
        n_factors: int = 50,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        n_epochs: int = 100,
        random_state: int = 42
    ):
        self.method = method
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
    
    def fit(self, interactions_df: pd.DataFrame):
        """Fit the matrix factorization model."""
        try:
            # Encode user and item IDs
            interactions_df = interactions_df.copy()
            interactions_df['user_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
            interactions_df['item_encoded'] = self.item_encoder.fit_transform(interactions_df['item_id'])
            
            self.global_mean = interactions_df['rating'].mean()
            
            # Choose algorithm
            if self.method == MFMethod.SVD and SURPRISE_AVAILABLE:
                self._fit_surprise_svd(interactions_df)
            elif self.method == MFMethod.NMF and SURPRISE_AVAILABLE:
                self._fit_surprise_nmf(interactions_df)
            elif self.method == MFMethod.ALS and IMPLICIT_AVAILABLE:
                self._fit_implicit_als(interactions_df)
            elif self.method == MFMethod.BPR and IMPLICIT_AVAILABLE:
                self._fit_implicit_bpr(interactions_df)
            else:
                # Fallback to custom SVD implementation
                self._fit_custom_svd(interactions_df)
            
            return self
            
        except Exception as e:
            logger.error(f"MF model fitting failed: {str(e)}")
            raise
    
    def _fit_surprise_svd(self, interactions_df: pd.DataFrame):
        """Fit using Surprise SVD algorithm."""
        try:
            # Prepare data for Surprise
            reader = Reader(rating_scale=(interactions_df['rating'].min(), 
                                        interactions_df['rating'].max()))
            
            data = Dataset.load_from_df(
                interactions_df[['user_id', 'item_id', 'rating']], 
                reader
            )
            
            # Train SVD model
            self.model = SVD(
                n_factors=self.n_factors,
                lr_all=self.learning_rate,
                reg_all=self.regularization,
                n_epochs=self.n_epochs,
                random_state=self.random_state
            )
            
            # Train on full dataset
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            
        except Exception as e:
            logger.error(f"Surprise SVD fitting failed: {str(e)}")
            raise
    
    def _fit_surprise_nmf(self, interactions_df: pd.DataFrame):
        """Fit using Surprise NMF algorithm."""
        try:
            # Prepare data for Surprise
            reader = Reader(rating_scale=(interactions_df['rating'].min(),
                                        interactions_df['rating'].max()))
            
            data = Dataset.load_from_df(
                interactions_df[['user_id', 'item_id', 'rating']],
                reader
            )
            
            # Train NMF model
            self.model = SurpriseNMF(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                random_state=self.random_state
            )
            
            # Train on full dataset
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            
        except Exception as e:
            logger.error(f"Surprise NMF fitting failed: {str(e)}")
            raise
    
    def _fit_implicit_als(self, interactions_df: pd.DataFrame):
        """Fit using Implicit ALS algorithm."""
        try:
            # Create sparse matrix
            n_users = len(self.user_encoder.classes_)
            n_items = len(self.item_encoder.classes_)
            
            # Convert ratings to confidence values
            confidence_values = interactions_df['rating'].values
            if interactions_df['rating'].max() <= 1:
                # Already in [0, 1] range
                confidence_values = confidence_values * 40  # Scale for implicit
            
            user_item_matrix = csr_matrix(
                (confidence_values, 
                 (interactions_df['user_encoded'].values, interactions_df['item_encoded'].values)),
                shape=(n_users, n_items)
            )
            
            # Train ALS model
            self.model = AlternatingLeastSquares(
                factors=self.n_factors,
                regularization=self.regularization,
                iterations=self.n_epochs,
                random_state=self.random_state
            )
            
            self.model.fit(user_item_matrix)
            
        except Exception as e:
            logger.error(f"Implicit ALS fitting failed: {str(e)}")
            raise
    
    def _fit_implicit_bpr(self, interactions_df: pd.DataFrame):
        """Fit using Implicit BPR algorithm."""
        try:
            # Create sparse matrix
            n_users = len(self.user_encoder.classes_)
            n_items = len(self.item_encoder.classes_)
            
            # Convert to implicit feedback (binary)
            implicit_values = np.ones(len(interactions_df))
            
            user_item_matrix = csr_matrix(
                (implicit_values,
                 (interactions_df['user_encoded'].values, interactions_df['item_encoded'].values)),
                shape=(n_users, n_items)
            )
            
            # Train BPR model
            self.model = BayesianPersonalizedRanking(
                factors=self.n_factors,
                learning_rate=self.learning_rate,
                regularization=self.regularization,
                iterations=self.n_epochs,
                random_state=self.random_state
            )
            
            self.model.fit(user_item_matrix)
            
        except Exception as e:
            logger.error(f"Implicit BPR fitting failed: {str(e)}")
            raise
    
    def _fit_custom_svd(self, interactions_df: pd.DataFrame):
        """Custom SVD implementation using scipy."""
        try:
            # Create user-item matrix
            user_item_matrix = interactions_df.pivot(
                index='user_encoded',
                columns='item_encoded', 
                values='rating'
            ).fillna(0)
            
            # Perform SVD
            U, sigma, Vt = svds(user_item_matrix.values, k=self.n_factors)
            
            # Store factors
            self.user_factors = U * np.sqrt(sigma)
            self.item_factors = Vt.T * np.sqrt(sigma)
            
            # Calculate biases
            self.user_biases = interactions_df.groupby('user_encoded')['rating'].mean() - self.global_mean
            self.item_biases = interactions_df.groupby('item_encoded')['rating'].mean() - self.global_mean
            
        except Exception as e:
            logger.error(f"Custom SVD fitting failed: {str(e)}")
            raise
    
    def predict(self, user_item_pairs: List[Tuple]) -> np.ndarray:
        """Predict ratings for user-item pairs."""
        try:
            predictions = []
            
            for user_id, item_id in user_item_pairs:
                # Handle unknown users/items
                try:
                    if hasattr(self.model, 'predict'):
                        # Surprise models
                        pred = self.model.predict(user_id, item_id)
                        predictions.append(pred.est)
                    else:
                        # Custom implementation
                        user_encoded = self.user_encoder.transform([user_id])[0]
                        item_encoded = self.item_encoder.transform([item_id])[0]
                        
                        if (user_encoded < len(self.user_factors) and 
                            item_encoded < len(self.item_factors)):
                            
                            pred = (self.global_mean + 
                                   self.user_biases.get(user_encoded, 0) +
                                   self.item_biases.get(item_encoded, 0) +
                                   np.dot(self.user_factors[user_encoded], 
                                         self.item_factors[item_encoded]))
                            predictions.append(pred)
                        else:
                            predictions.append(self.global_mean)
                
                except (ValueError, KeyError):
                    predictions.append(self.global_mean)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"MF prediction failed: {str(e)}")
            return np.array([self.global_mean] * len(user_item_pairs))
    
    def recommend(self, user_id: Any, n_recommendations: int = 10, filter_seen: bool = True) -> List[Tuple]:
        """Generate recommendations for a user."""
        try:
            # Handle unknown users
            try:
                user_encoded = self.user_encoder.transform([user_id])[0]
            except ValueError:
                return self._get_popular_items(n_recommendations)
            
            # Get all items
            all_items = self.item_encoder.classes_
            
            # Generate predictions for all items
            user_item_pairs = [(user_id, item) for item in all_items]
            predictions = self.predict(user_item_pairs)
            
            # Create item-score pairs
            item_scores = list(zip(all_items, predictions))
            
            # Filter seen items if requested
            if filter_seen and hasattr(self, 'seen_items'):
                seen_items = self.seen_items.get(user_id, set())
                item_scores = [(item, score) for item, score in item_scores 
                              if item not in seen_items]
            
            # Sort by score
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return item_scores[:n_recommendations]
            
        except Exception as e:
            logger.error(f"MF recommendation failed: {str(e)}")
            return self._get_popular_items(n_recommendations)
    
    def _get_popular_items(self, n_recommendations: int) -> List[Tuple]:
        """Get popular items as fallback."""
        try:
            # Simple popularity based on occurrence frequency
            popular_items = list(self.item_encoder.classes_[:n_recommendations])
            return [(item, self.global_mean) for item in popular_items]
        except:
            return []

class DeepLearningRecommender(BaseEstimator, RegressorMixin):
    """Deep Learning Recommender using neural networks."""
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_units: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        random_state: int = 42
    ):
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.model = None
        self.n_users = None
        self.n_items = None
    
    def fit(self, interactions_df: pd.DataFrame):
        """Fit the deep learning model."""
        try:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for deep learning recommender")
            
            # Encode user and item IDs
            interactions_df = interactions_df.copy()
            interactions_df['user_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
            interactions_df['item_encoded'] = self.item_encoder.fit_transform(interactions_df['item_id'])
            
            self.n_users = len(self.user_encoder.classes_)
            self.n_items = len(self.item_encoder.classes_)
            
            # Build model
            self.model = self._build_neural_cf_model()
            
            # Prepare training data
            user_input = interactions_df['user_encoded'].values
            item_input = interactions_df['item_encoded'].values
            ratings = interactions_df['rating'].values
            
            # Normalize ratings to [0, 1]
            rating_min, rating_max = ratings.min(), ratings.max()
            ratings_normalized = (ratings - rating_min) / (rating_max - rating_min)
            
            # Train model
            self.model.fit(
                [user_input, item_input],
                ratings_normalized,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
                ]
            )
            
            # Store normalization parameters
            self.rating_min = rating_min
            self.rating_max = rating_max
            
            return self
            
        except Exception as e:
            logger.error(f"Deep learning model fitting failed: {str(e)}")
            raise
    
    def _build_neural_cf_model(self):
        """Build Neural Collaborative Filtering model."""
        try:
            # User and item inputs
            user_input = tf.keras.Input(shape=(), name='user_id')
            item_input = tf.keras.Input(shape=(), name='item_id')
            
            # Embeddings
            user_embedding = tf.keras.layers.Embedding(
                self.n_users, self.embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
            )(user_input)
            item_embedding = tf.keras.layers.Embedding(
                self.n_items, self.embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
            )(item_input)
            
            # Flatten embeddings
            user_vec = tf.keras.layers.Flatten()(user_embedding)
            item_vec = tf.keras.layers.Flatten()(item_embedding)
            
            # Concatenate user and item vectors
            concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
            
            # Hidden layers
            x = concat
            for units in self.hidden_units:
                x = tf.keras.layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-6)
                )(x)
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            
            # Output layer
            output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            # Create model
            model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Neural CF model building failed: {str(e)}")
            raise
    
    def predict(self, user_item_pairs: List[Tuple]) -> np.ndarray:
        """Predict ratings for user-item pairs."""
        try:
            if self.model is None:
                raise ValueError("Model not fitted")
            
            predictions = []
            
            for user_id, item_id in user_item_pairs:
                try:
                    user_encoded = self.user_encoder.transform([user_id])[0]
                    item_encoded = self.item_encoder.transform([item_id])[0]
                    
                    pred_normalized = self.model.predict(
                        [np.array([user_encoded]), np.array([item_encoded])],
                        verbose=0
                    )[0][0]
                    
                    # Denormalize prediction
                    pred = pred_normalized * (self.rating_max - self.rating_min) + self.rating_min
                    predictions.append(pred)
                    
                except (ValueError, KeyError):
                    # Unknown user or item
                    global_mean = (self.rating_min + self.rating_max) / 2
                    predictions.append(global_mean)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Deep learning prediction failed: {str(e)}")
            global_mean = (self.rating_min + self.rating_max) / 2 if hasattr(self, 'rating_min') else 3.0
            return np.array([global_mean] * len(user_item_pairs))
    
    def recommend(self, user_id: Any, n_recommendations: int = 10, filter_seen: bool = True) -> List[Tuple]:
        """Generate recommendations for a user."""
        try:
            # Handle unknown users
            try:
                user_encoded = self.user_encoder.transform([user_id])[0]
            except ValueError:
                return self._get_popular_items(n_recommendations)
            
            # Get all items
            all_items = self.item_encoder.classes_
            
            # Generate predictions for all items
            user_item_pairs = [(user_id, item) for item in all_items]
            predictions = self.predict(user_item_pairs)
            
            # Create item-score pairs
            item_scores = list(zip(all_items, predictions))
            
            # Sort by score
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return item_scores[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Deep learning recommendation failed: {str(e)}")
            return self._get_popular_items(n_recommendations)
    
    def _get_popular_items(self, n_recommendations: int) -> List[Tuple]:
        """Get popular items as fallback."""
        try:
            # Simple popularity based on occurrence frequency
            popular_items = list(self.item_encoder.classes_[:n_recommendations])
            global_mean = (self.rating_min + self.rating_max) / 2 if hasattr(self, 'rating_min') else 3.0
            return [(item, global_mean) for item in popular_items]
        except:
            return []

class RecommenderSystem:
    """
    Comprehensive recommendation system with multiple algorithms,
    automatic selection, evaluation, and business intelligence.
    """
    
    def __init__(self, config: Optional[RecommenderConfig] = None):
        self.config = config or RecommenderConfig()
        self.models = {}
        self.best_model = None
        self.best_algorithm = None
        self.item_features = {}
        self.user_profiles = {}
        self.popular_items = []
        self.evaluation_results = {}
        self.recommendation_cache = {}
        
        logger.info("RecommenderSystem initialized")
    
    async def train_recommender(
        self,
        interactions_df: pd.DataFrame,
        item_features_df: Optional[pd.DataFrame] = None,
        user_features_df: Optional[pd.DataFrame] = None,
        algorithms: Optional[List[RecommenderType]] = None
    ) -> RecommenderReport:
        """
        Train recommendation system with multiple algorithms and evaluation.
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
            item_features_df: Optional item features DataFrame
            user_features_df: Optional user features DataFrame
            algorithms: List of algorithms to try (None for automatic selection)
            
        Returns:
            Comprehensive recommender evaluation report
        """
        try:
            logger.info(f"Training recommender system on {len(interactions_df)} interactions")
            start_time = datetime.now()
            
            # Data preprocessing and validation
            interactions_df = await self._preprocess_interactions(interactions_df)
            
            # Process features if provided
            if item_features_df is not None:
                self.item_features = await self._process_item_features(item_features_df)
            
            if user_features_df is not None:
                self.user_profiles = await self._process_user_features(user_features_df)
            
            # Determine algorithms to try
            if algorithms is None:
                algorithms = self._select_algorithms(interactions_df)
            
            logger.info(f"Training algorithms: {[alg.value for alg in algorithms]}")
            
            # Split data for evaluation
            train_df, test_df = await self._split_data(interactions_df)
            
            # Train models
            model_results = {}
            for algorithm in algorithms:
                try:
                    logger.info(f"Training {algorithm.value} model")
                    model, training_time = await self._train_single_model(
                        algorithm, train_df
                    )
                    
                    if model is not None:
                        self.models[algorithm.value] = model
                        model_results[algorithm.value] = {
                            'model': model,
                            'training_time': training_time
                        }
                        
                        logger.info(f"{algorithm.value} training completed in {training_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"{algorithm.value} training failed: {str(e)}")
                    continue
            
            if not model_results:
                raise ValueError("No models were successfully trained")
            
            # Evaluate models
            evaluation_results = await self._evaluate_models(
                model_results, train_df, test_df
            )
            
            # Select best model
            best_model_info = self._select_best_model(evaluation_results)
            self.best_model = best_model_info['model']
            self.best_algorithm = best_model_info['algorithm']
            
            # Calculate popular items for cold start
            self.popular_items = await self._calculate_popular_items(interactions_df)
            
            # Generate business metrics
            business_metrics = await self._calculate_business_metrics(
                interactions_df, evaluation_results
            )
            
            # Generate sample recommendations
            sample_recommendations = await self._generate_sample_recommendations(
                interactions_df, n_samples=5
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(evaluation_results, business_metrics)
            recommendations = self._generate_recommendations(evaluation_results, insights)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive report
            report = RecommenderReport(
                report_id=str(uuid.uuid4()),
                timestamp=start_time,
                algorithm_type=RecommenderType(self.best_algorithm),
                dataset_info={
                    'n_interactions': len(interactions_df),
                    'n_users': interactions_df['user_id'].nunique(),
                    'n_items': interactions_df['item_id'].nunique(),
                    'sparsity': 1 - len(interactions_df) / (
                        interactions_df['user_id'].nunique() * interactions_df['item_id'].nunique()
                    ),
                    'rating_range': (interactions_df['rating'].min(), interactions_df['rating'].max())
                },
                model_parameters=self._get_best_model_parameters(),
                evaluation_results=evaluation_results,
                business_metrics=business_metrics,
                performance_metrics={
                    'training_time': execution_time,
                    'models_trained': len(model_results),
                    'best_algorithm': self.best_algorithm
                },
                recommendation_samples=sample_recommendations,
                insights=insights,
                recommendations_for_improvement=recommendations,
                metadata={
                    'algorithms_tried': list(model_results.keys()),
                    'config': asdict(self.config)
                }
            )
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(report)
            
            logger.info(f"Recommender system training completed in {execution_time:.2f}s")
            logger.info(f"Best algorithm: {self.best_algorithm}")
            
            return report
            
        except Exception as e:
            logger.error(f"Recommender training failed: {str(e)}")
            # Return minimal report with error
            return RecommenderReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                algorithm_type=self.config.recommender_type,
                dataset_info={},
                model_parameters={},
                evaluation_results={},
                business_metrics={},
                performance_metrics={'error': str(e)},
                recommendation_samples=[],
                insights=[f"Training failed: {str(e)}"],
                recommendations_for_improvement=["Review data quality and system configuration"],
                metadata={'error': str(e)}
            )
    
    async def _preprocess_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess interaction data."""
        try:
            df = interactions_df.copy()
            
            # Required columns validation
            required_cols = ['user_id', 'item_id', 'rating']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['user_id', 'item_id'])
            
            # Filter users and items with minimum interactions
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()
            
            valid_users = user_counts[user_counts >= self.config.min_user_interactions].index
            valid_items = item_counts[item_counts >= self.config.min_item_interactions].index
            
            df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
            
            logger.info(f"After preprocessing: {len(df)} interactions, "
                       f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
            
            return df
            
        except Exception as e:
            logger.error(f"Interaction preprocessing failed: {str(e)}")
            return interactions_df
    
    async def _process_item_features(self, item_features_df: pd.DataFrame) -> Dict:
        """Process item features for content-based filtering."""
        try:
            features_dict = {}
            
            for _, row in item_features_df.iterrows():
                item_id = row['item_id']
                features = row.drop('item_id').to_dict()
                features_dict[item_id] = features
            
            return features_dict
            
        except Exception as e:
            logger.warning(f"Item features processing failed: {str(e)}")
            return {}
    
    async def _process_user_features(self, user_features_df: pd.DataFrame) -> Dict:
        """Process user features for enhanced recommendations."""
        try:
            profiles_dict = {}
            
            for _, row in user_features_df.iterrows():
                user_id = row['user_id']
                profile = row.drop('user_id').to_dict()
                profiles_dict[user_id] = profile
            
            return profiles_dict
            
        except Exception as e:
            logger.warning(f"User features processing failed: {str(e)}")
            return {}
    
    def _select_algorithms(self, interactions_df: pd.DataFrame) -> List[RecommenderType]:
        """Automatically select appropriate algorithms based on data characteristics."""
        try:
            algorithms = []
            
            n_users = interactions_df['user_id'].nunique()
            n_items = interactions_df['item_id'].nunique()
            n_interactions = len(interactions_df)
            sparsity = 1 - n_interactions / (n_users * n_items)
            
            # Always include popular baseline
            algorithms.append(RecommenderType.POPULARITY_BASED)
            
            # Matrix factorization - good for most cases
            algorithms.append(RecommenderType.MATRIX_FACTORIZATION)
            
            # Collaborative filtering - good for dense data
            if sparsity < 0.99 and n_users > 100 and n_items > 100:
                algorithms.append(RecommenderType.COLLABORATIVE_FILTERING)
            
            # Deep learning - good for large datasets
            if TENSORFLOW_AVAILABLE and n_interactions > 10000:
                algorithms.append(RecommenderType.DEEP_LEARNING)
            
            # Content-based - if item features available
            if self.item_features:
                algorithms.append(RecommenderType.CONTENT_BASED)
            
            return algorithms
            
        except Exception as e:
            logger.warning(f"Algorithm selection failed: {str(e)}")
            return [RecommenderType.MATRIX_FACTORIZATION]
    
    async def _split_data(self, interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data for training and evaluation."""
        try:
            if self.config.feedback_type == FeedbackType.IMPLICIT:
                # For implicit feedback, use temporal split if timestamp available
                if 'timestamp' in interactions_df.columns:
                    df_sorted = interactions_df.sort_values('timestamp')
                    split_point = int(len(df_sorted) * (1 - self.config.test_size))
                    train_df = df_sorted.iloc[:split_point]
                    test_df = df_sorted.iloc[split_point:]
                else:
                    # Random split
                    train_df, test_df = train_test_split(
                        interactions_df,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state
                    )
            else:
                # Stratified split by user
                train_df, test_df = train_test_split(
                    interactions_df,
                    test_size=self.config.test_size,
                    stratify=interactions_df['user_id'],
                    random_state=self.config.random_state
                )
            
            return train_df, test_df
            
        except Exception as e:
            logger.warning(f"Data splitting failed: {str(e)}")
            # Fallback to random split
            return train_test_split(
                interactions_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
    
    async def _train_single_model(
        self,
        algorithm: RecommenderType,
        train_df: pd.DataFrame
    ) -> Tuple[Optional[Any], float]:
        """Train a single recommendation model."""
        try:
            start_time = datetime.now()
            model = None
            
            if algorithm == RecommenderType.COLLABORATIVE_FILTERING:
                model = CollaborativeFilteringRecommender(
                    method=self.config.cf_method,
                    similarity_metric=self.config.cf_similarity_metric,
                    k_neighbors=self.config.cf_k_neighbors,
                    min_support=self.config.cf_min_support
                )
                model.fit(train_df)
                
            elif algorithm == RecommenderType.MATRIX_FACTORIZATION:
                model = MatrixFactorizationRecommender(
                    method=self.config.mf_method,
                    n_factors=self.config.mf_factors,
                    learning_rate=self.config.mf_learning_rate,
                    regularization=self.config.mf_regularization,
                    n_epochs=self.config.mf_epochs
                )
                model.fit(train_df)
                
            elif algorithm == RecommenderType.DEEP_LEARNING and TENSORFLOW_AVAILABLE:
                model = DeepLearningRecommender(
                    embedding_dim=self.config.dl_embedding_dim,
                    hidden_units=self.config.dl_hidden_units,
                    dropout_rate=self.config.dl_dropout_rate,
                    batch_size=self.config.dl_batch_size,
                    epochs=self.config.dl_epochs
                )
                model.fit(train_df)
                
            elif algorithm == RecommenderType.POPULARITY_BASED:
                model = PopularityBasedRecommender()
                model.fit(train_df)
            
            training_time = (datetime.now() - start_time).total_seconds()
            return model, training_time
            
        except Exception as e:
            logger.error(f"Model training failed for {algorithm.value}: {str(e)}")
            return None, 0.0

class PopularityBasedRecommender(BaseEstimator, RegressorMixin):
    """Simple popularity-based recommender as baseline."""
    
    def __init__(self):
        self.item_popularity = None
        self.global_mean = None
    
    def fit(self, interactions_df: pd.DataFrame):
        """Fit popularity-based model."""
        self.item_popularity = interactions_df.groupby('item_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        
        self.item_popularity.columns = ['item_id', 'count', 'mean_rating']
        self.item_popularity['popularity_score'] = (
            self.item_popularity['count'] * self.item_popularity['mean_rating']
        )
        
        self.item_popularity = self.item_popularity.sort_values(
            'popularity_score', ascending=False
        )
        
        self.global_mean = interactions_df['rating'].mean()
        return self
    
    def predict(self, user_item_pairs: List[Tuple]) -> np.ndarray:
        """Predict ratings based on item popularity."""
        predictions = []
        
        for user_id, item_id in user_item_pairs:
            item_info = self.item_popularity[
                self.item_popularity['item_id'] == item_id
            ]
            
            if len(item_info) > 0:
                predictions.append(item_info.iloc[0]['mean_rating'])
            else:
                predictions.append(self.global_mean)
        
        return np.array(predictions)
    
    def recommend(self, user_id: Any, n_recommendations: int = 10, filter_seen: bool = True) -> List[Tuple]:
        """Generate popularity-based recommendations."""
        top_items = self.item_popularity.head(n_recommendations)
        return [(row['item_id'], row['popularity_score']) 
                for _, row in top_items.iterrows()]

    async def _evaluate_models(
        self,
        model_results: Dict,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate all trained models."""
        try:
            evaluation_results = {}
            
            for algorithm, model_info in model_results.items():
                model = model_info['model']
                
                # Generate test predictions/recommendations
                test_users = test_df['user_id'].unique()
                
                # Create ground truth for each user
                ground_truth = {}
                for user_id in test_users:
                    user_items = test_df[test_df['user_id'] == user_id]
                    # For implicit feedback, use all interacted items
                    # For explicit feedback, use highly rated items
                    if self.config.feedback_type == FeedbackType.IMPLICIT:
                        ground_truth[user_id] = user_items['item_id'].tolist()
                    else:
                        # Use items with rating >= 4 as relevant
                        threshold = test_df['rating'].quantile(0.7)
                        relevant_items = user_items[user_items['rating'] >= threshold]
                        ground_truth[user_id] = relevant_items['item_id'].tolist()
                
                # Generate predictions for each user
                predictions = {}
                for user_id in test_users:
                    try:
                        recommendations = model.recommend(
                            user_id,
                            n_recommendations=max(self.config.evaluation_k_values),
                            filter_seen=True
                        )
                        predictions[user_id] = [item for item, score in recommendations]
                    except:
                        predictions[user_id] = []
                
                # Calculate metrics for different k values
                algorithm_results = {}
                for k in self.config.evaluation_k_values:
                    k_results = {}
                    
                    # Calculate metrics
                    precision_scores = []
                    recall_scores = []
                    ndcg_scores = []
                    
                    for user_id in test_users:
                        if user_id in ground_truth and user_id in predictions:
                            y_true = ground_truth[user_id]
                            y_pred = predictions[user_id][:k]
                            
                            if len(y_true) > 0:
                                precision = RecommenderEvaluator.precision_at_k(y_true, y_pred, k)
                                recall = RecommenderEvaluator.recall_at_k(y_true, y_pred, k)
                                ndcg = RecommenderEvaluator.ndcg_at_k(y_true, y_pred, k)
                                
                                precision_scores.append(precision)
                                recall_scores.append(recall)
                                ndcg_scores.append(ndcg)
                    
                    k_results['precision'] = np.mean(precision_scores) if precision_scores else 0.0
                    k_results['recall'] = np.mean(recall_scores) if recall_scores else 0.0
                    k_results['ndcg'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
                    
                    # F1 score
                    if k_results['precision'] + k_results['recall'] > 0:
                        k_results['f1_score'] = (
                            2 * k_results['precision'] * k_results['recall'] /
                            (k_results['precision'] + k_results['recall'])
                        )
                    else:
                        k_results['f1_score'] = 0.0
                    
                    algorithm_results[f'k_{k}'] = k_results
                
                # Calculate coverage and diversity
                all_items = set(train_df['item_id'].unique())
                coverage = RecommenderEvaluator.coverage(predictions, all_items)
                
                algorithm_results['overall'] = {
                    'coverage': coverage,
                    'training_time': model_info['training_time']
                }
                
                evaluation_results[algorithm] = algorithm_results
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}
    
    def _select_best_model(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Select the best performing model."""
        try:
            if not evaluation_results:
                raise ValueError("No evaluation results available")
            
            # Calculate composite score for each algorithm
            algorithm_scores = {}
            
            for algorithm, results in evaluation_results.items():
                # Use k=10 results for selection (or largest k available)
                k_key = 'k_10' if 'k_10' in results else list(results.keys())[0]
                
                if k_key in results:
                    k_results = results[k_key]
                    
                    # Weighted composite score
                    composite_score = (
                        0.4 * k_results.get('precision', 0) +
                        0.3 * k_results.get('recall', 0) +
                        0.3 * k_results.get('ndcg', 0)
                    )
                    
                    algorithm_scores[algorithm] = composite_score
            
            if not algorithm_scores:
                # Fallback to first available model
                first_algorithm = list(evaluation_results.keys())[0]
                return {
                    'algorithm': first_algorithm,
                    'model': self.models[first_algorithm],
                    'score': 0.0
                }
            
            # Select best algorithm
            best_algorithm = max(algorithm_scores, key=algorithm_scores.get)
            best_score = algorithm_scores[best_algorithm]
            
            return {
                'algorithm': best_algorithm,
                'model': self.models[best_algorithm],
                'score': best_score
            }
            
        except Exception as e:
            logger.error(f"Best model selection failed: {str(e)}")
            # Return first available model
            first_algorithm = list(self.models.keys())[0]
            return {
                'algorithm': first_algorithm,
                'model': self.models[first_algorithm],
                'score': 0.0
            }
    
    async def _calculate_popular_items(self, interactions_df: pd.DataFrame) -> List[Tuple]:
        """Calculate popular items for cold start scenarios."""
        try:
            item_stats = interactions_df.groupby('item_id').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            
            item_stats.columns = ['item_id', 'count', 'mean_rating']
            item_stats['popularity_score'] = (
                item_stats['count'] * item_stats['mean_rating']
            )
            
            popular_items = item_stats.sort_values(
                'popularity_score', ascending=False
            ).head(100)
            
            return [(row['item_id'], row['popularity_score']) 
                   for _, row in popular_items.iterrows()]
            
        except Exception as e:
            logger.warning(f"Popular items calculation failed: {str(e)}")
            return []
    
    async def _calculate_business_metrics(
        self,
        interactions_df: pd.DataFrame,
        evaluation_results: Dict
    ) -> Dict[str, Any]:
        """Calculate business-relevant metrics."""
        try:
            metrics = {}
            
            # Dataset characteristics
            n_users = interactions_df['user_id'].nunique()
            n_items = interactions_df['item_id'].nunique()
            n_interactions = len(interactions_df)
            
            metrics['dataset_metrics'] = {
                'sparsity': 1 - n_interactions / (n_users * n_items),
                'avg_interactions_per_user': n_interactions / n_users,
                'avg_interactions_per_item': n_interactions / n_items,
                'rating_distribution': interactions_df['rating'].value_counts().to_dict()
            }
            
            # Business impact estimation
            if self.best_algorithm and self.best_algorithm in evaluation_results:
                best_results = evaluation_results[self.best_algorithm]
                
                # Estimate potential business impact
                k_10_results = best_results.get('k_10', best_results.get('k_5', {}))
                
                if k_10_results:
                    precision = k_10_results.get('precision', 0)
                    
                    # Simplified business impact calculation
                    estimated_conversion_lift = precision * 0.1  # 10% base conversion
                    estimated_revenue_impact = estimated_conversion_lift * n_users * 50  # $50 avg order
                    
                    metrics['business_impact'] = {
                        'estimated_precision': precision,
                        'estimated_conversion_lift': estimated_conversion_lift,
                        'estimated_revenue_impact': estimated_revenue_impact,
                        'recommendation_coverage': k_10_results.get('coverage', 0)
                    }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Business metrics calculation failed: {str(e)}")
            return {}
    
    async def _generate_sample_recommendations(
        self,
        interactions_df: pd.DataFrame,
        n_samples: int = 5
    ) -> List[UserRecommendations]:
        """Generate sample recommendations for report."""
        try:
            sample_recommendations = []
            
            if self.best_model is None:
                return sample_recommendations
            
            # Select sample users
            active_users = interactions_df['user_id'].value_counts().head(20).index
            sample_users = np.random.choice(active_users, min(n_samples, len(active_users)), replace=False)
            
            for user_id in sample_users:
                try:
                    # Generate recommendations
                    recommendations = self.best_model.recommend(
                        user_id,
                        n_recommendations=self.config.top_k_recommendations,
                        filter_seen=self.config.filter_seen_items
                    )
                    
                    # Create recommendation items
                    rec_items = []
                    for i, (item_id, score) in enumerate(recommendations):
                        rec_item = RecommendationItem(
                            item_id=item_id,
                            score=float(score),
                            rank=i + 1,
                            explanation=self._generate_explanation(user_id, item_id) if self.config.enable_explanations else None,
                            confidence=min(1.0, abs(float(score)) / 5.0),  # Normalize to [0,1]
                            metadata={}
                        )
                        rec_items.append(rec_item)
                    
                    # Calculate diversity and novelty (simplified)
                    diversity_score = 0.7  # Placeholder
                    novelty_score = 0.6   # Placeholder
                    
                    user_rec = UserRecommendations(
                        user_id=user_id,
                        recommendations=rec_items,
                        algorithm_used=self.best_algorithm,
                        generation_time=datetime.now(),
                        user_profile=self.user_profiles.get(user_id, {}),
                        diversity_score=diversity_score,
                        novelty_score=novelty_score,
                        metadata={}
                    )
                    
                    sample_recommendations.append(user_rec)
                    
                except Exception as e:
                    logger.warning(f"Sample recommendation generation failed for user {user_id}: {str(e)}")
                    continue
            
            return sample_recommendations
            
        except Exception as e:
            logger.warning(f"Sample recommendations generation failed: {str(e)}")
            return []
    
    def _generate_explanation(self, user_id: Any, item_id: Any) -> str:
        """Generate explanation for why an item was recommended."""
        try:
            if self.config.explanation_method == 'feature_based' and item_id in self.item_features:
                features = self.item_features[item_id]
                return f"Recommended based on features: {', '.join(str(v) for v in list(features.values())[:3])}"
            
            elif self.config.explanation_method == 'collaborative':
                return "Users with similar preferences also liked this item"
            
            else:
                return "Recommended based on your past interactions"
                
        except Exception:
            return "Recommended for you"
    
    def _generate_insights(
        self,
        evaluation_results: Dict,
        business_metrics: Dict
    ) -> List[str]:
        """Generate insights from evaluation results."""
        try:
            insights = []
            
            if not evaluation_results:
                return ["No evaluation results available for insights generation."]
            
            # Performance insights
            if self.best_algorithm:
                best_results = evaluation_results[self.best_algorithm]
                k_10_results = best_results.get('k_10', best_results.get('k_5', {}))
                
                if k_10_results:
                    precision = k_10_results.get('precision', 0)
                    recall = k_10_results.get('recall', 0)
                    
                    if precision > 0.3:
                        insights.append(f"High precision achieved ({precision:.3f}) - recommendations are highly relevant")
                    elif precision < 0.1:
                        insights.append("Low precision detected - consider improving feature engineering or algorithm tuning")
                    
                    if recall > 0.2:
                        insights.append(f"Good recall performance ({recall:.3f}) - system captures user preferences well")
                    elif recall < 0.05:
                        insights.append("Low recall - system may be missing relevant items for users")
            
            # Algorithm comparison insights
            if len(evaluation_results) > 1:
                algorithm_scores = {}
                for alg, results in evaluation_results.items():
                    k_results = results.get('k_10', results.get('k_5', {}))
                    if k_results:
                        score = k_results.get('precision', 0) + k_results.get('recall', 0)
                        algorithm_scores[alg] = score
                
                if algorithm_scores:
                    best_alg = max(algorithm_scores, key=algorithm_scores.get)
                    worst_alg = min(algorithm_scores, key=algorithm_scores.get)
                    
                    insights.append(f"{best_alg} outperformed {worst_alg} in overall metrics")
            
            # Business insights
            if 'dataset_metrics' in business_metrics:
                sparsity = business_metrics['dataset_metrics'].get('sparsity', 0)
                if sparsity > 0.99:
                    insights.append("Very sparse dataset detected - consider content-based or hybrid approaches")
                elif sparsity < 0.9:
                    insights.append("Relatively dense dataset - collaborative filtering should work well")
            
            # Default insight
            if not insights:
                insights.append("Recommendation system successfully trained and evaluated")
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {str(e)}")
            return ["Recommendation system training completed"]
    
    def _generate_recommendations(
        self,
        evaluation_results: Dict,
        insights: List[str]
    ) -> List[str]:
        """Generate actionable recommendations for improvement."""
        try:
            recommendations = []
            
            if not evaluation_results:
                return ["Improve data quality and system configuration"]
            
            # Performance-based recommendations
            if self.best_algorithm:
                best_results = evaluation_results[self.best_algorithm]
                k_10_results = best_results.get('k_10', best_results.get('k_5', {}))
                
                if k_10_results:
                    precision = k_10_results.get('precision', 0)
                    recall = k_10_results.get('recall', 0)
                    
                    if precision < 0.2:
                        recommendations.append("Improve recommendation precision through better feature engineering or hyperparameter tuning")
                    
                    if recall < 0.1:
                        recommendations.append("Increase recall by expanding recommendation diversity or adjusting similarity thresholds")
                    
                    coverage = best_results.get('overall', {}).get('coverage', 0)
                    if coverage < 0.1:
                        recommendations.append("Improve item coverage by addressing popularity bias and cold start problems")
            
            # Algorithm-specific recommendations
            if self.best_algorithm == 'collaborative_filtering':
                recommendations.append("Consider matrix factorization or deep learning for improved scalability")
            elif self.best_algorithm == 'matrix_factorization':
                recommendations.append("Experiment with deep learning models for potential accuracy improvements")
            elif self.best_algorithm == 'popularity_based':
                recommendations.append("Implement personalized algorithms (CF or MF) for better user experience")
            
            # Data-based recommendations
            recommendations.append("Consider collecting additional user and item features for hybrid approaches")
            recommendations.append("Implement A/B testing to validate recommendation improvements in production")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Continue monitoring performance and consider periodic model retraining")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {str(e)}")
            return ["Monitor system performance and user feedback"]
    
    def _get_best_model_parameters(self) -> Dict[str, Any]:
        """Get parameters of the best performing model."""
        try:
            if self.best_algorithm and hasattr(self.best_model, 'get_params'):
                return self.best_model.get_params()
            else:
                return {}
        except:
            return {}
    
    async def _log_to_mlflow(self, report: RecommenderReport):
        """Log recommender results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"recommender_{report.algorithm_type.value}"):
                # Log parameters
                mlflow.log_param("algorithm_type", report.algorithm_type.value)
                mlflow.log_param("n_interactions", report.dataset_info.get('n_interactions', 0))
                mlflow.log_param("n_users", report.dataset_info.get('n_users', 0))
                mlflow.log_param("n_items", report.dataset_info.get('n_items', 0))
                mlflow.log_param("sparsity", report.dataset_info.get('sparsity', 0))
                
                # Log evaluation metrics
                for algorithm, results in report.evaluation_results.items():
                    for k_key, metrics in results.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    mlflow.log_metric(f"{algorithm}_{k_key}_{metric_name}", value)
                
                # Log business metrics
                if 'business_impact' in report.business_metrics:
                    for metric, value in report.business_metrics['business_impact'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"business_{metric}", value)
                
                # Log artifacts
                report_dict = asdict(report)
                report_dict['timestamp'] = report.timestamp.isoformat()
                
                with open("recommender_report.json", "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)
                mlflow.log_artifact("recommender_report.json")
                
                logger.info("Recommender results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def recommend_for_user(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        filter_seen: bool = True,
        diversity_boost: bool = False
    ) -> UserRecommendations:
        """Generate recommendations for a specific user."""
        try:
            if self.best_model is None:
                raise ValueError("No trained model available. Train the recommender first.")
            
            # Check cache first
            cache_key = f"{user_id}_{n_recommendations}_{filter_seen}_{diversity_boost}"
            if self.config.enable_caching and cache_key in self.recommendation_cache:
                cached_result = self.recommendation_cache[cache_key]
                # Check if cache is still valid
                cache_age = (datetime.now() - cached_result.generation_time).total_seconds() / 3600
                if cache_age < self.config.cache_ttl_hours:
                    return cached_result
            
            # Generate fresh recommendations
            recommendations = self.best_model.recommend(
                user_id,
                n_recommendations=n_recommendations,
                filter_seen=filter_seen
            )
            
            # Apply diversity boosting if requested
            if diversity_boost and self.item_features:
                recommendations = self._apply_diversity_boosting(recommendations)
            
            # Create recommendation items
            rec_items = []
            for i, (item_id, score) in enumerate(recommendations):
                rec_item = RecommendationItem(
                    item_id=item_id,
                    score=float(score),
                    rank=i + 1,
                    explanation=self._generate_explanation(user_id, item_id) if self.config.enable_explanations else None,
                    confidence=min(1.0, abs(float(score)) / 5.0),
                    metadata={}
                )
                rec_items.append(rec_item)
            
            # Calculate diversity and novelty
            diversity_score = self._calculate_recommendation_diversity(rec_items)
            novelty_score = self._calculate_recommendation_novelty(rec_items)
            
            user_recommendations = UserRecommendations(
                user_id=user_id,
                recommendations=rec_items,
                algorithm_used=self.best_algorithm,
                generation_time=datetime.now(),
                user_profile=self.user_profiles.get(user_id, {}),
                diversity_score=diversity_score,
                novelty_score=novelty_score,
                metadata={}
            )
            
            # Cache result
            if self.config.enable_caching:
                self.recommendation_cache[cache_key] = user_recommendations
            
            return user_recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed for user {user_id}: {str(e)}")
            # Return empty recommendations
            return UserRecommendations(
                user_id=user_id,
                recommendations=[],
                algorithm_used=self.best_algorithm or "unknown",
                generation_time=datetime.now(),
                user_profile={},
                diversity_score=0.0,
                novelty_score=0.0,
                metadata={'error': str(e)}
            )
    
    def _apply_diversity_boosting(self, recommendations: List[Tuple]) -> List[Tuple]:
        """Apply diversity boosting to recommendations."""
        try:
            if not self.item_features or len(recommendations) <= 1:
                return recommendations
            
            # Simple greedy diversification
            diversified = [recommendations[0]]  # Start with top recommendation
            
            for item_id, score in recommendations[1:]:
                # Calculate similarity to already selected items
                min_similarity = float('inf')
                
                for selected_item, _ in diversified:
                    similarity = self._calculate_item_similarity(item_id, selected_item)
                    min_similarity = min(min_similarity, similarity)
                
                # Boost score based on diversity
                diversity_bonus = (1 - min_similarity) * self.config.diversity_weight
                boosted_score = score + diversity_bonus
                
                diversified.append((item_id, boosted_score))
            
            # Re-sort by boosted scores
            diversified.sort(key=lambda x: x[1], reverse=True)
            return diversified
            
        except Exception as e:
            logger.warning(f"Diversity boosting failed: {str(e)}")
            return recommendations
    
    def _calculate_item_similarity(self, item1: Any, item2: Any) -> float:
        """Calculate similarity between two items based on features."""
        try:
            if item1 not in self.item_features or item2 not in self.item_features:
                return 0.5  # Default similarity
            
            features1 = self.item_features[item1]
            features2 = self.item_features[item2]
            
            # Simple Jaccard similarity for categorical features
            if isinstance(features1, dict) and isinstance(features2, dict):
                common_keys = set(features1.keys()) & set(features2.keys())
                if not common_keys:
                    return 0.0
                
                matches = sum(1 for key in common_keys if features1[key] == features2[key])
                return matches / len(common_keys)
            
            return 0.5  # Default
            
        except Exception:
            return 0.5
    
    def _calculate_recommendation_diversity(self, rec_items: List[RecommendationItem]) -> float:
        """Calculate diversity score for a list of recommendations."""
        try:
            if len(rec_items) <= 1:
                return 0.0
            
            similarities = []
            for i, item1 in enumerate(rec_items):
                for item2 in rec_items[i+1:]:
                    similarity = self._calculate_item_similarity(item1.item_id, item2.item_id)
                    similarities.append(similarity)
            
            return 1 - (sum(similarities) / len(similarities)) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_recommendation_novelty(self, rec_items: List[RecommendationItem]) -> float:
        """Calculate novelty score based on item popularity."""
        try:
            if not self.popular_items or not rec_items:
                return 0.5  # Default novelty
            
            # Create popularity ranking
            popular_item_ids = {item_id for item_id, _ in self.popular_items[:100]}
            
            # Calculate novelty as inverse of popularity
            novelty_scores = []
            for rec_item in rec_items:
                if rec_item.item_id in popular_item_ids:
                    novelty_scores.append(0.3)  # Popular items have low novelty
                else:
                    novelty_scores.append(0.8)  # Less popular items have high novelty
            
            return sum(novelty_scores) / len(novelty_scores)
            
        except Exception:
            return 0.5
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'models_trained': len(self.models),
                'best_algorithm': self.best_algorithm,
                'cache_size': len(self.recommendation_cache),
                'popular_items_count': len(self.popular_items),
                'item_features_count': len(self.item_features),
                'user_profiles_count': len(self.user_profiles),
                'configuration': asdict(self.config)
            }
            
            if self.best_model:
                status['model_ready'] = True
                if hasattr(self.best_model, 'get_params'):
                    status['model_parameters'] = self.best_model.get_params()
            else:
                status['model_ready'] = False
            
            return status
            
        except Exception as e:
            logger.error(f"System status generation failed: {str(e)}")
            return {'error': str(e)}

# Utility functions

def create_recommender_system(
    algorithm_type: str = 'matrix_factorization',
    enable_deep_learning: bool = True,
    enable_explanations: bool = True
) -> RecommenderSystem:
    """Factory function to create a RecommenderSystem."""
    config = RecommenderConfig()
    config.recommender_type = RecommenderType(algorithm_type)
    config.enable_explanations = enable_explanations
    
    if not enable_deep_learning:
        config.dl_epochs = 0  # Disable deep learning
    
    return RecommenderSystem(config)

async def quick_recommendation_analysis(
    interactions_df: pd.DataFrame,
    n_recommendations: int = 10
) -> Dict[str, Any]:
    """Quick recommendation analysis for simple use cases."""
    system = create_recommender_system()
    
    # Use simple configuration
    system.config.mf_epochs = 20  # Faster training
    system.config.evaluation_k_values = [5, 10]
    
    report = await system.train_recommender(interactions_df)
    
    # Generate sample recommendations
    sample_users = interactions_df['user_id'].unique()[:3]
    recommendations = {}
    
    for user_id in sample_users:
        user_recs = await system.recommend_for_user(user_id, n_recommendations)
        recommendations[user_id] = [
            (rec.item_id, rec.score) for rec in user_recs.recommendations
        ]
    
    return {
        'best_algorithm': report.algorithm_type.value,
        'evaluation_metrics': report.evaluation_results,
        'sample_recommendations': recommendations,
        'insights': report.insights
    }

def get_available_algorithms() -> Dict[str, bool]:
    """Get available recommendation algorithms."""
    return {
        'collaborative_filtering': True,
        'matrix_factorization': True,
        'popularity_based': True,
        'deep_learning': TENSORFLOW_AVAILABLE,
        'advanced_matrix_factorization': SURPRISE_AVAILABLE,
        'implicit_feedback': IMPLICIT_AVAILABLE,
        'content_based': True  # Basic implementation available
    }

# Example usage and testing
if __name__ == "__main__":
    async def test_recommender_system():
        """Test the recommender system functionality."""
        print("Testing Recommender System...")
        print("Available algorithms:", get_available_algorithms())
        
        # Create sample interaction data
        np.random.seed(42)
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        
        # Generate realistic interaction data
        user_ids = np.random.choice(range(1, n_users + 1), n_interactions)
        item_ids = np.random.choice(range(1, n_items + 1), n_interactions)
        
        # Generate ratings with some structure
        base_ratings = np.random.normal(3.5, 1.2, n_interactions)
        ratings = np.clip(base_ratings, 1, 5).round().astype(int)
        
        interactions_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': pd.date_range('2023-01-01', periods=n_interactions, freq='H')
        })
        
        # Remove duplicates
        interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'item_id'])
        
        print(f"Generated {len(interactions_df)} unique interactions")
        print(f"Users: {interactions_df['user_id'].nunique()}")
        print(f"Items: {interactions_df['item_id'].nunique()}")
        print(f"Rating distribution: {interactions_df['rating'].value_counts().to_dict()}")
        
        # Create and train recommender system
        system = create_recommender_system()
        
        # Configure for faster testing
        system.config.mf_epochs = 10
        system.config.cf_k_neighbors = 20
        system.config.evaluation_k_values = [5, 10]
        system.config.cross_validation_folds = 3
        
        # Train system
        report = await system.train_recommender(interactions_df)
        
        print(f"\nTraining Results:")
        print(f"Best algorithm: {report.algorithm_type.value}")
        print(f"Dataset sparsity: {report.dataset_info.get('sparsity', 0):.4f}")
        print(f"Training time: {report.performance_metrics.get('training_time', 0):.2f}s")
        
        # Print evaluation results
        print(f"\nEvaluation Results:")
        if report.evaluation_results:
            for algorithm, results in report.evaluation_results.items():
                for k_key, metrics in results.items():
                    if k_key.startswith('k_'):
                        print(f"{algorithm} @ {k_key}:")
                        for metric, value in metrics.items():
                            print(f"  {metric}: {value:.4f}")
        
        # Generate sample recommendations
        sample_user = interactions_df['user_id'].iloc[0]
        user_recommendations = await system.recommend_for_user(
            sample_user, n_recommendations=10, diversity_boost=True
        )
        
        print(f"\nSample Recommendations for User {sample_user}:")
        for rec in user_recommendations.recommendations[:5]:
            print(f"  Item {rec.item_id}: Score {rec.score:.4f}, Rank {rec.rank}")
            if rec.explanation:
                print(f"    Explanation: {rec.explanation}")
        
        print(f"Diversity Score: {user_recommendations.diversity_score:.4f}")
        print(f"Novelty Score: {user_recommendations.novelty_score:.4f}")
        
        # Test different users
        print(f"\nTesting recommendations for multiple users:")
        test_users = interactions_df['user_id'].unique()[:3]
        
        for user_id in test_users:
            try:
                recs = await system.recommend_for_user(user_id, n_recommendations=5)
                print(f"User {user_id}: {len(recs.recommendations)} recommendations generated")
                
                # Show top recommendation
                if recs.recommendations:
                    top_rec = recs.recommendations[0]
                    print(f"  Top: Item {top_rec.item_id} (Score: {top_rec.score:.4f})")
                    
            except Exception as e:
                print(f"User {user_id}: Error - {str(e)}")
        
        # Print insights and recommendations
        print(f"\nBusiness Insights:")
        for insight in report.insights[:3]:
            print(f"  - {insight}")
        
        print(f"\nRecommendations for Improvement:")
        for rec in report.recommendations_for_improvement[:3]:
            print(f"  - {rec}")
        
        # Test system status
        print(f"\nSystem Status:")
        status = system.get_system_status()
        for key, value in status.items():
            if key not in ['configuration', 'model_parameters']:
                print(f"  {key}: {value}")
        
        # Test quick analysis
        print(f"\nTesting Quick Analysis:")
        quick_results = await quick_recommendation_analysis(
            interactions_df.head(1000), n_recommendations=5
        )
        
        print(f"Quick analysis completed:")
        print(f"  Best algorithm: {quick_results['best_algorithm']}")
        print(f"  Sample recommendations generated: {len(quick_results['sample_recommendations'])}")
        print(f"  Insights: {len(quick_results['insights'])}")
        
        return report, user_recommendations
    
    # Run test
    import asyncio
    results = asyncio.run(test_recommender_system())

class HybridRecommender(BaseEstimator, RegressorMixin):
    """Hybrid recommender combining multiple approaches."""
    
    def __init__(
        self,
        algorithms: List[RecommenderType] = None,
        weights: Optional[List[float]] = None,
        combination_method: str = 'weighted_average'
    ):
        self.algorithms = algorithms or [
            RecommenderType.COLLABORATIVE_FILTERING,
            RecommenderType.MATRIX_FACTORIZATION,
            RecommenderType.POPULARITY_BASED
        ]
        self.weights = weights or [0.4, 0.4, 0.2]
        self.combination_method = combination_method
        self.models = {}
        self.trained = False
    
    def fit(self, interactions_df: pd.DataFrame):
        """Fit all component models."""
        try:
            for i, algorithm in enumerate(self.algorithms):
                if algorithm == RecommenderType.COLLABORATIVE_FILTERING:
                    model = CollaborativeFilteringRecommender()
                elif algorithm == RecommenderType.MATRIX_FACTORIZATION:
                    model = MatrixFactorizationRecommender()
                elif algorithm == RecommenderType.POPULARITY_BASED:
                    model = PopularityBasedRecommender()
                else:
                    continue
                
                model.fit(interactions_df)
                self.models[algorithm.value] = model
                logger.info(f"Hybrid component {algorithm.value} trained")
            
            self.trained = True
            return self
            
        except Exception as e:
            logger.error(f"Hybrid model training failed: {str(e)}")
            raise
    
    def predict(self, user_item_pairs: List[Tuple]) -> np.ndarray:
        """Predict using hybrid combination."""
        try:
            if not self.trained:
                raise ValueError("Model not trained")
            
            # Get predictions from all models
            all_predictions = {}
            for alg_name, model in self.models.items():
                try:
                    predictions = model.predict(user_item_pairs)
                    all_predictions[alg_name] = predictions
                except Exception as e:
                    logger.warning(f"Prediction failed for {alg_name}: {str(e)}")
                    continue
            
            if not all_predictions:
                raise ValueError("No model predictions available")
            
            # Combine predictions
            if self.combination_method == 'weighted_average':
                combined_predictions = np.zeros(len(user_item_pairs))
                total_weight = 0
                
                for i, (alg_name, predictions) in enumerate(all_predictions.items()):
                    weight = self.weights[i] if i < len(self.weights) else 1.0
                    combined_predictions += weight * predictions
                    total_weight += weight
                
                combined_predictions /= total_weight
                
            elif self.combination_method == 'rank_fusion':
                # Implement rank-based fusion
                combined_predictions = self._rank_fusion_predict(all_predictions, user_item_pairs)
            else:
                # Simple average
                predictions_array = np.array(list(all_predictions.values()))
                combined_predictions = np.mean(predictions_array, axis=0)
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"Hybrid prediction failed: {str(e)}")
            # Fallback to first available model
            first_model = list(self.models.values())[0]
            return first_model.predict(user_item_pairs)
    
    def _rank_fusion_predict(
        self, 
        all_predictions: Dict[str, np.ndarray], 
        user_item_pairs: List[Tuple]
    ) -> np.ndarray:
        """Combine predictions using rank fusion."""
        try:
            # Convert predictions to ranks
            rank_sums = np.zeros(len(user_item_pairs))
            
            for predictions in all_predictions.values():
                # Convert to ranks (lower rank = higher score)
                ranks = len(predictions) - stats.rankdata(predictions, method='ordinal')
                rank_sums += ranks
            
            # Convert back to scores (higher = better)
            max_rank_sum = np.max(rank_sums)
            return (max_rank_sum - rank_sums) / max_rank_sum
            
        except Exception as e:
            logger.warning(f"Rank fusion failed: {str(e)}")
            # Fallback to simple average
            predictions_array = np.array(list(all_predictions.values()))
            return np.mean(predictions_array, axis=0)
    
    def recommend(self, user_id: Any, n_recommendations: int = 10, filter_seen: bool = True) -> List[Tuple]:
        """Generate hybrid recommendations."""
        try:
            if not self.trained:
                raise ValueError("Model not trained")
            
            # Get recommendations from all models
            all_recommendations = {}
            for alg_name, model in self.models.items():
                try:
                    recs = model.recommend(user_id, n_recommendations * 2, filter_seen)
                    all_recommendations[alg_name] = dict(recs)  # Convert to dict for easier lookup
                except Exception as e:
                    logger.warning(f"Recommendations failed for {alg_name}: {str(e)}")
                    continue
            
            if not all_recommendations:
                return []
            
            # Combine recommendations
            if self.combination_method == 'weighted_average':
                return self._weighted_average_recommend(all_recommendations, n_recommendations)
            elif self.combination_method == 'rank_fusion':
                return self._rank_fusion_recommend(all_recommendations, n_recommendations)
            else:
                return self._simple_combine_recommend(all_recommendations, n_recommendations)
                
        except Exception as e:
            logger.error(f"Hybrid recommendation failed: {str(e)}")
            return []
    
    def _weighted_average_recommend(
        self, 
        all_recommendations: Dict[str, Dict], 
        n_recommendations: int
    ) -> List[Tuple]:
        """Combine recommendations using weighted average."""
        try:
            combined_scores = defaultdict(float)
            combined_weights = defaultdict(float)
            
            for i, (alg_name, recs) in enumerate(all_recommendations.items()):
                weight = self.weights[i] if i < len(self.weights) else 1.0
                
                for item_id, score in recs.items():
                    combined_scores[item_id] += weight * score
                    combined_weights[item_id] += weight
            
            # Normalize by total weights
            final_scores = []
            for item_id, total_score in combined_scores.items():
                normalized_score = total_score / combined_weights[item_id]
                final_scores.append((item_id, normalized_score))
            
            # Sort and return top N
            final_scores.sort(key=lambda x: x[1], reverse=True)
            return final_scores[:n_recommendations]
            
        except Exception as e:
            logger.warning(f"Weighted average combination failed: {str(e)}")
            return []
    
    def _rank_fusion_recommend(
        self, 
        all_recommendations: Dict[str, Dict], 
        n_recommendations: int
    ) -> List[Tuple]:
        """Combine recommendations using rank fusion."""
        try:
            # Get all unique items
            all_items = set()
            for recs in all_recommendations.values():
                all_items.update(recs.keys())
            
            # Calculate combined ranks
            combined_ranks = {}
            
            for item_id in all_items:
                rank_sum = 0
                count = 0
                
                for recs in all_recommendations.values():
                    if item_id in recs:
                        # Convert score to rank (simple approximation)
                        sorted_items = sorted(recs.items(), key=lambda x: x[1], reverse=True)
                        rank = next(i for i, (item, _) in enumerate(sorted_items) if item == item_id)
                        rank_sum += rank
                        count += 1
                
                # Average rank (lower is better)
                if count > 0:
                    avg_rank = rank_sum / count
                    combined_ranks[item_id] = avg_rank
            
            # Sort by rank and return top N
            sorted_items = sorted(combined_ranks.items(), key=lambda x: x[1])
            return [(item_id, 1.0 / (rank + 1)) for item_id, rank in sorted_items[:n_recommendations]]
            
        except Exception as e:
            logger.warning(f"Rank fusion combination failed: {str(e)}")
            return []
    
    def _simple_combine_recommend(
        self, 
        all_recommendations: Dict[str, Dict], 
        n_recommendations: int
    ) -> List[Tuple]:
        """Simple combination of recommendations."""
        try:
            combined_scores = defaultdict(float)
            
            for recs in all_recommendations.values():
                for item_id, score in recs.items():
                    combined_scores[item_id] += score
            
            # Sort and return top N
            sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:n_recommendations]
            
        except Exception as e:
            logger.warning(f"Simple combination failed: {str(e)}")
            return []

class ContextAwareRecommender(BaseEstimator, RegressorMixin):
    """Context-aware recommender that considers contextual information."""
    
    def __init__(
        self,
        base_recommender: BaseEstimator = None,
        context_features: List[str] = None,
        context_weight: float = 0.3
    ):
        self.base_recommender = base_recommender or MatrixFactorizationRecommender()
        self.context_features = context_features or ['time_of_day', 'day_of_week', 'season']
        self.context_weight = context_weight
        self.context_models = {}
        self.context_encoders = {}
        self.trained = False
    
    def fit(self, interactions_df: pd.DataFrame, context_df: Optional[pd.DataFrame] = None):
        """Fit context-aware model."""
        try:
            # Train base recommender
            self.base_recommender.fit(interactions_df)
            
            # Process context features if provided
            if context_df is not None:
                self._fit_context_models(interactions_df, context_df)
            
            self.trained = True
            return self
            
        except Exception as e:
            logger.error(f"Context-aware model training failed: {str(e)}")
            raise
    
    def _fit_context_models(self, interactions_df: pd.DataFrame, context_df: pd.DataFrame):
        """Fit context-specific models."""
        try:
            # Merge interactions with context
            merged_df = interactions_df.merge(context_df, on=['user_id', 'item_id'], how='left')
            
            # Train separate models for each context feature
            for feature in self.context_features:
                if feature in merged_df.columns:
                    # Encode context values
                    encoder = LabelEncoder()
                    context_values = merged_df[feature].fillna('unknown')
                    encoded_values = encoder.fit_transform(context_values)
                    
                    self.context_encoders[feature] = encoder
                    self.context_models[feature] = {}
                    
                    # Train model for each context value
                    for context_val in encoder.classes_:
                        context_mask = context_values == context_val
                        context_data = merged_df[context_mask]
                        
                        if len(context_data) > 10:  # Minimum data requirement
                            model = MatrixFactorizationRecommender(n_epochs=20)
                            model.fit(context_data[['user_id', 'item_id', 'rating']])
                            self.context_models[feature][context_val] = model
                            
        except Exception as e:
            logger.warning(f"Context model fitting failed: {str(e)}")
    
    def predict(self, user_item_pairs: List[Tuple], context: Optional[Dict] = None) -> np.ndarray:
        """Predict with context awareness."""
        try:
            # Base predictions
            base_predictions = self.base_recommender.predict(user_item_pairs)
            
            if context is None or not self.context_models:
                return base_predictions
            
            # Context-aware adjustments
            context_predictions = np.zeros_like(base_predictions)
            context_count = 0
            
            for feature, value in context.items():
                if (feature in self.context_models and 
                    feature in self.context_encoders and
                    value in self.context_models[feature]):
                    
                    context_model = self.context_models[feature][value]
                    ctx_pred = context_model.predict(user_item_pairs)
                    context_predictions += ctx_pred
                    context_count += 1
            
            if context_count > 0:
                context_predictions /= context_count
                
                # Combine base and context predictions
                final_predictions = (
                    (1 - self.context_weight) * base_predictions +
                    self.context_weight * context_predictions
                )
                return final_predictions
            
            return base_predictions
            
        except Exception as e:
            logger.error(f"Context-aware prediction failed: {str(e)}")
            return self.base_recommender.predict(user_item_pairs)
    
    def recommend(
        self, 
        user_id: Any, 
        n_recommendations: int = 10, 
        context: Optional[Dict] = None,
        filter_seen: bool = True
    ) -> List[Tuple]:
        """Generate context-aware recommendations."""
        try:
            if context and self.context_models:
                # Get candidate items
                base_recs = self.base_recommender.recommend(
                    user_id, n_recommendations * 2, filter_seen
                )
                
                # Re-rank with context
                if base_recs:
                    items = [item for item, _ in base_recs]
                    user_item_pairs = [(user_id, item) for item in items]
                    
                    context_scores = self.predict(user_item_pairs, context)
                    context_recs = list(zip(items, context_scores))
                    context_recs.sort(key=lambda x: x[1], reverse=True)
                    
                    return context_recs[:n_recommendations]
            
            # Fallback to base recommender
            return self.base_recommender.recommend(user_id, n_recommendations, filter_seen)
            
        except Exception as e:
            logger.error(f"Context-aware recommendation failed: {str(e)}")
            return self.base_recommender.recommend(user_id, n_recommendations, filter_seen)

# Advanced utility functions

def create_hybrid_recommender(
    algorithms: List[str] = None,
    weights: List[float] = None
) -> HybridRecommender:
    """Factory function to create a HybridRecommender."""
    if algorithms is None:
        algorithms = ['matrix_factorization', 'collaborative_filtering', 'popularity_based']
    
    algorithm_types = [RecommenderType(alg) for alg in algorithms]
    return HybridRecommender(algorithms=algorithm_types, weights=weights)

def create_context_aware_recommender(
    base_algorithm: str = 'matrix_factorization',
    context_features: List[str] = None
) -> ContextAwareRecommender:
    """Factory function to create a ContextAwareRecommender."""
    if base_algorithm == 'matrix_factorization':
        base_model = MatrixFactorizationRecommender()
    elif base_algorithm == 'collaborative_filtering':
        base_model = CollaborativeFilteringRecommender()
    else:
        base_model = PopularityBasedRecommender()
    
    return ContextAwareRecommender(
        base_recommender=base_model,
        context_features=context_features
    )

async def evaluate_recommendation_quality(
    interactions_df: pd.DataFrame,
    recommendations_dict: Dict[Any, List[Tuple]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, Dict[int, float]]:
    """Evaluate recommendation quality using multiple metrics."""
    try:
        # Create ground truth from interactions
        ground_truth = {}
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            
            if user_id not in ground_truth:
                ground_truth[user_id] = []
            
            # Consider high ratings as relevant (threshold can be adjusted)
            threshold = interactions_df['rating'].quantile(0.7)
            if rating >= threshold:
                ground_truth[user_id].append(item_id)
        
        results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_id in ground_truth:
                if user_id in recommendations_dict:
                    y_true = ground_truth[user_id]
                    y_pred = [item for item, _ in recommendations_dict[user_id][:k]]
                    
                    precision = RecommenderEvaluator.precision_at_k(y_true, y_pred, k)
                    recall = RecommenderEvaluator.recall_at_k(y_true, y_pred, k)
                    ndcg = RecommenderEvaluator.ndcg_at_k(y_true, y_pred, k)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    ndcg_scores.append(ndcg)
            
            results[k] = {
                'precision': np.mean(precision_scores) if precision_scores else 0.0,
                'recall': np.mean(recall_scores) if recall_scores else 0.0,
                'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Recommendation quality evaluation failed: {str(e)}")
        return {}

def calculate_recommendation_diversity(
    recommendations_dict: Dict[Any, List[Tuple]],
    item_features: Dict[Any, Dict] = None
) -> Dict[str, float]:
    """Calculate diversity metrics for recommendations."""
    try:
        if not recommendations_dict:
            return {}
        
        # Intra-list diversity (average diversity within each user's recommendations)
        intra_list_diversities = []
        
        for user_id, recs in recommendations_dict.items():
            if len(recs) > 1:
                if item_features:
                    diversity = RecommenderEvaluator.diversity_score(
                        [item for item, _ in recs], item_features
                    )
                else:
                    # Simple diversity based on item distribution
                    items = [item for item, _ in recs]
                    diversity = len(set(items)) / len(items)  # Unique items ratio
                
                intra_list_diversities.append(diversity)
        
        # Inter-list diversity (diversity across all recommendations)
        all_recommended_items = set()
        for recs in recommendations_dict.values():
            all_recommended_items.update(item for item, _ in recs)
        
        total_possible_items = len(all_recommended_items)
        avg_recommendations_per_user = np.mean([len(recs) for recs in recommendations_dict.values()])
        
        inter_list_diversity = total_possible_items / (len(recommendations_dict) * avg_recommendations_per_user)
        
        return {
            'intra_list_diversity': np.mean(intra_list_diversities) if intra_list_diversities else 0.0,
            'inter_list_diversity': min(1.0, inter_list_diversity),
            'catalog_coverage': len(all_recommended_items)
        }
        
    except Exception as e:
        logger.error(f"Diversity calculation failed: {str(e)}")
        return {}

def generate_recommendation_explanations(
    user_id: Any,
    recommendations: List[Tuple],
    interactions_df: pd.DataFrame,
    item_features: Dict = None,
    explanation_type: str = 'collaborative'
) -> List[str]:
    """Generate explanations for recommendations."""
    try:
        explanations = []
        
        # Get user's interaction history
        user_history = interactions_df[interactions_df['user_id'] == user_id]
        
        for item_id, score in recommendations:
            if explanation_type == 'collaborative':
                # Find similar items the user has interacted with
                user_items = user_history['item_id'].tolist()
                
                if user_items:
                    # Simplified similarity (in practice, use actual similarity measures)
                    explanation = f"Recommended because you liked similar items"
                else:
                    explanation = "Recommended based on overall popularity"
                    
            elif explanation_type == 'feature_based' and item_features and item_id in item_features:
                features = item_features[item_id]
                feature_str = ', '.join(str(v) for v in list(features.values())[:3])
                explanation = f"Recommended based on features: {feature_str}"
                
            elif explanation_type == 'popularity':
                explanation = f"Popular item (score: {score:.3f})"
                
            else:
                explanation = f"Recommended with confidence: {score:.3f}"
            
            explanations.append(explanation)
        
        return explanations
        
    except Exception as e:
        logger.warning(f"Explanation generation failed: {str(e)}")
        return ["Recommended for you"] * len(recommendations)

# Business intelligence functions

def analyze_recommendation_business_impact(
    interactions_df: pd.DataFrame,
    recommendations_dict: Dict[Any, List[Tuple]],
    conversion_rate: float = 0.05,
    avg_order_value: float = 50.0
) -> Dict[str, Any]:
    """Analyze potential business impact of recommendations."""
    try:
        analysis = {}
        
        # Basic metrics
        total_users = len(recommendations_dict)
        total_recommendations = sum(len(recs) for recs in recommendations_dict.values())
        avg_recs_per_user = total_recommendations / total_users if total_users > 0 else 0
        
        # Estimate potential impact
        potential_conversions = total_recommendations * conversion_rate
        estimated_revenue = potential_conversions * avg_order_value
        
        analysis['recommendation_metrics'] = {
            'total_users_served': total_users,
            'total_recommendations': total_recommendations,
            'avg_recommendations_per_user': avg_recs_per_user
        }
        
        analysis['business_impact'] = {
            'potential_conversions': potential_conversions,
            'estimated_revenue': estimated_revenue,
            'conversion_rate_assumed': conversion_rate,
            'avg_order_value_assumed': avg_order_value
        }
        
        # Coverage analysis
        all_items = set(interactions_df['item_id'].unique())
        recommended_items = set()
        for recs in recommendations_dict.values():
            recommended_items.update(item for item, _ in recs)
        
        catalog_coverage = len(recommended_items) / len(all_items) if all_items else 0
        
        analysis['coverage_metrics'] = {
            'catalog_coverage': catalog_coverage,
            'items_recommended': len(recommended_items),
            'total_catalog_size': len(all_items)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Business impact analysis failed: {str(e)}")
        return {}

def get_recommendation_insights(
    evaluation_results: Dict,
    diversity_results: Dict,
    business_impact: Dict
) -> List[str]:
    """Generate actionable insights from recommendation analysis."""
    try:
        insights = []
        
        # Performance insights
        if evaluation_results:
            for k, metrics in evaluation_results.items():
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                
                if precision > 0.3:
                    insights.append(f"High precision at k={k} ({precision:.3f}) indicates relevant recommendations")
                elif precision < 0.1:
                    insights.append(f"Low precision at k={k} suggests need for algorithm improvement")
                
                if recall > 0.2:
                    insights.append(f"Good recall at k={k} ({recall:.3f}) shows comprehensive coverage")
        
        # Diversity insights
        if diversity_results:
            intra_diversity = diversity_results.get('intra_list_diversity', 0)
            catalog_coverage = diversity_results.get('catalog_coverage', 0)
            
            if intra_diversity > 0.7:
                insights.append("High recommendation diversity provides good user experience")
            elif intra_diversity < 0.3:
                insights.append("Low diversity may lead to recommendation fatigue")
            
            if catalog_coverage > 1000:
                insights.append(f"Good catalog coverage ({catalog_coverage} items) promotes discovery")
        
        # Business insights
        if business_impact:
            estimated_revenue = business_impact.get('business_impact', {}).get('estimated_revenue', 0)
            coverage = business_impact.get('coverage_metrics', {}).get('catalog_coverage', 0)
            
            if estimated_revenue > 10000:
                insights.append(f"Significant revenue potential: ${estimated_revenue:,.0f}")
            
            if coverage > 0.5:
                insights.append(f"Good catalog coverage ({coverage:.1%}) supports long-tail items")
            elif coverage < 0.1:
                insights.append("Low catalog coverage may miss business opportunities")
        
        # Default insight
        if not insights:
            insights.append("Recommendation system analysis completed successfully")
        
        return insights
        
    except Exception as e:
        logger.warning(f"Insights generation failed: {str(e)}")
        return ["Analysis completed - review detailed metrics"]

# Export main classes and functions
__all__ = [
    'RecommenderSystem',
    'CollaborativeFilteringRecommender', 
    'MatrixFactorizationRecommender',
    'DeepLearningRecommender',
    'HybridRecommender',
    'ContextAwareRecommender',
    'PopularityBasedRecommender',
    'RecommenderEvaluator',
    'RecommenderConfig',
    'RecommenderReport',
    'UserRecommendations',
    'RecommendationItem',
    'create_recommender_system',
    'create_hybrid_recommender',
    'create_context_aware_recommender',
    'quick_recommendation_analysis',
    'evaluate_recommendation_quality',
    'calculate_recommendation_diversity',
    'generate_recommendation_explanations',
    'analyze_recommendation_business_impact',
    'get_recommendation_insights',
    'get_available_algorithms'
]
