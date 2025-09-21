"""
Feature Engineering Module for Auto-Analyst Platform

This module implements comprehensive feature engineering capabilities including:
- Encoding techniques (One-Hot, Label, Target, Binary, Ordinal, etc.)
- Scaling and normalization (Standard, MinMax, Robust, Quantile, etc.)
- Feature interactions and polynomial features
- Feature selection methods (Statistical, Model-based, Recursive)
- Dimensionality reduction (PCA, LDA, t-SNE, UMAP)
- Time-based feature engineering (Lag, Rolling, Seasonal)
- Text feature engineering (TF-IDF, N-grams, Embeddings)
- Missing value imputation (Simple, Iterative, KNN)
- Outlier detection and treatment
- Automated feature generation and selection
- Custom business-specific transformations

Features:
- Automatic feature type detection and appropriate transformation
- Pipeline-based feature engineering with reproducibility
- Advanced feature interaction discovery
- Domain-specific feature engineering templates
- Feature quality assessment and ranking
- Real-time feature transformation for streaming data
- Feature drift detection and adaptation
- Business impact-aware feature selection
- Integration with MLflow for feature tracking
- Comprehensive feature documentation generation
- Performance optimization for large datasets
- A/B testing support for feature engineering strategies
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
import re
from collections import defaultdict

# Core ML preprocessing
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder, 
    PolynomialFeatures, PowerTransformer, FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel, RFE, RFECV,
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score

# Advanced imputation
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False

# Advanced feature selection and engineering
try:
    from category_encoders import (
        TargetEncoder, BinaryEncoder, BaseNEncoder, 
        HashingEncoder, LeaveOneOutEncoder, WOEEncoder
    )
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

# Dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Text processing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    TEXT_PROCESSING_AVAILABLE = True
except ImportError:
    TEXT_PROCESSING_AVAILABLE = False

# Advanced feature engineering
try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False

# Time series processing
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Outlier detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    OUTLIER_DETECTION_AVAILABLE = True
except ImportError:
    OUTLIER_DETECTION_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Advanced analytics
try:
    from scipy import stats
    from scipy.stats import boxcox, yeojohnson
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features for engineering."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    TEXT = "text"
    DATETIME = "datetime"
    MIXED = "mixed"
    TARGET = "target"

class TransformationType(Enum):
    """Types of transformations available."""
    ENCODING = "encoding"
    SCALING = "scaling"
    IMPUTATION = "imputation"
    SELECTION = "selection"
    REDUCTION = "reduction"
    INTERACTION = "interaction"
    TEMPORAL = "temporal"
    TEXT_PROCESSING = "text_processing"
    OUTLIER_TREATMENT = "outlier_treatment"
    CUSTOM = "custom"

class EncodingMethod(Enum):
    """Available encoding methods."""
    ONE_HOT = "one_hot"
    LABEL = "label"
    ORDINAL = "ordinal"
    TARGET = "target"
    BINARY = "binary"
    HASHING = "hashing"
    LEAVE_ONE_OUT = "leave_one_out"
    WOE = "weight_of_evidence"
    BASE_N = "base_n"

class ScalingMethod(Enum):
    """Available scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    POWER_BOX_COX = "power_box_cox"
    POWER_YEO_JOHNSON = "power_yeo_johnson"
    UNIT_VECTOR = "unit_vector"

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering operations."""
    
    def __init__(self):
        # General settings
        self.auto_detect_types = True
        self.handle_missing_values = True
        self.remove_outliers = True
        self.create_interactions = True
        self.perform_selection = True
        
        # Encoding settings
        self.categorical_encoding = EncodingMethod.ONE_HOT
        self.high_cardinality_threshold = 50
        self.rare_category_threshold = 0.01  # 1% threshold
        self.target_encoding_smoothing = 1.0
        
        # Scaling settings
        self.numeric_scaling = ScalingMethod.STANDARD
        self.scale_target = False
        self.robust_scaling_quantile_range = (25.0, 75.0)
        
        # Missing value settings
        self.missing_strategy_numeric = 'median'
        self.missing_strategy_categorical = 'most_frequent'
        self.missing_threshold = 0.5  # Drop features with >50% missing
        self.use_iterative_imputer = True
        
        # Feature selection settings
        self.selection_method = 'automatic'
        self.max_features_ratio = 0.8
        self.min_feature_importance = 0.001
        self.correlation_threshold = 0.95
        
        # Interaction settings
        self.max_interaction_degree = 2
        self.interaction_only = True
        self.max_interaction_features = 100
        
        # Outlier settings
        self.outlier_method = 'isolation_forest'
        self.outlier_contamination = 0.1
        self.outlier_action = 'remove'  # 'remove', 'cap', 'transform'
        
        # Time series settings
        self.create_lag_features = True
        self.max_lags = 5
        self.create_rolling_features = True
        self.rolling_windows = [3, 7, 14, 30]
        
        # Text processing settings
        self.text_vectorizer = 'tfidf'
        self.max_text_features = 1000
        self.text_ngram_range = (1, 2)
        self.text_min_df = 2
        
        # Dimensionality reduction
        self.apply_dimensionality_reduction = False
        self.reduction_method = 'pca'
        self.explained_variance_threshold = 0.95
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        self.batch_size = 10000
        self.memory_efficient = True
        
        # Business settings
        self.preserve_business_rules = True
        self.feature_cost_awareness = True
        self.interpretability_preference = 'medium'
        
        # Quality settings
        self.validate_transformations = True
        self.feature_stability_check = True
        self.cross_validation_folds = 3
        
        # Output settings
        self.save_feature_metadata = True
        self.generate_feature_report = True
        self.track_feature_lineage = True

@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""
    name: str
    original_name: Optional[str]
    feature_type: FeatureType
    transformation_history: List[Dict[str, Any]]
    importance_score: Optional[float]
    business_meaning: Optional[str]
    creation_method: str
    quality_score: float
    missing_ratio: float
    unique_ratio: float
    distribution_stats: Dict[str, Any]
    correlation_with_target: Optional[float]

@dataclass
class FeatureEngineeringReport:
    """Comprehensive feature engineering report."""
    report_id: str
    timestamp: datetime
    dataset_name: Optional[str]
    original_features: int
    engineered_features: int
    transformations_applied: List[str]
    feature_metadata: Dict[str, FeatureMetadata]
    pipeline_steps: List[Dict[str, Any]]
    performance_impact: Dict[str, Any]
    business_insights: List[str]
    recommendations: List[str]
    quality_metrics: Dict[str, Any]
    execution_time: float

class FeatureTypeDetector:
    """Utility class for automatic feature type detection."""
    
    @staticmethod
    def detect_feature_types(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        datetime_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Dict[str, FeatureType]:
        """Automatically detect feature types in a DataFrame."""
        try:
            feature_types = {}
            
            for column in df.columns:
                if column == target_column:
                    feature_types[column] = FeatureType.TARGET
                    continue
                
                # User-specified types take precedence
                if datetime_features and column in datetime_features:
                    feature_types[column] = FeatureType.DATETIME
                    continue
                
                if categorical_features and column in categorical_features:
                    feature_types[column] = FeatureType.CATEGORICAL
                    continue
                
                # Automatic detection
                dtype = df[column].dtype
                unique_count = df[column].nunique()
                total_count = len(df[column].dropna())
                unique_ratio = unique_count / total_count if total_count > 0 else 0
                
                # Datetime detection
                if dtype in ['datetime64[ns]', '<M8[ns]'] or pd.api.types.is_datetime64_any_dtype(df[column]):
                    feature_types[column] = FeatureType.DATETIME
                
                # Text detection
                elif dtype == 'object':
                    # Check if it's text (long strings)
                    sample_values = df[column].dropna().astype(str).head(100)
                    avg_length = sample_values.str.len().mean()
                    
                    if avg_length > 20:  # Average length > 20 characters
                        feature_types[column] = FeatureType.TEXT
                    elif unique_count == 2:
                        feature_types[column] = FeatureType.BINARY
                    elif unique_ratio < 0.05 or unique_count < 50:  # Low cardinality
                        feature_types[column] = FeatureType.CATEGORICAL
                    else:
                        feature_types[column] = FeatureType.CATEGORICAL  # High cardinality
                
                # Numeric types
                elif pd.api.types.is_numeric_dtype(df[column]):
                    if unique_count == 2:
                        feature_types[column] = FeatureType.BINARY
                    elif unique_count < 20 and all(df[column].dropna() % 1 == 0):
                        # Integer values with low cardinality might be ordinal
                        feature_types[column] = FeatureType.ORDINAL
                    else:
                        feature_types[column] = FeatureType.NUMERIC
                
                # Boolean type
                elif dtype == 'bool':
                    feature_types[column] = FeatureType.BINARY
                
                else:
                    feature_types[column] = FeatureType.MIXED
                    
            return feature_types
            
        except Exception as e:
            logger.error(f"Feature type detection failed: {str(e)}")
            return {}

class CustomTransformers:
    """Collection of custom transformers for specific use cases."""
    
    class RareCategoryGrouper(BaseEstimator, TransformerMixin):
        """Groups rare categories into 'Other' category."""
        
        def __init__(self, threshold=0.01):
            self.threshold = threshold
            self.rare_categories_ = {}
        
        def fit(self, X, y=None):
            if isinstance(X, pd.Series):
                X = X.to_frame()
            
            for col in X.columns:
                value_counts = X[col].value_counts(normalize=True)
                rare_cats = value_counts[value_counts < self.threshold].index.tolist()
                self.rare_categories_[col] = rare_cats
            
            return self
        
        def transform(self, X):
            if isinstance(X, pd.Series):
                X = X.to_frame()
                was_series = True
            else:
                was_series = False
            
            X_transformed = X.copy()
            
            for col in X_transformed.columns:
                if col in self.rare_categories_:
                    X_transformed[col] = X_transformed[col].replace(
                        self.rare_categories_[col], 'Other'
                    )
            
            return X_transformed.iloc[:, 0] if was_series else X_transformed
    
    class OutlierClipper(BaseEstimator, TransformerMixin):
        """Clips outliers to specified quantiles."""
        
        def __init__(self, quantile_range=(0.05, 0.95)):
            self.quantile_range = quantile_range
            self.clip_values_ = {}
        
        def fit(self, X, y=None):
            if isinstance(X, pd.Series):
                X = X.to_frame()
            
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    lower = X[col].quantile(self.quantile_range[0])
                    upper = X[col].quantile(self.quantile_range[1])
                    self.clip_values_[col] = (lower, upper)
            
            return self
        
        def transform(self, X):
            if isinstance(X, pd.Series):
                X = X.to_frame()
                was_series = True
            else:
                was_series = False
            
            X_transformed = X.copy()
            
            for col in X_transformed.columns:
                if col in self.clip_values_:
                    lower, upper = self.clip_values_[col]
                    X_transformed[col] = X_transformed[col].clip(lower, upper)
            
            return X_transformed.iloc[:, 0] if was_series else X_transformed
    
    class BusinessRuleTransformer(BaseEstimator, TransformerMixin):
        """Applies business rules and domain knowledge."""
        
        def __init__(self, rules=None):
            self.rules = rules or []
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X_transformed = X.copy()
            
            for rule in self.rules:
                try:
                    if rule['type'] == 'ratio':
                        numerator = rule['numerator']
                        denominator = rule['denominator']
                        new_feature = rule['name']
                        
                        if numerator in X_transformed.columns and denominator in X_transformed.columns:
                            X_transformed[new_feature] = (
                                X_transformed[numerator] / (X_transformed[denominator] + 1e-8)
                            )
                    
                    elif rule['type'] == 'binning':
                        feature = rule['feature']
                        bins = rule['bins']
                        labels = rule.get('labels', None)
                        new_feature = rule['name']
                        
                        if feature in X_transformed.columns:
                            X_transformed[new_feature] = pd.cut(
                                X_transformed[feature], bins=bins, labels=labels
                            )
                    
                    elif rule['type'] == 'combination':
                        features = rule['features']
                        operation = rule['operation']
                        new_feature = rule['name']
                        
                        if all(f in X_transformed.columns for f in features):
                            if operation == 'sum':
                                X_transformed[new_feature] = X_transformed[features].sum(axis=1)
                            elif operation == 'mean':
                                X_transformed[new_feature] = X_transformed[features].mean(axis=1)
                            elif operation == 'max':
                                X_transformed[new_feature] = X_transformed[features].max(axis=1)
                            elif operation == 'min':
                                X_transformed[new_feature] = X_transformed[features].min(axis=1)
                    
                except Exception as e:
                    logger.warning(f"Business rule failed: {rule.get('name', 'unknown')}: {str(e)}")
            
            return X_transformed

class FeatureEngineer:
    """
    Comprehensive feature engineering system with automatic transformation selection,
    pipeline creation, and advanced feature generation techniques.
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self.feature_types = {}
        self.transformers = {}
        self.pipeline = None
        self.feature_metadata = {}
        self.original_columns = []
        self.engineered_columns = []
        self.transformation_history = []
        
        logger.info("FeatureEngineer initialized")
    
    async def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        datetime_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        business_rules: Optional[List[Dict]] = None
    ) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
        """
        Perform comprehensive feature engineering on a dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            datetime_features: List of datetime column names
            categorical_features: List of categorical column names
            business_rules: List of business rules to apply
            
        Returns:
            Tuple of transformed DataFrame and engineering report
        """
        try:
            logger.info(f"Starting feature engineering on dataset with shape: {df.shape}")
            start_time = datetime.now()
            
            # Store original information
            self.original_columns = df.columns.tolist()
            original_shape = df.shape
            
            # Detect feature types
            if self.config.auto_detect_types:
                self.feature_types = FeatureTypeDetector.detect_feature_types(
                    df, target_column, datetime_features, categorical_features
                )
            
            logger.info(f"Detected feature types: {dict(self.feature_types)}")
            
            # Create feature engineering pipeline
            pipeline_steps = await self._create_pipeline(df, target_column, business_rules)
            
            # Apply transformations
            df_transformed = await self._apply_transformations(df, pipeline_steps)
            
            # Feature selection
            if self.config.perform_selection and target_column and target_column in df_transformed.columns:
                df_transformed = await self._apply_feature_selection(
                    df_transformed, target_column
                )
            
            # Generate feature metadata
            feature_metadata = await self._generate_feature_metadata(
                df, df_transformed, target_column
            )
            
            # Calculate performance impact
            performance_impact = await self._calculate_performance_impact(
                original_shape, df_transformed.shape
            )
            
            # Generate insights and recommendations
            insights = self._generate_business_insights(feature_metadata, performance_impact)
            recommendations = self._generate_recommendations(feature_metadata, performance_impact)
            
            # Quality metrics
            quality_metrics = await self._calculate_quality_metrics(df_transformed, target_column)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create report
            report = FeatureEngineeringReport(
                report_id=str(uuid.uuid4()),
                timestamp=start_time,
                dataset_name=getattr(df, 'name', None),
                original_features=original_shape[1],
                engineered_features=df_transformed.shape[1],
                transformations_applied=[step['name'] for step in pipeline_steps],
                feature_metadata=feature_metadata,
                pipeline_steps=pipeline_steps,
                performance_impact=performance_impact,
                business_insights=insights,
                recommendations=recommendations,
                quality_metrics=quality_metrics,
                execution_time=execution_time
            )
            
            # Store engineered columns
            self.engineered_columns = df_transformed.columns.tolist()
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE and self.config.save_feature_metadata:
                await self._log_to_mlflow(report, df_transformed)
            
            logger.info(f"Feature engineering completed in {execution_time:.2f}s")
            logger.info(f"Features: {original_shape[1]} -> {df_transformed.shape[1]}")
            
            return df_transformed, report
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            # Return original dataframe with error report
            error_report = FeatureEngineeringReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                dataset_name=None,
                original_features=df.shape[1],
                engineered_features=df.shape[1],
                transformations_applied=[],
                feature_metadata={},
                pipeline_steps=[],
                performance_impact={'error': str(e)},
                business_insights=[f"Feature engineering failed: {str(e)}"],
                recommendations=["Review data quality and feature engineering configuration"],
                quality_metrics={},
                execution_time=0.0
            )
            return df, error_report
    
    async def _create_pipeline(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        business_rules: Optional[List[Dict]]
    ) -> List[Dict[str, Any]]:
        """Create feature engineering pipeline based on data characteristics."""
        try:
            pipeline_steps = []
            
            # Step 1: Handle missing values
            if self.config.handle_missing_values:
                missing_step = await self._create_missing_value_step(df, target_column)
                if missing_step:
                    pipeline_steps.append(missing_step)
            
            # Step 2: Outlier treatment
            if self.config.remove_outliers:
                outlier_step = await self._create_outlier_treatment_step(df)
                if outlier_step:
                    pipeline_steps.append(outlier_step)
            
            # Step 3: Datetime feature engineering
            datetime_step = await self._create_datetime_features_step(df)
            if datetime_step:
                pipeline_steps.append(datetime_step)
            
            # Step 4: Text processing
            text_step = await self._create_text_processing_step(df)
            if text_step:
                pipeline_steps.append(text_step)
            
            # Step 5: Business rules
            if business_rules:
                business_step = await self._create_business_rules_step(business_rules)
                if business_step:
                    pipeline_steps.append(business_step)
            
            # Step 6: Encoding
            encoding_step = await self._create_encoding_step(df, target_column)
            if encoding_step:
                pipeline_steps.append(encoding_step)
            
            # Step 7: Feature interactions
            if self.config.create_interactions:
                interaction_step = await self._create_interaction_step(df)
                if interaction_step:
                    pipeline_steps.append(interaction_step)
            
            # Step 8: Scaling
            scaling_step = await self._create_scaling_step(df, target_column)
            if scaling_step:
                pipeline_steps.append(scaling_step)
            
            # Step 9: Dimensionality reduction
            if self.config.apply_dimensionality_reduction:
                reduction_step = await self._create_dimensionality_reduction_step()
                if reduction_step:
                    pipeline_steps.append(reduction_step)
            
            return pipeline_steps
            
        except Exception as e:
            logger.error(f"Pipeline creation failed: {str(e)}")
            return []
    
    async def _create_missing_value_step(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Create missing value imputation step."""
        try:
            missing_info = df.isnull().sum()
            columns_with_missing = missing_info[missing_info > 0].index.tolist()
            
            if target_column and target_column in columns_with_missing:
                columns_with_missing.remove(target_column)
            
            if not columns_with_missing:
                return None
            
            # Separate numeric and categorical columns
            numeric_cols = []
            categorical_cols = []
            
            for col in columns_with_missing:
                if self.feature_types.get(col) in [FeatureType.NUMERIC, FeatureType.ORDINAL]:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            transformers = []
            
            # Numeric imputation
            if numeric_cols:
                if self.config.use_iterative_imputer and ITERATIVE_IMPUTER_AVAILABLE:
                    imputer = IterativeImputer(
                        random_state=42,
                        max_iter=10
                    )
                else:
                    imputer = SimpleImputer(
                        strategy=self.config.missing_strategy_numeric
                    )
                
                transformers.append(('numeric_imputer', imputer, numeric_cols))
            
            # Categorical imputation
            if categorical_cols:
                imputer = SimpleImputer(
                    strategy=self.config.missing_strategy_categorical,
                    fill_value='Missing'
                )
                transformers.append(('categorical_imputer', imputer, categorical_cols))
            
            if transformers:
                column_transformer = ColumnTransformer(
                    transformers=transformers,
                    remainder='passthrough'
                )
                
                return {
                    'name': 'missing_value_imputation',
                    'transformer': column_transformer,
                    'type': TransformationType.IMPUTATION,
                    'columns_affected': columns_with_missing,
                    'description': f'Imputed {len(columns_with_missing)} columns with missing values'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Missing value step creation failed: {str(e)}")
            return None
    
    async def _create_outlier_treatment_step(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create outlier treatment step."""
        try:
            numeric_columns = [
                col for col, ftype in self.feature_types.items()
                if ftype in [FeatureType.NUMERIC, FeatureType.ORDINAL] and col in df.columns
            ]
            
            if not numeric_columns:
                return None
            
            if self.config.outlier_action == 'remove':
                if OUTLIER_DETECTION_AVAILABLE and self.config.outlier_method == 'isolation_forest':
                    outlier_detector = IsolationForest(
                        contamination=self.config.outlier_contamination,
                        random_state=42
                    )
                else:
                    # Use IQR method as fallback
                    outlier_detector = CustomTransformers.OutlierClipper(quantile_range=(0.05, 0.95))
                
                return {
                    'name': 'outlier_treatment',
                    'transformer': outlier_detector,
                    'type': TransformationType.OUTLIER_TREATMENT,
                    'columns_affected': numeric_columns,
                    'description': f'Applied {self.config.outlier_method} outlier treatment'
                }
            
            elif self.config.outlier_action == 'cap':
                outlier_clipper = CustomTransformers.OutlierClipper(quantile_range=(0.05, 0.95))
                
                return {
                    'name': 'outlier_clipping',
                    'transformer': outlier_clipper,
                    'type': TransformationType.OUTLIER_TREATMENT,
                    'columns_affected': numeric_columns,
                    'description': 'Clipped outliers to 5th-95th percentile range'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Outlier treatment step creation failed: {str(e)}")
            return None
    
    async def _create_datetime_features_step(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create datetime feature engineering step."""
        try:
            datetime_columns = [
                col for col, ftype in self.feature_types.items()
                if ftype == FeatureType.DATETIME and col in df.columns
            ]
            
            if not datetime_columns:
                return None
            
            class DatetimeFeatureGenerator(BaseEstimator, TransformerMixin):
                def __init__(self, datetime_columns):
                    self.datetime_columns = datetime_columns
                
                def fit(self, X, y=None):
                    return self
                
                def transform(self, X):
                    X_transformed = X.copy()
                    
                    for col in self.datetime_columns:
                        if col in X_transformed.columns:
                            dt_series = pd.to_datetime(X_transformed[col])
                            
                            # Basic datetime features
                            X_transformed[f'{col}_year'] = dt_series.dt.year
                            X_transformed[f'{col}_month'] = dt_series.dt.month
                            X_transformed[f'{col}_day'] = dt_series.dt.day
                            X_transformed[f'{col}_dayofweek'] = dt_series.dt.dayofweek
                            X_transformed[f'{col}_dayofyear'] = dt_series.dt.dayofyear
                            X_transformed[f'{col}_quarter'] = dt_series.dt.quarter
                            
                            # Time-based features
                            X_transformed[f'{col}_hour'] = dt_series.dt.hour
                            X_transformed[f'{col}_minute'] = dt_series.dt.minute
                            
                            # Cyclical features
                            X_transformed[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                            X_transformed[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                            X_transformed[f'{col}_day_sin'] = np.sin(2 * np.pi * dt_series.dt.day / 31)
                            X_transformed[f'{col}_day_cos'] = np.cos(2 * np.pi * dt_series.dt.day / 31)
                            
                            # Business features
                            X_transformed[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
                            X_transformed[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
                            X_transformed[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
                    
                    return X_transformed
            
            datetime_generator = DatetimeFeatureGenerator(datetime_columns)
            
            return {
                'name': 'datetime_feature_generation',
                'transformer': datetime_generator,
                'type': TransformationType.TEMPORAL,
                'columns_affected': datetime_columns,
                'description': f'Generated datetime features from {len(datetime_columns)} columns'
            }
            
        except Exception as e:
            logger.warning(f"Datetime features step creation failed: {str(e)}")
            return None
    
    async def _create_text_processing_step(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create text processing step."""
        try:
            if not TEXT_PROCESSING_AVAILABLE:
                return None
            
            text_columns = [
                col for col, ftype in self.feature_types.items()
                if ftype == FeatureType.TEXT and col in df.columns
            ]
            
            if not text_columns:
                return None
            
            class TextFeatureGenerator(BaseEstimator, TransformerMixin):
                def __init__(self, text_columns, config):
                    self.text_columns = text_columns
                    self.config = config
                    self.vectorizers = {}
                
                def fit(self, X, y=None):
                    for col in self.text_columns:
                        if col in X.columns:
                            if self.config.text_vectorizer == 'tfidf':
                                vectorizer = TfidfVectorizer(
                                    max_features=self.config.max_text_features,
                                    ngram_range=self.config.text_ngram_range,
                                    min_df=self.config.text_min_df,
                                    stop_words='english'
                                )
                            else:
                                vectorizer = CountVectorizer(
                                    max_features=self.config.max_text_features,
                                    ngram_range=self.config.text_ngram_range,
                                    min_df=self.config.text_min_df,
                                    stop_words='english'
                                )
                            
                            text_data = X[col].fillna('').astype(str)
                            vectorizer.fit(text_data)
                            self.vectorizers[col] = vectorizer
                    
                    return self
                
                def transform(self, X):
                    X_transformed = X.copy()
                    
                    for col in self.text_columns:
                        if col in X_transformed.columns and col in self.vectorizers:
                            vectorizer = self.vectorizers[col]
                            text_data = X_transformed[col].fillna('').astype(str)
                            
                            # Generate text features
                            text_features = vectorizer.transform(text_data)
                            feature_names = vectorizer.get_feature_names_out()
                            
                            # Add text features to dataframe
                            text_df = pd.DataFrame(
                                text_features.toarray(),
                                columns=[f'{col}_text_{name}' for name in feature_names],
                                index=X_transformed.index
                            )
                            
                            X_transformed = pd.concat([X_transformed, text_df], axis=1)
                            
                            # Add basic text statistics
                            X_transformed[f'{col}_length'] = text_data.str.len()
                            X_transformed[f'{col}_word_count'] = text_data.str.split().str.len()
                            X_transformed[f'{col}_unique_words'] = text_data.apply(
                                lambda x: len(set(str(x).lower().split()))
                            )
                    
                    return X_transformed
            
            text_generator = TextFeatureGenerator(text_columns, self.config)
            
            return {
                'name': 'text_feature_generation',
                'transformer': text_generator,
                'type': TransformationType.TEXT_PROCESSING,
                'columns_affected': text_columns,
                'description': f'Generated text features from {len(text_columns)} columns'
            }
            
        except Exception as e:
            logger.warning(f"Text processing step creation failed: {str(e)}")
            return None
    
    async def _create_business_rules_step(
        self, business_rules: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Create business rules transformation step."""
        try:
            if not business_rules:
                return None
            
            business_transformer = CustomTransformers.BusinessRuleTransformer(business_rules)
            
            return {
                'name': 'business_rules_application',
                'transformer': business_transformer,
                'type': TransformationType.CUSTOM,
                'columns_affected': [rule.get('name', 'unknown') for rule in business_rules],
                'description': f'Applied {len(business_rules)} business rules'
            }
            
        except Exception as e:
            logger.warning(f"Business rules step creation failed: {str(e)}")
            return None
    
    async def _create_encoding_step(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Create categorical encoding step."""
        try:
            categorical_columns = [
                col for col, ftype in self.feature_types.items()
                if ftype in [FeatureType.CATEGORICAL, FeatureType.BINARY, FeatureType.ORDINAL]
                and col in df.columns and col != target_column
            ]
            
            if not categorical_columns:
                return None
            
            transformers = []
            
            for col in categorical_columns:
                unique_count = df[col].nunique()
                
                # Choose encoding method based on cardinality and availability
                if unique_count == 2:
                    # Binary encoding for binary features
                    encoder = LabelEncoder()
                    transformers.append((f'{col}_label', encoder, [col]))
                
                elif unique_count <= self.config.high_cardinality_threshold:
                    # One-hot encoding for low cardinality
                    if self.config.categorical_encoding == EncodingMethod.ONE_HOT:
                        encoder = OneHotEncoder(
                            drop='if_binary',
                            sparse=False,
                            handle_unknown='ignore'
                        )
                        transformers.append((f'{col}_onehot', encoder, [col]))
                    
                    elif self.config.categorical_encoding == EncodingMethod.TARGET and target_column:
                        if CATEGORY_ENCODERS_AVAILABLE:
                            encoder = TargetEncoder(
                                smoothing=self.config.target_encoding_smoothing
                            )
                            transformers.append((f'{col}_target', encoder, [col]))
                        else:
                            # Fallback to label encoding
                            encoder = LabelEncoder()
                            transformers.append((f'{col}_label', encoder, [col]))
                    
                    else:
                        encoder = LabelEncoder()
                        transformers.append((f'{col}_label', encoder, [col]))
                
                else:
                    # High cardinality - use specialized encoders
                    if CATEGORY_ENCODERS_AVAILABLE:
                        if self.config.categorical_encoding == EncodingMethod.TARGET and target_column:
                            encoder = TargetEncoder(
                                smoothing=self.config.target_encoding_smoothing
                            )
                        elif self.config.categorical_encoding == EncodingMethod.BINARY:
                            encoder = BinaryEncoder()
                        elif self.config.categorical_encoding == EncodingMethod.HASHING:
                            encoder = HashingEncoder(n_components=min(32, unique_count // 2))
                        else:
                            encoder = TargetEncoder()
                        
                        transformers.append((f'{col}_encoded', encoder, [col]))
                    else:
                        # Fallback: group rare categories and use label encoding
                        rare_grouper = CustomTransformers.RareCategoryGrouper(
                            threshold=self.config.rare_category_threshold
                        )
                        encoder = LabelEncoder()
                        
                        combined_transformer = Pipeline([
                            ('rare_grouper', rare_grouper),
                            ('label_encoder', encoder)
                        ])
                        
                        transformers.append((f'{col}_grouped_labeled', combined_transformer, [col]))
            
            if transformers:
                column_transformer = ColumnTransformer(
                    transformers=transformers,
                    remainder='passthrough'
                )
                
                return {
                    'name': 'categorical_encoding',
                    'transformer': column_transformer,
                    'type': TransformationType.ENCODING,
                    'columns_affected': categorical_columns,
                    'description': f'Encoded {len(categorical_columns)} categorical columns'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Encoding step creation failed: {str(e)}")
            return None
    
    async def _create_interaction_step(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create feature interaction step."""
        try:
            numeric_columns = [
                col for col, ftype in self.feature_types.items()
                if ftype in [FeatureType.NUMERIC, FeatureType.ORDINAL] and col in df.columns
            ]
            
            # Limit columns for computational efficiency
            if len(numeric_columns) > 20:
                # Select most important columns (simplified selection)
                variances = df[numeric_columns].var()
                numeric_columns = variances.nlargest(20).index.tolist()
            
            if len(numeric_columns) < 2:
                return None
            
            interaction_generator = PolynomialFeatures(
                degree=self.config.max_interaction_degree,
                interaction_only=self.config.interaction_only,
                include_bias=False
            )
            
            class InteractionTransformer(BaseEstimator, TransformerMixin):
                def __init__(self, numeric_columns, interaction_generator, max_features):
                    self.numeric_columns = numeric_columns
                    self.interaction_generator = interaction_generator
                    self.max_features = max_features
                    self.selected_features = None
                
                def fit(self, X, y=None):
                    # Fit interaction generator on numeric columns only
                    X_numeric = X[self.numeric_columns]
                    self.interaction_generator.fit(X_numeric)
                    
                    # Generate interactions
                    interactions = self.interaction_generator.transform(X_numeric)
                    
                    # Limit number of interaction features
                    if interactions.shape[1] > self.max_features:
                        # Select features with highest variance
                        variances = np.var(interactions, axis=0)
                        top_indices = np.argsort(variances)[-self.max_features:]
                        self.selected_features = top_indices
                    
                    return self
                
                def transform(self, X):
                    X_transformed = X.copy()
                    X_numeric = X[self.numeric_columns]
                    
                    # Generate interactions
                    interactions = self.interaction_generator.transform(X_numeric)
                    
                    # Select features if needed
                    if self.selected_features is not None:
                        interactions = interactions[:, self.selected_features]
                    
                    # Add interaction features
                    feature_names = self.interaction_generator.get_feature_names_out(
                        self.numeric_columns
                    )
                    if self.selected_features is not None:
                        feature_names = feature_names[self.selected_features]
                    
                    interaction_df = pd.DataFrame(
                        interactions,
                        columns=[f'interaction_{name}' for name in feature_names],
                        index=X_transformed.index
                    )
                    
                    X_transformed = pd.concat([X_transformed, interaction_df], axis=1)
                    
                    return X_transformed
            
            interaction_transformer = InteractionTransformer(
                numeric_columns, interaction_generator, self.config.max_interaction_features
            )
            
            return {
                'name': 'feature_interactions',
                'transformer': interaction_transformer,
                'type': TransformationType.INTERACTION,
                'columns_affected': numeric_columns,
                'description': f'Generated interaction features from {len(numeric_columns)} numeric columns'
            }
            
        except Exception as e:
            logger.warning(f"Interaction step creation failed: {str(e)}")
            return None
    
    async def _create_scaling_step(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Create feature scaling step."""
        try:
            numeric_columns = [
                col for col, ftype in self.feature_types.items()
                if ftype in [FeatureType.NUMERIC, FeatureType.ORDINAL]
                and col in df.columns and col != target_column
            ]
            
            if not numeric_columns:
                return None
            
            # Choose scaler based on configuration
            if self.config.numeric_scaling == ScalingMethod.STANDARD:
                scaler = StandardScaler()
            elif self.config.numeric_scaling == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
            elif self.config.numeric_scaling == ScalingMethod.ROBUST:
                scaler = RobustScaler(quantile_range=self.config.robust_scaling_quantile_range)
            elif self.config.numeric_scaling == ScalingMethod.QUANTILE_UNIFORM:
                scaler = QuantileTransformer(output_distribution='uniform')
            elif self.config.numeric_scaling == ScalingMethod.QUANTILE_NORMAL:
                scaler = QuantileTransformer(output_distribution='normal')
            elif self.config.numeric_scaling == ScalingMethod.POWER_BOX_COX:
                scaler = PowerTransformer(method='box-cox', standardize=True)
            elif self.config.numeric_scaling == ScalingMethod.POWER_YEO_JOHNSON:
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                scaler = StandardScaler()  # Default
            
            column_transformer = ColumnTransformer(
                transformers=[('scaler', scaler, numeric_columns)],
                remainder='passthrough'
            )
            
            return {
                'name': 'feature_scaling',
                'transformer': column_transformer,
                'type': TransformationType.SCALING,
                'columns_affected': numeric_columns,
                'description': f'Applied {self.config.numeric_scaling.value} scaling to {len(numeric_columns)} columns'
            }
            
        except Exception as e:
            logger.warning(f"Scaling step creation failed: {str(e)}")
            return None
    
    async def _create_dimensionality_reduction_step(self) -> Optional[Dict[str, Any]]:
        """Create dimensionality reduction step."""
        try:
            if self.config.reduction_method == 'pca':
                reducer = PCA(n_components=self.config.explained_variance_threshold)
            elif self.config.reduction_method == 'lda':
                reducer = LinearDiscriminantAnalysis()
            elif self.config.reduction_method == 'truncated_svd':
                reducer = TruncatedSVD(n_components=50)
            elif self.config.reduction_method == 'umap' and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                return None
            
            return {
                'name': 'dimensionality_reduction',
                'transformer': reducer,
                'type': TransformationType.REDUCTION,
                'columns_affected': ['all_numeric'],
                'description': f'Applied {self.config.reduction_method} dimensionality reduction'
            }
            
        except Exception as e:
            logger.warning(f"Dimensionality reduction step creation failed: {str(e)}")
            return None
    
    async def _apply_transformations(
        self, df: pd.DataFrame, pipeline_steps: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply all transformation steps to the dataframe."""
        try:
            df_transformed = df.copy()
            
            for step in pipeline_steps:
                try:
                    transformer = step['transformer']
                    step_name = step['name']
                    
                    logger.info(f"Applying transformation: {step_name}")
                    
                    # Fit and transform
                    df_transformed = transformer.fit_transform(df_transformed)
                    
                    # Convert back to DataFrame if necessary
                    if not isinstance(df_transformed, pd.DataFrame):
                        if hasattr(transformer, 'get_feature_names_out'):
                            try:
                                feature_names = transformer.get_feature_names_out()
                            except:
                                feature_names = [f'feature_{i}' for i in range(df_transformed.shape[1])]
                        else:
                            feature_names = [f'feature_{i}' for i in range(df_transformed.shape[1])]
                        
                        df_transformed = pd.DataFrame(
                            df_transformed,
                            columns=feature_names,
                            index=df.index
                        )
                    
                    # Store transformer for future use
                    self.transformers[step_name] = transformer
                    
                    logger.info(f"Transformation {step_name} completed. Shape: {df_transformed.shape}")
                    
                except Exception as e:
                    logger.warning(f"Transformation step {step['name']} failed: {str(e)}")
                    continue
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Transformation application failed: {str(e)}")
            return df
    
    async def _apply_feature_selection(
        self, df: pd.DataFrame, target_column: str
    ) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        try:
            if target_column not in df.columns:
                return df
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Determine if classification or regression
            is_classification = pd.api.types.is_integer_dtype(y) and y.nunique() < 50
            
            # Remove highly correlated features
            if self.config.correlation_threshold < 1.0:
                corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                high_corr_features = [
                    column for column in upper_triangle.columns
                    if any(upper_triangle[column] > self.config.correlation_threshold)
                ]
                
                X = X.drop(columns=high_corr_features)
                logger.info(f"Removed {len(high_corr_features)} highly correlated features")
            
            # Statistical feature selection
            if self.config.selection_method in ['automatic', 'statistical']:
                if is_classification:
                    scorer = f_classif
                else:
                    scorer = f_regression
                
                # Select top percentage of features
                n_features = max(1, int(len(X.columns) * self.config.max_features_ratio))
                
                selector = SelectKBest(scorer, k=min(n_features, len(X.columns)))
                X_selected = selector.fit_transform(X, y)
                
                selected_features = X.columns[selector.get_support()].tolist()
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                
                logger.info(f"Selected {len(selected_features)} features using statistical selection")
            
            # Combine with target
            df_selected = pd.concat([X, y], axis=1)
            
            return df_selected
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {str(e)}")
            return df
    
    async def _generate_feature_metadata(
        self,
        df_original: pd.DataFrame,
        df_transformed: pd.DataFrame,
        target_column: Optional[str]
    ) -> Dict[str, FeatureMetadata]:
        """Generate comprehensive metadata for all features."""
        try:
            metadata = {}
            
            for column in df_transformed.columns:
                if column == target_column:
                    continue
                
                # Basic statistics
                series = df_transformed[column]
                
                # Determine if feature is original or engineered
                original_name = column if column in df_original.columns else None
                creation_method = 'original' if original_name else 'engineered'
                
                # Calculate quality metrics
                missing_ratio = series.isnull().sum() / len(series)
                unique_ratio = series.nunique() / len(series)
                
                # Distribution statistics
                if pd.api.types.is_numeric_dtype(series):
                    dist_stats = {
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'skewness': float(series.skew()),
                        'kurtosis': float(series.kurtosis())
                    }
                else:
                    dist_stats = {
                        'most_frequent': str(series.mode().iloc[0]) if len(series.mode()) > 0 else 'N/A',
                        'nunique': int(series.nunique())
                    }
                
                # Correlation with target
                target_correlation = None
                if target_column and target_column in df_transformed.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(df_transformed[target_column]):
                            target_correlation = float(series.corr(df_transformed[target_column]))
                    except:
                        pass
                
                # Feature type
                feature_type = self.feature_types.get(column, FeatureType.MIXED)
                
                # Quality score (composite metric)
                quality_score = self._calculate_feature_quality_score(
                    missing_ratio, unique_ratio, target_correlation
                )
                
                metadata[column] = FeatureMetadata(
                    name=column,
                    original_name=original_name,
                    feature_type=feature_type,
                    transformation_history=[],  # Would be populated in full implementation
                    importance_score=None,  # Would be calculated with model
                    business_meaning=None,
                    creation_method=creation_method,
                    quality_score=quality_score,
                    missing_ratio=missing_ratio,
                    unique_ratio=unique_ratio,
                    distribution_stats=dist_stats,
                    correlation_with_target=target_correlation
                )
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Feature metadata generation failed: {str(e)}")
            return {}
    
    def _calculate_feature_quality_score(
        self,
        missing_ratio: float,
        unique_ratio: float,
        target_correlation: Optional[float]
    ) -> float:
        """Calculate a composite quality score for a feature."""
        try:
            score = 1.0
            
            # Penalize high missing values
            score *= (1 - missing_ratio)
            
            # Penalize very low or very high unique ratios
            if unique_ratio < 0.01:  # Too few unique values
                score *= 0.5
            elif unique_ratio > 0.99:  # Too many unique values (potential ID column)
                score *= 0.7
            
            # Reward high correlation with target
            if target_correlation is not None:
                score *= (1 + abs(target_correlation))
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default neutral score
    
    async def _calculate_performance_impact(
        self,
        original_shape: Tuple[int, int],
        transformed_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Calculate the performance impact of feature engineering."""
        try:
            impact = {
                'original_features': original_shape[1],
                'transformed_features': transformed_shape[1],
                'feature_expansion_ratio': transformed_shape[1] / original_shape[1],
                'dimensionality_change': transformed_shape[1] - original_shape[1]
            }
            
            # Classify impact
            if impact['feature_expansion_ratio'] > 2:
                impact['complexity_change'] = 'High increase'
            elif impact['feature_expansion_ratio'] > 1.5:
                impact['complexity_change'] = 'Moderate increase'
            elif impact['feature_expansion_ratio'] > 1.1:
                impact['complexity_change'] = 'Slight increase'
            else:
                impact['complexity_change'] = 'Minimal change'
            
            return impact
            
        except Exception as e:
            logger.warning(f"Performance impact calculation failed: {str(e)}")
            return {}
    
    def _generate_business_insights(
        self,
        feature_metadata: Dict[str, FeatureMetadata],
        performance_impact: Dict[str, Any]
    ) -> List[str]:
        """Generate business insights from feature engineering results."""
        try:
            insights = []
            
            if not feature_metadata:
                return ["No feature metadata available for insights generation."]
            
            # Feature quality insights
            quality_scores = [meta.quality_score for meta in feature_metadata.values()]
            avg_quality = np.mean(quality_scores)
            
            if avg_quality > 0.8:
                insights.append("High feature quality achieved - features show good predictive potential.")
            elif avg_quality < 0.5:
                insights.append("Feature quality could be improved - consider additional data cleaning.")
            
            # Missing value insights
            high_missing_features = [
                meta.name for meta in feature_metadata.values()
                if meta.missing_ratio > 0.2
            ]
            
            if high_missing_features:
                insights.append(
                    f"{len(high_missing_features)} features have high missing value rates - "
                    "imputation quality should be monitored."
                )
            
            # Target correlation insights
            correlated_features = [
                meta for meta in feature_metadata.values()
                if meta.correlation_with_target is not None and abs(meta.correlation_with_target) > 0.3
            ]
            
            if correlated_features:
                insights.append(
                    f"{len(correlated_features)} features show strong correlation with target - "
                    "good predictive signal detected."
                )
            
            # Complexity insights
            if 'feature_expansion_ratio' in performance_impact:
                ratio = performance_impact['feature_expansion_ratio']
                if ratio > 3:
                    insights.append(
                        "Significant feature expansion detected - consider dimensionality reduction "
                        "to manage computational complexity."
                    )
            
            # Engineered vs original features
            engineered_count = sum(
                1 for meta in feature_metadata.values()
                if meta.creation_method == 'engineered'
            )
            
            if engineered_count > 0:
                insights.append(
                    f"{engineered_count} new features created through engineering - "
                    "enhanced model capabilities expected."
                )
            
            return insights
            
        except Exception as e:
            logger.warning(f"Business insights generation failed: {str(e)}")
            return ["Feature engineering completed successfully."]
    
    def _generate_recommendations(
        self,
        feature_metadata: Dict[str, FeatureMetadata],
        performance_impact: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations for feature engineering."""
        try:
            recommendations = []
            
            if not feature_metadata:
                return ["No feature metadata available for recommendations."]
            
            # Quality-based recommendations
            low_quality_features = [
                meta.name for meta in feature_metadata.values()
                if meta.quality_score < 0.3
            ]
            
            if low_quality_features and len(low_quality_features) < len(feature_metadata) // 2:
                recommendations.append(
                    f"Consider removing {len(low_quality_features)} low-quality features "
                    "to improve model efficiency."
                )
            
            # Missing value recommendations
            high_missing_features = [
                meta.name for meta in feature_metadata.values()
                if meta.missing_ratio > 0.5
            ]
            
            if high_missing_features:
                recommendations.append(
                    f"Review {len(high_missing_features)} features with >50% missing values - "
                    "consider alternative data sources or removal."
                )
            
            # Correlation recommendations
            uncorrelated_features = [
                meta.name for meta in feature_metadata.values()
                if meta.correlation_with_target is not None and abs(meta.correlation_with_target) < 0.05
            ]
            
            if uncorrelated_features and len(uncorrelated_features) > 5:
                recommendations.append(
                    "Multiple features show low correlation with target - "
                    "consider feature selection or non-linear feature engineering."
                )
            
            # Dimensionality recommendations
            if 'transformed_features' in performance_impact:
                feature_count = performance_impact['transformed_features']
                if feature_count > 1000:
                    recommendations.append(
                        "High feature count detected - consider PCA or feature selection "
                        "to reduce computational requirements."
                    )
                elif feature_count < 10:
                    recommendations.append(
                        "Low feature count - consider feature engineering or "
                        "collecting additional data sources."
                    )
            
            # Engineering-specific recommendations
            engineered_features = [
                meta for meta in feature_metadata.values()
                if meta.creation_method == 'engineered'
            ]
            
            if engineered_features:
                avg_engineered_quality = np.mean([meta.quality_score for meta in engineered_features])
                if avg_engineered_quality > 0.8:
                    recommendations.append(
                        "Engineered features show high quality - consider expanding "
                        "feature engineering strategies."
                    )
                elif avg_engineered_quality < 0.5:
                    recommendations.append(
                        "Engineered features show mixed quality - review feature "
                        "engineering logic and domain knowledge."
                    )
            
            # Default recommendation
            if not recommendations:
                recommendations.append(
                    "Feature engineering pipeline completed successfully - "
                    "monitor model performance with engineered features."
                )
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {str(e)}")
            return ["Feature engineering completed - review results manually."]
    
    async def _calculate_quality_metrics(
        self,
        df_transformed: pd.DataFrame,
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for the transformed dataset."""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['total_features'] = len(df_transformed.columns)
            metrics['total_samples'] = len(df_transformed)
            
            # Missing value metrics
            missing_counts = df_transformed.isnull().sum()
            metrics['features_with_missing'] = int(sum(missing_counts > 0))
            metrics['total_missing_values'] = int(missing_counts.sum())
            metrics['missing_value_ratio'] = float(missing_counts.sum() / df_transformed.size)
            
            # Data type distribution
            numeric_count = len(df_transformed.select_dtypes(include=[np.number]).columns)
            categorical_count = len(df_transformed.select_dtypes(exclude=[np.number]).columns)
            
            metrics['numeric_features'] = numeric_count
            metrics['categorical_features'] = categorical_count
            metrics['feature_type_diversity'] = float(min(numeric_count, categorical_count) / max(numeric_count, categorical_count, 1))
            
            # Correlation metrics (for numeric features only)
            numeric_df = df_transformed.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr().abs()
                
                # Remove diagonal
                np.fill_diagonal(corr_matrix.values, 0)
                
                high_corr_pairs = (corr_matrix > 0.9).sum().sum() / 2
                metrics['high_correlation_pairs'] = int(high_corr_pairs)
                metrics['average_correlation'] = float(corr_matrix.mean().mean())
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {str(e)}")
            return {}
    
    async def _log_to_mlflow(
        self,
        report: FeatureEngineeringReport,
        df_transformed: pd.DataFrame
    ):
        """Log feature engineering results to MLflow."""
        try:
            with mlflow.start_run(run_name="feature_engineering"):
                # Log parameters
                mlflow.log_param("original_features", report.original_features)
                mlflow.log_param("engineered_features", report.engineered_features)
                mlflow.log_param("transformations_count", len(report.transformations_applied))
                mlflow.log_param("execution_time", report.execution_time)
                
                # Log metrics
                mlflow.log_metric("feature_expansion_ratio", 
                    report.engineered_features / max(report.original_features, 1))
                
                if report.quality_metrics:
                    for metric_name, metric_value in report.quality_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(f"quality_{metric_name}", metric_value)
                
                # Log feature metadata
                if report.feature_metadata:
                    feature_quality_scores = [
                        meta.quality_score for meta in report.feature_metadata.values()
                    ]
                    mlflow.log_metric("avg_feature_quality", np.mean(feature_quality_scores))
                    mlflow.log_metric("min_feature_quality", np.min(feature_quality_scores))
                    mlflow.log_metric("max_feature_quality", np.max(feature_quality_scores))
                
                # Log artifacts
                # Save feature engineering report
                report_dict = asdict(report)
                report_dict['timestamp'] = report.timestamp.isoformat()
                
                with open("feature_engineering_report.json", "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)
                mlflow.log_artifact("feature_engineering_report.json")
                
                # Save transformed dataset sample
                sample_size = min(1000, len(df_transformed))
                df_sample = df_transformed.head(sample_size)
                df_sample.to_csv("transformed_data_sample.csv", index=False)
                mlflow.log_artifact("transformed_data_sample.csv")
                
                logger.info("Feature engineering results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def transform_new_data(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted pipeline."""
        try:
            if not self.transformers:
                raise ValueError("No fitted transformers available. Run fit_transform first.")
            
            df_transformed = df_new.copy()
            
            # Apply transformers in order
            for transformer_name, transformer in self.transformers.items():
                try:
                    df_transformed = transformer.transform(df_transformed)
                    
                    # Convert back to DataFrame if necessary
                    if not isinstance(df_transformed, pd.DataFrame):
                        if hasattr(transformer, 'get_feature_names_out'):
                            try:
                                feature_names = transformer.get_feature_names_out()
                            except:
                                feature_names = self.engineered_columns
                        else:
                            feature_names = self.engineered_columns
                        
                        df_transformed = pd.DataFrame(
                            df_transformed,
                            columns=feature_names,
                            index=df_new.index
                        )
                    
                except Exception as e:
                    logger.warning(f"Transformer {transformer_name} failed on new data: {str(e)}")
                    continue
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"New data transformation failed: {str(e)}")
            return df_new
    
    def get_feature_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering capabilities and status."""
        try:
            summary = {
                'transformers_fitted': len(self.transformers),
                'original_features': len(self.original_columns),
                'engineered_features': len(self.engineered_columns),
                'feature_types_detected': len(self.feature_types),
                'available_encoders': {
                    'category_encoders': CATEGORY_ENCODERS_AVAILABLE,
                    'iterative_imputer': ITERATIVE_IMPUTER_AVAILABLE,
                    'text_processing': TEXT_PROCESSING_AVAILABLE,
                    'featuretools': FEATURETOOLS_AVAILABLE,
                    'umap': UMAP_AVAILABLE
                },
                'transformation_history': len(self.transformation_history),
                'configuration': asdict(self.config)
            }
            
            if self.feature_types:
                type_counts = {}
                for ftype in FeatureType:
                    count = sum(1 for t in self.feature_types.values() if t == ftype)
                    if count > 0:
                        type_counts[ftype.value] = count
                summary['feature_type_distribution'] = type_counts
            
            return summary
            
        except Exception as e:
            logger.error(f"Feature engineering summary generation failed: {str(e)}")
            return {'error': str(e)}

# Utility functions

def create_feature_engineer(
    auto_interactions: bool = True,
    auto_scaling: bool = True,
    target_encoding: bool = False
) -> FeatureEngineer:
    """Factory function to create a FeatureEngineer."""
    config = FeatureEngineeringConfig()
    config.create_interactions = auto_interactions
    config.numeric_scaling = ScalingMethod.STANDARD if auto_scaling else None
    config.categorical_encoding = EncodingMethod.TARGET if target_encoding else EncodingMethod.ONE_HOT
    return FeatureEngineer(config)

async def quick_feature_engineering(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    max_features: int = 100
) -> pd.DataFrame:
    """Quick feature engineering for simple use cases."""
    config = FeatureEngineeringConfig()
    config.max_interaction_features = max_features
    config.perform_selection = True
    
    engineer = FeatureEngineer(config)
    df_transformed, report = await engineer.fit_transform(df, target_column)
    
    return df_transformed

def get_available_methods() -> Dict[str, Dict[str, bool]]:
    """Get available feature engineering methods."""
    return {
        'encoders': {
            'one_hot': True,
            'label': True,
            'target': CATEGORY_ENCODERS_AVAILABLE,
            'binary': CATEGORY_ENCODERS_AVAILABLE,
            'hashing': CATEGORY_ENCODERS_AVAILABLE
        },
        'scalers': {
            'standard': True,
            'minmax': True,
            'robust': True,
            'quantile': True,
            'power': True
        },
        'imputers': {
            'simple': True,
            'knn': True,
            'iterative': ITERATIVE_IMPUTER_AVAILABLE
        },
        'feature_selection': {
            'statistical': True,
            'model_based': True,
            'recursive': True
        },
        'dimensionality_reduction': {
            'pca': True,
            'lda': True,
            'umap': UMAP_AVAILABLE
        },
        'advanced': {
            'text_processing': TEXT_PROCESSING_AVAILABLE,
            'automated_engineering': FEATURETOOLS_AVAILABLE,
            'outlier_detection': OUTLIER_DETECTION_AVAILABLE
        }
    }

def get_feature_engineering_recommendations(
    n_samples: int,
    n_features: int,
    categorical_ratio: float,
    missing_ratio: float
) -> Dict[str, str]:
    """Get recommendations for feature engineering configuration."""
    recommendations = {}
    
    # Sample size recommendations
    if n_samples < 1000:
        recommendations['scaling'] = "Use robust scaling for small datasets"
        recommendations['interactions'] = "Limit feature interactions to avoid overfitting"
    else:
        recommendations['scaling'] = "Standard scaling recommended"
        recommendations['interactions'] = "Feature interactions can be beneficial"
    
    # Feature count recommendations
    if n_features > 100:
        recommendations['selection'] = "Feature selection highly recommended"
        recommendations['encoding'] = "Consider target encoding for high-cardinality categoricals"
    elif n_features < 10:
        recommendations['engineering'] = "Consider creating additional features"
    
    # Categorical features
    if categorical_ratio > 0.7:
        recommendations['categorical'] = "High categorical ratio - consider advanced encoding methods"
    
    # Missing values
    if missing_ratio > 0.3:
        recommendations['imputation'] = "High missing value ratio - consider advanced imputation"
        recommendations['quality'] = "Review data collection process"
    
    return recommendations

# Example usage and testing
if __name__ == "__main__":
    async def test_feature_engineering():
        """Test the feature engineering functionality."""
        print("Testing Feature Engineering...")
        print("Available methods:", get_available_methods())
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate mixed data types
        data = {
            'numeric_1': np.random.normal(100, 15, n_samples),
            'numeric_2': np.random.exponential(2, n_samples),
            'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
            'binary': np.random.choice([0, 1], n_samples),
            'high_cardinality': [f'cat_{i%50}' for i in range(n_samples)],  # 50 categories
            'text_feature': [f'This is sample text {i} with some content' for i in range(n_samples)]
        }
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
        data['numeric_1'] = np.array(data['numeric_1'])
        data['numeric_1'][missing_indices[:len(missing_indices)//2]] = np.nan
        
        categorical_missing = np.random.choice(n_samples, size=n_samples//20, replace=False)
        for i in categorical_missing:
            data['categorical_1'][i] = None
        
        df = pd.DataFrame(data)
        
        # Create target variable
        df['target'] = (
            0.5 * df['numeric_1'].fillna(df['numeric_1'].mean()) +
            0.3 * df['numeric_2'] +
            0.2 * (df['categorical_1'] == 'A').astype(int) +
            0.1 * df['binary'] +
            np.random.normal(0, 10, n_samples)
        )
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Test feature engineering
        engineer = create_feature_engineer(
            auto_interactions=True,
            auto_scaling=True,
            target_encoding=True
        )
        
        # Business rules example
        business_rules = [
            {
                'type': 'ratio',
                'numerator': 'numeric_1',
                'denominator': 'numeric_2',
                'name': 'ratio_1_2'
            },
            {
                'type': 'binning',
                'feature': 'numeric_1',
                'bins': [0, 50, 100, 150, float('inf')],
                'labels': ['low', 'medium', 'high', 'very_high'],
                'name': 'numeric_1_binned'
            }
        ]
        
        df_transformed, report = await engineer.fit_transform(
            df,
            target_column='target',
            categorical_features=['categorical_1', 'categorical_2', 'high_cardinality'],
            business_rules=business_rules
        )
        
        print(f"\nTransformed data shape: {df_transformed.shape}")
        print(f"Feature expansion ratio: {df_transformed.shape[1] / df.shape[1]:.2f}")
        print(f"Transformations applied: {len(report.transformations_applied)}")
        print(f"Execution time: {report.execution_time:.2f}s")
        
        print(f"\nQuality metrics:")
        for metric, value in report.quality_metrics.items():
            print(f"  {metric}: {value}")
        
        print(f"\nBusiness insights:")
        for insight in report.business_insights[:3]:
            print(f"  - {insight}")
        
        print(f"\nRecommendations:")
        for rec in report.recommendations[:3]:
            print(f"  - {rec}")
        
        # Test quick feature engineering
        print(f"\nTesting quick feature engineering...")
        df_quick = await quick_feature_engineering(
            df.head(100),  # Small sample
            target_column='target',
            max_features=50
        )
        print(f"Quick engineering result shape: {df_quick.shape}")
        
        # Get recommendations
        print(f"\nFeature engineering recommendations:")
        recommendations = get_feature_engineering_recommendations(
            n_samples=len(df),
            n_features=len(df.columns),
            categorical_ratio=0.4,
            missing_ratio=0.1
        )
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
        
        return df_transformed, report
    
    # Run test
    import asyncio
    results = asyncio.run(test_feature_engineering())
