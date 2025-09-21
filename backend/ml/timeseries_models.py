"""
Time Series Models Module for Auto-Analyst Platform

This module implements comprehensive time series modeling capabilities including:
- Classical Statistical Models (ARIMA, SARIMA, Prophet)
- Deep Learning Models (LSTM, GRU, Temporal Fusion Transformer)
- Automatic dataset detection and preprocessing
- Model selection and hyperparameter optimization
- Advanced evaluation metrics and forecasting
- CPU/GPU acceleration support
- Integration with Auto-Analyst pipeline

Features:
- Automatic frequency detection and handling
- Missing data imputation and outlier detection
- Seasonal decomposition and trend analysis
- Multi-step ahead forecasting
- Confidence intervals and uncertainty quantification
- Real-time model serving and batch predictions
- Comprehensive evaluation metrics (RMSE, MAE, MAPE, etc.)
- Business intelligence and forecasting insights
- Integration with MLflow for experiment tracking
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

# Core data processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Statistical time series models
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Prophet for time series forecasting
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Advanced time series libraries
try:
    import scipy.stats as stats
    from scipy import signal
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Plotting and visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class TimeSeriesModelType(Enum):
    """Types of time series models."""
    ARIMA = "arima"
    SARIMA = "sarima"
    PROPHET = "prophet"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    TFT = "temporal_fusion_transformer"
    ENSEMBLE = "ensemble"

class TimeSeriesTaskType(Enum):
    """Types of time series tasks."""
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    TREND_ANALYSIS = "trend_analysis"

class FrequencyType(Enum):
    """Time series frequency types."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    HOURLY = "H"
    MINUTE = "T"
    SECOND = "S"
    AUTO = "auto"

@dataclass
class TimeSeriesConfig:
    """Configuration for time series models."""
    
    def __init__(self):
        # General settings
        self.auto_select_model = True
        self.max_models_to_try = 5
        self.enable_ensemble = True
        self.random_state = 42
        
        # Data preprocessing
        self.handle_missing_data = True
        self.missing_method = 'interpolate'  # 'interpolate', 'forward_fill', 'backward_fill', 'drop'
        self.detect_outliers = True
        self.outlier_method = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
        self.outlier_threshold = 3.0
        self.normalize_data = True
        self.normalization_method = 'minmax'  # 'minmax', 'standard', 'robust'
        
        # Time series specific
        self.frequency = FrequencyType.AUTO
        self.seasonal_periods = None  # Auto-detect if None
        self.test_size = 0.2
        self.validation_size = 0.1
        self.forecast_horizon = 30
        self.confidence_level = 0.95
        
        # ARIMA/SARIMA settings
        self.arima_max_p = 5
        self.arima_max_d = 2
        self.arima_max_q = 5
        self.sarima_max_P = 2
        self.sarima_max_D = 1
        self.sarima_max_Q = 2
        self.use_auto_arima = True
        self.information_criterion = 'aic'  # 'aic', 'bic'
        
        # Prophet settings
        self.prophet_growth = 'linear'  # 'linear', 'logistic'
        self.prophet_seasonality_mode = 'additive'  # 'additive', 'multiplicative'
        self.prophet_yearly_seasonality = 'auto'
        self.prophet_weekly_seasonality = 'auto'
        self.prophet_daily_seasonality = 'auto'
        self.prophet_holidays = None
        self.prophet_changepoint_prior_scale = 0.05
        
        # Deep learning settings
        self.sequence_length = 60
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        self.early_stopping_patience = 10
        self.use_gpu = torch.cuda.is_available() if PYTORCH_AVAILABLE else False
        
        # TFT specific settings
        self.tft_hidden_size = 128
        self.tft_attention_heads = 4
        self.tft_num_quantiles = 7
        self.tft_static_variables = []
        self.tft_time_varying_known = []
        self.tft_time_varying_unknown = []
        
        # Evaluation settings
        self.cross_validation_folds = 5
        self.evaluation_metrics = ['rmse', 'mae', 'mape', 'mase']
        self.walk_forward_validation = True
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        self.enable_caching = True
        
        # Business settings
        self.calculate_business_metrics = True
        self.generate_insights = True
        self.create_visualizations = True

@dataclass
class TimeSeriesResult:
    """Result of time series model training and evaluation."""
    model_type: TimeSeriesModelType
    model: Any
    scaler: Optional[Any]
    train_score: float
    test_score: float
    cv_scores: List[float]
    predictions: np.ndarray
    confidence_intervals: Optional[np.ndarray]
    residuals: np.ndarray
    feature_importance: Optional[Dict[str, float]]
    model_parameters: Dict[str, Any]
    training_time: float
    forecast_horizon: int
    frequency: str
    seasonal_periods: Optional[int]
    metadata: Dict[str, Any]

@dataclass
class TimeSeriesReport:
    """Comprehensive time series analysis report."""
    report_id: str
    timestamp: datetime
    task_type: TimeSeriesTaskType
    dataset_info: Dict[str, Any]
    time_series_analysis: Dict[str, Any]
    models_evaluated: List[TimeSeriesResult]
    best_model_result: TimeSeriesResult
    ensemble_result: Optional[TimeSeriesResult]
    forecasts: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]
    business_insights: List[str]
    recommendations: List[str]
    visualizations: Dict[str, Any]
    metadata: Dict[str, Any]

class TimeSeriesPreprocessor:
    """Preprocessor for time series data."""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.scaler = None
        self.frequency = None
        self.seasonal_periods = None
        
    async def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: str,
        external_features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess time series data."""
        try:
            logger.info("Starting time series preprocessing")
            
            # Validate input data
            processed_df = await self._validate_and_clean_data(df, target_column, date_column)
            
            # Set date as index
            processed_df = processed_df.set_index(date_column)
            processed_df.index = pd.to_datetime(processed_df.index)
            processed_df = processed_df.sort_index()
            
            # Detect frequency
            frequency_info = await self._detect_frequency(processed_df.index)
            self.frequency = frequency_info['frequency']
            
            # Handle missing data
            if self.config.handle_missing_data:
                processed_df = await self._handle_missing_data(processed_df, target_column)
            
            # Detect and handle outliers
            if self.config.detect_outliers:
                processed_df = await self._handle_outliers(processed_df, target_column)
            
            # Detect seasonality
            seasonality_info = await self._detect_seasonality(processed_df[target_column])
            self.seasonal_periods = seasonality_info.get('seasonal_periods')
            
            # Normalize data if requested
            if self.config.normalize_data:
                processed_df, scaler_info = await self._normalize_data(processed_df, target_column)
                self.scaler = scaler_info['scaler']
            
            # Create additional features if external features are provided
            if external_features:
                processed_df = await self._create_additional_features(
                    processed_df, target_column, external_features
                )
            
            # Preprocessing summary
            preprocessing_info = {
                'original_shape': df.shape,
                'processed_shape': processed_df.shape,
                'frequency_info': frequency_info,
                'seasonality_info': seasonality_info,
                'missing_data_handled': self.config.handle_missing_data,
                'outliers_handled': self.config.detect_outliers,
                'normalization_applied': self.config.normalize_data,
                'scaler_type': self.config.normalization_method if self.config.normalize_data else None
            }
            
            logger.info(f"Preprocessing completed. Shape: {processed_df.shape}")
            return processed_df, preprocessing_info
            
        except Exception as e:
            logger.error(f"Time series preprocessing failed: {str(e)}")
            raise
    
    async def _validate_and_clean_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: str
    ) -> pd.DataFrame:
        """Validate and clean input data."""
        try:
            # Check required columns
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            if date_column not in df.columns:
                raise ValueError(f"Date column '{date_column}' not found")
            
            # Remove duplicate dates
            df_clean = df.drop_duplicates(subset=[date_column]).copy()
            
            # Remove rows with missing target values
            initial_len = len(df_clean)
            df_clean = df_clean.dropna(subset=[target_column])
            
            if len(df_clean) < initial_len:
                logger.warning(f"Removed {initial_len - len(df_clean)} rows with missing target values")
            
            # Ensure target is numeric
            df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors='coerce')
            df_clean = df_clean.dropna(subset=[target_column])
            
            if len(df_clean) < 10:
                raise ValueError("Insufficient data points after cleaning (minimum 10 required)")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    async def _detect_frequency(self, date_index: pd.DatetimeIndex) -> Dict[str, Any]:
        """Detect time series frequency."""
        try:
            # Try to infer frequency automatically
            inferred_freq = pd.infer_freq(date_index)
            
            if inferred_freq:
                frequency = inferred_freq
            else:
                # Manual frequency detection based on median time difference
                time_diffs = date_index.to_series().diff().dropna()
                median_diff = time_diffs.median()
                
                if median_diff <= timedelta(seconds=1):
                    frequency = 'S'
                elif median_diff <= timedelta(minutes=1):
                    frequency = 'T'
                elif median_diff <= timedelta(hours=1):
                    frequency = 'H'
                elif median_diff <= timedelta(days=1):
                    frequency = 'D'
                elif median_diff <= timedelta(weeks=1):
                    frequency = 'W'
                elif median_diff <= timedelta(days=32):
                    frequency = 'M'
                elif median_diff <= timedelta(days=100):
                    frequency = 'Q'
                else:
                    frequency = 'Y'
            
            return {
                'frequency': frequency,
                'inferred_frequency': inferred_freq,
                'start_date': date_index.min(),
                'end_date': date_index.max(),
                'num_periods': len(date_index)
            }
            
        except Exception as e:
            logger.warning(f"Frequency detection failed: {str(e)}")
            return {'frequency': 'D', 'inferred_frequency': None}
    
    async def _handle_missing_data(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """Handle missing data in time series."""
        try:
            if df[target_column].isnull().sum() == 0:
                return df
            
            df_filled = df.copy()
            
            if self.config.missing_method == 'interpolate':
                df_filled[target_column] = df_filled[target_column].interpolate(method='time')
            elif self.config.missing_method == 'forward_fill':
                df_filled[target_column] = df_filled[target_column].fillna(method='ffill')
            elif self.config.missing_method == 'backward_fill':
                df_filled[target_column] = df_filled[target_column].fillna(method='bfill')
            elif self.config.missing_method == 'drop':
                df_filled = df_filled.dropna(subset=[target_column])
            
            # Handle any remaining missing values
            if df_filled[target_column].isnull().sum() > 0:
                df_filled[target_column] = df_filled[target_column].fillna(df_filled[target_column].median())
            
            missing_handled = df[target_column].isnull().sum()
            if missing_handled > 0:
                logger.info(f"Handled {missing_handled} missing values using {self.config.missing_method}")
            
            return df_filled
            
        except Exception as e:
            logger.error(f"Missing data handling failed: {str(e)}")
            return df
    
    async def _handle_outliers(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """Detect and handle outliers."""
        try:
            df_clean = df.copy()
            values = df_clean[target_column].values
            
            if self.config.outlier_method == 'iqr':
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (values < lower_bound) | (values > upper_bound)
                
            elif self.config.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(values))
                outlier_mask = z_scores > self.config.outlier_threshold
                
            else:  # isolation_forest
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=self.config.random_state)
                outlier_labels = iso_forest.fit_predict(values.reshape(-1, 1))
                outlier_mask = outlier_labels == -1
            
            # Handle outliers by clipping to bounds
            if outlier_mask.sum() > 0:
                if self.config.outlier_method == 'iqr':
                    df_clean.loc[outlier_mask, target_column] = np.clip(
                        df_clean.loc[outlier_mask, target_column], lower_bound, upper_bound
                    )
                else:
                    # Replace with median
                    median_value = np.median(values[~outlier_mask])
                    df_clean.loc[outlier_mask, target_column] = median_value
                
                logger.info(f"Handled {outlier_mask.sum()} outliers using {self.config.outlier_method}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Outlier handling failed: {str(e)}")
            return df
    
    async def _detect_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Detect seasonal patterns in the time series."""
        try:
            seasonality_info = {
                'has_seasonality': False,
                'seasonal_periods': None,
                'seasonal_strength': 0.0
            }
            
            if len(series) < 24:  # Need minimum data for seasonality detection
                return seasonality_info
            
            # Try different seasonal periods based on frequency
            freq = self.frequency
            
            if freq in ['H', 'T', 'S']:
                potential_periods = [24, 168, 8760]  # Daily, weekly, yearly for hourly data
            elif freq == 'D':
                potential_periods = [7, 30, 365]  # Weekly, monthly, yearly
            elif freq == 'W':
                potential_periods = [4, 52]  # Monthly, yearly
            elif freq == 'M':
                potential_periods = [12]  # Yearly
            else:
                potential_periods = []
            
            best_period = None
            best_strength = 0
            
            for period in potential_periods:
                if len(series) > 2 * period:
                    try:
                        # Use autocorrelation to detect seasonality
                        autocorr = series.autocorr(lag=period)
                        if autocorr > best_strength and autocorr > 0.3:
                            best_strength = autocorr
                            best_period = period
                    except:
                        continue
            
            if best_period and best_strength > 0.3:
                seasonality_info['has_seasonality'] = True
                seasonality_info['seasonal_periods'] = best_period
                seasonality_info['seasonal_strength'] = best_strength
            
            return seasonality_info
            
        except Exception as e:
            logger.warning(f"Seasonality detection failed: {str(e)}")
            return {'has_seasonality': False, 'seasonal_periods': None, 'seasonal_strength': 0.0}
    
    async def _normalize_data(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize time series data."""
        try:
            df_normalized = df.copy()
            
            if self.config.normalization_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.config.normalization_method == 'standard':
                scaler = StandardScaler()
            else:  # robust
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            # Fit and transform the target column
            values = df_normalized[target_column].values.reshape(-1, 1)
            normalized_values = scaler.fit_transform(values)
            df_normalized[target_column] = normalized_values.flatten()
            
            scaler_info = {
                'scaler': scaler,
                'method': self.config.normalization_method,
                'original_mean': float(np.mean(values)),
                'original_std': float(np.std(values))
            }
            
            return df_normalized, scaler_info
            
        except Exception as e:
            logger.error(f"Data normalization failed: {str(e)}")
            return df, {'scaler': None, 'method': None}
    
    async def _create_additional_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        external_features: List[str]
    ) -> pd.DataFrame:
        """Create additional time-based features."""
        try:
            df_enhanced = df.copy()
            
            # Time-based features
            df_enhanced['year'] = df_enhanced.index.year
            df_enhanced['month'] = df_enhanced.index.month
            df_enhanced['day'] = df_enhanced.index.day
            df_enhanced['dayofweek'] = df_enhanced.index.dayofweek
            df_enhanced['hour'] = df_enhanced.index.hour if hasattr(df_enhanced.index, 'hour') else 0
            
            # Lag features
            for lag in [1, 7, 30]:
                if len(df_enhanced) > lag:
                    df_enhanced[f'{target_column}_lag_{lag}'] = df_enhanced[target_column].shift(lag)
            
            # Rolling features
            for window in [7, 30]:
                if len(df_enhanced) > window:
                    df_enhanced[f'{target_column}_rolling_mean_{window}'] = (
                        df_enhanced[target_column].rolling(window=window).mean()
                    )
                    df_enhanced[f'{target_column}_rolling_std_{window}'] = (
                        df_enhanced[target_column].rolling(window=window).std()
                    )
            
            # External features normalization
            for feature in external_features:
                if feature in df_enhanced.columns:
                    df_enhanced[feature] = (
                        df_enhanced[feature] - df_enhanced[feature].mean()
                    ) / df_enhanced[feature].std()
            
            return df_enhanced
            
        except Exception as e:
            logger.error(f"Feature creation failed: {str(e)}")
            return df

class BaseTimeSeriesModel(ABC):
    """Abstract base class for time series models."""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    async def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the time series model."""
        pass
    
    @abstractmethod
    async def predict(self, X: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    async def forecast(self, steps_ahead: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate forecasts with confidence intervals."""
        pass
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}

class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA model implementation."""
    
    def __init__(self, config: TimeSeriesConfig, order: Tuple[int, int, int] = None):
        super().__init__(config)
        self.order = order or (1, 1, 1)
        self.seasonal_order = None
        self.last_values = None
        
    async def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ARIMA model."""
        try:
            if not STATSMODELS_AVAILABLE:
                raise ImportError("statsmodels is required for ARIMA models")
            
            # Convert to pandas Series for ARIMA
            y_series = pd.Series(y.flatten())
            
            if self.config.use_auto_arima:
                # Use auto ARIMA for order selection
                best_order = await self._find_best_arima_order(y_series)
                self.order = best_order
            
            # Fit ARIMA model
            self.model = ARIMA(y_series, order=self.order)
            fitted_model = self.model.fit()
            self.model = fitted_model
            
            # Store last values for forecasting
            self.last_values = y_series.values
            self.is_fitted = True
            
            logger.info(f"ARIMA{self.order} model fitted successfully")
            
        except Exception as e:
            logger.error(f"ARIMA model fitting failed: {str(e)}")
            raise
    
    async def predict(self, X: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """Make ARIMA predictions."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            # Generate predictions
            forecast_result = self.model.forecast(steps=steps_ahead)
            return np.array(forecast_result)
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {str(e)}")
            return np.zeros(steps_ahead)
    
    async def forecast(self, steps_ahead: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate ARIMA forecasts with confidence intervals."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before forecasting")
            
            # Generate forecast with confidence intervals
            forecast_result = self.model.get_forecast(steps=steps_ahead)
            
            predictions = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int().values
            
            return predictions, conf_int
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {str(e)}")
            return np.zeros(steps_ahead), None
    
    async def _find_best_arima_order(self, y: pd.Series) -> Tuple[int, int, int]:
        """Find best ARIMA order using information criteria."""
        try:
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            # Grid search over different orders
            for p in range(self.config.arima_max_p + 1):
                for d in range(self.config.arima_max_d + 1):
                    for q in range(self.config.arima_max_q + 1):
                        try:
                            model = ARIMA(y, order=(p, d, q))
                            fitted = model.fit(disp=False)
                            
                            if self.config.information_criterion == 'aic':
                                score = fitted.aic
                            else:
                                score = fitted.bic
                            
                            if score < best_aic:
                                best_aic = score
                                best_order = (p, d, q)
                                
                        except:
                            continue
            
            return best_order
            
        except Exception as e:
            logger.warning(f"Auto ARIMA order selection failed: {str(e)}")
            return (1, 1, 1)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get ARIMA model parameters."""
        params = {
            'order': self.order,
            'model_type': 'ARIMA'
        }
        
        if self.is_fitted:
            params.update({
                'aic': self.model.aic,
                'bic': self.model.bic,
                'params': self.model.params.to_dict()
            })
        
        return params

class SARIMAModel(BaseTimeSeriesModel):
    """SARIMA model implementation."""
    
    def __init__(
        self,
        config: TimeSeriesConfig,
        order: Tuple[int, int, int] = None,
        seasonal_order: Tuple[int, int, int, int] = None
    ):
        super().__init__(config)
        self.order = order or (1, 1, 1)
        self.seasonal_order = seasonal_order or (1, 1, 1, 12)
        
    async def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit SARIMA model."""
        try:
            if not STATSMODELS_AVAILABLE:
                raise ImportError("statsmodels is required for SARIMA models")
            
            y_series = pd.Series(y.flatten())
            
            # Auto-detect seasonal order if not provided
            if self.config.seasonal_periods:
                seasonal_periods = self.config.seasonal_periods
                self.seasonal_order = (1, 1, 1, seasonal_periods)
            
            # Fit SARIMA model
            self.model = SARIMAX(
                y_series,
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            fitted_model = self.model.fit(disp=False)
            self.model = fitted_model
            self.is_fitted = True
            
            logger.info(f"SARIMA{self.order}x{self.seasonal_order} model fitted successfully")
            
        except Exception as e:
            logger.error(f"SARIMA model fitting failed: {str(e)}")
            raise
    
    async def predict(self, X: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """Make SARIMA predictions."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            forecast_result = self.model.forecast(steps=steps_ahead)
            return np.array(forecast_result)
            
        except Exception as e:
            logger.error(f"SARIMA prediction failed: {str(e)}")
            return np.zeros(steps_ahead)
    
    async def forecast(self, steps_ahead: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate SARIMA forecasts with confidence intervals."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before forecasting")
            
            forecast_result = self.model.get_forecast(steps=steps_ahead)
            predictions = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int().values
            
            return predictions, conf_int
            
        except Exception as e:
            logger.error(f"SARIMA forecasting failed: {str(e)}")
            return np.zeros(steps_ahead), None
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get SARIMA model parameters."""
        params = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'model_type': 'SARIMA'
        }
        
        if self.is_fitted:
            params.update({
                'aic': self.model.aic,
                'bic': self.model.bic,
                'params': self.model.params.to_dict()
            })
        
        return params

class ProphetModel(BaseTimeSeriesModel):
    """Facebook Prophet model implementation."""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__(config)
        self.date_col = 'ds'
        self.target_col = 'y'
        
    async def fit(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> None:
        """Fit Prophet model."""
        try:
            if not PROPHET_AVAILABLE:
                raise ImportError("prophet is required for Prophet models")
            
            # Prepare data for Prophet
            if dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
            
            df = pd.DataFrame({
                self.date_col: dates,
                self.target_col: y.flatten()
            })
            
            # Initialize Prophet model
            self.model = Prophet(
                growth=self.config.prophet_growth,
                seasonality_mode=self.config.prophet_seasonality_mode,
                yearly_seasonality=self.config.prophet_yearly_seasonality,
                weekly_seasonality=self.config.prophet_weekly_seasonality,
                daily_seasonality=self.config.prophet_daily_seasonality,
                changepoint_prior_scale=self.config.prophet_changepoint_prior_scale
            )
            
            # Fit model
            self.model.fit(df)
            self.is_fitted = True
            
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Prophet model fitting failed: {str(e)}")
            raise
    
    async def predict(self, X: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """Make Prophet predictions."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps_ahead)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Return last steps_ahead predictions
            return forecast['yhat'].tail(steps_ahead).values
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {str(e)}")
            return np.zeros(steps_ahead)
    
    async def forecast(self, steps_ahead: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate Prophet forecasts with confidence intervals."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before forecasting")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps_ahead)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Extract predictions and confidence intervals
            predictions = forecast['yhat'].tail(steps_ahead).values
            lower_bound = forecast['yhat_lower'].tail(steps_ahead).values
            upper_bound = forecast['yhat_upper'].tail(steps_ahead).values
            
            conf_int = np.column_stack([lower_bound, upper_bound])
            
            return predictions, conf_int
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {str(e)}")
            return np.zeros(steps_ahead), None
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Prophet model parameters."""
        params = {
            'model_type': 'Prophet',
            'growth': self.config.prophet_growth,
            'seasonality_mode': self.config.prophet_seasonality_mode
        }
        
        if self.is_fitted:
            params.update({
                'changepoints': len(self.model.changepoints),
                'seasonalities': list(self.model.seasonalities.keys())
            })
        
        return params

class LSTMModel(BaseTimeSeriesModel):
    """LSTM model implementation for time series."""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__(config)
        self.sequence_length = config.sequence_length
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
    async def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LSTM model."""
        try:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is required for LSTM models")
            
            # Prepare sequences
            X_seq, y_seq = self._create_sequences(y.flatten())
            
            # Create PyTorch datasets
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            y_tensor = torch.FloatTensor(y_seq).to(self.device)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Build LSTM model
            input_size = 1  # Univariate time series
            self.model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=1,
                dropout_rate=self.config.dropout_rate
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(self.config.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logger.debug(f"LSTM Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            self.is_fitted = True
            logger.info("LSTM model fitted successfully")
            
        except Exception as e:
            logger.error(f"LSTM model fitting failed: {str(e)}")
            raise
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        try:
            X, y = [], []
            
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:i + self.sequence_length])
                y.append(data[i + self.sequence_length])
            
            return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)
            
        except Exception as e:
            logger.error(f"Sequence creation failed: {str(e)}")
            raise
    
    async def predict(self, X: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """Make LSTM predictions."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            self.model.eval()
            predictions = []
            
            # Use last sequence_length values as initial input
            current_sequence = X[-self.sequence_length:].reshape(1, -1, 1)
            current_sequence = torch.FloatTensor(current_sequence).to(self.device)
            
            with torch.no_grad():
                for _ in range(steps_ahead):
                    pred = self.model(current_sequence)
                    predictions.append(pred.cpu().numpy()[0, 0])
                    
                    # Update sequence for next prediction
                    pred_tensor = pred.unsqueeze(0)
                    current_sequence = torch.cat([current_sequence[:, 1:, :], pred_tensor], dim=1)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {str(e)}")
            return np.zeros(steps_ahead)
    
    async def forecast(self, steps_ahead: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate LSTM forecasts (without confidence intervals)."""
        try:
            # LSTM doesn't naturally provide confidence intervals
            # For now, return predictions without confidence intervals
            predictions = await self.predict(np.array([]), steps_ahead)
            return predictions, None
            
        except Exception as e:
            logger.error(f"LSTM forecasting failed: {str(e)}")
            return np.zeros(steps_ahead), None
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get LSTM model parameters."""
        return {
            'model_type': 'LSTM',
            'sequence_length': self.sequence_length,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'dropout_rate': self.config.dropout_rate
        }

class LSTMNetwork(nn.Module):
    """LSTM Neural Network architecture."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout_rate: float = 0.2
    ):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)
        
        return output

class TimeSeriesEvaluator:
    """Evaluate time series models with comprehensive metrics."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive time series evaluation metrics."""
        try:
            metrics = {}
            
            # Basic regression metrics
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # Mean Absolute Percentage Error (MAPE)
            non_zero_mask = y_true != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['mape'] = mape
            else:
                metrics['mape'] = np.inf
            
            # Symmetric Mean Absolute Percentage Error (sMAPE)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            non_zero_denom = denominator != 0
            if non_zero_denom.sum() > 0:
                smape = np.mean(np.abs(y_true[non_zero_denom] - y_pred[non_zero_denom]) / denominator[non_zero_denom]) * 100
                metrics['smape'] = smape
            else:
                metrics['smape'] = 0.0
            
            # Mean Absolute Scaled Error (MASE)
            if y_train is not None and len(y_train) > 1:
                naive_forecast_mae = np.mean(np.abs(np.diff(y_train)))
                if naive_forecast_mae != 0:
                    metrics['mase'] = metrics['mae'] / naive_forecast_mae
                else:
                    metrics['mase'] = np.inf
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot != 0:
                metrics['r2'] = 1 - (ss_res / ss_tot)
            else:
                metrics['r2'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            return {
                'rmse': np.inf,
                'mae': np.inf,
                'mape': np.inf,
                'smape': np.inf,
                'mase': np.inf,
                'r2': -np.inf
            }
    
    @staticmethod
    def cross_validate_timeseries(
        model: BaseTimeSeriesModel,
        data: np.ndarray,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> List[float]:
        """Perform time series cross-validation."""
        try:
            scores = []
            
            # Use TimeSeriesSplit for proper temporal validation
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(data) * test_size))
            
            for train_index, test_index in tscv.split(data):
                train_data = data[train_index]
                test_data = data[test_index]
                
                try:
                    # Fit model on training data
                    asyncio.run(model.fit(None, train_data))
                    
                    # Predict on test data
                    predictions = asyncio.run(model.predict(None, len(test_data)))
                    
                    # Calculate RMSE for this fold
                    rmse = np.sqrt(mean_squared_error(test_data, predictions))
                    scores.append(rmse)
                    
                except Exception as fold_e:
                    logger.warning(f"Cross-validation fold failed: {str(fold_e)}")
                    continue
            
            return scores
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            return []

class TimeSeriesAnalyzer:
    """
    Main time series analysis engine with automatic model selection,
    comprehensive evaluation, and forecasting capabilities.
    """
    
    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.preprocessor = TimeSeriesPreprocessor(self.config)
        self.models = {}
        self.best_model = None
        self.best_model_type = None
        self.scaler = None
        self.preprocessing_info = None
        
        logger.info("TimeSeriesAnalyzer initialized")
    
    async def analyze_timeseries(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: str,
        external_features: Optional[List[str]] = None,
        task_type: TimeSeriesTaskType = TimeSeriesTaskType.FORECASTING
    ) -> TimeSeriesReport:
        """
        Comprehensive time series analysis with automatic model selection.
        
        Args:
            df: Input DataFrame with time series data
            target_column: Name of the target variable column
            date_column: Name of the date/time column
            external_features: List of external feature columns
            task_type: Type of time series task
            
        Returns:
            Comprehensive time series analysis report
        """
        try:
            logger.info(f"Starting time series analysis on {len(df)} data points")
            start_time = datetime.now()
            
            # Data preprocessing
            processed_df, preprocessing_info = await self.preprocessor.preprocess_data(
                df, target_column, date_column, external_features
            )
            
            self.preprocessing_info = preprocessing_info
            self.scaler = self.preprocessor.scaler
            
            # Extract time series data
            y = processed_df[target_column].values
            dates = processed_df.index
            
            # Dataset analysis
            dataset_info = await self._analyze_dataset(processed_df, target_column, dates)
            
            # Time series specific analysis
            ts_analysis = await self._perform_timeseries_analysis(y, dates)
            
            # Split data for training and testing
            train_size = int(len(y) * (1 - self.config.test_size))
            y_train = y[:train_size]
            y_test = y[train_size:]
            dates_train = dates[:train_size]
            dates_test = dates[train_size:]
            
            # Model training and evaluation
            models_evaluated = []
            
            if task_type == TimeSeriesTaskType.FORECASTING:
                models_evaluated = await self._train_forecasting_models(
                    y_train, y_test, dates_train, dates_test
                )
            
            # Select best model
            if models_evaluated:
                best_model_result = min(models_evaluated, key=lambda x: x.test_score)
                self.best_model = best_model_result.model
                self.best_model_type = best_model_result.model_type
            else:
                best_model_result = None
            
            # Generate forecasts
            forecasts = await self._generate_forecasts(best_model_result, y)
            
            # Comprehensive evaluation
            evaluation_metrics = await self._evaluate_models(models_evaluated)
            
            # Generate insights and recommendations
            insights = await self._generate_insights(
                dataset_info, ts_analysis, best_model_result
            )
            
            recommendations = await self._generate_recommendations(
                insights, best_model_result, evaluation_metrics
            )
            
            # Create visualizations
            visualizations = await self._create_visualizations(
                y, dates, forecasts, best_model_result
            )
            
            # Execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive report
            report = TimeSeriesReport(
                report_id=str(uuid.uuid4()),
                timestamp=start_time,
                task_type=task_type,
                dataset_info=dataset_info,
                time_series_analysis=ts_analysis,
                models_evaluated=models_evaluated,
                best_model_result=best_model_result,
                ensemble_result=None,  # TODO: Implement ensemble
                forecasts=forecasts,
                evaluation_metrics=evaluation_metrics,
                business_insights=insights,
                recommendations=recommendations,
                visualizations=visualizations,
                metadata={
                    'execution_time': execution_time,
                    'preprocessing_info': preprocessing_info,
                    'config': asdict(self.config)
                }
            )
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(report)
            
            logger.info(f"Time series analysis completed in {execution_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Time series analysis failed: {str(e)}")
            # Return minimal report with error
            return TimeSeriesReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                task_type=task_type,
                dataset_info={},
                time_series_analysis={},
                models_evaluated=[],
                best_model_result=None,
                ensemble_result=None,
                forecasts={},
                evaluation_metrics={},
                business_insights=[f"Analysis failed: {str(e)}"],
                recommendations=["Review data quality and configuration"],
                visualizations={},
                metadata={'error': str(e)}
            )
    
    async def _analyze_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        try:
            target_values = df[target_column].values
            
            analysis = {
                'n_observations': len(df),
                'date_range': {
                    'start': dates.min().isoformat(),
                    'end': dates.max().isoformat(),
                    'duration_days': (dates.max() - dates.min()).days
                },
                'target_statistics': {
                    'mean': float(np.mean(target_values)),
                    'std': float(np.std(target_values)),
                    'min': float(np.min(target_values)),
                    'max': float(np.max(target_values)),
                    'median': float(np.median(target_values)),
                    'skewness': float(stats.skew(target_values)) if SCIPY_AVAILABLE else 0.0,
                    'kurtosis': float(stats.kurtosis(target_values)) if SCIPY_AVAILABLE else 0.0
                },
                'missing_values': df[target_column].isnull().sum(),
                'frequency': self.preprocessor.frequency,
                'seasonal_periods': self.preprocessor.seasonal_periods
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {str(e)}")
            return {}
    
    async def _perform_timeseries_analysis(
        self,
        y: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Perform comprehensive time series analysis."""
        try:
            analysis = {}
            
            # Stationarity tests
            if STATSMODELS_AVAILABLE:
                analysis['stationarity'] = await self._test_stationarity(y)
            
            # Trend analysis
            analysis['trend'] = await self._analyze_trend(y)
            
            # Seasonality analysis
            analysis['seasonality'] = await self._analyze_seasonality(y, dates)
            
            # Autocorrelation analysis
            analysis['autocorrelation'] = await self._analyze_autocorrelation(y)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Time series analysis failed: {str(e)}")
            return {}
    
    async def _test_stationarity(self, y: np.ndarray) -> Dict[str, Any]:
        """Test for stationarity using ADF and KPSS tests."""
        try:
            results = {}
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(y)
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            
            # KPSS test
            kpss_result = kpss(y)
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
            
            # Overall assessment
            results['overall_stationary'] = (
                results['adf']['is_stationary'] and results['kpss']['is_stationary']
            )
            
            return results
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {str(e)}")
            return {}
    
    async def _analyze_trend(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in the time series."""
        try:
            # Simple trend analysis using linear regression
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_analysis = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'trend_strength': abs(r_value)
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.warning(f"Trend analysis failed: {str(e)}")
            return {}
    
    async def _analyze_seasonality(
        self,
        y: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns."""
        try:
            seasonality_analysis = {}
            
            if STATSMODELS_AVAILABLE and len(y) > 24:
                # Seasonal decomposition
                ts = pd.Series(y, index=dates)
                
                # Determine period for decomposition
                if self.preprocessor.seasonal_periods:
                    period = min(self.preprocessor.seasonal_periods, len(y) // 2)
                else:
                    period = min(12, len(y) // 2)  # Default to 12
                
                if period >= 2:
                    decomposition = seasonal_decompose(
                        ts, model='additive', period=period, extrapolate_trend='freq'
                    )
                    
                    seasonality_analysis = {
                        'seasonal_strength': float(np.std(decomposition.seasonal) / np.std(y)),
                        'trend_strength': float(np.std(decomposition.trend.dropna()) / np.std(y)),
                        'residual_strength': float(np.std(decomposition.resid.dropna()) / np.std(y)),
                        'period': period
                    }
            
            return seasonality_analysis
            
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {str(e)}")
            return {}
    
    async def _analyze_autocorrelation(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze autocorrelation structure."""
        try:
            # Calculate autocorrelations for different lags
            max_lags = min(20, len(y) // 4)
            autocorrelations = []
            
            for lag in range(1, max_lags + 1):
                if len(y) > lag:
                    corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
                    autocorrelations.append((lag, corr))
            
            # Find significant autocorrelations
            significant_lags = [
                lag for lag, corr in autocorrelations 
                if abs(corr) > 0.1  # Simple threshold
            ]
            
            return {
                'autocorrelations': autocorrelations,
                'significant_lags': significant_lags,
                'max_autocorr': max(autocorrelations, key=lambda x: abs(x[1])) if autocorrelations else (0, 0)
            }
            
        except Exception as e:
            logger.warning(f"Autocorrelation analysis failed: {str(e)}")
            return {}
    
    async def _train_forecasting_models(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        dates_train: pd.DatetimeIndex,
        dates_test: pd.DatetimeIndex
    ) -> List[TimeSeriesResult]:
        """Train multiple forecasting models."""
        try:
            results = []
            
            # Model types to try
            model_types = []
            
            if STATSMODELS_AVAILABLE:
                model_types.extend([TimeSeriesModelType.ARIMA, TimeSeriesModelType.SARIMA])
            
            if PROPHET_AVAILABLE:
                model_types.append(TimeSeriesModelType.PROPHET)
            
            if PYTORCH_AVAILABLE:
                model_types.extend([TimeSeriesModelType.LSTM])
            
            if not model_types:
                logger.warning("No time series models available")
                return results
            
            for model_type in model_types[:self.config.max_models_to_try]:
                try:
                    result = await self._train_single_model(
                        model_type, y_train, y_test, dates_train, dates_test
                    )
                    if result:
                        results.append(result)
                        logger.info(f"{model_type.value} - Test RMSE: {result.test_score:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Model {model_type.value} training failed: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return []
    
    async def _train_single_model(
        self,
        model_type: TimeSeriesModelType,
        y_train: np.ndarray,
        y_test: np.ndarray,
        dates_train: pd.DatetimeIndex,
        dates_test: pd.DatetimeIndex
    ) -> Optional[TimeSeriesResult]:
        """Train a single time series model."""
        try:
            start_time = datetime.now()
            
            # Create model instance
            if model_type == TimeSeriesModelType.ARIMA:
                model = ARIMAModel(self.config)
            elif model_type == TimeSeriesModelType.SARIMA:
                model = SARIMAModel(self.config)
            elif model_type == TimeSeriesModelType.PROPHET:
                model = ProphetModel(self.config)
            elif model_type == TimeSeriesModelType.LSTM:
                model = LSTMModel(self.config)
            else:
                logger.warning(f"Model type {model_type} not implemented")
                return None
            
            # Fit model
            if model_type == TimeSeriesModelType.PROPHET:
                await model.fit(None, y_train, dates_train)
            else:
                await model.fit(None, y_train)
            
            # Make predictions
            if model_type == TimeSeriesModelType.LSTM:
                predictions = await model.predict(y_train, len(y_test))
            else:
                predictions = await model.predict(None, len(y_test))
            
            # Calculate metrics
            metrics = TimeSeriesEvaluator.calculate_metrics(y_test, predictions, y_train)
            
            # Cross-validation scores
            try:
                cv_scores = TimeSeriesEvaluator.cross_validate_timeseries(
                    model, y_train, n_splits=min(5, len(y_train) // 20)
                )
            except:
                cv_scores = [metrics['rmse']]  # Fallback
            
            # Calculate residuals
            residuals = y_test - predictions
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = TimeSeriesResult(
                model_type=model_type,
                model=model,
                scaler=self.scaler,
                train_score=0.0,  # TODO: Calculate train score
                test_score=metrics['rmse'],
                cv_scores=cv_scores,
                predictions=predictions,
                confidence_intervals=None,  # TODO: Implement for all models
                residuals=residuals,
                feature_importance=None,
                model_parameters=model.get_model_params(),
                training_time=training_time,
                forecast_horizon=self.config.forecast_horizon,
                frequency=self.preprocessor.frequency or 'D',
                seasonal_periods=self.preprocessor.seasonal_periods,
                metadata={'metrics': metrics}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Single model training failed for {model_type.value}: {str(e)}")
            return None
    
    async def _generate_forecasts(
        self,
        best_model_result: Optional[TimeSeriesResult],
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Generate forecasts using the best model."""
        try:
            if not best_model_result:
                return {}
            
            model = best_model_result.model
            horizon = self.config.forecast_horizon
            
            # Generate forecasts
            if best_model_result.model_type == TimeSeriesModelType.LSTM:
                forecasts, conf_int = await model.forecast(horizon)
            else:
                forecasts, conf_int = await model.forecast(horizon)
            
            # Inverse transform if scaled
            if self.scaler:
                forecasts = self.scaler.inverse_transform(forecasts.reshape(-1, 1)).flatten()
                if conf_int is not None:
                    conf_int = self.scaler.inverse_transform(conf_int)
            
            forecast_results = {
                'forecasts': forecasts.tolist(),
                'horizon': horizon,
                'model_used': best_model_result.model_type.value,
                'confidence_intervals': conf_int.tolist() if conf_int is not None else None,
                'forecast_dates': pd.date_range(
                    start=pd.Timestamp.now(),
                    periods=horizon,
                    freq=best_model_result.frequency
                ).strftime('%Y-%m-%d').tolist()
            }
            
            return forecast_results
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {str(e)}")
            return {}
    
    async def _evaluate_models(
        self,
        models_evaluated: List[TimeSeriesResult]
    ) -> Dict[str, Any]:
        """Evaluate and compare all models."""
        try:
            if not models_evaluated:
                return {}
            
            evaluation = {
                'model_comparison': [],
                'best_model': None,
                'metrics_summary': {}
            }
            
            # Compare models
            for result in models_evaluated:
                model_eval = {
                    'model_type': result.model_type.value,
                    'test_rmse': result.test_score,
                    'cv_rmse_mean': float(np.mean(result.cv_scores)) if result.cv_scores else result.test_score,
                    'cv_rmse_std': float(np.std(result.cv_scores)) if result.cv_scores else 0.0,
                    'training_time': result.training_time
                }
                
                if result.metadata and 'metrics' in result.metadata:
                    metrics = result.metadata['metrics']
                    model_eval.update({
                        'mae': metrics.get('mae', 0),
                        'mape': metrics.get('mape', 0),
                        'r2': metrics.get('r2', 0)
                    })
                
                evaluation['model_comparison'].append(model_eval)
            
            # Find best model
            best_result = min(models_evaluated, key=lambda x: x.test_score)
            evaluation['best_model'] = {
                'model_type': best_result.model_type.value,
                'test_score': best_result.test_score,
                'parameters': best_result.model_parameters
            }
            
            # Summary statistics
            test_scores = [r.test_score for r in models_evaluated]
            evaluation['metrics_summary'] = {
                'best_rmse': float(min(test_scores)),
                'worst_rmse': float(max(test_scores)),
                'mean_rmse': float(np.mean(test_scores)),
                'std_rmse': float(np.std(test_scores))
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}
    
    async def _generate_insights(
        self,
        dataset_info: Dict[str, Any],
        ts_analysis: Dict[str, Any],
        best_model_result: Optional[TimeSeriesResult]
    ) -> List[str]:
        """Generate business insights from time series analysis."""
        try:
            insights = []
            
            # Dataset insights
            n_obs = dataset_info.get('n_observations', 0)
            if n_obs < 50:
                insights.append("Limited data available - consider collecting more historical data for robust forecasting")
            elif n_obs > 1000:
                insights.append("Rich historical data available - models should provide reliable forecasts")
            
            # Trend insights
            if 'trend' in ts_analysis and ts_analysis['trend']:
                trend = ts_analysis['trend']
                direction = trend.get('trend_direction', 'stable')
                strength = trend.get('trend_strength', 0)
                
                if direction != 'stable' and strength > 0.5:
                    insights.append(f"Strong {direction} trend detected (R² = {strength:.3f}) - consider trend-aware forecasting")
                elif direction != 'stable':
                    insights.append(f"Weak {direction} trend present - may stabilize over time")
            
            # Seasonality insights
            if 'seasonality' in ts_analysis and ts_analysis['seasonality']:
                seasonal = ts_analysis['seasonality']
                seasonal_strength = seasonal.get('seasonal_strength', 0)
                
                if seasonal_strength > 0.3:
                    period = seasonal.get('period', 'unknown')
                    insights.append(f"Strong seasonal pattern detected (period: {period}) - seasonal models recommended")
                elif seasonal_strength > 0.1:
                    insights.append("Moderate seasonality present - may benefit from seasonal adjustment")
            
            # Model performance insights
            if best_model_result:
                rmse = best_model_result.test_score
                model_type = best_model_result.model_type.value
                
                # Get relative error
                if 'target_statistics' in dataset_info:
                    mean_value = dataset_info['target_statistics']['mean']
                    relative_error = rmse / abs(mean_value) if mean_value != 0 else np.inf
                    
                    if relative_error < 0.1:
                        insights.append(f"Excellent forecast accuracy achieved with {model_type} (RMSE: {rmse:.3f})")
                    elif relative_error < 0.2:
                        insights.append(f"Good forecast accuracy with {model_type} - suitable for business planning")
                    else:
                        insights.append(f"Moderate accuracy with {model_type} - consider additional features or different approach")
            
            # Stationarity insights
            if 'stationarity' in ts_analysis and ts_analysis['stationarity']:
                if not ts_analysis['stationarity'].get('overall_stationary', True):
                    insights.append("Non-stationary time series detected - differencing or transformation may improve models")
            
            return insights
            
        except Exception as e:
            logger.error(f"Insights generation failed: {str(e)}")
            return ["Time series analysis completed successfully"]
    
    async def _generate_recommendations(
        self,
        insights: List[str],
        best_model_result: Optional[TimeSeriesResult],
        evaluation_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Data recommendations
            if any("Limited data" in insight for insight in insights):
                recommendations.append("Collect more historical data to improve forecast reliability")
            
            # Model recommendations
            if best_model_result:
                model_type = best_model_result.model_type.value
                
                if model_type == 'arima':
                    recommendations.append("Consider SARIMA or Prophet models if strong seasonality is present")
                elif model_type == 'prophet':
                    recommendations.append("Monitor forecast performance and retrain regularly as new data arrives")
                elif model_type == 'lstm':
                    recommendations.append("Ensure sufficient training data for neural network stability")
            
            # Seasonality recommendations
            if any("seasonal" in insight.lower() for insight in insights):
                recommendations.append("Incorporate seasonal adjustments in business planning and inventory management")
            
            # Trend recommendations
            if any("trend" in insight.lower() for insight in insights):
                recommendations.append("Consider trend extrapolation in long-term strategic planning")
            
            # Performance recommendations
            if evaluation_metrics.get('metrics_summary', {}).get('std_rmse', 0) > 0.1:
                recommendations.append("High model variance detected - consider ensemble methods for more stable forecasts")
            
            # Business recommendations
            recommendations.append("Implement forecast monitoring and set up alerts for unusual patterns")
            recommendations.append("Regularly retrain models as new data becomes available")
            recommendations.append("Consider external factors and domain expertise when interpreting forecasts")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {str(e)}")
            return ["Monitor forecast performance and update models regularly"]
    
    async def _create_visualizations(
        self,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        forecasts: Dict[str, Any],
        best_model_result: Optional[TimeSeriesResult]
    ) -> Dict[str, Any]:
        """Create visualization data for time series analysis."""
        try:
            visualizations = {}
            
            # Time series plot
            visualizations['timeseries'] = {
                'type': 'line',
                'data': {
                    'dates': dates.strftime('%Y-%m-%d').tolist(),
                    'values': y.tolist(),
                    'title': 'Time Series Data'
                }
            }
            
            # Forecast plot
            if forecasts and 'forecasts' in forecasts:
                forecast_dates = forecasts.get('forecast_dates', [])
                forecast_values = forecasts.get('forecasts', [])
                
                visualizations['forecast'] = {
                    'type': 'line_with_forecast',
                    'data': {
                        'historical_dates': dates.strftime('%Y-%m-%d').tolist(),
                        'historical_values': y.tolist(),
                        'forecast_dates': forecast_dates,
                        'forecast_values': forecast_values,
                        'confidence_intervals': forecasts.get('confidence_intervals'),
                        'title': 'Time Series Forecast'
                    }
                }
            
            # Residuals plot
            if best_model_result and best_model_result.residuals is not None:
                visualizations['residuals'] = {
                    'type': 'scatter',
                    'data': {
                        'x': list(range(len(best_model_result.residuals))),
                        'y': best_model_result.residuals.tolist(),
                        'title': 'Model Residuals'
                    }
                }
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            return {}
    
    async def _log_to_mlflow(self, report: TimeSeriesReport):
        """Log time series analysis results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"timeseries_analysis_{report.task_type.value}"):
                # Log parameters
                mlflow.log_param("task_type", report.task_type.value)
                mlflow.log_param("n_observations", report.dataset_info.get('n_observations', 0))
                mlflow.log_param("forecast_horizon", self.config.forecast_horizon)
                
                # Log dataset metrics
                if 'target_statistics' in report.dataset_info:
                    stats = report.dataset_info['target_statistics']
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"data_{key}", value)
                
                # Log model metrics
                if report.best_model_result:
                    mlflow.log_metric("best_test_rmse", report.best_model_result.test_score)
                    mlflow.log_param("best_model_type", report.best_model_result.model_type.value)
                    mlflow.log_metric("training_time", report.best_model_result.training_time)
                
                # Log evaluation metrics
                if 'metrics_summary' in report.evaluation_metrics:
                    for key, value in report.evaluation_metrics['metrics_summary'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                
                # Log artifacts
                report_dict = asdict(report)
                report_dict['timestamp'] = report.timestamp.isoformat()
                
                with open("timeseries_report.json", "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)
                mlflow.log_artifact("timeseries_report.json")
                
                logger.info("Time series analysis results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def predict(
        self,
        steps_ahead: int = None,
        return_confidence_intervals: bool = True
    ) -> Dict[str, Any]:
        """Generate predictions using the best trained model."""
        try:
            if not self.best_model:
                raise ValueError("No trained model available. Run analyze_timeseries first.")
            
            steps_ahead = steps_ahead or self.config.forecast_horizon
            
            # Generate forecasts
            if self.best_model_type == TimeSeriesModelType.LSTM:
                predictions, conf_int = await self.best_model.forecast(steps_ahead)
            else:
                predictions, conf_int = await self.best_model.forecast(steps_ahead)
            
            # Inverse transform if scaled
            if self.scaler:
                predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                if conf_int is not None:
                    conf_int = self.scaler.inverse_transform(conf_int)
            
            result = {
                'predictions': predictions.tolist(),
                'steps_ahead': steps_ahead,
                'model_type': self.best_model_type.value,
                'prediction_dates': pd.date_range(
                    start=pd.Timestamp.now(),
                    periods=steps_ahead,
                    freq=self.preprocessor.frequency or 'D'
                ).strftime('%Y-%m-%d').tolist()
            }
            
            if return_confidence_intervals and conf_int is not None:
                result['confidence_intervals'] = conf_int.tolist()
            
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
            summary = {
                'best_model': {
                    'type': self.best_model_type.value if self.best_model_type else None,
                    'parameters': self.best_model.get_model_params() if self.best_model else {}
                },
                'preprocessing_applied': bool(self.preprocessing_info),
                'scaler_used': self.scaler is not None,
                'frequency_detected': self.preprocessor.frequency,
                'seasonal_periods': self.preprocessor.seasonal_periods,
                'models_available': len(self.models),
                'configuration': asdict(self.config)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Model summary generation failed: {str(e)}")
            return {'error': str(e)}

# Utility functions for integration with Auto-Analyst pipeline

def create_timeseries_analyzer(
    forecast_horizon: int = 30,
    enable_gpu: bool = False,
    auto_select_model: bool = True
) -> TimeSeriesAnalyzer:
    """Factory function to create TimeSeriesAnalyzer."""
    config = TimeSeriesConfig()
    config.forecast_horizon = forecast_horizon
    config.use_gpu = enable_gpu
    config.auto_select_model = auto_select_model
    
    return TimeSeriesAnalyzer(config)

async def quick_timeseries_forecast(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    forecast_periods: int = 30
) -> Dict[str, Any]:
    """Quick time series forecasting for simple use cases."""
    # Create analyzer with simplified configuration
    analyzer = create_timeseries_analyzer(
        forecast_horizon=forecast_periods,
        auto_select_model=True
    )
    
    # Run analysis
    report = await analyzer.analyze_timeseries(
        df, target_column, date_column
    )
    
    # Return simplified results
    return {
        'forecasts': report.forecasts.get('forecasts', []),
        'forecast_dates': report.forecasts.get('forecast_dates', []),
        'best_model': report.best_model_result.model_type.value if report.best_model_result else None,
        'test_rmse': report.best_model_result.test_score if report.best_model_result else None,
        'insights': report.business_insights[:3],
        'recommendations': report.recommendations[:3],
        'dataset_info': {
            'n_observations': report.dataset_info.get('n_observations', 0),
            'frequency': report.dataset_info.get('frequency', 'unknown')
        }
    }

def get_available_models() -> Dict[str, bool]:
    """Get available time series models and their status."""
    return {
        'arima': STATSMODELS_AVAILABLE,
        'sarima': STATSMODELS_AVAILABLE,
        'prophet': PROPHET_AVAILABLE,
        'exponential_smoothing': STATSMODELS_AVAILABLE,
        'lstm': PYTORCH_AVAILABLE,
        'gru': PYTORCH_AVAILABLE,
        'tensorflow_models': TENSORFLOW_AVAILABLE,
        'advanced_optimization': OPTUNA_AVAILABLE,
        'visualization': PLOTLY_AVAILABLE,
        'mlflow_tracking': MLFLOW_AVAILABLE
    }

def validate_timeseries_data(
    df: pd.DataFrame,
    target_column: str,
    date_column: str
) -> Dict[str, Any]:
    """Validate time series data quality."""
    try:
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check required columns
        if target_column not in df.columns:
            validation['is_valid'] = False
            validation['issues'].append(f"Target column '{target_column}' not found")
        
        if date_column not in df.columns:
            validation['is_valid'] = False
            validation['issues'].append(f"Date column '{date_column}' not found")
        
        if not validation['is_valid']:
            return validation
        
        # Check data types and content
        try:
            pd.to_datetime(df[date_column])
        except:
            validation['issues'].append("Date column cannot be converted to datetime")
            validation['is_valid'] = False
        
        try:
            pd.to_numeric(df[target_column], errors='coerce')
        except:
            validation['issues'].append("Target column contains non-numeric values")
            validation['is_valid'] = False
        
        # Check for sufficient data
        if len(df) < 10:
            validation['is_valid'] = False
            validation['issues'].append("Insufficient data points (minimum 10 required)")
        elif len(df) < 50:
            validation['warnings'].append("Limited data available (less than 50 points)")
            validation['recommendations'].append("Consider collecting more historical data")
        
        # Check for missing values
        missing_target = df[target_column].isnull().sum()
        missing_date = df[date_column].isnull().sum()
        
        if missing_target > 0:
            if missing_target > len(df) * 0.2:
                validation['is_valid'] = False
                validation['issues'].append(f"Too many missing values in target ({missing_target})")
            else:
                validation['warnings'].append(f"Missing values in target will be handled ({missing_target})")
        
        if missing_date > 0:
            validation['warnings'].append(f"Missing values in date column ({missing_date})")
        
        # Check for duplicates
        if df[date_column].duplicated().sum() > 0:
            validation['warnings'].append("Duplicate dates found - will be removed")
        
        return validation
        
    except Exception as e:
        return {
            'is_valid': False,
            'issues': [f"Validation failed: {str(e)}"],
            'warnings': [],
            'recommendations': []
        }

# Export main classes and functions
__all__ = [
    'TimeSeriesAnalyzer',
    'TimeSeriesConfig',
    'TimeSeriesReport',
    'TimeSeriesResult',
    'TimeSeriesPreprocessor',
    'BaseTimeSeriesModel',
    'ARIMAModel',
    'SARIMAModel',
    'ProphetModel',
    'LSTMModel',
    'TimeSeriesEvaluator',
    'create_timeseries_analyzer',
    'quick_timeseries_forecast',
    'get_available_models',
    'validate_timeseries_data'
]

# Example usage and testing
if __name__ == "__main__":
    async def test_timeseries_analysis():
        """Test the time series analysis functionality."""
        print("Testing Time Series Analysis...")
        print("Available models:", get_available_models())
        
        # Create sample time series data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Generate synthetic time series with trend and seasonality
        t = np.arange(len(dates))
        trend = 0.1 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 365.25)  # Yearly seasonality
        noise = np.random.normal(0, 2, len(dates))
        values = 100 + trend + seasonal + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'external_feature': np.random.randn(len(dates))
        })
        
        print(f"Generated time series data: {len(df)} observations")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Validate data
        validation = validate_timeseries_data(df, 'value', 'date')
        print(f"Data validation: {'✓' if validation['is_valid'] else '✗'}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        # Test quick forecasting
        print("\n=== Quick Forecasting Test ===")
        quick_results = await quick_timeseries_forecast(
            df, 'value', 'date', forecast_periods=30
        )
        
        print(f"Quick Forecast Results:")
        print(f"  Best Model: {quick_results['best_model']}")
        print(f"  Test RMSE: {quick_results['test_rmse']:.4f}" if quick_results['test_rmse'] else "No RMSE")
        print(f"  Forecast Length: {len(quick_results['forecasts'])}")
        print(f"  Dataset: {quick_results['dataset_info']['n_observations']} observations")
        
        # Test comprehensive analysis
        print("\n=== Comprehensive Analysis Test ===")
        analyzer = create_timeseries_analyzer(
            forecast_horizon=60,
            enable_gpu=False,
            auto_select_model=True
        )
        
        report = await analyzer.analyze_timeseries(
            df, 'value', 'date', external_features=['external_feature']
        )
        
        print(f"Comprehensive Analysis Results:")
        print(f"  Task Type: {report.task_type.value}")
        print(f"  Models Evaluated: {len(report.models_evaluated)}")
        
        if report.best_model_result:
            print(f"  Best Model: {report.best_model_result.model_type.value}")
            print(f"  Test RMSE: {report.best_model_result.test_score:.4f}")
            print(f"  Training Time: {report.best_model_result.training_time:.2f}s")
            print(f"  Forecast Horizon: {report.best_model_result.forecast_horizon}")
        
        # Dataset analysis
        print(f"\n  Dataset Analysis:")
        dataset_info = report.dataset_info
        print(f"    Observations: {dataset_info.get('n_observations', 0)}")
        print(f"    Date Range: {dataset_info.get('date_range', {}).get('duration_days', 0)} days")
        print(f"    Frequency: {dataset_info.get('frequency', 'unknown')}")
        print(f"    Seasonal Periods: {dataset_info.get('seasonal_periods', 'none')}")
        
        # Time series analysis
        if report.time_series_analysis:
            print(f"\n  Time Series Analysis:")
            ts_analysis = report.time_series_analysis
            
            if 'trend' in ts_analysis:
                trend_info = ts_analysis['trend']
                print(f"    Trend: {trend_info.get('trend_direction', 'unknown')} (strength: {trend_info.get('trend_strength', 0):.3f})")
            
            if 'stationarity' in ts_analysis:
                stat_info = ts_analysis['stationarity']
                print(f"    Stationary: {stat_info.get('overall_stationary', 'unknown')}")
        
        # Forecasts
        if report.forecasts:
            forecasts = report.forecasts
            print(f"\n  Forecasts:")
            print(f"    Horizon: {forecasts.get('horizon', 0)} periods")
            print(f"    Model Used: {forecasts.get('model_used', 'unknown')}")
            forecast_values = forecasts.get('forecasts', [])
            if forecast_values:
                print(f"    Sample Forecasts: {forecast_values[:5]}")
        
        # Business insights
        print(f"\n  Business Insights:")
        for i, insight in enumerate(report.business_insights[:3], 1):
            print(f"    {i}. {insight}")
        
        print(f"\n  Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"    {i}. {rec}")
        
        # Test individual model predictions
        if report.best_model_result:
            print(f"\n=== Individual Model Prediction Test ===")
            predictions = await analyzer.predict(steps_ahead=10, return_confidence_intervals=True)
            
            print(f"Predictions:")
            print(f"  Steps Ahead: {predictions['steps_ahead']}")
            print(f"  Model Type: {predictions['model_type']}")
            print(f"  Predictions: {predictions['predictions'][:5] if predictions['predictions'] else 'None'}")
            print(f"  Has Confidence Intervals: {'confidence_intervals' in predictions}")
        
        # Model summary
        print(f"\n=== Model Summary ===")
        summary = analyzer.get_model_summary()
        print(f"  Best Model Type: {summary['best_model']['type']}")
        print(f"  Preprocessing Applied: {summary['preprocessing_applied']}")
        print(f"  Scaler Used: {summary['scaler_used']}")
        print(f"  Frequency Detected: {summary['frequency_detected']}")
        
        return report
    
    # Run test
    import asyncio
    results = asyncio.run(test_timeseries_analysis())
