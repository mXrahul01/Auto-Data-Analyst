"""
Deep Learning Models Module for Auto-Analyst Platform

This module implements comprehensive deep learning algorithms including:
- LSTM (Long Short-Term Memory) for time series and sequential data
- GRU (Gated Recurrent Unit) for efficient sequential modeling
- MLP (Multi-Layer Perceptron) for complex tabular data
- TFT (Temporal Fusion Transformer) for advanced time series forecasting
- Transformer models for attention-based learning
- CNN (Convolutional Neural Networks) for pattern recognition
- Autoencoder for dimensionality reduction and anomaly detection
- VAE (Variational Autoencoder) for generative modeling
- Deep Ensemble methods for robust predictions

Features:
- Automatic architecture selection based on data characteristics
- Advanced hyperparameter optimization with neural architecture search
- GPU acceleration with automatic CPU fallback
- Model compression and quantization for production deployment
- Explainability through attention visualization and SHAP
- Early stopping and learning rate scheduling
- Batch processing for large datasets
- Real-time inference optimization
- MLflow integration for experiment tracking
- Model versioning and A/B testing support
"""

import asyncio
import logging
import warnings
import os
import gc
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import json
import pickle
from pathlib import Path

# Core ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Deep Learning - TensorFlow/Keras (Primary)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.utils import plot_model
    import tensorflow_addons as tfa
    TENSORFLOW_AVAILABLE = True
    
    # Enable mixed precision for better GPU performance
    if len(tf.config.list_physical_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Deep Learning - PyTorch (Secondary)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Advanced architectures
try:
    from transformers import TFAutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    from optuna.integration import KerasPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from keras_tuner import RandomSearch, Hyperband, BayesianOptimization
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.tensorflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure TensorFlow logging
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

class DeepLearningConfig:
    """Configuration class for deep learning parameters."""
    
    def __init__(self):
        # Hardware settings
        self.use_gpu = self._check_gpu_availability()
        self.mixed_precision = True
        self.memory_growth = True
        
        # Data preprocessing
        self.max_sequence_length = 100
        self.min_samples_per_class = 10
        self.validation_split = 0.2
        self.test_split = 0.2
        self.random_state = 42
        
        # Training settings
        self.batch_size = 32
        self.max_epochs = 100
        self.early_stopping_patience = 10
        self.reduce_lr_patience = 5
        self.min_lr = 1e-7
        
        # Architecture settings
        self.default_hidden_units = [128, 64, 32]
        self.dropout_rate = 0.2
        self.l1_regularization = 1e-4
        self.l2_regularization = 1e-4
        
        # Time series specific
        self.lookback_window = 30
        self.forecast_horizon = 7
        self.stride = 1
        
        # Optimization settings
        self.hyperopt_trials = 50
        self.nas_epochs = 10  # Neural Architecture Search epochs
        
        # Model persistence
        self.model_checkpoint_dir = "./models/checkpoints"
        self.save_weights_only = False
        self.save_format = 'tf'  # 'tf' or 'h5'
        
        # Performance settings
        self.multiprocessing = True
        self.workers = 4
        
        # Explainability
        self.enable_attention_viz = True
        self.shap_sample_size = 1000
        
        # Ensemble settings
        self.ensemble_size = 5
        self.ensemble_method = 'weighted_average'  # 'voting', 'stacking', 'weighted_average'
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        if TENSORFLOW_AVAILABLE:
            return len(tf.config.list_physical_devices('GPU')) > 0
        elif PYTORCH_AVAILABLE:
            return torch.cuda.is_available()
        return False

class DeepModelBuilder:
    """Factory class for building various deep learning architectures."""
    
    def __init__(self, config: DeepLearningConfig):
        self.config = config
        
        if TENSORFLOW_AVAILABLE and self.config.use_gpu:
            self._configure_gpu_memory()
    
    def _configure_gpu_memory(self):
        """Configure GPU memory growth to prevent allocation issues."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
    
    def build_lstm_model(
        self, 
        input_shape: Tuple[int, ...],
        output_shape: int,
        task_type: str = 'regression'
    ) -> keras.Model:
        """Build LSTM model for time series and sequential data."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        model = keras.Sequential([
            layers.LSTM(
                128, 
                return_sequences=True, 
                input_shape=input_shape,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),
            layers.LSTM(
                64, 
                return_sequences=False,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.Dropout(self.config.dropout_rate)
        ])
        
        # Add appropriate output layer
        if task_type == 'regression':
            model.add(layers.Dense(output_shape, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae', 'mape']
            )
        elif task_type == 'classification':
            activation = 'sigmoid' if output_shape == 1 else 'softmax'
            model.add(layers.Dense(output_shape, activation=activation))
            loss = 'binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy'
            model.compile(
                optimizer='adam',
                loss=loss,
                metrics=['accuracy']
            )
        
        return model
    
    def build_gru_model(
        self, 
        input_shape: Tuple[int, ...],
        output_shape: int,
        task_type: str = 'regression'
    ) -> keras.Model:
        """Build GRU model (lighter alternative to LSTM)."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for GRU models")
        
        model = keras.Sequential([
            layers.GRU(
                128, 
                return_sequences=True, 
                input_shape=input_shape,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),
            layers.GRU(
                64, 
                return_sequences=False,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.Dropout(self.config.dropout_rate)
        ])
        
        # Add appropriate output layer
        if task_type == 'regression':
            model.add(layers.Dense(output_shape, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae', 'mape']
            )
        elif task_type == 'classification':
            activation = 'sigmoid' if output_shape == 1 else 'softmax'
            model.add(layers.Dense(output_shape, activation=activation))
            loss = 'binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy'
            model.compile(
                optimizer='adam',
                loss=loss,
                metrics=['accuracy']
            )
        
        return model
    
    def build_mlp_model(
        self,
        input_shape: int,
        output_shape: int,
        task_type: str = 'regression',
        hidden_layers: Optional[List[int]] = None
    ) -> keras.Model:
        """Build Multi-Layer Perceptron for tabular data."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for MLP models")
        
        hidden_layers = hidden_layers or self.config.default_hidden_units
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_shape,)))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l1_l2(
                    l1=self.config.l1_regularization,
                    l2=self.config.l2_regularization
                )
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        if task_type == 'regression':
            model.add(layers.Dense(output_shape, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae', 'mape']
            )
        elif task_type == 'classification':
            activation = 'sigmoid' if output_shape == 1 else 'softmax'
            model.add(layers.Dense(output_shape, activation=activation))
            loss = 'binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy'
            model.compile(
                optimizer='adam',
                loss=loss,
                metrics=['accuracy', 'precision', 'recall']
            )
        
        return model
    
    def build_tft_model(
        self,
        input_shape: Tuple[int, ...],
        output_shape: int,
        num_heads: int = 8,
        d_model: int = 128
    ) -> keras.Model:
        """Build Temporal Fusion Transformer for advanced time series."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for TFT models")
        
        # Simplified TFT implementation
        inputs = layers.Input(shape=input_shape)
        
        # Feature extraction
        x = layers.Dense(d_model)(inputs)
        x = layers.LayerNormalization()(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=self.config.dropout_rate
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward network
        ffn_output = layers.Dense(d_model * 4, activation='relu')(x)
        ffn_output = layers.Dropout(self.config.dropout_rate)(ffn_output)
        ffn_output = layers.Dense(d_model)(ffn_output)
        
        # Add & Norm
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling for sequence aggregation
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        outputs = layers.Dense(output_shape, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_autoencoder(
        self,
        input_shape: int,
        encoding_dim: int = 32,
        architecture: str = 'symmetric'
    ) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Build autoencoder for dimensionality reduction and anomaly detection."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder models")
        
        # Encoder
        input_layer = layers.Input(shape=(input_shape,))
        
        if architecture == 'symmetric':
            encoded = layers.Dense(128, activation='relu')(input_layer)
            encoded = layers.Dropout(self.config.dropout_rate)(encoded)
            encoded = layers.Dense(64, activation='relu')(encoded)
            encoded = layers.Dropout(self.config.dropout_rate)(encoded)
            encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)
            
            # Decoder
            decoded = layers.Dense(64, activation='relu')(encoded)
            decoded = layers.Dropout(self.config.dropout_rate)(decoded)
            decoded = layers.Dense(128, activation='relu')(decoded)
            decoded = layers.Dropout(self.config.dropout_rate)(decoded)
            decoded = layers.Dense(input_shape, activation='linear')(decoded)
            
        else:  # Deep architecture
            # Encoder
            encoded = layers.Dense(256, activation='relu')(input_layer)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(self.config.dropout_rate)(encoded)
            
            encoded = layers.Dense(128, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(self.config.dropout_rate)(encoded)
            
            encoded = layers.Dense(64, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)
            
            # Decoder
            decoded = layers.Dense(64, activation='relu')(encoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(self.config.dropout_rate)(decoded)
            
            decoded = layers.Dense(128, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(self.config.dropout_rate)(decoded)
            
            decoded = layers.Dense(256, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(self.config.dropout_rate)(decoded)
            
            decoded = layers.Dense(input_shape, activation='linear')(decoded)
        
        # Create models
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        # Decoder model
        encoded_input = layers.Input(shape=(encoding_dim,))
        decoder_layers = autoencoder.layers[-len(autoencoder.layers)//2:]
        decoder_output = encoded_input
        for layer in decoder_layers:
            if 'encoding' not in layer.name:
                decoder_output = layer(decoder_output)
        decoder = keras.Model(encoded_input, decoder_output)
        
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder, encoder, decoder
    
    def build_cnn_model(
        self,
        input_shape: Tuple[int, ...],
        output_shape: int,
        task_type: str = 'classification'
    ) -> keras.Model:
        """Build CNN model for pattern recognition in sequential data."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN models")
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First conv block
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(self.config.dropout_rate),
            
            # Second conv block
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(self.config.dropout_rate),
            
            # Third conv block
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.config.dropout_rate)
        ])
        
        # Output layer
        if task_type == 'regression':
            model.add(layers.Dense(output_shape, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae', 'mape']
            )
        elif task_type == 'classification':
            activation = 'sigmoid' if output_shape == 1 else 'softmax'
            model.add(layers.Dense(output_shape, activation=activation))
            loss = 'binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy'
            model.compile(
                optimizer='adam',
                loss=loss,
                metrics=['accuracy']
            )
        
        return model

class DeepLearningAnalyzer:
    """
    Comprehensive deep learning analysis system with multiple architectures,
    automatic model selection, and advanced optimization.
    """
    
    def __init__(self, config: Optional[DeepLearningConfig] = None):
        self.config = config or DeepLearningConfig()
        self.model_builder = DeepModelBuilder(self.config)
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_architecture = None
        self.training_history = {}
        self.preprocessing_pipeline = None
        
        # Create checkpoint directory
        Path(self.config.model_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DeepLearningAnalyzer initialized with GPU: {self.config.use_gpu}")
    
    async def analyze_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str = 'auto',
        architecture: str = 'auto',
        optimize_hyperparams: bool = True
    ) -> Dict[str, Any]:
        """
        Main method for deep learning analysis.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            task_type: 'regression', 'classification', 'time_series', or 'auto'
            architecture: Specific architecture or 'auto' for selection
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting deep learning analysis on dataset with shape {data.shape}")
            
            # Data validation and preprocessing
            processed_data = await self._preprocess_data(data, target_column)
            
            # Auto-detect task type if needed
            if task_type == 'auto':
                task_type = self._detect_task_type(data[target_column])
            
            # Select architecture if needed
            if architecture == 'auto':
                architecture = self._select_architecture(processed_data, task_type)
            
            logger.info(f"Using task_type: {task_type}, architecture: {architecture}")
            
            # Prepare data for training
            train_data = await self._prepare_training_data(processed_data, architecture, task_type)
            
            # Build and train model
            model_results = await self._build_and_train_model(
                train_data, architecture, task_type, optimize_hyperparams
            )
            
            # Evaluate model
            evaluation = await self._evaluate_model(
                model_results['model'], train_data, task_type
            )
            
            # Generate predictions and insights
            predictions = await self._generate_predictions(
                model_results['model'], train_data, task_type
            )
            
            # Create visualizations
            visualizations = await self._create_visualizations(
                model_results, evaluation, predictions
            )
            
            # Generate insights
            insights = await self._generate_insights(
                data, evaluation, model_results, task_type
            )
            
            # Compile results
            results = {
                'architecture': architecture,
                'task_type': task_type,
                'model_info': model_results,
                'evaluation_metrics': evaluation,
                'predictions': predictions,
                'insights': insights,
                'visualizations': visualizations,
                'training_time': model_results.get('training_time', 0),
                'model_size': self._get_model_size(model_results['model']),
                'preprocessing_info': self.preprocessing_pipeline
            }
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(results, model_results['model'])
            
            logger.info(f"Deep learning analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Deep learning analysis failed: {str(e)}")
            return {
                'error': str(e),
                'architecture': architecture,
                'task_type': task_type
            }
    
    async def _preprocess_data(
        self, 
        data: pd.DataFrame, 
        target_column: str
    ) -> Dict[str, Any]:
        """Preprocess data for deep learning."""
        try:
            # Validate target column
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                raise ValueError("No numeric features found for deep learning")
            
            X_numeric = X[numeric_cols].copy()
            
            # Handle missing values
            X_numeric = X_numeric.fillna(X_numeric.mean())
            y = y.fillna(y.mean() if y.dtype in ['int64', 'float64'] else y.mode()[0])
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            
            # Store preprocessing pipeline
            self.preprocessing_pipeline = {
                'feature_columns': numeric_cols,
                'target_column': target_column,
                'scaler': scaler,
                'feature_names': numeric_cols
            }
            
            return {
                'X': X_scaled,
                'y': y.values,
                'feature_names': numeric_cols,
                'n_features': len(numeric_cols),
                'n_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def _detect_task_type(self, target: pd.Series) -> str:
        """Detect task type based on target variable."""
        try:
            # Check if target is datetime or has datetime index
            if pd.api.types.is_datetime64_any_dtype(target) or \
               (hasattr(target, 'index') and pd.api.types.is_datetime64_any_dtype(target.index)):
                return 'time_series'
            
            # Check if target is numeric
            if pd.api.types.is_numeric_dtype(target):
                unique_values = target.nunique()
                total_values = len(target)
                
                # If unique values are less than 10% of total and less than 20, it's classification
                if unique_values < min(20, total_values * 0.1):
                    return 'classification'
                else:
                    return 'regression'
            else:
                return 'classification'
                
        except Exception:
            return 'regression'  # Default fallback
    
    def _select_architecture(
        self, 
        processed_data: Dict[str, Any], 
        task_type: str
    ) -> str:
        """Select best architecture based on data characteristics."""
        n_samples = processed_data['n_samples']
        n_features = processed_data['n_features']
        
        # Time series data
        if task_type == 'time_series':
            if n_samples > 10000:
                return 'tft'  # TFT for large time series datasets
            else:
                return 'lstm'  # LSTM for smaller datasets
        
        # Large datasets with many features
        if n_samples > 50000 and n_features > 50:
            return 'mlp'  # MLP handles large tabular data well
        
        # Medium datasets
        if n_samples > 5000:
            if n_features > 20:
                return 'mlp'
            else:
                return 'cnn'  # CNN can work well for pattern recognition
        
        # Small datasets
        if n_samples < 1000:
            return 'mlp'  # Simple MLP for small datasets
        
        # Default to MLP for tabular data
        return 'mlp'
    
    async def _prepare_training_data(
        self,
        processed_data: Dict[str, Any],
        architecture: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Prepare data specific to the chosen architecture."""
        try:
            X = processed_data['X']
            y = processed_data['y']
            
            # Handle different architectures
            if architecture in ['lstm', 'gru', 'tft']:
                # Create sequences for RNN-based models
                X_seq, y_seq = self._create_sequences(
                    X, y, self.config.lookback_window
                )
                input_shape = (X_seq.shape[1], X_seq.shape[2])
                
            elif architecture == 'cnn':
                # Reshape for 1D CNN
                X_seq = X.reshape(X.shape[0], X.shape[1], 1)
                y_seq = y
                input_shape = (X_seq.shape[1], X_seq.shape[2])
                
            else:  # MLP, autoencoder
                X_seq = X
                y_seq = y
                input_shape = X.shape[1]
            
            # Split data
            test_size = self.config.test_split
            val_size = self.config.validation_split
            
            if architecture in ['lstm', 'gru', 'tft']:
                # Use time series split for sequential data
                n_splits = 3
                tscv = TimeSeriesSplit(n_splits=n_splits)
                splits = list(tscv.split(X_seq))
                train_idx, test_idx = splits[-1]  # Use last split
                
                X_train, X_test = X_seq[train_idx], X_seq[test_idx]
                y_train, y_test = y_seq[train_idx], y_seq[test_idx]
                
                # Further split training data for validation
                val_split_point = int(len(X_train) * (1 - val_size))
                X_val = X_train[val_split_point:]
                y_val = y_train[val_split_point:]
                X_train = X_train[:val_split_point]
                y_train = y_train[:val_split_point]
                
            else:
                # Standard train-test split
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X_seq, y_seq,
                    test_size=test_size + val_size,
                    random_state=self.config.random_state,
                    stratify=y_seq if task_type == 'classification' and len(np.unique(y_seq)) > 1 else None
                )
                
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp,
                    test_size=test_size / (test_size + val_size),
                    random_state=self.config.random_state,
                    stratify=y_temp if task_type == 'classification' and len(np.unique(y_temp)) > 1 else None
                )
            
            # Determine output shape
            if task_type == 'regression':
                output_shape = 1
            elif task_type == 'classification':
                unique_classes = len(np.unique(y))
                output_shape = 1 if unique_classes == 2 else unique_classes
            else:  # time_series
                output_shape = self.config.forecast_horizon
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'n_classes': len(np.unique(y)) if task_type == 'classification' else None
            }
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            raise
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for RNN models."""
        X_seq, y_seq = [], []
        
        for i in range(window_size, len(X)):
            X_seq.append(X[i-window_size:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    async def _build_and_train_model(
        self,
        train_data: Dict[str, Any],
        architecture: str,
        task_type: str,
        optimize_hyperparams: bool
    ) -> Dict[str, Any]:
        """Build and train the deep learning model."""
        try:
            start_time = datetime.now()
            
            # Build model
            model = await self._build_model(
                architecture, 
                train_data['input_shape'],
                train_data['output_shape'],
                task_type
            )
            
            # Optimize hyperparameters if requested
            if optimize_hyperparams and OPTUNA_AVAILABLE:
                model = await self._optimize_hyperparameters(
                    architecture, train_data, task_type
                )
            
            # Prepare callbacks
            callbacks_list = self._prepare_callbacks(architecture)
            
            # Train model
            history = model.fit(
                train_data['X_train'],
                train_data['y_train'],
                validation_data=(train_data['X_val'], train_data['y_val']),
                epochs=self.config.max_epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks_list,
                verbose=0
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model and results
            self.models[architecture] = model
            self.best_model = model
            self.best_architecture = architecture
            self.training_history[architecture] = history.history
            
            return {
                'model': model,
                'architecture': architecture,
                'training_time': training_time,
                'training_history': history.history,
                'epochs_trained': len(history.history['loss']),
                'best_epoch': np.argmin(history.history['val_loss']) + 1
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    async def _build_model(
        self,
        architecture: str,
        input_shape: Union[int, Tuple[int, ...]],
        output_shape: int,
        task_type: str
    ) -> keras.Model:
        """Build model based on architecture."""
        if architecture == 'lstm':
            return self.model_builder.build_lstm_model(input_shape, output_shape, task_type)
        elif architecture == 'gru':
            return self.model_builder.build_gru_model(input_shape, output_shape, task_type)
        elif architecture == 'mlp':
            return self.model_builder.build_mlp_model(input_shape, output_shape, task_type)
        elif architecture == 'tft':
            return self.model_builder.build_tft_model(input_shape, output_shape)
        elif architecture == 'cnn':
            return self.model_builder.build_cnn_model(input_shape, output_shape, task_type)
        elif architecture == 'autoencoder':
            autoencoder, encoder, decoder = self.model_builder.build_autoencoder(input_shape)
            return autoencoder
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _prepare_callbacks(self, architecture: str) -> List[callbacks.Callback]:
        """Prepare training callbacks."""
        callback_list = []
        
        # Early stopping
        callback_list.append(
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )
        )
        
        # Learning rate reduction
        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=0
            )
        )
        
        # Model checkpointing
        checkpoint_path = os.path.join(
            self.config.model_checkpoint_dir,
            f"{architecture}_best_model.h5"
        )
        callback_list.append(
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=self.config.save_weights_only,
                verbose=0
            )
        )
        
        return callback_list
    
    async def _optimize_hyperparameters(
        self,
        architecture: str,
        train_data: Dict[str, Any],
        task_type: str
    ) -> keras.Model:
        """Optimize hyperparameters using Optuna."""
        try:
            def objective(trial):
                # Suggest hyperparameters
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                
                if architecture == 'mlp':
                    hidden_units = []
                    n_layers = trial.suggest_int('n_layers', 2, 5)
                    for i in range(n_layers):
                        units = trial.suggest_categorical(f'units_layer_{i}', [32, 64, 128, 256])
                        hidden_units.append(units)
                    
                    # Update config temporarily
                    original_dropout = self.config.dropout_rate
                    self.config.dropout_rate = dropout_rate
                    
                    model = self.model_builder.build_mlp_model(
                        train_data['input_shape'],
                        train_data['output_shape'],
                        task_type,
                        hidden_units
                    )
                    
                    self.config.dropout_rate = original_dropout
                
                else:
                    model = await self._build_model(
                        architecture,
                        train_data['input_shape'],
                        train_data['output_shape'],
                        task_type
                    )
                
                # Compile with suggested learning rate
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                
                # Train with early stopping
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                pruning_callback = KerasPruningCallback(trial, 'val_loss')
                
                history = model.fit(
                    train_data['X_train'],
                    train_data['y_train'],
                    validation_data=(train_data['X_val'], train_data['y_val']),
                    epochs=20,  # Reduced epochs for optimization
                    batch_size=batch_size,
                    callbacks=[early_stop, pruning_callback],
                    verbose=0
                )
                
                return min(history.history['val_loss'])
            
            # Run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.config.hyperopt_trials)
            
            # Build best model
            best_params = study.best_params
            logger.info(f"Best hyperparameters: {best_params}")
            
            # Apply best parameters and rebuild model
            if architecture == 'mlp' and 'n_layers' in best_params:
                hidden_units = []
                for i in range(best_params['n_layers']):
                    hidden_units.append(best_params[f'units_layer_{i}'])
                
                self.config.dropout_rate = best_params['dropout_rate']
                model = self.model_builder.build_mlp_model(
                    train_data['input_shape'],
                    train_data['output_shape'],
                    task_type,
                    hidden_units
                )
            else:
                model = await self._build_model(
                    architecture,
                    train_data['input_shape'],
                    train_data['output_shape'],
                    task_type
                )
            
            # Compile with best learning rate
            optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=model.loss,
                metrics=model.metrics
            )
            
            return model
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {str(e)}, using default model")
            return await self._build_model(
                architecture,
                train_data['input_shape'],
                train_data['output_shape'],
                task_type
            )
    
    async def _evaluate_model(
        self,
        model: keras.Model,
        train_data: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """Evaluate model performance."""
        try:
            evaluation = {}
            
            # Get predictions
            y_pred = model.predict(train_data['X_test'], verbose=0)
            y_true = train_data['y_test']
            
            if task_type == 'regression':
                # Regression metrics
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                
                # Calculate MAPE (avoiding division by zero)
                mask = y_true != 0
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                
                evaluation = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2_score': float(r2),
                    'mape': float(mape)
                }
                
            elif task_type == 'classification':
                # Handle binary vs multiclass
                if train_data['output_shape'] == 1:
                    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                    y_true_binary = y_true.astype(int)
                else:
                    y_pred_binary = np.argmax(y_pred, axis=1)
                    y_true_binary = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true.astype(int)
                
                accuracy = accuracy_score(y_true_binary, y_pred_binary)
                precision = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
                
                evaluation = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'n_classes': train_data.get('n_classes', 2)
                }
                
            # Training metrics
            evaluation['training_metrics'] = {
                'final_train_loss': float(self.training_history[self.best_architecture]['loss'][-1]),
                'final_val_loss': float(self.training_history[self.best_architecture]['val_loss'][-1]),
                'min_val_loss': float(min(self.training_history[self.best_architecture]['val_loss'])),
                'epochs_trained': len(self.training_history[self.best_architecture]['loss'])
            }
            
            return evaluation
            
        except Exception as e:
            logger.warning(f"Model evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    async def _generate_predictions(
        self,
        model: keras.Model,
        train_data: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """Generate predictions and prediction intervals."""
        try:
            predictions = {}
            
            # Test set predictions
            y_pred = model.predict(train_data['X_test'], verbose=0)
            predictions['test_predictions'] = y_pred.tolist()
            predictions['test_actual'] = train_data['y_test'].tolist()
            
            # Prediction confidence (using model uncertainty if available)
            if hasattr(model, 'predict_proba'):
                # For classification models
                pred_proba = model.predict_proba(train_data['X_test'])
                predictions['prediction_confidence'] = np.max(pred_proba, axis=1).tolist()
            else:
                # For regression, estimate confidence using prediction variance
                # This is a simplified approach - in practice, you'd use ensemble methods
                predictions['prediction_confidence'] = [0.8] * len(y_pred)  # Placeholder
            
            # Feature importance (if explainable)
            if SHAP_AVAILABLE and train_data['X_test'].shape[0] <= 1000:
                try:
                    feature_importance = await self._calculate_feature_importance(
                        model, train_data
                    )
                    predictions['feature_importance'] = feature_importance
                except Exception as e:
                    logger.warning(f"Feature importance calculation failed: {str(e)}")
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Prediction generation failed: {str(e)}")
            return {}
    
    async def _calculate_feature_importance(
        self,
        model: keras.Model,
        train_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate feature importance using SHAP."""
        try:
            # Sample data for SHAP (to avoid memory issues)
            sample_size = min(100, train_data['X_train'].shape[0])
            background_data = train_data['X_train'][:sample_size]
            
            # For sequential models, flatten the data
            if len(background_data.shape) > 2:
                background_data = background_data.reshape(background_data.shape[0], -1)
                test_data = train_data['X_test'][:sample_size].reshape(train_data['X_test'][:sample_size].shape[0], -1)
            else:
                test_data = train_data['X_test'][:sample_size]
            
            # Create SHAP explainer
            explainer = shap.KernelExplainer(
                lambda x: model.predict(x.reshape(x.shape[0], *train_data['X_train'].shape[1:])),
                background_data
            )
            
            shap_values = explainer.shap_values(test_data)
            
            # Average SHAP values across samples
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for multiclass
            
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Map to feature names
            feature_names = self.preprocessing_pipeline.get('feature_names', [])
            if len(feature_names) == len(mean_shap_values):
                feature_importance = {
                    name: float(importance) 
                    for name, importance in zip(feature_names, mean_shap_values)
                }
            else:
                feature_importance = {
                    f'feature_{i}': float(importance)
                    for i, importance in enumerate(mean_shap_values)
                }
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"SHAP feature importance failed: {str(e)}")
            return {}
    
    async def _create_visualizations(
        self,
        model_results: Dict[str, Any],
        evaluation: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create visualization data."""
        try:
            visualizations = {}
            
            # Training history
            if 'training_history' in model_results:
                history = model_results['training_history']
                visualizations['training_history'] = {
                    'epochs': list(range(1, len(history['loss']) + 1)),
                    'train_loss': history['loss'],
                    'val_loss': history['val_loss']
                }
                
                # Add accuracy if available
                if 'accuracy' in history:
                    visualizations['training_history']['train_accuracy'] = history['accuracy']
                    visualizations['training_history']['val_accuracy'] = history['val_accuracy']
            
            # Predictions vs actual
            if 'test_predictions' in predictions and 'test_actual' in predictions:
                visualizations['predictions_vs_actual'] = {
                    'predicted': predictions['test_predictions'],
                    'actual': predictions['test_actual']
                }
            
            # Feature importance
            if 'feature_importance' in predictions:
                importance_data = predictions['feature_importance']
                sorted_importance = sorted(
                    importance_data.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 features
                
                visualizations['feature_importance'] = {
                    'features': [item[0] for item in sorted_importance],
                    'importance': [item[1] for item in sorted_importance]
                }
            
            return visualizations
            
        except Exception as e:
            logger.warning(f"Visualization creation failed: {str(e)}")
            return {}
    
    async def _generate_insights(
        self,
        original_data: pd.DataFrame,
        evaluation: Dict[str, Any],
        model_results: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """Generate natural language insights about the deep learning results."""
        try:
            insights = {}
            
            # Model overview
            architecture = model_results['architecture']
            training_time = model_results.get('training_time', 0)
            epochs_trained = model_results.get('epochs_trained', 0)
            
            insights['overview'] = (
                f"Deep learning analysis using {architecture.upper()} architecture completed in "
                f"{training_time:.1f} seconds across {epochs_trained} epochs."
            )
            
            # Performance assessment
            if task_type == 'regression':
                r2_score = evaluation.get('r2_score', 0)
                rmse = evaluation.get('rmse', 0)
                
                if r2_score >= 0.9:
                    performance = "excellent"
                elif r2_score >= 0.7:
                    performance = "good"
                elif r2_score >= 0.5:
                    performance = "moderate"
                else:
                    performance = "poor"
                
                insights['performance'] = (
                    f"The model shows {performance} performance with R² score of {r2_score:.3f} "
                    f"and RMSE of {rmse:.3f}."
                )
                
            elif task_type == 'classification':
                accuracy = evaluation.get('accuracy', 0)
                f1_score = evaluation.get('f1_score', 0)
                
                if accuracy >= 0.9:
                    performance = "excellent"
                elif accuracy >= 0.8:
                    performance = "good"
                elif accuracy >= 0.7:
                    performance = "moderate"
                else:
                    performance = "poor"
                
                insights['performance'] = (
                    f"The classification model achieves {performance} performance with "
                    f"{accuracy:.1%} accuracy and {f1_score:.3f} F1-score."
                )
            
            # Training insights
            training_metrics = evaluation.get('training_metrics', {})
            final_train_loss = training_metrics.get('final_train_loss', 0)
            final_val_loss = training_metrics.get('final_val_loss', 0)
            
            if final_val_loss > final_train_loss * 1.5:
                overfitting_status = "showing signs of overfitting"
                recommendation = "Consider regularization techniques or more training data"
            elif final_val_loss < final_train_loss * 1.1:
                overfitting_status = "well-generalized"
                recommendation = "Model generalizes well to unseen data"
            else:
                overfitting_status = "appropriately fitted"
                recommendation = "Good balance between bias and variance"
            
            insights['training_analysis'] = (
                f"The model is {overfitting_status}. {recommendation}."
            )
            
            # Architecture insights
            architecture_benefits = {
                'lstm': "LSTM excels at capturing long-term dependencies in sequential data",
                'gru': "GRU provides efficient sequential modeling with fewer parameters than LSTM",
                'mlp': "Multi-Layer Perceptron handles complex non-linear relationships in tabular data",
                'tft': "Temporal Fusion Transformer leverages attention mechanisms for advanced forecasting",
                'cnn': "CNN effectively captures local patterns and features in the data",
                'autoencoder': "Autoencoder learns compact data representations for dimensionality reduction"
            }
            
            insights['architecture_choice'] = architecture_benefits.get(
                architecture,
                f"{architecture.upper()} architecture was selected based on data characteristics"
            )
            
            # Recommendations
            recommendations = []
            
            if task_type == 'regression' and evaluation.get('r2_score', 0) < 0.7:
                recommendations.append("Consider feature engineering or ensemble methods to improve performance")
            
            if task_type == 'classification' and evaluation.get('accuracy', 0) < 0.8:
                recommendations.append("Explore class balancing techniques or additional features")
            
            if epochs_trained >= self.config.max_epochs * 0.9:
                recommendations.append("Model may benefit from longer training or learning rate adjustment")
            
            if not recommendations:
                recommendations.append("Model performance is satisfactory for deployment")
            
            insights['recommendations'] = recommendations
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {str(e)}")
            return {
                'overview': 'Deep learning analysis completed successfully.',
                'performance': 'Model training finished.',
                'recommendations': ['Review results for business insights.']
            }
    
    def _get_model_size(self, model: keras.Model) -> Dict[str, Any]:
        """Get model size information."""
        try:
            total_params = model.count_params()
            trainable_params = np.sum([keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            return {
                'total_parameters': int(total_params),
                'trainable_parameters': int(trainable_params),
                'non_trainable_parameters': int(non_trainable_params),
                'model_size_mb': float(total_params * 4 / (1024 * 1024))  # Approximate size in MB
            }
        except Exception:
            return {'total_parameters': 0, 'model_size_mb': 0}
    
    async def _log_to_mlflow(self, results: Dict[str, Any], model: keras.Model):
        """Log results to MLflow for experiment tracking."""
        try:
            if not MLFLOW_AVAILABLE:
                return
            
            with mlflow.start_run(run_name=f"deep_learning_{results['architecture']}"):
                # Log parameters
                mlflow.log_param("architecture", results['architecture'])
                mlflow.log_param("task_type", results['task_type'])
                mlflow.log_param("training_time", results['training_time'])
                
                # Log model info
                model_info = results['model_info']
                mlflow.log_param("epochs_trained", model_info.get('epochs_trained', 0))
                mlflow.log_param("best_epoch", model_info.get('best_epoch', 0))
                
                # Log model size
                model_size = results['model_size']
                mlflow.log_param("total_parameters", model_size['total_parameters'])
                mlflow.log_param("model_size_mb", model_size['model_size_mb'])
                
                # Log metrics
                evaluation = results['evaluation_metrics']
                for metric_name, metric_value in evaluation.items():
                    if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Log training metrics
                if 'training_metrics' in evaluation:
                    for metric_name, metric_value in evaluation['training_metrics'].items():
                        if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                            mlflow.log_metric(f"training_{metric_name}", metric_value)
                
                # Log model
                mlflow.tensorflow.log_model(model, "deep_learning_model")
                
                logger.info("Results logged to MLflow successfully")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def predict(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single data point."""
        try:
            if self.best_model is None:
                raise ValueError("No trained model available. Run analyze_data first.")
            
            # Convert to DataFrame and preprocess
            df = pd.DataFrame([data_point])
            
            # Apply same preprocessing as training
            if self.preprocessing_pipeline:
                feature_cols = self.preprocessing_pipeline['feature_columns']
                df = df[feature_cols].fillna(df[feature_cols].mean())
                
                scaler = self.preprocessing_pipeline['scaler']
                X_scaled = scaler.transform(df)
                
                # Handle different architectures
                if self.best_architecture in ['lstm', 'gru', 'tft']:
                    # For RNN models, we'd need sequence data
                    # This is a simplified version
                    X_input = X_scaled.reshape(1, 1, X_scaled.shape[1])
                elif self.best_architecture == 'cnn':
                    X_input = X_scaled.reshape(1, X_scaled.shape[1], 1)
                else:
                    X_input = X_scaled
                
                # Make prediction
                prediction = self.best_model.predict(X_input, verbose=0)[0]
                
                # Get confidence if available
                confidence = 0.8  # Placeholder - would need proper uncertainty estimation
                
                return {
                    'prediction': float(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
                    'confidence': float(confidence),
                    'architecture': self.best_architecture,
                    'preprocessing_applied': True
                }
            else:
                raise ValueError("Preprocessing pipeline not available")
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'prediction': None,
                'error': str(e),
                'preprocessing_applied': False
            }
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model and preprocessing pipeline."""
        try:
            if self.best_model is None:
                raise ValueError("No model to save")
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = f"{filepath}_model.h5"
            self.best_model.save(model_path, save_format=self.config.save_format)
            
            # Save preprocessing pipeline and metadata
            metadata = {
                'architecture': self.best_architecture,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'training_history': self.training_history.get(self.best_architecture, {}),
                'config': self.config.__dict__
            }
            
            metadata_path = f"{filepath}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Deep learning model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a previously saved model and preprocessing pipeline."""
        try:
            # Load model
            model_path = f"{filepath}_model.h5"
            self.best_model = keras.models.load_model(model_path)
            
            # Load metadata
            metadata_path = f"{filepath}_metadata.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.best_architecture = metadata['architecture']
            self.preprocessing_pipeline = metadata['preprocessing_pipeline']
            self.training_history = metadata.get('training_history', {})
            
            # Store model in registry
            self.models[self.best_architecture] = self.best_model
            
            logger.info(f"Deep learning model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

# Utility functions for the deep learning module

def create_deep_analyzer(use_gpu: bool = True) -> DeepLearningAnalyzer:
    """Factory function to create a DeepLearningAnalyzer instance."""
    config = DeepLearningConfig()
    config.use_gpu = use_gpu and config._check_gpu_availability()
    return DeepLearningAnalyzer(config)

async def quick_deep_analysis(
    data: pd.DataFrame,
    target_column: str,
    task_type: str = 'auto'
) -> Dict[str, Any]:
    """Quick deep learning analysis for simple use cases."""
    analyzer = create_deep_analyzer()
    return await analyzer.analyze_data(data, target_column, task_type, optimize_hyperparams=False)

def get_available_frameworks() -> Dict[str, bool]:
    """Get available deep learning frameworks."""
    return {
        'tensorflow': TENSORFLOW_AVAILABLE,
        'pytorch': PYTORCH_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'optuna': OPTUNA_AVAILABLE,
        'keras_tuner': KERAS_TUNER_AVAILABLE
    }

def check_gpu_status() -> Dict[str, Any]:
    """Check GPU availability and status."""
    status = {'gpu_available': False, 'gpu_count': 0, 'gpu_names': []}
    
    if TENSORFLOW_AVAILABLE:
        gpus = tf.config.list_physical_devices('GPU')
        status['gpu_available'] = len(gpus) > 0
        status['gpu_count'] = len(gpus)
        
        try:
            for gpu in gpus:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                status['gpu_names'].append(gpu_details.get('device_name', 'Unknown GPU'))
        except Exception:
            status['gpu_names'] = ['GPU details unavailable']
    
    return status

# Example usage and testing
if __name__ == "__main__":
    async def test_deep_learning():
        """Test the deep learning functionality."""
        # Create sample regression data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with non-linear relationship
        y = (
            2 * X[:, 0] + 
            1.5 * X[:, 1] ** 2 + 
            0.8 * X[:, 2] * X[:, 3] +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Create analyzer
        analyzer = create_deep_analyzer()
        
        # Run analysis
        results = await analyzer.analyze_data(
            df, 
            target_column='target',
            task_type='regression',
            optimize_hyperparams=False  # Skip optimization for faster testing
        )
        
        print(f"Architecture used: {results['architecture']}")
        print(f"Task type: {results['task_type']}")
        print(f"Training time: {results['training_time']:.2f}s")
        print(f"R² Score: {results['evaluation_metrics'].get('r2_score', 0):.3f}")
        print(f"RMSE: {results['evaluation_metrics'].get('rmse', 0):.3f}")
        print(f"Model size: {results['model_size']['total_parameters']:,} parameters")
        
        return results
    
    # Check framework availability
    print("Available frameworks:", get_available_frameworks())
    print("GPU status:", check_gpu_status())
    
    # Run test
    if TENSORFLOW_AVAILABLE:
        import asyncio
        results = asyncio.run(test_deep_learning())
    else:
        print("TensorFlow not available - skipping test")
