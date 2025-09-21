"""
Comprehensive Test Suite for ML Pipeline Components

This test module provides complete coverage for the Auto-Analyst ML pipeline
including the main AutoML pipeline, individual model modules, training,
prediction, and evaluation functionality.

Test Coverage:
- AutoML Pipeline (auto_pipeline.py) - Core AutoML functionality
- Individual Model Modules - Tabular, time series, text analysis models
- Training Pipeline - Model training execution and validation
- Prediction Pipeline - Inference and prediction generation
- Evaluation System - Metrics calculation and validation
- Edge Cases - Invalid data, missing values, error conditions
- Integration Tests - End-to-end pipeline functionality

Test Categories:
- Unit Tests: Individual component functionality
- Integration Tests: Component interaction and data flow
- Edge Case Tests: Error handling and boundary conditions
- Performance Tests: Training time and memory usage validation
- Data Quality Tests: Input validation and output integrity

Usage:
    # Run all ML tests
    pytest tests/test_ml.py -v
    
    # Run specific test class
    pytest tests/test_ml.py::TestAutoPipeline -v
    
    # Run with coverage
    pytest tests/test_ml.py --cov=ml --cov-report=html
    
    # Run performance tests
    pytest tests/test_ml.py -m performance
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import pickle
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Test framework imports
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ML pipeline imports - with error handling for missing dependencies
try:
    from ml import (
        auto_analyze_dataset, quick_auto_analysis, create_auto_pipeline,
        get_available_models, validate_installation
    )
    ML_PIPELINE_AVAILABLE = True
except ImportError as e:
    ML_PIPELINE_AVAILABLE = False
    pytest.skip(f"ML pipeline not available: {e}", allow_module_level=True)

try:
    from ml.tabular import TabularMLPipeline, TabularModelRegistry
    TABULAR_AVAILABLE = True
except ImportError:
    TABULAR_AVAILABLE = False

try:
    from ml.timeseries import TimeSeriesMLPipeline, ForecastingPipeline
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False

try:
    from ml.text import TextAnalysisPipeline, SentimentAnalyzer
    TEXT_AVAILABLE = True
except ImportError:
    TEXT_AVAILABLE = False

try:
    from ml.anomaly import AnomalyDetectionPipeline
    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False

try:
    from ml.clustering import ClusteringPipeline
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')

# Test markers
pytestmark = pytest.mark.ml


class TestMLPipelineInstallation:
    """Test ML pipeline installation and dependency validation."""
    
    def test_ml_pipeline_available(self):
        """Test that ML pipeline is properly installed and importable."""
        assert ML_PIPELINE_AVAILABLE, "ML pipeline should be available for testing"
    
    def test_validate_installation(self):
        """Test ML pipeline installation validation."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        status = validate_installation()
        
        assert isinstance(status, dict), "Validation should return dictionary"
        assert 'status' in status, "Status should include overall status"
        assert status['status'] in ['healthy', 'degraded', 'critical'], "Valid status values"
        
        if status['status'] != 'healthy':
            assert 'issues' in status, "Non-healthy status should include issues"
            assert isinstance(status['issues'], list), "Issues should be a list"
    
    def test_get_available_models(self):
        """Test retrieval of available ML models."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        models = get_available_models()
        
        assert isinstance(models, dict), "Available models should be dictionary"
        assert len(models) > 0, "Should have at least some models available"
        
        # Check model categories
        expected_categories = ['classification', 'regression', 'clustering']
        for category in expected_categories:
            if category in models:
                assert isinstance(models[category], list), f"{category} should be a list of models"
                assert len(models[category]) > 0, f"Should have {category} models"


class TestAutoPipeline:
    """Comprehensive tests for the main AutoML pipeline functionality."""
    
    @pytest.fixture
    def classification_dataset(self):
        """Create a sample classification dataset for testing."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=2,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def regression_dataset(self):
        """Create a sample regression dataset for testing."""
        X, y = make_regression(
            n_samples=1000,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def timeseries_dataset(self):
        """Create a sample time series dataset for testing."""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        
        # Generate synthetic time series data
        np.random.seed(42)
        trend = np.linspace(100, 200, 365)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)
        noise = np.random.normal(0, 5, 365)
        values = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'feature_1': np.random.normal(0, 1, 365),
            'feature_2': np.random.uniform(0, 1, 365)
        })
        
        return df
    
    @pytest.fixture
    def text_dataset(self):
        """Create a sample text analysis dataset for testing."""
        texts = [
            "This is a positive review of the product. I really love it!",
            "Terrible product, waste of money. Very disappointed.",
            "Average product, nothing special but does the job.",
            "Excellent quality and fast delivery. Highly recommended!",
            "Poor customer service and defective product received.",
            "Great value for money. Would buy again.",
            "Not what I expected. The description was misleading.",
            "Outstanding product quality. Exceeded my expectations.",
            "Mediocre at best. There are better alternatives available.",
            "Perfect for my needs. Exactly as described."
        ] * 50  # Repeat to get enough samples
        
        sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative'] * 100
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        return df
    
    def test_auto_analyze_dataset_classification(self, classification_dataset):
        """Test auto analysis pipeline with classification dataset."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        # Test basic auto analysis
        result = auto_analyze_dataset(
            df=classification_dataset,
            target_column='target',
            user_id='test_user',
            config={'max_models': 3, 'max_time': 60}
        )
        
        # Validate result structure
        assert isinstance(result, dict), "Result should be dictionary"
        
        # Check required keys
        required_keys = ['dataset_analysis', 'model_results', 'performance_metrics']
        for key in required_keys:
            assert key in result, f"Result should contain '{key}'"
        
        # Validate dataset analysis
        dataset_analysis = result['dataset_analysis']
        assert isinstance(dataset_analysis, dict), "Dataset analysis should be dict"
        assert 'task_type' in dataset_analysis, "Should detect task type"
        assert dataset_analysis['task_type'] == 'classification', "Should detect classification task"
        
        # Validate model results
        model_results = result['model_results']
        assert isinstance(model_results, dict), "Model results should be dict"
        assert 'best_model_name' in model_results, "Should identify best model"
        assert 'model_comparison' in model_results, "Should include model comparison"
        
        # Validate performance metrics
        performance_metrics = result['performance_metrics']
        assert isinstance(performance_metrics, dict), "Performance metrics should be dict"
        
        # Check classification metrics
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in classification_metrics:
            if metric in performance_metrics:
                assert isinstance(performance_metrics[metric], (int, float)), f"{metric} should be numeric"
                assert 0 <= performance_metrics[metric] <= 1, f"{metric} should be between 0 and 1"
    
    def test_auto_analyze_dataset_regression(self, regression_dataset):
        """Test auto analysis pipeline with regression dataset."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        result = auto_analyze_dataset(
            df=regression_dataset,
            target_column='target',
            user_id='test_user',
            config={'max_models': 3, 'max_time': 60}
        )
        
        # Validate result structure
        assert isinstance(result, dict), "Result should be dictionary"
        
        # Check task type detection
        dataset_analysis = result['dataset_analysis']
        assert dataset_analysis['task_type'] == 'regression', "Should detect regression task"
        
        # Check regression metrics
        performance_metrics = result['performance_metrics']
        regression_metrics = ['r2_score', 'rmse', 'mae']
        
        for metric in regression_metrics:
            if metric in performance_metrics:
                assert isinstance(performance_metrics[metric], (int, float)), f"{metric} should be numeric"
                
                # R2 score should be <= 1, RMSE and MAE should be >= 0
                if metric == 'r2_score':
                    assert performance_metrics[metric] <= 1, "R2 score should be <= 1"
                else:  # RMSE, MAE
                    assert performance_metrics[metric] >= 0, f"{metric} should be >= 0"
    
    def test_quick_auto_analysis(self, classification_dataset):
        """Test quick auto analysis functionality."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        result = quick_auto_analysis(
            df=classification_dataset,
            target_column='target'
        )
        
        # Validate result structure
        assert isinstance(result, dict), "Result should be dictionary"
        
        # Should have basic analysis results
        assert 'dataset_summary' in result, "Should include dataset summary"
        assert 'recommended_models' in result, "Should include model recommendations"
        assert 'quick_insights' in result, "Should include quick insights"
        
        # Validate dataset summary
        dataset_summary = result['dataset_summary']
        assert 'n_samples' in dataset_summary, "Should include sample count"
        assert 'n_features' in dataset_summary, "Should include feature count"
        assert 'task_type' in dataset_summary, "Should detect task type"
        
        # Validate model recommendations
        recommended_models = result['recommended_models']
        assert isinstance(recommended_models, list), "Recommendations should be list"
        assert len(recommended_models) > 0, "Should recommend at least one model"
    
    def test_create_auto_pipeline(self):
        """Test auto pipeline creation and configuration."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        # Test with default configuration
        pipeline = create_auto_pipeline()
        assert pipeline is not None, "Should create pipeline object"
        
        # Test with custom configuration
        config = {
            'max_models': 5,
            'max_time': 120,
            'enable_ensemble': True,
            'cross_validation_folds': 3
        }
        
        custom_pipeline = create_auto_pipeline(config)
        assert custom_pipeline is not None, "Should create pipeline with custom config"
    
    @pytest.mark.parametrize("invalid_config", [
        {'max_models': -1},  # Negative value
        {'max_time': 0},     # Zero time
        {'cross_validation_folds': 1},  # Invalid CV folds
        {'enable_ensemble': 'yes'},     # Wrong type
    ])
    def test_auto_pipeline_invalid_config(self, classification_dataset, invalid_config):
        """Test auto pipeline with invalid configurations."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        # Should handle invalid config gracefully
        with pytest.raises((ValueError, TypeError)) or warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = auto_analyze_dataset(
                df=classification_dataset,
                target_column='target',
                user_id='test_user',
                config=invalid_config
            )
            # If no exception, check that result is valid or contains error info
            if isinstance(result, dict):
                assert 'error' in result or 'dataset_analysis' in result
    
    def test_auto_analyze_missing_target(self, classification_dataset):
        """Test auto analysis with missing target column."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        with pytest.raises(ValueError):
            auto_analyze_dataset(
                df=classification_dataset,
                target_column='nonexistent_column',
                user_id='test_user'
            )
    
    def test_auto_analyze_empty_dataset(self):
        """Test auto analysis with empty dataset."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, Exception)):
            auto_analyze_dataset(
                df=empty_df,
                target_column='target',
                user_id='test_user'
            )
    
    def test_auto_analyze_single_column(self):
        """Test auto analysis with dataset having only target column."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        single_col_df = pd.DataFrame({'target': [0, 1, 0, 1, 0]})
        
        with pytest.raises((ValueError, Exception)):
            auto_analyze_dataset(
                df=single_col_df,
                target_column='target',
                user_id='test_user'
            )
    
    def test_auto_analyze_all_missing_values(self):
        """Test auto analysis with dataset containing all missing values."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        missing_df = pd.DataFrame({
            'feature_1': [np.nan] * 100,
            'feature_2': [np.nan] * 100,
            'target': [0, 1] * 50
        })
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, Exception)) or warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = auto_analyze_dataset(
                df=missing_df,
                target_column='target',
                user_id='test_user'
            )
            # If successful, should indicate data quality issues
            if isinstance(result, dict):
                dataset_analysis = result.get('dataset_analysis', {})
                assert 'data_quality_issues' in dataset_analysis or 'warnings' in result


@pytest.mark.skipif(not TABULAR_AVAILABLE, reason="Tabular ML module not available")
class TestTabularMLPipeline:
    """Test suite for tabular data ML pipeline functionality."""
    
    @pytest.fixture
    def tabular_pipeline(self):
        """Create a tabular ML pipeline instance."""
        return TabularMLPipeline()
    
    @pytest.fixture
    def sample_tabular_data(self):
        """Create sample tabular data for testing."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        
        return df, pd.Series(y, name='target')
    
    def test_tabular_pipeline_initialization(self, tabular_pipeline):
        """Test tabular pipeline initialization."""
        assert tabular_pipeline is not None, "Pipeline should be created"
        assert hasattr(tabular_pipeline, 'fit'), "Pipeline should have fit method"
        assert hasattr(tabular_pipeline, 'predict'), "Pipeline should have predict method"
    
    def test_tabular_model_registry(self):
        """Test tabular model registry functionality."""
        registry = TabularModelRegistry()
        
        # Test getting available models
        models = registry.get_available_models('classification')
        assert isinstance(models, dict), "Should return dictionary of models"
        assert len(models) > 0, "Should have classification models"
        
        # Test getting model by name
        model_names = list(models.keys())
        if model_names:
            model = registry.get_model(model_names[0])
            assert model is not None, "Should return model instance"
    
    def test_tabular_pipeline_fit_predict(self, tabular_pipeline, sample_tabular_data):
        """Test fitting and prediction with tabular pipeline."""
        X, y = sample_tabular_data
        
        # Fit the pipeline
        tabular_pipeline.fit(X, y, task_type='classification')
        
        # Make predictions
        predictions = tabular_pipeline.predict(X)
        
        # Validate predictions
        assert len(predictions) == len(X), "Should predict for all samples"
        assert all(pred in [0, 1] for pred in predictions), "Should predict valid classes"
        
        # Test prediction probabilities
        if hasattr(tabular_pipeline, 'predict_proba'):
            probabilities = tabular_pipeline.predict_proba(X)
            assert probabilities.shape == (len(X), 2), "Should return probabilities for all classes"
            assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
    
    def test_tabular_pipeline_evaluation(self, tabular_pipeline, sample_tabular_data):
        """Test tabular pipeline evaluation functionality."""
        X, y = sample_tabular_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit pipeline
        tabular_pipeline.fit(X_train, y_train, task_type='classification')
        
        # Evaluate
        if hasattr(tabular_pipeline, 'evaluate'):
            evaluation_results = tabular_pipeline.evaluate(X_test, y_test)
            
            # Check evaluation results
            assert isinstance(evaluation_results, dict), "Evaluation should return dict"
            
            # Check for common classification metrics
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in expected_metrics:
                if metric in evaluation_results:
                    assert isinstance(evaluation_results[metric], (int, float)), f"{metric} should be numeric"
    
    def test_tabular_pipeline_feature_importance(self, tabular_pipeline, sample_tabular_data):
        """Test feature importance extraction from tabular pipeline."""
        X, y = sample_tabular_data
        
        # Fit pipeline
        tabular_pipeline.fit(X, y, task_type='classification')
        
        # Get feature importance
        if hasattr(tabular_pipeline, 'get_feature_importance'):
            importance = tabular_pipeline.get_feature_importance()
            
            assert isinstance(importance, dict), "Feature importance should be dict"
            assert len(importance) == len(X.columns), "Should have importance for all features"
            
            # Check that all values are numeric
            for feature, imp_value in importance.items():
                assert isinstance(imp_value, (int, float)), f"Importance for {feature} should be numeric"
    
    def test_tabular_pipeline_with_missing_values(self, tabular_pipeline):
        """Test tabular pipeline handling of missing values."""
        # Create data with missing values
        X = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],
            'feature_2': [np.nan, 2, 3, 4, np.nan],
            'feature_3': [1, 2, 3, 4, 5]
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        # Should handle missing values gracefully
        try:
            tabular_pipeline.fit(X, y, task_type='classification')
            predictions = tabular_pipeline.predict(X)
            assert len(predictions) == len(X), "Should predict for all samples despite missing values"
        except Exception as e:
            # If error occurs, it should be a meaningful error
            assert "missing" in str(e).lower() or "nan" in str(e).lower(), "Error should be about missing values"
    
    def test_tabular_pipeline_categorical_features(self, tabular_pipeline):
        """Test tabular pipeline with categorical features."""
        # Create data with categorical features
        X = pd.DataFrame({
            'numeric_feature': [1, 2, 3, 4, 5],
            'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
            'ordinal_feature': ['low', 'medium', 'high', 'low', 'medium']
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        # Should handle categorical features
        try:
            tabular_pipeline.fit(X, y, task_type='classification')
            predictions = tabular_pipeline.predict(X)
            assert len(predictions) == len(X), "Should handle categorical features"
        except Exception as e:
            # If error occurs, should be about categorical handling
            assert any(word in str(e).lower() for word in ['categorical', 'string', 'object']), \
                "Error should be about categorical features"


@pytest.mark.skipif(not TIMESERIES_AVAILABLE, reason="Time series ML module not available")
class TestTimeSeriesMLPipeline:
    """Test suite for time series ML pipeline functionality."""
    
    @pytest.fixture
    def timeseries_pipeline(self):
        """Create a time series ML pipeline instance."""
        return TimeSeriesMLPipeline()
    
    @pytest.fixture
    def forecasting_pipeline(self):
        """Create a forecasting pipeline instance."""
        return ForecastingPipeline()
    
    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        # Generate synthetic time series
        np.random.seed(42)
        trend = np.linspace(100, 150, 200)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 365.25 * 4)
        noise = np.random.normal(0, 2, 200)
        values = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'exog_1': np.random.normal(0, 1, 200),
            'exog_2': np.random.uniform(0, 1, 200)
        })
        
        return df
    
    def test_timeseries_pipeline_initialization(self, timeseries_pipeline):
        """Test time series pipeline initialization."""
        assert timeseries_pipeline is not None, "Pipeline should be created"
        assert hasattr(timeseries_pipeline, 'fit'), "Pipeline should have fit method"
        assert hasattr(timeseries_pipeline, 'forecast'), "Pipeline should have forecast method"
    
    def test_forecasting_pipeline_fit_forecast(self, forecasting_pipeline, sample_timeseries_data):
        """Test fitting and forecasting with time series pipeline."""
        df = sample_timeseries_data
        
        # Fit the pipeline
        forecasting_pipeline.fit(
            df=df,
            time_column='date',
            target_column='value',
            forecast_horizon=30
        )
        
        # Make forecasts
        forecast_result = forecasting_pipeline.forecast(steps=30)
        
        # Validate forecast results
        assert isinstance(forecast_result, dict), "Forecast should return dict"
        assert 'forecasts' in forecast_result, "Should include forecast values"
        assert 'confidence_intervals' in forecast_result or 'predictions' in forecast_result, \
            "Should include predictions or confidence intervals"
        
        forecasts = forecast_result['forecasts']
        assert len(forecasts) == 30, "Should forecast requested number of steps"
        assert all(isinstance(f, (int, float)) for f in forecasts), "Forecasts should be numeric"
    
    def test_timeseries_decomposition(self, timeseries_pipeline, sample_timeseries_data):
        """Test time series decomposition functionality."""
        df = sample_timeseries_data
        
        if hasattr(timeseries_pipeline, 'decompose'):
            decomposition = timeseries_pipeline.decompose(
                df=df,
                time_column='date',
                target_column='value'
            )
            
            # Check decomposition components
            expected_components = ['trend', 'seasonal', 'residual']
            for component in expected_components:
                if component in decomposition:
                    assert len(decomposition[component]) == len(df), \
                        f"{component} should have same length as input data"
    
    def test_timeseries_anomaly_detection(self, timeseries_pipeline, sample_timeseries_data):
        """Test time series anomaly detection."""
        df = sample_timeseries_data
        
        if hasattr(timeseries_pipeline, 'detect_anomalies'):
            anomalies = timeseries_pipeline.detect_anomalies(
                df=df,
                time_column='date',
                target_column='value'
            )
            
            # Validate anomaly detection results
            assert isinstance(anomalies, (list, np.ndarray, pd.Series)), \
                "Anomalies should be list, array, or series"
            
            if len(anomalies) > 0:
                # Check that anomaly indices are valid
                assert all(0 <= idx < len(df) for idx in anomalies), \
                    "Anomaly indices should be within data range"
    
    def test_timeseries_with_missing_dates(self, forecasting_pipeline):
        """Test time series pipeline with missing dates."""
        # Create time series with missing dates
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # Remove some random dates
        np.random.seed(42)
        keep_indices = np.random.choice(100, size=80, replace=False)
        dates = dates[keep_indices]
        
        values = np.random.normal(100, 10, 80)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Should handle missing dates appropriately
        try:
            forecasting_pipeline.fit(
                df=df,
                time_column='date',
                target_column='value',
                forecast_horizon=10
            )
            
            forecast_result = forecasting_pipeline.forecast(steps=10)
            assert 'forecasts' in forecast_result, "Should still produce forecasts"
            
        except Exception as e:
            # If error, should be about missing dates or irregular frequency
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['frequency', 'missing', 'irregular', 'dates']), \
                "Error should be about missing dates or frequency issues"


@pytest.mark.skipif(not TEXT_AVAILABLE, reason="Text analysis ML module not available")
class TestTextAnalysisPipeline:
    """Test suite for text analysis ML pipeline functionality."""
    
    @pytest.fixture
    def text_pipeline(self):
        """Create a text analysis pipeline instance."""
        return TextAnalysisPipeline()
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """Create a sentiment analyzer instance."""
        return SentimentAnalyzer()
    
    @pytest.fixture
    def sample_text_data(self):
        """Create sample text data for testing."""
        texts = [
            "I love this product! It's amazing and works perfectly.",
            "This is terrible. Waste of money and poor quality.",
            "It's okay, nothing special but does what it's supposed to do.",
            "Excellent service and fast delivery. Highly recommend!",
            "Poor customer support. Very disappointing experience.",
            "Great value for the price. Would definitely buy again.",
            "Not as described. The product doesn't match the photos.",
            "Outstanding quality and exceeded my expectations completely.",
            "Average product. There are better alternatives available.",
            "Perfect for my needs. Exactly what I was looking for."
        ] * 20  # Repeat for more samples
        
        labels = ['positive', 'negative', 'neutral', 'positive', 'negative'] * 40
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': labels
        })
        
        return df
    
    def test_text_pipeline_initialization(self, text_pipeline):
        """Test text analysis pipeline initialization."""
        assert text_pipeline is not None, "Pipeline should be created"
        assert hasattr(text_pipeline, 'fit'), "Pipeline should have fit method"
        assert hasattr(text_pipeline, 'predict'), "Pipeline should have predict method"
    
    def test_sentiment_analysis(self, sentiment_analyzer, sample_text_data):
        """Test sentiment analysis functionality."""
        df = sample_text_data
        
        # Fit sentiment analyzer
        sentiment_analyzer.fit(df['text'], df['sentiment'])
        
        # Predict sentiments
        predictions = sentiment_analyzer.predict(df['text'])
        
        # Validate predictions
        assert len(predictions) == len(df), "Should predict for all texts"
        
        expected_sentiments = ['positive', 'negative', 'neutral']
        assert all(pred in expected_sentiments for pred in predictions), \
            "Predictions should be valid sentiment labels"
        
        # Test prediction probabilities
        if hasattr(sentiment_analyzer, 'predict_proba'):
            probabilities = sentiment_analyzer.predict_proba(df['text'])
            assert probabilities.shape[0] == len(df), "Should return probabilities for all texts"
            assert probabilities.shape[1] >= 2, "Should return probabilities for multiple classes"
    
    def test_text_classification_pipeline(self, text_pipeline, sample_text_data):
        """Test general text classification pipeline."""
        df = sample_text_data
        
        # Fit pipeline
        text_pipeline.fit(df['text'], df['sentiment'], task_type='classification')
        
        # Make predictions
        predictions = text_pipeline.predict(df['text'])
        
        # Validate predictions
        assert len(predictions) == len(df), "Should predict for all texts"
        
        # Check feature extraction
        if hasattr(text_pipeline, 'get_feature_importance'):
            feature_importance = text_pipeline.get_feature_importance()
            assert isinstance(feature_importance, dict), "Feature importance should be dict"
            assert len(feature_importance) > 0, "Should have feature importance values"
    
    def test_text_preprocessing(self, text_pipeline):
        """Test text preprocessing functionality."""
        if hasattr(text_pipeline, 'preprocess_text'):
            sample_texts = [
                "This is a SAMPLE text with UPPERCASE and numbers 123!",
                "Another text with special characters @#$% and URLs http://example.com",
                "Text with   multiple   spaces   and\ttabs\nand newlines"
            ]
            
            preprocessed = text_pipeline.preprocess_text(sample_texts)
            
            assert len(preprocessed) == len(sample_texts), "Should preprocess all texts"
            assert all(isinstance(text, str) for text in preprocessed), "Should return strings"
    
    def test_text_feature_extraction(self, text_pipeline, sample_text_data):
        """Test text feature extraction functionality."""
        df = sample_text_data
        
        if hasattr(text_pipeline, 'extract_features'):
            features = text_pipeline.extract_features(df['text'])
            
            # Check feature matrix
            assert features.shape[0] == len(df), "Should extract features for all texts"
            assert features.shape[1] > 0, "Should extract at least some features"
            
            # Features should be numeric
            if hasattr(features, 'toarray'):  # Sparse matrix
                features_dense = features.toarray()
                assert np.all(np.isfinite(features_dense)), "All features should be finite"
            else:  # Dense matrix
                assert np.all(np.isfinite(features)), "All features should be finite"
    
    def test_text_pipeline_empty_texts(self, text_pipeline):
        """Test text pipeline with empty or very short texts."""
        problematic_texts = ["", " ", "a", "  \n  ", "123"]
        labels = ["neutral"] * len(problematic_texts)
        
        try:
            text_pipeline.fit(problematic_texts, labels, task_type='classification')
            predictions = text_pipeline.predict(problematic_texts)
            assert len(predictions) == len(problematic_texts), "Should handle problematic texts"
            
        except Exception as e:
            # Should be a meaningful error about text processing
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['text', 'empty', 'short', 'feature']), \
                "Error should be about text processing issues"
    
    def test_text_pipeline_multilingual(self, text_pipeline):
        """Test text pipeline with multilingual content (if supported)."""
        multilingual_texts = [
            "This is English text",
            "Esto es texto en español",
            "Ceci est du texte français",
            "Dies ist deutscher Text",
            "これは日本語のテキストです"
        ]
        labels = ["neutral"] * len(multilingual_texts)
        
        try:
            text_pipeline.fit(multilingual_texts, labels, task_type='classification')
            predictions = text_pipeline.predict(multilingual_texts)
            
            # Should handle multilingual text (may not work well, but shouldn't crash)
            assert len(predictions) == len(multilingual_texts), "Should process all texts"
            
        except Exception as e:
            # If error occurs, should be about encoding or language processing
            error_msg = str(e).lower()
            valid_errors = ['encoding', 'unicode', 'character', 'language', 'text']
            assert any(word in error_msg for word in valid_errors), \
                "Error should be about text processing or encoding"


@pytest.mark.skipif(not ANOMALY_AVAILABLE, reason="Anomaly detection module not available")
class TestAnomalyDetectionPipeline:
    """Test suite for anomaly detection ML pipeline functionality."""
    
    @pytest.fixture
    def anomaly_pipeline(self):
        """Create an anomaly detection pipeline instance."""
        return AnomalyDetectionPipeline()
    
    @pytest.fixture
    def normal_data(self):
        """Create normal data without anomalies."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (1000, 5))
        return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
    
    @pytest.fixture
    def data_with_anomalies(self):
        """Create data with injected anomalies."""
        np.random.seed(42)
        
        # Normal data
        normal_data = np.random.normal(0, 1, (900, 5))
        
        # Anomalous data (outliers)
        anomalous_data = np.random.normal(5, 2, (100, 5))
        
        # Combine data
        all_data = np.vstack([normal_data, anomalous_data])
        df = pd.DataFrame(all_data, columns=[f'feature_{i}' for i in range(5)])
        
        # Create labels (1 for anomaly, 0 for normal)
        labels = np.hstack([np.zeros(900), np.ones(100)])
        
        return df, labels
    
    def test_anomaly_pipeline_initialization(self, anomaly_pipeline):
        """Test anomaly detection pipeline initialization."""
        assert anomaly_pipeline is not None, "Pipeline should be created"
        assert hasattr(anomaly_pipeline, 'fit'), "Pipeline should have fit method"
        assert hasattr(anomaly_pipeline, 'predict'), "Pipeline should have predict method"
    
    def test_anomaly_detection_unsupervised(self, anomaly_pipeline, data_with_anomalies):
        """Test unsupervised anomaly detection."""
        X, true_labels = data_with_anomalies
        
        # Fit pipeline (unsupervised)
        anomaly_pipeline.fit(X, method='unsupervised')
        
        # Detect anomalies
        anomaly_scores = anomaly_pipeline.predict(X)
        
        # Validate results
        assert len(anomaly_scores) == len(X), "Should return scores for all samples"
        assert all(isinstance(score, (int, float)) for score in anomaly_scores), \
            "Anomaly scores should be numeric"
        
        # Get binary predictions
        if hasattr(anomaly_pipeline, 'predict_binary'):
            binary_predictions = anomaly_pipeline.predict_binary(X)
            assert len(binary_predictions) == len(X), "Should return binary predictions for all samples"
            assert all(pred in [0, 1] for pred in binary_predictions), \
                "Binary predictions should be 0 or 1"
            
            # Check that some anomalies are detected
            anomaly_count = sum(binary_predictions)
            assert 0 < anomaly_count < len(X), "Should detect some but not all samples as anomalies"
    
    def test_anomaly_detection_supervised(self, anomaly_pipeline, data_with_anomalies):
        """Test supervised anomaly detection (if supported)."""
        X, y = data_with_anomalies
        
        try:
            # Fit pipeline with labels
            anomaly_pipeline.fit(X, y, method='supervised')
            
            # Make predictions
            predictions = anomaly_pipeline.predict(X)
            
            # Validate predictions
            assert len(predictions) == len(X), "Should predict for all samples"
            
            # Check prediction accuracy
            if hasattr(anomaly_pipeline, 'predict_binary'):
                binary_predictions = anomaly_pipeline.predict_binary(X)
                accuracy = sum(binary_predictions == y) / len(y)
                assert accuracy > 0.5, "Should achieve better than random accuracy"
                
        except NotImplementedError:
            pytest.skip("Supervised anomaly detection not implemented")
        except Exception as e:
            # Should be a meaningful error
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['supervised', 'labels', 'method']), \
                "Error should be about supervised learning"
    
    def test_anomaly_detection_threshold_tuning(self, anomaly_pipeline, data_with_anomalies):
        """Test anomaly detection threshold tuning."""
        X, y = data_with_anomalies
        
        # Fit pipeline
        anomaly_pipeline.fit(X, method='unsupervised')
        
        # Test different thresholds
        if hasattr(anomaly_pipeline, 'set_threshold'):
            thresholds = [0.1, 0.05, 0.01]
            
            for threshold in thresholds:
                anomaly_pipeline.set_threshold(threshold)
                binary_predictions = anomaly_pipeline.predict_binary(X)
                
                anomaly_rate = sum(binary_predictions) / len(binary_predictions)
                
                # Lower threshold should detect more anomalies
                assert 0 < anomaly_rate <= 1, f"Anomaly rate should be between 0 and 1 for threshold {threshold}"
    
    def test_anomaly_detection_feature_importance(self, anomaly_pipeline, data_with_anomalies):
        """Test feature importance for anomaly detection."""
        X, y = data_with_anomalies
        
        # Fit pipeline
        anomaly_pipeline.fit(X, method='unsupervised')
        
        # Get feature importance
        if hasattr(anomaly_pipeline, 'get_feature_importance'):
            importance = anomaly_pipeline.get_feature_importance()
            
            assert isinstance(importance, dict), "Feature importance should be dict"
            assert len(importance) == X.shape[1], "Should have importance for all features"
            
            # All importance values should be numeric and non-negative
            for feature, imp_value in importance.items():
                assert isinstance(imp_value, (int, float)), f"Importance for {feature} should be numeric"
                assert imp_value >= 0, f"Importance for {feature} should be non-negative"
    
    def test_anomaly_detection_normal_data_only(self, anomaly_pipeline, normal_data):
        """Test anomaly detection on purely normal data."""
        X = normal_data
        
        # Fit pipeline
        anomaly_pipeline.fit(X, method='unsupervised')
        
        # Predict on normal data
        if hasattr(anomaly_pipeline, 'predict_binary'):
            binary_predictions = anomaly_pipeline.predict_binary(X)
            
            # Should detect very few anomalies in normal data
            anomaly_rate = sum(binary_predictions) / len(binary_predictions)
            assert anomaly_rate < 0.1, "Should detect few anomalies in normal data"


@pytest.mark.skipif(not CLUSTERING_AVAILABLE, reason="Clustering module not available")
class TestClusteringPipeline:
    """Test suite for clustering ML pipeline functionality."""
    
    @pytest.fixture
    def clustering_pipeline(self):
        """Create a clustering pipeline instance."""
        return ClusteringPipeline()
    
    @pytest.fixture
    def clusterable_data(self):
        """Create data with natural clusters."""
        np.random.seed(42)
        
        # Create 3 distinct clusters
        cluster1 = np.random.normal([0, 0], 0.5, (100, 2))
        cluster2 = np.random.normal([3, 3], 0.5, (100, 2))
        cluster3 = np.random.normal([0, 3], 0.5, (100, 2))
        
        data = np.vstack([cluster1, cluster2, cluster3])
        true_labels = np.hstack([np.zeros(100), np.ones(100), np.full(100, 2)])
        
        df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
        
        return df, true_labels
    
    def test_clustering_pipeline_initialization(self, clustering_pipeline):
        """Test clustering pipeline initialization."""
        assert clustering_pipeline is not None, "Pipeline should be created"
        assert hasattr(clustering_pipeline, 'fit'), "Pipeline should have fit method"
        assert hasattr(clustering_pipeline, 'predict'), "Pipeline should have predict method"
    
    def test_kmeans_clustering(self, clustering_pipeline, clusterable_data):
        """Test K-means clustering functionality."""
        X, true_labels = clusterable_data
        
        # Fit clustering pipeline
        clustering_pipeline.fit(X, n_clusters=3, algorithm='kmeans')
        
        # Get cluster assignments
        cluster_labels = clustering_pipeline.predict(X)
        
        # Validate cluster assignments
        assert len(cluster_labels) == len(X), "Should assign cluster to all samples"
        assert len(set(cluster_labels)) <= 3, "Should not create more clusters than requested"
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels), \
            "Cluster labels should be integers"
        
        # Check cluster centers
        if hasattr(clustering_pipeline, 'get_cluster_centers'):
            centers = clustering_pipeline.get_cluster_centers()
            assert len(centers) <= 3, "Should not have more centers than clusters"
            assert all(len(center) == X.shape[1] for center in centers), \
                "Centers should have same dimensionality as data"
    
    def test_optimal_cluster_number(self, clustering_pipeline, clusterable_data):
        """Test automatic optimal cluster number detection."""
        X, true_labels = clusterable_data
        
        if hasattr(clustering_pipeline, 'find_optimal_clusters'):
            optimal_k = clustering_pipeline.find_optimal_clusters(X, max_clusters=10)
            
            assert isinstance(optimal_k, int), "Optimal clusters should be integer"
            assert 1 <= optimal_k <= 10, "Optimal clusters should be within valid range"
            
            # For our test data, should find around 3 clusters
            assert 2 <= optimal_k <= 5, "Should find reasonable number of clusters for test data"
    
    def test_clustering_evaluation_metrics(self, clustering_pipeline, clusterable_data):
        """Test clustering evaluation metrics."""
        X, true_labels = clusterable_data
        
        # Fit clustering
        clustering_pipeline.fit(X, n_clusters=3, algorithm='kmeans')
        cluster_labels = clustering_pipeline.predict(X)
        
        # Calculate evaluation metrics
        if hasattr(clustering_pipeline, 'evaluate'):
            metrics = clustering_pipeline.evaluate(X, cluster_labels, true_labels)
            
            assert isinstance(metrics, dict), "Metrics should be dictionary"
            
            # Check for common clustering metrics
            expected_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            for metric in expected_metrics:
                if metric in metrics:
                    assert isinstance(metrics[metric], (int, float)), f"{metric} should be numeric"
    
    def test_hierarchical_clustering(self, clustering_pipeline, clusterable_data):
        """Test hierarchical clustering if available."""
        X, true_labels = clusterable_data
        
        try:
            # Fit hierarchical clustering
            clustering_pipeline.fit(X, n_clusters=3, algorithm='hierarchical')
            cluster_labels = clustering_pipeline.predict(X)
            
            # Validate results
            assert len(cluster_labels) == len(X), "Should assign cluster to all samples"
            assert len(set(cluster_labels)) <= 3, "Should not create more clusters than requested"
            
            # Check dendrogram if available
            if hasattr(clustering_pipeline, 'get_dendrogram'):
                dendrogram_data = clustering_pipeline.get_dendrogram()
                assert isinstance(dendrogram_data, dict), "Dendrogram data should be dictionary"
                
        except (NotImplementedError, AttributeError):
            pytest.skip("Hierarchical clustering not available")
    
    def test_dbscan_clustering(self, clustering_pipeline, clusterable_data):
        """Test DBSCAN clustering if available."""
        X, true_labels = clusterable_data
        
        try:
            # Fit DBSCAN clustering
            clustering_pipeline.fit(X, algorithm='dbscan', eps=0.5, min_samples=5)
            cluster_labels = clustering_pipeline.predict(X)
            
            # Validate results
            assert len(cluster_labels) == len(X), "Should assign cluster to all samples"
            
            # DBSCAN can have noise points (label -1)
            unique_labels = set(cluster_labels)
            assert -1 in unique_labels or len(unique_labels) > 0, \
                "Should find clusters or noise points"
            
        except (NotImplementedError, AttributeError):
            pytest.skip("DBSCAN clustering not available")
    
    def test_clustering_with_noise(self, clustering_pipeline):
        """Test clustering with noisy data."""
        np.random.seed(42)
        
        # Create data with noise
        signal = np.random.normal([1, 1], 0.3, (200, 2))
        noise = np.random.uniform(-3, 3, (50, 2))
        
        noisy_data = np.vstack([signal, noise])
        df = pd.DataFrame(noisy_data, columns=['feature_1', 'feature_2'])
        
        # Should handle noisy data without crashing
        try:
            clustering_pipeline.fit(df, n_clusters=2, algorithm='kmeans')
            cluster_labels = clustering_pipeline.predict(df)
            
            assert len(cluster_labels) == len(df), "Should handle noisy data"
            
        except Exception as e:
            # If error occurs, should be meaningful
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['cluster', 'convergence', 'data']), \
                "Error should be about clustering issues"


@pytest.mark.performance
class TestMLPerformance:
    """Performance tests for ML pipeline components."""
    
    def test_training_time_reasonable(self, tmp_path):
        """Test that training completes within reasonable time."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        # Create moderately sized dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        # Time the training
        start_time = time.time()
        
        result = auto_analyze_dataset(
            df=df,
            target_column='target',
            user_id='performance_test',
            config={'max_models': 3, 'max_time': 180}  # 3 minutes max
        )
        
        training_time = time.time() - start_time
        
        # Should complete within reasonable time (5 minutes)
        assert training_time < 300, f"Training took too long: {training_time:.1f} seconds"
        
        # Should produce valid results
        assert isinstance(result, dict), "Should return valid results"
        assert 'performance_metrics' in result, "Should include performance metrics"
    
    @pytest.mark.skipif(not TABULAR_AVAILABLE, reason="Tabular module not available")
    def test_memory_usage_reasonable(self):
        """Test that ML pipeline doesn't consume excessive memory."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        X, y = make_classification(
            n_samples=10000,
            n_features=50,
            n_informative=25,
            random_state=42
        )
        
        # Train model
        pipeline = TabularMLPipeline()
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        target = pd.Series(y, name='target')
        
        pipeline.fit(df, target, task_type='classification')
        
        # Check memory usage after training
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not consume more than 1GB additional memory
        assert memory_increase < 1024, f"Memory increase too high: {memory_increase:.1f} MB"
    
    def test_batch_prediction_performance(self):
        """Test performance of batch predictions."""
        if not TABULAR_AVAILABLE:
            pytest.skip("Tabular module not available")
        
        # Create training data
        X_train, y_train = make_classification(
            n_samples=1000,
            n_features=10,
            random_state=42
        )
        
        # Create prediction data
        X_pred, _ = make_classification(
            n_samples=10000,  # Large prediction batch
            n_features=10,
            random_state=43
        )
        
        # Train model
        pipeline = TabularMLPipeline()
        df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        target_train = pd.Series(y_train, name='target')
        
        pipeline.fit(df_train, target_train, task_type='classification')
        
        # Time batch prediction
        df_pred = pd.DataFrame(X_pred, columns=[f'feature_{i}' for i in range(X_pred.shape[1])])
        
        start_time = time.time()
        predictions = pipeline.predict(df_pred)
        prediction_time = time.time() - start_time
        
        # Should predict reasonably fast (< 10 seconds for 10k samples)
        assert prediction_time < 10, f"Batch prediction too slow: {prediction_time:.2f} seconds"
        assert len(predictions) == len(df_pred), "Should predict for all samples"


class TestMLIntegration:
    """Integration tests for ML pipeline components working together."""
    
    @pytest.fixture
    def integration_dataset(self):
        """Create a comprehensive dataset for integration testing."""
        np.random.seed(42)
        
        # Create mixed data types
        n_samples = 1000
        
        # Numeric features
        numeric_features = np.random.normal(0, 1, (n_samples, 5))
        
        # Categorical features
        categories = ['A', 'B', 'C', 'D']
        categorical_feature = np.random.choice(categories, n_samples)
        
        # Binary feature
        binary_feature = np.random.choice([0, 1], n_samples)
        
        # Create target based on features (for meaningful relationships)
        target = (
            numeric_features[:, 0] * 2 + 
            numeric_features[:, 1] * 1.5 + 
            (categorical_feature == 'A').astype(int) * 3 +
            binary_feature * 2 +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Convert to classification target
        target_binary = (target > np.median(target)).astype(int)
        
        df = pd.DataFrame({
            'numeric_1': numeric_features[:, 0],
            'numeric_2': numeric_features[:, 1],
            'numeric_3': numeric_features[:, 2],
            'numeric_4': numeric_features[:, 3],
            'numeric_5': numeric_features[:, 4],
            'categorical': categorical_feature,
            'binary': binary_feature,
            'target_regression': target,
            'target_classification': target_binary
        })
        
        return df
    
    def test_end_to_end_classification_pipeline(self, integration_dataset):
        """Test complete end-to-end classification pipeline."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        df = integration_dataset.copy()
        
        # Run complete analysis
        result = auto_analyze_dataset(
            df=df,
            target_column='target_classification',
            user_id='integration_test',
            config={'max_models': 2, 'max_time': 120}
        )
        
        # Validate complete pipeline result
        assert isinstance(result, dict), "Should return dictionary result"
        
        # Check all major components
        required_sections = [
            'dataset_analysis',
            'model_results', 
            'performance_metrics',
            'feature_importance'
        ]
        
        for section in required_sections:
            assert section in result, f"Result should contain {section}"
            assert isinstance(result[section], dict), f"{section} should be dictionary"
        
        # Validate dataset analysis
        dataset_analysis = result['dataset_analysis']
        assert dataset_analysis['task_type'] == 'classification', "Should detect classification task"
        assert dataset_analysis['n_features'] > 0, "Should count features"
        assert dataset_analysis['n_samples'] > 0, "Should count samples"
        
        # Validate model results
        model_results = result['model_results']
        assert 'best_model_name' in model_results, "Should identify best model"
        assert 'model_comparison' in model_results, "Should compare models"
        
        # Validate performance metrics
        performance_metrics = result['performance_metrics']
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Should have at least one classification metric
        has_classification_metric = any(metric in performance_metrics for metric in classification_metrics)
        assert has_classification_metric, "Should include classification metrics"
        
        # Validate feature importance
        feature_importance = result['feature_importance']
        assert len(feature_importance) > 0, "Should calculate feature importance"
        
        # Feature importance should include original features
        original_features = ['numeric_1', 'numeric_2', 'categorical', 'binary']
        feature_names = list(feature_importance.keys())
        
        # Some original features should appear in importance (after preprocessing)
        has_original_features = any(
            any(orig_feat in feat_name for orig_feat in original_features)
            for feat_name in feature_names
        )
        assert has_original_features, "Feature importance should relate to original features"
    
    def test_end_to_end_regression_pipeline(self, integration_dataset):
        """Test complete end-to-end regression pipeline."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        df = integration_dataset.copy()
        
        # Run regression analysis
        result = auto_analyze_dataset(
            df=df,
            target_column='target_regression',
            user_id='integration_test',
            config={'max_models': 2, 'max_time': 120}
        )
        
        # Validate regression-specific results
        dataset_analysis = result['dataset_analysis']
        assert dataset_analysis['task_type'] == 'regression', "Should detect regression task"
        
        performance_metrics = result['performance_metrics']
        regression_metrics = ['r2_score', 'rmse', 'mae']
        
        # Should have regression metrics
        has_regression_metric = any(metric in performance_metrics for metric in regression_metrics)
        assert has_regression_metric, "Should include regression metrics"
        
        # R2 score should be reasonable for our synthetic data
        if 'r2_score' in performance_metrics:
            r2 = performance_metrics['r2_score']
            assert -1 <= r2 <= 1, "R2 score should be in valid range"
            # Should achieve decent performance on synthetic data
            assert r2 > 0.5, f"R2 score should be reasonable: {r2:.3f}"
    
    def test_pipeline_with_missing_values(self):
        """Test pipeline handling of missing values in integration scenario."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        # Create dataset with missing values
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        # Introduce missing values
        np.random.seed(42)
        
        # Random missing values
        for col in df.columns[:-1]:  # Exclude target
            missing_mask = np.random.random(len(df)) < 0.1  # 10% missing
            df.loc[missing_mask, col] = np.nan
        
        # Complete missing column
        df['feature_missing'] = np.nan
        
        # Should handle missing values gracefully
        result = auto_analyze_dataset(
            df=df,
            target_column='target',
            user_id='missing_values_test',
            config={'max_models': 2, 'max_time': 60}
        )
        
        # Should still produce results
        assert isinstance(result, dict), "Should handle missing values and return results"
        assert 'performance_metrics' in result, "Should still calculate performance metrics"
        
        # Check if missing value handling is noted
        dataset_analysis = result.get('dataset_analysis', {})
        if 'data_quality_issues' in dataset_analysis:
            issues = dataset_analysis['data_quality_issues']
            assert any('missing' in str(issue).lower() for issue in issues), \
                "Should identify missing value issues"
    
    def test_pipeline_error_recovery(self):
        """Test pipeline error recovery and graceful failure handling."""
        if not ML_PIPELINE_AVAILABLE:
            pytest.skip("ML pipeline not available")
        
        # Test with problematic dataset
        problematic_df = pd.DataFrame({
            'constant_feature': [1] * 100,  # No variance
            'target': [0, 1] * 50
        })
        
        # Should handle gracefully or provide meaningful error
        try:
            result = auto_analyze_dataset(
                df=problematic_df,
                target_column='target',
                user_id='error_test',
                config={'max_models': 1, 'max_time': 30}
            )
            
            # If successful, should indicate issues
            if isinstance(result, dict):
                # Should either have results with warnings or error information
                has_warnings = 'warnings' in result or 'data_quality_issues' in result.get('dataset_analysis', {})
                has_results = 'performance_metrics' in result
                
                assert has_warnings or has_results, "Should either produce results or indicate issues"
        
        except Exception as e:
            # If exception occurs, should be meaningful
            error_msg = str(e).lower()
            meaningful_errors = ['feature', 'variance', 'data', 'constant', 'training']
            assert any(word in error_msg for word in meaningful_errors), \
                f"Error should be meaningful: {str(e)}"


if __name__ == "__main__":
    # Run tests when file is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
