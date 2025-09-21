"""
Comprehensive pytest configuration and fixtures for Auto-Analyst backend testing.

This conftest.py file provides shared pytest fixtures, configurations, and utilities
for testing the Auto-Analyst backend application. It includes database setup,
authentication mocking, test data generation, and service mocking.

Fixtures Provided:
- FastAPI test client setup
- Database session management (test DB)
- Authentication and user management
- Sample datasets and file fixtures
- Service mocking (ML, data, insights)
- Temporary file and directory management
- Configuration and environment setup
- Cleanup and teardown utilities

Usage:
    Fixtures are automatically available in all test files:
    
    def test_example(client, auth_headers, sample_dataset):
        response = client.get("/api/endpoint", headers=auth_headers)
        assert response.status_code == 200

Dependencies:
- pytest: Core testing framework
- FastAPI: Web framework and test client
- SQLAlchemy: Database ORM and testing
- pandas: Data manipulation for test datasets
- numpy: Numerical operations
- tempfile: Temporary file management
- unittest.mock: Service mocking
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Iterator
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import warnings

# Core testing imports
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from fastapi import FastAPI
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Backend imports with error handling
try:
    from backend.main import app
    from backend.models import database, schemas
    from backend.models.database import Base, get_db
    from backend.core.config import settings
    from backend.services import (
        DataService, MLService, InsightsService, MLOpsService,
        get_data_service, get_ml_service, get_insights_service, get_mlops_service
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    # Create mock app for testing when backend is not available
    app = FastAPI(title="Mock Auto-Analyst API")
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "Mock API for testing"}

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Test database configuration
TEST_DATABASE_URL = "sqlite:///:memory:"
TEMP_FILE_PREFIX = "auto_analyst_test_"


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "api: mark test as API endpoint test")
    config.addinivalue_line("markers", "ml: mark test as ML pipeline test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "slow: mark test as slow running test")
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during testing


def pytest_sessionstart(session):
    """Actions to perform at the start of the test session."""
    print("\n🧪 Starting Auto-Analyst Backend Test Session")
    print("=" * 60)
    
    # Create temporary directories for testing
    temp_base = Path(tempfile.gettempdir()) / "auto_analyst_tests"
    temp_base.mkdir(exist_ok=True)
    
    # Set environment variables for test paths
    os.environ["TEST_TEMP_DIR"] = str(temp_base)
    os.environ["TEST_UPLOAD_DIR"] = str(temp_base / "uploads")
    os.environ["TEST_MODELS_DIR"] = str(temp_base / "models")
    
    # Create test directories
    for dir_path in [temp_base / "uploads", temp_base / "models", temp_base / "cache"]:
        dir_path.mkdir(exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    """Actions to perform at the end of the test session."""
    print(f"\n🏁 Test Session Completed (Exit Status: {exitstatus})")
    
    # Cleanup temporary directories
    temp_base = Path(tempfile.gettempdir()) / "auto_analyst_tests"
    if temp_base.exists():
        try:
            shutil.rmtree(temp_base)
            print("✅ Cleaned up temporary test files")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp files: {e}")


# Database Fixtures
@pytest.fixture(scope="session")
def test_engine():
    """Create a test database engine for the entire test session."""
    if not BACKEND_AVAILABLE:
        return None
    
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_engine) -> Generator[Session, None, None]:
    """
    Create a test database session for individual test functions.
    
    This fixture provides a clean database session for each test, ensuring
    test isolation and preventing data leakage between tests.
    
    Yields:
        Database session for testing
    """
    if not BACKEND_AVAILABLE or not test_engine:
        yield None
        return
    
    # Create a new session for each test
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    
    # Begin a transaction
    transaction = session.begin()
    
    try:
        yield session
    finally:
        # Rollback transaction and close session
        transaction.rollback()
        session.close()


@pytest.fixture
def override_get_db(test_db_session):
    """
    Override the database dependency for testing.
    
    This fixture replaces the production database with the test database
    for all FastAPI dependency injection.
    """
    if not BACKEND_AVAILABLE:
        return None
    
    def _override_get_db():
        try:
            yield test_db_session
        finally:
            pass
    
    return _override_get_db


# FastAPI Client Fixtures
@pytest.fixture
def client(override_get_db) -> TestClient:
    """
    Create a FastAPI test client.
    
    This fixture provides a test client for making HTTP requests to the
    FastAPI application with the test database injected.
    
    Returns:
        TestClient instance for API testing
    """
    if BACKEND_AVAILABLE and override_get_db:
        # Override database dependency
        app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clear dependency overrides
    if BACKEND_AVAILABLE:
        app.dependency_overrides.clear()


@pytest.fixture
def authenticated_client(client, test_user, auth_token) -> TestClient:
    """
    Create an authenticated FastAPI test client.
    
    This fixture provides a test client with authentication headers
    pre-configured for testing protected endpoints.
    
    Returns:
        TestClient with authentication configured
    """
    # Set default authentication headers
    client.headers.update({"Authorization": f"Bearer {auth_token}"})
    return client


# Authentication Fixtures
@pytest.fixture
def test_user() -> Dict[str, Any]:
    """
    Create a test user for authentication testing.
    
    Returns:
        Dictionary representing a test user
    """
    return {
        "id": 1,
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "is_active": True,
        "is_verified": True,
        "is_superuser": False,
        "created_at": datetime.now(),
        "last_login": datetime.now()
    }


@pytest.fixture
def admin_user() -> Dict[str, Any]:
    """
    Create an admin test user.
    
    Returns:
        Dictionary representing an admin user
    """
    return {
        "id": 2,
        "email": "admin@example.com", 
        "username": "admin",
        "full_name": "Admin User",
        "is_active": True,
        "is_verified": True,
        "is_superuser": True,
        "created_at": datetime.now(),
        "last_login": datetime.now()
    }


@pytest.fixture
def auth_token() -> str:
    """
    Create a mock authentication token.
    
    Returns:
        Mock JWT token string
    """
    return "test_jwt_token_12345"


@pytest.fixture
def auth_headers(auth_token) -> Dict[str, str]:
    """
    Create authentication headers for API requests.
    
    Args:
        auth_token: Authentication token
        
    Returns:
        Dictionary with authentication headers
    """
    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }


@pytest.fixture
def admin_headers(admin_user, auth_token) -> Dict[str, str]:
    """
    Create admin authentication headers.
    
    Returns:
        Dictionary with admin authentication headers
    """
    return {
        "Authorization": f"Bearer admin_{auth_token}",
        "Content-Type": "application/json",
        "X-User-Role": "admin"
    }


# Data Fixtures
@pytest.fixture
def sample_classification_dataset() -> pd.DataFrame:
    """
    Create a sample classification dataset for testing.
    
    Returns:
        Pandas DataFrame with classification data
    """
    np.random.seed(42)  # For reproducible tests
    
    n_samples = 1000
    n_features = 5
    
    # Generate synthetic features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create target based on features (for realistic relationships)
    target = (
        X[:, 0] * 2 + 
        X[:, 1] * 1.5 + 
        X[:, 2] * 0.5 + 
        np.random.normal(0, 0.5, n_samples)
    )
    y = (target > np.median(target)).astype(int)
    
    # Create DataFrame
    data = {
        **{f'feature_{i+1}': X[:, i] for i in range(n_features)},
        'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
        'binary_feature': np.random.choice([0, 1], n_samples),
        'target': y
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_dataset() -> pd.DataFrame:
    """
    Create a sample regression dataset for testing.
    
    Returns:
        Pandas DataFrame with regression data
    """
    np.random.seed(43)  # Different seed for variety
    
    n_samples = 800
    n_features = 4
    
    # Generate features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create continuous target
    target = (
        X[:, 0] * 3 + 
        X[:, 1] * 2 + 
        X[:, 2] * 1 + 
        X[:, 3] * 0.5 + 
        np.random.normal(0, 0.8, n_samples)
    )
    
    data = {
        **{f'numeric_feature_{i+1}': X[:, i] for i in range(n_features)},
        'categorical_col': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'target_value': target
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_timeseries_dataset() -> pd.DataFrame:
    """
    Create a sample time series dataset for testing.
    
    Returns:
        Pandas DataFrame with time series data
    """
    np.random.seed(44)
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=365*2, freq='D')  # 2 years of daily data
    
    # Generate time series with trend, seasonality, and noise
    trend = np.linspace(100, 200, len(dates))
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 10, len(dates))
    
    values = trend + seasonal + weekly + noise
    
    # Add external features
    external_feature_1 = np.random.normal(50, 15, len(dates))
    external_feature_2 = np.random.uniform(0, 100, len(dates))
    
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'external_1': external_feature_1,
        'external_2': external_feature_2,
        'day_of_week': dates.dayofweek,
        'month': dates.month
    })


@pytest.fixture
def sample_text_dataset() -> pd.DataFrame:
    """
    Create a sample text dataset for NLP testing.
    
    Returns:
        Pandas DataFrame with text data
    """
    texts = [
        "This product is absolutely amazing! I love it so much.",
        "Terrible quality and poor customer service. Very disappointed.",
        "It's okay, nothing special but gets the job done.",
        "Excellent value for money. Highly recommended!",
        "Worst purchase ever. Complete waste of money.",
        "Good product, fast delivery, satisfied with purchase.",
        "Average quality, could be better for the price.",
        "Outstanding! Exceeded all my expectations completely.",
        "Not bad, but there are better alternatives available.",
        "Perfect for what I needed. Great customer support too."
    ] * 50  # Repeat for sufficient data
    
    sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative'] * 100
    categories = ['electronics', 'clothing', 'books', 'home', 'sports'] * 100
    
    return pd.DataFrame({
        'text': texts,
        'sentiment': sentiments,
        'category': categories,
        'length': [len(text) for text in texts],
        'word_count': [len(text.split()) for text in texts]
    })


@pytest.fixture
def corrupted_dataset() -> pd.DataFrame:
    """
    Create a dataset with data quality issues for testing error handling.
    
    Returns:
        Pandas DataFrame with various data quality problems
    """
    np.random.seed(45)
    
    n_samples = 200
    
    # Create data with various issues
    data = {
        'feature_with_nulls': [np.random.normal() if np.random.random() > 0.3 else np.nan for _ in range(n_samples)],
        'constant_feature': [1] * n_samples,  # No variance
        'mostly_missing': [np.random.normal() if np.random.random() > 0.95 else np.nan for _ in range(n_samples)],
        'high_cardinality': [f'category_{i}' for i in range(n_samples)],  # Too many categories
        'mixed_types': [np.random.choice([1, 2, 'string', None]) for _ in range(n_samples)],
        'outliers': [np.random.normal() if np.random.random() > 0.05 else np.random.normal(0, 50) for _ in range(n_samples)],
        'target': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


# File Fixtures
@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test file operations.
    
    Yields:
        Path to temporary directory that will be cleaned up after test
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_FILE_PREFIX))
    
    try:
        yield temp_dir
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv_file(temp_directory, sample_classification_dataset) -> tuple[Path, str]:
    """
    Create a sample CSV file for upload testing.
    
    Args:
        temp_directory: Temporary directory fixture
        sample_classification_dataset: Sample dataset fixture
        
    Returns:
        Tuple of (file_path, filename)
    """
    filename = "test_dataset.csv"
    file_path = temp_directory / filename
    
    # Save dataset to CSV
    sample_classification_dataset.to_csv(file_path, index=False)
    
    return file_path, filename


@pytest.fixture
def sample_excel_file(temp_directory, sample_regression_dataset) -> tuple[Path, str]:
    """
    Create a sample Excel file for upload testing.
    
    Args:
        temp_directory: Temporary directory fixture
        sample_regression_dataset: Sample dataset fixture
        
    Returns:
        Tuple of (file_path, filename)
    """
    filename = "test_dataset.xlsx"
    file_path = temp_directory / filename
    
    # Save dataset to Excel
    try:
        sample_regression_dataset.to_excel(file_path, index=False)
    except ImportError:
        # If openpyxl not available, skip
        pytest.skip("openpyxl not available for Excel testing")
    
    return file_path, filename


@pytest.fixture
def sample_json_file(temp_directory, sample_text_dataset) -> tuple[Path, str]:
    """
    Create a sample JSON file for upload testing.
    
    Args:
        temp_directory: Temporary directory fixture
        sample_text_dataset: Sample dataset fixture
        
    Returns:
        Tuple of (file_path, filename)
    """
    filename = "test_dataset.json"
    file_path = temp_directory / filename
    
    # Save dataset to JSON
    sample_text_dataset.to_json(file_path, orient='records', indent=2)
    
    return file_path, filename


@pytest.fixture
def invalid_file(temp_directory) -> tuple[Path, str]:
    """
    Create an invalid file for error testing.
    
    Args:
        temp_directory: Temporary directory fixture
        
    Returns:
        Tuple of (file_path, filename)
    """
    filename = "invalid_file.txt"
    file_path = temp_directory / filename
    
    # Create file with invalid content
    with open(file_path, 'w') as f:
        f.write("This is not a valid dataset file.\nIt contains random text.")
    
    return file_path, filename


# Service Mock Fixtures
@pytest.fixture
def mock_data_service():
    """
    Create a mock DataService for testing.
    
    Returns:
        Mock DataService instance
    """
    mock_service = Mock(spec=DataService)
    
    # Configure common mock returns
    mock_service.load_dataset.return_value = Mock(
        dataframe=pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]}),
        info=Mock(
            n_rows=3,
            n_columns=2,
            column_names=['feature1', 'target'],
            file_size=1024,
            data_quality_score=0.8
        )
    )
    
    mock_service.preprocess_dataset.return_value = Mock(
        dataframe=pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]}),
        transformations_applied=['missing_values_handled'],
        quality_report={'overall_score': 0.85}
    )
    
    return mock_service


@pytest.fixture
def mock_ml_service():
    """
    Create a mock MLService for testing.
    
    Returns:
        Mock MLService instance
    """
    mock_service = Mock(spec=MLService)
    
    # Configure mock returns
    mock_service.create_analysis.return_value = "analysis_123"
    
    mock_service.get_analysis_status.return_value = {
        'analysis_id': 'analysis_123',
        'status': 'completed',
        'progress': 100.0,
        'execution_time': 150.0
    }
    
    mock_service.get_analysis_results.return_value = Mock(
        analysis_id='analysis_123',
        status='completed',
        best_model_name='RandomForestClassifier',
        performance_metrics={'accuracy': 0.85, 'f1_score': 0.83},
        execution_time=150.0
    )
    
    mock_service.run_analysis.return_value = "Analysis started in background"
    mock_service.cancel_analysis.return_value = True
    
    return mock_service


@pytest.fixture
def mock_insights_service():
    """
    Create a mock InsightsService for testing.
    
    Returns:
        Mock InsightsService instance
    """
    mock_service = Mock(spec=InsightsService)
    
    # Configure mock returns
    mock_insights_result = Mock(
        insights=[
            Mock(title="High Accuracy", description="Model achieved 85% accuracy", importance=0.9)
        ],
        recommendations=["Consider deploying model", "Monitor performance"],
        summary="Analysis completed successfully with good performance",
        confidence_score=0.85
    )
    
    mock_service.generate_insights.return_value = mock_insights_result
    mock_service.format_for_dashboard.return_value = {
        'insights': [],
        'recommendations': [],
        'visualizations': [],
        'summary': 'Mock dashboard data'
    }
    
    return mock_service


@pytest.fixture
def mock_mlops_service():
    """
    Create a mock MLOpsService for testing.
    
    Returns:
        Mock MLOpsService instance
    """
    mock_service = Mock(spec=MLOpsService)
    
    # Configure mock returns
    mock_service.start_ml_experiment.return_value = "experiment_123"
    mock_service.log_experiment_progress.return_value = None
    mock_service.complete_ml_experiment.return_value = {
        'experiment_id': 'experiment_123',
        'status': 'completed',
        'duration': 150.0
    }
    
    mock_service.get_service_health.return_value = {
        'status': 'healthy',
        'components': {'mlflow': True, 'feast': True},
        'timestamp': datetime.now().isoformat()
    }
    
    return mock_service


# Configuration Fixtures
@pytest.fixture
def test_config() -> Dict[str, Any]:
    """
    Create test configuration settings.
    
    Returns:
        Dictionary with test configuration
    """
    return {
        'testing': True,
        'database_url': TEST_DATABASE_URL,
        'upload_max_size': 10 * 1024 * 1024,  # 10MB
        'ml_timeout': 300,  # 5 minutes
        'cache_enabled': False,
        'log_level': 'WARNING',
        'secret_key': 'test_secret_key_for_testing_only',
        'algorithm': 'HS256',
        'access_token_expire_minutes': 60
    }


@pytest.fixture
def mock_settings(test_config):
    """
    Mock application settings for testing.
    
    Args:
        test_config: Test configuration fixture
        
    Returns:
        Mock settings object
    """
    mock_settings = Mock()
    
    for key, value in test_config.items():
        setattr(mock_settings, key.upper(), value)
    
    return mock_settings


# Analysis and Model Fixtures
@pytest.fixture
def sample_analysis_request() -> Dict[str, Any]:
    """
    Create a sample analysis request for testing.
    
    Returns:
        Dictionary representing an analysis request
    """
    return {
        "dataset_id": 1,
        "target_column": "target",
        "task_type": "classification",
        "execution_mode": "local_cpu",
        "config": {
            "max_models": 5,
            "max_time": 300,
            "cross_validation_folds": 5,
            "enable_ensemble": True,
            "generate_explanations": True
        }
    }


@pytest.fixture
def sample_analysis_response() -> Dict[str, Any]:
    """
    Create a sample analysis response for testing.
    
    Returns:
        Dictionary representing an analysis response
    """
    return {
        "id": "analysis_123",
        "dataset_id": 1,
        "user_id": 1,
        "status": "completed",
        "task_type": "classification",
        "target_column": "target",
        "execution_mode": "local_cpu",
        "progress": 100.0,
        "start_time": "2025-09-21T10:00:00Z",
        "end_time": "2025-09-21T10:02:30Z",
        "execution_time": 150.0,
        "best_model_name": "RandomForestClassifier",
        "performance_metrics": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "roc_auc": 0.92
        },
        "feature_importance": {
            "feature_1": 0.35,
            "feature_2": 0.25,
            "feature_3": 0.20,
            "categorical_feature": 0.15,
            "binary_feature": 0.05
        }
    }


@pytest.fixture
def sample_dataset_metadata() -> Dict[str, Any]:
    """
    Create sample dataset metadata for testing.
    
    Returns:
        Dictionary with dataset metadata
    """
    return {
        "id": 1,
        "name": "test_dataset.csv",
        "original_filename": "test_dataset.csv",
        "file_size": 2048,
        "file_path": "/tmp/datasets/test_dataset.csv",
        "status": "processed",
        "upload_time": "2025-09-21T10:00:00Z",
        "num_rows": 1000,
        "num_columns": 8,
        "column_names": [
            "feature_1", "feature_2", "feature_3", "feature_4", "feature_5",
            "categorical_feature", "binary_feature", "target"
        ],
        "column_types": {
            "feature_1": "numeric",
            "feature_2": "numeric",
            "feature_3": "numeric",
            "feature_4": "numeric",
            "feature_5": "numeric",
            "categorical_feature": "categorical",
            "binary_feature": "boolean",
            "target": "categorical"
        },
        "data_quality_score": 0.85,
        "missing_value_ratio": 0.02,
        "duplicate_ratio": 0.01
    }


# Utility Fixtures
@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for async tests.
    
    This fixture ensures that async tests have a proper event loop available.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_client(client):
    """
    Create an async-compatible test client.
    
    Args:
        client: FastAPI test client fixture
        
    Returns:
        Async-compatible test client
    """
    return client


# Dependency Override Fixtures
@pytest.fixture
def override_dependencies(
    mock_data_service, 
    mock_ml_service, 
    mock_insights_service, 
    mock_mlops_service
):
    """
    Override service dependencies for testing.
    
    This fixture replaces all service dependencies with mocks to isolate
    API testing from service implementation details.
    
    Args:
        mock_data_service: Mock data service fixture
        mock_ml_service: Mock ML service fixture
        mock_insights_service: Mock insights service fixture
        mock_mlops_service: Mock MLOps service fixture
    """
    if not BACKEND_AVAILABLE:
        return
    
    # Store original dependencies
    original_overrides = app.dependency_overrides.copy()
    
    # Override dependencies
    app.dependency_overrides[get_data_service] = lambda: mock_data_service
    app.dependency_overrides[get_ml_service] = lambda: mock_ml_service
    app.dependency_overrides[get_insights_service] = lambda: mock_insights_service
    app.dependency_overrides[get_mlops_service] = lambda: mock_mlops_service
    
    yield
    
    # Restore original dependencies
    app.dependency_overrides.clear()
    app.dependency_overrides.update(original_overrides)


# Performance Testing Fixtures
@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """
    Create a large dataset for performance testing.
    
    Returns:
        Large pandas DataFrame for performance tests
    """
    np.random.seed(46)
    
    n_samples = 50000
    n_features = 50
    
    # Generate large dataset
    data = {}
    for i in range(n_features):
        if i % 5 == 0:  # Every 5th feature is categorical
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
        else:
            data[f'num_feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create target
    target = np.random.choice([0, 1], n_samples)
    data['target'] = target
    
    return pd.DataFrame(data)


@pytest.fixture
def performance_timer():
    """
    Create a context manager for timing test operations.
    
    Returns:
        Context manager that tracks execution time
    """
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.duration = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
    
    import time
    return Timer


# Error Injection Fixtures
@pytest.fixture
def error_injection():
    """
    Create utilities for injecting errors during testing.
    
    Returns:
        Dictionary with error injection utilities
    """
    return {
        'database_error': lambda: Exception("Database connection failed"),
        'timeout_error': lambda: TimeoutError("Operation timed out"),
        'validation_error': lambda: ValueError("Invalid input data"),
        'auth_error': lambda: PermissionError("Authentication failed"),
        'file_error': lambda: FileNotFoundError("File not found"),
        'memory_error': lambda: MemoryError("Out of memory")
    }


# Cleanup Utilities
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Automatic cleanup fixture that runs after each test.
    
    This fixture automatically cleans up any test artifacts, resets
    global state, and ensures test isolation.
    """
    # Pre-test setup (if needed)
    yield
    
    # Post-test cleanup
    # Clear any global caches or state
    if BACKEND_AVAILABLE:
        # Clear dependency overrides
        app.dependency_overrides.clear()
    
    # Clean up any temporary files that might have been created
    temp_pattern = f"{TEMP_FILE_PREFIX}*"
    temp_dir = Path(tempfile.gettempdir())
    
    for temp_file in temp_dir.glob(temp_pattern):
        try:
            if temp_file.is_file():
                temp_file.unlink()
            elif temp_file.is_dir():
                shutil.rmtree(temp_file)
        except Exception:
            # Ignore cleanup errors
            pass


# Parameterized Test Data
@pytest.fixture(params=[
    "classification",
    "regression", 
    "clustering",
    "anomaly_detection"
])
def task_types(request):
    """
    Parameterized fixture for testing different ML task types.
    
    Args:
        request: Pytest request object
        
    Returns:
        Task type string for parameterized testing
    """
    return request.param


@pytest.fixture(params=[
    {"max_models": 3, "max_time": 60},
    {"max_models": 5, "max_time": 120},
    {"max_models": 8, "max_time": 300}
])
def analysis_configs(request):
    """
    Parameterized fixture for testing different analysis configurations.
    
    Args:
        request: Pytest request object
        
    Returns:
        Configuration dictionary for parameterized testing
    """
    return request.param


@pytest.fixture(params=[
    ("csv", "text/csv"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("json", "application/json")
])
def file_formats(request):
    """
    Parameterized fixture for testing different file formats.
    
    Args:
        request: Pytest request object
        
    Returns:
        Tuple of (file_extension, mime_type) for parameterized testing
    """
    return request.param


# Documentation and Help
def pytest_report_header(config):
    """
    Add custom header information to pytest reports.
    
    Args:
        config: Pytest configuration object
        
    Returns:
        String to add to test report header
    """
    header_lines = [
        "Auto-Analyst Backend Test Suite",
        f"Backend Available: {BACKEND_AVAILABLE}",
        f"Test Database: {TEST_DATABASE_URL}",
        f"Temp Directory: {os.environ.get('TEST_TEMP_DIR', 'Not Set')}"
    ]
    
    return "\n".join(header_lines)


# Export commonly used fixtures for easy import
__all__ = [
    # Client fixtures
    "client", "authenticated_client", "async_client",
    
    # Database fixtures  
    "test_db_session", "override_get_db",
    
    # Authentication fixtures
    "test_user", "admin_user", "auth_token", "auth_headers", "admin_headers",
    
    # Data fixtures
    "sample_classification_dataset", "sample_regression_dataset", 
    "sample_timeseries_dataset", "sample_text_dataset", "corrupted_dataset",
    
    # File fixtures
    "temp_directory", "sample_csv_file", "sample_excel_file", 
    "sample_json_file", "invalid_file",
    
    # Service mocks
    "mock_data_service", "mock_ml_service", "mock_insights_service", "mock_mlops_service",
    
    # Configuration
    "test_config", "mock_settings",
    
    # Analysis fixtures
    "sample_analysis_request", "sample_analysis_response", "sample_dataset_metadata",
    
    # Utilities
    "large_dataset", "performance_timer", "error_injection",
    
    # Parameterized fixtures
    "task_types", "analysis_configs", "file_formats"
]
