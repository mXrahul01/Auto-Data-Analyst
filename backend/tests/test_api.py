"""
Comprehensive Test Suite for Auto-Analyst FastAPI Endpoints

This test module provides complete coverage for all FastAPI endpoints in the
Auto-Analyst backend, including dataset management, ML training, analysis
retrieval, and dashboard functionality.

Test Coverage:
- Dataset Upload Endpoints - File upload, validation, metadata extraction
- Analysis Management - Creating, running, monitoring ML analyses
- Dashboard Endpoints - Results retrieval, insights, visualizations
- User Management - Authentication, authorization, user operations
- Health & Status - System health, service status monitoring
- Error Handling - Invalid requests, server errors, edge cases

Test Categories:
- Positive Tests: Valid requests and expected successful responses
- Negative Tests: Invalid inputs, error conditions, edge cases
- Authentication Tests: Login, logout, token validation, permissions
- Integration Tests: End-to-end workflow testing
- Performance Tests: Response times, large file handling
- Security Tests: Input validation, SQL injection prevention

Usage:
    # Run all API tests
    pytest tests/test_api.py -v
    
    # Run specific endpoint tests
    pytest tests/test_api.py::TestDatasetEndpoints -v
    
    # Run with coverage
    pytest tests/test_api.py --cov=backend --cov-report=html
    
    # Run security tests only
    pytest tests/test_api.py -m security
"""

import pytest
import json
import tempfile
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# FastAPI and HTTP testing imports
from fastapi.testclient import TestClient
from fastapi import status
from fastapi.security import HTTPBearer
from starlette.datastructures import UploadFile

# Backend imports - with error handling for missing modules
try:
    from backend.main import app
    from backend.models import schemas, database
    from backend.services import (
        DataService, MLService, InsightsService, 
        get_data_service, get_ml_service, get_insights_service
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    pytest.skip(f"Backend not available: {e}", allow_module_level=True)

# Test markers
pytestmark = pytest.mark.api


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    if not BACKEND_AVAILABLE:
        pytest.skip("Backend not available")
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    """Create authentication headers for protected endpoints."""
    return {
        "Authorization": "Bearer test_token_123",
        "Content-Type": "application/json"
    }


@pytest.fixture
def test_user():
    """Create a test user for authentication testing."""
    return {
        "id": 1,
        "email": "test@example.com",
        "username": "testuser",
        "is_active": True,
        "is_verified": True
    }


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for upload testing."""
    # Create sample data
    data = {
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(0, 1, 100),
        'feature_3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(data)
    
    # Create temporary CSV file
    csv_content = df.to_csv(index=False)
    
    return io.StringIO(csv_content), "test_dataset.csv", csv_content


@pytest.fixture
def sample_excel_file():
    """Create a sample Excel file for upload testing."""
    # Create sample data
    data = {
        'numeric_col': np.random.normal(50, 10, 50),
        'categorical_col': np.random.choice(['X', 'Y', 'Z'], 50),
        'target_col': np.random.normal(100, 20, 50)
    }
    df = pd.DataFrame(data)
    
    # Create temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        df.to_excel(tmp_file.name, index=False)
        tmp_file.seek(0)
        return tmp_file, "test_dataset.xlsx"


@pytest.fixture
def mock_dataset_response():
    """Mock dataset response for testing."""
    return {
        "id": 1,
        "name": "test_dataset.csv",
        "status": "processed",
        "file_size": 2048,
        "num_rows": 100,
        "num_columns": 4,
        "column_names": ["feature_1", "feature_2", "feature_3", "target"],
        "column_types": {
            "feature_1": "numeric",
            "feature_2": "numeric", 
            "feature_3": "categorical",
            "target": "numeric"
        },
        "upload_time": "2025-09-21T09:55:00Z",
        "file_path": "/tmp/datasets/test_dataset.csv"
    }


@pytest.fixture
def mock_analysis_response():
    """Mock analysis response for testing."""
    return {
        "id": "analysis_123",
        "dataset_id": 1,
        "status": "completed",
        "task_type": "classification",
        "target_column": "target",
        "execution_mode": "local_cpu",
        "progress": 100.0,
        "start_time": "2025-09-21T09:55:00Z",
        "end_time": "2025-09-21T09:57:30Z",
        "execution_time": 150.0,
        "best_model_name": "RandomForestClassifier",
        "performance_metrics": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85
        },
        "feature_importance": {
            "feature_1": 0.35,
            "feature_2": 0.25,
            "feature_3": 0.40
        }
    }


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "degraded", "critical"]
        assert "timestamp" in health_data
        assert "services" in health_data
    
    def test_service_status(self, client):
        """Test service status endpoint."""
        response = client.get("/status")
        
        assert response.status_code == status.HTTP_200_OK
        
        status_data = response.json()
        assert "backend_status" in status_data
        assert "database_status" in status_data
        assert "ml_pipeline_status" in status_data
        assert "services_status" in status_data
    
    def test_version_info(self, client):
        """Test version information endpoint."""
        response = client.get("/version")
        
        assert response.status_code == status.HTTP_200_OK
        
        version_data = response.json()
        assert "version" in version_data
        assert "build_time" in version_data or "timestamp" in version_data
        assert "components" in version_data


class TestAuthEndpoints:
    """Test authentication and authorization endpoints."""
    
    def test_login_success(self, client, test_user):
        """Test successful user login."""
        login_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        
        with patch("backend.auth.authenticate_user") as mock_auth:
            mock_auth.return_value = test_user
            
            response = client.post("/auth/login", data=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        auth_response = response.json()
        assert "access_token" in auth_response
        assert "token_type" in auth_response
        assert auth_response["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        login_data = {
            "username": "wronguser",
            "password": "wrongpassword"
        }
        
        with patch("backend.auth.authenticate_user") as mock_auth:
            mock_auth.return_value = None
            
            response = client.post("/auth/login", data=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        error_response = response.json()
        assert "detail" in error_response
        assert "invalid" in error_response["detail"].lower()
    
    def test_login_missing_credentials(self, client):
        """Test login with missing credentials."""
        incomplete_data = {
            "username": "testuser"
            # Missing password
        }
        
        response = client.post("/auth/login", data=incomplete_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_logout(self, client, auth_headers):
        """Test user logout."""
        response = client.post("/auth/logout", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        logout_response = response.json()
        assert logout_response.get("message") == "Successfully logged out" or "success" in str(logout_response).lower()
    
    def test_token_validation(self, client, auth_headers, test_user):
        """Test token validation endpoint."""
        with patch("backend.auth.get_current_user") as mock_user:
            mock_user.return_value = test_user
            
            response = client.get("/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        user_response = response.json()
        assert user_response["id"] == test_user["id"]
        assert user_response["email"] == test_user["email"]
    
    def test_invalid_token(self, client):
        """Test requests with invalid authentication token."""
        invalid_headers = {
            "Authorization": "Bearer invalid_token_xyz"
        }
        
        response = client.get("/auth/me", headers=invalid_headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestDatasetEndpoints:
    """Test dataset upload and management endpoints."""
    
    def test_upload_csv_file_success(self, client, auth_headers, sample_csv_file, mock_dataset_response):
        """Test successful CSV file upload."""
        file_obj, filename, content = sample_csv_file
        
        files = {
            "file": (filename, content, "text/csv")
        }
        
        with patch("backend.services.data_service.DataService") as mock_service:
            mock_service.return_value.load_dataset.return_value = Mock(
                info=Mock(
                    n_rows=100,
                    n_columns=4,
                    column_names=["feature_1", "feature_2", "feature_3", "target"],
                    file_size=2048
                )
            )
            
            response = client.post(
                "/datasets/upload",
                files=files,
                headers={"Authorization": auth_headers["Authorization"]}
            )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        upload_response = response.json()
        assert "id" in upload_response
        assert upload_response["name"] == filename
        assert upload_response["status"] in ["processing", "processed"]
        assert upload_response["num_rows"] > 0
        assert upload_response["num_columns"] > 0
    
    def test_upload_excel_file_success(self, client, auth_headers, sample_excel_file, mock_dataset_response):
        """Test successful Excel file upload."""
        file_obj, filename = sample_excel_file
        
        with open(file_obj.name, 'rb') as f:
            files = {
                "file": (filename, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            }
            
            with patch("backend.services.data_service.DataService") as mock_service:
                mock_service.return_value.load_dataset.return_value = Mock(
                    info=Mock(n_rows=50, n_columns=3, file_size=1024)
                )
                
                response = client.post(
                    "/datasets/upload",
                    files=files,
                    headers={"Authorization": auth_headers["Authorization"]}
                )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        upload_response = response.json()
        assert upload_response["name"] == filename
        assert "id" in upload_response
    
    def test_upload_file_too_large(self, client, auth_headers):
        """Test upload file that exceeds size limits."""
        # Create a mock large file
        large_content = "data" * (10 * 1024 * 1024)  # 40MB string
        
        files = {
            "file": ("large_file.csv", large_content, "text/csv")
        }
        
        response = client.post(
            "/datasets/upload",
            files=files,
            headers={"Authorization": auth_headers["Authorization"]}
        )
        
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    def test_upload_invalid_file_format(self, client, auth_headers):
        """Test upload of invalid file format."""
        # Create a text file with invalid content
        invalid_content = "This is not a valid dataset file format"
        
        files = {
            "file": ("invalid.txt", invalid_content, "text/plain")
        }
        
        response = client.post(
            "/datasets/upload",
            files=files,
            headers={"Authorization": auth_headers["Authorization"]}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        error_response = response.json()
        assert "unsupported" in error_response["detail"].lower() or "invalid" in error_response["detail"].lower()
    
    def test_upload_without_authentication(self, client, sample_csv_file):
        """Test file upload without authentication."""
        file_obj, filename, content = sample_csv_file
        
        files = {
            "file": (filename, content, "text/csv")
        }
        
        response = client.post("/datasets/upload", files=files)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_datasets_list(self, client, auth_headers):
        """Test retrieving list of user's datasets."""
        mock_datasets = [
            {
                "id": 1,
                "name": "dataset1.csv",
                "status": "processed",
                "upload_time": "2025-09-21T09:00:00Z",
                "num_rows": 100,
                "num_columns": 5
            },
            {
                "id": 2,
                "name": "dataset2.xlsx",
                "status": "processing",
                "upload_time": "2025-09-21T09:30:00Z",
                "num_rows": 200,
                "num_columns": 8
            }
        ]
        
        with patch("backend.crud.dataset.get_user_datasets") as mock_get:
            mock_get.return_value = mock_datasets
            
            response = client.get("/datasets/", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        datasets_response = response.json()
        assert isinstance(datasets_response, list)
        assert len(datasets_response) == 2
        assert datasets_response[0]["id"] == 1
        assert datasets_response[1]["id"] == 2
    
    def test_get_dataset_details(self, client, auth_headers, mock_dataset_response):
        """Test retrieving specific dataset details."""
        dataset_id = 1
        
        with patch("backend.crud.dataset.get_dataset") as mock_get:
            mock_get.return_value = mock_dataset_response
            
            response = client.get(f"/datasets/{dataset_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        dataset_response = response.json()
        assert dataset_response["id"] == dataset_id
        assert "column_names" in dataset_response
        assert "column_types" in dataset_response
        assert "num_rows" in dataset_response
    
    def test_get_nonexistent_dataset(self, client, auth_headers):
        """Test retrieving dataset that doesn't exist."""
        dataset_id = 99999
        
        with patch("backend.crud.dataset.get_dataset") as mock_get:
            mock_get.return_value = None
            
            response = client.get(f"/datasets/{dataset_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_dataset(self, client, auth_headers):
        """Test deleting a dataset."""
        dataset_id = 1
        
        with patch("backend.crud.dataset.delete_dataset") as mock_delete:
            mock_delete.return_value = True
            
            response = client.delete(f"/datasets/{dataset_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
    
    def test_delete_nonexistent_dataset(self, client, auth_headers):
        """Test deleting dataset that doesn't exist."""
        dataset_id = 99999
        
        with patch("backend.crud.dataset.delete_dataset") as mock_delete:
            mock_delete.return_value = False
            
            response = client.delete(f"/datasets/{dataset_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_dataset_preview(self, client, auth_headers):
        """Test getting dataset preview (first few rows)."""
        dataset_id = 1
        
        mock_preview_data = {
            "columns": ["feature_1", "feature_2", "target"],
            "data": [
                [1.5, 2.3, 0],
                [0.8, 1.9, 1],
                [2.1, 3.2, 0],
                [1.2, 2.8, 1],
                [0.9, 1.7, 0]
            ],
            "total_rows": 100,
            "preview_rows": 5
        }
        
        with patch("backend.services.data_service.DataService") as mock_service:
            mock_service.return_value.get_dataset_preview.return_value = mock_preview_data
            
            response = client.get(f"/datasets/{dataset_id}/preview", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        preview_response = response.json()
        assert "columns" in preview_response
        assert "data" in preview_response
        assert len(preview_response["data"]) <= 10  # Should be limited


class TestAnalysisEndpoints:
    """Test ML analysis creation and management endpoints."""
    
    def test_create_analysis_success(self, client, auth_headers):
        """Test successful analysis creation."""
        analysis_request = {
            "dataset_id": 1,
            "target_column": "target",
            "task_type": "classification",
            "execution_mode": "local_cpu",
            "config": {
                "max_models": 5,
                "max_time": 300
            }
        }
        
        mock_analysis_id = "analysis_123"
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.create_analysis.return_value = mock_analysis_id
            
            response = client.post(
                "/analyses/",
                json=analysis_request,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        create_response = response.json()
        assert create_response["analysis_id"] == mock_analysis_id
        assert create_response["status"] == "created"
    
    def test_create_analysis_invalid_dataset(self, client, auth_headers):
        """Test analysis creation with invalid dataset ID."""
        analysis_request = {
            "dataset_id": 99999,  # Non-existent dataset
            "target_column": "target",
            "task_type": "classification"
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.create_analysis.side_effect = ValueError("Dataset not found")
            
            response = client.post(
                "/analyses/",
                json=analysis_request,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        error_response = response.json()
        assert "dataset" in error_response["detail"].lower()
    
    def test_create_analysis_missing_target(self, client, auth_headers):
        """Test analysis creation with missing target column."""
        analysis_request = {
            "dataset_id": 1,
            "target_column": "nonexistent_column",
            "task_type": "classification"
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.create_analysis.side_effect = ValueError("Target column not found")
            
            response = client.post(
                "/analyses/",
                json=analysis_request,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        error_response = response.json()
        assert "target" in error_response["detail"].lower() or "column" in error_response["detail"].lower()
    
    def test_create_analysis_validation_error(self, client, auth_headers):
        """Test analysis creation with validation errors."""
        invalid_request = {
            "dataset_id": "invalid_id",  # Should be integer
            "target_column": "",  # Empty string
            "task_type": "invalid_task"  # Invalid task type
        }
        
        response = client.post(
            "/analyses/",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_analysis_status(self, client, auth_headers):
        """Test retrieving analysis status."""
        analysis_id = "analysis_123"
        
        mock_status = {
            "analysis_id": analysis_id,
            "status": "running",
            "progress": 65.0,
            "current_stage": "model_training",
            "execution_time": 120.5,
            "execution_mode": "local_cpu"
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.get_analysis_status.return_value = mock_status
            
            response = client.get(f"/analyses/{analysis_id}/status", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        status_response = response.json()
        assert status_response["analysis_id"] == analysis_id
        assert status_response["status"] == "running"
        assert "progress" in status_response
        assert "current_stage" in status_response
    
    def test_get_analysis_results(self, client, auth_headers, mock_analysis_response):
        """Test retrieving completed analysis results."""
        analysis_id = "analysis_123"
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.get_analysis_results.return_value = Mock(
                analysis_id=analysis_id,
                status="completed",
                best_model_name="RandomForestClassifier",
                performance_metrics={"accuracy": 0.85},
                execution_time=150.0,
                dashboard_data={"charts": [], "metrics": {}}
            )
            
            response = client.get(f"/analyses/{analysis_id}/results", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        results_response = response.json()
        assert results_response["analysis_id"] == analysis_id
        assert "performance_metrics" in results_response
        assert "best_model_name" in results_response
    
    def test_get_analysis_results_not_completed(self, client, auth_headers):
        """Test retrieving results for analysis that's not completed."""
        analysis_id = "analysis_running"
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.get_analysis_results.return_value = None
            
            response = client.get(f"/analyses/{analysis_id}/results", headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND or response.status_code == status.HTTP_202_ACCEPTED
    
    def test_cancel_analysis(self, client, auth_headers):
        """Test cancelling a running analysis."""
        analysis_id = "analysis_123"
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.cancel_analysis.return_value = True
            
            response = client.post(f"/analyses/{analysis_id}/cancel", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        cancel_response = response.json()
        assert cancel_response["message"] == "Analysis cancelled successfully" or "cancelled" in str(cancel_response).lower()
    
    def test_list_user_analyses(self, client, auth_headers):
        """Test listing user's analyses."""
        mock_analyses = [
            {
                "id": "analysis_1",
                "dataset_id": 1,
                "status": "completed",
                "task_type": "classification",
                "created_at": "2025-09-21T09:00:00Z"
            },
            {
                "id": "analysis_2",
                "dataset_id": 2,
                "status": "running",
                "task_type": "regression",
                "created_at": "2025-09-21T09:30:00Z"
            }
        ]
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.list_analyses.return_value = mock_analyses
            
            response = client.get("/analyses/", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        analyses_response = response.json()
        assert isinstance(analyses_response, list)
        assert len(analyses_response) == 2
    
    def test_run_analysis_background(self, client, auth_headers):
        """Test starting analysis execution in background."""
        analysis_id = "analysis_123"
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.run_analysis.return_value = "Analysis started in background"
            
            response = client.post(f"/analyses/{analysis_id}/run", headers=auth_headers)
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        
        run_response = response.json()
        assert "started" in str(run_response).lower()


class TestDashboardEndpoints:
    """Test dashboard and insights endpoints."""
    
    def test_get_dashboard_data(self, client, auth_headers):
        """Test retrieving dashboard data for an analysis."""
        analysis_id = "analysis_123"
        
        mock_dashboard_data = {
            "overview": {
                "analysis_id": analysis_id,
                "task_type": "classification",
                "model_performance": {"accuracy": 0.85}
            },
            "visualizations": [
                {
                    "type": "bar_chart",
                    "title": "Feature Importance",
                    "data": {"labels": ["f1", "f2"], "values": [0.6, 0.4]}
                }
            ],
            "insights": [
                "Model achieved 85% accuracy",
                "Feature 'f1' is most important"
            ],
            "recommendations": [
                "Consider feature engineering",
                "Try ensemble methods"
            ]
        }
        
        with patch("backend.services.insights_service.InsightsService") as mock_service:
            mock_service.return_value.format_for_dashboard.return_value = mock_dashboard_data
            
            response = client.get(f"/dashboard/{analysis_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        dashboard_response = response.json()
        assert "overview" in dashboard_response
        assert "visualizations" in dashboard_response
        assert "insights" in dashboard_response
    
    def test_get_insights(self, client, auth_headers):
        """Test retrieving insights for an analysis."""
        analysis_id = "analysis_123"
        
        mock_insights = {
            "analysis_id": analysis_id,
            "insights": [
                {
                    "title": "High Model Performance",
                    "description": "Your model achieved excellent accuracy of 85%",
                    "type": "performance",
                    "importance": "high"
                }
            ],
            "recommendations": [
                "Model is ready for deployment",
                "Consider A/B testing"
            ],
            "summary": "Analysis completed successfully with high performance model"
        }
        
        with patch("backend.services.insights_service.InsightsService") as mock_service:
            mock_service.return_value.generate_insights.return_value = Mock(
                insights=mock_insights["insights"],
                recommendations=mock_insights["recommendations"],
                summary=mock_insights["summary"]
            )
            
            response = client.get(f"/insights/{analysis_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        insights_response = response.json()
        assert "insights" in insights_response
        assert "recommendations" in insights_response
        assert "summary" in insights_response
    
    def test_get_model_explanations(self, client, auth_headers):
        """Test retrieving model explanations."""
        analysis_id = "analysis_123"
        
        mock_explanations = {
            "feature_importance": {
                "global": {"feature_1": 0.4, "feature_2": 0.3, "feature_3": 0.3},
                "local_examples": [
                    {"sample_id": 0, "explanation": {"feature_1": 0.5, "feature_2": -0.2}}
                ]
            },
            "shap_values": {
                "available": True,
                "summary_plot": "/tmp/shap_summary.png"
            }
        }
        
        with patch("backend.services.explainer_service.ExplainerService") as mock_service:
            mock_service.return_value.explain_model.return_value = mock_explanations
            
            response = client.get(f"/explanations/{analysis_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        explanations_response = response.json()
        assert "feature_importance" in explanations_response
    
    def test_get_performance_metrics(self, client, auth_headers):
        """Test retrieving detailed performance metrics."""
        analysis_id = "analysis_123"
        
        mock_metrics = {
            "classification_metrics": {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85,
                "auc_roc": 0.92
            },
            "confusion_matrix": [[80, 15], [10, 95]],
            "classification_report": {
                "0": {"precision": 0.89, "recall": 0.84, "f1-score": 0.86},
                "1": {"precision": 0.86, "recall": 0.90, "f1-score": 0.88}
            },
            "cross_validation_scores": [0.83, 0.87, 0.85, 0.82, 0.88]
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_analysis_result = Mock(performance_metrics=mock_metrics["classification_metrics"])
            mock_service.return_value.get_analysis_results.return_value = mock_analysis_result
            
            response = client.get(f"/metrics/{analysis_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        
        metrics_response = response.json()
        assert "classification_metrics" in metrics_response or "performance_metrics" in metrics_response
    
    def test_export_results(self, client, auth_headers):
        """Test exporting analysis results."""
        analysis_id = "analysis_123"
        
        mock_export_data = {
            "analysis_summary": {"id": analysis_id, "status": "completed"},
            "model_details": {"name": "RandomForest", "accuracy": 0.85},
            "export_format": "json",
            "export_timestamp": "2025-09-21T09:55:00Z"
        }
        
        with patch("backend.services.export_service.create_export") as mock_export:
            mock_export.return_value = mock_export_data
            
            response = client.get(
                f"/export/{analysis_id}",
                params={"format": "json"},
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        export_response = response.json()
        assert "analysis_summary" in export_response
        assert "model_details" in export_response


class TestPredictionEndpoints:
    """Test model prediction endpoints."""
    
    def test_make_predictions(self, client, auth_headers):
        """Test making predictions with trained model."""
        analysis_id = "analysis_123"
        
        prediction_request = {
            "features": {
                "feature_1": 1.5,
                "feature_2": 2.3,
                "feature_3": "A"
            }
        }
        
        mock_predictions = {
            "predictions": [1],
            "probabilities": [0.85, 0.15],
            "model_name": "RandomForestClassifier",
            "confidence": 0.85
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.get_model_predictions.return_value = mock_predictions
            
            response = client.post(
                f"/predict/{analysis_id}",
                json=prediction_request,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        prediction_response = response.json()
        assert "predictions" in prediction_response
        assert "model_name" in prediction_response
    
    def test_batch_predictions(self, client, auth_headers):
        """Test making batch predictions."""
        analysis_id = "analysis_123"
        
        batch_request = {
            "features": [
                {"feature_1": 1.5, "feature_2": 2.3, "feature_3": "A"},
                {"feature_1": 0.8, "feature_2": 1.9, "feature_3": "B"},
                {"feature_1": 2.1, "feature_2": 3.2, "feature_3": "C"}
            ]
        }
        
        mock_predictions = {
            "predictions": [1, 0, 1],
            "probabilities": [[0.85, 0.15], [0.25, 0.75], [0.92, 0.08]],
            "model_name": "RandomForestClassifier"
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.get_model_predictions.return_value = mock_predictions
            
            response = client.post(
                f"/predict/{analysis_id}/batch",
                json=batch_request,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        prediction_response = response.json()
        assert len(prediction_response["predictions"]) == 3
    
    def test_predictions_missing_features(self, client, auth_headers):
        """Test predictions with missing required features."""
        analysis_id = "analysis_123"
        
        incomplete_request = {
            "features": {
                "feature_1": 1.5
                # Missing required features
            }
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.get_model_predictions.side_effect = ValueError("Missing required features")
            
            response = client.post(
                f"/predict/{analysis_id}",
                json=incomplete_request,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        error_response = response.json()
        assert "missing" in error_response["detail"].lower() or "features" in error_response["detail"].lower()
    
    def test_predictions_model_not_ready(self, client, auth_headers):
        """Test predictions when model is not ready."""
        analysis_id = "analysis_not_completed"
        
        prediction_request = {
            "features": {"feature_1": 1.5, "feature_2": 2.3}
        }
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            mock_service.return_value.get_model_predictions.side_effect = RuntimeError("Model not ready")
            
            response = client.post(
                f"/predict/{analysis_id}",
                json=prediction_request,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST or response.status_code == status.HTTP_409_CONFLICT


@pytest.mark.security
class TestSecurityEndpoints:
    """Test security aspects of API endpoints."""
    
    def test_sql_injection_prevention(self, client, auth_headers):
        """Test SQL injection prevention in query parameters."""
        malicious_id = "1; DROP TABLE datasets; --"
        
        response = client.get(f"/datasets/{malicious_id}", headers=auth_headers)
        
        # Should return 404 or 400, not 500 (which would indicate SQL injection)
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND]
    
    def test_xss_prevention(self, client, auth_headers):
        """Test XSS prevention in request data."""
        xss_payload = "<script>alert('xss')</script>"
        
        analysis_request = {
            "dataset_id": 1,
            "target_column": xss_payload,
            "task_type": "classification"
        }
        
        response = client.post(
            "/analyses/",
            json=analysis_request,
            headers=auth_headers
        )
        
        # Should be rejected due to validation
        assert response.status_code == status.HTTP_400_BAD_REQUEST or response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_file_upload_security(self, client, auth_headers):
        """Test file upload security restrictions."""
        # Test executable file upload
        malicious_content = b"#!/bin/bash\necho 'malicious script'"
        
        files = {
            "file": ("malicious.sh", malicious_content, "application/x-sh")
        }
        
        response = client.post(
            "/datasets/upload",
            files=files,
            headers={"Authorization": auth_headers["Authorization"]}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting on endpoints."""
        # Make multiple rapid requests
        responses = []
        for _ in range(20):
            response = client.get("/datasets/", headers=auth_headers)
            responses.append(response.status_code)
        
        # Should either succeed all or start rate limiting
        rate_limited = any(code == status.HTTP_429_TOO_MANY_REQUESTS for code in responses)
        all_successful = all(code == status.HTTP_200_OK for code in responses)
        
        assert rate_limited or all_successful, "Should handle rate limiting appropriately"


@pytest.mark.performance
class TestPerformanceEndpoints:
    """Test performance aspects of API endpoints."""
    
    def test_endpoint_response_times(self, client, auth_headers):
        """Test that endpoints respond within reasonable time limits."""
        endpoints_to_test = [
            ("/health", "GET"),
            ("/datasets/", "GET"),
            ("/analyses/", "GET")
        ]
        
        for endpoint, method in endpoints_to_test:
            start_time = time.time()
            
            if method == "GET":
                response = client.get(endpoint, headers=auth_headers)
            elif method == "POST":
                response = client.post(endpoint, headers=auth_headers)
            
            response_time = time.time() - start_time
            
            # Endpoints should respond within 2 seconds
            assert response_time < 2.0, f"Endpoint {endpoint} took too long: {response_time:.2f}s"
            
            # Should not return server error
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_large_dataset_upload_handling(self, client, auth_headers):
        """Test handling of large dataset uploads."""
        # Create a moderately large dataset
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, 10000) 
            for i in range(20)
        })
        large_data['target'] = np.random.choice([0, 1], 10000)
        
        csv_content = large_data.to_csv(index=False)
        
        files = {
            "file": ("large_dataset.csv", csv_content, "text/csv")
        }
        
        start_time = time.time()
        
        with patch("backend.services.data_service.DataService") as mock_service:
            mock_service.return_value.load_dataset.return_value = Mock(
                info=Mock(n_rows=10000, n_columns=21, file_size=len(csv_content))
            )
            
            response = client.post(
                "/datasets/upload",
                files=files,
                headers={"Authorization": auth_headers["Authorization"]}
            )
        
        upload_time = time.time() - start_time
        
        # Should handle large upload within reasonable time (10 seconds)
        assert upload_time < 10.0, f"Large upload took too long: {upload_time:.2f}s"
        
        if response.status_code == status.HTTP_201_CREATED:
            upload_response = response.json()
            assert upload_response["num_rows"] == 10000


class TestErrorHandling:
    """Test comprehensive error handling across endpoints."""
    
    def test_internal_server_error_handling(self, client, auth_headers):
        """Test handling of internal server errors."""
        with patch("backend.services.ml_service.MLService") as mock_service:
            # Simulate internal server error
            mock_service.side_effect = Exception("Database connection failed")
            
            response = client.get("/analyses/", headers=auth_headers)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error_response = response.json()
        assert "detail" in error_response
        # Should not expose internal error details
        assert "Database connection failed" not in error_response["detail"]
    
    def test_validation_error_messages(self, client, auth_headers):
        """Test that validation errors provide helpful messages."""
        invalid_request = {
            "dataset_id": -1,  # Invalid ID
            "target_column": "",  # Empty string
            "task_type": "invalid_task",  # Invalid enum value
            "config": {
                "max_models": "not_a_number"  # Wrong type
            }
        }
        
        response = client.post(
            "/analyses/",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        error_response = response.json()
        assert "detail" in error_response
        
        # Should provide specific validation errors
        error_details = error_response["detail"]
        if isinstance(error_details, list):
            # FastAPI validation error format
            field_errors = [error["loc"][-1] for error in error_details]
            assert "dataset_id" in field_errors or "target_column" in field_errors
    
    def test_timeout_handling(self, client, auth_headers):
        """Test handling of request timeouts."""
        analysis_id = "analysis_123"
        
        with patch("backend.services.ml_service.MLService") as mock_service:
            # Simulate timeout
            import asyncio
            
            async def slow_operation(*args, **kwargs):
                await asyncio.sleep(30)  # Simulate very slow operation
                return {}
            
            mock_service.return_value.get_analysis_results = slow_operation
            
            start_time = time.time()
            response = client.get(f"/analyses/{analysis_id}/results", headers=auth_headers)
            request_time = time.time() - start_time
            
            # Should timeout and return error before 30 seconds
            assert request_time < 25.0, "Request should timeout before 25 seconds"
            assert response.status_code in [
                status.HTTP_408_REQUEST_TIMEOUT,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                status.HTTP_503_SERVICE_UNAVAILABLE
            ]


if __name__ == "__main__":
    # Run tests when file is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
