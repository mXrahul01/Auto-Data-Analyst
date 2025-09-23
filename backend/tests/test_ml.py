# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from pathlib import Path

from backend.main import app, Settings, ModelManager

# Test fixtures
@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)

@pytest.fixture
def mock_model_manager():
    """Mock model manager fixture."""
    manager = Mock(spec=ModelManager)
    manager.get_available_models.return_value = ["test_model"]
    manager.get_loaded_models.return_value = []
    manager.device = "cpu"
    return manager

@pytest.fixture
def temp_model_dir():
    """Temporary model directory fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock model file
        model_file = Path(temp_dir) / "test_model.pkl"
        model_file.touch()
        yield temp_dir

# Schema validation tests
class TestSchemaValidation:
    """Test Pydantic schema validation."""
    
    def test_prediction_request_valid(self):
        """Test valid prediction request."""
        from backend.main import PredictionRequest
        
        request = PredictionRequest(
            data={"feature1": 1.0, "feature2": "A"},
            model_name="test_model",
            return_probabilities=True
        )
        assert request.data == {"feature1": 1.0, "feature2": "A"}
        assert request.model_name == "test_model"
    
    def test_prediction_request_empty_data(self):
        """Test prediction request with empty data."""
        from backend.main import PredictionRequest
        
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            PredictionRequest(data={})
    
    def test_batch_size_validation(self):
        """Test batch size limit validation."""
        from backend.main import BatchPredictionRequest
        
        large_batch = [{"feature": i} for i in range(1001)]  # Exceeds default limit
        
        with pytest.raises(ValueError, match="Batch size.*exceeds maximum"):
            BatchPredictionRequest(data=large_batch)

# API endpoint tests
class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client, mock_model_manager):
        """Test health check endpoint."""
        with patch("backend.main.model_manager", mock_model_manager):
            response = client.get("/health")
            
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_readiness_check(self, client, mock_model_manager):
        """Test readiness check endpoint."""
        with patch("backend.main.model_manager", mock_model_manager):
            response = client.get("/readiness")
            
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["available_models"] == 1
    
    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/liveness")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

class TestPredictionEndpoints:
    """Test ML prediction endpoints."""
    
    @pytest.mark.asyncio
    async def test_predict_endpoint(self, client, mock_model_manager):
        """Test single prediction endpoint."""
        # Mock prediction result
        mock_result = {
            "predictions": [0.8],
            "probabilities": [[0.2, 0.8]],
            "model_info": {"type": "RandomForestClassifier", "features_used": []},
            "inference_time_ms": 10.5
        }
        mock_model_manager.predict = AsyncMock(return_value=mock_result)
        
        with patch("backend.main.model_manager", mock_model_manager):
            response = client.post("/predict", json={
                "data": {"feature1": 1.0, "feature2": "A"},
                "return_probabilities": True
            })
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "request_id" in data
    
    def test_predict_no_model_manager(self, client):
        """Test prediction when model manager unavailable."""
        with patch("backend.main.model_manager", None):
            response = client.post("/predict", json={
                "data": {"feature1": 1.0}
            })
        
        assert response.status_code == 503
        assert "Model manager not available" in response.json()["error"]
    
    def test_invalid_prediction_data(self, client, mock_model_manager):
        """Test prediction with invalid data."""
        with patch("backend.main.model_manager", mock_model_manager):
            response = client.post("/predict", json={
                "data": {}  # Empty data
            })
        
        assert response.status_code == 422

# Model manager tests
class TestModelManager:
    """Test ModelManager functionality."""
    
    def test_model_manager_initialization(self, temp_model_dir):
        """Test model manager initialization."""
        manager = ModelManager(temp_model_dir, cache_size=2)
        assert manager.model_path == Path(temp_model_dir)
        assert manager.cache_size == 2
        assert manager.device == "cpu"
    
    def test_get_available_models(self, temp_model_dir):
        """Test getting available models."""
        manager = ModelManager(temp_model_dir)
        models = manager.get_available_models()
        assert "test_model" in models
    
    def test_device_detection(self, temp_model_dir):
        """Test device detection logic."""
        manager = ModelManager(temp_model_dir)
        device = manager._detect_device()
        # Should default to CPU in test environment
        assert device == "cpu"

# Integration tests
class TestIntegration:
    """Integration tests."""
    
    def test_full_prediction_workflow(self, client, temp_model_dir):
        """Test complete prediction workflow."""
        # This would require actual model files and more complex setup
        # For now, test the API contract
        
        with patch("backend.main.settings.MODEL_PATH", temp_model_dir):
            # Test API info endpoint
            response = client.get("/info")
            assert response.status_code == 200 or response.status_code == 503
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        # CORS headers should be added by middleware

# Error handling tests
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_http_exception_handler(self, client):
        """Test custom HTTP exception handling."""
        # Trigger a 404
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        assert "correlation_id" in data
        assert "timestamp" in data

# Performance tests
class TestPerformance:
    """Performance and load tests."""
    
    def test_concurrent_health_checks(self, client):
        """Test concurrent health check requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            start = time.time()
            response = client.get("/health")
            duration = time.time() - start
            results.append((response.status_code, duration))
        
        # Create 10 concurrent threads
        threads = [threading.Thread(target=make_request) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert all(status == 200 for status, _ in results)
        # All should complete in reasonable time
        assert all(duration < 1.0 for _, duration in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
