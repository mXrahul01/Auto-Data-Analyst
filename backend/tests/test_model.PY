# tests/test_model.py
import pytest
import tempfile
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from backend.main import ModelManager

class MockModel:
    """Mock ML model for testing."""
    
    def __init__(self, name="MockModel"):
        self.name = name
        self.feature_names_in_ = ["feature1", "feature2"]
    
    def predict(self, X):
        """Mock predict method."""
        if isinstance(X, pd.DataFrame):
            return np.array([0.5] * len(X))
        return np.array([0.5])
    
    def predict_proba(self, X):
        """Mock predict_proba method."""
        if isinstance(X, pd.DataFrame):
            return np.array([[0.5, 0.5]] * len(X))
        return np.array([[0.5, 0.5]])

@pytest.fixture
def temp_model_dir_with_models():
    """Create temporary directory with mock model files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        
        # Create mock models
        mock_model1 = MockModel("model1")
        mock_model2 = MockModel("model2")
        
        # Save models
        joblib.dump(mock_model1, model_dir / "model1.pkl")
        joblib.dump(mock_model2, model_dir / "model2.pkl")
        
        yield temp_dir

@pytest.fixture
def model_manager(temp_model_dir_with_models):
    """Model manager fixture with mock models."""
    return ModelManager(temp_model_dir_with_models, cache_size=2)

class TestModelManager:
    """Test ModelManager functionality."""
    
    def test_initialization(self, temp_model_dir_with_models):
        """Test model manager initialization."""
        manager = ModelManager(temp_model_dir_with_models, cache_size=3)
        
        assert manager.model_path == Path(temp_model_dir_with_models)
        assert manager.cache_size == 3
        assert len(manager._models) == 0
        assert manager.device == "cpu"
    
    def test_get_available_models(self, model_manager):
        """Test getting available models."""
        available = model_manager.get_available_models()
        assert "model1" in available
        assert "model2" in available
        assert len(available) == 2
    
    def test_get_loaded_models_empty(self, model_manager):
        """Test getting loaded models when none loaded."""
        loaded = model_manager.get_loaded_models()
        assert loaded == []
    
    @pytest.mark.asyncio
    async def test_load_model(self, model_manager):
        """Test loading a model."""
        model = await model_manager.load_model("model1")
        
        assert model is not None
        assert model.name == "MockModel"
        
        # Check cache
        loaded_models = model_manager.get_loaded_models()
        assert "model1" in loaded_models
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, model_manager):
        """Test loading a nonexistent model."""
        with pytest.raises(FileNotFoundError):
            await model_manager.load_model("nonexistent_model")
    
    @pytest.mark.asyncio
    async def test_cache_management(self, model_manager):
        """Test model cache management."""
        # Set small cache size
        model_manager.cache_size = 1
        
        # Load first model
        model1 = await model_manager.load_model("model1")
        assert len(model_manager._models) == 1
        
        # Load second model (should evict first)
        model2 = await model_manager.load_model("model2")
        assert len(model_manager._models) == 1
        assert "model2" in model_manager._models
        assert "model1" not in model_manager._models
    
    @pytest.mark.asyncio
    async def test_cache_lru_behavior(self, model_manager):
        """Test LRU cache behavior."""
        # Load both models
        await model_manager.load_model("model1")
        await model_manager.load_model("model2")
        
        # Access model1 again (should move to end)
        await model_manager.load_model("model1")
        
        # Check order in cache
        cached_models = list(model_manager._models.keys())
        assert cached_models[-1] == "model1"  # Most recently used
    
    @pytest.mark.asyncio
    async def test_predict_single(self, model_manager):
        """Test single prediction."""
        data = {"feature1": 1.0, "feature2": 2.0}
        
        result = await model_manager.predict("model1", data)
        
        assert "predictions" in result
        assert "model_info" in result
        assert "inference_time_ms" in result
        
        predictions = result["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) == 1
    
    @pytest.mark.asyncio
    async def test_predict_batch(self, model_manager):
        """Test batch prediction."""
        data = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0}
        ]
        
        result = await model_manager.predict("model1", data)
        
        predictions = result["predictions"]
        assert len(predictions) == 2
    
    @pytest.mark.asyncio
    async def test_predict_with_probabilities(self, model_manager):
        """Test prediction with probabilities."""
        data = {"feature1": 1.0, "feature2": 2.0}
        
        result = await model_manager.predict(
            "model1", data, return_probabilities=True
        )
        
        assert "probabilities" in result
        probabilities = result["probabilities"]
        assert isinstance(probabilities, list)
        assert len(probabilities[0]) == 2  # Binary classification
    
    @pytest.mark.asyncio
    async def test_concurrent_model_loading(self, model_manager):
        """Test concurrent model loading."""
        # Create multiple concurrent load tasks
        tasks = [
            model_manager.load_model("model1"),
            model_manager.load_model("model2"),
            model_manager.load_model("model1"),  # Duplicate
        ]
        
        models = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(model is not None for model in models)
        
        # Cache should contain both models
        loaded = model_manager.get_loaded_models()
        assert "model1" in loaded
        assert "model2" in loaded
    
    @pytest.mark.asyncio
    async def test_predict_error_handling(self, model_manager):
        """Test prediction error handling."""
        # Test with model that doesn't exist
        with pytest.raises(Exception):
            await model_manager.predict("nonexistent", {"feature": 1})
    
    def test_thread_safety(self, model_manager):
        """Test thread safety of cache operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def load_model_thread(model_name):
            try:
                # Use asyncio.run for each thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                model = loop.run_until_complete(
                    model_manager.load_model(model_name)
                )
                results.append((model_name, model is not None))
            except Exception as e:
                errors.append(e)
            finally:
                loop.close()
        
        # Create multiple threads loading different models
        threads = []
        for i in range(5):
            model_name = "model1" if i % 2 == 0 else "model2"
            thread = threading.Thread(target=load_model_thread, args=(model_name,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(success for _, success in results)

class TestModelManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_model_directory(self):
        """Test behavior with empty model directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(temp_dir)
            available = manager.get_available_models()
            assert available == []
    
    def test_nonexistent_model_directory(self):
        """Test behavior with nonexistent model directory."""
        manager = ModelManager("/nonexistent/path")
        available = manager.get_available_models()
        assert available == []
    
    def test_corrupted_model_file(self, temp_model_dir_with_models):
        """Test loading corrupted model file."""
        model_dir = Path(temp_model_dir_with_models)
        
        # Create corrupted model file
        corrupted_file = model_dir / "corrupted.pkl"
        corrupted_file.write_text("not a valid pickle file")
        
        manager = ModelManager(temp_model_dir_with_models)
        
        # Should raise an exception
        with pytest.raises(Exception):
            asyncio.run(manager.load_model("corrupted"))
    
    @pytest.mark.asyncio
    async def test_model_without_predict_method(self, model_manager):
        """Test model without predict method."""
        # Mock a model without predict method
        bad_model = object()
        
        # Manually add to cache
        model_manager._models["bad_model"] = bad_model
        
        with pytest.raises(ValueError, match="Model does not have a predict method"):
            await model_manager.predict("bad_model", {"feature": 1})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
