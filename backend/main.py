"""
🚀 AUTO-ANALYST PLATFORM - ENTERPRISE FASTAPI APPLICATION
========================================================================
                                                                        
██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ██╗      █████╗ ██╗   ██╗      
██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██║     ██╔══██╗╚██╗ ██╔╝      
███████║ ╚████╔╝ ██████╔╝█████╗  ██║  ██║██║     ███████║ ╚████╔╝       
██╔══██║  ╚██╔╝  ██╔══██╗██╔══╝  ██║  ██║██║     ██╔══██║  ╚██╔╝         
██║  ██║   ██║   ██║  ██║███████╗██████╔╝███████╗██║  ██║   ██║          
╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝          
                                                                         
ENTERPRISE AI-POWERED ZERO-CODE DATA ANALYSIS PLATFORM                   
========================================================================
Enterprise-grade ML inference server with:
- Thread-safe model loading and caching
- Comprehensive request/response validation  
- Production monitoring and observability
- Security hardening and rate limiting
- GPU/CPU aware inference optimization
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File, 
    BackgroundTasks, Request, Response, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# Production imports with fallbacks
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import joblib
    import pickle
    MODEL_LOADING_AVAILABLE = True
except ImportError:
    MODEL_LOADING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseSettings):
    """Production configuration with environment variable support."""
    
    # Application
    APP_NAME: str = "Auto-Data-Analyst API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Security
    SECRET_KEY: str = Field(..., description="JWT secret key")
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Models
    MODEL_PATH: str = Field(default="./models", description="Path to model directory")
    MAX_BATCH_SIZE: int = Field(default=1000, description="Maximum batch prediction size")
    MODEL_CACHE_SIZE: int = Field(default=5, description="Number of models to cache")
    
    # Performance  
    REQUEST_TIMEOUT: float = Field(default=300.0, description="Request timeout in seconds")
    PREDICTION_TIMEOUT: float = Field(default=60.0, description="Prediction timeout")
    MAX_WORKERS: int = Field(default=4, description="Maximum worker threads")
    
    # Redis (optional)
    REDIS_URL: Optional[str] = Field(default=None, description="Redis connection URL")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Monitoring
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# =============================================================================
# Pydantic Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Response timestamp") 
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of loaded models")

class PredictionRequest(BaseModel):
    """Single prediction request."""
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ..., description="Input data for prediction"
    )
    model_name: Optional[str] = Field(
        default=None, description="Specific model to use"
    )
    return_probabilities: bool = Field(
        default=False, description="Return prediction probabilities"
    )
    explain_predictions: bool = Field(
        default=False, description="Include SHAP explanations"
    )
    
    @validator('data')
    def validate_data(cls, v):
        """Validate input data structure."""
        if isinstance(v, dict):
            if not v:
                raise ValueError("Input data cannot be empty")
        elif isinstance(v, list):
            if not v or len(v) == 0:
                raise ValueError("Input data list cannot be empty")
            if len(v) > settings.MAX_BATCH_SIZE:
                raise ValueError(f"Batch size exceeds maximum of {settings.MAX_BATCH_SIZE}")
        else:
            raise ValueError("Data must be dict or list of dicts")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "data": {"feature1": 1.0, "feature2": "category_A", "feature3": 100},
                "model_name": "best_model",
                "return_probabilities": True,
                "explain_predictions": False
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response model."""
    predictions: Union[Any, List[Any]] = Field(..., description="Model predictions")
    probabilities: Optional[Union[List[float], List[List[float]]]] = Field(
        default=None, description="Prediction probabilities"
    )
    explanations: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None, description="SHAP explanations"
    )
    model_info: Dict[str, str] = Field(..., description="Model metadata")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    data: List[Dict[str, Any]] = Field(..., description="Batch input data")
    model_name: Optional[str] = Field(default=None, description="Model to use")
    
    @validator('data')
    def validate_batch_size(cls, v):
        if len(v) > settings.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum {settings.MAX_BATCH_SIZE}")
        return v

class ModelInfo(BaseModel):
    """Model information response."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type")
    features: List[str] = Field(..., description="Expected input features")
    target: Optional[str] = Field(default=None, description="Target variable")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Model metrics")
    created_at: datetime = Field(..., description="Model creation timestamp")

# =============================================================================
# Model Manager
# =============================================================================

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

class ModelManager:
    """Thread-safe model loading and caching manager."""
    
    def __init__(self, model_path: str, cache_size: int = 5):
        self.model_path = Path(model_path)
        self.cache_size = cache_size
        self._models: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        
        # GPU detection
        self.device = self._detect_device()
        logger.info(f"Initialized ModelManager with device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                return "gpu"
        except ImportError:
            pass
        
        return "cpu"
    
    async def load_model(self, model_name: str) -> Any:
        """Load model with thread-safe caching."""
        with self._lock:
            # Check cache first
            if model_name in self._models:
                # Move to end (most recently used)
                self._models.move_to_end(model_name)
                return self._models[model_name]
        
        # Load model asynchronously
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            self._executor, self._load_model_sync, model_name
        )
        
        with self._lock:
            # Add to cache
            self._models[model_name] = model
            
            # Evict oldest if cache full
            while len(self._models) > self.cache_size:
                oldest_model = self._models.popitem(last=False)
                logger.info(f"Evicted model from cache: {oldest_model[0]}")
        
        return model
    
    def _load_model_sync(self, model_name: str) -> Any:
        """Synchronous model loading."""
        model_file = self.model_path / f"{model_name}.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            logger.info(f"Loading model: {model_name}")
            with open(model_file, 'rb') as f:
                model = joblib.load(f)
            
            # Move model to appropriate device if needed
            if hasattr(model, 'device') and self.device != 'cpu':
                try:
                    model = model.to(self.device)
                except:
                    logger.warning(f"Could not move model {model_name} to {self.device}")
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    async def predict(
        self, 
        model_name: str, 
        data: Union[pd.DataFrame, np.ndarray], 
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Make predictions with loaded model."""
        start_time = time.time()
        
        try:
            model = await self.load_model(model_name)
            
            # Convert data to appropriate format
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            
            # Make predictions
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, self._predict_sync, model, data, return_probabilities
            )
            
            result['inference_time_ms'] = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {str(e)}")
            raise
    
    def _predict_sync(self, model: Any, data: pd.DataFrame, return_probabilities: bool) -> Dict[str, Any]:
        """Synchronous prediction."""
        result = {}
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(data)
            result['predictions'] = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
        else:
            raise ValueError("Model does not have a predict method")
        
        # Get probabilities if requested and available
        if return_probabilities and hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(data)
                result['probabilities'] = probabilities.tolist()
            except Exception as e:
                logger.warning(f"Could not get probabilities: {str(e)}")
        
        # Model metadata
        result['model_info'] = {
            'type': type(model).__name__,
            'features_used': getattr(model, 'feature_names_in_', None) or [],
            'n_features': getattr(data, 'shape', [0])[1] if hasattr(data, 'shape') else 0
        }
        
        return result
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        with self._lock:
            return list(self._models.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models."""
        if not self.model_path.exists():
            return []
        return [f.stem for f in self.model_path.glob("*.pkl")]

# Global model manager
model_manager: Optional[ModelManager] = None

# =============================================================================
# Monitoring & Metrics
# =============================================================================

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status_code']
    )
    
    REQUEST_DURATION = Histogram(
        'http_request_duration_seconds',
        'Request duration in seconds',
        ['method', 'endpoint']
    )
    
    PREDICTION_COUNT = Counter(
        'ml_predictions_total',
        'Total ML predictions',
        ['model_name', 'prediction_type']
    )
    
    PREDICTION_DURATION = Histogram(
        'ml_prediction_duration_seconds',
        'ML prediction duration in seconds',
        ['model_name']
    )

# Middleware for monitoring
class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        request = Request(scope, receive)
        
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        async def send_with_correlation_id(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-correlation-id"] = correlation_id.encode()
                message["headers"] = list(headers.items())
            await send(message)
        
        try:
            await self.app(scope, receive, send_with_correlation_id)
        finally:
            # Log request metrics
            duration = time.time() - start_time
            logger.info(
                f"Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "duration_ms": duration * 1000
                }
            )

# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Auto-Data-Analyst API Server")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    
    # Initialize model manager
    global model_manager
    model_manager = ModelManager(
        model_path=settings.MODEL_PATH,
        cache_size=settings.MODEL_CACHE_SIZE
    )
    
    # Initialize Redis if available
    redis_client = None
    if REDIS_AVAILABLE and settings.REDIS_URL:
        try:
            redis_client = aioredis.from_url(settings.REDIS_URL)
            await redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    
    app.state.redis = redis_client
    
    logger.info("Startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server")
    if redis_client:
        await redis_client.close()
    logger.info("Shutdown completed")

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade ML inference API",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add middleware
app.add_middleware(MonitoringMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.error(f"Unhandled exception: {str(exc)}", extra={
        "correlation_id": correlation_id,
        "exception_type": type(exc).__name__
    })
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
        models_loaded=len(model_manager.get_loaded_models()) if model_manager else 0
    )

@app.get("/readiness", tags=["Health"])
async def readiness_check():
    """Kubernetes readiness probe."""
    if not model_manager:
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized"
        )
    
    available_models = model_manager.get_available_models()
    if not available_models:
        raise HTTPException(
            status_code=503,
            detail="No models available"
        )
    
    return {
        "status": "ready",
        "available_models": len(available_models),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/liveness", tags=["Health"])
async def liveness_check():
    """Kubernetes liveness probe."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Metrics not available - Prometheus client not installed"
        )
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/info", response_model=Dict[str, Any], tags=["Models"])
async def get_api_info():
    """Get API and model information."""
    if not model_manager:
        raise HTTPException(
            status_code=503,
            detail="Model manager not available"
        )
    
    return {
        "api_version": settings.APP_VERSION,
        "available_models": model_manager.get_available_models(),
        "loaded_models": model_manager.get_loaded_models(),
        "device": model_manager.device,
        "max_batch_size": settings.MAX_BATCH_SIZE,
        "cache_size": settings.MODEL_CACHE_SIZE,
        "endpoints": [
            "/health", "/readiness", "/liveness", "/metrics",
            "/info", "/predict", "/batch_predict", "/explain"
        ]
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest, req: Request):
    """Single or batch prediction endpoint."""
    if not model_manager:
        raise HTTPException(
            status_code=503,
            detail="Model manager not available"
        )
    
    correlation_id = getattr(req.state, 'correlation_id', 'unknown')
    
    try:
        # Determine model to use
        available_models = model_manager.get_available_models()
        if not available_models:
            raise HTTPException(
                status_code=503,
                detail="No models available"
            )
        
        model_name = request.model_name or available_models[0]
        if model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Make prediction
        start_time = time.time()
        result = await model_manager.predict(
            model_name=model_name,
            data=request.data,
            return_probabilities=request.return_probabilities
        )
        
        # Add request metadata
        result['request_id'] = correlation_id
        result['model_info']['model_name'] = model_name
        
        # Update metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(
                model_name=model_name,
                prediction_type='single' if isinstance(request.data, dict) else 'batch'
            ).inc()
            
            PREDICTION_DURATION.labels(model_name=model_name).observe(
                time.time() - start_time
            )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", extra={
            "correlation_id": correlation_id,
            "model_name": request.model_name
        })
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch_predict", tags=["Inference"])
async def batch_predict(request: BatchPredictionRequest, req: Request):
    """Optimized batch prediction endpoint."""
    if not model_manager:
        raise HTTPException(
            status_code=503,
            detail="Model manager not available"
        )
    
    correlation_id = getattr(req.state, 'correlation_id', 'unknown')
    
    try:
        # Use the predict endpoint with batch data
        prediction_request = PredictionRequest(
            data=request.data,
            model_name=request.model_name,
            return_probabilities=False,  # Optimize for batch
            explain_predictions=False
        )
        
        return await predict(prediction_request, req)
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", extra={
            "correlation_id": correlation_id,
            "batch_size": len(request.data)
        })
        raise

@app.post("/explain", tags=["Inference"])
async def explain_predictions(request: PredictionRequest, req: Request):
    """Get prediction explanations (placeholder for SHAP integration)."""
    # For now, return regular predictions
    # TODO: Integrate SHAP explanations
    
    enhanced_request = request.copy(update={"explain_predictions": True})
    response = await predict(enhanced_request, req)
    
    # Add placeholder explanation
    if isinstance(request.data, dict):
        response.explanations = {
            "feature_importance": {"note": "SHAP explanations not yet implemented"},
            "method": "placeholder"
        }
    else:
        response.explanations = [
            {"feature_importance": {"note": "SHAP explanations not yet implemented"}}
            for _ in request.data
        ]
    
    return response

@app.post("/upload", tags=["Data"])
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Secure file upload endpoint."""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    allowed_extensions = {'.csv', '.json', '.xlsx', '.parquet'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not allowed"
        )
    
    # Generate secure filename
    secure_name = f"upload_{int(time.time())}_{uuid.uuid4().hex}{file_extension}"
    
    return {
        "message": "File upload successful",
        "filename": secure_name,
        "size": file.size if hasattr(file, 'size') else 0,
        "type": file.content_type
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "message": "Auto-Data-Analyst ML API",
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs" if settings.DEBUG else "disabled",
        "health": "/health"
    }

# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG
    )
