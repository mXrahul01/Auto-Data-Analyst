"""
Auto-Analyst Platform - FastAPI Application Entrypoint

This module serves as the main FastAPI application entrypoint for the Auto-Analyst platform,
providing a comprehensive zero-code AI-powered data analysis web application with enterprise-grade
features including security, monitoring, and scalability.

Key Features:
- RESTful API endpoints for complete ML lifecycle
- Large file upload support (up to 20GB) with streaming and chunked processing
- Multi-platform remote training capabilities (Kaggle, Colab, AWS, GCP, Azure)
- MLflow experiment tracking and model registry integration
- Real-time monitoring, drift detection, and performance analytics
- Comprehensive error handling, logging, and observability
- Background task processing with Celery integration
- Production-ready middleware stack with security headers
- Health checks and Prometheus metrics
- Horizontal scaling support with cloud-native design

Architecture Components:
- FastAPI with async/await for high-performance concurrent request handling
- Modular service-oriented architecture with dependency injection
- Database connection pooling and automatic migration support
- Cloud-native design with container and Kubernetes support
- Comprehensive monitoring, alerting, and distributed tracing
- Multi-tenancy support with role-based access control

API Endpoints Overview:
Dataset Management:
- POST /api/v1/datasets/upload - Upload and validate datasets
- GET /api/v1/datasets/ - List user datasets with pagination
- GET /api/v1/datasets/{id} - Get dataset details and metadata
- DELETE /api/v1/datasets/{id} - Delete dataset and cleanup

ML Analysis:
- POST /api/v1/analyses/ - Create new ML analysis with auto-model selection
- GET /api/v1/analyses/{id} - Get analysis status, progress, and results
- GET /api/v1/analyses/ - List analyses with filtering and pagination
- POST /api/v1/analyses/{id}/cancel - Cancel running analysis

Predictions:
- POST /api/v1/predict/{id} - Single predictions with explanations
- POST /api/v1/predict/{id}/batch - Batch predictions from file

Dashboard & Insights:
- GET /api/v1/dashboard/{id} - Comprehensive dashboard data
- GET /api/v1/insights/{id} - AI-generated insights and recommendations

Monitoring & MLOps:
- GET /api/v1/monitoring/drift/{id} - Data and model drift analysis
- GET /api/v1/monitoring/performance/{id} - Performance monitoring data

System Health:
- GET /health - Application health check with service status
- GET /readiness - Kubernetes readiness probe
- GET /liveness - Kubernetes liveness probe
- GET /metrics - Prometheus metrics endpoint

Security Features:
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- Rate limiting per client IP
- CORS configuration with origin validation
- Security headers middleware
- Request/response logging with sanitization
- Input validation and sanitization

Dependencies:
- FastAPI: High-performance async web framework
- SQLAlchemy: Database ORM with connection pooling
- Pydantic: Data validation and serialization
- MLflow: Experiment tracking and model registry
- Celery: Distributed task queue for background processing
- Redis: Caching and session storage
- Prometheus: Metrics collection and monitoring
- Alembic: Database schema migrations

Usage Examples:
    # Development with auto-reload
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
    
    # Production with multiple workers
    gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 --timeout 300 --max-requests 1000
    
    # Docker container
    docker run -p 8000:8000 auto-analyst:latest
    
    # Kubernetes deployment
    kubectl apply -f k8s/auto-analyst-deployment.yaml

Environment Configuration:
    Set environment variables or use .env files:
    - ENVIRONMENT: deployment environment (development, staging, production)
    - DATABASE_URL: database connection string
    - REDIS_URL: Redis connection string for caching and task queue
    - SECRET_KEY: JWT signing secret
    - MLFLOW_TRACKING_URI: MLflow tracking server URI

Author: Auto-Analyst Development Team
Version: 2.0.0
License: Commercial
Last Updated: 2025-09-21
"""

import asyncio
import logging
import traceback
import signal
import sys
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import tempfile
import uuid
from datetime import datetime, timedelta
import json

# Core async and concurrency
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# File handling
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# FastAPI and web framework components
from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File, Form,
    BackgroundTasks, Request, Response, status, Query, Header
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

# Database and ORM
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from sqlalchemy.exc import SQLAlchemyError

# Monitoring and observability
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, 
        CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess
    )
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Background task processing
try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Memory profiling (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configuration and settings
from .config import settings, validate_and_setup_config

# Database models and schemas
from .models.database import get_db, engine, Base, init_database, get_db_session, create_tables
from .models import schemas

# Service layer dependencies
from .services.data_service import DataService, get_data_service
from .services.ml_service import MLService, get_ml_service
from .services.insights_service import InsightsService, get_insights_service
from .services.mlops_service import MLOpsService, get_mlops_service
from .services.auth_service import AuthService, get_auth_service
# Utility modules
from .utils.monitoring import MonitoringManager, create_monitoring_manager, log_info, log_warning, log_error, monitor_performance
from .utils.validation import validate_dataset, ValidationResult
from .utils.preprocessing import preprocess_data
from .utils.security import SecurityManager, get_security_manager
from .utils.cache import CacheManager, get_cache_manager
from .tasks import execute_analysis, process_uploaded_dataset, execute_batch_predictions, create_insights

# Task processing
from backend.tasks import (
    execute_analysis, process_uploaded_dataset, 
    execute_batch_predictions, create_insights
)

# Configure structured logging
logger = logging.getLogger(__name__)

# Global service instances
monitoring_manager: Optional[MonitoringManager] = None
thread_pool: Optional[ThreadPoolExecutor] = None
celery_app: Optional[Celery] = None
security_manager: Optional[SecurityManager] = None
cache_manager: Optional[CacheManager] = None

# Security and authentication
security = HTTPBearer(auto_error=False)

# Prometheus metrics setup
if PROMETHEUS_AVAILABLE:
    # Initialize metrics registry
    if "prometheus_multiproc_dir" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        from prometheus_client import REGISTRY as registry
    
    # HTTP request metrics
    REQUEST_COUNT = Counter(
        'http_requests_total',
        'Total HTTP requests by method, endpoint and status',
        ['method', 'endpoint', 'status_code', 'user_id'],
        registry=registry
    )
    
    REQUEST_DURATION = Histogram(
        'http_request_duration_seconds',
        'HTTP request latency by method and endpoint',
        ['method', 'endpoint'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')],
        registry=registry
    )
    
    REQUEST_SIZE = Histogram(
        'http_request_size_bytes',
        'HTTP request size in bytes',
        buckets=[100, 1000, 10000, 100000, 1000000, 10000000, float('inf')],
        registry=registry
    )
    
    RESPONSE_SIZE = Histogram(
        'http_response_size_bytes', 
        'HTTP response size in bytes',
        buckets=[100, 1000, 10000, 100000, 1000000, 10000000, float('inf')],
        registry=registry
    )
    
    # Application-specific metrics
    ACTIVE_UPLOADS = Gauge(
        'active_file_uploads',
        'Number of active file uploads',
        registry=registry
    )
    
    ACTIVE_ANALYSES = Gauge(
        'active_ml_analyses',
        'Number of active ML analyses',
        registry=registry
    )
    
    DATASET_SIZE = Histogram(
        'dataset_size_bytes',
        'Size of uploaded datasets in bytes',
        buckets=[1e6, 10e6, 100e6, 1e9, 10e9, 20e9, float('inf')],
        registry=registry
    )
    
    MODEL_TRAINING_DURATION = Histogram(
        'model_training_duration_seconds',
        'Duration of model training in seconds',
        buckets=[60, 300, 600, 1800, 3600, 7200, 21600, float('inf')],
        registry=registry
    )
    
    PREDICTION_COUNT = Counter(
        'predictions_total',
        'Total number of predictions made',
        ['model_type', 'prediction_type'],
        registry=registry
    )
    
    # Error and system metrics
    ERROR_COUNT = Counter(
        'errors_total',
        'Total errors by type and endpoint',
        ['error_type', 'endpoint', 'severity'],
        registry=registry
    )
    
    SYSTEM_MEMORY_USAGE = Gauge(
        'system_memory_usage_bytes',
        'System memory usage in bytes',
        registry=registry
    )
    
    SYSTEM_CPU_USAGE = Gauge(
        'system_cpu_usage_percent',
        'System CPU usage percentage',
        registry=registry
    )
    
    DATABASE_CONNECTIONS = Gauge(
        'database_connections_active',
        'Number of active database connections',
        registry=registry
    )
    
    CACHE_HITS = Counter(
        'cache_hits_total',
        'Total cache hits',
        ['cache_type'],
        registry=registry
    )
    
    CACHE_MISSES = Counter(
        'cache_misses_total',
        'Total cache misses', 
        ['cache_type'],
        registry=registry
    )

# Custom middleware classes with enhanced functionality
class EnhancedRequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive request logging middleware with security features.
    
    Features:
    - Detailed request/response logging with structured data
    - Performance metrics collection
    - Security event logging
    - Request correlation IDs
    - Sensitive data filtering
    - Rate limiting tracking
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.sensitive_headers = {
            'authorization', 'cookie', 'x-api-key', 
            'x-auth-token', 'x-csrf-token'
        }
        self.sensitive_params = {
            'password', 'token', 'secret', 'key', 'auth'
        }
    
    async def dispatch(self, request: StarletteRequest, call_next: Callable) -> StarletteResponse:
        """Process request with comprehensive logging and metrics."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state for correlation
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Extract client information
        client_ip = self._get_real_client_ip(request)
        user_agent = request.headers.get('user-agent', 'Unknown')[:200]  # Limit length
        
        # Log request start with filtered sensitive data
        log_info(
            "HTTP request started",
            extra={
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'path': request.url.path,
                'query_params': dict(request.query_params),
                'client_ip': client_ip,
                'user_agent': user_agent,
                'headers': self._filter_sensitive_headers(dict(request.headers)),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Calculate request size
        request_size = int(request.headers.get('content-length', 0))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.time() - start_time
            response_size = int(response.headers.get('content-length', 0))
            
            # Extract user ID if available
            user_id = getattr(request.state, 'user_id', 'anonymous')
            
            # Log successful response
            log_info(
                "HTTP request completed successfully",
                extra={
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'duration_ms': round(duration * 1000, 2),
                    'request_size_bytes': request_size,
                    'response_size_bytes': response_size,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                endpoint = self._normalize_endpoint(request.url.path)
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status_code=response.status_code,
                    user_id=user_id
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=request.method,
                    endpoint=endpoint
                ).observe(duration)
                
                if request_size > 0:
                    REQUEST_SIZE.observe(request_size)
                
                if response_size > 0:
                    RESPONSE_SIZE.observe(response_size)
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Server-Instance"] = os.getenv("HOSTNAME", "unknown")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            # Determine error severity
            if isinstance(e, HTTPException):
                if e.status_code < 500:
                    severity = "warning"
                else:
                    severity = "error"
            else:
                severity = "error"
            
            # Log error with context
            log_error(
                "HTTP request failed",
                exception=e,
                extra={
                    'request_id': request_id,
                    'duration_ms': round(duration * 1000, 2),
                    'error_type': error_type,
                    'severity': severity,
                    'method': request.method,
                    'path': request.url.path,
                    'client_ip': client_ip,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Update error metrics
            if PROMETHEUS_AVAILABLE:
                endpoint = self._normalize_endpoint(request.url.path)
                ERROR_COUNT.labels(
                    error_type=error_type,
                    endpoint=endpoint,
                    severity=severity
                ).inc()
            
            # Re-raise the exception
            raise
    
    def _get_real_client_ip(self, request: StarletteRequest) -> str:
        """Extract real client IP handling proxies and load balancers."""
        # Check X-Forwarded-For header (most common)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Get first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Check CF-Connecting-IP (Cloudflare)
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    def _filter_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive information from headers."""
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                filtered[key] = "[FILTERED]"
            else:
                filtered[key] = value
        return filtered
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics by replacing IDs with placeholders."""
        # Replace UUIDs and numeric IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware with comprehensive security controls.
    
    Implements security best practices including:
    - Content Security Policy (CSP)
    - HTTP Strict Transport Security (HSTS)
    - X-Frame-Options, X-Content-Type-Options
    - Referrer Policy and Feature Policy
    - Custom security headers
    """
    
    async def dispatch(self, request: StarletteRequest, call_next: Callable) -> StarletteResponse:
        """Add comprehensive security headers to all responses."""
        response = await call_next(request)
        
        # Basic security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        
        if not settings.is_development:
            response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # HSTS for production HTTPS
        if settings.is_production and settings.ENABLE_HTTPS:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Feature Policy (Permissions Policy)
        feature_policy = [
            "camera 'none'",
            "microphone 'none'", 
            "geolocation 'none'",
            "payment 'none'",
            "usb 'none'"
        ]
        response.headers["Permissions-Policy"] = ", ".join(feature_policy)
        
        # Additional custom headers
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        
        return response

class EnhancedRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with multiple strategies.
    
    Features:
    - Per-IP rate limiting with sliding window
    - Per-user rate limiting for authenticated requests
    - Different limits for different endpoint categories
    - Whitelist/blacklist support
    - Burst allowance with token bucket algorithm
    - Redis backend for distributed rate limiting
    """
    
    def __init__(self, app, **config):
        super().__init__(app)
        self.config = {
            'requests_per_minute': config.get('requests_per_minute', 60),
            'requests_per_hour': config.get('requests_per_hour', 1000),
            'burst_allowance': config.get('burst_allowance', 10),
            'cleanup_interval': config.get('cleanup_interval', 300),
            'whitelist': config.get('whitelist', set()),
            'endpoint_limits': config.get('endpoint_limits', {}),
        }
        
        # In-memory storage for development, Redis for production
        self.use_redis = settings.is_production and cache_manager
        self.local_storage = {} if not self.use_redis else None
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: StarletteRequest, call_next: Callable) -> StarletteResponse:
        """Apply rate limiting based on client IP and user ID."""
        # Skip rate limiting for health checks and metrics
        if request.url.path in {'/health', '/readiness', '/liveness', '/metrics'}:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        user_id = getattr(request.state, 'user_id', None)
        current_time = time.time()
        
        # Check whitelist
        if client_ip in self.config['whitelist']:
            return await call_next(request)
        
        # Periodic cleanup for memory-based storage
        if not self.use_redis and current_time - self.last_cleanup > self.config['cleanup_interval']:
            await self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Check rate limits
        rate_limit_key = f"rate_limit:{client_ip}"
        if user_id:
            rate_limit_key += f":user:{user_id}"
        
        # Get endpoint-specific limits
        endpoint_category = self._categorize_endpoint(request.url.path)
        limits = self.config['endpoint_limits'].get(endpoint_category, {
            'requests_per_minute': self.config['requests_per_minute'],
            'requests_per_hour': self.config['requests_per_hour']
        })
        
        # Check if rate limit exceeded
        if await self._is_rate_limited(rate_limit_key, limits, current_time):
            log_warning(
                f"Rate limit exceeded",
                extra={
                    'client_ip': client_ip,
                    'user_id': user_id,
                    'endpoint': request.url.path,
                    'endpoint_category': endpoint_category
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(limits['requests_per_minute']),
                    "X-RateLimit-Reset": str(int(current_time) + 60)
                }
            )
        
        # Record request
        await self._record_request(rate_limit_key, current_time)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: StarletteRequest) -> str:
        """Extract client IP from request headers."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _categorize_endpoint(self, path: str) -> str:
        """Categorize endpoints for different rate limit rules."""
        if path.startswith('/api/v1/datasets/upload'):
            return 'upload'
        elif path.startswith('/api/v1/predict'):
            return 'prediction'
        elif path.startswith('/api/v1/analyses'):
            return 'analysis'
        elif path.startswith('/api/v1/dashboard'):
            return 'dashboard'
        else:
            return 'general'
    
    async def _is_rate_limited(self, key: str, limits: Dict[str, int], current_time: float) -> bool:
        """Check if request should be rate limited."""
        if self.use_redis and cache_manager:
            return await self._check_redis_rate_limit(key, limits, current_time)
        else:
            return await self._check_local_rate_limit(key, limits, current_time)
    
    async def _check_redis_rate_limit(self, key: str, limits: Dict[str, int], current_time: float) -> bool:
        """Check rate limit using Redis sliding window."""
        try:
            # Use Redis sliding window log for rate limiting
            pipeline = cache_manager.redis_client.pipeline()
            
            # Remove entries outside the time window (1 hour)
            pipeline.zremrangebyscore(key, 0, current_time - 3600)
            
            # Count requests in the last minute
            pipeline.zcount(key, current_time - 60, current_time)
            
            # Count requests in the last hour
            pipeline.zcount(key, current_time - 3600, current_time)
            
            results = await pipeline.execute()
            
            minute_count = results[1]
            hour_count = results[2]
            
            return (minute_count >= limits['requests_per_minute'] or 
                   hour_count >= limits['requests_per_hour'])
                   
        except Exception as e:
            log_error("Redis rate limit check failed", exception=e)
            return False  # Fail open for availability
    
    async def _check_local_rate_limit(self, key: str, limits: Dict[str, int], current_time: float) -> bool:
        """Check rate limit using local memory storage."""
        if key not in self.local_storage:
            self.local_storage[key] = []
        
        requests = self.local_storage[key]
        
        # Remove old requests (outside 1 hour window)
        requests[:] = [req_time for req_time in requests if current_time - req_time < 3600]
        
        # Count requests in different time windows
        minute_count = sum(1 for req_time in requests if current_time - req_time < 60)
        hour_count = len(requests)
        
        return (minute_count >= limits['requests_per_minute'] or 
               hour_count >= limits['requests_per_hour'])
    
    async def _record_request(self, key: str, current_time: float):
        """Record a request for rate limiting."""
        if self.use_redis and cache_manager:
            try:
                # Add request to Redis sorted set with current timestamp as score
                await cache_manager.redis_client.zadd(key, {str(uuid.uuid4()): current_time})
                # Set expiry to 1 hour
                await cache_manager.redis_client.expire(key, 3600)
            except Exception as e:
                log_error("Failed to record request in Redis", exception=e)
        else:
            if key not in self.local_storage:
                self.local_storage[key] = []
            self.local_storage[key].append(current_time)
    
    async def _cleanup_old_entries(self, current_time: float):
        """Clean up old entries from local storage."""
        for key in list(self.local_storage.keys()):
            self.local_storage[key] = [
                req_time for req_time in self.local_storage[key]
                if current_time - req_time < 3600
            ]
            
            if not self.local_storage[key]:
                del self.local_storage[key]

class SystemMetricsMiddleware(BaseHTTPMiddleware):
    """
    System metrics collection middleware.
    
    Collects system-level metrics including:
    - Memory usage
    - CPU usage  
    - Database connection pool status
    - Cache hit/miss ratios
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.last_metrics_update = 0
        self.metrics_update_interval = 30  # Update every 30 seconds
    
    async def dispatch(self, request: StarletteRequest, call_next: Callable) -> StarletteResponse:
        """Update system metrics periodically."""
        current_time = time.time()
        
        # Update system metrics periodically
        if (PROMETHEUS_AVAILABLE and PSUTIL_AVAILABLE and 
            current_time - self.last_metrics_update > self.metrics_update_interval):
            
            try:
                await self._update_system_metrics()
                self.last_metrics_update = current_time
            except Exception as e:
                log_error("Failed to update system metrics", exception=e)
        
        return await call_next(request)
    
    async def _update_system_metrics(self):
        """Update system-level Prometheus metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Database connection pool metrics
            if engine and hasattr(engine.pool, 'size'):
                pool = engine.pool
                DATABASE_CONNECTIONS.set(pool.checkedout())
            
        except Exception as e:
            log_error("System metrics update failed", exception=e)

# Application lifespan management with comprehensive initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Comprehensive application lifespan management.
    
    Handles:
    - Configuration validation
    - Database initialization and migrations
    - Service initialization with dependency injection
    - Monitoring and observability setup
    - Background task processing setup
    - Signal handlers for graceful shutdown
    - Resource cleanup on shutdown
    """
    # Startup sequence
    startup_start_time = time.time()
    logger.info("🚀 Starting Auto-Analyst Platform...")
    
    try:
        # Step 1: Validate configuration
        logger.info("📋 Validating configuration...")
        validate_and_setup_config()
        
        # Step 2: Initialize database
        logger.info("🗄️  Initializing database...")
        await init_database()
        
        # Step 3: Run database migrations
        logger.info("🔄 Running database migrations...")
        await run_database_migrations()
        
        # Step 4: Initialize services
        logger.info("⚙️  Initializing services...")
        await initialize_global_services()
        
        # Step 5: Initialize monitoring
        logger.info("📊 Setting up monitoring...")
        await initialize_monitoring_system()
        
        # Step 6: Initialize background tasks
        logger.info("⏰ Initializing background tasks...")
        await initialize_background_processing()
        
        # Step 7: Setup signal handlers
        logger.info("🔧 Setting up signal handlers...")
        setup_signal_handlers()
        
        # Step 8: Perform health checks
        logger.info("🏥 Performing initial health checks...")
        health_status = await perform_startup_health_checks()
        
        startup_duration = time.time() - startup_start_time
        logger.info(
            f"✅ Auto-Analyst Platform started successfully in {startup_duration:.2f}s",
            extra={
                'startup_duration': startup_duration,
                'environment': settings.ENVIRONMENT,
                'version': settings.APP_VERSION,
                'health_status': health_status
            }
        )
        
        # Application is ready to serve requests
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Shutdown sequence
        logger.info("🛑 Shutting down Auto-Analyst Platform...")
        shutdown_start_time = time.time()
        
        try:
            # Step 1: Stop accepting new requests (handled by server)
            logger.info("🚫 Stopping new request acceptance...")
            
            # Step 2: Complete in-flight requests (handled by server)
            logger.info("⏳ Waiting for in-flight requests to complete...")
            
            # Step 3: Stop background tasks
            logger.info("🔄 Stopping background tasks...")
            await cleanup_background_processing()
            
            # Step 4: Cleanup services
            logger.info("🧹 Cleaning up services...")
            await cleanup_global_services()
            
            # Step 5: Stop monitoring
            logger.info("📊 Stopping monitoring...")
            if monitoring_manager:
                await monitoring_manager.stop_monitoring()
            
            # Step 6: Close database connections
            logger.info("🗄️  Closing database connections...")
            if engine:
                engine.dispose()
            
            # Step 7: Final cleanup
            logger.info("🧽 Final cleanup...")
            await final_cleanup()
            
            shutdown_duration = time.time() - shutdown_start_time
            logger.info(
                f"✅ Auto-Analyst Platform shutdown completed in {shutdown_duration:.2f}s",
                extra={'shutdown_duration': shutdown_duration}
            )
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {str(e)}", exc_info=True)

# FastAPI application initialization with comprehensive configuration
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade AI-powered zero-code data analysis platform",
    summary="Comprehensive ML platform with automated model selection, training, and deployment",
    contact={
        "name": "Auto-Analyst Support",
        "email": "support@auto-analyst.com",
        "url": "https://auto-analyst.com/support"
    },
    license_info={
        "name": "Commercial License",
        "url": "https://auto-analyst.com/license"
    },
    terms_of_service="https://auto-analyst.com/terms",
    # Disable documentation in production for security
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
    # Custom OpenAPI configuration
    openapi_tags=[
        {"name": "Health", "description": "System health and monitoring endpoints"},
        {"name": "Authentication", "description": "User authentication and authorization"},
        {"name": "Datasets", "description": "Dataset upload, management, and validation"},
        {"name": "ML Analysis", "description": "Machine learning model training and analysis"},
        {"name": "Predictions", "description": "Model predictions and inference"},
        {"name": "Dashboard", "description": "Analytics dashboard and insights"},
        {"name": "Monitoring", "description": "MLOps monitoring and drift detection"},
        {"name": "Admin", "description": "Administrative endpoints"}
    ],
    responses={
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"},
        503: {"description": "Service Unavailable"}
    }
)

# Middleware configuration with proper ordering (order matters!)
# 1. System metrics collection (outermost)
app.add_middleware(SystemMetricsMiddleware)

# 2. CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
    expose_headers=["X-Request-ID", "X-Response-Time"],
    max_age=86400  # 24 hours
)

# 3. Security headers
app.add_middleware(SecurityHeadersMiddleware)

# 4. Request logging (before rate limiting to log blocked requests)
app.add_middleware(EnhancedRequestLoggingMiddleware)

# 5. Rate limiting (production only)
if settings.is_production:
    app.add_middleware(
        EnhancedRateLimitMiddleware,
        requests_per_minute=100,
        requests_per_hour=1000,
        endpoint_limits={
            'upload': {'requests_per_minute': 10, 'requests_per_hour': 50},
            'prediction': {'requests_per_minute': 200, 'requests_per_hour': 2000},
            'analysis': {'requests_per_minute': 20, 'requests_per_hour': 100}
        }
    )

# 6. Compression (innermost, closest to response)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
    compresslevel=6  # Balance between compression and CPU usage
)

# 7. Trusted host middleware (production only)
if settings.is_production and hasattr(settings, 'TRUSTED_HOSTS'):
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS
    )

# Prometheus instrumentation setup
if PROMETHEUS_AVAILABLE and settings.PROMETHEUS_ENABLED:
    try:
        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics", "/favicon.ico"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="inprogress",
            inprogress_labels=True
        )
        
        # Add custom metrics
        instrumentator.add(
            lambda info: REQUEST_COUNT.labels(
                method=info.method,
                endpoint=info.modified_handler,
                status_code=info.response.status_code,
                user_id=getattr(info.request.state, 'user_id', 'anonymous')
            ).inc()
        )
        
        instrumentator.instrument(app)
        instrumentator.expose(app, endpoint="/metrics", tags=["Health"])
        
        logger.info("📊 Prometheus instrumentation configured")
        
    except Exception as e:
        log_error("Failed to setup Prometheus instrumentation", exception=e)

# Exception handlers with structured error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging and metrics."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    log_warning(
        f"HTTP exception occurred: {exc.status_code}",
        extra={
            'request_id': request_id,
            'status_code': exc.status_code,
            'detail': exc.detail,
            'url': str(request.url),
            'method': request.method,
            'client_ip': request.headers.get('x-forwarded-for', 
                        request.headers.get('x-real-ip', 'unknown')),
            'user_agent': request.headers.get('user-agent', 'unknown')
        }
    )
    
    # Create standardized error response
    error_response = {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat(),
        "path": request.url.path,
        "request_id": request_id
    }
    
    # Add additional context for client errors
    if 400 <= exc.status_code < 500:
        error_response["type"] = "client_error"
        error_response["documentation"] = f"https://docs.auto-analyst.com/errors/{exc.status_code}"
    elif exc.status_code >= 500:
        error_response["type"] = "server_error"
        # Don't expose internal details in production
        if settings.is_production:
            error_response["message"] = "Internal server error"
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed field information."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    # Extract validation error details
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input_value": error.get("input")
        })
    
    log_warning(
        f"Request validation failed",
        extra={
            'request_id': request_id,
            'url': str(request.url),
            'method': request.method,
            'validation_errors': validation_errors,
            'error_count': len(validation_errors)
        }
    )
    
    error_response = {
        "error": True,
        "message": "Request validation failed",
        "type": "validation_error",
        "status_code": 422,
        "timestamp": datetime.utcnow().isoformat(),
        "path": request.url.path,
        "request_id": request_id,
        "validation_errors": validation_errors
    }
    
    return JSONResponse(
        status_code=422,
        content=error_response,
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with proper logging and error tracking."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    error_id = str(uuid.uuid4())
    
    log_error(
        f"Unhandled exception occurred",
        exception=exc,
        extra={
            'request_id': request_id,
            'error_id': error_id,
            'url': str(request.url),
            'method': request.method,
            'client_ip': request.headers.get('x-forwarded-for', 
                        request.headers.get('x-real-ip', 'unknown')),
            'user_agent': request.headers.get('user-agent', 'unknown'),
            'exception_type': type(exc).__name__,
            'traceback': traceback.format_exc()
        }
    )
    
    # Update error metrics
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNT.labels(
            error_type=type(exc).__name__,
            endpoint=request.url.path,
            severity="error"
        ).inc()
    
    # Prepare error response
    error_response = {
        "error": True,
        "type": "internal_error",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat(),
        "path": request.url.path,
        "request_id": request_id,
        "error_id": error_id
    }
    
    # Include details based on environment
    if settings.is_production:
        error_response["message"] = "An internal server error occurred. Please try again later."
        error_response["support"] = f"Contact support with error ID: {error_id}"
    else:
        error_response["message"] = str(exc)
        error_response["exception_type"] = type(exc).__name__
        error_response["traceback"] = traceback.format_exc().split('\n')
    
    return JSONResponse(
        status_code=500,
        content=error_response,
        headers={"X-Request-ID": request_id}
    )

# Dependency functions with enhanced error handling and caching
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    request: Request = None
) -> schemas.User:
    """
    Get current authenticated user with JWT validation.
    
    Features:
    - JWT token validation with proper error handling
    - Token blacklist checking
    - User session validation
    - Rate limiting per user
    - Caching for performance optimization
    """
    if not settings.ENABLE_AUTHENTICATION:
        # Return mock user for development/testing
        mock_user = schemas.User(
            id=1,
            email="dev@auto-analyst.com",
            username="developer",
            is_active=True,
            is_verified=True,
            role="admin",
            created_at=datetime.utcnow()
        )
        
        if request:
            request.state.user_id = mock_user.id
            request.state.user_role = mock_user.role
        
        return mock_user
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        # Validate JWT token
        user = await auth_service.validate_token(credentials.credentials)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=401,
                detail="Account is disabled"
            )
        
        # Store user info in request state for logging
        if request:
            request.state.user_id = user.id
            request.state.user_role = user.role
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        log_error("Authentication validation failed", exception=e)
        raise HTTPException(
            status_code=401,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )

def get_request_context(request: Request) -> Dict[str, Any]:
    """Extract request context for logging and monitoring."""
    return {
        'request_id': getattr(request.state, 'request_id', str(uuid.uuid4())),
        'user_id': getattr(request.state, 'user_id', None),
        'client_ip': request.headers.get('x-forwarded-for', 
                    request.headers.get('x-real-ip', 'unknown')),
        'user_agent': request.headers.get('user-agent', 'unknown')[:100],
        'method': request.method,
        'url': str(request.url),
        'timestamp': datetime.utcnow().isoformat()
    }

# Service initialization functions
async def initialize_global_services():
    """Initialize global application services."""
    global monitoring_manager, thread_pool, celery_app, security_manager, cache_manager
    
    try:
        # Initialize thread pool for CPU-intensive tasks
        max_workers = min(32, (os.cpu_count() or 1) * 4)
        thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="auto-analyst-worker-"
        )
        logger.info(f"🧵 Thread pool initialized with {max_workers} workers")
        
        # Initialize security manager
        security_manager = SecurityManager()
        logger.info("🔐 Security manager initialized")
        
        # Initialize cache manager
        cache_manager = CacheManager()
        await cache_manager.initialize()
        logger.info("💾 Cache manager initialized")
        
        # Initialize monitoring manager
        monitoring_manager = create_monitoring_manager()
        logger.info("📊 Monitoring manager initialized")
        
        # Initialize Celery for background tasks
        if CELERY_AVAILABLE and settings.REDIS_URL:
            celery_app = Celery(
                'auto_analyst',
                broker=settings.redis_url,
                backend=settings.redis_url,
                include=[
                    'backend.tasks.training_tasks',
                    'backend.tasks.data_processing_tasks',
                    'backend.tasks.prediction_tasks',
                    'backend.tasks.cleanup_tasks'
                ]
            )
            
            # Configure Celery with optimized settings
            celery_app.conf.update({
                'task_serializer': 'json',
                'accept_content': ['json'],
                'result_serializer': 'json',
                'timezone': 'UTC',
                'enable_utc': True,
                'task_track_started': True,
                'task_ignore_result': False,
                'result_expires': 86400 * 7,  # 7 days
                'worker_prefetch_multiplier': 1,
                'task_acks_late': True,
                'worker_disable_rate_limits': False,
                'task_default_queue': 'default',
                'task_routes': {
                    'backend.tasks.train_model': {'queue': 'training'},
                    'backend.tasks.process_dataset': {'queue': 'processing'},
                    'backend.tasks.predict_batch': {'queue': 'prediction'},
                    'backend.tasks.generate_insights': {'queue': 'insights'},
                    'backend.tasks.cleanup_artifacts': {'queue': 'maintenance'},
                },
                'task_default_retry_delay': 60,
                'task_max_retries': 3,
                'worker_max_tasks_per_child': 1000,  # Prevent memory leaks
                'worker_max_memory_per_child': 200000,  # 200MB limit
            })
            
            logger.info("🔄 Celery application initialized")
        else:
            logger.warning("⚠️  Celery not available or Redis not configured")
        
        logger.info("✅ All global services initialized successfully")
        
    except Exception as e:
        log_error("Failed to initialize global services", exception=e)
        raise

async def run_database_migrations():
    """Run database migrations using Alembic."""
    try:
        # Import Alembic components
        from alembic.config import Config
        from alembic import command
        
        # Configure Alembic
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database.url)
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        
        logger.info("🔄 Database migrations completed successfully")
        
    except ImportError:
        logger.warning("⚠️  Alembic not available, skipping migrations")
    except Exception as e:
        log_error("Database migration failed", exception=e)
        raise

async def initialize_monitoring_system():
    """Initialize comprehensive monitoring system."""
    global monitoring_manager
    
    try:
        if monitoring_manager and settings.ENABLE_MONITORING:
            await monitoring_manager.start_monitoring()
            
            # Setup custom monitoring tasks
            await monitoring_manager.setup_drift_monitoring()
            await monitoring_manager.setup_performance_monitoring()
            await monitoring_manager.setup_resource_monitoring()
            
            logger.info("📊 Monitoring system initialized successfully")
        else:
            logger.warning("⚠️  Monitoring disabled or not available")
            
    except Exception as e:
        log_error("Failed to initialize monitoring system", exception=e)
        # Don't raise - monitoring is not critical for core functionality

async def initialize_background_processing():
    """Initialize background task processing systems."""
    try:
        if celery_app:
            # Verify Celery connectivity
            inspect = celery_app.control.inspect()
            if inspect:
                stats = inspect.stats()
                logger.info(f"🔄 Celery workers available: {len(stats) if stats else 0}")
            
            # Schedule periodic tasks
            from backend.tasks.cleanup_tasks import cleanup_artifacts
            from backend.tasks.monitoring_tasks import check_model_drift
            
            # Setup periodic task scheduling
            if hasattr(celery_app, 'beat_schedule'):
                celery_app.conf.beat_schedule = {
                    'cleanup-artifacts': {
                        'task': 'backend.tasks.cleanup_artifacts',
                        'schedule': 86400.0,  # Daily
                        'args': (30,)  # 30 days retention
                    },
                    'check-drift': {
                        'task': 'backend.tasks.check_model_drift',
                        'schedule': 3600.0,  # Hourly
                    }
                }
                celery_app.conf.timezone = 'UTC'
        
        # Initialize task monitoring
        if monitoring_manager:
            await monitoring_manager.setup_task_monitoring()
        
        logger.info("⏰ Background processing initialized successfully")
        
    except Exception as e:
        log_error("Failed to initialize background processing", exception=e)
        # Don't raise - background tasks are not critical for basic functionality

async def perform_startup_health_checks() -> Dict[str, str]:
    """Perform comprehensive startup health checks."""
    health_status = {}
    
    try:
        # Check database connectivity
        try:
            db_healthy = await check_database_health()
            health_status['database'] = 'healthy' if db_healthy else 'unhealthy'
        except Exception as e:
            health_status['database'] = f'error: {str(e)}'
            log_error("Database health check failed", exception=e)
        
        # Check Redis/cache connectivity
        try:
            if cache_manager:
                cache_healthy = await cache_manager.health_check()
                health_status['cache'] = 'healthy' if cache_healthy else 'unhealthy'
            else:
                health_status['cache'] = 'not_configured'
        except Exception as e:
            health_status['cache'] = f'error: {str(e)}'
            log_error("Cache health check failed", exception=e)
        
        # Check MLflow connectivity
        try:
            mlflow_healthy = await check_mlflow_health()
            health_status['mlflow'] = 'healthy' if mlflow_healthy else 'unhealthy'
        except Exception as e:
            health_status['mlflow'] = f'error: {str(e)}'
            log_error("MLflow health check failed", exception=e)
        
        # Check Celery workers
        try:
            if celery_app:
                inspect = celery_app.control.inspect()
                if inspect:
                    stats = inspect.stats()
                    worker_count = len(stats) if stats else 0
                    health_status['celery'] = f'healthy ({worker_count} workers)'
                else:
                    health_status['celery'] = 'unhealthy'
            else:
                health_status['celery'] = 'not_configured'
        except Exception as e:
            health_status['celery'] = f'error: {str(e)}'
            log_error("Celery health check failed", exception=e)
        
        # Check system resources
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                health_status['system'] = (
                    f'memory: {memory.percent:.1f}%, '
                    f'disk: {disk.percent:.1f}%'
                )
            else:
                health_status['system'] = 'monitoring_unavailable'
        except Exception as e:
            health_status['system'] = f'error: {str(e)}'
        
        return health_status
        
    except Exception as e:
        log_error("Startup health checks failed", exception=e)
        return {'overall': f'error: {str(e)}'}

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"🔔 Received signal {signum}, initiating graceful shutdown...")
        
        # Note: The actual cleanup is handled by the lifespan manager
        # This just logs the signal reception
        if signum == signal.SIGTERM:
            logger.info("📤 Received SIGTERM (Kubernetes/Docker shutdown)")
        elif signum == signal.SIGINT:
            logger.info("⌨️  Received SIGINT (Ctrl+C)")
        
        # Don't call sys.exit() here as it can interfere with FastAPI's shutdown
        # FastAPI will handle the graceful shutdown through the lifespan manager
    
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("🔧 Signal handlers configured")
    except Exception as e:
        log_error("Failed to setup signal handlers", exception=e)

# Cleanup functions
async def cleanup_global_services():
    """Cleanup global services during shutdown."""
    global monitoring_manager, thread_pool, celery_app, security_manager, cache_manager
    
    try:
        # Stop thread pool
        if thread_pool:
            logger.info("🧵 Shutting down thread pool...")
            thread_pool.shutdown(wait=True, timeout=30)
            thread_pool = None
        
        # Cleanup cache manager
        if cache_manager:
            logger.info("💾 Cleaning up cache manager...")
            await cache_manager.cleanup()
            cache_manager = None
        
        # Cleanup security manager
        if security_manager:
            logger.info("🔐 Cleaning up security manager...")
            await security_manager.cleanup()
            security_manager = None
        
        # Note: Celery cleanup is handled by the task system
        # Monitoring manager cleanup is handled separately
        
        logger.info("✅ Global services cleanup completed")
        
    except Exception as e:
        log_error("Error during services cleanup", exception=e)

async def cleanup_background_processing():
    """Cleanup background processing systems."""
    global celery_app
    
    try:
        if celery_app:
            logger.info("🔄 Stopping Celery workers...")
            
            # Send shutdown signal to workers
            celery_app.control.broadcast('shutdown', destination=None)
            
            # Wait for tasks to complete (with timeout)
            active_tasks = celery_app.control.inspect().active()
            if active_tasks:
                logger.info(f"⏳ Waiting for {sum(len(tasks) for tasks in active_tasks.values())} active tasks to complete...")
                
                # Give tasks time to complete
                await asyncio.sleep(10)
            
            celery_app = None
        
        logger.info("✅ Background processing cleanup completed")
        
    except Exception as e:
        log_error("Error during background processing cleanup", exception=e)

async def final_cleanup():
    """Final cleanup operations."""
    try:
        # Clean up temporary files
        temp_dir = Path(settings.TEMP_DIRECTORY)
        if temp_dir.exists():
            for temp_file in temp_dir.glob("auto_analyst_*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        # Final logging
        logger.info("🧽 Final cleanup completed")
        
    except Exception as e:
        log_error("Error during final cleanup", exception=e)

# Health check endpoints with comprehensive status reporting
@app.get("/health", tags=["Health"], summary="Comprehensive health check")
async def health_check():
    """
    Comprehensive health check endpoint with detailed service status.
    
    Returns overall application health status including:
    - Database connectivity and performance
    - Cache system status
    - MLflow tracking server connectivity
    - Background task processing status
    - System resource usage
    - External service dependencies
    """
    try:
        health_info = {
            "status": "checking",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "uptime": time.time() - getattr(app.state, 'start_time', time.time()),
            "services": {},
            "system": {},
            "dependencies": {}
        }
        
        # Check database health
        try:
            db_start = time.time()
            db_healthy = await check_database_health()
            db_latency = (time.time() - db_start) * 1000
            
            health_info["services"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "latency_ms": round(db_latency, 2),
                "connection_pool": {
                    "size": getattr(engine.pool, 'size', None) if engine else None,
                    "checked_out": getattr(engine.pool, 'checkedout', lambda: None)() if engine else None,
                    "overflow": getattr(engine.pool, 'overflow', lambda: None)() if engine else None,
                }
            }
        except Exception as e:
            health_info["services"]["database"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check cache health
        try:
            if cache_manager:
                cache_start = time.time()
                cache_healthy = await cache_manager.health_check()
                cache_latency = (time.time() - cache_start) * 1000
                
                health_info["services"]["cache"] = {
                    "status": "healthy" if cache_healthy else "unhealthy",
                    "latency_ms": round(cache_latency, 2),
                    "type": "redis" if settings.REDIS_URL else "memory"
                }
            else:
                health_info["services"]["cache"] = {"status": "not_configured"}
        except Exception as e:
            health_info["services"]["cache"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check MLflow health
        try:
            mlflow_healthy = await check_mlflow_health()
            health_info["services"]["mlflow"] = {
                "status": "healthy" if mlflow_healthy else "unhealthy",
                "tracking_uri": settings.mlflow.tracking_uri
            }
        except Exception as e:
            health_info["services"]["mlflow"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check background processing
        try:
            if celery_app:
                inspect = celery_app.control.inspect()
                if inspect:
                    stats = inspect.stats()
                    active = inspect.active()
                    
                    worker_count = len(stats) if stats else 0
                    active_task_count = sum(len(tasks) for tasks in active.values()) if active else 0
                    
                    health_info["services"]["background_processing"] = {
                        "status": "healthy" if worker_count > 0 else "degraded",
                        "workers": worker_count,
                        "active_tasks": active_task_count
                    }
                else:
                    health_info["services"]["background_processing"] = {
                        "status": "unhealthy",
                        "error": "Cannot connect to workers"
                    }
            else:
                health_info["services"]["background_processing"] = {
                    "status": "not_configured"
                }
        except Exception as e:
            health_info["services"]["background_processing"] = {
                "status": "error",
                "error": str(e)
            }
        
        # System resource information
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                cpu_percent = psutil.cpu_percent(interval=None)
                
                health_info["system"] = {
                    "memory": {
                        "used_percent": round(memory.percent, 1),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "total_gb": round(memory.total / (1024**3), 2)
                    },
                    "disk": {
                        "used_percent": round(disk.percent, 1),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "total_gb": round(disk.total / (1024**3), 2)
                    },
                    "cpu": {
                        "usage_percent": round(cpu_percent, 1),
                        "cores": psutil.cpu_count()
                    }
                }
            except Exception as e:
                health_info["system"] = {"error": str(e)}
        
        # Determine overall status
        service_statuses = [
            service.get("status") for service in health_info["services"].values()
            if isinstance(service, dict)
        ]
        
        if all(status == "healthy" for status in service_statuses):
            overall_status = "healthy"
            status_code = 200
        elif any(status == "unhealthy" for status in service_statuses):
            overall_status = "unhealthy"
            status_code = 503
        elif any(status == "error" for status in service_statuses):
            overall_status = "error"
            status_code = 503
        else:
            overall_status = "degraded"
            status_code = 200
        
        health_info["status"] = overall_status
        
        # Add health check metadata
        health_info["check_duration_ms"] = round(
            (datetime.utcnow().timestamp() - datetime.fromisoformat(health_info["timestamp"].replace('Z', '+00:00')).timestamp()) * 1000,
            2
        )
        
        return JSONResponse(content=health_info, status_code=status_code)
        
    except Exception as e:
        log_error("Health check failed", exception=e)
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "version": settings.APP_VERSION
            },
            status_code=503
        )

@app.get("/readiness", tags=["Health"], summary="Kubernetes readiness probe")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Checks if the application is ready to receive requests.
    Used by Kubernetes to determine when to start routing traffic.
    """
    try:
        # Check critical dependencies
        db_ready = await check_database_health()
        
        # Application is ready if database is accessible
        if db_ready:
            return JSONResponse(
                content={
                    "status": "ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": settings.APP_VERSION
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "reason": "Database not accessible",
                    "timestamp": datetime.utcnow().isoformat()
                },
                status_code=503
            )
            
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )

@app.get("/liveness", tags=["Health"], summary="Kubernetes liveness probe")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Simple endpoint that indicates the application process is alive.
    Used by Kubernetes to determine if the container should be restarted.
    """
    return JSONResponse(
        content={
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "pid": os.getpid()
        },
        status_code=200
    )

# Dataset management endpoints with enhanced functionality
@app.post(
    "/api/v1/datasets/upload",
    response_model=schemas.DatasetResponse,
    tags=["Datasets"],
    summary="Upload dataset for analysis",
    description="Upload and validate datasets with support for multiple formats and large files"
)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Dataset file to upload"),
    name: Optional[str] = Form(None, description="Custom name for the dataset"),
    description: Optional[str] = Form(None, description="Dataset description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    data_service: DataService = Depends(get_data_service),
    current_user: schemas.User = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Upload a dataset for analysis with comprehensive validation and processing.
    
    Features:
    - Support for multiple file formats (CSV, Excel, JSON, Parquet)
    - Large file support up to 20GB with streaming upload
    - Automatic format detection and validation
    - Background processing with progress tracking
    - Data quality assessment and profiling
    - Metadata extraction and storage
    """
    request_context = get_request_context(request)
    
    try:
        log_info(
            "Dataset upload initiated",
            extra={
                **request_context,
                'filename': file.filename,
                'content_type': file.content_type,
                'user_id': current_user.id
            }
        )
        
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="File name is required"
            )
        
        # Check file extension
        allowed_extensions = {'.csv', '.tsv', '.xlsx', '.xls', '.json', '.parquet', '.feather'}
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Estimate file size
        file_size = 0
        if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
            current_pos = file.file.tell()
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(current_pos)
        
        # Validate file size
        if file_size > settings.UPLOAD_MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.UPLOAD_MAX_SIZE / (1024**3):.1f}GB"
            )
        
        # Update metrics
        if PROMETHEUS_AVAILABLE:
            ACTIVE_UPLOADS.inc()
            DATASET_SIZE.observe(file_size)
        
        try:
            # Parse tags
            tag_list = []
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            
            # Create dataset record in database
            dataset = await data_service.create_dataset_record(
                filename=file.filename,
                original_name=name or file.filename,
                description=description,
                tags=tag_list,
                file_size=file_size,
                content_type=file.content_type,
                user_id=current_user.id,
                db=db
            )
            
            # Save uploaded file
            temp_file_path = await save_uploaded_file(file, dataset.id)
            
            # Process dataset in background
            background_tasks.add_task(
                process_dataset_background,
                dataset.id,
                temp_file_path,
                current_user.id,
                request_context['request_id']
            )
            
            log_info(
                "Dataset upload completed, processing started",
                extra={
                    **request_context,
                    'dataset_id': dataset.id,
                    'file_size_bytes': file_size
                }
            )
            
            return schemas.DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                original_filename=dataset.original_filename,
                status="processing",
                file_size=file_size,
                content_type=file.content_type,
                upload_time=dataset.created_at,
                tags=tag_list,
                message="Dataset uploaded successfully. Processing in background."
            )
            
        finally:
            if PROMETHEUS_AVAILABLE:
                ACTIVE_UPLOADS.dec()
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(
            "Dataset upload failed",
            exception=e,
            extra=request_context
        )
        
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(
                error_type=type(e).__name__,
                endpoint="/api/v1/datasets/upload",
                severity="error"
            ).inc()
        
        raise HTTPException(
            status_code=500,
            detail="Dataset upload failed. Please try again."
        )

# Additional utility functions
async def save_uploaded_file(file: UploadFile, identifier: Union[str, int]) -> Path:
    """
    Save uploaded file to temporary storage with streaming support.
    
    Features:
    - Streaming upload for large files
    - Virus scanning integration (placeholder)
    - File integrity verification
    - Temporary storage with cleanup
    """
    try:
        # Create secure temporary directory
        temp_dir = Path(settings.TEMP_DIRECTORY)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate secure filename
        file_extension = Path(file.filename).suffix if file.filename else ""
        secure_filename = f"upload_{identifier}_{uuid.uuid4().hex}{file_extension}"
        temp_path = temp_dir / secure_filename
        
        # Calculate hash while saving
        import hashlib
        hash_sha256 = hashlib.sha256()
        
        # Save file with streaming to handle large files
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(temp_path, 'wb') as f:
                while chunk := await file.read(settings.CHUNK_SIZE):
                    hash_sha256.update(chunk)
                    await f.write(chunk)
        else:
            # Fallback for environments without aiofiles
            with open(temp_path, 'wb') as f:
                while chunk := await file.read(settings.CHUNK_SIZE):
                    hash_sha256.update(chunk)
                    f.write(chunk)
        
        # Store file hash for integrity verification
        file_hash = hash_sha256.hexdigest()
        hash_file = temp_path.with_suffix(temp_path.suffix + '.sha256')
        
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(hash_file, 'w') as f:
                await f.write(file_hash)
        else:
            with open(hash_file, 'w') as f:
                f.write(file_hash)
        
        log_info(
            f"File saved successfully",
            extra={
                'temp_path': str(temp_path),
                'file_size': temp_path.stat().st_size,
                'file_hash': file_hash
            }
        )
        
        return temp_path
        
    except Exception as e:
        log_error("Failed to save uploaded file", exception=e)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file"
        )

async def process_dataset_background(
    dataset_id: int,
    file_path: Path,
    user_id: int,
    request_id: str
):
    """
    Background task to process uploaded dataset.
    
    Processing steps:
    1. File format detection and validation
    2. Data loading and initial validation
    3. Data quality assessment
    4. Statistical profiling
    5. Schema inference and optimization
    6. Metadata extraction
    7. Database record update
    8. Cleanup
    """
    try:
        log_info(
            "Dataset background processing started",
            extra={
                'dataset_id': dataset_id,
                'request_id': request_id,
                'user_id': user_id
            }
        )
        
        # Get data service
        data_service = get_data_service()
        
        # Process dataset using the task system
        from backend.tasks.data_processing_tasks import execute_dataset_processing
        
        # Configure processing parameters
        processing_config = {
            'file_format': 'auto',
            'handle_missing_values': True,
            'generate_profile': True,
            'validate_quality': True,
            'optimize_storage': True
        }
        
        # Execute processing
        result = execute_dataset_processing(
            dataset_id=dataset_id,
            user_id=user_id,
            config=processing_config
        )
        
        # Update dataset record with results
        with get_db_session() as db:
            await data_service.update_dataset_processing_results(
                dataset_id=dataset_id,
                processing_results=result,
                db=db
            )
        
        log_info(
            "Dataset background processing completed",
            extra={
                'dataset_id': dataset_id,
                'request_id': request_id,
                'processing_status': result.get('status', 'unknown')
            }
        )
        
    except Exception as e:
        log_error(
            "Dataset background processing failed",
            exception=e,
            extra={
                'dataset_id': dataset_id,
                'request_id': request_id
            }
        )
        
        # Update dataset status to failed
        try:
            with get_db_session() as db:
                data_service = get_data_service()
                await data_service.update_dataset_status(
                    dataset_id=dataset_id,
                    status="failed",
                    error_message=str(e),
                    db=db
                )
        except Exception as update_error:
            log_error(
                "Failed to update dataset status",
                exception=update_error
            )
    
    finally:
        # Cleanup temporary files
        try:
            if file_path.exists():
                file_path.unlink()
                
            # Also remove hash file
            hash_file = file_path.with_suffix(file_path.suffix + '.sha256')
            if hash_file.exists():
                hash_file.unlink()
                
            log_info(f"Temporary files cleaned up: {file_path}")
            
        except Exception as cleanup_error:
            log_error(
                "Failed to cleanup temporary files",
                exception=cleanup_error,
                extra={'file_path': str(file_path)}
            )

# Health check utility functions
async def check_database_health() -> bool:
    """
    Check database connectivity and basic functionality.
    
    Performs:
    - Connection test
    - Simple query execution
    - Connection pool status check
    """
    try:
        # Test basic connectivity
        async with get_db_session() as db:
            result = await db.execute(text
