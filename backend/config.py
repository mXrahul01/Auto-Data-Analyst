"""
Comprehensive Configuration Management for Auto-Analyst Platform

This module provides centralized, type-safe configuration management for the
Auto-Analyst platform with enterprise-grade features including multi-environment
support, cloud integrations, security, and performance optimization.

Key Features:
- Type-safe configuration with Pydantic validation
- Multi-environment support (development, staging, production, testing)
- Database configuration with connection pooling and optimization
- MLflow experiment tracking and model registry integration
- Feast feature store configuration
- Multi-cloud storage support (AWS S3, GCP, Azure)
- Remote compute integration (Kaggle, Colab, cloud platforms)
- Comprehensive monitoring and observability settings
- Security and authentication configuration with best practices
- Performance optimization and resource management
- Structured logging and metrics collection

Environment Variable Priority:
1. Environment variables (highest priority)
2. .env.{environment} files
3. .env file
4. Default values (lowest priority)

Required Environment Variables:
- ENVIRONMENT: deployment environment (development, staging, production)
- DATABASE_URL: database connection string
- SECRET_KEY: application secret key for JWT tokens

Usage:
    from backend.config import settings
    
    # Access configuration
    database_url = settings.DATABASE_URL
    mlflow_uri = settings.mlflow.tracking_uri
    
    # Environment checks
    if settings.is_production:
        # Production-specific logic
        pass

Author: Auto-Analyst Team
Version: 2.0.0
Last Updated: 2025-09-21
"""

import os
import secrets
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from functools import lru_cache

# Optional dependencies with graceful fallbacks
try:
    from pydantic import BaseSettings, Field, validator, root_validator
    from pydantic.types import SecretStr, PositiveInt, constr
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Graceful fallback for environments without pydantic
    BaseSettings = object
    Field = lambda default=None, **kwargs: default
    SecretStr = str
    PositiveInt = int
    constr = str
    PYDANTIC_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables with proper precedence
def _load_environment_variables() -> None:
    """Load environment variables with proper precedence handling."""
    if not DOTENV_AVAILABLE:
        logger.warning("python-dotenv not available, using only system environment variables")
        return
    
    try:
        # Load base .env file first
        base_env_path = Path(".env")
        if base_env_path.exists():
            load_dotenv(base_env_path)
            logger.debug(f"Loaded base environment from: {base_env_path}")
        
        # Load environment-specific .env file with override
        environment = os.getenv("ENVIRONMENT", "development")
        env_file_path = Path(f".env.{environment}")
        if env_file_path.exists():
            load_dotenv(env_file_path, override=True)
            logger.debug(f"Loaded environment-specific config from: {env_file_path}")
        
        # Load local .env.local for development overrides
        local_env_path = Path(".env.local")
        if local_env_path.exists():
            load_dotenv(local_env_path, override=True)
            logger.debug(f"Loaded local overrides from: {local_env_path}")
            
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        raise

# Load environment variables on module import
_load_environment_variables()

# Enumerations for type safety
class Environment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Logging severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class CloudProvider(str, Enum):
    """Supported cloud providers for storage and compute."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"

class ComputeBackend(str, Enum):
    """Supported compute backends for ML training."""
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"
    AWS_SAGEMAKER = "aws_sagemaker"
    GCP_VERTEX = "gcp_vertex_ai"
    AZURE_ML = "azure_ml"

class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"

class CacheBackend(str, Enum):
    """Supported cache backends."""
    REDIS = "redis"
    MEMCACHED = "memcached"
    MEMORY = "memory"

# Configuration helper classes
class DatabaseConfig:
    """Database configuration with connection pooling and optimization."""
    
    def __init__(self, settings_instance: 'Settings'):
        self.settings = settings_instance
        self._cached_url: Optional[str] = None
        self._cached_engine_config: Optional[Dict[str, Any]] = None
    
    @property
    def url(self) -> str:
        """Get properly formatted database URL with connection pooling."""
        if self._cached_url:
            return self._cached_url
        
        if self.settings.DATABASE_URL:
            self._cached_url = self.settings.DATABASE_URL
            return self._cached_url
        
        # Construct URL from individual components
        password = self.settings.DB_PASSWORD.get_secret_value() if hasattr(self.settings.DB_PASSWORD, 'get_secret_value') else str(self.settings.DB_PASSWORD)
        
        if self.settings.DB_TYPE == DatabaseType.POSTGRESQL:
            self._cached_url = (
                f"postgresql+psycopg2://{self.settings.DB_USER}:{password}"
                f"@{self.settings.DB_HOST}:{self.settings.DB_PORT}/{self.settings.DB_NAME}"
            )
        elif self.settings.DB_TYPE == DatabaseType.MYSQL:
            self._cached_url = (
                f"mysql+pymysql://{self.settings.DB_USER}:{password}"
                f"@{self.settings.DB_HOST}:{self.settings.DB_PORT}/{self.settings.DB_NAME}?charset=utf8mb4"
            )
        elif self.settings.DB_TYPE == DatabaseType.SQLITE:
            db_path = Path(self.settings.DB_NAME)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path
            self._cached_url = f"sqlite:///{db_path}"
        else:
            raise ValueError(f"Unsupported database type: {self.settings.DB_TYPE}")
        
        return self._cached_url
    
    @property
    def engine_config(self) -> Dict[str, Any]:
        """Get optimized SQLAlchemy engine configuration."""
        if self._cached_engine_config:
            return self._cached_engine_config
        
        base_config = {
            "echo": self.settings.DB_ECHO,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "future": True,  # Use SQLAlchemy 2.0 style
        }
        
        # Environment-specific optimizations
        if self.settings.ENVIRONMENT == Environment.PRODUCTION:
            base_config.update({
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30,
                "pool_recycle": 1800,  # 30 minutes
                "echo": False,  # Disable SQL logging in production
            })
        elif self.settings.ENVIRONMENT == Environment.STAGING:
            base_config.update({
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 20,
                "pool_recycle": 3600,
            })
        else:  # Development/Testing
            base_config.update({
                "pool_size": 5,
                "max_overflow": 10,
                "pool_timeout": 10,
                "pool_recycle": 7200,
            })
        
        # Database-specific optimizations
        if self.settings.DB_TYPE == DatabaseType.POSTGRESQL:
            base_config.update({
                "connect_args": {
                    "options": "-c timezone=utc",
                    "connect_timeout": 10,
                }
            })
        elif self.settings.DB_TYPE == DatabaseType.SQLITE:
            base_config.update({
                "connect_args": {
                    "check_same_thread": False,
                    "timeout": 20,
                }
            })
        
        self._cached_engine_config = base_config
        return self._cached_engine_config

class MLflowConfig:
    """MLflow experiment tracking and model registry configuration."""
    
    def __init__(self, settings_instance: 'Settings'):
        self.settings = settings_instance
        self._cached_tracking_uri: Optional[str] = None
    
    @property
    def tracking_uri(self) -> str:
        """Get MLflow tracking server URI with fallbacks."""
        if self._cached_tracking_uri:
            return self._cached_tracking_uri
        
        if self.settings.MLFLOW_TRACKING_URI:
            self._cached_tracking_uri = self.settings.MLFLOW_TRACKING_URI
            return self._cached_tracking_uri
        
        # Environment-specific defaults
        if self.settings.ENVIRONMENT == Environment.PRODUCTION:
            # Use cloud-based MLflow in production
            if self.settings.CLOUD_PROVIDER == CloudProvider.AWS:
                self._cached_tracking_uri = "https://mlflow-prod.example.com"  # Replace with actual URI
            else:
                self._cached_tracking_uri = "http://mlflow-server:5000"
        else:
            self._cached_tracking_uri = "file://./mlruns"
        
        return self._cached_tracking_uri
    
    @property
    def experiment_name(self) -> str:
        """Get default experiment name with environment prefix."""
        base_name = self.settings.MLFLOW_EXPERIMENT_NAME or "auto-analyst"
        return f"{base_name}-{self.settings.ENVIRONMENT.value}"
    
    @property
    def artifact_location(self) -> str:
        """Get artifact storage location based on cloud provider."""
        if self.settings.MLFLOW_ARTIFACT_ROOT:
            return self.settings.MLFLOW_ARTIFACT_ROOT
        
        if self.settings.CLOUD_PROVIDER == CloudProvider.AWS and self.settings.S3_BUCKET:
            return f"s3://{self.settings.S3_BUCKET}/mlflow-artifacts/{self.settings.ENVIRONMENT.value}"
        elif self.settings.CLOUD_PROVIDER == CloudProvider.GCP and self.settings.GCS_BUCKET:
            return f"gs://{self.settings.GCS_BUCKET}/mlflow-artifacts/{self.settings.ENVIRONMENT.value}"
        elif self.settings.CLOUD_PROVIDER == CloudProvider.AZURE and self.settings.AZURE_CONTAINER:
            return f"abfss://{self.settings.AZURE_CONTAINER}@{self.settings.AZURE_ACCOUNT}.dfs.core.windows.net/mlflow-artifacts/{self.settings.ENVIRONMENT.value}"
        else:
            artifacts_dir = Path("./artifacts/mlflow") / self.settings.ENVIRONMENT.value
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            return str(artifacts_dir.absolute())

class FeastConfig:
    """Feast feature store configuration with cloud backend support."""
    
    def __init__(self, settings_instance: 'Settings'):
        self.settings = settings_instance
    
    @property
    def repo_path(self) -> str:
        """Get Feast repository path."""
        if self.settings.FEAST_REPO_PATH:
            return self.settings.FEAST_REPO_PATH
        
        repo_path = Path("./feast_repo") / self.settings.ENVIRONMENT.value
        repo_path.mkdir(parents=True, exist_ok=True)
        return str(repo_path.absolute())
    
    @property
    def online_store_config(self) -> Dict[str, Any]:
        """Get optimized online store configuration."""
        # Production uses cloud-native stores
        if self.settings.ENVIRONMENT == Environment.PRODUCTION:
            if self.settings.CLOUD_PROVIDER == CloudProvider.AWS:
                return {
                    "type": "dynamodb",
                    "region": self.settings.AWS_REGION,
                    "table_name": f"feast-online-{self.settings.ENVIRONMENT.value}",
                    "ttl_duration_seconds": 86400,  # 24 hours
                }
            elif self.settings.REDIS_URL:
                return {
                    "type": "redis",
                    "connection_string": self.settings.redis_url,
                    "key_ttl_seconds": 86400,
                }
        
        # Development/Staging uses SQLite
        online_store_path = Path("./feast_online_store") / f"{self.settings.ENVIRONMENT.value}.db"
        online_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        return {
            "type": "sqlite",
            "path": str(online_store_path.absolute())
        }
    
    @property
    def offline_store_config(self) -> Dict[str, Any]:
        """Get offline store configuration for batch features."""
        if self.settings.CLOUD_PROVIDER == CloudProvider.AWS:
            return {
                "type": "redshift",
                "cluster_id": self.settings.REDSHIFT_CLUSTER_ID,
                "database": self.settings.REDSHIFT_DATABASE,
                "user": self.settings.REDSHIFT_USER,
                "password": self.settings.REDSHIFT_PASSWORD,
                "s3_staging_location": f"s3://{self.settings.S3_BUCKET}/feast-staging/{self.settings.ENVIRONMENT.value}",
                "workgroup": f"feast-{self.settings.ENVIRONMENT.value}",
            }
        elif self.settings.CLOUD_PROVIDER == CloudProvider.GCP:
            return {
                "type": "bigquery",
                "project_id": self.settings.GCP_PROJECT_ID,
                "dataset_id": f"feast_{self.settings.ENVIRONMENT.value}",
                "location": "US",  # Default location
            }
        else:
            # Local file-based store for development
            offline_store_path = Path("./feast_offline_store") / self.settings.ENVIRONMENT.value
            offline_store_path.mkdir(parents=True, exist_ok=True)
            return {
                "type": "file",
                "path": str(offline_store_path.absolute())
            }

# Main settings class with comprehensive validation
if PYDANTIC_AVAILABLE:
    class Settings(BaseSettings):
        """
        Comprehensive application settings with Pydantic validation.
        
        This class provides type-safe configuration management with automatic
        validation, environment-specific defaults, and cloud integration support.
        """
        
        # ===================
        # Core Application Settings
        # ===================
        ENVIRONMENT: Environment = Field(
            Environment.DEVELOPMENT,
            description="Deployment environment"
        )
        DEBUG: bool = Field(
            False,
            description="Enable debug mode and verbose logging"
        )
        TESTING: bool = Field(
            False,
            description="Enable testing mode with mocked services"
        )
        
        APP_NAME: str = Field(
            "Auto-Analyst",
            description="Application name for branding and logging"
        )
        APP_VERSION: str = Field(
            "2.0.0",
            description="Application version"
        )
        API_V1_STR: str = Field(
            "/api/v1",
            description="API version prefix for routing"
        )
        
        # ===================
        # Security Settings
        # ===================
        SECRET_KEY: SecretStr = Field(
            ...,
            description="Secret key for JWT tokens and session encryption"
        )
        ACCESS_TOKEN_EXPIRE_MINUTES: PositiveInt = Field(
            60 * 24 * 7,  # 7 days
            description="JWT access token expiration time in minutes"
        )
        REFRESH_TOKEN_EXPIRE_MINUTES: PositiveInt = Field(
            60 * 24 * 30,  # 30 days
            description="JWT refresh token expiration time in minutes"
        )
        
        # Password policy
        PASSWORD_MIN_LENGTH: PositiveInt = Field(
            8,
            description="Minimum password length requirement"
        )
        PASSWORD_REQUIRE_SPECIAL: bool = Field(
            True,
            description="Require special characters in passwords"
        )
        PASSWORD_REQUIRE_NUMBERS: bool = Field(
            True,
            description="Require numbers in passwords"
        )
        
        # ===================
        # Server Configuration
        # ===================
        HOST: str = Field(
            "0.0.0.0",
            description="Server bind host address"
        )
        PORT: PositiveInt = Field(
            8000,
            description="Server bind port"
        )
        WORKERS: PositiveInt = Field(
            1,
            description="Number of worker processes (production)"
        )
        RELOAD: bool = Field(
            False,
            description="Enable auto-reload for development"
        )
        
        # ===================
        # Database Configuration
        # ===================
        DATABASE_URL: Optional[str] = Field(
            None,
            description="Complete database connection URL"
        )
        DB_TYPE: DatabaseType = Field(
            DatabaseType.POSTGRESQL,
            description="Database type/driver"
        )
        DB_HOST: str = Field(
            "localhost",
            description="Database server host"
        )
        DB_PORT: PositiveInt = Field(
            5432,
            description="Database server port"
        )
        DB_USER: str = Field(
            "postgres",
            description="Database username"
        )
        DB_PASSWORD: SecretStr = Field(
            "",
            description="Database password"
        )
        DB_NAME: str = Field(
            "auto_analyst",
            description="Database name"
        )
        DB_ECHO: bool = Field(
            False,
            description="Enable SQLAlchemy query logging"
        )
        
        # Database pool settings
        DB_POOL_SIZE: PositiveInt = Field(
            10,
            description="Database connection pool size"
        )
        DB_MAX_OVERFLOW: PositiveInt = Field(
            20,
            description="Database connection pool max overflow"
        )
        
        # ===================
        # Cache Configuration
        # ===================
        CACHE_BACKEND: CacheBackend = Field(
            CacheBackend.REDIS,
            description="Cache backend type"
        )
        REDIS_URL: Optional[str] = Field(
            None,
            description="Redis connection URL"
        )
        REDIS_HOST: str = Field(
            "localhost",
            description="Redis server host"
        )
        REDIS_PORT: PositiveInt = Field(
            6379,
            description="Redis server port"
        )
        REDIS_DB: int = Field(
            0,
            description="Redis database number"
        )
        REDIS_PASSWORD: Optional[SecretStr] = Field(
            None,
            description="Redis authentication password"
        )
        REDIS_SSL: bool = Field(
            False,
            description="Enable Redis SSL/TLS connection"
        )
        
        # ===================
        # Cloud Provider Settings
        # ===================
        CLOUD_PROVIDER: CloudProvider = Field(
            CloudProvider.LOCAL,
            description="Primary cloud provider for storage and compute"
        )
        
        # AWS Configuration
        AWS_ACCESS_KEY_ID: Optional[SecretStr] = Field(
            None,
            description="AWS access key ID"
        )
        AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = Field(
            None,
            description="AWS secret access key"
        )
        AWS_REGION: str = Field(
            "us-east-1",
            description="AWS region for services"
        )
        S3_BUCKET: Optional[str] = Field(
            None,
            description="S3 bucket for data and artifact storage"
        )
        
        # AWS Data Services
        REDSHIFT_CLUSTER_ID: Optional[str] = Field(
            None,
            description="AWS Redshift cluster identifier"
        )
        REDSHIFT_DATABASE: Optional[str] = Field(
            None,
            description="Redshift database name"
        )
        REDSHIFT_USER: Optional[str] = Field(
            None,
            description="Redshift username"
        )
        REDSHIFT_PASSWORD: Optional[SecretStr] = Field(
            None,
            description="Redshift password"
        )
        
        # GCP Configuration
        GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(
            None,
            description="Path to GCP service account key file"
        )
        GCP_PROJECT_ID: Optional[str] = Field(
            None,
            description="GCP project identifier"
        )
        GCS_BUCKET: Optional[str] = Field(
            None,
            description="Google Cloud Storage bucket name"
        )
        
        # Azure Configuration
        AZURE_ACCOUNT: Optional[str] = Field(
            None,
            description="Azure storage account name"
        )
        AZURE_CONTAINER: Optional[str] = Field(
            None,
            description="Azure storage container name"
        )
        AZURE_ACCOUNT_KEY: Optional[SecretStr] = Field(
            None,
            description="Azure storage account key"
        )
        
        # ===================
        # MLflow Configuration
        # ===================
        MLFLOW_TRACKING_URI: Optional[str] = Field(
            None,
            description="MLflow tracking server URI"
        )
        MLFLOW_REGISTRY_URI: Optional[str] = Field(
            None,
            description="MLflow model registry URI"
        )
        MLFLOW_EXPERIMENT_NAME: Optional[str] = Field(
            None,
            description="Default MLflow experiment name"
        )
        MLFLOW_ARTIFACT_ROOT: Optional[str] = Field(
            None,
            description="MLflow artifact storage root directory"
        )
        MLFLOW_S3_ENDPOINT_URL: Optional[str] = Field(
            None,
            description="Custom S3 endpoint for MLflow artifacts"
        )
        
        # ===================
        # Feast Feature Store
        # ===================
        FEAST_REPO_PATH: Optional[str] = Field(
            None,
            description="Feast feature store repository path"
        )
        FEAST_ONLINE_STORE: str = Field(
            "redis",
            description="Feast online store backend type"
        )
        FEAST_OFFLINE_STORE: str = Field(
            "file",
            description="Feast offline store backend type"
        )
        
        # ===================
        # ML Pipeline Settings
        # ===================
        DEFAULT_COMPUTE_BACKEND: ComputeBackend = Field(
            ComputeBackend.LOCAL,
            description="Default compute backend for ML training"
        )
        ENABLE_REMOTE_TRAINING: bool = Field(
            True,
            description="Enable remote training on cloud platforms"
        )
        
        # Dataset limits
        MAX_DATASET_SIZE_GB: float = Field(
            20.0,
            description="Maximum dataset size in GB"
        )
        MAX_TRAINING_TIME_HOURS: PositiveInt = Field(
            24,
            description="Maximum training time in hours"
        )
        MAX_MODELS_PER_EXPERIMENT: PositiveInt = Field(
            50,
            description="Maximum models per experiment"
        )
        
        # ML defaults
        DEFAULT_TEST_SIZE: float = Field(
            0.2,
            ge=0.1,
            le=0.5,
            description="Default test split ratio"
        )
        DEFAULT_CV_FOLDS: PositiveInt = Field(
            5,
            description="Default cross-validation folds"
        )
        ENABLE_HYPERPARAMETER_TUNING: bool = Field(
            True,
            description="Enable automatic hyperparameter tuning"
        )
        ENABLE_ENSEMBLE_MODELS: bool = Field(
            True,
            description="Enable ensemble model creation"
        )
        ENABLE_DEEP_LEARNING: bool = Field(
            False,
            description="Enable deep learning models (requires GPU)"
        )
        
        # Remote training platforms
        KAGGLE_USERNAME: Optional[str] = Field(
            None,
            description="Kaggle account username"
        )
        KAGGLE_KEY: Optional[SecretStr] = Field(
            None,
            description="Kaggle API key"
        )
        COLAB_NOTEBOOK_TEMPLATE: Optional[str] = Field(
            None,
            description="Google Colab notebook template path"
        )
        
        # ===================
        # Monitoring & Observability
        # ===================
        ENABLE_MONITORING: bool = Field(
            True,
            description="Enable comprehensive monitoring"
        )
        ENABLE_DRIFT_DETECTION: bool = Field(
            True,
            description="Enable model and data drift detection"
        )
        DRIFT_DETECTION_THRESHOLD: float = Field(
            0.1,
            ge=0.01,
            le=1.0,
            description="Statistical threshold for drift detection"
        )
        MONITORING_RETENTION_DAYS: PositiveInt = Field(
            90,
            description="Monitoring data retention period in days"
        )
        
        # Prometheus metrics
        PROMETHEUS_ENABLED: bool = Field(
            True,
            description="Enable Prometheus metrics collection"
        )
        PROMETHEUS_PORT: PositiveInt = Field(
            8001,
            description="Prometheus metrics server port"
        )
        PROMETHEUS_METRICS_PATH: str = Field(
            "/metrics",
            description="Prometheus metrics endpoint path"
        )
        
        # ===================
        # Logging Configuration
        # ===================
        LOG_LEVEL: LogLevel = Field(
            LogLevel.INFO,
            description="Application logging level"
        )
        LOG_FORMAT: str = Field(
            "json",
            regex="^(json|text)$",
            description="Log output format"
        )
        LOG_FILE: Optional[str] = Field(
            None,
            description="Log file path for persistent logging"
        )
        LOG_ROTATION: bool = Field(
            True,
            description="Enable log file rotation"
        )
        LOG_MAX_SIZE: str = Field(
            "100MB",
            description="Maximum size per log file"
        )
        LOG_BACKUP_COUNT: PositiveInt = Field(
            5,
            description="Number of rotated log files to keep"
        )
        
        # Centralized logging
        ELASTICSEARCH_URL: Optional[str] = Field(
            None,
            description="Elasticsearch URL for centralized logging"
        )
        LOGSTASH_HOST: Optional[str] = Field(
            None,
            description="Logstash server host"
        )
        LOGSTASH_PORT: Optional[PositiveInt] = Field(
            5044,
            description="Logstash server port"
        )
        
        # ===================
        # Web Security Settings
        # ===================
        CORS_ORIGINS: List[str] = Field(
            ["*"],
            description="CORS allowed origins"
        )
        CORS_METHODS: List[str] = Field(
            ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            description="CORS allowed HTTP methods"
        )
        CORS_HEADERS: List[str] = Field(
            ["*"],
            description="CORS allowed headers"
        )
        ENABLE_HTTPS: bool = Field(
            False,
            description="Enable HTTPS/TLS encryption"
        )
        SSL_CERT_PATH: Optional[str] = Field(
            None,
            description="SSL/TLS certificate file path"
        )
        SSL_KEY_PATH: Optional[str] = Field(
            None,
            description="SSL/TLS private key file path"
        )
        
        # ===================
        # Performance Settings
        # ===================
        MAX_CONCURRENT_REQUESTS: PositiveInt = Field(
            100,
            description="Maximum concurrent HTTP requests"
        )
        REQUEST_TIMEOUT: PositiveInt = Field(
            300,
            description="HTTP request timeout in seconds"
        )
        UPLOAD_MAX_SIZE: PositiveInt = Field(
            21474836480,  # 20GB in bytes
            description="Maximum file upload size in bytes"
        )
        CHUNK_SIZE: PositiveInt = Field(
            8388608,  # 8MB in bytes
            description="File chunk size for streaming uploads"
        )
        
        # Caching settings
        ENABLE_CACHING: bool = Field(
            True,
            description="Enable HTTP response caching"
        )
        CACHE_TTL: PositiveInt = Field(
            3600,
            description="Cache time-to-live in seconds"
        )
        CACHE_MAX_SIZE: PositiveInt = Field(
            1000,
            description="Maximum number of cache entries"
        )
        
        # ===================
        # Storage Directories
        # ===================
        BASE_DIR: str = Field(
            str(Path(__file__).parent.parent),
            description="Application base directory"
        )
        UPLOAD_DIRECTORY: str = Field(
            "./uploads",
            description="File upload storage directory"
        )
        TEMP_DIRECTORY: str = Field(
            "./temp",
            description="Temporary files directory"
        )
        MODELS_DIRECTORY: str = Field(
            "./models",
            description="Trained models storage directory"
        )
        ARTIFACTS_DIRECTORY: str = Field(
            "./artifacts",
            description="ML artifacts storage directory"
        )
        DATASETS_DIRECTORY: str = Field(
            "./datasets",
            description="Processed datasets storage directory"
        )
        
        # ===================
        # External Services
        # ===================
        # SMTP Email configuration
        SMTP_HOST: Optional[str] = Field(
            None,
            description="SMTP server hostname"
        )
        SMTP_PORT: PositiveInt = Field(
            587,
            description="SMTP server port (587 for TLS, 465 for SSL)"
        )
        SMTP_USER: Optional[str] = Field(
            None,
            description="SMTP authentication username"
        )
        SMTP_PASSWORD: Optional[SecretStr] = Field(
            None,
            description="SMTP authentication password"
        )
        SMTP_TLS: bool = Field(
            True,
            description="Enable SMTP TLS encryption"
        )
        EMAIL_FROM: Optional[str] = Field(
            None,
            description="Default sender email address"
        )
        
        # Webhook configuration
        WEBHOOK_SECRET: Optional[SecretStr] = Field(
            None,
            description="Webhook signature verification secret"
        )
        WEBHOOK_TIMEOUT: PositiveInt = Field(
            30,
            description="Webhook request timeout in seconds"
        )
        
        # ===================
        # Development Settings
        # ===================
        AUTO_RELOAD: bool = Field(
            False,
            description="Enable automatic code reloading"
        )
        PROFILING_ENABLED: bool = Field(
            False,
            description="Enable performance profiling"
        )
        MOCK_EXTERNAL_SERVICES: bool = Field(
            False,
            description="Use mock implementations for external services"
        )
        
        class Config:
            """Pydantic model configuration."""
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True
            validate_assignment = True
            use_enum_values = True
            allow_population_by_field_name = True
        
        # ===================
        # Validation Methods
        # ===================
        
        @validator("SECRET_KEY", pre=True, always=True)
        def generate_secret_key(cls, v: Any) -> str:
            """Generate a secure secret key if not provided."""
            if not v or (isinstance(v, str) and len(v) < 32):
                logger.warning("Generating new SECRET_KEY. Set SECRET_KEY environment variable for production.")
                return secrets.token_urlsafe(32)
            return v
        
        @validator("DEBUG", pre=True, always=True)
        def set_debug_mode(cls, v: Any, values: Dict[str, Any]) -> bool:
            """Set debug mode based on environment."""
            environment = values.get("ENVIRONMENT", Environment.DEVELOPMENT)
            if environment == Environment.DEVELOPMENT:
                return True
            elif environment == Environment.TESTING:
                return True
            return bool(v)
        
        @validator("CORS_ORIGINS", pre=True)
        def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
            """Parse CORS origins from string or list."""
            if isinstance(v, str):
                return [origin.strip() for origin in v.split(",")]
            elif isinstance(v, list):
                return v
            return ["*"]
        
        @validator("LOG_LEVEL", pre=True)
        def validate_log_level(cls, v: Any, values: Dict[str, Any]) -> LogLevel:
            """Validate and adjust log level based on environment."""
            environment = values.get("ENVIRONMENT", Environment.DEVELOPMENT)
            
            if environment == Environment.DEVELOPMENT:
                return LogLevel.DEBUG
            elif environment == Environment.PRODUCTION:
                return LogLevel.WARNING if v == LogLevel.DEBUG else v
            
            return v
        
        @root_validator
        def validate_cloud_configuration(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            """Validate cloud provider configuration consistency."""
            cloud_provider = values.get("CLOUD_PROVIDER")
            
            if cloud_provider == CloudProvider.AWS:
                # Check AWS credentials
                if not values.get("AWS_ACCESS_KEY_ID") and not os.getenv("AWS_PROFILE"):
                    logger.warning("AWS credentials not configured. AWS features may not work.")
                
                # Validate S3 bucket for artifact storage
                if not values.get("S3_BUCKET") and values.get("ENVIRONMENT") == Environment.PRODUCTION:
                    logger.warning("S3_BUCKET not configured for production AWS deployment")
                    
            elif cloud_provider == CloudProvider.GCP:
                # Check GCP credentials
                if not values.get("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    logger.warning("GCP credentials not configured. GCP features may not work.")
                    
            elif cloud_provider == CloudProvider.AZURE:
                # Check Azure credentials
                if not values.get("AZURE_ACCOUNT_KEY") and not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
                    logger.warning("Azure credentials not configured. Azure features may not work.")
            
            return values
        
        @root_validator
        def validate_database_configuration(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            """Validate database configuration completeness."""
            database_url = values.get("DATABASE_URL")
            
            if not database_url:
                # Check individual database components
                required_fields = ["DB_HOST", "DB_USER", "DB_NAME"]
                missing_fields = [field for field in required_fields if not values.get(field)]
                
                if missing_fields:
                    logger.warning(f"Database configuration incomplete. Missing: {missing_fields}")
            
            return values
        
        @root_validator
        def validate_production_requirements(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            """Validate production deployment requirements."""
            if values.get("ENVIRONMENT") == Environment.PRODUCTION:
                warnings = []
                
                # Check security settings
                if not values.get("ENABLE_HTTPS"):
                    warnings.append("HTTPS should be enabled in production")
                
                if values.get("DEBUG"):
                    warnings.append("DEBUG mode should be disabled in production")
                    values["DEBUG"] = False
                
                if values.get("CORS_ORIGINS") == ["*"]:
                    warnings.append("CORS origins should be restricted in production")
                
                # Check monitoring
                if not values.get("ENABLE_MONITORING"):
                    warnings.append("Monitoring should be enabled in production")
                
                # Check MLflow configuration
                if not values.get("MLFLOW_TRACKING_URI"):
                    warnings.append("MLflow tracking URI should be configured for production")
                
                if warnings:
                    logger.warning(f"Production deployment warnings: {warnings}")
            
            return values
        
        # ===================
        # Configuration Properties
        # ===================
        
        @property
        def database(self) -> DatabaseConfig:
            """Get database configuration helper."""
            return DatabaseConfig(self)
        
        @property
        def mlflow(self) -> MLflowConfig:
            """Get MLflow configuration helper."""
            return MLflowConfig(self)
        
        @property
        def feast(self) -> FeastConfig:
            """Get Feast feature store configuration helper."""
            return FeastConfig(self)
        
        @property
        def is_production(self) -> bool:
            """Check if running in production environment."""
            return self.ENVIRONMENT == Environment.PRODUCTION
        
        @property
        def is_development(self) -> bool:
            """Check if running in development environment."""
            return self.ENVIRONMENT == Environment.DEVELOPMENT
        
        @property
        def is_staging(self) -> bool:
            """Check if running in staging environment."""
            return self.ENVIRONMENT == Environment.STAGING
        
        @property
        def is_testing(self) -> bool:
            """Check if running in testing environment."""
            return self.ENVIRONMENT == Environment.TESTING or self.TESTING
        
        @property
        def redis_url(self) -> str:
            """Get properly formatted Redis connection URL."""
            if self.REDIS_URL:
                return self.REDIS_URL
            
            # Construct Redis URL from components
            auth_part = ""
            if self.REDIS_PASSWORD:
                password = self.REDIS_PASSWORD.get_secret_value()
                auth_part = f":{password}@"
            
            protocol = "rediss" if self.REDIS_SSL else "redis"
            return f"{protocol}://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        
        # ===================
        # Utility Methods
        # ===================
        
        def get_database_url(self, include_password: bool = True) -> str:
            """Get database URL with optional password masking."""
            url = self.database.url
            if not include_password:
                # Mask password in URL for logging
                import re
                url = re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)
            return url
        
        @lru_cache(maxsize=1)
        def get_cloud_storage_config(self) -> Dict[str, Any]:
            """Get cloud storage configuration for the current provider."""
            if self.CLOUD_PROVIDER == CloudProvider.AWS:
                return {
                    "provider": "aws",
                    "region": self.AWS_REGION,
                    "bucket": self.S3_BUCKET,
                    "access_key": self.AWS_ACCESS_KEY_ID.get_secret_value() if self.AWS_ACCESS_KEY_ID else None,
                    "secret_key": self.AWS_SECRET_ACCESS_KEY.get_secret_value() if self.AWS_SECRET_ACCESS_KEY else None,
                    "endpoint_url": None,  # Use default AWS endpoints
                }
            elif self.CLOUD_PROVIDER == CloudProvider.GCP:
                return {
                    "provider": "gcp",
                    "project_id": self.GCP_PROJECT_ID,
                    "bucket": self.GCS_BUCKET,
                    "credentials_path": self.GOOGLE_APPLICATION_CREDENTIALS,
                }
            elif self.CLOUD_PROVIDER == CloudProvider.AZURE:
                return {
                    "provider": "azure",
                    "account_name": self.AZURE_ACCOUNT,
                    "container": self.AZURE_CONTAINER,
                    "account_key": self.AZURE_ACCOUNT_KEY.get_secret_value() if self.AZURE_ACCOUNT_KEY else None,
                }
            else:
                return {
                    "provider": "local",
                    "base_path": Path(self.BASE_DIR) / "storage"
                }
        
        @lru_cache(maxsize=1)
        def get_logging_config(self) -> Dict[str, Any]:
            """Get comprehensive logging configuration."""
            formatters = {
                "json": {
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(funcName)s %(message)s"
                },
                "text": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            }
            
            handlers = {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.LOG_LEVEL.value,
                    "formatter": self.LOG_FORMAT,
                    "stream": "ext://sys.stdout"
                }
            }
            
            # Add file handler if configured
            if self.LOG_FILE:
                log_file_path = Path(self.LOG_FILE)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if self.LOG_ROTATION:
                    handlers["file"] = {
                        "class": "logging.handlers.RotatingFileHandler",
                        "filename": str(log_file_path),
                        "maxBytes": self._parse_size(self.LOG_MAX_SIZE),
                        "backupCount": self.LOG_BACKUP_COUNT,
                        "level": self.LOG_LEVEL.value,
                        "formatter": self.LOG_FORMAT,
                        "encoding": "utf-8"
                    }
                else:
                    handlers["file"] = {
                        "class": "logging.FileHandler",
                        "filename": str(log_file_path),
                        "level": self.LOG_LEVEL.value,
                        "formatter": self.LOG_FORMAT,
                        "encoding": "utf-8"
                    }
            
            # Add Elasticsearch handler if configured
            if self.ELASTICSEARCH_URL:
                try:
                    handlers["elasticsearch"] = {
                        "class": "cmreslogging.CMRESHandler",
                        "hosts": [{"host": self.ELASTICSEARCH_URL}],
                        "es_index_name": f"auto-analyst-logs-{self.ENVIRONMENT.value}",
                        "level": self.LOG_LEVEL.value,
                        "formatter": "json"
                    }
                except ImportError:
                    logger.warning("elasticsearch-logging not installed, skipping Elasticsearch handler")
            
            loggers = {
                "": {  # Root logger
                    "level": self.LOG_LEVEL.value,
                    "handlers": list(handlers.keys()),
                    "propagate": False
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                },
                "uvicorn.access": {
                    "level": "INFO" if self.is_development else "WARNING",
                    "handlers": ["console"],
                    "propagate": False
                },
                "sqlalchemy.engine": {
                    "level": "INFO" if self.DB_ECHO else "WARNING",
                    "handlers": ["console"],
                    "propagate": False
                },
                "mlflow": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                },
                "boto3": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False
                },
                "botocore": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False
                }
            }
            
            return {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": formatters,
                "handlers": handlers,
                "loggers": loggers
            }
        
        def _parse_size(self, size_str: str) -> int:
            """Parse size string (e.g., '100MB') to bytes."""
            size_str = size_str.upper().strip()
            multipliers = {
                "B": 1,
                "KB": 1024,
                "MB": 1024 ** 2,
                "GB": 1024 ** 3,
                "TB": 1024 ** 4
            }
            
            for suffix, multiplier in multipliers.items():
                if size_str.endswith(suffix):
                    try:
                        number = float(size_str[:-len(suffix)])
                        return int(number * multiplier)
                    except ValueError:
                        logger.warning(f"Invalid size format: {size_str}")
                        return 100 * 1024 * 1024  # Default to 100MB
            
            # If no suffix, assume bytes
            try:
                return int(size_str)
            except ValueError:
                logger.warning(f"Invalid size format: {size_str}")
                return 100 * 1024 * 1024  # Default to 100MB
        
        def setup_directories(self) -> None:
            """Create all required directories with proper permissions."""
            directories = [
                self.UPLOAD_DIRECTORY,
                self.TEMP_DIRECTORY,
                self.MODELS_DIRECTORY,
                self.ARTIFACTS_DIRECTORY,
                self.DATASETS_DIRECTORY,
            ]
            
            for directory in directories:
                dir_path = Path(directory)
                if not dir_path.is_absolute():
                    dir_path = Path(self.BASE_DIR) / directory
                
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    # Set proper permissions (readable/writable by owner, readable by group)
                    if os.name != 'nt':  # Not Windows
                        os.chmod(dir_path, 0o755)
                    logger.debug(f"Created directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
                    raise
        
        def validate_configuration(self) -> List[str]:
            """Comprehensive configuration validation with actionable feedback."""
            issues = []
            
            # Critical security checks
            secret_key_str = self.SECRET_KEY.get_secret_value()
            if len(secret_key_str) < 32:
                issues.append("SECRET_KEY must be at least 32 characters long")
            
            # Database validation
            try:
                db_url = self.database.url
                if not db_url:
                    issues.append("Database configuration is incomplete or invalid")
            except Exception as e:
                issues.append(f"Database configuration error: {str(e)}")
            
            # Cloud provider validation
            if self.CLOUD_PROVIDER != CloudProvider.LOCAL:
                cloud_config = self.get_cloud_storage_config()
                
                if self.CLOUD_PROVIDER == CloudProvider.AWS:
                    if not cloud_config.get("access_key") and not os.getenv("AWS_PROFILE"):
                        issues.append("AWS credentials not configured (set AWS_ACCESS_KEY_ID or use AWS_PROFILE)")
                    if self.is_production and not cloud_config.get("bucket"):
                        issues.append("S3_BUCKET must be configured for production AWS deployments")
                
                elif self.CLOUD_PROVIDER == CloudProvider.GCP:
                    if not cloud_config.get("credentials_path"):
                        issues.append("GCP credentials not configured (set GOOGLE_APPLICATION_CREDENTIALS)")
                    if self.is_production and not cloud_config.get("bucket"):
                        issues.append("GCS_BUCKET must be configured for production GCP deployments")
                
                elif self.CLOUD_PROVIDER == CloudProvider.AZURE:
                    if not cloud_config.get("account_key"):
                        issues.append("Azure credentials not configured (set AZURE_ACCOUNT_KEY)")
            
            # Production-specific validations
            if self.is_production:
                if self.DEBUG:
                    issues.append("DEBUG mode must be disabled in production")
                
                if not self.ENABLE_HTTPS:
                    issues.append("HTTPS should be enabled in production (set ENABLE_HTTPS=true)")
                
                if self.CORS_ORIGINS == ["*"]:
                    issues.append("CORS origins should be restricted in production")
                
                if not self.ENABLE_MONITORING:
                    issues.append("Monitoring should be enabled in production")
                
                if self.LOG_LEVEL in [LogLevel.DEBUG]:
                    issues.append("Log level should be INFO or higher in production")
            
            # Remote training validation
            if self.ENABLE_REMOTE_TRAINING:
                if self.DEFAULT_COMPUTE_BACKEND == ComputeBackend.KAGGLE:
                    if not self.KAGGLE_USERNAME or not self.KAGGLE_KEY:
                        issues.append("Kaggle credentials required for Kaggle compute backend")
                
                elif self.DEFAULT_COMPUTE_BACKEND == ComputeBackend.COLAB:
                    if not self.COLAB_NOTEBOOK_TEMPLATE:
                        issues.append("Colab notebook template required for Colab compute backend")
            
            # Storage directory validation
            try:
                self.setup_directories()
            except Exception as e:
                issues.append(f"Failed to create required directories: {str(e)}")
            
            return issues

else:
    # Fallback configuration class for environments without Pydantic
    class Settings:
        """
        Fallback configuration class without Pydantic validation.
        
        This provides basic configuration management when Pydantic is not available,
        but lacks validation and type safety features.
        """
        
        def __init__(self):
            # Core settings
            self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
            self.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
            self.TESTING = os.getenv("TESTING", "false").lower() in ("true", "1", "yes")
            
            # Application settings
            self.APP_NAME = os.getenv("APP_NAME", "Auto-Analyst")
            self.APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
            self.SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
            
            # Server settings
            self.HOST = os.getenv("HOST", "0.0.0.0")
            self.PORT = int(os.getenv("PORT", "8000"))
            self.WORKERS = int(os.getenv("WORKERS", "1"))
            
            # Database settings
            self.DATABASE_URL = os.getenv("DATABASE_URL")
            self.DB_TYPE = os.getenv("DB_TYPE", "postgresql")
            self.DB_HOST = os.getenv("DB_HOST", "localhost")
            self.DB_PORT = int(os.getenv("DB_PORT", "5432"))
            self.DB_USER = os.getenv("DB_USER", "postgres")
            self.DB_PASSWORD = os.getenv("DB_PASSWORD", "")
            self.DB_NAME = os.getenv("DB_NAME", "auto_analyst")
            self.DB_ECHO = os.getenv("DB_ECHO", "false").lower() in ("true", "1", "yes")
            
            # Cache settings
            self.REDIS_URL = os.getenv("REDIS_URL")
            self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
            self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
            self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
            self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
            
            # Cloud settings
            self.CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "local")
            self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
            self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
            self.AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
            self.S3_BUCKET = os.getenv("S3_BUCKET")
            
            # MLflow settings
            self.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file://./mlruns")
            self.MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
            
            # Monitoring
            self.ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() in ("true", "1", "yes")
            self.PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() in ("true", "1", "yes")
            
            # Logging settings
            self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
            self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
            self.LOG_FILE = os.getenv("LOG_FILE")
            
            # Performance settings
            self.MAX_DATASET_SIZE_GB = float(os.getenv("MAX_DATASET_SIZE_GB", "20.0"))
            self.UPLOAD_MAX_SIZE = int(os.getenv("UPLOAD_MAX_SIZE", str(20 * 1024 * 1024 * 1024)))
            
            # Storage directories
            self.BASE_DIR = str(Path(__file__).parent.parent)
            self.UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY", "./uploads")
            self.TEMP_DIRECTORY = os.getenv("TEMP_DIRECTORY", "./temp")
            self.MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "./models")
            self.ARTIFACTS_DIRECTORY = os.getenv("ARTIFACTS_DIRECTORY", "./artifacts")
            self.DATASETS_DIRECTORY = os.getenv("DATASETS_DIRECTORY", "./datasets")
        
        @property
        def database(self) -> DatabaseConfig:
            """Get database configuration helper."""
            return DatabaseConfig(self)
        
        @property
        def mlflow(self) -> MLflowConfig:
            """Get MLflow configuration helper."""
            return MLflowConfig(self)
        
        @property
        def feast(self) -> FeastConfig:
            """Get Feast configuration helper."""
            return FeastConfig(self)
        
        @property
        def is_production(self) -> bool:
            """Check if running in production environment."""
            return self.ENVIRONMENT == "production"
        
        @property
        def is_development(self) -> bool:
            """Check if running in development environment."""
            return self.ENVIRONMENT == "development"
        
        @property
        def is_staging(self) -> bool:
            """Check if running in staging environment."""
            return self.ENVIRONMENT == "staging"
        
        @property
        def is_testing(self) -> bool:
            """Check if running in testing environment."""
            return self.ENVIRONMENT == "testing" or self.TESTING
        
        @property
        def redis_url(self) -> str:
            """Get Redis connection URL."""
            if self.REDIS_URL:
                return self.REDIS_URL
            
            auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            return f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        
        def setup_directories(self) -> None:
            """Create required directories."""
            directories = [
                self.UPLOAD_DIRECTORY,
                self.TEMP_DIRECTORY,
                self.MODELS_DIRECTORY,
                self.ARTIFACTS_DIRECTORY,
                self.DATASETS_DIRECTORY,
            ]
            
            for directory in directories:
                dir_path = Path(directory)
                if not dir_path.is_absolute():
                    dir_path = Path(self.BASE_DIR) / directory
                dir_path.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()

# Configuration initialization and validation
def validate_and_setup_config() -> None:
    """
    Validate configuration and setup necessary components.
    
    This function should be called once during application startup to ensure
    all configuration is valid and required directories exist.
    """
    try:
        # Validate configuration if available
        if PYDANTIC_AVAILABLE and hasattr(settings, 'validate_configuration'):
            issues = settings.validate_configuration()
            if issues:
                logger.warning(f"Configuration validation issues: {issues}")
                # In development, continue with warnings
                # In production, consider raising an exception
                if settings.is_production and any("must" in issue.lower() or "required" in issue.lower() for issue in issues):
                    raise ValueError(f"Critical configuration issues in production: {issues}")
        
        # Setup required directories
        settings.setup_directories()
        
        # Setup logging configuration
        if hasattr(settings, 'get_logging_config'):
            import logging.config
            try:
                logging_config = settings.get_logging_config()
                logging.config.dictConfig(logging_config)
                logger.info(f"Logging configuration applied for {settings.ENVIRONMENT} environment")
            except Exception as e:
                logger.error(f"Failed to apply logging configuration: {e}")
                # Fallback to basic logging
                _setup_basic_logging()
        else:
            _setup_basic_logging()
        
        # Log successful initialization
        logger.info(f"Configuration initialized successfully")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug mode: {getattr(settings, 'DEBUG', False)}")
        logger.info(f"Cloud provider: {getattr(settings, 'CLOUD_PROVIDER', 'local')}")
        
        # Log configuration summary in debug mode
        if hasattr(settings, 'DEBUG') and settings.DEBUG:
            config_summary = get_config_summary()
            logger.debug(f"Configuration summary: {config_summary}")
        
    except Exception as e:
        logger.error(f"Configuration initialization failed: {str(e)}")
        # Don't raise here to allow graceful degradation
        # The application can decide how to handle configuration errors

def _setup_basic_logging() -> None:
    """Setup basic logging configuration as fallback."""
    log_level = getattr(settings, 'LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Utility functions
@lru_cache(maxsize=1)
def get_config_summary() -> Dict[str, Any]:
    """
    Get configuration summary for debugging and monitoring.
    
    Returns:
        Dictionary containing key configuration information.
    """
    summary = {
        "app_name": getattr(settings, 'APP_NAME', 'Auto-Analyst'),
        "app_version": getattr(settings, 'APP_VERSION', '2.0.0'),
        "environment": getattr(settings, 'ENVIRONMENT', 'unknown'),
        "debug_mode": getattr(settings, 'DEBUG', False),
        "cloud_provider": getattr(settings, 'CLOUD_PROVIDER', 'local'),
        "monitoring_enabled": getattr(settings, 'ENABLE_MONITORING', False),
        "remote_training_enabled": getattr(settings, 'ENABLE_REMOTE_TRAINING', False),
        "database_configured": bool(getattr(settings, 'DATABASE_URL', None)),
        "redis_configured": bool(getattr(settings, 'REDIS_URL', None)),
        "mlflow_configured": bool(getattr(settings, 'MLFLOW_TRACKING_URI', None)),
    }
    
    # Add database URL (without password) if available
    try:
        if hasattr(settings, 'get_database_url'):
            summary["database_url"] = settings.get_database_url(include_password=False)
        elif hasattr(settings, 'database'):
            summary["database_url"] = "[configured]"
        else:
            summary["database_url"] = "[not configured]"
    except Exception:
        summary["database_url"] = "[configuration error]"
    
    return summary

def export_config_template() -> str:
    """
    Export a comprehensive configuration template for .env files.
    
    Returns:
        String containing a complete .env file template with documentation.
    """
    return """
# ===========================================
# Auto-Analyst Platform Configuration Template
# ===========================================
# Copy this to .env and customize for your environment
# Documentation: https://docs.auto-analyst.com/configuration

# ===================
# Environment Settings
# ===================
ENVIRONMENT=development  # development, staging, production, testing
DEBUG=true
TESTING=false

# ===================
# Application Settings
# ===================
APP_NAME=Auto-Analyst
APP_VERSION=2.0.0
SECRET_KEY=your-super-secret-key-at-least-32-characters-long
API_V1_STR=/api/v1

# ===================
# Server Configuration
# ===================
HOST=0.0.0.0
PORT=8000
WORKERS=1
RELOAD=true

# ===================
# Database Configuration
# ===================
# Option 1: Complete database URL
DATABASE_URL=postgresql://username:password@localhost:5432/auto_analyst

# Option 2: Individual components (used if DATABASE_URL not set)
DB_TYPE=postgresql  # postgresql, mysql, sqlite
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your-db-password
DB_NAME=auto_analyst
DB_ECHO=false
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# ===================
# Cache Configuration
# ===================
CACHE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
# Or individual components:
# REDIS_HOST=localhost
# REDIS_PORT=6379
# REDIS_DB=0
# REDIS_PASSWORD=your-redis-password
# REDIS_SSL=false

# ===================
# Cloud Provider Settings
# ===================
CLOUD_PROVIDER=local  # local, aws, gcp, azure

# AWS Settings
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
S3_BUCKET=your-s3-bucket

# AWS Data Services (optional)
REDSHIFT_CLUSTER_ID=your-redshift-cluster
REDSHIFT_DATABASE=dev
REDSHIFT_USER=awsuser
REDSHIFT_PASSWORD=your-redshift-password

# GCP Settings
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GCP_PROJECT_ID=your-gcp-project
GCS_BUCKET=your-gcs-bucket

# Azure Settings
AZURE_ACCOUNT=your-storage-account
AZURE_CONTAINER=your-container
AZURE_ACCOUNT_KEY=your-account-key

# ===================
# MLflow Configuration
# ===================
MLFLOW_TRACKING_URI=file://./mlruns
# For production, use a dedicated MLflow server:
# MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_REGISTRY_URI=
MLFLOW_EXPERIMENT_NAME=auto-analyst-experiment
MLFLOW_ARTIFACT_ROOT=./artifacts/mlflow

# ===================
# Feast Feature Store
# ===================
FEAST_REPO_PATH=./feast_repo
FEAST_ONLINE_STORE=redis
FEAST_OFFLINE_STORE=file

# ===================
# ML Pipeline Settings
# ===================
DEFAULT_COMPUTE_BACKEND=local  # local, kaggle, colab, aws_sagemaker, gcp_vertex_ai
ENABLE_REMOTE_TRAINING=true
MAX_DATASET_SIZE_GB=20.0
MAX_TRAINING_TIME_HOURS=24
MAX_MODELS_PER_EXPERIMENT=50
DEFAULT_TEST_SIZE=0.2
DEFAULT_CV_FOLDS=5
ENABLE_HYPERPARAMETER_TUNING=true
ENABLE_ENSEMBLE_MODELS=true
ENABLE_DEEP_LEARNING=false

# Remote Training Platforms
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-api-key
COLAB_NOTEBOOK_TEMPLATE=./templates/colab_template.ipynb

# ===================
# Monitoring & Observability
# ===================
ENABLE_MONITORING=true
ENABLE_DRIFT_DETECTION=true
DRIFT_DETECTION_THRESHOLD=0.1
MONITORING_RETENTION_DAYS=90

# Prometheus Metrics
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
PROMETHEUS_METRICS_PATH=/metrics

# ===================
# Logging Configuration
# ===================
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json, text
LOG_FILE=./logs/auto-analyst.log
LOG_ROTATION=true
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5

# Centralized Logging (optional)
ELASTICSEARCH_URL=http://elasticsearch:9200
LOGSTASH_HOST=logstash
LOGSTASH_PORT=5044

# ===================
# Security Settings
# ===================
ENABLE_HTTPS=false
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# Authentication
ACCESS_TOKEN_EXPIRE_MINUTES=10080  # 7 days
REFRESH_TOKEN_EXPIRE_MINUTES=43200  # 30 days
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_SPECIAL=true
PASSWORD_REQUIRE_NUMBERS=true

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS,PATCH
CORS_HEADERS=*

# ===================
# Performance Settings
# ===================
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=300
UPLOAD_MAX_SIZE=21474836480  # 20GB in bytes
CHUNK_SIZE=8388608  # 8MB in bytes

# Caching
ENABLE_CACHING=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# ===================
# Storage Directories
# ===================
BASE_DIR=/app
UPLOAD_DIRECTORY=./uploads
TEMP_DIRECTORY=./temp
MODELS_DIRECTORY=./models
ARTIFACTS_DIRECTORY=./artifacts
DATASETS_DIRECTORY=./datasets

# ===================
# External Services
# ===================
# Email/SMTP Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_TLS=true
EMAIL_FROM=noreply@auto-analyst.com

# Webhook Configuration
WEBHOOK_SECRET=your-webhook-secret
WEBHOOK_TIMEOUT=30

# ===================
# Development Settings
# ===================
AUTO_RELOAD=true
PROFILING_ENABLED=false
MOCK_EXTERNAL_SERVICES=false
""".strip()

# Initialize configuration on module import
try:
    validate_and_setup_config()
except Exception as e:
    # Log the error but don't crash the module import
    print(f"Configuration initialization warning: {str(e)}")

# Export public interface
__all__ = [
    "settings",
    "Environment",
    "LogLevel", 
    "CloudProvider",
    "ComputeBackend",
    "DatabaseType",
    "CacheBackend",
    "DatabaseConfig",
    "MLflowConfig",
    "FeastConfig",
    "validate_and_setup_config",
    "get_config_summary",
    "export_config_template"
]

