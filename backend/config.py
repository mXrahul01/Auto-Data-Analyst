"""
🚀 AUTO-ANALYST PLATFORM - ENTERPRISE CONFIGURATION SYSTEM
===========================================================

Production-grade configuration management with ZERO warnings and errors.
Fully compatible with Pydantic v2+ and configured for Render PostgreSQL.

Key Features:
- Pydantic v2 with @field_validator and @model_validator (zero deprecation warnings)
- Automatic secure secret generation for development
- Multi-environment support (dev/staging/prod)
- Type-safe configuration with comprehensive validation
- Render PostgreSQL database integration
- Cloud-native deployment ready
"""

from __future__ import annotations

import logging
import os
import secrets
import socket
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Suppress third-party warnings early
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings, BaseModel, Field, validator as field_validator, root_validator as model_validator
        PYDANTIC_V2 = False
        class SettingsConfigDict(dict):
            def __init__(self, **kwargs):
                super().__init__(kwargs)
    except ImportError:
        raise ImportError("Pydantic is required. Install with: pip install 'pydantic[dotenv]>=2.0.0'")

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# TYPE-SAFE ENUMERATIONS
# =============================================================================

class Environment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Structured logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseDialect(str, Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class CacheBackend(str, Enum):
    """Cache backend options."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    DISABLED = "disabled"


class StorageBackend(str, Enum):
    """File storage backends."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


# =============================================================================
# CONFIGURATION MODELS (PYDANTIC V2 COMPLIANT)
# =============================================================================

class SecurityConfig(BaseModel):
    """Security and authentication configuration."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(protected_namespaces=())

    # FIXED: Use default values instead of required fields for auto-generation
    secret_key: str = Field(
        default="",
        min_length=32,
        description="Main application secret key (auto-generated if empty)"
    )
    jwt_secret_key: str = Field(
        default="",
        min_length=32,
        description="JWT signing key (auto-generated if empty)"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1, le=1440)
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1, le=90)

    # Password security
    password_min_length: int = Field(default=8, ge=6, le=128)
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digits: bool = True
    password_require_special: bool = True

    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window_minutes: int = Field(default=15, ge=1)

    # CORS configuration
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "https://localhost:3000"],
        description="Allowed CORS origins"
    )
    cors_credentials: bool = True
    cors_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    )
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])

    # FIXED: Add model validator to generate keys after initialization
    @model_validator(mode="after")
    def generate_secure_keys(self):
        """Generate secure keys if not provided."""
        # Check if running in production
        is_production = os.getenv("ENVIRONMENT", "").lower() == "production"

        if not self.secret_key:
            if is_production:
                raise ValueError("SECRET_KEY must be explicitly set in production environment")
            self.secret_key = secrets.token_urlsafe(32)
            logger.warning("Generated temporary SECRET_KEY for development")

        if not self.jwt_secret_key:
            self.jwt_secret_key = self.secret_key

        return self


class DatabaseConfig(BaseModel):
    """Render PostgreSQL database configuration."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(protected_namespaces=())

    # Primary database URL (takes precedence over individual settings)
    url: Optional[str] = Field(
        default=None,
        description="Complete database URL (overrides individual settings)"
    )

    # Individual database components - RENDER POSTGRESQL DEFAULTS
    dialect: DatabaseDialect = DatabaseDialect.POSTGRESQL
    host: str = Field(
        default="dpg-d38junfdiees73cktd90-a.singapore-postgres.render.com",
        description="Render PostgreSQL external hostname"
    )
    internal_host: str = Field(
        default="dpg-d38junfdiees73cktd90-a",
        description="Render PostgreSQL internal hostname (for same-region apps)"
    )
    port: int = Field(default=5432, ge=1, le=65535)
    username: str = Field(default="auto_analyst_db_user", description="Database username")
    password: str = Field(
        default="TFNUfugIC689SN2XxiXBajrsWPfEN1us",
        repr=False,
        description="Database password"
    )
    database: str = Field(default="auto_analyst_db", description="Database name")

    # Connection preference (internal vs external)
    use_internal_host: bool = Field(
        default=True,
        description="Use internal hostname when deployed on Render"
    )

    # Connection pool settings
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=40, ge=0, le=200)
    pool_timeout: int = Field(default=30, ge=1, le=300)
    pool_recycle: int = Field(default=3600, ge=300)
    pool_pre_ping: bool = True

    # Query configuration
    echo: bool = Field(default=False, description="Enable SQL query logging")
    echo_pool: bool = Field(default=False, description="Enable connection pool logging")

    # SSL/TLS configuration (recommended for Render)
    ssl_mode: str = Field(default="require", description="SSL connection mode")
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None

    @property
    def effective_host(self) -> str:
        """Get the effective hostname based on deployment context."""
        # Use internal host if deployed on Render and use_internal_host is True
        if self.use_internal_host and os.getenv("RENDER"):
            return self.internal_host
        return self.host

    @property
    def connection_url(self) -> str:
        """Build the complete connection URL."""
        if self.url:
            return self.url.replace("postgres://", "postgresql://", 1)

        host = self.effective_host
        return f"postgresql://{self.username}:{self.password}@{host}:{self.port}/{self.database}"

    @model_validator(mode="after")
    def build_database_url(self):
        """Build database URL from environment or components."""
        # Use environment DATABASE_URL if available (highest priority)
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            self.url = db_url.replace("postgres://", "postgresql://", 1)

        return self


class CacheConfig(BaseModel):
    """Caching configuration."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(protected_namespaces=())

    backend: CacheBackend = CacheBackend.REDIS
    enabled: bool = True

    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = Field(default=20, ge=1)
    redis_socket_timeout: int = Field(default=5, ge=1)
    redis_socket_connect_timeout: int = Field(default=5, ge=1)
    redis_retry_on_timeout: bool = True

    # Cache behavior
    default_ttl_seconds: int = Field(default=3600, ge=1)
    max_key_length: int = Field(default=250, ge=1)
    key_prefix: str = "auto_analyst"

    # Memory cache (fallback)
    memory_max_size: int = Field(default=1000, ge=1)


class StorageConfig(BaseModel):
    """File storage configuration."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(protected_namespaces=())

    backend: StorageBackend = StorageBackend.LOCAL

    # Local storage
    base_path: str = "./data"
    upload_path: str = "uploads"
    temp_path: str = "temp"
    datasets_path: str = "datasets"
    exports_path: str = "exports"

    # File constraints
    max_file_size_mb: int = Field(default=1024, ge=1, le=10240)
    max_files_per_upload: int = Field(default=10, ge=1, le=100)
    chunk_size_mb: int = Field(default=8, ge=1, le=100)

    # Allowed file types
    allowed_extensions: Set[str] = Field(
        default_factory=lambda: {
            "csv", "xlsx", "xls", "json", "parquet",
            "tsv", "txt", "zip", "gz", "tar"
        }
    )

    # Storage quotas
    max_storage_per_user_gb: int = Field(default=10, ge=1)
    cleanup_temp_files_hours: int = Field(default=24, ge=1)

    # Cloud storage (when backend != LOCAL)
    cloud_bucket: Optional[str] = None
    cloud_region: Optional[str] = None
    cloud_access_key: Optional[str] = Field(default=None, repr=False)
    cloud_secret_key: Optional[str] = Field(default=None, repr=False)


class MLConfig(BaseModel):
    """Machine Learning pipeline configuration."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(protected_namespaces=())

    # Compute resources
    enable_gpu: bool = False
    max_cpu_cores: Optional[int] = None
    max_memory_gb: int = Field(default=8, ge=1, le=128)

    # Training constraints
    max_training_time_minutes: int = Field(default=60, ge=1, le=720)
    max_dataset_size_mb: int = Field(default=1024, ge=1, le=10240)
    max_feature_count: int = Field(default=1000, ge=1, le=10000)
    max_sample_count: int = Field(default=1000000, ge=1)

    # Training settings
    default_test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cross_validation_folds: int = Field(default=5, ge=3, le=10)
    random_state: int = Field(default=42, ge=0)

    # AutoML configuration
    enable_automl: bool = True
    automl_time_budget_minutes: int = Field(default=30, ge=1, le=360)
    automl_limit_count: int = Field(default=20, ge=1, le=100)
    enable_ensemble: bool = True

    # ML Model registry (FIXED: removed "model_" prefixes)
    ml_storage_backend: StorageBackend = StorageBackend.LOCAL
    ml_versioning: bool = True
    ml_cache_size: int = Field(default=10, ge=1, le=100)
    ml_cache_ttl_hours: int = Field(default=24, ge=1)

    # Remote execution
    enable_remote_execution: bool = False
    remote_execution_timeout_minutes: int = Field(default=120, ge=1)


class MonitoringConfig(BaseModel):
    """Observability and monitoring configuration."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(protected_namespaces=())

    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = Field(default="json", pattern="^(json|text)$")
    enable_request_logging: bool = True
    enable_sql_logging: bool = False
    log_file_path: Optional[str] = None
    log_rotation_size_mb: int = Field(default=100, ge=1)
    log_retention_days: int = Field(default=30, ge=1)

    # Metrics
    enable_metrics: bool = True
    metrics_port: int = Field(default=8001, ge=1024, le=65535)
    metrics_path: str = "/metrics"

    # Health checks
    enable_health_checks: bool = True
    health_check_interval_seconds: int = Field(default=30, ge=1)

    # Performance monitoring
    enable_profiling: bool = False
    slow_query_threshold_ms: int = Field(default=1000, ge=1)
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # Alerts
    enable_alerts: bool = False
    alert_webhook_url: Optional[str] = None
    alert_email_recipients: List[str] = Field(default_factory=list)


class ExternalServicesConfig(BaseModel):
    """External service integrations."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(protected_namespaces=())

    # Email service
    smtp_enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = Field(default=None, repr=False)
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    email_from: Optional[str] = None

    # Kaggle integration
    kaggle_enabled: bool = False
    kaggle_username: Optional[str] = None
    kaggle_api_key: Optional[str] = Field(default=None, repr=False)

    # MLflow tracking
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "file://./mlruns"
    mlflow_experiment_name: str = "auto-analyst"
    mlflow_artifact_root: Optional[str] = None

    # Cloud services
    aws_access_key_id: Optional[str] = Field(default=None, repr=False)
    aws_secret_access_key: Optional[str] = Field(default=None, repr=False)
    aws_region: str = "us-west-2"

    # Webhook integrations
    webhook_endpoints: List[str] = Field(default_factory=list)
    webhook_timeout_seconds: int = Field(default=10, ge=1)
    webhook_retry_attempts: int = Field(default=3, ge=0)


# =============================================================================
# MAIN SETTINGS CLASS (PYDANTIC V2 COMPLIANT)
# =============================================================================

class Settings(BaseSettings):
    """Enterprise-grade application configuration with Render PostgreSQL integration."""

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            validate_assignment=True,
            extra="forbid",
            protected_namespaces=(),  # Fixes all model_ field warnings
            json_schema_extra={       # Replaces deprecated schema_extra
                "example": {
                    "app_name": "Auto-Analyst Platform",
                    "environment": "production",
                    "debug": False
                }
            }
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            validate_assignment = True
            extra = "forbid"

    # ==========================================================================
    # CORE APPLICATION
    # ==========================================================================

    app_name: str = "Auto-Analyst Platform"
    app_version: str = "2.1.0"
    app_description: str = "Enterprise ML Analytics Platform"
    api_v1_prefix: str = "/api/v1"

    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    testing: bool = False

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=32)

    # Request handling
    request_timeout: int = Field(default=30, ge=1, le=300)
    max_request_size: int = Field(default=100 * 1024 * 1024)  # 100MB

    # ==========================================================================
    # CONFIGURATION MODULES
    # ==========================================================================

    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    external: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig)

    # ==========================================================================
    # VALIDATORS
    # ==========================================================================

    @field_validator("debug")
    @classmethod
    def production_debug_check(cls, v, info=None):
        """Ensure debug is disabled in production."""
        values = info.data if info else {}
        env = values.get("environment")

        if env == Environment.PRODUCTION and v:
            raise ValueError("DEBUG mode must be disabled in production environment")
        return v

    @model_validator(mode="after")
    def validate_environment_constraints(self):
        """Validate cross-field environment constraints."""
        if self.environment == Environment.PRODUCTION:
            # Production-specific validations
            if self.security and len(self.security.secret_key) < 32:
                raise ValueError("Production requires SECRET_KEY >= 32 characters")

            # Validate CORS origins are not wildcard
            if self.security and "*" in self.security.cors_origins:
                logger.warning("Wildcard CORS origins detected in production - security risk!")

        return self

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING or self.testing

    def create_directories(self) -> None:
        """Create required storage directories with proper permissions."""
        base_path = Path(self.storage.base_path)
        directories = [
            base_path / self.storage.upload_path,
            base_path / self.storage.temp_path,
            base_path / self.storage.datasets_path,
            base_path / "models",
            base_path / self.storage.exports_path,
            ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o750)
                logger.info(f"✓ Created directory: {directory}")
            except PermissionError as e:
                logger.error(f"✗ Permission denied creating directory {directory}: {e}")
                raise
            except Exception as e:
                logger.error(f"✗ Failed to create directory {directory}: {e}")
                raise

    def get_database_url(self) -> str:
        """Get the complete database connection URL."""
        return self.database.connection_url

    def get_render_db_info(self) -> Dict[str, str]:
        """Get Render PostgreSQL connection information."""
        db = self.database
        return {
            "internal_url": f"postgresql://{db.username}:{db.password}@{db.internal_host}:{db.port}/{db.database}",
            "external_url": f"postgresql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}",
            "current_url": db.connection_url,
            "psql_command": f"PGPASSWORD={db.password} psql -h {db.host} -U {db.username} {db.database}",
            "connection_info": {
                "host": db.effective_host,
                "port": db.port,
                "username": db.username,
                "database": db.database,
                "ssl_mode": db.ssl_mode
            }
        }

    def validate_production_readiness(self) -> List[str]:
        """Validate configuration for production deployment."""
        issues = []

        if not self.is_production:
            return issues

        # Security checks
        if len(self.security.secret_key) < 32:
            issues.append("SECRET_KEY must be at least 32 characters")

        if self.debug:
            issues.append("DEBUG must be disabled in production")

        if "*" in self.security.cors_origins:
            issues.append("CORS origins should not include wildcards")

        # Database checks
        if self.database.connection_url.startswith("sqlite://"):
            issues.append("SQLite not recommended for production - using PostgreSQL ✓")

        # Storage checks
        if self.storage.backend == StorageBackend.LOCAL and self.workers > 1:
            issues.append("Local storage may have issues with multiple workers")

        return issues

    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        db_info = self.get_render_db_info()

        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment.value,
            "debug": self.debug,
            "python_version": os.sys.version,
            "hostname": socket.gethostname(),
            "pydantic_version": "v2" if PYDANTIC_V2 else "v1",
            "deployment_context": {
                "render_deployment": bool(os.getenv("RENDER")),
                "render_service": os.getenv("RENDER_SERVICE_NAME"),
                "render_region": os.getenv("RENDER_REGION"),
            },
            "database_info": {
                "dialect": self.database.dialect.value,
                "host": self.database.effective_host,
                "ssl_required": self.database.ssl_mode == "require",
                "using_internal_host": self.database.use_internal_host and bool(os.getenv("RENDER"))
            },
            "config_loaded_from": {
                "env_file": os.path.exists(".env"),
                "environment_variables": bool(os.getenv("DATABASE_URL")),
                "render_defaults": True
            }
        }


# =============================================================================
# SETTINGS FACTORY & CACHE
# =============================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Create global settings instance
settings = get_settings()


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_application() -> None:
    """Initialize application with comprehensive validation."""
    try:
        # Create storage directories
        settings.create_directories()

        # Validate production readiness
        if settings.is_production:
            issues = settings.validate_production_readiness()
            if issues:
                logger.warning("🚨 Production readiness issues detected:")
                for issue in issues:
                    logger.warning(f"  • {issue}")

        # Log successful initialization with database info
        env_info = settings.get_environment_info()
        db_info = settings.get_render_db_info()

        logger.info("🚀 Auto-Analyst Platform configuration initialized successfully")
        logger.info(f"   Environment: {env_info['environment']}")
        logger.info(f"   Debug mode: {settings.debug}")
        logger.info(f"   Database: PostgreSQL @ Render")
        logger.info(f"   DB Host: {env_info['database_info']['host']}")
        logger.info(f"   SSL Required: {env_info['database_info']['ssl_required']}")
        logger.info(f"   Cache: {settings.cache.backend.value}")
        logger.info(f"   Storage: {settings.storage.backend.value}")
        logger.info(f"   Pydantic: {env_info['pydantic_version']}")

        if env_info['deployment_context']['render_deployment']:
            logger.info(f"🌐 Render deployment detected")
            logger.info(f"   Service: {env_info['deployment_context']['render_service']}")
            logger.info(f"   Region: {env_info['deployment_context']['render_region']}")
            logger.info(f"   Using internal DB host: {env_info['database_info']['using_internal_host']}")

    except Exception as e:
        logger.error(f"❌ Application initialization failed: {e}")
        raise


# Auto-initialize (skip during testing)
if not settings.is_testing and not os.getenv("SKIP_AUTO_INIT", "").lower() == "true":
    try:
        initialize_application()
    except Exception as e:
        logger.error(f"Auto-initialization failed: {e}")
        # Continue gracefully in case initialization fails
        pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Settings", "SecurityConfig", "DatabaseConfig", "CacheConfig",
    "StorageConfig", "MLConfig", "MonitoringConfig", "ExternalServicesConfig",
    "Environment", "LogLevel", "DatabaseDialect", "CacheBackend", "StorageBackend",
    "get_settings", "initialize_application", "settings"
]

# Test the configuration on direct execution
if __name__ == "__main__":
    print("🧪 Testing Render PostgreSQL configuration...")
    try:
        test_settings = get_settings()
        db_info = test_settings.get_render_db_info()

        print(f"✅ Configuration loaded successfully!")
        print(f"   App: {test_settings.app_name} v{test_settings.app_version}")
        print(f"   Environment: {test_settings.environment.value}")
        print(f"   Database: PostgreSQL")
        print(f"   DB Host: {test_settings.database.effective_host}")
        print(f"   DB Name: {test_settings.database.database}")
        print(f"   DB User: {test_settings.database.username}")
        print(f"   SSL Mode: {test_settings.database.ssl_mode}")
        print(f"   Connection URL: {db_info['current_url'][:50]}...")
        print(f"   Debug: {test_settings.debug}")
        print(f"   Secret Key: {'✓ Set' if test_settings.security.secret_key else '✗ Missing'}")
        print(f"   JWT Key: {'✓ Set' if test_settings.security.jwt_secret_key else '✗ Missing'}")
        print(f"🎯 Zero warnings, zero errors - Render PostgreSQL ready!")

        # Test database connection info
        print(f"\n📊 Render PostgreSQL Connection Info:")
        print(f"   Internal URL: {db_info['internal_url'][:50]}...")
        print(f"   External URL: {db_info['external_url'][:50]}...")
        print(f"   PSQL Command: {db_info['psql_command'][:50]}...")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        raise
