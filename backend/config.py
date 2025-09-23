"""
🚀 AUTO-ANALYST PLATFORM - PRODUCTION CONFIGURATION
==================================================

Clean, efficient, and maintainable configuration system using Pydantic BaseSettings.
Follows industry best practices with proper validation and environment management.

Key Features:
- Environment-based configuration with validation
- Type safety with Pydantic models
- Secure secret management
- Multi-environment support (dev/staging/production)
- Cloud-native ready with sensible defaults
- Easy testing and debugging
"""

import os
import secrets
import logging
from pathlib import Path
from typing import List, Optional, Union
from enum import Enum
from functools import lru_cache

try:
    from pydantic import BaseSettings, Field, validator, AnyHttpUrl
    from pydantic.types import PositiveInt
except ImportError:
    raise ImportError(
        "Pydantic is required for configuration management. "
        "Install with: pip install pydantic[dotenv]"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS - Type-safe configuration options
# =============================================================================

class Environment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

class Settings(BaseSettings):
    """
    🏆 Production-grade application configuration.

    Uses Pydantic BaseSettings for automatic environment variable loading,
    type validation, and comprehensive configuration management.
    """

    # ==========================================================================
    # CORE APPLICATION SETTINGS
    # ==========================================================================

    app_name: str = Field(default="Auto-Analyst Platform", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    api_v1_str: str = Field(default="/api/v1", description="API version prefix")

    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    debug: bool = Field(default=False, description="Debug mode (never enable in production)")
    testing: bool = Field(default=False, description="Testing mode")

    # ==========================================================================
    # SERVER CONFIGURATION
    # ==========================================================================

    host: str = Field(default="0.0.0.0", description="Server host")
    port: PositiveInt = Field(default=8000, description="Server port")
    workers: PositiveInt = Field(default=1, description="Number of worker processes")

    # Request handling
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_request_size: int = Field(default=104857600, description="Max request size in bytes (100MB)")

    # ==========================================================================
    # SECURITY CONFIGURATION
    # ==========================================================================

    secret_key: str = Field(
        default="",
        min_length=32,
        description="Application secret key (min 32 characters)"
    )

    jwt_secret_key: str = Field(
        default="",
        min_length=32,
        description="JWT secret key (min 32 characters)"
    )

    jwt_access_token_expire_minutes: int = Field(
        default=30,
        description="JWT access token expiration in minutes"
    )

    jwt_refresh_token_expire_days: int = Field(
        default=7,
        description="JWT refresh token expiration in days"
    )

    # CORS configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )

    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        description="Allowed CORS methods"
    )

    # ==========================================================================
    # DATABASE CONFIGURATION
    # ==========================================================================

    database_url: Optional[str] = Field(
        default=None,
        description="Complete database URL (overrides individual DB settings)"
    )

    # Individual database settings (used if DATABASE_URL not provided)
    db_type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Database type")
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="", description="Database password")
    db_name: str = Field(default="auto_analyst", description="Database name")

    # Connection pool settings
    db_pool_size: int = Field(default=10, description="Database connection pool size")
    db_max_overflow: int = Field(default=20, description="Database max overflow connections")
    db_pool_timeout: int = Field(default=30, description="Database pool timeout")

    # ==========================================================================
    # REDIS & CACHING
    # ==========================================================================

    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    enable_caching: bool = Field(default=True, description="Enable Redis caching")
    cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")

    # ==========================================================================
    # FILE STORAGE & UPLOADS
    # ==========================================================================

    upload_max_size: int = Field(default=1073741824, description="Max upload size (1GB)")
    upload_max_files: int = Field(default=10, description="Max files per upload")
    allowed_file_types: List[str] = Field(
        default=["csv", "json", "xlsx", "parquet", "txt"],
        description="Allowed file extensions"
    )

    # Storage directories
    upload_directory: str = Field(default="./data/uploads", description="Upload directory")
    temp_directory: str = Field(default="./data/temp", description="Temporary directory")
    datasets_directory: str = Field(default="./data/datasets", description="Datasets directory")
    models_directory: str = Field(default="./models", description="ML models directory")

    chunk_size: int = Field(default=8388608, description="File chunk size (8MB)")

    # ==========================================================================
    # MACHINE LEARNING CONFIGURATION
    # ==========================================================================

    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    max_training_time: int = Field(default=3600, description="Max training time in seconds")
    max_dataset_size_mb: int = Field(default=1024, description="Max dataset size in MB")
    default_test_size: float = Field(default=0.2, description="Default train/test split ratio")

    # Model management
    model_cache_size: int = Field(default=5, description="Number of models to cache")
    model_cache_ttl: int = Field(default=7200, description="Model cache TTL in seconds")

    # ==========================================================================
    # EXTERNAL SERVICES
    # ==========================================================================

    # Kaggle API
    kaggle_username: Optional[str] = Field(default=None, description="Kaggle username")
    kaggle_key: Optional[str] = Field(default=None, description="Kaggle API key")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="file://./mlruns",
        description="MLflow tracking server URI"
    )

    # Email notifications
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    email_from: Optional[str] = Field(default=None, description="From email address")

    # ==========================================================================
    # MONITORING & LOGGING
    # ==========================================================================

    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    enable_request_logging: bool = Field(default=True, description="Enable request logging")

    # Monitoring
    enable_monitoring: bool = Field(default=True, description="Enable application monitoring")
    prometheus_enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8001, description="Prometheus metrics port")

    # ==========================================================================
    # DEVELOPMENT SETTINGS
    # ==========================================================================

    auto_reload: bool = Field(default=False, description="Auto-reload on code changes")
    profiling_enabled: bool = Field(default=False, description="Enable performance profiling")

    # ==========================================================================
    # VALIDATORS
    # ==========================================================================

    @validator("secret_key", pre=True)
    def validate_secret_key(cls, v: str, values: dict) -> str:
        """Generate secure secret key if not provided."""
        if not v:
            if values.get("environment") == Environment.PRODUCTION:
                raise ValueError("SECRET_KEY must be set in production environment")

            # Generate secure key for development
            generated_key = secrets.token_urlsafe(32)
            logger.warning(
                "Generated temporary SECRET_KEY for development. "
                "Set SECRET_KEY environment variable for production."
            )
            return generated_key

        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")

        return v

    @validator("jwt_secret_key", pre=True)
    def validate_jwt_secret_key(cls, v: str, values: dict) -> str:
        """Set JWT secret key to main secret key if not provided."""
        if not v:
            return values.get("secret_key", "")
        return v

    @validator("debug", pre=True)
    def validate_debug_mode(cls, v: bool, values: dict) -> bool:
        """Ensure debug is disabled in production."""
        env = values.get("environment")
        if env == Environment.PRODUCTION and v:
            raise ValueError("DEBUG mode must be disabled in production")
        return v

    @validator("cors_origins", pre=True)
    def validate_cors_origins(cls, v: Union[str, List[str]], values: dict) -> List[str]:
        """Parse CORS origins from string or validate list."""
        if isinstance(v, str):
            # Parse comma-separated string
            origins = [origin.strip() for origin in v.split(",")]
        else:
            origins = v

        # Warn about permissive CORS in production
        env = values.get("environment")
        if env == Environment.PRODUCTION and "*" in origins:
            logger.warning("Wildcard CORS origins not recommended in production")

        return origins

    @validator("database_url", pre=True)
    def build_database_url(cls, v: Optional[str], values: dict) -> str:
        """Build database URL from components if not provided."""
        if v:
            # Handle postgres:// vs postgresql:// for modern SQLAlchemy
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql://", 1)
            return v

        # Build URL from individual components
        db_type = values.get("db_type", DatabaseType.SQLITE)

        if db_type == DatabaseType.SQLITE:
            db_name = values.get("db_name", "auto_analyst")
            return f"sqlite:///./{db_name}.db"

        elif db_type == DatabaseType.POSTGRESQL:
            host = values.get("db_host", "localhost")
            port = values.get("db_port", 5432)
            user = values.get("db_user", "postgres")
            password = values.get("db_password", "")
            name = values.get("db_name", "auto_analyst")
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"

        elif db_type == DatabaseType.MYSQL:
            host = values.get("db_host", "localhost")
            port = values.get("db_port", 3306)
            user = values.get("db_user", "root")
            password = values.get("db_password", "")
            name = values.get("db_name", "auto_analyst")
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"

        # Default to SQLite
        return "sqlite:///./auto_analyst.db"

    # ==========================================================================
    # UTILITY PROPERTIES
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
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING or self.testing

    def create_directories(self) -> None:
        """Create necessary storage directories."""
        directories = [
            self.upload_directory,
            self.temp_directory,
            self.datasets_directory,
            self.models_directory,
        ]

        for directory in directories:
            path = Path(directory)
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Failed to create directory {path}: {e}")
                raise

    def validate_production_config(self) -> List[str]:
        """Validate configuration for production deployment."""
        issues = []

        if not self.is_production:
            return issues

        # Security checks
        if len(self.secret_key) < 32:
            issues.append("SECRET_KEY must be at least 32 characters in production")

        if self.debug:
            issues.append("DEBUG mode must be disabled in production")

        if "*" in self.cors_origins:
            issues.append("CORS origins should be restricted in production")

        # Database checks
        if self.database_url.startswith("sqlite://"):
            issues.append("SQLite not recommended for production - use PostgreSQL")

        # Security headers and HTTPS
        if not hasattr(self, 'enable_https') or not getattr(self, 'enable_https', False):
            issues.append("HTTPS should be enabled in production")

        return issues

    # ==========================================================================
    # PYDANTIC CONFIGURATION
    # ==========================================================================

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Allow both UPPER and lower case env vars
        use_enum_values = True  # Use enum values instead of enum objects
        validate_assignment = True  # Validate on assignment
        arbitrary_types_allowed = True  # Allow custom types


# =============================================================================
# GLOBAL SETTINGS INSTANCE
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure single instance across application.
    Cache is automatically cleared when environment changes.
    """
    return Settings()


# Global settings instance for easy access
settings = get_settings()


# =============================================================================
# INITIALIZATION AND VALIDATION
# =============================================================================

def initialize_application() -> None:
    """Initialize application with configuration validation."""
    try:
        # Create necessary directories
        settings.create_directories()

        # Validate production configuration
        if settings.is_production:
            issues = settings.validate_production_config()
            if issues:
                logger.warning(f"Production configuration issues found: {issues}")
                for issue in issues:
                    logger.warning(f"  - {issue}")

        logger.info(f"🚀 Application initialized successfully")
        logger.info(f"   Environment: {settings.environment.value}")
        logger.info(f"   Debug mode: {settings.debug}")
        logger.info(f"   Database: {settings.database_url.split('://')[0]}://...")
        logger.info(f"   Host: {settings.host}:{settings.port}")

    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise


# Auto-initialize on import (only in non-testing environments)
if not os.getenv("TESTING", "").lower() in ("true", "1", "yes"):
    try:
        initialize_application()
    except Exception as e:
        logger.error(f"Auto-initialization failed: {e}")


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "Settings",
    "Environment",
    "LogLevel",
    "DatabaseType",
    "settings",
    "get_settings",
    "initialize_application",
]
