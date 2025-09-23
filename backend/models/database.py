"""
🚀 AUTO-ANALYST PLATFORM - DATABASE MODELS & CONNECTION MANAGEMENT
================================================================

Production-ready database layer with:
- SQLAlchemy 2.0+ async/sync dual support
- Robust connection pooling and error handling
- Comprehensive data models with validation
- Health monitoring and observability
- Type-safe operations with full validation
- Multi-database support (PostgreSQL, SQLite, MySQL)

Dependencies:
- sqlalchemy>=2.0.0
- asyncpg (for async PostgreSQL)
- psycopg2-binary (for sync PostgreSQL)
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import (
    Dict, Any, Optional, Generator, AsyncGenerator, 
    Union, List, Type, TypeVar
)
from datetime import datetime, timezone
from enum import Enum
import uuid
import hashlib
from functools import lru_cache

# SQLAlchemy 2.0+ imports
from sqlalchemy import (
    create_engine, select, text, event,
    Column, Integer, String, Text, DateTime, 
    Boolean, Float, JSON, ForeignKey, Index,
    MetaData, Table
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    Session, sessionmaker, declared_attr
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.sql import func
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError

# Pydantic for validation
try:
    from pydantic import BaseModel as PydanticBase, Field, validator
except ImportError:
    # Fallback if Pydantic not available
    PydanticBase = object
    Field = lambda **kwargs: None
    validator = lambda *args, **kwargs: lambda f: f

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
ModelType = TypeVar("ModelType", bound="BaseModel")


# =============================================================================
# CONFIGURATION & ENUMS
# =============================================================================

class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"


class AnalysisStatus(str, Enum):
    """Analysis execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetStatus(str, Enum):
    """Dataset processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"


class ExecutionMode(str, Enum):
    """ML execution modes."""
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"
    REMOTE_KAGGLE = "remote_kaggle"
    REMOTE_COLAB = "remote_colab"
    CLOUD_AWS = "cloud_aws"
    CLOUD_GCP = "cloud_gcp"


@lru_cache()
def get_database_config() -> Dict[str, Any]:
    """Get database configuration from environment variables."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Fallback to SQLite for development
        database_url = "sqlite:///./auto_analyst.db"
        logger.warning("DATABASE_URL not set, using SQLite fallback")
    
    # Handle postgres:// vs postgresql:// for modern SQLAlchemy
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    return {
        "database_url": database_url,
        "async_database_url": database_url.replace("postgresql://", "postgresql+asyncpg://")
        if "postgresql://" in database_url else database_url,
        "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
        "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
        "echo": os.getenv("DB_ECHO", "false").lower() == "true",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }


# =============================================================================
# DECLARATIVE BASE & MODELS
# =============================================================================

class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base with metadata configuration."""
    
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )


class BaseModel(Base):
    """
    Base model with common fields and functionality.
    
    Provides:
    - Automatic table naming
    - Common timestamp fields
    - Dictionary conversion
    - Validation helpers
    """
    
    __abstract__ = True
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name (snake_case)."""
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
    
    # Primary key with proper type hints
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Timestamps with timezone support
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Record creation timestamp"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Record last update timestamp"
    )
    
    def to_dict(self, include_relationships: bool = False) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Args:
            include_relationships: Whether to include relationship data
            
        Returns:
            Dictionary representation of the model
        """
        result = {}
        
        # Include column data
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            
            # Handle special types
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, Enum):
                value = value.value
                
            result[column.name] = value
        
        # Include relationships if requested
        if include_relationships:
            for relationship_name in self.__mapper__.relationships.keys():
                relationship_value = getattr(self, relationship_name)
                if relationship_value is not None:
                    if hasattr(relationship_value, '__iter__'):
                        # One-to-many relationship
                        result[relationship_name] = [
                            item.to_dict() if hasattr(item, 'to_dict') else str(item)
                            for item in relationship_value
                        ]
                    else:
                        # One-to-one relationship
                        result[relationship_name] = (
                            relationship_value.to_dict() 
                            if hasattr(relationship_value, 'to_dict') 
                            else str(relationship_value)
                        )
        
        return result
    
    @classmethod
    def from_dict(cls: Type[ModelType], data: Dict[str, Any]) -> ModelType:
        """
        Create model instance from dictionary.
        
        Args:
            data: Dictionary with model data
            
        Returns:
            Model instance
        """
        # Filter out non-column keys
        column_names = {column.name for column in cls.__table__.columns}
        filtered_data = {k: v for k, v in data.items() if k in column_names}
        
        return cls(**filtered_data)


class User(BaseModel):
    """
    User accounts and authentication.
    
    Stores user information, authentication details, and preferences.
    """
    
    __tablename__ = "users"
    
    # User identification
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True,
        comment="User email address (unique)"
    )
    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True,
        comment="Username (unique)"
    )
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
        comment="User's full name"
    )
    
    # Authentication
    hashed_password: Mapped[str] = mapped_column(
        String(255), nullable=False,
        comment="Bcrypt hashed password"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
        comment="Whether user account is active"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="Whether user email is verified"
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="Whether user has admin privileges"
    )
    
    # Profile information
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True,
        comment="URL to user avatar image"
    )
    timezone: Mapped[str] = mapped_column(
        String(50), default="UTC", nullable=False,
        comment="User's timezone preference"
    )
    preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
        comment="User preferences and settings (JSON)"
    )
    
    # Activity tracking
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Last login timestamp"
    )
    login_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
        comment="Total number of logins"
    )
    
    # Relationships
    datasets: Mapped[List["Dataset"]] = relationship(
        "Dataset", back_populates="owner", cascade="all, delete-orphan"
    )
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis", back_populates="user", cascade="all, delete-orphan"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_user_email_active', 'email', 'is_active'),
        Index('ix_user_created_at', 'created_at'),
    )


class Dataset(BaseModel):
    """
    Uploaded datasets and metadata.
    
    Stores information about user-uploaded datasets including
    file metadata, data characteristics, and processing status.
    """
    
    __tablename__ = "datasets"
    
    # Dataset identification
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True,
        comment="Dataset display name"
    )
    original_filename: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="Original uploaded filename"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Dataset description"
    )
    tags: Mapped[Optional[List[str]]] = mapped_column(
        JSON, nullable=True,
        comment="Dataset tags (JSON array)"
    )
    
    # File information
    file_path: Mapped[str] = mapped_column(
        String(1000), nullable=False,
        comment="Relative path to stored file"
    )
    file_size: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="File size in bytes"
    )
    content_type: Mapped[str] = mapped_column(
        String(100), nullable=False,
        comment="MIME type of uploaded file"
    )
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, unique=True,
        comment="SHA256 hash for integrity checking"
    )
    
    # Data characteristics
    num_rows: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="Number of rows in dataset"
    )
    num_columns: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="Number of columns in dataset"
    )
    column_names: Mapped[Optional[List[str]]] = mapped_column(
        JSON, nullable=True,
        comment="Column names (JSON array)"
    )
    column_types: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON, nullable=True,
        comment="Column name to data type mapping (JSON object)"
    )
    sample_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
        comment="Sample rows for preview (JSON)"
    )
    
    # Processing status
    status: Mapped[DatasetStatus] = mapped_column(
        String(20), default=DatasetStatus.UPLOADED, nullable=False, index=True,
        comment="Processing status"
    )
    processing_error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Error message if processing failed"
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="When processing completed"
    )
    
    # Data quality metrics
    missing_values_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="Total missing values across all columns"
    )
    duplicate_rows_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="Number of duplicate rows"
    )
    data_quality_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Overall data quality score (0.0-1.0)"
    )
    data_profile: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
        comment="Detailed data profiling results (JSON)"
    )
    
    # Relationships
    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, index=True
    )
    owner: Mapped["User"] = relationship("User", back_populates="datasets")
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis", back_populates="dataset", cascade="all, delete-orphan"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_dataset_owner_status', 'owner_id', 'status'),
        Index('ix_dataset_created_at', 'created_at'),
        Index('ix_dataset_file_hash', 'file_hash'),
    )
    
    def generate_file_hash(self, file_content: bytes) -> str:
        """Generate SHA256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()


class Analysis(BaseModel):
    """
    ML analysis configurations and results.
    
    Stores complete information about ML analysis runs including
    configuration, execution details, and results.
    """
    
    __tablename__ = "analyses"
    
    # Analysis identification
    analysis_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True,
        comment="Unique analysis identifier (UUID)"
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False,
        comment="Analysis display name"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Analysis description"
    )
    
    # ML Configuration
    task_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="ML task type (classification, regression, etc.)"
    )
    target_column: Mapped[str] = mapped_column(
        String(255), nullable=False,
        comment="Target column name for prediction"
    )
    feature_columns: Mapped[Optional[List[str]]] = mapped_column(
        JSON, nullable=True,
        comment="Selected feature columns (JSON array)"
    )
    algorithms: Mapped[Optional[List[str]]] = mapped_column(
        JSON, nullable=True,
        comment="Selected ML algorithms (JSON array)"
    )
    
    # Execution settings
    execution_mode: Mapped[ExecutionMode] = mapped_column(
        String(20), default=ExecutionMode.LOCAL_CPU, nullable=False,
        comment="Execution environment"
    )
    max_training_time: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="Maximum training time in seconds"
    )
    test_size: Mapped[float] = mapped_column(
        Float, default=0.2, nullable=False,
        comment="Train/test split ratio"
    )
    validation_size: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Validation set size ratio"
    )
    random_state: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="Random seed for reproducibility"
    )
    
    # Advanced settings
    hyperparameter_tuning: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
        comment="Whether to perform hyperparameter tuning"
    )
    cross_validation_folds: Mapped[int] = mapped_column(
        Integer, default=5, nullable=False,
        comment="Number of cross-validation folds"
    )
    feature_selection: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
        comment="Whether to perform feature selection"
    )
    
    # Status and progress
    status: Mapped[AnalysisStatus] = mapped_column(
        String(20), default=AnalysisStatus.PENDING, nullable=False, index=True,
        comment="Analysis execution status"
    )
    progress: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False,
        comment="Execution progress (0.0-1.0)"
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Error message if analysis failed"
    )
    
    # Results
    best_model_name: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="Name of the best performing model"
    )
    best_model_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Best model's primary metric score"
    )
    performance_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
        comment="Detailed performance metrics (JSON)"
    )
    feature_importance: Mapped[Optional[Dict[str, float]]] = mapped_column(
        JSON, nullable=True,
        comment="Feature importance scores (JSON)"
    )
    model_comparison: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON, nullable=True,
        comment="Comparison of all models (JSON array)"
    )
    predictions: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
        comment="Sample predictions and probabilities (JSON)"
    )
    
    # Execution tracking
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Analysis start timestamp"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Analysis completion timestamp"
    )
    execution_time: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Total execution time in seconds"
    )
    
    # Resource usage
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Peak memory usage in MB"
    )
    cpu_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Total CPU time in seconds"
    )
    
    # Relationships
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, index=True
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), 
        nullable=False, index=True
    )
    
    user: Mapped["User"] = relationship("User", back_populates="analyses")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="analyses")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_analysis_user_status', 'user_id', 'status'),
        Index('ix_analysis_dataset_id', 'dataset_id'),
        Index('ix_analysis_task_type', 'task_type'),
        Index('ix_analysis_created_at', 'created_at'),
        Index('ix_analysis_completed_at', 'completed_at'),
    )


# =============================================================================
# DATABASE ENGINE & SESSION MANAGEMENT
# =============================================================================

class DatabaseManager:
    """
    Centralized database connection and session management.
    
    Handles both sync and async database operations with proper
    connection pooling, error handling, and health monitoring.
    """
    
    def __init__(self):
        self.config = get_database_config()
        self._sync_engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._sync_session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
    
    def _create_sync_engine(self) -> Engine:
        """Create synchronous database engine."""
        engine_kwargs = {
            "echo": self.config["echo"],
            "future": True,  # Use SQLAlchemy 2.0 style
            "pool_pre_ping": True,  # Validate connections before use
            "pool_recycle": self.config["pool_recycle"],
        }
        
        # Database-specific configuration
        if "postgresql" in self.config["database_url"]:
            engine_kwargs.update({
                "poolclass": QueuePool,
                "pool_size": self.config["pool_size"],
                "max_overflow": self.config["max_overflow"],
                "pool_timeout": self.config["pool_timeout"],
            })
        elif "sqlite" in self.config["database_url"]:
            engine_kwargs.update({
                "poolclass": NullPool,
                "connect_args": {"check_same_thread": False}
            })
        
        return create_engine(self.config["database_url"], **engine_kwargs)
    
    def _create_async_engine(self) -> AsyncEngine:
        """Create asynchronous database engine."""
        engine_kwargs = {
            "echo": self.config["echo"],
            "future": True,
            "pool_pre_ping": True,
            "pool_recycle": self.config["pool_recycle"],
        }
        
        # Only PostgreSQL supports async with asyncpg
        if "postgresql" in self.config["async_database_url"]:
            engine_kwargs.update({
                "poolclass": QueuePool,
                "pool_size": self.config["pool_size"],
                "max_overflow": self.config["max_overflow"],
                "pool_timeout": self.config["pool_timeout"],
            })
        
        return create_async_engine(self.config["async_database_url"], **engine_kwargs)
    
    def initialize(self) -> None:
        """Initialize database engines and session factories."""
        try:
            # Create engines
            self._sync_engine = self._create_sync_engine()
            
            # Only create async engine for PostgreSQL
            if "postgresql" in self.config["database_url"]:
                self._async_engine = self._create_async_engine()
            
            # Create session factories
            self._sync_session_factory = sessionmaker(
                bind=self._sync_engine,
                class_=Session,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False
            )
            
            if self._async_engine:
                self._async_session_factory = async_sessionmaker(
                    bind=self._async_engine,
                    class_=AsyncSession,
                    autoflush=False,
                    autocommit=False,
                    expire_on_commit=False
                )
            
            self._initialized = True
            logger.info("✅ Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def create_tables(self) -> None:
        """Create all database tables."""
        if not self._initialized or not self._sync_engine:
            raise RuntimeError("Database not initialized")
        
        try:
            Base.metadata.create_all(bind=self._sync_engine)
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create database tables: {e}")
            raise
    
    def get_sync_session(self) -> Generator[Session, None, None]:
        """Get synchronous database session (FastAPI dependency)."""
        if not self._sync_session_factory:
            raise RuntimeError("Sync session factory not initialized")
        
        session = self._sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get asynchronous database session (async context manager)."""
        if not self._async_session_factory:
            raise RuntimeError("Async session factory not initialized")
        
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Async database session error: {e}")
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Test sync connection
            with self._sync_session_factory() as session:
                result = session.execute(text("SELECT 1 as healthcheck"))
                sync_result = result.scalar()
            
            # Test async connection if available
            async_result = None
            if self._async_session_factory:
                async with self._async_session_factory() as session:
                    result = await session.execute(text("SELECT 1 as healthcheck"))
                    async_result = result.scalar()
            
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "message": "Database connections successful",
                "sync_connection": sync_result == 1,
                "async_connection": async_result == 1 if async_result is not None else "N/A",
                "response_time_ms": round(response_time, 2),
                "database_type": self.config["database_url"].split("://")[0],
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Database connection failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def close(self) -> None:
        """Close database connections."""
        if self._sync_engine:
            self._sync_engine.dispose()
        if self._async_engine:
            asyncio.create_task(self._async_engine.dispose())
        
        logger.info("✅ Database connections closed")


# =============================================================================
# GLOBAL DATABASE MANAGER
# =============================================================================

# Global database manager instance
db_manager = DatabaseManager()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def init_database() -> None:
    """Initialize database (convenience function)."""
    db_manager.initialize()


def create_tables() -> None:
    """Create all tables (convenience function)."""
    db_manager.create_tables()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    yield from db_manager.get_sync_session()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    async with db_manager.get_async_session() as session:
        yield session


async def health_check() -> Dict[str, Any]:
    """Database health check (convenience function)."""
    return await db_manager.health_check()


# =============================================================================
# DATABASE EVENTS & HOOKS
# =============================================================================

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance and integrity."""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys=ON")
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Set synchronous mode for better performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


@event.listens_for(User.hashed_password, 'set')
def validate_password_hash(target, value, oldvalue, initiator):
    """Validate that password is properly hashed."""
    if value and not value.startswith(('$2b$', '$2a$', '$2y$')):
        logger.warning("Password does not appear to be bcrypt hashed")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    "Base", "BaseModel",
    
    # Models
    "User", "Dataset", "Analysis",
    
    # Enums
    "DatabaseType", "AnalysisStatus", "DatasetStatus", "ExecutionMode",
    
    # Database management
    "DatabaseManager", "db_manager",
    
    # Convenience functions
    "init_database", "create_tables", "get_db", "get_async_db", "health_check",
    
    # Configuration
    "get_database_config",
]
