"""
Database Models and Connection Management for Auto-Analyst Platform

This module provides SQLAlchemy ORM models and database connection management
for the Auto-Analyst zero-code AI-powered data analysis web application.

Features:
- Multi-database support (PostgreSQL, MySQL, SQLite)
- Environment-based configuration
- Connection pooling and retry logic
- Session management with context managers
- Base models with common functionality
- Production-ready error handling and logging
- FastAPI integration with dependency injection
- Migration-ready model definitions

Models:
- User: User accounts and authentication
- Dataset: Uploaded datasets and metadata
- Analysis: ML analysis results and configurations
- MLModel: Trained machine learning models
- Pipeline: ML pipeline definitions and runs
- KaggleToken: Remote execution credentials
- SystemConfig: Application configuration
- AuditLog: System activity logging

Usage:
    # Initialize database
    from backend.models.database import init_database, get_db
    
    init_database()
    
    # Use in FastAPI endpoints
    @app.get("/users/")
    def get_users(db: Session = Depends(get_db)):
        return db.query(User).all()
        
    # Direct usage
    with get_db_session() as db:
        user = User(username="john", email="john@example.com")
        db.add(user)
        db.commit()
"""

import os
import logging
import warnings
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, Union
from datetime import datetime, timezone
import uuid
from enum import Enum

# SQLAlchemy imports
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Boolean, 
    Float, JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint,
    event, exc, pool
)
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, Session, relationship, scoped_session
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID
from sqlalchemy.sql import func
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, NullPool

# Pydantic for settings validation
try:
    from pydantic import BaseSettings, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Configure logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress SQLAlchemy warnings
warnings.filterwarnings('ignore', category=exc.SAWarning)

# Database configuration
class DatabaseSettings:
    """Database configuration from environment variables."""
    
    def __init__(self):
        # Database URL - supports multiple formats
        self.DATABASE_URL = os.getenv(
            'DATABASE_URL', 
            'sqlite:///./auto_analyst.db'
        )
        
        # Individual connection parameters (fallback)
        self.DB_HOST = os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = int(os.getenv('DB_PORT', '5432'))
        self.DB_NAME = os.getenv('DB_NAME', 'auto_analyst')
        self.DB_USER = os.getenv('DB_USER', 'postgres')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD', '')
        self.DB_DRIVER = os.getenv('DB_DRIVER', 'postgresql')  # postgresql, mysql, sqlite
        
        # Connection pool settings
        self.DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
        self.DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '20'))
        self.DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        self.DB_POOL_RECYCLE = int(os.getenv('DB_POOL_RECYCLE', '3600'))
        
        # SQLAlchemy settings
        self.DB_ECHO = os.getenv('DB_ECHO', 'False').lower() == 'true'
        self.DB_ECHO_POOL = os.getenv('DB_ECHO_POOL', 'False').lower() == 'true'
        
        # Application settings
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    def get_database_url(self) -> str:
        """Get the complete database URL."""
        if self.DATABASE_URL and not self.DATABASE_URL.startswith('sqlite'):
            return self.DATABASE_URL
        elif self.DATABASE_URL.startswith('sqlite'):
            return self.DATABASE_URL
        else:
            # Construct URL from individual components
            if self.DB_DRIVER == 'postgresql':
                return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            elif self.DB_DRIVER == 'mysql':
                return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            else:
                return f"sqlite:///./{self.DB_NAME}.db"

# Global settings instance
db_settings = DatabaseSettings()

# SQLAlchemy setup
Base = declarative_base()

# Database engine and session
engine: Optional[Engine] = None
SessionLocal: Optional[sessionmaker] = None
ScopedSession: Optional[scoped_session] = None

class ModelMixin:
    """Base mixin for all database models with common functionality."""
    
    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name."""
        return cls.__name__.lower() + 's'
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        comment="Record creation timestamp"
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Record last update timestamp"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, uuid.UUID):
                value = str(value)
            result[column.name] = value
        return result
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"

class UUIDMixin:
    """Mixin for models that use UUID as primary key."""
    
    id = Column(
        String(36) if db_settings.DB_DRIVER != 'postgresql' else PostgreSQL_UUID(as_uuid=True),
        primary_key=True,
        default=lambda: str(uuid.uuid4()) if db_settings.DB_DRIVER != 'postgresql' else uuid.uuid4(),
        comment="Unique identifier"
    )

# Status enums
class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class DatasetStatus(str, Enum):
    """Dataset processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"

class AnalysisStatus(str, Enum):
    """Analysis execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionMode(str, Enum):
    """Execution mode for analyses."""
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"

# Model definitions

class User(Base, ModelMixin):
    """User model for authentication and user management."""
    
    __tablename__ = "users"
    
    # Basic user information
    username = Column(
        String(50), 
        unique=True, 
        nullable=False, 
        index=True,
        comment="Unique username"
    )
    email = Column(
        String(255), 
        unique=True, 
        nullable=False, 
        index=True,
        comment="User email address"
    )
    full_name = Column(
        String(100), 
        nullable=True,
        comment="User's full name"
    )
    
    # Authentication
    hashed_password = Column(
        String(255), 
        nullable=False,
        comment="Bcrypt hashed password"
    )
    is_active = Column(
        Boolean, 
        default=True, 
        nullable=False,
        comment="Whether the account is active"
    )
    is_superuser = Column(
        Boolean, 
        default=False, 
        nullable=False,
        comment="Whether the user has admin privileges"
    )
    status = Column(
        String(20),
        default=UserStatus.ACTIVE,
        nullable=False,
        comment="User account status"
    )
    
    # Profile information
    avatar_url = Column(
        String(500),
        nullable=True,
        comment="URL to user avatar image"
    )
    bio = Column(
        Text,
        nullable=True,
        comment="User biography"
    )
    
    # Settings
    preferences = Column(
        JSON,
        nullable=True,
        comment="User preferences as JSON"
    )
    
    # Timestamps
    last_login = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last login timestamp"
    )
    email_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Email verification timestamp"
    )
    
    # Relationships
    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")
    kaggle_tokens = relationship("KaggleToken", back_populates="user", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('idx_users_email_status', 'email', 'status'),
        Index('idx_users_username_active', 'username', 'is_active'),
        CheckConstraint('length(username) >= 3', name='username_min_length'),
        CheckConstraint('length(email) >= 5', name='email_min_length'),
    )

class UserSession(Base, ModelMixin):
    """User session management."""
    
    __tablename__ = "user_sessions"
    
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User ID"
    )
    session_token = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Session token"
    )
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="Session expiration time"
    )
    ip_address = Column(
        String(45),
        nullable=True,
        comment="Client IP address"
    )
    user_agent = Column(
        Text,
        nullable=True,
        comment="Client user agent"
    )
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether session is active"
    )
    
    # Relationships
    user = relationship("User")

class Dataset(Base, ModelMixin):
    """Dataset model for uploaded data files."""
    
    __tablename__ = "datasets"
    
    # Basic information
    name = Column(
        String(255),
        nullable=False,
        comment="Dataset name"
    )
    description = Column(
        Text,
        nullable=True,
        comment="Dataset description"
    )
    
    # File information
    original_filename = Column(
        String(500),
        nullable=False,
        comment="Original filename"
    )
    file_path = Column(
        String(1000),
        nullable=False,
        comment="Storage file path"
    )
    file_size = Column(
        Integer,
        nullable=False,
        comment="File size in bytes"
    )
    file_hash = Column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA256 hash of file content"
    )
    
    # Data characteristics
    num_rows = Column(
        Integer,
        nullable=True,
        comment="Number of rows in dataset"
    )
    num_columns = Column(
        Integer,
        nullable=True,
        comment="Number of columns in dataset"
    )
    column_names = Column(
        JSON,
        nullable=True,
        comment="List of column names"
    )
    column_types = Column(
        JSON,
        nullable=True,
        comment="Data types for each column"
    )
    
    # Processing status
    status = Column(
        String(20),
        default=DatasetStatus.UPLOADED,
        nullable=False,
        comment="Processing status"
    )
    processing_error = Column(
        Text,
        nullable=True,
        comment="Error message if processing failed"
    )
    
    # Metadata
    metadata = Column(
        JSON,
        nullable=True,
        comment="Additional dataset metadata"
    )
    tags = Column(
        JSON,
        nullable=True,
        comment="Dataset tags"
    )
    
    # Data quality metrics
    data_quality_score = Column(
        Float,
        nullable=True,
        comment="Overall data quality score (0-1)"
    )
    missing_value_ratio = Column(
        Float,
        nullable=True,
        comment="Ratio of missing values"
    )
    
    # Ownership
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Dataset owner"
    )
    is_public = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether dataset is publicly accessible"
    )
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    analyses = relationship("Analysis", back_populates="dataset", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('idx_datasets_owner_status', 'owner_id', 'status'),
        Index('idx_datasets_hash', 'file_hash'),
        CheckConstraint('file_size > 0', name='positive_file_size'),
        CheckConstraint('num_rows >= 0', name='non_negative_rows'),
        CheckConstraint('num_columns >= 0', name='non_negative_columns'),
    )

class Analysis(Base, ModelMixin):
    """Analysis model for ML analysis runs."""
    
    __tablename__ = "analyses"
    
    # Basic information
    name = Column(
        String(255),
        nullable=False,
        comment="Analysis name"
    )
    description = Column(
        Text,
        nullable=True,
        comment="Analysis description"
    )
    
    # Analysis configuration
    task_type = Column(
        String(50),
        nullable=False,
        comment="Type of ML task (classification, regression, etc.)"
    )
    dataset_type = Column(
        String(50),
        nullable=True,
        comment="Type of dataset (tabular, timeseries, text, etc.)"
    )
    target_column = Column(
        String(255),
        nullable=True,
        comment="Target column name for supervised learning"
    )
    
    # Execution details
    status = Column(
        String(20),
        default=AnalysisStatus.PENDING,
        nullable=False,
        comment="Analysis execution status"
    )
    execution_mode = Column(
        String(20),
        default=ExecutionMode.LOCAL,
        nullable=False,
        comment="Where analysis was executed"
    )
    
    # Configuration
    pipeline_config = Column(
        JSON,
        nullable=True,
        comment="Pipeline configuration parameters"
    )
    model_config = Column(
        JSON,
        nullable=True,
        comment="Model-specific configuration"
    )
    
    # Results
    best_model_name = Column(
        String(100),
        nullable=True,
        comment="Name of best performing model"
    )
    performance_metrics = Column(
        JSON,
        nullable=True,
        comment="Model performance metrics"
    )
    feature_importance = Column(
        JSON,
        nullable=True,
        comment="Feature importance scores"
    )
    
    # Execution metadata
    execution_time = Column(
        Float,
        nullable=True,
        comment="Execution time in seconds"
    )
    resource_usage = Column(
        JSON,
        nullable=True,
        comment="Resource usage statistics"
    )
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if analysis failed"
    )
    
    # Timestamps
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Analysis start time"
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Analysis completion time"
    )
    
    # Relationships
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who created the analysis"
    )
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Dataset used for analysis"
    )
    
    user = relationship("User", back_populates="analyses")
    dataset = relationship("Dataset", back_populates="analyses")
    ml_models = relationship("MLModel", back_populates="analysis", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('idx_analyses_user_status', 'user_id', 'status'),
        Index('idx_analyses_dataset_task', 'dataset_id', 'task_type'),
        Index('idx_analyses_created', 'created_at'),
    )

class MLModel(Base, ModelMixin):
    """ML model storage and metadata."""
    
    __tablename__ = "ml_models"
    
    # Basic information
    name = Column(
        String(255),
        nullable=False,
        comment="Model name"
    )
    description = Column(
        Text,
        nullable=True,
        comment="Model description"
    )
    version = Column(
        String(50),
        default="1.0.0",
        nullable=False,
        comment="Model version"
    )
    
    # Model details
    model_type = Column(
        String(100),
        nullable=False,
        comment="Type of model (RandomForest, XGBoost, etc.)"
    )
    algorithm_name = Column(
        String(100),
        nullable=False,
        comment="Algorithm name"
    )
    
    # Storage
    model_path = Column(
        String(1000),
        nullable=True,
        comment="Path to serialized model file"
    )
    model_size = Column(
        Integer,
        nullable=True,
        comment="Model file size in bytes"
    )
    
    # Performance
    performance_metrics = Column(
        JSON,
        nullable=True,
        comment="Model performance metrics"
    )
    training_metrics = Column(
        JSON,
        nullable=True,
        comment="Training process metrics"
    )
    
    # Configuration
    hyperparameters = Column(
        JSON,
        nullable=True,
        comment="Model hyperparameters"
    )
    features = Column(
        JSON,
        nullable=True,
        comment="List of feature names used"
    )
    preprocessing_steps = Column(
        JSON,
        nullable=True,
        comment="Preprocessing pipeline steps"
    )
    
    # Deployment
    is_deployed = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether model is deployed"
    )
    deployment_url = Column(
        String(500),
        nullable=True,
        comment="Model deployment endpoint URL"
    )
    
    # Relationships
    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent analysis"
    )
    
    analysis = relationship("Analysis", back_populates="ml_models")
    
    # Constraints
    __table_args__ = (
        Index('idx_ml_models_analysis', 'analysis_id'),
        Index('idx_ml_models_type', 'model_type'),
        UniqueConstraint('analysis_id', 'name', name='unique_model_per_analysis'),
    )

class Pipeline(Base, ModelMixin):
    """ML pipeline definitions and runs."""
    
    __tablename__ = "pipelines"
    
    # Basic information
    name = Column(
        String(255),
        nullable=False,
        comment="Pipeline name"
    )
    description = Column(
        Text,
        nullable=True,
        comment="Pipeline description"
    )
    
    # Pipeline definition
    pipeline_config = Column(
        JSON,
        nullable=False,
        comment="Complete pipeline configuration"
    )
    steps = Column(
        JSON,
        nullable=True,
        comment="Pipeline steps definition"
    )
    
    # Execution settings
    execution_mode = Column(
        String(20),
        default=ExecutionMode.LOCAL,
        nullable=False,
        comment="Execution environment"
    )
    max_execution_time = Column(
        Integer,
        default=3600,
        nullable=False,
        comment="Maximum execution time in seconds"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether pipeline is active"
    )
    
    # Ownership
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Pipeline owner"
    )
    
    # Relationships
    user = relationship("User")
    runs = relationship("PipelineRun", back_populates="pipeline", cascade="all, delete-orphan")

class PipelineRun(Base, ModelMixin):
    """Individual pipeline execution runs."""
    
    __tablename__ = "pipeline_runs"
    
    # Run identification
    run_id = Column(
        String(36),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique run identifier"
    )
    
    # Execution details
    status = Column(
        String(20),
        default=AnalysisStatus.PENDING,
        nullable=False,
        comment="Run status"
    )
    execution_mode = Column(
        String(20),
        nullable=False,
        comment="Execution environment"
    )
    
    # Configuration
    config_override = Column(
        JSON,
        nullable=True,
        comment="Configuration overrides for this run"
    )
    
    # Results
    results = Column(
        JSON,
        nullable=True,
        comment="Run results and outputs"
    )
    logs = Column(
        Text,
        nullable=True,
        comment="Execution logs"
    )
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if run failed"
    )
    
    # Performance
    execution_time = Column(
        Float,
        nullable=True,
        comment="Total execution time in seconds"
    )
    resource_usage = Column(
        JSON,
        nullable=True,
        comment="Resource usage metrics"
    )
    
    # Timestamps
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Run start time"
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Run completion time"
    )
    
    # Relationships
    pipeline_id = Column(
        Integer,
        ForeignKey("pipelines.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent pipeline"
    )
    
    pipeline = relationship("Pipeline", back_populates="runs")

class KaggleToken(Base, ModelMixin):
    """Kaggle API token storage for remote execution."""
    
    __tablename__ = "kaggle_tokens"
    
    # Token data (encrypted)
    encrypted_token = Column(
        Text,
        nullable=False,
        comment="Encrypted Kaggle API token"
    )
    username = Column(
        String(100),
        nullable=False,
        comment="Kaggle username"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether token is active"
    )
    last_verified = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last token verification time"
    )
    
    # Relationships
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Token owner"
    )
    
    user = relationship("User", back_populates="kaggle_tokens")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'username', name='unique_kaggle_token_per_user'),
    )

class RemoteExecution(Base, ModelMixin):
    """Remote execution tracking (Kaggle/Colab)."""
    
    __tablename__ = "remote_executions"
    
    # Execution details
    execution_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Remote execution identifier"
    )
    platform = Column(
        String(20),
        nullable=False,
        comment="Execution platform (kaggle, colab)"
    )
    status = Column(
        String(20),
        default=AnalysisStatus.PENDING,
        nullable=False,
        comment="Execution status"
    )
    
    # Configuration
    config = Column(
        JSON,
        nullable=True,
        comment="Execution configuration"
    )
    dataset_url = Column(
        String(500),
        nullable=True,
        comment="Remote dataset URL"
    )
    notebook_url = Column(
        String(500),
        nullable=True,
        comment="Remote notebook URL"
    )
    
    # Results
    results_url = Column(
        String(500),
        nullable=True,
        comment="Results download URL"
    )
    logs = Column(
        Text,
        nullable=True,
        comment="Execution logs"
    )
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if execution failed"
    )
    
    # Timestamps
    submitted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Submission time"
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Completion time"
    )
    
    # Relationships
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who submitted the execution"
    )
    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Related analysis"
    )
    
    user = relationship("User")
    analysis = relationship("Analysis")

class SystemConfig(Base, ModelMixin):
    """System configuration and settings."""
    
    __tablename__ = "system_config"
    
    key = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Configuration key"
    )
    value = Column(
        Text,
        nullable=True,
        comment="Configuration value"
    )
    value_type = Column(
        String(20),
        default="string",
        nullable=False,
        comment="Type of value (string, int, float, bool, json)"
    )
    description = Column(
        Text,
        nullable=True,
        comment="Configuration description"
    )
    is_secret = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether value should be treated as secret"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint("value_type IN ('string', 'int', 'float', 'bool', 'json')", name='valid_value_type'),
    )

class AuditLog(Base, ModelMixin):
    """Audit log for tracking system activities."""
    
    __tablename__ = "audit_logs"
    
    # Event details
    event_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of event"
    )
    event_description = Column(
        Text,
        nullable=True,
        comment="Event description"
    )
    
    # Context
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who performed the action"
    )
    resource_type = Column(
        String(50),
        nullable=True,
        comment="Type of resource affected"
    )
    resource_id = Column(
        String(50),
        nullable=True,
        comment="ID of resource affected"
    )
    
    # Request details
    ip_address = Column(
        String(45),
        nullable=True,
        comment="Client IP address"
    )
    user_agent = Column(
        Text,
        nullable=True,
        comment="Client user agent"
    )
    
    # Additional data
    metadata = Column(
        JSON,
        nullable=True,
        comment="Additional event metadata"
    )
    
    # Relationships
    user = relationship("User")
    
    # Constraints
    __table_args__ = (
        Index('idx_audit_logs_event_type', 'event_type'),
        Index('idx_audit_logs_user_created', 'user_id', 'created_at'),
        Index('idx_audit_logs_resource', 'resource_type', 'resource_id'),
    )

class ErrorLog(Base, ModelMixin):
    """Error logging for debugging and monitoring."""
    
    __tablename__ = "error_logs"
    
    # Error details
    error_type = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Type of error"
    )
    error_message = Column(
        Text,
        nullable=False,
        comment="Error message"
    )
    stack_trace = Column(
        Text,
        nullable=True,
        comment="Stack trace"
    )
    
    # Context
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User associated with the error"
    )
    request_path = Column(
        String(500),
        nullable=True,
        comment="Request path where error occurred"
    )
    request_method = Column(
        String(10),
        nullable=True,
        comment="HTTP method"
    )
    
    # Additional context
    metadata = Column(
        JSON,
        nullable=True,
        comment="Additional error context"
    )
    
    # Resolution
    is_resolved = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether error has been resolved"
    )
    resolved_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Resolution timestamp"
    )
    
    # Relationships
    user = relationship("User")
    
    # Constraints
    __table_args__ = (
        Index('idx_error_logs_type_created', 'error_type', 'created_at'),
        Index('idx_error_logs_resolved', 'is_resolved'),
    )

# Database engine and session management

def create_database_engine() -> Engine:
    """
    Create and configure the database engine.
    
    Returns:
        Configured SQLAlchemy engine
    """
    database_url = db_settings.get_database_url()
    
    # Engine configuration based on database type
    engine_kwargs = {
        'echo': db_settings.DB_ECHO,
        'echo_pool': db_settings.DB_ECHO_POOL,
        'future': True,  # Use SQLAlchemy 2.0 style
    }
    
    # Configure connection pooling
    if database_url.startswith('sqlite'):
        # SQLite-specific configuration
        engine_kwargs.update({
            'poolclass': NullPool,
            'connect_args': {
                'check_same_thread': False,
                'timeout': 30
            }
        })
    else:
        # PostgreSQL/MySQL configuration
        engine_kwargs.update({
            'poolclass': QueuePool,
            'pool_size': db_settings.DB_POOL_SIZE,
            'max_overflow': db_settings.DB_MAX_OVERFLOW,
            'pool_timeout': db_settings.DB_POOL_TIMEOUT,
            'pool_recycle': db_settings.DB_POOL_RECYCLE,
            'pool_pre_ping': True  # Validate connections
        })
    
    try:
        engine = create_engine(database_url, **engine_kwargs)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        logger.info(f"Database engine created successfully: {database_url.split('@')[-1] if '@' in database_url else database_url}")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise

def create_session_factory(engine: Engine) -> sessionmaker:
    """
    Create session factory with proper configuration.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        Configured sessionmaker
    """
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )

def init_database() -> None:
    """Initialize database engine and session factory."""
    global engine, SessionLocal, ScopedSession
    
    try:
        engine = create_database_engine()
        SessionLocal = create_session_factory(engine)
        ScopedSession = scoped_session(SessionLocal)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def create_tables() -> None:
    """Create all database tables."""
    if engine is None:
        raise RuntimeError("Database engine not initialized. Call init_database() first.")
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

def drop_tables() -> None:
    """Drop all database tables (use with caution)."""
    if engine is None:
        raise RuntimeError("Database engine not initialized. Call init_database() first.")
    
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
        
    except Exception as e:
        logger.error(f"Failed to drop database tables: {str(e)}")
        raise

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Yields:
        Database session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        session.close()

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Yields:
        Database session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def get_scoped_session() -> Session:
    """
    Get a scoped session for thread-safe operations.
    
    Returns:
        Scoped database session
    """
    if ScopedSession is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return ScopedSession()

def close_scoped_session() -> None:
    """Close and remove the scoped session."""
    if ScopedSession is not None:
        ScopedSession.remove()

def health_check() -> Dict[str, Any]:
    """
    Perform database health check.
    
    Returns:
        Health check results
    """
    try:
        with get_db_session() as db:
            # Simple query to test connection
            result = db.execute("SELECT 1 as health_check")
            row = result.fetchone()
            
            if row and row[0] == 1:
                return {
                    'status': 'healthy',
                    'database': 'connected',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'database': 'query_failed',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
    except Exception as e:
        return {
            'status': 'unhealthy',
            'database': 'connection_failed',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Event listeners for automatic behavior

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance."""
    if db_settings.get_database_url().startswith('sqlite'):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

@event.listens_for(User, 'before_insert')
def create_user_uuid(mapper, connection, target):
    """Generate UUID for new users if not provided."""
    if not target.id and hasattr(target, 'id'):
        target.id = str(uuid.uuid4())

# Utility functions

def get_database_info() -> Dict[str, Any]:
    """
    Get database configuration and status information.
    
    Returns:
        Database information dictionary
    """
    database_url = db_settings.get_database_url()
    
    # Hide sensitive information
    safe_url = database_url
    if '@' in safe_url:
        parts = safe_url.split('@')
        safe_url = f"{parts[0].split('://')[0]}://***:***@{parts[1]}"
    
    return {
        'database_url': safe_url,
        'driver': db_settings.DB_DRIVER,
        'pool_size': db_settings.DB_POOL_SIZE,
        'max_overflow': db_settings.DB_MAX_OVERFLOW,
        'echo': db_settings.DB_ECHO,
        'engine_initialized': engine is not None,
        'session_factory_initialized': SessionLocal is not None,
        'environment': db_settings.ENVIRONMENT
    }

# Export public API
__all__ = [
    # Base classes
    'Base', 'ModelMixin', 'UUIDMixin',
    
    # Models
    'User', 'UserSession', 'Dataset', 'Analysis', 'MLModel',
    'Pipeline', 'PipelineRun', 'KaggleToken', 'RemoteExecution',
    'SystemConfig', 'AuditLog', 'ErrorLog',
    
    # Enums
    'UserStatus', 'DatasetStatus', 'AnalysisStatus', 'ExecutionMode',
    
    # Database functions
    'init_database', 'create_tables', 'drop_tables',
    'get_db', 'get_db_session', 'get_scoped_session', 'close_scoped_session',
    'health_check', 'get_database_info',
    
    # Configuration
    'db_settings', 'DatabaseSettings'
]
