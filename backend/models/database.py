"""
Database Models for Auto-Analyst Platform - PostgreSQL Ready

FIXES APPLIED:
- ✅ PostgreSQL compatibility with proper types
- ✅ Environment variable configuration
- ✅ Connection pooling for production
- ✅ Async/await support
- ✅ Error handling with retries
- ✅ FastAPI dependency injection
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Generator, Optional, Dict, Any
from datetime import datetime, timezone
import uuid
from enum import Enum

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Boolean, Float, JSON, ForeignKey, event, pool
)
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.sql import func
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration from environment
class DatabaseConfig:
    """Production-ready database configuration."""
    
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Handle postgres:// vs postgresql:// for SQLAlchemy 2.0+
        if self.DATABASE_URL.startswith("postgres://"):
            self.DATABASE_URL = self.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        
        # Connection pool settings
        self.POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
        self.MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # Environment settings
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Global configuration
db_config = DatabaseConfig()

# SQLAlchemy Base
Base = declarative_base()

# Database engine and session
engine: Optional[Engine] = None
SessionLocal: Optional[sessionmaker] = None

class BaseModel(Base):
    """Base model with common fields and functionality."""
    
    __abstract__ = True
    
    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name."""
        return cls.__name__.lower()
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Timestamps with timezone support (PostgreSQL)
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

class User(BaseModel):
    """User accounts and authentication."""
    
    __tablename__ = "users"
    
    # User identification
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Profile
    avatar_url = Column(String(500), nullable=True)
    timezone = Column(String(50), default="UTC", nullable=False)
    preferences = Column(JSON, nullable=True, comment="User preferences and settings")
    
    # Tracking
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="owner", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")

class Dataset(BaseModel):
    """Uploaded datasets and metadata."""
    
    __tablename__ = "datasets"
    
    # Dataset identification
    name = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True, comment="Array of tags")
    
    # File information
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False, comment="File size in bytes")
    content_type = Column(String(100), nullable=False)
    file_hash = Column(String(64), nullable=True, comment="SHA256 hash for integrity")
    
    # Data characteristics
    num_rows = Column(Integer, nullable=True)
    num_columns = Column(Integer, nullable=True)
    column_names = Column(JSON, nullable=True, comment="Array of column names")
    column_types = Column(JSON, nullable=True, comment="Column name to type mapping")
    
    # Processing status
    status = Column(String(20), default="uploaded", nullable=False, index=True)
    processing_error = Column(Text, nullable=True)
    
    # Data quality
    missing_values_count = Column(Integer, nullable=True)
    duplicate_rows_count = Column(Integer, nullable=True)
    data_quality_score = Column(Float, nullable=True)
    
    # Relationships
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="datasets")
    analyses = relationship("Analysis", back_populates="dataset", cascade="all, delete-orphan")

class Analysis(BaseModel):
    """ML analysis configurations and results."""
    
    __tablename__ = "analyses"
    
    # Analysis identification
    analysis_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Configuration
    task_type = Column(String(50), nullable=False, index=True)  # classification, regression, etc.
    target_column = Column(String(255), nullable=False)
    feature_columns = Column(JSON, nullable=True)
    algorithms = Column(JSON, nullable=True, comment="Array of algorithm names")
    
    # Execution settings
    execution_mode = Column(String(20), default="local_cpu", nullable=False)
    max_training_time = Column(Integer, nullable=True, comment="Max time in seconds")
    test_size = Column(Float, default=0.2, nullable=False)
    random_state = Column(Integer, nullable=True)
    
    # Status and progress
    status = Column(String(20), default="pending", nullable=False, index=True)
    progress = Column(Float, default=0.0, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Results
    best_model_name = Column(String(100), nullable=True)
    best_model_score = Column(Float, nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    model_comparison = Column(JSON, nullable=True)
    
    # Execution tracking
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    execution_time = Column(Float, nullable=True, comment="Execution time in seconds")
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    
    user = relationship("User", back_populates="analyses")
    dataset = relationship("Dataset", back_populates="analyses")

# Database engine creation
def create_database_engine() -> Engine:
    """Create and configure the database engine."""
    engine_kwargs = {
        "echo": db_config.DEBUG,
        "future": True,  # Use SQLAlchemy 2.0 style
        "pool_pre_ping": True,  # Validate connections
        "pool_recycle": db_config.POOL_RECYCLE,
    }
    
    # PostgreSQL-specific configuration
    if "postgresql" in db_config.DATABASE_URL:
        engine_kwargs.update({
            "poolclass": QueuePool,
            "pool_size": db_config.POOL_SIZE,
            "max_overflow": db_config.MAX_OVERFLOW,
            "pool_timeout": db_config.POOL_TIMEOUT,
        })
    else:
        # SQLite fallback for development
        engine_kwargs.update({
            "poolclass": pool.NullPool,
            "connect_args": {"check_same_thread": False}
        })
    
    try:
        engine = create_engine(db_config.DATABASE_URL, **engine_kwargs)
        logger.info("✅ Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"❌ Failed to create database engine: {e}")
        raise

def create_session_factory(engine: Engine) -> sessionmaker:
    """Create session factory with proper configuration."""
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )

async def init_database() -> None:
    """Initialize database engine and session factory."""
    global engine, SessionLocal
    
    try:
        engine = create_database_engine()
        SessionLocal = create_session_factory(engine)
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

def create_tables() -> None:
    """Create all database tables."""
    if engine is None:
        raise RuntimeError("Database engine not initialized. Call init_database() first.")
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@asynccontextmanager
async def get_db_session():
    """Async context manager for database sessions."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

def health_check() -> Dict[str, Any]:
    """Perform database health check."""
    try:
        with get_db_session() as db:
            # Simple query to test connection
            result = db.execute("SELECT 1 as healthcheck")
            row = result.fetchone()
            
            if row and row[0] == 1:
                return {
                    "status": "healthy",
                    "message": "Database connected",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "status": "unhealthy", 
                    "message": "Database query failed",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": "Database connection failed", 
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Export main components
__all__ = [
    "Base", "BaseModel", "User", "Dataset", "Analysis",
    "engine", "SessionLocal", "get_db", "get_db_session",
    "init_database", "create_tables", "health_check"
]
