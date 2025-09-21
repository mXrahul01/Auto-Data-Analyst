"""
Pydantic Schemas for Auto-Analyst Platform

This module defines Pydantic models for request/response validation and serialization
in the Auto-Analyst zero-code AI-powered data analysis web application.

The schemas ensure data integrity, type validation, and API contract enforcement
across all FastAPI endpoints and backend services.

Schema Categories:
- User Management: Authentication, profiles, sessions
- Dataset Management: File uploads, metadata, processing
- ML Analytics: Analysis requests/responses, model metadata
- Pipeline Management: Pipeline configurations and runs
- Remote Execution: Kaggle/Colab integration schemas
- System Operations: Configuration, auditing, monitoring
- Common Utilities: Responses, pagination, validation

Features:
- Comprehensive field validation with constraints
- Custom validators for business logic
- Request/response schema variants
- Nested schema relationships
- Enum validation for status fields
- Date/time handling with timezone awareness
- File upload validation
- Security-focused field handling (password masking, etc.)
- Extensible design for future model additions

Usage:
    from backend.models.schemas import UserCreate, DatasetResponse
    
    # In FastAPI endpoints
    @app.post("/users/", response_model=UserResponse)
    def create_user(user: UserCreate, db: Session = Depends(get_db)):
        # Pydantic automatically validates the request
        return create_user_in_db(db, user)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
import re
from pathlib import Path

from pydantic import (
    BaseModel, Field, validator, root_validator, EmailStr,
    constr, conint, confloat, conlist, HttpUrl, Json
)
from pydantic.types import UUID4, SecretStr

# Configuration
class SchemaConfig:
    """Base configuration for all Pydantic schemas."""
    
    # General settings
    validate_assignment = True
    use_enum_values = True
    allow_population_by_field_name = True
    json_encoders = {
        datetime: lambda v: v.isoformat() if v else None,
        UUID4: lambda v: str(v) if v else None
    }
    
    # Schema generation
    schema_extra = {
        "example": "See specific schema examples"
    }

# Custom field types and validators
Username = constr(
    min_length=3, 
    max_length=50, 
    regex=r'^[a-zA-Z0-9_-]+$',
    description="Username with alphanumeric characters, underscores, and hyphens only"
)

Password = constr(
    min_length=8,
    max_length=128,
    description="Password must be at least 8 characters long"
)

DatasetName = constr(
    min_length=1,
    max_length=255,
    regex=r'^[a-zA-Z0-9\s\-_.()]+$',
    description="Dataset name with alphanumeric characters, spaces, and common symbols"
)

ModelName = constr(
    min_length=1,
    max_length=255,
    regex=r'^[a-zA-Z0-9\s\-_.()]+$',
    description="Model name with alphanumeric characters, spaces, and common symbols"
)

# Enums for status fields
class UserStatusEnum(str, Enum):
    """User account status options."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class DatasetStatusEnum(str, Enum):
    """Dataset processing status options."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"

class AnalysisStatusEnum(str, Enum):
    """Analysis execution status options."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionModeEnum(str, Enum):
    """Execution mode options."""
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"

class TaskTypeEnum(str, Enum):
    """ML task type options."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"

class DatasetTypeEnum(str, Enum):
    """Dataset type options."""
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"

# Base schema classes
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config(SchemaConfig):
        pass

class TimestampMixin(BaseSchema):
    """Mixin for schemas with timestamp fields."""
    
    created_at: datetime = Field(
        ...,
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        ...,
        description="Record last update timestamp"
    )

class IDMixin(BaseSchema):
    """Mixin for schemas with ID fields."""
    
    id: int = Field(
        ...,
        ge=1,
        description="Unique record identifier"
    )

# User schemas
class UserBase(BaseSchema):
    """Base user schema with common fields."""
    
    username: Username = Field(
        ...,
        description="Unique username for the account"
    )
    email: EmailStr = Field(
        ...,
        description="User's email address"
    )
    full_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's full name"
    )
    bio: Optional[str] = Field(
        None,
        max_length=1000,
        description="User biography or description"
    )
    avatar_url: Optional[HttpUrl] = Field(
        None,
        description="URL to user's avatar image"
    )
    
    class Config(SchemaConfig):
        schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "bio": "Data scientist and ML enthusiast",
                "avatar_url": "https://example.com/avatar.jpg"
            }
        }

class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    password: Password = Field(
        ...,
        description="User's password (minimum 8 characters)"
    )
    
    @validator('email')
    def email_must_be_valid(cls, v):
        """Validate email format."""
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseSchema):
    """Schema for updating user information."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=1000)
    avatar_url: Optional[HttpUrl] = None
    preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="User preferences as JSON object"
    )
    
    @validator('email')
    def email_must_be_valid(cls, v):
        """Validate email format if provided."""
        if v is not None:
            return v.lower()
        return v

class UserResponse(UserBase, IDMixin, TimestampMixin):
    """Schema for user responses."""
    
    status: UserStatusEnum = Field(
        ...,
        description="Current user account status"
    )
    is_active: bool = Field(
        ...,
        description="Whether the user account is active"
    )
    is_superuser: bool = Field(
        ...,
        description="Whether the user has administrative privileges"
    )
    last_login: Optional[datetime] = Field(
        None,
        description="Last login timestamp"
    )
    email_verified_at: Optional[datetime] = Field(
        None,
        description="Email verification timestamp"
    )
    preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="User preferences"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

class UserLogin(BaseSchema):
    """Schema for user login requests."""
    
    username: str = Field(
        ...,
        min_length=1,
        description="Username or email address"
    )
    password: str = Field(
        ...,
        min_length=1,
        description="User password"
    )
    remember_me: bool = Field(
        False,
        description="Whether to extend session duration"
    )

class UserSessionResponse(BaseSchema, IDMixin, TimestampMixin):
    """Schema for user session responses."""
    
    session_token: str = Field(
        ...,
        description="Session authentication token"
    )
    expires_at: datetime = Field(
        ...,
        description="Session expiration time"
    )
    is_active: bool = Field(
        ...,
        description="Whether session is currently active"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

# Dataset schemas
class DatasetBase(BaseSchema):
    """Base dataset schema with common fields."""
    
    name: DatasetName = Field(
        ...,
        description="Dataset name"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Dataset description"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Dataset tags for categorization"
    )
    is_public: bool = Field(
        False,
        description="Whether dataset is publicly accessible"
    )

class DatasetCreate(DatasetBase):
    """Schema for dataset creation."""
    
    pass

class DatasetUpload(BaseSchema):
    """Schema for dataset file upload."""
    
    name: DatasetName = Field(
        ...,
        description="Dataset name"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Dataset description"
    )
    # File will be handled separately in FastAPI endpoint
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        """Ensure name is not empty after stripping."""
        if not v.strip():
            raise ValueError('Dataset name cannot be empty')
        return v.strip()

class DatasetUpdate(BaseSchema):
    """Schema for updating dataset information."""
    
    name: Optional[DatasetName] = None
    description: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None

class DatasetResponse(DatasetBase, IDMixin, TimestampMixin):
    """Schema for dataset responses."""
    
    original_filename: str = Field(
        ...,
        description="Original uploaded filename"
    )
    file_size: int = Field(
        ...,
        ge=0,
        description="File size in bytes"
    )
    file_hash: Optional[str] = Field(
        None,
        description="SHA256 hash of file content"
    )
    
    # Data characteristics
    num_rows: Optional[int] = Field(
        None,
        ge=0,
        description="Number of rows in dataset"
    )
    num_columns: Optional[int] = Field(
        None,
        ge=0,
        description="Number of columns in dataset"
    )
    column_names: Optional[List[str]] = Field(
        None,
        description="List of column names"
    )
    column_types: Optional[Dict[str, str]] = Field(
        None,
        description="Data types for each column"
    )
    
    # Processing status
    status: DatasetStatusEnum = Field(
        ...,
        description="Dataset processing status"
    )
    processing_error: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )
    
    # Data quality metrics
    data_quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall data quality score (0-1)"
    )
    missing_value_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Ratio of missing values in dataset"
    )
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional dataset metadata"
    )
    
    # Owner information
    owner_id: int = Field(
        ...,
        description="ID of the dataset owner"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

class DatasetSummary(BaseSchema, IDMixin):
    """Lightweight dataset summary for listings."""
    
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    status: DatasetStatusEnum = Field(..., description="Processing status")
    num_rows: Optional[int] = Field(None, description="Number of rows")
    num_columns: Optional[int] = Field(None, description="Number of columns")
    file_size: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config(SchemaConfig):
        orm_mode = True

# Analysis schemas
class AnalysisBase(BaseSchema):
    """Base analysis schema."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Analysis name"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Analysis description"
    )
    task_type: TaskTypeEnum = Field(
        ...,
        description="Type of ML task to perform"
    )
    dataset_type: Optional[DatasetTypeEnum] = Field(
        None,
        description="Type of dataset being analyzed"
    )
    target_column: Optional[str] = Field(
        None,
        max_length=255,
        description="Target column name for supervised learning"
    )

class AnalysisRequest(AnalysisBase):
    """Schema for analysis creation requests."""
    
    dataset_id: int = Field(
        ...,
        ge=1,
        description="ID of the dataset to analyze"
    )
    execution_mode: ExecutionModeEnum = Field(
        ExecutionModeEnum.LOCAL,
        description="Execution environment for the analysis"
    )
    
    # Configuration options
    pipeline_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Pipeline configuration parameters"
    )
    model_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Model-specific configuration options"
    )
    
    @validator('pipeline_config')
    def validate_pipeline_config(cls, v):
        """Validate pipeline configuration."""
        if v is not None:
            # Add specific validation rules for pipeline config
            if 'max_execution_time' in v and v['max_execution_time'] <= 0:
                raise ValueError('max_execution_time must be positive')
        return v

class AnalysisUpdate(BaseSchema):
    """Schema for updating analysis information."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    
class AnalysisResponse(AnalysisBase, IDMixin, TimestampMixin):
    """Schema for analysis responses."""
    
    dataset_id: int = Field(
        ...,
        description="ID of the analyzed dataset"
    )
    user_id: int = Field(
        ...,
        description="ID of the user who created the analysis"
    )
    
    # Execution details
    status: AnalysisStatusEnum = Field(
        ...,
        description="Current analysis status"
    )
    execution_mode: ExecutionModeEnum = Field(
        ...,
        description="Execution environment used"
    )
    
    # Results
    best_model_name: Optional[str] = Field(
        None,
        description="Name of the best performing model"
    )
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Model performance metrics"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None,
        description="Feature importance scores"
    )
    
    # Configuration
    pipeline_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Pipeline configuration used"
    )
    model_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Model configuration used"
    )
    
    # Execution metadata
    execution_time: Optional[float] = Field(
        None,
        ge=0,
        description="Total execution time in seconds"
    )
    resource_usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Resource usage statistics"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if analysis failed"
    )
    
    # Timestamps
    started_at: Optional[datetime] = Field(
        None,
        description="Analysis start time"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Analysis completion time"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

class AnalysisSummary(BaseSchema, IDMixin):
    """Lightweight analysis summary for listings."""
    
    name: str = Field(..., description="Analysis name")
    task_type: TaskTypeEnum = Field(..., description="ML task type")
    status: AnalysisStatusEnum = Field(..., description="Analysis status")
    best_model_name: Optional[str] = Field(None, description="Best model name")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config(SchemaConfig):
        orm_mode = True

# ML Model schemas
class MLModelBase(BaseSchema):
    """Base ML model schema."""
    
    name: ModelName = Field(
        ...,
        description="Model name"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Model description"
    )
    version: str = Field(
        "1.0.0",
        max_length=50,
        description="Model version"
    )
    model_type: str = Field(
        ...,
        max_length=100,
        description="Type of model (e.g., RandomForest, XGBoost)"
    )
    algorithm_name: str = Field(
        ...,
        max_length=100,
        description="Algorithm name"
    )

class MLModelResponse(MLModelBase, IDMixin, TimestampMixin):
    """Schema for ML model responses."""
    
    analysis_id: int = Field(
        ...,
        description="ID of the parent analysis"
    )
    
    # Storage information
    model_size: Optional[int] = Field(
        None,
        ge=0,
        description="Model file size in bytes"
    )
    
    # Performance metrics
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Model performance metrics"
    )
    training_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Training process metrics"
    )
    
    # Configuration
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Model hyperparameters"
    )
    features: Optional[List[str]] = Field(
        None,
        description="List of feature names used"
    )
    preprocessing_steps: Optional[List[str]] = Field(
        None,
        description="Preprocessing pipeline steps"
    )
    
    # Deployment status
    is_deployed: bool = Field(
        False,
        description="Whether model is currently deployed"
    )
    deployment_url: Optional[HttpUrl] = Field(
        None,
        description="Model deployment endpoint URL"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

class ModelPredictionRequest(BaseSchema):
    """Schema for model prediction requests."""
    
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ...,
        description="Input data for prediction (single record or batch)"
    )
    return_probabilities: bool = Field(
        False,
        description="Whether to return prediction probabilities"
    )
    explain_predictions: bool = Field(
        False,
        description="Whether to include prediction explanations"
    )

class ModelPredictionResponse(BaseSchema):
    """Schema for model prediction responses."""
    
    predictions: Union[Any, List[Any]] = Field(
        ...,
        description="Model predictions"
    )
    probabilities: Optional[Union[List[float], List[List[float]]]] = Field(
        None,
        description="Prediction probabilities (if requested)"
    )
    explanations: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        None,
        description="Prediction explanations (if requested)"
    )
    model_info: Dict[str, str] = Field(
        ...,
        description="Information about the model used"
    )
    
    class Config(SchemaConfig):
        schema_extra = {
            "example": {
                "predictions": [0.85, 0.92],
                "probabilities": [[0.15, 0.85], [0.08, 0.92]],
                "model_info": {
                    "model_name": "RandomForest_v1",
                    "model_type": "RandomForestClassifier"
                }
            }
        }

class ModelTrainingRequest(BaseSchema):
    """Schema for model training requests."""
    
    dataset_id: int = Field(
        ...,
        ge=1,
        description="ID of the dataset to train on"
    )
    model_type: str = Field(
        ...,
        description="Type of model to train"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom hyperparameters"
    )
    cross_validation_folds: int = Field(
        5,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )
    test_size: float = Field(
        0.2,
        gt=0.0,
        lt=1.0,
        description="Proportion of dataset to use for testing"
    )

# Pipeline schemas
class PipelineBase(BaseSchema):
    """Base pipeline schema."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Pipeline name"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Pipeline description"
    )

class PipelineConfigSchema(BaseSchema):
    """Schema for pipeline configuration."""
    
    execution_mode: ExecutionModeEnum = Field(
        ExecutionModeEnum.LOCAL,
        description="Execution environment"
    )
    max_execution_time: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Maximum execution time in seconds"
    )
    enable_parallel: bool = Field(
        True,
        description="Whether to enable parallel processing"
    )
    optimization_budget: int = Field(
        100,
        ge=10,
        le=1000,
        description="Number of optimization trials"
    )
    enable_ensemble: bool = Field(
        True,
        description="Whether to enable ensemble methods"
    )
    generate_explanations: bool = Field(
        True,
        description="Whether to generate model explanations"
    )

class PipelineCreate(PipelineBase):
    """Schema for creating pipelines."""
    
    pipeline_config: PipelineConfigSchema = Field(
        ...,
        description="Pipeline configuration"
    )
    steps: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Pipeline steps definition"
    )

class PipelineResponse(PipelineBase, IDMixin, TimestampMixin):
    """Schema for pipeline responses."""
    
    user_id: int = Field(
        ...,
        description="ID of the pipeline owner"
    )
    pipeline_config: Dict[str, Any] = Field(
        ...,
        description="Pipeline configuration"
    )
    steps: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Pipeline steps"
    )
    execution_mode: ExecutionModeEnum = Field(
        ...,
        description="Execution environment"
    )
    is_active: bool = Field(
        ...,
        description="Whether pipeline is active"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

class PipelineRunRequest(BaseSchema):
    """Schema for pipeline run requests."""
    
    pipeline_id: int = Field(
        ...,
        ge=1,
        description="ID of the pipeline to run"
    )
    dataset_id: int = Field(
        ...,
        ge=1,
        description="ID of the dataset to process"
    )
    config_override: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration overrides for this run"
    )

class PipelineRunResponse(BaseSchema, IDMixin, TimestampMixin):
    """Schema for pipeline run responses."""
    
    run_id: str = Field(
        ...,
        description="Unique run identifier"
    )
    pipeline_id: int = Field(
        ...,
        description="ID of the executed pipeline"
    )
    status: AnalysisStatusEnum = Field(
        ...,
        description="Run status"
    )
    execution_mode: ExecutionModeEnum = Field(
        ...,
        description="Execution environment"
    )
    
    # Results
    results: Optional[Dict[str, Any]] = Field(
        None,
        description="Run results"
    )
    logs: Optional[str] = Field(
        None,
        description="Execution logs"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if run failed"
    )
    
    # Performance
    execution_time: Optional[float] = Field(
        None,
        ge=0,
        description="Execution time in seconds"
    )
    resource_usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Resource usage metrics"
    )
    
    # Timestamps
    started_at: Optional[datetime] = Field(
        None,
        description="Run start time"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Run completion time"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

# Kaggle Integration schemas
class KaggleTokenSchema(BaseSchema):
    """Schema for Kaggle token information (without sensitive data)."""
    
    username: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Kaggle username"
    )
    is_active: bool = Field(
        ...,
        description="Whether token is active"
    )
    last_verified: Optional[datetime] = Field(
        None,
        description="Last verification timestamp"
    )

class KaggleConnectionRequest(BaseSchema):
    """Schema for Kaggle connection requests."""
    
    username: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Kaggle username"
    )
    key: SecretStr = Field(
        ...,
        description="Kaggle API key (will be encrypted)"
    )
    
    @validator('username')
    def username_alphanumeric(cls, v):
        """Validate username format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must contain only alphanumeric characters, underscores, and hyphens')
        return v

class RemoteExecutionRequest(BaseSchema):
    """Schema for remote execution requests."""
    
    analysis_id: int = Field(
        ...,
        ge=1,
        description="ID of the analysis to execute remotely"
    )
    platform: Literal["kaggle", "colab"] = Field(
        ...,
        description="Remote execution platform"
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Platform-specific configuration"
    )

class RemoteExecutionResponse(BaseSchema, IDMixin, TimestampMixin):
    """Schema for remote execution responses."""
    
    execution_id: str = Field(
        ...,
        description="Remote execution identifier"
    )
    platform: str = Field(
        ...,
        description="Execution platform"
    )
    status: AnalysisStatusEnum = Field(
        ...,
        description="Execution status"
    )
    
    # URLs
    dataset_url: Optional[HttpUrl] = Field(
        None,
        description="Remote dataset URL"
    )
    notebook_url: Optional[HttpUrl] = Field(
        None,
        description="Remote notebook URL"
    )
    results_url: Optional[HttpUrl] = Field(
        None,
        description="Results download URL"
    )
    
    # Execution details
    logs: Optional[str] = Field(
        None,
        description="Execution logs"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if execution failed"
    )
    
    # Timestamps
    submitted_at: Optional[datetime] = Field(
        None,
        description="Submission timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )
    
    class Config(SchemaConfig):
        orm_mode = True

# Dashboard and Visualization schemas
class ChartDataSchema(BaseSchema):
    """Schema for chart data."""
    
    chart_type: Literal["bar", "line", "pie", "scatter", "histogram", "box"] = Field(
        ...,
        description="Type of chart"
    )
    title: str = Field(
        ...,
        max_length=200,
        description="Chart title"
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Chart data structure"
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Chart configuration options"
    )

class MetricsResponseSchema(BaseSchema):
    """Schema for metrics responses."""
    
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    roc_auc: Optional[float] = Field(None, ge=0.0, le=1.0)
    rmse: Optional[float] = Field(None, ge=0.0)
    mae: Optional[float] = Field(None, ge=0.0)
    r2_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    class Config(SchemaConfig):
        schema_extra = {
            "example": {
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.94,
                "f1_score": 0.935,
                "roc_auc": 0.97
            }
        }

class DashboardDataSchema(BaseSchema):
    """Schema for dashboard data."""
    
    overview: Dict[str, Any] = Field(
        ...,
        description="Overview statistics"
    )
    performance_metrics: MetricsResponseSchema = Field(
        ...,
        description="Model performance metrics"
    )
    charts: List[ChartDataSchema] = Field(
        ...,
        description="Visualization charts"
    )
    insights: List[str] = Field(
        ...,
        description="Generated insights"
    )
    recommendations: List[str] = Field(
        ...,
        description="Actionable recommendations"
    )

# Common response schemas
class StatusResponse(BaseSchema):
    """Generic status response schema."""
    
    status: Literal["success", "error", "pending", "processing"] = Field(
        ...,
        description="Operation status"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional data"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    
    class Config(SchemaConfig):
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "data": {"result_id": 123},
                "timestamp": "2025-09-21T08:51:00Z"
            }
        }

class ErrorResponse(BaseSchema):
    """Error response schema."""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for tracking"
    )
    
    class Config(SchemaConfig):
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data provided",
                "details": {"field": "email", "issue": "Invalid format"},
                "timestamp": "2025-09-21T08:51:00Z",
                "request_id": "req_123456"
            }
        }

class MessageResponse(BaseSchema):
    """Simple message response schema."""
    
    message: str = Field(
        ...,
        description="Response message"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )

# Pagination schemas
class PaginationParams(BaseSchema):
    """Schema for pagination parameters."""
    
    page: int = Field(
        1,
        ge=1,
        description="Page number (1-based)"
    )
    size: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of items per page"
    )
    sort_by: Optional[str] = Field(
        None,
        description="Field to sort by"
    )
    sort_order: Literal["asc", "desc"] = Field(
        "asc",
        description="Sort order"
    )

class PaginatedResponse(BaseSchema):
    """Generic paginated response schema."""
    
    items: List[Any] = Field(
        ...,
        description="List of items for current page"
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of items"
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page number"
    )
    size: int = Field(
        ...,
        ge=1,
        description="Items per page"
    )
    pages: int = Field(
        ...,
        ge=1,
        description="Total number of pages"
    )
    has_next: bool = Field(
        ...,
        description="Whether there's a next page"
    )
    has_previous: bool = Field(
        ...,
        description="Whether there's a previous page"
    )

# File upload schemas
class FileUploadResponse(BaseSchema):
    """Schema for file upload responses."""
    
    filename: str = Field(
        ...,
        description="Original filename"
    )
    file_path: str = Field(
        ...,
        description="Server file path"
    )
    file_size: int = Field(
        ...,
        ge=0,
        description="File size in bytes"
    )
    file_hash: str = Field(
        ...,
        description="File hash for integrity checking"
    )
    content_type: Optional[str] = Field(
        None,
        description="MIME content type"
    )
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Upload timestamp"
    )

# Health check schemas
class HealthCheckResponse(BaseSchema):
    """Schema for health check responses."""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall system health status"
    )
    version: str = Field(
        ...,
        description="Application version"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp"
    )
    services: Dict[str, str] = Field(
        ...,
        description="Status of individual services"
    )
    uptime: Optional[float] = Field(
        None,
        description="System uptime in seconds"
    )

# System configuration schemas
class ConfigValueSchema(BaseSchema):
    """Schema for configuration values."""
    
    key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Configuration key"
    )
    value: Optional[str] = Field(
        None,
        description="Configuration value"
    )
    value_type: Literal["string", "int", "float", "bool", "json"] = Field(
        "string",
        description="Value data type"
    )
    description: Optional[str] = Field(
        None,
        description="Configuration description"
    )
    is_secret: bool = Field(
        False,
        description="Whether value should be treated as secret"
    )

# Export all schemas for easy importing
__all__ = [
    # Base schemas and mixins
    "BaseSchema", "TimestampMixin", "IDMixin", "SchemaConfig",
    
    # Enums
    "UserStatusEnum", "DatasetStatusEnum", "AnalysisStatusEnum", 
    "ExecutionModeEnum", "TaskTypeEnum", "DatasetTypeEnum",
    
    # User schemas
    "UserBase", "UserCreate", "UserUpdate", "UserResponse", "UserLogin", "UserSessionResponse",
    
    # Dataset schemas
    "DatasetBase", "DatasetCreate", "DatasetUpload", "DatasetUpdate", 
    "DatasetResponse", "DatasetSummary",
    
    # Analysis schemas
    "AnalysisBase", "AnalysisRequest", "AnalysisUpdate", "AnalysisResponse", "AnalysisSummary",
    
    # ML Model schemas
    "MLModelBase", "MLModelResponse", "ModelPredictionRequest", 
    "ModelPredictionResponse", "ModelTrainingRequest",
    
    # Pipeline schemas
    "PipelineBase", "PipelineConfigSchema", "PipelineCreate", "PipelineResponse",
    "PipelineRunRequest", "PipelineRunResponse",
    
    # Kaggle integration schemas
    "KaggleTokenSchema", "KaggleConnectionRequest", "RemoteExecutionRequest", "RemoteExecutionResponse",
    
    # Dashboard schemas
    "ChartDataSchema", "MetricsResponseSchema", "DashboardDataSchema",
    
    # Common response schemas
    "StatusResponse", "ErrorResponse", "MessageResponse",
    
    # Pagination schemas
    "PaginationParams", "PaginatedResponse",
    
    # Utility schemas
    "FileUploadResponse", "HealthCheckResponse", "ConfigValueSchema",
    
    # Custom field types
    "Username", "Password", "DatasetName", "ModelName"
]
