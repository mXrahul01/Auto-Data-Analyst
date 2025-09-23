"""
🚀 AUTO-ANALYST PLATFORM - PYDANTIC VALIDATION SCHEMAS
=====================================================

Modern Pydantic v2 schemas for comprehensive API validation and serialization.
Organized by domain with proper separation of concerns and type safety.

Key Features:
- Pydantic v2 with modern field validators and model configuration
- Type-safe schemas with comprehensive validation
- Separated request/response schemas for clean API contracts
- Security-focused validation with proper sanitization
- Performance-optimized with efficient validation patterns
- Modular design for easy maintenance and extension

Structure:
- Base schemas and mixins for reusability
- User management (authentication, profiles, sessions)
- Dataset management (uploads, metadata, processing)
- ML analysis (requests, configurations, results)
- System utilities (pagination, responses, health checks)

Dependencies:
- pydantic>=2.0.0
- email-validator>=2.0.0
- python-multipart>=0.0.6 (for file uploads)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Literal, Annotated
from enum import Enum
import re
import uuid

# Pydantic v2 imports
from pydantic import (
    BaseModel, Field, field_validator, model_validator, computed_field,
    EmailStr, HttpUrl, ConfigDict
)
from pydantic.types import PositiveInt, NonNegativeInt, PositiveFloat


# =============================================================================
# CUSTOM ANNOTATED TYPES
# =============================================================================

Username = Annotated[str, Field(
    min_length=3,
    max_length=50,
    pattern=r'^[a-zA-Z0-9_-]+$',
    description="Username with alphanumeric, underscore, and hyphen only"
)]

DatasetName = Annotated[str, Field(
    min_length=1,
    max_length=255,
    pattern=r'^[a-zA-Z0-9\s\-_.()]+$',
    description="Dataset name with standard characters"
)]

AnalysisName = Annotated[str, Field(
    min_length=1,
    max_length=255,
    description="Analysis name"
)]

FileSize = Annotated[int, Field(ge=0, description="File size in bytes")]
Progress = Annotated[float, Field(ge=0.0, le=1.0, description="Progress ratio (0-1)")]
TestSize = Annotated[float, Field(gt=0.0, lt=1.0, description="Test split ratio")]


# =============================================================================
# ENUMERATIONS
# =============================================================================

class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


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
    ERROR = "error"
    ARCHIVED = "archived"


class AnalysisStatus(str, Enum):
    """Analysis execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    TEXT_ANALYSIS = "text_analysis"
    FORECASTING = "forecasting"


class ExecutionMode(str, Enum):
    """Execution environments."""
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"
    REMOTE_KAGGLE = "remote_kaggle"
    REMOTE_COLAB = "remote_colab"
    CLOUD_AWS = "cloud_aws"
    CLOUD_GCP = "cloud_gcp"


class ResponseStatus(str, Enum):
    """API response status."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PROCESSING = "processing"


# =============================================================================
# BASE SCHEMAS & MIXINS
# =============================================================================

class BaseSchema(BaseModel):
    """
    Base schema with common configuration and utilities.

    Provides consistent configuration and helper methods for all schemas.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        extra="forbid",
        populate_by_name=True,
    )


# Pure mixins (don't inherit from BaseModel to avoid MRO issues)
class TimestampMixin:
    """Mixin for schemas with timestamp fields."""

    created_at: datetime = Field(
        description="Record creation timestamp",
        examples=["2025-09-24T03:23:00+00:00"]
    )
    updated_at: datetime = Field(
        description="Record last update timestamp",
        examples=["2025-09-24T03:23:00+00:00"]
    )


class IDMixin:
    """Mixin for schemas with ID field."""

    id: PositiveInt = Field(
        description="Unique record identifier",
        examples=[1, 42, 123]
    )


class OwnershipMixin:
    """Mixin for schemas with ownership information."""

    user_id: PositiveInt = Field(
        description="ID of the owning user",
        examples=[1, 42]
    )


# =============================================================================
# USER MANAGEMENT SCHEMAS
# =============================================================================

class UserBase(BaseSchema):
    """Base user schema with common fields."""

    username: Username = Field(
        description="Unique username",
        examples=["john_doe", "data_scientist_2024"]
    )
    email: EmailStr = Field(
        description="User email address",
        examples=["john@example.com", "user@company.com"]
    )
    full_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's full name",
        examples=["John Doe", "Jane Smith"]
    )


class UserCreateRequest(UserBase):
    """Schema for user registration requests."""

    password: str = Field(
        min_length=8,
        max_length=128,
        description="Password (validated on server-side)",
        examples=["SecurePassword123!"]
    )
    confirm_password: str = Field(
        min_length=8,
        max_length=128,
        description="Password confirmation",
        examples=["SecurePassword123!"]
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email address."""
        return v.lower().strip()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format and reserved names."""
        reserved_names = {"admin", "api", "www", "mail", "ftp", "localhost"}
        if v.lower() in reserved_names:
            raise ValueError(f"Username '{v}' is reserved")
        return v.lower()

    @model_validator(mode="after")
    def validate_passwords_match(self):
        """Ensure password and confirmation match."""
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserUpdateRequest(BaseSchema):
    """Schema for user profile updates."""

    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=1000)
    avatar_url: Optional[HttpUrl] = None
    preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="User preferences as JSON"
    )

    @field_validator("email")
    @classmethod
    def validate_email_update(cls, v: Optional[str]) -> Optional[str]:
        """Validate email if provided."""
        return v.lower().strip() if v else None


class UserResponse(UserBase):
    """Schema for user data responses."""

    # From IDMixin
    id: PositiveInt = Field(description="Unique record identifier")

    # From TimestampMixin
    created_at: datetime = Field(description="Record creation timestamp")
    updated_at: datetime = Field(description="Record last update timestamp")

    # User-specific fields
    status: UserStatus = Field(description="Account status")
    is_active: bool = Field(description="Whether account is active")
    is_verified: bool = Field(description="Whether email is verified")
    is_superuser: bool = Field(description="Whether user is admin")

    last_login_at: Optional[datetime] = Field(
        None,
        description="Last login timestamp"
    )
    email_verified_at: Optional[datetime] = Field(
        None,
        description="Email verification timestamp"
    )

    # Statistics
    dataset_count: Optional[int] = Field(None, ge=0)
    analysis_count: Optional[int] = Field(None, ge=0)


class UserLoginRequest(BaseSchema):
    """Schema for user login requests."""

    identifier: str = Field(
        min_length=1,
        description="Username or email address",
        examples=["john_doe", "john@example.com"]
    )
    password: str = Field(
        min_length=1,
        description="User password"
    )
    remember_me: bool = Field(
        default=False,
        description="Extend session duration"
    )


class UserLoginResponse(BaseSchema):
    """Schema for login responses."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    user: UserResponse = Field(description="User information")


# =============================================================================
# DATASET MANAGEMENT SCHEMAS
# =============================================================================

class DatasetBase(BaseSchema):
    """Base dataset schema."""

    name: DatasetName = Field(description="Dataset display name")
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Dataset description"
    )
    tags: Optional[List[str]] = Field(
        None,
        max_length=20,
        description="Dataset tags for categorization"
    )
    is_public: bool = Field(
        default=False,
        description="Whether dataset is publicly accessible"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and clean tags."""
        if not v:
            return v

        # Clean and deduplicate tags
        cleaned_tags = []
        for tag in v:
            tag = tag.strip().lower()
            if tag and len(tag) <= 50 and tag not in cleaned_tags:
                # Only allow alphanumeric and basic characters
                if re.match(r'^[a-zA-Z0-9\s_-]+$', tag):
                    cleaned_tags.append(tag)

        return cleaned_tags[:20]  # Max 20 tags


class DatasetCreateRequest(DatasetBase):
    """Schema for dataset creation requests."""

    @field_validator("name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset name."""
        return v.strip()


class DatasetUpdateRequest(BaseSchema):
    """Schema for dataset updates."""

    name: Optional[DatasetName] = None
    description: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = Field(None, max_length=20)
    is_public: Optional[bool] = None


class DatasetResponse(DatasetBase):
    """Schema for dataset responses."""

    # From IDMixin
    id: PositiveInt = Field(description="Unique record identifier")

    # From TimestampMixin
    created_at: datetime = Field(description="Record creation timestamp")
    updated_at: datetime = Field(description="Record last update timestamp")

    # From OwnershipMixin
    user_id: PositiveInt = Field(description="ID of the owning user")

    # File information
    original_filename: str = Field(description="Original uploaded filename")
    file_size: FileSize = Field(description="File size in bytes")
    file_hash: Optional[str] = Field(
        None,
        pattern=r'^[a-fA-F0-9]{64}$',
        description="SHA256 hash of file content"
    )
    content_type: str = Field(description="MIME content type")

    # Data characteristics
    num_rows: Optional[NonNegativeInt] = Field(None, description="Number of rows")
    num_columns: Optional[NonNegativeInt] = Field(None, description="Number of columns")
    column_names: Optional[List[str]] = Field(None, description="Column names")
    column_types: Optional[Dict[str, str]] = Field(None, description="Column data types")

    # Processing status
    status: DatasetStatus = Field(description="Processing status")
    processing_error: Optional[str] = Field(None, description="Error if processing failed")
    processed_at: Optional[datetime] = Field(None, description="Processing completion time")

    # Data quality metrics
    data_quality_score: Optional[Progress] = Field(
        None,
        description="Data quality score (0-1)"
    )
    missing_value_ratio: Optional[Progress] = Field(
        None,
        description="Ratio of missing values"
    )

    # Statistics for analysis count
    analysis_count: Optional[NonNegativeInt] = Field(
        None,
        description="Number of analyses using this dataset"
    )


class DatasetSummary(BaseSchema):
    """Lightweight dataset summary for listings."""

    # From IDMixin
    id: PositiveInt = Field(description="Unique record identifier")

    name: str = Field(description="Dataset name")
    status: DatasetStatus = Field(description="Processing status")
    num_rows: Optional[NonNegativeInt] = None
    num_columns: Optional[NonNegativeInt] = None
    file_size: FileSize = Field(description="File size in bytes")
    created_at: datetime = Field(description="Creation timestamp")
    tags: Optional[List[str]] = Field(None, max_length=5)  # Limit for summary


class DatasetUploadResponse(BaseSchema):
    """Schema for file upload responses."""

    upload_id: str = Field(description="Temporary upload identifier")
    filename: str = Field(description="Uploaded filename")
    file_size: FileSize = Field(description="File size in bytes")
    content_type: str = Field(description="MIME content type")
    upload_url: Optional[HttpUrl] = Field(None, description="Signed upload URL if using cloud storage")
    expires_at: datetime = Field(description="Upload URL expiration")


# =============================================================================
# ML ANALYSIS SCHEMAS
# =============================================================================

class AnalysisBase(BaseSchema):
    """Base analysis schema."""

    name: AnalysisName = Field(description="Analysis name")
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Analysis description"
    )
    task_type: TaskType = Field(description="Type of ML task")


class AnalysisConfiguration(BaseSchema):
    """Analysis configuration schema."""

    target_column: Optional[str] = Field(
        None,
        max_length=255,
        description="Target column for supervised learning"
    )
    feature_columns: Optional[List[str]] = Field(
        None,
        max_length=1000,  # Reasonable limit on features
        description="Selected feature columns"
    )
    algorithms: Optional[List[str]] = Field(
        None,
        max_length=20,  # Reasonable limit on algorithms
        description="Selected ML algorithms"
    )

    # Training configuration
    test_size: TestSize = Field(
        default=0.2,
        description="Train/test split ratio"
    )
    validation_size: Optional[TestSize] = Field(
        None,
        description="Validation set ratio"
    )
    cross_validation_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of CV folds"
    )
    random_state: Optional[int] = Field(
        None,
        ge=0,
        description="Random seed for reproducibility"
    )

    # Advanced options
    hyperparameter_tuning: bool = Field(
        default=True,
        description="Enable hyperparameter tuning"
    )
    feature_selection: bool = Field(
        default=True,
        description="Enable automatic feature selection"
    )
    ensemble_methods: bool = Field(
        default=True,
        description="Enable ensemble methods"
    )

    @field_validator("algorithms")
    @classmethod
    def validate_algorithms(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate algorithm names."""
        if not v:
            return v

        valid_algorithms = {
            "random_forest", "xgboost", "lightgbm", "catboost",
            "logistic_regression", "svm", "naive_bayes", "knn",
            "linear_regression", "ridge", "lasso", "elastic_net",
            "decision_tree", "gradient_boosting", "ada_boost"
        }

        validated = []
        for algo in v:
            algo_clean = algo.lower().strip().replace("-", "_")
            if algo_clean in valid_algorithms:
                validated.append(algo_clean)

        return validated if validated else None


class AnalysisCreateRequest(AnalysisBase):
    """Schema for analysis creation requests."""

    dataset_id: PositiveInt = Field(description="Dataset ID to analyze")
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.LOCAL_CPU,
        description="Execution environment"
    )
    config: AnalysisConfiguration = Field(
        default_factory=AnalysisConfiguration,
        description="Analysis configuration"
    )

    # Resource limits
    max_execution_time: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Maximum execution time in seconds"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Execution priority (1=lowest, 10=highest)"
    )


class AnalysisUpdateRequest(BaseSchema):
    """Schema for analysis updates."""

    name: Optional[AnalysisName] = None
    description: Optional[str] = Field(None, max_length=2000)


class ModelPerformanceMetrics(BaseSchema):
    """Schema for model performance metrics."""

    # Classification metrics
    accuracy: Optional[Progress] = None
    precision: Optional[Progress] = None
    recall: Optional[Progress] = None
    f1_score: Optional[Progress] = None
    roc_auc: Optional[Progress] = None

    # Regression metrics
    mae: Optional[PositiveFloat] = None
    mse: Optional[PositiveFloat] = None
    rmse: Optional[PositiveFloat] = None
    r2_score: Optional[float] = Field(None, ge=-1.0, le=1.0)

    # Additional metrics
    log_loss: Optional[PositiveFloat] = None
    confusion_matrix: Optional[List[List[int]]] = None

    # Custom metrics
    custom_metrics: Optional[Dict[str, float]] = None


class AnalysisResponse(AnalysisBase):
    """Schema for analysis responses."""

    # From IDMixin
    id: PositiveInt = Field(description="Unique record identifier")

    # From TimestampMixin
    created_at: datetime = Field(description="Record creation timestamp")
    updated_at: datetime = Field(description="Record last update timestamp")

    # From OwnershipMixin
    user_id: PositiveInt = Field(description="ID of the owning user")

    analysis_id: str = Field(
        description="Unique analysis identifier",
        pattern=r'^[a-zA-Z0-9_-]+$'
    )
    dataset_id: PositiveInt = Field(description="Associated dataset ID")

    # Configuration
    config: AnalysisConfiguration = Field(description="Analysis configuration")
    execution_mode: ExecutionMode = Field(description="Execution environment")

    # Status and progress
    status: AnalysisStatus = Field(description="Current status")
    progress: Progress = Field(default=0.0, description="Progress (0-1)")
    error_message: Optional[str] = Field(None, description="Error if failed")

    # Results
    best_model_name: Optional[str] = Field(None, description="Best model name")
    performance_metrics: Optional[ModelPerformanceMetrics] = Field(
        None,
        description="Model performance metrics"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None,
        description="Feature importance scores"
    )
    model_comparison: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Model comparison results"
    )

    # Execution details
    execution_time: Optional[PositiveFloat] = Field(
        None,
        description="Total execution time in seconds"
    )
    memory_usage_mb: Optional[PositiveFloat] = Field(
        None,
        description="Peak memory usage in MB"
    )

    # Timestamps
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")


class AnalysisSummary(BaseSchema):
    """Lightweight analysis summary for listings."""

    # From IDMixin
    id: PositiveInt = Field(description="Unique record identifier")

    name: str = Field(description="Analysis name")
    task_type: TaskType = Field(description="ML task type")
    status: AnalysisStatus = Field(description="Current status")
    progress: Progress = Field(description="Progress (0-1)")
    best_model_name: Optional[str] = None
    execution_time: Optional[PositiveFloat] = None
    created_at: datetime = Field(description="Creation timestamp")


# =============================================================================
# PREDICTION SCHEMAS
# =============================================================================

class PredictionRequest(BaseSchema):
    """Schema for model prediction requests."""

    input_data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        description="Input data for prediction"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Return prediction probabilities"
    )
    return_explanations: bool = Field(
        default=False,
        description="Return prediction explanations"
    )

    @field_validator("input_data")
    @classmethod
    def validate_input_data(cls, v: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Validate input data structure."""
        if isinstance(v, dict):
            if not v:
                raise ValueError("Input data cannot be empty")
        elif isinstance(v, list):
            if not v or len(v) == 0:
                raise ValueError("Input data list cannot be empty")
            if len(v) > 1000:  # Reasonable batch size limit
                raise ValueError("Batch size too large (max 1000 records)")
        else:
            raise ValueError("Input data must be dict or list of dicts")

        return v


class PredictionResponse(BaseSchema):
    """Schema for prediction responses."""

    predictions: Union[Any, List[Any]] = Field(description="Model predictions")
    probabilities: Optional[Union[List[float], List[List[float]]]] = Field(
        None,
        description="Prediction probabilities"
    )
    explanations: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        None,
        description="Prediction explanations"
    )

    # Metadata
    model_info: Dict[str, str] = Field(description="Model information")
    prediction_time_ms: PositiveFloat = Field(description="Prediction time in ms")
    request_id: str = Field(description="Request identifier")


# =============================================================================
# SYSTEM & UTILITY SCHEMAS
# =============================================================================

class PaginationParams(BaseSchema):
    """Schema for pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Literal["asc", "desc"] = Field(default="asc", description="Sort order")


class PaginatedResponse(BaseSchema):
    """Generic paginated response schema."""

    items: List[Any] = Field(description="Items for current page")
    total: NonNegativeInt = Field(description="Total number of items")
    page: PositiveInt = Field(description="Current page number")
    size: PositiveInt = Field(description="Items per page")
    pages: PositiveInt = Field(description="Total number of pages")
    has_next: bool = Field(description="Has next page")
    has_previous: bool = Field(description="Has previous page")

    @computed_field
    @property
    def pagination_info(self) -> Dict[str, Any]:
        """Computed pagination information."""
        return {
            "current_page": self.page,
            "total_pages": self.pages,
            "has_next": self.has_next,
            "has_previous": self.has_previous,
            "items_on_page": len(self.items),
            "total_items": self.total
        }


class APIResponse(BaseSchema):
    """Generic API response wrapper."""

    status: ResponseStatus = Field(description="Response status")
    message: str = Field(description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )


class ErrorResponse(BaseSchema):
    """Error response schema."""

    status: Literal[ResponseStatus.ERROR] = ResponseStatus.ERROR
    error_type: str = Field(description="Error classification")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")


class HealthCheckResponse(BaseSchema):
    """Health check response schema."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall system health"
    )
    version: str = Field(description="Application version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp"
    )
    services: Dict[str, str] = Field(description="Individual service statuses")
    uptime_seconds: Optional[PositiveFloat] = Field(None, description="System uptime")

    @field_validator("services")
    @classmethod
    def validate_services(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate service status values."""
        valid_statuses = {"healthy", "degraded", "unhealthy", "unknown"}
        for service, status in v.items():
            if status not in valid_statuses:
                v[service] = "unknown"
        return v


class FileUploadResponse(BaseSchema):
    """File upload response schema."""

    filename: str = Field(description="Original filename")
    file_size: FileSize = Field(description="File size in bytes")
    file_hash: str = Field(
        pattern=r'^[a-fA-F0-9]{64}$',
        description="SHA256 hash"
    )
    content_type: str = Field(description="MIME content type")
    upload_id: str = Field(description="Upload identifier")
    storage_path: Optional[str] = Field(None, description="Storage path")
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Upload timestamp"
    )


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ValidationUtils:
    """Utility functions for common validations."""

    @staticmethod
    def validate_file_size(size: int, max_size: int = 1024 * 1024 * 1024) -> bool:
        """Validate file size is within limits."""
        return 0 < size <= max_size

    @staticmethod
    def validate_column_name(name: str) -> bool:
        """Validate column name format."""
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name))

    @staticmethod
    def sanitize_string(text: str) -> str:
        """Sanitize string input."""
        return re.sub(r'[<>"\']', '', text.strip())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base schemas
    "BaseSchema",

    # Enums
    "Environment", "UserStatus", "DatasetStatus", "AnalysisStatus",
    "TaskType", "ExecutionMode", "ResponseStatus",

    # User schemas
    "UserBase", "UserCreateRequest", "UserUpdateRequest", "UserResponse",
    "UserLoginRequest", "UserLoginResponse",

    # Dataset schemas
    "DatasetBase", "DatasetCreateRequest", "DatasetUpdateRequest",
    "DatasetResponse", "DatasetSummary", "DatasetUploadResponse",

    # Analysis schemas
    "AnalysisBase", "AnalysisConfiguration", "AnalysisCreateRequest",
    "AnalysisUpdateRequest", "AnalysisResponse", "AnalysisSummary",

    # Prediction schemas
    "PredictionRequest", "PredictionResponse", "ModelPerformanceMetrics",

    # System schemas
    "PaginationParams", "PaginatedResponse", "APIResponse",
    "ErrorResponse", "HealthCheckResponse", "FileUploadResponse",

    # Utilities
    "ValidationUtils",

    # Type aliases
    "Username", "DatasetName", "AnalysisName", "FileSize", "Progress", "TestSize",
]
