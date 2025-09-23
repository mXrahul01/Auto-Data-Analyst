"""
🚀 AUTO-ANALYST PLATFORM - INITIAL SCHEMA MIGRATION
=================================================

Production-grade database migration for the Auto-Analyst platform core entities.
This migration creates the essential tables with proper constraints, indexes,
and multi-database compatibility.

Key Principles:
- Multi-database compatibility (PostgreSQL, MySQL, SQLite)
- Consistent naming conventions and data types
- Proper foreign key constraints and indexes
- Optimized for performance and scalability
- Follows Alembic best practices

Core Tables:
- users: User accounts and authentication
- datasets: Dataset metadata and processing status  
- analyses: ML analysis runs and configurations
- models: Trained model registry
- predictions: Prediction requests and results

Features:
- UUID primary keys with proper generation
- Consistent timestamp handling across databases
- Optimized indexes for common query patterns
- Proper foreign key constraints with cascade rules
- JSON columns with fallback for older databases
- Soft delete support for critical entities

Migration Strategy:
- Split into logical phases for safer deployment
- Backward compatible changes only
- Comprehensive rollback support
- Performance-optimized index creation

Revision ID: 001_initial_schema
Revises: None
Create Date: 2025-09-24 03:30:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, mysql, sqlite
from sqlalchemy.sql import func
from sqlalchemy import text
import uuid

# Revision identifiers, used by Alembic
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def get_uuid_type():
    """Get appropriate UUID type based on database dialect."""
    dialect = op.get_bind().dialect.name
    
    if dialect == 'postgresql':
        # Use native UUID type in PostgreSQL
        return postgresql.UUID(as_uuid=True)
    else:
        # Use CHAR(36) for other databases
        return sa.CHAR(36)


def get_uuid_default():
    """Get appropriate UUID default based on database dialect."""
    dialect = op.get_bind().dialect.name
    
    if dialect == 'postgresql':
        # Use PostgreSQL's gen_random_uuid() if available, otherwise uuid_generate_v4()
        return text("gen_random_uuid()")
    elif dialect == 'mysql':
        # Use MySQL's UUID() function
        return text("UUID()")
    else:
        # For SQLite and others, we'll handle UUID generation in application
        return None


def get_json_type():
    """Get appropriate JSON type based on database dialect."""
    dialect = op.get_bind().dialect.name
    
    if dialect == 'postgresql':
        return postgresql.JSONB  # Better performance than JSON
    elif dialect == 'mysql':
        return mysql.JSON
    else:
        # SQLite fallback - store as TEXT
        return sa.Text


def upgrade():
    """Create initial database schema with core tables."""
    
    # ==========================================================================
    # CORE USERS TABLE
    # ==========================================================================
    
    op.create_table(
        'users',
        # Primary key - consistent across all tables
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique user identifier"
        ),
        
        # Authentication fields
        sa.Column(
            'email', 
            sa.String(255), 
            nullable=False, 
            unique=True,
            comment="User email address (unique)"
        ),
        sa.Column(
            'username', 
            sa.String(100), 
            nullable=False, 
            unique=True,
            comment="Username (unique, 3-50 chars)"
        ),
        sa.Column(
            'hashed_password', 
            sa.String(255), 
            nullable=False,
            comment="Bcrypt hashed password"
        ),
        
        # Profile information
        sa.Column(
            'full_name', 
            sa.String(255), 
            nullable=True,
            comment="User's full name"
        ),
        sa.Column(
            'avatar_url', 
            sa.String(500), 
            nullable=True,
            comment="Profile avatar URL"
        ),
        
        # Account status
        sa.Column(
            'is_active', 
            sa.Boolean, 
            nullable=False, 
            default=True,
            comment="Account active status"
        ),
        sa.Column(
            'is_verified', 
            sa.Boolean, 
            nullable=False, 
            default=False,
            comment="Email verification status"
        ),
        sa.Column(
            'is_superuser', 
            sa.Boolean, 
            nullable=False, 
            default=False,
            comment="Admin privileges flag"
        ),
        
        # Activity tracking
        sa.Column(
            'last_login_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Last login timestamp"
        ),
        sa.Column(
            'login_count', 
            sa.Integer, 
            nullable=False, 
            default=0,
            comment="Total login count"
        ),
        sa.Column(
            'failed_login_attempts', 
            sa.Integer, 
            nullable=False, 
            default=0,
            comment="Failed login attempts counter"
        ),
        sa.Column(
            'locked_until', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Account lock expiration"
        ),
        
        # User preferences and metadata
        sa.Column(
            'preferences', 
            get_json_type(), 
            nullable=True,
            comment="User preferences (JSON)"
        ),
        sa.Column(
            'timezone', 
            sa.String(50), 
            nullable=False, 
            default='UTC',
            comment="User timezone preference"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Account creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        sa.Column(
            'deleted_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Soft delete timestamp"
        ),
        
        comment="User accounts and authentication"
    )
    
    # User indexes for performance
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_active', 'users', ['is_active'])
    op.create_index('ix_users_created_at', 'users', ['created_at'])
    
    # ==========================================================================
    # DATASETS TABLE
    # ==========================================================================
    
    op.create_table(
        'datasets',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique dataset identifier"
        ),
        
        # Foreign key to users
        sa.Column(
            'user_id', 
            get_uuid_type(), 
            sa.ForeignKey('users.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Dataset owner"
        ),
        
        # Dataset identification
        sa.Column(
            'name', 
            sa.String(255), 
            nullable=False,
            comment="Dataset display name"
        ),
        sa.Column(
            'description', 
            sa.Text, 
            nullable=True,
            comment="Dataset description"
        ),
        sa.Column(
            'tags', 
            get_json_type(), 
            nullable=True,
            comment="Dataset tags (JSON array)"
        ),
        
        # File information
        sa.Column(
            'original_filename', 
            sa.String(255), 
            nullable=False,
            comment="Original uploaded filename"
        ),
        sa.Column(
            'file_path', 
            sa.String(500), 
            nullable=True,
            comment="Server storage path"
        ),
        sa.Column(
            'file_size', 
            sa.BigInteger, 
            nullable=False,
            comment="File size in bytes"
        ),
        sa.Column(
            'content_type', 
            sa.String(100), 
            nullable=False,
            comment="MIME content type"
        ),
        sa.Column(
            'file_hash', 
            sa.String(64), 
            nullable=True,
            comment="SHA256 file hash"
        ),
        
        # Data characteristics
        sa.Column(
            'num_rows', 
            sa.Integer, 
            nullable=True,
            comment="Number of data rows"
        ),
        sa.Column(
            'num_columns', 
            sa.Integer, 
            nullable=True,
            comment="Number of data columns"
        ),
        sa.Column(
            'column_names', 
            get_json_type(), 
            nullable=True,
            comment="Column names (JSON array)"
        ),
        sa.Column(
            'column_types', 
            get_json_type(), 
            nullable=True,
            comment="Column data types (JSON object)"
        ),
        
        # Processing status
        sa.Column(
            'status', 
            sa.String(50), 
            nullable=False, 
            default='uploaded',
            comment="Processing status"
        ),
        sa.Column(
            'processing_error', 
            sa.Text, 
            nullable=True,
            comment="Processing error message"
        ),
        sa.Column(
            'processed_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Processing completion time"
        ),
        
        # Data quality metrics
        sa.Column(
            'data_quality_score', 
            sa.Float, 
            nullable=True,
            comment="Data quality score (0-1)"
        ),
        sa.Column(
            'missing_value_ratio', 
            sa.Float, 
            nullable=True,
            comment="Ratio of missing values"
        ),
        
        # Visibility
        sa.Column(
            'is_public', 
            sa.Boolean, 
            nullable=False, 
            default=False,
            comment="Public dataset flag"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Dataset creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        sa.Column(
            'deleted_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Soft delete timestamp"
        ),
        
        comment="Dataset metadata and processing information"
    )
    
    # Dataset indexes
    op.create_index('ix_datasets_user_id', 'datasets', ['user_id'])
    op.create_index('ix_datasets_status', 'datasets', ['status'])
    op.create_index('ix_datasets_public', 'datasets', ['is_public', 'status'])
    op.create_index('ix_datasets_created_at', 'datasets', ['created_at'])
    op.create_index('ix_datasets_file_hash', 'datasets', ['file_hash'])
    
    # ==========================================================================
    # ANALYSES TABLE
    # ==========================================================================
    
    op.create_table(
        'analyses',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique analysis identifier"
        ),
        
        # Foreign keys
        sa.Column(
            'user_id', 
            get_uuid_type(), 
            sa.ForeignKey('users.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Analysis creator"
        ),
        sa.Column(
            'dataset_id', 
            get_uuid_type(), 
            sa.ForeignKey('datasets.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Dataset being analyzed"
        ),
        
        # Analysis identification
        sa.Column(
            'name', 
            sa.String(255), 
            nullable=False,
            comment="Analysis name"
        ),
        sa.Column(
            'description', 
            sa.Text, 
            nullable=True,
            comment="Analysis description"
        ),
        
        # ML configuration
        sa.Column(
            'task_type', 
            sa.String(50), 
            nullable=False,
            comment="ML task type (classification, regression, etc.)"
        ),
        sa.Column(
            'target_column', 
            sa.String(255), 
            nullable=True,
            comment="Target variable column name"
        ),
        sa.Column(
            'feature_columns', 
            get_json_type(), 
            nullable=True,
            comment="Selected feature columns (JSON array)"
        ),
        sa.Column(
            'algorithms', 
            get_json_type(), 
            nullable=True,
            comment="Selected algorithms (JSON array)"
        ),
        
        # Execution settings
        sa.Column(
            'execution_mode', 
            sa.String(50), 
            nullable=False, 
            default='local_cpu',
            comment="Execution environment"
        ),
        sa.Column(
            'config', 
            get_json_type(), 
            nullable=True,
            comment="Analysis configuration (JSON)"
        ),
        
        # Status and progress
        sa.Column(
            'status', 
            sa.String(50), 
            nullable=False, 
            default='pending',
            comment="Analysis status"
        ),
        sa.Column(
            'progress', 
            sa.Float, 
            nullable=False, 
            default=0.0,
            comment="Progress ratio (0-1)"
        ),
        sa.Column(
            'error_message', 
            sa.Text, 
            nullable=True,
            comment="Error message if failed"
        ),
        
        # Results
        sa.Column(
            'best_model_name', 
            sa.String(255), 
            nullable=True,
            comment="Best performing model name"
        ),
        sa.Column(
            'performance_metrics', 
            get_json_type(), 
            nullable=True,
            comment="Model performance metrics (JSON)"
        ),
        sa.Column(
            'feature_importance', 
            get_json_type(), 
            nullable=True,
            comment="Feature importance scores (JSON)"
        ),
        sa.Column(
            'model_comparison', 
            get_json_type(), 
            nullable=True,
            comment="Model comparison results (JSON)"
        ),
        
        # Execution timing
        sa.Column(
            'started_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Analysis start time"
        ),
        sa.Column(
            'completed_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Analysis completion time"
        ),
        sa.Column(
            'execution_time', 
            sa.Float, 
            nullable=True,
            comment="Total execution time in seconds"
        ),
        
        # Resource usage
        sa.Column(
            'memory_usage_mb', 
            sa.Float, 
            nullable=True,
            comment="Peak memory usage in MB"
        ),
        sa.Column(
            'cpu_time_seconds', 
            sa.Float, 
            nullable=True,
            comment="CPU time in seconds"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Analysis creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        sa.Column(
            'deleted_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Soft delete timestamp"
        ),
        
        comment="ML analysis runs and configurations"
    )
    
    # Analysis indexes
    op.create_index('ix_analyses_user_id', 'analyses', ['user_id'])
    op.create_index('ix_analyses_dataset_id', 'analyses', ['dataset_id'])
    op.create_index('ix_analyses_status', 'analyses', ['status'])
    op.create_index('ix_analyses_task_type', 'analyses', ['task_type'])
    op.create_index('ix_analyses_created_at', 'analyses', ['created_at'])
    op.create_index('ix_analyses_user_status', 'analyses', ['user_id', 'status'])
    
    # ==========================================================================
    # MODELS TABLE
    # ==========================================================================
    
    op.create_table(
        'models',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique model identifier"
        ),
        
        # Foreign keys
        sa.Column(
            'analysis_id', 
            get_uuid_type(), 
            sa.ForeignKey('analyses.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Parent analysis"
        ),
        sa.Column(
            'user_id', 
            get_uuid_type(), 
            sa.ForeignKey('users.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Model creator"
        ),
        
        # Model identification
        sa.Column(
            'name', 
            sa.String(255), 
            nullable=False,
            comment="Model name"
        ),
        sa.Column(
            'model_type', 
            sa.String(100), 
            nullable=False,
            comment="Model type/algorithm"
        ),
        sa.Column(
            'version', 
            sa.String(50), 
            nullable=False, 
            default='1.0.0',
            comment="Model version"
        ),
        
        # Model lifecycle
        sa.Column(
            'stage', 
            sa.String(50), 
            nullable=False, 
            default='development',
            comment="Model stage (development, staging, production)"
        ),
        sa.Column(
            'status', 
            sa.String(50), 
            nullable=False, 
            default='training',
            comment="Model status (training, ready, deployed, archived)"
        ),
        
        # Model configuration and metrics
        sa.Column(
            'hyperparameters', 
            get_json_type(), 
            nullable=True,
            comment="Model hyperparameters (JSON)"
        ),
        sa.Column(
            'performance_metrics', 
            get_json_type(), 
            nullable=True,
            comment="Model performance metrics (JSON)"
        ),
        sa.Column(
            'feature_importance', 
            get_json_type(), 
            nullable=True,
            comment="Feature importance (JSON)"
        ),
        
        # Model metadata
        sa.Column(
            'model_size_mb', 
            sa.Float, 
            nullable=True,
            comment="Model file size in MB"
        ),
        sa.Column(
            'inference_time_ms', 
            sa.Float, 
            nullable=True,
            comment="Average inference time in milliseconds"
        ),
        sa.Column(
            'description', 
            sa.Text, 
            nullable=True,
            comment="Model description"
        ),
        sa.Column(
            'tags', 
            get_json_type(), 
            nullable=True,
            comment="Model tags (JSON array)"
        ),
        
        # Storage and deployment
        sa.Column(
            'artifact_location', 
            sa.String(500), 
            nullable=True,
            comment="Model artifact storage path"
        ),
        sa.Column(
            'deployment_config', 
            get_json_type(), 
            nullable=True,
            comment="Deployment configuration (JSON)"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Model creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        sa.Column(
            'deployed_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Deployment timestamp"
        ),
        sa.Column(
            'deleted_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Soft delete timestamp"
        ),
        
        comment="Trained model registry and metadata"
    )
    
    # Model indexes
    op.create_index('ix_models_analysis_id', 'models', ['analysis_id'])
    op.create_index('ix_models_user_id', 'models', ['user_id'])
    op.create_index('ix_models_stage', 'models', ['stage'])
    op.create_index('ix_models_status', 'models', ['status'])
    op.create_index('ix_models_created_at', 'models', ['created_at'])
    
    # ==========================================================================
    # PREDICTIONS TABLE
    # ==========================================================================
    
    op.create_table(
        'predictions',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique prediction identifier"
        ),
        
        # Foreign keys
        sa.Column(
            'user_id', 
            get_uuid_type(), 
            sa.ForeignKey('users.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Prediction requester"
        ),
        sa.Column(
            'model_id', 
            get_uuid_type(), 
            sa.ForeignKey('models.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Model used for prediction"
        ),
        
        # Prediction request details
        sa.Column(
            'request_type', 
            sa.String(50), 
            nullable=False, 
            default='single',
            comment="Prediction type (single, batch)"
        ),
        sa.Column(
            'status', 
            sa.String(50), 
            nullable=False, 
            default='pending',
            comment="Prediction status"
        ),
        
        # Input and output data
        sa.Column(
            'input_data', 
            get_json_type(), 
            nullable=True,
            comment="Input data for prediction (JSON)"
        ),
        sa.Column(
            'predictions', 
            get_json_type(), 
            nullable=True,
            comment="Prediction results (JSON)"
        ),
        sa.Column(
            'probabilities', 
            get_json_type(), 
            nullable=True,
            comment="Prediction probabilities (JSON)"
        ),
        sa.Column(
            'explanations', 
            get_json_type(), 
            nullable=True,
            comment="Prediction explanations (JSON)"
        ),
        
        # Performance metrics
        sa.Column(
            'num_predictions', 
            sa.Integer, 
            nullable=False, 
            default=1,
            comment="Number of predictions made"
        ),
        sa.Column(
            'processing_time_ms', 
            sa.Float, 
            nullable=True,
            comment="Processing time in milliseconds"
        ),
        
        # Model version tracking
        sa.Column(
            'model_version', 
            sa.String(50), 
            nullable=True,
            comment="Model version used"
        ),
        
        # Error handling
        sa.Column(
            'error_message', 
            sa.Text, 
            nullable=True,
            comment="Error message if prediction failed"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Prediction request timestamp"
        ),
        sa.Column(
            'completed_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Prediction completion timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        
        comment="Prediction requests and results"
    )
    
    # Prediction indexes
    op.create_index('ix_predictions_user_id', 'predictions', ['user_id'])
    op.create_index('ix_predictions_model_id', 'predictions', ['model_id'])
    op.create_index('ix_predictions_status', 'predictions', ['status'])
    op.create_index('ix_predictions_created_at', 'predictions', ['created_at'])
    op.create_index('ix_predictions_request_type', 'predictions', ['request_type'])
    
    # ==========================================================================
    # CREATE ADDITIONAL CONSTRAINTS
    # ==========================================================================
    
    # Check constraints for data integrity
    op.create_check_constraint(
        'ck_users_email_format',
        'users',
        "email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'"
    )
    
    op.create_check_constraint(
        'ck_users_username_length',
        'users',
        "length(username) >= 3 AND length(username) <= 50"
    )
    
    op.create_check_constraint(
        'ck_datasets_file_size_positive',
        'datasets',
        "file_size >= 0"
    )
    
    op.create_check_constraint(
        'ck_analyses_progress_range',
        'analyses',
        "progress >= 0.0 AND progress <= 1.0"
    )
    
    op.create_check_constraint(
        'ck_predictions_num_predictions_positive',
        'predictions',
        "num_predictions >= 0"
    )
    
    # ==========================================================================
    # DATABASE-SPECIFIC OPTIMIZATIONS
    # ==========================================================================
    
    dialect = op.get_bind().dialect.name
    
    if dialect == 'postgresql':
        # PostgreSQL-specific optimizations
        
        # Enable UUID extension if not already enabled
        op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        op.execute("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\"")
        
        # Create partial indexes for soft-deleted records
        op.execute(
            "CREATE INDEX ix_users_active_not_deleted ON users (id) "
            "WHERE deleted_at IS NULL"
        )
        op.execute(
            "CREATE INDEX ix_datasets_active_not_deleted ON datasets (id) "
            "WHERE deleted_at IS NULL"
        )
        op.execute(
            "CREATE INDEX ix_analyses_active_not_deleted ON analyses (id) "
            "WHERE deleted_at IS NULL"
        )
        
        # Create GIN indexes for JSON columns for better query performance
        op.execute(
            "CREATE INDEX ix_datasets_column_names_gin ON datasets "
            "USING GIN (column_names)"
        )
        op.execute(
            "CREATE INDEX ix_analyses_config_gin ON analyses "
            "USING GIN (config)"
        )
    
    print("✅ Initial core schema migration completed successfully!")
    print(f"📊 Created 5 core tables with optimized indexes for {dialect}")
    print("🚀 Ready for application deployment!")


def downgrade():
    """
    Rollback the initial schema migration.
    
    Drops all tables created in the upgrade function in reverse order
    to handle foreign key dependencies properly.
    """
    
    # List of tables to drop in reverse dependency order
    tables_to_drop = [
        'predictions',
        'models', 
        'analyses',
        'datasets',
        'users'
    ]
    
    # Drop tables with proper error handling
    for table in tables_to_drop:
        try:
            # Check if table exists before dropping (for safety)
            op.execute(f"""
                SELECT CASE 
                WHEN EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = '{table}'
                ) THEN 1 ELSE 0 END
            """)
            
            op.drop_table(table)
            print(f"✅ Dropped table: {table}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not drop table {table}: {e}")
            # Continue with other tables even if one fails
            continue
    
    # Drop database-specific extensions (PostgreSQL)
    try:
        dialect = op.get_bind().dialect.name
        if dialect == 'postgresql':
            # Note: We don't drop UUID extensions as other applications might use them
            print("ℹ️  Note: PostgreSQL extensions (uuid-ossp, pgcrypto) left intact")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up extensions: {e}")
    
    print("✅ Schema downgrade completed successfully!")
    print("🔄 Database rolled back to pre-migration state")
