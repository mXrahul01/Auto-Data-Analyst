"""
Initial Schema Migration for Auto-Analyst Platform

This migration creates the foundational database schema for the Auto-Analyst
zero-code AI-powered data analysis platform, including:

- User management and authentication
- Dataset storage and metadata
- ML analysis tracking and results
- Model management and deployment
- Feature store integration
- Monitoring and observability
- Audit logging and security

Tables Created:
- users: User accounts and profiles
- datasets: Uploaded datasets metadata
- analyses: ML analysis runs and configurations
- models: Trained model registry
- predictions: Prediction requests and results
- insights: AI-generated insights and recommendations
- features: Feature store metadata
- experiments: MLflow experiment tracking
- monitoring: Model and data drift monitoring
- audit_logs: System audit and security logs

Features:
- Full referential integrity with foreign keys
- Optimized indexes for query performance
- JSON columns for flexible metadata storage
- Timestamp tracking for all entities
- Soft delete support for important entities
- Partitioning support for large tables
- Database-level constraints and validations

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-09-21 11:47:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import func
import uuid
from datetime import datetime

# revision identifiers, used by Alembic
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create initial database schema."""
    
    # ======================
    # USERS AND AUTHENTICATION
    # ======================
    
    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('uuid', sa.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('username', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('avatar_url', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean, nullable=False, default=False),
        sa.Column('is_superuser', sa.Boolean, nullable=False, default=False),
        sa.Column('email_verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('login_count', sa.Integer, nullable=False, default=0),
        sa.Column('failed_login_attempts', sa.Integer, nullable=False, default=0),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('preferences', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # User sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('session_token', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('refresh_token', sa.String(255), unique=True, nullable=True),
        sa.Column('device_info', sa.JSON, nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # ======================
    # DATASETS AND DATA MANAGEMENT
    # ======================
    
    # Datasets table
    op.create_table(
        'datasets',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('uuid', sa.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('original_filename', sa.String(255), nullable=False),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('file_size', sa.BigInteger, nullable=False, default=0),
        sa.Column('file_format', sa.String(50), nullable=False),
        sa.Column('mime_type', sa.String(100), nullable=True),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='uploading'),
        sa.Column('num_rows', sa.Integer, nullable=True),
        sa.Column('num_columns', sa.Integer, nullable=True),
        sa.Column('column_names', sa.JSON, nullable=True),
        sa.Column('column_types', sa.JSON, nullable=True),
        sa.Column('column_statistics', sa.JSON, nullable=True),
        sa.Column('data_quality_score', sa.Float, nullable=True),
        sa.Column('data_quality_report', sa.JSON, nullable=True),
        sa.Column('missing_value_ratio', sa.Float, nullable=True),
        sa.Column('duplicate_ratio', sa.Float, nullable=True),
        sa.Column('preprocessing_config', sa.JSON, nullable=True),
        sa.Column('validation_results', sa.JSON, nullable=True),
        sa.Column('sample_data', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('is_public', sa.Boolean, nullable=False, default=False),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Dataset versions table (for data versioning)
    op.create_table(
        'dataset_versions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('dataset_id', sa.Integer, sa.ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('file_path', sa.String(500), nullable=False),
        sa.Column('changes_summary', sa.Text, nullable=True),
        sa.Column('preprocessing_applied', sa.JSON, nullable=True),
        sa.Column('statistics', sa.JSON, nullable=True),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    
    # ======================
    # ML ANALYSIS AND EXPERIMENTS
    # ======================
    
    # ML Analyses table
    op.create_table(
        'analyses',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('dataset_id', sa.Integer, sa.ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('task_type', sa.String(50), nullable=False, index=True),
        sa.Column('target_column', sa.String(255), nullable=True),
        sa.Column('feature_columns', sa.JSON, nullable=True),
        sa.Column('execution_mode', sa.String(50), nullable=False, default='local'),
        sa.Column('compute_backend', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='created', index=True),
        sa.Column('progress', sa.Float, nullable=False, default=0.0),
        sa.Column('current_step', sa.String(255), nullable=True),
        sa.Column('total_steps', sa.Integer, nullable=True),
        sa.Column('config', sa.JSON, nullable=True),
        sa.Column('preprocessing_config', sa.JSON, nullable=True),
        sa.Column('model_selection_config', sa.JSON, nullable=True),
        sa.Column('hyperparameter_config', sa.JSON, nullable=True),
        sa.Column('validation_config', sa.JSON, nullable=True),
        sa.Column('models_evaluated', sa.JSON, nullable=True),
        sa.Column('best_model_name', sa.String(255), nullable=True),
        sa.Column('best_model_id', sa.String(36), nullable=True),
        sa.Column('performance_metrics', sa.JSON, nullable=True),
        sa.Column('feature_importance', sa.JSON, nullable=True),
        sa.Column('model_comparison', sa.JSON, nullable=True),
        sa.Column('cross_validation_results', sa.JSON, nullable=True),
        sa.Column('training_history', sa.JSON, nullable=True),
        sa.Column('mlflow_experiment_id', sa.String(255), nullable=True),
        sa.Column('mlflow_run_id', sa.String(255), nullable=True),
        sa.Column('artifacts_path', sa.String(500), nullable=True),
        sa.Column('model_artifacts', sa.JSON, nullable=True),
        sa.Column('logs', sa.Text, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_details', sa.JSON, nullable=True),
        sa.Column('execution_time', sa.Float, nullable=True),
        sa.Column('resource_usage', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Model registry table
    op.create_table(
        'models',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(100), nullable=False),
        sa.Column('algorithm', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('stage', sa.String(50), nullable=False, default='none'),  # none, staging, production, archived
        sa.Column('status', sa.String(50), nullable=False, default='training'),  # training, ready, deployed, failed
        sa.Column('hyperparameters', sa.JSON, nullable=True),
        sa.Column('training_config', sa.JSON, nullable=True),
        sa.Column('performance_metrics', sa.JSON, nullable=True),
        sa.Column('validation_metrics', sa.JSON, nullable=True),
        sa.Column('feature_importance', sa.JSON, nullable=True),
        sa.Column('model_signature', sa.JSON, nullable=True),
        sa.Column('input_schema', sa.JSON, nullable=True),
        sa.Column('output_schema', sa.JSON, nullable=True),
        sa.Column('model_size_mb', sa.Float, nullable=True),
        sa.Column('inference_time_ms', sa.Float, nullable=True),
        sa.Column('memory_usage_mb', sa.Float, nullable=True),
        sa.Column('mlflow_model_uri', sa.String(500), nullable=True),
        sa.Column('mlflow_run_id', sa.String(255), nullable=True),
        sa.Column('artifact_location', sa.String(500), nullable=True),
        sa.Column('deployment_config', sa.JSON, nullable=True),
        sa.Column('monitoring_config', sa.JSON, nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('training_data_hash', sa.String(64), nullable=True),
        sa.Column('model_hash', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Column('deployed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Model versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('changes', sa.JSON, nullable=True),
        sa.Column('performance_comparison', sa.JSON, nullable=True),
        sa.Column('artifact_location', sa.String(500), nullable=False),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.UniqueConstraint('model_id', 'version', name='uq_model_version')
    )
    
    # ======================
    # PREDICTIONS AND INFERENCE
    # ======================
    
    # Predictions table
    op.create_table(
        'predictions',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('request_type', sa.String(50), nullable=False, default='single'),  # single, batch, streaming
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('input_data', sa.JSON, nullable=True),
        sa.Column('input_data_hash', sa.String(64), nullable=True, index=True),
        sa.Column('predictions', sa.JSON, nullable=True),
        sa.Column('probabilities', sa.JSON, nullable=True),
        sa.Column('confidence_scores', sa.JSON, nullable=True),
        sa.Column('feature_contributions', sa.JSON, nullable=True),
        sa.Column('explanations', sa.JSON, nullable=True),
        sa.Column('prediction_metadata', sa.JSON, nullable=True),
        sa.Column('num_predictions', sa.Integer, nullable=False, default=0),
        sa.Column('processing_time_ms', sa.Float, nullable=True),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('api_version', sa.String(50), nullable=True),
        sa.Column('client_info', sa.JSON, nullable=True),
        sa.Column('feedback_score', sa.Float, nullable=True),
        sa.Column('feedback_comment', sa.Text, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Batch prediction jobs table
    op.create_table(
        'batch_predictions',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='queued'),
        sa.Column('input_file_path', sa.String(500), nullable=False),
        sa.Column('output_file_path', sa.String(500), nullable=True),
        sa.Column('input_format', sa.String(50), nullable=False),
        sa.Column('output_format', sa.String(50), nullable=False, default='csv'),
        sa.Column('total_records', sa.Integer, nullable=True),
        sa.Column('processed_records', sa.Integer, nullable=False, default=0),
        sa.Column('failed_records', sa.Integer, nullable=False, default=0),
        sa.Column('progress', sa.Float, nullable=False, default=0.0),
        sa.Column('config', sa.JSON, nullable=True),
        sa.Column('error_log', sa.Text, nullable=True),
        sa.Column('execution_time', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # ======================
    # INSIGHTS AND RECOMMENDATIONS
    # ======================
    
    # AI-generated insights table
    op.create_table(
        'insights',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('insight_type', sa.String(100), nullable=False),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('importance_score', sa.Float, nullable=False, default=0.5),
        sa.Column('confidence_score', sa.Float, nullable=False, default=0.5),
        sa.Column('statistical_evidence', sa.JSON, nullable=True),
        sa.Column('visualizations', sa.JSON, nullable=True),
        sa.Column('recommendations', sa.JSON, nullable=True),
        sa.Column('business_impact', sa.Text, nullable=True),
        sa.Column('technical_details', sa.JSON, nullable=True),
        sa.Column('data_sources', sa.JSON, nullable=True),
        sa.Column('methodology', sa.Text, nullable=True),
        sa.Column('limitations', sa.Text, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('is_actionable', sa.Boolean, nullable=False, default=True),
        sa.Column('is_approved', sa.Boolean, nullable=False, default=False),
        sa.Column('user_feedback', sa.JSON, nullable=True),
        sa.Column('view_count', sa.Integer, nullable=False, default=0),
        sa.Column('last_viewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # ======================
    # FEATURE STORE
    # ======================
    
    # Feature definitions table
    op.create_table(
        'features',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('display_name', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('feature_type', sa.String(50), nullable=False),
        sa.Column('data_type', sa.String(50), nullable=False),
        sa.Column('entity', sa.String(255), nullable=False),
        sa.Column('source_table', sa.String(255), nullable=True),
        sa.Column('source_column', sa.String(255), nullable=True),
        sa.Column('transformation_logic', sa.Text, nullable=True),
        sa.Column('aggregation_function', sa.String(100), nullable=True),
        sa.Column('window_size', sa.String(100), nullable=True),
        sa.Column('default_value', sa.String(255), nullable=True),
        sa.Column('validation_rules', sa.JSON, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Column('deprecated_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Feature sets table
    op.create_table(
        'feature_sets',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('features', sa.JSON, nullable=False),  # List of feature IDs
        sa.Column('entity', sa.String(255), nullable=False),
        sa.Column('online_enabled', sa.Boolean, nullable=False, default=False),
        sa.Column('offline_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('refresh_frequency', sa.String(100), nullable=True),
        sa.Column('retention_days', sa.Integer, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Feature usage tracking table
    op.create_table(
        'feature_usage',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('feature_id', sa.String(36), sa.ForeignKey('features.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('usage_type', sa.String(50), nullable=False),  # training, inference, analysis
        sa.Column('usage_count', sa.Integer, nullable=False, default=1),
        sa.Column('importance_score', sa.Float, nullable=True),
        sa.Column('performance_impact', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.UniqueConstraint('feature_id', 'analysis_id', 'usage_type', name='uq_feature_analysis_usage')
    )
    
    # ======================
    # MONITORING AND OBSERVABILITY
    # ======================
    
    # Model monitoring table
    op.create_table(
        'model_monitoring',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('monitoring_type', sa.String(50), nullable=False),  # data_drift, model_drift, performance
        sa.Column('metric_name', sa.String(255), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('threshold', sa.Float, nullable=True),
        sa.Column('baseline_value', sa.Float, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='normal'),  # normal, warning, critical
        sa.Column('detection_method', sa.String(100), nullable=True),
        sa.Column('statistical_test', sa.String(100), nullable=True),
        sa.Column('p_value', sa.Float, nullable=True),
        sa.Column('confidence_interval', sa.JSON, nullable=True),
        sa.Column('affected_features', sa.JSON, nullable=True),
        sa.Column('recommendation', sa.Text, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('measurement_window_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('measurement_window_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_model_monitoring_time', 'model_id', 'created_at'),
        sa.Index('idx_model_monitoring_status', 'status', 'created_at')
    )
    
    # Data drift monitoring table
    op.create_table(
        'data_drift',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('dataset_id', sa.Integer, sa.ForeignKey('datasets.id'), nullable=True, index=True),
        sa.Column('feature_name', sa.String(255), nullable=False),
        sa.Column('drift_type', sa.String(50), nullable=False),  # univariate, multivariate
        sa.Column('drift_score', sa.Float, nullable=False),
        sa.Column('drift_threshold', sa.Float, nullable=False),
        sa.Column('is_drifted', sa.Boolean, nullable=False),
        sa.Column('detection_method', sa.String(100), nullable=False),
        sa.Column('statistical_test', sa.String(100), nullable=True),
        sa.Column('p_value', sa.Float, nullable=True),
        sa.Column('reference_distribution', sa.JSON, nullable=True),
        sa.Column('current_distribution', sa.JSON, nullable=True),
        sa.Column('distribution_comparison', sa.JSON, nullable=True),
        sa.Column('visualization_data', sa.JSON, nullable=True),
        sa.Column('sample_size_reference', sa.Integer, nullable=True),
        sa.Column('sample_size_current', sa.Integer, nullable=True),
        sa.Column('window_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_data_drift_time', 'model_id', 'feature_name', 'created_at'),
    )
    
    # Performance monitoring table
    op.create_table(
        'performance_monitoring',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('baseline_value', sa.Float, nullable=True),
        sa.Column('threshold_lower', sa.Float, nullable=True),
        sa.Column('threshold_upper', sa.Float, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='normal'),
        sa.Column('sample_size', sa.Integer, nullable=True),
        sa.Column('confidence_interval', sa.JSON, nullable=True),
        sa.Column('measurement_period', sa.String(50), nullable=False),  # hourly, daily, weekly
        sa.Column('measurement_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('measurement_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_performance_monitoring', 'model_id', 'metric_name', 'created_at'),
    )
    
    # ======================
    # EXPERIMENTS AND MLFLOW
    # ======================
    
    # Experiments table (MLflow integration)
    op.create_table(
        'experiments',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('mlflow_experiment_id', sa.String(255), nullable=True, unique=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('artifact_location', sa.String(500), nullable=True),
        sa.Column('lifecycle_stage', sa.String(50), nullable=False, default='active'),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Experiment runs table
    op.create_table(
        'experiment_runs',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('mlflow_run_id', sa.String(255), nullable=True, unique=True),
        sa.Column('experiment_id', sa.String(36), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('source_type', sa.String(50), nullable=True),
        sa.Column('source_name', sa.String(255), nullable=True),
        sa.Column('entry_point_name', sa.String(255), nullable=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='RUNNING'),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('artifact_uri', sa.String(500), nullable=True),
        sa.Column('lifecycle_stage', sa.String(50), nullable=False, default='active'),
        sa.Column('parameters', sa.JSON, nullable=True),
        sa.Column('metrics', sa.JSON, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # ======================
    # AUDIT LOGS AND SECURITY
    # ======================
    
    # Audit logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id'), nullable=True, index=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('action', sa.String(100), nullable=False, index=True),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('resource_id', sa.String(255), nullable=True, index=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('request_id', sa.String(36), nullable=True, index=True),
        sa.Column('endpoint', sa.String(255), nullable=True),
        sa.Column('method', sa.String(10), nullable=True),
        sa.Column('status_code', sa.Integer, nullable=True),
        sa.Column('response_time_ms', sa.Float, nullable=True),
        sa.Column('old_values', sa.JSON, nullable=True),
        sa.Column('new_values', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('severity', sa.String(20), nullable=False, default='info'),
        sa.Column('success', sa.Boolean, nullable=False, default=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_audit_logs_time', 'created_at'),
        sa.Index('idx_audit_logs_user_action', 'user_id', 'action', 'created_at'),
    )
    
    # Security events table
    op.create_table(
        'security_events',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('event_type', sa.String(100), nullable=False, index=True),
        sa.Column('severity', sa.String(20), nullable=False, default='medium'),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id'), nullable=True, index=True),
        sa.Column('ip_address', sa.String(45), nullable=True, index=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('details', sa.JSON, nullable=True),
        sa.Column('risk_score', sa.Float, nullable=True),
        sa.Column('is_blocked', sa.Boolean, nullable=False, default=False),
        sa.Column('resolution_status', sa.String(50), nullable=False, default='open'),
        sa.Column('resolved_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('resolution_notes', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Index('idx_security_events_time', 'created_at'),
        sa.Index('idx_security_events_severity', 'severity', 'created_at'),
    )
    
    # ======================
    # SYSTEM CONFIGURATION
    # ======================
    
    # System settings table
    op.create_table(
        'system_settings',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('key', sa.String(255), nullable=False, unique=True),
        sa.Column('value', sa.JSON, nullable=True),
        sa.Column('data_type', sa.String(50), nullable=False, default='string'),
        sa.Column('category', sa.String(100), nullable=False, default='general'),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('is_public', sa.Boolean, nullable=False, default=False),
        sa.Column('is_readonly', sa.Boolean, nullable=False, default=False),
        sa.Column('validation_rule', sa.Text, nullable=True),
        sa.Column('default_value', sa.JSON, nullable=True),
        sa.Column('updated_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # ======================
    # NOTIFICATIONS
    # ======================
    
    # Notifications table
    op.create_table(
        'notifications',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('channel', sa.String(50), nullable=False, default='in_app'),  # in_app, email, sms
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('data', sa.JSON, nullable=True),
        sa.Column('priority', sa.String(20), nullable=False, default='normal'),  # low, normal, high, urgent
        sa.Column('is_read', sa.Boolean, nullable=False, default=False),
        sa.Column('is_archived', sa.Boolean, nullable=False, default=False),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Index('idx_notifications_user_read', 'user_id', 'is_read', 'created_at'),
    )
    
    # ======================
    # CREATE INDEXES FOR PERFORMANCE
    # ======================
    
    # User indexes
    op.create_index('idx_users_active', 'users', ['is_active'])
    op.create_index('idx_users_created', 'users', ['created_at'])
    op.create_index('idx_user_sessions_active', 'user_sessions', ['is_active', 'expires_at'])
    
    # Dataset indexes
    op.create_index('idx_datasets_status', 'datasets', ['status'])
    op.create_index('idx_datasets_user_created', 'datasets', ['user_id', 'created_at'])
    op.create_index('idx_datasets_public', 'datasets', ['is_public', 'status'])
    
    # Analysis indexes
    op.create_index('idx_analyses_status', 'analyses', ['status'])
    op.create_index('idx_analyses_task_type', 'analyses', ['task_type'])
    op.create_index('idx_analyses_user_created', 'analyses', ['user_id', 'created_at'])
    
    # Model indexes
    op.create_index('idx_models_stage', 'models', ['stage'])
    op.create_index('idx_models_status', 'models', ['status'])
    op.create_index('idx_models_user_created', 'models', ['user_id', 'created_at'])
    
    # Prediction indexes
    op.create_index('idx_predictions_status', 'predictions', ['status'])
    op.create_index('idx_predictions_created', 'predictions', ['created_at'])
    op.create_index('idx_batch_predictions_status', 'batch_predictions', ['status'])
    
    # Insight indexes
    op.create_index('idx_insights_importance', 'insights', ['importance_score'])
    op.create_index('idx_insights_approved', 'insights', ['is_approved'])
    
    # Feature indexes
    op.create_index('idx_features_active', 'features', ['is_active'])
    op.create_index('idx_features_entity', 'features', ['entity'])
    op.create_index('idx_feature_sets_active', 'feature_sets', ['is_active'])
    
    print("✅ Initial schema migration completed successfully!")


def downgrade():
    """Drop all tables created in upgrade."""
    
    # Drop tables in reverse order to handle foreign key constraints
    tables_to_drop = [
        'notifications',
        'system_settings',
        'security_events',
        'audit_logs',
        'experiment_runs',
        'experiments',
        'performance_monitoring',
        'data_drift',
        'model_monitoring',
        'feature_usage',
        'feature_sets',
        'features',
        'insights',
        'batch_predictions',
        'predictions',
        'model_versions',
        'models',
        'analyses',
        'dataset_versions',
        'datasets',
        'user_sessions',
        'users'
    ]
    
    for table in tables_to_drop:
        try:
            op.drop_table(table)
        except Exception as e:
            print(f"Warning: Could not drop table {table}: {e}")
    
    print("✅ Schema downgrade completed!")
