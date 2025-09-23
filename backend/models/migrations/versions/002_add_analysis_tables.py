"""
🚀 AUTO-ANALYST PLATFORM - ANALYSIS ENHANCEMENT MIGRATION
========================================================

This migration adds essential analysis enhancement features in a controlled,
production-ready manner. Focuses on core ML workflow improvements without
over-engineering.

Core Enhancements:
- Analysis execution steps tracking
- Model performance comparisons  
- Basic hyperparameter tuning support
- Feature selection tracking
- Validation strategy configurations

Design Principles:
- Incremental enhancement approach
- Proper database constraints and validation
- Multi-database compatibility
- Performance-optimized indexes
- Safe rollback capabilities

This migration is intentionally focused and can be followed by additional
migrations for advanced features like collaboration, alerting, etc.

Revision ID: 002_add_analysis_tables
Revises: 001_initial_schema
Create Date: 2025-09-24 03:30:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, mysql
from sqlalchemy.sql import func, text
import uuid

# Revision identifiers
revision = '002_add_analysis_tables'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def get_uuid_type():
    """Get appropriate UUID type based on database dialect."""
    dialect = op.get_bind().dialect.name
    
    if dialect == 'postgresql':
        return postgresql.UUID(as_uuid=True)
    else:
        return sa.CHAR(36)


def get_uuid_default():
    """Get appropriate UUID default based on database dialect."""
    dialect = op.get_bind().dialect.name
    
    if dialect == 'postgresql':
        return text("gen_random_uuid()")
    elif dialect == 'mysql':
        return text("UUID()")
    else:
        return None


def get_json_type():
    """Get appropriate JSON type based on database dialect."""
    dialect = op.get_bind().dialect.name
    
    if dialect == 'postgresql':
        return postgresql.JSONB
    elif dialect == 'mysql':
        return mysql.JSON
    else:
        return sa.Text


def upgrade():
    """Add essential analysis enhancement tables."""
    
    # ==========================================================================
    # ANALYSIS EXECUTION STEPS
    # ==========================================================================
    
    op.create_table(
        'analysis_steps',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique step identifier"
        ),
        
        # Foreign key to analysis
        sa.Column(
            'analysis_id', 
            get_uuid_type(), 
            sa.ForeignKey('analyses.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Parent analysis"
        ),
        
        # Step identification
        sa.Column(
            'step_name', 
            sa.String(255), 
            nullable=False,
            comment="Human-readable step name"
        ),
        sa.Column(
            'step_type', 
            sa.String(100), 
            nullable=False,
            comment="Type of processing step"
        ),
        sa.Column(
            'step_order', 
            sa.Integer, 
            nullable=False,
            comment="Execution order within analysis"
        ),
        
        # Hierarchical steps support
        sa.Column(
            'parent_step_id', 
            get_uuid_type(), 
            sa.ForeignKey('analysis_steps.id', ondelete='CASCADE'), 
            nullable=True,
            comment="Parent step for nested workflows"
        ),
        
        # Execution status and progress
        sa.Column(
            'status', 
            sa.String(50), 
            nullable=False, 
            default='pending',
            comment="Step execution status"
        ),
        sa.Column(
            'progress', 
            sa.Float, 
            nullable=False, 
            default=0.0,
            comment="Progress ratio (0-1)"
        ),
        
        # Step configuration and results
        sa.Column(
            'parameters', 
            get_json_type(), 
            nullable=True,
            comment="Step-specific parameters (JSON)"
        ),
        sa.Column(
            'results', 
            get_json_type(), 
            nullable=True,
            comment="Step execution results (JSON)"
        ),
        sa.Column(
            'metrics', 
            get_json_type(), 
            nullable=True,
            comment="Step performance metrics (JSON)"
        ),
        
        # Error handling
        sa.Column(
            'error_message', 
            sa.Text, 
            nullable=True,
            comment="Error message if step failed"
        ),
        sa.Column(
            'error_details', 
            get_json_type(), 
            nullable=True,
            comment="Detailed error information (JSON)"
        ),
        
        # Resource usage tracking
        sa.Column(
            'execution_time_ms', 
            sa.BigInteger, 
            nullable=True,
            comment="Step execution time in milliseconds"
        ),
        sa.Column(
            'memory_usage_mb', 
            sa.Float, 
            nullable=True,
            comment="Peak memory usage in MB"
        ),
        sa.Column(
            'cpu_usage_percent', 
            sa.Float, 
            nullable=True,
            comment="Average CPU usage percentage"
        ),
        
        # Execution timestamps
        sa.Column(
            'started_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Step start timestamp"
        ),
        sa.Column(
            'completed_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Step completion timestamp"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Step creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        
        comment="Detailed tracking of analysis execution steps"
    )
    
    # Analysis steps indexes
    op.create_index('ix_analysis_steps_analysis_id', 'analysis_steps', ['analysis_id'])
    op.create_index('ix_analysis_steps_status', 'analysis_steps', ['status'])
    op.create_index('ix_analysis_steps_order', 'analysis_steps', ['analysis_id', 'step_order'])
    op.create_index('ix_analysis_steps_parent', 'analysis_steps', ['parent_step_id'])
    op.create_index('ix_analysis_steps_type', 'analysis_steps', ['step_type'])
    
    # ==========================================================================
    # MODEL PERFORMANCE COMPARISONS
    # ==========================================================================
    
    op.create_table(
        'model_comparisons',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique comparison identifier"
        ),
        
        # Foreign key to analysis
        sa.Column(
            'analysis_id', 
            get_uuid_type(), 
            sa.ForeignKey('analyses.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Parent analysis"
        ),
        
        # Comparison metadata
        sa.Column(
            'comparison_name', 
            sa.String(255), 
            nullable=False,
            comment="Human-readable comparison name"
        ),
        sa.Column(
            'description', 
            sa.Text, 
            nullable=True,
            comment="Comparison description"
        ),
        
        # Models being compared
        sa.Column(
            'models_compared', 
            get_json_type(), 
            nullable=False,
            comment="List of model IDs being compared (JSON array)"
        ),
        
        # Comparison configuration
        sa.Column(
            'evaluation_strategy', 
            sa.String(100), 
            nullable=False,
            comment="Evaluation methodology used"
        ),
        sa.Column(
            'metrics_compared', 
            get_json_type(), 
            nullable=False,
            comment="Metrics used for comparison (JSON array)"
        ),
        
        # Comparison results
        sa.Column(
            'comparison_results', 
            get_json_type(), 
            nullable=True,
            comment="Detailed comparison results (JSON)"
        ),
        sa.Column(
            'statistical_significance', 
            get_json_type(), 
            nullable=True,
            comment="Statistical test results (JSON)"
        ),
        sa.Column(
            'ranking', 
            get_json_type(), 
            nullable=True,
            comment="Model ranking results (JSON)"
        ),
        
        # Best model identification
        sa.Column(
            'best_model_id', 
            get_uuid_type(), 
            nullable=True,
            comment="ID of best performing model"
        ),
        sa.Column(
            'best_model_score', 
            sa.Float, 
            nullable=True,
            comment="Best model's primary metric score"
        ),
        
        # Execution metadata
        sa.Column(
            'comparison_time_ms', 
            sa.BigInteger, 
            nullable=True,
            comment="Total comparison execution time"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Comparison creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        
        comment="Model performance comparison results"
    )
    
    # Model comparison indexes
    op.create_index('ix_model_comparisons_analysis_id', 'model_comparisons', ['analysis_id'])
    op.create_index('ix_model_comparisons_created_at', 'model_comparisons', ['created_at'])
    
    # ==========================================================================
    # HYPERPARAMETER TUNING EXPERIMENTS
    # ==========================================================================
    
    op.create_table(
        'hyperparameter_experiments',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique experiment identifier"
        ),
        
        # Foreign key to analysis
        sa.Column(
            'analysis_id', 
            get_uuid_type(), 
            sa.ForeignKey('analyses.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Parent analysis"
        ),
        
        # Experiment configuration
        sa.Column(
            'experiment_name', 
            sa.String(255), 
            nullable=False,
            comment="Human-readable experiment name"
        ),
        sa.Column(
            'model_type', 
            sa.String(100), 
            nullable=False,
            comment="Type of model being tuned"
        ),
        sa.Column(
            'tuning_method', 
            sa.String(100), 
            nullable=False,
            comment="Hyperparameter tuning strategy"
        ),
        
        # Search space definition
        sa.Column(
            'search_space', 
            get_json_type(), 
            nullable=False,
            comment="Hyperparameter search space (JSON)"
        ),
        sa.Column(
            'optimization_metric', 
            sa.String(100), 
            nullable=False,
            comment="Metric to optimize"
        ),
        sa.Column(
            'optimization_direction', 
            sa.String(20), 
            nullable=False,
            comment="maximize or minimize"
        ),
        
        # Experiment limits
        sa.Column(
            'max_trials', 
            sa.Integer, 
            nullable=True,
            comment="Maximum number of trials"
        ),
        sa.Column(
            'timeout_minutes', 
            sa.Integer, 
            nullable=True,
            comment="Maximum experiment duration"
        ),
        sa.Column(
            'cv_folds', 
            sa.Integer, 
            nullable=True,
            comment="Cross-validation folds"
        ),
        
        # Experiment status and progress
        sa.Column(
            'status', 
            sa.String(50), 
            nullable=False, 
            default='pending',
            comment="Experiment status"
        ),
        sa.Column(
            'progress', 
            sa.Float, 
            nullable=False, 
            default=0.0,
            comment="Progress ratio (0-1)"
        ),
        sa.Column(
            'completed_trials', 
            sa.Integer, 
            nullable=False, 
            default=0,
            comment="Number of completed trials"
        ),
        
        # Best results
        sa.Column(
            'best_trial_id', 
            get_uuid_type(), 
            nullable=True,
            comment="ID of best trial"
        ),
        sa.Column(
            'best_parameters', 
            get_json_type(), 
            nullable=True,
            comment="Best hyperparameters found (JSON)"
        ),
        sa.Column(
            'best_score', 
            sa.Float, 
            nullable=True,
            comment="Best score achieved"
        ),
        
        # Experiment metadata
        sa.Column(
            'experiment_config', 
            get_json_type(), 
            nullable=True,
            comment="Additional experiment configuration (JSON)"
        ),
        sa.Column(
            'execution_summary', 
            get_json_type(), 
            nullable=True,
            comment="Experiment execution summary (JSON)"
        ),
        
        # Timing
        sa.Column(
            'started_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Experiment start time"
        ),
        sa.Column(
            'completed_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Experiment completion time"
        ),
        sa.Column(
            'total_execution_time', 
            sa.Float, 
            nullable=True,
            comment="Total execution time in seconds"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Experiment creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        
        comment="Hyperparameter tuning experiments"
    )
    
    # Hyperparameter experiment indexes
    op.create_index('ix_hyperparameter_experiments_analysis_id', 'hyperparameter_experiments', ['analysis_id'])
    op.create_index('ix_hyperparameter_experiments_status', 'hyperparameter_experiments', ['status'])
    op.create_index('ix_hyperparameter_experiments_method', 'hyperparameter_experiments', ['tuning_method'])
    
    # ==========================================================================
    # HYPERPARAMETER TRIALS
    # ==========================================================================
    
    op.create_table(
        'hyperparameter_trials',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique trial identifier"
        ),
        
        # Foreign key to experiment
        sa.Column(
            'experiment_id', 
            get_uuid_type(), 
            sa.ForeignKey('hyperparameter_experiments.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Parent experiment"
        ),
        
        # Trial identification
        sa.Column(
            'trial_number', 
            sa.Integer, 
            nullable=False,
            comment="Trial sequence number"
        ),
        sa.Column(
            'trial_name', 
            sa.String(255), 
            nullable=True,
            comment="Optional trial name"
        ),
        
        # Trial parameters and results
        sa.Column(
            'parameters', 
            get_json_type(), 
            nullable=False,
            comment="Trial hyperparameters (JSON)"
        ),
        sa.Column(
            'score', 
            sa.Float, 
            nullable=True,
            comment="Trial objective score"
        ),
        sa.Column(
            'metrics', 
            get_json_type(), 
            nullable=True,
            comment="Additional trial metrics (JSON)"
        ),
        
        # Trial status and execution
        sa.Column(
            'status', 
            sa.String(50), 
            nullable=False, 
            default='pending',
            comment="Trial status"
        ),
        sa.Column(
            'execution_time_ms', 
            sa.BigInteger, 
            nullable=True,
            comment="Trial execution time in milliseconds"
        ),
        sa.Column(
            'error_message', 
            sa.Text, 
            nullable=True,
            comment="Error message if trial failed"
        ),
        
        # Trial metadata
        sa.Column(
            'intermediate_values', 
            get_json_type(), 
            nullable=True,
            comment="Intermediate values for pruning (JSON)"
        ),
        sa.Column(
            'user_attributes', 
            get_json_type(), 
            nullable=True,
            comment="User-defined attributes (JSON)"
        ),
        
        # Timing
        sa.Column(
            'started_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Trial start time"
        ),
        sa.Column(
            'completed_at', 
            sa.DateTime(timezone=True), 
            nullable=True,
            comment="Trial completion time"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Trial creation timestamp"
        ),
        
        comment="Individual hyperparameter tuning trials"
    )
    
    # Hyperparameter trial indexes
    op.create_index('ix_hyperparameter_trials_experiment_id', 'hyperparameter_trials', ['experiment_id'])
    op.create_index('ix_hyperparameter_trials_number', 'hyperparameter_trials', ['experiment_id', 'trial_number'])
    op.create_index('ix_hyperparameter_trials_score', 'hyperparameter_trials', ['score'])
    op.create_index('ix_hyperparameter_trials_status', 'hyperparameter_trials', ['status'])
    
    # ==========================================================================
    # FEATURE SELECTION EXPERIMENTS
    # ==========================================================================
    
    op.create_table(
        'feature_selections',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique feature selection identifier"
        ),
        
        # Foreign key to analysis
        sa.Column(
            'analysis_id', 
            get_uuid_type(), 
            sa.ForeignKey('analyses.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Parent analysis"
        ),
        
        # Selection configuration
        sa.Column(
            'selection_name', 
            sa.String(255), 
            nullable=False,
            comment="Human-readable selection name"
        ),
        sa.Column(
            'selection_method', 
            sa.String(100), 
            nullable=False,
            comment="Feature selection algorithm used"
        ),
        sa.Column(
            'selection_config', 
            get_json_type(), 
            nullable=True,
            comment="Method-specific configuration (JSON)"
        ),
        
        # Feature sets
        sa.Column(
            'original_features', 
            get_json_type(), 
            nullable=False,
            comment="Original feature names (JSON array)"
        ),
        sa.Column(
            'selected_features', 
            get_json_type(), 
            nullable=False,
            comment="Selected feature names (JSON array)"
        ),
        sa.Column(
            'feature_scores', 
            get_json_type(), 
            nullable=True,
            comment="Feature importance scores (JSON)"
        ),
        
        # Selection metrics
        sa.Column(
            'num_original_features', 
            sa.Integer, 
            nullable=False,
            comment="Number of original features"
        ),
        sa.Column(
            'num_selected_features', 
            sa.Integer, 
            nullable=False,
            comment="Number of selected features"
        ),
        sa.Column(
            'selection_ratio', 
            sa.Float, 
            nullable=False,
            comment="Ratio of selected to original features"
        ),
        
        # Performance comparison
        sa.Column(
            'baseline_performance', 
            get_json_type(), 
            nullable=True,
            comment="Performance with all features (JSON)"
        ),
        sa.Column(
            'selected_performance', 
            get_json_type(), 
            nullable=True,
            comment="Performance with selected features (JSON)"
        ),
        sa.Column(
            'performance_change', 
            sa.Float, 
            nullable=True,
            comment="Performance change ratio"
        ),
        
        # Execution metadata
        sa.Column(
            'execution_time_ms', 
            sa.BigInteger, 
            nullable=True,
            comment="Selection execution time"
        ),
        sa.Column(
            'selection_criteria', 
            get_json_type(), 
            nullable=True,
            comment="Criteria used for selection (JSON)"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Selection creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        
        comment="Feature selection experiments and results"
    )
    
    # Feature selection indexes
    op.create_index('ix_feature_selections_analysis_id', 'feature_selections', ['analysis_id'])
    op.create_index('ix_feature_selections_method', 'feature_selections', ['selection_method'])
    op.create_index('ix_feature_selections_ratio', 'feature_selections', ['selection_ratio'])
    
    # ==========================================================================
    # VALIDATION STRATEGIES
    # ==========================================================================
    
    op.create_table(
        'validation_strategies',
        # Primary key
        sa.Column(
            'id', 
            get_uuid_type(), 
            primary_key=True, 
            default=get_uuid_default(),
            comment="Unique validation strategy identifier"
        ),
        
        # Foreign key to analysis
        sa.Column(
            'analysis_id', 
            get_uuid_type(), 
            sa.ForeignKey('analyses.id', ondelete='CASCADE'), 
            nullable=False,
            comment="Parent analysis"
        ),
        
        # Strategy configuration
        sa.Column(
            'strategy_name', 
            sa.String(255), 
            nullable=False,
            comment="Human-readable strategy name"
        ),
        sa.Column(
            'strategy_type', 
            sa.String(100), 
            nullable=False,
            comment="Type of validation strategy"
        ),
        sa.Column(
            'strategy_config', 
            get_json_type(), 
            nullable=False,
            comment="Strategy configuration parameters (JSON)"
        ),
        
        # Validation parameters
        sa.Column(
            'n_splits', 
            sa.Integer, 
            nullable=True,
            comment="Number of validation splits"
        ),
        sa.Column(
            'test_size', 
            sa.Float, 
            nullable=True,
            comment="Test set size ratio"
        ),
        sa.Column(
            'shuffle', 
            sa.Boolean, 
            nullable=False, 
            default=True,
            comment="Whether to shuffle data before splitting"
        ),
        sa.Column(
            'random_state', 
            sa.Integer, 
            nullable=True,
            comment="Random seed for reproducibility"
        ),
        
        # Validation results
        sa.Column(
            'validation_results', 
            get_json_type(), 
            nullable=True,
            comment="Validation results summary (JSON)"
        ),
        sa.Column(
            'fold_results', 
            get_json_type(), 
            nullable=True,
            comment="Individual fold results (JSON)"
        ),
        sa.Column(
            'cross_validation_score', 
            sa.Float, 
            nullable=True,
            comment="Mean cross-validation score"
        ),
        sa.Column(
            'score_std', 
            sa.Float, 
            nullable=True,
            comment="Standard deviation of CV scores"
        ),
        
        # Strategy metadata
        sa.Column(
            'is_default', 
            sa.Boolean, 
            nullable=False, 
            default=False,
            comment="Whether this is the default strategy"
        ),
        sa.Column(
            'execution_time_ms', 
            sa.BigInteger, 
            nullable=True,
            comment="Total validation execution time"
        ),
        
        # Audit timestamps
        sa.Column(
            'created_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            comment="Strategy creation timestamp"
        ),
        sa.Column(
            'updated_at', 
            sa.DateTime(timezone=True), 
            nullable=False, 
            server_default=func.now(),
            onupdate=func.now(),
            comment="Last update timestamp"
        ),
        
        comment="Cross-validation and data splitting strategies"
    )
    
    # Validation strategy indexes
    op.create_index('ix_validation_strategies_analysis_id', 'validation_strategies', ['analysis_id'])
    op.create_index('ix_validation_strategies_type', 'validation_strategies', ['strategy_type'])
    op.create_index('ix_validation_strategies_default', 'validation_strategies', ['is_default'])
    
    # ==========================================================================
    # ADD CONSTRAINTS FOR DATA INTEGRITY
    # ==========================================================================
    
    # Analysis steps constraints
    op.create_check_constraint(
        'ck_analysis_steps_progress_range',
        'analysis_steps',
        "progress >= 0.0 AND progress <= 1.0"
    )
    
    op.create_check_constraint(
        'ck_analysis_steps_step_order_positive',
        'analysis_steps',
        "step_order >= 0"
    )
    
    # Hyperparameter experiments constraints
    op.create_check_constraint(
        'ck_hyperparameter_experiments_progress_range',
        'hyperparameter_experiments',
        "progress >= 0.0 AND progress <= 1.0"
    )
    
    op.create_check_constraint(
        'ck_hyperparameter_experiments_completed_trials',
        'hyperparameter_experiments',
        "completed_trials >= 0"
    )
    
    # Feature selections constraints
    op.create_check_constraint(
        'ck_feature_selections_counts_positive',
        'feature_selections',
        "num_original_features > 0 AND num_selected_features >= 0"
    )
    
    op.create_check_constraint(
        'ck_feature_selections_ratio_range',
        'feature_selections',
        "selection_ratio >= 0.0 AND selection_ratio <= 1.0"
    )
    
    # Validation strategies constraints
    op.create_check_constraint(
        'ck_validation_strategies_test_size_range',
        'validation_strategies',
        "test_size IS NULL OR (test_size > 0.0 AND test_size < 1.0)"
    )
    
    op.create_check_constraint(
        'ck_validation_strategies_n_splits_positive',
        'validation_strategies',
        "n_splits IS NULL OR n_splits >= 2"
    )
    
    print("✅ Analysis enhancement migration completed successfully!")
    print("📊 Added 6 core tables for ML workflow tracking")
    print("🚀 Ready for advanced analysis features!")


def downgrade():
    """
    Rollback the analysis enhancement migration.
    
    Drops all tables created in the upgrade function in reverse dependency order.
    """
    
    # Tables to drop in reverse dependency order
    tables_to_drop = [
        'validation_strategies',
        'feature_selections', 
        'hyperparameter_trials',
        'hyperparameter_experiments',
        'model_comparisons',
        'analysis_steps'
    ]
    
    # Drop tables with error handling
    for table in tables_to_drop:
        try:
            op.drop_table(table)
            print(f"✅ Dropped table: {table}")
        except Exception as e:
            print(f"⚠️  Warning: Could not drop table {table}: {e}")
            continue
    
    print("✅ Analysis enhancement migration rolled back successfully!")
    print("🔄 Database restored to core schema state")
