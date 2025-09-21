"""
Enhanced Analysis Tables Migration for Auto-Analyst Platform

This migration adds advanced analysis capabilities and monitoring features
to support comprehensive ML workflows including:

- Advanced analysis configurations and templates
- Model comparison and benchmarking tables
- Hyperparameter tuning tracking
- Feature selection and engineering tracking  
- Cross-validation and validation strategies
- Model ensemble configurations
- Advanced monitoring and alerting
- Collaboration and sharing features
- Analysis pipeline versioning
- Performance optimization tracking

New Tables:
- analysis_templates: Reusable analysis configurations
- analysis_steps: Detailed step tracking for analyses
- model_comparisons: Model benchmarking and comparison results
- hyperparameter_tuning: HPO tracking and results
- feature_selections: Feature engineering and selection tracking
- validation_strategies: Cross-validation configuration and results
- model_ensembles: Ensemble model configurations
- analysis_collaborators: Team collaboration on analyses
- analysis_comments: Discussion and feedback on analyses
- pipeline_versions: Analysis pipeline versioning
- performance_profiles: Performance optimization tracking
- alert_rules: Custom monitoring and alert configurations
- analysis_metrics: Time-series metrics for analyses
- model_lineage: Model heritage and relationship tracking

Enhanced Features:
- Advanced analytics workflows
- Team collaboration capabilities
- Performance monitoring and optimization
- Custom alerting and notifications  
- Model lineage and governance
- Pipeline versioning and reproducibility

Revision ID: 002_add_analysis_tables
Revises: 001_initial_schema
Create Date: 2025-09-21 11:49:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import func
import uuid
from datetime import datetime

# revision identifiers, used by Alembic
revision = '002_add_analysis_tables'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade():
    """Add enhanced analysis tables and features."""
    
    # ======================
    # ANALYSIS TEMPLATES AND CONFIGURATIONS
    # ======================
    
    # Analysis templates for reusable configurations
    op.create_table(
        'analysis_templates',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('display_name', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('category', sa.String(100), nullable=False, default='custom'),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('is_builtin', sa.Boolean, nullable=False, default=False),
        sa.Column('is_public', sa.Boolean, nullable=False, default=False),
        sa.Column('complexity_level', sa.String(20), nullable=False, default='intermediate'),  # beginner, intermediate, advanced
        sa.Column('estimated_duration_minutes', sa.Integer, nullable=True),
        sa.Column('required_columns', sa.JSON, nullable=True),
        sa.Column('optional_columns', sa.JSON, nullable=True),
        sa.Column('data_requirements', sa.JSON, nullable=True),
        sa.Column('preprocessing_config', sa.JSON, nullable=True),
        sa.Column('model_selection_config', sa.JSON, nullable=True),
        sa.Column('hyperparameter_config', sa.JSON, nullable=True),
        sa.Column('validation_config', sa.JSON, nullable=True),
        sa.Column('evaluation_metrics', sa.JSON, nullable=True),
        sa.Column('visualization_config', sa.JSON, nullable=True),
        sa.Column('deployment_config', sa.JSON, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('usage_count', sa.Integer, nullable=False, default=0),
        sa.Column('success_rate', sa.Float, nullable=True),
        sa.Column('avg_performance', sa.Float, nullable=True),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('changelog', sa.Text, nullable=True),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Column('deprecated_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Analysis steps for detailed execution tracking
    op.create_table(
        'analysis_steps',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('step_name', sa.String(255), nullable=False),
        sa.Column('step_type', sa.String(100), nullable=False),
        sa.Column('step_order', sa.Integer, nullable=False),
        sa.Column('parent_step_id', sa.String(36), sa.ForeignKey('analysis_steps.id'), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('progress', sa.Float, nullable=False, default=0.0),
        sa.Column('input_data', sa.JSON, nullable=True),
        sa.Column('output_data', sa.JSON, nullable=True),
        sa.Column('parameters', sa.JSON, nullable=True),
        sa.Column('results', sa.JSON, nullable=True),
        sa.Column('metrics', sa.JSON, nullable=True),
        sa.Column('artifacts', sa.JSON, nullable=True),
        sa.Column('logs', sa.Text, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('execution_time_ms', sa.BigInteger, nullable=True),
        sa.Column('memory_usage_mb', sa.Float, nullable=True),
        sa.Column('cpu_usage_percent', sa.Float, nullable=True),
        sa.Column('resource_usage', sa.JSON, nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Index('idx_analysis_steps_order', 'analysis_id', 'step_order'),
        sa.Index('idx_analysis_steps_status', 'status', 'created_at'),
    )
    
    # Analysis configurations for different experiment variations
    op.create_table(
        'analysis_configs',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('config_name', sa.String(255), nullable=False),
        sa.Column('config_type', sa.String(100), nullable=False),  # preprocessing, model_selection, hyperparameters, etc.
        sa.Column('config_data', sa.JSON, nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.UniqueConstraint('analysis_id', 'config_name', name='uq_analysis_config')
    )
    
    # ======================
    # MODEL COMPARISON AND BENCHMARKING
    # ======================
    
    # Model comparisons for benchmarking different algorithms
    op.create_table(
        'model_comparisons',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('comparison_name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('models_compared', sa.JSON, nullable=False),  # List of model IDs
        sa.Column('comparison_metrics', sa.JSON, nullable=False),
        sa.Column('evaluation_strategy', sa.String(100), nullable=False),
        sa.Column('statistical_tests', sa.JSON, nullable=True),
        sa.Column('significance_results', sa.JSON, nullable=True),
        sa.Column('ranking', sa.JSON, nullable=True),
        sa.Column('recommendations', sa.Text, nullable=True),
        sa.Column('visualization_data', sa.JSON, nullable=True),
        sa.Column('comparison_matrix', sa.JSON, nullable=True),
        sa.Column('execution_time_comparison', sa.JSON, nullable=True),
        sa.Column('resource_usage_comparison', sa.JSON, nullable=True),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Individual model evaluation results
    op.create_table(
        'model_evaluations',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('comparison_id', sa.String(36), sa.ForeignKey('model_comparisons.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('evaluation_type', sa.String(100), nullable=False),  # train, validation, test, cross_validation
        sa.Column('dataset_split', sa.String(50), nullable=True),
        sa.Column('fold_number', sa.Integer, nullable=True),  # For cross-validation
        sa.Column('metrics', sa.JSON, nullable=False),
        sa.Column('confusion_matrix', sa.JSON, nullable=True),
        sa.Column('classification_report', sa.JSON, nullable=True),
        sa.Column('roc_curve_data', sa.JSON, nullable=True),
        sa.Column('feature_importance', sa.JSON, nullable=True),
        sa.Column('prediction_probabilities', sa.JSON, nullable=True),
        sa.Column('residuals_data', sa.JSON, nullable=True),
        sa.Column('learning_curves', sa.JSON, nullable=True),
        sa.Column('validation_curves', sa.JSON, nullable=True),
        sa.Column('evaluation_time_ms', sa.BigInteger, nullable=True),
        sa.Column('sample_size', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_model_evaluations_type', 'evaluation_type', 'created_at'),
    )
    
    # ======================
    # HYPERPARAMETER TUNING
    # ======================
    
    # Hyperparameter tuning experiments
    op.create_table(
        'hyperparameter_tuning',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('model_type', sa.String(100), nullable=False),
        sa.Column('tuning_method', sa.String(100), nullable=False),  # grid_search, random_search, bayesian, optuna
        sa.Column('search_space', sa.JSON, nullable=False),
        sa.Column('optimization_metric', sa.String(100), nullable=False),
        sa.Column('optimization_direction', sa.String(20), nullable=False),  # minimize, maximize
        sa.Column('n_trials', sa.Integer, nullable=True),
        sa.Column('timeout_minutes', sa.Integer, nullable=True),
        sa.Column('cv_folds', sa.Integer, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='running'),
        sa.Column('progress', sa.Float, nullable=False, default=0.0),
        sa.Column('current_trial', sa.Integer, nullable=False, default=0),
        sa.Column('best_trial', sa.JSON, nullable=True),
        sa.Column('best_parameters', sa.JSON, nullable=True),
        sa.Column('best_score', sa.Float, nullable=True),
        sa.Column('all_trials', sa.JSON, nullable=True),
        sa.Column('optimization_history', sa.JSON, nullable=True),
        sa.Column('parameter_importance', sa.JSON, nullable=True),
        sa.Column('early_stopping_enabled', sa.Boolean, nullable=False, default=False),
        sa.Column('pruning_enabled', sa.Boolean, nullable=False, default=False),
        sa.Column('study_name', sa.String(255), nullable=True),
        sa.Column('sampler_config', sa.JSON, nullable=True),
        sa.Column('pruner_config', sa.JSON, nullable=True),
        sa.Column('execution_time_total', sa.Float, nullable=True),
        sa.Column('resource_usage', sa.JSON, nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Individual hyperparameter trials
    op.create_table(
        'hyperparameter_trials',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('tuning_id', sa.String(36), sa.ForeignKey('hyperparameter_tuning.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('trial_number', sa.Integer, nullable=False),
        sa.Column('parameters', sa.JSON, nullable=False),
        sa.Column('objective_value', sa.Float, nullable=True),
        sa.Column('metrics', sa.JSON, nullable=True),
        sa.Column('status', sa.String(50), nullable=False),  # complete, pruned, failed
        sa.Column('execution_time_ms', sa.BigInteger, nullable=True),
        sa.Column('intermediate_values', sa.JSON, nullable=True),  # For pruning
        sa.Column('user_attrs', sa.JSON, nullable=True),
        sa.Column('system_attrs', sa.JSON, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Index('idx_hyperparameter_trials_number', 'tuning_id', 'trial_number'),
        sa.Index('idx_hyperparameter_trials_value', 'objective_value'),
    )
    
    # ======================
    # FEATURE ENGINEERING AND SELECTION
    # ======================
    
    # Feature selection experiments
    op.create_table(
        'feature_selections',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('selection_name', sa.String(255), nullable=False),
        sa.Column('selection_method', sa.String(100), nullable=False),
        sa.Column('original_features', sa.JSON, nullable=False),
        sa.Column('selected_features', sa.JSON, nullable=False),
        sa.Column('feature_scores', sa.JSON, nullable=True),
        sa.Column('feature_rankings', sa.JSON, nullable=True),
        sa.Column('selection_criteria', sa.JSON, nullable=True),
        sa.Column('threshold_value', sa.Float, nullable=True),
        sa.Column('n_features_selected', sa.Integer, nullable=False),
        sa.Column('selection_ratio', sa.Float, nullable=False),
        sa.Column('performance_before', sa.JSON, nullable=True),
        sa.Column('performance_after', sa.JSON, nullable=True),
        sa.Column('performance_improvement', sa.Float, nullable=True),
        sa.Column('training_time_before', sa.Float, nullable=True),
        sa.Column('training_time_after', sa.Float, nullable=True),
        sa.Column('time_improvement', sa.Float, nullable=True),
        sa.Column('dimensionality_reduction', sa.Float, nullable=False),
        sa.Column('correlation_analysis', sa.JSON, nullable=True),
        sa.Column('mutual_information', sa.JSON, nullable=True),
        sa.Column('statistical_tests', sa.JSON, nullable=True),
        sa.Column('visualization_data', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    
    # Feature engineering operations
    op.create_table(
        'feature_engineering',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('operation_name', sa.String(255), nullable=False),
        sa.Column('operation_type', sa.String(100), nullable=False),  # transformation, aggregation, interaction, etc.
        sa.Column('operation_order', sa.Integer, nullable=False),
        sa.Column('input_features', sa.JSON, nullable=False),
        sa.Column('output_features', sa.JSON, nullable=False),
        sa.Column('transformation_config', sa.JSON, nullable=False),
        sa.Column('transformation_code', sa.Text, nullable=True),
        sa.Column('validation_results', sa.JSON, nullable=True),
        sa.Column('feature_statistics', sa.JSON, nullable=True),
        sa.Column('quality_metrics', sa.JSON, nullable=True),
        sa.Column('impact_analysis', sa.JSON, nullable=True),
        sa.Column('is_automated', sa.Boolean, nullable=False, default=False),
        sa.Column('automation_confidence', sa.Float, nullable=True),
        sa.Column('execution_time_ms', sa.BigInteger, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_feature_engineering_order', 'analysis_id', 'operation_order'),
    )
    
    # ======================
    # VALIDATION STRATEGIES
    # ======================
    
    # Validation strategy configurations
    op.create_table(
        'validation_strategies',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('strategy_name', sa.String(255), nullable=False),
        sa.Column('strategy_type', sa.String(100), nullable=False),  # train_test_split, cross_validation, time_series_split
        sa.Column('strategy_config', sa.JSON, nullable=False),
        sa.Column('n_splits', sa.Integer, nullable=True),
        sa.Column('test_size', sa.Float, nullable=True),
        sa.Column('validation_size', sa.Float, nullable=True),
        sa.Column('shuffle', sa.Boolean, nullable=False, default=True),
        sa.Column('stratify', sa.Boolean, nullable=False, default=True),
        sa.Column('random_state', sa.Integer, nullable=True),
        sa.Column('group_column', sa.String(255), nullable=True),
        sa.Column('time_column', sa.String(255), nullable=True),
        sa.Column('gap_size', sa.Integer, nullable=True),  # For time series validation
        sa.Column('results', sa.JSON, nullable=True),
        sa.Column('fold_results', sa.JSON, nullable=True),
        sa.Column('aggregated_metrics', sa.JSON, nullable=True),
        sa.Column('variance_analysis', sa.JSON, nullable=True),
        sa.Column('stability_score', sa.Float, nullable=True),
        sa.Column('cross_validation_score', sa.Float, nullable=True),
        sa.Column('std_score', sa.Float, nullable=True),
        sa.Column('is_default', sa.Boolean, nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Cross-validation fold details
    op.create_table(
        'validation_folds',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('strategy_id', sa.String(36), sa.ForeignKey('validation_strategies.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('fold_number', sa.Integer, nullable=False),
        sa.Column('train_indices', sa.JSON, nullable=True),  # For reproducibility
        sa.Column('validation_indices', sa.JSON, nullable=True),
        sa.Column('train_size', sa.Integer, nullable=False),
        sa.Column('validation_size', sa.Integer, nullable=False),
        sa.Column('fold_metrics', sa.JSON, nullable=True),
        sa.Column('fold_predictions', sa.JSON, nullable=True),
        sa.Column('execution_time_ms', sa.BigInteger, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_validation_folds_number', 'strategy_id', 'fold_number'),
    )
    
    # ======================
    # MODEL ENSEMBLES
    # ======================
    
    # Ensemble model configurations
    op.create_table(
        'model_ensembles',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('ensemble_name', sa.String(255), nullable=False),
        sa.Column('ensemble_type', sa.String(100), nullable=False),  # voting, bagging, boosting, stacking
        sa.Column('base_models', sa.JSON, nullable=False),  # List of model IDs
        sa.Column('ensemble_config', sa.JSON, nullable=False),
        sa.Column('voting_strategy', sa.String(50), nullable=True),  # hard, soft
        sa.Column('meta_learner_config', sa.JSON, nullable=True),  # For stacking
        sa.Column('weight_optimization', sa.Boolean, nullable=False, default=False),
        sa.Column('weights', sa.JSON, nullable=True),
        sa.Column('diversity_metrics', sa.JSON, nullable=True),
        sa.Column('performance_metrics', sa.JSON, nullable=True),
        sa.Column('individual_performances', sa.JSON, nullable=True),
        sa.Column('ensemble_performance', sa.JSON, nullable=True),
        sa.Column('performance_improvement', sa.Float, nullable=True),
        sa.Column('training_time_total', sa.Float, nullable=True),
        sa.Column('prediction_time_avg', sa.Float, nullable=True),
        sa.Column('model_complexity', sa.JSON, nullable=True),
        sa.Column('interpretability_score', sa.Float, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='created'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Ensemble member details
    op.create_table(
        'ensemble_members',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('ensemble_id', sa.String(36), sa.ForeignKey('model_ensembles.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('member_order', sa.Integer, nullable=False),
        sa.Column('weight', sa.Float, nullable=False, default=1.0),
        sa.Column('contribution_score', sa.Float, nullable=True),
        sa.Column('diversity_score', sa.Float, nullable=True),
        sa.Column('individual_performance', sa.JSON, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('added_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.UniqueConstraint('ensemble_id', 'model_id', name='uq_ensemble_member')
    )
    
    # ======================
    # COLLABORATION AND SHARING
    # ======================
    
    # Analysis collaborators for team work
    op.create_table(
        'analysis_collaborators',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('role', sa.String(50), nullable=False, default='viewer'),  # owner, editor, viewer, commentor
        sa.Column('permissions', sa.JSON, nullable=True),
        sa.Column('invited_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('invitation_status', sa.String(50), nullable=False, default='pending'),  # pending, accepted, declined
        sa.Column('invitation_message', sa.Text, nullable=True),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('access_count', sa.Integer, nullable=False, default=0),
        sa.Column('invited_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('responded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('analysis_id', 'user_id', name='uq_analysis_collaborator')
    )
    
    # Comments and discussions on analyses
    op.create_table(
        'analysis_comments',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('parent_comment_id', sa.String(36), sa.ForeignKey('analysis_comments.id'), nullable=True),
        sa.Column('comment_type', sa.String(50), nullable=False, default='general'),  # general, suggestion, issue, question
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('mentioned_users', sa.JSON, nullable=True),
        sa.Column('attachments', sa.JSON, nullable=True),
        sa.Column('is_resolved', sa.Boolean, nullable=False, default=False),
        sa.Column('resolved_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('likes_count', sa.Integer, nullable=False, default=0),
        sa.Column('replies_count', sa.Integer, nullable=False, default=0),
        sa.Column('is_edited', sa.Boolean, nullable=False, default=False),
        sa.Column('edit_history', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Index('idx_analysis_comments_parent', 'parent_comment_id', 'created_at'),
    )
    
    # Comment reactions (likes, etc.)
    op.create_table(
        'comment_reactions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('comment_id', sa.String(36), sa.ForeignKey('analysis_comments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('reaction_type', sa.String(50), nullable=False, default='like'),  # like, dislike, heart, etc.
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.UniqueConstraint('comment_id', 'user_id', 'reaction_type', name='uq_comment_reaction')
    )
    
    # ======================
    # PIPELINE VERSIONING
    # ======================
    
    # Analysis pipeline versions for reproducibility
    op.create_table(
        'pipeline_versions',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('version_name', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('pipeline_config', sa.JSON, nullable=False),
        sa.Column('dependencies', sa.JSON, nullable=True),
        sa.Column('environment_config', sa.JSON, nullable=True),
        sa.Column('data_hash', sa.String(64), nullable=True),
        sa.Column('code_hash', sa.String(64), nullable=True),
        sa.Column('config_hash', sa.String(64), nullable=True),
        sa.Column('reproducibility_score', sa.Float, nullable=True),
        sa.Column('differences_from_previous', sa.JSON, nullable=True),
        sa.Column('performance_comparison', sa.JSON, nullable=True),
        sa.Column('is_baseline', sa.Boolean, nullable=False, default=False),
        sa.Column('is_production', sa.Boolean, nullable=False, default=False),
        sa.Column('approval_status', sa.String(50), nullable=False, default='pending'),
        sa.Column('approved_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rollback_from_version', sa.String(36), sa.ForeignKey('pipeline_versions.id'), nullable=True),
        sa.Column('created_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.UniqueConstraint('analysis_id', 'version', name='uq_pipeline_version')
    )
    
    # ======================
    # PERFORMANCE OPTIMIZATION
    # ======================
    
    # Performance profiling results
    op.create_table(
        'performance_profiles',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('step_id', sa.String(36), sa.ForeignKey('analysis_steps.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('profile_type', sa.String(50), nullable=False),  # cpu, memory, io, network
        sa.Column('measurement_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('measurement_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('duration_ms', sa.BigInteger, nullable=False),
        sa.Column('cpu_usage_avg', sa.Float, nullable=True),
        sa.Column('cpu_usage_max', sa.Float, nullable=True),
        sa.Column('memory_usage_avg', sa.Float, nullable=True),
        sa.Column('memory_usage_max', sa.Float, nullable=True),
        sa.Column('memory_peak_mb', sa.Float, nullable=True),
        sa.Column('disk_io_read_mb', sa.Float, nullable=True),
        sa.Column('disk_io_write_mb', sa.Float, nullable=True),
        sa.Column('network_io_mb', sa.Float, nullable=True),
        sa.Column('gpu_usage_avg', sa.Float, nullable=True),
        sa.Column('gpu_memory_usage_mb', sa.Float, nullable=True),
        sa.Column('detailed_profile', sa.JSON, nullable=True),
        sa.Column('bottlenecks', sa.JSON, nullable=True),
        sa.Column('optimization_suggestions', sa.JSON, nullable=True),
        sa.Column('performance_score', sa.Float, nullable=True),
        sa.Column('efficiency_score', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_performance_profiles_time', 'analysis_id', 'measurement_start'),
    )
    
    # ======================
    # ADVANCED MONITORING AND ALERTING
    # ======================
    
    # Custom alert rules for analyses
    op.create_table(
        'alert_rules',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('rule_name', sa.String(255), nullable=False),
        sa.Column('rule_type', sa.String(100), nullable=False),  # threshold, anomaly, drift, performance
        sa.Column('metric_name', sa.String(255), nullable=False),
        sa.Column('condition', sa.String(100), nullable=False),  # gt, lt, gte, lte, eq, ne
        sa.Column('threshold_value', sa.Float, nullable=True),
        sa.Column('threshold_range', sa.JSON, nullable=True),  # For range-based alerts
        sa.Column('aggregation_window', sa.String(100), nullable=True),  # 5m, 1h, 1d
        sa.Column('aggregation_function', sa.String(50), nullable=True),  # avg, max, min, sum, count
        sa.Column('evaluation_frequency', sa.String(100), nullable=False, default='5m'),
        sa.Column('alert_channels', sa.JSON, nullable=True),  # email, slack, webhook
        sa.Column('severity', sa.String(20), nullable=False, default='warning'),
        sa.Column('is_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('cooldown_period', sa.Integer, nullable=False, default=300),  # seconds
        sa.Column('max_alerts_per_day', sa.Integer, nullable=False, default=10),
        sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('trigger_count', sa.Integer, nullable=False, default=0),
        sa.Column('false_positive_count', sa.Integer, nullable=False, default=0),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    
    # Alert instances/triggers
    op.create_table(
        'alert_instances',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('rule_id', sa.String(36), sa.ForeignKey('alert_rules.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('alert_status', sa.String(50), nullable=False, default='active'),  # active, resolved, acknowledged, silenced
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('triggered_value', sa.Float, nullable=True),
        sa.Column('threshold_value', sa.Float, nullable=True),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('context_data', sa.JSON, nullable=True),
        sa.Column('affected_metrics', sa.JSON, nullable=True),
        sa.Column('recommendation', sa.Text, nullable=True),
        sa.Column('notification_channels', sa.JSON, nullable=True),
        sa.Column('notification_status', sa.JSON, nullable=True),
        sa.Column('acknowledged_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_notes', sa.Text, nullable=True),
        sa.Column('is_false_positive', sa.Boolean, nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Index('idx_alert_instances_status', 'alert_status', 'created_at'),
    )
    
    # ======================
    # TIME-SERIES METRICS
    # ======================
    
    # Analysis metrics over time
    op.create_table(
        'analysis_metrics',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('analysis_id', sa.String(36), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('metric_name', sa.String(255), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('metric_type', sa.String(100), nullable=False),  # accuracy, precision, recall, f1, etc.
        sa.Column('metric_category', sa.String(100), nullable=False),  # performance, drift, quality, resource
        sa.Column('aggregation_period', sa.String(50), nullable=False, default='point'),  # point, hourly, daily, weekly
        sa.Column('measurement_window_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('measurement_window_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sample_size', sa.Integer, nullable=True),
        sa.Column('confidence_interval', sa.JSON, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_analysis_metrics_time', 'analysis_id', 'metric_name', 'created_at'),
        sa.Index('idx_analysis_metrics_category', 'metric_category', 'created_at'),
    )
    
    # ======================
    # MODEL LINEAGE AND GOVERNANCE
    # ======================
    
    # Model lineage tracking
    op.create_table(
        'model_lineage',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('parent_model_id', sa.String(36), sa.ForeignKey('models.id'), nullable=True, index=True),
        sa.Column('relationship_type', sa.String(100), nullable=False),  # derived_from, retrained_from, ensemble_of, transfer_from
        sa.Column('lineage_metadata', sa.JSON, nullable=True),
        sa.Column('inheritance_ratio', sa.Float, nullable=True),  # How much is inherited from parent
        sa.Column('differences', sa.JSON, nullable=True),
        sa.Column('performance_comparison', sa.JSON, nullable=True),
        sa.Column('data_lineage', sa.JSON, nullable=True),  # Data sources and transformations
        sa.Column('code_lineage', sa.JSON, nullable=True),  # Code changes and versions
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Index('idx_model_lineage_parent', 'parent_model_id', 'created_at'),
    )
    
    # Model governance and approval workflows
    op.create_table(
        'model_approvals',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('model_id', sa.String(36), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('approval_type', sa.String(100), nullable=False),  # deployment, production, retirement
        sa.Column('current_stage', sa.String(50), nullable=False),
        sa.Column('target_stage', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),  # pending, approved, rejected, cancelled
        sa.Column('requested_by', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('assigned_to', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('approval_criteria', sa.JSON, nullable=True),
        sa.Column('performance_requirements', sa.JSON, nullable=True),
        sa.Column('compliance_checklist', sa.JSON, nullable=True),
        sa.Column('risk_assessment', sa.JSON, nullable=True),
        sa.Column('business_justification', sa.Text, nullable=True),
        sa.Column('technical_review', sa.JSON, nullable=True),
        sa.Column('reviewer_comments', sa.Text, nullable=True),
        sa.Column('approval_conditions', sa.JSON, nullable=True),
        sa.Column('approved_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('approval_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejection_reason', sa.Text, nullable=True),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('priority', sa.String(20), nullable=False, default='medium'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
        sa.Index('idx_model_approvals_status', 'status', 'created_at'),
    )
    
    # ======================
    # CREATE ADDITIONAL INDEXES
    # ======================
    
    # Analysis template indexes
    op.create_index('idx_analysis_templates_category', 'analysis_templates', ['category', 'task_type'])
    op.create_index('idx_analysis_templates_public', 'analysis_templates', ['is_public', 'usage_count'])
    op.create_index('idx_analysis_templates_builtin', 'analysis_templates', ['is_builtin'])
    
    # Model comparison indexes
    op.create_index('idx_model_comparisons_analysis', 'model_comparisons', ['analysis_id', 'created_at'])
    
    # Hyperparameter tuning indexes
    op.create_index('idx_hyperparameter_tuning_status', 'hyperparameter_tuning', ['status', 'created_at'])
    op.create_index('idx_hyperparameter_tuning_method', 'hyperparameter_tuning', ['tuning_method'])
    
    # Feature selection indexes
    op.create_index('idx_feature_selections_method', 'feature_selections', ['selection_method'])
    
    # Collaboration indexes
    op.create_index('idx_analysis_collaborators_user', 'analysis_collaborators', ['user_id', 'role'])
    op.create_index('idx_analysis_collaborators_role', 'analysis_collaborators', ['role', 'invitation_status'])
    
    # Alert rule indexes
    op.create_index('idx_alert_rules_enabled', 'alert_rules', ['is_enabled', 'rule_type'])
    op.create_index('idx_alert_rules_user', 'alert_rules', ['user_id', 'is_enabled'])
    
    # Performance profile indexes
    op.create_index('idx_performance_profiles_type', 'performance_profiles', ['profile_type', 'created_at'])
    
    print("✅ Enhanced analysis tables migration completed successfully!")


def downgrade():
    """Drop enhanced analysis tables."""
    
    # Drop tables in reverse order
    enhanced_tables_to_drop = [
        'model_approvals',
        'model_lineage', 
        'analysis_metrics',
        'alert_instances',
        'alert_rules',
        'performance_profiles',
        'pipeline_versions',
        'comment_reactions',
        'analysis_comments',
        'analysis_collaborators',
        'ensemble_members',
        'model_ensembles',
        'validation_folds',
        'validation_strategies',
        'feature_engineering',
        'feature_selections',
        'hyperparameter_trials',
        'hyperparameter_tuning',
        'model_evaluations',
        'model_comparisons',
        'analysis_configs',
        'analysis_steps',
        'analysis_templates'
    ]
    
    for table in enhanced_tables_to_drop:
        try:
            op.drop_table(table)
        except Exception as e:
            print(f"Warning: Could not drop table {table}: {e}")
    
    print("✅ Enhanced analysis tables downgrade completed!")
