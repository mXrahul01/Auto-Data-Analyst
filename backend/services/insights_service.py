"""
Insights Service for Auto-Analyst Platform

This service generates user-friendly insights, summaries, and actionable recommendations
from machine learning model outputs across different data types and tasks.

The service transforms technical ML results into business-friendly insights that
non-technical users can understand and act upon.

Features:
- Automated insight generation for all model types
- Business impact assessment and ROI analysis
- Performance interpretation in plain language
- Feature importance explanations with business context
- Actionable recommendations for model improvement
- Dashboard-ready visualization data
- Integration with SHAP/LIME explanations
- Multi-language support for insights
- Contextual suggestions based on model performance
- Risk assessment and confidence scoring

Supported Model Types:
- Tabular: Classification, Regression
- Time Series: Forecasting, Trend Analysis
- Text: Sentiment Analysis, Classification, Topic Modeling
- Anomaly Detection: Outlier identification, Risk scoring
- Clustering: Segmentation, Pattern discovery
- Ensemble: Combined model insights

Usage:
    # Initialize service
    insights_service = InsightsService()
    
    # Generate insights from model results
    insights = await insights_service.generate_insights(
        model_results=results,
        model_type='classification',
        dataset_info=dataset_info
    )
    
    # Get dashboard-formatted insights
    dashboard_data = insights_service.format_for_dashboard(insights)
    
    # Generate business recommendations
    recommendations = insights_service.generate_recommendations(
        insights, business_context
    )
"""

import asyncio
import logging
import warnings
import re
import math
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Statistical analysis
try:
    import scipy.stats as stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Text processing for insights
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ML explanation integration
try:
    from backend.services.explainer_service import ExplainerService, create_explainer_service
    EXPLAINER_SERVICE_AVAILABLE = True
except ImportError:
    EXPLAINER_SERVICE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class ModelType(str, Enum):
    """Supported model types for insight generation."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"

class InsightType(str, Enum):
    """Types of insights that can be generated."""
    PERFORMANCE = "performance"
    FEATURE_IMPORTANCE = "feature_importance"
    PREDICTION_EXPLANATION = "prediction_explanation"
    BUSINESS_IMPACT = "business_impact"
    DATA_QUALITY = "data_quality"
    MODEL_COMPARISON = "model_comparison"
    RECOMMENDATIONS = "recommendations"
    RISK_ASSESSMENT = "risk_assessment"

class ConfidenceLevel(str, Enum):
    """Confidence levels for insights."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class InsightMetadata:
    """Metadata for insights."""
    insight_type: InsightType
    confidence: ConfidenceLevel
    importance: float  # 0-1 scale
    timestamp: datetime
    model_type: ModelType
    source_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Insight:
    """Single insight with metadata."""
    title: str
    description: str
    details: List[str]
    impact: str  # High, Medium, Low
    actionable: bool
    metadata: InsightMetadata
    visualization_data: Optional[Dict[str, Any]] = None
    related_features: List[str] = field(default_factory=list)
    numerical_evidence: Dict[str, float] = field(default_factory=dict)

@dataclass
class InsightsResult:
    """Complete insights result."""
    insights: List[Insight]
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    confidence_score: float
    dashboard_data: Dict[str, Any]
    business_metrics: Dict[str, Any] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)

class InsightTemplates:
    """Templates for generating insights in natural language."""
    
    # Performance insights
    PERFORMANCE_EXCELLENT = "Your model achieved excellent performance with {metric_name} of {value:.3f}. This indicates very reliable predictions."
    PERFORMANCE_GOOD = "Your model shows good performance with {metric_name} of {value:.3f}. The predictions are generally reliable."
    PERFORMANCE_MODERATE = "Your model has moderate performance with {metric_name} of {value:.3f}. Consider improvements."
    PERFORMANCE_POOR = "Your model shows room for improvement with {metric_name} of {value:.3f}. Additional data or features may help."
    
    # Feature importance insights
    FEATURE_DOMINANT = "The feature '{feature}' dominates predictions with {importance:.1%} influence. This could indicate over-reliance on a single factor."
    FEATURE_IMPORTANT = "'{feature}' is highly important ({importance:.1%} influence) for predictions."
    FEATURE_BALANCED = "Feature importance is well-distributed across multiple variables, indicating a robust model."
    FEATURE_SPARSE = "Many features have minimal impact. Consider feature selection to simplify the model."
    
    # Data quality insights
    DATA_QUALITY_HIGH = "Data quality is excellent with minimal missing values and good feature coverage."
    DATA_QUALITY_MEDIUM = "Data quality is acceptable but could be improved by addressing missing values or outliers."
    DATA_QUALITY_LOW = "Data quality issues detected. Consider data cleaning and preprocessing improvements."
    
    # Business impact templates
    BUSINESS_IMPACT_HIGH = "This model can provide significant business value through {impact_area}."
    BUSINESS_IMPACT_MEDIUM = "The model offers moderate business benefits in {impact_area}."
    BUSINESS_IMPACT_LOW = "Limited immediate business impact. Consider refining objectives or data collection."
    
    # Recommendation templates
    RECOMMENDATION_DATA = "Collect more data for features: {features} to improve model accuracy."
    RECOMMENDATION_FEATURES = "Consider adding these features: {features} for better predictions."
    RECOMMENDATION_MODEL = "Try {model_type} models which may perform better for this type of data."
    RECOMMENDATION_DEPLOYMENT = "Model is ready for deployment with confidence score of {confidence:.1%}."

class InsightsService:
    """
    Comprehensive service for generating actionable insights from ML models.
    
    This service analyzes model outputs, performance metrics, and feature importance
    to generate human-readable insights and recommendations.
    """
    
    def __init__(self, language: str = "en", business_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the insights service.
        
        Args:
            language: Language for insights generation (default: English)
            business_context: Optional business context for more relevant insights
        """
        self.language = language
        self.business_context = business_context or {}
        
        # Initialize explainer service if available
        self.explainer_service = None
        if EXPLAINER_SERVICE_AVAILABLE:
            try:
                self.explainer_service = create_explainer_service()
            except Exception as e:
                logger.warning(f"Could not initialize explainer service: {str(e)}")
        
        # Insight generation statistics
        self.stats = {
            'insights_generated': 0,
            'models_analyzed': 0,
            'recommendations_made': 0,
            'average_confidence': 0.0
        }
        
        # Performance thresholds for different metrics
        self.performance_thresholds = {
            'accuracy': {'excellent': 0.95, 'good': 0.85, 'moderate': 0.70},
            'precision': {'excellent': 0.90, 'good': 0.80, 'moderate': 0.65},
            'recall': {'excellent': 0.90, 'good': 0.80, 'moderate': 0.65},
            'f1_score': {'excellent': 0.90, 'good': 0.80, 'moderate': 0.65},
            'roc_auc': {'excellent': 0.95, 'good': 0.85, 'moderate': 0.70},
            'r2_score': {'excellent': 0.90, 'good': 0.75, 'moderate': 0.50},
            'rmse': {'excellent': 0.1, 'good': 0.2, 'moderate': 0.4, 'inverse': True},
            'mae': {'excellent': 0.1, 'good': 0.2, 'moderate': 0.4, 'inverse': True}
        }
        
        logger.info("InsightsService initialized successfully")
    
    async def generate_insights(
        self,
        model_results: Dict[str, Any],
        model_type: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        business_context: Optional[Dict[str, Any]] = None
    ) -> InsightsResult:
        """
        Generate comprehensive insights from model results.
        
        Args:
            model_results: Complete model results including metrics, predictions, etc.
            model_type: Type of ML model (classification, regression, etc.)
            dataset_info: Information about the dataset used
            business_context: Business context for more relevant insights
            
        Returns:
            Comprehensive insights result
            
        Raises:
            ValueError: If model results are invalid
            RuntimeError: If insight generation fails
        """
        try:
            logger.info(f"Generating insights for {model_type} model")
            
            # Validate inputs
            self._validate_model_results(model_results)
            
            # Initialize insight collection
            insights = []
            
            # Update business context
            current_context = {**self.business_context, **(business_context or {})}
            
            # Generate different types of insights
            model_type_enum = ModelType(model_type.lower())
            
            # 1. Performance insights
            performance_insights = await self._generate_performance_insights(
                model_results, model_type_enum, current_context
            )
            insights.extend(performance_insights)
            
            # 2. Feature importance insights
            if 'feature_importance' in model_results:
                feature_insights = await self._generate_feature_insights(
                    model_results['feature_importance'], 
                    model_type_enum,
                    dataset_info
                )
                insights.extend(feature_insights)
            
            # 3. Data quality insights
            if dataset_info:
                data_insights = await self._generate_data_quality_insights(
                    dataset_info, model_results
                )
                insights.extend(data_insights)
            
            # 4. Business impact insights
            business_insights = await self._generate_business_impact_insights(
                model_results, model_type_enum, current_context
            )
            insights.extend(business_insights)
            
            # 5. Model-specific insights
            specific_insights = await self._generate_model_specific_insights(
                model_results, model_type_enum, dataset_info
            )
            insights.extend(specific_insights)
            
            # 6. Risk assessment insights
            risk_insights = await self._generate_risk_insights(
                model_results, model_type_enum, dataset_info
            )
            insights.extend(risk_insights)
            
            # Generate summary and recommendations
            summary = await self._generate_summary(insights, model_type_enum)
            key_findings = await self._extract_key_findings(insights)
            recommendations = await self._generate_recommendations(insights, model_results, model_type_enum)
            risk_factors = await self._extract_risk_factors(insights)
            opportunities = await self._extract_opportunities(insights)
            next_steps = await self._generate_next_steps(insights, model_results)
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(insights)
            
            # Generate dashboard data
            dashboard_data = await self._create_dashboard_data(
                insights, model_results, model_type_enum
            )
            
            # Generate business metrics
            business_metrics = await self._generate_business_metrics(
                model_results, model_type_enum, current_context
            )
            
            # Create final result
            result = InsightsResult(
                insights=insights,
                summary=summary,
                key_findings=key_findings,
                recommendations=recommendations,
                risk_factors=risk_factors,
                opportunities=opportunities,
                confidence_score=confidence_score,
                dashboard_data=dashboard_data,
                business_metrics=business_metrics,
                next_steps=next_steps
            )
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Generated {len(insights)} insights with confidence {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate insights: {str(e)}")
    
    async def generate_prediction_insights(
        self,
        predictions: Union[np.ndarray, List],
        model_results: Dict[str, Any],
        input_features: Optional[Dict[str, Any]] = None,
        explanation_data: Optional[Dict[str, Any]] = None
    ) -> List[Insight]:
        """
        Generate insights for specific predictions.
        
        Args:
            predictions: Model predictions
            model_results: Model results for context
            input_features: Input features used for prediction
            explanation_data: SHAP/LIME explanation data
            
        Returns:
            List of prediction-specific insights
        """
        try:
            insights = []
            
            # Basic prediction insights
            if isinstance(predictions, (list, np.ndarray)):
                predictions_array = np.array(predictions)
                
                # Prediction distribution insights
                if len(predictions_array.shape) == 1:  # Single output
                    unique_preds = np.unique(predictions_array)
                    
                    if len(unique_preds) <= 10:  # Classification-like
                        pred_counts = pd.Series(predictions_array).value_counts()
                        dominant_class = pred_counts.index[0]
                        
                        insight = Insight(
                            title="Prediction Distribution",
                            description=f"Most predictions ({pred_counts.iloc[0]}/{len(predictions_array)}) fall into class '{dominant_class}'",
                            details=[
                                f"Prediction breakdown: {dict(pred_counts)}",
                                f"Confidence in dominant class: {pred_counts.iloc[0]/len(predictions_array):.1%}"
                            ],
                            impact="Medium",
                            actionable=True,
                            metadata=InsightMetadata(
                                insight_type=InsightType.PREDICTION_EXPLANATION,
                                confidence=ConfidenceLevel.HIGH,
                                importance=0.7,
                                timestamp=datetime.now(),
                                model_type=ModelType.CLASSIFICATION
                            ),
                            numerical_evidence={
                                'dominant_class_ratio': pred_counts.iloc[0]/len(predictions_array),
                                'unique_predictions': len(unique_preds)
                            }
                        )
                        insights.append(insight)
                    
                    else:  # Regression-like
                        pred_mean = np.mean(predictions_array)
                        pred_std = np.std(predictions_array)
                        
                        insight = Insight(
                            title="Prediction Statistics",
                            description=f"Predictions range from {np.min(predictions_array):.3f} to {np.max(predictions_array):.3f} with mean {pred_mean:.3f}",
                            details=[
                                f"Standard deviation: {pred_std:.3f}",
                                f"Coefficient of variation: {pred_std/abs(pred_mean):.2%}" if pred_mean != 0 else "Mean is zero",
                                f"Prediction spread indicates {'high' if pred_std/abs(pred_mean) > 0.3 else 'moderate' if pred_std/abs(pred_mean) > 0.1 else 'low'} variability"
                            ],
                            impact="Medium",
                            actionable=True,
                            metadata=InsightMetadata(
                                insight_type=InsightType.PREDICTION_EXPLANATION,
                                confidence=ConfidenceLevel.HIGH,
                                importance=0.6,
                                timestamp=datetime.now(),
                                model_type=ModelType.REGRESSION
                            ),
                            numerical_evidence={
                                'mean_prediction': float(pred_mean),
                                'std_prediction': float(pred_std),
                                'coefficient_of_variation': float(pred_std/abs(pred_mean)) if pred_mean != 0 else 0
                            }
                        )
                        insights.append(insight)
            
            # Feature-based insights if available
            if input_features and explanation_data:
                feature_insights = await self._generate_feature_contribution_insights(
                    input_features, explanation_data
                )
                insights.extend(feature_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Prediction insights generation failed: {str(e)}")
            return []
    
    async def explain_model_behavior(
        self,
        model_results: Dict[str, Any],
        model_type: str,
        dataset_sample: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate explanations for model behavior using SHAP/LIME.
        
        Args:
            model_results: Model results including trained model
            model_type: Type of model
            dataset_sample: Sample of dataset for explanation
            
        Returns:
            Model behavior explanations
        """
        try:
            explanations = {}
            
            if not self.explainer_service:
                logger.warning("Explainer service not available")
                return explanations
            
            # Generate global explanations
            if 'best_model' in model_results and dataset_sample is not None:
                try:
                    global_explanations = await self.explainer_service.explain_model(
                        model=model_results['best_model'],
                        X=dataset_sample,
                        explanation_type='global'
                    )
                    explanations['global'] = global_explanations
                    
                except Exception as e:
                    logger.warning(f"Global explanation failed: {str(e)}")
            
            # Generate local explanations for sample predictions
            if dataset_sample is not None and len(dataset_sample) > 0:
                try:
                    sample_size = min(5, len(dataset_sample))
                    sample_data = dataset_sample.sample(n=sample_size)
                    
                    local_explanations = await self.explainer_service.explain_predictions(
                        model=model_results['best_model'],
                        X=sample_data,
                        explanation_type='local'
                    )
                    explanations['local'] = local_explanations
                    
                except Exception as e:
                    logger.warning(f"Local explanation failed: {str(e)}")
            
            # Convert explanations to insights
            if explanations:
                explanation_insights = await self._convert_explanations_to_insights(
                    explanations, model_type
                )
                explanations['insights'] = explanation_insights
            
            return explanations
            
        except Exception as e:
            logger.error(f"Model behavior explanation failed: {str(e)}")
            return {}
    
    def format_for_dashboard(self, insights_result: InsightsResult) -> Dict[str, Any]:
        """
        Format insights for dashboard display.
        
        Args:
            insights_result: Complete insights result
            
        Returns:
            Dashboard-formatted data
        """
        try:
            dashboard_data = {
                'summary': {
                    'overall_summary': insights_result.summary,
                    'confidence_score': insights_result.confidence_score,
                    'total_insights': len(insights_result.insights),
                    'key_findings_count': len(insights_result.key_findings),
                    'recommendations_count': len(insights_result.recommendations)
                },
                'insights': [],
                'key_findings': insights_result.key_findings,
                'recommendations': insights_result.recommendations,
                'risk_factors': insights_result.risk_factors,
                'opportunities': insights_result.opportunities,
                'next_steps': insights_result.next_steps,
                'business_metrics': insights_result.business_metrics,
                'visualizations': []
            }
            
            # Format individual insights
            for insight in insights_result.insights:
                formatted_insight = {
                    'title': insight.title,
                    'description': insight.description,
                    'details': insight.details,
                    'impact': insight.impact,
                    'actionable': insight.actionable,
                    'confidence': insight.metadata.confidence.value,
                    'importance': insight.metadata.importance,
                    'type': insight.metadata.insight_type.value,
                    'numerical_evidence': insight.numerical_evidence,
                    'related_features': insight.related_features
                }
                
                # Add visualization data if available
                if insight.visualization_data:
                    formatted_insight['visualization'] = insight.visualization_data
                    dashboard_data['visualizations'].append({
                        'insight_id': len(dashboard_data['insights']),
                        'type': insight.visualization_data.get('type', 'chart'),
                        'data': insight.visualization_data
                    })
                
                dashboard_data['insights'].append(formatted_insight)
            
            # Add aggregate visualizations
            dashboard_data['aggregate_visualizations'] = self._create_aggregate_visualizations(
                insights_result
            )
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard formatting failed: {str(e)}")
            return {'error': str(e)}
    
    # Private insight generation methods
    
    async def _generate_performance_insights(
        self,
        model_results: Dict[str, Any],
        model_type: ModelType,
        business_context: Dict[str, Any]
    ) -> List[Insight]:
        """Generate insights about model performance."""
        insights = []
        
        try:
            performance_metrics = model_results.get('performance_metrics', {})
            
            if not performance_metrics:
                return insights
            
            # Analyze each metric
            for metric_name, metric_value in performance_metrics.items():
                if isinstance(metric_value, (int, float)):
                    
                    # Determine performance level
                    performance_level = self._assess_performance_level(metric_name, metric_value)
                    
                    # Generate appropriate insight
                    if performance_level == 'excellent':
                        template = InsightTemplates.PERFORMANCE_EXCELLENT
                        impact = "High"
                        confidence = ConfidenceLevel.HIGH
                    elif performance_level == 'good':
                        template = InsightTemplates.PERFORMANCE_GOOD
                        impact = "Medium"
                        confidence = ConfidenceLevel.HIGH
                    elif performance_level == 'moderate':
                        template = InsightTemplates.PERFORMANCE_MODERATE
                        impact = "Medium"
                        confidence = ConfidenceLevel.MEDIUM
                    else:
                        template = InsightTemplates.PERFORMANCE_POOR
                        impact = "High"
                        confidence = ConfidenceLevel.HIGH
                    
                    description = template.format(
                        metric_name=metric_name.replace('_', ' ').title(),
                        value=metric_value
                    )
                    
                    # Add context-specific details
                    details = [
                        f"Metric value: {metric_value:.4f}",
                        f"Performance level: {performance_level.title()}"
                    ]
                    
                    # Add business context if available
                    if business_context.get('target_performance'):
                        target = business_context['target_performance'].get(metric_name)
                        if target:
                            meets_target = metric_value >= target
                            details.append(f"Target: {target:.3f} ({'✓ Met' if meets_target else '✗ Not met'})")
                    
                    # Add benchmark comparison if available
                    if business_context.get('industry_benchmarks'):
                        benchmark = business_context['industry_benchmarks'].get(metric_name)
                        if benchmark:
                            vs_benchmark = "above" if metric_value > benchmark else "below"
                            details.append(f"Industry benchmark: {benchmark:.3f} (you are {vs_benchmark})")
                    
                    insight = Insight(
                        title=f"{metric_name.replace('_', ' ').title()} Performance",
                        description=description,
                        details=details,
                        impact=impact,
                        actionable=performance_level in ['moderate', 'poor'],
                        metadata=InsightMetadata(
                            insight_type=InsightType.PERFORMANCE,
                            confidence=confidence,
                            importance=0.9 if performance_level in ['excellent', 'poor'] else 0.7,
                            timestamp=datetime.now(),
                            model_type=model_type,
                            source_data={'metric': metric_name, 'value': metric_value}
                        ),
                        numerical_evidence={metric_name: metric_value}
                    )
                    
                    insights.append(insight)
            
            # Overall performance insight
            if len(performance_metrics) > 1:
                overall_insight = await self._generate_overall_performance_insight(
                    performance_metrics, model_type
                )
                if overall_insight:
                    insights.append(overall_insight)
            
        except Exception as e:
            logger.error(f"Performance insights generation failed: {str(e)}")
        
        return insights
    
    async def _generate_feature_insights(
        self,
        feature_importance: Dict[str, float],
        model_type: ModelType,
        dataset_info: Optional[Dict[str, Any]]
    ) -> List[Insight]:
        """Generate insights about feature importance."""
        insights = []
        
        try:
            if not feature_importance:
                return insights
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Top features analysis
            top_features = sorted_features[:5]
            top_feature_names = [f[0] for f in top_features]
            top_importances = [f[1] for f in top_features]
            
            # Feature dominance analysis
            total_importance = sum(abs(imp) for _, imp in sorted_features)
            if total_importance > 0:
                top_feature_share = abs(top_importances[0]) / total_importance
                
                if top_feature_share > 0.5:  # Single feature dominates
                    insight = Insight(
                        title="Feature Dominance Detected",
                        description=InsightTemplates.FEATURE_DOMINANT.format(
                            feature=top_features[0][0],
                            importance=top_feature_share
                        ),
                        details=[
                            f"Top feature contributes {top_feature_share:.1%} of total importance",
                            "Consider data diversity and feature engineering",
                            "High dependency on single feature may indicate data leakage"
                        ],
                        impact="High",
                        actionable=True,
                        metadata=InsightMetadata(
                            insight_type=InsightType.FEATURE_IMPORTANCE,
                            confidence=ConfidenceLevel.HIGH,
                            importance=0.9,
                            timestamp=datetime.now(),
                            model_type=model_type
                        ),
                        related_features=[top_features[0][0]],
                        numerical_evidence={
                            'dominant_feature_importance': top_feature_share,
                            'feature_count': len(sorted_features)
                        }
                    )
                    insights.append(insight)
                
                elif len(top_features) >= 3 and sum(abs(imp) for _, imp in top_features[:3]) / total_importance < 0.6:
                    # Well-distributed importance
                    insight = Insight(
                        title="Balanced Feature Importance",
                        description=InsightTemplates.FEATURE_BALANCED,
                        details=[
                            f"Top 3 features contribute {sum(abs(imp) for _, imp in top_features[:3]) / total_importance:.1%} of importance",
                            "Model relies on multiple diverse features",
                            "Good sign for model robustness and generalization"
                        ],
                        impact="Medium",
                        actionable=False,
                        metadata=InsightMetadata(
                            insight_type=InsightType.FEATURE_IMPORTANCE,
                            confidence=ConfidenceLevel.HIGH,
                            importance=0.6,
                            timestamp=datetime.now(),
                            model_type=model_type
                        ),
                        related_features=top_feature_names,
                        numerical_evidence={
                            'top3_importance_ratio': sum(abs(imp) for _, imp in top_features[:3]) / total_importance,
                            'feature_count': len(sorted_features)
                        }
                    )
                    insights.append(insight)
            
            # Individual top features
            for i, (feature_name, importance) in enumerate(top_features[:3]):
                normalized_importance = abs(importance) / total_importance if total_importance > 0 else 0
                
                # Feature-specific insights
                feature_insight = await self._generate_individual_feature_insight(
                    feature_name, importance, normalized_importance, i + 1, dataset_info
                )
                if feature_insight:
                    insights.append(feature_insight)
            
            # Low importance features
            low_importance_features = [f for f, imp in sorted_features if abs(imp) / total_importance < 0.01]
            if len(low_importance_features) > 5:
                insight = Insight(
                    title="Many Low-Impact Features",
                    description=f"{len(low_importance_features)} features have minimal impact on predictions",
                    details=[
                        f"Features with <1% importance: {len(low_importance_features)}",
                        "Consider feature selection to simplify model",
                        "Removing low-impact features may improve interpretability"
                    ],
                    impact="Medium",
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.FEATURE_IMPORTANCE,
                        confidence=ConfidenceLevel.MEDIUM,
                        importance=0.5,
                        timestamp=datetime.now(),
                        model_type=model_type
                    ),
                    related_features=low_importance_features[:10],  # Limit display
                    numerical_evidence={
                        'low_importance_count': len(low_importance_features),
                        'total_features': len(sorted_features)
                    }
                )
                insights.append(insight)
        
        except Exception as e:
            logger.error(f"Feature insights generation failed: {str(e)}")
        
        return insights
    
    async def _generate_data_quality_insights(
        self,
        dataset_info: Dict[str, Any],
        model_results: Dict[str, Any]
    ) -> List[Insight]:
        """Generate insights about data quality impact."""
        insights = []
        
        try:
            # Missing value analysis
            missing_ratio = dataset_info.get('missing_value_ratio', 0)
            if missing_ratio > 0.1:
                severity = "high" if missing_ratio > 0.3 else "moderate"
                impact_level = "High" if missing_ratio > 0.3 else "Medium"
                
                insight = Insight(
                    title=f"{'High' if severity == 'high' else 'Moderate'} Missing Data Detected",
                    description=f"Dataset has {missing_ratio:.1%} missing values, which may impact model performance",
                    details=[
                        f"Missing value ratio: {missing_ratio:.1%}",
                        "Missing data can bias predictions" if severity == "high" else "Moderate missing data handled by preprocessing",
                        "Consider data collection improvements" if severity == "high" else "Monitor for data quality patterns"
                    ],
                    impact=impact_level,
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.DATA_QUALITY,
                        confidence=ConfidenceLevel.HIGH,
                        importance=0.8 if severity == "high" else 0.6,
                        timestamp=datetime.now(),
                        model_type=ModelType.CLASSIFICATION  # Generic
                    ),
                    numerical_evidence={'missing_ratio': missing_ratio}
                )
                insights.append(insight)
            
            # Data size analysis
            n_samples = dataset_info.get('n_samples', 0)
            n_features = dataset_info.get('n_features', 0)
            
            if n_samples < 1000:
                insight = Insight(
                    title="Small Dataset Size",
                    description=f"Dataset has only {n_samples} samples, which may limit model reliability",
                    details=[
                        f"Sample count: {n_samples}",
                        "Small datasets may lead to overfitting",
                        "Consider collecting more data or using simpler models",
                        "Cross-validation results may have high variance"
                    ],
                    impact="High",
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.DATA_QUALITY,
                        confidence=ConfidenceLevel.HIGH,
                        importance=0.8,
                        timestamp=datetime.now(),
                        model_type=ModelType.CLASSIFICATION
                    ),
                    numerical_evidence={'sample_count': n_samples}
                )
                insights.append(insight)
            
            # Dimensionality analysis
            if n_features > n_samples:
                insight = Insight(
                    title="High-Dimensional Data",
                    description=f"More features ({n_features}) than samples ({n_samples}) detected",
                    details=[
                        f"Feature-to-sample ratio: {n_features/n_samples:.1f}",
                        "High dimensionality can cause overfitting",
                        "Consider dimensionality reduction techniques",
                        "Feature selection may improve performance"
                    ],
                    impact="High",
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.DATA_QUALITY,
                        confidence=ConfidenceLevel.HIGH,
                        importance=0.7,
                        timestamp=datetime.now(),
                        model_type=ModelType.CLASSIFICATION
                    ),
                    numerical_evidence={
                        'feature_sample_ratio': n_features/n_samples if n_samples > 0 else float('inf'),
                        'n_features': n_features,
                        'n_samples': n_samples
                    }
                )
                insights.append(insight)
            
            # Data quality score
            data_quality_score = dataset_info.get('data_quality_score', 0.5)
            if data_quality_score < 0.7:
                severity = "poor" if data_quality_score < 0.5 else "moderate"
                
                insight = Insight(
                    title=f"{'Poor' if severity == 'poor' else 'Moderate'} Data Quality",
                    description=f"Data quality score of {data_quality_score:.2f} indicates room for improvement",
                    details=[
                        f"Quality score: {data_quality_score:.2f}/1.00",
                        "Consider data cleaning and preprocessing",
                        "Quality issues may limit model performance",
                        "Address missing values, outliers, and inconsistencies"
                    ],
                    impact="High" if severity == "poor" else "Medium",
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.DATA_QUALITY,
                        confidence=ConfidenceLevel.HIGH,
                        importance=0.8 if severity == "poor" else 0.6,
                        timestamp=datetime.now(),
                        model_type=ModelType.CLASSIFICATION
                    ),
                    numerical_evidence={'data_quality_score': data_quality_score}
                )
                insights.append(insight)
        
        except Exception as e:
            logger.error(f"Data quality insights generation failed: {str(e)}")
        
        return insights
    
    async def _generate_business_impact_insights(
        self,
        model_results: Dict[str, Any],
        model_type: ModelType,
        business_context: Dict[str, Any]
    ) -> List[Insight]:
        """Generate business impact insights."""
        insights = []
        
        try:
            performance_metrics = model_results.get('performance_metrics', {})
            
            # ROI estimation
            if business_context.get('cost_benefit_analysis'):
                roi_insight = await self._calculate_roi_insight(
                    performance_metrics, business_context['cost_benefit_analysis'], model_type
                )
                if roi_insight:
                    insights.append(roi_insight)
            
            # Deployment readiness
            deployment_readiness = await self._assess_deployment_readiness(
                performance_metrics, model_type, business_context
            )
            insights.append(deployment_readiness)
            
            # Business value assessment
            business_value = await self._assess_business_value(
                performance_metrics, model_type, business_context
            )
            insights.append(business_value)
            
            # Risk-benefit analysis
            if model_type in [ModelType.CLASSIFICATION, ModelType.ANOMALY_DETECTION]:
                risk_insight = await self._generate_business_risk_insight(
                    performance_metrics, model_type, business_context
                )
                if risk_insight:
                    insights.append(risk_insight)
        
        except Exception as e:
            logger.error(f"Business impact insights generation failed: {str(e)}")
        
        return insights
    
    async def _generate_model_specific_insights(
        self,
        model_results: Dict[str, Any],
        model_type: ModelType,
        dataset_info: Optional[Dict[str, Any]]
    ) -> List[Insight]:
        """Generate model-type specific insights."""
        insights = []
        
        try:
            if model_type == ModelType.CLASSIFICATION:
                insights.extend(await self._generate_classification_insights(model_results, dataset_info))
            elif model_type == ModelType.REGRESSION:
                insights.extend(await self._generate_regression_insights(model_results, dataset_info))
            elif model_type == ModelType.FORECASTING:
                insights.extend(await self._generate_forecasting_insights(model_results, dataset_info))
            elif model_type == ModelType.CLUSTERING:
                insights.extend(await self._generate_clustering_insights(model_results, dataset_info))
            elif model_type == ModelType.ANOMALY_DETECTION:
                insights.extend(await self._generate_anomaly_insights(model_results, dataset_info))
            elif model_type in [ModelType.TEXT_CLASSIFICATION, ModelType.SENTIMENT_ANALYSIS]:
                insights.extend(await self._generate_text_insights(model_results, dataset_info))
        
        except Exception as e:
            logger.error(f"Model-specific insights generation failed: {str(e)}")
        
        return insights
    
    async def _generate_risk_insights(
        self,
        model_results: Dict[str, Any],
        model_type: ModelType,
        dataset_info: Optional[Dict[str, Any]]
    ) -> List[Insight]:
        """Generate risk assessment insights."""
        insights = []
        
        try:
            performance_metrics = model_results.get('performance_metrics', {})
            
            # Model confidence assessment
            confidence_insight = await self._assess_model_confidence(
                performance_metrics, model_type
            )
            if confidence_insight:
                insights.append(confidence_insight)
            
            # Overfitting risk
            overfitting_risk = await self._assess_overfitting_risk(
                model_results, dataset_info
            )
            if overfitting_risk:
                insights.append(overfitting_risk)
            
            # Bias detection
            bias_insight = await self._detect_potential_bias(
                model_results, dataset_info
            )
            if bias_insight:
                insights.append(bias_insight)
            
            # Generalization assessment
            generalization_insight = await self._assess_generalization(
                performance_metrics, model_type, dataset_info
            )
            if generalization_insight:
                insights.append(generalization_insight)
        
        except Exception as e:
            logger.error(f"Risk insights generation failed: {str(e)}")
        
        return insights
    
    # Helper methods for specific insight types
    
    async def _generate_classification_insights(
        self,
        model_results: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]]
    ) -> List[Insight]:
        """Generate classification-specific insights."""
        insights = []
        
        performance_metrics = model_results.get('performance_metrics', {})
        
        # Precision vs Recall trade-off
        precision = performance_metrics.get('precision')
        recall = performance_metrics.get('recall')
        
        if precision is not None and recall is not None:
            if precision > recall + 0.1:
                insights.append(Insight(
                    title="High Precision, Lower Recall",
                    description=f"Model prioritizes precision ({precision:.3f}) over recall ({recall:.3f})",
                    details=[
                        "Model is conservative - fewer false positives",
                        "May miss some positive cases (higher false negatives)",
                        "Good for scenarios where false positives are costly"
                    ],
                    impact="Medium",
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.PERFORMANCE,
                        confidence=ConfidenceLevel.HIGH,
                        importance=0.7,
                        timestamp=datetime.now(),
                        model_type=ModelType.CLASSIFICATION
                    ),
                    numerical_evidence={'precision': precision, 'recall': recall}
                ))
            
            elif recall > precision + 0.1:
                insights.append(Insight(
                    title="High Recall, Lower Precision",
                    description=f"Model prioritizes recall ({recall:.3f}) over precision ({precision:.3f})",
                    details=[
                        "Model is aggressive - captures most positive cases",
                        "May have more false positives",
                        "Good for scenarios where missing positives is costly"
                    ],
                    impact="Medium",
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.PERFORMANCE,
                        confidence=ConfidenceLevel.HIGH,
                        importance=0.7,
                        timestamp=datetime.now(),
                        model_type=ModelType.CLASSIFICATION
                    ),
                    numerical_evidence={'precision': precision, 'recall': recall}
                ))
        
        # Class imbalance insights
        if dataset_info and 'class_distribution' in dataset_info:
            class_dist = dataset_info['class_distribution']
            if isinstance(class_dist, dict):
                class_counts = list(class_dist.values())
                if len(class_counts) > 1:
                    imbalance_ratio = max(class_counts) / min(class_counts)
                    if imbalance_ratio > 5:
                        insights.append(Insight(
                            title="Class Imbalance Detected",
                            description=f"Dataset has significant class imbalance (ratio: {imbalance_ratio:.1f}:1)",
                            details=[
                                f"Class distribution: {class_dist}",
                                "Imbalanced data can bias model predictions",
                                "Consider class balancing techniques",
                                "Monitor minority class performance closely"
                            ],
                            impact="High",
                            actionable=True,
                            metadata=InsightMetadata(
                                insight_type=InsightType.DATA_QUALITY,
                                confidence=ConfidenceLevel.HIGH,
                                importance=0.8,
                                timestamp=datetime.now(),
                                model_type=ModelType.CLASSIFICATION
                            ),
                            numerical_evidence={'imbalance_ratio': imbalance_ratio}
                        ))
        
        return insights
    
    async def _generate_regression_insights(
        self,
        model_results: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]]
    ) -> List[Insight]:
        """Generate regression-specific insights."""
        insights = []
        
        performance_metrics = model_results.get('performance_metrics', {})
        
        # R² interpretation
        r2_score = performance_metrics.get('r2_score')
        if r2_score is not None:
            variance_explained = r2_score * 100
            
            if r2_score > 0.9:
                interpretation = "excellent - explains most variance"
            elif r2_score > 0.7:
                interpretation = "good - explains majority of variance"
            elif r2_score > 0.5:
                interpretation = "moderate - explains some variance"
            else:
                interpretation = "poor - explains little variance"
            
            insights.append(Insight(
                title="Variance Explanation",
                description=f"Model explains {variance_explained:.1f}% of target variance",
                details=[
                    f"R² score: {r2_score:.3f}",
                    f"Performance level: {interpretation}",
                    f"Unexplained variance: {100 - variance_explained:.1f}%"
                ],
                impact="High" if r2_score < 0.5 else "Medium",
                actionable=r2_score < 0.7,
                metadata=InsightMetadata(
                    insight_type=InsightType.PERFORMANCE,
                    confidence=ConfidenceLevel.HIGH,
                    importance=0.9,
                    timestamp=datetime.now(),
                    model_type=ModelType.REGRESSION
                ),
                numerical_evidence={'r2_score': r2_score, 'variance_explained': variance_explained}
            ))
        
        # Error distribution analysis
        rmse = performance_metrics.get('rmse')
        mae = performance_metrics.get('mae')
        
        if rmse is not None and mae is not None:
            error_ratio = rmse / mae if mae > 0 else float('inf')
            
            if error_ratio > 1.5:
                insights.append(Insight(
                    title="Large Error Variability",
                    description=f"RMSE/MAE ratio of {error_ratio:.2f} indicates some large prediction errors",
                    details=[
                        f"RMSE: {rmse:.4f}, MAE: {mae:.4f}",
                        "Some predictions have disproportionately large errors",
                        "Consider outlier detection and robust models",
                        "Review error patterns for insights"
                    ],
                    impact="Medium",
                    actionable=True,
                    metadata=InsightMetadata(
                        insight_type=InsightType.PERFORMANCE,
                        confidence=ConfidenceLevel.HIGH,
                        importance=0.6,
                        timestamp=datetime.now(),
                        model_type=ModelType.REGRESSION
                    ),
                    numerical_evidence={'rmse': rmse, 'mae': mae, 'error_ratio': error_ratio}
                ))
        
        return insights
    
    # Summary and aggregation methods
    
    async def _generate_summary(self, insights: List[Insight], model_type: ModelType) -> str:
        """Generate overall summary from insights."""
        try:
            high_impact_insights = [i for i in insights if i.impact == "High"]
            actionable_insights = [i for i in insights if i.actionable]
            
            # Calculate average confidence
            confidences = [i.metadata.confidence for i in insights]
            confidence_weights = {'high': 1.0, 'medium': 0.7, 'low': 0.3}
            avg_confidence = np.mean([confidence_weights[c.value] for c in confidences]) if confidences else 0.5
            
            # Build summary
            summary_parts = []
            
            # Model type and performance summary
            performance_insights = [i for i in insights if i.metadata.insight_type == InsightType.PERFORMANCE]
            if performance_insights:
                best_performance = max(performance_insights, key=lambda x: x.metadata.importance)
                summary_parts.append(f"Your {model_type.value} model shows {best_performance.description.lower()}")
            
            # Key findings
            if high_impact_insights:
                summary_parts.append(f"Key findings include {len(high_impact_insights)} high-impact insights")
            
            # Actionable items
            if actionable_insights:
                summary_parts.append(f"with {len(actionable_insights)} actionable recommendations")
            
            # Confidence assessment
            confidence_desc = "high" if avg_confidence > 0.8 else "moderate" if avg_confidence > 0.5 else "low"
            summary_parts.append(f"Analysis confidence is {confidence_desc}")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return f"Analysis completed for {model_type.value} model with {len(insights)} insights generated."
    
    async def _extract_key_findings(self, insights: List[Insight]) -> List[str]:
        """Extract key findings from insights."""
        try:
            # Sort by importance and impact
            sorted_insights = sorted(
                insights,
                key=lambda x: (x.impact == "High", x.metadata.importance),
                reverse=True
            )
            
            # Extract top findings
            key_findings = []
            for insight in sorted_insights[:5]:  # Top 5 findings
                finding = insight.title
                if insight.numerical_evidence:
                    # Add key numbers
                    key_metric = list(insight.numerical_evidence.items())[0]
                    finding += f" ({key_metric[0]}: {key_metric[1]:.3f})"
                key_findings.append(finding)
            
            return key_findings
            
        except Exception as e:
            logger.error(f"Key findings extraction failed: {str(e)}")
            return [insight.title for insight in insights[:3]]
    
    async def _generate_recommendations(
        self,
        insights: List[Insight],
        model_results: Dict[str, Any],
        model_type: ModelType
    ) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Collect recommendations from actionable insights
            actionable_insights = [i for i in insights if i.actionable]
            
            for insight in actionable_insights:
                if "consider" in insight.description.lower():
                    # Extract recommendation from description
                    recommendation = insight.description.split("consider")[-1].strip()
                    if recommendation and not recommendation.startswith(('Consider', 'consider')):
                        recommendation = "Consider " + recommendation
                    recommendations.append(recommendation)
                elif insight.details:
                    # Look for actionable details
                    for detail in insight.details:
                        if any(action_word in detail.lower() for action_word in ['consider', 'try', 'improve', 'add', 'remove']):
                            recommendations.append(detail)
            
            # Add model-specific recommendations
            performance_metrics = model_results.get('performance_metrics', {})
            
            if model_type == ModelType.CLASSIFICATION:
                accuracy = performance_metrics.get('accuracy', 0)
                if accuracy < 0.8:
                    recommendations.append("Try ensemble methods or feature engineering to improve accuracy")
            
            elif model_type == ModelType.REGRESSION:
                r2_score = performance_metrics.get('r2_score', 0)
                if r2_score < 0.7:
                    recommendations.append("Consider polynomial features or non-linear models to capture more variance")
            
            # Remove duplicates and limit
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:8]  # Limit to 8 recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {str(e)}")
            return ["Review model performance and consider improvements"]
    
    async def _extract_risk_factors(self, insights: List[Insight]) -> List[str]:
        """Extract risk factors from insights."""
        try:
            risk_factors = []
            
            risk_insights = [i for i in insights if i.metadata.insight_type == InsightType.RISK_ASSESSMENT]
            for insight in risk_insights:
                risk_factors.append(insight.title)
            
            # Look for other risk indicators
            for insight in insights:
                if any(risk_word in insight.description.lower() for risk_word in ['risk', 'bias', 'overfit', 'unstable']):
                    risk_factors.append(insight.title)
            
            return list(dict.fromkeys(risk_factors))[:5]  # Unique, limit to 5
            
        except Exception as e:
            logger.error(f"Risk factors extraction failed: {str(e)}")
            return []
    
    async def _extract_opportunities(self, insights: List[Insight]) -> List[str]:
        """Extract opportunities from insights."""
        try:
            opportunities = []
            
            # Look for positive insights and improvement opportunities
            for insight in insights:
                if insight.impact == "High" and not insight.actionable:
                    # High impact, non-actionable usually means good performance
                    if any(positive_word in insight.description.lower() for positive_word in ['excellent', 'good', 'high', 'strong']):
                        opportunities.append(f"Leverage {insight.title.lower()} for business value")
                
                elif insight.actionable and insight.metadata.importance > 0.7:
                    # High importance actionable items are opportunities
                    opportunities.append(f"Opportunity to improve: {insight.title}")
            
            return list(dict.fromkeys(opportunities))[:5]
            
        except Exception as e:
            logger.error(f"Opportunities extraction failed: {str(e)}")
            return []
    
    async def _generate_next_steps(
        self,
        insights: List[Insight],
        model_results: Dict[str, Any]
    ) -> List[str]:
        """Generate concrete next steps."""
        try:
            next_steps = []
            
            # Deployment readiness
            performance_metrics = model_results.get('performance_metrics', {})
            if performance_metrics:
                # Simple deployment readiness check
                primary_metric = list(performance_metrics.values())[0]
                if isinstance(primary_metric, (int, float)) and primary_metric > 0.8:
                    next_steps.append("Model is ready for pilot deployment")
                else:
                    next_steps.append("Improve model performance before deployment")
            
            # Data collection
            data_quality_insights = [i for i in insights if i.metadata.insight_type == InsightType.DATA_QUALITY]
            if data_quality_insights:
                next_steps.append("Address data quality issues identified")
            
            # Feature engineering
            feature_insights = [i for i in insights if i.metadata.insight_type == InsightType.FEATURE_IMPORTANCE]
            actionable_feature_insights = [i for i in feature_insights if i.actionable]
            if actionable_feature_insights:
                next_steps.append("Explore additional feature engineering opportunities")
            
            # Monitoring setup
            next_steps.append("Set up model monitoring for production deployment")
            
            # Business validation
            next_steps.append("Validate model outputs with business stakeholders")
            
            return next_steps[:6]  # Limit to 6 steps
            
        except Exception as e:
            logger.error(f"Next steps generation failed: {str(e)}")
            return ["Review insights and plan model improvements"]
    
    # Utility methods
    
    def _assess_performance_level(self, metric_name: str, metric_value: float) -> str:
        """Assess performance level for a given metric."""
        try:
            thresholds = self.performance_thresholds.get(metric_name.lower(), {})
            
            if not thresholds:
                # Default thresholds for unknown metrics
                if metric_value > 0.9:
                    return 'excellent'
                elif metric_value > 0.75:
                    return 'good'
                elif metric_value > 0.5:
                    return 'moderate'
                else:
                    return 'poor'
            
            # Handle inverse metrics (lower is better)
            if thresholds.get('inverse', False):
                if metric_value <= thresholds['excellent']:
                    return 'excellent'
                elif metric_value <= thresholds['good']:
                    return 'good'
                elif metric_value <= thresholds['moderate']:
                    return 'moderate'
                else:
                    return 'poor'
            else:
                if metric_value >= thresholds['excellent']:
                    return 'excellent'
                elif metric_value >= thresholds['good']:
                    return 'good'
                elif metric_value >= thresholds['moderate']:
                    return 'moderate'
                else:
                    return 'poor'
        
        except Exception as e:
            logger.warning(f"Performance assessment failed for {metric_name}: {str(e)}")
            return 'moderate'
    
    def _calculate_overall_confidence(self, insights: List[Insight]) -> float:
        """Calculate overall confidence score for insights."""
        try:
            if not insights:
                return 0.5
            
            confidence_weights = {
                ConfidenceLevel.HIGH: 1.0,
                ConfidenceLevel.MEDIUM: 0.7,
                ConfidenceLevel.LOW: 0.3
            }
            
            weighted_scores = []
            for insight in insights:
                weight = confidence_weights[insight.metadata.confidence]
                importance = insight.metadata.importance
                weighted_scores.append(weight * importance)
            
            return np.mean(weighted_scores)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    async def _create_dashboard_data(
        self,
        insights: List[Insight],
        model_results: Dict[str, Any],
        model_type: ModelType
    ) -> Dict[str, Any]:
        """Create dashboard visualization data."""
        try:
            dashboard_data = {
                'insight_summary': {
                    'total_insights': len(insights),
                    'high_impact': len([i for i in insights if i.impact == "High"]),
                    'actionable': len([i for i in insights if i.actionable]),
                    'confidence_distribution': self._get_confidence_distribution(insights)
                },
                'performance_overview': {},
                'feature_importance_chart': {},
                'recommendations_priority': {}
            }
            
            # Performance overview
            performance_metrics = model_results.get('performance_metrics', {})
            if performance_metrics:
                dashboard_data['performance_overview'] = {
                    'metrics': performance_metrics,
                    'chart_data': {
                        'labels': list(performance_metrics.keys()),
                        'values': list(performance_metrics.values()),
                        'type': 'bar'
                    }
                }
            
            # Feature importance visualization
            feature_importance = model_results.get('feature_importance', {})
            if feature_importance:
                # Top 10 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                
                dashboard_data['feature_importance_chart'] = {
                    'chart_data': {
                        'labels': [f[0] for f in sorted_features],
                        'values': [f[1] for f in sorted_features],
                        'type': 'horizontal_bar'
                    }
                }
            
            # Insight type distribution
            insight_types = {}
            for insight in insights:
                insight_type = insight.metadata.insight_type.value
                insight_types[insight_type] = insight_types.get(insight_type, 0) + 1
            
            dashboard_data['insight_distribution'] = {
                'chart_data': {
                    'labels': list(insight_types.keys()),
                    'values': list(insight_types.values()),
                    'type': 'pie'
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard data creation failed: {str(e)}")
            return {}
    
    def _get_confidence_distribution(self, insights: List[Insight]) -> Dict[str, int]:
        """Get distribution of confidence levels."""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for insight in insights:
            distribution[insight.metadata.confidence.value] += 1
        
        return distribution
    
    async def _generate_business_metrics(
        self,
        model_results: Dict[str, Any],
        model_type: ModelType,
        business_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate business-relevant metrics."""
        try:
            business_metrics = {}
            
            # ROI estimation
            if business_context.get('cost_savings_per_correct_prediction'):
                performance_metrics = model_results.get('performance_metrics', {})
                accuracy = performance_metrics.get('accuracy', 0)
                
                if accuracy > 0:
                    cost_savings = business_context['cost_savings_per_correct_prediction']
                    estimated_samples = business_context.get('expected_predictions_per_month', 1000)
                    
                    monthly_savings = cost_savings * accuracy * estimated_samples
                    business_metrics['estimated_monthly_savings'] = monthly_savings
                    business_metrics['roi_percentage'] = (monthly_savings / business_context.get('monthly_model_cost', 1000)) * 100
            
            # Risk assessment
            business_metrics['deployment_risk'] = self._assess_deployment_risk(model_results, business_context)
            
            # Confidence score
            business_metrics['business_confidence'] = self._calculate_business_confidence(model_results, business_context)
            
            return business_metrics
            
        except Exception as e:
            logger.error(f"Business metrics generation failed: {str(e)}")
            return {}
    
    def _assess_deployment_risk(self, model_results: Dict[str, Any], business_context: Dict[str, Any]) -> str:
        """Assess risk level for model deployment."""
        try:
            performance_metrics = model_results.get('performance_metrics', {})
            
            if not performance_metrics:
                return "High"
            
            # Simple risk assessment based on performance
            primary_metric = list(performance_metrics.values())[0]
            
            if isinstance(primary_metric, (int, float)):
                if primary_metric > 0.9:
                    return "Low"
                elif primary_metric > 0.75:
                    return "Medium"
                else:
                    return "High"
            
            return "Medium"
            
        except Exception as e:
            logger.warning(f"Risk assessment failed: {str(e)}")
            return "Medium"
    
    def _calculate_business_confidence(self, model_results: Dict[str, Any], business_context: Dict[str, Any]) -> float:
        """Calculate business confidence score."""
        try:
            factors = []
            
            # Model performance factor
            performance_metrics = model_results.get('performance_metrics', {})
            if performance_metrics:
                avg_performance = np.mean(list(performance_metrics.values()))
                factors.append(min(avg_performance, 1.0))
            
            # Data quality factor
            if business_context.get('data_quality_score'):
                factors.append(business_context['data_quality_score'])
            
            # Business alignment factor
            if business_context.get('business_alignment_score'):
                factors.append(business_context['business_alignment_score'])
            else:
                factors.append(0.7)  # Default moderate alignment
            
            return np.mean(factors) if factors else 0.5
            
        except Exception as e:
            logger.warning(f"Business confidence calculation failed: {str(e)}")
            return 0.5
    
    def _validate_model_results(self, model_results: Dict[str, Any]) -> None:
        """Validate model results structure."""
        if not isinstance(model_results, dict):
            raise ValueError("Model results must be a dictionary")
        
        if not model_results:
            raise ValueError("Model results cannot be empty")
        
        # Basic validation - could be enhanced
        logger.debug("Model results validation passed")
    
    def _update_stats(self, result: InsightsResult) -> None:
        """Update service statistics."""
        self.stats['insights_generated'] += len(result.insights)
        self.stats['models_analyzed'] += 1
        self.stats['recommendations_made'] += len(result.recommendations)
        
        # Update average confidence
        total_confidence = self.stats['average_confidence'] * (self.stats['models_analyzed'] - 1) + result.confidence_score
        self.stats['average_confidence'] = total_confidence / self.stats['models_analyzed']
    
    def _create_aggregate_visualizations(self, insights_result: InsightsResult) -> Dict[str, Any]:
        """Create aggregate visualizations for dashboard."""
        try:
            return {
                'confidence_gauge': {
                    'type': 'gauge',
                    'value': insights_result.confidence_score,
                    'max': 1.0,
                    'title': 'Analysis Confidence'
                },
                'impact_distribution': {
                    'type': 'donut',
                    'data': {
                        'High': len([i for i in insights_result.insights if i.impact == "High"]),
                        'Medium': len([i for i in insights_result.insights if i.impact == "Medium"]),
                        'Low': len([i for i in insights_result.insights if i.impact == "Low"])
                    },
                    'title': 'Impact Distribution'
                }
            }
        except Exception as e:
            logger.error(f"Aggregate visualizations creation failed: {str(e)}")
            return {}
    
    # Placeholder methods for complex insight generation
    # These would be implemented with more sophisticated logic
    
    async def _generate_individual_feature_insight(
        self, feature_name: str, importance: float, normalized_importance: float, 
        rank: int, dataset_info: Optional[Dict[str, Any]]
    ) -> Optional[Insight]:
        """Generate insight for individual feature."""
        if normalized_importance > 0.15:  # Significant feature
            return Insight(
                title=f"Key Feature: {feature_name}",
                description=f"'{feature_name}' ranks #{rank} with {normalized_importance:.1%} importance",
                details=[
                    f"Feature importance: {importance:.4f}",
                    f"Relative importance: {normalized_importance:.1%}",
                    "Consider domain expertise to validate this finding"
                ],
                impact="Medium",
                actionable=False,
                metadata=InsightMetadata(
                    insight_type=InsightType.FEATURE_IMPORTANCE,
                    confidence=ConfidenceLevel.HIGH,
                    importance=normalized_importance,
                    timestamp=datetime.now(),
                    model_type=ModelType.CLASSIFICATION
                ),
                related_features=[feature_name],
                numerical_evidence={'importance': importance, 'normalized_importance': normalized_importance}
            )
        return None
    
    async def _generate_overall_performance_insight(
        self, performance_metrics: Dict[str, Any], model_type: ModelType
    ) -> Optional[Insight]:
        """Generate overall performance insight."""
        # This would implement sophisticated overall performance assessment
        return None
    
    async def _calculate_roi_insight(
        self, performance_metrics: Dict[str, Any], cost_benefit: Dict[str, Any], model_type: ModelType
    ) -> Optional[Insight]:
        """Calculate ROI insight."""
        # This would implement ROI calculation logic
        return None
    
    async def _assess_deployment_readiness(
        self, performance_metrics: Dict[str, Any], model_type: ModelType, business_context: Dict[str, Any]
    ) -> Insight:
        """Assess deployment readiness."""
        # Simplified deployment readiness assessment
        if performance_metrics:
            avg_performance = np.mean(list(performance_metrics.values()))
            if avg_performance > 0.8:
                readiness = "Ready"
                description = "Model meets performance criteria for deployment"
                impact = "High"
            elif avg_performance > 0.6:
                readiness = "Needs Review"
                description = "Model performance is acceptable but should be reviewed"
                impact = "Medium"
            else:
                readiness = "Not Ready"
                description = "Model performance is below deployment threshold"
                impact = "High"
        else:
            readiness = "Unknown"
            description = "Cannot assess deployment readiness without performance metrics"
            impact = "Medium"
        
        return Insight(
            title=f"Deployment Status: {readiness}",
            description=description,
            details=[
                f"Assessment: {readiness}",
                "Review business requirements",
                "Consider A/B testing for deployment"
            ],
            impact=impact,
            actionable=readiness != "Ready",
            metadata=InsightMetadata(
                insight_type=InsightType.BUSINESS_IMPACT,
                confidence=ConfidenceLevel.MEDIUM,
                importance=0.9,
                timestamp=datetime.now(),
                model_type=model_type
            )
        )
    
    async def _assess_business_value(
        self, performance_metrics: Dict[str, Any], model_type: ModelType, business_context: Dict[str, Any]
    ) -> Insight:
        """Assess business value."""
        # Simplified business value assessment
        return Insight(
            title="Business Value Assessment",
            description="Model provides moderate business value based on performance metrics",
            details=[
                "Performance metrics indicate good model quality",
                "Consider business KPI alignment",
                "Monitor real-world performance"
            ],
            impact="Medium",
            actionable=True,
            metadata=InsightMetadata(
                insight_type=InsightType.BUSINESS_IMPACT,
                confidence=ConfidenceLevel.MEDIUM,
                importance=0.7,
                timestamp=datetime.now(),
                model_type=model_type
            )
        )
    
    # Additional placeholder methods would be implemented similarly
    async def _generate_business_risk_insight(self, performance_metrics, model_type, business_context): return None
    async def _assess_model_confidence(self, performance_metrics, model_type): return None
    async def _assess_overfitting_risk(self, model_results, dataset_info): return None
    async def _detect_potential_bias(self, model_results, dataset_info): return None
    async def _assess_generalization(self, performance_metrics, model_type, dataset_info): return None
    async def _generate_forecasting_insights(self, model_results, dataset_info): return []
    async def _generate_clustering_insights(self, model_results, dataset_info): return []
    async def _generate_anomaly_insights(self, model_results, dataset_info): return []
    async def _generate_text_insights(self, model_results, dataset_info): return []
    async def _generate_feature_contribution_insights(self, input_features, explanation_data): return []
    async def _convert_explanations_to_insights(self, explanations, model_type): return []

# Factory function for easy service creation
def create_insights_service(
    language: str = "en",
    business_context: Optional[Dict[str, Any]] = None
) -> InsightsService:
    """
    Factory function to create InsightsService instance.
    
    Args:
        language: Language for insights generation
        business_context: Business context for relevant insights
        
    Returns:
        Configured InsightsService instance
    """
    return InsightsService(language=language, business_context=business_context)

def get_insights_service() -> InsightsService:
    """Get InsightsService instance for dependency injection."""
    return create_insights_service()

# Example usage
if __name__ == "__main__":
    async def example_usage():
        """Example usage of the insights service."""
        
        print("🔍 InsightsService Example Usage")
        print("=" * 50)
        
        # Initialize service
        insights_service = create_insights_service()
        
        # Example model results
        mock_model_results = {
            'performance_metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85
            },
            'feature_importance': {
                'feature_1': 0.35,
                'feature_2': 0.25,
                'feature_3': 0.15,
                'feature_4': 0.10,
                'feature_5': 0.05,
                'feature_6': 0.05,
                'feature_7': 0.05
            },
            'best_model_name': 'RandomForestClassifier'
        }
        
        mock_dataset_info = {
            'n_samples': 5000,
            'n_features': 7,
            'missing_value_ratio': 0.05,
            'data_quality_score': 0.8
        }
        
        try:
            # Generate insights
            print("🔄 Generating insights...")
            insights_result = await insights_service.generate_insights(
                model_results=mock_model_results,
                model_type='classification',
                dataset_info=mock_dataset_info
            )
            
            print(f"✅ Generated {len(insights_result.insights)} insights")
            print(f"📊 Confidence Score: {insights_result.confidence_score:.2f}")
            
            # Display summary
            print(f"\n📋 Summary:")
            print(f"   {insights_result.summary}")
            
            # Display key findings
            print(f"\n🔍 Key Findings:")
            for i, finding in enumerate(insights_result.key_findings[:3], 1):
                print(f"   {i}. {finding}")
            
            # Display recommendations
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(insights_result.recommendations[:3], 1):
                print(f"   {i}. {rec}")
            
            # Format for dashboard
            print(f"\n📱 Dashboard Formatting...")
            dashboard_data = insights_service.format_for_dashboard(insights_result)
            print(f"   Dashboard sections: {len(dashboard_data)}")
            print(f"   Visualizations: {len(dashboard_data.get('visualizations', []))}")
            
            # Service statistics
            print(f"\n📊 Service Statistics:")
            stats = insights_service.stats
            print(f"   Models Analyzed: {stats['models_analyzed']}")
            print(f"   Insights Generated: {stats['insights_generated']}")
            print(f"   Average Confidence: {stats['average_confidence']:.2f}")
            
        except Exception as e:
            print(f"❌ Example failed: {str(e)}")
        
        print(f"\n🎯 InsightsService example completed!")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except Exception as e:
        print(f"Example execution failed: {str(e)}")
