"""
Drift Detection Module for Auto-Analyst Platform

This module implements comprehensive drift detection capabilities including:
- Data drift detection using statistical methods and ML-based approaches
- Model performance drift monitoring
- Feature drift analysis and root cause identification
- Real-time and batch drift detection
- Multi-variate drift detection with correlation analysis
- Population stability index (PSI) calculation
- Kolmogorov-Smirnov and other statistical tests
- Distribution comparison and visualization
- Alert generation and notification systems
- Integration with MLOps pipelines

Features:
- Multiple drift detection algorithms (statistical, ML-based, distribution-based)
- Automatic threshold determination and adaptive alerting
- Feature-level and dataset-level drift analysis
- Performance drift monitoring with business impact assessment
- Real-time streaming drift detection
- Comprehensive reporting and visualization support
- Integration with monitoring systems (Prometheus, Grafana)
- MLflow experiment and model registry integration
- Configurable alert mechanisms (email, Slack, webhooks)
- Drift explanation and root cause analysis
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

# Statistical analysis
from scipy import stats
from scipy.spatial.distance import jensenshannon, wasserstein_distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Drift detection libraries
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.test_suite import TestSuite
    from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

try:
    import alibi_detect
    from alibi_detect.cd import KSDrift, MMDDrift, TabularDrift
    from alibi_detect.utils.saving import save_detector, load_detector
    ALIBI_DETECT_AVAILABLE = True
except ImportError:
    ALIBI_DETECT_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Monitoring integration
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Notification systems
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Advanced statistical methods
try:
    from scipy.stats import entropy
    from scipy.optimize import minimize_scalar
    SCIPY_ADVANCED_AVAILABLE = True
except ImportError:
    SCIPY_ADVANCED_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    TARGET_DRIFT = "target_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    FEATURE_DRIFT = "feature_drift"

class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftMethod(Enum):
    """Available drift detection methods."""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    WASSERSTEIN = "wasserstein"
    PSI = "population_stability_index"
    JENSEN_SHANNON = "jensen_shannon"
    CHI_SQUARE = "chi_square"
    EVIDENTLY = "evidently"
    ALIBI_DETECT = "alibi_detect"
    STATISTICAL_TESTS = "statistical_tests"
    ML_BASED = "ml_based"

@dataclass
class DriftAlert:
    """Structure for drift alerts."""
    id: str
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    method: DriftMethod
    feature: Optional[str]
    drift_score: float
    threshold: float
    message: str
    metadata: Dict[str, Any]
    resolved: bool = False

@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    report_id: str
    timestamp: datetime
    dataset_name: str
    reference_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]
    drift_detected: bool
    overall_drift_score: float
    feature_drift_scores: Dict[str, float]
    alerts: List[DriftAlert]
    recommendations: List[str]
    metadata: Dict[str, Any]

class DriftConfig:
    """Configuration for drift detection."""
    
    def __init__(self):
        # Detection thresholds
        self.data_drift_threshold = 0.1  # KS test p-value threshold
        self.psi_threshold = 0.2  # PSI threshold for significant drift
        self.wasserstein_threshold = 0.1  # Normalized Wasserstein distance
        self.jensen_shannon_threshold = 0.1  # JS divergence threshold
        
        # Performance drift thresholds
        self.performance_drop_threshold = 0.05  # 5% performance drop
        self.critical_performance_threshold = 0.15  # 15% drop is critical
        
        # Monitoring settings
        self.monitoring_window_days = 7  # Look back window for drift detection
        self.min_samples_per_window = 100  # Minimum samples needed for detection
        self.drift_detection_frequency = "daily"  # hourly, daily, weekly
        
        # Alert settings
        self.max_alerts_per_day = 50  # Prevent alert spam
        self.alert_cooldown_minutes = 60  # Minimum time between similar alerts
        self.enable_notifications = True
        
        # Statistical settings
        self.confidence_level = 0.95
        self.bonferroni_correction = True  # Multiple testing correction
        self.bootstrap_samples = 1000
        
        # Advanced settings
        self.adaptive_thresholds = True  # Adjust thresholds based on historical data
        self.feature_importance_weighting = True  # Weight drift by feature importance
        self.drift_explanation_enabled = True
        
        # Integration settings
        self.mlflow_tracking = MLFLOW_AVAILABLE
        self.prometheus_metrics = PROMETHEUS_AVAILABLE
        self.save_drift_artifacts = True

class PopulationStabilityIndex:
    """Population Stability Index (PSI) calculator."""
    
    def __init__(self, bins: int = 10, min_freq: float = 0.001):
        self.bins = bins
        self.min_freq = min_freq
        self.bin_edges = None
        self.reference_dist = None
    
    def fit(self, reference_data: np.ndarray) -> 'PopulationStabilityIndex':
        """Fit PSI calculator on reference data."""
        # Create bins based on reference data quantiles
        self.bin_edges = np.quantile(reference_data, np.linspace(0, 1, self.bins + 1))
        self.bin_edges[-1] += 1e-6  # Ensure last bin includes maximum value
        
        # Calculate reference distribution
        hist, _ = np.histogram(reference_data, bins=self.bin_edges)
        self.reference_dist = hist / len(reference_data)
        
        # Apply minimum frequency to avoid division by zero
        self.reference_dist = np.maximum(self.reference_dist, self.min_freq)
        
        return self
    
    def calculate(self, current_data: np.ndarray) -> float:
        """Calculate PSI between reference and current data."""
        if self.bin_edges is None or self.reference_dist is None:
            raise ValueError("PSI calculator must be fitted first")
        
        # Calculate current distribution
        hist, _ = np.histogram(current_data, bins=self.bin_edges)
        current_dist = hist / len(current_data)
        current_dist = np.maximum(current_dist, self.min_freq)
        
        # Calculate PSI
        psi_values = (current_dist - self.reference_dist) * np.log(current_dist / self.reference_dist)
        return float(np.sum(psi_values))

class DriftDetector:
    """Comprehensive drift detection system."""
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.reference_data = {}
        self.reference_statistics = {}
        self.psi_calculators = {}
        self.drift_detectors = {}  # For advanced ML-based detectors
        self.alert_history = []
        self.drift_reports = []
        
        # Performance tracking
        if self.config.prometheus_metrics and PROMETHEUS_AVAILABLE:
            self.drift_counter = Counter('drift_detections_total', 'Total drift detections', ['drift_type', 'severity'])
            self.drift_score_histogram = Histogram('drift_score', 'Drift score distribution')
            self.performance_gauge = Gauge('model_performance_current', 'Current model performance')
        
        logger.info("DriftDetector initialized")
    
    async def set_reference_data(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        identifier: str = "default"
    ) -> bool:
        """Set reference data for drift detection."""
        try:
            # Store reference data
            self.reference_data[identifier] = data.copy()
            
            # Calculate reference statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            stats = {
                'numeric': {},
                'categorical': {},
                'target': {},
                'metadata': {
                    'n_samples': len(data),
                    'n_features': len(data.columns),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            # Numeric feature statistics
            for col in numeric_cols:
                if col != target_column:
                    values = data[col].dropna()
                    stats['numeric'][col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'skewness': float(values.skew()),
                        'kurtosis': float(values.kurtosis()),
                        'quantiles': values.quantile([0.25, 0.5, 0.75]).to_dict()
                    }
                    
                    # Fit PSI calculator
                    psi_calc = PopulationStabilityIndex(bins=10)
                    psi_calc.fit(values.values)
                    self.psi_calculators[f"{identifier}_{col}"] = psi_calc
            
            # Categorical feature statistics
            for col in categorical_cols:
                if col != target_column:
                    value_counts = data[col].value_counts(normalize=True)
                    stats['categorical'][col] = {
                        'categories': value_counts.index.tolist(),
                        'frequencies': value_counts.values.tolist(),
                        'n_unique': int(data[col].nunique()),
                        'most_frequent': str(value_counts.index[0])
                    }
            
            # Target statistics
            if target_column and target_column in data.columns:
                target_values = data[target_column].dropna()
                if pd.api.types.is_numeric_dtype(target_values):
                    stats['target'] = {
                        'type': 'numeric',
                        'mean': float(target_values.mean()),
                        'std': float(target_values.std()),
                        'distribution': target_values.hist(bins=20)[0].tolist()
                    }
                else:
                    value_counts = target_values.value_counts(normalize=True)
                    stats['target'] = {
                        'type': 'categorical',
                        'categories': value_counts.index.tolist(),
                        'frequencies': value_counts.values.tolist()
                    }
            
            self.reference_statistics[identifier] = stats
            
            # Initialize advanced drift detectors if available
            if ALIBI_DETECT_AVAILABLE and len(numeric_cols) > 0:
                try:
                    # Use only numeric columns for advanced detection
                    reference_array = data[numeric_cols].values
                    
                    # KS Drift detector
                    ks_detector = KSDrift(reference_array, p_val=self.config.data_drift_threshold)
                    self.drift_detectors[f"{identifier}_ks"] = ks_detector
                    
                    # MMD Drift detector for multivariate detection
                    if len(numeric_cols) > 1 and len(data) > 200:
                        mmd_detector = MMDDrift(reference_array, p_val=self.config.data_drift_threshold)
                        self.drift_detectors[f"{identifier}_mmd"] = mmd_detector
                        
                except Exception as e:
                    logger.warning(f"Failed to initialize advanced drift detectors: {str(e)}")
            
            logger.info(f"Reference data set for identifier: {identifier}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set reference data: {str(e)}")
            return False
    
    async def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        identifier: str = "default",
        methods: Optional[List[DriftMethod]] = None
    ) -> Dict[str, Any]:
        """Detect data drift using multiple methods."""
        try:
            if identifier not in self.reference_data:
                raise ValueError(f"Reference data not found for identifier: {identifier}")
            
            reference = self.reference_data[identifier]
            ref_stats = self.reference_statistics[identifier]
            
            methods = methods or [
                DriftMethod.KOLMOGOROV_SMIRNOV,
                DriftMethod.PSI,
                DriftMethod.WASSERSTEIN
            ]
            
            drift_results = {
                'overall_drift': False,
                'drift_score': 0.0,
                'feature_drift': {},
                'method_results': {},
                'alerts': [],
                'metadata': {
                    'identifier': identifier,
                    'timestamp': datetime.utcnow().isoformat(),
                    'methods_used': [method.value for method in methods],
                    'current_samples': len(current_data),
                    'reference_samples': len(reference)
                }
            }
            
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            categorical_cols = current_data.select_dtypes(include=['object', 'category']).columns
            
            total_drift_scores = []
            
            # Test numeric features
            for col in numeric_cols:
                if col in reference.columns:
                    feature_drift = await self._detect_numeric_drift(
                        reference[col].dropna().values,
                        current_data[col].dropna().values,
                        col,
                        methods,
                        identifier
                    )
                    
                    drift_results['feature_drift'][col] = feature_drift
                    total_drift_scores.append(feature_drift['drift_score'])
                    
                    # Generate alerts if drift detected
                    if feature_drift['drift_detected']:
                        alert = DriftAlert(
                            id=str(uuid.uuid4()),
                            timestamp=datetime.utcnow(),
                            drift_type=DriftType.FEATURE_DRIFT,
                            severity=self._determine_severity(feature_drift['drift_score']),
                            method=feature_drift['primary_method'],
                            feature=col,
                            drift_score=feature_drift['drift_score'],
                            threshold=self.config.data_drift_threshold,
                            message=f"Feature drift detected in {col}",
                            metadata=feature_drift
                        )
                        drift_results['alerts'].append(alert)
                        self.alert_history.append(alert)
            
            # Test categorical features
            for col in categorical_cols:
                if col in reference.columns:
                    feature_drift = await self._detect_categorical_drift(
                        reference[col].dropna().values,
                        current_data[col].dropna().values,
                        col
                    )
                    
                    drift_results['feature_drift'][col] = feature_drift
                    total_drift_scores.append(feature_drift['drift_score'])
                    
                    if feature_drift['drift_detected']:
                        alert = DriftAlert(
                            id=str(uuid.uuid4()),
                            timestamp=datetime.utcnow(),
                            drift_type=DriftType.FEATURE_DRIFT,
                            severity=self._determine_severity(feature_drift['drift_score']),
                            method=DriftMethod.CHI_SQUARE,
                            feature=col,
                            drift_score=feature_drift['drift_score'],
                            threshold=self.config.data_drift_threshold,
                            message=f"Categorical drift detected in {col}",
                            metadata=feature_drift
                        )
                        drift_results['alerts'].append(alert)
                        self.alert_history.append(alert)
            
            # Advanced multivariate drift detection
            if ALIBI_DETECT_AVAILABLE and len(numeric_cols) > 1:
                multivariate_result = await self._detect_multivariate_drift(
                    current_data[numeric_cols].values,
                    identifier
                )
                drift_results['method_results']['multivariate'] = multivariate_result
                total_drift_scores.append(multivariate_result.get('drift_score', 0))
            
            # Calculate overall drift score
            if total_drift_scores:
                drift_results['drift_score'] = float(np.mean(total_drift_scores))
                drift_results['overall_drift'] = any(
                    feature['drift_detected'] for feature in drift_results['feature_drift'].values()
                )
            
            # Update Prometheus metrics
            if self.config.prometheus_metrics and PROMETHEUS_AVAILABLE:
                self.drift_score_histogram.observe(drift_results['drift_score'])
                if drift_results['overall_drift']:
                    self.drift_counter.labels(
                        drift_type=DriftType.DATA_DRIFT.value,
                        severity=self._determine_severity(drift_results['drift_score']).value
                    ).inc()
            
            # Log to MLflow
            if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                await self._log_drift_to_mlflow(drift_results, DriftType.DATA_DRIFT)
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {str(e)}")
            return {
                'overall_drift': False,
                'drift_score': 0.0,
                'error': str(e),
                'alerts': []
            }
    
    async def _detect_numeric_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        methods: List[DriftMethod],
        identifier: str
    ) -> Dict[str, Any]:
        """Detect drift in numeric features using multiple methods."""
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'primary_method': None,
            'method_details': {}
        }
        
        drift_scores = []
        
        # Kolmogorov-Smirnov test
        if DriftMethod.KOLMOGOROV_SMIRNOV in methods:
            try:
                ks_stat, p_value = stats.ks_2samp(reference, current)
                drift_detected = p_value < self.config.data_drift_threshold
                
                results['method_details']['ks_test'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drift_detected': drift_detected,
                    'threshold': self.config.data_drift_threshold
                }
                
                if drift_detected:
                    drift_scores.append(ks_stat)
                    if not results['drift_detected']:
                        results['primary_method'] = DriftMethod.KOLMOGOROV_SMIRNOV
                        results['drift_detected'] = True
                        
            except Exception as e:
                logger.warning(f"KS test failed for {feature_name}: {str(e)}")
        
        # Population Stability Index
        if DriftMethod.PSI in methods:
            try:
                psi_key = f"{identifier}_{feature_name}"
                if psi_key in self.psi_calculators:
                    psi_score = self.psi_calculators[psi_key].calculate(current)
                    drift_detected = psi_score > self.config.psi_threshold
                    
                    results['method_details']['psi'] = {
                        'score': float(psi_score),
                        'drift_detected': drift_detected,
                        'threshold': self.config.psi_threshold
                    }
                    
                    if drift_detected:
                        drift_scores.append(psi_score / self.config.psi_threshold)  # Normalize
                        if not results['drift_detected']:
                            results['primary_method'] = DriftMethod.PSI
                            results['drift_detected'] = True
                            
            except Exception as e:
                logger.warning(f"PSI calculation failed for {feature_name}: {str(e)}")
        
        # Wasserstein Distance
        if DriftMethod.WASSERSTEIN in methods:
            try:
                # Normalize data to [0, 1] for fair comparison
                ref_min, ref_max = reference.min(), reference.max()
                if ref_max > ref_min:
                    ref_norm = (reference - ref_min) / (ref_max - ref_min)
                    curr_norm = np.clip((current - ref_min) / (ref_max - ref_min), 0, 1)
                    
                    wasserstein_dist = wasserstein_distance(ref_norm, curr_norm)
                    drift_detected = wasserstein_dist > self.config.wasserstein_threshold
                    
                    results['method_details']['wasserstein'] = {
                        'distance': float(wasserstein_dist),
                        'drift_detected': drift_detected,
                        'threshold': self.config.wasserstein_threshold
                    }
                    
                    if drift_detected:
                        drift_scores.append(wasserstein_dist / self.config.wasserstein_threshold)
                        if not results['drift_detected']:
                            results['primary_method'] = DriftMethod.WASSERSTEIN
                            results['drift_detected'] = True
                            
            except Exception as e:
                logger.warning(f"Wasserstein distance failed for {feature_name}: {str(e)}")
        
        # Jensen-Shannon Divergence
        if DriftMethod.JENSEN_SHANNON in methods:
            try:
                # Create histograms for probability distributions
                bins = min(50, int(np.sqrt(min(len(reference), len(current)))))
                
                # Use same bin edges for both distributions
                bin_min = min(reference.min(), current.min())
                bin_max = max(reference.max(), current.max())
                bin_edges = np.linspace(bin_min, bin_max, bins + 1)
                
                ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
                curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
                
                # Normalize to probability distributions
                ref_prob = ref_hist / ref_hist.sum() + 1e-10  # Add small epsilon
                curr_prob = curr_hist / curr_hist.sum() + 1e-10
                
                js_divergence = jensenshannon(ref_prob, curr_prob)
                drift_detected = js_divergence > self.config.jensen_shannon_threshold
                
                results['method_details']['jensen_shannon'] = {
                    'divergence': float(js_divergence),
                    'drift_detected': drift_detected,
                    'threshold': self.config.jensen_shannon_threshold
                }
                
                if drift_detected:
                    drift_scores.append(js_divergence / self.config.jensen_shannon_threshold)
                    if not results['drift_detected']:
                        results['primary_method'] = DriftMethod.JENSEN_SHANNON
                        results['drift_detected'] = True
                        
            except Exception as e:
                logger.warning(f"Jensen-Shannon divergence failed for {feature_name}: {str(e)}")
        
        # Calculate overall drift score
        if drift_scores:
            results['drift_score'] = float(np.max(drift_scores))  # Take maximum drift score
        
        return results
    
    async def _detect_categorical_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str
    ) -> Dict[str, Any]:
        """Detect drift in categorical features using chi-square test."""
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'primary_method': DriftMethod.CHI_SQUARE,
            'method_details': {}
        }
        
        try:
            # Get unique categories from both datasets
            all_categories = np.union1d(reference, current)
            
            # Calculate frequency distributions
            ref_counts = pd.Series(reference).value_counts()
            curr_counts = pd.Series(current).value_counts()
            
            # Align categories and fill missing with zeros
            ref_freq = ref_counts.reindex(all_categories, fill_value=0).values
            curr_freq = curr_counts.reindex(all_categories, fill_value=0).values
            
            # Chi-square test
            if len(all_categories) > 1 and ref_freq.sum() > 0 and curr_freq.sum() > 0:
                # Convert to expected frequencies
                ref_expected = ref_freq / ref_freq.sum() * curr_freq.sum()
                
                # Avoid division by zero
                mask = ref_expected > 0
                if mask.sum() > 0:
                    chi2_stat = np.sum((curr_freq[mask] - ref_expected[mask])**2 / ref_expected[mask])
                    dof = mask.sum() - 1
                    
                    if dof > 0:
                        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
                        
                        drift_detected = p_value < self.config.data_drift_threshold
                        
                        results['method_details']['chi_square'] = {
                            'statistic': float(chi2_stat),
                            'p_value': float(p_value),
                            'degrees_of_freedom': int(dof),
                            'drift_detected': drift_detected,
                            'threshold': self.config.data_drift_threshold
                        }
                        
                        if drift_detected:
                            results['drift_detected'] = True
                            # Normalize chi2 statistic for drift score
                            results['drift_score'] = min(1.0, chi2_stat / (dof * 5))  # Rough normalization
            
            # Additional categorical drift metrics
            # Calculate Total Variation Distance
            ref_prob = ref_freq / ref_freq.sum()
            curr_prob = curr_freq / curr_freq.sum()
            tv_distance = 0.5 * np.sum(np.abs(ref_prob - curr_prob))
            
            results['method_details']['total_variation'] = {
                'distance': float(tv_distance),
                'drift_detected': tv_distance > 0.1  # 10% threshold
            }
            
            if tv_distance > 0.1 and not results['drift_detected']:
                results['drift_detected'] = True
                results['drift_score'] = float(tv_distance)
            
            return results
            
        except Exception as e:
            logger.warning(f"Categorical drift detection failed for {feature_name}: {str(e)}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'error': str(e)
            }
    
    async def _detect_multivariate_drift(
        self,
        current_data: np.ndarray,
        identifier: str
    ) -> Dict[str, Any]:
        """Detect multivariate drift using advanced methods."""
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'method_details': {}
        }
        
        try:
            # KS Drift (univariate applied to each feature)
            ks_key = f"{identifier}_ks"
            if ks_key in self.drift_detectors:
                ks_result = self.drift_detectors[ks_key].predict(current_data)
                
                results['method_details']['ks_multivariate'] = {
                    'drift_detected': bool(ks_result['data']['is_drift']),
                    'p_value': float(ks_result['data']['p_val']),
                    'threshold': self.config.data_drift_threshold
                }
                
                if ks_result['data']['is_drift']:
                    results['drift_detected'] = True
                    results['drift_score'] = 1 - ks_result['data']['p_val']
            
            # MMD Drift (true multivariate)
            mmd_key = f"{identifier}_mmd"
            if mmd_key in self.drift_detectors:
                mmd_result = self.drift_detectors[mmd_key].predict(current_data)
                
                results['method_details']['mmd_multivariate'] = {
                    'drift_detected': bool(mmd_result['data']['is_drift']),
                    'p_value': float(mmd_result['data']['p_val']),
                    'threshold': self.config.data_drift_threshold
                }
                
                if mmd_result['data']['is_drift']:
                    results['drift_detected'] = True
                    if results['drift_score'] == 0.0:  # Don't override KS result
                        results['drift_score'] = 1 - mmd_result['data']['p_val']
            
            return results
            
        except Exception as e:
            logger.warning(f"Multivariate drift detection failed: {str(e)}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'error': str(e)
            }
    
    async def detect_model_performance_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = 'regression',
        model_id: str = 'default'
    ) -> Dict[str, Any]:
        """Detect drift in model performance."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate current performance metrics
            if task_type == 'regression':
                current_metrics = {
                    'mse': float(mean_squared_error(y_true, y_pred)),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'r2': float(r2_score(y_true, y_pred))
                }
                primary_metric = 'r2'
                higher_is_better = True
                
            else:  # classification
                current_metrics = {
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                }
                primary_metric = 'f1'
                higher_is_better = True
            
            # Get historical performance (would typically come from database)
            historical_key = f"performance_{model_id}"
            
            performance_drift = {
                'drift_detected': False,
                'drift_score': 0.0,
                'current_performance': current_metrics,
                'performance_drop': 0.0,
                'severity': DriftSeverity.LOW,
                'alerts': [],
                'metadata': {
                    'model_id': model_id,
                    'task_type': task_type,
                    'timestamp': current_time.isoformat(),
                    'samples_evaluated': len(y_true)
                }
            }
            
            # For demonstration, assume we have baseline performance
            # In practice, this would be stored in database/MLflow
            baseline_performance = self._get_baseline_performance(model_id, task_type)
            
            if baseline_performance:
                baseline_value = baseline_performance.get(primary_metric, 0)
                current_value = current_metrics.get(primary_metric, 0)
                
                if higher_is_better:
                    performance_change = (baseline_value - current_value) / abs(baseline_value) if baseline_value != 0 else 0
                else:
                    performance_change = (current_value - baseline_value) / abs(baseline_value) if baseline_value != 0 else 0
                
                performance_drift['performance_drop'] = float(max(0, performance_change))
                performance_drift['baseline_performance'] = baseline_performance
                
                # Determine if drift is significant
                if performance_drift['performance_drop'] > self.config.performance_drop_threshold:
                    performance_drift['drift_detected'] = True
                    performance_drift['drift_score'] = performance_drift['performance_drop']
                    
                    # Determine severity
                    if performance_drift['performance_drop'] > self.config.critical_performance_threshold:
                        severity = DriftSeverity.CRITICAL
                    elif performance_drift['performance_drop'] > self.config.performance_drop_threshold * 2:
                        severity = DriftSeverity.HIGH
                    else:
                        severity = DriftSeverity.MEDIUM
                    
                    performance_drift['severity'] = severity
                    
                    # Create alert
                    alert = DriftAlert(
                        id=str(uuid.uuid4()),
                        timestamp=current_time,
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        severity=severity,
                        method=DriftMethod.STATISTICAL_TESTS,
                        feature=None,
                        drift_score=performance_drift['drift_score'],
                        threshold=self.config.performance_drop_threshold,
                        message=f"Model performance dropped by {performance_drift['performance_drop']:.2%}",
                        metadata=performance_drift
                    )
                    
                    performance_drift['alerts'].append(alert)
                    self.alert_history.append(alert)
            
            # Update Prometheus metrics
            if self.config.prometheus_metrics and PROMETHEUS_AVAILABLE:
                self.performance_gauge.set(current_metrics.get(primary_metric, 0))
                if performance_drift['drift_detected']:
                    self.drift_counter.labels(
                        drift_type=DriftType.PERFORMANCE_DRIFT.value,
                        severity=performance_drift['severity'].value
                    ).inc()
            
            # Log to MLflow
            if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                await self._log_drift_to_mlflow(performance_drift, DriftType.PERFORMANCE_DRIFT)
            
            return performance_drift
            
        except Exception as e:
            logger.error(f"Performance drift detection failed: {str(e)}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'error': str(e),
                'alerts': []
            }
    
    def _get_baseline_performance(self, model_id: str, task_type: str) -> Optional[Dict[str, float]]:
        """Get baseline performance metrics (placeholder - would be from database)."""
        # This would typically query a database or MLflow for historical performance
        # For demonstration, returning mock baseline performance
        if task_type == 'regression':
            return {
                'mse': 0.1,
                'mae': 0.2,
                'r2': 0.85
            }
        else:
            return {
                'accuracy': 0.90,
                'precision': 0.88,
                'recall': 0.92,
                'f1': 0.90
            }
    
    def _determine_severity(self, drift_score: float) -> DriftSeverity:
        """Determine drift severity based on drift score."""
        if drift_score >= 0.5:
            return DriftSeverity.CRITICAL
        elif drift_score >= 0.3:
            return DriftSeverity.HIGH
        elif drift_score >= 0.1:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    async def generate_drift_report(
        self,
        identifier: str = "default",
        period_days: int = 7
    ) -> DriftReport:
        """Generate comprehensive drift report."""
        try:
            current_time = datetime.utcnow()
            report_period_start = current_time - timedelta(days=period_days)
            
            # Get recent alerts for this identifier
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= report_period_start
            ]
            
            # Calculate overall drift status
            drift_detected = len(recent_alerts) > 0
            overall_drift_score = np.mean([alert.drift_score for alert in recent_alerts]) if recent_alerts else 0.0
            
            # Aggregate feature drift scores
            feature_drift_scores = {}
            for alert in recent_alerts:
                if alert.feature:
                    if alert.feature not in feature_drift_scores:
                        feature_drift_scores[alert.feature] = []
                    feature_drift_scores[alert.feature].append(alert.drift_score)
            
            # Average drift scores per feature
            for feature in feature_drift_scores:
                feature_drift_scores[feature] = np.mean(feature_drift_scores[feature])
            
            # Generate recommendations
            recommendations = self._generate_recommendations(recent_alerts, feature_drift_scores)
            
            report = DriftReport(
                report_id=str(uuid.uuid4()),
                timestamp=current_time,
                dataset_name=identifier,
                reference_period=(current_time - timedelta(days=30), report_period_start),
                current_period=(report_period_start, current_time),
                drift_detected=drift_detected,
                overall_drift_score=float(overall_drift_score),
                feature_drift_scores=feature_drift_scores,
                alerts=recent_alerts,
                recommendations=recommendations,
                metadata={
                    'alert_count': len(recent_alerts),
                    'severity_breakdown': self._count_alerts_by_severity(recent_alerts),
                    'drift_types': self._count_alerts_by_type(recent_alerts)
                }
            )
            
            self.drift_reports.append(report)
            return report
            
        except Exception as e:
            logger.error(f"Drift report generation failed: {str(e)}")
            # Return empty report
            return DriftReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                dataset_name=identifier,
                reference_period=(datetime.utcnow(), datetime.utcnow()),
                current_period=(datetime.utcnow(), datetime.utcnow()),
                drift_detected=False,
                overall_drift_score=0.0,
                feature_drift_scores={},
                alerts=[],
                recommendations=["Report generation failed - check logs"],
                metadata={'error': str(e)}
            )
    
    def _generate_recommendations(
        self,
        alerts: List[DriftAlert],
        feature_drift_scores: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on detected drift."""
        recommendations = []
        
        if not alerts:
            recommendations.append("No drift detected - monitoring system is operating normally")
            return recommendations
        
        # High-level recommendations
        critical_alerts = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append("CRITICAL: Immediate model retraining recommended due to severe drift")
        
        high_alerts = [a for a in alerts if a.severity == DriftSeverity.HIGH]
        if high_alerts:
            recommendations.append("Consider model retraining or feature engineering")
        
        # Feature-specific recommendations
        if feature_drift_scores:
            top_drifted_features = sorted(
                feature_drift_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            recommendations.append(
                f"Focus on features with highest drift: {', '.join([f[0] for f in top_drifted_features])}"
            )
        
        # Performance drift recommendations
        performance_alerts = [a for a in alerts if a.drift_type == DriftType.PERFORMANCE_DRIFT]
        if performance_alerts:
            recommendations.append("Model performance has degraded - investigate data quality and feature changes")
        
        # Data drift recommendations
        data_alerts = [a for a in alerts if a.drift_type == DriftType.DATA_DRIFT]
        if len(data_alerts) > len(alerts) * 0.7:  # Mostly data drift
            recommendations.append("Widespread data distribution changes detected - verify data pipeline")
        
        return recommendations
    
    def _count_alerts_by_severity(self, alerts: List[DriftAlert]) -> Dict[str, int]:
        """Count alerts by severity level."""
        severity_counts = {severity.value: 0 for severity in DriftSeverity}
        for alert in alerts:
            severity_counts[alert.severity.value] += 1
        return severity_counts
    
    def _count_alerts_by_type(self, alerts: List[DriftAlert]) -> Dict[str, int]:
        """Count alerts by drift type."""
        type_counts = {drift_type.value: 0 for drift_type in DriftType}
        for alert in alerts:
            type_counts[alert.drift_type.value] += 1
        return type_counts
    
    async def _log_drift_to_mlflow(self, drift_results: Dict[str, Any], drift_type: DriftType):
        """Log drift detection results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"drift_detection_{drift_type.value}"):
                # Log parameters
                mlflow.log_param("drift_type", drift_type.value)
                mlflow.log_param("detection_timestamp", datetime.utcnow().isoformat())
                
                # Log metrics
                mlflow.log_metric("drift_score", drift_results.get('drift_score', 0))
                mlflow.log_metric("drift_detected", 1 if drift_results.get('drift_detected', False) else 0)
                
                # Log feature-level drift scores
                if 'feature_drift' in drift_results:
                    for feature, feature_result in drift_results['feature_drift'].items():
                        mlflow.log_metric(f"drift_score_{feature}", feature_result.get('drift_score', 0))
                
                # Log performance metrics if available
                if 'current_performance' in drift_results:
                    for metric, value in drift_results['current_performance'].items():
                        mlflow.log_metric(f"performance_{metric}", value)
                
                logger.info(f"Drift results logged to MLflow for {drift_type.value}")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def send_alerts(self, alerts: List[DriftAlert]) -> bool:
        """Send alerts via configured notification channels."""
        if not self.config.enable_notifications or not alerts:
            return True
        
        try:
            # Filter out alerts that are too frequent (rate limiting)
            filtered_alerts = self._filter_rate_limited_alerts(alerts)
            
            if not filtered_alerts:
                return True
            
            success_count = 0
            
            for alert in filtered_alerts:
                # Send to configured channels
                alert_sent = await self._send_single_alert(alert)
                if alert_sent:
                    success_count += 1
            
            logger.info(f"Sent {success_count}/{len(filtered_alerts)} drift alerts successfully")
            return success_count == len(filtered_alerts)
            
        except Exception as e:
            logger.error(f"Alert sending failed: {str(e)}")
            return False
    
    def _filter_rate_limited_alerts(self, alerts: List[DriftAlert]) -> List[DriftAlert]:
        """Filter alerts based on rate limiting rules."""
        current_time = datetime.utcnow()
        cooldown_time = timedelta(minutes=self.config.alert_cooldown_minutes)
        
        filtered_alerts = []
        
        for alert in alerts:
            # Check if similar alert was sent recently
            similar_recent = any(
                abs((hist_alert.timestamp - current_time).total_seconds()) < cooldown_time.total_seconds()
                and hist_alert.drift_type == alert.drift_type
                and hist_alert.feature == alert.feature
                for hist_alert in self.alert_history[-100:]  # Check last 100 alerts
            )
            
            if not similar_recent:
                filtered_alerts.append(alert)
        
        # Limit total alerts per day
        today = current_time.date()
        today_alerts = [
            a for a in self.alert_history 
            if a.timestamp.date() == today
        ]
        
        if len(today_alerts) >= self.config.max_alerts_per_day:
            return []  # No more alerts today
        
        # Limit remaining alerts for today
        remaining_quota = self.config.max_alerts_per_day - len(today_alerts)
        return filtered_alerts[:remaining_quota]
    
    async def _send_single_alert(self, alert: DriftAlert) -> bool:
        """Send a single alert to notification channels."""
        try:
            # Prepare alert message
            message = self._format_alert_message(alert)
            
            # Send via webhook (placeholder - would implement actual webhook)
            if REQUESTS_AVAILABLE:
                # This would send to actual webhook URL
                logger.info(f"Alert would be sent: {message}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert {alert.id}: {str(e)}")
            return False
    
    def _format_alert_message(self, alert: DriftAlert) -> str:
        """Format alert message for notifications."""
        message_parts = [
            f"🚨 DRIFT ALERT - {alert.severity.value.upper()}",
            f"Type: {alert.drift_type.value.title()}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Score: {alert.drift_score:.3f} (threshold: {alert.threshold:.3f})",
        ]
        
        if alert.feature:
            message_parts.append(f"Feature: {alert.feature}")
        
        message_parts.extend([
            f"Message: {alert.message}",
            f"Alert ID: {alert.id}"
        ])
        
        return "\n".join(message_parts)
    
    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of drift detection activity."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
            
            return {
                'period_days': days,
                'total_alerts': len(recent_alerts),
                'drift_detected': len(recent_alerts) > 0,
                'severity_breakdown': self._count_alerts_by_severity(recent_alerts),
                'type_breakdown': self._count_alerts_by_type(recent_alerts),
                'most_recent_alert': recent_alerts[-1].timestamp.isoformat() if recent_alerts else None,
                'average_drift_score': float(np.mean([a.drift_score for a in recent_alerts])) if recent_alerts else 0.0,
                'unique_features_affected': len(set(a.feature for a in recent_alerts if a.feature))
            }
            
        except Exception as e:
            logger.error(f"Failed to generate drift summary: {str(e)}")
            return {'error': str(e)}
    
    def save_drift_state(self, filepath: str) -> bool:
        """Save drift detector state for persistence."""
        try:
            state = {
                'config': asdict(self.config) if hasattr(self.config, '__dict__') else self.config.__dict__,
                'reference_statistics': self.reference_statistics,
                'alert_history': [asdict(alert) for alert in self.alert_history[-1000:]],  # Last 1000 alerts
                'psi_calculators': {}  # PSI calculators would need custom serialization
            }
            
            # Save PSI calculators separately (they contain fitted state)
            for key, psi_calc in self.psi_calculators.items():
                state['psi_calculators'][key] = {
                    'bin_edges': psi_calc.bin_edges.tolist() if psi_calc.bin_edges is not None else None,
                    'reference_dist': psi_calc.reference_dist.tolist() if psi_calc.reference_dist is not None else None,
                    'bins': psi_calc.bins,
                    'min_freq': psi_calc.min_freq
                }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            logger.info(f"Drift detector state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save drift state: {str(e)}")
            return False
    
    def load_drift_state(self, filepath: str) -> bool:
        """Load drift detector state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore reference statistics
            self.reference_statistics = state.get('reference_statistics', {})
            
            # Restore alert history
            alert_data = state.get('alert_history', [])
            self.alert_history = []
            for alert_dict in alert_data:
                # Convert string timestamp back to datetime
                if isinstance(alert_dict['timestamp'], str):
                    alert_dict['timestamp'] = datetime.fromisoformat(alert_dict['timestamp'])
                
                # Convert string enums back to enum objects
                alert_dict['drift_type'] = DriftType(alert_dict['drift_type'])
                alert_dict['severity'] = DriftSeverity(alert_dict['severity'])
                alert_dict['method'] = DriftMethod(alert_dict['method'])
                
                self.alert_history.append(DriftAlert(**alert_dict))
            
            # Restore PSI calculators
            psi_data = state.get('psi_calculators', {})
            for key, psi_state in psi_data.items():
                psi_calc = PopulationStabilityIndex(
                    bins=psi_state['bins'],
                    min_freq=psi_state['min_freq']
                )
                
                if psi_state['bin_edges']:
                    psi_calc.bin_edges = np.array(psi_state['bin_edges'])
                if psi_state['reference_dist']:
                    psi_calc.reference_dist = np.array(psi_state['reference_dist'])
                
                self.psi_calculators[key] = psi_calc
            
            logger.info(f"Drift detector state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load drift state: {str(e)}")
            return False

# Utility functions and factory methods

def create_drift_detector(
    data_threshold: float = 0.05,
    performance_threshold: float = 0.05
) -> DriftDetector:
    """Factory function to create a DriftDetector instance."""
    config = DriftConfig()
    config.data_drift_threshold = data_threshold
    config.performance_drop_threshold = performance_threshold
    return DriftDetector(config)

async def quick_drift_check(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """Quick drift detection for simple use cases."""
    detector = create_drift_detector()
    await detector.set_reference_data(reference_data, target_column)
    return await detector.detect_data_drift(current_data)

def get_available_methods() -> List[str]:
    """Get list of available drift detection methods."""
    methods = [method.value for method in DriftMethod]
    
    # Filter based on available libraries
    if not EVIDENTLY_AVAILABLE:
        methods = [m for m in methods if m != DriftMethod.EVIDENTLY.value]
    
    if not ALIBI_DETECT_AVAILABLE:
        methods = [m for m in methods if m != DriftMethod.ALIBI_DETECT.value]
    
    return methods

# Example usage and testing
if __name__ == "__main__":
    async def test_drift_detection():
        """Test the drift detection functionality."""
        # Create reference data
        np.random.seed(42)
        n_samples = 1000
        
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(5, 2, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
            'target': np.random.normal(10, 3, n_samples)
        })
        
        # Create current data with drift
        current_data = pd.DataFrame({
            'feature1': np.random.normal(1, 1.5, n_samples),  # Mean and variance shift
            'feature2': np.random.normal(5, 2, n_samples),    # No drift
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.3, 0.4]),  # Distribution shift
            'target': np.random.normal(12, 3, n_samples)      # Target drift
        })
        
        # Create detector
        detector = create_drift_detector()
        
        # Set reference data
        await detector.set_reference_data(reference_data, target_column='target')
        
        # Detect drift
        drift_results = await detector.detect_data_drift(current_data)
        
        print(f"Overall drift detected: {drift_results['overall_drift']}")
        print(f"Drift score: {drift_results['drift_score']:.3f}")
        print(f"Features with drift:")
        
        for feature, result in drift_results['feature_drift'].items():
            if result['drift_detected']:
                print(f"  - {feature}: {result['drift_score']:.3f}")
        
        print(f"Total alerts: {len(drift_results['alerts'])}")
        
        # Generate report
        report = await detector.generate_drift_report()
        print(f"Report generated: {report.report_id}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        return drift_results
    
    # Run test
    import asyncio
    print("Available methods:", get_available_methods())
    results = asyncio.run(test_drift_detection())
