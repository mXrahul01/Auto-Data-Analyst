"""
Comprehensive Monitoring and Observability Module for Auto-Analyst Backend

This module provides production-ready monitoring, logging, and alerting capabilities
for the Auto-Analyst platform, including ML pipeline observability, system health
tracking, and anomaly detection.

Features:
- Structured logging with configurable verbosity and formatting
- Prometheus-compatible metrics collection (counters, gauges, histograms)
- Model and data drift detection with Evidently integration
- Multi-channel alerting system (email, webhook, Slack)
- System health monitoring and resource tracking
- Performance metrics and execution time measurement
- Error tracking and exception handling
- Real-time dashboard metrics for ML pipelines

Components:
- MonitoringManager: Central coordinator for all monitoring activities
- MetricsCollector: Prometheus metrics collection and management
- DriftDetector: Model and data drift detection with alerts
- AlertManager: Multi-channel notification system
- PerformanceTracker: Execution time and resource monitoring
- SystemHealthMonitor: Infrastructure and service health tracking

Usage:
    # Initialize monitoring
    monitor = MonitoringManager()
    
    # Track metrics
    monitor.metrics.increment_counter('ml_training_started')
    monitor.metrics.observe_histogram('training_duration', 150.0)
    
    # Monitor drift
    drift_result = monitor.detect_drift(reference_data, current_data)
    
    # Send alerts
    monitor.alert_manager.send_alert('critical', 'Model drift detected')
    
    # Track performance
    with monitor.track_performance('data_preprocessing'):
        # Your code here
        pass

Dependencies:
- prometheus_client: Metrics collection and exposure
- evidently: Data and model drift detection (optional)
- smtplib: Email notifications
- requests: Webhook notifications
- psutil: System resource monitoring
- logging: Structured logging
"""

import asyncio
import logging
import time
import uuid
import json
import smtplib
import threading
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import traceback

# System monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Prometheus metrics imports
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Drift detection imports
try:
    import evidently
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

# Data processing imports
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# HTTP client for webhooks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='evidently')

# Configure module logger
logger = logging.getLogger(__name__)

class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(str, Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class DriftType(str, Enum):
    """Types of drift detection."""
    DATA_DRIFT = "data_drift"
    TARGET_DRIFT = "target_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"

class MonitoringStatus(str, Enum):
    """Monitoring system status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"

@dataclass
class MetricConfig:
    """Configuration for metric collection."""
    
    # Prometheus settings
    enable_prometheus: bool = True
    metrics_port: int = 8000
    metrics_path: str = "/metrics"
    
    # Collection settings
    collection_interval: float = 10.0  # seconds
    retention_days: int = 30
    
    # Performance settings
    enable_histograms: bool = True
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])

@dataclass
class AlertConfig:
    """Configuration for alerting system."""
    
    # Email settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    from_email: Optional[str] = None
    to_emails: List[str] = field(default_factory=list)
    
    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)
    webhook_timeout: int = 30
    
    # Slack settings
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#alerts"
    
    # Alert settings
    rate_limit_minutes: int = 5
    max_alerts_per_hour: int = 60
    enable_batching: bool = True
    batch_size: int = 10

@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    
    # Detection settings
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    confidence_level: float = 0.95
    
    # Monitoring frequency
    check_interval_hours: int = 1
    batch_size: int = 1000
    
    # Alert settings
    alert_on_drift: bool = True
    drift_alert_threshold: float = 0.2
    
    # Data settings
    reference_window_days: int = 30
    comparison_window_days: int = 7

@dataclass
class Alert:
    """Alert message structure."""
    
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

class MetricsCollector:
    """Prometheus-compatible metrics collection and management."""
    
    def __init__(self, config: MetricConfig, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            config: Metric collection configuration
            registry: Prometheus registry (creates new if None)
        """
        self.config = config
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        
        # Initialize core metrics if Prometheus is available
        if PROMETHEUS_AVAILABLE and config.enable_prometheus:
            self._initialize_core_metrics()
        
        logger.info("MetricsCollector initialized")
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core application metrics."""
        try:
            # System metrics
            self.metrics['system_cpu_usage'] = Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage',
                registry=self.registry
            )
            
            self.metrics['system_memory_usage'] = Gauge(
                'system_memory_usage_bytes',
                'System memory usage in bytes',
                registry=self.registry
            )
            
            self.metrics['system_disk_usage'] = Gauge(
                'system_disk_usage_percent',
                'System disk usage percentage',
                registry=self.registry
            )
            
            # Application metrics
            self.metrics['http_requests_total'] = Counter(
                'http_requests_total',
                'Total HTTP requests',
                ['method', 'endpoint', 'status'],
                registry=self.registry
            )
            
            self.metrics['http_request_duration'] = Histogram(
                'http_request_duration_seconds',
                'HTTP request duration in seconds',
                ['method', 'endpoint'],
                buckets=self.config.histogram_buckets,
                registry=self.registry
            )
            
            # ML pipeline metrics
            self.metrics['ml_analyses_total'] = Counter(
                'ml_analyses_total',
                'Total ML analyses',
                ['status', 'task_type'],
                registry=self.registry
            )
            
            self.metrics['ml_analysis_duration'] = Histogram(
                'ml_analysis_duration_seconds',
                'ML analysis duration in seconds',
                ['task_type'],
                buckets=self.config.histogram_buckets,
                registry=self.registry
            )
            
            self.metrics['model_accuracy'] = Gauge(
                'model_accuracy_score',
                'Model accuracy score',
                ['model_name', 'dataset_id'],
                registry=self.registry
            )
            
            self.metrics['data_drift_score'] = Gauge(
                'data_drift_score',
                'Data drift score',
                ['dataset_id', 'drift_type'],
                registry=self.registry
            )
            
            # Dataset metrics
            self.metrics['datasets_total'] = Counter(
                'datasets_total',
                'Total datasets processed',
                ['status', 'format'],
                registry=self.registry
            )
            
            self.metrics['dataset_size'] = Histogram(
                'dataset_size_bytes',
                'Dataset size in bytes',
                buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824],
                registry=self.registry
            )
            
            # Error metrics
            self.metrics['errors_total'] = Counter(
                'errors_total',
                'Total errors',
                ['error_type', 'service'],
                registry=self.registry
            )
            
            logger.info("Core metrics initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize core metrics: {str(e)}")
    
    def create_counter(self, name: str, description: str, labels: Optional[List[str]] = None) -> Optional[Counter]:
        """
        Create a new counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional labels for the metric
            
        Returns:
            Counter instance or None if Prometheus not available
        """
        if not PROMETHEUS_AVAILABLE or not self.config.enable_prometheus:
            return None
        
        try:
            counter = Counter(name, description, labels or [], registry=self.registry)
            self.metrics[name] = counter
            return counter
        except Exception as e:
            logger.error(f"Failed to create counter {name}: {str(e)}")
            return None
    
    def create_gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Optional[Gauge]:
        """
        Create a new gauge metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional labels for the metric
            
        Returns:
            Gauge instance or None if Prometheus not available
        """
        if not PROMETHEUS_AVAILABLE or not self.config.enable_prometheus:
            return None
        
        try:
            gauge = Gauge(name, description, labels or [], registry=self.registry)
            self.metrics[name] = gauge
            return gauge
        except Exception as e:
            logger.error(f"Failed to create gauge {name}: {str(e)}")
            return None
    
    def create_histogram(self, name: str, description: str, labels: Optional[List[str]] = None) -> Optional[Histogram]:
        """
        Create a new histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional labels for the metric
            
        Returns:
            Histogram instance or None if Prometheus not available
        """
        if not PROMETHEUS_AVAILABLE or not self.config.enable_prometheus:
            return None
        
        try:
            histogram = Histogram(
                name, description, labels or [], 
                buckets=self.config.histogram_buckets,
                registry=self.registry
            )
            self.metrics[name] = histogram
            return histogram
        except Exception as e:
            logger.error(f"Failed to create histogram {name}: {str(e)}")
            return None
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            labels: Label values
            value: Increment value
        """
        try:
            if name in self.metrics and hasattr(self.metrics[name], 'inc'):
                if labels:
                    self.metrics[name].labels(**labels).inc(value)
                else:
                    self.metrics[name].inc(value)
        except Exception as e:
            logger.error(f"Failed to increment counter {name}: {str(e)}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Value to set
            labels: Label values
        """
        try:
            if name in self.metrics and hasattr(self.metrics[name], 'set'):
                if labels:
                    self.metrics[name].labels(**labels).set(value)
                else:
                    self.metrics[name].set(value)
        except Exception as e:
            logger.error(f"Failed to set gauge {name}: {str(e)}")
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value in a histogram metric.
        
        Args:
            name: Histogram name
            value: Value to observe
            labels: Label values
        """
        try:
            if name in self.metrics and hasattr(self.metrics[name], 'observe'):
                if labels:
                    self.metrics[name].labels(**labels).observe(value)
                else:
                    self.metrics[name].observe(value)
        except Exception as e:
            logger.error(f"Failed to observe histogram {name}: {str(e)}")
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge('system_cpu_usage', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge('system_memory_usage', memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.set_gauge('system_disk_usage', disk_percent)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {str(e)}")
    
    def get_metrics_text(self) -> str:
        """
        Get metrics in Prometheus text format.
        
        Returns:
            Metrics as text string
        """
        if not PROMETHEUS_AVAILABLE or not self.config.enable_prometheus:
            return "# Prometheus metrics not available\n"
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics text: {str(e)}")
            return f"# Error generating metrics: {str(e)}\n"

class AlertManager:
    """Multi-channel alerting system for monitoring events."""
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.alert_history: List[Alert] = []
        self.rate_limiter: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        logger.info("AlertManager initialized")
    
    async def send_alert(
        self, 
        level: AlertLevel, 
        title: str, 
        message: str, 
        source: str = "monitoring",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert through configured channels.
        
        Args:
            level: Alert severity level
            title: Alert title
            message: Alert message
            source: Source of the alert
            metadata: Additional metadata
            
        Returns:
            True if alert was sent successfully
        """
        try:
            # Create alert object
            alert = Alert(
                id=str(uuid.uuid4()),
                level=level,
                title=title,
                message=message,
                timestamp=datetime.now(),
                source=source,
                metadata=metadata or {}
            )
            
            # Check rate limiting
            if not self._check_rate_limit(alert):
                logger.warning(f"Alert rate limited: {title}")
                return False
            
            # Store alert
            with self._lock:
                self.alert_history.append(alert)
                # Keep only recent alerts
                cutoff = datetime.now() - timedelta(days=7)
                self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff]
            
            # Send through configured channels
            success = True
            
            # Email notification
            if self.config.to_emails and self.config.smtp_server:
                email_success = await self._send_email_alert(alert)
                success = success and email_success
            
            # Webhook notification
            if self.config.webhook_urls:
                webhook_success = await self._send_webhook_alert(alert)
                success = success and webhook_success
            
            # Slack notification
            if self.config.slack_webhook_url:
                slack_success = await self._send_slack_alert(alert)
                success = success and slack_success
            
            if success:
                logger.info(f"Alert sent successfully: {title}")
            else:
                logger.error(f"Failed to send alert: {title}")
            
            return success
            
        except Exception as e:
            logger.error(f"Alert sending failed: {str(e)}")
            return False
    
    def _check_rate_limit(self, alert: Alert) -> bool:
        """
        Check if alert should be rate limited.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be sent
        """
        try:
            rate_limit_key = f"{alert.level.value}:{alert.title}"
            
            with self._lock:
                now = datetime.now()
                
                # Check if we've sent this type of alert recently
                if rate_limit_key in self.rate_limiter:
                    last_sent = self.rate_limiter[rate_limit_key]
                    time_diff = (now - last_sent).total_seconds() / 60
                    
                    if time_diff < self.config.rate_limit_minutes:
                        return False
                
                # Update rate limiter
                self.rate_limiter[rate_limit_key] = now
                
                # Clean up old entries
                cutoff = now - timedelta(minutes=self.config.rate_limit_minutes * 2)
                self.rate_limiter = {
                    k: v for k, v in self.rate_limiter.items() if v > cutoff
                }
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return True  # Allow alert on error
    
    async def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            if not self.config.smtp_server or not self.config.to_emails:
                return False
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.config.from_email or self.config.email_username
            msg['To'] = ', '.join(self.config.to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Level: {alert.level.value.upper()}
- Title: {alert.title}
- Message: {alert.message}
- Source: {alert.source}
- Timestamp: {alert.timestamp.isoformat()}
- Alert ID: {alert.id}

Metadata:
{json.dumps(alert.metadata, indent=2)}

---
Auto-Analyst Monitoring System
            """.strip()
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                if self.config.email_username and self.config.email_password:
                    server.login(self.config.email_username, self.config.email_password)
                
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Email alert failed: {str(e)}")
            return False
    
    async def _send_webhook_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available for webhook alerts")
            return False
        
        try:
            payload = {
                'id': alert.id,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            success = True
            for webhook_url in self.config.webhook_urls:
                try:
                    response = requests.post(
                        webhook_url,
                        json=payload,
                        timeout=self.config.webhook_timeout,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Webhook alert sent to {webhook_url}: {alert.title}")
                    else:
                        logger.error(f"Webhook alert failed for {webhook_url}: {response.status_code}")
                        success = False
                        
                except Exception as e:
                    logger.error(f"Webhook alert failed for {webhook_url}: {str(e)}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Webhook alert failed: {str(e)}")
            return False
    
    async def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert via Slack webhook."""
        if not REQUESTS_AVAILABLE or not self.config.slack_webhook_url:
            return False
        
        try:
            # Create Slack message
            color = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning", 
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "danger"
            }.get(alert.level, "warning")
            
            payload = {
                'channel': self.config.slack_channel,
                'username': 'Auto-Analyst Monitor',
                'icon_emoji': ':warning:',
                'attachments': [{
                    'color': color,
                    'title': f"[{alert.level.value.upper()}] {alert.title}",
                    'text': alert.message,
                    'fields': [
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Timestamp', 'value': alert.timestamp.isoformat(), 'short': True},
                        {'title': 'Alert ID', 'value': alert.id, 'short': False}
                    ],
                    'footer': 'Auto-Analyst Monitoring',
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=self.config.webhook_timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Slack alert failed: {str(e)}")
            return False
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """
        Get recent alerts within specified time window.
        
        Args:
            hours: Time window in hours
            
        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [alert for alert in self.alert_history if alert.timestamp > cutoff]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alerting statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        with self._lock:
            recent_alerts = self.get_recent_alerts(24)
            
            stats = {
                'total_alerts_24h': len(recent_alerts),
                'alerts_by_level': {},
                'alerts_by_source': {},
                'rate_limited_alerts': len(self.rate_limiter)
            }
            
            for alert in recent_alerts:
                # By level
                level = alert.level.value
                stats['alerts_by_level'][level] = stats['alerts_by_level'].get(level, 0) + 1
                
                # By source
                source = alert.source
                stats['alerts_by_source'][source] = stats['alerts_by_source'].get(source, 0) + 1
            
            return stats

class DriftDetector:
    """Model and data drift detection with alerting capabilities."""
    
    def __init__(self, config: DriftConfig, alert_manager: Optional[AlertManager] = None):
        """
        Initialize drift detector.
        
        Args:
            config: Drift detection configuration
            alert_manager: Alert manager for notifications
        """
        self.config = config
        self.alert_manager = alert_manager
        self.drift_history: List[Dict[str, Any]] = []
        
        logger.info("DriftDetector initialized")
    
    async def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current datasets.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset to compare
            feature_columns: Columns to analyze (all if None)
            
        Returns:
            Drift detection results
        """
        if not EVIDENTLY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.warning("Evidently or pandas not available for drift detection")
            return self._mock_drift_result(DriftType.DATA_DRIFT, False)
        
        try:
            # Prepare data
            if feature_columns:
                ref_data = reference_data[feature_columns].copy()
                cur_data = current_data[feature_columns].copy()
            else:
                ref_data = reference_data.copy()
                cur_data = current_data.copy()
            
            # Create drift report
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_data, current_data=cur_data)
            
            # Extract results
            drift_results = report.as_dict()
            dataset_drift = drift_results['metrics'][0]['result']
            
            # Process results
            drift_detected = dataset_drift.get('dataset_drift', False)
            drift_score = dataset_drift.get('drift_score', 0.0)
            
            # Feature-level drift
            feature_drifts = {}
            for feature_result in dataset_drift.get('drift_by_columns', []):
                feature_name = feature_result.get('column_name')
                feature_drift_score = feature_result.get('drift_score', 0)
                feature_drifts[feature_name] = feature_drift_score
            
            # Create result
            result = {
                'drift_type': DriftType.DATA_DRIFT.value,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'threshold': self.config.drift_threshold,
                'feature_drifts': feature_drifts,
                'timestamp': datetime.now().isoformat(),
                'sample_sizes': {
                    'reference': len(ref_data),
                    'current': len(cur_data)
                }
            }
            
            # Store result
            self.drift_history.append(result)
            
            # Send alert if drift detected and configured
            if (drift_detected and self.config.alert_on_drift and 
                drift_score > self.config.drift_alert_threshold and
                self.alert_manager):
                
                await self.alert_manager.send_alert(
                    level=AlertLevel.WARNING,
                    title="Data Drift Detected",
                    message=f"Data drift detected with score {drift_score:.3f} (threshold: {self.config.drift_alert_threshold:.3f})",
                    source="drift_detector",
                    metadata=result
                )
            
            logger.info(f"Data drift detection completed: drift_detected={drift_detected}, score={drift_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {str(e)}")
            return self._mock_drift_result(DriftType.DATA_DRIFT, False, error=str(e))
    
    async def detect_target_drift(
        self,
        reference_targets: pd.Series,
        current_targets: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect target drift between reference and current targets.
        
        Args:
            reference_targets: Reference target values
            current_targets: Current target values to compare
            
        Returns:
            Target drift detection results
        """
        if not EVIDENTLY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.warning("Evidently or pandas not available for drift detection")
            return self._mock_drift_result(DriftType.TARGET_DRIFT, False)
        
        try:
            # Prepare data
            ref_df = pd.DataFrame({'target': reference_targets})
            cur_df = pd.DataFrame({'target': current_targets})
            
            # Create drift report
            report = Report(metrics=[TargetDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            
            # Extract results
            drift_results = report.as_dict()
            target_drift = drift_results['metrics'][0]['result']
            
            drift_detected = target_drift.get('drift_detected', False)
            drift_score = target_drift.get('drift_score', 0.0)
            
            # Create result
            result = {
                'drift_type': DriftType.TARGET_DRIFT.value,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'threshold': self.config.drift_threshold,
                'timestamp': datetime.now().isoformat(),
                'sample_sizes': {
                    'reference': len(reference_targets),
                    'current': len(current_targets)
                }
            }
            
            # Store result
            self.drift_history.append(result)
            
            # Send alert if drift detected
            if (drift_detected and self.config.alert_on_drift and 
                drift_score > self.config.drift_alert_threshold and
                self.alert_manager):
                
                await self.alert_manager.send_alert(
                    level=AlertLevel.WARNING,
                    title="Target Drift Detected",
                    message=f"Target drift detected with score {drift_score:.3f}",
                    source="drift_detector",
                    metadata=result
                )
            
            logger.info(f"Target drift detection completed: drift_detected={drift_detected}, score={drift_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Target drift detection failed: {str(e)}")
            return self._mock_drift_result(DriftType.TARGET_DRIFT, False, error=str(e))
    
    def _mock_drift_result(
        self, 
        drift_type: DriftType, 
        drift_detected: bool,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create mock drift result when dependencies not available."""
        return {
            'drift_type': drift_type.value,
            'drift_detected': drift_detected,
            'drift_score': 0.0,
            'threshold': self.config.drift_threshold,
            'feature_drifts': {},
            'timestamp': datetime.now().isoformat(),
            'sample_sizes': {'reference': 0, 'current': 0},
            'error': error,
            'mock_result': True
        }
    
    def get_drift_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get drift detection history.
        
        Args:
            hours: Time window in hours
            
        Returns:
            List of drift detection results
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            result for result in self.drift_history
            if datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00')) > cutoff
        ]

class PerformanceTracker:
    """Performance monitoring and execution time tracking."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize performance tracker.
        
        Args:
            metrics_collector: Metrics collector for recording measurements
        """
        self.metrics_collector = metrics_collector
        self.active_tracks: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        logger.info("PerformanceTracker initialized")
    
    @contextmanager
    def track_performance(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for tracking operation performance.
        
        Args:
            operation_name: Name of the operation being tracked
            labels: Optional labels for metrics
            
        Yields:
            Performance tracking context
        """
        start_time = time.time()
        track_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            self.active_tracks[track_id] = start_time
        
        try:
            yield track_id
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            with self._lock:
                self.active_tracks.pop(track_id, None)
            
            # Record metrics
            if self.metrics_collector:
                # Try to find existing histogram or create generic one
                histogram_name = f"{operation_name}_duration"
                if histogram_name in self.metrics_collector.metrics:
                    self.metrics_collector.observe_histogram(histogram_name, duration, labels)
                else:
                    # Use generic performance histogram
                    generic_labels = {'operation': operation_name}
                    if labels:
                        generic_labels.update(labels)
                    self.metrics_collector.observe_histogram('operation_duration_seconds', duration, generic_labels)
            
            logger.debug(f"Operation '{operation_name}' completed in {duration:.3f} seconds")
    
    def measure_execution_time(self, func: Callable) -> Callable:
        """
        Decorator for measuring function execution time.
        
        Args:
            func: Function to measure
            
        Returns:
            Decorated function with timing
        """
        def wrapper(*args, **kwargs):
            with self.track_performance(func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    
    def get_active_operations(self) -> Dict[str, float]:
        """
        Get currently active operations and their durations.
        
        Returns:
            Dictionary mapping operation IDs to current duration
        """
        current_time = time.time()
        
        with self._lock:
            return {
                track_id: current_time - start_time
                for track_id, start_time in self.active_tracks.items()
            }

class SystemHealthMonitor:
    """System health monitoring and status reporting."""
    
    def __init__(
        self, 
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        """
        Initialize system health monitor.
        
        Args:
            metrics_collector: Metrics collector for recording health metrics
            alert_manager: Alert manager for health alerts
        """
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.health_checks: Dict[str, Callable] = {}
        self.last_check: Optional[datetime] = None
        self.health_status: MonitoringStatus = MonitoringStatus.HEALTHY
        
        # Register default health checks
        self._register_default_health_checks()
        
        logger.info("SystemHealthMonitor initialized")
    
    def _register_default_health_checks(self) -> None:
        """Register default system health checks."""
        
        def check_disk_space() -> Dict[str, Any]:
            """Check disk space availability."""
            if not PSUTIL_AVAILABLE:
                return {'status': 'unknown', 'message': 'psutil not available'}
            
            try:
                disk = psutil.disk_usage('/')
                free_percent = (disk.free / disk.total) * 100
                
                if free_percent < 5:
                    return {'status': 'critical', 'message': f'Disk space critically low: {free_percent:.1f}% free'}
                elif free_percent < 15:
                    return {'status': 'warning', 'message': f'Disk space low: {free_percent:.1f}% free'}
                else:
                    return {'status': 'healthy', 'message': f'Disk space OK: {free_percent:.1f}% free'}
                    
            except Exception as e:
                return {'status': 'error', 'message': f'Disk check failed: {str(e)}'}
        
        def check_memory_usage() -> Dict[str, Any]:
            """Check memory usage."""
            if not PSUTIL_AVAILABLE:
                return {'status': 'unknown', 'message': 'psutil not available'}
            
            try:
                memory = psutil.virtual_memory()
                used_percent = memory.percent
                
                if used_percent > 90:
                    return {'status': 'critical', 'message': f'Memory usage critically high: {used_percent:.1f}%'}
                elif used_percent > 80:
                    return {'status': 'warning', 'message': f'Memory usage high: {used_percent:.1f}%'}
                else:
                    return {'status': 'healthy', 'message': f'Memory usage OK: {used_percent:.1f}%'}
                    
            except Exception as e:
                return {'status': 'error', 'message': f'Memory check failed: {str(e)}'}
        
        def check_cpu_usage() -> Dict[str, Any]:
            """Check CPU usage."""
            if not PSUTIL_AVAILABLE:
                return {'status': 'unknown', 'message': 'psutil not available'}
            
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                
                if cpu_percent > 95:
                    return {'status': 'critical', 'message': f'CPU usage critically high: {cpu_percent:.1f}%'}
                elif cpu_percent > 85:
                    return {'status': 'warning', 'message': f'CPU usage high: {cpu_percent:.1f}%'}
                else:
                    return {'status': 'healthy', 'message': f'CPU usage OK: {cpu_percent:.1f}%'}
                    
            except Exception as e:
                return {'status': 'error', 'message': f'CPU check failed: {str(e)}'}
        
        # Register checks
        self.health_checks['disk_space'] = check_disk_space
        self.health_checks['memory_usage'] = check_memory_usage
        self.health_checks['cpu_usage'] = check_cpu_usage
    
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Name of the health check
            check_func: Function that returns health status dict
        """
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary with overall health status and individual check results
        """
        try:
            check_results = {}
            overall_status = MonitoringStatus.HEALTHY
            
            # Run each health check
            for check_name, check_func in self.health_checks.items():
                try:
                    result = check_func()
                    check_results[check_name] = result
                    
                    # Update overall status
                    check_status = result.get('status', 'unknown')
                    if check_status == 'critical':
                        overall_status = MonitoringStatus.CRITICAL
                    elif check_status == 'warning' and overall_status != MonitoringStatus.CRITICAL:
                        overall_status = MonitoringStatus.WARNING
                    elif check_status == 'error' and overall_status == MonitoringStatus.HEALTHY:
                        overall_status = MonitoringStatus.DEGRADED
                        
                except Exception as e:
                    check_results[check_name] = {
                        'status': 'error',
                        'message': f'Health check failed: {str(e)}'
                    }
                    if overall_status == MonitoringStatus.HEALTHY:
                        overall_status = MonitoringStatus.DEGRADED
            
            # Update system status
            previous_status = self.health_status
            self.health_status = overall_status
            self.last_check = datetime.now()
            
            # Send alerts for status changes
            if (self.alert_manager and previous_status != overall_status and 
                overall_status in [MonitoringStatus.CRITICAL, MonitoringStatus.WARNING]):
                
                await self.alert_manager.send_alert(
                    level=AlertLevel.CRITICAL if overall_status == MonitoringStatus.CRITICAL else AlertLevel.WARNING,
                    title=f"System Health Status Changed: {overall_status.value.upper()}",
                    message=f"System health status changed from {previous_status.value} to {overall_status.value}",
                    source="health_monitor",
                    metadata={'check_results': check_results}
                )
            
            # Prepare response
            health_report = {
                'status': overall_status.value,
                'timestamp': self.last_check.isoformat(),
                'checks': check_results,
                'summary': {
                    'total_checks': len(check_results),
                    'healthy_checks': len([r for r in check_results.values() if r.get('status') == 'healthy']),
                    'warning_checks': len([r for r in check_results.values() if r.get('status') == 'warning']),
                    'critical_checks': len([r for r in check_results.values() if r.get('status') == 'critical']),
                    'error_checks': len([r for r in check_results.values() if r.get('status') == 'error'])
                }
            }
            
            logger.debug(f"Health checks completed: status={overall_status.value}")
            return health_report
            
        except Exception as e:
            logger.error(f"Health check execution failed: {str(e)}")
            return {
                'status': MonitoringStatus.CRITICAL.value,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'checks': {}
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status without running checks.
        
        Returns:
            Current health status information
        """
        return {
            'status': self.health_status.value,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'registered_checks': list(self.health_checks.keys())
        }

class MonitoringManager:
    """Central coordinator for all monitoring activities."""
    
    def __init__(
        self,
        metric_config: Optional[MetricConfig] = None,
        alert_config: Optional[AlertConfig] = None,
        drift_config: Optional[DriftConfig] = None
    ):
        """
        Initialize monitoring manager.
        
        Args:
            metric_config: Metrics configuration
            alert_config: Alerting configuration
            drift_config: Drift detection configuration
        """
        self.metric_config = metric_config or MetricConfig()
        self.alert_config = alert_config or AlertConfig()
        self.drift_config = drift_config or DriftConfig()
        
        # Initialize components
        self.metrics = MetricsCollector(self.metric_config)
        self.alert_manager = AlertManager(self.alert_config)
        self.drift_detector = DriftDetector(self.drift_config, self.alert_manager)
        self.performance_tracker = PerformanceTracker(self.metrics)
        self.health_monitor = SystemHealthMonitor(self.metrics, self.alert_manager)
        
        # Background tasks
        self._monitoring_active = True
        self._background_tasks: List[asyncio.Task] = []
        
        logger.info("MonitoringManager initialized successfully")
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Start system metrics collection
            if self.metric_config.enable_prometheus:
                metrics_task = asyncio.create_task(self._system_metrics_loop())
                self._background_tasks.append(metrics_task)
            
            # Start health monitoring
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self._background_tasks.append(health_task)
            
            logger.info("Background monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        try:
            self._monitoring_active = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            logger.info("Background monitoring tasks stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
    
    async def _system_metrics_loop(self) -> None:
        """Background loop for collecting system metrics."""
        while self._monitoring_active:
            try:
                self.metrics.update_system_metrics()
                await asyncio.sleep(self.metric_config.collection_interval)
                
            except Exception as e:
                logger.error(f"System metrics collection failed: {str(e)}")
                await asyncio.sleep(self.metric_config.collection_interval)
    
    async def _health_monitoring_loop(self) -> None:
        """Background loop for health monitoring."""
        while self._monitoring_active:
            try:
                await self.health_monitor.run_health_checks()
                await asyncio.sleep(60)  # Check health every minute
                
            except Exception as e:
                logger.error(f"Health monitoring failed: {str(e)}")
                await asyncio.sleep(60)
    
    async def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_type: DriftType = DriftType.DATA_DRIFT,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            drift_type: Type of drift to detect
            target_column: Target column for target drift detection
            
        Returns:
            Drift detection results
        """
        try:
            if drift_type == DriftType.DATA_DRIFT:
                return await self.drift_detector.detect_data_drift(reference_data, current_data)
            
            elif drift_type == DriftType.TARGET_DRIFT and target_column:
                ref_targets = reference_data[target_column]
                cur_targets = current_data[target_column]
                return await self.drift_detector.detect_target_drift(ref_targets, cur_targets)
            
            else:
                raise ValueError(f"Unsupported drift type: {drift_type}")
                
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            return {
                'drift_type': drift_type.value,
                'drift_detected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring status.
        
        Returns:
            Dictionary with monitoring system status
        """
        try:
            return {
                'monitoring_active': self._monitoring_active,
                'components': {
                    'metrics_collector': {
                        'enabled': self.metric_config.enable_prometheus,
                        'prometheus_available': PROMETHEUS_AVAILABLE,
                        'total_metrics': len(self.metrics.metrics)
                    },
                    'alert_manager': {
                        'email_configured': bool(self.alert_config.smtp_server and self.alert_config.to_emails),
                        'webhook_configured': bool(self.alert_config.webhook_urls),
                        'slack_configured': bool(self.alert_config.slack_webhook_url),
                        'recent_alerts': len(self.alert_manager.get_recent_alerts())
                    },
                    'drift_detector': {
                        'enabled': self.drift_config.enable_drift_detection,
                        'evidently_available': EVIDENTLY_AVAILABLE,
                        'recent_detections': len(self.drift_detector.get_drift_history())
                    },
                    'health_monitor': self.health_monitor.get_health_status()
                },
                'background_tasks': len(self._background_tasks),
                'dependencies': {
                    'prometheus_client': PROMETHEUS_AVAILABLE,
                    'evidently': EVIDENTLY_AVAILABLE,
                    'psutil': PSUTIL_AVAILABLE,
                    'requests': REQUESTS_AVAILABLE,
                    'pandas': PANDAS_AVAILABLE
                }
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {str(e)}")
            return {'error': str(e)}

# Convenience functions for easy import and usage

def create_monitoring_manager(
    metric_config: Optional[MetricConfig] = None,
    alert_config: Optional[AlertConfig] = None,
    drift_config: Optional[DriftConfig] = None
) -> MonitoringManager:
    """
    Factory function to create a monitoring manager.
    
    Args:
        metric_config: Metrics configuration
        alert_config: Alerting configuration
        drift_config: Drift detection configuration
        
    Returns:
        Configured MonitoringManager instance
    """
    return MonitoringManager(metric_config, alert_config, drift_config)

def get_default_monitoring_manager() -> MonitoringManager:
    """Get monitoring manager with default configuration."""
    return create_monitoring_manager()

# Logging utilities
def log_info(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log info message with optional extra data."""
    logger.info(message, extra=extra)

def log_warning(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log warning message with optional extra data."""
    logger.warning(message, extra=extra)

def log_error(message: str, exception: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log error message with optional exception and extra data."""
    if exception:
        logger.error(f"{message}: {str(exception)}", extra=extra, exc_info=True)
    else:
        logger.error(message, extra=extra)

# Performance measurement utilities
def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Function '{func.__name__}' executed in {duration:.3f} seconds")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function '{func.__name__}' failed after {duration:.3f} seconds: {str(e)}")
            raise
    
    return wrapper

@contextmanager
def monitor_performance(operation_name: str, logger_instance: Optional[logging.Logger] = None):
    """
    Context manager for monitoring operation performance.
    
    Args:
        operation_name: Name of the operation
        logger_instance: Optional logger instance
        
    Yields:
        Performance monitoring context
    """
    log = logger_instance or logger
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        log.info(f"Operation '{operation_name}' completed in {duration:.3f} seconds")

# Health check utilities
def track_system_health() -> Dict[str, Any]:
    """
    Quick system health check.
    
    Returns:
        Basic system health information
    """
    try:
        health_info = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy'
        }
        
        if PSUTIL_AVAILABLE:
            health_info.update({
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            })
        
        return health_info
        
    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }

def alert_on_threshold(
    metric_name: str,
    current_value: float,
    threshold: float,
    alert_manager: Optional[AlertManager] = None,
    comparison: str = "greater"
) -> bool:
    """
    Send alert if metric value breaches threshold.
    
    Args:
        metric_name: Name of the metric
        current_value: Current metric value
        threshold: Threshold value
        alert_manager: Alert manager instance
        comparison: "greater" or "less" than threshold
        
    Returns:
        True if alert was sent
    """
    try:
        breach_detected = False
        
        if comparison == "greater" and current_value > threshold:
            breach_detected = True
        elif comparison == "less" and current_value < threshold:
            breach_detected = True
        
        if breach_detected and alert_manager:
            asyncio.create_task(alert_manager.send_alert(
                level=AlertLevel.WARNING,
                title=f"Threshold Breach: {metric_name}",
                message=f"Metric '{metric_name}' value {current_value} breached threshold {threshold}",
                source="threshold_monitor",
                metadata={
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'threshold': threshold,
                    'comparison': comparison
                }
            ))
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Threshold alert failed: {str(e)}")
        return False

# Export key functions and classes
__all__ = [
    # Main classes
    'MonitoringManager', 'MetricsCollector', 'AlertManager', 'DriftDetector',
    'PerformanceTracker', 'SystemHealthMonitor',
    
    # Configuration classes
    'MetricConfig', 'AlertConfig', 'DriftConfig',
    
    # Enums
    'AlertLevel', 'MetricType', 'DriftType', 'MonitoringStatus',
    
    # Factory functions
    'create_monitoring_manager', 'get_default_monitoring_manager',
    
    # Utility functions
    'log_info', 'log_warning', 'log_error',
    'measure_execution_time', 'monitor_performance',
    'track_system_health', 'alert_on_threshold'
]

# Initialize module
logger.info(f"Monitoring module loaded - Dependencies: Prometheus={PROMETHEUS_AVAILABLE}, Evidently={EVIDENTLY_AVAILABLE}, psutil={PSUTIL_AVAILABLE}")
