"""
🚀 AUTO-ANALYST PLATFORM - SYSTEM MONITORING UTILITIES
=====================================================

Production-ready monitoring utilities with focus on performance, security,
and maintainability. Provides essential monitoring capabilities without
over-engineering.

Key Features:
- Lightweight metrics collection with Prometheus integration
- Secure alert management with proper credential handling
- Performance tracking with minimal overhead
- Health monitoring with configurable checks
- Async-first design for scalability

Components:
- MetricsCollector: Prometheus-compatible metrics
- AlertManager: Multi-channel alerting with rate limiting
- HealthMonitor: System health checks and status tracking
- PerformanceTracker: Execution time and resource monitoring

Dependencies:
- prometheus_client: Metrics collection (optional)
- aiosmtp: Async email sending (optional)
- aiohttp: Async HTTP client (optional)
- psutil: System metrics (optional)
"""

import asyncio
import json
import logging
import os
import ssl
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, AsyncContextManager,
    Protocol, Union, Tuple
)

# Core dependencies
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Optional dependencies with graceful fallbacks
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, CollectorRegistry, 
        generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import aiosmtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    SMTP_AVAILABLE = True
except ImportError:
    SMTP_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """System health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Supported metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class MonitoringConfig(BaseModel):
    """Configuration for monitoring system."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    # General settings
    environment: str = Field(default="production", description="Environment name")
    service_name: str = Field(default="auto-analyst", description="Service identifier")
    
    # Metrics collection
    enable_metrics: bool = True
    metrics_port: int = Field(default=8000, ge=1000, le=65535)
    collection_interval_seconds: int = Field(default=30, ge=5, le=300)
    
    # Health monitoring
    enable_health_checks: bool = True
    health_check_interval_seconds: int = Field(default=60, ge=10, le=600)
    
    # Performance tracking
    enable_performance_tracking: bool = True
    slow_request_threshold_seconds: float = Field(default=5.0, gt=0.0)
    
    # Resource monitoring
    cpu_warning_threshold: float = Field(default=80.0, gt=0.0, le=100.0)
    memory_warning_threshold: float = Field(default=85.0, gt=0.0, le=100.0)
    disk_warning_threshold: float = Field(default=90.0, gt=0.0, le=100.0)


class AlertConfig(BaseModel):
    """Configuration for alert system."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_window_minutes: int = Field(default=15, ge=1, le=60)
    max_alerts_per_window: int = Field(default=10, ge=1, le=100)
    
    # Email configuration (from environment)
    smtp_host: Optional[str] = Field(default_factory=lambda: os.getenv("SMTP_HOST"))
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: Optional[str] = Field(default_factory=lambda: os.getenv("SMTP_USERNAME"))
    smtp_password: Optional[str] = Field(default_factory=lambda: os.getenv("SMTP_PASSWORD"))
    from_email: Optional[str] = Field(default_factory=lambda: os.getenv("FROM_EMAIL"))
    to_emails: List[str] = Field(default_factory=lambda: 
        os.getenv("ALERT_EMAILS", "").split(",") if os.getenv("ALERT_EMAILS") else []
    )
    
    # Webhook configuration
    webhook_urls: List[str] = Field(default_factory=lambda:
        os.getenv("WEBHOOK_URLS", "").split(",") if os.getenv("WEBHOOK_URLS") else []
    )
    webhook_timeout_seconds: int = Field(default=30, ge=5, le=120)
    
    @field_validator('to_emails', 'webhook_urls')
    @classmethod
    def filter_empty_strings(cls, v: List[str]) -> List[str]:
        """Filter out empty strings from lists."""
        return [item.strip() for item in v if item.strip()]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Alert:
    """Alert message structure."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: AlertLevel = AlertLevel.INFO
    title: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    service: str = "auto-analyst"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'service': self.service,
            'metadata': self.metadata
        }


@dataclass
class HealthCheck:
    """Health check result."""
    
    name: str
    status: HealthStatus
    message: str
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    load_average: Optional[Tuple[float, float, float]] = None
    uptime_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# CORE INTERFACES
# =============================================================================

class HealthCheckProtocol(Protocol):
    """Protocol for health check functions."""
    
    async def __call__(self) -> HealthCheck:
        """Execute health check and return result."""
        ...


class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collectors."""
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        ...
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        ...
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation."""
        ...


# =============================================================================
# METRICS COLLECTION
# =============================================================================

class PrometheusMetricsCollector:
    """Prometheus-compatible metrics collection."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize metrics collector."""
        self.config = config
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        
        if PROMETHEUS_AVAILABLE:
            self._initialize_core_metrics()
        else:
            logger.warning("Prometheus client not available - metrics collection disabled")
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core application metrics."""
        try:
            # System metrics
            self.metrics['cpu_usage'] = Gauge(
                'system_cpu_usage_percent',
                'CPU usage percentage',
                registry=self.registry
            )
            
            self.metrics['memory_usage'] = Gauge(
                'system_memory_usage_percent',
                'Memory usage percentage',
                registry=self.registry
            )
            
            self.metrics['disk_usage'] = Gauge(
                'system_disk_usage_percent',
                'Disk usage percentage',
                registry=self.registry
            )
            
            # Application metrics
            self.metrics['http_requests_total'] = Counter(
                'http_requests_total',
                'Total HTTP requests',
                ['method', 'endpoint', 'status_code'],
                registry=self.registry
            )
            
            self.metrics['request_duration'] = Histogram(
                'http_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint'],
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self.registry
            )
            
            # ML specific metrics
            self.metrics['ml_analyses_total'] = Counter(
                'ml_analyses_total',
                'Total ML analyses',
                ['status', 'task_type'],
                registry=self.registry
            )
            
            self.metrics['analysis_duration'] = Histogram(
                'ml_analysis_duration_seconds',
                'ML analysis duration',
                ['task_type'],
                buckets=[1, 10, 60, 300, 1800, 3600],
                registry=self.registry
            )
            
            # Health metrics
            self.metrics['health_check_status'] = Gauge(
                'health_check_status',
                'Health check status (1=healthy, 0=unhealthy)',
                ['check_name'],
                registry=self.registry
            )
            
            logger.info("Core metrics initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        try:
            if name in self.metrics and hasattr(self.metrics[name], 'inc'):
                if labels:
                    self.metrics[name].labels(**labels).inc()
                else:
                    self.metrics[name].inc()
        except Exception as e:
            logger.error(f"Failed to increment counter {name}: {e}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        try:
            if name in self.metrics and hasattr(self.metrics[name], 'set'):
                if labels:
                    self.metrics[name].labels(**labels).set(value)
                else:
                    self.metrics[name].set(value)
        except Exception as e:
            logger.error(f"Failed to set gauge {name}: {e}")
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation."""
        try:
            if name in self.metrics and hasattr(self.metrics[name], 'observe'):
                if labels:
                    self.metrics[name].labels(**labels).observe(value)
                else:
                    self.metrics[name].observe(value)
        except Exception as e:
            logger.error(f"Failed to record histogram {name}: {e}")
    
    def update_system_metrics(self, system_metrics: SystemMetrics) -> None:
        """Update system resource metrics."""
        try:
            self.set_gauge('cpu_usage', system_metrics.cpu_percent)
            self.set_gauge('memory_usage', system_metrics.memory_percent)
            self.set_gauge('disk_usage', system_metrics.disk_percent)
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available\n"
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return f"# Error: {e}\n"


class MockMetricsCollector:
    """Mock metrics collector for when Prometheus is not available."""
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Mock counter increment."""
        logger.debug(f"Mock counter increment: {name} {labels}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Mock gauge set."""
        logger.debug(f"Mock gauge set: {name}={value} {labels}")
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Mock histogram record."""
        logger.debug(f"Mock histogram record: {name}={value} {labels}")
    
    def update_system_metrics(self, system_metrics: SystemMetrics) -> None:
        """Mock system metrics update."""
        logger.debug(f"Mock system metrics: CPU={system_metrics.cpu_percent}%")
    
    def get_metrics_text(self) -> str:
        """Mock metrics text."""
        return "# Mock metrics - Prometheus not available\n"


# =============================================================================
# ALERT MANAGEMENT
# =============================================================================

class AlertManager:
    """Secure, rate-limited alert management."""
    
    def __init__(self, config: AlertConfig):
        """Initialize alert manager."""
        self.config = config
        self.alert_history: Dict[str, datetime] = {}  # For rate limiting
        self._lock = asyncio.Lock()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("AlertManager initialized")
    
    def _validate_config(self) -> None:
        """Validate alert configuration."""
        if self.config.to_emails and not self.config.smtp_host:
            logger.warning("Email alerts configured but SMTP host not provided")
        
        if self.config.webhook_urls:
            for url in self.config.webhook_urls:
                if not url.startswith(('http://', 'https://')):
                    logger.warning(f"Invalid webhook URL: {url}")
    
    async def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send alert through configured channels."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        # Check rate limiting
        if self.config.enable_rate_limiting:
            if not await self._check_rate_limit(alert):
                logger.debug(f"Alert rate limited: {title}")
                return False
        
        success = True
        
        # Send email alerts
        if self.config.to_emails and self.config.smtp_host:
            email_success = await self._send_email_alert(alert)
            success = success and email_success
        
        # Send webhook alerts
        if self.config.webhook_urls:
            webhook_success = await self._send_webhook_alerts(alert)
            success = success and webhook_success
        
        if success:
            logger.info(f"Alert sent: {title}")
        else:
            logger.error(f"Failed to send alert: {title}")
        
        return success
    
    async def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        async with self._lock:
            rate_key = f"{alert.level.value}:{alert.title}"
            now = datetime.now(timezone.utc)
            
            # Check if we've sent this alert recently
            if rate_key in self.alert_history:
                time_diff = now - self.alert_history[rate_key]
                window = timedelta(minutes=self.config.rate_limit_window_minutes)
                
                if time_diff < window:
                    return False
            
            # Update rate limit tracker
            self.alert_history[rate_key] = now
            
            # Clean old entries
            cutoff = now - timedelta(minutes=self.config.rate_limit_window_minutes * 2)
            self.alert_history = {
                k: v for k, v in self.alert_history.items() if v > cutoff
            }
            
            return True
    
    async def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not SMTP_AVAILABLE:
            logger.warning("aiosmtplib not available for email alerts")
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.config.from_email or "noreply@auto-analyst.com"
            msg['To'] = ', '.join(self.config.to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Level: {alert.level.value.upper()}
- Service: {alert.service}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Alert ID: {alert.id}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

---
Auto-Analyst Monitoring System
            """.strip()
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            smtp = aiosmtplib.SMTP(
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                use_tls=True
            )
            
            await smtp.connect()
            
            if self.config.smtp_username and self.config.smtp_password:
                await smtp.login(self.config.smtp_username, self.config.smtp_password)
            
            await smtp.send_message(msg)
            await smtp.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False
    
    async def _send_webhook_alerts(self, alert: Alert) -> bool:
        """Send alert via webhooks."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available for webhook alerts")
            return False
        
        success = True
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout_seconds)
        ) as session:
            
            for webhook_url in self.config.webhook_urls:
                try:
                    async with session.post(
                        webhook_url,
                        json=alert.to_dict(),
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            logger.info(f"Webhook alert sent to {webhook_url}")
                        else:
                            logger.error(f"Webhook alert failed: {response.status}")
                            success = False
                            
                except Exception as e:
                    logger.error(f"Webhook alert failed for {webhook_url}: {e}")
                    success = False
        
        return success


# =============================================================================
# HEALTH MONITORING
# =============================================================================

class HealthMonitor:
    """System health monitoring with configurable checks."""
    
    def __init__(self, config: MonitoringConfig, alert_manager: Optional[AlertManager] = None):
        """Initialize health monitor."""
        self.config = config
        self.alert_manager = alert_manager
        self.health_checks: Dict[str, HealthCheckProtocol] = {}
        self.last_check: Optional[datetime] = None
        self.current_status = HealthStatus.UNKNOWN
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("HealthMonitor initialized")
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        
        async def check_system_resources() -> HealthCheck:
            """Check system resource usage."""
            if not PSUTIL_AVAILABLE:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available"
                )
            
            try:
                start_time = time.time()
                
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Determine status
                if (cpu_percent > self.config.cpu_warning_threshold or
                    memory_percent > self.config.memory_warning_threshold or
                    disk_percent > self.config.disk_warning_threshold):
                    
                    status = HealthStatus.WARNING
                    message = f"High resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Resources OK - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%"
                
                return HealthCheck(
                    name="system_resources",
                    status=status,
                    message=message,
                    duration_ms=duration_ms,
                    metadata={
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'disk_percent': disk_percent
                    }
                )
                
            except Exception as e:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.CRITICAL,
                    message=f"Resource check failed: {e}"
                )
        
        async def check_disk_space() -> HealthCheck:
            """Check available disk space."""
            if not PSUTIL_AVAILABLE:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available"
                )
            
            try:
                start_time = time.time()
                disk_usage = psutil.disk_usage('/')
                free_gb = disk_usage.free / (1024**3)
                total_gb = disk_usage.total / (1024**3)
                used_percent = (disk_usage.used / disk_usage.total) * 100
                
                duration_ms = (time.time() - start_time) * 1000
                
                if used_percent > 95:
                    status = HealthStatus.CRITICAL
                    message = f"Disk space critically low: {free_gb:.1f}GB free ({100-used_percent:.1f}% free)"
                elif used_percent > 90:
                    status = HealthStatus.WARNING
                    message = f"Disk space low: {free_gb:.1f}GB free ({100-used_percent:.1f}% free)"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Disk space OK: {free_gb:.1f}GB free of {total_gb:.1f}GB"
                
                return HealthCheck(
                    name="disk_space",
                    status=status,
                    message=message,
                    duration_ms=duration_ms,
                    metadata={
                        'free_bytes': disk_usage.free,
                        'total_bytes': disk_usage.total,
                        'used_percent': used_percent
                    }
                )
                
            except Exception as e:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.CRITICAL,
                    message=f"Disk check failed: {e}"
                )
        
        # Register checks
        self.health_checks['system_resources'] = check_system_resources
        self.health_checks['disk_space'] = check_disk_space
    
    def register_health_check(self, name: str, check_func: HealthCheckProtocol) -> None:
        """Register a custom health check."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        check_results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                check_results[check_name] = result
                
                # Update overall status
                if result.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif result.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.WARNING
                    
            except Exception as e:
                logger.error(f"Health check failed: {check_name} - {e}")
                check_results[check_name] = HealthCheck(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check execution failed: {e}"
                )
                overall_status = HealthStatus.CRITICAL
        
        # Update status
        previous_status = self.current_status
        self.current_status = overall_status
        self.last_check = datetime.now(timezone.utc)
        
        # Send alert if status degraded
        if (self.alert_manager and 
            previous_status == HealthStatus.HEALTHY and 
            overall_status != HealthStatus.HEALTHY):
            
            await self.alert_manager.send_alert(
                level=AlertLevel.ERROR if overall_status == HealthStatus.CRITICAL else AlertLevel.WARNING,
                title=f"Health Status Changed: {overall_status.value.upper()}",
                message=f"System health degraded from {previous_status.value} to {overall_status.value}",
                metadata={'check_results': {name: result.message for name, result in check_results.items()}}
            )
        
        return {
            'status': overall_status.value,
            'timestamp': self.last_check.isoformat(),
            'checks': {name: {
                'status': result.status.value,
                'message': result.message,
                'duration_ms': result.duration_ms,
                'metadata': result.metadata
            } for name, result in check_results.items()},
            'summary': {
                'total': len(check_results),
                'healthy': len([r for r in check_results.values() if r.status == HealthStatus.HEALTHY]),
                'warning': len([r for r in check_results.values() if r.status == HealthStatus.WARNING]),
                'critical': len([r for r in check_results.values() if r.status == HealthStatus.CRITICAL])
            }
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current health status without running checks."""
        return {
            'status': self.current_status.value,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'registered_checks': list(self.health_checks.keys())
        }


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

class PerformanceTracker:
    """Lightweight performance monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollectorProtocol):
        """Initialize performance tracker."""
        self.metrics_collector = metrics_collector
        self.active_operations: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        logger.info("PerformanceTracker initialized")
    
    @asynccontextmanager
    async def track_operation(
        self, 
        operation_name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> AsyncContextManager[str]:
        """Track operation performance."""
        operation_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        async with self._lock:
            self.active_operations[operation_id] = start_time
        
        try:
            yield operation_id
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            async with self._lock:
                self.active_operations.pop(operation_id, None)
            
            # Record metrics
            self.metrics_collector.record_histogram(
                'request_duration', duration, 
                labels or {'operation': operation_name}
            )
            
            logger.debug(f"Operation '{operation_name}' took {duration:.3f}s")
    
    def track_function(self, operation_name: Optional[str] = None):
        """Decorator to track function performance."""
        def decorator(func):
            name = operation_name or func.__name__
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with self.track_operation(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        self.metrics_collector.record_histogram(
                            'request_duration', duration, {'operation': name}
                        )
                        return result
                    except Exception:
                        duration = time.time() - start_time
                        self.metrics_collector.record_histogram(
                            'request_duration', duration, {'operation': name, 'status': 'error'}
                        )
                        raise
                return sync_wrapper
                
        return decorator
    
    async def get_active_operations(self) -> Dict[str, float]:
        """Get currently running operations."""
        current_time = time.time()
        async with self._lock:
            return {
                op_id: current_time - start_time
                for op_id, start_time in self.active_operations.items()
            }


# =============================================================================
# SYSTEM RESOURCE MONITORING
# =============================================================================

class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollectorProtocol):
        """Initialize system monitor."""
        self.config = config
        self.metrics_collector = metrics_collector
        
        logger.info("SystemMonitor initialized")
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - returning empty metrics")
            return SystemMetrics()
        
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Get load average (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except (AttributeError, OSError):
                pass  # Not available on Windows
            
            # Calculate uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                load_average=load_avg,
                uptime_seconds=uptime_seconds
            )
            
            # Update Prometheus metrics
            self.metrics_collector.update_system_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()


# =============================================================================
# MAIN MONITORING COORDINATOR
# =============================================================================

class MonitoringManager:
    """Main monitoring coordinator."""
    
    def __init__(
        self,
        monitoring_config: Optional[MonitoringConfig] = None,
        alert_config: Optional[AlertConfig] = None
    ):
        """Initialize monitoring manager."""
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.alert_config = alert_config or AlertConfig()
        
        # Initialize components
        self.metrics_collector = (
            PrometheusMetricsCollector(self.monitoring_config)
            if PROMETHEUS_AVAILABLE
            else MockMetricsCollector()
        )
        
        self.alert_manager = AlertManager(self.alert_config)
        self.health_monitor = HealthMonitor(self.monitoring_config, self.alert_manager)
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.monitoring_config, self.metrics_collector)
        
        # Background tasks
        self._monitoring_active = False
        self._background_tasks: List[asyncio.Task] = []
        
        logger.info("MonitoringManager initialized successfully")
    
    async def start(self) -> None:
        """Start background monitoring tasks."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        
        try:
            # Start metrics collection
            if self.monitoring_config.enable_metrics:
                metrics_task = asyncio.create_task(self._metrics_collection_loop())
                self._background_tasks.append(metrics_task)
            
            # Start health monitoring
            if self.monitoring_config.enable_health_checks:
                health_task = asyncio.create_task(self._health_monitoring_loop())
                self._background_tasks.append(health_task)
            
            logger.info("Monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            await self.stop()
    
    async def stop(self) -> None:
        """Stop background monitoring tasks."""
        self._monitoring_active = False
        
        # Cancel all tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("Monitoring stopped")
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self._monitoring_active:
            try:
                await self.system_monitor.collect_system_metrics()
                await asyncio.sleep(self.monitoring_config.collection_interval_seconds)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)  # Shorter retry interval
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self._monitoring_active:
            try:
                await self.health_monitor.run_health_checks()
                await asyncio.sleep(self.monitoring_config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Shorter retry interval
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        return {
            'monitoring_active': self._monitoring_active,
            'config': {
                'metrics_enabled': self.monitoring_config.enable_metrics,
                'health_checks_enabled': self.monitoring_config.enable_health_checks,
                'performance_tracking_enabled': self.monitoring_config.enable_performance_tracking
            },
            'components': {
                'metrics_collector': type(self.metrics_collector).__name__,
                'alert_manager': 'AlertManager',
                'health_monitor': 'HealthMonitor',
                'performance_tracker': 'PerformanceTracker'
            },
            'dependencies': {
                'prometheus': PROMETHEUS_AVAILABLE,
                'psutil': PSUTIL_AVAILABLE,
                'aiohttp': AIOHTTP_AVAILABLE,
                'aiosmtplib': SMTP_AVAILABLE
            },
            'background_tasks': len(self._background_tasks)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_monitoring_manager(
    monitoring_config: Optional[MonitoringConfig] = None,
    alert_config: Optional[AlertConfig] = None
) -> MonitoringManager:
    """Create a monitoring manager with configuration."""
    return MonitoringManager(monitoring_config, alert_config)


def get_basic_health_status() -> Dict[str, Any]:
    """Quick health check without full monitoring setup."""
    try:
        if PSUTIL_AVAILABLE:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            status = "healthy"
            if cpu > 90 or memory > 90 or disk > 95:
                status = "critical"
            elif cpu > 80 or memory > 80 or disk > 90:
                status = "warning"
            
            return {
                'status': status,
                'cpu_percent': cpu,
                'memory_percent': memory,
                'disk_percent': disk,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                'status': 'unknown',
                'message': 'psutil not available',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    'MonitoringManager', 'AlertManager', 'HealthMonitor', 
    'PerformanceTracker', 'SystemMonitor',
    
    # Configuration
    'MonitoringConfig', 'AlertConfig',
    
    # Data models
    'Alert', 'HealthCheck', 'SystemMetrics',
    
    # Enums
    'AlertLevel', 'HealthStatus', 'MetricType',
    
    # Protocols
    'HealthCheckProtocol', 'MetricsCollectorProtocol',
    
    # Functions
    'create_monitoring_manager', 'get_basic_health_status'
]

# Module initialization
logger.info(f"Monitoring module loaded - Prometheus: {PROMETHEUS_AVAILABLE}, psutil: {PSUTIL_AVAILABLE}")
