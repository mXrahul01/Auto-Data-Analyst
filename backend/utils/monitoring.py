"""
🚀 AUTO-ANALYST PLATFORM - SYSTEM MONITORING UTILITIES
=====================================================

Production-ready system monitoring utilities for comprehensive application
observability, performance tracking, and health monitoring.

Key Features:
- Real-time system metrics collection
- Application performance monitoring (APM)
- Resource usage tracking
- Health check endpoints
- Alert management system
- Custom metrics and dashboards
- Error tracking and logging
- Database connection monitoring

Components:
- SystemMonitor: Core system metrics collection
- ApplicationMonitor: Application-specific monitoring
- ResourceMonitor: CPU, memory, disk, network monitoring
- HealthChecker: Service health checks
- MetricsCollector: Custom metrics aggregation
- AlertManager: Alert threshold management
- PerformanceProfiler: Code performance profiling

Dependencies:
- psutil: System information
- prometheus_client: Metrics collection
- asyncio: Async monitoring tasks
- logging: Structured logging
"""

import asyncio
import logging
import time
import threading
import socket
import platform
import gc
import sys
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Set,
    Callable, NamedTuple, Protocol
)
import warnings
import json
import os

# Core monitoring dependencies
import psutil
from pydantic import BaseModel, Field, ConfigDict

# Optional dependencies with graceful fallbacks
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class MonitoringLevel(str, Enum):
    """Monitoring detail levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ResourceType(str, Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"


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
    monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED
    collection_interval: int = Field(default=30, ge=1, le=300)
    retention_days: int = Field(default=30, ge=1, le=365)

    # Alerting
    enable_alerts: bool = True
    alert_cooldown_minutes: int = Field(default=15, ge=1)

    # Resource thresholds
    cpu_threshold_warning: float = Field(default=70.0, ge=0.0, le=100.0)
    cpu_threshold_critical: float = Field(default=90.0, ge=0.0, le=100.0)
    memory_threshold_warning: float = Field(default=80.0, ge=0.0, le=100.0)
    memory_threshold_critical: float = Field(default=95.0, ge=0.0, le=100.0)
    disk_threshold_warning: float = Field(default=85.0, ge=0.0, le=100.0)
    disk_threshold_critical: float = Field(default=95.0, ge=0.0, le=100.0)

    # Performance monitoring
    enable_profiling: bool = False
    slow_query_threshold_ms: int = Field(default=1000, ge=1)
    request_timeout_seconds: int = Field(default=30, ge=1)

    # Storage
    metrics_storage_path: str = "metrics"
    max_log_file_size_mb: int = Field(default=100, ge=1)

    # External services
    prometheus_port: Optional[int] = Field(default=8000, ge=1024, le=65535)
    webhook_url: Optional[str] = None
    redis_url: Optional[str] = None


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_freq_current: float = 0.0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Memory metrics
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_percent: float = 0.0

    # Disk metrics
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0

    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    network_connections: int = 0


@dataclass
class ProcessMetrics:
    """Process-specific metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Process info
    pid: int = 0
    name: str = ""
    status: str = ""
    create_time: float = 0.0

    # Resource usage
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0

    # I/O
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    num_threads: int = 0
    num_fds: int = 0

    # Network connections
    num_connections: int = 0


@dataclass
class ApplicationMetrics:
    """Application-level metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Request metrics
    total_requests: int = 0
    requests_per_second: float = 0.0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0

    # Database metrics
    db_connections_active: int = 0
    db_connections_total: int = 0
    avg_query_time_ms: float = 0.0
    slow_queries: int = 0

    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0

    # Custom metrics
    active_users: int = 0
    background_tasks: int = 0
    queue_size: int = 0


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert information."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    resource_type: ResourceType
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None


# =============================================================================
# BASE MONITORING CLASSES
# =============================================================================

class BaseMonitor(ABC):
    """Abstract base class for all monitors."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics specific to this monitor."""
        pass

    async def start_monitoring(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(f"{self.__class__.__name__} started")

    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(f"{self.__class__.__name__} stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = await self.collect_metrics()
                await self._process_metrics(metrics)
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.collection_interval)

    async def _process_metrics(self, metrics: Dict[str, Any]) -> None:
        """Process collected metrics (override in subclasses)."""
        pass


# =============================================================================
# SYSTEM RESOURCE MONITOR
# =============================================================================

class SystemMonitor(BaseMonitor):
    """Monitor system resources (CPU, memory, disk, network)."""

    def __init__(self, config: MonitoringConfig):
        super().__init__(config)
        self.metrics_history: List[SystemMetrics] = []
        self.last_disk_io = psutil.disk_io_counters()
        self.last_network_io = psutil.net_io_counters()
        self.last_collection_time = time.time()

    async def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        current_time = time.time()
        time_delta = current_time - self.last_collection_time

        metrics = SystemMetrics()

        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=1)
            metrics.cpu_count = psutil.cpu_count()

            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.cpu_freq_current = cpu_freq.current

            # Load average (Unix-like systems)
            try:
                metrics.load_average = psutil.getloadavg()
            except AttributeError:
                metrics.load_average = (0.0, 0.0, 0.0)

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_total_gb = memory.total / (1024**3)
            metrics.memory_available_gb = memory.available / (1024**3)
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_percent = memory.percent

            # Swap memory
            swap = psutil.swap_memory()
            metrics.swap_total_gb = swap.total / (1024**3)
            metrics.swap_used_gb = swap.used / (1024**3)
            metrics.swap_percent = swap.percent

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_total_gb = disk.total / (1024**3)
            metrics.disk_used_gb = disk.used / (1024**3)
            metrics.disk_free_gb = disk.free / (1024**3)
            metrics.disk_percent = (disk.used / disk.total) * 100

            # Disk I/O
            current_disk_io = psutil.disk_io_counters()
            if current_disk_io and self.last_disk_io:
                metrics.disk_read_bytes = current_disk_io.read_bytes - self.last_disk_io.read_bytes
                metrics.disk_write_bytes = current_disk_io.write_bytes - self.last_disk_io.write_bytes
            self.last_disk_io = current_disk_io

            # Network metrics
            current_network_io = psutil.net_io_counters()
            if current_network_io and self.last_network_io:
                metrics.network_bytes_sent = current_network_io.bytes_sent - self.last_network_io.bytes_sent
                metrics.network_bytes_recv = current_network_io.bytes_recv - self.last_network_io.bytes_recv
                metrics.network_packets_sent = current_network_io.packets_sent - self.last_network_io.packets_sent
                metrics.network_packets_recv = current_network_io.packets_recv - self.last_network_io.packets_recv

            self.last_network_io = current_network_io

            # Network connections
            metrics.network_connections = len(psutil.net_connections())

            self.last_collection_time = current_time

        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")

        return metrics

    async def _process_metrics(self, metrics: SystemMetrics) -> None:
        """Process system metrics and check thresholds."""
        # Store metrics
        self.metrics_history.append(metrics)

        # Keep only recent history
        max_history = int(timedelta(days=1).total_seconds() / self.config.collection_interval)
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]

        # Check thresholds and generate alerts
        await self._check_resource_thresholds(metrics)

    async def _check_resource_thresholds(self, metrics: SystemMetrics) -> None:
        """Check resource usage against configured thresholds."""
        checks = [
            (ResourceType.CPU, metrics.cpu_percent, self.config.cpu_threshold_warning, self.config.cpu_threshold_critical),
            (ResourceType.MEMORY, metrics.memory_percent, self.config.memory_threshold_warning, self.config.memory_threshold_critical),
            (ResourceType.DISK, metrics.disk_percent, self.config.disk_threshold_warning, self.config.disk_threshold_critical),
        ]

        for resource_type, current_value, warning_threshold, critical_threshold in checks:
            if current_value >= critical_threshold:
                await self._create_alert(
                    resource_type, AlertSeverity.CRITICAL,
                    current_value, critical_threshold
                )
            elif current_value >= warning_threshold:
                await self._create_alert(
                    resource_type, AlertSeverity.HIGH,
                    current_value, warning_threshold
                )

    async def _create_alert(
            self,
            resource_type: ResourceType,
            severity: AlertSeverity,
            current_value: float,
            threshold_value: float
    ) -> None:
        """Create and process resource threshold alert."""
        alert_id = f"{resource_type.value}_{int(time.time())}"

        alert = Alert(
            id=alert_id,
            severity=severity,
            title=f"{resource_type.value.upper()} Usage Alert",
            message=f"{resource_type.value.upper()} usage at {current_value:.1f}% (threshold: {threshold_value:.1f}%)",
            resource_type=resource_type,
            current_value=current_value,
            threshold_value=threshold_value
        )

        self.logger.warning(f"Alert created: {alert.title} - {alert.message}")

        # Here you would integrate with your alerting system
        # await self._send_alert(alert)

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get summarized metrics for the specified duration."""
        if not self.metrics_history:
            return {}

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {}

        return {
            "period_minutes": duration_minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "avg": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                "max": max(m.cpu_percent for m in recent_metrics),
                "min": min(m.cpu_percent for m in recent_metrics),
            },
            "memory": {
                "avg": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                "max": max(m.memory_percent for m in recent_metrics),
                "min": min(m.memory_percent for m in recent_metrics),
            },
            "disk": {
                "avg": sum(m.disk_percent for m in recent_metrics) / len(recent_metrics),
                "max": max(m.disk_percent for m in recent_metrics),
                "min": min(m.disk_percent for m in recent_metrics),
            }
        }


# =============================================================================
# APPLICATION PERFORMANCE MONITOR
# =============================================================================

class ApplicationMonitor(BaseMonitor):
    """Monitor application-specific metrics and performance."""

    def __init__(self, config: MonitoringConfig):
        super().__init__(config)
        self.request_metrics: List[float] = []
        self.error_count = 0
        self.total_requests = 0
        self.slow_queries = 0
        self.start_time = time.time()

        # Thread-safe counters
        self._lock = threading.Lock()

    async def collect_metrics(self) -> ApplicationMetrics:
        """Collect application performance metrics."""
        metrics = ApplicationMetrics()

        try:
            with self._lock:
                metrics.total_requests = self.total_requests

                # Calculate requests per second
                uptime_seconds = time.time() - self.start_time
                metrics.requests_per_second = self.total_requests / max(uptime_seconds, 1)

                # Calculate average response time
                if self.request_metrics:
                    metrics.avg_response_time_ms = sum(self.request_metrics) / len(self.request_metrics)

                # Calculate error rate
                metrics.error_rate = (self.error_count / max(self.total_requests, 1)) * 100

                metrics.slow_queries = self.slow_queries

            # Collect process-specific metrics
            current_process = psutil.Process()
            metrics.db_connections_active = len([
                conn for conn in current_process.connections()
                if conn.status == psutil.CONN_ESTABLISHED
            ])

            # Custom application metrics (override in subclass)
            custom_metrics = await self._collect_custom_metrics()
            metrics.active_users = custom_metrics.get('active_users', 0)
            metrics.background_tasks = custom_metrics.get('background_tasks', 0)
            metrics.queue_size = custom_metrics.get('queue_size', 0)

        except Exception as e:
            self.logger.error(f"Application metrics collection failed: {e}")

        return metrics

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect custom application metrics (override in subclass)."""
        return {}

    def record_request(self, response_time_ms: float, is_error: bool = False) -> None:
        """Record a request with its response time and error status."""
        with self._lock:
            self.total_requests += 1
            self.request_metrics.append(response_time_ms)

            if is_error:
                self.error_count += 1

            if response_time_ms > self.config.slow_query_threshold_ms:
                self.slow_queries += 1

            # Keep only recent request metrics
            if len(self.request_metrics) > 1000:
                self.request_metrics = self.request_metrics[-500:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        with self._lock:
            uptime_seconds = time.time() - self.start_time

            return {
                "uptime_seconds": uptime_seconds,
                "total_requests": self.total_requests,
                "requests_per_second": self.total_requests / max(uptime_seconds, 1),
                "error_count": self.error_count,
                "error_rate_percent": (self.error_count / max(self.total_requests, 1)) * 100,
                "slow_queries": self.slow_queries,
                "avg_response_time_ms": (
                    sum(self.request_metrics) / len(self.request_metrics)
                    if self.request_metrics else 0
                )
            }


# =============================================================================
# HEALTH CHECK SYSTEM
# =============================================================================

class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}

    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")

    async def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )

        start_time = time.time()

        try:
            check_func = self.checks[name]

            # Run check with timeout
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(
                    check_func(),
                    timeout=self.config.request_timeout_seconds
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, check_func
                )

            response_time_ms = (time.time() - start_time) * 1000

            # Parse result
            if isinstance(result, dict):
                health_check = HealthCheck(
                    name=name,
                    status=HealthStatus(result.get('status', HealthStatus.HEALTHY)),
                    message=result.get('message', 'OK'),
                    response_time_ms=response_time_ms,
                    details=result.get('details', {})
                )
            elif isinstance(result, bool):
                health_check = HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message='OK' if result else 'Check failed',
                    response_time_ms=response_time_ms
                )
            else:
                health_check = HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    response_time_ms=response_time_ms
                )

        except asyncio.TimeoutError:
            health_check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.config.request_timeout_seconds}s",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            health_check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )

        self.last_results[name] = health_check
        return health_check

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        if not self.checks:
            return {}

        # Run checks concurrently
        tasks = [self.run_check(name) for name in self.checks.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_results = {}
        for name, result in zip(self.checks.keys(), results):
            if isinstance(result, Exception):
                health_results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check exception: {str(result)}"
                )
            else:
                health_results[name] = result

        self.last_results.update(health_results)
        return health_results

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in self.last_results.values()]

        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif any(status == HealthStatus.UNKNOWN for status in statuses):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        overall_status = self.get_overall_health()

        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }

        for check in self.last_results.values():
            status_counts[check.status] += 1

        return {
            "overall_status": overall_status,
            "total_checks": len(self.last_results),
            "status_breakdown": {k.value: v for k, v in status_counts.items()},
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                    "timestamp": check.timestamp.isoformat()
                }
                for name, check in self.last_results.items()
            }
        }


# =============================================================================
# PERFORMANCE PROFILER
# =============================================================================

class PerformanceProfiler:
    """Code performance profiling and analysis."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.profiles: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling code blocks."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            self.record_execution_time(operation_name, execution_time)

    def record_execution_time(self, operation_name: str, time_ms: float) -> None:
        """Record execution time for an operation."""
        with self._lock:
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []

            self.profiles[operation_name].append(time_ms)

            # Keep only recent measurements
            if len(self.profiles[operation_name]) > 1000:
                self.profiles[operation_name] = self.profiles[operation_name][-500:]

    def get_profile_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get performance statistics for an operation."""
        with self._lock:
            if operation_name not in self.profiles:
                return {}

            times = self.profiles[operation_name]
            if not times:
                return {}

            return {
                "operation": operation_name,
                "sample_count": len(times),
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "median_time_ms": sorted(times)[len(times) // 2],
                "p95_time_ms": sorted(times)[int(len(times) * 0.95)],
                "p99_time_ms": sorted(times)[int(len(times) * 0.99)],
            }

    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all operations."""
        return {
            operation: self.get_profile_stats(operation)
            for operation in self.profiles.keys()
        }

    def profile_function(self, func: Callable) -> Callable:
        """Decorator for profiling function execution."""
        def wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def profile_async_function(self, func: Callable) -> Callable:
        """Decorator for profiling async function execution."""
        async def wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return await func(*args, **kwargs)
        return wrapper


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """Collect and aggregate custom metrics."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metrics: Dict[str, List[Tuple[float, datetime]]] = {}
        self._lock = threading.Lock()

        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        try:
            # Start Prometheus metrics server
            if self.config.prometheus_port:
                start_http_server(self.config.prometheus_port)
                self.logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
        except Exception as e:
            self.logger.warning(f"Failed to start Prometheus server: {e}")

    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self._record_metric(name, value, MetricType.COUNTER)

    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self._record_metric(name, value, MetricType.GAUGE)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self._record_metric(name, value, MetricType.HISTOGRAM)

    def _record_metric(self, name: str, value: float, metric_type: MetricType) -> None:
        """Record a metric value."""
        timestamp = datetime.now(timezone.utc)

        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []

            self.metrics[name].append((value, timestamp))

            # Keep only recent metrics
            cutoff_time = timestamp - timedelta(days=self.config.retention_days)
            self.metrics[name] = [
                (val, ts) for val, ts in self.metrics[name]
                if ts >= cutoff_time
            ]

    def get_metric_stats(self, name: str, duration_hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a metric over specified duration."""
        with self._lock:
            if name not in self.metrics:
                return {}

            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)
            recent_values = [
                value for value, timestamp in self.metrics[name]
                if timestamp >= cutoff_time
            ]

            if not recent_values:
                return {}

            return {
                "metric_name": name,
                "period_hours": duration_hours,
                "sample_count": len(recent_values),
                "sum": sum(recent_values),
                "avg": sum(recent_values) / len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "latest": recent_values[-1] if recent_values else None
            }

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            name: self.get_metric_stats(name)
            for name in self.metrics.keys()
        }


# =============================================================================
# MAIN MONITORING SYSTEM
# =============================================================================

class MonitoringSystem:
    """
    Main monitoring system orchestrator.

    Coordinates all monitoring components and provides unified interface
    for system observability.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize monitoring system."""
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.system_monitor = SystemMonitor(self.config)
        self.app_monitor = ApplicationMonitor(self.config)
        self.health_checker = HealthChecker(self.config)
        self.performance_profiler = PerformanceProfiler(self.config)
        self.metrics_collector = MetricsCollector(self.config)

        self._running = False
        self._monitors: List[BaseMonitor] = [
            self.system_monitor,
            self.app_monitor
        ]

        # Register default health checks
        self._register_default_health_checks()

    def _register_default_health_checks(self) -> None:
        """Register default system health checks."""
        self.health_checker.register_check("system_resources", self._check_system_resources)
        self.health_checker.register_check("disk_space", self._check_disk_space)
        self.health_checker.register_check("memory_usage", self._check_memory_usage)
        self.health_checker.register_check("python_process", self._check_python_process)

    async def start(self) -> None:
        """Start all monitoring components."""
        if self._running:
            return

        self.logger.info("Starting monitoring system...")

        # Start all monitors
        for monitor in self._monitors:
            await monitor.start_monitoring()

        self._running = True
        self.logger.info("Monitoring system started successfully")

    async def stop(self) -> None:
        """Stop all monitoring components."""
        if not self._running:
            return

        self.logger.info("Stopping monitoring system...")

        # Stop all monitors
        for monitor in self._monitors:
            await monitor.stop_monitoring()

        self._running = False
        self.logger.info("Monitoring system stopped")

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and metrics."""
        # Run health checks
        health_results = await self.health_checker.run_all_checks()

        # Get current metrics
        system_metrics = self.system_monitor.get_current_metrics()
        app_summary = self.app_monitor.get_performance_summary()

        # Get system summary
        system_summary = self.system_monitor.get_metrics_summary()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monitoring_config": {
                "level": self.config.monitoring_level,
                "collection_interval": self.config.collection_interval,
                "alerts_enabled": self.config.enable_alerts
            },
            "overall_health": self.health_checker.get_overall_health(),
            "health_checks": health_results,
            "system_metrics": {
                "current": system_metrics.__dict__ if system_metrics else None,
                "summary": system_summary
            },
            "application_performance": app_summary,
            "custom_metrics": self.metrics_collector.get_all_metrics_summary(),
            "performance_profiles": self.performance_profiler.get_all_profiles()
        }

    # Default health check implementations
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check overall system resource health."""
        metrics = self.system_monitor.get_current_metrics()
        if not metrics:
            return {"status": HealthStatus.UNKNOWN, "message": "No metrics available"}

        issues = []
        if metrics.cpu_percent > self.config.cpu_threshold_critical:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.config.memory_threshold_critical:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        if metrics.disk_percent > self.config.disk_threshold_critical:
            issues.append(f"High disk usage: {metrics.disk_percent:.1f}%")

        if issues:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": "; ".join(issues),
                "details": metrics.__dict__
            }
        else:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "System resources normal",
                "details": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "disk_percent": metrics.disk_percent
                }
            }

    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            used_percent = (disk.used / disk.total) * 100
            free_gb = disk.free / (1024**3)

            if used_percent > self.config.disk_threshold_critical:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Critical disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free"
                }
            elif used_percent > self.config.disk_threshold_warning:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"Low disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free"
                }
            else:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"Disk space OK: {used_percent:.1f}% used, {free_gb:.1f}GB free"
                }
        except Exception as e:
            return {"status": HealthStatus.UNKNOWN, "message": f"Disk check failed: {e}"}

    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()

            if memory.percent > self.config.memory_threshold_critical:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Critical memory usage: {memory.percent:.1f}%"
                }
            elif memory.percent > self.config.memory_threshold_warning:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"High memory usage: {memory.percent:.1f}%"
                }
            else:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"Memory usage OK: {memory.percent:.1f}%"
                }
        except Exception as e:
            return {"status": HealthStatus.UNKNOWN, "message": f"Memory check failed: {e}"}

    async def _check_python_process(self) -> Dict[str, Any]:
        """Check Python process health."""
        try:
            process = psutil.Process()

            details = {
                "pid": process.pid,
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_mb": process.memory_info().rss / (1024*1024),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }

            # Check for issues
            if process.cpu_percent() > 90:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"High process CPU usage: {process.cpu_percent():.1f}%",
                    "details": details
                }
            elif process.memory_percent() > 90:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"High process memory usage: {process.memory_percent():.1f}%",
                    "details": details
                }
            else:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Python process healthy",
                    "details": details
                }
        except Exception as e:
            return {"status": HealthStatus.UNKNOWN, "message": f"Process check failed: {e}"}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global monitoring system instance
_monitoring_system: Optional[MonitoringSystem] = None

def get_monitoring_system(config: Optional[MonitoringConfig] = None) -> MonitoringSystem:
    """Get or create global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem(config)
    return _monitoring_system


async def start_monitoring(config: Optional[MonitoringConfig] = None) -> MonitoringSystem:
    """Start monitoring system with optional configuration."""
    system = get_monitoring_system(config)
    await system.start()
    return system


async def stop_monitoring() -> None:
    """Stop the global monitoring system."""
    global _monitoring_system
    if _monitoring_system:
        await _monitoring_system.stop()


def record_request(response_time_ms: float, is_error: bool = False) -> None:
    """Record a request metric."""
    system = get_monitoring_system()
    system.app_monitor.record_request(response_time_ms, is_error)


def record_metric(name: str, value: float, metric_type: str = "gauge") -> None:
    """Record a custom metric."""
    system = get_monitoring_system()
    if metric_type == "counter":
        system.metrics_collector.record_counter(name, value)
    else:
        system.metrics_collector.record_gauge(name, value)


def profile_operation(operation_name: str):
    """Context manager for profiling operations."""
    system = get_monitoring_system()
    return system.performance_profiler.profile(operation_name)


def profile_function(func: Callable) -> Callable:
    """Decorator for profiling function execution."""
    system = get_monitoring_system()
    return system.performance_profiler.profile_function(func)


def profile_async_function(func: Callable) -> Callable:
    """Decorator for profiling async function execution."""
    system = get_monitoring_system()
    return system.performance_profiler.profile_async_function(func)


async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    system = get_monitoring_system()
    return await system.get_comprehensive_status()


def register_health_check(name: str, check_func: Callable) -> None:
    """Register a custom health check."""
    system = get_monitoring_system()
    system.health_checker.register_check(name, check_func)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    'MonitoringSystem', 'SystemMonitor', 'ApplicationMonitor',
    'HealthChecker', 'PerformanceProfiler', 'MetricsCollector',

    # Configuration and models
    'MonitoringConfig', 'SystemMetrics', 'ProcessMetrics', 'ApplicationMetrics',
    'HealthCheck', 'Alert',

    # Enums
    'MonitoringLevel', 'AlertSeverity', 'HealthStatus', 'MetricType', 'ResourceType',

    # Convenience functions
    'get_monitoring_system', 'start_monitoring', 'stop_monitoring',
    'record_request', 'record_metric', 'profile_operation',
    'profile_function', 'profile_async_function', 'get_system_status',
    'register_health_check',

    # Base classes
    'BaseMonitor'
]

# Initialize logging
logger.info("System monitoring utilities loaded successfully")
