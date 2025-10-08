"""
Backend Logger - Core Logging Orchestrator

Provides the main logging interface for the agentic AI microservice backend.
Handles structured logging, context management, and integration with various
backend components.
"""

import asyncio
import json
import logging
import sys
import traceback
import re
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import threading
import queue
import time
import yaml
import os
from collections import defaultdict

# Optional cryptography imports for log encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from .models import (
    LogEntry, LogLevel, LogCategory, LogContext,
    PerformanceMetrics, ErrorDetails, AgentMetrics,
    APIMetrics, DatabaseMetrics, LogConfiguration,
    LoggingMode, ModuleConfig, ConversationConfig, TierConfig
)
from .context import CorrelationContext, SystemContext
from .formatters import JSONFormatter, StructuredFormatter, ConversationFormatter, ColoredStructuredFormatter
from .handlers import AsyncFileHandler


# ============================================================================
# CONFIGURATION LOADERS
# ============================================================================

def load_config_from_yaml(yaml_path: str = "config/logging.yaml") -> LogConfiguration:
    """
    Load logging configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        LogConfiguration object
    """
    yaml_file = Path(yaml_path)

    if not yaml_file.exists():
        logging.warning(f"YAML config file not found: {yaml_path}, using defaults")
        return LogConfiguration()

    try:
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)

        if not yaml_data:
            logging.warning(f"Empty YAML config file: {yaml_path}, using defaults")
            return LogConfiguration()

        # Extract configuration sections
        global_config = yaml_data.get('global', {})
        conversation_config_data = yaml_data.get('conversation_layer', {})
        module_control_data = yaml_data.get('module_control', {})
        file_logging_data = yaml_data.get('file_logging', {})
        external_logging_data = yaml_data.get('external_logging', {})
        runtime_config_data = yaml_data.get('runtime', {})

        # Build module configs
        module_configs = {}
        for module_name, module_data in module_control_data.items():
            if isinstance(module_data, dict):
                module_configs[module_name] = ModuleConfig(
                    module_name=module_name,
                    enabled=module_data.get('enabled', False),
                    console_level=LogLevel(module_data.get('console_level', 'WARNING').upper()),
                    file_level=LogLevel(module_data.get('file_level', 'DEBUG').upper()),
                    console_output=module_data.get('console_output', False),
                    file_output=module_data.get('file_output', True)
                )

        # Build conversation config
        conversation_config = ConversationConfig(
            enabled=conversation_config_data.get('enabled', True),
            style=conversation_config_data.get('style', 'conversational'),
            show_reasoning=conversation_config_data.get('show_reasoning', True),
            show_tool_usage=conversation_config_data.get('show_tool_usage', True),
            show_tool_results=conversation_config_data.get('show_tool_results', True),
            show_responses=conversation_config_data.get('show_responses', True),
            emoji_enhanced=conversation_config_data.get('emoji_enhanced', True),
            max_reasoning_length=conversation_config_data.get('max_reasoning_length', 200),
            max_result_length=conversation_config_data.get('max_result_length', 500)
        )

        # Build main configuration
        config = LogConfiguration(
            # Global settings
            logging_mode=LoggingMode(global_config.get('mode', 'user').lower()),
            show_ids=global_config.get('show_ids', False),
            show_timestamps=global_config.get('show_timestamps', False),
            timestamp_format=global_config.get('timestamp_format', 'simple'),

            # Color settings
            enable_colors=global_config.get('enable_colors', True),
            color_scheme=global_config.get('color_scheme', 'default'),
            force_colors=global_config.get('force_colors', False),

            # Module configs
            module_configs=module_configs,

            # Conversation config
            conversation_config=conversation_config,

            # File logging
            file_directory=file_logging_data.get('directory', 'data/logs/backend'),
            file_format=file_logging_data.get('format', 'json'),
            separate_by_category=file_logging_data.get('separate_by_category', True),
            rotation_strategy=file_logging_data.get('rotation', {}).get('strategy', 'daily'),
            backup_count=file_logging_data.get('retention', {}).get('backup_count', 30),
            compress_old_logs=file_logging_data.get('compression', {}).get('enabled', True),

            # External logging
            suppress_noisy_loggers=external_logging_data.get('suppress_noisy_loggers', True),
            external_default_level=LogLevel(external_logging_data.get('default_level', 'ERROR').upper()),

            # Runtime
            hot_reload_enabled=runtime_config_data.get('hot_reload', {}).get('enabled', True),
            api_enabled=runtime_config_data.get('api_control', {}).get('enabled', True),
            allow_mode_switching=runtime_config_data.get('allow_mode_switching', True),
            allow_module_control=runtime_config_data.get('allow_module_control', True)
        )

        return config

    except Exception as e:
        logging.error(f"Failed to load YAML config: {e}")
        return LogConfiguration()


# ============================================================================
# PII REDACTION ENGINE
# ============================================================================

class PIIRedactor:
    """
    Production-ready PII redaction engine for log sanitization.

    Redacts sensitive information including:
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - Social Security Numbers
    - IP addresses
    - Custom patterns
    """

    # Regex patterns for common PII
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b')
    SSN_PATTERN = re.compile(r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b')
    IP_ADDRESS_PATTERN = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

    def __init__(self, config: LogConfiguration):
        self.config = config
        self.custom_patterns = [re.compile(pattern) for pattern in config.custom_redaction_patterns]

    def redact(self, text: str) -> str:
        """Redact PII from text."""
        if not self.config.enable_pii_redaction:
            return text

        if not isinstance(text, str):
            return text

        # Redact email addresses
        if self.config.redact_email_addresses:
            text = self.EMAIL_PATTERN.sub('[EMAIL_REDACTED]', text)

        # Redact phone numbers
        if self.config.redact_phone_numbers:
            text = self.PHONE_PATTERN.sub('[PHONE_REDACTED]', text)

        # Redact credit cards
        if self.config.redact_credit_cards:
            text = self.CREDIT_CARD_PATTERN.sub('[CREDIT_CARD_REDACTED]', text)

        # Redact SSNs
        if self.config.redact_ssn:
            text = self.SSN_PATTERN.sub('[SSN_REDACTED]', text)

        # Redact IP addresses
        if self.config.redact_ip_addresses:
            text = self.IP_ADDRESS_PATTERN.sub('[IP_REDACTED]', text)

        # Redact custom patterns
        for pattern in self.custom_patterns:
            text = pattern.sub('[REDACTED]', text)

        return text

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact PII from dictionary."""
        if not self.config.enable_pii_redaction:
            return data

        if not isinstance(data, dict):
            return data

        redacted = {}
        for key, value in data.items():
            if isinstance(value, str):
                redacted[key] = self.redact(value)
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [self.redact(item) if isinstance(item, str) else item for item in value]
            else:
                redacted[key] = value

        return redacted


# ============================================================================
# LOG SAMPLING ENGINE
# ============================================================================

class LogSampler:
    """
    Production-ready log sampling engine for high-volume scenarios.

    Features:
    - Adaptive sampling based on QPS
    - Always sample errors and critical logs
    - Category-based sampling rules
    - Deterministic sampling for correlation
    """

    def __init__(self, config: LogConfiguration):
        self.config = config
        self.request_count = 0
        self.last_reset = time.time()
        self.current_qps = 0.0
        self._lock = threading.Lock()

    def should_sample(self, log_entry: LogEntry) -> bool:
        """Determine if log entry should be sampled."""
        if not self.config.enable_sampling:
            return True

        # Always sample errors
        if self.config.always_sample_errors and log_entry.level in [LogLevel.ERROR, LogLevel.FATAL]:
            return True

        # Always sample specific categories
        if log_entry.category in self.config.always_sample_categories:
            return True

        # Update QPS calculation
        with self._lock:
            self.request_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_reset

            if elapsed >= 1.0:
                self.current_qps = self.request_count / elapsed
                self.request_count = 0
                self.last_reset = current_time

        # Only sample if QPS exceeds threshold
        if self.current_qps < self.config.sampling_threshold_qps:
            return True

        # Deterministic sampling based on correlation ID for consistency
        if log_entry.context and log_entry.context.correlation_id:
            # Use hash of correlation ID for deterministic sampling
            hash_value = int(hashlib.md5(log_entry.context.correlation_id.encode()).hexdigest(), 16)
            return (hash_value % 10000) < (self.config.sampling_rate * 10000)

        # Random sampling as fallback
        return random.random() < self.config.sampling_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            "enabled": self.config.enable_sampling,
            "current_qps": self.current_qps,
            "sampling_rate": self.config.sampling_rate,
            "threshold_qps": self.config.sampling_threshold_qps
        }


# ============================================================================
# ERROR AGGREGATION ENGINE
# ============================================================================

class ErrorAggregator:
    """
    Production-ready error aggregation engine for centralized error tracking.

    Supports:
    - Sentry integration
    - Rollbar integration
    - Custom error tracking services
    - Breadcrumb tracking
    - Error grouping
    """

    def __init__(self, config: LogConfiguration):
        self.config = config
        self.sentry_sdk = None
        self.breadcrumbs = []
        self._lock = threading.Lock()

        if config.enable_error_aggregation:
            self._initialize_service()

    def _initialize_service(self):
        """Initialize error aggregation service."""
        try:
            if self.config.error_aggregation_service == "sentry":
                self._initialize_sentry()
            elif self.config.error_aggregation_service == "rollbar":
                self._initialize_rollbar()
        except Exception as e:
            logging.error(f"Failed to initialize error aggregation service: {e}")

    def _initialize_sentry(self):
        """Initialize Sentry SDK."""
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            if not self.config.error_aggregation_dsn:
                logging.warning("Sentry DSN not configured, skipping initialization")
                return

            sentry_sdk.init(
                dsn=self.config.error_aggregation_dsn,
                environment=self.config.error_aggregation_environment,
                traces_sample_rate=self.config.error_aggregation_traces_sample_rate,
                sample_rate=self.config.error_aggregation_sample_rate,
                max_breadcrumbs=self.config.max_breadcrumbs,
                integrations=[
                    LoggingIntegration(
                        level=logging.INFO,
                        event_level=logging.ERROR
                    )
                ]
            )

            self.sentry_sdk = sentry_sdk
            logging.info("Sentry error aggregation initialized successfully")

        except ImportError:
            logging.warning("sentry-sdk not installed, error aggregation disabled")
        except Exception as e:
            logging.error(f"Failed to initialize Sentry: {e}")

    def _initialize_rollbar(self):
        """Initialize Rollbar SDK."""
        try:
            import rollbar

            if not self.config.error_aggregation_dsn:
                logging.warning("Rollbar access token not configured, skipping initialization")
                return

            rollbar.init(
                access_token=self.config.error_aggregation_dsn,
                environment=self.config.error_aggregation_environment
            )

            logging.info("Rollbar error aggregation initialized successfully")

        except ImportError:
            logging.warning("rollbar not installed, error aggregation disabled")
        except Exception as e:
            logging.error(f"Failed to initialize Rollbar: {e}")

    def capture_error(self, log_entry: LogEntry):
        """Capture error in aggregation service."""
        if not self.config.enable_error_aggregation:
            return

        if log_entry.level not in [LogLevel.ERROR, LogLevel.FATAL]:
            return

        try:
            if self.config.error_aggregation_service == "sentry" and self.sentry_sdk:
                self._capture_sentry_error(log_entry)
            elif self.config.error_aggregation_service == "rollbar":
                self._capture_rollbar_error(log_entry)
        except Exception as e:
            logging.error(f"Failed to capture error in aggregation service: {e}")

    def _capture_sentry_error(self, log_entry: LogEntry):
        """Capture error in Sentry."""
        if not self.sentry_sdk:
            return

        with self.sentry_sdk.push_scope() as scope:
            # Add context
            if log_entry.context:
                scope.set_context("log_context", log_entry.context.dict(exclude_none=True))

            # Add tags
            scope.set_tag("category", log_entry.category.value)
            scope.set_tag("component", log_entry.component)
            scope.set_tag("level", log_entry.level.value)

            # Add extra data
            if log_entry.data:
                scope.set_context("log_data", log_entry.data)

            if log_entry.performance:
                scope.set_context("performance", log_entry.performance.dict(exclude_none=True))

            # Add breadcrumbs
            if self.config.capture_breadcrumbs:
                for breadcrumb in self.breadcrumbs:
                    self.sentry_sdk.add_breadcrumb(breadcrumb)

            # Capture exception or message
            if log_entry.error_details and log_entry.error_details.stack_trace:
                self.sentry_sdk.capture_message(
                    log_entry.message,
                    level=log_entry.level.value.lower()
                )
            else:
                self.sentry_sdk.capture_message(
                    log_entry.message,
                    level=log_entry.level.value.lower()
                )

    def _capture_rollbar_error(self, log_entry: LogEntry):
        """Capture error in Rollbar."""
        try:
            import rollbar

            extra_data = {
                "category": log_entry.category.value,
                "component": log_entry.component,
                "context": log_entry.context.dict(exclude_none=True) if log_entry.context else {}
            }

            if log_entry.data:
                extra_data["data"] = log_entry.data

            if log_entry.performance:
                extra_data["performance"] = log_entry.performance.dict(exclude_none=True)

            rollbar.report_message(
                log_entry.message,
                level=log_entry.level.value.lower(),
                extra_data=extra_data
            )
        except Exception as e:
            logging.error(f"Failed to capture error in Rollbar: {e}")

    def add_breadcrumb(self, message: str, category: str = "default", level: str = "info", data: Dict[str, Any] = None):
        """Add breadcrumb for error context."""
        if not self.config.capture_breadcrumbs:
            return

        with self._lock:
            breadcrumb = {
                "message": message,
                "category": category,
                "level": level,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data or {}
            }

            self.breadcrumbs.append(breadcrumb)

            # Limit breadcrumbs
            if len(self.breadcrumbs) > self.config.max_breadcrumbs:
                self.breadcrumbs.pop(0)


# ============================================================================
# OPENTELEMETRY INTEGRATION
# ============================================================================

class OpenTelemetryIntegration:
    """
    Production-ready OpenTelemetry integration for distributed tracing.

    Features:
    - Trace context propagation
    - Span creation and management
    - Metrics collection
    - Log correlation
    """

    def __init__(self, config: LogConfiguration):
        self.config = config
        self.tracer = None
        self.meter = None
        self.logger_provider = None

        if config.enable_opentelemetry:
            self._initialize_opentelemetry()

    def _initialize_opentelemetry(self):
        """Initialize OpenTelemetry SDK."""
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

            # Create resource
            resource_attributes = {
                "service.name": self.config.otel_service_name,
                **self.config.otel_resource_attributes
            }
            resource = Resource.create(resource_attributes)

            # Initialize tracing
            if self.config.otel_traces_enabled and self.config.otel_exporter_endpoint:
                tracer_provider = TracerProvider(resource=resource)

                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.config.otel_exporter_endpoint,
                    insecure=True  # Use TLS in production
                )

                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)

                trace.set_tracer_provider(tracer_provider)
                self.tracer = trace.get_tracer(__name__)

                logging.info("OpenTelemetry tracing initialized successfully")

            # Initialize metrics
            if self.config.otel_metrics_enabled and self.config.otel_exporter_endpoint:
                metric_exporter = OTLPMetricExporter(
                    endpoint=self.config.otel_exporter_endpoint,
                    insecure=True  # Use TLS in production
                )

                metric_reader = PeriodicExportingMetricReader(metric_exporter)
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

                metrics.set_meter_provider(meter_provider)
                self.meter = metrics.get_meter(__name__)

                logging.info("OpenTelemetry metrics initialized successfully")

        except ImportError:
            logging.warning("OpenTelemetry SDK not installed, distributed tracing disabled")
        except Exception as e:
            logging.error(f"Failed to initialize OpenTelemetry: {e}")

    def create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a new span for tracing."""
        if not self.tracer:
            return None

        try:
            span = self.tracer.start_span(name)

            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            return span
        except Exception as e:
            logging.error(f"Failed to create span: {e}")
            return None

    def add_log_to_span(self, log_entry: LogEntry):
        """Add log entry as span event."""
        if not self.tracer:
            return

        try:
            from opentelemetry import trace

            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.add_event(
                    log_entry.message,
                    attributes={
                        "log.level": log_entry.level.value,
                        "log.category": log_entry.category.value,
                        "log.component": log_entry.component
                    }
                )
        except Exception as e:
            logging.error(f"Failed to add log to span: {e}")


# ============================================================================
# LOG METRICS COLLECTOR
# ============================================================================

class LogMetricsCollector:
    """
    Production-ready metrics collector for log analytics.

    Tracks:
    - Total log count by level/category
    - Error rates
    - Performance percentiles
    - Log volume trends
    """

    def __init__(self, config: LogConfiguration):
        self.config = config
        self.metrics = {
            "total_logs": 0,
            "logs_by_level": defaultdict(int),
            "logs_by_category": defaultdict(int),
            "logs_by_component": defaultdict(int),
            "error_count": 0,
            "warning_count": 0,
            "performance_samples": []
        }
        self._lock = threading.Lock()
        self.last_aggregation = time.time()

    def record_log(self, log_entry: LogEntry):
        """Record log entry metrics."""
        if not self.config.enable_log_metrics:
            return

        with self._lock:
            self.metrics["total_logs"] += 1
            self.metrics["logs_by_level"][log_entry.level.value] += 1
            self.metrics["logs_by_category"][log_entry.category.value] += 1
            self.metrics["logs_by_component"][log_entry.component] += 1

            if log_entry.level in [LogLevel.ERROR, LogLevel.FATAL]:
                self.metrics["error_count"] += 1
            elif log_entry.level == LogLevel.WARN:
                self.metrics["warning_count"] += 1

            # Track performance metrics
            if self.config.track_performance_percentiles and log_entry.performance:
                if log_entry.performance.duration_ms:
                    self.metrics["performance_samples"].append(log_entry.performance.duration_ms)

                    # Limit samples
                    if len(self.metrics["performance_samples"]) > 10000:
                        self.metrics["performance_samples"] = self.metrics["performance_samples"][-5000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            metrics = {
                "total_logs": self.metrics["total_logs"],
                "logs_by_level": dict(self.metrics["logs_by_level"]),
                "logs_by_category": dict(self.metrics["logs_by_category"]),
                "logs_by_component": dict(self.metrics["logs_by_component"]),
                "error_count": self.metrics["error_count"],
                "warning_count": self.metrics["warning_count"]
            }

            # Calculate error rate
            if self.metrics["total_logs"] > 0:
                metrics["error_rate"] = self.metrics["error_count"] / self.metrics["total_logs"]
            else:
                metrics["error_rate"] = 0.0

            # Calculate performance percentiles
            if self.config.track_performance_percentiles and self.metrics["performance_samples"]:
                sorted_samples = sorted(self.metrics["performance_samples"])
                count = len(sorted_samples)

                metrics["performance_percentiles"] = {
                    "p50": sorted_samples[int(count * 0.5)],
                    "p90": sorted_samples[int(count * 0.9)],
                    "p95": sorted_samples[int(count * 0.95)],
                    "p99": sorted_samples[int(count * 0.99)]
                }

            return metrics

    def reset_metrics(self):
        """Reset metrics counters."""
        with self._lock:
            self.metrics = {
                "total_logs": 0,
                "logs_by_level": defaultdict(int),
                "logs_by_category": defaultdict(int),
                "logs_by_component": defaultdict(int),
                "error_count": 0,
                "warning_count": 0,
                "performance_samples": []
            }
            self.last_aggregation = time.time()


def load_config_from_env() -> LogConfiguration:
    """
    Load logging configuration from environment variables.

    Returns:
        LogConfiguration object
    """
    try:
        # Import settings
        from app.config.settings import get_settings
        settings = get_settings()

        # Build module configs from environment
        module_configs = {}
        module_names = [
            'app.agents', 'app.rag', 'app.memory', 'app.llm', 'app.tools',
            'app.api', 'app.core', 'app.services', 'app.orchestration',
            'app.communication', 'app.config', 'app.models', 'app.optimization',
            'app.integrations'
        ]

        for module_name in module_names:
            # Get module-specific settings
            module_key = module_name.replace('.', '_').upper()
            enabled = getattr(settings, f'LOG_MODULE_{module_key.replace("APP_", "")}', False)
            level_str = getattr(settings, f'LOG_LEVEL_{module_key}', 'DEBUG')

            module_configs[module_name] = ModuleConfig(
                module_name=module_name,
                enabled=enabled,
                console_level=LogLevel(level_str.upper()),
                file_level=LogLevel.DEBUG,
                console_output=enabled,
                file_output=True
            )

        # Build conversation config
        conversation_config = ConversationConfig(
            enabled=settings.LOG_CONVERSATION_ENABLED,
            style=settings.LOG_CONVERSATION_STYLE,
            show_reasoning=settings.LOG_CONVERSATION_SHOW_REASONING,
            show_tool_usage=settings.LOG_CONVERSATION_SHOW_TOOL_USAGE,
            show_tool_results=settings.LOG_CONVERSATION_SHOW_TOOL_RESULTS,
            emoji_enhanced=settings.LOG_CONVERSATION_EMOJI_ENHANCED,
            max_reasoning_length=settings.LOG_CONVERSATION_MAX_REASONING_LENGTH,
            max_result_length=settings.LOG_CONVERSATION_MAX_RESULT_LENGTH
        )

        # Build main configuration
        config = LogConfiguration(
            # Global settings
            logging_mode=LoggingMode(settings.LOG_MODE.lower()),
            show_ids=settings.LOG_SHOW_IDS,
            show_timestamps=settings.LOG_SHOW_TIMESTAMPS,
            timestamp_format=settings.LOG_TIMESTAMP_FORMAT,

            # Color settings
            enable_colors=getattr(settings, 'LOG_ENABLE_COLORS', True),
            color_scheme=getattr(settings, 'LOG_COLOR_SCHEME', 'default'),
            force_colors=getattr(settings, 'LOG_FORCE_COLORS', False),

            # Module configs
            module_configs=module_configs,

            # Conversation config
            conversation_config=conversation_config,

            # File logging
            file_directory=settings.LOG_FILE_DIRECTORY,
            file_format=settings.LOG_FILE_FORMAT,
            separate_by_category=settings.LOG_SEPARATE_BY_CATEGORY,
            rotation_strategy=settings.LOG_ROTATION_STRATEGY,
            compress_old_logs=settings.LOG_COMPRESSION_ENABLED,

            # External logging
            suppress_noisy_loggers=settings.LOG_SUPPRESS_NOISY_LOGGERS,
            external_default_level=LogLevel(settings.LOG_EXTERNAL_DEFAULT_LEVEL.upper()),

            # Runtime
            hot_reload_enabled=settings.LOG_HOT_RELOAD_ENABLED,
            api_enabled=settings.LOG_API_ENABLED,
            allow_mode_switching=settings.LOG_ALLOW_MODE_SWITCHING,
            allow_module_control=settings.LOG_ALLOW_MODULE_CONTROL,

            # Advanced features (with safe defaults)
            enable_sampling=getattr(settings, 'LOG_ENABLE_SAMPLING', False),
            sampling_rate=getattr(settings, 'LOG_SAMPLING_RATE', 0.01),
            sampling_threshold_qps=getattr(settings, 'LOG_SAMPLING_THRESHOLD_QPS', 1000),

            enable_pii_redaction=getattr(settings, 'LOG_ENABLE_PII_REDACTION', False),
            redact_email_addresses=getattr(settings, 'LOG_REDACT_EMAIL', True),
            redact_phone_numbers=getattr(settings, 'LOG_REDACT_PHONE', True),
            redact_credit_cards=getattr(settings, 'LOG_REDACT_CREDIT_CARDS', True),
            redact_ssn=getattr(settings, 'LOG_REDACT_SSN', True),
            redact_ip_addresses=getattr(settings, 'LOG_REDACT_IP', False),

            enable_error_aggregation=getattr(settings, 'LOG_ENABLE_ERROR_AGGREGATION', False),
            error_aggregation_service=getattr(settings, 'LOG_ERROR_AGGREGATION_SERVICE', 'sentry'),
            error_aggregation_dsn=getattr(settings, 'LOG_ERROR_AGGREGATION_DSN', None),
            error_aggregation_environment=getattr(settings, 'LOG_ERROR_AGGREGATION_ENV', 'development'),

            enable_opentelemetry=getattr(settings, 'LOG_ENABLE_OPENTELEMETRY', False),
            otel_service_name=getattr(settings, 'LOG_OTEL_SERVICE_NAME', 'agentic-ai-backend'),
            otel_exporter_endpoint=getattr(settings, 'LOG_OTEL_EXPORTER_ENDPOINT', None),
            otel_traces_enabled=getattr(settings, 'LOG_OTEL_TRACES_ENABLED', True),
            otel_metrics_enabled=getattr(settings, 'LOG_OTEL_METRICS_ENABLED', True),

            enable_log_metrics=getattr(settings, 'LOG_ENABLE_METRICS', True),
            track_error_rates=getattr(settings, 'LOG_TRACK_ERROR_RATES', True),
            track_performance_percentiles=getattr(settings, 'LOG_TRACK_PERFORMANCE_PERCENTILES', True)
        )

        return config

    except Exception as e:
        logging.error(f"Failed to load config from environment: {e}")
        return LogConfiguration()


class ModuleController:
    """
    Controls logging for individual modules using Python's hierarchical logger system.

    This class manages per-module logging configuration, allowing granular control
    over which modules log to console/file and at what levels.
    """

    def __init__(self, config: LogConfiguration):
        self.config = config
        self.module_loggers: Dict[str, logging.Logger] = {}
        self.module_configs: Dict[str, ModuleConfig] = {}
        self._lock = threading.Lock()

        # Load module hierarchy from JSON
        self._load_module_hierarchy()

        # Initialize module configurations
        self._initialize_module_configs()

    def _load_module_hierarchy(self):
        """Load module hierarchy from configuration file"""
        hierarchy_file = Path("config/logging_module_hierarchy.json")
        if hierarchy_file.exists():
            try:
                with open(hierarchy_file, 'r') as f:
                    self.module_hierarchy = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load module hierarchy: {e}")
                self.module_hierarchy = {}
        else:
            self.module_hierarchy = {}

    def _initialize_module_configs(self):
        """Initialize module configurations from config"""
        # Use configurations from LogConfiguration if available
        if self.config.module_configs:
            self.module_configs = self.config.module_configs.copy()

        # Apply default configurations for all modules in hierarchy
        if self.module_hierarchy:
            self._apply_hierarchy_defaults(self.module_hierarchy.get('module_hierarchy', {}))

    def _apply_hierarchy_defaults(self, hierarchy: Dict[str, Any], parent_path: str = ""):
        """Recursively apply default configurations from hierarchy"""
        for module_name, module_info in hierarchy.items():
            if isinstance(module_info, dict):
                # Get module configuration
                default_level_str = module_info.get('default_level', 'WARNING')
                default_level = getattr(LogLevel, default_level_str, LogLevel.WARNING)
                log_category_str = module_info.get('log_category')
                log_category = getattr(LogCategory, log_category_str, None) if log_category_str else None

                # Create module config if not exists
                if module_name not in self.module_configs:
                    self.module_configs[module_name] = ModuleConfig(
                        module_name=module_name,
                        enabled=False,  # Disabled by default
                        console_level=default_level,
                        file_level=LogLevel.DEBUG,
                        console_output=False,
                        file_output=True,
                        description=module_info.get('description'),
                        log_category=log_category
                    )

                # Process sub-modules
                if 'sub_modules' in module_info:
                    self._apply_hierarchy_defaults(module_info['sub_modules'], module_name)

    def enable_module(self, module_name: str, console_level: LogLevel = LogLevel.DEBUG):
        """Enable logging for a specific module"""
        with self._lock:
            if module_name in self.module_configs:
                self.module_configs[module_name].enabled = True
                self.module_configs[module_name].console_level = console_level
                self.module_configs[module_name].console_output = True
            else:
                # Create new config
                self.module_configs[module_name] = ModuleConfig(
                    module_name=module_name,
                    enabled=True,
                    console_level=console_level,
                    file_level=LogLevel.DEBUG,
                    console_output=True,
                    file_output=True
                )

            # Apply to Python logger
            self._apply_module_config(module_name)

    def disable_module(self, module_name: str):
        """Disable logging for a specific module"""
        with self._lock:
            if module_name in self.module_configs:
                self.module_configs[module_name].enabled = False
                self.module_configs[module_name].console_output = False

            # Apply to Python logger
            self._apply_module_config(module_name)

    def set_module_level(self, module_name: str, console_level: LogLevel, file_level: LogLevel = None):
        """Set log levels for a specific module"""
        with self._lock:
            if module_name not in self.module_configs:
                self.module_configs[module_name] = ModuleConfig(
                    module_name=module_name,
                    enabled=True,
                    console_level=console_level,
                    file_level=file_level or LogLevel.DEBUG,
                    console_output=True,
                    file_output=True
                )
            else:
                self.module_configs[module_name].console_level = console_level
                if file_level:
                    self.module_configs[module_name].file_level = file_level

            # Apply to Python logger
            self._apply_module_config(module_name)

    def _apply_module_config(self, module_name: str):
        """Apply configuration to Python logger"""
        if module_name not in self.module_configs:
            return

        config = self.module_configs[module_name]
        logger = logging.getLogger(module_name)

        if config.enabled and config.console_output:
            # Set level to the more permissive of console and file levels
            min_level = min(
                self._log_level_to_int(config.console_level),
                self._log_level_to_int(config.file_level)
            )
            logger.setLevel(min_level)
        else:
            # Set to CRITICAL to effectively disable
            logger.setLevel(logging.CRITICAL)

    def _log_level_to_int(self, level: LogLevel) -> int:
        """Convert LogLevel to Python logging level integer"""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.FATAL: logging.CRITICAL
        }
        return level_map.get(level, logging.INFO)

    def get_module_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all modules"""
        with self._lock:
            return {
                name: {
                    'enabled': config.enabled,
                    'console_level': config.console_level.value,
                    'file_level': config.file_level.value,
                    'console_output': config.console_output,
                    'file_output': config.file_output,
                    'description': config.description
                }
                for name, config in self.module_configs.items()
            }

    def get_active_loggers(self) -> List[str]:
        """Get list of currently active (enabled) loggers"""
        with self._lock:
            return [
                name for name, config in self.module_configs.items()
                if config.enabled and config.console_output
            ]

    def is_module_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled for logging"""
        with self._lock:
            if module_name in self.module_configs:
                return self.module_configs[module_name].enabled
            return False

    def should_log_to_console(self, module_name: str, level: LogLevel) -> bool:
        """Check if a module should log to console at given level"""
        with self._lock:
            if module_name not in self.module_configs:
                return False

            config = self.module_configs[module_name]
            if not config.enabled or not config.console_output:
                return False

            return self._log_level_to_int(level) >= self._log_level_to_int(config.console_level)


class BackendLogger:
    """
    Main backend logger class that provides comprehensive logging capabilities
    for the agentic AI microservice.

    Features:
    - PII redaction
    - Log sampling for high-volume scenarios
    - Error aggregation (Sentry/Rollbar)
    - OpenTelemetry integration
    - Log metrics and analytics
    - Module-based control
    - Hot-reload configuration
    """

    def __init__(self, config: LogConfiguration = None):
        self.config = config or LogConfiguration()
        self.log_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.is_running = True
        self.worker_thread = None
        self.file_handlers = {}
        self.console_handler = None
        self.conversation_handler = None

        # Initialize module controller for granular logging control
        self.module_controller = ModuleController(self.config)

        # Initialize advanced features
        self.pii_redactor = PIIRedactor(self.config)
        self.log_sampler = LogSampler(self.config)
        self.error_aggregator = ErrorAggregator(self.config)
        self.otel_integration = OpenTelemetryIntegration(self.config)
        self.metrics_collector = LogMetricsCollector(self.config)

        # Initialize logging infrastructure
        self._setup_logging()
        self._start_worker_thread()

        # Suppress noisy external loggers
        if self.config.suppress_noisy_loggers:
            self._suppress_external_loggers()

        # Log system initialization
        self.info(
            "Backend logging system initialized",
            category=LogCategory.SYSTEM_HEALTH,
            component="BackendLogger",
            data={
                "mode": self.config.logging_mode.value,
                "modules_enabled": len(self.module_controller.get_active_loggers()),
                "pii_redaction": self.config.enable_pii_redaction,
                "sampling": self.config.enable_sampling,
                "error_aggregation": self.config.enable_error_aggregation,
                "opentelemetry": self.config.enable_opentelemetry
            }
        )
    
    def _setup_logging(self):
        """Setup logging handlers and formatters"""
        # Create logs directory
        logs_dir = Path(self.config.file_directory)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup file handlers for different categories
        if self.config.enable_file_output:
            for category in LogCategory:
                log_file = logs_dir / f"{category.value}_{datetime.now().strftime('%Y%m%d')}.log"
                handler = AsyncFileHandler(
                    filename=str(log_file),
                    max_bytes=self.config.max_log_file_size_mb * 1024 * 1024,
                    backup_count=self.config.max_log_files
                )

                if self.config.file_format == "json":
                    handler.setFormatter(JSONFormatter())
                else:
                    handler.setFormatter(StructuredFormatter())

                self.file_handlers[category] = handler

        # Setup console handler based on logging mode
        if self.config.enable_console_output:
            self.console_handler = logging.StreamHandler(sys.stdout)

            # Choose formatter based on mode and color settings
            if self.config.enable_colors:
                # Use colored formatter
                if self.config.logging_mode == LoggingMode.USER:
                    # User mode: minimal colored output
                    self.console_handler.setFormatter(
                        ColoredStructuredFormatter(
                            include_context=False,
                            include_metrics=False,
                            enable_colors=True,
                            force_colors=self.config.force_colors,
                            color_scheme=self.config.color_scheme
                        )
                    )
                elif self.config.logging_mode == LoggingMode.DEVELOPER:
                    # Developer mode: colored with context
                    self.console_handler.setFormatter(
                        ColoredStructuredFormatter(
                            include_context=True,
                            include_metrics=False,
                            enable_colors=True,
                            force_colors=self.config.force_colors,
                            color_scheme=self.config.color_scheme
                        )
                    )
                else:  # DEBUG mode
                    # Debug mode: full colored output
                    self.console_handler.setFormatter(
                        ColoredStructuredFormatter(
                            include_context=True,
                            include_metrics=True,
                            enable_colors=True,
                            force_colors=self.config.force_colors,
                            color_scheme=self.config.color_scheme
                        )
                    )
            else:
                # Use plain formatter (no colors)
                if self.config.logging_mode == LoggingMode.USER:
                    # User mode: minimal structured output
                    self.console_handler.setFormatter(StructuredFormatter(include_context=False, include_metrics=False))
                elif self.config.logging_mode == LoggingMode.DEVELOPER:
                    # Developer mode: structured with context
                    self.console_handler.setFormatter(StructuredFormatter(include_context=True, include_metrics=False))
                else:  # DEBUG mode
                    # Debug mode: full structured output
                    self.console_handler.setFormatter(StructuredFormatter(include_context=True, include_metrics=True))

        # Setup conversation handler for user-facing output
        if self.config.conversation_config.enabled:
            self.conversation_handler = logging.StreamHandler(sys.stdout)
            conversation_formatter_config = {
                'emoji_enhanced': self.config.conversation_config.emoji_enhanced,
                'show_reasoning': self.config.conversation_config.show_reasoning,
                'show_tool_usage': self.config.conversation_config.show_tool_usage,
                'show_tool_results': self.config.conversation_config.show_tool_results,
                'max_reasoning_length': self.config.conversation_config.max_reasoning_length,
                'max_result_length': self.config.conversation_config.max_result_length,
                'style': self.config.conversation_config.style,
                'enable_colors': self.config.enable_colors,
                'force_colors': self.config.force_colors
            }
            self.conversation_handler.setFormatter(ConversationFormatter(conversation_formatter_config))

    def _suppress_external_loggers(self):
        """Suppress noisy external library loggers"""
        noisy_loggers = [
            'chromadb', 'sentence_transformers', 'transformers', 'urllib3',
            'requests', 'httpx', 'httpcore', 'openai', 'anthropic', 'ollama',
            'playwright', 'selenium', 'asyncio', 'aiohttp'
        ]

        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(self.module_controller._log_level_to_int(self.config.external_default_level))
    
    def _start_worker_thread(self):
        """Start the background worker thread for async logging"""
        if self.config.enable_async_logging:
            self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.worker_thread.start()

        # Start hot-reload thread if enabled
        if self.config.hot_reload_enabled:
            self.reload_thread = threading.Thread(target=self._hot_reload_worker, daemon=True)
            self.reload_thread.start()

    def _hot_reload_worker(self):
        """Background worker that monitors configuration changes"""
        yaml_path = Path("config/logging.yaml")
        last_modified = yaml_path.stat().st_mtime if yaml_path.exists() else 0

        while self.is_running:
            try:
                time.sleep(self.config.reload_interval_seconds)

                # Check if YAML file has been modified
                if yaml_path.exists():
                    current_modified = yaml_path.stat().st_mtime

                    if current_modified > last_modified:
                        # Reload configuration
                        self.info(
                            "Configuration file changed, reloading...",
                            category=LogCategory.CONFIGURATION_MANAGEMENT,
                            component="BackendLogger"
                        )

                        try:
                            new_config = load_config_from_yaml(str(yaml_path))

                            # Update configuration
                            self.config = new_config
                            self.module_controller = ModuleController(new_config)

                            # Reinitialize advanced features with new config
                            self.pii_redactor = PIIRedactor(new_config)
                            self.log_sampler = LogSampler(new_config)
                            self.error_aggregator = ErrorAggregator(new_config)
                            self.otel_integration = OpenTelemetryIntegration(new_config)
                            self.metrics_collector = LogMetricsCollector(new_config)

                            # Re-setup logging with new configuration
                            self._setup_logging()

                            self.info(
                                "Configuration reloaded successfully",
                                category=LogCategory.CONFIGURATION_MANAGEMENT,
                                component="BackendLogger",
                                data={
                                    "mode": self.config.logging_mode.value,
                                    "modules_enabled": len(self.module_controller.get_active_loggers()),
                                    "pii_redaction": self.config.enable_pii_redaction,
                                    "sampling": self.config.enable_sampling,
                                    "error_aggregation": self.config.enable_error_aggregation,
                                    "opentelemetry": self.config.enable_opentelemetry
                                }
                            )

                            last_modified = current_modified

                        except Exception as e:
                            self.error(
                                "Failed to reload configuration",
                                category=LogCategory.CONFIGURATION_MANAGEMENT,
                                component="BackendLogger",
                                error=e
                            )

            except Exception as e:
                # Don't let hot-reload errors crash the worker
                print(f"Hot-reload worker error: {e}", file=sys.stderr)

    def _log_worker(self):
        """Background worker that processes log entries"""
        while self.is_running:
            try:
                # Process logs in batches
                logs_to_process = []

                # Collect logs with timeout
                try:
                    log_entry = self.log_queue.get(timeout=self.config.flush_interval_seconds)
                    logs_to_process.append(log_entry)

                    # Collect additional logs without blocking
                    while len(logs_to_process) < self.config.buffer_size:
                        try:
                            log_entry = self.log_queue.get_nowait()
                            logs_to_process.append(log_entry)
                        except queue.Empty:
                            break

                except queue.Empty:
                    continue

                # Process the batch
                for log_entry in logs_to_process:
                    # Apply sampling
                    if not self.log_sampler.should_sample(log_entry):
                        continue

                    # Apply PII redaction
                    if self.config.enable_pii_redaction:
                        log_entry.message = self.pii_redactor.redact(log_entry.message)
                        if log_entry.data:
                            log_entry.data = self.pii_redactor.redact_dict(log_entry.data)

                    # Collect metrics
                    self.metrics_collector.record_log(log_entry)

                    # Capture errors in aggregation service
                    self.error_aggregator.capture_error(log_entry)

                    # Add to OpenTelemetry span
                    self.otel_integration.add_log_to_span(log_entry)

                    # Write log entry
                    self._write_log_entry(log_entry)
                    
            except Exception as e:
                # Don't let logging errors crash the worker
                print(f"Logging worker error: {e}", file=sys.stderr)
    
    def _write_log_entry(self, log_entry: LogEntry):
        """Write a log entry to appropriate handlers"""
        try:
            # Write to file handler for the category
            if log_entry.category in self.file_handlers:
                handler = self.file_handlers[log_entry.category]
                record = self._create_log_record(log_entry)
                handler.emit(record)
            
            # Write to console if enabled and appropriate level
            if (self.console_handler and 
                self._should_log_to_console(log_entry.level)):
                record = self._create_log_record(log_entry)
                self.console_handler.emit(record)
                
        except Exception as e:
            print(f"Error writing log entry: {e}", file=sys.stderr)
    
    def _create_log_record(self, log_entry: LogEntry) -> logging.LogRecord:
        """Create a Python logging record from our log entry"""
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.FATAL: logging.CRITICAL
        }
        
        record = logging.LogRecord(
            name=log_entry.component,
            level=level_mapping.get(log_entry.level, logging.INFO),
            pathname="",
            lineno=0,
            msg=log_entry.message,
            args=(),
            exc_info=None
        )
        
        # Add our custom fields
        record.log_entry = log_entry
        return record
    
    def _should_log_to_console(self, level: LogLevel) -> bool:
        """Determine if a log level should be written to console"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARN: 2,
            LogLevel.ERROR: 3,
            LogLevel.FATAL: 4
        }
        
        return level_order.get(level, 1) >= level_order.get(self.config.log_level, 1)
    
    def _should_exclude_message(self, message: str) -> bool:
        """Check if message should be excluded based on patterns"""
        return any(pattern in message.lower() for pattern in self.config.exclude_patterns)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory,
        component: str,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[LogContext] = None,
        performance: Optional[PerformanceMetrics] = None,
        agent_metrics: Optional[AgentMetrics] = None,
        api_metrics: Optional[APIMetrics] = None,
        database_metrics: Optional[DatabaseMetrics] = None,
        error_details: Optional[ErrorDetails] = None,
        **kwargs
    ):
        """
        Main logging method that creates and queues log entries
        """
        # Skip if message should be excluded
        if self._should_exclude_message(message):
            return
        
        # Get or create context
        if context is None:
            context = CorrelationContext.get_context()
        
        # Add system information
        system_info = SystemContext.get_system_info()
        
        # Create log entry
        log_entry = LogEntry(
            level=level,
            category=category,
            message=message,
            component=component,
            context=context,
            data=data,
            performance=performance,
            agent_metrics=agent_metrics,
            api_metrics=api_metrics,
            database_metrics=database_metrics,
            error_details=error_details,
            hostname=system_info.get("hostname"),
            process_id=system_info.get("process_id"),
            thread_id=str(system_info.get("thread_id")),
            environment=kwargs.get("environment", "development"),
            version=kwargs.get("version", "1.0.0")
        )
        
        # Add to queue for async processing
        if self.config.enable_async_logging:
            try:
                self.log_queue.put_nowait(log_entry)
            except queue.Full:
                # If queue is full, write directly (blocking)
                self._write_log_entry(log_entry)
        else:
            # Synchronous logging
            self._write_log_entry(log_entry)
    
    # Convenience methods for different log levels
    def debug(self, message: str, category: LogCategory, component: str, **kwargs):
        """Log a debug message"""
        self.log(LogLevel.DEBUG, message, category, component, **kwargs)
    
    def info(self, message: str, category: LogCategory, component: str, **kwargs):
        """Log an info message"""
        self.log(LogLevel.INFO, message, category, component, **kwargs)
    
    def warn(self, message: str, category: LogCategory, component: str, **kwargs):
        """Log a warning message"""
        self.log(LogLevel.WARN, message, category, component, **kwargs)
    
    def error(self, message: str, category: LogCategory, component: str, 
              error: Exception = None, **kwargs):
        """Log an error message with optional exception details"""
        error_details = None
        if error:
            error_details = ErrorDetails(
                error_type=type(error).__name__,
                error_code=getattr(error, 'code', None),
                stack_trace=traceback.format_exc() if self.config.include_stack_trace else None,
                severity="high" if isinstance(error, (SystemError, MemoryError)) else "medium"
            )
        
        self.log(LogLevel.ERROR, message, category, component, 
                error_details=error_details, **kwargs)
    
    def fatal(self, message: str, category: LogCategory, component: str, 
             error: Exception = None, **kwargs):
        """Log a fatal error message"""
        error_details = None
        if error:
            error_details = ErrorDetails(
                error_type=type(error).__name__,
                error_code=getattr(error, 'code', None),
                stack_trace=traceback.format_exc() if self.config.include_stack_trace else None,
                severity="critical",
                user_impact="service_unavailable"
            )
        
        self.log(LogLevel.FATAL, message, category, component, 
                error_details=error_details, **kwargs)
    
    def get_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        levels: Optional[List[LogLevel]] = None,
        categories: Optional[List[LogCategory]] = None,
        components: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Retrieve logs based on filters (placeholder for file-based implementation)
        In a production system, this would read from log files or a database
        """
        # This is a simplified implementation
        # In practice, you'd read from log files or a log database
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "queue_size": self.log_queue.qsize() if hasattr(self.log_queue, 'qsize') else 0,
            "worker_running": self.is_running,
            "handlers_count": len(self.file_handlers),
            "logging_mode": self.config.logging_mode.value,
            "active_modules": len(self.module_controller.get_active_loggers()),
            "conversation_enabled": self.config.conversation_config.enabled,
            "config": self.config.dict()
        }

    # ========================================================================
    # REVOLUTIONARY LOGGING SYSTEM - MODE AND MODULE CONTROL
    # ========================================================================

    def set_mode(self, mode: Union[LoggingMode, str]):
        """
        Switch logging mode at runtime.

        Args:
            mode: LoggingMode enum or string ('user', 'developer', 'debug')
        """
        if not self.config.allow_mode_switching:
            self.warn(
                "Mode switching is disabled in configuration",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "BackendLogger"
            )
            return

        # Convert string to enum if needed
        if isinstance(mode, str):
            mode = LoggingMode(mode.lower())

        old_mode = self.config.logging_mode
        self.config.logging_mode = mode

        # Update console handler formatter based on new mode
        if self.console_handler:
            if mode == LoggingMode.USER:
                self.console_handler.setFormatter(StructuredFormatter(include_context=False, include_metrics=False))
            elif mode == LoggingMode.DEVELOPER:
                self.console_handler.setFormatter(StructuredFormatter(include_context=True, include_metrics=False))
            else:  # DEBUG
                self.console_handler.setFormatter(StructuredFormatter(include_context=True, include_metrics=True))

        self.info(
            f"Logging mode changed from {old_mode.value} to {mode.value}",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger"
        )

    def enable_module(self, module_name: str, level: Union[LogLevel, str] = LogLevel.DEBUG):
        """
        Enable logging for a specific module.

        Args:
            module_name: Module name (e.g., 'app.rag', 'app.agents')
            level: Log level for the module
        """
        if not self.config.allow_module_control:
            self.warn(
                "Module control is disabled in configuration",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "BackendLogger"
            )
            return

        # Convert string to enum if needed
        if isinstance(level, str):
            level = LogLevel(level.upper())

        self.module_controller.enable_module(module_name, level)

        self.info(
            f"Module '{module_name}' enabled at level {level.value}",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger"
        )

    def disable_module(self, module_name: str):
        """
        Disable logging for a specific module.

        Args:
            module_name: Module name (e.g., 'app.rag', 'app.agents')
        """
        if not self.config.allow_module_control:
            self.warn(
                "Module control is disabled in configuration",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "BackendLogger"
            )
            return

        self.module_controller.disable_module(module_name)

        self.info(
            f"Module '{module_name}' disabled",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger"
        )

    def set_module_level(self, module_name: str, console_level: Union[LogLevel, str],
                        file_level: Union[LogLevel, str] = None):
        """
        Set log levels for a specific module.

        Args:
            module_name: Module name (e.g., 'app.rag', 'app.agents')
            console_level: Console log level
            file_level: File log level (optional)
        """
        if not self.config.allow_module_control:
            self.warn(
                "Module control is disabled in configuration",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "BackendLogger"
            )
            return

        # Convert strings to enums if needed
        if isinstance(console_level, str):
            console_level = LogLevel(console_level.upper())
        if isinstance(file_level, str):
            file_level = LogLevel(file_level.upper())

        self.module_controller.set_module_level(module_name, console_level, file_level)

        self.info(
            f"Module '{module_name}' levels set: console={console_level.value}, file={file_level.value if file_level else 'unchanged'}",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger"
        )

    def get_module_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all modules"""
        return self.module_controller.get_module_status()

    def get_active_loggers(self) -> List[str]:
        """Get list of currently active (enabled) loggers"""
        return self.module_controller.get_active_loggers()

    def set_conversation_enabled(self, enabled: bool):
        """Enable or disable conversation layer"""
        self.config.conversation_config.enabled = enabled

        self.info(
            f"Conversation layer {'enabled' if enabled else 'disabled'}",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive logging metrics."""
        metrics = {
            "log_metrics": self.metrics_collector.get_metrics(),
            "sampling_stats": self.log_sampler.get_stats(),
            "queue_size": self.log_queue.qsize(),
            "queue_capacity": self.config.buffer_size,
            "active_modules": len(self.module_controller.get_active_loggers()),
            "configuration": {
                "mode": self.config.logging_mode.value,
                "pii_redaction": self.config.enable_pii_redaction,
                "sampling": self.config.enable_sampling,
                "error_aggregation": self.config.enable_error_aggregation,
                "opentelemetry": self.config.enable_opentelemetry
            }
        }

        return metrics

    def reset_metrics(self):
        """Reset all metrics counters."""
        self.metrics_collector.reset_metrics()

        self.info(
            "Metrics reset",
            LogCategory.SYSTEM_HEALTH,
            "BackendLogger"
        )

    def add_breadcrumb(self, message: str, category: str = "default", level: str = "info", data: Dict[str, Any] = None):
        """Add breadcrumb for error context tracking."""
        self.error_aggregator.add_breadcrumb(message, category, level, data)

    def create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create OpenTelemetry span for distributed tracing."""
        return self.otel_integration.create_span(name, attributes)

    def update_sampling_rate(self, rate: float):
        """Update log sampling rate at runtime."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Sampling rate must be between 0.0 and 1.0")

        self.config.sampling_rate = rate

        self.info(
            f"Sampling rate updated to {rate}",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger",
            data={"new_rate": rate}
        )

    def enable_pii_redaction(self, enabled: bool = True):
        """Enable or disable PII redaction at runtime."""
        self.config.enable_pii_redaction = enabled

        self.info(
            f"PII redaction {'enabled' if enabled else 'disabled'}",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger"
        )

    def enable_sampling(self, enabled: bool = True):
        """Enable or disable log sampling at runtime."""
        self.config.enable_sampling = enabled

        self.info(
            f"Log sampling {'enabled' if enabled else 'disabled'}",
            LogCategory.CONFIGURATION_MANAGEMENT,
            "BackendLogger"
        )

    def shutdown(self):
        """Shutdown the logging system gracefully"""
        self.info(
            "Backend logging system shutting down",
            category=LogCategory.SYSTEM_HEALTH,
            component="BackendLogger"
        )

        self.is_running = False

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

        # Close all handlers
        for handler in self.file_handlers.values():
            handler.close()

        if self.console_handler:
            self.console_handler.close()


# Global logger instance
_global_logger: Optional[BackendLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> BackendLogger:
    """Get the global backend logger instance"""
    global _global_logger
    
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = BackendLogger()
    
    return _global_logger


def configure_logger(config: LogConfiguration) -> BackendLogger:
    """Configure the global logger with custom settings"""
    global _global_logger
    
    with _logger_lock:
        if _global_logger:
            _global_logger.shutdown()
        _global_logger = BackendLogger(config)
    
    return _global_logger
