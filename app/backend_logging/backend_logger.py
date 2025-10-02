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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import threading
import queue
import time
import yaml
import os

from .models import (
    LogEntry, LogLevel, LogCategory, LogContext,
    PerformanceMetrics, ErrorDetails, AgentMetrics,
    APIMetrics, DatabaseMetrics, LogConfiguration,
    LoggingMode, ModuleConfig, ConversationConfig, TierConfig
)
from .context import CorrelationContext, SystemContext
from .formatters import JSONFormatter, StructuredFormatter, ConversationFormatter
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
            allow_module_control=settings.LOG_ALLOW_MODULE_CONTROL
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
                "modules_enabled": len(self.module_controller.get_active_loggers())
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

            # Choose formatter based on mode
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
                'style': self.config.conversation_config.style
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

                            # Re-setup logging with new configuration
                            self._setup_logging()

                            self.info(
                                "Configuration reloaded successfully",
                                category=LogCategory.CONFIGURATION_MANAGEMENT,
                                component="BackendLogger",
                                data={
                                    "mode": self.config.logging_mode.value,
                                    "modules_enabled": len(self.module_controller.get_active_loggers())
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
