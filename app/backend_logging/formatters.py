"""
Logging Formatters

Provides various formatters for log output including JSON and structured text formats.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from .models import LogEntry, LogLevel, LogCategory, LoggingMode

# Import Rich for color support
try:
    from rich.console import Console
    from rich.text import Text
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """
    JSON formatter that outputs log entries as structured JSON
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON"""
        try:
            # Get our custom log entry if available
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                log_dict = self._log_entry_to_dict(log_entry)
            else:
                # Fallback for standard logging records
                log_dict = self._standard_record_to_dict(record)
            
            return json.dumps(log_dict, separators=(',', ':'), default=str)
            
        except Exception as e:
            # Fallback to basic format if JSON serialization fails
            return f'{{"timestamp": "{datetime.utcnow().isoformat()}", "level": "ERROR", "message": "JSON formatting error: {str(e)}", "original_message": "{record.getMessage()}"}}'
    
    def _log_entry_to_dict(self, log_entry: LogEntry) -> Dict[str, Any]:
        """Convert LogEntry to dictionary"""
        log_dict = {
            "timestamp": log_entry.timestamp.isoformat(),
            "level": log_entry.level.value,
            "category": log_entry.category.value,
            "component": log_entry.component,
            "message": log_entry.message,
        }
        
        # Add context information
        if log_entry.context:
            context_dict = log_entry.context.dict(exclude_none=True)
            log_dict["context"] = context_dict
        
        # Add additional data
        if log_entry.data:
            log_dict["data"] = log_entry.data
        
        # Add metrics
        if log_entry.performance:
            log_dict["performance"] = log_entry.performance.dict(exclude_none=True)
        
        if log_entry.agent_metrics:
            log_dict["agent_metrics"] = log_entry.agent_metrics.dict(exclude_none=True)
        
        if log_entry.api_metrics:
            log_dict["api_metrics"] = log_entry.api_metrics.dict(exclude_none=True)
        
        if log_entry.database_metrics:
            log_dict["database_metrics"] = log_entry.database_metrics.dict(exclude_none=True)
        
        # Add error details
        if log_entry.error_details:
            log_dict["error_details"] = log_entry.error_details.dict(exclude_none=True)
        
        # Add system information
        if log_entry.hostname:
            log_dict["hostname"] = log_entry.hostname
        if log_entry.process_id:
            log_dict["process_id"] = log_entry.process_id
        if log_entry.thread_id:
            log_dict["thread_id"] = log_entry.thread_id
        if log_entry.environment:
            log_dict["environment"] = log_entry.environment
        if log_entry.version:
            log_dict["version"] = log_entry.version
        
        return log_dict
    
    def _standard_record_to_dict(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Convert standard logging record to dictionary"""
        log_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "category": "system_health",  # Default category
        }
        
        # Add exception information if present
        if record.exc_info:
            log_dict["error_details"] = {
                "stack_trace": self.formatException(record.exc_info)
            }
        
        return log_dict


class StructuredFormatter(logging.Formatter):
    """
    Human-readable structured formatter for console output
    """
    
    def __init__(self, include_context: bool = True, include_metrics: bool = False):
        super().__init__()
        self.include_context = include_context
        self.include_metrics = include_metrics
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as structured text"""
        try:
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                return self._format_log_entry(log_entry)
            else:
                return self._format_standard_record(record)
                
        except Exception as e:
            # Fallback format
            return f"[{datetime.utcnow().isoformat()}] [ERROR] [Formatter] Formatting error: {str(e)} | Original: {record.getMessage()}"
    
    def _format_log_entry(self, log_entry: LogEntry) -> str:
        """Format a LogEntry as structured text"""
        # Basic log line
        timestamp = log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [
            f"[{timestamp}]",
            f"[{log_entry.level.value}]",
            f"[{log_entry.category.value}]",
            f"[{log_entry.component}]",
            log_entry.message
        ]
        
        base_line = " ".join(parts)
        
        # Add context information
        context_parts = []
        if self.include_context and log_entry.context:
            if log_entry.context.correlation_id:
                context_parts.append(f"correlation_id={log_entry.context.correlation_id}")
            if log_entry.context.session_id:
                context_parts.append(f"session_id={log_entry.context.session_id}")
            if log_entry.context.agent_id:
                context_parts.append(f"agent_id={log_entry.context.agent_id}")
            if log_entry.context.request_id:
                context_parts.append(f"request_id={log_entry.context.request_id}")
        
        if context_parts:
            base_line += f" | Context: {', '.join(context_parts)}"
        
        # Add performance metrics
        if self.include_metrics and log_entry.performance:
            metrics_parts = []
            if log_entry.performance.duration_ms:
                metrics_parts.append(f"duration={log_entry.performance.duration_ms:.2f}ms")
            if log_entry.performance.memory_usage_mb:
                metrics_parts.append(f"memory={log_entry.performance.memory_usage_mb:.2f}MB")
            if log_entry.performance.cpu_usage_percent:
                metrics_parts.append(f"cpu={log_entry.performance.cpu_usage_percent:.1f}%")
            
            if metrics_parts:
                base_line += f" | Performance: {', '.join(metrics_parts)}"
        
        # Add data if present
        if log_entry.data:
            try:
                data_str = json.dumps(log_entry.data, separators=(',', ':'), default=str)
                if len(data_str) > 200:
                    data_str = data_str[:200] + "..."
                base_line += f" | Data: {data_str}"
            except Exception:
                base_line += f" | Data: <serialization_error>"
        
        # Add error details
        if log_entry.error_details:
            error_parts = []
            if log_entry.error_details.error_type:
                error_parts.append(f"type={log_entry.error_details.error_type}")
            if log_entry.error_details.error_code:
                error_parts.append(f"code={log_entry.error_details.error_code}")
            if log_entry.error_details.severity:
                error_parts.append(f"severity={log_entry.error_details.severity}")
            
            if error_parts:
                base_line += f" | Error: {', '.join(error_parts)}"
            
            # Add stack trace on new line if present
            if log_entry.error_details.stack_trace:
                base_line += f"\nStack Trace:\n{log_entry.error_details.stack_trace}"
        
        return base_line
    
    def _format_standard_record(self, record: logging.LogRecord) -> str:
        """Format a standard logging record"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [
            f"[{timestamp}]",
            f"[{record.levelname}]",
            f"[system_health]",
            f"[{record.name}]",
            record.getMessage()
        ]
        
        base_line = " ".join(parts)
        
        # Add exception information if present
        if record.exc_info:
            base_line += f"\nException:\n{self.formatException(record.exc_info)}"
        
        return base_line


class ColoredStructuredFormatter(logging.Formatter):
    """
    Production-ready color-coded structured formatter using Rich library.

    Features:
    - Color-coded log levels (DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=bold red)
    - Color-coded categories (AGENT=blue, RAG=purple, MEMORY=magenta, LLM=bright_blue, TOOL=orange3)
    - Color-coded modules (app.agents=blue, app.rag=purple, etc.)
    - Special green color for agent reasoning/thinking
    - Automatic TTY detection (no colors in pipes/files)
    - Cross-platform support (Windows, Linux, macOS)
    """

    # Color scheme for log levels
    LEVEL_COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold red',
    }

    # Color scheme for log categories
    CATEGORY_COLORS = {
        'AGENT_OPERATIONS': 'blue',
        'RAG_OPERATIONS': 'purple',
        'MEMORY_OPERATIONS': 'magenta',
        'LLM_OPERATIONS': 'bright_blue',
        'TOOL_OPERATIONS': 'orange3',
        'API_LAYER': 'green',
        'DATABASE_LAYER': 'cyan',
        'SECURITY_EVENTS': 'red',
        'ERROR_TRACKING': 'bright_red',
        'PERFORMANCE': 'yellow',
        'SYSTEM_HEALTH': 'white',
        'CONFIGURATION_MANAGEMENT': 'bright_cyan',
        'RESOURCE_MANAGEMENT': 'bright_magenta',
        'ORCHESTRATION': 'bright_yellow',
        'COMMUNICATION': 'bright_green',
        'SERVICE_OPERATIONS': 'cyan',
        'USER_INTERACTION': 'bright_green',
        'EXTERNAL_INTEGRATIONS': 'orange3',
    }

    # Color scheme for module names
    MODULE_COLORS = {
        'app.agents': 'blue',
        'app.rag': 'purple',
        'app.memory': 'magenta',
        'app.llm': 'bright_blue',
        'app.tools': 'orange3',
        'app.api': 'green',
        'app.core': 'white',
        'app.services': 'cyan',
        'app.config': 'bright_cyan',
        'app.models': 'bright_white',
        'app.optimization': 'yellow',
        'app.integrations': 'orange3',
        'app.orchestration': 'bright_yellow',
        'app.communication': 'bright_green',
    }

    # Special color for agent reasoning/thinking
    AGENT_REASONING_COLOR = 'bright_green'
    AGENT_RESPONSE_COLOR = 'bright_green'

    def __init__(self,
                 include_context: bool = True,
                 include_metrics: bool = False,
                 enable_colors: bool = True,
                 force_colors: bool = False,
                 color_scheme: str = "default"):
        super().__init__()
        self.include_context = include_context
        self.include_metrics = include_metrics
        self.enable_colors = enable_colors and RICH_AVAILABLE
        self.force_colors = force_colors
        self.color_scheme = color_scheme

        # Initialize Rich console
        if self.enable_colors:
            # Create custom theme
            custom_theme = Theme({
                "debug": self.LEVEL_COLORS['DEBUG'],
                "info": self.LEVEL_COLORS['INFO'],
                "warning": self.LEVEL_COLORS['WARNING'],
                "error": self.LEVEL_COLORS['ERROR'],
                "critical": self.LEVEL_COLORS['CRITICAL'],
                "agent_reasoning": self.AGENT_REASONING_COLOR,
                "agent_response": self.AGENT_RESPONSE_COLOR,
            })

            # Determine if we should use colors
            force_terminal = self.force_colors or None

            self.console = Console(
                theme=custom_theme,
                force_terminal=force_terminal,
                file=sys.stdout,
                legacy_windows=False  # Use modern Windows terminal features
            )
        else:
            self.console = None

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with color coding"""
        try:
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                return self._format_log_entry_colored(log_entry)
            else:
                return self._format_standard_record_colored(record)

        except Exception as e:
            # Fallback format without colors
            return f"[{datetime.utcnow().isoformat()}] [ERROR] [Formatter] Formatting error: {str(e)} | Original: {record.getMessage()}"

    def _format_log_entry_colored(self, log_entry: LogEntry) -> str:
        """Format a LogEntry with color coding using ANSI escape codes"""
        if not self.enable_colors or not self.console:
            # Fallback to non-colored format
            return self._format_log_entry_plain(log_entry)

        # Check if this is agent reasoning/thinking
        is_agent_reasoning = self._is_agent_reasoning(log_entry)
        is_agent_response = self._is_agent_response(log_entry)

        # Build colored text components
        text = Text()

        # Timestamp (if needed)
        timestamp = log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        text.append(f"[{timestamp}] ", style="dim")

        # Log level with color
        level_color = self.LEVEL_COLORS.get(log_entry.level.value, 'white')
        text.append(f"[{log_entry.level.value}] ", style=level_color + " bold")

        # Category with color
        category_color = self.CATEGORY_COLORS.get(log_entry.category.value.upper(), 'white')
        text.append(f"[{log_entry.category.value}] ", style=category_color + " bold")

        # Component/Module with color
        component_color = self._get_module_color(log_entry.component)
        text.append(f"[{log_entry.component}] ", style=component_color)

        # Message with special color for agent reasoning/responses
        if is_agent_reasoning:
            text.append(log_entry.message, style=self.AGENT_REASONING_COLOR + " bold")
        elif is_agent_response:
            text.append(log_entry.message, style=self.AGENT_RESPONSE_COLOR + " bold")
        else:
            # Use level color for regular messages
            text.append(log_entry.message, style=level_color)

        # Convert Rich Text to ANSI string with color codes
        # Use a temporary console with ANSI export enabled
        from io import StringIO
        string_buffer = StringIO()
        temp_console = Console(
            file=string_buffer,
            force_terminal=True,
            legacy_windows=False,
            width=200,
            color_system="auto"
        )
        temp_console.print(text, end="")
        base_line = string_buffer.getvalue()

        # Add context information (non-colored for readability)
        if self.include_context and log_entry.context:
            context_parts = []
            if log_entry.context.correlation_id:
                context_parts.append(f"correlation_id={log_entry.context.correlation_id}")
            if log_entry.context.session_id:
                context_parts.append(f"session_id={log_entry.context.session_id}")
            if log_entry.context.agent_id:
                context_parts.append(f"agent_id={log_entry.context.agent_id}")

            if context_parts:
                base_line += f" | Context: {', '.join(context_parts)}"

        # Add performance metrics
        if self.include_metrics and log_entry.performance:
            metrics_parts = []
            if log_entry.performance.duration_ms:
                metrics_parts.append(f"duration={log_entry.performance.duration_ms:.2f}ms")
            if log_entry.performance.memory_usage_mb:
                metrics_parts.append(f"memory={log_entry.performance.memory_usage_mb:.2f}MB")

            if metrics_parts:
                base_line += f" | Performance: {', '.join(metrics_parts)}"

        return base_line

    def _format_log_entry_plain(self, log_entry: LogEntry) -> str:
        """Format a LogEntry without colors (fallback)"""
        timestamp = log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [
            f"[{timestamp}]",
            f"[{log_entry.level.value}]",
            f"[{log_entry.category.value}]",
            f"[{log_entry.component}]",
            log_entry.message
        ]

        base_line = " ".join(parts)

        # Add context information
        if self.include_context and log_entry.context:
            context_parts = []
            if log_entry.context.correlation_id:
                context_parts.append(f"correlation_id={log_entry.context.correlation_id}")
            if log_entry.context.session_id:
                context_parts.append(f"session_id={log_entry.context.session_id}")
            if log_entry.context.agent_id:
                context_parts.append(f"agent_id={log_entry.context.agent_id}")

            if context_parts:
                base_line += f" | Context: {', '.join(context_parts)}"

        return base_line

    def _format_standard_record_colored(self, record: logging.LogRecord) -> str:
        """Format a standard logging record with colors"""
        if not self.enable_colors or not self.console:
            return self._format_standard_record_plain(record)

        text = Text()
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        text.append(f"[{timestamp}] ", style="dim")

        level_color = self.LEVEL_COLORS.get(record.levelname, 'white')
        text.append(f"[{record.levelname}] ", style=level_color + " bold")

        text.append(f"[system_health] ", style="white")
        text.append(f"[{record.name}] ", style="cyan")
        text.append(record.getMessage(), style=level_color)

        # Convert to ANSI string
        from io import StringIO
        string_buffer = StringIO()
        temp_console = Console(
            file=string_buffer,
            force_terminal=True,
            legacy_windows=False,
            width=200,
            color_system="auto"
        )
        temp_console.print(text, end="")
        return string_buffer.getvalue()

    def _format_standard_record_plain(self, record: logging.LogRecord) -> str:
        """Format a standard logging record without colors"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [
            f"[{timestamp}]",
            f"[{record.levelname}]",
            f"[system_health]",
            f"[{record.name}]",
            record.getMessage()
        ]
        return " ".join(parts)

    def _get_module_color(self, component: str) -> str:
        """Get color for a module/component name"""
        # Check for exact match first
        if component in self.MODULE_COLORS:
            return self.MODULE_COLORS[component]

        # Check for prefix match (e.g., app.agents.react -> app.agents)
        for module_prefix, color in self.MODULE_COLORS.items():
            if component.startswith(module_prefix):
                return color

        # Default color
        return 'white'

    def _is_agent_reasoning(self, log_entry: LogEntry) -> bool:
        """Check if this log entry is agent reasoning/thinking"""
        message_lower = log_entry.message.lower()

        # Check for reasoning keywords
        reasoning_keywords = [
            'thinking', 'reasoning', 'analyzing', 'considering',
            'evaluating', 'planning', 'deciding', 'thought',
            'reflection', 'deliberating', 'pondering'
        ]

        # Check if message contains reasoning keywords
        if any(keyword in message_lower for keyword in reasoning_keywords):
            return True

        # Check if category is agent operations and message suggests reasoning
        if log_entry.category == LogCategory.AGENT_OPERATIONS:
            if any(word in message_lower for word in ['think', 'reason', 'analyze', 'plan']):
                return True

        return False

    def _is_agent_response(self, log_entry: LogEntry) -> bool:
        """Check if this log entry is agent response/output"""
        message_lower = log_entry.message.lower()

        # Check for response keywords
        response_keywords = [
            'response:', 'answer:', 'result:', 'output:',
            'agent says', 'agent responds', 'final answer',
            'conclusion:', 'summary:'
        ]

        # Check if message contains response keywords
        if any(keyword in message_lower for keyword in response_keywords):
            return True

        # Check if category is agent operations and message suggests response
        if log_entry.category == LogCategory.AGENT_OPERATIONS:
            if any(word in message_lower for word in ['respond', 'answer', 'reply', 'output']):
                return True

        return False


class CompactFormatter(logging.Formatter):
    """
    Compact formatter for high-volume logging scenarios
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record in compact format"""
        try:
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                timestamp = log_entry.timestamp.strftime("%H:%M:%S.%f")[:-3]

                # Compact format: TIME LEVEL COMPONENT MESSAGE
                parts = [
                    timestamp,
                    log_entry.level.value[0],  # First letter of level
                    log_entry.component[:10],  # Truncated component name
                    log_entry.message[:100]    # Truncated message
                ]

                line = " ".join(parts)

                # Add correlation ID if present
                if log_entry.context and log_entry.context.correlation_id:
                    line += f" [{log_entry.context.correlation_id[:8]}]"

                return line
            else:
                timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
                return f"{timestamp} {record.levelname[0]} {record.name[:10]} {record.getMessage()[:100]}"

        except Exception:
            return f"{datetime.utcnow().strftime('%H:%M:%S')} E Formatter <error>"


class MetricsFormatter(logging.Formatter):
    """
    Specialized formatter for metrics and performance data
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format metrics-focused log entries"""
        try:
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry

                # Focus on metrics data
                metrics_data = {}

                if log_entry.performance:
                    metrics_data.update(log_entry.performance.dict(exclude_none=True))

                if log_entry.agent_metrics:
                    metrics_data.update({f"agent_{k}": v for k, v in log_entry.agent_metrics.dict(exclude_none=True).items()})

                if log_entry.api_metrics:
                    metrics_data.update({f"api_{k}": v for k, v in log_entry.api_metrics.dict(exclude_none=True).items()})

                if log_entry.database_metrics:
                    metrics_data.update({f"db_{k}": v for k, v in log_entry.database_metrics.dict(exclude_none=True).items()})

                # Create metrics line
                timestamp = log_entry.timestamp.isoformat()
                component = log_entry.component

                metrics_str = " ".join([f"{k}={v}" for k, v in metrics_data.items()])

                return f"{timestamp} {component} {metrics_str}"
            else:
                return record.getMessage()

        except Exception:
            return f"{datetime.utcnow().isoformat()} metrics_formatter_error"


class ConversationFormatter(logging.Formatter):
    """
    User-facing conversation formatter for clean, emoji-enhanced agent dialogue with color coding.

    This formatter creates clean, conversational output without technical details,
    correlation IDs, or timestamps. It's designed for end users interacting with agents.

    Features:
    - Bright green color for agent reasoning/thinking (easy to read)
    - Bright green color for agent responses (clear communication)
    - Color-coded tool usage and results
    - Emoji enhancement for visual clarity
    - Automatic TTY detection
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.emoji_enhanced = self.config.get('emoji_enhanced', True)
        self.show_reasoning = self.config.get('show_reasoning', True)
        self.show_tool_usage = self.config.get('show_tool_usage', True)
        self.show_tool_results = self.config.get('show_tool_results', True)
        self.max_reasoning_length = self.config.get('max_reasoning_length', 200)
        self.max_result_length = self.config.get('max_result_length', 500)
        self.style = self.config.get('style', 'conversational')
        self.enable_colors = self.config.get('enable_colors', True) and RICH_AVAILABLE

        # Initialize Rich console for color output
        if self.enable_colors:
            self.console = Console(
                file=sys.stdout,
                force_terminal=self.config.get('force_colors', False) or None,
                legacy_windows=False
            )
        else:
            self.console = None

    def format(self, record: logging.LogRecord) -> str:
        """Format a conversation log entry"""
        try:
            # Check if this is a conversation message
            if not hasattr(record, 'conversation_type'):
                return ""  # Not a conversation message, skip

            conversation_type = record.conversation_type
            message = record.getMessage()

            # Format based on conversation type
            if conversation_type == 'user_query':
                return self._format_user_query(message)
            elif conversation_type == 'agent_acknowledgment':
                return self._format_agent_acknowledgment(message)
            elif conversation_type == 'agent_thinking':
                return self._format_agent_thinking(message) if self.show_reasoning else ""
            elif conversation_type == 'agent_goal':
                return self._format_agent_goal(message)
            elif conversation_type == 'agent_decision':
                return self._format_agent_decision(message)
            elif conversation_type == 'tool_usage':
                return self._format_tool_usage(message) if self.show_tool_usage else ""
            elif conversation_type == 'tool_result':
                return self._format_tool_result(message) if self.show_tool_results else ""
            elif conversation_type == 'agent_action':
                return self._format_agent_action(message)
            elif conversation_type == 'agent_response':
                return self._format_agent_response(message)
            elif conversation_type == 'agent_insight':
                return self._format_agent_insight(message)
            elif conversation_type == 'error':
                return self._format_error(message)
            elif conversation_type == 'warning':
                return self._format_warning(message)
            elif conversation_type == 'success':
                return self._format_success(message)
            else:
                return message  # Unknown type, return as-is

        except Exception as e:
            # Fallback: return message without formatting
            return record.getMessage() if hasattr(record, 'getMessage') else str(record)

    def _format_user_query(self, message: str) -> str:
        """Format user query"""
        emoji = "🧑 " if self.emoji_enhanced else ""
        text = f"{emoji}User: {message}"
        return self._colorize(text, "bright_white")

    def _format_agent_acknowledgment(self, message: str) -> str:
        """Format agent acknowledgment"""
        emoji = "🤖 " if self.emoji_enhanced else ""
        text = f"{emoji}Agent: {message}"
        return self._colorize(text, "bright_cyan")

    def _format_agent_thinking(self, message: str) -> str:
        """Format agent thinking/reasoning with bright green color"""
        emoji = "🔍 " if self.emoji_enhanced else ""
        # Truncate if too long
        if len(message) > self.max_reasoning_length:
            message = message[:self.max_reasoning_length] + "..."
        text = f"{emoji}Thinking: {message}"
        return self._colorize(text, "bright_green bold")

    def _format_agent_goal(self, message: str) -> str:
        """Format agent goal (autonomous agents)"""
        emoji = "🎯 " if self.emoji_enhanced else ""
        text = f"{emoji}Goal: {message}"
        return self._colorize(text, "bright_yellow")

    def _format_agent_decision(self, message: str) -> str:
        """Format agent decision (autonomous agents) with bright green color"""
        emoji = "🧠 " if self.emoji_enhanced else ""
        text = f"{emoji}Decision: {message}"
        return self._colorize(text, "bright_green bold")

    def _format_tool_usage(self, message: str) -> str:
        """Format tool usage"""
        emoji = "🔧 " if self.emoji_enhanced else ""
        # Expected format: "tool_name|purpose"
        if "|" in message:
            tool_name, purpose = message.split("|", 1)
            text = f"{emoji}Using: {tool_name}\n   → {purpose}"
        else:
            text = f"{emoji}Using: {message}"
        return self._colorize(text, "orange3")

    def _format_tool_result(self, message: str) -> str:
        """Format tool result"""
        emoji = "✅ " if self.emoji_enhanced else ""
        # Truncate if too long
        if len(message) > self.max_result_length:
            message = message[:self.max_result_length] + "..."
        text = f"{emoji}{message}"
        return self._colorize(text, "cyan")

    def _format_agent_action(self, message: str) -> str:
        """Format agent action"""
        emoji = "⚙️ " if self.emoji_enhanced else ""
        text = f"{emoji}Action: {message}"
        return self._colorize(text, "bright_blue")

    def _format_agent_response(self, message: str) -> str:
        """Format agent final response with bright green color"""
        emoji = "💬 " if self.emoji_enhanced else ""
        text = f"{emoji}{message}"
        return self._colorize(text, "bright_green bold")

    def _format_agent_insight(self, message: str) -> str:
        """Format agent insight"""
        emoji = "💡 " if self.emoji_enhanced else ""
        text = f"{emoji}Insight: {message}"
        return self._colorize(text, "bright_yellow")

    def _format_error(self, message: str) -> str:
        """Format error message"""
        emoji = "❌ " if self.emoji_enhanced else ""
        text = f"{emoji}Error: {message}"
        return self._colorize(text, "red bold")

    def _format_warning(self, message: str) -> str:
        """Format warning message"""
        emoji = "⚠️ " if self.emoji_enhanced else ""
        text = f"{emoji}Warning: {message}"
        return self._colorize(text, "yellow")

    def _format_success(self, message: str) -> str:
        """Format success message"""
        emoji = "✅ " if self.emoji_enhanced else ""
        text = f"{emoji}{message}"
        return self._colorize(text, "green")

    def _colorize(self, text: str, style: str) -> str:
        """Apply color to text using Rich console and return ANSI string"""
        if not self.enable_colors or not self.console:
            return text

        try:
            rich_text = Text(text, style=style)

            # Convert to ANSI string
            from io import StringIO
            string_buffer = StringIO()
            temp_console = Console(
                file=string_buffer,
                force_terminal=True,
                legacy_windows=False,
                width=200,
                color_system="auto"
            )
            temp_console.print(rich_text, end="")
            return string_buffer.getvalue()
        except Exception:
            # Fallback to plain text if coloring fails
            return text
