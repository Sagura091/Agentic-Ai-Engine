"""
Logging Models and Data Structures

Defines the core data models for the backend logging system including
log entries, levels, categories, and context information.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class LogLevel(str, Enum):
    """Log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class LogCategory(str, Enum):
    """Log categories for better organization"""
    AGENT_OPERATIONS = "agent_operations"
    API_LAYER = "api_layer"
    DATABASE_LAYER = "database_layer"
    EXTERNAL_INTEGRATIONS = "external_integrations"
    SECURITY_EVENTS = "security_events"
    CONFIGURATION_MANAGEMENT = "configuration_management"
    RESOURCE_MANAGEMENT = "resource_management"
    SYSTEM_HEALTH = "system_health"
    ORCHESTRATION = "orchestration"
    PERFORMANCE = "performance"
    USER_INTERACTION = "user_interaction"
    ERROR_TRACKING = "error_tracking"


class LogContext(BaseModel):
    """Context information for log entries"""
    correlation_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    class Config:
        extra = "allow"


class PerformanceMetrics(BaseModel):
    """Performance metrics for operations"""
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    db_queries_count: Optional[int] = None
    api_calls_count: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None


class ErrorDetails(BaseModel):
    """Detailed error information"""
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    error_category: Optional[str] = None
    severity: Optional[str] = None
    user_impact: Optional[str] = None
    resolution_steps: Optional[List[str]] = None


class AgentMetrics(BaseModel):
    """Agent-specific metrics"""
    agent_type: Optional[str] = None
    agent_state: Optional[str] = None
    tools_used: Optional[List[str]] = None
    tasks_completed: Optional[int] = None
    tasks_failed: Optional[int] = None
    execution_time_ms: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    tokens_consumed: Optional[int] = None
    api_calls_made: Optional[int] = None


class APIMetrics(BaseModel):
    """API request/response metrics"""
    method: Optional[str] = None
    endpoint: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    rate_limit_remaining: Optional[int] = None
    authentication_method: Optional[str] = None


class DatabaseMetrics(BaseModel):
    """Database operation metrics"""
    query_type: Optional[str] = None
    table_name: Optional[str] = None
    execution_time_ms: Optional[float] = None
    rows_affected: Optional[int] = None
    connection_pool_size: Optional[int] = None
    active_connections: Optional[int] = None
    query_hash: Optional[str] = None


class LogEntry(BaseModel):
    """Complete log entry structure"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: LogLevel
    category: LogCategory
    message: str
    component: str
    
    # Context information
    context: LogContext = Field(default_factory=LogContext)
    
    # Additional data
    data: Optional[Dict[str, Any]] = None
    
    # Metrics
    performance: Optional[PerformanceMetrics] = None
    agent_metrics: Optional[AgentMetrics] = None
    api_metrics: Optional[APIMetrics] = None
    database_metrics: Optional[DatabaseMetrics] = None
    
    # Error information
    error_details: Optional[ErrorDetails] = None
    
    # System information
    hostname: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[str] = None
    
    # Environment
    environment: Optional[str] = None
    version: Optional[str] = None
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LogQuery(BaseModel):
    """Query parameters for log retrieval"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: Optional[List[LogLevel]] = None
    categories: Optional[List[LogCategory]] = None
    components: Optional[List[str]] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    search_term: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class LogStats(BaseModel):
    """Log statistics and aggregations"""
    total_logs: int
    logs_by_level: Dict[LogLevel, int]
    logs_by_category: Dict[LogCategory, int]
    logs_by_component: Dict[str, int]
    error_rate: float
    average_response_time: Optional[float] = None
    peak_memory_usage: Optional[float] = None
    active_agents: int
    failed_operations: int
    time_range: Dict[str, datetime]


class LogConfiguration(BaseModel):
    """Logging system configuration"""
    log_level: LogLevel = LogLevel.INFO
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_json_format: bool = True
    enable_async_logging: bool = True
    max_log_file_size_mb: int = 100
    max_log_files: int = 10
    log_retention_days: int = 30
    buffer_size: int = 1000
    flush_interval_seconds: int = 5
    enable_performance_logging: bool = True
    enable_agent_metrics: bool = True
    enable_api_metrics: bool = True
    enable_database_metrics: bool = True
    exclude_patterns: List[str] = Field(default_factory=list)
    include_stack_trace: bool = True
    correlation_id_header: str = "X-Correlation-ID"
    
    class Config:
        extra = "allow"
