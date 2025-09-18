"""
Monitoring and Analytics System for Multi-Agent Architecture.

This module provides comprehensive monitoring, analytics, and reporting
capabilities for the entire multi-agent system including performance
metrics, usage analytics, and intelligent insights.

Features:
- Real-time system monitoring
- Comprehensive analytics engine
- Performance reporting and dashboards
- Predictive analytics and insights
- Alert management and notifications
- Historical data analysis
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

import structlog
from pydantic import BaseModel, Field

from .performance_optimizer import PerformanceOptimizer, PerformanceMetrics
from .advanced_access_controls import AdvancedAccessController
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.rag.core.unified_memory_system import UnifiedMemorySystem
from app.tools.unified_tool_repository import UnifiedToolRepository
from app.communication.agent_communication_system import AgentCommunicationSystem

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics collected."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"


@dataclass
class Alert:
    """System alert structure."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    metric_type: MetricType
    
    # Alert details
    source_component: str
    threshold_value: float
    current_value: float
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Status
    is_active: bool = True
    acknowledged_by: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricCollector:
    """Metric collection configuration."""
    name: str
    description: str
    metric_type: MetricType
    collection_interval: int  # seconds
    retention_period: int     # hours
    
    # Thresholds for alerting
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Collection function
    collector_function: Optional[callable] = None
    
    # Status
    enabled: bool = True
    last_collected: Optional[datetime] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    title: str
    period_start: datetime
    period_end: datetime
    
    # System overview
    system_health_score: float  # 0-100
    total_operations: int
    avg_response_time: float
    error_rate: float
    
    # Component performance
    rag_performance: Dict[str, Any]
    memory_performance: Dict[str, Any]
    communication_performance: Dict[str, Any]
    tool_performance: Dict[str, Any]
    
    # Usage analytics
    agent_activity: Dict[str, Any]
    resource_utilization: Dict[str, Any]
    
    # Trends and insights
    performance_trends: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "system"


class MonitoringSystem:
    """
    Comprehensive Monitoring System.
    
    Provides real-time monitoring, alerting, and analytics for
    the entire multi-agent architecture.
    """
    
    def __init__(
        self,
        performance_optimizer: PerformanceOptimizer,
        access_controller: AdvancedAccessController,
        unified_rag: UnifiedRAGSystem,
        memory_system: UnifiedMemorySystem,
        tool_repository: UnifiedToolRepository,
        communication_system: AgentCommunicationSystem
    ):
        """Initialize the monitoring system."""
        self.performance_optimizer = performance_optimizer
        self.access_controller = access_controller
        self.unified_rag = unified_rag
        self.memory_system = memory_system
        self.tool_repository = tool_repository
        self.communication_system = communication_system
        
        # Metric collection
        self.metric_collectors: Dict[str, MetricCollector] = {}
        self.collected_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Alert management
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Analytics data
        self.analytics_data: Dict[str, Any] = {}
        self.trend_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Performance tracking
        self.stats = {
            "total_metrics_collected": 0,
            "active_alerts": 0,
            "resolved_alerts": 0,
            "system_uptime": 0,
            "last_health_check": None
        }
        
        self.start_time = datetime.utcnow()
        self.is_initialized = False
        
        logger.info("Monitoring system created")
    
    async def initialize(self) -> None:
        """Initialize the monitoring system."""
        try:
            if self.is_initialized:
                logger.warning("Monitoring system already initialized")
                return
            
            logger.info("Initializing monitoring system...")
            
            # Initialize metric collectors
            await self._initialize_metric_collectors()
            
            # Start background tasks
            asyncio.create_task(self._metric_collection_loop())
            asyncio.create_task(self._alert_processing_loop())
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._analytics_processing_loop())
            
            self.is_initialized = True
            logger.info("Monitoring system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {str(e)}")
            raise
    
    async def _initialize_metric_collectors(self) -> None:
        """Initialize all metric collectors."""
        try:
            # System performance metrics
            self.metric_collectors["cpu_usage"] = MetricCollector(
                name="CPU Usage",
                description="System CPU utilization percentage",
                metric_type=MetricType.PERFORMANCE,
                collection_interval=60,
                retention_period=168,  # 7 days
                warning_threshold=70.0,
                error_threshold=85.0,
                critical_threshold=95.0
            )
            
            self.metric_collectors["memory_usage"] = MetricCollector(
                name="Memory Usage",
                description="System memory utilization percentage",
                metric_type=MetricType.PERFORMANCE,
                collection_interval=60,
                retention_period=168,
                warning_threshold=75.0,
                error_threshold=90.0,
                critical_threshold=98.0
            )
            
            # RAG system metrics
            self.metric_collectors["rag_query_rate"] = MetricCollector(
                name="RAG Query Rate",
                description="Number of RAG queries per minute",
                metric_type=MetricType.USAGE,
                collection_interval=60,
                retention_period=168
            )
            
            self.metric_collectors["rag_response_time"] = MetricCollector(
                name="RAG Response Time",
                description="Average RAG query response time in milliseconds",
                metric_type=MetricType.PERFORMANCE,
                collection_interval=60,
                retention_period=168,
                warning_threshold=1000.0,
                error_threshold=5000.0,
                critical_threshold=10000.0
            )
            
            # Memory system metrics
            self.metric_collectors["memory_operations"] = MetricCollector(
                name="Memory Operations",
                description="Number of memory operations per minute",
                metric_type=MetricType.USAGE,
                collection_interval=60,
                retention_period=168
            )
            
            # Communication metrics
            self.metric_collectors["message_rate"] = MetricCollector(
                name="Message Rate",
                description="Number of messages per minute",
                metric_type=MetricType.USAGE,
                collection_interval=60,
                retention_period=168
            )
            
            # Security metrics
            self.metric_collectors["access_denials"] = MetricCollector(
                name="Access Denials",
                description="Number of access denials per minute",
                metric_type=MetricType.SECURITY,
                collection_interval=60,
                retention_period=168,
                warning_threshold=10.0,
                error_threshold=50.0,
                critical_threshold=100.0
            )
            
            logger.info(f"Initialized {len(self.metric_collectors)} metric collectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize metric collectors: {str(e)}")
    
    async def collect_metric(self, collector_name: str) -> Optional[float]:
        """Collect a specific metric."""
        try:
            if collector_name not in self.metric_collectors:
                logger.warning(f"Unknown metric collector: {collector_name}")
                return None
            
            collector = self.metric_collectors[collector_name]
            
            if not collector.enabled:
                return None
            
            # Collect metric based on type
            value = None
            
            if collector_name == "cpu_usage":
                metrics = await self.performance_optimizer.collect_metrics()
                value = metrics.cpu_usage
            
            elif collector_name == "memory_usage":
                metrics = await self.performance_optimizer.collect_metrics()
                value = metrics.memory_usage
            
            elif collector_name == "rag_query_rate":
                rag_stats = self.unified_rag.get_system_stats()
                value = rag_stats.get("total_queries", 0)
            
            elif collector_name == "rag_response_time":
                rag_stats = self.unified_rag.get_system_stats()
                value = rag_stats.get("avg_query_time", 0.0)
            
            elif collector_name == "memory_operations":
                memory_stats = self.memory_system.get_system_stats()
                value = memory_stats.get("memories_created", 0) + memory_stats.get("memories_retrieved", 0)
            
            elif collector_name == "message_rate":
                comm_stats = self.communication_system.get_system_stats()
                value = comm_stats.get("total_messages", 0)
            
            elif collector_name == "access_denials":
                access_stats = self.access_controller.get_access_stats()
                value = access_stats.get("deny_decisions", 0)
            
            if value is not None:
                # Store metric
                timestamp = datetime.utcnow()
                self.collected_metrics[collector_name].append((timestamp, value))
                self.trend_data[collector_name].append((timestamp, value))
                
                # Update collector
                collector.last_collected = timestamp
                
                # Check thresholds and generate alerts
                await self._check_thresholds(collector, value)
                
                self.stats["total_metrics_collected"] += 1
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to collect metric {collector_name}: {str(e)}")
            return None
    
    async def _check_thresholds(self, collector: MetricCollector, value: float) -> None:
        """Check metric thresholds and generate alerts."""
        try:
            alert_severity = None
            threshold_value = None
            
            # Determine alert severity
            if collector.critical_threshold and value >= collector.critical_threshold:
                alert_severity = AlertSeverity.CRITICAL
                threshold_value = collector.critical_threshold
            elif collector.error_threshold and value >= collector.error_threshold:
                alert_severity = AlertSeverity.ERROR
                threshold_value = collector.error_threshold
            elif collector.warning_threshold and value >= collector.warning_threshold:
                alert_severity = AlertSeverity.WARNING
                threshold_value = collector.warning_threshold
            
            if alert_severity:
                # Create alert
                alert_id = f"{collector.name}_{alert_severity.value}_{int(time.time())}"
                
                alert = Alert(
                    alert_id=alert_id,
                    title=f"{collector.name} {alert_severity.value.upper()}",
                    description=f"{collector.name} has exceeded {alert_severity.value} threshold",
                    severity=alert_severity,
                    metric_type=collector.metric_type,
                    source_component=collector.name,
                    threshold_value=threshold_value,
                    current_value=value
                )
                
                self.alerts[alert_id] = alert
                self.alert_history.append(alert)
                self.stats["active_alerts"] += 1
                
                logger.warning(f"Alert generated: {alert.title} - Current: {value}, Threshold: {threshold_value}")
            
        except Exception as e:
            logger.error(f"Failed to check thresholds: {str(e)}")
    
    async def generate_performance_report(
        self,
        period_hours: int = 24,
        include_recommendations: bool = True
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=period_hours)
            
            # Calculate system health score
            health_score = await self._calculate_system_health_score()
            
            # Collect component performance data
            rag_performance = await self._analyze_rag_performance(start_time, end_time)
            memory_performance = await self._analyze_memory_performance(start_time, end_time)
            communication_performance = await self._analyze_communication_performance(start_time, end_time)
            tool_performance = await self._analyze_tool_performance(start_time, end_time)
            
            # Analyze usage patterns
            agent_activity = await self._analyze_agent_activity(start_time, end_time)
            resource_utilization = await self._analyze_resource_utilization(start_time, end_time)
            
            # Generate trends and insights
            performance_trends = await self._analyze_performance_trends(start_time, end_time)
            
            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = await self._generate_recommendations(health_score, performance_trends)
            
            # Create report
            report = PerformanceReport(
                report_id=f"report_{int(time.time())}",
                title=f"Performance Report - {period_hours}h Period",
                period_start=start_time,
                period_end=end_time,
                system_health_score=health_score,
                total_operations=self._calculate_total_operations(start_time, end_time),
                avg_response_time=self._calculate_avg_response_time(start_time, end_time),
                error_rate=self._calculate_error_rate(start_time, end_time),
                rag_performance=rag_performance,
                memory_performance=memory_performance,
                communication_performance=communication_performance,
                tool_performance=tool_performance,
                agent_activity=agent_activity,
                resource_utilization=resource_utilization,
                performance_trends=performance_trends,
                recommendations=recommendations
            )
            
            logger.info(f"Generated performance report: {report.report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {str(e)}")
            raise
    
    async def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            scores = []
            
            # Performance score (based on response times and resource usage)
            latest_metrics = await self.performance_optimizer.collect_metrics()
            
            # CPU score (100 - usage percentage)
            cpu_score = max(0, 100 - latest_metrics.cpu_usage)
            scores.append(cpu_score)
            
            # Memory score
            memory_score = max(0, 100 - latest_metrics.memory_usage)
            scores.append(memory_score)
            
            # Response time score (inverse relationship)
            response_time_score = max(0, 100 - (latest_metrics.avg_response_time / 10))  # Normalize to 0-100
            scores.append(response_time_score)
            
            # Error rate score
            error_rate_score = max(0, 100 - (latest_metrics.error_rate * 10))  # Normalize to 0-100
            scores.append(error_rate_score)
            
            # Cache performance score
            cache_score = latest_metrics.cache_hit_rate
            scores.append(cache_score)
            
            # Calculate weighted average
            health_score = sum(scores) / len(scores) if scores else 0
            
            return min(100, max(0, health_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate system health score: {str(e)}")
            return 0.0
    
    async def _analyze_rag_performance(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze RAG system performance."""
        try:
            rag_stats = self.unified_rag.get_system_stats()
            
            return {
                "total_queries": rag_stats.get("total_queries", 0),
                "avg_query_time": rag_stats.get("avg_query_time", 0.0),
                "total_agents": rag_stats.get("total_agents", 0),
                "total_collections": rag_stats.get("total_collections", 0),
                "total_documents": rag_stats.get("total_documents", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze RAG performance: {str(e)}")
            return {}
    
    async def _analyze_memory_performance(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze memory system performance."""
        try:
            memory_stats = self.memory_system.get_system_stats()
            
            return {
                "total_memories": memory_stats.get("total_memories", 0),
                "memories_created": memory_stats.get("memories_created", 0),
                "memories_retrieved": memory_stats.get("memories_retrieved", 0),
                "memories_consolidated": memory_stats.get("memories_consolidated", 0),
                "cleanup_cycles": memory_stats.get("cleanup_cycles", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze memory performance: {str(e)}")
            return {}
    
    async def _analyze_communication_performance(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze communication system performance."""
        try:
            comm_stats = self.communication_system.get_system_stats()
            
            return {
                "total_messages": comm_stats.get("total_messages", 0),
                "messages_delivered": comm_stats.get("messages_delivered", 0),
                "messages_failed": comm_stats.get("messages_failed", 0),
                "active_agents": comm_stats.get("active_agents", 0),
                "active_channels": comm_stats.get("active_channels", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze communication performance: {str(e)}")
            return {}
    
    async def _analyze_tool_performance(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze tool repository performance."""
        try:
            tool_stats = self.tool_repository.get_repository_stats()
            
            return {
                "total_tools": tool_stats.get("total_tools", 0),
                "total_tool_calls": tool_stats.get("total_tool_calls", 0),
                "avg_tool_performance": tool_stats.get("avg_tool_performance", 0.0),
                "tools_by_category": tool_stats.get("tools_by_category", {}),
                "agent_profiles_count": tool_stats.get("agent_profiles_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze tool performance: {str(e)}")
            return {}
    
    async def _metric_collection_loop(self) -> None:
        """Background loop for metric collection."""
        while True:
            try:
                for collector_name, collector in self.metric_collectors.items():
                    if collector.enabled:
                        await self.collect_metric(collector_name)
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Metric collection loop error: {str(e)}")
    
    async def _alert_processing_loop(self) -> None:
        """Background loop for alert processing."""
        while True:
            try:
                await asyncio.sleep(300)  # Process alerts every 5 minutes
                
                # Auto-resolve alerts that are no longer active
                current_time = datetime.utcnow()
                for alert in list(self.alerts.values()):
                    if alert.is_active and (current_time - alert.created_at).total_seconds() > 3600:
                        # Auto-resolve alerts older than 1 hour
                        alert.is_active = False
                        alert.resolved_at = current_time
                        self.stats["active_alerts"] -= 1
                        self.stats["resolved_alerts"] += 1
                
            except Exception as e:
                logger.error(f"Alert processing loop error: {str(e)}")
    
    async def _health_check_loop(self) -> None:
        """Background loop for system health checks."""
        while True:
            try:
                await asyncio.sleep(300)  # Health check every 5 minutes
                
                # Update system uptime
                self.stats["system_uptime"] = (datetime.utcnow() - self.start_time).total_seconds()
                self.stats["last_health_check"] = datetime.utcnow()
                
                # Perform health checks on all components
                await self._perform_health_checks()
                
            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
    
    async def _analytics_processing_loop(self) -> None:
        """Background loop for analytics processing."""
        while True:
            try:
                await asyncio.sleep(3600)  # Process analytics every hour
                
                # Update analytics data
                await self._update_analytics_data()
                
            except Exception as e:
                logger.error(f"Analytics processing loop error: {str(e)}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            **self.stats,
            "metric_collectors_count": len(self.metric_collectors),
            "active_alerts_count": len([a for a in self.alerts.values() if a.is_active]),
            "total_alerts_count": len(self.alert_history),
            "metrics_collected_count": sum(len(metrics) for metrics in self.collected_metrics.values())
        }
