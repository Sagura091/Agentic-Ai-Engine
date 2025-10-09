"""
Monitoring and Analytics System - Streamlined Foundation.

This module provides simple monitoring and analytics capabilities
for the multi-agent system including basic performance metrics
and usage analytics.

Features:
- Basic system monitoring
- Simple analytics
- Performance reporting
- Alert management
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import structlog

from .performance_optimizer import PerformanceOptimizer, PerformanceMetrics
from .advanced_access_controls import AdvancedAccessController
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.tools.unified_tool_repository import UnifiedToolRepository
from app.communication.agent_communication_system import AgentCommunicationSystem

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MetricType(str, Enum):
    """Types of metrics collected."""
    PERFORMANCE = "performance"
    USAGE = "usage"


@dataclass
class Alert:
    """Simple system alert structure."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    created_at: datetime

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        severity: AlertSeverity = AlertSeverity.INFO
    ) -> "Alert":
        """Create a new alert."""
        return cls(
            alert_id=f"alert_{datetime.now().timestamp()}",
            title=title,
            description=description,
            severity=severity,
            created_at=datetime.now()
        )


@dataclass
class PerformanceReport:
    """Simple performance report."""
    report_id: str
    title: str
    system_health_score: float  # 0-100
    avg_response_time: float
    total_operations: int
    generated_at: datetime

    @classmethod
    def create(
        cls,
        title: str,
        system_health_score: float,
        avg_response_time: float,
        total_operations: int
    ) -> "PerformanceReport":
        """Create a new performance report."""
        return cls(
            report_id=f"report_{datetime.now().timestamp()}",
            title=title,
            system_health_score=system_health_score,
            avg_response_time=avg_response_time,
            total_operations=total_operations,
            generated_at=datetime.now()
        )


class MonitoringSystem:
    """
    Monitoring System - Streamlined Foundation.

    Provides simple monitoring and alerting for
    the multi-agent architecture.
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
        self.is_initialized = False

        # Simple tracking
        self.alerts: List[Alert] = []
        self.performance_reports: List[PerformanceReport] = []

        # Simple stats
        self.stats = {
            "total_metrics_collected": 0,
            "active_alerts": 0,
            "system_uptime": 0
        }

        self.start_time = datetime.now()

        logger.info("Monitoring system created")
    async def initialize(self) -> None:
        """Initialize the monitoring system."""
        try:
            if self.is_initialized:
                return

            self.is_initialized = True
            logger.info("Monitoring system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {str(e)}")
            raise
    async def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect basic system metrics."""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get metrics from performance optimizer
            metrics = await self.performance_optimizer.collect_metrics()

            # Check for alerts
            if metrics.cpu_usage > 80.0:
                alert = Alert.create(
                    title="High CPU Usage",
                    description=f"CPU usage is {metrics.cpu_usage:.1f}%",
                    severity=AlertSeverity.WARNING
                )
                self.alerts.append(alert)
                self.stats["active_alerts"] += 1

            if metrics.memory_usage > 85.0:
                alert = Alert.create(
                    title="High Memory Usage",
                    description=f"Memory usage is {metrics.memory_usage:.1f}%",
                    severity=AlertSeverity.ERROR
                )
                self.alerts.append(alert)
                self.stats["active_alerts"] += 1

            # Keep only recent alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]

            self.stats["total_metrics_collected"] += 1

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return PerformanceMetrics.create(0.0, 0.0, 0.0)
    async def generate_performance_report(self) -> PerformanceReport:
        """Generate a simple performance report."""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Collect current metrics
            metrics = await self.collect_system_metrics()

            # Calculate simple health score
            health_score = (100 - metrics.cpu_usage + 100 - metrics.memory_usage) / 2
            health_score = max(0, min(100, health_score))

            # Create report
            report = PerformanceReport.create(
                title="System Performance Report",
                system_health_score=health_score,
                avg_response_time=metrics.avg_response_time,
                total_operations=self.stats["total_metrics_collected"]
            )

            self.performance_reports.append(report)

            # Keep only recent reports
            if len(self.performance_reports) > 50:
                self.performance_reports = self.performance_reports[-50:]

            logger.info(f"Generated performance report with health score: {health_score:.1f}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate performance report: {str(e)}")
            raise
    def get_alerts(self) -> List[Alert]:
        """Get all alerts."""
        return self.alerts.copy()

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return [alert for alert in self.alerts if alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR]]

    def get_performance_reports(self) -> List[PerformanceReport]:
        """Get all performance reports."""
        return self.performance_reports.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "system_uptime": (datetime.now() - self.start_time).total_seconds(),
            "alerts_count": len(self.alerts),
            "reports_count": len(self.performance_reports)
        }
