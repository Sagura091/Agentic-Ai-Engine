"""
Performance Optimization Module for Multi-Agent Architecture.

This module provides comprehensive performance optimization capabilities
including system monitoring, resource optimization, caching strategies,
and intelligent load balancing.

Features:
- Real-time performance monitoring
- Intelligent resource optimization
- Advanced caching strategies
- Load balancing and scaling
- Performance analytics and reporting
- Automated optimization recommendations
"""

from .performance_optimizer import (
    PerformanceOptimizer,
    PerformanceMetrics,
    OptimizationStrategy,
    ResourceUsage
)

from .advanced_access_controls import (
    AdvancedAccessController,
    AccessRule,
    SecurityLevel,
    AccessAudit
)

from .monitoring_analytics import (
    MonitoringSystem,
    PerformanceReport
)

__all__ = [
    # Performance optimization
    "PerformanceOptimizer",
    "PerformanceMetrics",
    "OptimizationStrategy",
    "ResourceUsage",

    # Access controls
    "AdvancedAccessController",
    "AccessRule",
    "SecurityLevel",
    "AccessAudit",

    # Monitoring and analytics
    "MonitoringSystem",
    "PerformanceReport"
]

__version__ = "1.0.0"
__author__ = "Performance Team"
__description__ = "Comprehensive performance optimization and monitoring system"
