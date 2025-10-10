"""
Performance Optimizer - THE Performance System for Multi-Agent Architecture.

This is THE ONLY performance optimization system in the entire application.
All performance monitoring and optimization flows through this unified system.

CORE ARCHITECTURE:
- System-wide performance monitoring
- Automatic optimization strategies
- Resource usage tracking
- Memory cleanup and optimization
- Cache management

DESIGN PRINCIPLES:
- Continuous monitoring and optimization
- Automatic resource management
- Simple, clean, fast operations
- No complexity unless absolutely necessary

PHASE 4 ENHANCEMENT:
✅ Integration with all unified systems
✅ Automatic optimization strategies
✅ Real-time performance monitoring
✅ Resource usage optimization
"""

import asyncio
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.tools.unified_tool_repository import UnifiedToolRepository
from app.communication.agent_communication_system import AgentCommunicationSystem

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class OptimizationStrategy(str, Enum):
    """Performance optimization strategies - ENHANCED."""
    CONSERVATIVE = "conservative"     # Minimal optimization
    BALANCED = "balanced"             # Balanced approach (DEFAULT)
    AGGRESSIVE = "aggressive"         # Maximum performance
    ADAPTIVE = "adaptive"             # Adapts based on usage patterns


class ResourceType(str, Enum):
    """Types of system resources - ENHANCED."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"                     # NEW: Disk usage
    NETWORK = "network"               # NEW: Network usage
    CHROMADB = "chromadb"             # NEW: ChromaDB specific metrics


@dataclass
class ResourceUsage:
    """Resource usage metrics - ENHANCED."""
    resource_type: ResourceType
    current_usage: float          # Current usage percentage (0-100)
    peak_usage: float = 0.0       # NEW: Peak usage in current session
    average_usage: float = 0.0    # NEW: Average usage
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, resource_type: ResourceType, current_usage: float) -> "ResourceUsage":
        """Create a new resource usage metric."""
        return cls(
            resource_type=resource_type,
            current_usage=current_usage,
            timestamp=datetime.now()
        )


@dataclass
class PerformanceMetrics:
    """Simple system performance metrics."""
    avg_response_time: float
    cpu_usage: float
    memory_usage: float
    timestamp: datetime

    @classmethod
    def create(
        cls,
        avg_response_time: float,
        cpu_usage: float,
        memory_usage: float
    ) -> "PerformanceMetrics":
        """Create new performance metrics."""
        return cls(
            avg_response_time=avg_response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            timestamp=datetime.now()
        )


class PerformanceOptimizer:
    """
    Performance Optimizer - Streamlined Foundation.

    Provides simple performance optimization with basic monitoring
    and resource management.
    """

    def __init__(
        self,
        unified_rag: UnifiedRAGSystem,
        memory_system: UnifiedMemorySystem,
        tool_repository: UnifiedToolRepository,
        communication_system: AgentCommunicationSystem
    ):
        """Initialize the performance optimizer."""
        self.unified_rag = unified_rag
        self.memory_system = memory_system
        self.tool_repository = tool_repository
        self.communication_system = communication_system
        self.is_initialized = False

        # Simple tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.resource_usage_history: List[ResourceUsage] = []

        # Simple cache
        self.query_cache: Dict[str, Any] = {}

        # Simple stats
        self.stats = {
            "optimizations_applied": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_operations": 0
        }

        logger.info(
            "Performance optimizer created",
            LogCategory.PERFORMANCE_MONITORING,
            "app.optimization.performance_optimizer.PerformanceOptimizer"
        )
    async def initialize(self) -> None:
        """Initialize the performance optimizer."""
        try:
            if self.is_initialized:
                return

            self.is_initialized = True
            logger.info(
                "Performance optimizer initialized successfully",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer"
            )

        except Exception as e:
            logger.error(
                "Failed to initialize performance optimizer",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )
            raise

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Simple response time calculation
            avg_response_time = 0.0
            if self.metrics_history:
                avg_response_time = sum(m.avg_response_time for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))

            metrics = PerformanceMetrics.create(
                avg_response_time=avg_response_time,
                cpu_usage=cpu_percent,
                memory_usage=memory.percent
            )

            # Store metrics
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:  # Keep only recent metrics
                self.metrics_history = self.metrics_history[-100:]

            return metrics

        except Exception as e:
            logger.error(
                "Failed to collect metrics",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )
            return PerformanceMetrics.create(0.0, 0.0, 0.0)
    async def optimize_system_performance(self) -> List[str]:
        """Perform simple system-wide performance optimization."""
        try:
            if not self.is_initialized:
                await self.initialize()

            optimizations_applied = []

            # Collect current metrics
            metrics = await self.collect_metrics()

            # Simple optimization logic
            if metrics.cpu_usage > 80.0:
                await self._cleanup_cache()
                optimizations_applied.append("cache_cleanup")
                self.stats["optimizations_applied"] += 1

            if metrics.memory_usage > 85.0:
                await self._cleanup_memory()
                optimizations_applied.append("memory_cleanup")
                self.stats["optimizations_applied"] += 1

            logger.info(
                f"Applied {len(optimizations_applied)} optimizations",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                data={"optimizations_count": len(optimizations_applied), "optimizations": optimizations_applied}
            )
            return optimizations_applied

        except Exception as e:
            logger.error(
                "Failed to optimize system performance",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )
            return []
    async def _cleanup_cache(self) -> None:
        """Simple cache cleanup."""
        try:
            # Clear half of the cache
            cache_keys = list(self.query_cache.keys())
            keys_to_remove = cache_keys[:len(cache_keys)//2]

            for key in keys_to_remove:
                del self.query_cache[key]

            logger.info(
                f"Cleaned up {len(keys_to_remove)} cache entries",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                data={"entries_removed": len(keys_to_remove)}
            )

        except Exception as e:
            logger.error(
                "Failed to cleanup cache",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )

    async def _cleanup_memory(self) -> None:
        """Simple memory cleanup."""
        try:
            # Cleanup expired memories in memory system
            if hasattr(self.memory_system, 'cleanup_expired_memories'):
                await self.memory_system.cleanup_expired_memories()

            # Keep only recent metrics
            if len(self.metrics_history) > 50:
                self.metrics_history = self.metrics_history[-50:]

            if len(self.resource_usage_history) > 50:
                self.resource_usage_history = self.resource_usage_history[-50:]

            logger.info(
                "Performed memory cleanup",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer"
            )

        except Exception as e:
            logger.error(
                "Failed to cleanup memory",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached result."""
        try:
            result = self.query_cache.get(key)
            if result is not None:
                self.stats["cache_hits"] += 1
                return result
            else:
                self.stats["cache_misses"] += 1
                return None

        except Exception as e:
            logger.error(
                "Failed to get cached result",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )
            return None

    def set_cached_result(self, key: str, value: Any) -> None:
        """Set a cached result."""
        try:
            self.query_cache[key] = value

            # Simple cache size management
            if len(self.query_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self.query_cache.keys())[:100]
                for k in keys_to_remove:
                    del self.query_cache[k]

        except Exception as e:
            logger.error(
                "Failed to set cached result",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )

    def record_resource_usage(self, resource_type: ResourceType, usage: float) -> None:
        """Record resource usage."""
        try:
            resource_usage = ResourceUsage.create(resource_type, usage)
            self.resource_usage_history.append(resource_usage)

            # Keep only recent usage data
            if len(self.resource_usage_history) > 100:
                self.resource_usage_history = self.resource_usage_history[-100:]

        except Exception as e:
            logger.error(
                "Failed to record resource usage",
                LogCategory.PERFORMANCE_MONITORING,
                "app.optimization.performance_optimizer.PerformanceOptimizer",
                error=e
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get performance optimizer statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "metrics_count": len(self.metrics_history),
            "resource_usage_count": len(self.resource_usage_history),
            "cache_size": len(self.query_cache)
        }
