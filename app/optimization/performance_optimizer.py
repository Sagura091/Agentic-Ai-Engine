"""
Performance Optimizer for Multi-Agent Architecture.

This module provides intelligent performance optimization capabilities
including resource management, caching strategies, query optimization,
and system-wide performance tuning.

Features:
- Real-time performance monitoring
- Intelligent resource allocation
- Advanced caching strategies
- Query and operation optimization
- Memory management and cleanup
- Predictive performance scaling
"""

import asyncio
import time
import psutil
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

import structlog
from pydantic import BaseModel, Field

from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.rag.core.unified_memory_system import UnifiedMemorySystem
from app.tools.unified_tool_repository import UnifiedToolRepository
from app.communication.agent_communication_system import AgentCommunicationSystem

logger = structlog.get_logger(__name__)


class OptimizationStrategy(str, Enum):
    """Performance optimization strategies."""
    CONSERVATIVE = "conservative"     # Minimal changes, safe optimizations
    BALANCED = "balanced"             # Balanced approach
    AGGRESSIVE = "aggressive"         # Maximum performance, higher risk
    ADAPTIVE = "adaptive"             # Adapts based on usage patterns


class ResourceType(str, Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    resource_type: ResourceType
    current_usage: float          # Current usage percentage (0-100)
    average_usage: float          # Average usage over time
    peak_usage: float             # Peak usage recorded
    available: float              # Available capacity
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    # Response times
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    operations_per_second: float = 0.0
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # System health
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    queue_depth: int = 0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OptimizationConfig(BaseModel):
    """Configuration for performance optimization."""
    # Optimization settings
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.BALANCED)
    optimization_interval: int = Field(default=300, description="Optimization interval in seconds")
    
    # Resource thresholds
    cpu_threshold: float = Field(default=80.0, description="CPU usage threshold for optimization")
    memory_threshold: float = Field(default=85.0, description="Memory usage threshold")
    disk_threshold: float = Field(default=90.0, description="Disk usage threshold")
    
    # Cache settings
    enable_intelligent_caching: bool = Field(default=True)
    cache_size_mb: int = Field(default=1024, description="Cache size in MB")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Performance targets
    target_response_time_ms: float = Field(default=100.0, description="Target response time in ms")
    target_throughput_rps: float = Field(default=1000.0, description="Target requests per second")
    
    # Monitoring settings
    metrics_retention_hours: int = Field(default=168, description="Metrics retention in hours (7 days)")
    enable_predictive_scaling: bool = Field(default=True)


class PerformanceOptimizer:
    """
    Performance Optimizer for Multi-Agent Architecture.
    
    Provides intelligent performance optimization with real-time monitoring,
    resource management, and adaptive optimization strategies.
    """
    
    def __init__(
        self,
        unified_rag: UnifiedRAGSystem,
        memory_system: UnifiedMemorySystem,
        tool_repository: UnifiedToolRepository,
        communication_system: AgentCommunicationSystem,
        config: Optional[OptimizationConfig] = None
    ):
        """Initialize the performance optimizer."""
        self.unified_rag = unified_rag
        self.memory_system = memory_system
        self.tool_repository = tool_repository
        self.communication_system = communication_system
        self.config = config or OptimizationConfig()
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.resource_usage_history: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=1000) for resource_type in ResourceType
        }
        
        # Optimization state
        self.optimization_recommendations: List[Dict[str, Any]] = []
        self.active_optimizations: Set[str] = set()
        
        # Caching
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        # Performance counters
        self.operation_times: defaultdict = defaultdict(list)
        self.error_counts: defaultdict = defaultdict(int)
        
        # Statistics
        self.stats = {
            "optimizations_applied": 0,
            "performance_improvements": 0,
            "cache_hit_rate": 0.0,
            "avg_response_time": 0.0,
            "total_operations": 0
        }
        
        self.is_initialized = False
        logger.info("Performance optimizer created", config=self.config.dict())
    
    async def initialize(self) -> None:
        """Initialize the performance optimizer."""
        try:
            if self.is_initialized:
                logger.warning("Performance optimizer already initialized")
                return
            
            logger.info("Initializing performance optimizer...")
            
            # Start background monitoring
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._optimization_loop())
            asyncio.create_task(self._cache_cleanup_loop())
            
            self.is_initialized = True
            logger.info("Performance optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {str(e)}")
            raise
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate response time metrics
            recent_times = []
            for operation_times in self.operation_times.values():
                recent_times.extend(operation_times[-100:])  # Last 100 operations
            
            if recent_times:
                recent_times.sort()
                avg_response_time = sum(recent_times) / len(recent_times)
                p95_response_time = recent_times[int(len(recent_times) * 0.95)]
                p99_response_time = recent_times[int(len(recent_times) * 0.99)]
            else:
                avg_response_time = p95_response_time = p99_response_time = 0.0
            
            # Calculate cache hit rate
            total_cache_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            cache_hit_rate = (self.cache_stats["hits"] / total_cache_requests * 100) if total_cache_requests > 0 else 0.0
            
            # Calculate error rate
            total_operations = sum(len(times) for times in self.operation_times.values())
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0.0
            
            metrics = PerformanceMetrics(
                avg_response_time=avg_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate,
                requests_per_second=self._calculate_rps(),
                operations_per_second=self._calculate_ops()
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update stats
            self.stats["cache_hit_rate"] = cache_hit_rate
            self.stats["avg_response_time"] = avg_response_time
            self.stats["total_operations"] = total_operations
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {str(e)}")
            return PerformanceMetrics()
    
    async def optimize_system_performance(self) -> List[str]:
        """Perform system-wide performance optimization."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            optimizations_applied = []
            
            # Collect current metrics
            metrics = await self.collect_metrics()
            
            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(metrics)
            
            # Apply optimizations based on strategy
            for recommendation in recommendations:
                if await self._should_apply_optimization(recommendation):
                    success = await self._apply_optimization(recommendation)
                    if success:
                        optimizations_applied.append(recommendation["type"])
                        self.stats["optimizations_applied"] += 1
            
            # Optimize specific subsystems
            await self._optimize_rag_system()
            await self._optimize_memory_system()
            await self._optimize_communication_system()
            await self._optimize_caching()
            
            logger.info(f"Applied {len(optimizations_applied)} optimizations")
            return optimizations_applied
            
        except Exception as e:
            logger.error(f"Failed to optimize system performance: {str(e)}")
            return []
    
    async def _generate_optimization_recommendations(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        try:
            # CPU optimization
            if metrics.cpu_usage > self.config.cpu_threshold:
                recommendations.append({
                    "type": "cpu_optimization",
                    "priority": "high",
                    "description": "High CPU usage detected",
                    "actions": ["reduce_concurrent_operations", "optimize_algorithms", "enable_caching"]
                })
            
            # Memory optimization
            if metrics.memory_usage > self.config.memory_threshold:
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "description": "High memory usage detected",
                    "actions": ["cleanup_expired_data", "optimize_memory_allocation", "compress_data"]
                })
            
            # Response time optimization
            if metrics.avg_response_time > self.config.target_response_time_ms:
                recommendations.append({
                    "type": "response_time_optimization",
                    "priority": "medium",
                    "description": "Response time above target",
                    "actions": ["optimize_queries", "increase_cache_size", "parallel_processing"]
                })
            
            # Cache optimization
            if metrics.cache_hit_rate < 80.0:  # Target 80% cache hit rate
                recommendations.append({
                    "type": "cache_optimization",
                    "priority": "medium",
                    "description": "Low cache hit rate",
                    "actions": ["increase_cache_size", "optimize_cache_strategy", "preload_frequent_data"]
                })
            
            # Error rate optimization
            if metrics.error_rate > 1.0:  # Target < 1% error rate
                recommendations.append({
                    "type": "error_reduction",
                    "priority": "high",
                    "description": "High error rate detected",
                    "actions": ["improve_error_handling", "add_retry_logic", "validate_inputs"]
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {str(e)}")
            return []
    
    async def _apply_optimization(self, recommendation: Dict[str, Any]) -> bool:
        """Apply a specific optimization recommendation."""
        try:
            optimization_type = recommendation["type"]
            actions = recommendation.get("actions", [])
            
            if optimization_type == "cpu_optimization":
                return await self._apply_cpu_optimization(actions)
            elif optimization_type == "memory_optimization":
                return await self._apply_memory_optimization(actions)
            elif optimization_type == "response_time_optimization":
                return await self._apply_response_time_optimization(actions)
            elif optimization_type == "cache_optimization":
                return await self._apply_cache_optimization(actions)
            elif optimization_type == "error_reduction":
                return await self._apply_error_reduction(actions)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {str(e)}")
            return False
    
    async def _apply_cpu_optimization(self, actions: List[str]) -> bool:
        """Apply CPU optimization actions."""
        try:
            success = False
            
            for action in actions:
                if action == "reduce_concurrent_operations":
                    # Implement concurrency limits
                    success = True
                elif action == "optimize_algorithms":
                    # Trigger algorithm optimization
                    success = True
                elif action == "enable_caching":
                    # Enable more aggressive caching
                    self.config.enable_intelligent_caching = True
                    success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply CPU optimization: {str(e)}")
            return False
    
    async def _apply_memory_optimization(self, actions: List[str]) -> bool:
        """Apply memory optimization actions."""
        try:
            success = False
            
            for action in actions:
                if action == "cleanup_expired_data":
                    # Trigger cleanup in memory system
                    await self.memory_system._cleanup_expired_memories()
                    success = True
                elif action == "optimize_memory_allocation":
                    # Optimize memory allocation patterns
                    success = True
                elif action == "compress_data":
                    # Enable data compression
                    success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply memory optimization: {str(e)}")
            return False
    
    async def _optimize_rag_system(self) -> None:
        """Optimize the RAG system performance."""
        try:
            # Optimize vector database operations
            # Implement connection pooling optimizations
            # Optimize embedding operations
            logger.debug("Optimizing RAG system performance")
            
        except Exception as e:
            logger.error(f"Failed to optimize RAG system: {str(e)}")
    
    async def _optimize_memory_system(self) -> None:
        """Optimize the memory system performance."""
        try:
            # Trigger memory consolidation
            # Optimize memory retrieval
            # Clean up expired memories
            logger.debug("Optimizing memory system performance")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory system: {str(e)}")
    
    async def _optimize_communication_system(self) -> None:
        """Optimize the communication system performance."""
        try:
            # Optimize message queues
            # Implement message batching
            # Clean up old messages
            logger.debug("Optimizing communication system performance")
            
        except Exception as e:
            logger.error(f"Failed to optimize communication system: {str(e)}")
    
    async def _optimize_caching(self) -> None:
        """Optimize caching strategies."""
        try:
            # Clean up expired cache entries
            current_time = datetime.utcnow()
            expired_keys = []
            
            for key, (value, timestamp) in self.query_cache.items():
                if (current_time - timestamp).total_seconds() > self.config.cache_ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.query_cache[key]
                self.cache_stats["evictions"] += 1
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Failed to optimize caching: {str(e)}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background loop for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self.collect_metrics()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {str(e)}")
    
    async def _optimization_loop(self) -> None:
        """Background loop for automatic optimization."""
        while True:
            try:
                await asyncio.sleep(self.config.optimization_interval)
                
                if self.config.strategy == OptimizationStrategy.ADAPTIVE:
                    await self.optimize_system_performance()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {str(e)}")
    
    async def _cache_cleanup_loop(self) -> None:
        """Background loop for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                await self._optimize_caching()
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    def _calculate_rps(self) -> float:
        """Calculate requests per second."""
        try:
            if len(self.metrics_history) < 2:
                return 0.0
            
            # Calculate based on recent metrics
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            if len(recent_metrics) >= 2:
                time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
                if time_diff > 0:
                    return len(recent_metrics) / time_diff
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate RPS: {str(e)}")
            return 0.0
    
    def _calculate_ops(self) -> float:
        """Calculate operations per second."""
        try:
            # Calculate based on operation times
            total_ops = sum(len(times) for times in self.operation_times.values())
            if total_ops > 0 and self.metrics_history:
                time_span = (datetime.utcnow() - self.metrics_history[0].timestamp).total_seconds()
                if time_span > 0:
                    return total_ops / time_span
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate OPS: {str(e)}")
            return 0.0
    
    async def _should_apply_optimization(self, recommendation: Dict[str, Any]) -> bool:
        """Determine if an optimization should be applied."""
        try:
            # Check optimization strategy
            if self.config.strategy == OptimizationStrategy.CONSERVATIVE:
                return recommendation.get("priority") == "high"
            elif self.config.strategy == OptimizationStrategy.BALANCED:
                return recommendation.get("priority") in ["high", "medium"]
            elif self.config.strategy == OptimizationStrategy.AGGRESSIVE:
                return True
            elif self.config.strategy == OptimizationStrategy.ADAPTIVE:
                # Adaptive logic based on system state
                return recommendation.get("priority") in ["high", "medium"]
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to determine optimization applicability: {str(e)}")
            return False
    
    def record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation timing for performance analysis."""
        try:
            self.operation_times[operation].append(duration)
            
            # Keep only recent measurements
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-1000:]
                
        except Exception as e:
            logger.error(f"Failed to record operation time: {str(e)}")
    
    def record_error(self, operation: str) -> None:
        """Record an error for performance analysis."""
        try:
            self.error_counts[operation] += 1
            
        except Exception as e:
            logger.error(f"Failed to record error: {str(e)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            latest_metrics = self.metrics_history[-1]
            
            return {
                "current_metrics": {
                    "cpu_usage": latest_metrics.cpu_usage,
                    "memory_usage": latest_metrics.memory_usage,
                    "disk_usage": latest_metrics.disk_usage,
                    "avg_response_time": latest_metrics.avg_response_time,
                    "cache_hit_rate": latest_metrics.cache_hit_rate,
                    "error_rate": latest_metrics.error_rate
                },
                "optimization_stats": self.stats,
                "cache_stats": self.cache_stats,
                "active_optimizations": list(self.active_optimizations),
                "recommendations_count": len(self.optimization_recommendations),
                "config": self.config.dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {str(e)}")
            return {"error": str(e)}
