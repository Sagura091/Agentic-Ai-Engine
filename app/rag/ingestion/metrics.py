"""
Metrics collection and observability for RAG ingestion pipeline.

This module provides comprehensive metrics collection including:
- Counters (jobs, duplicates, errors)
- Timers (stage latencies with percentiles)
- Gauges (queue depths, active workers)
- Histograms (processing times)
- Prometheus-compatible output
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
import asyncio
import time

import structlog

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Counter:
    """Counter metric."""
    name: str
    value: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    
    def inc(self, amount: int = 1):
        """Increment counter."""
        self.value += amount
    
    def reset(self):
        """Reset counter."""
        self.value = 0


@dataclass
class Gauge:
    """Gauge metric."""
    name: str
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    
    def set(self, value: float):
        """Set gauge value."""
        self.value = value
    
    def inc(self, amount: float = 1.0):
        """Increment gauge."""
        self.value += amount
    
    def dec(self, amount: float = 1.0):
        """Decrement gauge."""
        self.value -= amount


@dataclass
class Histogram:
    """Histogram metric with percentile calculation."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=10000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def observe(self, value: float):
        """Observe a value."""
        self.values.append(value)
    
    def get_percentile(self, percentile: float) -> float:
        """
        Get percentile value.
        
        Args:
            percentile: Percentile (0.0-1.0)
            
        Returns:
            Percentile value
        """
        if not self.values:
            return 0.0
        
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * percentile)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        if not self.values:
            return {
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        sorted_values = sorted(self.values)
        count = len(sorted_values)
        total = sum(sorted_values)
        
        return {
            "count": count,
            "sum": total,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": total / count,
            "p50": self.get_percentile(0.50),
            "p95": self.get_percentile(0.95),
            "p99": self.get_percentile(0.99)
        }


class Timer:
    """Timer context manager for measuring durations."""
    
    def __init__(self, histogram: Histogram):
        """
        Initialize timer.
        
        Args:
            histogram: Histogram to record duration
        """
        self.histogram = histogram
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record duration."""
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration * 1000)  # Convert to milliseconds


class MetricsCollector:
    """
    Comprehensive metrics collector for RAG ingestion pipeline.
    
    Provides:
    - Counters for events (jobs, errors, duplicates)
    - Gauges for current state (queue depth, workers)
    - Histograms for distributions (latencies)
    - Timers for stage measurements
    - Prometheus-compatible output
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = asyncio.Lock()
        
        # Initialize standard metrics
        self._init_standard_metrics()
        
        logger.info("MetricsCollector initialized")
    
    def _init_standard_metrics(self):
        """Initialize standard metrics."""
        # Counters
        self._register_counter("ingest_jobs_total", {"status": "pending"})
        self._register_counter("ingest_jobs_total", {"status": "processing"})
        self._register_counter("ingest_jobs_total", {"status": "completed"})
        self._register_counter("ingest_jobs_total", {"status": "failed"})
        
        self._register_counter("duplicates_total", {"type": "exact"})
        self._register_counter("duplicates_total", {"type": "fuzzy"})
        
        self._register_counter("chunks_total", {"status": "created"})
        self._register_counter("chunks_total", {"status": "skipped"})
        self._register_counter("chunks_total", {"status": "updated"})
        
        self._register_counter("dlq_total", {"reason": "validation_error"})
        self._register_counter("dlq_total", {"reason": "processing_error"})
        self._register_counter("dlq_total", {"reason": "timeout"})
        
        self._register_counter("cache_requests_total", {"result": "hit"})
        self._register_counter("cache_requests_total", {"result": "miss"})
        
        # Gauges
        self._register_gauge("queue_depth", {})
        self._register_gauge("active_workers", {})
        self._register_gauge("cache_size", {})
        self._register_gauge("cache_hit_rate", {})
        
        # Histograms for stage timings
        self._register_histogram("stage_duration_ms", {"stage": "intake"})
        self._register_histogram("stage_duration_ms", {"stage": "validation"})
        self._register_histogram("stage_duration_ms", {"stage": "extraction"})
        self._register_histogram("stage_duration_ms", {"stage": "chunking"})
        self._register_histogram("stage_duration_ms", {"stage": "deduplication"})
        self._register_histogram("stage_duration_ms", {"stage": "embedding"})
        self._register_histogram("stage_duration_ms", {"stage": "indexing"})
        
        # Histograms for sizes
        self._register_histogram("document_size_bytes", {})
        self._register_histogram("chunk_size_chars", {})
    
    def _register_counter(self, name: str, labels: Dict[str, str]):
        """Register a counter."""
        key = self._make_key(name, labels)
        if key not in self._counters:
            self._counters[key] = Counter(name=name, labels=labels)
    
    def _register_gauge(self, name: str, labels: Dict[str, str]):
        """Register a gauge."""
        key = self._make_key(name, labels)
        if key not in self._gauges:
            self._gauges[key] = Gauge(name=name, labels=labels)
    
    def _register_histogram(self, name: str, labels: Dict[str, str]):
        """Register a histogram."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = Histogram(name=name, labels=labels)
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Make metric key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def inc_counter(self, name: str, labels: Optional[Dict[str, str]] = None, amount: int = 1):
        """
        Increment counter.
        
        Args:
            name: Counter name
            labels: Labels
            amount: Amount to increment
        """
        labels = labels or {}
        key = self._make_key(name, labels)
        
        if key not in self._counters:
            self._register_counter(name, labels)
        
        self._counters[key].inc(amount)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set gauge value.
        
        Args:
            name: Gauge name
            value: Value to set
            labels: Labels
        """
        labels = labels or {}
        key = self._make_key(name, labels)
        
        if key not in self._gauges:
            self._register_gauge(name, labels)
        
        self._gauges[key].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Observe histogram value.
        
        Args:
            name: Histogram name
            value: Value to observe
            labels: Labels
        """
        labels = labels or {}
        key = self._make_key(name, labels)
        
        if key not in self._histograms:
            self._register_histogram(name, labels)
        
        self._histograms[key].observe(value)
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> Timer:
        """
        Create timer context manager.
        
        Args:
            name: Histogram name for timing
            labels: Labels
            
        Returns:
            Timer context manager
        """
        labels = labels or {}
        key = self._make_key(name, labels)
        
        if key not in self._histograms:
            self._register_histogram(name, labels)
        
        return Timer(self._histograms[key])
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Metrics dictionary
        """
        metrics = {
            "counters": {},
            "gauges": {},
            "histograms": {}
        }
        
        # Collect counters
        for key, counter in self._counters.items():
            metrics["counters"][key] = {
                "value": counter.value,
                "labels": counter.labels
            }
        
        # Collect gauges
        for key, gauge in self._gauges.items():
            metrics["gauges"][key] = {
                "value": gauge.value,
                "labels": gauge.labels
            }
        
        # Collect histograms
        for key, histogram in self._histograms.items():
            metrics["histograms"][key] = {
                "stats": histogram.get_stats(),
                "labels": histogram.labels
            }
        
        return metrics
    
    def get_prometheus_format(self) -> str:
        """
        Get metrics in Prometheus text format.
        
        Returns:
            Prometheus-formatted metrics
        """
        lines = []
        
        # Counters
        for key, counter in self._counters.items():
            lines.append(f"# TYPE {counter.name} counter")
            label_str = ",".join(f'{k}="{v}"' for k, v in counter.labels.items())
            metric_line = f"{counter.name}"
            if label_str:
                metric_line += f"{{{label_str}}}"
            metric_line += f" {counter.value}"
            lines.append(metric_line)
        
        # Gauges
        for key, gauge in self._gauges.items():
            lines.append(f"# TYPE {gauge.name} gauge")
            label_str = ",".join(f'{k}="{v}"' for k, v in gauge.labels.items())
            metric_line = f"{gauge.name}"
            if label_str:
                metric_line += f"{{{label_str}}}"
            metric_line += f" {gauge.value}"
            lines.append(metric_line)
        
        # Histograms
        for key, histogram in self._histograms.items():
            lines.append(f"# TYPE {histogram.name} histogram")
            stats = histogram.get_stats()
            label_str = ",".join(f'{k}="{v}"' for k, v in histogram.labels.items())
            
            for stat_name, stat_value in stats.items():
                metric_line = f"{histogram.name}_{stat_name}"
                if label_str:
                    metric_line += f"{{{label_str}}}"
                metric_line += f" {stat_value}"
                lines.append(metric_line)
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics."""
        for counter in self._counters.values():
            counter.reset()
        
        for gauge in self._gauges.values():
            gauge.set(0.0)
        
        for histogram in self._histograms.values():
            histogram.values.clear()
        
        logger.info("All metrics reset")


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None
_metrics_lock = asyncio.Lock()


async def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance.
    
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        async with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    
    return _metrics_collector

