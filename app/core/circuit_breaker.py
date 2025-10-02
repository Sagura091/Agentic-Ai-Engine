"""
Advanced Circuit Breaker System for External Services.

This module provides production-grade circuit breaker implementation
for protecting the system from cascading failures when external
services are unavailable or slow.
"""

import asyncio
from typing import Optional, Callable, Any, Dict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    # Failure threshold
    failure_threshold: int = 5  # Number of failures before opening
    failure_timeout: int = 60   # Seconds to wait before half-open
    
    # Success threshold for half-open state
    success_threshold: int = 2  # Successful calls to close circuit
    
    # Timeout settings
    call_timeout: float = 30.0  # Timeout for individual calls
    
    # Monitoring window
    monitoring_window: int = 60  # Seconds to track failures
    
    # Recovery settings
    exponential_backoff: bool = True
    max_backoff_time: int = 300  # Max seconds for backoff


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    
    state_changes: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    def update_response_time(self, response_time: float):
        """Update response time metrics."""
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
        
        # Calculate running average
        total_successful = self.successful_calls
        if total_successful > 0:
            self.average_response_time = (
                (self.average_response_time * (total_successful - 1) + response_time) / total_successful
            )


class CircuitBreaker:
    """
    Advanced circuit breaker for external service calls.
    
    Implements the circuit breaker pattern to prevent cascading failures
    and provide graceful degradation when external services fail.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        
        self._lock = asyncio.Lock()
        self._failure_times: list[datetime] = []
        
        logger.info(f"Circuit breaker '{name}' initialized", state=self.state.value)
    
    async def call(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments for func
        
        Returns:
            Result from func or fallback
        
        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback provided
        """
        async with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.metrics.state_changes += 1
                    self.metrics.last_state_change = datetime.utcnow()
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    self.metrics.rejected_calls += 1
                    logger.warning(f"Circuit breaker '{self.name}' is OPEN, rejecting call")
                    
                    if fallback:
                        return await fallback(*args, **kwargs)
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        # Execute the call
        start_time = datetime.utcnow()
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.call_timeout
            )
            
            # Record success
            response_time = (datetime.utcnow() - start_time).total_seconds()
            await self._record_success(response_time)
            
            return result
            
        except asyncio.TimeoutError as e:
            logger.error(f"Circuit breaker '{self.name}' call timeout", timeout=self.config.call_timeout)
            await self._record_failure()
            
            if fallback:
                return await fallback(*args, **kwargs)
            raise
            
        except Exception as e:
            logger.error(f"Circuit breaker '{self.name}' call failed", error=str(e))
            await self._record_failure()
            
            if fallback:
                return await fallback(*args, **kwargs)
            raise
    
    async def _record_success(self, response_time: float):
        """Record successful call."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.utcnow()
            self.metrics.update_response_time(response_time)
            
            # If in HALF_OPEN state, check if we should close
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.metrics.state_changes += 1
                    self.metrics.last_state_change = datetime.utcnow()
                    logger.info(f"Circuit breaker '{self.name}' CLOSED after successful recovery")
    
    async def _record_failure(self):
        """Record failed call."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = datetime.utcnow()
            
            # Track failure times for monitoring window
            self._failure_times.append(datetime.utcnow())
            self._cleanup_old_failures()
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                if len(self._failure_times) >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.metrics.state_changes += 1
                    self.metrics.last_state_change = datetime.utcnow()
                    logger.warning(
                        f"Circuit breaker '{self.name}' OPENED",
                        failures=len(self._failure_times),
                        threshold=self.config.failure_threshold
                    )
            
            # If in HALF_OPEN state, reopen immediately
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.metrics.state_changes += 1
                self.metrics.last_state_change = datetime.utcnow()
                logger.warning(f"Circuit breaker '{self.name}' reopened after failed recovery attempt")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if not self.metrics.last_failure_time:
            return True
        
        # Calculate backoff time
        backoff_time = self.config.failure_timeout
        if self.config.exponential_backoff:
            # Exponential backoff based on state changes
            backoff_time = min(
                self.config.failure_timeout * (2 ** min(self.metrics.state_changes, 5)),
                self.config.max_backoff_time
            )
        
        time_since_failure = (datetime.utcnow() - self.metrics.last_failure_time).total_seconds()
        return time_since_failure >= backoff_time
    
    def _cleanup_old_failures(self):
        """Remove failures outside monitoring window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.monitoring_window)
        self._failure_times = [t for t in self._failure_times if t > cutoff_time]
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "rejected_calls": self.metrics.rejected_calls,
            "success_rate": (
                self.metrics.successful_calls / self.metrics.total_calls
                if self.metrics.total_calls > 0 else 0.0
            ),
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            "last_state_change": self.metrics.last_state_change.isoformat() if self.metrics.last_state_change else None,
            "state_changes": self.metrics.state_changes,
            "average_response_time": self.metrics.average_response_time,
            "min_response_time": self.metrics.min_response_time if self.metrics.min_response_time != float('inf') else 0.0,
            "max_response_time": self.metrics.max_response_time,
        }
    
    async def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
            self._failure_times.clear()
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    async def call(
        self,
        name: str,
        func: Callable,
        *args,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        breaker = await self.get_breaker(name, config)
        return await breaker.call(func, *args, fallback=fallback, **kwargs)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()

