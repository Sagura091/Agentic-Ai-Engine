"""
Reliability primitives for robust async operations.

This module provides production-grade patterns for:
- Timeout enforcement
- Retry with exponential backoff
- Circuit breaker pattern
- Resource limit management
"""

import asyncio
import time
import random
from typing import TypeVar, Callable, Any, Optional, Awaitable
from functools import wraps
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: float = 60.0  # Time to wait before half-open
    
    
@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.utcnow)
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by failing fast when a service is down.
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered, allow limited requests
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name for logging
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
        logger.info(
            "CircuitBreaker initialized",
            name=name,
            failure_threshold=self.config.failure_threshold,
            timeout_seconds=self.config.timeout_seconds
        )
    
    async def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.stats.state = CircuitState.OPEN
        self.stats.last_state_change = datetime.utcnow()
        logger.warning(
            "CircuitBreaker opened",
            name=self.name,
            failure_count=self.stats.failure_count
        )
    
    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.stats.state = CircuitState.HALF_OPEN
        self.stats.success_count = 0
        self.stats.failure_count = 0
        self.stats.last_state_change = datetime.utcnow()
        logger.info("CircuitBreaker half-opened", name=self.name)
    
    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.stats.state = CircuitState.CLOSED
        self.stats.success_count = 0
        self.stats.failure_count = 0
        self.stats.last_state_change = datetime.utcnow()
        logger.info("CircuitBreaker closed", name=self.name)
    
    async def _check_state(self) -> None:
        """Check if state should transition."""
        async with self._lock:
            if self.stats.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if self.stats.last_state_change:
                    elapsed = (datetime.utcnow() - self.stats.last_state_change).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        await self._transition_to_half_open()
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        await self._check_state()
        
        self.stats.total_calls += 1
        
        # Check if circuit is open
        if self.stats.state == CircuitState.OPEN:
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Record success
            async with self._lock:
                self.stats.success_count += 1
                self.stats.total_successes += 1
                
                # Transition from HALF_OPEN to CLOSED if enough successes
                if self.stats.state == CircuitState.HALF_OPEN:
                    if self.stats.success_count >= self.config.success_threshold:
                        await self._transition_to_closed()
            
            return result
            
        except Exception as e:
            # Record failure
            async with self._lock:
                self.stats.failure_count += 1
                self.stats.total_failures += 1
                self.stats.last_failure_time = datetime.utcnow()
                
                # Transition to OPEN if too many failures
                if self.stats.state == CircuitState.CLOSED:
                    if self.stats.failure_count >= self.config.failure_threshold:
                        await self._transition_to_open()
                elif self.stats.state == CircuitState.HALF_OPEN:
                    # Any failure in HALF_OPEN goes back to OPEN
                    await self._transition_to_open()
            
            raise
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self.stats


async def with_timeout(coro: Awaitable[T], timeout_seconds: float, operation_name: str = "operation") -> T:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        operation_name: Name for logging
        
    Returns:
        Coroutine result
        
    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        logger.error(
            "Operation timed out",
            operation=operation_name,
            timeout_seconds=timeout_seconds
        )
        raise


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff and jitter.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delay
        retriable_exceptions: Tuple of exceptions to retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(
                            "Retry succeeded",
                            function=func.__name__,
                            attempt=attempt
                        )
                    
                    return result
                    
                except retriable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            "All retry attempts failed",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(e)
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    
                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        "Retry attempt failed, retrying",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_seconds=delay,
                        error=str(e)
                    )
                    
                    await asyncio.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class ResourceLimiter:
    """
    Resource limiter using semaphore.
    
    Limits concurrent operations to prevent resource exhaustion.
    """
    
    def __init__(self, max_concurrent: int, name: str = "limiter"):
        """
        Initialize resource limiter.
        
        Args:
            max_concurrent: Maximum concurrent operations
            name: Limiter name for logging
        """
        self.max_concurrent = max_concurrent
        self.name = name
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._total_count = 0
        
        logger.info(
            "ResourceLimiter initialized",
            name=name,
            max_concurrent=max_concurrent
        )
    
    async def __aenter__(self):
        """Acquire resource."""
        await self._semaphore.acquire()
        self._active_count += 1
        self._total_count += 1
        
        logger.debug(
            "Resource acquired",
            name=self.name,
            active=self._active_count,
            max=self.max_concurrent
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release resource."""
        self._semaphore.release()
        self._active_count -= 1
        
        logger.debug(
            "Resource released",
            name=self.name,
            active=self._active_count
        )
    
    def get_stats(self) -> dict:
        """Get limiter statistics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "total_count": self._total_count
        }


async def run_with_semaphore(
    coro: Awaitable[T],
    semaphore: asyncio.Semaphore,
    operation_name: str = "operation"
) -> T:
    """
    Run coroutine with semaphore limiting.
    
    Args:
        coro: Coroutine to run
        semaphore: Semaphore for limiting
        operation_name: Name for logging
        
    Returns:
        Coroutine result
    """
    async with semaphore:
        logger.debug("Semaphore acquired", operation=operation_name)
        try:
            result = await coro
            return result
        finally:
            logger.debug("Semaphore released", operation=operation_name)

