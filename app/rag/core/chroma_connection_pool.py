"""
ChromaDB Connection Pool for Revolutionary RAG System.

This module provides connection pooling and performance optimizations
for ChromaDB to handle high-throughput RAG operations efficiently.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager
import threading
from queue import Queue, Empty
import weakref

import structlog
import chromadb
from chromadb.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for connection pool monitoring."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    peak_connections: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class PooledConnection:
    """A pooled ChromaDB connection with metadata."""
    client: chromadb.Client
    created_at: datetime
    last_used: datetime
    request_count: int = 0
    is_active: bool = False
    connection_id: str = ""
    
    def __post_init__(self):
        if not self.connection_id:
            self.connection_id = f"conn_{id(self.client)}"


class ChromaConnectionPool:
    """
    High-performance connection pool for ChromaDB.
    
    Features:
    - Connection pooling with configurable min/max connections
    - Automatic connection health monitoring
    - Request load balancing
    - Performance metrics and monitoring
    - Connection lifecycle management
    - Thread-safe operations
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: int = 30,
        idle_timeout: int = 300,  # 5 minutes
        health_check_interval: int = 60,  # 1 minute
        enable_telemetry: bool = False
    ):
        """Initialize the connection pool."""
        self.persist_directory = persist_directory
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval
        self.enable_telemetry = enable_telemetry
        
        # Connection management
        self._connections: List[PooledConnection] = []
        self._available_connections: Queue = Queue()
        self._lock = threading.RLock()
        self._stats = ConnectionStats()
        self._is_initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self._request_times: List[float] = []
        self._max_request_history = 1000
        
        logger.info(
            "ChromaDB connection pool initialized",
            min_connections=min_connections,
            max_connections=max_connections,
            persist_directory=persist_directory
        )
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._is_initialized:
            return
        
        try:
            # Create minimum connections
            for _ in range(self.min_connections):
                await self._create_connection()
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self._is_initialized = True
            self._stats.created_at = datetime.utcnow()
            
            logger.info(
                "Connection pool initialized successfully",
                connections=len(self._connections),
                min_connections=self.min_connections
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new ChromaDB connection."""
        try:
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=self.enable_telemetry,
                    allow_reset=True
                )
            )
            
            connection = PooledConnection(
                client=client,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow()
            )
            
            with self._lock:
                self._connections.append(connection)
                self._available_connections.put(connection)
                self._stats.total_connections += 1
                self._stats.idle_connections += 1
                
                if self._stats.total_connections > self._stats.peak_connections:
                    self._stats.peak_connections = self._stats.total_connections
            
            logger.debug(f"Created new connection: {connection.connection_id}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection: {str(e)}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool (context manager)."""
        connection = None
        start_time = time.time()
        
        try:
            # Get connection from pool
            connection = await self._acquire_connection()
            
            # Update stats
            with self._lock:
                self._stats.total_requests += 1
                self._stats.active_connections += 1
                self._stats.idle_connections -= 1
            
            yield connection.client
            
            # Mark as successful
            with self._lock:
                self._stats.successful_requests += 1
                
        except Exception as e:
            # Mark as failed
            with self._lock:
                self._stats.failed_requests += 1
            logger.error(f"Connection error: {str(e)}")
            raise
            
        finally:
            # Return connection to pool
            if connection:
                await self._release_connection(connection)
                
                # Update response time stats
                response_time = time.time() - start_time
                self._update_response_time_stats(response_time)
                
                with self._lock:
                    self._stats.active_connections -= 1
                    self._stats.idle_connections += 1
    
    async def _acquire_connection(self) -> PooledConnection:
        """Acquire a connection from the pool."""
        timeout_time = time.time() + self.connection_timeout
        
        while time.time() < timeout_time:
            try:
                # Try to get available connection
                connection = self._available_connections.get_nowait()
                
                # Check if connection is still healthy
                if await self._is_connection_healthy(connection):
                    connection.last_used = datetime.utcnow()
                    connection.request_count += 1
                    connection.is_active = True
                    return connection
                else:
                    # Remove unhealthy connection
                    await self._remove_connection(connection)
                    
            except Empty:
                # No available connections, try to create new one
                with self._lock:
                    if len(self._connections) < self.max_connections:
                        return await self._create_connection()
                
                # Wait a bit and retry
                await asyncio.sleep(0.1)
        
        raise TimeoutError("Failed to acquire connection within timeout")
    
    async def _release_connection(self, connection: PooledConnection) -> None:
        """Release a connection back to the pool."""
        connection.is_active = False
        connection.last_used = datetime.utcnow()
        
        # Check if connection should be kept
        if await self._should_keep_connection(connection):
            self._available_connections.put(connection)
        else:
            await self._remove_connection(connection)
    
    async def _is_connection_healthy(self, connection: PooledConnection) -> bool:
        """Check if a connection is healthy."""
        try:
            # Simple health check - try to list collections
            connection.client.list_collections()
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {str(e)}")
            return False
    
    async def _should_keep_connection(self, connection: PooledConnection) -> bool:
        """Determine if a connection should be kept in the pool."""
        # Check idle timeout
        idle_time = datetime.utcnow() - connection.last_used
        if idle_time > timedelta(seconds=self.idle_timeout):
            return False
        
        # Keep if we're at minimum connections
        with self._lock:
            if len(self._connections) <= self.min_connections:
                return True
        
        return True
    
    async def _remove_connection(self, connection: PooledConnection) -> None:
        """Remove a connection from the pool."""
        try:
            with self._lock:
                if connection in self._connections:
                    self._connections.remove(connection)
                    self._stats.total_connections -= 1
                    
                    if not connection.is_active:
                        self._stats.idle_connections -= 1
            
            # Close the connection (ChromaDB doesn't have explicit close)
            del connection.client
            
            logger.debug(f"Removed connection: {connection.connection_id}")
            
        except Exception as e:
            logger.error(f"Error removing connection: {str(e)}")
    
    def _update_response_time_stats(self, response_time: float) -> None:
        """Update response time statistics."""
        self._request_times.append(response_time)
        
        # Keep only recent request times
        if len(self._request_times) > self._max_request_history:
            self._request_times = self._request_times[-self._max_request_history:]
        
        # Calculate average
        if self._request_times:
            self._stats.average_response_time = sum(self._request_times) / len(self._request_times)
    
    async def _health_check_loop(self) -> None:
        """Background task for connection health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all connections."""
        unhealthy_connections = []
        
        with self._lock:
            connections_to_check = self._connections.copy()
        
        for connection in connections_to_check:
            if not connection.is_active:  # Only check idle connections
                if not await self._is_connection_healthy(connection):
                    unhealthy_connections.append(connection)
        
        # Remove unhealthy connections
        for connection in unhealthy_connections:
            await self._remove_connection(connection)
        
        # Ensure minimum connections
        with self._lock:
            current_count = len(self._connections)
            
        if current_count < self.min_connections:
            for _ in range(self.min_connections - current_count):
                try:
                    await self._create_connection()
                except Exception as e:
                    logger.error(f"Failed to create replacement connection: {str(e)}")
    
    def get_stats(self) -> ConnectionStats:
        """Get current connection pool statistics."""
        with self._lock:
            return ConnectionStats(
                total_connections=self._stats.total_connections,
                active_connections=self._stats.active_connections,
                idle_connections=self._stats.idle_connections,
                total_requests=self._stats.total_requests,
                successful_requests=self._stats.successful_requests,
                failed_requests=self._stats.failed_requests,
                average_response_time=self._stats.average_response_time,
                peak_connections=self._stats.peak_connections,
                created_at=self._stats.created_at
            )
    
    async def shutdown(self) -> None:
        """Shutdown the connection pool."""
        logger.info("Shutting down connection pool...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        with self._lock:
            connections_to_close = self._connections.copy()
            self._connections.clear()
        
        for connection in connections_to_close:
            try:
                del connection.client
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
        
        logger.info("Connection pool shutdown complete")


# Global connection pool instance
_connection_pool: Optional[ChromaConnectionPool] = None


async def get_connection_pool(
    persist_directory: str = "./data/chroma",
    **kwargs
) -> ChromaConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = ChromaConnectionPool(
            persist_directory=persist_directory,
            **kwargs
        )
        await _connection_pool.initialize()
    
    return _connection_pool


async def shutdown_connection_pool() -> None:
    """Shutdown the global connection pool."""
    global _connection_pool
    
    if _connection_pool:
        await _connection_pool.shutdown()
        _connection_pool = None
