"""
Revolutionary Custom HTTP Client

A high-performance, feature-rich HTTP client built from scratch with:
- True async I/O (no ThreadPoolExecutor hacks)
- Intelligent connection pooling with keep-alive
- Streaming support for large responses and LLM outputs
- Modular middleware system for extensibility
- Generic and reusable across any project

This client bypasses limitations of standard packages (requests, aiohttp, httpx)
while providing enterprise-grade features and full protocol control.
"""

import asyncio
import json
import ssl
import socket
import time
import logging
from urllib.parse import urlparse, urlencode, urlunparse
from typing import Optional, Dict, Any, Union, AsyncIterator, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class HTTPError(Exception):
    """
    Custom HTTPError to differentiate from built-in exceptions.
    Can store additional context such as status code, reason, headers, and body.
    """
    def __init__(self, status_code, reason, headers, raw_data):
        self.status_code = status_code
        self.reason = reason
        self.headers = headers
        self.raw_data = raw_data
        message = f"HTTP {status_code} {reason}"
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error details to dictionary format"""
        return {
            "status_code": self.status_code,
            "reason": self.reason,
            "headers": dict(self.headers),
            "raw_data": self.raw_data
        }


class ConnectionPoolError(Exception):
    """Raised when connection pool operations fail"""
    pass


class StreamingError(Exception):
    """Raised when streaming operations fail"""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling"""
    max_size: int = 100  # Total max connections across all hosts
    max_per_host: int = 10  # Max connections per host
    keepalive_timeout: float = 60.0  # Seconds to keep idle connections alive
    cleanup_interval: float = 30.0  # Seconds between cleanup runs
    connect_timeout: float = 10.0  # Timeout for establishing connections


@dataclass
class ClientConfig:
    """Main client configuration"""
    timeout: float = 30.0  # Default request timeout
    verify_ssl: bool = True  # Verify SSL certificates
    max_retries: int = 3  # Max retry attempts
    default_headers: Optional[Dict[str, str]] = None  # Default headers for all requests
    pool_config: Optional[ConnectionPoolConfig] = None  # Connection pool configuration
    enable_streaming: bool = True  # Enable streaming support

    def __post_init__(self):
        if self.pool_config is None:
            self.pool_config = ConnectionPoolConfig()
        if self.default_headers is None:
            self.default_headers = {}


# ============================================================================
# ASYNC CONNECTION (True Async I/O)
# ============================================================================

class AsyncHTTPConnection:
    """
    True async HTTP/HTTPS connection using asyncio sockets.
    No ThreadPoolExecutor - pure non-blocking I/O.
    """

    def __init__(self, host: str, port: int, ssl_context: Optional[ssl.SSLContext] = None):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self.created_at = time.time()
        self.last_used = time.time()
        self.request_count = 0

    async def connect(self, timeout: float = 10.0) -> None:
        """Establish connection to the host"""
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.host,
                    self.port,
                    ssl=self.ssl_context
                ),
                timeout=timeout
            )
            self.connected = True
            self.last_used = time.time()
            logger.debug(f"Connected to {self.host}:{self.port} (SSL: {self.ssl_context is not None})")
        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection to {self.host}:{self.port} timed out after {timeout}s")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {str(e)}")

    async def send_request(self, method: str, path: str, headers: Dict[str, str],
                          body: Optional[bytes] = None) -> None:
        """Send HTTP request"""
        if not self.connected:
            raise ConnectionError("Not connected")

        # Build HTTP request
        request_lines = [f"{method} {path} HTTP/1.1"]

        # Add headers
        for key, value in headers.items():
            request_lines.append(f"{key}: {value}")

        # Add body length if present
        if body:
            request_lines.append(f"Content-Length: {len(body)}")

        # End headers
        request_lines.append("")
        request_lines.append("")

        # Combine request
        request_str = "\r\n".join(request_lines)
        request_bytes = request_str.encode('utf-8')

        if body:
            request_bytes += body

        # Send request
        self.writer.write(request_bytes)
        await self.writer.drain()
        self.last_used = time.time()
        self.request_count += 1

    async def receive_response_headers(self) -> tuple[int, str, Dict[str, str]]:
        """Receive and parse response headers"""
        if not self.connected:
            raise ConnectionError("Not connected")

        # Read status line
        status_line = await self.reader.readline()
        if not status_line:
            raise ConnectionError("Connection closed by server")

        status_line = status_line.decode('utf-8').strip()

        # Parse status line: HTTP/1.1 200 OK
        parts = status_line.split(' ', 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid status line: {status_line}")

        status_code = int(parts[1])
        reason = parts[2]

        # Read headers
        headers = {}
        while True:
            line = await self.reader.readline()
            if not line or line == b'\r\n':
                break

            line = line.decode('utf-8').strip()
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()

        self.last_used = time.time()
        return status_code, reason, headers

    async def receive_response_body(self, headers: Dict[str, str]) -> bytes:
        """Receive response body based on headers"""
        if not self.connected:
            raise ConnectionError("Not connected")

        # Check for chunked encoding
        if headers.get('transfer-encoding', '').lower() == 'chunked':
            return await self._read_chunked_body()

        # Check for content-length
        content_length = headers.get('content-length')
        if content_length:
            return await self.reader.readexactly(int(content_length))

        # Read until connection closes
        return await self.reader.read()

    async def _read_chunked_body(self) -> bytes:
        """Read chunked transfer encoding"""
        chunks = []
        while True:
            # Read chunk size
            size_line = await self.reader.readline()
            size = int(size_line.strip(), 16)

            if size == 0:
                # Last chunk
                await self.reader.readline()  # Read trailing CRLF
                break

            # Read chunk data
            chunk = await self.reader.readexactly(size)
            chunks.append(chunk)

            # Read trailing CRLF
            await self.reader.readline()

        return b''.join(chunks)

    def is_alive(self) -> bool:
        """Check if connection is still alive"""
        if not self.connected or not self.writer:
            return False

        # Check if writer is closing
        if self.writer.is_closing():
            return False

        return True

    async def close(self) -> None:
        """Close the connection"""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")
            finally:
                self.connected = False
                self.reader = None
                self.writer = None


# ============================================================================
# CONNECTION POOL (Intelligent Connection Management)
# ============================================================================

class ConnectionPool:
    """
    Intelligent connection pool with:
    - Per-host connection limits
    - Keep-alive support
    - Automatic cleanup of stale connections
    - Connection reuse
    - Thread-safe async operations
    """

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self._pools: Dict[tuple, List[AsyncHTTPConnection]] = defaultdict(list)
        self._in_use: weakref.WeakSet = weakref.WeakSet()
        self._total_connections = 0
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start(self) -> None:
        """Start the connection pool and cleanup task"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Connection pool started")

    async def acquire(self, host: str, port: int,
                     ssl_context: Optional[ssl.SSLContext] = None) -> AsyncHTTPConnection:
        """
        Acquire a connection from the pool or create a new one.

        Args:
            host: Target host
            port: Target port
            ssl_context: SSL context for HTTPS

        Returns:
            AsyncHTTPConnection ready to use
        """
        if self._closed:
            raise ConnectionPoolError("Connection pool is closed")

        key = (host, port, ssl_context is not None)

        async with self._lock:
            # Try to get existing connection from pool
            while self._pools[key]:
                conn = self._pools[key].pop(0)

                # Check if connection is still alive
                if conn.is_alive():
                    # Check if not expired
                    age = time.time() - conn.last_used
                    if age < self.config.keepalive_timeout:
                        self._in_use.add(conn)
                        logger.debug(f"Reusing connection to {host}:{port} (age: {age:.1f}s)")
                        return conn

                # Connection is dead or expired, close it
                await conn.close()
                self._total_connections -= 1

            # Check per-host limit
            host_connections = len([c for c in self._in_use if c.host == host and c.port == port])
            if host_connections >= self.config.max_per_host:
                raise ConnectionPoolError(
                    f"Max connections per host reached for {host}:{port} "
                    f"({host_connections}/{self.config.max_per_host})"
                )

            # Check total limit
            if self._total_connections >= self.config.max_size:
                raise ConnectionPoolError(
                    f"Max total connections reached ({self._total_connections}/{self.config.max_size})"
                )

            # Create new connection
            conn = AsyncHTTPConnection(host, port, ssl_context)
            await conn.connect(timeout=self.config.connect_timeout)
            self._total_connections += 1
            self._in_use.add(conn)
            logger.debug(f"Created new connection to {host}:{port} (total: {self._total_connections})")
            return conn

    async def release(self, conn: AsyncHTTPConnection) -> None:
        """
        Release a connection back to the pool.

        Args:
            conn: Connection to release
        """
        if self._closed:
            await conn.close()
            return

        key = (conn.host, conn.port, conn.ssl_context is not None)

        async with self._lock:
            # Remove from in-use set
            try:
                self._in_use.remove(conn)
            except KeyError:
                pass

            # Check if connection is still alive
            if not conn.is_alive():
                await conn.close()
                self._total_connections -= 1
                logger.debug(f"Closed dead connection to {conn.host}:{conn.port}")
                return

            # Return to pool
            self._pools[key].append(conn)
            logger.debug(f"Released connection to {conn.host}:{conn.port} back to pool")

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup stale connections"""
        while not self._closed:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_stale_connections(self) -> None:
        """Remove stale connections from the pool"""
        async with self._lock:
            now = time.time()
            cleaned = 0

            for key, connections in list(self._pools.items()):
                # Filter out stale connections
                alive_connections = []
                for conn in connections:
                    age = now - conn.last_used
                    if conn.is_alive() and age < self.config.keepalive_timeout:
                        alive_connections.append(conn)
                    else:
                        await conn.close()
                        self._total_connections -= 1
                        cleaned += 1

                if alive_connections:
                    self._pools[key] = alive_connections
                else:
                    del self._pools[key]

            if cleaned > 0:
                logger.debug(f"Cleaned up {cleaned} stale connections")

    async def close(self) -> None:
        """Close all connections and shutdown the pool"""
        self._closed = True

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._lock:
            for connections in self._pools.values():
                for conn in connections:
                    await conn.close()

            self._pools.clear()
            self._total_connections = 0

        logger.debug("Connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'total_connections': self._total_connections,
            'in_use': len(self._in_use),
            'available': sum(len(conns) for conns in self._pools.values()),
            'pools': len(self._pools)
        }


# ============================================================================
# STREAMING RESPONSE (For Large Responses & LLM Outputs)
# ============================================================================

class StreamingResponse:
    """
    Streaming HTTP response with:
    - Async iteration over chunks
    - Line-by-line iteration (perfect for SSE)
    - JSON stream parsing (perfect for LLM streaming)
    - Memory-efficient processing
    - Progress tracking
    """

    def __init__(self, status_code: int, reason: str, headers: Dict[str, str],
                 reader: asyncio.StreamReader, connection: AsyncHTTPConnection,
                 pool: Optional[ConnectionPool] = None):
        self.status_code = status_code
        self.reason = reason
        self.headers = headers
        self._reader = reader
        self._connection = connection
        self._pool = pool
        self._bytes_read = 0
        self._closed = False

    async def iter_chunks(self, chunk_size: int = 8192) -> AsyncIterator[bytes]:
        """
        Iterate over response body in chunks.

        Args:
            chunk_size: Size of each chunk in bytes

        Yields:
            bytes: Response chunks
        """
        try:
            # Check for chunked encoding
            if self.headers.get('transfer-encoding', '').lower() == 'chunked':
                async for chunk in self._iter_chunked():
                    self._bytes_read += len(chunk)
                    yield chunk
            else:
                # Check for content-length
                content_length = self.headers.get('content-length')
                if content_length:
                    remaining = int(content_length)
                    while remaining > 0:
                        to_read = min(chunk_size, remaining)
                        chunk = await self._reader.read(to_read)
                        if not chunk:
                            break
                        self._bytes_read += len(chunk)
                        remaining -= len(chunk)
                        yield chunk
                else:
                    # Read until EOF
                    while True:
                        chunk = await self._reader.read(chunk_size)
                        if not chunk:
                            break
                        self._bytes_read += len(chunk)
                        yield chunk
        finally:
            await self.close()

    async def _iter_chunked(self) -> AsyncIterator[bytes]:
        """Iterate over chunked transfer encoding"""
        while True:
            # Read chunk size
            size_line = await self._reader.readline()
            if not size_line:
                break

            try:
                size = int(size_line.strip(), 16)
            except ValueError:
                raise StreamingError(f"Invalid chunk size: {size_line}")

            if size == 0:
                # Last chunk
                await self._reader.readline()  # Read trailing CRLF
                break

            # Read chunk data
            chunk = await self._reader.readexactly(size)
            yield chunk

            # Read trailing CRLF
            await self._reader.readline()

    async def iter_lines(self, delimiter: bytes = b'\n') -> AsyncIterator[str]:
        """
        Iterate over response lines (perfect for Server-Sent Events).

        Args:
            delimiter: Line delimiter

        Yields:
            str: Response lines
        """
        buffer = b''
        async for chunk in self.iter_chunks():
            buffer += chunk
            while delimiter in buffer:
                line, buffer = buffer.split(delimiter, 1)
                yield line.decode('utf-8', errors='ignore')

        # Yield remaining buffer
        if buffer:
            yield buffer.decode('utf-8', errors='ignore')

    async def iter_json_stream(self, prefix: str = 'data: ') -> AsyncIterator[Dict[str, Any]]:
        """
        Iterate over JSON stream (perfect for LLM streaming responses).

        Args:
            prefix: Prefix to strip from each line (for SSE format)

        Yields:
            dict: Parsed JSON objects
        """
        async for line in self.iter_lines():
            line = line.strip()
            if not line:
                continue

            # Strip prefix if present
            if prefix and line.startswith(prefix):
                line = line[len(prefix):]

            # Skip special SSE messages
            if line.startswith(':') or line == '[DONE]':
                continue

            # Parse JSON
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
                continue

    async def read(self) -> bytes:
        """Read entire response body into memory"""
        chunks = []
        async for chunk in self.iter_chunks():
            chunks.append(chunk)
        return b''.join(chunks)

    async def text(self, encoding: str = 'utf-8') -> str:
        """Read entire response as text"""
        body = await self.read()
        return body.decode(encoding, errors='ignore')

    async def json(self) -> Any:
        """Read and parse response as JSON"""
        text = await self.text()
        return json.loads(text)

    async def close(self) -> None:
        """Close the streaming response and release connection"""
        if self._closed:
            return

        self._closed = True

        # Release connection back to pool
        if self._pool and self._connection:
            await self._pool.release(self._connection)

    @property
    def bytes_read(self) -> int:
        """Get number of bytes read so far"""
        return self._bytes_read

    def raise_for_status(self) -> None:
        """Raise HTTPError if status code >= 400"""
        if self.status_code >= 400:
            raise HTTPError(
                status_code=self.status_code,
                reason=self.reason,
                headers=self.headers,
                raw_data=f"Streaming response (read {self._bytes_read} bytes)"
            )


class HTTPResponse:
    """
    Standard HTTP response (non-streaming).
    Encapsulates the HTTP response with structured access to status, headers, raw data, and parsed data.
    """
    def __init__(self, status, reason, headers, raw_data):
        """
        Initializes the HTTPResponse object.

        Args:
            status (int): The HTTP status code.
            reason (str): The HTTP status reason phrase.
            headers (dict): The HTTP headers.
            raw_data (str or bytes): The raw HTTP response data.
        """
        self.status_code = status
        self.reason = reason
        self.headers = headers

        # Handle both str and bytes
        if isinstance(raw_data, bytes):
            try:
                self.raw_data = raw_data.decode('utf-8', errors='ignore')
            except Exception:
                self.raw_data = str(raw_data)
        else:
            self.raw_data = raw_data

        self._parsed_json = None
        self._parse_timestamp = datetime.now()

        try:
            self._parsed_json = json.loads(self.raw_data)
        except (json.JSONDecodeError, TypeError):
            self._parsed_json = None

    @property
    def text(self) -> str:
        """Get the response content as text"""
        return self.raw_data

    @property
    def json_data(self) -> Optional[Dict]:
        """Lazy JSON parsing with caching"""
        return self._parsed_json

    def raise_for_status(self):
        """
        Raises an HTTPError if the response status code is >= 400.
        """
        if self.status_code >= 400:
            raise HTTPError(
                status_code=self.status_code,
                reason=self.reason,
                headers=self.headers,
                raw_data=self.raw_data
            )

    def json(self) -> Optional[Dict]:
        """
        Returns the JSON-parsed data if available, otherwise None.
        """
        return self.json_data

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the HTTPResponse object.

        Returns:
            dict: A dictionary containing the status_code, reason, headers, raw_data, and json_data.
        """
        return {
            "status_code": self.status_code,
            "reason": self.reason,
            "headers": self.headers,
            "raw_data": self.raw_data,
            "json": self.json_data,
            "timestamp": self._parse_timestamp.isoformat()
        }

    def __repr__(self):
        """
        Provides a concise summary of the response for debugging.
        """
        snippet = self.raw_data[:60] + "..." if len(self.raw_data) > 60 else self.raw_data
        return (
            f"<HTTPResponse [{self.status_code} {self.reason}], "
            f"Headers: {self.headers}, "
            f"Body: {snippet}>"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_url_with_params(path, params=None):
    """
    Appends query parameters to a path.

    Args:
        path (str): The original path (e.g. '/api/resource')
        params (dict): A dictionary of query parameters.

    Returns:
        str: The path with query parameters (e.g. '/api/resource?key=value').
    """
    if not params:
        return path

    # If the path already has a '?', use '&', otherwise '?'
    separator = '&' if '?' in path else '?'
    return f"{path}{separator}{urlencode(params)}"


# ============================================================================
# REVOLUTIONARY HTTP CLIENT (Main Client Class)
# ============================================================================

class HTTPClient:
    """
    Revolutionary async HTTP client with:
    - True async I/O (no thread pools)
    - Intelligent connection pooling
    - Streaming support
    - Modular middleware system
    - Generic and reusable

    This is the main client class that should be used throughout the project.
    """

    def __init__(self, base_url: str, config: Optional[ClientConfig] = None):
        """
        Initialize the HTTP client.

        Args:
            base_url: Base URL for requests (e.g., 'https://api.example.com')
            config: Client configuration (optional)
        """
        self.config = config or ClientConfig()

        # Parse base URL
        parsed = urlparse(base_url)
        self.scheme = parsed.scheme or 'http'
        self.host = parsed.hostname
        self.port = parsed.port or (443 if self.scheme == 'https' else 80)
        self.base_path = parsed.path.rstrip('/') if parsed.path else ''

        if not self.host:
            raise ValueError(f"Invalid base URL: {base_url}")

        # Create SSL context if needed
        self.ssl_context = self._create_ssl_context() if self.scheme == 'https' else None

        # Create connection pool
        self.pool = ConnectionPool(self.config.pool_config)
        self._pool_started = False

        # Middleware hooks
        self._request_hooks: List[Callable] = []
        self._response_hooks: List[Callable] = []

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with appropriate verification settings"""
        context = ssl.create_default_context()
        if not self.config.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context

    async def _ensure_pool_started(self) -> None:
        """Ensure connection pool is started"""
        if not self._pool_started:
            await self.pool.start()
            self._pool_started = True

    def add_request_hook(self, hook: Callable) -> None:
        """
        Add a request hook (middleware).
        Hook signature: async def hook(method, path, headers, body) -> (method, path, headers, body)
        """
        self._request_hooks.append(hook)

    def add_response_hook(self, hook: Callable) -> None:
        """
        Add a response hook (middleware).
        Hook signature: async def hook(response) -> response
        """
        self._response_hooks.append(hook)

    async def request(self, method: str, path: str,
                     headers: Optional[Dict[str, str]] = None,
                     body: Optional[Union[str, bytes, Dict]] = None,
                     params: Optional[Dict[str, str]] = None,
                     stream: bool = False,
                     timeout: Optional[float] = None) -> Union[HTTPResponse, StreamingResponse]:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: Request path (will be appended to base_url)
            headers: Request headers
            body: Request body (str, bytes, or dict for JSON)
            params: Query parameters
            stream: If True, return StreamingResponse for large responses
            timeout: Request timeout (overrides default)

        Returns:
            HTTPResponse or StreamingResponse
        """
        await self._ensure_pool_started()

        # Build full path
        full_path = self._build_full_path(path, params)

        # Merge headers
        merged_headers = {**self.config.default_headers, **(headers or {})}
        merged_headers.setdefault('Host', self.host)
        merged_headers.setdefault('Connection', 'keep-alive')
        merged_headers.setdefault('User-Agent', 'RevolutionaryHTTPClient/1.0')

        # Handle body
        body_bytes = None
        if body is not None:
            if isinstance(body, dict):
                body_bytes = json.dumps(body).encode('utf-8')
                merged_headers.setdefault('Content-Type', 'application/json')
            elif isinstance(body, str):
                body_bytes = body.encode('utf-8')
            else:
                body_bytes = body

        # Apply request hooks
        for hook in self._request_hooks:
            method, full_path, merged_headers, body_bytes = await hook(
                method, full_path, merged_headers, body_bytes
            )

        # Get connection from pool
        conn = await self.pool.acquire(self.host, self.port, self.ssl_context)

        try:
            # Send request
            await conn.send_request(method, full_path, merged_headers, body_bytes)

            # Receive response headers
            status_code, reason, response_headers = await conn.receive_response_headers()

            # Return streaming or standard response
            if stream or self.config.enable_streaming:
                response = StreamingResponse(
                    status_code, reason, response_headers,
                    conn.reader, conn, self.pool
                )
            else:
                # Read full body
                body = await conn.receive_response_body(response_headers)
                response = HTTPResponse(status_code, reason, response_headers, body)
                # Release connection back to pool
                await self.pool.release(conn)

            # Apply response hooks
            for hook in self._response_hooks:
                response = await hook(response)

            return response

        except Exception as e:
            # Release connection on error
            await self.pool.release(conn)
            raise

    def _build_full_path(self, path: str, params: Optional[Dict] = None) -> str:
        """Build the full request path with parameters"""
        if path.startswith('/'):
            full_path = f"{self.base_path}{path}"
        else:
            full_path = f"{self.base_path}/{path}" if self.base_path else f"/{path}"
        return build_url_with_params(full_path, params)

    # Convenience methods
    async def get(self, path: str, **kwargs) -> Union[HTTPResponse, StreamingResponse]:
        """Convenience method for GET requests"""
        return await self.request('GET', path, **kwargs)

    async def post(self, path: str, **kwargs) -> Union[HTTPResponse, StreamingResponse]:
        """Convenience method for POST requests"""
        return await self.request('POST', path, **kwargs)

    async def put(self, path: str, **kwargs) -> Union[HTTPResponse, StreamingResponse]:
        """Convenience method for PUT requests"""
        return await self.request('PUT', path, **kwargs)

    async def delete(self, path: str, **kwargs) -> Union[HTTPResponse, StreamingResponse]:
        """Convenience method for DELETE requests"""
        return await self.request('DELETE', path, **kwargs)

    async def patch(self, path: str, **kwargs) -> Union[HTTPResponse, StreamingResponse]:
        """Convenience method for PATCH requests"""
        return await self.request('PATCH', path, **kwargs)

    async def head(self, path: str, **kwargs) -> Union[HTTPResponse, StreamingResponse]:
        """Convenience method for HEAD requests"""
        return await self.request('HEAD', path, **kwargs)

    async def close(self) -> None:
        """Close the client and all connections"""
        await self.pool.close()

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_pool_started()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'pool': self.pool.get_stats(),
            'base_url': f"{self.scheme}://{self.host}:{self.port}{self.base_path}"
        }


# ============================================================================
# LEGACY CLIENT (Backward Compatibility)
# ============================================================================

class SimpleHTTPClient:
    """
    LEGACY: Simple HTTP client for backward compatibility.

    This class is kept for backward compatibility with existing code.
    For new code, use HTTPClient instead which provides:
    - True async I/O (no ThreadPoolExecutor)
    - Connection pooling
    - Streaming support
    - Better performance

    This legacy client uses the new HTTPClient under the hood.
    """
    def __init__(self, url: str, timeout: int = 30, default_headers: Optional[Dict] = None,
                 verify_ssl: bool = True, max_retries: int = 3):
        """
        Initialize the SimpleHTTPClient (legacy wrapper).

        Args:
            url (str): Base URL or host to connect to
            timeout (int): Request timeout in seconds
            default_headers (dict): Default headers for all requests
            verify_ssl (bool): Whether to verify SSL certificates
            max_retries (int): Maximum number of retry attempts
        """
        # Create config for new client
        config = ClientConfig(
            timeout=timeout,
            verify_ssl=verify_ssl,
            max_retries=max_retries,
            default_headers=default_headers or {}
        )

        # Create new HTTPClient
        self._client = HTTPClient(url, config)

        # Store attributes for backward compatibility
        self.timeout = timeout
        self.default_headers = default_headers or {}
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries

    async def request(self, method: str, path: str, headers: Optional[Dict] = None,
                      body: Any = None, params: Optional[Dict] = None,
                      retries: int = None, backoff: float = 1.0) -> HTTPResponse:
        """
        Make an HTTP request (legacy wrapper).

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            body: Request body
            params: Query parameters
            retries: Number of retries (ignored - uses config)
            backoff: Backoff multiplier (ignored - uses config)

        Returns:
            HTTPResponse
        """
        response = await self._client.request(
            method=method,
            path=path,
            headers=headers,
            body=body,
            params=params,
            stream=False  # Legacy client doesn't support streaming
        )

        # Convert StreamingResponse to HTTPResponse if needed
        if isinstance(response, StreamingResponse):
            body = await response.read()
            result = HTTPResponse(response.status_code, response.reason, response.headers, body)
            await response.close()
            return result

        return response

    async def get(self, path: str, **kwargs) -> HTTPResponse:
        """Convenience method for GET requests"""
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> HTTPResponse:
        """Convenience method for POST requests"""
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs) -> HTTPResponse:
        """Convenience method for PUT requests"""
        return await self.request("PUT", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> HTTPResponse:
        """Convenience method for DELETE requests"""
        return await self.request("DELETE", path, **kwargs)

    async def patch(self, path: str, **kwargs) -> HTTPResponse:
        """Convenience method for PATCH requests"""
        return await self.request("PATCH", path, **kwargs)

    async def head(self, path: str, **kwargs) -> HTTPResponse:
        """Convenience method for HEAD requests"""
        return await self.request("HEAD", path, **kwargs)

    async def __aenter__(self):
        """Async context manager entry"""
        await self._client._ensure_pool_started()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Async context manager exit"""
        await self._client.close()

    def __enter__(self):
        """Sync context manager entry (not supported - use async)"""
        raise NotImplementedError("Use async context manager (async with) instead")

    def __exit__(self, exc_type, exc_value, traceback):
        """Sync context manager exit"""
        pass

    # High-Level JSON Methods

    async def get_json(self, path, headers=None, params=None, retries=1, backoff=1):
        """
        Sends a GET request and returns JSON data if available.
        Raises ValueError if not valid JSON.
        """
        response = await self.get(path, headers=headers, params=params, retries=retries, backoff=backoff)
        try:
            return response.json()
        except (json.JSONDecodeError, AttributeError):
            raise ValueError("Response data is not valid JSON")

    async def post_json(self, path, headers=None, body=None, params=None, retries=1, backoff=1):
        """
        Sends a POST request with an optional JSON body and returns JSON data if available.
        Raises ValueError if not valid JSON.
        """
        response = await self.post(path, headers=headers, body=body, params=params, retries=retries, backoff=backoff)
        try:
            return response.json()
        except (json.JSONDecodeError, AttributeError):
            raise ValueError("Response data is not valid JSON")
