"""
Main HTTP Client for the Backend

This module provides the primary HTTP client used throughout the backend
for all HTTP requests, including Ollama connections. It bypasses proxy
issues and provides reliable connectivity with full async support.
"""

import asyncio
import http.client
import json
from urllib.parse import urlparse, urlencode
import time
import ssl
import socket
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

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

class HTTPResponse:
    """
    Encapsulates the HTTP response with structured access to status, headers, raw data, and parsed data.
    """
    def __init__(self, status, reason, headers, raw_data):
        """
        Initializes the HTTPResponse object.

        Args:
            status (int): The HTTP status code.
            reason (str): The HTTP status reason phrase.
            headers (dict): The HTTP headers.
            raw_data (str): The raw HTTP response data.
        """
        self.status_code = status
        self.reason = reason
        self.headers = headers
        self.raw_data = raw_data
        self._parsed_json = None
        self._parse_timestamp = datetime.now()

        try:
            self._parsed_json = json.loads(raw_data)
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


class SimpleHTTPClient:
    """
    A simple async HTTP/HTTPS client with convenience methods for GET, POST, PUT, DELETE, PATCH, HEAD,
    handling JSON bodies, timeouts, default headers, and optional context management.
    """
    def __init__(self, url: str, timeout: int = 30, default_headers: Optional[Dict] = None,
                 verify_ssl: bool = True, max_retries: int = 3):
        """
        Initialize the SimpleHTTPClient with enhanced SSL and retry options.

        Args:
            url (str): Base URL or host to connect to
            timeout (int): Request timeout in seconds
            default_headers (dict): Default headers for all requests
            verify_ssl (bool): Whether to verify SSL certificates
            max_retries (int): Maximum number of retry attempts
        """
        parsed_url = urlparse(url) if "://" in url else None

        if parsed_url:
            self.scheme = parsed_url.scheme or "http"
            self.host = parsed_url.hostname
            self.port = parsed_url.port
            self.base_path = parsed_url.path.rstrip("/") if parsed_url.path else ""
        else:
            self.scheme = "http"
            self.host = url
            self.port = None
            self.base_path = ""

        if not self.host:
            raise ValueError("Invalid URL or host provided.")

        self.timeout = timeout
        self.default_headers = default_headers or {}
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self._connection = None
        self._ssl_context = self._create_ssl_context() if self.scheme == "https" else None
        self._last_request_time = None
        self._request_count = 0
        self._executor = ThreadPoolExecutor(max_workers=10)

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with appropriate verification settings"""
        context = ssl.create_default_context()
        if not self.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context

    def _get_connection(self) -> Union[http.client.HTTPConnection, http.client.HTTPSConnection]:
        """
        Returns an HTTPConnection or HTTPSConnection with enhanced error handling.
        """
        try:
            if self.scheme == "https":
                return http.client.HTTPSConnection(
                    self.host,
                    self.port,
                    timeout=self.timeout,
                    context=self._ssl_context
                )
            return http.client.HTTPConnection(
                self.host,
                self.port,
                timeout=self.timeout
            )
        except Exception as e:
            logger.error(f"Failed to create connection: {str(e)}")
            raise ConnectionError(f"Failed to create connection to {self.host}: {str(e)}")

    async def request(self, method: str, path: str, headers: Optional[Dict] = None,
                      body: Any = None, params: Optional[Dict] = None,
                      retries: int = None, backoff: float = 1.0) -> HTTPResponse:
        """
        Enhanced async request method with better error handling and retry logic.

        Args:
            method (str): HTTP method
            path (str): Request path
            headers (dict): Request headers
            body (Any): Request body
            params (dict): Query parameters
            retries (int): Number of retries (overrides max_retries)
            backoff (float): Backoff multiplier between retries

        Returns:
            HTTPResponse: Response object

        Raises:
            HTTPError: For HTTP-level errors
            ConnectionError: For network-level errors
        """
        retries = retries if retries is not None else self.max_retries
        attempt = 0
        last_error = None

        while attempt <= retries:
            try:
                return await self._single_request(method, path, headers, body, params)
            except (socket.timeout, ConnectionError) as e:
                last_error = e
                attempt += 1
                if attempt <= retries:
                    wait_time = backoff * (2 ** (attempt - 1))
                    logger.warning(f"Request failed, retrying in {wait_time}s... ({attempt}/{retries})")
                    await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"Unexpected error during request: {str(e)}")
                raise

        raise last_error or ConnectionError("Max retries exceeded")

    async def _single_request(self, method: str, path: str, headers: Optional[Dict] = None,
                             body: Any = None, params: Optional[Dict] = None) -> HTTPResponse:
        """
        Perform a single async HTTP request with enhanced error handling.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_single_request,
            method, path, headers, body, params
        )

    def _sync_single_request(self, method: str, path: str, headers: Optional[Dict] = None,
                            body: Any = None, params: Optional[Dict] = None) -> HTTPResponse:
        """
        Perform a single synchronous HTTP request with enhanced error handling.
        """
        full_path = self._build_full_path(path, params)
        merged_headers = {**self.default_headers, **(headers or {})}

        if isinstance(body, dict):
            body = json.dumps(body)
            merged_headers.setdefault('Content-Type', 'application/json')

        conn = self._connection or self._get_connection()

        try:
            # Set socket timeout
            if hasattr(conn, 'sock') and conn.sock:
                conn.sock.settimeout(self.timeout)

            # Log request details at debug level
            logger.debug(f"Sending {method} request to {self.host}{full_path}")
            start_time = time.time()

            # Make the request
            conn.request(method, full_path, body, merged_headers)

            # Get the response with timeout handling
            try:
                response = conn.getresponse()
                raw_data = response.read().decode('utf-8')

                # Log response time
                elapsed = time.time() - start_time
                logger.debug(f"Response received in {elapsed:.2f}s: {response.status} {response.reason}")

                return HTTPResponse(
                    response.status,
                    response.reason,
                    dict(response.getheaders()),
                    raw_data
                )
            except socket.timeout:
                elapsed = time.time() - start_time
                logger.error(f"Request timed out after {elapsed:.2f}s")
                raise socket.timeout(f"Request timed out after {elapsed:.2f}s")

        except socket.timeout as e:
            # Convert socket.timeout to a more descriptive error
            raise TimeoutError(f"Request to {self.host}{full_path} timed out after {self.timeout}s") from e
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
        finally:
            if not self._connection:
                conn.close()

    def _build_full_path(self, path: str, params: Optional[Dict] = None) -> str:
        """Build the full request path with parameters"""
        if path.startswith("/"):
            full_path = f"{self.base_path}{path}"
        else:
            full_path = f"{self.base_path}/{path}"
        return build_url_with_params(full_path, params)

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
        self._connection = self._get_connection()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Async context manager exit"""
        if self._connection:
            self._connection.close()
        self._connection = None
        if self._executor:
            self._executor.shutdown(wait=False)

    def __enter__(self):
        """Sync context manager entry"""
        self._connection = self._get_connection()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Sync context manager exit"""
        if self._connection:
            self._connection.close()
        self._connection = None

    # High-Level JSON Methods

    async def get_json(self, path, headers=None, params=None, retries=1, backoff=1):
        """
        Sends a GET request and returns JSON data if available.
        Raises ValueError if not valid JSON.
        """
        response = await self.get(path, headers=headers, params=params, retries=retries, backoff=backoff)
        try:
            return response.json()
        except json.JSONDecodeError:
            raise ValueError("Response data is not valid JSON")

    async def post_json(self, path, headers=None, body=None, params=None, retries=1, backoff=1):
        """
        Sends a POST request with an optional JSON body and returns JSON data if available.
        Raises ValueError if not valid JSON.
        """
        response = await self.post(path, headers=headers, body=body, params=params, retries=retries, backoff=backoff)
        try:
            return response.json()
        except json.JSONDecodeError:
            raise ValueError("Response data is not valid JSON")
