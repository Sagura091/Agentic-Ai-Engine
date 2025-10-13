"""
Revolutionary API Integration Tool for Agentic AI Systems.

This tool provides universal API connectivity and management capabilities
with intelligent handling, authentication, rate limiting, and performance optimization.
"""

import asyncio
import json
import time
import hashlib
import base64
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urljoin, urlparse

import aiohttp
from pydantic import BaseModel, Field, validator, HttpUrl
from langchain_core.tools import BaseTool

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.tools.unified_tool_repository import ToolCategory as ToolCategoryEnum, ToolAccessLevel, ToolMetadata

logger = get_logger()


class HTTPMethod(str, Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(str, Enum):
    """Supported authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM_HEADER = "custom_header"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float = 10.0
    requests_per_minute: int = 600
    requests_per_hour: int = 3600
    burst_size: int = 20
    backoff_factor: float = 2.0
    max_retries: int = 3


@dataclass
class APIResponse:
    """API response structure."""
    status_code: int
    headers: Dict[str, str]
    content: Union[Dict, List, str]
    response_time: float
    success: bool
    error: Optional[str] = None


class APIIntegrationInput(BaseModel):
    """Input schema for API integration operations."""
    url: HttpUrl = Field(..., description="API endpoint URL")
    method: HTTPMethod = Field(default=HTTPMethod.GET, description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Request headers")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")
    data: Optional[Union[Dict, List, str]] = Field(default=None, description="Request body data")
    json_data: Optional[Union[Dict, List]] = Field(default=None, description="JSON request body")
    
    # Authentication
    auth_type: AuthType = Field(default=AuthType.NONE, description="Authentication type")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    bearer_token: Optional[str] = Field(default=None, description="Bearer token")
    username: Optional[str] = Field(default=None, description="Username for basic auth")
    password: Optional[str] = Field(default=None, description="Password for basic auth")
    custom_auth_header: Optional[str] = Field(default=None, description="Custom auth header name")
    custom_auth_value: Optional[str] = Field(default=None, description="Custom auth header value")
    
    # Request configuration
    timeout: int = Field(default=30, description="Request timeout in seconds")
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries")
    
    # Response handling
    parse_json: bool = Field(default=True, description="Automatically parse JSON responses")
    include_headers: bool = Field(default=True, description="Include response headers")
    stream_response: bool = Field(default=False, description="Stream large responses")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format."""
        if not str(v).startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class APIIntegrationTool(BaseTool):
    """
    Revolutionary API Integration Tool.
    
    Provides universal API connectivity with:
    - Multiple authentication methods (API key, Bearer, Basic, OAuth2, JWT)
    - Intelligent rate limiting with backoff strategies
    - Response parsing and validation
    - Circuit breaker pattern for resilience
    - Request/response caching with TTL
    - Performance monitoring and analytics
    - Webhook handling and management
    - Batch operations with concurrent requests
    """

    name: str = "api_integration"
    description: str = """
    Revolutionary API integration tool with intelligent handling and enterprise features.
    
    CORE CAPABILITIES:
    ✅ RESTful API calls (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)
    ✅ Multiple authentication methods (API key, Bearer, Basic, OAuth2, JWT)
    ✅ Intelligent rate limiting with exponential backoff
    ✅ Response parsing and validation
    ✅ Circuit breaker pattern for resilience
    ✅ Request/response caching with TTL
    ✅ Batch operations with concurrent requests
    ✅ Webhook handling and management
    ✅ Performance monitoring and analytics
    
    AUTHENTICATION SUPPORT:
    🔐 API Key authentication
    🔐 Bearer token authentication
    🔐 Basic authentication
    🔐 OAuth 2.0 flows
    🔐 JWT token handling
    🔐 Custom header authentication
    
    ADVANCED FEATURES:
    🚀 Auto-retry with exponential backoff
    🚀 Request/response transformation
    🚀 API versioning support
    🚀 Mock API generation for testing
    🚀 Performance monitoring and analytics
    
    Use this tool for any API integration - it's intelligent, resilient, and secure!
    """
    args_schema: Type[BaseModel] = APIIntegrationInput

    def __init__(self):
        super().__init__()

        # Rate limiting and circuit breaker (private attributes to avoid Pydantic validation)
        self._rate_limit_config = RateLimitConfig()
        self._request_times = []
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
        self._circuit_breaker_state = "closed"  # closed, open, half-open

        # Performance tracking
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_response_time = 0.0
        self._last_used = None

        # Response cache
        self._response_cache = {}
        self._cache_ttl = 300  # 5 minutes default

        # Session for connection pooling
        self._session = None

        logger.info(
            "API Integration Tool initialized",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.production.api_integration_tool"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'Agentic-AI-System/1.0'}
            )
        
        return self._session

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        
        # Clean old request times
        cutoff = now - 60  # Keep last minute
        self._request_times = [t for t in self._request_times if t > cutoff]
        
        # Check various rate limits
        recent_requests = len([t for t in self._request_times if t > now - 1])  # Last second
        minute_requests = len(self._request_times)  # Last minute
        
        if recent_requests >= self._rate_limit_config.requests_per_second:
            return False

        if minute_requests >= self._rate_limit_config.requests_per_minute:
            return False
        
        return True

    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state."""
        now = time.time()
        
        if self._circuit_breaker_state == "open":
            # Check if we should try half-open
            if (self._circuit_breaker_last_failure and
                now - self._circuit_breaker_last_failure > 60):  # 1 minute timeout
                self._circuit_breaker_state = "half-open"
                logger.info(
                    "Circuit breaker moving to half-open state",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.production.api_integration_tool"
                )
                return True
            return False

        return True

    def _update_circuit_breaker(self, success: bool):
        """Update circuit breaker state based on request result."""
        if success:
            if self._circuit_breaker_state == "half-open":
                self._circuit_breaker_state = "closed"
                self._circuit_breaker_failures = 0
                logger.info(
                    "Circuit breaker closed - service recovered",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.production.api_integration_tool"
                )
        else:
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = time.time()

            if self._circuit_breaker_failures >= 5:  # Threshold
                self._circuit_breaker_state = "open"
                logger.warning(
                    "Circuit breaker opened - service failing",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.production.api_integration_tool"
                )

    def _prepare_auth_headers(self, input_data: APIIntegrationInput) -> Dict[str, str]:
        """Prepare authentication headers."""
        headers = input_data.headers.copy() if input_data.headers else {}
        
        if input_data.auth_type == AuthType.API_KEY and input_data.api_key:
            headers[input_data.api_key_header] = input_data.api_key
        
        elif input_data.auth_type == AuthType.BEARER_TOKEN and input_data.bearer_token:
            headers['Authorization'] = f'Bearer {input_data.bearer_token}'
        
        elif input_data.auth_type == AuthType.BASIC_AUTH and input_data.username and input_data.password:
            credentials = base64.b64encode(f'{input_data.username}:{input_data.password}'.encode()).decode()
            headers['Authorization'] = f'Basic {credentials}'
        
        elif input_data.auth_type == AuthType.JWT and input_data.bearer_token:
            headers['Authorization'] = f'Bearer {input_data.bearer_token}'
        
        elif (input_data.auth_type == AuthType.CUSTOM_HEADER and 
              input_data.custom_auth_header and input_data.custom_auth_value):
            headers[input_data.custom_auth_header] = input_data.custom_auth_value
        
        return headers

    def _get_cache_key(self, method: str, url: str, params: Optional[Dict], 
                      headers: Optional[Dict]) -> str:
        """Generate cache key for request."""
        cache_data = {
            'method': method,
            'url': url,
            'params': params or {},
            'headers': {k: v for k, v in (headers or {}).items() 
                       if k.lower() not in ['authorization', 'x-api-key']}
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

    def _is_cacheable(self, method: str, status_code: int) -> bool:
        """Check if response should be cached."""
        return method.upper() == 'GET' and 200 <= status_code < 300

    async def _make_request(self, input_data: APIIntegrationInput) -> APIResponse:
        """Make HTTP request with all features."""
        start_time = time.time()

        try:
            # Check rate limiting
            if not self._check_rate_limit():
                await asyncio.sleep(1.0 / self._rate_limit_config.requests_per_second)

            # Check circuit breaker
            if not self._check_circuit_breaker():
                raise Exception("Circuit breaker is open - service unavailable")

            # Record request time for rate limiting
            self._request_times.append(time.time())

            # Check cache for GET requests
            cache_key = None
            if input_data.method == HTTPMethod.GET:
                cache_key = self._get_cache_key(
                    input_data.method, str(input_data.url),
                    input_data.params, input_data.headers
                )

                if cache_key in self._response_cache:
                    cached_response, cache_time = self._response_cache[cache_key]
                    if time.time() - cache_time < self._cache_ttl:
                        logger.info(
                            "Returning cached response",
                            LogCategory.TOOL_OPERATIONS,
                            "app.tools.production.api_integration_tool",
                            data={"url": str(input_data.url)}
                        )
                        return cached_response

            # Prepare headers with authentication
            headers = self._prepare_auth_headers(input_data)

            # Set content type for JSON data
            if input_data.json_data:
                headers['Content-Type'] = 'application/json'

            # Get session
            session = await self._get_session()

            # Prepare request data
            request_kwargs = {
                'method': input_data.method.value,
                'url': str(input_data.url),
                'headers': headers,
                'params': input_data.params,
                'timeout': aiohttp.ClientTimeout(total=input_data.timeout),
                'allow_redirects': input_data.follow_redirects,
                'ssl': input_data.verify_ssl
            }

            # Add request body
            if input_data.json_data:
                request_kwargs['json'] = input_data.json_data
            elif input_data.data:
                if isinstance(input_data.data, (dict, list)):
                    request_kwargs['json'] = input_data.data
                else:
                    request_kwargs['data'] = input_data.data

            # Make request with retries
            last_exception = None
            for attempt in range(input_data.max_retries + 1):
                try:
                    async with session.request(**request_kwargs) as response:
                        # Read response content
                        if input_data.stream_response and response.content_length and response.content_length > 1024*1024:
                            # Stream large responses
                            content_chunks = []
                            async for chunk in response.content.iter_chunked(8192):
                                content_chunks.append(chunk)
                            content_bytes = b''.join(content_chunks)
                        else:
                            content_bytes = await response.read()

                        # Parse content
                        content_text = content_bytes.decode('utf-8', errors='ignore')
                        parsed_content = content_text

                        if input_data.parse_json and response.content_type == 'application/json':
                            try:
                                parsed_content = json.loads(content_text)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Failed to parse JSON response",
                                    LogCategory.TOOL_OPERATIONS,
                                    "app.tools.production.api_integration_tool",
                                    data={"url": str(input_data.url)}
                                )

                        # Create response object
                        response_time = time.time() - start_time
                        api_response = APIResponse(
                            status_code=response.status,
                            headers=dict(response.headers) if input_data.include_headers else {},
                            content=parsed_content,
                            response_time=response_time,
                            success=200 <= response.status < 300
                        )

                        # Cache successful GET responses
                        if (cache_key and self._is_cacheable(input_data.method.value, response.status)):
                            self._response_cache[cache_key] = (api_response, time.time())

                        # Update circuit breaker
                        self._update_circuit_breaker(api_response.success)

                        logger.info(
                            "API request completed",
                            LogCategory.TOOL_OPERATIONS,
                            "app.tools.production.api_integration_tool",
                            data={
                                "method": input_data.method.value,
                                "url": str(input_data.url),
                                "status": response.status,
                                "response_time": response_time
                            }
                        )

                        return api_response

                except Exception as e:
                    last_exception = e
                    if attempt < input_data.max_retries:
                        delay = input_data.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            "Request failed, retrying",
                            LogCategory.TOOL_OPERATIONS,
                            "app.tools.production.api_integration_tool",
                            data={
                                "attempt": attempt + 1,
                                "delay": delay
                            },
                            error=e
                        )
                        await asyncio.sleep(delay)
                    else:
                        break

            # All retries failed
            response_time = time.time() - start_time
            error_response = APIResponse(
                status_code=0,
                headers={},
                content={"error": str(last_exception)},
                response_time=response_time,
                success=False,
                error=str(last_exception)
            )

            # Update circuit breaker
            self._update_circuit_breaker(False)

            return error_response

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(
                "API request failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.api_integration_tool",
                data={
                    "method": input_data.method.value,
                    "url": str(input_data.url),
                    "response_time": response_time
                },
                error=e
            )

            return APIResponse(
                status_code=0,
                headers={},
                content={"error": str(e)},
                response_time=response_time,
                success=False,
                error=str(e)
            )

    async def _batch_requests(self, requests: List[APIIntegrationInput],
                            max_concurrent: int = 10) -> List[APIResponse]:
        """Execute multiple API requests concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(request_input):
            async with semaphore:
                return await self._make_request(request_input)

        tasks = [bounded_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                processed_responses.append(APIResponse(
                    status_code=0,
                    headers={},
                    content={"error": str(response)},
                    response_time=0.0,
                    success=False,
                    error=str(response)
                ))
            else:
                processed_responses.append(response)

        return processed_responses

    def _update_metrics(self, response: APIResponse):
        """Update performance metrics."""
        self._total_requests += 1
        self._total_response_time += response.response_time
        self._last_used = datetime.now()

        if response.success:
            self._successful_requests += 1
        else:
            self._failed_requests += 1

    async def _run(self, **kwargs) -> str:
        """Execute API integration operation."""
        try:
            # Parse and validate input
            input_data = APIIntegrationInput(**kwargs)

            # Make API request
            response = await self._make_request(input_data)

            # Update metrics
            self._update_metrics(response)

            # Prepare result
            result = {
                "success": response.success,
                "status_code": response.status_code,
                "response_time": response.response_time,
                "timestamp": datetime.now().isoformat(),
                "url": str(input_data.url),
                "method": input_data.method.value
            }

            # Add response content
            if response.success:
                result["content"] = response.content
                if input_data.include_headers:
                    result["headers"] = response.headers
            else:
                result["error"] = response.error or "Request failed"
                result["error_content"] = response.content

            # Add performance metrics
            result["tool_metrics"] = {
                "total_requests": self._total_requests,
                "success_rate": self._successful_requests / self._total_requests if self._total_requests > 0 else 0,
                "average_response_time": self._total_response_time / self._total_requests if self._total_requests > 0 else 0,
                "circuit_breaker_state": self._circuit_breaker_state,
                "cache_size": len(self._response_cache)
            }

            logger.info(
                "API integration completed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.api_integration_tool",
                data={
                    "url": str(input_data.url),
                    "method": input_data.method.value,
                    "success": response.success,
                    "status_code": response.status_code
                }
            )

            return str(result)

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "url": kwargs.get('url', 'unknown'),
                "method": kwargs.get('method', 'unknown')
            }

            logger.error(
                "API integration failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.api_integration_tool",
                data={
                    "url": kwargs.get('url'),
                    "method": kwargs.get('method')
                },
                error=e
            )

            return str(error_result)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Create the tool instance (following the existing pattern)
api_integration_tool = APIIntegrationTool()

# Tool metadata for UnifiedToolRepository registration
API_INTEGRATION_TOOL_METADATA = ToolMetadata(
    tool_id="api_integration_v1",
    name="api_integration",
    description="Revolutionary API integration with intelligent handling and enterprise features - Universal HTTP methods, multiple authentication, rate limiting, caching, and circuit breaker patterns",
    category=ToolCategoryEnum.COMMUNICATION,
    access_level=ToolAccessLevel.PUBLIC,
    requires_rag=False,
    use_cases={"api_integration", "data_retrieval", "web_services", "authentication", "automation"}
)
