"""
Comprehensive test suite for API Integration Tool.

Tests all functionality including authentication, rate limiting, caching, and error handling.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import patch, MagicMock, AsyncMock
from aiohttp import ClientResponse, ClientSession
from aiohttp.test_utils import make_mocked_coro

from app.tools.production.api_integration_tool import (
    APIIntegrationTool,
    HTTPMethod,
    AuthType,
    APIIntegrationInput,
    APIResponse
)


class TestAPIIntegrationTool:
    """Test suite for API Integration Tool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance for testing."""
        return APIIntegrationTool()

    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        response = MagicMock(spec=ClientResponse)
        response.status = 200
        response.headers = {'Content-Type': 'application/json'}
        response.content_type = 'application/json'
        response.content_length = 100
        response.read = AsyncMock(return_value=b'{"message": "success"}')
        response.content.iter_chunked = AsyncMock(return_value=[b'{"message": "success"}'])
        return response

    @pytest.mark.asyncio
    async def test_simple_get_request(self, tool):
        """Test basic GET request functionality."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"data": "test"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            result = await tool._run(
                url="https://api.example.com/test",
                method=HTTPMethod.GET
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is True
            assert result_dict["status_code"] == 200
            assert result_dict["method"] == "GET"

    @pytest.mark.asyncio
    async def test_post_with_json_data(self, tool):
        """Test POST request with JSON data."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 201
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"id": 123, "created": true}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            test_data = {"name": "test", "value": 42}
            result = await tool._run(
                url="https://api.example.com/create",
                method=HTTPMethod.POST,
                json_data=test_data
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is True
            assert result_dict["status_code"] == 201

    @pytest.mark.asyncio
    async def test_api_key_authentication(self, tool):
        """Test API key authentication."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"authenticated": true}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            result = await tool._run(
                url="https://api.example.com/protected",
                method=HTTPMethod.GET,
                auth_type=AuthType.API_KEY,
                api_key="test-api-key-123",
                api_key_header="X-API-Key"
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is True
            
            # Verify API key was included in headers
            call_args = mock_session.request.call_args
            headers = call_args[1]['headers']
            assert headers['X-API-Key'] == "test-api-key-123"

    @pytest.mark.asyncio
    async def test_bearer_token_authentication(self, tool):
        """Test Bearer token authentication."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"user": "authenticated"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            result = await tool._run(
                url="https://api.example.com/user",
                method=HTTPMethod.GET,
                auth_type=AuthType.BEARER_TOKEN,
                bearer_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is True
            
            # Verify Bearer token was included
            call_args = mock_session.request.call_args
            headers = call_args[1]['headers']
            assert headers['Authorization'].startswith('Bearer ')

    @pytest.mark.asyncio
    async def test_basic_authentication(self, tool):
        """Test Basic authentication."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"access": "granted"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            result = await tool._run(
                url="https://api.example.com/secure",
                method=HTTPMethod.GET,
                auth_type=AuthType.BASIC_AUTH,
                username="testuser",
                password="testpass"
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is True
            
            # Verify Basic auth header was included
            call_args = mock_session.request.call_args
            headers = call_args[1]['headers']
            assert headers['Authorization'].startswith('Basic ')

    @pytest.mark.asyncio
    async def test_custom_headers(self, tool):
        """Test custom headers functionality."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"custom": "header_received"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            custom_headers = {
                "X-Custom-Header": "custom-value",
                "X-Client-Version": "1.0.0"
            }
            
            result = await tool._run(
                url="https://api.example.com/custom",
                method=HTTPMethod.GET,
                headers=custom_headers
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is True
            
            # Verify custom headers were included
            call_args = mock_session.request.call_args
            headers = call_args[1]['headers']
            assert headers['X-Custom-Header'] == "custom-value"
            assert headers['X-Client-Version'] == "1.0.0"

    @pytest.mark.asyncio
    async def test_query_parameters(self, tool):
        """Test query parameters functionality."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"filtered": "results"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            params = {
                "limit": 10,
                "offset": 0,
                "filter": "active"
            }
            
            result = await tool._run(
                url="https://api.example.com/items",
                method=HTTPMethod.GET,
                params=params
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is True
            
            # Verify params were included
            call_args = mock_session.request.call_args
            assert call_args[1]['params'] == params

    @pytest.mark.asyncio
    async def test_error_handling(self, tool):
        """Test error handling for failed requests."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 404
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"error": "Not found"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            result = await tool._run(
                url="https://api.example.com/nonexistent",
                method=HTTPMethod.GET
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is False
            assert result_dict["status_code"] == 404

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, tool):
        """Test retry mechanism with exponential backoff."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            
            # First two calls fail, third succeeds
            mock_session.request.side_effect = [
                Exception("Connection error"),
                Exception("Timeout error"),
                AsyncMock().__aenter__.return_value
            ]
            
            # Mock successful response for third attempt
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"retry": "success"}')
            mock_session.request.side_effect[-1] = mock_response
            
            mock_get_session.return_value = mock_session
            
            start_time = time.time()
            result = await tool._run(
                url="https://api.example.com/retry",
                method=HTTPMethod.GET,
                max_retries=2,
                retry_delay=0.1  # Short delay for testing
            )
            end_time = time.time()
            
            # Should have taken some time due to retries
            assert end_time - start_time >= 0.1  # At least one retry delay

    @pytest.mark.asyncio
    async def test_rate_limiting(self, tool):
        """Test rate limiting functionality."""
        # Set very low rate limit for testing
        tool.rate_limit_config.requests_per_second = 2.0
        
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"rate": "limited"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            # Make multiple rapid requests
            start_time = time.time()
            tasks = []
            for i in range(3):
                task = tool._run(
                    url=f"https://api.example.com/test{i}",
                    method=HTTPMethod.GET
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Should have been rate limited (taken longer than immediate)
            assert end_time - start_time >= 0.5  # Rate limiting delay

    @pytest.mark.asyncio
    async def test_response_caching(self, tool):
        """Test response caching for GET requests."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"cached": "response"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            # First request
            result1 = await tool._run(
                url="https://api.example.com/cacheable",
                method=HTTPMethod.GET
            )
            
            # Second identical request (should be cached)
            result2 = await tool._run(
                url="https://api.example.com/cacheable",
                method=HTTPMethod.GET
            )
            
            # Should have only made one actual HTTP request
            assert mock_session.request.call_count == 1
            
            # Both results should be successful
            result1_dict = eval(result1)
            result2_dict = eval(result2)
            assert result1_dict["success"] is True
            assert result2_dict["success"] is True

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, tool):
        """Test circuit breaker functionality."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            
            # Simulate multiple failures to trigger circuit breaker
            mock_session.request.side_effect = Exception("Service unavailable")
            mock_get_session.return_value = mock_session
            
            # Make multiple failing requests
            for i in range(6):  # Exceed failure threshold
                await tool._run(
                    url="https://api.example.com/failing",
                    method=HTTPMethod.GET,
                    max_retries=0  # No retries for faster testing
                )
            
            # Circuit breaker should now be open
            assert tool._circuit_breaker_state == "open"
            
            # Next request should fail immediately due to circuit breaker
            result = await tool._run(
                url="https://api.example.com/test",
                method=HTTPMethod.GET,
                max_retries=0
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is False
            assert "circuit breaker" in result_dict["error"].lower()

    @pytest.mark.asyncio
    async def test_performance_metrics(self, tool):
        """Test performance metrics tracking."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.content_type = 'application/json'
            mock_response.read = AsyncMock(return_value=b'{"metrics": "test"}')
            
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session
            
            # Make several requests
            for i in range(3):
                await tool._run(
                    url=f"https://api.example.com/metrics{i}",
                    method=HTTPMethod.GET
                )
            
            # Check metrics are updated
            assert tool._total_requests >= 3
            assert tool._successful_requests >= 3
            assert tool._total_response_time > 0
            assert tool._last_used is not None

    @pytest.mark.asyncio
    async def test_input_validation(self, tool):
        """Test input validation."""
        # Test invalid URL
        result = await tool._run(
            url="not-a-valid-url",
            method=HTTPMethod.GET
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is False
        assert "error" in result_dict

    @pytest.mark.asyncio
    async def test_timeout_handling(self, tool):
        """Test timeout handling."""
        with patch.object(tool, '_get_session') as mock_get_session:
            mock_session = AsyncMock(spec=ClientSession)
            mock_session.request.side_effect = asyncio.TimeoutError("Request timeout")
            mock_get_session.return_value = mock_session
            
            result = await tool._run(
                url="https://api.example.com/slow",
                method=HTTPMethod.GET,
                timeout=1,
                max_retries=0
            )
            
            result_dict = eval(result)
            assert result_dict["success"] is False
            assert "timeout" in result_dict["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
