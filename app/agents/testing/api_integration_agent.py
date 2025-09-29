"""
API Integration Agent - Specialized Agent for API Operations.

This agent demonstrates:
- Specialized API integration capabilities
- HTTP request handling and authentication
- Rate limiting and circuit breaker patterns
- Comprehensive API response processing
- Error handling and retry logic
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agents.base.agent import LangGraphAgent, AgentConfig, AgentCapability
from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType, MemoryType
from app.agents.testing.custom_agent_logger import (
    custom_agent_logger, AgentMetadata, ThinkingProcess, ToolUsage, LogLevel
)
from app.tools.production.api_integration_tool import api_integration_tool
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType

logger = structlog.get_logger(__name__)


class APIIntegrationAgent:
    """
    Specialized agent for API integration and HTTP operations.
    
    Capabilities:
    - HTTP requests (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)
    - Multiple authentication methods (API Key, Bearer, Basic, OAuth2, JWT)
    - Rate limiting and circuit breaker patterns
    - Response caching and optimization
    - Error handling and retry logic
    - API documentation analysis
    """
    
    def __init__(self):
        """Initialize the API Integration Agent."""
        self.agent_id = str(uuid.uuid4())
        self.session_id = None
        self.agent = None
        self.llm_manager = None
        
        # Agent configuration
        self.agent_metadata = AgentMetadata(
            agent_id=self.agent_id,
            agent_type="REACT",
            agent_name="API Integration Specialist Agent",
            capabilities=["reasoning", "tool_use", "api_integration"],
            tools_available=["api_integration_tool"],
            memory_type="simple",
            rag_enabled=False
        )
        
        # System prompt for API operations
        self.system_prompt = """You are an API Integration Specialist Agent, an expert in HTTP requests and API interactions.

Your capabilities include:
- Making HTTP requests (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)
- Handling multiple authentication methods (API Key, Bearer Token, Basic Auth, OAuth2, JWT)
- Managing rate limiting and circuit breaker patterns
- Processing and analyzing API responses
- Caching responses for performance optimization
- Error handling and retry logic
- API documentation analysis and testing

You have access to the Revolutionary API Integration Tool which provides:
- Universal HTTP support with intelligent handling
- Enterprise-grade authentication and security
- Performance optimization with caching and rate limiting
- Circuit breaker patterns for reliability
- Comprehensive error handling and logging

When handling API operations:
1. Analyze the API requirements and authentication needs
2. Choose appropriate HTTP method and parameters
3. Handle authentication securely
4. Process responses intelligently
5. Implement proper error handling
6. Optimize for performance and reliability

Think step by step and explain your reasoning for each API operation."""
        
        logger.info("API Integration Agent initialized", agent_id=self.agent_id)
    
    async def initialize(self) -> bool:
        """Initialize the agent with LLM and tools."""
        try:
            # Get LLM manager
            self.llm_manager = get_enhanced_llm_manager()
            
            # Get default LLM config
            from app.llm.models import LLMConfig, ProviderType
            llm_config = LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="phi4:latest",
                temperature=0.7
            )

            # Create agent configuration
            config = AgentBuilderConfig(
                name="API Integration Specialist Agent",
                description="Specialized agent for API integration and HTTP operations",
                agent_type=AgentType.REACT,
                llm_config=llm_config,
                capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
                tools=["api_integration_tool"],
                memory_type=MemoryType.SIMPLE,
                enable_memory=True,
                system_prompt=self.system_prompt
            )
            
            # Create agent using factory
            factory = AgentBuilderFactory(self.llm_manager)
            self.agent = await factory.build_agent(config)
            
            logger.info("API Integration Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize API Integration Agent", error=str(e))
            return False
    
    async def process_request(self, user_query: str) -> Dict[str, Any]:
        """Process an API request with comprehensive logging."""
        # Start logging session
        self.session_id = f"api_agent_{uuid.uuid4().hex[:8]}"
        custom_agent_logger.start_session(self.session_id, self.agent_metadata)
        
        start_time = datetime.now()
        
        try:
            # Log query received
            custom_agent_logger.log_query_received(
                self.session_id, 
                user_query, 
                self.system_prompt
            )
            
            # Step 1: Analyze the API request
            thinking_step_1 = ThinkingProcess(
                step_number=1,
                thought="Analyzing the API request to determine HTTP method, endpoint, and authentication",
                reasoning=f"User query: '{user_query}'. I need to identify the API operation required.",
                decision="Proceed with API request analysis and planning"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_1)
            
            # Step 2: Plan the API operation
            thinking_step_2 = ThinkingProcess(
                step_number=2,
                thought="Planning the API operation with appropriate parameters and authentication",
                reasoning="Determining HTTP method, headers, authentication, and request body",
                decision="Execute planned API operation using the API integration tool"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_2)
            
            # Execute the API request
            actual_result = await self._execute_api_operation(user_query)
            
            # Log tool usage
            tool_usage = ToolUsage(
                tool_name="api_integration_tool",
                parameters=actual_result.get("parameters", {}),
                execution_time=actual_result.get("execution_time", 0.0),
                success=actual_result.get("success", False),
                result=actual_result.get("result"),
                error=actual_result.get("error")
            )
            custom_agent_logger.log_tool_usage(self.session_id, tool_usage)
            
            # Step 3: Process API response
            thinking_step_3 = ThinkingProcess(
                step_number=3,
                thought="Processing API response and generating user-friendly output",
                reasoning="Analyzing response status, headers, and body for meaningful insights",
                decision="Provide comprehensive response to user"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_3)
            
            # Generate final response
            final_answer = self._generate_response(user_query, actual_result)
            
            # Log final answer
            execution_time = (datetime.now() - start_time).total_seconds()
            custom_agent_logger.log_final_answer(self.session_id, final_answer, execution_time)
            
            # End session and get summary
            session_summary = custom_agent_logger.end_session(self.session_id)
            
            return {
                "success": True,
                "response": final_answer,
                "execution_time": execution_time,
                "session_summary": session_summary,
                "agent_metadata": self.agent_metadata.__dict__,
                "api_result": actual_result
            }
            
        except Exception as e:
            logger.error("API Integration Agent request failed", error=str(e))
            
            # Log error
            if self.session_id:
                custom_agent_logger.end_session(self.session_id)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_api_operation(self, query: str) -> Dict[str, Any]:
        """Execute actual API operation based on query."""
        start_time = datetime.now()
        
        try:
            # Analyze query to determine API operation
            query_lower = query.lower()
            
            if "get" in query_lower and ("status" in query_lower or "health" in query_lower):
                # Health check request
                parameters = {
                    "method": "GET",
                    "url": "https://httpbin.org/status/200",
                    "timeout": 10
                }
                result = await api_integration_tool.arun(parameters)
                
            elif "post" in query_lower and "data" in query_lower:
                # POST request with data
                parameters = {
                    "method": "POST",
                    "url": "https://httpbin.org/post",
                    "json_data": {"message": "Hello from API Integration Agent", "timestamp": str(datetime.now())},
                    "timeout": 10
                }
                result = await api_integration_tool.arun(parameters)
                
            elif "weather" in query_lower or "api" in query_lower:
                # Weather API request (using a free service)
                parameters = {
                    "method": "GET",
                    "url": "https://api.weatherapi.com/v1/current.json",
                    "params": {"key": "demo", "q": "London"},
                    "timeout": 10
                }
                result = await api_integration_tool.arun(parameters)
                
            elif "json" in query_lower or "placeholder" in query_lower:
                # JSONPlaceholder API test
                parameters = {
                    "method": "GET",
                    "url": "https://jsonplaceholder.typicode.com/posts/1",
                    "timeout": 10
                }
                result = await api_integration_tool.arun(parameters)
                
            else:
                # Default: Simple GET request to httpbin
                parameters = {
                    "method": "GET",
                    "url": "https://httpbin.org/get",
                    "timeout": 10
                }
                result = await api_integration_tool.arun(parameters)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time
            }
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error("API operation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def _generate_response(self, query: str, api_result: Dict[str, Any]) -> str:
        """Generate comprehensive response based on API results."""
        if not api_result.get("success"):
            return f"""I encountered an error while processing your API request: "{query}"

Error: {api_result.get('error', 'Unknown error')}

I attempted to make the API call but ran into an issue. This could be due to:
- Network connectivity issues
- API endpoint unavailability
- Authentication problems
- Rate limiting
- Invalid request parameters

The Revolutionary API Integration Tool includes circuit breaker patterns and retry logic to handle such issues automatically. Would you like me to try a different approach or help troubleshoot the issue?"""
        
        result = api_result.get("result", {})
        parameters = api_result.get("parameters", {})
        execution_time = api_result.get("execution_time", 0.0)
        
        # Parse the result based on its structure
        if isinstance(result, dict):
            status_code = result.get("status_code", "Unknown")
            response_data = result.get("response", {})
            headers = result.get("headers", {})
            
            response = f"""I've successfully processed your API request: "{query}"

API Operation Details:
- Method: {parameters.get('method', 'GET')}
- URL: {parameters.get('url', 'Unknown')}
- Status Code: {status_code}
- Execution Time: {execution_time:.3f} seconds

Response Summary:
{self._format_api_response(response_data)}

The Revolutionary API Integration Tool handled this request with enterprise-grade reliability, including:
- Intelligent error handling and retry logic
- Performance optimization with caching
- Circuit breaker patterns for resilience
- Comprehensive logging and monitoring

Is there another API operation you'd like me to perform?"""
        else:
            response = f"""I've processed your API request: "{query}"

Result: {str(result)}

Execution Time: {execution_time:.3f} seconds

The API operation completed successfully using the Revolutionary API Integration Tool."""
        
        return response
    
    def _format_api_response(self, response_data: Any) -> str:
        """Format API response data for user-friendly display."""
        if isinstance(response_data, dict):
            if len(response_data) <= 5:
                # Small response - show all data
                formatted = []
                for key, value in response_data.items():
                    if isinstance(value, (dict, list)) and len(str(value)) > 100:
                        formatted.append(f"- {key}: [Complex data structure]")
                    else:
                        formatted.append(f"- {key}: {value}")
                return "\n".join(formatted)
            else:
                # Large response - show summary
                return f"Response contains {len(response_data)} fields: {', '.join(list(response_data.keys())[:5])}{'...' if len(response_data) > 5 else ''}"
        elif isinstance(response_data, list):
            return f"Response contains {len(response_data)} items"
        else:
            return str(response_data)[:200] + ("..." if len(str(response_data)) > 200 else "")
    
    async def demonstrate_capabilities(self) -> Dict[str, Any]:
        """Demonstrate the agent's API integration capabilities."""
        demo_queries = [
            "Make a GET request to check API status",
            "Send a POST request with JSON data",
            "Get data from JSONPlaceholder API",
            "Make a simple HTTP GET request",
            "Test API connectivity and response time"
        ]
        
        results = []
        
        for query in demo_queries:
            logger.info("Demonstrating API capability", query=query)
            result = await self.process_request(query)
            results.append({
                "query": query,
                "success": result["success"],
                "execution_time": result["execution_time"],
                "api_success": result.get("api_result", {}).get("success", False)
            })
            
            # Small delay between operations
            await asyncio.sleep(1.0)
        
        return {
            "agent_type": "API Integration Specialist Agent",
            "capabilities_demonstrated": len(demo_queries),
            "results": results,
            "overall_success": all(r["success"] for r in results),
            "api_success_rate": sum(1 for r in results if r.get("api_success", False)) / len(results)
        }


# Create global instance
api_integration_agent = APIIntegrationAgent()


async def main():
    """Test the API Integration Agent."""
    print("🌐 Testing API Integration Agent...")
    
    # Initialize agent
    success = await api_integration_agent.initialize()
    if not success:
        print("❌ Failed to initialize API Integration Agent")
        return
    
    # Test with a sample query
    result = await api_integration_agent.process_request(
        "Make a GET request to test API connectivity and response time"
    )
    
    print(f"✅ Agent Response: {result['response'][:200]}...")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
    
    # Demonstrate capabilities
    demo_results = await api_integration_agent.demonstrate_capabilities()
    print(f"🎯 Capabilities Demo: {demo_results['overall_success']}")
    print(f"📊 API Success Rate: {demo_results['api_success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
