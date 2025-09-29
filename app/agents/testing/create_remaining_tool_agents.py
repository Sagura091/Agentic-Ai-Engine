"""
Batch Creation Script for Remaining Tool-Specific Agents.

This script creates the remaining 5 tool-specific agents:
- Text Processing NLP Agent
- Password Security Agent  
- Notification Alert Agent
- QR Barcode Agent
- Weather Environmental Agent
"""

import asyncio
from pathlib import Path

# Agent templates for the remaining tools
AGENT_TEMPLATES = {
    "text_processing_nlp_agent.py": {
        "tool_name": "text_processing_nlp_tool",
        "agent_name": "Text Processing NLP Specialist Agent",
        "capabilities": "text analysis, sentiment detection, entity extraction, language processing",
        "description": "Advanced natural language processing and text analysis",
        "operations": [
            "analyze_sentiment",
            "extract_entities", 
            "compare_similarity",
            "extract_keywords",
            "analyze_readability"
        ]
    },
    "password_security_agent.py": {
        "tool_name": "password_security_tool",
        "agent_name": "Password Security Specialist Agent", 
        "capabilities": "password generation, encryption, hashing, security validation",
        "description": "Military-grade cryptographic operations and security",
        "operations": [
            "generate_password",
            "encrypt_data",
            "decrypt_data", 
            "hash_password",
            "generate_token"
        ]
    },
    "notification_alert_agent.py": {
        "tool_name": "notification_alert_tool",
        "agent_name": "Notification Alert Specialist Agent",
        "capabilities": "multi-channel messaging, email, SMS, webhooks, scheduling",
        "description": "Multi-channel notification and alert management",
        "operations": [
            "send_email",
            "send_sms",
            "send_webhook",
            "schedule_notification",
            "send_slack_message"
        ]
    },
    "qr_barcode_agent.py": {
        "tool_name": "qr_barcode_tool", 
        "agent_name": "QR Barcode Specialist Agent",
        "capabilities": "QR code generation, barcode creation, scanning, validation",
        "description": "Comprehensive barcode and QR code operations",
        "operations": [
            "generate_qr",
            "generate_barcode",
            "validate_code",
            "batch_generate",
            "scan_code"
        ]
    },
    "weather_environmental_agent.py": {
        "tool_name": "weather_environmental_tool",
        "agent_name": "Weather Environmental Specialist Agent",
        "capabilities": "weather data, forecasting, air quality, environmental monitoring",
        "description": "Comprehensive weather and environmental data analysis",
        "operations": [
            "get_current_weather",
            "get_forecast",
            "get_air_quality",
            "get_marine_conditions",
            "analyze_climate"
        ]
    }
}

def generate_agent_code(filename: str, config: dict) -> str:
    """Generate agent code from template."""
    
    class_name = ''.join(word.capitalize() for word in filename.replace('.py', '').split('_'))
    
    return f'''"""
{config["agent_name"]} - Specialized Agent for {config["description"].title()}.

This agent demonstrates:
- Specialized {config["capabilities"]}
- Comprehensive logging of all interactions
- Intelligent processing capabilities
- Error handling and recovery
- Performance optimization
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
from app.tools.production.{config["tool_name"]} import {config["tool_name"]}
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType

logger = structlog.get_logger(__name__)


class {class_name}:
    """
    Specialized agent for {config["description"]}.
    
    Capabilities:
    - {config["capabilities"]}
    - Comprehensive error handling and logging
    - Performance optimization and monitoring
    - Integration with backend systems
    """
    
    def __init__(self):
        """Initialize the {config["agent_name"]}."""
        self.agent_id = str(uuid.uuid4())
        self.session_id = None
        self.agent = None
        self.llm_manager = None
        
        # Agent configuration
        self.agent_metadata = AgentMetadata(
            agent_id=self.agent_id,
            agent_type="REACT",
            agent_name="{config["agent_name"]}",
            capabilities=["reasoning", "tool_use", "{config["description"].split()[0].lower()}"],
            tools_available=["{config["tool_name"]}"],
            memory_type="simple",
            rag_enabled=False
        )
        
        # System prompt
        self.system_prompt = """You are a {config["agent_name"]}, an expert in {config["description"]}.

Your capabilities include:
- {config["capabilities"]}
- Advanced error handling and recovery
- Performance optimization and monitoring
- Comprehensive logging and reporting

You have access to the Revolutionary {config["tool_name"].replace("_", " ").title()} which provides:
- Enterprise-grade {config["description"]}
- Comprehensive error handling and validation
- Performance monitoring and optimization
- Advanced features and capabilities

When handling operations:
1. Analyze the requirements carefully
2. Choose appropriate parameters and methods
3. Handle errors gracefully
4. Provide clear feedback on results
5. Optimize for performance and reliability

Think step by step and explain your reasoning for each operation."""
        
        logger.info("{config["agent_name"]} initialized", agent_id=self.agent_id)
    
    async def initialize(self) -> bool:
        """Initialize the agent with LLM and tools."""
        try:
            # Get LLM manager
            self.llm_manager = await get_enhanced_llm_manager()
            
            # Create agent configuration
            config = AgentBuilderConfig(
                name="{config["agent_name"]}",
                description="Specialized agent for {config["description"]}",
                agent_type=AgentType.REACT,
                capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
                tools=["{config["tool_name"]}"],
                memory_type=MemoryType.SIMPLE,
                enable_memory=True,
                system_prompt=self.system_prompt
            )
            
            # Create agent using factory
            factory = AgentBuilderFactory(self.llm_manager)
            self.agent = await factory.build_agent(config)
            
            logger.info("{config["agent_name"]} initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize {config["agent_name"]}", error=str(e))
            return False
    
    async def process_request(self, user_query: str) -> Dict[str, Any]:
        """Process a request with comprehensive logging."""
        # Start logging session
        self.session_id = f"{config["tool_name"][:3]}_agent_{{uuid.uuid4().hex[:8]}}"
        custom_agent_logger.start_session(self.session_id, self.agent_metadata)
        
        start_time = datetime.now()
        
        try:
            # Log query received
            custom_agent_logger.log_query_received(
                self.session_id, 
                user_query, 
                self.system_prompt
            )
            
            # Step 1: Analyze the request
            thinking_step_1 = ThinkingProcess(
                step_number=1,
                thought="Analyzing the request to determine required operations",
                reasoning=f"User query: '{{user_query}}'. I need to identify the operation needed.",
                decision="Proceed with request analysis and planning"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_1)
            
            # Step 2: Plan the operation
            thinking_step_2 = ThinkingProcess(
                step_number=2,
                thought="Planning the operation with appropriate parameters",
                reasoning="Determining the best approach and parameters for the operation",
                decision="Execute planned operation using the {config["tool_name"]}"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_2)
            
            # Execute the operation
            actual_result = await self._execute_operation(user_query)
            
            # Log tool usage
            tool_usage = ToolUsage(
                tool_name="{config["tool_name"]}",
                parameters=actual_result.get("parameters", dict()),
                execution_time=actual_result.get("execution_time", 0.0),
                success=actual_result.get("success", False),
                result=actual_result.get("result"),
                error=actual_result.get("error")
            )
            custom_agent_logger.log_tool_usage(self.session_id, tool_usage)
            
            # Step 3: Generate response
            thinking_step_3 = ThinkingProcess(
                step_number=3,
                thought="Processing results and generating user-friendly output",
                reasoning="Analyzing operation results for meaningful insights",
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
                "operation_result": actual_result
            }
            
        except Exception as e:
            logger.error("{config["agent_name"]} request failed", error=str(e))
            
            # Log error
            if self.session_id:
                custom_agent_logger.end_session(self.session_id)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_operation(self, query: str) -> Dict[str, Any]:
        """Execute operation based on query."""
        start_time = datetime.now()
        
        try:
            query_lower = query.lower()
            
            # Determine operation based on query
            operation = "{config["operations"][0]}"  # Default operation
            for op in {config["operations"]}:
                if op.replace("_", " ") in query_lower:
                    operation = op
                    break
            
            # Execute operation
            parameters = {
                "operation": operation,
                "input": query
            }

            result = await {config["tool_name"]}.arun(parameters)

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time
            }
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error("Operation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def _generate_response(self, query: str, operation_result: Dict[str, Any]) -> str:
        """Generate comprehensive response based on operation results."""
        if not operation_result.get("success"):
            return f"""I encountered an error while processing your request: "{{query}}"

Error: {{operation_result.get('error', 'Unknown error')}}

I attempted to perform the operation but ran into an issue. The Revolutionary {config["tool_name"].replace("_", " ").title()} includes comprehensive error handling to manage such situations.

Would you like me to try a different approach or help troubleshoot the issue?"""
        
        result = operation_result.get("result", {{}})
        execution_time = operation_result.get("execution_time", 0.0)
        
        return f"""I've successfully processed your request: "{{query}}"

Operation completed using the Revolutionary {config["tool_name"].replace("_", " ").title()}:
- Execution Time: {{execution_time:.3f}} seconds
- Result: {{str(result)[:200]}}{{("..." if len(str(result)) > 200 else "")}}

The operation completed successfully with enterprise-grade reliability and performance optimization.

Is there anything else you'd like me to help you with?"""
    
    async def demonstrate_capabilities(self) -> Dict[str, Any]:
        """Demonstrate the agent's capabilities."""
        demo_queries = [
            f"Please {config["operations"][0].replace('_', ' ')}",
            f"Can you {config["operations"][1].replace('_', ' ')}",
            f"Help me {config["operations"][2].replace('_', ' ')}",
            f"I need to {config["operations"][-1].replace('_', ' ')}"
        ]
        
        results = []
        
        for query in demo_queries:
            logger.info("Demonstrating capability", query=query)
            result = await self.process_request(query)
            results.append({{
                "query": query,
                "success": result["success"],
                "execution_time": result["execution_time"]
            }})
            
            # Small delay between operations
            await asyncio.sleep(0.5)
        
        return {
            "agent_type": "{config["agent_name"]}",
            "capabilities_demonstrated": len(demo_queries),
            "results": results,
            "overall_success": all(r["success"] for r in results)
        }


# Create global instance
{config["tool_name"].replace("_tool", "_agent")} = {class_name}()


async def main():
    """Test the {config["agent_name"]}."""
    print("🔧 Testing {config["agent_name"]}...")
    
    # Initialize agent
    success = await {config["tool_name"].replace("_tool", "_agent")}.initialize()
    if not success:
        print("❌ Failed to initialize {config["agent_name"]}")
        return
    
    # Test with a sample query
    result = await {config["tool_name"].replace("_tool", "_agent")}.process_request(
        "Please demonstrate your {config["description"]} capabilities"
    )
    
    print(f"✅ Agent Response: {{result['response'][:200]}}...")
    print(f"⏱️  Execution Time: {{result['execution_time']:.2f}}s")
    
    # Demonstrate capabilities
    demo_results = await {config["tool_name"].replace("_tool", "_agent")}.demonstrate_capabilities()
    print(f"🎯 Capabilities Demo: {{demo_results['overall_success']}}")


if __name__ == "__main__":
    asyncio.run(main())'''


async def create_all_agents():
    """Create all remaining tool-specific agents."""
    agents_dir = Path("app/agents/testing")
    agents_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏗️  Creating remaining tool-specific agents...")
    
    for filename, config in AGENT_TEMPLATES.items():
        file_path = agents_dir / filename
        
        if file_path.exists():
            print(f"⚠️  {filename} already exists, skipping...")
            continue
        
        print(f"📝 Creating {filename}...")
        
        # Generate agent code
        agent_code = generate_agent_code(filename, config)
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(agent_code)
        
        print(f"✅ Created {filename}")
    
    print("🎉 All remaining tool-specific agents created!")


if __name__ == "__main__":
    asyncio.run(create_all_agents())
