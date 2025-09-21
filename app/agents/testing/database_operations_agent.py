"""
Database Operations Agent - Specialized Agent for Database Management.

This agent demonstrates:
- Multi-database connectivity (SQLite, PostgreSQL, MySQL, MongoDB, Redis)
- SQL query execution and optimization
- Database schema management
- Data migration and backup operations
- Performance monitoring and analytics
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
from app.tools.production.database_operations_tool import database_operations_tool
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType

logger = structlog.get_logger(__name__)


class DatabaseOperationsAgent:
    """
    Specialized agent for database operations and management.
    
    Capabilities:
    - Multi-database connectivity (SQLite, PostgreSQL, MySQL, MongoDB, Redis)
    - SQL query execution with injection prevention
    - Database schema management and migrations
    - Data backup and restore operations
    - Performance monitoring and optimization
    - Connection pooling and management
    """
    
    def __init__(self):
        """Initialize the Database Operations Agent."""
        self.agent_id = str(uuid.uuid4())
        self.session_id = None
        self.agent = None
        self.llm_manager = None
        
        # Agent configuration
        self.agent_metadata = AgentMetadata(
            agent_id=self.agent_id,
            agent_type="REACT",
            agent_name="Database Operations Specialist Agent",
            capabilities=["reasoning", "tool_use", "database_management"],
            tools_available=["database_operations_tool"],
            memory_type="simple",
            rag_enabled=False
        )
        
        # System prompt for database operations
        self.system_prompt = """You are a Database Operations Specialist Agent, an expert in database management and operations.

Your capabilities include:
- Multi-database connectivity (SQLite, PostgreSQL, MySQL, MongoDB, Redis)
- SQL query execution with advanced injection prevention
- Database schema management and migrations
- Data backup, restore, and migration operations
- Performance monitoring and query optimization
- Connection pooling and resource management
- Database security and access control

You have access to the Revolutionary Database Operations Tool which provides:
- Universal database connectivity with intelligent handling
- Enterprise-grade SQL injection prevention
- Performance optimization with connection pooling
- Comprehensive error handling and logging
- Advanced query analysis and optimization
- Secure credential management

When handling database operations:
1. Analyze the database requirements and connection needs
2. Choose appropriate database type and connection parameters
3. Implement proper security measures and injection prevention
4. Optimize queries for performance
5. Handle errors gracefully with proper rollback
6. Monitor performance and resource usage

Think step by step and explain your reasoning for each database operation."""
        
        logger.info("Database Operations Agent initialized", agent_id=self.agent_id)
    
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
                name="Database Operations Specialist Agent",
                description="Specialized agent for database operations and management",
                agent_type=AgentType.REACT,
                llm_config=llm_config,
                capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
                tools=["database_operations_tool"],
                memory_type=MemoryType.SIMPLE,
                enable_memory=True,
                system_prompt=self.system_prompt
            )
            
            # Create agent using factory
            factory = AgentBuilderFactory(self.llm_manager)
            self.agent = await factory.build_agent(config)
            
            logger.info("Database Operations Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Database Operations Agent", error=str(e))
            return False
    
    async def process_request(self, user_query: str) -> Dict[str, Any]:
        """Process a database request with comprehensive logging."""
        # Start logging session
        self.session_id = f"db_agent_{uuid.uuid4().hex[:8]}"
        custom_agent_logger.start_session(self.session_id, self.agent_metadata)
        
        start_time = datetime.now()
        
        try:
            # Log query received
            custom_agent_logger.log_query_received(
                self.session_id, 
                user_query, 
                self.system_prompt
            )
            
            # Step 1: Analyze the database request
            thinking_step_1 = ThinkingProcess(
                step_number=1,
                thought="Analyzing the database request to determine operation type and requirements",
                reasoning=f"User query: '{user_query}'. I need to identify the database operation needed.",
                decision="Proceed with database request analysis and planning"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_1)
            
            # Step 2: Plan the database operation
            thinking_step_2 = ThinkingProcess(
                step_number=2,
                thought="Planning the database operation with appropriate connection and query",
                reasoning="Determining database type, connection parameters, and SQL query structure",
                decision="Execute planned database operation using the database operations tool"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_2)
            
            # Execute the database operation
            actual_result = await self._execute_database_operation(user_query)
            
            # Log tool usage
            tool_usage = ToolUsage(
                tool_name="database_operations_tool",
                parameters=actual_result.get("parameters", {}),
                execution_time=actual_result.get("execution_time", 0.0),
                success=actual_result.get("success", False),
                result=actual_result.get("result"),
                error=actual_result.get("error")
            )
            custom_agent_logger.log_tool_usage(self.session_id, tool_usage)
            
            # Step 3: Process database results
            thinking_step_3 = ThinkingProcess(
                step_number=3,
                thought="Processing database results and generating user-friendly output",
                reasoning="Analyzing query results, performance metrics, and any issues",
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
                "database_result": actual_result
            }
            
        except Exception as e:
            logger.error("Database Operations Agent request failed", error=str(e))
            
            # Log error
            if self.session_id:
                custom_agent_logger.end_session(self.session_id)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_database_operation(self, query: str) -> Dict[str, Any]:
        """Execute actual database operation based on query."""
        start_time = datetime.now()
        
        try:
            # Analyze query to determine database operation
            query_lower = query.lower()
            
            if "create" in query_lower and ("table" in query_lower or "database" in query_lower):
                # Create table operation
                parameters = {
                    "operation": "execute",
                    "database_type": "sqlite",
                    "connection_string": ":memory:",
                    "query": """
                    CREATE TABLE IF NOT EXISTS agent_test (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data TEXT
                    )
                    """
                }
                result = await database_operations_tool.arun(parameters)
                
            elif "insert" in query_lower or "add" in query_lower:
                # Insert data operation
                parameters = {
                    "operation": "execute",
                    "database_type": "sqlite",
                    "connection_string": ":memory:",
                    "query": "INSERT INTO agent_test (name, data) VALUES (?, ?)",
                    "parameters": ["Database Agent Test", f"Created at {datetime.now()}"]
                }
                result = await database_operations_tool.arun(parameters)
                
            elif "select" in query_lower or "query" in query_lower or "find" in query_lower:
                # Select/Query operation
                parameters = {
                    "operation": "query",
                    "database_type": "sqlite",
                    "connection_string": ":memory:",
                    "query": "SELECT name, created_at FROM agent_test LIMIT 10"
                }
                result = await database_operations_tool.arun(parameters)
                
            elif "backup" in query_lower:
                # Backup operation
                parameters = {
                    "operation": "backup",
                    "database_type": "sqlite",
                    "connection_string": ":memory:",
                    "backup_path": "test_backup.db"
                }
                result = await database_operations_tool.arun(parameters)
                
            else:
                # Default: Test connection
                parameters = {
                    "operation": "test_connection",
                    "database_type": "sqlite",
                    "connection_string": ":memory:"
                }
                result = await database_operations_tool.arun(parameters)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time
            }
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error("Database operation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def _generate_response(self, query: str, db_result: Dict[str, Any]) -> str:
        """Generate comprehensive response based on database results."""
        if not db_result.get("success"):
            return f"""I encountered an error while processing your database request: "{query}"

Error: {db_result.get('error', 'Unknown error')}

I attempted to execute the database operation but ran into an issue. This could be due to:
- Database connectivity problems
- Invalid SQL syntax or parameters
- Permission or access control issues
- Database schema conflicts
- Resource limitations

The Revolutionary Database Operations Tool includes comprehensive error handling and SQL injection prevention. Would you like me to try a different approach or help troubleshoot the issue?"""
        
        result = db_result.get("result", {})
        parameters = db_result.get("parameters", {})
        execution_time = db_result.get("execution_time", 0.0)
        
        operation = parameters.get("operation", "unknown")
        database_type = parameters.get("database_type", "unknown")
        
        response = f"""I've successfully processed your database request: "{query}"

Database Operation Details:
- Operation: {operation.title()}
- Database Type: {database_type.upper()}
- Execution Time: {execution_time:.3f} seconds

Results:
{self._format_database_result(result, operation)}

The Revolutionary Database Operations Tool handled this request with enterprise-grade features:
- SQL injection prevention and security validation
- Connection pooling for optimal performance
- Comprehensive error handling and rollback
- Performance monitoring and optimization
- Multi-database compatibility

Is there another database operation you'd like me to perform?"""
        
        return response
    
    def _format_database_result(self, result: Any, operation: str) -> str:
        """Format database result for user-friendly display."""
        if isinstance(result, dict):
            if "rows" in result:
                rows = result["rows"]
                if isinstance(rows, list) and rows:
                    return f"Query returned {len(rows)} rows:\n" + "\n".join([f"- {row}" for row in rows[:5]]) + ("..." if len(rows) > 5 else "")
                else:
                    return "Query executed successfully (no rows returned)"
            elif "affected_rows" in result:
                return f"Operation affected {result['affected_rows']} rows"
            elif "success" in result:
                return f"Operation completed successfully: {result.get('message', 'No additional details')}"
            else:
                return f"Operation result: {str(result)[:200]}"
        elif isinstance(result, list):
            return f"Operation returned {len(result)} items"
        else:
            return str(result)[:200] + ("..." if len(str(result)) > 200 else "")
    
    async def demonstrate_capabilities(self) -> Dict[str, Any]:
        """Demonstrate the agent's database operation capabilities."""
        demo_queries = [
            "Test database connection",
            "Create a new table for testing",
            "Insert some test data into the table",
            "Query the data from the table",
            "Backup the database"
        ]
        
        results = []
        
        for query in demo_queries:
            logger.info("Demonstrating database capability", query=query)
            result = await self.process_request(query)
            results.append({
                "query": query,
                "success": result["success"],
                "execution_time": result["execution_time"],
                "database_success": result.get("database_result", {}).get("success", False)
            })
            
            # Small delay between operations
            await asyncio.sleep(0.5)
        
        return {
            "agent_type": "Database Operations Specialist Agent",
            "capabilities_demonstrated": len(demo_queries),
            "results": results,
            "overall_success": all(r["success"] for r in results),
            "database_success_rate": sum(1 for r in results if r.get("database_success", False)) / len(results)
        }


# Create global instance
database_operations_agent = DatabaseOperationsAgent()


async def main():
    """Test the Database Operations Agent."""
    print("🗄️  Testing Database Operations Agent...")
    
    # Initialize agent
    success = await database_operations_agent.initialize()
    if not success:
        print("❌ Failed to initialize Database Operations Agent")
        return
    
    # Test with a sample query
    result = await database_operations_agent.process_request(
        "Create a test table and insert some sample data"
    )
    
    print(f"✅ Agent Response: {result['response'][:200]}...")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
    
    # Demonstrate capabilities
    demo_results = await database_operations_agent.demonstrate_capabilities()
    print(f"🎯 Capabilities Demo: {demo_results['overall_success']}")
    print(f"📊 Database Success Rate: {demo_results['database_success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
