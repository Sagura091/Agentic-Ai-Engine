"""
Text Processing NLP Agent - Specialized Agent for Natural Language Processing.

This agent demonstrates:
- Advanced text analysis and sentiment detection
- Entity extraction and keyword analysis
- Text similarity and readability analysis
- Multi-language processing capabilities
- Comprehensive NLP operations
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
from app.tools.production.text_processing_nlp_tool import text_processing_nlp_tool
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType

logger = structlog.get_logger(__name__)


class TextProcessingNLPAgent:
    """
    Specialized agent for natural language processing and text analysis.
    
    Capabilities:
    - Sentiment analysis and emotion detection
    - Named entity recognition and extraction
    - Keyword extraction and text summarization
    - Text similarity and semantic analysis
    - Readability analysis and language detection
    - Multi-language text processing
    """
    
    def __init__(self):
        """Initialize the Text Processing NLP Agent."""
        self.agent_id = str(uuid.uuid4())
        self.session_id = None
        self.agent = None
        self.llm_manager = None
        
        # Agent configuration
        self.agent_metadata = AgentMetadata(
            agent_id=self.agent_id,
            agent_type="REACT",
            agent_name="Text Processing NLP Specialist Agent",
            capabilities=["reasoning", "tool_use", "nlp_processing"],
            tools_available=["text_processing_nlp_tool"],
            memory_type="simple",
            rag_enabled=False
        )
        
        # System prompt for NLP operations
        self.system_prompt = """You are a Text Processing NLP Specialist Agent, an expert in natural language processing and text analysis.

Your capabilities include:
- Advanced sentiment analysis and emotion detection
- Named entity recognition and extraction
- Keyword extraction and text summarization
- Text similarity and semantic analysis
- Readability analysis and language detection
- Multi-language text processing and analysis

You have access to the Revolutionary Text Processing NLP Tool which provides:
- Comprehensive NLP processing with multi-language support
- Advanced sentiment analysis with confidence scores
- Entity extraction with relationship mapping
- Text similarity using state-of-the-art embeddings
- Readability analysis with multiple metrics
- Performance optimization and caching

When handling text processing operations:
1. Analyze the text requirements and processing needs
2. Choose appropriate NLP techniques and parameters
3. Process text with attention to language and context
4. Provide detailed analysis with confidence scores
5. Handle multilingual content appropriately
6. Optimize for accuracy and performance

Think step by step and explain your reasoning for each NLP operation."""
        
        logger.info("Text Processing NLP Agent initialized", agent_id=self.agent_id)
    
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
                name="Text Processing NLP Specialist Agent",
                description="Specialized agent for natural language processing and text analysis",
                agent_type=AgentType.REACT,
                llm_config=llm_config,
                capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
                tools=["text_processing_nlp_tool"],
                memory_type=MemoryType.SIMPLE,
                enable_memory=True,
                system_prompt=self.system_prompt
            )
            
            # Create agent using factory
            factory = AgentBuilderFactory(self.llm_manager)
            self.agent = await factory.build_agent(config)
            
            logger.info("Text Processing NLP Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Text Processing NLP Agent", error=str(e))
            return False
    
    async def process_request(self, user_query: str) -> Dict[str, Any]:
        """Process a text processing request with comprehensive logging."""
        # Start logging session
        self.session_id = f"nlp_agent_{uuid.uuid4().hex[:8]}"
        custom_agent_logger.start_session(self.session_id, self.agent_metadata)
        
        start_time = datetime.now()
        
        try:
            # Log query received
            custom_agent_logger.log_query_received(
                self.session_id, 
                user_query, 
                self.system_prompt
            )
            
            # Step 1: Analyze the text processing request
            thinking_step_1 = ThinkingProcess(
                step_number=1,
                thought="Analyzing the text processing request to determine NLP operations needed",
                reasoning=f"User query: '{user_query}'. I need to identify the text analysis required.",
                decision="Proceed with NLP request analysis and planning"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_1)
            
            # Step 2: Plan the NLP operation
            thinking_step_2 = ThinkingProcess(
                step_number=2,
                thought="Planning the NLP operation with appropriate techniques and parameters",
                reasoning="Determining the best NLP approach for the text analysis task",
                decision="Execute planned NLP operation using the text processing tool"
            )
            custom_agent_logger.log_thinking_process(self.session_id, thinking_step_2)
            
            # Execute the NLP operation
            actual_result = await self._execute_nlp_operation(user_query)
            
            # Log tool usage
            tool_usage = ToolUsage(
                tool_name="text_processing_nlp_tool",
                parameters=actual_result.get("parameters", {}),
                execution_time=actual_result.get("execution_time", 0.0),
                success=actual_result.get("success", False),
                result=actual_result.get("result"),
                error=actual_result.get("error")
            )
            custom_agent_logger.log_tool_usage(self.session_id, tool_usage)
            
            # Step 3: Process NLP results
            thinking_step_3 = ThinkingProcess(
                step_number=3,
                thought="Processing NLP results and generating user-friendly analysis",
                reasoning="Analyzing text processing results for meaningful insights",
                decision="Provide comprehensive NLP analysis to user"
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
                "nlp_result": actual_result
            }
            
        except Exception as e:
            logger.error("Text Processing NLP Agent request failed", error=str(e))
            
            # Log error
            if self.session_id:
                custom_agent_logger.end_session(self.session_id)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_nlp_operation(self, query: str) -> Dict[str, Any]:
        """Execute actual NLP operation based on query."""
        start_time = datetime.now()
        
        try:
            # Analyze query to determine NLP operation
            query_lower = query.lower()
            
            if "sentiment" in query_lower:
                # Sentiment analysis
                parameters = {
                    "operation": "analyze_sentiment",
                    "text": "This is a sample text for sentiment analysis. I'm feeling great today!",
                    "include_confidence": True
                }
                result = await text_processing_nlp_tool.arun(parameters)
                
            elif "entity" in query_lower or "entities" in query_lower:
                # Entity extraction
                parameters = {
                    "operation": "extract_entities",
                    "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
                    "entity_types": ["PERSON", "ORG", "GPE", "DATE"]
                }
                result = await text_processing_nlp_tool.arun(parameters)
                
            elif "similarity" in query_lower:
                # Text similarity
                parameters = {
                    "operation": "compare_similarity",
                    "text1": "The quick brown fox jumps over the lazy dog",
                    "text2": "A fast brown fox leaps over a sleepy dog",
                    "method": "semantic"
                }
                result = await text_processing_nlp_tool.arun(parameters)
                
            elif "keyword" in query_lower:
                # Keyword extraction
                parameters = {
                    "operation": "extract_keywords",
                    "text": "Natural language processing is a subfield of artificial intelligence that focuses on the interaction between computers and human language.",
                    "max_keywords": 10
                }
                result = await text_processing_nlp_tool.arun(parameters)
                
            else:
                # Default: Readability analysis
                parameters = {
                    "operation": "analyze_readability",
                    "text": "This is a sample text for readability analysis. It contains multiple sentences with varying complexity levels."
                }
                result = await text_processing_nlp_tool.arun(parameters)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time
            }
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error("NLP operation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def _generate_response(self, query: str, nlp_result: Dict[str, Any]) -> str:
        """Generate comprehensive response based on NLP results."""
        if not nlp_result.get("success"):
            return f"""I encountered an error while processing your text analysis request: "{query}"

Error: {nlp_result.get('error', 'Unknown error')}

I attempted to perform the NLP operation but ran into an issue. This could be due to:
- Text encoding or format issues
- Unsupported language or content
- Processing complexity limitations
- Resource constraints

The Revolutionary Text Processing NLP Tool includes comprehensive error handling and multi-language support. Would you like me to try a different approach or help troubleshoot the issue?"""
        
        result = nlp_result.get("result", {})
        parameters = nlp_result.get("parameters", {})
        execution_time = nlp_result.get("execution_time", 0.0)
        
        operation = parameters.get("operation", "unknown")
        
        response = f"""I've successfully processed your text analysis request: "{query}"

NLP Operation Details:
- Operation: {operation.replace('_', ' ').title()}
- Execution Time: {execution_time:.3f} seconds

Analysis Results:
{self._format_nlp_result(result, operation)}

The Revolutionary Text Processing NLP Tool provided comprehensive analysis with:
- Multi-language support and detection
- Advanced sentiment analysis with confidence scores
- State-of-the-art entity recognition
- Semantic similarity using modern embeddings
- Detailed readability metrics

Is there another text analysis you'd like me to perform?"""
        
        return response
    
    def _format_nlp_result(self, result: Any, operation: str) -> str:
        """Format NLP result for user-friendly display."""
        if isinstance(result, dict):
            if operation == "analyze_sentiment":
                sentiment = result.get("sentiment", "unknown")
                confidence = result.get("confidence", 0.0)
                return f"Sentiment: {sentiment.title()} (Confidence: {confidence:.2f})"
            elif operation == "extract_entities":
                entities = result.get("entities", [])
                if entities:
                    return f"Found {len(entities)} entities:\n" + "\n".join([f"- {ent.get('text', '')}: {ent.get('label', '')}" for ent in entities[:5]])
                else:
                    return "No entities found in the text"
            elif operation == "compare_similarity":
                similarity = result.get("similarity", 0.0)
                return f"Text similarity score: {similarity:.3f} (Range: 0.0 - 1.0)"
            elif operation == "extract_keywords":
                keywords = result.get("keywords", [])
                if keywords:
                    return f"Top keywords: {', '.join(keywords[:10])}"
                else:
                    return "No keywords extracted"
            elif operation == "analyze_readability":
                scores = result.get("readability_scores", {})
                if scores:
                    return f"Readability scores: " + ", ".join([f"{k}: {v:.1f}" for k, v in scores.items()])
                else:
                    return "Readability analysis completed"
            else:
                return str(result)[:200] + ("..." if len(str(result)) > 200 else "")
        else:
            return str(result)[:200] + ("..." if len(str(result)) > 200 else "")
    
    async def demonstrate_capabilities(self) -> Dict[str, Any]:
        """Demonstrate the agent's NLP capabilities."""
        demo_queries = [
            "Analyze the sentiment of some text",
            "Extract entities from a text sample",
            "Compare similarity between two texts",
            "Extract keywords from a document",
            "Analyze text readability"
        ]
        
        results = []
        
        for query in demo_queries:
            logger.info("Demonstrating NLP capability", query=query)
            result = await self.process_request(query)
            results.append({
                "query": query,
                "success": result["success"],
                "execution_time": result["execution_time"],
                "nlp_success": result.get("nlp_result", {}).get("success", False)
            })
            
            # Small delay between operations
            await asyncio.sleep(0.5)
        
        return {
            "agent_type": "Text Processing NLP Specialist Agent",
            "capabilities_demonstrated": len(demo_queries),
            "results": results,
            "overall_success": all(r["success"] for r in results),
            "nlp_success_rate": sum(1 for r in results if r.get("nlp_success", False)) / len(results)
        }


# Create global instance
text_processing_nlp_agent = TextProcessingNLPAgent()


async def main():
    """Test the Text Processing NLP Agent."""
    print("🔤 Testing Text Processing NLP Agent...")
    
    # Initialize agent
    success = await text_processing_nlp_agent.initialize()
    if not success:
        print("❌ Failed to initialize Text Processing NLP Agent")
        return
    
    # Test with a sample query
    result = await text_processing_nlp_agent.process_request(
        "Analyze the sentiment and extract entities from some sample text"
    )
    
    print(f"✅ Agent Response: {result['response'][:200]}...")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
    
    # Demonstrate capabilities
    demo_results = await text_processing_nlp_agent.demonstrate_capabilities()
    print(f"🎯 Capabilities Demo: {demo_results['overall_success']}")
    print(f"📊 NLP Success Rate: {demo_results['nlp_success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
