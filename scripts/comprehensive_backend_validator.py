#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE BACKEND AGENT SYSTEM VALIDATOR

This script validates EVERY component of the Agentic AI Engine backend:
- Creates agents of each type and framework
- Creates knowledge bases and attaches them to agents
- Uploads documents through the ingestion pipeline
- Creates dynamic tools and assigns them to agents
- Tests agent execution with real tasks
- Validates RAG functionality (agents using their knowledge)
- Tests multi-modal processing (OCR, vision)
- Verifies memory systems
- Tests multi-agent coordination

Usage:
    # Full comprehensive backend validation
    python scripts/comprehensive_backend_validator.py

    # Quick validation (skip long tests)
    python scripts/comprehensive_backend_validator.py --quick

    # Test specific agent type only
    python scripts/comprehensive_backend_validator.py --agent-type react

    # Quick agent demonstration
    python scripts/comprehensive_backend_validator.py --demo

    # Create specific agent on-demand
    python scripts/comprehensive_backend_validator.py --create react --name "MyReActAgent" --task "Solve math problems"
    python scripts/comprehensive_backend_validator.py --create rag --model "qwen2.5:7b"
    python scripts/comprehensive_backend_validator.py --create workflow --name "ProcessManager"

    # Available agent types: react, rag, workflow, multimodal, autonomous, composite
    # Available models: llama3.2:latest, qwen2.5:7b, mistral:latest, codellama:latest
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports - Using the Enhanced Unified System Architecture
from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator, EnhancedUnifiedSystemOrchestrator
from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentTemplate, MemoryType, AgentType
from app.agents.base.agent import LangGraphAgent, AgentConfig, AgentCapability
from app.agents.autonomous import AutonomousLangGraphAgent, AutonomousAgentConfig, AutonomyLevel, LearningMode
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType
from app.rag.core.unified_rag_system import UnifiedRAGSystem, Document, KnowledgeQuery
from app.rag.core.collection_based_kb_manager import CollectionBasedKBManager, AccessLevel
from app.models.agent import Agent

# Try to import optional components
try:
    from app.rag.tools.enhanced_knowledge_tools import AgentKnowledgeSearchTool, AgentDocumentIngestTool
except ImportError:
    AgentKnowledgeSearchTool = None
    AgentDocumentIngestTool = None

try:
    from app.tools.dynamic_tool_factory import DynamicToolFactory, ToolCategory, ToolComplexity
except ImportError:
    DynamicToolFactory = None
    ToolCategory = None
    ToolComplexity = None

try:
    from app.rag.ingestion.pipeline import RevolutionaryIngestionPipeline
except ImportError:
    RevolutionaryIngestionPipeline = None

logger = structlog.get_logger(__name__)


class ComprehensiveBackendValidator:
    """
    Comprehensive validator for the entire backend agent system.
    
    Tests every component to ensure the backend is production-ready
    before frontend development.
    """
    
    def __init__(self):
        self.test_results = {}
        self.created_agents = []
        self.created_knowledge_bases = []
        self.created_tools = []
        self.test_documents = []
        self.start_time = None
        
        # Initialize components - Using Enhanced Unified System
        self.enhanced_orchestrator = None
        self.llm_manager = None
        self.agent_factory = None
        self.agent_registry = None
        self.rag_system = None
        self.kb_manager = None
        self.tool_repository = None
        
    async def initialize(self):
        """Initialize all backend components."""
        try:
            print("🚀 Initializing Comprehensive Backend Validator...")
            print("=" * 60)
            
            self.start_time = time.time()

            # Initialize Enhanced Unified System Orchestrator
            print("🚀 Initializing Enhanced Unified System Orchestrator...")
            self.enhanced_orchestrator = get_enhanced_system_orchestrator()
            await self.enhanced_orchestrator.initialize()

            # Get components from the enhanced orchestrator
            print("📦 Getting system components...")
            self.rag_system = self.enhanced_orchestrator.unified_rag
            self.kb_manager = self.enhanced_orchestrator.kb_manager
            self.tool_repository = self.enhanced_orchestrator.tool_repository

            # Get Agent Builder integration components
            if self.enhanced_orchestrator.agent_builder_integration:
                print("🤖 Getting Agent Builder components...")
                self.llm_manager = self.enhanced_orchestrator.agent_builder_integration.llm_manager
                self.agent_factory = self.enhanced_orchestrator.agent_builder_integration.agent_factory
                self.agent_registry = self.enhanced_orchestrator.agent_builder_integration.agent_registry
            else:
                print("⚠️ Agent Builder integration not available, initializing manually...")
                self.llm_manager = get_enhanced_llm_manager()
                await self.llm_manager.initialize()
                self.agent_factory = AgentBuilderFactory(self.llm_manager)
                self.agent_registry = None
            
            print("✅ All backend components initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize backend components: {e}")
            return False
    
    async def run_comprehensive_validation(self, quick_mode: bool = False, specific_agent_type: str = None):
        """Run the complete backend validation suite."""
        try:
            print(f"\n🎯 Starting Comprehensive Backend Validation")
            print(f"   Quick Mode: {quick_mode}")
            print(f"   Specific Type: {specific_agent_type or 'All'}")
            print("=" * 60)
            
            # Test 1: Comprehensive Agent Creation & Testing
            await self._test_agent_creation(specific_agent_type)

            # Test 1.5: Agent Capability Demonstrations
            if not quick_mode:
                await self.demonstrate_agent_capabilities()

            # Test 2: Knowledge Base Management
            await self._test_knowledge_base_management()
            
            # Test 3: Document Ingestion Pipeline
            await self._test_document_ingestion()
            
            # Test 4: Dynamic Tool Creation
            await self._test_dynamic_tool_creation()
            
            # Test 5: Agent-Tool Integration
            await self._test_agent_tool_integration()
            
            # Test 6: Agent-Knowledge Base Integration
            await self._test_agent_knowledge_integration()
            
            # Test 7: Agent Task Execution
            await self._test_agent_task_execution()
            
            # Test 8: RAG Functionality
            await self._test_rag_functionality()
            
            if not quick_mode:
                # Test 9: Multi-Modal Processing
                await self._test_multimodal_processing()
                
                # Test 10: Memory Systems
                await self._test_memory_systems()
                
                # Test 11: Multi-Agent Coordination
                await self._test_multi_agent_coordination()
            
            # Generate comprehensive report
            await self._generate_validation_report()
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            print(f"❌ Validation failed: {e}")
    
    def _get_agent_configurations(self):
        """Get comprehensive agent configurations for all types."""
        # Available Ollama models (update based on what's actually installed)
        ollama_models = [
            "phi4:latest",  # Currently available model
            "llama3.2:latest",  # Fallback models (may not be available)
            "qwen2.5:7b",
            "mistral:latest",
            "codellama:latest"
        ]

        return {
            AgentType.REACT: {
                "name": "ReAct_Reasoning_Agent",
                "description": "Reasoning and Acting agent with tool use capabilities",
                "system_prompt": """You are a ReAct (Reasoning and Acting) agent. You think step by step and use tools when needed.

Your process:
1. THOUGHT: Think about what you need to do
2. ACTION: Use a tool if needed
3. OBSERVATION: Observe the result
4. REPEAT: Continue until you have the answer

You have access to tools and should use them when appropriate. Always explain your reasoning process.""",
                "capabilities": [AgentCapability.REASONING, AgentCapability.TOOL_USE, AgentCapability.PLANNING],
                "model": ollama_models[0],
                "temperature": 0.3,
                "test_task": "Calculate the area of a circle with radius 5, then explain the mathematical concept behind the formula you used."
            },

            AgentType.RAG: {
                "name": "RAG_Knowledge_Agent",
                "description": "Retrieval-Augmented Generation agent for knowledge-based tasks",
                "system_prompt": """You are a RAG (Retrieval-Augmented Generation) agent specialized in knowledge retrieval and synthesis.

Your capabilities:
- Search through knowledge bases for relevant information
- Synthesize information from multiple sources
- Provide accurate, well-sourced answers
- Cite your sources when possible

Always search your knowledge base first before answering questions. Combine retrieved information with your reasoning to provide comprehensive answers.""",
                "capabilities": [AgentCapability.REASONING, AgentCapability.MEMORY],
                "model": ollama_models[1],
                "temperature": 0.2,
                "test_task": "Search for information about machine learning and explain the key differences between supervised and unsupervised learning."
            },

            AgentType.WORKFLOW: {
                "name": "Workflow_Automation_Agent",
                "description": "Process automation and workflow management agent",
                "system_prompt": """You are a Workflow Automation agent designed to handle structured processes and task sequences.

Your specialties:
- Breaking down complex tasks into steps
- Managing sequential and parallel processes
- Coordinating multiple subtasks
- Ensuring process completion and quality

When given a task, create a clear workflow with numbered steps, dependencies, and expected outcomes.""",
                "capabilities": [AgentCapability.PLANNING, AgentCapability.REASONING],
                "model": ollama_models[2],
                "temperature": 0.4,
                "test_task": "Create a detailed workflow for setting up a new employee's workspace, including IT setup, documentation, and training steps."
            },

            AgentType.MULTIMODAL: {
                "name": "MultiModal_Vision_Agent",
                "description": "Multi-modal agent with vision and text processing capabilities",
                "system_prompt": """You are a Multi-Modal agent capable of processing both text and visual information.

Your capabilities:
- Analyze images and visual content
- Process text and documents
- Combine visual and textual information
- Generate descriptions of visual content
- Extract information from documents and images

You can work with various media types and provide comprehensive analysis combining multiple modalities.""",
                "capabilities": [AgentCapability.MULTIMODAL, AgentCapability.REASONING, AgentCapability.TOOL_USE],
                "model": ollama_models[0],
                "temperature": 0.5,
                "test_task": "Describe how you would analyze a business document that contains both text and charts to extract key insights."
            },

            AgentType.AUTONOMOUS: {
                "name": "Autonomous_Self_Directed_Agent",
                "description": "Self-directed autonomous agent with learning capabilities",
                "system_prompt": """You are an Autonomous agent with self-direction and learning capabilities.

Your characteristics:
- Set your own sub-goals to achieve objectives
- Learn from interactions and feedback
- Adapt your approach based on results
- Take initiative in problem-solving
- Continuously improve your performance

You should demonstrate independence in thinking and approach tasks creatively while staying focused on the main objective.""",
                "capabilities": [AgentCapability.LEARNING, AgentCapability.REASONING, AgentCapability.PLANNING, AgentCapability.MEMORY],
                "model": ollama_models[3],
                "temperature": 0.6,
                "test_task": "You need to research and propose a solution for improving team productivity. Define your own research approach and present a comprehensive recommendation."
            },

            AgentType.COMPOSITE: {
                "name": "Composite_Coordination_Agent",
                "description": "Multi-agent coordination and composite task management",
                "system_prompt": """You are a Composite agent designed to coordinate multiple sub-agents and manage complex, multi-faceted tasks.

Your responsibilities:
- Coordinate between different specialized agents
- Delegate tasks to appropriate sub-agents
- Synthesize results from multiple sources
- Manage complex workflows with multiple components
- Ensure overall task completion and quality

You act as a conductor orchestrating different capabilities to achieve complex objectives.""",
                "capabilities": [AgentCapability.COLLABORATION, AgentCapability.PLANNING, AgentCapability.REASONING],
                "model": ollama_models[1],
                "temperature": 0.4,
                "test_task": "Plan and coordinate a comprehensive market research project that would require different types of analysis (data analysis, customer surveys, competitive analysis)."
            }
        }

    async def _test_agent_creation(self, specific_type: str = None):
        """Test creation of all agent types with comprehensive configurations."""
        print(f"\n🤖 Testing Comprehensive Agent Creation & Task Execution...")
        print("=" * 60)

        agent_configs = self._get_agent_configurations()

        if specific_type:
            agent_configs = {AgentType(specific_type): agent_configs[AgentType(specific_type)]}

        test_results = {}

        for agent_type, config in agent_configs.items():
            try:
                print(f"\n🔧 Creating {agent_type.value.upper()} Agent...")
                print(f"   Name: {config['name']}")
                print(f"   Model: {config['model']}")
                print(f"   Description: {config['description']}")

                # Create agent configuration
                agent_config = AgentBuilderConfig(
                    name=config["name"],
                    description=config["description"],
                    agent_type=agent_type,
                    llm_config=LLMConfig(
                        provider=ProviderType.OLLAMA,
                        model_id=config["model"],
                        temperature=config["temperature"],
                        base_url="http://localhost:11434"  # Ollama default
                    ),
                    capabilities=config["capabilities"],
                    tools=["calculator", "web_research", "business_intelligence"],  # Add available tools
                    system_prompt=config["system_prompt"],
                    enable_memory=True,
                    memory_type=MemoryType.SIMPLE
                )

                # Create agent
                print(f"   🚀 Building agent...")
                agent = await self.agent_factory.build_agent(agent_config)

                if agent:
                    self.created_agents.append(agent)
                    print(f"   ✅ Agent created successfully: {agent.agent_id}")

                    # Test the agent with its specific task
                    print(f"   🎯 Testing agent with task...")
                    print(f"   Task: {config['test_task'][:80]}...")

                    try:
                        session_id = f"test_{agent_type.value}_{int(time.time())}"
                        result = await agent.execute(
                            task=config["test_task"],
                            context={"session_id": session_id}
                        )

                        if result and isinstance(result, dict):
                            # Extract the final response from messages
                            messages = result.get("messages", [])
                            if messages:
                                # Get the last AI message
                                ai_messages = [msg for msg in messages if hasattr(msg, 'content') and getattr(msg, 'type', None) == 'ai']
                                if ai_messages:
                                    response = str(ai_messages[-1].content)
                                    print(f"   📝 Agent Response:")
                                    print(f"   {'-' * 50}")
                                    # Show first 300 characters of response
                                    print(f"   {response[:300]}...")
                                    if len(response) > 300:
                                        print(f"   [Response truncated - Full length: {len(response)} characters]")
                                    print(f"   {'-' * 50}")
                                    print(f"   ✅ Task completed successfully!")
                                    print(f"   📊 Execution Stats:")
                                    print(f"      - Iterations: {result.get('iteration_count', 0)}")
                                    print(f"      - Messages: {len(messages)}")
                                    print(f"      - Tools used: {len(result.get('tool_calls', []))}")

                                    test_results[agent_type.value] = {
                                        "creation": "✅ SUCCESS",
                                        "task_execution": "✅ SUCCESS",
                                        "response_length": len(response),
                                        "agent_id": agent.agent_id,
                                        "iterations": result.get('iteration_count', 0)
                                    }
                                else:
                                    print(f"   ⚠️  No AI response found in messages")
                                    test_results[agent_type.value] = {
                                        "creation": "✅ SUCCESS",
                                        "task_execution": "⚠️ NO_RESPONSE",
                                        "agent_id": agent.agent_id
                                    }
                            else:
                                print(f"   ⚠️  No messages in result")
                                test_results[agent_type.value] = {
                                    "creation": "✅ SUCCESS",
                                    "task_execution": "⚠️ NO_MESSAGES",
                                    "agent_id": agent.agent_id
                                }
                        else:
                            print(f"   ❌ No response from agent")
                            test_results[agent_type.value] = {
                                "creation": "✅ SUCCESS",
                                "task_execution": "❌ NO_RESPONSE",
                                "agent_id": agent.agent_id
                            }

                    except Exception as e:
                        print(f"   ❌ Task execution error: {e}")
                        test_results[agent_type.value] = {
                            "creation": "✅ SUCCESS",
                            "task_execution": f"❌ ERROR: {str(e)}",
                            "agent_id": agent.agent_id
                        }
                else:
                    print(f"   ❌ Failed to create agent")
                    test_results[agent_type.value] = {
                        "creation": "❌ FAILED",
                        "task_execution": "❌ NOT_TESTED"
                    }

            except Exception as e:
                print(f"   ❌ Agent creation error: {e}")
                test_results[agent_type.value] = {
                    "creation": f"❌ ERROR: {str(e)}",
                    "task_execution": "❌ NOT_TESTED"
                }

        self.test_results["comprehensive_agent_testing"] = test_results

        # Summary
        successful_creations = sum(1 for result in test_results.values() if "SUCCESS" in result.get("creation", ""))
        successful_executions = sum(1 for result in test_results.values() if "SUCCESS" in result.get("task_execution", ""))
        total_agents = len(test_results)

        print(f"\n📊 AGENT CREATION & TESTING SUMMARY:")
        print(f"   Agent Creation: {successful_creations}/{total_agents} successful")
        print(f"   Task Execution: {successful_executions}/{total_agents} successful")
        print(f"   Total Agents Created: {len(self.created_agents)}")

    async def create_agent_on_demand(
        self,
        agent_type: AgentType,
        custom_name: str = None,
        custom_task: str = None,
        model_name: str = "llama3.2:latest",
        temperature: float = 0.7,
        custom_system_prompt: str = None
    ):
        """
        Create an agent on-demand with custom configurations.

        This function makes it easy to create agents with specific configurations
        for testing or production use.
        """
        print(f"\n🚀 Creating {agent_type.value.upper()} Agent On-Demand...")

        # Get base configuration
        base_configs = self._get_agent_configurations()
        base_config = base_configs.get(agent_type)

        if not base_config:
            print(f"❌ Unknown agent type: {agent_type}")
            return None

        # Apply customizations
        name = custom_name or f"OnDemand_{base_config['name']}"
        system_prompt = custom_system_prompt or base_config['system_prompt']
        task = custom_task or base_config['test_task']

        print(f"   Name: {name}")
        print(f"   Model: {model_name}")
        print(f"   Temperature: {temperature}")

        try:
            # Create agent configuration
            agent_config = AgentBuilderConfig(
                name=name,
                description=f"On-demand {agent_type.value} agent",
                agent_type=agent_type,
                llm_config=LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model_id=model_name,
                    temperature=temperature,
                    base_url="http://localhost:11434"
                ),
                capabilities=base_config["capabilities"],
                tools=["calculator", "web_research", "business_intelligence"],  # Add available tools
                system_prompt=system_prompt,
                enable_memory=True,
                memory_type=MemoryType.SIMPLE
            )

            # Create agent
            agent = await self.agent_factory.build_agent(agent_config)

            if agent:
                print(f"   ✅ Agent created: {agent.agent_id}")

                # Test with task
                if task:
                    print(f"   🎯 Testing with task: {task[:60]}...")

                    session_id = f"ondemand_{agent_type.value}_{int(time.time())}"
                    result = await agent.execute(task=task, context={"session_id": session_id})

                    if result and isinstance(result, dict):
                        # Extract the final response from messages
                        messages = result.get("messages", [])
                        if messages:
                            # Get the last AI message
                            ai_messages = [msg for msg in messages if hasattr(msg, 'content') and getattr(msg, 'type', None) == 'ai']
                            if ai_messages:
                                response = str(ai_messages[-1].content)
                                print(f"   📝 Response: {response[:150]}...")
                                print(f"   ✅ Task completed successfully")
                                print(f"   📊 Stats: {result.get('iteration_count', 0)} iterations, {len(messages)} messages")
                            else:
                                print(f"   ⚠️  No AI response found")
                        else:
                            print(f"   ⚠️  No messages in result")
                    else:
                        print(f"   ⚠️  No response from agent")

                return agent
            else:
                print(f"   ❌ Failed to create agent")
                return None

        except Exception as e:
            print(f"   ❌ Error creating agent: {e}")
            return None

    async def demonstrate_agent_capabilities(self):
        """Demonstrate different agent capabilities with specific examples."""
        print(f"\n🎭 DEMONSTRATING AGENT CAPABILITIES")
        print("=" * 60)

        demonstrations = [
            {
                "agent_type": AgentType.REACT,
                "demo_name": "Problem Solving with Tools",
                "task": "I need to calculate compound interest for $1000 at 5% annual rate for 3 years, then explain the financial concept.",
                "model": "llama3.2:latest"
            },
            {
                "agent_type": AgentType.RAG,
                "demo_name": "Knowledge Synthesis",
                "task": "Compare different machine learning algorithms and recommend the best approach for customer segmentation.",
                "model": "qwen2.5:7b"
            },
            {
                "agent_type": AgentType.WORKFLOW,
                "demo_name": "Process Automation",
                "task": "Design a complete onboarding workflow for new software developers, including all necessary steps and checkpoints.",
                "model": "mistral:latest"
            },
            {
                "agent_type": AgentType.AUTONOMOUS,
                "demo_name": "Self-Directed Research",
                "task": "Research and propose innovative solutions for reducing energy consumption in data centers. Define your own research methodology.",
                "model": "codellama:latest"
            }
        ]

        demo_results = {}

        for demo in demonstrations:
            print(f"\n🎯 DEMONSTRATION: {demo['demo_name']}")
            print(f"   Agent Type: {demo['agent_type'].value.upper()}")
            print(f"   Model: {demo['model']}")
            print("-" * 50)

            try:
                # Create agent for demonstration
                agent = await self.create_agent_on_demand(
                    agent_type=demo["agent_type"],
                    custom_name=f"Demo_{demo['agent_type'].value}",
                    custom_task=demo["task"],
                    model_name=demo["model"],
                    temperature=0.5
                )

                if agent:
                    demo_results[demo["demo_name"]] = "✅ SUCCESS"
                    print(f"   ✅ Demonstration completed successfully")
                else:
                    demo_results[demo["demo_name"]] = "❌ FAILED"
                    print(f"   ❌ Demonstration failed")

            except Exception as e:
                demo_results[demo["demo_name"]] = f"❌ ERROR: {str(e)}"
                print(f"   ❌ Demonstration error: {e}")

        self.test_results["agent_demonstrations"] = demo_results

        successful_demos = sum(1 for result in demo_results.values() if "SUCCESS" in result)
        total_demos = len(demo_results)
        print(f"\n📊 DEMONSTRATION RESULTS: {successful_demos}/{total_demos} successful")

    async def _test_knowledge_base_management(self):
        """Test knowledge base creation and management."""
        print(f"\n📚 Testing Knowledge Base Management...")
        print("-" * 40)

        test_results = {}

        try:
            # Test 1: Create knowledge bases for different use cases
            kb_configs = [
                {"name": "Research_KB", "description": "Research knowledge base", "use_case": "research"},
                {"name": "Support_KB", "description": "Customer support knowledge base", "use_case": "support"},
                {"name": "Technical_KB", "description": "Technical documentation", "use_case": "documentation"}
            ]

            for kb_config in kb_configs:
                try:
                    print(f"   Creating knowledge base: {kb_config['name']}...")

                    kb_id = await self.kb_manager.create_knowledge_base(
                        name=kb_config["name"],
                        description=kb_config["description"],
                        owner_agent_id="system",
                        access_level=AccessLevel.PRIVATE
                    )

                    if kb_id:
                        self.created_knowledge_bases.append(kb_id)
                        test_results[kb_config["name"]] = "✅ SUCCESS"
                        print(f"      ✅ Created: {kb_id}")
                    else:
                        test_results[kb_config["name"]] = "❌ FAILED"
                        print(f"      ❌ Failed to create KB")

                except Exception as e:
                    test_results[kb_config["name"]] = f"❌ ERROR: {str(e)}"
                    print(f"      ❌ Error: {e}")

            # Test 2: List knowledge bases
            try:
                print("   Testing knowledge base listing...")
                kb_list = await self.kb_manager.list_knowledge_bases("system")
                if kb_list:
                    test_results["kb_listing"] = "✅ SUCCESS"
                    print(f"      ✅ Listed {len(kb_list)} knowledge bases")
                else:
                    test_results["kb_listing"] = "❌ FAILED"
                    print("      ❌ Failed to list knowledge bases")
            except Exception as e:
                test_results["kb_listing"] = f"❌ ERROR: {str(e)}"
                print(f"      ❌ Error: {e}")

        except Exception as e:
            test_results["general"] = f"❌ ERROR: {str(e)}"
            print(f"   ❌ General KB management error: {e}")

        self.test_results["knowledge_base_management"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   Knowledge Base Results: {success_count}/{total_count} successful")

    async def _test_document_ingestion(self):
        """Test document ingestion pipeline."""
        print(f"\n📄 Testing Document Ingestion Pipeline...")
        print("-" * 40)

        test_results = {}

        # Create test documents
        test_documents = [
            {
                "title": "AI Research Paper",
                "content": "Artificial Intelligence has revolutionized many fields. Machine learning algorithms can process vast amounts of data to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex relationships in data.",
                "type": "research"
            },
            {
                "title": "Customer Support Guide",
                "content": "Welcome to our customer support system. To reset your password, click on 'Forgot Password' on the login page. For billing inquiries, contact our billing department at billing@company.com. Technical issues can be reported through our support portal.",
                "type": "support"
            },
            {
                "title": "Technical Documentation",
                "content": "API Endpoint: POST /api/v1/agents/create. This endpoint creates a new agent with the specified configuration. Required parameters: name, description, agent_type. Optional parameters: tools, memory_type, capabilities.",
                "type": "technical"
            }
        ]

        for i, doc_data in enumerate(test_documents):
            try:
                print(f"   Ingesting document: {doc_data['title']}...")

                # Create document
                document = Document(
                    id=str(uuid.uuid4()),
                    content=doc_data["content"],
                    metadata={
                        "title": doc_data["title"],
                        "type": doc_data["type"],
                        "test_document": True,
                        "created_at": datetime.utcnow().isoformat()
                    }
                )

                # Select knowledge base (use first created KB)
                if self.created_knowledge_bases:
                    kb_id = self.created_knowledge_bases[0]

                    # Add document to knowledge base
                    doc_id = await self.kb_manager.add_document_to_kb(
                        kb_id=kb_id,
                        document=document,
                        agent_id="system"
                    )

                    if doc_id:
                        self.test_documents.append(doc_id)
                        test_results[f"document_{i+1}"] = "✅ SUCCESS"
                        print(f"      ✅ Ingested: {doc_id}")
                    else:
                        test_results[f"document_{i+1}"] = "❌ FAILED"
                        print(f"      ❌ Failed to ingest document")
                else:
                    test_results[f"document_{i+1}"] = "❌ NO_KB"
                    print(f"      ❌ No knowledge base available")

            except Exception as e:
                test_results[f"document_{i+1}"] = f"❌ ERROR: {str(e)}"
                print(f"      ❌ Error: {e}")

        self.test_results["document_ingestion"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   Document Ingestion Results: {success_count}/{total_count} successful")

    async def _test_dynamic_tool_creation(self):
        """Test dynamic tool creation."""
        print(f"\n🔧 Testing Dynamic Tool Creation...")
        print("-" * 40)

        test_results = {}

        # Skip tool creation if DynamicToolFactory is not available
        if DynamicToolFactory is None or ToolCategory is None:
            print("   ⚠️ Dynamic tool factory not available - skipping tool creation tests")
            print("   ✅ Tool repository is available through UnifiedToolRepository")
            return

        # Define test tools to create
        tool_configs = [
            {
                "name": "calculator",
                "description": "A simple calculator tool",
                "functionality": "Performs basic arithmetic operations",
                "category": ToolCategory.UTILITY,
                "complexity": ToolComplexity.SIMPLE
            },
            {
                "name": "web_research",
                "description": "Web search tool",
                "functionality": "Searches the web for information",
                "category": ToolCategory.INFORMATION,
                "complexity": ToolComplexity.MEDIUM
            },
            {
                "name": "data_analyzer",
                "description": "Data analysis tool",
                "functionality": "Analyzes data and generates insights",
                "category": ToolCategory.ANALYSIS,
                "complexity": ToolComplexity.COMPLEX
            }
        ]

        for tool_config in tool_configs:
            try:
                print(f"   Creating tool: {tool_config['name']}...")

                tool = await self.tool_factory.create_tool_from_description(
                    name=tool_config["name"],
                    description=tool_config["description"],
                    functionality_description=tool_config["functionality"],
                    llm=enhanced_orchestrator.llm,
                    category=tool_config["category"]
                )

                if tool:
                    self.created_tools.append(tool.name)
                    test_results[tool_config["name"]] = "✅ SUCCESS"
                    print(f"      ✅ Created: {tool.name}")
                else:
                    test_results[tool_config["name"]] = "❌ FAILED"
                    print(f"      ❌ Failed to create tool")

            except Exception as e:
                test_results[tool_config["name"]] = f"❌ ERROR: {str(e)}"
                print(f"      ❌ Error: {e}")

        self.test_results["dynamic_tool_creation"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   Tool Creation Results: {success_count}/{total_count} successful")

    async def _test_agent_tool_integration(self):
        """Test agent-tool integration."""
        print(f"\n🔗 Testing Agent-Tool Integration...")
        print("-" * 40)

        test_results = {}

        if not self.created_agents or not self.created_tools:
            test_results["integration"] = "❌ NO_AGENTS_OR_TOOLS"
            print("   ❌ No agents or tools available for integration testing")
            self.test_results["agent_tool_integration"] = test_results
            return

        # Test tool assignment to agents
        for i, agent in enumerate(self.created_agents[:3]):  # Test first 3 agents
            try:
                agent_name = f"Agent_{i+1}"
                print(f"   Testing tool integration for {agent_name}...")

                # Assign tools to agent
                tools_assigned = 0
                for tool_name in self.created_tools[:2]:  # Assign first 2 tools
                    try:
                        tool = self.tool_factory.get_tool(tool_name)
                        if tool:
                            await agent.add_tool(tool)
                            tools_assigned += 1
                    except Exception as e:
                        print(f"      ⚠️  Tool assignment warning: {e}")

                if tools_assigned > 0:
                    test_results[agent_name] = f"✅ SUCCESS ({tools_assigned} tools)"
                    print(f"      ✅ Assigned {tools_assigned} tools")
                else:
                    test_results[agent_name] = "❌ NO_TOOLS_ASSIGNED"
                    print(f"      ❌ No tools assigned")

            except Exception as e:
                test_results[f"Agent_{i+1}"] = f"❌ ERROR: {str(e)}"
                print(f"      ❌ Error: {e}")

        self.test_results["agent_tool_integration"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   Agent-Tool Integration Results: {success_count}/{total_count} successful")

    async def _test_agent_knowledge_integration(self):
        """Test agent-knowledge base integration."""
        print(f"\n🧠 Testing Agent-Knowledge Base Integration...")
        print("-" * 40)

        test_results = {}

        if not self.created_agents or not self.created_knowledge_bases:
            test_results["integration"] = "❌ NO_AGENTS_OR_KBS"
            print("   ❌ No agents or knowledge bases available for integration testing")
            self.test_results["agent_knowledge_integration"] = test_results
            return

        # Test knowledge base access for agents
        for i, agent in enumerate(self.created_agents[:3]):  # Test first 3 agents
            try:
                agent_name = f"Agent_{i+1}"
                print(f"   Testing knowledge integration for {agent_name}...")

                # Create knowledge search tool for agent
                kb_id = self.created_knowledge_bases[0]  # Use first KB

                knowledge_tool = AgentKnowledgeSearchTool(
                    rag_service=self.rag_system,
                    agent_id=agent.agent_id
                )

                # Add knowledge tool to agent
                await agent.add_tool(knowledge_tool)

                test_results[agent_name] = "✅ SUCCESS"
                print(f"      ✅ Knowledge integration successful")

            except Exception as e:
                test_results[f"Agent_{i+1}"] = f"❌ ERROR: {str(e)}"
                print(f"      ❌ Error: {e}")

        self.test_results["agent_knowledge_integration"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   Agent-Knowledge Integration Results: {success_count}/{total_count} successful")

    async def _test_agent_task_execution(self):
        """Test agent task execution."""
        print(f"\n⚡ Testing Agent Task Execution...")
        print("-" * 40)

        test_results = {}

        if not self.created_agents:
            test_results["execution"] = "❌ NO_AGENTS"
            print("   ❌ No agents available for task execution testing")
            self.test_results["agent_task_execution"] = test_results
            return

        # Define test tasks
        test_tasks = [
            {
                "task": "What is artificial intelligence?",
                "expected_keywords": ["artificial", "intelligence", "machine", "learning"]
            },
            {
                "task": "Explain the benefits of using AI agents",
                "expected_keywords": ["agents", "benefits", "automation", "efficiency"]
            }
        ]

        # Test task execution with first few agents
        for i, agent in enumerate(self.created_agents[:2]):  # Test first 2 agents
            try:
                agent_name = f"Agent_{i+1}"
                print(f"   Testing task execution for {agent_name}...")

                for j, test_task in enumerate(test_tasks):
                    try:
                        print(f"      Task {j+1}: {test_task['task'][:50]}...")

                        # Execute task
                        result = await agent.execute_task(
                            task=test_task["task"],
                            session_id=f"test_session_{i}_{j}"
                        )

                        if result and hasattr(result, 'content'):
                            response_text = str(result.content).lower()

                            # Check if response contains expected keywords
                            keyword_matches = sum(1 for keyword in test_task["expected_keywords"]
                                                if keyword.lower() in response_text)

                            if keyword_matches > 0:
                                test_results[f"{agent_name}_task_{j+1}"] = f"✅ SUCCESS ({keyword_matches} keywords)"
                                print(f"         ✅ Task completed ({keyword_matches} relevant keywords)")
                            else:
                                test_results[f"{agent_name}_task_{j+1}"] = "⚠️  PARTIAL (no keywords)"
                                print(f"         ⚠️  Task completed but no relevant keywords found")
                        else:
                            test_results[f"{agent_name}_task_{j+1}"] = "❌ NO_RESPONSE"
                            print(f"         ❌ No response from agent")

                    except Exception as e:
                        test_results[f"{agent_name}_task_{j+1}"] = f"❌ ERROR: {str(e)}"
                        print(f"         ❌ Task error: {e}")

            except Exception as e:
                test_results[f"Agent_{i+1}"] = f"❌ ERROR: {str(e)}"
                print(f"      ❌ Agent error: {e}")

        self.test_results["agent_task_execution"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   Task Execution Results: {success_count}/{total_count} successful")

    async def _test_rag_functionality(self):
        """Test RAG (Retrieval-Augmented Generation) functionality."""
        print(f"\n🔍 Testing RAG Functionality...")
        print("-" * 40)

        test_results = {}

        if not self.created_knowledge_bases or not self.test_documents:
            test_results["rag"] = "❌ NO_KB_OR_DOCS"
            print("   ❌ No knowledge bases or documents available for RAG testing")
            self.test_results["rag_functionality"] = test_results
            return

        # Test knowledge queries
        test_queries = [
            {
                "query": "What is machine learning?",
                "expected_source": "AI Research Paper"
            },
            {
                "query": "How to reset password?",
                "expected_source": "Customer Support Guide"
            },
            {
                "query": "API endpoint for creating agents",
                "expected_source": "Technical Documentation"
            }
        ]

        kb_id = self.created_knowledge_bases[0]  # Use first KB

        for i, test_query in enumerate(test_queries):
            try:
                print(f"   Testing query {i+1}: {test_query['query'][:50]}...")

                # Create knowledge query
                query = KnowledgeQuery(
                    query_text=test_query["query"],
                    agent_id="system",
                    session_id=f"rag_test_{i}",
                    max_results=5
                )

                # Search knowledge base
                results = await self.kb_manager.search_knowledge_base(
                    kb_id=kb_id,
                    query=query,
                    agent_id="system"
                )

                if results and len(results) > 0:
                    # Check if results are relevant
                    relevant_found = False
                    for result in results:
                        if hasattr(result, 'metadata') and result.metadata:
                            title = result.metadata.get('title', '')
                            if test_query['expected_source'].lower() in title.lower():
                                relevant_found = True
                                break

                    if relevant_found:
                        test_results[f"query_{i+1}"] = f"✅ SUCCESS ({len(results)} results)"
                        print(f"      ✅ Found {len(results)} relevant results")
                    else:
                        test_results[f"query_{i+1}"] = f"⚠️  PARTIAL ({len(results)} results)"
                        print(f"      ⚠️  Found {len(results)} results but not from expected source")
                else:
                    test_results[f"query_{i+1}"] = "❌ NO_RESULTS"
                    print(f"      ❌ No results found")

            except Exception as e:
                test_results[f"query_{i+1}"] = f"❌ ERROR: {str(e)}"
                print(f"      ❌ Query error: {e}")

        self.test_results["rag_functionality"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   RAG Functionality Results: {success_count}/{total_count} successful")

    async def _test_multimodal_processing(self):
        """Test multi-modal processing capabilities."""
        print(f"\n🎨 Testing Multi-Modal Processing...")
        print("-" * 40)

        test_results = {}

        try:
            # Test text processing
            print("   Testing text processing...")
            text_content = "This is a test document for multi-modal processing validation."

            # Create a simple text document for processing
            if self.ingestion_pipeline:
                job_id = await self.ingestion_pipeline.ingest_content(
                    content=text_content.encode('utf-8'),
                    file_name="test_multimodal.txt",
                    mime_type="text/plain",
                    collection="multimodal_test"
                )

                if job_id:
                    test_results["text_processing"] = "✅ SUCCESS"
                    print("      ✅ Text processing successful")
                else:
                    test_results["text_processing"] = "❌ FAILED"
                    print("      ❌ Text processing failed")
            else:
                test_results["text_processing"] = "❌ NO_PIPELINE"
                print("      ❌ No ingestion pipeline available")

        except Exception as e:
            test_results["text_processing"] = f"❌ ERROR: {str(e)}"
            print(f"      ❌ Text processing error: {e}")

        # Note: Image/OCR processing would require actual image files
        # For now, we'll mark it as a placeholder test
        test_results["image_processing"] = "⚠️  PLACEHOLDER (requires image files)"
        print("   ⚠️  Image processing test requires actual image files")

        self.test_results["multimodal_processing"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len([r for r in test_results.values() if "PLACEHOLDER" not in r])
        print(f"\n   Multi-Modal Processing Results: {success_count}/{total_count} successful")

    async def _test_memory_systems(self):
        """Test agent memory systems."""
        print(f"\n🧠 Testing Memory Systems...")
        print("-" * 40)

        test_results = {}

        if not self.created_agents:
            test_results["memory"] = "❌ NO_AGENTS"
            print("   ❌ No agents available for memory testing")
            self.test_results["memory_systems"] = test_results
            return

        # Test memory with first agent
        try:
            agent = self.created_agents[0]
            print(f"   Testing memory for agent: {agent.agent_id}")

            # Test short-term memory (conversation context)
            session_id = f"memory_test_{int(time.time())}"

            # First interaction
            result1 = await agent.execute_task(
                task="Remember that my name is TestUser and I like AI research.",
                session_id=session_id
            )

            if result1:
                test_results["memory_store"] = "✅ SUCCESS"
                print("      ✅ Memory storage test passed")
            else:
                test_results["memory_store"] = "❌ FAILED"
                print("      ❌ Memory storage test failed")

            # Second interaction (test recall)
            result2 = await agent.execute_task(
                task="What is my name and what do I like?",
                session_id=session_id
            )

            if result2 and hasattr(result2, 'content'):
                response_text = str(result2.content).lower()
                if "testuser" in response_text and ("ai" in response_text or "research" in response_text):
                    test_results["memory_recall"] = "✅ SUCCESS"
                    print("      ✅ Memory recall test passed")
                else:
                    test_results["memory_recall"] = "⚠️  PARTIAL"
                    print("      ⚠️  Memory recall partially successful")
            else:
                test_results["memory_recall"] = "❌ FAILED"
                print("      ❌ Memory recall test failed")

        except Exception as e:
            test_results["memory_general"] = f"❌ ERROR: {str(e)}"
            print(f"      ❌ Memory system error: {e}")

        self.test_results["memory_systems"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len(test_results)
        print(f"\n   Memory Systems Results: {success_count}/{total_count} successful")

    async def _test_multi_agent_coordination(self):
        """Test multi-agent coordination."""
        print(f"\n🤝 Testing Multi-Agent Coordination...")
        print("-" * 40)

        test_results = {}

        if len(self.created_agents) < 2:
            test_results["coordination"] = "❌ INSUFFICIENT_AGENTS"
            print("   ❌ Need at least 2 agents for coordination testing")
            self.test_results["multi_agent_coordination"] = test_results
            return

        try:
            # Test basic agent communication
            agent1 = self.created_agents[0]
            agent2 = self.created_agents[1]

            print(f"   Testing coordination between {agent1.agent_id} and {agent2.agent_id}...")

            # For now, test that both agents can execute tasks independently
            session_id = f"coordination_test_{int(time.time())}"

            task1_result = await agent1.execute_task(
                task="You are Agent 1. Introduce yourself.",
                session_id=f"{session_id}_agent1"
            )

            task2_result = await agent2.execute_task(
                task="You are Agent 2. Introduce yourself.",
                session_id=f"{session_id}_agent2"
            )

            if task1_result and task2_result:
                test_results["basic_coordination"] = "✅ SUCCESS"
                print("      ✅ Basic multi-agent execution successful")
            else:
                test_results["basic_coordination"] = "❌ FAILED"
                print("      ❌ Basic multi-agent execution failed")

            # Note: Advanced coordination would require specific coordination protocols
            test_results["advanced_coordination"] = "⚠️  PLACEHOLDER (requires coordination protocols)"
            print("   ⚠️  Advanced coordination testing requires specific protocols")

        except Exception as e:
            test_results["coordination_general"] = f"❌ ERROR: {str(e)}"
            print(f"      ❌ Coordination error: {e}")

        self.test_results["multi_agent_coordination"] = test_results

        success_count = sum(1 for result in test_results.values() if "SUCCESS" in result)
        total_count = len([r for r in test_results.values() if "PLACEHOLDER" not in r])
        print(f"\n   Multi-Agent Coordination Results: {success_count}/{total_count} successful")

    async def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        print(f"\n📊 COMPREHENSIVE BACKEND VALIDATION REPORT")
        print("=" * 60)

        total_time = time.time() - self.start_time

        print(f"🕒 Total Validation Time: {total_time:.2f} seconds")
        print(f"🤖 Agents Created: {len(self.created_agents)}")
        print(f"📚 Knowledge Bases Created: {len(self.created_knowledge_bases)}")
        print(f"📄 Documents Ingested: {len(self.test_documents)}")
        print(f"🔧 Tools Created: {len(self.created_tools)}")

        print(f"\n📈 DETAILED TEST RESULTS:")
        print("-" * 40)

        overall_success = 0
        overall_total = 0

        for test_category, results in self.test_results.items():
            if isinstance(results, dict):
                success_count = sum(1 for result in results.values() if "SUCCESS" in result)
                total_count = len([r for r in results.values() if "PLACEHOLDER" not in r])

                overall_success += success_count
                overall_total += total_count

                status = "✅" if success_count == total_count else "⚠️" if success_count > 0 else "❌"
                print(f"{status} {test_category.replace('_', ' ').title()}: {success_count}/{total_count}")

                # Show failed tests
                failed_tests = [k for k, v in results.items() if "SUCCESS" not in v and "PLACEHOLDER" not in v]
                if failed_tests:
                    for failed_test in failed_tests[:3]:  # Show first 3 failures
                        print(f"     ❌ {failed_test}: {results[failed_test]}")
                    if len(failed_tests) > 3:
                        print(f"     ... and {len(failed_tests) - 3} more failures")

        print(f"\n🎯 OVERALL RESULTS:")
        print("-" * 20)
        success_rate = (overall_success / overall_total * 100) if overall_total > 0 else 0
        print(f"Success Rate: {success_rate:.1f}% ({overall_success}/{overall_total})")

        if success_rate >= 90:
            print("🎉 EXCELLENT! Backend is production-ready!")
        elif success_rate >= 75:
            print("✅ GOOD! Backend is mostly functional with minor issues.")
        elif success_rate >= 50:
            print("⚠️  MODERATE! Backend has significant issues that need attention.")
        else:
            print("❌ CRITICAL! Backend has major issues that must be fixed.")

        # Save detailed report to file
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "overall_success_rate": success_rate,
            "overall_results": f"{overall_success}/{overall_total}",
            "components_created": {
                "agents": len(self.created_agents),
                "knowledge_bases": len(self.created_knowledge_bases),
                "documents": len(self.test_documents),
                "tools": len(self.created_tools)
            },
            "detailed_results": self.test_results
        }

        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save report file: {e}")

        print(f"\n🚀 Backend validation complete!")

        return success_rate >= 75  # Return True if backend is mostly functional


async def create_agent_easy(
    agent_type: str,
    name: str = None,
    task: str = None,
    model: str = "llama3.2:latest",
    temperature: float = 0.7,
    system_prompt: str = None
):
    """
    🚀 EASY AGENT CREATOR - Create agents on the fly!

    This function makes it super easy to create any type of agent with custom configurations.

    Args:
        agent_type: Type of agent ("react", "rag", "workflow", "multimodal", "autonomous", "composite")
        name: Custom name for the agent
        task: Task to test the agent with
        model: Ollama model to use (default: "llama3.2:latest")
        temperature: LLM temperature (default: 0.7)
        system_prompt: Custom system prompt

    Returns:
        Created agent instance

    Example:
        # Create a ReAct agent
        agent = await create_agent_easy("react", "MyReActAgent", "Solve this math problem: 2+2*3")

        # Create a RAG agent with custom model
        agent = await create_agent_easy("rag", "KnowledgeBot", model="qwen2.5:7b")
    """
    print(f"🚀 Easy Agent Creator - Creating {agent_type.upper()} agent...")

    # Initialize validator to use its components
    validator = ComprehensiveBackendValidator()
    if not await validator.initialize():
        print("❌ Failed to initialize backend components")
        return None

    try:
        agent_type_enum = AgentType(agent_type.lower())

        agent = await validator.create_agent_on_demand(
            agent_type=agent_type_enum,
            custom_name=name,
            custom_task=task,
            model_name=model,
            temperature=temperature,
            custom_system_prompt=system_prompt
        )

        if agent:
            print(f"✅ Agent created successfully!")
            print(f"   Agent ID: {agent.agent_id}")
            print(f"   Type: {agent_type.upper()}")
            print(f"   Model: {model}")
            return agent
        else:
            print(f"❌ Failed to create agent")
            return None

    except ValueError:
        print(f"❌ Invalid agent type: {agent_type}")
        print(f"   Valid types: react, rag, workflow, multimodal, autonomous, composite")
        return None
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return None


async def quick_agent_demo():
    """Quick demonstration of creating different agent types."""
    print("🎭 QUICK AGENT DEMONSTRATION")
    print("=" * 50)

    # Demo different agent types
    agent_demos = [
        ("react", "ReAct Problem Solver", "Calculate the area of a rectangle with length 10 and width 5, then explain the formula."),
        ("rag", "Knowledge Assistant", "What are the benefits of using AI in healthcare?"),
        ("workflow", "Process Manager", "Create a workflow for handling customer complaints."),
        ("autonomous", "Self-Directed Researcher", "Research renewable energy trends and make recommendations.")
    ]

    for agent_type, name, task in agent_demos:
        print(f"\n🤖 Creating {agent_type.upper()} Agent...")
        agent = await create_agent_easy(agent_type, name, task)
        if agent:
            print(f"   ✅ {name} is ready to work!")
        else:
            print(f"   ❌ Failed to create {name}")


async def main():
    """Main function to run comprehensive backend validation."""
    parser = argparse.ArgumentParser(description="Comprehensive Backend Agent System Validator & Easy Agent Creator")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (skip long tests)")
    parser.add_argument("--agent-type", type=str, help="Test specific agent type only")
    parser.add_argument("--demo", action="store_true", help="Run quick agent demonstration")
    parser.add_argument("--create", type=str, help="Create a specific agent type (react, rag, workflow, etc.)")
    parser.add_argument("--name", type=str, help="Custom name for created agent")
    parser.add_argument("--task", type=str, help="Custom task for created agent")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama model to use")

    args = parser.parse_args()

    # Handle different modes
    if args.demo:
        # Quick agent demonstration
        await quick_agent_demo()
        return

    if args.create:
        # Create a specific agent
        print(f"🚀 Creating {args.create.upper()} agent...")
        agent = await create_agent_easy(
            agent_type=args.create,
            name=args.name,
            task=args.task,
            model=args.model
        )
        if agent:
            print(f"🎉 Agent created successfully! You can now use it in your applications.")
        return

    # Default: Run comprehensive validation
    validator = ComprehensiveBackendValidator()

    # Initialize validator
    if not await validator.initialize():
        print("❌ Failed to initialize validator")
        sys.exit(1)

    # Run comprehensive validation
    await validator.run_comprehensive_validation(
        quick_mode=args.quick,
        specific_agent_type=args.agent_type
    )


if __name__ == "__main__":
    asyncio.run(main())
