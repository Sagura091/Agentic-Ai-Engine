#!/usr/bin/env python3
"""
🚀 UNIVERSAL DOCUMENT MASTER AGENT
===================================
Interactive ReAct agent for document creation and manipulation.
Uses Revolutionary Universal Tools (Excel, Word, PDF) with complete power-user capabilities.

INTERACTIVE REACT AGENT:
🧠 Uses LLM reasoning to understand document requests
🛠️ Dynamically selects tools (Excel, Word, PDF) based on needs
🎯 Makes autonomous decisions about document creation
📊 Adapts approach based on document type and requirements
🔄 Responds to user queries interactively

ARCHITECTURE:
- ReAct Agent with interactive mode
- Dynamic tool selection from UnifiedToolRepository
- LLM-driven document planning and execution
- Tool-based document generation (Excel, Word, PDF)
- Learning from user interactions
"""

import os
import sys
import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure clean logging BEFORE importing anything else
# This suppresses all the noisy initialization logs
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(message)s",  # Clean format, no timestamps
    stream=sys.stdout
)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Import the complete production infrastructure
from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
from app.agents.factory import AgentBuilderFactory

# Import Revolutionary Logging System
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LoggingMode
from app.core.clean_logging import get_conversation_logger
import structlog

# Configure structlog for clean output (USER mode)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        # Minimal console output - just the message, no formatting
        structlog.processors.KeyValueRenderer(key_order=["event"]),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Set structlog root logger to WARNING to suppress initialization spam
logging.getLogger().setLevel(logging.WARNING)

# Agent ID - matches YAML configuration
AGENT_ID = "universal_document_master_agent"

# Initialize loggers
backend_logger = get_logger()
conversation_logger = get_conversation_logger(AGENT_ID)
structlog_logger = structlog.get_logger(__name__)


class UniversalDocumentMasterAgent:
    """
    🚀 Universal Document Master Agent

    Interactive ReAct agent that uses autonomous reasoning and dynamic tool selection
    to create and manipulate Excel, Word, and PDF documents.

    REACT BEHAVIORS:
    🧠 Uses LLM reasoning to analyze document requirements
    🛠️ Dynamically selects tools from UnifiedToolRepository based on needs
    🎯 Makes autonomous decisions about document creation strategies
    📊 Adapts methodology based on document type and context
    🔄 Responds to user queries interactively
    """

    def __init__(self):
        """Initialize the Universal Document Master Agent."""
        self.agent_id = AGENT_ID
        self.orchestrator = None
        self.agent = None
        self.llm_manager = None

        # Agent state
        self.is_initialized = False
        self.operation_history = []

        # Loggers - already configured globally, just use them!
        self.conversation_logger = conversation_logger
        self.backend_logger = backend_logger

        # Use conversation logger for user-facing output
        self.conversation_logger.info(f"Universal Document Master Agent created with ID: {self.agent_id}")

    async def initialize(self) -> bool:
        """Initialize the agent with ReAct capabilities."""
        try:
            # Use conversation logger for user-facing messages
            self.conversation_logger.agent_acknowledgment("Initializing Universal Document Master Agent...")

            # Backend log for technical tracking
            self.backend_logger.info(
                "Agent initialization started",
                LogCategory.AGENT_OPERATIONS,
                "UniversalDocumentMasterAgent",
                data={"agent_id": self.agent_id}
            )

            # Get the enhanced system orchestrator
            self.orchestrator = get_enhanced_system_orchestrator()
            if not self.orchestrator:
                self.conversation_logger.error("Failed to get system orchestrator")
                self.backend_logger.error(
                    "System orchestrator initialization failed",
                    LogCategory.AGENT_OPERATIONS,
                    "UniversalDocumentMasterAgent"
                )
                return False

            # Initialize the orchestrator if not already initialized
            if not self.orchestrator.is_initialized:
                await self.orchestrator.initialize()

            self.conversation_logger.success("System orchestrator connected and initialized")
            self.backend_logger.info(
                "System orchestrator initialized",
                LogCategory.AGENT_OPERATIONS,
                "UniversalDocumentMasterAgent"
            )

            # Get LLM manager from agent builder integration
            if hasattr(self.orchestrator, 'agent_builder_integration') and self.orchestrator.agent_builder_integration:
                self.llm_manager = self.orchestrator.agent_builder_integration.llm_manager
            else:
                # Fallback to creating our own ENHANCED LLM manager
                from app.llm.manager import get_enhanced_llm_manager
                self.llm_manager = get_enhanced_llm_manager()
                if not self.llm_manager.is_initialized():
                    await self.llm_manager.initialize()

            if not self.llm_manager:
                print("❌ Failed to get LLM manager")
                return False

            print("   ✅ LLM manager connected")

            # Try to build agent from YAML configuration
            try:
                from app.agents.factory import AgentBuilderFactory

                # Create agent factory
                factory = AgentBuilderFactory(self.llm_manager, self.orchestrator.memory_system)

                # Build agent from YAML configuration
                self.agent = await factory.build_agent_from_yaml(self.agent_id)

                print("   ✅ Agent built from YAML configuration")
                print(f"   ✅ Agent type: {self.agent.config.agent_type if hasattr(self.agent, 'config') else 'react'}")

            except Exception as yaml_error:
                print(f"   ⚠️ YAML config failed: {yaml_error}")
                import traceback
                traceback.print_exc()
                return False

            if not self.agent:
                print("❌ Failed to create agent")
                return False

            # Initialize agent memory and RAG
            await self.orchestrator.memory_system.create_agent_memory(self.agent_id)
            print("   🧠 Memory system initialized")

            # Create RAG ecosystem if enabled
            try:
                await self.orchestrator.unified_rag.create_agent_ecosystem(self.agent_id)
                print("   📚 Knowledge base created")
            except Exception as e:
                print(f"   ⚠️ RAG initialization warning: {str(e)}")

            # Create tool profile for the agent
            await self.orchestrator.tool_repository.create_agent_profile(self.agent_id)
            print("   🛠️ Tool profile created for dynamic tool selection")

            self.is_initialized = True
            print("✅ Universal Document Master Agent initialized successfully!")
            return True

        except Exception as e:
            self.conversation_logger.error(f"Failed to initialize agent: {str(e)}")
            self.backend_logger.error(
                f"Agent initialization failed: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "UniversalDocumentMasterAgent"
            )
            import traceback
            traceback.print_exc()
            return False

    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and execute document operations.

        This is the main interactive method - user asks, agent responds and acts.
        """
        try:
            if not self.agent:
                print("❌ Agent not initialized")
                return {"status": "error", "error": "Agent not initialized"}

            print(f"\n{'='*80}")
            print(f"📝 USER QUERY: {query}")
            print(f"{'='*80}\n")

            print("🧠 Sending query to ReAct agent for reasoning and execution...")

            # Execute the query using the ReAct agent
            result = await self.agent.execute(
                task=query,
                context={"session_id": f"doc_query_{uuid.uuid4().hex[:8]}"}
            )

            print("\n✅ Agent execution completed!")

            # Store in operation history
            self.operation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'result': result
            })

            return {
                "status": "success",
                "agent_result": result,
                "query": query,
                "execution_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.conversation_logger.error(f"Query processing failed: {str(e)}")
            self.backend_logger.error(
                "Query processing failed",
                LogCategory.AGENT_OPERATIONS,
                "UniversalDocumentMasterAgent",
                data={"error": str(e), "query": query}
            )
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def interactive_session(self):
        """Run an interactive session with the agent."""
        try:
            print("\n" + "="*80)
            print("🚀 UNIVERSAL DOCUMENT MASTER AGENT - INTERACTIVE SESSION")
            print("="*80)
            print("🧠 I can create Excel, Word, and PDF documents with power-user capabilities!")
            print("💡 Ask me to create any document, and I'll make it happen!")
            print()
            print("Commands:")
            print("  - Type your request (e.g., 'Create an Excel file with sales data')")
            print("  - Type 'history' to see operation history")
            print("  - Type 'help' for examples")
            print("  - Type 'exit' or 'quit' to end session")
            print("="*80)
            print()

            # Initialize if not already done
            if not self.is_initialized:
                print("🔧 Initializing agent...")
                if not await self.initialize():
                    print("❌ Failed to initialize agent. Exiting.")
                    return
                print("✅ Agent initialized successfully!")
                print()

            # Interactive loop
            while True:
                try:
                    # Get user input
                    user_input = input("\n📝 You: ").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        print("\n👋 Goodbye! Thanks for using Universal Document Master Agent!")
                        break

                    elif user_input.lower() == 'history':
                        print("\n📜 OPERATION HISTORY:")
                        print("-" * 80)
                        if self.operation_history:
                            for i, op in enumerate(self.operation_history[-10:], 1):
                                print(f"{i}. [{op['timestamp']}]")
                                print(f"   Query: {op['query']}")
                                print()
                        else:
                            print("No operations yet.")
                        print("-" * 80)
                        continue

                    elif user_input.lower() == 'help':
                        print("\n💡 EXAMPLE REQUESTS:")
                        print("-" * 80)
                        print("Excel:")
                        print("  • Create an Excel file with monthly sales data")
                        print("  • Make a spreadsheet with formulas for budget calculations")
                        print()
                        print("Word:")
                        print("  • Create a professional business letter")
                        print("  • Make a report with headings and tables")
                        print()
                        print("PDF:")
                        print("  • Create a PDF document")
                        print("  • Merge multiple PDF files")
                        print("-" * 80)
                        continue

                    # Process the query
                    result = await self.process_user_query(user_input)

                    if result["status"] != "success":
                        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"\n🤖 Agent Result: {result['agent_result']}")

                except KeyboardInterrupt:
                    print("\n\n👋 Session interrupted. Goodbye!")
                    break
                except Exception as e:
                    self.conversation_logger.error(f"Error: {str(e)}")
                    self.backend_logger.error(
                        f"Interactive session error: {str(e)}",
                        LogCategory.AGENT_OPERATIONS,
                        "UniversalDocumentMasterAgent"
                    )

        except Exception as e:
            self.conversation_logger.error(f"Interactive session failed: {str(e)}")
            self.backend_logger.error(
                f"Interactive session failed: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "UniversalDocumentMasterAgent"
            )


# ================================
# 🚀 MAIN LAUNCHER
# ================================

async def main():
    """Main launcher for the Universal Document Master Agent."""
    print("🤖 UNIVERSAL DOCUMENT MASTER AGENT")
    print("=" * 80)
    print("🧠 INTERACTIVE REACT AGENT WITH LLM REASONING AND DYNAMIC TOOL SELECTION")
    print()

    # Create and initialize agent
    agent = UniversalDocumentMasterAgent()

    # Run interactive session
    await agent.interactive_session()


if __name__ == "__main__":
    asyncio.run(main())
