#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        UNIVERSAL AGENT TEMPLATE                              ║
║                     Production-Ready Agent Shell                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 PURPOSE: Universal Python template that works with ANY YAML configuration
🎯 USAGE: Copy this file, change AGENT_ID, customize as needed
✅ YAML-DRIVEN: All configuration comes from the YAML file
🔧 MINIMAL: This is just a shell - YAML controls everything

═══════════════════════════════════════════════════════════════════════════════
QUICK START:
═══════════════════════════════════════════════════════════════════════════════
1. Copy this file: cp agent_template.py → your_agent.py
2. Copy YAML template: cp universal_agent_template.yaml → your_agent.yaml
3. Change AGENT_ID below to match your YAML's agent_id field
4. Customize YAML configuration (agent type, tools, personality, etc.)
5. Run: python data/agents/your_agent.py

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE DOES:
═══════════════════════════════════════════════════════════════════════════════
- Initializes the unified system orchestrator
- Loads agent configuration from YAML
- Creates the appropriate agent type (ReAct, Autonomous, RAG, etc.)
- Provides execution methods (execute_task, interactive_session)
- Handles cleanup and error management

═══════════════════════════════════════════════════════════════════════════════
WHAT THE YAML FILE DOES:
═══════════════════════════════════════════════════════════════════════════════
- Defines agent type (react, autonomous, rag, workflow, etc.)
- Configures LLM settings (provider, model, temperature, etc.)
- Specifies tools and capabilities
- Sets personality and communication style
- Defines system prompts and behavior
- Configures memory, RAG, performance settings
- Everything else!

═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
from app.agents.factory import AgentBuilderFactory
from app.backend_logging import get_logger, LogCategory

# Initialize logger
logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ⚠️ CHANGE THIS: Must match agent_id in your YAML file
AGENT_ID = "my_agent_template"

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class AgentTemplate:
    """
    Universal agent template that works with any YAML configuration.
    
    This class is a minimal shell - all configuration comes from the YAML file.
    The AgentBuilderFactory reads the YAML and creates the appropriate agent type
    (ReAct, Autonomous, RAG, etc.) with all specified tools, memory, and settings.
    """
    
    def __init__(self):
        """Initialize the agent template."""
        self.agent_id = AGENT_ID
        self.orchestrator = None
        self.agent = None
        self.is_initialized = False
        
        logger.info(
            f"Agent template created: {self.agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "agent_template"
        )
    
    async def initialize(self) -> bool:
        """
        Initialize the agent and all required systems.

        This method:
        1. Ensures system is ready (models downloaded, etc.)
        2. Initializes the unified system orchestrator
        3. Loads agent configuration from YAML
        4. Creates the agent using the factory (which reads YAML)
        5. Sets up memory, RAG, and tool systems

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info(
                f"Initializing agent: {self.agent_id}",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )

            # Ensure system is fully initialized (models downloaded, etc.)
            from app.core.system_initialization import ensure_system_ready

            print(f"🔍 Checking system initialization for agent: {self.agent_id}")
            system_ready = await ensure_system_ready(silent=True)

            if not system_ready:
                logger.warning(
                    "System initialization incomplete - some features may be unavailable",
                    LogCategory.AGENT_OPERATIONS,
                    "agent_template"
                )
                print("⚠️  Warning: Some models may be unavailable")
            else:
                print("✅ System ready")

            # Initialize the unified system orchestrator
            self.orchestrator = get_enhanced_system_orchestrator()
            await self.orchestrator.initialize()
            
            logger.info(
                "System orchestrator initialized",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )
            
            # Get LLM manager from orchestrator
            llm_manager = self.orchestrator.agent_builder_integration.llm_manager
            
            # Create agent factory
            factory = AgentBuilderFactory(
                llm_manager=llm_manager,
                memory_system=self.orchestrator.memory_system
            )
            
            # Build agent from YAML configuration
            # This is where the magic happens - factory reads YAML and creates
            # the appropriate agent type with all specified configuration
            logger.info(
                f"Building agent from YAML: {self.agent_id}",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )
            
            self.agent = await factory.build_agent_from_yaml(self.agent_id)
            
            logger.info(
                f"Agent built successfully: {type(self.agent).__name__}",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )
            
            # Initialize agent-specific memory
            await self.orchestrator.memory_system.create_agent_memory(self.agent_id)
            
            # Initialize agent-specific RAG ecosystem (if RAG enabled in YAML)
            await self.orchestrator.unified_rag.create_agent_ecosystem(self.agent_id)
            
            # Initialize agent-specific tool profile
            await self.orchestrator.tool_repository.create_agent_profile(self.agent_id)
            
            self.is_initialized = True
            
            logger.info(
                f"Agent initialization complete: {self.agent_id}",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to initialize agent: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "agent_template",
                data={"error": str(e), "agent_id": self.agent_id}
            )
            return False
    
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single task.
        
        Args:
            task: The task description or user query
            context: Optional context dictionary
            
        Returns:
            Dict containing the execution result
        """
        if not self.is_initialized:
            logger.info(
                "Agent not initialized, initializing now...",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )
            success = await self.initialize()
            if not success:
                return {
                    "success": False,
                    "error": "Failed to initialize agent",
                    "timestamp": datetime.now().isoformat()
                }
        
        try:
            logger.info(
                f"Executing task: {task[:100]}...",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )
            
            # Prepare context
            execution_context = context or {}
            execution_context.update({
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "orchestrator": self.orchestrator
            })
            
            # Execute the task using the agent
            result = await self.agent.execute(
                task=task,
                context=execution_context
            )
            
            logger.info(
                "Task execution complete",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Task execution failed: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "agent_template",
                data={"error": str(e), "task": task[:100]}
            )
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def interactive_session(self):
        """
        Run an interactive session with the agent.
        
        This method provides a simple command-line interface for interacting
        with the agent. Type 'exit', 'quit', or 'q' to end the session.
        """
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                print("❌ Failed to initialize agent. Exiting.")
                return
        
        print("\n" + "═" * 80)
        print(f"🤖 {self.agent_id.upper()} - Interactive Session")
        print("═" * 80)
        print("Type your queries below. Type 'exit', 'quit', or 'q' to end the session.")
        print("═" * 80 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Ending session. Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Execute the task
                print(f"\n🤖 {self.agent_id}: Processing...")
                result = await self.execute_task(user_input)
                
                # Display result
                if result.get("success", False):
                    print(f"\n✅ Result:\n{result.get('output', result)}")
                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logger.error(
                    f"Interactive session error: {str(e)}",
                    LogCategory.AGENT_OPERATIONS,
                    "agent_template"
                )
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.orchestrator:
                # Perform any necessary cleanup
                logger.info(
                    "Cleaning up agent resources",
                    LogCategory.AGENT_OPERATIONS,
                    "agent_template"
                )
            self.is_initialized = False
        except Exception as e:
            logger.error(
                f"Cleanup error: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "agent_template"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Main entry point for the agent."""
    agent = AgentTemplate()
    
    try:
        # Run interactive session
        await agent.interactive_session()
    finally:
        # Cleanup
        await agent.cleanup()


if __name__ == "__main__":
    # Run the agent
    asyncio.run(main())

