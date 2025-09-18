"""
Comprehensive Integration Test for Unified Multi-Agent System.

This test suite validates the complete unified architecture including:
- Unified RAG system with collection-based isolation
- Unified memory system with agent-specific collections
- Unified tool repository with access controls
- Agent communication and collaboration systems
- Performance optimization and monitoring
- Advanced access controls and security

Test Categories:
1. System Initialization and Health
2. Agent Lifecycle Management
3. Knowledge Base Operations
4. Memory System Operations
5. Tool Repository Operations
6. Communication System Operations
7. Performance and Monitoring
8. Security and Access Controls
9. End-to-End Workflows
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import unified system components
from app.core.unified_system_orchestrator import (
    UnifiedSystemOrchestrator,
    SystemConfig,
    get_system_orchestrator,
    shutdown_system
)
from app.rag.core.unified_rag_system import UnifiedRAGConfig
from app.rag.core.unified_memory_system import UnifiedMemoryConfig, MemoryType
from app.communication.agent_communication_system import CommunicationConfig
from app.optimization.performance_optimizer import OptimizationConfig
from app.optimization.advanced_access_controls import AccessAction, PolicyEffect


class TestUnifiedSystemIntegration:
    """Comprehensive integration tests for the unified multi-agent system."""
    
    @pytest.fixture(scope="class")
    async def system_orchestrator(self):
        """Create and initialize the unified system orchestrator."""
        config = SystemConfig(
            rag_config=UnifiedRAGConfig(
                chroma_host="localhost",
                chroma_port=8000,
                enable_connection_pooling=True,
                max_connections=10
            ),
            memory_config=UnifiedMemoryConfig(
                enable_memory_consolidation=True,
                consolidation_interval_hours=1,
                max_memories_per_agent=1000
            ),
            communication_config=CommunicationConfig(
                enable_message_persistence=True,
                max_message_history=1000
            ),
            optimization_config=OptimizationConfig(
                strategy="balanced",
                enable_intelligent_caching=True
            ),
            enable_monitoring=True,
            enable_optimization=True,
            enable_security=True
        )
        
        orchestrator = UnifiedSystemOrchestrator(config)
        await orchestrator.initialize()
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, system_orchestrator):
        """Test system initialization and health check."""
        # Verify system is initialized
        assert system_orchestrator.status.is_initialized
        assert system_orchestrator.status.is_running
        assert system_orchestrator.status.health_score > 0
        
        # Verify all core components are initialized
        required_components = [
            "unified_rag", "isolation_manager", "memory_system",
            "kb_manager", "memory_collections", "tool_repository",
            "communication_system", "knowledge_sharing", "collaboration_manager"
        ]
        
        for component in required_components:
            assert system_orchestrator.status.components_status.get(component, False), f"Component {component} not initialized"
        
        # Get system status
        status = system_orchestrator.get_system_status()
        assert status["is_initialized"]
        assert status["is_running"]
        assert status["health_score"] > 0
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_management(self, system_orchestrator):
        """Test complete agent lifecycle from creation to cleanup."""
        test_agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
        
        try:
            # 1. Create agent isolation profile
            isolation_profile = await system_orchestrator.isolation_manager.create_agent_isolation(test_agent_id)
            assert isolation_profile.agent_id == test_agent_id
            
            # 2. Create agent memory
            agent_memory = await system_orchestrator.memory_system.create_agent_memory(test_agent_id)
            assert agent_memory is not None
            
            # 3. Create agent tool profile
            tool_profile = await system_orchestrator.tool_repository.create_agent_profile(test_agent_id)
            assert tool_profile.agent_id == test_agent_id
            
            # 4. Register agent for communication
            await system_orchestrator.communication_system.register_agent(test_agent_id)
            
            # 5. Verify agent is fully set up
            agent_stats = system_orchestrator.unified_rag.get_agent_stats(test_agent_id)
            assert agent_stats is not None
            
        finally:
            # Cleanup agent (in a real implementation, we'd have cleanup methods)
            pass
    
    @pytest.mark.asyncio
    async def test_knowledge_base_operations(self, system_orchestrator):
        """Test knowledge base operations with collection isolation."""
        test_agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
        
        # Create agent
        await system_orchestrator.isolation_manager.create_agent_isolation(test_agent_id)
        
        # Create knowledge base
        kb_info = await system_orchestrator.kb_manager.create_knowledge_base(
            agent_id=test_agent_id,
            kb_name="test_knowledge_base",
            description="Test knowledge base for integration testing"
        )
        assert kb_info.name == "test_knowledge_base"
        assert kb_info.owner_agent_id == test_agent_id
        
        # Add documents to knowledge base
        test_documents = [
            {
                "content": "This is a test document about artificial intelligence.",
                "metadata": {"source": "test", "category": "ai"}
            },
            {
                "content": "This document discusses machine learning algorithms.",
                "metadata": {"source": "test", "category": "ml"}
            }
        ]
        
        for doc in test_documents:
            await system_orchestrator.kb_manager.add_document(
                agent_id=test_agent_id,
                kb_name="test_knowledge_base",
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # Query knowledge base
        results = await system_orchestrator.kb_manager.query_knowledge_base(
            agent_id=test_agent_id,
            kb_name="test_knowledge_base",
            query="artificial intelligence",
            top_k=5
        )
        assert len(results) > 0
        assert any("artificial intelligence" in result.get("content", "").lower() for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_system_operations(self, system_orchestrator):
        """Test unified memory system operations."""
        test_agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
        
        # Create agent memory
        await system_orchestrator.memory_system.create_agent_memory(test_agent_id)
        
        # Store different types of memories
        memories = [
            {
                "content": "I learned about neural networks today",
                "memory_type": MemoryType.EPISODIC,
                "metadata": {"topic": "learning", "importance": 0.8}
            },
            {
                "content": "Python is a programming language",
                "memory_type": MemoryType.SEMANTIC,
                "metadata": {"topic": "knowledge", "importance": 0.9}
            },
            {
                "content": "How to debug code effectively",
                "memory_type": MemoryType.PROCEDURAL,
                "metadata": {"topic": "skills", "importance": 0.7}
            }
        ]
        
        memory_ids = []
        for memory in memories:
            memory_id = await system_orchestrator.memory_system.store_memory(
                agent_id=test_agent_id,
                content=memory["content"],
                memory_type=memory["memory_type"],
                metadata=memory["metadata"]
            )
            memory_ids.append(memory_id)
            assert memory_id is not None
        
        # Retrieve memories
        retrieved_memories = await system_orchestrator.memory_system.retrieve_memories(
            agent_id=test_agent_id,
            query="learning programming",
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            limit=10
        )
        assert len(retrieved_memories) > 0
        
        # Test memory consolidation
        await system_orchestrator.memory_system.consolidate_memories(test_agent_id)
    
    @pytest.mark.asyncio
    async def test_tool_repository_operations(self, system_orchestrator):
        """Test unified tool repository operations."""
        test_agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
        
        # Create agent tool profile
        tool_profile = await system_orchestrator.tool_repository.create_agent_profile(test_agent_id)
        
        # Get available tools for agent
        available_tools = await system_orchestrator.tool_repository.get_available_tools(test_agent_id)
        assert len(available_tools) > 0
        
        # Assign tools to agent
        tool_assignments = await system_orchestrator.tool_repository.assign_tools_to_agent(
            agent_id=test_agent_id,
            tool_names=["calculator_tool", "business_intelligence_tool"]
        )
        assert len(tool_assignments) > 0
        
        # Get agent's assigned tools
        agent_tools = await system_orchestrator.tool_repository.get_agent_tools(test_agent_id)
        assert len(agent_tools) > 0
    
    @pytest.mark.asyncio
    async def test_communication_system_operations(self, system_orchestrator):
        """Test agent communication system operations."""
        agent1_id = f"test_agent_1_{uuid.uuid4().hex[:8]}"
        agent2_id = f"test_agent_2_{uuid.uuid4().hex[:8]}"
        
        # Register agents
        await system_orchestrator.communication_system.register_agent(agent1_id)
        await system_orchestrator.communication_system.register_agent(agent2_id)
        
        # Send message from agent1 to agent2
        message_id = await system_orchestrator.communication_system.send_message(
            sender_id=agent1_id,
            recipient_id=agent2_id,
            content="Hello from agent 1!",
            message_type="text"
        )
        assert message_id is not None
        
        # Retrieve messages for agent2
        messages = await system_orchestrator.communication_system.get_messages(
            agent_id=agent2_id,
            limit=10
        )
        assert len(messages) > 0
        assert any(msg.content == "Hello from agent 1!" for msg in messages)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, system_orchestrator):
        """Test performance optimization and monitoring."""
        if system_orchestrator.performance_optimizer:
            # Collect performance metrics
            metrics = await system_orchestrator.performance_optimizer.collect_metrics()
            assert metrics.cpu_usage >= 0
            assert metrics.memory_usage >= 0
            
            # Run performance optimization
            optimizations = await system_orchestrator.performance_optimizer.optimize_system_performance()
            assert isinstance(optimizations, list)
            
            # Get performance report
            report = system_orchestrator.performance_optimizer.get_performance_report()
            assert "current_metrics" in report
        
        if system_orchestrator.monitoring_system:
            # Generate performance report
            report = await system_orchestrator.monitoring_system.generate_performance_report(
                period_hours=1,
                include_recommendations=True
            )
            assert report.system_health_score >= 0
            assert report.system_health_score <= 100
    
    @pytest.mark.asyncio
    async def test_access_controls(self, system_orchestrator):
        """Test advanced access control system."""
        if system_orchestrator.access_controller:
            test_agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
            test_resource_id = f"test_resource_{uuid.uuid4().hex[:8]}"
            
            # Assign role to agent
            success = system_orchestrator.access_controller.assign_role_to_agent(test_agent_id, "user")
            assert success
            
            # Test access evaluation
            decision, applied_rules = await system_orchestrator.access_controller.evaluate_access(
                subject_id=test_agent_id,
                resource_id=test_resource_id,
                action=AccessAction.READ
            )
            assert decision in [PolicyEffect.ALLOW, PolicyEffect.DENY]
            
            # Get access statistics
            stats = system_orchestrator.access_controller.get_access_stats()
            assert "total_evaluations" in stats
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, system_orchestrator):
        """Test complete end-to-end workflow."""
        test_agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
        
        # 1. Create agent with full setup
        await system_orchestrator.isolation_manager.create_agent_isolation(test_agent_id)
        await system_orchestrator.memory_system.create_agent_memory(test_agent_id)
        await system_orchestrator.tool_repository.create_agent_profile(test_agent_id)
        await system_orchestrator.communication_system.register_agent(test_agent_id)
        
        # 2. Create knowledge base and add content
        kb_info = await system_orchestrator.kb_manager.create_knowledge_base(
            agent_id=test_agent_id,
            kb_name="workflow_test_kb",
            description="Knowledge base for workflow testing"
        )
        
        await system_orchestrator.kb_manager.add_document(
            agent_id=test_agent_id,
            kb_name="workflow_test_kb",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "workflow_test", "category": "ai"}
        )
        
        # 3. Store memory about the learning
        memory_id = await system_orchestrator.memory_system.store_memory(
            agent_id=test_agent_id,
            content="I added information about machine learning to my knowledge base",
            memory_type=MemoryType.EPISODIC,
            metadata={"action": "knowledge_addition", "importance": 0.8}
        )
        
        # 4. Query knowledge base
        kb_results = await system_orchestrator.kb_manager.query_knowledge_base(
            agent_id=test_agent_id,
            kb_name="workflow_test_kb",
            query="machine learning",
            top_k=5
        )
        
        # 5. Retrieve related memories
        memories = await system_orchestrator.memory_system.retrieve_memories(
            agent_id=test_agent_id,
            query="knowledge base machine learning",
            memory_types=[MemoryType.EPISODIC],
            limit=5
        )
        
        # 6. Verify workflow completed successfully
        assert len(kb_results) > 0
        assert len(memories) > 0
        assert any("machine learning" in result.get("content", "").lower() for result in kb_results)
        assert any("knowledge base" in memory.content.lower() for memory in memories)
    
    @pytest.mark.asyncio
    async def test_system_health_and_stats(self, system_orchestrator):
        """Test system health monitoring and statistics."""
        # Get comprehensive system status
        status = system_orchestrator.get_system_status()
        
        # Verify status structure
        assert "is_initialized" in status
        assert "is_running" in status
        assert "health_score" in status
        assert "components_status" in status
        assert "uptime_seconds" in status
        
        # Verify health score is reasonable
        assert 0 <= status["health_score"] <= 100
        
        # Verify uptime is positive
        assert status["uptime_seconds"] > 0
        
        # Get component-specific stats
        if system_orchestrator.unified_rag:
            rag_stats = system_orchestrator.unified_rag.get_system_stats()
            assert isinstance(rag_stats, dict)
        
        if system_orchestrator.memory_system:
            memory_stats = system_orchestrator.memory_system.get_system_stats()
            assert isinstance(memory_stats, dict)
        
        if system_orchestrator.tool_repository:
            tool_stats = system_orchestrator.tool_repository.get_repository_stats()
            assert isinstance(tool_stats, dict)
        
        if system_orchestrator.communication_system:
            comm_stats = system_orchestrator.communication_system.get_system_stats()
            assert isinstance(comm_stats, dict)


# Utility functions for testing
async def create_test_agent(orchestrator: UnifiedSystemOrchestrator, agent_id: str) -> str:
    """Create a fully configured test agent."""
    await orchestrator.isolation_manager.create_agent_isolation(agent_id)
    await orchestrator.memory_system.create_agent_memory(agent_id)
    await orchestrator.tool_repository.create_agent_profile(agent_id)
    await orchestrator.communication_system.register_agent(agent_id)
    return agent_id


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
