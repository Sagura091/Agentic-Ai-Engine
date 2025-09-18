"""
Performance and Integration Test Suite for Revolutionary RAG System.

Tests system performance, scalability, and integration with:
- Multi-agent concurrent operations
- Large-scale knowledge management
- Memory performance and caching
- Real-world usage scenarios
- Agent collaboration workflows
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import statistics

from app.rag.core.enhanced_rag_service import EnhancedRAGService
from app.rag.core.agent_knowledge_manager import KnowledgeScope, KnowledgePermission
from app.rag.core.knowledge_base import Document
from app.rag.tools.enhanced_knowledge_tools import (
    EnhancedKnowledgeSearchTool,
    AgentDocumentIngestTool,
    AgentMemoryTool
)


class TestPerformanceAndIntegration:
    """Performance and integration tests for the revolutionary RAG system."""
    
    @pytest.fixture
    async def performance_rag_service(self):
        """Create RAG service optimized for performance testing."""
        temp_dir = tempfile.mkdtemp()
        service = EnhancedRAGService()
        service.settings.chroma_persist_directory = temp_dir
        await service.initialize()
        yield service
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self, performance_rag_service):
        """Test concurrent operations across multiple agents."""
        service = performance_rag_service
        num_agents = 10
        operations_per_agent = 5
        
        async def agent_workflow(agent_id: str) -> Dict[str, Any]:
            """Simulate typical agent workflow."""
            start_time = time.time()
            
            # Create agent
            manager = await service.get_or_create_agent_manager(
                agent_id, "performance_test"
            )
            
            # Add documents
            for i in range(operations_per_agent):
                doc = Document(
                    title=f"Document {i} for {agent_id}",
                    content=f"This is test content {i} for agent {agent_id} with unique information",
                    metadata={"agent": agent_id, "doc_num": i}
                )
                await service.add_document(agent_id, doc, KnowledgeScope.PRIVATE)
            
            # Add memories
            for i in range(operations_per_agent):
                await service.add_memory(
                    agent_id,
                    f"Memory {i}: Learned something important about task {i}",
                    importance=0.5 + (i * 0.1),
                    tags=[f"task_{i}", "learning"]
                )
            
            # Perform searches
            search_results = []
            for i in range(operations_per_agent):
                result = await service.search_knowledge(
                    agent_id,
                    f"test content {i}",
                    top_k=5,
                    include_memories=True
                )
                search_results.append(len(result.results))
            
            end_time = time.time()
            
            return {
                "agent_id": agent_id,
                "execution_time": end_time - start_time,
                "documents_added": operations_per_agent,
                "memories_added": operations_per_agent,
                "searches_performed": operations_per_agent,
                "avg_search_results": statistics.mean(search_results)
            }
        
        # Run concurrent agent workflows
        start_time = time.time()
        
        tasks = [
            agent_workflow(f"perf_agent_{i:03d}")
            for i in range(num_agents)
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze performance results
        execution_times = [r["execution_time"] for r in results]
        avg_execution_time = statistics.mean(execution_times)
        max_execution_time = max(execution_times)
        
        # Performance assertions
        assert len(results) == num_agents
        assert avg_execution_time < 30.0  # Should complete within 30 seconds per agent
        assert max_execution_time < 60.0  # No agent should take more than 60 seconds
        assert total_time < 120.0  # Total concurrent execution under 2 minutes
        
        # Verify all operations completed successfully
        total_docs = sum(r["documents_added"] for r in results)
        total_memories = sum(r["memories_added"] for r in results)
        total_searches = sum(r["searches_performed"] for r in results)
        
        assert total_docs == num_agents * operations_per_agent
        assert total_memories == num_agents * operations_per_agent
        assert total_searches == num_agents * operations_per_agent
        
        # Verify service statistics
        stats = service.get_service_stats()
        assert stats["active_agents"] == num_agents
        assert stats["total_queries"] >= total_searches
        assert stats["total_documents"] >= total_docs
        
        print(f"\n🚀 Performance Test Results:")
        print(f"   Agents: {num_agents}")
        print(f"   Operations per agent: {operations_per_agent}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg agent time: {avg_execution_time:.2f}s")
        print(f"   Max agent time: {max_execution_time:.2f}s")
        print(f"   Total documents: {total_docs}")
        print(f"   Total memories: {total_memories}")
        print(f"   Total searches: {total_searches}")
    
    @pytest.mark.asyncio
    async def test_large_scale_knowledge_management(self, performance_rag_service):
        """Test handling large amounts of knowledge across agents."""
        service = performance_rag_service
        
        # Create agent for large-scale testing
        agent_id = "large_scale_agent"
        manager = await service.get_or_create_agent_manager(agent_id, "research")
        
        # Add large number of documents
        num_documents = 100
        document_size = 1000  # characters
        
        start_time = time.time()
        
        for i in range(num_documents):
            # Create document with substantial content
            content = f"Document {i}: " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
            content += f"Unique identifier: doc_{i}_unique_content_marker"
            
            doc = Document(
                title=f"Large Scale Document {i}",
                content=content,
                metadata={
                    "doc_id": i,
                    "category": f"category_{i % 10}",
                    "size": len(content)
                }
            )
            
            await service.add_document(agent_id, doc, KnowledgeScope.PRIVATE)
        
        ingestion_time = time.time() - start_time
        
        # Test search performance across large knowledge base
        search_start = time.time()
        
        # Perform various search queries
        search_queries = [
            "Lorem ipsum dolor",
            "unique identifier",
            f"doc_{num_documents//2}_unique",
            "category_5",
            "Large Scale Document"
        ]
        
        search_results = []
        for query in search_queries:
            result = await service.search_knowledge(
                agent_id,
                query,
                top_k=10,
                scopes=[KnowledgeScope.PRIVATE]
            )
            search_results.append({
                "query": query,
                "results_count": len(result.results),
                "processing_time": result.processing_time
            })
        
        search_time = time.time() - search_start
        
        # Performance assertions
        assert ingestion_time < 60.0  # Should ingest 100 docs within 60 seconds
        assert search_time < 10.0  # Should complete 5 searches within 10 seconds
        
        # Verify search quality
        for result in search_results:
            assert result["results_count"] > 0  # All queries should return results
        
        # Test specific document retrieval
        specific_search = await service.search_knowledge(
            agent_id,
            "doc_50_unique_content_marker",
            top_k=5
        )
        
        assert len(specific_search.results) > 0
        found_doc_50 = any(
            "doc_50_unique" in result.content 
            for result in specific_search.results
        )
        assert found_doc_50
        
        print(f"\n📊 Large Scale Test Results:")
        print(f"   Documents ingested: {num_documents}")
        print(f"   Ingestion time: {ingestion_time:.2f}s")
        print(f"   Search time: {search_time:.2f}s")
        print(f"   Avg search results: {statistics.mean([r['results_count'] for r in search_results]):.1f}")
    
    @pytest.mark.asyncio
    async def test_memory_performance_and_caching(self, performance_rag_service):
        """Test memory system performance and caching effectiveness."""
        service = performance_rag_service
        agent_id = "memory_perf_agent"
        
        # Create agent
        manager = await service.get_or_create_agent_manager(agent_id, "general")
        
        # Add large number of memories
        num_memories = 200
        memory_types = ["episodic", "semantic", "procedural"]
        
        start_time = time.time()
        
        for i in range(num_memories):
            memory_type = memory_types[i % len(memory_types)]
            importance = 0.1 + (i % 10) * 0.1  # Vary importance
            
            await service.add_memory(
                agent_id,
                f"Memory {i}: Important information about {memory_type} learning process {i}",
                memory_type=memory_type,
                importance=importance,
                context={"session": f"session_{i//10}", "topic": f"topic_{i%5}"},
                tags=[f"tag_{i%3}", memory_type, "performance_test"]
            )
        
        memory_creation_time = time.time() - start_time
        
        # Test memory search performance
        search_start = time.time()
        
        # First search (cache miss)
        result1 = await service.search_knowledge(
            agent_id,
            "important information about episodic",
            include_memories=True,
            top_k=20
        )
        
        first_search_time = time.time() - search_start
        
        # Second identical search (should hit cache)
        cache_start = time.time()
        result2 = await service.search_knowledge(
            agent_id,
            "important information about episodic",
            include_memories=True,
            top_k=20
        )
        cached_search_time = time.time() - cache_start
        
        # Test memory filtering by importance
        high_importance_result = await service.search_knowledge(
            agent_id,
            "learning process",
            include_memories=True,
            top_k=50
        )
        
        # Analyze results
        memory_results_1 = [r for r in result1.results if r.metadata.get("type") == "memory"]
        memory_results_2 = [r for r in result2.results if r.metadata.get("type") == "memory"]
        high_imp_memories = [
            r for r in high_importance_result.results 
            if r.metadata.get("type") == "memory" and r.metadata.get("importance", 0) > 0.7
        ]
        
        # Performance assertions
        assert memory_creation_time < 30.0  # Should create 200 memories within 30 seconds
        assert first_search_time < 5.0  # First search should complete within 5 seconds
        assert cached_search_time < first_search_time  # Cached search should be faster
        
        # Verify memory search quality
        assert len(memory_results_1) > 0
        assert len(memory_results_2) == len(memory_results_1)  # Same results from cache
        assert len(high_imp_memories) > 0  # Should find high importance memories
        
        # Verify importance-based ranking
        if len(high_imp_memories) > 1:
            importances = [r.metadata.get("importance", 0) for r in high_imp_memories[:5]]
            assert importances == sorted(importances, reverse=True)  # Should be sorted by importance
        
        print(f"\n🧠 Memory Performance Results:")
        print(f"   Memories created: {num_memories}")
        print(f"   Creation time: {memory_creation_time:.2f}s")
        print(f"   First search time: {first_search_time:.2f}s")
        print(f"   Cached search time: {cached_search_time:.2f}s")
        print(f"   Cache speedup: {first_search_time/cached_search_time:.1f}x")
        print(f"   Memory results found: {len(memory_results_1)}")
        print(f"   High importance memories: {len(high_imp_memories)}")
    
    @pytest.mark.asyncio
    async def test_agent_collaboration_workflow(self, performance_rag_service):
        """Test realistic agent collaboration scenarios."""
        service = performance_rag_service
        
        # Create collaborative agents
        research_agent = "research_collab_001"
        creative_agent = "creative_collab_001"
        technical_agent = "technical_collab_001"
        
        # Initialize agents
        await service.get_or_create_agent_manager(research_agent, "research")
        await service.get_or_create_agent_manager(creative_agent, "creative")
        await service.get_or_create_agent_manager(technical_agent, "technical")
        
        # Scenario: Research agent gathers information
        research_docs = [
            Document(
                title="Market Research Findings",
                content="User surveys indicate strong demand for AI-powered productivity tools",
                metadata={"phase": "research", "confidence": 0.9}
            ),
            Document(
                title="Competitive Analysis",
                content="Existing solutions lack personalization and multi-agent capabilities",
                metadata={"phase": "research", "competitive": True}
            )
        ]
        
        for doc in research_docs:
            await service.add_document(research_agent, doc, KnowledgeScope.DOMAIN)
        
        # Research agent creates insights
        await service.add_memory(
            research_agent,
            "Key insight: Users want personalized AI that learns from their behavior",
            memory_type="semantic",
            importance=0.9,
            tags=["insight", "user_needs", "personalization"]
        )
        
        # Creative agent searches for research insights
        creative_search = await service.search_knowledge(
            creative_agent,
            "user demand AI productivity personalization",
            scopes=[KnowledgeScope.DOMAIN, KnowledgeScope.GLOBAL],
            top_k=10
        )
        
        # Creative agent builds on research
        creative_doc = Document(
            title="Product Concept: Personalized AI Assistant",
            content="Based on research findings, propose AI assistant that adapts to user behavior and preferences",
            metadata={"phase": "ideation", "based_on": "research_findings"}
        )
        
        await service.add_document(creative_agent, creative_doc, KnowledgeScope.DOMAIN)
        
        # Technical agent searches for both research and creative work
        technical_search = await service.search_knowledge(
            technical_agent,
            "AI assistant personalization user behavior",
            scopes=[KnowledgeScope.DOMAIN],
            top_k=15
        )
        
        # Technical agent creates implementation plan
        technical_doc = Document(
            title="Technical Architecture for Personalized AI",
            content="Implementation plan: Use machine learning for behavior analysis and adaptive responses",
            metadata={"phase": "technical_design", "implements": "creative_concept"}
        )
        
        await service.add_document(technical_agent, technical_doc, KnowledgeScope.DOMAIN)
        
        # Verify collaboration effectiveness
        assert len(creative_search.results) > 0
        assert len(technical_search.results) > 0
        
        # Verify knowledge flow between agents
        research_content_found_by_creative = any(
            "User surveys indicate" in result.content 
            for result in creative_search.results
        )
        
        creative_content_found_by_technical = any(
            "Personalized AI Assistant" in result.content 
            for result in technical_search.results
        )
        
        assert research_content_found_by_creative
        assert creative_content_found_by_technical
        
        # Test final integration search
        integration_search = await service.search_knowledge(
            technical_agent,
            "productivity tools personalization implementation",
            scopes=[KnowledgeScope.DOMAIN],
            top_k=20
        )
        
        # Should find content from all phases
        phases_found = set()
        for result in integration_search.results:
            phase = result.metadata.get("phase")
            if phase:
                phases_found.add(phase)
        
        expected_phases = {"research", "ideation", "technical_design"}
        assert len(phases_found.intersection(expected_phases)) >= 2
        
        print(f"\n🤝 Collaboration Test Results:")
        print(f"   Agents involved: 3")
        print(f"   Documents shared: {len(research_docs) + 2}")
        print(f"   Creative search results: {len(creative_search.results)}")
        print(f"   Technical search results: {len(technical_search.results)}")
        print(f"   Integration search results: {len(integration_search.results)}")
        print(f"   Phases found in integration: {phases_found}")
        print(f"   Knowledge flow verified: ✅")
    
    @pytest.mark.asyncio
    async def test_tool_integration_performance(self, performance_rag_service):
        """Test performance of enhanced tools in realistic scenarios."""
        service = performance_rag_service
        
        # Create tools
        search_tool = EnhancedKnowledgeSearchTool(service)
        ingest_tool = AgentDocumentIngestTool(service)
        memory_tool = AgentMemoryTool(service)
        
        agent_id = "tool_perf_agent"
        num_operations = 20
        
        # Test tool performance
        start_time = time.time()
        
        # Simulate realistic tool usage pattern
        for i in range(num_operations):
            # Ingest document
            ingest_result = await ingest_tool._arun(
                title=f"Tool Test Document {i}",
                content=f"This is test content {i} for tool performance testing with unique data {i}",
                agent_id=agent_id,
                scope="private",
                metadata={"test_run": i, "tool": "ingest"}
            )
            
            # Create memory
            memory_result = await memory_tool._arun(
                content=f"Learned from document {i}: Important insight about tool performance",
                agent_id=agent_id,
                memory_type="episodic",
                importance=0.5 + (i % 5) * 0.1,
                tags=[f"document_{i}", "tool_test"]
            )
            
            # Search for content
            search_result = await search_tool._arun(
                query=f"test content {i} unique data",
                agent_id=agent_id,
                top_k=5,
                include_memories=True
            )
            
            # Verify all operations succeeded
            import json
            ingest_data = json.loads(ingest_result)
            memory_data = json.loads(memory_result)
            search_data = json.loads(search_result)
            
            assert ingest_data["success"] is True
            assert memory_data["success"] is True
            assert search_data["success"] is True
            assert len(search_data["results"]) > 0
        
        total_time = time.time() - start_time
        avg_time_per_operation = total_time / (num_operations * 3)  # 3 operations per iteration
        
        # Performance assertions
        assert total_time < 60.0  # Should complete within 60 seconds
        assert avg_time_per_operation < 1.0  # Each operation should take less than 1 second
        
        print(f"\n🔧 Tool Performance Results:")
        print(f"   Operations performed: {num_operations * 3}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per operation: {avg_time_per_operation:.3f}s")
        print(f"   Operations per second: {(num_operations * 3) / total_time:.1f}")
        
        # Verify final state
        final_search = await search_tool._arun(
            query="tool performance testing",
            agent_id=agent_id,
            top_k=50,
            include_memories=True
        )
        
        final_data = json.loads(final_search)
        assert len(final_data["results"]) >= num_operations  # Should find most documents and memories
