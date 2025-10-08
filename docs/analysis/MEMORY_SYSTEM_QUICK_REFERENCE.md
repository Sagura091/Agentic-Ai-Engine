# 🧠 MEMORY SYSTEM QUICK REFERENCE GUIDE

**For Developers Working with Agent Memory**

---

## 📚 MEMORY TYPES

### **Episodic Memory** 🎬
- **What:** Specific experiences and events
- **Example:** "User asked about quantum computing on 2024-01-15"
- **Use Case:** Remembering past conversations, interactions
- **Retention:** Medium-term (days to weeks)

### **Semantic Memory** 📖
- **What:** General knowledge and concepts
- **Example:** "Quantum computing uses qubits for computation"
- **Use Case:** Learned facts, domain knowledge
- **Retention:** Long-term (weeks to months)

### **Procedural Memory** 🔧
- **What:** Skills and procedures
- **Example:** "To analyze data: 1) Load file, 2) Clean data, 3) Run analysis"
- **Use Case:** Learned workflows, best practices
- **Retention:** Long-term (permanent)

### **Working Memory** 💭
- **What:** Current context and temporary information
- **Example:** "Currently processing user's request about stocks"
- **Use Case:** Active task context
- **Retention:** Very short-term (minutes)

### **Core Memory** 🎯
- **What:** Always-visible persistent context
- **Example:** "User prefers Python over JavaScript"
- **Use Case:** User preferences, agent identity
- **Retention:** Permanent

### **Resource Memory** 📁
- **What:** Documents and files
- **Example:** "research_paper.pdf - Quantum Computing Basics"
- **Use Case:** Knowledge base documents
- **Retention:** Permanent (until deleted)

### **Knowledge Vault** 🔐
- **What:** Sensitive information
- **Example:** "API key: sk-xxx" (encrypted)
- **Use Case:** Credentials, secrets
- **Retention:** Permanent (encrypted)

---

## 🔧 HOW TO USE MEMORY IN YOUR AGENT

### **1. Storing Memories**

```python
# Simple storage
memory_id = await agent.memory_system.add_memory(
    agent_id=agent.agent_id,
    memory_type=MemoryType.EPISODIC,
    content="User asked about machine learning",
    metadata={"topic": "ml", "session_id": session_id},
    importance=MemoryImportance.MEDIUM,
    emotional_valence=0.5,  # Positive interaction
    tags={"ml", "conversation"}
)

# Store with context
memory_id = await agent.memory_system.add_memory(
    agent_id=agent.agent_id,
    memory_type=MemoryType.SEMANTIC,
    content="Machine learning is a subset of AI",
    metadata={"domain": "ai", "source": "learned"},
    importance=MemoryImportance.HIGH,
    context={"learned_from": "user_interaction", "confidence": 0.9}
)
```

### **2. Retrieving Memories**

```python
# Active retrieval (context-aware)
result = await agent.memory_system.active_retrieve_memories(
    agent_id=agent.agent_id,
    current_task="Tell me about machine learning",
    conversation_context="Previous discussion about AI",
    max_memories=5,
    relevance_threshold=0.3
)

for memory in result.memories:
    print(f"Memory: {memory.content}")
    print(f"Relevance: {result.relevance_scores[memory.id]}")

# Search by query
memories = await agent.memory_system.search_memories(
    agent_id=agent.agent_id,
    query="machine learning",
    memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
    limit=10
)

# Search by tags
memories = await agent.memory_system.search_by_tags(
    agent_id=agent.agent_id,
    tags={"ml", "conversation"},
    limit=10
)
```

### **3. Using Memories in Agent Execution**

```python
async def execute_with_memory(self, task: str, session_id: str = None):
    """Execute task with memory integration."""
    
    # 1. Retrieve relevant memories
    result = await self.memory_system.active_retrieve_memories(
        agent_id=self.agent_id,
        current_task=task,
        max_memories=5
    )
    
    # 2. Build context from memories
    memory_context = ""
    if result.memories:
        memory_context = "\n\nRELEVANT PAST EXPERIENCES:\n"
        for mem in result.memories:
            memory_context += f"- {mem.content}\n"
    
    # 3. Execute with context
    enhanced_task = f"{task}\n{memory_context}"
    response = await self.llm.ainvoke(enhanced_task)
    
    # 4. Store outcome as memory
    await self.memory_system.add_memory(
        agent_id=self.agent_id,
        memory_type=MemoryType.EPISODIC,
        content=f"Task: {task} | Outcome: {response.content[:100]}",
        metadata={"session_id": session_id, "success": True},
        importance=MemoryImportance.MEDIUM
    )
    
    return response
```

---

## 🎯 MEMORY IMPORTANCE LEVELS

| Level | When to Use | Retention | Example |
|-------|-------------|-----------|---------|
| **CRITICAL** | Must never forget | Permanent | User's core preferences, agent identity |
| **HIGH** | Very important | Long-term | Key learnings, important decisions |
| **MEDIUM** | Moderately important | Medium-term | Regular interactions, useful facts |
| **LOW** | Nice to have | Short-term | Minor details, temporary notes |
| **TEMPORARY** | Discard soon | Very short | Intermediate calculations |

---

## 📊 MEMORY SYSTEM COMPONENTS

### **UnifiedMemorySystem** (Simple Memory)
- **Location:** `app/memory/unified_memory_system.py`
- **Use For:** Most agents
- **Features:** Fast, in-memory, RAG-integrated
- **Persistence:** ⚠️ Currently in-memory only (needs fix)

### **PersistentMemorySystem** (Advanced Memory)
- **Location:** `app/agents/autonomous/persistent_memory.py`
- **Use For:** Autonomous agents
- **Features:** PostgreSQL-backed, full persistence
- **Persistence:** ✅ Fully persistent

### **Memory Models**
- **Location:** `app/memory/memory_models.py`
- **Contains:** MemoryEntry, MemoryCollection, CoreMemoryBlock, etc.

### **Database Models**
- **Location:** `app/models/autonomous.py`
- **Tables:** `agent_memories`, `autonomous_agent_states`

---

## 🔍 ADVANCED FEATURES

### **Core Memory (Always-Visible Context)**

```python
# Set core memory
await agent.memory_system.set_core_memory(
    agent_id=agent.agent_id,
    block_type="user_preferences",
    content="User prefers concise answers and Python code examples"
)

# Get core memory
core_memory = await agent.memory_system.get_core_memory(agent.agent_id)
print(core_memory.user_preferences)
```

### **Resource Memory (Documents)**

```python
# Add document
resource_id = await agent.memory_system.add_resource(
    agent_id=agent.agent_id,
    title="Machine Learning Guide",
    summary="Comprehensive guide to ML algorithms",
    content=document_text,
    resource_type="document",
    metadata={"author": "John Doe", "year": 2024}
)

# Search documents
results = await agent.memory_system.search_resources(
    agent_id=agent.agent_id,
    query="neural networks",
    resource_type="document",
    top_k=5
)
```

### **Knowledge Vault (Sensitive Data)**

```python
# Store sensitive data
vault_id = await agent.memory_system.add_vault_entry(
    agent_id=agent.agent_id,
    entry_type="api_key",
    content="sk-1234567890abcdef",
    sensitivity=SensitivityLevel.SECRET,
    metadata={"service": "openai", "expires": "2025-12-31"}
)

# Retrieve (with access logging)
entry = await agent.memory_system.get_vault_entry(agent.agent_id, vault_id)
```

### **Memory Consolidation**

```python
# Manual consolidation
result = await agent.memory_system.run_consolidation_for_agent(agent.agent_id)
print(f"Processed: {result['memories_processed']}")
print(f"Promoted: {result['memories_promoted']}")
print(f"Forgotten: {result['memories_forgotten']}")
```

### **Learning from Experience**

```python
# Record learning experience
experience_id = await agent.memory_system.record_learning_experience(
    agent_id=agent.agent_id,
    task_type="data_analysis",
    task_context={"dataset": "sales_data", "goal": "predict_revenue"},
    performance_metrics={"accuracy": 0.92, "time_taken": 45.2},
    memories_used=["mem_123", "mem_456"],
    memories_created=["mem_789"],
    success=True
)
```

---

## ⚠️ CURRENT LIMITATIONS (TO BE FIXED)

1. **No Automatic Persistence** - Memories not saved to database automatically
2. **No Memory Loading** - Agents don't load memories on restart
3. **No Auto-Consolidation** - Consolidation must be triggered manually
4. **No Auto-Retrieval** - Agents don't retrieve memories automatically during execution

**See:** `docs/analysis/MEMORY_SYSTEM_FIX_ACTION_PLAN.md` for fixes

---

## 🧪 TESTING YOUR MEMORY INTEGRATION

```python
import pytest
from app.memory.memory_models import MemoryType, MemoryImportance

@pytest.mark.asyncio
async def test_agent_memory():
    # Create agent
    agent = await create_test_agent()
    
    # Store memory
    memory_id = await agent.memory_system.add_memory(
        agent_id=agent.agent_id,
        memory_type=MemoryType.EPISODIC,
        content="Test memory",
        importance=MemoryImportance.HIGH
    )
    
    # Retrieve memory
    memories = await agent.memory_system.search_memories(
        agent_id=agent.agent_id,
        query="Test memory"
    )
    
    assert len(memories) > 0
    assert memories[0].content == "Test memory"
```

---

## 📖 BEST PRACTICES

### ✅ DO

- **Store outcomes** after every significant action
- **Retrieve memories** before making decisions
- **Use appropriate memory types** (episodic for events, semantic for facts)
- **Set importance levels** correctly (critical for must-keep, temporary for discard)
- **Add metadata** for better searchability
- **Use tags** for categorization

### ❌ DON'T

- **Don't store sensitive data** in regular memory (use Knowledge Vault)
- **Don't store everything** (be selective, use importance levels)
- **Don't forget to retrieve** memories before decisions
- **Don't use working memory** for long-term storage
- **Don't ignore consolidation** (run periodically)

---

## 🔗 RELATED DOCUMENTATION

- **Full Analysis:** `docs/analysis/MEMORY_SYSTEM_DEEP_DIVE_ANALYSIS.md`
- **Fix Plan:** `docs/analysis/MEMORY_SYSTEM_FIX_ACTION_PLAN.md`
- **Memory System Docs:** `docs/system-documentation/MEMORY_SYSTEM_DOCUMENTATION.md`
- **RAG Integration:** `docs/system-documentation/RAG_SYSTEM_DOCUMENTATION.md`

---

## 🆘 TROUBLESHOOTING

### **Memories Not Persisting**
- **Cause:** Database persistence not implemented yet
- **Workaround:** Use PersistentMemorySystem for autonomous agents
- **Fix:** See action plan

### **Agent Doesn't Remember Past Interactions**
- **Cause:** Memories not loaded on agent creation
- **Workaround:** Manually load memories after creation
- **Fix:** See action plan

### **Memory Retrieval Returns Nothing**
- **Check:** Is memory system initialized?
- **Check:** Are memories actually stored?
- **Check:** Is relevance threshold too high?
- **Debug:** Lower threshold to 0.1 and check

### **Out of Memory Errors**
- **Cause:** Too many memories in cache
- **Solution:** Run consolidation to clean up
- **Solution:** Adjust cache size limits in config

---

**END OF QUICK REFERENCE**

