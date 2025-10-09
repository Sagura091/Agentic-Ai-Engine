# RAG System Basics

Learn how to use the Retrieval-Augmented Generation (RAG) system to give your agents access to custom knowledge bases.

## 📋 What You'll Learn

By the end of this tutorial, you will:

- ✅ Understand what RAG is and why it's useful
- ✅ Create a knowledge base
- ✅ Upload documents to the knowledge base
- ✅ Create an agent with RAG capabilities
- ✅ Query your knowledge base through the agent

**Estimated Time:** 25 minutes

## 🎯 Prerequisites

Before starting, make sure you've completed:
- **[Getting Started Tutorial](getting-started.md)** - Basic setup
- **[Build Your First Agent](first-agent.md)** - Agent creation
- The system is running

## 💡 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
- **Retrieval:** Finding relevant information from a knowledge base
- **Generation:** Using that information to generate responses

### Why Use RAG?

- **Custom Knowledge:** Give agents access to your specific documents
- **Up-to-Date Info:** Add new information without retraining models
- **Accurate Responses:** Ground responses in your actual data
- **Source Citations:** Know where information comes from

## 📚 Step 1: Create a Knowledge Base

Let's create a knowledge base for company documentation:

```python
import requests

# Create knowledge base
kb_config = {
    "name": "company_docs",
    "description": "Company policies, procedures, and documentation",
    "embedding_model": "all-MiniLM-L6-v2",  # Fast, good quality
    "chunk_size": 500,
    "chunk_overlap": 50
}

response = requests.post(
    "http://localhost:8888/api/v1/knowledge-bases",
    json=kb_config
)

kb = response.json()
kb_id = kb["id"]
print(f"✅ Created knowledge base: {kb_id}")
```

### Using curl:

```bash
curl -X POST "http://localhost:8888/api/v1/knowledge-bases" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "company_docs",
    "description": "Company policies and documentation",
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50
  }'
```

## 📄 Step 2: Upload Documents

Now let's add some documents to the knowledge base:

### Upload a Text File:

```python
# Upload a document
with open("employee_handbook.pdf", "rb") as f:
    files = {"file": f}
    data = {"knowledge_base_id": kb_id}
    
    response = requests.post(
        "http://localhost:8888/api/v1/documents/upload",
        files=files,
        data=data
    )

doc = response.json()
print(f"✅ Uploaded document: {doc['id']}")
print(f"   Chunks created: {doc['chunk_count']}")
```

### Upload Multiple Documents:

```python
import os

# Upload all PDFs in a directory
docs_dir = "company_docs/"
for filename in os.listdir(docs_dir):
    if filename.endswith(".pdf"):
        filepath = os.path.join(docs_dir, filename)
        
        with open(filepath, "rb") as f:
            files = {"file": f}
            data = {"knowledge_base_id": kb_id}
            
            response = requests.post(
                "http://localhost:8888/api/v1/documents/upload",
                files=files,
                data=data
            )
            
            print(f"✅ Uploaded: {filename}")
```

### Upload Text Directly:

```python
# Upload text content directly
text_content = """
Company Policy: Remote Work

Employees may work remotely up to 3 days per week.
Remote work must be approved by direct manager.
Core hours (10 AM - 3 PM) must be maintained.
"""

response = requests.post(
    "http://localhost:8888/api/v1/documents/text",
    json={
        "knowledge_base_id": kb_id,
        "content": text_content,
        "metadata": {
            "title": "Remote Work Policy",
            "category": "HR Policies",
            "version": "2.0"
        }
    }
)

print("✅ Uploaded text document")
```

## 🤖 Step 3: Create a RAG-Enabled Agent

Create an agent that can access the knowledge base:

```python
# Create RAG-enabled agent
agent_config = {
    "name": "hr_assistant",
    "agent_type": "react",
    "description": "HR assistant with access to company policies",
    "system_prompt": """You are an HR assistant with access to company policies and documentation.
    
    When answering questions:
    1. Search the knowledge base for relevant information
    2. Provide accurate answers based on the documents
    3. Cite the source documents
    4. If information isn't in the knowledge base, say so
    
    Always be helpful, accurate, and professional.""",
    
    "llm_config": {
        "provider": "ollama",
        "model_id": "llama3.2:latest",
        "temperature": 0.2  # Low temperature for accuracy
    },
    
    "rag_config": {
        "enabled": true,
        "knowledge_base_ids": [kb_id],
        "top_k": 5,  # Retrieve top 5 relevant chunks
        "similarity_threshold": 0.7
    },
    
    "tools": [
        "knowledge_search",  # RAG tool
        "document_ingest"    # For adding new docs
    ]
}

response = requests.post(
    "http://localhost:8888/api/v1/agents",
    json=agent_config
)

agent = response.json()
agent_id = agent["id"]
print(f"✅ Created RAG-enabled agent: {agent_id}")
```

## 💬 Step 4: Query the Knowledge Base

Now let's ask questions about the documents:

```python
# Ask a question
response = requests.post(
    f"http://localhost:8888/api/v1/agents/{agent_id}/chat",
    json={
        "message": "What is the company's remote work policy?"
    }
)

result = response.json()
print("Agent Response:")
print(result["response"])

print("\nSources Used:")
for source in result["sources"]:
    print(f"- {source['document']}: {source['relevance_score']}")
```

### Expected Response:

```
Agent Response:
According to the company's Remote Work Policy (version 2.0), employees may work 
remotely up to 3 days per week. Remote work must be approved by your direct manager, 
and you must maintain core hours from 10 AM to 3 PM.

Sources Used:
- Remote Work Policy: 0.92
- Employee Handbook Section 4.2: 0.85
```

## 🔍 Step 5: Advanced RAG Queries

### Semantic Search:

```python
# Direct knowledge base search
response = requests.post(
    f"http://localhost:8888/api/v1/knowledge-bases/{kb_id}/search",
    json={
        "query": "vacation days",
        "top_k": 3
    }
)

results = response.json()
for result in results["results"]:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Source: {result['metadata']['source']}\n")
```

### Filtered Search:

```python
# Search with metadata filters
response = requests.post(
    f"http://localhost:8888/api/v1/knowledge-bases/{kb_id}/search",
    json={
        "query": "benefits",
        "top_k": 5,
        "filters": {
            "category": "HR Policies",
            "version": "2.0"
        }
    }
)
```

## 📊 Step 6: Monitor RAG Performance

Check how well your RAG system is performing:

```python
# Get knowledge base statistics
stats = requests.get(
    f"http://localhost:8888/api/v1/knowledge-bases/{kb_id}/stats"
).json()

print(f"Total Documents: {stats['document_count']}")
print(f"Total Chunks: {stats['chunk_count']}")
print(f"Total Queries: {stats['query_count']}")
print(f"Average Relevance Score: {stats['avg_relevance_score']}")
```

## 🎨 Step 7: Optimize RAG Performance

### Adjust Chunk Size:

```python
# Update knowledge base configuration
requests.patch(
    f"http://localhost:8888/api/v1/knowledge-bases/{kb_id}",
    json={
        "chunk_size": 1000,  # Larger chunks for more context
        "chunk_overlap": 100  # More overlap for continuity
    }
)

# Re-process documents with new settings
requests.post(
    f"http://localhost:8888/api/v1/knowledge-bases/{kb_id}/reprocess"
)
```

### Adjust Retrieval Parameters:

```python
# Update agent RAG configuration
requests.patch(
    f"http://localhost:8888/api/v1/agents/{agent_id}",
    json={
        "rag_config": {
            "top_k": 10,  # Retrieve more chunks
            "similarity_threshold": 0.6,  # Lower threshold for more results
            "rerank": true  # Enable re-ranking for better results
        }
    }
)
```

## 🎯 Best Practices

### 1. **Document Preparation**
- Clean and format documents before upload
- Remove unnecessary content
- Add meaningful metadata
- Use consistent formatting

### 2. **Chunk Size**
- **Small chunks (200-500):** Better for precise answers
- **Medium chunks (500-1000):** Balanced approach
- **Large chunks (1000-2000):** More context, less precision

### 3. **Retrieval Settings**
- **top_k:** Start with 5, adjust based on results
- **similarity_threshold:** 0.7 is a good starting point
- **Enable reranking** for better accuracy

### 4. **Agent Configuration**
- Use low temperature (0.2-0.4) for factual accuracy
- Include clear instructions about citing sources
- Test with various question types

## 🚀 Advanced Features

### Hybrid Search:

Combine semantic search with keyword search:

```python
response = requests.post(
    f"http://localhost:8888/api/v1/knowledge-bases/{kb_id}/search",
    json={
        "query": "remote work policy",
        "search_type": "hybrid",  # Semantic + keyword
        "alpha": 0.7  # 70% semantic, 30% keyword
    }
)
```

### Multi-Knowledge Base Search:

Search across multiple knowledge bases:

```python
agent_config = {
    "rag_config": {
        "enabled": true,
        "knowledge_base_ids": [kb_id_1, kb_id_2, kb_id_3],
        "search_strategy": "parallel"  # Search all in parallel
    }
}
```

## 🛠️ Troubleshooting

### Low Relevance Scores

**Problem:** Search results have low relevance scores

**Solutions:**
- Improve document quality and formatting
- Adjust chunk size
- Lower similarity threshold
- Try different embedding models

### Agent Not Using RAG

**Problem:** Agent doesn't search knowledge base

**Solutions:**
- Ensure `knowledge_search` tool is included
- Make RAG usage explicit in system prompt
- Lower temperature for more deterministic behavior

### Slow Search Performance

**Problem:** Searches take too long

**Solutions:**
- Reduce top_k value
- Use smaller embedding model
- Enable caching
- Optimize chunk size

## 📚 Additional Resources

- **[RAG System Documentation](../reference/RAG_SYSTEM_DOCUMENTATION.md)**
- **[RAG Quick Start Guide](../guides/RAG_SYSTEM_QUICK_START.md)**
- **[Knowledge Base API Reference](../reference/API_SYSTEM_DOCUMENTATION.md)**
- **[Embedding Models Guide](../reference/LLM_SYSTEM_DOCUMENTATION.md)**

## 💡 Use Cases

- **Customer Support:** Answer questions from documentation
- **Internal Knowledge:** Company policies and procedures
- **Technical Documentation:** API docs, code examples
- **Research:** Academic papers, research notes
- **Legal:** Contracts, compliance documents

---

**Previous Tutorial:** [Build Your First Agent](first-agent.md) ←  
**Next Steps:** Explore [Advanced RAG Features](../guides/RAG_SYSTEM_QUICK_START.md) →

