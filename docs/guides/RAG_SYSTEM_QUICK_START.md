# RAG System Quick Start Guide

**Complete guide to using the fully integrated RAG system**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Creating a Knowledge Base](#creating-a-knowledge-base)
3. [Ingesting Documents](#ingesting-documents)
4. [Searching Knowledge](#searching-knowledge)
5. [Advanced Features](#advanced-features)
6. [Code Examples](#code-examples)

---

## System Overview

The RAG system provides:
- **ChromaDB** vector storage with agent isolation
- **40+ file formats** supported (PDF, DOCX, XLSX, images, audio, video, code)
- **Advanced retrieval** (query expansion, BM25, hybrid fusion, reranking, MMR)
- **Structured KB** (metadata indexing, relationships, deduplication)
- **Production-ready** (error handling, metrics, caching, async)

---

## Creating a Knowledge Base

### Option 1: Using UnifiedRAGSystem (Recommended)

```python
from app.rag.core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig

# Initialize RAG system
config = UnifiedRAGConfig(
    persist_directory="./data/chroma",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200
)

rag_system = UnifiedRAGSystem(config)
await rag_system.initialize()

# Create agent ecosystem (KB + Memory)
agent_id = "agent_123"
agent_collections = await rag_system.create_agent_ecosystem(agent_id)

print(f"Knowledge Base: {agent_collections.knowledge_collection}")
print(f"Short Memory: {agent_collections.short_memory_collection}")
print(f"Long Memory: {agent_collections.long_memory_collection}")
```

### Option 2: Using CollectionBasedKBManager

```python
from app.rag.core.collection_based_kb_manager import (
    CollectionBasedKBManager,
    AccessLevel
)

# Create KB manager
kb_manager = CollectionBasedKBManager(rag_system)

# Create knowledge base
kb_id = await kb_manager.create_knowledge_base(
    owner_agent_id="agent_123",
    name="Research Papers",
    description="AI research papers collection",
    access_level=AccessLevel.PRIVATE
)

print(f"Knowledge Base ID: {kb_id}")
```

---

## Ingesting Documents

### Option 1: Direct Document Addition

```python
from app.rag.core.unified_rag_system import Document

# Prepare documents
documents = [
    Document(
        id="doc_1",
        content="Quantum computing is a revolutionary technology...",
        metadata={
            "title": "Quantum Computing Intro",
            "author": "John Doe",
            "category": "research",
            "page_count": 10
        }
    ),
    Document(
        id="doc_2",
        content="Machine learning algorithms...",
        metadata={
            "title": "ML Basics",
            "author": "Jane Smith",
            "category": "tutorial"
        }
    )
]

# Add to knowledge base
success = await rag_system.add_documents(
    agent_id="agent_123",
    documents=documents,
    collection_type="knowledge"
)

print(f"Documents added: {success}")
```

### Option 2: Using Revolutionary Ingestion Pipeline (Full Features)

```python
from app.rag.ingestion.pipeline import (
    RevolutionaryIngestionPipeline,
    RevolutionaryIngestionConfig
)
from app.rag.core.collection_based_kb_manager import KnowledgeBaseInfo

# Get KB info
kb_info = kb_manager.knowledge_bases[kb_id]

# Configure pipeline
pipeline_config = RevolutionaryIngestionConfig(
    enable_deduplication=True,
    enable_semantic_chunking=True,
    chunk_size=1000,
    chunk_overlap=200,
    enable_multimodal=True,
    enable_ocr=True
)

# Create pipeline
pipeline = RevolutionaryIngestionPipeline(kb_info, pipeline_config)
await pipeline.initialize()

# Ingest file
job_id = await pipeline.ingest_file(
    file_path="research_paper.pdf",
    collection="kb_agent_123",
    metadata={
        "category": "research",
        "priority": "high",
        "tags": ["quantum", "computing"]
    }
)

# Monitor progress
while True:
    status = await pipeline.get_job_status(job_id)
    print(f"Progress: {status.progress * 100:.1f}%")
    
    if status.status in ["completed", "failed"]:
        break
    
    await asyncio.sleep(1)

print(f"Job completed: {status.status}")
print(f"Chunks created: {status.chunks_created}")
print(f"Processing time: {status.processing_time_ms}ms")
```

### Supported File Formats

**Documents**:
- PDF, DOCX, DOC, RTF, ODT, TXT, MD

**Spreadsheets**:
- XLSX, XLS, CSV, ODS

**Presentations**:
- PPTX, PPT, ODP

**Images**:
- PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP

**Audio**:
- MP3, WAV, FLAC, OGG, M4A

**Video**:
- MP4, AVI, MOV, MKV, WEBM

**Code**:
- PY, JS, TS, JAVA, CPP, C, GO, RS, etc.

**Archives**:
- ZIP, TAR, GZ, BZ2, 7Z

**Email**:
- EML, MSG

---

## Searching Knowledge

### Basic Search

```python
# Simple search
results = await rag_system.search_agent_knowledge(
    agent_id="agent_123",
    query="quantum computing applications",
    top_k=10
)

for doc in results:
    print(f"ID: {doc.id}")
    print(f"Content: {doc.content[:200]}...")
    print(f"Score: {doc.metadata.get('similarity_score', 0):.3f}")
    print("---")
```

### Advanced Search with Pipeline

```python
# Advanced search with all features
results = await rag_system.search_agent_knowledge(
    agent_id="agent_123",
    query="quantum computing applications",
    top_k=10,
    search_type="advanced",  # Uses full pipeline
    use_advanced_retrieval=True
)

for doc in results:
    metadata = doc.metadata
    print(f"ID: {doc.id}")
    print(f"Content: {doc.content[:200]}...")
    print(f"Final Score: {metadata.get('similarity_score', 0):.3f}")
    print(f"Dense Score: {metadata.get('dense_score', 0):.3f}")
    print(f"Sparse Score: {metadata.get('sparse_score', 0):.3f}")
    print(f"Rerank Score: {metadata.get('rerank_score', 0):.3f}")
    print(f"MMR Score: {metadata.get('mmr_score', 0):.3f}")
    print("---")
```

### Structured Search with Filters

```python
# Search with metadata filters and context expansion
results = await rag_system.search_agent_knowledge_structured(
    agent_id="agent_123",
    query="machine learning algorithms",
    top_k=10,
    content_types=["text", "code"],  # Filter by content type
    section_path="Chapter 3",  # Filter by section
    page_number=42,  # Filter by page
    metadata_filters={
        "category": "research",
        "confidence": {"min": 0.8}  # Range filter
    },
    expand_context=True,  # Include surrounding chunks
    context_size=2,  # 2 chunks before/after
    use_advanced_retrieval=True
)

for doc in results:
    metadata = doc.metadata
    print(f"ID: {doc.id}")
    print(f"Content Type: {metadata.get('content_type')}")
    print(f"Section: {metadata.get('section_path')}")
    print(f"Page: {metadata.get('page_number')}")
    print(f"Is Context: {metadata.get('is_context', False)}")
    if metadata.get('is_context'):
        print(f"Context For: {metadata.get('context_for')}")
    print("---")
```

### Search Types

```python
# Dense vector search (default)
results = await rag_system.search_agent_knowledge(
    agent_id="agent_123",
    query="quantum computing",
    search_type="dense"
)

# Sparse keyword search (BM25)
results = await rag_system.search_agent_knowledge(
    agent_id="agent_123",
    query="quantum computing",
    search_type="sparse"
)

# Hybrid search (dense + sparse)
results = await rag_system.search_agent_knowledge(
    agent_id="agent_123",
    query="quantum computing",
    search_type="hybrid"
)

# Advanced search (full pipeline)
results = await rag_system.search_agent_knowledge(
    agent_id="agent_123",
    query="quantum computing",
    search_type="advanced",
    use_advanced_retrieval=True
)
```

---

## Advanced Features

### 1. Query Expansion

Automatically expands queries with synonyms and related terms:

```python
from app.rag.retrieval.query_expansion import (
    get_query_expander,
    ExpansionStrategy
)

expander = await get_query_expander()

# WordNet expansion
result = await expander.expand_query(
    "quantum computing",
    strategy=ExpansionStrategy.WORDNET
)
print(f"Expanded queries: {result.expanded_queries}")
# Output: ["quantum computing", "quantum calculation", "quantum processing"]

# Hybrid expansion (WordNet + Semantic)
result = await expander.expand_query(
    "machine learning",
    strategy=ExpansionStrategy.HYBRID
)
```

### 2. Deduplication

Prevent duplicate content:

```python
from app.rag.core.deduplication_enforcer import (
    get_deduplication_enforcer,
    DuplicateAction,
    ConflictResolution
)

dedup = await get_deduplication_enforcer(
    fuzzy_threshold=0.95,
    default_action=DuplicateAction.SKIP,
    conflict_resolution=ConflictResolution.KEEP_EXISTING
)

# Check for duplicates
result = await dedup.check_duplicate(
    chunk_id="chunk_123",
    content="Some content...",
    content_sha="abc123...",
    norm_text_sha="def456...",
    metadata={"source": "doc1"}
)

if result.is_duplicate:
    print(f"Duplicate found: {result.duplicate_ids}")
    print(f"Action: {result.action}")
```

### 3. Metadata Indexing

Fast metadata filtering:

```python
from app.rag.core.metadata_index import (
    get_metadata_index_manager,
    TermFilter,
    RangeFilter
)

metadata_index = await get_metadata_index_manager()

# Query with filters
chunk_ids = metadata_index.query(
    term_filters=[
        TermFilter(field="content_type", values=["text", "code"]),
        TermFilter(field="language", values=["en"])
    ],
    range_filters=[
        RangeFilter(field="confidence", min_value=0.8, max_value=1.0),
        RangeFilter(field="page_number", min_value=1, max_value=10)
    ]
)

print(f"Matching chunks: {len(chunk_ids)}")

# Get facets
facets = metadata_index.get_facets(
    fields=["content_type", "language"],
    limit=10
)

for facet in facets:
    print(f"{facet.field}: {facet.values}")
```

### 4. Chunk Relationships

Navigate document structure:

```python
from app.rag.core.chunk_relationship_manager import (
    get_chunk_relationship_manager
)

rel_manager = await get_chunk_relationship_manager()

# Get surrounding chunks
surrounding = rel_manager.get_surrounding_chunks(
    chunk_id="chunk_123",
    context_size=2,
    include_siblings=True
)

print(f"Surrounding chunks: {surrounding}")

# Get parent document
parent_doc = rel_manager.get_parent_document("chunk_123")
print(f"Parent document: {parent_doc}")

# Traverse hierarchy
hierarchy = rel_manager.traverse_hierarchy(
    start_chunk_id="chunk_123",
    max_depth=3
)

print(f"Hierarchy: {hierarchy}")
```

### 5. Multimodal Indexing

Content-type-specific search:

```python
from app.rag.core.multimodal_indexer import (
    get_multimodal_indexer,
    ContentType
)

multimodal = await get_multimodal_indexer()

# Search specific content types
results = await multimodal.search(
    query="machine learning",
    content_types=[ContentType.CODE, ContentType.TEXT],
    top_k=10
)

for result in results:
    print(f"Chunk: {result.chunk_id}")
    print(f"Type: {result.content_type}")
    print(f"Score: {result.score}")
    print(f"Boosted Score: {result.boosted_score}")
```

---

## Code Examples

### Complete Workflow Example

```python
import asyncio
from app.rag.core.unified_rag_system import UnifiedRAGSystem, Document

async def main():
    # 1. Initialize RAG system
    rag = UnifiedRAGSystem()
    await rag.initialize()
    
    # 2. Create agent ecosystem
    agent_id = "research_agent"
    await rag.create_agent_ecosystem(agent_id)
    
    # 3. Add documents
    docs = [
        Document(
            id="paper_1",
            content="Quantum computing uses quantum mechanics...",
            metadata={"title": "Quantum Intro", "category": "research"}
        ),
        Document(
            id="paper_2",
            content="Machine learning is a subset of AI...",
            metadata={"title": "ML Basics", "category": "tutorial"}
        )
    ]
    
    await rag.add_documents(agent_id, docs)
    
    # 4. Search with advanced features
    results = await rag.search_agent_knowledge(
        agent_id=agent_id,
        query="quantum computing applications",
        top_k=5,
        search_type="advanced",
        use_advanced_retrieval=True
    )
    
    # 5. Display results
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"   Score: {doc.metadata.get('similarity_score', 0):.3f}")
        print(f"   Content: {doc.content[:150]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

### File Ingestion Example

```python
import asyncio
from pathlib import Path
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.rag.core.collection_based_kb_manager import CollectionBasedKBManager
from app.rag.ingestion.pipeline import (
    RevolutionaryIngestionPipeline,
    RevolutionaryIngestionConfig
)

async def ingest_files():
    # Initialize
    rag = UnifiedRAGSystem()
    await rag.initialize()
    
    kb_manager = CollectionBasedKBManager(rag)
    
    # Create KB
    agent_id = "doc_agent"
    kb_id = await kb_manager.create_knowledge_base(
        owner_agent_id=agent_id,
        name="Document Library"
    )
    
    # Setup pipeline
    kb_info = kb_manager.knowledge_bases[kb_id]
    config = RevolutionaryIngestionConfig(
        enable_deduplication=True,
        enable_semantic_chunking=True
    )
    
    pipeline = RevolutionaryIngestionPipeline(kb_info, config)
    await pipeline.initialize()
    
    # Ingest files
    files = [
        "documents/research.pdf",
        "documents/presentation.pptx",
        "documents/data.xlsx"
    ]
    
    for file_path in files:
        if Path(file_path).exists():
            job_id = await pipeline.ingest_file(
                file_path=file_path,
                collection=f"kb_agent_{agent_id}"
            )
            
            # Wait for completion
            while True:
                status = await pipeline.get_job_status(job_id)
                if status.status in ["completed", "failed"]:
                    break
                await asyncio.sleep(0.5)
            
            print(f"✅ {file_path}: {status.chunks_created} chunks")

if __name__ == "__main__":
    asyncio.run(ingest_files())
```

---

## Best Practices

1. **Always initialize** the RAG system before use
2. **Use agent ecosystems** for proper isolation
3. **Enable advanced retrieval** for best results
4. **Use structured search** when you need filtering
5. **Monitor ingestion jobs** for large files
6. **Enable deduplication** to save storage
7. **Use appropriate chunk sizes** (1000 tokens default)
8. **Add metadata** for better filtering
9. **Use context expansion** for better understanding
10. **Check metrics** for performance monitoring

---

## Troubleshooting

### Issue: "RAG system not initialized"
**Solution**: Call `await rag_system.initialize()` before use

### Issue: "Collection not found"
**Solution**: Create agent ecosystem first with `create_agent_ecosystem()`

### Issue: "No results returned"
**Solution**: Check if documents were added successfully, verify agent_id matches

### Issue: "Slow search performance"
**Solution**: Reduce `top_k`, disable compression, use basic search instead of advanced

### Issue: "Out of memory during ingestion"
**Solution**: Reduce batch_size, process files one at a time, reduce chunk_size

---

## Next Steps

- Read the [Integration Analysis](../analysis/RAG_SYSTEM_INTEGRATION_ANALYSIS.md) for technical details
- Explore [Advanced Retrieval Documentation](../system-documentation/ADVANCED_RETRIEVAL_DOCUMENTATION.md)
- Check [RAG System Documentation](../system-documentation/RAG_SYSTEM_DOCUMENTATION.md) for architecture
- Review [Phase 1 & 2 Completion](../analysis/PHASE_1_2_COMPLETION_REPORT.md) for features

---

**The RAG system is production-ready and fully functional. Happy building! 🚀**

