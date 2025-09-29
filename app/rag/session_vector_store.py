"""
Session Vector Store - In-Memory Document Storage for Sessions.

Provides session-scoped vector storage using existing RAG infrastructure
while maintaining complete isolation from permanent storage.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog
import numpy as np

from app.rag.core.embeddings import get_global_embedding_manager
from app.rag.core.batch_processor import get_batch_processor, EmbeddingBatchProcessor
from app.rag.ingestion.pipeline import RevolutionaryIngestionPipeline
from app.config.session_document_config import session_document_config
from app.models.session_document_models import SessionDocument

logger = structlog.get_logger(__name__)


class SessionVectorStore:
    """Session-scoped in-memory vector storage for document search."""

    def __init__(self):
        """Initialize the session vector store."""
        self.config = session_document_config.vector
        self.session_collections: Dict[str, Any] = {}
        self.embedding_manager = None
        self.ingestion_pipeline = None
        self.batch_processor = None
        self.is_initialized = False

        # Statistics
        self.stats = {
            "total_sessions": 0,
            "total_documents": 0,
            "search_queries": 0
        }
    
    async def initialize(self):
        """Initialize the session vector store."""
        if self.is_initialized:
            return

        if not self.config.enable_vector_search:
            self.is_initialized = True
            return

        try:
            # Initialize embedding manager
            self.embedding_manager = await get_global_embedding_manager()

            # Initialize batch processor for embeddings
            if self.embedding_manager:
                self.batch_processor = await get_batch_processor(
                    processor_name="session_embeddings",
                    processor_func=self.embedding_manager.generate_embeddings,
                    processor_type="embedding",
                    max_batch_size=50,
                    max_concurrent_batches=2
                )

            # Initialize ingestion pipeline
            self.ingestion_pipeline = RevolutionaryIngestionPipeline()
            await self.ingestion_pipeline.initialize()

            self.is_initialized = True
            logger.info("Session vector store initialized")

        except Exception as e:
            logger.error(f"Failed to initialize session vector store: {e}")
            self.is_initialized = True
    
    async def create_session_collection(self, session_id: str) -> bool:
        """Create a vector collection for a session."""
        try:
            if not self.is_initialized:
                await self.initialize()

            if not self.config.enable_vector_search:
                return True

            # Create in-memory collection for session
            self.session_collections[session_id] = {
                "documents": {},
                "embeddings": {},
                "metadata": {},
                "created_at": datetime.utcnow(),
                "document_count": 0
            }

            self.stats["total_sessions"] += 1
            logger.info(f"Session collection created: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create session collection {session_id}: {e}")
            return False
    
    async def add_document_to_session(self, session_id: str, document: SessionDocument) -> bool:
        """Add a document to the session vector collection."""
        try:
            if not self.config.enable_vector_search:
                return True

            # Ensure session collection exists
            if session_id not in self.session_collections:
                await self.create_session_collection(session_id)

            # Process document content
            chunks = await self._process_document(document)
            if not chunks:
                return False

            # Generate embeddings
            embeddings = await self._generate_embeddings(chunks)
            if not embeddings:
                return False

            # Store in session collection
            collection = self.session_collections[session_id]
            collection["documents"][document.document_id] = chunks
            collection["embeddings"][document.document_id] = embeddings
            collection["metadata"][document.document_id] = {
                "filename": document.filename,
                "content_type": document.content_type,
                "uploaded_at": document.uploaded_at.isoformat()
            }
            collection["document_count"] += 1
            self.stats["total_documents"] += 1

            logger.info(f"Document added to session {session_id}: {document.document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document to session {session_id}: {e}")
            return False
    
    async def search_session_documents(self, session_id: str, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search documents in a session using vector similarity."""
        try:
            if not self.config.enable_vector_search or session_id not in self.session_collections:
                return []

            collection = self.session_collections[session_id]
            if collection["document_count"] == 0:
                return []

            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)
            if query_embedding is None:
                return []

            # Calculate similarities
            results = []
            for doc_id, doc_embeddings in collection["embeddings"].items():
                max_similarity = 0.0
                best_chunk_idx = 0

                for idx, chunk_embedding in enumerate(doc_embeddings):
                    similarity = self._calculate_cosine_similarity(query_embedding, chunk_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_chunk_idx = idx

                if max_similarity >= self.config.similarity_threshold:
                    results.append({
                        "document_id": doc_id,
                        "similarity_score": float(max_similarity),
                        "content": collection["documents"][doc_id][best_chunk_idx],
                        "metadata": collection["metadata"][doc_id]
                    })

            # Sort and limit results
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_k = top_k or self.config.default_search_k
            results = results[:top_k]

            self.stats["search_queries"] += 1
            return results

        except Exception as e:
            logger.error(f"Session search failed for {session_id}: {e}")
            return []
    
    async def _process_document(self, document: SessionDocument) -> List[str]:
        """Process document and extract text chunks."""
        try:
            # Convert bytes to string
            if isinstance(document.content, bytes):
                try:
                    text_content = document.content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = document.content.decode('latin-1', errors='ignore')
            else:
                text_content = str(document.content)

            # Simple chunking strategy
            chunks = []
            chunk_size = self.config.chunk_size
            chunk_overlap = self.config.chunk_overlap

            for i in range(0, len(text_content), chunk_size - chunk_overlap):
                chunk = text_content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())

            return chunks if chunks else [text_content]

        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return []
    
    async def _generate_embeddings(self, text_chunks: List[str]) -> Optional[List[np.ndarray]]:
        """Generate embeddings for text chunks using batch processing."""
        try:
            if self.batch_processor:
                # Use batch processor for optimized embedding generation
                result = await self.batch_processor.generate_embeddings(text_chunks)
                if hasattr(result, 'embeddings'):
                    return [np.array(emb) for emb in result.embeddings]
                else:
                    return [np.array(emb) for emb in result]
            elif self.embedding_manager:
                # Fallback to direct embedding manager
                result = await self.embedding_manager.generate_embeddings(text_chunks)
                if hasattr(result, 'embeddings'):
                    return [np.array(emb) for emb in result.embeddings]
                else:
                    return [np.array(emb) for emb in result]
            return None
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None

    async def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate query embedding."""
        try:
            if self.embedding_manager:
                result = await self.embedding_manager.generate_embeddings([query])
                if hasattr(result, 'embeddings'):
                    return np.array(result.embeddings[0]) if result.embeddings else None
                else:
                    return np.array(result[0]) if result else None
            return None
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    async def remove_session_collection(self, session_id: str) -> bool:
        """Remove a session's vector collection."""
        try:
            if session_id in self.session_collections:
                del self.session_collections[session_id]
                self.stats["total_sessions"] -= 1
                logger.info(f"Session collection removed: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove session collection {session_id}: {e}")
            return False

    async def cleanup_expired_sessions(self) -> Dict[str, int]:
        """Clean up expired session collections."""
        try:
            current_time = datetime.utcnow()
            expiration_threshold = current_time - session_document_config.expiration.default_workspace_expiration

            expired_sessions = [
                session_id for session_id, collection in self.session_collections.items()
                if collection["created_at"] < expiration_threshold
            ]

            for session_id in expired_sessions:
                await self.remove_session_collection(session_id)

            return {"sessions_cleaned": len(expired_sessions)}

        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return {"error": str(e)}

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global vector store statistics."""
        return {
            **self.stats,
            "active_sessions": len(self.session_collections),
            "vector_search_enabled": self.config.enable_vector_search
        }


# Global session vector store instance
session_vector_store = SessionVectorStore()
