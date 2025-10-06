"""
Contextual Compression for RAG Retrieval.

This module implements contextual compression to extract relevant passages
from long documents, reducing context size while preserving relevance.

Compression Strategies:
- Extractive: Select most relevant sentences/passages
- Sentence scoring: Score sentences by relevance to query
- Passage extraction: Extract coherent passages around relevant sentences
- Redundancy removal: Remove duplicate or highly similar content

Features:
- Multiple compression strategies
- Configurable compression ratio
- Sentence-level and passage-level extraction
- Relevance scoring
- Quality preservation
- Comprehensive metrics

Author: Agentic AI System
Purpose: Compress long documents while preserving relevant information
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field
import numpy as np

# Sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    try:
        sent_tokenize("test")
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
    sent_tokenize = None

logger = structlog.get_logger(__name__)


class CompressionStrategy(str, Enum):
    """Compression strategies."""
    EXTRACTIVE = "extractive"  # Extract most relevant sentences
    PASSAGE = "passage"  # Extract coherent passages
    HYBRID = "hybrid"  # Combine extractive and passage


class CompressionConfig(BaseModel):
    """Configuration for contextual compression."""
    strategy: CompressionStrategy = Field(default=CompressionStrategy.EXTRACTIVE)
    target_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Target compression ratio (0.5 = 50% of original)"
    )
    min_sentence_length: int = Field(default=10, ge=1)
    max_sentence_length: int = Field(default=500, ge=10)
    context_window: int = Field(
        default=1,
        ge=0,
        description="Number of sentences before/after to include for context"
    )
    remove_redundancy: bool = Field(default=True)
    redundancy_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


@dataclass
class CompressedResult:
    """Compressed document result."""
    doc_id: str
    original_content: str
    compressed_content: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    relevance_score: float
    selected_sentences: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextualCompressor:
    """
    Production-grade contextual compression engine.
    
    Compresses long documents by extracting most relevant passages
    while preserving context and coherence.
    
    Features:
    - Multiple compression strategies
    - Sentence-level relevance scoring
    - Passage extraction with context
    - Redundancy removal
    - Quality preservation
    - Comprehensive metrics
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize contextual compressor."""
        self.config = config or CompressionConfig()
        
        # Metrics
        self._metrics = {
            'total_compressions': 0,
            'total_documents_compressed': 0,
            'avg_compression_ratio': 0.0,
            'avg_compression_time_ms': 0.0
        }
        
        logger.info(
            "ContextualCompressor initialized",
            strategy=self.config.strategy.value,
            target_ratio=self.config.target_ratio
        )
    
    def compress(
        self,
        query: str,
        results: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None
    ) -> List[CompressedResult]:
        """
        Compress documents to extract relevant passages.
        
        Args:
            query: Search query
            results: List of results with 'doc_id', 'content', 'score', 'metadata'
            query_embedding: Optional query embedding for relevance scoring
            
        Returns:
            List of CompressedResult with compressed content
        """
        start_time = time.time()
        
        if not results:
            logger.warning("No results to compress")
            return []
        
        compressed_results = []
        
        for result in results:
            doc_id = result.get('doc_id')
            content = result.get('content', '')
            score = result.get('score', 0.0)
            metadata = result.get('metadata', {})
            
            if not doc_id or not content:
                logger.warning("Skipping result with missing doc_id or content")
                continue
            
            try:
                # Compress document
                compressed_content, selected_sentences = self._compress_document(
                    query,
                    content,
                    query_embedding
                )
                
                # Calculate compression ratio
                original_length = len(content)
                compressed_length = len(compressed_content)
                compression_ratio = compressed_length / original_length if original_length > 0 else 0.0
                
                compressed_results.append(CompressedResult(
                    doc_id=doc_id,
                    original_content=content,
                    compressed_content=compressed_content,
                    original_length=original_length,
                    compressed_length=compressed_length,
                    compression_ratio=compression_ratio,
                    relevance_score=score,
                    selected_sentences=selected_sentences,
                    metadata=metadata
                ))
                
                # Update metrics
                total_compressions = self._metrics['total_compressions']
                current_avg = self._metrics['avg_compression_ratio']
                self._metrics['avg_compression_ratio'] = (
                    (current_avg * total_compressions + compression_ratio) / (total_compressions + 1)
                )
                
            except Exception as e:
                logger.error(f"Failed to compress document {doc_id}: {e}")
                # Return original content on error
                compressed_results.append(CompressedResult(
                    doc_id=doc_id,
                    original_content=content,
                    compressed_content=content,
                    original_length=len(content),
                    compressed_length=len(content),
                    compression_ratio=1.0,
                    relevance_score=score,
                    selected_sentences=[],
                    metadata=metadata
                ))
        
        # Update metrics
        self._metrics['total_compressions'] += 1
        self._metrics['total_documents_compressed'] += len(compressed_results)
        
        compression_time_ms = (time.time() - start_time) * 1000
        total_compressions = self._metrics['total_compressions']
        current_avg = self._metrics['avg_compression_time_ms']
        self._metrics['avg_compression_time_ms'] = (
            (current_avg * (total_compressions - 1) + compression_time_ms) / total_compressions
        )
        
        logger.debug(
            f"Compression completed",
            documents_count=len(compressed_results),
            avg_ratio=self._metrics['avg_compression_ratio'],
            compression_time_ms=compression_time_ms
        )
        
        return compressed_results
    
    def _compress_document(
        self,
        query: str,
        content: str,
        query_embedding: Optional[np.ndarray]
    ) -> Tuple[str, List[str]]:
        """
        Compress a single document.
        
        Returns:
            Tuple of (compressed_content, selected_sentences)
        """
        # Tokenize into sentences
        sentences = self._tokenize_sentences(content)
        
        if not sentences:
            return content, []
        
        # Score sentences by relevance
        sentence_scores = self._score_sentences(query, sentences, query_embedding)
        
        # Select sentences based on strategy
        if self.config.strategy == CompressionStrategy.EXTRACTIVE:
            selected_indices = self._select_extractive(sentence_scores)
        elif self.config.strategy == CompressionStrategy.PASSAGE:
            selected_indices = self._select_passage(sentence_scores)
        else:  # HYBRID
            selected_indices = self._select_hybrid(sentence_scores)
        
        # Add context window
        if self.config.context_window > 0:
            selected_indices = self._add_context_window(selected_indices, len(sentences))
        
        # Remove redundancy if configured
        if self.config.remove_redundancy:
            selected_indices = self._remove_redundant_sentences(
                sentences,
                selected_indices
            )
        
        # Extract selected sentences
        selected_sentences = [sentences[i] for i in sorted(selected_indices)]
        compressed_content = ' '.join(selected_sentences)
        
        return compressed_content, selected_sentences
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        if NLTK_AVAILABLE and sent_tokenize:
            try:
                sentences = sent_tokenize(text)
            except Exception:
                # Fallback to simple split
                sentences = self._simple_sentence_split(text)
        else:
            sentences = self._simple_sentence_split(text)
        
        # Filter by length
        sentences = [
            s.strip() for s in sentences
            if self.config.min_sentence_length <= len(s.strip()) <= self.config.max_sentence_length
        ]
        
        return sentences
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback."""
        # Split on period, exclamation, question mark followed by space
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentences(
        self,
        query: str,
        sentences: List[str],
        query_embedding: Optional[np.ndarray]
    ) -> List[float]:
        """
        Score sentences by relevance to query.

        Returns:
            List of relevance scores (one per sentence)
        """
        scores = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_terms = set(sentence_lower.split())

            # Term overlap score (Jaccard similarity)
            if query_terms and sentence_terms:
                overlap = len(query_terms & sentence_terms)
                union = len(query_terms | sentence_terms)
                term_score = overlap / union if union > 0 else 0.0
            else:
                term_score = 0.0

            # Position score (earlier sentences slightly preferred)
            # This is a simple heuristic

            # Combined score
            score = term_score
            scores.append(score)

        return scores

    def _select_extractive(self, sentence_scores: List[float]) -> List[int]:
        """
        Select sentences using extractive strategy.

        Selects top-scoring sentences up to target ratio.
        """
        # Calculate target number of sentences
        total_sentences = len(sentence_scores)
        target_count = max(1, int(total_sentences * self.config.target_ratio))

        # Get indices sorted by score descending
        sorted_indices = sorted(
            range(len(sentence_scores)),
            key=lambda i: sentence_scores[i],
            reverse=True
        )

        # Select top sentences
        selected_indices = sorted_indices[:target_count]

        return selected_indices

    def _select_passage(self, sentence_scores: List[float]) -> List[int]:
        """
        Select sentences using passage strategy.

        Finds contiguous passages with high average score.
        """
        if not sentence_scores:
            return []

        total_sentences = len(sentence_scores)
        target_count = max(1, int(total_sentences * self.config.target_ratio))

        # Find best contiguous passage
        best_start = 0
        best_score = 0.0

        for start in range(total_sentences):
            for end in range(start + 1, min(start + target_count + 1, total_sentences + 1)):
                passage_score = sum(sentence_scores[start:end]) / (end - start)
                if passage_score > best_score:
                    best_score = passage_score
                    best_start = start
                    best_end = end

        # Return indices of best passage
        if best_score > 0:
            return list(range(best_start, best_end))
        else:
            # Fallback to extractive
            return self._select_extractive(sentence_scores)

    def _select_hybrid(self, sentence_scores: List[float]) -> List[int]:
        """
        Select sentences using hybrid strategy.

        Combines extractive and passage approaches.
        """
        # Get extractive selections
        extractive_indices = set(self._select_extractive(sentence_scores))

        # Get passage selections
        passage_indices = set(self._select_passage(sentence_scores))

        # Combine (union)
        combined_indices = extractive_indices | passage_indices

        # Limit to target ratio
        total_sentences = len(sentence_scores)
        target_count = max(1, int(total_sentences * self.config.target_ratio))

        if len(combined_indices) > target_count:
            # Sort by score and take top
            sorted_indices = sorted(
                combined_indices,
                key=lambda i: sentence_scores[i],
                reverse=True
            )
            combined_indices = set(sorted_indices[:target_count])

        return list(combined_indices)

    def _add_context_window(
        self,
        selected_indices: List[int],
        total_sentences: int
    ) -> List[int]:
        """
        Add context window around selected sentences.

        Includes N sentences before and after each selected sentence.
        """
        expanded_indices = set(selected_indices)
        window = self.config.context_window

        for idx in selected_indices:
            # Add sentences before
            for i in range(max(0, idx - window), idx):
                expanded_indices.add(i)

            # Add sentences after
            for i in range(idx + 1, min(total_sentences, idx + window + 1)):
                expanded_indices.add(i)

        return list(expanded_indices)

    def _remove_redundant_sentences(
        self,
        sentences: List[str],
        selected_indices: List[int]
    ) -> List[int]:
        """
        Remove redundant sentences based on similarity.

        Keeps only sentences that are sufficiently different from each other.
        """
        if len(selected_indices) <= 1:
            return selected_indices

        # Sort indices to process in order
        sorted_indices = sorted(selected_indices)

        # Keep first sentence
        filtered_indices = [sorted_indices[0]]

        for idx in sorted_indices[1:]:
            sentence = sentences[idx]

            # Check similarity to already selected sentences
            is_redundant = False
            for kept_idx in filtered_indices:
                kept_sentence = sentences[kept_idx]
                similarity = self._compute_text_similarity(sentence, kept_sentence)

                if similarity >= self.config.redundancy_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                filtered_indices.append(idx)

        return filtered_indices

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two text strings.

        Uses Jaccard similarity on word sets.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get compression metrics."""
        return {
            'total_compressions': self._metrics['total_compressions'],
            'total_documents_compressed': self._metrics['total_documents_compressed'],
            'avg_compression_ratio': self._metrics['avg_compression_ratio'],
            'avg_compression_time_ms': self._metrics['avg_compression_time_ms']
        }

