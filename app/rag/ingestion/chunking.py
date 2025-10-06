"""
Semantic and layout-aware document chunking.

This module provides production-grade chunking that:
- Respects sentence boundaries
- Preserves document structure (headings, sections)
- Handles special content (tables, code, lists)
- Optimizes for retrieval quality
- Supports configurable chunk sizes with overlap
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class ContentType(str, Enum):
    """Types of content for specialized chunking."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    MARKDOWN = "markdown"


@dataclass
class ChunkConfig:
    """Configuration for semantic chunking."""
    min_chunk_size: int = 200  # Minimum tokens per chunk
    max_chunk_size: int = 800  # Maximum tokens per chunk
    overlap_percentage: float = 0.15  # 15% overlap
    respect_sentences: bool = True  # Don't break mid-sentence
    respect_paragraphs: bool = True  # Prefer paragraph boundaries
    respect_sections: bool = True  # Preserve section structure
    
    def get_overlap_size(self) -> int:
        """Calculate overlap size in tokens."""
        return int(self.max_chunk_size * self.overlap_percentage)


@dataclass
class Chunk:
    """A document chunk with metadata."""
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    section_path: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.content)
    
    @property
    def token_count_estimate(self) -> int:
        """Estimate token count (rough: 1 token ≈ 4 chars)."""
        return len(self.content) // 4


class SemanticChunker:
    """
    Semantic and layout-aware document chunker.
    
    Produces high-quality chunks that:
    - Respect natural boundaries (sentences, paragraphs, sections)
    - Preserve document structure
    - Optimize for retrieval quality
    - Handle special content types
    """
    
    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]+[\s\n]+')
    
    # Paragraph boundary patterns
    PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')
    
    # Heading patterns (Markdown-style)
    HEADING_PATTERN = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
    
    # Code block patterns
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`', re.MULTILINE)
    
    # Table patterns (simple detection)
    TABLE_PATTERN = re.compile(r'^\|.+\|$', re.MULTILINE)
    
    # List patterns
    LIST_PATTERN = re.compile(r'^[\s]*[-*+]\s+.+$|^[\s]*\d+\.\s+.+$', re.MULTILINE)
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize semantic chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        logger.info(
            "SemanticChunker initialized",
            min_size=self.config.min_chunk_size,
            max_size=self.config.max_chunk_size,
            overlap=f"{self.config.overlap_percentage*100}%"
        )
    
    def chunk_document(self, 
                      content: str,
                      content_type: ContentType = ContentType.TEXT,
                      section_path: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk document with semantic awareness.
        
        Args:
            content: Document content
            content_type: Type of content
            section_path: Section path (e.g., "2.1 Introduction")
            metadata: Additional metadata
            
        Returns:
            List of chunks
        """
        if not content or not content.strip():
            return []
        
        # Route to specialized chunker based on content type
        if content_type == ContentType.CODE:
            return self._chunk_code(content, section_path, metadata)
        elif content_type == ContentType.TABLE:
            return self._chunk_table(content, section_path, metadata)
        elif content_type == ContentType.MARKDOWN:
            return self._chunk_markdown(content, section_path, metadata)
        else:
            return self._chunk_text(content, section_path, metadata)
    
    def _chunk_text(self,
                   content: str,
                   section_path: Optional[str],
                   metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk plain text with sentence and paragraph awareness.
        
        Args:
            content: Text content
            section_path: Section path
            metadata: Metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Split into paragraphs first
        paragraphs = self.PARAGRAPH_BOUNDARY.split(content)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Estimate tokens
            para_tokens = len(para) // 4
            current_tokens = len(current_chunk) // 4
            
            # Check if adding this paragraph would exceed max size
            if current_tokens + para_tokens > self.config.max_chunk_size:
                # Save current chunk if it meets minimum size
                if current_tokens >= self.config.min_chunk_size:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        chunk_index=chunk_index,
                        section_path=section_path,
                        content_type=ContentType.TEXT,
                        metadata=metadata or {}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_size = self.config.get_overlap_size() * 4  # Convert tokens to chars
                    if len(current_chunk) > overlap_size:
                        # Take last sentences for overlap
                        overlap_text = self._get_last_sentences(current_chunk, overlap_size)
                        current_start = current_start + len(current_chunk) - len(overlap_text)
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_start = current_start + len(current_chunk)
                        current_chunk = para
                else:
                    # Current chunk too small, just add paragraph
                    current_chunk += "\n\n" + para if current_chunk else para
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                chunk_index=chunk_index,
                section_path=section_path,
                content_type=ContentType.TEXT,
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        logger.debug(
            "Text chunked",
            content_length=len(content),
            chunks_created=len(chunks),
            avg_chunk_size=sum(c.char_count for c in chunks) // len(chunks) if chunks else 0
        )
        
        return chunks
    
    def _get_last_sentences(self, text: str, max_chars: int) -> str:
        """
        Get last complete sentences up to max_chars.
        
        Args:
            text: Text to extract from
            max_chars: Maximum characters
            
        Returns:
            Last sentences
        """
        if len(text) <= max_chars:
            return text
        
        # Find sentence boundaries
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        # Build from end
        result = ""
        for sentence in reversed(sentences):
            if len(result) + len(sentence) <= max_chars:
                result = sentence + result
            else:
                break
        
        return result.strip()
    
    def _chunk_code(self,
                   content: str,
                   section_path: Optional[str],
                   metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk code with function/class awareness.
        
        For now, uses simple line-based chunking.
        TODO: Add AST-based chunking for better function/class boundaries.
        
        Args:
            content: Code content
            section_path: Section path
            metadata: Metadata
            
        Returns:
            List of chunks
        """
        # Simple implementation: chunk by lines
        lines = content.split('\n')
        chunks = []
        
        current_chunk_lines = []
        current_start = 0
        chunk_index = 0
        
        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            
            # Check if chunk is large enough
            chunk_text = '\n'.join(current_chunk_lines)
            if len(chunk_text) // 4 >= self.config.max_chunk_size:
                chunk = Chunk(
                    content=chunk_text,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    chunk_index=chunk_index,
                    section_path=section_path,
                    content_type=ContentType.CODE,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Overlap: keep last few lines
                overlap_lines = int(len(current_chunk_lines) * self.config.overlap_percentage)
                current_chunk_lines = current_chunk_lines[-overlap_lines:] if overlap_lines > 0 else []
                current_start = current_start + len(chunk_text) - len('\n'.join(current_chunk_lines))
        
        # Add final chunk
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunk = Chunk(
                content=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                chunk_index=chunk_index,
                section_path=section_path,
                content_type=ContentType.CODE,
                metadata=metadata or {}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_table(self,
                    content: str,
                    section_path: Optional[str],
                    metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk table content by row groups.

        Args:
            content: Table content
            section_path: Section path
            metadata: Metadata

        Returns:
            List of chunks
        """
        # Split by lines (rows)
        lines = content.split('\n')

        # Try to identify header
        header_lines = []
        data_lines = []

        for i, line in enumerate(lines):
            if i < 2:  # Assume first 2 lines might be header
                header_lines.append(line)
            else:
                data_lines.append(line)

        chunks = []
        chunk_index = 0

        # Create chunks with header + data rows
        current_rows = header_lines.copy()
        current_start = 0

        for line in data_lines:
            current_rows.append(line)

            # Check size
            chunk_text = '\n'.join(current_rows)
            if len(chunk_text) // 4 >= self.config.max_chunk_size:
                chunk = Chunk(
                    content=chunk_text,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    chunk_index=chunk_index,
                    section_path=section_path,
                    content_type=ContentType.TABLE,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with header
                current_rows = header_lines.copy()
                current_start = current_start + len(chunk_text)

        # Add final chunk
        if len(current_rows) > len(header_lines):
            chunk_text = '\n'.join(current_rows)
            chunk = Chunk(
                content=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                chunk_index=chunk_index,
                section_path=section_path,
                content_type=ContentType.TABLE,
                metadata=metadata or {}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_markdown(self,
                       content: str,
                       section_path: Optional[str],
                       metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk Markdown with heading awareness.

        Args:
            content: Markdown content
            section_path: Section path
            metadata: Metadata

        Returns:
            List of chunks
        """
        # Find all headings
        headings = []
        for match in self.HEADING_PATTERN.finditer(content):
            level = len(match.group().split()[0])  # Count #'s
            text = match.group().lstrip('#').strip()
            headings.append({
                'level': level,
                'text': text,
                'start': match.start(),
                'end': match.end()
            })

        if not headings:
            # No headings, treat as plain text
            return self._chunk_text(content, section_path, metadata)

        # Split content by sections
        sections = []
        for i, heading in enumerate(headings):
            start = heading['start']
            end = headings[i + 1]['start'] if i + 1 < len(headings) else len(content)
            section_content = content[start:end]
            section_path_new = f"{section_path}.{heading['text']}" if section_path else heading['text']
            sections.append({
                'path': section_path_new,
                'content': section_content,
                'start': start,
                'end': end
            })

        # Chunk each section
        all_chunks = []
        for section in sections:
            section_chunks = self._chunk_text(
                section['content'],
                section['path'],
                metadata
            )
            all_chunks.extend(section_chunks)

        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i

        return all_chunks


def chunk_document(content: str,
                  content_type: ContentType = ContentType.TEXT,
                  config: Optional[ChunkConfig] = None,
                  section_path: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
    """
    Convenience function to chunk a document.

    Args:
        content: Document content
        content_type: Type of content
        config: Chunking configuration
        section_path: Section path
        metadata: Additional metadata

    Returns:
        List of chunks
    """
    chunker = SemanticChunker(config)
    return chunker.chunk_document(content, content_type, section_path, metadata)

