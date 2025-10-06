"""
Metadata Indexing System.

This module provides fast metadata indexing for efficient filtering and faceted search.
It maintains separate indexes for different metadata fields to enable quick lookups
and aggregations without scanning the entire vector database.

Key Features:
- Multi-field indexing (content_type, language, date, author, source, etc.)
- Faceted search with dynamic facet generation
- Range queries (date ranges, numeric ranges)
- Aggregations (count, sum, avg, min, max)
- Query planning and optimization
- Efficient data structures (inverted indexes, B-trees)

Author: Agentic AI System
Purpose: Fast metadata filtering and faceted search
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from collections import defaultdict
import bisect

import structlog

logger = structlog.get_logger(__name__)


class IndexType(str, Enum):
    """Types of metadata indexes."""
    INVERTED = "inverted"  # For categorical fields (content_type, language)
    RANGE = "range"  # For numeric/date fields (page_number, timestamp)
    FULL_TEXT = "full_text"  # For text fields (title, author)
    COMPOSITE = "composite"  # For multi-field indexes


class AggregationType(str, Enum):
    """Types of aggregations."""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"
    DATE_HISTOGRAM = "date_histogram"
    TERMS = "terms"


@dataclass
class FacetValue:
    """A facet value with count."""
    value: Any
    count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Facet:
    """A facet with multiple values."""
    field: str
    values: List[FacetValue]
    total_count: int


@dataclass
class RangeFilter:
    """Range filter for numeric/date fields."""
    field: str
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    include_min: bool = True
    include_max: bool = True


@dataclass
class TermFilter:
    """Term filter for categorical fields."""
    field: str
    values: List[Any]
    operator: str = "OR"  # OR, AND, NOT


@dataclass
class QueryPlan:
    """Query execution plan."""
    filters: List[Any]
    estimated_results: int
    index_usage: List[str]
    execution_order: List[str]


class InvertedIndex:
    """
    Inverted index for categorical fields.
    
    Maps field values to sets of chunk IDs.
    Example: content_type -> {"code": {chunk1, chunk2}, "text": {chunk3}}
    """
    
    def __init__(self, field_name: str):
        """
        Initialize inverted index.
        
        Args:
            field_name: Name of the indexed field
        """
        self.field_name = field_name
        self._index: Dict[Any, Set[str]] = defaultdict(set)
        self._reverse_index: Dict[str, Any] = {}  # chunk_id -> value
        
    def add(self, chunk_id: str, value: Any) -> None:
        """
        Add a chunk to the index.
        
        Args:
            chunk_id: Chunk ID
            value: Field value
        """
        # Remove old value if exists
        if chunk_id in self._reverse_index:
            old_value = self._reverse_index[chunk_id]
            self._index[old_value].discard(chunk_id)
        
        # Add new value
        self._index[value].add(chunk_id)
        self._reverse_index[chunk_id] = value
    
    def remove(self, chunk_id: str) -> None:
        """
        Remove a chunk from the index.
        
        Args:
            chunk_id: Chunk ID
        """
        if chunk_id in self._reverse_index:
            value = self._reverse_index[chunk_id]
            self._index[value].discard(chunk_id)
            del self._reverse_index[chunk_id]
    
    def get(self, value: Any) -> Set[str]:
        """
        Get all chunks with a specific value.
        
        Args:
            value: Field value
            
        Returns:
            Set of chunk IDs
        """
        return self._index.get(value, set())
    
    def get_facets(self, limit: int = 10) -> List[FacetValue]:
        """
        Get facet values with counts.
        
        Args:
            limit: Maximum number of facet values
            
        Returns:
            List of facet values sorted by count
        """
        facets = [
            FacetValue(value=value, count=len(chunk_ids))
            for value, chunk_ids in self._index.items()
        ]
        
        # Sort by count descending
        facets.sort(key=lambda f: f.count, reverse=True)
        
        return facets[:limit]
    
    def get_all_values(self) -> Set[Any]:
        """
        Get all unique values in the index.
        
        Returns:
            Set of unique values
        """
        return set(self._index.keys())
    
    def size(self) -> int:
        """
        Get number of unique chunks in the index.
        
        Returns:
            Number of chunks
        """
        return len(self._reverse_index)


class RangeIndex:
    """
    Range index for numeric/date fields.
    
    Uses sorted lists for efficient range queries.
    """
    
    def __init__(self, field_name: str):
        """
        Initialize range index.
        
        Args:
            field_name: Name of the indexed field
        """
        self.field_name = field_name
        self._values: List[Tuple[Any, str]] = []  # (value, chunk_id) sorted by value
        self._chunk_to_value: Dict[str, Any] = {}
        self._sorted = True
    
    def add(self, chunk_id: str, value: Any) -> None:
        """
        Add a chunk to the index.
        
        Args:
            chunk_id: Chunk ID
            value: Field value (must be comparable)
        """
        # Remove old value if exists
        if chunk_id in self._chunk_to_value:
            self.remove(chunk_id)
        
        # Add new value
        self._values.append((value, chunk_id))
        self._chunk_to_value[chunk_id] = value
        self._sorted = False
    
    def remove(self, chunk_id: str) -> None:
        """
        Remove a chunk from the index.
        
        Args:
            chunk_id: Chunk ID
        """
        if chunk_id in self._chunk_to_value:
            value = self._chunk_to_value[chunk_id]
            self._values = [(v, cid) for v, cid in self._values if cid != chunk_id]
            del self._chunk_to_value[chunk_id]
            self._sorted = False
    
    def _ensure_sorted(self) -> None:
        """Ensure the values list is sorted."""
        if not self._sorted:
            self._values.sort(key=lambda x: x[0])
            self._sorted = True
    
    def range_query(
        self,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        include_min: bool = True,
        include_max: bool = True
    ) -> Set[str]:
        """
        Query chunks in a range.
        
        Args:
            min_value: Minimum value (inclusive/exclusive based on include_min)
            max_value: Maximum value (inclusive/exclusive based on include_max)
            include_min: Include minimum value
            include_max: Include maximum value
            
        Returns:
            Set of chunk IDs in range
        """
        self._ensure_sorted()
        
        result = set()
        
        for value, chunk_id in self._values:
            # Check minimum
            if min_value is not None:
                if include_min:
                    if value < min_value:
                        continue
                else:
                    if value <= min_value:
                        continue
            
            # Check maximum
            if max_value is not None:
                if include_max:
                    if value > max_value:
                        break
                else:
                    if value >= max_value:
                        break
            
            result.add(chunk_id)
        
        return result
    
    def get_min(self) -> Optional[Any]:
        """Get minimum value."""
        self._ensure_sorted()
        return self._values[0][0] if self._values else None
    
    def get_max(self) -> Optional[Any]:
        """Get maximum value."""
        self._ensure_sorted()
        return self._values[-1][0] if self._values else None
    
    def size(self) -> int:
        """Get number of chunks in the index."""
        return len(self._chunk_to_value)


class MetadataIndexManager:
    """
    Production-grade metadata index manager.

    Manages multiple indexes for different metadata fields and provides
    unified query interface with faceted search and aggregations.

    Features:
    - Multi-field indexing
    - Faceted search
    - Range queries
    - Aggregations
    - Query planning
    - Index optimization
    """

    def __init__(self):
        """Initialize metadata index manager."""
        # Inverted indexes for categorical fields
        self._inverted_indexes: Dict[str, InvertedIndex] = {}

        # Range indexes for numeric/date fields
        self._range_indexes: Dict[str, RangeIndex] = {}

        # Field type mapping
        self._field_types: Dict[str, IndexType] = {}

        # Metrics
        self._metrics = {
            'total_indexes': 0,
            'total_chunks_indexed': 0,
            'total_queries': 0,
            'total_facet_queries': 0,
            'total_range_queries': 0,
            'avg_query_time_ms': 0.0
        }

        logger.info("MetadataIndexManager initialized")

    def create_index(self, field_name: str, index_type: IndexType) -> bool:
        """
        Create an index for a field.

        Args:
            field_name: Name of the field to index
            index_type: Type of index to create

        Returns:
            True if successful
        """
        try:
            if index_type == IndexType.INVERTED:
                self._inverted_indexes[field_name] = InvertedIndex(field_name)
            elif index_type == IndexType.RANGE:
                self._range_indexes[field_name] = RangeIndex(field_name)
            else:
                logger.error(f"Unsupported index type: {index_type}")
                return False

            self._field_types[field_name] = index_type
            self._metrics['total_indexes'] += 1

            logger.info(f"Index created: {field_name} ({index_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to create index for {field_name}: {e}")
            return False

    def add_document(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Add a document to all indexes.

        Args:
            chunk_id: Chunk ID
            metadata: Metadata dictionary

        Returns:
            True if successful
        """
        try:
            # Add to inverted indexes
            for field_name, index in self._inverted_indexes.items():
                if field_name in metadata:
                    index.add(chunk_id, metadata[field_name])

            # Add to range indexes
            for field_name, index in self._range_indexes.items():
                if field_name in metadata:
                    value = metadata[field_name]
                    # Convert datetime to timestamp for comparison
                    if isinstance(value, datetime):
                        value = value.timestamp()
                    elif isinstance(value, date):
                        value = datetime.combine(value, datetime.min.time()).timestamp()
                    index.add(chunk_id, value)

            self._metrics['total_chunks_indexed'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to add document {chunk_id}: {e}")
            return False

    def remove_document(self, chunk_id: str) -> bool:
        """
        Remove a document from all indexes.

        Args:
            chunk_id: Chunk ID

        Returns:
            True if successful
        """
        try:
            # Remove from inverted indexes
            for index in self._inverted_indexes.values():
                index.remove(chunk_id)

            # Remove from range indexes
            for index in self._range_indexes.values():
                index.remove(chunk_id)

            self._metrics['total_chunks_indexed'] -= 1
            return True

        except Exception as e:
            logger.error(f"Failed to remove document {chunk_id}: {e}")
            return False

    def query(
        self,
        term_filters: Optional[List[TermFilter]] = None,
        range_filters: Optional[List[RangeFilter]] = None
    ) -> Set[str]:
        """
        Query indexes with filters.

        Args:
            term_filters: List of term filters
            range_filters: List of range filters

        Returns:
            Set of matching chunk IDs
        """
        import time
        start_time = time.time()

        try:
            result_sets: List[Set[str]] = []

            # Apply term filters
            if term_filters:
                for term_filter in term_filters:
                    if term_filter.field not in self._inverted_indexes:
                        logger.warning(f"No index for field: {term_filter.field}")
                        continue

                    index = self._inverted_indexes[term_filter.field]

                    if term_filter.operator == "OR":
                        # Union of all values
                        chunks = set()
                        for value in term_filter.values:
                            chunks.update(index.get(value))
                        result_sets.append(chunks)
                    elif term_filter.operator == "AND":
                        # Intersection of all values
                        chunks = None
                        for value in term_filter.values:
                            value_chunks = index.get(value)
                            if chunks is None:
                                chunks = value_chunks.copy()
                            else:
                                chunks.intersection_update(value_chunks)
                        result_sets.append(chunks or set())
                    elif term_filter.operator == "NOT":
                        # All chunks except these values
                        all_chunks = set()
                        for value in index.get_all_values():
                            all_chunks.update(index.get(value))

                        exclude_chunks = set()
                        for value in term_filter.values:
                            exclude_chunks.update(index.get(value))

                        result_sets.append(all_chunks - exclude_chunks)

            # Apply range filters
            if range_filters:
                for range_filter in range_filters:
                    if range_filter.field not in self._range_indexes:
                        logger.warning(f"No range index for field: {range_filter.field}")
                        continue

                    index = self._range_indexes[range_filter.field]

                    chunks = index.range_query(
                        min_value=range_filter.min_value,
                        max_value=range_filter.max_value,
                        include_min=range_filter.include_min,
                        include_max=range_filter.include_max
                    )
                    result_sets.append(chunks)
                    self._metrics['total_range_queries'] += 1

            # Intersect all result sets
            if result_sets:
                result = result_sets[0]
                for rs in result_sets[1:]:
                    result.intersection_update(rs)
            else:
                result = set()

            # Update metrics
            self._metrics['total_queries'] += 1
            query_time = (time.time() - start_time) * 1000
            self._metrics['avg_query_time_ms'] = (
                (self._metrics['avg_query_time_ms'] * (self._metrics['total_queries'] - 1) + query_time) /
                self._metrics['total_queries']
            )

            logger.debug(
                f"Query executed",
                term_filters=len(term_filters) if term_filters else 0,
                range_filters=len(range_filters) if range_filters else 0,
                results=len(result),
                time_ms=query_time
            )

            return result

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return set()

    def get_facets(
        self,
        fields: List[str],
        limit: int = 10,
        filters: Optional[List[Any]] = None
    ) -> List[Facet]:
        """
        Get facets for specified fields.

        Args:
            fields: List of fields to get facets for
            limit: Maximum number of facet values per field
            filters: Optional filters to apply before faceting

        Returns:
            List of facets
        """
        try:
            # Apply filters first if provided
            filtered_chunks = None
            if filters:
                term_filters = [f for f in filters if isinstance(f, TermFilter)]
                range_filters = [f for f in filters if isinstance(f, RangeFilter)]
                filtered_chunks = self.query(term_filters, range_filters)

            facets = []

            for field in fields:
                if field not in self._inverted_indexes:
                    logger.warning(f"No index for field: {field}")
                    continue

                index = self._inverted_indexes[field]

                # Get facet values
                if filtered_chunks is not None:
                    # Filter facets by filtered chunks
                    facet_counts: Dict[Any, int] = defaultdict(int)
                    for chunk_id in filtered_chunks:
                        if chunk_id in index._reverse_index:
                            value = index._reverse_index[chunk_id]
                            facet_counts[value] += 1

                    facet_values = [
                        FacetValue(value=value, count=count)
                        for value, count in facet_counts.items()
                    ]
                    facet_values.sort(key=lambda f: f.count, reverse=True)
                    facet_values = facet_values[:limit]
                    total_count = sum(f.count for f in facet_values)
                else:
                    # Get all facets
                    facet_values = index.get_facets(limit)
                    total_count = sum(f.count for f in facet_values)

                facets.append(Facet(
                    field=field,
                    values=facet_values,
                    total_count=total_count
                ))

            self._metrics['total_facet_queries'] += 1

            logger.debug(f"Facets generated for {len(fields)} fields")
            return facets

        except Exception as e:
            logger.error(f"Facet generation failed: {e}")
            return []

    def aggregate(
        self,
        field: str,
        aggregation_type: AggregationType,
        filters: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform aggregation on a field.

        Args:
            field: Field to aggregate
            aggregation_type: Type of aggregation
            filters: Optional filters to apply

        Returns:
            Aggregation result
        """
        try:
            # Apply filters
            filtered_chunks = None
            if filters:
                term_filters = [f for f in filters if isinstance(f, TermFilter)]
                range_filters = [f for f in filters if isinstance(f, RangeFilter)]
                filtered_chunks = self.query(term_filters, range_filters)

            # Get values
            if field in self._range_indexes:
                index = self._range_indexes[field]

                if filtered_chunks is not None:
                    values = [
                        index._chunk_to_value[chunk_id]
                        for chunk_id in filtered_chunks
                        if chunk_id in index._chunk_to_value
                    ]
                else:
                    values = list(index._chunk_to_value.values())

                if not values:
                    return {'error': 'No values found'}

                # Perform aggregation
                if aggregation_type == AggregationType.COUNT:
                    return {'count': len(values)}
                elif aggregation_type == AggregationType.SUM:
                    return {'sum': sum(values)}
                elif aggregation_type == AggregationType.AVG:
                    return {'avg': sum(values) / len(values)}
                elif aggregation_type == AggregationType.MIN:
                    return {'min': min(values)}
                elif aggregation_type == AggregationType.MAX:
                    return {'max': max(values)}
                elif aggregation_type == AggregationType.PERCENTILE:
                    # Calculate percentiles
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    return {
                        'p50': sorted_values[int(n * 0.5)],
                        'p95': sorted_values[int(n * 0.95)],
                        'p99': sorted_values[int(n * 0.99)]
                    }
                else:
                    return {'error': f'Unsupported aggregation type: {aggregation_type}'}

            elif field in self._inverted_indexes:
                index = self._inverted_indexes[field]

                if aggregation_type == AggregationType.COUNT:
                    if filtered_chunks is not None:
                        count = len(filtered_chunks)
                    else:
                        count = index.size()
                    return {'count': count}
                elif aggregation_type == AggregationType.TERMS:
                    # Return top terms
                    facet_values = index.get_facets(limit=100)
                    return {
                        'terms': [
                            {'value': fv.value, 'count': fv.count}
                            for fv in facet_values
                        ]
                    }
                else:
                    return {'error': f'Unsupported aggregation type for categorical field: {aggregation_type}'}

            else:
                return {'error': f'No index for field: {field}'}

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {'error': str(e)}

    def create_query_plan(
        self,
        term_filters: Optional[List[TermFilter]] = None,
        range_filters: Optional[List[RangeFilter]] = None
    ) -> QueryPlan:
        """
        Create an optimized query execution plan.

        Args:
            term_filters: List of term filters
            range_filters: List of range filters

        Returns:
            Query plan
        """
        filters = []
        index_usage = []
        execution_order = []
        estimated_results = 0

        # Estimate selectivity for each filter
        selectivity_scores = []

        if term_filters:
            for term_filter in term_filters:
                if term_filter.field in self._inverted_indexes:
                    index = self._inverted_indexes[term_filter.field]

                    # Estimate result size
                    result_size = 0
                    for value in term_filter.values:
                        result_size += len(index.get(value))

                    selectivity_scores.append((
                        result_size,
                        f"term_filter:{term_filter.field}",
                        term_filter
                    ))
                    index_usage.append(f"inverted_index:{term_filter.field}")

        if range_filters:
            for range_filter in range_filters:
                if range_filter.field in self._range_indexes:
                    index = self._range_indexes[range_filter.field]

                    # Estimate result size (rough estimate)
                    total_size = index.size()
                    result_size = total_size // 2  # Rough estimate

                    selectivity_scores.append((
                        result_size,
                        f"range_filter:{range_filter.field}",
                        range_filter
                    ))
                    index_usage.append(f"range_index:{range_filter.field}")

        # Sort by selectivity (most selective first)
        selectivity_scores.sort(key=lambda x: x[0])

        # Build execution order
        for size, name, filter_obj in selectivity_scores:
            execution_order.append(name)
            filters.append(filter_obj)

        # Estimate final result size (use most selective filter)
        if selectivity_scores:
            estimated_results = selectivity_scores[0][0]

        return QueryPlan(
            filters=filters,
            estimated_results=estimated_results,
            index_usage=index_usage,
            execution_order=execution_order
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get index metrics.

        Returns:
            Metrics dictionary
        """
        return {
            **self._metrics,
            'inverted_indexes': len(self._inverted_indexes),
            'range_indexes': len(self._range_indexes),
            'indexed_fields': list(self._field_types.keys())
        }

    def optimize_indexes(self) -> bool:
        """
        Optimize all indexes.

        Returns:
            True if successful
        """
        try:
            # Ensure all range indexes are sorted
            for index in self._range_indexes.values():
                index._ensure_sorted()

            logger.info("Indexes optimized")
            return True

        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return False


# Global singleton
_metadata_index_manager: Optional[MetadataIndexManager] = None
_manager_lock = asyncio.Lock()


async def get_metadata_index_manager() -> MetadataIndexManager:
    """
    Get or create metadata index manager singleton.

    Returns:
        MetadataIndexManager instance
    """
    global _metadata_index_manager

    async with _manager_lock:
        if _metadata_index_manager is None:
            _metadata_index_manager = MetadataIndexManager()

            # Create default indexes
            _metadata_index_manager.create_index('content_type', IndexType.INVERTED)
            _metadata_index_manager.create_index('language', IndexType.INVERTED)
            _metadata_index_manager.create_index('source_type', IndexType.INVERTED)
            _metadata_index_manager.create_index('page_number', IndexType.RANGE)
            _metadata_index_manager.create_index('chunk_index', IndexType.RANGE)
            _metadata_index_manager.create_index('confidence', IndexType.RANGE)

            logger.info("Default metadata indexes created")

        return _metadata_index_manager
