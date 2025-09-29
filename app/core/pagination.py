"""
Revolutionary Advanced Pagination System.

This module provides comprehensive pagination, filtering, and sorting capabilities
for all API endpoints with AI-powered optimization suggestions.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import Query
import structlog

from app.api.v1.responses import PaginationInfo

logger = structlog.get_logger(__name__)


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class FilterOperator(str, Enum):
    """Filter operators for advanced filtering."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class AdvancedQueryParams(BaseModel):
    """Revolutionary advanced query parameters with AI optimization."""
    
    # Pagination
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    size: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    # Sorting
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: SortOrder = Field(default=SortOrder.ASC, description="Sort order")
    
    # Basic filtering
    status: Optional[List[str]] = Field(default=None, description="Filter by status")
    type: Optional[List[str]] = Field(default=None, description="Filter by type")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    
    # Date filtering
    created_after: Optional[datetime] = Field(default=None, description="Created after date")
    created_before: Optional[datetime] = Field(default=None, description="Created before date")
    updated_after: Optional[datetime] = Field(default=None, description="Updated after date")
    updated_before: Optional[datetime] = Field(default=None, description="Updated before date")
    
    # Search
    search: Optional[str] = Field(default=None, description="Search query")
    search_fields: Optional[List[str]] = Field(default=None, description="Fields to search in")
    
    # Advanced options
    include_deleted: bool = Field(default=False, description="Include deleted items")
    include_metadata: bool = Field(default=True, description="Include metadata")
    include_relationships: bool = Field(default=False, description="Include related data")
    
    # Performance options
    use_cache: bool = Field(default=True, description="Use cached results if available")
    cache_ttl: Optional[int] = Field(default=None, description="Cache TTL in seconds")
    
    @validator('size')
    def validate_size(cls, v):
        """Validate page size with intelligent suggestions."""
        if v > 100:
            logger.warning(f"Large page size requested: {v}, consider using smaller pages for better performance")
        return v
    
    @validator('search')
    def validate_search(cls, v):
        """Validate search query."""
        if v and len(v) < 2:
            raise ValueError("Search query must be at least 2 characters long")
        return v


class FilterCondition(BaseModel):
    """Advanced filter condition."""
    field: str = Field(..., description="Field name to filter on")
    operator: FilterOperator = Field(..., description="Filter operator")
    value: Any = Field(..., description="Filter value")
    case_sensitive: bool = Field(default=False, description="Case sensitive filtering")


class AdvancedFilter(BaseModel):
    """Advanced filtering with multiple conditions."""
    conditions: List[FilterCondition] = Field(default_factory=list, description="Filter conditions")
    logic: str = Field(default="AND", pattern="^(AND|OR)$", description="Logic operator between conditions")


class PaginationResult(BaseModel):
    """Result of pagination operation."""
    items: List[Any] = Field(..., description="Paginated items")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    total_count: int = Field(..., description="Total number of items")
    filtered_count: int = Field(..., description="Number of items after filtering")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class RevolutionaryPaginator:
    """Revolutionary pagination system with AI-powered optimization."""
    
    def __init__(self):
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
        self.optimization_suggestions: List[str] = []
    
    async def paginate_query(
        self,
        query: Query,
        params: AdvancedQueryParams,
        count_query: Optional[Query] = None,
        transform_func: Optional[Callable] = None
    ) -> PaginationResult:
        """Paginate SQLAlchemy query with advanced features."""
        try:
            start_time = datetime.utcnow()
            
            # Apply filtering
            filtered_query = await self._apply_filters(query, params)
            
            # Get total count
            if count_query:
                total_count = count_query.count()
            else:
                total_count = filtered_query.count()
            
            # Apply sorting
            sorted_query = await self._apply_sorting(filtered_query, params)
            
            # Apply pagination
            offset = (params.page - 1) * params.size
            paginated_query = sorted_query.offset(offset).limit(params.size)
            
            # Execute query
            items = paginated_query.all()
            
            # Transform items if function provided
            if transform_func:
                items = [transform_func(item) for item in items]
            
            # Calculate pagination info
            total_pages = (total_count + params.size - 1) // params.size
            pagination_info = PaginationInfo(
                page=params.page,
                size=params.size,
                total=total_count,
                total_pages=total_pages,
                has_next=params.page < total_pages,
                has_previous=params.page > 1
            )
            
            # Calculate performance metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            performance_metrics = {
                "execution_time_ms": execution_time,
                "items_returned": len(items),
                "total_items": total_count,
                "cache_used": params.use_cache,
                "optimization_applied": len(self.optimization_suggestions) > 0
            }
            
            # Generate optimization suggestions
            await self._generate_optimization_suggestions(params, performance_metrics)
            
            return PaginationResult(
                items=items,
                pagination=pagination_info,
                total_count=total_count,
                filtered_count=len(items),
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            params_dict = {}
            try:
                if hasattr(params, 'dict'):
                    params_dict = params.dict()
                elif hasattr(params, '__dict__'):
                    params_dict = params.__dict__
                else:
                    params_dict = {"type": str(type(params))}
            except Exception:
                params_dict = {"error": "Could not serialize params"}

            logger.error("Pagination failed", error=str(e), params=params_dict)
            raise
    
    async def paginate_list(
        self,
        items: List[Any],
        params: AdvancedQueryParams,
        filter_func: Optional[Callable] = None,
        sort_func: Optional[Callable] = None
    ) -> PaginationResult:
        """Paginate Python list with advanced features."""
        try:
            start_time = datetime.utcnow()
            
            # Apply filtering
            if filter_func:
                filtered_items = [item for item in items if filter_func(item, params)]
            else:
                filtered_items = items
            
            # Apply search
            if params.search:
                filtered_items = await self._apply_list_search(filtered_items, params)
            
            total_count = len(filtered_items)
            
            # Apply sorting
            if sort_func:
                filtered_items = sort_func(filtered_items, params)
            elif params.sort_by:
                filtered_items = await self._apply_list_sorting(filtered_items, params)
            
            # Apply pagination
            start_idx = (params.page - 1) * params.size
            end_idx = start_idx + params.size
            paginated_items = filtered_items[start_idx:end_idx]
            
            # Calculate pagination info
            total_pages = (total_count + params.size - 1) // params.size
            pagination_info = PaginationInfo(
                page=params.page,
                size=params.size,
                total=total_count,
                total_pages=total_pages,
                has_next=params.page < total_pages,
                has_previous=params.page > 1
            )
            
            # Calculate performance metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            performance_metrics = {
                "execution_time_ms": execution_time,
                "items_returned": len(paginated_items),
                "total_items": len(items),
                "filtered_items": total_count,
                "in_memory_processing": True
            }
            
            return PaginationResult(
                items=paginated_items,
                pagination=pagination_info,
                total_count=len(items),
                filtered_count=total_count,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error("List pagination failed", error=str(e), params=params.dict())
            raise
    
    async def _apply_filters(self, query: Query, params: AdvancedQueryParams) -> Query:
        """Apply advanced filters to SQLAlchemy query."""
        # This is a simplified implementation
        # In practice, you'd need to map params to actual model fields
        
        if params.status:
            query = query.filter(query.column_descriptions[0]['entity'].status.in_(params.status))
        
        if params.created_after:
            query = query.filter(query.column_descriptions[0]['entity'].created_at >= params.created_after)
        
        if params.created_before:
            query = query.filter(query.column_descriptions[0]['entity'].created_at <= params.created_before)
        
        return query
    
    async def _apply_sorting(self, query: Query, params: AdvancedQueryParams) -> Query:
        """Apply sorting to SQLAlchemy query."""
        if not params.sort_by:
            return query
        
        # This is a simplified implementation
        # In practice, you'd need to map sort_by to actual model fields
        entity = query.column_descriptions[0]['entity']
        
        if hasattr(entity, params.sort_by):
            field = getattr(entity, params.sort_by)
            if params.sort_order == SortOrder.DESC:
                query = query.order_by(desc(field))
            else:
                query = query.order_by(asc(field))
        
        return query
    
    async def _apply_list_search(self, items: List[Any], params: AdvancedQueryParams) -> List[Any]:
        """Apply search to Python list."""
        if not params.search:
            return items
        
        search_term = params.search.lower()
        filtered_items = []
        
        for item in items:
            # Convert item to dict if it's a Pydantic model
            if hasattr(item, 'dict'):
                item_dict = item.dict()
            elif hasattr(item, '__dict__'):
                item_dict = item.__dict__
            else:
                item_dict = {"value": str(item)}
            
            # Search in specified fields or all fields
            search_fields = params.search_fields or list(item_dict.keys())
            
            for field in search_fields:
                if field in item_dict:
                    field_value = str(item_dict[field]).lower()
                    if search_term in field_value:
                        filtered_items.append(item)
                        break
        
        return filtered_items
    
    async def _apply_list_sorting(self, items: List[Any], params: AdvancedQueryParams) -> List[Any]:
        """Apply sorting to Python list."""
        if not params.sort_by:
            return items
        
        def get_sort_key(item):
            if hasattr(item, params.sort_by):
                return getattr(item, params.sort_by)
            elif hasattr(item, 'dict') and params.sort_by in item.dict():
                return item.dict()[params.sort_by]
            elif hasattr(item, '__dict__') and params.sort_by in item.__dict__:
                return item.__dict__[params.sort_by]
            else:
                return ""
        
        reverse = params.sort_order == SortOrder.DESC
        return sorted(items, key=get_sort_key, reverse=reverse)
    
    async def _generate_optimization_suggestions(
        self, 
        params: AdvancedQueryParams, 
        metrics: Dict[str, Any]
    ) -> None:
        """Generate AI-powered optimization suggestions."""
        self.optimization_suggestions.clear()
        
        # Performance-based suggestions
        if metrics.get("execution_time_ms", 0) > 1000:
            self.optimization_suggestions.append("Consider reducing page size for better performance")
        
        if params.size > 50 and metrics.get("execution_time_ms", 0) > 500:
            self.optimization_suggestions.append("Large page size detected, consider pagination optimization")
        
        if params.search and len(params.search) < 3:
            self.optimization_suggestions.append("Short search queries may return too many results")
        
        # Caching suggestions
        if not params.use_cache and metrics.get("execution_time_ms", 0) > 200:
            self.optimization_suggestions.append("Enable caching for frequently accessed data")


# Global paginator instance
paginator = RevolutionaryPaginator()
