"""
Web Research API endpoints for direct tool testing and usage.

This module provides REST API endpoints for testing and using the web research tools
directly, allowing users to search, scrape, and analyze web content.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl

from app.core.dependencies import get_current_user
from app.tools.web_tools_registry import get_web_research_tool, web_tools_registry

logger = structlog.get_logger(__name__)
router = APIRouter()


class WebSearchRequest(BaseModel):
    """Web search request model."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    num_results: int = Field(default=10, description="Number of results", ge=1, le=50)
    search_engine: str = Field(default="duckduckgo", description="Search engine")
    safe_search: bool = Field(default=True, description="Enable safe search")
    region: str = Field(default="us-en", description="Search region")


class WebScrapeRequest(BaseModel):
    """Web scraping request model."""
    url: HttpUrl = Field(..., description="URL to scrape")
    extract_links: bool = Field(default=False, description="Extract all links")
    extract_images: bool = Field(default=False, description="Extract image URLs")
    extract_text: bool = Field(default=True, description="Extract text content")
    css_selector: Optional[str] = Field(default=None, description="CSS selector")
    timeout: int = Field(default=30, description="Request timeout", ge=5, le=120)


class MultiScrapeRequest(BaseModel):
    """Multiple URL scraping request model."""
    urls: List[HttpUrl] = Field(..., description="URLs to scrape", min_items=1, max_items=20)
    extract_links: bool = Field(default=False, description="Extract all links")
    extract_images: bool = Field(default=False, description="Extract image URLs")
    extract_text: bool = Field(default=True, description="Extract text content")
    css_selector: Optional[str] = Field(default=None, description="CSS selector")
    timeout: int = Field(default=30, description="Request timeout", ge=5, le=120)


class WebResearchResponse(BaseModel):
    """Web research response model."""
    success: bool = Field(..., description="Operation success status")
    action: str = Field(..., description="Action performed")
    timestamp: datetime = Field(..., description="Response timestamp")
    data: Dict[str, Any] = Field(..., description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")


@router.post("/search", response_model=WebResearchResponse)
async def search_web(
    request: WebSearchRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> WebResearchResponse:
    """
    Search the web using the advanced web research tool.
    
    This endpoint allows you to search the web using multiple search engines
    and get structured results with URLs, titles, and snippets.
    
    Args:
        request: Web search parameters
        
    Returns:
        Search results with metadata
    """
    try:
        logger.info(f"Web search requested: {request.query}")
        
        # Get the web research tool
        web_tool = await get_web_research_tool()
        if not web_tool:
            raise HTTPException(status_code=500, detail="Web research tool not available")
        
        # Execute search
        result_json = await web_tool._arun(
            action="search",
            query=request.query,
            num_results=request.num_results,
            search_engine=request.search_engine,
            safe_search=request.safe_search,
            region=request.region
        )
        
        import json
        result_data = json.loads(result_json)
        
        return WebResearchResponse(
            success=result_data.get("success", False),
            action="search",
            timestamp=datetime.utcnow(),
            data=result_data,
            error=result_data.get("error")
        )
        
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return WebResearchResponse(
            success=False,
            action="search",
            timestamp=datetime.utcnow(),
            data={},
            error=str(e)
        )


@router.post("/scrape", response_model=WebResearchResponse)
async def scrape_webpage(
    request: WebScrapeRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> WebResearchResponse:
    """
    Scrape content from a webpage.
    
    This endpoint allows you to extract text, links, images, and metadata
    from any accessible webpage with advanced content extraction capabilities.
    
    Args:
        request: Web scraping parameters
        
    Returns:
        Extracted content and metadata
    """
    try:
        logger.info(f"Web scraping requested: {request.url}")
        
        # Get the web research tool
        web_tool = await get_web_research_tool()
        if not web_tool:
            raise HTTPException(status_code=500, detail="Web research tool not available")
        
        # Execute scraping
        result_json = await web_tool._arun(
            action="scrape",
            url=str(request.url),
            extract_links=request.extract_links,
            extract_images=request.extract_images,
            extract_text=request.extract_text,
            css_selector=request.css_selector,
            timeout=request.timeout
        )
        
        import json
        result_data = json.loads(result_json)
        
        return WebResearchResponse(
            success=result_data.get("success", False),
            action="scrape",
            timestamp=datetime.utcnow(),
            data=result_data,
            error=result_data.get("error")
        )
        
    except Exception as e:
        logger.error(f"Web scraping failed: {str(e)}")
        return WebResearchResponse(
            success=False,
            action="scrape",
            timestamp=datetime.utcnow(),
            data={},
            error=str(e)
        )


@router.post("/multi-scrape", response_model=WebResearchResponse)
async def scrape_multiple_webpages(
    request: MultiScrapeRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = Depends(get_current_user)
) -> WebResearchResponse:
    """
    Scrape content from multiple webpages.
    
    This endpoint allows you to scrape multiple URLs simultaneously
    with rate limiting and respectful crawling practices.
    
    Args:
        request: Multiple URL scraping parameters
        
    Returns:
        Combined results from all scraped URLs
    """
    try:
        logger.info(f"Multi-scraping requested for {len(request.urls)} URLs")
        
        # Get the web research tool
        web_tool = await get_web_research_tool()
        if not web_tool:
            raise HTTPException(status_code=500, detail="Web research tool not available")
        
        # Convert URLs to strings
        url_strings = [str(url) for url in request.urls]
        
        # Execute multi-scraping
        result_json = await web_tool._arun(
            action="multi_scrape",
            urls=url_strings,
            extract_links=request.extract_links,
            extract_images=request.extract_images,
            extract_text=request.extract_text,
            css_selector=request.css_selector,
            timeout=request.timeout
        )
        
        import json
        result_data = json.loads(result_json)
        
        return WebResearchResponse(
            success=result_data.get("success", False),
            action="multi_scrape",
            timestamp=datetime.utcnow(),
            data=result_data,
            error=result_data.get("error")
        )
        
    except Exception as e:
        logger.error(f"Multi-scraping failed: {str(e)}")
        return WebResearchResponse(
            success=False,
            action="multi_scrape",
            timestamp=datetime.utcnow(),
            data={},
            error=str(e)
        )


@router.get("/tools", response_model=Dict[str, Any])
async def list_web_tools(
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all available web research tools.
    
    Returns:
        Information about available web tools and their capabilities
    """
    try:
        tools = web_tools_registry.list_tools()
        
        tool_info = {}
        for tool_name in tools:
            tool = web_tools_registry.get_tool(tool_name)
            if tool:
                tool_info[tool_name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "available": True
                }
        
        return {
            "success": True,
            "tools": tool_info,
            "total_tools": len(tools),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list web tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.get("/tool/{tool_name}/info", response_model=Dict[str, Any])
async def get_tool_info(
    tool_name: str,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific web tool.
    
    Args:
        tool_name: Name of the tool to get info for
        
    Returns:
        Detailed tool information including capabilities and usage examples
    """
    try:
        tool = web_tools_registry.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        # Get tool metadata from production registry
        from app.tools.production_tool_system import production_tool_registry
        metadata = production_tool_registry.get_tool_metadata(tool_name)
        
        return {
            "success": True,
            "tool_name": tool_name,
            "description": tool.description,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tool info for {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool info: {str(e)}")


@router.post("/validate-url", response_model=Dict[str, Any])
async def validate_url(
    url: HttpUrl,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate a URL and check its accessibility.
    
    Args:
        url: URL to validate
        
    Returns:
        Validation results including accessibility status
    """
    try:
        # Get URL validator tool
        url_validator = web_tools_registry.get_tool("url_validator")
        if not url_validator:
            raise HTTPException(status_code=500, detail="URL validator not available")
        
        # Validate URL
        result = await url_validator.arun(str(url))
        
        return {
            "success": True,
            "url": str(url),
            "validation_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"URL validation failed: {str(e)}")
        return {
            "success": False,
            "url": str(url),
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Health check endpoint
@router.get("/health")
async def web_research_health() -> Dict[str, Any]:
    """
    Check the health of web research tools.
    
    Returns:
        Health status of web research capabilities
    """
    try:
        # Check if web research tool is available
        web_tool = await get_web_research_tool()
        tools_available = web_tools_registry.list_tools()
        
        return {
            "status": "healthy",
            "web_research_tool_available": web_tool is not None,
            "total_tools": len(tools_available),
            "available_tools": tools_available,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Web research health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
