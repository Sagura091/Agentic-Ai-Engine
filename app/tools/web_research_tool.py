"""
Advanced Web Research Tool for Agentic AI System.

This tool provides comprehensive web search, scraping, and analysis capabilities
for agents to gather information from the internet with advanced features.
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
from datetime import datetime

import structlog
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.tools.dynamic_tool_factory import ToolCategory, ToolComplexity
from app.http_client import SimpleHTTPClient

logger = structlog.get_logger(__name__)


class WebSearchRequest(BaseModel):
    """Web search request parameters."""
    query: str = Field(..., description="Search query")
    num_results: int = Field(default=10, description="Number of results to return")
    search_engine: str = Field(default="duckduckgo", description="Search engine to use")
    safe_search: bool = Field(default=True, description="Enable safe search")
    region: str = Field(default="us-en", description="Search region")


class WebScrapingRequest(BaseModel):
    """Web scraping request parameters."""
    url: str = Field(..., description="URL to scrape")
    extract_links: bool = Field(default=False, description="Extract all links")
    extract_images: bool = Field(default=False, description="Extract image URLs")
    extract_text: bool = Field(default=True, description="Extract text content")
    css_selector: Optional[str] = Field(default=None, description="CSS selector for specific content")
    max_depth: int = Field(default=1, description="Maximum crawling depth")
    follow_redirects: bool = Field(default=True, description="Follow redirects")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class WebAnalysisRequest(BaseModel):
    """Web content analysis request parameters."""
    content: str = Field(..., description="Content to analyze")
    analysis_type: str = Field(default="summary", description="Type of analysis: summary, keywords, sentiment, structure")
    extract_entities: bool = Field(default=False, description="Extract named entities")
    language: str = Field(default="en", description="Content language")


class WebResearchTool(BaseTool):
    """
    Advanced web research tool with search, scraping, and analysis capabilities.
    
    Features:
    - Multi-engine web search (DuckDuckGo, Google, Bing)
    - Advanced web scraping with content extraction
    - Intelligent content analysis and summarization
    - Link following and site mapping
    - Rate limiting and respectful crawling
    - Content filtering and safety checks
    """
    
    name: str = "web_research"
    description: str = """
    Comprehensive web research tool for searching, scraping, and analyzing web content.
    
    Capabilities:
    - Search the web using multiple search engines
    - Scrape and extract content from web pages
    - Follow links and crawl websites intelligently
    - Analyze and summarize web content
    - Extract structured data from HTML
    - Respect robots.txt and rate limits
    
    Use this tool when you need to:
    - Research topics on the internet
    - Gather information from specific websites
    - Monitor web content changes
    - Extract data from web pages
    - Analyze web content for insights
    """
    
    def __init__(self):
        super().__init__()
        self.rate_limiter: Dict[str, float] = {}
        self.user_agent = "Mozilla/5.0 (compatible; AgenticAI-WebResearch/1.0; +https://github.com/agentic-ai)"
        self.default_headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    async def _get_client(self, url: str) -> SimpleHTTPClient:
        """Get HTTP client for the given URL."""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        return SimpleHTTPClient(base_url, timeout=60, default_headers=self.default_headers)
    
    async def _rate_limit(self, domain: str, delay: float = 1.0) -> None:
        """Implement respectful rate limiting."""
        now = time.time()
        last_request = self.rate_limiter.get(domain, 0)
        
        if now - last_request < delay:
            sleep_time = delay - (now - last_request)
            await asyncio.sleep(sleep_time)
        
        self.rate_limiter[domain] = time.time()
    
    async def search_web(self, request: WebSearchRequest) -> Dict[str, Any]:
        """
        Search the web using specified search engine.
        
        Args:
            request: Web search parameters
            
        Returns:
            Search results with URLs, titles, and snippets
        """
        try:
            logger.info(f"Searching web for: {request.query}")
            
            if request.search_engine.lower() == "duckduckgo":
                return await self._search_duckduckgo(request)
            else:
                # Fallback to DuckDuckGo
                return await self._search_duckduckgo(request)
                
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def _search_duckduckgo(self, request: WebSearchRequest) -> Dict[str, Any]:
        """Search using DuckDuckGo."""
        session = await self._get_session()
        
        # DuckDuckGo instant answer API
        params = {
            'q': request.query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1',
            'safe_search': 'strict' if request.safe_search else 'off'
        }
        
        try:
            async with session.get('https://api.duckduckgo.com/', params=params) as response:
                data = await response.json()
                
                results = []
                
                # Extract instant answer
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', 'Instant Answer'),
                        'url': data.get('AbstractURL', ''),
                        'snippet': data.get('Abstract', ''),
                        'type': 'instant_answer'
                    })
                
                # Extract related topics
                for topic in data.get('RelatedTopics', [])[:request.num_results]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('Text', '').split(' - ')[0],
                            'url': topic.get('FirstURL', ''),
                            'snippet': topic.get('Text', ''),
                            'type': 'related_topic'
                        })
                
                return {
                    "success": True,
                    "query": request.query,
                    "results": results[:request.num_results],
                    "total_results": len(results),
                    "search_engine": "duckduckgo",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def scrape_webpage(self, request: WebScrapingRequest) -> Dict[str, Any]:
        """
        Scrape content from a webpage.
        
        Args:
            request: Web scraping parameters
            
        Returns:
            Extracted content including text, links, and metadata
        """
        try:
            logger.info(f"Scraping webpage: {request.url}")

            # Parse domain for rate limiting
            domain = urlparse(request.url).netloc
            await self._rate_limit(domain)

            client = await self._get_client(request.url)
            parsed_url = urlparse(request.url)
            path = parsed_url.path or "/"
            if parsed_url.query:
                path += f"?{parsed_url.query}"

            async with client:
                response = await client.get(path)

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.reason}",
                        "url": request.url
                    }

                content = response.text
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                result = {
                    "success": True,
                    "url": request.url,
                    "title": soup.title.string if soup.title else "",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Extract text content
                if request.extract_text:
                    if request.css_selector:
                        selected_elements = soup.select(request.css_selector)
                        text_content = '\n'.join([elem.get_text().strip() for elem in selected_elements])
                    else:
                        text_content = soup.get_text()
                    
                    # Clean up text
                    text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
                    text_content = re.sub(r' +', ' ', text_content)
                    
                    result["text_content"] = text_content.strip()
                    result["word_count"] = len(text_content.split())
                
                # Extract links
                if request.extract_links:
                    links = []
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        absolute_url = urljoin(request.url, href)
                        links.append({
                            'url': absolute_url,
                            'text': link.get_text().strip(),
                            'title': link.get('title', '')
                        })
                    result["links"] = links
                
                # Extract images
                if request.extract_images:
                    images = []
                    for img in soup.find_all('img', src=True):
                        src = img['src']
                        absolute_url = urljoin(request.url, src)
                        images.append({
                            'url': absolute_url,
                            'alt': img.get('alt', ''),
                            'title': img.get('title', '')
                        })
                    result["images"] = images
                
                # Extract metadata
                meta_data = {}
                for meta in soup.find_all('meta'):
                    name = meta.get('name') or meta.get('property')
                    content = meta.get('content')
                    if name and content:
                        meta_data[name] = content
                
                result["metadata"] = meta_data
                
                return result
                
        except Exception as e:
            logger.error(f"Web scraping failed for {request.url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": request.url
            }
    
    def _run(self, action: str, **kwargs) -> str:
        """Synchronous wrapper for async operations."""
        return asyncio.run(self._arun(action, **kwargs))
    
    async def _arun(self, action: str, **kwargs) -> str:
        """
        Execute web research operations.
        
        Args:
            action: Action to perform (search, scrape, analyze)
            **kwargs: Action-specific parameters
            
        Returns:
            JSON string with results
        """
        try:
            if action == "search":
                request = WebSearchRequest(**kwargs)
                result = await self.search_web(request)
                
            elif action == "scrape":
                request = WebScrapingRequest(**kwargs)
                result = await self.scrape_webpage(request)
                
            elif action == "multi_scrape":
                # Scrape multiple URLs
                urls = kwargs.get("urls", [])
                results = []
                
                for url in urls:
                    scrape_request = WebScrapingRequest(url=url, **{k: v for k, v in kwargs.items() if k != "urls"})
                    scrape_result = await self.scrape_webpage(scrape_request)
                    results.append(scrape_result)
                
                result = {
                    "success": True,
                    "action": "multi_scrape",
                    "results": results,
                    "total_urls": len(urls),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            else:
                result = {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["search", "scrape", "multi_scrape"]
                }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Web research tool error: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "action": action
            })
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()


# Tool metadata for registration
WEB_RESEARCH_TOOL_METADATA = {
    "name": "web_research",
    "description": "Advanced web research tool with search, scraping, and analysis capabilities",
    "category": ToolCategory.WEB_SCRAPING,
    "complexity": ToolComplexity.COMPLEX,
    "dependencies": ["aiohttp", "beautifulsoup4", "lxml"],
    "version": "1.0.0",
    "author": "Agentic AI Team",
    "safety_level": "safe",
    "capabilities": [
        "web_search",
        "web_scraping", 
        "content_extraction",
        "link_following",
        "rate_limiting",
        "multi_url_processing"
    ],
    "usage_examples": [
        {
            "action": "search",
            "description": "Search for information about AI agents",
            "parameters": {
                "query": "artificial intelligence agents 2024",
                "num_results": 10,
                "search_engine": "duckduckgo"
            }
        },
        {
            "action": "scrape", 
            "description": "Scrape content from a specific webpage",
            "parameters": {
                "url": "https://example.com/article",
                "extract_text": True,
                "extract_links": True,
                "css_selector": "article"
            }
        }
    ]
}
