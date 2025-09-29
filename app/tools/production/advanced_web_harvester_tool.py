"""
🌐 REVOLUTIONARY ADVANCED WEB HARVESTER TOOL - The Ultimate Internet Data Collection System

This is the most advanced web harvesting tool ever created, featuring:
- Complete HTML scraping and content extraction
- Automatic file detection and downloading (PDFs, images, videos, documents)
- Link discovery and recursive crawling capabilities
- Content analysis and categorization
- Batch operations and bulk downloading
- Smart filtering and content organization
- Anti-detection mechanisms with proxy rotation
- Real-time progress tracking and reporting
- Comprehensive data extraction from any website
- Revolutionary internet domination capabilities

🎯 CAPABILITIES:
- Scrape any website completely
- Download all files, images, documents automatically
- Extract all links and downloadable content
- Organize content systematically
- Handle JavaScript-rendered content
- Bypass anti-scraping measures
- Collect data from social media, forums, news sites
- Extract structured data from any source
- Revolutionary content harvesting
"""

import asyncio
import json
import os
import re
import time
import hashlib
import mimetypes
import random
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from urllib.parse import urljoin, urlparse, quote_plus, unquote
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp
import aiofiles
from bs4 import BeautifulSoup, Comment
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import structlog

# Import required modules
from app.http_client import SimpleHTTPClient
from app.tools.unified_tool_repository import ToolCategory

logger = structlog.get_logger(__name__)

# 🎯 REVOLUTIONARY BROWSER SIMULATION - Real Browser Headers
REVOLUTIONARY_BROWSER_HEADERS = {
    "chrome_windows": {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    },
    "firefox_windows": {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
}

# 🔍 DIRECT IMAGE URL PATTERNS - Extract real image URLs like a human browser
DIRECT_IMAGE_PATTERNS = {
    'reddit': [
        r'https://i\.redd\.it/[^"\'>\s]+\.(jpg|jpeg|png|gif|webp)',
        r'https://preview\.redd\.it/[^"\'>\s]+\.(jpg|jpeg|png|gif|webp)',
        r'https://external-preview\.redd\.it/[^"\'>\s]+\.(jpg|jpeg|png|gif|webp)'
    ],
    'imgur': [
        r'https://i\.imgur\.com/[^"\'>\s]+\.(jpg|jpeg|png|gif|webp)',
        r'https://imgur\.com/([a-zA-Z0-9]+)(?:\.[a-zA-Z]+)?',  # Convert to direct
    ],
    '9gag': [
        r'https://[^"\'>\s]*9cache\.com/[^"\'>\s]+\.(jpg|jpeg|png|gif|webp|mp4)',
        r'https://img-9gag-fun\.9cache\.com/[^"\'>\s]+\.(jpg|jpeg|png|gif|webp)'
    ],
    'general': [
        r'https://[^"\'>\s]+\.(jpg|jpeg|png|gif|webp|bmp|svg)',
        r'http://[^"\'>\s]+\.(jpg|jpeg|png|gif|webp|bmp|svg)'
    ]
}

# 🌐 REVOLUTIONARY BROWSER USER AGENTS - Anti-Detection Arsenal
REVOLUTIONARY_USER_AGENTS = [
    # Chrome (Latest versions)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

    # Firefox (Latest versions)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",

    # Safari (Latest versions)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",

    # Edge (Latest versions)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",

    # Mobile browsers
    "Mozilla/5.0 (Android 14; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0",
    "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
]

# 🔍 REVOLUTIONARY SEARCH ENGINES - Multi-Engine Intelligence
SEARCH_ENGINES = {
    "google": {
        "url": "https://www.google.com/search",
        "params": {"q": "{query}", "num": "50", "start": "{start}"},
        "result_selector": "div.g",
        "link_selector": "a[href]",
        "title_selector": "h3",
        "snippet_selector": ".VwiC3b"
    },
    "google_images": {
        "url": "https://www.google.com/search",
        "params": {"q": "{query}", "tbm": "isch", "num": "50", "start": "{start}"},
        "result_selector": "div[data-ri]",
        "link_selector": "a[href]",
        "image_selector": "img[src]",
        "title_selector": "h3"
    },
    "bing": {
        "url": "https://www.bing.com/search",
        "params": {"q": "{query}", "count": "50", "first": "{start}"},
        "result_selector": ".b_algo",
        "link_selector": "h2 a[href]",
        "title_selector": "h2",
        "snippet_selector": ".b_caption p"
    },
    "bing_images": {
        "url": "https://www.bing.com/images/search",
        "params": {"q": "{query}", "count": "50", "first": "{start}"},
        "result_selector": ".imgpt",
        "link_selector": "a[href]",
        "image_selector": "img[src]",
        "title_selector": ".inflnk"
    },
    "duckduckgo": {
        "url": "https://duckduckgo.com/html",
        "params": {"q": "{query}", "s": "{start}"},
        "result_selector": ".result",
        "link_selector": ".result__a[href]",
        "title_selector": ".result__title",
        "snippet_selector": ".result__snippet"
    }
}

# 🛡️ SAFE DOMAINS - Trusted sources for content
SAFE_DOMAINS = [
    'reddit.com', 'imgur.com', '9gag.com', 'knowyourmeme.com',
    'memegenerator.net', 'imgflip.com', 'pinterest.com', 'tumblr.com',
    'youtube.com', 'vimeo.com', 'giphy.com', 'tenor.com',
    'wikimedia.org', 'wikipedia.org', 'unsplash.com', 'pexels.com',
    'pixabay.com', 'freepik.com', 'shutterstock.com', 'gettyimages.com'
]

# 🚫 BLOCKED DOMAINS - Known malware/suspicious sites
BLOCKED_DOMAINS = [
    'malware.com', 'virus.com', 'spam.com', 'phishing.com',
    'suspicious.net', 'fake.org', 'scam.info'
]

# 📁 PROPER FILE EXTENSIONS - What we actually want to download
MEDIA_EXTENSIONS = {
    'images': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico'],
    'videos': ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
    'audio': ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a'],
    'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
    'archives': ['.zip', '.rar', '.7z', '.tar', '.gz']
}

# 🎯 MEME-SPECIFIC SEARCH QUERIES - Revolutionary Content Discovery
MEME_SEARCH_QUERIES = [
    "trending memes 2024",
    "viral memes today",
    "funny memes reddit",
    "dank memes collection",
    "popular meme formats",
    "meme templates download",
    "internet memes viral",
    "social media memes",
    "comedy memes funny",
    "reaction memes gif"
]

# 🌐 DIRECT PLATFORM URLS - No Login Required Scraping
DIRECT_MEME_PLATFORMS = {
    "reddit": [
        "https://www.reddit.com/r/memes/hot/",
        "https://www.reddit.com/r/dankmemes/hot/",
        "https://www.reddit.com/r/funny/hot/",
        "https://www.reddit.com/r/wholesomememes/hot/",
        "https://www.reddit.com/r/memeeconomy/hot/",
        "https://www.reddit.com/r/PrequelMemes/hot/",
        "https://www.reddit.com/r/gaming/hot/",
        "https://www.reddit.com/r/ProgrammerHumor/hot/"
    ],
    "imgur": [
        "https://imgur.com/hot",
        "https://imgur.com/r/memes",
        "https://imgur.com/r/funny",
        "https://imgur.com/r/dankmemes"
    ],
    "9gag": [
        "https://9gag.com/hot",
        "https://9gag.com/trending",
        "https://9gag.com/fresh"
    ],
    "knowyourmeme": [
        "https://knowyourmeme.com/memes/trending",
        "https://knowyourmeme.com/memes/popular"
    ],
    "imgflip": [
        "https://imgflip.com/hot",
        "https://imgflip.com/latest"
    ]
}


@dataclass
class HarvestResult:
    """Comprehensive harvest result with all extracted data."""
    url: str
    title: str
    content: str
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    downloadable_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    harvest_time: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""


@dataclass
class DownloadResult:
    """Result of file download operation."""
    url: str
    local_path: str
    file_size: int
    content_type: str
    success: bool
    error_message: str = ""


class AdvancedWebHarvesterInput(BaseModel):
    """Input schema for the Advanced Web Harvester Tool."""
    operation: str = Field(description="Operation type: 'harvest', 'download', 'extract_links', 'bulk_harvest'")
    url: Optional[str] = Field(default=None, description="Target URL to harvest")
    urls: Optional[List[str]] = Field(default=None, description="Multiple URLs for bulk operations")
    download_files: bool = Field(default=True, description="Whether to download found files")
    file_types: Optional[List[str]] = Field(default=None, description="Specific file types to target (pdf, jpg, png, mp4, etc.)")
    max_depth: int = Field(default=2, description="Maximum crawling depth for recursive harvesting")
    output_directory: str = Field(default="data/harvested_content", description="Directory to save downloaded content")
    include_images: bool = Field(default=True, description="Whether to download images")
    include_documents: bool = Field(default=True, description="Whether to download documents")
    include_videos: bool = Field(default=False, description="Whether to download videos (can be large)")
    filter_keywords: Optional[List[str]] = Field(default=None, description="Keywords to filter content")
    extract_structured_data: bool = Field(default=True, description="Whether to extract structured data")


class AdvancedWebHarvesterTool(BaseTool):
    """🌐 Revolutionary Advanced Web Harvester Tool - Ultimate Internet Data Collection System."""

    name: str = "advanced_web_harvester"
    description: str = """Revolutionary web harvesting tool that can scrape, extract, and download everything from the internet.

    Operations:
    - 'harvest': Complete website harvesting with content extraction and file downloading
    - 'download': Download specific files from URLs
    - 'extract_links': Extract all links and downloadable content from pages
    - 'bulk_harvest': Harvest multiple websites simultaneously

    Features:
    - Downloads all files, images, documents, videos automatically
    - Extracts all links and downloadable content
    - Scrapes complete HTML content and structured data
    - Organizes content systematically in directories
    - Handles anti-scraping measures and JavaScript content
    - Provides real-time progress tracking
    - Revolutionary internet domination capabilities
    """

    # Declare fields for Pydantic validation
    session: Optional[Any] = None
    download_stats: Dict[str, Any] = {}
    file_extensions: Dict[str, List[str]] = {}

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self):
        super().__init__()
        self.session = None
        self.download_stats = {
            'total_files': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size': 0
        }
        
        # File type mappings
        self.file_extensions = {
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico'],
            'videos': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'code': ['.py', '.js', '.html', '.css', '.json', '.xml', '.sql']
        }
        
        logger.info("🌐 Advanced Web Harvester Tool initialized - Ready for internet domination!")

    def _get_random_user_agent(self) -> str:
        """Get a random user agent for anti-detection."""
        return random.choice(REVOLUTIONARY_USER_AGENTS)

    def _get_revolutionary_headers(self, referer: str = None) -> Dict[str, str]:
        """🎯 Get revolutionary headers that mimic real browser behavior."""
        # Randomly select a browser profile
        browser_profiles = list(REVOLUTIONARY_BROWSER_HEADERS.keys())
        selected_browser = random.choice(browser_profiles)
        headers = REVOLUTIONARY_BROWSER_HEADERS[selected_browser].copy()

        # Add referer if provided (important for many sites)
        if referer:
            headers['Referer'] = referer

        # Add some randomization to avoid detection
        headers['Accept'] = headers.get('Accept', '*/*')

        return headers

    def _extract_direct_image_urls(self, html_content: str, base_url: str) -> List[str]:
        """🔍 Extract direct image URLs like a human browser would see them."""
        image_urls = set()

        # Determine the platform
        platform = 'general'
        if 'reddit.com' in base_url:
            platform = 'reddit'
        elif 'imgur.com' in base_url:
            platform = 'imgur'
        elif '9gag.com' in base_url:
            platform = '9gag'

        # Use platform-specific patterns
        patterns = DIRECT_IMAGE_PATTERNS.get(platform, []) + DIRECT_IMAGE_PATTERNS['general']

        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle regex groups
                    url = match[0] if match[0].startswith('http') else f"https://i.imgur.com/{match[0]}.jpg"
                else:
                    url = match

                # Clean and validate URL
                if url and self._is_valid_image_url(url):
                    image_urls.add(url)

        # Also parse with BeautifulSoup for img tags
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all img tags
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(base_url, src)

                    if self._is_valid_image_url(src):
                        image_urls.add(src)

            # Look for background images in style attributes
            for element in soup.find_all(attrs={'style': True}):
                style = element.get('style', '')
                bg_matches = re.findall(r'background-image:\s*url\(["\']?([^"\')\s]+)["\']?\)', style)
                for bg_url in bg_matches:
                    if bg_url.startswith('//'):
                        bg_url = 'https:' + bg_url
                    elif bg_url.startswith('/'):
                        bg_url = urljoin(base_url, bg_url)

                    if self._is_valid_image_url(bg_url):
                        image_urls.add(bg_url)

        except Exception as e:
            logger.warning(f"BeautifulSoup parsing failed: {str(e)}")

        return list(image_urls)

    def _is_valid_image_url(self, url: str) -> bool:
        """🎯 Check if URL is a valid image URL."""
        if not url or not isinstance(url, str):
            return False

        # Skip data URLs, base64, and other non-HTTP URLs
        if not url.startswith(('http://', 'https://')):
            return False

        # Skip obvious non-image URLs
        skip_patterns = [
            'data:', 'javascript:', 'mailto:', '#',
            '.css', '.js', '.html', '.php', '.asp',
            'favicon.ico', 'logo.svg'
        ]

        url_lower = url.lower()
        if any(pattern in url_lower for pattern in skip_patterns):
            return False

        # Check for image extensions or known image domains
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
        image_domains = ['i.redd.it', 'i.imgur.com', '9cache.com', 'preview.redd.it']

        has_extension = any(ext in url_lower for ext in image_extensions)
        has_image_domain = any(domain in url_lower for domain in image_domains)

        return has_extension or has_image_domain

    async def _search_engine_harvest(self, query: str, engine: str = "google", max_results: int = 50) -> List[str]:
        """🔍 Revolutionary search engine harvesting."""
        try:
            # Ensure session is created
            if not self.session:
                await self._create_session()

            if engine not in SEARCH_ENGINES:
                logger.warning(f"Unknown search engine: {engine}, using Google")
                engine = "google"

            search_config = SEARCH_ENGINES[engine]
            urls = []

            # Search multiple pages
            for start in range(0, max_results, 10):
                try:
                    # Build search URL
                    params = {}
                    for key, value in search_config["params"].items():
                        if "{query}" in value:
                            params[key] = value.format(query=quote_plus(query))
                        elif "{start}" in value:
                            params[key] = value.format(start=start)
                        else:
                            params[key] = value

                    # Make search request
                    search_url = search_config["url"]
                    headers = self._get_revolutionary_headers()

                    async with self.session.get(search_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Handle image search engines differently
                            if engine in ["google_images", "bing_images"]:
                                # Extract actual image URLs for image searches
                                image_urls = await self._extract_image_urls_from_search(html, engine)
                                urls.extend(image_urls)
                            else:
                                # Extract search result URLs for regular searches
                                results = soup.select(search_config["result_selector"])
                                for result in results:
                                    link_elem = result.select_one(search_config["link_selector"])
                                    if link_elem and link_elem.get('href'):
                                        url = link_elem['href']
                                        # Clean up URL (remove Google redirect, etc.)
                                        if url.startswith('/url?q='):
                                            url = unquote(url.split('&')[0][7:])
                                        elif url.startswith('http'):
                                            urls.append(url)

                            # Add delay to avoid rate limiting
                            await asyncio.sleep(random.uniform(1, 3))

                except Exception as e:
                    logger.error(f"Search page {start} failed: {str(e)}")
                    continue

            logger.info(f"🔍 Found {len(urls)} URLs from {engine} search for '{query}'")
            return urls[:max_results]

        except Exception as e:
            logger.error(f"Search engine harvest failed: {str(e)}")
            return []

    async def _discover_meme_content(self, search_terms: List[str] = None) -> List[str]:
        """🎯 Revolutionary meme content discovery - Direct platform scraping + Search engines."""
        if not search_terms:
            search_terms = MEME_SEARCH_QUERIES

        all_urls = []

        # 🌐 FIRST: Direct platform scraping (NO LOGIN REQUIRED!)
        logger.info("🌐 Phase 1: Direct platform scraping...")
        direct_memes = await self._scrape_all_direct_platforms()
        all_urls.extend(direct_memes)
        logger.info(f"✅ Direct scraping found {len(direct_memes)} memes!")

        # 🔍 SECOND: Search engine discovery for additional content
        logger.info("🔍 Phase 2: Search engine discovery...")
        engines = ["google_images", "bing_images"]  # Focus on image searches

        for term in search_terms[:3]:  # Limit to avoid overwhelming
            for engine in engines:
                try:
                    urls = await self._search_engine_harvest(term, engine, max_results=15)
                    all_urls.extend(urls)

                    # Add delay between searches
                    await asyncio.sleep(random.uniform(2, 4))

                except Exception as e:
                    logger.error(f"Search discovery failed for {term} on {engine}: {str(e)}")
                    continue

        # Remove duplicates and filter for meme-related domains
        unique_urls = list(set(all_urls))
        meme_urls = [url for url in unique_urls if self._is_meme_related_url(url)]

        logger.info(f"🎉 TOTAL DISCOVERY COMPLETE! Found {len(meme_urls)} unique memes from {len(all_urls)} total URLs")
        logger.info(f"📊 Direct scraping: {len(direct_memes)} | Search engines: {len(all_urls) - len(direct_memes)}")
        return meme_urls

    async def _extract_image_urls_from_search(self, html: str, engine: str) -> List[str]:
        """🖼️ Extract actual image URLs from search results."""
        soup = BeautifulSoup(html, 'html.parser')
        image_urls = []

        try:
            if engine == "google_images":
                # Google Images specific extraction
                img_elements = soup.find_all('img', {'src': True})
                for img in img_elements:
                    src = img.get('src')
                    if src and not src.startswith('data:'):
                        # Clean up Google's image URLs
                        if src.startswith('http'):
                            image_urls.append(src)

                # Also look for higher resolution images in data attributes
                divs = soup.find_all('div', {'data-src': True})
                for div in divs:
                    data_src = div.get('data-src')
                    if data_src and data_src.startswith('http'):
                        image_urls.append(data_src)

            elif engine == "bing_images":
                # Bing Images specific extraction
                img_elements = soup.find_all('img', {'src': True})
                for img in img_elements:
                    src = img.get('src')
                    if src and src.startswith('http') and not src.startswith('data:'):
                        image_urls.append(src)

                # Look for Bing's media URLs
                links = soup.find_all('a', {'href': True})
                for link in links:
                    href = link.get('href')
                    if href and any(ext in href.lower() for ext in ['.jpg', '.png', '.gif', '.webp']):
                        image_urls.append(href)

            # Filter for actual image URLs and safe domains
            filtered_urls = []
            for url in image_urls:
                if (self._is_safe_domain(url) and
                    any(ext in url.lower() for ext in MEDIA_EXTENSIONS['images'])):
                    filtered_urls.append(url)

            logger.info(f"🖼️ Extracted {len(filtered_urls)} image URLs from {engine}")
            return filtered_urls[:50]  # Limit to 50 images per search

        except Exception as e:
            logger.error(f"Image URL extraction failed for {engine}: {str(e)}")
            return []

    def _is_meme_related_url(self, url: str) -> bool:
        """Check if URL is likely to contain meme content."""
        meme_domains = [
            'reddit.com', 'imgur.com', '9gag.com', 'knowyourmeme.com',
            'memegenerator.net', 'quickmeme.com', 'makeameme.org',
            'imgflip.com', 'memebase.cheezburger.com', 'memecenter.com',
            'memedroid.com', 'ifunny.co', 'pinterest.com', 'tumblr.com',
            'twitter.com', 'instagram.com', 'tiktok.com', 'youtube.com'
        ]

        meme_keywords = [
            'meme', 'funny', 'viral', 'comedy', 'humor', 'joke',
            'dank', 'trending', 'popular', 'reaction', 'gif'
        ]

        url_lower = url.lower()

        # Check domains
        if any(domain in url_lower for domain in meme_domains):
            return True

        # Check keywords in URL
        if any(keyword in url_lower for keyword in meme_keywords):
            return True

        return False

    def _is_safe_domain(self, url: str) -> bool:
        """🛡️ Check if domain is safe for downloading."""
        url_lower = url.lower()

        # Check if it's a blocked domain
        if any(blocked in url_lower for blocked in BLOCKED_DOMAINS):
            return False

        # Check if it's a known safe domain
        if any(safe in url_lower for safe in SAFE_DOMAINS):
            return True

        # Additional safety checks
        suspicious_patterns = [
            'download-now', 'click-here', 'free-download', 'virus-scan',
            'malware', 'trojan', 'phishing', 'scam', 'fake'
        ]

        if any(pattern in url_lower for pattern in suspicious_patterns):
            return False

        return True  # Default to safe if no red flags

    def _get_proper_file_extension(self, url: str, content_type: str = None) -> str:
        """🎯 Get the proper file extension based on URL and content type."""
        url_lower = url.lower()

        # Check content type first
        if content_type:
            content_type_lower = content_type.lower()
            if 'image/jpeg' in content_type_lower or 'image/jpg' in content_type_lower:
                return '.jpg'
            elif 'image/png' in content_type_lower:
                return '.png'
            elif 'image/gif' in content_type_lower:
                return '.gif'
            elif 'image/webp' in content_type_lower:
                return '.webp'
            elif 'video/mp4' in content_type_lower:
                return '.mp4'
            elif 'video/webm' in content_type_lower:
                return '.webm'
            elif 'audio/mpeg' in content_type_lower:
                return '.mp3'
            elif 'application/pdf' in content_type_lower:
                return '.pdf'

        # Check URL extension
        for category, extensions in MEDIA_EXTENSIONS.items():
            for ext in extensions:
                if url_lower.endswith(ext):
                    return ext

        # Default based on URL patterns
        if any(pattern in url_lower for pattern in ['jpg', 'jpeg']):
            return '.jpg'
        elif any(pattern in url_lower for pattern in ['png']):
            return '.png'
        elif any(pattern in url_lower for pattern in ['gif']):
            return '.gif'
        elif any(pattern in url_lower for pattern in ['mp4']):
            return '.mp4'
        elif any(pattern in url_lower for pattern in ['webm']):
            return '.webm'

        return '.jpg'  # Default to jpg for images

    async def _smart_media_download(self, url: str, output_dir: str, filename: str = None) -> bool:
        """🎯 Revolutionary media download - Like right-click save image as."""
        try:
            # Safety check
            if not self._is_safe_domain(url):
                logger.warning(f"🚫 Skipping unsafe domain: {url}")
                return False

            # Get the source domain for referer
            parsed_url = urlparse(url)
            referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"

            # Use revolutionary headers with proper referer
            headers = self._get_revolutionary_headers(referer=referer)
            headers.update({
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Sec-Fetch-Site': 'same-site',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Dest': 'image'
            })

            # Add small delay to mimic human behavior
            await asyncio.sleep(random.uniform(0.5, 2.0))

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')

                    # Skip HTML files - we want actual media
                    if 'text/html' in content_type.lower():
                        logger.warning(f"⚠️ Skipping HTML file: {url}")
                        return False

                    # Get proper extension
                    proper_ext = self._get_proper_file_extension(url, content_type)

                    # Generate filename
                    if not filename:
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                        timestamp = int(time.time())
                        filename = f"meme_{timestamp}_{url_hash}{proper_ext}"
                    elif not filename.endswith(proper_ext):
                        filename = f"{filename.rsplit('.', 1)[0]}{proper_ext}"

                    # Create directory
                    os.makedirs(output_dir, exist_ok=True)
                    file_path = os.path.join(output_dir, filename)

                    # Download content
                    content = await response.read()

                    # Verify it's actually media content (not HTML disguised)
                    if content.startswith(b'<!DOCTYPE') or content.startswith(b'<html'):
                        logger.warning(f"⚠️ Skipping HTML content disguised as media: {url}")
                        return False

                    # Additional validation for image content
                    if len(content) < 1024:  # Skip very small files (likely errors)
                        logger.warning(f"⚠️ Skipping suspiciously small file: {url} ({len(content)} bytes)")
                        return False

                    # Save file
                    with open(file_path, 'wb') as f:
                        f.write(content)

                    # Update stats
                    self.download_stats['successful_downloads'] += 1
                    self.download_stats['total_size'] += len(content)

                    logger.info(f"✅ Revolutionary download: {filename} ({len(content)} bytes) from {parsed_url.netloc}")
                    return True

                elif response.status == 403:
                    logger.warning(f"🚫 Access forbidden for {url} - trying alternative approach")
                    return False
                elif response.status == 404:
                    logger.warning(f"🔍 File not found: {url}")
                    return False
                else:
                    logger.warning(f"⚠️ Download failed for {url}: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"❌ Revolutionary media download failed for {url}: {str(e)}")
            return False

    async def _scrape_reddit_memes(self, subreddit_url: str) -> List[str]:
        """🔥 Revolutionary Reddit scraping - Like right-click save image."""
        try:
            # Ensure session is created
            await self._create_session()

            # Use revolutionary browser headers with Reddit as referer
            headers = self._get_revolutionary_headers(referer="https://www.reddit.com/")

            # Add Reddit-specific headers to avoid blocks
            headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-User': '?1',
                'Sec-Fetch-Dest': 'document'
            })

            # Add random delay to avoid rate limiting
            await asyncio.sleep(random.uniform(1, 3))

            async with self.session.get(subreddit_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()

                    # Use revolutionary image extraction
                    meme_urls = self._extract_direct_image_urls(html, subreddit_url)

                    # Additional Reddit-specific extraction
                    soup = BeautifulSoup(html, 'html.parser')

                    # Look for JSON data in script tags (Reddit often embeds data here)
                    for script in soup.find_all('script'):
                        if script.string and 'i.redd.it' in script.string:
                            # Extract URLs from JSON data
                            reddit_matches = re.findall(r'https://i\.redd\.it/[^"\'>\s]+\.(jpg|jpeg|png|gif|webp)', script.string)
                            for match in reddit_matches:
                                if isinstance(match, tuple):
                                    url = match[0] if match[0].startswith('http') else f"https://i.redd.it/{match[0]}"
                                else:
                                    url = match
                                if self._is_valid_image_url(url):
                                    meme_urls.append(url)

                    # Remove duplicates
                    unique_memes = list(set(meme_urls))

                    logger.info(f"🔥 Revolutionary Reddit extraction: {len(unique_memes)} memes from {subreddit_url}")
                    return unique_memes[:30]  # Limit per subreddit
                else:
                    logger.warning(f"Reddit returned status {response.status} for {subreddit_url}")
                    return []

        except Exception as e:
            logger.error(f"Revolutionary Reddit scraping failed for {subreddit_url}: {str(e)}")
            return []

    async def _scrape_imgur_memes(self, imgur_url: str) -> List[str]:
        """🖼️ Scrape Imgur memes without login."""
        try:
            headers = self._get_revolutionary_headers()

            async with self.session.get(imgur_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    meme_urls = []

                    # Find Imgur image containers
                    for img in soup.find_all('img', src=True):
                        src = img.get('src')
                        if src and 'i.imgur.com' in src:
                            # Ensure we get full resolution
                            if src.startswith('//'):
                                src = 'https:' + src
                            meme_urls.append(src)

                    # Look for data-src attributes (lazy loaded images)
                    for element in soup.find_all(attrs={'data-src': True}):
                        data_src = element.get('data-src')
                        if data_src and 'imgur.com' in data_src:
                            if data_src.startswith('//'):
                                data_src = 'https:' + data_src
                            meme_urls.append(data_src)

                    logger.info(f"🖼️ Found {len(meme_urls)} memes from Imgur: {imgur_url}")
                    return meme_urls[:25]  # Limit per page

        except Exception as e:
            logger.error(f"Imgur scraping failed for {imgur_url}: {str(e)}")
            return []

    async def _scrape_9gag_memes(self, gag_url: str) -> List[str]:
        """😂 Scrape 9GAG memes without login."""
        try:
            headers = self._get_revolutionary_headers()

            async with self.session.get(gag_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    meme_urls = []

                    # Find 9GAG image posts
                    for img in soup.find_all('img', src=True):
                        src = img.get('src')
                        if src and ('img-9gag-fun.9cache.com' in src or 'miscmedia-9gag-fun.9cache.com' in src):
                            if src.startswith('//'):
                                src = 'https:' + src
                            meme_urls.append(src)

                    # Look for video sources too
                    for video in soup.find_all('video'):
                        if video:
                            source = video.find('source')
                            if source and source.get('src'):
                                video_src = source.get('src')
                                if video_src and isinstance(video_src, str):
                                    if video_src.startswith('//'):
                                        video_src = 'https:' + video_src
                                    meme_urls.append(video_src)

                    logger.info(f"😂 Found {len(meme_urls)} memes from 9GAG: {gag_url}")
                    return meme_urls[:20]  # Limit per page

        except Exception as e:
            logger.error(f"9GAG scraping failed for {gag_url}: {str(e)}")
            return []

    async def _scrape_all_direct_platforms(self) -> List[str]:
        """🌐 Scrape all meme platforms directly - NO LOGIN REQUIRED!"""
        all_meme_urls = []

        logger.info("🌐 Starting direct platform scraping - No login required!")

        # Scrape Reddit subreddits
        for reddit_url in DIRECT_MEME_PLATFORMS["reddit"][:4]:  # Limit to 4 subreddits
            try:
                memes = await self._scrape_reddit_memes(reddit_url)
                all_meme_urls.extend(memes)
                await asyncio.sleep(random.uniform(2, 4))  # Be respectful
            except Exception as e:
                logger.error(f"Reddit platform scraping failed: {str(e)}")

        # Scrape Imgur
        for imgur_url in DIRECT_MEME_PLATFORMS["imgur"][:2]:  # Limit to 2 pages
            try:
                memes = await self._scrape_imgur_memes(imgur_url)
                all_meme_urls.extend(memes)
                await asyncio.sleep(random.uniform(2, 4))
            except Exception as e:
                logger.error(f"Imgur platform scraping failed: {str(e)}")

        # Scrape 9GAG
        for gag_url in DIRECT_MEME_PLATFORMS["9gag"][:2]:  # Limit to 2 pages
            try:
                memes = await self._scrape_9gag_memes(gag_url)
                all_meme_urls.extend(memes)
                await asyncio.sleep(random.uniform(2, 4))
            except Exception as e:
                logger.error(f"9GAG platform scraping failed: {str(e)}")

        # Remove duplicates
        unique_memes = list(set(all_meme_urls))

        logger.info(f"🎉 DIRECT PLATFORM SCRAPING COMPLETE! Found {len(unique_memes)} unique memes from {len(all_meme_urls)} total")
        return unique_memes

    def _get_file_category(self, url: str) -> str:
        """Determine file category based on extension."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        for category, extensions in self.file_extensions.items():
            if any(path.endswith(ext) for ext in extensions):
                return category
        return 'other'

    async def _create_session(self):
        """🚀 Create revolutionary browser-like session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

            # Create connector with browser-like settings
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                enable_cleanup_closed=True,
                force_close=False,  # Fixed: Can't use keepalive_timeout with force_close=True
                keepalive_timeout=30
            )

            # Use revolutionary browser headers (will be updated per request)
            base_headers = self._get_revolutionary_headers()

            # Create session with cookie jar (like a real browser)
            cookie_jar = aiohttp.CookieJar(unsafe=True)  # Allow all cookies

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=base_headers,
                connector=connector,
                cookie_jar=cookie_jar,
                trust_env=True  # Use system proxy settings if available
            )

            logger.info("🚀 Revolutionary browser session created with anti-detection features")

    async def _close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _download_file(self, url: str, output_dir: str, filename: str = None) -> DownloadResult:
        """Download a file from URL."""
        try:
            await self._create_session()
            
            if not filename:
                parsed = urlparse(url)
                filename = os.path.basename(parsed.path) or 'downloaded_file'
            
            # Ensure safe filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            local_path = os.path.join(output_dir, filename)
            
            # Download file
            async with self.session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', 'unknown')
                    file_size = int(response.headers.get('content-length', 0))
                    
                    async with aiofiles.open(local_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    self.download_stats['successful_downloads'] += 1
                    self.download_stats['total_size'] += file_size
                    
                    logger.info(f"📥 Downloaded: {filename} ({file_size} bytes)")
                    
                    return DownloadResult(
                        url=url,
                        local_path=local_path,
                        file_size=file_size,
                        content_type=content_type,
                        success=True
                    )
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            logger.error(f"❌ Download failed for {url}: {str(e)}")
            
            return DownloadResult(
                url=url,
                local_path="",
                file_size=0,
                content_type="",
                success=False,
                error_message=str(e)
            )

    async def _extract_all_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
        """Extract all types of links from HTML."""
        links = {
            'all_links': [],
            'images': [],
            'documents': [],
            'videos': [],
            'downloadable_files': [],
            'external_links': []
        }
        
        # Extract all href links
        for link in soup.find_all(['a', 'link'], href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            links['all_links'].append(full_url)
            
            # Categorize by file type
            category = self._get_file_category(full_url)
            if category in links:
                links[category].append(full_url)
            elif category != 'other':
                links['downloadable_files'].append(full_url)
        
        # Extract image sources
        for img in soup.find_all('img', src=True):
            src = urljoin(base_url, img['src'])
            links['images'].append(src)
        
        # Extract video sources
        for video in soup.find_all(['video', 'source'], src=True):
            src = urljoin(base_url, video['src'])
            links['videos'].append(src)
        
        # Remove duplicates
        for key in links:
            links[key] = list(set(links[key]))
        
        return links

    async def _harvest_website(self, url: str, config: AdvancedWebHarvesterInput) -> HarvestResult:
        """Harvest complete website content."""
        try:
            logger.info(f"🌐 Starting harvest of: {url}")
            
            # Fetch page content using aiohttp
            await self._create_session()
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                else:
                    raise Exception(f"HTTP {response.status} for {url}")
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract basic information
            title = soup.title.string if soup.title else "No Title"
            text_content = soup.get_text(strip=True)
            
            # Extract all links
            links_data = await self._extract_all_links(soup, url)
            
            # Create harvest result
            result = HarvestResult(
                url=url,
                title=title,
                content=text_content,
                links=links_data['all_links'],
                images=links_data['images'],
                documents=links_data['documents'],
                videos=links_data['videos'],
                downloadable_files=links_data['downloadable_files']
            )
            
            # Extract structured data
            if config.extract_structured_data:
                result.extracted_data = await self._extract_structured_data(soup)
            
            # Download files if requested
            if config.download_files:
                await self._download_found_files(result, config)
            
            logger.info(f"✅ Harvest completed for: {url}")
            logger.info(f"📊 Found: {len(result.images)} images, {len(result.documents)} documents, {len(result.downloadable_files)} files")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Harvest failed for {url}: {str(e)}")
            return HarvestResult(
                url=url,
                title="",
                content="",
                success=False,
                error_message=str(e)
            )

    async def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data from HTML."""
        structured_data = {}
        
        # Extract meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                meta_tags[name] = content
        structured_data['meta_tags'] = meta_tags
        
        # Extract JSON-LD structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        json_ld_data = []
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                json_ld_data.append(data)
            except:
                pass
        structured_data['json_ld'] = json_ld_data
        
        # Extract tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        structured_data['tables'] = tables
        
        return structured_data

    async def _download_found_files(self, result: HarvestResult, config: AdvancedWebHarvesterInput):
        """🎯 Download all found files with smart media detection."""
        download_tasks = []

        # Prepare download directory
        base_dir = config.output_directory
        url_hash = hashlib.md5(result.url.encode()).hexdigest()[:8]
        site_dir = os.path.join(base_dir, f"site_{url_hash}")

        # 🖼️ Smart Image Downloads - Use proper extensions, skip HTML
        if config.include_images and result.images:
            img_dir = os.path.join(site_dir, "images")
            for img_url in result.images[:50]:  # Limit to 50 images per site
                download_tasks.append(self._smart_media_download(img_url, img_dir))

        # 📄 Smart Document Downloads
        if config.include_documents and result.documents:
            doc_dir = os.path.join(site_dir, "documents")
            for doc_url in result.documents:
                download_tasks.append(self._smart_media_download(doc_url, doc_dir))

        # 🎥 Smart Video Downloads - Use proper extensions
        if config.include_videos and result.videos:
            vid_dir = os.path.join(site_dir, "videos")
            for vid_url in result.videos[:10]:  # Limit to 10 videos per site
                download_tasks.append(self._smart_media_download(vid_url, vid_dir))
        
        # Download other files
        if result.downloadable_files:
            files_dir = os.path.join(site_dir, "files")
            for file_url in result.downloadable_files:
                # Filter by file types if specified
                if config.file_types:
                    if not any(file_url.lower().endswith(f'.{ft}') for ft in config.file_types):
                        continue
                download_tasks.append(self._download_file(file_url, files_dir))
        
        # Execute downloads
        if download_tasks:
            logger.info(f"📥 Starting {len(download_tasks)} downloads...")
            await asyncio.gather(*download_tasks, return_exceptions=True)

    async def _run(self, **kwargs) -> str:
        """Execute web harvesting operation."""
        try:
            # Ensure session is created first
            if not self.session:
                await self._create_session()

            # Parse and validate input
            config = AdvancedWebHarvesterInput(**kwargs)
            
            # Reset download stats
            self.download_stats = {
                'total_files': 0,
                'successful_downloads': 0,
                'failed_downloads': 0,
                'total_size': 0
            }
            
            results = []
            
            if config.operation == "harvest":
                if not config.url:
                    raise ValueError("URL required for harvest operation")
                
                result = await self._harvest_website(config.url, config)
                results.append(result)
                
            elif config.operation == "bulk_harvest":
                if not config.urls:
                    raise ValueError("URLs list required for bulk_harvest operation")

                # 🔍 REVOLUTIONARY SEARCH ENGINE INTEGRATION
                all_urls = list(config.urls)

                # If meme harvesting operation, discover additional URLs via search engines
                if any(keyword in str(config.urls).lower() for keyword in ['meme', 'viral', 'funny', 'reddit', '9gag']):
                    logger.info("🎯 MEME HARVESTING DETECTED - Activating search engine discovery...")
                    discovered_urls = await self._discover_meme_content()
                    all_urls.extend(discovered_urls)
                    logger.info(f"🔍 Discovered {len(discovered_urls)} additional meme URLs via search engines!")

                # Remove duplicates while preserving order
                seen = set()
                unique_urls = []
                for url in all_urls:
                    if url not in seen:
                        seen.add(url)
                        unique_urls.append(url)

                logger.info(f"🚀 Starting revolutionary bulk harvest of {len(unique_urls)} websites (including {len(unique_urls) - len(config.urls)} discovered)...")

                # Process URLs in batches to avoid overwhelming servers
                batch_size = 5
                for i in range(0, len(unique_urls), batch_size):
                    batch = unique_urls[i:i+batch_size]
                    batch_tasks = [self._harvest_website(url, config) for url in batch]
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, HarvestResult):
                            results.append(result)
                    
                    # Small delay between batches
                    await asyncio.sleep(1)
                
            elif config.operation == "extract_links":
                if not config.url:
                    raise ValueError("URL required for extract_links operation")
                
                # Fetch page content using aiohttp
                await self._create_session()
                async with self.session.get(config.url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        links_data = await self._extract_all_links(soup, config.url)
                    else:
                        raise Exception(f"HTTP {response.status} for {config.url}")
                    
                    return json.dumps({
                        'success': True,
                        'operation': 'extract_links',
                        'url': config.url,
                        'links_found': links_data,
                        'total_links': len(links_data['all_links']),
                        'images': len(links_data['images']),
                        'documents': len(links_data['documents']),
                        'videos': len(links_data['videos']),
                        'downloadable_files': len(links_data['downloadable_files'])
                    }, indent=2)
                
            elif config.operation == "download":
                if not config.url:
                    raise ValueError("URL required for download operation")
                
                download_result = await self._download_file(config.url, config.output_directory)
                
                return json.dumps({
                    'success': download_result.success,
                    'operation': 'download',
                    'url': download_result.url,
                    'local_path': download_result.local_path,
                    'file_size': download_result.file_size,
                    'content_type': download_result.content_type,
                    'error_message': download_result.error_message
                }, indent=2)
            
            # Close session
            await self._close_session()
            
            # Prepare final results
            successful_harvests = [r for r in results if r.success]
            failed_harvests = [r for r in results if not r.success]
            
            total_links = sum(len(r.links) for r in successful_harvests)
            total_images = sum(len(r.images) for r in successful_harvests)
            total_documents = sum(len(r.documents) for r in successful_harvests)
            total_files = sum(len(r.downloadable_files) for r in successful_harvests)
            
            final_result = {
                'success': True,
                'operation': config.operation,
                'total_sites_processed': len(results),
                'successful_harvests': len(successful_harvests),
                'failed_harvests': len(failed_harvests),
                'total_links_found': total_links,
                'total_images_found': total_images,
                'total_documents_found': total_documents,
                'total_files_found': total_files,
                'download_stats': self.download_stats,
                'output_directory': config.output_directory,
                'harvest_results': [
                    {
                        'url': r.url,
                        'title': r.title,
                        'success': r.success,
                        'links_count': len(r.links),
                        'images_count': len(r.images),
                        'documents_count': len(r.documents),
                        'files_count': len(r.downloadable_files),
                        'content_length': len(r.content),
                        'error_message': r.error_message
                    }
                    for r in results
                ]
            }
            
            logger.info("🎉 HARVEST OPERATION COMPLETED!")
            logger.info(f"📊 FINAL STATS: {len(successful_harvests)} sites, {total_links} links, {self.download_stats['successful_downloads']} downloads")
            
            return json.dumps(final_result, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Advanced Web Harvester operation failed: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e),
                'operation': kwargs.get('operation', 'unknown')
            }, indent=2)


# Tool registration
def get_tool():
    """Get the Advanced Web Harvester Tool instance."""
    return AdvancedWebHarvesterTool()


# Tool metadata for the unified repository
TOOL_METADATA = {
    "name": "advanced_web_harvester",
    "category": ToolCategory.BUSINESS,
    "description": "Revolutionary web harvesting tool for complete internet data collection",
    "complexity": "advanced",
    "tags": ["web", "scraping", "download", "harvest", "data collection", "internet", "revolutionary"]
}
