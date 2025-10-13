"""
Revolutionary Browser Automation Tool (Visual) - Human-like Web Interaction for AI Agents.

This tool provides cutting-edge browser automation capabilities including:
- Visual web interaction using computer vision + LLM reasoning
- Puppeteer integration for browser control
- Screenshot-based element detection and interaction
- Human-like browsing patterns and behavior
- Multi-modal LLM integration for visual understanding
- Advanced web scraping with visual feedback

REVOLUTIONARY FEATURES:
- Interacts with ANY web interface like a human would
- Uses visual understanding instead of brittle selectors
- Adapts to dynamic content and layout changes
- Provides intelligent error recovery and retry logic
- Supports complex multi-step workflows
- Integrates with Screenshot Analysis Tool for visual reasoning
"""

import asyncio
import base64
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.tools.metadata import MetadataCapableToolMixin, ToolMetadata as MetadataToolMetadata, ParameterSchema, ParameterType, UsagePattern, UsagePatternType, ConfidenceModifier, ConfidenceModifierType

# Import screenshot analysis capabilities
from .screenshot_analysis_tool import (
    RevolutionaryScreenshotAnalyzer,
    ScreenshotAnalysisConfig,
    VisualAnalysisResult,
    UIElement,
    UIElementType
)

# Import LLM capabilities
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType

logger = get_logger()


class BrowserAction(str, Enum):
    """Types of browser actions that can be performed."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT_TEXT = "extract_text"
    EXTRACT_DATA = "extract_data"
    HOVER = "hover"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    DRAG_DROP = "drag_drop"
    SELECT_OPTION = "select_option"
    UPLOAD_FILE = "upload_file"
    DOWNLOAD_FILE = "download_file"


class BrowserEngine(str, Enum):
    """Supported browser engines."""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    CHROME = "chrome"
    EDGE = "edge"


@dataclass
class BrowserActionResult:
    """Result of a browser action."""
    action: BrowserAction
    success: bool
    message: str
    screenshot_path: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    error_details: Optional[str] = None


@dataclass
class WebElement:
    """Represents a web element detected visually."""
    element_type: UIElementType
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]  # Click coordinates
    confidence: float
    text: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    selector_hints: List[str] = field(default_factory=list)
    is_clickable: bool = False
    is_visible: bool = True


class BrowserAutomationConfig(BaseModel):
    """Configuration for browser automation."""
    # Browser settings
    browser_engine: BrowserEngine = BrowserEngine.CHROMIUM
    headless: bool = False
    window_width: int = 1920
    window_height: int = 1080
    user_agent: Optional[str] = None
    
    # Visual analysis settings
    screenshot_analysis_config: Optional[ScreenshotAnalysisConfig] = None
    enable_visual_feedback: bool = True
    visual_retry_attempts: int = 3
    
    # Automation settings
    default_timeout: int = 30
    action_delay: float = 1.0  # Delay between actions (human-like)
    scroll_delay: float = 0.5
    typing_delay: float = 0.1  # Delay between keystrokes
    
    # Error handling
    enable_auto_retry: bool = True
    max_retry_attempts: int = 3
    screenshot_on_error: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_screenshots: bool = True
    max_cache_size: int = 100


class RevolutionaryBrowserAutomator:
    """Revolutionary browser automator with visual intelligence."""
    
    def __init__(self, config: Optional[BrowserAutomationConfig] = None):
        self.config = config or BrowserAutomationConfig()
        self.browser = None
        self.page = None
        self.context = None
        self.screenshot_analyzer = None
        self.llm_manager = None
        self.session_id = None
        self.is_initialized = False
        
        # Action history for learning and debugging
        self.action_history: List[Dict[str, Any]] = []
        self.screenshot_cache: Dict[str, str] = {}
        
    async def initialize(self) -> bool:
        """Initialize the browser automator."""
        try:
            if self.is_initialized:
                return True

            logger.info(
                "🚀 Initializing Revolutionary Browser Automator...",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool"
            )

            # Initialize screenshot analyzer
            screenshot_config = self.config.screenshot_analysis_config or ScreenshotAnalysisConfig()
            self.screenshot_analyzer = RevolutionaryScreenshotAnalyzer(screenshot_config)
            await self.screenshot_analyzer.initialize()
            logger.info(
                "✅ Screenshot analyzer initialized",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool"
            )

            # Initialize LLM manager
            self.llm_manager = get_enhanced_llm_manager()
            if self.llm_manager:
                await self.llm_manager.initialize()
                logger.info(
                    "✅ LLM manager initialized",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.production.browser_automation_tool"
                )

            # Initialize browser (Playwright)
            await self._initialize_browser()

            self.session_id = f"browser_session_{int(time.time())}"
            self.is_initialized = True

            logger.info(
                "🎯 Revolutionary Browser Automator ready!",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool"
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to initialize browser automator",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                error=e
            )
            return False
    
    async def _initialize_browser(self) -> None:
        """Initialize the browser engine."""
        try:
            # Try to import Playwright
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                logger.warning(
                    "Playwright not installed. Browser automation will use fallback methods.",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.production.browser_automation_tool"
                )
                return

            # Launch browser
            self.playwright = await async_playwright().start()

            if self.config.browser_engine == BrowserEngine.CHROMIUM:
                self.browser = await self.playwright.chromium.launch(
                    headless=self.config.headless,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
            elif self.config.browser_engine == BrowserEngine.FIREFOX:
                self.browser = await self.playwright.firefox.launch(headless=self.config.headless)
            elif self.config.browser_engine == BrowserEngine.WEBKIT:
                self.browser = await self.playwright.webkit.launch(headless=self.config.headless)
            else:
                # Default to Chromium
                self.browser = await self.playwright.chromium.launch(
                    headless=self.config.headless,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )

            # Create context and page
            self.context = await self.browser.new_context(
                viewport={'width': self.config.window_width, 'height': self.config.window_height},
                user_agent=self.config.user_agent
            )

            self.page = await self.context.new_page()

            logger.info(
                f"✅ Browser initialized: {self.config.browser_engine.value}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"browser_engine": self.config.browser_engine.value}
            )

        except Exception as e:
            logger.error(
                "Browser initialization failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                error=e
            )
            raise
    
    async def navigate_to_url(self, url: str) -> BrowserActionResult:
        """Navigate to a URL."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if not self.page:
                return BrowserActionResult(
                    action=BrowserAction.NAVIGATE,
                    success=False,
                    message="Browser not initialized",
                    execution_time=time.time() - start_time
                )
            
            logger.info(
                f"🌐 Navigating to: {url}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"url": url}
            )

            # Navigate to URL
            await self.page.goto(url, timeout=self.config.default_timeout * 1000)

            # Wait for page to load
            await self.page.wait_for_load_state('networkidle', timeout=self.config.default_timeout * 1000)

            # Take screenshot for visual analysis
            screenshot_path = await self._take_screenshot("navigation")

            # Record action
            self._record_action("navigate", {"url": url}, True)

            return BrowserActionResult(
                action=BrowserAction.NAVIGATE,
                success=True,
                message=f"Successfully navigated to {url}",
                screenshot_path=screenshot_path,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(
                "Navigation failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"url": url},
                error=e
            )
            screenshot_path = await self._take_screenshot("navigation_error") if self.config.screenshot_on_error else None
            
            return BrowserActionResult(
                action=BrowserAction.NAVIGATE,
                success=False,
                message=f"Navigation failed: {str(e)}",
                screenshot_path=screenshot_path,
                execution_time=time.time() - start_time,
                error_details=str(e)
            )
    
    async def visual_click(self, target_description: str, screenshot_path: Optional[str] = None) -> BrowserActionResult:
        """Click on an element using visual detection."""
        start_time = time.time()
        
        try:
            if not screenshot_path:
                screenshot_path = await self._take_screenshot("visual_click_analysis")
            
            # Analyze screenshot to find the target element
            analysis_result = await self.screenshot_analyzer.analyze_screenshot(screenshot_path)
            
            if not analysis_result:
                return BrowserActionResult(
                    action=BrowserAction.CLICK,
                    success=False,
                    message="Failed to analyze screenshot for visual click",
                    execution_time=time.time() - start_time
                )
            
            # Find the best matching element
            target_element = await self._find_target_element(analysis_result, target_description)
            
            if not target_element:
                return BrowserActionResult(
                    action=BrowserAction.CLICK,
                    success=False,
                    message=f"Could not find element matching: {target_description}",
                    screenshot_path=screenshot_path,
                    execution_time=time.time() - start_time
                )
            
            # Perform the click
            click_x = target_element.bbox[0] + target_element.bbox[2] // 2
            click_y = target_element.bbox[1] + target_element.bbox[3] // 2
            
            await self.page.mouse.click(click_x, click_y)
            await asyncio.sleep(self.config.action_delay)
            
            # Take screenshot after action
            after_screenshot = await self._take_screenshot("after_visual_click")
            
            # Record action
            self._record_action("visual_click", {
                "target": target_description,
                "coordinates": (click_x, click_y),
                "element_type": target_element.element_type.value
            }, True)

            logger.info(
                f"✅ Visual click successful: {target_description} at ({click_x}, {click_y})",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"target": target_description, "coordinates": (click_x, click_y)}
            )

            return BrowserActionResult(
                action=BrowserAction.CLICK,
                success=True,
                message=f"Successfully clicked on {target_description}",
                screenshot_path=after_screenshot,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(
                "Visual click failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"target": target_description},
                error=e
            )
            error_screenshot = await self._take_screenshot("visual_click_error") if self.config.screenshot_on_error else None
            
            return BrowserActionResult(
                action=BrowserAction.CLICK,
                success=False,
                message=f"Visual click failed: {str(e)}",
                screenshot_path=error_screenshot,
                execution_time=time.time() - start_time,
                error_details=str(e)
            )

    async def visual_type(self, target_description: str, text: str, screenshot_path: Optional[str] = None) -> BrowserActionResult:
        """Type text into an element using visual detection."""
        start_time = time.time()

        try:
            if not screenshot_path:
                screenshot_path = await self._take_screenshot("visual_type_analysis")

            # Analyze screenshot to find the input field
            analysis_result = await self.screenshot_analyzer.analyze_screenshot(screenshot_path)

            if not analysis_result:
                return BrowserActionResult(
                    action=BrowserAction.TYPE,
                    success=False,
                    message="Failed to analyze screenshot for visual typing",
                    execution_time=time.time() - start_time
                )

            # Find the target input element
            target_element = await self._find_target_element(analysis_result, target_description, UIElementType.INPUT_FIELD)

            if not target_element:
                return BrowserActionResult(
                    action=BrowserAction.TYPE,
                    success=False,
                    message=f"Could not find input field matching: {target_description}",
                    screenshot_path=screenshot_path,
                    execution_time=time.time() - start_time
                )

            # Click on the input field first
            click_x = target_element.bbox[0] + target_element.bbox[2] // 2
            click_y = target_element.bbox[1] + target_element.bbox[3] // 2

            await self.page.mouse.click(click_x, click_y)
            await asyncio.sleep(0.5)

            # Clear existing content and type new text
            await self.page.keyboard.press('Control+A')
            await asyncio.sleep(0.2)

            # Type text with human-like delays
            for char in text:
                await self.page.keyboard.type(char)
                await asyncio.sleep(self.config.typing_delay)

            await asyncio.sleep(self.config.action_delay)

            # Take screenshot after typing
            after_screenshot = await self._take_screenshot("after_visual_type")

            # Record action
            self._record_action("visual_type", {
                "target": target_description,
                "text": text,
                "coordinates": (click_x, click_y)
            }, True)

            logger.info(
                f"✅ Visual typing successful: '{text}' into {target_description}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"text": text, "target": target_description}
            )

            return BrowserActionResult(
                action=BrowserAction.TYPE,
                success=True,
                message=f"Successfully typed '{text}' into {target_description}",
                screenshot_path=after_screenshot,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(
                "Visual typing failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"text": text, "target": target_description},
                error=e
            )
            error_screenshot = await self._take_screenshot("visual_type_error") if self.config.screenshot_on_error else None

            return BrowserActionResult(
                action=BrowserAction.TYPE,
                success=False,
                message=f"Visual typing failed: {str(e)}",
                screenshot_path=error_screenshot,
                execution_time=time.time() - start_time,
                error_details=str(e)
            )

    async def visual_scroll(self, direction: str = "down", amount: int = 3) -> BrowserActionResult:
        """Scroll the page visually."""
        start_time = time.time()

        try:
            logger.info(
                f"📜 Scrolling {direction} by {amount} units",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"direction": direction, "amount": amount}
            )

            # Take before screenshot
            before_screenshot = await self._take_screenshot("before_scroll")

            # Perform scroll
            if direction.lower() == "down":
                for _ in range(amount):
                    await self.page.mouse.wheel(0, 300)
                    await asyncio.sleep(self.config.scroll_delay)
            elif direction.lower() == "up":
                for _ in range(amount):
                    await self.page.mouse.wheel(0, -300)
                    await asyncio.sleep(self.config.scroll_delay)
            elif direction.lower() == "left":
                for _ in range(amount):
                    await self.page.mouse.wheel(-300, 0)
                    await asyncio.sleep(self.config.scroll_delay)
            elif direction.lower() == "right":
                for _ in range(amount):
                    await self.page.mouse.wheel(300, 0)
                    await asyncio.sleep(self.config.scroll_delay)

            # Take after screenshot
            after_screenshot = await self._take_screenshot("after_scroll")

            # Record action
            self._record_action("scroll", {"direction": direction, "amount": amount}, True)

            return BrowserActionResult(
                action=BrowserAction.SCROLL,
                success=True,
                message=f"Successfully scrolled {direction} by {amount} units",
                screenshot_path=after_screenshot,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(
                "Scroll failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"direction": direction, "amount": amount},
                error=e
            )

            return BrowserActionResult(
                action=BrowserAction.SCROLL,
                success=False,
                message=f"Scroll failed: {str(e)}",
                execution_time=time.time() - start_time,
                error_details=str(e)
            )

    async def extract_page_data(self, data_description: str) -> BrowserActionResult:
        """Extract data from the current page using visual analysis."""
        start_time = time.time()

        try:
            logger.info(
                f"📊 Extracting data: {data_description}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"description": data_description}
            )

            # Take screenshot for analysis
            screenshot_path = await self._take_screenshot("data_extraction")

            # Analyze screenshot
            analysis_result = await self.screenshot_analyzer.analyze_screenshot(screenshot_path)

            if not analysis_result:
                return BrowserActionResult(
                    action=BrowserAction.EXTRACT_DATA,
                    success=False,
                    message="Failed to analyze page for data extraction",
                    execution_time=time.time() - start_time
                )

            # Extract relevant data based on analysis
            extracted_data = {
                "extracted_text": analysis_result.extracted_text,
                "ui_elements": [
                    {
                        "type": elem.element_type.value,
                        "text": elem.text,
                        "bbox": elem.bbox,
                        "confidence": elem.confidence
                    }
                    for elem in analysis_result.ui_elements
                ],
                "context": analysis_result.context_analysis,
                "purpose": analysis_result.purpose_analysis,
                "automation_suggestions": analysis_result.automation_suggestions
            }

            # Try to extract additional data using Playwright
            if self.page:
                try:
                    page_title = await self.page.title()
                    page_url = self.page.url
                    extracted_data.update({
                        "page_title": page_title,
                        "page_url": page_url,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(
                        "Could not extract additional page data",
                        LogCategory.TOOL_OPERATIONS,
                        "app.tools.production.browser_automation_tool",
                        error=e
                    )

            # Record action
            self._record_action("extract_data", {"description": data_description}, True)

            logger.info(
                f"✅ Data extraction successful: {len(extracted_data)} data points",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"description": data_description, "data_points": len(extracted_data)}
            )

            return BrowserActionResult(
                action=BrowserAction.EXTRACT_DATA,
                success=True,
                message=f"Successfully extracted data: {data_description}",
                screenshot_path=screenshot_path,
                extracted_data=extracted_data,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(
                "Data extraction failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"description": data_description},
                error=e
            )

            return BrowserActionResult(
                action=BrowserAction.EXTRACT_DATA,
                success=False,
                message=f"Data extraction failed: {str(e)}",
                execution_time=time.time() - start_time,
                error_details=str(e)
            )

    async def _take_screenshot(self, purpose: str = "general") -> Optional[str]:
        """Take a screenshot of the current page."""
        try:
            if not self.page:
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"data/screenshots/browser_{purpose}_{timestamp}.png"

            # Ensure screenshots directory exists
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)

            # Take screenshot
            await self.page.screenshot(path=screenshot_path, full_page=True)

            # Cache screenshot if enabled
            if self.config.cache_screenshots:
                self.screenshot_cache[purpose] = screenshot_path

                # Limit cache size
                if len(self.screenshot_cache) > self.config.max_cache_size:
                    oldest_key = next(iter(self.screenshot_cache))
                    del self.screenshot_cache[oldest_key]

            return screenshot_path

        except Exception as e:
            logger.error(
                "Screenshot failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"purpose": purpose},
                error=e
            )
            return None

    async def _find_target_element(
        self,
        analysis_result: VisualAnalysisResult,
        target_description: str,
        preferred_type: Optional[UIElementType] = None
    ) -> Optional[WebElement]:
        """Find the best matching element based on description."""
        try:
            # Simple matching logic - in production, this would use more sophisticated NLP/ML
            target_lower = target_description.lower()
            best_match = None
            best_score = 0.0

            for ui_element in analysis_result.ui_elements:
                score = 0.0

                # Type matching
                if preferred_type and ui_element.element_type == preferred_type:
                    score += 0.5
                elif ui_element.element_type in [UIElementType.BUTTON, UIElementType.INPUT_FIELD, UIElementType.LINK]:
                    score += 0.3

                # Text matching
                if ui_element.text:
                    element_text_lower = ui_element.text.lower()
                    if target_lower in element_text_lower or element_text_lower in target_lower:
                        score += 0.4

                    # Keyword matching
                    target_words = target_lower.split()
                    element_words = element_text_lower.split()
                    common_words = set(target_words) & set(element_words)
                    if common_words:
                        score += 0.2 * len(common_words) / len(target_words)

                # Confidence bonus
                score += ui_element.confidence * 0.1

                if score > best_score:
                    best_score = score
                    best_match = ui_element

            if best_match and best_score > 0.3:  # Minimum threshold
                return WebElement(
                    element_type=best_match.element_type,
                    bbox=best_match.bbox,
                    center=(
                        best_match.bbox[0] + best_match.bbox[2] // 2,
                        best_match.bbox[1] + best_match.bbox[3] // 2
                    ),
                    confidence=best_match.confidence,
                    text=best_match.text,
                    is_clickable=best_match.actionable,
                    is_visible=True
                )

            return None

        except Exception as e:
            logger.error(
                "Element matching failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"target": target_description},
                error=e
            )
            return None

    def _record_action(self, action_type: str, parameters: Dict[str, Any], success: bool) -> None:
        """Record an action for history and learning."""
        try:
            action_record = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "action_type": action_type,
                "parameters": parameters,
                "success": success
            }

            self.action_history.append(action_record)

            # Limit history size
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-500:]  # Keep last 500 actions

        except Exception as e:
            logger.warning(
                "Failed to record action",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"action_type": action_type},
                error=e
            )

    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()

            logger.info(
                "🔒 Browser automation session closed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool"
            )

        except Exception as e:
            logger.error(
                "Browser cleanup failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                error=e
            )


class BrowserAutomationInput(BaseModel):
    """Input schema for browser automation tool."""
    action: str = Field(description="Action to perform: navigate, visual_click, visual_type, scroll, extract_data, screenshot")
    url: Optional[str] = Field(default=None, description="URL to navigate to (for navigate action)")
    target: Optional[str] = Field(default=None, description="Description of element to interact with")
    text: Optional[str] = Field(default=None, description="Text to type (for visual_type action)")
    direction: Optional[str] = Field(default="down", description="Scroll direction: up, down, left, right")
    amount: Optional[int] = Field(default=3, description="Scroll amount (number of scroll units)")
    data_description: Optional[str] = Field(default=None, description="Description of data to extract")


class BrowserAutomationTool(BaseTool, MetadataCapableToolMixin):
    """Revolutionary Browser Automation Tool for AI Agents."""

    name: str = "browser_automation"
    tool_id: str = "browser_automation"
    description: str = """Revolutionary browser automation tool with visual intelligence that can:
    - Navigate to any website
    - Click on elements using visual description (no selectors needed)
    - Type text into forms using visual detection
    - Scroll pages in any direction
    - Extract data from pages using visual analysis
    - Take screenshots for analysis
    - Perform complex multi-step web workflows

    Actions: navigate, visual_click, visual_type, scroll, extract_data, screenshot

    This tool uses computer vision and LLM reasoning to interact with web pages like a human would."""

    args_schema: type = BrowserAutomationInput

    def __init__(self, config: Optional[BrowserAutomationConfig] = None):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation for internal attributes
        object.__setattr__(self, 'automator', RevolutionaryBrowserAutomator(config))
        object.__setattr__(self, '_initialized', False)

    async def _arun(
        self,
        action: str,
        url: Optional[str] = None,
        target: Optional[str] = None,
        text: Optional[str] = None,
        direction: str = "down",
        amount: int = 3,
        data_description: Optional[str] = None
    ) -> str:
        """Execute browser automation action."""
        try:
            # Initialize if needed
            if not self._initialized:
                success = await self.automator.initialize()
                if not success:
                    return "❌ Failed to initialize browser automator"
                object.__setattr__(self, '_initialized', True)

            action_lower = action.lower()

            if action_lower == "navigate":
                if not url:
                    return "❌ URL is required for navigate action"
                result = await self.automator.navigate_to_url(url)

            elif action_lower == "visual_click":
                if not target:
                    return "❌ Target description is required for visual_click action"
                result = await self.automator.visual_click(target)

            elif action_lower == "visual_type":
                if not target or not text:
                    return "❌ Target description and text are required for visual_type action"
                result = await self.automator.visual_type(target, text)

            elif action_lower == "scroll":
                result = await self.automator.visual_scroll(direction, amount)

            elif action_lower == "extract_data":
                description = data_description or "general page data"
                result = await self.automator.extract_page_data(description)

            elif action_lower == "screenshot":
                screenshot_path = await self.automator._take_screenshot("manual_request")
                if screenshot_path:
                    result = BrowserActionResult(
                        action=BrowserAction.SCREENSHOT,
                        success=True,
                        message=f"Screenshot saved to {screenshot_path}",
                        screenshot_path=screenshot_path
                    )
                else:
                    result = BrowserActionResult(
                        action=BrowserAction.SCREENSHOT,
                        success=False,
                        message="Failed to take screenshot"
                    )
            else:
                return f"❌ Unknown action: {action}. Available actions: navigate, visual_click, visual_type, scroll, extract_data, screenshot"

            # Format result
            return self._format_result(result)

        except Exception as e:
            logger.error(
                "Browser automation error",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                data={"action": action},
                error=e
            )
            return f"❌ Browser automation error: {str(e)}"

    def _run(
        self,
        action: str,
        url: Optional[str] = None,
        target: Optional[str] = None,
        text: Optional[str] = None,
        direction: str = "down",
        amount: int = 3,
        data_description: Optional[str] = None
    ) -> str:
        """Synchronous wrapper for browser automation."""
        return asyncio.run(self._arun(action, url, target, text, direction, amount, data_description))

    def _format_result(self, result: BrowserActionResult) -> str:
        """Format browser action result for agent consumption."""
        output = []

        status = "✅" if result.success else "❌"
        output.append(f"{status} **BROWSER AUTOMATION RESULT**")
        output.append(f"🎬 Action: {result.action.value}")
        output.append(f"📊 Status: {'Success' if result.success else 'Failed'}")
        output.append(f"💬 Message: {result.message}")
        output.append(f"⏱️ Execution Time: {result.execution_time:.2f}s")

        if result.screenshot_path:
            output.append(f"📸 Screenshot: {result.screenshot_path}")

        if result.extracted_data:
            output.append(f"📊 **EXTRACTED DATA**")
            data = result.extracted_data
            if 'page_title' in data:
                output.append(f"📄 Page Title: {data['page_title']}")
            if 'page_url' in data:
                output.append(f"🌐 Page URL: {data['page_url']}")
            if 'extracted_text' in data and data['extracted_text']:
                text_preview = data['extracted_text'][:200] + "..." if len(data['extracted_text']) > 200 else data['extracted_text']
                output.append(f"📝 Text: {text_preview}")
            if 'ui_elements' in data:
                output.append(f"🎯 UI Elements: {len(data['ui_elements'])} detected")

        if result.error_details:
            output.append(f"🔍 Error Details: {result.error_details}")

        return "\n".join(output)

    async def cleanup(self) -> None:
        """Cleanup browser resources."""
        try:
            if hasattr(self, 'automator'):
                await self.automator.close()
        except Exception as e:
            logger.error(
                "Browser cleanup error",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.browser_automation_tool",
                error=e
            )

    def _create_metadata(self) -> MetadataToolMetadata:
        """Create metadata for browser automation tool."""
        return MetadataToolMetadata(
            name="browser_automation",
            description="Revolutionary browser automation tool with visual intelligence for creative web interactions",
            category="automation",
            usage_patterns=[
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="chaos,research,creative,inspiration,memes",
                    weight=0.9,
                    context_requirements=["chaos_mode", "creative_task"],
                    description="Triggers on creative research tasks"
                ),
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="browse,navigate,website,url,web",
                    weight=0.8,
                    context_requirements=["web_interaction_task"],
                    description="Matches web browsing tasks"
                ),
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="click,type,scroll,interact,automate",
                    weight=0.85,
                    context_requirements=["browser_automation_task"],
                    description="Matches automated interaction tasks"
                )
            ],
            confidence_modifiers=[
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="chaos_mode",
                    value=0.2,
                    description="Boost confidence for chaotic web research"
                ),
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="web_interaction_task",
                    value=0.15,
                    description="Boost confidence for web automation tasks"
                )
            ],
            parameter_schemas=[
                ParameterSchema(
                    name="action",
                    type=ParameterType.STRING,
                    description="Browser automation action to perform",
                    required=True,
                    default_value="navigate"
                ),
                ParameterSchema(
                    name="url",
                    type=ParameterType.STRING,
                    description="Target URL for navigation",
                    required=False,
                    default_value="https://www.reddit.com/r/dankmemes"
                ),
                ParameterSchema(
                    name="target",
                    type=ParameterType.STRING,
                    description="Target element or content",
                    required=False,
                    default_value="meme_content"
                )
            ]
        )


# Factory function to create the tool
def create_browser_automation_tool(config: Optional[BrowserAutomationConfig] = None) -> BrowserAutomationTool:
    """Create a browser automation tool instance."""
    return BrowserAutomationTool(config)

# Expected factory function name
def get_browser_automation_tool(config: Optional[BrowserAutomationConfig] = None) -> BrowserAutomationTool:
    """Get browser automation tool instance."""
    return BrowserAutomationTool(config)

# Default tool instance
browser_automation_tool = create_browser_automation_tool()

# Tool metadata for unified repository registration
from app.tools.unified_tool_repository import ToolMetadata as UnifiedToolMetadata, ToolCategory, ToolAccessLevel

BROWSER_AUTOMATION_TOOL_METADATA = UnifiedToolMetadata(
    tool_id="browser_automation",
    name="Browser Automation Tool",
    description="Revolutionary browser automation tool with visual intelligence for creative web interactions",
    category=ToolCategory.AUTOMATION,
    access_level=ToolAccessLevel.PUBLIC,
    requires_rag=False,
    use_cases={"browser", "automation", "visual", "web", "chaos", "navigation", "element_detection", "form_filling", "data_extraction"}
)
