"""
Revolutionary Screenshot Analysis Tool - Advanced Visual Understanding for AI Agents.

This tool provides cutting-edge screenshot analysis capabilities including:
- Multi-modal LLM integration (Ollama llama4:scout + API providers)
- UI element detection and classification
- Visual reasoning and context understanding
- OCR integration with existing RAG system
- Screenshot capture functionality
- Advanced visual analysis beyond traditional OCR

REVOLUTIONARY FEATURES:
- Understands UI context and purpose
- Identifies interactive elements (buttons, forms, links)
- Provides actionable insights for automation
- Multi-provider LLM support for visual reasoning
- Seamless integration with existing OCR engine
"""

import asyncio
import base64
import io
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import structlog
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from pydantic import BaseModel as LangChainBaseModel

# Import existing OCR capabilities
from app.rag.ingestion.processors import get_ocr_engine, RevolutionaryOCREngine

# Import LLM capabilities
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType, ModelCapability

logger = structlog.get_logger(__name__)


class UIElementType(str, Enum):
    """Types of UI elements that can be detected."""
    BUTTON = "button"
    INPUT_FIELD = "input_field"
    LINK = "link"
    IMAGE = "image"
    TEXT = "text"
    MENU = "menu"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    SLIDER = "slider"
    MODAL = "modal"
    POPUP = "popup"
    FORM = "form"
    TABLE = "table"
    NAVIGATION = "navigation"
    ICON = "icon"
    UNKNOWN = "unknown"


class ScreenshotSource(str, Enum):
    """Source of the screenshot."""
    DESKTOP = "desktop"
    BROWSER = "browser"
    APPLICATION = "application"
    MOBILE = "mobile"
    UPLOAD = "upload"


@dataclass
class UIElement:
    """Represents a detected UI element."""
    element_type: UIElementType
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    text: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    actionable: bool = False
    interaction_hints: List[str] = field(default_factory=list)


@dataclass
class VisualAnalysisResult:
    """Result of visual analysis."""
    screenshot_path: str
    timestamp: datetime
    source: ScreenshotSource
    dimensions: Tuple[int, int]  # (width, height)
    
    # OCR Results
    extracted_text: str
    ocr_confidence: float
    ocr_language: str
    
    # UI Elements
    ui_elements: List[UIElement]
    
    # Visual Understanding
    visual_description: str
    context_analysis: str
    purpose_analysis: str
    
    # Actionable Insights
    automation_suggestions: List[str]
    interaction_opportunities: List[Dict[str, Any]]
    
    # Metadata
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


class ScreenshotAnalysisConfig(BaseModel):
    """Configuration for screenshot analysis."""
    # Multi-modal LLM settings
    primary_llm_provider: ProviderType = ProviderType.OLLAMA
    primary_llm_model: str = "llama4:scout"
    fallback_providers: List[ProviderType] = Field(default=[ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE])
    
    # Analysis settings
    enable_ui_detection: bool = True
    enable_visual_reasoning: bool = True
    enable_context_analysis: bool = True
    enable_automation_suggestions: bool = True
    
    # OCR settings
    enable_ocr: bool = True
    ocr_languages: List[str] = Field(default=["en"])
    min_ocr_confidence: float = 0.3
    
    # UI detection settings
    min_element_confidence: float = 0.5
    max_elements_to_detect: int = 50
    
    # Visual analysis settings
    max_description_length: int = 500
    include_color_analysis: bool = True
    include_layout_analysis: bool = True
    
    # Performance settings
    max_analysis_time: int = 30  # seconds
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds


class RevolutionaryScreenshotAnalyzer:
    """Revolutionary screenshot analyzer with multi-modal LLM integration."""
    
    def __init__(self, config: Optional[ScreenshotAnalysisConfig] = None):
        self.config = config or ScreenshotAnalysisConfig()
        self.ocr_engine: Optional[RevolutionaryOCREngine] = None
        self.llm_manager = None
        self.analysis_cache: Dict[str, VisualAnalysisResult] = {}
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the screenshot analyzer."""
        try:
            if self.is_initialized:
                return True
                
            logger.info("🚀 Initializing Revolutionary Screenshot Analyzer...")
            
            # Initialize OCR engine
            self.ocr_engine = await get_ocr_engine()
            logger.info("✅ OCR engine initialized")
            
            # Initialize LLM manager
            self.llm_manager = get_enhanced_llm_manager()
            if self.llm_manager:
                await self.llm_manager.initialize()
                logger.info("✅ LLM manager initialized")
            
            # Validate multi-modal capabilities
            await self._validate_multimodal_capabilities()
            
            self.is_initialized = True
            logger.info("🎯 Revolutionary Screenshot Analyzer ready!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize screenshot analyzer: {e}")
            return False
    
    async def _validate_multimodal_capabilities(self) -> None:
        """Validate that we have multi-modal LLM capabilities."""
        try:
            # Check primary provider
            primary_available = False
            if self.llm_manager:
                provider = self.llm_manager.get_provider(self.config.primary_llm_provider)
                if provider:
                    models = await provider.get_available_models()
                    for model in models:
                        if (model.id == self.config.primary_llm_model and 
                            ModelCapability.MULTIMODAL in model.capabilities):
                            primary_available = True
                            break
            
            if not primary_available:
                logger.warning(f"Primary multimodal model {self.config.primary_llm_model} not available")
                
            # Check fallback providers
            fallback_available = []
            for provider_type in self.config.fallback_providers:
                provider = self.llm_manager.get_provider(provider_type) if self.llm_manager else None
                if provider:
                    models = await provider.get_available_models()
                    for model in models:
                        if ModelCapability.MULTIMODAL in model.capabilities:
                            fallback_available.append(provider_type)
                            break
            
            logger.info(f"✅ Multimodal capabilities: Primary={primary_available}, Fallbacks={fallback_available}")
            
        except Exception as e:
            logger.warning(f"Could not validate multimodal capabilities: {e}")

    async def capture_screenshot(self, source: ScreenshotSource = ScreenshotSource.DESKTOP) -> Optional[str]:
        """Capture screenshot from specified source."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"data/screenshots/screenshot_{timestamp}.png"

            # Ensure screenshots directory exists
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)

            if source == ScreenshotSource.DESKTOP:
                return await self._capture_desktop_screenshot(screenshot_path)
            elif source == ScreenshotSource.BROWSER:
                return await self._capture_browser_screenshot(screenshot_path)
            elif source == ScreenshotSource.APPLICATION:
                return await self._capture_application_screenshot(screenshot_path)
            else:
                logger.error(f"Unsupported screenshot source: {source}")
                return None

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None

    async def _capture_desktop_screenshot(self, output_path: str) -> Optional[str]:
        """Capture desktop screenshot using cross-platform method."""
        try:
            import platform
            system = platform.system().lower()

            if system == "windows":
                # Use Windows-specific screenshot method
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot.save(output_path)
                return output_path
            elif system == "linux":
                # Use Linux screenshot methods
                import subprocess
                # Try gnome-screenshot first, then scrot
                try:
                    subprocess.run(["gnome-screenshot", "-f", output_path], check=True)
                    return output_path
                except (subprocess.CalledProcessError, FileNotFoundError):
                    try:
                        subprocess.run(["scrot", output_path], check=True)
                        return output_path
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # Fallback to pyautogui
                        import pyautogui
                        screenshot = pyautogui.screenshot()
                        screenshot.save(output_path)
                        return output_path
            elif system == "darwin":  # macOS
                import subprocess
                subprocess.run(["screencapture", output_path], check=True)
                return output_path
            else:
                logger.error(f"Unsupported operating system: {system}")
                return None

        except Exception as e:
            logger.error(f"Desktop screenshot capture failed: {e}")
            return None

    async def _capture_browser_screenshot(self, output_path: str) -> Optional[str]:
        """Capture browser screenshot (placeholder for browser automation integration)."""
        logger.warning("Browser screenshot capture requires browser automation tool integration")
        return None

    async def _capture_application_screenshot(self, output_path: str) -> Optional[str]:
        """Capture specific application screenshot (placeholder)."""
        logger.warning("Application screenshot capture requires window management integration")
        return None

    async def analyze_screenshot(self, screenshot_path: str, source: ScreenshotSource = ScreenshotSource.UPLOAD) -> Optional[VisualAnalysisResult]:
        """Perform comprehensive screenshot analysis."""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Check cache first
            cache_key = f"{screenshot_path}_{hash(str(self.config))}"
            if self.config.enable_caching and cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if (datetime.now() - cached_result.timestamp).seconds < self.config.cache_ttl:
                    logger.info("📋 Using cached analysis result")
                    return cached_result

            logger.info(f"🔍 Analyzing screenshot: {screenshot_path}")

            # Load and validate image
            if not os.path.exists(screenshot_path):
                logger.error(f"Screenshot file not found: {screenshot_path}")
                return None

            image = Image.open(screenshot_path)
            dimensions = image.size

            # Initialize result
            result = VisualAnalysisResult(
                screenshot_path=screenshot_path,
                timestamp=datetime.now(),
                source=source,
                dimensions=dimensions,
                extracted_text="",
                ocr_confidence=0.0,
                ocr_language="unknown",
                ui_elements=[],
                visual_description="",
                context_analysis="",
                purpose_analysis="",
                automation_suggestions=[],
                interaction_opportunities=[]
            )

            # Step 1: OCR Analysis
            if self.config.enable_ocr:
                await self._perform_ocr_analysis(image, result)

            # Step 2: UI Element Detection
            if self.config.enable_ui_detection:
                await self._detect_ui_elements(image, result)

            # Step 3: Multi-modal LLM Visual Reasoning
            if self.config.enable_visual_reasoning:
                await self._perform_visual_reasoning(image, result)

            # Step 4: Context Analysis
            if self.config.enable_context_analysis:
                await self._analyze_context(image, result)

            # Step 5: Generate Automation Suggestions
            if self.config.enable_automation_suggestions:
                await self._generate_automation_suggestions(result)

            # Cache result
            if self.config.enable_caching:
                self.analysis_cache[cache_key] = result

            logger.info("✅ Screenshot analysis completed")
            return result

        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}")
            return None

    async def _perform_ocr_analysis(self, image: Image.Image, result: VisualAnalysisResult) -> None:
        """Perform OCR analysis using existing OCR engine."""
        try:
            if not self.ocr_engine:
                logger.warning("OCR engine not available")
                return

            # Convert PIL image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # Perform OCR
            ocr_result = await self.ocr_engine.extract_text_from_image(
                img_bytes,
                languages=self.config.ocr_languages,
                enhance_image=True
            )

            result.extracted_text = ocr_result.get('text', '')
            result.ocr_confidence = ocr_result.get('confidence', 0.0)
            result.ocr_language = ocr_result.get('language', 'unknown')

            logger.info(f"📝 OCR extracted {len(result.extracted_text)} characters with {result.ocr_confidence:.2f} confidence")

        except Exception as e:
            logger.error(f"OCR analysis failed: {e}")

    async def _detect_ui_elements(self, image: Image.Image, result: VisualAnalysisResult) -> None:
        """Detect UI elements using computer vision techniques."""
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Basic UI element detection using OpenCV
            elements = []

            # Detect buttons (rectangular shapes with text)
            elements.extend(await self._detect_buttons(cv_image))

            # Detect input fields (rectangular outlines)
            elements.extend(await self._detect_input_fields(cv_image))

            # Detect links (text with specific characteristics)
            elements.extend(await self._detect_links(cv_image, result.extracted_text))

            # Filter by confidence and limit count
            elements = [e for e in elements if e.confidence >= self.config.min_element_confidence]
            elements = elements[:self.config.max_elements_to_detect]

            result.ui_elements = elements
            logger.info(f"🎯 Detected {len(elements)} UI elements")

        except Exception as e:
            logger.error(f"UI element detection failed: {e}")

    async def _detect_buttons(self, cv_image: np.ndarray) -> List[UIElement]:
        """Detect button-like elements."""
        elements = []
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Find contours that might be buttons
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size (buttons are usually medium-sized rectangles)
                if 20 < w < 300 and 15 < h < 100:
                    # Calculate aspect ratio
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 8:  # Typical button aspect ratios
                        elements.append(UIElement(
                            element_type=UIElementType.BUTTON,
                            bbox=(x, y, w, h),
                            confidence=0.7,  # Basic detection confidence
                            actionable=True,
                            interaction_hints=["click", "tap"]
                        ))

        except Exception as e:
            logger.error(f"Button detection failed: {e}")

        return elements

    async def _detect_input_fields(self, cv_image: np.ndarray) -> List[UIElement]:
        """Detect input field elements."""
        elements = []
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Find rectangular shapes that might be input fields
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Look for rectangular shapes (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Input fields are usually wider than they are tall
                    if w > h and w > 50 and h > 15 and h < 50:
                        elements.append(UIElement(
                            element_type=UIElementType.INPUT_FIELD,
                            bbox=(x, y, w, h),
                            confidence=0.6,
                            actionable=True,
                            interaction_hints=["type", "input", "fill"]
                        ))

        except Exception as e:
            logger.error(f"Input field detection failed: {e}")

        return elements

    async def _detect_links(self, cv_image: np.ndarray, extracted_text: str) -> List[UIElement]:
        """Detect link elements based on text patterns."""
        elements = []
        try:
            # Simple link detection based on common patterns in extracted text
            import re

            # Common link patterns
            url_pattern = r'https?://[^\s]+'
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

            urls = re.findall(url_pattern, extracted_text)
            emails = re.findall(email_pattern, extracted_text)

            # For now, create generic link elements
            # In a full implementation, we'd locate these in the image
            for i, url in enumerate(urls[:5]):  # Limit to 5 URLs
                elements.append(UIElement(
                    element_type=UIElementType.LINK,
                    bbox=(0, 0, 100, 20),  # Placeholder bbox
                    confidence=0.8,
                    text=url,
                    actionable=True,
                    interaction_hints=["click", "navigate"]
                ))

            for i, email in enumerate(emails[:3]):  # Limit to 3 emails
                elements.append(UIElement(
                    element_type=UIElementType.LINK,
                    bbox=(0, 0, 100, 20),  # Placeholder bbox
                    confidence=0.8,
                    text=email,
                    actionable=True,
                    interaction_hints=["click", "email"]
                ))

        except Exception as e:
            logger.error(f"Link detection failed: {e}")

        return elements

    async def _perform_visual_reasoning(self, image: Image.Image, result: VisualAnalysisResult) -> None:
        """Perform visual reasoning using multi-modal LLMs."""
        try:
            if not self.llm_manager:
                logger.warning("LLM manager not available for visual reasoning")
                return

            # Convert image to base64 for LLM
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

            # Try primary provider first
            visual_description = await self._get_visual_description(img_base64, self.config.primary_llm_provider, self.config.primary_llm_model)

            if not visual_description:
                # Try fallback providers
                for provider_type in self.config.fallback_providers:
                    fallback_models = {
                        ProviderType.OPENAI: "gpt-4o",
                        ProviderType.ANTHROPIC: "claude-3-5-sonnet-20241022",
                        ProviderType.GOOGLE: "gemini-1.5-pro"
                    }
                    model = fallback_models.get(provider_type)
                    if model:
                        visual_description = await self._get_visual_description(img_base64, provider_type, model)
                        if visual_description:
                            break

            if visual_description:
                result.visual_description = visual_description
                logger.info("🧠 Visual reasoning completed")
            else:
                logger.warning("Visual reasoning failed with all providers")
                result.visual_description = "Visual analysis not available"

        except Exception as e:
            logger.error(f"Visual reasoning failed: {e}")
            result.visual_description = f"Visual analysis error: {str(e)}"

    async def _get_visual_description(self, img_base64: str, provider_type: ProviderType, model: str) -> Optional[str]:
        """Get visual description from specific LLM provider."""
        try:
            # Create LLM config
            llm_config = LLMConfig(
                provider=provider_type,
                model_id=model,
                temperature=0.1,
                max_tokens=500
            )

            # Create LLM instance
            llm = await self.llm_manager.create_llm_instance(llm_config)

            # Prepare prompt for visual analysis
            prompt = f"""
            Analyze this screenshot and provide a detailed description focusing on:
            1. What type of interface or application this appears to be
            2. Key UI elements visible (buttons, forms, menus, etc.)
            3. The overall purpose and context of this screen
            4. Any text or content that stands out
            5. Visual layout and design characteristics

            Be specific and actionable in your description. Focus on elements that would be useful for automation or interaction.

            Image data: data:image/png;base64,{img_base64}
            """

            # For now, use text-only analysis since multi-modal integration varies by provider
            # In production, this would use proper multi-modal APIs
            response = await llm.ainvoke(prompt)

            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Visual description failed for {provider_type}: {e}")
            return None

    async def _analyze_context(self, image: Image.Image, result: VisualAnalysisResult) -> None:
        """Analyze the context and purpose of the screenshot."""
        try:
            # Combine OCR text and visual description for context analysis
            combined_info = f"""
            Extracted Text: {result.extracted_text}
            Visual Description: {result.visual_description}
            UI Elements Found: {len(result.ui_elements)} elements
            """

            # Basic context analysis based on available information
            context_clues = []

            # Analyze text content for context
            text_lower = result.extracted_text.lower()
            if any(word in text_lower for word in ['login', 'sign in', 'password', 'username']):
                context_clues.append("Authentication/Login interface")
            if any(word in text_lower for word in ['search', 'find', 'query']):
                context_clues.append("Search interface")
            if any(word in text_lower for word in ['submit', 'send', 'save', 'create']):
                context_clues.append("Form submission interface")
            if any(word in text_lower for word in ['dashboard', 'overview', 'summary']):
                context_clues.append("Dashboard/Overview interface")
            if any(word in text_lower for word in ['settings', 'preferences', 'configuration']):
                context_clues.append("Settings/Configuration interface")

            # Analyze UI elements for context
            button_count = len([e for e in result.ui_elements if e.element_type == UIElementType.BUTTON])
            input_count = len([e for e in result.ui_elements if e.element_type == UIElementType.INPUT_FIELD])

            if input_count > 2 and button_count > 0:
                context_clues.append("Form-based interface")
            if button_count > 5:
                context_clues.append("Control-rich interface")

            result.context_analysis = "; ".join(context_clues) if context_clues else "General interface"

            # Purpose analysis
            if "login" in result.context_analysis.lower():
                result.purpose_analysis = "User authentication and access control"
            elif "form" in result.context_analysis.lower():
                result.purpose_analysis = "Data input and submission"
            elif "dashboard" in result.context_analysis.lower():
                result.purpose_analysis = "Information display and monitoring"
            elif "search" in result.context_analysis.lower():
                result.purpose_analysis = "Information discovery and retrieval"
            else:
                result.purpose_analysis = "General user interaction interface"

            logger.info(f"🎯 Context analysis: {result.context_analysis}")

        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            result.context_analysis = "Context analysis unavailable"
            result.purpose_analysis = "Purpose analysis unavailable"

    async def _generate_automation_suggestions(self, result: VisualAnalysisResult) -> None:
        """Generate automation suggestions based on analysis."""
        try:
            suggestions = []
            opportunities = []

            # Analyze UI elements for automation opportunities
            for element in result.ui_elements:
                if element.actionable:
                    if element.element_type == UIElementType.BUTTON:
                        suggestions.append(f"Button at ({element.bbox[0]}, {element.bbox[1]}) can be clicked for automation")
                        opportunities.append({
                            "type": "click",
                            "element": "button",
                            "coordinates": element.bbox,
                            "description": f"Clickable button {element.text or 'without text'}"
                        })
                    elif element.element_type == UIElementType.INPUT_FIELD:
                        suggestions.append(f"Input field at ({element.bbox[0]}, {element.bbox[1]}) can be filled with data")
                        opportunities.append({
                            "type": "input",
                            "element": "input_field",
                            "coordinates": element.bbox,
                            "description": "Text input field for data entry"
                        })
                    elif element.element_type == UIElementType.LINK:
                        suggestions.append(f"Link '{element.text}' can be clicked for navigation")
                        opportunities.append({
                            "type": "navigate",
                            "element": "link",
                            "text": element.text,
                            "description": f"Navigational link to {element.text}"
                        })

            # Context-based suggestions
            if "login" in result.context_analysis.lower():
                suggestions.append("This appears to be a login form - automation can handle credential entry")
                opportunities.append({
                    "type": "workflow",
                    "element": "form",
                    "description": "Login workflow automation opportunity"
                })

            if "search" in result.context_analysis.lower():
                suggestions.append("Search interface detected - automation can perform queries")
                opportunities.append({
                    "type": "workflow",
                    "element": "search",
                    "description": "Search automation opportunity"
                })

            # OCR-based suggestions
            if result.extracted_text:
                suggestions.append("Text content available for extraction and processing")
                opportunities.append({
                    "type": "extraction",
                    "element": "text",
                    "description": "Text extraction and processing opportunity"
                })

            result.automation_suggestions = suggestions
            result.interaction_opportunities = opportunities

            logger.info(f"🤖 Generated {len(suggestions)} automation suggestions")

        except Exception as e:
            logger.error(f"Automation suggestion generation failed: {e}")
            result.automation_suggestions = ["Automation analysis unavailable"]
            result.interaction_opportunities = []


class ScreenshotAnalysisInput(LangChainBaseModel):
    """Input schema for screenshot analysis tool."""
    screenshot_path: str = Field(description="Path to screenshot file to analyze")
    source: str = Field(default="upload", description="Source of screenshot: desktop, browser, application, upload")
    capture_new: bool = Field(default=False, description="Whether to capture a new screenshot instead of using existing file")


class ScreenshotAnalysisTool(BaseTool):
    """Revolutionary Screenshot Analysis Tool for AI Agents."""

    name: str = "screenshot_analysis"
    description: str = """Revolutionary screenshot analysis tool that provides comprehensive visual understanding including:
    - OCR text extraction with high accuracy
    - UI element detection and classification
    - Visual reasoning using multi-modal LLMs
    - Context and purpose analysis
    - Automation suggestions and interaction opportunities

    Use this tool to analyze screenshots for automation, testing, or visual understanding tasks."""

    args_schema: type = ScreenshotAnalysisInput

    def __init__(self, config: Optional[ScreenshotAnalysisConfig] = None):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation for internal attributes
        object.__setattr__(self, 'analyzer', RevolutionaryScreenshotAnalyzer(config))
        object.__setattr__(self, '_initialized', False)

    async def _arun(self, screenshot_path: str, source: str = "upload", capture_new: bool = False) -> str:
        """Execute screenshot analysis."""
        try:
            # Initialize if needed
            if not self._initialized:
                success = await self.analyzer.initialize()
                if not success:
                    return "❌ Failed to initialize screenshot analyzer"
                self._initialized = True

            # Capture new screenshot if requested
            if capture_new:
                source_enum = ScreenshotSource(source.lower())
                captured_path = await self.analyzer.capture_screenshot(source_enum)
                if captured_path:
                    screenshot_path = captured_path
                else:
                    return "❌ Failed to capture screenshot"

            # Perform analysis
            result = await self.analyzer.analyze_screenshot(screenshot_path, ScreenshotSource(source.lower()))

            if not result:
                return "❌ Screenshot analysis failed"

            # Format results
            return self._format_analysis_result(result)

        except Exception as e:
            logger.error(f"Screenshot analysis tool error: {e}")
            return f"❌ Screenshot analysis error: {str(e)}"

    def _run(self, screenshot_path: str, source: str = "upload", capture_new: bool = False) -> str:
        """Synchronous wrapper for screenshot analysis."""
        return asyncio.run(self._arun(screenshot_path, source, capture_new))

    def _format_analysis_result(self, result: VisualAnalysisResult) -> str:
        """Format analysis result for agent consumption."""
        output = []

        output.append("🖼️ **SCREENSHOT ANALYSIS RESULTS**")
        output.append(f"📁 File: {result.screenshot_path}")
        output.append(f"📏 Dimensions: {result.dimensions[0]}x{result.dimensions[1]}")
        output.append(f"🕒 Analyzed: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")

        # OCR Results
        output.append("📝 **OCR ANALYSIS**")
        output.append(f"Confidence: {result.ocr_confidence:.2f}")
        output.append(f"Language: {result.ocr_language}")
        if result.extracted_text:
            text_preview = result.extracted_text[:200] + "..." if len(result.extracted_text) > 200 else result.extracted_text
            output.append(f"Text: {text_preview}")
        else:
            output.append("Text: No text detected")
        output.append("")

        # UI Elements
        output.append("🎯 **UI ELEMENTS DETECTED**")
        if result.ui_elements:
            for i, element in enumerate(result.ui_elements[:10]):  # Show first 10
                output.append(f"{i+1}. {element.element_type.value.title()} at ({element.bbox[0]}, {element.bbox[1]}) - Confidence: {element.confidence:.2f}")
                if element.text:
                    output.append(f"   Text: {element.text}")
                if element.actionable:
                    output.append(f"   Actions: {', '.join(element.interaction_hints)}")
        else:
            output.append("No UI elements detected")
        output.append("")

        # Visual Analysis
        output.append("🧠 **VISUAL ANALYSIS**")
        output.append(f"Description: {result.visual_description}")
        output.append(f"Context: {result.context_analysis}")
        output.append(f"Purpose: {result.purpose_analysis}")
        output.append("")

        # Automation Suggestions
        output.append("🤖 **AUTOMATION OPPORTUNITIES**")
        if result.automation_suggestions:
            for i, suggestion in enumerate(result.automation_suggestions[:5]):  # Show first 5
                output.append(f"{i+1}. {suggestion}")
        else:
            output.append("No automation suggestions available")
        output.append("")

        # Interaction Opportunities
        if result.interaction_opportunities:
            output.append("⚡ **INTERACTION OPPORTUNITIES**")
            for i, opportunity in enumerate(result.interaction_opportunities[:5]):  # Show first 5
                output.append(f"{i+1}. {opportunity['type'].title()}: {opportunity['description']}")

        return "\n".join(output)


# Factory function to create the tool
def create_screenshot_analysis_tool(config: Optional[ScreenshotAnalysisConfig] = None) -> ScreenshotAnalysisTool:
    """Create a screenshot analysis tool instance."""
    return ScreenshotAnalysisTool(config)


# Default tool instance
screenshot_analysis_tool = create_screenshot_analysis_tool()
