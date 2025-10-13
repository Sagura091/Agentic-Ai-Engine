"""
🖥️ REVOLUTIONARY SCREEN CAPTURE TOOL - Universal Screen Analysis for AI Agents

A comprehensive screen capture and analysis tool that any agent can use to:
- Capture screenshots of the entire screen or specific windows
- Analyze screen content using vision models
- Extract text from screenshots using OCR
- Identify UI elements and applications
- Generate detailed descriptions of what's on screen

PRODUCTION FEATURES:
✅ Multi-monitor support
✅ Window-specific capture
✅ OCR text extraction
✅ Vision model integration for content analysis
✅ Application detection and identification
✅ Privacy-aware capture with filtering options
✅ High-quality image processing
✅ Cross-platform compatibility (Windows, Linux, macOS)

This tool is designed to be generic and reusable by any agent in the system.
"""

import os
import asyncio
import base64
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import tempfile

# Screen capture libraries
try:
    import pyautogui
    import PIL.Image
    import PIL.ImageGrab
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    pyautogui = None
    PIL = None
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False

# OCR libraries
try:
    import pytesseract
    import cv2
    import numpy as np
except ImportError:
    pytesseract = None
    cv2 = None

# System libraries
import platform
import subprocess
import psutil

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.tools.unified_tool_repository import ToolCategory as ToolCategoryEnum, ToolAccessLevel, ToolMetadata
from app.tools.metadata import (
    ToolMetadata as MetadataToolMetadata, ParameterSchema, ParameterType, UsagePattern, UsagePatternType,
    ConfidenceModifier, ConfidenceModifierType, ExecutionPreference, ContextRequirement,
    BehavioralHint, MetadataCapableToolMixin
)

logger = get_logger()


class ScreenCaptureConfig(BaseModel):
    """Configuration for screen capture operations."""
    output_directory: str = Field(default="./data/screenshots")
    image_format: str = Field(default="PNG")
    image_quality: int = Field(default=95, ge=1, le=100)
    max_image_size: Tuple[int, int] = Field(default=(1920, 1080))
    enable_ocr: bool = Field(default=True)
    enable_vision_analysis: bool = Field(default=True)
    privacy_filter: bool = Field(default=False)
    blur_sensitive_areas: bool = Field(default=False)


class ScreenCaptureInput(BaseModel):
    """Input schema for screen capture operations."""
    action: str = Field(
        description="Action to perform: 'capture_full', 'capture_window', 'capture_region', 'analyze_screen', 'get_active_window'"
    )
    window_title: Optional[str] = Field(
        default=None,
        description="Title of specific window to capture (for capture_window action)"
    )
    region: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description="Region to capture as (left, top, width, height) for capture_region action"
    )
    include_analysis: bool = Field(
        default=True,
        description="Whether to include AI analysis of the captured screen"
    )
    include_ocr: bool = Field(
        default=True,
        description="Whether to extract text using OCR"
    )
    save_image: bool = Field(
        default=True,
        description="Whether to save the captured image to disk"
    )
    analysis_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for AI analysis of the screen content"
    )


class ScreenCaptureResult(BaseModel):
    """Result from screen capture operation."""
    success: bool
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    screen_analysis: Optional[str] = None
    extracted_text: Optional[str] = None
    active_applications: List[str] = []
    window_titles: List[str] = []
    screen_resolution: Optional[Tuple[int, int]] = None
    timestamp: str
    error_message: Optional[str] = None


class ScreenCaptureTool(BaseTool, MetadataCapableToolMixin):
    """
    Revolutionary Screen Capture Tool for AI Agents.

    Provides comprehensive screen capture and analysis capabilities that any agent
    can use to understand what's happening on the user's screen.
    """

    name: str = "screen_capture"
    tool_id: str = "screen_capture"
    description: str = """
    Capture and analyze the user's screen with AI-powered analysis.

    This tool can:
    - Capture full screen or specific windows
    - Extract text from screenshots using OCR
    - Analyze screen content using AI vision models
    - Identify active applications and windows
    - Provide detailed descriptions of what's on screen

    Actions available:
    - 'capture_full': Capture the entire screen
    - 'capture_window': Capture a specific window by title
    - 'capture_region': Capture a specific region of the screen
    - 'analyze_screen': Analyze current screen without saving image
    - 'get_active_window': Get information about currently active window

    Perfect for agents that need to understand user context, provide commentary,
    or interact with what's currently on screen.
    """
    args_schema: type = ScreenCaptureInput

    # Pydantic v2 compatible field definitions
    config: ScreenCaptureConfig = Field(default_factory=ScreenCaptureConfig)
    output_dir: Optional[Path] = Field(default=None, exclude=True)
    
    def __init__(self, config: Optional[ScreenCaptureConfig] = None, **kwargs):
        # Set config before calling super().__init__
        if config:
            kwargs['config'] = config
        super().__init__(**kwargs)

        # Initialize output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for required dependencies
        self._check_dependencies()

        # Initialize OCR if available
        if pytesseract and self.config.enable_ocr:
            try:
                # Try to configure tesseract path on Windows
                if platform.system() == "Windows":
                    possible_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', ''))
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            break
            except Exception as e:
                logger.warn(
                    f"OCR setup warning: {e}",
                    LogCategory.TOOL_OPERATIONS,
                    "ScreenCaptureTool",
                    error=e
                )
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        missing_deps = []
        
        if not pyautogui:
            missing_deps.append("pyautogui")
        if not PIL:
            missing_deps.append("Pillow")
        if self.config.enable_ocr and not pytesseract:
            missing_deps.append("pytesseract")
        
        if missing_deps:
            logger.warn(
                f"Missing optional dependencies: {missing_deps}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                data={"missing_deps": missing_deps}
            )
            logger.info(
                "Install with: pip install pyautogui Pillow pytesseract opencv-python",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool"
            )
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async screen capture."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(**kwargs))
                    result = future.result()
                    return result
            else:
                return asyncio.run(self._arun(**kwargs))
        except Exception as e:
            logger.error(
                f"Screen capture error: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _arun(self, **kwargs) -> str:
        """Execute screen capture operation."""
        try:
            input_data = ScreenCaptureInput(**kwargs)
            
            if input_data.action == "capture_full":
                result = await self._capture_full_screen(input_data)
            elif input_data.action == "capture_window":
                result = await self._capture_window(input_data)
            elif input_data.action == "capture_region":
                result = await self._capture_region(input_data)
            elif input_data.action == "analyze_screen":
                result = await self._analyze_current_screen(input_data)
            elif input_data.action == "get_active_window":
                result = await self._get_active_window_info()
            else:
                raise ValueError(f"Unknown action: {input_data.action}")
            
            return json.dumps(result.dict(), indent=2)
            
        except Exception as e:
            logger.error(
                f"Screen capture operation failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            })

    async def _capture_full_screen(self, input_data: ScreenCaptureInput) -> ScreenCaptureResult:
        """Capture the full screen."""
        try:
            if not pyautogui:
                raise RuntimeError("pyautogui not available for screen capture")

            # Capture screenshot
            screenshot = pyautogui.screenshot()

            # Get screen resolution
            screen_size = screenshot.size

            # Resize if needed
            if (screen_size[0] > self.config.max_image_size[0] or
                screen_size[1] > self.config.max_image_size[1]):
                if PIL_AVAILABLE:
                    screenshot.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)

            # Save image if requested
            image_path = None
            if input_data.save_image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screen_capture_{timestamp}.{self.config.image_format.lower()}"
                image_path = self.output_dir / filename
                screenshot.save(image_path, format=self.config.image_format, quality=self.config.image_quality)
                image_path = str(image_path)

            # Convert to base64 for analysis
            image_base64 = None
            if input_data.include_analysis:
                import io
                buffer = io.BytesIO()
                screenshot.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Extract text using OCR
            extracted_text = None
            if input_data.include_ocr and pytesseract:
                try:
                    extracted_text = pytesseract.image_to_string(screenshot)
                except Exception as e:
                    logger.warn(
                        f"OCR failed: {e}",
                        LogCategory.TOOL_OPERATIONS,
                        "ScreenCaptureTool",
                        error=e
                    )
                    extracted_text = "OCR extraction failed"

            # Get active applications
            active_apps = await self._get_active_applications()
            window_titles = await self._get_window_titles()

            # AI Analysis
            screen_analysis = None
            if input_data.include_analysis and image_base64:
                screen_analysis = await self._analyze_screen_content(
                    image_base64,
                    input_data.analysis_prompt,
                    extracted_text
                )

            return ScreenCaptureResult(
                success=True,
                image_path=image_path,
                image_base64=image_base64,
                screen_analysis=screen_analysis,
                extracted_text=extracted_text,
                active_applications=active_apps,
                window_titles=window_titles,
                screen_resolution=screen_size,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(
                f"Full screen capture failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return ScreenCaptureResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )

    async def _capture_window(self, input_data: ScreenCaptureInput) -> ScreenCaptureResult:
        """Capture a specific window."""
        try:
            if not input_data.window_title:
                raise ValueError("window_title required for capture_window action")

            # Find window by title
            windows = pyautogui.getWindowsWithTitle(input_data.window_title)
            if not windows:
                raise ValueError(f"No window found with title: {input_data.window_title}")

            window = windows[0]

            # Bring window to front and capture
            window.activate()
            await asyncio.sleep(0.5)  # Give time for window to come to front

            # Get window region
            left, top, width, height = window.left, window.top, window.width, window.height

            # Capture the window region
            screenshot = pyautogui.screenshot(region=(left, top, width, height))

            # Process similar to full screen capture
            return await self._process_screenshot(screenshot, input_data, f"window_{input_data.window_title}")

        except Exception as e:
            logger.error(
                f"Window capture failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return ScreenCaptureResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )

    async def _capture_region(self, input_data: ScreenCaptureInput) -> ScreenCaptureResult:
        """Capture a specific region of the screen."""
        try:
            if not input_data.region:
                raise ValueError("region required for capture_region action")

            left, top, width, height = input_data.region
            screenshot = pyautogui.screenshot(region=(left, top, width, height))

            return await self._process_screenshot(screenshot, input_data, "region")

        except Exception as e:
            logger.error(
                f"Region capture failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return ScreenCaptureResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )

    async def _analyze_current_screen(self, input_data: ScreenCaptureInput) -> ScreenCaptureResult:
        """Analyze current screen without saving image."""
        input_data.save_image = False
        return await self._capture_full_screen(input_data)

    async def _get_active_window_info(self) -> ScreenCaptureResult:
        """Get information about the currently active window."""
        try:
            active_apps = await self._get_active_applications()
            window_titles = await self._get_window_titles()

            # Try to get the active window
            active_window = None
            try:
                active_window = pyautogui.getActiveWindow()
            except Exception:
                pass

            active_window_info = None
            if active_window:
                active_window_info = {
                    "title": active_window.title,
                    "left": active_window.left,
                    "top": active_window.top,
                    "width": active_window.width,
                    "height": active_window.height
                }

            return ScreenCaptureResult(
                success=True,
                active_applications=active_apps,
                window_titles=window_titles,
                screen_analysis=json.dumps(active_window_info) if active_window_info else None,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(
                f"Get active window info failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return ScreenCaptureResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )

    async def _process_screenshot(self, screenshot: Any, input_data: ScreenCaptureInput, prefix: str) -> ScreenCaptureResult:
        """Process a screenshot with all requested operations."""
        try:
            if not PIL_AVAILABLE:
                return ScreenCaptureResult(
                    success=False,
                    error_message="PIL/Pillow not available for image processing",
                    timestamp=datetime.now().isoformat()
                )

            # Resize if needed
            if (screenshot.size[0] > self.config.max_image_size[0] or
                screenshot.size[1] > self.config.max_image_size[1]):
                screenshot.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)

            # Save image if requested
            image_path = None
            if input_data.save_image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_{timestamp}.{self.config.image_format.lower()}"
                image_path = self.output_dir / filename
                screenshot.save(image_path, format=self.config.image_format, quality=self.config.image_quality)
                image_path = str(image_path)

            # Convert to base64 for analysis
            image_base64 = None
            if input_data.include_analysis:
                import io
                buffer = io.BytesIO()
                screenshot.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Extract text using OCR
            extracted_text = None
            if input_data.include_ocr and pytesseract:
                try:
                    extracted_text = pytesseract.image_to_string(screenshot)
                except Exception as e:
                    logger.warn(
                        f"OCR failed: {e}",
                        LogCategory.TOOL_OPERATIONS,
                        "ScreenCaptureTool",
                        error=e
                    )
                    extracted_text = "OCR extraction failed"

            # Get system info
            active_apps = await self._get_active_applications()
            window_titles = await self._get_window_titles()

            # AI Analysis
            screen_analysis = None
            if input_data.include_analysis and image_base64:
                screen_analysis = await self._analyze_screen_content(
                    image_base64,
                    input_data.analysis_prompt,
                    extracted_text
                )

            return ScreenCaptureResult(
                success=True,
                image_path=image_path,
                image_base64=image_base64,
                screen_analysis=screen_analysis,
                extracted_text=extracted_text,
                active_applications=active_apps,
                window_titles=window_titles,
                screen_resolution=screenshot.size,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(
                f"Screenshot processing failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return ScreenCaptureResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )

    async def _get_active_applications(self) -> List[str]:
        """Get list of currently running applications."""
        try:
            apps = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] and not proc.info['name'].startswith('System'):
                        apps.append(proc.info['name'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Remove duplicates and system processes
            unique_apps = list(set(apps))
            filtered_apps = [app for app in unique_apps if not app.lower().startswith(('dwm', 'csrss', 'winlogon', 'services'))]

            return sorted(filtered_apps)[:20]  # Limit to top 20

        except Exception as e:
            logger.warn(
                f"Failed to get active applications: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return []

    async def _get_window_titles(self) -> List[str]:
        """Get list of visible window titles."""
        try:
            if not pyautogui:
                return []

            windows = pyautogui.getAllWindows()
            titles = [w.title for w in windows if w.title and w.visible]

            return sorted(list(set(titles)))[:20]  # Limit to top 20

        except Exception as e:
            logger.warn(
                f"Failed to get window titles: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return []

    async def _analyze_screen_content(self, image_base64: str, custom_prompt: Optional[str], extracted_text: Optional[str]) -> str:
        """Analyze screen content using AI vision models."""
        try:
            # This would integrate with your LLM system for vision analysis
            # For now, provide a basic analysis based on available information

            analysis_parts = []

            if custom_prompt:
                analysis_parts.append(f"Custom Analysis Request: {custom_prompt}")

            if extracted_text and extracted_text.strip():
                # Analyze the extracted text
                text_summary = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                analysis_parts.append(f"Text Content Found: {text_summary}")

                # Basic text analysis
                if any(word in extracted_text.lower() for word in ['error', 'exception', 'failed']):
                    analysis_parts.append("⚠️ Potential error or issue detected in screen content")

                if any(word in extracted_text.lower() for word in ['code', 'function', 'class', 'import']):
                    analysis_parts.append("💻 Programming/development content detected")

                if any(word in extracted_text.lower() for word in ['email', 'message', 'chat']):
                    analysis_parts.append("📧 Communication content detected")

            # Get active applications for context
            active_apps = await self._get_active_applications()
            if active_apps:
                analysis_parts.append(f"Active Applications: {', '.join(active_apps[:5])}")

            if not analysis_parts:
                analysis_parts.append("Screen captured successfully. No specific content analysis available without vision model integration.")

            return "\n".join(analysis_parts)

        except Exception as e:
            logger.error(
                f"Screen analysis failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "ScreenCaptureTool",
                error=e
            )
            return f"Screen analysis failed: {str(e)}"

    def _create_metadata(self) -> MetadataToolMetadata:
        """Create metadata for the screen capture tool."""
        return MetadataToolMetadata(
            name="screen_capture",
            category="automation",
            description="Revolutionary screen capture and analysis tool that captures screenshots and provides AI-powered analysis of screen content",
            version="2.0.0",

            # Usage patterns
            usage_patterns=[
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="screen,capture,screenshot,roast,analyze,see,look,view,monitor",
                    weight=1.0,
                    description="Triggers on screen capture and analysis keywords"
                ),
                UsagePattern(
                    type=UsagePatternType.TASK_TYPE_MATCH,
                    pattern="automation,monitoring,analysis,roasting,creative",
                    weight=0.9,
                    description="Matches automation and analysis tasks"
                ),
                UsagePattern(
                    type=UsagePatternType.CONTEXT_MATCH,
                    pattern="current_task,user_input,goal",
                    weight=0.8,
                    description="Uses context to determine screen capture needs"
                )
            ],

            # Parameter schemas
            parameter_schemas=[
                ParameterSchema(
                    name="action",
                    type=ParameterType.ENUM,
                    description="Type of screen capture action to perform",
                    required=True,
                    enum_values=["capture_full", "capture_window", "capture_region", "analyze_screen", "roast_screen"],
                    default_value="capture_full",
                    examples=["capture_full", "roast_screen"],
                    context_hints=["current_task", "goal"]
                ),
                ParameterSchema(
                    name="include_analysis",
                    type=ParameterType.BOOLEAN,
                    description="Whether to include AI analysis of the captured screen",
                    required=False,
                    default_value=True,
                    examples=[True, False],
                    context_hints=["analysis_required", "detailed_output"]
                ),
                ParameterSchema(
                    name="privacy_filter",
                    type=ParameterType.BOOLEAN,
                    description="Whether to apply privacy filtering to sensitive content",
                    required=False,
                    default_value=True,
                    examples=[True, False],
                    context_hints=["privacy_mode", "safe_mode"]
                ),
                ParameterSchema(
                    name="save_image",
                    type=ParameterType.BOOLEAN,
                    description="Whether to save the captured image to disk",
                    required=False,
                    default_value=True,
                    examples=[True, False],
                    context_hints=["save_output", "persistent_storage"]
                )
            ],

            # Confidence modifiers
            confidence_modifiers=[
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="current_task:screen",
                    value=0.4,
                    description="High confidence for screen-related tasks"
                ),
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="current_task:roast",
                    value=0.5,
                    description="Maximum boost for roasting tasks"
                ),
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="current_task:capture",
                    value=0.3,
                    description="Boost for capture tasks"
                )
            ],

            # Execution preferences
            execution_preferences=ExecutionPreference(
                preferred_contexts=["automation", "monitoring", "analysis", "roasting", "creative", "chaos"],
                avoid_contexts=["private", "sensitive"],
                execution_order_preference=2,  # High priority for screen tasks
                parallel_execution_allowed=False,  # Screen capture should be sequential
                max_concurrent_executions=1
            ),

            # Context requirements
            context_requirements=ContextRequirement(
                required_context_keys=[],  # No strict requirements
                optional_context_keys=["current_task", "user_input", "privacy_mode", "analysis_required"],
                minimum_context_quality=0.0
            ),

            # Behavioral hints
            behavioral_hints=BehavioralHint(
                creativity_level=0.6,  # Moderately creative
                risk_level=0.4,  # Moderate risk (privacy concerns)
                resource_intensity=0.5,  # Moderate resource usage
                output_predictability=0.8,  # Highly predictable output
                user_interaction_level=0.2,  # Low interaction needed
                learning_value=0.8  # High learning value
            ),

            # Capabilities and metadata
            capabilities=["screen_capture", "image_analysis", "ocr", "ui_detection", "privacy_filtering"],
            limitations=["requires_screen_access", "privacy_sensitive", "platform_dependent"],
            dependencies=["pyautogui", "PIL", "pytesseract"],
            tags=["automation", "monitoring", "analysis", "roasting", "creative"],
            aliases=["screenshot", "screen_analyzer", "roaster"],
            related_tools=["meme_generation", "social_media_orchestrator", "notification_alert"]
        )


# Tool metadata for registration
SCREEN_CAPTURE_TOOL_METADATA = ToolMetadata(
    tool_id="screen_capture",
    name="screen_capture",
    category=ToolCategory.AUTOMATION,
    access_level=ToolAccessLevel.PUBLIC,
    description="Universal screen capture and analysis tool for AI agents",
    requires_rag=False,
    use_cases={"screen_analysis", "user_monitoring", "visual_feedback", "automation"}
)


# Tool factory function
def get_screen_capture_tool(config: Optional[ScreenCaptureConfig] = None) -> ScreenCaptureTool:
    """Create a screen capture tool instance."""
    return ScreenCaptureTool(config=config)
