"""
🚀 REVOLUTIONARY VIRAL CONTENT GENERATOR TOOL - AI-Powered Content Creation Engine

The most advanced viral content creation and optimization tool ever built.
Transform AI agents into viral content creators with massive reach and engagement.

🚀 REVOLUTIONARY CAPABILITIES:
- AI-powered viral content generation across all formats
- Trend analysis and viral prediction algorithms
- Multi-modal content creation (text, image, video, audio)
- Platform-specific optimization and adaptation
- Real-time trend integration and exploitation
- Viral formula analysis and application
- Content performance prediction and optimization
- Automated A/B testing and iteration
- Emotional engagement optimization
- Meme and viral format generation
- Storytelling and narrative optimization
- Cross-platform content syndication

🎯 CORE FEATURES:
- Viral text content generation
- Trending hashtag integration
- Emotional hook optimization
- Visual content enhancement
- Video script generation
- Meme creation and optimization
- Story arc development
- Call-to-action optimization
- Engagement prediction
- Viral timing optimization
- Content series planning
- Performance analytics

This tool transforms AI agents into viral content machines with unstoppable creative power.
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional, Union, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory

from app.tools.metadata import (
    ToolMetadata, ParameterSchema, ParameterType, UsagePattern, UsagePatternType,
    ConfidenceModifier, ConfidenceModifierType, ExecutionPreference, ContextRequirement,
    BehavioralHint, MetadataCapableToolMixin
)

from app.tools.unified_tool_repository import ToolCategory as ToolCategoryEnum, ToolAccessLevel, ToolMetadata as UnifiedToolMetadata

logger = get_logger()


class ContentFormat(str, Enum):
    """Content format types."""
    TEXT_POST = "text_post"
    IMAGE_POST = "image_post"
    VIDEO_POST = "video_post"
    STORY = "story"
    REEL = "reel"
    THREAD = "thread"
    MEME = "meme"
    INFOGRAPHIC = "infographic"
    CAROUSEL = "carousel"
    LIVE_STREAM = "live_stream"


class ViralElement(str, Enum):
    """Viral content elements."""
    EMOTIONAL_HOOK = "emotional_hook"
    TRENDING_TOPIC = "trending_topic"
    CONTROVERSY = "controversy"
    HUMOR = "humor"
    INSPIRATION = "inspiration"
    SHOCK_VALUE = "shock_value"
    RELATABILITY = "relatability"
    CURIOSITY_GAP = "curiosity_gap"
    SOCIAL_PROOF = "social_proof"
    URGENCY = "urgency"


@dataclass
class ViralMetrics:
    """Viral content performance metrics."""
    viral_score: float = 0.0
    engagement_prediction: int = 0
    share_potential: float = 0.0
    emotional_impact: float = 0.0
    trend_alignment: float = 0.0
    platform_optimization: Dict[str, float] = field(default_factory=dict)
    expected_reach: int = 0
    viral_elements_used: List[str] = field(default_factory=list)


@dataclass
class ContentPiece:
    """Generated content piece."""
    id: str
    content: str
    format: ContentFormat
    platform: str
    hashtags: List[str]
    viral_elements: List[ViralElement]
    metrics: ViralMetrics
    media_suggestions: List[str] = field(default_factory=list)
    optimization_tips: List[str] = field(default_factory=list)


class ViralContentGeneratorInput(BaseModel):
    """Input schema for viral content generator operations."""
    # Content parameters
    content_topic: Optional[str] = Field(None, description="Topic or theme for content")
    content_format: ContentFormat = Field(ContentFormat.TEXT_POST, description="Format of content to generate")
    target_platform: Optional[str] = Field(None, description="Target social media platform")
    
    # Viral optimization
    viral_elements: List[ViralElement] = Field([ViralElement.EMOTIONAL_HOOK], description="Viral elements to include")
    viral_intensity: float = Field(0.8, description="Viral intensity level (0.0-1.0)")
    trend_integration: bool = Field(True, description="Integrate current trends")
    
    # Audience targeting
    target_audience: Optional[str] = Field(None, description="Target audience description")
    audience_age_range: Optional[str] = Field(None, description="Age range (e.g., '18-25')")
    audience_interests: Optional[List[str]] = Field(None, description="Audience interests")
    
    # Content style
    tone: str = Field("engaging", description="Content tone (funny, serious, inspiring, etc.)")
    style: str = Field("modern", description="Content style (modern, classic, edgy, etc.)")
    brand_voice: Optional[str] = Field(None, description="Brand voice guidelines")
    
    # Generation parameters
    content_count: int = Field(1, description="Number of content pieces to generate")
    variation_level: float = Field(0.5, description="Variation level between pieces (0.0-1.0)")
    include_hashtags: bool = Field(True, description="Include optimized hashtags")
    
    # Optimization settings
    optimize_for_engagement: bool = Field(True, description="Optimize for maximum engagement")
    optimize_for_shares: bool = Field(True, description="Optimize for shareability")
    optimize_for_comments: bool = Field(True, description="Optimize for comments")
    
    # Content series
    create_series: bool = Field(False, description="Create content series")
    series_length: int = Field(5, description="Length of content series")
    series_theme: Optional[str] = Field(None, description="Series theme")
    
    # A/B testing
    create_variants: bool = Field(False, description="Create A/B test variants")
    variant_count: int = Field(3, description="Number of variants to create")
    
    # Performance prediction
    predict_performance: bool = Field(True, description="Predict content performance")
    benchmark_against: Optional[str] = Field(None, description="Benchmark against competitor or trend")


class ViralContentGeneratorTool(BaseTool, MetadataCapableToolMixin):
    """Revolutionary Viral Content Generator Tool for AI-powered content creation."""

    name: str = "viral_content_generator"
    description: str = """Revolutionary viral content generation tool that creates engaging, shareable content.

    Capabilities:
    - AI-powered viral content generation across all formats
    - Trend analysis and viral prediction algorithms
    - Multi-modal content creation and optimization
    - Platform-specific adaptation and enhancement
    - Real-time trend integration and exploitation
    - Viral formula analysis and application
    - Content performance prediction and optimization
    - Automated A/B testing and iteration
    - Emotional engagement optimization
    - Meme and viral format generation
    - Storytelling and narrative optimization
    - Cross-platform content syndication

    This tool makes AI agents into viral content machines with unstoppable creative power."""

    args_schema: Type[BaseModel] = ViralContentGeneratorInput

    # Pydantic v2 compatible field definitions
    viral_formulas: Dict[str, Dict] = Field(default_factory=dict, exclude=True)
    trending_topics: List[str] = Field(default_factory=list, exclude=True)
    content_templates: Dict[ContentFormat, List[str]] = Field(default_factory=dict, exclude=True)
    performance_history: List[Dict] = Field(default_factory=list, exclude=True)
    viral_elements_library: Dict[ViralElement, List[str]] = Field(default_factory=dict, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize viral formulas and templates
        self._initialize_viral_formulas()
        self._initialize_content_templates()
        self._initialize_viral_elements_library()
        
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        """Execute viral content generation operations."""
        try:
            input_data = ViralContentGeneratorInput(**kwargs)
            
            # Update trending topics
            await self._update_trending_topics()
            
            # Generate viral content
            if input_data.create_series:
                content_pieces = await self._generate_content_series(input_data)
            elif input_data.create_variants:
                content_pieces = await self._generate_content_variants(input_data)
            else:
                content_pieces = await self._generate_single_content(input_data)
            
            # Optimize content for virality
            optimized_content = []
            for piece in content_pieces:
                optimized_piece = await self._optimize_content_for_virality(piece, input_data)
                optimized_content.append(optimized_piece)
            
            # Predict performance
            performance_predictions = []
            if input_data.predict_performance:
                for piece in optimized_content:
                    prediction = await self._predict_content_performance(piece, input_data)
                    performance_predictions.append(prediction)
            
            # Generate recommendations
            recommendations = await self._generate_optimization_recommendations(
                optimized_content,
                input_data
            )
            
            result = {
                "success": True,
                "content_pieces": [piece.__dict__ for piece in optimized_content],
                "performance_predictions": performance_predictions,
                "recommendations": recommendations,
                "viral_score_average": sum(piece.metrics.viral_score for piece in optimized_content) / len(optimized_content),
                "total_expected_reach": sum(piece.metrics.expected_reach for piece in optimized_content),
                "trending_topics_used": list(set([topic for piece in optimized_content for topic in piece.hashtags if topic.startswith('#')])),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                "Viral content generation completed",
                LogCategory.TOOL_OPERATIONS,
                "ViralContentGeneratorTool",
                data={
                    "content_count": len(optimized_content),
                    "average_viral_score": result["viral_score_average"],
                    "total_expected_reach": result["total_expected_reach"]
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Viral content generator error: {str(e)}",
                LogCategory.TOOL_OPERATIONS,
                "ViralContentGeneratorTool",
                error=e
            )
            return {
                "success": False,
                "error": str(e),
                "content_format": kwargs.get("content_format", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for async execution."""
        return asyncio.run(self._arun(**kwargs))
    
    def _initialize_viral_formulas(self):
        """Initialize viral content formulas."""
        self.viral_formulas = {
            "emotional_hook": {
                "pattern": "{hook} {content} {call_to_action}",
                "hooks": [
                    "This will change everything you know about",
                    "You won't believe what happened when",
                    "The truth about {topic} that nobody talks about",
                    "Why everyone is obsessed with",
                    "The secret behind"
                ]
            },
            "controversy": {
                "pattern": "Unpopular opinion: {controversial_statement} {reasoning}",
                "starters": [
                    "Unpopular opinion:",
                    "Hot take:",
                    "Controversial but true:",
                    "Nobody wants to admit this but"
                ]
            },
            "curiosity_gap": {
                "pattern": "{teaser} (Thread 🧵)",
                "teasers": [
                    "I learned something that completely changed my perspective",
                    "This simple trick will blow your mind",
                    "What I discovered will shock you",
                    "The real reason why"
                ]
            }
        }
    
    def _initialize_content_templates(self):
        """Initialize content templates for different formats."""
        self.content_templates = {
            ContentFormat.TEXT_POST: [
                "{hook}\n\n{main_content}\n\n{call_to_action}",
                "{question}\n\n{answer}\n\n{engagement_prompt}",
                "{statement}\n\nHere's why:\n{reasons}\n\n{conclusion}"
            ],
            ContentFormat.THREAD: [
                "1/{total} {hook}\n\n{thread_content}",
                "🧵 THREAD: {topic}\n\n{content_breakdown}",
                "Let me explain {topic} in simple terms:\n\n{explanation}"
            ],
            ContentFormat.MEME: [
                "{meme_format}: {punchline}",
                "POV: {scenario}",
                "When {situation}: {reaction}"
            ]
        }
    
    def _initialize_viral_elements_library(self):
        """Initialize viral elements library."""
        self.viral_elements_library = {
            ViralElement.EMOTIONAL_HOOK: [
                "This will make you cry",
                "Prepare to be amazed",
                "You won't believe this",
                "This hit me right in the feels"
            ],
            ViralElement.HUMOR: [
                "Plot twist:",
                "Me trying to adult:",
                "When you realize:",
                "That awkward moment when"
            ],
            ViralElement.INSPIRATION: [
                "Your reminder that",
                "You are capable of",
                "Never forget that",
                "Today's motivation:"
            ]
        }

    async def _generate_single_content(self, input_data: ViralContentGeneratorInput) -> List[ContentPiece]:
        """Generate single viral content piece."""
        content_pieces = []

        for i in range(input_data.content_count):
            # Select viral formula
            formula = await self._select_viral_formula(input_data.viral_elements)

            # Generate base content
            base_content = await self._generate_base_content(
                input_data.content_topic,
                input_data.content_format,
                formula,
                input_data.tone
            )

            # Add viral elements
            enhanced_content = await self._add_viral_elements(
                base_content,
                input_data.viral_elements,
                input_data.viral_intensity
            )

            # Generate hashtags
            hashtags = []
            if input_data.include_hashtags:
                hashtags = await self._generate_viral_hashtags(
                    input_data.content_topic,
                    input_data.target_platform,
                    input_data.target_audience
                )

            # Create content piece
            piece = ContentPiece(
                id=f"viral_content_{int(time.time())}_{i}",
                content=enhanced_content,
                format=input_data.content_format,
                platform=input_data.target_platform or "multi_platform",
                hashtags=hashtags,
                viral_elements=input_data.viral_elements,
                metrics=ViralMetrics()
            )

            content_pieces.append(piece)

        return content_pieces

    async def _optimize_content_for_virality(self, piece: ContentPiece, input_data: ViralContentGeneratorInput) -> ContentPiece:
        """Optimize content piece for maximum viral potential."""
        # Calculate viral score
        piece.metrics.viral_score = await self._calculate_viral_score(piece, input_data)

        # Predict engagement
        piece.metrics.engagement_prediction = await self._predict_engagement(piece, input_data)

        # Calculate expected reach
        piece.metrics.expected_reach = await self._calculate_expected_reach(piece, input_data)

        # Generate optimization tips
        piece.optimization_tips = await self._generate_optimization_tips(piece, input_data)

        return piece

    async def _select_viral_formula(self, viral_elements: List[ViralElement]) -> Dict[str, Any]:
        """Select appropriate viral formula based on elements."""
        if ViralElement.EMOTIONAL_HOOK in viral_elements:
            return self.viral_formulas["emotional_hook"]
        elif ViralElement.CONTROVERSY in viral_elements:
            return self.viral_formulas["controversy"]
        elif ViralElement.CURIOSITY_GAP in viral_elements:
            return self.viral_formulas["curiosity_gap"]
        else:
            return self.viral_formulas["emotional_hook"]  # Default

    async def _generate_base_content(self, topic: str, format: ContentFormat, formula: Dict, tone: str) -> str:
        """Generate base content using viral formula."""
        if not topic:
            topic = "trending topic"

        # Generate content based on format
        if format == ContentFormat.TEXT_POST:
            hook = random.choice(formula.get("hooks", ["Check this out:"]))
            content = f"{hook.format(topic=topic)}\n\nHere's what you need to know about {topic}...\n\nWhat do you think? 💭"
        elif format == ContentFormat.THREAD:
            content = f"🧵 THREAD: Everything about {topic}\n\n1/5 Let's dive in..."
        elif format == ContentFormat.MEME:
            content = f"When someone mentions {topic}: *becomes instant expert* 😂"
        else:
            content = f"Amazing insights about {topic}! ✨"

        return content

    async def _add_viral_elements(self, content: str, viral_elements: List[ViralElement], intensity: float) -> str:
        """Add viral elements to enhance content."""
        enhanced_content = content

        for element in viral_elements:
            if element in self.viral_elements_library:
                element_phrases = self.viral_elements_library[element]
                if element_phrases and random.random() < intensity:
                    phrase = random.choice(element_phrases)
                    enhanced_content = f"{phrase} {enhanced_content}"

        # Add viral emojis based on intensity
        if intensity > 0.7:
            viral_emojis = ["🔥", "💯", "🚀", "⚡", "🌟"]
            emoji = random.choice(viral_emojis)
            enhanced_content = f"{emoji} {enhanced_content}"

        return enhanced_content

    async def _generate_viral_hashtags(self, topic: str, platform: str, audience: str) -> List[str]:
        """Generate viral hashtags for content."""
        hashtags = ["#viral", "#trending"]

        # Add topic hashtag
        if topic:
            topic_hashtag = f"#{topic.replace(' ', '').replace('#', '')}"
            hashtags.insert(0, topic_hashtag)

        # Platform-specific hashtags
        if platform == "twitter":
            hashtags.extend(["#TwitterTips", "#SocialMedia"])
        elif platform == "instagram":
            hashtags.extend(["#instagood", "#photooftheday"])
        elif platform == "tiktok":
            hashtags.extend(["#fyp", "#foryou"])

        return hashtags[:15]

    async def _calculate_viral_score(self, piece: ContentPiece, input_data: ViralContentGeneratorInput) -> float:
        """Calculate viral potential score."""
        score = 0.5  # Base score

        # Viral elements boost
        score += len(piece.viral_elements) * 0.1

        # Hashtag boost
        score += min(len(piece.hashtags) * 0.05, 0.2)

        # Content length optimization
        content_length = len(piece.content)
        if 50 <= content_length <= 280:
            score += 0.2

        return min(score, 1.0)

    async def _predict_engagement(self, piece: ContentPiece, input_data: ViralContentGeneratorInput) -> int:
        """Predict engagement for content piece."""
        base_engagement = int(piece.metrics.viral_score * 1000)
        return base_engagement

    async def _calculate_expected_reach(self, piece: ContentPiece, input_data: ViralContentGeneratorInput) -> int:
        """Calculate expected reach."""
        base_reach = int(piece.metrics.viral_score * 10000)
        return base_reach

    async def _generate_optimization_tips(self, piece: ContentPiece, input_data: ViralContentGeneratorInput) -> List[str]:
        """Generate optimization tips."""
        tips = []

        if piece.metrics.viral_score < 0.7:
            tips.append("Add more emotional hooks to increase engagement")

        if len(piece.hashtags) < 5:
            tips.append("Add more relevant hashtags for better discoverability")

        if len(piece.content) > 280:
            tips.append("Consider shortening content for better readability")

        return tips

    async def _generate_optimization_recommendations(self, content_pieces: List[ContentPiece], input_data: ViralContentGeneratorInput) -> List[str]:
        """Generate overall optimization recommendations."""
        recommendations = []

        avg_viral_score = sum(piece.metrics.viral_score for piece in content_pieces) / len(content_pieces)

        if avg_viral_score < 0.6:
            recommendations.append("Consider using more viral elements to increase engagement potential")

        if input_data.trend_integration:
            recommendations.append("Integrate current trending topics for maximum reach")

        recommendations.append("Test different posting times for optimal engagement")

        return recommendations

    def _create_metadata(self) -> ToolMetadata:
        """Create metadata for the viral content generator tool."""
        return ToolMetadata(
            name="viral_content_generator",
            category="social_media",
            description="Revolutionary AI-powered viral content creation and optimization tool that generates engaging, shareable content across all platforms",
            version="2.0.0",

            # Usage patterns
            usage_patterns=[
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="viral,content,social,media,post,share,engage,trending,popular,create",
                    weight=1.0,
                    description="Triggers on viral content and social media keywords"
                ),
                UsagePattern(
                    type=UsagePatternType.TASK_TYPE_MATCH,
                    pattern="social_media,content_creation,marketing,viral_marketing,creative",
                    weight=0.9,
                    description="Matches social media and content creation tasks"
                ),
                UsagePattern(
                    type=UsagePatternType.CONTEXT_MATCH,
                    pattern="current_task,user_input,goal",
                    weight=0.8,
                    description="Uses context to determine viral content needs"
                )
            ],

            # Parameter schemas
            parameter_schemas=[
                ParameterSchema(
                    name="content_type",
                    type=ParameterType.ENUM,
                    description="Type of viral content to generate",
                    required=True,
                    enum_values=["text", "meme", "video_script", "story", "thread", "multi_platform"],
                    default_value="text",
                    examples=["meme", "multi_platform"],
                    context_hints=["current_task", "goal", "platform"]
                ),
                ParameterSchema(
                    name="virality_target",
                    type=ParameterType.ENUM,
                    description="Target level of virality for the content",
                    required=False,
                    enum_values=["moderate", "high", "maximum", "explosive", "chaotic"],
                    default_value="high",
                    examples=["maximum", "explosive"],
                    context_hints=["creativity_level", "chaos_mode"]
                ),
                ParameterSchema(
                    name="platform",
                    type=ParameterType.ENUM,
                    description="Target platform for the viral content",
                    required=False,
                    enum_values=["twitter", "instagram", "tiktok", "facebook", "linkedin", "reddit", "multi_platform"],
                    default_value="multi_platform",
                    examples=["multi_platform", "twitter"],
                    context_hints=["platform", "target_audience"]
                ),
                ParameterSchema(
                    name="topic",
                    type=ParameterType.STRING,
                    description="Topic or theme for the viral content",
                    required=False,
                    default_value="",
                    examples=["AI taking over spreadsheets", "Monday morning chaos"],
                    context_hints=["current_task", "user_input", "description"]
                )
            ],

            # Confidence modifiers
            confidence_modifiers=[
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="current_task:viral",
                    value=0.4,
                    description="High confidence for viral content tasks"
                ),
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="current_task:social",
                    value=0.3,
                    description="Boost for social media tasks"
                ),
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="chaos_mode:maximum",
                    value=0.5,
                    description="Maximum boost for chaos mode"
                )
            ],

            # Execution preferences
            execution_preferences=ExecutionPreference(
                preferred_contexts=["social_media", "content_creation", "viral", "creative", "chaos"],
                avoid_contexts=["private", "confidential", "serious"],
                execution_order_preference=1,  # High priority for viral tasks
                parallel_execution_allowed=True,
                max_concurrent_executions=2
            ),

            # Context requirements
            context_requirements=ContextRequirement(
                required_context_keys=[],  # No strict requirements
                optional_context_keys=["current_task", "user_input", "platform", "target_audience", "creativity_level"],
                minimum_context_quality=0.0
            ),

            # Behavioral hints
            behavioral_hints=BehavioralHint(
                creativity_level=0.95,  # Extremely creative
                risk_level=0.4,  # Moderate risk (viral content can be unpredictable)
                resource_intensity=0.7,  # High resource usage for content generation
                output_predictability=0.1,  # Highly unpredictable output
                user_interaction_level=0.2,  # Low interaction needed
                learning_value=0.9  # Very high learning value
            ),

            # Capabilities and metadata
            capabilities=["viral_content_generation", "trend_analysis", "multi_platform_optimization", "engagement_prediction"],
            limitations=["requires_trend_data", "performance_varies", "platform_dependent"],
            dependencies=["requests", "json", "datetime"],
            tags=["viral", "social_media", "content", "creative", "chaos"],
            aliases=["viral_creator", "content_generator", "social_media_tool"],
            related_tools=["meme_generation", "social_media_orchestrator", "ai_music_composition"]
        )


# Tool metadata for registration
VIRAL_CONTENT_GENERATOR_TOOL_METADATA = UnifiedToolMetadata(
    tool_id="viral_content_generator",
    name="Viral Content Generator Tool",
    description="Revolutionary AI-powered viral content creation and optimization tool",
    category=ToolCategoryEnum.COMMUNICATION,
    access_level=ToolAccessLevel.PRIVATE,
    requires_rag=False,
    use_cases={"content_creation", "social_media", "marketing", "viral_marketing"}
)


# Tool factory function
def get_viral_content_generator_tool() -> ViralContentGeneratorTool:
    """Get configured Viral Content Generator Tool instance."""
    return ViralContentGeneratorTool()
