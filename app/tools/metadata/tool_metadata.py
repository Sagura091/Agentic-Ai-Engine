"""
Core Tool Metadata System

Defines the complete metadata structure for tools to self-describe their
capabilities, usage patterns, and behavioral characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory

logger = get_logger()


class ParameterType(Enum):
    """Parameter types for tool parameters."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ENUM = "enum"
    FILE_PATH = "file_path"
    URL = "url"
    JSON = "json"


class UsagePatternType(Enum):
    """Types of usage patterns for tools."""
    KEYWORD_MATCH = "keyword_match"
    CONTEXT_MATCH = "context_match"
    TASK_TYPE_MATCH = "task_type_match"
    SEMANTIC_MATCH = "semantic_match"
    REGEX_MATCH = "regex_match"


class ConfidenceModifierType(Enum):
    """Types of confidence modifiers."""
    BOOST = "boost"
    PENALTY = "penalty"
    MULTIPLIER = "multiplier"
    THRESHOLD = "threshold"


@dataclass
class ParameterSchema:
    """Schema definition for a tool parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default_value: Any = None
    enum_values: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    examples: List[Any] = field(default_factory=list)
    context_hints: List[str] = field(default_factory=list)


@dataclass
class UsagePattern:
    """Pattern that indicates when a tool should be used."""
    type: UsagePatternType
    pattern: str
    weight: float = 1.0
    context_requirements: List[str] = field(default_factory=list)
    exclusion_patterns: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ConfidenceModifier:
    """Modifier that adjusts tool confidence based on context."""
    type: ConfidenceModifierType
    condition: str
    value: float
    description: str = ""


@dataclass
class ExecutionPreference:
    """Preferences for when and how a tool should be executed."""
    preferred_contexts: List[str] = field(default_factory=list)
    avoid_contexts: List[str] = field(default_factory=list)
    execution_order_preference: int = 0  # Lower numbers execute first
    parallel_execution_allowed: bool = True
    requires_user_confirmation: bool = False
    max_concurrent_executions: int = 1


@dataclass
class ContextRequirement:
    """Requirements for context that a tool needs to function properly."""
    required_context_keys: List[str] = field(default_factory=list)
    optional_context_keys: List[str] = field(default_factory=list)
    context_validators: List[Callable] = field(default_factory=list)
    minimum_context_quality: float = 0.0


@dataclass
class BehavioralHint:
    """Hints about tool behavior for decision making."""
    creativity_level: float = 0.5  # 0.0 = deterministic, 1.0 = highly creative
    risk_level: float = 0.5  # 0.0 = safe, 1.0 = high risk
    resource_intensity: float = 0.5  # 0.0 = lightweight, 1.0 = resource intensive
    output_predictability: float = 0.5  # 0.0 = unpredictable, 1.0 = predictable
    user_interaction_level: float = 0.0  # 0.0 = no interaction, 1.0 = high interaction
    learning_value: float = 0.5  # 0.0 = no learning, 1.0 = high learning value


@dataclass
class ToolMetadata:
    """Complete metadata description for a tool."""
    name: str
    category: str
    description: str
    version: str = "1.0.0"
    
    # Core functionality
    usage_patterns: List[UsagePattern] = field(default_factory=list)
    parameter_schemas: List[ParameterSchema] = field(default_factory=list)
    
    # Decision making support
    confidence_modifiers: List[ConfidenceModifier] = field(default_factory=list)
    execution_preferences: ExecutionPreference = field(default_factory=ExecutionPreference)
    context_requirements: ContextRequirement = field(default_factory=ContextRequirement)
    behavioral_hints: BehavioralHint = field(default_factory=BehavioralHint)
    
    # Capabilities and constraints
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Integration metadata
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    
    def get_parameter_schema(self, param_name: str) -> Optional[ParameterSchema]:
        """Get parameter schema by name."""
        for schema in self.parameter_schemas:
            if schema.name == param_name:
                return schema
        return None
    
    def get_usage_patterns_by_type(self, pattern_type: UsagePatternType) -> List[UsagePattern]:
        """Get usage patterns of a specific type."""
        return [p for p in self.usage_patterns if p.type == pattern_type]
    
    def get_confidence_modifiers_by_type(self, modifier_type: ConfidenceModifierType) -> List[ConfidenceModifier]:
        """Get confidence modifiers of a specific type."""
        return [m for m in self.confidence_modifiers if m.type == modifier_type]
    
    def matches_context(self, context: Dict[str, Any]) -> float:
        """Calculate how well this tool matches the given context."""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for pattern in self.usage_patterns:
                pattern_score = self._evaluate_usage_pattern(pattern, context)
                total_score += pattern_score * pattern.weight
                total_weight += pattern.weight
            
            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.warn(
                f"Context matching failed for tool {self.name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.tool_metadata",
                data={"tool_name": self.name},
                error=e
            )
            return 0.0

    def _evaluate_usage_pattern(self, pattern: UsagePattern, context: Dict[str, Any]) -> float:
        """Evaluate a single usage pattern against context."""
        try:
            if pattern.type == UsagePatternType.KEYWORD_MATCH:
                return self._evaluate_keyword_pattern(pattern, context)
            elif pattern.type == UsagePatternType.CONTEXT_MATCH:
                return self._evaluate_context_pattern(pattern, context)
            elif pattern.type == UsagePatternType.TASK_TYPE_MATCH:
                return self._evaluate_task_type_pattern(pattern, context)
            else:
                return 0.0

        except Exception as e:
            logger.warn(
                "Pattern evaluation failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.tool_metadata",
                data={"pattern": pattern.pattern},
                error=e
            )
            return 0.0
    
    def _evaluate_keyword_pattern(self, pattern: UsagePattern, context: Dict[str, Any]) -> float:
        """Evaluate keyword-based usage pattern."""
        text_to_search = ""
        
        # Combine relevant context text
        for key in ["current_task", "goal", "description", "user_input"]:
            if key in context:
                text_to_search += f" {str(context[key])}"
        
        text_to_search = text_to_search.lower()
        keywords = pattern.pattern.lower().split(",")
        
        matches = sum(1 for keyword in keywords if keyword.strip() in text_to_search)
        return matches / len(keywords) if keywords else 0.0
    
    def _evaluate_context_pattern(self, pattern: UsagePattern, context: Dict[str, Any]) -> float:
        """Evaluate context-based usage pattern."""
        required_keys = pattern.pattern.split(",")
        matches = sum(1 for key in required_keys if key.strip() in context)
        return matches / len(required_keys) if required_keys else 0.0
    
    def _evaluate_task_type_pattern(self, pattern: UsagePattern, context: Dict[str, Any]) -> float:
        """Evaluate task type-based usage pattern."""
        task_type = context.get("task_type", "").lower()
        pattern_types = [t.strip().lower() for t in pattern.pattern.split(",")]
        return 1.0 if task_type in pattern_types else 0.0
