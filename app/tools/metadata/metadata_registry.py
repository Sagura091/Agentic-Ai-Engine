"""
Tool Metadata Registry

Centralized registry for managing all tool metadata and providing
metadata-driven tool discovery and selection capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from .tool_metadata import ToolMetadata, UsagePatternType, ConfidenceModifierType

logger = get_logger()


class ToolMetadataRegistry:
    """Centralized registry for tool metadata."""
    
    def __init__(self):
        self._metadata: Dict[str, ToolMetadata] = {}
        self._category_index: Dict[str, List[str]] = defaultdict(list)
        self._tag_index: Dict[str, List[str]] = defaultdict(list)
        self._capability_index: Dict[str, List[str]] = defaultdict(list)
        self._pattern_index: Dict[UsagePatternType, List[str]] = defaultdict(list)
    
    def register_tool_metadata(self, metadata: ToolMetadata) -> None:
        """Register metadata for a tool."""
        try:
            self._metadata[metadata.name] = metadata
            self._update_indices(metadata)
            logger.info(
                f"Registered metadata for tool: {metadata.name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                data={"tool_name": metadata.name}
            )

        except Exception as e:
            logger.error(
                f"Failed to register metadata for tool {metadata.name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                data={"tool_name": metadata.name},
                error=e
            )
            raise

    def register_tool(self, tool) -> None:
        """Register a tool by extracting its metadata."""
        try:
            # Check if tool has metadata capability
            if hasattr(tool, '_create_metadata'):
                metadata = tool._create_metadata()
                self.register_tool_metadata(metadata)
            else:
                tool_name = getattr(tool, 'name', 'unknown')
                logger.warn(
                    f"Tool {tool_name} does not have metadata capability",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.metadata.metadata_registry",
                    data={"tool_name": tool_name}
                )

        except Exception as e:
            tool_name = getattr(tool, 'name', 'unknown')
            logger.error(
                f"Failed to register tool {tool_name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                data={"tool_name": tool_name},
                error=e
            )
            raise
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool."""
        return self._metadata.get(tool_name)
    
    def get_all_tool_metadata(self) -> Dict[str, ToolMetadata]:
        """Get all registered tool metadata."""
        return self._metadata.copy()
    
    def find_tools_by_category(self, category: str) -> List[ToolMetadata]:
        """Find all tools in a specific category."""
        tool_names = self._category_index.get(category, [])
        return [self._metadata[name] for name in tool_names if name in self._metadata]
    
    def find_tools_by_tag(self, tag: str) -> List[ToolMetadata]:
        """Find all tools with a specific tag."""
        tool_names = self._tag_index.get(tag, [])
        return [self._metadata[name] for name in tool_names if name in self._metadata]
    
    def find_tools_by_capability(self, capability: str) -> List[ToolMetadata]:
        """Find all tools with a specific capability."""
        tool_names = self._capability_index.get(capability, [])
        return [self._metadata[name] for name in tool_names if name in self._metadata]
    
    def find_tools_by_usage_pattern_type(self, pattern_type: UsagePatternType) -> List[ToolMetadata]:
        """Find all tools with a specific usage pattern type."""
        tool_names = self._pattern_index.get(pattern_type, [])
        return [self._metadata[name] for name in tool_names if name in self._metadata]
    
    def find_best_tools_for_context(self, context: Dict[str, Any], max_results: int = 10) -> List[Tuple[ToolMetadata, float]]:
        """Find the best tools for a given context, ranked by match score."""
        try:
            tool_scores = []
            
            for tool_name, metadata in self._metadata.items():
                match_score = metadata.matches_context(context)
                if match_score > 0:
                    tool_scores.append((metadata, match_score))
            
            # Sort by score descending and return top results
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            return tool_scores[:max_results]

        except Exception as e:
            logger.error(
                "Context-based tool search failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                error=e
            )
            return []

    def calculate_tool_confidence(self, tool_name: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for a tool in a given context."""
        try:
            metadata = self.get_tool_metadata(tool_name)
            if not metadata:
                return 0.0

            # Start with base match score
            base_confidence = metadata.matches_context(context)

            # Apply confidence modifiers
            for modifier in metadata.confidence_modifiers:
                if self._modifier_condition_met(modifier.condition, context):
                    if modifier.type == ConfidenceModifierType.BOOST:
                        base_confidence += modifier.value
                    elif modifier.type == ConfidenceModifierType.PENALTY:
                        base_confidence -= modifier.value
                    elif modifier.type == ConfidenceModifierType.MULTIPLIER:
                        base_confidence *= modifier.value
                    elif modifier.type == ConfidenceModifierType.THRESHOLD:
                        base_confidence = max(base_confidence, modifier.value)

            return max(0.0, min(1.0, base_confidence))

        except Exception as e:
            logger.warn(
                f"Confidence calculation failed for tool {tool_name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                data={"tool_name": tool_name},
                error=e
            )
            return 0.0
    
    def get_tools_by_execution_preference(self, context: Dict[str, Any]) -> List[Tuple[ToolMetadata, int]]:
        """Get tools sorted by execution order preference."""
        try:
            tools_with_preference = []
            
            for metadata in self._metadata.values():
                # Check if tool is suitable for context
                if self._tool_suitable_for_context(metadata, context):
                    preference_score = metadata.execution_preferences.execution_order_preference
                    tools_with_preference.append((metadata, preference_score))
            
            # Sort by preference (lower numbers first)
            tools_with_preference.sort(key=lambda x: x[1])
            return tools_with_preference

        except Exception as e:
            logger.error(
                "Execution preference sorting failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                error=e
            )
            return []
    
    def validate_tool_context_requirements(self, tool_name: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that context meets tool requirements."""
        try:
            metadata = self.get_tool_metadata(tool_name)
            if not metadata:
                return False, [f"Tool {tool_name} not found"]
            
            missing_requirements = []
            requirements = metadata.context_requirements
            
            # Check required context keys
            for required_key in requirements.required_context_keys:
                if required_key not in context:
                    missing_requirements.append(f"Missing required context key: {required_key}")
            
            # Run context validators
            for validator in requirements.context_validators:
                try:
                    if not validator(context):
                        missing_requirements.append(f"Context validation failed: {validator.__name__}")
                except Exception as e:
                    missing_requirements.append(f"Context validator error: {str(e)}")
            
            return len(missing_requirements) == 0, missing_requirements

        except Exception as e:
            logger.error(
                f"Context validation failed for tool {tool_name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                data={"tool_name": tool_name},
                error=e
            )
            return False, [f"Validation error: {str(e)}"]
    
    def _update_indices(self, metadata: ToolMetadata) -> None:
        """Update all indices with new metadata."""
        tool_name = metadata.name
        
        # Update category index
        self._category_index[metadata.category].append(tool_name)
        
        # Update tag index
        for tag in metadata.tags:
            self._tag_index[tag].append(tool_name)
        
        # Update capability index
        for capability in metadata.capabilities:
            self._capability_index[capability].append(tool_name)
        
        # Update pattern index
        for pattern in metadata.usage_patterns:
            self._pattern_index[pattern.type].append(tool_name)
    
    def _modifier_condition_met(self, condition: str, context: Dict[str, Any]) -> bool:
        """Check if a modifier condition is met."""
        try:
            # Simple condition evaluation - can be enhanced with more complex logic
            if ":" in condition:
                key, value = condition.split(":", 1)
                context_value = str(context.get(key, "")).lower()
                return value.lower() in context_value
            else:
                # Check if condition exists as a key in context
                return condition in context

        except Exception as e:
            logger.warn(
                "Condition evaluation failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                data={"condition": condition},
                error=e
            )
            return False

    def _tool_suitable_for_context(self, metadata: ToolMetadata, context: Dict[str, Any]) -> bool:
        """Check if a tool is suitable for the given context."""
        try:
            # Check if context matches preferred contexts
            preferences = metadata.execution_preferences

            if preferences.preferred_contexts:
                context_str = str(context).lower()
                if not any(pref.lower() in context_str for pref in preferences.preferred_contexts):
                    return False

            # Check if context matches avoided contexts
            if preferences.avoid_contexts:
                context_str = str(context).lower()
                if any(avoid.lower() in context_str for avoid in preferences.avoid_contexts):
                    return False

            return True

        except Exception as e:
            logger.warn(
                "Context suitability check failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_registry",
                data={"tool_name": metadata.name},
                error=e
            )
            return True  # Default to suitable if check fails


# Global registry instance
_global_registry = ToolMetadataRegistry()


def get_global_registry() -> ToolMetadataRegistry:
    """Get the global tool metadata registry."""
    return _global_registry


def register_tool_metadata(metadata: ToolMetadata) -> None:
    """Register metadata in the global registry."""
    _global_registry.register_tool_metadata(metadata)
