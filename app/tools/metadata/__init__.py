"""
Tool Metadata System - Configuration-Driven Tool Architecture

This module provides a complete metadata-driven system for tools that eliminates
hardcoded logic and enables fully configuration-driven agent behavior.
"""

from .tool_metadata import (
    ToolMetadata,
    ParameterSchema,
    ParameterType,
    UsagePattern,
    UsagePatternType,
    ConfidenceModifier,
    ConfidenceModifierType,
    ExecutionPreference,
    ContextRequirement,
    BehavioralHint
)

from .metadata_registry import ToolMetadataRegistry
from .parameter_generator import ParameterGenerator, ContextMatcher
from .metadata_interfaces import MetadataCapableToolMixin, ParameterGeneratorInterface

# Global registry instance
_global_registry = None

def get_global_registry() -> ToolMetadataRegistry:
    """Get the global tool metadata registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolMetadataRegistry()
    return _global_registry

__all__ = [
    'ToolMetadata',
    'ParameterSchema',
    'ParameterType',
    'UsagePattern',
    'UsagePatternType',
    'ConfidenceModifier',
    'ConfidenceModifierType',
    'ExecutionPreference',
    'ContextRequirement',
    'BehavioralHint',
    'ToolMetadataRegistry',
    'ParameterGenerator',
    'ContextMatcher',
    'MetadataCapableToolMixin',
    'ParameterGeneratorInterface',
    'get_global_registry'
]
