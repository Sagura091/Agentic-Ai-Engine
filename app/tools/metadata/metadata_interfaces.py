"""
Metadata System Interfaces

Defines interfaces and mixins for tools to implement metadata-driven behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from .tool_metadata import ToolMetadata
from .parameter_generator import ParameterGenerator

logger = get_logger()


class ParameterGeneratorInterface(ABC):
    """Interface for tools that can generate their own parameters."""
    
    @abstractmethod
    def generate_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for this tool based on context."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parameters for this tool."""
        pass


class MetadataCapableToolMixin:
    """Mixin that provides metadata capabilities to tools."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata: Optional[ToolMetadata] = None
        self._parameter_generator = ParameterGenerator()
    
    @property
    def metadata(self) -> Optional[ToolMetadata]:
        """Get tool metadata."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata: ToolMetadata) -> None:
        """Set tool metadata."""
        self._metadata = metadata
    
    def get_tool_metadata(self) -> ToolMetadata:
        """Get metadata for this tool. Must be implemented by subclasses."""
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata
    
    @abstractmethod
    def _create_metadata(self) -> ToolMetadata:
        """Create metadata for this tool. Must be implemented by subclasses."""
        pass
    
    def generate_parameters_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters using the metadata system."""
        try:
            metadata = self.get_tool_metadata()
            return self._parameter_generator.generate_parameters(metadata, context)
        except Exception as e:
            logger.error(
                f"Parameter generation failed for {self.__class__.__name__}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_interfaces",
                data={"class_name": self.__class__.__name__},
                error=e
            )
            return {}

    def validate_context_requirements(self, context: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate that context meets tool requirements."""
        try:
            metadata = self.get_tool_metadata()
            requirements = metadata.context_requirements
            missing_requirements = []

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
                f"Context validation failed for {self.__class__.__name__}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_interfaces",
                data={"class_name": self.__class__.__name__},
                error=e
            )
            return False, [f"Validation error: {str(e)}"]

    def calculate_confidence_for_context(self, context: Dict[str, Any]) -> float:
        """Calculate confidence score for this tool in the given context."""
        try:
            metadata = self.get_tool_metadata()
            return metadata.matches_context(context)
        except Exception as e:
            logger.warning(
                f"Confidence calculation failed for {self.__class__.__name__}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_interfaces",
                data={"class_name": self.__class__.__name__},
                error=e
            )
            return 0.0
    
    def is_suitable_for_context(self, context: Dict[str, Any]) -> bool:
        """Check if this tool is suitable for the given context."""
        try:
            metadata = self.get_tool_metadata()
            preferences = metadata.execution_preferences
            
            # Check preferred contexts
            if preferences.preferred_contexts:
                context_str = str(context).lower()
                if not any(pref.lower() in context_str for pref in preferences.preferred_contexts):
                    return False
            
            # Check avoided contexts
            if preferences.avoid_contexts:
                context_str = str(context).lower()
                if any(avoid.lower() in context_str for avoid in preferences.avoid_contexts):
                    return False
            
            return True

        except Exception as e:
            logger.warning(
                f"Context suitability check failed for {self.__class__.__name__}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_interfaces",
                data={"class_name": self.__class__.__name__},
                error=e
            )
            return True  # Default to suitable if check fails


class ConfigurableToolInterface(ABC):
    """Interface for tools that can be configured via metadata."""
    
    @abstractmethod
    def configure_from_metadata(self, metadata: ToolMetadata) -> None:
        """Configure tool behavior based on metadata."""
        pass
    
    @abstractmethod
    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for this tool."""
        pass


class ContextAwareToolInterface(ABC):
    """Interface for tools that are aware of their execution context."""
    
    @abstractmethod
    def update_context(self, context: Dict[str, Any]) -> None:
        """Update tool with new context information."""
        pass
    
    @abstractmethod
    def get_context_requirements(self) -> Dict[str, Any]:
        """Get the context requirements for this tool."""
        pass


class AdaptiveToolInterface(ABC):
    """Interface for tools that can adapt their behavior based on usage patterns."""
    
    @abstractmethod
    def learn_from_execution(self, parameters: Dict[str, Any], result: Any, feedback: Dict[str, Any]) -> None:
        """Learn from tool execution to improve future parameter generation."""
        pass
    
    @abstractmethod
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get metrics about tool adaptation and learning."""
        pass


class MetadataAwareBaseTool(MetadataCapableToolMixin):
    """Base class for tools that implement the metadata system."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register metadata when tool is created
        self._register_metadata()
    
    def _register_metadata(self) -> None:
        """Register this tool's metadata with the global registry."""
        try:
            from .metadata_registry import register_tool_metadata
            metadata = self.get_tool_metadata()
            register_tool_metadata(metadata)
            logger.info(
                f"Registered metadata for tool: {metadata.name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_interfaces",
                data={"tool_name": metadata.name}
            )
        except Exception as e:
            logger.error(
                f"Failed to register metadata for {self.__class__.__name__}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_interfaces",
                data={"class_name": self.__class__.__name__},
                error=e
            )
    
    def execute_with_context(self, context: Dict[str, Any], **kwargs) -> Any:
        """Execute tool with context-aware parameter generation."""
        try:
            # Validate context requirements
            is_valid, errors = self.validate_context_requirements(context)
            if not is_valid:
                raise ValueError(f"Context validation failed: {', '.join(errors)}")
            
            # Generate parameters from context
            generated_params = self.generate_parameters_from_context(context)
            
            # Merge with provided kwargs (kwargs take precedence)
            final_params = {**generated_params, **kwargs}
            
            # Execute the tool
            return self._execute_with_params(final_params)

        except Exception as e:
            logger.error(
                f"Context-aware execution failed for {self.__class__.__name__}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.metadata_interfaces",
                data={"class_name": self.__class__.__name__},
                error=e
            )
            raise
    
    @abstractmethod
    def _execute_with_params(self, parameters: Dict[str, Any]) -> Any:
        """Execute the tool with the given parameters. Must be implemented by subclasses."""
        pass
