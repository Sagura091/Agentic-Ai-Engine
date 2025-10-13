"""
Base Universal Tool

Base class for all Revolutionary Universal Tools.
Provides common functionality, patterns, and integration with the system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, List
from datetime import datetime
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.tools.unified_tool_repository import (
    ToolMetadata as UnifiedToolMetadata,
    ToolCategory,
    ToolAccessLevel,
)
from app.tools.metadata import (
    ToolMetadata as MetadataToolMetadata,
    ParameterSchema,
    ParameterType,
    UsagePattern,
    UsagePatternType,
    ConfidenceModifier,
    ConfidenceModifierType,
    ExecutionPreference,
    ContextRequirement,
    BehavioralHint,
)

from .error_handlers import UniversalToolError, ValidationError
from .validators import UniversalToolValidator
from .utils import cleanup_temp_files

logger = get_logger()


class BaseUniversalTool(BaseTool, ABC):
    """
    Base class for all Revolutionary Universal Tools.

    Provides:
    - Common initialization and setup
    - Error handling patterns
    - Logging integration
    - Validation utilities
    - Async operation support
    - Integration with UnifiedToolRepository
    - Metadata management
    - Cleanup and resource management
    """

    # Tool identification (override in subclasses)
    tool_id: str = "base_universal_tool"
    tool_version: str = "1.0.0"
    tool_category: ToolCategory = ToolCategory.PRODUCTIVITY

    # Tool configuration
    requires_rag: bool = False
    access_level: ToolAccessLevel = ToolAccessLevel.PUBLIC

    # Internal state (not Pydantic fields)
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **kwargs):
        """Initialize the Universal Tool."""
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'validator', UniversalToolValidator())
        object.__setattr__(self, '_initialized_at', datetime.now())
        object.__setattr__(self, '_execution_count', 0)
        object.__setattr__(self, '_error_count', 0)

        logger.info(
            "Universal Tool initialized",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.production.universal.shared.base_universal_tool",
            data={
                "tool_id": self.tool_id,
                "tool_name": self.name,
                "version": self.tool_version,
            }
        )
    
    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """
        Synchronous execution wrapper.
        
        This method wraps the async _arun method for synchronous execution.
        Override _arun in subclasses, not this method.
        """
        try:
            # Run async method in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new task
                import nest_asyncio
                nest_asyncio.apply()
            
            result = loop.run_until_complete(
                self._arun(*args, run_manager=run_manager, **kwargs)
            )
            return result
            
        except Exception as e:
            logger.error(
                "Tool execution failed (sync)",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.base_universal_tool",
                data={
                    "tool_id": self.tool_id,
                    "error_type": type(e).__name__,
                },
                error=e
            )
            raise
    
    async def _arun(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """
        Asynchronous execution wrapper.

        Override this method in subclasses to implement tool logic.
        This wrapper provides:
        - Execution tracking
        - Error handling
        - Logging
        - Cleanup
        """
        # Increment execution count
        object.__setattr__(self, '_execution_count', self._execution_count + 1)
        start_time = datetime.now()

        try:
            logger.info(
                "Tool execution started",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.base_universal_tool",
                data={
                    "tool_id": self.tool_id,
                    "execution_number": self._execution_count,
                }
            )

            # Call the actual implementation
            result = await self._execute(*args, **kwargs)

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "Tool execution completed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.base_universal_tool",
                data={
                    "tool_id": self.tool_id,
                    "execution_number": self._execution_count,
                    "execution_time": execution_time,
                }
            )

            return result

        except UniversalToolError as e:
            # Already logged by error class
            object.__setattr__(self, '_error_count', self._error_count + 1)
            raise

        except Exception as e:
            object.__setattr__(self, '_error_count', self._error_count + 1)
            logger.error(
                "Tool execution failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.base_universal_tool",
                data={
                    "tool_id": self.tool_id,
                    "execution_number": self._execution_count,
                    "error_type": type(e).__name__,
                },
                error=e
            )
            raise UniversalToolError(
                f"Tool execution failed: {str(e)}",
                original_exception=e,
            )

        finally:
            # Cleanup temporary files
            try:
                cleanup_temp_files()
            except Exception as e:
                logger.warning(
                    "Failed to cleanup temporary files",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.production.universal.shared.base_universal_tool",
                    data={
                        "tool_id": self.tool_id,
                    },
                    error=e
                )
    
    @abstractmethod
    async def _execute(self, *args, **kwargs) -> str:
        """
        Execute the tool's main logic.
        
        Override this method in subclasses to implement tool functionality.
        
        Returns:
            Tool execution result as string
        
        Raises:
            UniversalToolError: If execution fails
        """
        pass
    
    def get_unified_metadata(self) -> UnifiedToolMetadata:
        """
        Get metadata for UnifiedToolRepository registration.
        
        Returns:
            UnifiedToolMetadata instance
        """
        return UnifiedToolMetadata(
            tool_id=self.tool_id,
            name=self.name,
            description=self.description,
            category=self.tool_category,
            access_level=self.access_level,
            requires_rag=self.requires_rag,
            use_cases=self.get_use_cases(),
        )
    
    @abstractmethod
    def get_use_cases(self) -> set:
        """
        Get use cases for this tool.
        
        Returns:
            Set of use case strings
        """
        pass
    
    def get_advanced_metadata(self) -> MetadataToolMetadata:
        """
        Get advanced metadata for metadata system.
        
        Returns:
            MetadataToolMetadata instance
        """
        return MetadataToolMetadata(
            name=self.name,
            category=self.tool_category.value,
            description=self.description,
            version=self.tool_version,
            usage_patterns=self.get_usage_patterns(),
            parameter_schemas=self.get_parameter_schemas(),
            confidence_modifiers=self.get_confidence_modifiers(),
            execution_preferences=self.get_execution_preferences(),
            context_requirements=self.get_context_requirements(),
            behavioral_hints=self.get_behavioral_hints(),
        )
    
    def get_usage_patterns(self) -> List[UsagePattern]:
        """Get usage patterns for this tool. Override in subclasses."""
        return []
    
    def get_parameter_schemas(self) -> List[ParameterSchema]:
        """Get parameter schemas for this tool. Override in subclasses."""
        return []
    
    def get_confidence_modifiers(self) -> List[ConfidenceModifier]:
        """Get confidence modifiers for this tool. Override in subclasses."""
        return []
    
    def get_execution_preferences(self) -> ExecutionPreference:
        """Get execution preferences for this tool. Override in subclasses."""
        return ExecutionPreference()
    
    def get_context_requirements(self) -> ContextRequirement:
        """Get context requirements for this tool. Override in subclasses."""
        return ContextRequirement()
    
    def get_behavioral_hints(self) -> BehavioralHint:
        """Get behavioral hints for this tool. Override in subclasses."""
        return BehavioralHint()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tool statistics.
        
        Returns:
            Dictionary of statistics
        """
        uptime = (datetime.now() - self._initialized_at).total_seconds()
        
        return {
            "tool_id": self.tool_id,
            "tool_name": self.name,
            "version": self.tool_version,
            "initialized_at": self._initialized_at.isoformat(),
            "uptime_seconds": uptime,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._execution_count - self._error_count) / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            cleanup_temp_files()
        except:
            pass

