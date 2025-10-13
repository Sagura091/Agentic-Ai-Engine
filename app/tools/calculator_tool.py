"""
Simple Calculator Tool for Agentic AI Systems.

This tool provides basic mathematical operations for agents to perform calculations
during problem-solving tasks. It demonstrates simple tool usage patterns.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Type
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.tools.metadata import MetadataCapableToolMixin, ToolMetadata as MetadataToolMetadata, ParameterSchema, ParameterType, UsagePattern, UsagePatternType, ConfidenceModifier, ConfidenceModifierType

logger = get_logger()


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(..., description="Mathematical expression to calculate (e.g., '100 + 50 * 2')")
    precision: int = Field(default=2, description="Number of decimal places for result")


class CalculatorTool(BaseTool, MetadataCapableToolMixin):
    """Simple calculator tool for mathematical operations."""

    name: str = "calculator"
    description: str = "Perform mathematical calculations including arithmetic, percentages, and basic functions"
    args_schema: Type[BaseModel] = CalculatorInput
    tool_id: str = "calculator"

    def __init__(self):
        super().__init__()
        # Simple metadata tracking (not Pydantic fields)
        self._execution_history = []
        self._usage_count = 0
        self._last_used = None
        self._average_execution_time = 0.0
        self._success_rate = 1.0
        self._last_updated = datetime.utcnow()

    @property
    def execution_history(self):
        return self._execution_history

    @property
    def usage_count(self):
        return self._usage_count

    @property
    def last_used(self):
        return self._last_used

    @property
    def average_execution_time(self):
        return self._average_execution_time

    @property
    def success_rate(self):
        return self._success_rate

    @property
    def last_updated(self):
        return self._last_updated
    
    def _run(
        self,
        expression: str,
        precision: int = 2,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute calculator operation synchronously."""
        return asyncio.run(self._arun_implementation(expression, precision))
    
    async def _arun_implementation(
        self,
        expression: str,
        precision: int = 2,
        **kwargs
    ) -> str:
        """Execute calculator operation asynchronously."""
        start_time = time.time()
        
        try:
            # Update usage statistics
            self._usage_count += 1
            self._last_used = datetime.utcnow()
            
            # Log calculation start
            logger.info(
                "Calculator tool execution started",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.calculator_tool",
                data={
                    "tool_name": self.name,
                    "expression": expression,
                    "precision": precision,
                    "usage_count": self.usage_count
                }
            )
            
            # Sanitize and validate expression
            sanitized_expression = self._sanitize_expression(expression)
            
            # Perform calculation
            result = self._calculate(sanitized_expression)
            
            # Format result with specified precision
            if isinstance(result, float):
                formatted_result = round(result, precision)
            else:
                formatted_result = result
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, success=True)
            
            # Record execution
            execution_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "expression": expression,
                "sanitized_expression": sanitized_expression,
                "result": formatted_result,
                "precision": precision,
                "execution_time": execution_time,
                "success": True
            }
            self._execution_history.append(execution_record)

            # Log successful calculation
            logger.info(
                "Calculator tool execution completed successfully",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.calculator_tool",
                data={
                    "tool_name": self.name,
                    "expression": expression,
                    "result": formatted_result,
                    "execution_time": execution_time,
                    "usage_count": self.usage_count
                }
            )

            return f"Calculation: {expression} = {formatted_result}"

        except Exception as e:
            execution_time = time.time() - start_time

            # Update performance metrics for failure
            self._update_performance_metrics(execution_time, success=False)

            # Record failed execution
            execution_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "expression": expression,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
            self._execution_history.append(execution_record)

            # Log calculation error
            logger.error(
                "Calculator tool execution failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.calculator_tool",
                data={
                    "tool_name": self.name,
                    "expression": expression,
                    "execution_time": execution_time,
                    "usage_count": self.usage_count
                },
                error=e
            )
            
            return f"Calculation error: {str(e)}"
    
    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize mathematical expression for safe evaluation."""
        # Remove whitespace
        sanitized = expression.strip()
        
        # Replace common mathematical symbols
        replacements = {
            "×": "*",
            "÷": "/",
            "^": "**",
            "π": "3.14159265359",
            "e": "2.71828182846"
        }
        
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        
        # Basic validation - only allow safe characters
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in sanitized):
            raise ValueError(f"Invalid characters in expression: {expression}")
        
        return sanitized
    
    def _calculate(self, expression: str) -> float:
        """Safely calculate mathematical expression."""
        try:
            # Handle percentage calculations
            if "%" in expression:
                return self._handle_percentage(expression)
            
            # Use eval for basic arithmetic (sanitized input)
            # Note: In production, consider using a proper math parser
            result = eval(expression)
            
            if not isinstance(result, (int, float)):
                raise ValueError(f"Invalid calculation result type: {type(result)}")
            
            return float(result)
            
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {str(e)}")
    
    def _handle_percentage(self, expression: str) -> float:
        """Handle percentage calculations."""
        if " of " in expression:
            # Handle "X% of Y" format
            parts = expression.split(" of ")
            if len(parts) == 2:
                percent_part = parts[0].replace("%", "").strip()
                base_part = parts[1].strip()
                
                percent = float(percent_part)
                base = float(base_part)
                
                return (percent / 100) * base
        
        # Handle simple percentage conversion
        if expression.endswith("%"):
            number = float(expression.replace("%", "").strip())
            return number / 100
        
        raise ValueError(f"Invalid percentage expression: {expression}")
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update tool performance metrics."""
        # Update average execution time
        total_executions = self._usage_count
        current_avg = self._average_execution_time

        new_avg = ((current_avg * (total_executions - 1)) + execution_time) / total_executions
        self._average_execution_time = new_avg

        # Update success rate
        if total_executions == 1:
            self._success_rate = 1.0 if success else 0.0
        else:
            successful_executions = int(self._success_rate * (total_executions - 1))
            if success:
                successful_executions += 1

            self._success_rate = successful_executions / total_executions

        # Update last updated timestamp
        self._last_updated = datetime.utcnow()
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        return {
            "tool_name": self.name,
            "total_usage": self.usage_count,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "total_executions": len(self.execution_history),
            "successful_executions": sum(1 for exec in self.execution_history if exec["success"]),
            "failed_executions": sum(1 for exec in self.execution_history if not exec["success"]),
            "recent_executions": self.execution_history[-5:] if self.execution_history else []
        }

    def _create_metadata(self) -> MetadataToolMetadata:
        """Create metadata for calculator tool."""
        return MetadataToolMetadata(
            name="calculator",
            description="Simple calculator tool for mathematical operations with creative chaos capabilities",
            category="utility",
            usage_patterns=[
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="chaos,creative,chaotic,random,42,1337",
                    weight=0.9,
                    context_requirements=["chaos_mode", "creative_task"],
                    description="Triggers on creative chaos math tasks"
                ),
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="calculate,math,compute,add,subtract,multiply,divide",
                    weight=0.8,
                    context_requirements=["calculation_task"],
                    description="Matches basic calculation tasks"
                ),
                UsagePattern(
                    type=UsagePatternType.KEYWORD_MATCH,
                    pattern="expression,formula,equation,evaluate",
                    weight=0.85,
                    context_requirements=["mathematical_expression"],
                    description="Matches expression evaluation tasks"
                )
            ],
            confidence_modifiers=[
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="chaos_mode",
                    value=0.15,
                    description="Boost confidence for chaotic mathematical creativity"
                ),
                ConfidenceModifier(
                    type=ConfidenceModifierType.BOOST,
                    condition="calculation_task",
                    value=0.1,
                    description="Boost confidence for mathematical calculations"
                )
            ],
            parameter_schemas=[
                ParameterSchema(
                    name="expression",
                    type=ParameterType.STRING,
                    description="Mathematical expression to calculate",
                    required=True,
                    default_value="42 * 1337 / 69"
                ),
                ParameterSchema(
                    name="precision",
                    type=ParameterType.INTEGER,
                    description="Number of decimal places for result",
                    required=False,
                    default_value=4
                )
            ]
        )


# Create tool instance for registration
calculator_tool = CalculatorTool()

# Tool metadata for unified repository registration
from app.tools.unified_tool_repository import ToolMetadata as UnifiedToolMetadata, ToolCategory, ToolAccessLevel

CALCULATOR_TOOL_METADATA = UnifiedToolMetadata(
    tool_id="calculator",
    name="Calculator Tool",
    description="Simple calculator tool for mathematical operations with creative chaos capabilities",
    category=ToolCategory.COMPUTATION,
    access_level=ToolAccessLevel.PUBLIC,
    requires_rag=False,
    use_cases={"math", "calculation", "utility", "chaos", "arithmetic", "percentages"}
)
