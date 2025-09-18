"""
Simple Calculator Tool - Direct LangChain BaseTool implementation.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Type
from datetime import datetime

import structlog
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

logger = structlog.get_logger(__name__)


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(..., description="Mathematical expression to calculate (e.g., '100 + 50 * 2')")
    precision: int = Field(default=2, description="Number of decimal places for result")


class SimpleCalculatorTool(BaseTool):
    """Simple calculator tool for mathematical operations."""
    
    name: str = "calculator"
    description: str = "Perform mathematical calculations including arithmetic, percentages, and basic functions"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def __init__(self):
        super().__init__()
        self.usage_count = 0
        self.execution_history = []
    
    def _run(
        self,
        expression: str,
        precision: int = 2,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute calculator operation synchronously."""
        return asyncio.run(self._arun(expression, precision, run_manager))
    
    async def _arun(
        self,
        expression: str,
        precision: int = 2,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute calculator operation asynchronously."""
        start_time = time.time()
        
        try:
            # Update usage statistics
            self.usage_count += 1
            
            # Log calculation start
            logger.info(
                "Calculator tool execution started",
                tool_name=self.name,
                expression=expression,
                precision=precision,
                usage_count=self.usage_count
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
            
            # Record execution
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "expression": expression,
                "sanitized_expression": sanitized_expression,
                "result": formatted_result,
                "precision": precision,
                "execution_time": execution_time,
                "success": True
            }
            self.execution_history.append(execution_record)
            
            # Log successful calculation
            logger.info(
                "Calculator tool execution completed successfully",
                tool_name=self.name,
                expression=expression,
                result=formatted_result,
                execution_time=execution_time,
                usage_count=self.usage_count
            )
            
            return f"Calculation: {expression} = {formatted_result}"
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed execution
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "expression": expression,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
            self.execution_history.append(execution_record)
            
            # Log calculation error
            logger.error(
                "Calculator tool execution failed",
                tool_name=self.name,
                expression=expression,
                error=str(e),
                execution_time=execution_time,
                usage_count=self.usage_count
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
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        successful_executions = sum(1 for exec in self.execution_history if exec["success"])
        failed_executions = sum(1 for exec in self.execution_history if not exec["success"])
        
        return {
            "tool_name": self.name,
            "total_usage": self.usage_count,
            "total_executions": len(self.execution_history),
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / max(1, len(self.execution_history)),
            "recent_executions": self.execution_history[-5:] if self.execution_history else []
        }


# Create tool instance for registration
simple_calculator_tool = SimpleCalculatorTool()
