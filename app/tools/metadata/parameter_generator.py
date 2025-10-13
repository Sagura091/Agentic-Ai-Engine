"""
Dynamic Parameter Generation System

Provides context-aware parameter generation for tools based on their
metadata schemas and current execution context.
"""

from typing import Dict, List, Any, Optional, Union
import re
import json
from datetime import datetime

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from .tool_metadata import ToolMetadata, ParameterSchema, ParameterType

logger = get_logger()


class ContextMatcher:
    """Utility for matching context patterns and extracting values."""
    
    @staticmethod
    def extract_keywords(text: str, keywords: List[str]) -> List[str]:
        """Extract matching keywords from text."""
        text_lower = text.lower()
        return [kw for kw in keywords if kw.lower() in text_lower]
    
    @staticmethod
    def extract_numbers(text: str) -> List[Union[int, float]]:
        """Extract numbers from text."""
        numbers = []
        # Find integers and floats
        for match in re.finditer(r'-?\d+\.?\d*', text):
            try:
                num_str = match.group()
                if '.' in num_str:
                    numbers.append(float(num_str))
                else:
                    numbers.append(int(num_str))
            except ValueError:
                continue
        return numbers
    
    @staticmethod
    def extract_file_paths(text: str) -> List[str]:
        """Extract potential file paths from text."""
        # Simple file path patterns
        patterns = [
            r'[A-Za-z]:\\[^<>:"|?*\n\r]+',  # Windows paths
            r'/[^<>:"|?*\n\r]+',  # Unix paths
            r'[^/\\<>:"|?*\n\r]+\.[a-zA-Z0-9]+',  # Files with extensions
        ]
        
        paths = []
        for pattern in patterns:
            paths.extend(re.findall(pattern, text))
        return paths
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)
    
    @staticmethod
    def match_enum_values(text: str, enum_values: List[str]) -> Optional[str]:
        """Find the best matching enum value from text."""
        text_lower = text.lower()
        
        # Exact match first
        for value in enum_values:
            if value.lower() in text_lower:
                return value
        
        # Partial match
        for value in enum_values:
            if any(word in text_lower for word in value.lower().split('_')):
                return value
        
        return None


class ParameterGenerator:
    """Generates parameters for tools based on context and metadata."""
    
    def __init__(self):
        self.context_matcher = ContextMatcher()
    
    def generate_parameters(self, metadata: ToolMetadata, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for a tool based on context and metadata."""
        try:
            parameters = {}
            
            for param_schema in metadata.parameter_schemas:
                param_value = self._generate_parameter_value(param_schema, context, metadata)
                if param_value is not None:
                    parameters[param_schema.name] = param_value
                elif param_schema.required and param_schema.default_value is not None:
                    parameters[param_schema.name] = param_schema.default_value
            
            return parameters

        except Exception as e:
            logger.error(
                f"Parameter generation failed for tool {metadata.name}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.parameter_generator",
                data={"tool_name": metadata.name},
                error=e
            )
            return {}

    def _generate_parameter_value(self, param_schema: ParameterSchema, context: Dict[str, Any], metadata: ToolMetadata) -> Any:
        """Generate a single parameter value."""
        try:
            # Check for explicit parameter in context
            if param_schema.name in context:
                return self._validate_and_convert_value(context[param_schema.name], param_schema)

            # Use context hints to find parameter value
            for hint in param_schema.context_hints:
                if hint in context:
                    return self._validate_and_convert_value(context[hint], param_schema)

            # Generate based on parameter type and context
            return self._generate_by_type(param_schema, context, metadata)

        except Exception as e:
            logger.warning(
                f"Parameter value generation failed",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.metadata.parameter_generator",
                data={"param_name": param_schema.name},
                error=e
            )
            return param_schema.default_value
    
    def _generate_by_type(self, param_schema: ParameterSchema, context: Dict[str, Any], metadata: ToolMetadata) -> Any:
        """Generate parameter value based on type and context."""
        context_text = self._get_context_text(context)
        
        if param_schema.type == ParameterType.STRING:
            return self._generate_string_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.INTEGER:
            return self._generate_integer_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.FLOAT:
            return self._generate_float_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.BOOLEAN:
            return self._generate_boolean_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.LIST:
            return self._generate_list_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.DICT:
            return self._generate_dict_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.ENUM:
            return self._generate_enum_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.FILE_PATH:
            return self._generate_file_path_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.URL:
            return self._generate_url_parameter(param_schema, context_text, context)
        
        elif param_schema.type == ParameterType.JSON:
            return self._generate_json_parameter(param_schema, context_text, context)
        
        return param_schema.default_value
    
    def _generate_string_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> str:
        """Generate string parameter value."""
        # Look for relevant text in context
        param_name_lower = param_schema.name.lower()
        
        # Common string parameter patterns
        if any(word in param_name_lower for word in ['prompt', 'query', 'text', 'message']):
            return context.get('current_task', context.get('user_input', context_text[:200]))
        
        elif any(word in param_name_lower for word in ['style', 'mode', 'type']):
            # Extract style/mode from context
            for example in param_schema.examples:
                if str(example).lower() in context_text.lower():
                    return str(example)
        
        elif any(word in param_name_lower for word in ['name', 'title', 'label']):
            # Generate descriptive name
            task = context.get('current_task', '')
            if task:
                return f"Generated from: {task[:50]}"
        
        return param_schema.default_value or ""
    
    def _generate_integer_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> int:
        """Generate integer parameter value."""
        numbers = self.context_matcher.extract_numbers(context_text)
        integers = [n for n in numbers if isinstance(n, int)]
        
        if integers:
            value = integers[0]
            # Apply constraints
            if param_schema.min_value is not None:
                value = max(value, int(param_schema.min_value))
            if param_schema.max_value is not None:
                value = min(value, int(param_schema.max_value))
            return value
        
        return param_schema.default_value or 1
    
    def _generate_float_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> float:
        """Generate float parameter value."""
        numbers = self.context_matcher.extract_numbers(context_text)
        floats = [float(n) for n in numbers]
        
        if floats:
            value = floats[0]
            # Apply constraints
            if param_schema.min_value is not None:
                value = max(value, float(param_schema.min_value))
            if param_schema.max_value is not None:
                value = min(value, float(param_schema.max_value))
            return value
        
        return param_schema.default_value or 0.5
    
    def _generate_boolean_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> bool:
        """Generate boolean parameter value."""
        context_lower = context_text.lower()
        
        # Look for boolean indicators
        true_indicators = ['yes', 'true', 'enable', 'on', 'include', 'with', 'add']
        false_indicators = ['no', 'false', 'disable', 'off', 'exclude', 'without', 'skip']
        
        for indicator in true_indicators:
            if indicator in context_lower:
                return True
        
        for indicator in false_indicators:
            if indicator in context_lower:
                return False
        
        return param_schema.default_value or False
    
    def _generate_list_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> List[Any]:
        """Generate list parameter value."""
        # Try to extract list from context
        if param_schema.examples:
            # Use examples as template
            return param_schema.examples[:3]  # Limit to first 3 examples
        
        # Extract keywords as list
        keywords = re.findall(r'\b\w+\b', context_text)
        return keywords[:5] if keywords else []
    
    def _generate_dict_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dict parameter value."""
        if param_schema.examples and isinstance(param_schema.examples[0], dict):
            return param_schema.examples[0]
        
        # Create basic dict from context
        return {
            "source": "context_generated",
            "timestamp": datetime.now().isoformat(),
            "context_summary": context_text[:100]
        }
    
    def _generate_enum_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> str:
        """Generate enum parameter value."""
        if param_schema.enum_values:
            matched_value = self.context_matcher.match_enum_values(context_text, param_schema.enum_values)
            if matched_value:
                return matched_value
            # Return first enum value as default
            return param_schema.enum_values[0]
        
        return param_schema.default_value or ""
    
    def _generate_file_path_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> str:
        """Generate file path parameter value."""
        paths = self.context_matcher.extract_file_paths(context_text)
        if paths:
            return paths[0]
        
        # Generate default path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"./output/generated_{timestamp}.txt"
    
    def _generate_url_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> str:
        """Generate URL parameter value."""
        urls = self.context_matcher.extract_urls(context_text)
        if urls:
            return urls[0]
        
        return param_schema.default_value or "https://example.com"
    
    def _generate_json_parameter(self, param_schema: ParameterSchema, context_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON parameter value."""
        # Try to parse JSON from context
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, context_text)
        
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Return default JSON structure
        return {"generated": True, "source": "parameter_generator"}
    
    def _validate_and_convert_value(self, value: Any, param_schema: ParameterSchema) -> Any:
        """Validate and convert a value according to parameter schema."""
        try:
            if param_schema.type == ParameterType.STRING:
                return str(value)
            elif param_schema.type == ParameterType.INTEGER:
                return int(float(value))
            elif param_schema.type == ParameterType.FLOAT:
                return float(value)
            elif param_schema.type == ParameterType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ['true', '1', 'yes', 'on']
            elif param_schema.type == ParameterType.LIST:
                if isinstance(value, list):
                    return value
                return [value]
            elif param_schema.type == ParameterType.DICT:
                if isinstance(value, dict):
                    return value
                return {"value": value}
            else:
                return value
                
        except (ValueError, TypeError):
            return param_schema.default_value
    
    def _get_context_text(self, context: Dict[str, Any]) -> str:
        """Extract text content from context for analysis."""
        text_parts = []
        
        # Common text fields
        for key in ['current_task', 'goal', 'user_input', 'description', 'prompt']:
            if key in context:
                text_parts.append(str(context[key]))
        
        return " ".join(text_parts)
