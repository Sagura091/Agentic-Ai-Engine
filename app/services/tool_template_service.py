"""
Tool Template Service for Custom Tool Creation.

This service provides pre-built templates and code generation helpers
to make it easy for users to create custom tools.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ToolTemplate(BaseModel):
    """Tool template definition."""
    id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: str = Field(..., description="Tool category")
    complexity: str = Field(..., description="Complexity level")
    template_code: str = Field(..., description="Template code with placeholders")
    placeholders: List[str] = Field(..., description="List of placeholders to fill")
    example_values: Dict[str, str] = Field(..., description="Example values for placeholders")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    tags: List[str] = Field(default_factory=list, description="Template tags")


class ToolTemplateService:
    """Service for managing tool templates and code generation."""
    
    def __init__(self):
        """Initialize the template service."""
        self.templates: Dict[str, ToolTemplate] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in tool templates."""
        
        # Basic Tool Template
        basic_template = ToolTemplate(
            id="basic_tool",
            name="Basic Tool",
            description="Simple tool template for basic operations",
            category="utility",
            complexity="simple",
            template_code='''"""
{TOOL_DESCRIPTION}
"""

from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun


class {TOOL_NAME}Input(BaseModel):
    """Input schema for {TOOL_NAME}."""
    {INPUT_FIELDS}


class {TOOL_NAME}(BaseTool):
    """
    {TOOL_DESCRIPTION}
    """
    
    name: str = "{TOOL_NAME_LOWER}"
    description: str = "{TOOL_DESCRIPTION}"
    args_schema: Type[BaseModel] = {TOOL_NAME}Input
    
    def _run(
        self,
        {RUN_PARAMETERS},
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool."""
        try:
            # TODO: Implement your tool logic here
            {TOOL_IMPLEMENTATION}
            
            return f"Tool executed successfully with parameters: {{{RUN_PARAMETERS}}}"
            
        except Exception as e:
            return f"Tool execution failed: {{str(e)}}"
''',
            placeholders=[
                "TOOL_NAME", "TOOL_NAME_LOWER", "TOOL_DESCRIPTION", 
                "INPUT_FIELDS", "RUN_PARAMETERS", "TOOL_IMPLEMENTATION"
            ],
            example_values={
                "TOOL_NAME": "CalculatorTool",
                "TOOL_NAME_LOWER": "calculator",
                "TOOL_DESCRIPTION": "A simple calculator tool for mathematical operations",
                "INPUT_FIELDS": 'expression: str = Field(..., description="Mathematical expression to evaluate")',
                "RUN_PARAMETERS": "expression: str",
                "TOOL_IMPLEMENTATION": "result = eval(expression)  # Note: Use safe evaluation in production\n            return str(result)"
            },
            dependencies=["pydantic", "langchain"],
            tags=["basic", "utility", "template"]
        )
        
        # API Tool Template
        api_template = ToolTemplate(
            id="api_tool",
            name="API Tool",
            description="Template for tools that make API calls",
            category="communication",
            complexity="moderate",
            template_code='''"""
{TOOL_DESCRIPTION}
"""

import json
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun


class {TOOL_NAME}Input(BaseModel):
    """Input schema for {TOOL_NAME}."""
    {INPUT_FIELDS}


class {TOOL_NAME}(BaseTool):
    """
    {TOOL_DESCRIPTION}
    """
    
    name: str = "{TOOL_NAME_LOWER}"
    description: str = "{TOOL_DESCRIPTION}"
    args_schema: Type[BaseModel] = {TOOL_NAME}Input
    
    def __init__(self):
        super().__init__()
        self.base_url = "{API_BASE_URL}"
        self.headers = {{
            "Content-Type": "application/json",
            {API_HEADERS}
        }}
    
    async def _arun(
        self,
        {RUN_PARAMETERS},
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the API call asynchronously."""
        try:
            from app.http_client import SimpleHTTPClient

            # Prepare API request
            payload = {{
                {API_PAYLOAD}
            }}

            # Make API call using SimpleHTTPClient
            async with SimpleHTTPClient(self.base_url, timeout=30) as client:
                response = await client.post("/{API_ENDPOINT}", body=payload, headers=self.headers)
                response.raise_for_status()

            # Process response
            result = response.json()
            {RESPONSE_PROCESSING}

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"API call failed: {{str(e)}}"
''',
            placeholders=[
                "TOOL_NAME", "TOOL_NAME_LOWER", "TOOL_DESCRIPTION", "INPUT_FIELDS",
                "RUN_PARAMETERS", "API_BASE_URL", "API_HEADERS", "API_ENDPOINT",
                "API_PAYLOAD", "RESPONSE_PROCESSING"
            ],
            example_values={
                "TOOL_NAME": "WeatherTool",
                "TOOL_NAME_LOWER": "weather",
                "TOOL_DESCRIPTION": "Get weather information for a location",
                "INPUT_FIELDS": 'location: str = Field(..., description="Location to get weather for")',
                "RUN_PARAMETERS": "location: str",
                "API_BASE_URL": "https://api.weather.com",
                "API_HEADERS": '"Authorization": "Bearer YOUR_API_KEY"',
                "API_ENDPOINT": "current",
                "API_PAYLOAD": '"location": location',
                "RESPONSE_PROCESSING": "# Extract relevant weather data\n            weather_data = result.get('current', {})"
            },
            dependencies=["pydantic", "langchain", "requests"],
            tags=["api", "communication", "external"]
        )
        
        # Data Processing Tool Template
        data_template = ToolTemplate(
            id="data_processing_tool",
            name="Data Processing Tool",
            description="Template for tools that process and analyze data",
            category="computation",
            complexity="moderate",
            template_code='''"""
{TOOL_DESCRIPTION}
"""

import json
from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun


class {TOOL_NAME}Input(BaseModel):
    """Input schema for {TOOL_NAME}."""
    {INPUT_FIELDS}


class {TOOL_NAME}(BaseTool):
    """
    {TOOL_DESCRIPTION}
    """
    
    name: str = "{TOOL_NAME_LOWER}"
    description: str = "{TOOL_DESCRIPTION}"
    args_schema: Type[BaseModel] = {TOOL_NAME}Input
    
    def _run(
        self,
        {RUN_PARAMETERS},
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Process the data."""
        try:
            # Parse input data
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return "Invalid JSON data provided"
            
            # Process data
            {DATA_PROCESSING_LOGIC}
            
            # Return results
            return json.dumps({{
                "processed_data": processed_data,
                "summary": {{
                    "total_items": len(data) if isinstance(data, list) else 1,
                    "processing_method": "{PROCESSING_METHOD}",
                    "timestamp": datetime.now().isoformat()
                }}
            }}, indent=2)
            
        except Exception as e:
            return f"Data processing failed: {{str(e)}}"
''',
            placeholders=[
                "TOOL_NAME", "TOOL_NAME_LOWER", "TOOL_DESCRIPTION", "INPUT_FIELDS",
                "RUN_PARAMETERS", "DATA_PROCESSING_LOGIC", "PROCESSING_METHOD"
            ],
            example_values={
                "TOOL_NAME": "DataAnalyzerTool",
                "TOOL_NAME_LOWER": "data_analyzer",
                "TOOL_DESCRIPTION": "Analyze and process structured data",
                "INPUT_FIELDS": 'data: str = Field(..., description="JSON data to process")',
                "RUN_PARAMETERS": "data: str",
                "DATA_PROCESSING_LOGIC": "# Example: Calculate statistics\n            if isinstance(data, list):\n                processed_data = {\n                    'count': len(data),\n                    'sample': data[:5] if data else []\n                }\n            else:\n                processed_data = {'single_item': data}",
                "PROCESSING_METHOD": "statistical_analysis"
            },
            dependencies=["pydantic", "langchain", "json"],
            tags=["data", "analysis", "processing"]
        )
        
        # RAG Tool Template
        rag_template = ToolTemplate(
            id="rag_tool",
            name="RAG-Enabled Tool",
            description="Template for tools that use RAG (Retrieval-Augmented Generation)",
            category="rag_enabled",
            complexity="complex",
            template_code='''"""
{TOOL_DESCRIPTION}
"""

from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun


class {TOOL_NAME}Input(BaseModel):
    """Input schema for {TOOL_NAME}."""
    {INPUT_FIELDS}


class {TOOL_NAME}(BaseTool):
    """
    {TOOL_DESCRIPTION}
    
    This tool uses RAG capabilities to enhance its functionality.
    """
    
    name: str = "{TOOL_NAME_LOWER}"
    description: str = "{TOOL_DESCRIPTION}"
    args_schema: Type[BaseModel] = {TOOL_NAME}Input
    
    def __init__(self, rag_system=None):
        super().__init__()
        self.rag_system = rag_system
    
    def _run(
        self,
        {RUN_PARAMETERS},
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the RAG-enabled tool."""
        try:
            # Use RAG system if available
            if self.rag_system:
                # Search knowledge base
                search_results = self.rag_system.search(
                    query=query,
                    collection_name="{COLLECTION_NAME}",
                    limit=5
                )
                
                # Process RAG results
                {RAG_PROCESSING_LOGIC}
                
                return f"RAG-enhanced result: {{enhanced_result}}"
            else:
                # Fallback without RAG
                {FALLBACK_LOGIC}
                
                return f"Basic result: {{basic_result}}"
            
        except Exception as e:
            return f"RAG tool execution failed: {{str(e)}}"
''',
            placeholders=[
                "TOOL_NAME", "TOOL_NAME_LOWER", "TOOL_DESCRIPTION", "INPUT_FIELDS",
                "RUN_PARAMETERS", "COLLECTION_NAME", "RAG_PROCESSING_LOGIC", "FALLBACK_LOGIC"
            ],
            example_values={
                "TOOL_NAME": "KnowledgeSearchTool",
                "TOOL_NAME_LOWER": "knowledge_search",
                "TOOL_DESCRIPTION": "Search and retrieve information from knowledge base",
                "INPUT_FIELDS": 'query: str = Field(..., description="Search query")',
                "RUN_PARAMETERS": "query: str",
                "COLLECTION_NAME": "default",
                "RAG_PROCESSING_LOGIC": "# Process search results\n                context = '\\n'.join([doc.get('content', '') for doc in search_results])\n                enhanced_result = f'Found {len(search_results)} relevant documents: {context[:500]}...'",
                "FALLBACK_LOGIC": "# Basic search without RAG\n                basic_result = f'Searched for: {query} (RAG not available)'"
            },
            dependencies=["pydantic", "langchain"],
            tags=["rag", "knowledge", "search"]
        )
        
        # Store templates
        self.templates = {
            "basic_tool": basic_template,
            "api_tool": api_template,
            "data_processing_tool": data_template,
            "rag_tool": rag_template
        }
        
        logger.info("Loaded built-in tool templates", count=len(self.templates))
    
    def get_template(self, template_id: str) -> Optional[ToolTemplate]:
        """Get a specific template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[str] = None) -> List[ToolTemplate]:
        """List all available templates, optionally filtered by category."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return templates
    
    def generate_tool_code(
        self, 
        template_id: str, 
        values: Dict[str, str],
        custom_modifications: Optional[str] = None
    ) -> str:
        """
        Generate tool code from template with provided values.
        
        Args:
            template_id: Template to use
            values: Values to fill placeholders
            custom_modifications: Optional custom code modifications
            
        Returns:
            Generated tool code
        """
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Start with template code
        code = template.template_code
        
        # Fill placeholders
        for placeholder in template.placeholders:
            placeholder_key = f"{{{placeholder}}}"
            if placeholder in values:
                code = code.replace(placeholder_key, values[placeholder])
            elif placeholder in template.example_values:
                # Use example value if not provided
                code = code.replace(placeholder_key, template.example_values[placeholder])
            else:
                # Leave placeholder for manual completion
                code = code.replace(placeholder_key, f"# TODO: Replace {placeholder}")
        
        # Add custom modifications if provided
        if custom_modifications:
            code += f"\n\n# Custom modifications:\n{custom_modifications}"
        
        return code
    
    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """Get detailed information about a template."""
        template = self.get_template(template_id)
        if not template:
            return {}
        
        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "complexity": template.complexity,
            "placeholders": template.placeholders,
            "example_values": template.example_values,
            "dependencies": template.dependencies,
            "tags": template.tags,
            "code_preview": template.template_code[:500] + "..." if len(template.template_code) > 500 else template.template_code
        }
    
    def validate_template_values(self, template_id: str, values: Dict[str, str]) -> Dict[str, Any]:
        """Validate that provided values are sufficient for template generation."""
        template = self.get_template(template_id)
        if not template:
            return {"valid": False, "error": f"Template {template_id} not found"}
        
        missing_placeholders = []
        provided_placeholders = []
        
        for placeholder in template.placeholders:
            if placeholder in values:
                provided_placeholders.append(placeholder)
            elif placeholder not in template.example_values:
                missing_placeholders.append(placeholder)
        
        return {
            "valid": len(missing_placeholders) == 0,
            "missing_placeholders": missing_placeholders,
            "provided_placeholders": provided_placeholders,
            "total_placeholders": len(template.placeholders),
            "completion_percentage": (len(provided_placeholders) / len(template.placeholders)) * 100
        }


# Global service instance
tool_template_service = ToolTemplateService()
