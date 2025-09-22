"""
🎨 REVOLUTIONARY DOCUMENT TEMPLATE ENGINE
=========================================

Advanced template system for dynamic document generation.
Provides intelligent template creation, natural language processing,
and AI-powered content generation.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import structlog
from jinja2 import Template, Environment, BaseLoader
from io import BytesIO

logger = structlog.get_logger(__name__)


class DocumentTemplate:
    """Represents a document template with metadata and content."""
    
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        format: str,
        template_content: str,
        variables: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.format = format
        self.template_content = template_content
        self.variables = variables
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()


class NaturalLanguageProcessor:
    """
    🧠 NATURAL LANGUAGE PROCESSOR
    
    Converts natural language descriptions into structured document templates
    and content. Uses AI to understand intent and generate appropriate content.
    """
    
    def __init__(self):
        self.llm_manager = None
        logger.info("🧠 Natural Language Processor initialized")
    
    async def initialize(self):
        """Initialize AI components."""
        try:
            from app.llm.providers.manager import LLMProviderManager
            self.llm_manager = LLMProviderManager()
            await self.llm_manager.initialize()
            logger.info("✅ NLP processor ready with AI capabilities")
        except Exception as e:
            logger.warning(f"NLP processor running without AI: {str(e)}")
    
    async def parse_natural_language_request(self, request: str) -> Dict[str, Any]:
        """
        Parse natural language request into structured document requirements.
        
        Examples:
        - "Create a business report with sales data and charts"
        - "Generate an invoice template with customer details"
        - "Make a presentation about quarterly results"
        """
        try:
            if self.llm_manager:
                return await self._ai_parse_request(request)
            else:
                return await self._rule_based_parse(request)
        except Exception as e:
            logger.error(f"Natural language parsing failed: {str(e)}")
            return self._create_fallback_structure(request)
    
    async def _ai_parse_request(self, request: str) -> Dict[str, Any]:
        """Use AI to parse natural language request."""
        try:
            prompt = f"""
            Parse this document creation request into a structured format:
            "{request}"
            
            Return a JSON object with:
            - document_type: (report, invoice, presentation, letter, etc.)
            - format: (pdf, docx, xlsx, pptx)
            - sections: list of sections to include
            - variables: list of template variables needed
            - style: document style preferences
            
            Example output:
            {{
                "document_type": "business_report",
                "format": "pdf",
                "sections": [
                    {{"type": "title", "content": "Quarterly Sales Report"}},
                    {{"type": "section", "title": "Executive Summary"}},
                    {{"type": "section", "title": "Sales Data"}},
                    {{"type": "table", "title": "Sales by Region"}}
                ],
                "variables": [
                    {{"name": "quarter", "type": "string", "description": "Quarter period"}},
                    {{"name": "sales_data", "type": "table", "description": "Sales figures"}}
                ],
                "style": "professional"
            }}
            """
            
            # Get AI response
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                provider="ollama",  # Use available provider
                model="llama3.2:latest"  # Fallback model
            )
            
            # Parse JSON response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Extract JSON from response if wrapped in text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return self._create_fallback_structure(request)
                    
        except Exception as e:
            logger.error(f"AI parsing failed: {str(e)}")
            return self._rule_based_parse(request)
    
    async def _rule_based_parse(self, request: str) -> Dict[str, Any]:
        """Rule-based parsing for when AI is not available."""
        request_lower = request.lower()
        
        # Determine document type
        if any(word in request_lower for word in ["report", "analysis", "summary"]):
            document_type = "report"
            format = "pdf"
        elif any(word in request_lower for word in ["invoice", "bill", "receipt"]):
            document_type = "invoice"
            format = "pdf"
        elif any(word in request_lower for word in ["presentation", "slides", "deck"]):
            document_type = "presentation"
            format = "pptx"
        elif any(word in request_lower for word in ["spreadsheet", "data", "calculations"]):
            document_type = "spreadsheet"
            format = "xlsx"
        elif any(word in request_lower for word in ["letter", "memo", "document"]):
            document_type = "letter"
            format = "docx"
        else:
            document_type = "document"
            format = "pdf"
        
        # Create basic structure
        return {
            "document_type": document_type,
            "format": format,
            "sections": [
                {"type": "title", "content": "Document Title"},
                {"type": "section", "title": "Main Content"}
            ],
            "variables": [
                {"name": "title", "type": "string", "description": "Document title"},
                {"name": "content", "type": "text", "description": "Main content"}
            ],
            "style": "professional"
        }
    
    def _create_fallback_structure(self, request: str) -> Dict[str, Any]:
        """Create fallback structure when parsing fails."""
        return {
            "document_type": "document",
            "format": "pdf",
            "sections": [
                {"type": "title", "content": "Generated Document"},
                {"type": "paragraph", "content": f"Content based on: {request}"}
            ],
            "variables": [
                {"name": "title", "type": "string", "description": "Document title"}
            ],
            "style": "simple"
        }


class RevolutionaryTemplateEngine:
    """
    🎨 REVOLUTIONARY TEMPLATE ENGINE
    
    Advanced template system that provides:
    - Natural language to template conversion
    - Dynamic template generation
    - AI-powered content suggestions
    - Multi-format template support
    - Variable extraction and validation
    """
    
    def __init__(self):
        self.templates: Dict[str, DocumentTemplate] = {}
        self.nlp_processor = NaturalLanguageProcessor()
        self.jinja_env = Environment(loader=BaseLoader())
        logger.info("🎨 Revolutionary Template Engine initialized")
    
    async def initialize(self):
        """Initialize the template engine."""
        await self.nlp_processor.initialize()
        logger.info("✅ Template Engine ready!")
    
    async def create_template_from_natural_language(
        self,
        description: str,
        template_name: Optional[str] = None
    ) -> DocumentTemplate:
        """
        Create a document template from natural language description.
        
        Args:
            description: Natural language description of desired document
            template_name: Optional name for the template
        
        Returns:
            DocumentTemplate object ready for use
        """
        try:
            # Parse the natural language request
            parsed_request = await self.nlp_processor.parse_natural_language_request(description)
            
            # Generate template ID
            template_id = f"template_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create template name
            if not template_name:
                template_name = f"{parsed_request['document_type'].title()} Template"
            
            # Generate template content based on format
            template_content = await self._generate_template_content(parsed_request)
            
            # Create template object
            template = DocumentTemplate(
                template_id=template_id,
                name=template_name,
                description=description,
                format=parsed_request["format"],
                template_content=template_content,
                variables=parsed_request["variables"],
                metadata={
                    "document_type": parsed_request["document_type"],
                    "style": parsed_request.get("style", "professional"),
                    "sections": parsed_request["sections"]
                }
            )
            
            # Store template
            self.templates[template_id] = template
            
            logger.info(f"✅ Template created: {template_name} ({template_id})")
            return template
            
        except Exception as e:
            logger.error(f"Template creation failed: {str(e)}")
            raise
    
    async def _generate_template_content(self, parsed_request: Dict[str, Any]) -> str:
        """Generate template content based on parsed request."""
        format = parsed_request["format"]
        sections = parsed_request["sections"]
        
        if format == "docx":
            return self._generate_word_template(sections)
        elif format == "xlsx":
            return self._generate_excel_template(sections)
        elif format == "pptx":
            return self._generate_powerpoint_template(sections)
        elif format == "pdf":
            return self._generate_pdf_template(sections)
        else:
            return self._generate_generic_template(sections)
    
    def _generate_word_template(self, sections: List[Dict[str, Any]]) -> str:
        """Generate Word document template structure."""
        template_parts = []
        
        for section in sections:
            if section["type"] == "title":
                template_parts.append(f"# {{{{ title | default('{section['content']}') }}}}")
            elif section["type"] == "section":
                template_parts.append(f"\n## {{{{ {section['title'].lower().replace(' ', '_')} | default('{section['title']}') }}}}")
                template_parts.append("{{ content }}")
            elif section["type"] == "paragraph":
                template_parts.append(f"\n{section.get('content', '{{ paragraph_content }}')}")
            elif section["type"] == "table":
                template_parts.append(f"\n### {section.get('title', 'Table')}")
                template_parts.append("{{ table_data | table }}")
        
        return "\n".join(template_parts)
    
    def _generate_excel_template(self, sections: List[Dict[str, Any]]) -> str:
        """Generate Excel template structure."""
        template_data = {
            "sheets": []
        }
        
        current_sheet = {
            "name": "Data",
            "data": []
        }
        
        for section in sections:
            if section["type"] == "title":
                current_sheet["data"].append([f"{{{{ title | default('{section['content']}') }}}}"])
            elif section["type"] == "table":
                current_sheet["data"].append([f"{{{{ {section.get('title', 'table').lower().replace(' ', '_')}_headers }}}}"])
                current_sheet["data"].append([f"{{{{ {section.get('title', 'table').lower().replace(' ', '_')}_data }}}}"])
        
        template_data["sheets"].append(current_sheet)
        return json.dumps(template_data, indent=2)
    
    def _generate_powerpoint_template(self, sections: List[Dict[str, Any]]) -> str:
        """Generate PowerPoint template structure."""
        template_data = {
            "slides": []
        }
        
        for i, section in enumerate(sections):
            slide = {
                "slide_number": i + 1,
                "layout": "title_content" if section["type"] == "title" else "content"
            }
            
            if section["type"] == "title":
                slide["title"] = f"{{{{ title | default('{section['content']}') }}}}"
            elif section["type"] == "section":
                slide["title"] = f"{{{{ {section['title'].lower().replace(' ', '_')} | default('{section['title']}') }}}}"
                slide["content"] = "{{ content }}"
            
            template_data["slides"].append(slide)
        
        return json.dumps(template_data, indent=2)
    
    def _generate_pdf_template(self, sections: List[Dict[str, Any]]) -> str:
        """Generate PDF template structure."""
        template_parts = []
        
        for section in sections:
            if section["type"] == "title":
                template_parts.append(f"TITLE: {{{{ title | default('{section['content']}') }}}}")
            elif section["type"] == "section":
                template_parts.append(f"\nSECTION: {{{{ {section['title'].lower().replace(' ', '_')} | default('{section['title']}') }}}}")
                template_parts.append("CONTENT: {{ content }}")
            elif section["type"] == "paragraph":
                template_parts.append(f"\nPARAGRAPH: {section.get('content', '{{ paragraph_content }}')}")
            elif section["type"] == "table":
                template_parts.append(f"\nTABLE: {section.get('title', 'Table')}")
                template_parts.append("TABLE_DATA: {{ table_data }}")
        
        return "\n".join(template_parts)
    
    def _generate_generic_template(self, sections: List[Dict[str, Any]]) -> str:
        """Generate generic template structure."""
        template_parts = []
        
        for section in sections:
            if section["type"] == "title":
                template_parts.append(f"{{{{ title | default('{section['content']}') }}}}")
            elif section["type"] == "section":
                template_parts.append(f"\n{{{{ {section['title'].lower().replace(' ', '_')} | default('{section['title']}') }}}}")
            elif section["type"] == "paragraph":
                template_parts.append(f"\n{section.get('content', '{{ content }}')}")
        
        return "\n".join(template_parts)
    
    async def apply_template(
        self,
        template_id: str,
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply variables to a template and generate document structure.
        
        Args:
            template_id: ID of the template to use
            variables: Dictionary of variables to substitute
        
        Returns:
            Document structure ready for generation
        """
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Create Jinja2 template
            jinja_template = self.jinja_env.from_string(template.template_content)
            
            # Render template with variables
            rendered_content = jinja_template.render(**variables)
            
            # Create document structure
            document_structure = {
                "format": template.format,
                "content": rendered_content,
                "metadata": template.metadata,
                "variables_used": variables
            }
            
            logger.info(f"✅ Template applied: {template.name}")
            return document_structure
            
        except Exception as e:
            logger.error(f"Template application failed: {str(e)}")
            raise
    
    def get_template(self, template_id: str) -> Optional[DocumentTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates."""
        return [
            {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "format": template.format,
                "created_at": template.created_at.isoformat(),
                "variables": template.variables
            }
            for template in self.templates.values()
        ]
    
    async def generate_content_suggestions(
        self,
        document_type: str,
        context: Optional[str] = None
    ) -> List[str]:
        """Generate AI-powered content suggestions for document sections."""
        try:
            if self.nlp_processor.llm_manager:
                prompt = f"""
                Generate content suggestions for a {document_type} document.
                Context: {context or 'General business document'}
                
                Provide 5 specific, actionable content suggestions that would be appropriate
                for this type of document. Return as a JSON list of strings.
                """
                
                response = await self.nlp_processor.llm_manager.generate_response(
                    prompt=prompt,
                    provider="ollama",
                    model="llama3.2:latest"
                )
                
                try:
                    return json.loads(response)
                except:
                    # Fallback to basic suggestions
                    return self._get_basic_suggestions(document_type)
            else:
                return self._get_basic_suggestions(document_type)
                
        except Exception as e:
            logger.error(f"Content suggestion generation failed: {str(e)}")
            return self._get_basic_suggestions(document_type)
    
    def _get_basic_suggestions(self, document_type: str) -> List[str]:
        """Get basic content suggestions when AI is not available."""
        suggestions = {
            "report": [
                "Executive Summary",
                "Key Findings",
                "Data Analysis",
                "Recommendations",
                "Conclusion"
            ],
            "invoice": [
                "Company Information",
                "Client Details",
                "Itemized Services",
                "Payment Terms",
                "Total Amount"
            ],
            "presentation": [
                "Title Slide",
                "Agenda",
                "Main Content",
                "Key Points",
                "Next Steps"
            ],
            "letter": [
                "Header Information",
                "Greeting",
                "Main Message",
                "Call to Action",
                "Closing"
            ]
        }
        
        return suggestions.get(document_type, [
            "Introduction",
            "Main Content",
            "Supporting Details",
            "Summary",
            "Conclusion"
        ])
