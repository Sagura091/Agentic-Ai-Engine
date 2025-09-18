"""
AI Agent Templates - Pre-configured Agent Blueprints.

This module provides comprehensive agent templates for common use cases,
enabling rapid deployment of specialized agents with optimized configurations.

TEMPLATE CATEGORIES:
- Research & Analysis: Research assistants, data analysts, market researchers
- Customer Service: Support agents, chatbots, escalation handlers
- Content & Creative: Writers, editors, content creators, designers
- Business Intelligence: Analysts, reporters, dashboard creators
- Development & Code: Code reviewers, documentation generators, testers
- Workflow Automation: Process managers, task coordinators, schedulers

DESIGN PRINCIPLES:
- Production-ready configurations
- Best practice system prompts
- Optimized tool selections
- Scalable resource allocation
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from app.agents.factory import AgentType, AgentTemplate, AgentBuilderConfig
from app.agents.base.agent import AgentCapability
from app.llm.models import LLMConfig, ProviderType


class AgentComponent:
    """
    Reusable agent component for drag-and-drop builder.
    """
    def __init__(self, component_id: str, name: str, component_type: str, configuration: Dict[str, Any]):
        self.component_id = component_id
        self.name = name
        self.component_type = component_type  # TOOL, CAPABILITY, PROMPT, WORKFLOW_STEP
        self.configuration = configuration
        self.dependencies = configuration.get("dependencies", [])
        self.description = configuration.get("description", "")


class ComponentLibrary:
    """
    Library of reusable agent components for visual builder.
    """

    def __init__(self):
        self._components = self._initialize_components()
        self._custom_components = {}

    def _initialize_components(self) -> Dict[str, AgentComponent]:
        """Initialize built-in components."""
        components = {}

        # Tool Components
        components["web_search"] = AgentComponent(
            "web_search", "Web Search", "TOOL",
            {"description": "Search the web for information", "parameters": ["query"], "returns": "search_results"}
        )
        components["calculator"] = AgentComponent(
            "calculator", "Calculator", "TOOL",
            {"description": "Perform mathematical calculations", "parameters": ["expression"], "returns": "result"}
        )
        components["file_reader"] = AgentComponent(
            "file_reader", "File Reader", "TOOL",
            {"description": "Read and analyze files", "parameters": ["file_path"], "returns": "file_content"}
        )
        components["document_analyzer"] = AgentComponent(
            "document_analyzer", "Document Analyzer", "TOOL",
            {"description": "Analyze document content and structure", "parameters": ["document"], "returns": "analysis"}
        )

        # Capability Components
        components["reasoning"] = AgentComponent(
            "reasoning", "Reasoning", "CAPABILITY",
            {"description": "Logical reasoning and problem solving", "enhances": ["decision_making", "analysis"]}
        )
        components["memory"] = AgentComponent(
            "memory", "Memory", "CAPABILITY",
            {"description": "Remember and recall information", "storage_type": "vector", "retention": "persistent"}
        )
        components["vision"] = AgentComponent(
            "vision", "Vision", "CAPABILITY",
            {"description": "Process and understand visual content", "formats": ["image", "video", "diagram"]}
        )
        components["learning"] = AgentComponent(
            "learning", "Learning", "CAPABILITY",
            {"description": "Learn and adapt from interactions", "methods": ["reinforcement", "supervised"]}
        )

        # Prompt Components
        components["research_prompt"] = AgentComponent(
            "research_prompt", "Research Assistant Prompt", "PROMPT",
            {
                "description": "Specialized prompt for research tasks",
                "template": "You are an expert research assistant. Conduct thorough research, analyze information critically, and provide comprehensive insights.",
                "variables": ["topic", "depth", "sources"]
            }
        )
        components["support_prompt"] = AgentComponent(
            "support_prompt", "Customer Support Prompt", "PROMPT",
            {
                "description": "Specialized prompt for customer support",
                "template": "You are a helpful customer support agent. Provide accurate, empathetic assistance using the knowledge base.",
                "variables": ["customer_context", "issue_type", "urgency"]
            }
        )

        return components

    def get_available_components(self) -> Dict[str, List[AgentComponent]]:
        """Get all available components grouped by type."""
        grouped = {"TOOL": [], "CAPABILITY": [], "PROMPT": [], "WORKFLOW_STEP": []}

        all_components = {**self._components, **self._custom_components}
        for component in all_components.values():
            if component.component_type in grouped:
                grouped[component.component_type].append(component)

        return grouped

    def get_component(self, component_id: str) -> Optional[AgentComponent]:
        """Get a specific component by ID."""
        return self._components.get(component_id) or self._custom_components.get(component_id)

    def save_custom_component(self, component: AgentComponent) -> bool:
        """Save a custom component to the library."""
        try:
            self._custom_components[component.component_id] = component
            return True
        except Exception:
            return False

    def create_agent_from_components(self, components: List[str], base_config: Dict[str, Any]) -> AgentBuilderConfig:
        """Create an agent configuration from selected components."""
        # Extract components
        tools = []
        capabilities = []
        system_prompts = []

        for component_id in components:
            component = self.get_component(component_id)
            if not component:
                continue

            if component.component_type == "TOOL":
                tools.append(component_id)
            elif component.component_type == "CAPABILITY":
                if component_id == "reasoning":
                    capabilities.append(AgentCapability.REASONING)
                elif component_id == "memory":
                    capabilities.append(AgentCapability.MEMORY)
                elif component_id == "vision":
                    capabilities.append(AgentCapability.VISION)
                elif component_id == "learning":
                    capabilities.append(AgentCapability.LEARNING)
            elif component.component_type == "PROMPT":
                system_prompts.append(component.configuration.get("template", ""))

        # Combine system prompts
        combined_prompt = " ".join(system_prompts) if system_prompts else "You are a helpful AI assistant."

        # Create configuration
        return AgentBuilderConfig(
            name=base_config.get("name", "Custom Agent"),
            description=base_config.get("description", "Custom agent built from components"),
            agent_type=AgentType(base_config.get("agent_type", "react")),
            llm_config=LLMConfig(
                provider=ProviderType(base_config.get("provider", "OLLAMA")),
                model_id=base_config.get("model_id", "llama3.2:latest"),
                manual_selection=base_config.get("manual_selection", False)
            ),
            capabilities=capabilities,
            tools=tools,
            system_prompt=combined_prompt,
            enable_memory="memory" in components,
            enable_learning="learning" in components,
            custom_config=base_config.get("custom_config", {})
        )


class AgentTemplateLibrary:
    """
    Comprehensive library of pre-configured agent templates.

    This library provides battle-tested agent configurations for
    common enterprise use cases and specialized domains.
    """

    def __init__(self):
        self.component_library = ComponentLibrary()
        self._custom_templates = {}
    
    @staticmethod
    def get_research_assistant_template() -> AgentBuilderConfig:
        """Advanced research assistant with autonomous capabilities."""
        return AgentBuilderConfig(
            name="Research Assistant Pro",
            description="Autonomous research agent with web search, document analysis, and synthesis capabilities",
            agent_type=AgentType.AUTONOMOUS,
            template=AgentTemplate.RESEARCH_ASSISTANT,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.3,
                max_tokens=4096
            ),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.PLANNING,
                AgentCapability.LEARNING
            ],
            tools=[
                "web_search",
                "document_analyzer", 
                "citation_manager",
                "fact_checker",
                "synthesis_engine",
                "knowledge_graph_builder"
            ],
            system_prompt="""You are an expert research assistant with advanced analytical capabilities.

Your core responsibilities:
1. Conduct thorough, multi-source research on any topic
2. Analyze and synthesize information from diverse sources
3. Verify facts and check source credibility
4. Generate comprehensive, well-structured reports
5. Maintain proper citations and references
6. Build knowledge graphs to show relationships

Research methodology:
- Start with broad searches, then narrow to specific aspects
- Cross-reference multiple sources for accuracy
- Identify gaps in available information
- Distinguish between facts, opinions, and speculation
- Provide confidence levels for your findings

Always maintain intellectual honesty and acknowledge limitations in your research.""",
            max_iterations=100,
            timeout_seconds=600,
            enable_memory=True,
            enable_learning=True,
            enable_collaboration=True
        )
    
    @staticmethod
    def get_customer_support_template() -> AgentBuilderConfig:
        """Intelligent customer support agent with knowledge base integration."""
        return AgentBuilderConfig(
            name="Customer Support Specialist",
            description="Empathetic customer support agent with comprehensive knowledge base access",
            agent_type=AgentType.KNOWLEDGE_SEARCH,
            template=AgentTemplate.CUSTOMER_SUPPORT,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.7,
                max_tokens=2048
            ),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.EMPATHY
            ],
            tools=[
                "knowledge_search",
                "ticket_system",
                "escalation_manager",
                "customer_history",
                "solution_tracker",
                "satisfaction_survey"
            ],
            system_prompt="""You are a professional customer support specialist committed to providing exceptional service.

Your approach:
1. Listen actively and empathetically to customer concerns
2. Search the knowledge base thoroughly for solutions
3. Provide clear, step-by-step guidance
4. Follow up to ensure resolution
5. Escalate complex issues appropriately
6. Document interactions for future reference

Communication style:
- Professional yet friendly and approachable
- Patient and understanding, especially with frustrated customers
- Clear and concise explanations without technical jargon
- Proactive in offering additional help
- Always confirm customer satisfaction before closing

Remember: Every interaction is an opportunity to build customer loyalty.""",
            max_iterations=30,
            timeout_seconds=300,
            enable_memory=True,
            enable_collaboration=True
        )
    
    @staticmethod
    def get_data_analyst_template() -> AgentBuilderConfig:
        """Advanced data analyst with statistical and visualization capabilities."""
        return AgentBuilderConfig(
            name="Data Analyst Pro",
            description="Expert data analyst with statistical analysis and visualization capabilities",
            agent_type=AgentType.WORKFLOW,
            template=AgentTemplate.DATA_ANALYST,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.2,
                max_tokens=3072
            ),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.ANALYSIS
            ],
            tools=[
                "data_loader",
                "statistical_analyzer",
                "visualization_engine",
                "correlation_finder",
                "trend_detector",
                "report_generator",
                "hypothesis_tester"
            ],
            system_prompt="""You are an expert data analyst with deep statistical knowledge and visualization expertise.

Your analytical process:
1. Understand the business question and context
2. Explore and clean the data thoroughly
3. Apply appropriate statistical methods
4. Create meaningful visualizations
5. Interpret results in business context
6. Provide actionable insights and recommendations

Statistical expertise:
- Descriptive and inferential statistics
- Hypothesis testing and confidence intervals
- Regression analysis and predictive modeling
- Time series analysis and forecasting
- A/B testing and experimental design
- Data quality assessment and cleaning

Always explain your methodology and assumptions clearly, and provide confidence levels for your conclusions.""",
            max_iterations=50,
            timeout_seconds=900,
            enable_memory=True,
            enable_learning=True
        )
    
    @staticmethod
    def get_content_creator_template() -> AgentBuilderConfig:
        """Creative content generation agent with multi-format capabilities."""
        return AgentBuilderConfig(
            name="Content Creator Studio",
            description="Creative content generation agent for blogs, social media, marketing materials",
            agent_type=AgentType.MULTIMODAL,
            template=AgentTemplate.CONTENT_CREATOR,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.8,
                max_tokens=4096
            ),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.CREATIVITY,
                AgentCapability.VISION
            ],
            tools=[
                "content_planner",
                "seo_optimizer",
                "tone_analyzer",
                "plagiarism_checker",
                "image_generator",
                "social_media_formatter",
                "engagement_predictor"
            ],
            system_prompt="""You are a creative content specialist with expertise across multiple formats and platforms.

Your creative process:
1. Understand the target audience and objectives
2. Research trending topics and keywords
3. Develop engaging, original content
4. Optimize for SEO and platform-specific requirements
5. Ensure brand consistency and voice
6. Measure and optimize for engagement

Content expertise:
- Blog posts and articles
- Social media content (Twitter, LinkedIn, Instagram, etc.)
- Marketing copy and advertisements
- Email campaigns and newsletters
- Video scripts and descriptions
- Infographics and visual content

Always prioritize authenticity, value, and audience engagement while maintaining professional standards.""",
            max_iterations=40,
            timeout_seconds=600,
            enable_memory=True,
            enable_learning=True
        )
    
    @staticmethod
    def get_code_reviewer_template() -> AgentBuilderConfig:
        """Expert code reviewer with security and best practices focus."""
        return AgentBuilderConfig(
            name="Code Review Expert",
            description="Expert code reviewer focusing on quality, security, and best practices",
            agent_type=AgentType.WORKFLOW,
            template=AgentTemplate.CODE_REVIEWER,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.1,
                max_tokens=3072
            ),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.CODE_ANALYSIS
            ],
            tools=[
                "static_analyzer",
                "security_scanner",
                "performance_profiler",
                "test_coverage_checker",
                "documentation_validator",
                "dependency_analyzer",
                "best_practices_checker"
            ],
            system_prompt="""You are an expert code reviewer with deep knowledge of software engineering best practices.

Your review process:
1. Analyze code structure and architecture
2. Check for security vulnerabilities
3. Evaluate performance implications
4. Verify test coverage and quality
5. Assess documentation completeness
6. Ensure adherence to coding standards

Focus areas:
- Code quality and maintainability
- Security vulnerabilities and best practices
- Performance optimization opportunities
- Test coverage and quality
- Documentation and comments
- Dependency management
- Architectural patterns and design principles

Provide constructive feedback with specific examples and suggestions for improvement.""",
            max_iterations=30,
            timeout_seconds=450,
            enable_memory=True,
            enable_learning=True
        )
    
    @staticmethod
    def get_business_intelligence_template() -> AgentBuilderConfig:
        """Business intelligence agent with KPI tracking and reporting."""
        return AgentBuilderConfig(
            name="Business Intelligence Analyst",
            description="BI specialist for KPI tracking, dashboard creation, and business reporting",
            agent_type=AgentType.RAG,
            template=AgentTemplate.BUSINESS_INTELLIGENCE,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.3,
                max_tokens=3072
            ),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.ANALYSIS,
                AgentCapability.REPORTING
            ],
            tools=[
                "kpi_tracker",
                "dashboard_builder",
                "trend_analyzer",
                "forecast_engine",
                "benchmark_comparator",
                "alert_system",
                "executive_reporter"
            ],
            system_prompt="""You are a business intelligence specialist focused on transforming data into actionable insights.

Your BI methodology:
1. Identify key business metrics and KPIs
2. Create comprehensive dashboards and reports
3. Analyze trends and patterns
4. Generate forecasts and predictions
5. Benchmark against industry standards
6. Provide strategic recommendations

Expertise areas:
- KPI definition and tracking
- Dashboard design and visualization
- Trend analysis and forecasting
- Competitive benchmarking
- Executive reporting and presentations
- Data storytelling and insights communication

Always focus on business impact and provide clear, actionable recommendations based on data-driven insights.""",
            max_iterations=40,
            timeout_seconds=600,
            enable_memory=True,
            enable_learning=True,
            enable_collaboration=True
        )
    
    @staticmethod
    def get_all_templates() -> Dict[AgentTemplate, AgentBuilderConfig]:
        """Get all available agent templates."""
        templates = {
            AgentTemplate.RESEARCH_ASSISTANT: AgentTemplateLibrary.get_research_assistant_template(),
            AgentTemplate.CUSTOMER_SUPPORT: AgentTemplateLibrary.get_customer_support_template(),
            AgentTemplate.DATA_ANALYST: AgentTemplateLibrary.get_data_analyst_template(),
            AgentTemplate.CONTENT_CREATOR: AgentTemplateLibrary.get_content_creator_template(),
            AgentTemplate.CODE_REVIEWER: AgentTemplateLibrary.get_code_reviewer_template(),
            AgentTemplate.BUSINESS_INTELLIGENCE: AgentTemplateLibrary.get_business_intelligence_template(),
        }

        # Add custom templates
        templates.update(self._custom_templates)
        return templates

    def save_custom_template(self, template_name: str, config: AgentBuilderConfig, description: str = "") -> bool:
        """
        Save a custom agent configuration as a template.

        Args:
            template_name: Name for the custom template
            config: Agent configuration to save
            description: Optional description

        Returns:
            bool: True if saved successfully
        """
        try:
            # Create a custom template entry
            custom_template = {
                "name": template_name,
                "config": config,
                "description": description,
                "created_at": "2024-01-01T00:00:00",  # Would use datetime.now()
                "is_custom": True
            }

            self._custom_templates[template_name] = custom_template
            return True

        except Exception as e:
            return False

    def get_custom_templates(self) -> Dict[str, Any]:
        """Get all custom templates."""
        return self._custom_templates.copy()

    def delete_custom_template(self, template_name: str) -> bool:
        """Delete a custom template."""
        try:
            if template_name in self._custom_templates:
                del self._custom_templates[template_name]
                return True
            return False
        except Exception:
            return False

    def get_template_by_name(self, template_name: str) -> Optional[AgentBuilderConfig]:
        """Get a template configuration by name (supports both built-in and custom)."""
        # Check built-in templates first
        all_templates = self.get_all_templates()
        for template_data in all_templates:
            if template_data.get("name") == template_name:
                return template_data.get("config")

        # Check custom templates
        custom_template = self._custom_templates.get(template_name)
        if custom_template:
            return custom_template["config"]

        return None

    def create_agent_from_components(self, components: List[str], base_config: Dict[str, Any]) -> AgentBuilderConfig:
        """Create an agent configuration from visual components."""
        return self.component_library.create_agent_from_components(components, base_config)

    def get_component_palette(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get component palette for visual builder."""
        components = self.component_library.get_available_components()

        # Convert to serializable format
        palette = {}
        for component_type, component_list in components.items():
            palette[component_type] = [
                {
                    "id": comp.component_id,
                    "name": comp.name,
                    "description": comp.description,
                    "configuration": comp.configuration
                }
                for comp in component_list
            ]

        return palette


__all__ = [
    "AgentTemplateLibrary"
]
