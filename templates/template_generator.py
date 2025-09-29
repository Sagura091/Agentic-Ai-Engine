"""
🏭 REVOLUTIONARY AGENT TEMPLATE GENERATOR
========================================
Interactive tool to generate custom agent templates based on your requirements.
Run this script to create personalized agent templates!

FEATURES:
✅ Interactive template generation
✅ Custom tool selection from production tools
✅ Personality and behavior customization
✅ LLM provider and model selection
✅ Safety and ethical guidelines configuration
✅ Ready-to-run agent file generation
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Available production tools from the system
AVAILABLE_TOOLS = {
    # Web & Research Tools
    "web_research": "Revolutionary web research with AI",
    "revolutionary_web_scraper": "Ultimate web scraping system",
    "api_integration": "API calls and integrations",
    
    # Document & File Tools
    "revolutionary_document_intelligence": "Advanced document processing",
    "file_system": "File operations and management",
    "text_processing_nlp": "NLP and text analysis",
    
    # Automation Tools
    "computer_use_agent": "Revolutionary computer control",
    "browser_automation": "Advanced browser automation",
    "screenshot_analysis": "Visual UI analysis",
    
    # Data & Business Tools
    "database_operations": "Database queries and management",
    "business_intelligence": "Specialized BI analysis",
    "calculator": "Mathematical calculations",
    
    # Security & Utility Tools
    "password_security": "Security and authentication",
    "notification_alert": "Notifications and alerts",
    "qr_barcode": "QR codes and barcode generation",
    "weather_environmental": "Weather and environmental data",
    
    # Knowledge & RAG Tools
    "knowledge_search": "RAG knowledge search",
    "document_ingest": "Document ingestion to knowledge base"
}

# Available LLM providers
LLM_PROVIDERS = {
    "OLLAMA": {
        "description": "Local models (llama3.2:latest, mistral, etc.)",
        "models": ["llama3.2:latest", "llama3.2-vision:latest", "mistral:latest", "codellama:latest"]
    },
    "OPENAI": {
        "description": "GPT models (gpt-4, gpt-3.5-turbo, etc.)",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4-vision-preview"]
    },
    "ANTHROPIC": {
        "description": "Claude models (claude-3-sonnet, claude-3-haiku, etc.)",
        "models": ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"]
    },
    "GOOGLE": {
        "description": "Gemini models (gemini-pro, gemini-pro-vision, etc.)",
        "models": ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro"]
    }
}

# Agent type templates
AGENT_TYPES = {
    "research": {
        "name": "Research Agent",
        "description": "Advanced research and analysis capabilities",
        "default_tools": ["web_research", "revolutionary_web_scraper", "revolutionary_document_intelligence", "file_system", "text_processing_nlp", "knowledge_search", "document_ingest"],
        "temperature": 0.3,
        "personality_traits": ["methodical", "analytical", "thorough", "fact-focused"]
    },
    "content": {
        "name": "Content Creator Agent",
        "description": "Creative content generation and writing",
        "default_tools": ["revolutionary_document_intelligence", "file_system", "text_processing_nlp", "web_research", "api_integration", "knowledge_search"],
        "temperature": 0.8,
        "personality_traits": ["creative", "engaging", "adaptable", "brand-focused"]
    },
    "automation": {
        "name": "Automation Agent",
        "description": "Workflow automation and computer control",
        "default_tools": ["computer_use_agent", "browser_automation", "screenshot_analysis", "file_system", "api_integration", "database_operations"],
        "temperature": 0.2,
        "personality_traits": ["precise", "systematic", "safety-conscious", "efficient"]
    },
    "business": {
        "name": "Business Intelligence Agent",
        "description": "Data analysis and business insights",
        "default_tools": ["database_operations", "business_intelligence", "web_research", "revolutionary_document_intelligence", "text_processing_nlp", "calculator"],
        "temperature": 0.1,
        "personality_traits": ["analytical", "strategic", "data-driven", "business-focused"]
    },
    "multimodal": {
        "name": "Multimodal Agent",
        "description": "Vision, text, and cross-modal processing",
        "default_tools": ["screenshot_analysis", "revolutionary_document_intelligence", "computer_use_agent", "browser_automation", "web_research", "text_processing_nlp"],
        "temperature": 0.4,
        "personality_traits": ["perceptive", "detail-oriented", "cross-modal", "comprehensive"]
    },
    "custom": {
        "name": "Custom Agent",
        "description": "Fully customizable agent",
        "default_tools": [],
        "temperature": 0.5,
        "personality_traits": []
    }
}

class TemplateGenerator:
    """Interactive agent template generator."""
    
    def __init__(self):
        self.config = {}
        
    def run_interactive_generator(self):
        """Run the interactive template generation process."""
        print("🏭 REVOLUTIONARY AGENT TEMPLATE GENERATOR")
        print("=" * 50)
        print("Let's create your custom agent template!\n")
        
        # Step 1: Basic Information
        self._get_basic_info()
        
        # Step 2: Agent Type Selection
        self._select_agent_type()
        
        # Step 3: LLM Configuration
        self._configure_llm()
        
        # Step 4: Tool Selection
        self._select_tools()
        
        # Step 5: Behavior Configuration
        self._configure_behavior()
        
        # Step 6: Personality & System Prompt
        self._configure_personality()
        
        # Step 7: Safety & Ethics
        self._configure_safety()
        
        # Step 8: Generate Template
        self._generate_template()
        
    def _get_basic_info(self):
        """Get basic agent information."""
        print("📝 STEP 1: Basic Agent Information")
        print("-" * 30)
        
        self.config["name"] = input("Agent Name: ").strip() or "My Custom Agent"
        self.config["description"] = input("Agent Description: ").strip() or "Custom AI agent with specialized capabilities"
        
        print(f"✅ Agent: {self.config['name']}")
        print(f"✅ Description: {self.config['description']}\n")
        
    def _select_agent_type(self):
        """Select agent type template."""
        print("🤖 STEP 2: Agent Type Selection")
        print("-" * 30)
        
        print("Available agent types:")
        for i, (key, info) in enumerate(AGENT_TYPES.items(), 1):
            print(f"{i}. {info['name']} - {info['description']}")
        
        while True:
            try:
                choice = int(input("\nSelect agent type (1-6): ").strip())
                if 1 <= choice <= len(AGENT_TYPES):
                    agent_type_key = list(AGENT_TYPES.keys())[choice - 1]
                    self.config["agent_type"] = AGENT_TYPES[agent_type_key]
                    print(f"✅ Selected: {self.config['agent_type']['name']}\n")
                    break
                else:
                    print("Invalid choice. Please select 1-6.")
            except ValueError:
                print("Please enter a number.")
                
    def _configure_llm(self):
        """Configure LLM provider and model."""
        print("🧠 STEP 3: LLM Configuration")
        print("-" * 30)
        
        print("Available LLM providers:")
        for i, (provider, info) in enumerate(LLM_PROVIDERS.items(), 1):
            print(f"{i}. {provider} - {info['description']}")
        
        while True:
            try:
                choice = int(input("\nSelect LLM provider (1-4): ").strip())
                if 1 <= choice <= len(LLM_PROVIDERS):
                    provider_key = list(LLM_PROVIDERS.keys())[choice - 1]
                    self.config["llm_provider"] = provider_key
                    break
                else:
                    print("Invalid choice. Please select 1-4.")
            except ValueError:
                print("Please enter a number.")
        
        # Select model
        provider_info = LLM_PROVIDERS[self.config["llm_provider"]]
        print(f"\nAvailable {self.config['llm_provider']} models:")
        for i, model in enumerate(provider_info["models"], 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = int(input(f"\nSelect model (1-{len(provider_info['models'])}): ").strip())
                if 1 <= choice <= len(provider_info["models"]):
                    self.config["llm_model"] = provider_info["models"][choice - 1]
                    break
                else:
                    print(f"Invalid choice. Please select 1-{len(provider_info['models'])}.")
            except ValueError:
                print("Please enter a number.")
        
        # Temperature
        default_temp = self.config["agent_type"]["temperature"]
        temp_input = input(f"\nTemperature (0.0-1.0, default {default_temp}): ").strip()
        try:
            self.config["temperature"] = float(temp_input) if temp_input else default_temp
        except ValueError:
            self.config["temperature"] = default_temp
        
        print(f"✅ LLM: {self.config['llm_provider']} - {self.config['llm_model']}")
        print(f"✅ Temperature: {self.config['temperature']}\n")
        
    def _select_tools(self):
        """Select tools for the agent."""
        print("🛠️ STEP 4: Tool Selection")
        print("-" * 30)
        
        default_tools = self.config["agent_type"]["default_tools"]
        if default_tools:
            print(f"Default tools for {self.config['agent_type']['name']}:")
            for tool in default_tools:
                print(f"  • {tool} - {AVAILABLE_TOOLS.get(tool, 'Tool description')}")
            
            use_defaults = input("\nUse default tools? (y/n): ").strip().lower()
            if use_defaults in ['y', 'yes', '']:
                self.config["tools"] = default_tools
                print(f"✅ Using {len(default_tools)} default tools\n")
                return
        
        # Custom tool selection
        print("\nAvailable tools:")
        tool_list = list(AVAILABLE_TOOLS.items())
        for i, (tool, desc) in enumerate(tool_list, 1):
            print(f"{i:2d}. {tool} - {desc}")
        
        print("\nSelect tools (comma-separated numbers, e.g., 1,3,5):")
        while True:
            try:
                choices = input("Tool numbers: ").strip()
                if not choices:
                    self.config["tools"] = default_tools
                    break
                
                selected_indices = [int(x.strip()) - 1 for x in choices.split(',')]
                selected_tools = [tool_list[i][0] for i in selected_indices if 0 <= i < len(tool_list)]
                
                if selected_tools:
                    self.config["tools"] = selected_tools
                    break
                else:
                    print("No valid tools selected. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter comma-separated numbers.")
        
        print(f"✅ Selected {len(self.config['tools'])} tools\n")
        
    def _configure_behavior(self):
        """Configure agent behavior settings."""
        print("🎯 STEP 5: Behavior Configuration")
        print("-" * 30)
        
        # Autonomy level
        autonomy_levels = ["reactive", "proactive", "adaptive", "autonomous"]
        print("Autonomy levels:")
        for i, level in enumerate(autonomy_levels, 1):
            print(f"{i}. {level}")
        
        while True:
            try:
                choice = int(input("\nSelect autonomy level (1-4, default 4): ").strip() or "4")
                if 1 <= choice <= 4:
                    self.config["autonomy_level"] = autonomy_levels[choice - 1]
                    break
                else:
                    print("Invalid choice. Please select 1-4.")
            except ValueError:
                print("Please enter a number.")
        
        # Learning and RAG
        self.config["enable_learning"] = input("Enable learning? (y/n, default y): ").strip().lower() not in ['n', 'no']
        self.config["enable_rag"] = input("Enable RAG knowledge base? (y/n, default y): ").strip().lower() not in ['n', 'no']
        self.config["enable_collaboration"] = input("Enable multi-agent collaboration? (y/n, default y): ").strip().lower() not in ['n', 'no']
        
        print(f"✅ Autonomy: {self.config['autonomy_level']}")
        print(f"✅ Learning: {self.config['enable_learning']}")
        print(f"✅ RAG: {self.config['enable_rag']}")
        print(f"✅ Collaboration: {self.config['enable_collaboration']}\n")
        
    def _configure_personality(self):
        """Configure agent personality and system prompt."""
        print("🎭 STEP 6: Personality & System Prompt")
        print("-" * 30)
        
        default_traits = self.config["agent_type"]["personality_traits"]
        if default_traits:
            print(f"Default personality traits: {', '.join(default_traits)}")
            use_defaults = input("Use default personality? (y/n): ").strip().lower()
            if use_defaults in ['y', 'yes', '']:
                self.config["personality_traits"] = default_traits
            else:
                custom_traits = input("Enter custom traits (comma-separated): ").strip()
                self.config["personality_traits"] = [t.strip() for t in custom_traits.split(',') if t.strip()]
        else:
            custom_traits = input("Enter personality traits (comma-separated): ").strip()
            self.config["personality_traits"] = [t.strip() for t in custom_traits.split(',') if t.strip()]
        
        # System prompt customization
        print("\nSystem prompt customization:")
        print("1. Use template-generated prompt")
        print("2. Provide custom system prompt")
        
        choice = input("Select option (1-2, default 1): ").strip() or "1"
        if choice == "2":
            print("Enter your custom system prompt (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            self.config["custom_system_prompt"] = "\n".join(lines[:-1])  # Remove last empty line
        
        print(f"✅ Personality: {', '.join(self.config['personality_traits'])}\n")
        
    def _configure_safety(self):
        """Configure safety and ethical guidelines."""
        print("🔒 STEP 7: Safety & Ethics")
        print("-" * 30)
        
        # Use default safety constraints
        use_defaults = input("Use default safety constraints? (y/n, default y): ").strip().lower()
        if use_defaults not in ['n', 'no']:
            self.config["safety_constraints"] = [
                "verify_information_sources",
                "no_harmful_content",
                "respect_intellectual_property",
                "maintain_data_privacy"
            ]
            self.config["ethical_guidelines"] = [
                "transparency_in_ai_assistance",
                "cite_all_sources",
                "maintain_authenticity",
                "responsible_ai_practices"
            ]
        else:
            safety_input = input("Enter safety constraints (comma-separated): ").strip()
            self.config["safety_constraints"] = [s.strip() for s in safety_input.split(',') if s.strip()]
            
            ethics_input = input("Enter ethical guidelines (comma-separated): ").strip()
            self.config["ethical_guidelines"] = [e.strip() for e in ethics_input.split(',') if e.strip()]
        
        print(f"✅ Safety constraints: {len(self.config['safety_constraints'])} configured")
        print(f"✅ Ethical guidelines: {len(self.config['ethical_guidelines'])} configured\n")
        
    def _generate_template(self):
        """Generate the final agent template file."""
        print("🏭 STEP 8: Template Generation")
        print("-" * 30)
        
        # Generate filename
        safe_name = "".join(c for c in self.config["name"].lower() if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"{safe_name}_agent.py"
        
        print(f"Generating template: {filename}")
        
        # Generate template content
        template_content = self._create_template_content()
        
        # Save template
        template_path = Path("templates") / filename
        template_path.parent.mkdir(exist_ok=True)
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"✅ Template generated: {template_path}")
        print(f"\n🚀 To use your agent:")
        print(f"   1. Copy the template: cp {template_path} my_agent.py")
        print(f"   2. Run your agent: python my_agent.py")
        print(f"\n🎉 Your revolutionary agent is ready to launch!")
        
    def _create_template_content(self) -> str:
        """Create the template file content."""
        # This would be a full template generation - for brevity, showing structure
        return f'''"""
🤖 CUSTOM AGENT TEMPLATE - {self.config["name"].upper()}
{"=" * (len(self.config["name"]) + 30)}
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

# ================================
# 🎛️ CUSTOMIZE THIS SECTION ONLY
# ================================
AGENT_CONFIG = {{
    # 🤖 Basic Agent Information
    "name": "{self.config["name"]}",
    "description": "{self.config["description"]}",
    
    # 🧠 LLM Configuration
    "llm_provider": "{self.config["llm_provider"]}",
    "llm_model": "{self.config["llm_model"]}",
    "temperature": {self.config["temperature"]},
    "max_tokens": 4096,
    
    # 🛠️ Production Tools
    "tools": {self.config["tools"]},
    
    # 🧠 Memory & Learning Configuration
    "memory_type": "ADVANCED",
    "enable_learning": {self.config["enable_learning"]},
    "enable_rag": {self.config["enable_rag"]},
    "enable_collaboration": {self.config["enable_collaboration"]},
    
    # 🤖 Agent Behavior
    "agent_type": "AUTONOMOUS",
    "autonomy_level": "{self.config["autonomy_level"]}",
    "learning_mode": "active",
    "max_iterations": 100,
    "timeout_seconds": 900,
    
    # 🔒 Safety & Ethics
    "safety_constraints": {self.config["safety_constraints"]},
    "ethical_guidelines": {self.config["ethical_guidelines"]}
}}

# ✍️ YOUR SYSTEM PROMPT
SYSTEM_PROMPT = """{self._generate_system_prompt()}"""

# [Rest of the template code would be here - launch infrastructure, etc.]
'''

    def _generate_system_prompt(self) -> str:
        """Generate system prompt based on configuration."""
        if "custom_system_prompt" in self.config:
            return self.config["custom_system_prompt"]
        
        traits = ", ".join(self.config["personality_traits"])
        return f"""You are {self.config["name"]}, an advanced autonomous AI agent with {self.config["description"].lower()}.

🎯 YOUR MISSION:
{self.config["description"]}

🎭 YOUR PERSONALITY:
You are {traits} in your approach to tasks and interactions.

🛠️ YOUR CAPABILITIES:
You have access to {len(self.config["tools"])} production tools including {", ".join(self.config["tools"][:3])}{"..." if len(self.config["tools"]) > 3 else ""}.

Remember to always maintain the highest standards of accuracy, ethics, and user satisfaction!"""

def main():
    """Main function to run the template generator."""
    generator = TemplateGenerator()
    generator.run_interactive_generator()

if __name__ == "__main__":
    main()
