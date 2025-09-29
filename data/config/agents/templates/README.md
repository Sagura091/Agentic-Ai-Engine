# 🚀 **AGENT TEMPLATES - CREATE YOUR OWN AGENTS IN MINUTES**

Welcome to the **Revolutionary Agent Template System**! Create powerful, customized AI agents using simple YAML configuration files.

## 🎯 **What Are These Templates?**

These are **production-ready agent templates** that you can:
1. **Copy** - Choose the template that matches your needs
2. **Customize** - Edit the YAML configuration to fit your requirements  
3. **Deploy** - Use the AgentBuilderFactory to create your agent instantly

**No Python coding required!** Just edit YAML configuration files.

## 🤖 **Available Templates**

### **1. Basic Agent Template** (`basic_agent_template.yaml`)
- **Perfect for**: General-purpose tasks, beginners, simple automation
- **Capabilities**: Web research, text processing, general assistance
- **Complexity**: ⭐ Simple
- **Use cases**: Customer support, information lookup, basic task automation

### **2. Autonomous Agent Template** (`autonomous_agent_template.yaml`)
- **Perfect for**: Advanced automation, proactive assistance, learning systems
- **Capabilities**: Autonomous decision making, goal setting, continuous learning
- **Complexity**: ⭐⭐⭐⭐ Advanced
- **Use cases**: Personal assistants, business automation, intelligent monitoring

### **3. Business Agent Template** (`business_agent_template.yaml`)
- **Perfect for**: Business analysis, financial metrics, strategic insights
- **Capabilities**: Business intelligence, financial analysis, report generation
- **Complexity**: ⭐⭐⭐ Intermediate-Advanced
- **Use cases**: Business analytics, financial reporting, market research

### **4. Creative Agent Template** (`creative_agent_template.yaml`)
- **Perfect for**: Content creation, writing, marketing, artistic tasks
- **Capabilities**: Creative writing, content generation, social media, copywriting
- **Complexity**: ⭐⭐ Intermediate
- **Use cases**: Content marketing, blog writing, social media management

## 🚀 **Quick Start Guide**

### **Step 1: Choose Your Template**
```bash
# Copy the template that matches your needs
cp data/config/agents/templates/basic_agent_template.yaml data/config/agents/my_agent.yaml
```

### **Step 2: Customize Your Agent**
Edit the YAML file and change these key sections:

```yaml
# ===== AGENT IDENTITY =====
agent_id: "my_custom_agent"        # CHANGE THIS: Unique ID
name: "My Custom Agent"            # CHANGE THIS: Display name
description: "What my agent does"  # CHANGE THIS: Description

# ===== LLM CONFIGURATION =====
llm_config:
  provider: "ollama"               # CHANGE THIS: ollama, openai, anthropic, google
  model_id: "llama3.2:latest"     # CHANGE THIS: Your preferred model
  temperature: 0.7                 # CHANGE THIS: 0.0 (focused) to 1.0 (creative)

# ===== TOOL CONFIGURATION =====
use_cases:                         # CHANGE THIS: Add your use cases
  - "web_research"
  - "text_processing"
  - "your_custom_use_case"

# ===== SYSTEM PROMPT =====
system_prompt: |                  # CHANGE THIS: Define your agent's behavior
  You are my custom agent that specializes in...
```

### **Step 3: Create Your Agent**
```python
from app.agents.factory import AgentBuilderFactory

# Create the factory
factory = AgentBuilderFactory(llm_manager, memory_system)

# Build your agent from YAML
agent = await factory.build_agent_from_yaml("my_custom_agent")

# Your agent is ready to use!
result = await agent.execute("Your task here")
```

## 🎨 **Customization Guide**

### **🧠 LLM Configuration**
```yaml
llm_config:
  provider: "ollama"          # Choose: ollama, openai, anthropic, google
  model_id: "llama3.2:latest" # Your model
  temperature: 0.7            # Creativity: 0.0 (focused) to 1.0 (creative)
  max_tokens: 2048           # Response length limit
```

### **🛠️ Tool Configuration**
```yaml
use_cases:
  - "web_research"           # Web searching and information gathering
  - "business_analysis"      # Business intelligence and analysis
  - "document_generation"    # Document creation and processing
  - "text_processing"        # Text analysis and manipulation
  - "social_media"          # Social media content and management
  - "creative_writing"       # Creative content generation
  - "financial_analysis"     # Financial metrics and reporting
```

### **🧠 Memory Configuration**
```yaml
memory_config:
  memory_type: "simple"      # Options: none, simple, advanced, auto
  enable_short_term: true    # Recent conversation memory
  enable_long_term: true     # Persistent knowledge storage
  enable_episodic: false     # Experience-based memories (advanced)
```

### **🎭 Personality Configuration**
```yaml
personality:
  expertise_areas:           # What your agent specializes in
    - "your_domain"
    - "your_specialty"
  
  communication_style: "professional"  # professional, casual, friendly, creative
  creativity_level: "balanced"         # conservative, balanced, creative, highly_creative
  
  traits:                    # Personality characteristics
    - "helpful"
    - "accurate"
    - "professional"
```

## 🔧 **Advanced Configuration**

### **Autonomous Behavior** (Autonomous Template Only)
```yaml
autonomy_level: "proactive"           # reactive, proactive, adaptive, autonomous
decision_threshold: 0.7               # Confidence threshold (0.0-1.0)
enable_proactive_behavior: true       # Proactive assistance
enable_goal_setting: true             # Autonomous goal setting
enable_self_improvement: true         # Continuous learning
```

### **RAG Knowledge Base**
```yaml
rag_config:
  enable_rag: true                     # Enable knowledge base
  collection_name: "my_agent_kb"      # Unique collection name
  similarity_threshold: 0.7           # Search sensitivity
  max_results: 10                     # Number of results to retrieve
```

### **Performance Tuning**
```yaml
performance:
  timeout_seconds: 300                # Maximum execution time
  max_iterations: 50                  # Maximum reasoning steps
  enable_caching: true                # Cache results for speed
  enable_streaming: false             # Stream responses
```

## 📋 **Template Comparison**

| Feature | Basic | Autonomous | Business | Creative |
|---------|-------|------------|----------|----------|
| **Complexity** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Memory Type** | Simple | Advanced | Advanced | Simple |
| **Autonomy** | None | Full | Proactive | None |
| **Learning** | No | Yes | Yes | Yes |
| **RAG Enabled** | Optional | Yes | Yes | Yes |
| **Best For** | Beginners | Advanced Users | Business | Content |

## 🎯 **Use Case Examples**

### **Customer Support Agent**
```yaml
# Use: basic_agent_template.yaml
agent_id: "customer_support_agent"
use_cases: ["web_research", "text_processing", "general_assistance"]
communication_style: "friendly"
traits: ["helpful", "patient", "empathetic"]
```

### **Business Analyst Agent**
```yaml
# Use: business_agent_template.yaml
agent_id: "business_analyst_agent"
use_cases: ["business_analysis", "financial_analysis", "document_generation"]
expertise_areas: ["financial_analysis", "market_research", "strategic_planning"]
```

### **Content Creator Agent**
```yaml
# Use: creative_agent_template.yaml
agent_id: "content_creator_agent"
use_cases: ["content_creation", "social_media", "creative_writing"]
creativity_level: "highly_creative"
content_types: ["blog_posts", "social_media", "marketing_copy"]
```

### **Personal Assistant Agent**
```yaml
# Use: autonomous_agent_template.yaml
agent_id: "personal_assistant_agent"
autonomy_level: "proactive"
use_cases: ["autonomous_research", "goal_planning", "proactive_assistance"]
enable_proactive_behavior: true
```

## 🔍 **Testing Your Agent**

After creating your agent, test it thoroughly:

```python
# Test basic functionality
result = await agent.execute("Simple test task")

# Test tool usage
result = await agent.execute("Research the latest trends in AI")

# Test memory (if enabled)
await agent.execute("Remember that I prefer detailed explanations")
result = await agent.execute("Explain machine learning")  # Should use preference

# Test domain expertise
result = await agent.execute("Task specific to your agent's domain")
```

## 🚨 **Common Mistakes to Avoid**

1. **❌ Duplicate agent_id**: Each agent must have a unique identifier
2. **❌ Invalid provider**: Make sure your LLM provider is correctly configured
3. **❌ Mismatched use_cases**: Ensure use_cases match your available tools
4. **❌ Memory type mismatch**: Advanced memory requires autonomous agent type
5. **❌ Temperature extremes**: Very high (>0.9) or very low (<0.1) temperatures can cause issues

## 🎉 **Ready to Create Your Agent?**

1. Choose your template based on your needs
2. Copy it to `data/config/agents/your_agent_name.yaml`
3. Customize the configuration
4. Use `AgentBuilderFactory.build_agent_from_yaml()` to create it
5. Test and iterate!

**Your AI agent will be ready in minutes, not hours!** 🚀

## 📞 **Need Help?**

- Check the existing agent examples in `data/config/agents/`
- Review the configuration documentation
- Test with simple tasks first, then increase complexity
- Use the basic template if you're unsure where to start

**Happy Agent Building!** 🤖✨
