# 🚀 **UNIVERSAL AGENT TEMPLATE - CREATE ANY AGENT IN 10 MINUTES**

Welcome to the **Revolutionary Universal Agent Template System**! Create powerful, customized AI agents using a single comprehensive template that shows EVERY possible configuration option.

## 🎯 **What Is This?**

This is a **production-ready universal template system** with:
1. **ONE comprehensive YAML template** showing ALL configuration options
2. **ONE universal Python template** that works with ANY YAML configuration
3. **Complete documentation** of every field and option
4. **No coding required** - just copy, customize YAML, and run!

## ✨ **Why Universal Template?**

### **Old Approach (Confusing):**
- ❌ 4 different templates to choose from
- ❌ Duplication between templates
- ❌ Missing advanced options in simple templates
- ❌ Hard to know what's possible

### **New Approach (Simple & Powerful):**
- ✅ ONE template showing EVERYTHING
- ✅ See all options in one place
- ✅ Enable/disable what you need
- ✅ Clear documentation for each field
- ✅ Works for ANY agent type (ReAct, Autonomous, RAG, etc.)

## 📁 **Template Files**

### **1. `universal_agent_template.yaml`** (COMPREHENSIVE YAML)
- **1000+ lines** of fully documented configuration
- Shows **EVERY possible option** for agents
- Organized into **26 clear sections**
- Marks **REQUIRED** vs **OPTIONAL** fields
- Includes **examples and valid values**
- Works for **ALL agent types**

### **2. `agent_template.py`** (UNIVERSAL PYTHON SHELL)
- **Minimal Python wrapper** (~300 lines)
- Works with **ANY YAML configuration**
- Just change `AGENT_ID` constant
- Handles initialization, execution, cleanup
- Provides interactive session mode

## 🚀 **10-MINUTE QUICK START**

### **Step 1: Copy Templates**

```bash
# Copy Python template
cp data/agents/templates/agent_template.py data/agents/my_agent.py

# Copy YAML template
cp data/config/agents/templates/universal_agent_template.yaml data/config/agents/my_agent.yaml
```

### **Step 2: Update Python File (1 minute)**

Edit `data/agents/my_agent.py` and change ONE line:

```python
# Line 73: Change this to match your YAML's agent_id
AGENT_ID = "my_agent"  # Must match agent_id in YAML
```

That's it for Python! The template handles everything else.

### **Step 3: Customize YAML (5-8 minutes)**

Edit `data/config/agents/my_agent.yaml` and configure these REQUIRED fields:

```yaml
# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED FIELDS (must configure these)
# ═══════════════════════════════════════════════════════════════════════════════

agent_id: "my_agent"              # Must match Python AGENT_ID
name: "My Agent"                  # Human-readable name
description: "What my agent does" # Brief description
agent_type: "react"               # react, autonomous, rag, workflow, etc.
framework: "react"                # basic, react, or autonomous

llm_config:
  provider: "ollama"              # ollama, openai, anthropic, google
  model_id: "llama3.2:latest"    # Your model
  temperature: 0.7                # 0.0 (focused) to 1.0 (creative)
  max_tokens: 2048                # Response length

use_cases:                        # What your agent can do
  - "web_research"
  - "text_processing"
  - "general_assistance"

system_prompt: |                  # Define agent behavior
  You are an intelligent AI agent that...
```

**Optional**: Browse the template and enable/customize other sections as needed.

### **Step 4: Run Your Agent (1 minute)**

```bash
# Run in interactive mode
python data/agents/my_agent.py
```

Or use programmatically:

```python
from data.agents.my_agent import AgentTemplate

agent = AgentTemplate()
await agent.initialize()
result = await agent.execute_task("Your task here")
```

**Total time: ~10 minutes!** ⚡

## 📚 **Template Structure Overview**

The universal template has **26 comprehensive sections**:

### **Core Sections (REQUIRED)**

1. **Agent Identity** - agent_id, name, description
2. **Agent Type & Framework** - react, autonomous, rag, etc.
5. **LLM Configuration** - provider, model, temperature, etc.
6. **Tool Configuration** - use_cases or explicit tools
10. **System Prompt** - defines agent behavior

### **Optional Sections**

3-4. **Autonomy Settings** (autonomous agents only)
7. **Memory Configuration** - short-term, long-term, episodic, etc.
8. **RAG Configuration** - knowledge base integration
9. **Personality & Expertise** - communication style, traits
11. **Performance Settings** - timeouts, iterations, caching
12. **Learning & Adaptation** - continuous improvement
13. **Goal Management** (autonomous agents only)
14. **Collaboration** - multi-agent cooperation
15. **Safety & Ethics** - constraints and guidelines
16-18. **Execution, Capabilities, Monitoring**
19-24. **Advanced Autonomous** (decision patterns, behavioral rules)
25. **Domain-Specific** (business, trading, creative configs)
26. **Custom Configuration** - your own fields

## 🎯 **Agent Type Guide**

### **ReAct Agent** (Reasoning + Acting)

**Best for**: General tasks, research, analysis, problem-solving

**Configuration**:

```yaml
agent_type: "react"
framework: "react"
# Remove autonomy sections (3, 4, 13, 19-24)
```

**Use cases**: Customer support, research assistant, data analysis

### **Autonomous Agent** (Proactive & Self-Directed)

**Best for**: Long-running tasks, monitoring, proactive assistance

**Configuration**:

```yaml
agent_type: "autonomous"
framework: "autonomous"
autonomy_level: "proactive"  # or "autonomous"
enable_proactive_behavior: true
enable_goal_setting: true
# Keep all sections, customize autonomous behavior
```

**Use cases**: Personal assistant, business automation, trading bots

### **RAG Agent** (Knowledge-Enhanced)

**Best for**: Domain-specific expertise, document Q&A

**Configuration**:

```yaml
agent_type: "rag"
framework: "basic"
rag_config:
  enable_rag: true
  collection_name: "my_knowledge_base"
```

**Use cases**: Technical support, legal assistant, medical advisor

## 🎨 **Common Customization Patterns**

### **Pattern 1: Simple Research Agent**

```yaml
agent_id: "research_agent"
agent_type: "react"
use_cases: ["web_research", "text_processing"]
memory_config:
  memory_type: "simple"
temperature: 0.3  # Focused and accurate
```

### **Pattern 2: Creative Content Agent**

```yaml
agent_id: "content_creator"
agent_type: "react"
use_cases: ["content_creation", "creative_writing", "social_media"]
temperature: 0.9  # Highly creative
creativity_level: "maximum"
```

### **Pattern 3: Business Intelligence Agent**

```yaml
agent_id: "business_analyst"
agent_type: "autonomous"
use_cases: ["business_analysis", "financial_analysis", "excel_processing"]
autonomy_level: "proactive"
memory_config:
  memory_type: "advanced"
```

### **Pattern 4: Personal Assistant**

```yaml
agent_id: "personal_assistant"
agent_type: "autonomous"
autonomy_level: "autonomous"
enable_proactive_behavior: true
enable_goal_setting: true
enable_learning: true
memory_config:
  memory_type: "advanced"
  enable_episodic: true  # Remember interactions
```

## 📖 **Configuration Reference**

### **Available Use Cases**

```yaml
use_cases:
  # Research & Information
  - "web_research"           # Web searching and information gathering
  - "document_analysis"      # Document processing and analysis
  - "text_processing"        # Text analysis and manipulation

  # Business & Finance
  - "business_analysis"      # Business intelligence and analysis
  - "financial_analysis"     # Financial metrics and reporting
  - "excel_processing"       # Excel file operations

  # Content & Creative
  - "content_creation"       # Content generation
  - "creative_writing"       # Creative content
  - "social_media"          # Social media management

  # Technical
  - "code_generation"        # Code writing and analysis
  - "data_analysis"         # Data processing and insights

  # General
  - "general_assistance"     # General-purpose help
  - "autonomous_research"    # Autonomous information gathering
  - "goal_planning"         # Goal setting and planning
  - "proactive_assistance"  # Proactive help
```

### **LLM Providers**

```yaml
llm_config:
  provider: "ollama"         # Local models
  # provider: "openai"       # OpenAI GPT models
  # provider: "anthropic"    # Claude models
  # provider: "google"       # Gemini models
```

### **Agent Types**

- **react**: Reasoning + Acting (best for most use cases)
- **autonomous**: Self-directed, proactive agents
- **rag**: Knowledge-enhanced agents
- **workflow**: Multi-step workflow agents
- **multimodal**: Text, image, audio processing
- **composite**: Multiple agent types combined

### **Memory Types**

- **none**: No memory (stateless)
- **simple**: Basic conversation memory
- **advanced**: Full memory system (short-term, long-term, episodic)
- **auto**: Automatically choose based on agent type

### **Autonomy Levels**

- **reactive**: Responds only to user input
- **proactive**: Offers suggestions and assistance
- **adaptive**: Learns and adapts behavior
- **autonomous**: Fully self-directed

## 🔍 **Testing Your Agent**

### **Interactive Mode**

```bash
python data/agents/my_agent.py
```

### **Programmatic Usage**

```python
from data.agents.my_agent import AgentTemplate

# Initialize
agent = AgentTemplate()
await agent.initialize()

# Execute tasks
result = await agent.execute_task("Research the latest AI trends")
print(result)

# Cleanup
await agent.cleanup()
```

### **Test Checklist**

- ✅ Agent initializes without errors
- ✅ Responds to simple queries
- ✅ Uses tools correctly
- ✅ Memory works (if enabled)
- ✅ Follows personality/style settings
- ✅ Handles errors gracefully

## 🚨 **Common Issues & Solutions**

### **Issue: Agent fails to initialize**

**Solution**: Check that agent_id in YAML matches AGENT_ID in Python

### **Issue: Tools not working**

**Solution**: Verify use_cases are valid and tools are available

### **Issue: Memory not persisting**

**Solution**: Ensure memory_type is not "none" and memory system is initialized

### **Issue: Responses too short/long**

**Solution**: Adjust max_tokens in llm_config

### **Issue: Agent too creative/not creative enough**

**Solution**: Adjust temperature (0.0-1.0) and creativity_level

## 💡 **Pro Tips**

1. **Start Simple**: Begin with minimal configuration, add features incrementally
2. **Test Incrementally**: Test each new feature before adding more
3. **Use Comments**: Comment out sections you might need later
4. **Check Examples**: Review existing agents in `data/config/agents/`
5. **Monitor Logs**: Enable detailed logging for debugging
6. **Version Control**: Keep your YAML in version control
7. **Document Changes**: Add comments explaining your customizations

## 🎉 **You're Ready!**

You now have everything you need to create powerful AI agents in minutes:

✅ **Universal YAML template** with all options documented
✅ **Universal Python template** that works with any configuration
✅ **Complete documentation** and examples
✅ **10-minute quick start** guide

**Start building your agent now!** 🚀

## 📞 **Additional Resources**

- **Existing Agents**: `data/config/agents/` - Real-world examples
- **System Defaults**: `data/config/agent_defaults.yaml` - Default values
- **Main Documentation**: `data/config/README.md` - System overview
- **Agent Factory**: `app/agents/factory.py` - How agents are built

**Happy Agent Building!** 🤖✨
