# Build Your First Custom Agent

In this tutorial, you'll learn how to build a custom AI agent from scratch with specific capabilities and tools.

## 📋 What You'll Learn

By the end of this tutorial, you will:

- ✅ Understand agent types and their use cases
- ✅ Create a custom agent with specific tools
- ✅ Configure agent memory and behavior
- ✅ Test your agent with different tasks
- ✅ Monitor agent performance

**Estimated Time:** 30 minutes

## 🎯 Prerequisites

Before starting, make sure you've completed:
- **[Getting Started Tutorial](getting-started.md)** - Basic setup
- The system is running (`python -m app.main`)

## 🤖 Understanding Agent Types

The Agentic AI Engine supports three agent types:

### 1. **Basic Agents**
- Simple task-oriented agents
- No complex reasoning
- Best for: Simple Q&A, basic tasks

### 2. **ReAct Agents** (Recommended)
- Reasoning and Acting pattern
- Thought → Action → Observation cycles
- Best for: Complex tasks, tool usage, multi-step problems

### 3. **Autonomous Agents**
- Self-directed with goal management
- Learning from experience
- Best for: Long-running tasks, adaptive behavior

**For this tutorial, we'll build a ReAct agent.**

## 📝 Step 1: Plan Your Agent

Let's build a **Research Assistant Agent** that can:
- Search the web for information
- Analyze and summarize findings
- Save results to files

### Agent Specifications:
- **Name:** `research_assistant`
- **Type:** `react`
- **Tools:** `web_research`, `calculator`, `file_system`
- **Memory:** Enabled (to remember past research)

## 🛠️ Step 2: Create the Agent

### Using the API:

```python
import requests

# Agent configuration
agent_config = {
    "name": "research_assistant",
    "agent_type": "react",
    "description": "An AI research assistant that can search the web and analyze information",
    "system_prompt": """You are a professional research assistant. Your role is to:
    1. Search for accurate, up-to-date information
    2. Analyze and synthesize findings
    3. Provide clear, well-organized summaries
    4. Cite your sources
    
    Always be thorough, accurate, and objective in your research.""",
    
    "llm_config": {
        "provider": "ollama",
        "model_id": "llama3.2:latest",
        "temperature": 0.3,  # Lower temperature for more focused responses
        "max_tokens": 4000
    },
    
    "memory_config": {
        "enabled": true,
        "max_history": 50,
        "summarization_threshold": 20
    },
    
    "tools": [
        "web_research",
        "calculator",
        "file_system"
    ]
}

# Create the agent
response = requests.post(
    "http://localhost:8888/api/v1/agents",
    json=agent_config
)

agent = response.json()
agent_id = agent["id"]
print(f"✅ Created agent: {agent_id}")
```

### Using curl:

```bash
curl -X POST "http://localhost:8888/api/v1/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_assistant",
    "agent_type": "react",
    "description": "An AI research assistant",
    "system_prompt": "You are a professional research assistant...",
    "llm_config": {
      "provider": "ollama",
      "model_id": "llama3.2:latest",
      "temperature": 0.3,
      "max_tokens": 4000
    },
    "memory_config": {
      "enabled": true,
      "max_history": 50
    },
    "tools": ["web_research", "calculator", "file_system"]
  }'
```

## 🧪 Step 3: Test Your Agent

Let's test the agent with a research task:

```python
# Test 1: Simple research query
response = requests.post(
    f"http://localhost:8888/api/v1/agents/{agent_id}/chat",
    json={
        "message": "Research the latest developments in AI agents and summarize the top 3 trends."
    }
)

result = response.json()
print("Agent Response:")
print(result["response"])
print("\nTools Used:")
print(result["tools_used"])
```

### Expected Behavior:

The agent should:
1. **Think** about how to approach the task
2. **Use** the web_research tool to search
3. **Analyze** the findings
4. **Summarize** the top 3 trends
5. **Respond** with a clear summary

## 📊 Step 4: Monitor Agent Performance

Check how your agent is performing:

```python
# Get agent statistics
stats = requests.get(
    f"http://localhost:8888/api/v1/agents/{agent_id}/stats"
).json()

print(f"Total Interactions: {stats['total_interactions']}")
print(f"Average Response Time: {stats['avg_response_time']}s")
print(f"Tools Used: {stats['tools_usage']}")
print(f"Success Rate: {stats['success_rate']}%")
```

## 🎨 Step 5: Customize Agent Behavior

### Adjust Temperature

Temperature controls randomness:
- **0.0-0.3:** Focused, deterministic (good for research)
- **0.4-0.7:** Balanced (good for general tasks)
- **0.8-1.0:** Creative, varied (good for creative tasks)

```python
# Update agent configuration
requests.patch(
    f"http://localhost:8888/api/v1/agents/{agent_id}",
    json={
        "llm_config": {
            "temperature": 0.5  # More balanced
        }
    }
)
```

### Add More Tools

```python
# Add more tools to the agent
requests.patch(
    f"http://localhost:8888/api/v1/agents/{agent_id}",
    json={
        "tools": [
            "web_research",
            "calculator",
            "file_system",
            "database_operations",  # New tool
            "text_processing_nlp"   # New tool
        ]
    }
)
```

### Modify System Prompt

```python
# Update system prompt for different behavior
requests.patch(
    f"http://localhost:8888/api/v1/agents/{agent_id}",
    json={
        "system_prompt": """You are an expert research analyst specializing in technology trends.
        
        Your approach:
        1. Search multiple sources for comprehensive coverage
        2. Cross-reference information for accuracy
        3. Identify patterns and emerging trends
        4. Provide data-driven insights
        5. Include relevant statistics and examples
        
        Always maintain objectivity and cite your sources."""
    }
)
```

## 🔄 Step 6: Test Advanced Scenarios

### Multi-Step Research Task

```python
response = requests.post(
    f"http://localhost:8888/api/v1/agents/{agent_id}/chat",
    json={
        "message": """Research the following and create a report:
        1. Current state of LangChain framework
        2. Compare it with LlamaIndex
        3. Identify which is better for RAG applications
        4. Save your findings to a file called 'rag_frameworks_comparison.md'
        """
    }
)
```

The agent should:
1. Research LangChain
2. Research LlamaIndex
3. Compare both frameworks
4. Create a comparison report
5. Save to file using file_system tool

### Calculation Task

```python
response = requests.post(
    f"http://localhost:8888/api/v1/agents/{agent_id}/chat",
    json={
        "message": "If a company grows at 15% annually and starts with $1M revenue, what will revenue be in 5 years?"
    }
)
```

The agent should use the calculator tool for accurate computation.

## 📈 Step 7: Review Agent Memory

Check what the agent remembers:

```python
# Get agent memory
memory = requests.get(
    f"http://localhost:8888/api/v1/agents/{agent_id}/memory"
).json()

print("Recent Interactions:")
for interaction in memory["recent_interactions"]:
    print(f"- {interaction['timestamp']}: {interaction['summary']}")

print("\nLearned Patterns:")
for pattern in memory["learned_patterns"]:
    print(f"- {pattern}")
```

## 🎯 Best Practices

### 1. **Clear System Prompts**
- Define the agent's role clearly
- Specify expected behavior
- Include examples if needed

### 2. **Appropriate Tools**
- Only include tools the agent needs
- Too many tools can confuse the agent
- Start with 3-5 tools, add more as needed

### 3. **Temperature Settings**
- Research/Analysis: 0.2-0.4
- General Tasks: 0.5-0.7
- Creative Tasks: 0.7-0.9

### 4. **Memory Configuration**
- Enable memory for context-aware agents
- Set appropriate history limits
- Use summarization for long conversations

### 5. **Testing**
- Test with simple tasks first
- Gradually increase complexity
- Monitor performance metrics

## 🚀 Next Steps

Now that you've built your first custom agent, try:

1. **[RAG System Basics](rag-basics.md)** - Add knowledge retrieval
2. **[Agent Reference](../reference/AGENTS_SYSTEM_DOCUMENTATION.md)** - Learn all agent features
3. **[Tools Reference](../reference/TOOLS_SYSTEM_DOCUMENTATION.md)** - Explore available tools
4. **Build more agents** - Create specialized agents for different tasks

## 💡 Agent Ideas to Try

- **Code Assistant** - Helps with programming tasks
- **Data Analyst** - Analyzes data and creates visualizations
- **Content Writer** - Creates blog posts and articles
- **Customer Support** - Answers customer questions
- **Project Manager** - Helps organize and track projects

## 🛠️ Troubleshooting

### Agent Not Using Tools

**Problem:** Agent responds without using tools

**Solution:**
- Make tools more explicit in system prompt
- Lower temperature for more deterministic behavior
- Provide examples of tool usage in prompt

### Slow Response Times

**Problem:** Agent takes too long to respond

**Solution:**
- Reduce max_tokens
- Limit number of tools
- Use faster LLM model
- Enable caching

### Inconsistent Behavior

**Problem:** Agent behaves differently each time

**Solution:**
- Lower temperature (0.2-0.4)
- Make system prompt more specific
- Enable memory for consistency

## 📚 Additional Resources

- **[Agent System Documentation](../reference/AGENTS_SYSTEM_DOCUMENTATION.md)**
- **[Tools System Documentation](../reference/TOOLS_SYSTEM_DOCUMENTATION.md)**
- **[Memory System Documentation](../reference/MEMORY_SYSTEM_DOCUMENTATION.md)**
- **[LLM Configuration](../reference/LLM_SYSTEM_DOCUMENTATION.md)**

---

**Previous Tutorial:** [Getting Started](getting-started.md) ←  
**Next Tutorial:** [RAG System Basics](rag-basics.md) →

