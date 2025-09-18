# 🚀 **AGENTIC AI BACKEND API DOCUMENTATION**

## **📋 OVERVIEW**

This document provides comprehensive documentation for the **Agentic AI Backend API** - a revolutionary multi-framework agent development platform with advanced features including Agent DNA, real-time collaboration, marketplace, and analytics.

---

## **🏗️ ARCHITECTURE**

### **Core Components**
- **Multi-Framework Agent Engine** - Support for 6 agent frameworks
- **Agent DNA System** - Personality and behavior configuration
- **Real-Time Collaboration** - WebSocket-based collaborative editing
- **Agent Marketplace** - Template sharing and discovery
- **Advanced Analytics** - Performance monitoring and optimization
- **Code Generation** - Framework-specific code export

### **Supported Frameworks**
1. **Basic** - Simple LLM-based agents
2. **ReAct** - Reasoning and Acting with tool usage
3. **BDI** - Belief-Desire-Intention architecture
4. **CrewAI** - Role-based collaborative agents
5. **AutoGen** - Multi-agent conversation framework
6. **Swarm** - Lightweight agent coordination

---

## **🔗 API ENDPOINTS**

### **Agent Management**

#### **Create Multi-Framework Agent**
```http
POST /api/v1/agents/multi-framework
Content-Type: application/json

{
  "name": "Customer Support Agent",
  "description": "Intelligent customer service agent",
  "framework": "crewai",
  "agent_dna": {
    "identity": {
      "personality": {
        "creativity": 0.3,
        "analytical": 0.8,
        "empathy": 0.9,
        "assertiveness": 0.6,
        "curiosity": 0.7
      },
      "communication_style": "friendly"
    },
    "cognition": {
      "memory_architecture": "hybrid",
      "decision_making": "analytical",
      "learning_capability": "adaptive"
    },
    "behavior": {
      "autonomy_level": "proactive",
      "collaboration_style": "supportive",
      "error_handling": "graceful"
    }
  },
  "framework_config": {
    "framework_id": "crewai",
    "components": [
      {"type": "model", "config": {"name": "llama3.2:latest"}},
      {"type": "memory", "config": {"type": "hybrid"}},
      {"type": "tools", "config": {"tools": ["web_search", "calculator"]}}
    ]
  }
}
```

**Response:**
```json
{
  "success": true,
  "agent_id": "agent_12345",
  "message": "Multi-framework agent created successfully",
  "framework": "crewai",
  "capabilities": ["reasoning", "tool_usage", "collaboration"]
}
```

#### **Get Agent List**
```http
GET /api/v1/agents/
```

#### **Get Agent Details**
```http
GET /api/v1/agents/{agent_id}
```

#### **Update Agent**
```http
PUT /api/v1/agents/{agent_id}
```

#### **Delete Agent**
```http
DELETE /api/v1/agents/{agent_id}
```

---

### **Agent DNA Management**

#### **Validate Agent DNA**
```http
POST /api/v1/agents/dna/validate
Content-Type: application/json

{
  "identity": {
    "personality": {
      "creativity": 0.8,
      "analytical": 0.6,
      "empathy": 0.7,
      "assertiveness": 0.5,
      "curiosity": 0.9
    }
  },
  "cognition": {
    "memory_architecture": "hybrid",
    "decision_making": "intuitive"
  },
  "behavior": {
    "autonomy_level": "proactive",
    "collaboration_style": "cooperative"
  }
}
```

**Response:**
```json
{
  "is_valid": true,
  "warnings": [],
  "suggestions": ["Excellent DNA configuration!"],
  "score": 0.85
}
```

#### **Get DNA Presets**
```http
GET /api/v1/agents/dna/presets
```

#### **Update Agent DNA**
```http
POST /api/v1/agents/{agent_id}/dna
```

---

### **Marketplace**

#### **Get Templates**
```http
GET /api/v1/agents/marketplace/templates
```

**Response:**
```json
{
  "templates": [
    {
      "id": "customer-support",
      "name": "Customer Support Agent",
      "description": "Intelligent customer service agent with empathy",
      "author": "AgentCorp",
      "rating": 4.8,
      "downloads": 1250,
      "tags": ["customer-service", "support", "empathy"],
      "framework": "crewai",
      "price": "free",
      "thumbnail": "🎧"
    }
  ],
  "total": 3
}
```

#### **Use Template**
```http
POST /api/v1/agents/marketplace/templates/{template_id}/use
```

#### **Rate Template**
```http
POST /api/v1/agents/marketplace/templates/{template_id}/rate
Content-Type: application/json

{
  "score": 5,
  "review": "Excellent template, works perfectly!"
}
```

---

### **Analytics**

#### **Get Performance Analytics**
```http
GET /api/v1/agents/analytics/{agent_id}/performance
```

**Response:**
```json
{
  "agent_id": "agent_12345",
  "performance": {
    "response_time": 1.2,
    "success_rate": 94.5,
    "total_requests": 1847,
    "error_rate": 5.5,
    "uptime": 99.2
  },
  "usage": {
    "daily_active": 156,
    "weekly_active": 892,
    "monthly_active": 3421
  },
  "trends": {
    "response_time_trend": "improving",
    "usage_trend": "increasing"
  }
}
```

#### **Get Behavior Analytics**
```http
GET /api/v1/agents/analytics/{agent_id}/behavior
```

#### **Get Optimization Recommendations**
```http
GET /api/v1/agents/analytics/{agent_id}/recommendations
```

#### **Get System Analytics**
```http
GET /api/v1/agents/analytics/system/metrics
```

---

### **Code Generation**

#### **Generate Framework Code**
```http
POST /api/v1/agents/codegen/framework/{framework}
Content-Type: application/json

{
  "name": "MyAgent",
  "description": "Custom agent implementation",
  "tools": ["web_search", "calculator"],
  "capabilities": ["reasoning", "tool_usage"],
  "agent_dna": {
    "identity": {"personality": {"creativity": 0.8}},
    "cognition": {"memory_architecture": "hybrid"},
    "behavior": {"autonomy_level": "autonomous"}
  }
}
```

**Response:**
```json
{
  "framework": "react",
  "code": "# ReAct Agent Implementation\nfrom langchain.agents import AgentExecutor...",
  "language": "python",
  "filename": "myagent_agent.py",
  "metadata": {
    "generated_at": "2024-01-15T10:30:00Z",
    "framework_version": "1.0.0",
    "dependencies": ["langchain", "langchain-community", "ollama"]
  }
}
```

#### **Get Framework Templates**
```http
GET /api/v1/agents/codegen/templates/{framework}
```

---

## **🔌 WEBSOCKET ENDPOINTS**

### **Real-Time Agent Communication**
```
WebSocket: ws://localhost:8000/ws
```

### **Collaborative Editing**
```
WebSocket: ws://localhost:8000/collaboration/{workspace_id}
```

**Message Types:**
- `cursor_update` - User cursor position
- `selection_update` - Text selection changes
- `document_change` - Document modifications
- `comment_add` - Add comments
- `user_joined` - User joined workspace
- `user_left` - User left workspace

---

## **🚀 GETTING STARTED**

### **1. Start the Backend**
```bash
cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **2. Test API Health**
```bash
curl http://localhost:8000/health
```

### **3. Create Your First Agent**
```bash
curl -X POST http://localhost:8000/api/v1/agents/multi-framework \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Agent",
    "description": "My first agent",
    "framework": "basic"
  }'
```

---

## **📊 FEATURES SUMMARY**

✅ **Multi-Framework Support** - 6 different agent frameworks  
✅ **Agent DNA System** - Personality and behavior configuration  
✅ **Real-Time Collaboration** - WebSocket-based collaborative editing  
✅ **Agent Marketplace** - Template sharing and discovery  
✅ **Advanced Analytics** - Performance monitoring and insights  
✅ **Code Generation** - Framework-specific Python code export  
✅ **RESTful API** - Complete CRUD operations  
✅ **WebSocket Support** - Real-time communication  
✅ **Comprehensive Logging** - Structured logging and monitoring  

**Your backend is now a complete, production-ready agent development platform!** 🎉
