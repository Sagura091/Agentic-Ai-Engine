# 🔌 Enhanced API Contracts Specification

## Overview

This document defines the comprehensive API contracts for the enhanced agentic AI microservice, including REST endpoints, WebSocket connections, and OpenWebUI Pipelines integration.

## Core API Principles

### 1. **RESTful Design**
- Resource-based URLs with clear hierarchies
- Standard HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Consistent response formats with proper status codes
- Comprehensive error handling with detailed messages

### 2. **Real-time Communication**
- WebSocket endpoints for live monitoring and streaming
- Server-Sent Events for one-way real-time updates
- Proper connection management and reconnection logic

### 3. **OpenWebUI Compatibility**
- Pipelines framework implementation
- Dynamic model registration and discovery
- Seamless chat interface integration

## Enhanced Agent Management API

### Agent CRUD Operations

```yaml
# Create Agent
POST /api/v1/agents
Content-Type: application/json

{
  "name": "Research Assistant",
  "description": "Autonomous research and analysis agent",
  "agent_type": "research_agent",
  "config": {
    "model_name": "llama3.2:latest",
    "temperature": 0.7,
    "max_tokens": 2048,
    "system_prompt": "You are an autonomous research assistant...",
    "capabilities": ["reasoning", "tool_use", "memory", "planning"],
    "autonomy_level": "high",
    "tools": ["web_search", "document_analyzer", "data_processor"],
    "max_iterations": 50,
    "timeout_seconds": 300
  },
  "metadata": {
    "created_by": "user_id",
    "tags": ["research", "autonomous"],
    "version": "1.0.0"
  }
}

Response: 201 Created
{
  "agent_id": "uuid",
  "name": "Research Assistant",
  "status": "created",
  "capabilities": ["reasoning", "tool_use", "memory", "planning"],
  "created_at": "2024-01-01T00:00:00Z",
  "endpoints": {
    "execute": "/api/v1/agents/{agent_id}/execute",
    "stream": "/api/v1/agents/{agent_id}/stream",
    "state": "/api/v1/agents/{agent_id}/state",
    "websocket": "/ws/agents/{agent_id}/execution"
  }
}
```

### Agent Execution API

```yaml
# Execute Agent Task
POST /api/v1/agents/{agent_id}/execute
Content-Type: application/json

{
  "task": "Research the latest developments in quantum computing",
  "context": {
    "priority": "high",
    "deadline": "2024-01-02T00:00:00Z",
    "constraints": ["academic_sources_only", "peer_reviewed"]
  },
  "execution_mode": "autonomous",
  "streaming": false,
  "checkpoint_enabled": true
}

Response: 200 OK
{
  "execution_id": "uuid",
  "agent_id": "uuid",
  "status": "running",
  "started_at": "2024-01-01T00:00:00Z",
  "estimated_completion": "2024-01-01T00:15:00Z",
  "progress": {
    "current_step": "research_planning",
    "steps_completed": 0,
    "total_steps": 5,
    "percentage": 0
  },
  "monitoring": {
    "websocket_url": "/ws/agents/{agent_id}/execution",
    "status_endpoint": "/api/v1/agents/{agent_id}/executions/{execution_id}"
  }
}
```

### Agent Streaming API

```yaml
# Stream Agent Execution
GET /api/v1/agents/{agent_id}/stream/{execution_id}
Accept: text/event-stream

# Server-Sent Events Stream
data: {"type": "status", "status": "running", "step": "research_planning"}

data: {"type": "progress", "percentage": 20, "message": "Analyzing research requirements"}

data: {"type": "tool_call", "tool": "web_search", "args": {"query": "quantum computing 2024"}}

data: {"type": "tool_result", "tool": "web_search", "result": {"sources": [...]}}

data: {"type": "reasoning", "thought": "Based on search results, I need to analyze..."}

data: {"type": "completion", "status": "completed", "result": {...}}
```

## Enhanced Workflow Management API

### Hierarchical Workflow Creation

```yaml
# Create Hierarchical Workflow
POST /api/v1/workflows
Content-Type: application/json

{
  "name": "Research and Documentation Workflow",
  "description": "Hierarchical workflow for research and document creation",
  "workflow_type": "hierarchical",
  "definition": {
    "root_task": "Create comprehensive research report",
    "decomposition_strategy": "recursive",
    "coordination_pattern": "hierarchical",
    "subgraphs": [
      {
        "name": "research_team",
        "type": "research_subgraph",
        "agents": ["search_agent", "analysis_agent", "synthesis_agent"],
        "coordination": "supervisor_worker"
      },
      {
        "name": "document_team",
        "type": "document_subgraph",
        "agents": ["outline_agent", "writer_agent", "editor_agent"],
        "coordination": "sequential"
      }
    ],
    "integration_rules": {
      "research_to_document": "pass_synthesis_results",
      "error_handling": "graceful_degradation",
      "timeout_strategy": "partial_completion"
    }
  },
  "execution_config": {
    "max_parallel_subgraphs": 3,
    "checkpoint_frequency": "per_subgraph",
    "state_persistence": "full",
    "monitoring_level": "detailed"
  }
}

Response: 201 Created
{
  "workflow_id": "uuid",
  "name": "Research and Documentation Workflow",
  "status": "created",
  "hierarchy_level": 0,
  "subgraph_count": 2,
  "estimated_complexity": "high",
  "endpoints": {
    "execute": "/api/v1/workflows/{workflow_id}/execute",
    "subgraphs": "/api/v1/workflows/{workflow_id}/subgraphs",
    "checkpoints": "/api/v1/workflows/{workflow_id}/checkpoints",
    "monitoring": "/ws/workflows/{workflow_id}/monitor"
  }
}
```

### Subgraph Management

```yaml
# Get Workflow Subgraphs
GET /api/v1/workflows/{workflow_id}/subgraphs

Response: 200 OK
{
  "workflow_id": "uuid",
  "subgraphs": [
    {
      "subgraph_id": "uuid",
      "name": "research_team",
      "type": "research_subgraph",
      "status": "ready",
      "agents": [
        {
          "agent_id": "uuid",
          "name": "search_agent",
          "role": "worker",
          "status": "idle"
        }
      ],
      "execution_order": 1,
      "dependencies": [],
      "estimated_duration": "5-10 minutes"
    }
  ],
  "execution_graph": {
    "nodes": ["research_team", "document_team", "integration_node"],
    "edges": [
      {"from": "research_team", "to": "integration_node"},
      {"from": "integration_node", "to": "document_team"}
    ]
  }
}
```

## OpenWebUI Pipelines Integration

### Pipeline Registration

```python
# OpenWebUI Pipeline Implementation
class AgenticAIPipeline:
    """
    OpenWebUI Pipelines integration for seamless compatibility.
    """
    
    def __init__(self):
        self.type = "manifold"  # OpenWebUI pipeline type
        self.id = "agentic_ai_pipeline"
        self.name = "Agentic AI Pipeline"
        self.version = "1.0.0"
    
    async def on_startup(self):
        """Initialize pipeline and register agentic models."""
        self.models = await self.discover_agentic_models()
        return {
            "status": "initialized",
            "models": self.models,
            "capabilities": [
                "autonomous_agents",
                "hierarchical_workflows", 
                "real_time_monitoring",
                "state_persistence"
            ]
        }
    
    async def on_shutdown(self):
        """Cleanup pipeline resources."""
        await self.cleanup_agent_sessions()
        return {"status": "shutdown_complete"}
    
    async def inlet(self, body: dict, user: dict) -> dict:
        """Process incoming requests from OpenWebUI."""
        # Route to appropriate agent or workflow
        if body.get("model", "").startswith("agent:"):
            return await self.route_to_agent(body, user)
        elif body.get("model", "").startswith("workflow:"):
            return await self.route_to_workflow(body, user)
        else:
            return await self.route_to_default_agent(body, user)
    
    async def outlet(self, body: dict, user: dict) -> dict:
        """Process outgoing responses to OpenWebUI."""
        # Format agent responses for OpenWebUI chat interface
        if "agent_response" in body:
            return self.format_agent_response(body["agent_response"])
        return body
```

### Dynamic Model Registration

```yaml
# Register Agentic Models with OpenWebUI
POST /api/v1/openwebui/models/register

{
  "models": [
    {
      "id": "agent:research_assistant",
      "name": "Research Assistant Agent",
      "description": "Autonomous research and analysis agent",
      "capabilities": ["autonomous_research", "data_analysis", "report_generation"],
      "parameters": {
        "autonomy_level": {
          "type": "select",
          "options": ["low", "medium", "high"],
          "default": "medium"
        },
        "research_depth": {
          "type": "range",
          "min": 1,
          "max": 10,
          "default": 5
        }
      }
    },
    {
      "id": "workflow:research_and_document",
      "name": "Research & Documentation Workflow",
      "description": "Complete research and documentation pipeline",
      "capabilities": ["hierarchical_execution", "multi_agent_coordination"],
      "parameters": {
        "workflow_complexity": {
          "type": "select", 
          "options": ["simple", "standard", "complex"],
          "default": "standard"
        }
      }
    }
  ]
}

Response: 200 OK
{
  "registered_models": 2,
  "models": [
    {
      "id": "agent:research_assistant",
      "status": "registered",
      "openwebui_endpoint": "/api/v1/openwebui/chat/agent:research_assistant"
    }
  ]
}
```

## Ollama Integration API

### Model Management

```yaml
# Get Available Ollama Models
GET /api/v1/ollama/models

Response: 200 OK
{
  "models": [
    {
      "name": "llama3.2:latest",
      "size": "4.7GB",
      "status": "loaded",
      "capabilities": ["chat", "reasoning", "tool_use"],
      "performance_metrics": {
        "tokens_per_second": 45,
        "memory_usage": "3.2GB",
        "load_time": "12s"
      }
    }
  ],
  "connection_pool": {
    "active_connections": 5,
    "max_connections": 10,
    "health_status": "healthy"
  }
}
```

### Health Monitoring

```yaml
# Ollama Health Check
GET /api/v1/ollama/health

Response: 200 OK
{
  "status": "healthy",
  "instances": [
    {
      "url": "http://ollama:11434",
      "status": "online",
      "response_time": "45ms",
      "models_loaded": 2,
      "memory_usage": "6.4GB",
      "last_check": "2024-01-01T00:00:00Z"
    }
  ],
  "load_balancer": {
    "strategy": "round_robin",
    "active_instances": 1,
    "total_requests": 1250,
    "failed_requests": 3
  }
}
```

## WebSocket API Specifications

### Agent Execution Monitoring

```javascript
// WebSocket connection for real-time agent monitoring
const ws = new WebSocket('/ws/agents/{agent_id}/execution');

// Message types received:
{
  "type": "status_update",
  "agent_id": "uuid",
  "status": "running",
  "current_step": "reasoning",
  "timestamp": "2024-01-01T00:00:00Z"
}

{
  "type": "progress_update", 
  "agent_id": "uuid",
  "progress": {
    "percentage": 45,
    "current_task": "Analyzing search results",
    "estimated_remaining": "2 minutes"
  }
}

{
  "type": "tool_execution",
  "agent_id": "uuid",
  "tool": {
    "name": "web_search",
    "args": {"query": "quantum computing"},
    "status": "executing"
  }
}

{
  "type": "reasoning_step",
  "agent_id": "uuid", 
  "reasoning": {
    "thought": "Based on the search results...",
    "decision": "I need to analyze the academic papers",
    "next_action": "document_analyzer"
  }
}

{
  "type": "execution_complete",
  "agent_id": "uuid",
  "result": {
    "status": "completed",
    "outputs": {...},
    "execution_time": "8 minutes 32 seconds",
    "tokens_used": 2847
  }
}
```

### Workflow Monitoring

```javascript
// WebSocket connection for workflow monitoring
const ws = new WebSocket('/ws/workflows/{workflow_id}/monitor');

// Hierarchical workflow events:
{
  "type": "workflow_started",
  "workflow_id": "uuid",
  "subgraphs": ["research_team", "document_team"],
  "coordination_pattern": "hierarchical"
}

{
  "type": "subgraph_status",
  "workflow_id": "uuid",
  "subgraph_id": "uuid",
  "subgraph_name": "research_team",
  "status": "executing",
  "agents": [
    {"agent_id": "uuid", "name": "search_agent", "status": "running"},
    {"agent_id": "uuid", "name": "analysis_agent", "status": "waiting"}
  ]
}

{
  "type": "cross_subgraph_communication",
  "workflow_id": "uuid",
  "from_subgraph": "research_team",
  "to_subgraph": "document_team", 
  "message_type": "research_results",
  "data": {...}
}

{
  "type": "checkpoint_created",
  "workflow_id": "uuid",
  "checkpoint_id": "uuid",
  "checkpoint_type": "automatic",
  "state_size": "2.3MB"
}
```

## Error Handling and Response Formats

### Standard Error Response

```yaml
# Error Response Format
{
  "error": {
    "code": "AGENT_EXECUTION_FAILED",
    "message": "Agent execution failed due to tool timeout",
    "details": {
      "agent_id": "uuid",
      "execution_id": "uuid", 
      "failed_step": "tool_execution",
      "tool_name": "web_search",
      "error_type": "timeout",
      "retry_possible": true
    },
    "timestamp": "2024-01-01T00:00:00Z",
    "request_id": "uuid"
  }
}
```

### Status Codes

- `200 OK` - Successful operation
- `201 Created` - Resource created successfully
- `202 Accepted` - Request accepted for processing
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict
- `422 Unprocessable Entity` - Validation errors
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

This comprehensive API specification provides the foundation for building a production-ready agentic AI microservice with full OpenWebUI compatibility and advanced autonomous capabilities.
