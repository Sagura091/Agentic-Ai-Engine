# OpenWebUI Integration Guide

## Overview

The Agentic AI Microservice integrates seamlessly with OpenWebUI through the **OpenWebUI Pipelines framework**, exposing our LangChain/LangGraph agents as OpenAI-compatible models that can be selected and used directly in the OpenWebUI interface.

## Architecture

```mermaid
graph TB
    A[OpenWebUI Frontend] --> B[OpenWebUI Backend]
    B --> C[Agentic AI Microservice]
    C --> D[LangGraph Orchestrator]
    D --> E[Research Team Subgraph]
    D --> F[Document Team Subgraph]
    D --> G[General Agent]
    
    C --> H[Ollama LLMs]
    C --> I[PostgreSQL Database]
    C --> J[Redis State Store]
    
    subgraph "Agentic AI Service"
        C
        D
        E
        F
        G
    end
    
    subgraph "Existing Infrastructure"
        H
        I
        J
    end
```

## Key Features

### ✅ **Implemented Features**

1. **OpenAI-Compatible API Endpoints**
   - `/v1/models` - List available agent models
   - `/v1/chat/completions` - Chat completion with agents
   - Streaming and non-streaming responses
   - Full OpenAI API compatibility

2. **Agent Models Exposed to OpenWebUI**
   - `agentic-general` - General-purpose AI agent
   - `agentic-research` - Research specialist agent
   - `agentic-workflow` - Multi-agent workflow orchestrator

3. **LangGraph Hierarchical Workflows**
   - Research team subgraph (search → analysis → synthesis)
   - Document team subgraph (outline → write → edit)
   - Top-level orchestrator coordinating subgraphs

4. **Dynamic Agent Registration**
   - Runtime registration of custom agents
   - Automatic model discovery in OpenWebUI
   - RESTful agent management endpoints

5. **State Persistence**
   - Redis-backed LangGraph checkpoints
   - Conversation memory across interactions
   - Workflow state recovery

## Integration Setup

### 1. Docker Deployment

The service integrates with your existing Docker infrastructure:

```yaml
# Add to your existing docker-compose.yml
services:
  agents:
    build:
      context: ./agentic-ai-microservice
      dockerfile: Dockerfile
    container_name: agentic-agents
    ports:
      - "8001:8000"  # Agents API port
    environment:
      # Connect to existing services
      AGENTIC_DATABASE_URL: "postgresql://openwebui:password@postgres:5432/openwebui"
      AGENTIC_OLLAMA_BASE_URL: "http://ollama:11434"
      AGENTIC_OPENWEBUI_BASE_URL: "http://open-webui:8080"
      AGENTIC_REDIS_URL: "redis://redis:6379/1"
    networks:
      - aether-network
    depends_on:
      - postgres
      - ollama
      - redis
```

### 2. OpenWebUI Configuration

In OpenWebUI, add the Agentic AI service as a pipeline:

1. **Admin Panel → Settings → Pipelines**
2. **Add Pipeline URL**: `http://agents:8000`
3. **Enable Pipeline**: Toggle on
4. **Refresh Models**: Click to discover agent models

### 3. Model Selection

Once configured, you'll see new models in OpenWebUI:

- **Agentic General Assistant** - For general tasks and conversations
- **Agentic Research Agent** - For research and analysis tasks  
- **Agentic Multi-Agent Workflow** - For complex multi-step tasks

## Usage Examples

### Basic Chat with Agent

```javascript
// OpenWebUI automatically handles this, but the underlying API call is:
POST /v1/chat/completions
{
  "model": "agentic-general",
  "messages": [
    {"role": "user", "content": "Help me plan a project"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

### Research Workflow

```javascript
POST /v1/chat/completions
{
  "model": "agentic-research", 
  "messages": [
    {"role": "user", "content": "Research the latest trends in AI agents and provide a comprehensive analysis"}
  ]
}
```

### Multi-Agent Workflow

```javascript
POST /v1/chat/completions
{
  "model": "agentic-workflow",
  "messages": [
    {"role": "user", "content": "Research renewable energy technologies and write a technical report with recommendations"}
  ]
}
```

## API Endpoints

### Core OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available agent models |
| `/v1/chat/completions` | POST | Chat completion with agents |
| `/v1/models/{model_id}` | GET | Get specific model info |

### Agent Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/agents/register` | POST | Register custom agent |
| `/v1/agents/{agent_id}` | DELETE | Unregister agent |
| `/v1/health/openwebui` | GET | OpenWebUI integration health |

## Advanced Features

### Hierarchical Workflows

The system supports sophisticated multi-level workflows:

```python
# Research Team Subgraph
Research Supervisor → Search Agent → Analysis Agent → Synthesis Agent

# Document Team Subgraph  
Document Supervisor → Outline Agent → Writer Agent → Editor Agent

# Top-Level Orchestrator
Top Supervisor → Research Team → Document Team → Integration
```

### Custom Agent Registration

Register new agents dynamically:

```bash
curl -X POST http://localhost:8001/v1/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "custom-specialist",
    "name": "Custom Specialist Agent", 
    "description": "Specialized agent for custom tasks",
    "agent_type": "specialist",
    "capabilities": ["reasoning", "tool_use", "domain_expertise"]
  }'
```

### Streaming Responses

Enable real-time streaming in OpenWebUI:

```javascript
{
  "model": "agentic-general",
  "messages": [...],
  "stream": true  // Enables token-by-token streaming
}
```

## Monitoring and Debugging

### Health Checks

```bash
# Check overall service health
curl http://localhost:8001/health

# Check OpenWebUI integration specifically  
curl http://localhost:8001/v1/health/openwebui
```

### Logs

Monitor agent execution:

```bash
# View service logs
docker logs agentic-agents -f

# Check specific workflow execution
grep "workflow_id" /app/data/logs/agents.log
```

### Metrics

The service exposes Prometheus metrics:

```bash
curl http://localhost:8001/metrics
```

## Troubleshooting

### Common Issues

1. **Models not appearing in OpenWebUI**
   - Check pipeline URL configuration
   - Verify network connectivity between services
   - Check service health endpoints

2. **Agent execution failures**
   - Verify Ollama connectivity
   - Check Redis state store
   - Review agent logs for errors

3. **Slow response times**
   - Monitor LLM response times
   - Check Redis performance
   - Review workflow complexity

### Debug Mode

Enable debug logging:

```yaml
environment:
  AGENTIC_LOG_LEVEL: "DEBUG"
  AGENTIC_DEBUG: "true"
```

## Performance Considerations

### Optimization Tips

1. **Model Selection**: Choose appropriate models for tasks
2. **Caching**: Leverage Redis for state caching
3. **Concurrency**: Configure appropriate worker counts
4. **Timeouts**: Set reasonable timeout values

### Scaling

The service supports horizontal scaling:

```yaml
deploy:
  replicas: 3
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

## Security

### Authentication

The service integrates with OpenWebUI's authentication:

- User context passed through requests
- Session management via OpenWebUI
- API key validation for direct access

### Network Security

- Internal network communication only
- TLS encryption for external access
- Rate limiting and request validation

## Next Steps

1. **Custom Agent Development**: Create domain-specific agents
2. **Tool Integration**: Add external tools and APIs
3. **Workflow Optimization**: Enhance multi-agent coordination
4. **Performance Tuning**: Optimize for your specific use cases

For more detailed information, see the [API Documentation](./API_DOCUMENTATION.md) and [Development Guide](./DEVELOPMENT_GUIDE.md).
