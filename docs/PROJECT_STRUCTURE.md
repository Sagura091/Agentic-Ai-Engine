# Agentic AI Microservice Project Structure

## Overview
This document outlines the complete project structure for the containerized agentic AI microservice system.

## Directory Structure

```
agents-microservice/
├── README.md
├── ARCHITECTURE.md
├── CHANGELOG.md
├── LICENSE
├── .gitignore
├── .env.example
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── Dockerfile
├── Dockerfile.dev
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── pytest.ini
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── security.yml
├── scripts/
│   ├── start.sh
│   ├── test.sh
│   ├── build.sh
│   └── deploy.sh
├── docs/
│   ├── api/
│   ├── deployment/
│   ├── development/
│   └── examples/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── database.py
│   │   └── logging.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── security.py
│   │   ├── middleware.py
│   │   └── exceptions.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py
│   │   │   ├── endpoints/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agents.py
│   │   │   │   ├── workflows.py
│   │   │   │   ├── health.py
│   │   │   │   └── admin.py
│   │   │   └── dependencies.py
│   │   └── websocket/
│   │       ├── __init__.py
│   │       ├── manager.py
│   │       └── handlers.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── state.py
│   │   │   └── tools.py
│   │   ├── registry/
│   │   │   ├── __init__.py
│   │   │   ├── manager.py
│   │   │   └── loader.py
│   │   ├── factory/
│   │   │   ├── __init__.py
│   │   │   ├── builder.py
│   │   │   └── templates.py
│   │   ├── builtin/
│   │   │   ├── __init__.py
│   │   │   ├── research_agent.py
│   │   │   ├── coding_agent.py
│   │   │   ├── analysis_agent.py
│   │   │   └── supervisor_agent.py
│   │   └── custom/
│   │       ├── __init__.py
│   │       └── README.md
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── workflow_engine.py
│   │   ├── state_manager.py
│   │   ├── task_queue.py
│   │   └── scheduler.py
│   ├── langgraph/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   ├── subgraph_manager.py
│   │   ├── edge_processor.py
│   │   ├── checkpoint_system.py
│   │   └── state_persistence.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── openwebui/
│   │   │   ├── __init__.py
│   │   │   ├── pipelines/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent_pipe.py
│   │   │   │   ├── orchestrator_pipe.py
│   │   │   │   └── llm_selector_pipe.py
│   │   │   ├── filters/
│   │   │   │   ├── __init__.py
│   │   │   │   └── agent_filter.py
│   │   │   └── utils.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── ollama.py
│   │   │   ├── openai.py
│   │   │   ├── anthropic.py
│   │   │   └── selector.py
│   │   └── external/
│   │       ├── __init__.py
│   │       └── README.md
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── agent.py
│   │   │   ├── workflow.py
│   │   │   ├── session.py
│   │   │   └── user.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── workflow.py
│   │       ├── response.py
│   │       └── request.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py
│   │   ├── workflow_service.py
│   │   ├── llm_service.py
│   │   └── monitoring_service.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── helpers.py
│       ├── validators.py
│       └── decorators.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_agents/
│   │   ├── test_orchestration/
│   │   ├── test_langgraph/
│   │   └── test_integrations/
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api/
│   │   ├── test_workflows/
│   │   └── test_openwebui/
│   ├── e2e/
│   │   ├── __init__.py
│   │   └── test_full_workflow.py
│   └── fixtures/
│       ├── __init__.py
│       ├── agents.py
│       └── workflows.py
├── examples/
│   ├── README.md
│   ├── basic_agent/
│   ├── multi_agent_workflow/
│   ├── custom_agent_creation/
│   └── openwebui_integration/
├── deployment/
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── ingress.yaml
│   ├── helm/
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── monitoring/
│   ├── prometheus/
│   │   └── rules.yml
│   ├── grafana/
│   │   └── dashboards/
│   └── alerts/
│       └── rules.yml
└── data/
    ├── agents/
    ├── workflows/
    ├── checkpoints/
    └── logs/
```

## Key Components Description

### Core Application (`app/`)
- **main.py**: FastAPI application entry point
- **config/**: Configuration management and settings
- **core/**: Core utilities, dependencies, and middleware
- **api/**: REST API endpoints and WebSocket handlers

### Agent System (`app/agents/`)
- **base/**: Base agent classes and interfaces
- **registry/**: Agent registration and discovery
- **factory/**: Dynamic agent creation system
- **builtin/**: Pre-built agent templates
- **custom/**: User-defined custom agents

### Orchestration (`app/orchestration/`)
- **orchestrator.py**: Main orchestration engine
- **workflow_engine.py**: Workflow execution management
- **state_manager.py**: State persistence and recovery
- **task_queue.py**: Asynchronous task processing

### Orchestration (`app/orchestration/`)
- **orchestrator.py**: Main orchestration engine
- **enhanced_orchestrator.py**: Enhanced orchestration features
- **subgraphs.py**: LangGraph subgraphs and hierarchical workflows

### OpenWebUI Integration (`app/integrations/openwebui/`)
- **pipelines/**: OpenWebUI pipeline implementations
- **filters/**: Custom filters for data processing
- **utils.py**: Integration utilities and helpers

### Testing (`tests/`)
- **unit/**: Unit tests for individual components
- **integration/**: Integration tests for component interaction
- **e2e/**: End-to-end workflow testing
- **fixtures/**: Test data and mock objects

### Deployment (`deployment/`)
- **kubernetes/**: K8s manifests for production deployment
- **helm/**: Helm charts for easy deployment
- **terraform/**: Infrastructure as code

### Monitoring (`monitoring/`)
- **prometheus/**: Metrics collection configuration
- **grafana/**: Visualization dashboards
- **alerts/**: Alerting rules and notifications

## Container Architecture

### Development Environment
```yaml
version: '3.8'
services:
  agents:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
    volumes:
      - ./app:/app
      - ./data:/data
```

### Production Environment
```yaml
version: '3.8'
services:
  agents:
    image: agents-microservice:latest
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    volumes:
      - agents_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Integration with OpenWebUI

The service integrates with OpenWebUI through the Pipelines framework:

1. **Service Discovery**: OpenWebUI discovers the agent service via configured endpoint
2. **Pipeline Registration**: Agent pipelines register as available models
3. **Dynamic Routing**: Requests route to appropriate agents based on context
4. **State Persistence**: Conversation state maintained across interactions

## Next Steps

1. Create the basic project structure
2. Implement core FastAPI application
3. Build agent framework with LangGraph
4. Develop OpenWebUI integration
5. Add comprehensive testing
6. Create deployment configurations
