# 🏗️ Enhanced Agentic AI Microservice Architecture

## Overview

This document outlines the comprehensive architecture for the enhanced agentic AI microservice system, building upon the existing foundation while adding revolutionary capabilities for autonomous agent orchestration.

## Core Architectural Principles

### 1. **Agentic-First Design**
- True autonomous decision-making capabilities
- Self-directed task execution and planning
- Adaptive behavior based on context and feedback
- Emergent intelligence through agent collaboration

### 2. **LangChain/LangGraph Exclusive**
- All AI operations exclusively through LangChain/LangGraph
- No direct LLM API calls outside the framework
- Proper abstraction layers for model independence
- Advanced workflow orchestration with state persistence

### 3. **Ollama-Only LLM Operations**
- Exclusive use of Ollama for all LLM inference
- Connection pooling and load balancing
- Health monitoring and automatic failover
- Dynamic model selection based on task requirements

### 4. **Dual Frontend Architecture**
- Standalone React frontend for full workflow management
- OpenWebUI Pipelines integration for existing users
- Loose coupling enabling independent operation
- Seamless interoperability between interfaces

## Enhanced System Components

### Frontend Layer

#### Standalone Frontend (React + TypeScript)
```typescript
// Core Frontend Architecture
interface FrontendArchitecture {
  components: {
    visualWorkflowBuilder: "React Flow + Custom Nodes";
    agentBuilder: "Drag & Drop Interface";
    realTimeMonitoring: "WebSocket + Recharts";
    codeEditor: "Monaco Editor Integration";
  };
  state: {
    management: "Zustand + React Query";
    persistence: "LocalStorage + IndexedDB";
    synchronization: "WebSocket Real-time";
  };
  communication: {
    api: "Axios + Custom Hooks";
    websocket: "Socket.IO Client";
    streaming: "Server-Sent Events";
  };
}
```

#### Visual Workflow Builder
- **Node-based Interface**: Custom React Flow nodes for agents, tools, and decision points
- **Drag & Drop**: Intuitive agent creation and workflow design
- **Real-time Preview**: Live workflow execution visualization
- **Code Generation**: Automatic LangGraph code generation from visual designs

### Integration Layer

#### FastAPI Gateway Enhancement
```python
# Enhanced API Structure
class EnhancedAPIArchitecture:
    endpoints = {
        "agents": {
            "crud": "/api/v1/agents/{agent_id}",
            "execute": "/api/v1/agents/{agent_id}/execute",
            "stream": "/api/v1/agents/{agent_id}/stream",
            "state": "/api/v1/agents/{agent_id}/state",
        },
        "workflows": {
            "crud": "/api/v1/workflows/{workflow_id}",
            "execute": "/api/v1/workflows/{workflow_id}/execute",
            "subgraphs": "/api/v1/workflows/{workflow_id}/subgraphs",
            "checkpoints": "/api/v1/workflows/{workflow_id}/checkpoints",
        },
        "openwebui": {
            "pipelines": "/api/v1/openwebui/pipelines",
            "models": "/api/v1/openwebui/models",
            "chat": "/api/v1/openwebui/chat",
        },
        "ollama": {
            "models": "/api/v1/ollama/models",
            "health": "/api/v1/ollama/health",
            "generate": "/api/v1/ollama/generate",
        }
    }
    
    websocket_endpoints = {
        "agent_execution": "/ws/agents/{agent_id}/execution",
        "workflow_monitoring": "/ws/workflows/{workflow_id}/monitor",
        "system_events": "/ws/system/events",
    }
```

#### OpenWebUI Pipelines Integration
```python
# OpenWebUI Pipeline Architecture
class OpenWebUIPipeline:
    """
    OpenWebUI Pipelines integration for seamless compatibility.
    Implements the Pipelines framework for dynamic model registration.
    """
    
    def __init__(self):
        self.pipeline_id = "agentic-ai-pipeline"
        self.models = []
        self.agent_orchestrator = None
    
    async def on_startup(self):
        """Initialize pipeline and register models."""
        await self.register_agentic_models()
        await self.initialize_orchestrator()
    
    async def on_shutdown(self):
        """Cleanup pipeline resources."""
        await self.cleanup_orchestrator()
    
    async def inlet(self, body: dict, user: dict) -> dict:
        """Process incoming requests from OpenWebUI."""
        return await self.route_to_agent_system(body, user)
    
    async def outlet(self, body: dict, user: dict) -> dict:
        """Process outgoing responses to OpenWebUI."""
        return await self.format_agent_response(body, user)
```

### Core Agentic AI Engine

#### Enhanced Agent Orchestrator
```python
# Advanced Agent Orchestration
class AgenticOrchestrator:
    """
    Revolutionary agent orchestrator with true agentic capabilities.
    """
    
    def __init__(self):
        self.agent_registry = {}
        self.workflow_engine = None
        self.state_manager = None
        self.tool_manager = None
        self.decision_engine = None
    
    async def create_autonomous_agent(
        self, 
        agent_spec: AgentSpecification,
        autonomy_level: AutonomyLevel = AutonomyLevel.HIGH
    ) -> AutonomousAgent:
        """Create truly autonomous agents with decision-making capabilities."""
        
    async def orchestrate_multi_agent_workflow(
        self,
        workflow_spec: WorkflowSpecification,
        coordination_pattern: CoordinationPattern = CoordinationPattern.HIERARCHICAL
    ) -> WorkflowExecution:
        """Orchestrate complex multi-agent workflows with dynamic coordination."""
        
    async def enable_agent_collaboration(
        self,
        agents: List[AutonomousAgent],
        collaboration_mode: CollaborationMode = CollaborationMode.PEER_TO_PEER
    ) -> CollaborationSession:
        """Enable dynamic agent collaboration and knowledge sharing."""
```

#### Hierarchical Subgraph System
```python
# Advanced LangGraph Subgraph Architecture
class HierarchicalSubgraphSystem:
    """
    Advanced subgraph system for complex workflow decomposition.
    """
    
    def __init__(self):
        self.subgraph_registry = {}
        self.execution_hierarchy = {}
        self.state_synchronizer = None
        self.checkpoint_manager = None
    
    async def create_hierarchical_workflow(
        self,
        root_task: Task,
        decomposition_strategy: DecompositionStrategy = DecompositionStrategy.RECURSIVE
    ) -> HierarchicalWorkflow:
        """Create hierarchical workflows with automatic task decomposition."""
        
    async def manage_subgraph_execution(
        self,
        subgraph_id: str,
        execution_context: ExecutionContext
    ) -> SubgraphResult:
        """Manage complex subgraph execution with state synchronization."""
        
    async def coordinate_cross_subgraph_communication(
        self,
        source_subgraph: str,
        target_subgraph: str,
        message: CrossSubgraphMessage
    ) -> CommunicationResult:
        """Enable communication between different subgraphs."""
```

### LangChain/LangGraph Layer

#### Advanced Edge Routing System
```python
# Sophisticated Edge Routing
class AdvancedEdgeRouter:
    """
    Advanced edge routing system for complex conditional logic.
    """
    
    def __init__(self):
        self.routing_rules = {}
        self.decision_trees = {}
        self.learning_system = None
    
    async def create_conditional_edge(
        self,
        source_node: str,
        condition_func: Callable,
        target_mapping: Dict[str, str],
        fallback_target: str = "error_handler"
    ) -> ConditionalEdge:
        """Create sophisticated conditional edges with fallback handling."""
        
    async def implement_adaptive_routing(
        self,
        routing_context: RoutingContext,
        learning_enabled: bool = True
    ) -> AdaptiveRoute:
        """Implement adaptive routing that learns from execution patterns."""
        
    async def handle_dynamic_branching(
        self,
        branching_context: BranchingContext,
        max_parallel_branches: int = 5
    ) -> DynamicBranchResult:
        """Handle dynamic branching based on runtime conditions."""
```

#### Persistent State Management
```python
# Advanced State Persistence
class PersistentStateManager:
    """
    Advanced state management with checkpointing and recovery.
    """
    
    def __init__(self):
        self.checkpoint_store = None
        self.state_serializer = None
        self.recovery_engine = None
    
    async def create_checkpoint(
        self,
        workflow_id: str,
        state: WorkflowState,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC
    ) -> Checkpoint:
        """Create workflow checkpoints for recovery and resumption."""
        
    async def restore_from_checkpoint(
        self,
        checkpoint_id: str,
        restoration_strategy: RestorationStrategy = RestorationStrategy.EXACT
    ) -> RestoredWorkflow:
        """Restore workflow execution from checkpoint."""
        
    async def implement_state_versioning(
        self,
        state_id: str,
        versioning_strategy: VersioningStrategy = VersioningStrategy.SEMANTIC
    ) -> StateVersion:
        """Implement state versioning for workflow evolution."""
```

### Ollama Integration Layer

#### Connection Pool Management
```python
# Advanced Ollama Integration
class OllamaConnectionManager:
    """
    Advanced Ollama integration with connection pooling and load balancing.
    """
    
    def __init__(self):
        self.connection_pool = None
        self.load_balancer = None
        self.health_monitor = None
        self.model_manager = None
    
    async def initialize_connection_pool(
        self,
        pool_size: int = 10,
        max_connections: int = 50,
        connection_timeout: int = 30
    ) -> ConnectionPool:
        """Initialize optimized connection pool for Ollama."""
        
    async def implement_load_balancing(
        self,
        balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check_interval: int = 30
    ) -> LoadBalancer:
        """Implement intelligent load balancing across Ollama instances."""
        
    async def manage_model_lifecycle(
        self,
        model_spec: ModelSpecification,
        lifecycle_policy: LifecyclePolicy = LifecyclePolicy.LAZY_LOADING
    ) -> ModelLifecycle:
        """Manage model loading, unloading, and resource optimization."""
```

## Database Schema Design

### PostgreSQL Schema
```sql
-- Enhanced database schema for agent state and workflows
CREATE SCHEMA agentic_ai;

-- Agents table with enhanced capabilities
CREATE TABLE agentic_ai.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    capabilities TEXT[] DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'idle',
    autonomy_level VARCHAR(50) DEFAULT 'medium',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    metadata JSONB DEFAULT '{}'
);

-- Workflows table with hierarchical support
CREATE TABLE agentic_ai.workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(100) NOT NULL,
    definition JSONB NOT NULL,
    parent_workflow_id UUID REFERENCES agentic_ai.workflows(id),
    hierarchy_level INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'draft',
    version VARCHAR(50) DEFAULT '1.0.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    metadata JSONB DEFAULT '{}'
);

-- Subgraphs table for hierarchical workflows
CREATE TABLE agentic_ai.subgraphs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES agentic_ai.workflows(id),
    subgraph_name VARCHAR(255) NOT NULL,
    subgraph_definition JSONB NOT NULL,
    parent_subgraph_id UUID REFERENCES agentic_ai.subgraphs(id),
    execution_order INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Execution sessions with state tracking
CREATE TABLE agentic_ai.execution_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES agentic_ai.workflows(id),
    agent_id UUID REFERENCES agentic_ai.agents(id),
    session_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    current_state JSONB DEFAULT '{}',
    execution_context JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_details JSONB,
    metrics JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Checkpoints for state persistence
CREATE TABLE agentic_ai.checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES agentic_ai.execution_sessions(id),
    checkpoint_name VARCHAR(255),
    checkpoint_type VARCHAR(100) DEFAULT 'automatic',
    state_snapshot JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Tool registry and usage tracking
CREATE TABLE agentic_ai.tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    tool_type VARCHAR(100) NOT NULL,
    configuration JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Agent-tool associations
CREATE TABLE agentic_ai.agent_tools (
    agent_id UUID REFERENCES agentic_ai.agents(id),
    tool_id UUID REFERENCES agentic_ai.tools(id),
    is_enabled BOOLEAN DEFAULT true,
    configuration_override JSONB DEFAULT '{}',
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (agent_id, tool_id)
);

-- Indexes for performance
CREATE INDEX idx_agents_status ON agentic_ai.agents(status);
CREATE INDEX idx_agents_type ON agentic_ai.agents(agent_type);
CREATE INDEX idx_workflows_status ON agentic_ai.workflows(status);
CREATE INDEX idx_workflows_hierarchy ON agentic_ai.workflows(parent_workflow_id, hierarchy_level);
CREATE INDEX idx_sessions_status ON agentic_ai.execution_sessions(status);
CREATE INDEX idx_sessions_workflow ON agentic_ai.execution_sessions(workflow_id);
CREATE INDEX idx_checkpoints_session ON agentic_ai.checkpoints(session_id);
```

This enhanced architecture provides:

1. **True Agentic Capabilities**: Autonomous decision-making and self-directed execution
2. **Advanced LangGraph Integration**: Hierarchical subgraphs and sophisticated routing
3. **Comprehensive State Management**: Persistent checkpoints and recovery systems
4. **Dual Frontend Support**: Standalone React app + OpenWebUI integration
5. **Exclusive Ollama Integration**: Optimized connection pooling and model management
6. **Production-Ready Infrastructure**: Comprehensive monitoring and deployment support

The architecture maintains loose coupling while enabling seamless interoperability, following the Ollama-OpenWebUI pattern you specified.
