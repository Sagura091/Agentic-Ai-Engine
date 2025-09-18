# Agentic AI Microservice

A revolutionary agentic AI microservice with LangChain/LangGraph integration and OpenWebUI compatibility. This service operates as a separate containerized system that integrates seamlessly with your existing OpenWebUI and Ollama infrastructure.

## 🚀 Features

- **Containerized Architecture**: Runs as an independent Docker service that integrates with OpenWebUI
- **LangChain/LangGraph Integration**: Full support for complex agent workflows with subgraphs and edge nodes
- **OpenWebUI Pipelines**: Native integration through the OpenWebUI Pipelines framework
- **Multi-LLM Support**: Works with Ollama, OpenAI, and other LLM providers
- **Agent Orchestration**: Sophisticated multi-agent coordination and workflow management
- **State Persistence**: Redis-backed state management with checkpoint recovery
- **Production Ready**: Comprehensive monitoring, logging, and error handling

## 🏗️ Architecture

The system follows a microservice architecture pattern similar to how Ollama integrates with OpenWebUI:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenWebUI     │    │   Agent Service │    │     Ollama      │
│   Container     │◄──►│   Container     │◄──►│   Container     │
│   Port: 3000    │    │   Port: 8001    │    │   Port: 11434   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

- **FastAPI Microservice**: High-performance async web service
- **Agent Framework**: LangGraph-based agent system with state management
- **Orchestration Layer**: Multi-agent workflow coordination
- **OpenWebUI Integration**: Pipeline-based integration for seamless UI access
- **LLM Integration**: Dynamic model selection and load balancing

## 🛠️ Installation & Setup

### Prerequisites

- Docker and Docker Compose
- Your existing OpenWebUI/Ollama infrastructure
- Python 3.11+ (for development)

### Quick Start with Docker Compose

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agents-microservice
   ```

2. **Copy environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your specific settings
   ```

3. **Add to your existing Docker Compose**:
   
   You can either:
   
   **Option A**: Add our service to your existing `docker-compose.yml`:
   ```yaml
   # Add this to your existing services section
   agents:
     image: agentic-agents:latest
     container_name: agentic-agents
     restart: unless-stopped
     ports:
       - "8001:8000"
     environment:
       AGENTIC_DATABASE_URL: "postgresql://openwebui:your_secure_password_here@postgres:5432/openwebui"
       AGENTIC_OLLAMA_BASE_URL: "http://ollama:11434"
       AGENTIC_OPENWEBUI_BASE_URL: "http://open-webui:8080"
       AGENTIC_REDIS_URL: "redis://redis:6379/1"
     volumes:
       - agents_data:/app/data
     networks:
       - aether-network
     depends_on:
       - postgres
       - redis
       - ollama
   ```
   
   **Option B**: Use our separate compose file:
   ```bash
   # Start with your existing infrastructure
   docker-compose up -d
   
   # Add the agent service
   docker-compose -f docker-compose.agents.yml up -d
   ```

4. **Build and start the service**:
   ```bash
   docker-compose -f docker-compose.agents.yml up --build -d
   ```

5. **Verify the service is running**:
   ```bash
   curl http://localhost:8001/health
   ```

### Integration with OpenWebUI

The agent service automatically registers with OpenWebUI through the Pipelines framework:

1. **Configure OpenWebUI** to connect to the agent service:
   - Go to Admin Panel > Settings > Connections
   - Add new connection: `http://localhost:8001` (or your agent service URL)
   - The agents will appear as available models in the OpenWebUI interface

2. **Access agent capabilities** through the OpenWebUI chat interface:
   - Select an agent "model" from the dropdown
   - Start chatting to interact with the agent system
   - Agents can collaborate and hand off tasks automatically

## 🔧 Development

### Local Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Configure for local development
   ```

3. **Run database migrations**:
   ```bash
   alembic upgrade head
   ```

4. **Start the development server**:
   ```bash
   python -m app.main
   # Or use the development Docker container
   docker-compose -f docker-compose.agents.yml --profile dev up
   ```

### Project Structure

```
agents-microservice/
├── app/                          # Main application code
│   ├── agents/                   # Agent system
│   │   ├── base/                 # Base agent classes
│   │   ├── builtin/              # Pre-built agents
│   │   ├── custom/               # Custom agent definitions
│   │   ├── factory/              # Agent creation system
│   │   └── registry/             # Agent registration
│   ├── api/                      # REST API endpoints
│   ├── config/                   # Configuration management
│   ├── core/                     # Core utilities and middleware
│   ├── integrations/             # External service integrations
│   │   ├── openwebui/            # OpenWebUI pipelines
│   │   └── llm/                  # LLM provider integrations
│   ├── langgraph/                # LangGraph workflow engine
│   ├── models/                   # Database models and schemas
│   ├── orchestration/            # Agent orchestration
│   ├── services/                 # Business logic services
│   └── utils/                    # Utility functions
├── tests/                        # Test suite
├── docs/                         # Documentation
├── deployment/                   # Deployment configurations
└── examples/                     # Example agents and workflows
```

## 📊 Monitoring & Observability

The service includes comprehensive monitoring:

- **Health Checks**: `/health`, `/ready`, `/live` endpoints
- **Metrics**: Prometheus metrics at `/metrics`
- **Logging**: Structured JSON logging with request tracing
- **Performance**: Request duration and agent execution metrics

## 🔒 Security

- **Authentication**: JWT-based authentication system
- **Authorization**: Role-based access control
- **Security Headers**: Comprehensive security header middleware
- **Rate Limiting**: Built-in rate limiting protection
- **Input Validation**: Strict input validation and sanitization

## 🚀 Current Implementation Status

### ✅ Completed
- [x] Project structure and configuration
- [x] FastAPI application foundation
- [x] Docker containerization
- [x] Base agent framework
- [x] Configuration management
- [x] Health check endpoints
- [x] Middleware (logging, metrics, security)
- [x] Exception handling system

### 🚧 In Progress
- [ ] LangGraph integration and workflow engine
- [ ] Agent orchestration system
- [ ] OpenWebUI pipeline integration
- [ ] Built-in agent implementations
- [ ] State management and persistence

### 📋 Next Steps
- [ ] LLM integration layer
- [ ] Agent management API endpoints
- [ ] WebSocket support for real-time communication
- [ ] Comprehensive testing suite
- [ ] Documentation and examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 🔮 Roadmap

- **Phase 1**: Core agent framework and basic orchestration
- **Phase 2**: Advanced LangGraph workflows and subgraphs
- **Phase 3**: Multi-agent collaboration and delegation
- **Phase 4**: Learning and adaptation capabilities
- **Phase 5**: Advanced integrations and ecosystem expansion
#   A g e n t i c - A i - E n g i n e  
 