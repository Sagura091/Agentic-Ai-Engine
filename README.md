# Agentic AI Engine

> A multi-agent AI framework built on LangChain/LangGraph with FastAPI, featuring autonomous agents, RAG capabilities, and comprehensive tool integration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://github.com/langchain-ai/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Project Status](#project-status)
- [What This Project Is](#what-this-project-is)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🚦 Project Status

**Version:** 0.1.0 (Beta)
**Status:** Active Development
**Stability:** Core features functional, testing in progress
**Production Ready:** Not yet - See [Roadmap](#roadmap)

## 💡 What This Project Is

Agentic AI Engine is a Python-based multi-agent framework that provides:

- **Multi-Agent Support**: Create and manage multiple AI agents with different capabilities
- **LangChain/LangGraph Integration**: Built on industry-standard agent frameworks
- **RAG System**: Retrieval-Augmented Generation with ChromaDB vector storage
- **Memory Management**: Persistent agent memory across sessions
- **Tool Ecosystem**: Extensible tool system for agent capabilities
- **FastAPI Backend**: RESTful API with WebSocket support
- **Database Integration**: PostgreSQL for structured data, ChromaDB for vectors

## ⚡ Quick Start

### Prerequisites

- **Docker & Docker Compose** (required for PostgreSQL)
- **Python 3.11+** (for development)
- **8GB+ RAM** (recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sagura091/Agentic-Ai-Engine.git
cd Agentic-Ai-Engine

# 2. Start PostgreSQL
docker-compose up -d postgres

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run database migrations
python db/migrations/migrate_database.py migrate

# 5. Start the application
python -m app.main
```

The API will be available at `http://localhost:8888`

### Verify Installation

```bash
# Check health endpoint
curl http://localhost:8888/health

# View API documentation
open http://localhost:8888/docs
```

## 🎯 Core Features

### 1. Agent Framework

Create agents with different capabilities:

- **Basic Agents**: Simple task-oriented agents for straightforward operations
- **ReAct Agents**: Reasoning and Acting pattern with thought/action cycles
- **Autonomous Agents**: Self-directed agents with goal management and learning

### 2. Memory System

Agents maintain context through multiple memory types:

- **Short-term Memory**: Active conversation context
- **Long-term Memory**: Persistent knowledge across sessions
- **Episodic Memory**: Experience tracking and learning from past interactions
- **Working Memory**: Immediate task processing

### 3. RAG (Retrieval-Augmented Generation)

- **ChromaDB Integration**: Vector storage for semantic search
- **Document Processing**: Automatic chunking and embedding
- **Knowledge Bases**: Agent-specific and shared knowledge collections
- **Semantic Search**: Context-aware information retrieval

### 4. Tool System

Extensible tool framework with production-ready tools:

- **Web Research**: Web scraping and content extraction
- **Document Processing**: PDF, Word, Excel, and text file handling
- **Database Operations**: SQL query execution and data management
- **File System**: File and directory operations
- **API Integration**: HTTP client for external API calls

### 5. API & Integration

- **RESTful API**: FastAPI-based endpoints with automatic documentation
- **WebSocket Support**: Real-time bidirectional communication
- **OpenAPI/Swagger**: Interactive API documentation
- **OpenWebUI Integration**: Pipeline support for OpenWebUI

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory, organized by purpose:

### 🎓 [Tutorials](docs/tutorials/) - Learning-Oriented

Step-by-step guides for beginners:

- **[Getting Started](docs/tutorials/getting-started.md)** - Complete beginner tutorial from installation to first agent
- **[Build Your First Agent](docs/tutorials/first-agent.md)** - Create a custom AI agent step-by-step
- **[RAG System Basics](docs/tutorials/rag-basics.md)** - Learn how to use the RAG system

### 🛠️ [How-to Guides](docs/guides/) - Problem-Oriented

Practical guides for specific tasks:

- **[Installation Guide](docs/guides/FIRST_TIME_SETUP.md)** - Detailed installation instructions
- **[Configuration Guide](docs/guides/configuration.md)** - System configuration
- **[Deployment Guide](docs/guides/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[Development Guide](docs/guides/development.md)** - Development environment setup
- **[Error Handling Guide](docs/guides/error-handling.md)** - Error handling best practices
- **[Performance Guide](docs/guides/performance.md)** - Performance optimization
- **[RAG Quick Start](docs/guides/RAG_SYSTEM_QUICK_START.md)** - Quick start for RAG system

### 📖 [Reference](docs/reference/) - Information-Oriented

Technical reference documentation:

- **[Agents System](docs/reference/AGENTS_SYSTEM_DOCUMENTATION.md)** - Agent framework reference
- **[API Reference](docs/reference/API_SYSTEM_DOCUMENTATION.md)** - Complete API documentation
- **[Database](docs/reference/DATABASE_SYSTEM_DOCUMENTATION.md)** - Database schema and operations
- **[RAG System](docs/reference/RAG_SYSTEM_DOCUMENTATION.md)** - RAG system reference
- **[Tools](docs/reference/TOOLS_SYSTEM_DOCUMENTATION.md)** - Agent tools reference
- **[Memory System](docs/reference/MEMORY_SYSTEM_DOCUMENTATION.md)** - Memory system reference
- **[LLM System](docs/reference/LLM_SYSTEM_DOCUMENTATION.md)** - LLM provider integration
- **[And more...](docs/reference/)** - See all reference documentation

### 💡 [Explanation](docs/explanation/) - Understanding-Oriented

Conceptual discussions to deepen understanding:

- **[Architecture Overview](docs/explanation/architecture.md)** - System architecture and design
- **[Design Decisions](docs/explanation/design-decisions.md)** - Why we made certain choices
- **[Agent Workflows](docs/explanation/workflows.md)** - How agents work internally

### 📂 Special Topics

- **[Logging System](docs/logging/)** - Comprehensive logging documentation
- **[Architecture](docs/architecture/)** - Detailed architecture documentation
- **[API Contracts](docs/api/)** - API contract specifications

**👉 Start here:** [Documentation Index](docs/README.md)

## 🏗️ Architecture

### System Overview

```text
┌─────────────────────────────────────────┐
│         FastAPI Application             │
├─────────────────────────────────────────┤
│  API Endpoints  │  WebSocket Manager    │
├─────────────────────────────────────────┤
│         Agent Management                │
│  ├── Agent Factory                      │
│  ├── Agent Registry                     │
│  └── Agent Execution                    │
├─────────────────────────────────────────┤
│  Memory System  │  RAG System           │
│  ├── Short-term │  ├── ChromaDB         │
│  └── Long-term  │  └── Embeddings       │
├─────────────────────────────────────────┤
│  Tool Repository                        │
│  └── Production Tools                   │
├─────────────────────────────────────────┤
│         Data Layer                      │
│  ├── PostgreSQL (Structured)            │
│  ├── ChromaDB (Vectors)                 │
│  └── Redis (Cache)                      │
└─────────────────────────────────────────┘
```

### Core Components

#### Agent System

- **Agent Factory**: Creates and configures agents based on type
- **Agent Registry**: Manages active agent instances
- **Agent Execution**: Handles agent task execution and lifecycle

#### Memory & Knowledge

- **Memory System**: Manages short-term and long-term agent memory
- **RAG System**: Provides semantic search and knowledge retrieval
- **ChromaDB**: Vector database for embeddings and similarity search

#### Tools & Integration

- **Tool Repository**: Centralized tool management and execution
- **LLM Providers**: Support for Ollama, OpenAI, and other providers
- **API Layer**: RESTful and WebSocket endpoints

## ⚙️ Configuration

Configuration is managed through environment variables and `.env` files.

### Environment Variables

Create a `.env` file in the project root:

```env
# Server Configuration
AGENTIC_HOST=0.0.0.0
AGENTIC_PORT=8888
AGENTIC_DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/agentic_ai

# LLM Providers
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=your-key-here

# Security
SECRET_KEY=your-secret-key-change-this-in-production
JWT_SECRET_KEY=your-jwt-secret-change-this

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Redis (optional)
REDIS_URL=redis://localhost:6379
```

See `.env.example` for all available configuration options.

### Agent Configuration

Agents can be configured via YAML files in `data/config/agents/`:

```yaml
agent:
  name: "Research Assistant"
  type: "react"
  model: "llama3.1:8b"

tools:
  - "web_research"
  - "document_search"

memory:
  enable_learning: true
  memory_types: ["episodic", "semantic"]
```

## 💻 Development

### Project Structure

```text
app/
├── agents/          # Agent implementations
│   ├── base/        # Base agent classes
│   ├── react/       # ReAct agents
│   └── autonomous/  # Autonomous agents
├── api/             # FastAPI endpoints
│   └── v1/          # API version 1
├── config/          # Configuration management
├── core/            # Core utilities and orchestration
├── memory/          # Memory systems
├── models/          # Database models
├── rag/             # RAG implementation
├── services/        # Business logic
└── tools/           # Tool implementations

tests/               # Test suite
docs/                # Documentation
scripts/             # Utility scripts
db/                  # Database migrations
```

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_logging_system.py -v
```

**Note:** Test coverage is currently being expanded.

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Type checking
mypy app/

# Linting
flake8 app/
```

## 📚 API Documentation

Once the application is running, visit:

- **Interactive Docs (Swagger)**: http://localhost:8888/docs
- **ReDoc**: http://localhost:8888/redoc
- **OpenAPI JSON**: http://localhost:8888/openapi.json

### Example API Usage

#### Create an Agent

```python
import httpx

response = httpx.post("http://localhost:8888/api/v1/agents", json={
    "name": "Research Assistant",
    "type": "react",
    "model": "llama3.1:8b",
    "tools": ["web_research", "document_search"]
})

agent_id = response.json()["id"]
```

#### Execute a Task

```python
response = httpx.post(
    f"http://localhost:8888/api/v1/agents/{agent_id}/execute",
    json={"task": "Research recent developments in AI agents"}
)

result = response.json()["result"]
print(result)
```

#### WebSocket Communication

```python
import websockets
import json

async with websockets.connect("ws://localhost:8888/ws") as websocket:
    await websocket.send(json.dumps({
        "type": "agent_message",
        "agent_id": agent_id,
        "message": "Hello, agent!"
    }))

    response = await websocket.recv()
    print(json.loads(response))
```

## 🐳 Deployment

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

**Before deploying to production, ensure you:**

- [ ] Change all default secret keys
- [ ] Configure proper database credentials
- [ ] Set up SSL/TLS certificates
- [ ] Configure rate limiting
- [ ] Set up monitoring and logging
- [ ] Review security settings
- [ ] Run security audit
- [ ] Complete load testing
- [ ] Set up backup and recovery

See [docs/guides/DEPLOYMENT_GUIDE.md](docs/guides/DEPLOYMENT_GUIDE.md) for detailed instructions.

### Environment-Specific Configuration

```bash
# Development
export AGENTIC_DEBUG=true
python -m app.main

# Staging
export AGENTIC_DEBUG=false
export AGENTIC_ENV=staging
python -m app.main

# Production
export AGENTIC_DEBUG=false
export AGENTIC_ENV=production
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

## 🗺️ Roadmap

### Current Focus (v0.1.x)

- [ ] Expand test coverage to 70%+
- [ ] Simplify architecture and reduce code duplication
- [ ] Complete API documentation with examples
- [ ] Performance optimization and profiling
- [ ] Security hardening and audit

### Planned Features (v0.2.x)

- [ ] Enhanced monitoring with Prometheus/Grafana
- [ ] Multi-agent collaboration improvements
- [ ] Additional LLM provider support (Anthropic, Cohere)
- [ ] Workflow orchestration enhancements
- [ ] Production deployment templates (Kubernetes, AWS, GCP)

### Future Considerations (v0.3.x+)

- [ ] Distributed agent execution across multiple nodes
- [ ] Advanced learning capabilities and model fine-tuning
- [ ] Multi-modal agent support (vision, audio)
- [ ] Enterprise features (RBAC, audit logs, compliance)
- [ ] Agent marketplace and sharing platform

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

## 📝 Known Limitations

- Test coverage is currently limited (being actively expanded)
- Some advanced features are in development
- Performance optimization is ongoing
- Documentation is being improved continuously
#   A g e n t i c - A i - E n g i n e 
 
 
## 🎮 **Getting Started Examples**

### **Create Your First Agent in 30 Seconds**

```bash
# 1. Copy a template
cp templates/research_agent_template.py my_research_agent.py

# 2. Customize the config (optional)
# Edit the AGENT_CONFIG section at the top

# 3. Run your agent
python my_research_agent.py

# Your agent is now live with:
# ✅ Advanced RAG capabilities
# ✅ 8-type memory system
# ✅ Production tool access
# ✅ Autonomous operation
# ✅ Real-time learning
```

### **Create an Autonomous Trading Agent**

```yaml
# Save as: data/config/agents/trading_agent.yaml
agent:
  name: "Autonomous Trading Agent"
  framework: "bdi"
  autonomy_level: "autonomous"

tools:
  - "advanced_stock_trading"
  - "business_intelligence"

goals:
  - "Monitor market conditions"
  - "Execute profitable trades"
  - "Learn from outcomes"
```

```bash
# Agent automatically starts and operates 24/7
python -m app.agents.autonomous_agent --config trading_agent.yaml
```

## 🤝 **Community & Support**

### **📚 Documentation**
- **[Complete System Documentation](docs/system-documentation/)** - Comprehensive guides
- **[Architecture Diagrams](docs/architecture/)** - Visual system overview
- **[API Documentation](docs/api/)** - Complete API reference
- **[Setup Guides](docs/guides/)** - Step-by-step installation

### **🛠️ Development**
- **GitHub Repository**: [Agentic-Ai-Engine](https://github.com/Sagura091/Agentic-Ai-Engine)
- **Issues & Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Discussions
- **Community Chat**: Discord Server (Coming Soon)

### **🚀 Contributing**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 **Ready to Build the Future of AI?**

**The Agentic AI Engine isn't just a tool - it's a revolution in autonomous AI systems.**

Start building your autonomous agent empire today:

```bash
git clone https://github.com/Sagura091/Agentic-Ai-Engine.git
cd Agentic-Ai-Engine
docker-compose up --build -d
```

**Welcome to the future of autonomous AI. Welcome to the Agentic AI Engine.** 🚀

---

*Built with ❤️ by the Agentic AI Team | Powered by the most advanced multi-agent architecture ever created*