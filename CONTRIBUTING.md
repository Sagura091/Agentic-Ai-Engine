# Contributing to Agentic AI Engine

Thank you for your interest in contributing to the Agentic AI Engine! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)

## 🤝 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful** - Treat everyone with respect and kindness
- **Be collaborative** - Work together to improve the project
- **Be constructive** - Provide helpful feedback and suggestions
- **Be inclusive** - Welcome contributors of all backgrounds and skill levels

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.11+** installed
- **Docker Desktop** running (for PostgreSQL)
- **Git** for version control
- **A GitHub account**

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Agentic-Ai-Engine.git
   cd Agentic-Ai-Engine
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/Sagura091/Agentic-Ai-Engine.git
   ```

## 💻 Development Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available
```

### 2. Start PostgreSQL

```bash
# Start PostgreSQL container
docker-compose up -d postgres
```

### 3. Run Database Migrations

```bash
# Run migrations
python db/migrations/migrate_database.py migrate
```

### 4. Verify Setup

```bash
# Start the application
python -m app.main

# In another terminal, check health
curl http://localhost:8888/health
```

## 🛠️ How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes** - Fix issues and improve stability
- **New features** - Add new capabilities to the system
- **Documentation** - Improve or add documentation
- **Tests** - Add or improve test coverage
- **Performance** - Optimize code performance
- **Refactoring** - Improve code quality and structure

### Finding Issues to Work On

- Check the [Issues](https://github.com/Sagura091/Agentic-Ai-Engine/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it

### Creating a New Issue

Before creating a new issue:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** if available
3. **Provide clear descriptions** with steps to reproduce (for bugs)
4. **Include relevant information** (OS, Python version, error messages)

## 📝 Code Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications:

- **Line length**: 120 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized in groups (standard library, third-party, local)
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Example:

```python
from typing import Dict, List, Optional

import structlog
from pydantic import BaseModel

from app.models.agent import Agent

logger = structlog.get_logger(__name__)


class AgentService:
    """
    Service for managing AI agents.
    
    This service provides methods for creating, updating, and managing
    AI agents in the system.
    
    Attributes:
        db: Database session
        cache: Redis cache instance
    """
    
    def __init__(self, db, cache):
        """Initialize the agent service."""
        self.db = db
        self.cache = cache
    
    async def create_agent(
        self,
        name: str,
        agent_type: str,
        config: Optional[Dict] = None
    ) -> Agent:
        """
        Create a new AI agent.
        
        Args:
            name: Agent name
            agent_type: Type of agent (e.g., "react", "autonomous")
            config: Optional configuration dictionary
            
        Returns:
            Created agent instance
            
        Raises:
            ValueError: If agent_type is invalid
        """
        logger.info("Creating agent", name=name, agent_type=agent_type)
        # Implementation here
        pass
```

### Code Formatting

We use **Black** for code formatting:

```bash
# Format code
black app/ tests/

# Check formatting
black --check app/ tests/
```

### Linting

We use **Ruff** for linting:

```bash
# Run linter
ruff check app/ tests/

# Fix auto-fixable issues
ruff check --fix app/ tests/
```

## 🔄 Pull Request Process

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, well-documented code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for custom agent tools"
git commit -m "Fix memory leak in RAG system"
git commit -m "Update documentation for deployment"

# Bad commit messages (avoid these)
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Create a Pull Request

1. Go to the [repository](https://github.com/Sagura091/Agentic-Ai-Engine)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template with:
   - **Description** of changes
   - **Related issues** (if any)
   - **Testing** performed
   - **Screenshots** (if UI changes)

### 6. Code Review

- Address reviewer feedback promptly
- Make requested changes in new commits
- Keep the PR focused on a single feature/fix
- Be patient and respectful during review

### 7. Merge

Once approved, a maintainer will merge your PR. Thank you for contributing!

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=app tests/
```

### Writing Tests

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example:

```python
import pytest
from app.services.agent_service import AgentService


class TestAgentService:
    """Tests for AgentService."""
    
    @pytest.fixture
    def agent_service(self, db_session, redis_cache):
        """Create agent service instance."""
        return AgentService(db_session, redis_cache)
    
    async def test_create_agent_success(self, agent_service):
        """Test successful agent creation."""
        # Arrange
        name = "test_agent"
        agent_type = "react"
        
        # Act
        agent = await agent_service.create_agent(name, agent_type)
        
        # Assert
        assert agent.name == name
        assert agent.agent_type == agent_type
        assert agent.id is not None
```

## 📚 Documentation Guidelines

### Documentation Structure

Follow the [Divio Documentation System](https://documentation.divio.com/):

- **Tutorials** - Learning-oriented, step-by-step guides
- **How-to Guides** - Problem-oriented, specific tasks
- **Reference** - Information-oriented, technical descriptions
- **Explanation** - Understanding-oriented, conceptual discussions

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep documentation up-to-date with code changes

### Documentation Locations

- **Tutorials**: `docs/tutorials/`
- **How-to Guides**: `docs/guides/`
- **Reference**: `docs/reference/`
- **Explanation**: `docs/explanation/`

## 🎯 Project Structure

Understanding the project structure helps you contribute effectively:

```
Agentic-Ai-Engine/
├── app/                    # Main application code
│   ├── agents/            # Agent implementations
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Core system components
│   ├── models/            # Database models
│   ├── rag/               # RAG system
│   ├── services/          # Business logic services
│   └── tools/             # Agent tools
├── docs/                  # Documentation
├── tests/                 # Test suite
├── db/                    # Database migrations
└── data/                  # Data directory
```

## 🙏 Thank You!

Thank you for contributing to the Agentic AI Engine! Your contributions help make this project better for everyone.

If you have questions, feel free to:
- Open an issue
- Join our community discussions
- Reach out to the maintainers

Happy coding! 🚀

