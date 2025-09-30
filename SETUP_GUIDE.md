# 🚀 Agentic AI System - Complete Setup Guide

This guide will help you set up the complete Agentic AI system with database, data directories, and all necessary components.

## 📋 Prerequisites

- **Python 3.11+** (Python 3.13.5 works but 3.11.x recommended for better package compatibility)
- **Docker Desktop** (for PostgreSQL)
- **Git** (for version control)
- **Ollama** (optional, for local LLM inference)

## 🎯 Quick Start (Recommended)

### **Option 1: Automated Setup (Windows PowerShell)**

Run the enhanced setup script that does everything automatically:

```powershell
# Run the complete setup script
.\scripts\start-postgres.ps1
```

This script will:
1. ✅ Check Docker is running
2. ✅ Create all data directories (20+ directories)
3. ✅ Start PostgreSQL 17 in Docker
4. ✅ Run all database migrations
5. ✅ Initialize the system
6. ✅ Verify everything works

**Total time: ~2-3 minutes**

### **Option 2: Manual Setup**

If you prefer to run each step manually:

#### **Step 1: Start PostgreSQL**

```powershell
# Start PostgreSQL container
docker-compose up -d postgres

# Wait for it to be ready (check logs)
docker-compose logs -f postgres
# Look for: "database system is ready to accept connections"
# Press Ctrl+C to exit logs
```

#### **Step 2: Run Database Migrations**

```powershell
# Run all migrations
python db/migrations/run_all_migrations.py

# Verify migrations
python db/migrations/migrate_database.py status
```

#### **Step 3: Create Data Directories**

```powershell
# Test system initialization (creates directories)
python scripts/test_agent_standalone.py
```

## 📁 What Gets Created

### **Database Tables**
- ✅ **Extensions**: uuid-ossp, pg_trgm, btree_gin, btree_gist
- ✅ **Schemas**: agents, workflows, tools, rag, autonomous
- ✅ **Custom Types**: autonomy_level, goal_type, memory_type, etc.
- ✅ **Core Tables**: agents, workflows, tools, migration_history
- ✅ **Autonomous Tables**: autonomous_agent_states, autonomous_goals, autonomous_decisions, agent_memories, learning_experiences
- ✅ **Auth Tables**: users, conversations, messages, user_sessions
- ✅ **Knowledge Base Tables**: knowledge_bases, knowledge_base_access, documents, document_chunks

### **Data Directories**
```
data/
├── agents/              # Agent runtime files
├── autonomous/          # Autonomous agent persistence
├── cache/               # System-wide caching
├── checkpoints/         # Agent checkpoints
├── chroma/              # ChromaDB vector database
├── config/              # Configuration files
├── downloads/           # Downloaded files
├── generated_files/     # AI-generated documents
├── logs/                # Comprehensive logging
│   ├── agents/          # Agent-specific logs
│   └── backend/         # Backend system logs
├── memes/               # Meme generation
├── models/              # AI model storage
│   ├── embedding/
│   ├── llm/
│   ├── reranking/
│   └── vision/
├── outputs/             # System outputs
├── screenshots/         # Screenshot storage
├── session_documents/   # Session management
├── session_vectors/     # Session vector storage
├── templates/           # Document templates
├── temp/                # Temporary files
├── uploads/             # File uploads
└── workflows/           # Workflow definitions
```

## 🧪 Verify Setup

### **Test 1: Database Connection**

```powershell
# Check database tables
docker exec -it agentic-postgres psql -U agentic_user -d agentic_ai -c "\dt"

# Check schemas
docker exec -it agentic-postgres psql -U agentic_user -d agentic_ai -c "\dn"
```

### **Test 2: System Initialization**

```powershell
# Run standalone test
python scripts/test_agent_standalone.py
```

This will test:
- ✅ Configuration system
- ✅ Data directories
- ✅ Database connection
- ✅ Agent creation
- ✅ Autonomous persistence

### **Test 3: Start Backend (Optional)**

```powershell
# Start the FastAPI backend
python -m app.main

# Backend will be available at: http://localhost:8888
# API docs at: http://localhost:8888/docs
```

## 🤖 Using Agents

### **Option 1: With Backend Running**

```python
# Use the REST API
import requests

response = requests.post(
    "http://localhost:8888/api/v1/agents/create",
    json={
        "name": "my_agent",
        "agent_type": "basic",
        "llm_provider": "ollama",
        "llm_model": "llama3.2:latest"
    }
)
```

### **Option 2: Without Backend (Standalone)**

```python
import asyncio
from app.agents.agent_factory import AgentFactory
from app.models.agent import AgentConfig

async def main():
    # Create agent configuration
    config = AgentConfig(
        name="my_agent",
        description="My custom agent",
        agent_type="basic",
        llm_provider="ollama",
        llm_model="llama3.2:latest"
    )
    
    # Create agent
    factory = AgentFactory()
    agent = await factory.create_agent(config)
    
    # Use agent
    response = await agent.process("What is 2+2?")
    print(response)

asyncio.run(main())
```

### **Option 3: Autonomous Agents**

```python
import asyncio
from app.agents.autonomous_agent import AutonomousAgent

async def main():
    # Create autonomous agent
    agent = AutonomousAgent(
        name="autonomous_agent",
        autonomy_level="adaptive",
        learning_enabled=True
    )
    
    # Agent can set its own goals
    await agent.set_goal(
        title="Learn about Python",
        description="Research and learn Python best practices"
    )
    
    # Agent executes autonomously
    await agent.execute_autonomous_cycle()

asyncio.run(main())
```

## 🔧 Configuration

### **Environment Variables (.env)**

Key settings you might want to customize:

```bash
# Environment
AGENTIC_ENVIRONMENT=development
AGENTIC_LOG_LEVEL=DEBUG

# Server
AGENTIC_HOST=localhost
AGENTIC_PORT=8888

# LLM Provider
AGENTIC_DEFAULT_AGENT_MODEL=llama3.1:8b
AGENTIC_DEFAULT_AGENT_PROVIDER=ollama

# Database
AGENTIC_DATABASE_URL=postgresql://agentic_user:agentic_secure_password_2024@localhost:5432/agentic_ai
```

### **Data Configuration (data/config/)**

- `agent_defaults.yaml` - Default settings for all agents
- `user_config.yaml` - Your custom overrides
- `agents/*.yaml` - Agent-specific configurations

## 🛠️ Management Commands

### **Database**

```powershell
# Start PostgreSQL
docker-compose up -d postgres

# Stop PostgreSQL
docker-compose down

# Remove all data (WARNING: Deletes everything!)
docker-compose down -v

# Run migrations
python db/migrations/run_all_migrations.py

# Check migration status
python db/migrations/migrate_database.py status

# Database health check
python db/migrations/migrate_database.py health
```

### **System**

```powershell
# Start backend
python -m app.main

# Run tests
python -m pytest tests/ -v

# Test standalone agents
python scripts/test_agent_standalone.py
```

## 📊 Database Management

### **pgAdmin (Web UI)**

Access pgAdmin at: http://localhost:5050

- **Email**: admin@agentic.ai
- **Password**: admin_password_2024

Add server connection:
- **Host**: postgres (or localhost if accessing from host)
- **Port**: 5432
- **Database**: agentic_ai
- **Username**: agentic_user
- **Password**: agentic_secure_password_2024

### **Command Line**

```powershell
# Connect to database
docker exec -it agentic-postgres psql -U agentic_user -d agentic_ai

# List tables
\dt

# List schemas
\dn

# Describe table
\d table_name

# Run query
SELECT * FROM migration_history;
```

## 🐛 Troubleshooting

### **Issue: Docker not running**
```
Error: Docker is not running
```
**Solution**: Start Docker Desktop

### **Issue: PostgreSQL not ready**
```
Error: PostgreSQL failed to start within timeout
```
**Solution**: 
1. Check Docker logs: `docker-compose logs postgres`
2. Restart container: `docker-compose restart postgres`
3. Check port 5432 is not in use

### **Issue: Migration warnings**
```
Warning: SQL statement warning: InFailedSQLTransactionError
```
**Solution**: These warnings are normal! They occur when trying to create things that already exist. As long as the migration completes (exit code 0), everything is fine.

### **Issue: Column does not exist**
```
Error: column "execution_time_ms" does not exist
```
**Solution**: Add the missing column:
```powershell
docker exec -it agentic-postgres psql -U agentic_user -d agentic_ai -c "ALTER TABLE migration_history ADD COLUMN IF NOT EXISTS execution_time_ms INTEGER;"
```

## 📚 Next Steps

1. **Read the Documentation**: Check `docs/system-documentation/` for detailed guides
2. **Explore Examples**: Look at `examples/` for agent usage examples
3. **Run Tests**: Execute `python -m pytest tests/ -v` to see the system in action
4. **Create Your First Agent**: Use the examples above to create custom agents
5. **Join the Community**: Contribute and share your agents!

## 🎯 Summary

After running the setup, you have:
- ✅ PostgreSQL 17 running in Docker
- ✅ Complete database schema with all tables
- ✅ All data directories created
- ✅ System initialized and tested
- ✅ Ready to create and run agents
- ✅ Can work with or without the backend

**You're ready to build truly agentic AI systems!** 🚀

