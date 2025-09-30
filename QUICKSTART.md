# ⚡ Quick Start Guide - Agentic AI System

Get the complete Agentic AI system running in under 3 minutes with a single command!

## 📋 Prerequisites

Before running the setup, make sure you have:

- ✅ **Python 3.11+** installed (Python 3.13.5 works, but 3.11.x recommended)
- ✅ **Docker Desktop** installed and running
- ✅ **Git** (to clone the repository)

## 🚀 One-Command Setup

### **Windows Users**

#### Option 1: PowerShell (Recommended)
```powershell
.\setup.ps1
```

#### Option 2: Command Prompt
```cmd
setup.bat
```

### **Linux/Mac Users**

```bash
chmod +x setup.sh
./setup.sh
```

## 🎯 What the Setup Does

The setup script automatically performs these steps:

1. **Checks Docker** - Verifies Docker Desktop is running
2. **Starts PostgreSQL** - Launches PostgreSQL 17 in a Docker container
3. **Runs Migrations** - Creates all database tables and schemas
4. **Creates Directories** - Sets up the complete data directory structure (20+ directories)
5. **Checks Ollama** - Detects system specs and recommends the best multimodal model
6. **Pulls Model** - Downloads the recommended Ollama model (if Ollama is installed)
7. **Initializes Backend** - Runs backend once to create all data files
8. **Verifies Setup** - Runs health checks to ensure everything works
9. **Updates Config** - Sets the recommended model as default in .env

**Total time: 3-5 minutes** (depending on model download)

## ✅ What You Get

After the setup completes, you'll have:

### **Database**
- ✅ PostgreSQL 17 running in Docker
- ✅ Complete schema with all tables
- ✅ Extensions: uuid-ossp, pg_trgm, btree_gin, btree_gist
- ✅ Schemas: agents, workflows, tools, rag, autonomous
- ✅ Custom types for autonomous agents

### **Data Directories**
```
data/
├── agents/              # Agent runtime files
├── autonomous/          # Autonomous agent persistence
├── cache/               # System caching
├── checkpoints/         # Agent checkpoints
├── chroma/              # Vector database
├── config/              # Configuration files
├── logs/                # Comprehensive logging
├── models/              # AI model storage
├── workflows/           # Workflow definitions
└── ... (20+ more directories)
```

### **System Components**
- ✅ Configuration system loaded
- ✅ Database connection pool ready
- ✅ Migration tracking active
- ✅ All services initialized

## 🎮 Next Steps

### **1. Start the Backend (Optional)**

```bash
python -m app.main
```

The backend will be available at:
- **API**: http://localhost:8888
- **Docs**: http://localhost:8888/docs
- **Health**: http://localhost:8888/health

### **2. Test Standalone Agents**

You can use agents WITHOUT the backend running:

```bash
python scripts/test_agent_standalone.py
```

This demonstrates:
- ✅ Creating agents from Python code
- ✅ Using the database directly
- ✅ Autonomous agent operations
- ✅ RAG system integration

### **3. Create Your First Agent**

#### **Option A: With Backend (REST API)**

```python
import requests

response = requests.post(
    "http://localhost:8888/api/v1/agents/create",
    json={
        "name": "my_first_agent",
        "agent_type": "basic",
        "llm_provider": "ollama",
        "llm_model": "llama3.2:latest"
    }
)

agent = response.json()
print(f"Created agent: {agent['name']}")
```

#### **Option B: Without Backend (Direct Python)**

```python
import asyncio
from app.agents.agent_factory import AgentFactory
from app.models.agent import AgentConfig

async def main():
    # Create agent configuration
    config = AgentConfig(
        name="my_first_agent",
        description="My first autonomous agent",
        agent_type="basic",
        llm_provider="ollama",
        llm_model="llama3.2:latest"
    )
    
    # Create agent
    factory = AgentFactory()
    agent = await factory.create_agent(config)
    
    # Use agent
    response = await agent.process("Hello! What can you do?")
    print(response)

asyncio.run(main())
```

### **4. Access Database Management**

**pgAdmin** is available at: http://localhost:5050

- **Email**: admin@agentic.ai
- **Password**: admin_password_2024

Add server connection:
- **Host**: postgres
- **Port**: 5432
- **Database**: agentic_ai
- **Username**: agentic_user
- **Password**: agentic_secure_password_2024

## 🔧 Management Commands

### **Database**

```bash
# Stop PostgreSQL
docker-compose down

# Restart PostgreSQL
docker-compose restart postgres

# View logs
docker-compose logs -f postgres

# Remove all data (WARNING: Deletes everything!)
docker-compose down -v
```

### **Migrations**

```bash
# Run migrations manually
python db/migrations/run_all_migrations.py

# Check migration status
python db/migrations/migrate_database.py status

# Database health check
python db/migrations/migrate_database.py health
```

### **System**

```bash
# Start backend
python -m app.main

# Run tests
python -m pytest tests/ -v

# Test standalone agents
python scripts/test_agent_standalone.py
```

## 🐛 Troubleshooting

### **Issue: Docker not running**

```
Error: Docker is not running
```

**Solution**: Start Docker Desktop and wait for it to fully start, then run setup again.

### **Issue: Port 5432 already in use**

```
Error: Port 5432 is already allocated
```

**Solution**: 
1. Stop any existing PostgreSQL instances
2. Or change the port in `docker-compose.yml`

### **Issue: Python not found**

```
Error: Python is not installed
```

**Solution**: Install Python 3.11+ from https://www.python.org/

### **Issue: Migration warnings**

```
Warning: SQL statement warning: InFailedSQLTransactionError
```

**Solution**: These warnings are NORMAL! They occur when creating objects that already exist. As long as the setup completes successfully, everything is fine.

### **Issue: Permission denied (Linux/Mac)**

```
Error: Permission denied: ./setup.sh
```

**Solution**: Make the script executable:
```bash
chmod +x setup.sh
./setup.sh
```

## 📚 Learn More

- **Full Setup Guide**: See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed information
- **Documentation**: Check `docs/system-documentation/` for comprehensive guides
- **Examples**: Look at `examples/` for agent usage examples
- **API Docs**: Visit http://localhost:8888/docs when backend is running

## 🎯 Summary

You now have a complete, production-ready agentic AI system with:

- ✅ **Database**: PostgreSQL 17 with full schema
- ✅ **Data Directories**: Complete file structure
- ✅ **Configuration**: Environment and settings loaded
- ✅ **Agents**: Ready to create and run agents
- ✅ **RAG System**: Vector database and knowledge management
- ✅ **Autonomous Agents**: BDI architecture with learning
- ✅ **Workflows**: Multi-agent orchestration
- ✅ **Tools**: Extensible tool system

**You can now build truly autonomous AI agents!** 🚀

## 💡 Pro Tips

1. **Use without backend**: You don't need the FastAPI backend running to use agents - they work standalone!

2. **Configuration**: Customize settings in `.env` and `data/config/user_config.yaml`

3. **Logging**: Check `data/logs/` for detailed system and agent logs

4. **Database**: Use pgAdmin to explore the database schema and data

5. **Testing**: Run `python scripts/test_agent_standalone.py` to verify everything works

## 🆘 Need Help?

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs in `data/logs/`
3. Check Docker logs: `docker-compose logs postgres`
4. Verify Python dependencies: `pip install -r requirements.txt`
5. Read the full documentation in `docs/`

---

**Happy building!** 🎉

