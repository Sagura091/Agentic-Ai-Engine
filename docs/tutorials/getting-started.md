# Getting Started with Agentic AI Engine

Welcome! This tutorial will guide you through installing and running the Agentic AI Engine for the first time.

## 📋 What You'll Learn

By the end of this tutorial, you will:

- ✅ Install and set up the Agentic AI Engine
- ✅ Start the PostgreSQL database
- ✅ Run database migrations
- ✅ Start the FastAPI backend
- ✅ Create your first AI agent
- ✅ Make your first API call

**Estimated Time:** 15-20 minutes

## 🎯 Prerequisites

Before starting, make sure you have:

- **Python 3.11+** installed ([Download Python](https://www.python.org/downloads/))
- **Docker Desktop** installed and running ([Download Docker](https://www.docker.com/products/docker-desktop))
- **Git** installed ([Download Git](https://git-scm.com/downloads))
- **8GB+ RAM** (recommended)
- **Basic command line knowledge**

### Verify Prerequisites

```bash
# Check Python version (should be 3.11+)
python --version

# Check Docker is running
docker --version
docker ps

# Check Git
git --version
```

## 📥 Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Sagura091/Agentic-Ai-Engine.git

# Navigate to the project directory
cd Agentic-Ai-Engine
```

## 🐘 Step 2: Start PostgreSQL Database

The system uses PostgreSQL for structured data storage.

```bash
# Start PostgreSQL container
docker-compose up -d postgres

# Verify PostgreSQL is running
docker-compose ps
```

You should see output showing the `postgres` container is running.

### Wait for PostgreSQL to be Ready

```bash
# Check PostgreSQL logs
docker-compose logs -f postgres

# Look for this message:
# "database system is ready to accept connections"

# Press Ctrl+C to exit logs
```

## 📦 Step 3: Install Python Dependencies

```bash
# Install all required Python packages
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- LangChain/LangGraph (agent framework)
- SQLAlchemy (database ORM)
- ChromaDB (vector database)
- And many more dependencies

**Note:** This may take 5-10 minutes depending on your internet connection.

## 🗄️ Step 4: Run Database Migrations

Create all necessary database tables and schemas:

```bash
# Run migrations
python db/migrations/migrate_database.py migrate
```

You should see output indicating successful migration:
```
✅ Migration completed successfully
✅ All tables created
```

## 🚀 Step 5: Start the Application

```bash
# Start the FastAPI backend
python -m app.main
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

**The API is now running at:** `http://localhost:8888`

## ✅ Step 6: Verify Installation

Open a new terminal (keep the server running) and test the API:

```bash
# Check health endpoint
curl http://localhost:8888/health

# Expected response:
# {"status":"healthy","database":"connected","version":"0.1.0"}
```

### View API Documentation

Open your browser and visit:
- **Swagger UI:** http://localhost:8888/docs
- **ReDoc:** http://localhost:8888/redoc

You should see the interactive API documentation!

## 🤖 Step 7: Create Your First Agent

Let's create a simple agent using the API.

### Using curl:

```bash
curl -X POST "http://localhost:8888/api/v1/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_first_agent",
    "agent_type": "react",
    "description": "My first AI agent",
    "system_prompt": "You are a helpful AI assistant.",
    "llm_config": {
      "provider": "ollama",
      "model_id": "llama3.2:latest",
      "temperature": 0.7
    }
  }'
```

### Using Python:

```python
import requests

# Create agent
response = requests.post(
    "http://localhost:8888/api/v1/agents",
    json={
        "name": "my_first_agent",
        "agent_type": "react",
        "description": "My first AI agent",
        "system_prompt": "You are a helpful AI assistant.",
        "llm_config": {
            "provider": "ollama",
            "model_id": "llama3.2:latest",
            "temperature": 0.7
        }
    }
)

agent = response.json()
print(f"Created agent: {agent['id']}")
```

**Expected Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "my_first_agent",
  "agent_type": "react",
  "status": "active",
  "created_at": "2025-10-09T08:00:00Z"
}
```

## 💬 Step 8: Chat with Your Agent

Now let's send a message to your agent:

```bash
# Replace AGENT_ID with your agent's ID from Step 7
curl -X POST "http://localhost:8888/api/v1/agents/AGENT_ID/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Can you help me understand what you can do?"
  }'
```

You should receive a response from your agent!

## 🎉 Congratulations!

You've successfully:
- ✅ Installed the Agentic AI Engine
- ✅ Started the database and backend
- ✅ Created your first AI agent
- ✅ Sent your first message

## 🔍 What's Next?

Now that you have the basics working, explore more:

1. **[Build Your First Agent](first-agent.md)** - Learn to create custom agents
2. **[RAG System Basics](rag-basics.md)** - Add knowledge retrieval to your agents
3. **[Configuration Guide](../guides/configuration.md)** - Customize the system
4. **[API Reference](../reference/API_SYSTEM_DOCUMENTATION.md)** - Explore all API endpoints

## 🛠️ Troubleshooting

### PostgreSQL Won't Start

```bash
# Check if port 5432 is already in use
docker ps -a

# Stop any existing PostgreSQL containers
docker-compose down

# Start fresh
docker-compose up -d postgres
```

### Python Dependencies Fail to Install

```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing again
pip install -r requirements.txt
```

### Application Won't Start

```bash
# Check if port 8888 is already in use
# On Windows:
netstat -ano | findstr :8888

# On Linux/Mac:
lsof -i :8888

# Kill the process using the port or change the port in .env
```

### Agent Creation Fails

Make sure:
- PostgreSQL is running (`docker-compose ps`)
- Migrations completed successfully
- You're using a valid LLM provider (Ollama, OpenAI, etc.)

## 📚 Additional Resources

- **[Installation Guide](../guides/FIRST_TIME_SETUP.md)** - Detailed installation instructions
- **[Development Guide](../guides/development.md)** - Set up development environment
- **[Error Handling Guide](../guides/error-handling.md)** - Common errors and solutions
- **[Contributing Guide](../../CONTRIBUTING.md)** - Contribute to the project

## 💡 Tips

- **Keep the server running** - The FastAPI server needs to stay running to handle requests
- **Use the Swagger UI** - http://localhost:8888/docs is great for testing APIs
- **Check the logs** - The server logs show what's happening
- **Start simple** - Begin with basic agents before adding complexity

## 🤝 Need Help?

- **Documentation:** Check the [docs](../README.md)
- **Issues:** Report bugs on [GitHub Issues](https://github.com/Sagura091/Agentic-Ai-Engine/issues)
- **Community:** Join our community discussions

---

**Next Tutorial:** [Build Your First Agent](first-agent.md) →

