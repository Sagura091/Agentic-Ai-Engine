# 🚀 Automatic System Initialization

## Overview

The Agentic AI Engine now features **automatic system initialization** that ensures all required models and resources are available before any operation begins.

This system runs automatically at:
- ✅ **Backend startup** (`python -m app.main`)
- ✅ **Agent startup** (when running any agent)
- ✅ **Setup script** (`python setup_system.py`)

---

## What Gets Initialized

### 1. **Embedding Models**
- **Default Model**: `all-MiniLM-L6-v2` (384-dimensional, fast, general-purpose)
- **Vision Model**: `clip-vit-base-patch32` (512-dimensional, image-text understanding)
- **Storage**: `data/models/embedding/` and `data/models/vision/`

### 2. **Ollama LLM Models** (Optional)
- **Default Model**: `llama3.2-vision:11b` (or configured default)
- **Only if Ollama is installed** - gracefully skips if not available
- **Checks existing models** - only pulls if missing

### 3. **System Configuration**
- Creates initialization status file: `data/.initialization_status.json`
- Tracks which models are downloaded
- Records timestamps and versions

---

## How It Works

### Smart Model Detection

The system is **intelligent** and **efficient**:

1. **Checks HuggingFace Cache First**
   - If model exists in `~/.cache/huggingface/`, it reuses it
   - No duplicate downloads!

2. **Checks Centralized Storage**
   - If model exists in `data/models/`, it uses it
   - No re-downloads!

3. **Downloads Only If Missing**
   - Only downloads models that don't exist anywhere
   - Shows progress during download

4. **Idempotent**
   - Safe to run multiple times
   - Won't re-download existing models

### Initialization Flow

```
┌─────────────────────────────────────────┐
│  Backend/Agent Startup                  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Check .initialization_status.json      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Check Embedding Models                 │
│  - all-MiniLM-L6-v2                    │
│  - clip-vit-base-patch32               │
└─────────────────┬───────────────────────┘
                  │
                  ├─ Found in HF cache? → Copy to data/models/
                  ├─ Found in data/models/? → Use it
                  └─ Not found? → Download
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Check Ollama Models (if installed)     │
│  - List installed models via API       │
│  - Pull default if missing             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Update .initialization_status.json     │
│  - Record downloaded models            │
│  - Save timestamps                     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  System Ready! ✅                       │
└─────────────────────────────────────────┘
```

---

## Usage Examples

### 1. First-Time Setup

```bash
# Run complete setup (includes model initialization)
python setup_system.py

# Output:
# [1/9] Checking Docker... ✓
# [2/9] Starting PostgreSQL... ✓
# [3/9] Running migrations... ✓
# [4/9] Creating directories... ✓
# [5/9] Initializing essential models...
#    🔍 Checking system initialization...
#    📦 Checking embedding models...
#    ✅ Found in HuggingFace cache: all-MiniLM-L6-v2
#    ⬇️  Downloading: clip-vit-base-patch32... [████████] 100%
#    ✅ Downloaded: clip-vit-base-patch32
#    ✅ System initialization complete
# [6/9] Checking Ollama...
#    ✅ Ollama is running (3 models installed)
#    ✅ Default model available: llama3.2-vision:11b
# ...
# ✅ Setup complete!
```

### 2. Backend Startup

```bash
# Start the backend
python -m app.main

# Output:
# 🚀 Starting Agentic AI Microservice v0.1.0
# 🔍 Checking system initialization...
#    ✅ Already available: all-MiniLM-L6-v2
#    ✅ Already available: clip-vit-base-patch32
#    ✅ System initialization complete
# ⚙️  Initializing core systems...
# ✅ All services initialized successfully
# 🎯 System ready: Unlimited agents, Dynamic tools, True agentic AI
```

### 3. Agent Startup

```bash
# Run an agent
python data/agents/my_agent.py

# Output:
# 🔍 Checking system initialization for agent: my_agent
#    ✅ Already available: all-MiniLM-L6-v2
#    ✅ Already available: clip-vit-base-patch32
# ✅ System ready
# Initializing agent: my_agent
# ...
```

---

## Configuration

### Initialization Status File

Location: `data/.initialization_status.json`

```json
{
  "last_check": "2025-01-15T10:30:00Z",
  "embedding_models_ready": true,
  "ollama_ready": true,
  "system_ready": true,
  "embedding_models": [
    "sentence-transformers/all-MiniLM-L6-v2",
    "openai/clip-vit-base-patch32"
  ],
  "embedding_models_timestamp": "2025-01-15T10:30:00Z",
  "ollama_installed": true,
  "ollama_models": [
    "llama3.2-vision:11b",
    "llama3.2:3b"
  ],
  "ollama_default_model": "llama3.2-vision:11b"
}
```

### Environment Variables

```bash
# Skip model downloads (useful for CI/testing)
SKIP_MODEL_DOWNLOADS=true

# Set default Ollama model
AGENTIC_DEFAULT_AGENT_MODEL=llama3.2-vision:11b
```

---

## Troubleshooting

### Models Not Downloading

**Problem**: Models aren't downloading during initialization

**Solutions**:
1. Check internet connection
2. Check HuggingFace access (some models require authentication)
3. Check disk space in `data/models/`
4. Run manual initialization: `python scripts/initialize_models.py`

### Ollama Models Not Pulling

**Problem**: Ollama models aren't being pulled

**Solutions**:
1. Check if Ollama is running: `ollama list`
2. Check Ollama service: `http://localhost:11434/api/tags`
3. Pull manually: `ollama pull llama3.2-vision:11b`
4. Check Ollama logs for errors

### Initialization Takes Too Long

**Problem**: First-time initialization is slow

**Explanation**: This is normal! Model downloads can take several minutes:
- `all-MiniLM-L6-v2`: ~90 MB
- `clip-vit-base-patch32`: ~600 MB
- `llama3.2-vision:11b`: ~7 GB

**Solutions**:
- Be patient on first run
- Subsequent runs are instant (models already downloaded)
- Use faster internet connection for initial setup

---

## Advanced Usage

### Manual Initialization

```python
from app.core.system_initialization import ensure_system_ready

# Silent mode (no progress messages)
await ensure_system_ready(silent=True)

# Verbose mode (show progress)
await ensure_system_ready(silent=False)
```

### Check Initialization Status

```python
from pathlib import Path
import json

status_file = Path("data/.initialization_status.json")
if status_file.exists():
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    print(f"System ready: {status.get('system_ready')}")
    print(f"Models: {status.get('embedding_models')}")
```

### Force Re-initialization

```bash
# Delete status file to force re-check
rm data/.initialization_status.json

# Run setup again
python setup_system.py
```

---

## Benefits

✅ **Zero Manual Setup** - Models download automatically
✅ **No Duplicates** - Reuses HuggingFace cache
✅ **Fast Subsequent Runs** - Only checks, doesn't re-download
✅ **Graceful Degradation** - System works even if some models fail
✅ **Universal** - Works for backend, agents, and scripts
✅ **Idempotent** - Safe to run multiple times
✅ **Smart Detection** - Checks multiple locations before downloading

---

## Related Documentation

- [Model Management](../reference/MODEL_MANAGEMENT.md)
- [Setup Guide](../tutorials/SETUP.md)
- [Agent Template System](../reference/AGENT_TEMPLATES.md)

