# Development Launcher for Agentic AI System

This directory contains several scripts to launch both the FastAPI backend and React frontend simultaneously for development purposes.

## 🚀 Quick Start

### Option 1: Simple Launcher (Recommended)
```bash
# Python script that opens services in separate terminal windows
python simple_start.py
```

### Option 2: Integrated Launcher
```bash
# Python script that manages both services in one terminal
python start_dev.py
```

### Option 3: Platform-Specific Scripts

**Windows:**
```powershell
# PowerShell script
.\start-dev.ps1

# Batch file (fallback)
.\start-dev.bat
```

**Linux/macOS:**
```bash
# Shell script
./start-dev.sh
```

## 📋 Prerequisites

All launchers will automatically check for these prerequisites:

- **Python 3.8+** - For the FastAPI backend
- **Node.js 16+** - For the React frontend
- **npm** - Node package manager (automatically detected even if not in PATH)

## 🛠️ Launcher Options

### Simple Launcher (`simple_start.py`)

**Best for:** Most users, especially beginners

**Features:**
- ✅ Opens each service in a separate terminal window
- ✅ Easy to see logs for each service separately
- ✅ Simple to stop services (just close terminal windows)
- ✅ Automatically detects npm even when not in PATH
- ✅ Installs dependencies if needed

**Usage:**
```bash
python simple_start.py
```

**Services:**
- Backend: http://localhost:8888
- Frontend: http://localhost:5173
- API Docs: http://localhost:8888/docs

### Integrated Launcher (`start_dev.py`)

**Best for:** Advanced users who want integrated monitoring

**Features:**
- ✅ Manages both services in one terminal
- ✅ Real-time process monitoring
- ✅ Graceful shutdown with Ctrl+C
- ✅ Health checks and status monitoring
- ✅ Detailed startup progress

**Usage:**
```bash
python start_dev.py
```

### PowerShell Launcher (`start-dev.ps1`)

**Best for:** Windows users who prefer PowerShell

**Features:**
- ✅ Native Windows PowerShell integration
- ✅ Colored output and status indicators
- ✅ Command-line options for customization
- ✅ Automatic dependency checking

**Usage:**
```powershell
# Basic usage
.\start-dev.ps1

# Custom ports
.\start-dev.ps1 -BackendPort 9000 -FrontendPort 3000

# Skip prerequisite checks
.\start-dev.ps1 -SkipChecks

# Show help
.\start-dev.ps1 -Help
```

## 🔧 Configuration

### Default Ports
- **Backend:** 8888
- **Frontend:** 5173 (Vite dev server)

### Environment Variables

The system uses these environment variables (optional):

```bash
# Backend Configuration
AGENTIC_HOST=localhost
AGENTIC_PORT=8888
AGENTIC_DEBUG=true

# Frontend Configuration
VITE_API_URL=http://localhost:8888/api/v1
VITE_WS_URL=ws://localhost:8888
```

### Custom Configuration

You can create a `.env` file in the project root to override defaults:

```bash
# Copy the example file
cp .env.example .env

# Edit the configuration
# nano .env  # Linux/macOS
# notepad .env  # Windows
```

## 🐛 Troubleshooting

### Common Issues

1. **npm not found**
   - The launchers automatically detect npm in common locations
   - If you're using nvm or nvm4w, the scripts will find npm automatically
   - Supported locations: Node.js installation, nvm4w, standard paths

2. **Port already in use**
   - Backend (8888): Check if another FastAPI instance is running
   - Frontend (5173): Check if another Vite dev server is running
   - Use `netstat -ano | findstr :8888` (Windows) or `lsof -i :8888` (Linux/macOS)

3. **Dependencies not installed**
   - Backend: Run `pip install -e .` in the project root
   - Frontend: Run `npm install` in the frontend directory
   - The launchers will attempt to install dependencies automatically

4. **Permission errors on Linux/macOS**
   ```bash
   chmod +x start-dev.sh
   ./start-dev.sh
   ```

### Debug Mode

For detailed debugging, you can run services manually:

**Backend:**
```bash
python -m uvicorn app.main:app --host localhost --port 8888 --reload --log-level debug
```

**Frontend:**
```bash
cd frontend
npm run dev
```

## 📊 Service Status

### Health Checks

- **Backend Health:** http://localhost:8888/health
- **Frontend Status:** http://localhost:5173

### API Documentation

- **Interactive Docs:** http://localhost:8888/docs
- **ReDoc:** http://localhost:8888/redoc

## 🔄 Development Workflow

1. **Start Services:**
   ```bash
   python simple_start.py
   ```

2. **Develop:**
   - Backend code changes trigger automatic reload
   - Frontend changes trigger hot module replacement
   - Both services support live reloading

3. **Test:**
   - Backend: http://localhost:8888/docs
   - Frontend: http://localhost:5173

4. **Stop Services:**
   - Simple launcher: Close terminal windows
   - Integrated launcher: Press Ctrl+C

## 🚨 Known Issues

1. **Tool initialization warnings** - The backend shows some warnings about tool initialization. These are non-critical and don't affect functionality.

2. **Pydantic compatibility warnings** - Some LangChain components show deprecation warnings. These are informational only.

3. **Prometheus metrics duplication** - Multiple startups may cause metric registry conflicts. Restart clears this.

## 📝 Contributing

When adding new launcher features:

1. Update all launcher scripts consistently
2. Test on both Windows and Linux/macOS
3. Update this documentation
4. Ensure backward compatibility

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify prerequisites are installed
3. Try the simple launcher first
4. Check service logs in terminal windows
5. Test manual service startup for debugging
