# 🚀 Backend Launcher Guide

## Overview

The Agentic AI Engine backend can now be launched with a simple, clean Python script that:
- ✅ Validates configuration before startup
- ✅ Uses the unified configuration system
- ✅ Provides clear startup information
- ✅ Has zero bloat
- ✅ Works on all platforms

---

## Quick Start

### Launch the Backend

```bash
python run.py
```

That's it! The backend will:
1. Validate your configuration
2. Display startup information
3. Start the server on the configured port (default: 8888)

---

## What `run.py` Does

### 1. Configuration Validation
- Loads the unified configuration system
- Validates all settings (skips connectivity checks for faster startup)
- Reports any configuration issues

### 2. Startup Information
Displays:
- Application name
- Environment (development/staging/production)
- Host and port
- Documentation URL

### 3. Server Launch
- Starts Uvicorn with optimal settings
- Uses configuration from unified config
- Enables reload in debug mode
- Adjusts logging based on environment

---

## Configuration

The launcher uses settings from the unified configuration system.

### Environment Variables

Set these in your `.env` file or environment:

```env
# Server Configuration
SERVER__HOST=0.0.0.0
SERVER__PORT=8888
SERVER__DEBUG=false
SERVER__ENVIRONMENT=production

# See CONFIGURATION_GUIDE.md for all options
```

### Default Settings

If no configuration is provided:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8888
- **Environment**: development
- **Debug**: false

---

## Usage Examples

### Development Mode

```bash
# Set debug mode in .env
SERVER__DEBUG=true

# Run
python run.py
```

Features:
- Auto-reload on code changes
- Detailed logging
- Access logs enabled

### Production Mode

```bash
# Set production mode in .env
SERVER__ENVIRONMENT=production
SERVER__DEBUG=false

# Run
python run.py
```

Features:
- No auto-reload
- Minimal logging
- Optimized performance

### Custom Port

```bash
# Set custom port in .env
SERVER__PORT=9000

# Run
python run.py
```

---

## Accessing the Backend

Once started, access:

- **API**: http://localhost:8888
- **Interactive Docs**: http://localhost:8888/docs
- **OpenAPI Schema**: http://localhost:8888/openapi.json
- **Health Check**: http://localhost:8888/health

---

## Stopping the Server

Press `Ctrl+C` to gracefully stop the server.

The server will:
1. Stop accepting new requests
2. Complete ongoing requests
3. Shutdown services cleanly
4. Exit

---

## Troubleshooting

### Configuration Validation Fails

If you see configuration warnings:
1. Check the validation output
2. Review `CONFIGURATION_GUIDE.md`
3. Fix critical issues
4. Warnings are OK (server will still start)

### Port Already in Use

If port 8888 is already in use:
1. Change the port in `.env`:
   ```env
   SERVER__PORT=9000
   ```
2. Or stop the other service using port 8888

### Import Errors

If you see import errors:
1. Ensure you're in the correct directory
2. Check that all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

### Server Won't Start

If the server fails to start:
1. Check the error message
2. Verify database is running (if using PostgreSQL)
3. Verify Redis is running (if using Redis)
4. Check logs in `data/logs/`

---

## Advanced Usage

### Running with Custom Settings

You can override settings via environment variables:

```bash
# Windows PowerShell
$env:SERVER__PORT=9000
python run.py

# Linux/Mac
SERVER__PORT=9000 python run.py
```

### Running in Background

```bash
# Linux/Mac
nohup python run.py > backend.log 2>&1 &

# Windows (use a process manager like PM2 or NSSM)
```

### Running with Process Manager

For production, use a process manager:

**PM2** (Node.js required):
```bash
pm2 start run.py --name agentic-backend --interpreter python
pm2 save
pm2 startup
```

**Systemd** (Linux):
Create `/etc/systemd/system/agentic-backend.service`:
```ini
[Unit]
Description=Agentic AI Backend
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Agents
Environment="PATH=/path/to/python/bin"
ExecStart=/path/to/python run.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable agentic-backend
sudo systemctl start agentic-backend
```

---

## File Structure

```
Agents/
├── run.py                          # ← Main launcher (NEW)
├── app/
│   ├── main.py                     # FastAPI application
│   └── config/
│       ├── unified_config.py       # Unified configuration
│       ├── config_groups.py        # Configuration groups
│       └── config_validator.py     # Validation system
├── .env                            # Environment variables
└── data/
    └── logs/                       # Application logs
```

---

## Comparison with Old Method

### Old Way ❌
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload
```

Problems:
- No configuration validation
- Manual port/host specification
- No startup checks
- Verbose command

### New Way ✅
```bash
python run.py
```

Benefits:
- ✅ Automatic configuration validation
- ✅ Uses unified config settings
- ✅ Startup checks
- ✅ Clean, simple command
- ✅ Environment-aware
- ✅ Zero bloat

---

## Integration with Enhancements

The launcher integrates with all system enhancements:

### Configuration System
- Uses unified configuration
- Validates before startup
- Reports configuration status

### Error Handling
- Graceful error reporting
- Clean shutdown on errors
- Detailed error messages

### Performance
- Optimal Uvicorn settings
- Environment-based configuration
- Production-ready defaults

---

## Next Steps

1. **Configure**: Review `CONFIGURATION_GUIDE.md`
2. **Launch**: Run `python run.py`
3. **Test**: Access http://localhost:8888/docs
4. **Monitor**: Check logs in `data/logs/`
5. **Deploy**: Use process manager for production

---

## Support

For issues:
1. Check this guide
2. Review `CONFIGURATION_GUIDE.md`
3. Check application logs
4. Verify configuration with validation

---

## Summary

The new `run.py` launcher provides:
- ✅ **Simple**: One command to start
- ✅ **Clean**: No bloat, minimal code
- ✅ **Smart**: Validates configuration
- ✅ **Flexible**: Environment-aware
- ✅ **Production-ready**: Optimal defaults

**Just run `python run.py` and you're good to go!** 🚀

