# 🚀 Agentic AI Platform - Deployment Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.9 or higher
- **Node.js**: 16 or higher (for frontend)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **Storage**: Minimum 50GB free space
- **CPU**: 4+ cores recommended

### Required Services
- **PostgreSQL**: 13 or higher
- **Redis**: 6 or higher
- **ChromaDB**: Latest version
- **Docker**: 20.10 or higher (optional)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-org/agentic-ai-platform.git
cd agentic-ai-platform
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install frontend dependencies
cd frontend
npm install
```

### 3. Setup Database
```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run database migrations
python scripts/migrate_database.py

# Initialize database
python scripts/initialize_models.py
```

### 4. Start Services
```bash
# Start backend
python app/main.py

# Start frontend (in another terminal)
cd frontend
npm run dev
```

### 5. Access Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Development Setup

### 1. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Database Setup
```bash
# Create development database
createdb agentic_ai_dev

# Run migrations
alembic upgrade head

# Seed development data
python scripts/seed_dev_data.py
```

### 3. Development Tools
```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
flake8 app/
black app/
```

### 4. Development Workflow
```bash
# Start development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend with hot reload
cd frontend && npm run dev
```

## Production Deployment

### 1. Server Preparation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3.9 python3.9-venv python3-pip postgresql redis-server nginx

# Create application user
sudo useradd -m -s /bin/bash agenticai
sudo usermod -aG sudo agenticai
```

### 2. Application Deployment
```bash
# Clone repository
sudo -u agenticai git clone https://github.com/your-org/agentic-ai-platform.git /opt/agentic-ai

# Create virtual environment
sudo -u agenticai python3.9 -m venv /opt/agentic-ai/venv
sudo -u agenticai /opt/agentic-ai/venv/bin/pip install -r /opt/agentic-ai/requirements.txt

# Set permissions
sudo chown -R agenticai:agenticai /opt/agentic-ai
```

### 3. Database Configuration
```bash
# Create production database
sudo -u postgres createdb agentic_ai_prod

# Create database user
sudo -u postgres createuser agenticai
sudo -u postgres psql -c "ALTER USER agenticai PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agentic_ai_prod TO agenticai;"

# Run migrations
sudo -u agenticai /opt/agentic-ai/venv/bin/python /opt/agentic-ai/scripts/migrate_database.py
```

### 4. Systemd Service
```bash
# Create systemd service file
sudo nano /etc/systemd/system/agentic-ai.service
```

```ini
[Unit]
Description=Agentic AI Platform
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=agenticai
Group=agenticai
WorkingDirectory=/opt/agentic-ai
Environment=PATH=/opt/agentic-ai/venv/bin
ExecStart=/opt/agentic-ai/venv/bin/python app/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable agentic-ai
sudo systemctl start agentic-ai
```

### 5. Nginx Configuration
```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/agentic-ai
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /opt/agentic-ai/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/agentic-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Docker Deployment

### 1. Docker Compose Setup
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: agentic_ai
      POSTGRES_USER: agenticai
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  backend:
    build: .
    environment:
      DATABASE_URL: postgresql://agenticai:secure_password@postgres:5432/agentic_ai
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
  redis_data:
```

### 2. Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agenticai && chown -R agenticai:agenticai /app
USER agenticai

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "app/main.py"]
```

### 3. Deploy with Docker
```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Run database migrations
docker-compose -f docker-compose.prod.yml exec backend python scripts/migrate_database.py

# Check service status
docker-compose -f docker-compose.prod.yml ps
```

## Configuration

### 1. Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_ai
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# Monitoring
ENABLE_MONITORING=true
ENABLE_ALERTING=true
LOG_LEVEL=INFO

# Performance
CACHE_ENABLED=true
CACHE_TTL=3600
MAX_CONNECTIONS=100
```

### 2. Database Configuration
```python
# app/config/database.py
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo": False
}
```

### 3. Redis Configuration
```python
# app/config/redis.py
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None,
    "max_connections": 100,
    "retry_on_timeout": True
}
```

## Monitoring

### 1. Health Checks
```bash
# Check application health
curl http://localhost:8000/health

# Check database connection
curl http://localhost:8000/health/database

# Check Redis connection
curl http://localhost:8000/health/redis
```

### 2. Log Monitoring
```bash
# View application logs
sudo journalctl -u agentic-ai -f

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 3. Performance Monitoring
```bash
# Check system resources
htop
iostat -x 1
free -h

# Check database performance
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
```

### 4. Alerting Setup
```bash
# Configure email alerts
export SMTP_SERVER=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USERNAME=your-email@gmail.com
export SMTP_PASSWORD=your-app-password

# Configure Slack alerts
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

## Troubleshooting

### 1. Common Issues

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database connectivity
psql -h localhost -U agenticai -d agentic_ai_prod

# Check connection limits
sudo -u postgres psql -c "SHOW max_connections;"
```

#### Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis

# Test Redis connection
redis-cli ping

# Check Redis memory usage
redis-cli info memory
```

#### Application Issues
```bash
# Check application status
sudo systemctl status agentic-ai

# View application logs
sudo journalctl -u agentic-ai -n 100

# Check port usage
sudo netstat -tlnp | grep :8000
```

### 2. Performance Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check for memory leaks
sudo -u agenticai /opt/agentic-ai/venv/bin/python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### High CPU Usage
```bash
# Check CPU usage
top
htop

# Check for blocking queries
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

#### Slow Response Times
```bash
# Check Nginx logs for slow requests
sudo tail -f /var/log/nginx/access.log | grep -E "GET|POST"

# Check database query performance
sudo -u postgres psql -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

### 3. Security Issues

#### Authentication Problems
```bash
# Check JWT token validity
curl -H "Authorization: Bearer your-token" http://localhost:8000/api/v1/auth/verify

# Check session storage
redis-cli keys "session:*"
```

#### Rate Limiting Issues
```bash
# Check rate limit status
redis-cli keys "rate_limit:*"

# Clear rate limits (if needed)
redis-cli flushdb
```

### 4. Backup and Recovery

#### Database Backup
```bash
# Create database backup
sudo -u postgres pg_dump agentic_ai_prod > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database backup
sudo -u postgres psql agentic_ai_prod < backup_20240101_120000.sql
```

#### Application Backup
```bash
# Backup application data
tar -czf agentic_ai_backup_$(date +%Y%m%d_%H%M%S).tar.gz /opt/agentic-ai

# Restore application data
tar -xzf agentic_ai_backup_20240101_120000.tar.gz -C /
```

## Maintenance

### 1. Regular Maintenance Tasks
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update application
cd /opt/agentic-ai
sudo -u agenticai git pull
sudo -u agenticai /opt/agentic-ai/venv/bin/pip install -r requirements.txt

# Restart services
sudo systemctl restart agentic-ai
sudo systemctl restart nginx
```

### 2. Database Maintenance
```bash
# Analyze database
sudo -u postgres psql -c "ANALYZE;"

# Vacuum database
sudo -u postgres psql -c "VACUUM ANALYZE;"

# Check database size
sudo -u postgres psql -c "SELECT pg_size_pretty(pg_database_size('agentic_ai_prod'));"
```

### 3. Log Rotation
```bash
# Configure log rotation
sudo nano /etc/logrotate.d/agentic-ai
```

```
/var/log/agentic-ai/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 agenticai agenticai
    postrotate
        systemctl reload agentic-ai
    endscript
}
```

---

This deployment guide provides comprehensive instructions for deploying the Agentic AI Platform in various environments, from development to production.


