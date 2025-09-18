# Project Restoration Script
# Run this script after transferring the project to restore dependencies and setup

Write-Host "Starting project restoration after transfer..." -ForegroundColor Green

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+ first." -ForegroundColor Red
    exit 1
}

# Check if Node.js is available
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js first." -ForegroundColor Red
    exit 1
}

Write-Host "`n1. Installing Python dependencies..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "✓ Python dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ requirements.txt not found" -ForegroundColor Yellow
}

Write-Host "`n2. Installing Node.js dependencies..." -ForegroundColor Cyan
if (Test-Path "frontend/package.json") {
    Set-Location frontend
    npm install
    Set-Location ..
    Write-Host "✓ Node.js dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ frontend/package.json not found" -ForegroundColor Yellow
}

Write-Host "`n3. Creating necessary directories..." -ForegroundColor Cyan
$directories = @(
    "data/cache",
    "data/logs", 
    "data/uploaded_files",
    "data/uploads",
    "logs/backend",
    "logs/frontend"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✓ Created: $dir" -ForegroundColor Green
    }
}

Write-Host "`n4. Setting up environment..." -ForegroundColor Cyan
if (!(Test-Path ".env")) {
    $envTemplate = @"
# Environment Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/agents_db
CHROMA_PERSIST_DIRECTORY=./data/chroma
EMBEDDING_MODEL_PATH=./data/models/embedding
LOG_LEVEL=INFO
"@
    $envTemplate | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✓ Created .env template" -ForegroundColor Green
} else {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}

Write-Host "`n✅ Project restoration completed!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Configure your .env file with proper database credentials" -ForegroundColor White
Write-Host "2. Set up PostgreSQL database if needed" -ForegroundColor White  
Write-Host "3. Run database migrations: python -m app.migrations" -ForegroundColor White
Write-Host "4. Start the application: python -m app.main" -ForegroundColor White
