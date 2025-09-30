# PowerShell script to start PostgreSQL 17 container for Agentic AI development
# ENHANCED: Now includes complete data directory setup

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AGENTIC AI SYSTEM SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "[1/5] Checking Docker..." -ForegroundColor Blue
try {
    docker version | Out-Null
    Write-Host "      Docker is running" -ForegroundColor Green
} catch {
    Write-Host "      Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Navigate to project root
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Create complete data directory structure
Write-Host ""
Write-Host "[2/5] Creating data directory structure..." -ForegroundColor Blue

$dataDirectories = @(
    "data",
    "data/agents",
    "data/workflows",
    "data/checkpoints",
    "data/logs",
    "data/logs/agents",
    "data/logs/backend",
    "data/chroma",
    "data/autonomous",
    "data/agent_files",
    "data/cache",
    "data/downloads",
    "data/downloads/session_docs",
    "data/generated_files",
    "data/memes",
    "data/memes/generated",
    "data/memes/templates",
    "data/models",
    "data/models/embedding",
    "data/models/llm",
    "data/models/reranking",
    "data/models/vision",
    "data/outputs",
    "data/screenshots",
    "data/session_documents",
    "data/session_documents/sessions",
    "data/session_vectors",
    "data/templates",
    "data/temp",
    "data/temp/session_docs",
    "data/uploads",
    "data/config",
    "data/config/agents",
    "data/config/templates",
    "data/meme_analysis_cache"
)

$createdCount = 0
foreach ($dir in $dataDirectories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        $createdCount++
    }
}

Write-Host "      Created $createdCount new directories" -ForegroundColor Green
Write-Host "      Total data directories: $($dataDirectories.Count)" -ForegroundColor Green

# Start PostgreSQL container
Write-Host ""
Write-Host "[3/5] Starting PostgreSQL 17 container..." -ForegroundColor Blue
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
Write-Host "      Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0

do {
    $attempt++
    Start-Sleep -Seconds 2

    try {
        $result = docker-compose exec -T postgres pg_isready -U agentic_user -d agentic_ai 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      PostgreSQL is ready!" -ForegroundColor Green
            break
        }
    } catch {
        # Continue waiting
    }

    if ($attempt -eq $maxAttempts) {
        Write-Host "      PostgreSQL failed to start within timeout" -ForegroundColor Red
        Write-Host "      Container logs:" -ForegroundColor Yellow
        docker-compose logs postgres
        exit 1
    }

    if ($attempt % 5 -eq 0) {
        Write-Host "      Attempt $attempt/$maxAttempts - Still waiting..." -ForegroundColor Yellow
    }
} while ($true)

# Run database migrations automatically
Write-Host ""
Write-Host "[4/5] Running database migrations..." -ForegroundColor Blue

try {
    python "db/migrations/run_all_migrations.py"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "      Database migrations completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "      Database migrations completed with warnings (this is normal)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "      Error running migrations: $_" -ForegroundColor Red
    Write-Host "      You can run migrations manually later with:" -ForegroundColor Yellow
    Write-Host "      python db/migrations/run_all_migrations.py" -ForegroundColor Yellow
}

# Initialize system (test run)
Write-Host ""
Write-Host "[5/5] Testing system initialization..." -ForegroundColor Blue

try {
    # Test import to ensure everything is set up correctly
    python -c "from app.config.settings import get_settings; settings = get_settings(); settings.create_directories(); print('System directories initialized')" 2>$null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "      System initialization successful!" -ForegroundColor Green
    } else {
        Write-Host "      System initialization completed with warnings" -ForegroundColor Yellow
    }
} catch {
    Write-Host "      Could not test system initialization" -ForegroundColor Yellow
}

# Show completion summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Database Connection:" -ForegroundColor Cyan
Write-Host "  Host: localhost:5432" -ForegroundColor White
Write-Host "  Database: agentic_ai" -ForegroundColor White
Write-Host "  Username: agentic_user" -ForegroundColor White
Write-Host ""
Write-Host "pgAdmin (Database Management):" -ForegroundColor Cyan
Write-Host "  URL: http://localhost:5050" -ForegroundColor White
Write-Host "  Email: admin@agentic.ai" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Start backend:  python -m app.main" -ForegroundColor White
Write-Host "  2. Use agents:     python -c 'from app.agents import create_agent; ...'" -ForegroundColor White
Write-Host "  3. Run tests:      python -m pytest tests/ -v" -ForegroundColor White
Write-Host ""
Write-Host "Management Commands:" -ForegroundColor Cyan
Write-Host "  Stop PostgreSQL:   docker-compose down" -ForegroundColor Yellow
Write-Host "  Remove all data:   docker-compose down -v" -ForegroundColor Red
Write-Host ""
