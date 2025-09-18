# PowerShell script to start PostgreSQL 17 container for Agentic AI development

Write-Host "🚀 Starting PostgreSQL 17 container for Agentic AI..." -ForegroundColor Green

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Navigate to project root
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Create data directory if it doesn't exist
if (!(Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
    Write-Host "📁 Created data directory" -ForegroundColor Yellow
}

# Start PostgreSQL container
Write-Host "🐘 Starting PostgreSQL 17 container..." -ForegroundColor Blue
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
Write-Host "⏳ Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0

do {
    $attempt++
    Start-Sleep -Seconds 2
    
    try {
        $result = docker-compose exec -T postgres pg_isready -U agentic_user -d agentic_ai 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ PostgreSQL is ready!" -ForegroundColor Green
            break
        }
    } catch {
        # Continue waiting
    }
    
    if ($attempt -eq $maxAttempts) {
        Write-Host "❌ PostgreSQL failed to start within timeout" -ForegroundColor Red
        Write-Host "📋 Container logs:" -ForegroundColor Yellow
        docker-compose logs postgres
        exit 1
    }
    
    Write-Host "⏳ Attempt $attempt/$maxAttempts - PostgreSQL not ready yet..." -ForegroundColor Yellow
} while ($true)

# Show connection information
Write-Host ""
Write-Host "🎉 PostgreSQL 17 is now running!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Connection Details:" -ForegroundColor Cyan
Write-Host "  Host: localhost" -ForegroundColor White
Write-Host "  Port: 5432" -ForegroundColor White
Write-Host "  Database: agentic_ai" -ForegroundColor White
Write-Host "  Username: agentic_user" -ForegroundColor White
Write-Host "  Password: agentic_secure_password_2024" -ForegroundColor White
Write-Host ""
Write-Host "🔧 pgAdmin (Database Management):" -ForegroundColor Cyan
Write-Host "  URL: http://localhost:5050" -ForegroundColor White
Write-Host "  Email: admin@agentic.ai" -ForegroundColor White
Write-Host "  Password: admin_password_2024" -ForegroundColor White
Write-Host ""

# Ask if user wants to run database migrations
$runMigrations = Read-Host "🔄 Do you want to run database migrations now? (y/N)"
if ($runMigrations -eq "y" -or $runMigrations -eq "Y") {
    Write-Host "🔄 Running database migrations..." -ForegroundColor Blue
    
    try {
        Set-Location "app/models/database/migrations"
        python create_autonomous_tables.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Database migrations completed successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ Database migrations failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Error running migrations: $_" -ForegroundColor Red
    } finally {
        Set-Location $projectRoot
    }
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Run database migrations: python app/models/database/migrations/create_autonomous_tables.py" -ForegroundColor White
Write-Host "  2. Start the Agentic AI backend: python -m app.main" -ForegroundColor White
Write-Host "  3. Run tests: python -m pytest tests/test_truly_agentic_ai.py -v" -ForegroundColor White
Write-Host ""
Write-Host "🛑 To stop PostgreSQL: docker-compose down" -ForegroundColor Yellow
Write-Host "🗑️  To remove all data: docker-compose down -v" -ForegroundColor Red
