#!/bin/bash
# Bash script to start PostgreSQL 17 container for Agentic AI development

echo "🚀 Starting PostgreSQL 17 container for Agentic AI..."

# Check if Docker is running
if ! docker version >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Navigate to project root
cd "$(dirname "$0")/.."

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    mkdir -p data
    echo "📁 Created data directory"
fi

# Start PostgreSQL container
echo "🐘 Starting PostgreSQL 17 container..."
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    sleep 2
    
    if docker-compose exec -T postgres pg_isready -U agentic_user -d agentic_ai >/dev/null 2>&1; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ PostgreSQL failed to start within timeout"
        echo "📋 Container logs:"
        docker-compose logs postgres
        exit 1
    fi
    
    echo "⏳ Attempt $attempt/$max_attempts - PostgreSQL not ready yet..."
done

# Show connection information
echo ""
echo "🎉 PostgreSQL 17 is now running!"
echo ""
echo "📊 Connection Details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: agentic_ai"
echo "  Username: agentic_user"
echo "  Password: agentic_secure_password_2024"
echo ""
echo "🔧 pgAdmin (Database Management):"
echo "  URL: http://localhost:5050"
echo "  Email: admin@agentic.ai"
echo "  Password: admin_password_2024"
echo ""

# Ask if user wants to run database migrations
read -p "🔄 Do you want to run database migrations now? (y/N): " run_migrations
if [[ $run_migrations =~ ^[Yy]$ ]]; then
    echo "🔄 Running database migrations..."

    if python db/migrations/migrate_database.py migrate; then
        echo "✅ Database migrations completed successfully!"
    else
        echo "❌ Database migrations failed"
    fi
fi

echo ""
echo "🎯 Next Steps:"
echo "  1. Run database migrations: python app/models/database/migrations/create_autonomous_tables.py"
echo "  2. Start the Agentic AI backend: python -m app.main"
echo "  3. Run tests: python -m pytest tests/test_truly_agentic_ai.py -v"
echo ""
echo "🛑 To stop PostgreSQL: docker-compose down"
echo "🗑️  To remove all data: docker-compose down -v"
