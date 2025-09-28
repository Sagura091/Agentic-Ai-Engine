#!/bin/bash

# Complete Database Setup Script for Agentic AI System
# This script performs a complete database setup after Docker volumes are cleared:
# 1. Executes init-db.sql for basic table structure
# 2. Runs all migrations for full feature set
# 3. Verifies the setup was successful

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Database connection settings (adjust as needed)
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-agentic_ai}"
DB_USER="${DB_USER:-agentic_user}"
DB_PASSWORD="${DB_PASSWORD:-agentic_password}"

echo -e "${BLUE}🚀 AGENTIC AI DATABASE SETUP${NC}"
echo "=================================================="
echo "This script will set up your database from scratch."
echo "Use this after clearing Docker volumes or resetting the database."
echo ""

# Function to wait for database
wait_for_database() {
    echo -e "${YELLOW}🔄 Waiting for database connection...${NC}"
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Database connection established!${NC}"
            return 0
        else
            echo -e "${YELLOW}⏳ Attempt $attempt/$max_attempts failed, retrying in 2s...${NC}"
            sleep 2
            ((attempt++))
        fi
    done
    
    echo -e "${RED}❌ Failed to connect to database after $max_attempts attempts${NC}"
    return 1
}

# Function to execute init-db.sql
execute_init_sql() {
    echo -e "${YELLOW}🔧 Executing init-db.sql...${NC}"
    
    if [ ! -f "init-db.sql" ]; then
        echo -e "${RED}❌ init-db.sql not found in current directory${NC}"
        return 1
    fi
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "init-db.sql"; then
        echo -e "${GREEN}✅ init-db.sql executed successfully!${NC}"
        return 0
    else
        echo -e "${RED}❌ Error executing init-db.sql${NC}"
        return 1
    fi
}

# Function to run migrations
run_migrations() {
    echo -e "${YELLOW}🚀 Running database migrations...${NC}"
    
    if [ ! -f "db/migrations/run_all_migrations.py" ]; then
        echo -e "${RED}❌ Migration script not found at db/migrations/run_all_migrations.py${NC}"
        return 1
    fi
    
    if python db/migrations/run_all_migrations.py; then
        echo -e "${GREEN}✅ Database migrations completed successfully!${NC}"
        return 0
    else
        echo -e "${RED}❌ Migration failed${NC}"
        return 1
    fi
}

# Function to verify setup
verify_setup() {
    echo -e "${YELLOW}🔍 Verifying database setup...${NC}"
    
    # Check for key tables
    local tables=("agents" "workflows" "custom_tools" "users" "user_sessions")
    local found_tables=0
    
    for table in "${tables[@]}"; do
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM $table LIMIT 1;" >/dev/null 2>&1; then
            echo -e "${GREEN}   ✅ $table${NC}"
            ((found_tables++))
        else
            echo -e "${RED}   ❌ $table${NC}"
        fi
    done
    
    echo -e "${BLUE}📊 Found $found_tables/${#tables[@]} expected tables${NC}"
    
    if [ $found_tables -ge 3 ]; then
        echo -e "${GREEN}✅ Database setup verification passed!${NC}"
        return 0
    else
        echo -e "${RED}❌ Database setup verification failed!${NC}"
        return 1
    fi
}

# Main execution
main() {
    # Check if we're in the right directory
    if [ ! -f "init-db.sql" ] || [ ! -d "db/migrations" ]; then
        echo -e "${RED}❌ Please run this script from the project root directory${NC}"
        echo "   The script expects to find init-db.sql and db/migrations/ in the current directory"
        exit 1
    fi
    
    # Step 1: Wait for database
    if ! wait_for_database; then
        exit 1
    fi
    
    # Step 2: Execute init-db.sql
    if ! execute_init_sql; then
        exit 1
    fi
    
    # Step 3: Run migrations
    if ! run_migrations; then
        exit 1
    fi
    
    # Step 4: Verify setup
    if ! verify_setup; then
        echo -e "${YELLOW}⚠️  Setup completed with warnings. Some features may not be available.${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 DATABASE SETUP COMPLETE!${NC}"
    echo "=================================================="
    echo -e "${GREEN}✅ Your database is ready for use!${NC}"
    echo -e "${BLUE}🌐 You can now start the application:${NC}"
    echo "   Backend:  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo "   Frontend: cd frontend && npm run dev"
    echo -e "${BLUE}🎯 Create your admin user at: http://localhost:8000/register${NC}"
}

# Handle Ctrl+C
trap 'echo -e "\n${RED}❌ Setup interrupted by user${NC}"; exit 1' INT

# Run main function
main "$@"
