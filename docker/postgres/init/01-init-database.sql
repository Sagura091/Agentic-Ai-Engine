-- OPTIMIZED Agentic AI Database Initialization (Docker)
-- This script sets up the ESSENTIAL database structure for Docker container startup
--
-- OPTIMIZED SCHEMA: Focused on core functionality, removed unnecessary complexity
-- ✅ PRESERVED: Extensions, schemas, types, functions for full system support
-- ❌ REMOVED: Unnecessary tables (use migration system for full features)

-- Create extensions for advanced features
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS workflows;
CREATE SCHEMA IF NOT EXISTS tools;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS autonomous;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA agents TO agentic_user;
GRANT ALL PRIVILEGES ON SCHEMA workflows TO agentic_user;
GRANT ALL PRIVILEGES ON SCHEMA tools TO agentic_user;
GRANT ALL PRIVILEGES ON SCHEMA rag TO agentic_user;
GRANT ALL PRIVILEGES ON SCHEMA autonomous TO agentic_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA agents GRANT ALL ON TABLES TO agentic_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA workflows GRANT ALL ON TABLES TO agentic_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA tools GRANT ALL ON TABLES TO agentic_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT ALL ON TABLES TO agentic_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA autonomous GRANT ALL ON TABLES TO agentic_user;

-- Create custom types for autonomous agents
CREATE TYPE autonomy_level AS ENUM ('reactive', 'proactive', 'adaptive', 'autonomous');
CREATE TYPE goal_type AS ENUM ('achievement', 'maintenance', 'exploration', 'optimization', 'learning', 'collaboration');
CREATE TYPE goal_priority AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE goal_status AS ENUM ('pending', 'active', 'paused', 'completed', 'failed', 'cancelled');
CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural', 'working', 'emotional');
CREATE TYPE memory_importance AS ENUM ('temporary', 'low', 'medium', 'high', 'critical');

-- Create function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create function for generating UUIDs
CREATE OR REPLACE FUNCTION generate_uuid()
RETURNS UUID AS $$
BEGIN
    RETURN uuid_generate_v4();
END;
$$ language 'plpgsql';

-- Create migration tracking table to track applied migrations
CREATE TABLE IF NOT EXISTS migration_history (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    execution_time_ms INTEGER
);

-- Create indexes for migration tracking
CREATE INDEX IF NOT EXISTS idx_migration_history_migration_name ON migration_history(migration_name);
CREATE INDEX IF NOT EXISTS idx_migration_history_applied_at ON migration_history(applied_at);

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'OPTIMIZED Agentic AI Database initialized successfully!';
    RAISE NOTICE 'PostgreSQL version: %', version();
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'User: %', current_user;
    RAISE NOTICE '';
    RAISE NOTICE 'NEXT STEPS:';
    RAISE NOTICE '1. For basic functionality: Database is ready to use';
    RAISE NOTICE '2. For full features: Run migration system';
    RAISE NOTICE '   python db/migrations/migrate_database.py migrate';
    RAISE NOTICE '';
    RAISE NOTICE 'OPTIMIZED SCHEMA: Essential tables only, autonomous learning preserved';
END $$;
