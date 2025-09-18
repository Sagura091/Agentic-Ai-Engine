-- Initialize Agentic AI Database
-- This script sets up the initial database structure and extensions

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

-- Log initialization
INSERT INTO pg_stat_statements_info (dealloc) VALUES (0) ON CONFLICT DO NOTHING;

-- Create initial admin user (optional)
-- This can be used for application-level user management
-- CREATE TABLE IF NOT EXISTS app_users (
--     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     username VARCHAR(255) UNIQUE NOT NULL,
--     email VARCHAR(255) UNIQUE NOT NULL,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
--     updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
-- );

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Agentic AI Database initialized successfully';
    RAISE NOTICE 'PostgreSQL version: %', version();
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'User: %', current_user;
END $$;
