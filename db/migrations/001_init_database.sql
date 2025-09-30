-- 001_init_database.sql
-- OPTIMIZED Database Initialization for Agentic AI System
-- REVOLUTIONARY SIMPLIFIED SCHEMA - Essential Functionality Only
--
-- ✅ KEEPS: Core system, Autonomous learning, RAG, Workflows
-- ❌ REMOVES: Logs, Metrics, Audit, Model management, Project management
--
-- Migration Order: 001 (First - Database Foundation)
-- Dependencies: None
-- Next: 002_create_autonomous_tables.py

-- ===================================
-- EXTENSIONS & CORE SETUP
-- ===================================

-- Create extensions for advanced features
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ===================================
-- SCHEMA ORGANIZATION
-- ===================================

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

-- ===================================
-- CUSTOM TYPES FOR AUTONOMOUS AGENTS
-- ===================================

-- Autonomous agent types
CREATE TYPE autonomy_level AS ENUM ('reactive', 'proactive', 'adaptive', 'autonomous');
CREATE TYPE goal_type AS ENUM ('achievement', 'maintenance', 'exploration', 'optimization', 'learning', 'collaboration');
CREATE TYPE goal_priority AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE goal_status AS ENUM ('pending', 'active', 'paused', 'completed', 'failed', 'cancelled');
CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural', 'working', 'emotional');
CREATE TYPE memory_importance AS ENUM ('temporary', 'low', 'medium', 'high', 'critical');

-- ===================================
-- UTILITY FUNCTIONS
-- ===================================

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

-- ===================================
-- BASIC TABLES (from init-db.sql)
-- ===================================

-- Create agents table (SIMPLIFIED - removed unused performance tracking fields)
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type VARCHAR(100) NOT NULL DEFAULT 'general',

    -- LLM configuration
    model VARCHAR(255) NOT NULL DEFAULT 'llama3.2:latest',
    model_provider VARCHAR(50) NOT NULL DEFAULT 'ollama',
    temperature FLOAT DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 2048,

    -- Agent capabilities and tools
    capabilities JSONB DEFAULT '[]',
    tools JSONB DEFAULT '[]',
    system_prompt TEXT,

    -- Status and metadata
    status VARCHAR(50) DEFAULT 'active',
    last_activity TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Metadata
    agent_metadata JSONB DEFAULT '{}'

    -- REMOVED: autonomy_level, learning_mode, decision_threshold (handled by autonomous_agent_states)
    -- REMOVED: total_tasks_completed, total_tasks_failed, average_response_time (performance metrics not needed)
    -- REMOVED: duplicate metadata field
);

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    definition JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create tools table
CREATE TABLE IF NOT EXISTS tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tool_type VARCHAR(100) NOT NULL,
    configuration JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- ===================================
-- INDEXES FOR PERFORMANCE
-- ===================================

-- Basic table indexes
CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_created_at ON agents(created_at);

CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON workflows(created_at);

CREATE INDEX IF NOT EXISTS idx_tools_tool_type ON tools(tool_type);
CREATE INDEX IF NOT EXISTS idx_tools_status ON tools(status);
CREATE INDEX IF NOT EXISTS idx_tools_created_at ON tools(created_at);

-- JSONB indexes for metadata search
CREATE INDEX IF NOT EXISTS idx_agents_metadata_gin ON agents USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_workflows_metadata_gin ON workflows USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_tools_metadata_gin ON tools USING gin(metadata);

-- ===================================
-- TRIGGERS
-- ===================================

-- Update timestamp triggers
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_tools_updated_at BEFORE UPDATE ON tools FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===================================
-- MIGRATION TRACKING TABLE
-- ===================================

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

-- ===================================
-- INITIALIZATION LOGGING
-- ===================================

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Agentic AI Database initialized successfully';
    RAISE NOTICE 'PostgreSQL version: %', version();
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'User: %', current_user;
    RAISE NOTICE 'Schemas created: agents, workflows, tools, rag, autonomous';
    RAISE NOTICE 'Extensions enabled: uuid-ossp, pg_trgm, btree_gin, btree_gist';
    RAISE NOTICE 'Ready for migration 002_create_autonomous_tables.py';
END $$;
