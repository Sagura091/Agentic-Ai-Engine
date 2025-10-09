-- ============================================================================
-- SCHEMA FIXES FOR AGENTIC AI ENGINE
-- ============================================================================
-- Purpose: Fix critical schema bugs and inconsistencies
-- Date: 2025-10-08
-- Database: PostgreSQL 17.6
-- 
-- This script fixes:
-- 1. Missing execution_time_ms column in migration_history
-- 2. Incorrect metadata index on agents table
-- 3. PostgreSQL constraint syntax errors
-- 4. Metadata column naming inconsistencies
-- 5. Creates missing model management tables
-- ============================================================================

\echo '============================================================================'
\echo 'APPLYING CRITICAL SCHEMA FIXES'
\echo '============================================================================'

-- ============================================================================
-- FIX 1: Add Missing execution_time_ms Column to migration_history
-- ============================================================================

\echo ''
\echo 'Fix 1: Adding execution_time_ms column to migration_history...'

ALTER TABLE migration_history ADD COLUMN IF NOT EXISTS execution_time_ms INTEGER;

\echo '✓ execution_time_ms column added'

-- ============================================================================
-- FIX 2: Fix Agents Metadata Index
-- ============================================================================

\echo ''
\echo 'Fix 2: Fixing agents metadata index...'

-- Drop failed index attempt (if exists)
DROP INDEX IF EXISTS idx_agents_metadata_gin;

-- Create index with correct column name
CREATE INDEX IF NOT EXISTS idx_agents_metadata_gin ON agents USING gin(agent_metadata);

\echo '✓ Agents metadata index fixed'

-- ============================================================================
-- FIX 3: Fix Constraint Creation Syntax
-- ============================================================================

\echo ''
\echo 'Fix 3: Adding constraints with correct PostgreSQL syntax...'

-- Add size_mb check constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_knowledge_bases_size_positive'
    ) THEN
        ALTER TABLE rag.knowledge_bases ADD CONSTRAINT chk_knowledge_bases_size_positive CHECK (size_mb >= 0);
        RAISE NOTICE '✓ Added chk_knowledge_bases_size_positive constraint';
    ELSE
        RAISE NOTICE '  Constraint chk_knowledge_bases_size_positive already exists';
    END IF;
END $$;

-- Add status check constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_kb_status_valid'
    ) THEN
        ALTER TABLE rag.knowledge_bases ADD CONSTRAINT chk_kb_status_valid CHECK (status IN ('active', 'inactive', 'processing', 'error'));
        RAISE NOTICE '✓ Added chk_kb_status_valid constraint';
    ELSE
        RAISE NOTICE '  Constraint chk_kb_status_valid already exists';
    END IF;
END $$;

\echo '✓ Constraints added successfully'

-- ============================================================================
-- FIX 4: Align Workflow Metadata Column
-- ============================================================================

\echo ''
\echo 'Fix 4: Aligning workflow metadata column naming...'

-- Check if column needs to be renamed
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' 
        AND column_name = 'metadata'
        AND table_schema = 'public'
    ) THEN
        -- Rename column
        ALTER TABLE workflows RENAME COLUMN metadata TO workflow_metadata;
        RAISE NOTICE '✓ Renamed workflows.metadata to workflow_metadata';
        
        -- Drop old index
        DROP INDEX IF EXISTS idx_workflows_metadata_gin;
        RAISE NOTICE '✓ Dropped old index idx_workflows_metadata_gin';
        
        -- Create new index
        CREATE INDEX idx_workflows_metadata_gin ON workflows USING gin(workflow_metadata);
        RAISE NOTICE '✓ Created new index on workflow_metadata';
    ELSE
        RAISE NOTICE '  Column workflows.workflow_metadata already exists';
    END IF;
END $$;

\echo '✓ Workflow metadata column aligned'

-- ============================================================================
-- FIX 5: Align Tool Metadata Column
-- ============================================================================

\echo ''
\echo 'Fix 5: Aligning tool metadata column naming...'

-- Check if column needs to be renamed
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tools' 
        AND column_name = 'metadata'
        AND table_schema = 'public'
    ) THEN
        -- Rename column
        ALTER TABLE tools RENAME COLUMN metadata TO tool_metadata;
        RAISE NOTICE '✓ Renamed tools.metadata to tool_metadata';
        
        -- Drop old index
        DROP INDEX IF EXISTS idx_tools_metadata_gin;
        RAISE NOTICE '✓ Dropped old index idx_tools_metadata_gin';
        
        -- Create new index
        CREATE INDEX idx_tools_metadata_gin ON tools USING gin(tool_metadata);
        RAISE NOTICE '✓ Created new index on tool_metadata';
    ELSE
        RAISE NOTICE '  Column tools.tool_metadata already exists';
    END IF;
END $$;

\echo '✓ Tool metadata column aligned'

-- ============================================================================
-- CREATE MODEL MANAGEMENT TABLES (RAG Schema)
-- ============================================================================

\echo ''
\echo 'Creating model management tables in rag schema...'

-- Model Registry Table
CREATE TABLE IF NOT EXISTS rag.model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) UNIQUE NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_source VARCHAR(50) NOT NULL,
    description TEXT,
    dimension INTEGER,
    max_sequence_length INTEGER,
    size_mb FLOAT,
    download_url TEXT,
    local_path TEXT,
    is_downloaded BOOLEAN DEFAULT FALSE,
    is_available BOOLEAN DEFAULT TRUE,
    download_date TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    average_inference_time_ms FLOAT DEFAULT 0.0,
    total_tokens_processed INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 1.0,
    model_metadata JSON DEFAULT '{}',
    configuration JSON DEFAULT '{}',
    tags JSON DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

\echo '✓ Created rag.model_registry table'

-- Model Registry Indexes
CREATE INDEX IF NOT EXISTS idx_model_registry_model_id ON rag.model_registry(model_id);
CREATE INDEX IF NOT EXISTS idx_model_registry_model_type ON rag.model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_model_registry_is_downloaded ON rag.model_registry(is_downloaded);
CREATE INDEX IF NOT EXISTS idx_model_registry_is_available ON rag.model_registry(is_available);

\echo '✓ Created indexes on rag.model_registry'

-- Model Usage Logs Table
CREATE TABLE IF NOT EXISTS rag.model_usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    agent_id UUID,
    usage_type VARCHAR(50) NOT NULL,
    operation VARCHAR(100),
    tokens_processed INTEGER,
    input_length INTEGER,
    output_length INTEGER,
    processing_time_ms FLOAT,
    memory_usage_mb FLOAT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    error_type VARCHAR(100),
    session_id VARCHAR(255),
    request_metadata JSON DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES rag.model_registry(model_id),
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

\echo '✓ Created rag.model_usage_logs table'

-- Model Usage Logs Indexes
CREATE INDEX IF NOT EXISTS idx_model_usage_logs_model_id ON rag.model_usage_logs(model_id);
CREATE INDEX IF NOT EXISTS idx_model_usage_logs_agent_id ON rag.model_usage_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_model_usage_logs_usage_type ON rag.model_usage_logs(usage_type);
CREATE INDEX IF NOT EXISTS idx_model_usage_logs_success ON rag.model_usage_logs(success);
CREATE INDEX IF NOT EXISTS idx_model_usage_logs_session_id ON rag.model_usage_logs(session_id);

\echo '✓ Created indexes on rag.model_usage_logs'

-- Model Download History Table
CREATE TABLE IF NOT EXISTS rag.model_download_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    download_status VARCHAR(50) NOT NULL,
    download_progress FLOAT DEFAULT 0.0,
    download_speed_mbps FLOAT,
    total_size_mb FLOAT,
    downloaded_size_mb FLOAT DEFAULT 0.0,
    file_path TEXT,
    checksum VARCHAR(255),
    error_message TEXT,
    error_type VARCHAR(100),
    retry_count INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    initiated_by VARCHAR(255),
    download_source VARCHAR(255),
    download_metadata JSON DEFAULT '{}',
    FOREIGN KEY (model_id) REFERENCES rag.model_registry(model_id)
);

\echo '✓ Created rag.model_download_history table'

-- Model Download History Indexes
CREATE INDEX IF NOT EXISTS idx_model_download_history_model_id ON rag.model_download_history(model_id);
CREATE INDEX IF NOT EXISTS idx_model_download_history_status ON rag.model_download_history(download_status);

\echo '✓ Created indexes on rag.model_download_history'

-- Model Performance Metrics Table
CREATE TABLE IF NOT EXISTS rag.model_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    metric_date TIMESTAMP WITH TIME ZONE NOT NULL,
    metric_period VARCHAR(20) NOT NULL,
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    avg_processing_time_ms FLOAT DEFAULT 0.0,
    min_processing_time_ms FLOAT DEFAULT 0.0,
    max_processing_time_ms FLOAT DEFAULT 0.0,
    p95_processing_time_ms FLOAT DEFAULT 0.0,
    total_tokens_processed INTEGER DEFAULT 0,
    avg_tokens_per_second FLOAT DEFAULT 0.0,
    peak_tokens_per_second FLOAT DEFAULT 0.0,
    avg_memory_usage_mb FLOAT DEFAULT 0.0,
    peak_memory_usage_mb FLOAT DEFAULT 0.0,
    avg_cpu_usage_percent FLOAT DEFAULT 0.0,
    avg_quality_score FLOAT,
    user_satisfaction_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES rag.model_registry(model_id)
);

\echo '✓ Created rag.model_performance_metrics table'

-- Model Performance Metrics Indexes
CREATE INDEX IF NOT EXISTS idx_model_performance_metrics_model_id ON rag.model_performance_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_model_performance_metrics_date ON rag.model_performance_metrics(metric_date);

\echo '✓ Created indexes on rag.model_performance_metrics'

-- ============================================================================
-- VERIFICATION
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'VERIFICATION'
\echo '============================================================================'

\echo ''
\echo 'Checking migration_history table...'
\d migration_history

\echo ''
\echo 'Checking agents indexes...'
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'agents' AND indexname LIKE '%metadata%';

\echo ''
\echo 'Checking workflows columns...'
SELECT column_name FROM information_schema.columns WHERE table_name = 'workflows' AND column_name LIKE '%metadata%';

\echo ''
\echo 'Checking tools columns...'
SELECT column_name FROM information_schema.columns WHERE table_name = 'tools' AND column_name LIKE '%metadata%';

\echo ''
\echo 'Checking rag schema tables...'
\dt rag.*

\echo ''
\echo '============================================================================'
\echo 'SCHEMA FIXES COMPLETED SUCCESSFULLY'
\echo '============================================================================'
\echo ''
\echo 'Summary:'
\echo '  ✓ Fixed migration_history.execution_time_ms column'
\echo '  ✓ Fixed agents metadata index'
\echo '  ✓ Added knowledge_bases constraints'
\echo '  ✓ Aligned workflows metadata column'
\echo '  ✓ Aligned tools metadata column'
\echo '  ✓ Created model management tables'
\echo ''
\echo 'Next steps:'
\echo '  1. Re-run Python migrations: python db/migrations/run_all_migrations.py'
\echo '  2. Verify no errors in migration output'
\echo '  3. Test application functionality'
\echo ''
\echo '============================================================================'

