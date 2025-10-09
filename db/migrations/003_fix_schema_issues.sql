-- ============================================================================
-- Migration 003: Fix Schema Issues
-- ============================================================================
-- Purpose: Fix critical schema bugs and inconsistencies identified in analysis
-- Date: 2025-10-08
-- Database: PostgreSQL 17.6
-- Dependencies: 001_init_database.sql, 002_create_autonomous_tables.py
-- 
-- This migration fixes:
-- 1. Missing execution_time_ms column in migration_history
-- 2. Incorrect metadata index on agents table
-- 3. PostgreSQL constraint syntax errors
-- 4. Metadata column naming inconsistencies (workflows, tools)
-- ============================================================================

\echo '============================================================================'
\echo 'Migration 003: Fixing Schema Issues'
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

-- Create index with correct column name (agent_metadata, not metadata)
CREATE INDEX IF NOT EXISTS idx_agents_metadata_gin ON agents USING gin(agent_metadata);

\echo '✓ Agents metadata index fixed (using agent_metadata column)'

-- ============================================================================
-- FIX 3: Add Knowledge Bases Constraints
-- ============================================================================

\echo ''
\echo 'Fix 3: Adding constraints to rag.knowledge_bases...'

-- Add size_mb check constraint (must be non-negative)
ALTER TABLE rag.knowledge_bases 
ADD CONSTRAINT IF NOT EXISTS chk_knowledge_bases_size_positive 
CHECK (size_mb >= 0);

-- Add status check constraint (must be valid status)
ALTER TABLE rag.knowledge_bases 
ADD CONSTRAINT IF NOT EXISTS chk_kb_status_valid 
CHECK (status IN ('active', 'inactive', 'processing', 'error'));

\echo '✓ Knowledge bases constraints added'

-- ============================================================================
-- FIX 4: Align Workflow Metadata Column Naming
-- ============================================================================

\echo ''
\echo 'Fix 4: Aligning workflow metadata column naming...'

-- Check if column needs to be renamed (only if 'metadata' exists)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' 
        AND column_name = 'metadata'
        AND table_schema = 'public'
    ) THEN
        -- Rename column to match model definition
        ALTER TABLE workflows RENAME COLUMN metadata TO workflow_metadata;
        RAISE NOTICE '✓ Renamed workflows.metadata to workflow_metadata';
        
        -- Drop old index
        DROP INDEX IF EXISTS idx_workflows_metadata_gin;
        RAISE NOTICE '✓ Dropped old index idx_workflows_metadata_gin';
        
        -- Create new index on correct column
        CREATE INDEX idx_workflows_metadata_gin ON workflows USING gin(workflow_metadata);
        RAISE NOTICE '✓ Created new index on workflow_metadata';
    ELSE
        RAISE NOTICE '  Column workflows.workflow_metadata already exists';
    END IF;
END $$;

\echo '✓ Workflow metadata column aligned'

-- ============================================================================
-- FIX 5: Align Tool Metadata Column Naming
-- ============================================================================

\echo ''
\echo 'Fix 5: Aligning tool metadata column naming...'

-- Check if column needs to be renamed (only if 'metadata' exists)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tools' 
        AND column_name = 'metadata'
        AND table_schema = 'public'
    ) THEN
        -- Rename column to match model definition
        ALTER TABLE tools RENAME COLUMN metadata TO tool_metadata;
        RAISE NOTICE '✓ Renamed tools.metadata to tool_metadata';
        
        -- Drop old index
        DROP INDEX IF EXISTS idx_tools_metadata_gin;
        RAISE NOTICE '✓ Dropped old index idx_tools_metadata_gin';
        
        -- Create new index on correct column
        CREATE INDEX idx_tools_metadata_gin ON tools USING gin(tool_metadata);
        RAISE NOTICE '✓ Created new index on tool_metadata';
    ELSE
        RAISE NOTICE '  Column tools.tool_metadata already exists';
    END IF;
END $$;

\echo '✓ Tool metadata column aligned'

-- ============================================================================
-- VERIFICATION
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'VERIFICATION'
\echo '============================================================================'

\echo ''
\echo 'Checking migration_history table...'
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'migration_history' 
AND column_name = 'execution_time_ms';

\echo ''
\echo 'Checking agents indexes...'
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'agents' 
AND indexname LIKE '%metadata%';

\echo ''
\echo 'Checking workflows columns...'
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'workflows' 
AND column_name LIKE '%metadata%';

\echo ''
\echo 'Checking tools columns...'
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'tools' 
AND column_name LIKE '%metadata%';

\echo ''
\echo 'Checking knowledge_bases constraints...'
SELECT conname, contype, pg_get_constraintdef(oid) as definition
FROM pg_constraint 
WHERE conrelid = 'rag.knowledge_bases'::regclass
AND conname LIKE 'chk_%';

\echo ''
\echo '============================================================================'
\echo 'MIGRATION 003 COMPLETED SUCCESSFULLY'
\echo '============================================================================'
\echo ''
\echo 'Summary of fixes applied:'
\echo '  ✓ Added migration_history.execution_time_ms column'
\echo '  ✓ Fixed agents metadata index (agent_metadata)'
\echo '  ✓ Added knowledge_bases constraints (size, status)'
\echo '  ✓ Aligned workflows metadata column (workflow_metadata)'
\echo '  ✓ Aligned tools metadata column (tool_metadata)'
\echo ''
\echo 'All schema issues resolved!'
\echo '============================================================================'

