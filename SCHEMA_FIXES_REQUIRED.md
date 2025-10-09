# 🔧 DATABASE SCHEMA FIXES REQUIRED

## Quick Summary

**Status:** Database is 85% operational with 3 critical bugs blocking full functionality

**Critical Issues:**
1. ❌ `agents.metadata` index failed - column doesn't exist (should be `agent_metadata`)
2. ❌ `migration_history.execution_time_ms` column missing - Python migrations fail to record
3. ❌ PostgreSQL syntax errors in constraint creation (`IF NOT EXISTS` not supported)

**Missing Tables:** 15 optional tables (meme system, admin settings, session docs, model management)

---

## 🚨 CRITICAL FIXES (DO THESE FIRST)

### Fix 1: Add Missing Column to migration_history

**Problem:** Python migrations try to insert `execution_time_ms` but column doesn't exist

**SQL Fix:**
```sql
ALTER TABLE migration_history ADD COLUMN execution_time_ms INTEGER;
```

**Verification:**
```sql
\d migration_history
-- Should show: execution_time_ms | integer
```

---

### Fix 2: Fix Agents Metadata Index

**Problem:** SQL script tries to create index on `agents.metadata` but column is named `agent_metadata`

**SQL Fix:**
```sql
-- Drop failed index attempt (if exists)
DROP INDEX IF EXISTS idx_agents_metadata_gin;

-- Create index with correct column name
CREATE INDEX idx_agents_metadata_gin ON agents USING gin(agent_metadata);
```

**Verification:**
```sql
\d agents
-- Should show: "idx_agents_metadata_gin" gin (agent_metadata)
```

---

### Fix 3: Fix Constraint Creation Syntax

**Problem:** PostgreSQL doesn't support `IF NOT EXISTS` in `ALTER TABLE ADD CONSTRAINT`

**Current (FAILING):**
```sql
ALTER TABLE rag.knowledge_bases ADD CONSTRAINT IF NOT EXISTS chk_knowledge_bases_size_positive CHECK (size_mb >= 0);
```

**SQL Fix:**
```sql
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_knowledge_bases_size_positive'
    ) THEN
        ALTER TABLE rag.knowledge_bases ADD CONSTRAINT chk_knowledge_bases_size_positive CHECK (size_mb >= 0);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_kb_status_valid'
    ) THEN
        ALTER TABLE rag.knowledge_bases ADD CONSTRAINT chk_kb_status_valid CHECK (status IN ('active', 'inactive', 'processing', 'error'));
    END IF;
END $$;
```

---

## ⚠️ SCHEMA CONSISTENCY FIXES (RECOMMENDED)

### Fix 4: Align Workflow Metadata Column

**Problem:** SQL has `workflows.metadata` but model uses `workflow_metadata`

**Option A: Update Database (RECOMMENDED)**
```sql
ALTER TABLE workflows RENAME COLUMN metadata TO workflow_metadata;
DROP INDEX IF EXISTS idx_workflows_metadata_gin;
CREATE INDEX idx_workflows_metadata_gin ON workflows USING gin(workflow_metadata);
```

**Option B: Update Model**
```python
# In app/models/workflow.py, change:
workflow_metadata = Column(JSON, default=dict)
# To:
metadata = Column(JSON, default=dict)
```

---

### Fix 5: Align Tool Metadata Column

**Problem:** SQL has `tools.metadata` but model uses `tool_metadata`

**Option A: Update Database (RECOMMENDED)**
```sql
ALTER TABLE tools RENAME COLUMN metadata TO tool_metadata;
DROP INDEX IF EXISTS idx_tools_metadata_gin;
CREATE INDEX idx_tools_metadata_gin ON tools USING gin(tool_metadata);
```

**Option B: Update Model**
```python
# In app/models/tool.py, change:
tool_metadata = Column(JSON, default=dict)
# To:
metadata = Column(JSON, default=dict)
```

---

## 📋 MISSING TABLES ANALYSIS

### Model Management Tables (HIGH PRIORITY)

These tables are critical for the RAG system's model management functionality:

```sql
-- Create model_registry table in rag schema
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

CREATE INDEX idx_model_registry_model_id ON rag.model_registry(model_id);
CREATE INDEX idx_model_registry_model_type ON rag.model_registry(model_type);
CREATE INDEX idx_model_registry_is_downloaded ON rag.model_registry(is_downloaded);
CREATE INDEX idx_model_registry_is_available ON rag.model_registry(is_available);

-- Create model_usage_logs table in rag schema
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

CREATE INDEX idx_model_usage_logs_model_id ON rag.model_usage_logs(model_id);
CREATE INDEX idx_model_usage_logs_agent_id ON rag.model_usage_logs(agent_id);
CREATE INDEX idx_model_usage_logs_usage_type ON rag.model_usage_logs(usage_type);
CREATE INDEX idx_model_usage_logs_success ON rag.model_usage_logs(success);
CREATE INDEX idx_model_usage_logs_session_id ON rag.model_usage_logs(session_id);

-- Create model_download_history table in rag schema
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

CREATE INDEX idx_model_download_history_model_id ON rag.model_download_history(model_id);
CREATE INDEX idx_model_download_history_status ON rag.model_download_history(download_status);

-- Create model_performance_metrics table in rag schema
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

CREATE INDEX idx_model_performance_metrics_model_id ON rag.model_performance_metrics(model_id);
CREATE INDEX idx_model_performance_metrics_date ON rag.model_performance_metrics(metric_date);
```

---

### Meme System Tables (CONDITIONAL - Only if using meme features)

**Tables:** `memes`, `meme_analysis`, `generated_memes`, `meme_templates`, `meme_agent_states`, `meme_trends`

**Recommendation:** Create only if you're actively using meme collection/generation features

---

### Admin Settings Tables (CONDITIONAL - Only if using admin panel)

**Tables:** `admin_settings`, `admin_setting_history`, `system_configuration_cache`

**Recommendation:** Create only if you're using the enhanced admin panel

---

### Session Document Tables (CONDITIONAL - Only if using session workspace)

**Tables:** `session_documents`, `session_workspaces`

**Recommendation:** Create only if you're using session-based document workspace feature

---

## 🎯 EXECUTION PLAN

### Step 1: Apply Critical Fixes (5 minutes)
```bash
# Connect to database
docker exec -i agentic-postgres psql -U agentic_user -d agentic_ai

# Run fixes 1-3
ALTER TABLE migration_history ADD COLUMN execution_time_ms INTEGER;

DROP INDEX IF EXISTS idx_agents_metadata_gin;
CREATE INDEX idx_agents_metadata_gin ON agents USING gin(agent_metadata);

-- Add constraints with proper syntax (see Fix 3 above)
```

### Step 2: Apply Schema Consistency Fixes (5 minutes)
```bash
# Run fixes 4-5 (Option A - update database)
ALTER TABLE workflows RENAME COLUMN metadata TO workflow_metadata;
DROP INDEX IF EXISTS idx_workflows_metadata_gin;
CREATE INDEX idx_workflows_metadata_gin ON workflows USING gin(workflow_metadata);

ALTER TABLE tools RENAME COLUMN metadata TO tool_metadata;
DROP INDEX IF EXISTS idx_tools_metadata_gin;
CREATE INDEX idx_tools_metadata_gin ON tools USING gin(tool_metadata);
```

### Step 3: Create Model Management Tables (10 minutes)
```bash
# Run the model management table creation SQL from above
```

### Step 4: Verify Everything Works
```bash
# Re-run migrations
cd C:/Users/RR442821/Agentic-Ai-Engine
python db/migrations/run_all_migrations.py

# Should complete without errors
```

---

## ✅ VERIFICATION CHECKLIST

After applying fixes, verify:

- [ ] `migration_history` has `execution_time_ms` column
- [ ] `agents` table has GIN index on `agent_metadata`
- [ ] `workflows` table has `workflow_metadata` column (or model updated)
- [ ] `tools` table has `tool_metadata` column (or model updated)
- [ ] Constraints created successfully on `rag.knowledge_bases`
- [ ] Model management tables created in `rag` schema
- [ ] Python migrations run without errors
- [ ] No SQL warnings in migration output

---

## 📊 CURRENT STATUS

**Tables Created:** 35/50 (70%)
- ✅ Core system: 100%
- ✅ Autonomous agents: 100%
- ✅ RAG system: 60% (missing model management)
- ❌ Meme system: 0%
- ❌ Admin settings: 0%
- ❌ Session documents: 0%

**Critical Bugs:** 3 (all fixable in 15 minutes)

**Schema Consistency:** 2 mismatches (fixable in 5 minutes)

**Overall Health:** 85% operational, 15% needs fixes/additions

---

**See DATABASE_SCHEMA_ANALYSIS.md for complete detailed analysis**

