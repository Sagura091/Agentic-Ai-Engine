# 🚀 OPTIMIZED Agentic AI Database Migrations

## 📋 Overview

This directory contains the **OPTIMIZED** database migrations for the Agentic AI platform. We've streamlined from 30+ complex tables to **17 essential tables** focused on core functionality.

**✅ PRESERVED:** Autonomous agent learning, memory, conversations, RAG, workflows
**❌ REMOVED:** Logs, metrics, audit trails, model management, project management

## 🔄 Migration Order

The migrations must be run in this exact order to maintain referential integrity:

### 1. **001_init_database.sql** - Database Foundation
- **Purpose**: Initialize PostgreSQL with extensions, schemas, and basic tables
- **Dependencies**: None
- **Creates**:
  - Extensions: uuid-ossp, pg_trgm, btree_gin, btree_gist
  - Schemas: agents, workflows, tools, rag, autonomous
  - Custom types for autonomous agents
  - Basic tables: agents, workflows, tools
  - Utility functions and triggers

### 2. **002_create_autonomous_tables.py** - Autonomous Agent Learning System ✅
- **Purpose**: Create ESSENTIAL tables for autonomous agent memory and learning
- **Dependencies**: 001_init_database.sql (agents table)
- **Creates** (PRESERVED - Critical for Learning):
  - `autonomous_agent_states` → references `agents.id` ✅
  - `autonomous_goals` → references `autonomous_agent_states.id` ✅
  - `autonomous_decisions` → references `autonomous_agent_states.id` ✅
  - `agent_memories` → references `autonomous_agent_states.id` ✅ (Deep Memory)
  - `learning_experiences` → references `agents.id` ✅
- **REMOVED**: `performance_metrics` (system metrics, not learning)

### 3. **003_create_auth_tables.py** - Authentication & User Management ✅
- **Purpose**: Create ESSENTIAL user management and authentication system
- **Dependencies**: 002_create_autonomous_tables.py
- **Creates** (OPTIMIZED):
  - `users` (with integrated roles via user_group field) ✅
  - `user_sessions` → references `users.id` ✅
  - `conversations` → references `users.id`, `agents.id` ✅
  - `messages` → references `conversations.id` ✅
  - `user_api_keys` → references `users.id` ✅
  - `user_agents` → references `users.id`, `agents.id` ✅
  - `user_workflows` → references `users.id`, `workflows.id` ✅
- **REMOVED**: `projects`, `project_members`, `notifications` (not implemented)
- **REMOVED**: `roles`, `user_role_assignments` (roles now in users.user_group)

### 4. **004_create_enhanced_tables.py** - Knowledge Base System ✅
- **Purpose**: Create ESSENTIAL knowledge base system for RAG
- **Dependencies**: 003_create_auth_tables.py (users table)
- **Creates** (OPTIMIZED):
  - `knowledge_bases` → references `users.id` (created_by) ✅
  - `knowledge_base_access` → references `knowledge_bases.id`, `users.id` ✅
  - `user_sessions` (enhanced session management) ✅
- **REMOVED**: All model management tables (handled by Ollama/APIs)
- **REMOVED**: `knowledge_base_usage_logs`, `knowledge_base_templates` (unnecessary complexity)

### 5. **005_add_document_tables.py** - Document Storage & RAG
- **Purpose**: Create document storage and RAG system tables
- **Dependencies**: 004_create_enhanced_tables.py (knowledge_bases table)
- **Creates**:
  - `rag.documents` → references knowledge_base_id
  - `rag.document_chunks` → references `rag.documents.id`

## 🔗 Foreign Key Relationships

### Core Relationships:
- **Users** are the central entity that owns agents, workflows, projects, and knowledge bases
- **Agents** can have autonomous states, conversations, and performance metrics
- **Autonomous Agent States** contain goals, decisions, and memories
- **Projects** contain conversations and have members
- **Knowledge Bases** contain documents and have access controls
- **Documents** are chunked for RAG processing

### Key Constraints:
- All foreign keys use CASCADE DELETE where appropriate
- User-owned resources are deleted when user is deleted
- Agent states are deleted when agent is deleted
- Document chunks are deleted when document is deleted

## 🛠️ Usage

### Run All Migrations:
```bash
python db/migrations/migrate_database.py migrate
```

### Check Migration Status:
```bash
python db/migrations/migrate_database.py status
```

### Check Database Health:
```bash
python db/migrations/migrate_database.py health
```

### Run Individual Migration:
```bash
# SQL migrations
psql -d agentic_ai -f db/migrations/001_init_database.sql

# Python migrations
python db/migrations/002_create_autonomous_tables.py
```

## 🔒 Safety Features

- **IF NOT EXISTS** checks prevent overwriting existing tables
- **Transaction management** ensures atomic operations
- **Error handling** with rollback capabilities
- **Migration tracking** prevents duplicate runs
- **Backup recommendations** before running migrations

## 📊 OPTIMIZED Database Schema

The streamlined database schema includes:
- **17 essential tables** (reduced from 30+) with proper relationships ✅
- **UUID primary keys** for all entities ✅
- **JSONB columns** for flexible metadata ✅
- **Full-text search** capabilities for RAG ✅
- **Autonomous agent learning** with deep memory ✅
- **Performance indexes** for optimal queries ✅
- **REMOVED**: Unnecessary logs, metrics, audit trails, model management

## 🎯 Best Practices

1. **Always backup** before running migrations
2. **Run migrations in order** - never skip steps
3. **Test on development** environment first
4. **Monitor logs** during migration execution
5. **Verify relationships** after migration completion

## 🚨 Troubleshooting

### Common Issues:
- **Foreign key violations**: Ensure migrations run in correct order
- **Duplicate table errors**: Check if tables already exist
- **Permission errors**: Verify database user has CREATE privileges
- **Connection errors**: Ensure PostgreSQL is running and accessible

### Recovery:
- Check migration logs in `logs/` directory
- Use `migrate_database.py health` to diagnose issues
- Restore from backup if needed
- Contact support with error logs
