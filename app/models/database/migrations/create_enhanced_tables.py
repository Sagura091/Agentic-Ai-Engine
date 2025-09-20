"""
Production-Ready Database Migration System for Enhanced Platform Tables.

This migration creates all necessary tables for the enhanced platform with:
- Model management and tracking
- Knowledge base database persistence
- Enhanced user management and authentication
- API key management and rate limiting
- Proper async/await patterns
- Transaction management
- Error handling and rollback
- Index optimization
- Production-ready constraints

Tables created:
- Model Management: model_registry, model_usage_logs, model_download_history, model_performance_metrics
- Knowledge Bases: knowledge_bases, knowledge_base_access, knowledge_base_usage_logs, knowledge_base_templates
- Enhanced Users: users, user_sessions, roles, user_role_assignments, user_audit_logs
- API Management: api_keys, api_key_usage_logs, rate_limit_logs, api_quota_usage, api_endpoint_metrics
"""

import asyncio
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from app.models.database.base import Base, get_engine, get_session_factory
from app.config.settings import get_settings

# Import all new models to ensure they are registered
from app.models.model_management import (
    ModelRegistry, ModelUsageLog, ModelDownloadHistory, ModelPerformanceMetrics
)
from app.models.knowledge_base import (
    KnowledgeBase, KnowledgeBaseAccess, KnowledgeBaseUsageLog, KnowledgeBaseTemplate
)
from app.models.enhanced_user import (
    UserDB, UserSession, Role, UserRoleAssignment, UserAuditLog
)
from app.models.api_management import (
    APIKey, APIKeyUsageLog, RateLimitLog, APIQuotaUsage, APIEndpointMetrics
)

import structlog

logger = structlog.get_logger(__name__)


class EnhancedMigrationError(Exception):
    """Custom exception for enhanced migration errors."""
    pass


class EnhancedTablesMigration:
    """
    Production-ready migration system for enhanced platform tables.
    
    Features:
    - Async/await support
    - Transaction management
    - Error handling and rollback
    - Performance optimization
    - Index creation
    - Constraint management
    """
    
    def __init__(self):
        """Initialize the migration system."""
        self.engine = get_engine()
        self.session_factory = get_session_factory()
        self.migration_name = "create_enhanced_tables_v1"
        
    async def is_migration_applied(self, migration_name: str) -> bool:
        """Check if migration has already been applied."""
        try:
            async with self.session_factory() as session:
                # Check if migration tracking table exists
                result = await session.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'migration_history'
                    );
                """))
                
                table_exists = result.scalar()
                
                if not table_exists:
                    # Create migration tracking table
                    await session.execute(text("""
                        CREATE TABLE IF NOT EXISTS migration_history (
                            id SERIAL PRIMARY KEY,
                            migration_name VARCHAR(255) UNIQUE NOT NULL,
                            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            success BOOLEAN DEFAULT TRUE,
                            error_message TEXT
                        );
                    """))
                    await session.commit()
                    return False
                
                # Check if this specific migration was applied
                result = await session.execute(text("""
                    SELECT success FROM migration_history 
                    WHERE migration_name = :migration_name
                """), {"migration_name": migration_name})
                
                migration_record = result.fetchone()
                return migration_record is not None and migration_record[0]
                
        except Exception as e:
            logger.error("Error checking migration status", error=str(e))
            return False
    
    async def record_migration(self, migration_name: str, success: bool, error_message: Optional[str] = None) -> None:
        """Record migration attempt in history."""
        try:
            async with self.session_factory() as session:
                await session.execute(text("""
                    INSERT INTO migration_history (migration_name, success, error_message)
                    VALUES (:migration_name, :success, :error_message)
                    ON CONFLICT (migration_name) 
                    DO UPDATE SET 
                        success = EXCLUDED.success,
                        error_message = EXCLUDED.error_message,
                        applied_at = NOW()
                """), {
                    "migration_name": migration_name,
                    "success": success,
                    "error_message": error_message
                })
                await session.commit()
                
        except Exception as e:
            logger.error("Error recording migration", error=str(e))
    
    async def create_enhanced_tables(self) -> bool:
        """
        Create all enhanced platform tables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        migration_name = self.migration_name
        
        try:
            # Check if migration already applied
            if await self.is_migration_applied(migration_name):
                logger.info("Enhanced tables migration already applied", migration=migration_name)
                return True

            logger.info("Starting enhanced tables migration", migration=migration_name)

            async with self.session_factory() as session:
                async with session.begin():
                    # Create all tables using SQLAlchemy metadata
                    await session.run_sync(Base.metadata.create_all, self.engine)

                    # Create performance indices
                    await self._create_performance_indices(session)

                    # Create full-text search indices
                    await self._create_fulltext_indices(session)

                    # Create additional constraints
                    await self._create_additional_constraints(session)

                    # Insert default data
                    await self._insert_default_data(session)

                    logger.info("All enhanced tables created successfully")

            # Record successful migration
            await self.record_migration(migration_name, True)
            
            logger.info("Enhanced tables migration completed successfully", migration=migration_name)
            return True

        except Exception as e:
            error_msg = f"Enhanced tables migration failed: {str(e)}"
            logger.error(error_msg, migration=migration_name)
            
            # Record failed migration
            await self.record_migration(migration_name, False, error_msg)
            
            raise EnhancedMigrationError(error_msg) from e
    
    async def _create_performance_indices(self, session: AsyncSession) -> None:
        """Create performance-optimized indices."""
        indices = [
            # Model management indices
            "CREATE INDEX IF NOT EXISTS idx_model_registry_type_source ON rag.model_registry(model_type, model_source);",
            "CREATE INDEX IF NOT EXISTS idx_model_registry_downloaded ON rag.model_registry(is_downloaded) WHERE is_downloaded = true;",
            "CREATE INDEX IF NOT EXISTS idx_model_usage_logs_created_at ON rag.model_usage_logs(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_model_usage_logs_success ON rag.model_usage_logs(success, usage_type);",
            "CREATE INDEX IF NOT EXISTS idx_model_download_history_status ON rag.model_download_history(download_status, started_at);",
            
            # Knowledge base indices
            "CREATE INDEX IF NOT EXISTS idx_knowledge_bases_public_active ON rag.knowledge_bases(is_public, status) WHERE status = 'active';",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_base_access_active ON rag.knowledge_base_access(is_active, access_level) WHERE is_active = true;",
            "CREATE INDEX IF NOT EXISTS idx_kb_usage_logs_created_at ON rag.knowledge_base_usage_logs(created_at);",
            
            # User management indices
            "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active, is_verified) WHERE is_active = true;",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active, expires_at) WHERE is_active = true;",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);",
            "CREATE INDEX IF NOT EXISTS idx_user_audit_logs_created_at ON user_audit_logs(created_at);",
            
            # API management indices
            "CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active, is_revoked) WHERE is_active = true AND is_revoked = false;",
            "CREATE INDEX IF NOT EXISTS idx_api_key_usage_logs_created_at ON api_key_usage_logs(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_logs_violation ON rate_limit_logs(is_violation, created_at) WHERE is_violation = true;",
            "CREATE INDEX IF NOT EXISTS idx_api_quota_usage_date_type ON api_quota_usage(usage_date, quota_type);",
        ]
        
        for index_sql in indices:
            try:
                await session.execute(text(index_sql))
                logger.debug("Created index", sql=index_sql)
            except Exception as e:
                logger.warning("Failed to create index", sql=index_sql, error=str(e))
    
    async def _create_fulltext_indices(self, session: AsyncSession) -> None:
        """Create full-text search indices."""
        fulltext_indices = [
            # Knowledge base full-text search
            "CREATE INDEX IF NOT EXISTS idx_knowledge_bases_fulltext ON rag.knowledge_bases USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));",
            
            # Model registry full-text search
            "CREATE INDEX IF NOT EXISTS idx_model_registry_fulltext ON rag.model_registry USING gin(to_tsvector('english', model_name || ' ' || COALESCE(description, '')));",
        ]
        
        for index_sql in fulltext_indices:
            try:
                await session.execute(text(index_sql))
                logger.debug("Created full-text index", sql=index_sql)
            except Exception as e:
                logger.warning("Failed to create full-text index", sql=index_sql, error=str(e))
    
    async def _create_additional_constraints(self, session: AsyncSession) -> None:
        """Create additional database constraints."""
        constraints = [
            # Ensure positive values
            "ALTER TABLE rag.model_registry ADD CONSTRAINT chk_model_registry_size_positive CHECK (size_mb >= 0);",
            "ALTER TABLE rag.model_usage_logs ADD CONSTRAINT chk_model_usage_processing_time_positive CHECK (processing_time_ms >= 0);",
            "ALTER TABLE rag.knowledge_bases ADD CONSTRAINT chk_knowledge_bases_size_positive CHECK (size_mb >= 0);",
            "ALTER TABLE api_keys ADD CONSTRAINT chk_api_keys_rate_limits_positive CHECK (rate_limit_per_minute > 0);",
            
            # Ensure valid enum values
            "ALTER TABLE rag.model_registry ADD CONSTRAINT chk_model_type_valid CHECK (model_type IN ('embedding', 'reranking', 'vision', 'llm'));",
            "ALTER TABLE rag.model_registry ADD CONSTRAINT chk_model_source_valid CHECK (model_source IN ('huggingface', 'ollama', 'openai_api', 'anthropic_api', 'google_api'));",
            "ALTER TABLE rag.knowledge_bases ADD CONSTRAINT chk_kb_status_valid CHECK (status IN ('active', 'inactive', 'processing', 'error'));",
        ]
        
        for constraint_sql in constraints:
            try:
                await session.execute(text(constraint_sql))
                logger.debug("Created constraint", sql=constraint_sql)
            except Exception as e:
                logger.warning("Failed to create constraint", sql=constraint_sql, error=str(e))
    
    async def _insert_default_data(self, session: AsyncSession) -> None:
        """Insert default data for the enhanced tables."""
        try:
            # Insert default roles
            default_roles = [
                "INSERT INTO roles (id, name, display_name, description, is_system_role, permissions) VALUES (gen_random_uuid(), 'admin', 'Administrator', 'Full system access', true, '[\"*\"]') ON CONFLICT (name) DO NOTHING;",
                "INSERT INTO roles (id, name, display_name, description, is_system_role, permissions) VALUES (gen_random_uuid(), 'user', 'User', 'Standard user access', true, '[\"read\", \"create_agent\", \"use_rag\"]') ON CONFLICT (name) DO NOTHING;",
                "INSERT INTO roles (id, name, display_name, description, is_system_role, permissions) VALUES (gen_random_uuid(), 'developer', 'Developer', 'Developer access with API keys', true, '[\"read\", \"write\", \"api_access\"]') ON CONFLICT (name) DO NOTHING;",
            ]
            
            for role_sql in default_roles:
                await session.execute(text(role_sql))
            
            logger.info("Inserted default roles")
            
        except Exception as e:
            logger.warning("Failed to insert default data", error=str(e))


async def main():
    """Main migration execution function."""
    try:
        logger.info("Starting enhanced tables migration")
        
        migration = EnhancedTablesMigration()
        success = await migration.create_enhanced_tables()
        
        if success:
            logger.info("✅ Enhanced tables migration completed successfully!")
            return 0
        else:
            logger.error("❌ Enhanced tables migration failed!")
            return 1
            
    except Exception as e:
        logger.error("❌ Migration failed with exception", error=str(e))
        return 1


if __name__ == "__main__":
    # Run the migration
    exit_code = asyncio.run(main())
