"""
004_create_enhanced_tables.py
Production-Ready Database Migration System for Enhanced Platform Tables.

Migration Order: 004 (Fourth - Enhanced Platform Features)
Dependencies: 003_create_auth_tables.py
Next: 005_add_document_tables.py

This migration creates all necessary tables for the enhanced platform with:
- Model management and tracking
- Knowledge base database persistence
- Enhanced user management and authentication
- User-owned API key management (in auth.py)
- Proper async/await patterns
- Transaction management
- Error handling and rollback
- Index optimization
- Production-ready constraints

OPTIMIZED TABLES CREATED:
- Knowledge Bases: knowledge_bases, knowledge_base_access (ESSENTIAL for RAG)
- Enhanced Users: user_sessions (ESSENTIAL for session management)
- NOTE: User roles are integrated directly into users.user_group field (user/moderator/admin)

REMOVED UNNECESSARY TABLES:
- Model Management: model_registry, model_usage_logs, model_download_history, model_performance_metrics (handled by APIs)
- Logging/Audit: knowledge_base_usage_logs, user_audit_logs (not needed for core functionality)
- Templates: knowledge_base_templates (unnecessary complexity)
- RBAC Tables: roles, user_role_assignments (roles now in users.user_group field)
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Add project root to path for imports (go up two levels: migrations/ -> db/ -> project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.models.database.base import Base, get_engine, get_session_factory
from app.config.settings import get_settings

# Import ESSENTIAL models only (optimized schema)
from app.models.knowledge_base import (
    KnowledgeBase, KnowledgeBaseAccess
    # REMOVED: KnowledgeBaseUsageLog, KnowledgeBaseTemplate (unnecessary complexity)
)
from app.models.auth import UserDB
from app.models.enhanced_user import (
    UserSession
    # REMOVED: Role, UserRoleAssignment, UserAuditLog (roles in users.user_group, audit not needed)
)
# REMOVED: All model_management imports (handled by external APIs)
# REMOVED: API management models (replaced with user-owned API key system)

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

            # Create tables using engine connection
            async with self.engine.begin() as conn:
                # Create all tables using SQLAlchemy metadata
                await conn.run_sync(Base.metadata.create_all)
                logger.info("All enhanced tables created successfully")

            # Now use session for additional operations
            async with self.session_factory() as session:
                async with session.begin():
                    # Create performance indices
                    await self._create_performance_indices(session)

                    # Create full-text search indices
                    await self._create_fulltext_indices(session)

                    # Create additional constraints
                    await self._create_additional_constraints(session)

                    # Insert default data
                    await self._insert_default_data(session)

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
            # REMOVED: Model management indices (tables not created in optimized schema)
            # Knowledge base indices (ESSENTIAL)
            
            # Knowledge base indices (ESSENTIAL - only for tables that exist)
            "CREATE INDEX IF NOT EXISTS idx_knowledge_bases_public_active ON rag.knowledge_bases(is_public, status) WHERE status = 'active';",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_base_access_active ON rag.knowledge_base_access(is_active, access_level) WHERE is_active = true;",

            # User management indices (OPTIMIZED - only for columns that exist)
            "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = true;",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active, expires_at) WHERE is_active = true;",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);",
        ]
        
        for index_sql in indices:
            try:
                await session.execute(text(index_sql))
                logger.debug("Created index", sql=index_sql)
            except Exception as e:
                logger.warning("Failed to create index", sql=index_sql, error=str(e))
    
    async def _create_fulltext_indices(self, session: AsyncSession) -> None:
        """Create full-text search indices (OPTIMIZED - only for existing tables)."""
        fulltext_indices = [
            # Knowledge base full-text search (ESSENTIAL)
            "CREATE INDEX IF NOT EXISTS idx_knowledge_bases_fulltext ON rag.knowledge_bases USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));",
        ]

        for index_sql in fulltext_indices:
            try:
                await session.execute(text(index_sql))
                logger.debug("Created full-text index", sql=index_sql)
            except Exception as e:
                logger.warning("Failed to create full-text index", sql=index_sql, error=str(e))
    
    async def _create_additional_constraints(self, session: AsyncSession) -> None:
        """Create additional database constraints (OPTIMIZED - only for existing tables)."""
        constraints = [
            # Ensure positive values (only for tables that exist)
            "ALTER TABLE rag.knowledge_bases ADD CONSTRAINT IF NOT EXISTS chk_knowledge_bases_size_positive CHECK (size_mb >= 0);",

            # Ensure valid enum values (only for tables that exist)
            "ALTER TABLE rag.knowledge_bases ADD CONSTRAINT IF NOT EXISTS chk_kb_status_valid CHECK (status IN ('active', 'inactive', 'processing', 'error'));",
        ]

        for constraint_sql in constraints:
            try:
                await session.execute(text(constraint_sql))
                logger.debug("Created constraint", sql=constraint_sql)
            except Exception as e:
                logger.warning("Failed to create constraint", sql=constraint_sql, error=str(e))
    
    async def _insert_default_data(self, session: AsyncSession) -> None:
        """Insert default data for the enhanced tables (OPTIMIZED - roles removed, now in users.user_group)."""
        try:
            # REMOVED: Default roles insertion (roles table doesn't exist in optimized schema)
            # Roles are now handled via users.user_group field (user/moderator/admin)
            default_data = [
                # No default data needed for optimized schema
            ]

            # No default data to insert in optimized schema
            logger.info("No default data needed for optimized schema (roles in users.user_group)")
            
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
