"""
Production-Ready Database Migration System for Autonomous Agent Tables.

This migration creates all necessary tables for autonomous agent persistence with:
- Proper async/await patterns
- Transaction management
- Error handling and rollback
- Index optimization
- Production-ready constraints

Tables created:
- autonomous_agent_states: Core agent state persistence
- autonomous_goals: Goal management and tracking
- autonomous_decisions: Decision history and learning
- agent_memories: Memory storage and retrieval
- learning_experiences: Learning data and insights
- performance_metrics: Performance tracking
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
from app.models.autonomous import (
    AutonomousAgentState,
    AutonomousGoalDB,
    AutonomousDecisionDB,
    AgentMemoryDB,
    LearningExperienceDB,
    PerformanceMetricDB
)
from app.models.document import DocumentDB, DocumentChunkDB

import structlog

logger = structlog.get_logger(__name__)


class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass


class DatabaseMigrationManager:
    """
    Production-ready database migration manager.

    Handles all database migrations with proper async patterns,
    transaction management, and error recovery.
    """

    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.session_factory = None

    async def initialize(self) -> None:
        """Initialize the migration manager."""
        try:
            self.engine = get_engine()
            self.session_factory = get_session_factory()
            logger.info("Migration manager initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize migration manager", error=str(e))
            raise MigrationError(f"Migration manager initialization failed: {e}")

    async def check_database_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("SELECT 1"))
                result.scalar()
                logger.info("Database connection verified")
                return True
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            return False


    async def create_migration_tracking_table(self) -> None:
        """Create migration tracking table to track applied migrations."""
        try:
            async with self.session_factory() as session:
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS migration_history (
                        id SERIAL PRIMARY KEY,
                        migration_name VARCHAR(255) UNIQUE NOT NULL,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN NOT NULL DEFAULT TRUE,
                        error_message TEXT,
                        execution_time_ms INTEGER
                    );
                """))

                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_migration_history_migration_name
                    ON migration_history(migration_name);
                """))

                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_migration_history_applied_at
                    ON migration_history(applied_at);
                """))

                await session.commit()
                logger.info("Migration tracking table created")

        except Exception as e:
            logger.error("Failed to create migration tracking table", error=str(e))
            raise MigrationError(f"Migration tracking table creation failed: {e}")

    async def is_migration_applied(self, migration_name: str) -> bool:
        """Check if a migration has already been applied."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("""
                    SELECT success FROM migration_history
                    WHERE migration_name = :migration_name AND success = TRUE
                """), {"migration_name": migration_name})

                return result.scalar() is not None

        except Exception as e:
            logger.warning("Could not check migration status", migration=migration_name, error=str(e))
            return False

    async def record_migration(self, migration_name: str, success: bool,
                             execution_time_ms: int, error_message: Optional[str] = None) -> None:
        """Record migration execution in tracking table."""
        try:
            async with self.session_factory() as session:
                await session.execute(text("""
                    INSERT INTO migration_history (migration_name, success, execution_time_ms, error_message)
                    VALUES (:migration_name, :success, :execution_time_ms, :error_message)
                    ON CONFLICT (migration_name) DO UPDATE SET
                        applied_at = CURRENT_TIMESTAMP,
                        success = :success,
                        execution_time_ms = :execution_time_ms,
                        error_message = :error_message
                """), {
                    "migration_name": migration_name,
                    "success": success,
                    "execution_time_ms": execution_time_ms,
                    "error_message": error_message
                })
                await session.commit()

        except Exception as e:
            logger.error("Failed to record migration", migration=migration_name, error=str(e))

    async def create_autonomous_tables(self) -> bool:
        """Create all autonomous agent tables with proper async handling."""
        migration_name = "create_autonomous_tables_v1"
        start_time = datetime.now()

        try:
            # Check if migration already applied
            if await self.is_migration_applied(migration_name):
                logger.info("Migration already applied", migration=migration_name)
                return True

            logger.info("Starting autonomous tables migration", migration=migration_name)

            # Create tables using engine directly
            async with self.engine.begin() as conn:
                def create_tables(connection):
                    Base.metadata.create_all(connection)

                await conn.run_sync(create_tables)
                logger.info("All autonomous tables created successfully")

            # Record successful migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.record_migration(migration_name, True, execution_time)

            return True

        except Exception as e:
            # Record failed migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.record_migration(migration_name, False, execution_time, str(e))

            logger.error("Failed to create autonomous agent tables", error=str(e))
            raise MigrationError(f"Autonomous tables migration failed: {e}")

    async def _create_performance_indices(self, session: AsyncSession) -> None:
        """Create performance-optimized indices."""
        indices = [
            # Autonomous agent states indices
            "CREATE INDEX IF NOT EXISTS idx_autonomous_agent_states_agent_id ON autonomous_agent_states(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_agent_states_session_id ON autonomous_agent_states(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_agent_states_last_accessed ON autonomous_agent_states(last_accessed)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_agent_states_status ON autonomous_agent_states(status)",

            # Goals indices
            "CREATE INDEX IF NOT EXISTS idx_autonomous_goals_agent_state_id ON autonomous_goals(agent_state_id)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_goals_status ON autonomous_goals(status)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_goals_priority ON autonomous_goals(priority)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_goals_created_at ON autonomous_goals(created_at)",

            # Decisions indices
            "CREATE INDEX IF NOT EXISTS idx_autonomous_decisions_agent_state_id ON autonomous_decisions(agent_state_id)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_decisions_decision_type ON autonomous_decisions(decision_type)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_decisions_confidence ON autonomous_decisions(confidence)",
            "CREATE INDEX IF NOT EXISTS idx_autonomous_decisions_created_at ON autonomous_decisions(created_at)",

            # Memories indices
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_state_id ON agent_memories(agent_state_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_memory_type ON agent_memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_importance ON agent_memories(importance)",
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_session_id ON agent_memories(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_expires_at ON agent_memories(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_created_at ON agent_memories(created_at)",

            # Learning experiences indices
            "CREATE INDEX IF NOT EXISTS idx_learning_experiences_agent_state_id ON learning_experiences(agent_state_id)",
            "CREATE INDEX IF NOT EXISTS idx_learning_experiences_experience_type ON learning_experiences(experience_type)",
            "CREATE INDEX IF NOT EXISTS idx_learning_experiences_success ON learning_experiences(success)",
            "CREATE INDEX IF NOT EXISTS idx_learning_experiences_created_at ON learning_experiences(created_at)",

            # Performance metrics indices
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_agent_state_id ON performance_metrics(agent_state_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_metric_name ON performance_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at)",
        ]

        for index_sql in indices:
            try:
                await session.execute(text(index_sql))
                logger.debug("Created index", sql=index_sql.split("idx_")[1].split(" ")[0])
            except Exception as e:
                logger.warning("Failed to create index", sql=index_sql, error=str(e))

    async def _create_fulltext_indices(self, session: AsyncSession) -> None:
        """Create full-text search indices for content fields."""
        fulltext_indices = [
            """CREATE INDEX IF NOT EXISTS idx_agent_memories_content_gin
               ON agent_memories USING gin(to_tsvector('english', content))""",
            """CREATE INDEX IF NOT EXISTS idx_autonomous_goals_description_gin
               ON autonomous_goals USING gin(to_tsvector('english', description))""",
            """CREATE INDEX IF NOT EXISTS idx_autonomous_decisions_reasoning_gin
               ON autonomous_decisions USING gin(to_tsvector('english', reasoning::text))""",
        ]

        for index_sql in fulltext_indices:
            try:
                await session.execute(text(index_sql))
                logger.debug("Created full-text index")
            except Exception as e:
                logger.warning("Failed to create full-text index", error=str(e))

    async def _create_additional_constraints(self, session: AsyncSession) -> None:
        """Create additional database constraints for data integrity."""
        constraints = [
            # Ensure agent_id is not empty
            """ALTER TABLE autonomous_agent_states
               ADD CONSTRAINT IF NOT EXISTS chk_agent_id_not_empty
               CHECK (length(trim(agent_id)) > 0)""",

            # Ensure priority is within valid range
            """ALTER TABLE autonomous_goals
               ADD CONSTRAINT IF NOT EXISTS chk_priority_range
               CHECK (priority >= 0.0 AND priority <= 1.0)""",

            # Ensure confidence is within valid range
            """ALTER TABLE autonomous_decisions
               ADD CONSTRAINT IF NOT EXISTS chk_confidence_range
               CHECK (confidence >= 0.0 AND confidence <= 1.0)""",

            # Ensure importance is within valid range
            """ALTER TABLE agent_memories
               ADD CONSTRAINT IF NOT EXISTS chk_importance_range
               CHECK (importance >= 0.0 AND importance <= 1.0)""",
        ]

        for constraint_sql in constraints:
            try:
                await session.execute(text(constraint_sql))
                logger.debug("Created constraint")
            except Exception as e:
                logger.warning("Failed to create constraint", error=str(e))


    async def drop_autonomous_tables(self) -> bool:
        """Drop all autonomous agent tables (for testing/cleanup)."""
        migration_name = "drop_autonomous_tables_v1"
        start_time = datetime.now()

        try:
            logger.info("Starting autonomous tables cleanup", migration=migration_name)

            async with self.session_factory() as session:
                async with session.begin():
                    # Drop tables in reverse dependency order
                    tables_to_drop = [
                        "performance_metrics",
                        "learning_experiences",
                        "agent_memories",
                        "autonomous_decisions",
                        "autonomous_goals",
                        "autonomous_agent_states"
                    ]

                    for table_name in tables_to_drop:
                        await session.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                        logger.debug("Dropped table", table=table_name)

            # Record successful migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.record_migration(migration_name, True, execution_time)

            logger.info("Autonomous agent tables dropped successfully")
            return True

        except Exception as e:
            # Record failed migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.record_migration(migration_name, False, execution_time, str(e))

            logger.error("Failed to drop autonomous agent tables", error=str(e))
            return False

    async def verify_autonomous_tables(self) -> bool:
        """Verify that all autonomous agent tables exist and are accessible."""
        try:
            logger.info("Verifying autonomous agent tables")

            async with self.session_factory() as session:
                # Check each table exists and is accessible
                tables_to_check = [
                    "autonomous_agent_states",
                    "autonomous_goals",
                    "autonomous_decisions",
                    "agent_memories",
                    "learning_experiences",
                    "performance_metrics"
                ]

                verification_results = {}

                for table_name in tables_to_check:
                    try:
                        # Check table exists
                        result = await session.execute(text("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables
                                WHERE table_name = :table_name
                            )
                        """), {"table_name": table_name})

                        exists = result.scalar()
                        if not exists:
                            logger.error("Table does not exist", table=table_name)
                            return False

                        # Check table is accessible and get record count
                        count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = count_result.scalar()

                        verification_results[table_name] = {
                            "exists": True,
                            "accessible": True,
                            "record_count": count
                        }

                        logger.info("Table verified", table=table_name, records=count)

                    except Exception as e:
                        logger.error("Table verification failed", table=table_name, error=str(e))
                        verification_results[table_name] = {
                            "exists": False,
                            "accessible": False,
                            "error": str(e)
                        }
                        return False

            logger.info("All autonomous agent tables verified successfully", results=verification_results)
            return True

        except Exception as e:
            logger.error("Failed to verify autonomous agent tables", error=str(e))
            return False

    async def get_migration_status(self) -> Dict[str, Any]:
        """Get status of all migrations."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("""
                    SELECT migration_name, applied_at, success, execution_time_ms, error_message
                    FROM migration_history
                    ORDER BY applied_at DESC
                """))

                migrations = []
                for row in result:
                    migrations.append({
                        "name": row[0],
                        "applied_at": row[1],
                        "success": row[2],
                        "execution_time_ms": row[3],
                        "error_message": row[4]
                    })

                return {
                    "total_migrations": len(migrations),
                    "successful_migrations": len([m for m in migrations if m["success"]]),
                    "failed_migrations": len([m for m in migrations if not m["success"]]),
                    "migrations": migrations
                }

        except Exception as e:
            logger.error("Failed to get migration status", error=str(e))
            return {"error": str(e)}


# Global migration manager instance
migration_manager = DatabaseMigrationManager()


async def initialize_autonomous_database() -> bool:
    """Initialize the autonomous agent database with all required tables."""
    try:
        logger.info("Initializing autonomous agent database...")

        # Initialize migration manager
        await migration_manager.initialize()

        # Check database connection
        if not await migration_manager.check_database_connection():
            raise MigrationError("Database connection failed")

        # Create migration tracking table
        await migration_manager.create_migration_tracking_table()

        # Create autonomous tables
        success = await migration_manager.create_autonomous_tables()
        if not success:
            raise MigrationError("Failed to create autonomous tables")

        # Verify tables
        if not await migration_manager.verify_autonomous_tables():
            raise MigrationError("Table verification failed")

        logger.info("Autonomous agent database initialized successfully")
        return True

    except Exception as e:
        logger.error("Failed to initialize autonomous agent database", error=str(e))
        return False


async def drop_autonomous_database() -> bool:
    """Drop all autonomous agent tables."""
    try:
        await migration_manager.initialize()
        return await migration_manager.drop_autonomous_tables()
    except Exception as e:
        logger.error("Failed to drop autonomous database", error=str(e))
        return False


async def verify_autonomous_database() -> bool:
    """Verify autonomous agent database."""
    try:
        await migration_manager.initialize()
        return await migration_manager.verify_autonomous_tables()
    except Exception as e:
        logger.error("Failed to verify autonomous database", error=str(e))
        return False


async def get_database_migration_status() -> Dict[str, Any]:
    """Get database migration status."""
    try:
        await migration_manager.initialize()
        return await migration_manager.get_migration_status()
    except Exception as e:
        logger.error("Failed to get migration status", error=str(e))
        return {"error": str(e)}


def main():
    """Main entry point for running migrations."""
    async def run_migration():
        print("🚀 Starting autonomous agent database migration...")
        print("=" * 60)

        try:
            success = await initialize_autonomous_database()

            if success:
                print("✅ Autonomous agent database initialized successfully!")

                # Show migration status
                status = await get_database_migration_status()
                if "error" not in status:
                    print(f"📊 Migration Status:")
                    print(f"   Total migrations: {status['total_migrations']}")
                    print(f"   Successful: {status['successful_migrations']}")
                    print(f"   Failed: {status['failed_migrations']}")

                return 0
            else:
                print("❌ Failed to initialize autonomous agent database")
                return 1

        except Exception as e:
            print(f"❌ Migration failed with error: {e}")
            return 1

    return asyncio.run(run_migration())


if __name__ == "__main__":
    sys.exit(main())
