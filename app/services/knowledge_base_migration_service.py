"""
Knowledge Base Migration Service.

This service handles migrating knowledge base metadata from JSON files
to the PostgreSQL database for better scalability and querying.
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from sqlalchemy.exc import SQLAlchemyError

from app.models.database.base import get_session_factory
from app.models.knowledge_base import KnowledgeBase
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class KnowledgeBaseMigrationService:
    """Service for migrating knowledge bases from JSON to database."""
    
    def __init__(self):
        """Initialize the migration service."""
        self.session_factory = get_session_factory()
        self.settings = get_settings()
        self.json_file_path = Path("data/knowledge_bases.json")
        
    async def migrate_from_json(self) -> Dict[str, Any]:
        """
        Migrate knowledge bases from JSON file to database.
        
        Returns:
            Dict containing migration results
        """
        try:
            logger.info("Starting knowledge base migration from JSON to database")
            
            # Check if JSON file exists
            if not self.json_file_path.exists():
                logger.warning("Knowledge bases JSON file not found", path=str(self.json_file_path))
                return {
                    "success": True,
                    "message": "No JSON file to migrate",
                    "migrated_count": 0,
                    "skipped_count": 0,
                    "errors": []
                }
            
            # Load JSON data
            json_data = await self._load_json_data()
            if not json_data or "knowledge_bases" not in json_data:
                logger.warning("No knowledge bases found in JSON file")
                return {
                    "success": True,
                    "message": "No knowledge bases to migrate",
                    "migrated_count": 0,
                    "skipped_count": 0,
                    "errors": []
                }
            
            knowledge_bases = json_data["knowledge_bases"]
            logger.info("Found knowledge bases to migrate", count=len(knowledge_bases))
            
            # Migrate each knowledge base
            results = await self._migrate_knowledge_bases(knowledge_bases)
            
            # Create backup of JSON file
            if results["migrated_count"] > 0:
                await self._backup_json_file()
            
            logger.info(
                "Knowledge base migration completed",
                migrated=results["migrated_count"],
                skipped=results["skipped_count"],
                errors=len(results["errors"])
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Knowledge base migration failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "migrated_count": 0,
                "skipped_count": 0,
                "errors": [error_msg]
            }
    
    async def _load_json_data(self) -> Optional[Dict[str, Any]]:
        """Load knowledge bases from JSON file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load JSON file", error=str(e))
            return None
    
    async def _migrate_knowledge_bases(self, knowledge_bases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Migrate knowledge bases to database."""
        migrated_count = 0
        skipped_count = 0
        errors = []
        
        async with self.session_factory() as session:
            for kb_data in knowledge_bases:
                try:
                    # Check if knowledge base already exists
                    existing_kb = await session.execute(
                        select(KnowledgeBase).where(KnowledgeBase.id == kb_data["id"])
                    )
                    
                    if existing_kb.scalar_one_or_none():
                        logger.debug("Knowledge base already exists, skipping", kb_id=kb_data["id"])
                        skipped_count += 1
                        continue
                    
                    # Create new knowledge base record
                    kb_record = KnowledgeBase(
                        id=kb_data["id"],
                        name=kb_data["name"],
                        description=kb_data.get("description", ""),
                        use_case=kb_data.get("use_case", "general"),
                        tags=kb_data.get("tags", []),
                        is_public=kb_data.get("is_public", False),
                        created_by=kb_data.get("created_by", "system"),
                        document_count=kb_data.get("document_count", 0),
                        size_mb=kb_data.get("size_mb", 0.0),
                        created_at=self._parse_datetime(kb_data.get("created_at")),
                        status="active",
                        kb_metadata={
                            "migrated_from_json": True,
                            "migration_date": datetime.utcnow().isoformat(),
                            "original_data": kb_data
                        }
                    )
                    
                    session.add(kb_record)
                    await session.commit()
                    
                    logger.debug("Migrated knowledge base", kb_id=kb_data["id"], name=kb_data["name"])
                    migrated_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate knowledge base {kb_data.get('id', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    await session.rollback()
        
        return {
            "success": len(errors) == 0,
            "message": f"Migrated {migrated_count} knowledge bases, skipped {skipped_count}",
            "migrated_count": migrated_count,
            "skipped_count": skipped_count,
            "errors": errors
        }
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from JSON."""
        if not date_str:
            return None
        
        try:
            # Try parsing ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception:
            try:
                # Try parsing without timezone
                return datetime.fromisoformat(date_str)
            except Exception:
                logger.warning("Failed to parse datetime", date_str=date_str)
                return None
    
    async def _backup_json_file(self) -> None:
        """Create backup of JSON file after successful migration."""
        try:
            backup_path = self.json_file_path.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # Copy file to backup location
            import shutil
            shutil.copy2(self.json_file_path, backup_path)
            
            logger.info("Created backup of JSON file", backup_path=str(backup_path))
            
        except Exception as e:
            logger.warning("Failed to create backup of JSON file", error=str(e))
    
    async def export_to_json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export knowledge bases from database back to JSON format.
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Dict containing export results
        """
        try:
            logger.info("Starting knowledge base export to JSON")
            
            if not output_path:
                output_path = f"data/knowledge_bases_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            async with self.session_factory() as session:
                # Get all knowledge bases
                result = await session.execute(select(KnowledgeBase))
                knowledge_bases = result.scalars().all()
                
                # Convert to JSON format
                json_data = {
                    "knowledge_bases": [
                        {
                            "id": kb.id,
                            "name": kb.name,
                            "description": kb.description,
                            "use_case": kb.use_case,
                            "tags": kb.tags,
                            "is_public": kb.is_public,
                            "created_by": kb.created_by,
                            "created_at": kb.created_at.isoformat() if kb.created_at else None,
                            "document_count": kb.document_count,
                            "size_mb": kb.size_mb,
                            "status": kb.status,
                            "embedding_model": kb.embedding_model,
                            "metadata": kb.kb_metadata
                        }
                        for kb in knowledge_bases
                    ],
                    "exported_at": datetime.utcnow().isoformat(),
                    "total_count": len(knowledge_bases)
                }
                
                # Write to file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                logger.info("Knowledge bases exported to JSON", 
                           path=output_path, count=len(knowledge_bases))
                
                return {
                    "success": True,
                    "message": f"Exported {len(knowledge_bases)} knowledge bases",
                    "output_path": output_path,
                    "count": len(knowledge_bases)
                }
                
        except Exception as e:
            error_msg = f"Knowledge base export failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "output_path": None,
                "count": 0
            }
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        try:
            # Check JSON file
            json_exists = self.json_file_path.exists()
            json_count = 0
            
            if json_exists:
                json_data = await self._load_json_data()
                if json_data and "knowledge_bases" in json_data:
                    json_count = len(json_data["knowledge_bases"])
            
            # Check database
            async with self.session_factory() as session:
                result = await session.execute(select(KnowledgeBase))
                db_count = len(result.scalars().all())
            
            return {
                "json_file_exists": json_exists,
                "json_count": json_count,
                "database_count": db_count,
                "migration_needed": json_exists and json_count > 0,
                "status": "completed" if db_count > 0 else "pending" if json_exists else "no_data"
            }
            
        except Exception as e:
            logger.error("Failed to get migration status", error=str(e))
            return {
                "json_file_exists": False,
                "json_count": 0,
                "database_count": 0,
                "migration_needed": False,
                "status": "error",
                "error": str(e)
            }


# Global service instance
knowledge_base_migration_service = KnowledgeBaseMigrationService()


async def migrate_knowledge_bases():
    """Convenience function to run knowledge base migration."""
    return await knowledge_base_migration_service.migrate_from_json()
