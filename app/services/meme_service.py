"""
Meme Service - Database operations and business logic for meme management.

This service provides high-level operations for storing, retrieving, and managing
memes, analysis results, and generation history in the database.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4

import structlog
from sqlalchemy import select, update, delete, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database.base import get_session_factory
from app.models.meme import (
    MemeDB, MemeAnalysisDB, GeneratedMemeDB, MemeTemplateDB, 
    MemeAgentStateDB, MemeTrendDB
)
from app.tools.meme_collection_tool import MemeData
from app.tools.meme_analysis_tool import MemeAnalysisResult
from app.tools.meme_generation_tool import GeneratedMeme

logger = structlog.get_logger(__name__)


class MemeService:
    """Service for meme database operations."""
    
    def __init__(self):
        self.session_factory = get_session_factory()
    
    async def store_collected_meme(self, meme_data: MemeData) -> str:
        """Store a collected meme in the database."""
        try:
            async with self.session_factory() as session:
                # Check if meme already exists
                existing = await session.execute(
                    select(MemeDB).where(MemeDB.meme_id == meme_data.id)
                )
                if existing.scalar_one_or_none():
                    logger.info(f"Meme {meme_data.id} already exists, skipping")
                    return meme_data.id
                
                # Create new meme record
                meme_db = MemeDB(
                    meme_id=meme_data.id,
                    title=meme_data.title,
                    url=meme_data.url,
                    image_url=meme_data.image_url,
                    local_path=meme_data.local_path,
                    source=meme_data.source,
                    subreddit=meme_data.subreddit,
                    author=meme_data.author,
                    score=meme_data.score,
                    comments_count=meme_data.comments_count,
                    quality_score=meme_data.quality_score,
                    text_content=meme_data.text_content,
                    template_type=meme_data.template_type,
                    width=meme_data.dimensions[0] if meme_data.dimensions else 0,
                    height=meme_data.dimensions[1] if meme_data.dimensions else 0,
                    file_size=meme_data.file_size,
                    content_hash=meme_data.content_hash,
                    created_utc=meme_data.created_utc,
                    processed=meme_data.processed,
                    metadata=meme_data.metadata
                )
                
                session.add(meme_db)
                await session.commit()
                
                logger.info(f"Stored meme {meme_data.id} in database")
                return meme_data.id
                
        except Exception as e:
            logger.error(f"Failed to store meme {meme_data.id}: {str(e)}")
            raise
    
    async def store_meme_analysis(self, analysis_result: MemeAnalysisResult) -> str:
        """Store meme analysis results in the database."""
        try:
            async with self.session_factory() as session:
                # Get the meme record
                meme_result = await session.execute(
                    select(MemeDB).where(MemeDB.meme_id == analysis_result.meme_id)
                )
                meme_db = meme_result.scalar_one_or_none()
                
                if not meme_db:
                    logger.error(f"Meme {analysis_result.meme_id} not found for analysis storage")
                    return ""
                
                # Create analysis record
                analysis_id = f"analysis_{analysis_result.meme_id}_{int(datetime.now().timestamp())}"
                
                analysis_db = MemeAnalysisDB(
                    analysis_id=analysis_id,
                    meme_id=meme_db.id,
                    extracted_text=analysis_result.text_content,
                    text_regions=analysis_result.text_regions,
                    readability_score=analysis_result.readability_score,
                    template_matches=analysis_result.template_matches,
                    best_template_match=analysis_result.template_matches[0][0] if analysis_result.template_matches else None,
                    template_confidence=analysis_result.template_matches[0][1] if analysis_result.template_matches else 0.0,
                    visual_features=analysis_result.visual_features,
                    dominant_colors=analysis_result.color_palette,
                    complexity_score=analysis_result.complexity_score,
                    sentiment_score=analysis_result.sentiment_score,
                    humor_score=analysis_result.humor_score,
                    content_category=analysis_result.content_category,
                    detected_objects=analysis_result.detected_objects,
                    overall_quality_score=analysis_result.quality_score,
                    metadata=analysis_result.metadata
                )
                
                session.add(analysis_db)
                await session.commit()
                
                logger.info(f"Stored analysis {analysis_id} for meme {analysis_result.meme_id}")
                return analysis_id
                
        except Exception as e:
            logger.error(f"Failed to store analysis for meme {analysis_result.meme_id}: {str(e)}")
            raise
    
    async def store_generated_meme(self, generated_meme: GeneratedMeme, agent_id: str = None) -> str:
        """Store a generated meme in the database."""
        try:
            async with self.session_factory() as session:
                # Check if meme already exists
                existing = await session.execute(
                    select(GeneratedMemeDB).where(GeneratedMemeDB.meme_id == generated_meme.meme_id)
                )
                if existing.scalar_one_or_none():
                    logger.info(f"Generated meme {generated_meme.meme_id} already exists, skipping")
                    return generated_meme.meme_id
                
                # Create generated meme record
                generated_db = GeneratedMemeDB(
                    meme_id=generated_meme.meme_id,
                    prompt=generated_meme.prompt,
                    generation_method=generated_meme.generation_method,
                    template_used=generated_meme.template_used,
                    image_path=generated_meme.image_path,
                    text_elements=generated_meme.text_elements,
                    quality_score=generated_meme.quality_score,
                    humor_score=generated_meme.humor_score,
                    creativity_score=generated_meme.creativity_score,
                    generation_time=generated_meme.generation_time,
                    agent_id=agent_id,
                    metadata=generated_meme.metadata
                )
                
                session.add(generated_db)
                await session.commit()
                
                logger.info(f"Stored generated meme {generated_meme.meme_id} in database")
                return generated_meme.meme_id
                
        except Exception as e:
            logger.error(f"Failed to store generated meme {generated_meme.meme_id}: {str(e)}")
            raise
    
    async def get_memes_for_analysis(self, limit: int = 50, unanalyzed_only: bool = True) -> List[MemeDB]:
        """Get memes that need analysis."""
        try:
            async with self.session_factory() as session:
                query = select(MemeDB)
                
                if unanalyzed_only:
                    # Get memes without analysis results
                    query = query.outerjoin(MemeAnalysisDB).where(MemeAnalysisDB.id.is_(None))
                
                query = query.order_by(desc(MemeDB.collected_at)).limit(limit)
                
                result = await session.execute(query)
                memes = result.scalars().all()
                
                return list(memes)
                
        except Exception as e:
            logger.error(f"Failed to get memes for analysis: {str(e)}")
            return []
    
    async def get_popular_templates(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular meme templates based on usage."""
        try:
            async with self.session_factory() as session:
                # Get template usage from analysis results
                query = select(
                    MemeAnalysisDB.best_template_match,
                    func.count(MemeAnalysisDB.best_template_match).label('usage_count')
                ).where(
                    MemeAnalysisDB.best_template_match.is_not(None)
                ).group_by(
                    MemeAnalysisDB.best_template_match
                ).order_by(
                    desc('usage_count')
                ).limit(limit)
                
                result = await session.execute(query)
                templates = result.all()
                
                return [(template[0], template[1]) for template in templates]
                
        except Exception as e:
            logger.error(f"Failed to get popular templates: {str(e)}")
            return []
    
    async def get_trending_topics(self, hours: int = 24, limit: int = 10) -> List[str]:
        """Get trending topics based on recent meme content."""
        try:
            async with self.session_factory() as session:
                # Get recent memes
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                query = select(MemeDB.text_content, MemeDB.title).where(
                    MemeDB.collected_at >= cutoff_time
                ).order_by(desc(MemeDB.score))
                
                result = await session.execute(query)
                memes = result.all()
                
                # Extract topics from text content and titles
                topics = set()
                for meme in memes:
                    # Add title words
                    title_words = meme.title.lower().split()
                    topics.update(word for word in title_words if len(word) > 3)
                    
                    # Add text content
                    if meme.text_content:
                        for text in meme.text_content:
                            text_words = text.lower().split()
                            topics.update(word for word in text_words if len(word) > 3)
                
                # Return most common topics (simplified - could use more sophisticated analysis)
                return list(topics)[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get trending topics: {str(e)}")
            return []
    
    async def update_agent_state(self, agent_id: str, state_data: Dict[str, Any]) -> bool:
        """Update meme agent state in database."""
        try:
            async with self.session_factory() as session:
                # Check if agent state exists
                existing = await session.execute(
                    select(MemeAgentStateDB).where(MemeAgentStateDB.agent_id == agent_id)
                )
                agent_state = existing.scalar_one_or_none()
                
                if agent_state:
                    # Update existing state
                    for key, value in state_data.items():
                        if hasattr(agent_state, key):
                            setattr(agent_state, key, value)
                    agent_state.updated_at = datetime.utcnow()
                else:
                    # Create new agent state
                    agent_state = MemeAgentStateDB(
                        agent_id=agent_id,
                        **state_data
                    )
                    session.add(agent_state)
                
                await session.commit()
                logger.info(f"Updated agent state for {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update agent state for {agent_id}: {str(e)}")
            return False
    
    async def get_agent_state(self, agent_id: str) -> Optional[MemeAgentStateDB]:
        """Get meme agent state from database."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(MemeAgentStateDB).where(MemeAgentStateDB.agent_id == agent_id)
                )
                return result.scalar_one_or_none()
                
        except Exception as e:
            logger.error(f"Failed to get agent state for {agent_id}: {str(e)}")
            return None
    
    async def get_meme_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meme statistics."""
        try:
            async with self.session_factory() as session:
                # Total counts
                total_memes = await session.scalar(select(func.count(MemeDB.id)))
                total_analyzed = await session.scalar(select(func.count(MemeAnalysisDB.id)))
                total_generated = await session.scalar(select(func.count(GeneratedMemeDB.id)))
                
                # Quality statistics
                avg_quality = await session.scalar(select(func.avg(MemeDB.quality_score)))
                avg_analysis_quality = await session.scalar(select(func.avg(MemeAnalysisDB.overall_quality_score)))
                avg_generated_quality = await session.scalar(select(func.avg(GeneratedMemeDB.quality_score)))
                
                # Recent activity (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                recent_collected = await session.scalar(
                    select(func.count(MemeDB.id)).where(MemeDB.collected_at >= cutoff_time)
                )
                recent_generated = await session.scalar(
                    select(func.count(GeneratedMemeDB.id)).where(GeneratedMemeDB.created_at >= cutoff_time)
                )
                
                # Source breakdown
                source_query = select(
                    MemeDB.source,
                    func.count(MemeDB.source).label('count')
                ).group_by(MemeDB.source)
                source_result = await session.execute(source_query)
                source_breakdown = {row[0]: row[1] for row in source_result.all()}
                
                return {
                    'total_statistics': {
                        'total_memes_collected': total_memes or 0,
                        'total_memes_analyzed': total_analyzed or 0,
                        'total_memes_generated': total_generated or 0
                    },
                    'quality_metrics': {
                        'average_collection_quality': round(avg_quality or 0, 3),
                        'average_analysis_quality': round(avg_analysis_quality or 0, 3),
                        'average_generation_quality': round(avg_generated_quality or 0, 3)
                    },
                    'recent_activity': {
                        'memes_collected_24h': recent_collected or 0,
                        'memes_generated_24h': recent_generated or 0
                    },
                    'source_breakdown': source_breakdown
                }
                
        except Exception as e:
            logger.error(f"Failed to get meme statistics: {str(e)}")
            return {}
    
    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old meme data to manage storage."""
        try:
            async with self.session_factory() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                
                # Delete old low-quality memes
                old_memes_query = delete(MemeDB).where(
                    MemeDB.collected_at < cutoff_date,
                    MemeDB.quality_score < 0.3
                )
                old_memes_result = await session.execute(old_memes_query)
                
                # Delete old analysis results for deleted memes
                orphaned_analysis_query = delete(MemeAnalysisDB).where(
                    ~MemeAnalysisDB.meme_id.in_(select(MemeDB.id))
                )
                orphaned_analysis_result = await session.execute(orphaned_analysis_query)
                
                await session.commit()
                
                cleanup_stats = {
                    'deleted_memes': old_memes_result.rowcount,
                    'deleted_analysis': orphaned_analysis_result.rowcount
                }
                
                logger.info(f"Cleanup completed: {cleanup_stats}")
                return cleanup_stats
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return {'deleted_memes': 0, 'deleted_analysis': 0}


# Global service instance
meme_service = MemeService()
