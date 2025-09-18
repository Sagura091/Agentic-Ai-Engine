"""
Revolutionary Contextual Retrieval System for RAG 4.0.

This module provides advanced contextual retrieval capabilities including:
- Conversation-aware search and retrieval
- Personalized search based on user preferences and history
- Temporal retrieval with time-aware ranking
- Context understanding and semantic continuity
- Adaptive learning from user interactions
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re

import structlog
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from ..core.embeddings import EmbeddingManager
from ..core.caching import get_rag_cache, CacheType
from ..core.resilience_manager import get_resilience_manager

logger = structlog.get_logger(__name__)


class ContextType(Enum):
    """Types of context for retrieval."""
    CONVERSATION = "conversation"
    USER_PROFILE = "user_profile"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    COLLABORATIVE = "collaborative"
    DOMAIN = "domain"


class RetrievalMode(Enum):
    """Modes for contextual retrieval."""
    STANDARD = "standard"
    CONVERSATIONAL = "conversational"
    PERSONALIZED = "personalized"
    TEMPORAL = "temporal"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class ConversationContext:
    """Context from ongoing conversation."""
    conversation_id: str
    user_id: str
    messages: List[Dict[str, Any]]
    current_topic: Optional[str] = None
    intent_history: List[str] = None
    entity_mentions: List[str] = None
    semantic_thread: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.intent_history is None:
            self.intent_history = []
        if self.entity_mentions is None:
            self.entity_mentions = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class UserProfile:
    """User profile for personalized retrieval."""
    user_id: str
    preferences: Dict[str, Any]
    search_history: List[Dict[str, Any]]
    interaction_patterns: Dict[str, float]
    domain_expertise: Dict[str, float]
    learning_style: Optional[str] = None
    content_preferences: Dict[str, float] = None
    temporal_patterns: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.content_preferences is None:
            self.content_preferences = {}
        if self.temporal_patterns is None:
            self.temporal_patterns = {}


@dataclass
class ContextualQuery:
    """Query with contextual information."""
    query_text: str
    user_id: str
    conversation_context: Optional[ConversationContext] = None
    user_profile: Optional[UserProfile] = None
    temporal_context: Optional[Dict[str, Any]] = None
    domain_context: Optional[str] = None
    retrieval_mode: RetrievalMode = RetrievalMode.STANDARD
    context_weight: float = 0.5
    
    def __post_init__(self):
        if not hasattr(self, 'query_id'):
            self.query_id = str(uuid.uuid4())


@dataclass
class ContextualResult:
    """Result with contextual relevance scoring."""
    document_id: str
    content: str
    base_similarity: float
    contextual_relevance: float
    temporal_relevance: float
    personalization_score: float
    final_score: float
    context_explanation: Dict[str, Any]
    metadata: Dict[str, Any]


class ConversationTracker:
    """Tracks and analyzes conversation context."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_embeddings: Dict[str, List[float]] = {}
        
    async def update_conversation(
        self, 
        conversation_id: str, 
        user_id: str, 
        message: Dict[str, Any]
    ) -> ConversationContext:
        """Update conversation context with new message."""
        try:
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = ConversationContext(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    messages=[]
                )
            
            context = self.active_conversations[conversation_id]
            context.messages.append(message)
            
            # Analyze message for entities and intent
            entities = await self._extract_entities(message.get('content', ''))
            intent = await self._extract_intent(message.get('content', ''))
            
            context.entity_mentions.extend(entities)
            context.intent_history.append(intent)
            
            # Update current topic
            context.current_topic = await self._identify_topic(context.messages[-5:])
            
            # Generate semantic thread
            context.semantic_thread = await self._generate_semantic_thread(context)
            
            # Update conversation embedding
            await self._update_conversation_embedding(conversation_id, context)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to update conversation context: {str(e)}")
            raise
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text."""
        # Simple entity extraction - in production use advanced NER
        entities = []
        
        # Extract capitalized words (potential entities)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                if word not in ['The', 'This', 'That', 'When', 'Where', 'What', 'How']:
                    entities.append(word)
        
        return list(set(entities))
    
    async def _extract_intent(self, text: str) -> str:
        """Extract intent from text."""
        # Simple intent classification
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return 'question'
        elif any(word in text_lower for word in ['find', 'search', 'look', 'get']):
            return 'search'
        elif any(word in text_lower for word in ['explain', 'describe', 'tell']):
            return 'explanation'
        elif any(word in text_lower for word in ['help', 'assist', 'support']):
            return 'assistance'
        else:
            return 'general'
    
    async def _identify_topic(self, recent_messages: List[Dict[str, Any]]) -> str:
        """Identify current conversation topic."""
        if not recent_messages:
            return "general"
        
        # Combine recent messages
        combined_text = " ".join([msg.get('content', '') for msg in recent_messages])
        
        # Simple topic identification based on keywords
        topics = {
            'technology': ['ai', 'machine learning', 'software', 'programming', 'computer'],
            'science': ['research', 'study', 'experiment', 'theory', 'analysis'],
            'business': ['company', 'market', 'strategy', 'revenue', 'customer'],
            'education': ['learn', 'teach', 'course', 'student', 'knowledge'],
            'health': ['medical', 'health', 'treatment', 'patient', 'doctor']
        }
        
        text_lower = combined_text.lower()
        topic_scores = {}
        
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        
        return "general"
    
    async def _generate_semantic_thread(self, context: ConversationContext) -> str:
        """Generate semantic thread for conversation."""
        if not context.messages:
            return ""
        
        # Extract key concepts from recent messages
        recent_content = " ".join([
            msg.get('content', '') for msg in context.messages[-3:]
        ])
        
        # Simple semantic thread generation
        key_concepts = []
        if context.current_topic:
            key_concepts.append(context.current_topic)
        
        key_concepts.extend(context.entity_mentions[-5:])  # Recent entities
        
        return " -> ".join(key_concepts)
    
    async def _update_conversation_embedding(
        self, 
        conversation_id: str, 
        context: ConversationContext
    ) -> None:
        """Update conversation embedding."""
        try:
            # Combine recent conversation content
            recent_content = " ".join([
                msg.get('content', '') for msg in context.messages[-5:]
            ])
            
            if recent_content:
                embedding = await self.embedding_manager.embed_text(recent_content)
                self.conversation_embeddings[conversation_id] = embedding
                
        except Exception as e:
            logger.error(f"Failed to update conversation embedding: {str(e)}")


class UserProfileManager:
    """Manages user profiles for personalized retrieval."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences={},
                search_history=[],
                interaction_patterns={},
                domain_expertise={}
            )
        
        return self.user_profiles[user_id]
    
    async def update_profile_from_interaction(
        self, 
        user_id: str, 
        query: str, 
        results: List[ContextualResult],
        feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update user profile based on interaction."""
        try:
            profile = await self.get_or_create_profile(user_id)
            
            # Record search history
            interaction = {
                'timestamp': datetime.utcnow().isoformat(),
                'query': query,
                'results_count': len(results),
                'feedback': feedback
            }
            profile.search_history.append(interaction)
            
            # Update interaction patterns
            await self._update_interaction_patterns(profile, query, results, feedback)
            
            # Update domain expertise
            await self._update_domain_expertise(profile, query, results)
            
            # Update content preferences
            await self._update_content_preferences(profile, results, feedback)
            
            # Store interaction history
            self.interaction_history[user_id].append(interaction)
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {str(e)}")
    
    async def _update_interaction_patterns(
        self, 
        profile: UserProfile, 
        query: str, 
        results: List[ContextualResult],
        feedback: Optional[Dict[str, Any]]
    ) -> None:
        """Update user interaction patterns."""
        # Query length preference
        query_length = len(query.split())
        if 'query_length' not in profile.interaction_patterns:
            profile.interaction_patterns['query_length'] = query_length
        else:
            # Moving average
            profile.interaction_patterns['query_length'] = (
                profile.interaction_patterns['query_length'] * 0.8 + query_length * 0.2
            )
        
        # Result preference (if feedback provided)
        if feedback and 'rating' in feedback:
            rating = feedback['rating']
            if 'avg_rating' not in profile.interaction_patterns:
                profile.interaction_patterns['avg_rating'] = rating
            else:
                profile.interaction_patterns['avg_rating'] = (
                    profile.interaction_patterns['avg_rating'] * 0.9 + rating * 0.1
                )
    
    async def _update_domain_expertise(
        self, 
        profile: UserProfile, 
        query: str, 
        results: List[ContextualResult]
    ) -> None:
        """Update domain expertise based on queries and results."""
        # Simple domain classification
        domains = {
            'technology': ['ai', 'software', 'programming', 'computer', 'tech'],
            'science': ['research', 'study', 'experiment', 'scientific'],
            'business': ['business', 'market', 'strategy', 'company'],
            'education': ['education', 'learning', 'teaching', 'academic']
        }
        
        query_lower = query.lower()
        
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                if domain not in profile.domain_expertise:
                    profile.domain_expertise[domain] = 0.1
                else:
                    profile.domain_expertise[domain] = min(
                        1.0, profile.domain_expertise[domain] + 0.05
                    )
    
    async def _update_content_preferences(
        self, 
        profile: UserProfile, 
        results: List[ContextualResult],
        feedback: Optional[Dict[str, Any]]
    ) -> None:
        """Update content preferences based on results and feedback."""
        if not results:
            return
        
        # Analyze content types in results
        for result in results:
            content_type = result.metadata.get('content_type', 'text')
            
            if content_type not in profile.content_preferences:
                profile.content_preferences[content_type] = 0.5
            
            # Adjust based on feedback
            if feedback and 'rating' in feedback:
                rating = feedback['rating']
                adjustment = (rating - 3) * 0.1  # Scale -2 to +2 becomes -0.2 to +0.2
                profile.content_preferences[content_type] = max(
                    0.0, min(1.0, profile.content_preferences[content_type] + adjustment)
                )


class TemporalRetriever:
    """Handles temporal aspects of retrieval."""
    
    def __init__(self):
        self.temporal_weights = {
            'recency': 0.3,
            'relevance_decay': 0.2,
            'seasonal': 0.1,
            'trending': 0.4
        }
    
    async def calculate_temporal_relevance(
        self, 
        document_metadata: Dict[str, Any], 
        query_time: datetime
    ) -> float:
        """Calculate temporal relevance score."""
        try:
            doc_time = datetime.fromisoformat(
                document_metadata.get('created_at', query_time.isoformat())
            )
            
            # Recency score (exponential decay)
            time_diff = (query_time - doc_time).total_seconds()
            recency_score = np.exp(-time_diff / (30 * 24 * 3600))  # 30-day half-life
            
            # Relevance decay (content gets stale)
            content_type = document_metadata.get('content_type', 'general')
            decay_rates = {
                'news': 7 * 24 * 3600,      # 7 days
                'technical': 90 * 24 * 3600,  # 90 days
                'reference': 365 * 24 * 3600, # 1 year
                'general': 180 * 24 * 3600   # 180 days
            }
            
            decay_rate = decay_rates.get(content_type, decay_rates['general'])
            relevance_score = np.exp(-time_diff / decay_rate)
            
            # Trending score (simplified)
            trending_score = document_metadata.get('trending_score', 0.5)
            
            # Seasonal relevance (simplified)
            seasonal_score = 0.5  # Placeholder
            
            # Combine scores
            temporal_score = (
                self.temporal_weights['recency'] * recency_score +
                self.temporal_weights['relevance_decay'] * relevance_score +
                self.temporal_weights['trending'] * trending_score +
                self.temporal_weights['seasonal'] * seasonal_score
            )
            
            return min(1.0, temporal_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate temporal relevance: {str(e)}")
            return 0.5


class ContextualRetriever:
    """
    Revolutionary contextual retrieval system for RAG 4.0.
    
    Features:
    - Conversation-aware search and retrieval
    - Personalized search based on user behavior
    - Temporal retrieval with time-aware ranking
    - Context understanding and semantic continuity
    - Adaptive learning from user interactions
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        
        # Context managers
        self.conversation_tracker = ConversationTracker(embedding_manager)
        self.user_profile_manager = UserProfileManager(embedding_manager)
        self.temporal_retriever = TemporalRetriever()
        
        # Cache and resilience
        self.cache = None
        self.resilience_manager = None
        
        # Context weights
        self.context_weights = {
            'conversation': 0.25,
            'personalization': 0.25,
            'temporal': 0.20,
            'semantic': 0.30
        }
    
    async def initialize(self) -> None:
        """Initialize the contextual retriever."""
        try:
            self.cache = await get_rag_cache()
            self.resilience_manager = await get_resilience_manager()
            
            await self.resilience_manager.register_component(
                "contextual_retriever",
                recovery_strategies=["retry", "graceful_degradation"]
            )
            
            logger.info("Contextual retriever initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize contextual retriever: {str(e)}")
            raise
    
    async def contextual_search(
        self, 
        contextual_query: ContextualQuery,
        base_results: List[Dict[str, Any]]
    ) -> List[ContextualResult]:
        """Perform contextual search with awareness and personalization."""
        try:
            contextual_results = []
            
            for result in base_results:
                # Calculate base similarity (already provided)
                base_similarity = result.get('similarity', 0.0)
                
                # Calculate contextual relevance
                contextual_relevance = await self._calculate_contextual_relevance(
                    result, contextual_query
                )
                
                # Calculate temporal relevance
                temporal_relevance = await self.temporal_retriever.calculate_temporal_relevance(
                    result.get('metadata', {}), datetime.utcnow()
                )
                
                # Calculate personalization score
                personalization_score = await self._calculate_personalization_score(
                    result, contextual_query.user_profile
                )
                
                # Calculate final score
                final_score = self._calculate_final_score(
                    base_similarity,
                    contextual_relevance,
                    temporal_relevance,
                    personalization_score,
                    contextual_query.context_weight
                )
                
                # Create contextual result
                contextual_result = ContextualResult(
                    document_id=result.get('id', ''),
                    content=result.get('content', ''),
                    base_similarity=base_similarity,
                    contextual_relevance=contextual_relevance,
                    temporal_relevance=temporal_relevance,
                    personalization_score=personalization_score,
                    final_score=final_score,
                    context_explanation=await self._generate_context_explanation(
                        contextual_query, contextual_relevance, temporal_relevance, personalization_score
                    ),
                    metadata=result.get('metadata', {})
                )
                
                contextual_results.append(contextual_result)
            
            # Sort by final score
            contextual_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Update user profile with interaction
            if contextual_query.user_profile:
                await self.user_profile_manager.update_profile_from_interaction(
                    contextual_query.user_id,
                    contextual_query.query_text,
                    contextual_results
                )
            
            return contextual_results
            
        except Exception as e:
            await self.resilience_manager.record_error(
                "contextual_retriever",
                e,
                context={"query": contextual_query.query_text}
            )
            
            logger.error(f"Contextual search failed: {str(e)}")
            raise
    
    async def _calculate_contextual_relevance(
        self, 
        result: Dict[str, Any], 
        contextual_query: ContextualQuery
    ) -> float:
        """Calculate contextual relevance based on conversation and semantic context."""
        relevance_score = 0.0
        
        # Conversation context relevance
        if contextual_query.conversation_context:
            conv_relevance = await self._calculate_conversation_relevance(
                result, contextual_query.conversation_context
            )
            relevance_score += self.context_weights['conversation'] * conv_relevance
        
        # Semantic context relevance
        semantic_relevance = await self._calculate_semantic_relevance(
            result, contextual_query
        )
        relevance_score += self.context_weights['semantic'] * semantic_relevance
        
        return min(1.0, relevance_score)
    
    async def _calculate_conversation_relevance(
        self, 
        result: Dict[str, Any], 
        conversation_context: ConversationContext
    ) -> float:
        """Calculate relevance based on conversation context."""
        relevance = 0.0
        
        result_content = result.get('content', '').lower()
        
        # Topic relevance
        if conversation_context.current_topic:
            if conversation_context.current_topic.lower() in result_content:
                relevance += 0.3
        
        # Entity mention relevance
        entity_matches = sum(
            1 for entity in conversation_context.entity_mentions
            if entity.lower() in result_content
        )
        if conversation_context.entity_mentions:
            relevance += 0.4 * (entity_matches / len(conversation_context.entity_mentions))
        
        # Intent alignment
        if conversation_context.intent_history:
            recent_intent = conversation_context.intent_history[-1]
            intent_alignment = await self._calculate_intent_alignment(
                result, recent_intent
            )
            relevance += 0.3 * intent_alignment
        
        return min(1.0, relevance)
    
    async def _calculate_semantic_relevance(
        self, 
        result: Dict[str, Any], 
        contextual_query: ContextualQuery
    ) -> float:
        """Calculate semantic relevance."""
        # Placeholder for advanced semantic analysis
        # In production, use advanced NLP models for semantic understanding
        return 0.5
    
    async def _calculate_intent_alignment(
        self, 
        result: Dict[str, Any], 
        intent: str
    ) -> float:
        """Calculate how well result aligns with user intent."""
        result_content = result.get('content', '').lower()
        
        intent_indicators = {
            'question': ['answer', 'explanation', 'because', 'reason'],
            'search': ['find', 'locate', 'discover', 'identify'],
            'explanation': ['explain', 'describe', 'detail', 'overview'],
            'assistance': ['help', 'guide', 'support', 'assist']
        }
        
        indicators = intent_indicators.get(intent, [])
        matches = sum(1 for indicator in indicators if indicator in result_content)
        
        return min(1.0, matches / max(1, len(indicators)))
    
    async def _calculate_personalization_score(
        self, 
        result: Dict[str, Any], 
        user_profile: Optional[UserProfile]
    ) -> float:
        """Calculate personalization score based on user profile."""
        if not user_profile:
            return 0.5
        
        score = 0.0
        
        # Content type preference
        content_type = result.get('metadata', {}).get('content_type', 'text')
        if content_type in user_profile.content_preferences:
            score += 0.4 * user_profile.content_preferences[content_type]
        else:
            score += 0.2  # Default for unknown content types
        
        # Domain expertise alignment
        result_content = result.get('content', '').lower()
        for domain, expertise in user_profile.domain_expertise.items():
            if domain in result_content:
                score += 0.3 * expertise
                break
        
        # Historical interaction patterns
        if user_profile.interaction_patterns:
            avg_rating = user_profile.interaction_patterns.get('avg_rating', 3.0)
            score += 0.3 * (avg_rating / 5.0)  # Normalize to 0-1
        
        return min(1.0, score)
    
    def _calculate_final_score(
        self,
        base_similarity: float,
        contextual_relevance: float,
        temporal_relevance: float,
        personalization_score: float,
        context_weight: float
    ) -> float:
        """Calculate final contextual score."""
        # Weighted combination of all scores
        contextual_component = (
            contextual_relevance * 0.4 +
            temporal_relevance * 0.3 +
            personalization_score * 0.3
        )
        
        # Blend base similarity with contextual component
        final_score = (
            base_similarity * (1 - context_weight) +
            contextual_component * context_weight
        )
        
        return min(1.0, final_score)
    
    async def _generate_context_explanation(
        self,
        contextual_query: ContextualQuery,
        contextual_relevance: float,
        temporal_relevance: float,
        personalization_score: float
    ) -> Dict[str, Any]:
        """Generate explanation for contextual scoring."""
        explanation = {
            'contextual_factors': [],
            'temporal_factors': [],
            'personalization_factors': [],
            'overall_reasoning': ''
        }
        
        # Contextual factors
        if contextual_query.conversation_context:
            if contextual_query.conversation_context.current_topic:
                explanation['contextual_factors'].append(
                    f"Related to current topic: {contextual_query.conversation_context.current_topic}"
                )
            
            if contextual_query.conversation_context.entity_mentions:
                explanation['contextual_factors'].append(
                    f"Mentions entities from conversation: {', '.join(contextual_query.conversation_context.entity_mentions[-3:])}"
                )
        
        # Temporal factors
        if temporal_relevance > 0.7:
            explanation['temporal_factors'].append("Recent and relevant content")
        elif temporal_relevance > 0.4:
            explanation['temporal_factors'].append("Moderately recent content")
        else:
            explanation['temporal_factors'].append("Older content, may be less current")
        
        # Personalization factors
        if personalization_score > 0.7:
            explanation['personalization_factors'].append("Highly aligned with your preferences")
        elif personalization_score > 0.4:
            explanation['personalization_factors'].append("Somewhat aligned with your preferences")
        
        # Overall reasoning
        explanation['overall_reasoning'] = (
            f"This result was ranked considering conversation context ({contextual_relevance:.2f}), "
            f"temporal relevance ({temporal_relevance:.2f}), and your personal preferences ({personalization_score:.2f})"
        )
        
        return explanation


# Global contextual retriever instance
contextual_retriever = None


async def get_contextual_retriever(embedding_manager: EmbeddingManager) -> ContextualRetriever:
    """Get the global contextual retriever instance."""
    global contextual_retriever
    
    if contextual_retriever is None:
        contextual_retriever = ContextualRetriever(embedding_manager)
        await contextual_retriever.initialize()
    
    return contextual_retriever
