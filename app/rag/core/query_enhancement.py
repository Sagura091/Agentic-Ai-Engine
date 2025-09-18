"""
Advanced Query Enhancement Pipeline for RAG 4.0.

Implements query expansion, reformulation, multi-hop reasoning,
and confidence-based filtering for superior retrieval performance.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import spacy
from collections import defaultdict, Counter

from .knowledge_base import KnowledgeQuery, Document
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


class QueryType(str, Enum):
    """Types of queries for different enhancement strategies."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


class EnhancementStrategy(str, Enum):
    """Query enhancement strategies."""
    SYNONYM_EXPANSION = "synonym_expansion"
    CONTEXT_EXPANSION = "context_expansion"
    ENTITY_EXPANSION = "entity_expansion"
    TEMPORAL_EXPANSION = "temporal_expansion"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    REFORMULATION = "reformulation"


@dataclass
class QueryContext:
    """Context information for query enhancement."""
    agent_id: str
    conversation_history: List[str] = field(default_factory=list)
    recent_documents: List[Document] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    domain_context: Optional[str] = None
    temporal_context: Optional[datetime] = None
    session_id: Optional[str] = None


@dataclass
class EnhancedQuery:
    """Enhanced query with expansion and metadata."""
    original_query: str
    enhanced_query: str
    expansion_terms: List[str]
    query_type: QueryType
    confidence: float
    enhancement_strategies: List[EnhancementStrategy]
    reasoning_chain: List[str]
    metadata: Dict[str, Any]


@dataclass
class QueryEnhancementConfig:
    """Configuration for query enhancement."""
    enable_synonym_expansion: bool = True
    enable_context_expansion: bool = True
    enable_entity_expansion: bool = True
    enable_multi_hop: bool = True
    max_expansion_terms: int = 10
    min_confidence_threshold: float = 0.3
    context_window_size: int = 5
    synonym_similarity_threshold: float = 0.7
    enable_spacy: bool = True
    spacy_model: str = "en_core_web_sm"


class QueryClassifier:
    """Classifies queries to determine appropriate enhancement strategies."""
    
    def __init__(self):
        self.factual_patterns = [
            r'\b(what|who|when|where|which|how many|how much)\b',
            r'\b(define|definition|meaning|explain)\b',
            r'\b(is|are|was|were|does|do|did)\b.*\?'
        ]
        
        self.procedural_patterns = [
            r'\b(how to|how do|how can|steps|process|procedure)\b',
            r'\b(guide|tutorial|instructions|method)\b'
        ]
        
        self.comparative_patterns = [
            r'\b(compare|comparison|versus|vs|difference|similar|different)\b',
            r'\b(better|worse|best|worst|more|less)\b.*\bthan\b'
        ]
        
        self.temporal_patterns = [
            r'\b(before|after|during|since|until|recent|latest|current)\b',
            r'\b(yesterday|today|tomorrow|last|next|ago)\b',
            r'\b(\d{4}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2})\b'
        ]
        
        self.causal_patterns = [
            r'\b(why|because|cause|reason|result|effect|impact)\b',
            r'\b(leads to|results in|due to|caused by)\b'
        ]
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query type based on patterns."""
        query_lower = query.lower()
        
        # Check patterns in order of specificity
        if any(re.search(pattern, query_lower) for pattern in self.causal_patterns):
            return QueryType.CAUSAL
        elif any(re.search(pattern, query_lower) for pattern in self.comparative_patterns):
            return QueryType.COMPARATIVE
        elif any(re.search(pattern, query_lower) for pattern in self.procedural_patterns):
            return QueryType.PROCEDURAL
        elif any(re.search(pattern, query_lower) for pattern in self.temporal_patterns):
            return QueryType.TEMPORAL
        elif any(re.search(pattern, query_lower) for pattern in self.factual_patterns):
            return QueryType.FACTUAL
        else:
            return QueryType.CONCEPTUAL


class SynonymExpander:
    """Expands queries with synonyms and related terms."""
    
    def __init__(self, config: QueryEnhancementConfig):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model if enabled
        self.nlp = None
        if config.enable_spacy:
            try:
                self.nlp = spacy.load(config.spacy_model)
            except OSError:
                logger.warning(f"SpaCy model {config.spacy_model} not found, using NLTK only")
    
    async def expand_query(self, query: str, context: QueryContext) -> List[str]:
        """Expand query with synonyms and related terms."""
        expansion_terms = []
        
        try:
            # Tokenize and get POS tags
            tokens = word_tokenize(query.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract meaningful words (nouns, verbs, adjectives)
            meaningful_words = [
                word for word, pos in pos_tags
                if pos.startswith(('NN', 'VB', 'JJ')) and 
                word not in self.stop_words and 
                len(word) > 2
            ]
            
            # Get synonyms using WordNet
            for word in meaningful_words[:self.config.max_expansion_terms]:
                synonyms = self._get_wordnet_synonyms(word)
                expansion_terms.extend(synonyms[:3])  # Limit synonyms per word
            
            # Use spaCy for semantic similarity if available
            if self.nlp:
                spacy_terms = await self._get_spacy_expansions(query, meaningful_words)
                expansion_terms.extend(spacy_terms)
            
            # Remove duplicates and filter
            expansion_terms = list(set(expansion_terms))
            expansion_terms = [term for term in expansion_terms if term not in query.lower()]
            
            logger.debug(
                "Synonym expansion completed",
                original_terms=meaningful_words,
                expansion_terms=expansion_terms[:10]  # Log first 10
            )
            
            return expansion_terms[:self.config.max_expansion_terms]
            
        except Exception as e:
            logger.error("Synonym expansion failed", error=str(e))
            return []
    
    def _get_wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet."""
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym) > 2:
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    async def _get_spacy_expansions(self, query: str, meaningful_words: List[str]) -> List[str]:
        """Get semantic expansions using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(query)
            expansions = []
            
            # Get similar words based on word vectors
            for token in doc:
                if token.text.lower() in meaningful_words and token.has_vector:
                    # Find similar words in vocabulary
                    similar_words = []
                    for word in self.nlp.vocab:
                        if (word.has_vector and 
                            word.similarity(token) > self.config.synonym_similarity_threshold and
                            word.text != token.text):
                            similar_words.append((word.text, word.similarity(token)))
                    
                    # Sort by similarity and take top matches
                    similar_words.sort(key=lambda x: x[1], reverse=True)
                    expansions.extend([word for word, _ in similar_words[:3]])
            
            return expansions
            
        except Exception as e:
            logger.error("SpaCy expansion failed", error=str(e))
            return []


class ContextExpander:
    """Expands queries using conversation and agent context."""
    
    def __init__(self, config: QueryEnhancementConfig):
        self.config = config
    
    async def expand_with_context(self, query: str, context: QueryContext) -> List[str]:
        """Expand query using conversation and agent context."""
        expansion_terms = []
        
        try:
            # Extract terms from conversation history
            if context.conversation_history:
                history_terms = self._extract_context_terms(
                    context.conversation_history[-self.config.context_window_size:]
                )
                expansion_terms.extend(history_terms)
            
            # Extract terms from recent documents
            if context.recent_documents:
                doc_terms = self._extract_document_terms(context.recent_documents)
                expansion_terms.extend(doc_terms)
            
            # Add domain-specific terms
            if context.domain_context:
                domain_terms = self._get_domain_terms(context.domain_context)
                expansion_terms.extend(domain_terms)
            
            # Filter and rank terms
            filtered_terms = self._filter_context_terms(query, expansion_terms)
            
            logger.debug(
                "Context expansion completed",
                agent_id=context.agent_id,
                history_items=len(context.conversation_history),
                recent_docs=len(context.recent_documents),
                expansion_terms=len(filtered_terms)
            )
            
            return filtered_terms[:self.config.max_expansion_terms]
            
        except Exception as e:
            logger.error("Context expansion failed", error=str(e))
            return []
    
    def _extract_context_terms(self, history: List[str]) -> List[str]:
        """Extract relevant terms from conversation history."""
        terms = []
        
        for message in history:
            # Simple extraction - can be enhanced with NER
            words = word_tokenize(message.lower())
            pos_tags = pos_tag(words)
            
            # Extract nouns and proper nouns
            context_terms = [
                word for word, pos in pos_tags
                if pos.startswith(('NN', 'NNP')) and len(word) > 2
            ]
            terms.extend(context_terms)
        
        # Return most frequent terms
        term_counts = Counter(terms)
        return [term for term, _ in term_counts.most_common(10)]
    
    def _extract_document_terms(self, documents: List[Document]) -> List[str]:
        """Extract key terms from recent documents."""
        terms = []
        
        for doc in documents:
            # Extract from title and content
            text = f"{doc.title} {doc.content}"
            words = word_tokenize(text.lower())
            pos_tags = pos_tag(words)
            
            # Extract important terms
            doc_terms = [
                word for word, pos in pos_tags
                if pos.startswith(('NN', 'NNP', 'JJ')) and len(word) > 2
            ]
            terms.extend(doc_terms)
        
        # Return most frequent terms
        term_counts = Counter(terms)
        return [term for term, _ in term_counts.most_common(5)]
    
    def _get_domain_terms(self, domain: str) -> List[str]:
        """Get domain-specific expansion terms."""
        domain_mappings = {
            "research": ["study", "analysis", "methodology", "findings", "data"],
            "creative": ["design", "concept", "inspiration", "artistic", "innovation"],
            "technical": ["implementation", "architecture", "system", "framework", "solution"],
            "business": ["strategy", "market", "revenue", "customer", "growth"],
            "medical": ["treatment", "diagnosis", "patient", "clinical", "therapy"]
        }
        
        return domain_mappings.get(domain.lower(), [])
    
    def _filter_context_terms(self, query: str, terms: List[str]) -> List[str]:
        """Filter and rank context terms by relevance."""
        query_words = set(word_tokenize(query.lower()))
        
        # Filter out terms already in query
        filtered = [term for term in terms if term not in query_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in filtered:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms


class MultiHopReasoner:
    """Implements multi-hop reasoning for complex queries."""
    
    def __init__(self, config: QueryEnhancementConfig):
        self.config = config
    
    async def generate_reasoning_chain(
        self, 
        query: str, 
        query_type: QueryType,
        context: QueryContext
    ) -> List[str]:
        """Generate reasoning chain for multi-hop queries."""
        reasoning_chain = []
        
        try:
            if query_type == QueryType.CAUSAL:
                reasoning_chain = self._generate_causal_chain(query)
            elif query_type == QueryType.COMPARATIVE:
                reasoning_chain = self._generate_comparative_chain(query)
            elif query_type == QueryType.PROCEDURAL:
                reasoning_chain = self._generate_procedural_chain(query)
            elif query_type == QueryType.TEMPORAL:
                reasoning_chain = self._generate_temporal_chain(query, context)
            else:
                reasoning_chain = self._generate_basic_chain(query)
            
            logger.debug(
                "Reasoning chain generated",
                query_type=query_type.value,
                chain_length=len(reasoning_chain)
            )
            
            return reasoning_chain
            
        except Exception as e:
            logger.error("Reasoning chain generation failed", error=str(e))
            return [query]  # Fallback to original query
    
    def _generate_causal_chain(self, query: str) -> List[str]:
        """Generate reasoning chain for causal queries."""
        # Extract the main concept
        main_concept = self._extract_main_concept(query)
        
        return [
            f"What is {main_concept}?",
            f"What factors influence {main_concept}?",
            f"What are the effects of {main_concept}?",
            query  # Original query
        ]
    
    def _generate_comparative_chain(self, query: str) -> List[str]:
        """Generate reasoning chain for comparative queries."""
        # Extract entities being compared
        entities = self._extract_comparison_entities(query)
        
        if len(entities) >= 2:
            return [
                f"What is {entities[0]}?",
                f"What is {entities[1]}?",
                f"What are the characteristics of {entities[0]}?",
                f"What are the characteristics of {entities[1]}?",
                query
            ]
        else:
            return [query]
    
    def _generate_procedural_chain(self, query: str) -> List[str]:
        """Generate reasoning chain for procedural queries."""
        main_concept = self._extract_main_concept(query)
        
        return [
            f"What is {main_concept}?",
            f"What are the prerequisites for {main_concept}?",
            f"What are the steps involved in {main_concept}?",
            f"What are common challenges with {main_concept}?",
            query
        ]
    
    def _generate_temporal_chain(self, query: str, context: QueryContext) -> List[str]:
        """Generate reasoning chain for temporal queries."""
        main_concept = self._extract_main_concept(query)
        
        chain = [
            f"What is the history of {main_concept}?",
            f"What is the current state of {main_concept}?",
            query
        ]
        
        # Add temporal context if available
        if context.temporal_context:
            chain.insert(-1, f"What happened with {main_concept} around {context.temporal_context.year}?")
        
        return chain
    
    def _generate_basic_chain(self, query: str) -> List[str]:
        """Generate basic reasoning chain."""
        main_concept = self._extract_main_concept(query)
        
        return [
            f"What is {main_concept}?",
            f"What are examples of {main_concept}?",
            query
        ]
    
    def _extract_main_concept(self, query: str) -> str:
        """Extract the main concept from a query."""
        # Simple extraction - can be enhanced with NER
        words = word_tokenize(query)
        pos_tags = pos_tag(words)
        
        # Find the most important noun
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        
        if nouns:
            return nouns[0]  # Return first noun as main concept
        else:
            return "the topic"
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities being compared in a query."""
        # Look for patterns like "A vs B", "A and B", "A or B"
        comparison_patterns = [
            r'(\w+)\s+(?:vs|versus)\s+(\w+)',
            r'(\w+)\s+and\s+(\w+)',
            r'(\w+)\s+or\s+(\w+)',
            r'between\s+(\w+)\s+and\s+(\w+)'
        ]
        
        for pattern in comparison_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1), match.group(2)]
        
        return []


class QueryEnhancementEngine:
    """Main query enhancement engine coordinating all enhancement strategies."""
    
    def __init__(self, config: Optional[QueryEnhancementConfig] = None):
        self.config = config or QueryEnhancementConfig()
        self.classifier = QueryClassifier()
        self.synonym_expander = SynonymExpander(self.config)
        self.context_expander = ContextExpander(self.config)
        self.multi_hop_reasoner = MultiHopReasoner(self.config)
        
        self.enhancement_stats = {
            "total_enhancements": 0,
            "by_query_type": defaultdict(int),
            "by_strategy": defaultdict(int),
            "avg_expansion_terms": 0.0
        }
    
    async def enhance_query(
        self, 
        query: str, 
        context: QueryContext,
        strategies: Optional[List[EnhancementStrategy]] = None
    ) -> EnhancedQuery:
        """Enhance query using specified or automatic strategies."""
        try:
            self.enhancement_stats["total_enhancements"] += 1
            
            # Classify query type
            query_type = self.classifier.classify_query(query)
            self.enhancement_stats["by_query_type"][query_type.value] += 1
            
            # Determine enhancement strategies
            if strategies is None:
                strategies = self._select_strategies(query_type)
            
            # Apply enhancements
            expansion_terms = []
            reasoning_chain = [query]
            enhanced_query = query
            
            for strategy in strategies:
                self.enhancement_stats["by_strategy"][strategy.value] += 1
                
                if strategy == EnhancementStrategy.SYNONYM_EXPANSION and self.config.enable_synonym_expansion:
                    synonyms = await self.synonym_expander.expand_query(query, context)
                    expansion_terms.extend(synonyms)
                
                elif strategy == EnhancementStrategy.CONTEXT_EXPANSION and self.config.enable_context_expansion:
                    context_terms = await self.context_expander.expand_with_context(query, context)
                    expansion_terms.extend(context_terms)
                
                elif strategy == EnhancementStrategy.MULTI_HOP_REASONING and self.config.enable_multi_hop:
                    reasoning_chain = await self.multi_hop_reasoner.generate_reasoning_chain(
                        query, query_type, context
                    )
            
            # Build enhanced query
            if expansion_terms:
                unique_terms = list(set(expansion_terms))[:self.config.max_expansion_terms]
                enhanced_query = f"{query} {' '.join(unique_terms)}"
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, expansion_terms, query_type)
            
            # Update stats
            self._update_enhancement_stats(len(expansion_terms))
            
            enhanced = EnhancedQuery(
                original_query=query,
                enhanced_query=enhanced_query,
                expansion_terms=expansion_terms,
                query_type=query_type,
                confidence=confidence,
                enhancement_strategies=strategies,
                reasoning_chain=reasoning_chain,
                metadata={
                    "agent_id": context.agent_id,
                    "enhancement_timestamp": datetime.now().isoformat(),
                    "context_items": len(context.conversation_history),
                    "domain": context.domain_context
                }
            )
            
            logger.info(
                "Query enhancement completed",
                original_length=len(query),
                enhanced_length=len(enhanced_query),
                expansion_terms=len(expansion_terms),
                query_type=query_type.value,
                confidence=confidence
            )
            
            return enhanced
            
        except Exception as e:
            logger.error("Query enhancement failed", error=str(e))
            # Return minimal enhancement on failure
            return EnhancedQuery(
                original_query=query,
                enhanced_query=query,
                expansion_terms=[],
                query_type=QueryType.CONCEPTUAL,
                confidence=0.5,
                enhancement_strategies=[],
                reasoning_chain=[query],
                metadata={"error": str(e)}
            )
    
    def _select_strategies(self, query_type: QueryType) -> List[EnhancementStrategy]:
        """Select appropriate enhancement strategies based on query type."""
        base_strategies = [EnhancementStrategy.SYNONYM_EXPANSION]
        
        if query_type in [QueryType.CAUSAL, QueryType.COMPARATIVE, QueryType.PROCEDURAL]:
            base_strategies.append(EnhancementStrategy.MULTI_HOP_REASONING)
        
        base_strategies.extend([
            EnhancementStrategy.CONTEXT_EXPANSION,
            EnhancementStrategy.ENTITY_EXPANSION
        ])
        
        return base_strategies
    
    def _calculate_confidence(
        self, 
        query: str, 
        expansion_terms: List[str], 
        query_type: QueryType
    ) -> float:
        """Calculate confidence score for enhanced query."""
        base_confidence = 0.5
        
        # Boost confidence based on expansion terms
        if expansion_terms:
            expansion_boost = min(0.3, len(expansion_terms) * 0.05)
            base_confidence += expansion_boost
        
        # Boost confidence based on query type specificity
        type_boosts = {
            QueryType.FACTUAL: 0.2,
            QueryType.PROCEDURAL: 0.15,
            QueryType.COMPARATIVE: 0.1,
            QueryType.TEMPORAL: 0.1,
            QueryType.CAUSAL: 0.05,
            QueryType.CONCEPTUAL: 0.0
        }
        
        base_confidence += type_boosts.get(query_type, 0.0)
        
        return min(1.0, base_confidence)
    
    def _update_enhancement_stats(self, expansion_count: int) -> None:
        """Update enhancement statistics."""
        total = self.enhancement_stats["total_enhancements"]
        current_avg = self.enhancement_stats["avg_expansion_terms"]
        
        new_avg = ((current_avg * (total - 1)) + expansion_count) / total
        self.enhancement_stats["avg_expansion_terms"] = new_avg
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get current enhancement statistics."""
        return dict(self.enhancement_stats)
