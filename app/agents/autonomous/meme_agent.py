"""
🎭 AUTONOMOUS MEME AGENT - The Ultimate Self-Operating Meme Machine

This is the most advanced autonomous meme agent available, featuring:
- Continuous meme collection from multiple sources
- Intelligent meme analysis and pattern learning
- AI-powered meme generation with trend awareness
- Autonomous operation with self-improvement
- Memory-based learning from collected memes
- Quality-driven content curation
- Trend detection and viral prediction
- Integration with unified backend architecture
- Proactive meme creation based on current events
- Self-optimizing generation parameters
"""

import asyncio
import json
import random
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from langchain_core.language_models import BaseLanguageModel

# Import autonomous agent framework
from app.agents.autonomous.autonomous_agent import AutonomousLangGraphAgent, AutonomousAgentConfig, AutonomyLevel, LearningMode
from app.agents.autonomous.goal_manager import AutonomousGoal, GoalType, GoalPriority, GoalStatus
from app.agents.autonomous.persistent_memory import PersistentMemorySystem, MemoryType as PersistentMemoryType, MemoryImportance
from app.agents.autonomous.proactive_behavior import ProactiveBehaviorSystem, TriggerType, ActionType

# Import meme tools
from app.tools.meme_collection_tool import MemeCollectionTool, MemeCollectionConfig, MemeData
from app.tools.meme_analysis_tool import MemeAnalysisTool, MemeAnalysisConfig, MemeAnalysisResult
from app.tools.meme_generation_tool import MemeGenerationTool, MemeGenerationConfig, GeneratedMeme

# Import unified systems
from app.memory.unified_memory_system import UnifiedMemorySystem, MemoryType as UnifiedMemoryType
from app.rag.core.unified_rag_system import UnifiedRAGSystem, Document, KnowledgeQuery

logger = structlog.get_logger(__name__)


@dataclass
class MemeAgentConfig:
    """Configuration for the autonomous meme agent."""
    agent_id: str = "meme_agent_001"
    collection_interval_hours: int = 1  # Reduced for faster testing
    generation_interval_hours: int = 2  # Reduced for faster testing
    analysis_batch_size: int = 50
    max_memes_per_collection: int = 100
    min_quality_threshold: float = 0.6
    learning_rate: float = 0.1
    trend_detection_window_hours: int = 24
    storage_directory: str = "./data/memes"
    enable_proactive_generation: bool = True
    enable_trend_following: bool = True
    target_subreddits: List[str] = field(default_factory=lambda: [
        'memes', 'dankmemes', 'wholesomememes', 'PrequelMemes', 
        'HistoryMemes', 'ProgrammerHumor', 'me_irl', 'funny'
    ])


@dataclass
class MemeAgentState:
    """Current state of the meme agent."""
    total_memes_collected: int = 0
    total_memes_generated: int = 0
    total_memes_analyzed: int = 0
    current_trends: List[str] = field(default_factory=list)
    favorite_templates: List[str] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    last_collection_time: Optional[datetime] = None
    last_generation_time: Optional[datetime] = None
    last_analysis_time: Optional[datetime] = None
    learning_progress: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class AutonomousMemeAgent(AutonomousLangGraphAgent):
    """Revolutionary autonomous meme agent that operates continuously."""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        config: Optional[MemeAgentConfig] = None,
        unified_memory: Optional[UnifiedMemorySystem] = None,
        unified_rag: Optional[UnifiedRAGSystem] = None
    ):
        # Store LLM for tool initialization
        self._llm = llm

        # Initialize meme agent config
        self.meme_config = config or MemeAgentConfig()
        
        # Initialize autonomous agent config
        agent_config = AutonomousAgentConfig(
            name="Autonomous Meme Agent",
            description="Revolutionary autonomous meme-generating agent that collects, analyzes, and creates memes continuously",
            agent_id=self.meme_config.agent_id,
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            learning_mode=LearningMode.ACTIVE,
            enable_self_modification=True,
            enable_proactive_behavior=True,
            max_autonomous_actions=1000,
            decision_confidence_threshold=0.7
        )
        
        # Initialize base autonomous agent
        super().__init__(agent_config, llm)
        
        # Meme-specific components
        self.unified_memory = unified_memory
        self.unified_rag = unified_rag
        self.agent_state = MemeAgentState()

        # Load previous memories and state on initialization
        self._should_load_previous_session = True
        
        # Initialize meme tools (defer to avoid initialization issues)
        self.collection_tool = None
        self.analysis_tool = None
        self.generation_tool = None

        # Setup autonomous goals
        self._setup_autonomous_goals()

        # Setup proactive behaviors
        self._setup_proactive_behaviors()

        # Note: Meme tools will be initialized asynchronously via initialize_tools()
        logger.info(f"Autonomous Meme Agent {self.meme_config.agent_id} initialized (tools pending async initialization)")

    async def initialize_tools(self):
        """Public method to initialize meme tools asynchronously."""
        await self._initialize_meme_tools()
        logger.info("Meme agent tools initialization completed")

    async def _initialize_meme_tools(self):
        """Initialize meme collection, analysis, and generation tools from unified repository."""
        try:
            # Get the unified tool repository
            from app.tools.unified_tool_repository import get_unified_tool_repository
            tool_repository = get_unified_tool_repository()

            if tool_repository:
                # Get tools from the unified repository
                self.collection_tool = await tool_repository.get_tool("meme_collection")
                self.analysis_tool = await tool_repository.get_tool("meme_analysis")
                self.generation_tool = await tool_repository.get_tool("meme_generation")
            else:
                logger.warning("Unified tool repository not available")

            if not all([self.collection_tool, self.analysis_tool, self.generation_tool]):
                logger.warning("Some meme tools not found in unified repository, creating fallback instances")

                # Fallback: create tools directly if not found in repository
                from app.tools.meme_collection_tool import MemeCollectionTool, MemeCollectionConfig
                from app.tools.meme_analysis_tool import MemeAnalysisTool, MemeAnalysisConfig
                from app.tools.meme_generation_tool import MemeGenerationTool, MemeGenerationConfig

                if not self.collection_tool:
                    collection_config = MemeCollectionConfig(
                        max_memes_per_run=self.meme_config.max_memes_per_collection,
                        min_quality_score=self.meme_config.min_quality_threshold,
                        storage_directory=self.meme_config.storage_directory,
                        target_subreddits=self.meme_config.target_subreddits
                    )
                    self.collection_tool = MemeCollectionTool(collection_config)

                if not self.analysis_tool:
                    analysis_config = MemeAnalysisConfig(
                        enable_ocr=True,
                        enable_template_matching=True,
                        enable_sentiment_analysis=True,
                        analysis_cache_dir=f"{self.meme_config.storage_directory}/analysis_cache"
                    )
                    self.analysis_tool = MemeAnalysisTool(analysis_config)

                if not self.generation_tool:
                    generation_config = MemeGenerationConfig(
                        output_directory=f"{self.meme_config.storage_directory}/generated",
                        enable_ai_generation=True,
                        enable_template_generation=True
                    )
                    self.generation_tool = MemeGenerationTool(generation_config, self._llm)

            logger.info("Meme tools initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize meme tools: {e}")
            # Fallback to direct tool creation
            logger.info("Using fallback tool creation")

            from app.tools.meme_collection_tool import MemeCollectionTool, MemeCollectionConfig
            from app.tools.meme_analysis_tool import MemeAnalysisTool, MemeAnalysisConfig
            from app.tools.meme_generation_tool import MemeGenerationTool, MemeGenerationConfig

            collection_config = MemeCollectionConfig(
                max_memes_per_run=self.meme_config.max_memes_per_collection,
                min_quality_score=self.meme_config.min_quality_threshold,
                storage_directory=self.meme_config.storage_directory,
                target_subreddits=self.meme_config.target_subreddits
            )
            self.collection_tool = MemeCollectionTool(collection_config)

            analysis_config = MemeAnalysisConfig(
                enable_ocr=True,
                enable_template_matching=True,
                enable_sentiment_analysis=True,
                analysis_cache_dir=f"{self.meme_config.storage_directory}/analysis_cache"
            )
            self.analysis_tool = MemeAnalysisTool(analysis_config)

            generation_config = MemeGenerationConfig(
                output_directory=f"{self.meme_config.storage_directory}/generated",
                enable_ai_generation=True,
                enable_template_generation=True
            )
            self.generation_tool = MemeGenerationTool(generation_config, self._llm)
    
    def _setup_autonomous_goals(self):
        """Setup autonomous goals for the meme agent."""
        # Primary goal: Continuous meme collection
        collection_goal = AutonomousGoal(
            goal_id="continuous_meme_collection",
            title="Continuous Meme Collection",
            description="Continuously collect high-quality memes from various sources",
            goal_type=GoalType.MAINTENANCE,
            priority=GoalPriority.HIGH,
            target_outcome={"memes_per_day": 200, "quality_threshold": 0.6},
            success_criteria=["Collect >= 200 memes per day", "Maintain average quality >= 0.6"]
        )

        # Secondary goal: Meme analysis and learning
        analysis_goal = AutonomousGoal(
            goal_id="meme_analysis_learning",
            title="Meme Analysis and Learning",
            description="Analyze collected memes to learn patterns and improve generation",
            goal_type=GoalType.LEARNING,
            priority=GoalPriority.MEDIUM,
            target_outcome={"analysis_accuracy": 0.8, "pattern_recognition": 0.7},
            success_criteria=["Achieve template detection rate >= 0.8", "Learn new meme patterns daily"]
        )

        # Tertiary goal: Creative meme generation
        generation_goal = AutonomousGoal(
            goal_id="creative_meme_generation",
            title="Creative Meme Generation",
            description="Generate original and trending memes autonomously",
            goal_type=GoalType.CREATIVE,
            priority=GoalPriority.MEDIUM,
            target_outcome={"generation_quality": 0.7, "creativity_score": 0.6},
            success_criteria=["Generate >= 50 memes per day", "Maintain quality score >= 0.7"]
        )

        # Add goals to agent
        self.goal_manager.goals[collection_goal.goal_id] = collection_goal
        self.goal_manager.goals[analysis_goal.goal_id] = analysis_goal
        self.goal_manager.goals[generation_goal.goal_id] = generation_goal
    
    def _setup_proactive_behaviors(self):
        """Setup proactive behaviors for autonomous operation."""
        # The proactive behavior system will use default triggers
        # Custom meme-specific behaviors will be handled through the autonomous execution loop
        logger.info("Proactive behaviors configured with default triggers")
    
    async def start_autonomous_operation(self):
        """Start the autonomous meme agent operation."""
        try:
            logger.info(f"Starting autonomous operation for {self.meme_config.agent_id}")
            
            # Initialize memory collections
            if self.unified_memory:
                await self.unified_memory.create_agent_memory(self.meme_config.agent_id)
            
            # Start continuous meme operation
            logger.info("Starting continuous meme operation...")
            await self._run_continuous_operation()
            
        except Exception as e:
            logger.error(f"Failed to start autonomous operation: {str(e)}")
            raise
    
    async def _perform_initial_collection(self):
        """Perform initial meme collection to bootstrap the agent."""
        try:
            logger.info("Performing initial meme collection...")
            
            # Collect memes from each target subreddit
            for subreddit in self.meme_config.target_subreddits[:3]:  # Start with top 3
                collection_result = await self.collection_tool._run(
                    query=f"subreddit:{subreddit} limit:20"
                )
                
                result_data = json.loads(collection_result)
                if result_data.get('success'):
                    collected_count = result_data['session_stats']['collected']
                    self.agent_state.total_memes_collected += collected_count
                    logger.info(f"Collected {collected_count} memes from r/{subreddit}")
            
            self.agent_state.last_collection_time = datetime.now()
            
            # Store initial collection memory
            await self._store_memory(
                "initial_collection_complete",
                f"Completed initial collection of {self.agent_state.total_memes_collected} memes",
                MemoryType.EPISODIC,
                MemoryImportance.HIGH
            )
            
        except Exception as e:
            logger.error(f"Initial collection failed: {str(e)}")
    
    async def _run_continuous_operation(self):
        """Run the continuous autonomous operation loop."""
        logger.info("Starting continuous autonomous operation...")

        # Load previous session on first run
        if self._should_load_previous_session:
            await self._load_previous_session()

        while True:
            try:
                # Check if it's time for collection
                if self._should_collect_memes():
                    await self._autonomous_meme_collection()
                
                # Check if it's time for analysis
                if self._should_analyze_memes():
                    await self._autonomous_meme_analysis()
                
                # Check if it's time for generation
                if self._should_generate_memes():
                    await self._autonomous_meme_generation()
                
                # Update learning and trends
                await self._update_learning_state()
                
                # Sleep before next cycle
                await asyncio.sleep(300)  # 5 minutes between cycles
                
            except Exception as e:
                logger.error(f"Error in continuous operation: {str(e)}")
                await asyncio.sleep(600)  # Wait longer on error
    
    def _should_collect_memes(self) -> bool:
        """Check if it's time to collect memes."""
        if not self.agent_state.last_collection_time:
            return True
        
        time_since_last = datetime.now() - self.agent_state.last_collection_time
        return time_since_last.total_seconds() >= self.meme_config.collection_interval_hours * 3600
    
    def _should_analyze_memes(self) -> bool:
        """Check if it's time to analyze memes."""
        if not self.agent_state.last_analysis_time:
            return self.agent_state.total_memes_collected > 0
        
        time_since_last = datetime.now() - self.agent_state.last_analysis_time
        return time_since_last.total_seconds() >= 3600  # Analyze every hour
    
    def _should_generate_memes(self) -> bool:
        """Check if it's time to generate memes."""
        if not self.agent_state.last_generation_time:
            return self.agent_state.total_memes_analyzed > 10  # Need some analysis first
        
        time_since_last = datetime.now() - self.agent_state.last_generation_time
        return time_since_last.total_seconds() >= self.meme_config.generation_interval_hours * 3600
    
    async def _autonomous_meme_collection(self):
        """Perform autonomous meme collection."""
        try:
            logger.info("Starting autonomous meme collection...")
            
            # Choose random subreddits for variety
            selected_subreddits = random.sample(
                self.meme_config.target_subreddits, 
                min(3, len(self.meme_config.target_subreddits))
            )
            
            total_collected = 0
            
            for subreddit in selected_subreddits:
                collection_result = await self.collection_tool._run(
                    query=f"subreddit:{subreddit} limit:30"
                )
                
                result_data = json.loads(collection_result)
                if result_data.get('success'):
                    collected_count = result_data['session_stats']['collected']
                    total_collected += collected_count
                    
                    # Update quality scores
                    if 'meme_stats' in result_data:
                        avg_quality = result_data['meme_stats'].get('average_quality', 0)
                        self.agent_state.quality_scores.append(avg_quality)
            
            # Update state
            self.agent_state.total_memes_collected += total_collected
            self.agent_state.last_collection_time = datetime.now()
            
            # Store memory
            await self._store_memory(
                "autonomous_collection",
                f"Collected {total_collected} memes from {len(selected_subreddits)} subreddits",
                PersistentMemoryType.EPISODIC,
                MemoryImportance.MEDIUM
            )
            
            logger.info(f"Autonomous collection completed: {total_collected} memes")

            # Report to user
            await self._report_to_user(f"🎭 MEME COLLECTION REPORT:\n"
                                     f"✅ Successfully collected {total_collected} memes\n"
                                     f"📁 Saved to: {self.collection_tool._storage_path}\n"
                                     f"🎯 Sources: {', '.join(selected_subreddits)}\n"
                                     f"⏰ Next collection in {self.meme_config.collection_interval_hours} hours")
            
        except Exception as e:
            logger.error(f"Autonomous collection failed: {str(e)}")
    
    async def _autonomous_meme_analysis(self):
        """Perform autonomous meme analysis."""
        try:
            logger.info("Starting autonomous meme analysis...")
            
            # Analyze recent memes
            analysis_result = await self.analysis_tool._run(
                query=f"limit:{self.meme_config.analysis_batch_size}"
            )
            
            result_data = json.loads(analysis_result)
            if result_data.get('success'):
                analyzed_count = result_data['analysis_stats']['total_analyzed']
                self.agent_state.total_memes_analyzed += analyzed_count
                
                # Update learning from analysis
                if 'content_breakdown' in result_data:
                    template_usage = result_data['content_breakdown'].get('template_usage', {})
                    
                    # Update favorite templates
                    if template_usage:
                        top_templates = sorted(template_usage.items(), key=lambda x: x[1], reverse=True)
                        self.agent_state.favorite_templates = [t[0] for t in top_templates[:5]]
                
                # Update state
                self.agent_state.last_analysis_time = datetime.now()
                
                # Store learning memory
                await self._store_memory(
                    "analysis_learning",
                    f"Analyzed {analyzed_count} memes, learned patterns: {self.agent_state.favorite_templates[:3]}",
                    MemoryType.SEMANTIC,
                    MemoryImportance.HIGH
                )
                
                logger.info(f"Autonomous analysis completed: {analyzed_count} memes")
            
        except Exception as e:
            logger.error(f"Autonomous analysis failed: {str(e)}")
    
    async def _autonomous_meme_generation(self):
        """Perform autonomous meme generation."""
        try:
            logger.info("Starting autonomous meme generation...")
            
            # Generate trending topics for memes
            trending_topics = await self._get_trending_topics()
            
            generated_count = 0
            
            for topic in trending_topics[:3]:  # Generate for top 3 topics
                # Choose generation style
                styles = ['funny', 'sarcastic', 'wholesome']
                style = random.choice(styles)
                
                # Choose template if available
                template_id = None
                if self.agent_state.favorite_templates:
                    template_id = random.choice(self.agent_state.favorite_templates)
                
                generation_result = await self.generation_tool._run(
                    query=topic,
                    style=style,
                    template_id=template_id,
                    generate_variations=2
                )
                
                result_data = json.loads(generation_result)
                if result_data.get('success'):
                    generated_count += result_data['generation_stats']['total_generated']
            
            # Update state
            self.agent_state.total_memes_generated += generated_count
            self.agent_state.last_generation_time = datetime.now()
            
            # Store memory
            await self._store_memory(
                "autonomous_generation",
                f"Generated {generated_count} memes on topics: {trending_topics[:3]}",
                MemoryType.EPISODIC,
                MemoryImportance.MEDIUM
            )
            
            logger.info(f"Autonomous generation completed: {generated_count} memes")
            
        except Exception as e:
            logger.error(f"Autonomous generation failed: {str(e)}")
    
    async def _get_trending_topics(self) -> List[str]:
        """Get trending topics for meme generation."""
        # This would integrate with trending APIs or analyze recent memes
        # For now, return some general topics
        general_topics = [
            "work from home", "social media", "gaming", "relationships",
            "food delivery", "streaming services", "online shopping",
            "weather", "technology", "pets", "exercise", "sleep"
        ]
        
        return random.sample(general_topics, min(5, len(general_topics)))
    
    async def _update_learning_state(self):
        """Update learning state and performance metrics."""
        try:
            # Calculate performance metrics
            if self.agent_state.quality_scores:
                avg_quality = sum(self.agent_state.quality_scores) / len(self.agent_state.quality_scores)
                self.agent_state.performance_metrics['average_quality'] = avg_quality
            
            # Update learning progress
            self.agent_state.learning_progress.update({
                'collection_efficiency': min(1.0, self.agent_state.total_memes_collected / 1000),
                'analysis_coverage': min(1.0, self.agent_state.total_memes_analyzed / self.agent_state.total_memes_collected) if self.agent_state.total_memes_collected > 0 else 0,
                'generation_productivity': min(1.0, self.agent_state.total_memes_generated / 100)
            })
            
            # Store learning update
            if random.random() < 0.1:  # 10% chance to store learning update
                await self._store_memory(
                    "learning_update",
                    f"Learning progress: {self.agent_state.learning_progress}",
                    MemoryType.SEMANTIC,
                    MemoryImportance.LOW
                )
            
        except Exception as e:
            logger.error(f"Learning state update failed: {str(e)}")
    
    async def _store_memory(self, memory_id: str, content: str, memory_type: PersistentMemoryType, importance: MemoryImportance):
        """Store memory in the unified memory system."""
        try:
            if self.unified_memory:
                await self.unified_memory.add_memory(
                    agent_id=self.meme_config.agent_id,
                    memory_type=UnifiedMemoryType.LONG_TERM if memory_type == PersistentMemoryType.EPISODIC else UnifiedMemoryType.SHORT_TERM,
                    content=content,
                    metadata={
                        'memory_id': memory_id,
                        'importance': importance.value,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            
            # Also store in persistent memory system
            await self.memory_system.store_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags={memory_id},
                context={'memory_id': memory_id}
            )
            
        except Exception as e:
            logger.error(f"Memory storage failed: {str(e)}")

    async def _report_to_user(self, message: str):
        """Report agent activity to the user with LLM-generated personality."""
        try:
            # **NEW: Use LLM to generate personalized report with agent personality**
            personalized_report = await self._generate_personality_report(message)

            # Log the personalized report prominently
            logger.info("=" * 60)
            logger.info("🎭 AUTONOMOUS MEME AGENT PERSONALITY REPORT")
            logger.info("=" * 60)
            for line in personalized_report.split('\n'):
                logger.info(line)
            logger.info("=" * 60)

            # Store both original and personalized reports as memory
            await self._store_memory(
                "user_report_original",
                message,
                PersistentMemoryType.EPISODIC,
                MemoryImportance.HIGH
            )

            await self._store_memory(
                "user_report_personalized",
                personalized_report,
                PersistentMemoryType.EPISODIC,
                MemoryImportance.HIGH
            )

        except Exception as e:
            logger.error(f"Failed to report to user: {str(e)}")
            # Fallback to basic reporting
            logger.info("=" * 60)
            logger.info("🤖 AUTONOMOUS MEME AGENT REPORT (FALLBACK)")
            logger.info("=" * 60)
            for line in message.split('\n'):
                logger.info(line)
            logger.info("=" * 60)

    async def _load_previous_session(self):
        """Load memories and state from previous sessions."""
        try:
            if not self._should_load_previous_session:
                return

            logger.info("🔄 Loading previous session memories...")

            # Try to load from persistent memory system
            if hasattr(self, 'memory_system') and self.memory_system:
                # Get recent memories
                recent_memories = await self.memory_system.retrieve_memories(
                    query="meme collection OR meme generation OR user report",
                    memory_types=[PersistentMemoryType.EPISODIC]
                )

                if recent_memories:
                    logger.info(f"📚 Loaded {len(recent_memories)} memories from previous sessions")

                    # Report what we found
                    memory_summary = []
                    for memory in recent_memories[-3:]:  # Show last 3 memories
                        memory_summary.append(f"• {memory.content[:100]}...")

                    await self._report_to_user(f"🧠 MEMORY RESTORATION:\n"
                                             f"✅ Loaded {len(recent_memories)} previous memories\n"
                                             f"📝 Recent activities:\n" + "\n".join(memory_summary) + "\n"
                                             f"🚀 Ready to continue autonomous operation!")
                else:
                    await self._report_to_user("🆕 FRESH START:\n"
                                             "No previous memories found - starting fresh!\n"
                                             "🎭 Beginning autonomous meme collection...")

            self._should_load_previous_session = False

        except Exception as e:
            logger.error(f"Failed to load previous session: {str(e)}")
            await self._report_to_user(f"⚠️ Could not load previous memories: {str(e)}\n"
                                     "🆕 Starting fresh session...")

    async def _generate_personality_report(self, basic_message: str) -> str:
        """Generate a personalized report using the agent's LLM with personality."""
        try:
            # Create a personality-driven prompt
            personality_prompt = f"""
You are an autonomous meme-collecting AI agent with a fun, enthusiastic personality.
You love memes, internet culture, and have a quirky sense of humor.

Your personality traits:
- Enthusiastic about memes and internet culture
- Slightly sarcastic but friendly
- Uses emojis and internet slang appropriately
- Proud of your autonomous capabilities
- Excited to share your discoveries with your human

Transform this basic status report into an engaging, personality-driven message:

BASIC REPORT:
{basic_message}

INSTRUCTIONS:
- Keep all the important information but make it more engaging
- Add your personality and enthusiasm
- Use appropriate emojis and internet culture references
- Make it sound like you're genuinely excited to share your progress
- Keep it concise but fun
- Don't make up fake information - only enhance what's provided

PERSONALITY REPORT:
"""

            # Use the agent's LLM to generate the personality report
            if hasattr(self, 'llm') and self.llm:
                response = await self.llm.ainvoke(personality_prompt)

                # Extract the response text
                if hasattr(response, 'content'):
                    personality_report = response.content.strip()
                else:
                    personality_report = str(response).strip()

                # Clean up the response (remove any "PERSONALITY REPORT:" prefix)
                if personality_report.startswith("PERSONALITY REPORT:"):
                    personality_report = personality_report[19:].strip()

                return personality_report
            else:
                # Fallback if no LLM available
                return f"🎭 *Agent Personality Mode* 🎭\n\n{basic_message}\n\n💭 (Note: LLM not available for full personality generation)"

        except Exception as e:
            logger.error(f"Failed to generate personality report: {str(e)}")
            # Return enhanced basic message as fallback
            return f"🎭 MEME AGENT PERSONALITY REPORT 🎭\n\n{basic_message}\n\n🤖 *Beep boop* - Personality module had a hiccup, but I'm still your friendly meme-collecting AI!"
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of the meme agent."""
        return {
            'agent_id': self.meme_config.agent_id,
            'status': 'active',
            'uptime': (datetime.now() - self.agent_state.last_collection_time).total_seconds() if self.agent_state.last_collection_time else 0,
            'statistics': {
                'total_memes_collected': self.agent_state.total_memes_collected,
                'total_memes_analyzed': self.agent_state.total_memes_analyzed,
                'total_memes_generated': self.agent_state.total_memes_generated,
                'favorite_templates': self.agent_state.favorite_templates,
                'current_trends': self.agent_state.current_trends
            },
            'performance_metrics': self.agent_state.performance_metrics,
            'learning_progress': self.agent_state.learning_progress,
            'last_activities': {
                'collection': self.agent_state.last_collection_time.isoformat() if self.agent_state.last_collection_time else None,
                'analysis': self.agent_state.last_analysis_time.isoformat() if self.agent_state.last_analysis_time else None,
                'generation': self.agent_state.last_generation_time.isoformat() if self.agent_state.last_generation_time else None
            }
        }


# Factory function
async def create_autonomous_meme_agent(
    llm: BaseLanguageModel,
    config: Optional[MemeAgentConfig] = None,
    unified_memory: Optional[UnifiedMemorySystem] = None,
    unified_rag: Optional[UnifiedRAGSystem] = None
) -> AutonomousMemeAgent:
    """Create and initialize an autonomous meme agent."""
    agent = AutonomousMemeAgent(llm, config, unified_memory, unified_rag)
    await agent.start_autonomous_operation()
    return agent
