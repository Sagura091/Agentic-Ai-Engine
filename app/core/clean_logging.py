#!/usr/bin/env python3
"""
Clean Logging Configuration - Reduces spam and shows only essential agent output
"""

import logging
import sys
from typing import Optional, Dict

class CleanAgentFormatter(logging.Formatter):
    """Custom formatter that only shows essential agent messages."""
    
    def format(self, record):
        # Only show specific agent-related messages
        essential_keywords = [
            "🧠", "🤖", "✅", "❌", "🎯", "📊", "🔥", "💡", 
            "Agent Decision", "LLM REASONING", "FINAL DECISION",
            "Agent thinking", "Agent responding", "File generated",
            "Task completed", "Analysis complete"
        ]
        
        # Check if this is an essential message
        if any(keyword in record.getMessage() for keyword in essential_keywords):
            return super().format(record)
        
        # Skip non-essential messages
        return ""

def setup_clean_logging(
    agent_name: str = "Agent",
    log_level: str = "WARNING",
    show_agent_output: bool = True,
    log_file: Optional[str] = None
):
    """
    Setup clean logging configuration that reduces spam.
    
    Args:
        agent_name: Name of the agent for log formatting
        log_level: Logging level (WARNING, ERROR, CRITICAL)
        show_agent_output: Whether to show agent-specific output
        log_file: Optional file to write detailed logs to
    """
    
    # Set root logger to WARNING to reduce spam
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with clean formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if show_agent_output:
        # Use clean formatter for agent output
        formatter = CleanAgentFormatter(
            fmt=f'%(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        # Minimal formatter
        formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Optional file handler for detailed logs
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress specific noisy loggers
    noisy_loggers = [
        'chromadb',
        'sentence_transformers',
        'transformers',
        'urllib3',
        'requests',
        'httpx',
        'httpcore',
        'openai',
        'anthropic',
        'ollama',
        'playwright',
        'selenium'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Set app loggers to show only warnings and errors
    app_loggers = [
        'app.core',
        'app.rag',
        'app.memory',
        'app.tools',
        'app.agents'
    ]
    
    for logger_name in app_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def create_agent_logger(agent_name: str) -> logging.Logger:
    """Create a logger specifically for agent output."""
    logger = logging.getLogger(f"agent.{agent_name}")
    logger.setLevel(logging.INFO)
    return logger

# Convenience functions for agent logging
def agent_thinking(message: str, agent_name: str = "Agent"):
    """Log agent thinking process."""
    logger = create_agent_logger(agent_name)
    logger.info(f"🧠 {agent_name} thinking: {message}")

def agent_responding(message: str, agent_name: str = "Agent"):
    """Log agent response."""
    logger = create_agent_logger(agent_name)
    logger.info(f"🤖 {agent_name} responding: {message}")

def agent_decision(message: str, agent_name: str = "Agent"):
    """Log agent decision."""
    logger = create_agent_logger(agent_name)
    logger.info(f"🎯 {agent_name} decision: {message}")

def agent_file_generated(filename: str, agent_name: str = "Agent"):
    """Log file generation."""
    logger = create_agent_logger(agent_name)
    logger.info(f"📄 {agent_name} generated file: {filename}")

def agent_task_complete(task: str, duration: float, agent_name: str = "Agent"):
    """Log task completion."""
    logger = create_agent_logger(agent_name)
    logger.info(f"✅ {agent_name} completed task '{task}' in {duration:.2f}s")

def agent_error(error: str, agent_name: str = "Agent"):
    """Log agent error."""
    logger = create_agent_logger(agent_name)
    logger.error(f"❌ {agent_name} error: {error}")


# ============================================================================
# REVOLUTIONARY CONVERSATION LOGGER
# ============================================================================

class ConversationLogger:
    """
    Revolutionary conversation logger for clean, user-facing agent dialogue.

    This logger provides a clean, emoji-enhanced interface for logging agent
    conversations without technical details, correlation IDs, or timestamps.
    """

    def __init__(self, agent_name: str = "Agent", config: Optional[Dict] = None):
        """
        Initialize conversation logger.

        Args:
            agent_name: Name of the agent
            config: Optional configuration dictionary
        """
        self.agent_name = agent_name
        self.config = config or {}
        self.emoji_enhanced = self.config.get('emoji_enhanced', True)
        self.show_reasoning = self.config.get('show_reasoning', True)
        self.show_tool_usage = self.config.get('show_tool_usage', True)
        self.show_tool_results = self.config.get('show_tool_results', True)
        self.max_reasoning_length = self.config.get('max_reasoning_length', 200)
        self.max_result_length = self.config.get('max_result_length', 500)

        # Create dedicated conversation logger
        self.logger = logging.getLogger(f"conversation.{agent_name}")
        self.logger.setLevel(logging.INFO)

        # Ensure it has a handler
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text if too long"""
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def user_query(self, message: str):
        """Log user query"""
        emoji = "🧑 " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}User: {message}")

    def agent_acknowledgment(self, message: str):
        """Log agent acknowledgment"""
        emoji = "🤖 " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}Agent: {message}")

    def agent_thinking(self, message: str):
        """Log agent thinking/reasoning"""
        if not self.show_reasoning:
            return

        emoji = "🔍 " if self.emoji_enhanced else ""
        truncated = self._truncate(message, self.max_reasoning_length)
        self.logger.info(f"{emoji}Thinking: {truncated}")

    def agent_goal(self, message: str):
        """Log agent goal (autonomous agents)"""
        emoji = "🎯 " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}Goal: {message}")

    def agent_decision(self, message: str):
        """Log agent decision (autonomous agents)"""
        emoji = "🧠 " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}Decision: {message}")

    def tool_usage(self, tool_name: str, purpose: str = ""):
        """Log tool usage"""
        if not self.show_tool_usage:
            return

        emoji = "🔧 " if self.emoji_enhanced else ""
        if purpose:
            self.logger.info(f"{emoji}Using: {tool_name}\n   → {purpose}")
        else:
            self.logger.info(f"{emoji}Using: {tool_name}")

    def tool_result(self, message: str):
        """Log tool result"""
        if not self.show_tool_results:
            return

        emoji = "✅ " if self.emoji_enhanced else ""
        truncated = self._truncate(message, self.max_result_length)
        self.logger.info(f"{emoji}{truncated}")

    def agent_action(self, message: str):
        """Log agent action"""
        emoji = "⚙️ " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}Action: {message}")

    def agent_response(self, message: str):
        """Log agent final response"""
        emoji = "💬 " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}{message}")

    def agent_insight(self, message: str):
        """Log agent insight"""
        emoji = "💡 " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}Insight: {message}")

    def error(self, message: str):
        """Log error message"""
        emoji = "❌ " if self.emoji_enhanced else ""
        self.logger.error(f"{emoji}Error: {message}")

    def warning(self, message: str):
        """Log warning message"""
        emoji = "⚠️ " if self.emoji_enhanced else ""
        self.logger.warning(f"{emoji}Warning: {message}")

    def success(self, message: str):
        """Log success message"""
        emoji = "✅ " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}{message}")

    def info(self, message: str):
        """Log general informational message"""
        emoji = "ℹ️ " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}{message}")

    def progress(self, message: str):
        """Log progress update"""
        emoji = "⏳ " if self.emoji_enhanced else ""
        self.logger.info(f"{emoji}{message}")

    def separator(self):
        """Log a visual separator"""
        self.logger.info("")

    def header(self, message: str):
        """Log a header message"""
        emoji = "🚀 " if self.emoji_enhanced else ""
        self.logger.info(f"\n{emoji}{message}\n")


# ============================================================================
# GLOBAL CONVERSATION LOGGER HELPERS
# ============================================================================

# Global conversation logger instances (one per agent)
_conversation_loggers: Dict[str, ConversationLogger] = {}


def get_conversation_logger(agent_name: str = "Agent", config: Optional[Dict] = None) -> ConversationLogger:
    """Get or create a conversation logger for an agent (singleton per agent_name)"""
    if agent_name not in _conversation_loggers:
        _conversation_loggers[agent_name] = ConversationLogger(agent_name, config)
    return _conversation_loggers[agent_name]


def log_user_query(message: str, agent_name: str = "Agent"):
    """Log user query"""
    logger = get_conversation_logger(agent_name)
    logger.user_query(message)


def log_agent_thinking(message: str, agent_name: str = "Agent"):
    """Log agent thinking/reasoning"""
    logger = get_conversation_logger(agent_name)
    logger.agent_thinking(message)


def log_agent_action(message: str, agent_name: str = "Agent"):
    """Log agent action"""
    logger = get_conversation_logger(agent_name)
    logger.agent_action(message)


def log_tool_usage(tool_name: str, purpose: str = "", agent_name: str = "Agent"):
    """Log tool usage"""
    logger = get_conversation_logger(agent_name)
    logger.tool_usage(tool_name, purpose)


def log_agent_response(message: str, agent_name: str = "Agent"):
    """Log agent final response"""
    logger = get_conversation_logger(agent_name)
    logger.agent_response(message)
