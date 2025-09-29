#!/usr/bin/env python3
"""
Clean Logging Configuration - Reduces spam and shows only essential agent output
"""

import logging
import sys
from typing import Optional

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
