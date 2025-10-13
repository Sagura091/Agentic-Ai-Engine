"""
Universal System Initialization Manager

This module ensures the system is fully initialized before any operation:
- Checks if essential models are downloaded
- Downloads missing models automatically
- Configures system to use downloaded models
- Handles both embedding models and Ollama LLM models

Called from:
- Backend startup (app/main.py)
- Agent startup (data/agents/templates/agent_template.py)
- Setup script (setup_system.py)

Author: Agentic AI System
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import structlog

from app.backend_logging import get_logger, LogCategory

logger = get_logger(__name__)


class SystemInitializationManager:
    """Manages system initialization and model downloads."""
    
    def __init__(self, silent: bool = False):
        """
        Initialize the system initialization manager.
        
        Args:
            silent: If True, suppress progress messages (for background initialization)
        """
        self.silent = silent
        self.status_file = Path("data/.initialization_status.json")
        self.status: Dict[str, Any] = {}
        self._load_status()
    
    def _load_status(self):
        """Load initialization status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    self.status = json.load(f)
            except Exception as e:
                logger.warning(
                    "Failed to load initialization status",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.system_initialization",
                    error=e
                )
                self.status = {}
        else:
            self.status = {}
    
    def _save_status(self):
        """Save initialization status to file."""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(
                "Failed to save initialization status",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.system_initialization",
                error=e
            )
    
    def _log(self, message: str, level: str = "info"):
        """Log a message (respects silent mode)."""
        if not self.silent:
            if level == "info":
                print(f"   {message}")
            elif level == "success":
                print(f"   ✅ {message}")
            elif level == "warning":
                print(f"   ⚠️  {message}")
            elif level == "error":
                print(f"   ❌ {message}")
        
        # Always log to file
        log_func = getattr(logger, level if level != "success" else "info")
        log_func(
            message,
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.system_initialization"
        )
    
    async def ensure_system_ready(self) -> bool:
        """
        Ensure the system is fully initialized and ready.
        
        This is the main entry point. It:
        1. Checks if initialization is complete
        2. Downloads missing models
        3. Configures system to use models
        4. Updates initialization status
        
        Returns:
            True if system is ready, False if initialization failed
        """
        try:
            if not self.silent:
                print("\n🔍 Checking system initialization...")
            
            # Check embedding models
            embedding_ready = await self._ensure_embedding_models()
            
            # Check Ollama models (optional - don't fail if Ollama not installed)
            ollama_ready = await self._ensure_ollama_models()
            
            # Update status
            self.status.update({
                "last_check": datetime.utcnow().isoformat(),
                "embedding_models_ready": embedding_ready,
                "ollama_ready": ollama_ready,
                "system_ready": embedding_ready  # Ollama is optional
            })
            self._save_status()
            
            if embedding_ready:
                self._log("System initialization complete", "success")
                return True
            else:
                self._log("System initialization incomplete", "warning")
                return False
                
        except Exception as e:
            logger.error(
                "System initialization failed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.system_initialization",
                error=e
            )
            self._log(f"Initialization error: {e}", "error")
            return False
    
    async def _ensure_embedding_models(self) -> bool:
        """
        Ensure essential embedding models are available.
        
        Returns:
            True if models are ready, False otherwise
        """
        try:
            self._log("Checking embedding models...")
            
            # Import model initialization service
            from app.rag.core.model_initialization_service import get_model_initialization_service
            from app.rag.config.required_models import EMBEDDING_MODELS, VISION_MODELS
            
            # Define essential models
            essential_models = [
                EMBEDDING_MODELS["all-MiniLM-L6-v2"],  # Default embedding model
            ]
            
            # Add vision model if vision is enabled
            try:
                essential_models.append(VISION_MODELS["clip-ViT-B-32"])
            except KeyError:
                pass  # Vision model not required
            
            # Get initialization service
            service = await get_model_initialization_service(silent_mode=self.silent)
            
            # Ensure models are available (smart detection - checks HF cache first)
            self._log("Initializing models (this may take a few minutes on first run)...")
            locations = await service.ensure_models_available(essential_models)
            
            # Check results
            all_ready = True
            downloaded_models = []
            
            for model_spec in essential_models:
                location = locations.get(model_spec.model_id)
                
                if location and location.is_valid:
                    downloaded_models.append(model_spec.model_id)
                    
                    if location.source == "huggingface_cache":
                        self._log(f"Found in HuggingFace cache: {model_spec.local_name}", "success")
                    elif location.source == "centralized":
                        self._log(f"Already available: {model_spec.local_name}", "success")
                    else:
                        self._log(f"Downloaded: {model_spec.local_name}", "success")
                else:
                    self._log(f"Failed to initialize: {model_spec.local_name}", "error")
                    all_ready = False
            
            # Update status
            self.status["embedding_models"] = downloaded_models
            self.status["embedding_models_timestamp"] = datetime.utcnow().isoformat()
            
            return all_ready
            
        except Exception as e:
            logger.error(
                "Failed to ensure embedding models",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.system_initialization",
                error=e
            )
            self._log(f"Embedding model initialization failed: {e}", "error")
            return False
    
    async def _ensure_ollama_models(self) -> bool:
        """
        Ensure Ollama models are available (if Ollama is installed).
        
        Returns:
            True if Ollama is ready (or not installed), False if Ollama is installed but models failed
        """
        try:
            self._log("Checking Ollama installation...")
            
            # Check if Ollama is installed and running
            from app.config.settings import get_settings
            from app.http_client import HTTPClient, ClientConfig
            
            settings = get_settings()
            config = ClientConfig(timeout=5, verify_ssl=False)
            
            try:
                async with HTTPClient(settings.OLLAMA_BASE_URL, config) as client:
                    response = await client.get("/api/tags", stream=False)
                    
                    if response.status_code != 200:
                        self._log("Ollama not running (optional)", "info")
                        self.status["ollama_installed"] = False
                        return True  # Not installed is OK
                    
                    # Ollama is running - check for models
                    installed_models = response.json().get("models", [])
                    model_names = [m["name"] for m in installed_models]
                    
                    self._log(f"Ollama is running ({len(model_names)} models installed)", "success")
                    
                    # Check for default model
                    default_model = settings.AGENTIC_DEFAULT_AGENT_MODEL or "llama3.2-vision:11b"
                    
                    if any(default_model in name for name in model_names):
                        self._log(f"Default model available: {default_model}", "success")
                        self.status["ollama_installed"] = True
                        self.status["ollama_models"] = model_names
                        self.status["ollama_default_model"] = default_model
                        return True
                    
                    # Default model not installed - try to pull it
                    self._log(f"Pulling default Ollama model: {default_model}...")
                    self._log("This may take several minutes...", "info")
                    
                    success = await self._pull_ollama_model(default_model)
                    
                    if success:
                        self._log(f"Successfully pulled: {default_model}", "success")
                        self.status["ollama_installed"] = True
                        self.status["ollama_models"] = model_names + [default_model]
                        self.status["ollama_default_model"] = default_model
                        return True
                    else:
                        self._log(f"Failed to pull: {default_model}", "warning")
                        self._log(f"You can pull it manually: ollama pull {default_model}", "info")
                        return True  # Don't fail - Ollama is optional
                        
            except Exception as e:
                self._log("Ollama not available (optional)", "info")
                self.status["ollama_installed"] = False
                return True  # Not installed is OK
                
        except Exception as e:
            logger.warning(
                "Ollama check failed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.system_initialization",
                error=e
            )
            return True  # Don't fail - Ollama is optional
    
    async def _pull_ollama_model(self, model_name: str) -> bool:
        """Pull an Ollama model."""
        try:
            # Use existing Ollama manager if available
            try:
                from detect_system_and_model import OllamaManager
                return await OllamaManager.pull_model(model_name)
            except ImportError:
                # Fallback to direct API call
                from app.config.settings import get_settings
                from app.http_client import HTTPClient, ClientConfig
                
                settings = get_settings()
                config = ClientConfig(timeout=600, verify_ssl=False)  # 10 min timeout for pull
                
                async with HTTPClient(settings.OLLAMA_BASE_URL, config) as client:
                    response = await client.post(
                        "/api/pull",
                        json={"name": model_name},
                        stream=False
                    )
                    return response.status_code == 200
                    
        except Exception as e:
            logger.error(
                f"Failed to pull Ollama model: {model_name}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.system_initialization",
                error=e
            )
            return False


# Global instance
_initialization_manager: Optional[SystemInitializationManager] = None


async def ensure_system_ready(silent: bool = False) -> bool:
    """
    Ensure the system is fully initialized and ready.
    
    This is the main entry point for system initialization.
    Call this from:
    - Backend startup (app/main.py)
    - Agent startup (before running agent)
    - Setup scripts
    
    Args:
        silent: If True, suppress progress messages
    
    Returns:
        True if system is ready, False otherwise
    """
    global _initialization_manager
    
    if _initialization_manager is None:
        _initialization_manager = SystemInitializationManager(silent=silent)
    
    return await _initialization_manager.ensure_system_ready()

