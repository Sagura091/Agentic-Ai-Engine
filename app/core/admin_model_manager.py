"""
🚀 Revolutionary Admin Model Management System

Centralized model management for administrators with:
- Direct Ollama integration for model downloads
- Model validation and storage management
- Real-time user notifications
- Comprehensive model lifecycle management

ADMIN-ONLY FEATURES:
✅ Model Download & Validation
✅ Storage Deduplication
✅ User Notification Broadcasting
✅ Model Performance Tracking
✅ Centralized Model Registry
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import httpx

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

from .global_config_manager import global_config_manager, ConfigurationSection

_backend_logger = get_backend_logger()

# Import notification handler (will be initialized later to avoid circular imports)
_notification_handler = None


class ModelInfo:
    """Model information container."""
    
    def __init__(self, name: str, size: int = 0, digest: str = "", 
                 modified_at: str = "", model_type: str = "llm", 
                 provider: str = "ollama", capabilities: List[str] = None):
        self.name = name
        self.size = size
        self.digest = digest
        self.modified_at = modified_at
        self.model_type = model_type
        self.provider = provider
        self.capabilities = capabilities or []
        self.download_date = datetime.utcnow().isoformat()
        self.status = "available"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "digest": self.digest,
            "modified_at": self.modified_at,
            "model_type": self.model_type,
            "provider": self.provider,
            "capabilities": self.capabilities,
            "download_date": self.download_date,
            "status": self.status
        }


class AdminModelManager:
    """🚀 Revolutionary Admin Model Management System"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.models_directory = Path("data/models")
        self.registry_file = self.models_directory / "model_registry.json"
        self._model_registry: Dict[str, ModelInfo] = {}
        self._download_progress: Dict[str, Dict[str, Any]] = {}
        
        # Ensure directories exist
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load model registry from disk."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    for name, model_data in data.items():
                        self._model_registry[name] = ModelInfo(
                            name=model_data["name"],
                            size=model_data.get("size", 0),
                            digest=model_data.get("digest", ""),
                            modified_at=model_data.get("modified_at", ""),
                            model_type=model_data.get("model_type", "llm"),
                            provider=model_data.get("provider", "ollama"),
                            capabilities=model_data.get("capabilities", [])
                        )
                _backend_logger.info(
                    f"✅ Loaded {len(self._model_registry)} models from registry",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.admin_model_manager"
                )
        except Exception as e:
            _backend_logger.warn(
                f"⚠️ Failed to load model registry: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager"
            )

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        try:
            registry_data = {name: model.to_dict() for name, model in self._model_registry.items()}
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            _backend_logger.debug(
                "✅ Model registry saved",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager"
            )
        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to save model registry: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager"
            )
    
    async def check_ollama_connection(self) -> Dict[str, Any]:
        """Check Ollama connection and get available models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    if "models" in data:
                        for model in data["models"]:
                            models.append({
                                "name": model.get("name", "Unknown"),
                                "size": model.get("size", 0),
                                "digest": model.get("digest", ""),
                                "modified_at": model.get("modified_at", "")
                            })
                    
                    return {
                        "connected": True,
                        "url": self.ollama_url,
                        "models": models,
                        "model_count": len(models)
                    }
                else:
                    return {
                        "connected": False,
                        "url": self.ollama_url,
                        "models": [],
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except httpx.ConnectError:
            return {
                "connected": False,
                "url": self.ollama_url,
                "models": [],
                "error": "Connection refused - Ollama not running"
            }
        except Exception as e:
            return {
                "connected": False,
                "url": self.ollama_url,
                "models": [],
                "error": str(e)
            }
    
    async def download_model(self, model_name: str, admin_user_id: str) -> Dict[str, Any]:
        """Download a model via Ollama."""
        try:
            _backend_logger.info(
                f"🚀 Starting model download: {model_name}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager",
                data={"admin_user": admin_user_id}
            )
            
            # Check if already downloading
            if model_name in self._download_progress:
                return {
                    "success": False,
                    "message": f"Model {model_name} is already being downloaded",
                    "status": "already_downloading"
                }
            
            # Initialize download progress
            self._download_progress[model_name] = {
                "status": "starting",
                "progress": 0,
                "admin_user": admin_user_id,
                "start_time": datetime.utcnow().isoformat()
            }
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/pull",
                    json={"name": model_name, "stream": False}
                )
                
                if response.status_code == 200:
                    # Update progress
                    self._download_progress[model_name]["status"] = "completed"
                    self._download_progress[model_name]["progress"] = 100
                    
                    # Add to registry
                    model_info = ModelInfo(
                        name=model_name,
                        provider="ollama",
                        model_type="llm"
                    )
                    self._model_registry[model_name] = model_info
                    self._save_registry()
                    
                    # Broadcast model availability
                    await self._broadcast_model_availability(model_name, "added", admin_user_id)
                    
                    # Clean up progress
                    del self._download_progress[model_name]

                    _backend_logger.info(
                        f"✅ Model downloaded successfully: {model_name}",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.admin_model_manager"
                    )

                    return {
                        "success": True,
                        "message": f"Model {model_name} downloaded successfully",
                        "model_info": model_info.to_dict()
                    }
                else:
                    error_text = response.text
                    self._download_progress[model_name]["status"] = "failed"
                    self._download_progress[model_name]["error"] = error_text

                    return {
                        "success": False,
                        "message": f"Failed to download model: HTTP {response.status_code}",
                        "error": error_text
                    }

        except Exception as e:
            _backend_logger.error(
                f"❌ Model download failed: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager",
                data={"model": model_name}
            )
            
            if model_name in self._download_progress:
                self._download_progress[model_name]["status"] = "failed"
                self._download_progress[model_name]["error"] = str(e)
            
            return {
                "success": False,
                "message": f"Model download failed: {str(e)}",
                "error": str(e)
            }
    
    async def _broadcast_model_availability(self, model_name: str, action: str, admin_user_id: str) -> None:
        """🚀 Broadcast model availability changes to all users via notification system."""
        try:
            # Get notification handler
            global _notification_handler
            if _notification_handler is None:
                from ..api.websocket.notification_handlers import notification_handler
                _notification_handler = notification_handler

            # Get model details
            model_details = {}
            if model_name in self._model_registry:
                model_details = self._model_registry[model_name].to_dict()

            # Send notification via WebSocket
            result = await _notification_handler.send_model_availability_notification(
                model_name=model_name,
                action=action,
                admin_user_id=admin_user_id,
                model_details=model_details
            )

            # Also update global configuration to trigger observers
            changes = {
                "available_models_update": {
                    "model_name": model_name,
                    "action": action,  # "added", "removed", "updated"
                    "timestamp": datetime.utcnow().isoformat(),
                    "admin_user": admin_user_id,
                    "model_details": model_details
                }
            }
            
            await global_config_manager.update_configuration(
                section=ConfigurationSection.LLM_PROVIDERS,
                changes=changes,
                user_id=admin_user_id
            )
            
            _backend_logger.info(
                f"📢 Broadcasted model availability: {model_name} {action}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager"
            )

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to broadcast model availability: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager"
            )
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get all available models from registry."""
        return [model.to_dict() for model in self._model_registry.values()]
    
    def get_download_progress(self, model_name: str = None) -> Dict[str, Any]:
        """Get download progress for specific model or all downloads."""
        if model_name:
            return self._download_progress.get(model_name, {})
        return self._download_progress.copy()
    
    async def remove_model(self, model_name: str, admin_user_id: str) -> Dict[str, Any]:
        """Remove a model from the system."""
        try:
            # Remove from Ollama
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(f"{self.ollama_url}/api/delete", json={"name": model_name})
                
                if response.status_code == 200:
                    # Remove from registry
                    if model_name in self._model_registry:
                        del self._model_registry[model_name]
                        self._save_registry()
                    
                    # Broadcast removal
                    await self._broadcast_model_availability(model_name, "removed", admin_user_id)
                    
                    return {
                        "success": True,
                        "message": f"Model {model_name} removed successfully"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to remove model: HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to remove model: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager"
            )
            return {
                "success": False,
                "message": f"Failed to remove model: {str(e)}"
            }

    async def get_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get the current model registry."""
        try:
            registry = {}
            for model_name, model_info in self._model_registry.items():
                registry[model_name] = model_info.to_dict()
            return registry
        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to get model registry: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.admin_model_manager"
            )
            return {}


# Global instance
admin_model_manager = AdminModelManager()
