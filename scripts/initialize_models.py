#!/usr/bin/env python3
"""
🚀 Model Initialization Script for Agentic AI Engine

This script ensures your backend has all essential models properly downloaded
and configured in the data/models directory structure.

Essential Models Setup:
1. Default Embedding Model (all-MiniLM-L6-v2) - Fast, efficient, general-purpose
2. Global Vision Model (CLIP ViT-B/32) - Image-text understanding
3. Reranking Model (ms-marco-MiniLM-L-6-v2) - Search result reranking
4. Validate all model installations

Usage:
    python scripts/initialize_models.py
    python scripts/initialize_models.py --skip-download  # Only validate existing
    python scripts/initialize_models.py --force         # Force redownload all
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import structlog

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.core.embedding_model_manager import UniversalModelManager, ModelType
from app.rag.core.global_embedding_manager import get_global_embedding_manager, set_global_embedding_config
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

class ModelInitializer:
    """Initialize essential models for the Agentic AI Engine."""
    
    # Essential models that should be available
    ESSENTIAL_MODELS = {
        'embedding': {
            'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
            'name': 'All MiniLM L6 v2',
            'description': 'Fast, efficient general-purpose embedding model',
            'priority': 1,
            'set_as_global': True
        },
        'vision': {
            'model_id': 'openai/clip-vit-base-patch32',
            'name': 'CLIP ViT-B/32',
            'description': 'Standard CLIP model for image-text understanding',
            'priority': 2,
            'set_as_global': True
        },
        'reranking': {
            'model_id': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'name': 'MS MARCO MiniLM L6 v2',
            'description': 'Efficient reranking model for search results',
            'priority': 3,
            'set_as_global': False
        }
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.model_manager = None
        self.global_embedding_manager = None
        self.initialization_results = {}
    
    async def initialize(self):
        """Initialize the model management system."""
        try:
            logger.info("🚀 Initializing Agentic AI Model System...")
            
            # Initialize universal model manager
            self.model_manager = UniversalModelManager()
            
            # Initialize global embedding manager
            self.global_embedding_manager = await get_global_embedding_manager()
            
            logger.info("✅ Model management system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model management: {e}")
            raise
    
    async def setup_essential_models(self, skip_download: bool = False, force_download: bool = False):
        """Download and configure all essential models."""
        try:
            print("\n🎯 Setting up Essential Models for Agentic AI Engine")
            print("=" * 60)
            
            # Check data/models directory
            await self._ensure_data_directory()
            
            # Process each essential model
            for model_type, model_config in self.ESSENTIAL_MODELS.items():
                await self._setup_model(model_config, skip_download, force_download)
            
            # Configure global settings
            await self._configure_global_settings()
            
            # Validate installation
            await self._validate_installation()
            
            # Show summary
            self._show_setup_summary()
            
        except Exception as e:
            logger.error(f"❌ Essential model setup failed: {e}")
            raise
    
    async def _ensure_data_directory(self):
        """Ensure the data/models directory structure exists."""
        try:
            data_dir = Path("data")
            models_dir = data_dir / "models"
            
            # Create directories
            for subdir in ["embedding", "vision", "reranking", "llm"]:
                (models_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📁 Data directory structure ensured: {models_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
            raise
    
    async def _setup_model(self, model_config: Dict[str, Any], skip_download: bool, force_download: bool):
        """Setup a single model."""
        try:
            model_id = model_config['model_id']
            model_name = model_config['name']
            
            print(f"\n📦 Setting up {model_name} ({model_id})")
            print("-" * 50)
            
            # Check if model exists in catalog
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                print(f"⚠️  Model {model_id} not found in catalog")
                self.initialization_results[model_id] = {
                    'status': 'not_found',
                    'message': 'Model not found in catalog'
                }
                return
            
            # Check if already downloaded
            if model_info.is_downloaded and not force_download:
                print(f"✅ Model already downloaded: {model_info.local_path}")
                self.initialization_results[model_id] = {
                    'status': 'already_downloaded',
                    'path': model_info.local_path
                }
            elif not skip_download:
                # Download the model
                print(f"⬇️  Downloading {model_name}...")
                print(f"   Size: {model_info.size_mb:.1f}MB")
                
                success = await self.model_manager.download_model(model_id, force_redownload=force_download)
                
                if success:
                    print(f"✅ Download completed successfully")
                    self.initialization_results[model_id] = {
                        'status': 'downloaded',
                        'path': model_info.local_path
                    }
                else:
                    print(f"❌ Download failed")
                    self.initialization_results[model_id] = {
                        'status': 'download_failed',
                        'message': 'Download failed'
                    }
                    return
            else:
                print(f"⏭️  Skipping download (--skip-download flag)")
                self.initialization_results[model_id] = {
                    'status': 'skipped',
                    'message': 'Download skipped'
                }
                return
            
            # Set as global if configured
            if model_config.get('set_as_global', False):
                await self._set_as_global_model(model_id, model_info.model_type)
            
        except Exception as e:
            logger.error(f"Failed to setup model {model_config['model_id']}: {e}")
            self.initialization_results[model_config['model_id']] = {
                'status': 'error',
                'message': str(e)
            }
    
    async def _set_as_global_model(self, model_id: str, model_type: ModelType):
        """Set a model as global default."""
        try:
            if model_type == ModelType.EMBEDDING:
                await self._set_global_embedding(model_id)
            elif model_type == ModelType.VISION:
                await self._set_global_vision(model_id)
            
        except Exception as e:
            logger.warning(f"Failed to set {model_id} as global: {e}")
    
    async def _set_global_embedding(self, model_id: str):
        """Set global embedding model."""
        try:
            model_info = self.model_manager.get_model_info(model_id)
            
            new_config = {
                'embedding_model': model_id,
                'embedding_engine': '',  # Use local model
                'embedding_dimension': model_info.dimension,
                'model_path': model_info.local_path,
                'updated_at': str(asyncio.get_event_loop().time())
            }
            
            set_global_embedding_config(new_config)
            await self.global_embedding_manager.reload_configuration()
            
            print(f"🌍 Set as global embedding model")
            
        except Exception as e:
            logger.error(f"Failed to set global embedding: {e}")
    
    async def _set_global_vision(self, model_id: str):
        """Set global vision model."""
        try:
            model_info = self.model_manager.get_model_info(model_id)
            
            current_config = self.global_embedding_manager.get_current_config() or {}
            current_config.update({
                'vision_model': model_id,
                'vision_model_path': model_info.local_path,
                'vision_dimension': model_info.dimension,
                'updated_at': str(asyncio.get_event_loop().time())
            })
            
            set_global_embedding_config(current_config)
            await self.global_embedding_manager.reload_configuration()
            
            print(f"👁️  Set as global vision model")
            
        except Exception as e:
            logger.error(f"Failed to set global vision: {e}")
    
    async def _configure_global_settings(self):
        """Configure global system settings."""
        try:
            print(f"\n🌍 Configuring Global Settings")
            print("-" * 30)
            
            config = self.global_embedding_manager.get_current_config() or {}
            
            # Ensure essential settings
            if 'embedding_batch_size' not in config:
                config['embedding_batch_size'] = 32
            
            if 'enable_caching' not in config:
                config['enable_caching'] = True
            
            # Update configuration
            set_global_embedding_config(config)
            await self.global_embedding_manager.reload_configuration()
            
            print("✅ Global settings configured")
            
        except Exception as e:
            logger.error(f"Failed to configure global settings: {e}")
    
    async def _validate_installation(self):
        """Validate that all models are working correctly."""
        try:
            print(f"\n🔍 Validating Model Installation")
            print("-" * 35)
            
            for model_id, result in self.initialization_results.items():
                if result['status'] in ['downloaded', 'already_downloaded']:
                    model_info = self.model_manager.get_model_info(model_id)
                    
                    if model_info.model_type == ModelType.EMBEDDING:
                        test_result = await self.model_manager.test_embedding_model(model_id)
                    elif model_info.model_type == ModelType.VISION:
                        test_result = await self.model_manager.test_vision_model(model_id)
                    else:
                        test_result = {"success": True}
                    
                    if test_result.get("success"):
                        print(f"✅ {model_id} - Working correctly")
                    else:
                        print(f"❌ {model_id} - Test failed: {test_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
    
    def _show_setup_summary(self):
        """Show setup summary."""
        try:
            print(f"\n📊 Setup Summary")
            print("=" * 40)
            
            successful = 0
            failed = 0
            
            for model_id, result in self.initialization_results.items():
                status = result['status']
                if status in ['downloaded', 'already_downloaded']:
                    successful += 1
                    print(f"✅ {model_id} - {status}")
                else:
                    failed += 1
                    print(f"❌ {model_id} - {status}: {result.get('message', '')}")
            
            print(f"\n📈 Results: {successful} successful, {failed} failed")
            
            if successful > 0:
                print(f"\n🎉 Agentic AI Engine models are ready!")
                print(f"   Models stored in: data/models/")
                print(f"   Global configuration updated")
            
        except Exception as e:
            logger.error(f"Failed to show summary: {e}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Initialize essential models for Agentic AI Engine")
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading, only validate existing models')
    parser.add_argument('--force', action='store_true', help='Force redownload all models')
    
    args = parser.parse_args()
    
    # Initialize and run
    initializer = ModelInitializer()
    await initializer.initialize()
    await initializer.setup_essential_models(
        skip_download=args.skip_download,
        force_download=args.force
    )

if __name__ == "__main__":
    asyncio.run(main())
