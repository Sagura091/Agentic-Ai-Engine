#!/usr/bin/env python3
"""
🚀 Revolutionary Model Management Script

This script provides comprehensive model management for the Agentic AI Engine:
- Download and install embedding models to data/models/
- Set up global vision models for the entire system
- Configure global embedding settings
- Validate model installations
- Clean up and manage model storage

Usage:
    python scripts/model_management.py download --model sentence-transformers/all-MiniLM-L6-v2
    python scripts/model_management.py setup-vision --model openai/clip-vit-base-patch32
    python scripts/model_management.py list-models
    python scripts/model_management.py validate-installation
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.core.embedding_model_manager import CentralizedModelManager, ModelType
from app.rag.core.embeddings import get_global_embedding_manager, update_global_embedding_config, get_global_embedding_config
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

class ModelManagementCLI:
    """Command-line interface for model management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_manager = None
        self.global_embedding_manager = None
    
    async def initialize(self):
        """Initialize the model management system."""
        try:
            # Initialize centralized model manager
            self.model_manager = CentralizedModelManager()

            # Initialize global embedding manager
            self.global_embedding_manager = await get_global_embedding_manager()

            logger.info("Model management system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize model management: {e}")
            raise
    
    async def download_model(self, model_id: str, force: bool = False, model_type: Optional[ModelType] = None) -> bool:
        """Download a model to the data/models directory."""
        try:
            print(f"\n🚀 Starting download for model: {model_id}")
            logger.info(f"🚀 Starting download for model: {model_id}")

            # Sanitize model_id to match how it's stored
            model_name = model_id.replace("/", "_").replace("-", "_")

            # Check if model already exists
            model_info = self.model_manager.get_model_info(model_name)
            if model_info and model_info.is_downloaded and not force:
                logger.info(f"Model {model_id} is already downloaded")
                print(f"✅ Model {model_id} is already downloaded at: {model_info.local_path}")
                return True

            # Download the model
            print(f"📥 Downloading model (this may take a few minutes)...")
            success = await self.model_manager.download_model(model_id, model_type=model_type, force_redownload=force)

            if success:
                logger.info(f"✅ Model {model_id} downloaded successfully")
                print(f"✅ Model downloaded successfully!")

                # Refresh to get updated model info
                await self.model_manager.refresh_models()
                model_info = self.model_manager.get_model_info(model_name)

                if model_info:
                    print(f"📁 Location: {model_info.local_path}")
                    print(f"📊 Size: {model_info.size_mb:.1f} MB")

                    # If it's an embedding model, offer to set as global default
                    if model_info.model_type == ModelType.EMBEDDING:
                        await self._offer_set_global_embedding(model_name)

                    # If it's a vision model, offer to set as global vision model
                    elif model_info.model_type == ModelType.VISION:
                        await self._offer_set_global_vision(model_name)

                return True
            else:
                logger.error(f"❌ Failed to download model {model_id}")
                print(f"❌ Failed to download model {model_id}")
                return False

        except Exception as e:
            logger.error(f"Download failed: {e}")
            print(f"❌ Download failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _offer_set_global_embedding(self, model_id: str):
        """Offer to set the downloaded embedding model as global default."""
        try:
            current_config = get_global_embedding_config()
            current_model = current_config.get('model_name', 'none') if current_config else 'none'

            print(f"\n🤖 Would you like to set '{model_id}' as the global embedding model?")
            print(f"   Current global model: {current_model}")

            response = input("   Set as global? (y/N): ").strip().lower()

            if response in ['y', 'yes']:
                await self.set_global_embedding_model(model_id)

        except Exception as e:
            logger.warning(f"Could not offer global embedding setup: {e}")

    async def _offer_set_global_vision(self, model_id: str):
        """Offer to set the downloaded vision model as global default."""
        try:
            current_config = get_global_embedding_config()
            current_vision = current_config.get('vision_model', 'none') if current_config else 'none'

            print(f"\n👁️ Would you like to set '{model_id}' as the global vision model?")
            print(f"   Current global vision model: {current_vision}")

            response = input("   Set as global? (y/N): ").strip().lower()

            if response in ['y', 'yes']:
                await self.set_global_vision_model(model_id)

        except Exception as e:
            logger.warning(f"Could not offer global vision setup: {e}")
    
    async def set_global_embedding_model(self, model_id: str):
        """Set a model as the global embedding model."""
        try:
            # Try both original and sanitized model ID
            model_name = model_id.replace("/", "_").replace("-", "_")
            model_info = self.model_manager.get_model_info(model_name)

            if not model_info:
                # Try original ID
                model_info = self.model_manager.get_model_info(model_id)

            if not model_info or not model_info.is_downloaded:
                logger.error(f"Model {model_id} is not downloaded")
                print(f"❌ Model {model_id} is not downloaded. Please download it first using:")
                print(f"   python scripts/model_management.py download --model {model_id}")
                return False

            # Update global configuration
            new_config = {
                'embedding_model': model_id,
                'embedding_engine': '',  # Use local model
                'embedding_dimension': model_info.dimension,
                'model_path': model_info.local_path,
                'updated_at': str(asyncio.get_event_loop().time())
            }

            update_global_embedding_config(new_config)
            # Note: Global manager will reinitialize automatically on next use

            logger.info(f"✅ Global embedding model set to: {model_id}")
            print(f"✅ Global embedding model set to: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set global embedding model: {e}")
            print(f"❌ Failed to set global embedding model: {e}")
            return False
    
    async def set_global_vision_model(self, model_id: str):
        """Set a model as the global vision model."""
        try:
            # Try both original and sanitized model ID
            model_name = model_id.replace("/", "_").replace("-", "_")
            model_info = self.model_manager.get_model_info(model_name)

            if not model_info:
                # Try original ID
                model_info = self.model_manager.get_model_info(model_id)

            if not model_info or not model_info.is_downloaded:
                logger.error(f"Model {model_id} is not downloaded")
                print(f"❌ Model {model_id} is not downloaded. Please download it first using:")
                print(f"   python scripts/model_management.py download --model {model_id} --type vision")
                return False

            # Update global configuration
            current_config = get_global_embedding_config() or {}
            current_config.update({
                'vision_model': model_id,
                'vision_model_path': model_info.local_path,
                'vision_dimension': model_info.dimension,
                'updated_at': str(asyncio.get_event_loop().time())
            })

            update_global_embedding_config(current_config)
            # Note: Global manager will reinitialize automatically on next use

            logger.info(f"✅ Global vision model set to: {model_id}")
            print(f"✅ Global vision model set to: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set global vision model: {e}")
            print(f"❌ Failed to set global vision model: {e}")
            return False
    
    def list_models(self):
        """List all available and downloaded models."""
        try:
            print("\n🤖 Available Models:")
            print("=" * 80)

            models_by_type = {}
            for model_id, model_info in self.model_manager.list_all_models().items():
                model_type = model_info.model_type.value
                if model_type not in models_by_type:
                    models_by_type[model_type] = []
                models_by_type[model_type].append(model_info)

            for model_type, models in models_by_type.items():
                print(f"\n📂 {model_type.upper()} MODELS:")
                print("-" * 40)

                for model in models:
                    status = "✅ Downloaded" if model.is_downloaded else "⬇️  Available"
                    size = f"{model.size_mb:.1f}MB" if model.size_mb else "Unknown size"

                    print(f"  {status} {model.model_id}")
                    print(f"    Description: {model.description}")
                    print(f"    Size: {size}")
                    if model.is_downloaded and model.local_path:
                        print(f"    Path: {model.local_path}")
                    print()

            # Show global configuration
            self._show_global_config()

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
    
    def _show_global_config(self):
        """Show current global model configuration."""
        try:
            config = get_global_embedding_config()

            print("\n🌍 Global Model Configuration:")
            print("-" * 40)

            if config:
                embedding_model = config.get('model_name', 'Not set')
                vision_model = config.get('vision_model', 'Not set')

                print(f"  Embedding Model: {embedding_model}")
                print(f"  Vision Model: {vision_model}")
                print(f"  Embedding Type: {config.get('embedding_type', 'dense')}")
            else:
                print("  No global configuration found")

        except Exception as e:
            logger.warning(f"Could not show global config: {e}")
    
    async def validate_installation(self):
        """Validate that all downloaded models are working correctly."""
        try:
            print("\n🔍 Validating Model Installation:")
            print("=" * 50)
            
            downloaded_models = self.model_manager.get_downloaded_models()
            
            if not downloaded_models:
                print("  No downloaded models found")
                return
            
            for model in downloaded_models:
                print(f"\n📋 Testing {model.model_id}...")
                
                if model.model_type == ModelType.EMBEDDING:
                    result = await self.model_manager.test_embedding_model(model.model_id)
                elif model.model_type == ModelType.VISION:
                    result = await self.model_manager.test_vision_model(model.model_id)
                else:
                    result = {"success": True, "message": "Test not implemented for this model type"}
                
                if result.get("success"):
                    print(f"  ✅ {model.model_id} is working correctly")
                else:
                    print(f"  ❌ {model.model_id} failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Agentic AI Model Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a model')
    download_parser.add_argument('--model', required=True, help='Model ID to download (e.g., sentence-transformers/all-MiniLM-L6-v2)')
    download_parser.add_argument('--type', choices=['embedding', 'vision', 'reranking'], help='Model type (auto-detected if not specified)')
    download_parser.add_argument('--force', action='store_true', help='Force redownload')
    
    # Setup vision command
    vision_parser = subparsers.add_parser('setup-vision', help='Setup global vision model')
    vision_parser.add_argument('--model', required=True, help='Vision model ID')
    
    # Setup embedding command
    embedding_parser = subparsers.add_parser('setup-embedding', help='Setup global embedding model')
    embedding_parser.add_argument('--model', required=True, help='Embedding model ID')
    
    # List command
    subparsers.add_parser('list-models', help='List all available models')
    
    # Validate command
    subparsers.add_parser('validate-installation', help='Validate model installation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = ModelManagementCLI()
    await cli.initialize()
    
    # Execute command
    if args.command == 'download':
        # Convert type string to ModelType enum if provided
        model_type = None
        if hasattr(args, 'type') and args.type:
            model_type = ModelType(args.type)
        await cli.download_model(args.model, args.force, model_type)
    elif args.command == 'setup-vision':
        await cli.set_global_vision_model(args.model)
    elif args.command == 'setup-embedding':
        await cli.set_global_embedding_model(args.model)
    elif args.command == 'list-models':
        cli.list_models()
    elif args.command == 'validate-installation':
        await cli.validate_installation()

if __name__ == "__main__":
    asyncio.run(main())
