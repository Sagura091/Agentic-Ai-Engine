"""
Example: Model Management in RAG System

This example demonstrates how to:
1. Use the 3 default models (automatic)
2. List available models
3. Download additional models
4. Switch between models

The RAG system automatically downloads 3 default models on first run:
- all-MiniLM-L6-v2 (embedding)
- bge-reranker-base (reranking)
- clip-ViT-B-32 (vision)

Users can download and switch to other models at any time.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.core.unified_rag_system import UnifiedRAGSystem, RAGConfig


async def main():
    """Demonstrate model management."""
    
    print("\n" + "=" * 80)
    print("RAG SYSTEM - MODEL MANAGEMENT EXAMPLE")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # STEP 1: Initialize RAG System (3 default models auto-download)
    # ========================================================================
    print("Step 1: Initializing RAG system...")
    print("(Default models will be downloaded automatically if not present)\n")
    
    config = RAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_db_type="chroma",
        chroma_persist_directory="./data/chroma_db"
    )
    
    rag_system = UnifiedRAGSystem(config)
    await rag_system.initialize()
    
    print("✅ RAG system initialized\n")
    
    # ========================================================================
    # STEP 2: Check current models
    # ========================================================================
    print("Step 2: Checking current models...")
    current_models = rag_system.get_current_models()
    
    print("\nCurrent Models:")
    for model_type, model_id in current_models.items():
        print(f"  {model_type.capitalize()}: {model_id}")
    print()
    
    # ========================================================================
    # STEP 3: List available embedding models
    # ========================================================================
    print("Step 3: Listing available embedding models...")
    embedding_models = rag_system.list_available_models('embedding')
    
    print("\nAvailable Embedding Models:")
    for model in embedding_models:
        print(f"  • {model['short_name']}")
        print(f"    ID: {model['id']}")
        print(f"    Size: {model['size']}")
        print(f"    Description: {model['description']}")
        print(f"    Dimension: {model['dimension']}")
        print()
    
    # ========================================================================
    # STEP 4: Download a different embedding model
    # ========================================================================
    print("Step 4: Downloading a larger embedding model...")
    print("(This will be skipped if model already exists)\n")
    
    # Download all-mpnet-base-v2 (larger, more accurate)
    result = await rag_system.download_model('all-mpnet-base-v2', 'embedding')
    
    if result['success']:
        print(f"✅ {result['message']}")
        print(f"   Location: {result['location']}")
        print(f"   Size: {result['size_mb']:.1f} MB\n")
    else:
        print(f"❌ Download failed: {result['error']}\n")
    
    # ========================================================================
    # STEP 5: Switch to the new embedding model
    # ========================================================================
    print("Step 5: Switching to the new embedding model...")
    
    switch_result = await rag_system.switch_embedding_model('all-mpnet-base-v2')
    
    if switch_result['success']:
        print(f"✅ {switch_result['message']}")
        print(f"   Old model: {switch_result['old_model']}")
        print(f"   New model: {switch_result['new_model']}\n")
    else:
        print(f"❌ Switch failed: {switch_result['error']}\n")
    
    # ========================================================================
    # STEP 6: Verify the switch
    # ========================================================================
    print("Step 6: Verifying model switch...")
    current_models = rag_system.get_current_models()
    
    print("\nCurrent Models (after switch):")
    for model_type, model_id in current_models.items():
        print(f"  {model_type.capitalize()}: {model_id}")
    print()
    
    # ========================================================================
    # STEP 7: List all available models (all types)
    # ========================================================================
    print("Step 7: Listing all available models...")
    all_models = rag_system.list_available_models()
    
    print(f"\nTotal Available Models: {len(all_models)}")
    
    # Group by type
    by_type = {}
    for model in all_models:
        model_type = model['type']
        if model_type not in by_type:
            by_type[model_type] = []
        by_type[model_type].append(model)
    
    for model_type, models in by_type.items():
        print(f"\n{model_type.upper()} Models ({len(models)}):")
        for model in models:
            priority_icon = "⭐" if model['priority'] == 'required' else "📌" if model['priority'] == 'recommended' else "💡"
            print(f"  {priority_icon} {model['short_name']} - {model['size']}")
    
    print()
    
    # ========================================================================
    # STEP 8: Download and list reranking models
    # ========================================================================
    print("Step 8: Working with reranking models...")
    reranking_models = rag_system.list_available_models('reranking')
    
    print("\nAvailable Reranking Models:")
    for model in reranking_models:
        print(f"  • {model['short_name']} - {model['size']}")
    
    # Download a different reranker
    print("\nDownloading ms-marco-MiniLM-L-12-v2 reranker...")
    result = await rag_system.download_model('ms-marco-MiniLM-L-12-v2', 'reranking')
    
    if result['success']:
        print(f"✅ {result['message']}\n")
    else:
        print(f"❌ {result['error']}\n")
    
    # ========================================================================
    # STEP 9: Download vision model
    # ========================================================================
    print("Step 9: Working with vision models...")
    vision_models = rag_system.list_available_models('vision')
    
    print("\nAvailable Vision Models:")
    for model in vision_models:
        print(f"  • {model['short_name']} - {model['size']}")
    
    # The default clip-ViT-B-32 should already be downloaded
    print("\nDefault vision model (clip-ViT-B-32) is already available.\n")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✅ Model Management Features Demonstrated:")
    print("   1. Automatic download of 3 default models")
    print("   2. List available models by type")
    print("   3. Download additional models")
    print("   4. Switch between models")
    print("   5. Check current active models")
    print("\n💡 Key Points:")
    print("   • Default models are downloaded automatically (no CLI)")
    print("   • Users can download any model from the catalog")
    print("   • Models are stored in data/models/ (centralized)")
    print("   • Existing models are reused (no duplicates)")
    print("   • Switch models at any time")
    print("\n📁 Model Storage:")
    print("   data/models/")
    print("   ├── embedding/")
    print("   │   ├── all_MiniLM_L6_v2/")
    print("   │   └── all_mpnet_base_v2/")
    print("   ├── reranking/")
    print("   │   ├── bge_reranker_base/")
    print("   │   └── ms_marco_MiniLM_L_12_v2/")
    print("   └── vision/")
    print("       └── clip_ViT_B_32/")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

