#!/usr/bin/env python3
"""Minimal test to isolate the import issue."""

import sys

def test_minimal():
    try:
        print("Testing basic imports...")
        
        # Test basic imports first
        from app.rag.core.unified_rag_system import Document, KnowledgeQuery, KnowledgeResult
        print("✅ Basic classes OK")
        
        from app.rag.core.unified_rag_system import UnifiedRAGConfig
        print("✅ Config class OK")
        
        from app.rag.core.unified_rag_system import UnifiedRAGSystem
        print("✅ UnifiedRAGSystem OK")
        
        print("🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal()
    sys.exit(0 if success else 1)
