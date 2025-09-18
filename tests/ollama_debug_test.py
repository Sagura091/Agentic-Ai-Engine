#!/usr/bin/env python3
"""
Ollama Debug Test - Diagnose Ollama connection issues.

This script tests the Ollama connection at various levels to identify
where the connection is failing.
"""

import asyncio
import sys
import os
import httpx
import requests
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.llm.providers import OllamaProvider
    from app.llm.models import ProviderCredentials, ProviderType, LLMConfig
    from app.services.llm_service import get_llm_service, initialize_llm_service
    from app.config.settings import get_settings
    import structlog
    
    logger = structlog.get_logger(__name__)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_direct_ollama_connection():
    """Test direct connection to Ollama API."""
    print("🔧 Testing direct Ollama connection...")
    
    try:
        # Test with requests (synchronous)
        print("   Testing with requests library...")
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            print(f"   ✅ Requests successful: {len(models)} models found")
            print(f"   Models: {models}")
        else:
            print(f"   ❌ Requests failed: {response.status_code}")
            return False
        
        # Test with httpx (async)
        print("   Testing with httpx library...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                print(f"   ✅ HTTPX successful: {len(models)} models found")
                print(f"   Models: {models}")
                return True
            else:
                print(f"   ❌ HTTPX failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"   ❌ Direct connection failed: {e}")
        return False


async def test_ollama_provider():
    """Test OllamaProvider class directly."""
    print("\n🔧 Testing OllamaProvider class...")
    
    try:
        # Create provider with default credentials
        credentials = ProviderCredentials(
            provider=ProviderType.OLLAMA,
            base_url="http://localhost:11434"
        )
        
        provider = OllamaProvider(credentials)
        
        # Initialize provider
        print("   Initializing provider...")
        init_success = await provider.initialize()
        if not init_success:
            print("   ❌ Provider initialization failed")
            return False
        print("   ✅ Provider initialized")
        
        # Get available models
        print("   Getting available models...")
        models = await provider.get_available_models()
        print(f"   ✅ Found {len(models)} models:")
        for model in models:
            print(f"      - {model.id} ({model.name})")
        
        # Test model validation
        test_models = ["phi4:14b", "gpt-oss:120b", "nonexistent:model"]
        print("   Testing model validation...")
        for model_id in test_models:
            is_valid = await provider.validate_model(model_id)
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"      {model_id}: {status}")
        
        return len(models) > 0
        
    except Exception as e:
        print(f"   ❌ OllamaProvider test failed: {e}")
        logger.error("OllamaProvider test failed", error=str(e))
        return False


async def test_llm_service():
    """Test LLM service with Ollama."""
    print("\n🔧 Testing LLM Service...")
    
    try:
        # Initialize LLM service
        print("   Initializing LLM service...")
        service = await initialize_llm_service()
        print("   ✅ LLM service initialized")
        
        # Test provider connection
        print("   Testing Ollama provider connection...")
        ollama_status = await service.test_provider_connection("ollama")
        print(f"   Ollama status: {ollama_status}")
        
        if not ollama_status.get("is_available", False):
            print("   ❌ Ollama provider not available")
            return False
        
        # Get available models
        print("   Getting models from Ollama...")
        ollama_models = await service.get_models_by_provider("ollama")
        print(f"   ✅ Found {len(ollama_models)} models:")
        for model in ollama_models[:5]:  # Show first 5
            print(f"      - {model['id']}")
        
        # Test model validation
        test_models = ["phi4:14b", "gpt-oss:120b"]
        print("   Testing model validation through service...")
        for model_id in test_models:
            is_valid = await service.validate_model_config("ollama", model_id)
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"      {model_id}: {status}")
        
        return len(ollama_models) > 0
        
    except Exception as e:
        print(f"   ❌ LLM service test failed: {e}")
        logger.error("LLM service test failed", error=str(e))
        return False


async def test_llm_instance_creation():
    """Test creating actual LLM instances."""
    print("\n🔧 Testing LLM instance creation...")
    
    try:
        service = get_llm_service()
        
        # Test with phi4:14b
        test_configs = [
            {
                "provider": "ollama",
                "model_id": "phi4:14b",
                "temperature": 0.7,
                "max_tokens": 100
            },
            {
                "provider": "ollama", 
                "model_id": "gpt-oss:120b",
                "temperature": 0.7,
                "max_tokens": 100
            }
        ]
        
        for config in test_configs:
            print(f"   Testing {config['model_id']}...")
            try:
                llm = await service.create_llm_instance(config)
                print(f"   ✅ {config['model_id']} instance created successfully")
                print(f"      Type: {type(llm)}")
                
                # Test a simple generation
                print(f"   Testing generation with {config['model_id']}...")
                try:
                    response = await llm.agenerate(["Hello, can you respond to this test?"])
                    if response and response.generations:
                        text = response.generations[0][0].text
                        print(f"   ✅ Generation successful: {text[:100]}...")
                    else:
                        print("   ❌ Generation returned empty response")
                except Exception as gen_e:
                    print(f"   ❌ Generation failed: {gen_e}")
                
            except Exception as e:
                print(f"   ❌ {config['model_id']} instance creation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM instance creation test failed: {e}")
        logger.error("LLM instance creation test failed", error=str(e))
        return False


async def test_agent_creation_with_working_model():
    """Test agent creation with a working model."""
    print("\n🤖 Testing agent creation with working model...")
    
    try:
        from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType
        
        # First, find a working model
        service = get_llm_service()
        ollama_models = await service.get_models_by_provider("ollama")
        
        if not ollama_models:
            print("   ❌ No Ollama models available")
            return False
        
        # Try each model until we find one that works
        for model_info in ollama_models:
            model_id = model_info["id"]
            print(f"   Trying to create agent with {model_id}...")
            
            try:
                agent_id = await enhanced_orchestrator.create_agent_unlimited(
                    agent_type=AgentType.AUTONOMOUS,
                    name=f"TestAgent_{model_id.replace(':', '_')}",
                    description=f"Test agent using {model_id}",
                    config={
                        "model": model_id,
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "enable_tool_creation": True,
                        "enable_learning": True,
                        "autonomy_level": "high"
                    },
                    tools=[]
                )
                
                print(f"   ✅ Agent created successfully: {agent_id}")
                print(f"   Model: {model_id}")
                
                # Test a simple task
                print(f"   Testing simple task with agent...")
                result = await enhanced_orchestrator.execute_agent_task(
                    agent_id=agent_id,
                    task="Hello! Can you tell me that you're working correctly? Just respond with 'I am working correctly' if you can understand this.",
                    context={"test_mode": True}
                )
                
                if result and result.get('success', False):
                    response = result.get('response', '')
                    print(f"   ✅ Agent task successful: {response[:100]}...")
                    return True, agent_id, model_id
                else:
                    print(f"   ❌ Agent task failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"   ❌ Agent creation with {model_id} failed: {e}")
                continue
        
        print("   ❌ No working model found for agent creation")
        return False, None, None
        
    except Exception as e:
        print(f"   ❌ Agent creation test failed: {e}")
        logger.error("Agent creation test failed", error=str(e))
        return False, None, None


async def main():
    """Main diagnostic execution."""
    print("🚀 OLLAMA CONNECTION DIAGNOSTIC")
    print("=" * 60)
    print("Diagnosing Ollama connection issues step by step")
    print("=" * 60)
    
    tests = [
        ("Direct Ollama Connection", test_direct_ollama_connection),
        ("OllamaProvider Class", test_ollama_provider),
        ("LLM Service", test_llm_service),
        ("LLM Instance Creation", test_llm_instance_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{len(results) + 1}. {test_name}")
        try:
            success = await test_func()
            results.append((test_name, success))
            if not success:
                print(f"   ⚠️  {test_name} failed - stopping here to investigate")
                break
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
            break
    
    # If all basic tests pass, try agent creation
    if all(success for _, success in results):
        print(f"\n{len(results) + 1}. Agent Creation with Working Model")
        try:
            success, agent_id, model_id = await test_agent_creation_with_working_model()
            results.append(("Agent Creation", success))
            
            if success:
                print(f"\n🎉 SUCCESS! Agent created and working!")
                print(f"   Agent ID: {agent_id}")
                print(f"   Working Model: {model_id}")
                print(f"\n   You can now use this model for real agent testing:")
                print(f"   Model: {model_id}")
        except Exception as e:
            print(f"   ❌ Agent creation failed: {e}")
            results.append(("Agent Creation", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC RESULTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"\n🎯 Overall Score: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your Ollama connection is working perfectly!")
        print("   You can now create real agents and test them!")
    else:
        print(f"\n⚠️  ISSUE FOUND at step {passed_tests + 1}")
        print("   Check the error messages above to fix the connection issue")


if __name__ == "__main__":
    asyncio.run(main())
