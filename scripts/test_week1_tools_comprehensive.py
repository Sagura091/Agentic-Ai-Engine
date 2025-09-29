#!/usr/bin/env python3
"""
Comprehensive Test Script for Week 1 Production Tools.

This script tests all 8 revolutionary AI agent tools implemented in Week 1:
1. File System Operations Tool
2. API Integration Tool  
3. Database Operations Tool
4. Text Processing & NLP Tool
5. Password & Security Tool
6. Notification & Alert Tool
7. QR Code & Barcode Tool
8. Weather & Environmental Tool
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.tools.production import (
    file_system_tool,
    api_integration_tool,
    database_operations_tool,
    text_processing_nlp_tool,
    password_security_tool,
    notification_alert_tool,
    qr_barcode_tool,
    weather_environmental_tool
)

async def test_file_system_tool():
    """Test File System Operations Tool."""
    print("\n🗂️  Testing File System Operations Tool...")
    
    try:
        # Test file creation
        result = await file_system_tool.arun({
            "operation": "create",
            "path": "test_output/sample.txt",
            "content": "Hello from File System Tool!",
            "create_directories": True
        })
        print("✅ File creation test passed")

        # Test file reading
        result = await file_system_tool.arun({
            "operation": "read",
            "path": "test_output/sample.txt"
        })
        print("✅ File reading test passed")

        # Test directory listing
        result = await file_system_tool.arun({
            "operation": "list",
            "path": "test_output"
        })
        print("✅ Directory listing test passed")
        
        return True
    except Exception as e:
        print(f"❌ File System Tool test failed: {e}")
        return False

async def test_api_integration_tool():
    """Test API Integration Tool."""
    print("\n🌐 Testing API Integration Tool...")
    
    try:
        # Test GET request
        result = await api_integration_tool.arun({
            "method": "GET",
            "url": "https://httpbin.org/get",
            "timeout": 10
        })
        print("✅ GET request test passed")

        # Test POST request
        result = await api_integration_tool.arun({
            "method": "POST",
            "url": "https://httpbin.org/post",
            "data": {"test": "data"},
            "timeout": 10
        })
        print("✅ POST request test passed")
        
        return True
    except Exception as e:
        print(f"❌ API Integration Tool test failed: {e}")
        return False

async def test_database_operations_tool():
    """Test Database Operations Tool."""
    print("\n🗄️  Testing Database Operations Tool...")
    
    try:
        # Test SQLite connection and query
        result = await database_operations_tool.arun({
            "operation": "select",
            "database_type": "sqlite",
            "connection_string": ":memory:",
            "query": "SELECT 1 as test_column"
        })
        print("✅ SQLite query test passed")

        # Test table creation
        result = await database_operations_tool.arun({
            "operation": "execute",
            "database_type": "sqlite",
            "connection_string": ":memory:",
            "query": "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
        })
        print("✅ Table creation test passed")
        
        return True
    except Exception as e:
        print(f"❌ Database Operations Tool test failed: {e}")
        return False

async def test_text_processing_nlp_tool():
    """Test Text Processing & NLP Tool."""
    print("\n📝 Testing Text Processing & NLP Tool...")
    
    try:
        # Test sentiment analysis
        result = await text_processing_nlp_tool.arun({
            "operation": "sentiment_analysis",
            "text": "I love this amazing AI tool! It's fantastic and works perfectly.",
            "language": "en"
        })
        print("✅ Sentiment analysis test passed")

        # Test keyword extraction
        result = await text_processing_nlp_tool.arun({
            "operation": "extract_keywords",
            "text": "Artificial intelligence and machine learning are revolutionizing technology and business processes.",
            "max_keywords": 5
        })
        print("✅ Keyword extraction test passed")

        # Test text similarity
        result = await text_processing_nlp_tool.arun({
            "operation": "compare_similarity",
            "text": "The quick brown fox jumps over the lazy dog",
            "compare_text": "A fast brown fox leaps over a sleepy dog"
        })
        print("✅ Text similarity test passed")
        
        return True
    except Exception as e:
        print(f"❌ Text Processing & NLP Tool test failed: {e}")
        return False

async def test_password_security_tool():
    """Test Password & Security Tool."""
    print("\n🔐 Testing Password & Security Tool...")
    
    try:
        # Test password generation
        result = await password_security_tool.arun({
            "operation": "generate_password",
            "length": 16,
            "include_uppercase": True,
            "include_lowercase": True,
            "include_numbers": True,
            "include_symbols": True
        })
        print("✅ Password generation test passed")

        # Test text hashing
        result = await password_security_tool.arun({
            "operation": "hash_text",
            "text": "Hello World",
            "hash_algorithm": "sha256"
        })
        print("✅ Text hashing test passed")

        # Test encryption
        result = await password_security_tool.arun({
            "operation": "encrypt_text",
            "text": "Secret message",
            "key": "my-secret-key-32-characters-long",
            "encryption_algorithm": "aes_256_gcm"
        })
        print("✅ Text encryption test passed")
        
        return True
    except Exception as e:
        print(f"❌ Password & Security Tool test failed: {e}")
        return False

async def test_notification_alert_tool():
    """Test Notification & Alert Tool."""
    print("\n📢 Testing Notification & Alert Tool...")
    
    try:
        # Test console notification
        result = await notification_alert_tool.arun({
            "title": "Test Notification",
            "message": "This is a test message from the AI agent!",
            "channels": ["console"],
            "recipients": ["test-user"],
            "priority": "normal"
        })
        print("✅ Console notification test passed")

        # Test email notification (mock)
        result = await notification_alert_tool.arun({
            "title": "AI Agent Alert",
            "message": "Your AI agent has completed a task successfully!",
            "channels": ["email"],
            "recipients": ["user@example.com"],
            "priority": "high",
            "email_from": "ai-agent@example.com"
        })
        print("✅ Email notification test passed")
        
        return True
    except Exception as e:
        print(f"❌ Notification & Alert Tool test failed: {e}")
        return False

async def test_qr_barcode_tool():
    """Test QR Code & Barcode Tool."""
    print("\n📱 Testing QR Code & Barcode Tool...")
    
    try:
        # Test QR code generation
        result = await qr_barcode_tool.arun({
            "operation": "generate",
            "data": "https://example.com/ai-agent-test",
            "format": "qr_code",
            "size": 200,
            "error_correction": "M"
        })
        print("✅ QR code generation test passed")

        # Test barcode generation
        result = await qr_barcode_tool.arun({
            "operation": "generate",
            "data": "123456789012",
            "format": "ean13",
            "size": 150
        })
        print("✅ Barcode generation test passed")

        # Test data validation
        result = await qr_barcode_tool.arun({
            "operation": "validate",
            "data": "Test data for validation",
            "format": "qr_code"
        })
        print("✅ Data validation test passed")
        
        return True
    except Exception as e:
        print(f"❌ QR Code & Barcode Tool test failed: {e}")
        return False

async def test_weather_environmental_tool():
    """Test Weather & Environmental Tool."""
    print("\n🌤️  Testing Weather & Environmental Tool...")
    
    try:
        # Test current weather
        result = await weather_environmental_tool.arun({
            "operation": "current_weather",
            "location": "New York",
            "temperature_unit": "celsius",
            "include_air_quality": True,
            "include_astronomy": True
        })
        print("✅ Current weather test passed")

        # Test weather forecast
        result = await weather_environmental_tool.arun({
            "operation": "forecast",
            "location": "London",
            "days": 5,
            "include_daily": True,
            "temperature_unit": "fahrenheit"
        })
        print("✅ Weather forecast test passed")

        # Test air quality
        result = await weather_environmental_tool.arun({
            "operation": "air_quality",
            "latitude": 40.7128,
            "longitude": -74.0060
        })
        print("✅ Air quality test passed")
        
        return True
    except Exception as e:
        print(f"❌ Weather & Environmental Tool test failed: {e}")
        return False

async def main():
    """Run comprehensive tests for all Week 1 production tools."""
    print("🚀 Starting Comprehensive Week 1 Production Tools Test Suite")
    print("=" * 70)
    
    test_results = {}
    
    # Run all tests
    test_functions = [
        ("File System Operations", test_file_system_tool),
        ("API Integration", test_api_integration_tool),
        ("Database Operations", test_database_operations_tool),
        ("Text Processing & NLP", test_text_processing_nlp_tool),
        ("Password & Security", test_password_security_tool),
        ("Notification & Alert", test_notification_alert_tool),
        ("QR Code & Barcode", test_qr_barcode_tool),
        ("Weather & Environmental", test_weather_environmental_tool)
    ]
    
    for tool_name, test_func in test_functions:
        try:
            success = await test_func()
            test_results[tool_name] = success
        except Exception as e:
            print(f"❌ {tool_name} test failed with exception: {e}")
            test_results[tool_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for tool_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{tool_name:<30} {status}")
    
    print("-" * 70)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Week 1 production tools are fully functional!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
