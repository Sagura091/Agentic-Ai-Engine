#!/usr/bin/env python3
"""
Performance test script to verify the frontend performance improvements.

This script tests the API endpoints that are called during dashboard loading
to ensure they respond quickly after the optimizations.
"""

import asyncio
import time
import aiohttp
import json
from typing import List, Dict, Any

BASE_URL = "http://localhost:8888/api/v1"

async def test_endpoint_performance(session: aiohttp.ClientSession, endpoint: str, name: str) -> Dict[str, Any]:
    """Test a single endpoint and measure response time."""
    start_time = time.time()
    
    try:
        async with session.get(f"{BASE_URL}{endpoint}") as response:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            if response.status == 200:
                data = await response.json()
                return {
                    "endpoint": endpoint,
                    "name": name,
                    "status": "success",
                    "duration_ms": duration_ms,
                    "status_code": response.status,
                    "cache_control": response.headers.get("Cache-Control", "none"),
                    "etag": response.headers.get("ETag", "none")
                }
            else:
                return {
                    "endpoint": endpoint,
                    "name": name,
                    "status": "error",
                    "duration_ms": duration_ms,
                    "status_code": response.status,
                    "error": f"HTTP {response.status}"
                }
                
    except Exception as e:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        return {
            "endpoint": endpoint,
            "name": name,
            "status": "error",
            "duration_ms": duration_ms,
            "error": str(e)
        }

async def test_parallel_loading():
    """Test parallel loading of dashboard endpoints (simulating frontend behavior)."""
    endpoints = [
        ("/agents", "Agents List"),
        ("/workflows", "Workflows List"),
        ("/monitoring/system", "System Metrics")
    ]
    
    print("🚀 Testing Performance Improvements")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Test sequential loading (old behavior)
        print("\n📊 Sequential Loading (Old Behavior):")
        sequential_start = time.time()
        sequential_results = []
        
        for endpoint, name in endpoints:
            result = await test_endpoint_performance(session, endpoint, name)
            sequential_results.append(result)
            print(f"  {name}: {result['duration_ms']:.1f}ms ({result['status']})")
        
        sequential_total = (time.time() - sequential_start) * 1000
        print(f"  Total Sequential Time: {sequential_total:.1f}ms")
        
        # Test parallel loading (new behavior)
        print("\n⚡ Parallel Loading (New Behavior):")
        parallel_start = time.time()
        
        tasks = [
            test_endpoint_performance(session, endpoint, name)
            for endpoint, name in endpoints
        ]
        
        parallel_results = await asyncio.gather(*tasks)
        parallel_total = (time.time() - parallel_start) * 1000
        
        for result in parallel_results:
            print(f"  {result['name']}: {result['duration_ms']:.1f}ms ({result['status']})")
            if 'cache_control' in result:
                print(f"    Cache-Control: {result['cache_control']}")
        
        print(f"  Total Parallel Time: {parallel_total:.1f}ms")
        
        # Calculate improvement
        improvement = ((sequential_total - parallel_total) / sequential_total) * 100
        print(f"\n🎯 Performance Improvement: {improvement:.1f}% faster")
        print(f"   Time saved: {sequential_total - parallel_total:.1f}ms")
        
        # Test caching behavior
        print("\n🗄️  Testing Caching Behavior:")
        cache_start = time.time()
        cache_result = await test_endpoint_performance(session, "/monitoring/system", "System Metrics (Cached)")
        cache_total = (time.time() - cache_start) * 1000
        
        print(f"  Cached Request: {cache_result['duration_ms']:.1f}ms")
        if 'cache_control' in cache_result:
            print(f"  Cache Headers: {cache_result['cache_control']}")
        
        # Summary
        print("\n📈 Summary:")
        print(f"  - Sequential loading: {sequential_total:.1f}ms")
        print(f"  - Parallel loading: {parallel_total:.1f}ms")
        print(f"  - Performance gain: {improvement:.1f}%")
        print(f"  - Caching enabled: {'Yes' if any('max-age' in r.get('cache_control', '') for r in parallel_results) else 'No'}")
        
        # Performance targets
        if parallel_total < 1000:  # Less than 1 second
            print("  ✅ Performance target met (< 1 second)")
        else:
            print("  ❌ Performance target not met (> 1 second)")

if __name__ == "__main__":
    print("Performance Test for Frontend Improvements")
    print("Make sure the backend is running on localhost:8888")
    print()
    
    try:
        asyncio.run(test_parallel_loading())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
