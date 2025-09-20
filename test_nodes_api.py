#!/usr/bin/env python3
"""
Test script for the new nodes API
"""

import requests
import json
import time

def test_nodes_api():
    """Test the nodes API endpoints."""
    base_url = "http://localhost:8888/api/v1"
    
    print("🧪 Testing Advanced Node System API")
    print("=" * 50)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(5)
    
    try:
        # Test 1: List all nodes
        print("\n📋 Test 1: List all nodes")
        response = requests.get(f"{base_url}/nodes/")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            nodes = response.json()
            print(f"✅ Found {len(nodes)} registered nodes:")
            for node in nodes:
                print(f"  - {node['node_type']}: {node['name']} ({node['category']})")
                print(f"    Inputs: {len(node['input_ports'])}, Outputs: {len(node['output_ports'])}")
        else:
            print(f"❌ Error: {response.text[:200]}")
            return
        
        # Test 2: Get specific node definition
        if nodes:
            print(f"\n🔍 Test 2: Get specific node definition")
            first_node = nodes[0]['node_type']
            response = requests.get(f"{base_url}/nodes/{first_node}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                node_def = response.json()
                print(f"✅ Node definition for {first_node}:")
                print(f"  Name: {node_def['name']}")
                print(f"  Description: {node_def['description']}")
                print(f"  Category: {node_def['category']}")
                print(f"  Icon: {node_def['icon']}")
            else:
                print(f"❌ Error: {response.text[:200]}")
        
        # Test 3: List categories
        print(f"\n📂 Test 3: List node categories")
        response = requests.get(f"{base_url}/nodes/categories/")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            categories = response.json()
            print(f"✅ Available categories: {', '.join(categories)}")
        else:
            print(f"❌ Error: {response.text[:200]}")
        
        # Test 4: List port types
        print(f"\n🔌 Test 4: List port types")
        response = requests.get(f"{base_url}/nodes/port-types/")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            port_types = response.json()
            print(f"✅ Available port types: {', '.join(port_types)}")
        else:
            print(f"❌ Error: {response.text[:200]}")
        
        # Test 5: Bootstrap status
        print(f"\n🚀 Test 5: Bootstrap status")
        response = requests.get(f"{base_url}/nodes/bootstrap/status")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Bootstrap status:")
            print(f"  Initialized: {status['initialized']}")
            print(f"  Total nodes: {status['total_nodes']}")
        else:
            print(f"❌ Error: {response.text[:200]}")
        
        # Test 6: Execute a TIMER node
        print(f"\n⏰ Test 6: Execute TIMER node")
        timer_request = {
            "node_type": "TIMER",
            "node_config": {
                "timer_type": "delay",
                "duration_seconds": 0.5
            },
            "execution_context": {
                "node_id": "test_timer_1",
                "workflow_id": "test_workflow",
                "execution_id": "test_execution"
            }
        }
        
        response = requests.post(f"{base_url}/nodes/execute", json=timer_request)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Timer execution result:")
            print(f"  Success: {result['success']}")
            if result['success'] and result['data']:
                print(f"  Timer type: {result['data']['timer_type']}")
                print(f"  Execution time: {result['data']['execution_time_seconds']:.3f}s")
        else:
            print(f"❌ Error: {response.text[:200]}")
        
        print(f"\n🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the backend server.")
        print("   Make sure the backend is running on http://localhost:8888")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_nodes_api()
