#!/usr/bin/env python3
"""
Test script for workflow API endpoints.

This script tests the workflow creation, retrieval, update, and deletion
functionality to ensure the backend fixes are working correctly.
"""

import asyncio
import json
import sys
from typing import Dict, Any
import httpx
import structlog

# Configure logging
logger = structlog.get_logger(__name__)

# API Configuration
BASE_URL = "http://localhost:8888/api/v1"
TIMEOUT = 30.0

class WorkflowAPITester:
    """Test class for workflow API endpoints."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None
        self.created_workflow_id = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def test_create_workflow(self) -> Dict[str, Any]:
        """Test workflow creation from visual builder."""
        logger.info("Testing workflow creation...")
        
        # Sample workflow data matching frontend structure
        workflow_data = {
            "name": "Test Workflow API",
            "description": "A test workflow created via API",
            "nodes": [
                {
                    "id": "node-1",
                    "type": "agent",
                    "position": {"x": 100, "y": 100},
                    "data": {
                        "label": "Agent Node",
                        "config": {
                            "model": "llama3.2:latest",
                            "system_prompt": "You are a helpful assistant."
                        }
                    }
                },
                {
                    "id": "node-2",
                    "type": "output",
                    "position": {"x": 300, "y": 100},
                    "data": {
                        "label": "Output Node",
                        "config": {}
                    }
                }
            ],
            "edges": [
                {
                    "id": "edge-1",
                    "source": "node-1",
                    "target": "node-2",
                    "type": "default"
                }
            ],
            "status": "draft"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/workflows",
                json=workflow_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.created_workflow_id = result.get("id")
                logger.info("✅ Workflow creation successful", 
                          workflow_id=self.created_workflow_id,
                          name=result.get("name"))
                return result
            else:
                logger.error("❌ Workflow creation failed", 
                           status_code=response.status_code,
                           response=response.text)
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error("❌ Workflow creation exception", error=str(e))
            return {"error": str(e)}
    
    async def test_get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Test workflow retrieval."""
        logger.info("Testing workflow retrieval...", workflow_id=workflow_id)
        
        try:
            response = await self.client.get(f"{self.base_url}/workflows/{workflow_id}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Workflow retrieval successful", 
                          workflow_id=workflow_id,
                          name=result.get("name"))
                return result
            else:
                logger.error("❌ Workflow retrieval failed", 
                           status_code=response.status_code,
                           response=response.text)
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error("❌ Workflow retrieval exception", error=str(e))
            return {"error": str(e)}
    
    async def test_list_workflows(self) -> Dict[str, Any]:
        """Test workflow listing."""
        logger.info("Testing workflow listing...")
        
        try:
            response = await self.client.get(f"{self.base_url}/workflows")
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Workflow listing successful", count=len(result))
                return {"workflows": result, "count": len(result)}
            else:
                logger.error("❌ Workflow listing failed", 
                           status_code=response.status_code,
                           response=response.text)
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error("❌ Workflow listing exception", error=str(e))
            return {"error": str(e)}
    
    async def test_update_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Test workflow update."""
        logger.info("Testing workflow update...", workflow_id=workflow_id)
        
        # Updated workflow data
        updated_data = {
            "name": "Updated Test Workflow API",
            "description": "An updated test workflow",
            "nodes": [
                {
                    "id": "node-1",
                    "type": "agent",
                    "position": {"x": 150, "y": 150},
                    "data": {
                        "label": "Updated Agent Node",
                        "config": {
                            "model": "llama3.2:latest",
                            "system_prompt": "You are an updated helpful assistant."
                        }
                    }
                }
            ],
            "edges": [],
            "status": "active"
        }
        
        try:
            response = await self.client.put(
                f"{self.base_url}/workflows/{workflow_id}",
                json=updated_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Workflow update successful", 
                          workflow_id=workflow_id,
                          name=result.get("name"))
                return result
            else:
                logger.error("❌ Workflow update failed", 
                           status_code=response.status_code,
                           response=response.text)
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error("❌ Workflow update exception", error=str(e))
            return {"error": str(e)}
    
    async def test_delete_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Test workflow deletion."""
        logger.info("Testing workflow deletion...", workflow_id=workflow_id)
        
        try:
            response = await self.client.delete(f"{self.base_url}/workflows/{workflow_id}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Workflow deletion successful", workflow_id=workflow_id)
                return result
            else:
                logger.error("❌ Workflow deletion failed", 
                           status_code=response.status_code,
                           response=response.text)
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error("❌ Workflow deletion exception", error=str(e))
            return {"error": str(e)}
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        logger.info("🚀 Starting workflow API test suite...")
        
        results = {
            "create": None,
            "get": None,
            "list": None,
            "update": None,
            "delete": None,
            "success": False
        }
        
        # Test 1: Create workflow
        create_result = await self.test_create_workflow()
        results["create"] = create_result
        
        if "error" in create_result:
            logger.error("❌ Test suite failed at creation step")
            return results
        
        workflow_id = self.created_workflow_id
        if not workflow_id:
            logger.error("❌ No workflow ID returned from creation")
            return results
        
        # Test 2: Get workflow
        get_result = await self.test_get_workflow(workflow_id)
        results["get"] = get_result
        
        # Test 3: List workflows
        list_result = await self.test_list_workflows()
        results["list"] = list_result
        
        # Test 4: Update workflow
        update_result = await self.test_update_workflow(workflow_id)
        results["update"] = update_result
        
        # Test 5: Delete workflow
        delete_result = await self.test_delete_workflow(workflow_id)
        results["delete"] = delete_result
        
        # Check overall success
        success = all(
            "error" not in result for result in results.values() 
            if result is not None
        )
        results["success"] = success
        
        if success:
            logger.info("🎉 All workflow API tests passed!")
        else:
            logger.error("❌ Some workflow API tests failed")
        
        return results


async def main():
    """Main test function."""
    print("🧪 Workflow API Test Suite")
    print("=" * 50)
    
    async with WorkflowAPITester() as tester:
        results = await tester.run_full_test_suite()
        
        print("\n📊 Test Results:")
        print("=" * 50)
        for test_name, result in results.items():
            if test_name == "success":
                continue
            
            if result and "error" not in result:
                print(f"✅ {test_name.upper()}: PASSED")
            else:
                print(f"❌ {test_name.upper()}: FAILED")
                if result and "error" in result:
                    print(f"   Error: {result['error']}")
        
        print(f"\n🏆 Overall Result: {'PASSED' if results['success'] else 'FAILED'}")
        
        return 0 if results["success"] else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)
