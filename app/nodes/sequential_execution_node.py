"""
SEQUENTIAL_EXECUTION Node Implementation

This module implements the SEQUENTIAL_EXECUTION node for workflow orchestration.
The SEQUENTIAL_EXECUTION node executes multiple inputs in a defined order.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.core.node_registry import (
    RegisteredNode, NodePort, NodeConnectionRule, PortType, NodeCategory
)
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()


@dataclass
class SequentialExecutionContext:
    """Execution context for sequential operations."""
    node_id: str
    workflow_id: str
    execution_id: str
    configuration: Dict[str, Any]
    input_data: Dict[str, Any]
    start_time: datetime


class SequentialExecutionNodeExecutor:
    """Executor for SEQUENTIAL_EXECUTION nodes."""
    
    @staticmethod
    async def execute_sequential_node(node_config: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a SEQUENTIAL_EXECUTION node.
        
        Args:
            node_config: Node configuration including execution settings
            execution_context: Execution context with workflow information
            
        Returns:
            Dict containing execution results
        """
        try:
            context = SequentialExecutionContext(
                node_id=execution_context.get('node_id', ''),
                workflow_id=execution_context.get('workflow_id', ''),
                execution_id=execution_context.get('execution_id', ''),
                configuration=node_config,
                input_data=execution_context.get('input_data', {}),
                start_time=datetime.now()
            )
            
            backend_logger.info(
                "Starting SEQUENTIAL_EXECUTION node execution",
                LogCategory.ORCHESTRATION,
                "SequentialExecutionNode",
                data={
                    "node_id": context.node_id,
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id
                }
            )
            
            # Get execution configuration
            execution_mode = node_config.get('execution_mode', 'ordered')  # ordered, priority
            stop_on_error = node_config.get('stop_on_error', True)
            delay_between_steps = node_config.get('delay_between_steps', 0.0)
            
            # Process input data - expect multiple inputs to be executed sequentially
            input_data = context.input_data
            execution_steps = []
            
            # Extract execution steps from input data
            if isinstance(input_data, dict):
                # If input is a dict, treat each key-value pair as a step
                for key, value in input_data.items():
                    execution_steps.append({
                        "step_id": key,
                        "step_data": value,
                        "step_order": len(execution_steps)
                    })
            elif isinstance(input_data, list):
                # If input is a list, treat each item as a step
                for i, item in enumerate(input_data):
                    execution_steps.append({
                        "step_id": f"step_{i}",
                        "step_data": item,
                        "step_order": i
                    })
            else:
                # Single input, create one step
                execution_steps.append({
                    "step_id": "single_step",
                    "step_data": input_data,
                    "step_order": 0
                })
            
            # Sort steps if priority mode is enabled
            if execution_mode == 'priority':
                execution_steps = await SequentialExecutionNodeExecutor._sort_by_priority(
                    execution_steps, node_config
                )
            
            # Execute steps sequentially
            results = []
            total_steps = len(execution_steps)
            
            for i, step in enumerate(execution_steps):
                step_start_time = datetime.now()
                
                backend_logger.debug(
                    f"Executing step {i + 1}/{total_steps}: {step['step_id']}",
                    LogCategory.ORCHESTRATION,
                    "SequentialExecutionNode",
                    data={
                        "node_id": context.node_id,
                        "step_id": step['step_id'],
                        "step_order": step['step_order']
                    }
                )
                
                try:
                    # Execute the step
                    step_result = await SequentialExecutionNodeExecutor._execute_step(
                        step, context
                    )
                    
                    step_end_time = datetime.now()
                    step_duration = (step_end_time - step_start_time).total_seconds()
                    
                    step_result.update({
                        "step_duration_seconds": step_duration,
                        "step_completed_at": step_end_time.isoformat(),
                        "step_index": i,
                        "total_steps": total_steps
                    })
                    
                    results.append(step_result)
                    
                    # Add delay between steps if configured
                    if delay_between_steps > 0 and i < total_steps - 1:
                        await asyncio.sleep(delay_between_steps)
                    
                except Exception as step_error:
                    error_result = {
                        "step_id": step['step_id'],
                        "step_order": step['step_order'],
                        "success": False,
                        "error": str(step_error),
                        "step_duration_seconds": (datetime.now() - step_start_time).total_seconds(),
                        "step_index": i,
                        "total_steps": total_steps
                    }
                    
                    results.append(error_result)
                    
                    backend_logger.error(
                        f"Step execution failed: {step['step_id']}",
                        LogCategory.ORCHESTRATION,
                        "SequentialExecutionNode",
                        error=str(step_error),
                        data={
                            "node_id": context.node_id,
                            "step_id": step['step_id']
                        }
                    )
                    
                    if stop_on_error:
                        backend_logger.warning(
                            "Stopping sequential execution due to error",
                            LogCategory.ORCHESTRATION,
                            "SequentialExecutionNode",
                            data={"node_id": context.node_id, "failed_step": step['step_id']}
                        )
                        break
            
            execution_time = (datetime.now() - context.start_time).total_seconds()
            successful_steps = len([r for r in results if r.get('success', False)])
            failed_steps = len([r for r in results if not r.get('success', False)])
            
            backend_logger.info(
                "SEQUENTIAL_EXECUTION node execution completed",
                LogCategory.ORCHESTRATION,
                "SequentialExecutionNode",
                data={
                    "node_id": context.node_id,
                    "execution_time_seconds": execution_time,
                    "total_steps": total_steps,
                    "successful_steps": successful_steps,
                    "failed_steps": failed_steps
                }
            )
            
            return {
                "success": failed_steps == 0 or not stop_on_error,
                "data": {
                    "execution_mode": execution_mode,
                    "total_steps": total_steps,
                    "successful_steps": successful_steps,
                    "failed_steps": failed_steps,
                    "execution_time_seconds": execution_time,
                    "step_results": results,
                    "completed_at": datetime.now().isoformat()
                },
                "metadata": {
                    "node_type": "SEQUENTIAL_EXECUTION",
                    "execution_context": {
                        "node_id": context.node_id,
                        "workflow_id": context.workflow_id,
                        "execution_id": context.execution_id
                    }
                }
            }
            
        except Exception as e:
            backend_logger.error(
                "SEQUENTIAL_EXECUTION node execution failed",
                LogCategory.ORCHESTRATION,
                "SequentialExecutionNode",
                error=str(e),
                data={
                    "node_id": execution_context.get('node_id', ''),
                    "workflow_id": execution_context.get('workflow_id', '')
                }
            )
            
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "node_type": "SEQUENTIAL_EXECUTION",
                    "execution_context": execution_context
                }
            }
    
    @staticmethod
    async def _sort_by_priority(steps: List[Dict[str, Any]], node_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sort steps by priority if priority mode is enabled."""
        priority_field = node_config.get('priority_field', 'priority')
        
        # Add default priority if not present
        for step in steps:
            if priority_field not in step.get('step_data', {}):
                if isinstance(step['step_data'], dict):
                    step['step_data'][priority_field] = step['step_order']
                else:
                    # Can't add priority to non-dict data, use step order
                    step['priority'] = step['step_order']
        
        # Sort by priority (lower numbers = higher priority)
        def get_priority(step):
            if isinstance(step['step_data'], dict):
                return step['step_data'].get(priority_field, step['step_order'])
            else:
                return step.get('priority', step['step_order'])
        
        return sorted(steps, key=get_priority)
    
    @staticmethod
    async def _execute_step(step: Dict[str, Any], context: SequentialExecutionContext) -> Dict[str, Any]:
        """Execute a single step in the sequence."""
        step_data = step['step_data']
        
        # For now, this is a simple pass-through execution
        # In a more complex implementation, this could:
        # - Execute sub-workflows
        # - Call external APIs
        # - Perform data transformations
        # - Execute agent tasks
        
        # Simulate some processing time for demonstration
        processing_time = context.configuration.get('step_processing_time', 0.1)
        if processing_time > 0:
            await asyncio.sleep(processing_time)
        
        return {
            "step_id": step['step_id'],
            "step_order": step['step_order'],
            "success": True,
            "input_data": step_data,
            "output_data": step_data,  # Pass-through for now
            "processing_notes": "Sequential step executed successfully"
        }


def create_sequential_execution_node_definition() -> RegisteredNode:
    """Create the SEQUENTIAL_EXECUTION node definition."""
    return RegisteredNode(
        node_type="SEQUENTIAL_EXECUTION",
        name="Sequential Execution",
        description="Execute multiple inputs in a defined sequential order",
        category=NodeCategory.WORKFLOW,
        input_ports=[
            NodePort(
                id="input",
                name="Input Data",
                type=PortType.DATA,
                required=True,
                description="Data to be processed sequentially"
            ),
            NodePort(
                id="control",
                name="Control Signal",
                type=PortType.CONTROL,
                required=False,
                description="Optional control signal to trigger execution"
            )
        ],
        output_ports=[
            NodePort(
                id="output",
                name="Sequential Results",
                type=PortType.DATA,
                required=True,
                description="Results from sequential execution"
            ),
            NodePort(
                id="control_out",
                name="Control Output",
                type=PortType.CONTROL,
                required=False,
                description="Control signal after completion"
            )
        ],
        configuration_schema={
            "type": "object",
            "properties": {
                "execution_mode": {
                    "type": "string",
                    "enum": ["ordered", "priority"],
                    "default": "ordered",
                    "description": "Execution mode: ordered or priority-based"
                },
                "stop_on_error": {
                    "type": "boolean",
                    "default": True,
                    "description": "Stop execution if a step fails"
                },
                "delay_between_steps": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 60,
                    "default": 0.0,
                    "description": "Delay in seconds between steps"
                },
                "priority_field": {
                    "type": "string",
                    "default": "priority",
                    "description": "Field name for priority-based sorting"
                },
                "step_processing_time": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "default": 0.1,
                    "description": "Simulated processing time per step"
                }
            }
        },
        default_configuration={
            "execution_mode": "ordered",
            "stop_on_error": True,
            "delay_between_steps": 0.0,
            "priority_field": "priority",
            "step_processing_time": 0.1
        },
        execution_handler=SequentialExecutionNodeExecutor.execute_sequential_node,
        connection_rules=NodeConnectionRule(
            allowed_input_types=[PortType.DATA, PortType.CONTROL],
            allowed_output_types=[PortType.DATA, PortType.CONTROL],
            max_input_connections=-1,
            max_output_connections=-1
        ),
        icon="📋",
        color="bg-blue-500",
        execution_timeout=600  # 10 minutes max
    )
