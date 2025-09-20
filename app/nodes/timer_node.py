"""
TIMER Node Implementation

This module implements the TIMER node for workflow orchestration.
The TIMER node provides time-based triggers and delays in workflows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from app.core.node_registry import (
    RegisteredNode, NodePort, NodeConnectionRule, PortType, NodeCategory
)
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()


@dataclass
class TimerExecutionContext:
    """Execution context for timer operations."""
    node_id: str
    workflow_id: str
    execution_id: str
    configuration: Dict[str, Any]
    input_data: Dict[str, Any]
    start_time: datetime


class TimerNodeExecutor:
    """Executor for TIMER nodes."""
    
    @staticmethod
    async def execute_timer_node(node_config: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a TIMER node.
        
        Args:
            node_config: Node configuration including timer settings
            execution_context: Execution context with workflow information
            
        Returns:
            Dict containing execution results
        """
        try:
            context = TimerExecutionContext(
                node_id=execution_context.get('node_id', ''),
                workflow_id=execution_context.get('workflow_id', ''),
                execution_id=execution_context.get('execution_id', ''),
                configuration=node_config,
                input_data=execution_context.get('input_data', {}),
                start_time=datetime.now()
            )
            
            backend_logger.info(
                "Starting TIMER node execution",
                LogCategory.ORCHESTRATION,
                "TimerNode",
                data={
                    "node_id": context.node_id,
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id
                }
            )
            
            # Get timer configuration
            timer_type = node_config.get('timer_type', 'delay')  # delay, interval, schedule
            duration_seconds = node_config.get('duration_seconds', 1.0)
            repeat_count = node_config.get('repeat_count', 1)
            interval_seconds = node_config.get('interval_seconds', 1.0)
            
            results = []
            
            if timer_type == 'delay':
                # Simple delay timer
                result = await TimerNodeExecutor._execute_delay_timer(
                    duration_seconds, context
                )
                results.append(result)
                
            elif timer_type == 'interval':
                # Interval timer with repeats
                for i in range(repeat_count):
                    result = await TimerNodeExecutor._execute_interval_timer(
                        interval_seconds, i + 1, repeat_count, context
                    )
                    results.append(result)
                    
            elif timer_type == 'schedule':
                # Scheduled timer (future enhancement)
                result = await TimerNodeExecutor._execute_scheduled_timer(
                    node_config, context
                )
                results.append(result)
            
            else:
                raise ValueError(f"Unknown timer type: {timer_type}")
            
            execution_time = (datetime.now() - context.start_time).total_seconds()
            
            backend_logger.info(
                "TIMER node execution completed",
                LogCategory.ORCHESTRATION,
                "TimerNode",
                data={
                    "node_id": context.node_id,
                    "execution_time_seconds": execution_time,
                    "timer_type": timer_type,
                    "results_count": len(results)
                }
            )
            
            return {
                "success": True,
                "data": {
                    "timer_type": timer_type,
                    "execution_time_seconds": execution_time,
                    "results": results,
                    "trigger_count": len(results),
                    "completed_at": datetime.now().isoformat()
                },
                "metadata": {
                    "node_type": "TIMER",
                    "execution_context": {
                        "node_id": context.node_id,
                        "workflow_id": context.workflow_id,
                        "execution_id": context.execution_id
                    }
                }
            }
            
        except Exception as e:
            backend_logger.error(
                "TIMER node execution failed",
                LogCategory.ORCHESTRATION,
                "TimerNode",
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
                    "node_type": "TIMER",
                    "execution_context": execution_context
                }
            }
    
    @staticmethod
    async def _execute_delay_timer(duration_seconds: float, context: TimerExecutionContext) -> Dict[str, Any]:
        """Execute a simple delay timer."""
        start_time = datetime.now()
        
        backend_logger.debug(
            f"Starting delay timer for {duration_seconds} seconds",
            LogCategory.ORCHESTRATION,
            "TimerNode",
            data={"node_id": context.node_id, "duration": duration_seconds}
        )
        
        await asyncio.sleep(duration_seconds)
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        
        return {
            "trigger_type": "delay",
            "requested_duration": duration_seconds,
            "actual_duration": actual_duration,
            "triggered_at": end_time.isoformat(),
            "trigger_data": {
                "control_signal": True,
                "timestamp": end_time.isoformat()
            }
        }
    
    @staticmethod
    async def _execute_interval_timer(
        interval_seconds: float, 
        current_iteration: int, 
        total_iterations: int,
        context: TimerExecutionContext
    ) -> Dict[str, Any]:
        """Execute an interval timer."""
        start_time = datetime.now()
        
        backend_logger.debug(
            f"Interval timer iteration {current_iteration}/{total_iterations}",
            LogCategory.ORCHESTRATION,
            "TimerNode",
            data={
                "node_id": context.node_id,
                "interval": interval_seconds,
                "iteration": current_iteration
            }
        )
        
        await asyncio.sleep(interval_seconds)
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        
        return {
            "trigger_type": "interval",
            "iteration": current_iteration,
            "total_iterations": total_iterations,
            "requested_interval": interval_seconds,
            "actual_duration": actual_duration,
            "triggered_at": end_time.isoformat(),
            "trigger_data": {
                "control_signal": True,
                "timestamp": end_time.isoformat(),
                "iteration": current_iteration,
                "is_final": current_iteration == total_iterations
            }
        }
    
    @staticmethod
    async def _execute_scheduled_timer(node_config: Dict[str, Any], context: TimerExecutionContext) -> Dict[str, Any]:
        """Execute a scheduled timer (placeholder for future implementation)."""
        # This would implement cron-like scheduling
        # For now, just return a simple delay
        schedule_time = node_config.get('schedule_time', '1s')
        
        # Parse simple schedule format (e.g., "5s", "2m", "1h")
        if schedule_time.endswith('s'):
            seconds = float(schedule_time[:-1])
        elif schedule_time.endswith('m'):
            seconds = float(schedule_time[:-1]) * 60
        elif schedule_time.endswith('h'):
            seconds = float(schedule_time[:-1]) * 3600
        else:
            seconds = 1.0
        
        return await TimerNodeExecutor._execute_delay_timer(seconds, context)


def create_timer_node_definition() -> RegisteredNode:
    """Create the TIMER node definition."""
    return RegisteredNode(
        node_type="TIMER",
        name="Timer",
        description="Time-based triggers and delays for workflow control",
        category=NodeCategory.CONTROL_FLOW,
        input_ports=[
            NodePort(
                id="trigger",
                name="Trigger",
                type=PortType.CONTROL,
                required=False,
                description="Optional trigger to start the timer"
            )
        ],
        output_ports=[
            NodePort(
                id="output",
                name="Timer Output",
                type=PortType.CONTROL,
                required=True,
                description="Control signal triggered after timer completion"
            ),
            NodePort(
                id="data",
                name="Timer Data",
                type=PortType.JSON,
                required=False,
                description="Timer execution data and metadata"
            )
        ],
        configuration_schema={
            "type": "object",
            "properties": {
                "timer_type": {
                    "type": "string",
                    "enum": ["delay", "interval", "schedule"],
                    "default": "delay",
                    "description": "Type of timer operation"
                },
                "duration_seconds": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 3600,
                    "default": 1.0,
                    "description": "Duration in seconds for delay timer"
                },
                "repeat_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 1,
                    "description": "Number of repetitions for interval timer"
                },
                "interval_seconds": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 3600,
                    "default": 1.0,
                    "description": "Interval in seconds between triggers"
                },
                "schedule_time": {
                    "type": "string",
                    "default": "1s",
                    "description": "Schedule format (e.g., '5s', '2m', '1h')"
                }
            },
            "required": ["timer_type"]
        },
        default_configuration={
            "timer_type": "delay",
            "duration_seconds": 1.0,
            "repeat_count": 1,
            "interval_seconds": 1.0,
            "schedule_time": "1s"
        },
        execution_handler=TimerNodeExecutor.execute_timer_node,
        connection_rules=NodeConnectionRule(
            allowed_input_types=[PortType.CONTROL],
            allowed_output_types=[PortType.CONTROL, PortType.JSON],
            max_input_connections=1,
            max_output_connections=-1
        ),
        icon="⏰",
        color="bg-orange-500",
        execution_timeout=3600  # 1 hour max for long intervals
    )
