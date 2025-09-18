"""
Backend Logs API Endpoint

Provides REST API endpoints for querying, filtering, and managing backend logs.
Supports real-time log streaming and comprehensive log analytics.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import gzip

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import (
    LogLevel, LogCategory, LogQuery, LogStats, LogEntry, LogConfiguration
)

router = APIRouter()

# Get the global logger instance
backend_logger = get_logger()


class LogQueryRequest(BaseModel):
    """Request model for log queries"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: Optional[List[LogLevel]] = None
    categories: Optional[List[LogCategory]] = None
    components: Optional[List[str]] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    search_term: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class LogStreamMessage(BaseModel):
    """WebSocket message for log streaming"""
    type: str  # "log", "error", "ping"
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketManager:
    """Manages WebSocket connections for real-time log streaming"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: LogStreamMessage):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message.json())
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global WebSocket manager
ws_manager = WebSocketManager()


def read_log_files(
    category: Optional[LogCategory] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_term: Optional[str] = None,
    levels: Optional[List[LogLevel]] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Read and parse log files based on filters
    """
    logs = []
    logs_dir = Path("logs/backend")
    
    if not logs_dir.exists():
        return logs
    
    # Determine which files to read
    files_to_read = []
    
    if category:
        # Read specific category files
        pattern = f"{category.value}_*.log"
        files_to_read.extend(logs_dir.glob(pattern))
        # Also check for compressed files
        files_to_read.extend(logs_dir.glob(f"{pattern}.gz"))
    else:
        # Read all log files
        files_to_read.extend(logs_dir.glob("*.log"))
        files_to_read.extend(logs_dir.glob("*.log.gz"))
    
    # Filter by date if specified
    if start_date or end_date:
        filtered_files = []
        for file_path in files_to_read:
            try:
                # Extract date from filename (format: category_YYYYMMDD.log)
                date_part = file_path.stem.split('_')[-1]
                if date_part.isdigit() and len(date_part) == 8:
                    file_date = datetime.strptime(date_part, '%Y%m%d')
                    
                    if start_date and file_date < start_date.replace(hour=0, minute=0, second=0, microsecond=0):
                        continue
                    if end_date and file_date > end_date.replace(hour=23, minute=59, second=59, microsecond=999999):
                        continue
                    
                    filtered_files.append(file_path)
            except (ValueError, IndexError):
                # Include files that don't match the date pattern
                filtered_files.append(file_path)
        
        files_to_read = filtered_files
    
    # Read and parse log files
    for file_path in sorted(files_to_read):
        try:
            # Handle compressed files
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            for line in lines:
                try:
                    # Parse JSON log entry
                    if line.strip().startswith('{'):
                        log_data = json.loads(line.strip())
                    else:
                        # Handle structured text logs
                        # Extract JSON from structured log line
                        json_start = line.find('{')
                        if json_start != -1:
                            log_data = json.loads(line[json_start:].strip())
                        else:
                            continue
                    
                    # Apply filters
                    if levels and log_data.get('level') not in [level.value for level in levels]:
                        continue
                    
                    if search_term and search_term.lower() not in log_data.get('message', '').lower():
                        continue
                    
                    # Parse timestamp for time filtering
                    if start_date or end_date:
                        try:
                            log_time = datetime.fromisoformat(log_data.get('timestamp', '').replace('Z', '+00:00'))
                            if start_date and log_time < start_date:
                                continue
                            if end_date and log_time > end_date:
                                continue
                        except (ValueError, TypeError):
                            continue
                    
                    logs.append(log_data)
                    
                    # Limit results
                    if len(logs) >= limit:
                        return logs
                        
                except (json.JSONDecodeError, ValueError):
                    # Skip malformed log entries
                    continue
                    
        except Exception as e:
            # Log file read error, but continue with other files
            backend_logger.warn(
                f"Error reading log file {file_path}: {str(e)}",
                category=LogCategory.SYSTEM_HEALTH,
                component="BackendLogsAPI"
            )
            continue
    
    return logs


@router.get("/logs", summary="Query backend logs")
async def query_logs(
    start_time: Optional[datetime] = Query(None, description="Start time for log query"),
    end_time: Optional[datetime] = Query(None, description="End time for log query"),
    levels: Optional[List[LogLevel]] = Query(None, description="Log levels to include"),
    categories: Optional[List[LogCategory]] = Query(None, description="Log categories to include"),
    components: Optional[List[str]] = Query(None, description="Components to include"),
    correlation_id: Optional[str] = Query(None, description="Correlation ID to filter by"),
    search_term: Optional[str] = Query(None, description="Search term in log messages"),
    limit: int = Query(100, le=1000, description="Maximum number of logs to return"),
    offset: int = Query(0, ge=0, description="Number of logs to skip")
):
    """
    Query backend logs with various filters
    """
    try:
        # If no end time specified, default to now
        if start_time and not end_time:
            end_time = datetime.utcnow()
        
        # If no start time specified, default to last 24 hours
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.utcnow()
        
        logs = []
        
        # Read logs from each category if specified, otherwise all categories
        categories_to_read = categories or list(LogCategory)
        
        for category in categories_to_read:
            category_logs = read_log_files(
                category=category,
                start_date=start_time,
                end_date=end_time,
                search_term=search_term,
                levels=levels,
                limit=limit - len(logs)
            )
            logs.extend(category_logs)
            
            if len(logs) >= limit:
                break
        
        # Apply additional filters
        filtered_logs = []
        for log in logs:
            # Filter by correlation_id
            if correlation_id:
                log_correlation_id = log.get('context', {}).get('correlation_id')
                if log_correlation_id != correlation_id:
                    continue
            
            # Filter by components
            if components and log.get('component') not in components:
                continue
            
            filtered_logs.append(log)
        
        # Apply offset and limit
        result_logs = filtered_logs[offset:offset + limit]
        
        return {
            "logs": result_logs,
            "total": len(filtered_logs),
            "limit": limit,
            "offset": offset,
            "query": {
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "levels": levels,
                "categories": categories,
                "components": components,
                "correlation_id": correlation_id,
                "search_term": search_term
            }
        }
        
    except Exception as e:
        backend_logger.error(
            f"Error querying logs: {str(e)}",
            category=LogCategory.API_LAYER,
            component="BackendLogsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to query logs: {str(e)}")


@router.get("/logs/stats", response_model=LogStats, summary="Get backend log statistics")
async def get_log_stats(
    start_time: Optional[datetime] = Query(None, description="Start time for statistics"),
    end_time: Optional[datetime] = Query(None, description="End time for statistics")
):
    """
    Get comprehensive statistics about backend logs
    """
    try:
        # Default to last 24 hours if no time range specified
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()
        
        # Read all logs for the time period
        all_logs = []
        for category in LogCategory:
            category_logs = read_log_files(
                category=category,
                start_date=start_time,
                end_date=end_time,
                limit=10000  # Large limit for stats
            )
            all_logs.extend(category_logs)
        
        # Calculate statistics
        total_logs = len(all_logs)
        logs_by_level = {}
        logs_by_category = {}
        logs_by_component = {}
        error_count = 0
        response_times = []
        memory_usages = []
        active_agents = set()
        failed_operations = 0
        
        for log in all_logs:
            # Count by level
            level = log.get('level', 'UNKNOWN')
            logs_by_level[level] = logs_by_level.get(level, 0) + 1
            
            # Count by category
            category = log.get('category', 'unknown')
            logs_by_category[category] = logs_by_category.get(category, 0) + 1
            
            # Count by component
            component = log.get('component', 'unknown')
            logs_by_component[component] = logs_by_component.get(component, 0) + 1
            
            # Count errors
            if level in ['ERROR', 'FATAL']:
                error_count += 1
            
            # Collect performance metrics
            api_metrics = log.get('api_metrics', {})
            if api_metrics.get('response_time_ms'):
                response_times.append(api_metrics['response_time_ms'])
            
            performance = log.get('performance', {})
            if performance.get('memory_usage_mb'):
                memory_usages.append(performance['memory_usage_mb'])
            
            # Track active agents
            context = log.get('context', {})
            if context.get('agent_id'):
                active_agents.add(context['agent_id'])
            
            # Count failed operations
            if level == 'ERROR' and log.get('category') in ['agent_operations', 'api_layer']:
                failed_operations += 1
        
        # Calculate averages
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        peak_memory = max(memory_usages) if memory_usages else None
        error_rate = (error_count / total_logs * 100) if total_logs > 0 else 0
        
        return LogStats(
            total_logs=total_logs,
            logs_by_level=logs_by_level,
            logs_by_category=logs_by_category,
            logs_by_component=logs_by_component,
            error_rate=error_rate,
            average_response_time=avg_response_time,
            peak_memory_usage=peak_memory,
            active_agents=len(active_agents),
            failed_operations=failed_operations,
            time_range={
                "start": start_time,
                "end": end_time
            }
        )
        
    except Exception as e:
        backend_logger.error(
            f"Error getting log stats: {str(e)}",
            category=LogCategory.API_LAYER,
            component="BackendLogsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to get log statistics: {str(e)}")


@router.get("/logs/download", summary="Download backend logs")
async def download_logs(
    category: Optional[LogCategory] = Query(None, description="Log category to download"),
    date: Optional[str] = Query(None, description="Date in YYYYMMDD format"),
    format: str = Query("json", description="Download format: json or text")
):
    """
    Download backend logs for a specific category and date
    """
    try:
        # Default to today if no date specified
        if not date:
            date = datetime.now().strftime('%Y%m%d')
        else:
            # Validate date format
            try:
                datetime.strptime(date, '%Y%m%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYYMMDD")

        logs_dir = Path("logs/backend")

        if category:
            # Download specific category
            log_file = logs_dir / f"{category.value}_{date}.log"
            compressed_file = logs_dir / f"{category.value}_{date}.log.gz"

            # Check for compressed file first
            if compressed_file.exists():
                return FileResponse(
                    path=compressed_file,
                    filename=f"{category.value}_{date}.log.gz",
                    media_type="application/gzip"
                )
            elif log_file.exists():
                return FileResponse(
                    path=log_file,
                    filename=f"{category.value}_{date}.log",
                    media_type="text/plain"
                )
            else:
                raise HTTPException(status_code=404, detail="Log file not found")
        else:
            # Download all logs for the date
            log_files = list(logs_dir.glob(f"*_{date}.log*"))

            if not log_files:
                raise HTTPException(status_code=404, detail="No log files found for the specified date")

            # Create a combined download
            def generate_combined_logs():
                for log_file in sorted(log_files):
                    yield f"=== {log_file.name} ===\n"

                    try:
                        if log_file.suffix == '.gz':
                            with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                                for line in f:
                                    yield line
                        else:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    yield line
                    except Exception as e:
                        yield f"Error reading {log_file.name}: {str(e)}\n"

                    yield "\n"

            return StreamingResponse(
                generate_combined_logs(),
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=backend_logs_{date}.log"}
            )

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Error downloading logs: {str(e)}",
            category=LogCategory.API_LAYER,
            component="BackendLogsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to download logs: {str(e)}")


@router.delete("/logs/clear", summary="Clear backend logs")
async def clear_logs(
    category: Optional[LogCategory] = Query(None, description="Log category to clear"),
    date: Optional[str] = Query(None, description="Date in YYYYMMDD format"),
    confirm: bool = Query(False, description="Confirmation flag")
):
    """
    Clear backend logs for a specific category and date
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Must set confirm=true to clear logs")

    try:
        logs_dir = Path("logs/backend")

        if not logs_dir.exists():
            return {"status": "success", "message": "No logs directory found"}

        files_deleted = []

        if category and date:
            # Clear specific category and date
            patterns = [f"{category.value}_{date}.log", f"{category.value}_{date}.log.gz"]
        elif category:
            # Clear all files for category
            patterns = [f"{category.value}_*.log", f"{category.value}_*.log.gz"]
        elif date:
            # Clear all categories for date
            patterns = [f"*_{date}.log", f"*_{date}.log.gz"]
        else:
            # Clear all logs (dangerous!)
            patterns = ["*.log", "*.log.gz"]

        for pattern in patterns:
            for file_path in logs_dir.glob(pattern):
                try:
                    file_path.unlink()
                    files_deleted.append(file_path.name)
                except Exception as e:
                    backend_logger.error(
                        f"Error deleting log file {file_path}: {str(e)}",
                        category=LogCategory.SYSTEM_HEALTH,
                        component="BackendLogsAPI",
                        error=e
                    )

        backend_logger.info(
            f"Cleared {len(files_deleted)} log files",
            category=LogCategory.SYSTEM_HEALTH,
            component="BackendLogsAPI",
            data={"files_deleted": files_deleted}
        )

        return {
            "status": "success",
            "message": f"Cleared {len(files_deleted)} log files",
            "files_deleted": files_deleted
        }

    except Exception as e:
        backend_logger.error(
            f"Error clearing logs: {str(e)}",
            category=LogCategory.API_LAYER,
            component="BackendLogsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")


@router.websocket("/logs/stream")
async def stream_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time log streaming
    """
    await ws_manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_text(LogStreamMessage(
            type="connected",
            data={"message": "Connected to log stream"}
        ).json())

        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (like filter updates)
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(LogStreamMessage(
                        type="pong",
                        data={"timestamp": datetime.utcnow().isoformat()}
                    ).json())

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_text(LogStreamMessage(
                    type="error",
                    data={"error": str(e)}
                ).json())

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket)


@router.get("/logs/config", response_model=LogConfiguration, summary="Get logging configuration")
async def get_logging_config():
    """
    Get current logging configuration
    """
    try:
        return backend_logger.config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logging config: {str(e)}")


@router.put("/logs/config", summary="Update logging configuration")
async def update_logging_config(config: LogConfiguration):
    """
    Update logging configuration
    """
    try:
        # Update the logger configuration
        backend_logger.config = config

        backend_logger.info(
            "Logging configuration updated",
            category=LogCategory.CONFIGURATION_MANAGEMENT,
            component="BackendLogsAPI",
            data={"new_config": config.dict()}
        )

        return {"status": "success", "message": "Logging configuration updated"}

    except Exception as e:
        backend_logger.error(
            f"Error updating logging config: {str(e)}",
            category=LogCategory.API_LAYER,
            component="BackendLogsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to update logging config: {str(e)}")


@router.get("/logs/health", summary="Get logging system health")
async def get_logging_health():
    """
    Get health status of the logging system
    """
    try:
        stats = backend_logger.get_stats()

        # Check for potential issues
        issues = []

        if stats.get("queue_size", 0) > 500:
            issues.append("High log queue size")

        if not stats.get("worker_running", False):
            issues.append("Log worker thread not running")

        # Check disk space for logs directory
        logs_dir = Path("logs/backend")
        if logs_dir.exists():
            try:
                import shutil
                total, used, free = shutil.disk_usage(logs_dir)
                free_gb = free / (1024**3)

                if free_gb < 1:  # Less than 1GB free
                    issues.append(f"Low disk space: {free_gb:.2f}GB free")

                stats["disk_space"] = {
                    "total_gb": total / (1024**3),
                    "used_gb": used / (1024**3),
                    "free_gb": free_gb
                }
            except Exception:
                issues.append("Could not check disk space")

        health_status = "healthy" if not issues else "warning"

        return {
            "status": health_status,
            "issues": issues,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        backend_logger.error(
            f"Error getting logging health: {str(e)}",
            category=LogCategory.API_LAYER,
            component="BackendLogsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to get logging health: {str(e)}")
