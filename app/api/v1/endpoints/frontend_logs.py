"""
Frontend Logs API Endpoint
Receives and stores logs from the frontend application
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os
import logging
from pathlib import Path

router = APIRouter()

# Configure logging for frontend logs
frontend_logger = logging.getLogger("frontend_logs")
frontend_logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
logs_dir = Path("logs/frontend")
logs_dir.mkdir(parents=True, exist_ok=True)

# Create file handler for frontend logs
log_file = logs_dir / f"frontend_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)

# Add handler to logger
if not frontend_logger.handlers:
    frontend_logger.addHandler(file_handler)

class FrontendLogEntry(BaseModel):
    timestamp: str = Field(..., description="ISO timestamp of the log entry")
    level: str = Field(..., description="Log level (DEBUG, INFO, WARN, ERROR, FATAL)")
    message: str = Field(..., description="Log message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional log data")
    source: str = Field(..., description="Source of the log (e.g., Console, API, UserInteraction)")
    userAgent: str = Field(..., description="User agent string")
    url: str = Field(..., description="Current page URL")
    userId: Optional[str] = Field(None, description="User ID if available")
    sessionId: str = Field(..., description="Frontend session ID")
    stackTrace: Optional[str] = Field(None, description="Stack trace for errors")

class FrontendLogBatch(BaseModel):
    logs: List[FrontendLogEntry] = Field(..., description="Batch of log entries")
    sessionId: str = Field(..., description="Session ID for the batch")

class LogStats(BaseModel):
    totalLogs: int
    logsByLevel: Dict[str, int]
    logsBySource: Dict[str, int]
    sessionCount: int
    oldestLog: Optional[str]
    newestLog: Optional[str]

def write_log_to_file(log_entry: FrontendLogEntry):
    """Write a single log entry to the file"""
    try:
        # Format the log entry for file storage
        log_data = {
            "timestamp": log_entry.timestamp,
            "level": log_entry.level,
            "message": log_entry.message,
            "source": log_entry.source,
            "sessionId": log_entry.sessionId,
            "url": log_entry.url,
            "userAgent": log_entry.userAgent,
        }
        
        if log_entry.data:
            log_data["data"] = log_entry.data
            
        if log_entry.stackTrace:
            log_data["stackTrace"] = log_entry.stackTrace
            
        if log_entry.userId:
            log_data["userId"] = log_entry.userId

        # Write to file as JSON
        log_line = json.dumps(log_data, separators=(',', ':'))
        
        # Map frontend log levels to Python logging levels
        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "FATAL": logging.CRITICAL
        }
        
        python_level = level_mapping.get(log_entry.level, logging.INFO)
        frontend_logger.log(python_level, log_line)
        
    except Exception as e:
        # Don't let logging errors break the API
        print(f"Error writing frontend log: {e}")

@router.post("/logs", summary="Receive frontend log entry")
async def receive_log(
    log_entry: FrontendLogEntry,
    background_tasks: BackgroundTasks
):
    """
    Receive a single log entry from the frontend
    """
    try:
        # Process log in background to avoid blocking the response
        background_tasks.add_task(write_log_to_file, log_entry)
        
        return {"status": "success", "message": "Log received"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process log: {str(e)}")

@router.post("/logs/batch", summary="Receive batch of frontend logs")
async def receive_log_batch(
    log_batch: FrontendLogBatch,
    background_tasks: BackgroundTasks
):
    """
    Receive a batch of log entries from the frontend
    """
    try:
        # Process each log in the batch
        for log_entry in log_batch.logs:
            background_tasks.add_task(write_log_to_file, log_entry)
        
        return {
            "status": "success", 
            "message": f"Received {len(log_batch.logs)} logs for session {log_batch.sessionId}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process log batch: {str(e)}")

@router.get("/logs/stats", response_model=LogStats, summary="Get frontend logging statistics")
async def get_log_stats():
    """
    Get statistics about frontend logs
    """
    try:
        stats = {
            "totalLogs": 0,
            "logsByLevel": {},
            "logsBySource": {},
            "sessionCount": 0,
            "oldestLog": None,
            "newestLog": None
        }
        
        # Read today's log file
        today_log_file = logs_dir / f"frontend_{datetime.now().strftime('%Y%m%d')}.log"
        
        if today_log_file.exists():
            sessions = set()
            oldest_timestamp = None
            newest_timestamp = None
            
            with open(today_log_file, 'r') as f:
                for line in f:
                    try:
                        # Skip non-JSON lines (like log headers)
                        if not line.strip().startswith('{'):
                            continue
                            
                        log_data = json.loads(line.strip().split(' - ', 3)[-1])
                        stats["totalLogs"] += 1
                        
                        # Count by level
                        level = log_data.get("level", "UNKNOWN")
                        stats["logsByLevel"][level] = stats["logsByLevel"].get(level, 0) + 1
                        
                        # Count by source
                        source = log_data.get("source", "UNKNOWN")
                        stats["logsBySource"][source] = stats["logsBySource"].get(source, 0) + 1
                        
                        # Track sessions
                        session_id = log_data.get("sessionId")
                        if session_id:
                            sessions.add(session_id)
                        
                        # Track timestamps
                        timestamp = log_data.get("timestamp")
                        if timestamp:
                            if oldest_timestamp is None or timestamp < oldest_timestamp:
                                oldest_timestamp = timestamp
                            if newest_timestamp is None or timestamp > newest_timestamp:
                                newest_timestamp = timestamp
                                
                    except (json.JSONDecodeError, IndexError):
                        # Skip malformed lines
                        continue
            
            stats["sessionCount"] = len(sessions)
            stats["oldestLog"] = oldest_timestamp
            stats["newestLog"] = newest_timestamp
        
        return LogStats(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get log stats: {str(e)}")

@router.get("/logs/download", summary="Download frontend logs")
async def download_logs(date: Optional[str] = None):
    """
    Download frontend logs for a specific date (YYYYMMDD format)
    If no date provided, downloads today's logs
    """
    try:
        if date:
            # Validate date format
            try:
                datetime.strptime(date, '%Y%m%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYYMMDD")
            log_file = logs_dir / f"frontend_{date}.log"
        else:
            log_file = logs_dir / f"frontend_{datetime.now().strftime('%Y%m%d')}.log"
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=log_file,
            filename=f"frontend_logs_{date or datetime.now().strftime('%Y%m%d')}.log",
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download logs: {str(e)}")

@router.delete("/logs/clear", summary="Clear frontend logs")
async def clear_logs(date: Optional[str] = None):
    """
    Clear frontend logs for a specific date
    If no date provided, clears today's logs
    """
    try:
        if date:
            # Validate date format
            try:
                datetime.strptime(date, '%Y%m%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYYMMDD")
            log_file = logs_dir / f"frontend_{date}.log"
        else:
            log_file = logs_dir / f"frontend_{datetime.now().strftime('%Y%m%d')}.log"
        
        if log_file.exists():
            log_file.unlink()
            return {"status": "success", "message": f"Cleared logs for {date or 'today'}"}
        else:
            return {"status": "success", "message": "No logs found to clear"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")
