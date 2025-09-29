"""
Logging Handlers

Provides specialized logging handlers for file output, rotation, and async processing.
"""

import asyncio
import logging
import logging.handlers
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import gzip
import shutil


class AsyncFileHandler(logging.Handler):
    """
    Asynchronous file handler that writes logs without blocking
    """
    
    def __init__(self, filename: str, max_bytes: int = 0, backup_count: int = 0):
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.current_size = 0
        self.file_lock = threading.Lock()
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Open file
        self.stream = open(filename, 'a', encoding='utf-8')
        
        # Get current file size
        try:
            self.current_size = os.path.getsize(filename)
        except OSError:
            self.current_size = 0
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record"""
        try:
            msg = self.format(record)
            with self.file_lock:
                self._write_message(msg)
        except Exception:
            self.handleError(record)
    
    def _write_message(self, msg: str):
        """Write message to file with rotation check"""
        # Check if stream is available
        if self.stream is None:
            return  # Skip writing if stream is not available

        # Check if rotation is needed
        if self.max_bytes > 0 and self.current_size >= self.max_bytes:
            self._rotate_file()

        # Write message only if stream is still available
        if self.stream is not None:
            self.stream.write(msg + '\n')
            self.stream.flush()
            self.current_size += len(msg.encode('utf-8')) + 1
    
    def _rotate_file(self):
        """Rotate the log file"""
        if self.stream:
            self.stream.close()
        
        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            old_name = f"{self.filename}.{i}"
            new_name = f"{self.filename}.{i + 1}"
            
            if os.path.exists(old_name):
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
        
        # Move current file to .1
        if os.path.exists(self.filename):
            backup_name = f"{self.filename}.1"
            if os.path.exists(backup_name):
                os.remove(backup_name)
            os.rename(self.filename, backup_name)
        
        # Open new file
        self.stream = open(self.filename, 'a', encoding='utf-8')
        self.current_size = 0
    
    def close(self):
        """Close the handler"""
        with self.file_lock:
            if self.stream:
                self.stream.close()
                self.stream = None
        super().close()


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced rotating file handler with compression and cleanup
    """
    
    def __init__(self, filename: str, max_bytes: int = 0, backup_count: int = 0, 
                 compress_backups: bool = True, retention_days: int = 30):
        super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count)
        self.compress_backups = compress_backups
        self.retention_days = retention_days
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    def doRollover(self):
        """Perform file rollover with optional compression"""
        super().doRollover()
        
        # Compress backup files if enabled
        if self.compress_backups:
            self._compress_backup_files()
        
        # Clean up old files
        self._cleanup_old_files()
    
    def _compress_backup_files(self):
        """Compress backup log files"""
        for i in range(1, self.backupCount + 1):
            backup_file = f"{self.baseFilename}.{i}"
            compressed_file = f"{backup_file}.gz"
            
            if os.path.exists(backup_file) and not os.path.exists(compressed_file):
                try:
                    with open(backup_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file after compression
                    os.remove(backup_file)
                except Exception as e:
                    # Don't let compression errors break logging
                    print(f"Error compressing log file {backup_file}: {e}")
    
    def _cleanup_old_files(self):
        """Remove files older than retention period"""
        if self.retention_days <= 0:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        log_dir = Path(self.baseFilename).parent
        
        # Find old log files
        pattern = Path(self.baseFilename).name
        for file_path in log_dir.glob(f"{pattern}*"):
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
            except Exception as e:
                print(f"Error cleaning up old log file {file_path}: {e}")


class DailyRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Daily rotating file handler with date-based naming
    """
    
    def __init__(self, filename: str, backup_count: int = 30, compress_backups: bool = True):
        # Use daily rotation
        super().__init__(
            filename=filename,
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8'
        )
        self.compress_backups = compress_backups
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    def doRollover(self):
        """Perform daily rollover with compression"""
        super().doRollover()
        
        if self.compress_backups:
            self._compress_yesterday_file()
    
    def _compress_yesterday_file(self):
        """Compress yesterday's log file"""
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_suffix = yesterday.strftime("%Y-%m-%d")
        yesterday_file = f"{self.baseFilename}.{yesterday_suffix}"
        compressed_file = f"{yesterday_file}.gz"
        
        if os.path.exists(yesterday_file) and not os.path.exists(compressed_file):
            try:
                with open(yesterday_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original file after compression
                os.remove(yesterday_file)
            except Exception as e:
                print(f"Error compressing yesterday's log file: {e}")


class CategoryFileHandler(logging.Handler):
    """
    Handler that routes logs to different files based on category
    """
    
    def __init__(self, base_directory: str, max_bytes: int = 100 * 1024 * 1024, 
                 backup_count: int = 5):
        super().__init__()
        self.base_directory = Path(base_directory)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.handlers = {}
        
        # Ensure base directory exists
        self.base_directory.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record: logging.LogRecord):
        """Route log record to appropriate category file"""
        try:
            # Determine category
            category = "general"
            if hasattr(record, 'log_entry') and record.log_entry.category:
                category = record.log_entry.category.value
            
            # Get or create handler for category
            handler = self._get_category_handler(category)
            handler.emit(record)
            
        except Exception:
            self.handleError(record)
    
    def _get_category_handler(self, category: str) -> logging.Handler:
        """Get or create a handler for the specified category"""
        if category not in self.handlers:
            # Create date-based filename
            date_str = datetime.now().strftime("%Y%m%d")
            filename = self.base_directory / f"{category}_{date_str}.log"
            
            # Create rotating file handler
            handler = RotatingFileHandler(
                filename=str(filename),
                max_bytes=self.max_bytes,
                backup_count=self.backup_count
            )
            
            # Use the same formatter as this handler
            if self.formatter:
                handler.setFormatter(self.formatter)
            
            self.handlers[category] = handler
        
        return self.handlers[category]
    
    def close(self):
        """Close all category handlers"""
        for handler in self.handlers.values():
            handler.close()
        self.handlers.clear()
        super().close()


class BufferedHandler(logging.Handler):
    """
    Handler that buffers log records and flushes them in batches
    """
    
    def __init__(self, target_handler: logging.Handler, buffer_size: int = 100, 
                 flush_interval: float = 5.0):
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.last_flush = time.time()
        
        # Start flush timer
        self.flush_timer = threading.Timer(flush_interval, self._flush_buffer)
        self.flush_timer.daemon = True
        self.flush_timer.start()
    
    def emit(self, record: logging.LogRecord):
        """Add record to buffer"""
        with self.buffer_lock:
            self.buffer.append(record)
            
            # Flush if buffer is full or enough time has passed
            if (len(self.buffer) >= self.buffer_size or 
                time.time() - self.last_flush >= self.flush_interval):
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffered records to target handler"""
        with self.buffer_lock:
            if self.buffer:
                for record in self.buffer:
                    try:
                        self.target_handler.emit(record)
                    except Exception:
                        # Don't let individual record errors stop the flush
                        pass
                
                self.buffer.clear()
                self.last_flush = time.time()
        
        # Schedule next flush
        if hasattr(self, 'flush_timer'):
            self.flush_timer = threading.Timer(self.flush_interval, self._flush_buffer)
            self.flush_timer.daemon = True
            self.flush_timer.start()
    
    def close(self):
        """Close the handler and flush remaining records"""
        # Cancel timer
        if hasattr(self, 'flush_timer'):
            self.flush_timer.cancel()
        
        # Final flush
        self._flush_buffer()
        
        # Close target handler
        self.target_handler.close()
        super().close()


class WebSocketHandler(logging.Handler):
    """
    Handler that sends log records to WebSocket clients for real-time monitoring
    """
    
    def __init__(self):
        super().__init__()
        self.clients = set()
        self.client_lock = threading.Lock()
    
    def add_client(self, websocket):
        """Add a WebSocket client"""
        with self.client_lock:
            self.clients.add(websocket)
    
    def remove_client(self, websocket):
        """Remove a WebSocket client"""
        with self.client_lock:
            self.clients.discard(websocket)
    
    def emit(self, record: logging.LogRecord):
        """Send log record to all connected clients"""
        if not self.clients:
            return
        
        try:
            # Format the record
            message = self.format(record)
            
            # Send to all clients (in a separate thread to avoid blocking)
            threading.Thread(
                target=self._send_to_clients,
                args=(message,),
                daemon=True
            ).start()
            
        except Exception:
            self.handleError(record)
    
    def _send_to_clients(self, message: str):
        """Send message to all WebSocket clients"""
        with self.client_lock:
            disconnected_clients = set()
            
            for client in self.clients:
                try:
                    # This would need to be adapted based on your WebSocket implementation
                    # For example, with FastAPI WebSockets:
                    # asyncio.create_task(client.send_text(message))
                    pass
                except Exception:
                    # Mark client for removal
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected_clients
