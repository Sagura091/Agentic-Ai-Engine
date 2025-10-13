"""
Utilities for Revolutionary Universal Tools

Comprehensive utility functions for all Universal Tools.
NO SHORTCUTS - Complete utility coverage.
"""

import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Union, List, Dict
from functools import wraps

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from .error_handlers import FileOperationError, ValidationError

logger = get_logger()


# Global temp file registry for cleanup
_TEMP_FILES_REGISTRY: List[Path] = []


def ensure_async(func: Callable) -> Callable:
    """
    Decorator to ensure a function runs asynchronously.
    
    If the function is already async, returns it as-is.
    If the function is sync, wraps it to run in executor.
    
    Args:
        func: Function to wrap
    
    Returns:
        Async function
    """
    if asyncio.iscoroutinefunction(func):
        return func
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return async_wrapper


def sanitize_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Sanitize and normalize a file path.
    
    Args:
        path: Path to sanitize
        base_dir: Base directory to resolve relative paths against
    
    Returns:
        Sanitized Path object
    
    Raises:
        ValidationError: If path is invalid
    """
    try:
        # Convert to Path
        p = Path(path) if isinstance(path, str) else path
        
        # Resolve relative to base_dir if provided
        if base_dir:
            base = Path(base_dir) if isinstance(base_dir, str) else base_dir
            if not p.is_absolute():
                p = base / p
        
        # Normalize path (resolve .., ., etc.)
        p = p.resolve()
        
        logger.debug(
            "Path sanitized",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.production.universal.shared.utils",
            data={"original": str(path), "sanitized": str(p)}
        )
        return p
        
    except Exception as e:
        raise ValidationError(
            f"Failed to sanitize path: {str(e)}",
            field_name="path",
            invalid_value=str(path),
            original_exception=e
        )


def validate_file_exists(
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
) -> Path:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to file
        file_type: Optional file type description for error messages
    
    Returns:
        Path object
    
    Raises:
        FileOperationError: If file doesn't exist
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    if not path.exists():
        file_desc = f"{file_type} file" if file_type else "File"
        raise FileOperationError(
            f"{file_desc} does not exist: {path}",
            file_path=str(path),
            operation="validate_exists",
            recovery_suggestion="Ensure the file path is correct and the file exists"
        )
    
    if not path.is_file():
        raise FileOperationError(
            f"Path is not a file: {path}",
            file_path=str(path),
            operation="validate_is_file",
            recovery_suggestion="Provide a path to a file, not a directory"
        )
    
    return path


def get_file_extension(
    file_path: Union[str, Path],
    lowercase: bool = True,
) -> str:
    """
    Get file extension from path.
    
    Args:
        file_path: Path to file
        lowercase: Whether to return lowercase extension
    
    Returns:
        File extension (including dot)
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    ext = path.suffix
    return ext.lower() if lowercase else ext


def create_temp_file(
    suffix: Optional[str] = None,
    prefix: Optional[str] = "universal_tool_",
    dir: Optional[Union[str, Path]] = None,
    delete: bool = False,
) -> Path:
    """
    Create a temporary file.
    
    Args:
        suffix: File suffix/extension
        prefix: File prefix
        dir: Directory to create file in
        delete: Whether to auto-delete on close
    
    Returns:
        Path to temporary file
    """
    try:
        # Create temp file
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=str(dir) if dir else None,
        )
        
        # Close file descriptor
        os.close(fd)
        
        # Convert to Path
        path = Path(temp_path)
        
        # Register for cleanup if not auto-deleting
        if not delete:
            _TEMP_FILES_REGISTRY.append(path)
        
        logger.debug(
            "Temporary file created",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.production.universal.shared.utils",
            data={"path": str(path)}
        )
        return path
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to create temporary file: {str(e)}",
            operation="create_temp_file",
            original_exception=e
        )


def create_temp_dir(
    suffix: Optional[str] = None,
    prefix: Optional[str] = "universal_tool_",
    dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a temporary directory.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        dir: Parent directory
    
    Returns:
        Path to temporary directory
    """
    try:
        temp_dir = tempfile.mkdtemp(
            suffix=suffix,
            prefix=prefix,
            dir=str(dir) if dir else None,
        )
        
        path = Path(temp_dir)
        _TEMP_FILES_REGISTRY.append(path)
        
        logger.debug(
            "Temporary directory created",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.production.universal.shared.utils",
            data={"path": str(path)}
        )
        return path
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to create temporary directory: {str(e)}",
            operation="create_temp_dir",
            original_exception=e
        )


def cleanup_temp_files() -> int:
    """
    Clean up all registered temporary files and directories.
    
    Returns:
        Number of items cleaned up
    """
    cleaned = 0
    
    for path in _TEMP_FILES_REGISTRY[:]:  # Copy list to avoid modification during iteration
        try:
            if path.exists():
                if path.is_file():
                    path.unlink()
                    logger.debug(
                        "Temporary file deleted",
                        LogCategory.TOOL_OPERATIONS,
                        "app.tools.production.universal.shared.utils",
                        data={"path": str(path)}
                    )
                elif path.is_dir():
                    shutil.rmtree(path)
                    logger.debug(
                        "Temporary directory deleted",
                        LogCategory.TOOL_OPERATIONS,
                        "app.tools.production.universal.shared.utils",
                        data={"path": str(path)}
                    )
                cleaned += 1
            
            _TEMP_FILES_REGISTRY.remove(path)
            
        except Exception as e:
            logger.warn(
                "Failed to clean up temporary file",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.utils",
                data={"path": str(path)},
                error=e
            )
    
    if cleaned > 0:
        logger.info(
            f"Cleaned up {cleaned} temporary files/directories",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.production.universal.shared.utils",
            data={"cleaned_count": cleaned}
        )
    
    return cleaned


def ensure_directory_exists(
    dir_path: Union[str, Path],
    create: bool = True,
) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        dir_path: Path to directory
        create: Whether to create if doesn't exist
    
    Returns:
        Path object
    
    Raises:
        FileOperationError: If directory doesn't exist and create=False
    """
    path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    
    if not path.exists():
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(
                    "Directory created",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.production.universal.shared.utils",
                    data={"path": str(path)}
                )
            except Exception as e:
                raise FileOperationError(
                    f"Failed to create directory: {str(e)}",
                    file_path=str(path),
                    operation="create_directory",
                    original_exception=e
                )
        else:
            raise FileOperationError(
                f"Directory does not exist: {path}",
                file_path=str(path),
                operation="validate_directory_exists",
                recovery_suggestion="Create the directory or set create=True"
            )
    
    if not path.is_dir():
        raise FileOperationError(
            f"Path is not a directory: {path}",
            file_path=str(path),
            operation="validate_is_directory",
            recovery_suggestion="Provide a path to a directory, not a file"
        )
    
    return path


def safe_file_copy(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Safely copy a file.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to destination file
    
    Raises:
        FileOperationError: If copy fails
    """
    src = Path(source) if isinstance(source, str) else source
    dst = Path(destination) if isinstance(destination, str) else destination
    
    # Validate source exists
    validate_file_exists(src)
    
    # Check if destination exists
    if dst.exists() and not overwrite:
        raise FileOperationError(
            f"Destination file already exists: {dst}",
            file_path=str(dst),
            operation="copy_file",
            recovery_suggestion="Set overwrite=True or use a different destination"
        )
    
    try:
        # Ensure destination directory exists
        ensure_directory_exists(dst.parent)
        
        # Copy file
        shutil.copy2(src, dst)
        logger.debug(
            "File copied",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.production.universal.shared.utils",
            data={"source": str(src), "destination": str(dst)}
        )
        
        return dst
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to copy file: {str(e)}",
            file_path=str(src),
            operation="copy_file",
            context={"destination": str(dst)},
            original_exception=e
        )


def get_file_size_human(size_bytes: int) -> str:
    """
    Convert file size to human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

