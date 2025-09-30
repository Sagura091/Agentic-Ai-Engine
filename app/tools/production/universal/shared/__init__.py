"""
Shared utilities for Revolutionary Universal Tools

This module contains common base classes, error handlers, validators,
and utilities used across all Universal Tools.
"""

from .base_universal_tool import BaseUniversalTool
from .error_handlers import (
    UniversalToolError,
    FileOperationError,
    ValidationError,
    ConversionError,
    PermissionError as UniversalPermissionError,
)
from .validators import UniversalToolValidator
from .utils import (
    ensure_async,
    sanitize_path,
    validate_file_exists,
    get_file_extension,
    create_temp_file,
    cleanup_temp_files,
)

__all__ = [
    "BaseUniversalTool",
    "UniversalToolError",
    "FileOperationError",
    "ValidationError",
    "ConversionError",
    "UniversalPermissionError",
    "UniversalToolValidator",
    "ensure_async",
    "sanitize_path",
    "validate_file_exists",
    "get_file_extension",
    "create_temp_file",
    "cleanup_temp_files",
]

