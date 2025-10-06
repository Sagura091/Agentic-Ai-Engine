"""
Security and validation layer for document intake.

This module provides comprehensive security validation including:
- Magic-number MIME type detection
- Extension denylist enforcement
- File size limits
- Path traversal prevention
- Optional antivirus scanning hook
"""

import os
import mimetypes
from pathlib import Path
from typing import Optional, Set, Callable, Awaitable
import tempfile
import shutil

import structlog

from .models_result import ValidationResult, ErrorCode, ProcessingStage

logger = structlog.get_logger(__name__)


# Try to import python-magic for MIME detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("python-magic not available, falling back to mimetypes")


class IntakeGuard:
    """
    Security guard for document intake.
    
    Validates files before processing to prevent:
    - Malicious file execution
    - Resource exhaustion
    - Path traversal attacks
    - Malware injection
    """
    
    # Default blocked extensions (executables and scripts)
    DEFAULT_BLOCKED_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',  # Windows executables
        '.js', '.jse', '.vbs', '.vbe', '.wsf', '.wsh',   # Scripts
        '.ps1', '.psm1', '.psd1',                         # PowerShell
        '.msi', '.msp', '.mst',                           # Installers
        '.app', '.deb', '.rpm',                           # Unix executables
        '.sh', '.bash', '.zsh', '.fish',                  # Shell scripts
        '.jar', '.war',                                   # Java archives (can contain code)
        '.dll', '.so', '.dylib',                          # Libraries
    }
    
    # Default allowed MIME types (can be overridden)
    DEFAULT_ALLOWED_MIME_PREFIXES = {
        'text/',
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument',
        'application/vnd.oasis.opendocument',
        'application/json',
        'application/xml',
        'application/zip',
        'application/x-7z-compressed',
        'application/x-tar',
        'application/gzip',
        'image/',
        'video/',
        'audio/',
    }
    
    def __init__(self,
                 max_file_size_mb: int = 500,
                 blocked_extensions: Optional[Set[str]] = None,
                 allowed_mime_prefixes: Optional[Set[str]] = None,
                 enable_magic_mime: bool = True,
                 av_scan_hook: Optional[Callable[[bytes], Awaitable[bool]]] = None,
                 temp_dir: Optional[str] = None):
        """
        Initialize intake guard.
        
        Args:
            max_file_size_mb: Maximum file size in megabytes
            blocked_extensions: Set of blocked file extensions (with dots)
            allowed_mime_prefixes: Set of allowed MIME type prefixes
            enable_magic_mime: Use python-magic for MIME detection if available
            av_scan_hook: Optional async function for antivirus scanning
            temp_dir: Temporary directory for sandboxed operations
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.blocked_extensions = blocked_extensions or self.DEFAULT_BLOCKED_EXTENSIONS
        self.allowed_mime_prefixes = allowed_mime_prefixes or self.DEFAULT_ALLOWED_MIME_PREFIXES
        self.enable_magic_mime = enable_magic_mime and MAGIC_AVAILABLE
        self.av_scan_hook = av_scan_hook
        
        # Set up sandboxed temp directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "rag_ingestion_sandbox"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "IntakeGuard initialized",
            max_file_size_mb=max_file_size_mb,
            blocked_extensions_count=len(self.blocked_extensions),
            magic_available=self.enable_magic_mime,
            av_enabled=av_scan_hook is not None
        )
    
    def _detect_mime_type(self, file_path: Optional[Path] = None, content: Optional[bytes] = None) -> Optional[str]:
        """
        Detect MIME type using magic numbers or file extension.
        
        Args:
            file_path: Path to file (for extension-based detection)
            content: File content (for magic-number detection)
            
        Returns:
            Detected MIME type or None
        """
        mime_type = None
        
        # Try magic-number detection first (most reliable)
        if self.enable_magic_mime and content:
            try:
                mime = magic.Magic(mime=True)
                mime_type = mime.from_buffer(content)
                logger.debug("MIME detected via magic", mime_type=mime_type)
            except Exception as e:
                logger.warning("Magic MIME detection failed", error=str(e))
        
        # Fall back to extension-based detection
        if not mime_type and file_path:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            logger.debug("MIME detected via extension", mime_type=mime_type, file_path=str(file_path))
        
        return mime_type
    
    def _check_extension(self, file_path: Path, result: ValidationResult) -> None:
        """
        Check if file extension is blocked.
        
        Args:
            file_path: Path to file
            result: ValidationResult to update
        """
        extension = file_path.suffix.lower()
        
        if extension in self.blocked_extensions:
            result.add_error(
                ErrorCode.BLOCKED_EXTENSION,
                f"File extension '{extension}' is blocked for security reasons",
                ProcessingStage.INTAKE,
                extension=extension,
                file_name=file_path.name
            )
            logger.warning("Blocked extension detected", extension=extension, file_name=file_path.name)
    
    def _check_mime_type(self, mime_type: Optional[str], result: ValidationResult) -> None:
        """
        Check if MIME type is allowed.
        
        Args:
            mime_type: Detected MIME type
            result: ValidationResult to update
        """
        if not mime_type:
            result.add_warning("Could not detect MIME type")
            return
        
        # Check if MIME type matches any allowed prefix
        allowed = any(mime_type.startswith(prefix) for prefix in self.allowed_mime_prefixes)
        
        if not allowed:
            result.add_error(
                ErrorCode.INVALID_MIME_TYPE,
                f"MIME type '{mime_type}' is not allowed",
                ProcessingStage.INTAKE,
                mime_type=mime_type
            )
            logger.warning("Disallowed MIME type", mime_type=mime_type)
    
    def _check_file_size(self, file_size: int, result: ValidationResult) -> None:
        """
        Check if file size is within limits.
        
        Args:
            file_size: File size in bytes
            result: ValidationResult to update
        """
        if file_size > self.max_file_size_bytes:
            result.add_error(
                ErrorCode.FILE_TOO_LARGE,
                f"File size {file_size} bytes exceeds limit of {self.max_file_size_bytes} bytes",
                ProcessingStage.INTAKE,
                file_size=file_size,
                max_size=self.max_file_size_bytes
            )
            logger.warning("File too large", file_size=file_size, max_size=self.max_file_size_bytes)
    
    def _check_path_traversal(self, file_path: Path, result: ValidationResult) -> None:
        """
        Check for path traversal attempts.
        
        Args:
            file_path: Path to check
            result: ValidationResult to update
        """
        # Check for path traversal patterns
        path_str = str(file_path)
        
        if '..' in path_str or path_str.startswith('/') or ':' in path_str:
            result.add_error(
                ErrorCode.INVALID_FORMAT,
                "Path contains potentially malicious patterns",
                ProcessingStage.INTAKE,
                path=path_str
            )
            logger.warning("Path traversal attempt detected", path=path_str)
    
    async def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a file from disk.
        
        Args:
            file_path: Path to file
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(valid=True)
        
        # Check if file exists
        if not file_path.exists():
            result.add_error(
                ErrorCode.FILE_NOT_FOUND,
                f"File not found: {file_path}",
                ProcessingStage.INTAKE,
                file_path=str(file_path)
            )
            return result
        
        # Check path traversal
        self._check_path_traversal(file_path, result)
        if not result.valid:
            return result
        
        # Get file size
        file_size = file_path.stat().st_size
        result.file_size = file_size
        
        # Check file size
        self._check_file_size(file_size, result)
        if not result.valid:
            return result
        
        # Check extension
        self._check_extension(file_path, result)
        if not result.valid:
            return result
        
        # Read file content for MIME detection
        try:
            with open(file_path, 'rb') as f:
                # Read first 8KB for magic number detection
                content_sample = f.read(8192)
        except Exception as e:
            result.add_error(
                ErrorCode.INTERNAL_ERROR,
                f"Failed to read file: {str(e)}",
                ProcessingStage.INTAKE,
                error=str(e)
            )
            return result
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_path=file_path, content=content_sample)
        result.mime_type = mime_type
        
        # Check MIME type
        self._check_mime_type(mime_type, result)
        if not result.valid:
            return result
        
        # Optional: AV scan
        if self.av_scan_hook:
            try:
                # Read full file for AV scan
                with open(file_path, 'rb') as f:
                    full_content = f.read()
                
                is_clean = await self.av_scan_hook(full_content)
                if not is_clean:
                    result.add_error(
                        ErrorCode.MALWARE_DETECTED,
                        "File failed antivirus scan",
                        ProcessingStage.INTAKE
                    )
                    logger.error("Malware detected", file_path=str(file_path))
            except Exception as e:
                result.add_warning(f"AV scan failed: {str(e)}")
                logger.warning("AV scan error", error=str(e))
        
        return result
    
    async def validate_content(self, content: bytes, file_name: str) -> ValidationResult:
        """
        Validate file content directly.
        
        Args:
            content: File content
            file_name: Original file name
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(valid=True)
        
        # Get file size
        file_size = len(content)
        result.file_size = file_size
        
        # Check file size
        self._check_file_size(file_size, result)
        if not result.valid:
            return result
        
        # Check extension from file name
        file_path = Path(file_name)
        self._check_extension(file_path, result)
        if not result.valid:
            return result
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_path=file_path, content=content)
        result.mime_type = mime_type
        
        # Check MIME type
        self._check_mime_type(mime_type, result)
        if not result.valid:
            return result
        
        # Optional: AV scan
        if self.av_scan_hook:
            try:
                is_clean = await self.av_scan_hook(content)
                if not is_clean:
                    result.add_error(
                        ErrorCode.MALWARE_DETECTED,
                        "Content failed antivirus scan",
                        ProcessingStage.INTAKE
                    )
                    logger.error("Malware detected in content", file_name=file_name)
            except Exception as e:
                result.add_warning(f"AV scan failed: {str(e)}")
                logger.warning("AV scan error", error=str(e))
        
        return result
    
    def get_sandbox_path(self, file_name: str) -> Path:
        """
        Get a sandboxed path for temporary file operations.
        
        Args:
            file_name: Original file name
            
        Returns:
            Path in sandbox directory
        """
        # Sanitize file name
        safe_name = "".join(c for c in file_name if c.isalnum() or c in ('_', '-', '.'))
        
        # Create unique path
        return self.temp_dir / f"{os.getpid()}_{safe_name}"
    
    def cleanup_sandbox(self) -> None:
        """Clean up sandbox directory."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Sandbox cleaned up", temp_dir=str(self.temp_dir))
        except Exception as e:
            logger.warning("Failed to clean up sandbox", error=str(e))

