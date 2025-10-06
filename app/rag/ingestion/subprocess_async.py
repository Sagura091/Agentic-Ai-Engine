"""
Safe async subprocess execution with security controls.

This module provides secure subprocess execution with:
- Path validation and sanitization
- Command allowlisting
- Timeout enforcement
- Resource limits
- Proper cleanup
"""

import asyncio
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import structlog

from .safe_ops import with_timeout

logger = structlog.get_logger(__name__)


@dataclass
class SubprocessResult:
    """Result from subprocess execution."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: Optional[str] = None


class SafeSubprocess:
    """
    Safe subprocess executor with security controls.
    
    Provides secure subprocess execution with:
    - Command allowlisting (only approved commands)
    - Path validation (prevent path traversal)
    - Timeout enforcement
    - Resource limits
    - Proper cleanup
    """
    
    # Allowed commands (allowlist approach)
    ALLOWED_COMMANDS = {
        'ffmpeg',
        'ffprobe',
        'convert',  # ImageMagick
        'identify',  # ImageMagick
        'tesseract',
        'pdftotext',
        'pdfinfo',
        '7z',
        'unzip',
        'tar',
    }
    
    def __init__(self, 
                 allowed_commands: Optional[set] = None,
                 max_output_size: int = 100 * 1024 * 1024):  # 100MB
        """
        Initialize safe subprocess executor.
        
        Args:
            allowed_commands: Set of allowed command names (defaults to ALLOWED_COMMANDS)
            max_output_size: Maximum output size in bytes
        """
        self.allowed_commands = allowed_commands or self.ALLOWED_COMMANDS
        self.max_output_size = max_output_size
        
        logger.info(
            "SafeSubprocess initialized",
            allowed_commands=list(self.allowed_commands),
            max_output_size_mb=max_output_size / (1024 * 1024)
        )
    
    def _validate_command(self, command: str) -> bool:
        """
        Validate that command is in allowlist.
        
        Args:
            command: Command name
            
        Returns:
            True if allowed
        """
        # Extract base command name
        cmd_name = Path(command).name
        
        # Check against allowlist
        if cmd_name not in self.allowed_commands:
            logger.error(
                "Command not in allowlist",
                command=cmd_name,
                allowed=list(self.allowed_commands)
            )
            return False
        
        return True
    
    def _validate_path(self, path: str) -> bool:
        """
        Validate file path for security.
        
        Args:
            path: File path
            
        Returns:
            True if path is safe
        """
        # Check for path traversal patterns
        if '..' in path:
            logger.error("Path traversal detected", path=path)
            return False
        
        # Convert to absolute path and check it exists
        try:
            abs_path = Path(path).resolve()
            
            # Path should exist (for input files)
            # For output files, parent directory should exist
            if not abs_path.exists() and not abs_path.parent.exists():
                logger.error("Path does not exist", path=str(abs_path))
                return False
            
            return True
            
        except Exception as e:
            logger.error("Path validation failed", path=path, error=str(e))
            return False
    
    def _sanitize_args(self, args: List[str]) -> Tuple[bool, List[str]]:
        """
        Sanitize command arguments.
        
        Args:
            args: Command arguments
            
        Returns:
            Tuple of (is_valid, sanitized_args)
        """
        if not args:
            return False, []
        
        # Validate command
        if not self._validate_command(args[0]):
            return False, []
        
        sanitized = [args[0]]
        
        # Process remaining arguments
        for i, arg in enumerate(args[1:], 1):
            # Skip flags and options
            if arg.startswith('-'):
                sanitized.append(arg)
                continue
            
            # Check if this looks like a file path
            # (heuristic: contains path separators or file extensions)
            if '/' in arg or '\\' in arg or '.' in arg:
                # Validate path
                if not self._validate_path(arg):
                    logger.error("Invalid path in arguments", arg=arg, position=i)
                    return False, []
            
            sanitized.append(arg)
        
        return True, sanitized
    
    async def run(self,
                  args: List[str],
                  timeout: float = 60.0,
                  cwd: Optional[str] = None,
                  env: Optional[Dict[str, str]] = None,
                  check: bool = False) -> SubprocessResult:
        """
        Run command safely with timeout and validation.
        
        Args:
            args: Command and arguments as list
            timeout: Timeout in seconds
            cwd: Working directory
            env: Environment variables
            check: Raise exception on non-zero return code
            
        Returns:
            SubprocessResult
            
        Raises:
            ValueError: If command validation fails
            subprocess.CalledProcessError: If check=True and command fails
        """
        # Sanitize arguments
        is_valid, sanitized_args = self._sanitize_args(args)
        if not is_valid:
            raise ValueError(f"Command validation failed: {args[0]}")
        
        logger.debug(
            "Running subprocess",
            command=sanitized_args[0],
            args_count=len(sanitized_args),
            timeout=timeout
        )
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *sanitized_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )
            
            # Wait for completion with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                # Check output size
                if len(stdout_bytes) > self.max_output_size:
                    logger.warning(
                        "Subprocess output truncated",
                        size=len(stdout_bytes),
                        max_size=self.max_output_size
                    )
                    stdout_bytes = stdout_bytes[:self.max_output_size]
                
                # Decode output
                stdout = stdout_bytes.decode('utf-8', errors='replace')
                stderr = stderr_bytes.decode('utf-8', errors='replace')
                
                result = SubprocessResult(
                    returncode=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=False
                )
                
                if check and process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode,
                        sanitized_args,
                        stdout,
                        stderr
                    )
                
                logger.debug(
                    "Subprocess completed",
                    command=sanitized_args[0],
                    returncode=process.returncode,
                    stdout_size=len(stdout),
                    stderr_size=len(stderr)
                )
                
                return result
                
            except asyncio.TimeoutError:
                # Kill process on timeout
                logger.error(
                    "Subprocess timed out, killing",
                    command=sanitized_args[0],
                    timeout=timeout
                )
                
                try:
                    process.kill()
                    await process.wait()
                except Exception as e:
                    logger.warning("Failed to kill process", error=str(e))
                
                return SubprocessResult(
                    returncode=-1,
                    stdout="",
                    stderr=f"Process timed out after {timeout} seconds",
                    timed_out=True,
                    error="timeout"
                )
                
        except Exception as e:
            logger.error(
                "Subprocess execution failed",
                command=sanitized_args[0],
                error=str(e)
            )
            
            return SubprocessResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
                timed_out=False,
                error=str(e)
            )


# Global instance
_safe_subprocess: Optional[SafeSubprocess] = None
_subprocess_lock = asyncio.Lock()


async def get_safe_subprocess() -> SafeSubprocess:
    """
    Get global SafeSubprocess instance (singleton).
    
    Returns:
        SafeSubprocess instance
    """
    global _safe_subprocess
    
    if _safe_subprocess is None:
        async with _subprocess_lock:
            if _safe_subprocess is None:
                _safe_subprocess = SafeSubprocess()
    
    return _safe_subprocess


async def run_command(args: List[str], 
                     timeout: float = 60.0,
                     **kwargs) -> SubprocessResult:
    """
    Convenience function to run command safely.
    
    Args:
        args: Command and arguments
        timeout: Timeout in seconds
        **kwargs: Additional arguments for SafeSubprocess.run()
        
    Returns:
        SubprocessResult
    """
    subprocess_runner = await get_safe_subprocess()
    return await subprocess_runner.run(args, timeout=timeout, **kwargs)

