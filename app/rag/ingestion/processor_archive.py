"""
Archive processor for ZIP, TAR, 7Z, RAR files.

This module provides comprehensive archive processing:
- Recursive extraction
- Multiple format support (ZIP, TAR, GZ, BZ2, XZ, 7Z, RAR)
- Depth limiting
- Size limiting
- Malicious archive detection (zip bombs)
- Nested archive handling
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import tempfile
import zipfile
import tarfile
import shutil
import io

import structlog
import aiofiles

from .models_result import ProcessResult, ProcessorError, ErrorCode, ProcessingStage
from .subprocess_async import run_command
from .dependencies import get_dependency_checker

logger = structlog.get_logger(__name__)


class ArchiveProcessor:
    """
    Comprehensive archive processor with security controls.
    
    Features:
    - Multiple format support
    - Recursive extraction
    - Zip bomb detection
    - Size limits
    - Depth limits
    - File filtering
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        max_total_size: int = 1024 * 1024 * 1024,  # 1GB
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        compression_ratio_threshold: float = 100.0
    ):
        """
        Initialize archive processor.
        
        Args:
            max_depth: Maximum extraction depth
            max_total_size: Maximum total extracted size
            max_file_size: Maximum individual file size
            compression_ratio_threshold: Zip bomb detection threshold
        """
        self.max_depth = max_depth
        self.max_total_size = max_total_size
        self.max_file_size = max_file_size
        self.compression_ratio_threshold = compression_ratio_threshold
        
        self.dep_checker = get_dependency_checker()
        
        logger.info(
            "ArchiveProcessor initialized",
            max_depth=max_depth,
            max_total_size=max_total_size,
            max_file_size=max_file_size
        )
    
    async def process(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessResult:
        """
        Process archive file.
        
        Args:
            content: Archive file content
            filename: Filename
            metadata: Additional metadata
            
        Returns:
            ProcessResult with extracted file list
        """
        start_time = datetime.utcnow()
        errors = []
        
        try:
            # Detect archive type
            archive_type = self._detect_archive_type(filename, content)
            
            if not archive_type:
                return ProcessResult(
                    text="",
                    metadata=metadata or {},
                    errors=[ProcessorError(
                        code=ErrorCode.UNSUPPORTED_FORMAT,
                        message=f"Unsupported archive format: {filename}",
                        stage=ProcessingStage.EXTRACTION,
                        retriable=False
                    )],
                    processor_name="ArchiveProcessor",
                    processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
            
            # Check for zip bomb
            if await self._is_zip_bomb(content, archive_type):
                return ProcessResult(
                    text="",
                    metadata=metadata or {},
                    errors=[ProcessorError(
                        code=ErrorCode.SECURITY_VIOLATION,
                        message="Potential zip bomb detected",
                        stage=ProcessingStage.VALIDATION,
                        retriable=False
                    )],
                    processor_name="ArchiveProcessor",
                    processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
            
            # Extract archive
            extraction_result = await self._extract_archive(
                content,
                filename,
                archive_type,
                depth=0
            )
            
            # Build text summary
            file_list = extraction_result["files"]
            text_parts = [f"Archive: {filename}"]
            text_parts.append(f"Type: {archive_type}")
            text_parts.append(f"Files: {len(file_list)}")
            text_parts.append("\nContents:")
            
            for file_info in file_list[:100]:  # Limit to first 100 files
                text_parts.append(f"  - {file_info['path']} ({file_info['size']} bytes)")
            
            if len(file_list) > 100:
                text_parts.append(f"  ... and {len(file_list) - 100} more files")
            
            result_metadata = {
                **(metadata or {}),
                "archive_type": archive_type,
                "total_files": len(file_list),
                "total_size": extraction_result["total_size"],
                "extraction_depth": extraction_result["max_depth"],
                "files": file_list
            }
            
            return ProcessResult(
                text="\n".join(text_parts),
                metadata=result_metadata,
                errors=errors,
                processor_name="ArchiveProcessor",
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error("Archive processing failed", error=str(e), filename=filename)
            
            return ProcessResult(
                text="",
                metadata=metadata or {},
                errors=[ProcessorError(
                    code=ErrorCode.PROCESSING_FAILED,
                    message=f"Archive processing failed: {str(e)}",
                    stage=ProcessingStage.EXTRACTION,
                    retriable=True,
                    details={"error": str(e)}
                )],
                processor_name="ArchiveProcessor",
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def _detect_archive_type(self, filename: str, content: bytes) -> Optional[str]:
        """Detect archive type from filename and magic bytes."""
        # Check magic bytes
        if content.startswith(b'PK\x03\x04') or content.startswith(b'PK\x05\x06'):
            return 'zip'
        elif content.startswith(b'\x1f\x8b'):
            return 'gzip'
        elif content.startswith(b'BZh'):
            return 'bzip2'
        elif content.startswith(b'\xfd7zXZ\x00'):
            return 'xz'
        elif content.startswith(b'7z\xbc\xaf\x27\x1c'):
            return '7z'
        elif content.startswith(b'Rar!\x1a\x07'):
            return 'rar'
        
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext == '.zip':
            return 'zip'
        elif ext in ['.tar', '.tar.gz', '.tgz']:
            return 'tar'
        elif ext == '.gz':
            return 'gzip'
        elif ext in ['.bz2', '.tar.bz2', '.tbz2']:
            return 'bzip2'
        elif ext in ['.xz', '.tar.xz', '.txz']:
            return 'xz'
        elif ext == '.7z':
            return '7z'
        elif ext == '.rar':
            return 'rar'
        
        return None
    
    async def _is_zip_bomb(self, content: bytes, archive_type: str) -> bool:
        """Check if archive is a zip bomb."""
        try:
            compressed_size = len(content)
            
            if archive_type == 'zip':
                # Quick check using zipfile
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    uncompressed_size = sum(info.file_size for info in zf.filelist)
                    
                    if uncompressed_size > self.max_total_size:
                        return True
                    
                    if compressed_size > 0:
                        ratio = uncompressed_size / compressed_size
                        if ratio > self.compression_ratio_threshold:
                            logger.warning(
                                "High compression ratio detected",
                                ratio=ratio,
                                threshold=self.compression_ratio_threshold
                            )
                            return True
            
            return False
            
        except Exception as e:
            logger.warning("Zip bomb check failed", error=str(e))
            return False
    
    async def _extract_archive(
        self,
        content: bytes,
        filename: str,
        archive_type: str,
        depth: int
    ) -> Dict[str, Any]:
        """Extract archive recursively."""
        if depth >= self.max_depth:
            logger.warning("Maximum extraction depth reached", depth=depth)
            return {"files": [], "total_size": 0, "max_depth": depth}
        
        files = []
        total_size = 0
        max_depth = depth
        
        # Create temp directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write archive to temp file
            archive_path = temp_path / filename
            async with aiofiles.open(archive_path, 'wb') as f:
                await f.write(content)
            
            # Extract based on type
            extract_dir = temp_path / "extracted"
            extract_dir.mkdir()
            
            try:
                if archive_type == 'zip':
                    await self._extract_zip(archive_path, extract_dir)
                elif archive_type in ['tar', 'gzip', 'bzip2', 'xz']:
                    await self._extract_tar(archive_path, extract_dir, archive_type)
                elif archive_type == '7z':
                    await self._extract_7z(archive_path, extract_dir)
                elif archive_type == 'rar':
                    await self._extract_rar(archive_path, extract_dir)
                else:
                    raise ValueError(f"Unsupported archive type: {archive_type}")
                
                # Process extracted files
                for file_path in extract_dir.rglob('*'):
                    if not file_path.is_file():
                        continue
                    
                    file_size = file_path.stat().st_size
                    
                    # Check size limits
                    if file_size > self.max_file_size:
                        logger.warning("File too large, skipping", path=str(file_path), size=file_size)
                        continue
                    
                    if total_size + file_size > self.max_total_size:
                        logger.warning("Total size limit reached")
                        break
                    
                    total_size += file_size
                    
                    # Get relative path
                    rel_path = file_path.relative_to(extract_dir)
                    
                    files.append({
                        "path": str(rel_path),
                        "size": file_size,
                        "depth": depth
                    })
                    
                    # Check if file is also an archive
                    if self._is_archive(file_path.name):
                        async with aiofiles.open(file_path, 'rb') as f:
                            nested_content = await f.read()
                        
                        nested_type = self._detect_archive_type(file_path.name, nested_content)
                        
                        if nested_type:
                            nested_result = await self._extract_archive(
                                nested_content,
                                file_path.name,
                                nested_type,
                                depth + 1
                            )
                            
                            files.extend(nested_result["files"])
                            total_size += nested_result["total_size"]
                            max_depth = max(max_depth, nested_result["max_depth"])
                
            except Exception as e:
                logger.error("Extraction failed", error=str(e), archive_type=archive_type)
                raise
        
        return {
            "files": files,
            "total_size": total_size,
            "max_depth": max_depth
        }
    
    async def _extract_zip(self, archive_path: Path, extract_dir: Path):
        """Extract ZIP archive."""
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_dir)
    
    async def _extract_tar(self, archive_path: Path, extract_dir: Path, archive_type: str):
        """Extract TAR archive."""
        mode = 'r'
        if archive_type == 'gzip':
            mode = 'r:gz'
        elif archive_type == 'bzip2':
            mode = 'r:bz2'
        elif archive_type == 'xz':
            mode = 'r:xz'
        
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(extract_dir)
    
    async def _extract_7z(self, archive_path: Path, extract_dir: Path):
        """Extract 7Z archive using 7z command."""
        result = await run_command(
            ['7z', 'x', str(archive_path), f'-o{extract_dir}', '-y'],
            timeout=300.0
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"7z extraction failed: {result.stderr}")
    
    async def _extract_rar(self, archive_path: Path, extract_dir: Path):
        """Extract RAR archive using unrar command."""
        result = await run_command(
            ['unrar', 'x', '-y', str(archive_path), str(extract_dir)],
            timeout=300.0
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"unrar extraction failed: {result.stderr}")
    
    def _is_archive(self, filename: str) -> bool:
        """Check if filename is an archive."""
        archive_extensions = {
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
            '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz'
        }
        
        name_lower = filename.lower()
        return any(name_lower.endswith(ext) for ext in archive_extensions)

