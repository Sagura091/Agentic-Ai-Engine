"""
Comprehensive test suite for File System Operations Tool.

Tests all functionality including security, performance, and edge cases.
"""

import pytest
import asyncio
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.tools.production.file_system_tool import (
    FileSystemTool, 
    FileOperation, 
    CompressionFormat,
    FileSystemInput
)


class TestFileSystemTool:
    """Test suite for File System Operations Tool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance for testing."""
        return FileSystemTool()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_create_file_operation(self, tool):
        """Test file creation functionality."""
        result = await tool._run(
            operation=FileOperation.CREATE,
            path="test_files/example.txt",
            content="Hello, World!"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["operation"] == "create"
        assert "file_info" in result_dict
        
        # Verify file exists
        file_path = tool.sandbox_root / "test_files/example.txt"
        assert file_path.exists()
        assert file_path.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_create_directory_operation(self, tool):
        """Test directory creation functionality."""
        result = await tool._run(
            operation=FileOperation.CREATE,
            path="test_dirs/new_directory"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        
        # Verify directory exists
        dir_path = tool.sandbox_root / "test_dirs/new_directory"
        assert dir_path.exists()
        assert dir_path.is_dir()

    @pytest.mark.asyncio
    async def test_read_file_operation(self, tool):
        """Test file reading functionality."""
        # First create a file
        test_content = "Test content for reading"
        await tool._run(
            operation=FileOperation.CREATE,
            path="read_test.txt",
            content=test_content
        )
        
        # Then read it
        result = await tool._run(
            operation=FileOperation.READ,
            path="read_test.txt"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["content"] == test_content

    @pytest.mark.asyncio
    async def test_write_file_operation(self, tool):
        """Test file writing functionality."""
        test_content = "New content written to file"
        result = await tool._run(
            operation=FileOperation.WRITE,
            path="write_test.txt",
            content=test_content,
            overwrite=True
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["bytes_written"] == len(test_content.encode('utf-8'))

    @pytest.mark.asyncio
    async def test_delete_file_operation(self, tool):
        """Test file deletion functionality."""
        # Create file first
        await tool._run(
            operation=FileOperation.CREATE,
            path="delete_test.txt",
            content="To be deleted"
        )
        
        # Delete it
        result = await tool._run(
            operation=FileOperation.DELETE,
            path="delete_test.txt"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["total_deleted"] == 1
        
        # Verify file is gone
        file_path = tool.sandbox_root / "delete_test.txt"
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_copy_file_operation(self, tool):
        """Test file copying functionality."""
        # Create source file
        source_content = "Content to copy"
        await tool._run(
            operation=FileOperation.CREATE,
            path="source.txt",
            content=source_content
        )
        
        # Copy it
        result = await tool._run(
            operation=FileOperation.COPY,
            path="source.txt",
            destination="destination.txt"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["total_copied"] == 1
        
        # Verify both files exist
        source_path = tool.sandbox_root / "source.txt"
        dest_path = tool.sandbox_root / "destination.txt"
        assert source_path.exists()
        assert dest_path.exists()
        assert dest_path.read_text() == source_content

    @pytest.mark.asyncio
    async def test_move_file_operation(self, tool):
        """Test file moving functionality."""
        # Create source file
        source_content = "Content to move"
        await tool._run(
            operation=FileOperation.CREATE,
            path="move_source.txt",
            content=source_content
        )
        
        # Move it
        result = await tool._run(
            operation=FileOperation.MOVE,
            path="move_source.txt",
            destination="moved_file.txt"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        
        # Verify source is gone and destination exists
        source_path = tool.sandbox_root / "move_source.txt"
        dest_path = tool.sandbox_root / "moved_file.txt"
        assert not source_path.exists()
        assert dest_path.exists()
        assert dest_path.read_text() == source_content

    @pytest.mark.asyncio
    async def test_compress_operation(self, tool):
        """Test file compression functionality."""
        # Create test files
        await tool._run(
            operation=FileOperation.CREATE,
            path="compress_test/file1.txt",
            content="File 1 content"
        )
        await tool._run(
            operation=FileOperation.CREATE,
            path="compress_test/file2.txt",
            content="File 2 content"
        )
        
        # Compress directory
        result = await tool._run(
            operation=FileOperation.COMPRESS,
            path="compress_test",
            destination="test_archive.zip",
            compression_format=CompressionFormat.ZIP
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["compression_format"] == CompressionFormat.ZIP
        
        # Verify archive exists
        archive_path = tool.sandbox_root / "test_archive.zip"
        assert archive_path.exists()

    @pytest.mark.asyncio
    async def test_search_operation(self, tool):
        """Test file search functionality."""
        # Create test files
        await tool._run(
            operation=FileOperation.CREATE,
            path="search_test/python_file.py",
            content="print('Hello')"
        )
        await tool._run(
            operation=FileOperation.CREATE,
            path="search_test/text_file.txt",
            content="Some text"
        )
        await tool._run(
            operation=FileOperation.CREATE,
            path="search_test/another_python.py",
            content="import os"
        )
        
        # Search for Python files
        result = await tool._run(
            operation=FileOperation.SEARCH,
            path="search_test",
            pattern=r".*\.py$",
            recursive=True
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["total_found"] == 2
        
        # Verify found files are Python files
        found_files = [item["name"] for item in result_dict["found_items"]]
        assert "python_file.py" in found_files
        assert "another_python.py" in found_files
        assert "text_file.txt" not in found_files

    @pytest.mark.asyncio
    async def test_list_operation(self, tool):
        """Test directory listing functionality."""
        # Create test structure
        await tool._run(
            operation=FileOperation.CREATE,
            path="list_test/file1.txt",
            content="Content 1"
        )
        await tool._run(
            operation=FileOperation.CREATE,
            path="list_test/subdir/file2.txt",
            content="Content 2"
        )
        
        # List directory
        result = await tool._run(
            operation=FileOperation.LIST,
            path="list_test",
            recursive=True
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        assert result_dict["total_items"] >= 2
        assert result_dict["files_count"] >= 1
        assert result_dict["directories_count"] >= 1

    @pytest.mark.asyncio
    async def test_info_operation(self, tool):
        """Test file info functionality."""
        # Create test file
        test_content = "Test content for info"
        await tool._run(
            operation=FileOperation.CREATE,
            path="info_test.txt",
            content=test_content
        )
        
        # Get file info
        result = await tool._run(
            operation=FileOperation.INFO,
            path="info_test.txt"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is True
        
        file_info = result_dict["file_info"]
        assert file_info["name"] == "info_test.txt"
        assert file_info["size"] == len(test_content.encode('utf-8'))
        assert not file_info["is_directory"]
        assert file_info["checksum"] is not None

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, tool):
        """Test security against path traversal attacks."""
        with pytest.raises(ValueError, match="path traversal detected"):
            await tool._run(
                operation=FileOperation.CREATE,
                path="../../../etc/passwd",
                content="malicious content"
            )

    @pytest.mark.asyncio
    async def test_file_size_validation(self, tool):
        """Test file size limits."""
        large_content = "x" * (101 * 1024 * 1024)  # 101MB
        
        result = await tool._run(
            operation=FileOperation.WRITE,
            path="large_file.txt",
            content=large_content,
            max_size=100*1024*1024  # 100MB limit
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is False
        assert "too large" in result_dict["error"].lower()

    @pytest.mark.asyncio
    async def test_performance_metrics(self, tool):
        """Test performance metrics tracking."""
        # Perform several operations
        for i in range(3):
            await tool._run(
                operation=FileOperation.CREATE,
                path=f"perf_test_{i}.txt",
                content=f"Content {i}"
            )
        
        # Check metrics are updated
        assert tool._operation_count >= 3
        assert tool._success_count >= 3
        assert tool._total_execution_time > 0
        assert tool._last_used is not None

    @pytest.mark.asyncio
    async def test_error_handling(self, tool):
        """Test error handling for invalid operations."""
        # Try to read non-existent file
        result = await tool._run(
            operation=FileOperation.READ,
            path="non_existent_file.txt"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is False
        assert "not found" in result_dict["error"].lower()

    @pytest.mark.asyncio
    async def test_atomic_operations(self, tool):
        """Test atomic operations and rollback capability."""
        # This would be more complex in a real implementation
        # For now, test that operations either fully succeed or fully fail
        
        # Try to copy to an invalid destination
        await tool._run(
            operation=FileOperation.CREATE,
            path="atomic_test.txt",
            content="test content"
        )
        
        result = await tool._run(
            operation=FileOperation.COPY,
            path="atomic_test.txt",
            destination="invalid/../../path.txt"
        )
        
        result_dict = eval(result)
        assert result_dict["success"] is False
        
        # Original file should still exist
        original_path = tool.sandbox_root / "atomic_test.txt"
        assert original_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
