"""
Validators for Revolutionary Universal Tools

Comprehensive validation system for all Universal Tools.
NO SHORTCUTS - Complete validation coverage.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union, Dict, Callable
from enum import Enum
import structlog

from .error_handlers import ValidationError, SecurityError

logger = structlog.get_logger(__name__)


class FileType(Enum):
    """Supported file types."""
    # Excel
    XLSX = ".xlsx"
    XLSM = ".xlsm"
    XLS = ".xls"
    XLSB = ".xlsb"
    CSV = ".csv"
    
    # Word
    DOCX = ".docx"
    DOC = ".doc"
    RTF = ".rtf"
    ODT = ".odt"
    
    # PDF
    PDF = ".pdf"
    
    # PowerPoint
    PPTX = ".pptx"
    PPT = ".ppt"
    
    # Images
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    GIF = ".gif"
    BMP = ".bmp"
    TIFF = ".tiff"
    WEBP = ".webp"
    
    # Video
    MP4 = ".mp4"
    AVI = ".avi"
    MOV = ".mov"
    MKV = ".mkv"
    WMV = ".wmv"
    FLV = ".flv"
    
    # Audio
    MP3 = ".mp3"
    WAV = ".wav"
    FLAC = ".flac"
    AAC = ".aac"
    OGG = ".ogg"
    M4A = ".m4a"
    
    # Data
    JSON = ".json"
    XML = ".xml"
    PARQUET = ".parquet"
    FEATHER = ".feather"
    HDF5 = ".h5"
    
    # Text
    TXT = ".txt"
    MD = ".md"
    HTML = ".html"


class UniversalToolValidator:
    """
    Comprehensive validator for Universal Tools.
    
    Provides validation for:
    - File paths and existence
    - File types and formats
    - Input parameters
    - Security constraints
    - Data integrity
    """
    
    # Security: Blocked file extensions
    BLOCKED_EXTENSIONS = {
        ".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".sh",
        ".ps1", ".vbs", ".js", ".jar", ".app", ".deb", ".rpm"
    }
    
    # Security: Blocked path patterns
    BLOCKED_PATH_PATTERNS = [
        r"\.\.[\\/]",  # Parent directory traversal
        r"~[\\/]",     # Home directory shortcuts
    ]
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        "excel": 100 * 1024 * 1024,      # 100 MB
        "word": 50 * 1024 * 1024,        # 50 MB
        "pdf": 100 * 1024 * 1024,        # 100 MB
        "image": 50 * 1024 * 1024,       # 50 MB
        "video": 500 * 1024 * 1024,      # 500 MB
        "audio": 100 * 1024 * 1024,      # 100 MB
        "data": 500 * 1024 * 1024,       # 500 MB
        "default": 100 * 1024 * 1024,    # 100 MB
    }
    
    @staticmethod
    def validate_file_path(
        file_path: Union[str, Path],
        must_exist: bool = False,
        allowed_extensions: Optional[List[str]] = None,
        check_security: bool = True,
    ) -> Path:
        """
        Validate file path.
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            allowed_extensions: List of allowed file extensions
            check_security: Whether to perform security checks
        
        Returns:
            Validated Path object
        
        Raises:
            ValidationError: If validation fails
            SecurityError: If security check fails
        """
        try:
            # Convert to Path object
            path = Path(file_path) if isinstance(file_path, str) else file_path
            
            # Security checks
            if check_security:
                UniversalToolValidator._check_path_security(path)
            
            # Check existence
            if must_exist and not path.exists():
                raise ValidationError(
                    f"File does not exist: {path}",
                    field_name="file_path",
                    invalid_value=str(path),
                    recovery_suggestion="Ensure the file path is correct and the file exists"
                )
            
            # Check extension
            if allowed_extensions:
                ext = path.suffix.lower()
                if ext not in [e.lower() for e in allowed_extensions]:
                    raise ValidationError(
                        f"Invalid file extension: {ext}. Allowed: {allowed_extensions}",
                        field_name="file_path",
                        invalid_value=str(path),
                        expected_type=f"One of {allowed_extensions}",
                        recovery_suggestion=f"Use a file with one of these extensions: {allowed_extensions}"
                    )
            
            logger.debug("File path validated", path=str(path))
            return path
            
        except (ValidationError, SecurityError):
            raise
        except Exception as e:
            raise ValidationError(
                f"Failed to validate file path: {str(e)}",
                field_name="file_path",
                invalid_value=str(file_path),
                original_exception=e
            )
    
    @staticmethod
    def _check_path_security(path: Path) -> None:
        """
        Check path for security violations.
        
        Args:
            path: Path to check
        
        Raises:
            SecurityError: If security violation detected
        """
        path_str = str(path)
        
        # Check blocked extensions
        if path.suffix.lower() in UniversalToolValidator.BLOCKED_EXTENSIONS:
            raise SecurityError(
                f"Blocked file extension: {path.suffix}",
                violation_type="blocked_extension",
                context={"path": path_str, "extension": path.suffix},
                recovery_suggestion="This file type is not allowed for security reasons"
            )
        
        # Check blocked path patterns
        for pattern in UniversalToolValidator.BLOCKED_PATH_PATTERNS:
            if re.search(pattern, path_str):
                raise SecurityError(
                    f"Blocked path pattern detected: {pattern}",
                    violation_type="blocked_path_pattern",
                    context={"path": path_str, "pattern": pattern},
                    recovery_suggestion="Use relative paths within the workspace"
                )
    
    @staticmethod
    def validate_file_size(
        file_path: Union[str, Path],
        max_size: Optional[int] = None,
        file_category: str = "default",
    ) -> int:
        """
        Validate file size.
        
        Args:
            file_path: Path to file
            max_size: Maximum allowed size in bytes (overrides category default)
            file_category: Category for default size limit
        
        Returns:
            File size in bytes
        
        Raises:
            ValidationError: If file is too large
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise ValidationError(
                    f"File does not exist: {path}",
                    field_name="file_path",
                    invalid_value=str(path)
                )
            
            file_size = path.stat().st_size
            max_allowed = max_size or UniversalToolValidator.MAX_FILE_SIZES.get(
                file_category,
                UniversalToolValidator.MAX_FILE_SIZES["default"]
            )
            
            if file_size > max_allowed:
                raise ValidationError(
                    f"File too large: {file_size} bytes (max: {max_allowed} bytes)",
                    field_name="file_size",
                    invalid_value=file_size,
                    expected_type=f"<= {max_allowed} bytes",
                    recovery_suggestion=f"Use a file smaller than {max_allowed / (1024*1024):.1f} MB"
                )
            
            logger.debug("File size validated", path=str(path), size=file_size)
            return file_size
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Failed to validate file size: {str(e)}",
                field_name="file_size",
                original_exception=e
            )
    
    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any],
        required_fields: List[str],
    ) -> None:
        """
        Validate that required fields are present.
        
        Args:
            data: Data dictionary to validate
            required_fields: List of required field names
        
        Raises:
            ValidationError: If required fields are missing
        """
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {missing_fields}",
                field_name="required_fields",
                invalid_value=list(data.keys()),
                expected_type=f"Must include: {required_fields}",
                recovery_suggestion=f"Provide values for: {missing_fields}"
            )
    
    @staticmethod
    def validate_type(
        value: Any,
        expected_type: type,
        field_name: str,
    ) -> None:
        """
        Validate value type.
        
        Args:
            value: Value to validate
            expected_type: Expected type
            field_name: Name of field being validated
        
        Raises:
            ValidationError: If type is incorrect
        """
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Invalid type for {field_name}: {type(value).__name__}",
                field_name=field_name,
                invalid_value=str(value),
                expected_type=expected_type.__name__,
                recovery_suggestion=f"Provide a {expected_type.__name__} value"
            )
    
    @staticmethod
    def validate_range(
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        field_name: str = "value",
    ) -> None:
        """
        Validate numeric range.
        
        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Name of field being validated
        
        Raises:
            ValidationError: If value is out of range
        """
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{field_name} is below minimum: {value} < {min_value}",
                field_name=field_name,
                invalid_value=value,
                expected_type=f">= {min_value}",
                recovery_suggestion=f"Use a value >= {min_value}"
            )
        
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{field_name} is above maximum: {value} > {max_value}",
                field_name=field_name,
                invalid_value=value,
                expected_type=f"<= {max_value}",
                recovery_suggestion=f"Use a value <= {max_value}"
            )

