"""
Dependency checker for optional document processing libraries.

This module provides graceful degradation when optional dependencies
are not available, with clear error messages and capability detection.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    available: bool
    version: Optional[str] = None
    import_error: Optional[str] = None
    required_for: List[str] = None
    
    def __post_init__(self):
        if self.required_for is None:
            self.required_for = []


class DependencyChecker:
    """
    Checks availability of optional dependencies.
    
    Provides graceful degradation and clear error messages when
    optional dependencies are not available.
    """
    
    def __init__(self):
        """Initialize dependency checker."""
        self.dependencies: Dict[str, DependencyInfo] = {}
        self._check_all_dependencies()
    
    def _check_dependency(self, 
                         name: str, 
                         import_name: str,
                         required_for: List[str]) -> DependencyInfo:
        """
        Check if a dependency is available.
        
        Args:
            name: Display name
            import_name: Import name
            required_for: List of features requiring this dependency
            
        Returns:
            DependencyInfo
        """
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            
            info = DependencyInfo(
                name=name,
                available=True,
                version=version,
                required_for=required_for
            )
            
            logger.debug(
                "Dependency available",
                name=name,
                version=version
            )
            
            return info
            
        except ImportError as e:
            info = DependencyInfo(
                name=name,
                available=False,
                import_error=str(e),
                required_for=required_for
            )
            
            logger.warning(
                "Optional dependency not available",
                name=name,
                required_for=required_for,
                error=str(e)
            )
            
            return info
    
    def _check_all_dependencies(self):
        """Check all optional dependencies."""
        
        # OCR engines
        self.dependencies['pytesseract'] = self._check_dependency(
            'Tesseract OCR',
            'pytesseract',
            ['Image OCR', 'PDF OCR', 'Video frame OCR']
        )
        
        self.dependencies['easyocr'] = self._check_dependency(
            'EasyOCR',
            'easyocr',
            ['Advanced image OCR', 'Multi-language OCR']
        )
        
        self.dependencies['paddleocr'] = self._check_dependency(
            'PaddleOCR',
            'paddleocr',
            ['Advanced OCR', 'Layout analysis']
        )
        
        # Document processing
        self.dependencies['pypdf'] = self._check_dependency(
            'PyPDF',
            'pypdf',
            ['PDF text extraction']
        )
        
        self.dependencies['python-docx'] = self._check_dependency(
            'python-docx',
            'docx',
            ['DOCX processing']
        )
        
        self.dependencies['openpyxl'] = self._check_dependency(
            'openpyxl',
            'openpyxl',
            ['Excel processing']
        )
        
        self.dependencies['python-pptx'] = self._check_dependency(
            'python-pptx',
            'pptx',
            ['PowerPoint processing']
        )
        
        # Audio/Video
        self.dependencies['speech_recognition'] = self._check_dependency(
            'SpeechRecognition',
            'speech_recognition',
            ['Audio transcription', 'Video audio extraction']
        )
        
        self.dependencies['pydub'] = self._check_dependency(
            'pydub',
            'pydub',
            ['Audio processing']
        )
        
        # ML/NLP
        self.dependencies['sklearn'] = self._check_dependency(
            'scikit-learn',
            'sklearn',
            ['Text analysis', 'Clustering']
        )
        
        self.dependencies['langdetect'] = self._check_dependency(
            'langdetect',
            'langdetect',
            ['Language detection']
        )
        
        # Archive processing
        self.dependencies['py7zr'] = self._check_dependency(
            'py7zr',
            'py7zr',
            ['7z archive extraction']
        )
        
        self.dependencies['rarfile'] = self._check_dependency(
            'rarfile',
            'rarfile',
            ['RAR archive extraction']
        )
        
        # Email processing
        self.dependencies['email'] = self._check_dependency(
            'email',
            'email',
            ['Email processing']
        )
        
        # Code analysis
        self.dependencies['pygments'] = self._check_dependency(
            'Pygments',
            'pygments',
            ['Code syntax highlighting', 'Language detection']
        )
        
        logger.info(
            "Dependency check complete",
            total=len(self.dependencies),
            available=sum(1 for d in self.dependencies.values() if d.available),
            missing=sum(1 for d in self.dependencies.values() if not d.available)
        )
    
    def is_available(self, dependency: str) -> bool:
        """
        Check if dependency is available.
        
        Args:
            dependency: Dependency name
            
        Returns:
            True if available
        """
        info = self.dependencies.get(dependency)
        return info.available if info else False
    
    def get_info(self, dependency: str) -> Optional[DependencyInfo]:
        """
        Get dependency information.
        
        Args:
            dependency: Dependency name
            
        Returns:
            DependencyInfo or None
        """
        return self.dependencies.get(dependency)
    
    def require(self, dependency: str, feature: str) -> None:
        """
        Require a dependency for a feature.
        
        Raises ImportError if not available.
        
        Args:
            dependency: Dependency name
            feature: Feature name
            
        Raises:
            ImportError: If dependency not available
        """
        info = self.dependencies.get(dependency)
        
        if not info or not info.available:
            error_msg = (
                f"Feature '{feature}' requires '{dependency}' which is not installed. "
                f"Install it with: pip install {dependency}"
            )
            
            if info and info.import_error:
                error_msg += f"\nOriginal error: {info.import_error}"
            
            logger.error(
                "Required dependency not available",
                dependency=dependency,
                feature=feature
            )
            
            raise ImportError(error_msg)
    
    def get_available_features(self) -> Dict[str, List[str]]:
        """
        Get available features based on installed dependencies.
        
        Returns:
            Dictionary mapping feature categories to available features
        """
        features = {
            'ocr': [],
            'document': [],
            'audio_video': [],
            'archive': [],
            'ml_nlp': [],
            'code': []
        }
        
        # OCR
        if self.is_available('pytesseract'):
            features['ocr'].append('Tesseract OCR')
        if self.is_available('easyocr'):
            features['ocr'].append('EasyOCR')
        if self.is_available('paddleocr'):
            features['ocr'].append('PaddleOCR')
        
        # Documents
        if self.is_available('pypdf'):
            features['document'].append('PDF')
        if self.is_available('python-docx'):
            features['document'].append('DOCX')
        if self.is_available('openpyxl'):
            features['document'].append('Excel')
        if self.is_available('python-pptx'):
            features['document'].append('PowerPoint')
        
        # Audio/Video
        if self.is_available('speech_recognition'):
            features['audio_video'].append('Speech-to-text')
        if self.is_available('pydub'):
            features['audio_video'].append('Audio processing')
        
        # Archives
        if self.is_available('py7zr'):
            features['archive'].append('7z')
        if self.is_available('rarfile'):
            features['archive'].append('RAR')
        
        # ML/NLP
        if self.is_available('sklearn'):
            features['ml_nlp'].append('Text analysis')
        if self.is_available('langdetect'):
            features['ml_nlp'].append('Language detection')
        
        # Code
        if self.is_available('pygments'):
            features['code'].append('Syntax highlighting')
        
        return features
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of dependency status.
        
        Returns:
            Summary dictionary
        """
        return {
            'total_dependencies': len(self.dependencies),
            'available': sum(1 for d in self.dependencies.values() if d.available),
            'missing': sum(1 for d in self.dependencies.values() if not d.available),
            'dependencies': {
                name: {
                    'available': info.available,
                    'version': info.version,
                    'required_for': info.required_for
                }
                for name, info in self.dependencies.items()
            },
            'available_features': self.get_available_features()
        }


# Global instance
_dependency_checker: Optional[DependencyChecker] = None


def get_dependency_checker() -> DependencyChecker:
    """
    Get global dependency checker instance.
    
    Returns:
        DependencyChecker instance
    """
    global _dependency_checker
    
    if _dependency_checker is None:
        _dependency_checker = DependencyChecker()
    
    return _dependency_checker


def check_dependency(dependency: str) -> bool:
    """
    Check if dependency is available.
    
    Args:
        dependency: Dependency name
        
    Returns:
        True if available
    """
    checker = get_dependency_checker()
    return checker.is_available(dependency)


def require_dependency(dependency: str, feature: str) -> None:
    """
    Require a dependency for a feature.
    
    Args:
        dependency: Dependency name
        feature: Feature name
        
    Raises:
        ImportError: If dependency not available
    """
    checker = get_dependency_checker()
    checker.require(dependency, feature)

