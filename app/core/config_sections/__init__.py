"""
🚀 Revolutionary Configuration Section Managers

This package contains section-specific configuration managers that handle
validation, application, and rollback for different parts of the system.

Each section manager is responsible for:
- Validating configuration changes
- Applying changes to target systems
- Rolling back changes on failure
- Providing section-specific business logic
"""

from .base_section_manager import BaseConfigurationSectionManager
from .rag_section_manager import RAGSectionManager

__all__ = [
    "BaseConfigurationSectionManager",
    "RAGSectionManager"
]
