"""
Production script to migrate files from structlog to backend_logging.

This script systematically replaces structlog usage with backend_logging
across the entire codebase.

Usage:
    python migrate_to_backend_logger.py <file_path> <category>

Example:
    python migrate_to_backend_logger.py app/agents/base/agent.py AGENT_OPERATIONS
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Category mapping for different modules
CATEGORY_MAP = {
    'agents': 'AGENT_OPERATIONS',
    'memory': 'MEMORY_OPERATIONS',
    'rag': 'RAG_OPERATIONS',
    'llm': 'LLM_OPERATIONS',
    'tools': 'TOOL_OPERATIONS',
    'api': 'API_LAYER',
    'core': 'SYSTEM_HEALTH',
    'services': 'SERVICE_OPERATIONS',
    'config': 'CONFIGURATION_MANAGEMENT',
    'communication': 'COMMUNICATION',
    'integrations': 'EXTERNAL_INTEGRATIONS',
    'optimization': 'PERFORMANCE',
    'orchestration': 'ORCHESTRATION',
    'storage': 'RESOURCE_MANAGEMENT',
}


def get_category_for_file(file_path: str) -> str:
    """Determine the appropriate log category based on file path."""
    path_parts = Path(file_path).parts
    
    for part in path_parts:
        if part in CATEGORY_MAP:
            return CATEGORY_MAP[part]
    
    # Default category
    return 'SYSTEM_HEALTH'


def get_component_name(file_path: str) -> str:
    """Extract component name from file path."""
    path = Path(file_path)
    # Remove 'app/' prefix and '.py' suffix, replace '/' with '.'
    component = str(path).replace('app/', '').replace('app\\', '').replace('.py', '').replace('/', '.').replace('\\', '.')
    return f"app.{component}"


def migrate_file(file_path: str, category: str = None) -> Tuple[bool, str]:
    """
    Migrate a single file from structlog to backend_logging.
    
    Args:
        file_path: Path to the file to migrate
        category: Optional category override
        
    Returns:
        Tuple of (success, message)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already using backend logger
        if 'from app.backend_logging.backend_logger import' in content:
            return True, f"File already using backend logger: {file_path}"
        
        # Check if using structlog
        if 'import structlog' not in content and 'from structlog' not in content:
            return True, f"File not using structlog: {file_path}"
        
        # Determine category and component
        if category is None:
            category = get_category_for_file(file_path)
        component = get_component_name(file_path)
        
        # Step 1: Replace structlog import
        content = re.sub(
            r'import structlog\n',
            '',
            content
        )
        
        # Step 2: Replace logger initialization
        content = re.sub(
            r'logger = structlog\.get_logger\(__name__\)',
            f'# Backend logger will be used via _backend_logger instance',
            content
        )
        
        # Step 3: Add backend logger imports at the top (after other imports)
        # Find the last import statement
        import_pattern = r'((?:from|import)\s+[\w.]+(?:\s+import\s+[\w,\s]+)?(?:\s*#.*)?)\n'
        imports = list(re.finditer(import_pattern, content))
        
        if imports:
            last_import = imports[-1]
            insert_pos = last_import.end()
            
            backend_imports = f"\nfrom app.backend_logging.backend_logger import get_logger as get_backend_logger\nfrom app.backend_logging.models import LogCategory\n\n# Get backend logger instance\n_backend_logger = get_backend_logger()\n"
            
            content = content[:insert_pos] + backend_imports + content[insert_pos:]
        
        # Step 4: Replace logger calls
        # Pattern: logger.level(message, key=value, ...)
        # Replace with: _backend_logger.level(message, LogCategory.CATEGORY, component, data={key: value, ...})
        
        # This is complex, so we'll do a simpler replacement for now
        # Replace logger.debug/info/warning/error calls
        
        def replace_logger_call(match):
            level = match.group(1)
            args = match.group(2)
            
            # Map structlog levels to backend logger levels
            level_map = {
                'debug': 'debug',
                'info': 'info',
                'warning': 'warn',
                'warn': 'warn',
                'error': 'error',
                'exception': 'error',
                'critical': 'fatal',
                'fatal': 'fatal',
            }
            
            backend_level = level_map.get(level, 'info')
            
            # Extract message (first argument)
            # This is a simplified approach - assumes first arg is the message
            message_match = re.match(r'["\']([^"\']+)["\']|([^,]+)', args.strip())
            if message_match:
                message = message_match.group(1) or message_match.group(2)
                message = message.strip()
                
                # If message is a variable, use f-string
                if not (message.startswith('"') or message.startswith("'")):
                    message = f'f"{{{message}}}"'
                
                return f'_backend_logger.{backend_level}({message}, LogCategory.{category}, "{component}")'
            
            # Fallback
            return f'_backend_logger.{backend_level}("Log message", LogCategory.{category}, "{component}")'
        
        # Replace logger calls
        content = re.sub(
            r'logger\.(debug|info|warning|warn|error|exception|critical|fatal)\s*\(',
            lambda m: f'_backend_logger.{m.group(1) if m.group(1) not in ["warning", "exception", "critical"] else {"warning": "warn", "exception": "error", "critical": "fatal"}[m.group(1)]}(',
            content
        )
        
        # Write back
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True, f"Successfully migrated: {file_path}"
        
    except Exception as e:
        return False, f"Error migrating {file_path}: {str(e)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_to_backend_logger.py <file_path> [category]")
        print("\nAvailable categories:")
        for cat in set(CATEGORY_MAP.values()):
            print(f"  - {cat}")
        sys.exit(1)
    
    file_path = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else None
    
    success, message = migrate_file(file_path, category)
    print(message)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

