"""
Comprehensive Logger Migration Script

This script safely migrates all structlog logger calls to the backend_logging system.
It includes full error handling, backup creation, validation, and rollback capabilities.

Features:
- Creates backups before modification
- Validates Python syntax after changes
- Handles all logger call patterns (simple, f-strings, multi-line, kwargs)
- Automatic category detection based on file path
- Dry-run mode for testing
- Detailed logging of all changes
- Rollback capability on errors

Usage:
    # Dry run (no changes)
    python comprehensive_logger_migration.py --dry-run
    
    # Migrate all files
    python comprehensive_logger_migration.py
    
    # Migrate specific file
    python comprehensive_logger_migration.py --file app/memory/unified_memory_system.py
    
    # Migrate specific module
    python comprehensive_logger_migration.py --module agents
"""

import os
import re
import sys
import ast
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import json


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
    'backend_logging': 'SYSTEM_HEALTH',
}

# Level mapping
LEVEL_MAP = {
    'debug': 'debug',
    'info': 'info',
    'warning': 'warn',
    'warn': 'warn',
    'error': 'error',
    'exception': 'error',
    'critical': 'fatal',
    'fatal': 'fatal',
}


class LoggerMigration:
    """Handles safe migration of logger calls."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.backup_dir = Path("migration_backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.migration_log = []
        self.errors = []
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'files_skipped': 0,
            'files_failed': 0,
            'logger_calls_replaced': 0,
            'imports_updated': 0,
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.migration_log.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def get_category_for_file(self, file_path: Path) -> str:
        """Determine the appropriate log category based on file path."""
        parts = file_path.parts
        
        for part in parts:
            if part in CATEGORY_MAP:
                return CATEGORY_MAP[part]
        
        return 'SYSTEM_HEALTH'
    
    def get_component_name(self, file_path: Path) -> str:
        """Extract component name from file path."""
        # Remove 'app/' prefix and '.py' suffix, replace separators with dots
        path_str = str(file_path)
        if path_str.startswith('app/') or path_str.startswith('app\\'):
            path_str = path_str[4:]
        
        component = path_str.replace('.py', '').replace('/', '.').replace('\\', '.')
        return f"app.{component}"
    
    def create_backup(self, file_path: Path) -> Path:
        """Create a backup of the file."""
        if self.dry_run:
            return None
        
        backup_path = self.backup_dir / file_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def validate_python_syntax(self, content: str, file_path: Path) -> bool:
        """Validate that the content is valid Python."""
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            self.log(f"Syntax error in {file_path}: {e}", "ERROR")
            self.errors.append(f"Syntax error in {file_path}: {e}")
            return False
    
    def check_if_needs_migration(self, content: str) -> bool:
        """Check if file needs migration."""
        # Check if using structlog
        has_structlog = 'import structlog' in content or 'from structlog' in content
        
        # Check if already using backend logger
        has_backend_logger = 'from app.backend_logging.backend_logger import' in content
        
        # Check if has logger calls
        has_logger_calls = bool(re.search(r'\blogger\.(debug|info|warning|warn|error|exception|critical|fatal)\s*\(', content))
        
        return has_structlog and not has_backend_logger and has_logger_calls
    
    def update_imports(self, content: str, category: str) -> str:
        """Update imports to use backend_logging."""
        lines = content.split('\n')
        new_lines = []
        imports_added = False
        last_import_index = -1
        
        for i, line in enumerate(lines):
            # Remove structlog import
            if re.match(r'^\s*import structlog\s*$', line):
                continue
            
            # Remove structlog logger initialization
            if re.match(r'^\s*logger\s*=\s*structlog\.get_logger\(__name__\)\s*$', line):
                continue
            
            # Track last import line
            if re.match(r'^\s*(from|import)\s+', line) and not line.strip().startswith('#'):
                last_import_index = len(new_lines)
            
            new_lines.append(line)
        
        # Add backend_logging imports after last import
        if last_import_index >= 0 and not imports_added:
            backend_imports = [
                "",
                "from app.backend_logging.backend_logger import get_logger as get_backend_logger",
                "from app.backend_logging.models import LogCategory",
                "",
                "# Get backend logger instance",
                "_backend_logger = get_backend_logger()",
            ]
            new_lines = new_lines[:last_import_index + 1] + backend_imports + new_lines[last_import_index + 1:]
            self.stats['imports_updated'] += 1
        
        return '\n'.join(new_lines)
    
    def find_logger_calls(self, content: str) -> List[Tuple[int, int, str, str]]:
        """
        Find all logger calls in the content.
        
        Returns:
            List of (start_pos, end_pos, level, full_call) tuples
        """
        logger_calls = []
        
        # Pattern to match logger calls
        pattern = r'\blogger\.(debug|info|warning|warn|error|exception|critical|fatal)\s*\('
        
        for match in re.finditer(pattern, content):
            level = match.group(1)
            start_pos = match.start()
            
            # Find the matching closing parenthesis
            end_pos = self.find_matching_paren(content, match.end() - 1)
            
            if end_pos > 0:
                full_call = content[start_pos:end_pos + 1]
                logger_calls.append((start_pos, end_pos, level, full_call))
        
        return logger_calls
    
    def find_matching_paren(self, content: str, start_pos: int) -> int:
        """Find the matching closing parenthesis."""
        depth = 1
        i = start_pos + 1
        in_string = False
        string_char = None
        escape_next = False
        
        while i < len(content) and depth > 0:
            char = content[i]
            
            # Handle escape sequences
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                i += 1
                continue
            
            # Handle string literals
            if char in ['"', "'", '"""', "'''"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            
            # Count parentheses only outside strings
            if not in_string:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
            
            i += 1
        
        return i - 1 if depth == 0 else -1
    
    def replace_logger_call(self, call: str, level: str, category: str, component: str) -> str:
        """
        Replace a single logger call with backend_logger equivalent.
        
        Args:
            call: The full logger call string
            level: The log level (debug, info, etc.)
            category: The LogCategory to use
            component: The component name
            
        Returns:
            The replacement string
        """
        # Map the level
        backend_level = LEVEL_MAP.get(level, 'info')
        
        # Extract the arguments from the call
        # Pattern: logger.level(args)
        args_match = re.search(r'logger\.\w+\s*\((.*)\)', call, re.DOTALL)
        if not args_match:
            return call
        
        args_str = args_match.group(1).strip()
        
        # Parse the arguments
        # This is complex because we need to handle:
        # 1. Simple strings: logger.info("message")
        # 2. F-strings: logger.info(f"message {var}")
        # 3. Keyword args: logger.info("message", key=value, key2=value2)
        # 4. Multi-line calls
        
        # Try to extract the message (first argument)
        message = None
        data_dict = {}
        
        # Simple approach: split by comma, but respect strings and parentheses
        parts = self.smart_split_args(args_str)
        
        if parts:
            message = parts[0].strip()
            
            # Extract keyword arguments
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    data_dict[key.strip()] = value.strip()
        
        # Build the replacement
        if data_dict:
            # Has keyword arguments - put them in data dict
            data_items = ', '.join([f'"{k}": {v}' for k, v in data_dict.items()])
            replacement = f'_backend_logger.{backend_level}(\n            {message},\n            LogCategory.{category},\n            "{component}",\n            data={{{data_items}}}\n        )'
        else:
            # Simple call - just message
            replacement = f'_backend_logger.{backend_level}(\n            {message},\n            LogCategory.{category},\n            "{component}"\n        )'
        
        return replacement
    
    def smart_split_args(self, args_str: str) -> List[str]:
        """
        Split arguments by comma, respecting strings, parentheses, and brackets.
        """
        parts = []
        current = []
        depth = 0
        in_string = False
        string_char = None
        escape_next = False
        
        for char in args_str:
            if escape_next:
                current.append(char)
                escape_next = False
                continue
            
            if char == '\\':
                current.append(char)
                escape_next = True
                continue
            
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                current.append(char)
                continue
            
            if not in_string:
                if char in '([{':
                    depth += 1
                elif char in ')]}':
                    depth -= 1
                elif char == ',' and depth == 0:
                    parts.append(''.join(current))
                    current = []
                    continue
            
            current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts

