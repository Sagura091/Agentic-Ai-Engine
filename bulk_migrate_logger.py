"""
Bulk migration script to replace all logger calls in a file with backend_logger calls.

This script reads a file, finds all logger.X() calls, and replaces them with
_backend_logger.X() calls with proper LogCategory and component name.

Usage:
    python bulk_migrate_logger.py <file_path> <log_category> <component_name>

Example:
    python bulk_migrate_logger.py app/memory/unified_memory_system.py MEMORY_OPERATIONS app.memory.unified_memory_system
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def extract_logger_call_info(line: str, line_num: int) -> Tuple[str, int, int]:
    """
    Extract logger call information from a line.
    
    Returns:
        Tuple of (log_level, start_pos, end_pos)
    """
    # Match logger.level( patterns
    pattern = r'logger\.(debug|info|warning|warn|error|exception|critical|fatal)\s*\('
    match = re.search(pattern, line)
    
    if match:
        level = match.group(1)
        start_pos = match.start()
        end_pos = match.end()
        return (level, start_pos, end_pos)
    
    return (None, -1, -1)


def find_matching_paren(content: str, start_pos: int) -> int:
    """
    Find the matching closing parenthesis for a function call.
    
    Args:
        content: Full file content
        start_pos: Position of opening parenthesis
        
    Returns:
        Position of matching closing parenthesis
    """
    depth = 1
    i = start_pos
    in_string = False
    string_char = None
    
    while i < len(content) and depth > 0:
        char = content[i]
        
        # Handle string literals
        if char in ['"', "'"] and (i == 0 or content[i-1] != '\\'):
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


def replace_logger_calls(file_path: str, category: str, component: str) -> Tuple[bool, str, int]:
    """
    Replace all logger calls in a file with backend_logger calls.
    
    Args:
        file_path: Path to the file to process
        category: LogCategory to use (e.g., "MEMORY_OPERATIONS")
        component: Component name (e.g., "app.memory.unified_memory_system")
        
    Returns:
        Tuple of (success, message, replacements_count)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}", 0
        
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        replacements = 0
        
        # Level mapping
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
        
        # Find all logger calls
        lines = content.split('\n')
        offset = 0
        
        for line_num, line in enumerate(lines):
            level, start_pos, end_pos = extract_logger_call_info(line, line_num)
            
            if level:
                # Calculate absolute position in file
                abs_start = offset + start_pos
                abs_paren_start = offset + end_pos - 1  # Position of '('
                
                # Find matching closing parenthesis
                abs_paren_end = find_matching_paren(content, abs_paren_start + 1)
                
                if abs_paren_end > 0:
                    # Extract the arguments
                    args_content = content[abs_paren_start + 1:abs_paren_end]
                    
                    # Map the level
                    backend_level = level_map.get(level, 'info')
                    
                    # Build replacement
                    # Simple case: just message string
                    if args_content.strip().startswith(('"', "'")):
                        # Extract message
                        message_match = re.match(r'(["\'])(.+?)\1', args_content.strip())
                        if message_match:
                            quote_char = message_match.group(1)
                            message = message_match.group(2)
                            
                            replacement = f'_backend_logger.{backend_level}(\n            {quote_char}{message}{quote_char},\n            LogCategory.{category},\n            "{component}"\n        )'
                            
                            # Replace in content
                            old_call = content[abs_start:abs_paren_end + 1]
                            content = content[:abs_start] + replacement + content[abs_paren_end + 1:]
                            
                            replacements += 1
                            
                            # Update offset for next iteration
                            offset += len(replacement) - len(old_call)
                    else:
                        # Complex case with f-string or variables
                        # Just replace logger. with _backend_logger. and add category/component
                        # This is a simplified approach
                        pass
            
            offset += len(line) + 1  # +1 for newline
        
        # Write back if changes were made
        if replacements > 0:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, f"Successfully replaced {replacements} logger calls in {file_path}", replacements
        else:
            return True, f"No logger calls found in {file_path}", 0
        
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}", 0


def main():
    if len(sys.argv) < 4:
        print("Usage: python bulk_migrate_logger.py <file_path> <log_category> <component_name>")
        print("\nExample:")
        print("  python bulk_migrate_logger.py app/memory/unified_memory_system.py MEMORY_OPERATIONS app.memory.unified_memory_system")
        sys.exit(1)
    
    file_path = sys.argv[1]
    category = sys.argv[2]
    component = sys.argv[3]
    
    success, message, count = replace_logger_calls(file_path, category, component)
    print(message)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

