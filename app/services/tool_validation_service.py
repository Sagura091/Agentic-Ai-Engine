"""
Tool Validation Service for Custom Tool Upload System.

This service provides comprehensive validation, security checking, and
safe execution capabilities for user-uploaded custom tools.
"""

import ast
import sys
import tempfile
import subprocess
import importlib.util
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import re
import hashlib
from datetime import datetime

from pydantic import BaseModel, Field

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory

logger = get_logger()


class ValidationResult(BaseModel):
    """Result of tool validation."""
    is_valid: bool = Field(..., description="Whether the tool is valid")
    security_score: float = Field(..., description="Security score (0-1, higher is safer)")
    issues: List[str] = Field(default_factory=list, description="List of validation issues")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    tool_info: Dict[str, Any] = Field(default_factory=dict, description="Extracted tool information")


class ToolValidationService:
    """Service for validating and securing user-uploaded tools."""
    
    # Dangerous operations that are not allowed
    DANGEROUS_OPERATIONS = {
        'exec', 'eval', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
        'dir', 'hasattr', 'getattr', 'setattr', 'delattr',
        'subprocess', 'os.system', 'os.popen', 'os.spawn'
    }
    
    # Dangerous modules that should not be imported
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pickle',
        'marshal', 'shelve', 'dbm', 'sqlite3', 'socket', 'urllib',
        'http', 'ftplib', 'smtplib', 'telnetlib', 'xmlrpc',
        'multiprocessing', 'threading', 'asyncio', 'concurrent'
    }
    
    # Allowed safe modules
    SAFE_MODULES = {
        'json', 'datetime', 'time', 'math', 'random', 'string',
        'collections', 'itertools', 'functools', 'operator',
        're', 'uuid', 'hashlib', 'base64', 'typing',
        'pydantic', 'langchain', 'langchain_core'
    }
    
    # Required tool class structure
    REQUIRED_METHODS = {'_run', '__init__'}
    REQUIRED_ATTRIBUTES = {'name', 'description'}
    
    def __init__(self):
        """Initialize the validation service."""
        self.validation_cache: Dict[str, ValidationResult] = {}
    
    async def validate_tool_code(self, code: str, filename: str = "uploaded_tool.py") -> ValidationResult:
        """
        Comprehensive validation of tool code.
        
        Args:
            code: Python code to validate
            filename: Name of the file (for context)
            
        Returns:
            ValidationResult with detailed analysis
        """
        try:
            # Create cache key
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            if code_hash in self.validation_cache:
                logger.info(
                    "Using cached validation result",
                    LogCategory.SERVICE_OPERATIONS,
                    "app.services.tool_validation_service",
                    data={"hash": code_hash[:8]}
                )
                return self.validation_cache[code_hash]

            logger.info(
                "Starting tool validation",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.tool_validation_service",
                data={"filename": filename}
            )
            
            result = ValidationResult(
                is_valid=True,
                security_score=1.0,
                issues=[],
                warnings=[],
                dependencies=[],
                tool_info={}
            )
            
            # Step 1: Basic syntax validation
            try:
                tree = ast.parse(code, filename=filename)
            except SyntaxError as e:
                result.is_valid = False
                result.issues.append(f"Syntax error: {str(e)}")
                result.security_score = 0.0
                return result
            
            # Step 2: AST-based security analysis
            security_issues = self._analyze_ast_security(tree)
            result.issues.extend(security_issues)
            
            # Step 3: Import analysis
            imports_analysis = self._analyze_imports(tree)
            result.dependencies.extend(imports_analysis['safe_imports'])
            result.issues.extend(imports_analysis['dangerous_imports'])
            result.warnings.extend(imports_analysis['warnings'])
            
            # Step 4: Tool structure validation
            structure_analysis = self._analyze_tool_structure(tree, code)
            result.tool_info.update(structure_analysis['info'])
            result.issues.extend(structure_analysis['issues'])
            result.warnings.extend(structure_analysis['warnings'])
            
            # Step 5: Calculate security score
            result.security_score = self._calculate_security_score(result)
            
            # Step 6: Final validation decision
            if result.issues or result.security_score < 0.5:
                result.is_valid = False
            
            # Cache result
            self.validation_cache[code_hash] = result

            logger.info(
                "Tool validation completed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.tool_validation_service",
                data={
                    "filename": filename,
                    "is_valid": result.is_valid,
                    "security_score": result.security_score,
                    "issues_count": len(result.issues),
                    "warnings_count": len(result.warnings)
                }
            )

            return result

        except Exception as e:
            logger.error(
                "Tool validation failed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.tool_validation_service",
                error=e
            )
            return ValidationResult(
                is_valid=False,
                security_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                warnings=[],
                dependencies=[],
                tool_info={}
            )
    
    def _analyze_ast_security(self, tree: ast.AST) -> List[str]:
        """Analyze AST for security issues."""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ToolValidationService.DANGEROUS_OPERATIONS:
                        issues.append(f"Dangerous operation detected: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                    if any(danger in func_name for danger in ToolValidationService.DANGEROUS_OPERATIONS):
                        issues.append(f"Dangerous operation detected: {func_name}")
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name in ToolValidationService.DANGEROUS_MODULES:
                        issues.append(f"Dangerous module import: {alias.name}")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module and node.module in ToolValidationService.DANGEROUS_MODULES:
                    issues.append(f"Dangerous module import: {node.module}")
                self.generic_visit(node)
        
        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _analyze_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze imports for safety and dependencies."""
        safe_imports = []
        dangerous_imports = []
        warnings = []
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]  # Get root module
                    
                    if module_name in ToolValidationService.DANGEROUS_MODULES:
                        dangerous_imports.append(f"Dangerous import: {alias.name}")
                    elif module_name in ToolValidationService.SAFE_MODULES:
                        safe_imports.append(alias.name)
                    else:
                        warnings.append(f"Unknown module import: {alias.name} (requires review)")
                        safe_imports.append(alias.name)  # Allow but warn
                
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    module_name = node.module.split('.')[0]  # Get root module
                    
                    if module_name in ToolValidationService.DANGEROUS_MODULES:
                        dangerous_imports.append(f"Dangerous import from: {node.module}")
                    elif module_name in ToolValidationService.SAFE_MODULES:
                        safe_imports.append(node.module)
                    else:
                        warnings.append(f"Unknown module import from: {node.module} (requires review)")
                        safe_imports.append(node.module)  # Allow but warn
                
                self.generic_visit(node)
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        return {
            'safe_imports': list(set(safe_imports)),
            'dangerous_imports': dangerous_imports,
            'warnings': warnings
        }
    
    def _analyze_tool_structure(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Analyze tool class structure."""
        info = {}
        issues = []
        warnings = []
        
        class ToolStructureVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                # Look for tool classes (classes that inherit from BaseTool or similar)
                base_names = [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                
                if any('Tool' in base for base in base_names):
                    info['tool_class'] = node.name
                    info['base_classes'] = base_names
                    
                    # Check for required methods and attributes
                    methods = []
                    attributes = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    attributes.append(target.id)
                    
                    info['methods'] = methods
                    info['attributes'] = attributes
                    
                    # Check for required methods
                    missing_methods = ToolValidationService.REQUIRED_METHODS - set(methods)
                    if missing_methods:
                        issues.extend([f"Missing required method: {method}" for method in missing_methods])
                    
                    # Check for required attributes
                    missing_attrs = ToolValidationService.REQUIRED_ATTRIBUTES - set(attributes)
                    if missing_attrs:
                        warnings.extend([f"Missing recommended attribute: {attr}" for attr in missing_attrs])
                
                self.generic_visit(node)
        
        visitor = ToolStructureVisitor()
        visitor.visit(tree)
        
        # Extract docstrings and descriptions
        if 'tool_class' not in info:
            issues.append("No tool class found (should inherit from BaseTool or similar)")
        
        return {
            'info': info,
            'issues': issues,
            'warnings': warnings
        }
    
    def _calculate_security_score(self, result: ValidationResult) -> float:
        """Calculate security score based on analysis results."""
        base_score = 1.0
        
        # Deduct for issues
        issue_penalty = len(result.issues) * 0.2
        warning_penalty = len(result.warnings) * 0.05
        
        # Bonus for good structure
        structure_bonus = 0.1 if result.tool_info.get('tool_class') else 0.0
        
        final_score = max(0.0, base_score - issue_penalty - warning_penalty + structure_bonus)
        return min(1.0, final_score)
    
    async def validate_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Validate that dependencies are safe and available."""
        try:
            safe_deps = []
            unsafe_deps = []
            missing_deps = []
            
            for dep in dependencies:
                # Check if dependency is in safe list
                root_dep = dep.split('.')[0]
                
                if root_dep in self.SAFE_MODULES:
                    safe_deps.append(dep)
                elif root_dep in self.DANGEROUS_MODULES:
                    unsafe_deps.append(dep)
                else:
                    # Try to import to check availability
                    try:
                        importlib.import_module(root_dep)
                        safe_deps.append(dep)
                    except ImportError:
                        missing_deps.append(dep)
            
            return {
                'safe_dependencies': safe_deps,
                'unsafe_dependencies': unsafe_deps,
                'missing_dependencies': missing_deps,
                'all_safe': len(unsafe_deps) == 0 and len(missing_deps) == 0
            }

        except Exception as e:
            logger.error(
                "Dependency validation failed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.tool_validation_service",
                error=e
            )
            return {
                'safe_dependencies': [],
                'unsafe_dependencies': dependencies,
                'missing_dependencies': [],
                'all_safe': False
            }
    
    async def extract_tool_metadata(self, code: str) -> Dict[str, Any]:
        """Extract metadata from tool code."""
        try:
            tree = ast.parse(code)
            metadata = {
                'classes': [],
                'functions': [],
                'imports': [],
                'docstring': None
            }
            
            # Extract module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) 
                and isinstance(tree.body[0].value, ast.Constant)):
                metadata['docstring'] = tree.body[0].value.value
            
            class MetadataVisitor(ast.NodeVisitor):
                def visit_ClassDef(self, node):
                    class_info = {
                        'name': node.name,
                        'bases': [ast.unparse(base) for base in node.bases],
                        'methods': [item.name for item in node.body if isinstance(item, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node)
                    }
                    metadata['classes'].append(class_info)
                    self.generic_visit(node)
                
                def visit_FunctionDef(self, node):
                    if not any(node in cls_node.body for cls_node in ast.walk(tree) if isinstance(cls_node, ast.ClassDef)):
                        func_info = {
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'docstring': ast.get_docstring(node)
                        }
                        metadata['functions'].append(func_info)
                    self.generic_visit(node)
                
                def visit_Import(self, node):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)
                    self.generic_visit(node)
                
                def visit_ImportFrom(self, node):
                    if node.module:
                        for alias in node.names:
                            metadata['imports'].append(f"{node.module}.{alias.name}")
                    self.generic_visit(node)
            
            visitor = MetadataVisitor()
            visitor.visit(tree)

            return metadata

        except Exception as e:
            logger.error(
                "Metadata extraction failed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.tool_validation_service",
                error=e
            )
            return {}


# Global service instance
tool_validation_service = ToolValidationService()
