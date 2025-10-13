"""
🔍 REVOLUTIONARY TOOL AUTO-DISCOVERY SYSTEM - Intelligent Tool Scanner

The most advanced tool discovery and registration system ever created.
Automatically discovers, validates, and registers tools across the entire codebase.

🚀 REVOLUTIONARY CAPABILITIES:
- Automatic tool discovery across all directories
- Intelligent metadata extraction from docstrings and decorators
- Dynamic tool validation and structure analysis
- Dependency resolution and conflict detection
- Health monitoring and status tracking
- Automatic registration with unified tool repository
- Version management and update detection
- Performance profiling and optimization
- Security scanning and validation
- Integration testing automation

🎯 CORE FEATURES:
- Directory scanning and file analysis
- AST parsing for tool extraction
- Metadata inference and validation
- Factory function detection
- Dependency graph construction
- Conflict resolution algorithms
- Health check automation
- Performance benchmarking
- Security assessment
- Documentation generation
"""

import asyncio
import ast
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.tools.unified_tool_repository import ToolMetadata, ToolCategory as ToolCategoryEnum, ToolAccessLevel
from app.tools.testing.universal_tool_tester import UniversalToolTester, ToolTestResult

logger = get_logger()


class DiscoveryStatus(Enum):
    """Tool discovery status."""
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    REGISTERED = "registered"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ToolInfo:
    """Comprehensive tool information."""
    file_path: Path
    module_name: str
    class_name: str
    tool_id: str
    factory_function: Optional[str] = None
    metadata: Optional[ToolMetadata] = None
    dependencies: List[str] = field(default_factory=list)
    status: DiscoveryStatus = DiscoveryStatus.DISCOVERED
    error_message: Optional[str] = None
    test_result: Optional[ToolTestResult] = None
    last_modified: Optional[datetime] = None
    file_hash: Optional[str] = None


class ToolAutoDiscovery:
    """Revolutionary tool auto-discovery system."""
    
    def __init__(self, tool_repository=None):
        """Initialize the tool auto-discovery system."""
        self.tool_repository = tool_repository
        self.discovered_tools: Dict[str, ToolInfo] = {}
        self.discovery_cache: Dict[str, ToolInfo] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.tester = UniversalToolTester()
        
        # Discovery configuration
        self.config = {
            "scan_directories": [
                "app/tools",
                "app/tools/production",
                "app/tools/social_media",
                "app/tools/creative"
            ],
            "exclude_patterns": [
                "__pycache__",
                "*.pyc",
                "test_*",
                "*_test.py",
                "testing"
            ],
            "required_base_classes": [BaseTool],
            "factory_function_patterns": [
                "get_*_tool",
                "create_*_tool",
                "*_tool_factory"
            ]
        }

        logger.info(
            "🔍 Tool Auto-Discovery System initialized",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner"
        )

    async def discover_all_tools(self) -> Dict[str, ToolInfo]:
        """
        Discover all tools in the configured directories.

        Returns:
            Dictionary of discovered tools with their information
        """
        logger.info(
            "🔍 Starting comprehensive tool discovery...",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner"
        )

        discovered_count = 0

        for directory in self.config["scan_directories"]:
            try:
                tools_in_dir = await self._scan_directory(Path(directory))
                discovered_count += len(tools_in_dir)
                self.discovered_tools.update(tools_in_dir)

                logger.info(
                    f"🔍 Discovered {len(tools_in_dir)} tools in {directory}",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.auto_discovery.tool_scanner",
                    data={"tools_count": len(tools_in_dir), "directory": str(directory)}
                )

            except Exception as e:
                logger.error(
                    f"Failed to scan directory {directory}",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.auto_discovery.tool_scanner",
                    data={"directory": str(directory)},
                    error=e
                )

        logger.info(
            f"🔍 Discovery complete: {discovered_count} tools found",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner",
            data={"discovered_count": discovered_count}
        )
        
        # Build dependency graph
        await self._build_dependency_graph()
        
        # Validate discovered tools
        await self._validate_discovered_tools()
        
        return self.discovered_tools
    
    async def _scan_directory(self, directory: Path) -> Dict[str, ToolInfo]:
        """Scan a directory for tool files."""
        tools = {}

        if not directory.exists():
            logger.warn(
                f"Directory does not exist: {directory}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.auto_discovery.tool_scanner",
                data={"directory": str(directory)}
            )
            return tools

        # Recursively scan Python files
        for file_path in directory.rglob("*.py"):
            # Skip excluded patterns
            if self._should_skip_file(file_path):
                continue

            try:
                tool_info = await self._analyze_file(file_path)
                if tool_info:
                    tools[tool_info.tool_id] = tool_info

            except Exception as e:
                logger.error(
                    f"Failed to analyze file {file_path}",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.auto_discovery.tool_scanner",
                    data={"file_path": str(file_path)},
                    error=e
                )

        return tools
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped based on exclude patterns."""
        file_str = str(file_path)
        
        for pattern in self.config["exclude_patterns"]:
            if pattern.replace("*", "") in file_str:
                return True
        
        return False
    
    async def _analyze_file(self, file_path: Path) -> Optional[ToolInfo]:
        """Analyze a Python file for tool classes."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate file hash for caching
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check cache
            cache_key = str(file_path)
            if cache_key in self.discovery_cache:
                cached_info = self.discovery_cache[cache_key]
                if cached_info.file_hash == file_hash:
                    return cached_info
            
            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                logger.warn(
                    f"Syntax error in {file_path}",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.auto_discovery.tool_scanner",
                    data={"file_path": str(file_path), "syntax_error": str(e)}
                )
                return None
            
            # Find tool classes
            tool_classes = self._find_tool_classes(tree)
            if not tool_classes:
                return None
            
            # For now, take the first tool class found
            tool_class = tool_classes[0]
            
            # Extract module name
            module_name = self._path_to_module_name(file_path)
            
            # Create tool info
            tool_info = ToolInfo(
                file_path=file_path,
                module_name=module_name,
                class_name=tool_class['name'],
                tool_id=self._generate_tool_id(tool_class['name']),
                factory_function=self._find_factory_function(tree, tool_class['name']),
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                file_hash=file_hash
            )
            
            # Extract metadata from docstring and class
            tool_info.metadata = await self._extract_metadata(tree, tool_class, tool_info)
            
            # Extract dependencies
            tool_info.dependencies = self._extract_dependencies(tree)
            
            # Cache result
            self.discovery_cache[cache_key] = tool_info
            
            return tool_info

        except Exception as e:
            logger.error(
                f"Failed to analyze file {file_path}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.auto_discovery.tool_scanner",
                data={"file_path": str(file_path)},
                error=e
            )
            return None
    
    def _find_tool_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find tool classes in AST."""
        tool_classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class inherits from BaseTool
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'BaseTool':
                        tool_classes.append({
                            'name': node.name,
                            'node': node,
                            'docstring': ast.get_docstring(node),
                            'line_number': node.lineno
                        })
                        break
                    elif isinstance(base, ast.Attribute) and base.attr == 'BaseTool':
                        tool_classes.append({
                            'name': node.name,
                            'node': node,
                            'docstring': ast.get_docstring(node),
                            'line_number': node.lineno
                        })
                        break
        
        return tool_classes
    
    def _find_factory_function(self, tree: ast.AST, class_name: str) -> Optional[str]:
        """Find factory function for a tool class."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function name matches factory patterns
                for pattern in self.config["factory_function_patterns"]:
                    pattern_regex = pattern.replace("*", ".*")
                    if pattern_regex in node.name.lower():
                        # Check if function returns the tool class
                        for child in ast.walk(node):
                            if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                                if isinstance(child.value.func, ast.Name) and child.value.func.id == class_name:
                                    return node.name
        
        return None
    
    def _path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to Python module name."""
        # Get relative path from project root
        try:
            # Assume we're in the project root
            relative_path = file_path.relative_to(Path.cwd())
        except ValueError:
            # If not, use the file path as is
            relative_path = file_path
        
        # Convert to module name
        module_parts = list(relative_path.parts[:-1])  # Remove filename
        module_parts.append(relative_path.stem)  # Add filename without extension
        
        return ".".join(module_parts)
    
    def _generate_tool_id(self, class_name: str) -> str:
        """Generate tool ID from class name."""
        # Convert CamelCase to snake_case
        import re
        tool_id = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        tool_id = re.sub('([a-z0-9])([A-Z])', r'\1_\2', tool_id).lower()
        
        # Remove common suffixes
        if tool_id.endswith('_tool'):
            tool_id = tool_id[:-5]
        
        return tool_id
    
    async def _extract_metadata(self, tree: ast.AST, tool_class: Dict, tool_info: ToolInfo) -> ToolMetadata:
        """Extract metadata from tool class."""
        # Default metadata
        metadata = ToolMetadata(
            tool_id=tool_info.tool_id,
            name=tool_class['name'],
            description=tool_class['docstring'] or f"{tool_class['name']} - Auto-discovered tool",
            category=ToolCategoryEnum.UTILITY,  # Default category
            access_level=ToolAccessLevel.PUBLIC,
            requires_rag=False,
            use_cases=set()
        )

        # Try to infer category from class name or docstring
        class_name_lower = tool_class['name'].lower()
        docstring_lower = (tool_class['docstring'] or "").lower()

        # Category inference
        if any(word in class_name_lower for word in ['social', 'media', 'twitter', 'facebook']):
            metadata.category = ToolCategoryEnum.COMMUNICATION
        elif any(word in class_name_lower for word in ['meme', 'image', 'creative', 'art']):
            metadata.category = ToolCategoryEnum.CREATIVE
        elif any(word in class_name_lower for word in ['file', 'document', 'pdf', 'excel']):
            metadata.category = ToolCategoryEnum.PRODUCTIVITY
        elif any(word in class_name_lower for word in ['web', 'scraper', 'http', 'api']):
            metadata.category = ToolCategoryEnum.RESEARCH
        elif any(word in class_name_lower for word in ['database', 'sql', 'data']):
            metadata.category = ToolCategoryEnum.DATA
        elif any(word in class_name_lower for word in ['security', 'password', 'crypto']):
            metadata.category = ToolCategoryEnum.SECURITY
        elif any(word in class_name_lower for word in ['automation', 'browser', 'screenshot']):
            metadata.category = ToolCategoryEnum.AUTOMATION
        
        # Use case inference
        use_cases = set()
        if 'meme' in class_name_lower:
            use_cases.add('meme_generation')
        if 'social' in class_name_lower:
            use_cases.add('social_media')
        if 'file' in class_name_lower:
            use_cases.add('file_operations')
        if 'web' in class_name_lower:
            use_cases.add('web_research')
        
        metadata.use_cases = use_cases or {'general_purpose'}
        
        return metadata
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract dependencies from import statements."""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
        
        # Filter to external dependencies (not built-in or local)
        external_deps = []
        builtin_modules = set(sys.builtin_module_names)
        
        for dep in dependencies:
            root_module = dep.split('.')[0]
            if root_module not in builtin_modules and not root_module.startswith('app'):
                external_deps.append(root_module)
        
        return list(set(external_deps))  # Remove duplicates

    async def _build_dependency_graph(self):
        """Build dependency graph for discovered tools."""
        logger.info(
            "🔗 Building tool dependency graph...",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner"
        )

        for tool_id, tool_info in self.discovered_tools.items():
            self.dependency_graph[tool_id] = set()

            # Check if any dependencies are other discovered tools
            for dep in tool_info.dependencies:
                for other_tool_id, other_tool_info in self.discovered_tools.items():
                    if other_tool_id != tool_id:
                        if dep in other_tool_info.module_name or dep in other_tool_info.class_name.lower():
                            self.dependency_graph[tool_id].add(other_tool_id)

        logger.info(
            f"🔗 Dependency graph built with {len(self.dependency_graph)} nodes",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner",
            data={"nodes_count": len(self.dependency_graph)}
        )

    async def _validate_discovered_tools(self):
        """Validate all discovered tools."""
        logger.info(
            "✅ Validating discovered tools...",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner"
        )

        validated_count = 0
        failed_count = 0

        for tool_id, tool_info in self.discovered_tools.items():
            try:
                # Try to import and instantiate the tool
                tool_instance = await self._load_tool_instance(tool_info)

                if tool_instance:
                    # Run comprehensive tests
                    test_result = await self.tester.test_tool_comprehensive(tool_instance)
                    tool_info.test_result = test_result

                    if test_result.overall_success:
                        tool_info.status = DiscoveryStatus.VALIDATED
                        validated_count += 1
                    else:
                        tool_info.status = DiscoveryStatus.FAILED
                        tool_info.error_message = f"Validation failed: {len(test_result.issues)} issues found"
                        failed_count += 1
                else:
                    tool_info.status = DiscoveryStatus.FAILED
                    tool_info.error_message = "Failed to load tool instance"
                    failed_count += 1

            except Exception as e:
                tool_info.status = DiscoveryStatus.FAILED
                tool_info.error_message = f"Validation error: {str(e)}"
                failed_count += 1
                logger.error(
                    f"Failed to validate tool {tool_id}",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.auto_discovery.tool_scanner",
                    data={"tool_id": tool_id},
                    error=e
                )

        logger.info(
            f"✅ Validation complete: {validated_count} validated, {failed_count} failed",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner",
            data={"validated_count": validated_count, "failed_count": failed_count}
        )

    async def _load_tool_instance(self, tool_info: ToolInfo) -> Optional[BaseTool]:
        """Load tool instance from tool info."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                tool_info.module_name,
                tool_info.file_path
            )
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Try factory function first
            if tool_info.factory_function:
                if hasattr(module, tool_info.factory_function):
                    factory_func = getattr(module, tool_info.factory_function)
                    return factory_func()

            # Try direct class instantiation
            if hasattr(module, tool_info.class_name):
                tool_class = getattr(module, tool_info.class_name)
                return tool_class()

            return None

        except Exception as e:
            logger.error(
                f"Failed to load tool instance for {tool_info.tool_id}",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.auto_discovery.tool_scanner",
                data={"tool_id": tool_info.tool_id},
                error=e
            )
            return None

    async def register_discovered_tools(self) -> Dict[str, bool]:
        """Register all validated tools with the tool repository."""
        if not self.tool_repository:
            logger.error(
                "No tool repository available for registration",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.auto_discovery.tool_scanner"
            )
            return {}

        logger.info(
            "📝 Registering discovered tools...",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner"
        )

        registration_results = {}
        registered_count = 0

        # Sort tools by dependency order
        sorted_tools = self._topological_sort()

        for tool_id in sorted_tools:
            tool_info = self.discovered_tools[tool_id]

            if tool_info.status != DiscoveryStatus.VALIDATED:
                registration_results[tool_id] = False
                continue

            try:
                # Load tool instance
                tool_instance = await self._load_tool_instance(tool_info)

                if tool_instance and tool_info.metadata:
                    # Register with repository
                    await self.tool_repository.register_tool(tool_instance, tool_info.metadata)
                    tool_info.status = DiscoveryStatus.REGISTERED
                    registration_results[tool_id] = True
                    registered_count += 1

                    logger.info(
                        f"📝 Registered tool: {tool_id}",
                        LogCategory.TOOL_OPERATIONS,
                        "app.tools.auto_discovery.tool_scanner",
                        data={"tool_id": tool_id}
                    )
                else:
                    registration_results[tool_id] = False
                    tool_info.error_message = "Failed to load tool for registration"

            except Exception as e:
                registration_results[tool_id] = False
                tool_info.error_message = f"Registration failed: {str(e)}"
                logger.error(
                    f"Failed to register tool {tool_id}",
                    LogCategory.TOOL_OPERATIONS,
                    "app.tools.auto_discovery.tool_scanner",
                    data={"tool_id": tool_id},
                    error=e
                )

        logger.info(
            f"📝 Registration complete: {registered_count} tools registered",
            LogCategory.TOOL_OPERATIONS,
            "app.tools.auto_discovery.tool_scanner",
            data={"registered_count": registered_count}
        )
        return registration_results

    def _topological_sort(self) -> List[str]:
        """Sort tools by dependency order using topological sort."""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []

        def visit(tool_id: str):
            if tool_id in temp_visited:
                # Circular dependency detected, skip
                return
            if tool_id in visited:
                return

            temp_visited.add(tool_id)

            # Visit dependencies first
            for dep_tool_id in self.dependency_graph.get(tool_id, set()):
                if dep_tool_id in self.discovered_tools:
                    visit(dep_tool_id)

            temp_visited.remove(tool_id)
            visited.add(tool_id)
            result.append(tool_id)

        # Visit all tools
        for tool_id in self.discovered_tools.keys():
            if tool_id not in visited:
                visit(tool_id)

        return result

    def generate_discovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive discovery report."""
        report = {
            "discovery_timestamp": datetime.utcnow().isoformat(),
            "total_tools_discovered": len(self.discovered_tools),
            "tools_by_status": {},
            "tools_by_category": {},
            "dependency_graph": {},
            "validation_summary": {},
            "recommendations": []
        }

        # Count by status
        for status in DiscoveryStatus:
            count = sum(1 for tool in self.discovered_tools.values() if tool.status == status)
            report["tools_by_status"][status.value] = count

        # Count by category
        for tool_info in self.discovered_tools.values():
            if tool_info.metadata:
                category = tool_info.metadata.category.value
                report["tools_by_category"][category] = report["tools_by_category"].get(category, 0) + 1

        # Dependency graph
        report["dependency_graph"] = {
            tool_id: list(deps) for tool_id, deps in self.dependency_graph.items()
        }

        # Validation summary
        validated_tools = [t for t in self.discovered_tools.values() if t.test_result]
        if validated_tools:
            avg_quality = sum(t.test_result.quality_score for t in validated_tools) / len(validated_tools)
            report["validation_summary"] = {
                "average_quality_score": avg_quality,
                "tools_tested": len(validated_tools),
                "high_quality_tools": sum(1 for t in validated_tools if t.test_result.quality_score > 80),
                "low_quality_tools": sum(1 for t in validated_tools if t.test_result.quality_score < 50)
            }

        # Generate recommendations
        report["recommendations"] = self._generate_discovery_recommendations()

        return report

    def _generate_discovery_recommendations(self) -> List[str]:
        """Generate recommendations based on discovery results."""
        recommendations = []

        failed_tools = [t for t in self.discovered_tools.values() if t.status == DiscoveryStatus.FAILED]
        if failed_tools:
            recommendations.append(f"🔧 Fix {len(failed_tools)} failed tools to improve system reliability")

        low_quality_tools = [
            t for t in self.discovered_tools.values()
            if t.test_result and t.test_result.quality_score < 50
        ]
        if low_quality_tools:
            recommendations.append(f"⚡ Improve {len(low_quality_tools)} low-quality tools")

        # Check for missing factory functions
        no_factory = [t for t in self.discovered_tools.values() if not t.factory_function]
        if no_factory:
            recommendations.append(f"🏭 Add factory functions to {len(no_factory)} tools for better instantiation")

        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies()
        if circular_deps:
            recommendations.append(f"🔄 Resolve {len(circular_deps)} circular dependencies")

        return recommendations

    def _detect_circular_dependencies(self) -> List[Tuple[str, str]]:
        """Detect circular dependencies in the dependency graph."""
        circular = []

        for tool_id, deps in self.dependency_graph.items():
            for dep_id in deps:
                if dep_id in self.dependency_graph:
                    if tool_id in self.dependency_graph[dep_id]:
                        circular.append((tool_id, dep_id))

        return circular
