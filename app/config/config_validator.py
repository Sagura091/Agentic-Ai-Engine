"""
Configuration Validator - Startup Validation System.

This module provides comprehensive validation of all configuration settings
at application startup, ensuring the system is properly configured before
accepting requests.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis
from pydantic import ValidationError

from app.config.config_groups import (
    ServerConfig,
    DatabaseConfig,
    RedisConfig,
    LLMProviderConfig,
    RAGConfig,
    SecurityConfig,
    PerformanceConfig,
    LoggingConfig,
    ConfigProfile
)


class ValidationLevel(str, Enum):
    """Validation severity levels."""
    CRITICAL = "critical"  # Must pass or app won't start
    WARNING = "warning"    # Should pass but app can start
    INFO = "info"          # Informational only


class ValidationResult:
    """Result of a configuration validation check."""
    
    def __init__(
        self,
        check_name: str,
        passed: bool,
        level: ValidationLevel,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.check_name = check_name
        self.passed = passed
        self.level = level
        self.message = message
        self.details = details or {}
    
    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"{status} [{self.level.value.upper()}] {self.check_name}: {self.message}"


class ConfigValidator:
    """Comprehensive configuration validator."""
    
    def __init__(
        self,
        server_config: ServerConfig,
        database_config: DatabaseConfig,
        redis_config: RedisConfig,
        llm_config: LLMProviderConfig,
        rag_config: RAGConfig,
        security_config: SecurityConfig,
        performance_config: PerformanceConfig,
        logging_config: LoggingConfig
    ):
        self.server = server_config
        self.database = database_config
        self.redis = redis_config
        self.llm = llm_config
        self.rag = rag_config
        self.security = security_config
        self.performance = performance_config
        self.logging = logging_config
        
        self.results: List[ValidationResult] = []
    
    async def validate_all(self, skip_connectivity: bool = False) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.
        
        Args:
            skip_connectivity: Skip connectivity checks (useful for testing)
        
        Returns:
            Tuple of (all_critical_passed, results)
        """
        self.results = []
        
        # Configuration validation
        self._validate_server_config()
        self._validate_database_config()
        self._validate_redis_config()
        self._validate_llm_config()
        self._validate_rag_config()
        self._validate_security_config()
        self._validate_performance_config()
        self._validate_logging_config()
        
        # Connectivity validation (if not skipped)
        if not skip_connectivity:
            await self._validate_database_connectivity()
            await self._validate_redis_connectivity()
            await self._validate_llm_connectivity()
        
        # Check if all critical validations passed
        critical_failures = [r for r in self.results if not r.passed and r.level == ValidationLevel.CRITICAL]
        all_critical_passed = len(critical_failures) == 0
        
        return all_critical_passed, self.results
    
    def _validate_server_config(self):
        """Validate server configuration."""
        # Port validation
        if self.server.port < 1024 and os.geteuid() != 0 if hasattr(os, 'geteuid') else False:
            self.results.append(ValidationResult(
                "server.port",
                False,
                ValidationLevel.CRITICAL,
                f"Port {self.server.port} requires root privileges"
            ))
        else:
            self.results.append(ValidationResult(
                "server.port",
                True,
                ValidationLevel.INFO,
                f"Server port {self.server.port} is valid"
            ))
        
        # Environment validation
        if self.server.environment == ConfigProfile.PRODUCTION and self.server.debug:
            self.results.append(ValidationResult(
                "server.debug",
                False,
                ValidationLevel.WARNING,
                "Debug mode should be disabled in production"
            ))
        
        # CORS validation
        if self.server.environment == ConfigProfile.PRODUCTION and "*" in self.server.cors_origins:
            self.results.append(ValidationResult(
                "server.cors_origins",
                False,
                ValidationLevel.WARNING,
                "Wildcard CORS origins should not be used in production"
            ))
        else:
            self.results.append(ValidationResult(
                "server.cors_origins",
                True,
                ValidationLevel.INFO,
                f"CORS configured with {len(self.server.cors_origins)} origins"
            ))
    
    def _validate_database_config(self):
        """Validate database configuration."""
        # URL validation
        if "localhost" in self.database.database_url and self.server.environment == ConfigProfile.PRODUCTION:
            self.results.append(ValidationResult(
                "database.url",
                False,
                ValidationLevel.WARNING,
                "Using localhost database in production is not recommended"
            ))
        
        # Pool size validation
        if self.database.pool_size < 10:
            self.results.append(ValidationResult(
                "database.pool_size",
                False,
                ValidationLevel.WARNING,
                f"Pool size {self.database.pool_size} may be too small for production"
            ))
        else:
            self.results.append(ValidationResult(
                "database.pool_size",
                True,
                ValidationLevel.INFO,
                f"Database pool configured: {self.database.pool_size} connections"
            ))
    
    def _validate_redis_config(self):
        """Validate Redis configuration."""
        if "localhost" in self.redis.redis_url and self.server.environment == ConfigProfile.PRODUCTION:
            self.results.append(ValidationResult(
                "redis.url",
                False,
                ValidationLevel.WARNING,
                "Using localhost Redis in production is not recommended"
            ))
        else:
            self.results.append(ValidationResult(
                "redis.url",
                True,
                ValidationLevel.INFO,
                "Redis URL configured"
            ))
    
    def _validate_llm_config(self):
        """Validate LLM provider configuration."""
        enabled_providers = self.llm.get_enabled_providers()
        
        if not enabled_providers:
            self.results.append(ValidationResult(
                "llm.providers",
                False,
                ValidationLevel.CRITICAL,
                "No LLM providers are enabled"
            ))
        else:
            self.results.append(ValidationResult(
                "llm.providers",
                True,
                ValidationLevel.INFO,
                f"Enabled providers: {', '.join(enabled_providers)}"
            ))
        
        # Validate API keys for enabled providers
        if self.llm.openai_enabled and not self.llm.openai_api_key:
            self.results.append(ValidationResult(
                "llm.openai_api_key",
                False,
                ValidationLevel.CRITICAL,
                "OpenAI is enabled but API key is missing"
            ))
        
        if self.llm.anthropic_enabled and not self.llm.anthropic_api_key:
            self.results.append(ValidationResult(
                "llm.anthropic_api_key",
                False,
                ValidationLevel.CRITICAL,
                "Anthropic is enabled but API key is missing"
            ))
        
        if self.llm.google_enabled and not self.llm.google_api_key:
            self.results.append(ValidationResult(
                "llm.google_api_key",
                False,
                ValidationLevel.CRITICAL,
                "Google is enabled but API key is missing"
            ))
    
    def _validate_rag_config(self):
        """Validate RAG configuration."""
        # Check persist directory
        persist_path = Path(self.rag.chroma_persist_directory)
        
        if not persist_path.exists():
            try:
                persist_path.mkdir(parents=True, exist_ok=True)
                self.results.append(ValidationResult(
                    "rag.persist_directory",
                    True,
                    ValidationLevel.INFO,
                    f"Created ChromaDB directory: {persist_path}"
                ))
            except Exception as e:
                self.results.append(ValidationResult(
                    "rag.persist_directory",
                    False,
                    ValidationLevel.CRITICAL,
                    f"Cannot create ChromaDB directory: {e}"
                ))
        else:
            self.results.append(ValidationResult(
                "rag.persist_directory",
                True,
                ValidationLevel.INFO,
                f"ChromaDB directory exists: {persist_path}"
            ))
        
        # Validate chunk sizes
        if self.rag.chunk_overlap >= self.rag.chunk_size:
            self.results.append(ValidationResult(
                "rag.chunk_overlap",
                False,
                ValidationLevel.WARNING,
                "Chunk overlap should be smaller than chunk size"
            ))
    
    def _validate_security_config(self):
        """Validate security configuration."""
        # Secret key validation
        if self.security.secret_key == "your-secret-key-change-this":
            self.results.append(ValidationResult(
                "security.secret_key",
                False,
                ValidationLevel.CRITICAL,
                "Default secret key detected - MUST be changed for production"
            ))
        elif len(self.security.secret_key) < 32:
            self.results.append(ValidationResult(
                "security.secret_key",
                False,
                ValidationLevel.WARNING,
                "Secret key should be at least 32 characters"
            ))
        else:
            self.results.append(ValidationResult(
                "security.secret_key",
                True,
                ValidationLevel.INFO,
                "Secret key is properly configured"
            ))
        
        # Token expiration validation
        if self.security.access_token_expire_minutes > 1440:  # 24 hours
            self.results.append(ValidationResult(
                "security.access_token_expire_minutes",
                False,
                ValidationLevel.WARNING,
                "Access token expiration is very long (>24 hours)"
            ))
    
    def _validate_performance_config(self):
        """Validate performance configuration."""
        # Worker count validation
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if self.performance.async_worker_count > cpu_count * 4:
            self.results.append(ValidationResult(
                "performance.async_worker_count",
                False,
                ValidationLevel.WARNING,
                f"Worker count ({self.performance.async_worker_count}) is very high for {cpu_count} CPUs"
            ))
        else:
            self.results.append(ValidationResult(
                "performance.async_worker_count",
                True,
                ValidationLevel.INFO,
                f"Worker count: {self.performance.async_worker_count} (CPUs: {cpu_count})"
            ))
    
    def _validate_logging_config(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if self.logging.log_level not in valid_levels:
            self.results.append(ValidationResult(
                "logging.log_level",
                False,
                ValidationLevel.WARNING,
                f"Invalid log level: {self.logging.log_level}"
            ))
        else:
            self.results.append(ValidationResult(
                "logging.log_level",
                True,
                ValidationLevel.INFO,
                f"Log level: {self.logging.log_level}"
            ))
    
    async def _validate_database_connectivity(self):
        """Validate database connectivity."""
        try:
            conn = await asyncpg.connect(self.database.database_url, timeout=10)
            await conn.close()
            self.results.append(ValidationResult(
                "database.connectivity",
                True,
                ValidationLevel.INFO,
                "Database connection successful"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                "database.connectivity",
                False,
                ValidationLevel.CRITICAL,
                f"Database connection failed: {str(e)}"
            ))
    
    async def _validate_redis_connectivity(self):
        """Validate Redis connectivity."""
        try:
            r = redis.from_url(self.redis.redis_url, decode_responses=True)
            await r.ping()
            await r.close()
            self.results.append(ValidationResult(
                "redis.connectivity",
                True,
                ValidationLevel.INFO,
                "Redis connection successful"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                "redis.connectivity",
                False,
                ValidationLevel.WARNING,
                f"Redis connection failed: {str(e)}"
            ))
    
    async def _validate_llm_connectivity(self):
        """Validate LLM provider connectivity."""
        # Validate Ollama
        if self.llm.ollama_enabled:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.llm.ollama_base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            self.results.append(ValidationResult(
                                "llm.ollama.connectivity",
                                True,
                                ValidationLevel.INFO,
                                "Ollama connection successful"
                            ))
                        else:
                            self.results.append(ValidationResult(
                                "llm.ollama.connectivity",
                                False,
                                ValidationLevel.WARNING,
                                f"Ollama returned status {resp.status}"
                            ))
            except Exception as e:
                self.results.append(ValidationResult(
                    "llm.ollama.connectivity",
                    False,
                    ValidationLevel.WARNING,
                    f"Ollama connection failed: {str(e)}"
                ))
    
    def print_results(self):
        """Print validation results in a formatted way."""
        print("\n" + "="*80)
        print("CONFIGURATION VALIDATION RESULTS")
        print("="*80 + "\n")
        
        # Group by level
        critical = [r for r in self.results if r.level == ValidationLevel.CRITICAL]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING]
        info = [r for r in self.results if r.level == ValidationLevel.INFO]
        
        # Print critical issues
        if critical:
            print("🔴 CRITICAL ISSUES:")
            for result in critical:
                print(f"  {result}")
            print()
        
        # Print warnings
        if warnings:
            print("⚠️  WARNINGS:")
            for result in warnings:
                print(f"  {result}")
            print()
        
        # Print info (only failed ones)
        failed_info = [r for r in info if not r.passed]
        if failed_info:
            print("ℹ️  INFORMATION:")
            for result in failed_info:
                print(f"  {result}")
            print()
        
        # Summary
        total = len(self.results)
        passed = len([r for r in self.results if r.passed])
        failed = total - passed
        
        print("="*80)
        print(f"SUMMARY: {passed}/{total} checks passed, {failed} failed")
        print("="*80 + "\n")

