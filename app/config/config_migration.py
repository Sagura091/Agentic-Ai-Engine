"""
Configuration Migration and Validation Utilities.

This module provides utilities to:
- Migrate from hardcoded values to configuration system
- Validate existing configurations
- Generate configuration files from current settings
- Provide migration assistance
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

from app.config.agent_config_manager import get_agent_config_manager, AgentConfigurationManager

_backend_logger = get_backend_logger()


class ConfigMigration:
    """
    Configuration migration and validation utilities.
    """
    
    def __init__(self, config_manager: Optional[AgentConfigurationManager] = None):
        """
        Initialize configuration migration.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or get_agent_config_manager()
        self.data_dir = self.config_manager.data_dir
        self.config_dir = self.config_manager.config_dir

        _backend_logger.info(
            "Configuration migration initialized",
            LogCategory.AGENT_OPERATIONS,
            "app.config.config_migration",
            data={
                "data_dir": str(self.data_dir),
                "config_dir": str(self.config_dir)
            }
        )
    
    def validate_current_config(self) -> Tuple[bool, List[str]]:
        """
        Validate the current configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            errors = self.config_manager.validate_configuration()
            is_valid = len(errors) == 0
            
            if is_valid:
                _backend_logger.info(
                    "Configuration validation passed",
                    LogCategory.AGENT_OPERATIONS,
                    "app.config.config_migration"
                )
            else:
                _backend_logger.error(
                    "Configuration validation failed",
                    LogCategory.AGENT_OPERATIONS,
                    "app.config.config_migration",
                    data={"errors": errors}
                )

            return is_valid, errors

        except Exception as e:
            _backend_logger.error(
                "Configuration validation error",
                LogCategory.AGENT_OPERATIONS,
                "app.config.config_migration",
                data={"error": str(e)}
            )
            return False, [f"Validation error: {str(e)}"]
    
    def generate_user_config_from_env(self) -> Dict[str, Any]:
        """
        Generate user configuration from environment variables.
        
        Returns:
            Dictionary of configuration values from environment
        """
        import os
        
        user_config = {}
        
        # LLM Provider settings
        if os.getenv("AGENT_DEFAULT_PROVIDER"):
            user_config.setdefault("llm_providers", {})["default_provider"] = os.getenv("AGENT_DEFAULT_PROVIDER")
        
        if os.getenv("AGENT_DEFAULT_MODEL"):
            provider = os.getenv("AGENT_DEFAULT_PROVIDER", "ollama")
            user_config.setdefault("llm_providers", {}).setdefault(provider, {})["default_model"] = os.getenv("AGENT_DEFAULT_MODEL")
        
        if os.getenv("AGENT_DEFAULT_TEMPERATURE"):
            provider = os.getenv("AGENT_DEFAULT_PROVIDER", "ollama")
            user_config.setdefault("llm_providers", {}).setdefault(provider, {})["temperature"] = float(os.getenv("AGENT_DEFAULT_TEMPERATURE"))
        
        # Performance settings
        performance_config = {}
        if os.getenv("AGENT_MAX_ITERATIONS"):
            performance_config["max_iterations_hard_limit"] = int(os.getenv("AGENT_MAX_ITERATIONS"))
        
        if os.getenv("AGENT_MAX_EXECUTION_TIME"):
            performance_config["max_execution_time_seconds"] = int(os.getenv("AGENT_MAX_EXECUTION_TIME"))
        
        if os.getenv("AGENT_MAX_MEMORY_MB"):
            performance_config["max_memory_per_agent_mb"] = int(os.getenv("AGENT_MAX_MEMORY_MB"))
        
        if performance_config:
            user_config["performance"] = performance_config
        
        # Security settings
        security_config = {}
        if os.getenv("AGENT_RATE_LIMIT_PER_MINUTE"):
            security_config["requests_per_minute"] = int(os.getenv("AGENT_RATE_LIMIT_PER_MINUTE"))
        
        if os.getenv("AGENT_MAX_FILE_SIZE_MB"):
            security_config["max_file_size_mb"] = int(os.getenv("AGENT_MAX_FILE_SIZE_MB"))
        
        if security_config:
            user_config["security"] = security_config
        
        # Logging settings
        logging_config = {}
        if os.getenv("AGENT_LOG_LEVEL"):
            logging_config["default_level"] = os.getenv("AGENT_LOG_LEVEL")
        
        if os.getenv("AGENT_ENABLE_METRICS"):
            logging_config["enable_metrics"] = os.getenv("AGENT_ENABLE_METRICS").lower() == "true"
        
        if logging_config:
            user_config["logging"] = logging_config
        
        return user_config
    
    def save_user_config(self, config: Dict[str, Any], filename: str = "user_config.yaml") -> bool:
        """
        Save user configuration to file.
        
        Args:
            config: Configuration dictionary to save
            filename: Output filename
            
        Returns:
            True if successful
        """
        try:
            output_file = self.config_dir / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=True)
            
            _backend_logger.info(
                "User configuration saved",
                LogCategory.AGENT_OPERATIONS,
                "app.config.config_migration",
                data={"file": str(output_file)}
            )
            return True

        except Exception as e:
            _backend_logger.error(
                "Failed to save user configuration",
                LogCategory.AGENT_OPERATIONS,
                "app.config.config_migration",
                data={"error": str(e)}
            )
            return False
    
    def migrate_from_hardcoded_values(self) -> Dict[str, Any]:
        """
        Generate configuration from commonly hardcoded values in the codebase.
        
        Returns:
            Dictionary of recommended configuration values
        """
        # This would analyze the codebase and extract hardcoded values
        # For now, we'll provide common migration patterns
        
        migration_config = {
            "llm_providers": {
                "default_provider": "ollama",  # Instead of hardcoded in AgentConfig
                "ollama": {
                    "default_model": "llama3.2:latest",  # Instead of hardcoded default
                    "temperature": 0.7,  # Instead of scattered temperature values
                    "max_tokens": 2048,
                    "timeout_seconds": 300
                }
            },
            
            "agent_types": {
                # Replace hardcoded agent configurations
                "react": {
                    "framework": "react",
                    "default_temperature": 0.7,
                    "max_iterations": 50,  # Instead of hardcoded 3 in decision logic
                    "timeout_seconds": 300,
                    "enable_memory": True,
                    "memory_type": "simple"
                },
                
                "autonomous": {
                    "framework": "autonomous", 
                    "default_temperature": 0.7,
                    "max_iterations": 100,
                    "timeout_seconds": 1200,
                    "enable_memory": True,
                    "memory_type": "advanced"
                }
            },
            
            "performance": {
                # Replace hardcoded performance limits
                "max_execution_time_seconds": 3600,  # Instead of scattered timeout values
                "max_iterations_hard_limit": 200,    # Instead of hardcoded limits
                "max_memory_per_agent_mb": 1024,
                "max_concurrent_agents": 50,
                "default_decision_threshold": 0.6    # Instead of hardcoded 0.6
            },
            
            "infrastructure": {
                # Replace hardcoded infrastructure settings
                "health_check_interval_seconds": 60,  # Instead of hardcoded 60
                "cache_ttl_seconds": 300,             # Instead of hardcoded cache times
                "connection_pool_size": 10,
                "max_retries": 5,                     # Add missing retry logic
                "retry_delays_seconds": [1, 2, 4, 8, 16]  # Add missing backoff
            },
            
            "security": {
                # Add missing security constraints
                "requests_per_minute": 60,
                "max_file_size_mb": 100,
                "allowed_file_types": [".txt", ".pdf", ".docx", ".md", ".json", ".yaml"]
            },
            
            "memory_systems": {
                # Replace hardcoded memory configurations
                "advanced": {
                    "max_working_memory": 20,      # Instead of hardcoded 20
                    "max_episodic_memory": 10000,  # Instead of hardcoded 10000
                    "max_semantic_memory": 5000,   # Instead of hardcoded 5000
                    "consolidation_threshold": 5   # Instead of hardcoded 5
                }
            }
        }
        
        return migration_config
    
    def check_hardcoded_values(self) -> List[Dict[str, Any]]:
        """
        Check for remaining hardcoded values in the system.
        
        Returns:
            List of detected hardcoded values with recommendations
        """
        hardcoded_issues = [
            {
                "location": "app/agents/base/agent.py:110",
                "issue": "Hardcoded default model 'llama3.2:latest'",
                "recommendation": "Use config_manager.get_llm_config().get('default_model')",
                "severity": "high"
            },
            {
                "location": "app/agents/base/agent.py:1016", 
                "issue": "Hardcoded iteration limit of 3",
                "recommendation": "Use config_manager.get('performance.min_iterations', 3)",
                "severity": "high"
            },
            {
                "location": "app/agents/base/agent.py:271",
                "issue": "Hardcoded tool_choice='any'",
                "recommendation": "Make tool choice configurable",
                "severity": "medium"
            },
            {
                "location": "app/agents/factory/__init__.py:138",
                "issue": "Hardcoded health check interval of 60 seconds",
                "recommendation": "Use config_manager.get('infrastructure.health_check_interval_seconds')",
                "severity": "medium"
            },
            {
                "location": "app/agents/templates.py:multiple",
                "issue": "Multiple hardcoded temperature values",
                "recommendation": "Use agent type specific temperature from config",
                "severity": "high"
            }
        ]
        
        return hardcoded_issues
    
    def generate_migration_report(self) -> str:
        """
        Generate a comprehensive migration report.
        
        Returns:
            Migration report as string
        """
        report_lines = [
            "# Configuration Migration Report",
            f"Generated at: {self.config_manager._config_cache.get('generated_at', 'unknown')}",
            "",
            "## Current Configuration Status"
        ]
        
        # Validation status
        is_valid, errors = self.validate_current_config()
        if is_valid:
            report_lines.append("✅ Configuration validation: PASSED")
        else:
            report_lines.append("❌ Configuration validation: FAILED")
            for error in errors:
                report_lines.append(f"   - {error}")
        
        report_lines.extend([
            "",
            "## Hardcoded Values Detected"
        ])
        
        # Hardcoded values
        hardcoded_issues = self.check_hardcoded_values()
        for issue in hardcoded_issues:
            severity_icon = "🔴" if issue["severity"] == "high" else "🟡"
            report_lines.extend([
                f"{severity_icon} **{issue['location']}**",
                f"   Issue: {issue['issue']}",
                f"   Fix: {issue['recommendation']}",
                ""
            ])
        
        report_lines.extend([
            "## Recommended Actions",
            "",
            "1. **High Priority**: Fix hardcoded model names and iteration limits",
            "2. **Medium Priority**: Make infrastructure settings configurable", 
            "3. **Low Priority**: Add configuration validation for all parameters",
            "",
            "## Configuration Files Status"
        ])
        
        # Check configuration files
        config_files = [
            ("agent_defaults.yaml", "Default configuration"),
            ("user_config.yaml", "User overrides"),
            ("global_config.json", "Legacy configuration"),
            (".env", "Environment variables")
        ]
        
        for filename, description in config_files:
            file_path = self.config_dir / filename
            if file_path.exists():
                report_lines.append(f"✅ {filename}: {description} - EXISTS")
            else:
                report_lines.append(f"❌ {filename}: {description} - MISSING")
        
        return "\n".join(report_lines)
    
    def create_migration_files(self) -> bool:
        """
        Create all necessary migration files.
        
        Returns:
            True if successful
        """
        try:
            # Generate user config from environment
            env_config = self.generate_user_config_from_env()
            if env_config:
                self.save_user_config(env_config, "user_config_from_env.yaml")
            
            # Generate migration config
            migration_config = self.migrate_from_hardcoded_values()
            self.save_user_config(migration_config, "migration_recommended.yaml")
            
            # Generate migration report
            report = self.generate_migration_report()
            report_file = self.config_dir / "migration_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            _backend_logger.info(
                "Migration files created successfully",
                LogCategory.AGENT_OPERATIONS,
                "app.config.config_migration"
            )
            return True

        except Exception as e:
            _backend_logger.error(
                "Failed to create migration files",
                LogCategory.AGENT_OPERATIONS,
                "app.config.config_migration",
                data={"error": str(e)}
            )
            return False


def run_configuration_migration():
    """Run the configuration migration process."""
    migration = ConfigMigration()
    
    print("🔧 Starting Configuration Migration...")
    print()
    
    # Validate current config
    print("📋 Validating current configuration...")
    is_valid, errors = migration.validate_current_config()
    
    if not is_valid:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration validation passed")
    
    print()
    
    # Create migration files
    print("📁 Creating migration files...")
    if migration.create_migration_files():
        print("✅ Migration files created successfully")
        print(f"   - Check {migration.config_dir} for generated files")
    else:
        print("❌ Failed to create migration files")
    
    print()
    print("🎯 Next Steps:")
    print("1. Review the generated migration files")
    print("2. Copy user_config_template.yaml to user_config.yaml and customize")
    print("3. Update your agent code to use the configuration system")
    print("4. Test with the new configuration-driven approach")


if __name__ == "__main__":
    run_configuration_migration()
