-- Create admin settings tables for enhanced admin settings functionality
-- Based on the migration file: app/database/migrations/add_admin_settings_tables.py

-- Create admin_settings table
CREATE TABLE IF NOT EXISTS admin_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category VARCHAR(100) NOT NULL,
    key VARCHAR(200) NOT NULL,
    value JSONB NOT NULL,
    default_value JSONB NOT NULL,
    setting_type VARCHAR(50) NOT NULL,
    description TEXT,
    security_level VARCHAR(50) DEFAULT 'admin_only',
    requires_restart BOOLEAN DEFAULT FALSE,
    validation_rules JSONB DEFAULT '{}',
    enum_values JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by UUID,
    is_active BOOLEAN DEFAULT TRUE,
    is_system_managed BOOLEAN DEFAULT FALSE
);

-- Create unique index on category + key
CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_settings_category_key 
ON admin_settings (category, key);

-- Create additional indexes
CREATE INDEX IF NOT EXISTS idx_admin_settings_category ON admin_settings (category);
CREATE INDEX IF NOT EXISTS idx_admin_settings_updated ON admin_settings (updated_at);
CREATE INDEX IF NOT EXISTS idx_admin_settings_active ON admin_settings (is_active);

-- Create admin_setting_history table
CREATE TABLE IF NOT EXISTS admin_setting_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setting_id UUID NOT NULL,
    category VARCHAR(100) NOT NULL,
    key VARCHAR(200) NOT NULL,
    old_value JSONB,
    new_value JSONB NOT NULL,
    change_reason TEXT,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    changed_by UUID NOT NULL,
    system_restart_required BOOLEAN DEFAULT FALSE
);

-- Create indexes for admin_setting_history
CREATE INDEX IF NOT EXISTS idx_admin_setting_history_setting_id ON admin_setting_history (setting_id);
CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_at ON admin_setting_history (changed_at);
CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_by ON admin_setting_history (changed_by);

-- Create system_configuration_cache table
CREATE TABLE IF NOT EXISTS system_configuration_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_name VARCHAR(100) NOT NULL UNIQUE,
    configuration JSONB NOT NULL,
    configuration_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    requires_restart BOOLEAN DEFAULT FALSE
);

-- Create indexes for system_configuration_cache
CREATE INDEX IF NOT EXISTS idx_system_config_cache_component ON system_configuration_cache (component_name);
CREATE INDEX IF NOT EXISTS idx_system_config_cache_updated ON system_configuration_cache (updated_at);

-- Insert some default settings to get started
INSERT INTO admin_settings (category, key, value, default_value, setting_type, description) VALUES
('system_configuration', 'app_name', '"Agentic AI Microservice"', '"Agentic AI Platform"', 'string', 'Application name displayed in UI'),
('system_configuration', 'app_version', '"0.1.0"', '"0.1.0"', 'string', 'Application version'),
('system_configuration', 'debug_mode', 'false', 'false', 'boolean', 'Enable debug mode'),
('system_configuration', 'max_concurrent_agents', '10', '10', 'integer', 'Maximum number of concurrent agents'),
('llm_providers', 'ollama_enabled', 'true', 'true', 'boolean', 'Enable Ollama provider'),
('llm_providers', 'ollama_base_url', '"http://localhost:11434"', '"http://localhost:11434"', 'string', 'Ollama base URL'),
('llm_providers', 'openai_enabled', 'false', 'false', 'boolean', 'Enable OpenAI provider'),
('llm_providers', 'anthropic_enabled', 'false', 'false', 'boolean', 'Enable Anthropic provider'),
('rag_system', 'chunk_size', '1000', '1000', 'integer', 'Default chunk size for document processing'),
('rag_system', 'chunk_overlap', '200', '200', 'integer', 'Overlap between chunks'),
('rag_system', 'top_k', '5', '5', 'integer', 'Number of top results to retrieve')
ON CONFLICT (category, key) DO NOTHING;

GRANT ALL PRIVILEGES ON admin_settings TO agentic_user;
GRANT ALL PRIVILEGES ON admin_setting_history TO agentic_user;
GRANT ALL PRIVILEGES ON system_configuration_cache TO agentic_user;
