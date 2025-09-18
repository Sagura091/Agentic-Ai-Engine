-- Database initialization script for Agentic AI System
-- This script sets up the basic database structure

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type VARCHAR(100) NOT NULL DEFAULT 'general',
    model VARCHAR(255) NOT NULL DEFAULT 'llama3.2:latest',
    capabilities JSONB DEFAULT '[]',
    tools JSONB DEFAULT '[]',
    system_prompt TEXT,
    temperature FLOAT DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 2048,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    nodes JSONB DEFAULT '[]',
    edges JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_executed TIMESTAMP WITH TIME ZONE,
    execution_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- Create workflow_executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID REFERENCES workflows(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'running',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- Create custom_tools table
CREATE TABLE IF NOT EXISTS custom_tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100) DEFAULT 'general',
    parameters JSONB DEFAULT '[]',
    code TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create agent_conversations table
CREATE TABLE IF NOT EXISTS agent_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    session_id VARCHAR(255),
    messages JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create system_metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_created_at ON agents(created_at);

CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON workflows(created_at);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at);

CREATE INDEX IF NOT EXISTS idx_custom_tools_name ON custom_tools(name);
CREATE INDEX IF NOT EXISTS idx_custom_tools_category ON custom_tools(category);

CREATE INDEX IF NOT EXISTS idx_agent_conversations_agent_id ON agent_conversations(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_conversations_session_id ON agent_conversations(session_id);

CREATE INDEX IF NOT EXISTS idx_system_metrics_type_name ON system_metrics(metric_type, metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_custom_tools_updated_at BEFORE UPDATE ON custom_tools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_conversations_updated_at BEFORE UPDATE ON agent_conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing
INSERT INTO agents (name, description, agent_type, model, capabilities, system_prompt) VALUES
('General Assistant', 'A general-purpose AI assistant', 'general', 'llama3.2:latest', '["reasoning", "conversation"]', 'You are a helpful AI assistant.'),
('Research Specialist', 'Specialized in research and analysis', 'research', 'llama3.1:latest', '["research", "analysis", "reasoning"]', 'You are a research specialist focused on gathering and analyzing information.'),
('Code Generator', 'Specialized in code generation and programming', 'workflow', 'qwen2.5:latest', '["code-generation", "reasoning"]', 'You are a programming assistant specialized in code generation and software development.');

INSERT INTO custom_tools (name, description, category, parameters, code) VALUES
('simple_calculator', 'A basic calculator for mathematical operations', 'calculations', 
'[{"name": "expression", "type": "string", "description": "Mathematical expression to evaluate", "required": true}]',
'def execute_tool(**kwargs):
    expression = kwargs.get("expression", "")
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"'),
('text_analyzer', 'Analyze text for various metrics', 'text-processing',
'[{"name": "text", "type": "string", "description": "Text to analyze", "required": true}]',
'def execute_tool(**kwargs):
    text = kwargs.get("text", "")
    word_count = len(text.split())
    char_count = len(text)
    return f"Words: {word_count}, Characters: {char_count}"');

-- Create a view for agent statistics
CREATE OR REPLACE VIEW agent_stats AS
SELECT 
    agent_type,
    COUNT(*) as total_agents,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_agents,
    AVG(temperature) as avg_temperature
FROM agents
GROUP BY agent_type;

-- Create a view for workflow statistics
CREATE OR REPLACE VIEW workflow_stats AS
SELECT 
    status,
    COUNT(*) as total_workflows,
    AVG(execution_count) as avg_executions
FROM workflows
GROUP BY status;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agentic;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agentic;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO agentic;
