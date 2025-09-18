-- Create Autonomous Agent Tables
-- This script creates all tables needed for truly agentic AI

-- Create autonomous agent states table
CREATE TABLE IF NOT EXISTS autonomous_agent_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL,
    session_id VARCHAR(255),
    autonomy_level VARCHAR(50) NOT NULL DEFAULT 'adaptive',
    decision_confidence FLOAT DEFAULT 0.0,
    learning_enabled BOOLEAN DEFAULT true,
    current_task TEXT,
    tools_available JSONB DEFAULT '[]'::jsonb,
    outputs JSONB DEFAULT '{}'::jsonb,
    errors JSONB DEFAULT '[]'::jsonb,
    iteration_count INTEGER DEFAULT 0,
    max_iterations INTEGER DEFAULT 50,
    custom_state JSONB DEFAULT '{}'::jsonb,
    goal_stack JSONB DEFAULT '[]'::jsonb,
    context_memory JSONB DEFAULT '{}'::jsonb,
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    self_initiated_tasks JSONB DEFAULT '[]'::jsonb,
    proactive_actions JSONB DEFAULT '[]'::jsonb,
    emergent_behaviors JSONB DEFAULT '[]'::jsonb,
    collaboration_state JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE
);

-- Create autonomous goals table
CREATE TABLE IF NOT EXISTS autonomous_goals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    goal_id VARCHAR(255) UNIQUE NOT NULL,
    agent_state_id UUID REFERENCES autonomous_agent_states(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    goal_type VARCHAR(50) NOT NULL,
    priority VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    target_outcome JSONB DEFAULT '{}'::jsonb,
    success_criteria JSONB DEFAULT '[]'::jsonb,
    context JSONB DEFAULT '{}'::jsonb,
    goal_metadata JSONB DEFAULT '{}'::jsonb,
    progress FLOAT DEFAULT 0.0,
    completion_confidence FLOAT DEFAULT 0.0,
    estimated_effort FLOAT DEFAULT 1.0,
    actual_effort FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deadline TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create autonomous decisions table
CREATE TABLE IF NOT EXISTS autonomous_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id VARCHAR(255) UNIQUE NOT NULL,
    agent_state_id UUID REFERENCES autonomous_agent_states(id) ON DELETE CASCADE,
    decision_type VARCHAR(100) NOT NULL,
    context JSONB DEFAULT '{}'::jsonb,
    options_considered JSONB DEFAULT '[]'::jsonb,
    chosen_option JSONB DEFAULT '{}'::jsonb,
    confidence FLOAT NOT NULL,
    reasoning JSONB DEFAULT '[]'::jsonb,
    expected_outcome JSONB DEFAULT '{}'::jsonb,
    actual_outcome JSONB DEFAULT '{}'::jsonb,
    learning_value FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create agent memories table
CREATE TABLE IF NOT EXISTS agent_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id VARCHAR(255) UNIQUE NOT NULL,
    agent_state_id UUID REFERENCES autonomous_agent_states(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    context JSONB DEFAULT '{}'::jsonb,
    importance VARCHAR(50) NOT NULL,
    emotional_valence FLOAT DEFAULT 0.0,
    tags JSONB DEFAULT '[]'::jsonb,
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create learning experiences table
CREATE TABLE IF NOT EXISTS learning_experiences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experience_id VARCHAR(255) UNIQUE NOT NULL,
    agent_state_id UUID REFERENCES autonomous_agent_states(id) ON DELETE CASCADE,
    experience_type VARCHAR(100) NOT NULL,
    context JSONB DEFAULT '{}'::jsonb,
    outcome JSONB DEFAULT '{}'::jsonb,
    feedback TEXT,
    learning_value FLOAT DEFAULT 0.0,
    insights JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_state_id UUID REFERENCES autonomous_agent_states(id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    context JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for performance
CREATE INDEX IF NOT EXISTS idx_autonomous_agent_states_agent_id ON autonomous_agent_states(agent_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_agent_states_session_id ON autonomous_agent_states(session_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_agent_states_last_accessed ON autonomous_agent_states(last_accessed);

CREATE INDEX IF NOT EXISTS idx_autonomous_goals_agent_state_id ON autonomous_goals(agent_state_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_goals_status ON autonomous_goals(status);
CREATE INDEX IF NOT EXISTS idx_autonomous_goals_priority ON autonomous_goals(priority);

CREATE INDEX IF NOT EXISTS idx_autonomous_decisions_agent_state_id ON autonomous_decisions(agent_state_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_decisions_decision_type ON autonomous_decisions(decision_type);

CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_state_id ON agent_memories(agent_state_id);
CREATE INDEX IF NOT EXISTS idx_agent_memories_memory_type ON agent_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_agent_memories_importance ON agent_memories(importance);
CREATE INDEX IF NOT EXISTS idx_agent_memories_session_id ON agent_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_memories_expires_at ON agent_memories(expires_at);

CREATE INDEX IF NOT EXISTS idx_learning_experiences_agent_state_id ON learning_experiences(agent_state_id);
CREATE INDEX IF NOT EXISTS idx_learning_experiences_experience_type ON learning_experiences(experience_type);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_agent_state_id ON performance_metrics(agent_state_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_metric_name ON performance_metrics(metric_name);

-- Create full-text search indices for content
CREATE INDEX IF NOT EXISTS idx_agent_memories_content_gin ON agent_memories USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_autonomous_goals_description_gin ON autonomous_goals USING gin(to_tsvector('english', description));

-- Create triggers for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_autonomous_agent_states_updated_at BEFORE UPDATE ON autonomous_agent_states FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_autonomous_goals_updated_at BEFORE UPDATE ON autonomous_goals FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_autonomous_decisions_updated_at BEFORE UPDATE ON autonomous_decisions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agent_memories_updated_at BEFORE UPDATE ON agent_memories FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_learning_experiences_updated_at BEFORE UPDATE ON learning_experiences FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Log successful creation
DO $$
BEGIN
    RAISE NOTICE 'Autonomous agent tables created successfully';
    RAISE NOTICE 'Tables: autonomous_agent_states, autonomous_goals, autonomous_decisions, agent_memories, learning_experiences, performance_metrics';
END $$;
