/**
 * Data validation and alignment utilities for frontend-backend communication.
 * 
 * This module ensures proper data formatting and validation for all API requests
 * to maintain consistency between frontend and backend data structures.
 */

import { z } from 'zod'

// Agent-related schemas
export const AgentCreateSchema = z.object({
  name: z.string().min(1, 'Agent name is required').max(100, 'Name too long'),
  description: z.string().min(1, 'Description is required').max(500, 'Description too long'),
  agent_type: z.enum(['basic', 'autonomous', 'research', 'creative', 'optimization', 'custom']),
  model: z.string().min(1, 'Model is required'),
  model_provider: z.string().default('ollama'),
  capabilities: z.array(z.string()).default([]),
  tools: z.array(z.string()).default([]),
  system_prompt: z.string().optional(),
  temperature: z.number().min(0).max(2).default(0.7),
  max_tokens: z.number().min(256).max(8192).default(2048),
  config: z.record(z.any()).optional()
})

export const AgentChatSchema = z.object({
  message: z.string().min(1, 'Message is required'),
  agent_id: z.string().optional(),
  agent_type: z.string().optional(),
  model: z.string().optional(),
  conversation_id: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  max_tokens: z.number().min(256).max(8192).optional(),
  context: z.record(z.any()).optional()
})

// RAG-related schemas
export const KnowledgeSearchSchema = z.object({
  query: z.string().min(1, 'Search query is required'),
  collection: z.string().optional(),
  top_k: z.number().min(1).max(50).default(10),
  filters: z.record(z.any()).default({})
})

export const DocumentIngestSchema = z.object({
  title: z.string().min(1, 'Document title is required'),
  content: z.string().min(1, 'Document content is required'),
  collection: z.string().optional(),
  metadata: z.record(z.any()).default({}),
  document_type: z.string().default('text')
})

export const FileUploadSchema = z.object({
  file: z.instanceof(File, { message: 'Valid file is required' }),
  collection: z.string().optional()
})

// Embedding-related schemas
export const EmbeddingConfigSchema = z.object({
  current_model: z.string().min(1, 'Model name is required'),
  batch_size: z.number().min(1).max(128).default(32),
  max_length: z.number().min(128).max(2048).default(512),
  normalize: z.boolean().default(true),
  cache_embeddings: z.boolean().default(true),
  device: z.enum(['auto', 'cpu', 'cuda']).default('auto')
})

// Task execution schemas
export const TaskExecutionSchema = z.object({
  agent_id: z.string().min(1, 'Agent ID is required'),
  task: z.string().min(1, 'Task description is required'),
  context: z.object({
    task_type: z.enum(['document_analysis', 'knowledge_search', 'data_extraction', 'content_generation', 'custom']).optional(),
    input_data: z.record(z.any()).optional(),
    execution_config: z.object({
      use_rag: z.boolean().default(true),
      rag_collection: z.string().optional(),
      max_tokens: z.number().min(256).max(8192).default(2048),
      temperature: z.number().min(0).max(2).default(0.7),
      timeout_seconds: z.number().min(30).max(1800).default(300)
    }).optional()
  }).optional()
})

// Workflow-related schemas
export const WorkflowExecutionSchema = z.object({
  task: z.string().min(1, 'Task is required'),
  workflow_type: z.string().optional(),
  model: z.string().optional(),
  context: z.record(z.any()).optional(),
  timeout: z.number().min(30).max(1800).optional()
})

// Type definitions derived from schemas
export type AgentCreateData = z.infer<typeof AgentCreateSchema>
export type AgentChatData = z.infer<typeof AgentChatSchema>
export type KnowledgeSearchData = z.infer<typeof KnowledgeSearchSchema>
export type DocumentIngestData = z.infer<typeof DocumentIngestSchema>
export type FileUploadData = z.infer<typeof FileUploadSchema>
export type EmbeddingConfigData = z.infer<typeof EmbeddingConfigSchema>
export type TaskExecutionData = z.infer<typeof TaskExecutionSchema>
export type WorkflowExecutionData = z.infer<typeof WorkflowExecutionSchema>

// Validation functions
export const validateAgentCreate = (data: unknown): AgentCreateData => {
  return AgentCreateSchema.parse(data)
}

export const validateAgentChat = (data: unknown): AgentChatData => {
  return AgentChatSchema.parse(data)
}

export const validateKnowledgeSearch = (data: unknown): KnowledgeSearchData => {
  return KnowledgeSearchSchema.parse(data)
}

export const validateDocumentIngest = (data: unknown): DocumentIngestData => {
  return DocumentIngestSchema.parse(data)
}

export const validateFileUpload = (data: unknown): FileUploadData => {
  return FileUploadSchema.parse(data)
}

export const validateEmbeddingConfig = (data: unknown): EmbeddingConfigData => {
  return EmbeddingConfigSchema.parse(data)
}

export const validateTaskExecution = (data: unknown): TaskExecutionData => {
  return TaskExecutionSchema.parse(data)
}

export const validateWorkflowExecution = (data: unknown): WorkflowExecutionData => {
  return WorkflowExecutionSchema.parse(data)
}

// Data transformation utilities
export const transformAgentDataForBackend = (frontendData: any): AgentCreateData => {
  return validateAgentCreate({
    name: frontendData.name,
    description: frontendData.description,
    agent_type: frontendData.agentType || frontendData.agent_type || 'basic',
    model: frontendData.model || 'llama3.2:latest',
    model_provider: frontendData.modelProvider || frontendData.model_provider || 'ollama',
    capabilities: Array.isArray(frontendData.capabilities) ? frontendData.capabilities : [],
    tools: Array.isArray(frontendData.tools) ? frontendData.tools : [],
    system_prompt: frontendData.systemPrompt || frontendData.system_prompt,
    temperature: typeof frontendData.temperature === 'number' ? frontendData.temperature : 0.7,
    max_tokens: typeof frontendData.maxTokens === 'number' ? frontendData.maxTokens : 
                typeof frontendData.max_tokens === 'number' ? frontendData.max_tokens : 2048,
    config: frontendData.config || {}
  })
}

export const transformChatDataForBackend = (frontendData: any): AgentChatData => {
  return validateAgentChat({
    message: frontendData.message,
    agent_id: frontendData.agentId || frontendData.agent_id,
    agent_type: frontendData.agentType || frontendData.agent_type,
    model: frontendData.model,
    conversation_id: frontendData.conversationId || frontendData.conversation_id,
    temperature: frontendData.temperature,
    max_tokens: frontendData.maxTokens || frontendData.max_tokens,
    context: frontendData.context || {}
  })
}

export const transformRAGSearchForBackend = (frontendData: any): KnowledgeSearchData => {
  return validateKnowledgeSearch({
    query: frontendData.query,
    collection: frontendData.collection,
    top_k: frontendData.topK || frontendData.top_k || 10,
    filters: frontendData.filters || {}
  })
}

export const transformTaskDataForBackend = (frontendData: any): TaskExecutionData => {
  return validateTaskExecution({
    agent_id: frontendData.agentId || frontendData.agent_id,
    task: frontendData.taskDescription || frontendData.task,
    context: {
      task_type: frontendData.taskType || frontendData.task_type,
      input_data: frontendData.inputData || frontendData.input_data || {},
      execution_config: {
        use_rag: frontendData.executionConfig?.useRAG ?? frontendData.execution_config?.use_rag ?? true,
        rag_collection: frontendData.executionConfig?.ragCollection || frontendData.execution_config?.rag_collection,
        max_tokens: frontendData.executionConfig?.maxTokens || frontendData.execution_config?.max_tokens || 2048,
        temperature: frontendData.executionConfig?.temperature || frontendData.execution_config?.temperature || 0.7,
        timeout_seconds: frontendData.executionConfig?.timeoutSeconds || frontendData.execution_config?.timeout_seconds || 300
      }
    }
  })
}

// Response validation schemas
export const AgentResponseSchema = z.object({
  agent_id: z.string(),
  name: z.string(),
  description: z.string(),
  agent_type: z.string(),
  model: z.string(),
  status: z.enum(['active', 'idle', 'executing', 'error']),
  capabilities: z.array(z.string()),
  tools: z.array(z.string()),
  created_at: z.string(),
  last_activity: z.string().optional()
})

export const ChatResponseSchema = z.object({
  response: z.string(),
  agent_id: z.string().optional(),
  conversation_id: z.string().optional(),
  execution_time: z.number().optional(),
  tokens_used: z.number().optional(),
  model_used: z.string().optional()
})

export const RAGSearchResponseSchema = z.object({
  success: z.boolean(),
  query: z.string(),
  results: z.array(z.object({
    content: z.string(),
    score: z.number(),
    metadata: z.record(z.any())
  })),
  total_results: z.number(),
  processing_time: z.number(),
  collection: z.string()
})

// Response validation functions
export const validateAgentResponse = (data: unknown) => {
  return AgentResponseSchema.parse(data)
}

export const validateChatResponse = (data: unknown) => {
  return ChatResponseSchema.parse(data)
}

export const validateRAGSearchResponse = (data: unknown) => {
  return RAGSearchResponseSchema.parse(data)
}

// Error handling utilities
export const formatValidationError = (error: z.ZodError): string => {
  return error.errors.map(err => `${err.path.join('.')}: ${err.message}`).join(', ')
}

export const isValidationError = (error: any): error is z.ZodError => {
  return error instanceof z.ZodError
}

// File validation utilities
export const validateFileType = (file: File, allowedTypes: string[]): boolean => {
  return allowedTypes.some(type => {
    if (type.startsWith('.')) {
      return file.name.toLowerCase().endsWith(type.toLowerCase())
    }
    return file.type.toLowerCase().includes(type.toLowerCase())
  })
}

export const validateFileSize = (file: File, maxSizeMB: number): boolean => {
  const maxSizeBytes = maxSizeMB * 1024 * 1024
  return file.size <= maxSizeBytes
}

export const ALLOWED_DOCUMENT_TYPES = ['.pdf', '.docx', '.txt', '.md', '.json', '.csv']
export const MAX_FILE_SIZE_MB = 50

export const validateDocumentFile = (file: File): { valid: boolean; error?: string } => {
  if (!validateFileType(file, ALLOWED_DOCUMENT_TYPES)) {
    return {
      valid: false,
      error: `File type not supported. Allowed types: ${ALLOWED_DOCUMENT_TYPES.join(', ')}`
    }
  }
  
  if (!validateFileSize(file, MAX_FILE_SIZE_MB)) {
    return {
      valid: false,
      error: `File size too large. Maximum size: ${MAX_FILE_SIZE_MB}MB`
    }
  }
  
  return { valid: true }
}
