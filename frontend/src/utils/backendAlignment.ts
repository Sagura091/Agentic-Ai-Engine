/**
 * Backend alignment utilities for ensuring proper API communication.
 * 
 * This module provides utilities to ensure frontend requests are properly
 * formatted for the backend API endpoints and responses are correctly handled.
 */

import { 
  transformAgentDataForBackend,
  transformChatDataForBackend,
  transformRAGSearchForBackend,
  transformTaskDataForBackend,
  validateAgentResponse,
  validateChatResponse,
  validateRAGSearchResponse,
  formatValidationError,
  isValidationError
} from './dataValidation'

/**
 * Normalize API endpoint path to ensure trailing slash consistency
 * Backend expects trailing slashes to avoid 307 redirects
 */
export function normalizeApiPath(path: string): string {
  // Remove any existing trailing slash
  const cleanPath = path.replace(/\/$/, '');

  // Add trailing slash for consistency with backend expectations
  return `${cleanPath}/`;
}

/**
 * Build full API URL with proper base path and trailing slash
 */
export function buildApiUrl(endpoint: string, baseUrl: string = ''): string {
  const base = baseUrl || (import.meta.env.VITE_SERVER_URL || 'http://localhost:8888');
  const normalizedEndpoint = normalizeApiPath(endpoint);

  // Ensure we have /api/v1 prefix if not already present
  const apiPath = normalizedEndpoint.startsWith('/api/v1')
    ? normalizedEndpoint
    : `/api/v1${normalizedEndpoint}`;

  return `${base}${apiPath}`;
}

// Backend API endpoint mappings
export const API_ENDPOINTS = {
  // Health and system endpoints
  HEALTH: {
    CHECK: normalizeApiPath('/health'),
    STATUS: normalizeApiPath('/health/status'),
    DETAILED: normalizeApiPath('/health/detailed')
  },

  // Models endpoints
  MODELS: {
    LIST: normalizeApiPath('/models'),
    DETAILS: (modelId: string) => normalizeApiPath(`/models/${modelId}`)
  },

  // Agent endpoints
  AGENTS: {
    LIST: '/agents',
    CREATE: '/agents',
    CHAT: '/agents/chat',
    TEST_CONFIG: '/agents/test-config',
    TEMPLATES: '/agents/templates',
    HISTORY: (agentId: string) => `/agents/${agentId}/history`,
    METRICS: (agentId: string) => `/agents/${agentId}/metrics`
  },
  
  // Enhanced orchestration endpoints
  ORCHESTRATION: {
    CREATE_AGENT: '/orchestration/agents',
    CREATE_TOOL: '/orchestration/tools',
    EXECUTE_TASK: '/orchestration/agents/execute',
    ASSIGN_TOOLS: '/orchestration/agents/assign-tools',
    LIST_AGENTS: '/orchestration/agents',
    LIST_TOOLS: '/orchestration/tools',
    AGENT_PERFORMANCE: (agentId: string) => `/orchestration/agents/${agentId}/performance`,
    SYSTEM_STATUS: '/orchestration/system/status',
    METRICS: '/orchestration/metrics'
  },
  
  // RAG endpoints
  RAG: {
    HEALTH: normalizeApiPath('/rag/health'),
    SEARCH: normalizeApiPath('/rag/search'),
    INGEST_DOCUMENT: normalizeApiPath('/rag/ingest/document'),
    INGEST_FILE: normalizeApiPath('/rag/ingest/file'),
    INGESTION_STATUS: (jobId: string) => normalizeApiPath(`/rag/ingest/status/${jobId}`),
    COLLECTIONS: normalizeApiPath('/rag/collections'),
    COLLECTION_DETAILS: (name: string) => normalizeApiPath(`/rag/collections/${name}`),
    EMBEDDINGS: {
      MODELS: normalizeApiPath('/rag/embeddings/models'),
      GENERATE: normalizeApiPath('/rag/embeddings/generate')
    },
    STATS: '/rag/stats'
  },
  
  // Workflow endpoints
  WORKFLOWS: {
    EXECUTE: '/workflows/execute',
    TEMPLATES: '/workflows/templates',
    HISTORY: '/workflows/history',
    DETAILS: (workflowId: string) => `/workflows/${workflowId}`,
    CREATE: '/workflows/create'
  },
  
  // LLM provider endpoints
  LLM: {
    PROVIDERS: '/llm/providers',
    MODELS: '/llm/models',
    MODELS_BY_PROVIDER: (provider: string) => `/llm/models/${provider}`,
    REGISTER_PROVIDER: '/llm/providers/register',
    TEST_MODEL: '/llm/test/model',
    TEST_PROVIDER: (provider: string) => `/llm/test/providers/${provider}`,
    TEST_ALL_PROVIDERS: '/llm/test/providers',
    TEST_CONFIG: '/llm/test/config',
    DEFAULT_CONFIG: '/llm/default-config'
  },
  
  // Settings
  SETTINGS: {
    GET: '/settings',
    UPDATE: '/settings',
    PREFERENCES: '/settings/preferences',
    EXPORT: '/settings/export',
    IMPORT: '/settings/import'
  }
} as const

// Request transformation utilities
export class BackendRequestTransformer {
  static transformAgentCreate(frontendData: any) {
    try {
      return transformAgentDataForBackend(frontendData)
    } catch (error) {
      if (isValidationError(error)) {
        throw new Error(`Invalid agent data: ${formatValidationError(error)}`)
      }
      throw error
    }
  }
  
  static transformAgentChat(frontendData: any) {
    try {
      return transformChatDataForBackend(frontendData)
    } catch (error) {
      if (isValidationError(error)) {
        throw new Error(`Invalid chat data: ${formatValidationError(error)}`)
      }
      throw error
    }
  }
  
  static transformRAGSearch(frontendData: any) {
    try {
      return transformRAGSearchForBackend(frontendData)
    } catch (error) {
      if (isValidationError(error)) {
        throw new Error(`Invalid search data: ${formatValidationError(error)}`)
      }
      throw error
    }
  }
  
  static transformTaskExecution(frontendData: any) {
    try {
      return transformTaskDataForBackend(frontendData)
    } catch (error) {
      if (isValidationError(error)) {
        throw new Error(`Invalid task data: ${formatValidationError(error)}`)
      }
      throw error
    }
  }
  
  static transformFileUpload(file: File, collection?: string) {
    const formData = new FormData()
    formData.append('file', file)
    if (collection) {
      formData.append('collection', collection)
    }
    return formData
  }
}

// Response validation utilities
export class BackendResponseValidator {
  static validateAgentResponse(data: unknown) {
    try {
      return validateAgentResponse(data)
    } catch (error) {
      if (isValidationError(error)) {
        console.warn(`Invalid agent response: ${formatValidationError(error)}`)
        return data // Return raw data if validation fails
      }
      throw error
    }
  }
  
  static validateChatResponse(data: unknown) {
    try {
      return validateChatResponse(data)
    } catch (error) {
      if (isValidationError(error)) {
        console.warn(`Invalid chat response: ${formatValidationError(error)}`)
        return data
      }
      throw error
    }
  }
  
  static validateRAGSearchResponse(data: unknown) {
    try {
      return validateRAGSearchResponse(data)
    } catch (error) {
      if (isValidationError(error)) {
        console.warn(`Invalid RAG search response: ${formatValidationError(error)}`)
        return data
      }
      throw error
    }
  }
}

// Error handling utilities
export class BackendErrorHandler {
  static handleAPIError(error: any): string {
    if (error.response) {
      // HTTP error response
      const status = error.response.status
      const data = error.response.data
      
      switch (status) {
        case 400:
          return `Bad Request: ${data.detail || data.message || 'Invalid request data'}`
        case 401:
          return 'Unauthorized: Please check your authentication'
        case 403:
          return 'Forbidden: You do not have permission to perform this action'
        case 404:
          return 'Not Found: The requested resource was not found'
        case 422:
          return `Validation Error: ${data.detail || 'Invalid input data'}`
        case 429:
          return 'Rate Limited: Too many requests, please try again later'
        case 500:
          return 'Server Error: An internal server error occurred'
        case 502:
          return 'Bad Gateway: The server is temporarily unavailable'
        case 503:
          return 'Service Unavailable: The service is temporarily down'
        default:
          return `HTTP ${status}: ${data.detail || data.message || 'An error occurred'}`
      }
    } else if (error.request) {
      // Network error
      return 'Network Error: Unable to connect to the server'
    } else {
      // Other error
      return error.message || 'An unexpected error occurred'
    }
  }
  
  static isRetryableError(error: any): boolean {
    if (error.response) {
      const status = error.response.status
      return status >= 500 || status === 429 || status === 408
    }
    return true // Network errors are retryable
  }
}

// Request retry utilities
export class RequestRetryHandler {
  static async withRetry<T>(
    requestFn: () => Promise<T>,
    maxRetries: number = 3,
    delayMs: number = 1000
  ): Promise<T> {
    let lastError: any
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await requestFn()
      } catch (error) {
        lastError = error
        
        if (attempt === maxRetries || !BackendErrorHandler.isRetryableError(error)) {
          break
        }
        
        // Exponential backoff
        const delay = delayMs * Math.pow(2, attempt)
        await new Promise(resolve => setTimeout(resolve, delay))
      }
    }
    
    throw lastError
  }
}

// Data consistency utilities
export class DataConsistencyChecker {
  static checkAgentDataConsistency(frontendAgent: any, backendAgent: any): boolean {
    const requiredFields = ['id', 'name', 'description', 'agent_type', 'model']
    
    for (const field of requiredFields) {
      if (!backendAgent[field]) {
        console.warn(`Missing required field in backend agent: ${field}`)
        return false
      }
    }
    
    return true
  }
  
  static checkRAGDataConsistency(searchRequest: any, searchResponse: any): boolean {
    if (searchRequest.query !== searchResponse.query) {
      console.warn('Query mismatch between request and response')
      return false
    }
    
    if (!Array.isArray(searchResponse.results)) {
      console.warn('Invalid results format in RAG response')
      return false
    }
    
    return true
  }
}

// Configuration alignment utilities
export class ConfigurationAligner {
  static alignEmbeddingConfig(frontendConfig: any) {
    return {
      current_model: frontendConfig.currentModel || frontendConfig.current_model,
      batch_size: frontendConfig.batchSize || frontendConfig.batch_size || 32,
      max_length: frontendConfig.maxLength || frontendConfig.max_length || 512,
      normalize: frontendConfig.normalize ?? true,
      cache_embeddings: frontendConfig.cacheEmbeddings ?? frontendConfig.cache_embeddings ?? true,
      device: frontendConfig.device || 'auto'
    }
  }
  
  static alignAgentConfig(frontendConfig: any) {
    return {
      model: frontendConfig.model || 'llama3.2:latest',
      temperature: frontendConfig.temperature ?? 0.7,
      max_tokens: frontendConfig.maxTokens || frontendConfig.max_tokens || 2048,
      top_p: frontendConfig.topP || frontendConfig.top_p,
      top_k: frontendConfig.topK || frontendConfig.top_k,
      frequency_penalty: frontendConfig.frequencyPenalty || frontendConfig.frequency_penalty,
      presence_penalty: frontendConfig.presencePenalty || frontendConfig.presence_penalty,
      ...frontendConfig
    }
  }
}
