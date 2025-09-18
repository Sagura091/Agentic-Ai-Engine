import axios from 'axios'
import { io, Socket } from 'socket.io-client'
import {
  BackendRequestTransformer,
  BackendResponseValidator,
  BackendErrorHandler,
  RequestRetryHandler,
  API_ENDPOINTS
} from '../utils/backendAlignment'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8888/api/v1'
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'http://localhost:8888'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available (optional in development)
    const token = localStorage.getItem('auth-token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }

    // Add development headers
    if (window.location.hostname === 'localhost') {
      config.headers['X-Development-Mode'] = 'true'
    }

    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor with enhanced error handling
api.interceptors.response.use(
  (response) => {
    // Return the full response object to preserve status and headers
    return response
  },
  (error) => {
    // Enhanced error handling using BackendErrorHandler
    const errorMessage = BackendErrorHandler.handleAPIError(error)

    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth-token')
      // Don't redirect in development
      if (window.location.hostname !== 'localhost') {
        window.location.href = '/login'
      }
    }

    // Log error for debugging
    console.error('API Error:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      statusText: error.response?.statusText,
      message: errorMessage,
      data: error.response?.data,
      fullError: error
    })

    return Promise.reject(new Error(errorMessage))
  }
)

// Agent API
export const agentApi = {
  // Get all active agents
  getAgents: async () => {
    const response = await api.get('/agents')
    return response.data
  },

  // Chat with an agent
  chatWithAgent: async (data: {
    message: string
    agent_id?: string
    agent_type?: string
    model?: string
    conversation_id?: string
    temperature?: number
    max_tokens?: number
    context?: any
  }) => {
    const transformedData = BackendRequestTransformer.transformAgentChat(data)
    const response = await RequestRetryHandler.withRetry(() =>
      api.post(API_ENDPOINTS.AGENTS.CHAT, transformedData)
    )
    return BackendResponseValidator.validateChatResponse(response.data)
  },

  // Test agent configuration before creation
  testAgentConfig: async (data: {
    name: string
    description: string
    model: string
    model_provider: string
    temperature: number
    max_tokens: number
    capabilities?: string[]
    tools?: string[]
    system_prompt?: string
  }) => {
    const transformedData = BackendRequestTransformer.transformAgentCreate(data)
    return RequestRetryHandler.withRetry(() =>
      api.post(API_ENDPOINTS.AGENTS.TEST_CONFIG, transformedData)
    )
  },

  // Create a custom agent
  createAgent: async (data: {
    name: string
    description: string
    agent_type?: string
    model?: string
    capabilities?: string[]
    tools?: string[]
    system_prompt?: string
    temperature?: number
    max_tokens?: number
  }) => {
    const transformedData = BackendRequestTransformer.transformAgentCreate(data)
    const response = await RequestRetryHandler.withRetry(() =>
      api.post(API_ENDPOINTS.AGENTS.CREATE, transformedData)
    )
    return BackendResponseValidator.validateAgentResponse(response.data)
  },

  // Get agent templates
  getTemplates: async () => {
    const response = await api.get('/agents/templates')
    return response.data
  },

  // Get agent execution history
  getAgentHistory: async (agentId: string, params?: {
    limit?: number
    offset?: number
  }) => {
    const response = await api.get(`/agents/${agentId}/history`, { params })
    return response.data
  },

  // Get agent performance metrics
  getAgentMetrics: (agentId: string, params?: {
    timeframe?: string
  }) => api.get(`/agents/${agentId}/metrics`, { params }),
}

// Workflow API
export const workflowApi = {
  // Execute a workflow
  executeWorkflow: (data: {
    task: string
    workflow_type?: string
    model?: string
    context?: any
    timeout?: number
  }) => api.post('/workflows/execute', data),

  // Get workflow templates
  getTemplates: () => api.get('/workflows/templates'),

  // Get workflow execution history
  getHistory: (params?: {
    limit?: number
    offset?: number
    status?: string
  }) => api.get('/workflows/history', { params }),

  // Get workflow details
  getWorkflow: (workflowId: string) => api.get(`/workflows/${workflowId}`),

  // Create custom workflow
  createWorkflow: (data: {
    name: string
    description: string
    workflow_type?: string
    agents?: string[]
    steps?: any[]
    configuration?: any
  }) => api.post('/workflows/create', data),
}

// Models API
export const modelsApi = {
  // Get available Ollama models
  getModels: () => api.get('/models'),

  // Get model details
  getModelDetails: (modelName: string) => api.get(`/models/${modelName}`),

  // Test model availability
  testModel: (modelName: string) => api.post(`/models/${modelName}/test`),
}

// Health API
export const healthApi = {
  // Get system health
  getHealth: () => api.get('/health'),

  // Get service status
  getStatus: () => api.get('/health/status'),

  // Get system metrics
  getMetrics: () => api.get('/health/metrics'),
}

// Monitoring API
export const monitoringApi = {
  // Get real-time agent activity
  getAgentActivity: (params?: {
    timeframe?: string
    agent_type?: string
  }) => api.get('/monitoring/agents', { params }),

  // Get workflow activity
  getWorkflowActivity: (params?: {
    timeframe?: string
    workflow_type?: string
  }) => api.get('/monitoring/workflows', { params }),

  // Get system performance metrics
  getSystemMetrics: (params?: {
    timeframe?: string
    metric_type?: string
  }) => api.get('/monitoring/system', { params }),

  // Get error logs
  getErrorLogs: (params?: {
    limit?: number
    severity?: string
    component?: string
  }) => api.get('/monitoring/errors', { params }),
}

// Enhanced Orchestration API (Revolutionary Agentic AI)
export const enhancedOrchestrationApi = {
  // Create unlimited agents
  createUnlimitedAgent: (data: {
    agent_type: string
    name: string
    description: string
    config?: any
    tools?: string[]
  }) => api.post('/orchestration/unlimited/agent', data),

  // Create unlimited tools
  createUnlimitedTool: (data: {
    name: string
    description: string
    functionality_description: string
    assign_to_agent?: string
    make_global?: boolean
  }) => api.post('/orchestration/unlimited/tool', data),

  // Get system status
  getSystemStatus: () => api.get('/orchestration/system/status'),

  // Get orchestration metrics
  getOrchestrationMetrics: () => api.get('/orchestration/metrics'),

  // Create agent with enhanced capabilities
  createAgent: (data: {
    agent_type: 'basic' | 'autonomous' | 'research' | 'creative' | 'optimization' | 'custom'
    name: string
    description: string
    model?: string
    tools?: string[]
    config?: any
  }) => api.post('/orchestration/agents', data),

  // Create dynamic tool
  createDynamicTool: (data: {
    name: string
    description: string
    functionality_description: string
    category: string
    assign_to_agent?: string
    make_global?: boolean
  }) => api.post('/orchestration/tools', data),

  // Assign tools to agent
  assignToolsToAgent: (data: {
    agent_id: string
    tool_names: string[]
  }) => api.post('/orchestration/agents/assign-tools', data),

  // Execute agent task
  executeAgentTask: async (data: {
    agent_id: string
    task: string
    context?: any
  }) => {
    const transformedData = BackendRequestTransformer.transformTaskExecution(data)
    return RequestRetryHandler.withRetry(() =>
      api.post(API_ENDPOINTS.ORCHESTRATION.EXECUTE_TASK, transformedData)
    )
  },

  // Get agent performance
  getAgentPerformance: (agent_id: string) => api.get(`/orchestration/agents/${agent_id}/performance`),

  // List all enhanced agents
  listEnhancedAgents: () => api.get('/orchestration/agents'),

  // List all dynamic tools
  listDynamicTools: () => api.get('/orchestration/tools'),
}

// Autonomous Agents API
export const autonomousAgentsApi = {
  // Create autonomous agent
  createAutonomousAgent: (data: {
    name: string
    description: string
    autonomy_level?: string
    learning_mode?: string
    config?: any
  }) => api.post('/autonomous/agents', data),

  // List autonomous agents
  listAutonomousAgents: () => api.get('/autonomous/agents'),

  // Get autonomous agent details
  getAutonomousAgent: (agent_id: string) => api.get(`/autonomous/agents/${agent_id}`),

  // Execute autonomous agent task
  executeAutonomousTask: (data: {
    agent_id: string
    task: string
    context?: any
  }) => api.post('/autonomous/agents/execute', data),

  // Get learning statistics
  getLearningStats: (agent_id: string) => api.get(`/autonomous/agents/${agent_id}/learning`),
}

// Settings API
export const settingsApi = {
  // Get application settings
  getSettings: () => api.get('/settings'),

  // Update settings
  updateSettings: (data: any) => api.put('/settings', data),

  // Get user preferences
  getPreferences: () => api.get('/settings/preferences'),

  // Update user preferences
  updatePreferences: (data: any) => api.put('/settings/preferences', data),

  // Export configuration
  exportConfig: () => api.get('/settings/export'),

  // Import configuration
  importConfig: (data: any) => api.post('/settings/import', data),
}

// RAG API for Knowledge Management
export const ragApi = {
  // Search knowledge base
  searchKnowledge: async (data: {
    query: string
    collection?: string
    top_k?: number
    filters?: any
  }) => {
    const transformedData = BackendRequestTransformer.transformRAGSearch(data)
    const response = await RequestRetryHandler.withRetry(() =>
      api.post(API_ENDPOINTS.RAG.SEARCH, transformedData)
    )
    return BackendResponseValidator.validateRAGSearchResponse(response)
  },

  // Ingest document
  ingestDocument: (data: {
    title: string
    content: string
    collection?: string
    metadata?: any
    document_type?: string
  }) => api.post('/rag/ingest/document', data),

  // Upload file for ingestion
  uploadFile: async (file: File, collection?: string) => {
    const formData = BackendRequestTransformer.transformFileUpload(file, collection)

    return RequestRetryHandler.withRetry(() =>
      api.post(API_ENDPOINTS.RAG.INGEST_FILE, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
    )
  },

  // Get ingestion status
  getIngestionStatus: (jobId: string) => api.get(`/rag/ingest/status/${jobId}`),

  // Create collection
  createCollection: (data: { name: string }) => api.post('/rag/collections', data),

  // List collections
  listCollections: async () => {
    const response = await api.get('/rag/collections')
    return response.data
  },

  // Get collection details
  getCollection: (collectionName: string) => api.get(`/rag/collections/${collectionName}`),

  // Delete collection
  deleteCollection: (collectionName: string) => api.delete(`/rag/collections/${collectionName}`),

  // Get RAG statistics
  getStats: async () => {
    const response = await api.get('/rag/stats')
    return response.data
  },

  // Get system health
  getHealth: async () => {
    const response = await api.get('/rag/health')
    return response.data
  },
}

// Embedding Models API (Combined Global Configuration + Model Management)
export const embeddingApi = {
  // Global Configuration Methods
  // Get current embedding configuration
  getEmbeddingConfig: async () => {
    const response = await api.get('/rag/embeddings/config')
    return response.data
  },

  // Update embedding configuration
  updateEmbeddingConfig: async (config: {
    embedding_engine: string
    embedding_model: string
    embedding_batch_size: number
    openai_config?: {
      url: string
      key: string
    }
    ollama_config?: {
      url: string
      key: string
    }
    azure_openai_config?: {
      url: string
      key: string
      version: string
    }
  }) => {
    const response = await api.post('/rag/embeddings/config', config)
    return response.data
  },

  // Test embedding connection
  testEmbeddingConnection: async (config: {
    embedding_engine: string
    embedding_model: string
    openai_config?: {
      url: string
      key: string
    }
    ollama_config?: {
      url: string
      key: string
    }
    azure_openai_config?: {
      url: string
      key: string
      version: string
    }
  }) => {
    const response = await api.post('/rag/embeddings/test', config)
    return response.data
  },

  // Model Management Methods
  // List available embedding models
  listModels: async () => {
    const response = await api.get('/rag/embeddings/models')
    return response.data
  },

  // Get available embedding models (alias for compatibility)
  getAvailableModels: async () => {
    const response = await api.get('/rag/embeddings/models')
    return response.data.models || []
  },

  // Download embedding model
  downloadModel: (modelName: string) => api.post('/rag/embeddings/download', { model_name: modelName }),

  // Get model status
  getModelStatus: (modelName: string) => api.get(`/rag/embeddings/models/${modelName}/status`),

  // Test embedding model
  testModel: (modelName: string, text: string) => api.post('/rag/embeddings/test', {
    model_name: modelName,
    text
  }),

  // Legacy methods for compatibility
  // Get current embedding configuration (legacy)
  getConfig: () => api.get('/rag/embeddings/config'),

  // Update embedding configuration (legacy)
  updateConfig: (config: any) => api.put('/rag/embeddings/config', config),
}



// Knowledge Base Management API
export const knowledgeBaseApi = {
  // Create knowledge base
  createKnowledgeBase: async (data: {
    name: string
    description?: string
    use_case: string
    tags?: string[]
    is_public?: boolean
  }) => {
    const response = await api.post('/rag/knowledge-bases', data)
    return response.data
  },

  // List knowledge bases
  listKnowledgeBases: async () => {
    const response = await api.get('/rag/knowledge-bases')
    // Handle different response structures
    if (response.data && Array.isArray(response.data.knowledge_bases)) {
      return response.data.knowledge_bases
    } else if (response.data && Array.isArray(response.data)) {
      return response.data
    } else {
      return []
    }
  },

  // Upload document to knowledge base
  uploadDocument: async (kbId: string, file: File, title?: string, metadata?: any) => {
    const formData = new FormData()
    formData.append('file', file)
    if (title) formData.append('title', title)
    if (metadata) formData.append('metadata', JSON.stringify(metadata))

    const response = await api.post(`/rag/knowledge-bases/${kbId}/documents`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Search knowledge base
  searchKnowledgeBase: async (kbId: string, query: string, topK: number = 5) => {
    const formData = new FormData()
    formData.append('query', query)
    formData.append('top_k', topK.toString())

    const response = await api.post(`/rag/knowledge-bases/${kbId}/search`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Delete knowledge base
  deleteKnowledgeBase: async (kbId: string) => {
    const response = await api.delete(`/rag/knowledge-bases/${kbId}`)
    return response.data
  },

  // Get single knowledge base
  getKnowledgeBase: async (kbId: string) => {
    const response = await api.get(`/rag/knowledge-bases/${kbId}`)
    return response.data
  },

  // Get documents in knowledge base
  getDocuments: async (kbId: string) => {
    const response = await api.get(`/rag/knowledge-bases/${kbId}/documents`)
    return response.data
  },

  // Get document chunks (for viewing embeddings)
  getDocumentChunks: async (kbId: string, docId: string) => {
    const response = await api.get(`/rag/knowledge-bases/${kbId}/documents/${docId}/chunks`)
    return response.data
  },

  // Delete document
  deleteDocument: async (kbId: string, docId: string) => {
    const response = await api.delete(`/rag/knowledge-bases/${kbId}/documents/${docId}`)
    return response.data
  },

  // Update knowledge base
  updateKnowledgeBase: async (kbId: string, data: {
    description?: string
    tags?: string[]
    is_public?: boolean
  }) => {
    const response = await api.put(`/rag/knowledge-bases/${kbId}`, data)
    return response.data
  },

  // Get use cases - removed (unlimited knowledge base creation!)

  // Search within knowledge base
  searchKnowledgeBase: async (kbId: string, query: string, limit?: number) => {
    const response = await api.post('/rag/search', {
      query,
      collection: kbId,
      limit: limit || 10
    })
    return response.data
  },
}

// File upload utility (legacy - use ragApi.uploadFile for RAG)
export const uploadFile = async (file: File, endpoint: string) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post(endpoint, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
}

// WebSocket connection helper
export const getWebSocketUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = import.meta.env.VITE_WS_URL || `${protocol}//${window.location.host}`
  return host
}

// WebSocket API for real-time features
export class WebSocketAPI {
  private socket: Socket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private eventListeners: Map<string, Set<(data: any) => void>> = new Map()

  connect(): Promise<Socket> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve(this.socket)
        return
      }

      this.socket = io(WS_BASE_URL, {
        transports: ['websocket', 'polling'],
        timeout: 20000,
        forceNew: true
      })

      this.socket.on('connect', () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
        resolve(this.socket!)
      })

      this.socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason)
        if (reason === 'io server disconnect') {
          this.handleReconnect()
        }
      })

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error)
        this.handleReconnect()
        reject(error)
      })

      // Setup event forwarding
      this.setupEventForwarding()
    })
  }

  private setupEventForwarding() {
    if (!this.socket) return

    // Workflow execution events
    this.socket.on('workflow_started', (data) => this.forwardEvent('workflow_started', data))
    this.socket.on('workflow_progress', (data) => this.forwardEvent('workflow_progress', data))
    this.socket.on('workflow_completed', (data) => this.forwardEvent('workflow_completed', data))
    this.socket.on('workflow_error', (data) => this.forwardEvent('workflow_error', data))

    // Node execution events
    this.socket.on('node_started', (data) => this.forwardEvent('node_started', data))
    this.socket.on('node_completed', (data) => this.forwardEvent('node_completed', data))
    this.socket.on('node_error', (data) => this.forwardEvent('node_error', data))

    // Collaboration events
    this.socket.on('canvas_updated', (data) => this.forwardEvent('canvas_updated', data))
    this.socket.on('user_joined', (data) => this.forwardEvent('user_joined', data))
    this.socket.on('user_left', (data) => this.forwardEvent('user_left', data))
  }

  private forwardEvent(event: string, data: any) {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach(callback => callback(data))
    }
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
        this.connect()
      }, this.reconnectDelay * this.reconnectAttempts)
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    this.eventListeners.clear()
  }

  emit(event: string, data?: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data)
    }
  }

  on(event: string, callback: (data: any) => void) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set())
    }
    this.eventListeners.get(event)!.add(callback)
  }

  off(event: string, callback?: (data: any) => void) {
    if (callback) {
      const listeners = this.eventListeners.get(event)
      if (listeners) {
        listeners.delete(callback)
      }
    } else {
      this.eventListeners.delete(event)
    }
  }

  // Workflow-specific methods
  joinWorkflowRoom(workflowId: string) {
    this.emit('join_workflow', { workflowId })
  }

  leaveWorkflowRoom(workflowId: string) {
    this.emit('leave_workflow', { workflowId })
  }

  updateCanvas(workflowId: string, canvasData: any) {
    this.emit('update_canvas', { workflowId, canvasData })
  }

  executeWorkflow(workflowId: string, workflowData: any) {
    this.emit('execute_workflow', { workflowId, workflowData })
  }

  stopWorkflow(workflowId: string) {
    this.emit('stop_workflow', { workflowId })
  }
}

// LLM Providers API
export const llmProvidersApi = {
  getProviders: async (): Promise<any> => {
    const response = await api.get('/llm/providers')
    return response // axios interceptor already returns response.data
  },

  getAllModels: async (): Promise<any> => {
    const response = await api.get('/llm/models')
    return response // axios interceptor already returns response.data
  },

  getModelsByProvider: async (provider: string): Promise<any> => {
    const response = await api.get(`/llm/models/${provider}`)
    return response // axios interceptor already returns response.data
  },

  registerProvider: async (credentials: {
    provider: string
    api_key?: string
    base_url?: string
    organization?: string
    project?: string
    additional_headers?: Record<string, string>
  }): Promise<any> => {
    const response = await api.post('/llm/providers/register', credentials)
    return response.data
  },

  testModel: async (provider: string, model_id: string): Promise<any> => {
    const response = await api.post('/llm/test/model', { provider, model_id })
    return response.data
  },

  testProvider: async (provider: string): Promise<any> => {
    const response = await api.get(`/llm/test/providers/${provider}`)
    return response.data
  },

  testAllProviders: async (): Promise<any> => {
    const response = await api.get('/llm/test/providers')
    return response.data
  },

  testConfig: async (config: {
    provider: string
    model_id: string
    model_name?: string
    temperature: number
    max_tokens: number
    top_p?: number
    top_k?: number
    frequency_penalty?: number
    presence_penalty?: number
    additional_params?: Record<string, any>
  }): Promise<any> => {
    const response = await api.post('/llm/test/config', config)
    return response.data
  },

  getDefaultConfig: async (): Promise<any> => {
    const response = await api.get('/llm/default-config')
    return response.data
  }
}

// Singleton instance
export const wsApi = new WebSocketAPI()

export default api
