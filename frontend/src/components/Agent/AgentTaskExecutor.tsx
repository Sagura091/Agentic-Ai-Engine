import React, { useState, useEffect } from 'react'
import { useMutation, useQuery } from 'react-query'
import { 
  Play, 
  Pause, 
  Square, 
  Upload, 
  FileText, 
  Brain, 
  Zap,
  CheckCircle,
  AlertCircle,
  Loader2,
  Clock,
  Database,
  Search,
  Settings
} from 'lucide-react'
import { agentApi, ragApi, enhancedOrchestrationApi } from '../../services/api'
import { useAgent } from '../../contexts/AgentContext'
import toast from 'react-hot-toast'

interface TaskExecutionRequest {
  agent_id: string
  task_type: 'document_analysis' | 'knowledge_search' | 'data_extraction' | 'content_generation' | 'custom'
  task_description: string
  input_data?: {
    files?: File[]
    text_content?: string
    search_query?: string
    collection?: string
    parameters?: Record<string, any>
  }
  execution_config?: {
    use_rag: boolean
    rag_collection?: string
    max_tokens?: number
    temperature?: number
    timeout_seconds?: number
  }
}

interface TaskExecutionResult {
  task_id: string
  agent_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  result?: any
  error_message?: string
  execution_time_ms?: number
  tokens_used?: number
  rag_sources?: Array<{
    content: string
    score: number
    metadata: Record<string, any>
  }>
  created_at: string
  completed_at?: string
}

const AgentTaskExecutor: React.FC = () => {
  const { agents, selectedAgent } = useAgent()
  const [taskRequest, setTaskRequest] = useState<TaskExecutionRequest>({
    agent_id: '',
    task_type: 'document_analysis',
    task_description: '',
    input_data: {},
    execution_config: {
      use_rag: true,
      max_tokens: 2048,
      temperature: 0.7,
      timeout_seconds: 300
    }
  })
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [executionResults, setExecutionResults] = useState<TaskExecutionResult[]>([])
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false)

  // Fetch available collections for RAG
  const { data: collections = [] } = useQuery('rag-collections', ragApi.listCollections)

  // Execute task mutation
  const executeTaskMutation = useMutation(
    async (request: TaskExecutionRequest) => {
      // First upload files if any
      if (selectedFiles.length > 0) {
        const uploadPromises = selectedFiles.map(file => 
          ragApi.uploadFile(file, request.execution_config?.rag_collection)
        )
        await Promise.all(uploadPromises)
        toast.success(`Uploaded ${selectedFiles.length} files to knowledge base`)
      }

      // Execute the agent task
      return enhancedOrchestrationApi.executeAgentTask({
        agent_id: request.agent_id,
        task: request.task_description,
        context: {
          task_type: request.task_type,
          input_data: request.input_data,
          execution_config: request.execution_config
        }
      })
    },
    {
      onSuccess: (result) => {
        setExecutionResults(prev => [result, ...prev])
        toast.success('Task executed successfully')
        // Reset form
        setTaskRequest(prev => ({ ...prev, task_description: '' }))
        setSelectedFiles([])
      },
      onError: (error: any) => {
        toast.error(`Task execution failed: ${error.message}`)
      }
    }
  )

  // Update agent_id when selectedAgent changes
  useEffect(() => {
    if (selectedAgent) {
      setTaskRequest(prev => ({ ...prev, agent_id: selectedAgent.id }))
    }
  }, [selectedAgent])

  const handleExecuteTask = () => {
    if (!taskRequest.agent_id) {
      toast.error('Please select an agent')
      return
    }
    if (!taskRequest.task_description.trim()) {
      toast.error('Please provide a task description')
      return
    }

    const requestWithFiles = {
      ...taskRequest,
      input_data: {
        ...taskRequest.input_data,
        files: selectedFiles
      }
    }

    executeTaskMutation.mutate(requestWithFiles)
  }

  const handleFileUpload = (files: FileList | null) => {
    if (files) {
      setSelectedFiles(Array.from(files))
    }
  }

  const getTaskTypeIcon = (taskType: string) => {
    switch (taskType) {
      case 'document_analysis':
        return <FileText className="h-4 w-4" />
      case 'knowledge_search':
        return <Search className="h-4 w-4" />
      case 'data_extraction':
        return <Database className="h-4 w-4" />
      case 'content_generation':
        return <Brain className="h-4 w-4" />
      default:
        return <Zap className="h-4 w-4" />
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <Zap className="h-6 w-6 text-blue-500" />
            <span>Agent Task Executor</span>
          </h2>
          <p className="text-muted-foreground mt-1">
            Execute tasks with agents using RAG-enhanced knowledge retrieval
          </p>
        </div>
        
        <button
          onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
          className="btn-secondary flex items-center space-x-2"
        >
          <Settings className="h-4 w-4" />
          <span>Advanced Config</span>
        </button>
      </div>

      {/* Task Configuration */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4">Task Configuration</h3>
        
        <div className="space-y-4">
          {/* Agent Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Select Agent</label>
            <select
              value={taskRequest.agent_id}
              onChange={(e) => setTaskRequest(prev => ({ ...prev, agent_id: e.target.value }))}
              className="input"
            >
              <option value="">Choose an agent...</option>
              {agents.map(agent => (
                <option key={agent.id} value={agent.id}>
                  {agent.name} ({agent.agent_type})
                </option>
              ))}
            </select>
          </div>

          {/* Task Type */}
          <div>
            <label className="block text-sm font-medium mb-2">Task Type</label>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {[
                { value: 'document_analysis', label: 'Document Analysis' },
                { value: 'knowledge_search', label: 'Knowledge Search' },
                { value: 'data_extraction', label: 'Data Extraction' },
                { value: 'content_generation', label: 'Content Generation' },
                { value: 'custom', label: 'Custom Task' }
              ].map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => setTaskRequest(prev => ({ ...prev, task_type: value as any }))}
                  className={`p-3 rounded-lg border text-sm font-medium transition-colors flex items-center space-x-2 ${
                    taskRequest.task_type === value
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border hover:border-primary/50'
                  }`}
                >
                  {getTaskTypeIcon(value)}
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Task Description */}
          <div>
            <label className="block text-sm font-medium mb-2">Task Description</label>
            <textarea
              value={taskRequest.task_description}
              onChange={(e) => setTaskRequest(prev => ({ ...prev, task_description: e.target.value }))}
              placeholder="Describe what you want the agent to do..."
              className="input min-h-[100px] resize-none"
            />
          </div>

          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium mb-2">Upload Files (Optional)</label>
            <div className="border-2 border-dashed border-border rounded-lg p-4">
              <input
                type="file"
                multiple
                onChange={(e) => handleFileUpload(e.target.files)}
                className="hidden"
                id="file-upload"
                accept=".pdf,.docx,.txt,.md,.json,.csv"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center space-y-2"
              >
                <Upload className="h-8 w-8 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">
                  Click to upload files or drag and drop
                </span>
                <span className="text-xs text-muted-foreground">
                  PDF, DOCX, TXT, MD, JSON, CSV
                </span>
              </label>
              
              {selectedFiles.length > 0 && (
                <div className="mt-3 space-y-1">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="flex items-center space-x-2 text-sm">
                      <FileText className="h-4 w-4" />
                      <span>{file.name}</span>
                      <span className="text-muted-foreground">({(file.size / 1024).toFixed(1)} KB)</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* RAG Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="use-rag"
                checked={taskRequest.execution_config?.use_rag || false}
                onChange={(e) => setTaskRequest(prev => ({
                  ...prev,
                  execution_config: { ...prev.execution_config!, use_rag: e.target.checked }
                }))}
                className="rounded"
              />
              <label htmlFor="use-rag" className="text-sm font-medium">
                Use RAG Knowledge Base
              </label>
            </div>

            {taskRequest.execution_config?.use_rag && (
              <div>
                <select
                  value={taskRequest.execution_config?.rag_collection || ''}
                  onChange={(e) => setTaskRequest(prev => ({
                    ...prev,
                    execution_config: { ...prev.execution_config!, rag_collection: e.target.value }
                  }))}
                  className="input"
                >
                  <option value="">All Collections</option>
                  {collections.map((collection: any) => (
                    <option key={collection.name} value={collection.name}>
                      {collection.name} ({collection.document_count} docs)
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Advanced Configuration */}
          {showAdvancedConfig && (
            <div className="border-t pt-4 space-y-4">
              <h4 className="font-medium">Advanced Configuration</h4>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Max Tokens</label>
                  <input
                    type="number"
                    value={taskRequest.execution_config?.max_tokens || 2048}
                    onChange={(e) => setTaskRequest(prev => ({
                      ...prev,
                      execution_config: { ...prev.execution_config!, max_tokens: parseInt(e.target.value) }
                    }))}
                    className="input"
                    min="256"
                    max="8192"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Temperature</label>
                  <input
                    type="number"
                    value={taskRequest.execution_config?.temperature || 0.7}
                    onChange={(e) => setTaskRequest(prev => ({
                      ...prev,
                      execution_config: { ...prev.execution_config!, temperature: parseFloat(e.target.value) }
                    }))}
                    className="input"
                    min="0"
                    max="2"
                    step="0.1"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Timeout (seconds)</label>
                  <input
                    type="number"
                    value={taskRequest.execution_config?.timeout_seconds || 300}
                    onChange={(e) => setTaskRequest(prev => ({
                      ...prev,
                      execution_config: { ...prev.execution_config!, timeout_seconds: parseInt(e.target.value) }
                    }))}
                    className="input"
                    min="30"
                    max="1800"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Execute Button */}
          <div className="flex justify-end">
            <button
              onClick={handleExecuteTask}
              disabled={executeTaskMutation.isLoading || !taskRequest.agent_id || !taskRequest.task_description.trim()}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {executeTaskMutation.isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span>Execute Task</span>
            </button>
          </div>
        </div>
      </div>

      {/* Execution Results */}
      {executionResults.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Execution Results</h3>
          
          {executionResults.map((result, index) => (
            <div key={result.task_id || index} className="card p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(result.status)}
                  <span className="font-medium">Task {result.task_id}</span>
                  <span className="text-sm text-muted-foreground">
                    Agent: {agents.find(a => a.id === result.agent_id)?.name || result.agent_id}
                  </span>
                </div>
                
                <div className="text-sm text-muted-foreground">
                  {result.execution_time_ms && `${result.execution_time_ms}ms`}
                  {result.tokens_used && ` • ${result.tokens_used} tokens`}
                </div>
              </div>
              
              {result.status === 'completed' && result.result && (
                <div className="space-y-3">
                  <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
                    <h5 className="font-medium text-green-800 dark:text-green-200 mb-2">Result:</h5>
                    <p className="text-sm text-green-700 dark:text-green-300 whitespace-pre-wrap">
                      {typeof result.result === 'string' ? result.result : JSON.stringify(result.result, null, 2)}
                    </p>
                  </div>
                  
                  {result.rag_sources && result.rag_sources.length > 0 && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                      <h5 className="font-medium text-blue-800 dark:text-blue-200 mb-2">
                        RAG Sources ({result.rag_sources.length}):
                      </h5>
                      <div className="space-y-2">
                        {result.rag_sources.slice(0, 3).map((source, idx) => (
                          <div key={idx} className="text-sm">
                            <div className="flex justify-between items-center mb-1">
                              <span className="font-medium">Source {idx + 1}</span>
                              <span className="text-blue-600 dark:text-blue-400">
                                Score: {(source.score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <p className="text-blue-700 dark:text-blue-300">
                              {source.content.substring(0, 150)}...
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {result.status === 'failed' && result.error_message && (
                <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg">
                  <h5 className="font-medium text-red-800 dark:text-red-200 mb-2">Error:</h5>
                  <p className="text-sm text-red-700 dark:text-red-300">{result.error_message}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default AgentTaskExecutor
