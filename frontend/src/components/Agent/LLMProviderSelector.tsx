import React, { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { 
  Bot, 
  Key, 
  Settings, 
  AlertCircle, 
  CheckCircle, 
  Loader2,
  Eye,
  EyeOff
} from 'lucide-react'
import { llmProvidersApi } from '../../services/api'
import toast from 'react-hot-toast'

interface LLMConfig {
  provider: string
  model_id: string
  model_name?: string
  temperature: number
  max_tokens: number
  top_p?: number
  top_k?: number
  frequency_penalty?: number
  presence_penalty?: number
  api_key?: string
  base_url?: string
  organization?: string
  project?: string
}

interface LLMProviderSelectorProps {
  value: LLMConfig
  onChange: (config: LLMConfig) => void
  className?: string
}

interface ProviderModel {
  id: string
  name: string
  provider: string
  description?: string
  capabilities: string[]
  max_tokens?: number
  context_length?: number
  status: string
}

const LLMProviderSelector: React.FC<LLMProviderSelectorProps> = ({
  value,
  onChange,
  className = ''
}) => {
  const [showApiKey, setShowApiKey] = useState(false)
  const [testingConnection, setTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')

  // Fetch available providers
  const { data: providersData, isLoading: providersLoading } = useQuery(
    'llm-providers',
    () => llmProvidersApi.getProviders(),
    {
      staleTime: 5 * 60 * 1000, // 5 minutes
      onError: (error: any) => {
        console.error('Failed to fetch providers:', error)
        toast.error('Failed to load LLM providers')
      }
    }
  )

  // Fetch models for selected provider
  const { data: modelsData, isLoading: modelsLoading, error: modelsError, refetch: refetchModels } = useQuery(
    ['llm-models', value.provider],
    async () => {
      console.log(`🚀 Query function called for provider: ${value.provider}`)

      if (!value.provider) {
        console.log('❌ No provider selected, skipping model fetch')
        return { models: [] }
      }

      console.log(`🔍 Fetching models for provider: ${value.provider}`)
      try {
        const result = await llmProvidersApi.getModelsByProvider(value.provider)
        console.log(`✅ Raw result from API:`, result)
        console.log(`✅ Result type:`, typeof result)
        console.log(`✅ Result keys:`, result ? Object.keys(result) : 'no keys')
        console.log(`✅ Models in result:`, result?.models)
        console.log(`📊 Number of models: ${result?.models?.length || 0}`)
        return result
      } catch (error) {
        console.error(`❌ Error fetching models for ${value.provider}:`, error)
        throw error
      }
    },
    {
      enabled: !!value.provider,
      staleTime: 0, // No caching for debugging
      cacheTime: 0, // No caching for debugging
      retry: 1,
      refetchOnMount: true,
      refetchOnWindowFocus: false,
      onSuccess: (data) => {
        console.log(`🎉 Query success for ${value.provider}:`, data)
      },
      onError: (error: any) => {
        console.error(`❌ Query error for ${value.provider}:`, error)
        toast.error(`Failed to load models for ${value.provider}: ${error.message}`)
      }
    }
  )

  const providers = providersData?.providers || []
  const models: ProviderModel[] = modelsData?.models || []

  // Remove direct Ollama fetching - use backend API only

  // Force refetch when provider changes
  React.useEffect(() => {
    if (value.provider) {
      console.log(`🔄 Provider changed to: ${value.provider}, triggering refetch`)
      refetchModels()
    }
  }, [value.provider, refetchModels])

  // Debug logging
  React.useEffect(() => {
    console.log(`🔍 Provider: ${value.provider}`)
    console.log(`📊 Models data:`, modelsData)
    console.log(`📊 Models array:`, models)
    console.log(`📊 Models count:`, models.length)
    console.log(`⏳ Loading:`, modelsLoading)
    console.log(`❌ Error:`, modelsError)

    if (models.length > 0) {
      console.log('📋 Available models:', models.map(m => m.name || m.id))
    } else if (value.provider) {
      console.log(`⚠️ No models found for provider: ${value.provider}`)
    }

    if (modelsError) {
      console.error('❌ Models error details:', modelsError)
    }
  }, [value.provider, models, modelsError, modelsData, modelsLoading])

  // Test connection when provider or API key changes
  const testConnection = async () => {
    if (!value.provider) return

    setTestingConnection(true)
    setConnectionStatus('idle')

    try {
      // Register credentials if API key is provided
      if (value.api_key && value.provider !== 'ollama') {
        await llmProvidersApi.registerProvider({
          provider: value.provider,
          api_key: value.api_key,
          base_url: value.base_url,
          organization: value.organization,
          project: value.project
        })
      }

      // Test the connection
      const result = await llmProvidersApi.testProvider(value.provider)
      
      if (result.test_result?.is_available) {
        setConnectionStatus('success')
        toast.success(`Connected to ${value.provider} successfully`)
      } else {
        setConnectionStatus('error')
        toast.error(result.test_result?.error_message || `Failed to connect to ${value.provider}`)
      }
    } catch (error: any) {
      setConnectionStatus('error')
      toast.error(error.message || `Failed to test ${value.provider} connection`)
    } finally {
      setTestingConnection(false)
    }
  }

  const handleProviderChange = (provider: string) => {
    console.log(`🔄 Changing provider to: ${provider}`)

    // Set default base_url for Ollama
    const defaultBaseUrl = provider === 'ollama' ? 'http://localhost:11434' : ''

    onChange({
      ...value,
      provider,
      model_id: '', // Reset model when provider changes
      api_key: '',
      base_url: defaultBaseUrl,
      organization: '',
      project: ''
    })
    setConnectionStatus('idle')
  }

  const handleModelChange = (model_id: string) => {
    const selectedModel = models.find(m => m.id === model_id)

    console.log(`🔄 Selected model: ${model_id}`, selectedModel)

    onChange({
      ...value,
      model_id,
      model_name: selectedModel?.name || model_id,
      max_tokens: selectedModel?.max_tokens || value.max_tokens
    })
  }

  const requiresApiKey = (provider: string) => {
    return provider !== 'ollama'
  }

  const getProviderIcon = (provider: string) => {
    switch (provider) {
      case 'ollama': return '🦙'
      case 'openai': return '🤖'
      case 'anthropic': return '🧠'
      case 'google': return '🌟'
      default: return '⚡'
    }
  }

  const getConnectionStatusIcon = () => {
    switch (connectionStatus) {
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error': return <AlertCircle className="h-4 w-4 text-red-500" />
      default: return null
    }
  }

  if (providersLoading) {
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm text-muted-foreground">Loading LLM providers...</span>
        </div>
      </div>
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Provider Selection Dropdown */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-foreground">
          LLM Provider *
        </label>
        <select
          value={value.provider}
          onChange={(e) => handleProviderChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="">Select a provider</option>
          <option value="ollama">🦙 Ollama (Local)</option>
          <option value="openai">🤖 OpenAI</option>
          <option value="anthropic">🧠 Anthropic</option>
          <option value="google">🌟 Google</option>
        </select>
      </div>

      {/* Provider-Specific Configuration */}
      {value.provider === 'ollama' && (
        <div className="space-y-3">
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className="text-blue-600">🦙</div>
              <div className="flex-1">
                <h4 className="text-sm font-medium text-blue-800">Ollama (Local)</h4>
                <p className="text-xs text-blue-600">Using backend connection to localhost:11434</p>
                {models.length > 0 && (
                  <p className="text-xs text-green-600 mt-1">✅ {models.length} models available</p>
                )}
              </div>
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={async () => {
                    console.log('🔄 Manual refetch triggered for Ollama models')
                    refetchModels()
                  }}
                  disabled={modelsLoading}
                  className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {modelsLoading ? 'Loading...' : 'Refresh Models'}
                </button>
                <button
                  type="button"
                  onClick={async () => {
                    console.log('🧪 Direct API test for Ollama models')
                    try {
                      const response = await fetch('http://localhost:8888/api/v1/llm/models/ollama')
                      const data = await response.json()
                      console.log('🧪 Direct API response:', data)
                      console.log('🧪 Models in response:', data.models?.length || 0)
                    } catch (error) {
                      console.error('🧪 Direct API test failed:', error)
                    }
                  }}
                  className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Test API
                </button>
                <button
                  type="button"
                  onClick={() => {
                    console.log('🔍 Current state debug:')
                    console.log('  Provider:', value.provider)
                    console.log('  ModelsData:', modelsData)
                    console.log('  Models array:', models)
                    console.log('  Loading:', modelsLoading)
                    console.log('  Error:', modelsError)
                  }}
                  className="px-3 py-1 text-xs bg-purple-600 text-white rounded hover:bg-purple-700"
                >
                  Debug
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {value.provider === 'openai' && (
        <div className="space-y-3">
          <label className="block text-sm font-medium text-foreground">
            OpenAI API Key *
          </label>
          <div className="relative">
            <input
              type={showApiKey ? 'text' : 'password'}
              value={value.api_key || ''}
              onChange={(e) => onChange({ ...value, api_key: e.target.value })}
              placeholder="sk-..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-20"
            />
            <div className="absolute inset-y-0 right-0 flex items-center space-x-1 pr-2">
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="p-1 text-gray-400 hover:text-gray-600"
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
              <button
                type="button"
                onClick={testConnection}
                disabled={!value.api_key || testingConnection}
                className="px-2 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                {testingConnection ? 'Testing...' : 'Connect'}
              </button>
            </div>
          </div>
          {connectionStatus === 'success' && (
            <div className="text-xs text-green-600 flex items-center space-x-1">
              <CheckCircle className="h-3 w-3" />
              <span>Connected successfully</span>
            </div>
          )}
          {connectionStatus === 'error' && (
            <div className="text-xs text-red-600 flex items-center space-x-1">
              <AlertCircle className="h-3 w-3" />
              <span>Connection failed</span>
            </div>
          )}
        </div>
      )}

      {value.provider === 'anthropic' && (
        <div className="space-y-3">
          <label className="block text-sm font-medium text-foreground">
            Anthropic API Key *
          </label>
          <div className="relative">
            <input
              type={showApiKey ? 'text' : 'password'}
              value={value.api_key || ''}
              onChange={(e) => onChange({ ...value, api_key: e.target.value })}
              placeholder="sk-ant-..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-20"
            />
            <div className="absolute inset-y-0 right-0 flex items-center space-x-1 pr-2">
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="p-1 text-gray-400 hover:text-gray-600"
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
              <button
                type="button"
                onClick={testConnection}
                disabled={!value.api_key || testingConnection}
                className="px-2 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                {testingConnection ? 'Testing...' : 'Connect'}
              </button>
            </div>
          </div>
          {connectionStatus === 'success' && (
            <div className="text-xs text-green-600 flex items-center space-x-1">
              <CheckCircle className="h-3 w-3" />
              <span>Connected successfully</span>
            </div>
          )}
          {connectionStatus === 'error' && (
            <div className="text-xs text-red-600 flex items-center space-x-1">
              <AlertCircle className="h-3 w-3" />
              <span>Connection failed</span>
            </div>
          )}
        </div>
      )}

      {value.provider === 'google' && (
        <div className="space-y-3">
          <label className="block text-sm font-medium text-foreground">
            Google API Key *
          </label>
          <div className="relative">
            <input
              type={showApiKey ? 'text' : 'password'}
              value={value.api_key || ''}
              onChange={(e) => onChange({ ...value, api_key: e.target.value })}
              placeholder="AIza..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-20"
            />
            <div className="absolute inset-y-0 right-0 flex items-center space-x-1 pr-2">
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="p-1 text-gray-400 hover:text-gray-600"
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
              <button
                type="button"
                onClick={testConnection}
                disabled={!value.api_key || testingConnection}
                className="px-2 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                {testingConnection ? 'Testing...' : 'Connect'}
              </button>
            </div>
          </div>
          {connectionStatus === 'success' && (
            <div className="text-xs text-green-600 flex items-center space-x-1">
              <CheckCircle className="h-3 w-3" />
              <span>Connected successfully</span>
            </div>
          )}
          {connectionStatus === 'error' && (
            <div className="text-xs text-red-600 flex items-center space-x-1">
              <AlertCircle className="h-3 w-3" />
              <span>Connection failed</span>
            </div>
          )}
        </div>
      )}

      {/* Model Selection */}
      {value.provider && (
        <div className="space-y-3">
          <label className="block text-sm font-medium text-foreground">
            Model *
          </label>

          {/* Show models for Ollama immediately or for API providers after successful connection */}
          {(value.provider === 'ollama' || connectionStatus === 'success') ? (
            <>
              {modelsLoading ? (
                <div className="flex items-center space-x-2 p-3 border border-gray-300 rounded-lg">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm text-gray-600">
                    Loading {value.provider} models...
                  </span>
                </div>
              ) : (
                <select
                  value={value.model_id}
                  onChange={(e) => handleModelChange(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Select a model</option>
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name} {model.max_tokens && `(${model.max_tokens} tokens)`}
                    </option>
                  ))}
                </select>
              )}
            </>
          ) : value.provider !== 'ollama' ? (
            <div className="p-3 border border-yellow-300 bg-yellow-50 rounded-lg">
              <p className="text-sm text-yellow-800">
                Please connect to {value.provider} first to load available models.
              </p>
            </div>
          ) : null}
        </div>
      )}

      {/* Advanced Parameters */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Settings className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Model Parameters</span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Temperature: {value.temperature}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={value.temperature}
              onChange={(e) => onChange({ ...value, temperature: parseFloat(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Focused</span>
              <span>Creative</span>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Max Tokens
            </label>
            <input
              type="number"
              min="1"
              max="32768"
              value={value.max_tokens}
              onChange={(e) => onChange({ ...value, max_tokens: parseInt(e.target.value) || 2048 })}
              className="input"
            />
          </div>

          {/* Additional parameters for OpenAI/Anthropic */}
          {(value.provider === 'openai' || value.provider === 'anthropic') && (
            <>
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Top P
                </label>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={value.top_p || ''}
                  onChange={(e) => onChange({ ...value, top_p: e.target.value ? parseFloat(e.target.value) : undefined })}
                  placeholder="0.9"
                  className="input"
                />
              </div>

              {value.provider === 'openai' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Frequency Penalty
                    </label>
                    <input
                      type="number"
                      min="-2"
                      max="2"
                      step="0.1"
                      value={value.frequency_penalty || ''}
                      onChange={(e) => onChange({ ...value, frequency_penalty: e.target.value ? parseFloat(e.target.value) : undefined })}
                      placeholder="0"
                      className="input"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Presence Penalty
                    </label>
                    <input
                      type="number"
                      min="-2"
                      max="2"
                      step="0.1"
                      value={value.presence_penalty || ''}
                      onChange={(e) => onChange({ ...value, presence_penalty: e.target.value ? parseFloat(e.target.value) : undefined })}
                      placeholder="0"
                      className="input"
                    />
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default LLMProviderSelector
