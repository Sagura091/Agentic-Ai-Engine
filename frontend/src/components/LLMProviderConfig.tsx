import React, { useState, useEffect } from 'react'
import { 
  Settings, 
  Eye, 
  EyeOff, 
  CheckCircle, 
  XCircle, 
  Loader2,
  AlertTriangle,
  Key,
  Server,
  Zap
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import toast from 'react-hot-toast'

interface LLMProvider {
  id: string
  name: string
  description: string
  enabled: boolean
  api_key?: string
  base_url?: string
  models: string[]
  status: 'connected' | 'disconnected' | 'error' | 'testing'
  last_tested?: string
}

interface LLMProviderConfigProps {
  onProviderUpdate?: (providers: LLMProvider[]) => void
}

export const LLMProviderConfig: React.FC<LLMProviderConfigProps> = ({ onProviderUpdate }) => {
  const [providers, setProviders] = useState<LLMProvider[]>([
    {
      id: 'ollama',
      name: 'Ollama',
      description: 'Local LLM inference server',
      enabled: true,
      base_url: 'http://localhost:11434',
      models: [],
      status: 'disconnected'
    },
    {
      id: 'openai',
      name: 'OpenAI',
      description: 'GPT models from OpenAI',
      enabled: false,
      api_key: '',
      base_url: 'https://api.openai.com/v1',
      models: ['gpt-4', 'gpt-3.5-turbo'],
      status: 'disconnected'
    },
    {
      id: 'anthropic',
      name: 'Anthropic',
      description: 'Claude models from Anthropic',
      enabled: false,
      api_key: '',
      base_url: 'https://api.anthropic.com',
      models: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
      status: 'disconnected'
    },
    {
      id: 'google',
      name: 'Google AI',
      description: 'Gemini models from Google',
      enabled: false,
      api_key: '',
      base_url: 'https://generativelanguage.googleapis.com',
      models: ['gemini-pro', 'gemini-pro-vision'],
      status: 'disconnected'
    }
  ])

  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({})
  const [testingProvider, setTestingProvider] = useState<string | null>(null)

  useEffect(() => {
    loadProviderConfigs()
  }, [])

  const loadProviderConfigs = async () => {
    try {
      const response = await fetch('/api/v1/llm-providers')
      if (response.ok) {
        const configs = await response.json()
        setProviders(configs)
      }
    } catch (error) {
      console.error('Failed to load provider configs:', error)
    }
  }

  const updateProvider = (providerId: string, updates: Partial<LLMProvider>) => {
    setProviders(prev => 
      prev.map(provider => 
        provider.id === providerId 
          ? { ...provider, ...updates }
          : provider
      )
    )
  }

  const testProviderConnection = async (providerId: string) => {
    const provider = providers.find(p => p.id === providerId)
    if (!provider) return

    setTestingProvider(providerId)
    updateProvider(providerId, { status: 'testing' })

    try {
      const response = await fetch('/api/v1/llm-providers/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider_id: providerId,
          api_key: provider.api_key,
          base_url: provider.base_url
        })
      })

      const result = await response.json()
      
      if (response.ok && result.success) {
        updateProvider(providerId, { 
          status: 'connected',
          models: result.models || provider.models,
          last_tested: new Date().toISOString()
        })
        toast.success(`${provider.name} connection successful`)
      } else {
        updateProvider(providerId, { status: 'error' })
        toast.error(`${provider.name} connection failed: ${result.error || 'Unknown error'}`)
      }
    } catch (error) {
      updateProvider(providerId, { status: 'error' })
      toast.error(`${provider.name} connection failed`)
    } finally {
      setTestingProvider(null)
    }
  }

  const saveProviderConfig = async (providerId: string) => {
    const provider = providers.find(p => p.id === providerId)
    if (!provider) return

    try {
      const response = await fetch(`/api/v1/llm-providers/${providerId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(provider)
      })

      if (response.ok) {
        toast.success(`${provider.name} configuration saved`)
        if (onProviderUpdate) {
          onProviderUpdate(providers)
        }
      } else {
        toast.error(`Failed to save ${provider.name} configuration`)
      }
    } catch (error) {
      toast.error(`Failed to save ${provider.name} configuration`)
    }
  }

  const toggleApiKeyVisibility = (providerId: string) => {
    setShowApiKeys(prev => ({
      ...prev,
      [providerId]: !prev[providerId]
    }))
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />
      case 'testing':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
      case 'error':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      case 'testing':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">LLM Provider Configuration</h2>
          <p className="text-muted-foreground">Configure and manage your AI model providers</p>
        </div>
        <Button onClick={loadProviderConfigs} variant="outline">
          <Settings className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid gap-6">
        {providers.map((provider) => (
          <Card key={provider.id}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-primary/10 rounded-lg">
                    {provider.id === 'ollama' && <Server className="h-5 w-5" />}
                    {provider.id === 'openai' && <Zap className="h-5 w-5" />}
                    {provider.id === 'anthropic' && <Key className="h-5 w-5" />}
                    {provider.id === 'google' && <Settings className="h-5 w-5" />}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">{provider.name}</h3>
                    <p className="text-sm text-muted-foreground">{provider.description}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <Badge className={getStatusColor(provider.status)}>
                    {getStatusIcon(provider.status)}
                    <span className="ml-1 capitalize">{provider.status}</span>
                  </Badge>
                  
                  <Switch
                    checked={provider.enabled}
                    onCheckedChange={(enabled) => updateProvider(provider.id, { enabled })}
                  />
                </div>
              </CardTitle>
            </CardHeader>
            
            <CardContent className="space-y-4">
              {/* Base URL Configuration */}
              <div className="space-y-2">
                <Label htmlFor={`${provider.id}-url`}>Base URL</Label>
                <Input
                  id={`${provider.id}-url`}
                  value={provider.base_url || ''}
                  onChange={(e) => updateProvider(provider.id, { base_url: e.target.value })}
                  placeholder="Enter base URL"
                  disabled={!provider.enabled}
                />
              </div>

              {/* API Key Configuration (for non-Ollama providers) */}
              {provider.id !== 'ollama' && (
                <div className="space-y-2">
                  <Label htmlFor={`${provider.id}-key`}>API Key</Label>
                  <div className="relative">
                    <Input
                      id={`${provider.id}-key`}
                      type={showApiKeys[provider.id] ? 'text' : 'password'}
                      value={provider.api_key || ''}
                      onChange={(e) => updateProvider(provider.id, { api_key: e.target.value })}
                      placeholder="Enter API key"
                      disabled={!provider.enabled}
                      className="pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => toggleApiKeyVisibility(provider.id)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showApiKeys[provider.id] ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Available Models */}
              {provider.models.length > 0 && (
                <div className="space-y-2">
                  <Label>Available Models</Label>
                  <div className="flex flex-wrap gap-2">
                    {provider.models.map((model) => (
                      <Badge key={model} variant="outline">
                        {model}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Last Tested */}
              {provider.last_tested && (
                <div className="text-sm text-muted-foreground">
                  Last tested: {new Date(provider.last_tested).toLocaleString()}
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex space-x-3 pt-4">
                <Button
                  onClick={() => testProviderConnection(provider.id)}
                  disabled={!provider.enabled || testingProvider === provider.id}
                  variant="outline"
                >
                  {testingProvider === provider.id ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <CheckCircle className="h-4 w-4 mr-2" />
                  )}
                  Test Connection
                </Button>
                
                <Button
                  onClick={() => saveProviderConfig(provider.id)}
                  disabled={!provider.enabled}
                >
                  Save Configuration
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

export default LLMProviderConfig
