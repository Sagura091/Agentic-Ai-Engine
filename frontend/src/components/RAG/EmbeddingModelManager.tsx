import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { 
  Download, 
  CheckCircle, 
  AlertCircle, 
  Loader2, 
  Settings, 
  Brain,
  Zap,
  Star,
  Info,
  Trash2,
  Play
} from 'lucide-react'
import { embeddingApi } from '../../services/api'
import toast from 'react-hot-toast'

interface EmbeddingModel {
  name: string
  display_name: string
  description: string
  dimensions: number
  size_mb: number
  status: 'available' | 'downloading' | 'downloaded' | 'error'
  download_progress?: number
  provider: string
  recommended: boolean
  performance_score: number
  use_cases: string[]
  requirements: {
    min_ram_gb: number
    gpu_required: boolean
    disk_space_mb: number
  }
}

interface EmbeddingConfig {
  current_model: string
  batch_size: number
  max_length: number
  normalize: boolean
  cache_embeddings: boolean
  device: string
}

const EmbeddingModelManager: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [showConfig, setShowConfig] = useState(false)
  const [testText, setTestText] = useState('')
  const [testResults, setTestResults] = useState<any>(null)
  
  const queryClient = useQueryClient()

  // Fetch available embedding models
  const { data: models = [], isLoading: modelsLoading } = useQuery(
    'embedding-models',
    embeddingApi.listModels,
    {
      refetchInterval: 3000, // Refresh every 3 seconds for download progress
    }
  )

  // Fetch current embedding configuration
  const { data: config, isLoading: configLoading } = useQuery(
    'embedding-config',
    embeddingApi.getConfig
  )

  // Download model mutation
  const downloadMutation = useMutation(embeddingApi.downloadModel, {
    onSuccess: (data, modelName) => {
      toast.success(`Started downloading ${modelName}`)
      queryClient.invalidateQueries('embedding-models')
    },
    onError: (error: any, modelName) => {
      toast.error(`Failed to download ${modelName}: ${error.message}`)
    }
  })

  // Test model mutation
  const testMutation = useMutation(
    ({ modelName, text }: { modelName: string; text: string }) =>
      embeddingApi.testModel(modelName, text),
    {
      onSuccess: (data) => {
        setTestResults(data)
        toast.success('Model test completed')
      },
      onError: (error: any) => {
        toast.error(`Model test failed: ${error.message}`)
      }
    }
  )

  // Update config mutation
  const updateConfigMutation = useMutation(embeddingApi.updateConfig, {
    onSuccess: () => {
      toast.success('Configuration updated successfully')
      queryClient.invalidateQueries('embedding-config')
      setShowConfig(false)
    },
    onError: (error: any) => {
      toast.error(`Failed to update configuration: ${error.message}`)
    }
  })

  const handleDownload = (modelName: string) => {
    downloadMutation.mutate(modelName)
  }

  const handleTest = (modelName: string) => {
    if (!testText.trim()) {
      toast.error('Please enter test text')
      return
    }
    testMutation.mutate({ modelName, text: testText })
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'downloaded':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'downloading':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Download className="h-4 w-4 text-gray-500" />
    }
  }

  const getPerformanceColor = (score: number) => {
    if (score >= 0.9) return 'text-green-600'
    if (score >= 0.7) return 'text-yellow-600'
    return 'text-red-600'
  }

  const recommendedModels = models.filter((model: EmbeddingModel) => model.recommended)
  const otherModels = models.filter((model: EmbeddingModel) => !model.recommended)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <Brain className="h-6 w-6 text-purple-500" />
            <span>Embedding Models</span>
          </h2>
          <p className="text-muted-foreground mt-1">
            Manage and configure embedding models for enhanced RAG performance
          </p>
        </div>
        
        <button
          onClick={() => setShowConfig(!showConfig)}
          className="btn-secondary flex items-center space-x-2"
        >
          <Settings className="h-4 w-4" />
          <span>Configuration</span>
        </button>
      </div>

      {/* Current Configuration */}
      {config && (
        <div className="card p-4 bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
          <div className="flex items-center space-x-2 mb-2">
            <Zap className="h-4 w-4 text-blue-500" />
            <span className="font-medium">Current Model: {config.current_model}</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-muted-foreground">
            <div>Batch Size: {config.batch_size}</div>
            <div>Max Length: {config.max_length}</div>
            <div>Device: {config.device}</div>
            <div>Cache: {config.cache_embeddings ? 'Enabled' : 'Disabled'}</div>
          </div>
        </div>
      )}

      {/* Model Testing */}
      <div className="card p-4">
        <h3 className="font-semibold mb-3 flex items-center space-x-2">
          <Play className="h-4 w-4" />
          <span>Test Embedding Model</span>
        </h3>
        
        <div className="space-y-3">
          <div className="flex space-x-3">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="input flex-1"
            >
              <option value="">Select a model to test</option>
              {models
                .filter((model: EmbeddingModel) => model.status === 'downloaded')
                .map((model: EmbeddingModel) => (
                  <option key={model.name} value={model.name}>
                    {model.display_name}
                  </option>
                ))}
            </select>
            
            <button
              onClick={() => selectedModel && handleTest(selectedModel)}
              disabled={!selectedModel || !testText.trim() || testMutation.isLoading}
              className="btn-primary disabled:opacity-50"
            >
              {testMutation.isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              Test
            </button>
          </div>
          
          <textarea
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="Enter text to generate embeddings..."
            className="input min-h-[80px] resize-none"
          />
          
          {testResults && (
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
              <div className="text-sm space-y-1">
                <div><strong>Embedding Dimensions:</strong> {testResults.dimensions}</div>
                <div><strong>Processing Time:</strong> {testResults.processing_time_ms}ms</div>
                <div><strong>Model:</strong> {testResults.model_name}</div>
                <div className="mt-2">
                  <strong>Sample Values:</strong> [{testResults.embedding_sample.join(', ')}...]
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recommended Models */}
      {recommendedModels.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
            <Star className="h-5 w-5 text-yellow-500" />
            <span>Recommended Models</span>
          </h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {recommendedModels.map((model: EmbeddingModel) => (
              <ModelCard
                key={model.name}
                model={model}
                onDownload={handleDownload}
                isDownloading={downloadMutation.isLoading}
                getStatusIcon={getStatusIcon}
                getPerformanceColor={getPerformanceColor}
              />
            ))}
          </div>
        </div>
      )}

      {/* Other Models */}
      {otherModels.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">All Models</h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {otherModels.map((model: EmbeddingModel) => (
              <ModelCard
                key={model.name}
                model={model}
                onDownload={handleDownload}
                isDownloading={downloadMutation.isLoading}
                getStatusIcon={getStatusIcon}
                getPerformanceColor={getPerformanceColor}
              />
            ))}
          </div>
        </div>
      )}

      {/* Configuration Modal */}
      {showConfig && config && (
        <ConfigurationModal
          config={config}
          onSave={(newConfig) => updateConfigMutation.mutate(newConfig)}
          onClose={() => setShowConfig(false)}
          isLoading={updateConfigMutation.isLoading}
        />
      )}
    </div>
  )
}

// Model Card Component
interface ModelCardProps {
  model: EmbeddingModel
  onDownload: (modelName: string) => void
  isDownloading: boolean
  getStatusIcon: (status: string) => React.ReactNode
  getPerformanceColor: (score: number) => string
}

const ModelCard: React.FC<ModelCardProps> = ({
  model,
  onDownload,
  isDownloading,
  getStatusIcon,
  getPerformanceColor
}) => (
  <div className="card p-4 relative">
    {model.recommended && (
      <div className="absolute top-2 right-2">
        <Star className="h-4 w-4 text-yellow-500 fill-current" />
      </div>
    )}
    
    <div className="flex items-center justify-between mb-3">
      <div>
        <h4 className="font-semibold">{model.display_name}</h4>
        <p className="text-sm text-muted-foreground">{model.provider}</p>
      </div>
      {getStatusIcon(model.status)}
    </div>
    
    <p className="text-sm text-muted-foreground mb-3">{model.description}</p>
    
    <div className="space-y-2 text-sm mb-4">
      <div className="flex justify-between">
        <span>Dimensions:</span>
        <span>{model.dimensions.toLocaleString()}</span>
      </div>
      <div className="flex justify-between">
        <span>Size:</span>
        <span>{model.size_mb} MB</span>
      </div>
      <div className="flex justify-between">
        <span>Performance:</span>
        <span className={getPerformanceColor(model.performance_score)}>
          {(model.performance_score * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex justify-between">
        <span>RAM Required:</span>
        <span>{model.requirements.min_ram_gb} GB</span>
      </div>
    </div>
    
    {model.use_cases.length > 0 && (
      <div className="mb-4">
        <p className="text-xs font-medium text-muted-foreground mb-1">Use Cases:</p>
        <div className="flex flex-wrap gap-1">
          {model.use_cases.map((useCase, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-xs rounded"
            >
              {useCase}
            </span>
          ))}
        </div>
      </div>
    )}
    
    {model.status === 'downloading' && model.download_progress && (
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span>Downloading...</span>
          <span>{model.download_progress.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${model.download_progress}%` }}
          />
        </div>
      </div>
    )}
    
    <div className="flex space-x-2">
      {model.status === 'available' && (
        <button
          onClick={() => onDownload(model.name)}
          disabled={isDownloading}
          className="btn-primary flex-1 text-sm"
        >
          <Download className="h-3 w-3 mr-1" />
          Download
        </button>
      )}
      
      {model.status === 'downloaded' && (
        <button className="btn-secondary flex-1 text-sm">
          <CheckCircle className="h-3 w-3 mr-1" />
          Downloaded
        </button>
      )}
      
      {model.status === 'error' && (
        <button
          onClick={() => onDownload(model.name)}
          className="btn-destructive flex-1 text-sm"
        >
          <AlertCircle className="h-3 w-3 mr-1" />
          Retry
        </button>
      )}
    </div>
  </div>
)

// Configuration Modal Component
interface ConfigurationModalProps {
  config: EmbeddingConfig
  onSave: (config: EmbeddingConfig) => void
  onClose: () => void
  isLoading: boolean
}

const ConfigurationModal: React.FC<ConfigurationModalProps> = ({
  config,
  onSave,
  onClose,
  isLoading
}) => {
  const [formData, setFormData] = useState(config)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSave(formData)
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold mb-4">Embedding Configuration</h3>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Batch Size</label>
            <input
              type="number"
              value={formData.batch_size}
              onChange={(e) => setFormData({ ...formData, batch_size: parseInt(e.target.value) })}
              className="input"
              min="1"
              max="128"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Max Length</label>
            <input
              type="number"
              value={formData.max_length}
              onChange={(e) => setFormData({ ...formData, max_length: parseInt(e.target.value) })}
              className="input"
              min="128"
              max="2048"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Device</label>
            <select
              value={formData.device}
              onChange={(e) => setFormData({ ...formData, device: e.target.value })}
              className="input"
            >
              <option value="auto">Auto</option>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA (GPU)</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="normalize"
              checked={formData.normalize}
              onChange={(e) => setFormData({ ...formData, normalize: e.target.checked })}
              className="rounded"
            />
            <label htmlFor="normalize" className="text-sm">Normalize embeddings</label>
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="cache"
              checked={formData.cache_embeddings}
              onChange={(e) => setFormData({ ...formData, cache_embeddings: e.target.checked })}
              className="rounded"
            />
            <label htmlFor="cache" className="text-sm">Cache embeddings</label>
          </div>
          
          <div className="flex space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary flex-1"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary flex-1"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default EmbeddingModelManager
