import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { 
  Settings, 
  Database, 
  Zap, 
  CheckCircle, 
  AlertTriangle, 
  Loader2,
  RefreshCw,
  Server,
  Key,
  Globe
} from 'lucide-react';
import toast from 'react-hot-toast';
import { embeddingApi } from '../../services/api';

interface EmbeddingConfig {
  embedding_engine: string;
  embedding_model: string;
  embedding_batch_size: number;
  openai_config: {
    url: string;
    key: string;
  };
  ollama_config: {
    url: string;
    key: string;
  };
  azure_openai_config: {
    url: string;
    key: string;
    version: string;
  };
}

const EmbeddingSettings: React.FC = () => {
  const [embeddingEngine, setEmbeddingEngine] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('sentence-transformers/all-MiniLM-L6-v2');
  const [embeddingBatchSize, setEmbeddingBatchSize] = useState(32);
  
  // Provider configurations
  const [openaiConfig, setOpenaiConfig] = useState({ url: '', key: '' });
  const [ollamaConfig, setOllamaConfig] = useState({ url: 'http://localhost:11434', key: '' });
  const [azureConfig, setAzureConfig] = useState({ url: '', key: '', version: '' });
  
  const [isUpdating, setIsUpdating] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);

  const queryClient = useQueryClient();

  // Fetch current embedding configuration
  const { data: embeddingConfig, isLoading } = useQuery(
    'embedding-config',
    embeddingApi.getEmbeddingConfig,
    {
      onSuccess: (data) => {
        if (data) {
          setEmbeddingEngine(data.embedding_engine || '');
          setEmbeddingModel(data.embedding_model || 'sentence-transformers/all-MiniLM-L6-v2');
          setEmbeddingBatchSize(data.embedding_batch_size || 32);
          setOpenaiConfig(data.openai_config || { url: '', key: '' });
          setOllamaConfig(data.ollama_config || { url: 'http://localhost:11434', key: '' });
          setAzureConfig(data.azure_openai_config || { url: '', key: '', version: '' });
        }
      }
    }
  );

  // Update embedding configuration mutation
  const updateConfigMutation = useMutation(embeddingApi.updateEmbeddingConfig, {
    onSuccess: () => {
      toast.success('Embedding configuration updated successfully!');
      queryClient.invalidateQueries('embedding-config');
      setIsUpdating(false);
    },
    onError: (error: any) => {
      toast.error(`Failed to update embedding configuration: ${error.message}`);
      setIsUpdating(false);
    }
  });

  const handleUpdateConfig = async () => {
    // Validation
    if (embeddingEngine === 'openai' && (!openaiConfig.url || !openaiConfig.key)) {
      toast.error('OpenAI URL and API Key are required');
      return;
    }
    
    if (embeddingEngine === 'azure_openai' && (!azureConfig.url || !azureConfig.key || !azureConfig.version)) {
      toast.error('Azure OpenAI URL, API Key, and Version are required');
      return;
    }

    if (embeddingEngine === 'ollama' && !ollamaConfig.url) {
      toast.error('Ollama URL is required');
      return;
    }

    setIsUpdating(true);
    
    await updateConfigMutation.mutateAsync({
      embedding_engine: embeddingEngine,
      embedding_model: embeddingModel,
      embedding_batch_size: embeddingBatchSize,
      openai_config: openaiConfig,
      ollama_config: ollamaConfig,
      azure_openai_config: azureConfig
    });
  };

  const testConnection = async () => {
    setTestingConnection(true);
    try {
      // Test the connection based on the selected engine
      await embeddingApi.testEmbeddingConnection({
        embedding_engine: embeddingEngine,
        embedding_model: embeddingModel,
        openai_config: openaiConfig,
        ollama_config: ollamaConfig,
        azure_openai_config: azureConfig
      });
      toast.success('Connection test successful!');
    } catch (error: any) {
      toast.error(`Connection test failed: ${error.message}`);
    } finally {
      setTestingConnection(false);
    }
  };

  const resetToDefaults = () => {
    setEmbeddingEngine('');
    setEmbeddingModel('sentence-transformers/all-MiniLM-L6-v2');
    setEmbeddingBatchSize(32);
    setOpenaiConfig({ url: '', key: '' });
    setOllamaConfig({ url: 'http://localhost:11434', key: '' });
    setAzureConfig({ url: '', key: '', version: '' });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading embedding configuration...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Embedding Settings</h2>
          <p className="text-muted-foreground">
            Configure global embedding model settings for all knowledge bases
          </p>
        </div>
        <Badge variant="outline" className="flex items-center space-x-1">
          <Database className="h-3 w-3" />
          <span>Global Configuration</span>
        </Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Embedding Engine Configuration</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Embedding Engine Selection */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Embedding Engine
            </label>
            <select
              value={embeddingEngine}
              onChange={(e) => setEmbeddingEngine(e.target.value)}
              className="input w-full"
            >
              <option value="">Default (Sentence Transformers)</option>
              <option value="openai">OpenAI</option>
              <option value="azure_openai">Azure OpenAI</option>
              <option value="ollama">Ollama</option>
            </select>
            <p className="text-xs text-muted-foreground mt-1">
              Choose the embedding engine for all knowledge bases
            </p>
          </div>

          {/* Provider-specific configurations */}
          {embeddingEngine === 'openai' && (
            <div className="space-y-4 p-4 bg-muted rounded-lg">
              <h4 className="font-medium flex items-center space-x-2">
                <Globe className="h-4 w-4" />
                <span>OpenAI Configuration</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">API Base URL</label>
                  <input
                    type="text"
                    value={openaiConfig.url}
                    onChange={(e) => setOpenaiConfig({ ...openaiConfig, url: e.target.value })}
                    placeholder="https://api.openai.com/v1"
                    className="input w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">API Key</label>
                  <input
                    type="password"
                    value={openaiConfig.key}
                    onChange={(e) => setOpenaiConfig({ ...openaiConfig, key: e.target.value })}
                    placeholder="sk-..."
                    className="input w-full"
                  />
                </div>
              </div>
            </div>
          )}

          {embeddingEngine === 'ollama' && (
            <div className="space-y-4 p-4 bg-muted rounded-lg">
              <h4 className="font-medium flex items-center space-x-2">
                <Server className="h-4 w-4" />
                <span>Ollama Configuration</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">API Base URL</label>
                  <input
                    type="text"
                    value={ollamaConfig.url}
                    onChange={(e) => setOllamaConfig({ ...ollamaConfig, url: e.target.value })}
                    placeholder="http://localhost:11434"
                    className="input w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">API Key (Optional)</label>
                  <input
                    type="password"
                    value={ollamaConfig.key}
                    onChange={(e) => setOllamaConfig({ ...ollamaConfig, key: e.target.value })}
                    placeholder="Optional API key"
                    className="input w-full"
                  />
                </div>
              </div>
            </div>
          )}

          {embeddingEngine === 'azure_openai' && (
            <div className="space-y-4 p-4 bg-muted rounded-lg">
              <h4 className="font-medium flex items-center space-x-2">
                <Key className="h-4 w-4" />
                <span>Azure OpenAI Configuration</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">API Base URL</label>
                  <input
                    type="text"
                    value={azureConfig.url}
                    onChange={(e) => setAzureConfig({ ...azureConfig, url: e.target.value })}
                    placeholder="https://your-resource.openai.azure.com"
                    className="input w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">API Key</label>
                  <input
                    type="password"
                    value={azureConfig.key}
                    onChange={(e) => setAzureConfig({ ...azureConfig, key: e.target.value })}
                    placeholder="Your Azure API key"
                    className="input w-full"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium mb-1">API Version</label>
                  <input
                    type="text"
                    value={azureConfig.version}
                    onChange={(e) => setAzureConfig({ ...azureConfig, version: e.target.value })}
                    placeholder="2023-05-15"
                    className="input w-full"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Embedding Model */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Embedding Model
            </label>
            <input
              type="text"
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              placeholder="sentence-transformers/all-MiniLM-L6-v2"
              className="input w-full"
            />
            <p className="text-xs text-muted-foreground mt-1">
              {embeddingEngine === 'ollama' 
                ? 'Enter the Ollama model name (e.g., nomic-embed-text)'
                : 'Enter the embedding model identifier'
              }
            </p>
          </div>

          {/* Batch Size */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Batch Size
            </label>
            <input
              type="number"
              value={embeddingBatchSize}
              onChange={(e) => setEmbeddingBatchSize(parseInt(e.target.value) || 32)}
              min="1"
              max="100"
              className="input w-full"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Number of documents to process in each batch (1-100)
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between pt-4 border-t border-border">
            <div className="flex space-x-3">
              <Button
                onClick={testConnection}
                disabled={testingConnection || !embeddingModel}
                variant="outline"
                className="flex items-center space-x-2"
              >
                {testingConnection ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4" />
                )}
                <span>Test Connection</span>
              </Button>
              
              <Button
                onClick={resetToDefaults}
                variant="ghost"
                className="flex items-center space-x-2"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Reset to Defaults</span>
              </Button>
            </div>

            <Button
              onClick={handleUpdateConfig}
              disabled={isUpdating || !embeddingModel}
              className="flex items-center space-x-2"
            >
              {isUpdating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <CheckCircle className="h-4 w-4" />
              )}
              <span>Save Configuration</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Current Configuration Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="h-5 w-5" />
            <span>Current Configuration Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-lg font-semibold text-foreground">
                {embeddingEngine || 'Default'}
              </div>
              <div className="text-sm text-muted-foreground">Engine</div>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-lg font-semibold text-foreground truncate">
                {embeddingModel}
              </div>
              <div className="text-sm text-muted-foreground">Model</div>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-lg font-semibold text-foreground">
                {embeddingBatchSize}
              </div>
              <div className="text-sm text-muted-foreground">Batch Size</div>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-blue-600" />
              <span className="text-sm text-blue-800 dark:text-blue-200">
                This configuration applies to all knowledge bases in the system. Changes will affect new document embeddings.
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EmbeddingSettings;
