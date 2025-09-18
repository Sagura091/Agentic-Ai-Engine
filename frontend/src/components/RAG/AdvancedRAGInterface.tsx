import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { 
  Database, 
  Search, 
  Brain, 
  Zap, 
  Settings, 
  BarChart3,
  TrendingUp,
  FileText,
  Upload,
  Download,
  Cpu,
  Network,
  Target,
  Eye,
  Filter,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Loader2,
  Plus,
  Trash2,
  Edit,
  Copy,
  Share2,
  Lock,
  Unlock,
  Clock,
  Activity,
  Layers,
  Globe,
  Code,
  Gauge
} from 'lucide-react';
import { useQuery, useMutation } from 'react-query';
import { ragApi, embeddingApi, knowledgeBaseApi } from '../../services/api';
import toast from 'react-hot-toast';

interface EmbeddingModel {
  id: string;
  name: string;
  provider: 'sentence-transformers' | 'openai' | 'cohere' | 'custom';
  dimensions: number;
  max_tokens: number;
  status: 'active' | 'inactive' | 'loading';
  performance_metrics: {
    accuracy_score: number;
    speed_ms: number;
    memory_usage_mb: number;
    cost_per_1k_tokens: number;
  };
  supported_languages: string[];
  use_cases: string[];
}

interface ChromaCollection {
  id: string;
  name: string;
  embedding_model: string;
  document_count: number;
  vector_count: number;
  size_mb: number;
  created_at: string;
  last_updated: string;
  metadata: Record<string, any>;
  health_status: 'healthy' | 'degraded' | 'error';
  performance_metrics: {
    avg_query_time_ms: number;
    cache_hit_rate: number;
    index_efficiency: number;
  };
}

interface RAGPipeline {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'configuring';
  components: {
    retriever: {
      type: 'dense' | 'sparse' | 'hybrid';
      model: string;
      top_k: number;
      similarity_threshold: number;
    };
    reranker: {
      enabled: boolean;
      model?: string;
      top_k?: number;
    };
    generator: {
      model: string;
      temperature: number;
      max_tokens: number;
    };
  };
  performance_metrics: {
    retrieval_accuracy: number;
    generation_quality: number;
    latency_ms: number;
    throughput_qps: number;
  };
  knowledge_bases: string[];
}

interface SearchResult {
  id: string;
  content: string;
  metadata: Record<string, any>;
  score: number;
  source: string;
  chunk_id: string;
  embedding_model: string;
  retrieved_at: string;
}

export const AdvancedRAGInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'embeddings' | 'collections' | 'pipelines' | 'search' | 'analytics'>('overview');
  
  // Search and retrieval states
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  
  // Pipeline states
  const [selectedPipeline, setSelectedPipeline] = useState<RAGPipeline | null>(null);
  const [pipelineConfig, setPipelineConfig] = useState<any>({});
  
  // Advanced search options
  const [searchOptions, setSearchOptions] = useState({
    search_type: 'hybrid' as 'dense' | 'sparse' | 'hybrid',
    top_k: 10,
    similarity_threshold: 0.7,
    enable_reranking: true,
    include_metadata: true,
    filter_conditions: {} as Record<string, any>
  });

  // Fetch embedding models
  const { data: embeddingModels, isLoading: modelsLoading } = useQuery(
    'embedding-models',
    () => embeddingApi.getAvailableModels(),
    {
      refetchInterval: 30000
    }
  );

  // Fetch ChromaDB collections
  const { data: chromaCollections, isLoading: collectionsLoading } = useQuery(
    'chroma-collections',
    () => ragApi.getChromaCollections(),
    {
      refetchInterval: 10000
    }
  );

  // Fetch RAG pipelines
  const { data: ragPipelines, isLoading: pipelinesLoading } = useQuery(
    'rag-pipelines',
    () => ragApi.getRAGPipelines(),
    {
      refetchInterval: 15000
    }
  );

  // Advanced semantic search
  const performAdvancedSearch = async () => {
    if (!searchQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    if (selectedCollections.length === 0) {
      toast.error('Please select at least one collection to search');
      return;
    }

    setIsSearching(true);
    setSearchResults([]);

    try {
      const searchRequest = {
        query: searchQuery,
        collections: selectedCollections,
        search_type: searchOptions.search_type,
        top_k: searchOptions.top_k,
        similarity_threshold: searchOptions.similarity_threshold,
        enable_reranking: searchOptions.enable_reranking,
        include_metadata: searchOptions.include_metadata,
        filters: searchOptions.filter_conditions
      };

      const response = await ragApi.advancedSearch(searchRequest);
      setSearchResults(response.data.results);
      
      toast.success(`Found ${response.data.results.length} relevant results`);
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  // Create new RAG pipeline
  const createRAGPipeline = async (config: any) => {
    try {
      const response = await ragApi.createRAGPipeline(config);
      toast.success('RAG pipeline created successfully');
      return response.data;
    } catch (error) {
      console.error('Pipeline creation error:', error);
      toast.error('Failed to create RAG pipeline');
    }
  };

  // Optimize collection performance
  const optimizeCollection = async (collectionId: string) => {
    try {
      await ragApi.optimizeCollection(collectionId);
      toast.success('Collection optimization started');
    } catch (error) {
      console.error('Optimization error:', error);
      toast.error('Failed to optimize collection');
    }
  };

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-6 w-6 text-purple-500" />
              <div>
                <CardTitle className="text-xl">Advanced RAG System Management</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Comprehensive RAG 4.0 system with ChromaDB, embedding models, and advanced retrieval
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <Activity className="h-3 w-3" />
                RAG 4.0 Active
              </Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-muted p-1 rounded-lg w-fit">
        {[
          { id: 'overview', label: 'System Overview', icon: Gauge },
          { id: 'embeddings', label: 'Embedding Models', icon: Cpu },
          { id: 'collections', label: 'ChromaDB Collections', icon: Database },
          { id: 'pipelines', label: 'RAG Pipelines', icon: Network },
          { id: 'search', label: 'Advanced Search', icon: Search },
          { id: 'analytics', label: 'Analytics', icon: BarChart3 }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <tab.icon className="h-4 w-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* System Overview Tab */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* System Status */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Activity className="h-5 w-5" />
                <span>System Status</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Embedding Models:</span>
                  <Badge variant="outline">{embeddingModels?.length || 0} Active</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Collections:</span>
                  <Badge variant="outline">{chromaCollections?.length || 0} Collections</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">RAG Pipelines:</span>
                  <Badge variant="outline">{ragPipelines?.length || 0} Pipelines</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Total Documents:</span>
                  <Badge variant="outline">
                    {chromaCollections?.reduce((sum, col) => sum + col.document_count, 0) || 0}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <TrendingUp className="h-5 w-5" />
                <span>Performance Metrics</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-muted-foreground">Avg Query Time:</span>
                    <span className="text-sm font-medium">
                      {chromaCollections?.length > 0 
                        ? Math.round(chromaCollections.reduce((sum, col) => sum + col.performance_metrics.avg_query_time_ms, 0) / chromaCollections.length)
                        : 0}ms
                    </span>
                  </div>
                  <Progress value={75} className="h-2" />
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-muted-foreground">Cache Hit Rate:</span>
                    <span className="text-sm font-medium">
                      {chromaCollections?.length > 0 
                        ? Math.round(chromaCollections.reduce((sum, col) => sum + col.performance_metrics.cache_hit_rate, 0) / chromaCollections.length * 100)
                        : 0}%
                    </span>
                  </div>
                  <Progress value={85} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-muted-foreground">Index Efficiency:</span>
                    <span className="text-sm font-medium">
                      {chromaCollections?.length > 0 
                        ? Math.round(chromaCollections.reduce((sum, col) => sum + col.performance_metrics.index_efficiency, 0) / chromaCollections.length * 100)
                        : 0}%
                    </span>
                  </div>
                  <Progress value={92} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Zap className="h-5 w-5" />
                <span>Quick Actions</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button className="w-full justify-start" variant="outline">
                <Plus className="h-4 w-4 mr-2" />
                Create New Pipeline
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Upload className="h-4 w-4 mr-2" />
                Import Knowledge Base
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                Optimize Collections
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <BarChart3 className="h-4 w-4 mr-2" />
                View Analytics
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Embedding Models Tab */}
      {activeTab === 'embeddings' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Cpu className="h-5 w-5" />
                <span>Available Embedding Models</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-3">
                  {embeddingModels?.map((model: EmbeddingModel) => (
                    <div key={model.id} className="p-4 border border-border rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h3 className="font-medium text-sm">{model.name}</h3>
                          <p className="text-xs text-muted-foreground mt-1">
                            {model.provider} • {model.dimensions} dimensions
                          </p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {model.provider}
                          </Badge>
                          <Badge
                            variant={model.status === 'active' ? 'default' : 'secondary'}
                            className="text-xs"
                          >
                            {model.status}
                          </Badge>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-xs mb-3">
                        <div>
                          <span className="text-muted-foreground">Accuracy:</span>
                          <p className="font-medium">{Math.round(model.performance_metrics.accuracy_score * 100)}%</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Speed:</span>
                          <p className="font-medium">{model.performance_metrics.speed_ms}ms</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Memory:</span>
                          <p className="font-medium">{model.performance_metrics.memory_usage_mb}MB</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Cost:</span>
                          <p className="font-medium">${model.performance_metrics.cost_per_1k_tokens}</p>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-1 mb-3">
                        {model.use_cases.slice(0, 3).map((useCase, index) => (
                          <span key={index} className="px-2 py-1 bg-muted text-xs rounded">
                            {useCase}
                          </span>
                        ))}
                        {model.use_cases.length > 3 && (
                          <span className="px-2 py-1 bg-muted text-xs rounded">
                            +{model.use_cases.length - 3} more
                          </span>
                        )}
                      </div>

                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">
                          {model.supported_languages.length} languages supported
                        </span>
                        <div className="flex space-x-1">
                          <Button size="sm" variant="outline" className="h-7 px-2">
                            <Eye className="h-3 w-3" />
                          </Button>
                          <Button size="sm" variant="outline" className="h-7 px-2">
                            <Settings className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Plus className="h-5 w-5" />
                <span>Add New Model</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Model Provider
                </label>
                <select className="input w-full">
                  <option value="">Select provider...</option>
                  <option value="sentence-transformers">Sentence Transformers</option>
                  <option value="openai">OpenAI</option>
                  <option value="cohere">Cohere</option>
                  <option value="custom">Custom Model</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Model Name
                </label>
                <Input placeholder="e.g., all-MiniLM-L6-v2" />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Configuration
                </label>
                <Textarea
                  placeholder="Enter model configuration as JSON..."
                  rows={6}
                  className="font-mono text-sm"
                />
              </div>

              <Button className="w-full">
                <Plus className="h-4 w-4 mr-2" />
                Add Embedding Model
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ChromaDB Collections Tab */}
      {activeTab === 'collections' && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Database className="h-5 w-5" />
                  <span>ChromaDB Collections</span>
                </CardTitle>
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Collection
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
                {chromaCollections?.map((collection: ChromaCollection) => (
                  <div key={collection.id} className="p-4 border border-border rounded-lg">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-medium text-sm">{collection.name}</h3>
                        <p className="text-xs text-muted-foreground mt-1">
                          {collection.embedding_model}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge
                          variant={collection.health_status === 'healthy' ? 'default' : 'destructive'}
                          className="text-xs"
                        >
                          {collection.health_status === 'healthy' && <CheckCircle className="h-3 w-3 mr-1" />}
                          {collection.health_status === 'degraded' && <AlertTriangle className="h-3 w-3 mr-1" />}
                          {collection.health_status === 'error' && <XCircle className="h-3 w-3 mr-1" />}
                          {collection.health_status}
                        </Badge>
                      </div>
                    </div>

                    <div className="space-y-2 text-xs mb-3">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Documents:</span>
                        <span className="font-medium">{collection.document_count.toLocaleString()}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Vectors:</span>
                        <span className="font-medium">{collection.vector_count.toLocaleString()}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Size:</span>
                        <span className="font-medium">{collection.size_mb.toFixed(1)} MB</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Query Time:</span>
                        <span className="font-medium">{collection.performance_metrics.avg_query_time_ms}ms</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">
                        Updated {new Date(collection.last_updated).toLocaleDateString()}
                      </span>
                      <div className="flex space-x-1">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 px-2"
                          onClick={() => optimizeCollection(collection.id)}
                        >
                          <RefreshCw className="h-3 w-3" />
                        </Button>
                        <Button size="sm" variant="outline" className="h-7 px-2">
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button size="sm" variant="outline" className="h-7 px-2">
                          <Settings className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Advanced Search Tab */}
      {activeTab === 'search' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Search className="h-5 w-5" />
                  <span>Advanced Semantic Search</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex space-x-2">
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter your semantic search query..."
                    className="flex-1"
                    onKeyPress={(e) => e.key === 'Enter' && performAdvancedSearch()}
                  />
                  <Button
                    onClick={performAdvancedSearch}
                    disabled={!searchQuery.trim() || isSearching || selectedCollections.length === 0}
                  >
                    {isSearching ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Searching...
                      </>
                    ) : (
                      <>
                        <Search className="h-4 w-4 mr-2" />
                        Search
                      </>
                    )}
                  </Button>
                </div>

                {/* Search Options */}
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">
                      Search Type
                    </label>
                    <select
                      value={searchOptions.search_type}
                      onChange={(e) => setSearchOptions(prev => ({
                        ...prev,
                        search_type: e.target.value as any
                      }))}
                      className="input text-sm"
                    >
                      <option value="dense">Dense (Semantic)</option>
                      <option value="sparse">Sparse (Keyword)</option>
                      <option value="hybrid">Hybrid (Best)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">
                      Top K Results
                    </label>
                    <Input
                      type="number"
                      value={searchOptions.top_k}
                      onChange={(e) => setSearchOptions(prev => ({
                        ...prev,
                        top_k: parseInt(e.target.value)
                      }))}
                      min="1"
                      max="100"
                      className="text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">
                      Similarity Threshold
                    </label>
                    <Input
                      type="number"
                      value={searchOptions.similarity_threshold}
                      onChange={(e) => setSearchOptions(prev => ({
                        ...prev,
                        similarity_threshold: parseFloat(e.target.value)
                      }))}
                      min="0"
                      max="1"
                      step="0.1"
                      className="text-sm"
                    />
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="enable-reranking"
                      checked={searchOptions.enable_reranking}
                      onChange={(e) => setSearchOptions(prev => ({
                        ...prev,
                        enable_reranking: e.target.checked
                      }))}
                      className="rounded border-border"
                    />
                    <label htmlFor="enable-reranking" className="text-sm text-foreground">
                      Enable reranking
                    </label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="include-metadata"
                      checked={searchOptions.include_metadata}
                      onChange={(e) => setSearchOptions(prev => ({
                        ...prev,
                        include_metadata: e.target.checked
                      }))}
                      className="rounded border-border"
                    />
                    <label htmlFor="include-metadata" className="text-sm text-foreground">
                      Include metadata
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center space-x-2">
                    <FileText className="h-5 w-5" />
                    <span>Search Results ({searchResults.length})</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    <div className="space-y-3">
                      {searchResults.map((result) => (
                        <div key={result.id} className="p-4 border border-border rounded-lg">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <FileText className="h-4 w-4 text-blue-500" />
                              <span className="font-medium text-sm">
                                {result.metadata?.title || result.source}
                              </span>
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {Math.round(result.score * 100)}% match
                            </Badge>
                          </div>
                          <p className="text-sm text-foreground mb-2 leading-relaxed">
                            {result.content}
                          </p>
                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span>Model: {result.embedding_model}</span>
                            <span>Retrieved: {new Date(result.retrieved_at).toLocaleTimeString()}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Search Control Panel */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Filter className="h-5 w-5" />
                  <span>Collection Selection</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {chromaCollections?.map((collection) => (
                    <div key={collection.id} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id={`collection-${collection.id}`}
                        checked={selectedCollections.includes(collection.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedCollections(prev => [...prev, collection.id]);
                          } else {
                            setSelectedCollections(prev => prev.filter(id => id !== collection.id));
                          }
                        }}
                        className="rounded border-border"
                      />
                      <label htmlFor={`collection-${collection.id}`} className="text-sm text-foreground flex-1">
                        {collection.name}
                      </label>
                      <Badge variant="outline" className="text-xs">
                        {collection.document_count}
                      </Badge>
                    </div>
                  ))}
                </div>
                <div className="mt-3 pt-3 border-t border-border">
                  <Button
                    onClick={() => setSelectedCollections(
                      selectedCollections.length === chromaCollections?.length
                        ? []
                        : chromaCollections?.map(c => c.id) || []
                    )}
                    variant="outline"
                    size="sm"
                    className="w-full"
                  >
                    {selectedCollections.length === chromaCollections?.length ? 'Deselect All' : 'Select All'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};
