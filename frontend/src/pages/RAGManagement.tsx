import React, { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { useNavigate } from 'react-router-dom'
import {
  Upload,
  Search,
  Database,
  Download,
  Settings,
  FileText,
  Brain,
  Zap,
  CheckCircle,
  AlertCircle,
  Loader2,
  Trash2,
  Eye,
  Plus,
  BarChart3,
  Users,
  Activity,
  Globe,
  Image,
  ArrowRight,
  Sparkles,
  Target,
  TrendingUp,
  MessageSquare
} from 'lucide-react'
import { ragApi, embeddingApi } from '../services/api'
import toast from 'react-hot-toast'

interface EmbeddingModel {
  name: string
  description: string
  dimensions: number
  size_mb: number
  status: 'available' | 'downloading' | 'downloaded' | 'error'
  download_progress?: number
  provider: string
  recommended: boolean
}

interface Collection {
  name: string
  document_count: number
  size_mb: number
  created_at: string
  last_updated: string
  embedding_model: string
}

interface KnowledgeDocument {
  id: string
  title: string
  content_preview: string
  collection: string
  chunks_count: number
  created_at: string
  metadata: Record<string, any>
}

const RAGManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'documents' | 'collections' | 'embeddings' | 'search'>('overview')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedCollection, setSelectedCollection] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [isSearching, setIsSearching] = useState(false)

  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Fetch embedding models
  const { data: embeddingModels = [], isLoading: modelsLoading } = useQuery(
    'embedding-models',
    embeddingApi.listModels,
    {
      refetchInterval: 5000, // Refresh every 5 seconds to update download progress
    }
  )

  // Fetch collections
  const { data: collections = [], isLoading: collectionsLoading } = useQuery(
    'rag-collections',
    ragApi.listCollections
  )

  // Fetch RAG statistics
  const { data: ragStats } = useQuery('rag-stats', ragApi.getStats)

  // Download embedding model mutation
  const downloadModelMutation = useMutation(embeddingApi.downloadModel, {
    onSuccess: () => {
      toast.success('Model download started')
      queryClient.invalidateQueries('embedding-models')
    },
    onError: (error: any) => {
      toast.error(`Failed to download model: ${error.message}`)
    }
  })

  // Upload file mutation
  const uploadFileMutation = useMutation(
    ({ file, collection }: { file: File; collection?: string }) => 
      ragApi.uploadFile(file, collection),
    {
      onSuccess: (data) => {
        toast.success(`File uploaded successfully: ${data.filename}`)
        setSelectedFile(null)
        queryClient.invalidateQueries('rag-collections')
      },
      onError: (error: any) => {
        toast.error(`Upload failed: ${error.message}`)
      }
    }
  )

  // Search knowledge mutation
  const searchMutation = useMutation(ragApi.searchKnowledge, {
    onSuccess: (data) => {
      setSearchResults(data.results || [])
      setIsSearching(false)
    },
    onError: (error: any) => {
      toast.error(`Search failed: ${error.message}`)
      setIsSearching(false)
    }
  })

  const handleFileUpload = async () => {
    if (!selectedFile) return
    
    setIsUploading(true)
    try {
      await uploadFileMutation.mutateAsync({ 
        file: selectedFile, 
        collection: selectedCollection || undefined 
      })
    } finally {
      setIsUploading(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    
    setIsSearching(true)
    await searchMutation.mutateAsync({
      query: searchQuery,
      collection: selectedCollection || undefined,
      top_k: 10
    })
  }

  const handleDownloadModel = (modelName: string) => {
    downloadModelMutation.mutate(modelName)
  }

  const getModelStatusIcon = (status: string) => {
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">RAG Management</h1>
          <p className="text-muted-foreground mt-1">
            Manage knowledge base, embeddings, and document retrieval
          </p>
        </div>
        
        {ragStats && (
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-blue-500" />
              <span>{ragStats.total_documents} docs</span>
            </div>
            <div className="flex items-center space-x-2">
              <Brain className="h-4 w-4 text-purple-500" />
              <span>{ragStats.total_collections} collections</span>
            </div>
            <div className="flex items-center space-x-2">
              <Zap className="h-4 w-4 text-yellow-500" />
              <span>{ragStats.total_embeddings} embeddings</span>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex items-center space-x-1 bg-muted p-1 rounded-lg w-fit">
        {[
          { id: 'overview', label: 'Overview', icon: Eye },
          { id: 'documents', label: 'Documents', icon: FileText },
          { id: 'collections', label: 'Collections', icon: Database },
          { id: 'embeddings', label: 'Embeddings', icon: Brain },
          { id: 'search', label: 'Search', icon: Search }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as any)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
              activeTab === id
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Revolutionary RAG 4.0 Features */}
            <div className="card p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center space-x-2">
                <Sparkles className="h-6 w-6 text-purple-500" />
                <span>Revolutionary RAG 4.0 Features</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Advanced Analytics */}
                <div
                  onClick={() => navigate('/rag/analytics')}
                  className="card p-4 cursor-pointer hover:shadow-lg transition-all duration-200 border-2 border-transparent hover:border-purple-200"
                >
                  <div className="flex items-center justify-between mb-3">
                    <BarChart3 className="h-8 w-8 text-purple-500" />
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <h4 className="font-semibold mb-2">Advanced Analytics</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Search pattern analysis, query optimization, and intelligent insights
                  </p>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full">NEW</span>
                    <span className="text-muted-foreground">Revolutionary</span>
                  </div>
                </div>

                {/* Collaborative Workspace */}
                <div
                  onClick={() => navigate('/rag/collaboration')}
                  className="card p-4 cursor-pointer hover:shadow-lg transition-all duration-200 border-2 border-transparent hover:border-blue-200"
                >
                  <div className="flex items-center justify-between mb-3">
                    <Users className="h-8 w-8 text-blue-500" />
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <h4 className="font-semibold mb-2">Collaborative Workspace</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Shared workspaces, knowledge contribution, and team collaboration
                  </p>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full">NEW</span>
                    <span className="text-muted-foreground">Team Intelligence</span>
                  </div>
                </div>

                {/* Performance Monitoring */}
                <div
                  onClick={() => navigate('/rag/performance')}
                  className="card p-4 cursor-pointer hover:shadow-lg transition-all duration-200 border-2 border-transparent hover:border-green-200"
                >
                  <div className="flex items-center justify-between mb-3">
                    <Activity className="h-8 w-8 text-green-500" />
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <h4 className="font-semibold mb-2">Performance Monitoring</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Real-time system analytics and performance optimization
                  </p>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full">LIVE</span>
                    <span className="text-muted-foreground">Real-time</span>
                  </div>
                </div>

                {/* Knowledge Graph */}
                <div
                  onClick={() => navigate('/rag/knowledge-graph')}
                  className="card p-4 cursor-pointer hover:shadow-lg transition-all duration-200 border-2 border-transparent hover:border-orange-200"
                >
                  <div className="flex items-center justify-between mb-3">
                    <Globe className="h-8 w-8 text-orange-500" />
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <h4 className="font-semibold mb-2">Knowledge Graph</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Entity relationships, graph-based search, and knowledge mapping
                  </p>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2 py-1 bg-orange-100 text-orange-800 rounded-full">BETA</span>
                    <span className="text-muted-foreground">Graph Intelligence</span>
                  </div>
                </div>

                {/* Multi-Modal RAG */}
                <div
                  onClick={() => navigate('/rag/multimodal')}
                  className="card p-4 cursor-pointer hover:shadow-lg transition-all duration-200 border-2 border-transparent hover:border-pink-200"
                >
                  <div className="flex items-center justify-between mb-3">
                    <Image className="h-8 w-8 text-pink-500" />
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <h4 className="font-semibold mb-2">Multi-Modal RAG</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Process images, audio, video, and documents with unified search
                  </p>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2 py-1 bg-pink-100 text-pink-800 rounded-full">NEW</span>
                    <span className="text-muted-foreground">Cross-Modal</span>
                  </div>
                </div>

                {/* Revolutionary Search */}
                <div
                  onClick={() => navigate('/rag/search')}
                  className="card p-4 cursor-pointer hover:shadow-lg transition-all duration-200 border-2 border-transparent hover:border-indigo-200"
                >
                  <div className="flex items-center justify-between mb-3">
                    <Target className="h-8 w-8 text-indigo-500" />
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <h4 className="font-semibold mb-2">Revolutionary Search</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Advanced search modes with contextual filters and AI optimization
                  </p>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-full">ENHANCED</span>
                    <span className="text-muted-foreground">AI-Powered</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">System Status</p>
                    <p className="text-2xl font-bold text-green-600">Operational</p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-green-500" />
                </div>
              </div>

              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">RAG Version</p>
                    <p className="text-2xl font-bold">4.0</p>
                  </div>
                  <Sparkles className="h-8 w-8 text-purple-500" />
                </div>
              </div>

              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Features Active</p>
                    <p className="text-2xl font-bold">10+</p>
                  </div>
                  <Zap className="h-8 w-8 text-yellow-500" />
                </div>
              </div>

              <div className="card p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Performance</p>
                    <p className="text-2xl font-bold text-blue-600">Excellent</p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-blue-500" />
                </div>
              </div>
            </div>

            {/* Getting Started */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4">Getting Started with RAG 4.0</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-2">1. Upload Documents</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Start by uploading your documents to create knowledge bases
                  </p>
                  <button
                    onClick={() => setActiveTab('documents')}
                    className="btn-primary text-sm"
                  >
                    Upload Documents
                  </button>
                </div>

                <div>
                  <h4 className="font-medium mb-2">2. Explore Analytics</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    View comprehensive analytics and optimization insights
                  </p>
                  <button
                    onClick={() => navigate('/rag/analytics')}
                    className="btn-secondary text-sm"
                  >
                    View Analytics
                  </button>
                </div>

                <div>
                  <h4 className="font-medium mb-2">3. Try Advanced Search</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Experience revolutionary search with multiple modes
                  </p>
                  <button
                    onClick={() => navigate('/rag/search')}
                    className="btn-secondary text-sm"
                  >
                    Advanced Search
                  </button>
                </div>

                <div>
                  <h4 className="font-medium mb-2">4. Collaborate</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Create workspaces and collaborate with your team
                  </p>
                  <button
                    onClick={() => navigate('/rag/collaboration')}
                    className="btn-secondary text-sm"
                  >
                    Start Collaborating
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <div className="space-y-6">
            {/* Upload Section */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Upload className="h-5 w-5" />
                <span>Upload Documents</span>
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-1">
                    <input
                      type="file"
                      onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                      accept=".pdf,.docx,.txt,.md,.json,.csv"
                      className="input"
                    />
                  </div>
                  
                  <select
                    value={selectedCollection}
                    onChange={(e) => setSelectedCollection(e.target.value)}
                    className="input w-48"
                  >
                    <option value="">Default Collection</option>
                    {Array.isArray(collections) && collections.map((collection: Collection) => (
                      <option key={collection.name} value={collection.name}>
                        {collection.name}
                      </option>
                    ))}
                  </select>
                  
                  <button
                    onClick={handleFileUpload}
                    disabled={!selectedFile || isUploading}
                    className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isUploading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Upload className="h-4 w-4" />
                    )}
                    Upload
                  </button>
                </div>
                
                <p className="text-sm text-muted-foreground">
                  Supported formats: PDF, DOCX, TXT, MD, JSON, CSV
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Collections Tab */}
        {activeTab === 'collections' && (
          <div className="space-y-4">
            {collectionsLoading ? (
              <div className="text-center py-8">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                <p>Loading collections...</p>
              </div>
            ) : Array.isArray(collections) && collections.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {collections.map((collection: Collection) => (
                  <div key={collection.name} className="card p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold">{collection.name}</h4>
                      <Database className="h-4 w-4 text-blue-500" />
                    </div>
                    <div className="space-y-1 text-sm text-muted-foreground">
                      <p>{collection.document_count} documents</p>
                      <p>{collection.size_mb.toFixed(1)} MB</p>
                      <p>Model: {collection.embedding_model}</p>
                      <p>Updated: {new Date(collection.last_updated).toLocaleDateString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No collections found</h3>
                <p className="text-muted-foreground">Upload documents to create collections</p>
              </div>
            )}
          </div>
        )}

        {/* Embeddings Tab */}
        {activeTab === 'embeddings' && (
          <div className="space-y-4">
            {modelsLoading ? (
              <div className="text-center py-8">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                <p>Loading embedding models...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {Array.isArray(embeddingModels) && embeddingModels.map((model: EmbeddingModel) => (
                  <div key={model.name} className="card p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-semibold">{model.name}</h4>
                        {model.recommended && (
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                            Recommended
                          </span>
                        )}
                      </div>
                      {getModelStatusIcon(model.status)}
                    </div>
                    
                    <p className="text-sm text-muted-foreground mb-3">{model.description}</p>
                    
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Dimensions:</span>
                        <span>{model.dimensions}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Size:</span>
                        <span>{model.size_mb} MB</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Provider:</span>
                        <span>{model.provider}</span>
                      </div>
                    </div>
                    
                    {model.status === 'downloading' && model.download_progress && (
                      <div className="mt-3">
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
                    
                    <div className="mt-4 flex space-x-2">
                      {model.status === 'available' && (
                        <button
                          onClick={() => handleDownloadModel(model.name)}
                          disabled={downloadModelMutation.isLoading}
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
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Search Tab */}
        {activeTab === 'search' && (
          <div className="space-y-6">
            {/* Search Interface */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                <Search className="h-5 w-5" />
                <span>Knowledge Search</span>
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-1">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search knowledge base..."
                      className="input"
                      onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                    />
                  </div>
                  
                  <select
                    value={selectedCollection}
                    onChange={(e) => setSelectedCollection(e.target.value)}
                    className="input w-48"
                  >
                    <option value="">All Collections</option>
                    {Array.isArray(collections) && collections.map((collection: Collection) => (
                      <option key={collection.name} value={collection.name}>
                        {collection.name}
                      </option>
                    ))}
                  </select>
                  
                  <button
                    onClick={handleSearch}
                    disabled={!searchQuery.trim() || isSearching}
                    className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isSearching ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Search className="h-4 w-4" />
                    )}
                    Search
                  </button>
                </div>
              </div>
            </div>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="space-y-4">
                <h4 className="text-lg font-semibold">Search Results ({searchResults.length})</h4>
                {searchResults.map((result, index) => (
                  <div key={index} className="card p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h5 className="font-medium">{result.metadata?.title || `Result ${index + 1}`}</h5>
                      <span className="text-sm text-muted-foreground">
                        Score: {(result.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      {result.content.substring(0, 200)}...
                    </p>
                    <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                      <span>Collection: {result.metadata?.collection || 'default'}</span>
                      <span>Source: {result.metadata?.source || 'unknown'}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default RAGManagement
