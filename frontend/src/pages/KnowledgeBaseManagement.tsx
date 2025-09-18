import React, { useState, useCallback, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { useNavigate } from 'react-router-dom'
import {
  Plus,
  Database,
  Upload,
  Search,
  Settings,
  Trash2,
  Edit3,
  FileText,
  Users,
  Globe,
  Lock,
  Tag,
  Calendar,
  BarChart3,
  BookOpen,
  Folder,
  Filter,
  CheckCircle,
  AlertCircle,
  Loader2,
  Download,
  Eye,
  Brain,
  Zap,
  Target,
  X
} from 'lucide-react'
import toast from 'react-hot-toast'
import { knowledgeBaseApi, ragApi, enhancedOrchestrationApi } from '../services/api'

interface KnowledgeBase {
  id: string
  name: string
  description?: string
  use_case?: string
  tags: string[]
  document_count: number
  size_mb: number
  created_at: string
  created_by?: string
  embedding_model?: string
  is_public: boolean
}

const KnowledgeBaseManagement: React.FC = () => {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<'overview' | 'create' | 'browse' | 'upload' | 'search'>('overview')
  const [searchQuery, setSearchQuery] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedKB, setSelectedKB] = useState<KnowledgeBase | null>(null)

  // Document upload state
  const [uploadingFiles, setUploadingFiles] = useState<{[key: string]: boolean}>({})
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Search modal state
  const [showSearchModal, setShowSearchModal] = useState(false)
  const [searchModalKB, setSearchModalKB] = useState<KnowledgeBase | null>(null)
  const [modalSearchQuery, setModalSearchQuery] = useState('')
  const [modalSearchResults, setModalSearchResults] = useState<any[]>([])
  const [isModalSearching, setIsModalSearching] = useState(false)

  // Form states
  const [newKBName, setNewKBName] = useState('')
  const [newKBDescription, setNewKBDescription] = useState('')
  const [newKBTags, setNewKBTags] = useState<string[]>([])
  const [newKBIsPublic, setNewKBIsPublic] = useState(false)
  const [newKBUseCase, setNewKBUseCase] = useState('general')

  // File upload states
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const [isUploading, setIsUploading] = useState(false)
  const [uploadResults, setUploadResults] = useState<any[]>([])

  // Search states
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchKB, setSearchKB] = useState<string>('')

  const queryClient = useQueryClient()

  // Fetch knowledge bases
  const { data: knowledgeBases = [], isLoading: kbLoading } = useQuery(
    'knowledge-bases',
    knowledgeBaseApi.listKnowledgeBases,
    {
      onSuccess: (data) => {
        console.log('Knowledge bases loaded:', data)
      },
      onError: (error) => {
        console.error('Failed to load knowledge bases:', error)
      }
    }
  )

  // Create knowledge base mutation
  const createKBMutation = useMutation(knowledgeBaseApi.createKnowledgeBase, {
    onSuccess: (data) => {
      console.log('🎉 Knowledge base creation response:', data)
      toast.success('Knowledge base created successfully!')
      setShowCreateModal(false)
      resetForm()
      queryClient.invalidateQueries('knowledge-bases')

      // Navigate to the newly created knowledge base page
      console.log('🔍 Full response data:', JSON.stringify(data, null, 2))

      if (data && data.knowledge_base && data.knowledge_base.id) {
        const kbId = data.knowledge_base.id
        console.log('🚀 Navigating to KB:', kbId)
        console.log('🔗 Navigation URL:', `/knowledge-bases/${kbId}`)

        // Add a small delay to ensure the toast is shown before navigation
        setTimeout(() => {
          navigate(`/knowledge-bases/${kbId}`)
          console.log('✅ Navigation executed')
        }, 500)
      } else {
        console.error('❌ No KB ID in response:', data)
        console.error('❌ Response structure:', {
          hasData: !!data,
          hasKnowledgeBase: !!(data && data.knowledge_base),
          hasId: !!(data && data.knowledge_base && data.knowledge_base.id),
          dataKeys: data ? Object.keys(data) : 'no data',
          kbKeys: (data && data.knowledge_base) ? Object.keys(data.knowledge_base) : 'no kb'
        })
        toast.error('Knowledge base created but navigation failed - check console')
      }
    },
    onError: (error: any) => {
      toast.error(`Failed to create knowledge base: ${error.message}`)
    }
  })

  // Delete knowledge base mutation
  const deleteKBMutation = useMutation(knowledgeBaseApi.deleteKnowledgeBase, {
    onSuccess: () => {
      toast.success('Knowledge base deleted successfully!')
      queryClient.invalidateQueries('knowledge-bases')
    },
    onError: (error: any) => {
      toast.error(`Failed to delete knowledge base: ${error.message}`)
    }
  })

  // Document upload mutation
  const uploadDocumentMutation = useMutation(
    ({ kbId, file, title, metadata }: { kbId: string; file: File; title?: string; metadata?: any }) =>
      knowledgeBaseApi.uploadDocument(kbId, file, title, metadata),
    {
      onSuccess: (data, variables) => {
        toast.success(`Document "${variables.file.name}" uploaded successfully!`)
        queryClient.invalidateQueries('knowledge-bases')
        // Remove file from selected files
        setSelectedFiles(prev => prev.filter(f => f !== variables.file))
      },
      onError: (error: any, variables) => {
        toast.error(`Failed to upload "${variables.file.name}": ${error.message}`)
      }
    }
  )

  const resetForm = () => {
    setNewKBName('')
    setNewKBDescription('')
    setNewKBTags([])
    setNewKBIsPublic(false)
    setNewKBUseCase('general')
  }

  // Enhanced file upload functionality
  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setSelectedFiles(prev => [...prev, ...files])
  }, [])

  const removeFile = useCallback((index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }, [])

  const uploadFiles = async () => {
    if (!selectedKB) {
      toast.error('Please select a knowledge base first')
      return
    }

    if (selectedFiles.length === 0) {
      toast.error('Please select files to upload')
      return
    }

    setIsUploading(true)
    setUploadResults([])
    const results: any[] = []

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i]
        setUploadProgress(prev => ({ ...prev, [file.name]: 0 }))

        try {
          // Upload file to knowledge base
          const result = await ragApi.uploadFile(file, selectedKB.id)

          setUploadProgress(prev => ({ ...prev, [file.name]: 100 }))
          results.push({
            file: file.name,
            status: 'success',
            message: 'File uploaded and processed successfully',
            chunks: result.chunks_created || 0
          })
        } catch (error: any) {
          setUploadProgress(prev => ({ ...prev, [file.name]: -1 }))
          results.push({
            file: file.name,
            status: 'error',
            message: error.message || 'Upload failed'
          })
        }
      }

      setUploadResults(results)
      const successCount = results.filter(r => r.status === 'success').length
      const errorCount = results.filter(r => r.status === 'error').length

      if (successCount > 0) {
        toast.success(`${successCount} files uploaded successfully`)
      }
      if (errorCount > 0) {
        toast.error(`${errorCount} files failed to upload`)
      }

      // Refresh knowledge base data
      queryClient.invalidateQueries('knowledge-bases')

    } catch (error: any) {
      toast.error('Upload process failed: ' + error.message)
    } finally {
      setIsUploading(false)
    }
  }

  // New document upload functions
  const removeSelectedFile = (fileToRemove: File) => {
    setSelectedFiles(prev => prev.filter(f => f !== fileToRemove))
  }

  const uploadDocumentToKB = async (kbId: string, file: File) => {
    setUploadingFiles(prev => ({ ...prev, [`${kbId}-${file.name}`]: true }))

    try {
      await uploadDocumentMutation.mutateAsync({
        kbId,
        file,
        title: file.name,
        metadata: {
          uploaded_via: 'knowledge_base_management',
          file_size: file.size,
          file_type: file.type
        }
      })
    } finally {
      setUploadingFiles(prev => {
        const newState = { ...prev }
        delete newState[`${kbId}-${file.name}`]
        return newState
      })
    }
  }

  // Search modal functions
  const openSearchModal = (kb: KnowledgeBase) => {
    setSearchModalKB(kb)
    setShowSearchModal(true)
    setModalSearchQuery('')
    setModalSearchResults([])
  }

  const searchInModal = async () => {
    if (!modalSearchQuery.trim() || !searchModalKB) return

    setIsModalSearching(true)
    try {
      const result = await knowledgeBaseApi.searchKnowledgeBase(
        searchModalKB.id,
        modalSearchQuery.trim(),
        10
      )
      setModalSearchResults(result.results || [])
      toast.success(`Found ${result.results?.length || 0} results`)
    } catch (error: any) {
      toast.error(`Search failed: ${error.message}`)
      setModalSearchResults([])
    } finally {
      setIsModalSearching(false)
    }
  }

  // Enhanced search functionality
  const searchKnowledgeBase = async () => {
    if (!searchQuery.trim()) {
      toast.error('Please enter a search query')
      return
    }

    if (!searchKB) {
      toast.error('Please select a knowledge base to search')
      return
    }

    setIsSearching(true)
    try {
      const results = await knowledgeBaseApi.searchKnowledgeBase(searchKB, searchQuery.trim(), 10)
      setSearchResults(results.results || [])
      toast.success(`Found ${results.results?.length || 0} results`)
    } catch (error: any) {
      toast.error('Search failed: ' + error.message)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleCreateKB = async () => {
    if (!newKBName.trim()) {
      toast.error('Please enter a knowledge base name')
      return
    }

    await createKBMutation.mutateAsync({
      name: newKBName.trim(),
      description: newKBDescription.trim(),
      use_case: newKBUseCase,
      tags: newKBTags,
      is_public: newKBIsPublic
    })
  }

  const handleDeleteKB = async (kbId: string) => {
    if (window.confirm('Are you sure you want to delete this knowledge base? This action cannot be undone.')) {
      await deleteKBMutation.mutateAsync(kbId)
    }
  }

  const addTag = (tag: string) => {
    if (tag.trim() && !newKBTags.includes(tag.trim())) {
      setNewKBTags([...newKBTags, tag.trim()])
    }
  }

  const removeTag = (tagToRemove: string) => {
    setNewKBTags(newKBTags.filter(tag => tag !== tagToRemove))
  }

  const filteredKnowledgeBases = knowledgeBases.filter((kb: KnowledgeBase) => {
    const matchesSearch = !searchQuery ||
      kb.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      kb.description?.toLowerCase().includes(searchQuery.toLowerCase())

    return matchesSearch
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Revolutionary Knowledge Base Management</h1>
          <p className="text-muted-foreground mt-1">
            Create unlimited knowledge bases for any purpose - powered by ChromaDB and RAG
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn btn-primary flex items-center space-x-2"
        >
          <Plus className="h-4 w-4" />
          <span>Create Knowledge Base</span>
        </button>
      </div>

      {/* Enhanced Tabs */}
      <div className="flex space-x-1 bg-muted p-1 rounded-lg w-fit">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'browse', label: 'Browse', icon: Database },
          { id: 'upload', label: 'Upload Files', icon: Upload },
          { id: 'search', label: 'Search', icon: Search },
          { id: 'create', label: 'Create New', icon: Plus },
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

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="card p-4">
              <div className="flex items-center space-x-3">
                <Database className="h-8 w-8 text-blue-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Total Knowledge Bases</p>
                  <p className="text-2xl font-bold">{knowledgeBases.length}</p>
                </div>
              </div>
            </div>
            
            <div className="card p-4">
              <div className="flex items-center space-x-3">
                <FileText className="h-8 w-8 text-green-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Total Documents</p>
                  <p className="text-2xl font-bold">
                    {knowledgeBases.reduce((sum: number, kb: KnowledgeBase) => sum + kb.document_count, 0)}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="card p-4">
              <div className="flex items-center space-x-3">
                <BarChart3 className="h-8 w-8 text-purple-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Total Size</p>
                  <p className="text-2xl font-bold">
                    {knowledgeBases.reduce((sum: number, kb: KnowledgeBase) => sum + kb.size_mb, 0).toFixed(1)} MB
                  </p>
                </div>
              </div>
            </div>
            
            <div className="card p-4">
              <div className="flex items-center space-x-3">
                <Globe className="h-8 w-8 text-orange-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Public KBs</p>
                  <p className="text-2xl font-bold">
                    {knowledgeBases.filter((kb: KnowledgeBase) => kb.is_public).length}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">Revolutionary Knowledge Management</h3>
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Database className="h-5 w-5 text-blue-500" />
                <span>Unlimited knowledge base creation</span>
              </div>
              <div className="flex items-center space-x-3">
                <BookOpen className="h-5 w-5 text-green-500" />
                <span>ChromaDB vector storage for semantic search</span>
              </div>
              <div className="flex items-center space-x-3">
                <Upload className="h-5 w-5 text-purple-500" />
                <span>Multi-format document ingestion</span>
              </div>
              <div className="flex items-center space-x-3">
                <Search className="h-5 w-5 text-orange-500" />
                <span>AI-powered retrieval and embeddings</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Browse Tab */}
      {activeTab === 'browse' && (
        <div className="space-y-4">
          {/* Search */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search knowledge bases..."
                className="input"
              />
            </div>
          </div>

          {/* Knowledge Bases Grid */}
          {kbLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
              <p className="mt-2 text-muted-foreground">Loading knowledge bases...</p>
            </div>
          ) : filteredKnowledgeBases.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredKnowledgeBases.map((kb: KnowledgeBase) => (
                <div key={kb.id} className="card p-4 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Database className="h-5 w-5 text-blue-500" />
                      <h4 className="font-semibold">{kb.name}</h4>
                    </div>
                    <div className="flex items-center space-x-1">
                      {kb.is_public ? (
                        <Globe className="h-4 w-4 text-green-500" title="Public" />
                      ) : (
                        <Lock className="h-4 w-4 text-gray-500" title="Private" />
                      )}
                      <button
                        onClick={() => handleDeleteKB(kb.id)}
                        className="text-red-500 hover:text-red-700 p-1"
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  
                  {kb.description && (
                    <p className="text-sm text-muted-foreground mb-3">{kb.description}</p>
                  )}
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Documents:</span>
                      <span className="font-medium">{kb.document_count}</span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Size:</span>
                      <span className="font-medium">{kb.size_mb.toFixed(1)} MB</span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Embedding Model:</span>
                      <span className="font-medium text-xs">{kb.embedding_model}</span>
                    </div>
                  </div>
                  
                  {kb.tags.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1">
                      {kb.tags.map((tag, index) => (
                        <span key={index} className="px-2 py-1 bg-muted text-xs rounded">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                  
                  <div className="mt-4 flex space-x-2">
                    <button
                      onClick={() => navigate(`/knowledge-bases/${kb.id}`)}
                      className="btn btn-outline flex-1 text-sm"
                    >
                      <Upload className="h-4 w-4 mr-1" />
                      Manage KB
                    </button>
                    <button
                      onClick={() => openSearchModal(kb)}
                      className="btn btn-outline text-sm"
                    >
                      <Search className="h-4 w-4 mr-1" />
                      Search
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No Knowledge Bases Found</h3>
              <p className="text-muted-foreground mb-4">
                {searchQuery
                  ? 'No knowledge bases match your search.'
                  : 'Create your first revolutionary knowledge base to get started!'
                }
              </p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="btn btn-primary"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Knowledge Base
              </button>
            </div>
          )}
        </div>
      )}

      {/* Upload Files Tab */}
      {activeTab === 'upload' && (
        <div className="space-y-6">
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
              <Upload className="h-5 w-5" />
              <span>Upload Documents</span>
            </h2>

            {/* Knowledge Base Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-foreground mb-2">
                Select Knowledge Base
              </label>
              <select
                value={selectedKB?.id || ''}
                onChange={(e) => {
                  const kb = knowledgeBases.find(k => k.id === e.target.value)
                  setSelectedKB(kb || null)
                }}
                className="input w-full"
              >
                <option value="">Choose a knowledge base...</option>
                {knowledgeBases.map((kb) => (
                  <option key={kb.id} value={kb.id}>
                    {kb.name} ({kb.document_count} documents)
                  </option>
                ))}
              </select>
            </div>

            {/* File Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-foreground mb-2">
                Select Files
              </label>
              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                <input
                  type="file"
                  multiple
                  accept=".pdf,.txt,.md,.docx,.doc,.rtf"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium text-foreground mb-2">
                    Drop files here or click to browse
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Supports PDF, TXT, MD, DOCX, DOC, RTF files
                  </p>
                </label>
              </div>
            </div>

            {/* Selected Files */}
            {selectedFiles.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-foreground mb-3">
                  Selected Files ({selectedFiles.length})
                </h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div className="flex items-center space-x-3">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <p className="text-sm font-medium">{file.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {uploadProgress[file.name] !== undefined && (
                          <div className="flex items-center space-x-2">
                            {uploadProgress[file.name] === -1 ? (
                              <AlertCircle className="h-4 w-4 text-red-500" />
                            ) : uploadProgress[file.name] === 100 ? (
                              <CheckCircle className="h-4 w-4 text-green-500" />
                            ) : (
                              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                            )}
                          </div>
                        )}
                        <button
                          onClick={() => removeFile(index)}
                          className="text-red-500 hover:text-red-700"
                          disabled={isUploading}
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Button */}
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                {selectedFiles.length > 0 && selectedKB && (
                  <p>Ready to upload {selectedFiles.length} files to "{selectedKB.name}"</p>
                )}
              </div>
              <button
                onClick={uploadFiles}
                disabled={!selectedKB || selectedFiles.length === 0 || isUploading}
                className="btn-primary"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Files
                  </>
                )}
              </button>
            </div>

            {/* Upload Results */}
            {uploadResults.length > 0 && (
              <div className="mt-6 space-y-2">
                <h3 className="text-sm font-medium text-foreground">Upload Results</h3>
                {uploadResults.map((result, index) => (
                  <div key={index} className={`p-3 rounded-lg border ${
                    result.status === 'success'
                      ? 'bg-green-50 border-green-200 text-green-800 dark:bg-green-900/20 dark:border-green-800 dark:text-green-200'
                      : 'bg-red-50 border-red-200 text-red-800 dark:bg-red-900/20 dark:border-red-800 dark:text-red-200'
                  }`}>
                    <div className="flex items-center space-x-2">
                      {result.status === 'success' ? (
                        <CheckCircle className="h-4 w-4" />
                      ) : (
                        <AlertCircle className="h-4 w-4" />
                      )}
                      <span className="font-medium">{result.file}</span>
                    </div>
                    <p className="text-sm mt-1">{result.message}</p>
                    {result.chunks && (
                      <p className="text-xs mt-1">Created {result.chunks} text chunks</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Search Tab */}
      {activeTab === 'search' && (
        <div className="space-y-6">
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
              <Search className="h-5 w-5" />
              <span>Search Knowledge Bases</span>
            </h2>

            <div className="space-y-4">
              {/* Knowledge Base Selection */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Select Knowledge Base
                </label>
                <select
                  value={searchKB}
                  onChange={(e) => setSearchKB(e.target.value)}
                  className="input w-full"
                >
                  <option value="">Choose a knowledge base to search...</option>
                  {knowledgeBases.map((kb) => (
                    <option key={kb.id} value={kb.id}>
                      {kb.name} ({kb.document_count} documents)
                    </option>
                  ))}
                </select>
              </div>

              {/* Search Query */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Search Query
                </label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter your search query..."
                    className="input flex-1"
                    onKeyPress={(e) => e.key === 'Enter' && searchKnowledgeBase()}
                  />
                  <button
                    onClick={searchKnowledgeBase}
                    disabled={!searchKB || !searchQuery.trim() || isSearching}
                    className="btn-primary"
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
                  </button>
                </div>
              </div>
            </div>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-foreground mb-4">
                  Search Results ({searchResults.length})
                </h3>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {searchResults.map((result, index) => (
                    <div key={index} className="border border-border rounded-lg p-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium text-sm">
                            {result.metadata?.source || `Document ${index + 1}`}
                          </span>
                        </div>
                        <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded">
                          Score: {(result.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-sm text-foreground mb-2 leading-relaxed">
                        {result.content}
                      </p>
                      {result.metadata && (
                        <div className="text-xs text-muted-foreground">
                          {result.metadata.page && <span>Page {result.metadata.page} • </span>}
                          {result.metadata.chunk_id && <span>Chunk {result.metadata.chunk_id}</span>}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {searchResults.length === 0 && searchQuery && !isSearching && (
              <div className="mt-6 text-center py-8">
                <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No Results Found</h3>
                <p className="text-muted-foreground">
                  Try adjusting your search query or selecting a different knowledge base.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create New Tab */}
      {activeTab === 'create' && (
        <div className="space-y-6">
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
              <Plus className="h-5 w-5" />
              <span>Create New Knowledge Base</span>
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Name *
                </label>
                <input
                  type="text"
                  value={newKBName}
                  onChange={(e) => setNewKBName(e.target.value)}
                  placeholder="Enter knowledge base name"
                  className="input w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Description
                </label>
                <textarea
                  value={newKBDescription}
                  onChange={(e) => setNewKBDescription(e.target.value)}
                  placeholder="Describe the purpose of this knowledge base"
                  className="input w-full min-h-20 resize-none"
                  rows={3}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Use Case *
                </label>
                <select
                  value={newKBUseCase}
                  onChange={(e) => setNewKBUseCase(e.target.value)}
                  className="input w-full"
                >
                  <option value="general">General Purpose</option>
                  <option value="customer_support">Customer Support</option>
                  <option value="research">Research & Development</option>
                  <option value="legal">Legal Documents</option>
                  <option value="technical">Technical Documentation</option>
                  <option value="marketing">Marketing Content</option>
                  <option value="hr">Human Resources</option>
                  <option value="finance">Finance & Accounting</option>
                  <option value="education">Education & Training</option>
                  <option value="healthcare">Healthcare</option>
                </select>
              </div>



              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="is-public"
                  checked={newKBIsPublic}
                  onChange={(e) => setNewKBIsPublic(e.target.checked)}
                  className="rounded border-border"
                />
                <label htmlFor="is-public" className="text-sm text-foreground">
                  Make this knowledge base public (accessible to all agents)
                </label>
              </div>

              {/* Global Embedding Info */}
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                <div className="flex items-start space-x-3">
                  <Database className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100">
                      Global Embedding Configuration
                    </h4>
                    <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                      This knowledge base will use the global embedding model configured in Settings → Embedding & RAG.
                      All knowledge bases share the same embedding configuration for consistency.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-end space-x-3 pt-4">
                <button
                  onClick={resetForm}
                  className="btn-ghost"
                >
                  Reset
                </button>
                <button
                  onClick={handleCreateKB}
                  disabled={!newKBName.trim() || createKBMutation.isLoading}
                  className="btn-primary"
                >
                  {createKBMutation.isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Knowledge Base
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-background rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold mb-4">Create New Knowledge Base</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Name *</label>
                <input
                  type="text"
                  value={newKBName}
                  onChange={(e) => setNewKBName(e.target.value)}
                  placeholder="Enter knowledge base name"
                  className="input w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Description</label>
                <textarea
                  value={newKBDescription}
                  onChange={(e) => setNewKBDescription(e.target.value)}
                  placeholder="Describe the purpose of this knowledge base"
                  className="input w-full h-20 resize-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Use Case *</label>
                <select
                  value={newKBUseCase}
                  onChange={(e) => setNewKBUseCase(e.target.value)}
                  className="input w-full"
                >
                  <option value="general">General Purpose</option>
                  <option value="customer_support">Customer Support</option>
                  <option value="research">Research & Development</option>
                  <option value="legal">Legal Documents</option>
                  <option value="technical">Technical Documentation</option>
                  <option value="marketing">Marketing Content</option>
                  <option value="hr">Human Resources</option>
                  <option value="finance">Finance & Accounting</option>
                  <option value="education">Education & Training</option>
                  <option value="healthcare">Healthcare</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Tags</label>
                <div className="flex flex-wrap gap-1 mb-2">
                  {newKBTags.map((tag, index) => (
                    <span key={index} className="px-2 py-1 bg-muted text-xs rounded flex items-center">
                      {tag}
                      <button
                        onClick={() => removeTag(tag)}
                        className="ml-1 text-red-500 hover:text-red-700"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
                <input
                  type="text"
                  placeholder="Add tags (press Enter)"
                  className="input w-full"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      addTag(e.currentTarget.value)
                      e.currentTarget.value = ''
                    }
                  }}
                />
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="isPublic"
                  checked={newKBIsPublic}
                  onChange={(e) => setNewKBIsPublic(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="isPublic" className="text-sm">
                  Make this knowledge base public
                </label>
              </div>
            </div>

            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => {
                  setShowCreateModal(false)
                  resetForm()
                }}
                className="btn btn-outline flex-1"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateKB}
                disabled={createKBMutation.isLoading}
                className="btn btn-primary flex-1"
              >
                {createKBMutation.isLoading ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Document Upload Modal */}
      {selectedKB && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-background rounded-lg p-6 w-full max-w-lg mx-4">
            <h3 className="text-lg font-semibold mb-4">
              Upload Documents to "{selectedKB.name}"
            </h3>

            <div className="border-2 border-dashed border-muted rounded-lg p-8 text-center">
              <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground mb-4">
                Drag and drop files here, or click to select files
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.pdf,.doc,.docx,.md,.json,.csv"
                className="hidden"
                onChange={handleFileSelect}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="btn btn-outline cursor-pointer"
              >
                Select Files
              </button>
            </div>

            {/* Selected Files */}
            {selectedFiles.length > 0 && (
              <div className="mt-4">
                <h4 className="font-medium mb-2">Selected Files ({selectedFiles.length})</h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4 text-blue-500" />
                        <span className="text-sm">{file.name}</span>
                        <span className="text-xs text-muted-foreground">
                          ({(file.size / 1024).toFixed(1)} KB)
                        </span>
                      </div>
                      <button
                        onClick={() => removeSelectedFile(file)}
                        className="text-red-500 hover:text-red-700"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => {
                  setSelectedKB(null)
                  setSelectedFiles([])
                }}
                className="btn btn-outline flex-1"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  if (selectedFiles.length === 0) {
                    toast.error('Please select files to upload')
                    return
                  }

                  // Upload all selected files
                  for (const file of selectedFiles) {
                    await uploadDocumentToKB(selectedKB!.id, file)
                  }

                  // Close modal and clear files
                  setSelectedKB(null)
                  setSelectedFiles([])
                }}
                disabled={selectedFiles.length === 0 || uploadDocumentMutation.isLoading}
                className="btn btn-primary flex-1 disabled:opacity-50"
              >
                {uploadDocumentMutation.isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload {selectedFiles.length > 0 ? `(${selectedFiles.length})` : ''}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Search Modal */}
      {showSearchModal && searchModalKB && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-background rounded-lg p-6 w-full max-w-2xl mx-4 max-h-[80vh] overflow-hidden flex flex-col">
            <h3 className="text-lg font-semibold mb-4">
              Search in "{searchModalKB.name}"
            </h3>

            <div className="flex space-x-2 mb-4">
              <input
                type="text"
                value={modalSearchQuery}
                onChange={(e) => setModalSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && searchInModal()}
                placeholder="Enter your search query..."
                className="input flex-1"
              />
              <button
                onClick={searchInModal}
                disabled={!modalSearchQuery.trim() || isModalSearching}
                className="btn btn-primary disabled:opacity-50"
              >
                {isModalSearching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Search className="h-4 w-4" />
                )}
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              {modalSearchResults.length > 0 ? (
                <div className="space-y-3">
                  {modalSearchResults.map((result, index) => (
                    <div key={index} className="p-3 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-blue-600">
                          Score: {(result.score * 100).toFixed(1)}%
                        </span>
                        {result.metadata?.title && (
                          <span className="text-sm text-muted-foreground">
                            {result.metadata.title}
                          </span>
                        )}
                      </div>
                      <p className="text-sm">{result.content}</p>
                      {result.metadata && Object.keys(result.metadata).length > 0 && (
                        <div className="mt-2 text-xs text-muted-foreground">
                          {Object.entries(result.metadata)
                            .filter(([key]) => key !== 'title')
                            .map(([key, value]) => (
                              <span key={key} className="mr-3">
                                {key}: {String(value)}
                              </span>
                            ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : modalSearchQuery && !isModalSearching ? (
                <div className="text-center py-8">
                  <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">No results found for "{modalSearchQuery}"</p>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">Enter a search query to find relevant documents</p>
                </div>
              )}
            </div>

            <div className="flex justify-end mt-4">
              <button
                onClick={() => {
                  setShowSearchModal(false)
                  setSearchModalKB(null)
                  setModalSearchQuery('')
                  setModalSearchResults([])
                }}
                className="btn btn-outline"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default KnowledgeBaseManagement
