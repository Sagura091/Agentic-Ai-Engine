import React, { useState, useRef, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import {
  ArrowLeft,
  Upload,
  FileText,
  Image,
  File,
  Search,
  Eye,
  Download,
  Trash2,
  Brain,
  Zap,
  Database,
  Settings,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Loader2,
  X,
  Plus,
  Grid,
  List,
  Filter,
  MoreVertical,
  Clock,
  User,
  Tag,
  Hash
} from 'lucide-react'
import toast from 'react-hot-toast'
import { knowledgeBaseApi } from '../services/api'

interface Document {
  id: string
  title: string
  filename: string
  content_type: string
  size: number
  uploaded_at: string
  uploaded_by: string
  status: 'processing' | 'completed' | 'failed'
  chunk_count?: number
  embedding_model?: string
  metadata?: any
}

interface DocumentChunk {
  id: string
  content: string
  embedding_vector: number[]
  metadata: any
  score?: number
}

const KnowledgeBaseDetail: React.FC = () => {
  const { kbId } = useParams<{ kbId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)

  // State management
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploadProgress, setUploadProgress] = useState<{[key: string]: number}>({})
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [showDocumentViewer, setShowDocumentViewer] = useState(false)
  const [documentChunks, setDocumentChunks] = useState<DocumentChunk[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<string>('all')

  // Fetch knowledge base details
  const { data: knowledgeBase, isLoading: kbLoading } = useQuery(
    ['knowledge-base', kbId],
    () => knowledgeBaseApi.getKnowledgeBase(kbId!),
    { enabled: !!kbId }
  )

  // Fetch documents in this knowledge base
  const { data: documents = [], isLoading: docsLoading, refetch: refetchDocuments } = useQuery(
    ['knowledge-base-documents', kbId],
    () => knowledgeBaseApi.getDocuments(kbId!),
    { enabled: !!kbId }
  )

  // Upload document mutation
  const uploadMutation = useMutation(
    ({ file, title, metadata }: { file: File; title?: string; metadata?: any }) =>
      knowledgeBaseApi.uploadDocument(kbId!, file, title, metadata),
    {
      onSuccess: (data, variables) => {
        toast.success(`Document "${variables.file.name}" uploaded successfully!`)
        setSelectedFiles(prev => prev.filter(f => f !== variables.file))
        refetchDocuments()
      },
      onError: (error: any, variables) => {
        toast.error(`Failed to upload "${variables.file.name}": ${error.message}`)
      }
    }
  )

  // Delete document mutation
  const deleteMutation = useMutation(
    (docId: string) => knowledgeBaseApi.deleteDocument(kbId!, docId),
    {
      onSuccess: () => {
        toast.success('Document deleted successfully!')
        refetchDocuments()
      },
      onError: (error: any) => {
        toast.error(`Failed to delete document: ${error.message}`)
      }
    }
  )

  // File handling
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setSelectedFiles(prev => [...prev, ...files])
  }

  const removeSelectedFile = (fileToRemove: File) => {
    setSelectedFiles(prev => prev.filter(f => f !== fileToRemove))
  }

  const uploadAllFiles = async () => {
    if (selectedFiles.length === 0) {
      toast.error('Please select files to upload')
      return
    }

    for (const file of selectedFiles) {
      await uploadMutation.mutateAsync({
        file,
        title: file.name,
        metadata: {
          uploaded_via: 'knowledge_base_detail',
          file_size: file.size,
          file_type: file.type
        }
      })
    }
  }

  // Document viewing
  const viewDocument = async (doc: Document) => {
    setSelectedDocument(doc)
    setShowDocumentViewer(true)
    
    try {
      // Fetch document chunks to show how it was embedded
      const chunks = await knowledgeBaseApi.getDocumentChunks(kbId!, doc.id)
      setDocumentChunks(chunks)
    } catch (error: any) {
      toast.error(`Failed to load document chunks: ${error.message}`)
    }
  }

  // Filter documents
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesFilter = filterType === 'all' || doc.content_type.includes(filterType)
    return matchesSearch && matchesFilter
  })

  // Get file icon
  const getFileIcon = (contentType: string) => {
    if (contentType.includes('image')) return <Image className="h-5 w-5 text-blue-500" />
    if (contentType.includes('pdf')) return <FileText className="h-5 w-5 text-red-500" />
    if (contentType.includes('text')) return <FileText className="h-5 w-5 text-green-500" />
    return <File className="h-5 w-5 text-gray-500" />
  }

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  if (kbLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (!knowledgeBase) {
    return (
      <div className="text-center py-12">
        <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-medium mb-2">Knowledge Base Not Found</h3>
        <p className="text-muted-foreground mb-4">
          The requested knowledge base could not be found.
        </p>
        <button
          onClick={() => navigate('/knowledge-bases')}
          className="btn btn-primary"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Knowledge Bases
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/knowledge-bases')}
            className="btn btn-outline"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </button>
          <div>
            <h1 className="text-2xl font-bold flex items-center space-x-2">
              <Database className="h-6 w-6 text-blue-500" />
              <span>{knowledgeBase.name}</span>
            </h1>
            <p className="text-muted-foreground">{knowledgeBase.description}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn btn-primary"
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Documents
          </button>
          <button className="btn btn-outline">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </button>
        </div>
      </div>

      {/* Knowledge Base Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center space-x-2">
            <FileText className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm text-muted-foreground">Documents</p>
              <p className="text-xl font-semibold">{documents.length}</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-purple-500" />
            <div>
              <p className="text-sm text-muted-foreground">Total Chunks</p>
              <p className="text-xl font-semibold">
                {documents.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0)}
              </p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-yellow-500" />
            <div>
              <p className="text-sm text-muted-foreground">Embedding Model</p>
              <p className="text-sm font-medium">{knowledgeBase.embedding_model || 'Global'}</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center space-x-2">
            <Database className="h-5 w-5 text-green-500" />
            <div>
              <p className="text-sm text-muted-foreground">Total Size</p>
              <p className="text-xl font-semibold">
                {formatFileSize(documents.reduce((sum, doc) => sum + doc.size, 0))}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* File Upload Area */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4">📄 Revolutionary Document Ingestion</h3>
        
        <div className="border-2 border-dashed border-muted rounded-lg p-8 text-center mb-4">
          <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground mb-4">
            Drag and drop files here, or click to select files
          </p>
          <p className="text-sm text-muted-foreground mb-4">
            Supports: PDF, DOCX, TXT, MD, Images, and more
          </p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.pdf,.doc,.docx,.md,.json,.csv,.png,.jpg,.jpeg,.gif"
            className="hidden"
            onChange={handleFileSelect}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn btn-outline"
          >
            <Plus className="h-4 w-4 mr-2" />
            Select Files
          </button>
        </div>

        {/* Selected Files */}
        {selectedFiles.length > 0 && (
          <div className="space-y-2 mb-4">
            <h4 className="font-medium">Selected Files ({selectedFiles.length})</h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {selectedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                  <div className="flex items-center space-x-2">
                    {getFileIcon(file.type)}
                    <span className="text-sm">{file.name}</span>
                    <span className="text-xs text-muted-foreground">
                      ({formatFileSize(file.size)})
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
            <button
              onClick={uploadAllFiles}
              disabled={uploadMutation.isLoading}
              className="btn btn-primary w-full"
            >
              {uploadMutation.isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Processing & Embedding...
                </>
              ) : (
                <>
                  <Brain className="h-4 w-4 mr-2" />
                  Upload & Process ({selectedFiles.length} files)
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Documents Section */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold">📚 Documents & Embeddings</h3>
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-2">
              <Search className="h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input w-64"
              />
            </div>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="input w-32"
            >
              <option value="all">All Types</option>
              <option value="pdf">PDF</option>
              <option value="text">Text</option>
              <option value="image">Images</option>
            </select>
            <div className="flex border rounded">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 ${viewMode === 'grid' ? 'bg-primary text-primary-foreground' : ''}`}
              >
                <Grid className="h-4 w-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 ${viewMode === 'list' ? 'bg-primary text-primary-foreground' : ''}`}
              >
                <List className="h-4 w-4" />
              </button>
            </div>
            <button
              onClick={() => refetchDocuments()}
              className="btn btn-outline"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </div>

        {docsLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : filteredDocuments.length > 0 ? (
          <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' : 'space-y-2'}>
            {filteredDocuments.map((doc) => (
              <div
                key={doc.id}
                className={`border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
                  viewMode === 'list' ? 'flex items-center justify-between' : ''
                }`}
                onClick={() => viewDocument(doc)}
              >
                <div className={`flex items-start space-x-3 ${viewMode === 'list' ? 'flex-1' : ''}`}>
                  {getFileIcon(doc.content_type)}
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium truncate">{doc.title}</h4>
                    <p className="text-sm text-muted-foreground truncate">{doc.filename}</p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                      <span className="flex items-center space-x-1">
                        <Clock className="h-3 w-3" />
                        <span>{new Date(doc.uploaded_at).toLocaleDateString()}</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <Hash className="h-3 w-3" />
                        <span>{doc.chunk_count || 0} chunks</span>
                      </span>
                      <span>{formatFileSize(doc.size)}</span>
                    </div>
                  </div>
                </div>

                <div className={`flex items-center space-x-2 ${viewMode === 'list' ? '' : 'mt-3'}`}>
                  <div className="flex items-center space-x-1">
                    {doc.status === 'completed' && <CheckCircle className="h-4 w-4 text-green-500" />}
                    {doc.status === 'processing' && <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />}
                    {doc.status === 'failed' && <AlertCircle className="h-4 w-4 text-red-500" />}
                    <span className="text-xs capitalize">{doc.status}</span>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      viewDocument(doc)
                    }}
                    className="btn btn-outline btn-sm"
                  >
                    <Eye className="h-3 w-3 mr-1" />
                    View
                  </button>

                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      if (confirm('Are you sure you want to delete this document?')) {
                        deleteMutation.mutate(doc.id)
                      }
                    }}
                    className="btn btn-outline btn-sm text-red-500 hover:text-red-700"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No Documents Found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery || filterType !== 'all'
                ? 'No documents match your search criteria.'
                : 'Upload your first document to get started with this knowledge base!'
              }
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn btn-primary"
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload Documents
            </button>
          </div>
        )}
      </div>

      {/* Document Viewer Modal */}
      {showDocumentViewer && selectedDocument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-background rounded-lg w-full max-w-6xl mx-4 max-h-[90vh] overflow-hidden flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                {getFileIcon(selectedDocument.content_type)}
                <div>
                  <h3 className="text-lg font-semibold">{selectedDocument.title}</h3>
                  <p className="text-sm text-muted-foreground">{selectedDocument.filename}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <button className="btn btn-outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </button>
                <button
                  onClick={() => {
                    setShowDocumentViewer(false)
                    setSelectedDocument(null)
                    setDocumentChunks([])
                  }}
                  className="btn btn-outline"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-hidden flex">
              {/* Document Info Sidebar */}
              <div className="w-80 border-r p-6 overflow-y-auto">
                <h4 className="font-semibold mb-4">📊 Document Analytics</h4>

                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Status</label>
                    <div className="flex items-center space-x-2 mt-1">
                      {selectedDocument.status === 'completed' && <CheckCircle className="h-4 w-4 text-green-500" />}
                      {selectedDocument.status === 'processing' && <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />}
                      {selectedDocument.status === 'failed' && <AlertCircle className="h-4 w-4 text-red-500" />}
                      <span className="capitalize">{selectedDocument.status}</span>
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground">File Size</label>
                    <p className="mt-1">{formatFileSize(selectedDocument.size)}</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Content Type</label>
                    <p className="mt-1">{selectedDocument.content_type}</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Uploaded</label>
                    <p className="mt-1">{new Date(selectedDocument.uploaded_at).toLocaleString()}</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Uploaded By</label>
                    <p className="mt-1">{selectedDocument.uploaded_by}</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Embedding Model</label>
                    <p className="mt-1">{selectedDocument.embedding_model || 'Global Model'}</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Total Chunks</label>
                    <div className="flex items-center space-x-2 mt-1">
                      <Brain className="h-4 w-4 text-purple-500" />
                      <span className="font-semibold">{selectedDocument.chunk_count || 0}</span>
                    </div>
                  </div>

                  {selectedDocument.metadata && (
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Metadata</label>
                      <div className="mt-1 text-xs bg-muted p-2 rounded">
                        <pre>{JSON.stringify(selectedDocument.metadata, null, 2)}</pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Document Chunks */}
              <div className="flex-1 p-6 overflow-y-auto">
                <h4 className="font-semibold mb-4">🧠 Embedding Chunks & Vectors</h4>

                {documentChunks.length > 0 ? (
                  <div className="space-y-4">
                    {documentChunks.map((chunk, index) => (
                      <div key={chunk.id} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-blue-600">
                            Chunk #{index + 1}
                          </span>
                          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                            <span>Vector Dim: {chunk.embedding_vector?.length || 0}</span>
                            {chunk.score && (
                              <span className="bg-green-100 text-green-800 px-2 py-1 rounded">
                                Score: {(chunk.score * 100).toFixed(1)}%
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="bg-muted p-3 rounded mb-3">
                          <p className="text-sm">{chunk.content}</p>
                        </div>

                        {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                          <div className="text-xs">
                            <span className="font-medium">Metadata: </span>
                            {Object.entries(chunk.metadata).map(([key, value]) => (
                              <span key={key} className="mr-3">
                                {key}: {String(value)}
                              </span>
                            ))}
                          </div>
                        )}

                        {chunk.embedding_vector && chunk.embedding_vector.length > 0 && (
                          <details className="mt-2">
                            <summary className="text-xs text-muted-foreground cursor-pointer">
                              View Embedding Vector ({chunk.embedding_vector.length} dimensions)
                            </summary>
                            <div className="mt-2 text-xs bg-gray-50 p-2 rounded max-h-32 overflow-y-auto">
                              <code>
                                [{chunk.embedding_vector.slice(0, 10).map(v => v.toFixed(4)).join(', ')}
                                {chunk.embedding_vector.length > 10 ? ', ...' : ''}]
                              </code>
                            </div>
                          </details>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-medium mb-2">No Chunks Available</h3>
                    <p className="text-muted-foreground">
                      {selectedDocument.status === 'processing'
                        ? 'Document is still being processed and embedded...'
                        : 'This document has not been processed yet or failed to process.'
                      }
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default KnowledgeBaseDetail
