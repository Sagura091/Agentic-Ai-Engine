import React, { useState, useCallback, useRef } from 'react';
import { 
  Upload, 
  Image, 
  Music, 
  Video, 
  FileText, 
  X, 
  Check, 
  AlertCircle,
  Eye,
  Play,
  Pause,
  Download,
  Trash2,
  Search,
  Filter,
  Grid,
  List,
  Zap,
  Brain,
  Layers
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';

// Types for multi-modal content
interface MultiModalFile {
  id: string;
  name: string;
  type: 'image' | 'audio' | 'video' | 'document' | 'text';
  size: number;
  file: File;
  uploadProgress: number;
  processingStatus: 'pending' | 'uploading' | 'processing' | 'completed' | 'failed';
  extractedFeatures?: Record<string, any>;
  textDescription?: string;
  embeddings?: number[];
  thumbnail?: string;
  preview?: string;
}

interface CrossModalSearchResult {
  id: string;
  type: string;
  name: string;
  similarity: number;
  description: string;
  thumbnail?: string;
  crossModalRelevance: number;
}

const MultiModalUploader: React.FC = () => {
  // File management state
  const [files, setFiles] = useState<MultiModalFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  
  // UI state
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filterType, setFilterType] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Cross-modal search
  const [searchResults, setSearchResults] = useState<CrossModalSearchResult[]>([]);
  const [searchMode, setSearchMode] = useState<'text' | 'image' | 'audio' | 'video'>('text');
  const [crossModalQuery, setCrossModalQuery] = useState('');
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  // File type configurations
  const fileTypeConfig = {
    image: {
      icon: <Image className="h-5 w-5" />,
      color: 'text-blue-500',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      accept: 'image/*',
      maxSize: 10 * 1024 * 1024, // 10MB
      label: 'Images'
    },
    audio: {
      icon: <Music className="h-5 w-5" />,
      color: 'text-green-500',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      accept: 'audio/*',
      maxSize: 50 * 1024 * 1024, // 50MB
      label: 'Audio'
    },
    video: {
      icon: <Video className="h-5 w-5" />,
      color: 'text-purple-500',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      accept: 'video/*',
      maxSize: 100 * 1024 * 1024, // 100MB
      label: 'Videos'
    },
    document: {
      icon: <FileText className="h-5 w-5" />,
      color: 'text-orange-500',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200',
      accept: '.pdf,.doc,.docx,.txt,.md',
      maxSize: 25 * 1024 * 1024, // 25MB
      label: 'Documents'
    },
    text: {
      icon: <FileText className="h-5 w-5" />,
      color: 'text-gray-500',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      accept: '.txt,.md,.json,.csv',
      maxSize: 5 * 1024 * 1024, // 5MB
      label: 'Text Files'
    }
  };

  // Determine file type from file
  const getFileType = (file: File): MultiModalFile['type'] => {
    if (file.type.startsWith('image/')) return 'image';
    if (file.type.startsWith('audio/')) return 'audio';
    if (file.type.startsWith('video/')) return 'video';
    if (file.type === 'application/pdf' || file.name.endsWith('.doc') || file.name.endsWith('.docx')) return 'document';
    return 'text';
  };

  // Handle file selection
  const handleFileSelect = useCallback((selectedFiles: FileList | null) => {
    if (!selectedFiles) return;

    const newFiles: MultiModalFile[] = [];
    
    Array.from(selectedFiles).forEach((file) => {
      const fileType = getFileType(file);
      const config = fileTypeConfig[fileType];
      
      // Validate file size
      if (file.size > config.maxSize) {
        toast.error(`File ${file.name} is too large. Maximum size: ${config.maxSize / (1024 * 1024)}MB`);
        return;
      }
      
      const multiModalFile: MultiModalFile = {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: file.name,
        type: fileType,
        size: file.size,
        file,
        uploadProgress: 0,
        processingStatus: 'pending'
      };
      
      newFiles.push(multiModalFile);
    });
    
    setFiles(prev => [...prev, ...newFiles]);
    
    // Start processing files
    newFiles.forEach(processFile);
    
    toast.success(`Added ${newFiles.length} file(s) for processing`);
  }, []);

  // Process individual file
  const processFile = async (file: MultiModalFile) => {
    try {
      // Update status to uploading
      updateFileStatus(file.id, { processingStatus: 'uploading' });
      
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise(resolve => setTimeout(resolve, 100));
        updateFileStatus(file.id, { uploadProgress: progress });
      }
      
      // Update status to processing
      updateFileStatus(file.id, { processingStatus: 'processing' });
      
      // Generate thumbnail/preview
      const preview = await generatePreview(file);
      updateFileStatus(file.id, { thumbnail: preview });
      
      // Simulate feature extraction
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock extracted features
      const extractedFeatures = await extractFeatures(file);
      const textDescription = await generateTextDescription(file, extractedFeatures);
      
      // Update file with results
      updateFileStatus(file.id, {
        processingStatus: 'completed',
        extractedFeatures,
        textDescription,
        embeddings: Array.from({ length: 384 }, () => Math.random()) // Mock embeddings
      });
      
      toast.success(`Processed ${file.name} successfully`);
      
    } catch (error) {
      console.error('File processing failed:', error);
      updateFileStatus(file.id, { processingStatus: 'failed' });
      toast.error(`Failed to process ${file.name}`);
    }
  };

  // Update file status
  const updateFileStatus = (fileId: string, updates: Partial<MultiModalFile>) => {
    setFiles(prev => prev.map(file => 
      file.id === fileId ? { ...file, ...updates } : file
    ));
  };

  // Generate preview for file
  const generatePreview = async (file: MultiModalFile): Promise<string> => {
    if (file.type === 'image') {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.readAsDataURL(file.file);
      });
    }
    
    // For other types, return placeholder
    return '';
  };

  // Extract features from file
  const extractFeatures = async (file: MultiModalFile): Promise<Record<string, any>> => {
    // Mock feature extraction based on file type
    const baseFeatures = {
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type,
      processedAt: new Date().toISOString()
    };
    
    switch (file.type) {
      case 'image':
        return {
          ...baseFeatures,
          dimensions: { width: 1920, height: 1080 },
          colorProfile: 'RGB',
          dominantColors: ['#FF5733', '#33FF57', '#3357FF'],
          objects: ['person', 'building', 'sky'],
          scenes: ['outdoor', 'urban']
        };
      
      case 'audio':
        return {
          ...baseFeatures,
          duration: 180.5,
          sampleRate: 44100,
          channels: 2,
          audioType: 'speech',
          transcript: 'This is a sample audio transcript...'
        };
      
      case 'video':
        return {
          ...baseFeatures,
          duration: 300.0,
          resolution: '1920x1080',
          fps: 30,
          hasAudio: true,
          keyFrames: 15,
          scenes: ['indoor', 'presentation']
        };
      
      default:
        return baseFeatures;
    }
  };

  // Generate text description
  const generateTextDescription = async (file: MultiModalFile, features: Record<string, any>): Promise<string> => {
    switch (file.type) {
      case 'image':
        return `Image showing ${features.objects?.join(', ')} in ${features.scenes?.join(', ')} setting with dominant colors ${features.dominantColors?.slice(0, 2).join(', ')}`;
      
      case 'audio':
        return `${features.audioType} audio (${features.duration}s): ${features.transcript}`;
      
      case 'video':
        return `Video content (${features.duration}s) at ${features.resolution} resolution showing ${features.scenes?.join(', ')} scenes`;
      
      default:
        return `${file.type} file: ${file.name}`;
    }
  };

  // Cross-modal search
  const performCrossModalSearch = async () => {
    if (!crossModalQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Simulate cross-modal search
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock search results
      const mockResults: CrossModalSearchResult[] = files
        .filter(file => file.processingStatus === 'completed')
        .map(file => ({
          id: file.id,
          type: file.type,
          name: file.name,
          similarity: Math.random() * 0.4 + 0.6, // 0.6-1.0
          description: file.textDescription || '',
          thumbnail: file.thumbnail,
          crossModalRelevance: Math.random() * 0.3 + 0.7 // 0.7-1.0
        }))
        .sort((a, b) => b.crossModalRelevance - a.crossModalRelevance);
      
      setSearchResults(mockResults);
      toast.success(`Found ${mockResults.length} cross-modal results`);
      
    } catch (error) {
      console.error('Cross-modal search failed:', error);
      toast.error('Cross-modal search failed');
    } finally {
      setIsProcessing(false);
    }
  };

  // Drag and drop handlers
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files);
    }
  }, [handleFileSelect]);

  // Filter files
  const filteredFiles = files.filter(file => {
    const typeMatch = filterType === 'all' || file.type === filterType;
    const searchMatch = !searchQuery || 
      file.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (file.textDescription && file.textDescription.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return typeMatch && searchMatch;
  });

  // File selection handlers
  const toggleFileSelection = (fileId: string) => {
    setSelectedFiles(prev => {
      const newSet = new Set(prev);
      if (newSet.has(fileId)) {
        newSet.delete(fileId);
      } else {
        newSet.add(fileId);
      }
      return newSet;
    });
  };

  const selectAllFiles = () => {
    setSelectedFiles(new Set(filteredFiles.map(f => f.id)));
  };

  const clearSelection = () => {
    setSelectedFiles(new Set());
  };

  // Delete selected files
  const deleteSelectedFiles = () => {
    setFiles(prev => prev.filter(file => !selectedFiles.has(file.id)));
    setSelectedFiles(new Set());
    toast.success('Selected files deleted');
  };

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Layers className="h-6 w-6 text-indigo-500" />
              <div>
                <CardTitle className="text-xl">Multi-Modal Content Manager</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Upload, process, and search across images, audio, video, and documents
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <Brain className="h-3 w-3" />
                {files.filter(f => f.processingStatus === 'completed').length} processed
              </Badge>
              <Badge variant="outline" className="flex items-center gap-1">
                <Zap className="h-3 w-3" />
                {files.filter(f => f.processingStatus === 'processing').length} processing
              </Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="upload">Upload & Process</TabsTrigger>
          <TabsTrigger value="search">Cross-Modal Search</TabsTrigger>
          <TabsTrigger value="manage">Manage Content</TabsTrigger>
        </TabsList>

        {/* Upload Tab */}
        <TabsContent value="upload" className="space-y-4">
          {/* Upload Zone */}
          <Card>
            <CardContent className="p-6">
              <div
                ref={dropZoneRef}
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive 
                    ? 'border-indigo-500 bg-indigo-50' 
                    : 'border-gray-300 hover:border-gray-400'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">Upload Multi-Modal Content</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Drag and drop files here, or click to select files
                </p>
                
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-4">
                  {Object.entries(fileTypeConfig).map(([type, config]) => (
                    <div key={type} className={`p-2 rounded border ${config.bgColor} ${config.borderColor}`}>
                      <div className={`flex items-center gap-1 ${config.color}`}>
                        {config.icon}
                        <span className="text-xs font-medium">{config.label}</span>
                      </div>
                    </div>
                  ))}
                </div>
                
                <Button 
                  onClick={() => fileInputRef.current?.click()}
                  className="mb-2"
                >
                  Select Files
                </Button>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  accept="image/*,audio/*,video/*,.pdf,.doc,.docx,.txt,.md"
                  onChange={(e) => handleFileSelect(e.target.files)}
                />
                
                <p className="text-xs text-muted-foreground">
                  Supports images, audio, video, and documents up to 100MB
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Processing Queue */}
          {files.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Processing Queue</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {files.slice(-5).map((file) => (
                    <div key={file.id} className="flex items-center gap-3 p-3 border rounded">
                      <div className={`p-2 rounded ${fileTypeConfig[file.type].bgColor}`}>
                        {fileTypeConfig[file.type].icon}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium truncate">{file.name}</span>
                          <Badge 
                            variant={file.processingStatus === 'completed' ? 'default' : 'secondary'}
                            className="text-xs"
                          >
                            {file.processingStatus === 'completed' && <Check className="h-3 w-3 mr-1" />}
                            {file.processingStatus === 'failed' && <AlertCircle className="h-3 w-3 mr-1" />}
                            {file.processingStatus}
                          </Badge>
                        </div>
                        
                        {file.processingStatus === 'uploading' && (
                          <Progress value={file.uploadProgress} className="h-2" />
                        )}
                        
                        {file.processingStatus === 'processing' && (
                          <div className="flex items-center gap-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-500"></div>
                            <span className="text-xs text-muted-foreground">Extracting features...</span>
                          </div>
                        )}
                        
                        {file.textDescription && (
                          <p className="text-xs text-muted-foreground mt-1 truncate">
                            {file.textDescription}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Cross-Modal Search Tab */}
        <TabsContent value="search" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Cross-Modal Search</CardTitle>
              <p className="text-sm text-muted-foreground">
                Search across different content types using text, image, or other modalities
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Select value={searchMode} onValueChange={(value: any) => setSearchMode(value)}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="text">Text Query</SelectItem>
                    <SelectItem value="image">Image Query</SelectItem>
                    <SelectItem value="audio">Audio Query</SelectItem>
                    <SelectItem value="video">Video Query</SelectItem>
                  </SelectContent>
                </Select>
                
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Enter your cross-modal search query..."
                    value={crossModalQuery}
                    onChange={(e) => setCrossModalQuery(e.target.value)}
                    className="pl-10"
                    onKeyPress={(e) => e.key === 'Enter' && performCrossModalSearch()}
                  />
                </div>
                
                <Button 
                  onClick={performCrossModalSearch}
                  disabled={isProcessing || !crossModalQuery.trim()}
                >
                  {isProcessing ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <Search className="h-4 w-4" />
                  )}
                </Button>
              </div>
              
              {searchResults.length > 0 && (
                <div className="space-y-3">
                  <h3 className="font-medium">Search Results ({searchResults.length})</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {searchResults.map((result) => (
                      <div key={result.id} className="border rounded p-3">
                        <div className="flex items-center gap-2 mb-2">
                          {fileTypeConfig[result.type as keyof typeof fileTypeConfig].icon}
                          <span className="text-sm font-medium truncate">{result.name}</span>
                        </div>
                        
                        {result.thumbnail && (
                          <img 
                            src={result.thumbnail} 
                            alt={result.name}
                            className="w-full h-24 object-cover rounded mb-2"
                          />
                        )}
                        
                        <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
                          {result.description}
                        </p>
                        
                        <div className="flex justify-between text-xs">
                          <span>Similarity: {(result.similarity * 100).toFixed(1)}%</span>
                          <span>Relevance: {(result.crossModalRelevance * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Manage Content Tab */}
        <TabsContent value="manage" className="space-y-4">
          {/* Controls */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search files..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 w-64"
                    />
                  </div>
                  
                  <Select value={filterType} onValueChange={setFilterType}>
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Types</SelectItem>
                      {Object.entries(fileTypeConfig).map(([type, config]) => (
                        <SelectItem key={type} value={type}>{config.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex items-center gap-2">
                  {selectedFiles.size > 0 && (
                    <>
                      <Badge variant="outline">{selectedFiles.size} selected</Badge>
                      <Button variant="outline" size="sm" onClick={deleteSelectedFiles}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                      <Button variant="outline" size="sm" onClick={clearSelection}>
                        Clear
                      </Button>
                    </>
                  )}
                  
                  <Button variant="outline" size="sm" onClick={selectAllFiles}>
                    Select All
                  </Button>
                  
                  <div className="flex border rounded">
                    <Button
                      variant={viewMode === 'grid' ? 'default' : 'ghost'}
                      size="sm"
                      onClick={() => setViewMode('grid')}
                    >
                      <Grid className="h-4 w-4" />
                    </Button>
                    <Button
                      variant={viewMode === 'list' ? 'default' : 'ghost'}
                      size="sm"
                      onClick={() => setViewMode('list')}
                    >
                      <List className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* File Grid/List */}
          <Card>
            <CardContent className="p-4">
              {filteredFiles.length === 0 ? (
                <div className="text-center py-8">
                  <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-muted-foreground">No files uploaded yet</p>
                </div>
              ) : (
                <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4' : 'space-y-2'}>
                  {filteredFiles.map((file) => (
                    <div
                      key={file.id}
                      className={`border rounded p-3 cursor-pointer transition-colors ${
                        selectedFiles.has(file.id) ? 'border-indigo-500 bg-indigo-50' : 'hover:border-gray-400'
                      }`}
                      onClick={() => toggleFileSelection(file.id)}
                    >
                      {viewMode === 'grid' ? (
                        <div>
                          {file.thumbnail ? (
                            <img 
                              src={file.thumbnail} 
                              alt={file.name}
                              className="w-full h-32 object-cover rounded mb-2"
                            />
                          ) : (
                            <div className={`w-full h-32 rounded mb-2 flex items-center justify-center ${fileTypeConfig[file.type].bgColor}`}>
                              {fileTypeConfig[file.type].icon}
                            </div>
                          )}
                          
                          <h3 className="font-medium text-sm truncate mb-1">{file.name}</h3>
                          <p className="text-xs text-muted-foreground mb-2">
                            {(file.size / 1024 / 1024).toFixed(1)} MB
                          </p>
                          
                          {file.textDescription && (
                            <p className="text-xs text-muted-foreground line-clamp-2">
                              {file.textDescription}
                            </p>
                          )}
                        </div>
                      ) : (
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded ${fileTypeConfig[file.type].bgColor}`}>
                            {fileTypeConfig[file.type].icon}
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <h3 className="font-medium text-sm truncate">{file.name}</h3>
                            <p className="text-xs text-muted-foreground">
                              {(file.size / 1024 / 1024).toFixed(1)} MB • {file.type}
                            </p>
                          </div>
                          
                          <Badge 
                            variant={file.processingStatus === 'completed' ? 'default' : 'secondary'}
                            className="text-xs"
                          >
                            {file.processingStatus}
                          </Badge>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MultiModalUploader;
