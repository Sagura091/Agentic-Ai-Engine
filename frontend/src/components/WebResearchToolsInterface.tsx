import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Globe, 
  Search, 
  ExternalLink, 
  Download, 
  FileText, 
  Image, 
  Video,
  Database,
  Cpu,
  Zap,
  Settings,
  Plus,
  Trash2,
  Eye,
  CheckCircle,
  XCircle,
  Loader2,
  AlertTriangle,
  TrendingUp,
  BarChart3,
  Clock,
  Target
} from 'lucide-react';
import { useQuery, useMutation } from 'react-query';
import { webResearchApi, toolsApi } from '../services/api';
import toast from 'react-hot-toast';

interface WebSearchResult {
  id: string;
  title: string;
  url: string;
  snippet: string;
  source: string;
  relevance_score: number;
  timestamp: string;
  content_type: 'article' | 'video' | 'image' | 'document' | 'social';
  extracted_data?: Record<string, any>;
}

interface ExternalTool {
  id: string;
  name: string;
  description: string;
  category: 'web_scraping' | 'api_integration' | 'data_processing' | 'analysis' | 'automation';
  status: 'active' | 'inactive' | 'error';
  usage_count: number;
  last_used: string;
  configuration: Record<string, any>;
  capabilities: string[];
}

interface ResearchSession {
  id: string;
  query: string;
  results: WebSearchResult[];
  tools_used: string[];
  start_time: string;
  end_time?: string;
  status: 'active' | 'completed' | 'failed';
  insights: string[];
}

export const WebResearchToolsInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'research' | 'tools' | 'sessions' | 'insights'>('research');
  
  // Research states
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<WebSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedResults, setSelectedResults] = useState<string[]>([]);
  
  // Tool states
  const [availableTools, setAvailableTools] = useState<ExternalTool[]>([]);
  const [selectedTool, setSelectedTool] = useState<ExternalTool | null>(null);
  const [toolConfig, setToolConfig] = useState<Record<string, any>>({});
  
  // Session states
  const [currentSession, setCurrentSession] = useState<ResearchSession | null>(null);
  const [researchHistory, setResearchHistory] = useState<ResearchSession[]>([]);
  
  // Advanced options
  const [searchOptions, setSearchOptions] = useState({
    max_results: 10,
    include_images: false,
    include_videos: false,
    time_range: 'any' as 'any' | 'day' | 'week' | 'month' | 'year',
    language: 'en',
    region: 'us',
    safe_search: true
  });

  // Fetch available external tools
  const { data: externalTools } = useQuery(
    'external-tools',
    () => toolsApi.getExternalTools(),
    {
      onSuccess: (data) => setAvailableTools(data),
      refetchInterval: 30000 // Refresh every 30 seconds
    }
  );

  // Enhanced web search with multiple sources
  const performWebSearch = async () => {
    if (!searchQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    setIsSearching(true);
    setSearchResults([]);

    try {
      // Start new research session
      const sessionId = `research_${Date.now()}`;
      const newSession: ResearchSession = {
        id: sessionId,
        query: searchQuery,
        results: [],
        tools_used: [],
        start_time: new Date().toISOString(),
        status: 'active',
        insights: []
      };
      setCurrentSession(newSession);

      // Perform comprehensive web search
      const searchRequest = {
        query: searchQuery,
        options: searchOptions,
        session_id: sessionId
      };

      const response = await webResearchApi.comprehensiveSearch(searchRequest);
      const results = response.data.results;

      setSearchResults(results);
      setCurrentSession(prev => prev ? {
        ...prev,
        results,
        status: 'completed',
        end_time: new Date().toISOString()
      } : null);

      // Add to research history
      setResearchHistory(prev => [newSession, ...prev.slice(0, 9)]);

      toast.success(`Found ${results.length} results`);
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Search failed. Please try again.');
      setCurrentSession(prev => prev ? { ...prev, status: 'failed' } : null);
    } finally {
      setIsSearching(false);
    }
  };

  // Extract data from selected results
  const extractDataFromResults = async () => {
    if (selectedResults.length === 0) {
      toast.error('Please select results to extract data from');
      return;
    }

    try {
      const extractionRequest = {
        result_ids: selectedResults,
        extraction_type: 'comprehensive',
        session_id: currentSession?.id
      };

      const response = await webResearchApi.extractData(extractionRequest);
      
      // Update results with extracted data
      setSearchResults(prev => prev.map(result => {
        const extracted = response.data.extractions.find((e: any) => e.result_id === result.id);
        return extracted ? { ...result, extracted_data: extracted.data } : result;
      }));

      toast.success('Data extraction completed');
    } catch (error) {
      console.error('Extraction error:', error);
      toast.error('Data extraction failed');
    }
  };

  // Execute external tool
  const executeExternalTool = async (tool: ExternalTool, config: Record<string, any>) => {
    try {
      const executionRequest = {
        tool_id: tool.id,
        configuration: config,
        input_data: selectedResults.length > 0 ? 
          searchResults.filter(r => selectedResults.includes(r.id)) : 
          searchResults,
        session_id: currentSession?.id
      };

      const response = await toolsApi.executeExternalTool(executionRequest);
      
      toast.success(`${tool.name} executed successfully`);
      
      // Update tool usage
      setAvailableTools(prev => prev.map(t => 
        t.id === tool.id ? { 
          ...t, 
          usage_count: t.usage_count + 1, 
          last_used: new Date().toISOString() 
        } : t
      ));

      return response.data;
    } catch (error) {
      console.error('Tool execution error:', error);
      toast.error(`Failed to execute ${tool.name}`);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Globe className="h-6 w-6 text-blue-500" />
              <div>
                <CardTitle className="text-xl">Web Research & External Tools</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Comprehensive web research with advanced data extraction and external tool integration
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {currentSession && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  Active Session
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-muted p-1 rounded-lg w-fit">
        {[
          { id: 'research', label: 'Web Research', icon: Search },
          { id: 'tools', label: 'External Tools', icon: Cpu },
          { id: 'sessions', label: 'Research Sessions', icon: Clock },
          { id: 'insights', label: 'Insights', icon: TrendingUp }
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

      {/* Research Tab */}
      {activeTab === 'research' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Search Interface */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Search className="h-5 w-5" />
                  <span>Advanced Web Search</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex space-x-2">
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter your research query..."
                    className="flex-1"
                    onKeyPress={(e) => e.key === 'Enter' && performWebSearch()}
                    disabled={isSearching}
                  />
                  <Button
                    onClick={performWebSearch}
                    disabled={!searchQuery.trim() || isSearching}
                    className="flex items-center space-x-2"
                  >
                    {isSearching ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Searching...</span>
                      </>
                    ) : (
                      <>
                        <Search className="h-4 w-4" />
                        <span>Search</span>
                      </>
                    )}
                  </Button>
                </div>

                {/* Search Options */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">
                      Max Results
                    </label>
                    <select
                      value={searchOptions.max_results}
                      onChange={(e) => setSearchOptions(prev => ({ 
                        ...prev, 
                        max_results: parseInt(e.target.value) 
                      }))}
                      className="input text-sm"
                    >
                      <option value={5}>5 results</option>
                      <option value={10}>10 results</option>
                      <option value={20}>20 results</option>
                      <option value={50}>50 results</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">
                      Time Range
                    </label>
                    <select
                      value={searchOptions.time_range}
                      onChange={(e) => setSearchOptions(prev => ({ 
                        ...prev, 
                        time_range: e.target.value as any 
                      }))}
                      className="input text-sm"
                    >
                      <option value="any">Any time</option>
                      <option value="day">Past day</option>
                      <option value="week">Past week</option>
                      <option value="month">Past month</option>
                      <option value="year">Past year</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center space-x-4 text-sm">
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="include-images"
                      checked={searchOptions.include_images}
                      onChange={(e) => setSearchOptions(prev => ({ 
                        ...prev, 
                        include_images: e.target.checked 
                      }))}
                      className="rounded border-border"
                    />
                    <label htmlFor="include-images" className="text-foreground">
                      Include images
                    </label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="include-videos"
                      checked={searchOptions.include_videos}
                      onChange={(e) => setSearchOptions(prev => ({ 
                        ...prev, 
                        include_videos: e.target.checked 
                      }))}
                      className="rounded border-border"
                    />
                    <label htmlFor="include-videos" className="text-foreground">
                      Include videos
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg flex items-center space-x-2">
                      <FileText className="h-5 w-5" />
                      <span>Search Results ({searchResults.length})</span>
                    </CardTitle>
                    <div className="flex items-center space-x-2">
                      <Button
                        onClick={extractDataFromResults}
                        disabled={selectedResults.length === 0}
                        variant="outline"
                        size="sm"
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Extract Data
                      </Button>
                      <Button
                        onClick={() => setSelectedResults(
                          selectedResults.length === searchResults.length ? [] : searchResults.map(r => r.id)
                        )}
                        variant="outline"
                        size="sm"
                      >
                        {selectedResults.length === searchResults.length ? 'Deselect All' : 'Select All'}
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    <div className="space-y-3">
                      {searchResults.map((result) => (
                        <div
                          key={result.id}
                          className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                            selectedResults.includes(result.id)
                              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                              : 'border-border hover:border-muted-foreground'
                          }`}
                          onClick={() => {
                            setSelectedResults(prev =>
                              prev.includes(result.id)
                                ? prev.filter(id => id !== result.id)
                                : [...prev, result.id]
                            );
                          }}
                        >
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              {result.content_type === 'video' && <Video className="h-4 w-4 text-red-500" />}
                              {result.content_type === 'image' && <Image className="h-4 w-4 text-green-500" />}
                              {result.content_type === 'article' && <FileText className="h-4 w-4 text-blue-500" />}
                              {result.content_type === 'document' && <FileText className="h-4 w-4 text-purple-500" />}
                              <h3 className="font-medium text-sm line-clamp-2">{result.title}</h3>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Badge variant="outline" className="text-xs">
                                {Math.round(result.relevance_score * 100)}%
                              </Badge>
                              <a
                                href={result.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-500 hover:text-blue-700"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <ExternalLink className="h-4 w-4" />
                              </a>
                            </div>
                          </div>
                          <p className="text-sm text-muted-foreground mb-2 line-clamp-3">
                            {result.snippet}
                          </p>
                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span>{result.source}</span>
                            <span>{new Date(result.timestamp).toLocaleDateString()}</span>
                          </div>
                          {result.extracted_data && (
                            <div className="mt-2 p-2 bg-muted rounded text-xs">
                              <span className="font-medium">Extracted Data Available</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Research Control Panel */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Settings className="h-5 w-5" />
                  <span>Research Control</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {currentSession ? (
                  <div className="space-y-3">
                    <div>
                      <span className="text-sm text-muted-foreground">Current Query:</span>
                      <p className="text-sm font-medium mt-1">{currentSession.query}</p>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Status:</span>
                      <Badge variant="outline" className="flex items-center gap-1">
                        {currentSession.status === 'active' && <Loader2 className="h-3 w-3 animate-spin" />}
                        {currentSession.status === 'completed' && <CheckCircle className="h-3 w-3 text-green-500" />}
                        {currentSession.status === 'failed' && <XCircle className="h-3 w-3 text-red-500" />}
                        {currentSession.status}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Results:</span>
                      <span className="text-sm font-medium">{currentSession.results.length}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Selected:</span>
                      <span className="text-sm font-medium">{selectedResults.length}</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-sm text-muted-foreground">
                      Start a research session
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Tools */}
            {availableTools.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center space-x-2">
                    <Zap className="h-5 w-5" />
                    <span>Quick Tools</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {availableTools.slice(0, 3).map((tool) => (
                      <button
                        key={tool.id}
                        onClick={() => executeExternalTool(tool, {})}
                        disabled={selectedResults.length === 0}
                        className="w-full p-3 text-left border border-border rounded-lg hover:border-muted-foreground transition-colors disabled:opacity-50"
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium text-sm">{tool.name}</p>
                            <p className="text-xs text-muted-foreground">{tool.description}</p>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            {tool.category}
                          </Badge>
                        </div>
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* External Tools Tab */}
      {activeTab === 'tools' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Cpu className="h-5 w-5" />
                <span>Available External Tools</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-3">
                  {availableTools.map((tool) => (
                    <div
                      key={tool.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedTool?.id === tool.id
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-border hover:border-muted-foreground'
                      }`}
                      onClick={() => setSelectedTool(tool)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h3 className="font-medium text-sm">{tool.name}</h3>
                          <p className="text-xs text-muted-foreground mt-1">{tool.description}</p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {tool.category}
                          </Badge>
                          <Badge
                            variant={tool.status === 'active' ? 'default' : 'secondary'}
                            className="text-xs"
                          >
                            {tool.status}
                          </Badge>
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Used {tool.usage_count} times</span>
                        <span>Last used: {new Date(tool.last_used).toLocaleDateString()}</span>
                      </div>
                      {tool.capabilities.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {tool.capabilities.slice(0, 3).map((capability, index) => (
                            <span key={index} className="px-2 py-1 bg-muted text-xs rounded">
                              {capability}
                            </span>
                          ))}
                          {tool.capabilities.length > 3 && (
                            <span className="px-2 py-1 bg-muted text-xs rounded">
                              +{tool.capabilities.length - 3} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Tool Configuration */}
          {selectedTool && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Settings className="h-5 w-5" />
                  <span>{selectedTool.name} Configuration</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm text-muted-foreground mb-3">{selectedTool.description}</p>

                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Tool Category
                      </label>
                      <Badge variant="outline">{selectedTool.category}</Badge>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Capabilities
                      </label>
                      <div className="flex flex-wrap gap-1">
                        {selectedTool.capabilities.map((capability, index) => (
                          <span key={index} className="px-2 py-1 bg-muted text-xs rounded">
                            {capability}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Configuration
                      </label>
                      <Textarea
                        value={JSON.stringify(toolConfig, null, 2)}
                        onChange={(e) => {
                          try {
                            setToolConfig(JSON.parse(e.target.value));
                          } catch {
                            // Invalid JSON, keep current config
                          }
                        }}
                        placeholder="Enter tool configuration as JSON..."
                        rows={6}
                        className="font-mono text-sm"
                      />
                    </div>

                    <Button
                      onClick={() => executeExternalTool(selectedTool, toolConfig)}
                      className="w-full"
                      disabled={selectedResults.length === 0 && searchResults.length === 0}
                    >
                      <Zap className="h-4 w-4 mr-2" />
                      Execute Tool
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};
