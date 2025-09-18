import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  MessageSquare, 
  User, 
  Clock, 
  Brain, 
  Target, 
  TrendingUp,
  History,
  Settings,
  Lightbulb,
  Search,
  Filter,
  RotateCcw,
  Bookmark,
  Star,
  ThumbsUp,
  ThumbsDown,
  Eye,
  EyeOff,
  Layers,
  Zap,
  Users,
  Calendar,
  Tag,
  ArrowRight,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { toast } from 'sonner';

// Types for contextual search
interface ConversationContext {
  conversationId: string;
  userId: string;
  currentTopic?: string;
  entityMentions: string[];
  intentHistory: string[];
  semanticThread?: string;
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
  }>;
}

interface UserProfile {
  userId: string;
  preferences: Record<string, any>;
  searchHistory: Array<{
    query: string;
    timestamp: string;
    results: number;
    rating?: number;
  }>;
  domainExpertise: Record<string, number>;
  contentPreferences: Record<string, number>;
  interactionPatterns: Record<string, number>;
}

interface ContextualResult {
  id: string;
  content: string;
  title: string;
  baseSimilarity: number;
  contextualRelevance: number;
  temporalRelevance: number;
  personalizationScore: number;
  finalScore: number;
  contextExplanation: {
    contextualFactors: string[];
    temporalFactors: string[];
    personalizationFactors: string[];
    overallReasoning: string;
  };
  metadata: Record<string, any>;
}

interface SearchSettings {
  retrievalMode: 'standard' | 'conversational' | 'personalized' | 'temporal' | 'adaptive' | 'hybrid';
  contextWeight: number;
  temporalWeight: number;
  personalizationWeight: number;
  conversationDepth: number;
  enableLearning: boolean;
}

const ContextualSearchInterface: React.FC = () => {
  // Search state
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ContextualResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  // Context state
  const [conversationContext, setConversationContext] = useState<ConversationContext>({
    conversationId: 'conv-' + Date.now(),
    userId: 'user-123',
    entityMentions: [],
    intentHistory: [],
    messages: []
  });

  const [userProfile, setUserProfile] = useState<UserProfile>({
    userId: 'user-123',
    preferences: {},
    searchHistory: [],
    domainExpertise: {
      technology: 0.8,
      science: 0.6,
      business: 0.4
    },
    contentPreferences: {
      technical: 0.9,
      tutorial: 0.7,
      reference: 0.8
    },
    interactionPatterns: {
      avgRating: 4.2,
      queryLength: 8.5
    }
  });

  // UI state
  const [searchSettings, setSearchSettings] = useState<SearchSettings>({
    retrievalMode: 'hybrid',
    contextWeight: 0.5,
    temporalWeight: 0.3,
    personalizationWeight: 0.4,
    conversationDepth: 5,
    enableLearning: true
  });

  const [showSettings, setShowSettings] = useState(false);
  const [showExplanations, setShowExplanations] = useState(true);
  const [selectedResult, setSelectedResult] = useState<ContextualResult | null>(null);
  const [expandedExplanations, setExpandedExplanations] = useState<Set<string>>(new Set());

  // Refs
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Perform contextual search
  const performSearch = useCallback(async () => {
    if (!query.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    setIsSearching(true);

    try {
      // Add to conversation context
      const newMessage = {
        role: 'user' as const,
        content: query,
        timestamp: new Date().toISOString()
      };

      const updatedContext = {
        ...conversationContext,
        messages: [...conversationContext.messages, newMessage]
      };
      setConversationContext(updatedContext);

      // Add to search history
      setSearchHistory(prev => [query, ...prev.slice(0, 9)]);

      // Simulate contextual search API call
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Mock contextual results
      const mockResults: ContextualResult[] = [
        {
          id: '1',
          title: 'Advanced Machine Learning Techniques',
          content: 'Comprehensive guide to modern ML algorithms including deep learning, reinforcement learning, and neural networks...',
          baseSimilarity: 0.85,
          contextualRelevance: 0.92,
          temporalRelevance: 0.78,
          personalizationScore: 0.88,
          finalScore: 0.86,
          contextExplanation: {
            contextualFactors: [
              'Related to current topic: machine learning',
              'Mentions entities from conversation: algorithms, neural networks'
            ],
            temporalFactors: ['Recent and relevant content'],
            personalizationFactors: ['Highly aligned with your technical preferences'],
            overallReasoning: 'This result was ranked considering conversation context (0.92), temporal relevance (0.78), and your personal preferences (0.88)'
          },
          metadata: {
            contentType: 'technical',
            domain: 'technology',
            createdAt: '2024-01-15T10:00:00Z'
          }
        },
        {
          id: '2',
          title: 'Introduction to AI Ethics',
          content: 'Exploring the ethical implications of artificial intelligence and machine learning systems in modern society...',
          baseSimilarity: 0.72,
          contextualRelevance: 0.68,
          temporalRelevance: 0.85,
          personalizationScore: 0.65,
          finalScore: 0.73,
          contextExplanation: {
            contextualFactors: [
              'Related to AI and ethics discussion'
            ],
            temporalFactors: ['Very recent content, highly current'],
            personalizationFactors: ['Moderately aligned with your preferences'],
            overallReasoning: 'This result was ranked considering conversation context (0.68), temporal relevance (0.85), and your personal preferences (0.65)'
          },
          metadata: {
            contentType: 'reference',
            domain: 'technology',
            createdAt: '2024-01-20T14:30:00Z'
          }
        },
        {
          id: '3',
          title: 'Data Science Best Practices',
          content: 'Essential practices for data scientists including data preprocessing, model validation, and deployment strategies...',
          baseSimilarity: 0.79,
          contextualRelevance: 0.75,
          temporalRelevance: 0.65,
          personalizationScore: 0.82,
          finalScore: 0.75,
          contextExplanation: {
            contextualFactors: [
              'Related to data science methodology'
            ],
            temporalFactors: ['Moderately recent content'],
            personalizationFactors: ['Well aligned with your technical background'],
            overallReasoning: 'This result was ranked considering conversation context (0.75), temporal relevance (0.65), and your personal preferences (0.82)'
          },
          metadata: {
            contentType: 'tutorial',
            domain: 'technology',
            createdAt: '2024-01-10T09:15:00Z'
          }
        }
      ];

      setResults(mockResults);

      // Update user profile with search
      const searchEntry = {
        query,
        timestamp: new Date().toISOString(),
        results: mockResults.length
      };

      setUserProfile(prev => ({
        ...prev,
        searchHistory: [searchEntry, ...prev.searchHistory.slice(0, 19)]
      }));

      toast.success(`Found ${mockResults.length} contextual results`);

    } catch (error) {
      console.error('Contextual search failed:', error);
      toast.error('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
    }
  }, [query, conversationContext, searchSettings]);

  // Rate result
  const rateResult = useCallback(async (resultId: string, rating: number) => {
    try {
      // Update user profile with rating
      setUserProfile(prev => {
        const updatedHistory = prev.searchHistory.map(entry => 
          entry.query === query ? { ...entry, rating } : entry
        );
        
        return {
          ...prev,
          searchHistory: updatedHistory,
          interactionPatterns: {
            ...prev.interactionPatterns,
            avgRating: (prev.interactionPatterns.avgRating * 0.9) + (rating * 0.1)
          }
        };
      });

      toast.success('Thank you for your feedback!');
    } catch (error) {
      console.error('Failed to rate result:', error);
      toast.error('Failed to save rating');
    }
  }, [query]);

  // Toggle explanation expansion
  const toggleExplanation = (resultId: string) => {
    setExpandedExplanations(prev => {
      const newSet = new Set(prev);
      if (newSet.has(resultId)) {
        newSet.delete(resultId);
      } else {
        newSet.add(resultId);
      }
      return newSet;
    });
  };

  // Handle search on Enter
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      performSearch();
    }
  };

  // Clear conversation context
  const clearContext = () => {
    setConversationContext({
      conversationId: 'conv-' + Date.now(),
      userId: conversationContext.userId,
      entityMentions: [],
      intentHistory: [],
      messages: []
    });
    toast.success('Conversation context cleared');
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
                <CardTitle className="text-xl">Contextual Search Interface</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Conversation-aware and personalized search with temporal intelligence
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <MessageSquare className="h-3 w-3" />
                {conversationContext.messages.length} messages
              </Badge>
              <Badge variant="outline" className="flex items-center gap-1">
                <History className="h-3 w-3" />
                {userProfile.searchHistory.length} searches
              </Badge>
              <Button variant="outline" size="sm" onClick={() => setShowSettings(!showSettings)}>
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Context Panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Search Settings */}
          {showSettings && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Search Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-xs font-medium mb-2 block">Retrieval Mode</label>
                  <Select 
                    value={searchSettings.retrievalMode} 
                    onValueChange={(value: any) => setSearchSettings(prev => ({ ...prev, retrievalMode: value }))}
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="standard">Standard</SelectItem>
                      <SelectItem value="conversational">Conversational</SelectItem>
                      <SelectItem value="personalized">Personalized</SelectItem>
                      <SelectItem value="temporal">Temporal</SelectItem>
                      <SelectItem value="adaptive">Adaptive</SelectItem>
                      <SelectItem value="hybrid">Hybrid</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-xs font-medium mb-2 block">
                    Context Weight: {searchSettings.contextWeight.toFixed(2)}
                  </label>
                  <Slider
                    value={[searchSettings.contextWeight]}
                    onValueChange={([value]) => setSearchSettings(prev => ({ ...prev, contextWeight: value }))}
                    max={1}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium mb-2 block">
                    Temporal Weight: {searchSettings.temporalWeight.toFixed(2)}
                  </label>
                  <Slider
                    value={[searchSettings.temporalWeight]}
                    onValueChange={([value]) => setSearchSettings(prev => ({ ...prev, temporalWeight: value }))}
                    max={1}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium mb-2 block">
                    Personalization: {searchSettings.personalizationWeight.toFixed(2)}
                  </label>
                  <Slider
                    value={[searchSettings.personalizationWeight]}
                    onValueChange={([value]) => setSearchSettings(prev => ({ ...prev, personalizationWeight: value }))}
                    max={1}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium">Enable Learning</label>
                  <Switch 
                    checked={searchSettings.enableLearning} 
                    onCheckedChange={(checked) => setSearchSettings(prev => ({ ...prev, enableLearning: checked }))}
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Conversation Context */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Conversation Context</CardTitle>
                <Button variant="outline" size="sm" onClick={clearContext}>
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {conversationContext.currentTopic && (
                <div>
                  <label className="text-xs font-medium">Current Topic</label>
                  <Badge variant="outline" className="mt-1">
                    <Tag className="h-3 w-3 mr-1" />
                    {conversationContext.currentTopic}
                  </Badge>
                </div>
              )}

              {conversationContext.entityMentions.length > 0 && (
                <div>
                  <label className="text-xs font-medium mb-1 block">Entity Mentions</label>
                  <div className="flex flex-wrap gap-1">
                    {conversationContext.entityMentions.slice(-5).map((entity, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {entity}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {conversationContext.messages.length > 0 && (
                <div>
                  <label className="text-xs font-medium mb-1 block">Recent Messages</label>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {conversationContext.messages.slice(-3).map((message, index) => (
                      <div key={index} className="text-xs p-2 bg-muted rounded">
                        <div className="flex items-center gap-1 mb-1">
                          {message.role === 'user' ? <User className="h-3 w-3" /> : <Brain className="h-3 w-3" />}
                          <span className="font-medium capitalize">{message.role}</span>
                        </div>
                        <p className="line-clamp-2">{message.content}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* User Profile */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">User Profile</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <label className="text-xs font-medium mb-1 block">Domain Expertise</label>
                <div className="space-y-1">
                  {Object.entries(userProfile.domainExpertise).map(([domain, level]) => (
                    <div key={domain} className="flex justify-between items-center">
                      <span className="text-xs capitalize">{domain}</span>
                      <div className="flex items-center gap-2">
                        <Progress value={level * 100} className="w-16 h-2" />
                        <span className="text-xs text-muted-foreground">{(level * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <label className="text-xs font-medium mb-1 block">Content Preferences</label>
                <div className="space-y-1">
                  {Object.entries(userProfile.contentPreferences).map(([type, preference]) => (
                    <div key={type} className="flex justify-between items-center">
                      <span className="text-xs capitalize">{type}</span>
                      <div className="flex items-center gap-2">
                        <Progress value={preference * 100} className="w-16 h-2" />
                        <span className="text-xs text-muted-foreground">{(preference * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <label className="text-xs font-medium mb-1 block">Interaction Patterns</label>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>Avg Rating</span>
                    <span>{userProfile.interactionPatterns.avgRating?.toFixed(1)}/5.0</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>Query Length</span>
                    <span>{userProfile.interactionPatterns.queryLength?.toFixed(1)} words</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Search History */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Recent Searches</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {searchHistory.slice(0, 5).map((search, index) => (
                  <div
                    key={index}
                    className="text-xs p-2 bg-muted rounded cursor-pointer hover:bg-muted/80"
                    onClick={() => setQuery(search)}
                  >
                    <div className="flex items-center gap-1">
                      <History className="h-3 w-3" />
                      <span className="truncate">{search}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Search Area */}
        <div className="lg:col-span-3 space-y-4">
          {/* Search Input */}
          <Card>
            <CardContent className="p-4">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    ref={searchInputRef}
                    placeholder="Enter your contextual search query..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="pl-10"
                  />
                </div>
                
                <Button 
                  onClick={performSearch}
                  disabled={isSearching || !query.trim()}
                  className="min-w-[100px]"
                >
                  {isSearching ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <>
                      <Search className="h-4 w-4 mr-2" />
                      Search
                    </>
                  )}
                </Button>
              </div>
              
              <div className="flex items-center justify-between mt-3">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    Mode: {searchSettings.retrievalMode}
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    Context: {(searchSettings.contextWeight * 100).toFixed(0)}%
                  </Badge>
                </div>
                
                <div className="flex items-center gap-2">
                  <Switch 
                    checked={showExplanations} 
                    onCheckedChange={setShowExplanations}
                    id="show-explanations"
                  />
                  <label htmlFor="show-explanations" className="text-xs">Show explanations</label>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Search Results */}
          {results.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Contextual Search Results</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Results ranked by contextual relevance, temporal factors, and personalization
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.map((result, index) => (
                    <div key={result.id} className="border rounded p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <h3 className="font-semibold text-lg mb-1">{result.title}</h3>
                          <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                            {result.content}
                          </p>
                          
                          {/* Score Breakdown */}
                          <div className="flex items-center gap-4 mb-2">
                            <div className="flex items-center gap-1">
                              <Target className="h-3 w-3 text-blue-500" />
                              <span className="text-xs">Final: {(result.finalScore * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <MessageSquare className="h-3 w-3 text-green-500" />
                              <span className="text-xs">Context: {(result.contextualRelevance * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3 text-orange-500" />
                              <span className="text-xs">Temporal: {(result.temporalRelevance * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <User className="h-3 w-3 text-purple-500" />
                              <span className="text-xs">Personal: {(result.personalizationScore * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                          
                          {/* Metadata */}
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary" className="text-xs">
                              {result.metadata.contentType}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              {result.metadata.domain}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {new Date(result.metadata.createdAt).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        
                        {/* Rating */}
                        <div className="flex items-center gap-1 ml-4">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => rateResult(result.id, 5)}
                          >
                            <ThumbsUp className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => rateResult(result.id, 1)}
                          >
                            <ThumbsDown className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      
                      {/* Context Explanation */}
                      {showExplanations && (
                        <div className="border-t pt-3">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleExplanation(result.id)}
                            className="mb-2"
                          >
                            <Lightbulb className="h-4 w-4 mr-2" />
                            Why this result?
                            {expandedExplanations.has(result.id) ? 
                              <ChevronUp className="h-4 w-4 ml-2" /> : 
                              <ChevronDown className="h-4 w-4 ml-2" />
                            }
                          </Button>
                          
                          {expandedExplanations.has(result.id) && (
                            <div className="bg-muted/50 rounded p-3 space-y-2">
                              {result.contextExplanation.contextualFactors.length > 0 && (
                                <div>
                                  <h4 className="text-xs font-medium mb-1">Contextual Factors:</h4>
                                  <ul className="text-xs space-y-1">
                                    {result.contextExplanation.contextualFactors.map((factor, i) => (
                                      <li key={i} className="flex items-start gap-1">
                                        <ArrowRight className="h-3 w-3 mt-0.5 text-muted-foreground" />
                                        {factor}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              
                              {result.contextExplanation.temporalFactors.length > 0 && (
                                <div>
                                  <h4 className="text-xs font-medium mb-1">Temporal Factors:</h4>
                                  <ul className="text-xs space-y-1">
                                    {result.contextExplanation.temporalFactors.map((factor, i) => (
                                      <li key={i} className="flex items-start gap-1">
                                        <ArrowRight className="h-3 w-3 mt-0.5 text-muted-foreground" />
                                        {factor}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              
                              {result.contextExplanation.personalizationFactors.length > 0 && (
                                <div>
                                  <h4 className="text-xs font-medium mb-1">Personalization Factors:</h4>
                                  <ul className="text-xs space-y-1">
                                    {result.contextExplanation.personalizationFactors.map((factor, i) => (
                                      <li key={i} className="flex items-start gap-1">
                                        <ArrowRight className="h-3 w-3 mt-0.5 text-muted-foreground" />
                                        {factor}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              
                              <div className="pt-2 border-t">
                                <p className="text-xs text-muted-foreground">
                                  <strong>Overall:</strong> {result.contextExplanation.overallReasoning}
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Empty State */}
          {results.length === 0 && !isSearching && (
            <Card>
              <CardContent className="p-8 text-center">
                <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">Contextual Search Ready</h3>
                <p className="text-muted-foreground mb-4">
                  Enter a query to experience conversation-aware and personalized search
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-blue-500" />
                    <span>Conversation-aware</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-orange-500" />
                    <span>Temporal intelligence</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <User className="h-4 w-4 text-purple-500" />
                    <span>Personalized results</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContextualSearchInterface;
