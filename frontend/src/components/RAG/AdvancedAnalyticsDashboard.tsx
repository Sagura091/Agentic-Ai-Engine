import React, { useState, useEffect, useCallback } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  PieChart, 
  Activity, 
  Search, 
  Users, 
  Clock, 
  Target, 
  AlertTriangle, 
  CheckCircle, 
  Lightbulb,
  Zap,
  Brain,
  Eye,
  Settings,
  Download,
  RefreshCw,
  Filter,
  Calendar,
  ArrowUp,
  ArrowDown,
  Minus,
  Star,
  ThumbsUp,
  ThumbsDown,
  Database,
  Globe,
  Cpu,
  HardDrive,
  Network,
  Timer
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';

// Types for analytics
interface SearchPattern {
  patternId: string;
  patternType: string;
  description: string;
  frequency: number;
  usersAffected: number;
  avgSatisfaction: number;
  optimizationPotential: number;
  examples: string[];
  metadata: Record<string, any>;
}

interface QueryOptimization {
  originalQuery: string;
  optimizedQuery: string;
  optimizationType: string;
  expectedImprovement: number;
  confidence: number;
  reasoning: string;
  metadata: Record<string, any>;
}

interface AnalyticsInsight {
  id: string;
  insightType: 'trend' | 'anomaly' | 'optimization' | 'recommendation' | 'warning' | 'success';
  analyticsType: 'search_patterns' | 'query_optimization' | 'user_behavior' | 'performance' | 'knowledge_base' | 'content_quality';
  title: string;
  description: string;
  impactScore: number;
  confidence: number;
  actionableRecommendations: string[];
  supportingData: Record<string, any>;
  generatedAt: string;
}

interface PerformanceMetrics {
  avgResponseTime: number;
  p95ResponseTime: number;
  cacheHitRate: number;
  searchSuccessRate: number;
  userSatisfaction: number;
  queryComplexityScore: number;
  knowledgeCoverage: number;
  systemEfficiency: number;
}

interface AnalyticsSummary {
  totalInsights: number;
  highImpactInsights: number;
  insightCategories: Record<string, number>;
  searchAnalytics: {
    totalEvents: number;
    identifiedPatterns: number;
    recentEvents: number;
  };
  performanceMetrics: PerformanceMetrics;
  topInsights: AnalyticsInsight[];
  generatedAt: string;
}

const AdvancedAnalyticsDashboard: React.FC = () => {
  // Analytics state
  const [analyticsSummary, setAnalyticsSummary] = useState<AnalyticsSummary | null>(null);
  const [searchPatterns, setSearchPatterns] = useState<SearchPattern[]>([]);
  const [insights, setInsights] = useState<AnalyticsInsight[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // UI state
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const [selectedInsightType, setSelectedInsightType] = useState('all');
  const [selectedTab, setSelectedTab] = useState('overview');

  // Insight type configurations
  const insightTypeConfig = {
    trend: { icon: <TrendingUp className="h-4 w-4" />, color: 'text-blue-600', bg: 'bg-blue-50', label: 'Trend' },
    anomaly: { icon: <AlertTriangle className="h-4 w-4" />, color: 'text-red-600', bg: 'bg-red-50', label: 'Anomaly' },
    optimization: { icon: <Zap className="h-4 w-4" />, color: 'text-yellow-600', bg: 'bg-yellow-50', label: 'Optimization' },
    recommendation: { icon: <Lightbulb className="h-4 w-4" />, color: 'text-green-600', bg: 'bg-green-50', label: 'Recommendation' },
    warning: { icon: <AlertTriangle className="h-4 w-4" />, color: 'text-orange-600', bg: 'bg-orange-50', label: 'Warning' },
    success: { icon: <CheckCircle className="h-4 w-4" />, color: 'text-emerald-600', bg: 'bg-emerald-50', label: 'Success' }
  };

  // Load analytics data
  const loadAnalyticsData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Simulate API calls
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Mock analytics summary
      const mockSummary: AnalyticsSummary = {
        totalInsights: 12,
        highImpactInsights: 4,
        insightCategories: {
          search_patterns: 5,
          performance: 3,
          user_behavior: 2,
          knowledge_base: 2
        },
        searchAnalytics: {
          totalEvents: 1247,
          identifiedPatterns: 8,
          recentEvents: 156
        },
        performanceMetrics: {
          avgResponseTime: 0.85,
          p95ResponseTime: 2.1,
          cacheHitRate: 0.78,
          searchSuccessRate: 0.92,
          userSatisfaction: 0.84,
          queryComplexityScore: 0.65,
          knowledgeCoverage: 0.88,
          systemEfficiency: 0.91
        },
        topInsights: [],
        generatedAt: new Date().toISOString()
      };

      // Mock search patterns
      const mockPatterns: SearchPattern[] = [
        {
          patternId: 'pattern-1',
          patternType: 'similar_queries',
          description: 'Frequent similar queries about machine learning algorithms',
          frequency: 45,
          usersAffected: 12,
          avgSatisfaction: 0.72,
          optimizationPotential: 0.85,
          examples: ['ML algorithms', 'machine learning methods', 'AI algorithms'],
          metadata: { category: 'technical' }
        },
        {
          patternId: 'pattern-2',
          patternType: 'zero_results',
          description: 'Queries returning no results for advanced topics',
          frequency: 23,
          usersAffected: 8,
          avgSatisfaction: 0.15,
          optimizationPotential: 0.95,
          examples: ['quantum computing basics', 'advanced neural architectures'],
          metadata: { category: 'knowledge_gap' }
        },
        {
          patternId: 'pattern-3',
          patternType: 'slow_queries',
          description: 'Complex queries with high response times',
          frequency: 18,
          usersAffected: 6,
          avgSatisfaction: 0.58,
          optimizationPotential: 0.75,
          examples: ['comprehensive analysis of deep learning', 'detailed comparison of frameworks'],
          metadata: { category: 'performance' }
        }
      ];

      // Mock insights
      const mockInsights: AnalyticsInsight[] = [
        {
          id: 'insight-1',
          insightType: 'optimization',
          analyticsType: 'search_patterns',
          title: 'High Optimization Potential: Similar Queries',
          description: 'Frequent similar queries about machine learning algorithms show 85% optimization potential',
          impactScore: 0.85,
          confidence: 0.9,
          actionableRecommendations: [
            'Create query templates for common patterns',
            'Implement auto-complete suggestions',
            'Add query expansion for related terms'
          ],
          supportingData: { frequency: 45, users: 12 },
          generatedAt: new Date().toISOString()
        },
        {
          id: 'insight-2',
          insightType: 'warning',
          analyticsType: 'performance',
          title: 'Cache Hit Rate Below Optimal',
          description: 'Cache hit rate is 78%, below the optimal threshold of 85%',
          impactScore: 0.7,
          confidence: 0.8,
          actionableRecommendations: [
            'Review cache configuration and size',
            'Analyze cache eviction policies',
            'Consider warming up cache with popular queries'
          ],
          supportingData: { currentRate: 0.78, targetRate: 0.85 },
          generatedAt: new Date().toISOString()
        },
        {
          id: 'insight-3',
          insightType: 'recommendation',
          analyticsType: 'knowledge_base',
          title: 'Knowledge Gap in Advanced Topics',
          description: 'Zero-result queries indicate gaps in advanced technical content',
          impactScore: 0.95,
          confidence: 0.85,
          actionableRecommendations: [
            'Expand knowledge base coverage for advanced topics',
            'Implement fuzzy search capabilities',
            'Add query suggestion system'
          ],
          supportingData: { zeroResultQueries: 23, affectedUsers: 8 },
          generatedAt: new Date().toISOString()
        }
      ];

      mockSummary.topInsights = mockInsights;

      setAnalyticsSummary(mockSummary);
      setSearchPatterns(mockPatterns);
      setInsights(mockInsights);

    } catch (error) {
      console.error('Failed to load analytics data:', error);
      toast.error('Failed to load analytics data');
    } finally {
      setIsLoading(false);
    }
  }, [selectedTimeRange]);

  // Refresh analytics
  const refreshAnalytics = useCallback(async () => {
    await loadAnalyticsData();
    toast.success('Analytics data refreshed');
  }, [loadAnalyticsData]);

  // Filter insights
  const filteredInsights = insights.filter(insight => 
    selectedInsightType === 'all' || insight.insightType === selectedInsightType
  );

  // Initialize data
  useEffect(() => {
    loadAnalyticsData();
  }, [loadAnalyticsData]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading analytics data...</p>
        </div>
      </div>
    );
  }

  if (!analyticsSummary) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">No analytics data available</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl flex items-center gap-2">
                <BarChart3 className="h-6 w-6" />
                Advanced Analytics Dashboard
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Comprehensive insights and optimization recommendations
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1d">Last 24h</SelectItem>
                  <SelectItem value="7d">Last 7 days</SelectItem>
                  <SelectItem value="30d">Last 30 days</SelectItem>
                  <SelectItem value="90d">Last 90 days</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" onClick={refreshAnalytics}>
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="patterns">Search Patterns</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Insights</p>
                    <p className="text-2xl font-bold">{analyticsSummary.totalInsights}</p>
                  </div>
                  <Brain className="h-8 w-8 text-purple-500" />
                </div>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <ArrowUp className="h-3 w-3 mr-1" />
                    {analyticsSummary.highImpactInsights} high impact
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Search Events</p>
                    <p className="text-2xl font-bold">{analyticsSummary.searchAnalytics.totalEvents}</p>
                  </div>
                  <Search className="h-8 w-8 text-blue-500" />
                </div>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <Activity className="h-3 w-3 mr-1" />
                    {analyticsSummary.searchAnalytics.recentEvents} recent
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">User Satisfaction</p>
                    <p className="text-2xl font-bold">
                      {(analyticsSummary.performanceMetrics.userSatisfaction * 100).toFixed(0)}%
                    </p>
                  </div>
                  <ThumbsUp className="h-8 w-8 text-green-500" />
                </div>
                <Progress value={analyticsSummary.performanceMetrics.userSatisfaction * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">System Efficiency</p>
                    <p className="text-2xl font-bold">
                      {(analyticsSummary.performanceMetrics.systemEfficiency * 100).toFixed(0)}%
                    </p>
                  </div>
                  <Cpu className="h-8 w-8 text-orange-500" />
                </div>
                <Progress value={analyticsSummary.performanceMetrics.systemEfficiency * 100} className="mt-2" />
              </CardContent>
            </Card>
          </div>

          {/* Performance Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Average Response Time</span>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">
                        {analyticsSummary.performanceMetrics.avgResponseTime.toFixed(2)}s
                      </span>
                      <Timer className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Cache Hit Rate</span>
                    <div className="flex items-center gap-2">
                      <Progress 
                        value={analyticsSummary.performanceMetrics.cacheHitRate * 100} 
                        className="w-20 h-2" 
                      />
                      <span className="text-sm font-medium">
                        {(analyticsSummary.performanceMetrics.cacheHitRate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Search Success Rate</span>
                    <div className="flex items-center gap-2">
                      <Progress 
                        value={analyticsSummary.performanceMetrics.searchSuccessRate * 100} 
                        className="w-20 h-2" 
                      />
                      <span className="text-sm font-medium">
                        {(analyticsSummary.performanceMetrics.searchSuccessRate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Knowledge Coverage</span>
                    <div className="flex items-center gap-2">
                      <Progress 
                        value={analyticsSummary.performanceMetrics.knowledgeCoverage * 100} 
                        className="w-20 h-2" 
                      />
                      <span className="text-sm font-medium">
                        {(analyticsSummary.performanceMetrics.knowledgeCoverage * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Insight Categories</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(analyticsSummary.insightCategories).map(([category, count]) => (
                    <div key={category} className="flex justify-between items-center">
                      <span className="text-sm capitalize">{category.replace('_', ' ')}</span>
                      <div className="flex items-center gap-2">
                        <Progress 
                          value={(count / analyticsSummary.totalInsights) * 100} 
                          className="w-20 h-2" 
                        />
                        <span className="text-sm font-medium">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Top Insights */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Top Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analyticsSummary.topInsights.slice(0, 3).map((insight) => (
                  <div key={insight.id} className="flex items-start gap-3 p-3 border rounded">
                    <div className={`p-2 rounded ${insightTypeConfig[insight.insightType].bg}`}>
                      {insightTypeConfig[insight.insightType].icon}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium">{insight.title}</h3>
                        <Badge variant="outline" className="text-xs">
                          Impact: {(insight.impactScore * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        {insight.description}
                      </p>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="text-xs">
                          {insightTypeConfig[insight.insightType].label}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {insight.analyticsType.replace('_', ' ')}
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="text-xs text-muted-foreground">
                      {new Date(insight.generatedAt).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Search Patterns Tab */}
        <TabsContent value="patterns" className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Search Patterns Analysis</h2>
            <Badge variant="outline">
              {searchPatterns.length} patterns identified
            </Badge>
          </div>

          <div className="space-y-4">
            {searchPatterns.map((pattern) => (
              <Card key={pattern.patternId}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium">{pattern.description}</h3>
                        <Badge variant="outline" className="text-xs">
                          {pattern.patternType.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span>Frequency: {pattern.frequency}</span>
                        <span>Users: {pattern.usersAffected}</span>
                        <span>Satisfaction: {(pattern.avgSatisfaction * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-sm font-medium mb-1">
                        Optimization Potential
                      </div>
                      <div className="flex items-center gap-2">
                        <Progress value={pattern.optimizationPotential * 100} className="w-20 h-2" />
                        <span className="text-sm">{(pattern.optimizationPotential * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                  
                  {pattern.examples.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-muted-foreground mb-2">Example queries:</p>
                      <div className="flex flex-wrap gap-1">
                        {pattern.examples.map((example, index) => (
                          <Badge key={index} variant="secondary" className="text-xs">
                            "{example}"
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <h2 className="text-lg font-semibold">Performance Analytics</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Response Time</p>
                    <p className="text-2xl font-bold">
                      {analyticsSummary.performanceMetrics.avgResponseTime.toFixed(2)}s
                    </p>
                  </div>
                  <Timer className="h-8 w-8 text-blue-500" />
                </div>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    P95: {analyticsSummary.performanceMetrics.p95ResponseTime.toFixed(2)}s
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Cache Hit Rate</p>
                    <p className="text-2xl font-bold">
                      {(analyticsSummary.performanceMetrics.cacheHitRate * 100).toFixed(0)}%
                    </p>
                  </div>
                  <Database className="h-8 w-8 text-green-500" />
                </div>
                <Progress value={analyticsSummary.performanceMetrics.cacheHitRate * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Search Success</p>
                    <p className="text-2xl font-bold">
                      {(analyticsSummary.performanceMetrics.searchSuccessRate * 100).toFixed(0)}%
                    </p>
                  </div>
                  <Target className="h-8 w-8 text-purple-500" />
                </div>
                <Progress value={analyticsSummary.performanceMetrics.searchSuccessRate * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Query Complexity</p>
                    <p className="text-2xl font-bold">
                      {(analyticsSummary.performanceMetrics.queryComplexityScore * 100).toFixed(0)}%
                    </p>
                  </div>
                  <Brain className="h-8 w-8 text-orange-500" />
                </div>
                <Progress value={analyticsSummary.performanceMetrics.queryComplexityScore * 100} className="mt-2" />
              </CardContent>
            </Card>
          </div>

          {/* Performance Trends Placeholder */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Performance Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center border-2 border-dashed border-muted rounded">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">Performance trend charts would be displayed here</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Analytics Insights</h2>
            <Select value={selectedInsightType} onValueChange={setSelectedInsightType}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Insights</SelectItem>
                <SelectItem value="optimization">Optimization</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="recommendation">Recommendation</SelectItem>
                <SelectItem value="trend">Trend</SelectItem>
                <SelectItem value="anomaly">Anomaly</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-4">
            {filteredInsights.map((insight) => (
              <Card key={insight.id}>
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded ${insightTypeConfig[insight.insightType].bg}`}>
                      {insightTypeConfig[insight.insightType].icon}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium">{insight.title}</h3>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            Impact: {(insight.impactScore * 100).toFixed(0)}%
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Confidence: {(insight.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mb-3">
                        {insight.description}
                      </p>
                      
                      <div className="mb-3">
                        <p className="text-xs font-medium mb-2">Recommended Actions:</p>
                        <ul className="text-xs text-muted-foreground space-y-1">
                          {insight.actionableRecommendations.map((recommendation, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <span className="text-purple-500">•</span>
                              {recommendation}
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="text-xs">
                          {insightTypeConfig[insight.insightType].label}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {insight.analyticsType.replace('_', ' ')}
                        </Badge>
                        <span className="text-xs text-muted-foreground ml-auto">
                          {new Date(insight.generatedAt).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Optimization Tab */}
        <TabsContent value="optimization" className="space-y-4">
          <h2 className="text-lg font-semibold">Query Optimization</h2>

          <Card>
            <CardContent className="p-8 text-center">
              <Zap className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Query Optimization Engine</h3>
              <p className="text-muted-foreground mb-4">
                Advanced query optimization recommendations and improvements
              </p>
              <Button>
                <Settings className="h-4 w-4 mr-2" />
                Configure Optimization Rules
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdvancedAnalyticsDashboard;
