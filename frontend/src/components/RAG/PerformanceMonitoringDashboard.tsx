import React, { useState, useEffect, useCallback } from 'react';
import { 
  Activity, 
  BarChart3, 
  TrendingUp, 
  Zap, 
  Clock, 
  Database, 
  Search,
  Brain,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Download,
  Settings,
  Eye,
  Target,
  Layers,
  Globe,
  Users,
  Server,
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
import { Switch } from '@/components/ui/switch';
import { toast } from 'sonner';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

// Types for monitoring data
interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: number;
  active_connections: number;
  response_time: number;
}

interface RAGMetrics {
  total_searches: number;
  avg_response_time: number;
  cache_hit_rate: number;
  embedding_generation_time: number;
  index_size: number;
  active_collections: number;
}

interface SearchAnalytics {
  popular_queries: Array<{ query: string; count: number }>;
  search_patterns: Array<{ time: string; searches: number }>;
  success_rate: number;
  avg_results_per_query: number;
}

interface ComponentHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'critical';
  uptime: number;
  error_rate: number;
  last_check: string;
}

const PerformanceMonitoringDashboard: React.FC = () => {
  // State for metrics and analytics
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 0,
    network_io: 0,
    active_connections: 0,
    response_time: 0
  });

  const [ragMetrics, setRAGMetrics] = useState<RAGMetrics>({
    total_searches: 0,
    avg_response_time: 0,
    cache_hit_rate: 0,
    embedding_generation_time: 0,
    index_size: 0,
    active_collections: 0
  });

  const [searchAnalytics, setSearchAnalytics] = useState<SearchAnalytics>({
    popular_queries: [],
    search_patterns: [],
    success_rate: 0,
    avg_results_per_query: 0
  });

  const [componentHealth, setComponentHealth] = useState<ComponentHealth[]>([]);
  
  // UI state
  const [isRealTime, setIsRealTime] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState('5');
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  const [isLoading, setIsLoading] = useState(false);

  // Mock data for demonstration
  const mockSystemData = [
    { time: '00:00', cpu: 45, memory: 62, searches: 120 },
    { time: '00:05', cpu: 52, memory: 65, searches: 145 },
    { time: '00:10', cpu: 48, memory: 63, searches: 132 },
    { time: '00:15', cpu: 55, memory: 68, searches: 167 },
    { time: '00:20', cpu: 43, memory: 61, searches: 98 },
    { time: '00:25', cpu: 49, memory: 64, searches: 156 }
  ];

  const mockSearchModes = [
    { name: 'Semantic', value: 45, color: '#8884d8' },
    { name: 'Hybrid', value: 30, color: '#82ca9d' },
    { name: 'Neural', value: 15, color: '#ffc658' },
    { name: 'Graph', value: 10, color: '#ff7300' }
  ];

  // Fetch metrics from API
  const fetchMetrics = useCallback(async () => {
    setIsLoading(true);
    try {
      // Simulate API calls - replace with actual API endpoints
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mock data updates
      setSystemMetrics({
        cpu_usage: Math.random() * 100,
        memory_usage: Math.random() * 100,
        disk_usage: Math.random() * 100,
        network_io: Math.random() * 1000,
        active_connections: Math.floor(Math.random() * 500),
        response_time: Math.random() * 200
      });

      setRAGMetrics({
        total_searches: Math.floor(Math.random() * 10000),
        avg_response_time: Math.random() * 500,
        cache_hit_rate: Math.random() * 100,
        embedding_generation_time: Math.random() * 100,
        index_size: Math.random() * 1000,
        active_collections: Math.floor(Math.random() * 50)
      });

      setSearchAnalytics({
        popular_queries: [
          { query: 'AI architecture', count: 245 },
          { query: 'machine learning', count: 189 },
          { query: 'neural networks', count: 156 },
          { query: 'deep learning', count: 134 },
          { query: 'data science', count: 98 }
        ],
        search_patterns: mockSystemData,
        success_rate: 95 + Math.random() * 5,
        avg_results_per_query: 8 + Math.random() * 4
      });

      setComponentHealth([
        {
          name: 'Search Engine',
          status: 'healthy',
          uptime: 99.9,
          error_rate: 0.1,
          last_check: new Date().toISOString()
        },
        {
          name: 'Vector Store',
          status: 'healthy',
          uptime: 99.8,
          error_rate: 0.2,
          last_check: new Date().toISOString()
        },
        {
          name: 'Embedding Service',
          status: 'degraded',
          uptime: 98.5,
          error_rate: 1.5,
          last_check: new Date().toISOString()
        },
        {
          name: 'Cache System',
          status: 'healthy',
          uptime: 99.9,
          error_rate: 0.1,
          last_check: new Date().toISOString()
        }
      ]);

    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      toast.error('Failed to fetch monitoring data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Real-time updates
  useEffect(() => {
    fetchMetrics();
    
    if (isRealTime) {
      const interval = setInterval(fetchMetrics, parseInt(refreshInterval) * 1000);
      return () => clearInterval(interval);
    }
  }, [fetchMetrics, isRealTime, refreshInterval]);

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'degraded': return 'text-yellow-600 bg-yellow-100';
      case 'unhealthy': return 'text-orange-600 bg-orange-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4" />;
      case 'degraded': return <AlertTriangle className="h-4 w-4" />;
      case 'unhealthy': return <XCircle className="h-4 w-4" />;
      case 'critical': return <XCircle className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  // Export metrics data
  const exportMetrics = () => {
    const data = {
      timestamp: new Date().toISOString(),
      system_metrics: systemMetrics,
      rag_metrics: ragMetrics,
      search_analytics: searchAnalytics,
      component_health: componentHealth
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rag_metrics_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success('Metrics exported successfully');
  };

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="h-6 w-6 text-blue-500" />
              <div>
                <CardTitle className="text-xl">RAG Performance Monitoring</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Real-time analytics and system health monitoring
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={isRealTime}
                  onCheckedChange={setIsRealTime}
                  id="real-time"
                />
                <label htmlFor="real-time" className="text-sm">Real-time</label>
              </div>
              <Select value={refreshInterval} onValueChange={setRefreshInterval}>
                <SelectTrigger className="w-20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1s</SelectItem>
                  <SelectItem value="5">5s</SelectItem>
                  <SelectItem value="10">10s</SelectItem>
                  <SelectItem value="30">30s</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" onClick={fetchMetrics} disabled={isLoading}>
                <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
              <Button variant="outline" size="sm" onClick={exportMetrics}>
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Searches</p>
                <p className="text-2xl font-bold">{ragMetrics.total_searches.toLocaleString()}</p>
              </div>
              <Search className="h-8 w-8 text-blue-500" />
            </div>
            <div className="mt-2">
              <Badge variant="secondary" className="text-xs">
                <TrendingUp className="h-3 w-3 mr-1" />
                +12% from last hour
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Response Time</p>
                <p className="text-2xl font-bold">{ragMetrics.avg_response_time.toFixed(0)}ms</p>
              </div>
              <Timer className="h-8 w-8 text-green-500" />
            </div>
            <div className="mt-2">
              <Progress value={Math.max(0, 100 - ragMetrics.avg_response_time / 5)} className="h-2" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Cache Hit Rate</p>
                <p className="text-2xl font-bold">{ragMetrics.cache_hit_rate.toFixed(1)}%</p>
              </div>
              <Database className="h-8 w-8 text-purple-500" />
            </div>
            <div className="mt-2">
              <Progress value={ragMetrics.cache_hit_rate} className="h-2" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">System Health</p>
                <p className="text-2xl font-bold text-green-600">Healthy</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
            <div className="mt-2">
              <Badge variant="outline" className="text-xs">
                All systems operational
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analytics */}
      <Tabs defaultValue="performance" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="search">Search Analytics</TabsTrigger>
          <TabsTrigger value="system">System Health</TabsTrigger>
          <TabsTrigger value="components">Components</TabsTrigger>
        </TabsList>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Response Time Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={mockSystemData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="searches" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Search Mode Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={mockSearchModes}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {mockSearchModes.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">System Resource Usage</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={mockSystemData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="cpu" stackId="1" stroke="#8884d8" fill="#8884d8" />
                  <Area type="monotone" dataKey="memory" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Search Analytics Tab */}
        <TabsContent value="search" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Popular Search Queries</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {searchAnalytics.popular_queries.map((query, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm">{query.query}</span>
                      <div className="flex items-center gap-2">
                        <Progress value={(query.count / 250) * 100} className="w-20 h-2" />
                        <span className="text-xs text-muted-foreground">{query.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Search Success Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Success Rate</span>
                      <span>{searchAnalytics.success_rate.toFixed(1)}%</span>
                    </div>
                    <Progress value={searchAnalytics.success_rate} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Avg Results per Query</span>
                      <span>{searchAnalytics.avg_results_per_query.toFixed(1)}</span>
                    </div>
                    <Progress value={(searchAnalytics.avg_results_per_query / 20) * 100} className="h-2" />
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {ragMetrics.active_collections}
                      </div>
                      <div className="text-sm text-muted-foreground">Active Collections</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {ragMetrics.index_size.toFixed(0)}MB
                      </div>
                      <div className="text-sm text-muted-foreground">Index Size</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* System Health Tab */}
        <TabsContent value="system" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">CPU Usage</span>
                  <Cpu className="h-4 w-4 text-blue-500" />
                </div>
                <div className="text-2xl font-bold mb-2">{systemMetrics.cpu_usage.toFixed(1)}%</div>
                <Progress value={systemMetrics.cpu_usage} className="h-2" />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Memory Usage</span>
                  <HardDrive className="h-4 w-4 text-green-500" />
                </div>
                <div className="text-2xl font-bold mb-2">{systemMetrics.memory_usage.toFixed(1)}%</div>
                <Progress value={systemMetrics.memory_usage} className="h-2" />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Network I/O</span>
                  <Network className="h-4 w-4 text-purple-500" />
                </div>
                <div className="text-2xl font-bold mb-2">{systemMetrics.network_io.toFixed(0)} MB/s</div>
                <Progress value={(systemMetrics.network_io / 1000) * 100} className="h-2" />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">System Performance Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={mockSystemData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="cpu" stroke="#8884d8" strokeWidth={2} />
                  <Line type="monotone" dataKey="memory" stroke="#82ca9d" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Components Tab */}
        <TabsContent value="components" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {componentHealth.map((component, index) => (
              <Card key={index}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold">{component.name}</h3>
                    <Badge className={getStatusColor(component.status)}>
                      {getStatusIcon(component.status)}
                      <span className="ml-1 capitalize">{component.status}</span>
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Uptime</span>
                      <span>{component.uptime}%</span>
                    </div>
                    <Progress value={component.uptime} className="h-2" />
                    
                    <div className="flex justify-between text-sm">
                      <span>Error Rate</span>
                      <span>{component.error_rate}%</span>
                    </div>
                    <Progress value={component.error_rate} className="h-2" />
                    
                    <div className="text-xs text-muted-foreground mt-2">
                      Last check: {new Date(component.last_check).toLocaleTimeString()}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PerformanceMonitoringDashboard;
