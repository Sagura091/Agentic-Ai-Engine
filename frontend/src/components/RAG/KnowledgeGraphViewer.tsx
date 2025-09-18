import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Network, 
  Share2, 
  Search, 
  Filter, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw,
  Download,
  Settings,
  Eye,
  EyeOff,
  Layers,
  Target,
  GitBranch,
  Users,
  MapPin,
  Briefcase,
  Lightbulb,
  FileText,
  Tag,
  HelpCircle
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
import { toast } from 'sonner';

// Types for knowledge graph
interface GraphNode {
  id: string;
  name: string;
  type: string;
  description?: string;
  properties?: Record<string, any>;
  confidence: number;
  x?: number;
  y?: number;
  size?: number;
  color?: string;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
  confidence: number;
  properties?: Record<string, any>;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    total_entities: number;
    total_relationships: number;
    entity_types: Record<string, number>;
    relationship_types: Record<string, number>;
  };
}

interface GraphPath {
  entities: GraphNode[];
  relationships: GraphEdge[];
  total_weight: number;
  confidence: number;
  path_length: number;
}

const KnowledgeGraphViewer: React.FC = () => {
  // Graph data state
  const [graphData, setGraphData] = useState<GraphData>({
    nodes: [],
    edges: [],
    stats: {
      total_entities: 0,
      total_relationships: 0,
      entity_types: {},
      relationship_types: {}
    }
  });

  // UI state
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredNodes, setFilteredNodes] = useState<GraphNode[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Visualization settings
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force');
  const [showLabels, setShowLabels] = useState(true);
  const [showEdgeLabels, setShowEdgeLabels] = useState(false);
  const [nodeSize, setNodeSize] = useState([10]);
  const [edgeThickness, setEdgeThickness] = useState([2]);
  const [zoomLevel, setZoomLevel] = useState([1]);

  // Filters
  const [entityTypeFilter, setEntityTypeFilter] = useState('all');
  const [relationshipTypeFilter, setRelationshipTypeFilter] = useState('all');
  const [confidenceThreshold, setConfidenceThreshold] = useState([0.5]);

  // Path finding
  const [pathSource, setPathSource] = useState('');
  const [pathTarget, setPathTarget] = useState('');
  const [foundPath, setFoundPath] = useState<GraphPath | null>(null);

  // Canvas ref for graph visualization
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Entity type configurations
  const entityTypeConfig = {
    person: { icon: <Users className="h-4 w-4" />, color: '#3B82F6', label: 'Person' },
    organization: { icon: <Briefcase className="h-4 w-4" />, color: '#10B981', label: 'Organization' },
    location: { icon: <MapPin className="h-4 w-4" />, color: '#F59E0B', label: 'Location' },
    concept: { icon: <Lightbulb className="h-4 w-4" />, color: '#8B5CF6', label: 'Concept' },
    technology: { icon: <Settings className="h-4 w-4" />, color: '#EF4444', label: 'Technology' },
    document: { icon: <FileText className="h-4 w-4" />, color: '#6B7280', label: 'Document' },
    keyword: { icon: <Tag className="h-4 w-4" />, color: '#EC4899', label: 'Keyword' },
    unknown: { icon: <HelpCircle className="h-4 w-4" />, color: '#9CA3AF', label: 'Unknown' }
  };

  // Load graph data
  const loadGraphData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Simulate API call - replace with actual API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock graph data
      const mockNodes: GraphNode[] = [
        {
          id: '1',
          name: 'Artificial Intelligence',
          type: 'concept',
          description: 'The simulation of human intelligence in machines',
          confidence: 0.95,
          properties: { importance: 'high' }
        },
        {
          id: '2',
          name: 'Machine Learning',
          type: 'concept',
          description: 'A subset of AI that enables machines to learn',
          confidence: 0.92,
          properties: { importance: 'high' }
        },
        {
          id: '3',
          name: 'Neural Networks',
          type: 'technology',
          description: 'Computing systems inspired by biological neural networks',
          confidence: 0.88,
          properties: { complexity: 'high' }
        },
        {
          id: '4',
          name: 'Deep Learning',
          type: 'concept',
          description: 'Machine learning using deep neural networks',
          confidence: 0.90,
          properties: { importance: 'high' }
        },
        {
          id: '5',
          name: 'TensorFlow',
          type: 'technology',
          description: 'Open-source machine learning framework',
          confidence: 0.85,
          properties: { type: 'framework' }
        },
        {
          id: '6',
          name: 'Google',
          type: 'organization',
          description: 'Technology company that developed TensorFlow',
          confidence: 0.98,
          properties: { industry: 'technology' }
        }
      ];

      const mockEdges: GraphEdge[] = [
        {
          id: 'e1',
          source: '2',
          target: '1',
          type: 'part_of',
          weight: 0.9,
          confidence: 0.95
        },
        {
          id: 'e2',
          source: '4',
          target: '2',
          type: 'part_of',
          weight: 0.8,
          confidence: 0.90
        },
        {
          id: 'e3',
          source: '3',
          target: '4',
          type: 'used_in',
          weight: 0.85,
          confidence: 0.88
        },
        {
          id: 'e4',
          source: '5',
          target: '2',
          type: 'implements',
          weight: 0.7,
          confidence: 0.85
        },
        {
          id: 'e5',
          source: '6',
          target: '5',
          type: 'created_by',
          weight: 0.95,
          confidence: 0.98
        }
      ];

      const mockStats = {
        total_entities: mockNodes.length,
        total_relationships: mockEdges.length,
        entity_types: {
          concept: 3,
          technology: 2,
          organization: 1
        },
        relationship_types: {
          part_of: 2,
          used_in: 1,
          implements: 1,
          created_by: 1
        }
      };

      setGraphData({
        nodes: mockNodes,
        edges: mockEdges,
        stats: mockStats
      });

      toast.success('Knowledge graph loaded successfully');

    } catch (error) {
      console.error('Failed to load graph data:', error);
      toast.error('Failed to load knowledge graph');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Search entities
  const searchEntities = useCallback(async (query: string) => {
    if (!query.trim()) {
      setFilteredNodes([]);
      return;
    }

    const filtered = graphData.nodes.filter(node =>
      node.name.toLowerCase().includes(query.toLowerCase()) ||
      (node.description && node.description.toLowerCase().includes(query.toLowerCase()))
    );

    setFilteredNodes(filtered);
  }, [graphData.nodes]);

  // Find path between entities
  const findPath = useCallback(async () => {
    if (!pathSource || !pathTarget) {
      toast.error('Please select both source and target entities');
      return;
    }

    try {
      // Simulate path finding - replace with actual API
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mock path result
      const mockPath: GraphPath = {
        entities: [
          graphData.nodes.find(n => n.id === pathSource)!,
          graphData.nodes.find(n => n.id === pathTarget)!
        ],
        relationships: [
          graphData.edges.find(e => e.source === pathSource && e.target === pathTarget) ||
          graphData.edges.find(e => e.source === pathTarget && e.target === pathSource)!
        ],
        total_weight: 0.8,
        confidence: 0.85,
        path_length: 1
      };

      setFoundPath(mockPath);
      toast.success(`Found path with ${mockPath.path_length} hop(s)`);

    } catch (error) {
      console.error('Path finding failed:', error);
      toast.error('Failed to find path between entities');
    }
  }, [pathSource, pathTarget, graphData]);

  // Filter nodes by type and confidence
  const getFilteredNodes = useCallback(() => {
    return graphData.nodes.filter(node => {
      const typeMatch = entityTypeFilter === 'all' || node.type === entityTypeFilter;
      const confidenceMatch = node.confidence >= confidenceThreshold[0];
      return typeMatch && confidenceMatch;
    });
  }, [graphData.nodes, entityTypeFilter, confidenceThreshold]);

  // Filter edges by type and confidence
  const getFilteredEdges = useCallback(() => {
    const filteredNodes = getFilteredNodes();
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    
    return graphData.edges.filter(edge => {
      const typeMatch = relationshipTypeFilter === 'all' || edge.type === relationshipTypeFilter;
      const confidenceMatch = edge.confidence >= confidenceThreshold[0];
      const nodesExist = nodeIds.has(edge.source) && nodeIds.has(edge.target);
      return typeMatch && confidenceMatch && nodesExist;
    });
  }, [graphData.edges, relationshipTypeFilter, confidenceThreshold, getFilteredNodes]);

  // Initialize graph
  useEffect(() => {
    loadGraphData();
  }, [loadGraphData]);

  // Update search results
  useEffect(() => {
    searchEntities(searchQuery);
  }, [searchQuery, searchEntities]);

  // Export graph data
  const exportGraph = () => {
    const exportData = {
      nodes: getFilteredNodes(),
      edges: getFilteredEdges(),
      stats: graphData.stats,
      exported_at: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `knowledge_graph_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);

    toast.success('Graph data exported successfully');
  };

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Network className="h-6 w-6 text-purple-500" />
              <div>
                <CardTitle className="text-xl">Knowledge Graph Viewer</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Interactive visualization of entity relationships and knowledge connections
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <GitBranch className="h-3 w-3" />
                {graphData.stats.total_entities} entities
              </Badge>
              <Badge variant="outline" className="flex items-center gap-1">
                <Share2 className="h-3 w-3" />
                {graphData.stats.total_relationships} relationships
              </Badge>
              <Button variant="outline" size="sm" onClick={exportGraph}>
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-[calc(100vh-200px)]">
        {/* Controls Panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Search */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Search Entities</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search entities..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              
              {filteredNodes.length > 0 && (
                <div className="max-h-32 overflow-y-auto space-y-1">
                  {filteredNodes.map((node) => (
                    <div
                      key={node.id}
                      className="p-2 rounded border cursor-pointer hover:bg-muted"
                      onClick={() => setSelectedNode(node)}
                    >
                      <div className="flex items-center gap-2">
                        {entityTypeConfig[node.type as keyof typeof entityTypeConfig]?.icon}
                        <span className="text-sm font-medium">{node.name}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Filters */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Filters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-xs font-medium mb-2 block">Entity Type</label>
                <Select value={entityTypeFilter} onValueChange={setEntityTypeFilter}>
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    {Object.entries(entityTypeConfig).map(([type, config]) => (
                      <SelectItem key={type} value={type}>
                        <div className="flex items-center gap-2">
                          {config.icon}
                          {config.label}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-xs font-medium mb-2 block">Relationship Type</label>
                <Select value={relationshipTypeFilter} onValueChange={setRelationshipTypeFilter}>
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="part_of">Part Of</SelectItem>
                    <SelectItem value="related_to">Related To</SelectItem>
                    <SelectItem value="created_by">Created By</SelectItem>
                    <SelectItem value="used_in">Used In</SelectItem>
                    <SelectItem value="implements">Implements</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-xs font-medium mb-2 block">
                  Confidence Threshold: {confidenceThreshold[0].toFixed(2)}
                </label>
                <Slider
                  value={confidenceThreshold}
                  onValueChange={setConfidenceThreshold}
                  max={1}
                  min={0}
                  step={0.1}
                  className="w-full"
                />
              </div>
            </CardContent>
          </Card>

          {/* Visualization Settings */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Visualization</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-xs font-medium mb-2 block">Layout Algorithm</label>
                <Select value={layoutAlgorithm} onValueChange={setLayoutAlgorithm}>
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="force">Force-Directed</SelectItem>
                    <SelectItem value="hierarchical">Hierarchical</SelectItem>
                    <SelectItem value="circular">Circular</SelectItem>
                    <SelectItem value="grid">Grid</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-xs font-medium">Show Labels</label>
                <Switch checked={showLabels} onCheckedChange={setShowLabels} />
              </div>

              <div className="flex items-center justify-between">
                <label className="text-xs font-medium">Show Edge Labels</label>
                <Switch checked={showEdgeLabels} onCheckedChange={setShowEdgeLabels} />
              </div>

              <div>
                <label className="text-xs font-medium mb-2 block">
                  Node Size: {nodeSize[0]}
                </label>
                <Slider
                  value={nodeSize}
                  onValueChange={setNodeSize}
                  max={20}
                  min={5}
                  step={1}
                  className="w-full"
                />
              </div>
            </CardContent>
          </Card>

          {/* Path Finding */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Path Finding</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <label className="text-xs font-medium mb-1 block">Source Entity</label>
                <Select value={pathSource} onValueChange={setPathSource}>
                  <SelectTrigger className="h-8">
                    <SelectValue placeholder="Select source" />
                  </SelectTrigger>
                  <SelectContent>
                    {graphData.nodes.map((node) => (
                      <SelectItem key={node.id} value={node.id}>
                        {node.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-xs font-medium mb-1 block">Target Entity</label>
                <Select value={pathTarget} onValueChange={setPathTarget}>
                  <SelectTrigger className="h-8">
                    <SelectValue placeholder="Select target" />
                  </SelectTrigger>
                  <SelectContent>
                    {graphData.nodes.map((node) => (
                      <SelectItem key={node.id} value={node.id}>
                        {node.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Button onClick={findPath} size="sm" className="w-full">
                <Target className="h-4 w-4 mr-2" />
                Find Path
              </Button>

              {foundPath && (
                <div className="p-2 bg-muted rounded text-xs">
                  <div className="font-medium">Path Found:</div>
                  <div>Length: {foundPath.path_length} hop(s)</div>
                  <div>Confidence: {(foundPath.confidence * 100).toFixed(1)}%</div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Graph Visualization */}
        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Graph Visualization</CardTitle>
                <div className="flex items-center space-x-2">
                  <Button variant="outline" size="sm">
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="h-[calc(100%-80px)]">
              <div
                ref={containerRef}
                className="w-full h-full border rounded bg-muted/10 flex items-center justify-center"
              >
                {isLoading ? (
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mx-auto mb-2"></div>
                    <p className="text-sm text-muted-foreground">Loading knowledge graph...</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <Network className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                    <p className="text-sm text-muted-foreground">
                      Interactive graph visualization will be rendered here
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      {getFilteredNodes().length} nodes, {getFilteredEdges().length} edges visible
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Details Panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Selected Entity/Relationship */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Details</CardTitle>
            </CardHeader>
            <CardContent>
              {selectedNode ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    {entityTypeConfig[selectedNode.type as keyof typeof entityTypeConfig]?.icon}
                    <span className="font-medium">{selectedNode.name}</span>
                  </div>
                  
                  <div>
                    <Badge variant="outline" className="text-xs">
                      {entityTypeConfig[selectedNode.type as keyof typeof entityTypeConfig]?.label}
                    </Badge>
                  </div>
                  
                  {selectedNode.description && (
                    <p className="text-sm text-muted-foreground">
                      {selectedNode.description}
                    </p>
                  )}
                  
                  <div className="text-xs">
                    <div>Confidence: {(selectedNode.confidence * 100).toFixed(1)}%</div>
                  </div>
                  
                  {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
                    <div>
                      <div className="text-xs font-medium mb-1">Properties:</div>
                      <div className="space-y-1">
                        {Object.entries(selectedNode.properties).map(([key, value]) => (
                          <div key={key} className="text-xs">
                            <span className="font-medium">{key}:</span> {String(value)}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Select a node or edge to view details
                </p>
              )}
            </CardContent>
          </Card>

          {/* Graph Statistics */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Graph Statistics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <div className="font-medium">Entities</div>
                  <div className="text-muted-foreground">{graphData.stats.total_entities}</div>
                </div>
                <div>
                  <div className="font-medium">Relationships</div>
                  <div className="text-muted-foreground">{graphData.stats.total_relationships}</div>
                </div>
              </div>
              
              <Separator />
              
              <div>
                <div className="text-xs font-medium mb-2">Entity Types</div>
                <div className="space-y-1">
                  {Object.entries(graphData.stats.entity_types).map(([type, count]) => (
                    <div key={type} className="flex justify-between text-xs">
                      <span className="flex items-center gap-1">
                        {entityTypeConfig[type as keyof typeof entityTypeConfig]?.icon}
                        {entityTypeConfig[type as keyof typeof entityTypeConfig]?.label}
                      </span>
                      <span>{count}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <Separator />
              
              <div>
                <div className="text-xs font-medium mb-2">Relationship Types</div>
                <div className="space-y-1">
                  {Object.entries(graphData.stats.relationship_types).map(([type, count]) => (
                    <div key={type} className="flex justify-between text-xs">
                      <span>{type.replace('_', ' ')}</span>
                      <span>{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeGraphViewer;
