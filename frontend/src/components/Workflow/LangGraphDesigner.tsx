import React, { useState, useCallback, useRef, useEffect } from 'react'
import {
  GitBranch,
  Plus,
  Play,
  Square,
  Diamond,
  Circle,
  ArrowRight,
  Settings,
  Save,
  Download,
  Upload,
  Trash2,
  Copy,
  Brain,
  Activity,
  CheckCircle,
  AlertTriangle,
  Clock,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize,
  Zap,
  BarChart3,
  MessageSquare,
  Bot,
  Users,
  Eye,
  Layers,
  Network,
  Cpu,
  Database,
  Globe,
  Target,
  TrendingUp,
  Pause,
  SkipForward,
  Rewind,
  Monitor,
  Code,
  FileText,
  Workflow,
  Share2,
  Lock,
  Unlock,
  Timer,
  Gauge
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useQuery, useMutation } from 'react-query';
import { enhancedOrchestrationApi, workflowApi } from '../../services/api';
import toast from 'react-hot-toast';

interface WorkflowNode {
  id: string
  type: 'start' | 'end' | 'agent' | 'decision' | 'subgraph' | 'supervisor' | 'tool' | 'knowledge_base' | 'human_input' | 'conditional_router'
  position: { x: number; y: number }
  data: {
    label: string
    agentType?: string
    agentId?: string
    model?: string
    description?: string
    condition?: string
    subgraphType?: 'research_team' | 'document_team' | 'analysis_team' | 'creative_team' | 'validation_team'
    supervisorConfig?: {
      maxWorkers: number
      strategy: 'parallel' | 'sequential' | 'conditional' | 'dynamic'
      failureHandling: 'retry' | 'skip' | 'abort'
      timeout: number
    }
    toolConfig?: {
      toolId: string
      parameters: Record<string, any>
      retryCount: number
    }
    knowledgeBaseConfig?: {
      knowledgeBaseId: string
      searchType: 'semantic' | 'keyword' | 'hybrid'
      maxResults: number
    }
    humanInputConfig?: {
      prompt: string
      timeout: number
      required: boolean
    }
    routingConfig?: {
      conditions: Array<{
        condition: string
        targetNodeId: string
        priority: number
      }>
      defaultTarget: string
    }
  }
  status?: 'idle' | 'running' | 'completed' | 'failed' | 'waiting'
  executionTime?: number
  lastExecuted?: string
}

interface WorkflowEdge {
  id: string
  source: string
  target: string
  type: 'default' | 'conditional' | 'parallel' | 'sequential'
  condition?: string
  label?: string
  weight?: number
  executionOrder?: number
}

interface WorkflowExecution {
  id: string
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed'
  startTime?: string
  endTime?: string
  currentNode?: string
  executionLog: Array<{
    nodeId: string
    timestamp: string
    status: 'started' | 'completed' | 'failed'
    duration?: number
    output?: any
    error?: string
  }>
  metrics: {
    totalNodes: number
    completedNodes: number
    failedNodes: number
    totalExecutionTime: number
    averageNodeTime: number
  }
}

interface LangGraphDesignerProps {
  isOpen: boolean
  onClose: () => void
  workflowData?: any
}

const LangGraphDesigner: React.FC<LangGraphDesignerProps> = ({
  isOpen,
  onClose,
  workflowData
}) => {
  // Enhanced state management
  const [activeTab, setActiveTab] = useState<'design' | 'execution' | 'monitoring' | 'templates'>('design');
  const [nodes, setNodes] = useState<WorkflowNode[]>([
    {
      id: 'start',
      type: 'start',
      position: { x: 100, y: 100 },
      data: { label: 'Start' },
      status: 'idle'
    }
  ])
  const [edges, setEdges] = useState<WorkflowEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null)
  const [isAddingNode, setIsAddingNode] = useState(false)
  const [isAddingEdge, setIsAddingEdge] = useState(false)
  const [edgeStart, setEdgeStart] = useState<string | null>(null)
  const [workflowName, setWorkflowName] = useState('New Multi-Agent Workflow')
  const [workflowDescription, setWorkflowDescription] = useState('')
  const [currentExecution, setCurrentExecution] = useState<WorkflowExecution | null>(null)
  const [executionHistory, setExecutionHistory] = useState<WorkflowExecution[]>([])
  const [isExecuting, setIsExecuting] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const canvasRef = useRef<HTMLDivElement>(null)

  // Enhanced node types for multi-agent orchestration
  const nodeTypes = [
    { type: 'agent', label: 'Agent Node', icon: Bot, color: 'bg-blue-500', description: 'Individual AI agent' },
    { type: 'supervisor', label: 'Supervisor', icon: Users, color: 'bg-green-500', description: 'Multi-agent supervisor' },
    { type: 'decision', label: 'Decision Node', icon: Diamond, color: 'bg-yellow-500', description: 'Conditional routing' },
    { type: 'conditional_router', label: 'Smart Router', icon: Share2, color: 'bg-orange-500', description: 'Advanced routing logic' },
    { type: 'subgraph', label: 'Subgraph', icon: Network, color: 'bg-purple-500', description: 'Nested workflow' },
    { type: 'tool', label: 'Tool Node', icon: Zap, color: 'bg-cyan-500', description: 'External tool execution' },
    { type: 'knowledge_base', label: 'Knowledge Base', icon: Database, color: 'bg-indigo-500', description: 'RAG knowledge access' },
    { type: 'human_input', label: 'Human Input', icon: MessageSquare, color: 'bg-pink-500', description: 'Human-in-the-loop' },
    { type: 'end', label: 'End Node', icon: Target, color: 'bg-red-500', description: 'Workflow termination' }
  ]

  // Fetch available agents for workflow nodes
  const { data: availableAgents } = useQuery(
    'available-agents',
    () => enhancedOrchestrationApi.listAgents(),
    {
      refetchInterval: 10000 // Refresh every 10 seconds
    }
  );

  // Fetch available tools
  const { data: availableTools } = useQuery(
    'available-tools',
    () => enhancedOrchestrationApi.listDynamicTools(),
    {
      refetchInterval: 30000 // Refresh every 30 seconds
    }
  );

  // Fetch available knowledge bases
  const { data: availableKnowledgeBases } = useQuery(
    'available-knowledge-bases',
    () => enhancedOrchestrationApi.listKnowledgeBases(),
    {
      refetchInterval: 30000
    }
  );

  // Workflow execution mutation
  const executeWorkflowMutation = useMutation(
    (workflowConfig: any) => workflowApi.executeWorkflow(workflowConfig),
    {
      onSuccess: (data) => {
        setCurrentExecution(data);
        setIsExecuting(true);
        toast.success('Workflow execution started');
      },
      onError: (error) => {
        console.error('Workflow execution failed:', error);
        toast.error('Failed to start workflow execution');
      }
    }
  );

  // Enhanced workflow execution
  const executeWorkflow = async () => {
    if (nodes.length < 2) {
      toast.error('Workflow must have at least a start and end node');
      return;
    }

    const workflowConfig = {
      name: workflowName,
      description: workflowDescription,
      nodes: nodes.map(node => ({
        id: node.id,
        type: node.type,
        data: node.data,
        position: node.position
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type,
        condition: edge.condition,
        label: edge.label
      })),
      execution_config: {
        parallel_execution: true,
        failure_handling: 'continue',
        timeout: 3600000, // 1 hour
        enable_monitoring: true
      }
    };

    try {
      await executeWorkflowMutation.mutateAsync(workflowConfig);
    } catch (error) {
      console.error('Workflow execution error:', error);
    }
  };

  // Pause workflow execution
  const pauseWorkflow = async () => {
    if (!currentExecution) return;

    try {
      await workflowApi.pauseWorkflow(currentExecution.id);
      setIsPaused(true);
      toast.success('Workflow paused');
    } catch (error) {
      toast.error('Failed to pause workflow');
    }
  };

  // Resume workflow execution
  const resumeWorkflow = async () => {
    if (!currentExecution) return;

    try {
      await workflowApi.resumeWorkflow(currentExecution.id);
      setIsPaused(false);
      toast.success('Workflow resumed');
    } catch (error) {
      toast.error('Failed to resume workflow');
    }
  };

  // Stop workflow execution
  const stopWorkflow = async () => {
    if (!currentExecution) return;

    try {
      await workflowApi.stopWorkflow(currentExecution.id);
      setIsExecuting(false);
      setIsPaused(false);
      setCurrentExecution(prev => prev ? { ...prev, status: 'completed' } : null);
      toast.success('Workflow stopped');
    } catch (error) {
      toast.error('Failed to stop workflow');
    }
  };

  const subgraphTypes = [
    { 
      type: 'research_team', 
      label: 'Research Team',
      description: 'Specialized research agents for information gathering',
      agents: ['Research Specialist', 'Data Analyst', 'Web Searcher']
    },
    { 
      type: 'document_team', 
      label: 'Document Team',
      description: 'Document processing and analysis agents',
      agents: ['Document Parser', 'Content Analyzer', 'Summary Generator']
    },
    { 
      type: 'analysis_team', 
      label: 'Analysis Team',
      description: 'Data analysis and insight generation agents',
      agents: ['Data Scientist', 'Statistician', 'Insight Generator']
    }
  ]

  const addNode = useCallback((type: WorkflowNode['type'], position: { x: number; y: number }) => {
    const newNode: WorkflowNode = {
      id: `${type}_${Date.now()}`,
      type,
      position,
      data: {
        label: type === 'agent' ? 'New Agent' : 
               type === 'decision' ? 'Decision Point' :
               type === 'subgraph' ? 'Subgraph' :
               type === 'supervisor' ? 'Supervisor' :
               'End',
        agentType: type === 'agent' ? 'general' : undefined,
        model: type === 'agent' ? 'llama3.2:latest' : undefined,
        subgraphType: type === 'subgraph' ? 'research_team' : undefined,
        supervisorConfig: type === 'supervisor' ? {
          maxWorkers: 3,
          strategy: 'parallel'
        } : undefined
      }
    }
    setNodes(prev => [...prev, newNode])
    setIsAddingNode(false)
  }, [])

  const deleteNode = useCallback((nodeId: string) => {
    setNodes(prev => prev.filter(node => node.id !== nodeId))
    setEdges(prev => prev.filter(edge => edge.source !== nodeId && edge.target !== nodeId))
    setSelectedNode(null)
  }, [])

  const addEdge = useCallback((sourceId: string, targetId: string) => {
    const newEdge: WorkflowEdge = {
      id: `edge_${Date.now()}`,
      source: sourceId,
      target: targetId,
      type: 'default'
    }
    setEdges(prev => [...prev, newEdge])
    setIsAddingEdge(false)
    setEdgeStart(null)
  }, [])

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (isAddingNode) {
      const rect = canvasRef.current?.getBoundingClientRect()
      if (rect) {
        const position = {
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        }
        // Show node type selector
        setIsAddingNode(false)
      }
    }
  }

  const handleNodeClick = (nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    
    if (isAddingEdge) {
      if (edgeStart && edgeStart !== nodeId) {
        addEdge(edgeStart, nodeId)
      } else {
        setEdgeStart(nodeId)
      }
    } else {
      setSelectedNode(nodeId)
    }
  }

  const updateNode = useCallback((nodeId: string, updates: Partial<WorkflowNode['data']>) => {
    setNodes(prev => prev.map(node => 
      node.id === nodeId 
        ? { ...node, data: { ...node.data, ...updates } }
        : node
    ))
  }, [])

  const getNodeIcon = (type: string) => {
    const nodeType = nodeTypes.find(nt => nt.type === type)
    if (nodeType) {
      const Icon = nodeType.icon
      return <Icon className="h-4 w-4" />
    }
    return <Circle className="h-4 w-4" />
  }

  const getNodeColor = (type: string) => {
    const nodeType = nodeTypes.find(nt => nt.type === type)
    return nodeType?.color || 'bg-gray-500'
  }

  const exportWorkflow = () => {
    const workflowData = {
      name: workflowName,
      nodes,
      edges,
      metadata: {
        created: new Date().toISOString(),
        version: '1.0'
      }
    }
    
    const blob = new Blob([JSON.stringify(workflowData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${workflowName.replace(/\s+/g, '_')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const saveWorkflow = async () => {
    // Implementation would save to backend
    console.log('Saving workflow:', { name: workflowName, nodes, edges })
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-background">
      {/* Header */}
      <div className="h-16 border-b border-border bg-card px-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <GitBranch className="h-6 w-6 text-primary" />
          <input
            type="text"
            value={workflowName}
            onChange={(e) => setWorkflowName(e.target.value)}
            className="input w-64"
            placeholder="Workflow name..."
          />
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsAddingNode(true)}
            className={`btn-secondary ${isAddingNode ? 'bg-primary text-primary-foreground' : ''}`}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Node
          </button>
          
          <button
            onClick={() => setIsAddingEdge(!isAddingEdge)}
            className={`btn-secondary ${isAddingEdge ? 'bg-primary text-primary-foreground' : ''}`}
          >
            <ArrowRight className="h-4 w-4 mr-2" />
            Add Edge
          </button>
          
          <div className="w-px h-6 bg-border" />
          
          <button onClick={saveWorkflow} className="btn-secondary">
            <Save className="h-4 w-4 mr-2" />
            Save
          </button>
          
          <button onClick={exportWorkflow} className="btn-secondary">
            <Download className="h-4 w-4 mr-2" />
            Export
          </button>
          
          <button className="btn-primary">
            <Play className="h-4 w-4 mr-2" />
            Execute
          </button>
          
          <button onClick={onClose} className="btn-ghost">
            Close
          </button>
        </div>
      </div>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Node Palette */}
        {isAddingNode && (
          <div className="w-80 bg-card border-r border-border p-4">
            <h3 className="text-lg font-semibold text-foreground mb-4">Add Node</h3>
            <div className="space-y-2">
              {nodeTypes.map((nodeType) => {
                const Icon = nodeType.icon
                return (
                  <button
                    key={nodeType.type}
                    onClick={() => addNode(nodeType.type as WorkflowNode['type'], { x: 200, y: 200 })}
                    className="w-full flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary transition-colors text-left"
                  >
                    <div className={`p-2 rounded-lg ${nodeType.color} text-white`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <span className="font-medium text-foreground">{nodeType.label}</span>
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* Canvas */}
        <div className="flex-1 relative overflow-hidden">
          <div
            ref={canvasRef}
            className="w-full h-full bg-muted/10 relative overflow-auto cursor-crosshair"
            onClick={handleCanvasClick}
          >
            {/* Grid Background */}
            <div 
              className="absolute inset-0 opacity-20"
              style={{
                backgroundImage: `
                  linear-gradient(hsl(var(--muted-foreground)) 1px, transparent 1px),
                  linear-gradient(90deg, hsl(var(--muted-foreground)) 1px, transparent 1px)
                `,
                backgroundSize: '20px 20px'
              }}
            />

            {/* Edges */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              {edges.map((edge) => {
                const sourceNode = nodes.find(n => n.id === edge.source)
                const targetNode = nodes.find(n => n.id === edge.target)
                
                if (!sourceNode || !targetNode) return null
                
                const x1 = sourceNode.position.x + 60
                const y1 = sourceNode.position.y + 30
                const x2 = targetNode.position.x + 60
                const y2 = targetNode.position.y + 30
                
                return (
                  <g key={edge.id}>
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke="hsl(var(--primary))"
                      strokeWidth="2"
                      markerEnd="url(#arrowhead)"
                      className={selectedEdge === edge.id ? 'stroke-primary' : 'stroke-muted-foreground'}
                    />
                    {edge.label && (
                      <text
                        x={(x1 + x2) / 2}
                        y={(y1 + y2) / 2 - 5}
                        textAnchor="middle"
                        className="fill-foreground text-xs"
                      >
                        {edge.label}
                      </text>
                    )}
                  </g>
                )
              })}
              
              {/* Arrow marker */}
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon
                    points="0 0, 10 3.5, 0 7"
                    fill="hsl(var(--primary))"
                  />
                </marker>
              </defs>
            </svg>

            {/* Nodes */}
            {nodes.map((node) => (
              <div
                key={node.id}
                className={`absolute p-3 border-2 rounded-lg cursor-pointer transition-all min-w-32 ${
                  selectedNode === node.id 
                    ? 'border-primary ring-2 ring-primary/20 bg-card' 
                    : 'border-border hover:border-primary/50 bg-card'
                } ${isAddingEdge && edgeStart === node.id ? 'ring-2 ring-yellow-400' : ''}`}
                style={{
                  left: node.position.x,
                  top: node.position.y
                }}
                onClick={(e) => handleNodeClick(node.id, e)}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <div className={`p-1 rounded ${getNodeColor(node.type)} text-white`}>
                    {getNodeIcon(node.type)}
                  </div>
                  <span className="text-sm font-medium text-foreground">{node.data.label}</span>
                  {selectedNode === node.id && node.id !== 'start' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteNode(node.id)
                      }}
                      className="ml-auto p-1 rounded hover:bg-destructive/10 text-destructive"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
                
                {node.data.agentType && (
                  <div className="text-xs text-muted-foreground">
                    {node.data.agentType} • {node.data.model}
                  </div>
                )}
                
                {node.data.subgraphType && (
                  <div className="text-xs text-muted-foreground">
                    {subgraphTypes.find(st => st.type === node.data.subgraphType)?.label}
                  </div>
                )}
                
                {node.data.supervisorConfig && (
                  <div className="text-xs text-muted-foreground">
                    {node.data.supervisorConfig.strategy} • {node.data.supervisorConfig.maxWorkers} workers
                  </div>
                )}
              </div>
            ))}

            {/* Empty State */}
            {nodes.length === 1 && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                  <GitBranch className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-foreground mb-2">
                    Design Your LangGraph Workflow
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    Add nodes and connect them to create a multi-agent workflow
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Properties Panel */}
        {selectedNode && (
          <div className="w-80 bg-card border-l border-border p-4 overflow-y-auto">
            <h3 className="text-lg font-semibold text-foreground mb-4">Node Properties</h3>
            {(() => {
              const node = nodes.find(n => n.id === selectedNode)
              if (!node) return null

              return (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">Label</label>
                    <input
                      type="text"
                      value={node.data.label}
                      onChange={(e) => updateNode(selectedNode, { label: e.target.value })}
                      className="input"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">Description</label>
                    <textarea
                      value={node.data.description || ''}
                      onChange={(e) => updateNode(selectedNode, { description: e.target.value })}
                      className="input resize-none"
                      rows={3}
                      placeholder="Describe this node's purpose..."
                    />
                  </div>

                  {node.type === 'agent' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">Agent Type</label>
                        <select
                          value={node.data.agentType || 'general'}
                          onChange={(e) => updateNode(selectedNode, { agentType: e.target.value })}
                          className="input"
                        >
                          <option value="general">General</option>
                          <option value="research">Research</option>
                          <option value="workflow">Workflow</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">Model</label>
                        <select
                          value={node.data.model || 'llama3.2:latest'}
                          onChange={(e) => updateNode(selectedNode, { model: e.target.value })}
                          className="input"
                        >
                          <option value="llama3.2:latest">Llama 3.2</option>
                          <option value="llama3.1:latest">Llama 3.1</option>
                          <option value="qwen2.5:latest">Qwen 2.5</option>
                        </select>
                      </div>
                    </>
                  )}

                  {node.type === 'subgraph' && (
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">Subgraph Type</label>
                      <select
                        value={node.data.subgraphType || 'research_team'}
                        onChange={(e) => updateNode(selectedNode, { subgraphType: e.target.value as any })}
                        className="input"
                      >
                        {subgraphTypes.map(st => (
                          <option key={st.type} value={st.type}>{st.label}</option>
                        ))}
                      </select>
                      
                      {node.data.subgraphType && (
                        <div className="mt-2 p-2 bg-muted rounded text-xs">
                          <p className="text-muted-foreground mb-1">
                            {subgraphTypes.find(st => st.type === node.data.subgraphType)?.description}
                          </p>
                          <p className="font-medium">Agents:</p>
                          <ul className="list-disc list-inside text-muted-foreground">
                            {subgraphTypes.find(st => st.type === node.data.subgraphType)?.agents.map(agent => (
                              <li key={agent}>{agent}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {node.type === 'supervisor' && node.data.supervisorConfig && (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">Strategy</label>
                        <select
                          value={node.data.supervisorConfig.strategy}
                          onChange={(e) => updateNode(selectedNode, { 
                            supervisorConfig: { 
                              ...node.data.supervisorConfig!, 
                              strategy: e.target.value as any 
                            } 
                          })}
                          className="input"
                        >
                          <option value="parallel">Parallel</option>
                          <option value="sequential">Sequential</option>
                          <option value="conditional">Conditional</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Max Workers: {node.data.supervisorConfig.maxWorkers}
                        </label>
                        <input
                          type="range"
                          min="1"
                          max="10"
                          value={node.data.supervisorConfig.maxWorkers}
                          onChange={(e) => updateNode(selectedNode, { 
                            supervisorConfig: { 
                              ...node.data.supervisorConfig!, 
                              maxWorkers: parseInt(e.target.value) 
                            } 
                          })}
                          className="w-full"
                        />
                      </div>
                    </>
                  )}

                  {node.type === 'decision' && (
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">Condition</label>
                      <textarea
                        value={node.data.condition || ''}
                        onChange={(e) => updateNode(selectedNode, { condition: e.target.value })}
                        className="input resize-none font-mono text-sm"
                        rows={3}
                        placeholder="Enter condition logic..."
                      />
                    </div>
                  )}
                </div>
              )
            })()}
          </div>
        )}
      </div>
    </div>
  )
}

export default LangGraphDesigner
