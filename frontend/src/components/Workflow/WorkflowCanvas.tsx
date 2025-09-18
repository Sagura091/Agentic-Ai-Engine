import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import {
  Bot,
  Plus,
  GitBranch,
  Play,
  Save,
  Download,
  Upload,
  Trash2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize,
  Grid,
  Eye,
  Settings,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Brain,
  Users,
  MessageSquare,
  BarChart3,
  Copy,
  Square
} from 'lucide-react'
import { CanvasErrorBoundary } from '../ErrorBoundary'
import {
  generateUUID,
  rafThrottle,
  transformPoint,
  constrainToBounds,
  snapToGrid,
  debounce
} from '../../utils'
import { wsApi } from '../../services/api'

interface ConnectionPort {
  id: string
  type: 'input' | 'output'
  dataType: 'any' | 'text' | 'number' | 'boolean' | 'object' | 'array'
  position: { x: number; y: number }
  connected: boolean
  label?: string
}

interface WorkflowNode {
  id: string
  type: 'agent' | 'decision' | 'start' | 'end' | 'subgraph' | 'supervisor' | 'custom'
  position: { x: number; y: number }
  size: { width: number; height: number }
  data: {
    label: string
    agentType?: string
    model?: string
    description?: string
    status?: 'idle' | 'running' | 'completed' | 'error' | 'warning'
    executionTime?: number
    lastExecuted?: string
    customConfig?: any
    template?: string
  }
  ports: {
    inputs: ConnectionPort[]
    outputs: ConnectionPort[]
  }
  style?: {
    backgroundColor?: string
    borderColor?: string
    textColor?: string
    icon?: string
  }
  validation?: {
    isValid: boolean
    errors: string[]
    warnings: string[]
  }
}

interface WorkflowEdge {
  id: string
  source: string
  target: string
  sourcePort: string
  targetPort: string
  type: 'default' | 'conditional' | 'data' | 'control'
  label?: string
  animated?: boolean
  style?: {
    stroke?: string
    strokeWidth?: number
    strokeDasharray?: string
  }
  data?: {
    condition?: string
    dataType?: string
    lastDataFlow?: string
  }
}

interface CanvasState {
  zoom: number
  pan: { x: number; y: number }
  selection: string[]
  dragState: {
    isDragging: boolean
    dragType: 'node' | 'canvas' | 'connection' | 'selection'
    startPos: { x: number; y: number }
    currentPos: { x: number; y: number }
    draggedNode?: string
    connectionStart?: {
      nodeId: string;
      portId: string;
      position: { x: number; y: number }
    }
  }
  executionState: {
    isExecuting: boolean
    currentNode?: string
    executionPath: string[]
    startTime?: number
  }
  clipboard: {
    nodes: WorkflowNode[]
    edges: WorkflowEdge[]
  }
  history: {
    past: Array<{ nodes: WorkflowNode[]; edges: WorkflowEdge[] }>
    present: { nodes: WorkflowNode[]; edges: WorkflowEdge[] }
    future: Array<{ nodes: WorkflowNode[]; edges: WorkflowEdge[] }>
  }
  settings: {
    snapToGrid: boolean
    gridSize: number
    showGrid: boolean
    showMinimap: boolean
    autoSave: boolean
  }
}

const WorkflowCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const executionTimeoutRef = useRef<number | null>(null)
  const workflowIdRef = useRef<string>(generateUUID())

  const [nodes, setNodes] = useState<WorkflowNode[]>([
    {
      id: 'start',
      type: 'start',
      position: { x: 200, y: 150 },
      size: { width: 120, height: 60 },
      data: {
        label: 'Start',
        status: 'idle'
      },
      ports: {
        inputs: [],
        outputs: [{
          id: 'start-out',
          type: 'output',
          dataType: 'any',
          position: { x: 120, y: 30 },
          connected: false,
          label: 'Flow'
        }]
      },
      validation: {
        isValid: true,
        errors: [],
        warnings: []
      }
    }
  ])
  const [edges, setEdges] = useState<WorkflowEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null)
  const [isAddingNode, setIsAddingNode] = useState(false)
  const [canvasState, setCanvasState] = useState<CanvasState>({
    zoom: 1,
    pan: { x: 0, y: 0 },
    selection: [],
    dragState: {
      isDragging: false,
      dragType: 'canvas',
      startPos: { x: 0, y: 0 },
      currentPos: { x: 0, y: 0 }
    },
    executionState: {
      isExecuting: false,
      executionPath: []
    },
    clipboard: {
      nodes: [],
      edges: []
    },
    history: {
      past: [],
      present: { nodes: [], edges: [] },
      future: []
    },
    settings: {
      snapToGrid: true,
      gridSize: 20,
      showGrid: true,
      showMinimap: false,
      autoSave: true
    }
  })

  // Auto-validate workflow when nodes or edges change
  useEffect(() => {
    validateWorkflow()
  }, [nodes, edges])

  // WebSocket integration for real-time collaboration
  useEffect(() => {
    const initWebSocket = async () => {
      try {
        await wsApi.connect()
        wsApi.joinWorkflowRoom(workflowIdRef.current)

        // Listen for real-time updates
        wsApi.on('canvas_updated', (data) => {
          if (data.workflowId === workflowIdRef.current) {
            setNodes(data.nodes)
            setEdges(data.edges)
          }
        })

        wsApi.on('workflow_started', (data) => {
          if (data.workflowId === workflowIdRef.current) {
            setCanvasState(prev => ({
              ...prev,
              executionState: {
                isExecuting: true,
                currentNode: data.startNode,
                executionPath: [data.startNode],
                startTime: Date.now()
              }
            }))
          }
        })

        wsApi.on('node_started', (data) => {
          if (data.workflowId === workflowIdRef.current) {
            setNodes(prev => prev.map(node =>
              node.id === data.nodeId
                ? { ...node, data: { ...node.data, status: 'running' as const } }
                : node
            ))
            setCanvasState(prev => ({
              ...prev,
              executionState: {
                ...prev.executionState,
                currentNode: data.nodeId,
                executionPath: [...prev.executionState.executionPath, data.nodeId]
              }
            }))
          }
        })

        wsApi.on('node_completed', (data) => {
          if (data.workflowId === workflowIdRef.current) {
            setNodes(prev => prev.map(node =>
              node.id === data.nodeId
                ? {
                    ...node,
                    data: {
                      ...node.data,
                      status: 'completed' as const,
                      executionTime: data.executionTime,
                      lastExecuted: new Date().toISOString()
                    }
                  }
                : node
            ))
          }
        })

        wsApi.on('workflow_completed', (data) => {
          if (data.workflowId === workflowIdRef.current) {
            setCanvasState(prev => ({
              ...prev,
              executionState: {
                ...prev.executionState,
                isExecuting: false,
                currentNode: undefined
              }
            }))
          }
        })

      } catch (error) {
        console.error('Failed to initialize WebSocket:', error)
      }
    }

    initWebSocket()

    return () => {
      wsApi.leaveWorkflowRoom(workflowIdRef.current)
      wsApi.off('canvas_updated')
      wsApi.off('workflow_started')
      wsApi.off('node_started')
      wsApi.off('node_completed')
      wsApi.off('workflow_completed')
    }
  }, [])

  // Cleanup execution timeout on unmount
  useEffect(() => {
    return () => {
      if (executionTimeoutRef.current) {
        clearTimeout(executionTimeoutRef.current)
      }
    }
  }, [])

  // Advanced Node Templates
  const nodeTemplates = {
    agent: {
      general: {
        label: 'General Agent',
        description: 'Multi-purpose AI agent for general tasks',
        icon: '🤖',
        ports: {
          inputs: [
            { id: 'input', type: 'input' as const, dataType: 'text' as const, position: { x: 0, y: 30 }, connected: false, label: 'Input' }
          ],
          outputs: [
            { id: 'output', type: 'output' as const, dataType: 'text' as const, position: { x: 160, y: 30 }, connected: false, label: 'Output' },
            { id: 'error', type: 'output' as const, dataType: 'object' as const, position: { x: 160, y: 50 }, connected: false, label: 'Error' }
          ]
        },
        size: { width: 160, height: 80 }
      },
      research: {
        label: 'Research Agent',
        description: 'Specialized agent for research and data gathering',
        icon: '🔍',
        ports: {
          inputs: [
            { id: 'query', type: 'input' as const, dataType: 'text' as const, position: { x: 0, y: 25 }, connected: false, label: 'Query' },
            { id: 'sources', type: 'input' as const, dataType: 'array' as const, position: { x: 0, y: 45 }, connected: false, label: 'Sources' }
          ],
          outputs: [
            { id: 'results', type: 'output' as const, dataType: 'object' as const, position: { x: 180, y: 25 }, connected: false, label: 'Results' },
            { id: 'summary', type: 'output' as const, dataType: 'text' as const, position: { x: 180, y: 45 }, connected: false, label: 'Summary' },
            { id: 'citations', type: 'output' as const, dataType: 'array' as const, position: { x: 180, y: 65 }, connected: false, label: 'Citations' }
          ]
        },
        size: { width: 180, height: 90 }
      },
      workflow: {
        label: 'Workflow Agent',
        description: 'Agent specialized in workflow orchestration',
        icon: '⚡',
        ports: {
          inputs: [
            { id: 'task', type: 'input' as const, dataType: 'object' as const, position: { x: 0, y: 30 }, connected: false, label: 'Task' },
            { id: 'context', type: 'input' as const, dataType: 'object' as const, position: { x: 0, y: 50 }, connected: false, label: 'Context' }
          ],
          outputs: [
            { id: 'result', type: 'output' as const, dataType: 'object' as const, position: { x: 170, y: 30 }, connected: false, label: 'Result' },
            { id: 'status', type: 'output' as const, dataType: 'text' as const, position: { x: 170, y: 50 }, connected: false, label: 'Status' }
          ]
        },
        size: { width: 170, height: 80 }
      }
    },
    decision: {
      conditional: {
        label: 'Conditional',
        description: 'Route flow based on conditions',
        icon: '🔀',
        ports: {
          inputs: [
            { id: 'input', type: 'input' as const, dataType: 'any' as const, position: { x: 0, y: 30 }, connected: false, label: 'Input' }
          ],
          outputs: [
            { id: 'true', type: 'output' as const, dataType: 'any' as const, position: { x: 140, y: 20 }, connected: false, label: 'True' },
            { id: 'false', type: 'output' as const, dataType: 'any' as const, position: { x: 140, y: 40 }, connected: false, label: 'False' }
          ]
        },
        size: { width: 140, height: 60 }
      },
      switch: {
        label: 'Switch',
        description: 'Multi-way conditional routing',
        icon: '🎛️',
        ports: {
          inputs: [
            { id: 'input', type: 'input' as const, dataType: 'any' as const, position: { x: 0, y: 40 }, connected: false, label: 'Input' }
          ],
          outputs: [
            { id: 'case1', type: 'output' as const, dataType: 'any' as const, position: { x: 150, y: 20 }, connected: false, label: 'Case 1' },
            { id: 'case2', type: 'output' as const, dataType: 'any' as const, position: { x: 150, y: 35 }, connected: false, label: 'Case 2' },
            { id: 'case3', type: 'output' as const, dataType: 'any' as const, position: { x: 150, y: 50 }, connected: false, label: 'Case 3' },
            { id: 'default', type: 'output' as const, dataType: 'any' as const, position: { x: 150, y: 65 }, connected: false, label: 'Default' }
          ]
        },
        size: { width: 150, height: 85 }
      }
    },
    subgraph: {
      research_team: {
        label: 'Research Team',
        description: 'Collaborative research subgraph',
        icon: '👥',
        ports: {
          inputs: [
            { id: 'task', type: 'input' as const, dataType: 'text' as const, position: { x: 0, y: 35 }, connected: false, label: 'Task' }
          ],
          outputs: [
            { id: 'report', type: 'output' as const, dataType: 'object' as const, position: { x: 200, y: 35 }, connected: false, label: 'Report' }
          ]
        },
        size: { width: 200, height: 70 }
      },
      document_team: {
        label: 'Document Team',
        description: 'Document processing subgraph',
        icon: '📄',
        ports: {
          inputs: [
            { id: 'documents', type: 'input' as const, dataType: 'array' as const, position: { x: 0, y: 35 }, connected: false, label: 'Documents' }
          ],
          outputs: [
            { id: 'processed', type: 'output' as const, dataType: 'array' as const, position: { x: 200, y: 35 }, connected: false, label: 'Processed' }
          ]
        },
        size: { width: 200, height: 70 }
      }
    },
    supervisor: {
      parallel: {
        label: 'Parallel Supervisor',
        description: 'Execute multiple agents in parallel',
        icon: '⚡',
        ports: {
          inputs: [
            { id: 'tasks', type: 'input' as const, dataType: 'array' as const, position: { x: 0, y: 40 }, connected: false, label: 'Tasks' }
          ],
          outputs: [
            { id: 'results', type: 'output' as const, dataType: 'array' as const, position: { x: 190, y: 40 }, connected: false, label: 'Results' }
          ]
        },
        size: { width: 190, height: 80 }
      },
      sequential: {
        label: 'Sequential Supervisor',
        description: 'Execute agents in sequence',
        icon: '🔄',
        ports: {
          inputs: [
            { id: 'tasks', type: 'input' as const, dataType: 'array' as const, position: { x: 0, y: 40 }, connected: false, label: 'Tasks' }
          ],
          outputs: [
            { id: 'results', type: 'output' as const, dataType: 'array' as const, position: { x: 190, y: 40 }, connected: false, label: 'Results' }
          ]
        },
        size: { width: 190, height: 80 }
      }
    },
    custom: {
      api_call: {
        label: 'API Call',
        description: 'Make HTTP API requests',
        icon: '🌐',
        ports: {
          inputs: [
            { id: 'url', type: 'input' as const, dataType: 'text' as const, position: { x: 0, y: 25 }, connected: false, label: 'URL' },
            { id: 'method', type: 'input' as const, dataType: 'text' as const, position: { x: 0, y: 40 }, connected: false, label: 'Method' },
            { id: 'data', type: 'input' as const, dataType: 'object' as const, position: { x: 0, y: 55 }, connected: false, label: 'Data' }
          ],
          outputs: [
            { id: 'response', type: 'output' as const, dataType: 'object' as const, position: { x: 160, y: 35 }, connected: false, label: 'Response' },
            { id: 'error', type: 'output' as const, dataType: 'object' as const, position: { x: 160, y: 55 }, connected: false, label: 'Error' }
          ]
        },
        size: { width: 160, height: 80 }
      },
      data_transform: {
        label: 'Data Transform',
        description: 'Transform and process data',
        icon: '🔄',
        ports: {
          inputs: [
            { id: 'input', type: 'input' as const, dataType: 'any' as const, position: { x: 0, y: 30 }, connected: false, label: 'Input' },
            { id: 'transform', type: 'input' as const, dataType: 'text' as const, position: { x: 0, y: 50 }, connected: false, label: 'Transform' }
          ],
          outputs: [
            { id: 'output', type: 'output' as const, dataType: 'any' as const, position: { x: 170, y: 40 }, connected: false, label: 'Output' }
          ]
        },
        size: { width: 170, height: 80 }
      }
    }
  }

  const addNode = useCallback((type: WorkflowNode['type'], subtype: string, position: { x: number; y: number }) => {
    const nodeId = generateUUID()

    // Apply snap to grid if enabled
    const finalPosition = canvasState.settings.snapToGrid
      ? snapToGrid(position, canvasState.settings.gridSize)
      : position

    // Handle end node specially
    if (type === 'end') {
      const newNode: WorkflowNode = {
        id: nodeId,
        type: 'end',
        position: finalPosition,
        size: { width: 100, height: 60 },
        data: {
          label: 'End',
          description: 'Workflow termination point',
          status: 'idle'
        },
        ports: {
          inputs: [{
            id: generateUUID(),
            type: 'input',
            dataType: 'any',
            position: { x: 0, y: 30 },
            connected: false,
            label: 'Input'
          }],
          outputs: []
        },
        validation: {
          isValid: true,
          errors: [],
          warnings: []
        }
      }

      const newNodes = [...nodes, newNode]
      setNodes(newNodes)
      saveToHistory(newNodes, edges)
      setIsAddingNode(false)

      // Broadcast to other users
      wsApi.updateCanvas(workflowIdRef.current, { nodes: newNodes, edges })
      return
    }

    const template = nodeTemplates[type]?.[subtype]
    if (!template) return

    const newNode: WorkflowNode = {
      id: nodeId,
      type,
      position: finalPosition,
      size: template.size,
      data: {
        label: template.label,
        description: template.description,
        status: 'idle',
        template: subtype,
        agentType: type === 'agent' ? subtype : undefined,
        model: type === 'agent' ? 'llama3.2:latest' : undefined
      },
      ports: {
        inputs: template.ports.inputs.map(port => ({ ...port, id: generateUUID() })),
        outputs: template.ports.outputs.map(port => ({ ...port, id: generateUUID() }))
      },
      style: {
        icon: template.icon
      },
      validation: {
        isValid: true,
        errors: [],
        warnings: []
      }
    }

    const newNodes = [...nodes, newNode]
    setNodes(newNodes)
    saveToHistory(newNodes, edges)
    setIsAddingNode(false)

    // Broadcast to other users
    wsApi.updateCanvas(workflowIdRef.current, { nodes: newNodes, edges })
  }, [nodes, edges, canvasState.settings])

  // History management
  const saveToHistory = useCallback((newNodes: WorkflowNode[], newEdges: WorkflowEdge[]) => {
    setCanvasState(prev => ({
      ...prev,
      history: {
        past: [...prev.history.past, prev.history.present],
        present: { nodes: newNodes, edges: newEdges },
        future: []
      }
    }))
  }, [])

  const undo = useCallback(() => {
    setCanvasState(prev => {
      if (prev.history.past.length === 0) return prev

      const previous = prev.history.past[prev.history.past.length - 1]
      const newPast = prev.history.past.slice(0, -1)

      setNodes(previous.nodes)
      setEdges(previous.edges)

      return {
        ...prev,
        history: {
          past: newPast,
          present: previous,
          future: [prev.history.present, ...prev.history.future]
        }
      }
    })
  }, [])

  const redo = useCallback(() => {
    setCanvasState(prev => {
      if (prev.history.future.length === 0) return prev

      const next = prev.history.future[0]
      const newFuture = prev.history.future.slice(1)

      setNodes(next.nodes)
      setEdges(next.edges)

      return {
        ...prev,
        history: {
          past: [...prev.history.past, prev.history.present],
          present: next,
          future: newFuture
        }
      }
    })
  }, [])

  const deleteNode = useCallback((nodeId: string) => {
    const newNodes = nodes.filter(node => node.id !== nodeId)
    const newEdges = edges.filter(edge => edge.source !== nodeId && edge.target !== nodeId)

    setNodes(newNodes)
    setEdges(newEdges)
    setSelectedNode(null)
    saveToHistory(newNodes, newEdges)

    // Broadcast to other users
    wsApi.updateCanvas(workflowIdRef.current, { nodes: newNodes, edges: newEdges })
    validateWorkflow()
  }, [nodes, edges])

  const updateNode = useCallback((nodeId: string, updates: Partial<WorkflowNode['data']>) => {
    const newNodes = nodes.map(node =>
      node.id === nodeId
        ? { ...node, data: { ...node.data, ...updates } }
        : node
    )

    setNodes(newNodes)
    saveToHistory(newNodes, edges)

    // Broadcast to other users
    wsApi.updateCanvas(workflowIdRef.current, { nodes: newNodes, edges })
    validateWorkflow()
  }, [nodes, edges])

  // Copy/Paste functionality
  const copySelectedNodes = useCallback(() => {
    const selectedNodes = nodes.filter(node => canvasState.selection.includes(node.id))
    const selectedEdges = edges.filter(edge =>
      canvasState.selection.includes(edge.source) && canvasState.selection.includes(edge.target)
    )

    setCanvasState(prev => ({
      ...prev,
      clipboard: {
        nodes: selectedNodes,
        edges: selectedEdges
      }
    }))
  }, [nodes, edges, canvasState.selection])

  const pasteNodes = useCallback(() => {
    if (canvasState.clipboard.nodes.length === 0) return

    const offset = { x: 50, y: 50 }
    const newNodes = canvasState.clipboard.nodes.map(node => ({
      ...node,
      id: generateUUID(),
      position: {
        x: node.position.x + offset.x,
        y: node.position.y + offset.y
      },
      ports: {
        inputs: node.ports.inputs.map(port => ({ ...port, id: generateUUID(), connected: false })),
        outputs: node.ports.outputs.map(port => ({ ...port, id: generateUUID(), connected: false }))
      }
    }))

    const allNodes = [...nodes, ...newNodes]
    setNodes(allNodes)
    saveToHistory(allNodes, edges)

    // Select pasted nodes
    setCanvasState(prev => ({
      ...prev,
      selection: newNodes.map(node => node.id)
    }))

    // Broadcast to other users
    wsApi.updateCanvas(workflowIdRef.current, { nodes: allNodes, edges })
  }, [nodes, edges, canvasState.clipboard])

  // Connection Port System
  const startConnection = useCallback((nodeId: string, portId: string, portType: 'input' | 'output', event: React.MouseEvent) => {
    if (portType === 'output') {
      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      const sourceNode = nodes.find(n => n.id === nodeId)
      const sourcePort = sourceNode?.ports.outputs.find(p => p.id === portId)
      if (!sourceNode || !sourcePort) return

      const portWorldPos = {
        x: sourceNode.position.x + sourcePort.position.x,
        y: sourceNode.position.y + sourcePort.position.y
      }

      setCanvasState(prev => ({
        ...prev,
        dragState: {
          ...prev.dragState,
          isDragging: true,
          dragType: 'connection',
          startPos: portWorldPos,
          currentPos: portWorldPos,
          connectionStart: {
            nodeId,
            portId,
            position: portWorldPos
          }
        }
      }))
    }
  }, [nodes])

  const completeConnection = useCallback((targetNodeId: string, targetPortId: string) => {
    const { connectionStart } = canvasState.dragState
    if (!connectionStart) return

    // Validate connection
    const sourceNode = nodes.find(n => n.id === connectionStart.nodeId)
    const targetNode = nodes.find(n => n.id === targetNodeId)
    const sourcePort = sourceNode?.ports.outputs.find(p => p.id === connectionStart.portId)
    const targetPort = targetNode?.ports.inputs.find(p => p.id === targetPortId)

    if (!sourceNode || !targetNode || !sourcePort || !targetPort) return
    if (sourceNode.id === targetNode.id) return // No self-connections
    if (targetPort.connected) return // Target port already connected

    // Check for circular dependencies
    const wouldCreateCycle = (sourceId: string, targetId: string): boolean => {
      const visited = new Set<string>()
      const stack = [targetId]

      while (stack.length > 0) {
        const currentId = stack.pop()!
        if (currentId === sourceId) return true
        if (visited.has(currentId)) continue

        visited.add(currentId)
        const outgoingEdges = edges.filter(edge => edge.source === currentId)
        stack.push(...outgoingEdges.map(edge => edge.target))
      }

      return false
    }

    if (wouldCreateCycle(connectionStart.nodeId, targetNodeId)) {
      console.warn('Connection would create a cycle')
      return
    }

    // Check data type compatibility
    const isCompatible = sourcePort.dataType === 'any' ||
                        targetPort.dataType === 'any' ||
                        sourcePort.dataType === targetPort.dataType

    const newEdge: WorkflowEdge = {
      id: generateUUID(),
      source: connectionStart.nodeId,
      target: targetNodeId,
      sourcePort: connectionStart.portId,
      targetPort: targetPortId,
      type: isCompatible ? 'data' : 'default',
      animated: false,
      style: {
        stroke: isCompatible ? '#10b981' : '#ef4444',
        strokeWidth: 2
      },
      data: {
        dataType: sourcePort.dataType
      }
    }

    const newEdges = [...edges, newEdge]
    setEdges(newEdges)

    // Update port connection status
    const newNodes = nodes.map(node => {
      if (node.id === connectionStart.nodeId) {
        return {
          ...node,
          ports: {
            ...node.ports,
            outputs: node.ports.outputs.map(port =>
              port.id === connectionStart.portId ? { ...port, connected: true } : port
            )
          }
        }
      }
      if (node.id === targetNodeId) {
        return {
          ...node,
          ports: {
            ...node.ports,
            inputs: node.ports.inputs.map(port =>
              port.id === targetPortId ? { ...port, connected: true } : port
            )
          }
        }
      }
      return node
    })

    setNodes(newNodes)
    saveToHistory(newNodes, newEdges)

    setCanvasState(prev => ({
      ...prev,
      dragState: {
        ...prev.dragState,
        isDragging: false,
        dragType: 'canvas',
        connectionStart: undefined
      }
    }))

    // Broadcast to other users
    wsApi.updateCanvas(workflowIdRef.current, { nodes: newNodes, edges: newEdges })
    validateWorkflow()
  }, [canvasState.dragState, nodes, edges])

  // Node Dragging System with proper coordinate transformation
  const startNodeDrag = useCallback((nodeId: string, startPos: { x: number; y: number }) => {
    setCanvasState(prev => ({
      ...prev,
      dragState: {
        ...prev.dragState,
        isDragging: true,
        dragType: 'node',
        startPos,
        currentPos: startPos,
        draggedNode: nodeId
      }
    }))
  }, [])

  const updateNodeDrag = useCallback(rafThrottle((currentPos: { x: number; y: number }) => {
    const { dragState } = canvasState
    if (!dragState.isDragging || dragState.dragType !== 'node' || !dragState.draggedNode) return

    const deltaX = currentPos.x - dragState.startPos.x
    const deltaY = currentPos.y - dragState.startPos.y

    setNodes(prev => prev.map(node => {
      if (node.id === dragState.draggedNode) {
        let newPosition = {
          x: node.position.x + deltaX,
          y: node.position.y + deltaY
        }

        // Apply snap to grid if enabled
        if (canvasState.settings.snapToGrid) {
          newPosition = snapToGrid(newPosition, canvasState.settings.gridSize)
        }

        // Constrain to canvas bounds
        const canvasRect = canvasRef.current?.getBoundingClientRect()
        if (canvasRect) {
          const bounds = {
            x: -canvasState.pan.x / canvasState.zoom,
            y: -canvasState.pan.y / canvasState.zoom,
            width: canvasRect.width / canvasState.zoom,
            height: canvasRect.height / canvasState.zoom
          }
          newPosition = constrainToBounds(newPosition, bounds, node.size)
        }

        return { ...node, position: newPosition }
      }
      return node
    }))

    setCanvasState(prev => ({
      ...prev,
      dragState: {
        ...prev.dragState,
        startPos: currentPos
      }
    }))
  }), [canvasState])

  const endNodeDrag = useCallback(() => {
    const { dragState } = canvasState
    if (dragState.isDragging && dragState.dragType === 'node') {
      // Save to history when drag ends
      saveToHistory(nodes, edges)

      // Broadcast to other users
      wsApi.updateCanvas(workflowIdRef.current, { nodes, edges })
    }

    setCanvasState(prev => ({
      ...prev,
      dragState: {
        ...prev.dragState,
        isDragging: false,
        dragType: 'canvas',
        draggedNode: undefined
      }
    }))
  }, [canvasState, nodes, edges])

  // Workflow Validation System
  const validateWorkflow = useCallback(() => {
    setNodes(prev => prev.map(node => {
      const errors: string[] = []
      const warnings: string[] = []

      // Check for unconnected required inputs
      const requiredInputs = node.ports.inputs.filter(port => port.dataType !== 'any')
      const unconnectedRequired = requiredInputs.filter(port => !port.connected)
      if (unconnectedRequired.length > 0) {
        warnings.push(`Unconnected inputs: ${unconnectedRequired.map(p => p.label).join(', ')}`)
      }

      // Check for isolated nodes (except start/end)
      if (node.type !== 'start' && node.type !== 'end') {
        const hasInputConnections = node.ports.inputs.some(port => port.connected)
        const hasOutputConnections = node.ports.outputs.some(port => port.connected)
        if (!hasInputConnections && !hasOutputConnections) {
          warnings.push('Node is isolated (no connections)')
        }
      }

      // Type-specific validations
      if (node.type === 'decision' && !node.data.customConfig?.condition) {
        errors.push('Decision node requires a condition')
      }

      return {
        ...node,
        validation: {
          isValid: errors.length === 0,
          errors,
          warnings
        }
      }
    }))
  }, [])



  // Live Execution System with proper cleanup
  const startExecution = useCallback(async () => {
    // Clear any existing execution timeout
    if (executionTimeoutRef.current) {
      clearTimeout(executionTimeoutRef.current)
    }

    setCanvasState(prev => ({
      ...prev,
      executionState: {
        isExecuting: true,
        currentNode: 'start',
        executionPath: ['start'],
        startTime: Date.now()
      }
    }))

    // Notify backend to start execution
    wsApi.executeWorkflow(workflowIdRef.current, { nodes, edges })

    // Simulate execution flow for demo purposes
    const startNode = nodes.find(n => n.type === 'start')
    if (!startNode) return

    const executionQueue = [startNode.id]
    const executedNodes = new Set<string>()
    const timeouts: number[] = []

    const executeNode = async (nodeId: string): Promise<void> => {
      return new Promise((resolve) => {
        // Update current executing node
        setCanvasState(prev => ({
          ...prev,
          executionState: {
            ...prev.executionState,
            currentNode: nodeId,
            executionPath: [...prev.executionState.executionPath, nodeId]
          }
        }))

        // Update node status
        setNodes(prev => prev.map(node =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, status: 'running' as const } }
            : node
        ))

        // Simulate execution time
        const timeout = setTimeout(() => {
          const executionTime = 1000 + Math.random() * 2000

          // Mark as completed
          setNodes(prev => prev.map(node =>
            node.id === nodeId
              ? {
                  ...node,
                  data: {
                    ...node.data,
                    status: 'completed' as const,
                    executionTime,
                    lastExecuted: new Date().toISOString()
                  }
                }
              : node
          ))

          resolve()
        }, 1000 + Math.random() * 2000)

        timeouts.push(timeout)
      })
    }

    try {
      while (executionQueue.length > 0) {
        const currentNodeId = executionQueue.shift()!
        if (executedNodes.has(currentNodeId)) continue

        await executeNode(currentNodeId)
        executedNodes.add(currentNodeId)

        // Find next nodes
        const outgoingEdges = edges.filter(edge => edge.source === currentNodeId)
        outgoingEdges.forEach(edge => {
          if (!executedNodes.has(edge.target)) {
            executionQueue.push(edge.target)

            // Animate edge
            setEdges(prev => prev.map(e =>
              e.id === edge.id
                ? { ...e, animated: true, style: { ...e.style, stroke: '#10b981' } }
                : e
            ))
          }
        })
      }
    } catch (error) {
      console.error('Execution error:', error)
    } finally {
      // Clear all timeouts
      timeouts.forEach(timeout => clearTimeout(timeout))

      // Execution complete
      setCanvasState(prev => ({
        ...prev,
        executionState: {
          ...prev.executionState,
          isExecuting: false,
          currentNode: undefined
        }
      }))
    }
  }, [nodes, edges])

  const stopExecution = useCallback(() => {
    // Clear any existing execution timeout
    if (executionTimeoutRef.current) {
      clearTimeout(executionTimeoutRef.current)
    }

    // Notify backend to stop execution
    wsApi.stopWorkflow(workflowIdRef.current)

    setCanvasState(prev => ({
      ...prev,
      executionState: {
        isExecuting: false,
        executionPath: [],
        currentNode: undefined,
        startTime: undefined
      }
    }))

    // Reset node statuses
    setNodes(prev => prev.map(node => ({
      ...node,
      data: { ...node.data, status: 'idle' as const }
    })))

    // Reset edge animations
    setEdges(prev => prev.map(edge => ({
      ...edge,
      animated: false,
      style: { ...edge.style, stroke: edge.data?.dataType === 'any' ? '#6b7280' : '#10b981' }
    })))
  }, [])

  // Auto-layout Algorithm
  const autoLayout = useCallback(() => {
    if (nodes.length === 0) return

    const layoutNodes = [...nodes]
    const layoutEdges = [...edges]

    // Find start node
    const startNode = layoutNodes.find(n => n.type === 'start')
    if (!startNode) return

    // Hierarchical layout
    const levels: string[][] = []
    const visited = new Set<string>()
    const queue = [{ nodeId: startNode.id, level: 0 }]

    // Build level hierarchy
    while (queue.length > 0) {
      const { nodeId, level } = queue.shift()!
      if (visited.has(nodeId)) continue

      visited.add(nodeId)

      if (!levels[level]) levels[level] = []
      levels[level].push(nodeId)

      // Find connected nodes
      const outgoingEdges = layoutEdges.filter(edge => edge.source === nodeId)
      outgoingEdges.forEach(edge => {
        if (!visited.has(edge.target)) {
          queue.push({ nodeId: edge.target, level: level + 1 })
        }
      })
    }

    // Position nodes
    const levelSpacing = 300
    const nodeSpacing = 150
    const startX = 100
    const startY = 100

    levels.forEach((levelNodes, levelIndex) => {
      const levelY = startY + levelIndex * levelSpacing
      const totalWidth = (levelNodes.length - 1) * nodeSpacing
      const startLevelX = startX - totalWidth / 2

      levelNodes.forEach((nodeId, nodeIndex) => {
        const nodeX = startLevelX + nodeIndex * nodeSpacing

        const nodeIndex2 = layoutNodes.findIndex(n => n.id === nodeId)
        if (nodeIndex2 !== -1) {
          layoutNodes[nodeIndex2] = {
            ...layoutNodes[nodeIndex2],
            position: { x: nodeX, y: levelY }
          }
        }
      })
    })

    setNodes(layoutNodes)
    saveToHistory(layoutNodes, edges)

    // Broadcast to other users
    wsApi.updateCanvas(workflowIdRef.current, { nodes: layoutNodes, edges })
  }, [nodes, edges])

  // Force-directed layout for complex graphs
  const forceLayout = useCallback(() => {
    if (nodes.length === 0) return

    const layoutNodes = [...nodes]
    const iterations = 100
    const repulsionForce = 5000
    const attractionForce = 0.1
    const damping = 0.9

    for (let i = 0; i < iterations; i++) {
      const forces: { [nodeId: string]: { x: number; y: number } } = {}

      // Initialize forces
      layoutNodes.forEach(node => {
        forces[node.id] = { x: 0, y: 0 }
      })

      // Repulsion between all nodes
      for (let j = 0; j < layoutNodes.length; j++) {
        for (let k = j + 1; k < layoutNodes.length; k++) {
          const node1 = layoutNodes[j]
          const node2 = layoutNodes[k]

          const dx = node2.position.x - node1.position.x
          const dy = node2.position.y - node1.position.y
          const distance = Math.sqrt(dx * dx + dy * dy) || 1

          const force = repulsionForce / (distance * distance)
          const fx = (dx / distance) * force
          const fy = (dy / distance) * force

          forces[node1.id].x -= fx
          forces[node1.id].y -= fy
          forces[node2.id].x += fx
          forces[node2.id].y += fy
        }
      }

      // Attraction along edges
      edges.forEach(edge => {
        const sourceNode = layoutNodes.find(n => n.id === edge.source)
        const targetNode = layoutNodes.find(n => n.id === edge.target)

        if (sourceNode && targetNode) {
          const dx = targetNode.position.x - sourceNode.position.x
          const dy = targetNode.position.y - sourceNode.position.y
          const distance = Math.sqrt(dx * dx + dy * dy) || 1

          const force = attractionForce * distance
          const fx = (dx / distance) * force
          const fy = (dy / distance) * force

          forces[sourceNode.id].x += fx
          forces[sourceNode.id].y += fy
          forces[targetNode.id].x -= fx
          forces[targetNode.id].y -= fy
        }
      })

      // Apply forces
      layoutNodes.forEach(node => {
        const force = forces[node.id]
        node.position.x += force.x * damping
        node.position.y += force.y * damping
      })
    }

    setNodes(layoutNodes)
    saveToHistory(layoutNodes, edges)

    // Broadcast to other users
    wsApi.updateCanvas(workflowIdRef.current, { nodes: layoutNodes, edges })
  }, [nodes, edges])

  const getNodeIcon = (node: WorkflowNode) => {
    const { type, data, style } = node

    if (style?.icon) {
      return <span className="text-lg">{style.icon}</span>
    }

    switch (type) {
      case 'agent':
        return <Bot className="h-4 w-4" />
      case 'decision':
        return <GitBranch className="h-4 w-4" />
      case 'start':
        return <Play className="h-4 w-4" />
      case 'end':
        return <div className="h-4 w-4 bg-current rounded-full" />
      case 'subgraph':
        return <Users className="h-4 w-4" />
      case 'supervisor':
        return <Settings className="h-4 w-4" />
      case 'custom':
        return <Zap className="h-4 w-4" />
      default:
        return <div className="h-4 w-4 bg-current rounded" />
    }
  }

  const getNodeStatusIcon = (status?: string) => {
    switch (status) {
      case 'running':
        return <Activity className="h-3 w-3 text-blue-500 animate-pulse" />
      case 'completed':
        return <CheckCircle className="h-3 w-3 text-green-500" />
      case 'error':
        return <AlertTriangle className="h-3 w-3 text-red-500" />
      case 'warning':
        return <AlertTriangle className="h-3 w-3 text-yellow-500" />
      default:
        return <Clock className="h-3 w-3 text-gray-400" />
    }
  }

  // Canvas Interaction Handlers with proper coordinate transformation
  const handleCanvasMouseDown = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return

    const screenPos = { x: e.clientX - rect.left, y: e.clientY - rect.top }
    const worldPos = transformPoint(screenPos, {
      x: canvasState.pan.x,
      y: canvasState.pan.y,
      scale: canvasState.zoom
    })

    if (isAddingNode) {
      setIsAddingNode(false)
      return
    }

    // Clear selection if clicking on empty space
    if (e.target === canvasRef.current) {
      setCanvasState(prev => ({ ...prev, selection: [] }))
      setSelectedNode(null)
    }

    // Start canvas panning
    setCanvasState(prev => ({
      ...prev,
      dragState: {
        ...prev.dragState,
        isDragging: true,
        dragType: 'canvas',
        startPos: { x: e.clientX, y: e.clientY },
        currentPos: { x: e.clientX, y: e.clientY }
      }
    }))
  }, [canvasState, isAddingNode])

  const handleCanvasMouseMove = useCallback(rafThrottle((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return

    const screenPos = { x: e.clientX - rect.left, y: e.clientY - rect.top }
    const worldPos = transformPoint(screenPos, {
      x: canvasState.pan.x,
      y: canvasState.pan.y,
      scale: canvasState.zoom
    })

    // Update current position for connection preview
    setCanvasState(prev => ({
      ...prev,
      dragState: {
        ...prev.dragState,
        currentPos: worldPos
      }
    }))

    if (!canvasState.dragState.isDragging) return

    if (canvasState.dragState.dragType === 'canvas') {
      const deltaX = e.clientX - canvasState.dragState.startPos.x
      const deltaY = e.clientY - canvasState.dragState.startPos.y

      setCanvasState(prev => ({
        ...prev,
        pan: {
          x: prev.pan.x + deltaX,
          y: prev.pan.y + deltaY
        },
        dragState: {
          ...prev.dragState,
          startPos: { x: e.clientX, y: e.clientY }
        }
      }))
    } else if (canvasState.dragState.dragType === 'node') {
      updateNodeDrag(worldPos)
    }
  }), [canvasState, updateNodeDrag])

  const handleCanvasMouseUp = useCallback(() => {
    if (canvasState.dragState.dragType === 'node') {
      endNodeDrag()
    } else if (canvasState.dragState.dragType === 'connection') {
      // Cancel connection if not completed
      setCanvasState(prev => ({
        ...prev,
        dragState: {
          ...prev.dragState,
          isDragging: false,
          dragType: 'canvas',
          connectionStart: undefined
        }
      }))
    }

    setCanvasState(prev => ({
      ...prev,
      dragState: {
        ...prev.dragState,
        isDragging: false,
        dragType: 'canvas'
      }
    }))
  }, [canvasState.dragState.dragType, endNodeDrag])

  const handleCanvasWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()

    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return

    const mousePos = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    }

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1
    const newZoom = Math.max(0.05, Math.min(5, canvasState.zoom * zoomFactor))

    // Zoom towards mouse position
    const zoomRatio = newZoom / canvasState.zoom
    const newPan = {
      x: mousePos.x - (mousePos.x - canvasState.pan.x) * zoomRatio,
      y: mousePos.y - (mousePos.y - canvasState.pan.y) * zoomRatio
    }

    setCanvasState(prev => ({
      ...prev,
      zoom: newZoom,
      pan: newPan
    }))
  }, [canvasState.zoom, canvasState.pan])

  const zoomIn = useCallback(() => {
    setCanvasState(prev => ({
      ...prev,
      zoom: Math.min(5, prev.zoom * 1.2)
    }))
  }, [])

  const zoomOut = useCallback(() => {
    setCanvasState(prev => ({
      ...prev,
      zoom: Math.max(0.05, prev.zoom / 1.2)
    }))
  }, [])

  const resetZoom = useCallback(() => {
    setCanvasState(prev => ({
      ...prev,
      zoom: 1,
      pan: { x: 0, y: 0 }
    }))
  }, [])

  const fitToScreen = useCallback(() => {
    if (nodes.length === 0) return

    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return

    // Calculate bounding box of all nodes
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity

    nodes.forEach(node => {
      minX = Math.min(minX, node.position.x)
      minY = Math.min(minY, node.position.y)
      maxX = Math.max(maxX, node.position.x + node.size.width)
      maxY = Math.max(maxY, node.position.y + node.size.height)
    })

    const contentWidth = maxX - minX
    const contentHeight = maxY - minY
    const padding = 100

    const scaleX = (rect.width - padding * 2) / contentWidth
    const scaleY = (rect.height - padding * 2) / contentHeight
    const scale = Math.min(scaleX, scaleY, 2) // Max zoom of 2x

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2

    const panX = rect.width / 2 - centerX * scale
    const panY = rect.height / 2 - centerY * scale

    setCanvasState(prev => ({
      ...prev,
      zoom: scale,
      pan: { x: panX, y: panY }
    }))
  }, [nodes])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Prevent shortcuts when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'z':
            e.preventDefault()
            if (e.shiftKey) {
              redo()
            } else {
              undo()
            }
            break
          case 'c':
            e.preventDefault()
            copySelectedNodes()
            break
          case 'v':
            e.preventDefault()
            pasteNodes()
            break
          case 'a':
            e.preventDefault()
            setCanvasState(prev => ({
              ...prev,
              selection: nodes.map(node => node.id)
            }))
            break
          case 's':
            e.preventDefault()
            // Save workflow
            console.log('Save workflow')
            break
        }
      } else {
        switch (e.key) {
          case 'Delete':
          case 'Backspace':
            e.preventDefault()
            if (selectedNode) {
              deleteNode(selectedNode)
            }
            break
          case 'Escape':
            e.preventDefault()
            setSelectedNode(null)
            setCanvasState(prev => ({ ...prev, selection: [] }))
            setIsAddingNode(false)
            break
          case ' ':
            e.preventDefault()
            if (canvasState.executionState.isExecuting) {
              stopExecution()
            } else {
              startExecution()
            }
            break
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedNode, nodes, canvasState.executionState.isExecuting, undo, redo, copySelectedNodes, pasteNodes, deleteNode, startExecution, stopExecution])



  const getNodeColor = (node: WorkflowNode) => {
    const { type, data, validation } = node

    // Status-based coloring
    if (data.status === 'running') {
      return 'bg-blue-100 border-blue-400 text-blue-800 dark:bg-blue-900/30 dark:border-blue-600 dark:text-blue-300 shadow-lg shadow-blue-200 dark:shadow-blue-900/50'
    }
    if (data.status === 'completed') {
      return 'bg-green-100 border-green-400 text-green-800 dark:bg-green-900/30 dark:border-green-600 dark:text-green-300'
    }
    if (data.status === 'error') {
      return 'bg-red-100 border-red-400 text-red-800 dark:bg-red-900/30 dark:border-red-600 dark:text-red-300'
    }
    if (!validation?.isValid) {
      return 'bg-red-50 border-red-300 text-red-700 dark:bg-red-900/20 dark:border-red-700 dark:text-red-400'
    }
    if (validation?.warnings.length > 0) {
      return 'bg-yellow-50 border-yellow-300 text-yellow-700 dark:bg-yellow-900/20 dark:border-yellow-700 dark:text-yellow-400'
    }

    // Type-based coloring
    switch (type) {
      case 'agent':
        return 'bg-blue-50 border-blue-300 text-blue-800 dark:bg-blue-900/20 dark:border-blue-700 dark:text-blue-400'
      case 'decision':
        return 'bg-yellow-50 border-yellow-300 text-yellow-800 dark:bg-yellow-900/20 dark:border-yellow-700 dark:text-yellow-400'
      case 'start':
        return 'bg-green-50 border-green-300 text-green-800 dark:bg-green-900/20 dark:border-green-700 dark:text-green-400'
      case 'end':
        return 'bg-red-50 border-red-300 text-red-800 dark:bg-red-900/20 dark:border-red-700 dark:text-red-400'
      case 'subgraph':
        return 'bg-purple-50 border-purple-300 text-purple-800 dark:bg-purple-900/20 dark:border-purple-700 dark:text-purple-400'
      case 'supervisor':
        return 'bg-indigo-50 border-indigo-300 text-indigo-800 dark:bg-indigo-900/20 dark:border-indigo-700 dark:text-indigo-400'
      case 'custom':
        return 'bg-gray-50 border-gray-300 text-gray-800 dark:bg-gray-900/20 dark:border-gray-700 dark:text-gray-400'
      default:
        return 'bg-gray-50 border-gray-300 text-gray-800 dark:bg-gray-900/20 dark:border-gray-700 dark:text-gray-400'
    }
  }

  return (
    <CanvasErrorBoundary>
      <div className="h-[800px] bg-muted/20 border-2 border-dashed border-muted-foreground/20 rounded-lg relative overflow-hidden">
        {/* Enhanced Toolbar */}
        <div className="absolute top-4 left-4 z-50 flex items-center space-x-2">
        {/* Main Tools */}
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-1 shadow-lg">
          <button
            onClick={() => setIsAddingNode(!isAddingNode)}
            className={`p-2 rounded-md transition-colors ${
              isAddingNode
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-accent'
            }`}
            title="Add Node"
          >
            <Plus className="h-4 w-4" />
          </button>
          <button
            onClick={autoLayout}
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Hierarchical Layout"
          >
            <Brain className="h-4 w-4" />
          </button>
          <button
            onClick={forceLayout}
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Force-Directed Layout"
          >
            <GitBranch className="h-4 w-4" />
          </button>
          <button
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Save Workflow"
          >
            <Save className="h-4 w-4" />
          </button>
          <button
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Export Workflow"
          >
            <Download className="h-4 w-4" />
          </button>
          <button
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Import Workflow"
          >
            <Upload className="h-4 w-4" />
          </button>
        </div>

        {/* Execution Controls */}
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-1 shadow-lg">
          {canvasState.executionState.isExecuting ? (
            <button
              onClick={stopExecution}
              className="p-2 rounded-md hover:bg-red-100 dark:hover:bg-red-900/20 text-red-600 transition-colors"
              title="Stop Execution"
            >
              <div className="h-4 w-4 bg-current rounded-sm" />
            </button>
          ) : (
            <button
              onClick={startExecution}
              className="p-2 rounded-md hover:bg-green-100 dark:hover:bg-green-900/20 text-green-600 transition-colors"
              title="Start Execution"
            >
              <Play className="h-4 w-4" />
            </button>
          )}
          <button
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Debug Mode"
          >
            <Activity className="h-4 w-4" />
          </button>
        </div>

        {/* Edit Controls */}
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-1 shadow-lg">
          <button
            onClick={undo}
            disabled={canvasState.history.past.length === 0}
            className="p-2 rounded-md hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Undo (Ctrl+Z)"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
          <button
            onClick={redo}
            disabled={canvasState.history.future.length === 0}
            className="p-2 rounded-md hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Redo (Ctrl+Shift+Z)"
          >
            <RotateCcw className="h-4 w-4 scale-x-[-1]" />
          </button>
          <button
            onClick={copySelectedNodes}
            disabled={canvasState.selection.length === 0}
            className="p-2 rounded-md hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Copy (Ctrl+C)"
          >
            <Copy className="h-4 w-4" />
          </button>
          <button
            onClick={pasteNodes}
            disabled={canvasState.clipboard.nodes.length === 0}
            className="p-2 rounded-md hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Paste (Ctrl+V)"
          >
            <Copy className="h-4 w-4 scale-x-[-1]" />
          </button>
          <button
            onClick={() => {
              setCanvasState(prev => ({
                ...prev,
                selection: nodes.map(node => node.id)
              }))
            }}
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Select All (Ctrl+A)"
          >
            <Square className="h-4 w-4" />
          </button>
        </div>

        {/* View Controls */}
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-1 shadow-lg">
          <button
            onClick={zoomIn}
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            onClick={zoomOut}
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            onClick={resetZoom}
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Reset Zoom"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
          <button
            onClick={fitToScreen}
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Fit to Screen"
          >
            <Maximize className="h-4 w-4" />
          </button>
          <button
            onClick={() => setCanvasState(prev => ({
              ...prev,
              settings: {
                ...prev.settings,
                showGrid: !prev.settings.showGrid
              }
            }))}
            className={`p-2 rounded-md transition-colors ${
              canvasState.settings.showGrid
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-accent'
            }`}
            title="Toggle Grid"
          >
            <Grid className="h-4 w-4" />
          </button>
        </div>

        {/* Settings Panel Toggle */}
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-1 shadow-lg">
          <button
            className="p-2 rounded-md hover:bg-accent transition-colors"
            title="Canvas Settings"
          >
            <Settings className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Execution Status */}
      {canvasState.executionState.isExecuting && (
        <div className="absolute top-4 right-4 z-20 bg-background/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
          <div className="flex items-center space-x-2">
            <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
            <span className="text-sm font-medium">Executing Workflow</span>
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            Current: {canvasState.executionState.currentNode}
          </div>
          <div className="text-xs text-muted-foreground">
            Steps: {canvasState.executionState.executionPath.length}
          </div>
        </div>
      )}

      {/* Performance & Status Indicators */}
      <div className="absolute bottom-4 right-4 z-20 space-y-2">
        {/* Zoom Indicator */}
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg px-3 py-1 shadow-lg">
          <span className="text-xs font-medium">{Math.round(canvasState.zoom * 100)}%</span>
        </div>

        {/* Performance Monitor */}
        {canvasState.executionState.isExecuting && (
          <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-2 shadow-lg">
            <div className="text-xs space-y-1">
              <div className="flex items-center justify-between space-x-2">
                <span className="text-muted-foreground">Execution:</span>
                <span className="font-medium text-blue-600">
                  {canvasState.executionState.startTime
                    ? `${Math.round((Date.now() - canvasState.executionState.startTime) / 1000)}s`
                    : '0s'
                  }
                </span>
              </div>
              <div className="flex items-center justify-between space-x-2">
                <span className="text-muted-foreground">Current:</span>
                <span className="font-medium">
                  {nodes.find(n => n.id === canvasState.executionState.currentNode)?.data.label || 'None'}
                </span>
              </div>
              <div className="flex items-center justify-between space-x-2">
                <span className="text-muted-foreground">Progress:</span>
                <span className="font-medium">
                  {canvasState.executionState.executionPath.length}/{nodes.length}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Collaboration Status */}
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-2 shadow-lg">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-xs text-muted-foreground">Live</span>
          </div>
        </div>
      </div>

      {/* Enhanced Node Palette */}
      {isAddingNode && (
        <div className="absolute top-20 left-4 z-40 bg-background/95 backdrop-blur-sm border border-border rounded-lg shadow-xl p-4 w-80 max-h-[600px] overflow-y-auto">
          <h3 className="text-lg font-semibold text-foreground mb-4">Node Library</h3>

          {/* Agent Nodes */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-muted-foreground mb-2 flex items-center">
              <Bot className="h-4 w-4 mr-2" />
              Agent Nodes
            </h4>
            <div className="space-y-2">
              {Object.entries(nodeTemplates.agent).map(([key, template]) => (
                <button
                  key={key}
                  onClick={() => addNode('agent', key, { x: 300, y: 200 })}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent/50 transition-all text-left group"
                >
                  <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/20 text-blue-600 group-hover:bg-blue-200 dark:group-hover:bg-blue-900/40">
                    <span className="text-lg">{template.icon}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{template.label}</div>
                    <div className="text-xs text-muted-foreground truncate">{template.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Decision Nodes */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-muted-foreground mb-2 flex items-center">
              <GitBranch className="h-4 w-4 mr-2" />
              Decision Nodes
            </h4>
            <div className="space-y-2">
              {Object.entries(nodeTemplates.decision).map(([key, template]) => (
                <button
                  key={key}
                  onClick={() => addNode('decision', key, { x: 300, y: 200 })}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent/50 transition-all text-left group"
                >
                  <div className="p-2 rounded-lg bg-yellow-100 dark:bg-yellow-900/20 text-yellow-600 group-hover:bg-yellow-200 dark:group-hover:bg-yellow-900/40">
                    <span className="text-lg">{template.icon}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{template.label}</div>
                    <div className="text-xs text-muted-foreground truncate">{template.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Subgraph Nodes */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-muted-foreground mb-2 flex items-center">
              <Users className="h-4 w-4 mr-2" />
              Subgraph Nodes
            </h4>
            <div className="space-y-2">
              {Object.entries(nodeTemplates.subgraph).map(([key, template]) => (
                <button
                  key={key}
                  onClick={() => addNode('subgraph', key, { x: 300, y: 200 })}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent/50 transition-all text-left group"
                >
                  <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/20 text-purple-600 group-hover:bg-purple-200 dark:group-hover:bg-purple-900/40">
                    <span className="text-lg">{template.icon}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{template.label}</div>
                    <div className="text-xs text-muted-foreground truncate">{template.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Supervisor Nodes */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-muted-foreground mb-2 flex items-center">
              <Settings className="h-4 w-4 mr-2" />
              Supervisor Nodes
            </h4>
            <div className="space-y-2">
              {Object.entries(nodeTemplates.supervisor).map(([key, template]) => (
                <button
                  key={key}
                  onClick={() => addNode('supervisor', key, { x: 300, y: 200 })}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent/50 transition-all text-left group"
                >
                  <div className="p-2 rounded-lg bg-indigo-100 dark:bg-indigo-900/20 text-indigo-600 group-hover:bg-indigo-200 dark:group-hover:bg-indigo-900/40">
                    <span className="text-lg">{template.icon}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{template.label}</div>
                    <div className="text-xs text-muted-foreground truncate">{template.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Custom Nodes */}
          <div className="mb-4">
            <h4 className="text-sm font-medium text-muted-foreground mb-2 flex items-center">
              <Zap className="h-4 w-4 mr-2" />
              Custom Nodes
            </h4>
            <div className="space-y-2">
              {Object.entries(nodeTemplates.custom).map(([key, template]) => (
                <button
                  key={key}
                  onClick={() => addNode('custom', key, { x: 300, y: 200 })}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent/50 transition-all text-left group"
                >
                  <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-900/20 text-gray-600 group-hover:bg-gray-200 dark:group-hover:bg-gray-900/40">
                    <span className="text-lg">{template.icon}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{template.label}</div>
                    <div className="text-xs text-muted-foreground truncate">{template.description}</div>
                  </div>
                </button>
              ))}

              {/* End Node */}
              <button
                onClick={() => addNode('end', 'default', { x: 300, y: 200 })}
                className="w-full flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent/50 transition-all text-left group"
              >
                <div className="p-2 rounded-lg bg-red-100 dark:bg-red-900/20 text-red-600 group-hover:bg-red-200 dark:group-hover:bg-red-900/40">
                  <div className="h-4 w-4 bg-current rounded-full" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-foreground">End Node</div>
                  <div className="text-xs text-muted-foreground">Workflow termination point</div>
                </div>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Grid Background - Fixed Layer */}
      {canvasState.settings.showGrid && (
        <div
          className="absolute inset-0 pointer-events-none opacity-20"
          style={{
            backgroundImage: `
              linear-gradient(hsl(var(--muted-foreground)) 1px, transparent 1px),
              linear-gradient(90deg, hsl(var(--muted-foreground)) 1px, transparent 1px)
            `,
            backgroundSize: `${canvasState.settings.gridSize * canvasState.zoom}px ${canvasState.settings.gridSize * canvasState.zoom}px`,
            backgroundPosition: `${canvasState.pan.x % (canvasState.settings.gridSize * canvasState.zoom)}px ${canvasState.pan.y % (canvasState.settings.gridSize * canvasState.zoom)}px`
          }}
        />
      )}

      {/* Enhanced Canvas */}
      <div
        ref={canvasRef}
        className="relative w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
        onWheel={handleCanvasWheel}
      >
        {/* Content Layer with Transform */}
        <div
          className="absolute inset-0"
          style={{
            transform: `translate(${canvasState.pan.x}px, ${canvasState.pan.y}px) scale(${canvasState.zoom})`,
            transformOrigin: '0 0'
          }}
        >

        {/* Connection Lines (SVG) */}
        <svg
          ref={svgRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ overflow: 'visible' }}
        >
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
            <marker
              id="arrowhead-animated"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill="#10b981"
              />
            </marker>
          </defs>

          {edges.map((edge) => {
            const sourceNode = nodes.find(n => n.id === edge.source)
            const targetNode = nodes.find(n => n.id === edge.target)
            const sourcePort = sourceNode?.ports.outputs.find(p => p.id === edge.sourcePort)
            const targetPort = targetNode?.ports.inputs.find(p => p.id === edge.targetPort)

            if (!sourceNode || !targetNode || !sourcePort || !targetPort) return null

            const x1 = sourceNode.position.x + sourcePort.position.x
            const y1 = sourceNode.position.y + sourcePort.position.y
            const x2 = targetNode.position.x + targetPort.position.x
            const y2 = targetNode.position.y + targetPort.position.y

            // Bezier curve calculation
            const dx = x2 - x1
            const dy = y2 - y1
            const controlOffset = Math.max(50, Math.abs(dx) * 0.5)
            const cx1 = x1 + controlOffset
            const cy1 = y1
            const cx2 = x2 - controlOffset
            const cy2 = y2

            return (
              <g key={edge.id}>
                <path
                  d={`M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`}
                  stroke={edge.style?.stroke || '#6b7280'}
                  strokeWidth={edge.style?.strokeWidth || 2}
                  strokeDasharray={edge.style?.strokeDasharray}
                  fill="none"
                  markerEnd={edge.animated ? "url(#arrowhead-animated)" : "url(#arrowhead)"}
                  className={`transition-all duration-300 ${
                    edge.animated ? 'animate-pulse' : ''
                  } ${
                    selectedEdge === edge.id ? 'stroke-primary' : ''
                  }`}
                  onClick={() => setSelectedEdge(edge.id)}
                  style={{ cursor: 'pointer', pointerEvents: 'stroke' }}
                />

                {/* Edge Label */}
                {edge.label && (
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 - 8}
                    textAnchor="middle"
                    className="fill-foreground text-xs font-medium pointer-events-none"
                    style={{ userSelect: 'none' }}
                  >
                    {edge.label}
                  </text>
                )}

                {/* Data Flow Animation */}
                {edge.animated && (
                  <circle
                    r="3"
                    fill="#10b981"
                    className="opacity-80"
                  >
                    <animateMotion
                      dur="2s"
                      repeatCount="indefinite"
                      path={`M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`}
                    />
                  </circle>
                )}
              </g>
            )
          })}

          {/* Active Connection Line */}
          {canvasState.dragState.isDragging &&
           canvasState.dragState.dragType === 'connection' &&
           canvasState.dragState.connectionStart && (
            <line
              x1={canvasState.dragState.connectionStart.position.x}
              y1={canvasState.dragState.connectionStart.position.y}
              x2={canvasState.dragState.currentPos.x}
              y2={canvasState.dragState.currentPos.y}
              stroke="#3b82f6"
              strokeWidth="2"
              strokeDasharray="5,5"
              className="animate-pulse"
            />
          )}
        </svg>

        {/* Enhanced Interactive Nodes */}
        {nodes.map((node) => (
          <div
            key={node.id}
            className={`absolute border-2 rounded-lg cursor-move transition-all duration-200 ${
              getNodeColor(node)
            } ${
              selectedNode === node.id ? 'ring-2 ring-primary ring-offset-2 shadow-lg' : 'hover:shadow-md'
            } ${
              canvasState.executionState.currentNode === node.id ? 'animate-pulse shadow-lg shadow-blue-200 dark:shadow-blue-900/50' : ''
            }`}
            style={{
              left: node.position.x,
              top: node.position.y,
              width: node.size.width,
              height: node.size.height
            }}
            onMouseDown={(e) => {
              e.stopPropagation()
              const rect = canvasRef.current?.getBoundingClientRect()
              if (!rect) return

              const x = (e.clientX - rect.left - canvasState.pan.x) / canvasState.zoom
              const y = (e.clientY - rect.top - canvasState.pan.y) / canvasState.zoom

              setSelectedNode(node.id)
              startNodeDrag(node.id, { x, y })
            }}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedNode(node.id)
            }}
          >
            {/* Node Header */}
            <div className="flex items-center justify-between p-3 border-b border-current/20">
              <div className="flex items-center space-x-2 flex-1 min-w-0">
                {getNodeIcon(node)}
                <span className="text-sm font-medium truncate">{node.data.label}</span>
                {getNodeStatusIcon(node.data.status)}
              </div>

              {/* Node Actions */}
              {selectedNode === node.id && (
                <div className="flex items-center space-x-1 ml-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      // Open node settings
                    }}
                    className="p-1 rounded hover:bg-current/10 transition-colors"
                    title="Node Settings"
                  >
                    <Settings className="h-3 w-3" />
                  </button>
                  {node.id !== 'start' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteNode(node.id)
                      }}
                      className="p-1 rounded hover:bg-red-200 dark:hover:bg-red-900/20 text-red-600 transition-colors"
                      title="Delete Node"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Node Content */}
            <div className="p-3">
              {/* Node Type Info */}
              {node.data.agentType && (
                <div className="text-xs text-muted-foreground mb-2">
                  {node.data.agentType} • {node.data.model}
                </div>
              )}

              {/* Execution Info */}
              {node.data.executionTime && (
                <div className="text-xs text-muted-foreground mb-2">
                  Last: {node.data.executionTime.toFixed(0)}ms
                </div>
              )}

              {/* Validation Errors */}
              {node.validation && !node.validation.isValid && (
                <div className="text-xs text-red-600 mb-2">
                  {node.validation.errors[0]}
                </div>
              )}

              {/* Validation Warnings */}
              {node.validation && node.validation.warnings.length > 0 && (
                <div className="text-xs text-yellow-600 mb-2">
                  {node.validation.warnings[0]}
                </div>
              )}
            </div>

            {/* Input Ports */}
            {node.ports.inputs.map((port) => (
              <div
                key={port.id}
                className={`absolute w-3 h-3 border-2 border-background rounded-full cursor-pointer transition-all ${
                  port.connected
                    ? 'bg-green-500 hover:bg-green-600'
                    : 'bg-gray-400 hover:bg-gray-500'
                } ${
                  port.dataType === 'text' ? 'border-blue-300' :
                  port.dataType === 'number' ? 'border-green-300' :
                  port.dataType === 'boolean' ? 'border-purple-300' :
                  port.dataType === 'object' ? 'border-orange-300' :
                  port.dataType === 'array' ? 'border-red-300' :
                  'border-gray-300'
                }`}
                style={{
                  left: port.position.x - 6,
                  top: port.position.y - 6
                }}
                onMouseDown={(e) => {
                  e.stopPropagation()
                  if (canvasState.dragState.isDragging &&
                      canvasState.dragState.dragType === 'connection' &&
                      canvasState.dragState.connectionStart) {
                    completeConnection(node.id, port.id)
                  }
                }}
                title={`${port.label} (${port.dataType})`}
              />
            ))}

            {/* Output Ports */}
            {node.ports.outputs.map((port) => (
              <div
                key={port.id}
                className={`absolute w-3 h-3 border-2 border-background rounded-full cursor-pointer transition-all ${
                  port.connected
                    ? 'bg-green-500 hover:bg-green-600'
                    : 'bg-gray-400 hover:bg-gray-500'
                } ${
                  port.dataType === 'text' ? 'border-blue-300' :
                  port.dataType === 'number' ? 'border-green-300' :
                  port.dataType === 'boolean' ? 'border-purple-300' :
                  port.dataType === 'object' ? 'border-orange-300' :
                  port.dataType === 'array' ? 'border-red-300' :
                  'border-gray-300'
                }`}
                style={{
                  left: port.position.x - 6,
                  top: port.position.y - 6
                }}
                onMouseDown={(e) => {
                  e.stopPropagation()
                  startConnection(node.id, port.id, 'output', e)
                }}
                title={`${port.label} (${port.dataType})`}
              />
            ))}

            {/* Port Labels */}
            {selectedNode === node.id && (
              <>
                {node.ports.inputs.map((port) => (
                  <div
                    key={`${port.id}-label`}
                    className="absolute text-xs text-muted-foreground pointer-events-none"
                    style={{
                      left: port.position.x - 40,
                      top: port.position.y - 2,
                      textAlign: 'right',
                      width: '35px'
                    }}
                  >
                    {port.label}
                  </div>
                ))}
                {node.ports.outputs.map((port) => (
                  <div
                    key={`${port.id}-label`}
                    className="absolute text-xs text-muted-foreground pointer-events-none"
                    style={{
                      left: port.position.x + 10,
                      top: port.position.y - 2
                    }}
                  >
                    {port.label}
                  </div>
                ))}
              </>
            )}
          </div>
        ))}

        {/* Enhanced Empty State */}
        {nodes.length === 1 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center max-w-md">
              <div className="relative mb-6">
                <GitBranch className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                <div className="absolute -top-2 -right-2">
                  <Brain className="h-6 w-6 text-primary animate-pulse" />
                </div>
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-3">
                Design Your AI Workflow
              </h3>
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Create powerful multi-agent workflows with drag-and-drop simplicity.
                Connect AI agents, add decision points, and build complex automation flows.
              </p>
              <div className="space-y-3 pointer-events-auto">
                <button
                  onClick={() => setIsAddingNode(true)}
                  className="btn-primary inline-flex items-center px-6 py-3 text-base"
                >
                  <Plus className="h-5 w-5 mr-2" />
                  Add Your First Node
                </button>
                <div className="flex items-center justify-center space-x-4 text-sm text-muted-foreground">
                  <div className="flex items-center">
                    <Brain className="h-4 w-4 mr-1" />
                    AI-Powered
                  </div>
                  <div className="flex items-center">
                    <Zap className="h-4 w-4 mr-1" />
                    Real-time
                  </div>
                  <div className="flex items-center">
                    <Users className="h-4 w-4 mr-1" />
                    Collaborative
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Workflow Statistics Overlay */}
        {nodes.length > 1 && (
          <div className="absolute bottom-4 left-4 z-10 bg-background/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex items-center justify-between space-x-4">
                <span>Nodes:</span>
                <span className="font-medium">{nodes.length}</span>
              </div>
              <div className="flex items-center justify-between space-x-4">
                <span>Connections:</span>
                <span className="font-medium">{edges.length}</span>
              </div>
              <div className="flex items-center justify-between space-x-4">
                <span>Valid:</span>
                <span className={`font-medium ${
                  nodes.every(n => n.validation?.isValid) ? 'text-green-600' : 'text-red-600'
                }`}>
                  {nodes.filter(n => n.validation?.isValid).length}/{nodes.length}
                </span>
              </div>
            </div>
          </div>
        )}
        </div>
      </div>

      {/* Enhanced Properties Panel */}
      {selectedNode && (
        <div className="absolute top-4 right-4 w-80 z-30 bg-background/95 backdrop-blur-sm border border-border rounded-lg shadow-xl max-h-[calc(100vh-8rem)] overflow-y-auto">
          {(() => {
            const node = nodes.find(n => n.id === selectedNode)
            if (!node) return null

            return (
              <>
                {/* Panel Header */}
                <div className="sticky top-0 bg-background/95 backdrop-blur-sm border-b border-border p-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-foreground flex items-center">
                      {getNodeIcon(node)}
                      <span className="ml-2">Node Properties</span>
                    </h3>
                    <button
                      onClick={() => setSelectedNode(null)}
                      className="p-1 rounded hover:bg-accent transition-colors"
                    >
                      <span className="sr-only">Close</span>
                      ×
                    </button>
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">
                    {node.type} • {node.id}
                  </div>
                </div>

                <div className="p-4 space-y-6">
                  {/* Basic Properties */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-foreground border-b border-border pb-2">
                      Basic Configuration
                    </h4>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Label
                      </label>
                      <input
                        type="text"
                        value={node.data.label}
                        onChange={(e) => updateNode(selectedNode, { label: e.target.value })}
                        className="input text-sm"
                        placeholder="Node label..."
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Description
                      </label>
                      <textarea
                        value={node.data.description || ''}
                        onChange={(e) => updateNode(selectedNode, { description: e.target.value })}
                        className="input text-sm resize-none"
                        rows={3}
                        placeholder="Describe this node's purpose..."
                      />
                    </div>
                  </div>

                  {/* Agent-specific Properties */}
                  {node.type === 'agent' && (
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-foreground border-b border-border pb-2">
                        Agent Configuration
                      </h4>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Agent Type
                        </label>
                        <select
                          value={node.data.agentType || 'general'}
                          onChange={(e) => updateNode(selectedNode, { agentType: e.target.value })}
                          className="input text-sm"
                        >
                          <option value="general">General Agent</option>
                          <option value="research">Research Agent</option>
                          <option value="workflow">Workflow Agent</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Model
                        </label>
                        <select
                          value={node.data.model || 'llama3.2:latest'}
                          onChange={(e) => updateNode(selectedNode, { model: e.target.value })}
                          className="input text-sm"
                        >
                          <option value="llama3.2:latest">Llama 3.2 Latest</option>
                          <option value="llama3.1:latest">Llama 3.1 Latest</option>
                          <option value="qwen2.5:latest">Qwen 2.5 Latest</option>
                          <option value="mistral:latest">Mistral Latest</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Temperature: {node.data.customConfig?.temperature || 0.7}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={node.data.customConfig?.temperature || 0.7}
                          onChange={(e) => updateNode(selectedNode, {
                            customConfig: {
                              ...node.data.customConfig,
                              temperature: parseFloat(e.target.value)
                            }
                          })}
                          className="w-full"
                        />
                      </div>
                    </div>
                  )}

                  {/* Decision-specific Properties */}
                  {node.type === 'decision' && (
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-foreground border-b border-border pb-2">
                        Decision Logic
                      </h4>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Condition
                        </label>
                        <textarea
                          value={node.data.customConfig?.condition || ''}
                          onChange={(e) => updateNode(selectedNode, {
                            customConfig: {
                              ...node.data.customConfig,
                              condition: e.target.value
                            }
                          })}
                          className="input text-sm resize-none font-mono"
                          rows={4}
                          placeholder="Enter condition logic (e.g., input.value > 0.5)"
                        />
                      </div>
                    </div>
                  )}

                  {/* Connection Ports */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-foreground border-b border-border pb-2">
                      Connection Ports
                    </h4>

                    {/* Input Ports */}
                    {node.ports.inputs.length > 0 && (
                      <div>
                        <h5 className="text-xs font-medium text-muted-foreground mb-2">Input Ports</h5>
                        <div className="space-y-2">
                          {node.ports.inputs.map((port) => (
                            <div key={port.id} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                              <div className="flex items-center space-x-2">
                                <div className={`w-2 h-2 rounded-full ${
                                  port.connected ? 'bg-green-500' : 'bg-gray-400'
                                }`} />
                                <span className="text-sm">{port.label}</span>
                              </div>
                              <span className="text-xs text-muted-foreground">{port.dataType}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Output Ports */}
                    {node.ports.outputs.length > 0 && (
                      <div>
                        <h5 className="text-xs font-medium text-muted-foreground mb-2">Output Ports</h5>
                        <div className="space-y-2">
                          {node.ports.outputs.map((port) => (
                            <div key={port.id} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                              <div className="flex items-center space-x-2">
                                <div className={`w-2 h-2 rounded-full ${
                                  port.connected ? 'bg-green-500' : 'bg-gray-400'
                                }`} />
                                <span className="text-sm">{port.label}</span>
                              </div>
                              <span className="text-xs text-muted-foreground">{port.dataType}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Execution Status */}
                  {node.data.status && node.data.status !== 'idle' && (
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-foreground border-b border-border pb-2">
                        Execution Status
                      </h4>

                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Status:</span>
                          <div className="flex items-center space-x-1">
                            {getNodeStatusIcon(node.data.status)}
                            <span className="text-sm capitalize">{node.data.status}</span>
                          </div>
                        </div>

                        {node.data.executionTime && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Duration:</span>
                            <span className="text-sm">{node.data.executionTime.toFixed(0)}ms</span>
                          </div>
                        )}

                        {node.data.lastExecuted && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Last Run:</span>
                            <span className="text-sm">{new Date(node.data.lastExecuted).toLocaleTimeString()}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Validation Status */}
                  {node.validation && (!node.validation.isValid || node.validation.warnings.length > 0) && (
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-foreground border-b border-border pb-2">
                        Validation
                      </h4>

                      {node.validation.errors.length > 0 && (
                        <div className="space-y-2">
                          <h5 className="text-xs font-medium text-red-600">Errors</h5>
                          {node.validation.errors.map((error, index) => (
                            <div key={index} className="flex items-start space-x-2 p-2 bg-red-50 dark:bg-red-900/20 rounded">
                              <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                              <span className="text-sm text-red-700 dark:text-red-400">{error}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {node.validation.warnings.length > 0 && (
                        <div className="space-y-2">
                          <h5 className="text-xs font-medium text-yellow-600">Warnings</h5>
                          {node.validation.warnings.map((warning, index) => (
                            <div key={index} className="flex items-start space-x-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                              <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                              <span className="text-sm text-yellow-700 dark:text-yellow-400">{warning}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Advanced Actions */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-foreground border-b border-border pb-2">
                      Actions
                    </h4>

                    <div className="grid grid-cols-2 gap-2">
                      <button className="btn-secondary text-xs py-2">
                        <Eye className="h-3 w-3 mr-1" />
                        Preview
                      </button>
                      <button className="btn-secondary text-xs py-2">
                        <Settings className="h-3 w-3 mr-1" />
                        Configure
                      </button>
                      <button className="btn-secondary text-xs py-2">
                        <BarChart3 className="h-3 w-3 mr-1" />
                        Analytics
                      </button>
                      <button className="btn-secondary text-xs py-2">
                        <MessageSquare className="h-3 w-3 mr-1" />
                        Test
                      </button>
                    </div>
                  </div>
                </div>
              </>
            )
          })()}
        </div>
      )}
    </div>
    </CanvasErrorBoundary>
  )
}

export default WorkflowCanvas
