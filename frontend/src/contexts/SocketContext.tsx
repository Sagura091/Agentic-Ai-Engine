import React, { createContext, useContext, useEffect, useState } from 'react'
import { io, Socket } from 'socket.io-client'
import toast from 'react-hot-toast'

interface SocketContextType {
  socket: Socket | null
  isConnected: boolean
  executeAgent: (data: {
    agentId: string
    message: string
    options?: any
  }) => void
  executeWorkflow: (data: {
    workflowId: string
    task: string
    options?: any
  }) => void
  createAgentViaSocket: (data: {
    agentType: string
    name: string
    description: string
    config?: any
    tools?: string[]
  }) => void
  createToolViaSocket: (data: {
    name: string
    description: string
    functionalityDescription: string
    assignToAgent?: string
    makeGlobal?: boolean
  }) => void
  getSystemStatus: () => void
  joinAgentMonitoring: () => void
  joinWorkflowMonitoring: () => void
}

const SocketContext = createContext<SocketContextType | undefined>(undefined)

export function SocketProvider({ children }: { children: React.ReactNode }) {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    const socketInstance = io(import.meta.env.VITE_SERVER_URL || 'http://localhost:8888', {
      transports: ['websocket', 'polling'],
      timeout: 20000,
    })

    socketInstance.on('connect', () => {
      console.log('Connected to server')
      setIsConnected(true)
      toast.success('Connected to Agentic AI Service')
    })

    socketInstance.on('disconnect', () => {
      console.log('Disconnected from server')
      setIsConnected(false)
      toast.error('Disconnected from Agentic AI Service')
    })

    socketInstance.on('connect_error', (error) => {
      console.error('Connection error:', error)
      setIsConnected(false)
      toast.error('Failed to connect to Agentic AI Service')
    })

    // Agent execution events
    socketInstance.on('agent-execution-started', (data) => {
      console.log('Agent execution started:', data)
      toast.loading(`Agent ${data.agentId} is processing...`, {
        id: `agent-${data.agentId}`,
      })
    })

    socketInstance.on('agent-execution-completed', (data) => {
      console.log('Agent execution completed:', data)
      toast.success(`Agent completed in ${data.executionTime?.toFixed(2)}s`, {
        id: `agent-${data.agentId}`,
      })
    })

    socketInstance.on('agent-execution-error', (data) => {
      console.error('Agent execution error:', data)
      toast.error(`Agent execution failed: ${data.error}`, {
        id: `agent-${data.agentId}`,
      })
    })

    // Agent creation events
    socketInstance.on('agent_created', (data) => {
      console.log('Agent created:', data)
      toast.success(`Agent "${data.agent_info?.name || data.agent_id}" created successfully!`)
    })

    socketInstance.on('tool_created', (data) => {
      console.log('Tool created:', data)
      toast.success(`Tool "${data.tool_name}" created successfully!`)
    })

    // Workflow execution events
    socketInstance.on('workflow-execution-started', (data) => {
      console.log('Workflow execution started:', data)
      toast.loading(`Workflow ${data.workflowId} is running...`, {
        id: `workflow-${data.workflowId}`,
      })
    })

    socketInstance.on('workflow-execution-completed', (data) => {
      console.log('Workflow execution completed:', data)
      toast.success(`Workflow completed in ${data.executionTime?.toFixed(2)}s`, {
        id: `workflow-${data.workflowId}`,
      })
    })

    socketInstance.on('workflow-execution-error', (data) => {
      console.error('Workflow execution error:', data)
      toast.error(`Workflow execution failed: ${data.error}`, {
        id: `workflow-${data.workflowId}`,
      })
    })

    // Monitoring events
    socketInstance.on('agent-activity', (data) => {
      console.log('Agent activity:', data)
      // This can be used to update real-time monitoring dashboards
    })

    socketInstance.on('workflow-activity', (data) => {
      console.log('Workflow activity:', data)
      // This can be used to update real-time monitoring dashboards
    })

    setSocket(socketInstance)

    return () => {
      socketInstance.disconnect()
    }
  }, [])

  const executeAgent = (data: {
    agentId: string
    message: string
    options?: any
  }) => {
    if (socket && isConnected) {
      socket.emit('execute_agent', {
        type: 'execute_agent',
        data: {
          agent_id: data.agentId,
          task: data.message,
          context: data.options || {}
        }
      })
    } else {
      toast.error('Not connected to server')
    }
  }

  const executeWorkflow = (data: {
    workflowId: string
    task: string
    options?: any
  }) => {
    if (socket && isConnected) {
      socket.emit('execute_workflow', {
        type: 'execute_workflow',
        data: {
          workflow_id: data.workflowId,
          task: data.task,
          workflow_type: data.options?.workflowType || 'multi_agent',
          context: data.options || {}
        }
      })
    } else {
      toast.error('Not connected to server')
    }
  }

  const createAgentViaSocket = (data: {
    agentType: string
    name: string
    description: string
    config?: any
    tools?: string[]
  }) => {
    if (socket && isConnected) {
      socket.emit('create_agent', {
        type: 'create_agent',
        data: {
          agent_type: data.agentType,
          name: data.name,
          description: data.description,
          config: data.config || {},
          tools: data.tools || []
        }
      })
    } else {
      toast.error('Not connected to server')
    }
  }

  const createToolViaSocket = (data: {
    name: string
    description: string
    functionalityDescription: string
    assignToAgent?: string
    makeGlobal?: boolean
  }) => {
    if (socket && isConnected) {
      socket.emit('create_tool', {
        type: 'create_tool',
        data: {
          name: data.name,
          description: data.description,
          functionality_description: data.functionalityDescription,
          assign_to_agent: data.assignToAgent,
          make_global: data.makeGlobal || false
        }
      })
    } else {
      toast.error('Not connected to server')
    }
  }

  const getSystemStatus = () => {
    if (socket && isConnected) {
      socket.emit('get_system_status', {
        type: 'get_system_status',
        data: {}
      })
    } else {
      toast.error('Not connected to server')
    }
  }

  const joinAgentMonitoring = () => {
    if (socket && isConnected) {
      socket.emit('join-agent-monitoring')
    }
  }

  const joinWorkflowMonitoring = () => {
    if (socket && isConnected) {
      socket.emit('join-workflow-monitoring')
    }
  }

  const value: SocketContextType = {
    socket,
    isConnected,
    executeAgent,
    executeWorkflow,
    createAgentViaSocket,
    createToolViaSocket,
    getSystemStatus,
    joinAgentMonitoring,
    joinWorkflowMonitoring,
  }

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  )
}

export function useSocket() {
  const context = useContext(SocketContext)
  if (context === undefined) {
    throw new Error('useSocket must be used within a SocketProvider')
  }
  return context
}
