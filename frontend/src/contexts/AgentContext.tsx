import React, { createContext, useContext, useReducer } from 'react'
import { useQuery } from 'react-query'
import { agentApi, enhancedOrchestrationApi } from '../services/api'
import { useSocket } from './SocketContext'

export interface Agent {
  id: string
  name: string
  description: string
  agent_type: string
  model: string
  status: 'active' | 'idle' | 'executing' | 'error'
  capabilities: string[]
  tools: string[]
  created_at: string
  last_activity?: string
  conversation_id?: string
  system_prompt?: string
  temperature?: number
  max_tokens?: number
}

export interface AgentTemplate {
  id: string
  name: string
  description: string
  agent_type: string
  capabilities: string[]
  system_prompt: string
  icon: string
  category: string
}

interface AgentState {
  agents: Agent[]
  templates: AgentTemplate[]
  selectedAgent: Agent | null
  isLoading: boolean
  error: string | null
}

type AgentAction =
  | { type: 'SET_AGENTS'; payload: Agent[] }
  | { type: 'SET_TEMPLATES'; payload: AgentTemplate[] }
  | { type: 'ADD_AGENT'; payload: Agent }
  | { type: 'UPDATE_AGENT'; payload: { id: string; updates: Partial<Agent> } }
  | { type: 'REMOVE_AGENT'; payload: string }
  | { type: 'SELECT_AGENT'; payload: Agent | null }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }

const initialState: AgentState = {
  agents: [],
  templates: [],
  selectedAgent: null,
  isLoading: false,
  error: null,
}

function agentReducer(state: AgentState, action: AgentAction): AgentState {
  switch (action.type) {
    case 'SET_AGENTS':
      return { ...state, agents: action.payload }
    case 'SET_TEMPLATES':
      return { ...state, templates: action.payload }
    case 'ADD_AGENT':
      return { ...state, agents: [...state.agents, action.payload] }
    case 'UPDATE_AGENT':
      return {
        ...state,
        agents: state.agents.map(agent =>
          agent.id === action.payload.id
            ? { ...agent, ...action.payload.updates }
            : agent
        ),
        selectedAgent:
          state.selectedAgent?.id === action.payload.id
            ? { ...state.selectedAgent, ...action.payload.updates }
            : state.selectedAgent,
      }
    case 'REMOVE_AGENT':
      return {
        ...state,
        agents: state.agents.filter(agent => agent.id !== action.payload),
        selectedAgent:
          state.selectedAgent?.id === action.payload ? null : state.selectedAgent,
      }
    case 'SELECT_AGENT':
      return { ...state, selectedAgent: action.payload }
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload }
    case 'SET_ERROR':
      return { ...state, error: action.payload }
    default:
      return state
  }
}

interface AgentContextType extends AgentState {
  createAgent: (agentData: Partial<Agent>) => Promise<Agent>
  updateAgent: (id: string, updates: Partial<Agent>) => void
  deleteAgent: (id: string) => void
  selectAgent: (agent: Agent | null) => void
  chatWithAgent: (agentId: string, message: string, options?: any) => Promise<any>
  getAgentHistory: (agentId: string) => Promise<any>
  getAgentMetrics: (agentId: string) => Promise<any>
  refreshAgents: () => void
}

const AgentContext = createContext<AgentContextType | undefined>(undefined)

export function AgentProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(agentReducer, initialState)
  const { createAgentViaSocket, socket } = useSocket()

  // Fetch agents
  const { data: agentsData, refetch: refetchAgents } = useQuery(
    'agents',
    agentApi.getAgents,
    {
      onSuccess: (data) => {
        dispatch({ type: 'SET_AGENTS', payload: (data as any).active_agents || [] })
      },
      onError: (error: any) => {
        dispatch({ type: 'SET_ERROR', payload: error.message })
      },
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  )

  // Fetch agent templates
  const { data: templatesData } = useQuery(
    'agent-templates',
    agentApi.getTemplates,
    {
      onSuccess: (data) => {
        dispatch({ type: 'SET_TEMPLATES', payload: (data as any).templates || [] })
      },
      staleTime: 5 * 60 * 1000, // Templates don't change often
    }
  )

  const createAgent = async (agentData: Partial<Agent>): Promise<Agent> => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true })

      // Use Socket.IO for agent creation to bypass proxy issues
      return new Promise((resolve, reject) => {
        if (!socket || !socket.connected) {
          reject(new Error('Not connected to server'))
          return
        }

        // Set up one-time listeners for the response
        const handleAgentCreated = (data: any) => {
          socket.off('agent_created', handleAgentCreated)
          socket.off('error', handleError)

          const newAgent: Agent = {
            id: data.agent_id,
            name: agentData.name || 'Unnamed Agent',
            description: agentData.description || 'No description provided',
            agent_type: (agentData as any).agent_type || 'basic',
            model: (agentData as any).model || 'llama3.2:latest',
            status: 'active',
            capabilities: (agentData as any).capabilities || [],
            tools: (agentData as any).tools || [],
            created_at: new Date(),
            last_activity: new Date()
          }

          dispatch({ type: 'ADD_AGENT', payload: newAgent })
          resolve(newAgent)
        }

        const handleError = (error: any) => {
          socket.off('agent_created', handleAgentCreated)
          socket.off('error', handleError)
          reject(new Error(error.message || 'Failed to create agent'))
        }

        // Listen for responses
        socket.on('agent_created', handleAgentCreated)
        socket.on('error', handleError)

        // Send agent creation request via Socket.IO
        createAgentViaSocket({
          agentType: (agentData as any).agent_type || 'basic',
          name: agentData.name || 'Unnamed Agent',
          description: agentData.description || 'No description provided',
          config: {
            model: (agentData as any).model || 'llama3.2:latest',
            temperature: (agentData as any).temperature || 0.7,
            max_tokens: (agentData as any).max_tokens || 2048,
            ...(agentData as any).config || {}
          },
          tools: (agentData as any).tools || []
        })

        // Set timeout for the request
        setTimeout(() => {
          socket.off('agent_created', handleAgentCreated)
          socket.off('error', handleError)
          reject(new Error('Agent creation timeout'))
        }, 30000) // 30 second timeout
      })
    } catch (error: any) {
      dispatch({ type: 'SET_ERROR', payload: error.message })
      throw error
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false })
    }
  }

  const updateAgent = (id: string, updates: Partial<Agent>) => {
    dispatch({ type: 'UPDATE_AGENT', payload: { id, updates } })
  }

  const deleteAgent = (id: string) => {
    dispatch({ type: 'REMOVE_AGENT', payload: id })
  }

  const selectAgent = (agent: Agent | null) => {
    dispatch({ type: 'SELECT_AGENT', payload: agent })
  }

  const chatWithAgent = async (agentId: string, message: string, options?: any) => {
    try {
      // Update agent status to executing
      updateAgent(agentId, { status: 'executing' })
      
      const response = await agentApi.chatWithAgent({
        message,
        agent_type: options?.agent_type || 'general',
        model: options?.model || 'llama3.2:latest',
        conversation_id: options?.conversation_id,
        ...options,
      })

      // Update agent status back to active
      updateAgent(agentId, { 
        status: 'active',
        last_activity: new Date().toISOString()
      })

      return response
    } catch (error: any) {
      // Update agent status to error
      updateAgent(agentId, { status: 'error' })
      throw error
    }
  }

  const getAgentHistory = async (agentId: string) => {
    return await agentApi.getAgentHistory(agentId)
  }

  const getAgentMetrics = async (agentId: string) => {
    return await agentApi.getAgentMetrics(agentId)
  }

  const refreshAgents = () => {
    refetchAgents()
  }

  const value: AgentContextType = {
    ...state,
    createAgent,
    updateAgent,
    deleteAgent,
    selectAgent,
    chatWithAgent,
    getAgentHistory,
    getAgentMetrics,
    refreshAgents,
  }

  return (
    <AgentContext.Provider value={value}>
      {children}
    </AgentContext.Provider>
  )
}

export function useAgent() {
  const context = useContext(AgentContext)
  if (context === undefined) {
    throw new Error('useAgent must be used within an AgentProvider')
  }
  return context
}
