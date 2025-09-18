import React, { useState, useRef, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { Send, Bot, User, Loader2, Settings, MoreVertical } from 'lucide-react'
import { useAgent } from '../contexts/AgentContext'
import { useSocket } from '../contexts/SocketContext'
import { formatDistanceToNow } from 'date-fns'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  agentId?: string
  executionTime?: number
}

const AgentChat: React.FC = () => {
  const { agentId } = useParams<{ agentId: string }>()
  const { agents, selectedAgent, selectAgent, chatWithAgent } = useAgent()
  const { isConnected } = useSocket()
  
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId] = useState(`conv_${Date.now()}`)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Select agent if agentId is provided in URL
  useEffect(() => {
    if (agentId && agents.length > 0) {
      const agent = agents.find(a => a.id === agentId)
      if (agent) {
        selectAgent(agent)
      }
    }
  }, [agentId, agents, selectAgent])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !selectedAgent || isLoading) return

    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const response = await chatWithAgent(selectedAgent.id, userMessage.content, {
        conversation_id: conversationId,
        agent_type: selectedAgent.agent_type,
        model: selectedAgent.model
      })

      const assistantMessage: Message = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        agentId: selectedAgent.id,
        executionTime: response.execution_time
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error: any) {
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        role: 'system',
        content: `Error: ${error.message}`,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border bg-card">
        <div className="flex items-center space-x-3">
          {selectedAgent ? (
            <>
              <div className="p-2 rounded-lg bg-primary/10">
                <Bot className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h2 className="font-semibold text-foreground">{selectedAgent.name}</h2>
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <span>{selectedAgent.agent_type}</span>
                  <span>•</span>
                  <span>{selectedAgent.model}</span>
                  <span>•</span>
                  <div className={`flex items-center space-x-1 ${
                    selectedAgent.status === 'active' ? 'text-green-600' :
                    selectedAgent.status === 'executing' ? 'text-yellow-600' :
                    'text-gray-600'
                  }`}>
                    <div className={`h-2 w-2 rounded-full ${
                      selectedAgent.status === 'active' ? 'bg-green-500 animate-pulse' :
                      selectedAgent.status === 'executing' ? 'bg-yellow-500 animate-pulse' :
                      'bg-gray-400'
                    }`} />
                    <span className="capitalize">{selectedAgent.status}</span>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div>
              <h2 className="font-semibold text-foreground">Agent Chat</h2>
              <p className="text-sm text-muted-foreground">Select an agent to start chatting</p>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <div className={`px-2 py-1 rounded-full text-xs font-medium ${
            isConnected 
              ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
              : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
          }`}>
            {isConnected ? 'Connected' : 'Offline'}
          </div>
          
          <button className="p-2 rounded-md hover:bg-accent transition-colors">
            <Settings className="h-4 w-4" />
          </button>
          
          <button className="p-2 rounded-md hover:bg-accent transition-colors">
            <MoreVertical className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Agent Selection */}
      {!selectedAgent && (
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-md">
            <Bot className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-foreground mb-2">
              Choose an Agent
            </h3>
            <p className="text-muted-foreground mb-6">
              Select an agent from the list below to start a conversation
            </p>
            
            <div className="space-y-2">
              {agents.length > 0 ? (
                agents.slice(0, 3).map((agent) => (
                  <button
                    key={agent.id}
                    onClick={() => selectAgent(agent)}
                    className="w-full p-3 text-left border border-border rounded-lg hover:border-primary/50 hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Bot className="h-4 w-4 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium text-foreground">{agent.name}</p>
                        <p className="text-sm text-muted-foreground">{agent.description}</p>
                      </div>
                    </div>
                  </button>
                ))
              ) : (
                <p className="text-muted-foreground">No agents available</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      {selectedAgent && (
        <>
          <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {messages.length === 0 && (
              <div className="text-center py-8">
                <Bot className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium text-foreground mb-2">
                  Start a conversation
                </h3>
                <p className="text-muted-foreground">
                  Send a message to {selectedAgent.name} to begin
                </p>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[70%] rounded-lg p-3 ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : message.role === 'system'
                      ? 'bg-destructive/10 text-destructive border border-destructive/20'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    {message.role !== 'user' && (
                      <div className="flex-shrink-0 mt-1">
                        {message.role === 'assistant' ? (
                          <Bot className="h-4 w-4" />
                        ) : (
                          <div className="h-4 w-4 rounded-full bg-current opacity-50" />
                        )}
                      </div>
                    )}
                    <div className="flex-1">
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                      <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                        <span>
                          {formatDistanceToNow(message.timestamp, { addSuffix: true })}
                        </span>
                        {message.executionTime && (
                          <span>{message.executionTime.toFixed(2)}s</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-muted text-muted-foreground rounded-lg p-3 max-w-[70%]">
                  <div className="flex items-center space-x-2">
                    <Bot className="h-4 w-4" />
                    <div className="flex items-center space-x-1">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-border p-4 bg-card">
            <div className="flex items-end space-x-2">
              <div className="flex-1">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={`Message ${selectedAgent.name}...`}
                  className="input"
                  disabled={isLoading || !isConnected}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading || !isConnected}
                className="btn-primary p-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </button>
            </div>
            
            {!isConnected && (
              <p className="text-xs text-destructive mt-2">
                Disconnected from server. Reconnecting...
              </p>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default AgentChat
