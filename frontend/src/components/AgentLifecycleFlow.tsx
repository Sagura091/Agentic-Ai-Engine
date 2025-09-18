import React, { useState } from 'react'
import { 
  ArrowRight, 
  Brain, 
  Zap, 
  MessageSquare, 
  CheckCircle,
  Play,
  Settings,
  Bot
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useNavigate } from 'react-router-dom'
import ConversationalAgentCreator from './ConversationalAgentCreator'
import AutonomousTaskExecutor from './AutonomousTaskExecutor'

interface AgentConfig {
  name: string
  description: string
  capabilities: string[]
  tools: string[]
  model: string
  temperature: number
}

interface AgentLifecycleFlowProps {
  initialStep?: 'create' | 'execute' | 'chat'
}

export const AgentLifecycleFlow: React.FC<AgentLifecycleFlowProps> = ({ 
  initialStep = 'create' 
}) => {
  const navigate = useNavigate()
  const [currentStep, setCurrentStep] = useState<'create' | 'execute' | 'chat'>(initialStep)
  const [createdAgent, setCreatedAgent] = useState<AgentConfig | null>(null)
  const [agentId, setAgentId] = useState<string | null>(null)

  const steps = [
    {
      id: 'create',
      title: 'Create Agent',
      description: 'Design your agent through conversation',
      icon: Brain,
      color: 'bg-blue-500',
      completed: !!createdAgent
    },
    {
      id: 'execute',
      title: 'Execute Tasks',
      description: 'Run autonomous tasks with reasoning',
      icon: Zap,
      color: 'bg-purple-500',
      completed: false
    },
    {
      id: 'chat',
      title: 'Chat & Interact',
      description: 'Ongoing conversation with your agent',
      icon: MessageSquare,
      color: 'bg-green-500',
      completed: false
    }
  ]

  const handleAgentCreated = async (agentConfig: AgentConfig) => {
    try {
      // Create the agent via API
      const response = await fetch('/api/v1/agents/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(agentConfig)
      })

      if (response.ok) {
        const result = await response.json()
        setCreatedAgent(agentConfig)
        setAgentId(result.agent_id)
        setCurrentStep('execute')
      } else {
        throw new Error('Failed to create agent')
      }
    } catch (error) {
      console.error('Error creating agent:', error)
    }
  }

  const handleTaskExecuted = () => {
    setCurrentStep('chat')
  }

  const handleStepClick = (stepId: 'create' | 'execute' | 'chat') => {
    if (stepId === 'create') {
      setCurrentStep('create')
    } else if (stepId === 'execute' && createdAgent) {
      setCurrentStep('execute')
    } else if (stepId === 'chat' && agentId) {
      setCurrentStep('chat')
    }
  }

  const navigateToChat = () => {
    if (agentId) {
      navigate(`/chat/${agentId}`)
    }
  }

  const renderStepContent = () => {
    switch (currentStep) {
      case 'create':
        return (
          <ConversationalAgentCreator 
            onAgentCreated={handleAgentCreated}
          />
        )
      
      case 'execute':
        return (
          <div className="space-y-6">
            {createdAgent && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Bot className="h-5 w-5" />
                    Agent Created: {createdAgent.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground mb-4">{createdAgent.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {createdAgent.capabilities.map((capability) => (
                      <Badge key={capability} variant="secondary">
                        {capability}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
            
            <AutonomousTaskExecutor 
              agentId={agentId || 'default'}
              onExecutionComplete={handleTaskExecuted}
            />
          </div>
        )
      
      case 'chat':
        return (
          <div className="space-y-6">
            {createdAgent && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    Agent Ready: {createdAgent.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground mb-4">
                    Your agent has been created and tested. You can now chat with it directly.
                  </p>
                  <Button onClick={navigateToChat} className="w-full">
                    <MessageSquare className="h-4 w-4 mr-2" />
                    Start Chatting with {createdAgent.name}
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        )
      
      default:
        return null
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Progress Steps */}
      <Card>
        <CardHeader>
          <CardTitle className="text-center">Agent Lifecycle Flow</CardTitle>
          <p className="text-center text-muted-foreground">
            Create, test, and interact with your AI agents
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = currentStep === step.id
              const isClickable = step.id === 'create' || 
                                 (step.id === 'execute' && createdAgent) ||
                                 (step.id === 'chat' && agentId)
              
              return (
                <React.Fragment key={step.id}>
                  <div 
                    className={`flex flex-col items-center space-y-2 ${
                      isClickable ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'
                    }`}
                    onClick={() => isClickable && handleStepClick(step.id as any)}
                  >
                    <div className={`
                      w-12 h-12 rounded-full flex items-center justify-center text-white
                      ${isActive ? step.color : step.completed ? 'bg-green-500' : 'bg-gray-400'}
                      ${isActive ? 'ring-4 ring-blue-200' : ''}
                      transition-all duration-200
                    `}>
                      {step.completed ? (
                        <CheckCircle className="h-6 w-6" />
                      ) : (
                        <Icon className="h-6 w-6" />
                      )}
                    </div>
                    
                    <div className="text-center">
                      <h3 className={`text-sm font-semibold ${
                        isActive ? 'text-primary' : 'text-foreground'
                      }`}>
                        {step.title}
                      </h3>
                      <p className="text-xs text-muted-foreground max-w-24">
                        {step.description}
                      </p>
                    </div>
                  </div>
                  
                  {index < steps.length - 1 && (
                    <ArrowRight className="h-6 w-6 text-muted-foreground" />
                  )}
                </React.Fragment>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-3 justify-center">
            <Button
              variant={currentStep === 'create' ? 'default' : 'outline'}
              onClick={() => setCurrentStep('create')}
            >
              <Brain className="h-4 w-4 mr-2" />
              Create New Agent
            </Button>
            
            <Button
              variant={currentStep === 'execute' ? 'default' : 'outline'}
              onClick={() => createdAgent && setCurrentStep('execute')}
              disabled={!createdAgent}
            >
              <Zap className="h-4 w-4 mr-2" />
              Execute Tasks
            </Button>
            
            <Button
              variant={currentStep === 'chat' ? 'default' : 'outline'}
              onClick={navigateToChat}
              disabled={!agentId}
            >
              <MessageSquare className="h-4 w-4 mr-2" />
              Chat with Agent
            </Button>
            
            <Button
              variant="outline"
              onClick={() => navigate('/agents')}
            >
              <Settings className="h-4 w-4 mr-2" />
              Agent Builder
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Current Step Content */}
      <div className="min-h-[600px]">
        {renderStepContent()}
      </div>

      {/* Help Text */}
      <Card className="bg-muted/50">
        <CardContent className="pt-6">
          <div className="text-center space-y-2">
            <h4 className="font-semibold text-foreground">How it works:</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-muted-foreground">
              <div className="flex items-start space-x-2">
                <Brain className="h-4 w-4 mt-0.5 text-blue-500" />
                <div>
                  <strong>Create:</strong> Talk to our AI to design your perfect agent through natural conversation
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <Zap className="h-4 w-4 mt-0.5 text-purple-500" />
                <div>
                  <strong>Execute:</strong> Give your agent autonomous tasks and watch it reason through solutions
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <MessageSquare className="h-4 w-4 mt-0.5 text-green-500" />
                <div>
                  <strong>Chat:</strong> Interact directly with your agent for ongoing assistance and collaboration
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default AgentLifecycleFlow
