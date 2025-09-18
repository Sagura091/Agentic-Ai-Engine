import React from 'react'
import { Brain, Zap, MessageSquare, ArrowRight } from 'lucide-react'
import AgentLifecycleFlow from '../components/AgentLifecycleFlow'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

const ConversationalAgents: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: 'Conversational Agent Creation',
      description: 'Design your AI agents through natural conversation with our LLM',
      benefits: [
        'No technical knowledge required',
        'Natural language interface',
        'Intelligent suggestions',
        'Iterative refinement'
      ]
    },
    {
      icon: Zap,
      title: 'Autonomous Task Execution',
      description: 'Watch your agents work independently with full reasoning visibility',
      benefits: [
        'Real-time reasoning display',
        'Step-by-step process tracking',
        'Autonomous problem solving',
        'Performance monitoring'
      ]
    },
    {
      icon: MessageSquare,
      title: 'Direct Agent Interaction',
      description: 'Chat directly with your created agents for ongoing collaboration',
      benefits: [
        'Persistent conversations',
        'Context awareness',
        'Multi-turn dialogue',
        'Personalized responses'
      ]
    }
  ]

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-foreground">
          Conversational AI Agents
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Create, test, and interact with AI agents through natural conversation. 
          Experience true agentic AI with visible reasoning and autonomous task execution.
        </p>
        <div className="flex justify-center space-x-2">
          <Badge variant="secondary">No Code Required</Badge>
          <Badge variant="secondary">Real-time Reasoning</Badge>
          <Badge variant="secondary">Autonomous Execution</Badge>
          <Badge variant="secondary">Direct Chat</Badge>
        </div>
      </div>

      {/* Features Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {features.map((feature, index) => {
          const Icon = feature.icon
          return (
            <Card key={feature.title} className="relative">
              <CardHeader>
                <CardTitle className="flex items-center space-x-3">
                  <div className="p-2 bg-primary/10 rounded-lg">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">{feature.title}</h3>
                    <div className="flex items-center space-x-2 mt-1">
                      <Badge variant="outline" className="text-xs">
                        Step {index + 1}
                      </Badge>
                    </div>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  {feature.description}
                </p>
                <ul className="space-y-2">
                  {feature.benefits.map((benefit) => (
                    <li key={benefit} className="flex items-center space-x-2 text-sm">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full" />
                      <span>{benefit}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
              
              {/* Arrow for flow indication */}
              {index < features.length - 1 && (
                <div className="hidden md:block absolute -right-3 top-1/2 transform -translate-y-1/2 z-10">
                  <div className="bg-background border border-border rounded-full p-2">
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </div>
              )}
            </Card>
          )
        })}
      </div>

      {/* How It Works */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20">
        <CardHeader>
          <CardTitle className="text-center">How It Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div className="space-y-3">
              <div className="w-12 h-12 bg-blue-500 text-white rounded-full flex items-center justify-center mx-auto">
                <Brain className="h-6 w-6" />
              </div>
              <h4 className="font-semibold">1. Describe Your Needs</h4>
              <p className="text-sm text-muted-foreground">
                Tell our AI what kind of agent you want to create. Use natural language to describe its purpose, capabilities, and behavior.
              </p>
            </div>
            
            <div className="space-y-3">
              <div className="w-12 h-12 bg-purple-500 text-white rounded-full flex items-center justify-center mx-auto">
                <Zap className="h-6 w-6" />
              </div>
              <h4 className="font-semibold">2. Test Autonomous Tasks</h4>
              <p className="text-sm text-muted-foreground">
                Give your agent tasks to complete autonomously. Watch its reasoning process in real-time as it works through problems.
              </p>
            </div>
            
            <div className="space-y-3">
              <div className="w-12 h-12 bg-green-500 text-white rounded-full flex items-center justify-center mx-auto">
                <MessageSquare className="h-6 w-6" />
              </div>
              <h4 className="font-semibold">3. Chat & Collaborate</h4>
              <p className="text-sm text-muted-foreground">
                Interact directly with your agent through chat. Build ongoing relationships and leverage its capabilities for your work.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Agent Lifecycle Flow */}
      <AgentLifecycleFlow />

      {/* Benefits Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-center">Why Choose Conversational Agent Creation?</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-semibold text-lg">For Everyone</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>No programming or technical skills required</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>Natural language interface - just describe what you want</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>Instant feedback and iterative improvement</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>Visual reasoning makes AI behavior transparent</span>
                </li>
              </ul>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-semibold text-lg">Advanced Capabilities</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>True autonomous reasoning and problem solving</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>Real-time visibility into agent thought processes</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>Persistent agent memory and learning</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>Multi-provider LLM support for optimal performance</span>
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Call to Action */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="pt-6">
          <div className="text-center space-y-4">
            <h3 className="text-2xl font-bold text-foreground">
              Ready to Create Your First Conversational Agent?
            </h3>
            <p className="text-muted-foreground">
              Start with a simple conversation and watch as your AI agent comes to life
            </p>
            <div className="text-sm text-muted-foreground">
              ✨ The agent lifecycle flow above is fully interactive - click "Create New Agent" to begin!
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ConversationalAgents
