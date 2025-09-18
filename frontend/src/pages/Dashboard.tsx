import React from 'react'
import { useQuery } from 'react-query'
import { Link } from 'react-router-dom'
import { 
  Bot, 
  GitBranch, 
  MessageSquare, 
  Activity,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertCircle,
  Plus,
  ArrowRight
} from 'lucide-react'
import { useAgent } from '../contexts/AgentContext'
import { useSocket } from '../contexts/SocketContext'
import { healthApi, monitoringApi } from '../services/api'

const Dashboard: React.FC = () => {
  const { agents, templates } = useAgent()
  const { isConnected } = useSocket()

  // Fetch system health
  const { data: healthData } = useQuery('health', healthApi.getHealth, {
    refetchInterval: 30000,
  })

  // Fetch recent activity
  const { data: agentActivity } = useQuery('agent-activity', 
    () => monitoringApi.getAgentActivity({ timeframe: '24h' }), {
    refetchInterval: 60000,
  })

  const { data: workflowActivity } = useQuery('workflow-activity',
    () => monitoringApi.getWorkflowActivity({ timeframe: '24h' }), {
    refetchInterval: 60000,
  })

  // Calculate stats
  const activeAgents = agents.filter(agent => agent.status === 'active').length
  const executingAgents = agents.filter(agent => agent.status === 'executing').length
  const totalAgents = agents.length

  const stats = [
    {
      name: 'Active Agents',
      value: activeAgents,
      total: totalAgents,
      icon: Bot,
      color: 'text-green-600',
      bgColor: 'bg-green-100 dark:bg-green-900/20',
      change: '+2 from yesterday',
      changeType: 'positive' as const,
    },
    {
      name: 'Running Workflows',
      value: executingAgents,
      total: undefined,
      icon: GitBranch,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100 dark:bg-blue-900/20',
      change: '+1 from last hour',
      changeType: 'positive' as const,
    },
    {
      name: 'Conversations Today',
      value: agentActivity?.total_conversations || 0,
      total: undefined,
      icon: MessageSquare,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100 dark:bg-purple-900/20',
      change: '+12% from yesterday',
      changeType: 'positive' as const,
    },
    {
      name: 'System Health',
      value: healthData?.status === 'healthy' ? 100 : 0,
      total: 100,
      icon: Activity,
      color: isConnected ? 'text-green-600' : 'text-red-600',
      bgColor: isConnected ? 'bg-green-100 dark:bg-green-900/20' : 'bg-red-100 dark:bg-red-900/20',
      change: isConnected ? 'All systems operational' : 'Connection issues',
      changeType: isConnected ? 'positive' as const : 'negative' as const,
    },
  ]

  const quickActions = [
    {
      name: 'Create Agent',
      description: 'Build a new AI agent with custom capabilities',
      href: '/agents',
      icon: Bot,
      color: 'bg-blue-500',
    },
    {
      name: 'Design Workflow',
      description: 'Create multi-agent workflows for complex tasks',
      href: '/workflows',
      icon: GitBranch,
      color: 'bg-purple-500',
    },
    {
      name: 'Start Chat',
      description: 'Begin a conversation with an existing agent',
      href: '/chat',
      icon: MessageSquare,
      color: 'bg-green-500',
    },
    {
      name: 'View Monitoring',
      description: 'Monitor agent performance and system health',
      href: '/monitoring',
      icon: Activity,
      color: 'bg-orange-500',
    },
  ]

  const recentAgents = agents.slice(0, 3)
  const popularTemplates = templates.slice(0, 4)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Welcome to your Agentic AI control center
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            isConnected 
              ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
              : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
          }`}>
            <div className={`h-2 w-2 rounded-full ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <div key={stat.name} className="card p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    {stat.name}
                  </p>
                  <div className="flex items-baseline space-x-2">
                    <p className="text-2xl font-semibold text-foreground">
                      {stat.value}
                    </p>
                    {stat.total && (
                      <p className="text-sm text-muted-foreground">
                        / {stat.total}
                      </p>
                    )}
                  </div>
                  <p className={`text-xs mt-1 ${
                    stat.changeType === 'positive' 
                      ? 'text-green-600 dark:text-green-400' 
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    {stat.change}
                  </p>
                </div>
                <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                  <Icon className={`h-6 w-6 ${stat.color}`} />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-xl font-semibold text-foreground">Quick Actions</h2>
          <p className="text-muted-foreground">Get started with common tasks</p>
        </div>
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {quickActions.map((action) => {
              const Icon = action.icon
              return (
                <Link
                  key={action.name}
                  to={action.href}
                  className="group p-4 rounded-lg border border-border hover:border-primary/50 hover:shadow-md transition-all duration-200"
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${action.color} text-white`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-foreground group-hover:text-primary">
                        {action.name}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {action.description}
                      </p>
                    </div>
                    <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>
                </Link>
              )
            })}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Agents */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-foreground">Recent Agents</h2>
              <Link 
                to="/agents" 
                className="text-sm text-primary hover:text-primary/80 font-medium"
              >
                View all
              </Link>
            </div>
          </div>
          <div className="card-content space-y-3">
            {recentAgents.length > 0 ? (
              recentAgents.map((agent) => (
                <div key={agent.id} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-accent/50 transition-colors">
                  <div className={`p-2 rounded-lg ${
                    agent.status === 'active' ? 'bg-green-100 dark:bg-green-900/20' :
                    agent.status === 'executing' ? 'bg-yellow-100 dark:bg-yellow-900/20' :
                    'bg-gray-100 dark:bg-gray-900/20'
                  }`}>
                    <Bot className={`h-4 w-4 ${
                      agent.status === 'active' ? 'text-green-600' :
                      agent.status === 'executing' ? 'text-yellow-600' :
                      'text-gray-600'
                    }`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-foreground truncate">{agent.name}</p>
                    <p className="text-sm text-muted-foreground truncate">{agent.description}</p>
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    agent.status === 'active' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
                    agent.status === 'executing' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' :
                    'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
                  }`}>
                    {agent.status}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <Bot className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                <p className="text-muted-foreground">No agents created yet</p>
                <Link 
                  to="/agents" 
                  className="inline-flex items-center mt-2 text-sm text-primary hover:text-primary/80"
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Create your first agent
                </Link>
              </div>
            )}
          </div>
        </div>

        {/* Popular Templates */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-foreground">Popular Templates</h2>
              <Link 
                to="/agents" 
                className="text-sm text-primary hover:text-primary/80 font-medium"
              >
                Browse all
              </Link>
            </div>
          </div>
          <div className="card-content space-y-3">
            {popularTemplates.map((template) => (
              <div key={template.id} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-accent/50 transition-colors">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-foreground truncate">{template.name}</p>
                  <p className="text-sm text-muted-foreground truncate">{template.description}</p>
                </div>
                <div className="px-2 py-1 rounded-full text-xs font-medium bg-accent text-accent-foreground">
                  {template.category}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
