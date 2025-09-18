import React from 'react'
import { Link } from 'react-router-dom'
import { 
  Bot, 
  MessageSquare, 
  Settings, 
  Play, 
  Pause, 
  MoreVertical,
  Clock,
  Activity
} from 'lucide-react'
import { Agent } from '../../contexts/AgentContext'
import { formatDistanceToNow } from 'date-fns'

interface AgentCardProps {
  agent: Agent
}

const AgentCard: React.FC<AgentCardProps> = ({ agent }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
      case 'executing':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
      case 'idle':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
      case 'error':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
      case 'executing':
        return <Activity className="h-3 w-3 text-yellow-600 animate-spin" />
      case 'idle':
        return <div className="h-2 w-2 bg-gray-400 rounded-full" />
      case 'error':
        return <div className="h-2 w-2 bg-red-500 rounded-full" />
      default:
        return <div className="h-2 w-2 bg-gray-400 rounded-full" />
    }
  }

  return (
    <div className="group card p-6 hover:shadow-lg hover:border-primary/50 transition-all duration-200">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <Bot className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
              {agent.name}
            </h3>
            <p className="text-sm text-muted-foreground">{agent.agent_type}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
            {getStatusIcon(agent.status)}
            <span className="capitalize">{agent.status}</span>
          </div>
          
          <button className="p-1 rounded-md hover:bg-accent opacity-0 group-hover:opacity-100 transition-opacity">
            <MoreVertical className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>
      </div>

      {/* Description */}
      <p className="text-muted-foreground text-sm mb-4 line-clamp-2">
        {agent.description}
      </p>

      {/* Model and Capabilities */}
      <div className="space-y-3 mb-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Model:</span>
          <span className="font-medium text-foreground">{agent.model}</span>
        </div>
        
        {agent.capabilities && agent.capabilities.length > 0 && (
          <div>
            <span className="text-sm text-muted-foreground mb-2 block">Capabilities:</span>
            <div className="flex flex-wrap gap-1">
              {agent.capabilities.slice(0, 3).map((capability) => (
                <span
                  key={capability}
                  className="px-2 py-1 bg-accent text-accent-foreground rounded text-xs"
                >
                  {capability}
                </span>
              ))}
              {agent.capabilities.length > 3 && (
                <span className="px-2 py-1 bg-muted text-muted-foreground rounded text-xs">
                  +{agent.capabilities.length - 3} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Last Activity */}
      {agent.last_activity && (
        <div className="flex items-center space-x-2 text-xs text-muted-foreground mb-4">
          <Clock className="h-3 w-3" />
          <span>
            Last active {formatDistanceToNow(new Date(agent.last_activity), { addSuffix: true })}
          </span>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center space-x-2">
        <Link
          to={`/chat/${agent.id}`}
          className="flex-1 flex items-center justify-center space-x-2 py-2 px-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors text-sm"
        >
          <MessageSquare className="h-4 w-4" />
          <span>Chat</span>
        </Link>
        
        <button
          className="flex items-center justify-center p-2 border border-border rounded-md hover:bg-accent transition-colors"
          title={agent.status === 'active' ? 'Pause Agent' : 'Start Agent'}
        >
          {agent.status === 'active' ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </button>
        
        <button
          className="flex items-center justify-center p-2 border border-border rounded-md hover:bg-accent transition-colors"
          title="Agent Settings"
        >
          <Settings className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}

export default AgentCard
