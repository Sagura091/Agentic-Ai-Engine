import React, { useState, useEffect } from 'react'
import { 
  Wifi, 
  WifiOff, 
  Server, 
  Database, 
  Bot, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Activity,
  Zap,
  RefreshCw
} from 'lucide-react'
import { useSocket } from '../contexts/SocketContext'
import toast from 'react-hot-toast'

interface SystemHealth {
  backend_status: 'healthy' | 'degraded' | 'down'
  database_status: 'connected' | 'disconnected' | 'error'
  ollama_status: 'available' | 'unavailable' | 'error'
  agents_count: number
  active_conversations: number
  system_uptime: string
  last_check: string
  response_time_ms: number
}

interface SystemStatusProps {
  compact?: boolean
  showDetails?: boolean
}

export const SystemStatus: React.FC<SystemStatusProps> = ({ 
  compact = false, 
  showDetails = true 
}) => {
  const { isConnected } = useSocket()
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const checkSystemHealth = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/v1/system/health')
      if (response.ok) {
        const health = await response.json()
        setSystemHealth(health)
        setLastUpdate(new Date())
      } else {
        throw new Error(`Health check failed: ${response.status}`)
      }
    } catch (error) {
      console.error('System health check failed:', error)
      setSystemHealth({
        backend_status: 'down',
        database_status: 'error',
        ollama_status: 'error',
        agents_count: 0,
        active_conversations: 0,
        system_uptime: 'Unknown',
        last_check: new Date().toISOString(),
        response_time_ms: 0
      })
      toast.error('System health check failed')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    checkSystemHealth()
    
    if (autoRefresh) {
      const interval = setInterval(checkSystemHealth, 30000) // Check every 30 seconds
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'connected':
      case 'available':
        return 'text-green-500'
      case 'degraded':
        return 'text-yellow-500'
      case 'down':
      case 'disconnected':
      case 'unavailable':
      case 'error':
        return 'text-red-500'
      default:
        return 'text-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'connected':
      case 'available':
        return <CheckCircle className="h-4 w-4" />
      case 'degraded':
        return <AlertTriangle className="h-4 w-4" />
      case 'down':
      case 'disconnected':
      case 'unavailable':
      case 'error':
        return <AlertTriangle className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  if (compact) {
    return (
      <div className="flex items-center space-x-2">
        <div className={`flex items-center space-x-1 ${isConnected ? 'text-green-500' : 'text-red-500'}`}>
          {isConnected ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />}
          <span className="text-xs">{isConnected ? 'Connected' : 'Offline'}</span>
        </div>
        
        {systemHealth && (
          <div className={`flex items-center space-x-1 ${getStatusColor(systemHealth.backend_status)}`}>
            <Server className="h-4 w-4" />
            <span className="text-xs capitalize">{systemHealth.backend_status}</span>
          </div>
        )}
        
        <button
          onClick={checkSystemHealth}
          disabled={isLoading}
          className="p-1 hover:bg-accent rounded"
          title="Refresh system status"
        >
          <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    )
  }

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <Activity className="h-5 w-5" />
          System Status
        </h3>
        
        <div className="flex items-center space-x-2">
          <label className="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            <span>Auto-refresh</span>
          </label>
          
          <button
            onClick={checkSystemHealth}
            disabled={isLoading}
            className="flex items-center space-x-1 px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Connection Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
          <div className={isConnected ? 'text-green-500' : 'text-red-500'}>
            {isConnected ? <Wifi className="h-5 w-5" /> : <WifiOff className="h-5 w-5" />}
          </div>
          <div>
            <p className="text-sm font-medium">WebSocket</p>
            <p className="text-xs text-muted-foreground">
              {isConnected ? 'Connected' : 'Disconnected'}
            </p>
          </div>
        </div>

        {systemHealth && (
          <>
            <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
              <div className={getStatusColor(systemHealth.backend_status)}>
                <Server className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-medium">Backend</p>
                <p className="text-xs text-muted-foreground capitalize">
                  {systemHealth.backend_status}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
              <div className={getStatusColor(systemHealth.database_status)}>
                <Database className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-medium">Database</p>
                <p className="text-xs text-muted-foreground capitalize">
                  {systemHealth.database_status}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
              <div className={getStatusColor(systemHealth.ollama_status)}>
                <Bot className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-medium">Ollama</p>
                <p className="text-xs text-muted-foreground capitalize">
                  {systemHealth.ollama_status}
                </p>
              </div>
            </div>
          </>
        )}
      </div>

      {/* System Metrics */}
      {systemHealth && showDetails && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-foreground">System Metrics</h4>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-primary">{systemHealth.agents_count}</div>
              <div className="text-xs text-muted-foreground">Active Agents</div>
            </div>
            
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-primary">{systemHealth.active_conversations}</div>
              <div className="text-xs text-muted-foreground">Conversations</div>
            </div>
            
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-primary">{systemHealth.response_time_ms}ms</div>
              <div className="text-xs text-muted-foreground">Response Time</div>
            </div>
            
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-sm font-bold text-primary">{systemHealth.system_uptime}</div>
              <div className="text-xs text-muted-foreground">Uptime</div>
            </div>
          </div>
        </div>
      )}

      {/* Last Update */}
      {lastUpdate && (
        <div className="mt-4 pt-3 border-t border-border">
          <p className="text-xs text-muted-foreground">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
      )}
    </div>
  )
}

export default SystemStatus
