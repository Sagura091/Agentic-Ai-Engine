import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { 
  Activity, 
  Bot, 
  GitBranch, 
  TrendingUp, 
  Clock, 
  AlertTriangle,
  CheckCircle,
  RefreshCw
} from 'lucide-react'
import { healthApi, monitoringApi } from '../services/api'
import { useSocket } from '../contexts/SocketContext'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

const Monitoring: React.FC = () => {
  const [timeframe, setTimeframe] = useState('24h')
  const { isConnected, joinAgentMonitoring, joinWorkflowMonitoring } = useSocket()

  // Join monitoring rooms on mount
  React.useEffect(() => {
    joinAgentMonitoring()
    joinWorkflowMonitoring()
  }, [joinAgentMonitoring, joinWorkflowMonitoring])

  // Fetch system health
  const { data: healthData, refetch: refetchHealth } = useQuery(
    'system-health',
    healthApi.getHealth,
    { refetchInterval: 30000 }
  )

  // Fetch agent activity
  const { data: agentActivity } = useQuery(
    ['agent-activity', timeframe],
    () => monitoringApi.getAgentActivity({ timeframe }),
    { refetchInterval: 60000 }
  )

  // Fetch workflow activity
  const { data: workflowActivity } = useQuery(
    ['workflow-activity', timeframe],
    () => monitoringApi.getWorkflowActivity({ timeframe }),
    { refetchInterval: 60000 }
  )

  // Fetch system metrics
  const { data: systemMetrics } = useQuery(
    ['system-metrics', timeframe],
    () => monitoringApi.getSystemMetrics({ timeframe }),
    { refetchInterval: 30000 }
  )

  // Mock data for charts
  const activityData = [
    { time: '00:00', agents: 12, workflows: 3 },
    { time: '04:00', agents: 8, workflows: 1 },
    { time: '08:00', agents: 25, workflows: 7 },
    { time: '12:00', agents: 35, workflows: 12 },
    { time: '16:00', agents: 28, workflows: 8 },
    { time: '20:00', agents: 18, workflows: 4 },
  ]

  const performanceData = [
    { metric: 'Response Time', value: 1.2, unit: 's', status: 'good' },
    { metric: 'Success Rate', value: 96.5, unit: '%', status: 'good' },
    { metric: 'Memory Usage', value: 68, unit: '%', status: 'warning' },
    { metric: 'CPU Usage', value: 45, unit: '%', status: 'good' },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'good':
        return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400'
      case 'warning':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400'
      case 'error':
      case 'unhealthy':
        return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400'
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'good':
        return <CheckCircle className="h-4 w-4" />
      case 'warning':
        return <AlertTriangle className="h-4 w-4" />
      case 'error':
      case 'unhealthy':
        return <AlertTriangle className="h-4 w-4" />
      default:
        return <Activity className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">System Monitoring</h1>
          <p className="text-muted-foreground mt-1">
            Real-time monitoring of agents, workflows, and system performance
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="input w-auto"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          
          <button
            onClick={() => refetchHealth()}
            className="btn-secondary inline-flex items-center"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">System Health</p>
              <p className="text-2xl font-semibold text-foreground">
                {healthData?.status === 'healthy' ? 'Healthy' : 'Issues'}
              </p>
            </div>
            <div className={`p-3 rounded-lg ${
              healthData?.status === 'healthy' 
                ? 'bg-green-100 dark:bg-green-900/20' 
                : 'bg-red-100 dark:bg-red-900/20'
            }`}>
              {getStatusIcon(healthData?.status || 'unknown')}
            </div>
          </div>
          <div className={`flex items-center space-x-1 mt-2 px-2 py-1 rounded-full text-xs font-medium w-fit ${
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

        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Active Agents</p>
              <p className="text-2xl font-semibold text-foreground">
                {agentActivity?.active_agents || 0}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/20">
              <Bot className="h-6 w-6 text-blue-600" />
            </div>
          </div>
          <p className="text-xs text-green-600 dark:text-green-400 mt-2">
            +2 from yesterday
          </p>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Running Workflows</p>
              <p className="text-2xl font-semibold text-foreground">
                {workflowActivity?.running_workflows || 0}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-purple-100 dark:bg-purple-900/20">
              <GitBranch className="h-6 w-6 text-purple-600" />
            </div>
          </div>
          <p className="text-xs text-green-600 dark:text-green-400 mt-2">
            +1 from last hour
          </p>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Avg Response Time</p>
              <p className="text-2xl font-semibold text-foreground">1.2s</p>
            </div>
            <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900/20">
              <Clock className="h-6 w-6 text-green-600" />
            </div>
          </div>
          <p className="text-xs text-green-600 dark:text-green-400 mt-2">
            -0.3s from yesterday
          </p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Activity Chart */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-xl font-semibold text-foreground">Activity Over Time</h2>
            <p className="text-muted-foreground">Agent and workflow activity trends</p>
          </div>
          <div className="card-content">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={activityData}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="agents" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={2}
                  name="Agents"
                />
                <Line 
                  type="monotone" 
                  dataKey="workflows" 
                  stroke="hsl(var(--destructive))" 
                  strokeWidth={2}
                  name="Workflows"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-xl font-semibold text-foreground">Performance Metrics</h2>
            <p className="text-muted-foreground">System performance indicators</p>
          </div>
          <div className="card-content space-y-4">
            {performanceData.map((metric) => (
              <div key={metric.metric} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getStatusColor(metric.status)}`}>
                    {getStatusIcon(metric.status)}
                  </div>
                  <span className="font-medium text-foreground">{metric.metric}</span>
                </div>
                <div className="text-right">
                  <span className="text-lg font-semibold text-foreground">
                    {metric.value}{metric.unit}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Service Status */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-xl font-semibold text-foreground">Service Status</h2>
          <p className="text-muted-foreground">Status of all system components</p>
        </div>
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {healthData?.services && Object.entries(healthData.services).map(([service, status]) => (
              <div key={service} className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getStatusColor(status as string)}`}>
                    {getStatusIcon(status as string)}
                  </div>
                  <span className="font-medium text-foreground capitalize">
                    {service.replace('_', ' ')}
                  </span>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(status as string)}`}>
                  {status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Activity Log */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-xl font-semibold text-foreground">Recent Activity</h2>
          <p className="text-muted-foreground">Latest system events and activities</p>
        </div>
        <div className="card-content">
          <div className="space-y-3">
            {[
              { time: '2 minutes ago', event: 'Agent "Research Specialist" completed task', type: 'success' },
              { time: '5 minutes ago', event: 'Workflow "Data Analysis" started execution', type: 'info' },
              { time: '8 minutes ago', event: 'New agent "Code Assistant" created', type: 'info' },
              { time: '12 minutes ago', event: 'System health check completed', type: 'success' },
              { time: '15 minutes ago', event: 'Agent "General Assistant" went idle', type: 'warning' },
            ].map((activity, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-accent/50 transition-colors">
                <div className={`p-1 rounded-full ${
                  activity.type === 'success' ? 'bg-green-100 dark:bg-green-900/20' :
                  activity.type === 'warning' ? 'bg-yellow-100 dark:bg-yellow-900/20' :
                  'bg-blue-100 dark:bg-blue-900/20'
                }`}>
                  <div className={`h-2 w-2 rounded-full ${
                    activity.type === 'success' ? 'bg-green-500' :
                    activity.type === 'warning' ? 'bg-yellow-500' :
                    'bg-blue-500'
                  }`} />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-foreground">{activity.event}</p>
                  <p className="text-xs text-muted-foreground">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Monitoring
