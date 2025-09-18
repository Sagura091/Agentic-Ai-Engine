import React, { useState, useEffect } from 'react'
import { 
  Bug, 
  X, 
  Copy, 
  Download, 
  RefreshCw, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity,
  Database,
  Wifi,
  Server
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useError } from '../contexts/ErrorContext'
import { useSocket } from '../contexts/SocketContext'
import toast from 'react-hot-toast'

interface DebugPanelProps {
  isOpen: boolean
  onClose: () => void
}

interface SystemDiagnostics {
  frontend: {
    version: string
    buildTime: string
    environment: string
    userAgent: string
    viewport: string
    memory?: any
  }
  backend: {
    status: 'connected' | 'disconnected' | 'error'
    version?: string
    uptime?: string
    health?: any
  }
  network: {
    online: boolean
    effectiveType?: string
    downlink?: number
    rtt?: number
  }
  performance: {
    loadTime: number
    renderTime: number
    memoryUsage?: any
  }
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ isOpen, onClose }) => {
  const { errors, clearAllErrors } = useError()
  const { isConnected } = useSocket()
  const [diagnostics, setDiagnostics] = useState<SystemDiagnostics | null>(null)
  const [apiLogs, setApiLogs] = useState<any[]>([])
  const [isRefreshing, setIsRefreshing] = useState(false)

  useEffect(() => {
    if (isOpen) {
      collectDiagnostics()
    }
  }, [isOpen])

  const collectDiagnostics = async () => {
    setIsRefreshing(true)
    
    try {
      // Frontend diagnostics
      const frontend = {
        version: import.meta.env.VITE_APP_VERSION || 'development',
        buildTime: import.meta.env.VITE_BUILD_TIME || 'unknown',
        environment: import.meta.env.MODE,
        userAgent: navigator.userAgent,
        viewport: `${window.innerWidth}x${window.innerHeight}`,
        memory: (performance as any).memory ? {
          used: Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024),
          total: Math.round((performance as any).memory.totalJSHeapSize / 1024 / 1024),
          limit: Math.round((performance as any).memory.jsHeapSizeLimit / 1024 / 1024)
        } : undefined
      }

      // Network diagnostics
      const connection = (navigator as any).connection
      const network = {
        online: navigator.onLine,
        effectiveType: connection?.effectiveType,
        downlink: connection?.downlink,
        rtt: connection?.rtt
      }

      // Performance diagnostics
      const performanceEntries = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
      const performanceDiag = {
        loadTime: Math.round(performanceEntries.loadEventEnd - performanceEntries.navigationStart),
        renderTime: Math.round(performanceEntries.domContentLoadedEventEnd - performanceEntries.navigationStart),
        memoryUsage: frontend.memory
      }

      // Backend diagnostics
      let backend = {
        status: 'disconnected' as const,
        version: undefined,
        uptime: undefined,
        health: undefined
      }

      try {
        const response = await fetch('/api/v1/system/health')
        if (response.ok) {
          const health = await response.json()
          backend = {
            status: 'connected',
            version: health.version,
            uptime: health.uptime,
            health
          }
        }
      } catch (error) {
        backend.status = 'error'
      }

      setDiagnostics({
        frontend,
        backend,
        network,
        performance: performanceDiag
      })
    } catch (error) {
      console.error('Failed to collect diagnostics:', error)
    } finally {
      setIsRefreshing(false)
    }
  }

  const exportDiagnostics = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      diagnostics,
      errors: errors.map(error => ({
        ...error,
        timestamp: error.timestamp.toISOString()
      })),
      apiLogs,
      userAgent: navigator.userAgent,
      url: window.location.href
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `debug-report-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    toast.success('Debug report exported')
  }

  const copyDiagnostics = () => {
    const text = JSON.stringify(diagnostics, null, 2)
    navigator.clipboard.writeText(text)
    toast.success('Diagnostics copied to clipboard')
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-background border border-border rounded-lg shadow-lg w-full max-w-6xl h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center space-x-2">
            <Bug className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Debug Panel</h2>
            <Badge variant="outline">
              {errors.length} errors
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              onClick={collectDiagnostics}
              disabled={isRefreshing}
              variant="outline"
              size="sm"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            
            <Button
              onClick={exportDiagnostics}
              variant="outline"
              size="sm"
            >
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            
            <Button
              onClick={onClose}
              variant="outline"
              size="sm"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          <Tabs defaultValue="diagnostics" className="h-full flex flex-col">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="diagnostics">System Diagnostics</TabsTrigger>
              <TabsTrigger value="errors">Errors ({errors.length})</TabsTrigger>
              <TabsTrigger value="network">Network</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
            </TabsList>
            
            <div className="flex-1 overflow-auto p-4">
              <TabsContent value="diagnostics" className="space-y-4">
                {diagnostics && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Frontend Status */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Activity className="h-4 w-4" />
                          Frontend
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Version:</span>
                          <span className="font-mono">{diagnostics.frontend.version}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Environment:</span>
                          <Badge variant="outline">{diagnostics.frontend.environment}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>Viewport:</span>
                          <span className="font-mono">{diagnostics.frontend.viewport}</span>
                        </div>
                        {diagnostics.frontend.memory && (
                          <div className="flex justify-between">
                            <span>Memory:</span>
                            <span className="font-mono">
                              {diagnostics.frontend.memory.used}MB / {diagnostics.frontend.memory.total}MB
                            </span>
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {/* Backend Status */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Server className="h-4 w-4" />
                          Backend
                          <Badge className={
                            diagnostics.backend.status === 'connected' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }>
                            {diagnostics.backend.status}
                          </Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        {diagnostics.backend.version && (
                          <div className="flex justify-between">
                            <span>Version:</span>
                            <span className="font-mono">{diagnostics.backend.version}</span>
                          </div>
                        )}
                        {diagnostics.backend.uptime && (
                          <div className="flex justify-between">
                            <span>Uptime:</span>
                            <span className="font-mono">{diagnostics.backend.uptime}</span>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <span>WebSocket:</span>
                          <Badge className={isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                            {isConnected ? 'Connected' : 'Disconnected'}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Network Status */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Wifi className="h-4 w-4" />
                          Network
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Status:</span>
                          <Badge className={diagnostics.network.online ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                            {diagnostics.network.online ? 'Online' : 'Offline'}
                          </Badge>
                        </div>
                        {diagnostics.network.effectiveType && (
                          <div className="flex justify-between">
                            <span>Connection:</span>
                            <span className="font-mono">{diagnostics.network.effectiveType}</span>
                          </div>
                        )}
                        {diagnostics.network.downlink && (
                          <div className="flex justify-between">
                            <span>Downlink:</span>
                            <span className="font-mono">{diagnostics.network.downlink} Mbps</span>
                          </div>
                        )}
                        {diagnostics.network.rtt && (
                          <div className="flex justify-between">
                            <span>RTT:</span>
                            <span className="font-mono">{diagnostics.network.rtt}ms</span>
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {/* Performance */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Clock className="h-4 w-4" />
                          Performance
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Load Time:</span>
                          <span className="font-mono">{diagnostics.performance.loadTime}ms</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Render Time:</span>
                          <span className="font-mono">{diagnostics.performance.renderTime}ms</span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="errors" className="space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold">Application Errors</h3>
                  <Button onClick={clearAllErrors} variant="outline" size="sm">
                    Clear All
                  </Button>
                </div>
                
                <div className="space-y-2">
                  {errors.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <CheckCircle className="h-8 w-8 mx-auto mb-2" />
                      No errors recorded
                    </div>
                  ) : (
                    errors.map((error) => (
                      <Card key={error.id}>
                        <CardContent className="pt-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <AlertTriangle className="h-4 w-4 text-red-500" />
                                <Badge variant="outline">{error.type}</Badge>
                                <Badge className={
                                  error.severity === 'critical' ? 'bg-red-100 text-red-800' :
                                  error.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                                  error.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-gray-100 text-gray-800'
                                }>
                                  {error.severity}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {error.timestamp.toLocaleTimeString()}
                                </span>
                              </div>
                              <p className="text-sm font-medium">{error.message}</p>
                              {error.source && (
                                <p className="text-xs text-muted-foreground mt-1">
                                  Source: {error.source}
                                </p>
                              )}
                              {error.details && (
                                <details className="mt-2">
                                  <summary className="text-xs cursor-pointer text-muted-foreground">
                                    Show details
                                  </summary>
                                  <pre className="text-xs bg-muted p-2 rounded mt-1 overflow-x-auto">
                                    {error.details}
                                  </pre>
                                </details>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </TabsContent>

              <TabsContent value="network">
                <div className="text-center py-8 text-muted-foreground">
                  Network monitoring coming soon...
                </div>
              </TabsContent>

              <TabsContent value="performance">
                <div className="text-center py-8 text-muted-foreground">
                  Performance monitoring coming soon...
                </div>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </div>
    </div>
  )
}

export default DebugPanel
