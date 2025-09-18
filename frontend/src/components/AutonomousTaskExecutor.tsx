import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import {
  Play,
  Brain,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  Eye,
  Settings,
  Activity,
  Pause,
  Square,
  BarChart3,
  Target,
  Lightbulb,
  Cpu,
  Database,
  Globe,
  MessageSquare,
  TrendingUp,
  AlertTriangle,
  Loader2
} from 'lucide-react';
import { useQuery } from 'react-query';
import { enhancedOrchestrationApi, autonomousAgentsApi } from '../services/api';
import toast from 'react-hot-toast';

interface ReasoningStep {
  step_number: number;
  step_type: 'analysis' | 'planning' | 'execution' | 'validation' | 'learning';
  description: string;
  details?: Record<string, any>;
  timestamp: string;
  duration_ms?: number;
  success?: boolean;
  tools_used?: string[];
  knowledge_accessed?: string[];
}

interface TaskExecution {
  execution_id: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  reasoning_steps: ReasoningStep[];
  current_step?: string;
  result?: Record<string, any>;
  error?: string;
  start_time?: string;
  end_time?: string;
  total_duration_ms?: number;
  performance_metrics?: {
    reasoning_time_ms: number;
    tool_execution_time_ms: number;
    knowledge_retrieval_time_ms: number;
    total_tokens_used: number;
    success_rate: number;
  };
}

interface AgentMetrics {
  total_executions: number;
  successful_executions: number;
  average_execution_time_ms: number;
  tools_usage_count: Record<string, number>;
  knowledge_bases_accessed: string[];
  learning_progress: number;
  autonomy_score: number;
}

interface AutonomousTaskExecutorProps {
  agentId?: string;
  onExecutionComplete?: (result: any) => void;
}

export const AutonomousTaskExecutor: React.FC<AutonomousTaskExecutorProps> = ({
  agentId = 'default_agent',
  onExecutionComplete
}) => {
  const [taskDescription, setTaskDescription] = useState('');
  const [execution, setExecution] = useState<TaskExecution | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  // Enhanced monitoring states
  const [activeView, setActiveView] = useState<'execution' | 'reasoning' | 'metrics' | 'history'>('execution');
  const [executionHistory, setExecutionHistory] = useState<TaskExecution[]>([]);
  const [realTimeMetrics, setRealTimeMetrics] = useState<AgentMetrics | null>(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  // Execution configuration
  const [executionConfig, setExecutionConfig] = useState({
    show_reasoning: true,
    enable_learning: true,
    autonomy_level: 'adaptive' as 'basic' | 'adaptive' | 'advanced',
    max_execution_time_ms: 300000, // 5 minutes
    enable_proactive_behavior: false
  });
  const [showReasoning, setShowReasoning] = useState(true);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const reasoningEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    reasoningEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [execution?.reasoning_steps]);

  // Fetch agent metrics
  const { data: agentMetrics } = useQuery(
    ['agent-metrics', agentId],
    () => autonomousAgentsApi.getLearningStats(agentId),
    {
      refetchInterval: 5000, // Update every 5 seconds
      enabled: !!agentId
    }
  );

  // Enhanced task execution with real-time monitoring
  const executeTask = async () => {
    if (!taskDescription.trim()) {
      toast.error('Please enter a task description');
      return;
    }

    setIsExecuting(true);
    setIsPaused(false);
    setExecution(null);

    try {
      // Create enhanced execution request
      const executionRequest = {
        agent_id: agentId,
        task: taskDescription,
        context: {
          execution_config: executionConfig,
          enable_real_time_monitoring: true,
          session_id: `autonomous_${Date.now()}`
        }
      };

      const response = await enhancedOrchestrationApi.executeAgentTask(executionRequest);
      const data: TaskExecution = response.data;

      setExecution(data);

      // Add to execution history
      setExecutionHistory(prev => [data, ...prev.slice(0, 9)]); // Keep last 10 executions

      // Set up real-time monitoring
      if (data.execution_id) {
        setupRealTimeMonitoring(data.execution_id);
      }

      if (data.status === 'completed' && onExecutionComplete) {
        onExecutionComplete(data.result);
      }

    } catch (error) {
      console.error('Error executing task:', error);
      setExecution({
        execution_id: 'error',
        status: 'failed',
        reasoning_steps: [],
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    } finally {
      setIsExecuting(false);
    }
  };

  // Enhanced real-time monitoring setup
  const setupRealTimeMonitoring = (executionId: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/autonomous/monitoring/${executionId}`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Real-time monitoring connected');
      setWsConnection(ws);
      toast.success('Real-time monitoring active');
    };

    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);

      if (update.type === 'execution_complete') {
        ws.close();
        setWsConnection(null);
        setIsExecuting(false);
        toast.success('Task execution completed');
        return;
      }

      if (update.type === 'reasoning_step') {
        setExecution(prev => {
          if (!prev) return prev;
          return {
            ...prev,
            reasoning_steps: [...prev.reasoning_steps, update.step],
            current_step: update.step.description
          };
        });
      }

      if (update.type === 'metrics_update') {
        setRealTimeMetrics(update.metrics);
      }

      if (update.type === 'status_change') {
        setExecution(prev => {
          if (!prev) return prev;
          return {
            ...prev,
            status: update.status,
            error: update.error
          };
        });
      }

      // Add real-time reasoning step
      setExecution(prev => {
        if (!prev) return prev;
        
        const newStep: ReasoningStep = {
          step_number: update.step_number,
          step_type: update.step_type,
          description: update.description,
          timestamp: update.timestamp
        };

        return {
          ...prev,
          reasoning_steps: [...prev.reasoning_steps, newStep],
          current_step: update.description
        };
      });
    };

    ws.onclose = () => {
      console.log('Real-time monitoring disconnected');
      setWsConnection(null);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('Real-time monitoring connection failed');
      setWsConnection(null);
    };
  };

  // Enhanced execution controls
  const pauseExecution = async () => {
    if (!execution?.execution_id) return;

    try {
      // For now, simulate pause - in production this would call the backend
      setIsPaused(true);
      toast.success('Execution paused');
    } catch (error) {
      toast.error('Failed to pause execution');
    }
  };

  const resumeExecution = async () => {
    if (!execution?.execution_id) return;

    try {
      // For now, simulate resume - in production this would call the backend
      setIsPaused(false);
      toast.success('Execution resumed');
    } catch (error) {
      toast.error('Failed to resume execution');
    }
  };

  const stopExecution = async () => {
    if (!execution?.execution_id) return;

    try {
      setIsExecuting(false);
      setIsPaused(false);
      if (wsConnection) {
        wsConnection.close();
        setWsConnection(null);
      }
      setExecution(prev => prev ? { ...prev, status: 'cancelled' } : null);
      toast.success('Execution stopped');
    } catch (error) {
      toast.error('Failed to stop execution');
    }
  };

  const getStatusIcon = () => {
    if (!execution) return <Clock className="h-4 w-4" />;
    
    switch (execution.status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'running':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusLabel = () => {
    if (!execution) return 'Ready';
    
    switch (execution.status) {
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      case 'running':
        return 'Running';
      case 'pending':
        return 'Pending';
      default:
        return 'Unknown';
    }
  };

  const getStepTypeIcon = (stepType: string) => {
    switch (stepType) {
      case 'task_analysis':
        return <Eye className="h-4 w-4 text-blue-500" />;
      case 'planning':
        return <Settings className="h-4 w-4 text-purple-500" />;
      case 'execution':
        return <Zap className="h-4 w-4 text-orange-500" />;
      case 'completion':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Brain className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStepTypeLabel = (stepType: string) => {
    switch (stepType) {
      case 'task_analysis':
        return 'Analysis';
      case 'planning':
        return 'Planning';
      case 'execution':
        return 'Execution';
      case 'completion':
        return 'Completion';
      case 'error':
        return 'Error';
      default:
        return 'Reasoning';
    }
  };

  const progress = execution ? 
    (execution.reasoning_steps.length / Math.max(4, execution.reasoning_steps.length)) * 100 : 0;

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Enhanced Header with Real-time Status */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-6 w-6 text-purple-500" />
              <div>
                <CardTitle className="text-xl">Autonomous Task Execution Dashboard</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Real-time monitoring and control for autonomous agent execution
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center gap-1">
                {getStatusIcon()}
                {getStatusLabel()}
              </Badge>
              {wsConnection && (
                <Badge variant="outline" className="flex items-center gap-1 text-green-600">
                  <Activity className="h-3 w-3" />
                  Live
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Enhanced Navigation Tabs */}
      <div className="flex space-x-1 bg-muted p-1 rounded-lg w-fit">
        {[
          { id: 'execution', label: 'Execution', icon: Play },
          { id: 'reasoning', label: 'Reasoning', icon: Brain },
          { id: 'metrics', label: 'Metrics', icon: BarChart3 },
          { id: 'history', label: 'History', icon: Clock }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveView(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
              activeView === tab.id
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <tab.icon className="h-4 w-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Execution Tab */}
      {activeView === 'execution' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Task Input and Controls */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Target className="h-5 w-5" />
                  <span>Task Configuration</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Task Description
                  </label>
                  <Textarea
                    value={taskDescription}
                    onChange={(e) => setTaskDescription(e.target.value)}
                    placeholder="Describe the task you want the agent to execute autonomously..."
                    rows={3}
                    disabled={isExecuting}
                    className="min-h-20 resize-none"
                  />
                </div>

                {/* Advanced Configuration */}
                <div className="space-y-3">
                  <button
                    type="button"
                    onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                    className="flex items-center space-x-2 text-sm text-muted-foreground hover:text-foreground"
                  >
                    <Settings className="h-4 w-4" />
                    <span>Advanced Options</span>
                  </button>

                  {showAdvancedOptions && (
                    <div className="space-y-3 p-4 bg-muted/50 rounded-lg">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-sm font-medium text-foreground mb-1">
                            Autonomy Level
                          </label>
                          <select
                            value={executionConfig.autonomy_level}
                            onChange={(e) => setExecutionConfig(prev => ({
                              ...prev,
                              autonomy_level: e.target.value as any
                            }))}
                            className="input text-sm"
                            disabled={isExecuting}
                          >
                            <option value="basic">Basic</option>
                            <option value="adaptive">Adaptive</option>
                            <option value="advanced">Advanced</option>
                          </select>
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-foreground mb-1">
                            Max Duration (minutes)
                          </label>
                          <input
                            type="number"
                            value={executionConfig.max_execution_time_ms / 60000}
                            onChange={(e) => setExecutionConfig(prev => ({
                              ...prev,
                              max_execution_time_ms: parseInt(e.target.value) * 60000
                            }))}
                            className="input text-sm"
                            min="1"
                            max="60"
                            disabled={isExecuting}
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id="show-reasoning"
                            checked={executionConfig.show_reasoning}
                            onChange={(e) => setExecutionConfig(prev => ({
                              ...prev,
                              show_reasoning: e.target.checked
                            }))}
                            className="rounded border-border"
                            disabled={isExecuting}
                          />
                          <label htmlFor="show-reasoning" className="text-sm text-foreground">
                            Show real-time reasoning
                          </label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id="enable-learning"
                            checked={executionConfig.enable_learning}
                            onChange={(e) => setExecutionConfig(prev => ({
                              ...prev,
                              enable_learning: e.target.checked
                            }))}
                            className="rounded border-border"
                            disabled={isExecuting}
                          />
                          <label htmlFor="enable-learning" className="text-sm text-foreground">
                            Enable learning from execution
                          </label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id="proactive-behavior"
                            checked={executionConfig.enable_proactive_behavior}
                            onChange={(e) => setExecutionConfig(prev => ({
                              ...prev,
                              enable_proactive_behavior: e.target.checked
                            }))}
                            className="rounded border-border"
                            disabled={isExecuting}
                          />
                          <label htmlFor="proactive-behavior" className="text-sm text-foreground">
                            Enable proactive behavior
                          </label>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Execution Controls */}
                <div className="flex items-center justify-between pt-4 border-t border-border">
                  <div className="flex items-center space-x-2">
                    {!isExecuting ? (
                      <Button
                        onClick={executeTask}
                        disabled={!taskDescription.trim()}
                        className="flex items-center space-x-2"
                      >
                        <Play className="h-4 w-4" />
                        <span>Execute Task</span>
                      </Button>
                    ) : (
                      <div className="flex items-center space-x-2">
                        {!isPaused ? (
                          <Button
                            onClick={pauseExecution}
                            variant="outline"
                            className="flex items-center space-x-2"
                          >
                            <Pause className="h-4 w-4" />
                            <span>Pause</span>
                          </Button>
                        ) : (
                          <Button
                            onClick={resumeExecution}
                            className="flex items-center space-x-2"
                          >
                            <Play className="h-4 w-4" />
                            <span>Resume</span>
                          </Button>
                        )}
                        <Button
                          onClick={stopExecution}
                          variant="destructive"
                          className="flex items-center space-x-2"
                        >
                          <Square className="h-4 w-4" />
                          <span>Stop</span>
                        </Button>
                      </div>
                    )}
                  </div>

                  {execution && (
                    <div className="text-sm text-muted-foreground">
                      {execution.reasoning_steps.length} reasoning steps
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Real-time Status Panel */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Activity className="h-5 w-5" />
                  <span>Live Status</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {execution ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Status:</span>
                      <Badge variant="outline" className="flex items-center gap-1">
                        {getStatusIcon()}
                        {getStatusLabel()}
                      </Badge>
                    </div>

                    {execution.current_step && (
                      <div>
                        <span className="text-sm text-muted-foreground">Current Step:</span>
                        <p className="text-sm font-medium mt-1">{execution.current_step}</p>
                      </div>
                    )}

                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-muted-foreground">Progress:</span>
                        <span className="text-sm font-medium">{Math.round(progress)}%</span>
                      </div>
                      <Progress value={progress} className="h-2" />
                    </div>

                    {execution.start_time && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Duration:</span>
                        <span className="text-sm font-medium">
                          {execution.end_time
                            ? `${Math.round((new Date(execution.end_time).getTime() - new Date(execution.start_time).getTime()) / 1000)}s`
                            : `${Math.round((Date.now() - new Date(execution.start_time).getTime()) / 1000)}s`
                          }
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-sm text-muted-foreground">
                      No active execution
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Metrics */}
            {realTimeMetrics && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5" />
                    <span>Quick Metrics</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-muted-foreground">Success Rate:</span>
                      <p className="font-medium">{Math.round(realTimeMetrics.success_rate * 100)}%</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Autonomy Score:</span>
                      <p className="font-medium">{Math.round(realTimeMetrics.autonomy_score * 100)}%</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Total Executions:</span>
                      <p className="font-medium">{realTimeMetrics.total_executions}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Learning Progress:</span>
                      <p className="font-medium">{Math.round(realTimeMetrics.learning_progress * 100)}%</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* Execution Progress */}
      {execution && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Execution Progress
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
            {execution.current_step && (
              <div className="text-sm text-gray-600">
                Current: {execution.current_step}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Reasoning Process */}
      {execution && showReasoning && (
        <Card className="flex-1">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Reasoning Process
              {wsConnection && (
                <Badge variant="outline" className="text-xs">
                  Live Updates
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1">
            <ScrollArea className="h-96">
              <div className="space-y-4">
                {execution.reasoning_steps.map((step, index) => (
                  <div key={index} className="flex gap-3 p-3 bg-gray-50 rounded-lg">
                    <div className="flex-shrink-0 mt-1">
                      {getStepTypeIcon(step.step_type)}
                    </div>
                    <div className="flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="text-xs">
                          Step {step.step_number}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {getStepTypeLabel(step.step_type)}
                        </Badge>
                        <span className="text-xs text-gray-500 ml-auto">
                          {new Date(step.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-800">{step.description}</p>
                      {step.details && (
                        <details className="text-xs text-gray-600">
                          <summary className="cursor-pointer hover:text-gray-800">
                            View details
                          </summary>
                          <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-x-auto">
                            {JSON.stringify(step.details, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                ))}
                {isExecuting && execution.reasoning_steps.length === 0 && (
                  <div className="flex items-center justify-center p-8 text-gray-500">
                    <div className="text-center space-y-2">
                      <Activity className="h-8 w-8 mx-auto animate-pulse" />
                      <p>Agent is thinking...</p>
                    </div>
                  </div>
                )}
              </div>
              <div ref={reasoningEndRef} />
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {execution && execution.status === 'completed' && execution.result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Task Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="p-4 bg-green-50 rounded-lg">
              <pre className="text-sm text-green-800 whitespace-pre-wrap">
                {typeof execution.result === 'string' 
                  ? execution.result 
                  : JSON.stringify(execution.result, null, 2)
                }
              </pre>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error */}
      {execution && execution.status === 'failed' && execution.error && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <XCircle className="h-5 w-5 text-red-500" />
              Execution Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="p-4 bg-red-50 rounded-lg">
              <p className="text-sm text-red-800">{execution.error}</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AutonomousTaskExecutor;
