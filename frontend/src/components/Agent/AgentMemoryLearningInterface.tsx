import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Memory, 
  TrendingUp, 
  BarChart3, 
  Clock, 
  Target,
  Lightbulb,
  Activity,
  Eye,
  Settings,
  Download,
  Upload,
  RefreshCw,
  Filter,
  Search,
  Plus,
  Trash2,
  Edit,
  Copy,
  Share2,
  Lock,
  Unlock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Zap,
  Network,
  Database,
  FileText,
  MessageSquare,
  Users,
  Globe,
  Code,
  Gauge,
  Calendar,
  Star,
  Award,
  Cpu
} from 'lucide-react';
import { useQuery, useMutation } from 'react-query';
import { agentMemoryApi, learningApi, autonomousAgentsApi } from '../../services/api';
import toast from 'react-hot-toast';

interface AgentMemory {
  id: string;
  agent_id: string;
  memory_type: 'episodic' | 'semantic' | 'procedural' | 'working' | 'meta';
  content: string;
  context: Record<string, any>;
  importance_score: number;
  access_count: number;
  created_at: string;
  last_accessed: string;
  tags: string[];
  related_memories: string[];
  emotional_valence: number; // -1 to 1
  confidence_score: number;
}

interface LearningProgress {
  agent_id: string;
  skill_areas: Array<{
    name: string;
    current_level: number;
    target_level: number;
    progress_rate: number;
    last_improvement: string;
    milestones: Array<{
      name: string;
      achieved: boolean;
      date?: string;
    }>;
  }>;
  learning_metrics: {
    total_learning_sessions: number;
    successful_adaptations: number;
    knowledge_retention_rate: number;
    transfer_learning_score: number;
    meta_learning_ability: number;
  };
  behavioral_patterns: Array<{
    pattern_name: string;
    frequency: number;
    effectiveness: number;
    context: string[];
    trend: 'improving' | 'stable' | 'declining';
  }>;
}

interface AutonomousBehavior {
  id: string;
  agent_id: string;
  behavior_type: 'proactive' | 'reactive' | 'exploratory' | 'collaborative' | 'adaptive';
  description: string;
  trigger_conditions: string[];
  success_rate: number;
  execution_count: number;
  last_executed: string;
  impact_score: number;
  learning_source: 'experience' | 'observation' | 'instruction' | 'inference';
}

export const AgentMemoryLearningInterface: React.FC<{ agentId: string }> = ({ agentId }) => {
  const [activeTab, setActiveTab] = useState<'memories' | 'learning' | 'behaviors' | 'insights' | 'analytics'>('memories');
  
  // Memory states
  const [memories, setMemories] = useState<AgentMemory[]>([]);
  const [selectedMemory, setSelectedMemory] = useState<AgentMemory | null>(null);
  const [memoryFilter, setMemoryFilter] = useState({
    type: 'all' as string,
    importance_min: 0,
    search_query: '',
    date_range: 'all' as string
  });
  
  // Learning states
  const [learningProgress, setLearningProgress] = useState<LearningProgress | null>(null);
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null);
  
  // Behavior states
  const [autonomousBehaviors, setAutonomousBehaviors] = useState<AutonomousBehavior[]>([]);
  const [selectedBehavior, setSelectedBehavior] = useState<AutonomousBehavior | null>(null);

  // Fetch agent memories
  const { data: agentMemories, isLoading: memoriesLoading } = useQuery(
    ['agent-memories', agentId],
    () => agentMemoryApi.getAgentMemories(agentId),
    {
      onSuccess: (data) => setMemories(data),
      refetchInterval: 10000 // Refresh every 10 seconds
    }
  );

  // Fetch learning progress
  const { data: agentLearning, isLoading: learningLoading } = useQuery(
    ['agent-learning', agentId],
    () => learningApi.getLearningProgress(agentId),
    {
      onSuccess: (data) => setLearningProgress(data),
      refetchInterval: 15000 // Refresh every 15 seconds
    }
  );

  // Fetch autonomous behaviors
  const { data: agentBehaviors, isLoading: behaviorsLoading } = useQuery(
    ['agent-behaviors', agentId],
    () => autonomousAgentsApi.getAutonomousBehaviors(agentId),
    {
      onSuccess: (data) => setAutonomousBehaviors(data),
      refetchInterval: 20000 // Refresh every 20 seconds
    }
  );

  // Filter memories based on current filters
  const filteredMemories = memories.filter(memory => {
    if (memoryFilter.type !== 'all' && memory.memory_type !== memoryFilter.type) return false;
    if (memory.importance_score < memoryFilter.importance_min) return false;
    if (memoryFilter.search_query && !memory.content.toLowerCase().includes(memoryFilter.search_query.toLowerCase())) return false;
    return true;
  });

  // Create new memory
  const createMemory = async (memoryData: Partial<AgentMemory>) => {
    try {
      const response = await agentMemoryApi.createMemory(agentId, memoryData);
      setMemories(prev => [response.data, ...prev]);
      toast.success('Memory created successfully');
    } catch (error) {
      console.error('Memory creation error:', error);
      toast.error('Failed to create memory');
    }
  };

  // Consolidate memories
  const consolidateMemories = async () => {
    try {
      await agentMemoryApi.consolidateMemories(agentId);
      toast.success('Memory consolidation started');
    } catch (error) {
      console.error('Memory consolidation error:', error);
      toast.error('Failed to start memory consolidation');
    }
  };

  // Trigger learning session
  const triggerLearningSession = async (skillArea: string) => {
    try {
      await learningApi.triggerLearningSession(agentId, skillArea);
      toast.success('Learning session initiated');
    } catch (error) {
      console.error('Learning session error:', error);
      toast.error('Failed to start learning session');
    }
  };

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-6 w-6 text-purple-500" />
              <div>
                <CardTitle className="text-xl">Agent Memory & Learning System</CardTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  Comprehensive memory management, learning progress tracking, and autonomous behavior analysis
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <Activity className="h-3 w-3" />
                Agent: {agentId}
              </Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-muted p-1 rounded-lg w-fit">
        {[
          { id: 'memories', label: 'Memory System', icon: Memory },
          { id: 'learning', label: 'Learning Progress', icon: TrendingUp },
          { id: 'behaviors', label: 'Autonomous Behaviors', icon: Zap },
          { id: 'insights', label: 'Learning Insights', icon: Lightbulb },
          { id: 'analytics', label: 'Performance Analytics', icon: BarChart3 }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <tab.icon className="h-4 w-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Memory System Tab */}
      {activeTab === 'memories' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Memory List */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg flex items-center space-x-2">
                    <Memory className="h-5 w-5" />
                    <span>Agent Memories ({filteredMemories.length})</span>
                  </CardTitle>
                  <div className="flex items-center space-x-2">
                    <Button onClick={consolidateMemories} variant="outline" size="sm">
                      <RefreshCw className="h-4 w-4 mr-1" />
                      Consolidate
                    </Button>
                    <Button size="sm">
                      <Plus className="h-4 w-4 mr-1" />
                      Add Memory
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {/* Memory Filters */}
                <div className="grid grid-cols-4 gap-3 mb-4">
                  <select
                    value={memoryFilter.type}
                    onChange={(e) => setMemoryFilter(prev => ({ ...prev, type: e.target.value }))}
                    className="input text-sm"
                  >
                    <option value="all">All Types</option>
                    <option value="episodic">Episodic</option>
                    <option value="semantic">Semantic</option>
                    <option value="procedural">Procedural</option>
                    <option value="working">Working</option>
                    <option value="meta">Meta</option>
                  </select>

                  <Input
                    type="number"
                    placeholder="Min importance"
                    value={memoryFilter.importance_min}
                    onChange={(e) => setMemoryFilter(prev => ({ 
                      ...prev, 
                      importance_min: parseFloat(e.target.value) || 0 
                    }))}
                    className="text-sm"
                    min="0"
                    max="1"
                    step="0.1"
                  />

                  <Input
                    placeholder="Search memories..."
                    value={memoryFilter.search_query}
                    onChange={(e) => setMemoryFilter(prev => ({ ...prev, search_query: e.target.value }))}
                    className="text-sm"
                  />

                  <select
                    value={memoryFilter.date_range}
                    onChange={(e) => setMemoryFilter(prev => ({ ...prev, date_range: e.target.value }))}
                    className="input text-sm"
                  >
                    <option value="all">All Time</option>
                    <option value="today">Today</option>
                    <option value="week">This Week</option>
                    <option value="month">This Month</option>
                  </select>
                </div>

                <ScrollArea className="h-96">
                  <div className="space-y-3">
                    {filteredMemories.map((memory) => (
                      <div
                        key={memory.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                          selectedMemory?.id === memory.id
                            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                            : 'border-border hover:border-muted-foreground'
                        }`}
                        onClick={() => setSelectedMemory(memory)}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline" className="text-xs">
                              {memory.memory_type}
                            </Badge>
                            <Badge 
                              variant={memory.importance_score > 0.7 ? 'default' : 'secondary'}
                              className="text-xs"
                            >
                              {Math.round(memory.importance_score * 100)}% important
                            </Badge>
                          </div>
                          <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                            <Eye className="h-3 w-3" />
                            <span>{memory.access_count}</span>
                          </div>
                        </div>
                        
                        <p className="text-sm text-foreground mb-2 line-clamp-3">
                          {memory.content}
                        </p>
                        
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <div className="flex items-center space-x-2">
                            <span>Confidence: {Math.round(memory.confidence_score * 100)}%</span>
                            {memory.emotional_valence !== 0 && (
                              <span className={memory.emotional_valence > 0 ? 'text-green-600' : 'text-red-600'}>
                                {memory.emotional_valence > 0 ? '😊' : '😔'} {Math.abs(memory.emotional_valence).toFixed(2)}
                              </span>
                            )}
                          </div>
                          <span>{new Date(memory.created_at).toLocaleDateString()}</span>
                        </div>
                        
                        {memory.tags.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {memory.tags.slice(0, 3).map((tag, index) => (
                              <span key={index} className="px-2 py-1 bg-muted text-xs rounded">
                                {tag}
                              </span>
                            ))}
                            {memory.tags.length > 3 && (
                              <span className="px-2 py-1 bg-muted text-xs rounded">
                                +{memory.tags.length - 3} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {/* Memory Details Panel */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Eye className="h-5 w-5" />
                  <span>Memory Details</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {selectedMemory ? (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Memory Type
                      </label>
                      <Badge variant="outline">{selectedMemory.memory_type}</Badge>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Content
                      </label>
                      <p className="text-sm text-foreground p-3 bg-muted rounded">
                        {selectedMemory.content}
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm font-medium text-foreground mb-1">
                          Importance
                        </label>
                        <div className="flex items-center space-x-2">
                          <Progress value={selectedMemory.importance_score * 100} className="flex-1 h-2" />
                          <span className="text-sm font-medium">
                            {Math.round(selectedMemory.importance_score * 100)}%
                          </span>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-foreground mb-1">
                          Confidence
                        </label>
                        <div className="flex items-center space-x-2">
                          <Progress value={selectedMemory.confidence_score * 100} className="flex-1 h-2" />
                          <span className="text-sm font-medium">
                            {Math.round(selectedMemory.confidence_score * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Access Statistics
                      </label>
                      <div className="text-sm space-y-1">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Access Count:</span>
                          <span className="font-medium">{selectedMemory.access_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Created:</span>
                          <span className="font-medium">{new Date(selectedMemory.created_at).toLocaleDateString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Last Accessed:</span>
                          <span className="font-medium">{new Date(selectedMemory.last_accessed).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>

                    {selectedMemory.tags.length > 0 && (
                      <div>
                        <label className="block text-sm font-medium text-foreground mb-2">
                          Tags
                        </label>
                        <div className="flex flex-wrap gap-1">
                          {selectedMemory.tags.map((tag, index) => (
                            <span key={index} className="px-2 py-1 bg-muted text-xs rounded">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex space-x-2">
                      <Button size="sm" variant="outline" className="flex-1">
                        <Edit className="h-3 w-3 mr-1" />
                        Edit
                      </Button>
                      <Button size="sm" variant="outline" className="flex-1">
                        <Copy className="h-3 w-3 mr-1" />
                        Copy
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Memory className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-sm text-muted-foreground">
                      Select a memory to view details
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Learning Progress Tab */}
      {activeTab === 'learning' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <TrendingUp className="h-5 w-5" />
                <span>Skill Development</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {learningProgress ? (
                <div className="space-y-4">
                  {learningProgress.skill_areas.map((skill, index) => (
                    <div
                      key={index}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedSkill === skill.name
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-border hover:border-muted-foreground'
                      }`}
                      onClick={() => setSelectedSkill(skill.name)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium text-sm">{skill.name}</h3>
                        <Badge variant="outline" className="text-xs">
                          Level {skill.current_level}/{skill.target_level}
                        </Badge>
                      </div>

                      <div className="mb-3">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-muted-foreground">Progress</span>
                          <span className="text-xs font-medium">
                            {Math.round((skill.current_level / skill.target_level) * 100)}%
                          </span>
                        </div>
                        <Progress value={(skill.current_level / skill.target_level) * 100} className="h-2" />
                      </div>

                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Rate: +{skill.progress_rate.toFixed(2)}/day</span>
                        <span>Last: {new Date(skill.last_improvement).toLocaleDateString()}</span>
                      </div>

                      <div className="mt-2 flex flex-wrap gap-1">
                        {skill.milestones.map((milestone, idx) => (
                          <span
                            key={idx}
                            className={`px-2 py-1 text-xs rounded ${
                              milestone.achieved
                                ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-200'
                                : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                            }`}
                          >
                            {milestone.achieved && <CheckCircle className="h-3 w-3 inline mr-1" />}
                            {milestone.name}
                          </span>
                        ))}
                      </div>

                      <Button
                        onClick={() => triggerLearningSession(skill.name)}
                        size="sm"
                        variant="outline"
                        className="w-full mt-3"
                      >
                        <Zap className="h-3 w-3 mr-1" />
                        Trigger Learning Session
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <TrendingUp className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-sm text-muted-foreground">
                    No learning progress data available
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <BarChart3 className="h-5 w-5" />
                <span>Learning Metrics</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {learningProgress ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">
                        {learningProgress.learning_metrics.total_learning_sessions}
                      </div>
                      <div className="text-xs text-muted-foreground">Learning Sessions</div>
                    </div>

                    <div className="text-center p-3 bg-muted rounded-lg">
                      <div className="text-2xl font-bold text-green-600">
                        {learningProgress.learning_metrics.successful_adaptations}
                      </div>
                      <div className="text-xs text-muted-foreground">Successful Adaptations</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-muted-foreground">Knowledge Retention</span>
                        <span className="text-sm font-medium">
                          {Math.round(learningProgress.learning_metrics.knowledge_retention_rate * 100)}%
                        </span>
                      </div>
                      <Progress value={learningProgress.learning_metrics.knowledge_retention_rate * 100} className="h-2" />
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-muted-foreground">Transfer Learning</span>
                        <span className="text-sm font-medium">
                          {Math.round(learningProgress.learning_metrics.transfer_learning_score * 100)}%
                        </span>
                      </div>
                      <Progress value={learningProgress.learning_metrics.transfer_learning_score * 100} className="h-2" />
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-muted-foreground">Meta-Learning Ability</span>
                        <span className="text-sm font-medium">
                          {Math.round(learningProgress.learning_metrics.meta_learning_ability * 100)}%
                        </span>
                      </div>
                      <Progress value={learningProgress.learning_metrics.meta_learning_ability * 100} className="h-2" />
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-foreground mb-3">Behavioral Patterns</h4>
                    <div className="space-y-2">
                      {learningProgress.behavioral_patterns.map((pattern, index) => (
                        <div key={index} className="p-3 border border-border rounded-lg">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium">{pattern.pattern_name}</span>
                            <Badge
                              variant={
                                pattern.trend === 'improving' ? 'default' :
                                pattern.trend === 'stable' ? 'secondary' : 'destructive'
                              }
                              className="text-xs"
                            >
                              {pattern.trend}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground mb-2">
                            Frequency: {pattern.frequency} • Effectiveness: {Math.round(pattern.effectiveness * 100)}%
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {pattern.context.slice(0, 2).map((ctx, idx) => (
                              <span key={idx} className="px-2 py-1 bg-muted text-xs rounded">
                                {ctx}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-sm text-muted-foreground">
                    No learning metrics available
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Autonomous Behaviors Tab */}
      {activeTab === 'behaviors' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Zap className="h-5 w-5" />
                  <span>Autonomous Behaviors ({autonomousBehaviors.length})</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {autonomousBehaviors.map((behavior) => (
                    <div
                      key={behavior.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedBehavior?.id === behavior.id
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-border hover:border-muted-foreground'
                      }`}
                      onClick={() => setSelectedBehavior(behavior)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h3 className="font-medium text-sm">{behavior.behavior_type}</h3>
                          <p className="text-xs text-muted-foreground mt-1">{behavior.description}</p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {Math.round(behavior.success_rate * 100)}% success
                          </Badge>
                          <Badge variant="secondary" className="text-xs">
                            {behavior.learning_source}
                          </Badge>
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-3 text-xs mb-3">
                        <div>
                          <span className="text-muted-foreground">Executions:</span>
                          <p className="font-medium">{behavior.execution_count}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Impact Score:</span>
                          <p className="font-medium">{behavior.impact_score.toFixed(2)}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Last Executed:</span>
                          <p className="font-medium">{new Date(behavior.last_executed).toLocaleDateString()}</p>
                        </div>
                      </div>

                      <div>
                        <span className="text-xs text-muted-foreground">Trigger Conditions:</span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {behavior.trigger_conditions.slice(0, 3).map((condition, index) => (
                            <span key={index} className="px-2 py-1 bg-muted text-xs rounded">
                              {condition}
                            </span>
                          ))}
                          {behavior.trigger_conditions.length > 3 && (
                            <span className="px-2 py-1 bg-muted text-xs rounded">
                              +{behavior.trigger_conditions.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Eye className="h-5 w-5" />
                  <span>Behavior Details</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {selectedBehavior ? (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Behavior Type
                      </label>
                      <Badge variant="outline">{selectedBehavior.behavior_type}</Badge>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Description
                      </label>
                      <p className="text-sm text-foreground p-3 bg-muted rounded">
                        {selectedBehavior.description}
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Success Rate
                      </label>
                      <div className="flex items-center space-x-2">
                        <Progress value={selectedBehavior.success_rate * 100} className="flex-1 h-2" />
                        <span className="text-sm font-medium">
                          {Math.round(selectedBehavior.success_rate * 100)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-1">
                        Learning Source
                      </label>
                      <Badge variant="secondary">{selectedBehavior.learning_source}</Badge>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Trigger Conditions
                      </label>
                      <div className="space-y-1">
                        {selectedBehavior.trigger_conditions.map((condition, index) => (
                          <div key={index} className="text-sm p-2 bg-muted rounded">
                            {condition}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <span className="text-muted-foreground">Executions:</span>
                        <p className="font-medium">{selectedBehavior.execution_count}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Impact Score:</span>
                        <p className="font-medium">{selectedBehavior.impact_score.toFixed(2)}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Zap className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-sm text-muted-foreground">
                      Select a behavior to view details
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Learning Insights Tab */}
      {activeTab === 'insights' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Lightbulb className="h-5 w-5" />
                <span>Learning Insights</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Lightbulb className="h-4 w-4 text-blue-600" />
                    <span className="font-medium text-blue-800 dark:text-blue-200">Key Learning Pattern</span>
                  </div>
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    Agent shows strongest learning in collaborative scenarios, with 40% faster skill acquisition when working with other agents.
                  </p>
                </div>

                <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <TrendingUp className="h-4 w-4 text-green-600" />
                    <span className="font-medium text-green-800 dark:text-green-200">Performance Improvement</span>
                  </div>
                  <p className="text-sm text-green-700 dark:text-green-300">
                    Memory consolidation efficiency has improved by 25% over the last week, indicating better long-term retention.
                  </p>
                </div>

                <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <AlertTriangle className="h-4 w-4 text-orange-600" />
                    <span className="font-medium text-orange-800 dark:text-orange-200">Learning Challenge</span>
                  </div>
                  <p className="text-sm text-orange-700 dark:text-orange-300">
                    Agent struggles with abstract reasoning tasks, showing 30% lower success rate compared to concrete problem-solving.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Award className="h-5 w-5" />
                <span>Learning Achievements</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
                  <div className="p-2 bg-yellow-100 dark:bg-yellow-900/20 rounded-full">
                    <Star className="h-4 w-4 text-yellow-600" />
                  </div>
                  <div>
                    <p className="font-medium text-sm">Fast Learner</p>
                    <p className="text-xs text-muted-foreground">Achieved 90% skill proficiency in record time</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-full">
                    <Network className="h-4 w-4 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium text-sm">Collaborative Expert</p>
                    <p className="text-xs text-muted-foreground">Excels in multi-agent coordination tasks</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
                  <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-full">
                    <Brain className="h-4 w-4 text-green-600" />
                  </div>
                  <div>
                    <p className="font-medium text-sm">Memory Master</p>
                    <p className="text-xs text-muted-foreground">Maintains 95% knowledge retention rate</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 p-3 bg-muted rounded-lg">
                  <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-full">
                    <Zap className="h-4 w-4 text-purple-600" />
                  </div>
                  <div>
                    <p className="font-medium text-sm">Autonomous Pioneer</p>
                    <p className="text-xs text-muted-foreground">Developed 15 unique autonomous behaviors</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Performance Analytics Tab */}
      {activeTab === 'analytics' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <BarChart3 className="h-5 w-5" />
                <span>Performance Analytics</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{memories.length}</div>
                    <div className="text-xs text-muted-foreground">Total Memories</div>
                  </div>

                  <div className="text-center p-3 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {autonomousBehaviors.length}
                    </div>
                    <div className="text-xs text-muted-foreground">Behaviors</div>
                  </div>

                  <div className="text-center p-3 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">
                      {learningProgress?.skill_areas.length || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Skills</div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-foreground mb-3">Memory Distribution</h4>
                  <div className="space-y-2">
                    {['episodic', 'semantic', 'procedural', 'working', 'meta'].map((type) => {
                      const count = memories.filter(m => m.memory_type === type).length;
                      const percentage = memories.length > 0 ? (count / memories.length) * 100 : 0;
                      return (
                        <div key={type} className="flex items-center space-x-3">
                          <span className="text-sm text-muted-foreground w-20 capitalize">{type}:</span>
                          <Progress value={percentage} className="flex-1 h-2" />
                          <span className="text-sm font-medium w-12">{count}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-foreground mb-3">Behavior Success Rates</h4>
                  <div className="space-y-2">
                    {autonomousBehaviors.slice(0, 5).map((behavior) => (
                      <div key={behavior.id} className="flex items-center space-x-3">
                        <span className="text-sm text-muted-foreground w-20 truncate">{behavior.behavior_type}:</span>
                        <Progress value={behavior.success_rate * 100} className="flex-1 h-2" />
                        <span className="text-sm font-medium w-12">{Math.round(behavior.success_rate * 100)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Gauge className="h-5 w-5" />
                <span>System Health</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Memory System Health</span>
                    <Badge variant="default" className="text-xs">Excellent</Badge>
                  </div>
                  <Progress value={95} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Learning System Health</span>
                    <Badge variant="default" className="text-xs">Good</Badge>
                  </div>
                  <Progress value={87} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Behavior System Health</span>
                    <Badge variant="default" className="text-xs">Excellent</Badge>
                  </div>
                  <Progress value={92} className="h-2" />
                </div>

                <div className="pt-4 border-t border-border">
                  <h4 className="text-sm font-medium text-foreground mb-3">Recent Activity</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Last Memory Created:</span>
                      <span className="font-medium">2 hours ago</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Last Learning Session:</span>
                      <span className="font-medium">1 day ago</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Last Behavior Execution:</span>
                      <span className="font-medium">30 minutes ago</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};
