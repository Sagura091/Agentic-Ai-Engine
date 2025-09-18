import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Users, 
  Plus, 
  Share2, 
  MessageSquare, 
  ThumbsUp, 
  ThumbsDown,
  Edit,
  Eye,
  Crown,
  Shield,
  UserPlus,
  Settings,
  Bell,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Star,
  TrendingUp,
  Activity,
  FileText,
  Tag,
  MessageCircle,
  Lightbulb,
  Target,
  Award,
  Zap,
  Globe,
  Lock,
  Unlock
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';

// Types for collaborative features
interface Workspace {
  id: string;
  name: string;
  description: string;
  type: 'private' | 'team' | 'organization' | 'public' | 'research' | 'project';
  ownerId: string;
  members: Array<{
    userId: string;
    name: string;
    role: 'owner' | 'admin' | 'editor' | 'contributor' | 'reviewer' | 'viewer';
    joinedAt: string;
    avatar?: string;
  }>;
  knowledgeBases: string[];
  settings: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

interface KnowledgeContribution {
  id: string;
  contributorId: string;
  contributorName: string;
  workspaceId: string;
  type: 'document' | 'annotation' | 'correction' | 'enhancement' | 'review' | 'rating' | 'tag' | 'summary';
  targetDocumentId?: string;
  content: string;
  status: 'pending' | 'approved' | 'rejected';
  votes: Record<string, number>; // userId -> vote (1, 0, -1)
  metadata: Record<string, any>;
  createdAt: string;
}

interface CollaborativeSession {
  id: string;
  workspaceId: string;
  participants: Array<{
    userId: string;
    name: string;
    avatar?: string;
    status: 'active' | 'idle' | 'away';
  }>;
  activeDocumentId?: string;
  startedAt: string;
  lastActivity: string;
}

interface CollaborativeIntelligence {
  workspaceId: string;
  knowledgeQualityScore: number;
  collaborationActivity: {
    activeContributors: number;
    contributionsPerDay: number;
    collaborationFrequency: string;
    peakActivityHours: number[];
  };
  contributionPatterns: {
    contributionTypes: Record<string, number>;
    topContributors: Record<string, number>;
    totalContributions: number;
  };
  knowledgeGaps: string[];
  trendingTopics: string[];
  expertRecommendations: Array<{
    userId: string;
    expertiseScore: number;
    recommendationType: string;
    reason: string;
  }>;
}

const CollaborativeWorkspace: React.FC = () => {
  // Workspace state
  const [currentWorkspace, setCurrentWorkspace] = useState<Workspace | null>(null);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);

  // Collaboration state
  const [contributions, setContributions] = useState<KnowledgeContribution[]>([]);
  const [activeSessions, setActiveSessions] = useState<CollaborativeSession[]>([]);
  const [intelligence, setIntelligence] = useState<CollaborativeIntelligence | null>(null);

  // UI state
  const [selectedTab, setSelectedTab] = useState('overview');
  const [showCreateContribution, setShowCreateContribution] = useState(false);
  const [newContribution, setNewContribution] = useState({
    type: 'annotation' as const,
    content: '',
    targetDocumentId: ''
  });

  // Role configurations
  const roleConfig = {
    owner: { icon: <Crown className="h-4 w-4" />, color: 'text-yellow-600', label: 'Owner' },
    admin: { icon: <Shield className="h-4 w-4" />, color: 'text-red-600', label: 'Admin' },
    editor: { icon: <Edit className="h-4 w-4" />, color: 'text-blue-600', label: 'Editor' },
    contributor: { icon: <UserPlus className="h-4 w-4" />, color: 'text-green-600', label: 'Contributor' },
    reviewer: { icon: <Eye className="h-4 w-4" />, color: 'text-purple-600', label: 'Reviewer' },
    viewer: { icon: <Eye className="h-4 w-4" />, color: 'text-gray-600', label: 'Viewer' }
  };

  // Workspace type configurations
  const workspaceTypeConfig = {
    private: { icon: <Lock className="h-4 w-4" />, color: 'text-gray-600', label: 'Private' },
    team: { icon: <Users className="h-4 w-4" />, color: 'text-blue-600', label: 'Team' },
    organization: { icon: <Shield className="h-4 w-4" />, color: 'text-purple-600', label: 'Organization' },
    public: { icon: <Globe className="h-4 w-4" />, color: 'text-green-600', label: 'Public' },
    research: { icon: <Lightbulb className="h-4 w-4" />, color: 'text-orange-600', label: 'Research' },
    project: { icon: <Target className="h-4 w-4" />, color: 'text-indigo-600', label: 'Project' }
  };

  // Load workspace data
  const loadWorkspaceData = useCallback(async () => {
    try {
      // Simulate API calls
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Mock workspace data
      const mockWorkspaces: Workspace[] = [
        {
          id: 'ws-1',
          name: 'AI Research Team',
          description: 'Collaborative workspace for AI research and development',
          type: 'research',
          ownerId: 'user-1',
          members: [
            {
              userId: 'user-1',
              name: 'Dr. Sarah Chen',
              role: 'owner',
              joinedAt: '2024-01-01T00:00:00Z',
              avatar: '/avatars/sarah.jpg'
            },
            {
              userId: 'user-2',
              name: 'Alex Rodriguez',
              role: 'editor',
              joinedAt: '2024-01-05T00:00:00Z',
              avatar: '/avatars/alex.jpg'
            },
            {
              userId: 'user-3',
              name: 'Emily Johnson',
              role: 'contributor',
              joinedAt: '2024-01-10T00:00:00Z',
              avatar: '/avatars/emily.jpg'
            }
          ],
          knowledgeBases: ['kb-ai-papers', 'kb-research-notes'],
          settings: {
            autoApproveContributions: false,
            requireReview: true,
            publicVisibility: false
          },
          createdAt: '2024-01-01T00:00:00Z',
          updatedAt: '2024-01-15T00:00:00Z'
        }
      ];

      setWorkspaces(mockWorkspaces);
      setCurrentWorkspace(mockWorkspaces[0]);

      // Mock contributions
      const mockContributions: KnowledgeContribution[] = [
        {
          id: 'contrib-1',
          contributorId: 'user-2',
          contributorName: 'Alex Rodriguez',
          workspaceId: 'ws-1',
          type: 'annotation',
          targetDocumentId: 'doc-123',
          content: 'This section needs clarification on the mathematical formulation.',
          status: 'pending',
          votes: { 'user-1': 1, 'user-3': 1 },
          metadata: { section: 'methodology' },
          createdAt: '2024-01-15T10:00:00Z'
        },
        {
          id: 'contrib-2',
          contributorId: 'user-3',
          contributorName: 'Emily Johnson',
          workspaceId: 'ws-1',
          type: 'enhancement',
          content: 'Added comprehensive examples and use cases for better understanding.',
          status: 'approved',
          votes: { 'user-1': 1, 'user-2': 1 },
          metadata: { category: 'examples' },
          createdAt: '2024-01-14T15:30:00Z'
        }
      ];

      setContributions(mockContributions);

      // Mock collaborative intelligence
      const mockIntelligence: CollaborativeIntelligence = {
        workspaceId: 'ws-1',
        knowledgeQualityScore: 0.85,
        collaborationActivity: {
          activeContributors: 3,
          contributionsPerDay: 2.5,
          collaborationFrequency: 'high',
          peakActivityHours: [9, 10, 14, 15, 16]
        },
        contributionPatterns: {
          contributionTypes: {
            annotation: 5,
            enhancement: 3,
            correction: 2,
            review: 4
          },
          topContributors: {
            'user-2': 6,
            'user-3': 4,
            'user-1': 4
          },
          totalContributions: 14
        },
        knowledgeGaps: [
          'Mathematical proofs section needs more detail',
          'Implementation examples are sparse'
        ],
        trendingTopics: [
          'neural networks',
          'transformer architecture',
          'attention mechanisms',
          'optimization techniques'
        ],
        expertRecommendations: [
          {
            userId: 'user-2',
            expertiseScore: 8.5,
            recommendationType: 'active_contributor',
            reason: 'High-quality contributions with score 8.5'
          }
        ]
      };

      setIntelligence(mockIntelligence);

    } catch (error) {
      console.error('Failed to load workspace data:', error);
      toast.error('Failed to load workspace data');
    }
  }, []);

  // Submit contribution
  const submitContribution = useCallback(async () => {
    if (!newContribution.content.trim()) {
      toast.error('Please enter contribution content');
      return;
    }

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));

      const contribution: KnowledgeContribution = {
        id: `contrib-${Date.now()}`,
        contributorId: 'current-user',
        contributorName: 'Current User',
        workspaceId: currentWorkspace?.id || '',
        type: newContribution.type,
        targetDocumentId: newContribution.targetDocumentId || undefined,
        content: newContribution.content,
        status: 'pending',
        votes: {},
        metadata: {},
        createdAt: new Date().toISOString()
      };

      setContributions(prev => [contribution, ...prev]);
      setNewContribution({ type: 'annotation', content: '', targetDocumentId: '' });
      setShowCreateContribution(false);

      toast.success('Contribution submitted successfully');

    } catch (error) {
      console.error('Failed to submit contribution:', error);
      toast.error('Failed to submit contribution');
    }
  }, [newContribution, currentWorkspace]);

  // Vote on contribution
  const voteContribution = useCallback(async (contributionId: string, vote: number) => {
    try {
      setContributions(prev => prev.map(contrib => 
        contrib.id === contributionId 
          ? { ...contrib, votes: { ...contrib.votes, 'current-user': vote } }
          : contrib
      ));

      toast.success('Vote recorded');

    } catch (error) {
      console.error('Failed to vote:', error);
      toast.error('Failed to record vote');
    }
  }, []);

  // Initialize data
  useEffect(() => {
    loadWorkspaceData();
  }, [loadWorkspaceData]);

  if (!currentWorkspace) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                {workspaceTypeConfig[currentWorkspace.type].icon}
                <div>
                  <CardTitle className="text-xl">{currentWorkspace.name}</CardTitle>
                  <p className="text-sm text-muted-foreground mt-1">
                    {currentWorkspace.description}
                  </p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <Users className="h-3 w-3" />
                {currentWorkspace.members.length} members
              </Badge>
              <Badge variant="outline" className="flex items-center gap-1">
                <Activity className="h-3 w-3" />
                {contributions.length} contributions
              </Badge>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="contributions">Contributions</TabsTrigger>
          <TabsTrigger value="members">Members</TabsTrigger>
          <TabsTrigger value="intelligence">Intelligence</TabsTrigger>
          <TabsTrigger value="sessions">Sessions</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Knowledge Quality</p>
                    <p className="text-2xl font-bold">
                      {intelligence ? (intelligence.knowledgeQualityScore * 100).toFixed(0) : 0}%
                    </p>
                  </div>
                  <Star className="h-8 w-8 text-yellow-500" />
                </div>
                <Progress value={intelligence ? intelligence.knowledgeQualityScore * 100 : 0} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Active Contributors</p>
                    <p className="text-2xl font-bold">
                      {intelligence?.collaborationActivity.activeContributors || 0}
                    </p>
                  </div>
                  <Users className="h-8 w-8 text-blue-500" />
                </div>
                <div className="mt-2">
                  <Badge variant="secondary" className="text-xs">
                    <TrendingUp className="h-3 w-3 mr-1" />
                    {intelligence?.collaborationActivity.collaborationFrequency || 'low'} activity
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Daily Contributions</p>
                    <p className="text-2xl font-bold">
                      {intelligence?.collaborationActivity.contributionsPerDay.toFixed(1) || '0.0'}
                    </p>
                  </div>
                  <Activity className="h-8 w-8 text-green-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Pending Reviews</p>
                    <p className="text-2xl font-bold">
                      {contributions.filter(c => c.status === 'pending').length}
                    </p>
                  </div>
                  <Clock className="h-8 w-8 text-orange-500" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {contributions.slice(0, 5).map((contribution) => (
                  <div key={contribution.id} className="flex items-center gap-3 p-3 border rounded">
                    <Avatar className="h-8 w-8">
                      <AvatarFallback>{contribution.contributorName[0]}</AvatarFallback>
                    </Avatar>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium">{contribution.contributorName}</span>
                        <Badge variant="outline" className="text-xs">
                          {contribution.type}
                        </Badge>
                        <Badge 
                          variant={contribution.status === 'approved' ? 'default' : 'secondary'}
                          className="text-xs"
                        >
                          {contribution.status === 'approved' && <CheckCircle className="h-3 w-3 mr-1" />}
                          {contribution.status === 'rejected' && <XCircle className="h-3 w-3 mr-1" />}
                          {contribution.status === 'pending' && <Clock className="h-3 w-3 mr-1" />}
                          {contribution.status}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-1">
                        {contribution.content}
                      </p>
                    </div>
                    
                    <div className="text-xs text-muted-foreground">
                      {new Date(contribution.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Contributions Tab */}
        <TabsContent value="contributions" className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Knowledge Contributions</h2>
            <Button onClick={() => setShowCreateContribution(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Contribution
            </Button>
          </div>

          {/* Create Contribution Modal */}
          {showCreateContribution && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Create New Contribution</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Contribution Type</label>
                  <Select 
                    value={newContribution.type} 
                    onValueChange={(value: any) => setNewContribution(prev => ({ ...prev, type: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="annotation">Annotation</SelectItem>
                      <SelectItem value="correction">Correction</SelectItem>
                      <SelectItem value="enhancement">Enhancement</SelectItem>
                      <SelectItem value="review">Review</SelectItem>
                      <SelectItem value="summary">Summary</SelectItem>
                      <SelectItem value="tag">Tag</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Target Document (Optional)</label>
                  <Input
                    placeholder="Document ID or leave empty for general contribution"
                    value={newContribution.targetDocumentId}
                    onChange={(e) => setNewContribution(prev => ({ ...prev, targetDocumentId: e.target.value }))}
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Content</label>
                  <Textarea
                    placeholder="Describe your contribution..."
                    value={newContribution.content}
                    onChange={(e) => setNewContribution(prev => ({ ...prev, content: e.target.value }))}
                    rows={4}
                  />
                </div>

                <div className="flex gap-2">
                  <Button onClick={submitContribution}>
                    Submit Contribution
                  </Button>
                  <Button variant="outline" onClick={() => setShowCreateContribution(false)}>
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Contributions List */}
          <div className="space-y-4">
            {contributions.map((contribution) => (
              <Card key={contribution.id}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <Avatar className="h-8 w-8">
                        <AvatarFallback>{contribution.contributorName[0]}</AvatarFallback>
                      </Avatar>
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{contribution.contributorName}</span>
                          <Badge variant="outline" className="text-xs">
                            {contribution.type}
                          </Badge>
                          <Badge 
                            variant={contribution.status === 'approved' ? 'default' : 'secondary'}
                            className="text-xs"
                          >
                            {contribution.status}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {new Date(contribution.createdAt).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    
                    {contribution.status === 'pending' && (
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => voteContribution(contribution.id, 1)}
                        >
                          <ThumbsUp className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => voteContribution(contribution.id, -1)}
                        >
                          <ThumbsDown className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                  
                  <p className="text-sm mb-3">{contribution.content}</p>
                  
                  {Object.keys(contribution.votes).length > 0 && (
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <ThumbsUp className="h-3 w-3" />
                        {Object.values(contribution.votes).filter(v => v > 0).length} upvotes
                      </span>
                      <span className="flex items-center gap-1">
                        <ThumbsDown className="h-3 w-3" />
                        {Object.values(contribution.votes).filter(v => v < 0).length} downvotes
                      </span>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Members Tab */}
        <TabsContent value="members" className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Workspace Members</h2>
            <Button variant="outline">
              <UserPlus className="h-4 w-4 mr-2" />
              Invite Member
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {currentWorkspace.members.map((member) => (
              <Card key={member.userId}>
                <CardContent className="p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <Avatar className="h-10 w-10">
                      <AvatarImage src={member.avatar} />
                      <AvatarFallback>{member.name[0]}</AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <h3 className="font-medium">{member.name}</h3>
                      <div className="flex items-center gap-1">
                        {roleConfig[member.role].icon}
                        <span className={`text-sm ${roleConfig[member.role].color}`}>
                          {roleConfig[member.role].label}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-xs text-muted-foreground">
                    Joined {new Date(member.joinedAt).toLocaleDateString()}
                  </div>
                  
                  {intelligence && intelligence.contributionPatterns.topContributors[member.userId] && (
                    <div className="mt-2">
                      <Badge variant="secondary" className="text-xs">
                        <Award className="h-3 w-3 mr-1" />
                        {intelligence.contributionPatterns.topContributors[member.userId]} contributions
                      </Badge>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Intelligence Tab */}
        <TabsContent value="intelligence" className="space-y-4">
          <h2 className="text-lg font-semibold">Collaborative Intelligence</h2>

          {intelligence && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Knowledge Gaps */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Knowledge Gaps</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {intelligence.knowledgeGaps.map((gap, index) => (
                      <div key={index} className="flex items-center gap-2 p-2 bg-muted rounded">
                        <AlertCircle className="h-4 w-4 text-orange-500" />
                        <span className="text-sm">{gap}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Trending Topics */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Trending Topics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {intelligence.trendingTopics.map((topic, index) => (
                      <Badge key={index} variant="outline" className="flex items-center gap-1">
                        <TrendingUp className="h-3 w-3" />
                        {topic}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Contribution Patterns */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Contribution Patterns</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(intelligence.contributionPatterns.contributionTypes).map(([type, count]) => (
                      <div key={type} className="flex justify-between items-center">
                        <span className="text-sm capitalize">{type}</span>
                        <div className="flex items-center gap-2">
                          <Progress value={(count / intelligence.contributionPatterns.totalContributions) * 100} className="w-20 h-2" />
                          <span className="text-sm text-muted-foreground">{count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Expert Recommendations */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Expert Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {intelligence.expertRecommendations.map((expert, index) => (
                      <div key={index} className="flex items-center gap-3 p-3 border rounded">
                        <Avatar className="h-8 w-8">
                          <AvatarFallback>E</AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-medium">Expert User</span>
                            <Badge variant="outline" className="text-xs">
                              Score: {expert.expertiseScore.toFixed(1)}
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground">{expert.reason}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Sessions Tab */}
        <TabsContent value="sessions" className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Collaborative Sessions</h2>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Start Session
            </Button>
          </div>

          <Card>
            <CardContent className="p-8 text-center">
              <MessageSquare className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No Active Sessions</h3>
              <p className="text-muted-foreground mb-4">
                Start a collaborative session to work together in real-time
              </p>
              <Button>
                <Zap className="h-4 w-4 mr-2" />
                Start Collaboration
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CollaborativeWorkspace;
