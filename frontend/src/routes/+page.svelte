<script lang="ts">
	import { onMount } from 'svelte';
	import {
		agents,
		workflows,
		systemMetrics,
		connectionStatus,
		notificationActions
	} from '$stores';

	// SvelteKit props
	export let params: Record<string, string> = {};
	import { apiClient } from '$services/api';
	import { websocketService } from '$services/websocket';
	import { 
		Bot, 
		Workflow, 
		Zap, 
		Activity, 
		TrendingUp, 
		Users, 
		Clock, 
		CheckCircle,
		AlertTriangle,
		Plus,
		ArrowRight,
		Sparkles,
		Brain,
		Network
	} from 'lucide-svelte';
	
	// State
	let loading = true;
	let stats = {
		totalAgents: 0,
		activeAgents: 0,
		totalWorkflows: 0,
		activeWorkflows: 0,
		totalExecutions: 0,
		successRate: 95.8,
		avgResponseTime: 1.2
	};
	
	// Recent activity - loaded from backend
	let recentActivity: any[] = [];
	
	onMount(async () => {
		console.log('📊 Dashboard mounting...');

		// Load initial data
		await loadDashboardData();

		// Subscribe to real-time updates
		websocketService.subscribeToSystemMetrics((metrics) => {
			console.log('📈 Received system metrics via WebSocket:', metrics);
			systemMetrics.set(metrics);
			updateStats(metrics);
		});

		// Request initial system status
		websocketService.requestSystemStatus();

		// Test WebSocket connection manually
		setTimeout(() => {
			console.log('🧪 Testing WebSocket connection...');
			const testWs = new WebSocket('ws://localhost:8888/ws');
			testWs.onopen = () => {
				console.log('✅ Direct WebSocket test: CONNECTED');
				testWs.close();
			};
			testWs.onerror = (error) => {
				console.error('❌ Direct WebSocket test: FAILED', error);
			};
			testWs.onclose = (event) => {
				console.log('🔌 Direct WebSocket test: CLOSED', event.code, event.reason);
			};
		}, 2000);

		loading = false;
	});
	
	async function loadDashboardData() {
		try {
			// Load all data in parallel for much faster performance
			const [agentsResponse, workflowsResponse, metricsResponse] = await Promise.all([
				apiClient.getAgents(),
				apiClient.getWorkflows(),
				apiClient.getSystemMetrics()
			]);

			// Process agents data
			if (agentsResponse?.success && agentsResponse?.data) {
				agents.set(agentsResponse.data);
				stats.totalAgents = agentsResponse.data.length || 0;
				stats.activeAgents = agentsResponse.data.filter(a => a?.status === 'running')?.length || 0;

				// Generate recent activity from agents data
				recentActivity = agentsResponse.data.slice(0, 3).map((agent: any, index: number) => ({
					id: agent.id || `activity-${index}`,
					type: 'agent_status',
					title: `Agent: ${agent.name || 'Unnamed Agent'}`,
					description: `Status: ${agent.status || 'idle'} | Type: ${agent.type || 'unknown'}`,
					timestamp: agent.updated_at || agent.created_at || new Date().toISOString(),
					icon: Bot,
					color: agent.status === 'running' ? 'text-green-400' : 'text-blue-400'
				}));
			}

			// Process workflows data
			if (workflowsResponse?.success && workflowsResponse?.data) {
				workflows.set(workflowsResponse.data);
				stats.totalWorkflows = workflowsResponse.data.length || 0;
				stats.activeWorkflows = workflowsResponse.data.filter(w => w?.status === 'active')?.length || 0;
			}

			// Process system metrics
			if (metricsResponse?.success && metricsResponse?.data) {
				systemMetrics.set(metricsResponse.data);
				updateStats(metricsResponse.data);
			}

			// If no agents, show system status activity
			if (recentActivity.length === 0) {
				recentActivity = [
					{
						id: 'system-ready',
						type: 'system_status',
						title: 'System Online',
						description: 'Agentic AI backend is running and ready for agent creation',
						timestamp: new Date().toISOString(),
						icon: CheckCircle,
						color: 'text-green-400'
					}
				];
			}

		} catch (error) {
			console.error('Failed to load dashboard data:', error);
			notificationActions.add({
				type: 'error',
				title: 'Dashboard Load Failed',
				message: 'Unable to load dashboard data. Please refresh the page.'
			});

			// Show error activity
			recentActivity = [
				{
					id: 'connection-error',
					type: 'system_error',
					title: 'Connection Error',
					description: 'Unable to connect to backend. Please check if the server is running.',
					timestamp: new Date().toISOString(),
					icon: AlertTriangle,
					color: 'text-red-400'
				}
			];
		}
	}
	
	function updateStats(metrics: any) {
		if (metrics) {
			stats.totalExecutions = metrics.total_executions || 0;
			stats.successRate = metrics.success_rate || 0;
			stats.avgResponseTime = metrics.average_response_time || 0;
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function timeAgo(timestamp: string): string {
		const now = new Date();
		const time = new Date(timestamp);
		const diffInMinutes = Math.floor((now.getTime() - time.getTime()) / (1000 * 60));
		
		if (diffInMinutes < 1) return 'Just now';
		if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
		if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
		return `${Math.floor(diffInMinutes / 1440)}d ago`;
	}
</script>

<svelte:head>
	<title>Dashboard - Agentic AI</title>
</svelte:head>

<div class="dashboard-container">
	<!-- Welcome Header -->
	<div class="welcome-section">
		<div class="welcome-content">
			<div class="welcome-text">
				<h1 class="welcome-title">
					Welcome to <span class="text-gradient">Agentic AI</span>
				</h1>
				<p class="welcome-subtitle">
					The most revolutionary platform for building, testing, and deploying intelligent AI agents.
					Create extraordinary workflows that push the boundaries of what's possible.
				</p>
			</div>
			
			<div class="quick-actions">
				<a href="/agents" class="action-btn primary">
					<Bot class="w-5 h-5" />
					<span>Create Agent</span>
					<ArrowRight class="w-4 h-4" />
				</a>
				<a href="/workflows" class="action-btn secondary">
					<Workflow class="w-5 h-5" />
					<span>Build Workflow</span>
				</a>
			</div>
		</div>
		
		<!-- Connection Status Indicator -->
		<div class="connection-indicator">
			{#if $connectionStatus === 'connected'}
				<div class="status-badge connected">
					<div class="status-dot"></div>
					<span>System Online</span>
				</div>
			{:else if $connectionStatus === 'connecting'}
				<div class="status-badge connecting">
					<div class="status-dot"></div>
					<span>Connecting...</span>
				</div>
			{:else}
				<div class="status-badge disconnected">
					<div class="status-dot"></div>
					<span>Offline</span>
				</div>
			{/if}
		</div>
	</div>
	
	<!-- Stats Grid -->
	<div class="stats-grid">
		<div class="stat-card">
			<div class="stat-header">
				<Bot class="stat-icon text-accent-blue" />
				<span class="stat-label">Total Agents</span>
			</div>
			<div class="stat-value">{formatNumber(stats.totalAgents)}</div>
			<div class="stat-change positive">
				<TrendingUp class="w-4 h-4" />
				<span>{stats.activeAgents} active</span>
			</div>
		</div>
		
		<div class="stat-card">
			<div class="stat-header">
				<Workflow class="stat-icon text-accent-purple" />
				<span class="stat-label">Workflows</span>
			</div>
			<div class="stat-value">{formatNumber(stats.totalWorkflows)}</div>
			<div class="stat-change positive">
				<Activity class="w-4 h-4" />
				<span>{stats.activeWorkflows} running</span>
			</div>
		</div>
		
		<div class="stat-card">
			<div class="stat-header">
				<Zap class="stat-icon text-accent-green" />
				<span class="stat-label">Executions</span>
			</div>
			<div class="stat-value">{formatNumber(stats.totalExecutions)}</div>
			<div class="stat-change positive">
				<TrendingUp class="w-4 h-4" />
				<span>+12% today</span>
			</div>
		</div>
		
		<div class="stat-card">
			<div class="stat-header">
				<CheckCircle class="stat-icon text-accent-orange" />
				<span class="stat-label">Success Rate</span>
			</div>
			<div class="stat-value">{stats.successRate.toFixed(1)}%</div>
			<div class="stat-change positive">
				<TrendingUp class="w-4 h-4" />
				<span>Excellent</span>
			</div>
		</div>
	</div>
	
	<!-- Main Content Grid -->
	<div class="content-grid">
		<!-- Recent Activity -->
		<div class="activity-section">
			<div class="section-header">
				<h2 class="section-title">Recent Activity</h2>
				<button class="view-all-btn">View All</button>
			</div>
			
			<div class="activity-list">
				{#each recentActivity as activity (activity.id)}
					<div class="activity-item">
						<div class="activity-icon {activity.color}">
							<svelte:component this={activity.icon} class="w-5 h-5" />
						</div>
						<div class="activity-content">
							<h3 class="activity-title">{activity.title}</h3>
							<p class="activity-description">{activity.description}</p>
							<span class="activity-time">{timeAgo(activity.timestamp)}</span>
						</div>
					</div>
				{/each}
			</div>
		</div>
		
		<!-- System Health -->
		<div class="health-section">
			<div class="section-header">
				<h2 class="section-title">System Health</h2>
				<div class="health-status">
					<CheckCircle class="w-5 h-5 text-green-400" />
					<span class="text-green-400">All Systems Operational</span>
				</div>
			</div>
			
			<div class="health-metrics">
				<div class="metric-item">
					<div class="metric-header">
						<Brain class="w-4 h-4 text-accent-blue" />
						<span class="metric-name">AI Engine</span>
					</div>
					<div class="metric-bar">
						<div class="metric-fill" style="width: 98%"></div>
					</div>
					<span class="metric-value">98%</span>
				</div>
				
				<div class="metric-item">
					<div class="metric-header">
						<Network class="w-4 h-4 text-accent-purple" />
						<span class="metric-name">Network</span>
					</div>
					<div class="metric-bar">
						<div class="metric-fill" style="width: 95%"></div>
					</div>
					<span class="metric-value">95%</span>
				</div>
				
				<div class="metric-item">
					<div class="metric-header">
						<Activity class="w-4 h-4 text-accent-green" />
						<span class="metric-name">Performance</span>
					</div>
					<div class="metric-bar">
						<div class="metric-fill" style="width: 92%"></div>
					</div>
					<span class="metric-value">92%</span>
				</div>
			</div>
		</div>
		
		<!-- Quick Start -->
		<div class="quickstart-section">
			<div class="section-header">
				<h2 class="section-title">Quick Start</h2>
			</div>
			
			<div class="quickstart-cards">
				<a href="/agents" class="quickstart-card group">
					<Bot class="quickstart-icon text-accent-blue" />
					<h3 class="quickstart-title">Create Your First Agent</h3>
					<p class="quickstart-description">
						Build an intelligent AI agent in minutes with our visual builder
					</p>
					<ArrowRight class="quickstart-arrow w-4 h-4" />
				</a>

				<a href="/workflows" class="quickstart-card group">
					<Workflow class="quickstart-icon text-accent-purple" />
					<h3 class="quickstart-title">Design a Workflow</h3>
					<p class="quickstart-description">
						Connect multiple agents for complex automation tasks
					</p>
					<ArrowRight class="quickstart-arrow w-4 h-4" />
				</a>

				<a href="/testing" class="quickstart-card group">
					<Zap class="quickstart-icon text-accent-green" />
					<h3 class="quickstart-title">Test & Deploy</h3>
					<p class="quickstart-description">
						Validate your agents with our comprehensive testing suite
					</p>
					<ArrowRight class="quickstart-arrow w-4 h-4" />
				</a>
			</div>
		</div>
	</div>
</div>

<style lang="postcss">
	.dashboard-container {
		@apply space-y-8;
	}
	
	/* Welcome Section */
	.welcome-section {
		@apply flex items-center justify-between p-8 rounded-2xl;
		background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
		border: 1px solid rgba(59, 130, 246, 0.2);
	}
	
	.welcome-content {
		@apply flex-1 space-y-6;
	}
	
	.welcome-text {
		@apply space-y-3;
	}
	
	.welcome-title {
		@apply text-3xl font-bold text-white;
	}
	
	.welcome-subtitle {
		@apply text-lg text-dark-300 max-w-2xl;
	}
	
	.quick-actions {
		@apply flex items-center space-x-4;
	}
	
	.action-btn {
		@apply inline-flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-200;
		text-decoration: none;
	}
	
	.action-btn.primary {
		@apply bg-primary-600 text-white hover:bg-primary-700 hover:scale-105;
	}
	
	.action-btn.secondary {
		@apply bg-dark-800 text-dark-300 hover:bg-dark-700 hover:text-white;
	}
	
	.connection-indicator {
		@apply flex-shrink-0;
	}
	
	.status-badge {
		@apply flex items-center space-x-2 px-4 py-2 rounded-lg border;
	}
	
	.status-badge.connected {
		@apply bg-green-500/10 border-green-500/30 text-green-400;
	}
	
	.status-badge.connecting {
		@apply bg-yellow-500/10 border-yellow-500/30 text-yellow-400;
	}
	
	.status-badge.disconnected {
		@apply bg-red-500/10 border-red-500/30 text-red-400;
	}
	
	.status-dot {
		@apply w-2 h-2 rounded-full animate-pulse;
	}
	
	.status-badge.connected .status-dot {
		@apply bg-green-400;
	}
	
	.status-badge.connecting .status-dot {
		@apply bg-yellow-400;
	}
	
	.status-badge.disconnected .status-dot {
		@apply bg-red-400;
	}
	
	/* Stats Grid */
	.stats-grid {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6;
	}
	
	.stat-card {
		@apply p-6 bg-dark-800 border border-dark-700 rounded-xl hover:border-dark-600 transition-all duration-200;
	}
	
	.stat-header {
		@apply flex items-center space-x-3 mb-4;
	}
	
	.stat-icon {
		@apply w-8 h-8;
	}
	
	.stat-label {
		@apply text-sm font-medium text-dark-400;
	}
	
	.stat-value {
		@apply text-3xl font-bold text-white mb-2;
	}
	
	.stat-change {
		@apply flex items-center space-x-1 text-sm;
	}
	
	.stat-change.positive {
		@apply text-green-400;
	}
	
	/* Content Grid */
	.content-grid {
		@apply grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8;
	}
	
	.activity-section,
	.health-section,
	.quickstart-section {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6;
	}
	
	.section-header {
		@apply flex items-center justify-between mb-6;
	}
	
	.section-title {
		@apply text-lg font-semibold text-white;
	}
	
	.view-all-btn {
		@apply text-sm text-primary-400 hover:text-primary-300 transition-colors duration-200;
	}
	
	/* Activity List */
	.activity-list {
		@apply space-y-4;
	}
	
	.activity-item {
		@apply flex items-start space-x-3 p-3 rounded-lg hover:bg-dark-700/50 transition-colors duration-200;
	}
	
	.activity-icon {
		@apply flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center bg-dark-700;
	}
	
	.activity-content {
		@apply flex-1 min-w-0 space-y-1;
	}
	
	.activity-title {
		@apply text-sm font-medium text-white;
	}
	
	.activity-description {
		@apply text-sm text-dark-300;
	}
	
	.activity-time {
		@apply text-xs text-dark-400;
	}
	
	/* Health Metrics */
	.health-status {
		@apply flex items-center space-x-2 text-sm;
	}
	
	.health-metrics {
		@apply space-y-4;
	}
	
	.metric-item {
		@apply space-y-2;
	}
	
	.metric-header {
		@apply flex items-center justify-between;
	}
	
	.metric-name {
		@apply flex items-center space-x-2 text-sm text-dark-300;
	}
	
	.metric-bar {
		@apply h-2 bg-dark-700 rounded-full overflow-hidden;
	}
	
	.metric-fill {
		@apply h-full bg-gradient-to-r from-accent-blue to-accent-purple transition-all duration-500;
	}
	
	.metric-value {
		@apply text-sm font-medium text-white;
	}
	
	/* Quick Start Cards */
	.quickstart-cards {
		@apply space-y-4;
	}
	
	.quickstart-card {
		@apply block p-4 rounded-lg border border-dark-700 hover:border-dark-600 transition-all duration-200;
		text-decoration: none;
	}

	.quickstart-card:hover {
		background-color: rgba(51, 65, 85, 0.3); /* hover:bg-dark-700/30 */
	}
	
	.quickstart-icon {
		@apply w-8 h-8 mb-3;
	}
	
	.quickstart-title {
		@apply text-base font-semibold text-white mb-2;
	}
	
	.quickstart-description {
		@apply text-sm text-dark-300 mb-3;
	}
	
	.quickstart-arrow {
		@apply text-dark-400 group-hover:text-primary-400 group-hover:translate-x-1 transition-all duration-200;
	}
	
	/* Mobile Responsiveness */
	@media (max-width: 768px) {
		.welcome-section {
			@apply flex-col items-start space-y-6 p-6;
		}
		
		.quick-actions {
			@apply flex-col w-full space-y-3 space-x-0;
		}
		
		.action-btn {
			@apply w-full justify-center;
		}
		
		.content-grid {
			@apply grid-cols-1;
		}
	}
</style>
