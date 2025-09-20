<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import {
		agents,
		agentsLoading,
		agentsError,
		agentBuilder,
		notificationActions
	} from '$stores';

	// SvelteKit props
	export let params: Record<string, string> = {};
	import { apiClient } from '$services/api';
	import { websocketService } from '$services/websocket';
	import { 
		Bot, 
		Plus, 
		Search, 
		Filter, 
		MoreVertical, 
		Play, 
		Pause, 
		Settings, 
		Trash2,
		Eye,
		Copy,
		Edit,
		Zap,
		Brain,
		Activity,
		Clock,
		TrendingUp
	} from 'lucide-svelte';
	import type { Agent, AgentStatus, AgentType } from '$types';
	
	// State
	let searchQuery = '';
	let selectedFilter = 'all';
	let viewMode = 'grid'; // 'grid' | 'list'
	let showFilters = false;
	
	// Filters
	const filterOptions = [
		{ value: 'all', label: 'All Agents', count: 0 },
		{ value: 'running', label: 'Running', count: 0 },
		{ value: 'idle', label: 'Idle', count: 0 },
		{ value: 'error', label: 'Error', count: 0 }
	];
	
	// Agent type colors
	const agentTypeColors: Record<string, string> = {
		'react': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
		'knowledge_search': 'bg-green-500/20 text-green-400 border-green-500/30',
		'rag': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
		'workflow': 'bg-orange-500/20 text-orange-400 border-orange-500/30',
		'multimodal': 'bg-pink-500/20 text-pink-400 border-pink-500/30',
		'composite': 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30',
		'autonomous': 'bg-red-500/20 text-red-400 border-red-500/30'
	};
	
	// Status colors
	const statusColors: Record<AgentStatus, string> = {
		'idle': 'bg-gray-500/20 text-gray-400',
		'running': 'bg-green-500/20 text-green-400',
		'paused': 'bg-yellow-500/20 text-yellow-400',
		'error': 'bg-red-500/20 text-red-400',
		'completed': 'bg-blue-500/20 text-blue-400'
	};
	
	onMount(async () => {
		await loadAgents();
		
		// Subscribe to real-time agent updates
		websocketService.on('agent_status_update' as any, (data) => {
			updateAgentStatus(data.agent_id, data.status);
		});
	});
	
	async function loadAgents() {
		agentsLoading.set(true);
		agentsError.set(null);
		
		try {
			const response = await apiClient.getAgents();
			if (response.success && response.data) {
				agents.set(response.data);
				updateFilterCounts(response.data);
			} else {
				throw new Error(response.error || 'Failed to load agents');
			}
		} catch (error) {
			console.error('Failed to load agents:', error);
			agentsError.set(error instanceof Error ? error.message : 'Unknown error');
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Agents',
				message: 'Unable to load agents. Please try again.'
			});
		} finally {
			agentsLoading.set(false);
		}
	}
	
	function updateFilterCounts(agentList: Agent[]) {
		const safeAgentList = agentList || [];
		filterOptions[0].count = safeAgentList.length;
		filterOptions[1].count = safeAgentList.filter(a => a.status === 'running').length;
		filterOptions[2].count = safeAgentList.filter(a => a.status === 'idle').length;
		filterOptions[3].count = safeAgentList.filter(a => a.status === 'error').length;
	}
	
	function updateAgentStatus(agentId: string, status: AgentStatus) {
		agents.update(list => 
			list.map(agent => 
				agent.id === agentId ? { ...agent, status } : agent
			)
		);
	}
	
	// Filter and search agents
	$: filteredAgents = ($agents || []).filter(agent => {
		// Apply status filter
		if (selectedFilter !== 'all' && agent.status !== selectedFilter) {
			return false;
		}
		
		// Apply search filter
		if (searchQuery.trim()) {
			const query = searchQuery.toLowerCase();
			return (
				agent.name.toLowerCase().includes(query) ||
				agent.description.toLowerCase().includes(query) ||
				agent.type.toLowerCase().includes(query)
			);
		}
		
		return true;
	});
	
	// Agent actions
	async function executeAgent(agent: Agent) {
		try {
			notificationActions.add({
				type: 'info',
				title: 'Agent Execution Started',
				message: `${agent.name} is now running...`
			});
			
			const response = await apiClient.executeAgent(agent.id, {
				message: 'Test execution from dashboard'
			});
			
			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Agent Executed Successfully',
					message: `${agent.name} completed execution`
				});
			}
		} catch (error) {
			notificationActions.add({
				type: 'error',
				title: 'Agent Execution Failed',
				message: `Failed to execute ${agent.name}`
			});
		}
	}
	
	async function deleteAgent(agent: Agent) {
		if (!confirm(`Are you sure you want to delete "${agent.name}"?`)) {
			return;
		}
		
		try {
			const response = await apiClient.deleteAgent(agent.id);
			if (response.success) {
				agents.update(list => list.filter(a => a.id !== agent.id));
				notificationActions.add({
					type: 'success',
					title: 'Agent Deleted',
					message: `${agent.name} has been deleted`
				});
			}
		} catch (error) {
			notificationActions.add({
				type: 'error',
				title: 'Delete Failed',
				message: `Failed to delete ${agent.name}`
			});
		}
	}
	
	function createNewAgent() {
		agentBuilder.actions.reset();
		goto('/agents/create');
	}
	
	function editAgent(agent: Agent) {
		goto(`/agents/${agent.id}/edit`);
	}
	
	function viewAgent(agent: Agent) {
		goto(`/agents/${agent.id}`);
	}
	
	function duplicateAgent(agent: Agent) {
		// TODO: Implement agent duplication
		notificationActions.add({
			type: 'info',
			title: 'Feature Coming Soon',
			message: 'Agent duplication will be available soon'
		});
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString('en-US', {
			month: 'short',
			day: 'numeric',
			year: 'numeric'
		});
	}
	
	function getAgentTypeLabel(type: AgentType): string {
		const labels: Record<AgentType, string> = {
			'react': 'ReAct Agent',
			'knowledge_search': 'Knowledge Search',
			'rag': 'RAG Agent',
			'workflow': 'Workflow Agent',
			'multimodal': 'Multimodal Agent',
			'composite': 'Composite Agent',
			'autonomous': 'Autonomous Agent'
		};
		return labels[type] || type;
	}
</script>

<svelte:head>
	<title>Agent Builder - Agentic AI</title>
</svelte:head>

<div class="agents-container">
	<!-- Header Section -->
	<div class="agents-header">
		<div class="header-content">
			<div class="header-text">
				<h1 class="page-title">AI Agent Builder</h1>
				<p class="page-subtitle">
					Create, manage, and deploy intelligent AI agents with our revolutionary visual builder.
				</p>
			</div>
			
			<button class="create-btn" on:click={createNewAgent}>
				<Plus class="w-5 h-5" />
				<span>Create Agent</span>
			</button>
		</div>
		
		<!-- Stats Bar -->
		<div class="stats-bar">
			<div class="stat-item">
				<Bot class="w-5 h-5 text-accent-blue" />
				<span class="stat-value">{$agents?.length || 0}</span>
				<span class="stat-label">Total Agents</span>
			</div>
			<div class="stat-item">
				<Activity class="w-5 h-5 text-accent-green" />
				<span class="stat-value">{($agents || []).filter(a => a.status === 'running').length || 0}</span>
				<span class="stat-label">Running</span>
			</div>
			<div class="stat-item">
				<TrendingUp class="w-5 h-5 text-accent-purple" />
				<span class="stat-value">
					{$agents?.length > 0 ? Math.round($agents.reduce((acc, a) => acc + (a.performance_metrics?.success_rate || 0), 0) / $agents.length) : 0}%
				</span>
				<span class="stat-label">Avg Success Rate</span>
			</div>
		</div>
	</div>
	
	<!-- Controls Section -->
	<div class="controls-section">
		<!-- Search Bar -->
		<div class="search-container">
			<Search class="search-icon w-5 h-5" />
			<input
				type="text"
				placeholder="Search agents by name, description, or type..."
				class="search-input"
				bind:value={searchQuery}
			/>
		</div>
		
		<!-- Filters -->
		<div class="filters-container">
			<button 
				class="filter-toggle"
				on:click={() => showFilters = !showFilters}
			>
				<Filter class="w-4 h-4" />
				<span>Filters</span>
			</button>
			
			{#if showFilters}
				<div class="filter-dropdown">
					{#each filterOptions as option}
						<button
							class="filter-option {selectedFilter === option.value ? 'active' : ''}"
							on:click={() => selectedFilter = option.value}
						>
							<span class="filter-label">{option.label}</span>
							<span class="filter-count">{option.count}</span>
						</button>
					{/each}
				</div>
			{/if}
		</div>
		
		<!-- View Toggle -->
		<div class="view-toggle">
			<button
				class="view-btn {viewMode === 'grid' ? 'active' : ''}"
				on:click={() => viewMode = 'grid'}
			>
				Grid
			</button>
			<button
				class="view-btn {viewMode === 'list' ? 'active' : ''}"
				on:click={() => viewMode = 'list'}
			>
				List
			</button>
		</div>
	</div>
	
	<!-- Agents Grid/List -->
	{#if $agentsLoading}
		<div class="loading-state">
			<div class="loading-spinner">
				<Brain class="w-8 h-8 text-accent-blue animate-spin" />
			</div>
			<p class="loading-text">Loading your agents...</p>
		</div>
	{:else if $agentsError}
		<div class="error-state">
			<div class="error-icon">
				<Bot class="w-12 h-12 text-red-400" />
			</div>
			<h3 class="error-title">Failed to Load Agents</h3>
			<p class="error-message">{$agentsError}</p>
			<button class="retry-btn" on:click={loadAgents}>
				Try Again
			</button>
		</div>
	{:else if filteredAgents.length === 0}
		<div class="empty-state">
			{#if searchQuery.trim() || selectedFilter !== 'all'}
				<div class="empty-icon">
					<Search class="w-12 h-12 text-dark-500" />
				</div>
				<h3 class="empty-title">No agents found</h3>
				<p class="empty-message">
					Try adjusting your search or filter criteria.
				</p>
				<button class="clear-filters-btn" on:click={() => { searchQuery = ''; selectedFilter = 'all'; }}>
					Clear Filters
				</button>
			{:else}
				<div class="empty-icon">
					<Bot class="w-12 h-12 text-dark-500" />
				</div>
				<h3 class="empty-title">No agents yet</h3>
				<p class="empty-message">
					Create your first AI agent to get started with the revolutionary agent builder.
				</p>
				<button class="create-first-btn" on:click={createNewAgent}>
					<Plus class="w-5 h-5" />
					<span>Create Your First Agent</span>
				</button>
			{/if}
		</div>
	{:else}
		<div class="agents-grid {viewMode}-view">
			{#each filteredAgents as agent (agent.id)}
				<div class="agent-card">
					<!-- Card Header -->
					<div class="card-header">
						<div class="agent-info">
							<div class="agent-icon">
								<Bot class="w-6 h-6 text-accent-blue" />
							</div>
							<div class="agent-details">
								<h3 class="agent-name">{agent.name}</h3>
								<div class="agent-meta">
									<span class="agent-type {agentTypeColors[agent.type] || ''}">
										{getAgentTypeLabel(agent.type)}
									</span>
									<span class="agent-status {statusColors[agent.status]}">
										{agent.status}
									</span>
								</div>
							</div>
						</div>
						
						<!-- Actions Menu -->
						<div class="card-actions">
							<button class="action-btn primary" on:click={() => executeAgent(agent)}>
								<Play class="w-4 h-4" />
							</button>
							<div class="dropdown">
								<button class="action-btn secondary">
									<MoreVertical class="w-4 h-4" />
								</button>
								<div class="dropdown-menu">
									<button class="menu-item" on:click={() => viewAgent(agent)}>
										<Eye class="w-4 h-4" />
										<span>View Details</span>
									</button>
									<button class="menu-item" on:click={() => editAgent(agent)}>
										<Edit class="w-4 h-4" />
										<span>Edit</span>
									</button>
									<button class="menu-item" on:click={() => duplicateAgent(agent)}>
										<Copy class="w-4 h-4" />
										<span>Duplicate</span>
									</button>
									<hr class="menu-divider" />
									<button class="menu-item danger" on:click={() => deleteAgent(agent)}>
										<Trash2 class="w-4 h-4" />
										<span>Delete</span>
									</button>
								</div>
							</div>
						</div>
					</div>
					
					<!-- Card Content -->
					<div class="card-content">
						<p class="agent-description">{agent.description}</p>
						
						<!-- Performance Metrics -->
						<div class="metrics-grid">
							<div class="metric-item">
								<Zap class="w-4 h-4 text-accent-green" />
								<span class="metric-value">{agent.performance_metrics.total_executions}</span>
								<span class="metric-label">Executions</span>
							</div>
							<div class="metric-item">
								<TrendingUp class="w-4 h-4 text-accent-blue" />
								<span class="metric-value">{agent.performance_metrics.success_rate.toFixed(1)}%</span>
								<span class="metric-label">Success Rate</span>
							</div>
							<div class="metric-item">
								<Clock class="w-4 h-4 text-accent-purple" />
								<span class="metric-value">{agent.performance_metrics.average_response_time.toFixed(1)}s</span>
								<span class="metric-label">Avg Response</span>
							</div>
						</div>
					</div>
					
					<!-- Card Footer -->
					<div class="card-footer">
						<div class="footer-info">
							<span class="created-date">Created {formatDate(agent.created_at)}</span>
							<span class="model-info">Model: {agent.model}</span>
						</div>
						<div class="tool-count">
							{agent.tools.length} tools
						</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style lang="postcss">
	.agents-container {
		@apply space-y-8;
	}
	
	/* Header Section */
	.agents-header {
		@apply space-y-6;
	}
	
	.header-content {
		@apply flex items-center justify-between;
	}
	
	.header-text {
		@apply space-y-2;
	}
	
	.page-title {
		@apply text-3xl font-bold text-white;
	}
	
	.page-subtitle {
		@apply text-lg text-dark-300 max-w-2xl;
	}
	
	.create-btn {
		@apply inline-flex items-center space-x-2 px-6 py-3 bg-primary-600 text-white rounded-xl font-medium hover:bg-primary-700 hover:scale-105 transition-all duration-200;
	}
	
	.stats-bar {
		@apply flex items-center space-x-8 p-6 bg-dark-800 border border-dark-700 rounded-xl;
	}
	
	.stat-item {
		@apply flex items-center space-x-3;
	}
	
	.stat-value {
		@apply text-2xl font-bold text-white;
	}
	
	.stat-label {
		@apply text-sm text-dark-400;
	}
	
	/* Controls Section */
	.controls-section {
		@apply flex items-center justify-between space-x-4;
	}
	
	.search-container {
		@apply relative flex-1 max-w-md;
	}
	
	.search-icon {
		@apply absolute left-3 top-1/2 transform -translate-y-1/2 text-dark-400;
	}
	
	.search-input {
		@apply w-full pl-10 pr-4 py-3 bg-dark-800 border border-dark-700 rounded-xl text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent;
	}
	
	.filters-container {
		@apply relative;
	}
	
	.filter-toggle {
		@apply flex items-center space-x-2 px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl text-dark-300 hover:text-white hover:bg-dark-700 transition-all duration-200;
	}
	
	.filter-dropdown {
		@apply absolute top-full right-0 mt-2 w-48 bg-dark-800 border border-dark-700 rounded-xl shadow-xl z-10;
	}
	
	.filter-option {
		@apply flex items-center justify-between w-full px-4 py-3 text-left text-dark-300 hover:text-white hover:bg-dark-700 first:rounded-t-xl last:rounded-b-xl transition-colors duration-200;
	}
	
	.filter-option.active {
		@apply text-primary-400 bg-primary-500/10;
	}
	
	.filter-count {
		@apply text-xs bg-dark-700 px-2 py-1 rounded-full;
	}
	
	.view-toggle {
		@apply flex bg-dark-800 border border-dark-700 rounded-xl overflow-hidden;
	}
	
	.view-btn {
		@apply px-4 py-3 text-sm font-medium text-dark-300 hover:text-white transition-colors duration-200;
	}
	
	.view-btn.active {
		@apply text-white bg-primary-600;
	}
	
	/* Loading/Error/Empty States */
	.loading-state,
	.error-state,
	.empty-state {
		@apply flex flex-col items-center justify-center py-16 text-center;
	}
	
	.loading-spinner,
	.error-icon,
	.empty-icon {
		@apply mb-4;
	}
	
	.loading-text,
	.error-title,
	.empty-title {
		@apply text-xl font-semibold text-white mb-2;
	}
	
	.error-message,
	.empty-message {
		@apply text-dark-400 mb-6 max-w-md;
	}
	
	.retry-btn,
	.clear-filters-btn,
	.create-first-btn {
		@apply inline-flex items-center space-x-2 px-6 py-3 bg-primary-600 text-white rounded-xl font-medium hover:bg-primary-700 transition-colors duration-200;
	}
	
	/* Agents Grid */
	.agents-grid {
		display: grid;
		gap: 1.5rem; /* gap-6 */
	}

	.agents-grid.grid-view {
		grid-template-columns: 1fr;
	}

	@media (min-width: 768px) {
		.agents-grid.grid-view {
			grid-template-columns: repeat(2, 1fr);
		}
	}

	@media (min-width: 1024px) {
		.agents-grid.grid-view {
			grid-template-columns: repeat(3, 1fr);
		}
	}

	.agents-grid.list-view {
		grid-template-columns: 1fr;
	}
	
	.agent-card {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6 hover:border-dark-600 hover:bg-dark-700/50 transition-all duration-200;
	}
	
	.card-header {
		@apply flex items-start justify-between mb-4;
	}
	
	.agent-info {
		@apply flex items-start space-x-3 flex-1;
	}
	
	.agent-icon {
		@apply w-12 h-12 bg-dark-700 rounded-xl flex items-center justify-center;
	}
	
	.agent-details {
		@apply space-y-2;
	}
	
	.agent-name {
		@apply text-lg font-semibold text-white;
	}
	
	.agent-meta {
		@apply flex items-center space-x-2;
	}
	
	.agent-type,
	.agent-status {
		@apply px-2 py-1 text-xs font-medium rounded-lg border;
	}
	
	.card-actions {
		@apply flex items-center space-x-2;
	}
	
	.action-btn {
		@apply p-2 rounded-lg transition-all duration-200;
	}
	
	.action-btn.primary {
		@apply bg-primary-600 text-white hover:bg-primary-700;
	}
	
	.action-btn.secondary {
		@apply bg-dark-700 text-dark-300 hover:bg-dark-600 hover:text-white;
	}
	
	.dropdown {
		@apply relative;
	}
	
	.dropdown-menu {
		@apply absolute top-full right-0 mt-2 w-48 bg-dark-800 border border-dark-700 rounded-xl shadow-xl z-10 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200;
	}
	
	.menu-item {
		@apply flex items-center space-x-3 w-full px-4 py-3 text-left text-dark-300 hover:text-white hover:bg-dark-700 first:rounded-t-xl last:rounded-b-xl transition-colors duration-200;
	}
	
	.menu-item.danger {
		@apply text-red-400 hover:text-red-300 hover:bg-red-500/10;
	}
	
	.menu-divider {
		@apply border-dark-700;
	}
	
	.card-content {
		@apply space-y-4 mb-4;
	}
	
	.agent-description {
		@apply text-dark-300 text-sm line-clamp-2;
	}
	
	.metrics-grid {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 1rem; /* gap-4 */
	}
	
	.metric-item {
		@apply flex flex-col items-center space-y-1 p-3 bg-dark-700/50 rounded-lg;
	}
	
	.metric-value {
		@apply text-sm font-semibold text-white;
	}
	
	.metric-label {
		@apply text-xs text-dark-400;
	}
	
	.card-footer {
		@apply flex items-center justify-between pt-4 border-t border-dark-700;
	}
	
	.footer-info {
		@apply space-y-1;
	}
	
	.created-date,
	.model-info {
		@apply block text-xs text-dark-400;
	}
	
	.tool-count {
		@apply text-xs text-dark-400 bg-dark-700 px-2 py-1 rounded-lg;
	}
	
	/* Mobile Responsiveness */
	@media (max-width: 768px) {
		.header-content {
			@apply flex-col items-start space-y-4;
		}
		
		.stats-bar {
			@apply flex-col space-y-4 space-x-0;
		}
		
		.controls-section {
			@apply flex-col space-y-4 space-x-0;
		}
		
		.search-container {
			@apply max-w-none;
		}
	}
</style>
