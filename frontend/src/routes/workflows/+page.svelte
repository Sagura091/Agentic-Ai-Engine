<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { 
		workflows, 
		workflowsLoading, 
		workflowsError,
		notificationActions 
	} from '$stores';
	import { apiClient } from '$services/api';
	import { websocketService } from '$services/websocket';
	import { 
		Workflow, 
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
		Network,
		Activity,
		Clock,
		TrendingUp,
		Users,
		Zap
	} from 'lucide-svelte';
	import type { Workflow as WorkflowType, WorkflowStatus } from '$types';
	
	// State
	let searchQuery = '';
	let selectedFilter = 'all';
	let viewMode = 'grid'; // 'grid' | 'list'
	let showFilters = false;
	
	// Filters
	const filterOptions = [
		{ value: 'all', label: 'All Workflows', count: 0 },
		{ value: 'active', label: 'Active', count: 0 },
		{ value: 'draft', label: 'Draft', count: 0 },
		{ value: 'paused', label: 'Paused', count: 0 },
		{ value: 'error', label: 'Error', count: 0 }
	];
	
	// Status colors
	const statusColors: Record<WorkflowStatus, string> = {
		'draft': 'bg-gray-500/20 text-gray-400',
		'active': 'bg-green-500/20 text-green-400',
		'paused': 'bg-yellow-500/20 text-yellow-400',
		'completed': 'bg-blue-500/20 text-blue-400',
		'error': 'bg-red-500/20 text-red-400'
	};
	
	onMount(async () => {
		await loadWorkflows();
		
		// Subscribe to real-time workflow updates
		websocketService.on('workflow_status_update' as any, (data) => {
			updateWorkflowStatus(data.workflow_id, data.status);
		});
	});
	
	async function loadWorkflows() {
		workflowsLoading.set(true);
		workflowsError.set(null);
		
		try {
			const response = await apiClient.getWorkflows();
			if (response.success && response.data) {
				workflows.set(response.data);
				updateFilterCounts(response.data);
			} else {
				throw new Error(response.error || 'Failed to load workflows');
			}
		} catch (error) {
			console.error('Failed to load workflows:', error);
			workflowsError.set(error instanceof Error ? error.message : 'Unknown error');
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Workflows',
				message: 'Unable to load workflows. Please try again.'
			});
		} finally {
			workflowsLoading.set(false);
		}
	}
	
	function updateFilterCounts(workflowList: WorkflowType[]) {
		filterOptions[0].count = workflowList.length;
		filterOptions[1].count = workflowList.filter(w => w.status === 'active').length;
		filterOptions[2].count = workflowList.filter(w => w.status === 'draft').length;
		filterOptions[3].count = workflowList.filter(w => w.status === 'paused').length;
		filterOptions[4].count = workflowList.filter(w => w.status === 'error').length;
	}
	
	function updateWorkflowStatus(workflowId: string, status: WorkflowStatus) {
		workflows.update(list => 
			list.map(workflow => 
				workflow.id === workflowId ? { ...workflow, status } : workflow
			)
		);
	}
	
	// Filter and search workflows
	$: filteredWorkflows = $workflows.filter(workflow => {
		// Apply status filter
		if (selectedFilter !== 'all' && workflow.status !== selectedFilter) {
			return false;
		}
		
		// Apply search filter
		if (searchQuery.trim()) {
			const query = searchQuery.toLowerCase();
			return (
				workflow.name.toLowerCase().includes(query) ||
				workflow.description.toLowerCase().includes(query)
			);
		}
		
		return true;
	});
	
	// Workflow actions
	async function executeWorkflow(workflow: WorkflowType) {
		try {
			notificationActions.add({
				type: 'info',
				title: 'Workflow Execution Started',
				message: `${workflow.name} is now running...`
			});
			
			const response = await apiClient.executeWorkflow(workflow.id, {
				message: 'Test execution from dashboard'
			});
			
			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Workflow Executed Successfully',
					message: `${workflow.name} completed execution`
				});
			}
		} catch (error) {
			notificationActions.add({
				type: 'error',
				title: 'Workflow Execution Failed',
				message: `Failed to execute ${workflow.name}`
			});
		}
	}
	
	async function deleteWorkflow(workflow: WorkflowType) {
		if (!confirm(`Are you sure you want to delete "${workflow.name}"?`)) {
			return;
		}
		
		try {
			const response = await apiClient.deleteWorkflow(workflow.id);
			if (response.success) {
				workflows.update(list => list.filter(w => w.id !== workflow.id));
				notificationActions.add({
					type: 'success',
					title: 'Workflow Deleted',
					message: `${workflow.name} has been deleted`
				});
			}
		} catch (error) {
			notificationActions.add({
				type: 'error',
				title: 'Delete Failed',
				message: `Failed to delete ${workflow.name}`
			});
		}
	}
	
	function createNewWorkflow() {
		goto('/workflows/create');
	}
	
	function editWorkflow(workflow: WorkflowType) {
		goto(`/workflows/${workflow.id}/edit`);
	}
	
	function viewWorkflow(workflow: WorkflowType) {
		goto(`/workflows/${workflow.id}`);
	}
	
	function duplicateWorkflow(workflow: WorkflowType) {
		// TODO: Implement workflow duplication
		notificationActions.add({
			type: 'info',
			title: 'Feature Coming Soon',
			message: 'Workflow duplication will be available soon'
		});
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString('en-US', {
			month: 'short',
			day: 'numeric',
			year: 'numeric'
		});
	}
	
	function getStatusLabel(status: WorkflowStatus): string {
		const labels: Record<WorkflowStatus, string> = {
			'draft': 'Draft',
			'active': 'Active',
			'paused': 'Paused',
			'completed': 'Completed',
			'error': 'Error'
		};
		return labels[status] || status;
	}
</script>

<svelte:head>
	<title>Workflow Builder - Agentic AI</title>
</svelte:head>

<div class="workflows-container">
	<!-- Header Section -->
	<div class="workflows-header">
		<div class="header-content">
			<div class="header-text">
				<h1 class="page-title">Workflow Builder</h1>
				<p class="page-subtitle">
					Design and orchestrate complex multi-agent workflows with our revolutionary visual builder.
				</p>
			</div>
			
			<button class="create-btn" on:click={createNewWorkflow}>
				<Plus class="w-5 h-5" />
				<span>Create Workflow</span>
			</button>
		</div>
		
		<!-- Stats Bar -->
		<div class="stats-bar">
			<div class="stat-item">
				<Workflow class="w-5 h-5 text-accent-blue" />
				<span class="stat-value">{$workflows.length}</span>
				<span class="stat-label">Total Workflows</span>
			</div>
			<div class="stat-item">
				<Activity class="w-5 h-5 text-accent-green" />
				<span class="stat-value">{$workflows.filter(w => w.status === 'active').length}</span>
				<span class="stat-label">Active</span>
			</div>
			<div class="stat-item">
				<Users class="w-5 h-5 text-accent-purple" />
				<span class="stat-value">
					{$workflows.reduce((acc, w) => acc + w.agents.length, 0)}
				</span>
				<span class="stat-label">Total Agents</span>
			</div>
			<div class="stat-item">
				<Network class="w-5 h-5 text-accent-orange" />
				<span class="stat-value">
					{$workflows.reduce((acc, w) => acc + w.connections.length, 0)}
				</span>
				<span class="stat-label">Connections</span>
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
				placeholder="Search workflows by name or description..."
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
	
	<!-- Workflows Grid/List -->
	{#if $workflowsLoading}
		<div class="loading-state">
			<div class="loading-spinner">
				<Network class="w-8 h-8 text-accent-blue animate-spin" />
			</div>
			<p class="loading-text">Loading your workflows...</p>
		</div>
	{:else if $workflowsError}
		<div class="error-state">
			<div class="error-icon">
				<Workflow class="w-12 h-12 text-red-400" />
			</div>
			<h3 class="error-title">Failed to Load Workflows</h3>
			<p class="error-message">{$workflowsError}</p>
			<button class="retry-btn" on:click={loadWorkflows}>
				Try Again
			</button>
		</div>
	{:else if filteredWorkflows.length === 0}
		<div class="empty-state">
			{#if searchQuery.trim() || selectedFilter !== 'all'}
				<div class="empty-icon">
					<Search class="w-12 h-12 text-dark-500" />
				</div>
				<h3 class="empty-title">No workflows found</h3>
				<p class="empty-message">
					Try adjusting your search or filter criteria.
				</p>
				<button class="clear-filters-btn" on:click={() => { searchQuery = ''; selectedFilter = 'all'; }}>
					Clear Filters
				</button>
			{:else}
				<div class="empty-icon">
					<Workflow class="w-12 h-12 text-dark-500" />
				</div>
				<h3 class="empty-title">No workflows yet</h3>
				<p class="empty-message">
					Create your first multi-agent workflow to get started with advanced automation.
				</p>
				<button class="create-first-btn" on:click={createNewWorkflow}>
					<Plus class="w-5 h-5" />
					<span>Create Your First Workflow</span>
				</button>
			{/if}
		</div>
	{:else}
		<div class="workflows-grid {viewMode}-view">
			{#each filteredWorkflows as workflow (workflow.id)}
				<div class="workflow-card">
					<!-- Card Header -->
					<div class="card-header">
						<div class="workflow-info">
							<div class="workflow-icon">
								<Network class="w-6 h-6 text-accent-purple" />
							</div>
							<div class="workflow-details">
								<h3 class="workflow-name">{workflow.name}</h3>
								<div class="workflow-meta">
									<span class="workflow-status {statusColors[workflow.status]}">
										{getStatusLabel(workflow.status)}
									</span>
								</div>
							</div>
						</div>
						
						<!-- Actions Menu -->
						<div class="card-actions">
							<button class="action-btn primary" on:click={() => executeWorkflow(workflow)}>
								<Play class="w-4 h-4" />
							</button>
							<div class="dropdown">
								<button class="action-btn secondary">
									<MoreVertical class="w-4 h-4" />
								</button>
								<div class="dropdown-menu">
									<button class="menu-item" on:click={() => viewWorkflow(workflow)}>
										<Eye class="w-4 h-4" />
										<span>View Details</span>
									</button>
									<button class="menu-item" on:click={() => editWorkflow(workflow)}>
										<Edit class="w-4 h-4" />
										<span>Edit</span>
									</button>
									<button class="menu-item" on:click={() => duplicateWorkflow(workflow)}>
										<Copy class="w-4 h-4" />
										<span>Duplicate</span>
									</button>
									<hr class="menu-divider" />
									<button class="menu-item danger" on:click={() => deleteWorkflow(workflow)}>
										<Trash2 class="w-4 h-4" />
										<span>Delete</span>
									</button>
								</div>
							</div>
						</div>
					</div>
					
					<!-- Card Content -->
					<div class="card-content">
						<p class="workflow-description">{workflow.description}</p>
						
						<!-- Workflow Visualization -->
						<div class="workflow-preview">
							<div class="preview-header">
								<span class="preview-title">Workflow Structure</span>
							</div>
							<div class="preview-content">
								<!-- Simple node visualization -->
								<div class="node-chain">
									{#each workflow.agents.slice(0, 4) as agent, index}
										<div class="preview-node">
											<div class="node-icon">
												<Zap class="w-3 h-3" />
											</div>
										</div>
										{#if index < Math.min(workflow.agents.length - 1, 3)}
											<div class="node-connector"></div>
										{/if}
									{/each}
									{#if workflow.agents.length > 4}
										<div class="more-nodes">+{workflow.agents.length - 4}</div>
									{/if}
								</div>
							</div>
						</div>
						
						<!-- Metrics Grid -->
						<div class="metrics-grid">
							<div class="metric-item">
								<Users class="w-4 h-4 text-accent-blue" />
								<span class="metric-value">{workflow.agents.length}</span>
								<span class="metric-label">Agents</span>
							</div>
							<div class="metric-item">
								<Network class="w-4 h-4 text-accent-green" />
								<span class="metric-value">{workflow.connections.length}</span>
								<span class="metric-label">Connections</span>
							</div>
							<div class="metric-item">
								<Activity class="w-4 h-4 text-accent-purple" />
								<span class="metric-value">{workflow.status === 'active' ? 'Running' : 'Idle'}</span>
								<span class="metric-label">Status</span>
							</div>
						</div>
					</div>
					
					<!-- Card Footer -->
					<div class="card-footer">
						<div class="footer-info">
							<span class="created-date">Created {formatDate(workflow.created_at)}</span>
							<span class="updated-date">Updated {formatDate(workflow.updated_at)}</span>
						</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style lang="postcss">
	.workflows-container {
		@apply space-y-8;
	}
	
	/* Header Section */
	.workflows-header {
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
	
	/* Workflows Grid */
	.workflows-grid {
		display: grid;
		gap: 1.5rem; /* gap-6 */
	}

	.workflows-grid.grid-view {
		grid-template-columns: 1fr;
	}

	@media (min-width: 768px) {
		.workflows-grid.grid-view {
			grid-template-columns: repeat(2, 1fr);
		}
	}

	@media (min-width: 1024px) {
		.workflows-grid.grid-view {
			grid-template-columns: repeat(3, 1fr);
		}
	}

	.workflows-grid.list-view {
		grid-template-columns: 1fr;
	}
	
	.workflow-card {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6 hover:border-dark-600 hover:bg-dark-700/50 transition-all duration-200;
	}
	
	.card-header {
		@apply flex items-start justify-between mb-4;
	}
	
	.workflow-info {
		@apply flex items-start space-x-3 flex-1;
	}
	
	.workflow-icon {
		@apply w-12 h-12 bg-dark-700 rounded-xl flex items-center justify-center;
	}
	
	.workflow-details {
		@apply space-y-2;
	}
	
	.workflow-name {
		@apply text-lg font-semibold text-white;
	}
	
	.workflow-meta {
		@apply flex items-center space-x-2;
	}
	
	.workflow-status {
		@apply px-2 py-1 text-xs font-medium rounded-lg;
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
	
	.workflow-description {
		@apply text-dark-300 text-sm line-clamp-2;
	}
	
	/* Workflow Preview */
	.workflow-preview {
		@apply bg-dark-700/50 rounded-lg p-4;
	}
	
	.preview-header {
		@apply mb-3;
	}
	
	.preview-title {
		@apply text-xs font-medium text-dark-400 uppercase tracking-wide;
	}
	
	.preview-content {
		@apply flex justify-center;
	}
	
	.node-chain {
		@apply flex items-center space-x-2;
	}
	
	.preview-node {
		@apply w-8 h-8 bg-dark-600 rounded-lg flex items-center justify-center;
	}
	
	.node-icon {
		@apply text-accent-blue;
	}
	
	.node-connector {
		@apply w-4 h-0.5 bg-dark-600;
	}
	
	.more-nodes {
		@apply text-xs text-dark-400 bg-dark-600 px-2 py-1 rounded;
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
		@apply pt-4 border-t border-dark-700;
	}
	
	.footer-info {
		@apply space-y-1;
	}
	
	.created-date,
	.updated-date {
		@apply block text-xs text-dark-400;
	}
	
	/* Mobile Responsiveness */
	@media (max-width: 768px) {
		.header-content {
			@apply flex-col items-start space-y-4;
		}
		
		.stats-bar {
			display: grid;
			grid-template-columns: repeat(2, 1fr);
			gap: 1rem; /* gap-4 */
		}
		
		.controls-section {
			@apply flex-col space-y-4 space-x-0;
		}
		
		.search-container {
			@apply max-w-none;
		}
	}
</style>
