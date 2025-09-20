<!--
Revolutionary Tool Marketplace
Complete tool ecosystem with marketplace, custom builder, and management
-->

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { 
		tools, 
		toolsLoading, 
		toolsError,
		notificationActions 
	} from '$lib/stores';
	import { toolService } from '$lib/services/toolService';
	import { websocketService } from '$lib/services/websocket';
	import { 
		Wrench, 
		Search, 
		Filter, 
		Grid, 
		List, 
		Settings, 
		Trash2,
		Plus,
		RefreshCw,
		Download,
		Upload,
		Star,
		TrendingUp,
		Code,
		Zap,
		Eye,
		Share,
		Lock,
		Unlock,
		CheckCircle,
		AlertCircle,
		Loader,
		Package,
		Users,
		Clock,
		Award,
		ShoppingCart,
		Heart,
		ExternalLink
	} from 'lucide-svelte';
	import type { Tool, ToolCategory, ToolStatus } from '$lib/types';
	import ToolCard from '$lib/components/tools/ToolCard.svelte';
	import ToolMarketplace from '$lib/components/tools/ToolMarketplace.svelte';
	import CustomToolBuilder from '$lib/components/tools/CustomToolBuilder.svelte';
	import ToolTester from '$lib/components/tools/ToolTester.svelte';

	// State
	let activeTab: 'installed' | 'marketplace' | 'custom' | 'analytics' = 'installed';
	let searchQuery = '';
	let selectedCategory: ToolCategory | 'all' = 'all';
	let selectedStatus: ToolStatus | 'all' = 'all';
	let viewMode: 'grid' | 'list' = 'grid';
	let showMarketplace = false;
	let showCustomBuilder = false;
	let showToolTester = false;
	let selectedTool: Tool | null = null;
	let sortBy = 'name';
	let sortOrder: 'asc' | 'desc' = 'asc';

	// Filters
	const categoryOptions = [
		{ value: 'all', label: 'All Categories' },
		{ value: 'computation', label: 'Computation' },
		{ value: 'communication', label: 'Communication' },
		{ value: 'research', label: 'Research' },
		{ value: 'business', label: 'Business' },
		{ value: 'utility', label: 'Utility' },
		{ value: 'custom', label: 'Custom' }
	];

	const statusOptions = [
		{ value: 'all', label: 'All Status' },
		{ value: 'installed', label: 'Installed' },
		{ value: 'available', label: 'Available' },
		{ value: 'updating', label: 'Updating' },
		{ value: 'error', label: 'Error' }
	];

	const sortOptions = [
		{ value: 'name', label: 'Name' },
		{ value: 'rating', label: 'Rating' },
		{ value: 'downloads', label: 'Downloads' },
		{ value: 'updated', label: 'Last Updated' },
		{ value: 'created', label: 'Created' }
	];

	// Analytics data
	let analyticsData = {
		totalTools: 0,
		installedTools: 0,
		customTools: 0,
		toolUsage: 0,
		avgRating: 0,
		topCategories: [] as Array<{ category: string; count: number }>
	};

	// Real-time updates
	let unsubscribeWebSocket: (() => void) | null = null;

	onMount(async () => {
		await loadData();
		setupRealTimeUpdates();
	});

	onDestroy(() => {
		if (unsubscribeWebSocket) {
			unsubscribeWebSocket();
		}
	});

	async function loadData() {
		toolsLoading.set(true);
		toolsError.set(null);

		try {
			// Load installed tools
			const toolsResponse = await toolService.getInstalledTools();
			if (toolsResponse.success && toolsResponse.data) {
				tools.set(toolsResponse.data);
			}

			// Load analytics
			await loadAnalytics();

		} catch (error) {
			console.error('Failed to load tools:', error);
			toolsError.set(error instanceof Error ? error.message : 'Unknown error');
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Tools',
				message: 'Unable to load tools. Please try again.'
			});
		} finally {
			toolsLoading.set(false);
		}
	}

	async function loadAnalytics() {
		try {
			const response = await toolService.getToolAnalytics();
			if (response.success && response.data) {
				analyticsData = response.data;
			}
		} catch (error) {
			console.error('Failed to load analytics:', error);
		}
	}

	function setupRealTimeUpdates() {
		// Subscribe to tool installation updates
		unsubscribeWebSocket = websocketService.on('tool_installation_update' as any, (data) => {
			updateToolStatus(data.tool_id, data.status, data.progress);
		});

		// Subscribe to tool usage updates
		websocketService.on('tool_usage_update' as any, (data) => {
			analyticsData.toolUsage = data.total_usage;
		});
	}

	function updateToolStatus(toolId: string, status: ToolStatus, progress?: number) {
		tools.update(toolList => 
			toolList.map(tool => 
				tool.id === toolId 
					? { ...tool, status, installProgress: progress }
					: tool
			)
		);
	}

	async function handleInstallTool(tool: Tool) {
		try {
			const result = await toolService.installTool(tool.id);
			if (result.success) {
				notificationActions.add({
					type: 'success',
					title: 'Installation Started',
					message: `${tool.name} installation has begun.`
				});
				
				updateToolStatus(tool.id, 'installing');
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Install error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Installation Failed',
				message: `Failed to install ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	async function handleUninstallTool(tool: Tool) {
		if (!confirm(`Are you sure you want to uninstall ${tool.name}?`)) {
			return;
		}

		try {
			const result = await toolService.uninstallTool(tool.id);
			if (result.success) {
				notificationActions.add({
					type: 'success',
					title: 'Tool Uninstalled',
					message: `${tool.name} has been uninstalled.`
				});
				
				updateToolStatus(tool.id, 'available');
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Uninstall error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Uninstall Failed',
				message: `Failed to uninstall ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	async function handleUpdateTool(tool: Tool) {
		try {
			const result = await toolService.updateTool(tool.id);
			if (result.success) {
				notificationActions.add({
					type: 'success',
					title: 'Update Started',
					message: `${tool.name} update has begun.`
				});
				
				updateToolStatus(tool.id, 'updating');
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Update error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Update Failed',
				message: `Failed to update ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	// Computed properties
	$: filteredTools = $tools.filter(tool => {
		// Search filter
		if (searchQuery && !tool.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
			!tool.description.toLowerCase().includes(searchQuery.toLowerCase()) &&
			!tool.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))) {
			return false;
		}

		// Category filter
		if (selectedCategory !== 'all' && tool.category !== selectedCategory) {
			return false;
		}

		// Status filter
		if (selectedStatus !== 'all' && tool.status !== selectedStatus) {
			return false;
		}

		return true;
	}).sort((a, b) => {
		let aVal: any, bVal: any;

		switch (sortBy) {
			case 'name':
				aVal = a.name.toLowerCase();
				bVal = b.name.toLowerCase();
				break;
			case 'rating':
				aVal = a.rating || 0;
				bVal = b.rating || 0;
				break;
			case 'downloads':
				aVal = a.downloads || 0;
				bVal = b.downloads || 0;
				break;
			case 'updated':
				aVal = new Date(a.updatedAt || 0).getTime();
				bVal = new Date(b.updatedAt || 0).getTime();
				break;
			case 'created':
				aVal = new Date(a.createdAt || 0).getTime();
				bVal = new Date(b.createdAt || 0).getTime();
				break;
			default:
				return 0;
		}

		if (sortOrder === 'asc') {
			return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
		} else {
			return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
		}
	});

	function getCategoryIcon(category: ToolCategory): any {
		switch (category) {
			case 'computation': return Zap;
			case 'communication': return Users;
			case 'research': return Search;
			case 'business': return TrendingUp;
			case 'utility': return Settings;
			case 'custom': return Code;
			default: return Wrench;
		}
	}

	function getStatusColor(status: ToolStatus): string {
		switch (status) {
			case 'installed': return 'text-green-400';
			case 'available': return 'text-blue-400';
			case 'installing': return 'text-yellow-400';
			case 'updating': return 'text-purple-400';
			case 'error': return 'text-red-400';
			default: return 'text-dark-400';
		}
	}
</script>

<svelte:head>
	<title>Tool Marketplace - Agentic AI Platform</title>
	<meta name="description" content="Browse, install, and manage AI tools and extensions" />
</svelte:head>

<div class="tools-page p-6 space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold text-white">Tool Marketplace</h1>
			<p class="text-dark-300 mt-1">Browse, install, and manage AI tools and extensions</p>
		</div>
		<div class="flex items-center space-x-3">
			<button
				on:click={() => showCustomBuilder = true}
				class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
			>
				<Code class="w-4 h-4" />
				<span>Create Tool</span>
			</button>
			<button
				on:click={() => showMarketplace = true}
				class="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
			>
				<ShoppingCart class="w-4 h-4" />
				<span>Browse Marketplace</span>
			</button>
			<button
				on:click={loadData}
				disabled={$toolsLoading}
				class="bg-dark-700 hover:bg-dark-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
			>
				<RefreshCw class="w-4 h-4 {$toolsLoading ? 'animate-spin' : ''}" />
				<span>Refresh</span>
			</button>
		</div>
	</div>

	<!-- Analytics Cards -->
	<div class="grid grid-cols-1 md:grid-cols-6 gap-4">
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Total Tools</p>
					<p class="text-2xl font-bold text-white">{analyticsData.totalTools}</p>
				</div>
				<Package class="w-8 h-8 text-blue-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Installed</p>
					<p class="text-2xl font-bold text-white">{analyticsData.installedTools}</p>
				</div>
				<CheckCircle class="w-8 h-8 text-green-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Custom</p>
					<p class="text-2xl font-bold text-white">{analyticsData.customTools}</p>
				</div>
				<Code class="w-8 h-8 text-purple-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Usage</p>
					<p class="text-2xl font-bold text-white">{analyticsData.toolUsage}</p>
				</div>
				<TrendingUp class="w-8 h-8 text-orange-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Avg Rating</p>
					<p class="text-2xl font-bold text-white">{analyticsData.avgRating.toFixed(1)}</p>
				</div>
				<Star class="w-8 h-8 text-yellow-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Categories</p>
					<p class="text-2xl font-bold text-white">{analyticsData.topCategories.length}</p>
				</div>
				<Award class="w-8 h-8 text-pink-400" />
			</div>
		</div>
	</div>

	<!-- Navigation Tabs -->
	<div class="bg-dark-800 border border-dark-700 rounded-lg">
		<div class="flex border-b border-dark-700">
			<button
				on:click={() => activeTab = 'installed'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'installed' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<CheckCircle class="w-4 h-4" />
					<span>Installed Tools</span>
				</div>
			</button>
			<button
				on:click={() => activeTab = 'marketplace'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'marketplace' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<ShoppingCart class="w-4 h-4" />
					<span>Marketplace</span>
				</div>
			</button>
			<button
				on:click={() => activeTab = 'custom'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'custom' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<Code class="w-4 h-4" />
					<span>Custom Tools</span>
				</div>
			</button>
			<button
				on:click={() => activeTab = 'analytics'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'analytics' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<TrendingUp class="w-4 h-4" />
					<span>Analytics</span>
				</div>
			</button>
		</div>

		<!-- Tab Content -->
		<div class="p-6">
			<!-- Installed Tools Tab -->
			{#if activeTab === 'installed'}
				<div class="space-y-4">
					<!-- Filters and Search -->
					<div class="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
						<!-- Search -->
						<div class="relative flex-1 max-w-md">
							<Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
							<input
								type="text"
								bind:value={searchQuery}
								placeholder="Search tools..."
								class="w-full pl-10 pr-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
							/>
						</div>

						<!-- Filters -->
						<div class="flex items-center space-x-4">
							<!-- Category Filter -->
							<select
								bind:value={selectedCategory}
								class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
							>
								{#each categoryOptions as option}
									<option value={option.value}>{option.label}</option>
								{/each}
							</select>

							<!-- Status Filter -->
							<select
								bind:value={selectedStatus}
								class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
							>
								{#each statusOptions as option}
									<option value={option.value}>{option.label}</option>
								{/each}
							</select>

							<!-- Sort -->
							<select
								bind:value={sortBy}
								class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
							>
								{#each sortOptions as option}
									<option value={option.value}>{option.label}</option>
								{/each}
							</select>

							<!-- Sort Order -->
							<button
								on:click={() => sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'}
								class="bg-dark-700 hover:bg-dark-600 border border-dark-600 rounded-lg px-3 py-2 text-white transition-colors"
								title="Toggle sort order"
							>
								{sortOrder === 'asc' ? '↑' : '↓'}
							</button>

							<!-- View Mode -->
							<div class="flex items-center bg-dark-700 border border-dark-600 rounded-lg">
								<button
									on:click={() => viewMode = 'grid'}
									class="px-3 py-2 text-white transition-colors {viewMode === 'grid' ? 'bg-primary-600' : 'hover:bg-dark-600'}"
								>
									<Grid class="w-4 h-4" />
								</button>
								<button
									on:click={() => viewMode = 'list'}
									class="px-3 py-2 text-white transition-colors {viewMode === 'list' ? 'bg-primary-600' : 'hover:bg-dark-600'}"
								>
									<List class="w-4 h-4" />
								</button>
							</div>
						</div>
					</div>

					<!-- Tools Grid/List -->
					{#if $toolsLoading}
						<div class="flex items-center justify-center py-12">
							<div class="text-center">
								<Loader class="w-8 h-8 text-primary-500 animate-spin mx-auto mb-4" />
								<p class="text-dark-300">Loading tools...</p>
							</div>
						</div>
					{:else if filteredTools.length === 0}
						<div class="text-center py-12">
							<Wrench class="w-12 h-12 text-dark-400 mx-auto mb-4" />
							<h3 class="text-lg font-semibold text-white mb-2">No Tools Found</h3>
							<p class="text-dark-300 mb-4">
								{searchQuery || selectedCategory !== 'all' || selectedStatus !== 'all' 
									? 'No tools match your current filters.' 
									: 'No tools installed. Browse the marketplace to install tools.'}
							</p>
							<button
								on:click={() => showMarketplace = true}
								class="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
							>
								Browse Marketplace
							</button>
						</div>
					{:else}
						<div class="tools-container {viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}">
							{#each filteredTools as tool (tool.id)}
								<ToolCard
									{tool}
									{viewMode}
									on:install={() => handleInstallTool(tool)}
									on:uninstall={() => handleUninstallTool(tool)}
									on:update={() => handleUpdateTool(tool)}
									on:test={() => { selectedTool = tool; showToolTester = true; }}
									on:configure={() => selectedTool = tool}
									on:view={() => selectedTool = tool}
								/>
							{/each}
						</div>
					{/if}
				</div>
			{/if}

			<!-- Marketplace Tab -->
			{#if activeTab === 'marketplace'}
				<ToolMarketplace
					on:install={(event) => handleInstallTool(event.detail)}
				/>
			{/if}

			<!-- Custom Tools Tab -->
			{#if activeTab === 'custom'}
				<div class="space-y-4">
					<div class="flex items-center justify-between">
						<h3 class="text-lg font-semibold text-white">Custom Tools</h3>
						<button
							on:click={() => showCustomBuilder = true}
							class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
						>
							<Plus class="w-4 h-4" />
							<span>Create Custom Tool</span>
						</button>
					</div>

					<!-- Custom tools list would go here -->
					<div class="text-center py-12">
						<Code class="w-12 h-12 text-dark-400 mx-auto mb-4" />
						<h3 class="text-lg font-semibold text-white mb-2">No Custom Tools</h3>
						<p class="text-dark-300 mb-4">Create your first custom tool to extend the platform.</p>
						<button
							on:click={() => showCustomBuilder = true}
							class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors"
						>
							Create Tool
						</button>
					</div>
				</div>
			{/if}

			<!-- Analytics Tab -->
			{#if activeTab === 'analytics'}
				<div class="space-y-6">
					<h3 class="text-lg font-semibold text-white">Tool Analytics</h3>
					
					<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
						<div class="bg-dark-700 border border-dark-600 rounded-lg p-6">
							<h4 class="text-white font-medium mb-4">Top Categories</h4>
							<div class="space-y-3">
								{#each analyticsData.topCategories as category}
									<div class="flex items-center justify-between">
										<div class="flex items-center space-x-2">
											<svelte:component this={getCategoryIcon(category.category)} class="w-4 h-4 text-primary-400" />
											<span class="text-dark-300 capitalize">{category.category}</span>
										</div>
										<span class="text-white">{category.count}</span>
									</div>
								{/each}
							</div>
						</div>
						
						<div class="bg-dark-700 border border-dark-600 rounded-lg p-6">
							<h4 class="text-white font-medium mb-4">Usage Statistics</h4>
							<div class="space-y-3">
								<div class="flex justify-between">
									<span class="text-dark-300">Total Usage</span>
									<span class="text-white">{analyticsData.toolUsage}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-dark-300">Average Rating</span>
									<span class="text-white">{analyticsData.avgRating.toFixed(1)}/5.0</span>
								</div>
							</div>
						</div>
					</div>
				</div>
			{/if}
		</div>
	</div>
</div>

<!-- Tool Marketplace Modal -->
{#if showMarketplace}
	<ToolMarketplace
		on:close={() => showMarketplace = false}
		on:install={(event) => {
			handleInstallTool(event.detail);
			showMarketplace = false;
		}}
	/>
{/if}

<!-- Custom Tool Builder Modal -->
{#if showCustomBuilder}
	<CustomToolBuilder
		on:close={() => showCustomBuilder = false}
		on:create={(event) => {
			loadData();
			showCustomBuilder = false;
		}}
	/>
{/if}

<!-- Tool Tester Modal -->
{#if showToolTester && selectedTool}
	<ToolTester
		tool={selectedTool}
		on:close={() => { showToolTester = false; selectedTool = null; }}
	/>
{/if}

<style lang="postcss">
	.tools-page {
		animation: fadeIn 0.6s ease-out;
	}

	@keyframes fadeIn {
		from {
			opacity: 0;
			transform: translateY(20px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.tools-container {
		animation: slideUp 0.4s ease-out;
	}

	@keyframes slideUp {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
</style>
