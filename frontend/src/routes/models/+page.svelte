<!--
Revolutionary Model Management Interface
Complete model marketplace, download manager, and configuration system
-->

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { 
		models, 
		modelsLoading, 
		modelsError,
		notificationActions 
	} from '$lib/stores';
	import { modelService } from '$lib/services/modelService';
	import { websocketService } from '$lib/services/websocket';
	import { 
		Download, 
		Search, 
		Filter, 
		Grid, 
		List, 
		Settings, 
		Trash2,
		Play,
		Pause,
		RefreshCw,
		HardDrive,
		Cpu,
		Zap,
		Eye,
		Star,
		TrendingUp,
		Clock,
		CheckCircle,
		AlertCircle,
		Loader,
		Plus,
		ExternalLink
	} from 'lucide-svelte';
	import type { Model, ModelType, ModelStatus } from '$lib/types';
	import ModelCard from '$lib/components/models/ModelCard.svelte';
	import ModelDownloader from '$lib/components/models/ModelDownloader.svelte';
	import ModelMarketplace from '$lib/components/models/ModelMarketplace.svelte';

	// State
	let searchQuery = '';
	let selectedFilter = 'all';
	let selectedType: ModelType | 'all' = 'all';
	let viewMode: 'grid' | 'list' = 'grid';
	let showMarketplace = false;
	let showDownloader = false;
	let selectedModel: Model | null = null;
	let sortBy = 'name';
	let sortOrder: 'asc' | 'desc' = 'asc';

	// Filters
	const filterOptions = [
		{ value: 'all', label: 'All Models', count: 0 },
		{ value: 'downloaded', label: 'Downloaded', count: 0 },
		{ value: 'downloading', label: 'Downloading', count: 0 },
		{ value: 'available', label: 'Available', count: 0 },
		{ value: 'error', label: 'Error', count: 0 }
	];

	const typeOptions = [
		{ value: 'all', label: 'All Types' },
		{ value: 'llm', label: 'Language Models' },
		{ value: 'embedding', label: 'Embedding Models' },
		{ value: 'reranking', label: 'Reranking Models' },
		{ value: 'vision', label: 'Vision Models' },
		{ value: 'audio', label: 'Audio Models' }
	];

	const sortOptions = [
		{ value: 'name', label: 'Name' },
		{ value: 'size', label: 'Size' },
		{ value: 'downloads', label: 'Downloads' },
		{ value: 'rating', label: 'Rating' },
		{ value: 'updated', label: 'Last Updated' }
	];

	// Status colors
	const statusColors: Record<ModelStatus, string> = {
		'available': 'bg-blue-500/20 text-blue-400',
		'downloading': 'bg-yellow-500/20 text-yellow-400',
		'downloaded': 'bg-green-500/20 text-green-400',
		'error': 'bg-red-500/20 text-red-400',
		'updating': 'bg-purple-500/20 text-purple-400'
	};

	// Real-time updates
	let unsubscribeWebSocket: (() => void) | null = null;

	onMount(async () => {
		await loadModels();
		setupRealTimeUpdates();
	});

	onDestroy(() => {
		if (unsubscribeWebSocket) {
			unsubscribeWebSocket();
		}
	});

	async function loadModels() {
		modelsLoading.set(true);
		modelsError.set(null);

		try {
			const response = await modelService.getModels();
			if (response.success && response.data) {
				models.set(response.data);
				updateFilterCounts(response.data);
			} else {
				throw new Error(response.error || 'Failed to load models');
			}
		} catch (error) {
			console.error('Failed to load models:', error);
			modelsError.set(error instanceof Error ? error.message : 'Unknown error');
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Models',
				message: 'Unable to load models. Please try again.'
			});
		} finally {
			modelsLoading.set(false);
		}
	}

	function setupRealTimeUpdates() {
		// Subscribe to model download progress
		unsubscribeWebSocket = websocketService.on('model_download_progress' as any, (data) => {
			updateModelProgress(data.model_id, data.progress, data.status);
		});

		// Subscribe to model status updates
		websocketService.on('model_status_update' as any, (data) => {
			updateModelStatus(data.model_id, data.status, data.metadata);
		});
	}

	function updateModelProgress(modelId: string, progress: number, status: ModelStatus) {
		models.update(modelList => 
			modelList.map(model => 
				model.id === modelId 
					? { ...model, downloadProgress: progress, status }
					: model
			)
		);
	}

	function updateModelStatus(modelId: string, status: ModelStatus, metadata?: any) {
		models.update(modelList => 
			modelList.map(model => 
				model.id === modelId 
					? { ...model, status, ...metadata }
					: model
			)
		);
	}

	function updateFilterCounts(modelList: Model[]) {
		filterOptions[0].count = modelList.length;
		filterOptions[1].count = modelList.filter(m => m.status === 'downloaded').length;
		filterOptions[2].count = modelList.filter(m => m.status === 'downloading').length;
		filterOptions[3].count = modelList.filter(m => m.status === 'available').length;
		filterOptions[4].count = modelList.filter(m => m.status === 'error').length;
	}

	async function handleDownloadModel(model: Model) {
		try {
			const result = await modelService.downloadModel(model.id);
			if (result.success) {
				notificationActions.add({
					type: 'success',
					title: 'Download Started',
					message: `${model.name} download has begun.`
				});
				
				// Update model status immediately
				updateModelStatus(model.id, 'downloading');
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Download error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Download Failed',
				message: `Failed to download ${model.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	async function handleDeleteModel(model: Model) {
		if (!confirm(`Are you sure you want to delete ${model.name}? This action cannot be undone.`)) {
			return;
		}

		try {
			const result = await modelService.deleteModel(model.id);
			if (result.success) {
				notificationActions.add({
					type: 'success',
					title: 'Model Deleted',
					message: `${model.name} has been deleted.`
				});
				
				// Update model status
				updateModelStatus(model.id, 'available');
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Delete error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Delete Failed',
				message: `Failed to delete ${model.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	async function handleCancelDownload(model: Model) {
		try {
			const result = await modelService.cancelDownload(model.id);
			if (result.success) {
				notificationActions.add({
					type: 'info',
					title: 'Download Cancelled',
					message: `${model.name} download has been cancelled.`
				});
				
				updateModelStatus(model.id, 'available');
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Cancel error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Cancel Failed',
				message: `Failed to cancel download: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	// Computed properties
	$: filteredModels = $models.filter(model => {
		// Search filter
		if (searchQuery && !model.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
			!model.description.toLowerCase().includes(searchQuery.toLowerCase()) &&
			!model.provider.toLowerCase().includes(searchQuery.toLowerCase())) {
			return false;
		}

		// Status filter
		if (selectedFilter !== 'all') {
			if (selectedFilter === 'downloaded' && model.status !== 'downloaded') return false;
			if (selectedFilter === 'downloading' && model.status !== 'downloading') return false;
			if (selectedFilter === 'available' && model.status !== 'available') return false;
			if (selectedFilter === 'error' && model.status !== 'error') return false;
		}

		// Type filter
		if (selectedType !== 'all' && model.type !== selectedType) {
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
			case 'size':
				aVal = a.size || 0;
				bVal = b.size || 0;
				break;
			case 'downloads':
				aVal = a.downloads || 0;
				bVal = b.downloads || 0;
				break;
			case 'rating':
				aVal = a.rating || 0;
				bVal = b.rating || 0;
				break;
			case 'updated':
				aVal = new Date(a.updatedAt || 0).getTime();
				bVal = new Date(b.updatedAt || 0).getTime();
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

	function toggleSort(field: string) {
		if (sortBy === field) {
			sortOrder = sortOrder === 'asc' ? 'desc' : 'asc';
		} else {
			sortBy = field;
			sortOrder = 'asc';
		}
	}

	function getModelTypeIcon(type: ModelType): any {
		switch (type) {
			case 'llm': return Cpu;
			case 'embedding': return Zap;
			case 'reranking': return TrendingUp;
			case 'vision': return Eye;
			case 'audio': return Play;
			default: return Settings;
		}
	}

	function formatFileSize(bytes: number): string {
		if (!bytes) return 'Unknown';
		const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
		const i = Math.floor(Math.log(bytes) / Math.log(1024));
		return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
	}
</script>

<svelte:head>
	<title>Model Management - Agentic AI Platform</title>
	<meta name="description" content="Manage AI models, download from marketplace, and configure model settings" />
</svelte:head>

<div class="models-page p-6 space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold text-white">Model Management</h1>
			<p class="text-dark-300 mt-1">Download, configure, and manage AI models</p>
		</div>
		<div class="flex items-center space-x-3">
			<button
				on:click={() => showMarketplace = true}
				class="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
			>
				<Plus class="w-4 h-4" />
				<span>Browse Marketplace</span>
			</button>
			<button
				on:click={loadModels}
				disabled={$modelsLoading}
				class="bg-dark-700 hover:bg-dark-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
			>
				<RefreshCw class="w-4 h-4 {$modelsLoading ? 'animate-spin' : ''}" />
				<span>Refresh</span>
			</button>
		</div>
	</div>

	<!-- Stats Cards -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Total Models</p>
					<p class="text-2xl font-bold text-white">{$models.length}</p>
				</div>
				<HardDrive class="w-8 h-8 text-blue-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Downloaded</p>
					<p class="text-2xl font-bold text-white">{filterOptions[1].count}</p>
				</div>
				<CheckCircle class="w-8 h-8 text-green-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Downloading</p>
					<p class="text-2xl font-bold text-white">{filterOptions[2].count}</p>
				</div>
				<Loader class="w-8 h-8 text-yellow-400" />
			</div>
		</div>
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div>
					<p class="text-dark-300 text-sm">Storage Used</p>
					<p class="text-2xl font-bold text-white">
						{formatFileSize($models.filter(m => m.status === 'downloaded').reduce((sum, m) => sum + (m.size || 0), 0))}
					</p>
				</div>
				<HardDrive class="w-8 h-8 text-purple-400" />
			</div>
		</div>
	</div>

	<!-- Filters and Search -->
	<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
		<div class="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
			<!-- Search -->
			<div class="relative flex-1 max-w-md">
				<Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
				<input
					type="text"
					bind:value={searchQuery}
					placeholder="Search models..."
					class="w-full pl-10 pr-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
				/>
			</div>

			<!-- Filters -->
			<div class="flex items-center space-x-4">
				<!-- Status Filter -->
				<select
					bind:value={selectedFilter}
					class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
				>
					{#each filterOptions as option}
						<option value={option.value}>{option.label} ({option.count})</option>
					{/each}
				</select>

				<!-- Type Filter -->
				<select
					bind:value={selectedType}
					class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
				>
					{#each typeOptions as option}
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
	</div>

	<!-- Loading State -->
	{#if $modelsLoading}
		<div class="flex items-center justify-center py-12">
			<div class="text-center">
				<Loader class="w-8 h-8 text-primary-500 animate-spin mx-auto mb-4" />
				<p class="text-dark-300">Loading models...</p>
			</div>
		</div>
	{/if}

	<!-- Error State -->
	{#if $modelsError}
		<div class="bg-red-500/10 border border-red-500/20 rounded-lg p-6 text-center">
			<AlertCircle class="w-8 h-8 text-red-400 mx-auto mb-4" />
			<h3 class="text-lg font-semibold text-red-400 mb-2">Failed to Load Models</h3>
			<p class="text-red-300 mb-4">{$modelsError}</p>
			<button
				on:click={loadModels}
				class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
			>
				Try Again
			</button>
		</div>
	{/if}

	<!-- Models Grid/List -->
	{#if !$modelsLoading && !$modelsError}
		{#if filteredModels.length === 0}
			<div class="text-center py-12">
				<HardDrive class="w-12 h-12 text-dark-400 mx-auto mb-4" />
				<h3 class="text-lg font-semibold text-white mb-2">No Models Found</h3>
				<p class="text-dark-300 mb-4">
					{searchQuery || selectedFilter !== 'all' || selectedType !== 'all' 
						? 'No models match your current filters.' 
						: 'No models available. Browse the marketplace to download models.'}
				</p>
				<button
					on:click={() => showMarketplace = true}
					class="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
				>
					Browse Marketplace
				</button>
			</div>
		{:else}
			<div class="models-container {viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}">
				{#each filteredModels as model (model.id)}
					<ModelCard
						{model}
						{viewMode}
						on:download={() => handleDownloadModel(model)}
						on:delete={() => handleDeleteModel(model)}
						on:cancel={() => handleCancelDownload(model)}
						on:configure={() => selectedModel = model}
						on:view={() => selectedModel = model}
					/>
				{/each}
			</div>
		{/if}
	{/if}
</div>

<!-- Model Marketplace Modal -->
{#if showMarketplace}
	<ModelMarketplace
		on:close={() => showMarketplace = false}
		on:download={(event) => {
			handleDownloadModel(event.detail);
			showMarketplace = false;
		}}
	/>
{/if}

<!-- Model Downloader Modal -->
{#if showDownloader}
	<ModelDownloader
		on:close={() => showDownloader = false}
		on:download={(event) => {
			handleDownloadModel(event.detail);
			showDownloader = false;
		}}
	/>
{/if}

<style lang="postcss">
	.models-page {
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

	.models-container {
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
