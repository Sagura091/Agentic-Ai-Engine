<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Download,
		Settings,
		Database,
		Brain,
		Search,
		Save,
		RefreshCw,
		CheckCircle,
		AlertCircle,
		Trash2,
		Eye,
		FileText,
		Image,
		Zap,
		BarChart3,
		Globe,
		Link,
		TestTube,
		Loader,
		Monitor,
		HardDrive,
		Cpu,
		Activity,
		Upload,
		Play,
		X,
		Plus
	} from 'lucide-svelte';
	import { apiClient } from '$services/api';
	import { notificationActions } from '$stores';

	// Active tab state
	let activeTab = 'models';

	// Tab configuration
	const tabs = [
		{ id: 'models', name: 'Models', icon: Brain, description: 'Download and manage embedding models' },
		{ id: 'search', name: 'Search', icon: Search, description: 'Configure search and retrieval settings' },
		{ id: 'processing', name: 'Processing', icon: FileText, description: 'Document processing and OCR settings' },
		{ id: 'performance', name: 'Performance', icon: Zap, description: 'Performance and caching configuration' },
		{ id: 'storage', name: 'Storage', icon: HardDrive, description: 'Storage paths and database settings' },
		{ id: 'knowledge', name: 'Knowledge Bases', icon: Database, description: 'Manage knowledge bases and collections' },
		{ id: 'monitoring', name: 'Monitoring', icon: Monitor, description: 'System statistics and health monitoring' }
	];

	// Comprehensive RAG Configuration
	let ragConfig = {
		// Core Embedding Settings
		embedding_engine: '',
		embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
		embedding_batch_size: 32,
		embedding_auto_update: true,

		// Search & Retrieval Configuration
		retrieval_strategy: 'hybrid', // semantic, keyword, hybrid, sparse, dense
		search_mode: 'hybrid', // semantic, hybrid, neural, graph, multimodal
		top_k: 5,
		similarity_threshold: 0.7,
		max_results: 20,

		// Chunking Configuration
		chunk_size: 1000,
		chunk_overlap: 200,
		chunk_strategy: 'recursive', // recursive, semantic, fixed, adaptive

		// Hybrid Search Configuration
		enable_hybrid_search: true,
		hybrid_bm25_weight: 0.3,
		hybrid_semantic_weight: 0.7,

		// Reranking Configuration
		rerank_enabled: true,
		rerank_model: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
		rerank_top_k: 10,

		// OCR & Document Processing
		ocr_enabled: true,
		ocr_engine: 'tesseract', // tesseract, easyocr, paddleocr
		ocr_languages: ['en'],
		pdf_processing_enabled: true,
		image_processing_enabled: true,

		// Performance Configuration
		max_concurrent_operations: 10,
		connection_pool_size: 10,
		cache_enabled: true,
		cache_ttl: 3600,

		// Storage Configuration
		data_dir: './data',
		models_dir: './data/models',
		vector_db_dir: './data/chroma',
		cache_dir: './data/cache',

		// Advanced Features
		contextual_search: true,
		query_expansion: true,
		enable_access_control: true,
		bypass_embedding_and_retrieval: false
	};

	// Available Models and State
	let availableModels = {
		embedding: [],
		reranking: [],
		vision: [],
		llm: []
	};
	let downloadedModels = new Set();
	let downloadingModels = new Set();
	let loadingConfig = false;
	let savingConfig = false;
	let testingModel = null;

	// Model Download State
	let customModelUrl = '';
	let customModelName = '';
	let customModelType = 'embedding';
	let showCustomModelForm = false;

	// Additional State for Complete Backend Integration
	let knowledgeBases = [];
	let systemStats = {};
	let uploadProgress = {};
	let collections = [];

	// We now load models dynamically from the backend

	onMount(async () => {
		await loadRagConfiguration();
		await loadAvailableModels();
		await loadKnowledgeBases();
		await loadSystemStatistics();
		await loadCollections();
	});

	async function loadRagConfiguration() {
		loadingConfig = true;
		try {
			const response = await apiClient.getRagConfiguration();
			if (response.success) {
				ragConfig = { ...ragConfig, ...response.data };
			}
		} catch (error) {
			console.error('Failed to load RAG configuration:', error);
			notificationActions.add({
				type: 'error',
				title: 'Configuration Error',
				message: 'Failed to load RAG configuration'
			});
		} finally {
			loadingConfig = false;
		}
	}

	async function loadAvailableModels() {
		try {
			const response = await apiClient.getAvailableEmbeddingModels();
			if (response.success && response.data.models_by_type) {
				availableModels = response.data.models_by_type;

				// Create a set of downloaded model IDs for quick lookup
				downloadedModels = new Set();
				Object.values(availableModels).flat().forEach(model => {
					if (model.is_downloaded) {
						downloadedModels.add(model.model_id);
					}
				});
			}
		} catch (error) {
			console.error('Failed to load available models:', error);
			notificationActions.add({
				type: 'error',
				title: 'Load Error',
				message: 'Failed to load available models'
			});
		}
	}

	async function loadKnowledgeBases() {
		try {
			const response = await apiClient.getKnowledgeBases();
			if (response.success && response.data) {
				knowledgeBases = response.data;
			}
		} catch (error) {
			console.error('Failed to load knowledge bases:', error);
		}
	}

	async function loadSystemStatistics() {
		try {
			const response = await apiClient.getRagStatistics();
			if (response.success && response.data) {
				systemStats = response.data;
			}
		} catch (error) {
			console.error('Failed to load system statistics:', error);
		}
	}

	async function loadCollections() {
		try {
			const response = await apiClient.getCollections();
			if (response.success && response.data) {
				collections = response.data;
			}
		} catch (error) {
			console.error('Failed to load collections:', error);
		}
	}

	async function downloadModel(modelId: string) {
		downloadingModels.add(modelId);
		downloadingModels = downloadingModels;

		try {
			const response = await apiClient.downloadEmbeddingModel(modelId, false);

			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Model Download Started',
					message: `${modelId} download started. Models are saved to data/models directory.`
				});

				// Poll for download completion
				pollDownloadProgress(modelId);
			} else {
				throw new Error(response.error || 'Download failed');
			}
		} catch (error) {
			console.error('Failed to download model:', error);
			notificationActions.add({
				type: 'error',
				title: 'Download Failed',
				message: `Failed to download ${modelId}: ${error.message}`
			});
			downloadingModels.delete(modelId);
			downloadingModels = downloadingModels;
		}
	}

	async function pollDownloadProgress(modelId: string) {
		const maxAttempts = 60; // 5 minutes max
		let attempts = 0;

		const checkProgress = async () => {
			try {
				attempts++;
				const response = await apiClient.getModelDownloadProgress(modelId);

				if (response.success && response.data.status === 'completed') {
					downloadingModels.delete(modelId);
					downloadingModels = downloadingModels;
					downloadedModels.add(modelId);
					downloadedModels = downloadedModels;

					notificationActions.add({
						type: 'success',
						title: 'Download Complete',
						message: `${modelId} has been downloaded successfully`
					});

					// Reload models to get updated status
					await loadAvailableModels();
					return;
				}

				if (response.success && response.data.status === 'failed') {
					downloadingModels.delete(modelId);
					downloadingModels = downloadingModels;

					notificationActions.add({
						type: 'error',
						title: 'Download Failed',
						message: `${modelId} download failed`
					});
					return;
				}

				// Continue polling if still downloading and haven't exceeded max attempts
				if (attempts < maxAttempts && downloadingModels.has(modelId)) {
					setTimeout(checkProgress, 5000); // Check every 5 seconds
				} else if (attempts >= maxAttempts) {
					downloadingModels.delete(modelId);
					downloadingModels = downloadingModels;

					notificationActions.add({
						type: 'warning',
						title: 'Download Status Unknown',
						message: `${modelId} download status could not be determined. Check the backend logs.`
					});
				}
			} catch (error) {
				console.error('Failed to check download progress:', error);
			}
		};

		// Start checking after 2 seconds
		setTimeout(checkProgress, 2000);
	}

	async function downloadCustomModel() {
		if (!customModelUrl || !customModelName || !customModelType) {
			notificationActions.add({
				type: 'error',
				title: 'Validation Error',
				message: 'Please provide model URL, name, and type'
			});
			return;
		}

		try {
			// First add the custom model to the backend catalog
			const addResponse = await apiClient.addCustomModel({
				model_id: customModelUrl,
				name: customModelName,
				description: `Custom ${customModelType} model`,
				model_type: customModelType,
				model_source: 'huggingface',
				download_url: customModelUrl,
				size_mb: 100, // Default size estimate
				tags: ['custom']
			});

			if (addResponse.success) {
				// Then download it
				await downloadModel(customModelUrl);
				customModelUrl = '';
				customModelName = '';
				customModelType = 'embedding';
				showCustomModelForm = false;
			} else {
				throw new Error(addResponse.error || 'Failed to add custom model');
			}
		} catch (error) {
			console.error('Failed to add custom model:', error);
			notificationActions.add({
				type: 'error',
				title: 'Custom Model Error',
				message: `Failed to add custom model: ${error.message}`
			});
		}
	}

	async function saveConfiguration() {
		savingConfig = true;
		try {
			const response = await apiClient.updateRagConfiguration(ragConfig);
			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Configuration Saved',
					message: 'RAG configuration updated successfully and applied to backend'
				});
			}
		} catch (error) {
			console.error('Failed to save configuration:', error);
			notificationActions.add({
				type: 'error',
				title: 'Save Failed',
				message: 'Failed to save RAG configuration'
			});
		} finally {
			savingConfig = false;
		}
	}

	async function testEmbeddingModel(modelId: string) {
		testingModel = modelId;
		try {
			const response = await apiClient.testEmbeddingModel(modelId, 'This is a test sentence for embedding generation.');

			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Model Test Successful',
					message: `${modelId} is working correctly`
				});
			}
		} catch (error) {
			notificationActions.add({
				type: 'error',
				title: 'Model Test Failed',
				message: `Failed to test ${modelId}`
			});
		} finally {
			testingModel = null;
		}
	}

	function isModelDownloaded(modelId: string): boolean {
		return downloadedModels.has(modelId);
	}

	function getPerformanceColor(performance: string): string {
		switch (performance) {
			case 'fast': return 'text-green-400';
			case 'medium': return 'text-yellow-400';
			case 'slow': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}

	function getAccuracyColor(accuracy: string): string {
		switch (accuracy) {
			case 'excellent': return 'text-green-400';
			case 'good': return 'text-blue-400';
			case 'fair': return 'text-yellow-400';
			default: return 'text-gray-400';
		}
	}
</script>

<div class="rag-settings">
	<!-- Header -->
	<div class="settings-header">
		<div class="header-content">
			<div class="header-icon">
				<Database class="w-8 h-8 text-primary-400" />
			</div>
			<div class="header-text">
				<h1 class="header-title">RAG Configuration</h1>
				<p class="header-description">Configure Retrieval-Augmented Generation settings, models, and processing options</p>
			</div>
		</div>
		<div class="header-actions">
			<button
				class="btn-secondary"
				on:click={loadRagConfiguration}
				disabled={loadingConfig}
			>
				<RefreshCw class="w-4 h-4 {loadingConfig ? 'animate-spin' : ''}" />
				Refresh
			</button>
			<button
				class="btn-primary"
				on:click={saveConfiguration}
				disabled={savingConfig}
			>
				<Save class="w-4 h-4" />
				{savingConfig ? 'Saving...' : 'Save Configuration'}
			</button>
		</div>
	</div>

	<!-- Tab Navigation -->
	<div class="tab-navigation">
		{#each tabs as tab}
			<button
				class="tab-button {activeTab === tab.id ? 'active' : ''}"
				on:click={() => activeTab = tab.id}
			>
				<svelte:component this={tab.icon} class="w-5 h-5" />
				<div class="tab-content">
					<span class="tab-name">{tab.name}</span>
					<span class="tab-description">{tab.description}</span>
				</div>
			</button>
		{/each}
	</div>

	<!-- Tab Content -->
	<div class="tab-content-container">
		{#if activeTab === 'models'}
			<!-- Models Tab -->
			<div class="tab-panel">
				<div class="panel-header">
					<h2 class="panel-title">Model Management</h2>
					<button
						class="btn-accent"
						on:click={() => showCustomModelForm = !showCustomModelForm}
					>
						<Plus class="w-4 h-4" />
						Add Custom Model
					</button>
				</div>

				{#if showCustomModelForm}
					<div class="custom-model-form">
						<div class="form-grid">
							<div class="form-group">
								<label class="form-label">Model Name</label>
								<input
									type="text"
									class="form-input"
									placeholder="e.g., My Custom Model"
									bind:value={customModelName}
								/>
							</div>
							<div class="form-group">
								<label class="form-label">Model URL or HuggingFace ID</label>
								<input
									type="text"
									class="form-input"
									placeholder="e.g., sentence-transformers/model-name"
									bind:value={customModelUrl}
								/>
							</div>
							<div class="form-group">
								<label class="form-label">Model Type</label>
								<select class="form-select" bind:value={customModelType}>
									<option value="embedding">Embedding Model</option>
									<option value="reranking">Reranking Model</option>
									<option value="vision">Vision Model</option>
									<option value="llm">LLM Model</option>
								</select>
							</div>
						</div>
						<div class="form-actions">
							<button class="btn-primary" on:click={downloadCustomModel}>
								<Download class="w-4 h-4" />
								Add & Download Model
							</button>
							<button class="btn-secondary" on:click={() => showCustomModelForm = false}>
								<X class="w-4 h-4" />
								Cancel
							</button>
						</div>
					</div>
				{/if}

				<!-- Embedding Models -->
				<div class="model-section">
					<h3 class="section-title">Embedding Models</h3>
					<div class="models-grid">
						{#each availableModels.embedding as model}
							<div class="model-card">
								<div class="model-header">
									<h5 class="model-name">{model.name}</h5>
									<div class="model-badges">
										<span class="performance-badge {getPerformanceColor(model.performance_tier || 'medium')}">
											{model.performance_tier || 'medium'}
										</span>
										<span class="accuracy-badge text-blue-400">
											{model.dimension}d
										</span>
									</div>
								</div>
								<p class="model-description">{model.description}</p>
								<div class="model-meta">
									<span class="model-size">{model.size_mb}MB</span>
									<span class="model-id">{model.model_id}</span>
								</div>
								<div class="model-actions">
									{#if isModelDownloaded(model.model_id)}
										<button
											class="btn-success-sm"
											on:click={() => testEmbeddingModel(model.model_id)}
											disabled={testingModel === model.model_id}
										>
											{#if testingModel === model.model_id}
												<Loader class="w-4 h-4 animate-spin" />
											{:else}
												<TestTube class="w-4 h-4" />
											{/if}
											Test
										</button>
										<span class="status-downloaded">
											<CheckCircle class="w-4 h-4" />
											Downloaded
										</span>
									{:else}
										<button
											class="btn-primary-sm"
											on:click={() => downloadModel(model.model_id)}
											disabled={downloadingModels.has(model.model_id)}
										>
											<Download class="w-4 h-4 {downloadingModels.has(model.model_id) ? 'animate-pulse' : ''}" />
											{downloadingModels.has(model.model_id) ? 'Downloading...' : 'Download'}
										</button>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				</div>

				<!-- Reranking Models -->
				<div class="model-section">
					<h3 class="section-title">Reranking Models</h3>
					<div class="models-grid">
						{#each availableModels.reranking as model}
							<div class="model-card">
								<div class="model-header">
									<h5 class="model-name">{model.name}</h5>
									<div class="model-badges">
										<span class="performance-badge {getPerformanceColor(model.performance_tier || 'medium')}">
											{model.performance_tier || 'medium'}
										</span>
										<span class="accuracy-badge text-green-400">
											rerank
										</span>
									</div>
								</div>
								<p class="model-description">{model.description}</p>
								<div class="model-meta">
									<span class="model-size">{model.size_mb}MB</span>
									<span class="model-id">{model.model_id}</span>
								</div>
								<div class="model-actions">
									{#if isModelDownloaded(model.model_id)}
										<span class="status-downloaded">
											<CheckCircle class="w-4 h-4" />
											Downloaded
										</span>
									{:else}
										<button
											class="btn-primary-sm"
											on:click={() => downloadModel(model.model_id)}
											disabled={downloadingModels.has(model.model_id)}
										>
											<Download class="w-4 h-4 {downloadingModels.has(model.model_id) ? 'animate-pulse' : ''}" />
											{downloadingModels.has(model.model_id) ? 'Downloading...' : 'Download'}
										</button>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				</div>

				<!-- Vision Models -->
				{#if availableModels.vision && availableModels.vision.length > 0}
					<div class="model-section">
						<h3 class="section-title">Vision Models</h3>
						<div class="models-grid">
							{#each availableModels.vision as model}
								<div class="model-card">
									<div class="model-header">
										<h5 class="model-name">{model.name}</h5>
										<div class="model-badges">
											<span class="performance-badge {getPerformanceColor(model.performance_tier || 'medium')}">
												{model.performance_tier || 'medium'}
											</span>
											<span class="accuracy-badge text-purple-400">
												vision
											</span>
										</div>
									</div>
									<p class="model-description">{model.description}</p>
									<div class="model-meta">
										<span class="model-size">{model.size_mb}MB</span>
										<span class="model-id">{model.model_id}</span>
									</div>
									<div class="model-actions">
										{#if isModelDownloaded(model.model_id)}
											<span class="status-downloaded">
												<CheckCircle class="w-4 h-4" />
												Downloaded
											</span>
										{:else}
											<button
												class="btn-primary-sm"
												on:click={() => downloadModel(model.model_id)}
												disabled={downloadingModels.has(model.model_id)}
											>
												<Download class="w-4 h-4 {downloadingModels.has(model.model_id) ? 'animate-pulse' : ''}" />
												{downloadingModels.has(model.model_id) ? 'Downloading...' : 'Download'}
											</button>
										{/if}
									</div>
								</div>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		{:else if activeTab === 'search'}
			<!-- Search Configuration Tab -->
			<div class="tab-panel">
				<div class="panel-header">
					<h2 class="panel-title">Search & Retrieval Configuration</h2>
				</div>

				<div class="config-sections">
					<div class="config-section">
						<h3 class="config-title">Search Strategy</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">Retrieval Strategy</label>
								<select class="form-select" bind:value={ragConfig.retrieval_strategy}>
									<option value="semantic">Semantic Search</option>
									<option value="keyword">Keyword Search</option>
									<option value="hybrid">Hybrid Search</option>
									<option value="sparse">Sparse Search</option>
									<option value="dense">Dense Search</option>
								</select>
							</div>
							<div class="form-group">
								<label class="form-label">Search Mode</label>
								<select class="form-select" bind:value={ragConfig.search_mode}>
									<option value="semantic">Semantic</option>
									<option value="hybrid">Hybrid</option>
									<option value="neural">Neural</option>
									<option value="graph">Graph</option>
									<option value="multimodal">Multimodal</option>
								</select>
							</div>
							<div class="form-group">
								<label class="form-label">Top K Results</label>
								<input type="number" class="form-input" bind:value={ragConfig.top_k} min="1" max="100" />
							</div>
							<div class="form-group">
								<label class="form-label">Similarity Threshold</label>
								<input type="number" class="form-input" bind:value={ragConfig.similarity_threshold} min="0" max="1" step="0.01" />
							</div>
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">Hybrid Search Settings</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.enable_hybrid_search} />
									Enable Hybrid Search
								</label>
							</div>
							<div class="form-group">
								<label class="form-label">BM25 Weight</label>
								<input type="number" class="form-input" bind:value={ragConfig.hybrid_bm25_weight} min="0" max="1" step="0.1" />
							</div>
							<div class="form-group">
								<label class="form-label">Semantic Weight</label>
								<input type="number" class="form-input" bind:value={ragConfig.hybrid_semantic_weight} min="0" max="1" step="0.1" />
							</div>
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">Reranking Configuration</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.rerank_enabled} />
									Enable Reranking
								</label>
							</div>
							<div class="form-group">
								<label class="form-label">Rerank Model</label>
								<select class="form-select" bind:value={ragConfig.rerank_model}>
									{#each availableModels.reranking as model}
										<option value={model.model_id}>{model.name}</option>
									{/each}
								</select>
							</div>
							<div class="form-group">
								<label class="form-label">Rerank Top K</label>
								<input type="number" class="form-input" bind:value={ragConfig.rerank_top_k} min="1" max="50" />
							</div>
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">Advanced Features</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.contextual_search} />
									Contextual Search
								</label>
							</div>
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.query_expansion} />
									Query Expansion
								</label>
							</div>
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.enable_access_control} />
									Access Control
								</label>
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if activeTab === 'processing'}
			<!-- Document Processing Tab -->
			<div class="tab-panel">
				<div class="panel-header">
					<h2 class="panel-title">Document Processing & OCR</h2>
				</div>

				<div class="config-sections">
					<div class="config-section">
						<h3 class="config-title">Chunking Configuration</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">Chunk Size</label>
								<input type="number" class="form-input" bind:value={ragConfig.chunk_size} min="100" max="5000" />
							</div>
							<div class="form-group">
								<label class="form-label">Chunk Overlap</label>
								<input type="number" class="form-input" bind:value={ragConfig.chunk_overlap} min="0" max="1000" />
							</div>
							<div class="form-group">
								<label class="form-label">Chunk Strategy</label>
								<select class="form-select" bind:value={ragConfig.chunk_strategy}>
									<option value="recursive">Recursive</option>
									<option value="semantic">Semantic</option>
									<option value="fixed">Fixed</option>
									<option value="adaptive">Adaptive</option>
								</select>
							</div>
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">OCR Configuration</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.ocr_enabled} />
									Enable OCR
								</label>
							</div>
							<div class="form-group">
								<label class="form-label">OCR Engine</label>
								<select class="form-select" bind:value={ragConfig.ocr_engine}>
									<option value="tesseract">Tesseract</option>
									<option value="easyocr">EasyOCR</option>
									<option value="paddleocr">PaddleOCR</option>
								</select>
							</div>
							<div class="form-group">
								<label class="form-label">OCR Languages</label>
								<input type="text" class="form-input" bind:value={ragConfig.ocr_languages} placeholder="en,es,fr" />
							</div>
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">Document Types</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.pdf_processing_enabled} />
									PDF Processing
								</label>
							</div>
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.image_processing_enabled} />
									Image Processing
								</label>
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if activeTab === 'performance'}
			<!-- Performance Tab -->
			<div class="tab-panel">
				<div class="panel-header">
					<h2 class="panel-title">Performance & Caching</h2>
				</div>

				<div class="config-sections">
					<div class="config-section">
						<h3 class="config-title">Performance Settings</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">Max Concurrent Operations</label>
								<input type="number" class="form-input" bind:value={ragConfig.max_concurrent_operations} min="1" max="100" />
							</div>
							<div class="form-group">
								<label class="form-label">Connection Pool Size</label>
								<input type="number" class="form-input" bind:value={ragConfig.connection_pool_size} min="1" max="50" />
							</div>
							<div class="form-group">
								<label class="form-label">Embedding Batch Size</label>
								<input type="number" class="form-input" bind:value={ragConfig.embedding_batch_size} min="1" max="128" />
							</div>
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">Caching Configuration</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">
									<input type="checkbox" bind:checked={ragConfig.cache_enabled} />
									Enable Caching
								</label>
							</div>
							<div class="form-group">
								<label class="form-label">Cache TTL (seconds)</label>
								<input type="number" class="form-input" bind:value={ragConfig.cache_ttl} min="60" max="86400" />
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if activeTab === 'storage'}
			<!-- Storage Tab -->
			<div class="tab-panel">
				<div class="panel-header">
					<h2 class="panel-title">Storage Configuration</h2>
				</div>

				<div class="config-sections">
					<div class="config-section">
						<h3 class="config-title">Directory Paths</h3>
						<div class="config-grid">
							<div class="form-group">
								<label class="form-label">Data Directory</label>
								<input type="text" class="form-input" bind:value={ragConfig.data_dir} />
							</div>
							<div class="form-group">
								<label class="form-label">Models Directory</label>
								<input type="text" class="form-input" bind:value={ragConfig.models_dir} />
							</div>
							<div class="form-group">
								<label class="form-label">Vector Database Directory</label>
								<input type="text" class="form-input" bind:value={ragConfig.vector_db_dir} />
							</div>
							<div class="form-group">
								<label class="form-label">Cache Directory</label>
								<input type="text" class="form-input" bind:value={ragConfig.cache_dir} />
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if activeTab === 'knowledge'}
			<!-- Knowledge Base Management Tab -->
			<div class="tab-panel">
				<div class="panel-header">
					<h2 class="panel-title">Knowledge Base Management</h2>
					<button class="btn-primary" on:click={() => {}}>
						<Plus class="w-4 h-4" />
						Create Knowledge Base
					</button>
				</div>

				<div class="config-sections">
					<div class="config-section">
						<h3 class="config-title">Knowledge Bases</h3>
						<div class="knowledge-bases-grid">
							{#each knowledgeBases as kb}
								<div class="knowledge-base-card">
									<div class="kb-header">
										<h4 class="kb-name">{kb.name}</h4>
										<div class="kb-actions">
											<button class="btn-secondary-sm" title="View Details">
												<Eye class="w-4 h-4" />
											</button>
											<button class="btn-danger-sm" title="Delete">
												<Trash2 class="w-4 h-4" />
											</button>
										</div>
									</div>
									<p class="kb-description">{kb.description || 'No description'}</p>
									<div class="kb-stats">
										<span class="kb-stat">
											<FileText class="w-4 h-4" />
											{kb.document_count || 0} docs
										</span>
										<span class="kb-stat">
											<HardDrive class="w-4 h-4" />
											{kb.size_mb || 0}MB
										</span>
									</div>
								</div>
							{:else}
								<div class="empty-state">
									<Database class="w-12 h-12 text-gray-500" />
									<h3>No Knowledge Bases</h3>
									<p>Create your first knowledge base to get started</p>
								</div>
							{/each}
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">Collections</h3>
						<div class="collections-list">
							{#each collections as collection}
								<div class="collection-item">
									<div class="collection-info">
										<span class="collection-name">{collection}</span>
									</div>
									<button class="btn-danger-sm" title="Delete Collection">
										<Trash2 class="w-4 h-4" />
									</button>
								</div>
							{:else}
								<div class="empty-state-small">
									<p>No collections found</p>
								</div>
							{/each}
						</div>
					</div>
				</div>
			</div>
		{:else if activeTab === 'monitoring'}
			<!-- System Monitoring Tab -->
			<div class="tab-panel">
				<div class="panel-header">
					<h2 class="panel-title">System Monitoring</h2>
					<button class="btn-secondary" on:click={() => { loadSystemStatistics(); }}>
						<RefreshCw class="w-4 h-4" />
						Refresh
					</button>
				</div>

				<div class="config-sections">
					<div class="config-section">
						<h3 class="config-title">System Statistics</h3>
						<div class="stats-grid">
							<div class="stat-card">
								<div class="stat-icon">
									<Database class="w-6 h-6 text-blue-400" />
								</div>
								<div class="stat-content">
									<div class="stat-value">{systemStats.total_agents || 0}</div>
									<div class="stat-label">Total Agents</div>
								</div>
							</div>
							<div class="stat-card">
								<div class="stat-icon">
									<FileText class="w-6 h-6 text-green-400" />
								</div>
								<div class="stat-content">
									<div class="stat-value">{systemStats.total_documents || 0}</div>
									<div class="stat-label">Total Documents</div>
								</div>
							</div>
							<div class="stat-card">
								<div class="stat-icon">
									<Search class="w-6 h-6 text-purple-400" />
								</div>
								<div class="stat-content">
									<div class="stat-value">{systemStats.total_queries || 0}</div>
									<div class="stat-label">Total Queries</div>
								</div>
							</div>
							<div class="stat-card">
								<div class="stat-icon">
									<HardDrive class="w-6 h-6 text-orange-400" />
								</div>
								<div class="stat-content">
									<div class="stat-value">{systemStats.total_collections || 0}</div>
									<div class="stat-label">Collections</div>
								</div>
							</div>
						</div>
					</div>

					<div class="config-section">
						<h3 class="config-title">System Health</h3>
						<div class="health-indicators">
							<div class="health-item">
								<div class="health-status status-good"></div>
								<span>Vector Database</span>
								<span class="health-value">Connected</span>
							</div>
							<div class="health-item">
								<div class="health-status status-good"></div>
								<span>Embedding Service</span>
								<span class="health-value">Active</span>
							</div>
							<div class="health-item">
								<div class="health-status status-good"></div>
								<span>Document Processing</span>
								<span class="health-value">Ready</span>
							</div>
						</div>
					</div>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.rag-settings {
		padding: 1.5rem;
		max-width: 80rem;
		margin: 0 auto;
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.settings-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		background-color: #1f2937;
		border-radius: 0.75rem;
		padding: 1.5rem;
		border: 1px solid #374151;
	}

	.header-content {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.header-icon {
		padding: 0.75rem;
		background-color: rgba(59, 130, 246, 0.1);
		border-radius: 0.75rem;
	}

	.header-text {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.header-title {
		font-size: 1.5rem;
		font-weight: 700;
		color: white;
	}

	.header-description {
		color: #9ca3af;
	}

	.header-actions {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.tab-navigation {
		display: flex;
		background-color: #1f2937;
		border-radius: 0.75rem;
		padding: 0.5rem;
		border: 1px solid #374151;
		overflow-x: auto;
	}

	.tab-button {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.75rem 1rem;
		border-radius: 0.5rem;
		transition: all 0.2s;
		min-width: 0;
		flex-shrink: 0;
		background: none;
		border: none;
		color: #9ca3af;
		cursor: pointer;
	}

	.tab-button:hover {
		background-color: #374151;
	}

	.tab-button.active {
		background-color: #3b82f6;
		color: white;
	}

	.tab-content {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	.tab-name {
		font-weight: 500;
		font-size: 0.875rem;
		white-space: nowrap;
	}

	.tab-description {
		font-size: 0.75rem;
		color: #9ca3af;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.tab-button.active .tab-description {
		color: #dbeafe;
	}

	.tab-content-container {
		background-color: #1f2937;
		border-radius: 0.75rem;
		border: 1px solid #374151;
	}

	.tab-panel {
		padding: 1.5rem;
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding-bottom: 1rem;
		border-bottom: 1px solid #374151;
	}

	.panel-title {
		font-size: 1.25rem;
		font-weight: 600;
		color: white;
	}

	.custom-model-form {
		background-color: #374151;
		border-radius: 0.5rem;
		padding: 1rem;
		border: 1px solid #4b5563;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.form-grid {
		display: grid;
		grid-template-columns: 1fr;
		gap: 1rem;
	}

	@media (min-width: 768px) {
		.form-grid {
			grid-template-columns: 1fr 1fr;
		}
	}

	.form-group {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.form-label {
		display: block;
		font-size: 0.875rem;
		font-weight: 500;
		color: #d1d5db;
	}

	.form-input, .form-select {
		width: 100%;
		padding: 0.5rem 0.75rem;
		background-color: #4b5563;
		border: 1px solid #6b7280;
		border-radius: 0.5rem;
		color: white;
	}

	.form-input::placeholder {
		color: #9ca3af;
	}

	.form-input:focus, .form-select:focus {
		outline: none;
		border-color: #3b82f6;
		box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5);
	}

	.form-actions {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding-top: 0.5rem;
	}

	.model-section {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.section-title {
		font-size: 1.125rem;
		font-weight: 600;
		color: white;
	}

	.model-category {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.category-title {
		font-size: 1rem;
		font-weight: 500;
		color: #d1d5db;
	}

	.models-grid {
		display: grid;
		grid-template-columns: 1fr;
		gap: 1rem;
	}

	@media (min-width: 1024px) {
		.models-grid {
			grid-template-columns: 1fr 1fr;
		}
	}

	@media (min-width: 1280px) {
		.models-grid {
			grid-template-columns: 1fr 1fr 1fr;
		}
	}

	.model-card {
		background-color: #374151;
		border-radius: 0.5rem;
		padding: 1rem;
		border: 1px solid #4b5563;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		transition: border-color 0.2s;
	}

	.model-card:hover {
		border-color: rgba(59, 130, 246, 0.5);
	}

	.model-header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
	}

	.model-name {
		font-weight: 500;
		color: white;
	}

	.model-badges {
		display: flex;
		gap: 0.5rem;
	}

	.performance-badge, .accuracy-badge {
		padding: 0.25rem 0.5rem;
		font-size: 0.75rem;
		border-radius: 9999px;
		background-color: #4b5563;
		font-weight: 500;
	}

	.model-description {
		font-size: 0.875rem;
		color: #9ca3af;
	}

	.model-meta {
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-size: 0.75rem;
		color: #6b7280;
	}

	.model-size {
		font-weight: 500;
	}

	.model-id {
		font-family: monospace;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.model-actions {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.status-downloaded {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		color: #10b981;
		font-size: 0.875rem;
	}

	.config-sections {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.config-section {
		background-color: #374151;
		border-radius: 0.5rem;
		padding: 1rem;
		border: 1px solid #4b5563;
	}

	.config-title {
		font-size: 1.125rem;
		font-weight: 500;
		color: white;
		margin-bottom: 1rem;
	}

	.config-grid {
		display: grid;
		grid-template-columns: 1fr;
		gap: 1rem;
	}

	@media (min-width: 768px) {
		.config-grid {
			grid-template-columns: 1fr 1fr;
		}
	}

	@media (min-width: 1024px) {
		.config-grid {
			grid-template-columns: 1fr 1fr 1fr;
		}
	}

	/* Button Styles */
	.btn-primary {
		padding: 0.5rem 1rem;
		background-color: #3b82f6;
		color: white;
		border-radius: 0.5rem;
		font-weight: 500;
		transition: background-color 0.2s;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		border: none;
		cursor: pointer;
	}

	.btn-primary:hover:not(:disabled) {
		background-color: #2563eb;
	}

	.btn-primary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-secondary {
		padding: 0.5rem 1rem;
		background-color: #4b5563;
		color: #d1d5db;
		border-radius: 0.5rem;
		font-weight: 500;
		transition: background-color 0.2s;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		border: none;
		cursor: pointer;
	}

	.btn-secondary:hover:not(:disabled) {
		background-color: #6b7280;
	}

	.btn-secondary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-accent {
		padding: 0.5rem 1rem;
		background-color: #f59e0b;
		color: white;
		border-radius: 0.5rem;
		font-weight: 500;
		transition: background-color 0.2s;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		border: none;
		cursor: pointer;
	}

	.btn-accent:hover {
		background-color: #d97706;
	}

	.btn-primary-sm {
		padding: 0.375rem 0.75rem;
		background-color: #3b82f6;
		color: white;
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-weight: 500;
		transition: background-color 0.2s;
		display: flex;
		align-items: center;
		gap: 0.25rem;
		border: none;
		cursor: pointer;
	}

	.btn-primary-sm:hover:not(:disabled) {
		background-color: #2563eb;
	}

	.btn-primary-sm:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-success-sm {
		padding: 0.375rem 0.75rem;
		background-color: #10b981;
		color: white;
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-weight: 500;
		transition: background-color 0.2s;
		display: flex;
		align-items: center;
		gap: 0.25rem;
		border: none;
		cursor: pointer;
	}

	.btn-success-sm:hover:not(:disabled) {
		background-color: #059669;
	}

	.btn-success-sm:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	/* Knowledge Base Management Styles */
	.knowledge-bases-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
		gap: 1rem;
	}

	.knowledge-base-card {
		background-color: #374151;
		border-radius: 0.5rem;
		padding: 1rem;
		border: 1px solid #4b5563;
	}

	.kb-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.5rem;
	}

	.kb-name {
		font-weight: 600;
		color: white;
		margin: 0;
	}

	.kb-actions {
		display: flex;
		gap: 0.5rem;
	}

	.kb-description {
		color: #9ca3af;
		margin: 0 0 1rem 0;
		font-size: 0.875rem;
	}

	.kb-stats {
		display: flex;
		gap: 1rem;
	}

	.kb-stat {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		color: #d1d5db;
		font-size: 0.875rem;
	}

	.collections-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.collection-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem;
		background-color: #374151;
		border-radius: 0.5rem;
		border: 1px solid #4b5563;
	}

	.collection-name {
		color: white;
		font-weight: 500;
	}

	/* System Monitoring Styles */
	.stats-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 1rem;
	}

	.stat-card {
		background-color: #374151;
		border-radius: 0.5rem;
		padding: 1rem;
		border: 1px solid #4b5563;
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.stat-icon {
		padding: 0.75rem;
		background-color: rgba(59, 130, 246, 0.1);
		border-radius: 0.5rem;
	}

	.stat-content {
		display: flex;
		flex-direction: column;
	}

	.stat-value {
		font-size: 1.5rem;
		font-weight: 700;
		color: white;
	}

	.stat-label {
		color: #9ca3af;
		font-size: 0.875rem;
	}

	.health-indicators {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.health-item {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 0.75rem;
		background-color: #374151;
		border-radius: 0.5rem;
		border: 1px solid #4b5563;
	}

	.health-status {
		width: 0.75rem;
		height: 0.75rem;
		border-radius: 50%;
	}

	.status-good {
		background-color: #10b981;
	}

	.status-warning {
		background-color: #f59e0b;
	}

	.status-error {
		background-color: #ef4444;
	}

	.health-value {
		margin-left: auto;
		color: #10b981;
		font-weight: 500;
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 3rem;
		text-align: center;
		color: #9ca3af;
	}

	.empty-state h3 {
		margin: 1rem 0 0.5rem 0;
		color: white;
	}

	.empty-state-small {
		padding: 2rem;
		text-align: center;
		color: #9ca3af;
	}

	.btn-secondary-sm {
		padding: 0.375rem 0.75rem;
		background-color: #4b5563;
		color: #d1d5db;
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-weight: 500;
		transition: background-color 0.2s;
		display: flex;
		align-items: center;
		gap: 0.25rem;
		border: none;
		cursor: pointer;
	}

	.btn-secondary-sm:hover {
		background-color: #6b7280;
	}

	.btn-danger-sm {
		padding: 0.375rem 0.75rem;
		background-color: #dc2626;
		color: white;
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-weight: 500;
		transition: background-color 0.2s;
		display: flex;
		align-items: center;
		gap: 0.25rem;
		border: none;
		cursor: pointer;
	}

	.btn-danger-sm:hover {
		background-color: #b91c1c;
	}
</style>