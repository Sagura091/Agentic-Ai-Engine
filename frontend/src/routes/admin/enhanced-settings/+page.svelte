<script lang="ts">
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { authStore } from '$lib/stores/auth';
	import {
		Settings,
		Shield,
		Brain,
		Workflow,
		Target,
		Wrench,
		Globe,
		Archive,
		Bell,
		ChevronRight,
		ChevronDown,
		Save,
		RefreshCw,
		AlertTriangle,
		CheckCircle,
		XCircle,
		Info,
		Bot,
		HardDrive,
		Activity,
		Download,
		Eye,
		FileText,
		Search,
		Zap,
		Database,
		Cpu,
		BarChart3,
		Layers,
		Sparkles
	} from 'lucide-svelte';
	import ModelDownloadModal from '$lib/components/ModelDownloadModal.svelte';

	// ============================================================================
	// TYPES AND INTERFACES
	// ============================================================================

	interface SettingCategory {
		id: string;
		name: string;
		description: string;
		icon: string;
		color: string;
	}

	interface CategoryGroup {
		name: string;
		description: string;
		categories: SettingCategory[];
	}

	interface SettingDefinition {
		key: string;
		type: string;
		value: any;
		default: any;
		description: string;
		requires_restart: boolean;
		is_sensitive: boolean;
		enum_values?: string[];
		min_value?: number;
		max_value?: number;
		validation_rules?: Record<string, any>;
	}

	// ============================================================================
	// STORES AND STATE
	// ============================================================================

	const loading = writable(false);
	const error = writable<string | null>(null);
	const success = writable<string | null>(null);
	const categories = writable<Record<string, CategoryGroup>>({});
	const activeCategory = writable<string>('system_configuration');
	const categorySettings = writable<Record<string, SettingDefinition>>({});
	const expandedGroups = writable<Set<string>>(new Set(['core']));
	const unsavedChanges = writable<Record<string, any>>({});
	const validationErrors = writable<Record<string, string[]>>({});

	// 🚀 Revolutionary RAG System Stores
	const modelDownloadModal = writable<{
		isOpen: boolean;
		modelType: 'embedding' | 'vision' | 'reranking';
		currentModel: string;
	}>({
		isOpen: false,
		modelType: 'embedding',
		currentModel: ''
	});

	const ragTemplates = writable<Record<string, any>>({});
	const availableModels = writable<Record<string, any[]>>({
		embedding: [],
		vision: [],
		reranking: []
	});

	// 🚀 Revolutionary RAG Tabs System
	const ragActiveTab = writable<string>('models');
	const ragTabs = [
		{ id: 'models', name: 'Models & Downloads', icon: 'Download', description: 'Embedding, Vision & Reranking Models' },
		{ id: 'ocr', name: 'OCR Configuration', icon: 'FileText', description: 'Tesseract, EasyOCR, PaddleOCR' },
		{ id: 'processing', name: 'Document Processing', icon: 'FileSearch', description: 'Chunking, Preprocessing, File Handling' },
		{ id: 'retrieval', name: 'Retrieval & Search', icon: 'Search', description: 'Top-k, Scoring, Hybrid Search' },
		{ id: 'performance', name: 'Performance', icon: 'Zap', description: 'Caching, Concurrency, Optimization' },
		{ id: 'templates', name: 'Templates', icon: 'Sparkles', description: 'Pre-configured RAG Templates' }
	];

	// ============================================================================
	// REACTIVE DECLARATIONS
	// ============================================================================

	$: user = $authStore.user;
	$: isAdmin = user?.user_group === 'admin';

	// ============================================================================
	// API FUNCTIONS
	// ============================================================================

	async function fetchCategories() {
		try {
			loading.set(true);
			const response = await fetch('/api/v1/admin/enhanced-settings/categories', {
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				}
			});

			if (!response.ok) {
				throw new Error(`Failed to fetch categories: ${response.statusText}`);
			}

			const result = await response.json();
			if (result.success) {
				categories.set(result.data);
			} else {
				throw new Error(result.message || 'Failed to fetch categories');
			}
		} catch (err) {
			console.error('Error fetching categories:', err);
			error.set(err instanceof Error ? err.message : 'Failed to fetch categories');
		} finally {
			loading.set(false);
		}
	}

	async function fetchCategorySettings(categoryId: string) {
		try {
			loading.set(true);
			const response = await fetch(`/api/v1/admin/enhanced-settings/values/${categoryId}`, {
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				}
			});

			if (!response.ok) {
				throw new Error(`Failed to fetch settings: ${response.statusText}`);
			}

			const result = await response.json();
			if (result.success) {
				categorySettings.set(result.data);
				// Clear unsaved changes when loading new category
				unsavedChanges.set({});
				validationErrors.set({});
			} else {
				throw new Error(result.message || 'Failed to fetch settings');
			}
		} catch (err) {
			console.error('Error fetching category settings:', err);
			error.set(err instanceof Error ? err.message : 'Failed to fetch settings');
		} finally {
			loading.set(false);
		}
	}

	async function validateSetting(key: string, value: any, categoryId: string) {
		try {
			const response = await fetch('/api/v1/admin/enhanced-settings/validate', {
				method: 'POST',
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					category: categoryId,
					key: key,
					value: value
				})
			});

			if (!response.ok) {
				throw new Error(`Validation failed: ${response.statusText}`);
			}

			const result = await response.json();
			return result.data;
		} catch (err) {
			console.error('Error validating setting:', err);
			return { is_valid: false, errors: ['Validation request failed'] };
		}
	}

	async function saveSetting(key: string, value: any, categoryId: string) {
		try {
			const response = await fetch('/api/v1/admin/enhanced-settings/update', {
				method: 'POST',
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					category: categoryId,
					key: key,
					value: value
				})
			});

			if (!response.ok) {
				throw new Error(`Failed to save setting: ${response.statusText}`);
			}

			const result = await response.json();
			if (result.success) {
				// Remove from unsaved changes
				unsavedChanges.update(changes => {
					const newChanges = { ...changes };
					delete newChanges[key];
					return newChanges;
				});

				// Clear validation errors for this setting
				validationErrors.update(errors => {
					const newErrors = { ...errors };
					delete newErrors[key];
					return newErrors;
				});

				success.set(`Setting "${key}" saved successfully`);
				
				if (result.data.requires_restart) {
					success.set(`Setting "${key}" saved successfully. System restart required.`);
				}

				// Refresh the category settings
				await fetchCategorySettings(categoryId);
			} else {
				throw new Error(result.message || 'Failed to save setting');
			}
		} catch (err) {
			console.error('Error saving setting:', err);
			error.set(err instanceof Error ? err.message : 'Failed to save setting');
		}
	}

	// ============================================================================
	// EVENT HANDLERS
	// ============================================================================

	function handleCategorySelect(categoryId: string) {
		activeCategory.set(categoryId);
		fetchCategorySettings(categoryId);
	}

	function toggleGroup(groupId: string) {
		expandedGroups.update(groups => {
			const newGroups = new Set(groups);
			if (newGroups.has(groupId)) {
				newGroups.delete(groupId);
			} else {
				newGroups.add(groupId);
			}
			return newGroups;
		});
	}

	async function handleSettingChange(key: string, value: any) {
		// Update unsaved changes
		unsavedChanges.update(changes => ({
			...changes,
			[key]: value
		}));

		// Validate the setting
		const validationResult = await validateSetting(key, value, $activeCategory);
		
		if (!validationResult.is_valid) {
			validationErrors.update(errors => ({
				...errors,
				[key]: validationResult.errors
			}));
		} else {
			validationErrors.update(errors => {
				const newErrors = { ...errors };
				delete newErrors[key];
				return newErrors;
			});
		}
	}

	async function handleSaveSetting(key: string) {
		const value = $unsavedChanges[key];
		if (value !== undefined) {
			await saveSetting(key, value, $activeCategory);
		}
	}

	async function handleSaveAll() {
		const changes = $unsavedChanges;
		for (const [key, value] of Object.entries(changes)) {
			await saveSetting(key, value, $activeCategory);
		}
	}

	function clearMessages() {
		error.set(null);
		success.set(null);
	}

	// ============================================================================
	// 🚀 REVOLUTIONARY RAG SYSTEM FUNCTIONS
	// ============================================================================

	function openModelDownloadModal(modelType: 'embedding' | 'vision' | 'reranking', currentModel: string = '') {
		modelDownloadModal.set({
			isOpen: true,
			modelType,
			currentModel
		});
	}

	function closeModelDownloadModal() {
		modelDownloadModal.update(state => ({
			...state,
			isOpen: false
		}));
	}

	async function handleModelDownloadComplete(event: CustomEvent) {
		const { modelType, modelId, modelName } = event.detail;

		// Update the current model setting
		const settingKey = modelType === 'embedding' ? 'embedding_model' :
						  modelType === 'vision' ? 'primary_vision_model' : 'reranking_model';

		await handleSettingChange(`rag_configuration.${settingKey}`, modelId);

		// Show success message
		success.set(`${modelName} downloaded and activated successfully!`);

		// Close modal
		closeModelDownloadModal();

		// Clear success message after 5 seconds
		setTimeout(() => success.set(null), 5000);
	}

	async function loadRAGTemplates() {
		try {
			// TODO: Replace with actual API call
			const templates = {
				general_purpose: {
					name: 'General Purpose',
					description: 'Balanced settings for general knowledge retrieval',
					settings: {
						top_k: 10,
						score_threshold: 0.7,
						enable_reranking: true,
						enable_query_expansion: true
					}
				},
				research_assistant: {
					name: 'Research Assistant',
					description: 'Optimized for academic and research tasks',
					settings: {
						top_k: 15,
						score_threshold: 0.8,
						enable_reranking: true,
						enable_query_expansion: true,
						enable_contextual_retrieval: true
					}
				},
				code_helper: {
					name: 'Code Helper',
					description: 'Specialized for programming and technical documentation',
					settings: {
						top_k: 8,
						score_threshold: 0.75,
						enable_reranking: true,
						chunking_strategy: 'semantic'
					}
				}
			};

			ragTemplates.set(templates);
		} catch (err) {
			console.error('Failed to load RAG templates:', err);
		}
	}

	async function applyRAGTemplate(templateId: string) {
		try {
			const templates = $ragTemplates;
			const template = templates[templateId];

			if (!template) {
				throw new Error('Template not found');
			}

			// Apply template settings
			for (const [key, value] of Object.entries(template.settings)) {
				await handleSettingChange(`rag_configuration.${key}`, value);
			}

			success.set(`Applied ${template.name} template successfully!`);
			setTimeout(() => success.set(null), 5000);

		} catch (err) {
			error.set(`Failed to apply template: ${err.message}`);
			setTimeout(() => error.set(null), 5000);
		}
	}

	// ============================================================================
	// 🚀 REVOLUTIONARY RAG TABS FUNCTIONS
	// ============================================================================

	function filterRAGSettingsByTab(settings: Record<string, SettingDefinition>, tabId: string): Record<string, SettingDefinition> {
		const filtered: Record<string, SettingDefinition> = {};

		for (const [key, setting] of Object.entries(settings)) {
			const shouldInclude = (() => {
				switch (tabId) {
					case 'models':
						return key.includes('embedding_model') || key.includes('vision_model') || key.includes('reranking') || key.includes('model_');
					case 'ocr':
						return key.includes('ocr') || key.includes('tesseract') || key.includes('easyocr') || key.includes('paddleocr');
					case 'processing':
						return key.includes('chunk') || key.includes('preprocessing') || key.includes('file_') || key.includes('document_');
					case 'retrieval':
						return key.includes('top_k') || key.includes('score_') || key.includes('search') || key.includes('retrieval') || key.includes('hybrid');
					case 'performance':
						return key.includes('cache') || key.includes('concurrent') || key.includes('batch') || key.includes('performance') || key.includes('optimization');
					case 'templates':
						return false; // Templates are handled separately
					default:
						return true;
				}
			})();

			if (shouldInclude) {
				filtered[key] = setting;
			}
		}

		return filtered;
	}

	function selectRAGTab(tabId: string) {
		ragActiveTab.set(tabId);
	}

	// ============================================================================
	// LIFECYCLE
	// ============================================================================

	onMount(async () => {
		// Check if user is admin
		if (!isAdmin) {
			goto('/dashboard');
			return;
		}

		// Load initial data
		await fetchCategories();

		// Load default category settings
		if ($activeCategory) {
			await fetchCategorySettings($activeCategory);
		}

		// Load RAG-specific data
		await loadRAGTemplates();

		// Clear messages after 5 seconds
		const messageTimer = setInterval(() => {
			clearMessages();
		}, 5000);

		return () => {
			clearInterval(messageTimer);
		};
	});

	// ============================================================================
	// HELPER FUNCTIONS
	// ============================================================================

	function getIconComponent(iconName: string) {
		const iconMap: Record<string, any> = {
			'🔧': Settings,
			'🔐': Shield,
			'🤖': Bot,
			'🧠': Brain,
			'🔄': Workflow,
			'🎯': Target,
			'🗄️': HardDrive,
			'📊': Activity,
			'🛠️': Wrench,
			'🌐': Globe,
			'📦': Archive,
			'📧': Bell
		};
		return iconMap[iconName] || Settings;
	}


</script>

<!-- ============================================================================ -->
<!-- STYLES -->
<!-- ============================================================================ -->
<style>
	.glass {
		background: rgba(255, 255, 255, 0.05);
		backdrop-filter: blur(20px);
		-webkit-backdrop-filter: blur(20px);
		border: 1px solid rgba(255, 255, 255, 0.1);
	}

	.line-clamp-2 {
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
</style>

<!-- ============================================================================ -->
<!-- TEMPLATE -->
<!-- ============================================================================ -->

<!-- Revolutionary Compact Settings Interface -->
<div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
	<!-- Compact Header -->
	<div class="glass border-b border-white/10 sticky top-0 z-50">
		<div class="max-w-7xl mx-auto px-4 py-3">
			<div class="flex items-center justify-between">
				<div class="flex items-center space-x-3">
					<div class="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white">
						<Settings class="w-5 h-5" />
					</div>
					<div>
						<h1 class="text-xl font-bold text-white">Enhanced Settings</h1>
						<p class="text-xs text-white/60">System Configuration</p>
					</div>
				</div>

				<!-- Compact Status Bar -->
				<div class="flex items-center space-x-2">
					{#if Object.keys($unsavedChanges).length > 0}
						<div class="flex items-center space-x-1 px-2 py-1 bg-amber-500/20 text-amber-300 rounded-md text-xs">
							<AlertTriangle class="w-3 h-3" />
							<span>{Object.keys($unsavedChanges).length}</span>
						</div>
					{/if}

					{#if $loading}
						<div class="flex items-center space-x-1 px-2 py-1 bg-blue-500/20 text-blue-300 rounded-md text-xs">
							<RefreshCw class="w-3 h-3 animate-spin" />
						</div>
					{/if}

					<!-- Quick Save All Button -->
					{#if Object.keys($unsavedChanges).length > 0}
						<button
							on:click={handleSaveAll}
							class="px-3 py-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white text-xs rounded-md transition-all duration-200 flex items-center space-x-1"
						>
							<Save class="w-3 h-3" />
							<span>Save All</span>
						</button>
					{/if}
				</div>
			</div>
		</div>
	</div>

	<!-- Compact Messages -->
	{#if $error}
		<div class="max-w-7xl mx-auto px-4 py-2">
			<div class="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center space-x-2">
				<XCircle class="w-4 h-4 text-red-400 flex-shrink-0" />
				<p class="text-red-300 text-sm flex-1">{$error}</p>
				<button on:click={clearMessages} class="text-red-400 hover:text-red-300">
					<XCircle class="w-3 h-3" />
				</button>
			</div>
		</div>
	{/if}

	{#if $success}
		<div class="max-w-7xl mx-auto px-4 py-2">
			<div class="bg-green-500/10 border border-green-500/20 rounded-lg p-3 flex items-center space-x-2">
				<CheckCircle class="w-4 h-4 text-green-400 flex-shrink-0" />
				<p class="text-green-300 text-sm flex-1">{$success}</p>
				<button on:click={clearMessages} class="text-green-400 hover:text-green-300">
					<XCircle class="w-3 h-3" />
				</button>
			</div>
		</div>
	{/if}

	<!-- Revolutionary Compact Main Content -->
	<div class="max-w-7xl mx-auto px-4 py-4">
		<div class="grid grid-cols-1 lg:grid-cols-5 gap-4">
			<!-- Compact Sidebar - Categories -->
			<div class="lg:col-span-1">
				<div class="glass rounded-xl border border-white/10 p-4 sticky top-20">
					<h2 class="text-sm font-semibold text-white mb-3 flex items-center space-x-2">
						<Settings class="w-4 h-4" />
						<span>Categories</span>
					</h2>

					{#if Object.keys($categories).length > 0}
						{#each Object.entries($categories) as [groupId, group]}
							<div class="mb-3">
								<button
									on:click={() => toggleGroup(groupId)}
									class="w-full flex items-center justify-between p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-white/80 hover:text-white"
								>
									<span class="font-medium text-sm">{group.name}</span>
									{#if $expandedGroups.has(groupId)}
										<ChevronDown class="w-3 h-3" />
									{:else}
										<ChevronRight class="w-3 h-3" />
									{/if}
								</button>

								{#if $expandedGroups.has(groupId)}
									<div class="mt-1 space-y-1 pl-1">
										{#each group.categories as category}
											<button
												on:click={() => handleCategorySelect(category.id)}
												class="w-full flex items-center space-x-2 p-2 rounded-lg transition-all duration-200 text-left {$activeCategory === category.id
													? 'bg-gradient-to-r from-purple-500/30 to-pink-500/30 text-white border border-purple-400/30'
													: 'hover:bg-white/5 text-white/70 hover:text-white'}"
											>
												<svelte:component
													this={getIconComponent(category.icon)}
													class="w-3 h-3 flex-shrink-0"
												/>
												<div class="min-w-0 flex-1">
													<div class="font-medium text-xs truncate">
														{category.name}
													</div>
													<div class="text-xs opacity-60 truncate">
														{category.description}
													</div>
												</div>
											</button>
										{/each}
									</div>
								{/if}
							</div>
						{/each}
					{:else}
						<div class="text-center py-6">
							<Settings class="w-8 h-8 text-white/40 mx-auto mb-2" />
							<p class="text-white/60 text-xs">Loading...</p>
						</div>
					{/if}
				</div>
			</div>

			<!-- Revolutionary Compact Settings Panel -->
			<div class="lg:col-span-4">
				<div class="glass rounded-xl border border-white/10 p-4">
					{#if Object.keys($categorySettings).length > 0}

						{#if $activeCategory === 'rag_configuration'}
							<!-- 🚀 Revolutionary RAG Tabbed Interface -->
							<div class="space-y-4">
								<!-- RAG Tabs Navigation -->
								<div class="flex flex-wrap gap-2 p-1 bg-white/5 rounded-lg border border-white/10">
									{#each ragTabs as tab}
										<button
											on:click={() => selectRAGTab(tab.id)}
											class="flex-1 min-w-0 px-3 py-2 rounded-md text-xs font-medium transition-all duration-200 {$ragActiveTab === tab.id
												? 'bg-gradient-to-r from-purple-500/30 to-pink-500/30 text-white border border-purple-400/30'
												: 'text-white/70 hover:text-white hover:bg-white/5'}"
										>
											<div class="flex items-center justify-center space-x-1">
												<svelte:component this={getIconComponent(tab.icon)} class="w-3 h-3" />
												<span class="truncate">{tab.name}</span>
											</div>
										</button>
									{/each}
								</div>

								<!-- RAG Tab Content -->
								<div class="min-h-[400px]">
									{#if $ragActiveTab === 'templates'}
										<!-- RAG Templates Tab -->
										{#if Object.keys($ragTemplates).length > 0}
											<div class="space-y-4">
												<div class="flex items-center space-x-2 mb-4">
													<Sparkles class="w-5 h-5 text-purple-400" />
													<h3 class="text-lg font-semibold text-white">RAG Templates</h3>
													<span class="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded-full">Revolutionary</span>
												</div>
												<p class="text-sm text-white/60 mb-4">
													Apply pre-configured RAG templates optimized for different use cases. These templates automatically configure multiple settings for optimal performance.
												</p>

												<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
													{#each Object.entries($ragTemplates) as [templateId, template]}
														<div class="bg-white/5 rounded-lg p-3 border border-white/10 hover:border-purple-400/30 transition-all duration-200">
															<div class="flex items-start justify-between mb-2">
																<div class="flex-1">
																	<h4 class="font-medium text-white text-sm">{template.name}</h4>
																	<p class="text-xs text-white/60 mt-1">{template.description}</p>
																</div>
															</div>

															<!-- Template Settings Preview -->
															<div class="text-xs text-white/50 mb-3">
																{#each Object.entries(template.settings).slice(0, 3) as [key, value]}
																	<div class="flex justify-between">
																		<span>{key.replace(/_/g, ' ')}</span>
																		<span>{value}</span>
																	</div>
																{/each}
																{#if Object.keys(template.settings).length > 3}
																	<div class="text-center text-white/40 mt-1">
																		+{Object.keys(template.settings).length - 3} more settings
																	</div>
																{/if}
															</div>

															<button
																on:click={() => applyRAGTemplate(templateId)}
																class="w-full px-3 py-2 bg-gradient-to-r from-purple-500/20 to-pink-500/20 hover:from-purple-500/30 hover:to-pink-500/30 border border-purple-400/30 text-purple-300 rounded-lg transition-all duration-200 text-xs flex items-center justify-center space-x-1"
															>
																<Zap class="w-3 h-3" />
																<span>Apply Template</span>
															</button>
														</div>
													{/each}
												</div>
											</div>
										{:else}
											<div class="text-center py-12">
												<Sparkles class="w-12 h-12 text-white/20 mx-auto mb-4" />
												<h3 class="text-lg font-medium text-white mb-2">No Templates Available</h3>
												<p class="text-white/60 text-sm">RAG templates are loading...</p>
											</div>
										{/if}
									{:else}
										<!-- RAG Settings Tab Content -->
										{@const filteredSettings = filterRAGSettingsByTab($categorySettings, $ragActiveTab)}
										{#if Object.keys(filteredSettings).length > 0}
											<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
												{#each Object.entries(filteredSettings) as [key, setting]}
								<div class="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-white/20 transition-all duration-200">
									<div class="flex items-start justify-between mb-3">
										<div class="flex-1 min-w-0">
											<div class="flex items-center space-x-2 mb-1">
												<label for={key} class="text-sm font-medium text-white truncate">
													{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
												</label>
												{#if setting.requires_restart}
													<div class="flex items-center space-x-1">
														<AlertTriangle class="w-3 h-3 text-amber-400" />
														<span class="text-xs text-amber-300">Restart</span>
													</div>
												{/if}
											</div>
											<p class="text-xs text-white/60 line-clamp-2">
												{setting.description}
											</p>
										</div>

										{#if $unsavedChanges[key] !== undefined}
											<button
												on:click={() => handleSaveSetting(key)}
												class="ml-2 px-2 py-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white text-xs rounded-md transition-all duration-200 flex items-center space-x-1 flex-shrink-0"
											>
												<Save class="w-3 h-3" />
												<span>Save</span>
											</button>
										{/if}
									</div>
									
									<!-- Revolutionary RAG Settings Input -->
									<div class="mt-2">
										{#if key.includes('embedding_model') || key.includes('vision_model')}
											<!-- Model Selection with Download Button -->
											<div class="flex items-center space-x-2">
												<select
													id={key}
													value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
													on:change={(e) => handleSettingChange(key, e.currentTarget.value)}
													class="flex-1 px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
												>
													{#if setting.enum_values}
														{#each setting.enum_values as option}
															<option value={option} class="bg-slate-800 text-white">{option}</option>
														{/each}
													{:else}
														<option value={setting.value} class="bg-slate-800 text-white">{setting.value}</option>
													{/if}
												</select>
												<button
													on:click={() => openModelDownloadModal(
														key.includes('embedding') ? 'embedding' : 'vision',
														$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value
													)}
													class="px-3 py-2 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white rounded-lg transition-all duration-200 flex items-center space-x-1 text-xs"
													title="Download Model"
												>
													<Download class="w-3 h-3" />
													<span>Download</span>
												</button>
											</div>

										{:else if setting.type === 'boolean'}
											<label for={key} class="flex items-center space-x-2 cursor-pointer">
												<div class="relative">
													<input
														id={key}
														type="checkbox"
														checked={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
														on:change={(e) => handleSettingChange(key, e.currentTarget.checked)}
														class="sr-only"
													/>
													<div class="w-10 h-5 bg-white/20 rounded-full transition-colors duration-200 {($unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value) ? 'bg-gradient-to-r from-green-500 to-emerald-500' : 'bg-white/20'}">
														<div class="w-4 h-4 bg-white rounded-full shadow-md transform transition-transform duration-200 {($unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value) ? 'translate-x-5' : 'translate-x-0.5'} mt-0.5"></div>
													</div>
												</div>
												<span class="text-xs text-white/80">
													{$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value ? 'Enabled' : 'Disabled'}
												</span>
											</label>

										{:else if setting.type === 'enum' && setting.enum_values}
											<select
												id={key}
												value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
												on:change={(e) => handleSettingChange(key, e.currentTarget.value)}
												class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
											>
												{#each setting.enum_values as option}
													<option value={option} class="bg-slate-800 text-white">{option}</option>
												{/each}
											</select>

										{:else if setting.type === 'integer'}
											<input
												id={key}
												type="number"
												value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
												min={setting.min_value}
												max={setting.max_value}
												on:input={(e) => handleSettingChange(key, parseInt(e.currentTarget.value))}
												class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
											/>
										{:else if setting.type === 'float'}
											<input
												id={key}
												type="number"
												step="0.1"
												value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
												min={setting.min_value}
												max={setting.max_value}
												on:input={(e) => handleSettingChange(key, parseFloat(e.currentTarget.value))}
												class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
											/>
										{:else}
											<input
												id={key}
												type={setting.is_sensitive ? 'password' : 'text'}
												value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
												on:input={(e) => handleSettingChange(key, e.currentTarget.value)}
												class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
											/>
										{/if}
										
										<!-- Compact Validation Errors -->
										{#if $validationErrors[key]}
											<div class="mt-2 space-y-1">
												{#each $validationErrors[key] as error}
													<div class="flex items-center space-x-1 text-red-400 text-xs">
														<XCircle class="w-3 h-3 flex-shrink-0" />
														<span>{error}</span>
													</div>
												{/each}
											</div>
										{/if}

										<!-- Compact Default Value Info -->
										<div class="flex items-center space-x-1 text-xs text-white/40 mt-1">
											<Info class="w-3 h-3" />
											<span>Default: {JSON.stringify(setting.default)}</span>
										</div>
									</div>
								</div>
													{/each}
												</div>
											{:else}
												<div class="text-center py-12">
													<div class="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
														<Settings class="w-8 h-8 text-white/40" />
													</div>
													<h3 class="text-lg font-medium text-white mb-2">No Settings in This Tab</h3>
													<p class="text-white/60 text-sm">This tab doesn't contain any settings yet.</p>
												</div>
											{/if}
										{/if}
									</div>
								</div>
							{:else}
								<!-- Standard Settings Grid for Non-RAG Categories -->
								<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
									{#each Object.entries($categorySettings) as [key, setting]}
										<div class="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-white/20 transition-all duration-200">
											<div class="flex items-start justify-between mb-3">
												<div class="flex-1 min-w-0">
													<div class="flex items-center space-x-2 mb-1">
														<label for={key} class="text-sm font-medium text-white truncate">
															{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
														</label>
														{#if setting.requires_restart}
															<div class="flex items-center space-x-1">
																<AlertTriangle class="w-3 h-3 text-amber-400" />
																<span class="text-xs text-amber-300">Restart</span>
															</div>
														{/if}
													</div>
													<p class="text-xs text-white/60 line-clamp-2">
														{setting.description}
													</p>
												</div>

												{#if $unsavedChanges[key] !== undefined}
													<button
														on:click={() => handleSaveSetting(key)}
														class="ml-2 px-2 py-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white text-xs rounded-md transition-all duration-200 flex items-center space-x-1 flex-shrink-0"
													>
														<Save class="w-3 h-3" />
														<span>Save</span>
													</button>
												{/if}
											</div>

											<!-- Standard Settings Input -->
											<div class="mt-2">
												{#if setting.type === 'boolean'}
													<label for={key} class="flex items-center space-x-2 cursor-pointer">
														<div class="relative">
															<input
																id={key}
																type="checkbox"
																checked={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
																on:change={(e) => handleSettingChange(key, e.currentTarget.checked)}
																class="sr-only"
															/>
															<div class="w-10 h-5 bg-white/20 rounded-full transition-colors duration-200 {($unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value) ? 'bg-gradient-to-r from-green-500 to-emerald-500' : 'bg-white/20'}">
																<div class="w-4 h-4 bg-white rounded-full shadow-md transform transition-transform duration-200 {($unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value) ? 'translate-x-5' : 'translate-x-0.5'} mt-0.5"></div>
															</div>
														</div>
														<span class="text-xs text-white/80">
															{$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value ? 'Enabled' : 'Disabled'}
														</span>
													</label>

												{:else if setting.type === 'enum' && setting.enum_values}
													<select
														id={key}
														value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
														on:change={(e) => handleSettingChange(key, e.currentTarget.value)}
														class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
													>
														{#each setting.enum_values as option}
															<option value={option} class="bg-slate-800 text-white">{option}</option>
														{/each}
													</select>

												{:else if setting.type === 'integer'}
													<input
														id={key}
														type="number"
														value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
														min={setting.min_value}
														max={setting.max_value}
														on:input={(e) => handleSettingChange(key, parseInt(e.currentTarget.value))}
														class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
													/>
												{:else if setting.type === 'float'}
													<input
														id={key}
														type="number"
														step="0.1"
														value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
														min={setting.min_value}
														max={setting.max_value}
														on:input={(e) => handleSettingChange(key, parseFloat(e.currentTarget.value))}
														class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
													/>
												{:else}
													<input
														id={key}
														type={setting.is_sensitive ? 'password' : 'text'}
														value={$unsavedChanges[key] !== undefined ? $unsavedChanges[key] : setting.value}
														on:input={(e) => handleSettingChange(key, e.currentTarget.value)}
														class="w-full px-3 py-2 text-sm bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
													/>
												{/if}
											</div>

											<!-- Validation Errors -->
											{#if $validationErrors[key] && $validationErrors[key].length > 0}
												<div class="mt-2 space-y-1">
													{#each $validationErrors[key] as errorMsg}
														<p class="text-xs text-red-400 flex items-center space-x-1">
															<AlertTriangle class="w-3 h-3" />
															<span>{errorMsg}</span>
														</p>
													{/each}
												</div>
											{/if}
										</div>
									{/each}
								</div>
							{/if}

					{:else}
						<!-- Modern Empty State -->
						<div class="text-center py-12">
							<div class="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
								<Settings class="w-8 h-8 text-white/40" />
							</div>
							<h3 class="text-lg font-medium text-white mb-2">Select a Category</h3>
							<p class="text-white/60 text-sm">Choose a setting category to configure system options.</p>
						</div>
					{/if}
				</div>
			</div>
		</div>
	</div>
</div>

<!-- 🚀 Revolutionary Model Download Modal -->
<ModelDownloadModal
	bind:isOpen={$modelDownloadModal.isOpen}
	modelType={$modelDownloadModal.modelType}
	currentModel={$modelDownloadModal.currentModel}
	on:close={closeModelDownloadModal}
	on:download-complete={handleModelDownloadComplete}
/>
