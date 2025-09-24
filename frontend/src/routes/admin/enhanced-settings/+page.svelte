<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { writable, get } from 'svelte/store';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { authStore } from '$lib/stores/auth';
	import {
		configurationChanges,
		type ConfigurationChange
	} from '$lib/services/websocket';

	// Accept params prop to avoid Svelte warnings
	export let params: Record<string, string> = {};
	import {
		Settings,
		Activity
	} from 'lucide-svelte';
	import ModelDownloadModal from '$lib/components/ModelDownloadModal.svelte';
	import RAGSettingsPanel from '$lib/components/admin/RAGSettingsPanel.svelte';
	import LLMProvidersPanel from '$lib/components/admin/LLMProvidersPanel.svelte';
	import DatabaseStoragePanel from '$lib/components/admin/DatabaseStoragePanel.svelte';
	import SettingsCategoryPanel from '$lib/components/admin/SettingsCategoryPanel.svelte';

	// ============================================================================
	// TYPES AND INTERFACES - IMPORTED FROM CENTRALIZED TYPES
	// ============================================================================

	// All types are now imported from './types/EnhancedSettingsTypes'

	// ============================================================================
	// STORES AND STATE - IMPORTED FROM CENTRALIZED STORE
	// ============================================================================

	import { enhancedSettingsStore } from './stores/EnhancedSettingsStore';
	import { enhancedSettingsAPI } from './services/EnhancedSettingsAPI';
	import { RAGSettingsManager } from './managers/RAGSettingsManager';
	import { LLMProvidersManager } from './managers/LLMProvidersManager';
	import { DatabaseStorageManager } from './managers/DatabaseStorageManager';

	// Import UI components
	import EnhancedSettingsHeader from './components/EnhancedSettingsHeader.svelte';
	import EnhancedSettingsSidebar from './components/EnhancedSettingsSidebar.svelte';
	import EnhancedSettingsMessages from './components/EnhancedSettingsMessages.svelte';
	import EnhancedSettingsStatusBar from './components/EnhancedSettingsStatusBar.svelte';

	import type {
		SettingCategory,
		CategoryGroup,
		SettingDefinition
	} from './types/EnhancedSettingsTypes';

	// Destructure all stores from centralized store
	const {
		// Core stores
		loading,
		error,
		success,
		categories,
		activeCategory,
		categorySettings,
		expandedGroups,
		unsavedChanges,
		validationErrors,

		// RAG System stores
		modelDownloadModal,
		ragTemplates,
		availableModels,
		ragActiveTab,
		ragTabs,

		// LLM Provider stores
		llmActiveTab,
		llmTabs,
		llmProviderStatus,
		llmAvailableModels,
		llmProviderTemplates,
		ollamaConnectionStatus,
		llmUnsavedChanges,
		llmHasUnsavedChanges,
		llmSaving,
		llmSaveStatus,

		// Database Storage stores
		databaseActiveTab,
		databaseTabs,
		databaseConnectionStatus,
		databaseUnsavedChanges
	} = enhancedSettingsStore;

	// LLM original values tracking
	let llmOriginalValues: Record<string, any> = {};

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

				// Initialize LLM original values for change tracking
				if (categoryId === 'llm_providers') {
					initializeLLMOriginalValues(result.data);
				}

				// Initialize Database Storage original values for change tracking
				if (categoryId === 'database_storage') {
					initializeDatabaseOriginalValues(result.data);
				}
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
				// First, refresh the category settings to get the updated values
				await fetchCategorySettings(categoryId);

				// Then remove from unsaved changes (after we have the new values loaded)
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
		// Note: saveSetting already reloads settings and clears individual changes
		// Just ensure all changes are cleared and reload once more to be safe
		unsavedChanges.set({});
		validationErrors.set({});
		await fetchCategorySettings($activeCategory);
	}

	async function handleDatabaseSettingChange(key: string, value: any) {
		// Update database unsaved changes
		databaseUnsavedChanges.update(changes => ({
			...changes,
			[key]: value
		}));

		// Validate the setting
		const fullKey = `database_storage.${key}`;
		const validationResult = await validateSetting(fullKey, value, 'database_storage');

		if (!validationResult.is_valid) {
			validationErrors.update(errors => ({
				...errors,
				[fullKey]: validationResult.errors
			}));
		} else {
			validationErrors.update(errors => {
				const newErrors = { ...errors };
				delete newErrors[fullKey];
				return newErrors;
			});
		}
	}

	function clearMessages() {
		enhancedSettingsStore.clearMessages();
	}

	// ============================================================================
	// COMPONENT EVENT HANDLERS
	// ============================================================================

	// ============================================================================
	// COMPONENT EVENT HANDLERS - REMOVED DUPLICATES
	// ============================================================================

	// 🚀 Real-time configuration change handler
	function handleRealTimeConfigurationChange(change: ConfigurationChange) {
		console.log('⚙️ Real-time configuration change:', change);

		// Show notification to user
		const message = `Configuration updated by ${change.admin_user}: ${change.section}.${change.setting_key}`;

		// Add visual indicator
		success.set(`🔄 ${message}`);

		// Auto-refresh the affected category if it's currently active
		if (change.section === $activeCategory) {
			setTimeout(async () => {
				await fetchCategorySettings($activeCategory);
				success.set(`✅ Settings refreshed - ${change.section} updated by ${change.admin_user}`);
			}, 1000);
		}

		// Clear the message after 8 seconds (longer for real-time updates)
		setTimeout(() => {
			success.set(null);
		}, 8000);
	}

	// ============================================================================
	// 🚀 REVOLUTIONARY RAG SYSTEM FUNCTIONS
	// ============================================================================

	function openModelDownloadModal(modelType: 'embedding' | 'vision' | 'reranking', currentModel: string = '') {
		if (ragManager) {
			ragManager.openModelDownloadModal(modelType, currentModel);
		}
	}

	function closeModelDownloadModal() {
		if (ragManager) {
			ragManager.closeModelDownloadModal();
		}
	}

	async function handleModelDownloadComplete(event: CustomEvent) {
		const { modelType, modelId, modelName } = event.detail;
		if (ragManager) {
			await ragManager.handleModelDownloadComplete(modelType, modelId, modelName);
		}
	}

	async function loadRAGTemplates() {
		if (ragManager) {
			await ragManager.loadRAGTemplates();
		}
	}

	async function applyRAGTemplate(templateId: string) {
		if (ragManager) {
			await ragManager.applyRAGTemplate(templateId);
		}
	}

	async function downloadEmbeddingModel(modelId: string) {
		try {
			addMessage('info', `Starting download of embedding model ${modelId}...`);

			const response = await fetch('/api/v1/embedding-models/download', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${$authStore.token}`
				},
				body: JSON.stringify({
					model_id: modelId,
					force_redownload: false
				})
			});

			const result = await response.json();

			if (result.success) {
				addMessage('success', `Embedding model ${modelId} download started successfully`);
			} else {
				addMessage('error', result.message || `Failed to download embedding model ${modelId}`);
			}
		} catch (error) {
			console.error('Error downloading embedding model:', error);
			addMessage('error', `Error downloading embedding model ${modelId}: ${error.message}`);
		}
	}

	async function downloadVisionModel(modelId: string) {
		try {
			addMessage('info', `Starting download of vision model ${modelId}...`);

			const response = await fetch('/api/v1/enhanced-admin-settings/download-model', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${$authStore.token}`
				},
				body: JSON.stringify({
					model_id: modelId,
					model_type: 'vision',
					is_public: true,
					force_redownload: false
				})
			});

			const result = await response.json();

			if (result.success) {
				addMessage('success', `Vision model ${modelId} download started successfully`);
			} else {
				addMessage('error', result.message || `Failed to download vision model ${modelId}`);
			}
		} catch (error) {
			console.error('Error downloading vision model:', error);
			addMessage('error', `Error downloading vision model ${modelId}: ${error.message}`);
		}
	}

	// ============================================================================
	// 🚀 REVOLUTIONARY RAG TABS FUNCTIONS
	// ============================================================================





	// 🚀 Revolutionary LLM Providers Change Tracking Functions
	function trackLLMChange(key: string, value: any) {
		llmUnsavedChanges.update(changes => {
			const newChanges = { ...changes };

			// If value matches original, remove from changes
			if (llmOriginalValues[key] !== undefined && llmOriginalValues[key] === value) {
				delete newChanges[key];
			} else {
				newChanges[key] = value;
			}

			// Update has changes flag
			llmHasUnsavedChanges.set(Object.keys(newChanges).length > 0);

			return newChanges;
		});
	}

	function initializeLLMOriginalValues(settings: Record<string, any>) {
		llmOriginalValues = {};
		Object.entries(settings).forEach(([key, setting]) => {
			if (key.startsWith('llm_providers.')) {
				const cleanKey = key.replace('llm_providers.', '');
				llmOriginalValues[cleanKey] = setting.value;
			}
		});
	}

	function initializeDatabaseOriginalValues(settings: Record<string, any>) {
		const databaseOriginalValues: Record<string, any> = {};
		Object.entries(settings).forEach(([key, setting]) => {
			// Keys no longer have database_storage prefix in the new API response format
			databaseOriginalValues[key] = setting.value;
		});
		// Initialize unsaved changes with current values
		databaseUnsavedChanges.set(databaseOriginalValues);
	}

	async function saveLLMProviderChanges() {
		llmSaving.set(true);
		llmSaveStatus.set({ type: null, message: '' });

		try {
			const changes = get(llmUnsavedChanges);
			const updates = Object.entries(changes).map(([key, value]) => ({
				category: 'llm_providers',
				key: key,
				value: value,
				validate_only: false
			}));

			const response = await fetch('/api/v1/admin/enhanced-settings/bulk-update', {
				method: 'POST',
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					updates: updates,
					validate_all: true
				})
			});

			const result = await response.json();

			if (result.success) {
				// First, reload the category settings to get the updated values from the server
				await fetchCategorySettings('llm_providers');

				// Then clear unsaved changes (after we have the new values loaded)
				llmUnsavedChanges.set({});
				llmHasUnsavedChanges.set(false);

				// Update original values with the new values
				Object.entries(changes).forEach(([key, value]) => {
					llmOriginalValues[key] = value;
				});

				// Show success message
				llmSaveStatus.set({
					type: 'success',
					message: `✅ ${Object.keys(changes).length} LLM provider settings updated in real-time!${result.data?.llm_update?.success ? ' All backend systems updated.' : ''}`
				});

				// Refresh category settings to show updated values
				await fetchCategorySettings('llm_providers');

			} else {
				llmSaveStatus.set({
					type: 'error',
					message: `❌ Failed to save changes: ${result.message || 'Unknown error'}`
				});
			}

		} catch (error) {
			console.error('Error saving LLM provider changes:', error);
			llmSaveStatus.set({
				type: 'error',
				message: `❌ Network error: ${error.message}`
			});
		} finally {
			llmSaving.set(false);

			// Clear status message after 5 seconds
			setTimeout(() => {
				llmSaveStatus.set({ type: null, message: '' });
			}, 5000);
		}
	}

	function discardLLMChanges() {
		llmUnsavedChanges.set({});
		llmHasUnsavedChanges.set(false);
		llmSaveStatus.set({ type: null, message: '' });
	}

	async function loadLLMProviderStatus() {
		if (llmManager) {
			await llmManager.loadLLMProviderStatus();
		}
	}

	async function loadLLMAvailableModels() {
		if (llmManager) {
			await llmManager.loadLLMAvailableModels();
		}
	}

	async function loadLLMProviderTemplates() {
		if (llmManager) {
			await llmManager.loadLLMProviderTemplates();
		}
	}

	// 🚀 Enhanced Ollama Connection & Model Management - DELEGATED TO MANAGER
	async function checkOllamaConnection() {
		if (llmManager) {
			await llmManager.checkOllamaConnection();
		}
	}

	// Helper functions now handled by managers

	async function downloadOllamaModel(modelName: string) {
		if (llmManager) {
			await llmManager.downloadOllamaModel(modelName);
		}
	}

	async function applyLLMTemplate(templateId: string) {
		if (llmManager) {
			await llmManager.applyLLMTemplate(templateId);
		}
	}



	async function loadDatabaseConnectionStatus() {
		if (databaseManager) {
			await databaseManager.loadDatabaseConnectionStatus();
		}
	}

	async function testDatabaseConnection(connectionType: 'postgresql' | 'vector' | 'redis', dbType?: string) {
		if (databaseManager) {
			await databaseManager.testDatabaseConnection(connectionType, dbType);
		}
	}

	// ============================================================================
	// LIFECYCLE
	// ============================================================================

	// Manager instances
	let ragManager: RAGSettingsManager;
	let llmManager: LLMProvidersManager;
	let databaseManager: DatabaseStorageManager;

	onMount(async () => {
		// Check if user is admin
		if (!isAdmin) {
			goto('/dashboard');
			return;
		}

		// Initialize managers
		ragManager = new RAGSettingsManager($authStore.token || '');
		llmManager = new LLMProvidersManager($authStore.token || '');
		databaseManager = new DatabaseStorageManager($authStore.token || '');

		// Load initial data
		await fetchCategories();

		// Load default category settings
		if ($activeCategory) {
			await fetchCategorySettings($activeCategory);
		}

		// Initialize all managers
		await ragManager.initialize();
		await llmManager.initialize();
		await databaseManager.initialize();

		// Clear messages after 5 seconds
		const messageTimer = setInterval(() => {
			clearMessages();
		}, 5000);

		// 🚀 Setup real-time configuration change handling
		const unsubscribeConfigChanges = configurationChanges.subscribe((changes) => {
			// Handle configuration changes from other admins
			changes.forEach((change) => {
				handleRealTimeConfigurationChange(change);
			});
		});

		// Setup model download progress handling
		const handleModelProgress = (event: CustomEvent) => {
			const progress = event.detail;
			console.log('📥 Model download progress:', progress);
			// Update UI based on progress
		};

		window.addEventListener('model-download-progress', handleModelProgress);

		return () => {
			clearInterval(messageTimer);
			unsubscribeConfigChanges();
			window.removeEventListener('model-download-progress', handleModelProgress);
		};
	});

	onDestroy(() => {
		// Cleanup WebSocket connection if needed
		// webSocketService.disconnect(); // Don't disconnect as other pages might use it
	});

	// ============================================================================
	// HELPER FUNCTIONS - NOW HANDLED BY COMPONENTS
	// ============================================================================


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
	<!-- Enhanced Settings Header Component -->
	<EnhancedSettingsHeader
		unsavedChanges={$unsavedChanges}
		loading={$loading}
		on:saveAll={handleSaveAll}
	/>

	<!-- Enhanced Settings Messages Component -->
	<EnhancedSettingsMessages
		error={$error}
		success={$success}
		on:clearMessages={clearMessages}
	/>

	<!-- Enhanced Settings Status Bar Component -->
	<EnhancedSettingsStatusBar
		websocketConnected={true}
		configurationVerified={true}
		lastUpdateTime={new Date()}
	/>

	<!-- Revolutionary Compact Main Content -->
	<div class="max-w-7xl mx-auto px-4 py-4">
		<div class="grid grid-cols-1 lg:grid-cols-5 gap-4">
			<!-- Enhanced Settings Sidebar Component -->
			<EnhancedSettingsSidebar
				categories={$categories}
				activeCategory={$activeCategory}
				expandedGroups={$expandedGroups}
				on:categorySelect={(e) => handleCategorySelect(e.detail.categoryId)}
				on:toggleGroup={(e) => toggleGroup(e.detail.groupId)}
			/>

			<!-- Revolutionary Compact Settings Panel -->
			<div class="lg:col-span-4">
				<div class="glass rounded-xl border border-white/10 p-4">
					{#if Object.keys($categorySettings).length > 0}

						{#if $activeCategory === 'rag_configuration'}
							<!-- RAG Settings Panel Component -->
							<RAGSettingsPanel
								categorySettings={$categorySettings}
								unsavedChanges={$unsavedChanges}
								validationErrors={$validationErrors}
								ragTemplates={$ragTemplates}
								authToken={$authStore.token || ''}
								on:settingChange={(e) => handleSettingChange(e.detail.key, e.detail.value)}
								on:saveSetting={(e) => saveSetting(e.detail.key, e.detail.value, $activeCategory)}
								on:downloadEmbeddingModel={(e) => downloadEmbeddingModel(e.detail.modelId)}
								on:downloadVisionModel={(e) => downloadVisionModel(e.detail.modelId)}
							/>





						{:else if $activeCategory === 'llm_providers'}
							<LLMProvidersPanel
								categorySettings={$categorySettings}
								unsavedChanges={$llmUnsavedChanges}
								validationErrors={$validationErrors}
								llmProviderStatus={$llmProviderStatus}
								llmAvailableModels={$llmAvailableModels}
								llmProviderTemplates={$llmProviderTemplates}
								ollamaConnectionStatus={$ollamaConnectionStatus}
								authToken={$authStore.token || ''}
								on:settingChange={(e) => handleSettingChange(e.detail.key, e.detail.value)}
								on:saveSetting={(e) => handleSaveSetting(e.detail.key)}
								on:applyTemplate={(e) => applyLLMTemplate(e.detail.templateId)}
								on:checkOllamaConnection={checkOllamaConnection}
								on:refreshProviderStatus={loadLLMProviderStatus}
								on:downloadModel={(e) => downloadOllamaModel(e.detail.modelName)}
								on:saveAllChanges={handleSaveAll}
							/>









						{:else if $activeCategory === 'database_storage'}
							<DatabaseStoragePanel
								categorySettings={$categorySettings}
								unsavedChanges={$unsavedChanges}
								validationErrors={$validationErrors}
								databaseConnectionStatus={$databaseConnectionStatus}
								authToken={$authStore.token || ''}
								on:settingChange={(e) => handleSettingChange(e.detail.key, e.detail.value)}
								on:saveSetting={(e) => saveSetting(e.detail.key, e.detail.value, $activeCategory)}
								on:refreshDatabaseStatus={loadDatabaseConnectionStatus}
								on:testVectorConnection={(e) => testDatabaseConnection('vector', e.detail.dbType)}
							/>



							{:else}
								<SettingsCategoryPanel
									categorySettings={$categorySettings}
									unsavedChanges={$unsavedChanges}
									validationErrors={$validationErrors}
									on:settingChange={(e) => handleSettingChange(e.detail.key, e.detail.value)}
									on:saveSetting={(e) => handleSaveSetting(e.detail.key)}
								/>
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
