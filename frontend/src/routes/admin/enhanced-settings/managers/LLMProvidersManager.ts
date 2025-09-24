/**
 * LLM Providers Manager
 * 
 * This manager handles all LLM provider functionality including Ollama integration,
 * provider status management, and model operations.
 */

import { get } from 'svelte/store';
import { enhancedSettingsStore } from '../stores/EnhancedSettingsStore';
import { enhancedSettingsAPI } from '../services/EnhancedSettingsAPI';

/**
 * LLM Providers Manager Class
 */
export class LLMProvidersManager {
	private authToken: string;
	private originalValues: Record<string, any> = {};

	constructor(authToken: string) {
		this.authToken = authToken;
	}

	// ============================================================================
	// PROVIDER STATUS MANAGEMENT
	// ============================================================================

	/**
	 * Load LLM provider status
	 */
	async loadLLMProviderStatus(): Promise<void> {
		try {
			const response = await enhancedSettingsAPI.loadLLMProviderStatus(this.authToken);
			
			if (response.success && response.data) {
				enhancedSettingsStore.llmProviderStatus.set(response.data);
			} else {
				console.error('Failed to load LLM provider status:', response.error);
			}
		} catch (error) {
			console.error('Error loading LLM provider status:', error);
		}
	}

	/**
	 * Load available LLM models
	 */
	async loadLLMAvailableModels(): Promise<void> {
		try {
			const response = await enhancedSettingsAPI.loadLLMAvailableModels(this.authToken);
			
			if (response.success && response.data) {
				enhancedSettingsStore.llmAvailableModels.set(response.data);
			} else {
				console.error('Failed to load LLM available models:', response.error);
			}
		} catch (error) {
			console.error('Error loading LLM available models:', error);
		}
	}

	/**
	 * Load LLM provider templates
	 */
	async loadLLMProviderTemplates(): Promise<void> {
		try {
			const response = await enhancedSettingsAPI.loadLLMProviderTemplates(this.authToken);
			
			if (response.success && response.data) {
				enhancedSettingsStore.llmProviderTemplates.set(response.data);
			} else {
				console.error('Failed to load LLM provider templates:', response.error);
			}
		} catch (error) {
			console.error('Error loading LLM provider templates:', error);
		}
	}

	// ============================================================================
	// OLLAMA INTEGRATION
	// ============================================================================

	/**
	 * Check Ollama connection status
	 */
	async checkOllamaConnection(): Promise<void> {
		try {
			const response = await enhancedSettingsAPI.checkOllamaConnection(this.authToken);
			
			if (response.success && response.data) {
				enhancedSettingsStore.ollamaConnectionStatus.set(response.data);
			} else {
				enhancedSettingsStore.ollamaConnectionStatus.set({
					connected: false,
					url: 'http://localhost:11434',
					models: [],
					error: response.error || 'Connection failed'
				});
			}
		} catch (error) {
			console.error('Error checking Ollama connection:', error);
			enhancedSettingsStore.ollamaConnectionStatus.set({
				connected: false,
				url: 'http://localhost:11434',
				models: [],
				error: 'Connection error'
			});
		}
	}

	/**
	 * Download Ollama model
	 */
	async downloadOllamaModel(modelName: string): Promise<boolean> {
		try {
			enhancedSettingsStore.setGlobalLoading(true);

			const response = await enhancedSettingsAPI.downloadOllamaModel(modelName, this.authToken);
			
			if (response.success) {
				enhancedSettingsStore.setSuccess(`Ollama model ${modelName} download started`);
				
				// Refresh Ollama connection status to get updated model list
				await this.checkOllamaConnection();
				return true;
			} else {
				enhancedSettingsStore.setError(response.error || 'Failed to download Ollama model');
				return false;
			}
		} catch (error) {
			console.error('Error downloading Ollama model:', error);
			enhancedSettingsStore.setError('Failed to download Ollama model');
			return false;
		} finally {
			enhancedSettingsStore.setGlobalLoading(false);
		}
	}

	/**
	 * Get popular Ollama models
	 */
	getPopularOllamaModels(): Array<{
		name: string;
		description: string;
		size: string;
		tags: string[];
		capabilities: string[];
	}> {
		return [
			{
				name: 'llama3.2:latest',
				description: 'Latest Llama 3.2 model with improved performance',
				size: '2.0GB',
				tags: ['general', 'chat', 'instruct'],
				capabilities: ['text-generation', 'conversation', 'reasoning']
			},
			{
				name: 'llama3.1:8b',
				description: 'Llama 3.1 8B parameter model',
				size: '4.7GB',
				tags: ['general', 'chat', 'code'],
				capabilities: ['text-generation', 'code-generation', 'reasoning']
			},
			{
				name: 'codellama:7b',
				description: 'Code Llama 7B specialized for code generation',
				size: '3.8GB',
				tags: ['code', 'programming'],
				capabilities: ['code-generation', 'code-completion', 'debugging']
			},
			{
				name: 'mistral:7b',
				description: 'Mistral 7B high-performance language model',
				size: '4.1GB',
				tags: ['general', 'chat', 'efficient'],
				capabilities: ['text-generation', 'conversation', 'analysis']
			},
			{
				name: 'phi3:mini',
				description: 'Microsoft Phi-3 Mini compact model',
				size: '2.3GB',
				tags: ['compact', 'efficient', 'chat'],
				capabilities: ['text-generation', 'conversation', 'reasoning']
			},
			{
				name: 'gemma2:2b',
				description: 'Google Gemma 2B lightweight model',
				size: '1.6GB',
				tags: ['lightweight', 'efficient', 'general'],
				capabilities: ['text-generation', 'conversation']
			}
		];
	}

	// ============================================================================
	// CHANGE TRACKING SYSTEM
	// ============================================================================

	/**
	 * Track LLM provider changes
	 */
	trackLLMChange(key: string, value: any): void {
		const currentChanges = get(enhancedSettingsStore.llmUnsavedChanges);
		const newChanges = { ...currentChanges };

		if (this.originalValues[key] !== undefined && this.originalValues[key] === value) {
			// Value reverted to original, remove from changes
			delete newChanges[key];
		} else {
			// Value changed, add to changes
			newChanges[key] = value;
		}

		enhancedSettingsStore.llmUnsavedChanges.set(newChanges);
		enhancedSettingsStore.llmHasUnsavedChanges.set(Object.keys(newChanges).length > 0);
	}

	/**
	 * Initialize LLM original values
	 */
	initializeLLMOriginalValues(settings: Record<string, any>): void {
		this.originalValues = {};
		Object.entries(settings).forEach(([key, setting]) => {
			if (key.startsWith('llm_providers.')) {
				this.originalValues[key] = setting.value;
			}
		});
	}

	/**
	 * Save LLM provider changes
	 */
	async saveLLMProviderChanges(): Promise<boolean> {
		try {
			enhancedSettingsStore.llmSaving.set(true);
			enhancedSettingsStore.llmSaveStatus.set({ type: null, message: '' });

			const changes = get(enhancedSettingsStore.llmUnsavedChanges);
			
			if (Object.keys(changes).length === 0) {
				enhancedSettingsStore.llmSaveStatus.set({ 
					type: 'success', 
					message: 'No changes to save' 
				});
				return true;
			}

			const response = await enhancedSettingsAPI.saveLLMProviderChanges(changes, this.authToken);
			
			if (response.success) {
				// Update original values with saved changes
				Object.entries(changes).forEach(([key, value]) => {
					this.originalValues[key] = value;
				});

				// Clear unsaved changes
				enhancedSettingsStore.llmUnsavedChanges.set({});
				enhancedSettingsStore.llmHasUnsavedChanges.set(false);
				enhancedSettingsStore.llmSaveStatus.set({ 
					type: 'success', 
					message: 'LLM provider settings saved successfully' 
				});

				// Refresh provider status
				await this.loadLLMProviderStatus();
				return true;
			} else {
				enhancedSettingsStore.llmSaveStatus.set({ 
					type: 'error', 
					message: response.error || 'Failed to save LLM provider settings' 
				});
				return false;
			}
		} catch (error) {
			console.error('Error saving LLM provider changes:', error);
			enhancedSettingsStore.llmSaveStatus.set({ 
				type: 'error', 
				message: 'Failed to save LLM provider settings' 
			});
			return false;
		} finally {
			enhancedSettingsStore.llmSaving.set(false);
		}
	}

	/**
	 * Discard LLM provider changes
	 */
	discardLLMChanges(): void {
		enhancedSettingsStore.llmUnsavedChanges.set({});
		enhancedSettingsStore.llmHasUnsavedChanges.set(false);
		enhancedSettingsStore.llmSaveStatus.set({ type: null, message: '' });
	}

	/**
	 * Apply LLM provider template
	 */
	async applyLLMTemplate(templateId: string): Promise<boolean> {
		try {
			const templates = get(enhancedSettingsStore.llmProviderTemplates);
			const template = templates[templateId];

			if (!template) {
				enhancedSettingsStore.setError('Template not found');
				return false;
			}

			const response = await enhancedSettingsAPI.applyLLMTemplate(templateId, this.authToken);
			
			if (response.success) {
				enhancedSettingsStore.setSuccess(`LLM template "${template.name}" applied successfully`);
				
				// Refresh provider status and settings
				await this.loadLLMProviderStatus();
				return true;
			} else {
				enhancedSettingsStore.setError(response.error || 'Failed to apply LLM template');
				return false;
			}
		} catch (error) {
			console.error('Error applying LLM template:', error);
			enhancedSettingsStore.setError('Failed to apply LLM template');
			return false;
		}
	}

	/**
	 * Initialize LLM providers manager
	 */
	async initialize(): Promise<void> {
		try {
			await Promise.all([
				this.loadLLMProviderStatus(),
				this.loadLLMAvailableModels(),
				this.loadLLMProviderTemplates(),
				this.checkOllamaConnection()
			]);
		} catch (error) {
			console.error('Error initializing LLM providers manager:', error);
		}
	}

	/**
	 * Update auth token
	 */
	updateAuthToken(newToken: string): void {
		this.authToken = newToken;
	}
}
