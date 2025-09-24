/**
 * RAG Settings Manager
 * 
 * This manager handles all RAG-specific functionality including model downloads,
 * template management, and configuration operations.
 */

import { get } from 'svelte/store';
import { enhancedSettingsStore } from '../stores/EnhancedSettingsStore';
import { enhancedSettingsAPI } from '../services/EnhancedSettingsAPI';
import type { ModelType } from '../types/EnhancedSettingsTypes';

/**
 * RAG Settings Manager Class
 */
export class RAGSettingsManager {
	private authToken: string;

	constructor(authToken: string) {
		this.authToken = authToken;
	}

	// ============================================================================
	// MODEL DOWNLOAD MANAGEMENT
	// ============================================================================

	/**
	 * Open model download modal
	 */
	openModelDownloadModal(modelType: ModelType, currentModel: string = ''): void {
		enhancedSettingsStore.modelDownloadModal.set({
			isOpen: true,
			modelType,
			currentModel
		});
	}

	/**
	 * Close model download modal
	 */
	closeModelDownloadModal(): void {
		enhancedSettingsStore.modelDownloadModal.update(state => ({
			...state,
			isOpen: false
		}));
	}

	/**
	 * Handle model download completion
	 */
	async handleModelDownloadComplete(modelType: ModelType, modelId: string, modelName: string): Promise<void> {
		try {
			// Update the current model setting
			const settingKey = modelType === 'embedding' ? 'embedding_model' :
							  modelType === 'vision' ? 'primary_vision_model' : 'reranking_model';

			// Update the setting through the store
			const currentChanges = get(enhancedSettingsStore.unsavedChanges);
			enhancedSettingsStore.unsavedChanges.set({
				...currentChanges,
				[`rag_configuration.${settingKey}`]: modelId
			});

			// Close the modal
			this.closeModelDownloadModal();

			// Show success message
			enhancedSettingsStore.setSuccess(`${modelName} model downloaded and configured successfully`);

			// Refresh available models
			await this.loadAvailableModels();
		} catch (error) {
			console.error('Error handling model download completion:', error);
			enhancedSettingsStore.setError('Failed to configure downloaded model');
		}
	}

	/**
	 * Load available models
	 */
	async loadAvailableModels(): Promise<void> {
		try {
			const response = await enhancedSettingsAPI.loadAvailableModels(this.authToken);
			
			if (response.success && response.data) {
				enhancedSettingsStore.availableModels.set(response.data);
			} else {
				console.error('Failed to load available models:', response.error);
			}
		} catch (error) {
			console.error('Error loading available models:', error);
		}
	}

	/**
	 * Download model
	 */
	async downloadModel(modelType: ModelType, modelId: string): Promise<boolean> {
		try {
			enhancedSettingsStore.setGlobalLoading(true);

			const response = await enhancedSettingsAPI.downloadModel(modelType, modelId, this.authToken);
			
			if (response.success) {
				enhancedSettingsStore.setSuccess(`Model ${modelId} download started successfully`);
				return true;
			} else {
				enhancedSettingsStore.setError(response.error || 'Failed to start model download');
				return false;
			}
		} catch (error) {
			console.error('Error downloading model:', error);
			enhancedSettingsStore.setError('Failed to download model');
			return false;
		} finally {
			enhancedSettingsStore.setGlobalLoading(false);
		}
	}

	// ============================================================================
	// TEMPLATE MANAGEMENT
	// ============================================================================

	/**
	 * Load RAG templates
	 */
	async loadRAGTemplates(): Promise<void> {
		try {
			const response = await enhancedSettingsAPI.loadRAGTemplates(this.authToken);
			
			if (response.success && response.data) {
				enhancedSettingsStore.ragTemplates.set(response.data);
			} else {
				console.error('Failed to load RAG templates:', response.error);
			}
		} catch (error) {
			console.error('Error loading RAG templates:', error);
		}
	}

	/**
	 * Apply RAG template
	 */
	async applyRAGTemplate(templateId: string): Promise<boolean> {
		try {
			enhancedSettingsStore.setGlobalLoading(true);

			const templates = get(enhancedSettingsStore.ragTemplates);
			const template = templates[templateId];

			if (!template) {
				enhancedSettingsStore.setError('Template not found');
				return false;
			}

			const response = await enhancedSettingsAPI.applyRAGTemplate(templateId, this.authToken);
			
			if (response.success) {
				// Update unsaved changes with template configuration
				const currentChanges = get(enhancedSettingsStore.unsavedChanges);
				const templateChanges: Record<string, any> = {};

				// Apply template configuration to unsaved changes
				Object.entries(template.configuration).forEach(([key, value]) => {
					templateChanges[`rag_configuration.${key}`] = value;
				});

				enhancedSettingsStore.unsavedChanges.set({
					...currentChanges,
					...templateChanges
				});

				enhancedSettingsStore.setSuccess(`RAG template "${template.name}" applied successfully`);
				return true;
			} else {
				enhancedSettingsStore.setError(response.error || 'Failed to apply RAG template');
				return false;
			}
		} catch (error) {
			console.error('Error applying RAG template:', error);
			enhancedSettingsStore.setError('Failed to apply RAG template');
			return false;
		} finally {
			enhancedSettingsStore.setGlobalLoading(false);
		}
	}

	// ============================================================================
	// CONFIGURATION MANAGEMENT
	// ============================================================================

	/**
	 * Get popular embedding models
	 */
	getPopularEmbeddingModels(): Array<{ name: string; description: string; size: string }> {
		return [
			{
				name: 'sentence-transformers/all-MiniLM-L6-v2',
				description: 'Fast and efficient general-purpose embedding model',
				size: '90MB'
			},
			{
				name: 'sentence-transformers/all-mpnet-base-v2',
				description: 'High-quality general-purpose embedding model',
				size: '420MB'
			},
			{
				name: 'BAAI/bge-small-en-v1.5',
				description: 'Compact high-performance embedding model',
				size: '130MB'
			},
			{
				name: 'BAAI/bge-base-en-v1.5',
				description: 'Balanced performance and size embedding model',
				size: '440MB'
			},
			{
				name: 'BAAI/bge-large-en-v1.5',
				description: 'Large high-performance embedding model',
				size: '1.3GB'
			}
		];
	}

	/**
	 * Get popular vision models
	 */
	getPopularVisionModels(): Array<{ name: string; description: string; size: string }> {
		return [
			{
				name: 'openai/clip-vit-base-patch32',
				description: 'CLIP vision model for image understanding',
				size: '600MB'
			},
			{
				name: 'openai/clip-vit-large-patch14',
				description: 'Large CLIP vision model for better accuracy',
				size: '1.7GB'
			},
			{
				name: 'google/vit-base-patch16-224',
				description: 'Vision Transformer for image classification',
				size: '330MB'
			}
		];
	}

	/**
	 * Get popular reranking models
	 */
	getPopularRerankingModels(): Array<{ name: string; description: string; size: string }> {
		return [
			{
				name: 'BAAI/bge-reranker-base',
				description: 'Base reranking model for search result refinement',
				size: '280MB'
			},
			{
				name: 'BAAI/bge-reranker-large',
				description: 'Large reranking model for better accuracy',
				size: '1.1GB'
			},
			{
				name: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
				description: 'Compact cross-encoder for reranking',
				size: '90MB'
			}
		];
	}

	/**
	 * Initialize RAG settings manager
	 */
	async initialize(): Promise<void> {
		try {
			await Promise.all([
				this.loadRAGTemplates(),
				this.loadAvailableModels()
			]);
		} catch (error) {
			console.error('Error initializing RAG settings manager:', error);
		}
	}

	/**
	 * Update auth token
	 */
	updateAuthToken(newToken: string): void {
		this.authToken = newToken;
	}
}
