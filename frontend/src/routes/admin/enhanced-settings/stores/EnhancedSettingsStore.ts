/**
 * Enhanced Settings Store - Centralized State Management
 * 
 * This store manages all state for the Enhanced Admin Settings page,
 * providing a single source of truth for all reactive data.
 */

import { writable, type Writable } from 'svelte/store';
import type { 
	SettingCategory, 
	CategoryGroup, 
	SettingDefinition,
	ModelDownloadModalState,
	TabDefinition,
	DatabaseConnectionStatus,
	LLMProviderStatus,
	OllamaConnectionStatus,
	LLMSaveStatus
} from '../types/EnhancedSettingsTypes';

/**
 * Core Enhanced Settings Store
 */
class EnhancedSettingsStore {
	// ============================================================================
	// CORE STORES
	// ============================================================================
	
	public readonly loading: Writable<boolean> = writable(false);
	public readonly error: Writable<string | null> = writable(null);
	public readonly success: Writable<string | null> = writable(null);
	public readonly categories: Writable<Record<string, CategoryGroup>> = writable({});
	public readonly activeCategory: Writable<string> = writable('system_configuration');
	public readonly categorySettings: Writable<Record<string, SettingDefinition>> = writable({});
	public readonly expandedGroups: Writable<Set<string>> = writable(new Set(['core']));
	public readonly unsavedChanges: Writable<Record<string, any>> = writable({});
	public readonly validationErrors: Writable<Record<string, string[]>> = writable({});

	// ============================================================================
	// RAG SYSTEM STORES
	// ============================================================================
	
	public readonly modelDownloadModal: Writable<ModelDownloadModalState> = writable({
		isOpen: false,
		modelType: 'embedding',
		currentModel: ''
	});

	public readonly ragTemplates: Writable<Record<string, any>> = writable({});
	public readonly availableModels: Writable<Record<string, any[]>> = writable({
		embedding: [],
		vision: [],
		reranking: []
	});

	public readonly ragActiveTab: Writable<string> = writable('models');
	public readonly ragTabs: TabDefinition[] = [
		{ id: 'models', name: 'Models & Downloads', icon: 'Download', description: 'Embedding, Vision & Reranking Models' },
		{ id: 'ocr', name: 'OCR Configuration', icon: 'FileText', description: 'Tesseract, EasyOCR, PaddleOCR' },
		{ id: 'processing', name: 'Document Processing', icon: 'FileSearch', description: 'Chunking, Preprocessing, File Handling' },
		{ id: 'retrieval', name: 'Retrieval & Search', icon: 'Search', description: 'Top-k, Scoring, Hybrid Search' },
		{ id: 'performance', name: 'Performance', icon: 'Zap', description: 'Caching, Concurrency, Optimization' },
		{ id: 'templates', name: 'Templates', icon: 'Sparkles', description: 'Pre-configured RAG Templates' }
	];

	// ============================================================================
	// LLM PROVIDER STORES
	// ============================================================================
	
	public readonly llmActiveTab: Writable<string> = writable('providers');
	public readonly llmTabs: TabDefinition[] = [
		{ id: 'providers', name: 'Providers & Status', icon: 'Brain', description: 'Provider Configuration & Status' },
		{ id: 'models', name: 'Models & Downloads', icon: 'Download', description: 'Available Models & Downloads' },
		{ id: 'configuration', name: 'Configuration', icon: 'Settings', description: 'Provider Settings & Parameters' },
		{ id: 'performance', name: 'Performance', icon: 'Zap', description: 'Timeouts, Limits & Optimization' },
		{ id: 'security', name: 'Security', icon: 'Shield', description: 'API Keys, Authentication & Security' },
		{ id: 'templates', name: 'Templates', icon: 'Sparkles', description: 'Pre-configured Provider Templates' }
	];

	public readonly llmProviderStatus: Writable<Record<string, LLMProviderStatus>> = writable({});
	public readonly llmAvailableModels: Writable<Record<string, any[]>> = writable({});
	public readonly llmProviderTemplates: Writable<Record<string, any>> = writable({});
	public readonly ollamaConnectionStatus: Writable<OllamaConnectionStatus> = writable({
		connected: false,
		url: 'http://localhost:11434',
		models: [],
		error: undefined
	});

	// LLM Change Tracking System
	public readonly llmUnsavedChanges: Writable<Record<string, any>> = writable({});
	public readonly llmHasUnsavedChanges: Writable<boolean> = writable(false);
	public readonly llmSaving: Writable<boolean> = writable(false);
	public readonly llmSaveStatus: Writable<LLMSaveStatus> = writable({ type: null, message: '' });

	// ============================================================================
	// DATABASE STORAGE STORES
	// ============================================================================
	
	public readonly databaseActiveTab: Writable<string> = writable('postgresql');
	public readonly databaseTabs: TabDefinition[] = [
		{ id: 'postgresql', name: 'PostgreSQL', icon: 'Database', description: 'PostgreSQL Database Configuration' },
		{ id: 'vector', name: 'Vector DB', icon: 'Search', description: 'ChromaDB & PgVector Configuration' },
		{ id: 'redis', name: 'Redis Cache', icon: 'Zap', description: 'Redis Caching Configuration' },
		{ id: 'performance', name: 'Performance', icon: 'TrendingUp', description: 'Query Optimization & Monitoring' },
		{ id: 'maintenance', name: 'Maintenance', icon: 'Settings', description: 'Migrations, Schema & Monitoring' }
	];

	public readonly databaseConnectionStatus: Writable<Record<string, DatabaseConnectionStatus>> = writable({
		postgresql: { connected: false, status: 'unknown' },
		vector: { connected: false, status: 'unknown', type: 'auto' },
		redis: { connected: false, status: 'unknown' }
	});
	public readonly databaseUnsavedChanges: Writable<Record<string, any>> = writable({});

	// ============================================================================
	// STORE COMPOSITION METHODS
	// ============================================================================

	/**
	 * Reset all stores to initial state
	 */
	public resetAllStores(): void {
		this.loading.set(false);
		this.error.set(null);
		this.success.set(null);
		this.categories.set({});
		this.activeCategory.set('system_configuration');
		this.categorySettings.set({});
		this.expandedGroups.set(new Set(['core']));
		this.unsavedChanges.set({});
		this.validationErrors.set({});
		
		// Reset RAG stores
		this.modelDownloadModal.set({ isOpen: false, modelType: 'embedding', currentModel: '' });
		this.ragTemplates.set({});
		this.availableModels.set({ embedding: [], vision: [], reranking: [] });
		this.ragActiveTab.set('models');
		
		// Reset LLM stores
		this.llmActiveTab.set('providers');
		this.llmProviderStatus.set({});
		this.llmAvailableModels.set({});
		this.llmProviderTemplates.set({});
		this.ollamaConnectionStatus.set({ connected: false, url: 'http://localhost:11434', models: [], error: undefined });
		this.llmUnsavedChanges.set({});
		this.llmHasUnsavedChanges.set(false);
		this.llmSaving.set(false);
		this.llmSaveStatus.set({ type: null, message: '' });
		
		// Reset Database stores
		this.databaseActiveTab.set('postgresql');
		this.databaseConnectionStatus.set({
			postgresql: { connected: false, status: 'unknown' },
			vector: { connected: false, status: 'unknown', type: 'auto' },
			redis: { connected: false, status: 'unknown' }
		});
		this.databaseUnsavedChanges.set({});
	}

	/**
	 * Clear all error and success messages
	 */
	public clearMessages(): void {
		this.error.set(null);
		this.success.set(null);
	}

	/**
	 * Set loading state for all operations
	 */
	public setGlobalLoading(loading: boolean): void {
		this.loading.set(loading);
	}

	/**
	 * Set error message
	 */
	public setError(message: string): void {
		this.error.set(message);
		this.success.set(null);
	}

	/**
	 * Set success message
	 */
	public setSuccess(message: string): void {
		this.success.set(message);
		this.error.set(null);
	}
}

// Create and export singleton instance
export const enhancedSettingsStore = new EnhancedSettingsStore();
