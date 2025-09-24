/**
 * Enhanced Settings API Service
 * 
 * This service handles all API communications for the Enhanced Admin Settings,
 * providing a centralized interface for backend interactions with proper
 * error handling and retry logic.
 */

import type {
	APIResponse,
	CategoryGroup,
	SettingDefinition,
	ValidationResponse,
	SettingUpdateRequest,
	SettingValidationRequest,
	LLMProviderStatus,
	DatabaseConnectionStatus,
	OllamaConnectionStatus
} from '../types/EnhancedSettingsTypes';

/**
 * Enhanced Settings API Service Class
 */
class EnhancedSettingsAPIService {
	private baseUrl = '/api/v1/admin/enhanced-settings';
	private defaultHeaders = {
		'Content-Type': 'application/json'
	};

	/**
	 * Get authorization headers with current token
	 */
	private getAuthHeaders(token: string): Record<string, string> {
		return {
			...this.defaultHeaders,
			'Authorization': `Bearer ${token}`
		};
	}

	/**
	 * Generic API request handler with error handling and retry logic
	 */
	private async makeRequest<T>(
		endpoint: string,
		options: RequestInit = {},
		token: string,
		retries: number = 3
	): Promise<APIResponse<T>> {
		const url = `${this.baseUrl}${endpoint}`;
		const requestOptions: RequestInit = {
			...options,
			headers: {
				...this.getAuthHeaders(token),
				...options.headers
			}
		};

		for (let attempt = 1; attempt <= retries; attempt++) {
			try {
				const response = await fetch(url, requestOptions);
				
				if (!response.ok) {
					throw new Error(`HTTP ${response.status}: ${response.statusText}`);
				}

				const data = await response.json();
				return data;
			} catch (error) {
				console.error(`API request failed (attempt ${attempt}/${retries}):`, error);
				
				if (attempt === retries) {
					return {
						success: false,
						error: error instanceof Error ? error.message : 'Unknown error occurred'
					};
				}
				
				// Wait before retry (exponential backoff)
				await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
			}
		}

		return {
			success: false,
			error: 'Maximum retry attempts exceeded'
		};
	}

	// ============================================================================
	// CORE SETTINGS API
	// ============================================================================

	/**
	 * Fetch all setting categories
	 */
	async fetchCategories(token: string): Promise<APIResponse<Record<string, CategoryGroup>>> {
		return this.makeRequest<Record<string, CategoryGroup>>('/categories', { method: 'GET' }, token);
	}

	/**
	 * Fetch settings for a specific category
	 */
	async fetchCategorySettings(categoryId: string, token: string): Promise<APIResponse<Record<string, SettingDefinition>>> {
		return this.makeRequest<Record<string, SettingDefinition>>(`/values/${categoryId}`, { method: 'GET' }, token);
	}

	/**
	 * Validate a setting value
	 */
	async validateSetting(request: SettingValidationRequest, token: string): Promise<APIResponse<ValidationResponse>> {
		return this.makeRequest<ValidationResponse>('/validate', {
			method: 'POST',
			body: JSON.stringify(request)
		}, token);
	}

	/**
	 * Update a setting value
	 */
	async updateSetting(request: SettingUpdateRequest, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/update', {
			method: 'POST',
			body: JSON.stringify(request)
		}, token);
	}

	/**
	 * Batch update multiple settings
	 */
	async batchUpdateSettings(requests: SettingUpdateRequest[], token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/batch-update', {
			method: 'POST',
			body: JSON.stringify({ updates: requests })
		}, token);
	}

	// ============================================================================
	// RAG SYSTEM API
	// ============================================================================

	/**
	 * Load RAG templates
	 */
	async loadRAGTemplates(token: string): Promise<APIResponse<Record<string, any>>> {
		return this.makeRequest<Record<string, any>>('/rag-templates', { method: 'GET' }, token);
	}

	/**
	 * Apply RAG template
	 */
	async applyRAGTemplate(templateId: string, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/rag-templates/apply', {
			method: 'POST',
			body: JSON.stringify({ template_id: templateId })
		}, token);
	}

	/**
	 * Load available models
	 */
	async loadAvailableModels(token: string): Promise<APIResponse<Record<string, any[]>>> {
		return this.makeRequest<Record<string, any[]>>('/models/available', { method: 'GET' }, token);
	}

	/**
	 * Download model
	 */
	async downloadModel(modelType: string, modelId: string, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/models/download', {
			method: 'POST',
			body: JSON.stringify({ model_type: modelType, model_id: modelId })
		}, token);
	}

	// ============================================================================
	// LLM PROVIDERS API
	// ============================================================================

	/**
	 * Load LLM provider status
	 */
	async loadLLMProviderStatus(token: string): Promise<APIResponse<Record<string, LLMProviderStatus>>> {
		return this.makeRequest<Record<string, LLMProviderStatus>>('/llm-providers/status', { method: 'GET' }, token);
	}

	/**
	 * Load available LLM models
	 */
	async loadLLMAvailableModels(token: string): Promise<APIResponse<Record<string, any[]>>> {
		return this.makeRequest<Record<string, any[]>>('/llm-providers/available-models', { method: 'GET' }, token);
	}

	/**
	 * Load LLM provider templates
	 */
	async loadLLMProviderTemplates(token: string): Promise<APIResponse<Record<string, any>>> {
		return this.makeRequest<Record<string, any>>('/llm-providers/templates', { method: 'GET' }, token);
	}

	/**
	 * Check Ollama connection status
	 */
	async checkOllamaConnection(token: string): Promise<APIResponse<OllamaConnectionStatus>> {
		return this.makeRequest<OllamaConnectionStatus>('/llm-providers/ollama-status', { method: 'GET' }, token);
	}

	/**
	 * Download Ollama model
	 */
	async downloadOllamaModel(modelName: string, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/llm-providers/ollama-download', {
			method: 'POST',
			body: JSON.stringify({ model_name: modelName })
		}, token);
	}

	/**
	 * Apply LLM provider template
	 */
	async applyLLMTemplate(templateId: string, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/llm-providers/templates/apply', {
			method: 'POST',
			body: JSON.stringify({ template_id: templateId })
		}, token);
	}

	/**
	 * Save LLM provider changes
	 */
	async saveLLMProviderChanges(changes: Record<string, any>, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/llm-providers/save-changes', {
			method: 'POST',
			body: JSON.stringify({ changes })
		}, token);
	}

	// ============================================================================
	// DATABASE STORAGE API
	// ============================================================================

	/**
	 * Load database connection status
	 */
	async loadDatabaseConnectionStatus(token: string): Promise<APIResponse<Record<string, DatabaseConnectionStatus>>> {
		return this.makeRequest<Record<string, DatabaseConnectionStatus>>('/database-storage/status', { method: 'GET' }, token);
	}

	/**
	 * Test database connection
	 */
	async testDatabaseConnection(connectionType: string, token: string, dbType?: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/database-storage/test-connection', {
			method: 'POST',
			body: JSON.stringify({ 
				connection_type: connectionType,
				db_type: dbType 
			})
		}, token);
	}

	/**
	 * Save database configuration
	 */
	async saveDatabaseConfiguration(configuration: Record<string, any>, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/database-storage/save-configuration', {
			method: 'POST',
			body: JSON.stringify({ configuration })
		}, token);
	}

	// ============================================================================
	// SYSTEM OPERATIONS API
	// ============================================================================

	/**
	 * Restart system service
	 */
	async restartSystem(token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/system/restart', { method: 'POST' }, token);
	}

	/**
	 * Get system health status
	 */
	async getSystemHealth(token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/system/health', { method: 'GET' }, token);
	}

	/**
	 * Export configuration
	 */
	async exportConfiguration(token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/system/export-config', { method: 'GET' }, token);
	}

	/**
	 * Import configuration
	 */
	async importConfiguration(configData: any, token: string): Promise<APIResponse<any>> {
		return this.makeRequest<any>('/system/import-config', {
			method: 'POST',
			body: JSON.stringify({ config: configData })
		}, token);
	}
}

// Create and export singleton instance
export const enhancedSettingsAPI = new EnhancedSettingsAPIService();
