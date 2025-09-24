/**
 * Enhanced Settings Types - Comprehensive Type Definitions
 * 
 * This file contains all TypeScript interfaces, types, and type guards
 * for the Enhanced Admin Settings system.
 */

// ============================================================================
// CORE SETTING TYPES
// ============================================================================

export interface SettingCategory {
	id: string;
	name: string;
	description: string;
	icon: string;
	color: string;
}

export interface CategoryGroup {
	name: string;
	description: string;
	categories: SettingCategory[];
}

export interface SettingDefinition {
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
	display_name?: string;
	current_value?: any;
	default_value?: any;
	options?: Array<{ value: string; label: string }>;
}

// ============================================================================
// UI COMPONENT TYPES
// ============================================================================

export interface TabDefinition {
	id: string;
	name: string;
	icon: string;
	description: string;
}

export interface ModelDownloadModalState {
	isOpen: boolean;
	modelType: 'embedding' | 'vision' | 'reranking';
	currentModel: string;
}

// ============================================================================
// RAG SYSTEM TYPES
// ============================================================================

export interface RAGTemplate {
	id: string;
	name: string;
	description: string;
	configuration: Record<string, any>;
	category: string;
	tags: string[];
}

export interface ModelInfo {
	id: string;
	name: string;
	description: string;
	size: number;
	type: 'embedding' | 'vision' | 'reranking';
	provider: string;
	downloaded: boolean;
	downloadUrl?: string;
}

// ============================================================================
// LLM PROVIDER TYPES
// ============================================================================

export interface LLMProviderStatus {
	id: string;
	name: string;
	connected: boolean;
	status: 'online' | 'offline' | 'error' | 'unknown';
	lastChecked: string;
	error?: string;
	models?: string[];
	capabilities?: string[];
}

export interface OllamaConnectionStatus {
	connected: boolean;
	url: string;
	models: OllamaModel[];
	error?: string;
}

export interface OllamaModel {
	name: string;
	size: number;
	digest: string;
	modified_at: string;
	details?: {
		format: string;
		family: string;
		families: string[];
		parameter_size: string;
		quantization_level: string;
	};
}

export interface LLMSaveStatus {
	type: 'success' | 'error' | null;
	message: string;
}

export interface LLMProviderTemplate {
	id: string;
	name: string;
	description: string;
	provider: string;
	configuration: Record<string, any>;
	tags: string[];
}

// ============================================================================
// DATABASE STORAGE TYPES
// ============================================================================

export interface DatabaseConnectionStatus {
	connected: boolean;
	status: 'connected' | 'disconnected' | 'error' | 'unknown' | 'configured';
	type?: string;
	lastChecked?: string;
	error?: string;
	details?: Record<string, any>;
}

export interface DatabaseConfiguration {
	host: string;
	port: number;
	database: string;
	username: string;
	password: string;
	ssl_mode?: string;
	pool_size?: number;
	max_overflow?: number;
	pool_timeout?: number;
	pool_recycle?: number;
}

// ============================================================================
// API RESPONSE TYPES
// ============================================================================

export interface APIResponse<T = any> {
	success: boolean;
	data?: T;
	error?: string;
	message?: string;
}

export interface ValidationResponse {
	valid: boolean;
	errors?: string[];
	warnings?: string[];
}

export interface SettingUpdateRequest {
	key: string;
	value: any;
	category: string;
}

export interface SettingValidationRequest {
	key: string;
	value: any;
	category: string;
}

// ============================================================================
// WEBSOCKET TYPES
// ============================================================================

export interface ConfigurationChange {
	type: 'setting_changed' | 'category_updated' | 'system_restart_required';
	category: string;
	key?: string;
	value?: any;
	timestamp: string;
	user?: string;
}

// ============================================================================
// EVENT TYPES
// ============================================================================

export interface SettingChangeEvent {
	key: string;
	value: any;
}

export interface SaveSettingEvent {
	key: string;
	value?: any;
}

export interface ModelDownloadEvent {
	modelType: 'embedding' | 'vision' | 'reranking';
	modelId: string;
	modelName: string;
}

export interface TemplateApplyEvent {
	templateId: string;
}

export interface DatabaseTestEvent {
	dbType: string;
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

export function isSettingDefinition(obj: any): obj is SettingDefinition {
	return obj && 
		typeof obj.key === 'string' &&
		typeof obj.type === 'string' &&
		typeof obj.description === 'string' &&
		typeof obj.requires_restart === 'boolean' &&
		typeof obj.is_sensitive === 'boolean';
}

export function isCategoryGroup(obj: any): obj is CategoryGroup {
	return obj &&
		typeof obj.name === 'string' &&
		typeof obj.description === 'string' &&
		Array.isArray(obj.categories);
}

export function isAPIResponse<T>(obj: any): obj is APIResponse<T> {
	return obj && typeof obj.success === 'boolean';
}

export function isValidationResponse(obj: any): obj is ValidationResponse {
	return obj && typeof obj.valid === 'boolean';
}

export function isConfigurationChange(obj: any): obj is ConfigurationChange {
	return obj &&
		typeof obj.type === 'string' &&
		typeof obj.category === 'string' &&
		typeof obj.timestamp === 'string';
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

export type SettingValueType = string | number | boolean | object | null;

export type CategoryId = 
	| 'system_configuration'
	| 'security_authentication'
	| 'llm_providers'
	| 'rag_configuration'
	| 'database_storage'
	| 'performance_monitoring'
	| 'workflow_execution'
	| 'agent_management'
	| 'tool_repository'
	| 'notification_system';

export type ModelType = 'embedding' | 'vision' | 'reranking';

export type DatabaseType = 'postgresql' | 'vector' | 'redis';

export type LLMProvider = 'openai' | 'anthropic' | 'ollama' | 'huggingface' | 'local';

export type ConnectionStatus = 'connected' | 'disconnected' | 'error' | 'unknown' | 'configured';

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

export const SETTING_TYPES = [
	'string',
	'number',
	'boolean',
	'select',
	'multiselect',
	'password',
	'textarea',
	'json',
	'url',
	'email',
	'file',
	'directory'
] as const;

export type SettingType = typeof SETTING_TYPES[number];

export const MODEL_TYPES: ModelType[] = ['embedding', 'vision', 'reranking'];

export const DATABASE_TYPES: DatabaseType[] = ['postgresql', 'vector', 'redis'];

export const LLM_PROVIDERS: LLMProvider[] = ['openai', 'anthropic', 'ollama', 'huggingface', 'local'];
