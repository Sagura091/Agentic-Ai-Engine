/**
 * Database Storage Manager
 * 
 * This manager handles all database storage functionality including connection testing,
 * configuration management, and database operations.
 */

import { get } from 'svelte/store';
import { enhancedSettingsStore } from '../stores/EnhancedSettingsStore';
import { enhancedSettingsAPI } from '../services/EnhancedSettingsAPI';
import type { DatabaseType } from '../types/EnhancedSettingsTypes';

/**
 * Database Storage Manager Class
 */
export class DatabaseStorageManager {
	private authToken: string;
	private originalValues: Record<string, any> = {};

	constructor(authToken: string) {
		this.authToken = authToken;
	}

	// ============================================================================
	// CONNECTION MANAGEMENT
	// ============================================================================

	/**
	 * Load database connection status
	 */
	async loadDatabaseConnectionStatus(): Promise<void> {
		try {
			const response = await enhancedSettingsAPI.loadDatabaseConnectionStatus(this.authToken);
			
			if (response.success && response.data) {
				enhancedSettingsStore.databaseConnectionStatus.set(response.data);
			} else {
				console.error('Failed to load database connection status:', response.error);
			}
		} catch (error) {
			console.error('Error loading database connection status:', error);
		}
	}

	/**
	 * Test database connection
	 */
	async testDatabaseConnection(connectionType: DatabaseType, dbType?: string): Promise<boolean> {
		try {
			enhancedSettingsStore.setGlobalLoading(true);

			const response = await enhancedSettingsAPI.testDatabaseConnection(connectionType, this.authToken, dbType);
			
			if (response.success) {
				// Update connection status
				const currentStatus = get(enhancedSettingsStore.databaseConnectionStatus);
				enhancedSettingsStore.databaseConnectionStatus.set({
					...currentStatus,
					[connectionType]: {
						...currentStatus[connectionType],
						connected: true,
						status: 'connected',
						lastChecked: new Date().toISOString(),
						error: undefined
					}
				});

				enhancedSettingsStore.setSuccess(`${connectionType.toUpperCase()} connection test successful`);
				return true;
			} else {
				// Update connection status with error
				const currentStatus = get(enhancedSettingsStore.databaseConnectionStatus);
				enhancedSettingsStore.databaseConnectionStatus.set({
					...currentStatus,
					[connectionType]: {
						...currentStatus[connectionType],
						connected: false,
						status: 'error',
						lastChecked: new Date().toISOString(),
						error: response.error
					}
				});

				enhancedSettingsStore.setError(response.error || `${connectionType.toUpperCase()} connection test failed`);
				return false;
			}
		} catch (error) {
			console.error('Error testing database connection:', error);
			enhancedSettingsStore.setError('Failed to test database connection');
			return false;
		} finally {
			enhancedSettingsStore.setGlobalLoading(false);
		}
	}

	// ============================================================================
	// CONFIGURATION MANAGEMENT
	// ============================================================================

	/**
	 * Handle database setting change
	 */
	handleDatabaseSettingChange(key: string, value: any): void {
		// Update database unsaved changes
		const currentChanges = get(enhancedSettingsStore.databaseUnsavedChanges);
		enhancedSettingsStore.databaseUnsavedChanges.set({
			...currentChanges,
			[key]: value
		});

		// Also update global unsaved changes for consistency
		const globalChanges = get(enhancedSettingsStore.unsavedChanges);
		enhancedSettingsStore.unsavedChanges.set({
			...globalChanges,
			[key]: value
		});
	}

	/**
	 * Initialize database original values
	 */
	initializeDatabaseOriginalValues(settings: Record<string, any>): void {
		this.originalValues = {};
		Object.entries(settings).forEach(([key, setting]) => {
			// Keys no longer have database_storage prefix in the new API response format
			if (key.includes('database') || key.includes('postgresql') || key.includes('redis') || key.includes('vector')) {
				this.originalValues[key] = setting.value;
			}
		});
	}

	/**
	 * Save database configuration
	 */
	async saveDatabaseConfiguration(): Promise<boolean> {
		try {
			enhancedSettingsStore.setGlobalLoading(true);

			const changes = get(enhancedSettingsStore.databaseUnsavedChanges);
			
			if (Object.keys(changes).length === 0) {
				enhancedSettingsStore.setSuccess('No database changes to save');
				return true;
			}

			const response = await enhancedSettingsAPI.saveDatabaseConfiguration(changes, this.authToken);
			
			if (response.success) {
				// Update original values with saved changes
				Object.entries(changes).forEach(([key, value]) => {
					this.originalValues[key] = value;
				});

				// Clear unsaved changes
				enhancedSettingsStore.databaseUnsavedChanges.set({});

				// Also clear from global unsaved changes
				const globalChanges = get(enhancedSettingsStore.unsavedChanges);
				const updatedGlobalChanges = { ...globalChanges };
				Object.keys(changes).forEach(key => {
					delete updatedGlobalChanges[key];
				});
				enhancedSettingsStore.unsavedChanges.set(updatedGlobalChanges);

				enhancedSettingsStore.setSuccess('Database configuration saved successfully');

				// Refresh connection status
				await this.loadDatabaseConnectionStatus();
				return true;
			} else {
				enhancedSettingsStore.setError(response.error || 'Failed to save database configuration');
				return false;
			}
		} catch (error) {
			console.error('Error saving database configuration:', error);
			enhancedSettingsStore.setError('Failed to save database configuration');
			return false;
		} finally {
			enhancedSettingsStore.setGlobalLoading(false);
		}
	}

	// ============================================================================
	// DATABASE UTILITIES
	// ============================================================================

	/**
	 * Get PostgreSQL connection string template
	 */
	getPostgreSQLConnectionTemplate(): string {
		return 'postgresql://username:password@localhost:5432/database_name';
	}

	/**
	 * Get Redis connection string template
	 */
	getRedisConnectionTemplate(): string {
		return 'redis://localhost:6379/0';
	}

	/**
	 * Get ChromaDB connection template
	 */
	getChromaDBConnectionTemplate(): string {
		return 'http://localhost:8000';
	}

	/**
	 * Validate PostgreSQL connection string
	 */
	validatePostgreSQLConnection(connectionString: string): { valid: boolean; errors: string[] } {
		const errors: string[] = [];
		
		if (!connectionString) {
			errors.push('Connection string is required');
			return { valid: false, errors };
		}

		// Basic PostgreSQL connection string validation
		const postgresqlRegex = /^postgresql:\/\/([^:]+):([^@]+)@([^:]+):(\d+)\/(.+)$/;
		
		if (!postgresqlRegex.test(connectionString)) {
			errors.push('Invalid PostgreSQL connection string format');
			errors.push('Expected format: postgresql://username:password@host:port/database');
		}

		return { valid: errors.length === 0, errors };
	}

	/**
	 * Validate Redis connection string
	 */
	validateRedisConnection(connectionString: string): { valid: boolean; errors: string[] } {
		const errors: string[] = [];
		
		if (!connectionString) {
			errors.push('Connection string is required');
			return { valid: false, errors };
		}

		// Basic Redis connection string validation
		const redisRegex = /^redis:\/\/([^:]+):(\d+)(\/\d+)?$/;
		
		if (!redisRegex.test(connectionString)) {
			errors.push('Invalid Redis connection string format');
			errors.push('Expected format: redis://host:port/database');
		}

		return { valid: errors.length === 0, errors };
	}

	/**
	 * Get database performance recommendations
	 */
	getDatabasePerformanceRecommendations(): Array<{
		category: string;
		title: string;
		description: string;
		impact: 'high' | 'medium' | 'low';
	}> {
		return [
			{
				category: 'PostgreSQL',
				title: 'Connection Pooling',
				description: 'Use connection pooling to reduce connection overhead and improve performance',
				impact: 'high'
			},
			{
				category: 'PostgreSQL',
				title: 'Index Optimization',
				description: 'Create appropriate indexes on frequently queried columns',
				impact: 'high'
			},
			{
				category: 'Vector Database',
				title: 'Embedding Dimensions',
				description: 'Choose optimal embedding dimensions based on your use case',
				impact: 'medium'
			},
			{
				category: 'Redis',
				title: 'Memory Management',
				description: 'Configure appropriate memory limits and eviction policies',
				impact: 'medium'
			},
			{
				category: 'General',
				title: 'Regular Maintenance',
				description: 'Schedule regular database maintenance tasks like VACUUM and ANALYZE',
				impact: 'medium'
			}
		];
	}

	/**
	 * Get database health metrics
	 */
	async getDatabaseHealthMetrics(): Promise<Record<string, any>> {
		try {
			// This would typically call a health endpoint
			// For now, return mock data based on connection status
			const connectionStatus = get(enhancedSettingsStore.databaseConnectionStatus);
			
			return {
				postgresql: {
					status: connectionStatus.postgresql?.connected ? 'healthy' : 'unhealthy',
					connections: connectionStatus.postgresql?.connected ? 5 : 0,
					queries_per_second: connectionStatus.postgresql?.connected ? 12.5 : 0
				},
				vector: {
					status: connectionStatus.vector?.connected ? 'healthy' : 'unhealthy',
					collections: connectionStatus.vector?.connected ? 3 : 0,
					documents: connectionStatus.vector?.connected ? 1250 : 0
				},
				redis: {
					status: connectionStatus.redis?.connected ? 'healthy' : 'unhealthy',
					memory_usage: connectionStatus.redis?.connected ? '45MB' : '0MB',
					hit_rate: connectionStatus.redis?.connected ? '92%' : '0%'
				}
			};
		} catch (error) {
			console.error('Error getting database health metrics:', error);
			return {};
		}
	}

	/**
	 * Initialize database storage manager
	 */
	async initialize(): Promise<void> {
		try {
			await this.loadDatabaseConnectionStatus();
		} catch (error) {
			console.error('Error initializing database storage manager:', error);
		}
	}

	/**
	 * Update auth token
	 */
	updateAuthToken(newToken: string): void {
		this.authToken = newToken;
	}
}
