<script lang="ts">
	import { onMount } from 'svelte';
	import { 
		theme, 
		notificationActions 
	} from '$stores';
	import { apiClient } from '$services/api';
	import {
		Settings,
		User,
		Key,
		Server,
		Palette,
		Bell,
		Shield,
		Database,
		Save,
		RefreshCw,
		Eye,
		EyeOff,
		Plus,
		Trash2,
		Edit,
		Check,
		X,
		Search
	} from 'lucide-svelte';
	
	// Settings categories
	const settingsCategories = [
		{ id: 'general', name: 'General', icon: Settings },
		{ id: 'appearance', name: 'Appearance', icon: Palette },
		{ id: 'notifications', name: 'Notifications', icon: Bell },
		{ id: 'users', name: 'User Management', icon: User },
		{ id: 'api-keys', name: 'API Keys', icon: Key },
		{ id: 'models', name: 'Models', icon: Server },
		{ id: 'rag', name: 'RAG & Knowledge', icon: Database },
		{ id: 'security', name: 'Security', icon: Shield },
		{ id: 'system', name: 'System', icon: Database }
	];
	
	// State
	let activeCategory = 'general';
	let isSaving = false;
	let showApiKey = {};
	let editingUser = null;
	let newApiKey = { name: '', provider: '', key: '' };
	let showNewApiKeyForm = false;
	
	// Settings data
	let settings = {
		general: {
			platform_name: 'Agentic AI Platform',
			platform_description: 'Revolutionary AI Agent Builder',
			default_language: 'en',
			timezone: 'UTC',
			max_agents_per_user: 50,
			max_workflows_per_user: 20
		},
		appearance: {
			theme: 'dark',
			sidebar_collapsed: false,
			animations_enabled: true,
			compact_mode: false
		},
		notifications: {
			email_notifications: true,
			push_notifications: true,
			agent_status_alerts: true,
			workflow_alerts: true,
			system_alerts: true,
			alert_frequency: 'immediate'
		},
		security: {
			session_timeout: 24,
			require_2fa: false,
			password_min_length: 8,
			max_login_attempts: 5,
			api_rate_limit: 1000
		},
		system: {
			auto_backup: true,
			backup_frequency: 'daily',
			log_retention_days: 30,
			max_concurrent_agents: 100,
			default_memory_type: 'auto'
		}
	};
	
	// Mock data
	let users = [
		{
			id: '1',
			username: 'admin',
			email: 'admin@example.com',
			role: 'admin',
			status: 'active',
			created_at: '2024-01-01T00:00:00Z',
			last_login: '2024-01-15T10:30:00Z'
		},
		{
			id: '2',
			username: 'user1',
			email: 'user1@example.com',
			role: 'user',
			status: 'active',
			created_at: '2024-01-05T00:00:00Z',
			last_login: '2024-01-14T15:45:00Z'
		}
	];
	
	let apiKeys = [
		{
			id: '1',
			name: 'OpenAI GPT-4',
			provider: 'openai',
			key: 'sk-...abc123',
			created_at: '2024-01-01T00:00:00Z',
			last_used: '2024-01-15T10:30:00Z'
		},
		{
			id: '2',
			name: 'Anthropic Claude',
			provider: 'anthropic',
			key: 'sk-...def456',
			created_at: '2024-01-05T00:00:00Z',
			last_used: '2024-01-14T15:45:00Z'
		}
	];
	
	let models = [
		{
			id: '1',
			name: 'GPT-4 Turbo',
			provider: 'openai',
			model_id: 'gpt-4-turbo',
			status: 'active',
			max_tokens: 4096,
			cost_per_token: 0.00003
		},
		{
			id: '2',
			name: 'Claude 3 Sonnet',
			provider: 'anthropic',
			model_id: 'claude-3-sonnet',
			status: 'active',
			max_tokens: 4096,
			cost_per_token: 0.000015
		}
	];
	
	onMount(async () => {
		await loadSettings();
	});
	
	async function loadSettings() {
		try {
			// In a real implementation, this would load from the API
			// const response = await apiClient.getSettings();
			// if (response.success) {
			//     settings = response.data;
			// }
		} catch (error) {
			console.error('Failed to load settings:', error);
		}
	}
	
	async function saveSettings() {
		isSaving = true;
		
		try {
			// In a real implementation, this would save to the API
			// const response = await apiClient.updateSettings(settings);
			
			// Simulate API call
			await new Promise(resolve => setTimeout(resolve, 1000));
			
			// Update theme if changed
			if (settings.appearance.theme !== $theme) {
				theme.set(settings.appearance.theme as any);
			}
			
			notificationActions.add({
				type: 'success',
				title: 'Settings Saved',
				message: 'Your settings have been updated successfully'
			});
		} catch (error) {
			console.error('Failed to save settings:', error);
			notificationActions.add({
				type: 'error',
				title: 'Save Failed',
				message: 'Failed to save settings. Please try again.'
			});
		} finally {
			isSaving = false;
		}
	}
	
	function toggleApiKeyVisibility(keyId: string) {
		showApiKey[keyId] = !showApiKey[keyId];
	}
	
	function addApiKey() {
		if (!newApiKey.name || !newApiKey.provider || !newApiKey.key) {
			notificationActions.add({
				type: 'error',
				title: 'Validation Error',
				message: 'Please fill in all fields'
			});
			return;
		}
		
		const newKey = {
			id: crypto.randomUUID(),
			...newApiKey,
			created_at: new Date().toISOString(),
			last_used: null
		};
		
		apiKeys = [...apiKeys, newKey];
		newApiKey = { name: '', provider: '', key: '' };
		showNewApiKeyForm = false;
		
		notificationActions.add({
			type: 'success',
			title: 'API Key Added',
			message: 'New API key has been added successfully'
		});
	}
	
	function deleteApiKey(keyId: string) {
		if (!confirm('Are you sure you want to delete this API key?')) {
			return;
		}
		
		apiKeys = apiKeys.filter(key => key.id !== keyId);
		
		notificationActions.add({
			type: 'success',
			title: 'API Key Deleted',
			message: 'API key has been removed'
		});
	}
	
	function editUser(user: any) {
		editingUser = { ...user };
	}
	
	function saveUser() {
		if (!editingUser) return;
		
		users = users.map(user => 
			user.id === editingUser.id ? editingUser : user
		);
		
		editingUser = null;
		
		notificationActions.add({
			type: 'success',
			title: 'User Updated',
			message: 'User information has been updated'
		});
	}
	
	function cancelEditUser() {
		editingUser = null;
	}
	
	function deleteUser(userId: string) {
		if (!confirm('Are you sure you want to delete this user?')) {
			return;
		}
		
		users = users.filter(user => user.id !== userId);
		
		notificationActions.add({
			type: 'success',
			title: 'User Deleted',
			message: 'User has been removed from the system'
		});
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}
</script>

<svelte:head>
	<title>Settings - Agentic AI</title>
</svelte:head>

<div class="settings-container">
	<!-- Header -->
	<div class="settings-header">
		<div class="header-content">
			<div class="header-text">
				<h1 class="page-title">Settings</h1>
				<p class="page-subtitle">
					Configure your AI agent platform settings and preferences.
				</p>
			</div>
			
			<button 
				class="save-btn"
				on:click={saveSettings}
				disabled={isSaving}
			>
				{#if isSaving}
					<RefreshCw class="w-4 h-4 animate-spin" />
					<span>Saving...</span>
				{:else}
					<Save class="w-4 h-4" />
					<span>Save Changes</span>
				{/if}
			</button>
		</div>
	</div>
	
	<!-- Settings Interface -->
	<div class="settings-interface">
		<!-- Settings Navigation -->
		<div class="settings-nav">
			<nav class="nav-list">
				{#each settingsCategories as category}
					<button
						class="nav-item {activeCategory === category.id ? 'active' : ''}"
						on:click={() => activeCategory = category.id}
					>
						<svelte:component this={category.icon} class="w-5 h-5" />
						<span>{category.name}</span>
					</button>
				{/each}
			</nav>
		</div>
		
		<!-- Settings Content -->
		<div class="settings-content">
			{#if activeCategory === 'general'}
				<!-- General Settings -->
				<div class="settings-section">
					<h2 class="section-title">General Settings</h2>
					<div class="form-grid">
						<div class="form-group">
							<label class="form-label">Platform Name</label>
							<input
								type="text"
								class="form-input"
								bind:value={settings.general.platform_name}
							/>
						</div>
						
						<div class="form-group">
							<label class="form-label">Platform Description</label>
							<input
								type="text"
								class="form-input"
								bind:value={settings.general.platform_description}
							/>
						</div>
						
						<div class="form-group">
							<label class="form-label">Default Language</label>
							<select class="form-select" bind:value={settings.general.default_language}>
								<option value="en">English</option>
								<option value="es">Spanish</option>
								<option value="fr">French</option>
								<option value="de">German</option>
							</select>
						</div>
						
						<div class="form-group">
							<label class="form-label">Timezone</label>
							<select class="form-select" bind:value={settings.general.timezone}>
								<option value="UTC">UTC</option>
								<option value="America/New_York">Eastern Time</option>
								<option value="America/Los_Angeles">Pacific Time</option>
								<option value="Europe/London">London</option>
							</select>
						</div>
						
						<div class="form-group">
							<label class="form-label">Max Agents per User</label>
							<input
								type="number"
								class="form-input"
								bind:value={settings.general.max_agents_per_user}
							/>
						</div>
						
						<div class="form-group">
							<label class="form-label">Max Workflows per User</label>
							<input
								type="number"
								class="form-input"
								bind:value={settings.general.max_workflows_per_user}
							/>
						</div>
					</div>
				</div>
				
			{:else if activeCategory === 'appearance'}
				<!-- Appearance Settings -->
				<div class="settings-section">
					<h2 class="section-title">Appearance Settings</h2>
					<div class="form-grid">
						<div class="form-group">
							<label class="form-label">Theme</label>
							<select class="form-select" bind:value={settings.appearance.theme}>
								<option value="dark">Dark</option>
								<option value="light">Light</option>
								<option value="auto">Auto</option>
							</select>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.appearance.sidebar_collapsed}
								/>
								<span class="checkbox-label">Collapse Sidebar by Default</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.appearance.animations_enabled}
								/>
								<span class="checkbox-label">Enable Animations</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.appearance.compact_mode}
								/>
								<span class="checkbox-label">Compact Mode</span>
							</label>
						</div>
					</div>
				</div>
				
			{:else if activeCategory === 'notifications'}
				<!-- Notification Settings -->
				<div class="settings-section">
					<h2 class="section-title">Notification Settings</h2>
					<div class="form-grid">
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.notifications.email_notifications}
								/>
								<span class="checkbox-label">Email Notifications</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.notifications.push_notifications}
								/>
								<span class="checkbox-label">Push Notifications</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.notifications.agent_status_alerts}
								/>
								<span class="checkbox-label">Agent Status Alerts</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.notifications.workflow_alerts}
								/>
								<span class="checkbox-label">Workflow Alerts</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={settings.notifications.system_alerts}
								/>
								<span class="checkbox-label">System Alerts</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-label">Alert Frequency</label>
							<select class="form-select" bind:value={settings.notifications.alert_frequency}>
								<option value="immediate">Immediate</option>
								<option value="hourly">Hourly Digest</option>
								<option value="daily">Daily Digest</option>
								<option value="weekly">Weekly Digest</option>
							</select>
						</div>
					</div>
				</div>
				
			{:else if activeCategory === 'users'}
				<!-- User Management -->
				<div class="settings-section">
					<div class="section-header">
						<h2 class="section-title">User Management</h2>
						<button class="add-btn">
							<Plus class="w-4 h-4" />
							<span>Add User</span>
						</button>
					</div>
					
					<div class="users-table">
						<div class="table-header">
							<div class="header-cell">User</div>
							<div class="header-cell">Role</div>
							<div class="header-cell">Status</div>
							<div class="header-cell">Last Login</div>
							<div class="header-cell">Actions</div>
						</div>
						
						{#each users as user (user.id)}
							<div class="table-row">
								{#if editingUser && editingUser.id === user.id}
									<div class="cell">
										<input
											type="text"
											class="inline-input"
											bind:value={editingUser.username}
										/>
										<input
											type="email"
											class="inline-input"
											bind:value={editingUser.email}
										/>
									</div>
									<div class="cell">
										<select class="inline-select" bind:value={editingUser.role}>
											<option value="admin">Admin</option>
											<option value="user">User</option>
											<option value="viewer">Viewer</option>
										</select>
									</div>
									<div class="cell">
										<select class="inline-select" bind:value={editingUser.status}>
											<option value="active">Active</option>
											<option value="inactive">Inactive</option>
											<option value="suspended">Suspended</option>
										</select>
									</div>
									<div class="cell">
										{formatDate(user.last_login)}
									</div>
									<div class="cell">
										<div class="action-buttons">
											<button class="action-btn success" on:click={saveUser}>
												<Check class="w-4 h-4" />
											</button>
											<button class="action-btn secondary" on:click={cancelEditUser}>
												<X class="w-4 h-4" />
											</button>
										</div>
									</div>
								{:else}
									<div class="cell">
										<div class="user-info">
											<span class="username">{user.username}</span>
											<span class="email">{user.email}</span>
										</div>
									</div>
									<div class="cell">
										<span class="role-badge {user.role}">{user.role}</span>
									</div>
									<div class="cell">
										<span class="status-badge {user.status}">{user.status}</span>
									</div>
									<div class="cell">
										{formatDate(user.last_login)}
									</div>
									<div class="cell">
										<div class="action-buttons">
											<button class="action-btn secondary" on:click={() => editUser(user)}>
												<Edit class="w-4 h-4" />
											</button>
											<button class="action-btn danger" on:click={() => deleteUser(user.id)}>
												<Trash2 class="w-4 h-4" />
											</button>
										</div>
									</div>
								{/if}
							</div>
						{/each}
					</div>
				</div>
				
			{:else if activeCategory === 'rag'}
				<!-- RAG Settings -->
				<div class="settings-section">
					<div class="section-header">
						<h2 class="section-title">RAG & Knowledge Base</h2>
						<a href="/settings/rag" class="add-btn">
							<Database class="w-4 h-4" />
							<span>Configure RAG</span>
						</a>
					</div>

					<div class="rag-overview">
						<p class="section-description">
							Configure Retrieval-Augmented Generation settings, embedding models, chunking parameters, and knowledge base management.
						</p>

						<div class="rag-features">
							<div class="feature-item">
								<div class="feature-icon">
									<Database class="w-5 h-5 text-accent-blue" />
								</div>
								<div class="feature-content">
									<h4 class="feature-title">Embedding Models</h4>
									<p class="feature-description">Download and manage embedding models for semantic search</p>
								</div>
							</div>

							<div class="feature-item">
								<div class="feature-icon">
									<Settings class="w-5 h-5 text-accent-green" />
								</div>
								<div class="feature-content">
									<h4 class="feature-title">Chunking Configuration</h4>
									<p class="feature-description">Set chunk size, overlap, and retrieval strategies</p>
								</div>
							</div>

							<div class="feature-item">
								<div class="feature-icon">
									<Search class="w-5 h-5 text-accent-purple" />
								</div>
								<div class="feature-content">
									<h4 class="feature-title">Hybrid Search</h4>
									<p class="feature-description">Combine semantic and keyword search with reranking</p>
								</div>
							</div>
						</div>
					</div>
				</div>

			{:else if activeCategory === 'api-keys'}
				<!-- API Keys Management -->
				<div class="settings-section">
					<div class="section-header">
						<h2 class="section-title">API Keys</h2>
						<button class="add-btn" on:click={() => showNewApiKeyForm = true}>
							<Plus class="w-4 h-4" />
							<span>Add API Key</span>
						</button>
					</div>
					
					{#if showNewApiKeyForm}
						<div class="api-key-form">
							<div class="form-grid">
								<div class="form-group">
									<label class="form-label">Name</label>
									<input
										type="text"
										class="form-input"
										placeholder="e.g., OpenAI GPT-4"
										bind:value={newApiKey.name}
									/>
								</div>
								
								<div class="form-group">
									<label class="form-label">Provider</label>
									<select class="form-select" bind:value={newApiKey.provider}>
										<option value="">Select Provider</option>
										<option value="openai">OpenAI</option>
										<option value="anthropic">Anthropic</option>
										<option value="google">Google</option>
										<option value="cohere">Cohere</option>
									</select>
								</div>
								
								<div class="form-group full-width">
									<label class="form-label">API Key</label>
									<input
										type="password"
										class="form-input"
										placeholder="Enter your API key..."
										bind:value={newApiKey.key}
									/>
								</div>
							</div>
							
							<div class="form-actions">
								<button class="action-btn primary" on:click={addApiKey}>
									Add API Key
								</button>
								<button class="action-btn secondary" on:click={() => showNewApiKeyForm = false}>
									Cancel
								</button>
							</div>
						</div>
					{/if}
					
					<div class="api-keys-list">
						{#each apiKeys as apiKey (apiKey.id)}
							<div class="api-key-item">
								<div class="api-key-info">
									<h4 class="api-key-name">{apiKey.name}</h4>
									<span class="api-key-provider">{apiKey.provider}</span>
								</div>
								
								<div class="api-key-details">
									<div class="api-key-value">
										{#if showApiKey[apiKey.id]}
											<span class="key-text">{apiKey.key}</span>
										{:else}
											<span class="key-text">{'*'.repeat(20)}</span>
										{/if}
										<button 
											class="toggle-visibility"
											on:click={() => toggleApiKeyVisibility(apiKey.id)}
										>
											{#if showApiKey[apiKey.id]}
												<EyeOff class="w-4 h-4" />
											{:else}
												<Eye class="w-4 h-4" />
											{/if}
										</button>
									</div>
									
									<div class="api-key-meta">
										<span class="created-date">Created: {formatDate(apiKey.created_at)}</span>
										{#if apiKey.last_used}
											<span class="last-used">Last used: {formatDate(apiKey.last_used)}</span>
										{/if}
									</div>
								</div>
								
								<div class="api-key-actions">
									<button class="action-btn danger" on:click={() => deleteApiKey(apiKey.id)}>
										<Trash2 class="w-4 h-4" />
									</button>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.settings-container {
		@apply space-y-6;
	}
	
	/* Header */
	.settings-header {
		@apply space-y-4;
	}
	
	.header-content {
		@apply flex items-center justify-between;
	}
	
	.header-text {
		@apply space-y-2;
	}
	
	.page-title {
		@apply text-3xl font-bold text-white;
	}
	
	.page-subtitle {
		@apply text-lg text-dark-300 max-w-2xl;
	}
	
	.save-btn {
		@apply inline-flex items-center space-x-2 px-6 py-3 bg-primary-600 text-white rounded-xl font-medium hover:bg-primary-700 disabled:opacity-50 transition-all duration-200;
	}
	
	/* Settings Interface */
	.settings-interface {
		@apply grid grid-cols-1 lg:grid-cols-4 gap-6;
	}
	
	.settings-nav {
		@apply lg:col-span-1;
	}
	
	.nav-list {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-2 space-y-1;
	}
	
	.nav-item {
		@apply flex items-center space-x-3 w-full px-4 py-3 text-left text-dark-300 hover:text-white hover:bg-dark-700 rounded-lg transition-all duration-200;
	}
	
	.nav-item.active {
		@apply text-white bg-primary-600;
	}
	
	.settings-content {
		@apply lg:col-span-3;
	}
	
	/* Settings Sections */
	.settings-section {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6 space-y-6;
	}
	
	.section-header {
		@apply flex items-center justify-between;
	}
	
	.section-title {
		@apply text-2xl font-bold text-white;
	}
	
	.add-btn {
		@apply inline-flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors duration-200;
	}
	
	/* Forms */
	.form-grid {
		@apply grid grid-cols-1 md:grid-cols-2 gap-6;
	}
	
	.form-group {
		@apply space-y-2;
	}
	
	.form-group.full-width {
		@apply md:col-span-2;
	}
	
	.form-label {
		@apply block text-sm font-medium text-white;
	}
	
	.form-input,
	.form-select {
		@apply w-full px-4 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500;
	}
	
	.form-checkbox {
		@apply flex items-center space-x-3 cursor-pointer;
	}
	
	.checkbox-label {
		@apply text-sm text-white;
	}
	
	/* Tables */
	.users-table {
		@apply space-y-2;
	}
	
	.table-header {
		@apply grid grid-cols-5 gap-4 p-4 bg-dark-700 rounded-lg font-medium text-dark-300;
	}
	
	.header-cell {
		@apply text-sm;
	}
	
	.table-row {
		@apply grid grid-cols-5 gap-4 p-4 bg-dark-700/50 rounded-lg items-center;
	}
	
	.cell {
		@apply text-sm;
	}
	
	.user-info {
		@apply space-y-1;
	}
	
	.username {
		@apply block font-medium text-white;
	}
	
	.email {
		@apply block text-dark-400;
	}
	
	.role-badge,
	.status-badge {
		@apply px-2 py-1 text-xs font-medium rounded-lg;
	}
	
	.role-badge.admin {
		@apply bg-red-500/20 text-red-400;
	}
	
	.role-badge.user {
		@apply bg-blue-500/20 text-blue-400;
	}
	
	.role-badge.viewer {
		@apply bg-gray-500/20 text-gray-400;
	}
	
	.status-badge.active {
		@apply bg-green-500/20 text-green-400;
	}
	
	.status-badge.inactive {
		@apply bg-gray-500/20 text-gray-400;
	}
	
	.status-badge.suspended {
		@apply bg-red-500/20 text-red-400;
	}
	
	.inline-input,
	.inline-select {
		@apply w-full px-2 py-1 bg-dark-600 border border-dark-500 rounded text-white text-sm mb-1;
	}
	
	.action-buttons {
		@apply flex items-center space-x-2;
	}
	
	.action-btn {
		@apply p-2 rounded-lg transition-colors duration-200;
	}
	
	.action-btn.primary {
		@apply bg-primary-600 text-white hover:bg-primary-700;
	}
	
	.action-btn.secondary {
		@apply bg-dark-600 text-dark-300 hover:bg-dark-500 hover:text-white;
	}
	
	.action-btn.success {
		@apply bg-green-600 text-white hover:bg-green-700;
	}
	
	.action-btn.danger {
		@apply bg-red-600 text-white hover:bg-red-700;
	}
	
	/* API Keys */
	.api-key-form {
		@apply bg-dark-700 rounded-lg p-6 space-y-6;
	}
	
	.form-actions {
		@apply flex items-center space-x-3;
	}
	
	.api-keys-list {
		@apply space-y-4;
	}
	
	.api-key-item {
		@apply flex items-center justify-between p-4 bg-dark-700 rounded-lg;
	}
	
	.api-key-info {
		@apply space-y-1;
	}
	
	.api-key-name {
		@apply text-lg font-medium text-white;
	}
	
	.api-key-provider {
		@apply text-sm text-dark-400 capitalize;
	}
	
	.api-key-details {
		@apply flex-1 mx-6 space-y-2;
	}
	
	.api-key-value {
		@apply flex items-center space-x-2;
	}
	
	.key-text {
		@apply font-mono text-sm text-dark-300;
	}
	
	.toggle-visibility {
		@apply p-1 text-dark-400 hover:text-white transition-colors duration-200;
	}
	
	.api-key-meta {
		@apply flex items-center space-x-4 text-xs text-dark-400;
	}
	
	.api-key-actions {
		@apply flex items-center space-x-2;
	}

	/* RAG Settings */
	.rag-overview {
		@apply space-y-6;
	}

	.section-description {
		@apply text-dark-300 leading-relaxed;
	}

	.rag-features {
		@apply grid grid-cols-1 md:grid-cols-3 gap-4;
	}

	.feature-item {
		@apply flex items-start space-x-3 p-4 bg-dark-700 rounded-lg;
	}

	.feature-icon {
		@apply p-2 bg-dark-600 rounded-lg;
	}

	.feature-content {
		@apply space-y-1;
	}

	.feature-title {
		@apply font-medium text-white;
	}

	.feature-description {
		@apply text-sm text-dark-300;
	}

	/* Mobile Responsiveness */
	@media (max-width: 1024px) {
		.settings-interface {
			@apply grid-cols-1;
		}
		
		.nav-list {
			@apply flex overflow-x-auto space-y-0 space-x-2 p-2;
		}
		
		.nav-item {
			@apply flex-shrink-0;
		}
		
		.form-grid {
			@apply grid-cols-1;
		}
		
		.table-header,
		.table-row {
			@apply grid-cols-1 gap-2;
		}
		
		.header-cell {
			@apply hidden;
		}
		
		.cell {
			@apply flex items-center justify-between border-b border-dark-600 pb-2 last:border-b-0;
		}
		
		.cell::before {
			content: attr(data-label);
			@apply font-medium text-dark-400 mr-2;
		}
	}
</style>
