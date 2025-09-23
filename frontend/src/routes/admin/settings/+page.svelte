<script lang="ts">
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { goto } from '$app/navigation';
	import { fly } from 'svelte/transition';
	import {
		Settings, Users, TrendingUp, Shield,
		Monitor, Save, RefreshCw, AlertTriangle,
		Check, AlertCircle, Search, ChevronLeft, ChevronRight
	} from 'lucide-svelte';

	let loading = false;
	let activeTab = 'overview';
	let saveMessage = '';

	// System data
	let systemStats = {
		total_users: 0,
		active_users: 0,
		admin_users: 0,
		total_agents: 0,
		total_workflows: 0,
		system_uptime: 'N/A',
		database_size: 'N/A'
	};

	let userManagement = {
		users: [],
		total_count: 0,
		active_count: 0,
		admin_count: 0
	};

	let systemSettings = {
		general: {},
		security: {},
		agents: {},
		models: {},
		monitoring: {}
	};

	// User management filters
	let userSearch = '';
	let currentPage = 1;
	let usersPerPage = 20;

	const tabs = [
		{ id: 'overview', label: 'Overview', icon: TrendingUp },
		{ id: 'users', label: 'User Management', icon: Users },
		{ id: 'system', label: 'System Settings', icon: Settings },
		{ id: 'security', label: 'Security', icon: Shield },
		{ id: 'monitoring', label: 'Monitoring', icon: Monitor }
	];

	onMount(() => {
		// Check if user is admin
		if (!$authStore.user?.user_group || $authStore.user.user_group !== 'admin') {
			goto('/dashboard');
			return;
		}
		
		loadSystemStats();
		loadSystemSettings();
	});

	async function loadSystemStats() {
		loading = true;
		try {
			const response = await fetch('/api/admin/settings/stats', {
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				}
			});

			if (response.ok) {
				const data = await response.json();
				if (data.success) {
					systemStats = data.data;
				}
			}
		} catch (error) {
			console.error('Failed to load system stats:', error);
		} finally {
			loading = false;
		}
	}

	async function loadUserManagement() {
		loading = true;
		try {
			const params = new URLSearchParams({
				page: currentPage.toString(),
				limit: usersPerPage.toString()
			});
			
			if (userSearch) {
				params.append('search', userSearch);
			}

			const response = await fetch(`/api/admin/settings/users?${params}`, {
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				}
			});

			if (response.ok) {
				const data = await response.json();
				if (data.success) {
					userManagement = data.data;
				}
			}
		} catch (error) {
			console.error('Failed to load user management data:', error);
		} finally {
			loading = false;
		}
	}

	async function loadSystemSettings() {
		loading = true;
		try {
			const response = await fetch('/api/admin/settings/system', {
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				}
			});

			if (response.ok) {
				const data = await response.json();
				if (data.success) {
					systemSettings = data.data;
				}
			}
		} catch (error) {
			console.error('Failed to load system settings:', error);
		} finally {
			loading = false;
		}
	}

	async function updateSystemSettings(category: string, settings: any) {
		loading = true;
		try {
			const response = await fetch('/api/admin/settings/system', {
				method: 'POST',
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					category: category,
					settings: settings
				})
			});

			const data = await response.json();
			if (data.success) {
				saveMessage = `${category} settings updated successfully`;
				// Update local settings
				systemSettings[category] = { ...systemSettings[category], ...settings };
			} else {
				saveMessage = data.message || 'Failed to update settings';
			}
		} catch (error) {
			saveMessage = 'Failed to update settings';
		} finally {
			loading = false;
		}
	}

	function formatDate(dateString: string) {
		if (!dateString) return 'Never';
		return new Date(dateString).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	function getRoleColor(role: string) {
		switch (role) {
			case 'admin': return 'text-red-400 bg-red-500/20';
			case 'moderator': return 'text-yellow-400 bg-yellow-500/20';
			default: return 'text-blue-400 bg-blue-500/20';
		}
	}

	// Handle tab changes
	$: if (activeTab === 'users' && userManagement.users.length === 0) {
		loadUserManagement();
	}

	// Handle user search
	$: if (userSearch !== undefined) {
		currentPage = 1;
		if (activeTab === 'users') {
			loadUserManagement();
		}
	}

	// Clear save message after 3 seconds
	$: if (saveMessage) {
		setTimeout(() => {
			saveMessage = '';
		}, 3000);
	}
</script>

<svelte:head>
	<title>Admin Settings - Agentic AI Platform</title>
</svelte:head>

<div class="min-h-screen p-6">
	<div class="max-w-7xl mx-auto">
		<!-- Header -->
		<div class="mb-8">
			<div class="flex items-center space-x-3 mb-2">
				<div class="w-10 h-10 bg-gradient-to-br from-red-400 to-orange-400 rounded-xl flex items-center justify-center">
					<Shield class="w-5 h-5 text-white" />
				</div>
				<div>
					<h1 class="text-3xl font-bold text-white">Admin Settings</h1>
					<p class="text-white/60">System administration and configuration</p>
				</div>
			</div>
			
			<!-- Warning Banner -->
			<div class="glass rounded-xl p-4 border border-yellow-500/30 bg-yellow-500/10">
				<div class="flex items-center space-x-3">
					<AlertTriangle class="w-5 h-5 text-yellow-400" />
					<div>
						<p class="text-yellow-400 font-medium">Administrator Access</p>
						<p class="text-yellow-300/80 text-sm">You have full system access. Please use these settings carefully.</p>
					</div>
				</div>
			</div>
		</div>

		<div class="flex flex-col lg:flex-row gap-6">
			<!-- Sidebar Navigation -->
			<div class="lg:w-64">
				<div class="glass rounded-2xl p-4 sticky top-6">
					<nav class="space-y-2">
						{#each tabs as tab}
							<button
								on:click={() => activeTab = tab.id}
								class="w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200
									{activeTab === tab.id 
										? 'bg-gradient-to-r from-red-500/20 to-orange-500/20 text-white border border-red-400/30' 
										: 'text-white/70 hover:text-white hover:bg-white/5'
									}"
							>
								<svelte:component this={tab.icon} class="w-5 h-5" />
								<span class="font-medium">{tab.label}</span>
							</button>
						{/each}
					</nav>
				</div>
			</div>

			<!-- Main Content -->
			<div class="flex-1">
				{#if loading && activeTab === 'overview'}
					<!-- Loading State -->
					<div class="glass rounded-2xl p-8 flex items-center justify-center">
						<div class="w-8 h-8 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
					</div>
				{:else}
					<!-- Overview Tab -->
					{#if activeTab === 'overview'}
						<div class="space-y-6">
							<!-- System Stats Cards -->
							<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
								<div class="glass rounded-xl p-6">
									<div class="flex items-center justify-between">
										<div>
											<p class="text-white/60 text-sm">Total Users</p>
											<p class="text-2xl font-bold text-white">{systemStats.total_users}</p>
										</div>
										<Users class="w-8 h-8 text-blue-400" />
									</div>
								</div>

								<div class="glass rounded-xl p-6">
									<div class="flex items-center justify-between">
										<div>
											<p class="text-white/60 text-sm">Active Users</p>
											<p class="text-2xl font-bold text-white">{systemStats.active_users}</p>
										</div>
										<Monitor class="w-8 h-8 text-green-400" />
									</div>
								</div>

								<div class="glass rounded-xl p-6">
									<div class="flex items-center justify-between">
										<div>
											<p class="text-white/60 text-sm">Total Agents</p>
											<p class="text-2xl font-bold text-white">{systemStats.total_agents}</p>
										</div>
										<Settings class="w-8 h-8 text-purple-400" />
									</div>
								</div>

								<div class="glass rounded-xl p-6">
									<div class="flex items-center justify-between">
										<div>
											<p class="text-white/60 text-sm">Admin Users</p>
											<p class="text-2xl font-bold text-white">{systemStats.admin_users}</p>
										</div>
										<Shield class="w-8 h-8 text-red-400" />
									</div>
								</div>
							</div>

							<!-- System Information -->
							<div class="glass rounded-2xl p-6">
								<h3 class="text-xl font-semibold text-white mb-4">System Information</h3>
								<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
									<div>
										<h4 class="text-white font-medium mb-2">System Uptime</h4>
										<p class="text-white/80">{systemStats.system_uptime}</p>
									</div>
									<div>
										<h4 class="text-white font-medium mb-2">Database Size</h4>
										<p class="text-white/80">{systemStats.database_size}</p>
									</div>
								</div>
							</div>

							<!-- Quick Actions -->
							<div class="glass rounded-2xl p-6">
								<h3 class="text-xl font-semibold text-white mb-4">Quick Actions</h3>
								<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
									<button
										on:click={() => activeTab = 'users'}
										class="btn-secondary flex items-center justify-center space-x-2"
									>
										<Users class="w-4 h-4" />
										<span>Manage Users</span>
									</button>
									
									<button
										on:click={() => activeTab = 'system'}
										class="btn-secondary flex items-center justify-center space-x-2"
									>
										<Settings class="w-4 h-4" />
										<span>System Settings</span>
									</button>
									
									<button
										on:click={loadSystemStats}
										disabled={loading}
										class="btn-secondary flex items-center justify-center space-x-2"
									>
										<RefreshCw class="w-4 h-4 {loading ? 'animate-spin' : ''}" />
										<span>Refresh Stats</span>
									</button>
								</div>
							</div>
						</div>
					{/if}

					<!-- User Management Tab -->
					{#if activeTab === 'users'}
						<div class="space-y-6">
							<!-- Search and Filters -->
							<div class="glass rounded-2xl p-6">
								<div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
									<h3 class="text-xl font-semibold text-white">User Management</h3>

									<div class="flex items-center space-x-3">
										<div class="relative">
											<Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-white/60" />
											<input
												type="text"
												bind:value={userSearch}
												placeholder="Search users..."
												class="form-input pl-10 w-64"
											/>
										</div>
									</div>
								</div>
							</div>

							<!-- Users Table -->
							<div class="glass rounded-2xl overflow-hidden">
								{#if loading}
									<div class="p-8 flex items-center justify-center">
										<div class="w-6 h-6 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
									</div>
								{:else}
									<div class="overflow-x-auto">
										<table class="w-full">
											<thead class="bg-white/5 border-b border-white/10">
												<tr>
													<th class="text-left p-4 text-white font-medium">User</th>
													<th class="text-left p-4 text-white font-medium">Role</th>
													<th class="text-left p-4 text-white font-medium">Status</th>
													<th class="text-left p-4 text-white font-medium">Last Login</th>
													<th class="text-left p-4 text-white font-medium">Actions</th>
												</tr>
											</thead>
											<tbody>
												{#each userManagement.users as user}
													<tr class="border-b border-white/5 hover:bg-white/5">
														<td class="p-4">
															<div>
																<p class="text-white font-medium">{user.name || user.username}</p>
																<p class="text-white/60 text-sm">{user.email}</p>
															</div>
														</td>
														<td class="p-4">
															<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium capitalize {getRoleColor(user.user_group)}">
																{user.user_group}
															</span>
														</td>
														<td class="p-4">
															<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium {user.is_active ? 'text-green-400 bg-green-500/20' : 'text-red-400 bg-red-500/20'}">
																{user.is_active ? 'Active' : 'Inactive'}
															</span>
														</td>
														<td class="p-4">
															<span class="text-white/80 text-sm">{formatDate(user.last_login)}</span>
														</td>
														<td class="p-4">
															<button class="text-white/60 hover:text-white text-sm">Edit</button>
														</td>
													</tr>
												{/each}
											</tbody>
										</table>
									</div>

									<!-- Pagination -->
									{#if userManagement.total_count > usersPerPage}
										<div class="p-4 border-t border-white/10 flex items-center justify-between">
											<p class="text-white/60 text-sm">
												Showing {((currentPage - 1) * usersPerPage) + 1} to {Math.min(currentPage * usersPerPage, userManagement.total_count)} of {userManagement.total_count} users
											</p>

											<div class="flex items-center space-x-2">
												<button
													on:click={() => { currentPage--; loadUserManagement(); }}
													disabled={currentPage <= 1}
													class="p-2 text-white/60 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
												>
													<ChevronLeft class="w-4 h-4" />
												</button>

												<span class="text-white px-3 py-1 bg-white/10 rounded">
													{currentPage}
												</span>

												<button
													on:click={() => { currentPage++; loadUserManagement(); }}
													disabled={currentPage >= Math.ceil(userManagement.total_count / usersPerPage)}
													class="p-2 text-white/60 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
												>
													<ChevronRight class="w-4 h-4" />
												</button>
											</div>
										</div>
									{/if}
								{/if}
							</div>
						</div>
					{/if}

					<!-- System Settings Tab -->
					{#if activeTab === 'system'}
						<div class="space-y-6">
							<div class="glass rounded-2xl p-6">
								<h3 class="text-xl font-semibold text-white mb-6">System Configuration</h3>

								<!-- General Settings -->
								<div class="space-y-6">
									<div>
										<h4 class="text-lg font-medium text-white mb-4">General Settings</h4>
										<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
											<div>
												<label for="app-name" class="block text-white font-medium mb-2">Application Name</label>
												<input
													id="app-name"
													type="text"
													bind:value={systemSettings.general.app_name}
													class="form-input w-full"
												/>
											</div>
											<div>
												<label for="max-users" class="block text-white font-medium mb-2">Max Users</label>
												<input
													id="max-users"
													type="number"
													bind:value={systemSettings.general.max_users}
													class="form-input w-full"
												/>
											</div>
										</div>

										<div class="mt-4">
											<label class="flex items-center space-x-3">
												<input
													type="checkbox"
													bind:checked={systemSettings.general.registration_enabled}
													class="w-5 h-5 rounded border-white/20 bg-white/10 text-primary-500 focus:ring-primary-400/50"
												/>
												<span class="text-white">Enable User Registration</span>
											</label>
										</div>

										<button
											on:click={() => updateSystemSettings('general', systemSettings.general)}
											disabled={loading}
											class="mt-4 btn-primary flex items-center space-x-2"
										>
											<Save class="w-4 h-4" />
											<span>Save General Settings</span>
										</button>
									</div>

									<!-- Security Settings -->
									<div class="border-t border-white/10 pt-6">
										<h4 class="text-lg font-medium text-white mb-4">Security Settings</h4>
										<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
											<div>
												<label for="password-min-length" class="block text-white font-medium mb-2">Minimum Password Length</label>
												<input
													id="password-min-length"
													type="number"
													bind:value={systemSettings.security.password_min_length}
													class="form-input w-full"
												/>
											</div>
											<div>
												<label for="session-timeout" class="block text-white font-medium mb-2">Session Timeout (seconds)</label>
												<input
													id="session-timeout"
													type="number"
													bind:value={systemSettings.security.session_timeout}
													class="form-input w-full"
												/>
											</div>
										</div>

										<button
											on:click={() => updateSystemSettings('security', systemSettings.security)}
											disabled={loading}
											class="mt-4 btn-primary flex items-center space-x-2"
										>
											<Save class="w-4 h-4" />
											<span>Save Security Settings</span>
										</button>
									</div>
								</div>
							</div>
						</div>
					{/if}
				{/if}
			</div>
		</div>

		<!-- Save Message -->
		{#if saveMessage}
			<div
				class="fixed bottom-6 right-6 glass rounded-xl p-4 flex items-center space-x-3"
				transition:fly={{ y: 50, duration: 300 }}
			>
				{#if saveMessage.includes('successfully')}
					<Check class="w-5 h-5 text-green-400" />
					<span class="text-green-400">{saveMessage}</span>
				{:else}
					<AlertCircle class="w-5 h-5 text-red-400" />
					<span class="text-red-400">{saveMessage}</span>
				{/if}
			</div>
		{/if}
	</div>
</div>
