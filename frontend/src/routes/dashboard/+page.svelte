<script lang="ts">
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { goto } from '$app/navigation';
	import { User, LogOut, Zap, Activity, Brain, Sparkles, Crown } from 'lucide-svelte';
	import { fly, scale } from 'svelte/transition';
	import { quintOut } from 'svelte/easing';
	
	let mounted = false;
	
	onMount(() => {
		mounted = true;
		
		// Redirect if not authenticated
		if (!$authStore.isAuthenticated) {
			goto('/login');
		}
	});
	
	async function handleLogout() {
		await authStore.logout();
	}
</script>

<svelte:head>
	<title>Dashboard - Agentic AI</title>
</svelte:head>

{#if mounted && $authStore.isAuthenticated}
	<div class="min-h-screen p-6">
		<!-- Header -->
		<header 
			class="glass rounded-2xl p-6 mb-8 flex items-center justify-between"
			transition:fly={{ y: -50, duration: 600, easing: quintOut }}
		>
			<div class="flex items-center space-x-4">
				<div class="w-12 h-12 bg-gradient-to-br from-primary-400 to-accent-400 rounded-full flex items-center justify-center">
					<Brain class="w-6 h-6 text-white" />
				</div>
				<div>
					<h1 class="text-2xl font-bold text-white">Welcome back, {$authStore.user?.full_name || $authStore.user?.username}!</h1>
					<p class="text-white/70">Ready to revolutionize with AI?</p>
				</div>
			</div>
			
			<button
				on:click={handleLogout}
				class="flex items-center space-x-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-200 rounded-lg transition-colors duration-300"
			>
				<LogOut class="w-4 h-4" />
				<span>Logout</span>
			</button>
		</header>
		
		<!-- Dashboard Content -->
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
			<!-- User Info Card -->
			<div 
				class="card"
				transition:scale={{ duration: 400, delay: 200, easing: quintOut }}
			>
				<div class="flex items-center space-x-4 mb-4">
					<div class="w-10 h-10 bg-gradient-to-br from-primary-400 to-accent-400 rounded-full flex items-center justify-center">
						<User class="w-5 h-5 text-white" />
					</div>
					<h3 class="text-xl font-semibold text-white">Profile</h3>
				</div>
				<div class="space-y-2 text-white/80">
					<p><span class="font-medium">Username:</span> {$authStore.user?.username}</p>
					<p><span class="font-medium">Email:</span> {$authStore.user?.email}</p>
					<p><span class="font-medium">Status:</span>
						<span class="text-green-400">{$authStore.user?.is_active ? 'Active' : 'Inactive'}</span>
					</p>
					{#if $authStore.user?.is_admin || $authStore.user?.is_superuser}
						<p><span class="font-medium">Role:</span>
							<span class="text-yellow-400 flex items-center gap-1">
								<Crown class="w-4 h-4" />
								Administrator
							</span>
						</p>
					{/if}
				</div>
			</div>
			
			<!-- Activity Card -->
			<div 
				class="card"
				transition:scale={{ duration: 400, delay: 400, easing: quintOut }}
			>
				<div class="flex items-center space-x-4 mb-4">
					<div class="w-10 h-10 bg-gradient-to-br from-secondary-400 to-primary-400 rounded-full flex items-center justify-center">
						<Activity class="w-5 h-5 text-white" />
					</div>
					<h3 class="text-xl font-semibold text-white">Activity</h3>
				</div>
				<div class="space-y-2 text-white/80">
					<p><span class="font-medium">Last Login:</span> {$authStore.user?.last_login ? new Date($authStore.user.last_login).toLocaleDateString() : 'First time!'}</p>
					<p><span class="font-medium">Account Created:</span> {new Date($authStore.user?.created_at || '').toLocaleDateString()}</p>
				</div>
			</div>
			
			<!-- AI Features Card -->
			<div 
				class="card"
				transition:scale={{ duration: 400, delay: 600, easing: quintOut }}
			>
				<div class="flex items-center space-x-4 mb-4">
					<div class="w-10 h-10 bg-gradient-to-br from-accent-400 to-secondary-400 rounded-full flex items-center justify-center">
						<Sparkles class="w-5 h-5 text-white animate-pulse" />
					</div>
					<h3 class="text-xl font-semibold text-white">AI Features</h3>
				</div>
				<div class="space-y-3">
					<button class="w-full p-3 bg-gradient-to-r from-primary-500/20 to-accent-500/20 hover:from-primary-500/30 hover:to-accent-500/30 rounded-lg text-white transition-all duration-300 btn-hover">
						Launch AI Agent
					</button>
					<button class="w-full p-3 bg-gradient-to-r from-secondary-500/20 to-primary-500/20 hover:from-secondary-500/30 hover:to-primary-500/30 rounded-lg text-white transition-all duration-300 btn-hover">
						Create Workflow
					</button>
				</div>
			</div>
		</div>
		
		<!-- Success Message -->
		<div 
			class="mt-8 glass rounded-2xl p-6 text-center"
			transition:fly={{ y: 50, duration: 600, delay: 800, easing: quintOut }}
		>
			<div class="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-green-400 to-blue-400 rounded-full mb-4 animate-bounce-subtle">
				<Zap class="w-8 h-8 text-white" />
			</div>
			<h2 class="text-2xl font-bold text-white mb-2">🎉 Authentication Successful!</h2>
			<p class="text-white/70 text-lg">You have successfully logged into the revolutionary Agentic AI platform. The frontend and backend are now fully connected and working together seamlessly!</p>
		</div>
	</div>
{/if}
