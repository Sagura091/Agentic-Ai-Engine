<script lang="ts">
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { goto } from '$app/navigation';

	// Accept params prop to avoid Svelte warnings
	export let params: Record<string, string> = {};
	
	onMount(() => {
		// Redirect based on authentication status
		if ($authStore.isAuthenticated) {
			goto('/dashboard');
		} else {
			goto('/login');
		}
	});
</script>

<svelte:head>
	<title>Agentic AI - Revolutionary AI Platform</title>
</svelte:head>

<!-- Loading screen while redirecting -->
<div class="min-h-screen flex items-center justify-center">
	<div class="text-center">
		<div class="w-16 h-16 border-4 border-white/30 border-t-white rounded-full animate-spin mx-auto mb-4"></div>
		<p class="text-white/70 text-lg">Loading Agentic AI...</p>
	</div>
</div>
