<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import AppLayout from '$lib/components/AppLayout.svelte';

	// Accept params prop to avoid Svelte warnings
	export let params: Record<string, string> = {};

	// Initialize auth store on app load
	onMount(() => {
		authStore.initialize();
	});

	// Reactive statement to handle authentication redirects
	$: if ($page.url.pathname !== '/login' && $page.url.pathname !== '/register' && !$authStore.isAuthenticated) {
		// Only redirect to login if we're not already on auth pages and user is not authenticated
		if (typeof window !== 'undefined') {
			goto('/login');
		}
	}

	// Check if current page should use the app layout (exclude auth pages)
	$: isAuthPage = $page.url.pathname === '/login' || $page.url.pathname === '/register';
	$: shouldUseAppLayout = $authStore.isAuthenticated && !isAuthPage;
</script>

<svelte:head>
	<title>Agentic AI - Revolutionary AI Platform</title>
	<meta name="description" content="Revolutionary agentic AI platform with advanced automation and intelligent agents" />
</svelte:head>

{#if shouldUseAppLayout}
	<!-- Use App Layout with Navigation for authenticated pages -->
	<AppLayout>
		<slot />
	</AppLayout>
{:else}
	<!-- Use simple layout for auth pages -->
	<div class="min-h-screen animated-bg">
		<!-- Background decorative elements -->
		<div class="fixed inset-0 overflow-hidden pointer-events-none">
			<div class="absolute -top-40 -right-40 w-80 h-80 bg-white/10 rounded-full blur-3xl animate-pulse"></div>
			<div class="absolute -bottom-40 -left-40 w-80 h-80 bg-white/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
			<div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-float"></div>
		</div>

		<!-- Main content -->
		<main class="relative z-10">
			<slot />
		</main>
	</div>
{/if}

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		overflow-x: hidden;
	}
</style>
