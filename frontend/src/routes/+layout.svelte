<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';

	// SvelteKit props
	export let params: Record<string, string> = {};
	import { 
		theme, 
		connectionStatus, 
		notifications, 
		notificationActions,
		loadUIState 
	} from '$stores';
	import { websocketService } from '$services/websocket';
	import { apiClient } from '$services/api';
	
	// Components
	import Sidebar from '$components/layout/Sidebar.svelte';
	import Header from '$components/layout/Header.svelte';
	import NotificationCenter from '$components/ui/NotificationCenter.svelte';
	import LoadingOverlay from '$components/ui/LoadingOverlay.svelte';
	
	// State
	let mounted = false;
	let initializing = true;
	
	onMount(async () => {
		console.log('🚀 Initializing Revolutionary AI Agent Builder Platform...');
		
		// Load saved UI state
		loadUIState();
		
		// Mark as initialized immediately for faster perceived performance
		initializing = false;
		mounted = true;

		// Initialize WebSocket connection with detailed logging
		console.log('🔌 Attempting WebSocket connection...');
		connectionStatus.set('connecting');

		websocketService.connect().then(() => {
			console.log('✅ WebSocket connection promise resolved');
		}).catch((error) => {
			console.error('❌ WebSocket connection promise rejected:', error);
			console.warn('WebSocket connection failed, continuing without real-time features:', error.message);
		});

		// Subscribe to connection status changes from WebSocket service
		websocketService.connected.subscribe(connected => {
			console.log('🔄 WebSocket connected state changed:', connected);
			connectionStatus.set(connected ? 'connected' : 'disconnected');
			if (connected) {
				console.log('✅ WebSocket connection established - UI updated');
			} else {
				console.log('❌ WebSocket disconnected - UI updated');
			}
		});

		// Subscribe to system notifications
		websocketService.on('notification' as any, (data) => {
			notificationActions.add({
				type: data.type || 'info',
				title: data.title || 'System Notification',
				message: data.message || 'No message provided'
			});
		});

		// Subscribe to errors
		websocketService.on('error' as any, (data) => {
			notificationActions.add({
				type: 'error',
				title: 'System Error',
				message: data.message || 'An error occurred'
			});
		});
		
		// Test API connection in background (non-blocking)
		apiClient.healthCheck().then(healthCheck => {
			if (healthCheck.success) {
				console.log('✅ Backend API connection established');
			} else {
				throw new Error('Health check failed');
			}
		}).catch(error => {
			console.error('Backend API connection failed:', error);
			notificationActions.add({
				type: 'warning', // Changed from error to warning to be less alarming
				title: 'Backend Connection Issue',
				message: 'Backend API connection is slow or unavailable. Some features may be limited.'
			});
		});
		
		// Initialize theme
		const savedTheme = localStorage.getItem('theme') || 'dark';
		theme.set(savedTheme as any);
		document.documentElement.classList.toggle('dark', savedTheme === 'dark');
		
		// Theme change handler
		theme.subscribe(value => {
			document.documentElement.classList.toggle('dark', value === 'dark');
			localStorage.setItem('theme', value);
		});
		
		console.log('🎉 Platform initialized successfully!');

		// Welcome notification (reduced delay)
		setTimeout(() => {
			notificationActions.add({
				type: 'success',
				title: 'Welcome to Agentic AI!',
				message: 'The most revolutionary AI agent builder platform is ready to use.'
			});
		}, 500); // Reduced from 1000ms to 500ms
	});
	
	// Cleanup on destroy
	onMount(() => {
		return () => {
			websocketService.disconnect();
		};
	});
	
	// Handle page changes
	$: if (mounted) {
		// Update page title based on current route
		const routeTitles: Record<string, string> = {
			'/': 'Dashboard',
			'/agents': 'Agent Builder',
			'/workflows': 'Workflow Builder',
			'/testing': 'Testing Lab',
			'/monitoring': 'System Monitoring',
			'/settings': 'Settings'
		};
		
		const title = routeTitles[$page.route.id || '/'] || 'Agentic AI';
		document.title = `${title} - Revolutionary AI Agent Builder`;
	}
	
	// Keyboard shortcuts
	function handleKeydown(event: KeyboardEvent) {
		// Ctrl/Cmd + K for command palette (future feature)
		if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
			event.preventDefault();
			// TODO: Open command palette
		}
		
		// Ctrl/Cmd + / for help (future feature)
		if ((event.ctrlKey || event.metaKey) && event.key === '/') {
			event.preventDefault();
			// TODO: Open help
		}
	}
</script>

<svelte:window on:keydown={handleKeydown} />

<svelte:head>
	<title>Agentic AI - Revolutionary Agent Builder Platform</title>
	<meta name="description" content="Build, test, and deploy intelligent AI agents with the most advanced visual workflow builder." />
</svelte:head>

{#if initializing}
	<LoadingOverlay 
		message="Initializing revolutionary agent builder..."
		subMessage="Connecting to backend services..."
	/>
{:else}
	<div class="app-container min-h-screen bg-dark-900 text-white">
		<!-- Main Layout -->
		<div class="flex h-screen overflow-hidden">
			<!-- Sidebar -->
			<Sidebar />
			
			<!-- Main Content Area -->
			<div class="flex-1 flex flex-col overflow-hidden">
				<!-- Header -->
				<Header />
				
				<!-- Page Content -->
				<main class="flex-1 overflow-auto bg-dark-900">
					<div class="container mx-auto px-6 py-8 max-w-7xl">
						<slot />
					</div>
				</main>
			</div>
		</div>
		
		<!-- Notification Center -->
		<NotificationCenter />
		
		<!-- Global Loading States -->
		{#if $connectionStatus === 'connecting'}
			<div class="fixed bottom-4 right-4 z-50">
				<div class="glass rounded-lg p-3 flex items-center space-x-2">
					<div class="w-4 h-4 border-2 border-accent-blue border-t-transparent rounded-full animate-spin"></div>
					<span class="text-sm text-dark-300">Connecting...</span>
				</div>
			</div>
		{/if}
		
		{#if $connectionStatus === 'disconnected'}
			<div class="fixed bottom-4 right-4 z-50">
				<div class="bg-red-500/20 border border-red-500/30 rounded-lg p-3 flex items-center space-x-2">
					<div class="w-2 h-2 bg-red-500 rounded-full"></div>
					<span class="text-sm text-red-300">Disconnected</span>
				</div>
			</div>
		{/if}
	</div>
{/if}

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
		background-color: #0f172a;
		color: white;
		overflow-x: hidden;
	}
	
	:global(.app-container) {
		min-height: 100vh;
		background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
	}
	
	:global(.glass) {
		background: rgba(255, 255, 255, 0.05);
		backdrop-filter: blur(10px);
		border: 1px solid rgba(255, 255, 255, 0.1);
	}
	
	:global(.text-gradient) {
		background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
	}
	
	/* Custom scrollbar for the entire app */
	:global(::-webkit-scrollbar) {
		width: 8px;
		height: 8px;
	}
	
	:global(::-webkit-scrollbar-track) {
		background: #1e293b;
		border-radius: 4px;
	}
	
	:global(::-webkit-scrollbar-thumb) {
		background: #475569;
		border-radius: 4px;
	}
	
	:global(::-webkit-scrollbar-thumb:hover) {
		background: #64748b;
	}
	
	/* Smooth transitions for all elements */
	:global(*) {
		transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
	}
	
	/* Focus styles */
	:global(:focus-visible) {
		outline: 2px solid #3b82f6;
		outline-offset: 2px;
		border-radius: 4px;
	}
	
	/* Selection styles */
	:global(::selection) {
		background: rgba(59, 130, 246, 0.3);
		color: white;
	}
</style>
