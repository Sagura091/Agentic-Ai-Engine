<!--
Enhanced Settings Header Component

This component renders the header section of the Enhanced Admin Settings page,
including status indicators, save functionality, and glass morphism styling.
-->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import {
		Settings,
		AlertTriangle,
		RefreshCw,
		Save,
		XCircle
	} from 'lucide-svelte';

	// Props
	export let unsavedChanges: Record<string, any> = {};
	export let loading: boolean = false;

	// Event dispatcher
	const dispatch = createEventDispatcher();

	// Event handlers
	function handleSaveAll() {
		dispatch('saveAll');
	}
</script>

<!-- Revolutionary Compact Settings Interface Header -->
<div class="glass border-b border-white/10 sticky top-0 z-50">
	<div class="max-w-7xl mx-auto px-4 py-3">
		<div class="flex items-center justify-between">
			<div class="flex items-center space-x-3">
				<div class="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white">
					<Settings class="w-5 h-5" />
				</div>
				<div>
					<h1 class="text-xl font-bold text-white">Enhanced Settings</h1>
					<p class="text-xs text-white/60">System Configuration</p>
				</div>
			</div>

			<!-- Compact Status Bar -->
			<div class="flex items-center space-x-2">
				{#if Object.keys(unsavedChanges).length > 0}
					<div class="flex items-center space-x-1 px-2 py-1 bg-amber-500/20 text-amber-300 rounded-md text-xs">
						<AlertTriangle class="w-3 h-3" />
						<span>{Object.keys(unsavedChanges).length}</span>
					</div>
				{/if}

				{#if loading}
					<div class="flex items-center space-x-1 px-2 py-1 bg-blue-500/20 text-blue-300 rounded-md text-xs">
						<RefreshCw class="w-3 h-3 animate-spin" />
					</div>
				{/if}

				<!-- Quick Save All Button -->
				{#if Object.keys(unsavedChanges).length > 0}
					<button
						on:click={handleSaveAll}
						class="px-3 py-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white text-xs rounded-md transition-all duration-200 flex items-center space-x-1"
					>
						<Save class="w-3 h-3" />
						<span>Save All</span>
					</button>
				{/if}
			</div>
		</div>
	</div>
</div>

<style>
	.glass {
		background: rgba(255, 255, 255, 0.05);
		backdrop-filter: blur(10px);
		-webkit-backdrop-filter: blur(10px);
		border: 1px solid rgba(255, 255, 255, 0.1);
	}
</style>
