<!--
Enhanced Settings Messages Component

This component renders error and success notifications for the Enhanced Admin Settings page,
maintaining consistent styling and user interaction patterns.
-->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import {
		XCircle,
		CheckCircle
	} from 'lucide-svelte';

	// Props
	export let error: string | null = null;
	export let success: string | null = null;

	// Event dispatcher
	const dispatch = createEventDispatcher();

	// Event handlers
	function handleClearMessages() {
		dispatch('clearMessages');
	}
</script>

<!-- Compact Messages -->
{#if error}
	<div class="max-w-7xl mx-auto px-4 py-2">
		<div class="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center space-x-2">
			<XCircle class="w-4 h-4 text-red-400 flex-shrink-0" />
			<p class="text-red-300 text-sm flex-1">{error}</p>
			<button 
				on:click={handleClearMessages} 
				class="text-red-400 hover:text-red-300 transition-colors"
				aria-label="Clear error message"
			>
				<XCircle class="w-3 h-3" />
			</button>
		</div>
	</div>
{/if}

{#if success}
	<div class="max-w-7xl mx-auto px-4 py-2">
		<div class="bg-green-500/10 border border-green-500/20 rounded-lg p-3 flex items-center space-x-2">
			<CheckCircle class="w-4 h-4 text-green-400 flex-shrink-0" />
			<p class="text-green-300 text-sm flex-1">{success}</p>
			<button 
				on:click={handleClearMessages} 
				class="text-green-400 hover:text-green-300 transition-colors"
				aria-label="Clear success message"
			>
				<XCircle class="w-3 h-3" />
			</button>
		</div>
	</div>
{/if}
