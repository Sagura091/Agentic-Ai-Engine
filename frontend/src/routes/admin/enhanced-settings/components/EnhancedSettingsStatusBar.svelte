<!--
Enhanced Settings Status Bar Component

This component renders the system status indicators for the Enhanced Admin Settings page,
including WebSocket status, configuration status, and real-time update indicators.
-->

<script lang="ts">
	import { Activity } from 'lucide-svelte';

	// Props
	export let websocketConnected: boolean = true;
	export let configurationVerified: boolean = true;
	export let lastUpdateTime: Date = new Date();

	// Reactive statements
	$: formattedTime = lastUpdateTime.toLocaleTimeString();
</script>

<!-- System Status Indicator -->
<div class="max-w-7xl mx-auto px-4 py-1">
	<div class="flex items-center justify-between text-xs text-gray-400">
		<div class="flex items-center space-x-4">
			<!-- WebSocket Status -->
			<div class="flex items-center space-x-1">
				<div class="w-2 h-2 rounded-full {websocketConnected ? 'bg-green-400' : 'bg-red-400'}"></div>
				<span>{websocketConnected ? 'Real-time Updates Active' : 'Connection Lost'}</span>
			</div>

			<!-- Configuration Status -->
			<div class="flex items-center space-x-1">
				<Activity class="w-3 h-3" />
				<span>{configurationVerified ? 'Configuration Verified' : 'Configuration Issues'}</span>
			</div>
		</div>

		<!-- Last Update Time -->
		<div class="text-gray-500">
			Last updated: {formattedTime}
		</div>
	</div>
</div>
