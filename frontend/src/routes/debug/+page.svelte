<script lang="ts">
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { apiService } from '$lib/services/api';

	let debugInfo = {
		mounted: false,
		apiCallMade: false,
		apiResult: null as any,
		apiError: null as any,
		authStoreResult: null as any,
		authStoreError: null as any
	};

	onMount(async () => {
		console.log('🎬 Debug page: Starting tests');
		debugInfo.mounted = true;
		debugInfo = { ...debugInfo }; // Trigger reactivity

		// Test 1: Direct API call
		try {
			console.log('🔍 Debug: Testing direct API call');
			debugInfo.apiCallMade = true;
			debugInfo = { ...debugInfo };
			
			const result = await apiService.checkFirstTimeSetup();
			console.log('✅ Debug: Direct API call successful:', result);
			debugInfo.apiResult = result;
			debugInfo = { ...debugInfo };
		} catch (error) {
			console.error('❌ Debug: Direct API call failed:', error);
			debugInfo.apiError = error;
			debugInfo = { ...debugInfo };
		}

		// Test 2: Auth store call
		try {
			console.log('🔍 Debug: Testing auth store call');
			const authResult = await authStore.checkFirstTimeSetup();
			console.log('✅ Debug: Auth store call successful:', authResult);
			debugInfo.authStoreResult = authResult;
			debugInfo = { ...debugInfo };
		} catch (error) {
			console.error('❌ Debug: Auth store call failed:', error);
			debugInfo.authStoreError = error;
			debugInfo = { ...debugInfo };
		}

		console.log('🏁 Debug page: All tests completed');
	});
</script>

<svelte:head>
	<title>Debug - First Time Setup</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
	<div class="max-w-4xl mx-auto">
		<h1 class="text-4xl font-bold text-white mb-8">🔍 First-Time Setup Debug</h1>
		
		<div class="grid gap-6">
			<!-- Mount Status -->
			<div class="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
				<h2 class="text-xl font-semibold text-white mb-4">📱 Component Status</h2>
				<div class="space-y-2">
					<div class="flex items-center space-x-2">
						<div class="w-3 h-3 rounded-full {debugInfo.mounted ? 'bg-green-400' : 'bg-red-400'}"></div>
						<span class="text-white">Component Mounted: {debugInfo.mounted ? 'Yes' : 'No'}</span>
					</div>
				</div>
			</div>

			<!-- Direct API Test -->
			<div class="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
				<h2 class="text-xl font-semibold text-white mb-4">🌐 Direct API Test</h2>
				<div class="space-y-4">
					<div class="flex items-center space-x-2">
						<div class="w-3 h-3 rounded-full {debugInfo.apiCallMade ? 'bg-green-400' : 'bg-gray-400'}"></div>
						<span class="text-white">API Call Made: {debugInfo.apiCallMade ? 'Yes' : 'No'}</span>
					</div>
					
					{#if debugInfo.apiResult}
						<div class="bg-green-500/20 border border-green-500/30 rounded-lg p-4">
							<h3 class="text-green-300 font-semibold mb-2">✅ API Success</h3>
							<pre class="text-green-200 text-sm overflow-auto">{JSON.stringify(debugInfo.apiResult, null, 2)}</pre>
						</div>
					{/if}
					
					{#if debugInfo.apiError}
						<div class="bg-red-500/20 border border-red-500/30 rounded-lg p-4">
							<h3 class="text-red-300 font-semibold mb-2">❌ API Error</h3>
							<pre class="text-red-200 text-sm overflow-auto">{JSON.stringify(debugInfo.apiError, null, 2)}</pre>
						</div>
					{/if}
				</div>
			</div>

			<!-- Auth Store Test -->
			<div class="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
				<h2 class="text-xl font-semibold text-white mb-4">🔐 Auth Store Test</h2>
				<div class="space-y-4">
					{#if debugInfo.authStoreResult !== null}
						<div class="bg-green-500/20 border border-green-500/30 rounded-lg p-4">
							<h3 class="text-green-300 font-semibold mb-2">✅ Auth Store Success</h3>
							<p class="text-green-200">Result: {debugInfo.authStoreResult}</p>
						</div>
					{/if}
					
					{#if debugInfo.authStoreError}
						<div class="bg-red-500/20 border border-red-500/30 rounded-lg p-4">
							<h3 class="text-red-300 font-semibold mb-2">❌ Auth Store Error</h3>
							<pre class="text-red-200 text-sm overflow-auto">{JSON.stringify(debugInfo.authStoreError, null, 2)}</pre>
						</div>
					{/if}
				</div>
			</div>

			<!-- Manual Test Button -->
			<div class="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
				<h2 class="text-xl font-semibold text-white mb-4">🔧 Manual Tests</h2>
				<div class="space-y-4">
					<button 
						on:click={async () => {
							try {
								const result = await fetch('/api/v1/auth/setup/status');
								const data = await result.json();
								console.log('Manual fetch result:', data);
								alert('Manual fetch success: ' + JSON.stringify(data));
							} catch (error) {
								console.error('Manual fetch error:', error);
								alert('Manual fetch error: ' + error);
							}
						}}
						class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
					>
						Test Manual Fetch
					</button>
					
					<button 
						on:click={() => {
							window.location.href = '/login';
						}}
						class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-colors"
					>
						Go to Login Page
					</button>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	pre {
		white-space: pre-wrap;
		word-break: break-all;
	}
</style>
