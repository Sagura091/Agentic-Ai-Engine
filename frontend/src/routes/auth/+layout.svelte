<!--
Revolutionary Authentication Layout
Clean, modern authentication interface with animated backgrounds
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { authService } from '$lib/services/authService';
	import { theme } from '$lib/stores';
	import { Sparkles, Shield, Zap } from 'lucide-svelte';

	// Animated background particles
	let particles: Array<{ x: number; y: number; size: number; speed: number }> = [];
	let animationFrame: number;

	onMount(() => {
		// Check if already authenticated
		if (authService.isAuthenticated()) {
			goto('/');
			return;
		}

		// Initialize animated background
		initializeParticles();
		animate();

		return () => {
			if (animationFrame) {
				cancelAnimationFrame(animationFrame);
			}
		};
	});

	function initializeParticles() {
		particles = Array.from({ length: 50 }, () => ({
			x: Math.random() * window.innerWidth,
			y: Math.random() * window.innerHeight,
			size: Math.random() * 3 + 1,
			speed: Math.random() * 0.5 + 0.1
		}));
	}

	function animate() {
		particles = particles.map(particle => ({
			...particle,
			y: particle.y - particle.speed,
			x: particle.x + Math.sin(particle.y * 0.01) * 0.5
		})).filter(particle => particle.y > -10)
		.concat(
			particles.length < 50 ? [{
				x: Math.random() * window.innerWidth,
				y: window.innerHeight + 10,
				size: Math.random() * 3 + 1,
				speed: Math.random() * 0.5 + 0.1
			}] : []
		);

		animationFrame = requestAnimationFrame(animate);
	}

	$: currentPath = $page.url.pathname;
</script>

<div class="auth-layout min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 relative overflow-hidden">
	<!-- Animated Background -->
	<div class="absolute inset-0 overflow-hidden">
		{#each particles as particle}
			<div 
				class="absolute rounded-full bg-primary-400/20 animate-pulse"
				style="left: {particle.x}px; top: {particle.y}px; width: {particle.size}px; height: {particle.size}px;"
			></div>
		{/each}
	</div>

	<!-- Background Grid -->
	<div class="absolute inset-0 bg-grid-pattern opacity-5"></div>

	<!-- Header -->
	<header class="relative z-10 p-6">
		<div class="flex items-center space-x-3">
			<div class="w-10 h-10 bg-gradient-to-r from-primary-500 to-accent-blue rounded-xl flex items-center justify-center">
				<Sparkles class="w-6 h-6 text-white" />
			</div>
			<div>
				<h1 class="text-xl font-bold text-white">Agentic AI Platform</h1>
				<p class="text-sm text-dark-300">Revolutionary AI Agent Builder</p>
			</div>
		</div>
	</header>

	<!-- Main Content -->
	<main class="relative z-10 flex items-center justify-center min-h-[calc(100vh-120px)] px-6">
		<div class="w-full max-w-md">
			<!-- Auth Card -->
			<div class="bg-dark-800/80 backdrop-blur-xl border border-dark-700/50 rounded-2xl p-8 shadow-2xl">
				<slot />
			</div>

			<!-- Features Preview -->
			<div class="mt-8 grid grid-cols-3 gap-4 text-center">
				<div class="bg-dark-800/40 backdrop-blur-sm rounded-xl p-4 border border-dark-700/30">
					<Shield class="w-6 h-6 text-green-400 mx-auto mb-2" />
					<p class="text-xs text-dark-300">Enterprise Security</p>
				</div>
				<div class="bg-dark-800/40 backdrop-blur-sm rounded-xl p-4 border border-dark-700/30">
					<Zap class="w-6 h-6 text-yellow-400 mx-auto mb-2" />
					<p class="text-xs text-dark-300">Lightning Fast</p>
				</div>
				<div class="bg-dark-800/40 backdrop-blur-sm rounded-xl p-4 border border-dark-700/30">
					<Sparkles class="w-6 h-6 text-purple-400 mx-auto mb-2" />
					<p class="text-xs text-dark-300">AI Powered</p>
				</div>
			</div>
		</div>
	</main>

	<!-- Footer -->
	<footer class="relative z-10 p-6 text-center">
		<p class="text-sm text-dark-400">
			© 2024 Agentic AI Platform. All rights reserved.
		</p>
	</footer>
</div>

<style lang="postcss">
	.bg-grid-pattern {
		background-image: 
			linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
			linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px);
		background-size: 50px 50px;
	}

	.auth-layout {
		animation: fadeIn 0.8s ease-out;
	}

	@keyframes fadeIn {
		from {
			opacity: 0;
			transform: translateY(20px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
</style>
