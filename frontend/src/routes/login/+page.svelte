<script lang="ts">
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { goto } from '$app/navigation';
	import { Mail, Lock, Eye, EyeOff, Zap, Sparkles, ArrowRight, User, Crown, Shield } from 'lucide-svelte';
	import { fly, scale } from 'svelte/transition';
	import { quintOut, elasticOut } from 'svelte/easing';
	import RegisterModal from '$lib/components/RegisterModal.svelte';

	let credentials = {
		username_or_email: '',
		password: ''
	};

	let registrationData = {
		username: '',
		email: '',
		password: '',
		confirmPassword: '',
		full_name: ''
	};

	let showPassword = false;
	let showRegPassword = false;
	let showRegConfirmPassword = false;
	let isSubmitting = false;
	let showRegisterModal = false;
	let mounted = false;
	let isFirstTimeSetup = false;
	let showRegistrationForm = false;
	let registrationErrors: Record<string, string> = {};

	$: passwordsMatch = registrationData.password === registrationData.confirmPassword;
	$: isRegFormValid = registrationData.username && registrationData.email && registrationData.password && passwordsMatch;

	onMount(async () => {
		mounted = true;
		// Clear any existing errors
		authStore.clearError();

		// Redirect if already authenticated
		if ($authStore.isAuthenticated) {
			goto('/dashboard');
			return;
		}

		// Check if this is first-time setup
		console.log('Checking first-time setup...');
		try {
			isFirstTimeSetup = await authStore.checkFirstTimeSetup();
			console.log('First-time setup result:', isFirstTimeSetup);
			if (isFirstTimeSetup) {
				showRegistrationForm = true;
				console.log('Showing registration form for first-time setup');
			} else {
				console.log('Showing login form - users exist');
			}
		} catch (error) {
			console.error('Error checking first-time setup:', error);
		}
	});

	async function handleLogin() {
		if (!credentials.username_or_email || !credentials.password) return;

		isSubmitting = true;

		try {
			await authStore.login(credentials);
		} catch (error) {
			console.error('Login failed:', error);
		} finally {
			isSubmitting = false;
		}
	}

	function validateRegistration() {
		registrationErrors = {};

		if (!registrationData.username) registrationErrors.username = 'Username is required';
		if (!registrationData.email) registrationErrors.email = 'Email is required';
		if (!registrationData.password) registrationErrors.password = 'Password is required';
		if (registrationData.password.length < 8) registrationErrors.password = 'Password must be at least 8 characters';
		if (!passwordsMatch) registrationErrors.confirmPassword = 'Passwords do not match';

		return Object.keys(registrationErrors).length === 0;
	}

	async function handleRegistration() {
		if (!validateRegistration()) return;

		isSubmitting = true;

		try {
			await authStore.register({
				username: registrationData.username,
				email: registrationData.email,
				password: registrationData.password,
				full_name: registrationData.full_name || undefined
			});
		} catch (error) {
			console.error('Registration failed:', error);
		} finally {
			isSubmitting = false;
		}
	}

	function openRegisterModal() {
		showRegisterModal = true;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !isSubmitting) {
			if (showRegistrationForm) {
				handleRegistration();
			} else {
				handleLogin();
			}
		}
	}
</script>

<svelte:head>
	<title>Login - Agentic AI</title>
</svelte:head>

<svelte:window on:keydown={handleKeydown} />

<div class="min-h-screen flex items-center justify-center p-4 relative overflow-hidden">
	<!-- Modern animated background -->
	<div class="absolute inset-0 overflow-hidden">
		<!-- Gradient mesh background -->
		<div class="absolute inset-0 bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900"></div>
		<div class="absolute inset-0 bg-gradient-to-tl from-blue-800/50 via-transparent to-cyan-800/50"></div>

		<!-- Floating geometric shapes -->
		<div class="absolute top-1/4 left-1/4 w-32 h-32 bg-gradient-to-br from-blue-400/20 to-purple-400/20 rounded-full blur-xl animate-float"></div>
		<div class="absolute top-3/4 right-1/4 w-24 h-24 bg-gradient-to-br from-pink-400/20 to-red-400/20 rounded-full blur-xl animate-float" style="animation-delay: 2s;"></div>
		<div class="absolute top-1/2 left-3/4 w-40 h-40 bg-gradient-to-br from-cyan-400/20 to-blue-400/20 rounded-full blur-xl animate-float" style="animation-delay: 4s;"></div>

		<!-- Animated grid pattern -->
		<div class="absolute inset-0 opacity-10">
			<div class="absolute inset-0" style="background-image: radial-gradient(circle at 1px 1px, rgba(255,255,255,0.3) 1px, transparent 0); background-size: 50px 50px;"></div>
		</div>
	</div>
	
	<!-- Main container -->
	{#if mounted}
		<div
			class="glass-strong rounded-3xl p-8 w-full max-w-lg relative overflow-hidden shadow-2xl border border-white/10"
			transition:scale={{ duration: 600, easing: quintOut }}
		>
			<!-- Animated gradient overlay -->
			<div class="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-pink-500/5 animate-gradient-x"></div>

			<!-- Floating decorative elements -->
			<div class="absolute -top-10 -right-10 w-20 h-20 bg-gradient-to-br from-blue-400/20 to-purple-400/20 rounded-full blur-2xl animate-pulse"></div>
			<div class="absolute -bottom-10 -left-10 w-20 h-20 bg-gradient-to-br from-pink-400/20 to-blue-400/20 rounded-full blur-2xl animate-pulse delay-1000"></div>

			<!-- Header -->
			<div class="text-center mb-8 relative z-10">
				{#if showRegistrationForm}
					<div
						class="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-emerald-400 to-blue-400 rounded-full mb-6 shadow-lg animate-bounce-subtle"
						transition:scale={{ duration: 800, delay: 200, easing: elasticOut }}
					>
						<Crown class="w-10 h-10 text-white animate-pulse" />
					</div>

					<h1
						class="text-4xl font-bold text-white mb-3"
						transition:fly={{ y: 30, duration: 600, delay: 400, easing: quintOut }}
					>
						<span class="bg-gradient-to-r from-emerald-400 to-blue-400 bg-clip-text text-transparent">
							Welcome, Admin
						</span>
					</h1>

					<p
						class="text-white/80 text-lg mb-2"
						transition:fly={{ y: 30, duration: 600, delay: 600, easing: quintOut }}
					>
						Create your administrator account
					</p>
					<p
						class="text-emerald-400/80 text-sm flex items-center justify-center gap-2"
						transition:fly={{ y: 30, duration: 600, delay: 800, easing: quintOut }}
					>
						<Shield class="w-4 h-4" />
						First user becomes system administrator
					</p>
				{:else}
					<div
						class="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-400 to-purple-400 rounded-full mb-6 shadow-lg animate-bounce-subtle"
						transition:fly={{ y: -50, duration: 800, delay: 200, easing: quintOut }}
					>
						<Zap class="w-10 h-10 text-white animate-pulse" />
					</div>

					<h1
						class="text-4xl font-bold text-white mb-3"
						transition:fly={{ y: 30, duration: 600, delay: 400, easing: quintOut }}
					>
						<span class="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
							Welcome Back
						</span>
					</h1>

					<p
						class="text-white/70 text-lg"
						transition:fly={{ y: 30, duration: 600, delay: 600, easing: quintOut }}
					>
						Sign in to your revolutionary AI platform
					</p>
				{/if}
			</div>
			
			<!-- Conditional Form Rendering -->
			{#if showRegistrationForm}
				<!-- Registration Form -->
				<form
					on:submit|preventDefault={handleRegistration}
					class="space-y-5 relative z-10"
					transition:fly={{ y: 50, duration: 600, delay: 800, easing: quintOut }}
				>
					<!-- Full Name -->
					<div class="relative group">
						<div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
							<User class="h-5 w-5 text-white/50 group-focus-within:text-emerald-400 transition-colors duration-300" />
						</div>
						<input
							type="text"
							bind:value={registrationData.full_name}
							placeholder="Full Name (Optional)"
							class="form-input pl-12 group-focus-within:scale-105 transition-transform duration-300"
						/>
					</div>

					<!-- Username -->
					<div class="relative group">
						<div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
							<User class="h-5 w-5 text-white/50 group-focus-within:text-emerald-400 transition-colors duration-300" />
						</div>
						<input
							type="text"
							bind:value={registrationData.username}
							placeholder="Username"
							required
							class="form-input pl-12 group-focus-within:scale-105 transition-transform duration-300 {registrationErrors.username ? 'border-red-400' : ''}"
						/>
						{#if registrationErrors.username}
							<p class="mt-1 text-sm text-red-400">{registrationErrors.username}</p>
						{/if}
					</div>

					<!-- Email -->
					<div class="relative group">
						<div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
							<Mail class="h-5 w-5 text-white/50 group-focus-within:text-emerald-400 transition-colors duration-300" />
						</div>
						<input
							type="email"
							bind:value={registrationData.email}
							placeholder="Email Address"
							required
							class="form-input pl-12 group-focus-within:scale-105 transition-transform duration-300 {registrationErrors.email ? 'border-red-400' : ''}"
						/>
						{#if registrationErrors.email}
							<p class="mt-1 text-sm text-red-400">{registrationErrors.email}</p>
						{/if}
					</div>

					<!-- Password -->
					<div class="relative group">
						<div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
							<Lock class="h-5 w-5 text-white/50 group-focus-within:text-emerald-400 transition-colors duration-300" />
						</div>
						{#if showRegPassword}
							<input
								type="text"
								bind:value={registrationData.password}
								placeholder="Password"
								required
								class="form-input pl-12 pr-12 group-focus-within:scale-105 transition-transform duration-300 {registrationErrors.password ? 'border-red-400' : ''}"
							/>
						{:else}
							<input
								type="password"
								bind:value={registrationData.password}
								placeholder="Password"
								required
								class="form-input pl-12 pr-12 group-focus-within:scale-105 transition-transform duration-300 {registrationErrors.password ? 'border-red-400' : ''}"
							/>
						{/if}
						<button
							type="button"
							on:click={() => showRegPassword = !showRegPassword}
							class="absolute inset-y-0 right-0 pr-4 flex items-center text-white/50 hover:text-white/70 transition-colors duration-200"
						>
							{#if showRegPassword}
								<EyeOff class="h-5 w-5" />
							{:else}
								<Eye class="h-5 w-5" />
							{/if}
						</button>
						{#if registrationErrors.password}
							<p class="mt-1 text-sm text-red-400">{registrationErrors.password}</p>
						{/if}
					</div>

					<!-- Confirm Password -->
					<div class="relative group">
						<div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
							<Lock class="h-5 w-5 text-white/50 group-focus-within:text-emerald-400 transition-colors duration-300" />
						</div>
						{#if showRegConfirmPassword}
							<input
								type="text"
								bind:value={registrationData.confirmPassword}
								placeholder="Confirm Password"
								required
								class="form-input pl-12 pr-12 group-focus-within:scale-105 transition-transform duration-300 {registrationErrors.confirmPassword ? 'border-red-400' : ''}"
							/>
						{:else}
							<input
								type="password"
								bind:value={registrationData.confirmPassword}
								placeholder="Confirm Password"
								required
								class="form-input pl-12 pr-12 group-focus-within:scale-105 transition-transform duration-300 {registrationErrors.confirmPassword ? 'border-red-400' : ''}"
							/>
						{/if}
						<button
							type="button"
							on:click={() => showRegConfirmPassword = !showRegConfirmPassword}
							class="absolute inset-y-0 right-0 pr-4 flex items-center text-white/50 hover:text-white/70 transition-colors duration-200"
						>
							{#if showRegConfirmPassword}
								<EyeOff class="h-5 w-5" />
							{:else}
								<Eye class="h-5 w-5" />
							{/if}
						</button>
						{#if registrationErrors.confirmPassword}
							<p class="mt-1 text-sm text-red-400">{registrationErrors.confirmPassword}</p>
						{/if}
					</div>

					<!-- Register Button -->
					<button
						type="submit"
						disabled={isSubmitting || !isRegFormValid}
						class="w-full py-4 px-6 bg-gradient-to-r from-emerald-500 to-blue-500 text-white font-bold rounded-xl
							   hover:from-emerald-600 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2
							   disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 btn-hover pulse-glow
							   flex items-center justify-center space-x-2 text-lg
							   {isSubmitting ? 'animate-pulse' : ''}"
					>
						{#if isSubmitting}
							<span class="loading-dots">Creating Admin Account</span>
						{:else}
							<Crown class="w-5 h-5" />
							<span>Create Admin Account</span>
						{/if}
					</button>
				</form>
			{:else}
				<!-- Login Form -->
				<form
					on:submit|preventDefault={handleLogin}
					class="space-y-6 relative z-10"
					transition:fly={{ y: 50, duration: 600, delay: 800, easing: quintOut }}
				>
					<!-- Email/Username Input -->
					<div class="relative group">
						<div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
							<Mail class="h-5 w-5 text-white/50 group-focus-within:text-blue-400 transition-colors duration-300" />
						</div>
						<input
							type="text"
							bind:value={credentials.username_or_email}
							placeholder="Email or Username"
							required
							class="form-input pl-12 group-focus-within:scale-105 transition-transform duration-300"
						/>
					</div>
				
					<!-- Password Input -->
					<div class="relative group">
						<div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
							<Lock class="h-5 w-5 text-white/50 group-focus-within:text-blue-400 transition-colors duration-300" />
						</div>
						{#if showPassword}
							<input
								type="text"
								bind:value={credentials.password}
								placeholder="Password"
								required
								class="form-input pl-12 pr-12 group-focus-within:scale-105 transition-transform duration-300"
							/>
						{:else}
							<input
								type="password"
								bind:value={credentials.password}
								placeholder="Password"
								required
								class="form-input pl-12 pr-12 group-focus-within:scale-105 transition-transform duration-300"
							/>
						{/if}
						<button
							type="button"
							on:click={() => showPassword = !showPassword}
							class="absolute inset-y-0 right-0 pr-4 flex items-center text-white/50 hover:text-white/70 transition-colors duration-200"
						>
							{#if showPassword}
								<EyeOff class="h-5 w-5" />
							{:else}
								<Eye class="h-5 w-5" />
							{/if}
						</button>
					</div>

					<!-- Login Button -->
					<button
						type="submit"
						disabled={isSubmitting || !credentials.username_or_email || !credentials.password}
						class="w-full py-4 px-6 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-bold rounded-xl
							   hover:from-blue-600 hover:to-purple-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
							   disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 btn-hover pulse-glow
							   flex items-center justify-center space-x-2 text-lg
							   {isSubmitting ? 'animate-pulse' : ''}"
					>
						{#if isSubmitting}
							<span class="loading-dots">Signing In</span>
						{:else}
							<span>Sign In</span>
							<ArrowRight class="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
						{/if}
					</button>
				</form>

				<!-- Register Link (only show if not first-time setup) -->
				<div
					class="mt-8 text-center relative z-10"
					transition:fly={{ y: 30, duration: 600, delay: 1000, easing: quintOut }}
				>
					<p class="text-white/60 mb-4">Don't have an account?</p>
					<button
						on:click={openRegisterModal}
						class="inline-flex items-center space-x-2 text-blue-300 hover:text-blue-200 font-semibold
							   transition-all duration-300 hover:scale-105 group"
					>
						<Sparkles class="w-4 h-4 group-hover:animate-spin" />
						<span>Create Account</span>
						<ArrowRight class="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" />
					</button>
				</div>
			{/if}
			
			<!-- Error Display -->
			{#if $authStore.error}
				<div
					class="mt-6 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-200 text-sm relative z-10 animate-slide-down"
					transition:fly={{ y: -20, duration: 300 }}
				>
					<div class="flex items-center space-x-2">
						<div class="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
						<span>{$authStore.error}</span>
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>

<!-- Register Modal -->
<RegisterModal 
	bind:isOpen={showRegisterModal}
	on:close={() => showRegisterModal = false}
	on:success={() => {
		showRegisterModal = false;
		// Success is handled by the auth store redirect
	}}
/>

<style>
	:global(.group:focus-within .form-input) {
		box-shadow: 0 0 0 1px rgba(102, 126, 234, 0.5), 0 0 20px rgba(102, 126, 234, 0.3);
	}
</style>
