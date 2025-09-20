<!--
Revolutionary Login Page
Advanced authentication with biometric support, SSO, and security features
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { authService } from '$lib/services/authService';
	import { notificationActions } from '$lib/stores';
	import { 
		Eye, 
		EyeOff, 
		Mail, 
		Lock, 
		LogIn, 
		Fingerprint,
		Smartphone,
		Github,
		Google,
		Shield,
		AlertCircle,
		Loader
	} from 'lucide-svelte';

	// Form state
	let email = '';
	let password = '';
	let rememberMe = false;
	let showPassword = false;
	let isLoading = false;
	let errors: Record<string, string> = {};

	// Advanced features
	let biometricSupported = false;
	let twoFactorRequired = false;
	let twoFactorCode = '';
	let securityKeySupported = false;

	// Rate limiting
	let loginAttempts = 0;
	let isRateLimited = false;
	let rateLimitReset = 0;

	onMount(async () => {
		// Check biometric support
		if ('credentials' in navigator && 'create' in navigator.credentials) {
			biometricSupported = true;
		}

		// Check WebAuthn support
		if ('credentials' in navigator && 'get' in navigator.credentials) {
			securityKeySupported = true;
		}

		// Load saved email
		const savedEmail = localStorage.getItem('saved_email');
		if (savedEmail) {
			email = savedEmail;
			rememberMe = true;
		}

		// Check rate limiting
		checkRateLimit();
	});

	function checkRateLimit() {
		const attempts = localStorage.getItem('login_attempts');
		const lastAttempt = localStorage.getItem('last_login_attempt');
		
		if (attempts && lastAttempt) {
			loginAttempts = parseInt(attempts);
			const timeSinceLastAttempt = Date.now() - parseInt(lastAttempt);
			
			if (loginAttempts >= 5 && timeSinceLastAttempt < 15 * 60 * 1000) {
				isRateLimited = true;
				rateLimitReset = parseInt(lastAttempt) + 15 * 60 * 1000;
				startRateLimitTimer();
			}
		}
	}

	function startRateLimitTimer() {
		const timer = setInterval(() => {
			if (Date.now() >= rateLimitReset) {
				isRateLimited = false;
				loginAttempts = 0;
				localStorage.removeItem('login_attempts');
				localStorage.removeItem('last_login_attempt');
				clearInterval(timer);
			}
		}, 1000);
	}

	function validateForm(): boolean {
		errors = {};

		if (!email) {
			errors.email = 'Email is required';
		} else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
			errors.email = 'Please enter a valid email address';
		}

		if (!password) {
			errors.password = 'Password is required';
		} else if (password.length < 8) {
			errors.password = 'Password must be at least 8 characters';
		}

		if (twoFactorRequired && !twoFactorCode) {
			errors.twoFactor = '2FA code is required';
		}

		return Object.keys(errors).length === 0;
	}

	async function handleLogin() {
		if (!validateForm() || isRateLimited) return;

		isLoading = true;
		errors = {};

		try {
			const result = await authService.login({
				email,
				password,
				rememberMe,
				twoFactorCode: twoFactorRequired ? twoFactorCode : undefined
			});

			if (result.success) {
				// Save email if remember me is checked
				if (rememberMe) {
					localStorage.setItem('saved_email', email);
				} else {
					localStorage.removeItem('saved_email');
				}

				// Clear rate limiting
				localStorage.removeItem('login_attempts');
				localStorage.removeItem('last_login_attempt');

				notificationActions.add({
					type: 'success',
					title: 'Welcome back!',
					message: 'You have been successfully logged in.'
				});

				// Redirect to dashboard or intended page
				const redirectTo = new URLSearchParams(window.location.search).get('redirect') || '/';
				goto(redirectTo);

			} else if (result.requiresTwoFactor) {
				twoFactorRequired = true;
				notificationActions.add({
					type: 'info',
					title: '2FA Required',
					message: 'Please enter your two-factor authentication code.'
				});

			} else {
				// Handle login failure
				loginAttempts++;
				localStorage.setItem('login_attempts', loginAttempts.toString());
				localStorage.setItem('last_login_attempt', Date.now().toString());

				if (loginAttempts >= 5) {
					isRateLimited = true;
					rateLimitReset = Date.now() + 15 * 60 * 1000;
					startRateLimitTimer();
				}

				errors.general = result.error || 'Invalid email or password';
			}

		} catch (error) {
			console.error('Login error:', error);
			errors.general = 'An unexpected error occurred. Please try again.';
		} finally {
			isLoading = false;
		}
	}

	async function handleBiometricLogin() {
		if (!biometricSupported) return;

		try {
			isLoading = true;
			const result = await authService.biometricLogin();
			
			if (result.success) {
				notificationActions.add({
					type: 'success',
					title: 'Biometric Login Successful',
					message: 'Welcome back!'
				});
				goto('/');
			} else {
				errors.general = result.error || 'Biometric authentication failed';
			}
		} catch (error) {
			console.error('Biometric login error:', error);
			errors.general = 'Biometric authentication is not available';
		} finally {
			isLoading = false;
		}
	}

	async function handleSSOLogin(provider: 'google' | 'github' | 'microsoft') {
		try {
			isLoading = true;
			const result = await authService.ssoLogin(provider);
			
			if (result.success) {
				// Redirect will be handled by the SSO provider
				window.location.href = result.redirectUrl;
			} else {
				errors.general = result.error || `${provider} login failed`;
			}
		} catch (error) {
			console.error('SSO login error:', error);
			errors.general = 'Social login is not available';
		} finally {
			isLoading = false;
		}
	}

	function getRateLimitTimeRemaining(): string {
		const remaining = Math.ceil((rateLimitReset - Date.now()) / 1000 / 60);
		return `${remaining} minute${remaining !== 1 ? 's' : ''}`;
	}
</script>

<svelte:head>
	<title>Login - Agentic AI Platform</title>
	<meta name="description" content="Sign in to your Agentic AI Platform account" />
</svelte:head>

<div class="login-form space-y-6">
	<!-- Header -->
	<div class="text-center">
		<h2 class="text-2xl font-bold text-white mb-2">Welcome Back</h2>
		<p class="text-dark-300">Sign in to continue building revolutionary AI agents</p>
	</div>

	<!-- Rate Limit Warning -->
	{#if isRateLimited}
		<div class="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-start space-x-3">
			<AlertCircle class="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
			<div>
				<p class="text-red-400 font-medium">Too many login attempts</p>
				<p class="text-red-300 text-sm">Please wait {getRateLimitTimeRemaining()} before trying again.</p>
			</div>
		</div>
	{/if}

	<!-- General Error -->
	{#if errors.general}
		<div class="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-start space-x-3">
			<AlertCircle class="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
			<p class="text-red-400">{errors.general}</p>
		</div>
	{/if}

	<form on:submit|preventDefault={handleLogin} class="space-y-4">
		<!-- Email Field -->
		<div>
			<label for="email" class="block text-sm font-medium text-dark-200 mb-2">
				Email Address
			</label>
			<div class="relative">
				<Mail class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
				<input
					id="email"
					type="email"
					bind:value={email}
					disabled={isLoading || isRateLimited}
					class="w-full pl-10 pr-4 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
					placeholder="Enter your email"
					autocomplete="email"
				/>
			</div>
			{#if errors.email}
				<p class="mt-1 text-sm text-red-400">{errors.email}</p>
			{/if}
		</div>

		<!-- Password Field -->
		<div>
			<label for="password" class="block text-sm font-medium text-dark-200 mb-2">
				Password
			</label>
			<div class="relative">
				<Lock class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
				{#if showPassword}
					<input
						id="password"
						type="text"
						bind:value={password}
						disabled={isLoading || isRateLimited}
						class="w-full pl-10 pr-12 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
						placeholder="Enter your password"
						autocomplete="current-password"
					/>
				{:else}
					<input
						id="password"
						type="password"
						bind:value={password}
						disabled={isLoading || isRateLimited}
						class="w-full pl-10 pr-12 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
						placeholder="Enter your password"
						autocomplete="current-password"
					/>
				{/if}
				<button
					type="button"
					on:click={() => showPassword = !showPassword}
					class="absolute right-3 top-1/2 transform -translate-y-1/2 text-dark-400 hover:text-white transition-colors"
				>
					{#if showPassword}
						<EyeOff class="w-5 h-5" />
					{:else}
						<Eye class="w-5 h-5" />
					{/if}
				</button>
			</div>
			{#if errors.password}
				<p class="mt-1 text-sm text-red-400">{errors.password}</p>
			{/if}
		</div>

		<!-- 2FA Code (if required) -->
		{#if twoFactorRequired}
			<div>
				<label for="twoFactor" class="block text-sm font-medium text-dark-200 mb-2">
					Two-Factor Authentication Code
				</label>
				<div class="relative">
					<Shield class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
					<input
						id="twoFactor"
						type="text"
						bind:value={twoFactorCode}
						disabled={isLoading}
						class="w-full pl-10 pr-4 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
						placeholder="Enter 6-digit code"
						maxlength="6"
						autocomplete="one-time-code"
					/>
				</div>
				{#if errors.twoFactor}
					<p class="mt-1 text-sm text-red-400">{errors.twoFactor}</p>
				{/if}
			</div>
		{/if}

		<!-- Remember Me & Forgot Password -->
		<div class="flex items-center justify-between">
			<label class="flex items-center space-x-2 cursor-pointer">
				<input
					type="checkbox"
					bind:checked={rememberMe}
					disabled={isLoading || isRateLimited}
					class="w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500 focus:ring-2"
				/>
				<span class="text-sm text-dark-300">Remember me</span>
			</label>
			<a
				href="/auth/forgot-password"
				class="text-sm text-primary-400 hover:text-primary-300 transition-colors"
			>
				Forgot password?
			</a>
		</div>

		<!-- Login Button -->
		<button
			type="submit"
			disabled={isLoading || isRateLimited}
			class="w-full bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-700 hover:to-primary-600 disabled:from-dark-600 disabled:to-dark-600 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
		>
			{#if isLoading}
				<Loader class="w-5 h-5 animate-spin" />
				<span>Signing in...</span>
			{:else}
				<LogIn class="w-5 h-5" />
				<span>Sign In</span>
			{/if}
		</button>
	</form>

	<!-- Alternative Login Methods -->
	<div class="space-y-4">
		<!-- Divider -->
		<div class="relative">
			<div class="absolute inset-0 flex items-center">
				<div class="w-full border-t border-dark-600"></div>
			</div>
			<div class="relative flex justify-center text-sm">
				<span class="px-2 bg-dark-800 text-dark-400">Or continue with</span>
			</div>
		</div>

		<!-- Biometric Login -->
		{#if biometricSupported}
			<button
				type="button"
				on:click={handleBiometricLogin}
				disabled={isLoading || isRateLimited}
				class="w-full bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
			>
				<Fingerprint class="w-5 h-5" />
				<span>Use Biometric</span>
			</button>
		{/if}

		<!-- SSO Buttons -->
		<div class="grid grid-cols-2 gap-3">
			<button
				type="button"
				on:click={() => handleSSOLogin('google')}
				disabled={isLoading || isRateLimited}
				class="bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
			>
				<Google class="w-5 h-5" />
				<span>Google</span>
			</button>
			<button
				type="button"
				on:click={() => handleSSOLogin('github')}
				disabled={isLoading || isRateLimited}
				class="bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
			>
				<Github class="w-5 h-5" />
				<span>GitHub</span>
			</button>
		</div>
	</div>

	<!-- Sign Up Link -->
	<div class="text-center">
		<p class="text-dark-400">
			Don't have an account?
			<a
				href="/auth/register"
				class="text-primary-400 hover:text-primary-300 font-medium transition-colors"
			>
				Sign up for free
			</a>
		</p>
	</div>
</div>

<style lang="postcss">
	.login-form {
		animation: slideUp 0.6s ease-out;
	}

	@keyframes slideUp {
		from {
			opacity: 0;
			transform: translateY(30px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
</style>
