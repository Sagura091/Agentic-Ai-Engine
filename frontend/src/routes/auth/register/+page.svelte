<!--
Revolutionary Registration Page
Advanced user registration with validation, security features, and onboarding
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
		User, 
		UserPlus, 
		Building,
		Shield,
		Check,
		X,
		AlertCircle,
		Loader,
		Github,
		Google
	} from 'lucide-svelte';

	// Form state
	let formData = {
		name: '',
		email: '',
		password: '',
		confirmPassword: '',
		organization: '',
		role: 'developer',
		agreeToTerms: false,
		subscribeNewsletter: true
	};

	let showPassword = false;
	let showConfirmPassword = false;
	let isLoading = false;
	let errors: Record<string, string> = {};
	let step = 1; // Multi-step registration
	let passwordStrength = 0;

	// Role options
	const roleOptions = [
		{ value: 'developer', label: 'Developer', description: 'Build and deploy AI agents' },
		{ value: 'researcher', label: 'Researcher', description: 'Experiment with AI models' },
		{ value: 'business', label: 'Business User', description: 'Use pre-built solutions' },
		{ value: 'student', label: 'Student', description: 'Learn AI development' },
		{ value: 'enterprise', label: 'Enterprise', description: 'Large-scale deployments' }
	];

	// Password requirements
	const passwordRequirements = [
		{ id: 'length', label: 'At least 8 characters', test: (pwd: string) => pwd.length >= 8 },
		{ id: 'uppercase', label: 'One uppercase letter', test: (pwd: string) => /[A-Z]/.test(pwd) },
		{ id: 'lowercase', label: 'One lowercase letter', test: (pwd: string) => /[a-z]/.test(pwd) },
		{ id: 'number', label: 'One number', test: (pwd: string) => /\d/.test(pwd) },
		{ id: 'special', label: 'One special character', test: (pwd: string) => /[!@#$%^&*(),.?":{}|<>]/.test(pwd) }
	];

	onMount(() => {
		// Check if already authenticated
		if (authService.isAuthenticated()) {
			goto('/');
		}
	});

	$: {
		// Calculate password strength
		passwordStrength = passwordRequirements.filter(req => req.test(formData.password)).length;
	}

	function validateStep1(): boolean {
		errors = {};

		if (!formData.name.trim()) {
			errors.name = 'Full name is required';
		} else if (formData.name.trim().length < 2) {
			errors.name = 'Name must be at least 2 characters';
		}

		if (!formData.email) {
			errors.email = 'Email is required';
		} else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
			errors.email = 'Please enter a valid email address';
		}

		return Object.keys(errors).length === 0;
	}

	function validateStep2(): boolean {
		errors = {};

		if (!formData.password) {
			errors.password = 'Password is required';
		} else if (passwordStrength < 4) {
			errors.password = 'Password does not meet security requirements';
		}

		if (!formData.confirmPassword) {
			errors.confirmPassword = 'Please confirm your password';
		} else if (formData.password !== formData.confirmPassword) {
			errors.confirmPassword = 'Passwords do not match';
		}

		return Object.keys(errors).length === 0;
	}

	function validateStep3(): boolean {
		errors = {};

		if (!formData.agreeToTerms) {
			errors.terms = 'You must agree to the Terms of Service';
		}

		return Object.keys(errors).length === 0;
	}

	function nextStep() {
		let isValid = false;

		switch (step) {
			case 1:
				isValid = validateStep1();
				break;
			case 2:
				isValid = validateStep2();
				break;
			case 3:
				isValid = validateStep3();
				break;
		}

		if (isValid && step < 3) {
			step++;
		} else if (isValid && step === 3) {
			handleRegister();
		}
	}

	function prevStep() {
		if (step > 1) {
			step--;
			errors = {};
		}
	}

	async function handleRegister() {
		if (!validateStep3()) return;

		isLoading = true;
		errors = {};

		try {
			const result = await authService.register({
				name: formData.name.trim(),
				email: formData.email.toLowerCase().trim(),
				password: formData.password,
				organization: formData.organization.trim() || undefined,
				role: formData.role,
				subscribe_newsletter: formData.subscribeNewsletter
			});

			if (result.success) {
				if (result.requiresVerification) {
					notificationActions.add({
						type: 'success',
						title: 'Registration Successful!',
						message: 'Please check your email to verify your account.'
					});
					goto('/auth/verify-email');
				} else {
					notificationActions.add({
						type: 'success',
						title: 'Welcome to Agentic AI Platform!',
						message: 'Your account has been created successfully.'
					});
					goto('/onboarding');
				}
			} else {
				errors.general = result.error || 'Registration failed';
			}

		} catch (error) {
			console.error('Registration error:', error);
			errors.general = 'An unexpected error occurred. Please try again.';
		} finally {
			isLoading = false;
		}
	}

	async function handleSSORegister(provider: 'google' | 'github') {
		try {
			isLoading = true;
			const result = await authService.ssoLogin(provider);
			
			if (result.success) {
				// Redirect will be handled by the SSO provider
				window.location.href = result.redirectUrl;
			} else {
				errors.general = result.error || `${provider} registration failed`;
			}
		} catch (error) {
			console.error('SSO registration error:', error);
			errors.general = 'Social registration is not available';
		} finally {
			isLoading = false;
		}
	}

	function getPasswordStrengthColor(): string {
		switch (passwordStrength) {
			case 0:
			case 1: return 'bg-red-500';
			case 2: return 'bg-orange-500';
			case 3: return 'bg-yellow-500';
			case 4: return 'bg-green-500';
			case 5: return 'bg-emerald-500';
			default: return 'bg-gray-500';
		}
	}

	function getPasswordStrengthText(): string {
		switch (passwordStrength) {
			case 0:
			case 1: return 'Weak';
			case 2: return 'Fair';
			case 3: return 'Good';
			case 4: return 'Strong';
			case 5: return 'Very Strong';
			default: return '';
		}
	}
</script>

<svelte:head>
	<title>Sign Up - Agentic AI Platform</title>
	<meta name="description" content="Create your Agentic AI Platform account" />
</svelte:head>

<div class="register-form space-y-6">
	<!-- Header -->
	<div class="text-center">
		<h2 class="text-2xl font-bold text-white mb-2">Create Your Account</h2>
		<p class="text-dark-300">Join thousands of developers building revolutionary AI agents</p>
	</div>

	<!-- Progress Indicator -->
	<div class="flex items-center justify-center space-x-4 mb-8">
		{#each [1, 2, 3] as stepNum}
			<div class="flex items-center">
				<div class="w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all duration-200 {
					stepNum < step ? 'bg-primary-500 text-white' :
					stepNum === step ? 'bg-primary-500 text-white' :
					'bg-dark-600 text-dark-400'
				}">
					{stepNum < step ? '✓' : stepNum}
				</div>
				{#if stepNum < 3}
					<div class="w-12 h-0.5 mx-2 {stepNum < step ? 'bg-primary-500' : 'bg-dark-600'}"></div>
				{/if}
			</div>
		{/each}
	</div>

	<!-- General Error -->
	{#if errors.general}
		<div class="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-start space-x-3">
			<AlertCircle class="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
			<p class="text-red-400">{errors.general}</p>
		</div>
	{/if}

	<form on:submit|preventDefault={nextStep} class="space-y-6">
		<!-- Step 1: Basic Information -->
		{#if step === 1}
			<div class="space-y-4">
				<h3 class="text-lg font-semibold text-white mb-4">Basic Information</h3>

				<!-- Full Name -->
				<div>
					<label for="name" class="block text-sm font-medium text-dark-200 mb-2">
						Full Name
					</label>
					<div class="relative">
						<User class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
						<input
							id="name"
							type="text"
							bind:value={formData.name}
							disabled={isLoading}
							class="w-full pl-10 pr-4 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
							placeholder="Enter your full name"
							autocomplete="name"
						/>
					</div>
					{#if errors.name}
						<p class="mt-1 text-sm text-red-400">{errors.name}</p>
					{/if}
				</div>

				<!-- Email -->
				<div>
					<label for="email" class="block text-sm font-medium text-dark-200 mb-2">
						Email Address
					</label>
					<div class="relative">
						<Mail class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
						<input
							id="email"
							type="email"
							bind:value={formData.email}
							disabled={isLoading}
							class="w-full pl-10 pr-4 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
							placeholder="Enter your email"
							autocomplete="email"
						/>
					</div>
					{#if errors.email}
						<p class="mt-1 text-sm text-red-400">{errors.email}</p>
					{/if}
				</div>

				<!-- Organization (Optional) -->
				<div>
					<label for="organization" class="block text-sm font-medium text-dark-200 mb-2">
						Organization <span class="text-dark-400">(Optional)</span>
					</label>
					<div class="relative">
						<Building class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
						<input
							id="organization"
							type="text"
							bind:value={formData.organization}
							disabled={isLoading}
							class="w-full pl-10 pr-4 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
							placeholder="Company or organization name"
							autocomplete="organization"
						/>
					</div>
				</div>
			</div>
		{/if}

		<!-- Step 2: Security -->
		{#if step === 2}
			<div class="space-y-4">
				<h3 class="text-lg font-semibold text-white mb-4">Security & Role</h3>

				<!-- Password -->
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
								bind:value={formData.password}
								disabled={isLoading}
								class="w-full pl-10 pr-12 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
								placeholder="Create a strong password"
								autocomplete="new-password"
							/>
						{:else}
							<input
								id="password"
								type="password"
								bind:value={formData.password}
								disabled={isLoading}
								class="w-full pl-10 pr-12 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
								placeholder="Create a strong password"
								autocomplete="new-password"
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

					<!-- Password Strength Indicator -->
					{#if formData.password}
						<div class="mt-2">
							<div class="flex items-center justify-between mb-1">
								<span class="text-xs text-dark-300">Password Strength</span>
								<span class="text-xs font-medium {
									passwordStrength >= 4 ? 'text-green-400' :
									passwordStrength >= 3 ? 'text-yellow-400' :
									'text-red-400'
								}">{getPasswordStrengthText()}</span>
							</div>
							<div class="w-full bg-dark-600 rounded-full h-2">
								<div 
									class="h-2 rounded-full transition-all duration-300 {getPasswordStrengthColor()}"
									style="width: {(passwordStrength / 5) * 100}%"
								></div>
							</div>
						</div>

						<!-- Password Requirements -->
						<div class="mt-3 space-y-1">
							{#each passwordRequirements as req}
								<div class="flex items-center space-x-2 text-xs">
									{#if req.test(formData.password)}
										<Check class="w-3 h-3 text-green-400" />
										<span class="text-green-400">{req.label}</span>
									{:else}
										<X class="w-3 h-3 text-dark-400" />
										<span class="text-dark-400">{req.label}</span>
									{/if}
								</div>
							{/each}
						</div>
					{/if}

					{#if errors.password}
						<p class="mt-1 text-sm text-red-400">{errors.password}</p>
					{/if}
				</div>

				<!-- Confirm Password -->
				<div>
					<label for="confirmPassword" class="block text-sm font-medium text-dark-200 mb-2">
						Confirm Password
					</label>
					<div class="relative">
						<Lock class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
						{#if showConfirmPassword}
							<input
								id="confirmPassword"
								type="text"
								bind:value={formData.confirmPassword}
								disabled={isLoading}
								class="w-full pl-10 pr-12 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
								placeholder="Confirm your password"
								autocomplete="new-password"
							/>
						{:else}
							<input
								id="confirmPassword"
								type="password"
								bind:value={formData.confirmPassword}
								disabled={isLoading}
								class="w-full pl-10 pr-12 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
								placeholder="Confirm your password"
								autocomplete="new-password"
							/>
						{/if}
						<button
							type="button"
							on:click={() => showConfirmPassword = !showConfirmPassword}
							class="absolute right-3 top-1/2 transform -translate-y-1/2 text-dark-400 hover:text-white transition-colors"
						>
							{#if showConfirmPassword}
								<EyeOff class="w-5 h-5" />
							{:else}
								<Eye class="w-5 h-5" />
							{/if}
						</button>
					</div>
					{#if errors.confirmPassword}
						<p class="mt-1 text-sm text-red-400">{errors.confirmPassword}</p>
					{/if}
				</div>

				<!-- Role Selection -->
				<div>
					<label class="block text-sm font-medium text-dark-200 mb-3">
						What best describes your role?
					</label>
					<div class="grid grid-cols-1 gap-3">
						{#each roleOptions as role}
							<label class="relative flex items-start p-3 bg-dark-700 border border-dark-600 rounded-lg cursor-pointer hover:bg-dark-600 transition-colors">
								<input
									type="radio"
									bind:group={formData.role}
									value={role.value}
									class="mt-1 w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 focus:ring-primary-500 focus:ring-2"
								/>
								<div class="ml-3">
									<div class="text-sm font-medium text-white">{role.label}</div>
									<div class="text-xs text-dark-300">{role.description}</div>
								</div>
							</label>
						{/each}
					</div>
				</div>
			</div>
		{/if}

		<!-- Step 3: Terms & Preferences -->
		{#if step === 3}
			<div class="space-y-4">
				<h3 class="text-lg font-semibold text-white mb-4">Terms & Preferences</h3>

				<!-- Terms Agreement -->
				<div>
					<label class="flex items-start space-x-3 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={formData.agreeToTerms}
							disabled={isLoading}
							class="mt-1 w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500 focus:ring-2"
						/>
						<div class="text-sm text-dark-300">
							I agree to the 
							<a href="/terms" target="_blank" class="text-primary-400 hover:text-primary-300 underline">Terms of Service</a>
							and 
							<a href="/privacy" target="_blank" class="text-primary-400 hover:text-primary-300 underline">Privacy Policy</a>
						</div>
					</label>
					{#if errors.terms}
						<p class="mt-1 text-sm text-red-400">{errors.terms}</p>
					{/if}
				</div>

				<!-- Newsletter Subscription -->
				<div>
					<label class="flex items-start space-x-3 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={formData.subscribeNewsletter}
							disabled={isLoading}
							class="mt-1 w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500 focus:ring-2"
						/>
						<div class="text-sm text-dark-300">
							Subscribe to our newsletter for updates, tips, and new features
						</div>
					</label>
				</div>

				<!-- Account Summary -->
				<div class="bg-dark-700/50 border border-dark-600 rounded-lg p-4 space-y-2">
					<h4 class="text-sm font-medium text-white">Account Summary</h4>
					<div class="text-sm text-dark-300 space-y-1">
						<div><span class="text-dark-400">Name:</span> {formData.name}</div>
						<div><span class="text-dark-400">Email:</span> {formData.email}</div>
						<div><span class="text-dark-400">Role:</span> {roleOptions.find(r => r.value === formData.role)?.label}</div>
						{#if formData.organization}
							<div><span class="text-dark-400">Organization:</span> {formData.organization}</div>
						{/if}
					</div>
				</div>
			</div>
		{/if}

		<!-- Navigation Buttons -->
		<div class="flex items-center justify-between pt-4">
			{#if step > 1}
				<button
					type="button"
					on:click={prevStep}
					disabled={isLoading}
					class="px-6 py-2 text-dark-300 hover:text-white transition-colors"
				>
					← Back
				</button>
			{:else}
				<div></div>
			{/if}

			<button
				type="submit"
				disabled={isLoading}
				class="bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-700 hover:to-primary-600 disabled:from-dark-600 disabled:to-dark-600 text-white font-medium py-3 px-6 rounded-lg transition-all duration-200 flex items-center space-x-2"
			>
				{#if isLoading}
					<Loader class="w-5 h-5 animate-spin" />
					<span>Creating Account...</span>
				{:else if step < 3}
					<span>Continue</span>
					<span>→</span>
				{:else}
					<UserPlus class="w-5 h-5" />
					<span>Create Account</span>
				{/if}
			</button>
		</div>
	</form>

	<!-- Alternative Registration (only on step 1) -->
	{#if step === 1}
		<div class="space-y-4">
			<!-- Divider -->
			<div class="relative">
				<div class="absolute inset-0 flex items-center">
					<div class="w-full border-t border-dark-600"></div>
				</div>
				<div class="relative flex justify-center text-sm">
					<span class="px-2 bg-dark-800 text-dark-400">Or sign up with</span>
				</div>
			</div>

			<!-- SSO Buttons -->
			<div class="grid grid-cols-2 gap-3">
				<button
					type="button"
					on:click={() => handleSSORegister('google')}
					disabled={isLoading}
					class="bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
				>
					<Google class="w-5 h-5" />
					<span>Google</span>
				</button>
				<button
					type="button"
					on:click={() => handleSSORegister('github')}
					disabled={isLoading}
					class="bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
				>
					<Github class="w-5 h-5" />
					<span>GitHub</span>
				</button>
			</div>
		</div>
	{/if}

	<!-- Sign In Link -->
	<div class="text-center">
		<p class="text-dark-400">
			Already have an account?
			<a
				href="/auth/login"
				class="text-primary-400 hover:text-primary-300 font-medium transition-colors"
			>
				Sign in
			</a>
		</p>
	</div>
</div>

<style lang="postcss">
	.register-form {
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
