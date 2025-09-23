<script lang="ts">
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { fly, scale } from 'svelte/transition';
	import { 
		User, Mail, Calendar, Shield, Activity, 
		Edit3, Camera, Save, X, Check, AlertCircle
	} from 'lucide-svelte';

	let loading = false;
	let editing = false;
	let saveMessage = '';
	let avatarInput: HTMLInputElement;

	// Profile data
	let profile = {
		id: '',
		username: '',
		email: '',
		name: '',
		bio: '',
		avatar_url: '',
		user_group: '',
		is_active: false,
		created_at: '',
		last_login: '',
		stats: {
			total_logins: 0,
			agents_created: 0,
			workflows_created: 0,
			last_activity: ''
		}
	};

	// Edit form data
	let editForm = {
		name: '',
		bio: '',
		avatar_url: ''
	};

	onMount(() => {
		loadProfile();
	});

	async function loadProfile() {
		loading = true;
		try {
			const response = await fetch('/api/user/profile/', {
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				}
			});

			if (response.ok) {
				const data = await response.json();
				if (data.success) {
					profile = data.data;
					// Initialize edit form with current data
					editForm = {
						name: profile.name || '',
						bio: profile.bio || '',
						avatar_url: profile.avatar_url || ''
					};
				}
			}
		} catch (error) {
			console.error('Failed to load profile:', error);
		} finally {
			loading = false;
		}
	}

	async function updateProfile() {
		loading = true;
		try {
			const response = await fetch('/api/user/profile/', {
				method: 'PUT',
				headers: {
					'Authorization': `Bearer ${$authStore.token}`,
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(editForm)
			});

			const data = await response.json();
			if (data.success) {
				saveMessage = 'Profile updated successfully';
				// Update profile data
				profile.name = editForm.name;
				profile.bio = editForm.bio;
				profile.avatar_url = editForm.avatar_url;
				editing = false;
			} else {
				saveMessage = data.message || 'Failed to update profile';
			}
		} catch (error) {
			saveMessage = 'Failed to update profile';
		} finally {
			loading = false;
		}
	}

	async function uploadAvatar(event: Event) {
		const target = event.target as HTMLInputElement;
		const file = target.files?.[0];
		
		if (!file) return;

		// Validate file type and size
		if (!file.type.startsWith('image/')) {
			saveMessage = 'Please select an image file';
			return;
		}

		if (file.size > 5 * 1024 * 1024) {
			saveMessage = 'File size must be less than 5MB';
			return;
		}

		loading = true;
		try {
			const formData = new FormData();
			formData.append('file', file);

			const response = await fetch('/api/user/profile/avatar', {
				method: 'POST',
				headers: {
					'Authorization': `Bearer ${$authStore.token}`
				},
				body: formData
			});

			const data = await response.json();
			if (data.success) {
				editForm.avatar_url = data.data.avatar_url;
				saveMessage = 'Avatar uploaded successfully';
			} else {
				saveMessage = data.message || 'Failed to upload avatar';
			}
		} catch (error) {
			saveMessage = 'Failed to upload avatar';
		} finally {
			loading = false;
		}
	}

	function cancelEdit() {
		editing = false;
		// Reset form to current profile data
		editForm = {
			name: profile.name || '',
			bio: profile.bio || '',
			avatar_url: profile.avatar_url || ''
		};
	}

	function formatDate(dateString: string) {
		if (!dateString) return 'Never';
		return new Date(dateString).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'long',
			day: 'numeric'
		});
	}

	function getRoleColor(role: string) {
		switch (role) {
			case 'admin': return 'text-red-400 bg-red-500/20';
			case 'moderator': return 'text-yellow-400 bg-yellow-500/20';
			default: return 'text-blue-400 bg-blue-500/20';
		}
	}

	// Clear save message after 3 seconds
	$: if (saveMessage) {
		setTimeout(() => {
			saveMessage = '';
		}, 3000);
	}
</script>

<svelte:head>
	<title>Profile - Agentic AI Platform</title>
</svelte:head>

<div class="min-h-screen p-6">
	<div class="max-w-4xl mx-auto">
		<!-- Header -->
		<div class="mb-8">
			<h1 class="text-3xl font-bold text-white mb-2">User Profile</h1>
			<p class="text-white/60">Manage your public identity and profile information</p>
		</div>

		{#if loading && !profile.id}
			<!-- Loading State -->
			<div class="glass rounded-2xl p-8 flex items-center justify-center">
				<div class="w-8 h-8 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
			</div>
		{:else}
			<!-- Profile Content -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Profile Card -->
				<div class="lg:col-span-1">
					<div class="glass rounded-2xl p-6 sticky top-6">
						<!-- Avatar Section -->
						<div class="text-center mb-6">
							<div class="relative inline-block">
								{#if editForm.avatar_url || profile.avatar_url}
									<img 
										src={editForm.avatar_url || profile.avatar_url} 
										alt="Profile Avatar"
										class="w-24 h-24 rounded-full object-cover border-4 border-white/20"
									/>
								{:else}
									<div class="w-24 h-24 rounded-full bg-gradient-to-br from-primary-400 to-accent-400 flex items-center justify-center border-4 border-white/20">
										<User class="w-12 h-12 text-white" />
									</div>
								{/if}
								
								{#if editing}
									<button
										on:click={() => avatarInput.click()}
										class="absolute -bottom-2 -right-2 w-8 h-8 bg-primary-500 hover:bg-primary-600 rounded-full flex items-center justify-center transition-colors"
									>
										<Camera class="w-4 h-4 text-white" />
									</button>
									<input
										bind:this={avatarInput}
										type="file"
										accept="image/*"
										on:change={uploadAvatar}
										class="hidden"
									/>
								{/if}
							</div>
							
							<h2 class="text-xl font-bold text-white mt-4">
								{profile.name || profile.username}
							</h2>
							<p class="text-white/60">@{profile.username}</p>
							
							<!-- Role Badge -->
							<div class="mt-3">
								<span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium capitalize {getRoleColor(profile.user_group)}">
									<Shield class="w-4 h-4 mr-1" />
									{profile.user_group}
								</span>
							</div>
						</div>

						<!-- Quick Stats -->
						<div class="space-y-3">
							<div class="flex items-center space-x-3 text-white/80">
								<Mail class="w-4 h-4 text-white/60" />
								<span class="text-sm">{profile.email}</span>
							</div>
							
							<div class="flex items-center space-x-3 text-white/80">
								<Calendar class="w-4 h-4 text-white/60" />
								<span class="text-sm">Joined {formatDate(profile.created_at)}</span>
							</div>
							
							<div class="flex items-center space-x-3 text-white/80">
								<Activity class="w-4 h-4 text-white/60" />
								<span class="text-sm">Last login {formatDate(profile.last_login)}</span>
							</div>
						</div>

						<!-- Edit Button -->
						{#if !editing}
							<button
								on:click={() => editing = true}
								class="w-full mt-6 btn-primary flex items-center justify-center space-x-2"
							>
								<Edit3 class="w-4 h-4" />
								<span>Edit Profile</span>
							</button>
						{:else}
							<div class="flex space-x-2 mt-6">
								<button
									on:click={updateProfile}
									disabled={loading}
									class="flex-1 btn-primary flex items-center justify-center space-x-2"
								>
									<Save class="w-4 h-4" />
									<span>Save</span>
								</button>
								<button
									on:click={cancelEdit}
									class="flex-1 btn-secondary flex items-center justify-center space-x-2"
								>
									<X class="w-4 h-4" />
									<span>Cancel</span>
								</button>
							</div>
						{/if}
					</div>
				</div>

				<!-- Main Content -->
				<div class="lg:col-span-2 space-y-6">
					<!-- Bio Section -->
					<div class="glass rounded-2xl p-6">
						<h3 class="text-xl font-semibold text-white mb-4">About</h3>
						
						{#if editing}
							<div class="space-y-4">
								<div>
									<label for="name-input" class="block text-white font-medium mb-2">Display Name</label>
									<input
										id="name-input"
										type="text"
										bind:value={editForm.name}
										placeholder="Your display name"
										class="form-input w-full"
									/>
								</div>
								
								<div>
									<label for="bio-input" class="block text-white font-medium mb-2">Bio</label>
									<textarea
										id="bio-input"
										bind:value={editForm.bio}
										placeholder="Tell us about yourself..."
										rows="4"
										class="form-input w-full resize-none"
									></textarea>
								</div>
							</div>
						{:else}
							<div class="space-y-4">
								<div>
									<h4 class="text-white font-medium mb-2">Display Name</h4>
									<p class="text-white/80">{profile.name || 'Not set'}</p>
								</div>
								
								<div>
									<h4 class="text-white font-medium mb-2">Bio</h4>
									<p class="text-white/80">{profile.bio || 'No bio available'}</p>
								</div>
							</div>
						{/if}
					</div>

					<!-- Activity Stats -->
					<div class="glass rounded-2xl p-6">
						<h3 class="text-xl font-semibold text-white mb-4">Activity Statistics</h3>
						
						<div class="grid grid-cols-2 md:grid-cols-3 gap-4">
							<div class="text-center">
								<div class="text-2xl font-bold text-primary-400">{profile.stats.total_logins}</div>
								<div class="text-white/60 text-sm">Total Logins</div>
							</div>
							
							<div class="text-center">
								<div class="text-2xl font-bold text-accent-400">{profile.stats.agents_created}</div>
								<div class="text-white/60 text-sm">Agents Created</div>
							</div>
							
							<div class="text-center">
								<div class="text-2xl font-bold text-secondary-400">{profile.stats.workflows_created}</div>
								<div class="text-white/60 text-sm">Workflows Created</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		{/if}

		<!-- Save Message -->
		{#if saveMessage}
			<div 
				class="fixed bottom-6 right-6 glass rounded-xl p-4 flex items-center space-x-3"
				transition:fly={{ y: 50, duration: 300 }}
			>
				{#if saveMessage.includes('successfully')}
					<Check class="w-5 h-5 text-green-400" />
					<span class="text-green-400">{saveMessage}</span>
				{:else}
					<AlertCircle class="w-5 h-5 text-red-400" />
					<span class="text-red-400">{saveMessage}</span>
				{/if}
			</div>
		{/if}
	</div>
</div>
