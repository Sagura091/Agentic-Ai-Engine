<!--
Revolutionary Collaboration Interface
Real-time collaborative editing, team management, and communication
-->

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { 
		collaborationSessions,
		collaborationLoading,
		collaborationError,
		currentUser,
		notificationActions 
	} from '$lib/stores';
	import { collaborationService } from '$lib/services/collaborationService';
	import { websocketService } from '$lib/services/websocket';
	import { 
		Users, 
		Video, 
		MessageSquare, 
		Share, 
		Settings, 
		Plus,
		RefreshCw,
		Play,
		Pause,
		Eye,
		Edit,
		Lock,
		Unlock,
		Crown,
		UserCheck,
		UserX,
		Clock,
		Activity,
		GitBranch,
		FileText,
		Zap,
		Heart,
		Star,
		AlertCircle,
		CheckCircle,
		Loader,
		Camera,
		Mic,
		MicOff,
		VideoOff,
		Phone,
		PhoneOff,
		Screen,
		MoreVertical
	} from 'lucide-svelte';
	import type { CollaborationSession, TeamMember, CollaborationActivity } from '$lib/types';
	import CollaborativeEditor from '$lib/components/collaboration/CollaborativeEditor.svelte';
	import TeamManagement from '$lib/components/collaboration/TeamManagement.svelte';
	import ActivityFeed from '$lib/components/collaboration/ActivityFeed.svelte';
	import VideoChat from '$lib/components/collaboration/VideoChat.svelte';
	import CommentSystem from '$lib/components/collaboration/CommentSystem.svelte';

	// State
	let activeTab: 'sessions' | 'teams' | 'activity' | 'settings' = 'sessions';
	let selectedSession: CollaborationSession | null = null;
	let showEditor = false;
	let showTeamManagement = false;
	let showVideoChat = false;
	let searchQuery = '';
	let filterStatus = 'all';
	let sortBy = 'updated';

	// Real-time collaboration state
	let activeSessions: Map<string, CollaborationSession> = new Map();
	let onlineUsers: Map<string, any> = new Map();
	let realtimeActivities: CollaborationActivity[] = [];

	// Video/Audio state
	let isVideoEnabled = false;
	let isAudioEnabled = false;
	let isScreenSharing = false;
	let participants: any[] = [];

	// Real-time updates
	let unsubscribeWebSocket: (() => void) | null = null;

	onMount(async () => {
		await loadData();
		setupRealTimeUpdates();
	});

	onDestroy(() => {
		if (unsubscribeWebSocket) {
			unsubscribeWebSocket();
		}
		// Leave any active sessions
		if (selectedSession) {
			collaborationService.leaveSession(selectedSession.id);
		}
	});

	async function loadData() {
		collaborationLoading.set(true);
		collaborationError.set(null);

		try {
			// Load collaboration sessions
			const sessionsResponse = await collaborationService.getSessions();
			if (sessionsResponse.success && sessionsResponse.data) {
				collaborationSessions.set(sessionsResponse.data);
			}

			// Load recent activities
			const activitiesResponse = await collaborationService.getRecentActivities();
			if (activitiesResponse.success && activitiesResponse.data) {
				realtimeActivities = activitiesResponse.data;
			}

		} catch (error) {
			console.error('Failed to load collaboration data:', error);
			collaborationError.set(error instanceof Error ? error.message : 'Unknown error');
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Collaboration Data',
				message: 'Unable to load collaboration sessions. Please try again.'
			});
		} finally {
			collaborationLoading.set(false);
		}
	}

	function setupRealTimeUpdates() {
		// Subscribe to session updates
		unsubscribeWebSocket = websocketService.on('collaboration_session_update' as any, (data) => {
			updateSession(data.session_id, data.updates);
		});

		// Subscribe to user presence updates
		websocketService.on('user_presence_update' as any, (data) => {
			updateUserPresence(data.user_id, data.status, data.metadata);
		});

		// Subscribe to real-time activities
		websocketService.on('collaboration_activity' as any, (data) => {
			addRealtimeActivity(data);
		});

		// Subscribe to collaborative editing events
		websocketService.on('collaborative_edit' as any, (data) => {
			handleCollaborativeEdit(data);
		});

		// Subscribe to video chat events
		websocketService.on('video_chat_event' as any, (data) => {
			handleVideoChatEvent(data);
		});
	}

	function updateSession(sessionId: string, updates: any) {
		collaborationSessions.update(sessions => 
			sessions.map(session => 
				session.id === sessionId 
					? { ...session, ...updates }
					: session
			)
		);

		activeSessions.set(sessionId, { ...activeSessions.get(sessionId), ...updates });
	}

	function updateUserPresence(userId: string, status: string, metadata: any) {
		onlineUsers.set(userId, { userId, status, ...metadata });
	}

	function addRealtimeActivity(activity: CollaborationActivity) {
		realtimeActivities = [activity, ...realtimeActivities.slice(0, 49)]; // Keep last 50
	}

	function handleCollaborativeEdit(data: any) {
		// Handle real-time collaborative editing events
		if (selectedSession && data.session_id === selectedSession.id) {
			// Forward to collaborative editor component
		}
	}

	function handleVideoChatEvent(data: any) {
		// Handle video chat events (join, leave, mute, etc.)
		switch (data.event) {
			case 'user_joined':
				participants = [...participants, data.user];
				break;
			case 'user_left':
				participants = participants.filter(p => p.id !== data.user.id);
				break;
			case 'user_muted':
				participants = participants.map(p => 
					p.id === data.user.id ? { ...p, audioEnabled: false } : p
				);
				break;
			// ... handle other events
		}
	}

	async function handleCreateSession() {
		const name = prompt('Enter session name:');
		if (!name) return;

		try {
			const result = await collaborationService.createSession({
				name: name.trim(),
				type: 'agent_editing',
				isPublic: false,
				maxParticipants: 10
			});

			if (result.success) {
				notificationActions.add({
					type: 'success',
					title: 'Session Created',
					message: `Collaboration session "${name}" has been created.`
				});
				await loadData();
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Create session error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Creation Failed',
				message: `Failed to create session: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	async function handleJoinSession(session: CollaborationSession) {
		try {
			const result = await collaborationService.joinSession(session.id);
			if (result.success) {
				selectedSession = session;
				showEditor = true;
				
				notificationActions.add({
					type: 'success',
					title: 'Joined Session',
					message: `You have joined "${session.name}".`
				});
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Join session error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Join Failed',
				message: `Failed to join session: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	async function handleLeaveSession(session: CollaborationSession) {
		try {
			const result = await collaborationService.leaveSession(session.id);
			if (result.success) {
				if (selectedSession?.id === session.id) {
					selectedSession = null;
					showEditor = false;
				}
				
				notificationActions.add({
					type: 'info',
					title: 'Left Session',
					message: `You have left "${session.name}".`
				});
			} else {
				throw new Error(result.error);
			}
		} catch (error) {
			console.error('Leave session error:', error);
			notificationActions.add({
				type: 'error',
				title: 'Leave Failed',
				message: `Failed to leave session: ${error instanceof Error ? error.message : 'Unknown error'}`
			});
		}
	}

	async function toggleVideoChat() {
		if (!selectedSession) return;

		if (showVideoChat) {
			// Leave video chat
			await collaborationService.leaveVideoChat(selectedSession.id);
			showVideoChat = false;
			isVideoEnabled = false;
			isAudioEnabled = false;
		} else {
			// Join video chat
			const result = await collaborationService.joinVideoChat(selectedSession.id);
			if (result.success) {
				showVideoChat = true;
				isAudioEnabled = true;
			}
		}
	}

	async function toggleVideo() {
		if (!selectedSession) return;
		
		isVideoEnabled = !isVideoEnabled;
		await collaborationService.toggleVideo(selectedSession.id, isVideoEnabled);
	}

	async function toggleAudio() {
		if (!selectedSession) return;
		
		isAudioEnabled = !isAudioEnabled;
		await collaborationService.toggleAudio(selectedSession.id, isAudioEnabled);
	}

	async function toggleScreenShare() {
		if (!selectedSession) return;
		
		isScreenSharing = !isScreenSharing;
		await collaborationService.toggleScreenShare(selectedSession.id, isScreenSharing);
	}

	// Computed properties
	$: filteredSessions = $collaborationSessions.filter(session => {
		if (searchQuery && !session.name.toLowerCase().includes(searchQuery.toLowerCase())) {
			return false;
		}

		if (filterStatus !== 'all') {
			if (filterStatus === 'active' && !session.isActive) return false;
			if (filterStatus === 'inactive' && session.isActive) return false;
		}

		return true;
	});

	function getSessionStatusColor(session: CollaborationSession): string {
		if (session.isActive) return 'text-green-400';
		return 'text-dark-400';
	}

	function formatLastActivity(date: string): string {
		const now = new Date();
		const activityDate = new Date(date);
		const diffMs = now.getTime() - activityDate.getTime();
		const diffMins = Math.floor(diffMs / 60000);
		const diffHours = Math.floor(diffMins / 60);
		const diffDays = Math.floor(diffHours / 24);

		if (diffMins < 1) return 'Just now';
		if (diffMins < 60) return `${diffMins}m ago`;
		if (diffHours < 24) return `${diffHours}h ago`;
		return `${diffDays}d ago`;
	}
</script>

<svelte:head>
	<title>Collaboration - Agentic AI Platform</title>
	<meta name="description" content="Real-time collaboration, team management, and communication" />
</svelte:head>

<div class="collaboration-page p-6 space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold text-white">Collaboration Hub</h1>
			<p class="text-dark-300 mt-1">Real-time collaboration, team management, and communication</p>
		</div>
		<div class="flex items-center space-x-3">
			{#if selectedSession}
				<div class="flex items-center space-x-2 bg-dark-800 border border-dark-700 rounded-lg px-3 py-2">
					<div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
					<span class="text-sm text-white">In Session: {selectedSession.name}</span>
				</div>
				<button
					on:click={toggleVideoChat}
					class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
				>
					{#if showVideoChat}
						<PhoneOff class="w-4 h-4" />
						<span>Leave Call</span>
					{:else}
						<Video class="w-4 h-4" />
						<span>Join Call</span>
					{/if}
				</button>
			{/if}
			<button
				on:click={handleCreateSession}
				class="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
			>
				<Plus class="w-4 h-4" />
				<span>New Session</span>
			</button>
			<button
				on:click={loadData}
				disabled={$collaborationLoading}
				class="bg-dark-700 hover:bg-dark-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
			>
				<RefreshCw class="w-4 h-4 {$collaborationLoading ? 'animate-spin' : ''}" />
				<span>Refresh</span>
			</button>
		</div>
	</div>

	<!-- Active Session Controls -->
	{#if selectedSession && showVideoChat}
		<div class="bg-dark-800 border border-dark-700 rounded-lg p-4">
			<div class="flex items-center justify-between">
				<div class="flex items-center space-x-4">
					<div class="flex items-center space-x-2">
						<div class="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
						<span class="text-white font-medium">Live Session</span>
					</div>
					<div class="text-sm text-dark-300">
						{participants.length} participant{participants.length !== 1 ? 's' : ''}
					</div>
				</div>
				<div class="flex items-center space-x-2">
					<button
						on:click={toggleAudio}
						class="p-2 rounded-lg transition-colors {isAudioEnabled ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'}"
						title={isAudioEnabled ? 'Mute' : 'Unmute'}
					>
						{#if isAudioEnabled}
							<Mic class="w-4 h-4 text-white" />
						{:else}
							<MicOff class="w-4 h-4 text-white" />
						{/if}
					</button>
					<button
						on:click={toggleVideo}
						class="p-2 rounded-lg transition-colors {isVideoEnabled ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'}"
						title={isVideoEnabled ? 'Turn off camera' : 'Turn on camera'}
					>
						{#if isVideoEnabled}
							<Camera class="w-4 h-4 text-white" />
						{:else}
							<VideoOff class="w-4 h-4 text-white" />
						{/if}
					</button>
					<button
						on:click={toggleScreenShare}
						class="p-2 rounded-lg transition-colors {isScreenSharing ? 'bg-blue-600 hover:bg-blue-700' : 'bg-dark-600 hover:bg-dark-500'}"
						title={isScreenSharing ? 'Stop sharing' : 'Share screen'}
					>
						<Screen class="w-4 h-4 text-white" />
					</button>
				</div>
			</div>
		</div>
	{/if}

	<!-- Navigation Tabs -->
	<div class="bg-dark-800 border border-dark-700 rounded-lg">
		<div class="flex border-b border-dark-700">
			<button
				on:click={() => activeTab = 'sessions'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'sessions' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<Users class="w-4 h-4" />
					<span>Sessions</span>
				</div>
			</button>
			<button
				on:click={() => activeTab = 'teams'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'teams' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<Crown class="w-4 h-4" />
					<span>Teams</span>
				</div>
			</button>
			<button
				on:click={() => activeTab = 'activity'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'activity' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<Activity class="w-4 h-4" />
					<span>Activity</span>
				</div>
			</button>
			<button
				on:click={() => activeTab = 'settings'}
				class="px-6 py-3 text-sm font-medium transition-colors {
					activeTab === 'settings' 
						? 'text-primary-400 border-b-2 border-primary-400' 
						: 'text-dark-300 hover:text-white'
				}"
			>
				<div class="flex items-center space-x-2">
					<Settings class="w-4 h-4" />
					<span>Settings</span>
				</div>
			</button>
		</div>

		<!-- Tab Content -->
		<div class="p-6">
			<!-- Sessions Tab -->
			{#if activeTab === 'sessions'}
				<div class="space-y-4">
					<!-- Sessions Header -->
					<div class="flex items-center justify-between">
						<div class="flex items-center space-x-4">
							<div class="relative">
								<Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-dark-400" />
								<input
									type="text"
									bind:value={searchQuery}
									placeholder="Search sessions..."
									class="pl-10 pr-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
								/>
							</div>
							<select
								bind:value={filterStatus}
								class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
							>
								<option value="all">All Sessions</option>
								<option value="active">Active</option>
								<option value="inactive">Inactive</option>
							</select>
						</div>
					</div>

					<!-- Sessions List -->
					{#if $collaborationLoading}
						<div class="flex items-center justify-center py-12">
							<div class="text-center">
								<Loader class="w-8 h-8 text-primary-500 animate-spin mx-auto mb-4" />
								<p class="text-dark-300">Loading sessions...</p>
							</div>
						</div>
					{:else if filteredSessions.length === 0}
						<div class="text-center py-12">
							<Users class="w-12 h-12 text-dark-400 mx-auto mb-4" />
							<h3 class="text-lg font-semibold text-white mb-2">No Sessions Found</h3>
							<p class="text-dark-300 mb-4">
								{searchQuery ? 'No sessions match your search.' : 'Create your first collaboration session.'}
							</p>
							<button
								on:click={handleCreateSession}
								class="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
							>
								Create Session
							</button>
						</div>
					{:else}
						<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
							{#each filteredSessions as session (session.id)}
								<div class="bg-dark-700 border border-dark-600 rounded-lg p-6 hover:bg-dark-600 transition-colors">
									<div class="flex items-start justify-between mb-4">
										<div>
											<h3 class="font-semibold text-white mb-1">{session.name}</h3>
											<p class="text-sm text-dark-300">{session.description || 'No description'}</p>
										</div>
										<div class="flex items-center space-x-1">
											<div class="w-2 h-2 rounded-full {session.isActive ? 'bg-green-400' : 'bg-dark-400'}"></div>
											<span class="text-xs {getSessionStatusColor(session)}">
												{session.isActive ? 'Active' : 'Inactive'}
											</span>
										</div>
									</div>

									<div class="flex items-center justify-between text-sm text-dark-300 mb-4">
										<div class="flex items-center space-x-4">
											<div class="flex items-center space-x-1">
												<Users class="w-4 h-4" />
												<span>{session.participants?.length || 0}</span>
											</div>
											<div class="flex items-center space-x-1">
												<Clock class="w-4 h-4" />
												<span>{formatLastActivity(session.lastActivity)}</span>
											</div>
										</div>
									</div>

									<div class="flex items-center space-x-2">
										{#if session.isActive}
											<button
												on:click={() => handleJoinSession(session)}
												class="flex-1 bg-primary-600 hover:bg-primary-700 text-white py-2 px-3 rounded text-sm transition-colors"
											>
												Join Session
											</button>
										{:else}
											<button
												on:click={() => handleJoinSession(session)}
												class="flex-1 bg-dark-600 hover:bg-dark-500 text-white py-2 px-3 rounded text-sm transition-colors"
											>
												Resume Session
											</button>
										{/if}
										<button
											class="p-2 text-dark-400 hover:text-white transition-colors"
											title="More options"
										>
											<MoreVertical class="w-4 h-4" />
										</button>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{/if}

			<!-- Teams Tab -->
			{#if activeTab === 'teams'}
				<TeamManagement />
			{/if}

			<!-- Activity Tab -->
			{#if activeTab === 'activity'}
				<ActivityFeed activities={realtimeActivities} />
			{/if}

			<!-- Settings Tab -->
			{#if activeTab === 'settings'}
				<div class="space-y-6">
					<h3 class="text-lg font-semibold text-white">Collaboration Settings</h3>
					
					<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
						<div class="bg-dark-700 border border-dark-600 rounded-lg p-6">
							<h4 class="text-white font-medium mb-4">Default Permissions</h4>
							<div class="space-y-3">
								<label class="flex items-center space-x-3">
									<input type="checkbox" class="w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500" />
									<span class="text-dark-300">Allow guests to join sessions</span>
								</label>
								<label class="flex items-center space-x-3">
									<input type="checkbox" class="w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500" />
									<span class="text-dark-300">Enable video chat by default</span>
								</label>
								<label class="flex items-center space-x-3">
									<input type="checkbox" class="w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500" />
									<span class="text-dark-300">Auto-save collaborative changes</span>
								</label>
							</div>
						</div>
						
						<div class="bg-dark-700 border border-dark-600 rounded-lg p-6">
							<h4 class="text-white font-medium mb-4">Notification Preferences</h4>
							<div class="space-y-3">
								<label class="flex items-center space-x-3">
									<input type="checkbox" class="w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500" />
									<span class="text-dark-300">Session invitations</span>
								</label>
								<label class="flex items-center space-x-3">
									<input type="checkbox" class="w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500" />
									<span class="text-dark-300">Comments and mentions</span>
								</label>
								<label class="flex items-center space-x-3">
									<input type="checkbox" class="w-4 h-4 text-primary-500 bg-dark-700 border-dark-600 rounded focus:ring-primary-500" />
									<span class="text-dark-300">Activity updates</span>
								</label>
							</div>
						</div>
					</div>
				</div>
			{/if}
		</div>
	</div>
</div>

<!-- Collaborative Editor Modal -->
{#if showEditor && selectedSession}
	<CollaborativeEditor
		session={selectedSession}
		on:close={() => { showEditor = false; selectedSession = null; }}
		on:leave={() => handleLeaveSession(selectedSession)}
	/>
{/if}

<!-- Video Chat Component -->
{#if showVideoChat && selectedSession}
	<VideoChat
		session={selectedSession}
		{participants}
		{isVideoEnabled}
		{isAudioEnabled}
		{isScreenSharing}
		on:toggleVideo={toggleVideo}
		on:toggleAudio={toggleAudio}
		on:toggleScreenShare={toggleScreenShare}
		on:leave={toggleVideoChat}
	/>
{/if}

<!-- Team Management Modal -->
{#if showTeamManagement}
	<TeamManagement
		on:close={() => showTeamManagement = false}
	/>
{/if}

<style lang="postcss">
	.collaboration-page {
		animation: fadeIn 0.6s ease-out;
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
