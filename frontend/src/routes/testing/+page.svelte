<script lang="ts">
	import { onMount } from 'svelte';
	import { 
		agents, 
		testing, 
		notificationActions 
	} from '$stores';
	import { apiClient } from '$services/api';
	import { websocketService } from '$services/websocket';
	import { 
		TestTube, 
		Send, 
		Bot, 
		User, 
		Play, 
		Square, 
		RotateCcw, 
		Download,
		Settings,
		MessageSquare,
		Zap,
		Clock,
		CheckCircle,
		AlertCircle,
		Loader
	} from 'lucide-svelte';
	import type { Agent } from '$types';
	
	// State
	let selectedAgent: Agent | null = null;
	let chatInput = '';
	let isExecuting = false;
	let showSettings = false;
	
	// Test settings
	let testSettings = {
		temperature: 0.7,
		maxTokens: 1000,
		streaming: true,
		saveHistory: true
	};
	
	// Mock test scenarios for demonstration
	let testScenarios = [
		{
			id: '1',
			name: 'Basic Greeting',
			input: 'Hello! How can you help me today?',
			category: 'General'
		},
		{
			id: '2',
			name: 'Complex Query',
			input: 'Can you analyze the quarterly sales data and provide insights on customer behavior trends?',
			category: 'Analysis'
		},
		{
			id: '3',
			name: 'Multi-step Task',
			input: 'I need to plan a marketing campaign for a new product launch. Can you help me create a comprehensive strategy?',
			category: 'Planning'
		},
		{
			id: '4',
			name: 'Error Handling',
			input: 'What happens when you encounter invalid data or unclear instructions?',
			category: 'Edge Cases'
		}
	];
	
	onMount(async () => {
		// Load agents if not already loaded
		if ($agents.length === 0) {
			await loadAgents();
		}
		
		// Subscribe to real-time agent responses
		websocketService.on('agent_response' as any, (data) => {
			if (data.agent_id === selectedAgent?.id) {
				addMessage({
					id: crypto.randomUUID(),
					type: 'agent',
					content: data.response,
					timestamp: new Date().toISOString(),
					metadata: data.metadata
				});
				isExecuting = false;
			}
		});
		
		// Subscribe to agent errors
		websocketService.on('agent_error' as any, (data) => {
			if (data.agent_id === selectedAgent?.id) {
				addMessage({
					id: crypto.randomUUID(),
					type: 'error',
					content: `Error: ${data.error}`,
					timestamp: new Date().toISOString()
				});
				isExecuting = false;
			}
		});
	});
	
	async function loadAgents() {
		try {
			const response = await apiClient.getAgents();
			if (response.success && response.data) {
				agents.set(response.data);
				// Auto-select first agent if available
				if (response.data.length > 0) {
					selectedAgent = response.data[0];
				}
			}
		} catch (error) {
			console.error('Failed to load agents:', error);
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Agents',
				message: 'Unable to load agents for testing'
			});
		}
	}
	
	function addMessage(message: any) {
		testing.chatMessages.update(messages => [...messages, message]);
		
		// Auto-scroll to bottom
		setTimeout(() => {
			const chatContainer = document.getElementById('chat-container');
			if (chatContainer) {
				chatContainer.scrollTop = chatContainer.scrollHeight;
			}
		}, 100);
	}
	
	async function sendMessage() {
		if (!chatInput.trim() || !selectedAgent || isExecuting) return;
		
		const userMessage = {
			id: crypto.randomUUID(),
			type: 'user',
			content: chatInput.trim(),
			timestamp: new Date().toISOString()
		};
		
		addMessage(userMessage);
		
		// Clear input
		const input = chatInput.trim();
		chatInput = '';
		isExecuting = true;
		
		// Add thinking indicator
		const thinkingMessage = {
			id: 'thinking',
			type: 'thinking',
			content: 'Agent is thinking...',
			timestamp: new Date().toISOString()
		};
		addMessage(thinkingMessage);
		
		try {
			const response = await apiClient.executeAgent(selectedAgent.id, {
				message: input,
				settings: testSettings
			});
			
			// Remove thinking indicator
			testing.chatMessages.update(messages => 
				messages.filter(m => m.id !== 'thinking')
			);
			
			if (response.success) {
				addMessage({
					id: crypto.randomUUID(),
					type: 'agent',
					content: response.data.response || 'Agent executed successfully',
					timestamp: new Date().toISOString(),
					metadata: response.data.metadata
				});
			} else {
				throw new Error(response.error || 'Agent execution failed');
			}
		} catch (error) {
			// Remove thinking indicator
			testing.chatMessages.update(messages => 
				messages.filter(m => m.id !== 'thinking')
			);
			
			addMessage({
				id: crypto.randomUUID(),
				type: 'error',
				content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
				timestamp: new Date().toISOString()
			});
		} finally {
			isExecuting = false;
		}
	}
	
	function selectAgent(agent: Agent) {
		selectedAgent = agent;
		// Clear chat when switching agents
		testing.actions.clearChat();
	}
	
	function clearChat() {
		testing.actions.clearChat();
	}
	
	function stopExecution() {
		isExecuting = false;
		// Remove thinking indicator
		testing.chatMessages.update(messages => 
			messages.filter(m => m.id !== 'thinking')
		);
		
		notificationActions.add({
			type: 'info',
			title: 'Execution Stopped',
			message: 'Agent execution has been stopped'
		});
	}
	
	function useTestScenario(scenario: any) {
		chatInput = scenario.input;
	}
	
	function exportChatHistory() {
		const history = $testing.chatMessages.map(msg => ({
			timestamp: msg.timestamp,
			type: msg.type,
			content: msg.content
		}));
		
		const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `chat-history-${selectedAgent?.name || 'agent'}-${new Date().toISOString().split('T')[0]}.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
		
		notificationActions.add({
			type: 'success',
			title: 'Chat History Exported',
			message: 'Chat history has been downloaded'
		});
	}
	
	function formatTime(timestamp: string): string {
		return new Date(timestamp).toLocaleTimeString('en-US', {
			hour: '2-digit',
			minute: '2-digit'
		});
	}
	
	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}
</script>

<svelte:head>
	<title>Testing Lab - Agentic AI</title>
</svelte:head>

<div class="testing-container">
	<!-- Header -->
	<div class="testing-header">
		<div class="header-content">
			<div class="header-text">
				<h1 class="page-title">Testing Lab</h1>
				<p class="page-subtitle">
					Test and validate your AI agents in real-time with our comprehensive testing environment.
				</p>
			</div>
			
			<div class="header-actions">
				<button 
					class="action-btn secondary"
					on:click={() => showSettings = !showSettings}
				>
					<Settings class="w-4 h-4" />
					<span>Settings</span>
				</button>
				
				{#if $testing.chatMessages.length > 0}
					<button class="action-btn secondary" on:click={exportChatHistory}>
						<Download class="w-4 h-4" />
						<span>Export</span>
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Main Testing Interface -->
	<div class="testing-interface">
		<!-- Left Sidebar - Agent Selection & Scenarios -->
		<div class="testing-sidebar">
			<!-- Agent Selection -->
			<div class="sidebar-section">
				<h3 class="section-title">Select Agent</h3>
				<div class="agent-list">
					{#each $agents as agent (agent.id)}
						<button
							class="agent-item {selectedAgent?.id === agent.id ? 'selected' : ''}"
							on:click={() => selectAgent(agent)}
						>
							<div class="agent-icon">
								<Bot class="w-5 h-5 text-accent-blue" />
							</div>
							<div class="agent-info">
								<span class="agent-name">{agent.name}</span>
								<span class="agent-type">{agent.type}</span>
							</div>
							<div class="agent-status {agent.status === 'running' ? 'active' : ''}">
								<div class="status-dot"></div>
							</div>
						</button>
					{/each}
				</div>
			</div>
			
			<!-- Test Scenarios -->
			<div class="sidebar-section">
				<h3 class="section-title">Test Scenarios</h3>
				<div class="scenario-list">
					{#each testScenarios as scenario (scenario.id)}
						<button
							class="scenario-item"
							on:click={() => useTestScenario(scenario)}
						>
							<div class="scenario-content">
								<h4 class="scenario-name">{scenario.name}</h4>
								<p class="scenario-preview">{scenario.input.slice(0, 60)}...</p>
								<span class="scenario-category">{scenario.category}</span>
							</div>
						</button>
					{/each}
				</div>
			</div>
			
			<!-- Settings Panel -->
			{#if showSettings}
				<div class="sidebar-section">
					<h3 class="section-title">Test Settings</h3>
					<div class="settings-form">
						<div class="form-group">
							<label class="form-label">Temperature</label>
							<input
								type="range"
								min="0"
								max="1"
								step="0.1"
								class="form-range"
								bind:value={testSettings.temperature}
							/>
							<span class="range-value">{testSettings.temperature}</span>
						</div>
						
						<div class="form-group">
							<label class="form-label">Max Tokens</label>
							<input
								type="number"
								min="100"
								max="4000"
								step="100"
								class="form-input"
								bind:value={testSettings.maxTokens}
							/>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={testSettings.streaming}
								/>
								<span class="checkbox-label">Enable Streaming</span>
							</label>
						</div>
						
						<div class="form-group">
							<label class="form-checkbox">
								<input
									type="checkbox"
									bind:checked={testSettings.saveHistory}
								/>
								<span class="checkbox-label">Save Chat History</span>
							</label>
						</div>
					</div>
				</div>
			{/if}
		</div>
		
		<!-- Main Chat Interface -->
		<div class="chat-interface">
			{#if selectedAgent}
				<!-- Chat Header -->
				<div class="chat-header">
					<div class="chat-agent-info">
						<div class="agent-avatar">
							<Bot class="w-6 h-6 text-accent-blue" />
						</div>
						<div class="agent-details">
							<h3 class="agent-name">{selectedAgent.name}</h3>
							<p class="agent-description">{selectedAgent.description}</p>
						</div>
					</div>
					
					<div class="chat-actions">
						{#if isExecuting}
							<button class="action-btn danger" on:click={stopExecution}>
								<Square class="w-4 h-4" />
								<span>Stop</span>
							</button>
						{/if}
						
						<button class="action-btn secondary" on:click={clearChat}>
							<RotateCcw class="w-4 h-4" />
							<span>Clear</span>
						</button>
					</div>
				</div>
				
				<!-- Chat Messages -->
				<div id="chat-container" class="chat-messages">
					{#if $testing.chatMessages.length === 0}
						<div class="empty-chat">
							<div class="empty-icon">
								<MessageSquare class="w-12 h-12 text-dark-500" />
							</div>
							<h3 class="empty-title">Start Testing</h3>
							<p class="empty-message">
								Send a message to start testing your agent. Try one of the test scenarios or create your own.
							</p>
						</div>
					{:else}
						{#each $testing.chatMessages as message (message.id)}
							<div class="message {message.type}">
								{#if message.type === 'user'}
									<div class="message-avatar">
										<User class="w-5 h-5" />
									</div>
								{:else if message.type === 'agent'}
									<div class="message-avatar agent">
										<Bot class="w-5 h-5" />
									</div>
								{:else if message.type === 'thinking'}
									<div class="message-avatar thinking">
										<Loader class="w-5 h-5 animate-spin" />
									</div>
								{:else if message.type === 'error'}
									<div class="message-avatar error">
										<AlertCircle class="w-5 h-5" />
									</div>
								{/if}
								
								<div class="message-content">
									<div class="message-text">{message.content}</div>
									<div class="message-meta">
										<span class="message-time">{formatTime(message.timestamp)}</span>
										{#if message.metadata}
											<span class="message-tokens">
												{message.metadata.tokens_used} tokens
											</span>
											<span class="message-duration">
												{message.metadata.duration}ms
											</span>
										{/if}
									</div>
								</div>
							</div>
						{/each}
					{/if}
				</div>
				
				<!-- Chat Input -->
				<div class="chat-input-container">
					<div class="input-wrapper">
						<textarea
							class="chat-input"
							placeholder="Type your message to test the agent..."
							rows="3"
							bind:value={chatInput}
							on:keydown={handleKeydown}
							disabled={isExecuting}
						></textarea>
						
						<button
							class="send-btn"
							on:click={sendMessage}
							disabled={!chatInput.trim() || isExecuting}
						>
							{#if isExecuting}
								<Loader class="w-5 h-5 animate-spin" />
							{:else}
								<Send class="w-5 h-5" />
							{/if}
						</button>
					</div>
					
					<div class="input-footer">
						<span class="input-hint">Press Enter to send, Shift+Enter for new line</span>
						{#if isExecuting}
							<span class="execution-status">
								<Zap class="w-4 h-4 text-accent-blue animate-pulse" />
								Agent is processing...
							</span>
						{/if}
					</div>
				</div>
			{:else}
				<!-- No Agent Selected -->
				<div class="no-agent-selected">
					<div class="no-agent-icon">
						<TestTube class="w-16 h-16 text-dark-500" />
					</div>
					<h3 class="no-agent-title">Select an Agent to Test</h3>
					<p class="no-agent-message">
						Choose an agent from the sidebar to start testing its capabilities and performance.
					</p>
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.testing-container {
		@apply space-y-6;
	}
	
	/* Header */
	.testing-header {
		@apply space-y-4;
	}
	
	.header-content {
		@apply flex items-center justify-between;
	}
	
	.header-text {
		@apply space-y-2;
	}
	
	.page-title {
		@apply text-3xl font-bold text-white;
	}
	
	.page-subtitle {
		@apply text-lg text-dark-300 max-w-2xl;
	}
	
	.header-actions {
		@apply flex items-center space-x-3;
	}
	
	.action-btn {
		@apply inline-flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200;
	}
	
	.action-btn.secondary {
		@apply bg-dark-800 text-dark-300 hover:bg-dark-700 hover:text-white;
	}
	
	.action-btn.danger {
		@apply bg-red-600 text-white hover:bg-red-700;
	}
	
	/* Testing Interface */
	.testing-interface {
		@apply grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-200px)];
	}
	
	.testing-sidebar {
		@apply lg:col-span-1 space-y-6 overflow-y-auto;
	}
	
	.sidebar-section {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-4;
	}
	
	.section-title {
		@apply text-lg font-semibold text-white mb-4;
	}
	
	/* Agent List */
	.agent-list {
		@apply space-y-2;
	}
	
	.agent-item {
		@apply flex items-center space-x-3 w-full p-3 bg-dark-700 hover:bg-dark-600 rounded-lg transition-all duration-200 text-left;
	}
	
	.agent-item.selected {
		@apply bg-primary-600/20 border border-primary-500/30;
	}
	
	.agent-icon {
		@apply w-10 h-10 bg-dark-600 rounded-lg flex items-center justify-center;
	}
	
	.agent-info {
		@apply flex-1 space-y-1;
	}
	
	.agent-name {
		@apply block text-sm font-medium text-white;
	}
	
	.agent-type {
		@apply block text-xs text-dark-400;
	}
	
	.agent-status {
		@apply flex items-center;
	}
	
	.status-dot {
		@apply w-2 h-2 bg-gray-500 rounded-full;
	}
	
	.agent-status.active .status-dot {
		@apply bg-green-500 animate-pulse;
	}
	
	/* Scenario List */
	.scenario-list {
		@apply space-y-2;
	}
	
	.scenario-item {
		@apply w-full p-3 bg-dark-700 hover:bg-dark-600 rounded-lg transition-all duration-200 text-left;
	}
	
	.scenario-content {
		@apply space-y-2;
	}
	
	.scenario-name {
		@apply text-sm font-medium text-white;
	}
	
	.scenario-preview {
		@apply text-xs text-dark-300;
	}
	
	.scenario-category {
		@apply inline-block px-2 py-1 text-xs bg-dark-600 text-dark-400 rounded;
	}
	
	/* Settings Form */
	.settings-form {
		@apply space-y-4;
	}
	
	.form-group {
		@apply space-y-2;
	}
	
	.form-label {
		@apply block text-sm font-medium text-white;
	}
	
	.form-input,
	.form-range {
		@apply w-full px-3 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white;
	}
	
	.range-value {
		@apply text-sm text-dark-400;
	}
	
	.form-checkbox {
		@apply flex items-center space-x-2 cursor-pointer;
	}
	
	.checkbox-label {
		@apply text-sm text-white;
	}
	
	/* Chat Interface */
	.chat-interface {
		@apply lg:col-span-3 bg-dark-800 border border-dark-700 rounded-xl flex flex-col;
	}
	
	.chat-header {
		@apply flex items-center justify-between p-4 border-b border-dark-700;
	}
	
	.chat-agent-info {
		@apply flex items-center space-x-3;
	}
	
	.agent-avatar {
		@apply w-12 h-12 bg-dark-700 rounded-xl flex items-center justify-center;
	}
	
	.agent-details {
		@apply space-y-1;
	}
	
	.chat-actions {
		@apply flex items-center space-x-2;
	}
	
	/* Chat Messages */
	.chat-messages {
		@apply flex-1 overflow-y-auto p-4 space-y-4;
	}
	
	.empty-chat {
		@apply flex flex-col items-center justify-center h-full text-center;
	}
	
	.empty-icon {
		@apply mb-4;
	}
	
	.empty-title {
		@apply text-xl font-semibold text-white mb-2;
	}
	
	.empty-message {
		@apply text-dark-400 max-w-md;
	}
	
	.message {
		@apply flex items-start space-x-3;
	}
	
	.message.user {
		@apply flex-row-reverse space-x-reverse;
	}
	
	.message-avatar {
		@apply w-8 h-8 bg-dark-700 rounded-lg flex items-center justify-center text-dark-400;
	}
	
	.message-avatar.agent {
		@apply bg-primary-600/20 text-primary-400;
	}
	
	.message-avatar.thinking {
		@apply bg-yellow-600/20 text-yellow-400;
	}
	
	.message-avatar.error {
		@apply bg-red-600/20 text-red-400;
	}
	
	.message-content {
		@apply flex-1 space-y-1;
	}
	
	.message.user .message-content {
		@apply text-right;
	}
	
	.message-text {
		@apply p-3 rounded-lg text-white;
		background: rgba(55, 65, 81, 0.5);
	}
	
	.message.user .message-text {
		@apply bg-primary-600/20;
	}
	
	.message.error .message-text {
		@apply bg-red-600/20 text-red-300;
	}
	
	.message-meta {
		@apply flex items-center space-x-2 text-xs text-dark-400;
	}
	
	.message.user .message-meta {
		@apply justify-end;
	}
	
	/* Chat Input */
	.chat-input-container {
		@apply p-4 border-t border-dark-700;
	}
	
	.input-wrapper {
		@apply relative flex items-end space-x-3;
	}
	
	.chat-input {
		@apply flex-1 p-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 resize-none focus:outline-none focus:ring-2 focus:ring-primary-500;
	}
	
	.send-btn {
		@apply p-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200;
	}
	
	.input-footer {
		@apply flex items-center justify-between mt-2 text-xs text-dark-400;
	}
	
	.execution-status {
		@apply flex items-center space-x-1 text-accent-blue;
	}
	
	/* No Agent Selected */
	.no-agent-selected {
		@apply flex flex-col items-center justify-center h-full text-center;
	}
	
	.no-agent-icon {
		@apply mb-6;
	}
	
	.no-agent-title {
		@apply text-2xl font-semibold text-white mb-2;
	}
	
	.no-agent-message {
		@apply text-dark-400 max-w-md;
	}
	
	/* Mobile Responsiveness */
	@media (max-width: 1024px) {
		.testing-interface {
			@apply grid-cols-1 h-auto;
		}
		
		.testing-sidebar {
			@apply order-2;
		}
		
		.chat-interface {
			@apply order-1 h-96;
		}
	}
</style>
