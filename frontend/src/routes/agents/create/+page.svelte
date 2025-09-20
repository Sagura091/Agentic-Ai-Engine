<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { agentBuilder, notificationActions } from '$stores';
	import { apiClient } from '$services/api';
	import { 
		ArrowLeft, 
		ArrowRight, 
		Check, 
		Bot, 
		Brain, 
		Zap, 
		Settings,
		Sparkles,
		ChevronRight
	} from 'lucide-svelte';
	import type { AgentType, AgentTemplate, AgentCapability, MemoryType } from '$types';
	
	// Agent Builder Steps
	const steps = [
		{ id: 1, title: 'Basic Info', description: 'Name, description, and type' },
		{ id: 2, title: 'Template', description: 'Choose a starting template' },
		{ id: 3, title: 'Configuration', description: 'Model, capabilities, and tools' },
		{ id: 4, title: 'Review', description: 'Review and create agent' }
	];
	
	// Agent Types
	const agentTypes = [
		{
			type: 'react' as AgentType,
			name: 'ReAct Agent',
			description: 'Reasoning and Acting agent with tool use capabilities',
			icon: Brain,
			color: 'text-blue-400',
			features: ['Reasoning', 'Tool Use', 'Problem Solving']
		},
		{
			type: 'knowledge_search' as AgentType,
			name: 'Knowledge Search',
			description: 'Specialized in searching and retrieving information',
			icon: Bot,
			color: 'text-green-400',
			features: ['Information Retrieval', 'Search', 'Knowledge Base']
		},
		{
			type: 'rag' as AgentType,
			name: 'RAG Agent',
			description: 'Retrieval-Augmented Generation for document-based tasks',
			icon: Sparkles,
			color: 'text-purple-400',
			features: ['Document Processing', 'Context Retrieval', 'Generation']
		},
		{
			type: 'workflow' as AgentType,
			name: 'Workflow Agent',
			description: 'Custom process automation and workflow execution',
			icon: Settings,
			color: 'text-orange-400',
			features: ['Process Automation', 'Custom Workflows', 'Integration']
		},
		{
			type: 'multimodal' as AgentType,
			name: 'Multimodal Agent',
			description: 'Handles text, images, audio, and other media types',
			icon: Zap,
			color: 'text-pink-400',
			features: ['Vision', 'Audio Processing', 'Multi-format Input']
		},
		{
			type: 'autonomous' as AgentType,
			name: 'Autonomous Agent',
			description: 'Self-directed agent with advanced reasoning capabilities',
			icon: Brain,
			color: 'text-red-400',
			features: ['Self-Direction', 'Advanced Reasoning', 'Goal Setting']
		}
	];
	
	// Agent Templates
	const templates = [
		{
			template: 'research_assistant' as AgentTemplate,
			name: 'Research Assistant',
			description: 'Helps with research tasks, data analysis, and information gathering',
			icon: '🔬',
			compatibleTypes: ['react', 'knowledge_search', 'rag']
		},
		{
			template: 'customer_support' as AgentTemplate,
			name: 'Customer Support',
			description: 'Provides customer service and support interactions',
			icon: '🎧',
			compatibleTypes: ['react', 'rag']
		},
		{
			template: 'data_analyst' as AgentTemplate,
			name: 'Data Analyst',
			description: 'Analyzes data, creates reports, and provides insights',
			icon: '📊',
			compatibleTypes: ['react', 'workflow']
		},
		{
			template: 'creative_writer' as AgentTemplate,
			name: 'Creative Writer',
			description: 'Generates creative content, stories, and marketing copy',
			icon: '✍️',
			compatibleTypes: ['react', 'multimodal']
		},
		{
			template: 'code_reviewer' as AgentTemplate,
			name: 'Code Reviewer',
			description: 'Reviews code, suggests improvements, and finds bugs',
			icon: '💻',
			compatibleTypes: ['react', 'workflow']
		},
		{
			template: 'custom' as AgentTemplate,
			name: 'Custom Agent',
			description: 'Start from scratch with a blank configuration',
			icon: '🛠️',
			compatibleTypes: ['react', 'knowledge_search', 'rag', 'workflow', 'multimodal', 'autonomous']
		}
	];
	
	// Capabilities
	const capabilities = [
		{ id: 'reasoning' as AgentCapability, name: 'Reasoning', description: 'Logical thinking and problem solving' },
		{ id: 'tool_use' as AgentCapability, name: 'Tool Use', description: 'Ability to use external tools and APIs' },
		{ id: 'memory' as AgentCapability, name: 'Memory', description: 'Remember context and past interactions' },
		{ id: 'planning' as AgentCapability, name: 'Planning', description: 'Create and execute multi-step plans' },
		{ id: 'learning' as AgentCapability, name: 'Learning', description: 'Adapt and improve from experience' },
		{ id: 'collaboration' as AgentCapability, name: 'Collaboration', description: 'Work with other agents' },
		{ id: 'vision' as AgentCapability, name: 'Vision', description: 'Process and understand images' },
		{ id: 'audio' as AgentCapability, name: 'Audio', description: 'Process speech and audio content' }
	];
	
	// Memory Types
	const memoryTypes = [
		{ type: 'auto' as MemoryType, name: 'Automatic', description: 'System chooses optimal memory type' },
		{ type: 'simple' as MemoryType, name: 'Simple', description: 'Basic short-term and long-term memory' },
		{ type: 'advanced' as MemoryType, name: 'Advanced', description: 'Episodic, semantic, and procedural memory' },
		{ type: 'none' as MemoryType, name: 'None', description: 'No persistent memory' }
	];
	
	// State
	let currentStep = 1;
	let isCreating = false;
	
	// Get compatible templates for selected agent type
	$: compatibleTemplates = $agentBuilder.basicInfo.type 
		? templates.filter(t => t.compatibleTypes.includes($agentBuilder.basicInfo.type))
		: templates;
	
	onMount(() => {
		// Reset builder state
		agentBuilder.actions.reset();
	});
	
	function nextStep() {
		if (validateCurrentStep()) {
			currentStep = Math.min(currentStep + 1, 4);
			agentBuilder.actions.setStep(currentStep);
		}
	}
	
	function prevStep() {
		currentStep = Math.max(currentStep - 1, 1);
		agentBuilder.actions.setStep(currentStep);
	}
	
	function validateCurrentStep(): boolean {
		switch (currentStep) {
			case 1:
				return $agentBuilder.basicInfo.name.trim() !== '' && 
				       $agentBuilder.basicInfo.description.trim() !== '' &&
				       $agentBuilder.basicInfo.type !== '';
			case 2:
				return $agentBuilder.basicInfo.template !== undefined;
			case 3:
				return $agentBuilder.configuration.model !== '' &&
				       $agentBuilder.configuration.capabilities.length > 0;
			default:
				return true;
		}
	}
	
	async function createAgent() {
		if (!validateCurrentStep()) return;
		
		isCreating = true;
		
		try {
			const agentData = {
				name: $agentBuilder.basicInfo.name,
				description: $agentBuilder.basicInfo.description,
				type: $agentBuilder.basicInfo.type,
				template: $agentBuilder.basicInfo.template,
				model: $agentBuilder.configuration.model,
				capabilities: $agentBuilder.configuration.capabilities,
				memory_type: $agentBuilder.configuration.memory_type,
				tools: $agentBuilder.configuration.tools,
				is_public: false
			};
			
			const response = await apiClient.createAgent(agentData);
			
			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Agent Created Successfully!',
					message: `${agentData.name} is ready to use.`
				});
				
				// Reset builder and navigate to agents list
				agentBuilder.actions.reset();
				goto('/agents');
			} else {
				throw new Error(response.error || 'Failed to create agent');
			}
		} catch (error) {
			console.error('Failed to create agent:', error);
			notificationActions.add({
				type: 'error',
				title: 'Agent Creation Failed',
				message: error instanceof Error ? error.message : 'Unknown error occurred'
			});
		} finally {
			isCreating = false;
		}
	}
	
	function selectAgentType(type: AgentType) {
		agentBuilder.basicInfo.update(info => ({ ...info, type }));
		// Reset template when type changes
		agentBuilder.basicInfo.update(info => ({ ...info, template: undefined }));
	}
	
	function selectTemplate(template: AgentTemplate) {
		agentBuilder.basicInfo.update(info => ({ ...info, template }));
	}
	
	function toggleCapability(capability: AgentCapability) {
		agentBuilder.configuration.update(config => ({
			...config,
			capabilities: config.capabilities.includes(capability)
				? config.capabilities.filter(c => c !== capability)
				: [...config.capabilities, capability]
		}));
	}
</script>

<svelte:head>
	<title>Create Agent - Agentic AI</title>
</svelte:head>

<div class="create-agent-container">
	<!-- Header -->
	<div class="create-header">
		<button class="back-btn" on:click={() => goto('/agents')}>
			<ArrowLeft class="w-5 h-5" />
			<span>Back to Agents</span>
		</button>
		
		<div class="header-content">
			<h1 class="page-title">Create New Agent</h1>
			<p class="page-subtitle">Build an intelligent AI agent with our revolutionary visual builder</p>
		</div>
	</div>
	
	<!-- Progress Steps -->
	<div class="progress-container">
		<div class="progress-steps">
			{#each steps as step, index}
				<div class="step-item {currentStep >= step.id ? 'active' : ''} {currentStep > step.id ? 'completed' : ''}">
					<div class="step-circle">
						{#if currentStep > step.id}
							<Check class="w-4 h-4" />
						{:else}
							<span class="step-number">{step.id}</span>
						{/if}
					</div>
					<div class="step-content">
						<h3 class="step-title">{step.title}</h3>
						<p class="step-description">{step.description}</p>
					</div>
					{#if index < steps.length - 1}
						<ChevronRight class="step-arrow w-5 h-5" />
					{/if}
				</div>
			{/each}
		</div>
	</div>
	
	<!-- Step Content -->
	<div class="step-content-container">
		{#if currentStep === 1}
			<!-- Step 1: Basic Info -->
			<div class="step-panel">
				<div class="panel-header">
					<h2 class="panel-title">Basic Information</h2>
					<p class="panel-subtitle">Let's start with the basics about your agent</p>
				</div>
				
				<div class="form-grid">
					<div class="form-group">
						<label class="form-label" for="agent-name">Agent Name</label>
						<input
							id="agent-name"
							type="text"
							class="form-input"
							placeholder="Enter a name for your agent..."
							bind:value={$agentBuilder.basicInfo.name}
						/>
					</div>
					
					<div class="form-group">
						<label class="form-label" for="agent-description">Description</label>
						<textarea
							id="agent-description"
							class="form-textarea"
							placeholder="Describe what your agent will do..."
							rows="3"
							bind:value={$agentBuilder.basicInfo.description}
						></textarea>
					</div>
					
					<div class="form-group full-width">
						<label class="form-label">Agent Type</label>
						<div class="type-grid">
							{#each agentTypes as type}
								<button
									class="type-card {$agentBuilder.basicInfo.type === type.type ? 'selected' : ''}"
									on:click={() => selectAgentType(type.type)}
								>
									<div class="type-icon {type.color}">
										<svelte:component this={type.icon} class="w-6 h-6" />
									</div>
									<div class="type-content">
										<h3 class="type-name">{type.name}</h3>
										<p class="type-description">{type.description}</p>
										<div class="type-features">
											{#each type.features as feature}
												<span class="feature-tag">{feature}</span>
											{/each}
										</div>
									</div>
								</button>
							{/each}
						</div>
					</div>
				</div>
			</div>
			
		{:else if currentStep === 2}
			<!-- Step 2: Template Selection -->
			<div class="step-panel">
				<div class="panel-header">
					<h2 class="panel-title">Choose a Template</h2>
					<p class="panel-subtitle">Start with a pre-configured template or build from scratch</p>
				</div>
				
				<div class="template-grid">
					{#each compatibleTemplates as template}
						<button
							class="template-card {$agentBuilder.basicInfo.template === template.template ? 'selected' : ''}"
							on:click={() => selectTemplate(template.template)}
						>
							<div class="template-icon">
								<span class="template-emoji">{template.icon}</span>
							</div>
							<div class="template-content">
								<h3 class="template-name">{template.name}</h3>
								<p class="template-description">{template.description}</p>
							</div>
							{#if $agentBuilder.basicInfo.template === template.template}
								<div class="selected-indicator">
									<Check class="w-4 h-4" />
								</div>
							{/if}
						</button>
					{/each}
				</div>
			</div>
			
		{:else if currentStep === 3}
			<!-- Step 3: Configuration -->
			<div class="step-panel">
				<div class="panel-header">
					<h2 class="panel-title">Configuration</h2>
					<p class="panel-subtitle">Configure your agent's capabilities and settings</p>
				</div>
				
				<div class="config-sections">
					<!-- Model Selection -->
					<div class="config-section">
						<h3 class="section-title">Language Model</h3>
						<div class="form-group">
							<select class="form-select" bind:value={$agentBuilder.configuration.model}>
								<option value="">Select a model...</option>
								<option value="gpt-4-turbo">GPT-4 Turbo</option>
								<option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
								<option value="claude-3-sonnet">Claude 3 Sonnet</option>
								<option value="llama-2-70b">Llama 2 70B</option>
								<option value="mistral-7b">Mistral 7B</option>
							</select>
						</div>
					</div>
					
					<!-- Capabilities -->
					<div class="config-section">
						<h3 class="section-title">Capabilities</h3>
						<div class="capabilities-grid">
							{#each capabilities as capability}
								<button
									class="capability-card {$agentBuilder.configuration.capabilities.includes(capability.id) ? 'selected' : ''}"
									on:click={() => toggleCapability(capability.id)}
								>
									<div class="capability-content">
										<h4 class="capability-name">{capability.name}</h4>
										<p class="capability-description">{capability.description}</p>
									</div>
									{#if $agentBuilder.configuration.capabilities.includes(capability.id)}
										<div class="capability-check">
											<Check class="w-4 h-4" />
										</div>
									{/if}
								</button>
							{/each}
						</div>
					</div>
					
					<!-- Memory Type -->
					<div class="config-section">
						<h3 class="section-title">Memory System</h3>
						<div class="memory-grid">
							{#each memoryTypes as memory}
								<button
									class="memory-card {$agentBuilder.configuration.memory_type === memory.type ? 'selected' : ''}"
									on:click={() => agentBuilder.configuration.update(c => ({ ...c, memory_type: memory.type }))}
								>
									<h4 class="memory-name">{memory.name}</h4>
									<p class="memory-description">{memory.description}</p>
								</button>
							{/each}
						</div>
					</div>
				</div>
			</div>
			
		{:else if currentStep === 4}
			<!-- Step 4: Review -->
			<div class="step-panel">
				<div class="panel-header">
					<h2 class="panel-title">Review & Create</h2>
					<p class="panel-subtitle">Review your agent configuration before creating</p>
				</div>
				
				<div class="review-sections">
					<div class="review-section">
						<h3 class="review-title">Basic Information</h3>
						<div class="review-content">
							<div class="review-item">
								<span class="review-label">Name:</span>
								<span class="review-value">{$agentBuilder.basicInfo.name}</span>
							</div>
							<div class="review-item">
								<span class="review-label">Type:</span>
								<span class="review-value">{agentTypes.find(t => t.type === $agentBuilder.basicInfo.type)?.name}</span>
							</div>
							<div class="review-item">
								<span class="review-label">Template:</span>
								<span class="review-value">{templates.find(t => t.template === $agentBuilder.basicInfo.template)?.name}</span>
							</div>
							<div class="review-item">
								<span class="review-label">Description:</span>
								<span class="review-value">{$agentBuilder.basicInfo.description}</span>
							</div>
						</div>
					</div>
					
					<div class="review-section">
						<h3 class="review-title">Configuration</h3>
						<div class="review-content">
							<div class="review-item">
								<span class="review-label">Model:</span>
								<span class="review-value">{$agentBuilder.configuration.model}</span>
							</div>
							<div class="review-item">
								<span class="review-label">Memory:</span>
								<span class="review-value">{memoryTypes.find(m => m.type === $agentBuilder.configuration.memory_type)?.name}</span>
							</div>
							<div class="review-item">
								<span class="review-label">Capabilities:</span>
								<div class="review-tags">
									{#each $agentBuilder.configuration.capabilities as cap}
										<span class="review-tag">{capabilities.find(c => c.id === cap)?.name}</span>
									{/each}
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Navigation -->
	<div class="navigation-container">
		<div class="nav-buttons">
			{#if currentStep > 1}
				<button class="nav-btn secondary" on:click={prevStep}>
					<ArrowLeft class="w-4 h-4" />
					<span>Previous</span>
				</button>
			{/if}
			
			<div class="nav-spacer"></div>
			
			{#if currentStep < 4}
				<button 
					class="nav-btn primary" 
					on:click={nextStep}
					disabled={!validateCurrentStep()}
				>
					<span>Next</span>
					<ArrowRight class="w-4 h-4" />
				</button>
			{:else}
				<button 
					class="nav-btn primary create-btn" 
					on:click={createAgent}
					disabled={!validateCurrentStep() || isCreating}
				>
					{#if isCreating}
						<div class="spinner"></div>
						<span>Creating...</span>
					{:else}
						<Bot class="w-4 h-4" />
						<span>Create Agent</span>
					{/if}
				</button>
			{/if}
		</div>
	</div>
</div>

<style>
	.create-agent-container {
		@apply max-w-6xl mx-auto space-y-8;
	}
	
	/* Header */
	.create-header {
		@apply space-y-4;
	}
	
	.back-btn {
		@apply inline-flex items-center space-x-2 text-dark-400 hover:text-white transition-colors duration-200;
	}
	
	.header-content {
		@apply space-y-2;
	}
	
	.page-title {
		@apply text-3xl font-bold text-white;
	}
	
	.page-subtitle {
		@apply text-lg text-dark-300;
	}
	
	/* Progress Steps */
	.progress-container {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6;
	}
	
	.progress-steps {
		@apply flex items-center justify-between;
	}
	
	.step-item {
		@apply flex items-center space-x-3 flex-1;
	}
	
	.step-item.active .step-circle {
		@apply bg-primary-600 text-white border-primary-600;
	}
	
	.step-item.completed .step-circle {
		@apply bg-green-600 text-white border-green-600;
	}
	
	.step-circle {
		@apply w-10 h-10 rounded-full border-2 border-dark-600 bg-dark-700 flex items-center justify-center text-dark-400 font-medium;
	}
	
	.step-content {
		@apply space-y-1;
	}
	
	.step-title {
		@apply text-sm font-medium text-white;
	}
	
	.step-description {
		@apply text-xs text-dark-400;
	}
	
	.step-arrow {
		@apply text-dark-600 mx-4;
	}
	
	/* Step Content */
	.step-content-container {
		@apply bg-dark-800 border border-dark-700 rounded-xl;
	}
	
	.step-panel {
		@apply p-8;
	}
	
	.panel-header {
		@apply mb-8 text-center;
	}
	
	.panel-title {
		@apply text-2xl font-bold text-white mb-2;
	}
	
	.panel-subtitle {
		@apply text-dark-300;
	}
	
	/* Forms */
	.form-grid {
		@apply grid grid-cols-1 lg:grid-cols-2 gap-6;
	}
	
	.form-group {
		@apply space-y-2;
	}
	
	.form-group.full-width {
		@apply lg:col-span-2;
	}
	
	.form-label {
		@apply block text-sm font-medium text-white;
	}
	
	.form-input,
	.form-textarea,
	.form-select {
		@apply w-full px-4 py-3 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent;
	}
	
	/* Type Grid */
	.type-grid {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4;
	}
	
	.type-card {
		@apply p-4 bg-dark-700 border border-dark-600 rounded-lg hover:border-dark-500 transition-all duration-200 text-left;
	}
	
	.type-card.selected {
		@apply border-primary-500 bg-primary-500/10;
	}
	
	.type-icon {
		@apply w-12 h-12 rounded-lg bg-dark-600 flex items-center justify-center mb-3;
	}
	
	.type-content {
		@apply space-y-2;
	}
	
	.type-name {
		@apply text-lg font-semibold text-white;
	}
	
	.type-description {
		@apply text-sm text-dark-300;
	}
	
	.type-features {
		@apply flex flex-wrap gap-1;
	}
	
	.feature-tag {
		@apply px-2 py-1 text-xs bg-dark-600 text-dark-300 rounded;
	}
	
	/* Template Grid */
	.template-grid {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4;
	}
	
	.template-card {
		@apply relative p-6 bg-dark-700 border border-dark-600 rounded-lg hover:border-dark-500 transition-all duration-200 text-left;
	}
	
	.template-card.selected {
		@apply border-primary-500 bg-primary-500/10;
	}
	
	.template-icon {
		@apply mb-4;
	}
	
	.template-emoji {
		@apply text-3xl;
	}
	
	.template-content {
		@apply space-y-2;
	}
	
	.template-name {
		@apply text-lg font-semibold text-white;
	}
	
	.template-description {
		@apply text-sm text-dark-300;
	}
	
	.selected-indicator {
		@apply absolute top-4 right-4 w-6 h-6 bg-primary-600 text-white rounded-full flex items-center justify-center;
	}
	
	/* Configuration */
	.config-sections {
		@apply space-y-8;
	}
	
	.config-section {
		@apply space-y-4;
	}
	
	.section-title {
		@apply text-lg font-semibold text-white;
	}
	
	.capabilities-grid {
		@apply grid grid-cols-1 md:grid-cols-2 gap-3;
	}
	
	.capability-card {
		@apply relative flex items-center justify-between p-4 bg-dark-700 border border-dark-600 rounded-lg hover:border-dark-500 transition-all duration-200;
	}
	
	.capability-card.selected {
		@apply border-primary-500 bg-primary-500/10;
	}
	
	.capability-content {
		@apply space-y-1;
	}
	
	.capability-name {
		@apply text-sm font-medium text-white;
	}
	
	.capability-description {
		@apply text-xs text-dark-400;
	}
	
	.capability-check {
		@apply w-5 h-5 bg-primary-600 text-white rounded flex items-center justify-center;
	}
	
	.memory-grid {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3;
	}
	
	.memory-card {
		@apply p-4 bg-dark-700 border border-dark-600 rounded-lg hover:border-dark-500 transition-all duration-200 text-left;
	}
	
	.memory-card.selected {
		@apply border-primary-500 bg-primary-500/10;
	}
	
	.memory-name {
		@apply text-sm font-medium text-white mb-1;
	}
	
	.memory-description {
		@apply text-xs text-dark-400;
	}
	
	/* Review */
	.review-sections {
		@apply space-y-6;
	}
	
	.review-section {
		@apply bg-dark-700 rounded-lg p-6;
	}
	
	.review-title {
		@apply text-lg font-semibold text-white mb-4;
	}
	
	.review-content {
		@apply space-y-3;
	}
	
	.review-item {
		@apply flex items-start space-x-3;
	}
	
	.review-label {
		@apply text-sm font-medium text-dark-400 min-w-[100px];
	}
	
	.review-value {
		@apply text-sm text-white;
	}
	
	.review-tags {
		@apply flex flex-wrap gap-2;
	}
	
	.review-tag {
		@apply px-2 py-1 text-xs bg-primary-500/20 text-primary-400 rounded;
	}
	
	/* Navigation */
	.navigation-container {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6;
	}
	
	.nav-buttons {
		@apply flex items-center;
	}
	
	.nav-spacer {
		@apply flex-1;
	}
	
	.nav-btn {
		@apply inline-flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200;
	}
	
	.nav-btn.primary {
		@apply bg-primary-600 text-white hover:bg-primary-700;
	}
	
	.nav-btn.secondary {
		@apply bg-dark-700 text-dark-300 hover:bg-dark-600 hover:text-white;
	}
	
	.nav-btn:disabled {
		@apply opacity-50 cursor-not-allowed;
	}
	
	.spinner {
		@apply w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin;
	}
	
	/* Mobile Responsiveness */
	@media (max-width: 768px) {
		.progress-steps {
			@apply flex-col space-y-4;
		}
		
		.step-item {
			@apply flex-col text-center;
		}
		
		.step-arrow {
			@apply hidden;
		}
		
		.form-grid {
			@apply grid-cols-1;
		}
		
		.type-grid,
		.template-grid {
			@apply grid-cols-1;
		}
		
		.capabilities-grid,
		.memory-grid {
			@apply grid-cols-1;
		}
	}
</style>
