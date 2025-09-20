<!--
Revolutionary Visual Agent Builder - Enhanced Visual Agent Creation
Complete visual workflow builder with template decomposition and node editing
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { 
		ArrowLeft, Play, Save, Eye, Grid, Palette, 
		Monitor, Layers, Workflow, Sparkles, Settings,
		Bot, Brain, Zap, FileText, Database, Globe
	} from 'lucide-svelte';
	import type { AgentTemplate } from '$lib/types';
	import type { AgentWorkflow } from '$lib/types/nodes';
	import { notificationActions } from '$lib/stores';
	import { apiClient } from '$lib/services/api';
	import { workflowExecutor } from '$lib/services/workflowExecutor';
	
	// Visual Components
	import NodePalette from '$lib/components/nodes/NodePalette.svelte';
	import VisualCanvas from '$lib/components/nodes/VisualCanvas.svelte';
	import { TemplateDecomposer } from '$lib/components/nodes/TemplateDecomposer';

	// State
	let currentWorkflow: AgentWorkflow | null = null;
	let showNodePalette = true;
	let isExecuting = false;
	let executionResults: any = null;
	let selectedTemplate: AgentTemplate | null = null;
	let showTemplateSelector = true;

	// Templates with enhanced metadata
	const templates = [
		{
			template: 'research_assistant' as AgentTemplate,
			name: 'Research Assistant',
			description: 'Advanced research workflow with web search, knowledge base integration, and synthesis',
			icon: '🔬',
			color: 'bg-blue-500',
			complexity: 'Complex',
			nodeCount: 7,
			estimatedTime: '45s',
			features: ['Web Search', 'Knowledge Base', 'Memory', 'Synthesis']
		},
		{
			template: 'customer_support' as AgentTemplate,
			name: 'Customer Support',
			description: 'Intelligent customer support with intent classification and knowledge base integration',
			icon: '🎧',
			color: 'bg-green-500',
			complexity: 'Medium',
			nodeCount: 6,
			estimatedTime: '15s',
			features: ['Intent Classification', 'Knowledge Base', 'Response Generation']
		},
		{
			template: 'data_analyst' as AgentTemplate,
			name: 'Data Analyst',
			description: 'Comprehensive data analysis with visualization generation',
			icon: '📊',
			color: 'bg-purple-500',
			complexity: 'Medium',
			nodeCount: 5,
			estimatedTime: '30s',
			features: ['Data Processing', 'Analysis', 'Visualization']
		},
		{
			template: 'creative_writer' as AgentTemplate,
			name: 'Creative Writer',
			description: 'Creative content generation with style enhancement',
			icon: '✍️',
			color: 'bg-pink-500',
			complexity: 'Simple',
			nodeCount: 4,
			estimatedTime: '20s',
			features: ['Creative Writing', 'Style Enhancement']
		},
		{
			template: 'code_reviewer' as AgentTemplate,
			name: 'Code Reviewer',
			description: 'Comprehensive code review with security and performance analysis',
			icon: '💻',
			color: 'bg-indigo-500',
			complexity: 'Complex',
			nodeCount: 6,
			estimatedTime: '35s',
			features: ['Code Analysis', 'Security Scan', 'Performance Check']
		},
		{
			template: 'custom' as AgentTemplate,
			name: 'Custom Agent',
			description: 'Blank canvas for building custom agent workflows',
			icon: '🛠️',
			color: 'bg-gray-500',
			complexity: 'Simple',
			nodeCount: 2,
			estimatedTime: '5s',
			features: ['Blank Canvas', 'Full Customization']
		}
	];

	onMount(() => {
		// Initialize with custom template by default
		selectTemplate('custom');
	});

	// Template selection
	function selectTemplate(templateName: AgentTemplate) {
		selectedTemplate = templateName;
		
		// Create workflow from template
		currentWorkflow = TemplateDecomposer.createWorkflowFromTemplate(templateName);
		
		// Hide template selector and show visual builder
		showTemplateSelector = false;
		
		notificationActions.add({
			type: 'success',
			title: 'Template Loaded',
			message: `${templates.find(t => t.template === templateName)?.name} template loaded successfully`
		});
	}

	// Workflow management
	function handleWorkflowChange(event: CustomEvent<{ workflow: AgentWorkflow }>) {
		currentWorkflow = event.detail.workflow;
	}

	// Execute workflow
	async function executeWorkflow() {
		if (!currentWorkflow) return;
		
		isExecuting = true;
		executionResults = null;
		
		try {
			// Get input from user (simplified for demo)
			const inputs = {
				input_query: 'Test input for workflow execution'
			};
			
			const execution = await workflowExecutor.executeWorkflow(currentWorkflow, inputs);
			executionResults = execution;
			
			notificationActions.add({
				type: execution.status === 'completed' ? 'success' : 'error',
				title: 'Workflow Execution',
				message: execution.status === 'completed' 
					? `Workflow completed in ${execution.metrics.executionTime}ms`
					: `Workflow failed: ${execution.errors[0]?.message || 'Unknown error'}`
			});
		} catch (error) {
			console.error('Workflow execution failed:', error);
			notificationActions.add({
				type: 'error',
				title: 'Execution Failed',
				message: error instanceof Error ? error.message : 'Unknown error occurred'
			});
		} finally {
			isExecuting = false;
		}
	}

	// Save workflow as agent
	async function saveAsAgent() {
		if (!currentWorkflow) return;
		
		try {
			const agentData = {
				name: currentWorkflow.name,
				description: currentWorkflow.description,
				type: 'workflow',
				template: currentWorkflow.template,
				workflow_definition: {
					nodes: currentWorkflow.nodes,
					connections: currentWorkflow.connections
				},
				is_public: false
			};
			
			const response = await apiClient.createAgent(agentData);
			
			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Agent Created',
					message: `${agentData.name} has been saved as an agent`
				});
				
				goto('/agents');
			} else {
				throw new Error(response.error || 'Failed to create agent');
			}
		} catch (error) {
			console.error('Failed to save agent:', error);
			notificationActions.add({
				type: 'error',
				title: 'Save Failed',
				message: error instanceof Error ? error.message : 'Unknown error occurred'
			});
		}
	}

	// Back to template selector
	function backToTemplates() {
		showTemplateSelector = true;
		currentWorkflow = null;
		selectedTemplate = null;
	}

	// Get complexity color
	function getComplexityColor(complexity: string) {
		switch (complexity) {
			case 'Simple': return 'text-green-600 bg-green-100';
			case 'Medium': return 'text-yellow-600 bg-yellow-100';
			case 'Complex': return 'text-red-600 bg-red-100';
			default: return 'text-gray-600 bg-gray-100';
		}
	}
</script>

<div class="visual-builder-container">
	<!-- Header -->
	<div class="builder-header">
		<div class="header-left">
			<button
				class="back-btn"
				on:click={() => goto('/agents')}
				title="Back to Agents"
			>
				<ArrowLeft class="w-4 h-4" />
			</button>
			
			<div class="header-info">
				<h1 class="page-title">
					{#if showTemplateSelector}
						Visual Agent Builder
					{:else}
						{currentWorkflow?.name || 'Untitled Workflow'}
					{/if}
				</h1>
				<p class="page-subtitle">
					{#if showTemplateSelector}
						Choose a template to start building your agent visually
					{:else}
						Drag and drop nodes to build your agent workflow
					{/if}
				</p>
			</div>
		</div>

		{#if !showTemplateSelector}
			<div class="header-actions">
				<button
					class="action-btn secondary"
					on:click={backToTemplates}
					title="Change Template"
				>
					<Layers class="w-4 h-4" />
					Templates
				</button>
				
				<button
					class="action-btn secondary"
					on:click={() => showNodePalette = !showNodePalette}
					title="Toggle Node Palette"
				>
					<Palette class="w-4 h-4" />
					{showNodePalette ? 'Hide' : 'Show'} Palette
				</button>
				
				<button
					class="action-btn"
					on:click={executeWorkflow}
					disabled={isExecuting || !currentWorkflow}
					title="Test Workflow"
				>
					{#if isExecuting}
						<div class="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
					{:else}
						<Play class="w-4 h-4" />
					{/if}
					Test
				</button>
				
				<button
					class="action-btn primary"
					on:click={saveAsAgent}
					disabled={!currentWorkflow}
					title="Save as Agent"
				>
					<Save class="w-4 h-4" />
					Save Agent
				</button>
			</div>
		{/if}
	</div>

	<!-- Template Selector -->
	{#if showTemplateSelector}
		<div class="template-selector">
			<div class="selector-header">
				<h2 class="selector-title">Choose Your Agent Template</h2>
				<p class="selector-subtitle">
					Each template provides a pre-built workflow that you can customize and extend
				</p>
			</div>

			<div class="templates-grid">
				{#each templates as template}
					<button
						class="template-card"
						on:click={() => selectTemplate(template.template)}
					>
						<div class="template-header">
							<div class="template-icon {template.color}">
								<span class="text-2xl">{template.icon}</span>
							</div>
							
							<div class="template-meta">
								<span class="complexity-badge {getComplexityColor(template.complexity)}">
									{template.complexity}
								</span>
								<span class="node-count">{template.nodeCount} nodes</span>
							</div>
						</div>

						<div class="template-content">
							<h3 class="template-name">{template.name}</h3>
							<p class="template-description">{template.description}</p>
							
							<div class="template-features">
								{#each template.features as feature}
									<span class="feature-tag">{feature}</span>
								{/each}
							</div>
							
							<div class="template-stats">
								<span class="stat">
									<Monitor class="w-3 h-3" />
									~{template.estimatedTime}
								</span>
							</div>
						</div>
					</button>
				{/each}
			</div>
		</div>
	{:else}
		<!-- Visual Builder -->
		<div class="visual-builder">
			<!-- Node Palette -->
			{#if showNodePalette}
				<div class="palette-container">
					<NodePalette
						on:dragStart={(e) => console.log('Drag start:', e.detail)}
						on:addNode={(e) => console.log('Add node:', e.detail)}
					/>
				</div>
			{/if}

			<!-- Canvas Container -->
			<div class="canvas-container" class:full-width={!showNodePalette}>
				{#if currentWorkflow}
					<VisualCanvas
						workflow={currentWorkflow}
						on:workflowChange={handleWorkflowChange}
						on:execute={executeWorkflow}
						on:save={saveAsAgent}
					/>
				{/if}
			</div>
		</div>
	{/if}

	<!-- Execution Results Panel -->
	{#if executionResults}
		<div class="results-panel">
			<div class="results-header">
				<h3 class="results-title">Execution Results</h3>
				<button
					class="close-results"
					on:click={() => executionResults = null}
				>
					×
				</button>
			</div>
			
			<div class="results-content">
				<div class="result-status {executionResults.status}">
					Status: {executionResults.status}
				</div>
				
				<div class="result-metrics">
					<span>Nodes: {executionResults.metrics.completedNodes}/{executionResults.metrics.totalNodes}</span>
					<span>Time: {executionResults.metrics.executionTime}ms</span>
				</div>
				
				{#if executionResults.errors.length > 0}
					<div class="result-errors">
						<h4>Errors:</h4>
						{#each executionResults.errors as error}
							<div class="error-item">
								<strong>{error.nodeId}:</strong> {error.message}
							</div>
						{/each}
					</div>
				{/if}
				
				{#if Object.keys(executionResults.results).length > 0}
					<div class="result-data">
						<h4>Results:</h4>
						<pre>{JSON.stringify(executionResults.results, null, 2)}</pre>
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.visual-builder-container {
		@apply h-screen flex flex-col bg-gray-50;
	}

	.builder-header {
		@apply flex items-center justify-between p-4 bg-white border-b border-gray-200 shadow-sm;
	}

	.header-left {
		@apply flex items-center gap-4;
	}

	.back-btn {
		@apply p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200;
	}

	.header-info {
		@apply min-w-0;
	}

	.page-title {
		@apply text-xl font-bold text-gray-900 truncate;
	}

	.page-subtitle {
		@apply text-sm text-gray-600 truncate;
	}

	.header-actions {
		@apply flex items-center gap-2;
	}

	.action-btn {
		@apply flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors duration-200;
		@apply disabled:opacity-50 disabled:cursor-not-allowed;
	}

	.action-btn.secondary {
		@apply bg-gray-100 text-gray-700 hover:bg-gray-200;
	}

	.action-btn.primary {
		@apply bg-blue-600 text-white hover:bg-blue-700;
	}

	.action-btn:not(.secondary):not(.primary) {
		@apply bg-white text-gray-700 border border-gray-300 hover:bg-gray-50;
	}

	.template-selector {
		@apply flex-1 p-8 overflow-y-auto;
	}

	.selector-header {
		@apply text-center mb-8;
	}

	.selector-title {
		@apply text-2xl font-bold text-gray-900 mb-2;
	}

	.selector-subtitle {
		@apply text-gray-600 max-w-2xl mx-auto;
	}

	.templates-grid {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto;
	}

	.template-card {
		@apply bg-white rounded-xl border border-gray-200 p-6 hover:shadow-lg hover:border-blue-300 transition-all duration-200;
		@apply text-left;
	}

	.template-header {
		@apply flex items-start justify-between mb-4;
	}

	.template-icon {
		@apply w-12 h-12 rounded-lg flex items-center justify-center;
	}

	.template-meta {
		@apply flex flex-col items-end gap-1;
	}

	.complexity-badge {
		@apply px-2 py-1 text-xs font-medium rounded-full;
	}

	.node-count {
		@apply text-xs text-gray-500;
	}

	.template-content {
		@apply space-y-3;
	}

	.template-name {
		@apply font-semibold text-gray-900;
	}

	.template-description {
		@apply text-sm text-gray-600 leading-relaxed;
	}

	.template-features {
		@apply flex flex-wrap gap-1;
	}

	.feature-tag {
		@apply px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded-md;
	}

	.template-stats {
		@apply flex items-center gap-4 pt-2 border-t border-gray-100;
	}

	.stat {
		@apply flex items-center gap-1 text-xs text-gray-500;
	}

	.visual-builder {
		@apply flex-1 flex overflow-hidden;
	}

	.palette-container {
		@apply flex-shrink-0;
	}

	.canvas-container {
		@apply flex-1 relative;
	}

	.canvas-container.full-width {
		@apply w-full;
	}

	.results-panel {
		@apply fixed bottom-4 right-4 w-96 bg-white rounded-lg shadow-xl border border-gray-200 z-50;
		@apply max-h-96 overflow-hidden;
	}

	.results-header {
		@apply flex items-center justify-between p-4 border-b border-gray-200;
	}

	.results-title {
		@apply font-semibold text-gray-900;
	}

	.close-results {
		@apply w-6 h-6 flex items-center justify-center rounded hover:bg-gray-100;
	}

	.results-content {
		@apply p-4 space-y-3 overflow-y-auto max-h-80;
	}

	.result-status {
		@apply px-3 py-1 rounded-full text-sm font-medium;
	}

	.result-status.completed {
		@apply bg-green-100 text-green-800;
	}

	.result-status.failed {
		@apply bg-red-100 text-red-800;
	}

	.result-status.running {
		@apply bg-blue-100 text-blue-800;
	}

	.result-metrics {
		@apply flex gap-4 text-sm text-gray-600;
	}

	.result-errors {
		@apply space-y-2;
	}

	.error-item {
		@apply text-sm text-red-600 bg-red-50 p-2 rounded;
	}

	.result-data {
		@apply space-y-2;
	}

	.result-data pre {
		@apply text-xs bg-gray-100 p-2 rounded overflow-x-auto;
	}
</style>
