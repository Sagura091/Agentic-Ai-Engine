<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { goto } from '$app/navigation';
	import {
		workflowBuilder,
		agents,
		notificationActions
	} from '$stores';
	import { apiClient } from '$services/api';

	// Extract individual stores from workflowBuilder
	const { nodes, edges, selectedNode } = workflowBuilder;
	import { 
		ArrowLeft, 
		Save, 
		Play, 
		Plus, 
		Trash2, 
		Settings,
		Network,
		Bot,
		Zap,
		GitBranch,
		MessageSquare,
		Clock,
		CheckCircle,
		AlertTriangle
	} from 'lucide-svelte';
	import type { Agent } from '$types';
	
	// Workflow Builder State
	let workflowName = '';
	let workflowDescription = '';
	let canvasContainer: HTMLElement;
	let isDragging = false;
	let draggedNodeType = '';
	let canvasOffset = { x: 0, y: 0 };
	let zoom = 1;
	let isSaving = false;
	
	// Node Types
	const nodeTypes = [
		{
			type: 'agent',
			name: 'Agent Node',
			icon: Bot,
			color: 'bg-blue-500',
			description: 'Execute an AI agent'
		},
		{
			type: 'condition',
			name: 'Condition',
			icon: GitBranch,
			color: 'bg-yellow-500',
			description: 'Branch based on conditions'
		},
		{
			type: 'input',
			name: 'Input',
			icon: MessageSquare,
			color: 'bg-green-500',
			description: 'Receive user input'
		},
		{
			type: 'output',
			name: 'Output',
			icon: Zap,
			color: 'bg-purple-500',
			description: 'Send output to user'
		},
		{
			type: 'delay',
			name: 'Delay',
			icon: Clock,
			color: 'bg-orange-500',
			description: 'Add time delay'
		},
		{
			type: 'webhook',
			name: 'Webhook',
			icon: Network,
			color: 'bg-red-500',
			description: 'Call external API'
		}
	];
	
	// Connection state
	let isConnecting = false;
	let connectionStart: { nodeId: string; port: string } | null = null;
	let tempConnection: { x1: number; y1: number; x2: number; y2: number } | null = null;
	
	onMount(async () => {
		// Load agents if not already loaded
		if ($agents.length === 0) {
			await loadAgents();
		}
		
		// Initialize canvas
		setupCanvas();
		
		// Reset workflow builder
		workflowBuilder.actions.reset();
	});
	
	onDestroy(() => {
		// Clean up any event listeners
		document.removeEventListener('mousemove', handleMouseMove);
		document.removeEventListener('mouseup', handleMouseUp);
	});
	
	async function loadAgents() {
		try {
			const response = await apiClient.getAgents();
			if (response.success && response.data) {
				agents.set(response.data);
			}
		} catch (error) {
			console.error('Failed to load agents:', error);
		}
	}
	
	function setupCanvas() {
		if (canvasContainer) {
			canvasContainer.addEventListener('drop', handleDrop);
			canvasContainer.addEventListener('dragover', handleDragOver);
		}
	}
	
	function handleDragStart(event: DragEvent, nodeType: string) {
		if (event.dataTransfer) {
			event.dataTransfer.setData('text/plain', nodeType);
			draggedNodeType = nodeType;
			isDragging = true;
		}
	}
	
	function handleDragOver(event: DragEvent) {
		event.preventDefault();
	}
	
	function handleDrop(event: DragEvent) {
		event.preventDefault();
		
		if (!draggedNodeType) return;
		
		const rect = canvasContainer.getBoundingClientRect();
		const x = (event.clientX - rect.left - canvasOffset.x) / zoom;
		const y = (event.clientY - rect.top - canvasOffset.y) / zoom;
		
		const newNode = {
			id: crypto.randomUUID(),
			type: draggedNodeType,
			position: { x, y },
			data: {
				label: getNodeTypeInfo(draggedNodeType)?.name || draggedNodeType,
				config: {}
			}
		};
		
		workflowBuilder.actions.addNode(newNode);
		
		isDragging = false;
		draggedNodeType = '';
		
		notificationActions.add({
			type: 'success',
			title: 'Node Added',
			message: `${newNode.data.label} node added to workflow`
		});
	}
	
	function getNodeTypeInfo(type: string) {
		return nodeTypes.find(nt => nt.type === type);
	}
	
	function selectNode(nodeId: string) {
		selectedNode.set(nodeId);
	}
	
	function deleteNode(nodeId: string) {
		workflowBuilder.actions.removeNode(nodeId);
		notificationActions.add({
			type: 'info',
			title: 'Node Deleted',
			message: 'Node removed from workflow'
		});
	}
	
	function startConnection(nodeId: string, port: string, event: MouseEvent) {
		event.stopPropagation();
		isConnecting = true;
		connectionStart = { nodeId, port };
		
		document.addEventListener('mousemove', handleMouseMove);
		document.addEventListener('mouseup', handleMouseUp);
	}
	
	function handleMouseMove(event: MouseEvent) {
		if (isConnecting && connectionStart && canvasContainer) {
			const rect = canvasContainer.getBoundingClientRect();
			const x2 = event.clientX - rect.left;
			const y2 = event.clientY - rect.top;
			
			// Find the start position of the connection
			const startNode = $nodes.find(n => n.id === connectionStart!.nodeId);
			if (startNode) {
				const x1 = startNode.position.x * zoom + canvasOffset.x + 100; // Approximate node center
				const y1 = startNode.position.y * zoom + canvasOffset.y + 50;
				
				tempConnection = { x1, y1, x2, y2 };
			}
		}
	}
	
	function handleMouseUp(event: MouseEvent) {
		if (isConnecting) {
			// Check if we're over a valid connection target
			const target = event.target as HTMLElement;
			const targetNodeId = target.closest('[data-node-id]')?.getAttribute('data-node-id');
			const targetPort = target.closest('[data-port]')?.getAttribute('data-port');
			
			if (targetNodeId && targetPort && connectionStart && targetNodeId !== connectionStart.nodeId) {
				// Create connection
				const newEdge = {
					id: crypto.randomUUID(),
					source: connectionStart.nodeId,
					target: targetNodeId,
					sourceHandle: connectionStart.port,
					targetHandle: targetPort
				};
				
				workflowBuilder.actions.addEdge(newEdge);
				
				notificationActions.add({
					type: 'success',
					title: 'Connection Created',
					message: 'Nodes connected successfully'
				});
			}
			
			// Clean up
			isConnecting = false;
			connectionStart = null;
			tempConnection = null;
			
			document.removeEventListener('mousemove', handleMouseMove);
			document.removeEventListener('mouseup', handleMouseUp);
		}
	}
	
	function deleteConnection(edgeId: string) {
		workflowBuilder.actions.removeEdge(edgeId);
		notificationActions.add({
			type: 'info',
			title: 'Connection Deleted',
			message: 'Connection removed from workflow'
		});
	}
	
	async function saveWorkflow() {
		if (!workflowName.trim()) {
			notificationActions.add({
				type: 'error',
				title: 'Validation Error',
				message: 'Please enter a workflow name'
			});
			return;
		}
		
		if ($nodes.length === 0) {
			notificationActions.add({
				type: 'error',
				title: 'Validation Error',
				message: 'Please add at least one node to the workflow'
			});
			return;
		}

		isSaving = true;

		try {
			const workflowData = {
				name: workflowName,
				description: workflowDescription,
				nodes: $nodes,
				edges: $edges,
				status: 'draft'
			};
			
			const response = await apiClient.createWorkflow(workflowData);
			
			if (response.success) {
				notificationActions.add({
					type: 'success',
					title: 'Workflow Saved',
					message: `${workflowName} has been saved successfully`
				});
				
				// Reset builder and navigate back
				workflowBuilder.actions.reset();
				goto('/workflows');
			} else {
				throw new Error(response.error || 'Failed to save workflow');
			}
		} catch (error) {
			console.error('Failed to save workflow:', error);
			notificationActions.add({
				type: 'error',
				title: 'Save Failed',
				message: error instanceof Error ? error.message : 'Unknown error occurred'
			});
		} finally {
			isSaving = false;
		}
	}
	
	async function testWorkflow() {
		if ($nodes.length === 0) {
			notificationActions.add({
				type: 'error',
				title: 'Cannot Test',
				message: 'Please add nodes to the workflow before testing'
			});
			return;
		}
		
		notificationActions.add({
			type: 'info',
			title: 'Test Started',
			message: 'Testing workflow execution...'
		});
		
		// TODO: Implement workflow testing
		setTimeout(() => {
			notificationActions.add({
				type: 'success',
				title: 'Test Completed',
				message: 'Workflow test completed successfully'
			});
		}, 2000);
	}
	
	function zoomIn() {
		zoom = Math.min(zoom * 1.2, 3);
	}
	
	function zoomOut() {
		zoom = Math.max(zoom / 1.2, 0.3);
	}
	
	function resetZoom() {
		zoom = 1;
		canvasOffset = { x: 0, y: 0 };
	}
</script>

<svelte:head>
	<title>Create Workflow - Agentic AI</title>
</svelte:head>

<div class="workflow-builder">
	<!-- Header -->
	<div class="builder-header">
		<div class="header-left">
			<button class="back-btn" on:click={() => goto('/workflows')}>
				<ArrowLeft class="w-5 h-5" />
				<span>Back to Workflows</span>
			</button>
			
			<div class="workflow-info">
				<input
					type="text"
					placeholder="Enter workflow name..."
					class="workflow-name-input"
					bind:value={workflowName}
				/>
				<input
					type="text"
					placeholder="Enter description..."
					class="workflow-description-input"
					bind:value={workflowDescription}
				/>
			</div>
		</div>
		
		<div class="header-actions">
			<button class="action-btn secondary" on:click={testWorkflow}>
				<Play class="w-4 h-4" />
				<span>Test</span>
			</button>
			
			<button 
				class="action-btn primary" 
				on:click={saveWorkflow}
				disabled={isSaving}
			>
				{#if isSaving}
					<div class="spinner"></div>
					<span>Saving...</span>
				{:else}
					<Save class="w-4 h-4" />
					<span>Save Workflow</span>
				{/if}
			</button>
		</div>
	</div>
	
	<!-- Main Builder Interface -->
	<div class="builder-interface">
		<!-- Node Palette -->
		<div class="node-palette">
			<h3 class="palette-title">Node Types</h3>
			<div class="node-types">
				{#each nodeTypes as nodeType}
					<div
						class="node-type-item"
						draggable="true"
						on:dragstart={(e) => handleDragStart(e, nodeType.type)}
					>
						<div class="node-type-icon {nodeType.color}">
							<svelte:component this={nodeType.icon} class="w-5 h-5 text-white" />
						</div>
						<div class="node-type-info">
							<h4 class="node-type-name">{nodeType.name}</h4>
							<p class="node-type-description">{nodeType.description}</p>
						</div>
					</div>
				{/each}
			</div>
		</div>
		
		<!-- Canvas -->
		<div class="canvas-container">
			<!-- Canvas Controls -->
			<div class="canvas-controls">
				<button class="control-btn" on:click={zoomIn}>+</button>
				<span class="zoom-level">{Math.round(zoom * 100)}%</span>
				<button class="control-btn" on:click={zoomOut}>-</button>
				<button class="control-btn" on:click={resetZoom}>Reset</button>
			</div>
			
			<!-- Canvas -->
			<div 
				class="workflow-canvas"
				bind:this={canvasContainer}
				style="transform: scale({zoom}) translate({canvasOffset.x}px, {canvasOffset.y}px)"
			>
				<!-- Grid Background -->
				<div class="canvas-grid"></div>
				
				<!-- Connections/Edges -->
				<svg class="connections-layer">
					{#each $edges as edge (edge.id)}
						{@const sourceNode = $nodes.find(n => n.id === edge.source)}
						{@const targetNode = $nodes.find(n => n.id === edge.target)}
						{#if sourceNode && targetNode}
							<line
								x1={sourceNode.position.x + 200}
								y1={sourceNode.position.y + 50}
								x2={targetNode.position.x}
								y2={targetNode.position.y + 50}
								stroke="#6366f1"
								stroke-width="2"
								marker-end="url(#arrowhead)"
								class="connection-line"
								on:click={() => deleteConnection(edge.id)}
							/>
						{/if}
					{/each}
					
					<!-- Temporary connection while dragging -->
					{#if tempConnection}
						<line
							x1={tempConnection.x1}
							y1={tempConnection.y1}
							x2={tempConnection.x2}
							y2={tempConnection.y2}
							stroke="#6366f1"
							stroke-width="2"
							stroke-dasharray="5,5"
							class="temp-connection"
						/>
					{/if}
					
					<!-- Arrow marker definition -->
					<defs>
						<marker
							id="arrowhead"
							markerWidth="10"
							markerHeight="7"
							refX="9"
							refY="3.5"
							orient="auto"
						>
							<polygon
								points="0 0, 10 3.5, 0 7"
								fill="#6366f1"
							/>
						</marker>
					</defs>
				</svg>
				
				<!-- Nodes -->
				{#each $nodes as node (node.id)}
					{@const nodeTypeInfo = getNodeTypeInfo(node.type)}
					<div
						class="workflow-node {$selectedNode === node.id ? 'selected' : ''}"
						style="left: {node.position.x}px; top: {node.position.y}px"
						data-node-id={node.id}
						on:click={() => selectNode(node.id)}
					>
						<!-- Node Header -->
						<div class="node-header {nodeTypeInfo?.color || 'bg-gray-500'}">
							<div class="node-icon">
								{#if nodeTypeInfo?.icon}
									<svelte:component this={nodeTypeInfo.icon} class="w-4 h-4 text-white" />
								{/if}
							</div>
							<span class="node-title">{node.data.label}</span>
							<button 
								class="node-delete"
								on:click|stopPropagation={() => deleteNode(node.id)}
							>
								<Trash2 class="w-3 h-3" />
							</button>
						</div>
						
						<!-- Node Content -->
						<div class="node-content">
							{#if node.type === 'agent'}
								<select class="node-select">
									<option value="">Select Agent...</option>
									{#each $agents as agent}
										<option value={agent.id}>{agent.name}</option>
									{/each}
								</select>
							{:else if node.type === 'condition'}
								<input 
									type="text" 
									placeholder="Condition expression..."
									class="node-input"
								/>
							{:else if node.type === 'delay'}
								<input 
									type="number" 
									placeholder="Delay (seconds)"
									class="node-input"
								/>
							{:else}
								<div class="node-placeholder">
									Configure {nodeTypeInfo?.name || 'Node'}
								</div>
							{/if}
						</div>
						
						<!-- Connection Ports -->
						<div class="connection-ports">
							<div 
								class="input-port"
								data-port="input"
								on:mousedown={(e) => startConnection(node.id, 'input', e)}
							></div>
							<div 
								class="output-port"
								data-port="output"
								on:mousedown={(e) => startConnection(node.id, 'output', e)}
							></div>
						</div>
					</div>
				{/each}
				
				<!-- Drop Zone Hint -->
				{#if isDragging}
					<div class="drop-zone-hint">
						Drop node here to add to workflow
					</div>
				{/if}
			</div>
		</div>
		
		<!-- Properties Panel -->
		<div class="properties-panel">
			<h3 class="panel-title">Properties</h3>
			
			{#if $selectedNode}
				{#each $nodes as node}
					{#if node.id === $selectedNode}
						<div class="property-section">
							<h4 class="property-title">Node Configuration</h4>
							<div class="property-form">
								<div class="form-group">
									<label class="form-label">Label</label>
									<input
										type="text"
										class="form-input"
										bind:value={node.data.label}
									/>
								</div>

							{#if node.type === 'agent'}
								<div class="form-group">
									<label class="form-label">Agent</label>
									<select class="form-select">
										<option value="">Select Agent...</option>
										{#each $agents as agent}
											<option value={agent.id}>{agent.name}</option>
										{/each}
									</select>
								</div>
							{/if}
						</div>
					</div>
					{/if}
				{/each}
			{:else}
				<div class="no-selection">
					<Network class="w-12 h-12 text-dark-500 mb-4" />
					<p class="text-dark-400">Select a node to configure its properties</p>
				</div>
			{/if}
			
			<!-- Workflow Stats -->
			<div class="workflow-stats">
				<h4 class="stats-title">Workflow Stats</h4>
				<div class="stats-grid">
					<div class="stat-item">
						<span class="stat-value">{$nodes.length}</span>
						<span class="stat-label">Nodes</span>
					</div>
					<div class="stat-item">
						<span class="stat-value">{$edges.length}</span>
						<span class="stat-label">Connections</span>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	.workflow-builder {
		@apply h-screen flex flex-col bg-dark-900;
	}
	
	/* Header */
	.builder-header {
		@apply flex items-center justify-between p-4 bg-dark-800 border-b border-dark-700;
	}
	
	.header-left {
		@apply flex items-center space-x-4;
	}
	
	.back-btn {
		@apply inline-flex items-center space-x-2 text-dark-400 hover:text-white transition-colors duration-200;
	}
	
	.workflow-info {
		@apply space-y-2;
	}
	
	.workflow-name-input,
	.workflow-description-input {
		@apply block px-3 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500;
	}
	
	.workflow-name-input {
		@apply text-lg font-semibold;
	}
	
	.workflow-description-input {
		@apply text-sm;
	}
	
	.header-actions {
		@apply flex items-center space-x-3;
	}
	
	.action-btn {
		@apply inline-flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200;
	}
	
	.action-btn.primary {
		@apply bg-primary-600 text-white hover:bg-primary-700;
	}
	
	.action-btn.secondary {
		@apply bg-dark-700 text-dark-300 hover:bg-dark-600 hover:text-white;
	}
	
	.action-btn:disabled {
		@apply opacity-50 cursor-not-allowed;
	}
	
	.spinner {
		@apply w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin;
	}
	
	/* Builder Interface */
	.builder-interface {
		@apply flex-1 flex overflow-hidden;
	}
	
	/* Node Palette */
	.node-palette {
		@apply w-64 bg-dark-800 border-r border-dark-700 p-4 overflow-y-auto;
	}
	
	.palette-title {
		@apply text-lg font-semibold text-white mb-4;
	}
	
	.node-types {
		@apply space-y-2;
	}
	
	.node-type-item {
		@apply flex items-center space-x-3 p-3 bg-dark-700 rounded-lg cursor-move hover:bg-dark-600 transition-colors duration-200;
	}
	
	.node-type-icon {
		@apply w-10 h-10 rounded-lg flex items-center justify-center;
	}
	
	.node-type-info {
		@apply flex-1;
	}
	
	.node-type-name {
		@apply text-sm font-medium text-white;
	}
	
	.node-type-description {
		@apply text-xs text-dark-400;
	}
	
	/* Canvas */
	.canvas-container {
		@apply flex-1 relative overflow-hidden bg-dark-900;
	}
	
	.canvas-controls {
		@apply absolute top-4 right-4 flex items-center space-x-2 bg-dark-800 border border-dark-700 rounded-lg p-2 z-10;
	}
	
	.control-btn {
		@apply w-8 h-8 bg-dark-700 text-white rounded hover:bg-dark-600 transition-colors duration-200;
	}
	
	.zoom-level {
		@apply text-sm text-dark-300 px-2;
	}
	
	.workflow-canvas {
		@apply relative w-full h-full transform-gpu;
		transform-origin: top left;
	}
	
	.canvas-grid {
		@apply absolute inset-0 opacity-20;
		background-image: 
			linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
			linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
		background-size: 20px 20px;
	}
	
	.connections-layer {
		@apply absolute inset-0 pointer-events-none;
		width: 100%;
		height: 100%;
	}
	
	.connection-line {
		@apply cursor-pointer;
		pointer-events: stroke;
	}
	
	.connection-line:hover {
		stroke-width: 3;
	}
	
	.temp-connection {
		pointer-events: none;
	}
	
	/* Workflow Nodes */
	.workflow-node {
		@apply absolute w-48 bg-dark-800 border border-dark-700 rounded-lg shadow-lg cursor-pointer;
	}
	
	.workflow-node.selected {
		@apply border-primary-500 shadow-primary-500/20;
	}
	
	.node-header {
		@apply flex items-center justify-between p-3 rounded-t-lg;
	}
	
	.node-icon {
		@apply flex items-center justify-center;
	}
	
	.node-title {
		@apply flex-1 text-sm font-medium text-white mx-2;
	}
	
	.node-delete {
		@apply p-1 text-white/70 hover:text-white hover:bg-white/10 rounded transition-colors duration-200;
	}
	
	.node-content {
		@apply p-3 border-t border-dark-700;
	}
	
	.node-select,
	.node-input {
		@apply w-full px-2 py-1 bg-dark-700 border border-dark-600 rounded text-white text-sm;
	}
	
	.node-placeholder {
		@apply text-xs text-dark-400 text-center py-2;
	}
	
	.connection-ports {
		@apply relative;
	}
	
	.input-port,
	.output-port {
		@apply absolute w-3 h-3 bg-primary-600 rounded-full cursor-crosshair;
	}
	
	.input-port {
		@apply -left-1.5 top-1/2 transform -translate-y-1/2;
	}
	
	.output-port {
		@apply -right-1.5 top-1/2 transform -translate-y-1/2;
	}
	
	.drop-zone-hint {
		@apply absolute inset-0 flex items-center justify-center bg-primary-600/10 border-2 border-dashed border-primary-600 rounded-lg text-primary-400 font-medium;
	}
	
	/* Properties Panel */
	.properties-panel {
		@apply w-80 bg-dark-800 border-l border-dark-700 p-4 overflow-y-auto;
	}
	
	.panel-title {
		@apply text-lg font-semibold text-white mb-4;
	}
	
	.property-section {
		@apply mb-6;
	}
	
	.property-title {
		@apply text-sm font-medium text-white mb-3;
	}
	
	.property-form {
		@apply space-y-3;
	}
	
	.form-group {
		@apply space-y-1;
	}
	
	.form-label {
		@apply block text-xs font-medium text-dark-400;
	}
	
	.form-input,
	.form-select {
		@apply w-full px-3 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white text-sm;
	}
	
	.no-selection {
		@apply flex flex-col items-center justify-center py-8 text-center;
	}
	
	.workflow-stats {
		@apply mt-6 pt-6 border-t border-dark-700;
	}
	
	.stats-title {
		@apply text-sm font-medium text-white mb-3;
	}
	
	.stats-grid {
		@apply grid grid-cols-2 gap-3;
	}
	
	.stat-item {
		@apply text-center p-3 bg-dark-700 rounded-lg;
	}
	
	.stat-value {
		@apply block text-lg font-bold text-white;
	}
	
	.stat-label {
		@apply block text-xs text-dark-400;
	}
</style>
