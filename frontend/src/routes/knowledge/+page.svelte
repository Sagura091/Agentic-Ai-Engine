<!--
🚀 Revolutionary Knowledge Management Hub
Next-generation RAG system with AI-powered document processing, semantic search, and intelligent knowledge organization
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { notificationActions } from '$lib/stores';
	import { apiClient } from '$lib/services/api';
	import CreateKnowledgeBaseModal from '$lib/components/knowledge/CreateKnowledgeBaseModal.svelte';
	import KnowledgeBaseDetailView from '$lib/components/knowledge/KnowledgeBaseDetailView.svelte';
	import {
		Database, Brain, Sparkles, Plus, Search, Upload, BarChart3,
		FileText, Clock, Zap, Activity, Eye,
		Grid, List, Loader, CheckCircle
	} from 'lucide-svelte';

	// 🎯 State Management
	let currentView: 'main' | 'detail' = 'main';
	let activeTab: 'overview' | 'knowledge-bases' | 'documents' | 'search' | 'analytics' = 'overview';
	let loading = false;
	let knowledgeBases: any[] = [];
	let selectedKB: any = null;
	let viewMode: 'grid' | 'list' = 'grid';
	let searchQuery = '';
	let showCreateModal = false;

	// 📊 Statistics
	let stats = {
		totalKnowledgeBases: 0,
		totalDocuments: 0,
		totalChunks: 0,
		storageUsed: 0,
		searchQueries: 0,
		avgRelevanceScore: 0,
		processingJobs: 0,
		embeddingModels: 0
	};

	// 🚀 Lifecycle
	onMount(async () => {
		await loadData();
		startAnimations();
	});

	// 📊 Data Loading
	async function loadData() {
		loading = true;
		try {
			const kbResponse = await apiClient.getKnowledgeBases();
			if (kbResponse.success && kbResponse.data) {
				knowledgeBases = kbResponse.data;
				stats.totalKnowledgeBases = knowledgeBases.length;
			}

			const statsResponse = await apiClient.getRagStatistics();
			if (statsResponse.success && statsResponse.data) {
				stats = { ...stats, ...statsResponse.data };
			}
		} catch (error) {
			console.error('Failed to load data:', error);
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Data',
				message: 'Unable to load knowledge base data.'
			});
		} finally {
			loading = false;
		}
	}

	// 🎨 Animations
	function startAnimations() {
		const cards = document.querySelectorAll('.animate-card');
		cards.forEach((card, index) => {
			setTimeout(() => {
				card.classList.add('animate-in');
			}, index * 100);
		});
	}

	// 🎯 Actions
	function createKnowledgeBase() {
		showCreateModal = true;
	}

	function handleKnowledgeBaseCreated(event: CustomEvent) {
		const newKB = event.detail;
		knowledgeBases = [...knowledgeBases, newKB];
		stats.totalKnowledgeBases = knowledgeBases.length;

		// Automatically view the newly created knowledge base
		viewKnowledgeBase(newKB);
	}

	function viewKnowledgeBase(kb: any) {
		selectedKB = kb;
		currentView = 'detail';
	}

	function backToMain() {
		selectedKB = null;
		currentView = 'main';
		activeTab = 'knowledge-bases';
	}

	// 📊 Computed
	$: filteredKnowledgeBases = knowledgeBases.filter(kb => 
		kb.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
		kb.description.toLowerCase().includes(searchQuery.toLowerCase())
	);
</script>

<!-- 🎨 Main Container -->
{#if currentView === 'main'}
<div class="knowledge-hub">
	<!-- 🌟 Hero Header -->
	<div class="hero-header">
		<div class="hero-content">
			<div class="hero-text">
				<h1 class="hero-title">
					<Database class="w-8 h-8 text-accent-blue" />
					Knowledge Management Hub
					<Sparkles class="w-6 h-6 text-accent-purple animate-pulse" />
				</h1>
				<p class="hero-subtitle">
					Intelligent document processing, semantic search, and AI-powered knowledge organization
				</p>
			</div>
			<div class="hero-actions">
				<button
					class="btn-primary"
					on:click={createKnowledgeBase}
				>
					<Plus class="w-5 h-5" />
					Create Knowledge Base
				</button>
			</div>
		</div>
	</div>

	<!-- 📊 Stats Dashboard -->
	<div class="stats-grid">
		<div class="stat-card animate-card">
			<div class="stat-icon bg-blue-500/20">
				<Database class="w-6 h-6 text-blue-400" />
			</div>
			<div class="stat-content">
				<div class="stat-value">{stats.totalKnowledgeBases}</div>
				<div class="stat-label">Knowledge Bases</div>
			</div>
		</div>
		
		<div class="stat-card animate-card">
			<div class="stat-icon bg-green-500/20">
				<FileText class="w-6 h-6 text-green-400" />
			</div>
			<div class="stat-content">
				<div class="stat-value">{stats.totalDocuments}</div>
				<div class="stat-label">Documents</div>
			</div>
		</div>
		
		<div class="stat-card animate-card">
			<div class="stat-icon bg-purple-500/20">
				<Brain class="w-6 h-6 text-purple-400" />
			</div>
			<div class="stat-content">
				<div class="stat-value">{stats.totalChunks}</div>
				<div class="stat-label">Vector Chunks</div>
			</div>
		</div>
		
		<div class="stat-card animate-card">
			<div class="stat-icon bg-orange-500/20">
				<Activity class="w-6 h-6 text-orange-400" />
			</div>
			<div class="stat-content">
				<div class="stat-value">{stats.processingJobs}</div>
				<div class="stat-label">Processing Jobs</div>
			</div>
		</div>
	</div>

	<!-- 🎛️ Navigation Tabs -->
	<div class="tab-navigation">
		<button 
			class="tab-btn {activeTab === 'overview' ? 'active' : ''}"
			on:click={() => activeTab = 'overview'}
		>
			<Eye class="w-5 h-5" />
			Overview
		</button>
		<button 
			class="tab-btn {activeTab === 'knowledge-bases' ? 'active' : ''}"
			on:click={() => activeTab = 'knowledge-bases'}
		>
			<Database class="w-5 h-5" />
			Knowledge Bases
		</button>
		<button 
			class="tab-btn {activeTab === 'documents' ? 'active' : ''}"
			on:click={() => activeTab = 'documents'}
		>
			<FileText class="w-5 h-5" />
			Documents
		</button>
		<button 
			class="tab-btn {activeTab === 'search' ? 'active' : ''}"
			on:click={() => activeTab = 'search'}
		>
			<Search class="w-5 h-5" />
			Search
		</button>
		<button 
			class="tab-btn {activeTab === 'analytics' ? 'active' : ''}"
			on:click={() => activeTab = 'analytics'}
		>
			<BarChart3 class="w-5 h-5" />
			Analytics
		</button>
	</div>

	<!-- 📋 Tab Content -->
	<div class="tab-content">
		{#if activeTab === 'overview'}
			<!-- 🌟 Overview Tab -->
			<div class="overview-content">
				<div class="overview-grid">
					<!-- Quick Actions -->
					<div class="overview-card animate-card">
						<div class="card-header">
							<Zap class="w-6 h-6 text-accent-blue" />
							<h3>Quick Actions</h3>
						</div>
						<div class="quick-actions">
							<button class="quick-action-btn" on:click={createKnowledgeBase}>
								<Plus class="w-5 h-5" />
								New Knowledge Base
							</button>
							<button class="quick-action-btn" on:click={() => activeTab = 'search'}>
								<Search class="w-5 h-5" />
								Search Knowledge
							</button>
							<button class="quick-action-btn" on:click={() => activeTab = 'analytics'}>
								<BarChart3 class="w-5 h-5" />
								View Analytics
							</button>
						</div>
					</div>

					<!-- Recent Activity -->
					<div class="overview-card animate-card">
						<div class="card-header">
							<Clock class="w-6 h-6 text-accent-green" />
							<h3>Recent Activity</h3>
						</div>
						<div class="activity-list">
							<div class="activity-item">
								<CheckCircle class="w-4 h-4 text-green-400" />
								<span>Document processed successfully</span>
								<span class="activity-time">2 min ago</span>
							</div>
							<div class="activity-item">
								<Upload class="w-4 h-4 text-blue-400" />
								<span>New document uploaded</span>
								<span class="activity-time">5 min ago</span>
							</div>
							<div class="activity-item">
								<Database class="w-4 h-4 text-purple-400" />
								<span>Knowledge base created</span>
								<span class="activity-time">1 hour ago</span>
							</div>
						</div>
					</div>

					<!-- System Health -->
					<div class="overview-card animate-card">
						<div class="card-header">
							<Activity class="w-6 h-6 text-accent-orange" />
							<h3>System Health</h3>
						</div>
						<div class="health-metrics">
							<div class="health-metric">
								<div class="metric-label">Processing Queue</div>
								<div class="metric-value text-green-400">Healthy</div>
							</div>
							<div class="health-metric">
								<div class="metric-label">Vector Database</div>
								<div class="metric-value text-green-400">Online</div>
							</div>
							<div class="health-metric">
								<div class="metric-label">Embedding Models</div>
								<div class="metric-value text-blue-400">{stats.embeddingModels} Available</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		{:else if activeTab === 'knowledge-bases'}
			<!-- 📚 Knowledge Bases Tab -->
			<div class="knowledge-bases-content">
				<!-- Controls -->
				<div class="content-controls">
					<div class="search-bar">
						<Search class="w-5 h-5 text-dark-400" />
						<input 
							type="text" 
							placeholder="Search knowledge bases..."
							bind:value={searchQuery}
							class="search-input"
						/>
					</div>
					<div class="view-controls">
						<button 
							class="view-btn {viewMode === 'grid' ? 'active' : ''}"
							on:click={() => viewMode = 'grid'}
						>
							<Grid class="w-4 h-4" />
						</button>
						<button 
							class="view-btn {viewMode === 'list' ? 'active' : ''}"
							on:click={() => viewMode = 'list'}
						>
							<List class="w-4 h-4" />
						</button>
					</div>
				</div>

				<!-- Knowledge Bases Grid -->
				{#if loading}
					<div class="loading-state">
						<Loader class="w-8 h-8 animate-spin text-accent-blue" />
						<p>Loading knowledge bases...</p>
					</div>
				{:else if filteredKnowledgeBases.length === 0}
					<div class="empty-state">
						<Database class="w-16 h-16 text-dark-400" />
						<h3>No Knowledge Bases Found</h3>
						<p>Create your first knowledge base to get started</p>
						<button class="btn-primary" on:click={createKnowledgeBase}>
							<Plus class="w-5 h-5" />
							Create Knowledge Base
						</button>
					</div>
				{:else}
					<div class="kb-grid {viewMode}">
						{#each filteredKnowledgeBases as kb (kb.id)}
							<div class="kb-card animate-card">
								<div class="kb-header">
									<div class="kb-icon">
										<Database class="w-6 h-6 text-accent-blue" />
									</div>
									<div class="kb-info">
										<h4 class="kb-name">{kb.name}</h4>
										<p class="kb-description">{kb.description}</p>
									</div>
								</div>
								<div class="kb-stats">
									<div class="kb-stat">
										<FileText class="w-4 h-4" />
										<span>{kb.document_count || 0} docs</span>
									</div>
									<div class="kb-stat">
										<Brain class="w-4 h-4" />
										<span>{kb.chunk_count || 0} chunks</span>
									</div>
								</div>
								<div class="kb-actions">
									<button class="kb-action-btn" on:click={() => viewKnowledgeBase(kb)}>
										<Eye class="w-4 h-4" />
										View & Manage
									</button>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>

		{:else}
			<!-- 🚧 Other Tabs -->
			<div class="coming-soon">
				<Sparkles class="w-16 h-16 text-accent-purple" />
				<h3>Coming Soon</h3>
				<p>This feature is under development</p>
			</div>
		{/if}
	</div>
</div>
{:else if currentView === 'detail' && selectedKB}
	<KnowledgeBaseDetailView
		knowledgeBase={selectedKB}
		on:back={backToMain}
	/>
{/if}

<!-- 🎨 Modals -->
<CreateKnowledgeBaseModal
	bind:isOpen={showCreateModal}
	on:created={handleKnowledgeBaseCreated}
	on:close={() => showCreateModal = false}
/>

<style>
	/* 🎨 Knowledge Hub Styles */
	.knowledge-hub {
		min-height: 100vh;
		background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
		padding: 1.5rem;
	}

	/* 🌟 Hero Header */
	.hero-header {
		margin-bottom: 2rem;
		position: relative;
		overflow: hidden;
		border-radius: 1rem;
		background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
		border: 1px solid rgba(59, 130, 246, 0.2);
		backdrop-filter: blur(10px);
	}

	.hero-content {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 2rem;
		position: relative;
		z-index: 10;
	}

	.hero-title {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		font-size: 2.25rem;
		font-weight: 700;
		color: white;
		margin-bottom: 0.5rem;
	}

	.hero-subtitle {
		font-size: 1.125rem;
		color: #9ca3af;
		max-width: 32rem;
	}

	.hero-actions {
		display: flex;
		gap: 1rem;
	}

	/* 🎯 Buttons */
	.btn-primary {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.75rem 1.5rem;
		background: linear-gradient(135deg, #3b82f6 0%, #9333ea 100%);
		color: white;
		border-radius: 0.75rem;
		font-weight: 500;
		border: none;
		cursor: pointer;
		transition: all 0.3s ease;
		box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
	}

	.btn-primary:hover {
		background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
		box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
		transform: scale(1.05);
	}

	.btn-secondary {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.75rem 1.5rem;
		background: #374151;
		border: 1px solid #4b5563;
		color: white;
		border-radius: 0.75rem;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.3s ease;
	}

	.btn-secondary:hover {
		background: #4b5563;
		border-color: #6b7280;
	}

	/* 📊 Stats Grid */
	.stats-grid {
		display: grid;
		grid-template-columns: repeat(1, 1fr);
		gap: 1.5rem;
		margin-bottom: 2rem;
	}

	@media (min-width: 768px) {
		.stats-grid {
			grid-template-columns: repeat(2, 1fr);
		}
	}

	@media (min-width: 1024px) {
		.stats-grid {
			grid-template-columns: repeat(4, 1fr);
		}
	}

	.stat-card {
		background: rgba(31, 41, 55, 0.5);
		backdrop-filter: blur(10px);
		border: 1px solid #374151;
		border-radius: 0.75rem;
		padding: 1.5rem;
		transition: all 0.3s ease;
		opacity: 0;
		transform: translateY(20px);
	}

	.stat-card:hover {
		background: rgba(31, 41, 55, 0.7);
		transform: scale(1.05);
	}

	.stat-card.animate-in {
		opacity: 1;
		transform: translateY(0);
		transition: all 0.6s ease-out;
	}

	.stat-icon {
		width: 3rem;
		height: 3rem;
		border-radius: 0.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		margin-bottom: 1rem;
	}

	.stat-value {
		font-size: 1.875rem;
		font-weight: 700;
		color: white;
		margin-bottom: 0.25rem;
	}

	.stat-label {
		font-size: 0.875rem;
		color: #9ca3af;
	}

	/* 🎛️ Tab Navigation */
	.tab-navigation {
		display: flex;
		gap: 0.5rem;
		margin-bottom: 2rem;
		background: rgba(31, 41, 55, 0.5);
		backdrop-filter: blur(10px);
		border: 1px solid #374151;
		border-radius: 0.75rem;
		padding: 0.5rem;
	}

	.tab-btn {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.75rem 1rem;
		border-radius: 0.5rem;
		font-weight: 500;
		transition: all 0.3s ease;
		color: #9ca3af;
		background: none;
		border: none;
		cursor: pointer;
	}

	.tab-btn:hover {
		color: white;
		background: #374151;
	}

	.tab-btn.active {
		background: linear-gradient(135deg, #3b82f6 0%, #9333ea 100%);
		color: white;
		box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
	}

	/* 📋 Tab Content */
	.tab-content {
		min-height: 24rem;
	}

	/* 🌟 Overview Content */
	.overview-grid {
		display: grid;
		grid-template-columns: repeat(1, 1fr);
		gap: 1.5rem;
	}

	@media (min-width: 1024px) {
		.overview-grid {
			grid-template-columns: repeat(3, 1fr);
		}
	}

	.overview-card {
		background: rgba(31, 41, 55, 0.5);
		backdrop-filter: blur(10px);
		border: 1px solid #374151;
		border-radius: 0.75rem;
		padding: 1.5rem;
		transition: all 0.3s ease;
		opacity: 0;
		transform: translateY(20px);
	}

	.overview-card:hover {
		background: rgba(31, 41, 55, 0.7);
	}

	.overview-card.animate-in {
		opacity: 1;
		transform: translateY(0);
		transition: all 0.6s ease-out;
	}

	.card-header {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 1rem;
	}

	.card-header h3 {
		font-size: 1.125rem;
		font-weight: 600;
		color: white;
	}

	/* Quick Actions */
	.quick-actions {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.quick-action-btn {
		width: 100%;
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.75rem;
		background: #374151;
		color: white;
		border-radius: 0.5rem;
		border: none;
		cursor: pointer;
		transition: all 0.3s ease;
	}

	.quick-action-btn:hover {
		background: #4b5563;
		transform: scale(1.05);
	}

	/* Activity List */
	.activity-list {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.activity-item {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.75rem;
		background: rgba(55, 65, 81, 0.5);
		border-radius: 0.5rem;
	}

	.activity-time {
		font-size: 0.75rem;
		color: #9ca3af;
		margin-left: auto;
	}

	/* Health Metrics */
	.health-metrics {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.health-metric {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem;
		background: rgba(55, 65, 81, 0.5);
		border-radius: 0.5rem;
	}

	.metric-label {
		color: #d1d5db;
	}

	.metric-value {
		font-weight: 500;
	}

	/* 📚 Knowledge Bases Content */
	.content-controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 1.5rem;
		gap: 1rem;
	}

	.search-bar {
		position: relative;
		flex: 1;
		max-width: 28rem;
	}

	.search-bar svg {
		position: absolute;
		left: 0.75rem;
		top: 50%;
		transform: translateY(-50%);
		pointer-events: none;
	}

	.search-input {
		width: 100%;
		padding: 0.75rem 1rem 0.75rem 2.5rem;
		background: #1f2937;
		border: 1px solid #374151;
		border-radius: 0.75rem;
		color: white;
		transition: all 0.3s ease;
	}

	.search-input::placeholder {
		color: #9ca3af;
	}

	.search-input:focus {
		outline: none;
		border-color: #3b82f6;
		box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
	}

	.view-controls {
		display: flex;
		gap: 0.5rem;
		background: #1f2937;
		border: 1px solid #374151;
		border-radius: 0.5rem;
		padding: 0.25rem;
	}

	.view-btn {
		padding: 0.5rem;
		border-radius: 0.375rem;
		transition: all 0.3s ease;
		color: #9ca3af;
		background: none;
		border: none;
		cursor: pointer;
	}

	.view-btn:hover {
		color: white;
		background: #4b5563;
	}

	.view-btn.active {
		background: #3b82f6;
		color: white;
	}

	/* Loading & Empty States */
	.loading-state, .empty-state, .coming-soon {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 4rem 0;
		text-align: center;
	}

	.empty-state h3, .coming-soon h3 {
		font-size: 1.25rem;
		font-weight: 600;
		color: white;
		margin: 1rem 0 0.5rem 0;
	}

	.empty-state p, .coming-soon p {
		color: #9ca3af;
		margin-bottom: 1.5rem;
	}

	/* Knowledge Base Grid */
	.kb-grid.grid {
		display: grid;
		grid-template-columns: repeat(1, 1fr);
		gap: 1.5rem;
	}

	@media (min-width: 768px) {
		.kb-grid.grid {
			grid-template-columns: repeat(2, 1fr);
		}
	}

	@media (min-width: 1024px) {
		.kb-grid.grid {
			grid-template-columns: repeat(3, 1fr);
		}
	}

	.kb-grid.list {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.kb-card {
		background: rgba(31, 41, 55, 0.5);
		backdrop-filter: blur(10px);
		border: 1px solid #374151;
		border-radius: 0.75rem;
		padding: 1.5rem;
		transition: all 0.3s ease;
		opacity: 0;
		transform: translateY(20px);
	}

	.kb-card:hover {
		background: rgba(31, 41, 55, 0.7);
		border-color: rgba(59, 130, 246, 0.5);
		transform: scale(1.05);
	}

	.kb-card.animate-in {
		opacity: 1;
		transform: translateY(0);
		transition: all 0.6s ease-out;
	}

	.kb-header {
		display: flex;
		align-items: flex-start;
		gap: 1rem;
		margin-bottom: 1rem;
	}

	.kb-icon {
		width: 3rem;
		height: 3rem;
		background: rgba(59, 130, 246, 0.2);
		border-radius: 0.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.kb-name {
		font-size: 1.125rem;
		font-weight: 600;
		color: white;
		margin-bottom: 0.25rem;
	}

	.kb-description {
		font-size: 0.875rem;
		color: #9ca3af;
	}

	.kb-stats {
		display: flex;
		gap: 1rem;
		margin-bottom: 1rem;
		font-size: 0.875rem;
		color: #d1d5db;
	}

	.kb-stat {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.kb-actions {
		display: flex;
		gap: 0.5rem;
	}

	.kb-action-btn {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		background: #374151;
		color: white;
		border-radius: 0.5rem;
		font-size: 0.875rem;
		border: none;
		cursor: pointer;
		transition: all 0.3s ease;
	}

	.kb-action-btn:hover {
		background: #4b5563;
	}

	/* 📄 Documents Content */
	.documents-content {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.kb-header-section {
		background: rgba(31, 41, 55, 0.5);
		backdrop-filter: blur(10px);
		border: 1px solid #374151;
		border-radius: 0.75rem;
		padding: 1.5rem;
	}

	.back-btn {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 1rem;
		background: #374151;
		color: #d1d5db;
		border: none;
		border-radius: 0.5rem;
		cursor: pointer;
		transition: all 0.3s ease;
		margin-bottom: 1.5rem;
		font-size: 0.875rem;
	}

	.back-btn:hover {
		background: #4b5563;
		color: white;
	}

	.kb-info-header {
		display: flex;
		align-items: center;
		gap: 1.5rem;
	}

	.kb-icon-large {
		width: 4rem;
		height: 4rem;
		background: rgba(59, 130, 246, 0.2);
		border-radius: 0.75rem;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.kb-details {
		flex: 1;
	}

	.kb-title {
		font-size: 1.5rem;
		font-weight: 700;
		color: white;
		margin-bottom: 0.5rem;
	}

	.kb-subtitle {
		color: #9ca3af;
		margin-bottom: 1rem;
	}

	.kb-meta {
		display: flex;
		gap: 1.5rem;
	}

	.meta-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: #d1d5db;
		font-size: 0.875rem;
	}

	.kb-actions-header {
		display: flex;
		gap: 1rem;
	}

	.documents-list {
		background: rgba(31, 41, 55, 0.5);
		backdrop-filter: blur(10px);
		border: 1px solid #374151;
		border-radius: 0.75rem;
		padding: 2rem;
		min-height: 20rem;
	}

	/* 🎨 Animations */
	@keyframes fadeInUp {
		from {
			opacity: 0;
			transform: translateY(20px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.animate-card {
		animation: fadeInUp 0.6s ease-out forwards;
	}

	/* 🌟 Accent Colors */
	.text-accent-blue { color: #60a5fa; }
	.text-accent-purple { color: #a78bfa; }
	.text-accent-green { color: #4ade80; }
	.text-accent-orange { color: #fb923c; }

	/* 📱 Responsive Design */
	@media (max-width: 768px) {
		.hero-content {
			flex-direction: column;
			gap: 1.5rem;
			text-align: center;
		}

		.hero-actions {
			flex-direction: column;
			width: 100%;
		}

		.stats-grid {
			grid-template-columns: repeat(2, 1fr);
		}

		.overview-grid {
			grid-template-columns: repeat(1, 1fr);
		}

		.content-controls {
			flex-direction: column;
			gap: 1rem;
		}

		.search-bar {
			max-width: none;
		}
	}
</style>
