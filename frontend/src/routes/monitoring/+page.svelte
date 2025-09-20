<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { 
		systemMetrics, 
		connectionStatus, 
		agents, 
		workflows,
		notificationActions 
	} from '$stores';
	import { websocketService } from '$services/websocket';
	import { apiClient } from '$services/api';
	import { 
		Activity, 
		Server, 
		Cpu, 
		HardDrive, 
		Wifi, 
		AlertTriangle, 
		CheckCircle, 
		Clock,
		TrendingUp,
		TrendingDown,
		Zap,
		Bot,
		Network,
		Eye,
		RefreshCw,
		Download,
		Settings
	} from 'lucide-svelte';
	import type { SystemMetrics } from '$types';
	
	// State
	let refreshInterval: NodeJS.Timeout;
	let isRefreshing = false;
	let selectedTimeRange = '1h';
	let autoRefresh = true;
	
	// Mock metrics data for demonstration
	let metricsHistory: SystemMetrics[] = [];
	let alertsData = [
		{
			id: '1',
			type: 'warning',
			title: 'High CPU Usage',
			message: 'CPU usage has been above 80% for the last 5 minutes',
			timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
			resolved: false
		},
		{
			id: '2',
			type: 'info',
			title: 'Agent Deployment',
			message: 'New agent "Customer Support Bot" deployed successfully',
			timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
			resolved: true
		},
		{
			id: '3',
			type: 'error',
			title: 'Workflow Failure',
			message: 'Workflow "Data Processing Pipeline" failed with timeout error',
			timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
			resolved: false
		}
	];
	
	// Time range options
	const timeRanges = [
		{ value: '15m', label: '15 minutes' },
		{ value: '1h', label: '1 hour' },
		{ value: '6h', label: '6 hours' },
		{ value: '24h', label: '24 hours' },
		{ value: '7d', label: '7 days' }
	];
	
	onMount(async () => {
		await loadInitialData();
		setupRealTimeUpdates();
		
		if (autoRefresh) {
			startAutoRefresh();
		}
	});
	
	onDestroy(() => {
		if (refreshInterval) {
			clearInterval(refreshInterval);
		}
	});
	
	async function loadInitialData() {
		isRefreshing = true;
		
		try {
			// Load system metrics
			await loadSystemMetrics();
			
			// Load agents and workflows if not already loaded
			if ($agents.length === 0) {
				const agentsResponse = await apiClient.getAgents();
				if (agentsResponse.success && agentsResponse.data) {
					agents.set(agentsResponse.data);
				}
			}
			
			if ($workflows.length === 0) {
				const workflowsResponse = await apiClient.getWorkflows();
				if (workflowsResponse.success && workflowsResponse.data) {
					workflows.set(workflowsResponse.data);
				}
			}
		} catch (error) {
			console.error('Failed to load monitoring data:', error);
			notificationActions.add({
				type: 'error',
				title: 'Failed to Load Data',
				message: 'Unable to load monitoring data'
			});
		} finally {
			isRefreshing = false;
		}
	}
	
	async function loadSystemMetrics() {
		try {
			// Mock system metrics - in real implementation, this would call the API
			const mockMetrics: SystemMetrics = {
				timestamp: new Date().toISOString(),
				cpu_usage: Math.random() * 100,
				memory_usage: Math.random() * 100,
				disk_usage: Math.random() * 100,
				network_io: {
					bytes_sent: Math.floor(Math.random() * 1000000),
					bytes_received: Math.floor(Math.random() * 1000000)
				},
				active_agents: $agents.filter(a => a.status === 'running').length,
				active_workflows: $workflows.filter(w => w.status === 'active').length,
				total_requests: Math.floor(Math.random() * 10000),
				error_rate: Math.random() * 5,
				response_time: Math.random() * 1000
			};
			
			systemMetrics.set(mockMetrics);
			
			// Add to history for charts
			metricsHistory = [...metricsHistory.slice(-59), mockMetrics]; // Keep last 60 data points
		} catch (error) {
			console.error('Failed to load system metrics:', error);
		}
	}
	
	function setupRealTimeUpdates() {
		// Subscribe to real-time system metrics
		websocketService.on('system_metrics' as any, (data: SystemMetrics) => {
			systemMetrics.set(data);
			metricsHistory = [...metricsHistory.slice(-59), data];
		});
		
		// Subscribe to alerts
		websocketService.on('system_alert' as any, (alert: any) => {
			alertsData = [alert, ...alertsData];
			
			notificationActions.add({
				type: alert.type,
				title: alert.title,
				message: alert.message
			});
		});
	}
	
	function startAutoRefresh() {
		refreshInterval = setInterval(() => {
			if (autoRefresh) {
				loadSystemMetrics();
			}
		}, 30000); // Refresh every 30 seconds
	}
	
	function toggleAutoRefresh() {
		autoRefresh = !autoRefresh;
		
		if (autoRefresh) {
			startAutoRefresh();
		} else if (refreshInterval) {
			clearInterval(refreshInterval);
		}
	}
	
	async function manualRefresh() {
		await loadSystemMetrics();
		notificationActions.add({
			type: 'success',
			title: 'Data Refreshed',
			message: 'System metrics updated successfully'
		});
	}
	
	function exportMetrics() {
		const data = {
			timestamp: new Date().toISOString(),
			current_metrics: $systemMetrics,
			history: metricsHistory,
			alerts: alertsData
		};
		
		const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `system-metrics-${new Date().toISOString().split('T')[0]}.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
		
		notificationActions.add({
			type: 'success',
			title: 'Metrics Exported',
			message: 'System metrics have been downloaded'
		});
	}
	
	function resolveAlert(alertId: string) {
		alertsData = alertsData.map(alert => 
			alert.id === alertId ? { ...alert, resolved: true } : alert
		);
		
		notificationActions.add({
			type: 'success',
			title: 'Alert Resolved',
			message: 'Alert has been marked as resolved'
		});
	}
	
	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}
	
	function formatTime(timestamp: string): string {
		return new Date(timestamp).toLocaleTimeString('en-US', {
			hour: '2-digit',
			minute: '2-digit'
		});
	}
	
	function getStatusColor(value: number, thresholds: { warning: number; critical: number }): string {
		if (value >= thresholds.critical) return 'text-red-400';
		if (value >= thresholds.warning) return 'text-yellow-400';
		return 'text-green-400';
	}
	
	function getStatusIcon(value: number, thresholds: { warning: number; critical: number }) {
		if (value >= thresholds.critical) return AlertTriangle;
		if (value >= thresholds.warning) return AlertTriangle;
		return CheckCircle;
	}
</script>

<svelte:head>
	<title>System Monitoring - Agentic AI</title>
</svelte:head>

<div class="monitoring-container">
	<!-- Header -->
	<div class="monitoring-header">
		<div class="header-content">
			<div class="header-text">
				<h1 class="page-title">System Monitoring</h1>
				<p class="page-subtitle">
					Real-time monitoring and analytics for your AI agent platform.
				</p>
			</div>
			
			<div class="header-actions">
				<div class="time-range-selector">
					<select class="time-select" bind:value={selectedTimeRange}>
						{#each timeRanges as range}
							<option value={range.value}>{range.label}</option>
						{/each}
					</select>
				</div>
				
				<button 
					class="action-btn {autoRefresh ? 'active' : 'secondary'}"
					on:click={toggleAutoRefresh}
				>
					<RefreshCw class="w-4 h-4 {autoRefresh ? 'animate-spin' : ''}" />
					<span>Auto Refresh</span>
				</button>
				
				<button class="action-btn secondary" on:click={manualRefresh}>
					<RefreshCw class="w-4 h-4" />
					<span>Refresh</span>
				</button>
				
				<button class="action-btn secondary" on:click={exportMetrics}>
					<Download class="w-4 h-4" />
					<span>Export</span>
				</button>
			</div>
		</div>
		
		<!-- Connection Status -->
		<div class="connection-status">
			<div class="status-indicator {$connectionStatus}">
				<div class="status-dot"></div>
				<span class="status-text">
					{$connectionStatus === 'connected' ? 'Connected' : 
					 $connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
				</span>
			</div>
		</div>
	</div>
	
	<!-- System Overview Cards -->
	{#if $systemMetrics}
		<div class="metrics-grid">
			<!-- CPU Usage -->
			<div class="metric-card">
				<div class="metric-header">
					<div class="metric-icon">
						<Cpu class="w-6 h-6 text-accent-blue" />
					</div>
					<div class="metric-info">
						<h3 class="metric-title">CPU Usage</h3>
						<div class="metric-status">
							<svelte:component 
								this={getStatusIcon($systemMetrics.cpu_usage, { warning: 70, critical: 90 })} 
								class="w-4 h-4 {getStatusColor($systemMetrics.cpu_usage, { warning: 70, critical: 90 })}" 
							/>
						</div>
					</div>
				</div>
				<div class="metric-value">
					<span class="value-number">{$systemMetrics.cpu_usage.toFixed(1)}%</span>
				</div>
				<div class="metric-chart">
					<!-- Simple sparkline chart -->
					<div class="sparkline">
						{#each metricsHistory.slice(-20) as metric, index}
							<div 
								class="sparkline-bar"
								style="height: {metric.cpu_usage}%"
							></div>
						{/each}
					</div>
				</div>
			</div>
			
			<!-- Memory Usage -->
			<div class="metric-card">
				<div class="metric-header">
					<div class="metric-icon">
						<HardDrive class="w-6 h-6 text-accent-green" />
					</div>
					<div class="metric-info">
						<h3 class="metric-title">Memory Usage</h3>
						<div class="metric-status">
							<svelte:component 
								this={getStatusIcon($systemMetrics.memory_usage, { warning: 80, critical: 95 })} 
								class="w-4 h-4 {getStatusColor($systemMetrics.memory_usage, { warning: 80, critical: 95 })}" 
							/>
						</div>
					</div>
				</div>
				<div class="metric-value">
					<span class="value-number">{$systemMetrics.memory_usage.toFixed(1)}%</span>
				</div>
				<div class="metric-chart">
					<div class="sparkline">
						{#each metricsHistory.slice(-20) as metric, index}
							<div 
								class="sparkline-bar green"
								style="height: {metric.memory_usage}%"
							></div>
						{/each}
					</div>
				</div>
			</div>
			
			<!-- Active Agents -->
			<div class="metric-card">
				<div class="metric-header">
					<div class="metric-icon">
						<Bot class="w-6 h-6 text-accent-purple" />
					</div>
					<div class="metric-info">
						<h3 class="metric-title">Active Agents</h3>
						<div class="metric-status">
							<CheckCircle class="w-4 h-4 text-green-400" />
						</div>
					</div>
				</div>
				<div class="metric-value">
					<span class="value-number">{$systemMetrics.active_agents}</span>
					<span class="value-total">/ {$agents.length}</span>
				</div>
				<div class="metric-trend">
					<TrendingUp class="w-4 h-4 text-green-400" />
					<span class="trend-text">+2 from last hour</span>
				</div>
			</div>
			
			<!-- Active Workflows -->
			<div class="metric-card">
				<div class="metric-header">
					<div class="metric-icon">
						<Network class="w-6 h-6 text-accent-orange" />
					</div>
					<div class="metric-info">
						<h3 class="metric-title">Active Workflows</h3>
						<div class="metric-status">
							<CheckCircle class="w-4 h-4 text-green-400" />
						</div>
					</div>
				</div>
				<div class="metric-value">
					<span class="value-number">{$systemMetrics.active_workflows}</span>
					<span class="value-total">/ {$workflows.length}</span>
				</div>
				<div class="metric-trend">
					<TrendingUp class="w-4 h-4 text-green-400" />
					<span class="trend-text">Stable</span>
				</div>
			</div>
			
			<!-- Response Time -->
			<div class="metric-card">
				<div class="metric-header">
					<div class="metric-icon">
						<Clock class="w-6 h-6 text-accent-yellow" />
					</div>
					<div class="metric-info">
						<h3 class="metric-title">Avg Response Time</h3>
						<div class="metric-status">
							<CheckCircle class="w-4 h-4 text-green-400" />
						</div>
					</div>
				</div>
				<div class="metric-value">
					<span class="value-number">{$systemMetrics.response_time.toFixed(0)}ms</span>
				</div>
				<div class="metric-trend">
					<TrendingDown class="w-4 h-4 text-green-400" />
					<span class="trend-text">-15ms from last hour</span>
				</div>
			</div>
			
			<!-- Error Rate -->
			<div class="metric-card">
				<div class="metric-header">
					<div class="metric-icon">
						<AlertTriangle class="w-6 h-6 text-accent-red" />
					</div>
					<div class="metric-info">
						<h3 class="metric-title">Error Rate</h3>
						<div class="metric-status">
							<svelte:component 
								this={getStatusIcon($systemMetrics.error_rate, { warning: 2, critical: 5 })} 
								class="w-4 h-4 {getStatusColor($systemMetrics.error_rate, { warning: 2, critical: 5 })}" 
							/>
						</div>
					</div>
				</div>
				<div class="metric-value">
					<span class="value-number">{$systemMetrics.error_rate.toFixed(2)}%</span>
				</div>
				<div class="metric-trend">
					<TrendingDown class="w-4 h-4 text-green-400" />
					<span class="trend-text">-0.5% from last hour</span>
				</div>
			</div>
		</div>
	{/if}
	
	<!-- Alerts Section -->
	<div class="alerts-section">
		<div class="section-header">
			<h2 class="section-title">Recent Alerts</h2>
			<div class="alert-summary">
				<span class="alert-count error">{alertsData.filter(a => a.type === 'error' && !a.resolved).length} Errors</span>
				<span class="alert-count warning">{alertsData.filter(a => a.type === 'warning' && !a.resolved).length} Warnings</span>
			</div>
		</div>
		
		<div class="alerts-list">
			{#each alertsData.slice(0, 10) as alert (alert.id)}
				<div class="alert-item {alert.type} {alert.resolved ? 'resolved' : ''}">
					<div class="alert-icon">
						{#if alert.type === 'error'}
							<AlertTriangle class="w-5 h-5 text-red-400" />
						{:else if alert.type === 'warning'}
							<AlertTriangle class="w-5 h-5 text-yellow-400" />
						{:else}
							<CheckCircle class="w-5 h-5 text-blue-400" />
						{/if}
					</div>
					
					<div class="alert-content">
						<div class="alert-header">
							<h4 class="alert-title">{alert.title}</h4>
							<span class="alert-time">{formatTime(alert.timestamp)}</span>
						</div>
						<p class="alert-message">{alert.message}</p>
					</div>
					
					{#if !alert.resolved && alert.type !== 'info'}
						<button 
							class="resolve-btn"
							on:click={() => resolveAlert(alert.id)}
						>
							Resolve
						</button>
					{/if}
				</div>
			{/each}
		</div>
	</div>
</div>

<style>
	.monitoring-container {
		@apply space-y-8;
	}
	
	/* Header */
	.monitoring-header {
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
	
	.time-range-selector {
		@apply relative;
	}
	
	.time-select {
		@apply px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white text-sm;
	}
	
	.action-btn {
		@apply inline-flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200;
	}
	
	.action-btn.secondary {
		@apply bg-dark-800 text-dark-300 hover:bg-dark-700 hover:text-white;
	}
	
	.action-btn.active {
		@apply bg-primary-600 text-white;
	}
	
	.connection-status {
		@apply flex justify-end;
	}
	
	.status-indicator {
		@apply flex items-center space-x-2 px-3 py-2 rounded-lg;
	}
	
	.status-indicator.connected {
		@apply bg-green-500/20 text-green-400;
	}
	
	.status-indicator.connecting {
		@apply bg-yellow-500/20 text-yellow-400;
	}
	
	.status-indicator.disconnected {
		@apply bg-red-500/20 text-red-400;
	}
	
	.status-dot {
		@apply w-2 h-2 rounded-full;
	}
	
	.status-indicator.connected .status-dot {
		@apply bg-green-400 animate-pulse;
	}
	
	.status-indicator.connecting .status-dot {
		@apply bg-yellow-400 animate-pulse;
	}
	
	.status-indicator.disconnected .status-dot {
		@apply bg-red-400;
	}
	
	.status-text {
		@apply text-sm font-medium;
	}
	
	/* Metrics Grid */
	.metrics-grid {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6;
	}
	
	.metric-card {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6 space-y-4;
	}
	
	.metric-header {
		@apply flex items-center justify-between;
	}
	
	.metric-icon {
		@apply w-12 h-12 bg-dark-700 rounded-xl flex items-center justify-center;
	}
	
	.metric-info {
		@apply flex-1 ml-4;
	}
	
	.metric-title {
		@apply text-sm font-medium text-dark-300;
	}
	
	.metric-status {
		@apply mt-1;
	}
	
	.metric-value {
		@apply flex items-baseline space-x-2;
	}
	
	.value-number {
		@apply text-2xl font-bold text-white;
	}
	
	.value-total {
		@apply text-sm text-dark-400;
	}
	
	.metric-trend {
		@apply flex items-center space-x-2 text-sm;
	}
	
	.trend-text {
		@apply text-dark-400;
	}
	
	.metric-chart {
		@apply h-8;
	}
	
	.sparkline {
		@apply flex items-end space-x-1 h-full;
	}
	
	.sparkline-bar {
		@apply w-1 bg-accent-blue rounded-t opacity-70;
		min-height: 2px;
	}
	
	.sparkline-bar.green {
		@apply bg-accent-green;
	}
	
	/* Alerts Section */
	.alerts-section {
		@apply bg-dark-800 border border-dark-700 rounded-xl p-6;
	}
	
	.section-header {
		@apply flex items-center justify-between mb-6;
	}
	
	.section-title {
		@apply text-xl font-semibold text-white;
	}
	
	.alert-summary {
		@apply flex items-center space-x-4;
	}
	
	.alert-count {
		@apply text-sm font-medium;
	}
	
	.alert-count.error {
		@apply text-red-400;
	}
	
	.alert-count.warning {
		@apply text-yellow-400;
	}
	
	.alerts-list {
		@apply space-y-3;
	}
	
	.alert-item {
		@apply flex items-start space-x-4 p-4 rounded-lg border;
	}
	
	.alert-item.error {
		@apply bg-red-500/10 border-red-500/30;
	}
	
	.alert-item.warning {
		@apply bg-yellow-500/10 border-yellow-500/30;
	}
	
	.alert-item.info {
		@apply bg-blue-500/10 border-blue-500/30;
	}
	
	.alert-item.resolved {
		@apply opacity-60;
	}
	
	.alert-icon {
		@apply flex-shrink-0 mt-0.5;
	}
	
	.alert-content {
		@apply flex-1 space-y-1;
	}
	
	.alert-header {
		@apply flex items-center justify-between;
	}
	
	.alert-title {
		@apply text-sm font-medium text-white;
	}
	
	.alert-time {
		@apply text-xs text-dark-400;
	}
	
	.alert-message {
		@apply text-sm text-dark-300;
	}
	
	.resolve-btn {
		@apply px-3 py-1 bg-primary-600 text-white text-sm rounded hover:bg-primary-700 transition-colors duration-200;
	}
	
	/* Mobile Responsiveness */
	@media (max-width: 768px) {
		.header-content {
			@apply flex-col items-start space-y-4;
		}
		
		.header-actions {
			@apply flex-wrap;
		}
		
		.metrics-grid {
			@apply grid-cols-1;
		}
	}
</style>
