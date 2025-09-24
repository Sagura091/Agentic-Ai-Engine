<!--
Enhanced Settings Sidebar Component

This component renders the sidebar navigation for the Enhanced Admin Settings page,
including category management, selection, and collapsible group functionality.
-->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import {
		Settings,
		ChevronRight,
		ChevronDown,
		Shield,
		Brain,
		Workflow,
		Target,
		Wrench,
		Globe,
		Archive,
		Bell,
		Bot,
		HardDrive,
		Activity,
		Database,
		Cpu,
		BarChart3,
		Layers,
		Sparkles
	} from 'lucide-svelte';
	import type { CategoryGroup } from '../types/EnhancedSettingsTypes';

	// Props
	export let categories: Record<string, CategoryGroup> = {};
	export let activeCategory: string = '';
	export let expandedGroups: Set<string> = new Set(['core']);

	// Event dispatcher
	const dispatch = createEventDispatcher();

	// Icon mapping
	function getIconComponent(iconName: string) {
		const iconMap: Record<string, any> = {
			'🔧': Settings,
			'🔐': Shield,
			'🧠': Brain,
			'⚡': Workflow,
			'🎯': Target,
			'🔨': Wrench,
			'🌐': Globe,
			'📦': Archive,
			'🔔': Bell,
			'🤖': Bot,
			'💾': HardDrive,
			'📊': Activity,
			'🗄️': Database,
			'⚙️': Cpu,
			'📈': BarChart3,
			'🏗️': Layers,
			'✨': Sparkles,
			Settings,
			Shield,
			Brain,
			Workflow,
			Target,
			Wrench,
			Globe,
			Archive,
			Bell,
			Bot,
			HardDrive,
			Activity,
			Database,
			Cpu,
			BarChart3,
			Layers,
			Sparkles
		};
		return iconMap[iconName] || Settings;
	}

	// Event handlers
	function handleCategorySelect(categoryId: string) {
		dispatch('categorySelect', { categoryId });
	}

	function handleToggleGroup(groupId: string) {
		dispatch('toggleGroup', { groupId });
	}
</script>

<!-- Compact Sidebar - Categories -->
<div class="lg:col-span-1">
	<div class="glass rounded-xl border border-white/10 p-4 sticky top-20">
		<h2 class="text-sm font-semibold text-white mb-3 flex items-center space-x-2">
			<Settings class="w-4 h-4" />
			<span>Categories</span>
		</h2>

		{#if Object.keys(categories).length > 0}
			{#each Object.entries(categories) as [groupId, group]}
				<div class="mb-3">
					<button
						on:click={() => handleToggleGroup(groupId)}
						class="w-full flex items-center justify-between p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-white/80 hover:text-white"
					>
						<span class="font-medium text-sm">{group.name}</span>
						{#if expandedGroups.has(groupId)}
							<ChevronDown class="w-3 h-3" />
						{:else}
							<ChevronRight class="w-3 h-3" />
						{/if}
					</button>

					{#if expandedGroups.has(groupId)}
						<div class="mt-1 space-y-1 pl-1">
							{#each group.categories as category}
								<button
									on:click={() => handleCategorySelect(category.id)}
									class="w-full flex items-center space-x-2 p-2 rounded-lg transition-all duration-200 text-left {activeCategory === category.id
										? 'bg-gradient-to-r from-purple-500/30 to-pink-500/30 text-white border border-purple-400/30'
										: 'hover:bg-white/5 text-white/70 hover:text-white'}"
								>
									<svelte:component
										this={getIconComponent(category.icon)}
										class="w-3 h-3 flex-shrink-0"
									/>
									<div class="min-w-0 flex-1">
										<div class="font-medium text-xs truncate">
											{category.name}
										</div>
										<div class="text-xs opacity-60 truncate">
											{category.description}
										</div>
									</div>
								</button>
							{/each}
						</div>
					{/if}
				</div>
			{/each}
		{:else}
			<div class="text-center py-6">
				<Settings class="w-8 h-8 text-white/40 mx-auto mb-2" />
				<p class="text-white/60 text-xs">Loading...</p>
			</div>
		{/if}
	</div>
</div>

<style>
	.glass {
		background: rgba(255, 255, 255, 0.05);
		backdrop-filter: blur(10px);
		-webkit-backdrop-filter: blur(10px);
		border: 1px solid rgba(255, 255, 255, 0.1);
	}
</style>
