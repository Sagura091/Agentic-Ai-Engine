
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	export interface AppTypes {
		RouteId(): "/" | "/agents" | "/agents/create-visual" | "/agents/create" | "/auth" | "/auth/login" | "/auth/register" | "/collaboration" | "/knowledge" | "/models" | "/monitoring" | "/settings" | "/settings/rag" | "/testing" | "/tools" | "/workflows" | "/workflows/create";
		RouteParams(): {
			
		};
		LayoutParams(): {
			"/": Record<string, never>;
			"/agents": Record<string, never>;
			"/agents/create-visual": Record<string, never>;
			"/agents/create": Record<string, never>;
			"/auth": Record<string, never>;
			"/auth/login": Record<string, never>;
			"/auth/register": Record<string, never>;
			"/collaboration": Record<string, never>;
			"/knowledge": Record<string, never>;
			"/models": Record<string, never>;
			"/monitoring": Record<string, never>;
			"/settings": Record<string, never>;
			"/settings/rag": Record<string, never>;
			"/testing": Record<string, never>;
			"/tools": Record<string, never>;
			"/workflows": Record<string, never>;
			"/workflows/create": Record<string, never>
		};
		Pathname(): "/" | "/agents" | "/agents/" | "/agents/create-visual" | "/agents/create-visual/" | "/agents/create" | "/agents/create/" | "/auth" | "/auth/" | "/auth/login" | "/auth/login/" | "/auth/register" | "/auth/register/" | "/collaboration" | "/collaboration/" | "/knowledge" | "/knowledge/" | "/models" | "/models/" | "/monitoring" | "/monitoring/" | "/settings" | "/settings/" | "/settings/rag" | "/settings/rag/" | "/testing" | "/testing/" | "/tools" | "/tools/" | "/workflows" | "/workflows/" | "/workflows/create" | "/workflows/create/";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): string & {};
	}
}