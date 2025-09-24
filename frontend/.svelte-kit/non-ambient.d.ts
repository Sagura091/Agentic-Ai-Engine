
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
		RouteId(): "/" | "/admin" | "/admin/enhanced-settings" | "/admin/enhanced-settings/components" | "/admin/enhanced-settings/managers" | "/admin/enhanced-settings/services" | "/admin/enhanced-settings/stores" | "/admin/enhanced-settings/types" | "/admin/settings" | "/agents" | "/dashboard" | "/debug" | "/login" | "/profile" | "/test" | "/workflows";
		RouteParams(): {
			
		};
		LayoutParams(): {
			"/": Record<string, never>;
			"/admin": Record<string, never>;
			"/admin/enhanced-settings": Record<string, never>;
			"/admin/enhanced-settings/components": Record<string, never>;
			"/admin/enhanced-settings/managers": Record<string, never>;
			"/admin/enhanced-settings/services": Record<string, never>;
			"/admin/enhanced-settings/stores": Record<string, never>;
			"/admin/enhanced-settings/types": Record<string, never>;
			"/admin/settings": Record<string, never>;
			"/agents": Record<string, never>;
			"/dashboard": Record<string, never>;
			"/debug": Record<string, never>;
			"/login": Record<string, never>;
			"/profile": Record<string, never>;
			"/test": Record<string, never>;
			"/workflows": Record<string, never>
		};
		Pathname(): "/" | "/admin" | "/admin/" | "/admin/enhanced-settings" | "/admin/enhanced-settings/" | "/admin/enhanced-settings/components" | "/admin/enhanced-settings/components/" | "/admin/enhanced-settings/managers" | "/admin/enhanced-settings/managers/" | "/admin/enhanced-settings/services" | "/admin/enhanced-settings/services/" | "/admin/enhanced-settings/stores" | "/admin/enhanced-settings/stores/" | "/admin/enhanced-settings/types" | "/admin/enhanced-settings/types/" | "/admin/settings" | "/admin/settings/" | "/agents" | "/agents/" | "/dashboard" | "/dashboard/" | "/debug" | "/debug/" | "/login" | "/login/" | "/profile" | "/profile/" | "/test" | "/test/" | "/workflows" | "/workflows/";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): "/favicon.png" | string & {};
	}
}