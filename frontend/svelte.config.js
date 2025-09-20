import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	// Compiler options for compatibility
	compilerOptions: {
		legacy: true
	},

	kit: {
		// adapter-auto only supports some environments, see https://kit.svelte.dev/docs/adapter-auto for a list.
		// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
		// See https://kit.svelte.dev/docs/adapters for more information about adapters.
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		
		// Enable service worker for offline functionality
		serviceWorker: {
			register: false // We'll handle this manually for better control
		},
		
		// Prerender configuration
		prerender: {
			handleHttpError: 'warn',
			handleMissingId: 'warn'
		},
		
		// CSP configuration for security
		csp: {
			mode: 'auto',
			directives: {
				'script-src': ['self', 'unsafe-inline', 'unsafe-eval', 'blob:', 'data:'],
				'worker-src': ['self', 'blob:'],
				'connect-src': ['self', 'ws:', 'wss:', 'http:', 'https:'],
				'img-src': ['self', 'data:', 'blob:', 'https:'],
				'style-src': ['self', 'unsafe-inline', 'https://fonts.googleapis.com'],
				'font-src': ['self', 'https://fonts.gstatic.com']
			}
		},
		
		// Alias configuration
		alias: {
			$components: 'src/lib/components',
			$stores: 'src/lib/stores',
			$services: 'src/lib/services',
			$utils: 'src/lib/utils',
			$types: 'src/lib/types'
		}
	}
};

export default config;
