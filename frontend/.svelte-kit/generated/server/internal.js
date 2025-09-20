
import root from '../root.svelte';
import { set_building, set_prerendering } from '__sveltekit/environment';
import { set_assets } from '__sveltekit/paths';
import { set_manifest, set_read_implementation } from '__sveltekit/server';
import { set_private_env, set_public_env } from '../../../node_modules/@sveltejs/kit/src/runtime/shared-server.js';

export const options = {
	app_template_contains_nonce: false,
	csp: {"mode":"auto","directives":{"worker-src":["self","blob:"],"connect-src":["self","ws:","wss:","http:","https:"],"font-src":["self","https://fonts.gstatic.com"],"img-src":["self","data:","blob:","https:"],"script-src":["self","unsafe-inline","unsafe-eval","blob:","data:"],"style-src":["self","unsafe-inline","https://fonts.googleapis.com"],"upgrade-insecure-requests":false,"block-all-mixed-content":false},"reportOnly":{"upgrade-insecure-requests":false,"block-all-mixed-content":false}},
	csrf_check_origin: true,
	csrf_trusted_origins: [],
	embedded: false,
	env_public_prefix: 'PUBLIC_',
	env_private_prefix: '',
	hash_routing: false,
	hooks: null, // added lazily, via `get_hooks`
	preload_strategy: "modulepreload",
	root,
	service_worker: false,
	service_worker_options: null,
	templates: {
		app: ({ head, body, assets, nonce, env }) => "<!doctype html>\n<html lang=\"en\" class=\"dark\">\n\t<head>\n\t\t<meta charset=\"utf-8\" />\n\t\t<link rel=\"icon\" href=\"" + assets + "/favicon.png\" />\n\t\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n\t\t<meta name=\"theme-color\" content=\"#0f172a\" />\n\t\t<meta name=\"description\" content=\"Revolutionary AI Agent Builder Platform - Create, test, and deploy intelligent agents with the most advanced visual workflow builder.\" />\n\t\t<meta name=\"keywords\" content=\"AI, agents, workflow, automation, machine learning, artificial intelligence, no-code, low-code\" />\n\t\t\n\t\t<!-- Open Graph / Facebook -->\n\t\t<meta property=\"og:type\" content=\"website\" />\n\t\t<meta property=\"og:title\" content=\"Agentic AI - Revolutionary Agent Builder\" />\n\t\t<meta property=\"og:description\" content=\"The most extraordinary frontend for building intelligent AI agents with visual workflows.\" />\n\t\t<meta property=\"og:image\" content=\"" + assets + "/og-image.png\" />\n\t\t\n\t\t<!-- Twitter -->\n\t\t<meta property=\"twitter:card\" content=\"summary_large_image\" />\n\t\t<meta property=\"twitter:title\" content=\"Agentic AI - Revolutionary Agent Builder\" />\n\t\t<meta property=\"twitter:description\" content=\"The most extraordinary frontend for building intelligent AI agents with visual workflows.\" />\n\t\t<meta property=\"twitter:image\" content=\"" + assets + "/twitter-image.png\" />\n\t\t\n\t\t<!-- Preload critical fonts -->\n\t\t<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />\n\t\t<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />\n\t\t<link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap\" rel=\"stylesheet\" />\n\t\t<link href=\"https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@100..800&display=swap\" rel=\"stylesheet\" />\n\t\t\n\t\t<!-- Preload critical resources -->\n\t\t<!-- Service worker disabled for now to avoid 404 errors -->\n\t\t\n\t\t<!-- Global error handler -->\n\t\t<script>\n\t\t\twindow.addEventListener('error', (event) => {\n\t\t\t\tconsole.error('Global error:', event.error);\n\t\t\t\t// Send to error tracking service\n\t\t\t});\n\t\t\t\n\t\t\twindow.addEventListener('unhandledrejection', (event) => {\n\t\t\t\tconsole.error('Unhandled promise rejection:', event.reason);\n\t\t\t\t// Send to error tracking service\n\t\t\t});\n\t\t</script>\n\t\t\n\t\t<!-- Performance monitoring -->\n\t\t<script>\n\t\t\twindow.addEventListener('load', () => {\n\t\t\t\tif ('performance' in window) {\n\t\t\t\t\tconst perfData = performance.getEntriesByType('navigation')[0];\n\t\t\t\t\tconsole.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');\n\t\t\t\t}\n\t\t\t});\n\t\t</script>\n\t\t\n\t\t" + head + "\n\t</head>\n\t<body \n\t\tdata-sveltekit-preload-data=\"hover\" \n\t\tclass=\"bg-dark-900 text-white antialiased overflow-x-hidden\"\n\t\tstyle=\"font-family: 'Inter', sans-serif;\"\n\t>\n\t\t<!-- Loading screen -->\n\t\t<div id=\"initial-loading\" class=\"fixed inset-0 z-50 flex items-center justify-center bg-dark-900\">\n\t\t\t<div class=\"text-center\">\n\t\t\t\t<div class=\"relative\">\n\t\t\t\t\t<!-- Animated logo/spinner -->\n\t\t\t\t\t<div class=\"w-16 h-16 mx-auto mb-4 relative\">\n\t\t\t\t\t\t<div class=\"absolute inset-0 rounded-full border-4 border-accent-blue/20\"></div>\n\t\t\t\t\t\t<div class=\"absolute inset-0 rounded-full border-4 border-transparent border-t-accent-blue animate-spin\"></div>\n\t\t\t\t\t\t<div class=\"absolute inset-2 rounded-full bg-gradient-to-br from-accent-blue to-accent-purple opacity-20\"></div>\n\t\t\t\t\t</div>\n\t\t\t\t\t\n\t\t\t\t\t<!-- Loading text -->\n\t\t\t\t\t<h1 class=\"text-2xl font-bold text-gradient mb-2\">Agentic AI</h1>\n\t\t\t\t\t<p class=\"text-dark-400 text-sm\">Initializing revolutionary agent builder...</p>\n\t\t\t\t\t\n\t\t\t\t\t<!-- Progress bar -->\n\t\t\t\t\t<div class=\"w-64 h-1 bg-dark-800 rounded-full mx-auto mt-4 overflow-hidden\">\n\t\t\t\t\t\t<div class=\"h-full bg-gradient-to-r from-accent-blue to-accent-purple animate-pulse\"></div>\n\t\t\t\t\t</div>\n\t\t\t\t</div>\n\t\t\t</div>\n\t\t</div>\n\t\t\n\t\t<!-- Main app container -->\n\t\t<div style=\"display: contents\">" + body + "</div>\n\t\t\n\t\t<!-- Remove loading screen when app is ready -->\n\t\t<script>\n\t\t\tdocument.addEventListener('DOMContentLoaded', () => {\n\t\t\t\t// Remove loading screen immediately when DOM is ready for faster perceived performance\n\t\t\t\tconst loadingScreen = document.getElementById('initial-loading');\n\t\t\t\tif (loadingScreen) {\n\t\t\t\t\t// Much faster transition\n\t\t\t\t\tloadingScreen.style.opacity = '0';\n\t\t\t\t\tloadingScreen.style.transition = 'opacity 0.2s ease-out';\n\t\t\t\t\tsetTimeout(() => {\n\t\t\t\t\t\tloadingScreen.remove();\n\t\t\t\t\t}, 200); // Reduced from 500ms to 200ms\n\t\t\t\t}\n\t\t\t});\n\t\t</script>\n\t\t\n\t\t<!-- Global styles for better UX -->\n\t\t<style>\n\t\t\t/* Smooth scrolling */\n\t\t\thtml {\n\t\t\t\tscroll-behavior: smooth;\n\t\t\t}\n\t\t\t\n\t\t\t/* Custom scrollbar */\n\t\t\t::-webkit-scrollbar {\n\t\t\t\twidth: 8px;\n\t\t\t\theight: 8px;\n\t\t\t}\n\t\t\t\n\t\t\t::-webkit-scrollbar-track {\n\t\t\t\tbackground: #1e293b;\n\t\t\t}\n\t\t\t\n\t\t\t::-webkit-scrollbar-thumb {\n\t\t\t\tbackground: #475569;\n\t\t\t\tborder-radius: 4px;\n\t\t\t}\n\t\t\t\n\t\t\t::-webkit-scrollbar-thumb:hover {\n\t\t\t\tbackground: #64748b;\n\t\t\t}\n\t\t\t\n\t\t\t/* Selection styles */\n\t\t\t::selection {\n\t\t\t\tbackground: rgba(59, 130, 246, 0.3);\n\t\t\t\tcolor: white;\n\t\t\t}\n\t\t\t\n\t\t\t/* Focus styles */\n\t\t\t:focus-visible {\n\t\t\t\toutline: 2px solid #3b82f6;\n\t\t\t\toutline-offset: 2px;\n\t\t\t}\n\t\t\t\n\t\t\t/* Disable text selection on UI elements */\n\t\t\t.no-select {\n\t\t\t\t-webkit-user-select: none;\n\t\t\t\t-moz-user-select: none;\n\t\t\t\t-ms-user-select: none;\n\t\t\t\tuser-select: none;\n\t\t\t}\n\t\t\t\n\t\t\t/* Glass morphism utility */\n\t\t\t.glass-effect {\n\t\t\t\tbackground: rgba(255, 255, 255, 0.05);\n\t\t\t\tbackdrop-filter: blur(10px);\n\t\t\t\tborder: 1px solid rgba(255, 255, 255, 0.1);\n\t\t\t}\n\t\t</style>\n\t</body>\n</html>\n",
		error: ({ status, message }) => "<!doctype html>\n<html lang=\"en\">\n\t<head>\n\t\t<meta charset=\"utf-8\" />\n\t\t<title>" + message + "</title>\n\n\t\t<style>\n\t\t\tbody {\n\t\t\t\t--bg: white;\n\t\t\t\t--fg: #222;\n\t\t\t\t--divider: #ccc;\n\t\t\t\tbackground: var(--bg);\n\t\t\t\tcolor: var(--fg);\n\t\t\t\tfont-family:\n\t\t\t\t\tsystem-ui,\n\t\t\t\t\t-apple-system,\n\t\t\t\t\tBlinkMacSystemFont,\n\t\t\t\t\t'Segoe UI',\n\t\t\t\t\tRoboto,\n\t\t\t\t\tOxygen,\n\t\t\t\t\tUbuntu,\n\t\t\t\t\tCantarell,\n\t\t\t\t\t'Open Sans',\n\t\t\t\t\t'Helvetica Neue',\n\t\t\t\t\tsans-serif;\n\t\t\t\tdisplay: flex;\n\t\t\t\talign-items: center;\n\t\t\t\tjustify-content: center;\n\t\t\t\theight: 100vh;\n\t\t\t\tmargin: 0;\n\t\t\t}\n\n\t\t\t.error {\n\t\t\t\tdisplay: flex;\n\t\t\t\talign-items: center;\n\t\t\t\tmax-width: 32rem;\n\t\t\t\tmargin: 0 1rem;\n\t\t\t}\n\n\t\t\t.status {\n\t\t\t\tfont-weight: 200;\n\t\t\t\tfont-size: 3rem;\n\t\t\t\tline-height: 1;\n\t\t\t\tposition: relative;\n\t\t\t\ttop: -0.05rem;\n\t\t\t}\n\n\t\t\t.message {\n\t\t\t\tborder-left: 1px solid var(--divider);\n\t\t\t\tpadding: 0 0 0 1rem;\n\t\t\t\tmargin: 0 0 0 1rem;\n\t\t\t\tmin-height: 2.5rem;\n\t\t\t\tdisplay: flex;\n\t\t\t\talign-items: center;\n\t\t\t}\n\n\t\t\t.message h1 {\n\t\t\t\tfont-weight: 400;\n\t\t\t\tfont-size: 1em;\n\t\t\t\tmargin: 0;\n\t\t\t}\n\n\t\t\t@media (prefers-color-scheme: dark) {\n\t\t\t\tbody {\n\t\t\t\t\t--bg: #222;\n\t\t\t\t\t--fg: #ddd;\n\t\t\t\t\t--divider: #666;\n\t\t\t\t}\n\t\t\t}\n\t\t</style>\n\t</head>\n\t<body>\n\t\t<div class=\"error\">\n\t\t\t<span class=\"status\">" + status + "</span>\n\t\t\t<div class=\"message\">\n\t\t\t\t<h1>" + message + "</h1>\n\t\t\t</div>\n\t\t</div>\n\t</body>\n</html>\n"
	},
	version_hash: "1x3blq"
};

export async function get_hooks() {
	let handle;
	let handleFetch;
	let handleError;
	let handleValidationError;
	let init;
	

	let reroute;
	let transport;
	

	return {
		handle,
		handleFetch,
		handleError,
		handleValidationError,
		init,
		reroute,
		transport
	};
}

export { set_assets, set_building, set_manifest, set_prerendering, set_private_env, set_public_env, set_read_implementation };
