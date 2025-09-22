import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '0.0.0.0',
		port: 5173,
		proxy: {
			'/api': {
				target: 'http://localhost:8888',
				changeOrigin: true,
				secure: false
			},
			'/socket.io': {
				target: 'http://localhost:8888',
				changeOrigin: true,
				secure: false,
				ws: true
			}
		}
	},
	build: {
		target: 'esnext',
		rollupOptions: {
			output: {
				manualChunks: {
					'vendor': ['svelte', '@sveltejs/kit'],
					'ui': ['bits-ui', 'lucide-svelte'],
					'editor': ['codemirror', '@codemirror/lang-javascript', '@codemirror/lang-python'],
					'charts': ['chart.js'],
					'utils': ['dayjs', 'uuid', 'fuse.js']
				}
			}
		}
	},
	optimizeDeps: {
		include: [
			'socket.io-client',
			'chart.js',
			'dayjs',
			'uuid',
			'fuse.js',
			'bits-ui',
			'lucide-svelte'
		]
	},
	define: {
		global: 'globalThis'
	}
});
