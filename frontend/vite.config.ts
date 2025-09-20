import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
// import { viteStaticCopy } from 'vite-plugin-static-copy'; // Disabled for performance

export default defineConfig({
	plugins: [
		sveltekit(),
		// Disable heavy WASM file copying for better initial load performance
		// These will be loaded on-demand when needed
		// viteStaticCopy({
		// 	targets: [
		// 		{
		// 			src: 'node_modules/@huggingface/transformers/dist/onnx-wasm-simd.wasm',
		// 			dest: 'static'
		// 		},
		// 		{
		// 			src: 'node_modules/@huggingface/transformers/dist/onnx-wasm-threaded.wasm',
		// 			dest: 'static'
		// 		},
		// 		{
		// 			src: 'node_modules/pyodide/pyodide.asm.wasm',
		// 			dest: 'static'
		// 		},
		// 		{
		// 			src: 'node_modules/pyodide/pyodide.asm.js',
		// 			dest: 'static'
		// 		}
		// 	]
		// })
	],
	server: {
		host: '0.0.0.0',
		port: 3000,
		proxy: {
			'/api': {
				target: 'http://localhost:8888',
				changeOrigin: true,
				secure: false
			},
			'/ws': {
				target: 'ws://localhost:8888',
				ws: true,
				changeOrigin: true
			},
			'/socket.io': {
				target: 'http://localhost:8888',
				changeOrigin: true,
				ws: true
			}
		}
	},
	preview: {
		host: '0.0.0.0',
		port: 4173,
		proxy: {
			'/api': {
				target: 'http://localhost:8888',
				changeOrigin: true,
				secure: false
			},
			'/ws': {
				target: 'ws://localhost:8888',
				ws: true,
				changeOrigin: true
			},
			'/socket.io': {
				target: 'http://localhost:8888',
				changeOrigin: true,
				ws: true
			}
		}
	},
	build: {
		target: 'esnext',
		rollupOptions: {
			output: {
				manualChunks: {
					// Core UI components - load immediately
					'vendor-core': ['svelte', '@sveltejs/kit'],
					// UI libraries - load on demand
					'vendor-ui': ['chart.js', 'mermaid'],
					// Heavy AI libraries - load only when needed (lazy)
					'vendor-ai-lazy': ['@huggingface/transformers', 'pyodide'],
					// Editor components - load when editor is used
					'vendor-editor': ['@tiptap/core', '@tiptap/starter-kit', 'codemirror'],
					// Utility libraries - smaller, can load early
					'vendor-utils': ['d3', 'fuse.js', 'marked', 'yaml']
				}
			}
		},
		chunkSizeWarningLimit: 2000 // Increase limit to reduce warnings
	},
	optimizeDeps: {
		// Only include essential dependencies for initial load
		include: [
			'socket.io-client', // Needed for real-time features
			'fuse.js', // Lightweight search
			'marked', // Markdown parsing
			'yaml' // Configuration parsing
		],
		// Exclude ALL heavy dependencies to prevent them from blocking initial load
		exclude: [
			// Flow and diagram libraries
			'@xyflow/svelte',
			'mermaid',
			'd3',

			// AI and ML libraries (VERY HEAVY)
			'@huggingface/transformers',
			'@mediapipe/tasks-vision',
			'pyodide',
			'@pyscript/core',

			// Rich text editor (heavy)
			'@tiptap/core',
			'@tiptap/starter-kit',
			'@tiptap/extension-bubble-menu',
			'@tiptap/extension-code-block-lowlight',
			'@tiptap/extension-drag-handle',
			'@tiptap/extension-file-handler',
			'@tiptap/extension-floating-menu',
			'@tiptap/extension-highlight',
			'@tiptap/extension-image',
			'@tiptap/extension-link',
			'@tiptap/extension-list',
			'@tiptap/extension-mention',
			'@tiptap/extension-table',
			'@tiptap/extension-typography',
			'@tiptap/extension-youtube',
			'@tiptap/extensions',
			'@tiptap/pm',
			'@tiptap/suggestion',

			// Code editors
			'codemirror',
			'@codemirror/lang-javascript',
			'@codemirror/lang-python',
			'@codemirror/language-data',
			'@codemirror/theme-one-dark',
			'codemirror-lang-elixir',
			'codemirror-lang-hcl',
			'lowlight',
			'highlight.js',

			// Charts and visualization
			'chart.js',
			'leaflet',
			'panzoom',

			// PDF and document processing
			'pdfjs-dist',
			'jspdf',
			'html2canvas-pro',

			// Heavy utility libraries
			'katex',
			'prosemirror-collab',
			'prosemirror-commands',
			'prosemirror-example-setup',
			'prosemirror-history',
			'prosemirror-keymap',
			'prosemirror-markdown',
			'prosemirror-model',
			'prosemirror-schema-basic',
			'prosemirror-schema-list',
			'prosemirror-state',
			'prosemirror-tables',
			'prosemirror-view'
		]
	},
	define: {
		global: 'globalThis'
	},
	worker: {
		format: 'es'
	}
});
