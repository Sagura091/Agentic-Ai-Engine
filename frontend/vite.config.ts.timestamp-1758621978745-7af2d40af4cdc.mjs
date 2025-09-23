// vite.config.ts
import { sveltekit } from "file:///C:/Users/Bab18/Desktop/Agents/frontend/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///C:/Users/Bab18/Desktop/Agents/frontend/node_modules/vite/dist/node/index.js";
var vite_config_default = defineConfig({
  plugins: [sveltekit()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        secure: false
      },
      "/socket.io": {
        target: "http://localhost:8000",
        changeOrigin: true,
        secure: false,
        ws: true
      }
    }
  },
  build: {
    target: "esnext",
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor": ["svelte", "@sveltejs/kit"],
          "ui": ["bits-ui", "lucide-svelte"],
          "editor": ["codemirror", "@codemirror/lang-javascript", "@codemirror/lang-python"],
          "charts": ["chart.js"],
          "utils": ["dayjs", "uuid", "fuse.js"]
        }
      }
    }
  },
  optimizeDeps: {
    include: [
      "socket.io-client",
      "chart.js",
      "dayjs",
      "uuid",
      "fuse.js",
      "bits-ui",
      "lucide-svelte"
    ]
  },
  define: {
    global: "globalThis"
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxCYWIxOFxcXFxEZXNrdG9wXFxcXEFnZW50c1xcXFxmcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcQmFiMThcXFxcRGVza3RvcFxcXFxBZ2VudHNcXFxcZnJvbnRlbmRcXFxcdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0M6L1VzZXJzL0JhYjE4L0Rlc2t0b3AvQWdlbnRzL2Zyb250ZW5kL3ZpdGUuY29uZmlnLnRzXCI7aW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSAnQHN2ZWx0ZWpzL2tpdC92aXRlJztcbmltcG9ydCB7IGRlZmluZUNvbmZpZyB9IGZyb20gJ3ZpdGUnO1xuXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVDb25maWcoe1xuXHRwbHVnaW5zOiBbc3ZlbHRla2l0KCldLFxuXHRzZXJ2ZXI6IHtcblx0XHRob3N0OiAnMC4wLjAuMCcsXG5cdFx0cG9ydDogNTE3Myxcblx0XHRwcm94eToge1xuXHRcdFx0Jy9hcGknOiB7XG5cdFx0XHRcdHRhcmdldDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODAwMCcsXG5cdFx0XHRcdGNoYW5nZU9yaWdpbjogdHJ1ZSxcblx0XHRcdFx0c2VjdXJlOiBmYWxzZVxuXHRcdFx0fSxcblx0XHRcdCcvc29ja2V0LmlvJzoge1xuXHRcdFx0XHR0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgwMDAnLFxuXHRcdFx0XHRjaGFuZ2VPcmlnaW46IHRydWUsXG5cdFx0XHRcdHNlY3VyZTogZmFsc2UsXG5cdFx0XHRcdHdzOiB0cnVlXG5cdFx0XHR9XG5cdFx0fVxuXHR9LFxuXHRidWlsZDoge1xuXHRcdHRhcmdldDogJ2VzbmV4dCcsXG5cdFx0cm9sbHVwT3B0aW9uczoge1xuXHRcdFx0b3V0cHV0OiB7XG5cdFx0XHRcdG1hbnVhbENodW5rczoge1xuXHRcdFx0XHRcdCd2ZW5kb3InOiBbJ3N2ZWx0ZScsICdAc3ZlbHRlanMva2l0J10sXG5cdFx0XHRcdFx0J3VpJzogWydiaXRzLXVpJywgJ2x1Y2lkZS1zdmVsdGUnXSxcblx0XHRcdFx0XHQnZWRpdG9yJzogWydjb2RlbWlycm9yJywgJ0Bjb2RlbWlycm9yL2xhbmctamF2YXNjcmlwdCcsICdAY29kZW1pcnJvci9sYW5nLXB5dGhvbiddLFxuXHRcdFx0XHRcdCdjaGFydHMnOiBbJ2NoYXJ0LmpzJ10sXG5cdFx0XHRcdFx0J3V0aWxzJzogWydkYXlqcycsICd1dWlkJywgJ2Z1c2UuanMnXVxuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXHR9LFxuXHRvcHRpbWl6ZURlcHM6IHtcblx0XHRpbmNsdWRlOiBbXG5cdFx0XHQnc29ja2V0LmlvLWNsaWVudCcsXG5cdFx0XHQnY2hhcnQuanMnLFxuXHRcdFx0J2RheWpzJyxcblx0XHRcdCd1dWlkJyxcblx0XHRcdCdmdXNlLmpzJyxcblx0XHRcdCdiaXRzLXVpJyxcblx0XHRcdCdsdWNpZGUtc3ZlbHRlJ1xuXHRcdF1cblx0fSxcblx0ZGVmaW5lOiB7XG5cdFx0Z2xvYmFsOiAnZ2xvYmFsVGhpcydcblx0fVxufSk7XG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQWdULFNBQVMsaUJBQWlCO0FBQzFVLFNBQVMsb0JBQW9CO0FBRTdCLElBQU8sc0JBQVEsYUFBYTtBQUFBLEVBQzNCLFNBQVMsQ0FBQyxVQUFVLENBQUM7QUFBQSxFQUNyQixRQUFRO0FBQUEsSUFDUCxNQUFNO0FBQUEsSUFDTixNQUFNO0FBQUEsSUFDTixPQUFPO0FBQUEsTUFDTixRQUFRO0FBQUEsUUFDUCxRQUFRO0FBQUEsUUFDUixjQUFjO0FBQUEsUUFDZCxRQUFRO0FBQUEsTUFDVDtBQUFBLE1BQ0EsY0FBYztBQUFBLFFBQ2IsUUFBUTtBQUFBLFFBQ1IsY0FBYztBQUFBLFFBQ2QsUUFBUTtBQUFBLFFBQ1IsSUFBSTtBQUFBLE1BQ0w7QUFBQSxJQUNEO0FBQUEsRUFDRDtBQUFBLEVBQ0EsT0FBTztBQUFBLElBQ04sUUFBUTtBQUFBLElBQ1IsZUFBZTtBQUFBLE1BQ2QsUUFBUTtBQUFBLFFBQ1AsY0FBYztBQUFBLFVBQ2IsVUFBVSxDQUFDLFVBQVUsZUFBZTtBQUFBLFVBQ3BDLE1BQU0sQ0FBQyxXQUFXLGVBQWU7QUFBQSxVQUNqQyxVQUFVLENBQUMsY0FBYywrQkFBK0IseUJBQXlCO0FBQUEsVUFDakYsVUFBVSxDQUFDLFVBQVU7QUFBQSxVQUNyQixTQUFTLENBQUMsU0FBUyxRQUFRLFNBQVM7QUFBQSxRQUNyQztBQUFBLE1BQ0Q7QUFBQSxJQUNEO0FBQUEsRUFDRDtBQUFBLEVBQ0EsY0FBYztBQUFBLElBQ2IsU0FBUztBQUFBLE1BQ1I7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxJQUNEO0FBQUEsRUFDRDtBQUFBLEVBQ0EsUUFBUTtBQUFBLElBQ1AsUUFBUTtBQUFBLEVBQ1Q7QUFDRCxDQUFDOyIsCiAgIm5hbWVzIjogW10KfQo=
