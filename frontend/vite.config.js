import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const target = process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  envDir: false,
  server: {
    host: '0.0.0.0',
    strictPort: true,
    watch: { usePolling: true },
    proxy: {
      '/api': {
        target,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/static/swagger': { target, changeOrigin: true },
      '/openapi.json': { target, changeOrigin: true },
    },
  },
});
