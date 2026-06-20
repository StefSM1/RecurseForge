import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    hmr: {
      port: 5174,
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8100',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://localhost:8100',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
