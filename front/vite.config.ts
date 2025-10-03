/// <reference types="vitest" />

import legacy from '@vitejs/plugin-legacy'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    legacy()
  ],
  optimizeDeps: {
    exclude: ['@openfluke/portal'] // Prevents pre-bundling; allows relative asset imports (?raw/?url) and import.meta.url to resolve to actual dist/ files
  },
  assetsInclude: ['**/*.wasm'], // Treats .wasm as static assets (emits correctly in builds)
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  }
})