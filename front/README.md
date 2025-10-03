Isocard depends on loading `jolt-physics` (WebAssembly) at runtime.  
You need a bundler that:

- Excludes `jolt-physics` from dependency pre-bundling
- Serves `.wasm` files as static assets

### Recommended

- [Vite](https://vitejs.dev/) (React, Vue, Svelte, Solid, Astro, etc.)
- Ionic React (uses Vite under the hood)
- Next.js 14+ (with Turbopack/Webpack configured for WASM)

### Not Supported

- Create React App (CRA) without ejecting
- React Native (use a WebView if you want mobile)
- Any bundler that cannot handle `.wasm` as static files


ionic start simple-iso blank --type=react --vite --capacitor


bun add jolt-physics three @openfluke/isocard

# If developing isocard locally
bun add isocard@link:isocard


bun run dev



# change vite.config.ts

```
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
```