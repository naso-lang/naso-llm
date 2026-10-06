import { defineConfig } from 'vite';

// No `path` alias here: the app uses relative imports throughout, and pulling in
// `node:path` would drag @types/node into the browser-side tsconfig program.
export default defineConfig({
  root: '.',
  publicDir: 'public',
  base: '/',
  build: {
    outDir: 'dist',
    target: 'esnext',
    minify: 'esbuild',
    rollupOptions: {
      input: 'index.html',
    },
  },
  server: {
    port: 3000,
    host: true,
  },
  // The wasm-pack package is pre-bundled glue + .wasm; Vite's dep optimizer
  // would try to rewrite its import.meta.url-relative WASM fetch and break it.
  optimizeDeps: {
    exclude: ['nasoc-wasm'],
  },
  worker: {
    format: 'es',
  },
});
