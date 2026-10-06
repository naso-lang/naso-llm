import { defineConfig } from 'vite';

// No `path` alias here: the app uses relative imports throughout, and pulling in
// `node:path` would drag @types/node into the browser-side tsconfig program.
//
// The deployed app is NOT at a domain root: GitHub Pages serves it from
// https://<owner>.github.io/<repo>/, so every asset URL must carry that prefix.
// `base` is read from the environment so one build config works for both:
//   * default '/'          -- `npm run dev`, `npm run preview`
//   * GH_PAGES=1           -- the GitHub Pages deploy (see `npm run deploy`)
// The service worker also derives its URLs at runtime (see public/sw.js), so it
// keeps working under either base without a second copy of this value.
//
// Read through globalThis rather than a bare `process`: this file is part of the
// browser tsc program, whose `types` deliberately excludes node.
const env = ((globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env) ?? {};
const base = env.GH_PAGES ? '/naso-llm/' : (env.BASE ?? '/');

export default defineConfig({
  root: '.',
  publicDir: 'public',
  base,
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
