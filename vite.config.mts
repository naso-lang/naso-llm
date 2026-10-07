/**
 * Vite config, plus a small plugin that serves the LOCAL pre-quantised
 * checkpoint (NPQ1) during dev and carries it into `dist` on build.
 *
 * Lives in a `.mts` file: the browser tsc program excludes it, so no Node types
 * leak into app code (which must not use `process`/`fs`), while Vite still finds
 * it. The few host callbacks are given minimal local types for the same reason.
 */
import { defineConfig } from 'vite';
// `createRequire` keeps the node builtins out of the static module graph: this
// config runs under Node (Vite), but the same folder is compiled by the browser
// tsc program, which must not resolve `node:*` or see Node globals.
import { createRequire } from 'node:module';

const require_ = createRequire(import.meta.url);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const fs: any = require_('node:fs');
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const pathMod: any = require_('node:path');
const { existsSync, statSync, mkdirSync, copyFileSync, createReadStream } = fs;
const { join } = pathMod;

/**
 * `base` is read from the environment so one build config works for both:
 *   * default '/'          -- `npm run dev`, `npm run preview`
 *   * GH_PAGES=1           -- the GitHub Pages deploy (see `npm run deploy`)
 * The deployed app is NOT at a domain root (GitHub Pages serves it from
 * https://<owner>.github.io/<repo>/), so every asset URL must carry the prefix.
 * The service worker derives its URLs at runtime too (public/sw.js), so it keeps
 * working under either base without a second copy of this value.
 *
 * Read through globalThis rather than a bare `process`: the browser tsc program
 * has no `process` type.
 */
const env = ((globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env) ?? {};
const base = env.GH_PAGES ? '/naso-llm/' : (env.BASE ?? '/');

/**
 * Where the artifact built by `npm run quantize:build` lives. Overridable so a
 * CI job or a different model can point elsewhere.
 */
const NPQ_SRC = env.NASO_NPQ ?? '/var/tmp/model.int8.npq';

/** The one artifact that exists today; the URL filename is its identity. */
const NPQ_NAME = 'smollm2-135m-int8.npq';

type Req = { url?: string; method?: string };
type Res = { statusCode: number; setHeader(k: string, v: string): void; end(b?: string): void };
type Next = () => void;
type Handler = (req: Req, res: Res, next: Next) => void;
/** `connect`'s `use` overloads are route-aware; only the 1-arg form is used. */
type Middlewares = { use(fn: Handler): unknown };

/**
 * Serve `${base}model/<name>` from the build output directory.
 *
 * Hosting the artifact is a deployment choice, so it is deliberately NOT
 * committed (a 163 MB binary in git is the wrong home for it) and is not in
 * `public/` (which is size-capped and copied wholesale). For local work it only
 * needs to be reachable at that URL. If the file is missing the request 404s and
 * the app falls back to the f32 safetensors path, so this is never fatal.
 */
function serveArtifact(mw: Middlewares, distDir: string) {
  const mount = `${base.replace(/\/$/, '')}/model/`;
  mw.use((req, res, next) => {
    const path = (req.url ?? '').split('?')[0];
    if (!path.startsWith(mount)) return next();
    const name = path.slice(mount.length);
    if (name !== NPQ_NAME) return next();
    // Source of truth differs by mode: the dev artifact in /var/tmp, or whatever
    // `writeBundle` copied into dist/model.
    const file = existsSync(NPQ_SRC)
      ? NPQ_SRC
      : (existsSync(join(distDir, 'model', name)) ? join(distDir, 'model', name) : null);
    if (!file) return next();
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Length', String(statSync(file).size));
    res.setHeader('Accept-Ranges', 'bytes');
    // HEAD must report metadata only. Streaming `createReadStream(file).pipe(res)`
    // for a HEAD would still read all 163 MB off disk and push it down a socket
    // the browser discards -- so the "is a local artifact present?" probe stays
    // one cheap round-trip instead of a free re-download.
    if (req.method === 'HEAD') { res.end(); return; }
    createReadStream(file).pipe(res);
  });
}

export default defineConfig({
  root: '.',
  publicDir: 'public',
  base,
  plugins: [
    {
      name: 'naso-local-model',
      configureServer(server: { middlewares: Middlewares }) {
        serveArtifact(server.middlewares, 'dist');
      },
      configurePreviewServer(server: { middlewares: Middlewares }) {
        serveArtifact(server.middlewares, 'dist');
      },
      // Copy the artifact beside the bundle so a LOCAL `vite preview` can serve
      // it via the same /model/ route. Gated OFF for GH Pages deploys: the
      // production site serves the f32 safetensors (no local artifact is hosted
      // there -- Cloudflare is dev-only), and shipping a 163 MB binary into the
      // `dist` that gh-pages uploads is wasted quota for no benefit.
      writeBundle(options: { dir?: string }) {
        if (!existsSync(NPQ_SRC)) return;
        if (env.GH_PAGES) return;
        const outDir = join(options.dir ?? 'dist', 'model');
        mkdirSync(outDir, { recursive: true });
        copyFileSync(NPQ_SRC, join(outDir, NPQ_NAME));
        console.log(`[naso-local-model] packaged ${NPQ_NAME} -> ${outDir} (set NASO_PACKAGE_MODEL=1 to disable this copy)`);
      },
    },
  ],
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