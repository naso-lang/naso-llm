/**
 * Service worker: make the chat work offline after the first visit.
 *
 * Three caches, because the three kinds of asset have different lifetimes:
 *
 *   CORE    the shell: index.html, the bundled JS, the WASM compiler, the .naso
 *           sources, the hand-written matmul shaders. Versioned with the build,
 *           so a deploy replaces it wholesale. Precached on install.
 *   MODEL   the checkpoint weights and tokenizer from Hugging Face. Large (269 MB
 *           for SmolLM2-135M) and immutable, so kept across deploys -- losing it
 *           would mean re-downloading a quarter-gigabyte.
 *   RUNTIME everything else same-origin: cache-first with a background refresh.
 *
 * Strategy per request:
 *   * cross-origin (the Hugging Face checkpoint): cache-first, never revalidate.
 *     This is what makes a second visit and every later conversation offline.
 *   * navigation: network-first with the cached shell as fallback, so a deploy is
 *     picked up while online but the app still opens offline.
 *   * same-origin static: cache-first.
 */

const VERSION = 'v4';
const CORE = `naso-llm-core-${VERSION}`;
const MODEL = 'naso-llm-models-v1';
const RUNTIME = `naso-llm-runtime-${VERSION}`;
const KEEP = new Set([CORE, MODEL, RUNTIME]);

/**
 * The app is not always served from a domain root: GitHub Pages serves it from
 * https://<owner>.github.io/<repo>/, so a hardcoded '/kernels/x' would 404 and
 * silently break the shell cache. BASE is derived from the worker's own URL
 * (`/naso-llm/sw.js` -> `/naso-llm/`, `/sw.js` -> `/`), so the same file is
 * correct under every base without a build-time substitution.
 */
const BASE = new URL('./', self.location.href).pathname;
const at = (p) => new URL(p.replace(/^\//, ''), new URL(BASE, self.location.origin)).pathname;

// The shell. Kept exact: everything listed must be fetchable at install time, or
// the worker should fail loudly rather than half-cache.
const CORE_ASSETS = [
  at('/'),
  at('/index.html'),
  at('/manifest.json'),
  at('/favicon.svg'),
  at('/matmul.wgsl'),
  at('/matmul_i8.wgsl'),
  at('/pkg/nasoc_wasm.js'),
  at('/pkg/nasoc_wasm_bg.wasm'),
  at('/kernels/quantize_int8.naso'),
  at('/kernels/quantize_int8_symmetric.wgsl'),
  at('/kernels/quantize_int8_symmetric.abi.json'),
  at('/kernels/dequantize_int8_symmetric.wgsl'),
  at('/kernels/dequantize_int8_symmetric.abi.json'),
  at('/kernels/scale_clamp_f32.wgsl'),
  at('/kernels/scale_clamp_f32.abi.json'),
  at('/kernels/silu_f32.wgsl'),
  at('/kernels/silu_f32.abi.json'),
  at('/kernels/relu_scale_f32.wgsl'),
  at('/kernels/relu_scale_f32.abi.json'),
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CORE);
    // Cache each asset individually so one 404 does not abort the install and
    // leave a half-populated shell.
    await Promise.all(CORE_ASSETS.map(async (url) => {
      try {
        const res = await fetch(url, { cache: 'reload' });
        if (res.ok) await cache.put(url, res);
        else console.warn('[sw] skip', url, res.status);
      } catch (e) {
        console.warn('[sw] skip', url, String(e));
      }
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((n) => !KEEP.has(n)).map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

/** Cache-first, with the network result filling the cache when it is used. */
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok || res.type === 'opaque') await cache.put(request, res.clone());
  return res;
}

/**
 * Hugging Face: serve a stored entry if there is one, else fetch and return the
 * response WITHOUT cloning it into the cache.
 *
 * The page owns caching these files (it tees the response it is already
 * downloading into this same cache), so letting the worker clone and store too
 * would hold a second full copy of a 269 MB body in memory and duplicate the
 * write. `cache.match` finds page-written entries, so offline revisits still work.
 *
 * This is a simplification, not a bug fix: the clone that used to be here was
 * suspected of causing `TypeError: Failed to fetch`, but that was disproven --
 * the old handler passes the same large fetch on both localhost and the live
 * site. Those failures were Hugging Face's resolve rate limit.
 */
async function hfCacheFirst(request) {
  const cache = await caches.open(MODEL);
  const hit = await cache.match(request);
  if (hit) return hit;
  return fetch(request);
}

/** Network-first: fresh when online, cached shell when not. */
async function networkFirst(request, cacheName, fallback) {
  const cache = await caches.open(cacheName);
  try {
    const res = await fetch(request);
    if (res.ok) await cache.put(request, res.clone());
    return res;
  } catch {
    const hit = await cache.match(request);
    if (hit) return hit;
    if (fallback) {
      const shell = await cache.match(fallback);
      if (shell) return shell;
    }
    throw new Error('offline and not cached');
  }
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);

  // Hugging Face: serve a stored entry if there is one, else a pure pass-through.
  //
  // The page owns caching these files: `fetchWithProgress` tees the response it
  // is already downloading into this same cache. Letting the worker also clone
  // and store here would keep a second full copy of a 269 MB body in memory and
  // duplicate the write, for no benefit. `cache.match` finds page-written
  // entries regardless of who stored them, so offline revisits still work.
  //
  // (An earlier version of this comment claimed this clone was causing a
  // `TypeError: Failed to fetch`. That was disproven: the old handler passes the
  // same large fetch on localhost and on the live site. The real cause of those
  // failures was Hugging Face's resolve rate limit, exhausted by repeated test
  // downloads of the checkpoint.)
  if (url.origin === 'https://huggingface.co' || url.hostname.endsWith('.hf.co')) {
    event.respondWith(hfCacheFirst(request));
    return;
  }

  if (url.origin !== self.location.origin) return;

  // A navigation goes to the network first so a new deploy is picked up, then to
  // the cached shell for offline.
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request, RUNTIME, at('/index.html')));
    return;
  }

  event.respondWith(cacheFirst(request, RUNTIME));
});

// The page can ask the worker to warm the model cache, or to clear it.
self.addEventListener('message', (event) => {
  const data = event.data;
  if (!data || typeof data !== 'object') return;

  if (data.type === 'CACHE_MODEL' && data.url) {
    event.waitUntil((async () => {
      const cache = await caches.open(MODEL);
      // The body is streamed to disk by the browser, never buffered here, so a
      // 269 MB checkpoint does not have to fit in memory as one ArrayBuffer.
      const res = await fetch(data.url);
      if (res.ok) await cache.put(data.url, res);
      event.source?.postMessage({ type: 'MODEL_CACHED', url: data.url, ok: res.ok });
    })());
  }

  if (data.type === 'CLEAR_CACHE') {
    event.waitUntil(caches.keys().then((names) => Promise.all(names.map((n) => caches.delete(n)))));
  }
});
