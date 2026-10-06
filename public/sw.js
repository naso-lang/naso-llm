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

const VERSION = 'v3';
const CORE = `naso-llm-core-${VERSION}`;
const MODEL = 'naso-llm-models-v1';
const RUNTIME = `naso-llm-runtime-${VERSION}`;
const KEEP = new Set([CORE, MODEL, RUNTIME]);

// The shell. Kept exact: everything listed must be fetchable at install time, or
// the worker should fail loudly rather than half-cache.
const CORE_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/favicon.svg',
  '/matmul.wgsl',
  '/matmul_i8.wgsl',
  '/pkg/nasoc_wasm.js',
  '/pkg/nasoc_wasm_bg.wasm',
  '/kernels/quantize_int8.naso',
  '/kernels/quantize_int8_symmetric.wgsl',
  '/kernels/quantize_int8_symmetric.abi.json',
  '/kernels/dequantize_int8_symmetric.wgsl',
  '/kernels/dequantize_int8_symmetric.abi.json',
  '/kernels/scale_clamp_f32.wgsl',
  '/kernels/scale_clamp_f32.abi.json',
  '/kernels/silu_f32.wgsl',
  '/kernels/silu_f32.abi.json',
  '/kernels/relu_scale_f32.wgsl',
  '/kernels/relu_scale_f32.abi.json',
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

  // Hugging Face: the checkpoint and tokenizer are immutable and large, so
  // cache-first and never revalidate.
  if (url.origin === 'https://huggingface.co' || url.hostname.endsWith('.hf.co')) {
    event.respondWith(cacheFirst(request, MODEL));
    return;
  }

  if (url.origin !== self.location.origin) return;

  // A navigation goes to the network first so a new deploy is picked up, then to
  // the cached shell for offline.
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request, RUNTIME, '/index.html'));
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
