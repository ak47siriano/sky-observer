const CACHE_VERSION = "v3";
const STATIC_CACHE = `sky-observer-static-${CACHE_VERSION}`;
const RUNTIME_CACHE = `sky-observer-runtime-${CACHE_VERSION}`;
const TILE_CACHE = `sky-observer-tiles-${CACHE_VERSION}`;
const MAX_TILE_ENTRIES = 250;

const APP_SHELL = ["/", "/index.html", "/logo.png", "/manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(STATIC_CACHE)
      .then((cache) => cache.addAll(APP_SHELL))
      .catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  const activeCaches = new Set([STATIC_CACHE, RUNTIME_CACHE, TILE_CACHE]);
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => !activeCaches.has(k)).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

async function trimCache(cacheName, maxEntries) {
  try {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    if (keys.length > maxEntries) {
      const excess = keys.length - maxEntries;
      await Promise.all(keys.slice(0, excess).map((req) => cache.delete(req)));
    }
  } catch {
    // ignore cache trim errors
  }
}

function offlineFallbackResponse(request) {
  if (request.mode === "navigate") {
    return caches.match("/index.html").then(
      (cached) =>
        cached ||
        new Response(
          "<!doctype html><html><body style='background:#07090f;color:#e2e8f0;font-family:sans-serif;display:grid;place-items:center;height:100vh;margin:0'><div><h2>Sky Observer — Offline</h2><p>Waiting for local backend connection...</p></div></body></html>",
          { status: 503, headers: { "Content-Type": "text/html" } }
        )
    );
  }
  return new Response("", { status: 503, statusText: "Offline" });
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Never cache live API, WebSocket, or dynamic image/overlay streams
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/ws/")) {
    return;
  }

  // Map tiles (RainViewer / OSM / Carto): stale-while-revalidate with bounded LRU-style trim
  if (
    url.hostname.includes("tile") ||
    url.hostname.includes("openstreetmap.org") ||
    url.hostname.includes("cartocdn.com") ||
    url.hostname.includes("rainviewer.com")
  ) {
    event.respondWith(
      caches.open(TILE_CACHE).then(async (cache) => {
        const cached = await cache.match(request);
        const networkPromise = fetch(request)
          .then((res) => {
            if (res && (res.status === 200 || res.type === "opaque")) {
              cache.put(request, res.clone()).then(() => trimCache(TILE_CACHE, MAX_TILE_ENTRIES));
            }
            return res;
          })
          .catch(() => cached || offlineFallbackResponse(request));
        return cached || networkPromise;
      })
    );
    return;
  }

  // Immutable hashed Vite assets (/assets/*): cache-first
  if (url.pathname.startsWith("/assets/")) {
    event.respondWith(
      caches.match(request).then(
        (cached) =>
          cached ||
          fetch(request)
            .then((res) => {
              if (res && res.status === 200) {
                const clone = res.clone();
                caches.open(STATIC_CACHE).then((cache) => cache.put(request, clone));
              }
              return res;
            })
            .catch(() => offlineFallbackResponse(request))
      )
    );
    return;
  }

  // App shell & navigation: network-first with cached fallback
  event.respondWith(
    fetch(request)
      .then((res) => {
        if (res && res.status === 200) {
          const clone = res.clone();
          caches.open(RUNTIME_CACHE).then((cache) => cache.put(request, clone));
        }
        return res;
      })
      .catch(() => caches.match(request).then((cached) => cached || offlineFallbackResponse(request)))
  );
});
