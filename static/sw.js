// Minimal service worker — required by Chrome/Android for the
// "Add to Home Screen" install prompt to appear. We intentionally do
// NOT cache camera/API requests (this app is live and session-based),
// so this just passes requests straight through.
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  event.respondWith(fetch(event.request));
});
