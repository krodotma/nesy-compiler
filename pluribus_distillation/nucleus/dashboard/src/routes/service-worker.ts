/// <reference lib="webworker" />

const CACHE_NAME = 'pluribus-dashboard-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/manifest.json',
  '/global.css',
];

const sw = self as unknown as ServiceWorkerGlobalScope;

sw.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

sw.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

sw.addEventListener('fetch', (event) => {
  // Skip cross-origin requests
  if (!event.request.url.startsWith(sw.location.origin)) return;

  // Skip API requests (except maybe static ones, but keep it simple)
  if (event.request.url.includes('/api/')) return;

  event.respondWith(
    caches.match(event.request).then((response) => {
      if (response) {
        return response;
      }
      return fetch(event.request).then((response) => {
        // Cache successful GET requests for assets
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }
        
        // Cache fonts, images, JS, CSS
        if (event.request.destination === 'font' || 
            event.request.destination === 'image' || 
            event.request.destination === 'script' || 
            event.request.destination === 'style') {
            const responseToCache = response.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, responseToCache);
            });
        }
        
        return response;
      });
    })
  );
});
