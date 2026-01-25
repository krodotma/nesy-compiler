/**
 * Vite Configuration for Pluribus Dashboard
 *
 * Supports:
 * - Qwik City SSR/SSG
 * - Lit Element compilation
 * - TypeScript paths
 * - Tailwind CSS
 */

import { defineConfig } from 'vite';
import { qwikVite } from '@builder.io/qwik/optimizer';
import { qwikCity } from '@builder.io/qwik-city/vite';
import tsconfigPaths from 'vite-tsconfig-paths';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

function crossOriginIsolationHeaders() {
  // Use 'credentialless' instead of 'require-corp' to allow loading
  // external resources (like noVNC iframe) without CORS headers.
  // SharedArrayBuffer still works with credentialless COEP.
  const headers = {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'credentialless',
  } as const;

  return {
    name: 'pluribus-cross-origin-isolation-headers',
    configureServer(server: { middlewares: { use: Function } }) {
      server.middlewares.use((req: { url: string }, res: { setHeader: (k: string, v: string) => void }, next: Function) => {
        // Apply isolation headers
        for (const [key, value] of Object.entries(headers)) {
          res.setHeader(key, value);
        }
        
        // Ensure index.html is served as text/html (fix for regression where it might be served as text/plain or missing type)
        if (req.url === '/' || req.url?.endsWith('.html')) {
          res.setHeader('Content-Type', 'text/html; charset=utf-8');
        }
        
        next();
      });
    },
    configurePreviewServer(server: { middlewares: { use: Function } }) {
      server.middlewares.use((req: { url: string }, res: { setHeader: (k: string, v: string) => void }, next: Function) => {
        for (const [key, value] of Object.entries(headers)) {
          res.setHeader(key, value);
        }
        if (req.url === '/' || req.url?.endsWith('.html')) {
          res.setHeader('Content-Type', 'text/html; charset=utf-8');
        }
        next();
      });
    },
  };
}

export default defineConfig(() => {
  const here = dirname(fileURLToPath(import.meta.url));
  const e2ePort = (process.env.E2E_PORT || '').trim();
  return {
    define: {
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
      __BUILD_COMMIT__: JSON.stringify(
        process.env.GIT_COMMIT ||
          process.env.VERCEL_GIT_COMMIT_SHA ||
          process.env.GITHUB_SHA ||
          'dev'
      ),
      __E2E__: JSON.stringify(!!e2ePort),
    },
    plugins: [
      crossOriginIsolationHeaders(),
      qwikCity(),
      qwikVite({
        debug: false,
        verbose: false,
      }),
      tsconfigPaths(),
    ],
    server: {
      host: '0.0.0.0',
      port: parseInt(process.env.VITE_PORT || process.env.PORT || '5173'),
      hmr: {
        overlay: false,
      },
      // Warm up frequently used files to reduce cold start (Vite 5+)
      warmup: {
        clientFiles: [
          'src/routes/index.tsx',
          'src/routes/layout.tsx',
          'src/components/nav/BicameralNav.tsx',
          'src/components/supermotd.tsx',
          'src/components/EventVisualization.tsx',
          'src/components/DiagnosticsPanel.tsx',
          'src/components/LoadingStage.tsx',
        ],
      },
      // Allow HTTPS proxy domains (official domains + fallbacks)
      allowedHosts: [
        'localhost',
        '127.0.0.1',
        ...(e2ePort ? [`localhost:${e2ePort}`, `127.0.0.1:${e2ePort}`] : []),
        '69.169.104.17',
        'kroma.live',
        '.kroma.live',
        'kareem.movie',
        '69-169-104-17.sslip.io',
      ],
      fs: {
        allow: [
          // Permit importing canonical proto schemas living in `nucleus/proto`.
          resolve(here, '..'),
        ],
      },
      proxy: {
        // WebSocket proxies for terminal and chat
        '^/terminal(?:$|/)': { target: 'ws://127.0.0.1:9200', ws: true, changeOrigin: true },
        '^/crush(?:$|/)': { target: 'ws://127.0.0.1:9200', ws: true, changeOrigin: true },
        '^/plurichat(?:$|/)': { target: 'ws://127.0.0.1:9200', ws: true, changeOrigin: true },
        // Bus WebSocket - primary connection for events
        '/ws/bus': { target: 'ws://127.0.0.1:9200', ws: true, changeOrigin: true },
        // World Router - unified inference gateway
        '^/v1(?:$|/)': { target: 'http://127.0.0.1:8080', changeOrigin: true },
        '^/v1beta(?:$|/)': { target: 'http://127.0.0.1:8080', changeOrigin: true },
        '^/cua(?:$|/)': { target: 'http://127.0.0.1:8080', changeOrigin: true },
        '^/identity(?:$|/)': { target: 'http://127.0.0.1:8080', changeOrigin: true },
        '^/a2a(?:$|/)': { target: 'http://127.0.0.1:8080', changeOrigin: true },
        '^/\\.well-known/agent\\.json$': { target: 'http://127.0.0.1:8080', changeOrigin: true },
        // REST APIs - session, agents, io-buffer
        '/api/session': {
          target: 'http://127.0.0.1:9201',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/session/, '/session'),
        },
        '/api/agents': {
          target: 'http://127.0.0.1:9201',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/agents/, '/agents'),
        },
        '/api/io-buffer': {
          target: 'http://127.0.0.1:9201',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/io-buffer/, '/io-buffer'),
        },
        '/api/bus/events': {
          target: 'http://127.0.0.1:9201',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/bus\/events/, '/events'),
        },
        '/api/metrics/snapshot': {
          target: 'http://127.0.0.1:9201',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/metrics\/snapshot/, '/metrics/snapshot'),
        },
        '/api/dialogos/health': {
          target: 'http://127.0.0.1:9201',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/dialogos\/health/, '/dialogos/health'),
        },
        // Tools API - git, fs, sota, module
        '/api/git': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/git/, '/git') },
        '/api/fs': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/fs/, '/fs') },
        '/api/sota': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/sota/, '/sota') },
        '/api/metatest': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/metatest/, '/metatest') },
        '/api/module': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/module/, '/module') },
        // SemOps schema + CRUD (git_server.py)
        '/api/semops': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/semops/, '/semops') },
        // PBLBC purge + cache actions
        '/api/pblbc': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/pblbc/, '/pblbc') },
        // Rhizome API (MCP JSON-RPC)
        '/api/rhizome': { target: 'http://127.0.0.1:9100', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/rhizome/, '') },
        // Emit endpoint (bus-bridge REST: POST /publish)
        '/api/emit': { target: 'http://127.0.0.1:9201', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/emit/, '/publish') },
        // Portal asset ingest endpoint (bus-bridge REST: POST /portal/assets)
        '/api/portal/assets': { target: 'http://127.0.0.1:9201', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/portal\/assets/, '/portal/assets') },
        // Browser daemon status API
        '/api/browser': { target: 'http://127.0.0.1:9300', changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/browser/, '/browser') },
      },
      headers: {
        'Cache-Control': 'no-store',
      },
    },
    preview: {
      port: 4173,
      headers: {
        'Cache-Control': 'public, max-age=600',
      },
    },
    build: {
      target: 'es2022',
      sourcemap: true,
      rollupOptions: {
        output: {
          // Split heavy libraries into separate chunks for better initial load
          manualChunks(id) {
            // Auralux neural voice (ONNX runtime) - lazy loaded
            if (id.includes('/auralux/')) {
              return 'auralux';
            }
            // Lit elements
            if (id.includes('node_modules/lit') || id.includes('node_modules/@lit')) {
              return 'lit-core';
            }
            // Three.js (474KB) - only load when SphereAtlas is visible
            if (id.includes('node_modules/three')) {
              return 'three';
            }
            // WebLLM/MLC-AI (5.5MB) - only load when WebLLM widget is used
            if (id.includes('node_modules/@mlc-ai') || id.includes('node_modules/web-llm')) {
              return 'webllm';
            }
            // xterm (276KB) - only load when terminal is visible
            if (id.includes('node_modules/xterm')) {
              return 'xterm';
            }
            // Packery layout library
            if (id.includes('node_modules/packery')) {
              return 'packery';
            }
          },
        },
      },
    },
    optimizeDeps: {
      include: ['lit', '@lit/reactive-element'],
      // Pre-bundle to reduce dev startup time
      entries: ['src/routes/index.tsx', 'src/routes/layout.tsx'],
    },
    // Support both .ts and .tsx
    esbuild: {
      jsxFactory: 'h',
      jsxFragment: 'Fragment',
    },
  };
});
