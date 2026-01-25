import type { RequestHandler } from '@builder.io/qwik-city';

export const onRequest: RequestHandler = async ({ headers, url, next }) => {
  // Enable cross-origin isolation (required for SharedArrayBuffer).
  // Use 'credentialless' instead of 'require-corp' for COEP to allow
  // loading external resources like noVNC iframe without CORS headers.
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
  await next();
};
