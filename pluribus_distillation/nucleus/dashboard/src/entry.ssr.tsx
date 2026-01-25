/**
 * Qwik City SSR Entry Point
 */

import { renderToStream, type RenderToStreamOptions } from '@builder.io/qwik/server';
import { manifest } from '@qwik-client-manifest';
import Root from './root';

export default function (opts: RenderToStreamOptions) {
  return renderToStream(<Root />, {
    manifest,
    ...opts,
    preloader: {
      // Reduced preloading to speed up initial render
      // WebLLM (5.3MB) should NOT be preloaded
      ssrPreloads: 2,
      ssrPreloadProbability: 0.5,
      maxIdlePreloads: 8,
    },
    containerAttributes: {
      lang: 'en-us',
      ...opts.containerAttributes,
    },
  });
}
