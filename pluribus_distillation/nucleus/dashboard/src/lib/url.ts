/**
 * URL Utility Library for Pluribus Dashboard
 *
 * Provides consistent URL building and parsing for dashboard views.
 * Used by both client-side navigation and server-side routing.
 *
 * @module lib/url
 */

/**
 * Valid dashboard view names.
 * Keep in sync with routes/index.tsx activeView type.
 */
export const DASHBOARD_VIEWS = [
  'home',
  'studio',
  'bus',
  'events',
  'agents',
  'requests',
  'sota',
  'semops',
  'services',
  'rhizome',
  'git',
  'terminal',
  'plurichat',
  'webllm',
  'voice',
  'distill',
  'diagnostics',
  'generative',
  'browser-auth',
] as const;

export type DashboardView = (typeof DASHBOARD_VIEWS)[number];

/**
 * Parameters that can be passed to dashboard views via URL query params.
 */
export interface DashboardUrlParams {
  /** Event filter pattern (glob or regex) */
  filter?: string;
  /** Event topic to focus on */
  topic?: string;
  /** Request ID to highlight */
  reqId?: string;
  /** Trace ID for flow tracking */
  traceId?: string;
  /** Agent ID to focus on */
  agentId?: string;
  /** SOTA item ID */
  sotaId?: string;
  /** Git commit SHA */
  commit?: string;
  /** Generic search query */
  q?: string;
  /** Any additional custom params */
  [key: string]: string | undefined;
}

/**
 * Result of parsing a dashboard URL.
 */
export interface ParsedDashboardUrl {
  /** The view name (defaults to 'home' if invalid/missing) */
  view: DashboardView;
  /** Extracted parameters */
  params: DashboardUrlParams;
  /** Whether the URL had a valid view param */
  isValid: boolean;
}

/**
 * Build a dashboard URL for a specific view with optional parameters.
 *
 * @param view - The dashboard view to navigate to
 * @param params - Optional query parameters
 * @param base - Base URL (defaults to '/')
 * @returns Full URL string
 *
 * @example
 * buildDashboardUrl('events', { filter: 'strp.*' })
 * // => '/?view=events&filter=strp.*'
 *
 * buildDashboardUrl('git', { commit: 'abc123' })
 * // => '/?view=git&commit=abc123'
 */
export function buildDashboardUrl(
  view: DashboardView,
  params?: DashboardUrlParams,
  base: string = '/'
): string {
  const url = new URL(base, 'http://localhost');

  // Always set view param (except for home which is default)
  if (view !== 'home') {
    url.searchParams.set('view', view);
  }

  // Add additional params
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, value);
      }
    }
  }

  // Return pathname + search (without origin)
  return url.pathname + url.search;
}

/**
 * Parse the view name from a URL.
 *
 * @param url - URL string or URL object to parse
 * @returns The view name, defaulting to 'home' if invalid
 *
 * @example
 * parseViewFromUrl('/?view=events')
 * // => 'events'
 *
 * parseViewFromUrl('/?view=invalid')
 * // => 'home'
 */
export function parseViewFromUrl(url: string | URL): DashboardView {
  try {
    const urlObj = typeof url === 'string' ? new URL(url, 'http://localhost') : url;
    const view = urlObj.searchParams.get('view');

    if (!view) return 'home';

    // Validate against known views
    if ((DASHBOARD_VIEWS as readonly string[]).includes(view)) {
      return view as DashboardView;
    }

    return 'home';
  } catch {
    return 'home';
  }
}

/**
 * Extract deep-link parameters from a URL.
 *
 * @param url - URL string or URL object to parse
 * @returns Object containing all recognized parameters
 *
 * @example
 * getDeepLinkParams('/?view=events&filter=strp.*&topic=agent.status')
 * // => { filter: 'strp.*', topic: 'agent.status' }
 */
export function getDeepLinkParams(url: string | URL): DashboardUrlParams {
  try {
    const urlObj = typeof url === 'string' ? new URL(url, 'http://localhost') : url;
    const params: DashboardUrlParams = {};

    // Known deep-link parameter keys
    const knownKeys = [
      'filter',
      'topic',
      'reqId',
      'traceId',
      'agentId',
      'sotaId',
      'commit',
      'q',
    ];

    for (const key of knownKeys) {
      const value = urlObj.searchParams.get(key);
      if (value) {
        params[key] = value;
      }
    }

    // Also capture any other params (for extensibility)
    urlObj.searchParams.forEach((value, key) => {
      if (key !== 'view' && !knownKeys.includes(key) && value) {
        params[key] = value;
      }
    });

    return params;
  } catch {
    return {};
  }
}

/**
 * Parse a URL into view and parameters.
 *
 * @param url - URL string or URL object to parse
 * @returns ParsedDashboardUrl with view, params, and validity flag
 *
 * @example
 * parseDashboardUrl('/?view=agents&agentId=claude-opus')
 * // => { view: 'agents', params: { agentId: 'claude-opus' }, isValid: true }
 */
export function parseDashboardUrl(url: string | URL): ParsedDashboardUrl {
  const view = parseViewFromUrl(url);
  const params = getDeepLinkParams(url);

  // Check if URL had a valid view
  let isValid = true;
  try {
    const urlObj = typeof url === 'string' ? new URL(url, 'http://localhost') : url;
    const rawView = urlObj.searchParams.get('view');
    if (rawView && !(DASHBOARD_VIEWS as readonly string[]).includes(rawView)) {
      isValid = false;
    }
  } catch {
    isValid = false;
  }

  return { view, params, isValid };
}

/**
 * Check if a view name is valid.
 *
 * @param view - String to check
 * @returns True if valid dashboard view
 */
export function isValidView(view: string): view is DashboardView {
  return (DASHBOARD_VIEWS as readonly string[]).includes(view);
}

/**
 * Update the current URL with new view/params without full navigation.
 * Only works in browser context.
 *
 * @param view - New view (or null to keep current)
 * @param params - Parameters to set/update
 * @param replace - Use replaceState instead of pushState
 */
export function updateUrlParams(
  view?: DashboardView | null,
  params?: DashboardUrlParams,
  replace: boolean = false
): void {
  if (typeof window === 'undefined') return;

  const currentUrl = new URL(window.location.href);

  if (view !== null && view !== undefined) {
    if (view === 'home') {
      currentUrl.searchParams.delete('view');
    } else {
      currentUrl.searchParams.set('view', view);
    }
  }

  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null || value === '') {
        currentUrl.searchParams.delete(key);
      } else {
        currentUrl.searchParams.set(key, value);
      }
    }
  }

  const newUrl = currentUrl.pathname + currentUrl.search;

  if (replace) {
    window.history.replaceState({}, '', newUrl);
  } else {
    window.history.pushState({}, '', newUrl);
  }
}
