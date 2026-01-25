/**
 * Qwik City Preview Entry Point
 *
 * Used by `vite preview` to serve production build locally.
 */

import { createQwikCity } from '@builder.io/qwik-city/middleware/node';
import qwikCityPlan from '@qwik-city-plan';
import render from './entry.ssr';

export default createQwikCity({
  render,
  qwikCityPlan: { ...qwikCityPlan, trailingSlash: false },
});
