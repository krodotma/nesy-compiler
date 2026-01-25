/**
 * LazyComponentWrapper - Lazy-load heavy components on demand
 * Part of WebUI Performance Optimization Phase 2
 */
import { component$, useSignal, useVisibleTask$, Slot } from '@builder.io/qwik';

interface LazyWrapperProps {
  loader: () => Promise<any>;
  fallback?: string;
}

/**
 * Wraps components for on-visibility lazy loading
 * Usage: <LazyComponentWrapper loader={() => import('../components/HeavyComponent')}>
 */
export const LazyComponentWrapper = component$<LazyWrapperProps>((props) => {
  const loaded = useSignal(false);
  const isVisible = useSignal(false);
  
  useVisibleTask$(({ track }) => {
    track(() => isVisible.value);
    if (isVisible.value && !loaded.value) {
      props.loader().then(() => {
        loaded.value = true;
      });
    }
  });

  return (
    <div 
      class={loaded.value ? '' : 'lazy-loading'}
      data-visible={isVisible.value}
    >
      {loaded.value ? <Slot /> : (
        <div class="loading-skeleton">
          {props.fallback || 'Loading...'}
        </div>
      )}
    </div>
  );
});
