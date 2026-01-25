
import { generateChromaRegistry } from '../lib/art/chroma-registry';

// Web Worker for generating the chroma registry without blocking the main thread.
// Phase 1 Step 3: Performance Optimization

self.onmessage = (event) => {
  if (event.data.type === 'GENERATE_REGISTRY') {
    const start = performance.now();
    const registry = generateChromaRegistry();
    const duration = performance.now() - start;
    
    self.postMessage({
      type: 'REGISTRY_GENERATED',
      payload: {
        registry,
        duration
      }
    });
  }
};
