import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    root: path.resolve(__dirname),
    include: ['src/__tests__/**/*.test.ts'],
  },
  resolve: {
    alias: {
      '@nesy/core': path.resolve(__dirname, '../core/src/index.ts'),
      '@nesy/pipeline': path.resolve(__dirname, '../pipeline/src/index.ts'),
      '@nesy/integration': path.resolve(__dirname, '../integration/src/index.ts'),
      '@nesy/prompt': path.resolve(__dirname, '../prompt/src/index.ts'),
      '@nesy/learning': path.resolve(__dirname, '../learning/src/index.ts'),
      '@nesy/puzzle': path.resolve(__dirname, '../puzzle/src/index.ts'),
    },
  },
});
