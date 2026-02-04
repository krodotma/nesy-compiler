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
      '@nesy/temporal': path.resolve(__dirname, './src/index.ts'),
    },
  },
});
