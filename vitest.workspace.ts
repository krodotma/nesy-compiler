import {{ defineWorkspace }} from 'vitest/config';

export default defineWorkspace([
  'packages/core',
  'packages/pipeline',
  'packages/prompt',
  'packages/integration',
  'packages/learning',
  'packages/puzzle',
]);
