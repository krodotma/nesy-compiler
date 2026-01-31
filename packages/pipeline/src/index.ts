/**
 * @nesy/pipeline - Compilation pipeline stages
 *
 * 5 stages: PERCEIVE → GROUND → CONSTRAIN → VERIFY → EMIT
 */

export * from './stages/perceive';
export * from './stages/ground';
export * from './stages/constrain';
export * from './stages/verify';
export * from './stages/emit';
export * from './ir';
export * from './compiler';
