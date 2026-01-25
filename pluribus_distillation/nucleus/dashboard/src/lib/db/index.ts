/**
 * Database Module Exports
 *
 * Provides Prisma-style type-safe database operations for Pluribus entities.
 */

export {
  // Main client
  PrismaClient,
  getPrismaClient,

  // Entity types
  type Service,
  type ServiceInstance,
  type BusEvent,
  type Agent,
  type Artifact,
  type SotaTool,

  // Utility functions
  query,
  syncBusEvents,
} from './prisma_adapter';
