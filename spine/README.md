# Pluribus Spine (Registry + Coordination)

This repo isolates the core registry/spine layer for Pluribus. It contains the LMDB-backed CentralRegistry, module orchestration, and the CMP-LARGE service registry.

## Packages
- `@ark/core` — core types and URN/ring semantics
- `@ark/bus` — isomorphic event bus
- `@ark/registry` — service registry (genotype/phenotype)
- `@ark/spine` — CentralRegistry + SpineCoordinator

## Registry Specs
Canonical registry and schema assets live under `registry/`.

## Development

```bash
pnpm install
pnpm test
```

## Notes
This repo is extracted from `neo-pluribus` to serve as a stable System-of-Record for all holon siblings.
