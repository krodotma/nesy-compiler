# Pluribus Dialogos

Dialogos is the message control system and trace pipeline for Pluribus. This repo extracts the core services, protocols, and tests so it can evolve independently while remaining a first-class Spine customer.

## Contents
- `services/` — dialogos runtime services and indexer
- `protocols/` — Dialogos protocol and omega topic pairs
- `deploy/` — systemd units
- `tests/` — core integration/system tests
- `docs/` — UI integration notes
- `ui/` — VIL/UX surface

## Relationship to Spine
Dialogos should register itself and its trace schemas in the Spine registry so all holon siblings can query and reason over dialogos state.
