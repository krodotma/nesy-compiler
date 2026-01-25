---
description: Migrate Antigravity Playground to Pluribus Workspace
---

# Playground → Workspace Migration

Automates migration of Antigravity playground projects to the permanent Pluribus workspace.

## Usage

```bash
# Migrate current playground to Pluribus workspace
/migrate-to-pluribus <playground-name>

# Example:
/migrate-to-pluribus solitary-einstein
```

## What It Does

1. Creates workspace directory: `~/pluribus/` (Antigravity default)
2. Copies all files from playground
3. Preserves git history, configuration, and artifacts
4. Optionally removes playground after confirmation

## Manual Execution

```bash
PLAYGROUND_NAME="solitary-einstein"  # Replace with your playground

# Create workspace
mkdir -p ~/pluribus

# Copy all files
cp -r /Users/kroma/.gemini/antigravity/playground/$PLAYGROUND_NAME/* ~/pluribus/

# Verify
ls -la ~/pluribus/

# Open in Antigravity
# Use UI: Click "Move to folder" → Select ~/pluribus/
```

## Workspace Path

**Permanent location**: `~/pluribus/`

This is separate from:
- ❌ Antigravity playground: `/Users/kroma/.gemini/antigravity/playground/`
- ❌ VPS mount: `/Users/kroma/Remote_Volumes/pluribus/remote_pluribus/`

## Post-Migration

After migration, you can:
- Have multiple conversations in the same workspace
- Track changes with git
- Deploy to VPS via: `scp -r ~/pluribus/* root@kroma.live:/pluribus/`

## Arguments

`<playground-name>`: Name of playground to migrate (default: solitary-einstein)

## Notes

- Migration preserves all files and history
- Does NOT automatically delete playground (manual cleanup)
- Workspace supports multiple concurrent conversations
