---
description: Pluribus SSH Filesystem Rules (CRITICAL)
---

# Pluribus SSH Filesystem Protocol

> **SEE ALSO**: `.agent/workflows/dkin-paip-registry.md` for protocol version specs.

## TOPOLOGY (MEMORIZE THIS)

```
/Users/kroma/Remote_Volumes/pluribus/remote_pluribus/   <- REAL SSHFS MOUNT (macFUSE)
    └── .pluribus/bus/events.ndjson                     <- VPS Bus
    └── nucleus/tools/                                  <- VPS Tools
    └── ...

/Users/kroma/.gemini/.../solitary-einstein/pluribus/remote_pluribus/  <- LOCAL SHADOW (DO NOT USE)
```

**CRITICAL**: Always use `/Users/kroma/Remote_Volumes/pluribus/remote_pluribus/` for VPS writes, NOT the path inside your workspace.

## CRITICAL RULES

1. **`remote_pluribus` IS THE VPS**: Everything inside `pluribus/remote_pluribus/` is a LIVE SSH mount to the VPS. Writes here go directly to the VPS filesystem.

2. **NEVER CREATE LOCAL SHADOWS**: If `remote_pluribus` appears empty or missing subdirectories, DO NOT create them locally. This creates a "shadow" that blocks the real mount.

3. **ALWAYS PUSH TO REMOTE**: Any Pluribus bus, DKIN, or tool operations MUST target `pluribus/remote_pluribus/...` to reach the VPS.

4. **CHECK BEFORE CREATING**: If a path looks missing, ASK THE USER before creating directories. It may be a mount timing issue.

## VERIFICATION

Before writing to any Pluribus path:
```bash
ls -la pluribus/remote_pluribus/
# Should show VPS contents, not empty/local files
```

If it shows files YOU created (not from VPS), you have shadowed the mount. DELETE your local creations and remount.
