---
description: Antigravity Shell Fix (tselector -c flag)
---

# Antigravity Tselector Shell Fix

When Antigravity runs inside tmux (via the `antigravity` session), the shell is set to `tselector` which is a tmux session manager - NOT a command executor.

## Problem

Antigravity's `run_command` tool invokes commands via `$SHELL -c "command"`. When `tselector` is the shell, it doesn't understand `-c` and fails with:
```
tselector: unknown arguments: -c <command>
```

## Solution

Added `-c` flag handling to `/Users/kroma/.local/bin/tselector` that forwards commands to `/bin/bash`:

```bash
-c)
  # Forward -c commands to a real shell for command execution
  # This allows tools like Antigravity run_command to work
  shift
  exec /bin/bash -c "$*"
  ;;
```

## Canonical Source

The fixed `tselector` is tracked at:
- **Local**: `/Users/kroma/.local/bin/tselector`
- **Repo**: `/Users/kroma/pluribus_evolution/nucleus/tools/tselector`
- **VPS**: `/pluribus/nucleus/tools/tselector`

## Restore If Needed

```bash
cp /pluribus/nucleus/tools/tselector ~/.local/bin/tselector
chmod +x ~/.local/bin/tselector
```

## Related

- This fix was made on 2026-01-24 during conversation `1b3cf5c9-e0af-4416-898a-fc3f6d02409c`
- Emitted bus event documenting fix
