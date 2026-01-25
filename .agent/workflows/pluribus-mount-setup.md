---
description: Pluribus SSHFS Auto-Mount Configuration
---

# Pluribus SSHFS Mount Setup

Auto-mounts `/pluribus` from VPS on every login/reboot.

## LaunchAgent Configuration

**File**: `~/Library/LaunchAgents/com.kroma.pluribus.mount.plist`

**Mount Point**: `/Users/kroma/Remote_Volumes/pluribus/remote_pluribus/`

**Target**: `root@kroma.live:/pluribus`

## SSHFS Options Used

```
-o reconnect                # Auto-reconnect on connection drop
-o ServerAliveInterval=15   # Send keepalive every 15s
-o ServerAliveCountMax=3    # Disconnect after 3 failed keepalives
-o auto_cache               # Enable FUSE caching
-o defer_permissions        # Defer permission checks to server
-o noappledouble            # Disable ._* AppleDouble files
-o nolocalcaches            # Disable local caching for consistency
```

## Management Commands

```bash
# Load (enable auto-mount)
launchctl load ~/Library/LaunchAgents/com.kroma.pluribus.mount.plist

# Unload (disable auto-mount)
launchctl unload ~/Library/LaunchAgents/com.kroma.pluribus.mount.plist

# Check if running
launchctl list | grep pluribus

# View logs
tail -f /tmp/pluribus-mount.log
tail -f /tmp/pluribus-mount-error.log

# Manual unmount if needed
diskutil unmount force /Users/kroma/Remote_Volumes/pluribus/remote_pluribus
```

## Troubleshooting

If mount fails or hangs after reboot:

1. Check logs: `/tmp/pluribus-mount-error.log`
2. Verify SSH key authentication: `ssh root@kroma.live`
3. Ensure macFUSE is installed: `ls /Library/Filesystems/macfuse.fs`
4. Restart LaunchAgent:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.kroma.pluribus.mount.plist
   launchctl load ~/Library/LaunchAgents/com.kroma.pluribus.mount.plist
   ```
