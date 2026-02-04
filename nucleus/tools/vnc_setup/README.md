# Pluribus VNC Server Setup

VNC server for browser-based OAuth authentication to web services.

## Connection Details

- **Host**: 69.169.104.17
- **Port**: 5901
- **Display**: :1
- **Password**: See `/pluribus/.pluribus/vnc_password.txt`
- **Resolution**: 1280x1024

## Connect with VNC Client

```bash
# Using TigerVNC Viewer
vncviewer 69.169.104.17:5901

# Using RealVNC, TightVNC, or other clients
# Connect to: 69.169.104.17:5901
# Enter password when prompted
```

## Launch Browser for OAuth

From the server terminal or xterm in VNC session:

```bash
# Launch Firefox to ChatGPT
DISPLAY=:1 firefox https://chat.openai.com &

# Launch Firefox to Claude
DISPLAY=:1 firefox https://claude.ai &

# Launch Firefox to Gemini
DISPLAY=:1 firefox https://gemini.google.com &

# Or use the helper script
bash /pluribus/nucleus/tools/vnc_setup/launch_browser.sh https://chat.openai.com
```

## Service Management

```bash
# Check status
systemctl status pluribus-vnc.service

# Restart
systemctl restart pluribus-vnc.service

# Stop
systemctl stop pluribus-vnc.service

# Start
systemctl start pluribus-vnc.service

# View logs
journalctl -u pluribus-vnc.service -f
```

## Files

| File | Purpose |
|------|---------|
| `/etc/systemd/system/pluribus-vnc.service` | Systemd service file |
| `/root/.vnc/xstartup` | VNC session startup script |
| `/pluribus/.pluribus/vnc_passwd` | Encrypted VNC password |
| `/pluribus/.pluribus/vnc_password.txt` | Plain text password (for reference) |

## Security Notes

- VNC is listening on all interfaces (0.0.0.0:5901)
- Consider using SSH tunneling for secure access:
  ```bash
  ssh -L 5901:localhost:5901 user@69.169.104.17
  # Then connect VNC client to localhost:5901
  ```
- The password file is stored with restricted permissions (600)

## Troubleshooting

### Service won't start
```bash
journalctl -u pluribus-vnc.service -n 50 --no-pager
```

### Display issues
```bash
# Kill any stale VNC sessions
vncserver -kill :1
# Restart service
systemctl restart pluribus-vnc.service
```

### Browser won't launch
```bash
# Verify DISPLAY is set
export DISPLAY=:1
# Test with xterm
xterm &
```
