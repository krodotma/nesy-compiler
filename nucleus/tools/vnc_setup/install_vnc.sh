#!/bin/bash
# VNC Server Setup Script for Pluribus
# Purpose: Enable browser-based OAuth authentication for web services
# Usage: sudo bash /pluribus/pluribus_next/tools/vnc_setup/install_vnc.sh

set -e

echo "=== Pluribus VNC Server Setup ==="

# Configuration
VNC_PORT=5901
VNC_DISPLAY=:1
VNC_GEOMETRY="1280x1024"
VNC_DEPTH=24
VNC_PASSWD_DIR="/pluribus/.pluribus"
VNC_PASSWD_FILE="${VNC_PASSWD_DIR}/vnc_passwd"

# 1. Install required packages
echo "[1/6] Installing packages..."
apt-get update
apt-get install -y \
    tigervnc-standalone-server \
    tigervnc-common \
    x11-xserver-utils \
    dbus-x11 \
    xterm \
    openbox \
    firefox || apt-get install -y chromium-browser

# 2. Create VNC password
echo "[2/6] Creating VNC password..."
mkdir -p "${VNC_PASSWD_DIR}"

# Generate a random password and store it
VNC_PASSWORD=$(openssl rand -base64 12 | tr -dc 'a-zA-Z0-9' | head -c 12)
echo "VNC Password: ${VNC_PASSWORD}"
echo "${VNC_PASSWORD}" > "${VNC_PASSWD_DIR}/vnc_password.txt"
chmod 600 "${VNC_PASSWD_DIR}/vnc_password.txt"

# Create VNC password file using vncpasswd
echo "${VNC_PASSWORD}" | vncpasswd -f > "${VNC_PASSWD_FILE}"
chmod 600 "${VNC_PASSWD_FILE}"

echo "Password saved to: ${VNC_PASSWD_DIR}/vnc_password.txt"

# 3. Create VNC xstartup script
echo "[3/6] Creating VNC startup script..."
mkdir -p /root/.vnc
cat > /root/.vnc/xstartup << 'XSTARTUP'
#!/bin/bash
# VNC Startup Script for Pluribus Browser Authentication

unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

# Start dbus
eval $(dbus-launch --sh-syntax)
export DBUS_SESSION_BUS_ADDRESS

# Set up display
export DISPLAY=:1

# Start window manager (openbox is lightweight)
openbox &

# Launch xterm for debugging
xterm -geometry 80x24+10+10 &

echo "VNC session ready for browser authentication"
XSTARTUP
chmod +x /root/.vnc/xstartup

# 4. Create systemd service
echo "[4/6] Creating systemd service..."
cat > /etc/systemd/system/pluribus-vnc.service << 'SERVICE'
[Unit]
Description=Pluribus VNC Server for Browser Authentication
After=network.target

[Service]
Type=simple
User=root
Environment=DISPLAY=:1
Environment=HOME=/root

# Kill any existing VNC on display :1
ExecStartPre=-/usr/bin/vncserver -kill :1

# Start VNC server
ExecStart=/usr/bin/vncserver :1 \
    -geometry 1280x1024 \
    -depth 24 \
    -rfbport 5901 \
    -localhost no \
    -PasswordFile /pluribus/.pluribus/vnc_passwd \
    -fg

ExecStop=/usr/bin/vncserver -kill :1

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

# 5. Start and enable service
echo "[5/6] Starting VNC service..."
systemctl daemon-reload
systemctl enable pluribus-vnc.service
systemctl start pluribus-vnc.service

# Wait for startup
sleep 3

# 6. Verify
echo "[6/6] Verifying VNC server..."
systemctl status pluribus-vnc.service --no-pager || true
ss -tlnp | grep 5901 || netstat -tlnp | grep 5901 || true

echo ""
echo "=== VNC Server Setup Complete ==="
echo ""
echo "Connection Details:"
echo "  Host: $(hostname -I | awk '{print $1}'):5901"
echo "  Display: :1"
echo "  Password: ${VNC_PASSWORD}"
echo "  Password file: ${VNC_PASSWD_DIR}/vnc_password.txt"
echo ""
echo "Connect using a VNC client (RealVNC, TigerVNC Viewer, etc.)"
echo "  Example: vncviewer $(hostname -I | awk '{print $1}'):5901"
echo ""
echo "To launch Firefox in the VNC session:"
echo "  DISPLAY=:1 firefox &"
echo ""
