#!/bin/bash
# Pluribus VPS /tmp Protection Setup Script
# Run after boot recovery to prevent future /tmp explosions
#
# Usage: sudo bash install_tmp_protection.sh
#
# What this does:
# 1. Installs tmpjanitor.py (Python cleanup script)
# 2. Installs systemd timer (runs every 5 min)
# 3. Installs tmpfiles.d config (system-level cleanup)
# 4. Optionally configures /tmp as tmpfs (RAM-backed, hard limit)
# 5. Sets up cron fallback

set -e

PLURIBUS_ROOT="${PLURIBUS_ROOT:-/pluribus}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Pluribus /tmp Protection Setup ==="
echo ""

# Check root
if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root"
   exit 1
fi

# === 1. Install tmpjanitor.py ===
echo "[1/5] Installing tmpjanitor.py..."
if [[ -f "$SCRIPT_DIR/../nucleus/tools/tmpjanitor.py" ]]; then
    cp "$SCRIPT_DIR/../nucleus/tools/tmpjanitor.py" "$PLURIBUS_ROOT/nucleus/tools/"
    chmod +x "$PLURIBUS_ROOT/nucleus/tools/tmpjanitor.py"
    echo "  ✓ Installed to $PLURIBUS_ROOT/nucleus/tools/tmpjanitor.py"
else
    echo "  ⚠ tmpjanitor.py not found in script dir, skipping"
fi

# === 2. Install systemd units ===
echo "[2/5] Installing systemd units..."
if [[ -f "$SCRIPT_DIR/systemd/pluribus-tmpjanitor.service" ]]; then
    cp "$SCRIPT_DIR/systemd/pluribus-tmpjanitor.service" /etc/systemd/system/
    cp "$SCRIPT_DIR/systemd/pluribus-tmpjanitor.timer" /etc/systemd/system/
    
    systemctl daemon-reload
    systemctl enable pluribus-tmpjanitor.timer
    systemctl start pluribus-tmpjanitor.timer
    
    echo "  ✓ Installed and enabled pluribus-tmpjanitor.timer"
else
    echo "  ⚠ systemd units not found, creating inline..."
    
    cat > /etc/systemd/system/pluribus-tmpjanitor.service << 'EOF'
[Unit]
Description=Pluribus /tmp Janitorial Service

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /pluribus/nucleus/tools/tmpjanitor.py
User=root
Nice=19
EOF

    cat > /etc/systemd/system/pluribus-tmpjanitor.timer << 'EOF'
[Unit]
Description=Run Pluribus tmpjanitor every 5 minutes

[Timer]
OnBootSec=1min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

    systemctl daemon-reload
    systemctl enable pluribus-tmpjanitor.timer
    systemctl start pluribus-tmpjanitor.timer
    echo "  ✓ Created and enabled timer"
fi

# === 3. Install tmpfiles.d config ===
echo "[3/5] Installing tmpfiles.d config..."
if [[ -f "$SCRIPT_DIR/tmpfiles.d/pluribus.conf" ]]; then
    cp "$SCRIPT_DIR/tmpfiles.d/pluribus.conf" /etc/tmpfiles.d/
    echo "  ✓ Installed /etc/tmpfiles.d/pluribus.conf"
else
    echo "  ⚠ Creating inline tmpfiles.d config..."
    cat > /etc/tmpfiles.d/pluribus.conf << 'EOF'
# Clean /tmp files older than 1 hour
d /tmp 1777 root root 1h
d /var/tmp 1777 root root 6h
R /tmp/npm-* - - - 10m
R /tmp/vite-* - - - 10m
R /tmp/core.* - - - 0
EOF
    echo "  ✓ Created /etc/tmpfiles.d/pluribus.conf"
fi

# === 4. Optional: Configure tmpfs for /tmp ===
echo "[4/5] Configuring tmpfs for /tmp..."

# Check if already tmpfs
if mount | grep -q "tmpfs on /tmp"; then
    echo "  ✓ /tmp is already tmpfs"
else
    read -p "  Mount /tmp as tmpfs with 512M limit? (recommended) [y/N]: " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        # Add to fstab if not present
        if ! grep -q "tmpfs /tmp" /etc/fstab; then
            echo "tmpfs /tmp tmpfs defaults,noatime,nosuid,nodev,mode=1777,size=512M 0 0" >> /etc/fstab
            echo "  ✓ Added tmpfs entry to /etc/fstab"
        fi
        
        # Mount now (if possible)
        echo "  ⚠ tmpfs will take effect on next boot"
        echo "  ⚠ To mount now: umount /tmp && mount /tmp (requires no files in use)"
    else
        echo "  → Skipped tmpfs setup"
    fi
fi

# === 5. Install cron fallback ===
echo "[5/5] Installing cron fallback..."
CRON_LINE="*/10 * * * * root /usr/bin/python3 $PLURIBUS_ROOT/nucleus/tools/tmpjanitor.py --json >> /var/log/pluribus-tmpjanitor-cron.log 2>&1"

if ! grep -q "tmpjanitor" /etc/crontab 2>/dev/null; then
    echo "$CRON_LINE" >> /etc/crontab
    echo "  ✓ Added cron fallback (every 10 min)"
else
    echo "  ✓ Cron entry already exists"
fi

# === Create log file ===
touch /var/log/pluribus-tmpjanitor.log
chmod 644 /var/log/pluribus-tmpjanitor.log

# === Summary ===
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Protection layers installed:"
echo "  1. tmpjanitor.py      → Python cleanup (100MB limit, 1hr age)"
echo "  2. systemd timer      → Runs every 5 min"
echo "  3. tmpfiles.d         → System-level cleanup on boot"
echo "  4. tmpfs (if enabled) → RAM-backed /tmp with 512M hard limit"
echo "  5. cron fallback      → Every 10 min backup"
echo ""
echo "To test: python3 $PLURIBUS_ROOT/nucleus/tools/tmpjanitor.py --check"
echo "To force cleanup: python3 $PLURIBUS_ROOT/nucleus/tools/tmpjanitor.py"
echo ""
echo "Logs: /var/log/pluribus-tmpjanitor.log"
