#!/bin/bash
# noVNC + websockify startup script for Pluribus
# Provides web-based VNC access through Caddy HTTPS proxy
#
# Usage: ./novnc_start.sh [start|stop|restart|status]
#
# This script:
# 1. Ensures TigerVNC is running on display :1 (port 5901)
# 2. Starts websockify to bridge WebSocket to VNC
# 3. websockify listens on port 6080 and serves noVNC static files

set -e

NOVNC_DIR="/usr/share/novnc"
VNC_PORT="5901"
WEBSOCKIFY_PORT="6080"
VNC_DISPLAY=":1"
VNC_RESOLUTION="1920x1080"
VNC_DEPTH="24"
VNC_PASSWORD_FILE="/pluribus/.pluribus/vnc_password.txt"
VNC_PASSWD_FILE="/pluribus/.pluribus/vnc_passwd"
PIDFILE_VNC="/tmp/novnc-vnc.pid"
PIDFILE_WS="/tmp/novnc-websockify.pid"
LOGFILE="/tmp/novnc.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

start_vnc() {
    # Check if VNC is already running
    if pgrep -f "Xtigervnc $VNC_DISPLAY" > /dev/null 2>&1 || pgrep -f "Xvnc $VNC_DISPLAY" > /dev/null 2>&1; then
        log "VNC server already running on display $VNC_DISPLAY"
        return 0
    fi

    log "Starting TigerVNC server on display $VNC_DISPLAY..."

    # Ensure VNC password exists (and matches the operator-visible file).
    # Use the `-PasswordFile` flow so short PINs work (TigerVNC accepts them via `vncpasswd -f`).
    # Set NOVNC_KEEP_PASSWD=1 to skip re-seeding.
    if [ ! -f "$VNC_PASSWD_FILE" ] || [ "${NOVNC_KEEP_PASSWD:-0}" != "1" ]; then
        log "Ensuring VNC password files..."
        mkdir -p /pluribus/.pluribus

        VNC_PASSWORD=""
        if [ -f "$VNC_PASSWORD_FILE" ]; then
            VNC_PASSWORD="$(tr -d '\r\n' < "$VNC_PASSWORD_FILE" | head -c 64)"
        fi
        if [ -z "$VNC_PASSWORD" ]; then
            VNC_PASSWORD="pluribus"
            if command -v openssl >/dev/null 2>&1; then
                VNC_PASSWORD="$(openssl rand -base64 18 | tr -dc 'a-zA-Z0-9' | head -c 12)"
            fi
            printf "%s\n" "$VNC_PASSWORD" > "$VNC_PASSWORD_FILE"
        fi

        # Generate the TigerVNC password file (binary, 8 bytes).
        # NOTE: vncpasswd truncates passwords to 8 chars at auth time.
        printf "%s\n" "$VNC_PASSWORD" | vncpasswd -f > "$VNC_PASSWD_FILE"
        chmod 600 "$VNC_PASSWORD_FILE" "$VNC_PASSWD_FILE" 2>/dev/null || true
    fi

    # Start VNC server
    vncserver $VNC_DISPLAY \
        -geometry "$VNC_RESOLUTION" \
        -depth "$VNC_DEPTH" \
        -rfbport "$VNC_PORT" \
        -localhost no \
        -PasswordFile "$VNC_PASSWD_FILE" \
        -UseBlacklist 0 \
        -SecurityTypes VncAuth \
        -xstartup /root/.vnc/xstartup \
        2>&1 | tee -a "$LOGFILE"

    sleep 2

    # Verify it started
    if pgrep -f "Xtigervnc $VNC_DISPLAY" > /dev/null 2>&1 || pgrep -f "Xvnc $VNC_DISPLAY" > /dev/null 2>&1; then
        (pgrep -f "Xtigervnc $VNC_DISPLAY" || pgrep -f "Xvnc $VNC_DISPLAY") > "$PIDFILE_VNC"
        log "VNC server started successfully (PID: $(cat $PIDFILE_VNC))"
        return 0
    else
        log "ERROR: Failed to start VNC server"
        return 1
    fi
}

start_websockify() {
    # Check if websockify is already running
    if pgrep -f "websockify.*$WEBSOCKIFY_PORT" > /dev/null 2>&1; then
        log "websockify already running on port $WEBSOCKIFY_PORT"
        return 0
    fi

    log "Starting websockify on port $WEBSOCKIFY_PORT -> VNC port $VNC_PORT..."

    # Start websockify with noVNC web directory
    # --web serves the noVNC static files
    websockify \
        --web="$NOVNC_DIR" \
        --wrap-mode=ignore \
        $WEBSOCKIFY_PORT \
        localhost:$VNC_PORT \
        >> "$LOGFILE" 2>&1 &

    WS_PID=$!
    echo $WS_PID > "$PIDFILE_WS"

    sleep 2

    # Verify it started
    if kill -0 $WS_PID 2>/dev/null; then
        log "websockify started successfully (PID: $WS_PID)"
        return 0
    else
        log "ERROR: Failed to start websockify"
        return 1
    fi
}

stop_vnc() {
    log "Stopping VNC server..."
    if [ -f "$PIDFILE_VNC" ]; then
        kill $(cat "$PIDFILE_VNC") 2>/dev/null || true
        rm -f "$PIDFILE_VNC"
    fi
    vncserver -kill $VNC_DISPLAY 2>/dev/null || true
    pkill -f "Xtigervnc $VNC_DISPLAY" 2>/dev/null || true
    pkill -f "Xvnc $VNC_DISPLAY" 2>/dev/null || true
    log "VNC server stopped"
}

stop_websockify() {
    log "Stopping websockify..."
    if [ -f "$PIDFILE_WS" ]; then
        kill $(cat "$PIDFILE_WS") 2>/dev/null || true
        rm -f "$PIDFILE_WS"
    fi
    pkill -f "websockify.*$WEBSOCKIFY_PORT" 2>/dev/null || true
    log "websockify stopped"
}

status() {
    echo "=== noVNC Status ==="
    echo ""

    if pgrep -f "Xtigervnc $VNC_DISPLAY" > /dev/null 2>&1 || pgrep -f "Xvnc $VNC_DISPLAY" > /dev/null 2>&1; then
        echo "VNC Server: RUNNING (display $VNC_DISPLAY, port $VNC_PORT)"
        pgrep -a -f "Xtigervnc $VNC_DISPLAY" || pgrep -a -f "Xvnc $VNC_DISPLAY"
    else
        echo "VNC Server: STOPPED"
    fi

    echo ""

    if pgrep -f "websockify.*$WEBSOCKIFY_PORT" > /dev/null 2>&1; then
        echo "websockify: RUNNING (port $WEBSOCKIFY_PORT -> $VNC_PORT)"
        pgrep -a -f "websockify.*$WEBSOCKIFY_PORT"
    else
        echo "websockify: STOPPED"
    fi

    echo ""
    echo "Access URLs:"
    echo "  Local: http://localhost:$WEBSOCKIFY_PORT/vnc.html"
    echo "  Via Caddy: https://kroma.live/vnc/vnc.html"
}

case "${1:-start}" in
    start)
        log "=== Starting noVNC services ==="
        start_vnc
        start_websockify
        status
        ;;
    stop)
        log "=== Stopping noVNC services ==="
        stop_websockify
        stop_vnc
        ;;
    restart)
        log "=== Restarting noVNC services ==="
        stop_websockify
        stop_vnc
        sleep 2
        start_vnc
        start_websockify
        status
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
