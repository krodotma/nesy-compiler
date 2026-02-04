#!/bin/bash
# Sky Connect: The "Nexus" Reconnect Tool (Mosh/Autossh)
# Usage: ./sky_connect.sh [user@host]

TARGET="${1:-vps3231363}"

function tssh() {
    echo "[Sky] Connecting to $1..."
    
    # Strategy 1: Mosh (UDP, Roaming)
    if command -v mosh >/dev/null; then
        echo "[Sky] Attempting Mosh..."
        mosh "$1" -- tmux attach || tmux new-session
        return
    fi

    # Strategy 2: Autossh (TCP Keepalive)
    if command -v autossh >/dev/null; then
        echo "[Sky] Autossh fallback..."
        export AUTOSSH_POLL=10
        export AUTOSSH_LOGFILE=/tmp/autossh.log
        autossh -M 0 -t "$1" "tmux attach || tmux new-session"
        return
    fi

    # Strategy 3: Standard SSH (Hardened)
    echo "[Sky] SSH fallback (Hardened)..."
    ssh -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o ControlMaster=auto \
        -o ControlPath=~/.ssh/sockets/%r@%h-%p \
        -o ControlPersist=600 \
        -t "$1" "tmux attach || tmux new-session"
}

# Nexus Reconnect Loop
while true; do
    tssh "$TARGET"
    echo "[Sky] Connection lost. Retrying in 3 seconds..."
    sleep 3
done
