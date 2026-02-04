#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/docs/ingresses}"
OUT_FILE="${OUT_FILE:-$OUT_DIR/ingress_inventory.md}"

mkdir -p "$OUT_DIR"

have() { command -v "$1" >/dev/null 2>&1; }

ts="$(date -Is 2>/dev/null || date)"
host="$(hostname 2>/dev/null || echo unknown)"

{
  echo "# Ingress Inventory"
  echo
  echo "- Generated: \`$ts\`"
  echo "- Hostname: \`$host\`"
  echo

  echo "## Listening sockets"
  echo
  if have ss; then
    echo '```'
    ss -lntup || true
    echo '```'
  else
    echo "- \`ss\` not available"
  fi
  echo

  echo "## Network"
  echo
  if have ip; then
    echo "### Interfaces"
    echo '```'
    ip -br address || true
    echo '```'
    echo
    echo "### Routes"
    echo '```'
    ip route || true
    echo '```'
  else
    echo "- \`ip\` not available"
  fi
  echo

  echo "## SSH authorized keys (fingerprints only)"
  echo
  if have ssh-keygen; then
    for f in /root/.ssh/authorized_keys /root/.ssh/authorized_keys2; do
      if [[ -f "$f" ]]; then
        echo "### \`$f\`"
        echo '```'
        awk 'NF && $1 !~ /^#/' "$f" | while read -r line; do
          printf '%s\n' "$line" | ssh-keygen -lf - 2>/dev/null || true
        done
        echo '```'
        echo
      fi
    done
  else
    echo "- \`ssh-keygen\` not available"
  fi

  echo "## Firewall"
  echo
  if have ufw; then
    echo '```'
    ufw status verbose || true
    echo '```'
  else
    echo "- \`ufw\` not installed (intentionally not auto-enabled)"
  fi
  echo
} >"$OUT_FILE"

echo "Wrote inventory to $OUT_FILE"

