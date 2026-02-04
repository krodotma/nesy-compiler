#!/usr/bin/env bash
set -euo pipefail

# Install/refresh operator-sugar shims in a bin directory.
# Default: ~/.local/bin (no sudo). Use --system for /usr/local/bin.
#
# Shims:
# - PLURIBUSCHECK
# - PBPAIR
#
# Usage:
#   ./install_operator_shims.sh
#   ./install_operator_shims.sh --system

prefix=""
if [[ "${1:-}" == "--system" ]]; then
  prefix="/usr/local/bin"
  shift
else
  prefix="${HOME}/.local/bin"
fi

mkdir -p "$prefix"

src_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ln -sf "$src_dir/PLURIBUSCHECK" "$prefix/PLURIBUSCHECK"
ln -sf "$src_dir/PBPAIR" "$prefix/PBPAIR"
ln -sf "$src_dir/install_sota_deps.sh" "$prefix/install_sota_deps"

echo "installed shims into: $prefix"
echo "ensure PATH contains: $prefix"

# Optional: Trigger SOTA dep installation if in a venv
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "[+] Virtual environment detected, installing SOTA dependencies..."
  bash "$src_dir/install_sota_deps.sh"
fi

