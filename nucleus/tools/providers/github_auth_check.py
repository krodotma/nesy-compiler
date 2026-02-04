#!/usr/bin/env python3
"""Fast GitHub auth check (gh) — no network mutation, no secrets.

Checks:
1) `gh` is installed
2) `gh auth status -h github.com` returns success

Outputs a short hint on stdout/stderr; never prints tokens.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    try:
        p = subprocess.run(["gh", "--version"], check=False, capture_output=True, text=True, timeout=3)
        if p.returncode != 0:
            sys.stderr.write("github: gh not available\n")
            return 1
    except Exception:
        sys.stderr.write("github: gh not available\n")
        return 1

    try:
        p = subprocess.run(["gh", "auth", "status", "-h", "github.com"], check=False, capture_output=True, text=True, timeout=5)
    except Exception as e:
        sys.stderr.write(f"github: gh auth status failed: {e}\n")
        return 1

    if p.returncode == 0:
        sys.stdout.write("github: logged in\n")
        return 0

    err = (p.stderr or p.stdout or "").strip().splitlines()
    hint = err[-1] if err else "not logged in"
    sys.stderr.write(f"github: {hint}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

