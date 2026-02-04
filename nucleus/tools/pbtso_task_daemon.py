#!/usr/bin/env python3
"""
PBTSO Task Ingress Shim
=======================
Canonical implementation lives in PIA:
  /root/pib/pia/orchestrator/pbtso_task_daemon.py

This wrapper keeps Pluribus tooling stable while delegating to PIA.
"""
from __future__ import annotations

import importlib.util
import os
import sys


PIA_ROOT = os.environ.get("PIA_ROOT", "/root/pib/pia")
TARGET = os.path.join(PIA_ROOT, "orchestrator", "pbtso_task_daemon.py")

if not os.path.exists(TARGET):
    print(f"CRITICAL: PIA task ingress not found at {TARGET}", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("main_module", TARGET)
module = importlib.util.module_from_spec(spec)
# dataclasses requires the module to be in sys.modules
sys.modules["main_module"] = module
sys.modules["__main__"] = module
spec.loader.exec_module(module)

if __name__ == "__main__":
    main_func = getattr(module, "main", None)
    if callable(main_func):
        raise SystemExit(main_func())
