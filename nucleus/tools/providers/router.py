#!/usr/bin/env python3
"""
Provider Router Shim
====================
Canonical implementation lives in the PIA monorepo:
  /root/pib/pluribus/nucleus/tools/providers/router.py
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys

TARGET = os.environ.get(
    "PLURIBUS_PROVIDER_ROUTER",
    "/root/pib/pluribus/nucleus/tools/providers/router.py",
)

if not os.path.exists(TARGET):
    print(f"CRITICAL: provider router not found at {TARGET}", file=sys.stderr)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("main_module", TARGET)
module = importlib.util.module_from_spec(spec)
# dataclasses require module presence in sys.modules
sys.modules["main_module"] = module
sys.modules["__main__"] = module
spec.loader.exec_module(module)

if __name__ == "__main__":
    main_func = getattr(module, "main", None)
    if callable(main_func):
        try:
            params = inspect.signature(main_func).parameters
        except Exception:
            params = {}
        if not params:
            raise SystemExit(main_func())
        raise SystemExit(main_func(sys.argv[1:]))
