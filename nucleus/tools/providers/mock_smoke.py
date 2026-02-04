#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="mock_smoke.py", description="Deterministic mock provider for offline STRp tests.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default=os.environ.get("MOCK_MODEL") or "mock-1")
    args = p.parse_args(argv)

    h = hashlib.sha256(args.prompt.encode("utf-8", errors="replace")).hexdigest()[:12]
    payload = {
        "summary": f"mock:{args.model}:{h}",
        "claims": [{"text": "mock provider used", "confidence": 1.0}],
        "gaps": [],
        "definitions": [],
        "counterexamples": [],
        "citations": [],
        "next_actions": ["replace mock provider with real provider"],
        "meta": {"provider": "mock", "model": args.model, "ts": time.time()},
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

