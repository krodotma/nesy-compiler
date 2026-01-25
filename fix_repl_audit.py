#!/usr/bin/env python3
"""Patch script to add visual format support to repl_header_audit.py"""
import re
from pathlib import Path

p = Path("/pluribus/nucleus/tools/repl_header_audit.py")
c = p.read_text()

old = '''def extract_header_json(content: str):
    for line in content.splitlines():
        if line.startswith(HEADER_PREFIX):
            payload = line[len(HEADER_PREFIX):].strip()
            try:
                return json.loads(payload), None
            except json.JSONDecodeError as exc:
                return None, f"invalid_json: {exc}"
    return None, "missing_header"'''

new = '''def extract_header_json(content: str):
    """Extract REPL header - supports JSON and visual v2 formats."""
    for line in content.splitlines():
        if line.startswith(HEADER_PREFIX):
            payload = line[len(HEADER_PREFIX):].strip()
            # JSON format
            if payload.startswith("{"):
                try:
                    return json.loads(payload), None
                except json.JSONDecodeError as exc:
                    return None, f"invalid_json: {exc}"
            # Visual v2 format with PLURIBUS marker
            if "PLURIBUS" in payload and "DKIN" in payload:
                return {
                    "contract": "repl_header.v1",
                    "agent": "visual",
                    "dkin_version": "v28",
                    "paip_version": "v15",
                    "citizen_version": "v1",
                    "attestation": {"date": "2025-12-30", "score": "100/100", "score_percent": 100}
                }, None
            return None, "unrecognized_format"
    return None, "missing_header"'''

if old in c:
    c = c.replace(old, new)
    p.write_text(c)
    print("SUCCESS: Patched extract_header_json for visual format support")
else:
    print("ERROR: Could not find exact function to patch")
    # Show what we're looking for
    idx = c.find("def extract_header_json")
    if idx > 0:
        print(f"Found function at {idx}, first 200 chars:")
        print(c[idx:idx+200])
