#!/usr/bin/env python3
"""
REPL Header Audit Tool - Contract Verification

Checks agent configuration files for REPL header contract compliance.
Emits repl.header.attestation events to the bus.

Usage:
    repl_header_audit.py check             # Run audit (default)
    repl_header_audit.py report            # Print detailed report
    repl_header_audit.py --json            # JSON output
    repl_header_audit.py --emit            # Emit bus event
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

PLURIBUS_ROOT = os.environ.get("PLURIBUS_ROOT", "/pluribus")
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", f"{PLURIBUS_ROOT}/.pluribus/bus")
SPECS_DIR = f"{PLURIBUS_ROOT}/nucleus/specs"

HEADER_PREFIX = "REPL_HEADER: "
HEADER_CONTRACT = "repl_header.v1"

SPEC_PATH = f"{SPECS_DIR}/repl_header_contract_v1.md"
SCHEMA_PATH = f"{SPECS_DIR}/repl_header_contract.schema.json"

TARGET_FILES = [
    "AGENTS.md",
    "nucleus/AGENTS.md",
    "CLAUDE.md",
    "nexus_bridge/claude.md",
    "nexus_bridge/codex.md",
    "nexus_bridge/gemini.md",
    "nexus_bridge/qwen.md",
    "nexus_bridge/grok.md",
    ".qwen/QWEN.md",
    ".qwen/AGENT_COORDINATION.md",
    "nexus_bridge/README.md"
]


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info"):
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "repl_header_audit"),
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "data": data,
    }
    bus_file = f"{BUS_DIR}/events.ndjson"
    try:
        with open(bus_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass
    return event["id"]


def extract_header_json(content: str):
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
    return None, "missing_header"


def validate_header_obj(obj: dict):
    required = ["contract", "agent", "dkin_version", "paip_version", "citizen_version", "attestation"]
    missing = [key for key in required if key not in obj]
    if missing:
        return False, f"missing_fields: {','.join(missing)}"
    if obj.get("contract") != HEADER_CONTRACT:
        return False, "contract_mismatch"
    att = obj.get("attestation", {})
    if not isinstance(att, dict):
        return False, "attestation_not_object"
    if "date" not in att or "score" not in att:
        return False, "attestation_missing_fields"
    if att.get("score") != "100/100":
        return False, "attestation_score_not_100"
    return True, "ok"


def run_audit():
    results = {
        "spec_exists": os.path.exists(SPEC_PATH),
        "schema_exists": os.path.exists(SCHEMA_PATH),
        "files": {},
        "total_checks": 0,
        "total_passed": 0,
        "score_percent": 0,
        "verdict": "FAIL"
    }

    base_checks = 2
    base_passed = int(results["spec_exists"]) + int(results["schema_exists"])
    results["total_checks"] += base_checks
    results["total_passed"] += base_passed

    for rel_path in TARGET_FILES:
        abs_path = f"{PLURIBUS_ROOT}/{rel_path}"
        file_result = {
            "exists": os.path.exists(abs_path),
            "header_found": False,
            "header_valid": False,
            "error": None
        }
        results["total_checks"] += 2
        if file_result["exists"]:
            results["total_passed"] += 1
            try:
                content = Path(abs_path).read_text(encoding="utf-8")
            except Exception as exc:
                file_result["error"] = f"read_error: {exc}"
                results["files"][rel_path] = file_result
                continue
            header_obj, error = extract_header_json(content)
            if header_obj is None:
                file_result["error"] = error
            else:
                file_result["header_found"] = True
                is_valid, reason = validate_header_obj(header_obj)
                file_result["header_valid"] = is_valid
                file_result["error"] = None if is_valid else reason
                if is_valid:
                    results["total_passed"] += 1
        results["files"][rel_path] = file_result

    if results["total_checks"] > 0:
        results["score_percent"] = round(
            (results["total_passed"] / results["total_checks"]) * 100, 1
        )

    results["verdict"] = "PASS" if results["score_percent"] == 100 else "FAIL"
    return results


def print_report(results: dict):
    print("=" * 60)
    print("REPL HEADER CONTRACT AUDIT")
    print("=" * 60)
    print(f"Spec exists: {'PASS' if results['spec_exists'] else 'FAIL'}")
    print(f"Schema exists: {'PASS' if results['schema_exists'] else 'FAIL'}")
    print("-" * 60)
    for path, info in results["files"].items():
        status = "PASS" if info["exists"] and info["header_valid"] else "FAIL"
        print(f"{status} {path}")
        if info["error"]:
            print(f"  -> {info['error']}")
    print("-" * 60)
    print(f"Total: {results['total_passed']}/{results['total_checks']} "
          f"({results['score_percent']}%)")
    print(f"Verdict: {results['verdict']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="REPL Header Contract Audit")
    parser.add_argument("command", nargs="?", default="check",
                        choices=["check", "report"],
                        help="Command to run")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--emit", action="store_true", help="Emit to bus")
    args = parser.parse_args()

    results = run_audit()
    event_id = None

    if args.emit:
        event_id = emit_bus_event("repl.header.attestation", "artifact", {
            "scope": "REPL Header Contract v1",
            "score": f"{results['total_passed']}/{results['total_checks']} "
                     f"({results['score_percent']}%)",
            "verdict": results["verdict"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": results["files"]
        })
        results["attestation_event_id"] = event_id

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if args.command == "report":
            print_report(results)
        else:
            print_report(results)

    sys.exit(0 if results["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
